# Copyright 2026 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Specializations for uint8 x int4 x86 dot kernel generators."""

# pylint: disable=missing-class-docstring
# pylint: disable=invalid-name

from ynnpack.kernels.dot.generator.dot_base import generate_dot_kernels
from ynnpack.kernels.dot.generator.x86 import x86
from ynnpack.kernels.dot.generator.x86 import x86_avx
from ynnpack.kernels.dot.generator.x86 import x86_avx512


class x86_uint8_int4_int32(x86):

  def __init__(self, arch, vector_bits, tile_shape):
    super().__init__(
        arch, "uint8_int4_int32", "int32_t", vector_bits, tile_shape
    )
    self.a_type = "uint8_t"
    self.b_type = "int4x2"
    self.flags += ["dot_flag::consistent_arithmetic"]

  def b_tile_size_bytes(self):
    return f"{self.tile_shape[1] * self.tile_shape[2] // 2}"

  def header(self):
    return super().header() + """
#include <immintrin.h>

namespace {

YNN_INTRINSIC int64_t unaligned_load_u8x8(const uint8_t* ptr) {
  int64_t value;
  memcpy(&value, ptr, sizeof(int64_t));
  return value;
}

}  // namespace
"""

  def load_a_tile(self, i, k):
    a = f"unaligned_load_u8x8({self.a_ptr(i, k)})"
    if self.bits == 512:
      return f"__m512i a_{i}_{k} = _mm512_set1_epi64({a});\n"
    else:
      return f"__m{self.bits}i a_{i}_{k} = {self._mm()}_set1_epi64x({a});\n"

  def load_b_tile(self, k, j):
    bits = self.bits
    mm = self._mm()
    n = self.tile_shape[1] // 2
    return f"""
__m{bits}i b_{k}_{j}_eo = {mm}_load_si{bits}({self.b_ptr(k, j, f"__m{bits}i")});
__m{bits}i b_{k}_{j}_e = {mm}_and_si{bits}(b_{k}_{j}_eo, mask);
__m{bits}i b_{k}_{j}_o = {mm}_and_si{bits}({mm}_srli_epi16(b_{k}_{j}_eo, 4), mask);
__m{bits}i b_{k}_{j+0} = {mm}_unpacklo_epi8(b_{k}_{j}_e, b_{k}_{j}_o);
__m{bits}i b_{k}_{j+n} = {mm}_unpackhi_epi8(b_{k}_{j}_e, b_{k}_{j}_o);
b_{k}_{j+0} = {mm}_shuffle_epi8(sign_lut, b_{k}_{j+0});
b_{k}_{j+n} = {mm}_shuffle_epi8(sign_lut, b_{k}_{j+n});
"""

  def product(self, i, j, k):
    mm = self._mm()
    n = self.tile_shape[1] // 2
    a_ik = f"a_{i}_{k}"
    b_kj0 = f"b_{k}_{j+0}"
    b_kjn = f"b_{k}_{j+n}"
    c_ij0 = f"c_{i}_{j+0}"
    c_ijn = f"c_{i}_{j+n}"
    if k == 0:
      # The first tile initializes the local accumulator of int16 values.
      result = f"""
__m{self.bits}i {c_ij0}_block = {mm}_maddubs_epi16({a_ik}, {b_kj0});
__m{self.bits}i {c_ijn}_block = {mm}_maddubs_epi16({a_ik}, {b_kjn});
"""
    else:
      # Subsequent tiles add to the local accumulator. We can add 16 values of k
      # without overflow (32768 / (8*256) = 16). The accumulators are split in 2
      assert k <= 32
      result = f"""
{c_ij0}_block = {mm}_add_epi16({c_ij0}_block, {mm}_maddubs_epi16({a_ik}, {b_kj0}));
{c_ijn}_block = {mm}_add_epi16({c_ijn}_block, {mm}_maddubs_epi16({a_ik}, {b_kjn}));
"""
    if k + self.tile_shape[2] == self.block_shape[2]:
      # On the last tile, widen to int32, and add to the global accumulator.
      result += f"""
{c_ij0} = {mm}_sub_epi32({c_ij0}, {mm}_madd_epi16({c_ij0}_block, {mm}_set1_epi16(-1)));
{c_ijn} = {mm}_sub_epi32({c_ijn}, {mm}_madd_epi16({c_ijn}_block, {mm}_set1_epi16(-1)));
"""
    return result

  def init_c_tile(self, i, j):
    mm = self._mm()
    n = self.tile_shape[1] // 2
    return f"""
__m{self.bits}i c_{i}_{j+0} = {mm}_setzero_si{self.bits}();
__m{self.bits}i c_{i}_{j+n} = {mm}_setzero_si{self.bits}();
"""

  def finalize_c_tile(self, i, j):
    mm = self._mm()
    n = self.tile_shape[1] // 2
    return f"c_{i}_{j} = {mm}_hadd_epi32(c_{i}_{j}, c_{i}_{j+n});\n"


class x86_avx2_uint8_int4_int32(x86_uint8_int4_int32, x86_avx):

  def __init__(self, arch="avx2", vector_bits=256):
    super().__init__(arch, vector_bits, tile_shape=(1, 8, 8))

  def begin_func(self, func_name):
    result = super().begin_func(func_name)
    result += """
  const __m256i mask = _mm256_set1_epi8(0x0F);
  const __m256i sign_lut = _mm256_setr_epi8(
      0, 1, 2, 3, 4, 5, 6, 7, -8, -7, -6, -5, -4, -3, -2, -1,
      0, 1, 2, 3, 4, 5, 6, 7, -8, -7, -6, -5, -4, -3, -2, -1);
"""
    return result


class x86_avx512_uint8_int4_int32(x86_uint8_int4_int32, x86_avx512):

  def __init__(self, arch="avx512", vector_bits=512):
    super().__init__(arch, vector_bits, tile_shape=(1, 16, 8))

  def header(self):
    return super().header() + """
namespace {

YNN_INTRINSIC __m512i _mm512_hadd_epi32(__m512i a, __m512i b) {
  __m512i t0 = _mm512_unpacklo_epi64(a, b);
  __m512i t1 = _mm512_unpackhi_epi64(a, b);
  __m512i even = _mm512_add_epi32(t0, _mm512_srli_epi64(t0, 32));
  __m512i odd = _mm512_add_epi32(t1, _mm512_slli_epi64(t1, 32));
  return _mm512_mask_blend_epi32(0xAAAA, even, odd);
}

}  // namespace
"""

  def begin_func(self, func_name):
    result = super().begin_func(func_name)
    result += """
  const __m512i mask = _mm512_set1_epi8(0x0F);
  const __m512i sign_lut = _mm512_set_epi8(
      -1, -2, -3, -4, -5, -6, -7, -8, 7, 6, 5, 4, 3, 2, 1, 0,
      -1, -2, -3, -4, -5, -6, -7, -8, 7, 6, 5, 4, 3, 2, 1, 0,
      -1, -2, -3, -4, -5, -6, -7, -8, 7, 6, 5, 4, 3, 2, 1, 0,
      -1, -2, -3, -4, -5, -6, -7, -8, 7, 6, 5, 4, 3, 2, 1, 0);
"""
    return result


class x86_avx512vnni_uint8_int4_int32(x86_avx512_uint8_int4_int32):

  def __init__(self, arch="avx512vnni"):
    super().__init__(arch)

  def product(self, i, j, k):
    mm = self._mm()
    return f"""
c_{i}_{j+0} = {mm}_dpbusd_epi32(c_{i}_{j+0}, a_{i}_{k}, b_{k}_{j+0});
c_{i}_{j+8} = {mm}_dpbusd_epi32(c_{i}_{j+8}, a_{i}_{k}, b_{k}_{j+8});
"""


generate_dot_kernels(
    x86_avx2_uint8_int4_int32(),
    [
        (1, 16, 32),
        (2, 16, 32),
        (3, 16, 32),
        (1, 8, 32),
        (3, 8, 32),
        (4, 8, 32),
    ],
)

generate_dot_kernels(
    x86_avx512_uint8_int4_int32(),
    [
        (1, 32, 32),
        (2, 32, 32),
        (3, 32, 32),
        (4, 32, 32),
        (5, 32, 32),
        (1, 16, 32),
        (5, 16, 32),
        (8, 16, 32),
    ],
)

generate_dot_kernels(
    x86_avx512vnni_uint8_int4_int32(),
    [
        (1, 32, 8),
        (2, 32, 8),
        (3, 32, 8),
        (4, 32, 8),
        (5, 32, 8),
        (1, 16, 8),
        (5, 16, 8),
        (8, 16, 8),
        (12, 16, 8),
    ],
)
