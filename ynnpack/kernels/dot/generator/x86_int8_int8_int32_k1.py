# Copyright 2025 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Specializations for int8 x86 dot kernel generators."""

# pylint: disable=missing-class-docstring
# pylint: disable=invalid-name

from ynnpack.kernels.dot.generator.dot_base import generate_dot_kernels
from ynnpack.kernels.dot.generator.x86 import x86
from ynnpack.kernels.dot.generator.x86 import x86_avx
from ynnpack.kernels.dot.generator.x86 import x86_avx512


class x86_int8_int8_int32_k1(x86):

  def header(self):
    return super().header() + """

namespace {

// `_mm_loadu_si16` is missing in GCC < 11.
YNN_INTRINSIC int16_t unaligned_load_int8x2(const int8_t* ptr) {
  int16_t value;
  memcpy(&value, ptr, sizeof(int16_t));
  return value;
}

}  // namespace
"""

  def b_alignment_bytes(self):
    # This kernel loads half-vectors at a time from b.
    return self.tile_shape[1] * self.tile_shape[2] // 2

  def load_a_tile(self, i, k):
    if k % 2 != 0:
      return ""
    mm = self._mm()
    bits = self.bits
    if self.block_shape[2] % 2 == 0:
      a = f"unaligned_load_int8x2({self.a_ptr(i, k)})"
      return (
          f"__m{bits}i a_{i}_{k} ="
          f" {mm}_cvtepi8_epi16({self._mm(bits//2)}_set1_epi16({a}));\n"
      )
    # Convert to int16 while we load
    return f"__m{bits}i a_{i}_{k} = {mm}_set1_epi16(*{self.a_ptr(i, k)});\n"

  def load_b_tile(self, k, j):
    if k % 2 != 0:
      return ""
    bits = self.bits
    mm = self._mm()
    mm_2 = self._mm(bits // 2)
    m_2 = f"__m{bits // 2}i"
    j0 = j
    j1 = j + self.tile_shape[1] // 2

    # We're loading two tiles at once.
    b0 = f"{mm_2}_loadu_si{bits // 2}({self.b_ptr(k + 0, j, m_2)})"
    if self.block_shape[2] % 2 == 0:
      b1 = f"{mm_2}_loadu_si{bits // 2}({self.b_ptr(k + 1, j, m_2)})"
    else:
      b1 = f"{mm_2}_setzero_si{bits // 2}()"

    return f"""
const {m_2} b_r0_{k}_{j} = {b0};
const {m_2} b_r1_{k}_{j} = {b1};
__m{bits}i b_{k}_{j0} =
    {mm}_cvtepi8_epi16({mm_2}_unpacklo_epi8(b_r0_{k}_{j}, b_r1_{k}_{j}));
__m{bits}i b_{k}_{j1} =
    {mm}_cvtepi8_epi16({mm_2}_unpackhi_epi8(b_r0_{k}_{j}, b_r1_{k}_{j}));
"""

  def product(self, i, j, k):
    if k % 2 != 0:
      return ""
    mm = self._mm()
    j0 = j
    j1 = j + self.tile_shape[1] // 2
    c_ij0 = f"c_{i}_{j0}"
    c_ij1 = f"c_{i}_{j1}"
    return f"""
{c_ij0} = {mm}_add_epi32({c_ij0}, {mm}_madd_epi16(a_{i}_{k}, b_{k}_{j0}));
{c_ij1} = {mm}_add_epi32({c_ij1}, {mm}_madd_epi16(a_{i}_{k}, b_{k}_{j1}));
"""


class x86_avx2_int8_int8_int32_k1(x86_avx, x86_int8_int8_int32_k1):

  def __init__(self, arch="avx2", vector_bits=256):
    super().__init__(
        arch, "int8_int8_int32", "int32_t", vector_bits, tile_shape=(1, 16, 1)
    )
    self.a_type = "int8_t"
    self.b_type = "int8_t"
    self.flags += ["dot_flag::consistent_arithmetic"]


class x86_avx512_int8_int8_int32_k1(x86_avx512, x86_int8_int8_int32_k1):

  def __init__(self, arch="avx512", vector_bits=512):
    super().__init__(
        arch, "int8_int8_int32", "int32_t", vector_bits, tile_shape=(1, 32, 1)
    )
    self.a_type = "int8_t"
    self.b_type = "int8_t"
    self.flags += ["dot_flag::consistent_arithmetic"]

  def finalize_c_tile(self, i, j):
    c0 = f"c_{i}_{j}"
    c1 = f"c_{i}_{j + self.tile_shape[1] // 2}"
    return f"""
__m512i {c0}_perm = _mm512_shuffle_i64x2({c0}, {c1}, 0x44);
{c1} = _mm512_shuffle_i64x2({c0}, {c1}, 0xEE);
{c0} = {c0}_perm;
"""


generate_dot_kernels(
    x86_avx2_int8_int8_int32_k1(),
    [
        (1, 32, 2),
        (2, 32, 2),
        (1, 16, 2),
        (3, 16, 2),
        (4, 16, 2),
        (5, 16, 2),
    ],
)

generate_dot_kernels(
    x86_avx512_int8_int8_int32_k1(),
    [
        (1, 64, 2),
        (2, 64, 2),
        (3, 64, 2),
        (1, 32, 2),
        (4, 32, 2),
        (5, 32, 2),
        (6, 32, 2),
        (8, 32, 2),
    ],
)
