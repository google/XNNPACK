# Copyright 2026 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Specializations for uint8 x int2 x86 dot kernel generators."""

# pylint: disable=missing-class-docstring
# pylint: disable=invalid-name

from ynnpack.kernels.dot.generator.dot_base import generate_dot_kernels
from ynnpack.kernels.dot.generator.x86 import x86
from ynnpack.kernels.dot.generator.x86 import x86_avx
from ynnpack.kernels.dot.generator.x86 import x86_avx512


class x86_uint8_int2_int32(x86):

  def __init__(self, arch, vector_bits, tile_shape):
    super().__init__(
        arch, "uint8_int2_int32", "int32_t", vector_bits, tile_shape
    )
    self.a_type = "uint8_t"
    self.b_type = "int2x4"
    self.flags += ["dot_flag::consistent_arithmetic"]

  def b_tile_size_bytes(self):
    return f"{self.tile_shape[1] * self.tile_shape[2] // 4}"

  def broadcast_128(self):
    if self.bits == 256:
      return "_mm256_broadcastsi128_si256"
    elif self.bits == 512:
      return "_mm512_broadcast_i32x4"
    else:
      raise ValueError(f"Unsupported bits: {self.bits}")

  def begin_func(self, func_name):
    mm = self._mm()
    bits = self.bits
    broadcast_128 = self.broadcast_128()
    return super().begin_func(func_name) + f"""
const __m{bits}i mask_k02 = {mm}_set1_epi8(0x33);
const __m{bits}i mask_k13 = {mm}_set1_epi8(0xCC);
const __m{bits}i sign_lut = {broadcast_128}(_mm_setr_epi8(0, 1, -2, -1, 0, 1, -2, -1, 0, 1, -2, -1, 0, 1, -2, -1));
const __m{bits}i shuf_a0 = {broadcast_128}(_mm_setr_epi8(0, 4, 8, 12, 0, 4, 8, 12, 0, 4, 8, 12, 0, 4, 8, 12));
const __m{bits}i shuf_a1 = {broadcast_128}(_mm_setr_epi8(1, 5, 9, 13, 1, 5, 9, 13, 1, 5, 9, 13, 1, 5, 9, 13));
const __m{bits}i shuf_a2 = {broadcast_128}(_mm_setr_epi8(2, 6, 10, 14, 2, 6, 10, 14, 2, 6, 10, 14, 2, 6, 10, 14));
const __m{bits}i shuf_a3 = {broadcast_128}(_mm_setr_epi8(3, 7, 11, 15, 3, 7, 11, 15, 3, 7, 11, 15, 3, 7, 11, 15));
"""

  def load_b_tile(self, k, j):
    bits = self.bits
    mm = self._mm()
    # Unpack 2-bit data to 8-bit, leaving every 4th value of K in 4 registers.
    # The sign_lut effectively computes x % 4, but shuffle_epi8 has a special
    # case for when the upper bits are set. so we need to mask off the bits that
    # would pollute the upper bit of each 8-bit lane before applying sign_lut.
    return f"""
__m{bits}i b_{k+0}_{j} = {mm}_load_si{bits}({self.b_ptr(k, j, f"__m{bits}i")});
__m{bits}i b_{k+1}_{j} = {mm}_and_si{bits}(b_{k+0}_{j}, mask_k13);
b_{k+0}_{j} = {mm}_and_si{bits}(b_{k+0}_{j}, mask_k02);

__m{bits}i b_{k+3}_{j} = {mm}_srli_epi32(b_{k+1}_{j}, 6);
__m{bits}i b_{k+2}_{j} = {mm}_srli_epi32(b_{k+0}_{j}, 4);
b_{k+1}_{j} = {mm}_srli_epi32(b_{k+1}_{j}, 2);

b_{k+0}_{j} = {mm}_shuffle_epi8(sign_lut, b_{k+0}_{j});
b_{k+1}_{j} = {mm}_shuffle_epi8(sign_lut, b_{k+1}_{j});
b_{k+2}_{j} = {mm}_shuffle_epi8(sign_lut, b_{k+2}_{j});
b_{k+3}_{j} = {mm}_shuffle_epi8(sign_lut, b_{k+3}_{j});
"""

  def load_a_tile(self, i, k):
    # shuffle A to match this ordering, enabling 4-way dot products using either
    # dpbusd or pmaddubsw + pmaddwd.
    mm = self._mm()
    a_ptr = self.a_ptr(i, k)
    a_128 = f"_mm_loadu_si128(reinterpret_cast<const __m128i*>({a_ptr}))"
    return f"""
__m{self.bits}i a_{i}_{k+0} = {self.broadcast_128()}({a_128});
__m{self.bits}i a_{i}_{k+3} = {mm}_shuffle_epi8(a_{i}_{k+0}, shuf_a3);
__m{self.bits}i a_{i}_{k+2} = {mm}_shuffle_epi8(a_{i}_{k+0}, shuf_a2);
__m{self.bits}i a_{i}_{k+1} = {mm}_shuffle_epi8(a_{i}_{k+0}, shuf_a1);
a_{i}_{k+0} = {mm}_shuffle_epi8(a_{i}_{k+0}, shuf_a0);
"""

  def product(self, i, j, k):
    mm = self._mm()
    a_ik0 = f"a_{i}_{k+0}"
    a_ik1 = f"a_{i}_{k+1}"
    a_ik2 = f"a_{i}_{k+2}"
    a_ik3 = f"a_{i}_{k+3}"
    b_k0j = f"b_{k+0}_{j}"
    b_k1j = f"b_{k+1}_{j}"
    b_k2j = f"b_{k+2}_{j}"
    b_k3j = f"b_{k+3}_{j}"
    c_ij = f"c_{i}_{j}"
    if k == 0:
      # The first tile initializes the local accumulator of int16 values.
      result = f"""
__m{self.bits}i {c_ij}_block = {mm}_maddubs_epi16({a_ik0}, {b_k0j});
"""
    else:
      # Subsequent tiles add to the local accumulator. We can add 64 values of k
      # without overflow (32768 / (2*256) = 64).
      assert k <= 64
      result = f"""
{c_ij}_block = {mm}_add_epi16({c_ij}_block, {mm}_maddubs_epi16({a_ik0}, {b_k0j}));
"""
    result += f"""
{c_ij}_block = {mm}_add_epi16({c_ij}_block, {mm}_maddubs_epi16({a_ik1}, {b_k1j}));
{c_ij}_block = {mm}_add_epi16({c_ij}_block, {mm}_maddubs_epi16({a_ik2}, {b_k2j}));
{c_ij}_block = {mm}_add_epi16({c_ij}_block, {mm}_maddubs_epi16({a_ik3}, {b_k3j}));
"""
    if k + self.tile_shape[2] == self.block_shape[2]:
      # On the last tile, widen to int32, and add to the global accumulator.
      result += f"""
{c_ij} = {mm}_sub_epi32({c_ij}, {mm}_madd_epi16({c_ij}_block, {mm}_set1_epi16(-1)));
"""
    return result


class x86_avx2_uint8_int2_int32(x86_uint8_int2_int32, x86_avx):

  def __init__(self, arch="avx2", vector_bits=256):
    super().__init__(arch, vector_bits, tile_shape=(1, 8, 16))


class x86_avx512_uint8_int2_int32(x86_uint8_int2_int32, x86_avx512):

  def __init__(self, arch="avx512", vector_bits=512):
    super().__init__(arch, vector_bits, tile_shape=(1, 16, 16))


class x86_avx512vnni_uint8_int2_int32(x86_avx512_uint8_int2_int32):

  def __init__(self, arch="avx512vnni"):
    super().__init__(arch)

  def product(self, i, j, k):
    mm = self._mm()
    return f"""
c_{i}_{j} = {mm}_dpbusd_epi32(c_{i}_{j}, a_{i}_{k+0}, b_{k+0}_{j});
c_{i}_{j} = {mm}_dpbusd_epi32(c_{i}_{j}, a_{i}_{k+1}, b_{k+1}_{j});
c_{i}_{j} = {mm}_dpbusd_epi32(c_{i}_{j}, a_{i}_{k+2}, b_{k+2}_{j});
c_{i}_{j} = {mm}_dpbusd_epi32(c_{i}_{j}, a_{i}_{k+3}, b_{k+3}_{j});
"""


generate_dot_kernels(
    x86_avx2_uint8_int2_int32(),
    [
        (1, 32, 32),
        (2, 32, 32),
        (3, 32, 32),
        (4, 32, 32),
    ],
)

generate_dot_kernels(
    x86_avx512_uint8_int2_int32(),
    [
        (1, 64, 32),
        (2, 64, 32),
        (3, 64, 32),
        (4, 64, 32),
    ],
)

generate_dot_kernels(
    x86_avx512vnni_uint8_int2_int32(),
    [
        (1, 64, 16),
        (2, 64, 16),
        (3, 64, 16),
        (4, 64, 16),
    ],
)
