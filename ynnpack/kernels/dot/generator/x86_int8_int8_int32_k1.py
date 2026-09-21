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
    return self.tile_shape[1] * self.tile_shape[2] // 4

  def init_c_tile(self, i, j):
    mm = self._mm()
    return f"__m{self.bits}i c_{i}_{j} = {mm}_setzero_si{self.bits}();\n"

  def load_a_tile(self, i, k):
    if self.block_shape[2] % 2 == 0 and k % 2 != 0:
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
    if self.block_shape[2] % 2 == 0 and k % 2 != 0:
      return ""
    mm = self._mm()
    bits = self.bits
    b0_kj = self.b_ptr(k, j, f"__m{bits//4}i")
    b0_kj = f"{self._mm(bits//4)}_loadu_si{bits//4}({b0_kj})"
    # We are using madd_epi16, which is a 2-way int16 dot product. We need to
    # load each int8 weight into an int16, and then insert either the next k
    # weight (when block_k is even) or an int16 zero in the upper 16 bits.
    mask = f"{mm}_set1_epi32(0xffff)"
    b0 = f"{mm}_and_si{bits}({mm}_cvtepi8_epi32({b0_kj}), {mask})"
    if self.block_shape[2] % 2 == 0:
      b1_kj = self.b_ptr(k + 1, j, f"__m{bits//4}i")
      b1_kj = f"{self._mm(bits//4)}_loadu_si{bits//4}({b1_kj})"
      b1 = f"{mm}_slli_epi32({mm}_cvtepi8_epi32({b1_kj}), 16)"
      return f"__m{bits}i b_{k}_{j} = {mm}_or_si{bits}({b0}, {b1});\n"
    return f"__m{bits}i b_{k}_{j} = {b0};\n"

  def product(self, i, j, k):
    if self.block_shape[2] % 2 == 0 and k % 2 != 0:
      return ""
    mm = self._mm()
    c = f"c_{i}_{j}"
    a = f"a_{i}_{k}"
    b = f"b_{k}_{j}"
    return f"{c} = {mm}_add_epi32({c}, {mm}_madd_epi16({a}, {b}));\n"


class x86_avx2_int8_int8_int32_k1(x86_avx, x86_int8_int8_int32_k1):

  def __init__(self, arch="avx2", vector_bits=256):
    super().__init__(
        arch, "int8_int8_int32", "int32_t", vector_bits, tile_shape=(1, 8, 1)
    )
    self.a_type = "int8_t"
    self.b_type = "int8_t"
    self.flags += ["dot_flag::consistent_arithmetic"]

  def header(self):
    return super().header() + """
using __m64i = void;
"""


class x86_avx512_int8_int8_int32_k1(x86_avx512, x86_int8_int8_int32_k1):

  def __init__(self, arch="avx512", vector_bits=512):
    super().__init__(
        arch, "int8_int8_int32", "int32_t", vector_bits, tile_shape=(1, 16, 1)
    )
    self.a_type = "int8_t"
    self.b_type = "int8_t"
    self.flags += ["dot_flag::consistent_arithmetic"]


generate_dot_kernels(
    x86_avx2_int8_int8_int32_k1(),
    [
        (1, 32, 2),
        (2, 32, 2),
        (1, 16, 2),
        (3, 16, 2),
        (4, 16, 2),
        (5, 16, 2),
        (8, 8, 2),
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
        (16, 16, 2),
    ],
)
