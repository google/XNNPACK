# Copyright 2025 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Specializations for int8 x86 dot kernel generators."""

# pylint: disable=missing-class-docstring
# pylint: disable=invalid-name

from ynnpack.kernels.dot.generator.x86 import x86
from ynnpack.kernels.dot.generator.x86 import x86_avx
from ynnpack.kernels.dot.generator.x86 import x86_avx512


class x86_int8_int8_int32_k1(x86):
  def b_alignment_required(self):
    # This kernel loads half-vectors at a time from b.
    return self.tile_shape[1] * self.tile_shape[2] // 4

  def init_c_tile(self, i, j):
    mm = self._mm()
    return f"__m{self.bits}i c_{i}_{j} = {mm}_setzero_si{self.bits}();\n"

  def load_a_tile(self, i, k):
    mm = self._mm()
    bits = self.bits
    # Convert to int16 while we load
    return f"__m{bits}i a_{i}_{k} = {mm}_set1_epi16(*{self.a_ptr(i, k)});\n"

  def load_b_tile(self, k, j):
    b_kj = self.b_ptr(k, j, f"__m{self.bits//4}i")
    b_kj = f"{self._mm(self.bits//4)}_loadu_si{self.bits//4}({b_kj})"
    mm = self._mm()
    bits = self.bits
    # We are using madd_epi16, which is a 2-way int16 dot product. We need to
    # load each int8 weight into an int16, and then insert another int16 zero.
    # We do this by using cvtepi8_epi32, and then masking off the upper 16 bits.
    mask = f"{mm}_set1_epi32(0xffff)"
    return (
        f"__m{bits}i b_{k}_{j} = {mm}_and_si{bits}({mm}_cvtepi8_epi32({b_kj}),"
        f" {mask});\n"
    )

  def product(self, i, j, k):
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
