# Copyright 2025 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Specializations for uint8 x86 dot kernel generators."""

# pylint: disable=missing-class-docstring
# pylint: disable=invalid-name

from ynnpack.kernels.dot.generator.dot_base import generate_dot_kernels
from ynnpack.kernels.dot.generator.x86 import x86_avx512


class x86_avx512vnni_uint8_int8_int32(x86_avx512):
  def __init__(self, vector_bits=512):
    super().__init__("avx512vnni", "uint8_int8_int32", "int32_t", vector_bits, (1, 16, 4))
    self.a_type = "uint8_t"
    self.b_type = "int8_t"
    self.flags += ["dot_flag::consistent_arithmetic"]

  def header(self):
    return super().header() + """

namespace {

YNN_INTRINSIC int32_t unaligned_load_u8x4(const uint8_t* ptr) {
    int32_t value;
    memcpy(&value, ptr, sizeof(int32_t));
    return value;
}

}  // namespace
"""

  def load_a_tile(self, i, k):
    bits = self.bits
    a = f"unaligned_load_u8x4({self.a_ptr(i, k)})"
    return f"__m{bits}i a_{i}_{k} = _mm{self.bits}_set1_epi32({a});\n"

  def load_b_tile(self, k, j):
    bits = self.bits
    b_ptr = self.b_ptr(k, j, f"__m{bits}i")
    return f"__m{bits}i b_{k}_{j} = _mm{bits}_load_si{bits}({b_ptr});\n"

  def product(self, i, j, k):
    mm = self._mm()
    c_ij = f"c_{i}_{j}"
    return f"{c_ij} = {mm}_dpbusd_epi32({c_ij}, a_{i}_{k}, b_{k}_{j});\n"


generate_dot_kernels(
    x86_avx512vnni_uint8_int8_int32(),
    [
        (1, 64, 4),
        (2, 64, 4),
        (3, 64, 4),
        (4, 64, 4),
        (5, 64, 4),
        (1, 32, 4),
        (2, 32, 4),
        (3, 32, 4),
        (4, 32, 4),
        (5, 32, 4),
        (6, 32, 4),
        (8, 32, 4),
        (10, 32, 4),
        (16, 16, 4),
    ],
)
