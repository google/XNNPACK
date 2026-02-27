# Copyright 2025 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Specializations for uint8 x86 dot kernel generators."""

# pylint: disable=missing-class-docstring
# pylint: disable=invalid-name

from ynnpack.kernels.dot.generator.x86 import x86_avx512


class x86_avx512vnni_uint8_int8_int32(x86_avx512):
  def __init__(self, vector_bits=512):
    super().__init__("avx512vnni", "uint8_int8_int32", "int32_t", vector_bits, (1, 16, 4))
    self.a_type = "uint8_t"
    self.b_type = "int8_t"
    self.flags += ["dot_flag::consistent_arithmetic"]

  def header(self):
    return super().header() + f"""

namespace {{

YNN_INTRINSIC __m{self.bits}i unaligned_load_broadcast_4xuint8(const uint8_t* ptr) {{
    uint32_t value;
    memcpy(&value, ptr, sizeof(uint32_t));
    return _mm{self.bits}_set1_epi32(value);
}}

}}  // namespace
"""

  def load_a_tile(self, i, k):
    return (
        f"__m{self.bits}i a_{i}_{k} = unaligned_load_broadcast_4xuint8({self.a_ptr(i, k)});\n"
    )

  def load_b_tile(self, k, j):
    return (
        f"__m{self.bits}i b_{k}_{j} ="
        f" _mm{self.bits}_load_si{self.bits}({self.b_ptr(k, j, f'__m{self.bits}i')});\n"
    )

  def product(self, i, j, k):
    return f"c_{i}_{j} = _mm{self.bits}_dpbusd_epi32(c_{i}_{j}, a_{i}_{k}, b_{k}_{j});\n"
