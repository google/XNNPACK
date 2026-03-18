# Copyright 2025 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Specializations for bf16 x86 dot kernel generators."""

# pylint: disable=missing-class-docstring
# pylint: disable=invalid-name

from ynnpack.kernels.dot.generator.x86 import x86
from ynnpack.kernels.dot.generator.x86 import x86_avx
from ynnpack.kernels.dot.generator.x86 import x86_avx512


class x86_bf16_bf16_fp32_k1(x86):
  def __init__(self, arch, bits, tile_shape):
    super().__init__(arch, "bf16_bf16_fp32", "float", bits, tile_shape)
    self.a_type = "bfloat16"
    self.b_type = "bfloat16"

  def header(self):
    return super().header() + f"""

namespace {{

struct bfloat16 {{
  std::uint16_t value;
}};

YNN_INTRINSIC __m{self.bits}i unaligned_load_broadcast_bf16(const bfloat16* ptr) {{
    int16_t value;
    memcpy(&value, ptr, sizeof(int16_t));
    return {self._mm()}_set1_epi16(value);
}}

}}  // namespace
"""

  def load_a_tile(self, i, k):
    bits = self.bits
    cast = f"{self._mm()}_castsi{bits}_ps"
    bitwise_and = f"{self._mm()}_and_si{bits}"
    # We want to broadcast a single bf16 to a vector of f32. We implement this
    # with a 16-bit broadcast, and then masking off the low 16 bits of each
    # 32-bit value.
    a_ik = f"unaligned_load_broadcast_bf16({self.a_ptr(i, k)})"
    mask = f"{self._mm()}_set1_epi32(0xffff0000)"
    return f"__m{bits} a_{i}_{k} = {cast}({bitwise_and}({a_ik}, {mask}));\n"

  def load_b_tile(self, k, j):
    bits = self.bits
    cast = f"{self._mm()}_castsi{bits}_ps"
    ptr = self.b_ptr(k, j, f"__m{bits//2}i")
    b_kj = f"{self._mm(bits//2)}_loadu_si{bits//2}({ptr})"
    b_kj = f"{self._mm()}_cvtepi16_epi32({b_kj})"
    return f"""
__m{bits} b_{k}_{j} = {cast}({self._mm()}_slli_epi32({b_kj}, 16));
"""


class x86_avx2_bf16_bf16_fp32_k1(x86_bf16_bf16_fp32_k1, x86_avx):
  def __init__(self, arch="avx2", bits=256, tile_shape=(1, 8, 1)):
    super().__init__(arch, bits, tile_shape)

  def product(self, i, j, k):
    add = f"{self._mm()}_add_ps"
    mul = f"{self._mm()}_mul_ps"
    return f"""
c_{i}_{j} = {add}(c_{i}_{j}, {mul}(a_{i}_{k}, b_{k}_{j}));
"""


class x86_avx2_fma3_bf16_bf16_fp32_k1(x86_avx2_bf16_bf16_fp32_k1):
  def __init__(self, arch="avx2_fma3", bits=256, tile_shape=(1, 8, 1)):
    super().__init__(arch, bits, tile_shape)

  def product(self, i, j, k):
    return f"""
c_{i}_{j} = {self._mm()}_fmadd_ps(a_{i}_{k}, b_{k}_{j}, c_{i}_{j});
"""


class x86_avx512_bf16_bf16_fp32_k1(x86_bf16_bf16_fp32_k1, x86_avx512):

  def __init__(self, arch="avx512", bits=512, tile_shape=(1, 16, 1)):
    super().__init__(arch, bits, tile_shape)

  def product(self, i, j, k):
    return f"""
c_{i}_{j} = {self._mm()}_fmadd_ps(a_{i}_{k}, b_{k}_{j}, c_{i}_{j});
"""
