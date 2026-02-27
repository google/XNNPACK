# Copyright 2025 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Specializations for fp32 x86 dot kernel generators."""

# pylint: disable=invalid-name
# pylint: disable=missing-class-docstring

from ynnpack.kernels.dot.generator.x86 import x86
from ynnpack.kernels.dot.generator.x86 import x86_avx
from ynnpack.kernels.dot.generator.x86 import x86_avx512
from ynnpack.kernels.dot.generator.x86 import x86_sse2


class x86_fp32(x86):
  def __init__(self, arch, bits, tile_shape):
    super().__init__(arch, "fp32", "float", bits, tile_shape)
    self.a_type = "float"
    self.b_type = "float"

  def header(self):
    return super().header() + f"""

namespace {{

YNN_INTRINSIC __m{self.bits} unaligned_load_broadcast(const float* ptr) {{
    float value;
    memcpy(&value, ptr, sizeof(float));
    return {self._mm()}_set1_ps(value);
}}

}}  // namespace
"""

  def load_a_tile(self, i, k):
    return (
        f"__m{self.bits} a_{i}_{k} = unaligned_load_broadcast({self.a_ptr(i, k)});\n"
    )

  def load_b_tile(self, k, j):
    return (
        f"__m{self.bits} b_{k}_{j} = {self._mm()}_load_ps({self.b_ptr(k, j)});\n"
    )

  def product(self, i, j, k):
    mul = f"{self._mm()}_mul_ps(a_{i}_{k}, b_{k}_{j})"
    return f"c_{i}_{j} = {self._mm()}_add_ps(c_{i}_{j}, {mul});\n"


class x86_sse2_fp32(x86_fp32, x86_sse2):
  def __init__(self):
    super().__init__("sse2", 128, (1, 4, 1))


class x86_avx_fp32(x86_fp32, x86_avx):
  def __init__(self, arch="avx"):
    super().__init__(arch, 256, (1, 8, 1))
    self.flags += ["dot_flag::unaligned_b"]

  def b_alignment_required(self):
    return 1

  def load_b_tile(self, k, j):
    ptr = self.b_ptr(k, j)
    if self.n != "N":
      return f"__m256 b_{k}_{j} = _mm256_loadu_ps({ptr});\n"
    else:
      mask_ptr = f"&mask_table[sub_sat(8, sub_sat({self.n}, {j}))"
      mask = f"_mm256_loadu_si256((const __m256i*) {mask_ptr}])"
      return f"__m256 b_{k}_{j} = _mm256_maskload_ps({ptr}, {mask});\n"


class x86_fma3_fp32(x86_avx_fp32):
  def __init__(self):
    super().__init__("fma3")
    self.flags += ["dot_flag::consistent_arithmetic"]

  def product(self, i, j, k):
    return f"c_{i}_{j} = {self._mm()}_fmadd_ps(a_{i}_{k}, b_{k}_{j}, c_{i}_{j});\n"


class x86_avx512_fp32(x86_fp32, x86_avx512):
  def __init__(self):
    super().__init__("avx512", 512, (1, 16, 1))
    self.flags += ["dot_flag::consistent_arithmetic"]
    self.flags += ["dot_flag::unaligned_b"]

  def b_alignment_required(self):
    return 1

  def load_b_tile(self, k, j):
    mm = self._mm()
    ptr = self.b_ptr(k, j)
    if self.n != "N":
      return f"__m512 b_{k}_{j} = _mm512_loadu_ps({ptr});\n"
    else:
      zero = "_mm512_setzero_ps()"
      mask = f"(uint32_t)((1 << min(16, sub_sat({self.n}, {j}))) - 1)"
      mask = f"_cvtu32_mask16({mask})"
      return f"__m512 b_{k}_{j} = {mm}_mask_loadu_ps({zero}, {mask}, {ptr});\n"

  def product(self, i, j, k):
    return f"c_{i}_{j} = {self._mm()}_fmadd_ps(a_{i}_{k}, b_{k}_{j}, c_{i}_{j});\n"
