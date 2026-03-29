# Copyright 2025 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Specializations for fp64 x86 dot kernel generators."""

# pylint: disable=invalid-name
# pylint: disable=missing-class-docstring

from ynnpack.kernels.dot.generator.dot_base import generate_dot_kernels
from ynnpack.kernels.dot.generator.x86 import x86
from ynnpack.kernels.dot.generator.x86 import x86_avx
from ynnpack.kernels.dot.generator.x86 import x86_avx512


class x86_fp64(x86):
  def __init__(self, arch, bits, tile_shape):
    super().__init__(arch, "fp64", "double", bits, tile_shape)
    self.a_type = "double"
    self.b_type = "double"

  def header(self):
    return super().header() + f"""

namespace {{

YNN_INTRINSIC __m{self.bits}d unaligned_load_broadcast(const double* ptr) {{
    double value;
    memcpy(&value, ptr, sizeof(double));
    return {self._mm()}_set1_pd(value);
}}

}}  // namespace
"""

  def load_a_tile(self, i, k):
    return (
        f"__m{self.bits}d a_{i}_{k} ="
        f" unaligned_load_broadcast({self.a_ptr(i, k)});\n"
    )

  def load_b_tile(self, k, j):
    return (
        f"__m{self.bits}d b_{k}_{j} ="
        f" {self._mm()}_load_pd({self.b_ptr(k, j)});\n"
    )

  def product(self, i, j, k):
    mul = f"{self._mm()}_mul_pd(a_{i}_{k}, b_{k}_{j})"
    return f"c_{i}_{j} = {self._mm()}_add_pd(c_{i}_{j}, {mul});\n"


class x86_avx_fp64(x86_fp64, x86_avx):
  def __init__(self, arch="avx"):
    super().__init__(arch, 256, (1, 4, 1))
    self.flags += ["dot_flag::unaligned_b"]

  def b_alignment_required(self):
    return 1

  def load_b_tile(self, k, j):
    ptr = self.b_ptr(k, j)
    if self.n != "N":
      return f"__m256d b_{k}_{j} = _mm256_loadu_pd({ptr});\n"
    else:
      mask_ptr = f"&mask_table[sub_sat(8, sub_sat({self.n}, {j}) * 2)"
      mask = f"_mm256_loadu_si256((const __m256i*) {mask_ptr}])"
      return f"__m256d b_{k}_{j} = _mm256_maskload_pd({ptr}, {mask});\n"


class x86_fma3_fp64(x86_avx_fp64):
  def __init__(self):
    super().__init__("fma3")
    self.flags += ["dot_flag::consistent_arithmetic"]

  def product(self, i, j, k):
    return (
        f"c_{i}_{j} = {self._mm()}_fmadd_pd(a_{i}_{k}, b_{k}_{j}, c_{i}_{j});\n"
    )


class x86_avx512_fp64(x86_fp64, x86_avx512):
  def __init__(self):
    super().__init__("avx512", 512, (1, 8, 1))
    self.flags += ["dot_flag::consistent_arithmetic"]
    self.flags += ["dot_flag::unaligned_b"]

  def b_alignment_required(self):
    return 1

  def load_b_tile(self, k, j):
    mm = self._mm()
    ptr = self.b_ptr(k, j)
    if self.n != "N":
      return f"__m512d b_{k}_{j} = _mm512_loadu_pd({ptr});\n"
    else:
      zero = "_mm512_setzero_pd()"
      mask = f"(uint32_t)((1 << min(8, sub_sat({self.n}, {j}))) - 1)"
      mask = f"_cvtu32_mask16({mask})"
      return f"__m512d b_{k}_{j} = {mm}_mask_loadu_pd({zero}, {mask}, {ptr});\n"

  def product(self, i, j, k):
    return (
        f"c_{i}_{j} = {self._mm()}_fmadd_pd(a_{i}_{k}, b_{k}_{j}, c_{i}_{j});\n"
    )


generate_dot_kernels(
    x86_avx_fp64(),
    [
        (1, 16, 1),
        (2, 16, 1),
        (2, 8, 1),
        (3, 8, 1),
        (4, 8, 1),
        (4, 4, 1),
        (6, 4, 1),
        (8, 4, 1),
    ],
)

generate_dot_kernels(
    x86_fma3_fp64(),
    [
        (1, 16, 1),
        (2, 16, 1),
        (2, 8, 1),
        (3, 8, 1),
        (4, 8, 1),
        (5, 8, 1),
        (6, 8, 1),
        (8, 4, 1),
    ],
)

generate_dot_kernels(
    x86_avx512_fp64(),
    [
        (1, 32, 1),
        (2, 32, 1),
        (3, 32, 1),
        (4, 32, 1),
        (5, 32, 1),
        (2, 16, 1),
        (3, 16, 1),
        (4, 16, 1),
        (5, 16, 1),
        (5, 8, 1),
    ],
)
