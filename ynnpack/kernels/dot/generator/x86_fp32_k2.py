# Copyright 2025 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Specializations for fp32 x86 dot kernel generators."""

# pylint: disable=missing-class-docstring
# pylint: disable=invalid-name

from ynnpack.kernels.dot.generator.dot_base import generate_dot_kernels
from ynnpack.kernels.dot.generator.x86 import x86_avx
from ynnpack.kernels.dot.generator.x86 import x86_avx512


class x86_avx2_fp32_k2(x86_avx):
  """Generator for fp32 x86 avx2 for tile_k = 2."""

  def __init__(self, arch="avx2", bits=256, tile_shape=(1, 4, 2)):
    super().__init__(arch, "fp32", "float", bits, tile_shape)
    self.a_type = "float"
    self.b_type = "float"
    # This kernel already has 2 accumulators per tile in m.
    self.min_tiles = max(1, self.min_tiles // 2)

  def header(self):
    return super().header() + """
namespace {

YNN_INTRINSIC __m256 unaligned_load_broadcast_f32x2(const float* ptr) {
    double value;
    memcpy(&value, ptr, sizeof(value));
    return _mm256_castpd_ps(_mm256_set1_pd(value));
}

}  // namespace
"""

  def init_c_tile(self, i, j):
    return f"__m256 c_{i}_{j} = _mm256_setzero_ps();\n"

  def finalize_c_tile(self, i, j):
    if j % 8 != 0:
      return ""
    c_ij = f"c_{i}_{j}"
    mm = self._mm()
    if j + 4 < self.block_shape[1]:
      result = f"{c_ij} = {mm}_hadd_ps({c_ij}, c_{i}_{j+4});\n"
    else:
      result = f"{c_ij} = {mm}_hadd_ps({c_ij}, {mm}_setzero_ps());\n"
    result += (
        f"{c_ij} = {mm}_castpd_ps({mm}_permute4x64_pd({mm}_castps_pd({c_ij}),"
        " 216));\n"
    )
    return result

  def add_c_tile(self, i, j):
    return super().add_c_tile(i, j) if j % 8 == 0 else ""

  def store_c_tile(self, i, j):
    return super().store_c_tile(i, j) if j % 8 == 0 else ""

  def load_a_tile(self, i, k):
    a_ptr = self.a_ptr(i, k)
    return f"__m256 a_{i}_{k} = unaligned_load_broadcast_f32x2({a_ptr});\n"

  def load_b_tile(self, k, j):
    return f"__m256 b_{k}_{j} = _mm256_load_ps({self.b_ptr(k, j)});\n"

  def product(self, i, j, k):
    c_ij = f"c_{i}_{j}"
    mm = self._mm()
    return f"{c_ij} = {mm}_add_ps({c_ij}, {mm}_mul_ps(a_{i}_{k}, b_{k}_{j}));\n"


class x86_avx2_fma3_fp32_k2(x86_avx2_fp32_k2):
  def __init__(self):
    super().__init__("avx2_fma3")

  def product(self, i, j, k):
    return f"c_{i}_{j} = _mm256_fmadd_ps(a_{i}_{k}, b_{k}_{j}, c_{i}_{j});\n"


class x86_avx512_fp32_k2(x86_avx512):
  """Generator for fp32 x86 avx512 for tile_k = 2."""

  def __init__(self, arch="avx512", bits=512, tile_shape=(1, 8, 2)):
    super().__init__(arch, "fp32", "float", bits, tile_shape)
    self.a_type = "float"
    self.b_type = "float"
    # This kernel already has 2 accumulators per tile in m.
    self.min_tiles = max(1, self.min_tiles // 2)

  def header(self):
    return super().header() + """
namespace {

YNN_INTRINSIC __m512 unaligned_load_broadcast_f32x2(const float* ptr) {
    double value;
    memcpy(&value, ptr, sizeof(value));
    return _mm512_castpd_ps(_mm512_set1_pd(value));
}

YNN_INTRINSIC __m512 odd_to_even(__m512 x) {
  return _mm512_castsi512_ps(_mm512_srli_epi64(_mm512_castps_si512(x), 32));
}

}  // namespace
"""

  def init_c_tile(self, i, j):
    return f"__m512 c_{i}_{j} = _mm512_setzero_ps();\n"

  def finalize_c_tile(self, i, j):
    if j % 16 != 0:
      return ""
    even = (
        "_mm512_set_epi32(30, 28, 26, 24, 22, 20, 18, 16, 14, 12, 10, 8, 6, 4,"
        " 2, 0)"
    )
    if j + 8 < self.block_shape[1]:
      # Handle two tiles at once
      return f"""
c_{i}_{j+0} = _mm512_add_ps(c_{i}_{j+0}, odd_to_even(c_{i}_{j+0}));
c_{i}_{j+8} = _mm512_add_ps(c_{i}_{j+8}, odd_to_even(c_{i}_{j+8}));
c_{i}_{j} = _mm512_permutex2var_ps(c_{i}_{j}, {even}, c_{i}_{j+8});
"""
    else:
      # There's only one tile remaining.
      return f"""
c_{i}_{j} = _mm512_add_ps(c_{i}_{j}, odd_to_even(c_{i}_{j}));
c_{i}_{j} = _mm512_permutexvar_ps({even}, c_{i}_{j});
"""

  def add_c_tile(self, i, j):
    return super().add_c_tile(i, j) if j % 16 == 0 else ""

  def store_c_tile(self, i, j):
    return super().store_c_tile(i, j) if j % 16 == 0 else ""

  def load_a_tile(self, i, k):
    a_ptr = self.a_ptr(i, k)
    return f"__m512 a_{i}_{k} = unaligned_load_broadcast_f32x2({a_ptr});\n"

  def load_b_tile(self, k, j):
    return f"__m512 b_{k}_{j} = _mm512_load_ps({self.b_ptr(k, j)});\n"

  def product(self, i, j, k):
    return f"c_{i}_{j} = _mm512_fmadd_ps(a_{i}_{k}, b_{k}_{j}, c_{i}_{j});\n"


generate_dot_kernels(
    x86_avx2_fp32_k2(),
    [
        (1, 16, 2),
        (2, 16, 2),
        (1, 8, 2),
        (3, 8, 2),
        (4, 8, 2),
        (5, 8, 2),
        (6, 8, 2),
        (4, 4, 2),
        (5, 4, 2),
        (6, 4, 2),
        (8, 4, 2),
    ],
)

generate_dot_kernels(
    x86_avx2_fma3_fp32_k2(),
    [
        (1, 16, 2),
        (2, 16, 2),
        (1, 8, 2),
        (3, 8, 2),
        (4, 8, 2),
        (5, 8, 2),
        (6, 8, 2),
        (4, 4, 2),
        (5, 4, 2),
        (6, 4, 2),
        (8, 4, 2),
    ],
)

generate_dot_kernels(
    x86_avx512_fp32_k2(),
    [
        (1, 32, 2),
        (2, 32, 2),
        (3, 32, 2),
        (4, 32, 2),
        (5, 32, 2),
        # The kernels which are commented out should be good, but for some
        # reason they don't perform well. They don't seem to spill, so keeping
        # them until we understand why are they slower.
        # (6, 32, 2),
        (1, 16, 2),
        (4, 16, 2),
        # (5, 16, 2),
        # (6, 16, 2),
        # (8, 16, 2),
        # (10, 16, 2),
        (4, 8, 2),
        (8, 8, 2),
        # (12, 8, 2),
        # (16, 8, 2),
    ],
)
