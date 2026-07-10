# Copyright 2026 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""fp32 x86 dot kernel specialized for n == 1."""

# pylint: disable=missing-class-docstring
# pylint: disable=invalid-name

from ynnpack.kernels.dot.generator.dot_base import dot_base
from ynnpack.kernels.dot.generator.dot_base import generate_dot_kernels


class x86_avx2_fma3_fp32_k8(dot_base):
  """fp32 AVX2-FMA3 dot kernel specialized for n == 1."""

  def __init__(self):
    super().__init__("avx2_fma3", "fp32")
    self.a_type = "float"
    self.b_type = "float"
    self.c_type = "float"
    self.tile_shape = (1, 1, 8)

  def header(self):
    return """\
#include <immintrin.h>
""" + super().header() + """
namespace {

YNN_INTRINSIC float horizontal_sum(__m256 x) {
  __m128 lo = _mm256_castps256_ps128(x);
  __m128 hi = _mm256_extractf128_ps(x, 1);
  __m128 sum4 = _mm_add_ps(lo, hi);
  __m128 sum2 = _mm_add_ps(sum4, _mm_movehl_ps(sum4, sum4));
  __m128 sum1 = _mm_add_ss(sum2, _mm_shuffle_ps(sum2, sum2, 1));
  return _mm_cvtss_f32(sum1);
}

}  // namespace
"""

  def init_c_tile(self, i, j):
    return f"__m256 c_{i}_{j} = _mm256_setzero_ps();\n"

  def load_a_tile(self, i, k):
    return f"__m256 a_{i}_{k} = _mm256_loadu_ps({self.a_ptr(i, k)});\n"

  def load_b_tile(self, k, j):
    return f"__m256 b_{k}_{j} = _mm256_loadu_ps({self.b_ptr(k, j)});\n"

  def product(self, i, j, k):
    return f"c_{i}_{j} = _mm256_fmadd_ps(a_{i}_{k}, b_{k}_{j}, c_{i}_{j});\n"

  def finalize_c_tile(self, i, j):
    # Horizontal-sum the 8-wide accumulator (which holds 8 independent
    # partial sums, one per SIMD lane, interleaved across k) down to the
    # single scalar dot product for this row.
    return f"float cs_{i}_{j} = horizontal_sum(c_{i}_{j});\n"

  def add_c_tile(self, i, j):
    return f"cs_{i}_{j} += *{self.c_in_ptr(i, j)};\n"

  def store_c_tile(self, i, j):
    return f"*{self.c_out_ptr(i, j)} = cs_{i}_{j};\n"

  # n is always 1 (tile_shape[1] == block_shape[1] == 1), so the "tail" case
  # (n < tile_shape[1]) never actually triggers at runtime; these just need to
  # be valid, and reusing the main-case codegen is correct since there's only
  # ever one possible n.
  def add_c_tile_tail(self, i, j, n):
    return self.add_c_tile(i, j)

  def store_c_tile_tail(self, i, j, n):
    return self.store_c_tile(i, j)


generate_dot_kernels(
    x86_avx2_fma3_fp32_k8(),
    [
        (1, 1, 8),
        (2, 1, 8),
        (4, 1, 8),
        (8, 1, 8),
    ],
)
