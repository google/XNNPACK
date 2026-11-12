# Copyright 2026 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""bf16 x86 dot kernel specialized for n == 1."""

# pylint: disable=missing-class-docstring
# pylint: disable=invalid-name

from ynnpack.kernels.dot.generator.dot_base import dot_base
from ynnpack.kernels.dot.generator.dot_base import generate_dot_kernels


class x86_avx2_fma3_bf16_bf16_fp32_k16(dot_base):
  """bf16 AVX2-FMA3 dot kernel specialized for n == 1."""

  def __init__(self):
    super().__init__("avx2_fma3", "bf16_bf16_fp32")
    self.a_type = "bfloat16"
    self.a_load_type = "std::uint16_t"
    self.b_type = "bfloat16"
    self.b_load_type = "std::uint16_t"
    self.c_type = "float"
    self.tile_shape = (1, 1, 16)

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
    if self.block_shape[0] <= 4:
      return (
          f"__m256 c0_{i}_{j} = _mm256_setzero_ps();\n"
          f"__m256 c1_{i}_{j} = _mm256_setzero_ps();\n"
      )
    return f"__m256 c_{i}_{j} = _mm256_setzero_ps();\n"

  def load_a_tile(self, i, k):
    ptr = self.a_ptr(i, k, "__m256i")
    mask = "_mm256_set1_epi32(0xffff0000)"
    return (
        f"__m256i a_raw_{i}_{k} = _mm256_loadu_si256({ptr});\n"
        f"__m256 a0_{i}_{k} ="
        f" _mm256_castsi256_ps(_mm256_slli_epi32(a_raw_{i}_{k}, 16));\n"
        f"__m256 a1_{i}_{k} ="
        f" _mm256_castsi256_ps(_mm256_and_si256(a_raw_{i}_{k}, {mask}));\n"
    )

  def load_b_tile(self, k, j):
    ptr = self.b_ptr(k, j, "__m256i")
    mask = "_mm256_set1_epi32(0xffff0000)"
    return (
        f"__m256i b_raw_{k}_{j} = _mm256_loadu_si256({ptr});\n"
        f"__m256 b0_{k}_{j} ="
        f" _mm256_castsi256_ps(_mm256_slli_epi32(b_raw_{k}_{j}, 16));\n"
        f"__m256 b1_{k}_{j} ="
        f" _mm256_castsi256_ps(_mm256_and_si256(b_raw_{k}_{j}, {mask}));\n"
    )

  def product(self, i, j, k):
    if self.block_shape[0] <= 4:
      return (
          f"c0_{i}_{j} = _mm256_fmadd_ps(a0_{i}_{k}, b0_{k}_{j}, c0_{i}_{j});\n"
          f"c1_{i}_{j} = _mm256_fmadd_ps(a1_{i}_{k}, b1_{k}_{j}, c1_{i}_{j});\n"
      )
    return (
        f"c_{i}_{j} = _mm256_fmadd_ps(a0_{i}_{k}, b0_{k}_{j}, c_{i}_{j});\n"
        f"c_{i}_{j} = _mm256_fmadd_ps(a1_{i}_{k}, b1_{k}_{j}, c_{i}_{j});\n"
    )

  def generate_block(self, a_tail=False, b_tail=False):
    load_a = self.load_a_tile_tail if a_tail else self.load_a_tile
    load_b = self.load_b_tile_tail if b_tail else self.load_b_tile
    result = ""
    for k in range(0, self.block_shape[2], self.tile_shape[2]):
      for j in range(0, self.block_shape[1], self.tile_shape[1]):
        result += load_b(k, j)
      for i in range(0, self.block_shape[0], self.tile_shape[0]):
        result += load_a(i, k)
        for j in range(0, self.block_shape[1], self.tile_shape[1]):
          result += self.product(i, j, k)
    result += "\n"
    return result

  def finalize_c_tile(self, i, j):
    if self.block_shape[0] <= 4:
      return (
          f"float cs_{i}_{j} ="
          f" horizontal_sum(_mm256_add_ps(c0_{i}_{j}, c1_{i}_{j}));\n"
      )
    return f"float cs_{i}_{j} = horizontal_sum(c_{i}_{j});\n"

  def add_c_tile(self, i, j):
    return f"cs_{i}_{j} += *{self.c_in_ptr(i, j)};\n"

  def store_c_tile(self, i, j):
    return f"*{self.c_out_ptr(i, j)} = cs_{i}_{j};\n"

  def add_c_tile_tail(self, i, j, n):
    return self.add_c_tile(i, j)

  def store_c_tile_tail(self, i, j, n):
    return self.store_c_tile(i, j)


generate_dot_kernels(
    x86_avx2_fma3_bf16_bf16_fp32_k16(),
    [
        (1, 1, 16),
        (2, 1, 16),
        (4, 1, 16),
        (8, 1, 16),
    ],
)
