# Copyright 2025 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Specializations for fp32 x86 dot kernel generators."""

# pylint: disable=missing-class-docstring
# pylint: disable=invalid-name

from ynnpack.kernels.dot.generator.x86 import x86_avx512


class x86_avx512_fp32_k4(x86_avx512):
  """Generator for fp32 x86 avx512 for tile_k = 4."""

  def __init__(self, arch="avx512", bits=512, tile_shape=(1, 4, 4)):
    super().__init__(arch, "fp32", "float", bits, tile_shape)
    self.a_type = "float"
    self.b_type = "float"

  def header(self):
    return super().header() + """
namespace {

YNN_INTRINSIC __m512 unaligned_load_broadcast_x4(const float* ptr) {
    __m128 value = _mm_loadu_ps(ptr);
    return _mm512_broadcast_f32x4(value);
}

YNN_INTRINSIC __m512 odd_to_even(__m512 x) {
  return _mm512_castsi512_ps(_mm512_srli_epi64(_mm512_castps_si512(x), 32));
}

YNN_INTRINSIC __m512 high_to_low(__m512 x) {
  return _mm512_castps256_ps512(_mm256_castpd_ps(_mm512_extractf64x4_pd(_mm512_castps_pd(x), 1)));
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
      # Handle 1x4 tiles at once
      return f"""
c_{i+0}_{j+0} = _mm512_add_ps(c_{i+0}_{j+0}, odd_to_even(c_{i+0}_{j+0}));
c_{i+0}_{j+4} = _mm512_add_ps(c_{i+0}_{j+4}, odd_to_even(c_{i+0}_{j+4}));
c_{i+0}_{j+8} = _mm512_add_ps(c_{i+0}_{j+8}, odd_to_even(c_{i+0}_{j+8}));
c_{i+0}_{j+12} = _mm512_add_ps(c_{i+0}_{j+12}, odd_to_even(c_{i+0}_{j+12}));
c_{i+0}_{j+0} = _mm512_permutex2var_ps(c_{i+0}_{j+0}, {even}, c_{i+0}_{j+4});
c_{i+0}_{j+8} = _mm512_permutex2var_ps(c_{i+0}_{j+8}, {even}, c_{i+0}_{j+12});
c_{i+0}_{j+0} = _mm512_add_ps(c_{i+0}_{j+0}, odd_to_even(c_{i+0}_{j+0}));
c_{i+0}_{j+8} = _mm512_add_ps(c_{i+0}_{j+8}, odd_to_even(c_{i+0}_{j+8}));
c_{i+0}_{j} = _mm512_permutex2var_ps(c_{i+0}_{j}, {even}, c_{i+0}_{j+8});
"""
    elif j + 4 < self.block_shape[1]:
      if i % 2 != 0:
        return ""
      # Handle 2x2 tiles at once
      return f"""
c_{i+0}_{j+0} = _mm512_add_ps(c_{i+0}_{j+0}, odd_to_even(c_{i+0}_{j+0}));
c_{i+0}_{j+4} = _mm512_add_ps(c_{i+0}_{j+4}, odd_to_even(c_{i+0}_{j+4}));
c_{i+1}_{j+0} = _mm512_add_ps(c_{i+1}_{j+0}, odd_to_even(c_{i+1}_{j+0}));
c_{i+1}_{j+4} = _mm512_add_ps(c_{i+1}_{j+4}, odd_to_even(c_{i+1}_{j+4}));
c_{i+0}_{j+0} = _mm512_permutex2var_ps(c_{i+0}_{j+0}, {even}, c_{i+0}_{j+4});
c_{i+1}_{j+0} = _mm512_permutex2var_ps(c_{i+1}_{j+0}, {even}, c_{i+1}_{j+4});
c_{i+0}_{j+0} = _mm512_add_ps(c_{i+0}_{j+0}, odd_to_even(c_{i+0}_{j+0}));
c_{i+1}_{j+0} = _mm512_add_ps(c_{i+1}_{j+0}, odd_to_even(c_{i+1}_{j+0}));
c_{i+0}_{j+0} = _mm512_permutex2var_ps(c_{i+0}_{j+0}, {even}, c_{i+1}_{j+0+0});
c_{i+1}_{j+0} = high_to_low(c_{i+0}_{j+0});
"""
    else:
      # There's only one tile remaining.
      return f"""
c_{i+0}_{j+0} = _mm512_add_ps(c_{i+0}_{j+0}, odd_to_even(c_{i+0}_{j+0}));
c_{i+0}_{j+0} = _mm512_permutexvar_ps({even}, c_{i+0}_{j});
c_{i+0}_{j+0} = _mm512_add_ps(c_{i+0}_{j+0}, odd_to_even(c_{i+0}_{j+0}));
c_{i+0}_{j} = _mm512_permutexvar_ps({even}, c_{i+0}_{j});
"""

  def add_c_tile(self, i, j):
    return super().add_c_tile(i, j) if j % 16 == 0 else ""

  def store_c_tile(self, i, j):
    return super().store_c_tile(i, j) if j % 16 == 0 else ""

  def load_a_tile(self, i, k):
    return (
        f"__m512 a_{i}_{k} = unaligned_load_broadcast_x4({self.a_ptr(i, k)});\n"
    )

  def load_b_tile(self, k, j):
    return f"__m512 b_{k}_{j} = _mm512_load_ps({self.b_ptr(k, j)});\n"

  def product(self, i, j, k):
    return f"c_{i}_{j} = _mm512_fmadd_ps(a_{i}_{k}, b_{k}_{j}, c_{i}_{j});\n"
