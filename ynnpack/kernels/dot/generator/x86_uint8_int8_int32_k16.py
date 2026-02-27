# Copyright 2025 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Specializations for uint8 x86 dot kernel generators."""

from ynnpack.kernels.dot.generator.x86 import x86_avx


# In this generator, we tell the base class that we are generating 128-bit
# tiles, but we accumulate in 512-bit vectors.
class x86_avx512vnni_uint8_int8_int32_k16(x86_avx):  # pylint: disable=invalid-name
  """Generates tile_k=16 avx512vnni dot kernels."""

  def __init__(self):
    super().__init__(
        "avx512vnni", "uint8_int8_int32", "int32_t", 128, (1, 4, 16)
    )
    self.a_type = "uint8_t"
    self.b_type = "int8_t"
    self.flags += ["dot_flag::consistent_arithmetic"]
    # This kernel already has 2 accumulators per tile.
    self.min_tiles = max(1, self.min_tiles // 4)

  def header(self):
    return super().header() + """

namespace {

YNN_INTRINSIC __m512i _mm512_hadd_epi32(__m512i a, __m512i b) {
    __m512i even = _mm512_permutex2var_epi32(a, _mm512_set_epi32(30, 28, 26, 24, 22, 20, 18, 16, 14, 12, 10, 8, 6, 4, 2, 0), b);
    __m512i odd = _mm512_permutex2var_epi32(a, _mm512_set_epi32(31, 29, 27, 25, 23, 21, 19, 17, 15, 13, 11, 9, 7, 5, 3, 1), b);
    return _mm512_add_epi32(even, odd);
}

}  // namespace
"""

  def init_c_tile(self, i, j):
    if i % 4 != 0:
      return ""
    result = f"""
__m512i c16_{i+0}_{j+0} = _mm512_setzero_si512();
__m512i c16_{i+1}_{j+0} = _mm512_setzero_si512();
__m512i c16_{i+2}_{j+0} = _mm512_setzero_si512();
__m512i c16_{i+3}_{j+0} = _mm512_setzero_si512();
"""
    return result

  # Here, we process 4 rows of 512-bit accumulators, and turn them into 4 rows
  # of 128-bit outputs.
  def finalize_c_tile(self, i, j):
    if i % 4 != 0:
      return ""
    return f"""
c16_{i+0}_{j} = _mm512_hadd_epi32(c16_{i+0}_{j}, c16_{i+1}_{j});
c16_{i+2}_{j} = _mm512_hadd_epi32(c16_{i+2}_{j}, c16_{i+3}_{j});
c16_{i+0}_{j} = _mm512_hadd_epi32(c16_{i+0}_{j}, c16_{i+2}_{j});
__m128i c_{i+0}_{j} = _mm512_extracti32x4_epi32(c16_{i}_{j}, 0);
__m128i c_{i+1}_{j} = _mm512_extracti32x4_epi32(c16_{i}_{j}, 1);
__m128i c_{i+2}_{j} = _mm512_extracti32x4_epi32(c16_{i}_{j}, 2);
__m128i c_{i+3}_{j} = _mm512_extracti32x4_epi32(c16_{i}_{j}, 3);
"""

  def load_a_tile(self, i, k):
    return (
        f"__m512i a_{i}_{k} ="
        f" _mm512_broadcast_i32x4(_mm_loadu_si128({self.a_ptr(i, k, '__m128i')}));\n"
    )

  def load_b_tile(self, k, j):
    return (
        f"__m512i b_{k}_{j} ="
        f" _mm512_load_si512({self.b_ptr(k, j, '__m512i')});\n"
    )

  def product(self, i, j, k):
    return (
        f"c16_{i}_{j} = _mm512_dpbusd_epi32(c16_{i}_{j}, a_{i}_{k},"
        f" b_{k}_{j});\n"
    )
