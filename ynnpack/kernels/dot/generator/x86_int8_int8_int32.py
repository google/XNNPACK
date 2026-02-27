# Copyright 2025 Google LLC
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Specializations for int8 x86 dot kernel generators."""

# pylint: disable=missing-class-docstring
# pylint: disable=invalid-name

from ynnpack.kernels.dot.generator.x86 import x86_avx
from ynnpack.kernels.dot.generator.x86 import x86_avx512


class x86_avx2_int8_int8_int32(x86_avx):
  def __init__(self, arch="avx2", vector_bits=256):
    super().__init__(arch, "int8_int8_int32", "int32_t", vector_bits, tile_shape=(1, 8, 4))
    self.a_type = "int8_t"
    self.b_type = "int8_t"
    self.flags += ["dot_flag::consistent_arithmetic"]
    # This kernel already has 2 accumulators per tile in m.
    self.min_tiles = max(1, self.min_tiles // 2)

  def header(self):
    return super().header() + """

namespace {

YNN_INTRINSIC int32_t unaligned_load_int8x4(const void* ptr) {
    int32_t value;
    memcpy(&value, ptr, sizeof(int32_t));
    return value;
}

}  // namespace
"""

  def b_alignment_required(self):
    # This kernel loads half-vectors at a time from b.
    return self.tile_shape[1] * self.tile_shape[2] // 2

  # In this kernel, we load 4 values of A and B at a time, and do a 2-way dot
  # product, resulting in 2 values (so we need 2 accumulator registers per
  # tile). We accumulate these 2 values for each value of k, only summing the
  # result into the final result tile after the loops over k.
  def finalize_c_tile(self, i, j):
    # Horizontally add pairs of values, and swap the middle two 64-bits of each
    # 128-bit result.
    return f"""
c_{i}_{j+0} = {self._mm()}_hadd_epi32(c_{i}_{j+0}, c_{i}_{j+4});
c_{i}_{j} = _mm256_permute4x64_epi64(c_{i}_{j}, 216);
"""

  def init_c_tile(self, i, j):
    return f"""
__m{self.bits}i c_{i}_{j+0} = {self._mm()}_setzero_si{self.bits}();
__m{self.bits}i c_{i}_{j+4} = {self._mm()}_setzero_si{self.bits}();
"""

  def load_a_tile(self, i, k):
    return f"__m{self.bits}i a_{i}_{k} = {self._mm()}_cvtepi8_epi16({self._mm(self.bits//2)}_set1_epi32(unaligned_load_int8x4({self.a_ptr(i, k)})));\n"

  def load_b_tile(self, k, j):
    return f"""
__m{self.bits}i b_{k}_{j+0} = {self._mm()}_cvtepi8_epi16({self._mm(self.bits//2)}_load_si{self.bits//2}({self.b_ptr(k, j+0, f'__m{self.bits//2}i')}));
__m{self.bits}i b_{k}_{j+4} = {self._mm()}_cvtepi8_epi16({self._mm(self.bits//2)}_load_si{self.bits//2}({self.b_ptr(k, j+4, f'__m{self.bits//2}i')}));
"""

  def product(self, i, j, k):
    return f"""
c_{i}_{j+0} = {self._mm()}_add_epi32(c_{i}_{j+0}, {self._mm()}_madd_epi16(a_{i}_{k}, b_{k}_{j+0}));
c_{i}_{j+4} = {self._mm()}_add_epi32(c_{i}_{j+4}, {self._mm()}_madd_epi16(a_{i}_{k}, b_{k}_{j+4}));
"""


class x86_avx512_int8_int8_int32(x86_avx512):

  def __init__(self, arch="avx512", vector_bits=512):
    super().__init__(arch, "int8_int8_int32", "int32_t", vector_bits, tile_shape=(1, 16, 4))
    self.a_type = "int8_t"
    self.b_type = "int8_t"
    self.flags += ["dot_flag::consistent_arithmetic"]
    # This kernel already has 2 accumulators per tile.
    self.min_tiles = max(1, self.min_tiles // 2)

  def header(self):
    return super().header() + """

namespace {

YNN_INTRINSIC int32_t unaligned_load_int8x4(const void* ptr) {
    int32_t value;
    memcpy(&value, ptr, sizeof(int32_t));
    return value;
}

YNN_INTRINSIC __m512i _mm512_hadd_epi32(__m512i a, __m512i b) {
    __m512i even = _mm512_permutex2var_epi32(a, _mm512_set_epi32(30, 28, 26, 24, 22, 20, 18, 16, 14, 12, 10, 8, 6, 4, 2, 0), b);
    __m512i odd = _mm512_permutex2var_epi32(a, _mm512_set_epi32(31, 29, 27, 25, 23, 21, 19, 17, 15, 13, 11, 9, 7, 5, 3, 1), b);
    return _mm512_add_epi32(even, odd);
}

}  // namespace
"""

  def b_alignment_required(self):
    # This kernel loads half-vectors at a time from b.
    return self.tile_shape[1] * self.tile_shape[2] // 2

  # In this kernel, we load 4 values of A and B at a time, and do a 2-way dot
  # product, resulting in 2 values (so we need 2 accumulator registers per
  # tile). We accumulate these 2 values for each value of k, only summing the
  # result into the final result tile after the loops over k.
  def finalize_c_tile(self, i, j):
    return f"""
c_{i}_{j+0} = {self._mm()}_hadd_epi32(c_{i}_{j+0}, c_{i}_{j+8});
"""

  def init_c_tile(self, i, j):
    return f"""
__m{self.bits}i c_{i}_{j+0} = {self._mm()}_setzero_si{self.bits}();
__m{self.bits}i c_{i}_{j+8} = {self._mm()}_setzero_si{self.bits}();
"""

  def load_a_tile(self, i, k):
    return f"__m{self.bits}i a_{i}_{k} = {self._mm()}_cvtepi8_epi16({self._mm(self.bits//2)}_set1_epi32(unaligned_load_int8x4({self.a_ptr(i, k)})));\n"

  def load_b_tile(self, k, j):
    return f"""
__m{self.bits}i b_{k}_{j+0} = {self._mm()}_cvtepi8_epi16({self._mm(self.bits//2)}_load_si{self.bits//2}({self.b_ptr(k, j+0, f'__m{self.bits//2}i')}));
__m{self.bits}i b_{k}_{j+8} = {self._mm()}_cvtepi8_epi16({self._mm(self.bits//2)}_load_si{self.bits//2}({self.b_ptr(k, j+8, f'__m{self.bits//2}i')}));
"""

  def product(self, i, j, k):
    return f"""
c_{i}_{j+0} = {self._mm()}_add_epi32(c_{i}_{j+0}, {self._mm()}_madd_epi16(a_{i}_{k}, b_{k}_{j+0}));
c_{i}_{j+8} = {self._mm()}_add_epi32(c_{i}_{j+8}, {self._mm()}_madd_epi16(a_{i}_{k}, b_{k}_{j+8}));
"""
