// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "ynnpack/base/simd/x86_sse41.h"

#include <cassert>
#include <cstddef>
#include <cstdint>

#include "ynnpack/base/simd/multi_vec.h"
#include "ynnpack/kernels/reduce/generic.h"
#include "ynnpack/kernels/reduce/min_max_accumulator.h"
#include "ynnpack/kernels/reduce/sum_accumulator.h"

namespace ynn {

namespace simd {

using s32x4x4 = multi_vec<s32x4, 4>;

s32x4x4& operator+=(s32x4x4& a, s8x16 b) {
  s32x4 b_0(_mm_cvtepi8_epi32(b.v));
  s32x4 b_1(_mm_cvtepi8_epi32(_mm_srli_si128(b.v, 4)));
  s32x4 b_2(_mm_cvtepi8_epi32(_mm_srli_si128(b.v, 8)));
  s32x4 b_3(_mm_cvtepi8_epi32(_mm_srli_si128(b.v, 12)));

  a.v[0] += b_0;
  a.v[1] += b_1;
  a.v[2] += b_2;
  a.v[3] += b_3;
  return a;
}

s32x4x4& operator+=(s32x4x4& a, u8x16 b) {
  s32x4 b_0(_mm_cvtepu8_epi32(b.v));
  s32x4 b_1(_mm_cvtepu8_epi32(_mm_srli_si128(b.v, 4)));
  s32x4 b_2(_mm_cvtepu8_epi32(_mm_srli_si128(b.v, 8)));
  s32x4 b_3(_mm_cvtepu8_epi32(_mm_srli_si128(b.v, 12)));

  a.v[0] += b_0;
  a.v[1] += b_1;
  a.v[2] += b_2;
  a.v[3] += b_3;
  return a;
}

}  // namespace simd

using simd::s32x4;
using simd::s32x4x4;
using simd::s8x16;
using simd::u8x16;

MIN_MAX_KERNEL(min_max_int8_4x16_sse41, s8x16, s8x16, int8_t, 16);
MIN_MAX_KERNEL(min_int8_4x16_sse41, s8x16, dummy_t, int8_t, 16);
MIN_MAX_KERNEL(max_int8_4x16_sse41, dummy_t, s8x16, int8_t, 16);

void sum_int8_int32_sse2(size_t n, size_t k3, size_t k2, size_t k1,
                         size_t a_stride_n, size_t a_stride_k3,
                         size_t a_stride_k2, const void* a, size_t, void* c);
void sum_uint8_int32_sse2(size_t n, size_t k3, size_t k2, size_t k1,
                          size_t a_stride_n, size_t a_stride_k3,
                          size_t a_stride_k2, const void* a, size_t, void* c);

void sum_int8_int32_sse41(size_t n, size_t k3, size_t k2, size_t k1,
                          size_t a_stride_n, size_t a_stride_k3,
                          size_t a_stride_k2, const void* a, size_t, void* c) {
  if (k1 == 1 && a_stride_n == sizeof(int8_t)) {
    tiled_reduce<sum_accumulator_k1_1<s8x16, s32x4x4>, int8_t, int32_t>(
        n, k3, k2, a_stride_k3, a_stride_k2,
        reinterpret_cast<const int8_t*>(a),
        /*C_stride_m=*/0, reinterpret_cast<int32_t*>(c));
  } else {
    sum_int8_int32_sse2(
        n, k3, k2, k1, a_stride_n, a_stride_k3, a_stride_k2, a, 0, c);
  }
}

void sum_uint8_int32_sse41(size_t n, size_t k3, size_t k2, size_t k1,
                           size_t a_stride_n, size_t a_stride_k3,
                           size_t a_stride_k2, const void* a, size_t, void* c) {
  if (k1 == 1 && a_stride_n == sizeof(uint8_t)) {
    tiled_reduce<sum_accumulator_k1_1<u8x16, s32x4x4>, uint8_t, int32_t>(
        n, k3, k2, a_stride_k3, a_stride_k2,
        reinterpret_cast<const uint8_t*>(a),
        /*C_stride_m=*/0, reinterpret_cast<int32_t*>(c));
  } else {
      sum_uint8_int32_sse2(
          n, k3, k2, k1, a_stride_n, a_stride_k3, a_stride_k2, a, 0, c);
  }
}

}  // namespace ynn
