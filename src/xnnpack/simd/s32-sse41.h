// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//

#ifndef __XNNPACK_SRC_XNNPACK_SIMD_S32_SSE2_H_
#define __XNNPACK_SRC_XNNPACK_SIMD_S32_SSE2_H_

#include <assert.h>
#include <smmintrin.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include "xnnpack/common.h"
#include "xnnpack/unaligned.h"

// SIMD vector type for s32 using SSE41.
typedef __m128i xnn_simd_s32_t;
#define xnn_simd_size_s32 4
#define xnn_simd_log2_size_s32 2
#define xnn_simd_bytes_s32 (xnn_simd_size_s32 * sizeof(int32_t))

#define XNN_SIMD_CONST_S32(var, val) \
  const xnn_simd_s32_t var = _mm_set1_epi32(val);

// Arithmetic operations.

static XNN_INLINE xnn_simd_s32_t xnn_mul_s32(xnn_simd_s32_t a,
                                             xnn_simd_s32_t b) {
  return _mm_mullo_epi32(a, b);
}

static XNN_INLINE xnn_simd_s32_t xnn_max_s32(xnn_simd_s32_t a,
                                             xnn_simd_s32_t b) {
  return _mm_max_epi32(a, b);
}

static XNN_INLINE xnn_simd_s32_t xnn_min_s32(xnn_simd_s32_t a,
                                             xnn_simd_s32_t b) {
  return _mm_min_epi32(a, b);
}

static XNN_INLINE xnn_simd_s32_t xnn_clz_s32(xnn_simd_s32_t a) {

  xnn_simd_s32_t shuffled = _mm_shuffle_epi32(a, _MM_SHUFFLE(2, 3, 0, 1));
  xnn_simd_s32_t low_half = _mm_unpacklo_epi32(a, shuffled);
  xnn_simd_s32_t high_half = _mm_unpackhi_epi32(a, shuffled);
  __m128d low = _mm_cvtepi32_pd(low_half);
  __m128d high = _mm_cvtepi32_pd(high_half);
  xnn_simd_s32_t low_a = _mm_castpd_si128(low);
  xnn_simd_s32_t high_a = _mm_castpd_si128(high);

  xnn_simd_s32_t shift_low = _mm_srli_epi64(low_a, 52);
  xnn_simd_s32_t shift_high = _mm_srli_epi64(high_a, 52);

  xnn_simd_s32_t low_exp =
      _mm_shuffle_epi32(shift_low, _MM_SHUFFLE(3, 1, 2, 0));
  xnn_simd_s32_t high_exp =
      _mm_shuffle_epi32(shift_high, _MM_SHUFFLE(2, 0, 3, 1));

  xnn_simd_s32_t exponent = _mm_blend_epi16(low_exp, high_exp, 0b11110000);

  exponent = _mm_and_si128(exponent, _mm_set1_epi32(0x7FF));

  xnn_simd_s32_t result = _mm_sub_epi32(_mm_set1_epi32(31), _mm_sub_epi32(exponent, _mm_set1_epi32(1023)));

  xnn_simd_s32_t zero = _mm_setzero_si128();
  xnn_simd_s32_t mask = _mm_cmpgt_epi32(zero, a);
  result = _mm_andnot_si128(mask, result);

  xnn_simd_s32_t thirty_two = _mm_set1_epi32(32);
  xnn_simd_s32_t maskz = _mm_cmpeq_epi32(a, zero);

  result = _mm_blendv_epi8(result, thirty_two, maskz);

  return result;
}

// Load/store operations.

static XNN_INLINE xnn_simd_s32_t xnn_loadu_s32(const int32_t* ptr) {
  return _mm_loadu_si128((const __m128i*)ptr);
}

static XNN_INLINE xnn_simd_s32_t xnn_load_s32(const int32_t* ptr) {
  return _mm_load_si128((const __m128i*)ptr);
}

static XNN_INLINE void xnn_storeu_s32(int32_t* ptr, xnn_simd_s32_t v) {
  _mm_storeu_si128((__m128i*)ptr, v);
}

static XNN_INLINE void xnn_store_s32(float* ptr, xnn_simd_s32_t v) {
  _mm_store_si128((__m128i*)ptr, v);
}

static XNN_INLINE xnn_simd_s32_t xnn_set1_s32(int32_t v) {
  return _mm_set1_epi32(v);
}

static XNN_INLINE xnn_simd_s32_t xnn_set1_or_load_s32(const int32_t* v) {
#if XNN_ARCH_X86
  return _mm_load_si128((const __m128i*)v);
#else
  return _mm_set1_epi32(*v);
#endif
}

// Tail load/store operations.

static XNN_INLINE xnn_simd_s32_t
xnn_load_tail_s32(const int32_t* input, size_t num_elements) XNN_OOB_READS {
  assert(num_elements > 0);
  assert(num_elements < xnn_simd_size_s32);
  return _mm_loadu_si128((const __m128i*)input);
}

static XNN_INLINE void xnn_store_tail_s32(int32_t* output, xnn_simd_s32_t v,
                                          size_t num_elements) {
  assert(num_elements > 0);
  assert(num_elements < xnn_simd_size_s32);

  if (num_elements & 2) {
    _mm_storel_epi64((__m128i*)output, v);
    v = _mm_unpackhi_epi64(v, v);
    output += 2;
  }
  if (num_elements & 1) {
    unaligned_store_u32(output, (uint32_t)_mm_cvtsi128_si32(v));
  }
}

#endif  // __XNNPACK_SRC_XNNPACK_SIMD_S32_SSE2_H_
