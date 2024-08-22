// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//

#ifndef __XNNPACK_SRC_XNNPACK_SIMD_S32_AVX512F_H_
#define __XNNPACK_SRC_XNNPACK_SIMD_S32_AVX512F_H_

#include <assert.h>
#include <immintrin.h>
#include <stddef.h>
#include <stdint.h>

#include "xnnpack/common.h"
#include "xnnpack/intrinsics-polyfill.h"
#include "xnnpack/unaligned.h"

// SIMD vector type for s32 using SSE41.
typedef __m512i xnn_simd_s32_t;
#define xnn_simd_size_s32 16
#define xnn_simd_log2_size_s32 4
#define xnn_simd_bytes_s32 (xnn_simd_size_s32 * sizeof(int32_t))

#define XNN_SIMD_CONST_S32(var, val) \
  const xnn_simd_s32_t var = _mm_set1_epi32(val);

// Arithmetic operations.

static XNN_INLINE xnn_simd_s32_t xnn_mul_s32(xnn_simd_s32_t a,
                                             xnn_simd_s32_t b) {
  return _mm512_mullo_epi32(a, b);
}

static XNN_INLINE xnn_simd_s32_t xnn_max_s32(xnn_simd_s32_t a,
                                             xnn_simd_s32_t b) {
  return _mm512_max_epi32(a, b);
}

static XNN_INLINE xnn_simd_s32_t xnn_min_s32(xnn_simd_s32_t a,
                                             xnn_simd_s32_t b) {
  return _mm512_min_epi32(a, b);
}

static XNN_INLINE xnn_simd_s32_t xnn_clz_s32(xnn_simd_s32_t a) {
  __m256i lowi = _mm512_extracti64x4_epi64(a, 0);
  __m256i highi = _mm512_extracti64x4_epi64(a, 1);

  __m512d low = _mm512_cvtepi32_pd(lowi);
  __m512d high = _mm512_cvtepi32_pd(highi);

  xnn_simd_s32_t low_a = _mm512_castpd_si512(low);
  xnn_simd_s32_t high_a = _mm512_castpd_si512(high);

  xnn_simd_s32_t shift_low = _mm512_srli_epi64(low_a, 52);
  xnn_simd_s32_t shift_high =  _mm512_srli_epi64(high_a, 52);

  xnn_simd_s32_t mask =
      _mm512_setr_epi32(1, 3, 5, 7, 9, 11, 13, 15, 0, 2, 4, 6, 8, 10, 12, 14);
  xnn_simd_s32_t mlow = _mm512_permutexvar_epi32(mask, shift_low);
  xnn_simd_s32_t mhigh = _mm512_permutexvar_epi32(mask, shift_high);

  __m256i ai = _mm512_extracti64x4_epi64(mlow, 1);
  __m256i bi = _mm512_extracti64x4_epi64(mhigh, 1);

  xnn_simd_s32_t exponent =
      _mm512_inserti64x4(_mm512_castsi256_si512(ai), bi, 1);

  exponent = _mm512_and_epi32(exponent, _mm512_set1_epi32(0x7FF));

  xnn_simd_s32_t result =
      _mm512_sub_epi32(_mm512_set1_epi32(31),
                       _mm512_sub_epi32(exponent, _mm512_set1_epi32(1023)));

  xnn_simd_s32_t zero = _mm512_setzero_si512();

  __mmask16 maskl = _mm512_cmpge_epi32_mask(a, zero);
  result = _mm512_maskz_mov_epi32(maskl, result);
  __mmask16 maskz = _mm512_cmpeq_epi32_mask(a, zero);
  result = _mm512_mask_mov_epi32(result, maskz, _mm512_set1_epi32(32));
  return result;
}

// Load/store operations.

static XNN_INLINE xnn_simd_s32_t xnn_loadu_s32(const int32_t* ptr) {
  return _mm512_loadu_si512(ptr);
}

static XNN_INLINE xnn_simd_s32_t xnn_load_s32(const int32_t* ptr) {
  return _mm512_load_si512(ptr);
}

static XNN_INLINE void xnn_storeu_s32(int32_t* ptr, xnn_simd_s32_t v) {
  _mm512_storeu_si512((__m512i*)ptr, v);
}

static XNN_INLINE void xnn_store_s32(float* ptr, xnn_simd_s32_t v) {
  _mm512_store_si512((__m512i*)ptr, v);
}

static XNN_INLINE xnn_simd_s32_t xnn_set1_s32(int32_t v) {
  return _mm512_set1_epi32(v);
}

static XNN_INLINE xnn_simd_s32_t xnn_set1_or_load_s32(const int32_t* v) {
#if XNN_ARCH_X86
  return _mm512_load_epi32((const __m128i*)v);
#else
  return _mm512_set1_epi32(*v);
#endif
}

// Tail load/store operations.

static XNN_INLINE xnn_simd_s32_t
xnn_load_tail_s32(const int32_t* input, size_t num_elements) XNN_OOB_READS {
  assert(num_elements > 0);
  assert(num_elements < xnn_simd_size_s32);
  const __mmask16 vmask = _cvtu32_mask16((uint32_t) ((UINT32_C(1) << num_elements) - UINT32_C(1)));
  return _mm512_maskz_loadu_epi32(vmask, input);
}

static XNN_INLINE void xnn_store_tail_s32(int32_t* output, xnn_simd_s32_t v,
                                          size_t num_elements) {
  assert(num_elements > 0);
  assert(num_elements < xnn_simd_size_s32);

  const __mmask16 vmask =
      _cvtu32_mask16((uint32_t)((UINT32_C(1) << num_elements) - UINT32_C(1)));
  _mm512_mask_storeu_epi32(output, vmask, v);
}

#endif  // __XNNPACK_SRC_XNNPACK_SIMD_S32_AVX512F_H_
