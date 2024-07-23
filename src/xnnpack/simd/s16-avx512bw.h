// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//

#ifndef __XNNPACK_SRC_XNNPACK_SIMD_S16_AVX512F_H_
#define __XNNPACK_SRC_XNNPACK_SIMD_S16_AVX512F_H_

#include <assert.h>
#include <immintrin.h>
#include <stddef.h>
#include <stdint.h>

#include "xnnpack/common.h"
#include "xnnpack/intrinsics-polyfill.h"
#include "xnnpack/unaligned.h"

// SIMD vector type for s16 using SSE41.
typedef __m512i xnn_simd_s16_t;
#define xnn_simd_size_s16 32
#define xnn_simd_log2_size_s16 5
#define xnn_simd_bytes_s16 (xnn_simd_size_s16 * sizeof(int16_t))

#define XNN_SIMD_CONST_S16(var, val) \
  const xnn_simd_s16 var = _mm_set1_epi16(val);

// Arithmetic operations.
static XNN_INLINE __m512i xnn_low_cvt_s16_s32(xnn_simd_s16_t a) {
  return _mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64(a, 0));
}

static XNN_INLINE __m512i xnn_high_cvt_s16_s32(xnn_simd_s16_t a) {
  return _mm512_cvtepi16_epi32(_mm512_extracti64x4_epi64(a, 1));
}

// Load/store operations.

static XNN_INLINE xnn_simd_s16_t xnn_loadu_s16(const int16_t* ptr) {
  return _mm512_loadu_epi16(ptr);
}

static XNN_INLINE xnn_simd_s16_t xnn_load_s16(const int16_t* ptr) {
  return _mm512_load_si512(ptr);
}

static XNN_INLINE void xnn_storeu_s16(int16_t* ptr, xnn_simd_s16_t v) {
  _mm512_storeu_epi16(ptr, v);
}

static XNN_INLINE void xnn_store_s16(int16_t* ptr, xnn_simd_s16_t v) {
  _mm512_storeu_si512(ptr, v);
}

static XNN_INLINE xnn_simd_s16_t xnn_set1_s16(int16_t v) {
  return _mm512_set1_epi16(v);
}

static XNN_INLINE xnn_simd_s16_t xnn_set1_or_load_s16(const int16_t* v) {
#if XNN_ARCH_X86
  return _mm512_load_epiint16_t((const __m128i*)v);
#else
  return _mm512_set1_epi16(*v);
#endif
}

// Tail load/store operations.

static XNN_INLINE xnn_simd_s16_t
xnn_load_tail_s16(const int16_t* input, size_t num_elements) XNN_OOB_READS {
  assert(num_elements > 0);
  assert(num_elements < xnn_simd_size_s16);
  return _mm512_loadu_epi16(input);
}

static XNN_INLINE void xnn_store_tail_s16(int16_t* output, xnn_simd_s16_t v,
                                          size_t num_elements) {
  assert(num_elements > 0);
  assert(num_elements < xnn_simd_size_s16);

  const __mmask32 vmask =
      _cvtu32_mask32((uint32_t)((UINT32_C(1) << num_elements) - UINT32_C(1)));
  _mm512_mask_storeu_epi16(output, vmask, v);
}

#endif  // __XNNPACK_SRC_XNNPACK_SIMD_S16_AVX512F_H_
