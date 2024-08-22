// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//

#ifndef __XNNPACK_SRC_XNNPACK_SIMD_S8_AVX512F_H_
#define __XNNPACK_SRC_XNNPACK_SIMD_S8_AVX512F_H_

#include <assert.h>
#include <immintrin.h>
#include <stddef.h>
#include <stdint.h>

#include "xnnpack/common.h"
#include "xnnpack/intrinsics-polyfill.h"
#include "xnnpack/unaligned.h"

// SIMD vector type for s8 using SSE41.
typedef __m512i xnn_simd_s8_t;
#define xnn_simd_size_s8 64
#define xnn_simd_log2_size_s8 6
#define xnn_simd_bytes_s8 (xnn_simd_size_s8 * sizeof(int8_t))

#define XNN_SIMD_CONST_S8(var, val) \
  const xnn_simd_s8 var = _mm_set1_epi8(val);

// Arithmetic operations.

static XNN_INLINE xnn_simd_s8_t xnn_add_s8(xnn_simd_s8_t a,
                                             xnn_simd_s8_t b) {
  return _mm512_add_epi8(a, b);
}

static XNN_INLINE xnn_simd_s8_t xnn_max_s8(xnn_simd_s8_t a,
                                             xnn_simd_s8_t b) {
  return _mm512_max_epi8(a, b);
}

static XNN_INLINE xnn_simd_s8_t xnn_min_s8(xnn_simd_s8_t a,
                                             xnn_simd_s8_t b) {
  return _mm512_min_epi8(a, b);
}

static XNN_INLINE __m512i xnn_low_cvt_s8_s16(xnn_simd_s8_t a) {
  return _mm512_cvtepi8_epi16(_mm512_extracti64x4_epi64(a, 0));
}

static XNN_INLINE __m512i xnn_high_cvt_s8_s32(xnn_simd_s8_t a) {
  return _mm512_cvtepi8_epi16(_mm512_extracti64x4_epi64(a, 1));
}

// Load/store operations.

static XNN_INLINE xnn_simd_s8_t xnn_loadu_s8(const int8_t* ptr) {
  return _mm512_loadu_si512(ptr);
}

static XNN_INLINE xnn_simd_s8_t xnn_load_s8(const int8_t* ptr) {
  return _mm512_load_si512(ptr);
}

static XNN_INLINE void xnn_storeu_s8(int8_t* ptr, xnn_simd_s8_t v) {
  _mm512_storeu_si512(ptr, v);
}

static XNN_INLINE void xnn_store_s8(int8_t* ptr, xnn_simd_s8_t v) {
  _mm512_storeu_si512(ptr, v);
}

static XNN_INLINE xnn_simd_s8_t xnn_set1_s8(int8_t v) {
  return _mm512_set1_epi8(v);
}

static XNN_INLINE xnn_simd_s8_t xnn_set1_or_load_s8_t(const int8_t* v) {
  return _mm512_set1_epi8(*v);
}

// Tail load/store operations.

static XNN_INLINE xnn_simd_s8_t
xnn_load_tail_s8(const int8_t* input, size_t num_elements) XNN_OOB_READS {
  assert(num_elements > 0);
  assert(num_elements <= xnn_simd_size_s8);
  return _mm512_loadu_epi8(input);
}

static XNN_INLINE void xnn_store_tail_s8(int8_t* output, xnn_simd_s8_t v,
                                          size_t num_elements) {
  assert(num_elements > 0);
  assert(num_elements < xnn_simd_size_s8);

  const __mmask64 vmask =
      _cvtu64_mask64((uint64_t)((UINT64_C(1) << num_elements) - UINT64_C(1)));
  _mm512_mask_storeu_epi8(output, vmask, v);
}

#endif  // __XNNPACK_SRC_XNNPACK_SIMD_S8_AVX512F_H_
