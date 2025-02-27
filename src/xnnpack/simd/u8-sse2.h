// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//

#ifndef __XNNPACK_SRC_XNNPACK_SIMD_U8_SSE2_H_
#define __XNNPACK_SRC_XNNPACK_SIMD_U8_SSE2_H_

#include <assert.h>
#include <smmintrin.h>
#include <stddef.h>
#include <stdint.h>

#include "xnnpack/common.h"
#include "xnnpack/intrinsics-polyfill.h"

// SIMD vector type for u8 using SSE41.
typedef __m128i xnn_simd_u8_t;
#define xnn_simd_size_u8 16
#define xnn_simd_log2_size_u8 0
#define xnn_simd_bytes_u8 (xnn_simd_size_u8 * sizeof(uint8_t))

#define XNN_SIMD_CONST_U8(var, val) \
  const xnn_simd_u8_t var = _mm_set1_epi8(val);

// Arithmetic operations.

static XNN_INLINE xnn_simd_u8_t xnn_add_u8(xnn_simd_u8_t a, xnn_simd_u8_t b) {
  return _mm_add_epi8(a, b);
}

static XNN_INLINE xnn_simd_u8_t xnn_max_u8(xnn_simd_u8_t a, xnn_simd_u8_t b) {
  return _mm_max_epu8(a, b);
}

static XNN_INLINE xnn_simd_u8_t xnn_min_u8(xnn_simd_u8_t a, xnn_simd_u8_t b) {
  return _mm_min_epu8(a, b);
}

static XNN_INLINE xnn_simd_u8_t xnn_xor_u8(xnn_simd_u8_t a, xnn_simd_u8_t b) {
  return _mm_xor_si128(a, b);
}

// Load/store operations.

static XNN_INLINE xnn_simd_u8_t xnn_loadu_u8(const uint8_t* ptr) {
  return _mm_loadu_si128((const __m128i*)ptr);
}

static XNN_INLINE xnn_simd_u8_t xnn_load_u8(const uint8_t* ptr) {
  return _mm_load_si128((const __m128i*)ptr);
}

static XNN_INLINE void xnn_storeu_u8(uint8_t* ptr, xnn_simd_u8_t v) {
  _mm_storeu_si128((__m128i*)ptr, v);
}

static XNN_INLINE void xnn_store_u8(float* ptr, xnn_simd_u8_t v) {
  _mm_store_si128((__m128i*)ptr, v);
}

static XNN_INLINE xnn_simd_u8_t xnn_set1_u8(uint8_t v) {
  return _mm_set1_epi8(v);
}

static XNN_INLINE xnn_simd_u8_t xnn_set1_or_load_u8(const uint8_t* v) {
#if XNN_ARCH_X86
  return _mm_load_si128((const __m128i*)v);
#else
  return _mm_set1_epi8(*v);
#endif
}

// Tail load/store operations.

static XNN_INLINE xnn_simd_u8_t
xnn_load_tail_u8(const uint8_t* input, size_t num_elements) XNN_OOB_READS {
  assert(num_elements > 0);
  assert(num_elements < xnn_simd_size_u8);
  return _mm_loadu_si128((const __m128i*)input);
}

static XNN_INLINE void xnn_store_tail_u8(uint8_t* output, xnn_simd_u8_t v,
                                         size_t num_elements) {
  assert(num_elements > 0);
  assert(num_elements < xnn_simd_size_u8);

  if (num_elements & (8 * sizeof(uint8_t))) {
    _mm_storel_epi64((__m128i*)output, v);
    v = _mm_unpackhi_epi64(v, v);
    output += 8;
  }
  if (num_elements & (4 * sizeof(uint8_t))) {
    _mm_storeu_si32(output, v);
    v = _mm_srli_epi64(v, 32);
    output += 4;
  }
  if (num_elements & (2 * sizeof(uint8_t))) {
    _mm_storeu_si16(output, v);
    v = _mm_srli_epi32(v, 16);
    output += 2;
  }
  if (num_elements & (1 * sizeof(uint8_t))) {
    *output = (uint8_t)_mm_cvtsi128_si32(v);
  }
}

#endif  // __XNNPACK_SRC_XNNPACK_SIMD_U8_SSE2_H_
