// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//

#ifndef XNNPACK_SRC_XNNPACK_SIMD_S32_SSE41_H_
#define XNNPACK_SRC_XNNPACK_SIMD_S32_SSE41_H_

#include <assert.h>
#include <smmintrin.h>
#include <stddef.h>
#include <stdint.h>

#include "src/xnnpack/common.h"
#include "src/xnnpack/unaligned.h"

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

static XNN_INLINE xnn_simd_s32_t xnn_sub_s32(xnn_simd_s32_t a,
                                             xnn_simd_s32_t b) {
  return _mm_sub_epi32(a, b);
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

static XNN_INLINE void xnn_store_s32(int32_t* ptr, xnn_simd_s32_t v) {
  _mm_store_si128((__m128i*)ptr, v);
}

static XNN_INLINE xnn_simd_s32_t xnn_set1_s32(int32_t v) {
  return _mm_set1_epi32(v);
}

// Tail load/store operations.

static XNN_INLINE xnn_simd_s32_t
xnn_load_tail_s32(const int32_t* input, size_t num_elements) XNN_OOB_READS {
  assert(num_elements <= xnn_simd_size_s32);
  return _mm_loadu_si128((const __m128i*)input);
}

static XNN_INLINE xnn_simd_s32_t xnn_load_tail_safe_s32(const int32_t* input,
                                                        size_t num_elements) {
  assert(num_elements <= xnn_simd_size_s32);

  XNN_ALIGN(16) int32_t padded[4];
  int32_t* dst = padded;
  switch (num_elements) {
    case 4:
      *dst++ = *input++;
      XNN_FALLTHROUGH
    case 3:
      *dst++ = *input++;
      XNN_FALLTHROUGH
    case 2:
      *dst++ = *input++;
      XNN_FALLTHROUGH
    case 1:
      *dst++ = *input++;
      XNN_FALLTHROUGH
    default:
      break;
  }
  return _mm_loadu_si128((const __m128i*)padded);
}

static XNN_INLINE void xnn_store_tail_s32(int32_t* output, xnn_simd_s32_t v,
                                          size_t num_elements) {
  assert(num_elements <= xnn_simd_size_s32);

  if (num_elements & 2) {
    _mm_storel_epi64((__m128i*)output, v);
    v = _mm_unpackhi_epi64(v, v);
    output += 2;
  }
  if (num_elements & 1) {
    unaligned_store_u32(output, (uint32_t)_mm_cvtsi128_si32(v));
  }
}

// Conversion operations.

static XNN_INLINE __m128 xnn_cvt_f32_s32(xnn_simd_s32_t a) {
  return _mm_cvtepi32_ps(a);
}

#endif  // XNNPACK_SRC_XNNPACK_SIMD_S32_SSE41_H_
