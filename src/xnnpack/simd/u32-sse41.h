// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//

#ifndef __XNNPACK_SRC_XNNPACK_SIMD_U32_SSE41_H_
#define __XNNPACK_SRC_XNNPACK_SIMD_U32_SSE41_H_

#include <assert.h>
#include <smmintrin.h>
#include <stddef.h>
#include <stdint.h>

#include "xnnpack/common.h"
#include "xnnpack/unaligned.h"

// SIMD vector type for u32 using SSE41.
typedef __m128i xnn_simd_u32_t;
#define xnn_simd_size_u32 4
#define xnn_simd_log2_size_u32 2
#define xnn_simd_bytes_u32 (xnn_simd_size_u32 * sizeof(uint32_t))

#define XNN_SIMD_CONST_U32(var, val) \
  const xnn_simd_u32_t var = _mm_set1_epi32(val);

// Arithmetic operations.

static XNN_INLINE xnn_simd_u32_t xnn_mul_u32(xnn_simd_u32_t a,
                                             xnn_simd_u32_t b) {
  return _mm_mullo_epi32(a, b);
}

static XNN_INLINE xnn_simd_u32_t xnn_max_u32(xnn_simd_u32_t a,
                                             xnn_simd_u32_t b) {
  return _mm_max_epi32(a, b);
}

static XNN_INLINE xnn_simd_u32_t xnn_min_u32(xnn_simd_u32_t a,
                                             xnn_simd_u32_t b) {
  return _mm_min_epi32(a, b);
}

static XNN_INLINE xnn_simd_u32_t xnn_sub_u32(xnn_simd_u32_t a,
                                             xnn_simd_u32_t b) {
  return _mm_sub_epi32(a, b);
}

static XNN_INLINE __m128 xnn_cvt_f32_u32(xnn_simd_u32_t a);
static XNN_INLINE __m128 xnn_subw_f32_u32(xnn_simd_u32_t a,
                                          xnn_simd_u32_t b) {
  __m128i mask = _mm_cmpeq_epi32(a, _mm_max_epu32(a, b));
  __m128i result_32_variant1 = _mm_sub_epi32(a, b);
  __m128i result_32_variant2 = _mm_sub_epi32(b, a);
  __m128i result_32 = _mm_blendv_epi8(result_32_variant2, result_32_variant1,
                                      mask);
  __m128i sign = _mm_blendv_epi8(_mm_set1_epi32(INT32_C(-1)),
                                 _mm_set1_epi32(INT32_C(1)), mask);
  return _mm_mul_ps(xnn_cvt_f32_u32(result_32), _mm_cvtepi32_ps(sign));
}

// Load/store operations.

static XNN_INLINE xnn_simd_u32_t xnn_loadu_u32(const uint32_t* ptr) {
  return _mm_loadu_si128((const __m128i*)ptr);
}

static XNN_INLINE xnn_simd_u32_t xnn_load_u32(const uint32_t* ptr) {
  return _mm_load_si128((const __m128i*)ptr);
}

static XNN_INLINE void xnn_storeu_u32(uint32_t* ptr, xnn_simd_u32_t v) {
  _mm_storeu_si128((__m128i*)ptr, v);
}

static XNN_INLINE void xnn_store_u32(float* ptr, xnn_simd_u32_t v) {
  _mm_store_si128((__m128i*)ptr, v);
}

static XNN_INLINE xnn_simd_u32_t xnn_set1_u32(uint32_t v) {
  return _mm_set1_epi32(v);
}

static XNN_INLINE xnn_simd_u32_t xnn_set1_or_load_u32(const uint32_t* v) {
#if XNN_ARCH_X86
  return _mm_load_si128((const __m128i*)v);
#else
  return _mm_set1_epi32(*v);
#endif
}

// Tail load/store operations.

static XNN_INLINE xnn_simd_u32_t
xnn_load_tail_u32(const uint32_t* input, size_t num_elements) XNN_OOB_READS {
  assert(num_elements > 0);
  assert(num_elements < xnn_simd_size_u32);
  return _mm_loadu_si128((const __m128i*)input);
}

static XNN_INLINE void xnn_store_tail_u32(uint32_t* output, xnn_simd_u32_t v,
                                          size_t num_elements) {
  assert(num_elements > 0);
  assert(num_elements < xnn_simd_size_u32);

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

static XNN_INLINE __m128 xnn_cvt_f32_u32(xnn_simd_u32_t a) {
  const __m128 two16 = _mm_set1_ps(0x1.0p16f);  // Equivalent to 65536.0f
  __m128i hi = _mm_srli_epi32(a, 16);
  __m128i lo = _mm_srli_epi32(_mm_slli_epi32(a, 16), 16);
  __m128 fhi = _mm_mul_ps(_mm_cvtepi32_ps(hi), two16);
  __m128 flo = _mm_cvtepi32_ps(lo);

  return _mm_add_ps(fhi, flo);
}

#endif  // __XNNPACK_SRC_XNNPACK_SIMD_S32_SSE41_H_
