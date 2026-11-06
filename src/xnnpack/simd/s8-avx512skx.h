// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//

#ifndef XNNPACK_SRC_XNNPACK_SIMD_S8_AVX512SKX_H_
#define XNNPACK_SRC_XNNPACK_SIMD_S8_AVX512SKX_H_

#include <assert.h>
#include <immintrin.h>
#include <stddef.h>
#include <stdint.h>

#include "src/xnnpack/common.h"
#include "src/xnnpack/intrinsics-polyfill.h"

// SIMD vector type for s8 using AVX512SKX.
typedef __m512i xnn_simd_s8_t;
#define xnn_simd_size_s8 64
#define xnn_simd_log2_size_s8 6
#define xnn_simd_bytes_s8 (xnn_simd_size_s8 * sizeof(int8_t))

#define XNN_SIMD_CONST_S8(var, val) \
  const xnn_simd_s8_t var = _mm512_set1_epi8(val);

// Arithmetic operations.

static XNN_INLINE xnn_simd_s8_t xnn_add_s8(xnn_simd_s8_t a, xnn_simd_s8_t b) {
  return _mm512_add_epi8(a, b);
}

static XNN_INLINE xnn_simd_s8_t xnn_max_s8(xnn_simd_s8_t a, xnn_simd_s8_t b) {
  return _mm512_max_epi8(a, b);
}

static XNN_INLINE xnn_simd_s8_t xnn_min_s8(xnn_simd_s8_t a, xnn_simd_s8_t b) {
  return _mm512_min_epi8(a, b);
}

static XNN_INLINE int8_t xnn_reduce_max_s8(xnn_simd_s8_t a) {
  const __m256i vmax32 = _mm256_max_epi8(_mm512_castsi512_si256(a),
                                         _mm512_extracti64x4_epi64(a, 1));
  const __m128i vmax16 = _mm_max_epi8(_mm256_castsi256_si128(vmax32),
                                      _mm256_extracti128_si256(vmax32, 1));
  __m128i vmax = _mm_max_epi8(vmax16, _mm_unpackhi_epi64(vmax16, vmax16));
  vmax = _mm_max_epi8(vmax, _mm_srli_epi64(vmax, 32));
  vmax = _mm_max_epi8(vmax, _mm_srli_epi32(vmax, 16));
  vmax = _mm_max_epi8(vmax, _mm_srli_epi16(vmax, 8));
  return (int8_t)_mm_cvtsi128_si32(vmax);
}

static XNN_INLINE int8_t xnn_reduce_min_s8(xnn_simd_s8_t a) {
  const __m256i vmin32 = _mm256_min_epi8(_mm512_castsi512_si256(a),
                                         _mm512_extracti64x4_epi64(a, 1));
  const __m128i vmin16 = _mm_min_epi8(_mm256_castsi256_si128(vmin32),
                                      _mm256_extracti128_si256(vmin32, 1));
  __m128i vmin = _mm_min_epi8(vmin16, _mm_unpackhi_epi64(vmin16, vmin16));
  vmin = _mm_min_epi8(vmin, _mm_srli_epi64(vmin, 32));
  vmin = _mm_min_epi8(vmin, _mm_srli_epi32(vmin, 16));
  vmin = _mm_min_epi8(vmin, _mm_srli_epi16(vmin, 8));
  return (int8_t)_mm_cvtsi128_si32(vmin);
}

static XNN_INLINE xnn_simd_s8_t xnn_xor_s8(xnn_simd_s8_t a, xnn_simd_s8_t b) {
  return _mm512_xor_si512(a, b);
}

// Load/store operations.

static XNN_INLINE xnn_simd_s8_t xnn_loadu_s8(const int8_t* ptr) {
  return _mm512_loadu_si512((const __m512i*)ptr);
}

static XNN_INLINE xnn_simd_s8_t xnn_load_s8(const int8_t* ptr) {
  return _mm512_load_si512((const __m512i*)ptr);
}

static XNN_INLINE void xnn_storeu_s8(int8_t* ptr, xnn_simd_s8_t v) {
  _mm512_storeu_si512((__m512i*)ptr, v);
}

static XNN_INLINE void xnn_store_s8(int8_t* ptr, xnn_simd_s8_t v) {
  _mm512_store_si512((__m512i*)ptr, v);
}

static XNN_INLINE xnn_simd_s8_t xnn_set1_s8(int8_t v) {
  return _mm512_set1_epi8(v);
}

// Tail load/store operations.

static XNN_INLINE xnn_simd_s8_t xnn_load_tail_s8(const int8_t* input,
                                                 size_t num_elements) {
  assert(num_elements > 0);
  assert(num_elements < xnn_simd_size_s8);
  const __mmask64 vmask =
      _cvtu64_mask64((uint64_t)((UINT64_C(1) << num_elements) - UINT64_C(1)));
  return _mm512_maskz_loadu_epi8(vmask, input);
}

static XNN_INLINE xnn_simd_s8_t xnn_load_tail_safe_s8(const int8_t* input,
                                                      size_t num_elements) {
  return xnn_load_tail_s8(input, num_elements);
}

static XNN_INLINE void xnn_store_tail_s8(int8_t* output, xnn_simd_s8_t v,
                                         size_t num_elements) {
  assert(num_elements > 0);
  assert(num_elements < xnn_simd_size_s8);

  const __mmask64 vmask =
      _cvtu64_mask64((uint64_t)((UINT64_C(1) << num_elements) - UINT64_C(1)));
  _mm512_mask_storeu_epi8(output, vmask, v);
}

#endif  // XNNPACK_SRC_XNNPACK_SIMD_S8_AVX512SKX_H_
