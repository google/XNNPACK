// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//

#ifndef __XNNPACK_SRC_XNNPACK_SIMD_S16_AVX512SKX_H_
#define __XNNPACK_SRC_XNNPACK_SIMD_S16_AVX512SKX_H_

#include <assert.h>
#include <immintrin.h>
#include <stddef.h>
#include <stdint.h>

#include "src/xnnpack/common.h"

// SIMD vector type for s16 using AVX512SKX.
typedef __m512i xnn_simd_s16_t;
#define xnn_simd_size_s16 32
#define xnn_simd_log2_size_s16 5
#define xnn_simd_bytes_s16 (xnn_simd_size_s16 * sizeof(int16))

#define XNN_SIMD_CONST_S16(var, val) \
  const xnn_simd_s16_t var = _mm512_set1_epi16(val);

// Arithmetic operations.

static XNN_INLINE xnn_simd_s16_t xnn_min_s16(xnn_simd_s16_t a,
                                             xnn_simd_s16_t b) {
  return _mm512_min_epi16(a, b);
}

static XNN_INLINE xnn_simd_s16_t xnn_max_s16(xnn_simd_s16_t a,
                                             xnn_simd_s16_t b) {
  return _mm512_max_epi16(a, b);
}

static XNN_INLINE xnn_simd_s16_t xnn_signcomplement_s16(xnn_simd_s16_t x) {
  XNN_SIMD_CONST_S16(nonsign_mask, 0x7FFF);
  return _mm512_xor_si512(_mm512_and_si512(x, nonsign_mask),
                          _mm512_srai_epi16(x, 15));
}

// Load/store operations.

static XNN_INLINE xnn_simd_s16_t xnn_loadu_s16(const int16_t* ptr) {
  return _mm512_loadu_si512(ptr);
}

static XNN_INLINE xnn_simd_s16_t xnn_load_s16(const int16_t* ptr) {
  return _mm512_load_si512(ptr);
}

static XNN_INLINE void xnn_storeu_s16(int16_t* ptr, xnn_simd_s16_t v) {
  _mm512_storeu_si512(ptr, v);
}

static XNN_INLINE void xnn_store_s16(int16_t* ptr, xnn_simd_s16_t v) {
  _mm512_storeu_si512(ptr, v);
}

static XNN_INLINE xnn_simd_s16_t xnn_set1_s16_t(int16_t v) {
  return _mm512_set1_epi16(v);
}

// Tail load/store operations.

static XNN_INLINE xnn_simd_s16_t xnn_load_tail_s16(const int16_t* input,
                                                   size_t num_elements) {
  assert(num_elements > 0);
  assert(num_elements < xnn_simd_size_s16);
  const __mmask32 vmask =
      _cvtu32_mask32((uint32_t)((UINT32_C(1) << num_elements) - UINT32_C(1)));
  return _mm512_maskz_loadu_epi16(vmask, input);
}

static XNN_INLINE xnn_simd_s16_t xnn_load_tail_safe_s16(const int16_t* input,
                                                        size_t num_elements) {
  return xnn_load_tail_s16(input, num_elements);
}

static XNN_INLINE void xnn_store_tail_s16(int16_t* output, xnn_simd_s16_t v,
                                          size_t num_elements) {
  assert(num_elements > 0);
  assert(num_elements < xnn_simd_size_s16);

  const __mmask32 vmask =
      _cvtu32_mask32((uint32_t)((UINT32_C(1) << num_elements) - UINT32_C(1)));
  _mm512_mask_storeu_epi16(output, vmask, v);
}

#endif  // __XNNPACK_SRC_XNNPACK_SIMD_S16_AVX512SKX_H_
