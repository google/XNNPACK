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

// Load/store operations.

static XNN_INLINE xnn_simd_s32_t xnn_loadu_s32(const int32_t* ptr) {
  return _mm512_loadu_epi32(ptr);
}

static XNN_INLINE xnn_simd_s32_t xnn_load_s32(const int32_t* ptr) {
  return _mm512_load_epi32(ptr);
}

static XNN_INLINE void xnn_storeu_s32(int32_t* ptr, xnn_simd_s32_t v) {
  _mm512_storeu_epi32(ptr, v);
}

static XNN_INLINE void xnn_store_s32(float* ptr, xnn_simd_s32_t v) {
  _mm512_store_epi32(ptr, v);
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
