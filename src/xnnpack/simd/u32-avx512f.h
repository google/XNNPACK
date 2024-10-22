// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//

#ifndef __XNNPACK_SRC_XNNPACK_SIMD_U32_AVX512F_H_
#define __XNNPACK_SRC_XNNPACK_SIMD_U32_AVX512F_H_

#include <assert.h>
#include <immintrin.h>
#include <stddef.h>
#include <stdint.h>

#include "xnnpack/common.h"
#include "xnnpack/intrinsics-polyfill.h"
#include "xnnpack/unaligned.h"

// SIMD vector type for u32 using SSE41.
typedef __m512i xnn_simd_u32_t;
#define xnn_simd_size_u32 16
#define xnn_simd_log2_size_u32 4
#define xnn_simd_bytes_u32 (xnn_simd_size_u32 * sizeof(uint32_t))

#define XNN_SIMD_CONST_U32(var, val) \
  const xnn_simd_u32_t var = _mm512_set1_epi32(val);

// Arithmetic operations.

static XNN_INLINE xnn_simd_u32_t xnn_mul_u32(xnn_simd_u32_t a,
                                             xnn_simd_u32_t b) {
  return _mm512_mullo_epi32(a, b);
}

static XNN_INLINE xnn_simd_u32_t xnn_max_u32(xnn_simd_u32_t a,
                                             xnn_simd_u32_t b) {
  return _mm512_max_epi32(a, b);
}

static XNN_INLINE xnn_simd_u32_t xnn_min_u32(xnn_simd_u32_t a,
                                             xnn_simd_u32_t b) {
  return _mm512_min_epi32(a, b);
}

static XNN_INLINE xnn_simd_u32_t xnn_sub_u32(xnn_simd_u32_t a,
                                             xnn_simd_u32_t b) {
  return _mm512_sub_epi32(a, b);
}

static XNN_INLINE __m512 xnn_cvt_f32_u32(xnn_simd_u32_t a);
static XNN_INLINE __m512 xnn_subw_f32_u32(xnn_simd_u32_t a,
                                          xnn_simd_u32_t b) {
  __mmask16 mask = _mm512_cmpeq_epi32_mask(a, _mm512_max_epu32(a, b));
  __m512i result_32_variant1 = _mm512_sub_epi32(a, b);
  __m512i result_32_variant2 = _mm512_sub_epi32(b, a);
  __m512i result_32 = _mm512_mask_blend_epi32(mask, result_32_variant2,
                                              result_32_variant1);
  __m512i sign = _mm512_mask_blend_epi32(mask, _mm512_set1_epi32(INT32_C(-1)),
                                         _mm512_set1_epi32(INT32_C(1)));
  return _mm512_mul_ps(xnn_cvt_f32_u32(result_32), _mm512_cvtepi32_ps(sign));
}

// Load/store operations.

static XNN_INLINE xnn_simd_u32_t xnn_loadu_u32(const uint32_t* ptr) {
  return _mm512_loadu_epi32(ptr);
}

static XNN_INLINE xnn_simd_u32_t xnn_load_u32(const uint32_t* ptr) {
  return _mm512_load_epi32(ptr);
}

static XNN_INLINE void xnn_storeu_u32(uint32_t* ptr, xnn_simd_u32_t v) {
  _mm512_storeu_epi32(ptr, v);
}

static XNN_INLINE void xnn_store_u32(float* ptr, xnn_simd_u32_t v) {
  _mm512_store_epi32(ptr, v);
}

static XNN_INLINE xnn_simd_u32_t xnn_set1_u32(uint32_t v) {
  return _mm512_set1_epi32(v);
}

static XNN_INLINE xnn_simd_u32_t xnn_set1_or_load_u32(const uint32_t* v) {
#if XNN_ARCH_X86
  return _mm512_load_epi32((const __m128i*)v);
#else
  return _mm512_set1_epi32(*v);
#endif
}

// Tail load/store operations.

static XNN_INLINE xnn_simd_u32_t
xnn_load_tail_u32(const uint32_t* input, size_t num_elements) XNN_OOB_READS {
  assert(num_elements > 0);
  assert(num_elements < xnn_simd_size_u32);
  const __mmask16 vmask = _cvtu32_mask16((uint32_t) ((UINT32_C(1) << num_elements) - UINT32_C(1)));
  return _mm512_maskz_loadu_epi32(vmask, input);
}

static XNN_INLINE void xnn_store_tail_u32(uint32_t* output, xnn_simd_u32_t v,
                                          size_t num_elements) {
  assert(num_elements > 0);
  assert(num_elements < xnn_simd_size_u32);

  const __mmask16 vmask =
      _cvtu32_mask16((uint32_t)((UINT32_C(1) << num_elements) - UINT32_C(1)));
  _mm512_mask_storeu_epi32(output, vmask, v);
}

// Conversion operations.

static XNN_INLINE __m512 xnn_cvt_f32_u32(xnn_simd_u32_t a) {
  const __m512 two16 = _mm512_set1_ps(0x1.0p16f);  // Equivalent to 65536.0f
  __m512i hi = _mm512_srli_epi32(a, 16);
  __m512i lo = _mm512_srli_epi32(_mm512_slli_epi32(a, 16), 16);
  __m512 fhi = _mm512_mul_ps(_mm512_cvtepi32_ps(hi), two16);
  __m512 flo = _mm512_cvtepi32_ps(lo);

  return _mm512_add_ps(fhi, flo);
}

#endif  // __XNNPACK_SRC_XNNPACK_SIMD_U32_AVX512F_H_
