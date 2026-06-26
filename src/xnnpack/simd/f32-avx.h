// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//

#ifndef XNNPACK_SRC_XNNPACK_SIMD_F32_AVX_H_
#define XNNPACK_SRC_XNNPACK_SIMD_F32_AVX_H_

#include "src/xnnpack/common.h"
#include "src/xnnpack/simd/f32-avx-base.h"  // IWYU pragma: export

// Whether or not this architecture has native fused multiply-add support.
#define XNN_SIMD_HAS_NATIVE_FMA 0

static XNN_INLINE xnn_simd_f32_t xnn_fmadd_f32(xnn_simd_f32_t a,
                                               xnn_simd_f32_t b,
                                               xnn_simd_f32_t c) {
  return _mm256_add_ps(_mm256_mul_ps(a, b), c);
}

static XNN_INLINE xnn_simd_f32_t xnn_fnmadd_f32(xnn_simd_f32_t a,
                                                xnn_simd_f32_t b,
                                                xnn_simd_f32_t c) {
  return _mm256_sub_ps(c, _mm256_mul_ps(a, b));
}

static XNN_INLINE xnn_simd_f32_t xnn_fmsub_f32(xnn_simd_f32_t a,
                                               xnn_simd_f32_t b,
                                               xnn_simd_f32_t c) {
  return _mm256_sub_ps(_mm256_mul_ps(a, b), c);
}

// This intrinsic is not particularly efficient. If you really need this
// intrinsic, consider using `avx2` instead.
static XNN_INLINE xnn_simd_f32_t xnn_sll_f32(xnn_simd_f32_t a, uint8_t bits) {
  __m128i a_lo = _mm256_castsi256_si128(_mm256_castps_si256(a));
  __m128i a_hi = _mm256_extractf128_si256(_mm256_castps_si256(a), 1);
  __m128i res_lo = _mm_slli_epi32(a_lo, bits);
  __m128i res_hi = _mm_slli_epi32(a_hi, bits);
  return _mm256_castsi256_ps(
      _mm256_insertf128_si256(_mm256_castsi128_si256(res_lo), res_hi, 1));
}

// This intrinsic is not particularly efficient. If you really need this
// intrinsic, consider using `avx2` instead.
static XNN_INLINE xnn_simd_f32_t xnn_srl_f32(xnn_simd_f32_t a, uint8_t bits) {
  __m128i a_lo = _mm256_castsi256_si128(_mm256_castps_si256(a));
  __m128i a_hi = _mm256_extractf128_si256(_mm256_castps_si256(a), 1);
  __m128i res_lo = _mm_srli_epi32(a_lo, bits);
  __m128i res_hi = _mm_srli_epi32(a_hi, bits);
  return _mm256_castsi256_ps(
      _mm256_insertf128_si256(_mm256_castsi128_si256(res_lo), res_hi, 1));
}

// This intrinsic is not particularly efficient. If you really need this
// intrinsic, consider using `avx2` instead.
static XNN_INLINE xnn_simd_f32_t xnn_sra_f32(xnn_simd_f32_t a, uint8_t bits) {
  __m128i a_lo = _mm256_castsi256_si128(_mm256_castps_si256(a));
  __m128i a_hi = _mm256_extractf128_si256(_mm256_castps_si256(a), 1);
  __m128i res_lo = _mm_srai_epi32(a_lo, bits);
  __m128i res_hi = _mm_srai_epi32(a_hi, bits);
  return _mm256_castsi256_ps(
      _mm256_insertf128_si256(_mm256_castsi128_si256(res_lo), res_hi, 1));
}

static XNN_INLINE xnn_simd_f32_t xnn_cmpeq_f32(xnn_simd_f32_t a,
                                               xnn_simd_f32_t b) {
  return _mm256_cmp_ps(a, b, _CMP_EQ_OQ);
}

static XNN_INLINE xnn_simd_f32_t xnn_cmpneq_f32(xnn_simd_f32_t a,
  xnn_simd_f32_t b) {
return _mm256_cmp_ps(a, b, _CMP_NEQ_OQ);
}

#if (XNN_ENABLE_F16C && \
     (!defined(__clang__) || defined(__F16C__))) || \
    defined(__F16C__)
#include "src/xnnpack/math.h"

typedef __m128i xnn_simd_f16_t;

static XNN_INLINE xnn_simd_f16_t xnn_loadu_f16(const xnn_float16* ptr) {
  return _mm_loadu_si128((const __m128i*)ptr);
}

static XNN_INLINE void xnn_storeu_f16(xnn_float16* ptr, xnn_simd_f16_t v) {
  _mm_storeu_si128((__m128i*)ptr, v);
}

static XNN_INLINE xnn_simd_f32_t xnn_cvt_f32_f16(xnn_simd_f16_t a) {
  return _mm256_cvtph_ps(a);
}

static XNN_INLINE xnn_simd_f16_t xnn_cvt_f16_f32(xnn_simd_f32_t a) {
  return _mm256_cvtps_ph(a, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
}

static XNN_INLINE xnn_simd_f16_t xnn_load_tail_f16(const xnn_float16* input,
                                                   size_t num_elements) {
  XNN_ALIGN(16) xnn_float16 padded[8] = {0};
  memcpy(padded, input, num_elements * sizeof(xnn_float16));
  return _mm_loadu_si128((const __m128i*)padded);
}

static XNN_INLINE void xnn_store_tail_f16(xnn_float16* output, xnn_simd_f16_t v,
                                          size_t num_elements) {
  if (num_elements == 8) {
    _mm_storeu_si128((__m128i*)output, v);
  } else {
    XNN_ALIGN(16) xnn_float16 padded[8];
    _mm_storeu_si128((__m128i*)padded, v);
    memcpy(output, padded, num_elements * sizeof(xnn_float16));
  }
}
#endif  // XNN_ENABLE_F16C || defined(__F16C__)

#endif  // XNNPACK_SRC_XNNPACK_SIMD_F32_AVX_H_
