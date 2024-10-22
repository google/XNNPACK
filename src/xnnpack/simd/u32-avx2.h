// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//

#ifndef __XNNPACK_SRC_XNNPACK_SIMD_U32_AVX2_H_
#define __XNNPACK_SRC_XNNPACK_SIMD_U32_AVX2_H_

#include <assert.h>
#include <immintrin.h>
#include <stddef.h>
#include <stdint.h>

#include "xnnpack/common.h"
#include "xnnpack/unaligned.h"

// SIMD vector type for u32 using AVX2.
typedef __m256i xnn_simd_u32_t;
#define xnn_simd_size_u32 8
#define xnn_simd_log2_size_u32 3
#define xnn_simd_bytes_u32 (xnn_simd_size_u32 * sizeof(uint32_t))

#define XNN_SIMD_CONST_u32(var, val) \
  const xnn_simd_u32_t var = _mm256_set1_epi32(val);

// Mask table used for masked load/store operations.
static const int32_t mask_table_avx_u32[14] = {-1, -1, -1, -1, -1, -1, -1,
                                               0,  0,  0,  0,  0,  0,  0};
// Arithmetic operations.
static XNN_INLINE xnn_simd_u32_t xnn_mul_u32(xnn_simd_u32_t a,
                                             xnn_simd_u32_t b) {
  return _mm256_mullo_epi32(a, b);
}

static XNN_INLINE xnn_simd_u32_t xnn_max_u32(xnn_simd_u32_t a,
                                             xnn_simd_u32_t b) {
  return _mm256_max_epu32(a, b);
}

static XNN_INLINE xnn_simd_u32_t xnn_min_u32(xnn_simd_u32_t a,
                                             xnn_simd_u32_t b) {
  return _mm256_min_epu32(a, b);
}

static XNN_INLINE xnn_simd_u32_t xnn_sub_u32(xnn_simd_u32_t a,
                                             xnn_simd_u32_t b) {
  return _mm256_sub_epi32(a, b);
}

static XNN_INLINE __m256 xnn_cvt_f32_u32(xnn_simd_u32_t a);

static XNN_INLINE __m256 xnn_subw_f32_u32(xnn_simd_u32_t a,
                                          xnn_simd_u32_t b) {
  __m256i mask = _mm256_cmpeq_epi32(a, _mm256_max_epu32(a, b));
  __m256i result_32_variant1 = _mm256_sub_epi32(a, b);
  __m256i result_32_variant2 = _mm256_sub_epi32(b, a);
  __m256i result_32 = _mm256_blendv_epi8(result_32_variant2, result_32_variant1,
                                         mask);
  __m256i sign = _mm256_blendv_epi8(_mm256_set1_epi32(INT32_C(-1)),
                                    _mm256_set1_epi32(INT32_C(1)), mask);
  return _mm256_mul_ps(xnn_cvt_f32_u32(result_32), _mm256_cvtepi32_ps(sign));
}

// Load/store operations.

static XNN_INLINE xnn_simd_u32_t xnn_loadu_u32(const uint32_t* ptr) {
  return _mm256_loadu_si256((const __m256i*)ptr);
}

static XNN_INLINE xnn_simd_u32_t xnn_load_u32(const uint32_t* ptr) {
  return _mm256_load_si256((const __m256i*)ptr);
}

static XNN_INLINE void xnn_storeu_u32(uint32_t* ptr, xnn_simd_u32_t v) {
  _mm256_storeu_si256((__m256i*)ptr, v);
}

static XNN_INLINE void xnn_store_u32(uint32_t* ptr, xnn_simd_u32_t v) {
  _mm256_store_si256((__m256i*)ptr, v);
}

static XNN_INLINE xnn_simd_u32_t xnn_set1_u32(uint32_t v) {
  return _mm256_set1_epi32(v);
}

static XNN_INLINE xnn_simd_u32_t xnn_set1_or_load_u32(const uint32_t* v) {
#if XNN_ARCH_X86
  return _mm256_load_si256((const __m256i*)v);
#else
  return _mm256_set1_epi32(*v);
#endif
}

// Tail load/store operations.

static XNN_INLINE xnn_simd_u32_t
xnn_load_tail_u32(const uint32_t* input, size_t num_elements) {
  assert(num_elements > 0);
  assert(num_elements < xnn_simd_size_u32);

  const __m256i vmask = _mm256_loadu_si256(
      (const __m256i*)(&mask_table_avx_u32[7] - num_elements));
  return _mm256_maskload_epi32((const int32_t*) input, vmask);
}

static XNN_INLINE void xnn_store_tail_u32(uint32_t* output, xnn_simd_u32_t v,
                                          size_t num_elements) {
  assert(num_elements > 0);
  assert(num_elements < xnn_simd_size_u32);

  __m128i v_lo = _mm256_castsi256_si128(v);
  if (num_elements & 4) {
    _mm_storeu_si128((__m128i*)output, v_lo);
    v_lo = _mm256_extractf128_si256(v, 1);
    output += 4;
  }
  if (num_elements & 2) {
    _mm_storel_epi64((__m128i*)output, v_lo);
    v_lo = _mm_unpackhi_epi64(v_lo, v_lo);
    output += 2;
  }
  if (num_elements & 1) {
    unaligned_store_u32(output, (uint32_t)_mm_cvtsi128_si32(v_lo));
  }
}

// Conversion operations.

static XNN_INLINE __m256 xnn_cvt_f32_u32(xnn_simd_u32_t a) {
  const __m256 two16 = _mm256_set1_ps(0x1.0p16f);  // Equivalent to 65536.0f
  __m256i hi = _mm256_srli_epi32(a, 16);
  __m256i lo = _mm256_and_si256(a, _mm256_set1_epi32(0xFFFF));
  __m256 fhi = _mm256_mul_ps(_mm256_cvtepi32_ps(hi), two16);
  __m256 flo = _mm256_cvtepi32_ps(lo);

  return _mm256_add_ps(fhi, flo);
}

#endif  // __XNNPACK_SRC_XNNPACK_SIMD_U32_AVX2_H_
