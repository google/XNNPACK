// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//

#ifndef __XNNPACK_SRC_XNNPACK_SIMD_S8_AVX2_H_
#define __XNNPACK_SRC_XNNPACK_SIMD_S8_AVX2_H_

#include <assert.h>
#include <immintrin.h>
#include <stddef.h>
#include <stdint.h>

#include "xnnpack/common.h"
#include "xnnpack/unaligned.h"

// SIMD vector type for s8 using AVX2.
typedef __m256i xnn_simd_s8_t;
#define xnn_simd_size_s8 32
#define xnn_simd_log2_size_s8 5
#define xnn_simd_bytes_s8 (xnn_simd_size_s8 * sizeof(int8_t))

#define XNN_SIMD_CONST_S8(var, val) \
  const xnn_simd_s8_t var = _mm256_set1_epi8(val);

// Mask table used for masked load/store operations.
static const int32_t mask_table_avx_s8[64] = {
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1,
    0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,
    0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0,  0};

// Arithmetic operations.

static XNN_INLINE xnn_simd_s8_t xnn_add_s8(xnn_simd_s8_t a,
                                             xnn_simd_s8_t b) {
  return _mm256_add_epi8(a, b);
}

static XNN_INLINE xnn_simd_s8_t xnn_max_s8(xnn_simd_s8_t a,
                                             xnn_simd_s8_t b) {
  return _mm256_max_epi8(a, b);
}

static XNN_INLINE xnn_simd_s8_t xnn_min_s8(xnn_simd_s8_t a,
                                             xnn_simd_s8_t b) {
  return _mm256_min_epi8(a, b);
}

static XNN_INLINE __m256i xnn_low_cvt_s8_s16(xnn_simd_s8_t a) {
  return _mm256_cvtepi8_epi16(_mm256_castsi256_si128(a));
}

static XNN_INLINE __m256i xnn_high_cvt_s8_s16(xnn_simd_s8_t a) {
  return _mm256_cvtepi8_epi16(_mm256_extracti128_si256(a, 1));
}

// Load/store operations.

static XNN_INLINE xnn_simd_s8_t xnn_loadu_s8(const int8_t* ptr) {
  return _mm256_loadu_si256((const __m256i*)ptr);
}

static XNN_INLINE xnn_simd_s8_t xnn_load_s8(const int8_t* ptr) {
  return _mm256_load_si256((const __m256i*)ptr);
}

static XNN_INLINE void xnn_storeu_s8(int8_t* ptr, xnn_simd_s8_t v) {
  _mm256_storeu_si256((__m256i*)ptr, v);
}

static XNN_INLINE void xnn_store_s8(int8_t* ptr, xnn_simd_s8_t v) {
  _mm256_store_si256((__m256i*)ptr, v);
}

static XNN_INLINE xnn_simd_s8_t xnn_set1_s8(int8_t v) {
  return _mm256_set1_epi8(v);
}

static XNN_INLINE xnn_simd_s8_t xnn_set1_or_load_s8(const int8_t* v) {
#if XNN_ARCH_X86
  return _mm256_load_si256((const __m256i*)v);
#else
  return _mm256_set1_epi8(*v);
#endif
}

// Tail load/store operations.

static XNN_INLINE xnn_simd_s8_t
xnn_load_tail_s8(const int8_t* input, size_t num_elements) XNN_OOB_READS {
  assert(num_elements > 0);
  assert(num_elements < xnn_simd_size_s8);
  const __m256i vmask = _mm256_loadu_si256(
      (const __m256i*) (&mask_table_avx_s8[32 - num_elements]));
  return _mm256_maskload_epi32((const int32_t*) input, vmask);
}

static XNN_INLINE void xnn_store_tail_s8(int8_t* output, xnn_simd_s8_t v,
                                          size_t num_elements) {
  assert(num_elements > 0);
  assert(num_elements < xnn_simd_size_s8);

  __m128i v_lo = _mm256_castsi256_si128(v);
  if (num_elements & 16) {
    _mm_storeu_si128((__m128i*)output, v_lo);
    v_lo = _mm256_extractf128_si256(v, 1);
    output += 16;
  }
  if (num_elements & 8) {
    _mm_storel_epi64((__m128i*)output, v_lo);
    v_lo = _mm_unpackhi_epi64(v_lo, v_lo);
    output += 8;
  }
  if (num_elements & 4) {
    unaligned_store_u32(output, (uint32_t)_mm_cvtsi128_si32(v_lo));
    v_lo = _mm_srli_epi64(v_lo, 32);
    output += 4;
  }
  if (num_elements & 2) {
    unaligned_store_u16(output, (uint16_t)_mm_extract_epi16(v_lo, 0));
    v_lo = _mm_srli_epi32(v_lo, 16);
    output += 2;
  }
  if (num_elements & 1){
    unaligned_store_u8(output, (uint8_t)_mm_extract_epi8(v_lo, 0));
  }
}

#endif  // __XNNPACK_SRC_XNNPACK_SIMD_S8_AVX2_H_
