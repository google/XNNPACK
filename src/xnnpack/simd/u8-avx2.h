// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//

#ifndef XNNPACK_SRC_XNNPACK_SIMD_U8_AVX2_H_
#define XNNPACK_SRC_XNNPACK_SIMD_U8_AVX2_H_

#include <assert.h>
#include <immintrin.h>
#include <stddef.h>
#include <stdint.h>

#include "src/xnnpack/common.h"
#include "src/xnnpack/unaligned.h"

// SIMD vector type for u8 using AVX2.
typedef __m256i xnn_simd_u8_t;
#define xnn_simd_size_u8 32
#define xnn_simd_log2_size_u8 5
#define xnn_simd_bytes_u8 (xnn_simd_size_u8 * sizeof(uint8_t))

#define XNN_SIMD_CONST_U8(var, val) \
  const xnn_simd_u8_t var = _mm256_set1_epi8((char)(uint8_t)(val));

// Arithmetic operations.

static XNN_INLINE xnn_simd_u8_t xnn_add_u8(xnn_simd_u8_t a, xnn_simd_u8_t b) {
  return _mm256_add_epi8(a, b);
}

static XNN_INLINE xnn_simd_u8_t xnn_max_u8(xnn_simd_u8_t a, xnn_simd_u8_t b) {
  return _mm256_max_epu8(a, b);
}

static XNN_INLINE xnn_simd_u8_t xnn_min_u8(xnn_simd_u8_t a, xnn_simd_u8_t b) {
  return _mm256_min_epu8(a, b);
}

static XNN_INLINE uint8_t xnn_reduce_max_u8(xnn_simd_u8_t a) {
  const __m128i vmax16 =
      _mm_max_epu8(_mm256_castsi256_si128(a), _mm256_extracti128_si256(a, 1));
  __m128i vmax = _mm_max_epu8(vmax16, _mm_unpackhi_epi64(vmax16, vmax16));
  vmax = _mm_max_epu8(vmax, _mm_srli_epi64(vmax, 32));
  vmax = _mm_max_epu8(vmax, _mm_srli_epi32(vmax, 16));
  vmax = _mm_max_epu8(vmax, _mm_srli_epi16(vmax, 8));
  return (uint8_t)_mm_cvtsi128_si32(vmax);
}

static XNN_INLINE uint8_t xnn_reduce_min_u8(xnn_simd_u8_t a) {
  const __m128i vmin16 =
      _mm_min_epu8(_mm256_castsi256_si128(a), _mm256_extracti128_si256(a, 1));
  __m128i vmin = _mm_min_epu8(vmin16, _mm_unpackhi_epi64(vmin16, vmin16));
  vmin = _mm_min_epu8(vmin, _mm_srli_epi64(vmin, 32));
  vmin = _mm_min_epu8(vmin, _mm_srli_epi32(vmin, 16));
  vmin = _mm_min_epu8(vmin, _mm_srli_epi16(vmin, 8));
  return (uint8_t)_mm_cvtsi128_si32(vmin);
}

static XNN_INLINE xnn_simd_u8_t xnn_xor_u8(xnn_simd_u8_t a, xnn_simd_u8_t b) {
  return _mm256_xor_si256(a, b);
}

// Load/store operations.

static XNN_INLINE xnn_simd_u8_t xnn_loadu_u8(const uint8_t* ptr) {
  return _mm256_loadu_si256((const __m256i*)ptr);
}

static XNN_INLINE xnn_simd_u8_t xnn_load_u8(const uint8_t* ptr) {
  return _mm256_load_si256((const __m256i*)ptr);
}

static XNN_INLINE void xnn_storeu_u8(uint8_t* ptr, xnn_simd_u8_t v) {
  _mm256_storeu_si256((__m256i*)ptr, v);
}

static XNN_INLINE void xnn_store_u8(uint8_t* ptr, xnn_simd_u8_t v) {
  _mm256_store_si256((__m256i*)ptr, v);
}

static XNN_INLINE xnn_simd_u8_t xnn_set1_u8(uint8_t v) {
  return _mm256_set1_epi8((char)v);
}

// Tail load/store operations.

static XNN_INLINE xnn_simd_u8_t
xnn_load_tail_u8(const uint8_t* input, size_t num_elements) XNN_OOB_READS {
  assert(num_elements > 0);
  assert(num_elements < xnn_simd_size_u8);
  const __m128i low = _mm_loadu_si128((const __m128i*)input);
  const __m128i high =
      num_elements > 16 ? _mm_loadu_si128((const __m128i*)(input + 16)) : low;
  return _mm256_inserti128_si256(_mm256_castsi128_si256(low), high, 1);
}

static XNN_INLINE xnn_simd_u8_t xnn_load_tail_safe_u8(const uint8_t* input,
                                                      size_t num_elements) {
  assert(num_elements > 0);
  assert(num_elements < xnn_simd_size_u8);

  XNN_ALIGN(32) uint8_t padded[32];
  uint8_t* d = &padded[0];
  switch (num_elements) {
    case 31:
      *d++ = *input++;
      XNN_FALLTHROUGH
    case 30:
      *d++ = *input++;
      XNN_FALLTHROUGH
    case 29:
      *d++ = *input++;
      XNN_FALLTHROUGH
    case 28:
      *d++ = *input++;
      XNN_FALLTHROUGH
    case 27:
      *d++ = *input++;
      XNN_FALLTHROUGH
    case 26:
      *d++ = *input++;
      XNN_FALLTHROUGH
    case 25:
      *d++ = *input++;
      XNN_FALLTHROUGH
    case 24:
      *d++ = *input++;
      XNN_FALLTHROUGH
    case 23:
      *d++ = *input++;
      XNN_FALLTHROUGH
    case 22:
      *d++ = *input++;
      XNN_FALLTHROUGH
    case 21:
      *d++ = *input++;
      XNN_FALLTHROUGH
    case 20:
      *d++ = *input++;
      XNN_FALLTHROUGH
    case 19:
      *d++ = *input++;
      XNN_FALLTHROUGH
    case 18:
      *d++ = *input++;
      XNN_FALLTHROUGH
    case 17:
      *d++ = *input++;
      XNN_FALLTHROUGH
    case 16:
      *d++ = *input++;
      XNN_FALLTHROUGH
    case 15:
      *d++ = *input++;
      XNN_FALLTHROUGH
    case 14:
      *d++ = *input++;
      XNN_FALLTHROUGH
    case 13:
      *d++ = *input++;
      XNN_FALLTHROUGH
    case 12:
      *d++ = *input++;
      XNN_FALLTHROUGH
    case 11:
      *d++ = *input++;
      XNN_FALLTHROUGH
    case 10:
      *d++ = *input++;
      XNN_FALLTHROUGH
    case 9:
      *d++ = *input++;
      XNN_FALLTHROUGH
    case 8:
      *d++ = *input++;
      XNN_FALLTHROUGH
    case 7:
      *d++ = *input++;
      XNN_FALLTHROUGH
    case 6:
      *d++ = *input++;
      XNN_FALLTHROUGH
    case 5:
      *d++ = *input++;
      XNN_FALLTHROUGH
    case 4:
      *d++ = *input++;
      XNN_FALLTHROUGH
    case 3:
      *d++ = *input++;
      XNN_FALLTHROUGH
    case 2:
      *d++ = *input++;
      XNN_FALLTHROUGH
    case 1:
      *d++ = *input++;
      XNN_FALLTHROUGH
    default:
      break;
  }
  return _mm256_load_si256((const __m256i*)&padded[0]);
}

static XNN_INLINE void xnn_store_tail_u8(uint8_t* output, xnn_simd_u8_t v,
                                         size_t num_elements) {
  assert(num_elements > 0);
  assert(num_elements < xnn_simd_size_u8);

  __m128i v_low = _mm256_castsi256_si128(v);
  if (num_elements & (16 * sizeof(uint8_t))) {
    _mm_storeu_si128((__m128i*)output, v_low);
    v_low = _mm256_extracti128_si256(v, 1);
    output += 16;
  }
  if (num_elements & (8 * sizeof(uint8_t))) {
    _mm_storel_epi64((__m128i*)output, v_low);
    v_low = _mm_unpackhi_epi64(v_low, v_low);
    output += 8;
  }
  if (num_elements & (4 * sizeof(uint8_t))) {
    unaligned_store_u32(output, (uint32_t)_mm_cvtsi128_si32(v_low));
    v_low = _mm_srli_epi64(v_low, 32);
    output += 4;
  }
  if (num_elements & (2 * sizeof(uint8_t))) {
    unaligned_store_u16(output, (uint16_t)_mm_extract_epi16(v_low, 0));
    v_low = _mm_srli_epi32(v_low, 16);
    output += 2;
  }
  if (num_elements & (1 * sizeof(uint8_t))) {
    *output = (uint8_t)_mm_cvtsi128_si32(v_low);
  }
}

#endif  // XNNPACK_SRC_XNNPACK_SIMD_U8_AVX2_H_
