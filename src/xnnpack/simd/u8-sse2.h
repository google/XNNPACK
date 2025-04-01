// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//

#ifndef __XNNPACK_SRC_XNNPACK_SIMD_U8_SSE2_H_
#define __XNNPACK_SRC_XNNPACK_SIMD_U8_SSE2_H_

#include <assert.h>
#include <emmintrin.h>
#include <stddef.h>
#include <stdint.h>

#include "src/xnnpack/common.h"
#include "src/xnnpack/unaligned.h"

// SIMD vector type for u8 using SSE41.
typedef __m128i xnn_simd_u8_t;
#define xnn_simd_size_u8 16
#define xnn_simd_log2_size_u8 0
#define xnn_simd_bytes_u8 (xnn_simd_size_u8 * sizeof(uint8_t))

#define XNN_SIMD_CONST_U8(var, val) \
  const xnn_simd_u8_t var = _mm_set1_epi8(val);

// Arithmetic operations.

static XNN_INLINE xnn_simd_u8_t xnn_add_u8(xnn_simd_u8_t a, xnn_simd_u8_t b) {
  return _mm_add_epi8(a, b);
}

static XNN_INLINE xnn_simd_u8_t xnn_max_u8(xnn_simd_u8_t a, xnn_simd_u8_t b) {
  return _mm_max_epu8(a, b);
}

static XNN_INLINE xnn_simd_u8_t xnn_min_u8(xnn_simd_u8_t a, xnn_simd_u8_t b) {
  return _mm_min_epu8(a, b);
}

static XNN_INLINE uint8_t xnn_horizontal_max_u8(xnn_simd_u8_t a) {
  xnn_simd_u8_t vmax = _mm_max_epu8(a, _mm_unpackhi_epi64(a, a));
  vmax = _mm_max_epu8(vmax, _mm_srli_epi64(vmax, 32));
  vmax = _mm_max_epu8(vmax, _mm_srli_epi32(vmax, 16));
  vmax = _mm_max_epu8(vmax, _mm_srli_epi16(vmax, 8));
  return (uint8_t)_mm_cvtsi128_si32(vmax);
}

static XNN_INLINE uint8_t xnn_horizontal_min_u8(xnn_simd_u8_t a) {
  xnn_simd_u8_t vmin = _mm_min_epu8(a, _mm_unpackhi_epi64(a, a));
  vmin = _mm_min_epu8(vmin, _mm_srli_epi64(vmin, 32));
  vmin = _mm_min_epu8(vmin, _mm_srli_epi32(vmin, 16));
  vmin = _mm_min_epu8(vmin, _mm_srli_epi16(vmin, 8));
  return (uint8_t)_mm_cvtsi128_si32(vmin);
}

static XNN_INLINE xnn_simd_u8_t xnn_xor_u8(xnn_simd_u8_t a, xnn_simd_u8_t b) {
  return _mm_xor_si128(a, b);
}

// Load/store operations.

static XNN_INLINE xnn_simd_u8_t xnn_loadu_u8(const uint8_t* ptr) {
  return _mm_loadu_si128((const __m128i*)ptr);
}

static XNN_INLINE xnn_simd_u8_t xnn_load_u8(const uint8_t* ptr) {
  return _mm_load_si128((const __m128i*)ptr);
}

static XNN_INLINE void xnn_storeu_u8(uint8_t* ptr, xnn_simd_u8_t v) {
  _mm_storeu_si128((__m128i*)ptr, v);
}

static XNN_INLINE void xnn_store_u8(uint8_t* ptr, xnn_simd_u8_t v) {
  _mm_store_si128((__m128i*)ptr, v);
}

static XNN_INLINE xnn_simd_u8_t xnn_set1_u8(uint8_t v) {
  return _mm_set1_epi8(v);
}

// Tail load/store operations.

static XNN_INLINE xnn_simd_u8_t
xnn_load_tail_u8(const uint8_t* input, size_t num_elements) XNN_OOB_READS {
  assert(num_elements > 0);
  assert(num_elements < xnn_simd_size_u8);
  return _mm_loadu_si128((const __m128i*)input);
}

static XNN_INLINE xnn_simd_u8_t xnn_load_tail_safe_u8(const uint8_t* input,
                                                      size_t num_elements) {
  assert(num_elements > 0);
  assert(num_elements < xnn_simd_size_u8);

  XNN_ALIGN(16) uint8_t padded[16];
  uint8_t* d = &padded[0];
  switch (num_elements) {
    case 15:
      *d++ = *input++;
    case 14:
      *d++ = *input++;
    case 13:
      *d++ = *input++;
    case 12:
      *d++ = *input++;
    case 11:
      *d++ = *input++;
    case 10:
      *d++ = *input++;
    case 9:
      *d++ = *input++;
    case 8:
      *d++ = *input++;
    case 7:
      *d++ = *input++;
    case 6:
      *d++ = *input++;
    case 5:
      *d++ = *input++;
    case 4:
      *d++ = *input++;
    case 3:
      *d++ = *input++;
    case 2:
      *d++ = *input++;
    case 1:
      *d++ = *input++;
  }
  return _mm_load_si128((const __m128i*)&padded[0]);
}

static XNN_INLINE void xnn_store_tail_u8(uint8_t* output, xnn_simd_u8_t v,
                                         size_t num_elements) {
  assert(num_elements > 0);
  assert(num_elements < xnn_simd_size_u8);

  if (num_elements & (8 * sizeof(uint8_t))) {
    _mm_storel_epi64((__m128i*)output, v);
    v = _mm_unpackhi_epi64(v, v);
    output += 8;
  }
  if (num_elements & (4 * sizeof(uint8_t))) {
    unaligned_store_u32(output, (uint32_t)_mm_cvtsi128_si32(v));
    v = _mm_srli_epi64(v, 32);
    output += 4;
  }
  if (num_elements & (2 * sizeof(uint8_t))) {
    unaligned_store_u16(output, (uint16_t)_mm_extract_epi16(v, 0));
    v = _mm_srli_epi32(v, 16);
    output += 2;
  }
  if (num_elements & (1 * sizeof(uint8_t))) {
    *output = (uint8_t)_mm_cvtsi128_si32(v);
  }
}

#endif  // __XNNPACK_SRC_XNNPACK_SIMD_U8_SSE2_H_
