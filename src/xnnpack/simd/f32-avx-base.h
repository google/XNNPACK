// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//

// This header file contains portable SIMD wrappers common to AVX, AVX2, and
// FMA3. Do not include this header directly.
//
// Portable SIMD wrapper headers including this file need to implement the
// following wrappers themselves:
//
//   - xnn_cmpeq_f32
//   - xnn_fmadd_f32
//   - xnn_fmsub_f32
//   - xnn_fnmadd_f32
//   - xnn_sll_f32
//   - xnn_srl_f32

#ifndef __XNNPACK_SRC_XNNPACK_SIMD_F32_AVX_BASE_H_
#define __XNNPACK_SRC_XNNPACK_SIMD_F32_AVX_BASE_H_

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include <immintrin.h>

#include "xnnpack/common.h"


// SIMD vector type for f32 using AVX.
typedef __m256 xnn_simd_f32_t;
#define xnn_simd_size_f32 8
#define xnn_simd_log2_size_f32 3
#define xnn_simd_bytes_f32 (xnn_simd_size_f32 * sizeof(float))

#define XNN_SIMD_CONST_F32(var, val) \
  const xnn_simd_f32_t var = _mm256_set1_ps(val);

#define XNN_SIMD_CONST_F32_FROM_INT32(var, val) \
  const xnn_simd_f32_t var = _mm256_castsi256_ps(_mm256_set1_epi32(val));

// Include the header for generic functions _after_ declaring the arch-specific
// types and sizes.
#include "xnnpack/simd/f32-generic-functions.h"

// Mask table used for masked load/store operations.
static const int32_t mask_table_avx_f32[14] = {-1, -1, -1, -1, -1, -1, -1,
                                               0,  0,  0,  0,  0,  0,  0};

// Arithmetic operations.
static XNN_INLINE xnn_simd_f32_t xnn_zero_f32() { return _mm256_setzero_ps(); }

static XNN_INLINE xnn_simd_f32_t xnn_add_f32(xnn_simd_f32_t a,
                                             xnn_simd_f32_t b) {
  return _mm256_add_ps(a, b);
}

static XNN_INLINE xnn_simd_f32_t xnn_mul_f32(xnn_simd_f32_t a,
                                             xnn_simd_f32_t b) {
  return _mm256_mul_ps(a, b);
}

static XNN_INLINE xnn_simd_f32_t xnn_sub_f32(xnn_simd_f32_t a,
                                             xnn_simd_f32_t b) {
  return _mm256_sub_ps(a, b);
}

static XNN_INLINE xnn_simd_f32_t xnn_div_f32(xnn_simd_f32_t a,
                                             xnn_simd_f32_t b) {
  return _mm256_div_ps(a, b);
}

static XNN_INLINE xnn_simd_f32_t xnn_max_f32(xnn_simd_f32_t a,
                                             xnn_simd_f32_t b) {
  return _mm256_max_ps(a, b);
}

static XNN_INLINE xnn_simd_f32_t xnn_min_f32(xnn_simd_f32_t a,
                                             xnn_simd_f32_t b) {
  return _mm256_min_ps(a, b);
}

static XNN_INLINE xnn_simd_f32_t xnn_abs_f32(xnn_simd_f32_t a) {
  XNN_SIMD_CONST_F32_FROM_INT32(vnonsign_mask, 0x7FFFFFFFUL);
  return _mm256_and_ps(a, vnonsign_mask);
}

static XNN_INLINE xnn_simd_f32_t xnn_neg_f32(xnn_simd_f32_t a) {
  return xnn_sub_f32(xnn_zero_f32(), a);
}

// Logical operations.
static XNN_INLINE xnn_simd_f32_t xnn_and_f32(xnn_simd_f32_t a,
                                             xnn_simd_f32_t b) {
  return _mm256_and_ps(a, b);
}

static XNN_INLINE xnn_simd_f32_t xnn_or_f32(xnn_simd_f32_t a,
                                            xnn_simd_f32_t b) {
  return _mm256_or_ps(a, b);
}

static XNN_INLINE xnn_simd_f32_t xnn_xor_f32(xnn_simd_f32_t a,
                                             xnn_simd_f32_t b) {
  return _mm256_xor_ps(a, b);
}

// Special functions.
#define XNN_SIMD_HAVE_RCP_F32 1
#define XNN_SIMD_NUM_RCP_ITER_F32 1
static XNN_INLINE xnn_simd_f32_t xnn_rcp_f32(xnn_simd_f32_t a) {
  return _mm256_rcp_ps(a);
}

#define XNN_SIMD_HAVE_RSQRT_F32 1
#define XNN_SIMD_NUM_RSQRT_ITER_F32 1
static XNN_INLINE xnn_simd_f32_t xnn_rsqrt_f32(xnn_simd_f32_t a) {
  return _mm256_rsqrt_ps(a);
}

static XNN_INLINE xnn_simd_f32_t xnn_getexp_f32(xnn_simd_f32_t a) {
  return xnn_generic_getexp_f32(a);
}

// Load/store operations.
static XNN_INLINE xnn_simd_f32_t xnn_loadu_f32(const float* ptr) {
  return _mm256_loadu_ps(ptr);
}

static XNN_INLINE xnn_simd_f32_t xnn_load_f32(const float* ptr) {
  return _mm256_load_ps(ptr);
}

static XNN_INLINE void xnn_storeu_f32(float* ptr, xnn_simd_f32_t v) {
  _mm256_storeu_ps(ptr, v);
}

static XNN_INLINE void xnn_store_f32(float* ptr, xnn_simd_f32_t v) {
  _mm256_store_ps(ptr, v);
}

static XNN_INLINE xnn_simd_f32_t xnn_set1_f32(float v) {
  return _mm256_set1_ps(v);
}

static XNN_INLINE xnn_simd_f32_t xnn_set1_or_load_f32(const float* v) {
  return _mm256_set1_ps(*v);
}

// Tail load/store operations.
static XNN_INLINE xnn_simd_f32_t xnn_load_tail_f32(const float* input,
                                                   size_t num_elements) {
  assert(num_elements > 0);
  assert(num_elements < xnn_simd_size_f32);

  const __m256i vmask = _mm256_loadu_si256(
      (const __m256i*)(&mask_table_avx_f32[7] - num_elements));
  return _mm256_maskload_ps(input, vmask);
}

static XNN_INLINE void xnn_store_tail_f32(float* output, xnn_simd_f32_t v,
                                          size_t num_elements) {
  assert(num_elements > 0);
  assert(num_elements < xnn_simd_size_f32);

  __m128 v_lo = _mm256_castps256_ps128(v);
  if (num_elements & 4) {
    _mm_storeu_ps(output, v_lo);
    v_lo = _mm256_extractf128_ps(v, 1);
    output += 4;
  }
  if (num_elements & 2) {
    _mm_storel_pi((__m64*)output, v_lo);
    v_lo = _mm_movehl_ps(v_lo, v_lo);
    output += 2;
  }
  if (num_elements & 1) {
    _mm_store_ss(output, v_lo);
  }
}

#endif  // __XNNPACK_SRC_XNNPACK_SIMD_F32_AVX_BASE_H_
