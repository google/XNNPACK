// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//

#ifndef THIRD_PARTY_XNNPACK_INCLUDE_SIMD_F32_SSE2_H_
#define THIRD_PARTY_XNNPACK_INCLUDE_SIMD_F32_SSE2_H_

#include <assert.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <xmmintrin.h>
#include <xnnpack/common.h>
#include <xnnpack/math.h>

// SIMD vector type for f32 using SSE2.
typedef __m128 xnn_simd_f32_t;
#define xnn_simd_size_f32 4
static const size_t xnn_simd_log2_size_f32 = 2;
static const size_t xnn_simd_bytes_f32 = xnn_simd_size_f32 * sizeof(float);

#if XNN_COMPILER_MSVC
#define xnn_simd_static_init_f32(var, val) \
  const xnn_simd_f32_t var = _mm_set1_ps(val);

#define xnn_simd_static_init_i32(var, val) \
  const __m128i var = _mm_set1_epi32(val);
#else
#define xnn_simd_static_init_f32(var, val) \
  static const xnn_simd_f32_t var = {(val), (val), (val), (val)};

#define xnn_simd_static_init_i32(var, val)                     \
  static const __m128i var = {((uint64_t)(val) << 32) | (val), \
                              ((uint64_t)(val) << 32) | (val)};
#endif  // XNN_COMPILER_MSVC

// Arithmetic operations.

XNN_INLINE xnn_simd_f32_t xnn_zero_f32() { return _mm_setzero_ps(); }

XNN_INLINE xnn_simd_f32_t xnn_add_f32(xnn_simd_f32_t a, xnn_simd_f32_t b) {
  return _mm_add_ps(a, b);
}

XNN_INLINE xnn_simd_f32_t xnn_mul_f32(xnn_simd_f32_t a, xnn_simd_f32_t b) {
  return _mm_mul_ps(a, b);
}

XNN_INLINE xnn_simd_f32_t xnn_fmadd_f32(xnn_simd_f32_t a, xnn_simd_f32_t b,
                                        xnn_simd_f32_t c) {
  return _mm_add_ps(_mm_mul_ps(a, b), c);
}

XNN_INLINE xnn_simd_f32_t xnn_fnmadd_f32(xnn_simd_f32_t a, xnn_simd_f32_t b,
                                         xnn_simd_f32_t c) {
  return _mm_sub_ps(c, _mm_mul_ps(a, b));
}

XNN_INLINE xnn_simd_f32_t xnn_sub_f32(xnn_simd_f32_t a, xnn_simd_f32_t b) {
  return _mm_sub_ps(a, b);
}

XNN_INLINE xnn_simd_f32_t xnn_div_f32(xnn_simd_f32_t a, xnn_simd_f32_t b) {
  return _mm_div_ps(a, b);
}

XNN_INLINE xnn_simd_f32_t xnn_max_f32(xnn_simd_f32_t a, xnn_simd_f32_t b) {
  return _mm_max_ps(a, b);
}

XNN_INLINE xnn_simd_f32_t xnn_min_f32(xnn_simd_f32_t a, xnn_simd_f32_t b) {
  return _mm_min_ps(a, b);
}

XNN_INLINE xnn_simd_f32_t xnn_abs_f32(xnn_simd_f32_t a) {
  xnn_simd_static_init_i32(vnonsign_mask, 0x7FFFFFFFUL);
  return _mm_and_ps(a, _mm_castsi128_ps(vnonsign_mask));
}

XNN_INLINE xnn_simd_f32_t xnn_neg_f32(xnn_simd_f32_t a) {
  xnn_simd_static_init_f32(vsign_mask, -0.0f);
  return _mm_xor_ps(a, vsign_mask);
}

// Logical operations.

XNN_INLINE xnn_simd_f32_t xnn_and_f32(xnn_simd_f32_t a, xnn_simd_f32_t b) {
  return _mm_and_ps(a, b);
}

XNN_INLINE xnn_simd_f32_t xnn_or_f32(xnn_simd_f32_t a, xnn_simd_f32_t b) {
  return _mm_or_ps(a, b);
}

XNN_INLINE xnn_simd_f32_t xnn_xor_f32(xnn_simd_f32_t a, xnn_simd_f32_t b) {
  return _mm_xor_ps(a, b);
}

// Special functions.

#define XNN_SIMD_HAVE_RCP_F32 1
#define XNN_SIMD_NUM_RCP_ITER_F32 1
XNN_INLINE xnn_simd_f32_t xnn_rcp_f32(xnn_simd_f32_t a) {
  return _mm_rcp_ps(a);
}

#define XNN_SIMD_HAVE_RSQRT_F32 1
#define XNN_SIMD_NUM_RSQRT_ITER_F32 1
XNN_INLINE xnn_simd_f32_t xnn_rsqrt_f32(xnn_simd_f32_t a) {
  return _mm_rsqrt_ps(a);
}

// Load/store operations.

XNN_INLINE xnn_simd_f32_t xnn_loadu_f32(const float* ptr) {
  return _mm_loadu_ps(ptr);
}

XNN_INLINE xnn_simd_f32_t xnn_load_f32(const float* ptr) {
  return _mm_load_ps(ptr);
}

XNN_INLINE void xnn_storeu_f32(float* ptr, xnn_simd_f32_t v) {
  _mm_storeu_ps(ptr, v);
}

XNN_INLINE void xnn_store_f32(float* ptr, xnn_simd_f32_t v) {
  _mm_store_ps(ptr, v);
}

XNN_INLINE xnn_simd_f32_t xnn_set1_f32(float v) { return _mm_set1_ps(v); }

XNN_INLINE xnn_simd_f32_t xnn_set1_or_load_f32(const float* v) {
#if XNN_ARCH_X86
  return _mm_load_ps(v);
#else
  return _mm_set1_ps(*v);
#endif
}

// Tail load/store operations.

XNN_INLINE xnn_simd_f32_t xnn_load_tail_f32(const float* input,
                                            size_t num_elements) XNN_OOB_READS {
  assert(num_elements > 0);
  assert(num_elements < xnn_simd_size_f32);
  return _mm_loadu_ps(input);
}

XNN_INLINE void xnn_store_tail_f32(float* output, xnn_simd_f32_t v,
                                   size_t num_elements) {
  assert(num_elements > 0);
  assert(num_elements < xnn_simd_size_f32);

  if (num_elements & 2) {
    _mm_storel_pi((__m64*)output, v);
    v = _mm_movehl_ps(v, v);
    output += 2;
  }
  if (num_elements & 1) {
    _mm_store_ss(output, v);
  }
}

#endif  // THIRD_PARTY_XNNPACK_INCLUDE_SIMD_F32_SSE2_H_
