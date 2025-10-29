// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//

#ifndef XNNPACK_SRC_XNNPACK_SIMD_F32_SSE2_BASE_H_
#define XNNPACK_SRC_XNNPACK_SIMD_F32_SSE2_BASE_H_

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

// Header file for SSE2 intrinsics.
#include <emmintrin.h>

#include "src/xnnpack/common.h"

// SIMD vector type for f32 using SSE2.
typedef __m128 xnn_simd_f32_t;
#define xnn_simd_size_f32 4
#define xnn_simd_log2_size_f32 2
#define xnn_simd_bytes_f32 (xnn_simd_size_f32 * sizeof(float))

#define XNN_SIMD_CONST_F32(var, val) \
  const xnn_simd_f32_t var = _mm_set1_ps(val);

#define XNN_SIMD_CONST_F32_FROM_INT32(var, val) \
  const xnn_simd_f32_t var = _mm_castsi128_ps(_mm_set1_epi32(val));

// Arithmetic operations.

static XNN_INLINE xnn_simd_f32_t xnn_zero_f32() { return _mm_setzero_ps(); }

static XNN_INLINE xnn_simd_f32_t xnn_add_f32(xnn_simd_f32_t a,
                                             xnn_simd_f32_t b) {
  return _mm_add_ps(a, b);
}

static XNN_INLINE xnn_simd_f32_t xnn_mul_f32(xnn_simd_f32_t a,
                                             xnn_simd_f32_t b) {
  return _mm_mul_ps(a, b);
}

static XNN_INLINE xnn_simd_f32_t xnn_sub_f32(xnn_simd_f32_t a,
                                             xnn_simd_f32_t b) {
  return _mm_sub_ps(a, b);
}

static XNN_INLINE xnn_simd_f32_t xnn_div_f32(xnn_simd_f32_t a,
                                             xnn_simd_f32_t b) {
  return _mm_div_ps(a, b);
}

static XNN_INLINE xnn_simd_f32_t xnn_max_f32(xnn_simd_f32_t a,
                                             xnn_simd_f32_t b) {
  return _mm_max_ps(a, b);
}

static XNN_INLINE xnn_simd_f32_t xnn_min_f32(xnn_simd_f32_t a,
                                             xnn_simd_f32_t b) {
  return _mm_min_ps(a, b);
}

static XNN_INLINE xnn_simd_f32_t xnn_abs_f32(xnn_simd_f32_t a) {
  XNN_SIMD_CONST_F32_FROM_INT32(vnonsign_mask, 0x7FFFFFFFUL);
  return _mm_and_ps(a, vnonsign_mask);
}

static XNN_INLINE xnn_simd_f32_t xnn_neg_f32(xnn_simd_f32_t a) {
  return xnn_sub_f32(xnn_zero_f32(), a);
}

// Logical operations.

static XNN_INLINE xnn_simd_f32_t xnn_and_f32(xnn_simd_f32_t a,
                                             xnn_simd_f32_t b) {
  return _mm_and_ps(a, b);
}

static XNN_INLINE xnn_simd_f32_t xnn_or_f32(xnn_simd_f32_t a,
                                            xnn_simd_f32_t b) {
  return _mm_or_ps(a, b);
}

static XNN_INLINE xnn_simd_f32_t xnn_xor_f32(xnn_simd_f32_t a,
                                             xnn_simd_f32_t b) {
  return _mm_xor_ps(a, b);
}

static XNN_INLINE xnn_simd_f32_t xnn_andnot_f32(xnn_simd_f32_t a,
                                                xnn_simd_f32_t b) {
  return _mm_andnot_ps(a, b);
}

static XNN_INLINE xnn_simd_f32_t xnn_sll_f32(xnn_simd_f32_t a, uint8_t bits) {
  return _mm_castsi128_ps(_mm_slli_epi32(_mm_castps_si128(a), bits));
}

static XNN_INLINE xnn_simd_f32_t xnn_srl_f32(xnn_simd_f32_t a, uint8_t bits) {
  return _mm_castsi128_ps(_mm_srli_epi32(_mm_castps_si128(a), bits));
}

static XNN_INLINE xnn_simd_f32_t xnn_sra_f32(xnn_simd_f32_t a, uint8_t bits) {
  return _mm_castsi128_ps(_mm_srai_epi32(_mm_castps_si128(a), bits));
}

static XNN_INLINE xnn_simd_f32_t xnn_cmpeq_f32(xnn_simd_f32_t a,
                                               xnn_simd_f32_t b) {
  return _mm_castsi128_ps(
      _mm_cmpeq_epi32(_mm_castps_si128(a), _mm_castps_si128(b)));
}

static XNN_INLINE xnn_simd_f32_t xnn_cmpneq_f32(xnn_simd_f32_t a,
                                                xnn_simd_f32_t b) {
  return _mm_cmpneq_ps(a, b);
}

static XNN_INLINE xnn_simd_f32_t xnn_round_f32(xnn_simd_f32_t a) {
  // Any input larger than 2^23 is already an integer value since its fractional
  // bits will no longer fit in the mantissa. We create a filter for these that
  // also catches all non-finite values in `a` (compares with NaN are always
  // `false`).
  XNN_SIMD_CONST_F32(vmax_non_int_val, 8388608.0f);  // 2^23.
  const xnn_simd_f32_t vfilter = _mm_cmplt_ps(xnn_abs_f32(a), vmax_non_int_val);

  // Round by converting to `int` and back.
  const xnn_simd_f32_t vresult = _mm_cvtepi32_ps(_mm_cvtps_epi32(a));

  // Apply the non-finite value filter to replace any non-finite input with `a`.
  return _mm_or_ps(_mm_and_ps(vfilter, vresult), _mm_andnot_ps(vfilter, a));
}

static XNN_INLINE float xnn_reduce_add_f32(xnn_simd_f32_t a) {
  a = _mm_add_ps(a, _mm_movehl_ps(a, a));
  a = _mm_add_ss(a, _mm_shuffle_ps(a, a, _MM_SHUFFLE(1, 1, 1, 1)));
  return _mm_cvtss_f32(a);
}

static XNN_INLINE float xnn_reduce_min_f32(xnn_simd_f32_t a) {
  a = _mm_min_ps(a, _mm_movehl_ps(a, a));
  a = _mm_min_ss(a, _mm_shuffle_ps(a, a, _MM_SHUFFLE(1, 1, 1, 1)));
  return _mm_cvtss_f32(a);
}

static XNN_INLINE float xnn_reduce_max_f32(xnn_simd_f32_t a) {
  a = _mm_max_ps(a, _mm_movehl_ps(a, a));
  a = _mm_max_ss(a, _mm_shuffle_ps(a, a, _MM_SHUFFLE(1, 1, 1, 1)));
  return _mm_cvtss_f32(a);
}

// Special functions.

#define XNN_SIMD_HAVE_RCP_F32 1
#define XNN_SIMD_NUM_RCP_ITER_F32 1
static XNN_INLINE xnn_simd_f32_t xnn_rcp_f32(xnn_simd_f32_t a) {
  return _mm_rcp_ps(a);
}

#define XNN_SIMD_HAVE_RSQRT_F32 1
#define XNN_SIMD_NUM_RSQRT_ITER_F32 1
static XNN_INLINE xnn_simd_f32_t xnn_rsqrt_f32(xnn_simd_f32_t a) {
  return _mm_rsqrt_ps(a);
}

static XNN_INLINE xnn_simd_f32_t xnn_sqrt_f32(xnn_simd_f32_t a) {
  return _mm_sqrt_ps(a);
}

// Load/store operations.

static XNN_INLINE xnn_simd_f32_t xnn_loadu_f32(const float* ptr) {
  return _mm_loadu_ps(ptr);
}

static XNN_INLINE xnn_simd_f32_t xnn_load_f32(const float* ptr) {
  return _mm_load_ps(ptr);
}

static XNN_INLINE void xnn_storeu_f32(float* ptr, xnn_simd_f32_t v) {
  _mm_storeu_ps(ptr, v);
}

static XNN_INLINE void xnn_store_f32(float* ptr, xnn_simd_f32_t v) {
  _mm_store_ps(ptr, v);
}

static XNN_INLINE xnn_simd_f32_t xnn_set1_f32(float v) {
  return _mm_set1_ps(v);
}

// Tail load/store operations.

static XNN_INLINE xnn_simd_f32_t
xnn_load_tail_f32(const float* input, size_t num_elements) XNN_OOB_READS {
  assert(num_elements > 0);
  assert(num_elements < xnn_simd_size_f32);
  return _mm_loadu_ps(input);
}

// TODO: Use direct load of 1,2 or 3 floats
// Consider clearing pad values to 0
static XNN_INLINE xnn_simd_f32_t xnn_load_tail_safe_f32(const float* input,
                                                        size_t num_elements) {
  assert(num_elements <= xnn_simd_size_f32);

  XNN_ALIGN(16) float padded[4] = {0.0f};
  float* dst = padded;
  switch (num_elements) {
    case 4:
      *dst++ = *input++;
      XNN_FALLTHROUGH
    case 3:
      *dst++ = *input++;
      XNN_FALLTHROUGH
    case 2:
      *dst++ = *input++;
      XNN_FALLTHROUGH
    case 1:
      *dst++ = *input++;
      XNN_FALLTHROUGH
    default:
      break;
  }
  return _mm_load_ps(padded);
}

static XNN_INLINE void xnn_store_tail_f32(float* output, xnn_simd_f32_t v,
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

#endif  // XNNPACK_SRC_XNNPACK_SIMD_F32_SSE2_BASE_H_
