// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//

#ifndef THIRD_PARTY_XNNPACK_SRC_XNNPACK_SIMD_F16_AVX512F_H_
#define THIRD_PARTY_XNNPACK_SRC_XNNPACK_SIMD_F16_AVX512F_H_

#include <assert.h>
#include <immintrin.h>
#include <stddef.h>
#include <stdint.h>

#include "src/xnnpack/common.h"
#include "src/xnnpack/math.h"

// SIMD vector type for f32 using AVX512F.
typedef __m512h xnn_simd_f16_t;
#define xnn_simd_size_f16 32
#define xnn_simd_log2_size_f16 5
#define xnn_simd_bytes_f16 (xnn_simd_size_f16 * sizeof(xnn_float16))

#define XNN_SIMD_CONST_F16(var, val) \
  const xnn_simd_f16_t var = _mm512_set1_ph(val);
#define XNN_SIMD_CONST_F16_FROM_INT16(var, val) \
  const xnn_simd_f16_t var = _mm512_castsi512_ph(_mm512_set1_epi16(val));

#if XNN_HAVE_FLOAT16

#if defined(__clang__) && (__clang_major__ < 19)
static XNN_INLINE __m512h xnn_broadcast_16_512_workaround(uint16_t x) {
  uint32_t bits = (uint32_t)x | ((uint32_t)x) << 16;
  __asm__ volatile("" : "=m"(bits) : "m"(bits));
  return (__m512h)_mm512_castsi512_pd(_mm512_set1_epi32(bits));
}
#define XNN_SIMD_CONST_F16_FROM_FLOAT(var, val)               \
  const xnn_simd_f16_t var = xnn_broadcast_16_512_workaround( \
      xnn_float16_to_bits(xnn_float16_from_float(val)))
#else
#define XNN_SIMD_CONST_F16_FROM_FLOAT(var, val) \
  const xnn_simd_f16_t var = _mm512_set1_ph(xnn_float16_from_float(val))
#endif  // Old Clang workaround

#else
#define XNN_SIMD_CONST_F16_FROM_FLOAT(var, val) \
  XNN_SIMD_CONST_F16_FROM_INT16(                \
      var, xnn_float16_to_bits(xnn_float16_from_float(val)))
#endif  // XNN_HAVE_FLOAT16

// Whether or not this architecture has native fused multiply-add support.
#define XNN_SIMD_HAS_NATIVE_FMA 1

// Arithmetic operations.
static XNN_INLINE xnn_simd_f16_t xnn_zero_f16() { return _mm512_setzero_ph(); }

static XNN_INLINE xnn_simd_f16_t xnn_add_f16(xnn_simd_f16_t a,
                                             xnn_simd_f16_t b) {
  return _mm512_add_ph(a, b);
}

static XNN_INLINE xnn_simd_f16_t xnn_mul_f16(xnn_simd_f16_t a,
                                             xnn_simd_f16_t b) {
  return _mm512_mul_ph(a, b);
}

static XNN_INLINE xnn_simd_f16_t xnn_fmadd_f16(xnn_simd_f16_t a,
                                               xnn_simd_f16_t b,
                                               xnn_simd_f16_t c) {
  return _mm512_fmadd_ph(a, b, c);
}

static XNN_INLINE xnn_simd_f16_t xnn_fnmadd_f16(xnn_simd_f16_t a,
                                                xnn_simd_f16_t b,
                                                xnn_simd_f16_t c) {
  return _mm512_fnmadd_ph(a, b, c);
}

static XNN_INLINE xnn_simd_f16_t xnn_fmsub_f16(xnn_simd_f16_t a,
                                               xnn_simd_f16_t b,
                                               xnn_simd_f16_t c) {
  return _mm512_fmsub_ph(a, b, c);
}

static XNN_INLINE xnn_simd_f16_t xnn_sub_f16(xnn_simd_f16_t a,
                                             xnn_simd_f16_t b) {
  return _mm512_sub_ph(a, b);
}

static XNN_INLINE xnn_simd_f16_t xnn_max_f16(xnn_simd_f16_t a,
                                             xnn_simd_f16_t b) {
  return _mm512_max_ph(a, b);
}

static XNN_INLINE xnn_simd_f16_t xnn_min_f16(xnn_simd_f16_t a,
                                             xnn_simd_f16_t b) {
  return _mm512_min_ph(a, b);
}

static XNN_INLINE xnn_simd_f16_t xnn_abs_f16(xnn_simd_f16_t a) {
  XNN_SIMD_CONST_F16_FROM_INT16(vnonsign_mask, 0x7FFF);
  return _mm512_castsi512_ph(_mm512_and_si512(
      _mm512_castph_si512(a), _mm512_castph_si512(vnonsign_mask)));
}

static XNN_INLINE xnn_simd_f16_t xnn_neg_f16(xnn_simd_f16_t a) {
  return xnn_sub_f16(xnn_zero_f16(), a);
}

static XNN_INLINE xnn_simd_f16_t xnn_round_f16(xnn_simd_f16_t a) {
  return _mm512_roundscale_ph(a, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
}

// Logical operations.
static XNN_INLINE xnn_simd_f16_t xnn_and_f16(xnn_simd_f16_t a,
                                             xnn_simd_f16_t b) {
  return _mm512_castsi512_ph(
      _mm512_and_si512(_mm512_castph_si512(a), _mm512_castph_si512(b)));
}

static XNN_INLINE xnn_simd_f16_t xnn_or_f16(xnn_simd_f16_t a,
                                            xnn_simd_f16_t b) {
  return _mm512_castsi512_ph(
      _mm512_or_si512(_mm512_castph_si512(a), _mm512_castph_si512(b)));
}

static XNN_INLINE xnn_simd_f16_t xnn_xor_f16(xnn_simd_f16_t a,
                                             xnn_simd_f16_t b) {
  return _mm512_castsi512_ph(
      _mm512_xor_si512(_mm512_castph_si512(a), _mm512_castph_si512(b)));
}

static XNN_INLINE xnn_simd_f16_t xnn_sll_f16(xnn_simd_f16_t a, uint8_t bits) {
  return _mm512_castsi512_ph(_mm512_slli_epi16(_mm512_castph_si512(a), bits));
}

static XNN_INLINE xnn_simd_f16_t xnn_srl_f16(xnn_simd_f16_t a, uint8_t bits) {
  return _mm512_castsi512_ph(_mm512_srli_epi16(_mm512_castph_si512(a), bits));
}

static XNN_INLINE xnn_simd_f16_t xnn_sra_f16(xnn_simd_f16_t a, uint8_t bits) {
  return _mm512_castsi512_ph(_mm512_srai_epi16(_mm512_castph_si512(a), bits));
}

static XNN_INLINE xnn_simd_f16_t xnn_cmpeq_f16(xnn_simd_f16_t a,
                                               xnn_simd_f16_t b) {
  return _mm512_castsi512_ph(
      _mm512_maskz_set1_epi16(_mm512_cmp_ph_mask(a, b, _CMP_EQ_OQ), 0xFFFF));
}

// Special functions.
#define XNN_SIMD_HAVE_RCP_F16 1
#define XNN_SIMD_NUM_RCP_ITER_F16 0
static XNN_INLINE xnn_simd_f16_t xnn_rcp_f16(xnn_simd_f16_t a) {
  return _mm512_rcp_ph(a);
}

static XNN_INLINE xnn_simd_f16_t xnn_div_f16(xnn_simd_f16_t a,
                                             xnn_simd_f16_t b) {
  return _mm512_mul_ph(a, _mm512_rcp_ph(b));
}

#define XNN_SIMD_HAVE_RSQRT_F16 1
#define XNN_SIMD_NUM_RSQRT_ITER_F16 0
static XNN_INLINE xnn_simd_f16_t xnn_rsqrt_f16(xnn_simd_f16_t a) {
  return _mm512_rsqrt_ph(a);
}

// Load/store operations.
static XNN_INLINE xnn_simd_f16_t xnn_loadu_f16(const xnn_float16* ptr) {
  return _mm512_loadu_ph(ptr);
}

static XNN_INLINE xnn_simd_f16_t xnn_load_f16(const xnn_float16* ptr) {
  return _mm512_load_ph(ptr);
}

static XNN_INLINE void xnn_storeu_f16(xnn_float16* ptr, xnn_simd_f16_t v) {
  _mm512_storeu_ph(ptr, v);
}

static XNN_INLINE void xnn_store_f16(xnn_float16* ptr, xnn_simd_f16_t v) {
  _mm512_store_ph(ptr, v);
}

static XNN_INLINE xnn_simd_f16_t xnn_set1_f16(xnn_float16 v) {
#if XNN_HAVE_FLOAT16
  return _mm512_set1_ph(v);
#else
  return _mm512_castsi512_ph(_mm512_set1_epi16(v.value));
#endif  // XNN_HAVE_FLOAT16
}

// Tail load/store operations.
static XNN_INLINE xnn_simd_f16_t xnn_load_tail_f16(const xnn_float16* input,
                                                   size_t num_elements) {
  assert(num_elements > 0);
  assert(num_elements < xnn_simd_size_f16);

  const __mmask32 vmask =
      ((uint32_t)((UINT32_C(1) << num_elements) - UINT32_C(1)));
  return _mm512_castsi512_ph(_mm512_maskz_loadu_epi16(vmask, input));
}

static XNN_INLINE xnn_simd_f16_t
xnn_load_tail_safe_f16(const xnn_float16* input, size_t num_elements) {
  return xnn_load_tail_f16(input, num_elements);
}

static XNN_INLINE void xnn_store_tail_f16(xnn_float16* output, xnn_simd_f16_t v,
                                          size_t num_elements) {
  assert(num_elements > 0);
  assert(num_elements < xnn_simd_size_f16);

  const __mmask32 vmask =
      _cvtu32_mask32((uint32_t)((UINT32_C(1) << num_elements) - UINT32_C(1)));
  _mm512_mask_storeu_epi16(output, vmask, _mm512_castph_si512(v));
}

#endif  // THIRD_PARTY_XNNPACK_SRC_XNNPACK_SIMD_F16_AVX512F_H_
