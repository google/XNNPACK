// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//

#ifndef XNNPACK_SRC_XNNPACK_SIMD_F32_AVX512F_H_
#define XNNPACK_SRC_XNNPACK_SIMD_F32_AVX512F_H_

#include <assert.h>
#include <immintrin.h>
#include <stddef.h>
#include <stdint.h>

#include "src/xnnpack/common.h"
#include "src/xnnpack/intrinsics-polyfill.h"
#include "src/xnnpack/math.h"
#include "src/xnnpack/unaligned.h"

// SIMD vector type for f32 using AVX512F.
typedef __m512 xnn_simd_f32_t;
#define xnn_simd_size_f32 16
#define xnn_simd_log2_size_f32 4
#define xnn_simd_bytes_f32 (xnn_simd_size_f32 * sizeof(float))

#define XNN_SIMD_CONST_F32(var, val) \
  const xnn_simd_f32_t var = _mm512_set1_ps(val);

#define XNN_SIMD_CONST_F32_FROM_INT32(var, val) \
  const __m512 var = _mm512_castsi512_ps(_mm512_set1_epi32(val));

// Whether or not this architecture has native fused multiply-add support.
#define XNN_SIMD_HAS_NATIVE_FMA 1
// Arithmetic operations.
static XNN_INLINE xnn_simd_f32_t xnn_zero_f32() { return _mm512_setzero_ps(); }

static XNN_INLINE xnn_simd_f32_t xnn_add_f32(xnn_simd_f32_t a,
                                             xnn_simd_f32_t b) {
  return _mm512_add_ps(a, b);
}

static XNN_INLINE xnn_simd_f32_t xnn_mul_f32(xnn_simd_f32_t a,
                                             xnn_simd_f32_t b) {
  return _mm512_mul_ps(a, b);
}

static XNN_INLINE xnn_simd_f32_t xnn_fmadd_f32(xnn_simd_f32_t a,
                                               xnn_simd_f32_t b,
                                               xnn_simd_f32_t c) {
  return _mm512_fmadd_ps(a, b, c);
}

static XNN_INLINE xnn_simd_f32_t xnn_fnmadd_f32(xnn_simd_f32_t a,
                                                xnn_simd_f32_t b,
                                                xnn_simd_f32_t c) {
  return _mm512_fnmadd_ps(a, b, c);
}

static XNN_INLINE xnn_simd_f32_t xnn_fmsub_f32(xnn_simd_f32_t a,
                                               xnn_simd_f32_t b,
                                               xnn_simd_f32_t c) {
  return _mm512_fmsub_ps(a, b, c);
}

static XNN_INLINE xnn_simd_f32_t xnn_sub_f32(xnn_simd_f32_t a,
                                             xnn_simd_f32_t b) {
  return _mm512_sub_ps(a, b);
}

static XNN_INLINE xnn_simd_f32_t xnn_div_f32(xnn_simd_f32_t a,
                                             xnn_simd_f32_t b) {
  return _mm512_div_ps(a, b);
}

static XNN_INLINE xnn_simd_f32_t xnn_max_f32(xnn_simd_f32_t a,
                                             xnn_simd_f32_t b) {
  return _mm512_max_ps(a, b);
}

static XNN_INLINE xnn_simd_f32_t xnn_min_f32(xnn_simd_f32_t a,
                                             xnn_simd_f32_t b) {
  return _mm512_min_ps(a, b);
}

static XNN_INLINE xnn_simd_f32_t xnn_abs_f32(xnn_simd_f32_t a) {
  XNN_SIMD_CONST_F32_FROM_INT32(vnonsign_mask, 0x7FFFFFFFUL);
  return _mm512_castsi512_ps(_mm512_and_epi32(
      _mm512_castps_si512(a), _mm512_castps_si512(vnonsign_mask)));
}

static XNN_INLINE xnn_simd_f32_t xnn_neg_f32(xnn_simd_f32_t a) {
  return xnn_sub_f32(xnn_zero_f32(), a);
}

static XNN_INLINE xnn_simd_f32_t xnn_round_f32(xnn_simd_f32_t a) {
  return _mm512_roundscale_ps(a, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
}

static XNN_INLINE float xnn_reduce_add_f32(xnn_simd_f32_t a) {
  // Not using _mm512_reduce_add_ps because we're trying to produce numerically
  // consistent results with AVX.
  __m256 a256 = _mm256_add_ps(
      _mm512_castps512_ps256(a),
      _mm256_castpd_ps(_mm512_extractf64x4_pd(_mm512_castps_pd(a), 1)));
  __m128 a128 =
      _mm_add_ps(_mm256_castps256_ps128(a256), _mm256_extractf128_ps(a256, 1));
  a128 = _mm_add_ps(a128, _mm_movehl_ps(a128, a128));
  a128 = _mm_add_ss(a128, _mm_movehdup_ps(a128));
  return _mm_cvtss_f32(a128);
}

static XNN_INLINE float xnn_reduce_min_f32(xnn_simd_f32_t a) {
  return _mm512_reduce_min_ps(a);
}

static XNN_INLINE float xnn_reduce_max_f32(xnn_simd_f32_t a) {
  return _mm512_reduce_max_ps(a);
}

// Logical operations.
static XNN_INLINE xnn_simd_f32_t xnn_and_f32(xnn_simd_f32_t a,
                                             xnn_simd_f32_t b) {
  return _mm512_castsi512_ps(
      _mm512_and_epi32(_mm512_castps_si512(a), _mm512_castps_si512(b)));
}

static XNN_INLINE xnn_simd_f32_t xnn_or_f32(xnn_simd_f32_t a,
                                            xnn_simd_f32_t b) {
  return _mm512_castsi512_ps(
      _mm512_or_epi32(_mm512_castps_si512(a), _mm512_castps_si512(b)));
}

static XNN_INLINE xnn_simd_f32_t xnn_xor_f32(xnn_simd_f32_t a,
                                             xnn_simd_f32_t b) {
  return _mm512_castsi512_ps(
      _mm512_xor_epi32(_mm512_castps_si512(a), _mm512_castps_si512(b)));
}

static XNN_INLINE xnn_simd_f32_t xnn_andnot_f32(xnn_simd_f32_t a,
                                                xnn_simd_f32_t b) {
  return _mm512_castsi512_ps(
      _mm512_andnot_epi32(_mm512_castps_si512(a), _mm512_castps_si512(b)));
}

static XNN_INLINE xnn_simd_f32_t xnn_sll_f32(xnn_simd_f32_t a, uint8_t bits) {
  return _mm512_castsi512_ps(_mm512_slli_epi32(_mm512_castps_si512(a), bits));
}

static XNN_INLINE xnn_simd_f32_t xnn_srl_f32(xnn_simd_f32_t a, uint8_t bits) {
  return _mm512_castsi512_ps(_mm512_srli_epi32(_mm512_castps_si512(a), bits));
}

static XNN_INLINE xnn_simd_f32_t xnn_sra_f32(xnn_simd_f32_t a, uint8_t bits) {
  return _mm512_castsi512_ps(_mm512_srai_epi32(_mm512_castps_si512(a), bits));
}

static XNN_INLINE xnn_simd_f32_t xnn_cmpeq_f32(xnn_simd_f32_t a,
                                               xnn_simd_f32_t b) {
  return _mm512_castsi512_ps(_mm512_maskz_set1_epi32(
      _mm512_cmp_ps_mask(a, b, _CMP_EQ_OQ), 0xFFFFFFFF));
}

static XNN_INLINE xnn_simd_f32_t xnn_cmpneq_f32(xnn_simd_f32_t a,
                                                xnn_simd_f32_t b) {
  return _mm512_castsi512_ps(_mm512_maskz_set1_epi32(
      _mm512_cmp_ps_mask(a, b, _CMP_NEQ_OQ), 0xFFFFFFFF));
}

// Special functions.
#define XNN_SIMD_HAVE_RCP_F32 1
#define XNN_SIMD_NUM_RCP_ITER_F32 1
static XNN_INLINE xnn_simd_f32_t xnn_rcp_f32(xnn_simd_f32_t a) {
  return _mm512_rcp14_ps(a);
}

#define XNN_SIMD_HAVE_RSQRT_F32 1
#define XNN_SIMD_NUM_RSQRT_ITER_F32 1
static XNN_INLINE xnn_simd_f32_t xnn_rsqrt_f32(xnn_simd_f32_t a) {
  return _mm512_rsqrt14_ps(a);
}

static XNN_INLINE xnn_simd_f32_t xnn_sqrt_f32(xnn_simd_f32_t a) {
  return _mm512_sqrt_ps(a);
}

// Load/store operations.
static XNN_INLINE xnn_simd_f32_t xnn_loadu_f32(const float* ptr) {
  return _mm512_loadu_ps(ptr);
}

static XNN_INLINE xnn_simd_f32_t xnn_load_f32(const float* ptr) {
  return _mm512_load_ps(ptr);
}

static XNN_INLINE void xnn_storeu_f32(float* ptr, xnn_simd_f32_t v) {
  _mm512_storeu_ps(ptr, v);
}

static XNN_INLINE void xnn_store_f32(float* ptr, xnn_simd_f32_t v) {
  _mm512_store_ps(ptr, v);
}

static XNN_INLINE xnn_simd_f32_t xnn_set1_f32(float v) {
  return _mm512_set1_ps(v);
}

// Tail load/store operations.
static XNN_INLINE xnn_simd_f32_t xnn_load_tail_f32(const float* input,
                                                   size_t num_elements) {
  assert(num_elements > 0);
  assert(num_elements < xnn_simd_size_f32);

  const __mmask16 vmask =
      _cvtu32_mask16((uint32_t)((UINT32_C(1) << num_elements) - UINT32_C(1)));
  return _mm512_maskz_loadu_ps(vmask, input);
}

static XNN_INLINE xnn_simd_f32_t xnn_load_tail_safe_f32(const float* input,
                                                        size_t num_elements) {
  return xnn_load_tail_f32(input, num_elements);
}

static XNN_INLINE void xnn_store_tail_f32(float* output, xnn_simd_f32_t v,
                                          size_t num_elements) {
  assert(num_elements > 0);
  assert(num_elements < xnn_simd_size_f32);

  const __mmask16 vmask =
      _cvtu32_mask16((uint32_t)((UINT32_C(1) << num_elements) - UINT32_C(1)));
  _mm512_mask_storeu_ps(output, vmask, v);
}

typedef __m256i xnn_simd_f16_t;

static XNN_INLINE xnn_simd_f16_t xnn_loadu_f16(const xnn_float16* ptr) {
  return _mm256_loadu_si256((const __m256i*)ptr);
}

static XNN_INLINE void xnn_storeu_f16(xnn_float16* ptr, xnn_simd_f16_t v) {
  _mm256_storeu_si256((__m256i*)ptr, v);
}

static XNN_INLINE xnn_simd_f32_t xnn_cvt_f32_f16(xnn_simd_f16_t a) {
  return _mm512_cvtph_ps(a);
}

static XNN_INLINE xnn_simd_f16_t xnn_cvt_f16_f32(xnn_simd_f32_t a) {
  return _mm512_cvtps_ph(a, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_NO_EXC);
}

static XNN_INLINE xnn_simd_f16_t xnn_load_tail_f16(const xnn_float16* input,
                                                   size_t num_elements) {
  assert(num_elements > 0);
  assert(num_elements <= xnn_simd_size_f32);
  if (num_elements == 16) {
    return _mm256_loadu_si256((const __m256i*)input);
  }
  const uint16_t* p = (const uint16_t*)input + num_elements;
  __m128i v_hi = _mm_setzero_si128();
  if (num_elements & 1) {
    p -= 1;
    v_hi = _mm_cvtsi32_si128(*p);
  }
  if (num_elements & 2) {
    p -= 2;
    v_hi = _mm_slli_epi64(v_hi, 32);
    v_hi = _mm_or_si128(v_hi, _mm_cvtsi32_si128(unaligned_load_s32(p)));
  }
  if (num_elements & 4) {
    p -= 4;
    v_hi = _mm_unpacklo_epi64(_mm_loadl_epi64((const __m128i*)p), v_hi);
  }
  if (num_elements & 8) {
    return _mm256_inserti128_si256(
        _mm256_castsi128_si256(_mm_loadu_si128((const __m128i*)input)), v_hi,
        1);
  }
  return _mm256_zextsi128_si256(v_hi);
}

static XNN_INLINE void xnn_store_tail_f16(xnn_float16* output, xnn_simd_f16_t v,
                                          size_t num_elements) {
  assert(num_elements > 0);
  assert(num_elements <= xnn_simd_size_f32);
  if (num_elements == 16) {
    _mm256_storeu_si256((__m256i*)output, v);
    return;
  }
  __m128i v_low = _mm256_castsi256_si128(v);
  if (num_elements & 8) {
    _mm_storeu_si128((__m128i*)output, v_low);
    v_low = _mm256_extracti128_si256(v, 1);
    output += 8;
  }
  if (num_elements & 4) {
    _mm_storel_epi64((__m128i*)output, v_low);
    v_low = _mm_unpackhi_epi64(v_low, v_low);
    output += 4;
  }
  if (num_elements & 2) {
    unaligned_store_u32(output, (uint32_t)_mm_cvtsi128_si32(v_low));
    v_low = _mm_srli_epi64(v_low, 32);
    output += 2;
  }
  if (num_elements & 1) {
    unaligned_store_u16(output, (uint16_t)_mm_extract_epi16(v_low, 0));
  }
}

#endif  // XNNPACK_SRC_XNNPACK_SIMD_F32_AVX512F_H_
