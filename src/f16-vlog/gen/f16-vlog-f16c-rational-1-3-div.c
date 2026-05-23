// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/f16-vlog/f16c.c.in
//   Generator: tools/xngen
//
// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include <immintrin.h>

#include "src/xnnpack/common.h"
#include "src/xnnpack/intrinsics-polyfill.h"
#include "src/xnnpack/math.h"
#include "src/xnnpack/microparams.h"
#include "src/xnnpack/vunary.h"


void xnn_f16_vlog_ukernel__f16c_rational_1_3_div_u8(
    size_t batch,
    const xnn_float16* input,
    xnn_float16* output,
    const struct xnn_f16_default_params* unused_params) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(uint16_t) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const uint16_t* i = (const uint16_t*) input;
  uint16_t* o = (uint16_t*) output;

  const __m256 vone = _mm256_set1_ps(1.0f);
  const __m256 vln2 = _mm256_set1_ps(0.69314718f);
  const __m256 vmantissa_bits_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x007FFFFF));

  const __m256 vsqrt2 = _mm256_set1_ps(1.4142134190e+00f);
  const __m256 vsqrt1_2 = _mm256_set1_ps(7.0710688829e-01f);

  const __m256 vbeta_1 = _mm256_set1_ps(4.9951171875e-01f);
  const __m256 vbeta_2 = _mm256_set1_ps(-8.8439941406e-02f);
  const __m256 vbeta_3 = _mm256_set1_ps(4.8828125000e-02f);

  const __m256 vsign_mask = _mm256_set1_ps(-0.0f);
  const __m256 vsign_and_exp_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0xFF800000));
  const __m256 vbias_256 = _mm256_set1_ps(256.0f);
  const __m256 vbias_383 = _mm256_set1_ps(383.0f);

  for (; batch >= 8 * sizeof(uint16_t); batch -= 8 * sizeof(uint16_t)) {
    __m256 vx = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i));
    i += 8;

    vx = _mm256_mul_ps(vx, vsqrt2);

    const __m256 vzero_mask = _mm256_cmp_ps(vx, _mm256_setzero_ps(), _CMP_EQ_OQ);
    const __m256 va_ext = _mm256_or_ps(_mm256_and_ps(vzero_mask, vsign_mask), vx);
    const __m256 vtmp = _mm256_and_ps(va_ext, vsign_and_exp_mask);
    const __m128i vlo = _mm256_castsi256_si128(_mm256_castps_si256(vtmp));
    const __m128i vhi = _mm256_extractf128_si256(_mm256_castps_si256(vtmp), 1);
    __m256 vexp = _mm256_castsi256_ps(_mm256_insertf128_si256(_mm256_castsi128_si256(_mm_srai_epi32(vlo, 8)), _mm_srai_epi32(vhi, 8), 1));
    vexp = _mm256_sub_ps(_mm256_or_ps(vbias_256, vexp), vbias_383);

    vx = _mm256_or_ps(_mm256_and_ps(vx, vmantissa_bits_mask), vone);
    vx = _mm256_sub_ps(_mm256_mul_ps(vx, vsqrt1_2), vone);

    __m256 vq = _mm256_add_ps(_mm256_mul_ps(vx, vbeta_3), vbeta_2);
    vq = _mm256_add_ps(_mm256_mul_ps(vx, vq), vbeta_1);
    vq = _mm256_add_ps(_mm256_mul_ps(vx, vq), vone);

    __m256 vy = _mm256_div_ps(vx, vq);
    vy = _mm256_add_ps(_mm256_mul_ps(vexp, vln2), vy);

    _mm_storeu_si128((__m128i*) o, _mm256_cvtps_ph(vy, _MM_FROUND_TO_NEAREST_INT));
    o += 8;
  }
  if XNN_UNLIKELY(batch != 0) {
    __m256 vx = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i));

    vx = _mm256_mul_ps(vx, vsqrt2);

    const __m256 vzero_mask = _mm256_cmp_ps(vx, _mm256_setzero_ps(), _CMP_EQ_OQ);
    const __m256 va_ext = _mm256_or_ps(_mm256_and_ps(vzero_mask, vsign_mask), vx);
    const __m256 vtmp = _mm256_and_ps(va_ext, vsign_and_exp_mask);
    const __m128i vlo = _mm256_castsi256_si128(_mm256_castps_si256(vtmp));
    const __m128i vhi = _mm256_extractf128_si256(_mm256_castps_si256(vtmp), 1);
    __m256 vexp = _mm256_castsi256_ps(_mm256_insertf128_si256(_mm256_castsi128_si256(_mm_srai_epi32(vlo, 8)), _mm_srai_epi32(vhi, 8), 1));
    vexp = _mm256_sub_ps(_mm256_or_ps(vbias_256, vexp), vbias_383);

    vx = _mm256_or_ps(_mm256_and_ps(vx, vmantissa_bits_mask), vone);
    vx = _mm256_sub_ps(_mm256_mul_ps(vx, vsqrt1_2), vone);

    __m256 vq = _mm256_add_ps(_mm256_mul_ps(vx, vbeta_3), vbeta_2);
    vq = _mm256_add_ps(_mm256_mul_ps(vx, vq), vbeta_1);
    vq = _mm256_add_ps(_mm256_mul_ps(vx, vq), vone);

    __m256 vy = _mm256_div_ps(vx, vq);
    vy = _mm256_add_ps(_mm256_mul_ps(vexp, vln2), vy);

    __m128i vh = _mm256_cvtps_ph(vy, _MM_FROUND_TO_NEAREST_INT);
    if (batch & (4 * sizeof(uint16_t))) {
      _mm_storel_epi64((__m128i*) o, vh);
      vh = _mm_unpackhi_epi64(vh, vh);
      o += 4;
    }
    if (batch & (2 * sizeof(uint16_t))) {
      _mm_storeu_si32(o, vh);
      vh = _mm_srli_epi64(vh, 32);
      o += 2;
    }
    if (batch & (1 * sizeof(uint16_t))) {
      *o = (uint16_t) _mm_extract_epi16(vh, 0);
    }
  }
}
#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include <immintrin.h>

#include "src/xnnpack/common.h"
#include "src/xnnpack/intrinsics-polyfill.h"
#include "src/xnnpack/math.h"
#include "src/xnnpack/microparams.h"
#include "src/xnnpack/vunary.h"


void xnn_f16_vlog_ukernel__f16c_rational_1_3_div_u16(
    size_t batch,
    const xnn_float16* input,
    xnn_float16* output,
    const struct xnn_f16_default_params* unused_params) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(uint16_t) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const uint16_t* i = (const uint16_t*) input;
  uint16_t* o = (uint16_t*) output;

  const __m256 vone = _mm256_set1_ps(1.0f);
  const __m256 vln2 = _mm256_set1_ps(0.69314718f);
  const __m256 vmantissa_bits_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x007FFFFF));

  const __m256 vsqrt2 = _mm256_set1_ps(1.4142134190e+00f);
  const __m256 vsqrt1_2 = _mm256_set1_ps(7.0710688829e-01f);

  const __m256 vbeta_1 = _mm256_set1_ps(4.9951171875e-01f);
  const __m256 vbeta_2 = _mm256_set1_ps(-8.8439941406e-02f);
  const __m256 vbeta_3 = _mm256_set1_ps(4.8828125000e-02f);

  const __m256 vsign_mask = _mm256_set1_ps(-0.0f);
  const __m256 vsign_and_exp_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0xFF800000));
  const __m256 vbias_256 = _mm256_set1_ps(256.0f);
  const __m256 vbias_383 = _mm256_set1_ps(383.0f);

  for (; batch >= 16 * sizeof(uint16_t); batch -= 16 * sizeof(uint16_t)) {
    __m256 vx01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i + 0)));
    __m256 vx89ABCDEF = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i + 8)));
    i += 16;

    vx01234567 = _mm256_mul_ps(vx01234567, vsqrt2);
    vx89ABCDEF = _mm256_mul_ps(vx89ABCDEF, vsqrt2);

    const __m256 vzero_mask01234567 = _mm256_cmp_ps(vx01234567, _mm256_setzero_ps(), _CMP_EQ_OQ);
    const __m256 va_ext01234567 = _mm256_or_ps(_mm256_and_ps(vzero_mask01234567, vsign_mask), vx01234567);
    const __m256 vtmp01234567 = _mm256_and_ps(va_ext01234567, vsign_and_exp_mask);
    const __m128i vlo01234567 = _mm256_castsi256_si128(_mm256_castps_si256(vtmp01234567));
    const __m128i vhi01234567 = _mm256_extractf128_si256(_mm256_castps_si256(vtmp01234567), 1);
    __m256 vexp01234567 = _mm256_castsi256_ps(_mm256_insertf128_si256(_mm256_castsi128_si256(_mm_srai_epi32(vlo01234567, 8)), _mm_srai_epi32(vhi01234567, 8), 1));
    vexp01234567 = _mm256_sub_ps(_mm256_or_ps(vbias_256, vexp01234567), vbias_383);
    const __m256 vzero_mask89ABCDEF = _mm256_cmp_ps(vx89ABCDEF, _mm256_setzero_ps(), _CMP_EQ_OQ);
    const __m256 va_ext89ABCDEF = _mm256_or_ps(_mm256_and_ps(vzero_mask89ABCDEF, vsign_mask), vx89ABCDEF);
    const __m256 vtmp89ABCDEF = _mm256_and_ps(va_ext89ABCDEF, vsign_and_exp_mask);
    const __m128i vlo89ABCDEF = _mm256_castsi256_si128(_mm256_castps_si256(vtmp89ABCDEF));
    const __m128i vhi89ABCDEF = _mm256_extractf128_si256(_mm256_castps_si256(vtmp89ABCDEF), 1);
    __m256 vexp89ABCDEF = _mm256_castsi256_ps(_mm256_insertf128_si256(_mm256_castsi128_si256(_mm_srai_epi32(vlo89ABCDEF, 8)), _mm_srai_epi32(vhi89ABCDEF, 8), 1));
    vexp89ABCDEF = _mm256_sub_ps(_mm256_or_ps(vbias_256, vexp89ABCDEF), vbias_383);

    vx01234567 = _mm256_or_ps(_mm256_and_ps(vx01234567, vmantissa_bits_mask), vone);
    vx01234567 = _mm256_sub_ps(_mm256_mul_ps(vx01234567, vsqrt1_2), vone);
    vx89ABCDEF = _mm256_or_ps(_mm256_and_ps(vx89ABCDEF, vmantissa_bits_mask), vone);
    vx89ABCDEF = _mm256_sub_ps(_mm256_mul_ps(vx89ABCDEF, vsqrt1_2), vone);

    __m256 vq01234567 = _mm256_add_ps(_mm256_mul_ps(vx01234567, vbeta_3), vbeta_2);
    vq01234567 = _mm256_add_ps(_mm256_mul_ps(vx01234567, vq01234567), vbeta_1);
    vq01234567 = _mm256_add_ps(_mm256_mul_ps(vx01234567, vq01234567), vone);
    __m256 vq89ABCDEF = _mm256_add_ps(_mm256_mul_ps(vx89ABCDEF, vbeta_3), vbeta_2);
    vq89ABCDEF = _mm256_add_ps(_mm256_mul_ps(vx89ABCDEF, vq89ABCDEF), vbeta_1);
    vq89ABCDEF = _mm256_add_ps(_mm256_mul_ps(vx89ABCDEF, vq89ABCDEF), vone);

    __m256 vy01234567 = _mm256_div_ps(vx01234567, vq01234567);
    __m256 vy89ABCDEF = _mm256_div_ps(vx89ABCDEF, vq89ABCDEF);

    vy01234567 = _mm256_add_ps(_mm256_mul_ps(vexp01234567, vln2), vy01234567);
    vy89ABCDEF = _mm256_add_ps(_mm256_mul_ps(vexp89ABCDEF, vln2), vy89ABCDEF);

    _mm_storeu_si128((__m128i*) (o + 0), _mm256_cvtps_ph(vy01234567, _MM_FROUND_TO_NEAREST_INT));
    _mm_storeu_si128((__m128i*) (o + 8), _mm256_cvtps_ph(vy89ABCDEF, _MM_FROUND_TO_NEAREST_INT));
    o += 16;
  }
  for (; batch >= 8 * sizeof(uint16_t); batch -= 8 * sizeof(uint16_t)) {
    __m256 vx = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i));
    i += 8;

    vx = _mm256_mul_ps(vx, vsqrt2);

    const __m256 vzero_mask = _mm256_cmp_ps(vx, _mm256_setzero_ps(), _CMP_EQ_OQ);
    const __m256 va_ext = _mm256_or_ps(_mm256_and_ps(vzero_mask, vsign_mask), vx);
    const __m256 vtmp = _mm256_and_ps(va_ext, vsign_and_exp_mask);
    const __m128i vlo = _mm256_castsi256_si128(_mm256_castps_si256(vtmp));
    const __m128i vhi = _mm256_extractf128_si256(_mm256_castps_si256(vtmp), 1);
    __m256 vexp = _mm256_castsi256_ps(_mm256_insertf128_si256(_mm256_castsi128_si256(_mm_srai_epi32(vlo, 8)), _mm_srai_epi32(vhi, 8), 1));
    vexp = _mm256_sub_ps(_mm256_or_ps(vbias_256, vexp), vbias_383);

    vx = _mm256_or_ps(_mm256_and_ps(vx, vmantissa_bits_mask), vone);
    vx = _mm256_sub_ps(_mm256_mul_ps(vx, vsqrt1_2), vone);

    __m256 vq = _mm256_add_ps(_mm256_mul_ps(vx, vbeta_3), vbeta_2);
    vq = _mm256_add_ps(_mm256_mul_ps(vx, vq), vbeta_1);
    vq = _mm256_add_ps(_mm256_mul_ps(vx, vq), vone);

    __m256 vy = _mm256_div_ps(vx, vq);
    vy = _mm256_add_ps(_mm256_mul_ps(vexp, vln2), vy);

    _mm_storeu_si128((__m128i*) o, _mm256_cvtps_ph(vy, _MM_FROUND_TO_NEAREST_INT));
    o += 8;
  }
  if XNN_UNLIKELY(batch != 0) {
    __m256 vx = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i));

    vx = _mm256_mul_ps(vx, vsqrt2);

    const __m256 vzero_mask = _mm256_cmp_ps(vx, _mm256_setzero_ps(), _CMP_EQ_OQ);
    const __m256 va_ext = _mm256_or_ps(_mm256_and_ps(vzero_mask, vsign_mask), vx);
    const __m256 vtmp = _mm256_and_ps(va_ext, vsign_and_exp_mask);
    const __m128i vlo = _mm256_castsi256_si128(_mm256_castps_si256(vtmp));
    const __m128i vhi = _mm256_extractf128_si256(_mm256_castps_si256(vtmp), 1);
    __m256 vexp = _mm256_castsi256_ps(_mm256_insertf128_si256(_mm256_castsi128_si256(_mm_srai_epi32(vlo, 8)), _mm_srai_epi32(vhi, 8), 1));
    vexp = _mm256_sub_ps(_mm256_or_ps(vbias_256, vexp), vbias_383);

    vx = _mm256_or_ps(_mm256_and_ps(vx, vmantissa_bits_mask), vone);
    vx = _mm256_sub_ps(_mm256_mul_ps(vx, vsqrt1_2), vone);

    __m256 vq = _mm256_add_ps(_mm256_mul_ps(vx, vbeta_3), vbeta_2);
    vq = _mm256_add_ps(_mm256_mul_ps(vx, vq), vbeta_1);
    vq = _mm256_add_ps(_mm256_mul_ps(vx, vq), vone);

    __m256 vy = _mm256_div_ps(vx, vq);
    vy = _mm256_add_ps(_mm256_mul_ps(vexp, vln2), vy);

    __m128i vh = _mm256_cvtps_ph(vy, _MM_FROUND_TO_NEAREST_INT);
    if (batch & (4 * sizeof(uint16_t))) {
      _mm_storel_epi64((__m128i*) o, vh);
      vh = _mm_unpackhi_epi64(vh, vh);
      o += 4;
    }
    if (batch & (2 * sizeof(uint16_t))) {
      _mm_storeu_si32(o, vh);
      vh = _mm_srli_epi64(vh, 32);
      o += 2;
    }
    if (batch & (1 * sizeof(uint16_t))) {
      *o = (uint16_t) _mm_extract_epi16(vh, 0);
    }
  }
}
#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include <immintrin.h>

#include "src/xnnpack/common.h"
#include "src/xnnpack/intrinsics-polyfill.h"
#include "src/xnnpack/math.h"
#include "src/xnnpack/microparams.h"
#include "src/xnnpack/vunary.h"


void xnn_f16_vlog_ukernel__f16c_rational_1_3_div_u24(
    size_t batch,
    const xnn_float16* input,
    xnn_float16* output,
    const struct xnn_f16_default_params* unused_params) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(uint16_t) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const uint16_t* i = (const uint16_t*) input;
  uint16_t* o = (uint16_t*) output;

  const __m256 vone = _mm256_set1_ps(1.0f);
  const __m256 vln2 = _mm256_set1_ps(0.69314718f);
  const __m256 vmantissa_bits_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x007FFFFF));

  const __m256 vsqrt2 = _mm256_set1_ps(1.4142134190e+00f);
  const __m256 vsqrt1_2 = _mm256_set1_ps(7.0710688829e-01f);

  const __m256 vbeta_1 = _mm256_set1_ps(4.9951171875e-01f);
  const __m256 vbeta_2 = _mm256_set1_ps(-8.8439941406e-02f);
  const __m256 vbeta_3 = _mm256_set1_ps(4.8828125000e-02f);

  const __m256 vsign_mask = _mm256_set1_ps(-0.0f);
  const __m256 vsign_and_exp_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0xFF800000));
  const __m256 vbias_256 = _mm256_set1_ps(256.0f);
  const __m256 vbias_383 = _mm256_set1_ps(383.0f);

  for (; batch >= 24 * sizeof(uint16_t); batch -= 24 * sizeof(uint16_t)) {
    __m256 vx01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i + 0)));
    __m256 vx89ABCDEF = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i + 8)));
    __m256 vxGHIJKLMN = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i + 16)));
    i += 24;

    vx01234567 = _mm256_mul_ps(vx01234567, vsqrt2);
    vx89ABCDEF = _mm256_mul_ps(vx89ABCDEF, vsqrt2);
    vxGHIJKLMN = _mm256_mul_ps(vxGHIJKLMN, vsqrt2);

    const __m256 vzero_mask01234567 = _mm256_cmp_ps(vx01234567, _mm256_setzero_ps(), _CMP_EQ_OQ);
    const __m256 va_ext01234567 = _mm256_or_ps(_mm256_and_ps(vzero_mask01234567, vsign_mask), vx01234567);
    const __m256 vtmp01234567 = _mm256_and_ps(va_ext01234567, vsign_and_exp_mask);
    const __m128i vlo01234567 = _mm256_castsi256_si128(_mm256_castps_si256(vtmp01234567));
    const __m128i vhi01234567 = _mm256_extractf128_si256(_mm256_castps_si256(vtmp01234567), 1);
    __m256 vexp01234567 = _mm256_castsi256_ps(_mm256_insertf128_si256(_mm256_castsi128_si256(_mm_srai_epi32(vlo01234567, 8)), _mm_srai_epi32(vhi01234567, 8), 1));
    vexp01234567 = _mm256_sub_ps(_mm256_or_ps(vbias_256, vexp01234567), vbias_383);
    const __m256 vzero_mask89ABCDEF = _mm256_cmp_ps(vx89ABCDEF, _mm256_setzero_ps(), _CMP_EQ_OQ);
    const __m256 va_ext89ABCDEF = _mm256_or_ps(_mm256_and_ps(vzero_mask89ABCDEF, vsign_mask), vx89ABCDEF);
    const __m256 vtmp89ABCDEF = _mm256_and_ps(va_ext89ABCDEF, vsign_and_exp_mask);
    const __m128i vlo89ABCDEF = _mm256_castsi256_si128(_mm256_castps_si256(vtmp89ABCDEF));
    const __m128i vhi89ABCDEF = _mm256_extractf128_si256(_mm256_castps_si256(vtmp89ABCDEF), 1);
    __m256 vexp89ABCDEF = _mm256_castsi256_ps(_mm256_insertf128_si256(_mm256_castsi128_si256(_mm_srai_epi32(vlo89ABCDEF, 8)), _mm_srai_epi32(vhi89ABCDEF, 8), 1));
    vexp89ABCDEF = _mm256_sub_ps(_mm256_or_ps(vbias_256, vexp89ABCDEF), vbias_383);
    const __m256 vzero_maskGHIJKLMN = _mm256_cmp_ps(vxGHIJKLMN, _mm256_setzero_ps(), _CMP_EQ_OQ);
    const __m256 va_extGHIJKLMN = _mm256_or_ps(_mm256_and_ps(vzero_maskGHIJKLMN, vsign_mask), vxGHIJKLMN);
    const __m256 vtmpGHIJKLMN = _mm256_and_ps(va_extGHIJKLMN, vsign_and_exp_mask);
    const __m128i vloGHIJKLMN = _mm256_castsi256_si128(_mm256_castps_si256(vtmpGHIJKLMN));
    const __m128i vhiGHIJKLMN = _mm256_extractf128_si256(_mm256_castps_si256(vtmpGHIJKLMN), 1);
    __m256 vexpGHIJKLMN = _mm256_castsi256_ps(_mm256_insertf128_si256(_mm256_castsi128_si256(_mm_srai_epi32(vloGHIJKLMN, 8)), _mm_srai_epi32(vhiGHIJKLMN, 8), 1));
    vexpGHIJKLMN = _mm256_sub_ps(_mm256_or_ps(vbias_256, vexpGHIJKLMN), vbias_383);

    vx01234567 = _mm256_or_ps(_mm256_and_ps(vx01234567, vmantissa_bits_mask), vone);
    vx01234567 = _mm256_sub_ps(_mm256_mul_ps(vx01234567, vsqrt1_2), vone);
    vx89ABCDEF = _mm256_or_ps(_mm256_and_ps(vx89ABCDEF, vmantissa_bits_mask), vone);
    vx89ABCDEF = _mm256_sub_ps(_mm256_mul_ps(vx89ABCDEF, vsqrt1_2), vone);
    vxGHIJKLMN = _mm256_or_ps(_mm256_and_ps(vxGHIJKLMN, vmantissa_bits_mask), vone);
    vxGHIJKLMN = _mm256_sub_ps(_mm256_mul_ps(vxGHIJKLMN, vsqrt1_2), vone);

    __m256 vq01234567 = _mm256_add_ps(_mm256_mul_ps(vx01234567, vbeta_3), vbeta_2);
    vq01234567 = _mm256_add_ps(_mm256_mul_ps(vx01234567, vq01234567), vbeta_1);
    vq01234567 = _mm256_add_ps(_mm256_mul_ps(vx01234567, vq01234567), vone);
    __m256 vq89ABCDEF = _mm256_add_ps(_mm256_mul_ps(vx89ABCDEF, vbeta_3), vbeta_2);
    vq89ABCDEF = _mm256_add_ps(_mm256_mul_ps(vx89ABCDEF, vq89ABCDEF), vbeta_1);
    vq89ABCDEF = _mm256_add_ps(_mm256_mul_ps(vx89ABCDEF, vq89ABCDEF), vone);
    __m256 vqGHIJKLMN = _mm256_add_ps(_mm256_mul_ps(vxGHIJKLMN, vbeta_3), vbeta_2);
    vqGHIJKLMN = _mm256_add_ps(_mm256_mul_ps(vxGHIJKLMN, vqGHIJKLMN), vbeta_1);
    vqGHIJKLMN = _mm256_add_ps(_mm256_mul_ps(vxGHIJKLMN, vqGHIJKLMN), vone);

    __m256 vy01234567 = _mm256_div_ps(vx01234567, vq01234567);
    __m256 vy89ABCDEF = _mm256_div_ps(vx89ABCDEF, vq89ABCDEF);
    __m256 vyGHIJKLMN = _mm256_div_ps(vxGHIJKLMN, vqGHIJKLMN);

    vy01234567 = _mm256_add_ps(_mm256_mul_ps(vexp01234567, vln2), vy01234567);
    vy89ABCDEF = _mm256_add_ps(_mm256_mul_ps(vexp89ABCDEF, vln2), vy89ABCDEF);
    vyGHIJKLMN = _mm256_add_ps(_mm256_mul_ps(vexpGHIJKLMN, vln2), vyGHIJKLMN);

    _mm_storeu_si128((__m128i*) (o + 0), _mm256_cvtps_ph(vy01234567, _MM_FROUND_TO_NEAREST_INT));
    _mm_storeu_si128((__m128i*) (o + 8), _mm256_cvtps_ph(vy89ABCDEF, _MM_FROUND_TO_NEAREST_INT));
    _mm_storeu_si128((__m128i*) (o + 16), _mm256_cvtps_ph(vyGHIJKLMN, _MM_FROUND_TO_NEAREST_INT));
    o += 24;
  }
  for (; batch >= 8 * sizeof(uint16_t); batch -= 8 * sizeof(uint16_t)) {
    __m256 vx = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i));
    i += 8;

    vx = _mm256_mul_ps(vx, vsqrt2);

    const __m256 vzero_mask = _mm256_cmp_ps(vx, _mm256_setzero_ps(), _CMP_EQ_OQ);
    const __m256 va_ext = _mm256_or_ps(_mm256_and_ps(vzero_mask, vsign_mask), vx);
    const __m256 vtmp = _mm256_and_ps(va_ext, vsign_and_exp_mask);
    const __m128i vlo = _mm256_castsi256_si128(_mm256_castps_si256(vtmp));
    const __m128i vhi = _mm256_extractf128_si256(_mm256_castps_si256(vtmp), 1);
    __m256 vexp = _mm256_castsi256_ps(_mm256_insertf128_si256(_mm256_castsi128_si256(_mm_srai_epi32(vlo, 8)), _mm_srai_epi32(vhi, 8), 1));
    vexp = _mm256_sub_ps(_mm256_or_ps(vbias_256, vexp), vbias_383);

    vx = _mm256_or_ps(_mm256_and_ps(vx, vmantissa_bits_mask), vone);
    vx = _mm256_sub_ps(_mm256_mul_ps(vx, vsqrt1_2), vone);

    __m256 vq = _mm256_add_ps(_mm256_mul_ps(vx, vbeta_3), vbeta_2);
    vq = _mm256_add_ps(_mm256_mul_ps(vx, vq), vbeta_1);
    vq = _mm256_add_ps(_mm256_mul_ps(vx, vq), vone);

    __m256 vy = _mm256_div_ps(vx, vq);
    vy = _mm256_add_ps(_mm256_mul_ps(vexp, vln2), vy);

    _mm_storeu_si128((__m128i*) o, _mm256_cvtps_ph(vy, _MM_FROUND_TO_NEAREST_INT));
    o += 8;
  }
  if XNN_UNLIKELY(batch != 0) {
    __m256 vx = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i));

    vx = _mm256_mul_ps(vx, vsqrt2);

    const __m256 vzero_mask = _mm256_cmp_ps(vx, _mm256_setzero_ps(), _CMP_EQ_OQ);
    const __m256 va_ext = _mm256_or_ps(_mm256_and_ps(vzero_mask, vsign_mask), vx);
    const __m256 vtmp = _mm256_and_ps(va_ext, vsign_and_exp_mask);
    const __m128i vlo = _mm256_castsi256_si128(_mm256_castps_si256(vtmp));
    const __m128i vhi = _mm256_extractf128_si256(_mm256_castps_si256(vtmp), 1);
    __m256 vexp = _mm256_castsi256_ps(_mm256_insertf128_si256(_mm256_castsi128_si256(_mm_srai_epi32(vlo, 8)), _mm_srai_epi32(vhi, 8), 1));
    vexp = _mm256_sub_ps(_mm256_or_ps(vbias_256, vexp), vbias_383);

    vx = _mm256_or_ps(_mm256_and_ps(vx, vmantissa_bits_mask), vone);
    vx = _mm256_sub_ps(_mm256_mul_ps(vx, vsqrt1_2), vone);

    __m256 vq = _mm256_add_ps(_mm256_mul_ps(vx, vbeta_3), vbeta_2);
    vq = _mm256_add_ps(_mm256_mul_ps(vx, vq), vbeta_1);
    vq = _mm256_add_ps(_mm256_mul_ps(vx, vq), vone);

    __m256 vy = _mm256_div_ps(vx, vq);
    vy = _mm256_add_ps(_mm256_mul_ps(vexp, vln2), vy);

    __m128i vh = _mm256_cvtps_ph(vy, _MM_FROUND_TO_NEAREST_INT);
    if (batch & (4 * sizeof(uint16_t))) {
      _mm_storel_epi64((__m128i*) o, vh);
      vh = _mm_unpackhi_epi64(vh, vh);
      o += 4;
    }
    if (batch & (2 * sizeof(uint16_t))) {
      _mm_storeu_si32(o, vh);
      vh = _mm_srli_epi64(vh, 32);
      o += 2;
    }
    if (batch & (1 * sizeof(uint16_t))) {
      *o = (uint16_t) _mm_extract_epi16(vh, 0);
    }
  }
}
#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include <immintrin.h>

#include "src/xnnpack/common.h"
#include "src/xnnpack/intrinsics-polyfill.h"
#include "src/xnnpack/math.h"
#include "src/xnnpack/microparams.h"
#include "src/xnnpack/vunary.h"


void xnn_f16_vlog_ukernel__f16c_rational_1_3_div_u32(
    size_t batch,
    const xnn_float16* input,
    xnn_float16* output,
    const struct xnn_f16_default_params* unused_params) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(uint16_t) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const uint16_t* i = (const uint16_t*) input;
  uint16_t* o = (uint16_t*) output;

  const __m256 vone = _mm256_set1_ps(1.0f);
  const __m256 vln2 = _mm256_set1_ps(0.69314718f);
  const __m256 vmantissa_bits_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0x007FFFFF));

  const __m256 vsqrt2 = _mm256_set1_ps(1.4142134190e+00f);
  const __m256 vsqrt1_2 = _mm256_set1_ps(7.0710688829e-01f);

  const __m256 vbeta_1 = _mm256_set1_ps(4.9951171875e-01f);
  const __m256 vbeta_2 = _mm256_set1_ps(-8.8439941406e-02f);
  const __m256 vbeta_3 = _mm256_set1_ps(4.8828125000e-02f);

  const __m256 vsign_mask = _mm256_set1_ps(-0.0f);
  const __m256 vsign_and_exp_mask = _mm256_castsi256_ps(_mm256_set1_epi32(0xFF800000));
  const __m256 vbias_256 = _mm256_set1_ps(256.0f);
  const __m256 vbias_383 = _mm256_set1_ps(383.0f);

  for (; batch >= 32 * sizeof(uint16_t); batch -= 32 * sizeof(uint16_t)) {
    __m256 vx01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i + 0)));
    __m256 vx89ABCDEF = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i + 8)));
    __m256 vxGHIJKLMN = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i + 16)));
    __m256 vxOPQRSTUV = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i + 24)));
    i += 32;

    vx01234567 = _mm256_mul_ps(vx01234567, vsqrt2);
    vx89ABCDEF = _mm256_mul_ps(vx89ABCDEF, vsqrt2);
    vxGHIJKLMN = _mm256_mul_ps(vxGHIJKLMN, vsqrt2);
    vxOPQRSTUV = _mm256_mul_ps(vxOPQRSTUV, vsqrt2);

    const __m256 vzero_mask01234567 = _mm256_cmp_ps(vx01234567, _mm256_setzero_ps(), _CMP_EQ_OQ);
    const __m256 va_ext01234567 = _mm256_or_ps(_mm256_and_ps(vzero_mask01234567, vsign_mask), vx01234567);
    const __m256 vtmp01234567 = _mm256_and_ps(va_ext01234567, vsign_and_exp_mask);
    const __m128i vlo01234567 = _mm256_castsi256_si128(_mm256_castps_si256(vtmp01234567));
    const __m128i vhi01234567 = _mm256_extractf128_si256(_mm256_castps_si256(vtmp01234567), 1);
    __m256 vexp01234567 = _mm256_castsi256_ps(_mm256_insertf128_si256(_mm256_castsi128_si256(_mm_srai_epi32(vlo01234567, 8)), _mm_srai_epi32(vhi01234567, 8), 1));
    vexp01234567 = _mm256_sub_ps(_mm256_or_ps(vbias_256, vexp01234567), vbias_383);
    const __m256 vzero_mask89ABCDEF = _mm256_cmp_ps(vx89ABCDEF, _mm256_setzero_ps(), _CMP_EQ_OQ);
    const __m256 va_ext89ABCDEF = _mm256_or_ps(_mm256_and_ps(vzero_mask89ABCDEF, vsign_mask), vx89ABCDEF);
    const __m256 vtmp89ABCDEF = _mm256_and_ps(va_ext89ABCDEF, vsign_and_exp_mask);
    const __m128i vlo89ABCDEF = _mm256_castsi256_si128(_mm256_castps_si256(vtmp89ABCDEF));
    const __m128i vhi89ABCDEF = _mm256_extractf128_si256(_mm256_castps_si256(vtmp89ABCDEF), 1);
    __m256 vexp89ABCDEF = _mm256_castsi256_ps(_mm256_insertf128_si256(_mm256_castsi128_si256(_mm_srai_epi32(vlo89ABCDEF, 8)), _mm_srai_epi32(vhi89ABCDEF, 8), 1));
    vexp89ABCDEF = _mm256_sub_ps(_mm256_or_ps(vbias_256, vexp89ABCDEF), vbias_383);
    const __m256 vzero_maskGHIJKLMN = _mm256_cmp_ps(vxGHIJKLMN, _mm256_setzero_ps(), _CMP_EQ_OQ);
    const __m256 va_extGHIJKLMN = _mm256_or_ps(_mm256_and_ps(vzero_maskGHIJKLMN, vsign_mask), vxGHIJKLMN);
    const __m256 vtmpGHIJKLMN = _mm256_and_ps(va_extGHIJKLMN, vsign_and_exp_mask);
    const __m128i vloGHIJKLMN = _mm256_castsi256_si128(_mm256_castps_si256(vtmpGHIJKLMN));
    const __m128i vhiGHIJKLMN = _mm256_extractf128_si256(_mm256_castps_si256(vtmpGHIJKLMN), 1);
    __m256 vexpGHIJKLMN = _mm256_castsi256_ps(_mm256_insertf128_si256(_mm256_castsi128_si256(_mm_srai_epi32(vloGHIJKLMN, 8)), _mm_srai_epi32(vhiGHIJKLMN, 8), 1));
    vexpGHIJKLMN = _mm256_sub_ps(_mm256_or_ps(vbias_256, vexpGHIJKLMN), vbias_383);
    const __m256 vzero_maskOPQRSTUV = _mm256_cmp_ps(vxOPQRSTUV, _mm256_setzero_ps(), _CMP_EQ_OQ);
    const __m256 va_extOPQRSTUV = _mm256_or_ps(_mm256_and_ps(vzero_maskOPQRSTUV, vsign_mask), vxOPQRSTUV);
    const __m256 vtmpOPQRSTUV = _mm256_and_ps(va_extOPQRSTUV, vsign_and_exp_mask);
    const __m128i vloOPQRSTUV = _mm256_castsi256_si128(_mm256_castps_si256(vtmpOPQRSTUV));
    const __m128i vhiOPQRSTUV = _mm256_extractf128_si256(_mm256_castps_si256(vtmpOPQRSTUV), 1);
    __m256 vexpOPQRSTUV = _mm256_castsi256_ps(_mm256_insertf128_si256(_mm256_castsi128_si256(_mm_srai_epi32(vloOPQRSTUV, 8)), _mm_srai_epi32(vhiOPQRSTUV, 8), 1));
    vexpOPQRSTUV = _mm256_sub_ps(_mm256_or_ps(vbias_256, vexpOPQRSTUV), vbias_383);

    vx01234567 = _mm256_or_ps(_mm256_and_ps(vx01234567, vmantissa_bits_mask), vone);
    vx01234567 = _mm256_sub_ps(_mm256_mul_ps(vx01234567, vsqrt1_2), vone);
    vx89ABCDEF = _mm256_or_ps(_mm256_and_ps(vx89ABCDEF, vmantissa_bits_mask), vone);
    vx89ABCDEF = _mm256_sub_ps(_mm256_mul_ps(vx89ABCDEF, vsqrt1_2), vone);
    vxGHIJKLMN = _mm256_or_ps(_mm256_and_ps(vxGHIJKLMN, vmantissa_bits_mask), vone);
    vxGHIJKLMN = _mm256_sub_ps(_mm256_mul_ps(vxGHIJKLMN, vsqrt1_2), vone);
    vxOPQRSTUV = _mm256_or_ps(_mm256_and_ps(vxOPQRSTUV, vmantissa_bits_mask), vone);
    vxOPQRSTUV = _mm256_sub_ps(_mm256_mul_ps(vxOPQRSTUV, vsqrt1_2), vone);

    __m256 vq01234567 = _mm256_add_ps(_mm256_mul_ps(vx01234567, vbeta_3), vbeta_2);
    vq01234567 = _mm256_add_ps(_mm256_mul_ps(vx01234567, vq01234567), vbeta_1);
    vq01234567 = _mm256_add_ps(_mm256_mul_ps(vx01234567, vq01234567), vone);
    __m256 vq89ABCDEF = _mm256_add_ps(_mm256_mul_ps(vx89ABCDEF, vbeta_3), vbeta_2);
    vq89ABCDEF = _mm256_add_ps(_mm256_mul_ps(vx89ABCDEF, vq89ABCDEF), vbeta_1);
    vq89ABCDEF = _mm256_add_ps(_mm256_mul_ps(vx89ABCDEF, vq89ABCDEF), vone);
    __m256 vqGHIJKLMN = _mm256_add_ps(_mm256_mul_ps(vxGHIJKLMN, vbeta_3), vbeta_2);
    vqGHIJKLMN = _mm256_add_ps(_mm256_mul_ps(vxGHIJKLMN, vqGHIJKLMN), vbeta_1);
    vqGHIJKLMN = _mm256_add_ps(_mm256_mul_ps(vxGHIJKLMN, vqGHIJKLMN), vone);
    __m256 vqOPQRSTUV = _mm256_add_ps(_mm256_mul_ps(vxOPQRSTUV, vbeta_3), vbeta_2);
    vqOPQRSTUV = _mm256_add_ps(_mm256_mul_ps(vxOPQRSTUV, vqOPQRSTUV), vbeta_1);
    vqOPQRSTUV = _mm256_add_ps(_mm256_mul_ps(vxOPQRSTUV, vqOPQRSTUV), vone);

    __m256 vy01234567 = _mm256_div_ps(vx01234567, vq01234567);
    __m256 vy89ABCDEF = _mm256_div_ps(vx89ABCDEF, vq89ABCDEF);
    __m256 vyGHIJKLMN = _mm256_div_ps(vxGHIJKLMN, vqGHIJKLMN);
    __m256 vyOPQRSTUV = _mm256_div_ps(vxOPQRSTUV, vqOPQRSTUV);

    vy01234567 = _mm256_add_ps(_mm256_mul_ps(vexp01234567, vln2), vy01234567);
    vy89ABCDEF = _mm256_add_ps(_mm256_mul_ps(vexp89ABCDEF, vln2), vy89ABCDEF);
    vyGHIJKLMN = _mm256_add_ps(_mm256_mul_ps(vexpGHIJKLMN, vln2), vyGHIJKLMN);
    vyOPQRSTUV = _mm256_add_ps(_mm256_mul_ps(vexpOPQRSTUV, vln2), vyOPQRSTUV);

    _mm_storeu_si128((__m128i*) (o + 0), _mm256_cvtps_ph(vy01234567, _MM_FROUND_TO_NEAREST_INT));
    _mm_storeu_si128((__m128i*) (o + 8), _mm256_cvtps_ph(vy89ABCDEF, _MM_FROUND_TO_NEAREST_INT));
    _mm_storeu_si128((__m128i*) (o + 16), _mm256_cvtps_ph(vyGHIJKLMN, _MM_FROUND_TO_NEAREST_INT));
    _mm_storeu_si128((__m128i*) (o + 24), _mm256_cvtps_ph(vyOPQRSTUV, _MM_FROUND_TO_NEAREST_INT));
    o += 32;
  }
  for (; batch >= 8 * sizeof(uint16_t); batch -= 8 * sizeof(uint16_t)) {
    __m256 vx = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i));
    i += 8;

    vx = _mm256_mul_ps(vx, vsqrt2);

    const __m256 vzero_mask = _mm256_cmp_ps(vx, _mm256_setzero_ps(), _CMP_EQ_OQ);
    const __m256 va_ext = _mm256_or_ps(_mm256_and_ps(vzero_mask, vsign_mask), vx);
    const __m256 vtmp = _mm256_and_ps(va_ext, vsign_and_exp_mask);
    const __m128i vlo = _mm256_castsi256_si128(_mm256_castps_si256(vtmp));
    const __m128i vhi = _mm256_extractf128_si256(_mm256_castps_si256(vtmp), 1);
    __m256 vexp = _mm256_castsi256_ps(_mm256_insertf128_si256(_mm256_castsi128_si256(_mm_srai_epi32(vlo, 8)), _mm_srai_epi32(vhi, 8), 1));
    vexp = _mm256_sub_ps(_mm256_or_ps(vbias_256, vexp), vbias_383);

    vx = _mm256_or_ps(_mm256_and_ps(vx, vmantissa_bits_mask), vone);
    vx = _mm256_sub_ps(_mm256_mul_ps(vx, vsqrt1_2), vone);

    __m256 vq = _mm256_add_ps(_mm256_mul_ps(vx, vbeta_3), vbeta_2);
    vq = _mm256_add_ps(_mm256_mul_ps(vx, vq), vbeta_1);
    vq = _mm256_add_ps(_mm256_mul_ps(vx, vq), vone);

    __m256 vy = _mm256_div_ps(vx, vq);
    vy = _mm256_add_ps(_mm256_mul_ps(vexp, vln2), vy);

    _mm_storeu_si128((__m128i*) o, _mm256_cvtps_ph(vy, _MM_FROUND_TO_NEAREST_INT));
    o += 8;
  }
  if XNN_UNLIKELY(batch != 0) {
    __m256 vx = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i));

    vx = _mm256_mul_ps(vx, vsqrt2);

    const __m256 vzero_mask = _mm256_cmp_ps(vx, _mm256_setzero_ps(), _CMP_EQ_OQ);
    const __m256 va_ext = _mm256_or_ps(_mm256_and_ps(vzero_mask, vsign_mask), vx);
    const __m256 vtmp = _mm256_and_ps(va_ext, vsign_and_exp_mask);
    const __m128i vlo = _mm256_castsi256_si128(_mm256_castps_si256(vtmp));
    const __m128i vhi = _mm256_extractf128_si256(_mm256_castps_si256(vtmp), 1);
    __m256 vexp = _mm256_castsi256_ps(_mm256_insertf128_si256(_mm256_castsi128_si256(_mm_srai_epi32(vlo, 8)), _mm_srai_epi32(vhi, 8), 1));
    vexp = _mm256_sub_ps(_mm256_or_ps(vbias_256, vexp), vbias_383);

    vx = _mm256_or_ps(_mm256_and_ps(vx, vmantissa_bits_mask), vone);
    vx = _mm256_sub_ps(_mm256_mul_ps(vx, vsqrt1_2), vone);

    __m256 vq = _mm256_add_ps(_mm256_mul_ps(vx, vbeta_3), vbeta_2);
    vq = _mm256_add_ps(_mm256_mul_ps(vx, vq), vbeta_1);
    vq = _mm256_add_ps(_mm256_mul_ps(vx, vq), vone);

    __m256 vy = _mm256_div_ps(vx, vq);
    vy = _mm256_add_ps(_mm256_mul_ps(vexp, vln2), vy);

    __m128i vh = _mm256_cvtps_ph(vy, _MM_FROUND_TO_NEAREST_INT);
    if (batch & (4 * sizeof(uint16_t))) {
      _mm_storel_epi64((__m128i*) o, vh);
      vh = _mm_unpackhi_epi64(vh, vh);
      o += 4;
    }
    if (batch & (2 * sizeof(uint16_t))) {
      _mm_storeu_si32(o, vh);
      vh = _mm_srli_epi64(vh, 32);
      o += 2;
    }
    if (batch & (1 * sizeof(uint16_t))) {
      *o = (uint16_t) _mm_extract_epi16(vh, 0);
    }
  }
}
