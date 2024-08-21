// Auto-generated file. Do not edit!
//   Template: src/f32-qs8-vcvt/avx512skx.c.in
//   Generator: tools/xngen
//
// Copyright 2021 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <immintrin.h>

#include "xnnpack/common.h"
#include "xnnpack/intrinsics-polyfill.h"
#include "xnnpack/vcvt.h"


void xnn_f32_qs8_vcvt_ukernel__avx512skx_u64(
    size_t batch,
    const float* input,
    int8_t* output,
    const union xnn_f32_qs8_cvt_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  XNN_ALIGN(64) static const uint32_t shuffle512_mask[16] = {0, 4, 8, 12, 1, 5, 9, 13, 2, 6, 10, 14, 3, 7, 11, 15};


  const __m512 vscale = _mm512_set1_ps(params->scalar.scale);
  const __m512 voutput_max_less_zero_point = _mm512_set1_ps((float) ((int32_t) params->scalar.output_max - (int32_t) params->scalar.output_zero_point));
  const __m512i voutput_zero_point = _mm512_set1_epi16(params->scalar.output_zero_point);
  const __m512i vshuffle512_mask = _mm512_load_si512(shuffle512_mask);
  const __m512i voutput_min = _mm512_set1_epi8(params->scalar.output_min);
  XNN_FORCE_REALIZATION(vscale);
  XNN_FORCE_REALIZATION(voutput_max_less_zero_point);
  XNN_FORCE_REALIZATION(voutput_zero_point);
  XNN_FORCE_REALIZATION(voutput_min);
  for (; batch >= 64 * sizeof(float); batch -= 64 * sizeof(float)) {
    __m512 vx0123 = _mm512_loadu_ps(input);
    __m512 vx4567 = _mm512_loadu_ps(input + 16);
    __m512 vx89AB = _mm512_loadu_ps(input + 32);
    __m512 vxCDEF = _mm512_loadu_ps(input + 48);
    input += 64;

    vx0123 = _mm512_mul_ps(vx0123, vscale);
    vx4567 = _mm512_mul_ps(vx4567, vscale);
    vx89AB = _mm512_mul_ps(vx89AB, vscale);
    vxCDEF = _mm512_mul_ps(vxCDEF, vscale);

    vx0123 = _mm512_min_ps(vx0123, voutput_max_less_zero_point);
    vx4567 = _mm512_min_ps(vx4567, voutput_max_less_zero_point);
    vx89AB = _mm512_min_ps(vx89AB, voutput_max_less_zero_point);
    vxCDEF = _mm512_min_ps(vxCDEF, voutput_max_less_zero_point);

    const __m512i vacc0123 = _mm512_cvtps_epi32(vx0123);
    const __m512i vacc4567 = _mm512_cvtps_epi32(vx4567);
    const __m512i vacc89AB = _mm512_cvtps_epi32(vx89AB);
    const __m512i vaccCDEF = _mm512_cvtps_epi32(vxCDEF);

    __m512i vacc04152637 = _mm512_packs_epi32(vacc0123, vacc4567);
    __m512i vacc8C9DAEBF = _mm512_packs_epi32(vacc89AB, vaccCDEF);

    vacc04152637 = _mm512_adds_epi16(vacc04152637, voutput_zero_point);
    vacc8C9DAEBF = _mm512_adds_epi16(vacc8C9DAEBF, voutput_zero_point);

    __m512i vy048C159D26AE37BF = _mm512_packs_epi16(vacc04152637, vacc8C9DAEBF);

    vy048C159D26AE37BF = _mm512_max_epi8(vy048C159D26AE37BF, voutput_min);

    const __m512i vy0123456789ABCDEF = _mm512_permutexvar_epi32(vshuffle512_mask, vy048C159D26AE37BF);

    _mm512_storeu_si512(output, vy0123456789ABCDEF);
    output += 64;
  }
  for (; batch >= 16 * sizeof(float); batch -= 16 * sizeof(float)) {
    __m512 vx0123 = _mm512_loadu_ps(input);
    vx0123 = _mm512_mul_ps(vx0123, vscale);
    vx0123 = _mm512_min_ps(vx0123, voutput_max_less_zero_point);
    input += 16;

    const __m512i vacc0123 = _mm512_cvtps_epi32(vx0123);

    __m256i vacc0213 = _mm256_packs_epi32(_mm512_castsi512_si256(vacc0123), _mm512_extracti32x8_epi32(vacc0123, 1));
    vacc0213 = _mm256_adds_epi16(vacc0213, _mm512_castsi512_si256(voutput_zero_point));
    const __m128i vy0213 = _mm_packs_epi16(_mm256_castsi256_si128(vacc0213), _mm256_extracti128_si256(vacc0213, 1));
    __m128i vy0123 = _mm_shuffle_epi32(vy0213, _MM_SHUFFLE(3, 1, 2, 0));
    vy0123 = _mm_max_epi8(vy0123, _mm512_castsi512_si128(voutput_min));

    _mm_storeu_si128((__m128i*) output, vy0123);
    output += 16;
  }
  if XNN_UNLIKELY(batch != 0) {
    assert(batch >= 1 * sizeof(float));
    assert(batch <= 15 * sizeof(float));

    // Prepare mask for valid elements (depends on batch).
    batch >>= XNN_LOG2_SIZEOF_FLOAT;
    const __mmask16 vmask = _cvtu32_mask16((uint32_t) ((UINT32_C(1) << batch) - UINT32_C(1)));

    __m512 vx0123 = _mm512_maskz_loadu_ps(vmask, input);
    vx0123 = _mm512_mul_ps(vx0123, vscale);
    vx0123 = _mm512_min_ps(vx0123, voutput_max_less_zero_point);

    const __m512i vacc0123 = _mm512_cvtps_epi32(vx0123);

    __m256i vacc0213 = _mm256_packs_epi32(_mm512_castsi512_si256(vacc0123), _mm512_extracti32x8_epi32(vacc0123, 1));
    vacc0213 = _mm256_adds_epi16(vacc0213, _mm512_castsi512_si256(voutput_zero_point));
    const __m128i vy0213 = _mm_packs_epi16(_mm256_castsi256_si128(vacc0213), _mm256_extracti128_si256(vacc0213, 1));
    __m128i vy0123 = _mm_shuffle_epi32(vy0213, _MM_SHUFFLE(3, 1, 2, 0));
    vy0123 = _mm_max_epi8(vy0123, _mm512_castsi512_si128(voutput_min));

    _mm_mask_storeu_epi8(output, vmask, vy0123);
  }
}
