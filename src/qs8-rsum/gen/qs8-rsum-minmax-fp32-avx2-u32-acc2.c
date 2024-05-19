// Auto-generated file. Do not edit!
//   Template: src/qs8-rsum/avx2.c.in
//   Generator: tools/xngen
//
// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <immintrin.h>

#include <xnnpack/common.h>
#include <xnnpack/math.h>
#include <xnnpack/reduce.h>


void xnn_qs8_rsum_minmax_fp32_ukernel__avx2_u32_acc2(
    size_t batch,
    const int8_t* input,
    int8_t* output,
    const union xnn_qs8_avgpool_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(input != NULL);
  assert(output != NULL);
  assert(params != NULL);

  __m256i vacc0 = _mm256_setzero_si256();
  __m256i vacc1 = _mm256_setzero_si256();

  for (; batch >= 32 * sizeof(int8_t); batch -= 32 * sizeof(int8_t)) {
    const __m256i vt0 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) input + 0));
    const __m256i vt1 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) input + 8));
    const __m256i vt2 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) input + 16));
    const __m256i vt3 = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) input + 24));
    input += 32;

    vacc0 = _mm256_add_epi32(vacc0, vt0);
    vacc1 = _mm256_add_epi32(vacc1, vt1);
    vacc0 = _mm256_add_epi32(vacc0, vt2);
    vacc1 = _mm256_add_epi32(vacc1, vt3);
  }
  vacc0 = _mm256_add_epi32(vacc0, vacc1);

  for (; batch >= 8 * sizeof(int8_t); batch -= 8 * sizeof(int8_t)) {
    const __m256i vt = _mm256_cvtepi8_epi32(_mm_loadl_epi64((const __m128i*) input));
    input += 8;

    vacc0 = _mm256_add_epi32(vacc0, vt);
  }
  if XNN_UNLIKELY(batch != 0) {
    assert(batch >= 1 * sizeof(int8_t));
    assert(batch <= 7 * sizeof(int8_t));

    const __m128i vmask = _mm_loadl_epi64((const __m128i*) ((uintptr_t) &params->fp32_avx2.mask_table[7] - batch));
    const __m256i vt = _mm256_cvtepi8_epi32(_mm_and_si128(_mm_loadl_epi64((const __m128i*) input), vmask));

    vacc0 = _mm256_add_epi32(vacc0, vt);
  }

  __m128i vacc = _mm_add_epi32(_mm256_castsi256_si128(vacc0), _mm256_extractf128_si256(vacc0, 1));
  vacc = _mm_add_epi32(vacc, _mm_castps_si128(_mm_movehl_ps(_mm_castsi128_ps(vacc), _mm_castsi128_ps(vacc))));
  vacc = _mm_hadd_epi32(vacc, vacc);

  const int32_t vinit_bias = params->fp32_avx2.init_bias;
  const float vscale = params->fp32_avx2.scale;
  const int32_t output_min = params->fp32_avx2.output_min;
  const int32_t output_max = params->fp32_avx2.output_max;
  const float vmagic_bias = params->fp32_avx2.magic_bias;
  const int32_t vmagic_bias_less_output_zero_point = params->fp32_avx2.magic_bias_less_output_zero_point;

  float vfpacc = (float) (_mm_cvtsi128_si32(vacc) + vinit_bias) * vscale;
  vfpacc += vmagic_bias;
  int32_t vout = (int32_t) float_as_uint32(vfpacc);
  vout -= vmagic_bias_less_output_zero_point;
  vout = math_max_s32(vout, output_min);
  vout = math_min_s32(vout, output_max);
  *output += (int8_t) vout;
}
