// Auto-generated file. Do not edit!
//   Template: src/qs8-rsum/ssse3.c.in
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

void xnn_qs8_rsum_minmax_fp32_ukernel__ssse3_u64_acc4(
    size_t batch,
    const int8_t* input,
    int8_t* output,
    const union xnn_qs8_avgpool_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(input != NULL);
  assert(output != NULL);
  assert(params != NULL);

  const __m128i vone = _mm_loadu_si128((const __m128i*) &params->fp32_ssse3.onemask_table[0]);
  const __m128i vone_16 = _mm_srli_epi16(vone, 8);
  __m128i vacc0 = _mm_setzero_si128();
  __m128i vacc1 = _mm_setzero_si128();
  __m128i vacc2 = _mm_setzero_si128();
  __m128i vacc3 = _mm_setzero_si128();

  for (; batch >= 512; batch -= 512) {
    __m128i vacc16_0 = _mm_setzero_si128();
    __m128i vacc16_1 = _mm_setzero_si128();
    __m128i vacc16_2 = _mm_setzero_si128();
    __m128i vacc16_3 = _mm_setzero_si128();
    for (size_t current_batch = 512; current_batch > 0; current_batch -= 64) {
      const __m128i vt0 = _mm_maddubs_epi16(vone, _mm_loadu_si128((const __m128i*) input)); input += 16;
      const __m128i vt1 = _mm_maddubs_epi16(vone, _mm_loadu_si128((const __m128i*) input)); input += 16;
      const __m128i vt2 = _mm_maddubs_epi16(vone, _mm_loadu_si128((const __m128i*) input)); input += 16;
      const __m128i vt3 = _mm_maddubs_epi16(vone, _mm_loadu_si128((const __m128i*) input)); input += 16;
      vacc16_0 = _mm_add_epi16(vacc16_0, vt0);
      vacc16_1 = _mm_add_epi16(vacc16_1, vt1);
      vacc16_2 = _mm_add_epi16(vacc16_2, vt2);
      vacc16_3 = _mm_add_epi16(vacc16_3, vt3);
    }
    vacc0 = _mm_add_epi32(vacc0, _mm_madd_epi16(vone_16, vacc16_0));
    vacc1 = _mm_add_epi32(vacc1, _mm_madd_epi16(vone_16, vacc16_1));
    vacc2 = _mm_add_epi32(vacc2, _mm_madd_epi16(vone_16, vacc16_2));
    vacc3 = _mm_add_epi32(vacc3, _mm_madd_epi16(vone_16, vacc16_3));
  }
  if (XNN_UNLIKELY(batch != 0)) {
    __m128i vacc16_0 = _mm_setzero_si128();
    __m128i vacc16_1 = _mm_setzero_si128();
    __m128i vacc16_2 = _mm_setzero_si128();
    __m128i vacc16_3 = _mm_setzero_si128();
    for (; batch >= 64; batch -= 64) {
      const __m128i vt0 = _mm_maddubs_epi16(vone, _mm_loadu_si128((const __m128i*) input)); input += 16;
      const __m128i vt1 = _mm_maddubs_epi16(vone, _mm_loadu_si128((const __m128i*) input)); input += 16;
      const __m128i vt2 = _mm_maddubs_epi16(vone, _mm_loadu_si128((const __m128i*) input)); input += 16;
      const __m128i vt3 = _mm_maddubs_epi16(vone, _mm_loadu_si128((const __m128i*) input)); input += 16;
      vacc16_0 = _mm_add_epi16(vacc16_0, vt0);
      vacc16_1 = _mm_add_epi16(vacc16_1, vt1);
      vacc16_2 = _mm_add_epi16(vacc16_2, vt2);
      vacc16_3 = _mm_add_epi16(vacc16_3, vt3);
    }
    vacc16_0 = _mm_add_epi16(vacc16_0, vacc16_1);
    vacc16_2 = _mm_add_epi16(vacc16_2, vacc16_3);
    vacc16_0 = _mm_add_epi16(vacc16_0, vacc16_2);
    for (; batch >= 16; batch -= 16) {
      const __m128i vt = _mm_maddubs_epi16(vone, _mm_loadu_si128((const __m128i*) input)); input += 16;
      vacc16_0 = _mm_add_epi16(vacc16_0, vt);
    }
    if (XNN_UNLIKELY(batch != 0)) {
      assert(batch >= 1 && batch <= 15);
      const __m128i vonemask = _mm_loadu_si128((const __m128i*) &params->fp32_ssse3.onemask_table[16 - batch]);
      const __m128i vt = _mm_maddubs_epi16(vonemask, _mm_loadu_si128((const __m128i*) input));
      vacc16_0 = _mm_add_epi16(vacc16_0, vt);
    }
    vacc0 = _mm_add_epi32(vacc0, _mm_madd_epi16(vone_16, vacc16_0));
  }
  vacc0 = _mm_add_epi32(vacc0, vacc1);
  vacc2 = _mm_add_epi32(vacc2, vacc3);
  vacc0 = _mm_add_epi32(vacc0, vacc2);

  __m128i vacc_lo = _mm_hadd_epi32(vacc0, vacc0);
  vacc_lo = _mm_hadd_epi32(vacc_lo, vacc_lo);
  const int32_t vacc = _mm_cvtsi128_si32(vacc_lo);

  const int32_t vinit_bias = params->fp32_ssse3.init_bias;
  const float vscale = params->fp32_ssse3.scale;
  const int32_t output_min = params->fp32_ssse3.output_min;
  const int32_t output_max = params->fp32_ssse3.output_max;
  const float vmagic_bias = params->fp32_ssse3.magic_bias;
  const int32_t vmagic_bias_less_output_zero_point = params->fp32_ssse3.magic_bias_less_output_zero_point;

  float vfpacc = (float) (vacc + vinit_bias) * vscale;
  vfpacc += vmagic_bias;
  int32_t vout = (int32_t) float_as_uint32(vfpacc);
  vout -= vmagic_bias_less_output_zero_point;
  vout = math_max_s32(vout, output_min);
  vout = math_min_s32(vout, output_max);
  *output += (int8_t) vout;
}
