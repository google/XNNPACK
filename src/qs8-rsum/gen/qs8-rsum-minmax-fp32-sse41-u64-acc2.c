// Auto-generated file. Do not edit!
//   Template: src/qs8-rsum/sse41.c.in
//   Generator: tools/xngen
//
// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stdio.h>

#include <immintrin.h>

#include <xnnpack/common.h>
#include <xnnpack/math.h>
#include <xnnpack/reduce.h>

void xnn_qs8_rsum_minmax_fp32_ukernel__sse41_u64_acc2(
    size_t batch,
    const int8_t* input,
    int8_t* output,
    const union xnn_qs8_avgpool_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(input != NULL);
  assert(output != NULL);
  assert(params != NULL);

  __m128i vacc0 = _mm_setzero_si128();
  __m128i vacc1 = _mm_setzero_si128();
  int num_batches = batch  >> 9;
  for (; num_batches > 0; --num_batches) {
    __m128i vacc16_0 = _mm_setzero_si128();
    __m128i vacc16_1 = _mm_setzero_si128();
    for (size_t current_batch = 512; current_batch > 0; current_batch -= 64) {
      const __m128i vt0 = _mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*) input)); input += 8;
      const __m128i vt1 = _mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*) input)); input += 8;
      const __m128i vt2 = _mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*) input)); input += 8;
      const __m128i vt3 = _mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*) input)); input += 8;
      const __m128i vt4 = _mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*) input)); input += 8;
      const __m128i vt5 = _mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*) input)); input += 8;
      const __m128i vt6 = _mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*) input)); input += 8;
      const __m128i vt7 = _mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*) input)); input += 8;

      vacc16_0 = _mm_add_epi16(vacc16_0, vt0);
      vacc16_1 = _mm_add_epi16(vacc16_1, vt1);
      vacc16_0 = _mm_add_epi16(vacc16_0, vt2);
      vacc16_1 = _mm_add_epi16(vacc16_1, vt3);
      vacc16_0 = _mm_add_epi16(vacc16_0, vt4);
      vacc16_1 = _mm_add_epi16(vacc16_1, vt5);
      vacc16_0 = _mm_add_epi16(vacc16_0, vt6);
      vacc16_1 = _mm_add_epi16(vacc16_1, vt7);
    }
    vacc0 = _mm_add_epi32(vacc0, _mm_add_epi32(_mm_cvtepi16_epi32(vacc16_0), _mm_cvtepi16_epi32(_mm_srli_si128(vacc16_0, 8))));
    vacc1 = _mm_add_epi32(vacc1, _mm_add_epi32(_mm_cvtepi16_epi32(vacc16_1), _mm_cvtepi16_epi32(_mm_srli_si128(vacc16_1, 8))));
    batch -= 512;
  }
  if (XNN_UNLIKELY(batch != 0)) {
    __m128i vacc16_0 = _mm_setzero_si128();
    __m128i vacc16_1 = _mm_setzero_si128();
    for (; batch >= 64; batch -= 64) {
      const __m128i vt0 = _mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*) input)); input += 8;
      const __m128i vt1 = _mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*) input)); input += 8;
      const __m128i vt2 = _mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*) input)); input += 8;
      const __m128i vt3 = _mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*) input)); input += 8;
      const __m128i vt4 = _mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*) input)); input += 8;
      const __m128i vt5 = _mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*) input)); input += 8;
      const __m128i vt6 = _mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*) input)); input += 8;
      const __m128i vt7 = _mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*) input)); input += 8;
      vacc16_0 = _mm_add_epi16(vacc16_0, vt0);
      vacc16_1 = _mm_add_epi16(vacc16_1, vt1);
      vacc16_0 = _mm_add_epi16(vacc16_0, vt2);
      vacc16_1 = _mm_add_epi16(vacc16_1, vt3);
      vacc16_0 = _mm_add_epi16(vacc16_0, vt4);
      vacc16_1 = _mm_add_epi16(vacc16_1, vt5);
      vacc16_0 = _mm_add_epi16(vacc16_0, vt6);
      vacc16_1 = _mm_add_epi16(vacc16_1, vt7);
    }

    vacc16_0 = _mm_add_epi16(vacc16_0, vacc16_1);

    for (; batch >= 8; batch -= 8) {
      const __m128i vt7 = _mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*) input)); input += 8;
      vacc16_0 = _mm_add_epi16(vacc16_0, vt7);
    }
    if (XNN_UNLIKELY(batch != 0)) {
      __m128i vt = _mm_loadl_epi64((const __m128i*) input);
      const __m128i vmask = _mm_loadl_epi64((const __m128i*) &params->fp32_sse4.mask_table[7 - batch]);
      vt = _mm_cvtepi8_epi16(_mm_and_si128(vt, vmask));
      vacc16_0 = _mm_add_epi16(vacc16_0, vt);
    }
    vacc0 = _mm_add_epi32(vacc0, _mm_add_epi32(_mm_cvtepi16_epi32(vacc16_0), _mm_cvtepi16_epi32(_mm_srli_si128(vacc16_0, 8))));
  }

  vacc0 = _mm_add_epi32(vacc0, vacc1);

  __m128i vacc_lo = _mm_hadd_epi32(vacc0, vacc0);
  vacc_lo = _mm_hadd_epi32(vacc_lo, vacc_lo);

  const int32_t vinit_bias = params->fp32_sse4.init_bias[0];
  const float vscale = params->fp32_sse4.scale[0];
  const int32_t output_min = params->fp32_sse4.output_min[0];
  const int32_t output_max = params->fp32_sse4.output_max[0];
  const float vmagic_bias = params->fp32_sse4.magic_bias[0];
  const int32_t vmagic_bias_less_output_zero_point = params->fp32_sse4.magic_bias_less_output_zero_point[0];

  float vfpacc = (float) (_mm_cvtsi128_si32(vacc_lo) + vinit_bias) * vscale;
  vfpacc += vmagic_bias;
  int32_t vout = (int32_t) float_as_uint32(vfpacc);
  vout -= vmagic_bias_less_output_zero_point;
  vout = math_max_s32(vout, output_min);
  vout = math_min_s32(vout, output_max);
  *output += (int8_t) vout;
}
