// Auto-generated file. Do not edit!
//   Template: src/qs8-rsum/avx512skx.c.in
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

void xnn_qs8_rsum_minmax_fp32_ukernel__avx512skx_u128(
    size_t batch,
    const int8_t* input,
    int8_t* output,
    const union xnn_qs8_avgpool_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(input != NULL);
  assert(output != NULL);
  assert(params != NULL);

  __m512i vacc0 = _mm512_setzero_si512();
  // 256 int8s may be summed into an int16 before overflowing.
  // There are 32 lanes in the accumulator register and 1 registers.
  int num_batches = (batch + 8191) >> 13;
  const __m512i vone = _mm512_set1_epi8(1);
  for (; num_batches > 0; --num_batches) {
    __m512i vacc16_0 = _mm512_setzero_si512();
    for (int current_batch = min(batch, 8192); current_batch >= 128; current_batch -= 128) {
      const __m512i vt0 = _mm512_maddubs_epi16(vone, _mm512_loadu_si512((const __m512i*) input)); input += 64;
      const __m512i vt1 = _mm512_maddubs_epi16(vone, _mm512_loadu_si512((const __m512i*) input)); input += 64;

      vacc16_0 = _mm512_add_epi16(vacc16_0, vt0);
      vacc16_0 = _mm512_add_epi16(vacc16_0, vt1);
    }
    __m512i left0 = _mm512_cvtepi16_epi32(_mm512_castsi512_si256(vacc16_0));
    __m512i right0 = _mm512_cvtepi16_epi32(_mm512_extracti32x8_epi32(vacc16_0, 1));
    vacc0 = _mm512_add_epi32(vacc0, _mm512_add_epi32(left0, right0));
    batch = (batch >= 8192 ? (batch - 8192) : batch & 127);
  }
  if (XNN_UNLIKELY(batch != 0)) {
    __m512i vacc16 = _mm512_setzero_si512();
    for (; batch >= 64; batch -= 64) {
      const __m512i vt = _mm512_maddubs_epi16(vone, _mm512_loadu_epi8((const __m512i*) input)); input += 64;
      vacc16 = _mm512_add_epi16(vacc16, vt);
    }
    if (XNN_UNLIKELY(batch != 0)) {
      const __mmask64 vmask = _cvtu64_mask64((uint64_t) ((UINT64_C(1) << (batch & 63)) - UINT64_C(1)));
      const __m512i vt = _mm512_maddubs_epi16(vone, _mm512_maskz_loadu_epi8(vmask, (const __m512i*) input));
      vacc16 = _mm512_add_epi16(vacc16, vt);
    }
    __m512i left = _mm512_cvtepi16_epi32(_mm512_castsi512_si256(vacc16));
    __m512i right = _mm512_cvtepi16_epi32(_mm512_extracti32x8_epi32(vacc16, 1));
    vacc0 = _mm512_add_epi32(vacc0, _mm512_add_epi32(left, right));
  }

  int32_t res = _mm512_reduce_add_epi32(vacc0);

  const int32_t vinit_bias = params->fp32_avx2.init_bias[0];
  const float vscale = params->fp32_avx2.scale[0];
  const int32_t output_min = params->fp32_avx2.output_min[0];
  const int32_t output_max = params->fp32_avx2.output_max[0];
  const float vmagic_bias = params->fp32_avx2.magic_bias[0];
  const int32_t vmagic_bias_less_output_zero_point = params->fp32_avx2.magic_bias_less_output_zero_point[0];

  float vfpacc = (float) (res + vinit_bias) * vscale;
  vfpacc += vmagic_bias;
  int32_t vout = (int32_t) float_as_uint32(vfpacc);
  vout -= vmagic_bias_less_output_zero_point;
  vout = math_max_s32(vout, output_min);
  vout = math_min_s32(vout, output_max);
  *output += (int8_t) vout;
}
