// Auto-generated file. Do not edit!
//   Template: src/qs8-rsum/avx512vnni.c.in
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

void xnn_qs8_rsum_minmax_fp32_ukernel__avx512vnni_u64(
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
  const __m512i vone = _mm512_set1_epi8(1);
  for (; batch >= 64; batch -= 64) {
    vacc0 = _mm512_dpbusd_epi32(vacc0, vone, _mm512_loadu_si512((const __m512i*) input)); input += 64;
  }
  if (XNN_UNLIKELY(batch != 0)) {
    for (; batch >= 64; batch -= 64) {
      vacc0 = _mm512_dpbusd_epi32(vacc0, vone, _mm512_loadu_si512((const __m512i*) input)); input += 64;
    }
    if (XNN_UNLIKELY(batch != 0)) {
      const __mmask64 vmask = _cvtu64_mask64((uint64_t) ((UINT64_C(1) << (batch & 63)) - UINT64_C(1)));
      vacc0 = _mm512_dpbusd_epi32(vacc0, vone, _mm512_maskz_loadu_epi8(vmask, (const __m512i*) input)); input += 64;
    }
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
