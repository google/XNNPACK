// Auto-generated file. Do not edit!
//   Template: src/qs8-vaddc/scalar.c.in
//   Generator: tools/xngen
//
// Copyright 2021 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <xnnpack/math.h>
#include <xnnpack/vadd.h>


void xnn_qs8_vaddc_minmax_ukernel__scalar_x4(
    size_t n,
    const int8_t* input_a,
    const int8_t* input_b,
    int8_t* output,
    const union xnn_qs8_add_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_DISABLE_TSAN
{
  const int32_t vbias = params->scalar.bias + (int32_t) *input_b * params->scalar.b_multiplier;
  const int32_t va_multiplier = params->scalar.a_multiplier;
  const int32_t vrounding = params->scalar.rounding;
  const uint32_t vshift = params->scalar.shift;
  const int32_t voutput_min_less_zero_point = params->scalar.output_min_less_zero_point;
  const int32_t voutput_max_less_zero_point = params->scalar.output_max_less_zero_point;
  const int32_t voutput_zero_point = params->scalar.output_zero_point;

  for (; n >= 4 * sizeof(int8_t); n -= 4 * sizeof(int8_t)) {
    const int32_t va0 = input_a[0];
    const int32_t va1 = input_a[1];
    const int32_t va2 = input_a[2];
    const int32_t va3 = input_a[3];
    input_a += 4;

    const int32_t vacc0 = vbias + va0 * va_multiplier;
    const int32_t vacc1 = vbias + va1 * va_multiplier;
    const int32_t vacc2 = vbias + va2 * va_multiplier;
    const int32_t vacc3 = vbias + va3 * va_multiplier;
    input_b += 4;

    int32_t vout0 = asr_s32(vacc0 + vrounding, vshift);
    int32_t vout1 = asr_s32(vacc1 + vrounding, vshift);
    int32_t vout2 = asr_s32(vacc2 + vrounding, vshift);
    int32_t vout3 = asr_s32(vacc3 + vrounding, vshift);

    vout0 = math_max_s32(vout0, voutput_min_less_zero_point);
    vout1 = math_max_s32(vout1, voutput_min_less_zero_point);
    vout2 = math_max_s32(vout2, voutput_min_less_zero_point);
    vout3 = math_max_s32(vout3, voutput_min_less_zero_point);

    vout0 = math_min_s32(vout0, voutput_max_less_zero_point);
    vout1 = math_min_s32(vout1, voutput_max_less_zero_point);
    vout2 = math_min_s32(vout2, voutput_max_less_zero_point);
    vout3 = math_min_s32(vout3, voutput_max_less_zero_point);

    vout0 += voutput_zero_point;
    vout1 += voutput_zero_point;
    vout2 += voutput_zero_point;
    vout3 += voutput_zero_point;

    output[0] = (int8_t) vout0;
    output[1] = (int8_t) vout1;
    output[2] = (int8_t) vout2;
    output[3] = (int8_t) vout3;
    output += 4;
  }
  if XNN_UNLIKELY(n != 0) {
    do {
      const int32_t va = *input_a++;
      const int32_t vacc = vbias + va * va_multiplier;

      int32_t vout = asr_s32(vacc + vrounding, vshift);
      vout = math_max_s32(vout, voutput_min_less_zero_point);
      vout = math_min_s32(vout, voutput_max_less_zero_point);
      *output++ = (int8_t) (vout + voutput_zero_point);

      n -= sizeof(int8_t);
    } while (n != 0);
  }
}
