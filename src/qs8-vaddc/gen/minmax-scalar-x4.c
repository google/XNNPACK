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
  const int32_t vzero_point_product =
    params->scalar.zero_point_product + (int32_t) *input_b * params->scalar.y_multiplier;
  const int32_t va_multiplier = params->scalar.x_multiplier;
  const uint32_t vshift = params->scalar.shift;
  const int32_t vremainder_mask = params->scalar.remainder_mask;
  const int32_t vremainder_threshold = params->scalar.remainder_threshold;
  const int32_t voutput_zero_point = params->scalar.output_zero_point;
  const int32_t voutput_min = params->scalar.output_min;
  const int32_t voutput_max = params->scalar.output_max;

  for (; n >= 4 * sizeof(int8_t); n -= 4 * sizeof(int8_t)) {
    const int32_t va0 = input_a[0];
    const int32_t va1 = input_a[1];
    const int32_t va2 = input_a[2];
    const int32_t va3 = input_a[3];
    input_a += 4;

    const int32_t vacc0 = vzero_point_product + va0 * va_multiplier;
    const int32_t vacc1 = vzero_point_product + va1 * va_multiplier;
    const int32_t vacc2 = vzero_point_product + va2 * va_multiplier;
    const int32_t vacc3 = vzero_point_product + va3 * va_multiplier;
    input_b += 4;

    const int32_t vrem0 = (vacc0 & vremainder_mask) - (int32_t) (vacc0 < 0);
    const int32_t vrem1 = (vacc1 & vremainder_mask) - (int32_t) (vacc1 < 0);
    const int32_t vrem2 = (vacc2 & vremainder_mask) - (int32_t) (vacc2 < 0);
    const int32_t vrem3 = (vacc3 & vremainder_mask) - (int32_t) (vacc3 < 0);

    int32_t vout0 = asr_s32(vacc0, vshift) + (int32_t) (vrem0 > vremainder_threshold);
    int32_t vout1 = asr_s32(vacc1, vshift) + (int32_t) (vrem1 > vremainder_threshold);
    int32_t vout2 = asr_s32(vacc2, vshift) + (int32_t) (vrem2 > vremainder_threshold);
    int32_t vout3 = asr_s32(vacc3, vshift) + (int32_t) (vrem3 > vremainder_threshold);

    vout0 += voutput_zero_point;
    vout1 += voutput_zero_point;
    vout2 += voutput_zero_point;
    vout3 += voutput_zero_point;

    vout0 = math_max_s32(vout0, voutput_min);
    vout1 = math_max_s32(vout1, voutput_min);
    vout2 = math_max_s32(vout2, voutput_min);
    vout3 = math_max_s32(vout3, voutput_min);

    vout0 = math_min_s32(vout0, voutput_max);
    vout1 = math_min_s32(vout1, voutput_max);
    vout2 = math_min_s32(vout2, voutput_max);
    vout3 = math_min_s32(vout3, voutput_max);

    output[0] = vout0;
    output[1] = vout1;
    output[2] = vout2;
    output[3] = vout3;
    output += 4;
  }
  if XNN_UNLIKELY(n != 0) {
    do {
      const int32_t va = *input_a++;
      const int32_t vacc = vzero_point_product + va * va_multiplier;

      const int32_t vrem = (vacc & vremainder_mask) - (int32_t) (vacc < 0);
      int32_t vout = asr_s32(vacc, vshift) + (int32_t) (vrem > vremainder_threshold);
      vout += voutput_zero_point;
      vout = math_max_s32(vout, voutput_min);
      vout = math_min_s32(vout, voutput_max);
      *output++ = vout;

      n -= sizeof(int8_t);
    } while (n != 0);
  }
}
