// Auto-generated file. Do not edit!
//   Template: src/qs8-vadd/scalar.c.in
//   Generator: tools/xngen
//
// Copyright 2021 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include "xnnpack/math.h"
#include "xnnpack/vbinary.h"


void xnn_qu8_vadd_minmax_ukernel__scalar_u4(
    size_t batch,
    const uint8_t* input_a,
    const uint8_t* input_b,
    uint8_t* output,
    const union xnn_qu8_add_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(uint8_t) == 0);
  assert(input_a != NULL);
  assert(input_b != NULL);
  assert(output != NULL);

  const int32_t vbias = params->scalar.bias;
  const int32_t va_multiplier = params->scalar.a_multiplier;
  const int32_t vb_multiplier = params->scalar.b_multiplier;
  const uint32_t vshift = params->scalar.shift;
  const int32_t voutput_min = params->scalar.output_min;
  const int32_t voutput_max = params->scalar.output_max;
  const int32_t voutput_zero_point = params->scalar.output_zero_point;

  for (; batch >= 4 * sizeof(uint8_t); batch -= 4 * sizeof(uint8_t)) {
    const int32_t va0 = input_a[0];
    const int32_t va1 = input_a[1];
    const int32_t va2 = input_a[2];
    const int32_t va3 = input_a[3];
    input_a += 4;

    const int32_t vb0 = input_b[0];
    int32_t vacc0 = vbias + va0 * va_multiplier;
    const int32_t vb1 = input_b[1];
    int32_t vacc1 = vbias + va1 * va_multiplier;
    const int32_t vb2 = input_b[2];
    int32_t vacc2 = vbias + va2 * va_multiplier;
    const int32_t vb3 = input_b[3];
    int32_t vacc3 = vbias + va3 * va_multiplier;
    input_b += 4;

    vacc0 += vb0 * vb_multiplier;
    vacc1 += vb1 * vb_multiplier;
    vacc2 += vb2 * vb_multiplier;
    vacc3 += vb3 * vb_multiplier;

    int32_t vout0 = math_asr_s32(vacc0, vshift);
    int32_t vout1 = math_asr_s32(vacc1, vshift);
    int32_t vout2 = math_asr_s32(vacc2, vshift);
    int32_t vout3 = math_asr_s32(vacc3, vshift);

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

    output[0] = (uint8_t) vout0;
    output[1] = (uint8_t) vout1;
    output[2] = (uint8_t) vout2;
    output[3] = (uint8_t) vout3;
    output += 4;
  }
  if XNN_UNLIKELY(batch != 0) {
    do {
      const int32_t va = *input_a++;
      const int32_t vb = *input_b++;
      const int32_t vacc = vbias + va * va_multiplier + vb * vb_multiplier;

      int32_t vout = math_asr_s32(vacc, vshift);
      vout += voutput_zero_point;
      vout = math_max_s32(vout, voutput_min);
      vout = math_min_s32(vout, voutput_max);
      *output++ = (uint8_t) vout;

      batch -= sizeof(uint8_t);
    } while (batch != 0);
  }
}
