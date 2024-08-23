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


void xnn_qs8_vadd_minmax_ukernel__scalar_u2(
    size_t batch,
    const int8_t* input_a,
    const int8_t* input_b,
    int8_t* output,
    const union xnn_qs8_add_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(int8_t) == 0);
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

  for (; batch >= 2 * sizeof(int8_t); batch -= 2 * sizeof(int8_t)) {
    const int32_t va0 = input_a[0];
    const int32_t va1 = input_a[1];
    input_a += 2;

    const int32_t vb0 = input_b[0];
    int32_t vacc0 = vbias + va0 * va_multiplier;
    const int32_t vb1 = input_b[1];
    int32_t vacc1 = vbias + va1 * va_multiplier;
    input_b += 2;

    vacc0 += vb0 * vb_multiplier;
    vacc1 += vb1 * vb_multiplier;

    int32_t vout0 = math_asr_s32(vacc0, vshift);
    int32_t vout1 = math_asr_s32(vacc1, vshift);

    vout0 += voutput_zero_point;
    vout1 += voutput_zero_point;

    vout0 = math_max_s32(vout0, voutput_min);
    vout1 = math_max_s32(vout1, voutput_min);

    vout0 = math_min_s32(vout0, voutput_max);
    vout1 = math_min_s32(vout1, voutput_max);

    output[0] = (int8_t) vout0;
    output[1] = (int8_t) vout1;
    output += 2;
  }
  if XNN_UNLIKELY(batch != 0) {
    const int32_t va = *input_a;
    const int32_t vb = *input_b;
    const int32_t vacc = vbias + va * va_multiplier + vb * vb_multiplier;

    int32_t vout = math_asr_s32(vacc, vshift);
    vout += voutput_zero_point;
    vout = math_max_s32(vout, voutput_min);
    vout = math_min_s32(vout, voutput_max);
    *output++ = (int8_t) vout;
  }
}
