// Auto-generated file. Do not edit!
//   Template: src/qs8-vcvt/scalar.c.in
//   Generator: tools/xngen
//
// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include "xnnpack/math.h"
#include "xnnpack/vcvt.h"


void xnn_qs8_vcvt_ukernel__scalar_u2(
    size_t batch,
    const int8_t* input,
    int8_t* output,
    const union xnn_qs8_cvt_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(int8_t) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const int32_t vbias = 
      ((int32_t) params->scalar.output_zero_point << 8) -
      (int32_t) params->scalar.multiplier * (int32_t) params->scalar.input_zero_point + 
      INT32_C(0x80);
  const int32_t vmultiplier = params->scalar.multiplier;
  for (; batch >= 2 * sizeof(int8_t); batch -= 2 * sizeof(int8_t)) {
    int32_t vacc0 = input[0];
    int32_t vacc1 = input[1];
    input += 2;

    vacc0 = vbias + vacc0 * vmultiplier;
    vacc1 = vbias + vacc1 * vmultiplier;

    int32_t vout0 = math_asr_s32(vacc0, 8);
    int32_t vout1 = math_asr_s32(vacc1, 8);

    vout0 = math_max_s32(vout0, -128);
    vout1 = math_max_s32(vout1, -128);

    vout0 = math_min_s32(vout0, 127);
    vout1 = math_min_s32(vout1, 127);

    output[0] = (int8_t) vout0;
    output[1] = (int8_t) vout1;
    output += 2;
  }
  if XNN_UNLIKELY(batch != 0) {
    int32_t vacc = *input;
    vacc = vbias + vacc * vmultiplier;

    int32_t vout = math_asr_s32(vacc, 8);
    vout = math_max_s32(vout, -128);
    vout = math_min_s32(vout, 127);
    *output = (int8_t) vout;
  }
}
