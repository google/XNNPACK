// Auto-generated file. Do not edit!
//   Template: src/qs8-vlrelu/scalar-andxor.c.in
//   Generator: tools/xngen
//
// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include "xnnpack/math.h"
#include "xnnpack/vlrelu.h"


void xnn_qu8_vlrelu_ukernel__scalar_andxor_u2(
    size_t batch,
    const uint8_t* input,
    uint8_t* output,
    const union xnn_qu8_lrelu_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(uint8_t) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const int32_t vinput_zero_point = params->scalar.input_zero_point;
  const int32_t vmultiplier_diff = params->scalar.negative_multiplier ^ params->scalar.positive_multiplier;
  const int32_t vmultiplier_base = params->scalar.positive_multiplier;
  const int32_t vbias = (params->scalar.output_zero_point << 8) + 128;
  for (; batch >= 2 * sizeof(uint8_t); batch -= 2 * sizeof(uint8_t)) {
    int32_t vacc0 = (int32_t) input[0];
    int32_t vacc1 = (int32_t) input[1];
    input += 2;

    vacc0 -= vinput_zero_point;
    vacc1 -= vinput_zero_point;

    int32_t vmultiplier0 = math_asr_s32(vacc0, 31);
    int32_t vmultiplier1 = math_asr_s32(vacc1, 31);

    vmultiplier0 &= vmultiplier_diff;
    vmultiplier1 &= vmultiplier_diff;

    vmultiplier0 ^= vmultiplier_base;
    vmultiplier1 ^= vmultiplier_base;

    vacc0 = vbias + vacc0 * vmultiplier0;
    vacc1 = vbias + vacc1 * vmultiplier1;

    int32_t vout0 = math_asr_s32(vacc0, 8);
    int32_t vout1 = math_asr_s32(vacc1, 8);

    vout0 = math_max_s32(vout0, 0);
    vout1 = math_max_s32(vout1, 0);

    vout0 = math_min_s32(vout0, 255);
    vout1 = math_min_s32(vout1, 255);

    output[0] = (uint8_t) vout0;
    output[1] = (uint8_t) vout1;
    output += 2;
  }
  if XNN_UNLIKELY(batch != 0) {
    int32_t vacc = (int32_t) *input++ - vinput_zero_point;
    const int32_t vmultiplier = vmultiplier_base ^ (vmultiplier_diff & math_asr_s32(vacc, 31));
    vacc = vbias + vacc * vmultiplier;

    int32_t vout = math_asr_s32(vacc, 8);
    vout = math_max_s32(vout, 0);
    vout = math_min_s32(vout, 255);
    *output = (uint8_t) vout;
  }
}
