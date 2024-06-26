// Auto-generated file. Do not edit!
//   Template: src/qs8-vhswish/scalar.c.in
//   Generator: tools/xngen
//
// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include "xnnpack/math.h"
#include "xnnpack/vhswish.h"


void xnn_qs8_vhswish_ukernel__scalar_u1(
    size_t batch,
    const int8_t* input,
    int8_t* output,
    const union xnn_qs8_hswish_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(int8_t) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const uint32_t vinput_zero_point = (uint32_t) params->scalar.input_zero_point;
  const int32_t voutput_zero_point = params->scalar.output_zero_point;
  const int32_t vinput_scale_div_mantissa = params->scalar.input_scale_div_mantissa;
  const int32_t vinput_scale_div_exp = params->scalar.input_scale_div_exp;
  const int32_t vscale_ratio = params->scalar.scale_ratio;
  do {
    const int32_t vacc = (int32_t) ((vinput_zero_point - (uint32_t) *input++) << 7);
    int32_t vin = vacc * vinput_scale_div_mantissa;
    if (vinput_scale_div_exp > 0) {
      vin <<= vinput_scale_div_exp;
    } else {
      vin >>= -vinput_scale_div_exp;
    }
    vin -= 16384;
    vin = math_min_s32(vin, 0);
    vin = math_max_s32(vin, -32768);

    int32_t vout = math_asr_s32(vacc * vscale_ratio + INT32_C(0x4000), 15);
    vout = math_asr_s32(vin * vout + INT32_C(0x4000), 15) + voutput_zero_point;
    vout = math_max_s32(vout, -128);
    vout = math_min_s32(vout, 127);
    *output++ = (int8_t) vout;

    batch -= sizeof(int8_t);
  } while (batch != 0);
}
