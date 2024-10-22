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
#include "xnnpack/vunary.h"


void xnn_qu8_vhswish_ukernel__scalar_u4(
    size_t batch,
    const uint8_t* input,
    uint8_t* output,
    const union xnn_qu8_hswish_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(uint8_t) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const uint32_t vinput_zero_point = (uint32_t) params->scalar.input_zero_point;
  const int32_t voutput_zero_point = params->scalar.output_zero_point;
  const int32_t vinput_scale_div_mantissa = params->scalar.input_scale_div_mantissa;
  const int32_t vinput_scale_div_exp = params->scalar.input_scale_div_exp;
  const int32_t vscale_ratio = params->scalar.scale_ratio;
  for (; batch >= 4 * sizeof(uint8_t); batch -= 4 * sizeof(uint8_t)) {
    int32_t vacc0 = (int32_t) ((vinput_zero_point - (uint32_t) input[0]) << 7);
    int32_t vacc1 = (int32_t) ((vinput_zero_point - (uint32_t) input[1]) << 7);
    int32_t vacc2 = (int32_t) ((vinput_zero_point - (uint32_t) input[2]) << 7);
    int32_t vacc3 = (int32_t) ((vinput_zero_point - (uint32_t) input[3]) << 7);
    input += 4;

    int32_t vin0 = vacc0 * vinput_scale_div_mantissa;
    int32_t vin1 = vacc1 * vinput_scale_div_mantissa;
    int32_t vin2 = vacc2 * vinput_scale_div_mantissa;
    int32_t vin3 = vacc3 * vinput_scale_div_mantissa;

    if (vinput_scale_div_exp > 0) {
      vin0 <<= vinput_scale_div_exp;
      vin1 <<= vinput_scale_div_exp;
      vin2 <<= vinput_scale_div_exp;
      vin3 <<= vinput_scale_div_exp;
    } else {
      vin0 >>= -vinput_scale_div_exp;
      vin1 >>= -vinput_scale_div_exp;
      vin2 >>= -vinput_scale_div_exp;
      vin3 >>= -vinput_scale_div_exp;
    }

    vin0 -= 16384;
    vin1 -= 16384;
    vin2 -= 16384;
    vin3 -= 16384;

    vin0 = math_min_s32(vin0, 0);
    vin1 = math_min_s32(vin1, 0);
    vin2 = math_min_s32(vin2, 0);
    vin3 = math_min_s32(vin3, 0);

    vin0 = math_max_s32(vin0, -32768);
    vin1 = math_max_s32(vin1, -32768);
    vin2 = math_max_s32(vin2, -32768);
    vin3 = math_max_s32(vin3, -32768);

    int32_t vout0 = math_asr_s32(vacc0 * vscale_ratio + INT32_C(0x4000), 15);
    int32_t vout1 = math_asr_s32(vacc1 * vscale_ratio + INT32_C(0x4000), 15);
    int32_t vout2 = math_asr_s32(vacc2 * vscale_ratio + INT32_C(0x4000), 15);
    int32_t vout3 = math_asr_s32(vacc3 * vscale_ratio + INT32_C(0x4000), 15);

    vout0 = math_asr_s32(vin0 * vout0 + INT32_C(0x4000), 15) + voutput_zero_point;
    vout1 = math_asr_s32(vin1 * vout1 + INT32_C(0x4000), 15) + voutput_zero_point;
    vout2 = math_asr_s32(vin2 * vout2 + INT32_C(0x4000), 15) + voutput_zero_point;
    vout3 = math_asr_s32(vin3 * vout3 + INT32_C(0x4000), 15) + voutput_zero_point;

    vout0 = math_max_s32(vout0, 0);
    vout1 = math_max_s32(vout1, 0);
    vout2 = math_max_s32(vout2, 0);
    vout3 = math_max_s32(vout3, 0);

    vout0 = math_min_s32(vout0, 255);
    vout1 = math_min_s32(vout1, 255);
    vout2 = math_min_s32(vout2, 255);
    vout3 = math_min_s32(vout3, 255);

    output[0] = (uint8_t) vout0;
    output[1] = (uint8_t) vout1;
    output[2] = (uint8_t) vout2;
    output[3] = (uint8_t) vout3;
    output += 4;
  }
  if XNN_UNLIKELY(batch != 0) {
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
      vout = math_asr_s32(vin * vout, 15) + voutput_zero_point;
      vout = math_max_s32(vout, 0);
      vout = math_min_s32(vout, 255);
      *output++ = (uint8_t) vout;

      batch -= sizeof(uint8_t);
    } while (batch != 0);
  }
}
