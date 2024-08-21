// Auto-generated file. Do not edit!
//   Template: src/f32-qs8-vcvt/scalar-fmagic.c.in
//   Generator: tools/xngen
//
// Copyright 2021 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include "xnnpack/common.h"
#include "xnnpack/math.h"
#include "xnnpack/vcvt.h"
#include <fp16/fp16.h>

void xnn_f16_qs8_vcvt_ukernel__scalar_fmagic_u3(
    size_t batch,
    const void* input,
    int8_t* output,
    const union xnn_f16_qs8_cvt_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(uint16_t) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const uint16_t* i = (const uint16_t*) input;
  const float vscale = params->scalar.scale;
  const float voutput_min_less_zero_point = (float) ((int32_t) params->scalar.output_min - (int32_t) params->scalar.output_zero_point);
  const float voutput_max_less_zero_point = (float) ((int32_t) params->scalar.output_max - (int32_t) params->scalar.output_zero_point);
  const float vmagic_bias = 12582912.0f;
  const int32_t vmagic_bias_less_zero_point = INT32_C(0x4B400000) - (int32_t) params->scalar.output_zero_point;

  for (; batch >= 3 * sizeof(uint16_t); batch -= 3 * sizeof(uint16_t)) {
    float vx0 = fp16_ieee_to_fp32_value(i[0]);
    float vx1 = fp16_ieee_to_fp32_value(i[1]);
    float vx2 = fp16_ieee_to_fp32_value(i[2]);
    i += 3;

    vx0 *= vscale;
    vx1 *= vscale;
    vx2 *= vscale;

    vx0 = math_max_f32(vx0, voutput_min_less_zero_point);
    vx1 = math_max_f32(vx1, voutput_min_less_zero_point);
    vx2 = math_max_f32(vx2, voutput_min_less_zero_point);

    vx0 = math_min_f32(vx0, voutput_max_less_zero_point);
    vx1 = math_min_f32(vx1, voutput_max_less_zero_point);
    vx2 = math_min_f32(vx2, voutput_max_less_zero_point);

    vx0 += vmagic_bias;
    vx1 += vmagic_bias;
    vx2 += vmagic_bias;

    int32_t vy0 = (int32_t) float_as_uint32(vx0);
    int32_t vy1 = (int32_t) float_as_uint32(vx1);
    int32_t vy2 = (int32_t) float_as_uint32(vx2);

    vy0 -= vmagic_bias_less_zero_point;
    vy1 -= vmagic_bias_less_zero_point;
    vy2 -= vmagic_bias_less_zero_point;

    output[0] = (int8_t) vy0;
    output[1] = (int8_t) vy1;
    output[2] = (int8_t) vy2;
    output += 3;
  }
  if XNN_UNLIKELY(batch != 0) {
    do {
      float vx = fp16_ieee_to_fp32_value(*i++);
      vx *= vscale;
      vx = math_max_f32(vx, voutput_min_less_zero_point);
      vx = math_min_f32(vx, voutput_max_less_zero_point);
      vx += vmagic_bias;

      int32_t vy = (int32_t) float_as_uint32(vx);
      vy -= vmagic_bias_less_zero_point;

      *output++ = (int8_t) vy;

      batch -= sizeof(uint16_t);
    } while (batch != 0);
  }
}
