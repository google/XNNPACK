// Auto-generated file. Do not edit!
//   Template: src/f32-qs8-vcvt/scalar-lrintf.c.in
//   Generator: tools/xngen
//
// Copyright 2021 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <math.h>

#include "xnnpack/common.h"
#include "xnnpack/math.h"
#include "xnnpack/vcvt.h"


void xnn_f32_qs8_vcvt_ukernel__scalar_lrintf_u4(
    size_t batch,
    const float* input,
    int8_t* output,
    const union xnn_f32_qs8_cvt_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const float vscale = params->scalar.scale;
  const float voutput_min_less_zero_point = (float) ((int32_t) params->scalar.output_min - (int32_t) params->scalar.output_zero_point);
  const float voutput_max_less_zero_point = (float) ((int32_t) params->scalar.output_max - (int32_t) params->scalar.output_zero_point);
  const int32_t voutput_zero_point = params->scalar.output_zero_point;

  for (; batch >= 4 * sizeof(float); batch -= 4 * sizeof(float)) {
    float vx0 = input[0];
    float vx1 = input[1];
    float vx2 = input[2];
    float vx3 = input[3];
    input += 4;

    vx0 *= vscale;
    vx1 *= vscale;
    vx2 *= vscale;
    vx3 *= vscale;

    vx0 = math_max_f32(vx0, voutput_min_less_zero_point);
    vx1 = math_max_f32(vx1, voutput_min_less_zero_point);
    vx2 = math_max_f32(vx2, voutput_min_less_zero_point);
    vx3 = math_max_f32(vx3, voutput_min_less_zero_point);

    vx0 = math_min_f32(vx0, voutput_max_less_zero_point);
    vx1 = math_min_f32(vx1, voutput_max_less_zero_point);
    vx2 = math_min_f32(vx2, voutput_max_less_zero_point);
    vx3 = math_min_f32(vx3, voutput_max_less_zero_point);

    int32_t vy0 = (int32_t) lrintf(vx0);
    int32_t vy1 = (int32_t) lrintf(vx1);
    int32_t vy2 = (int32_t) lrintf(vx2);
    int32_t vy3 = (int32_t) lrintf(vx3);

    vy0 += voutput_zero_point;
    vy1 += voutput_zero_point;
    vy2 += voutput_zero_point;
    vy3 += voutput_zero_point;

    output[0] = (int8_t) vy0;
    output[1] = (int8_t) vy1;
    output[2] = (int8_t) vy2;
    output[3] = (int8_t) vy3;
    output += 4;
  }
  if XNN_UNLIKELY(batch != 0) {
    do {
      float vx = *input++;
      vx *= vscale;
      vx = math_max_f32(vx, voutput_min_less_zero_point);
      vx = math_min_f32(vx, voutput_max_less_zero_point);

      int32_t vy = (int32_t) lrintf(vx);
      vy += voutput_zero_point;

      *output++ = (int8_t) vy;

      batch -= sizeof(float);
    } while (batch != 0);
  }
}
