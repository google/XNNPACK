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


void xnn_f32_qs8_vcvt_ukernel__scalar_lrintf_u1(
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
