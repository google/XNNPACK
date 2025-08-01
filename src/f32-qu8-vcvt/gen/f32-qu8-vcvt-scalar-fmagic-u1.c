// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/f32-qs8-vcvt/scalar-fmagic.c.in
//   Generator: tools/xngen
//
// Copyright 2021 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include "src/xnnpack/math.h"
#include "src/xnnpack/microparams.h"
#include "src/xnnpack/vcvt.h"

void xnn_f32_qu8_vcvt_ukernel__scalar_fmagic_u1(
    size_t batch,
    const float* input,
    uint8_t* output,
    const struct xnn_f32_qu8_cvt_params* restrict params)
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const float* i = input;
  const float vscale = params->scalar.scale;
  const float voutput_min_less_zero_point = (float) ((int32_t) 0 - (int32_t) params->scalar.output_zero_point);
  const float voutput_max_less_zero_point = (float) ((int32_t) 255 - (int32_t) params->scalar.output_zero_point);
  const float vmagic_bias = 12582912.0f;
  const int32_t vmagic_bias_less_zero_point = INT32_C(0x4B400000) - (int32_t) params->scalar.output_zero_point;

  do {
    float vx = *i++;
    vx *= vscale;
    vx = math_max_f32(vx, voutput_min_less_zero_point);
    vx = math_min_f32(vx, voutput_max_less_zero_point);
    vx += vmagic_bias;

    int32_t vy = (int32_t) float_as_uint32(vx);
    vy -= vmagic_bias_less_zero_point;

    *output++ = (uint8_t) vy;

    batch -= sizeof(float);
  } while (batch != 0);
}
