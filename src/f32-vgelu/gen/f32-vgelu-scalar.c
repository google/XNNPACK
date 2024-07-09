// Auto-generated file. Do not edit!
//   Template: src/f32-vgelu/scalar.c.in
//   Generator: tools/xngen
//
// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <math.h>

#include "xnnpack/common.h"
#include "xnnpack/vunary.h"

#ifndef M_SQRT1_2
#define M_SQRT1_2 0.7071067811865475244
#endif

void xnn_f32_vgelu_ukernel__scalar_u1(
    size_t batch,
    const float* input,
    float* output,
    const union xnn_f32_default_params unused_params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  for (; batch >= sizeof(float); batch -= sizeof(float)) {
    const float vx = *input++;
    const float vy = vx * 0.5f * (1.0f + erff(vx * M_SQRT1_2));
    *output++ = vy;
  }
}
void xnn_f32_vgelu_ukernel__scalar_u2(
    size_t batch,
    const float* input,
    float* output,
    const union xnn_f32_default_params unused_params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  for (; batch >= 2 * sizeof(float); batch -= 2 * sizeof(float)) {
    const float vx_0 = input[0];
    const float vx_1 = input[1];
    input += 2;

    float vy_0 = erff(vx_0 * M_SQRT1_2);
    float vy_1 = erff(vx_1 * M_SQRT1_2);
    vy_0 = 1.0f + vy_0;
    vy_1 = 1.0f + vy_1;
    vy_0 = vx_0 * 0.5f * vy_0;
    vy_1 = vx_1 * 0.5f * vy_1;

    output[0] = vy_0;
    output[1] = vy_1;
    output += 2;
  }
  if XNN_UNLIKELY(batch != 0) {
    const float vx = *input;
    const float vy = vx * 0.5f * (1.0f + erff(vx * M_SQRT1_2));
    *output = vy;
  }
}
void xnn_f32_vgelu_ukernel__scalar_u4(
    size_t batch,
    const float* input,
    float* output,
    const union xnn_f32_default_params unused_params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  for (; batch >= 4 * sizeof(float); batch -= 4 * sizeof(float)) {
    const float vx_0 = input[0];
    const float vx_1 = input[1];
    const float vx_2 = input[2];
    const float vx_3 = input[3];
    input += 4;

    float vy_0 = erff(vx_0 * M_SQRT1_2);
    float vy_1 = erff(vx_1 * M_SQRT1_2);
    float vy_2 = erff(vx_2 * M_SQRT1_2);
    float vy_3 = erff(vx_3 * M_SQRT1_2);
    vy_0 = 1.0f + vy_0;
    vy_1 = 1.0f + vy_1;
    vy_2 = 1.0f + vy_2;
    vy_3 = 1.0f + vy_3;
    vy_0 = vx_0 * 0.5f * vy_0;
    vy_1 = vx_1 * 0.5f * vy_1;
    vy_2 = vx_2 * 0.5f * vy_2;
    vy_3 = vx_3 * 0.5f * vy_3;

    output[0] = vy_0;
    output[1] = vy_1;
    output[2] = vy_2;
    output[3] = vy_3;
    output += 4;
  }
  if XNN_UNLIKELY(batch != 0) {
    do {
      const float vx = *input++;
      const float vy = vx * 0.5f * (1.0f + erff(vx * M_SQRT1_2));
      *output++ = vy;
      batch -= sizeof(float);
    } while (batch != 0);
  }
}
