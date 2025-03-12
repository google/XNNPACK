// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/f32-vapproxgelu/scalar.c.in
//   Generator: tools/xngen
//
// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <math.h>

#include "src/xnnpack/common.h"
#include "src/xnnpack/vunary.h"

#ifndef M_SQRT_2_DIV_PI
#define M_SQRT_2_DIV_PI 0.7978845608028654
#endif

void xnn_f32_vapproxgelu_ukernel__scalar_u1(
    size_t batch,
    const float* input,
    float* output,
    const struct xnn_f32_default_params unused_params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  for (; batch >= sizeof(float); batch -= sizeof(float)) {
    const float vx = *input++;
    const float vy = vx * 0.5f * (1.0f + tanhf(M_SQRT_2_DIV_PI * vx * (1.0f + 0.044715f * vx * vx)));
    *output++ = vy;
  }
}
void xnn_f32_vapproxgelu_ukernel__scalar_u2(
    size_t batch,
    const float* input,
    float* output,
    const struct xnn_f32_default_params unused_params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  for (; batch >= 2 * sizeof(float); batch -= 2 * sizeof(float)) {
    const float vx_0 = input[0];
    const float vx_1 = input[1];
    input += 2;

    float vy_0 = tanhf(M_SQRT_2_DIV_PI * vx_0 * (1.0f + 0.044715f * vx_0 * vx_0));
    float vy_1 = tanhf(M_SQRT_2_DIV_PI * vx_1 * (1.0f + 0.044715f * vx_1 * vx_1));
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
    const float vy = vx * 0.5f * (1.0f + tanhf(M_SQRT_2_DIV_PI * vx * (1.0f + 0.044715f * vx * vx)));
    *output = vy;
  }
}
void xnn_f32_vapproxgelu_ukernel__scalar_u4(
    size_t batch,
    const float* input,
    float* output,
    const struct xnn_f32_default_params unused_params[restrict XNN_MIN_ELEMENTS(1)])
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

    float vy_0 = tanhf(M_SQRT_2_DIV_PI * vx_0 * (1.0f + 0.044715f * vx_0 * vx_0));
    float vy_1 = tanhf(M_SQRT_2_DIV_PI * vx_1 * (1.0f + 0.044715f * vx_1 * vx_1));
    float vy_2 = tanhf(M_SQRT_2_DIV_PI * vx_2 * (1.0f + 0.044715f * vx_2 * vx_2));
    float vy_3 = tanhf(M_SQRT_2_DIV_PI * vx_3 * (1.0f + 0.044715f * vx_3 * vx_3));
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
      const float vy = vx * 0.5f * (1.0f + tanhf(M_SQRT_2_DIV_PI * vx * (1.0f + 0.044715f * vx * vx)));
      *output++ = vy;
      batch -= sizeof(float);
    } while (batch != 0);
  }
}
