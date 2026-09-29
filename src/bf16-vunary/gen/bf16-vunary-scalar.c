// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/bf16-vunary/scalar.c.in
//   Generator: tools/xngen
//
// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <math.h>
#include <stddef.h>

#include "src/xnnpack/math.h"
#include "src/xnnpack/microparams.h"
#include "src/xnnpack/vunary.h"


void xnn_bf16_vsqr_ukernel__scalar_u4(
    size_t batch,
    const xnn_bfloat16* input,
    xnn_bfloat16* output,
    const struct xnn_bf16_default_params* restrict params)
{
  assert(batch != 0);
  assert(batch % sizeof(xnn_bfloat16) == 0);
  assert(input != NULL);
  assert(output != NULL);

  for (; batch >= 4 * sizeof(xnn_bfloat16);
       batch -= 4 * sizeof(xnn_bfloat16)) {
    const float vx0 = xnn_bfloat16_to_float(input[0]);
    const float vx1 = xnn_bfloat16_to_float(input[1]);
    const float vx2 = xnn_bfloat16_to_float(input[2]);
    const float vx3 = xnn_bfloat16_to_float(input[3]);
    input += 4;

    const float vy0 = vx0 * vx0;
    const float vy1 = vx1 * vx1;
    const float vy2 = vx2 * vx2;
    const float vy3 = vx3 * vx3;

    output[0] = xnn_bfloat16_from_float(vy0);
    output[1] = xnn_bfloat16_from_float(vy1);
    output[2] = xnn_bfloat16_from_float(vy2);
    output[3] = xnn_bfloat16_from_float(vy3);
    output += 4;
  }

  while (batch != 0) {
    const float vx = xnn_bfloat16_to_float(*input++);
    const float vy = vx * vx;
    *output++ = xnn_bfloat16_from_float(vy);
    batch -= sizeof(xnn_bfloat16);
  }
}
void xnn_bf16_vrsqrt_ukernel__scalar_u4(
    size_t batch,
    const xnn_bfloat16* input,
    xnn_bfloat16* output,
    const struct xnn_bf16_default_params* restrict params)
{
  assert(batch != 0);
  assert(batch % sizeof(xnn_bfloat16) == 0);
  assert(input != NULL);
  assert(output != NULL);

  for (; batch >= 4 * sizeof(xnn_bfloat16);
       batch -= 4 * sizeof(xnn_bfloat16)) {
    const float vx0 = xnn_bfloat16_to_float(input[0]);
    const float vx1 = xnn_bfloat16_to_float(input[1]);
    const float vx2 = xnn_bfloat16_to_float(input[2]);
    const float vx3 = xnn_bfloat16_to_float(input[3]);
    input += 4;

    const float vy0 = 1.0f / sqrtf(vx0);
    const float vy1 = 1.0f / sqrtf(vx1);
    const float vy2 = 1.0f / sqrtf(vx2);
    const float vy3 = 1.0f / sqrtf(vx3);

    output[0] = xnn_bfloat16_from_float(vy0);
    output[1] = xnn_bfloat16_from_float(vy1);
    output[2] = xnn_bfloat16_from_float(vy2);
    output[3] = xnn_bfloat16_from_float(vy3);
    output += 4;
  }

  while (batch != 0) {
    const float vx = xnn_bfloat16_to_float(*input++);
    const float vy = 1.0f / sqrtf(vx);
    *output++ = xnn_bfloat16_from_float(vy);
    batch -= sizeof(xnn_bfloat16);
  }
}
void xnn_bf16_vsigmoid_ukernel__scalar_u4(
    size_t batch,
    const xnn_bfloat16* input,
    xnn_bfloat16* output,
    const struct xnn_bf16_default_params* restrict params)
{
  assert(batch != 0);
  assert(batch % sizeof(xnn_bfloat16) == 0);
  assert(input != NULL);
  assert(output != NULL);

  for (; batch >= 4 * sizeof(xnn_bfloat16);
       batch -= 4 * sizeof(xnn_bfloat16)) {
    const float vx0 = xnn_bfloat16_to_float(input[0]);
    const float vx1 = xnn_bfloat16_to_float(input[1]);
    const float vx2 = xnn_bfloat16_to_float(input[2]);
    const float vx3 = xnn_bfloat16_to_float(input[3]);
    input += 4;

    const float vy0 = 1.0f / (1.0f + expf(-vx0));
    const float vy1 = 1.0f / (1.0f + expf(-vx1));
    const float vy2 = 1.0f / (1.0f + expf(-vx2));
    const float vy3 = 1.0f / (1.0f + expf(-vx3));

    output[0] = xnn_bfloat16_from_float(vy0);
    output[1] = xnn_bfloat16_from_float(vy1);
    output[2] = xnn_bfloat16_from_float(vy2);
    output[3] = xnn_bfloat16_from_float(vy3);
    output += 4;
  }

  while (batch != 0) {
    const float vx = xnn_bfloat16_to_float(*input++);
    const float vy = 1.0f / (1.0f + expf(-vx));
    *output++ = xnn_bfloat16_from_float(vy);
    batch -= sizeof(xnn_bfloat16);
  }
}
