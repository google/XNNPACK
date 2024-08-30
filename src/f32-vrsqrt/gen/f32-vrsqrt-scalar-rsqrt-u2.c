// Auto-generated file. Do not edit!
//   Template: src/f32-vrsqrt/scalar-rsqrt.c.in
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


void xnn_f32_vrsqrt_ukernel__scalar_rsqrt_u2(
    size_t batch,
    const float* input,
    float* output,
    const struct xnn_f32_rsqrt_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  for (; batch >= 2 * sizeof(float); batch -= 2 * sizeof(float)) {
    const float vx0 = input[0];
    const float vx1 = input[1];
    input += 2;

    const float vt0 = sqrtf(vx0);
    const float vt1 = sqrtf(vx1);
    const float vy0 = 1.0f / vt0;
    const float vy1 = 1.0f / vt1;

    output[0] = vy0;
    output[1] = vy1;
    output += 2;
  }
  if XNN_UNLIKELY(batch != 0) {
    const float vx = *input;
    const float vy = 1.0f / sqrtf(vx);
    *output = vy;
  }
}
