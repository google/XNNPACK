// clang-format off
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

#include "src/xnnpack/common.h"
#include "src/xnnpack/vunary.h"


void xnn_f32_vrsqrt_ukernel__scalar_rsqrt_u1(
    size_t batch,
    const float* input,
    float* output,
    const struct xnn_f32_default_params* restrict params)
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  for (; batch >= sizeof(float); batch -= sizeof(float)) {
    const float vx = *input++;
    const float vy = 1.0f / sqrtf(vx);
    *output++ = vy;
  }
}
