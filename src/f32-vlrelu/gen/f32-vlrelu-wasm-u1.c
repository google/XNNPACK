// Auto-generated file. Do not edit!
//   Template: src/f32-vlrelu/wasm.c.in
//   Generator: tools/xngen
//
// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include "xnnpack/common.h"
#include "xnnpack/vunary.h"


void xnn_f32_vlrelu_ukernel__wasm_u1(
    size_t batch,
    const float* input,
    float* output,
    const union xnn_f32_lrelu_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const float vslope = params->scalar.slope;
  const float vzero = 0.0f;

  do {
    const float vx = *input++;
    const float vnegx = __builtin_wasm_min_f32(vx, vzero);
    float vacc = vnegx * vslope;
    const float vposx = __builtin_wasm_max_f32(vx, vzero);
    vacc += vposx;
    *output++ = vacc;
    batch -= sizeof(float);
  } while (batch != 0);
}
