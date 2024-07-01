// Auto-generated file. Do not edit!
//   Template: src/f32-vclamp/scalar.c.in
//   Generator: tools/xngen
//
// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include "xnnpack/common.h"
#include "xnnpack/math.h"
#include "xnnpack/vunary.h"


void xnn_f32_vclamp_ukernel__wasm_u1(
    size_t batch,
    const float* input,
    float* output,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const float vy_min = params->scalar.min;
  const float vy_max = params->scalar.max;

  for (; batch >= sizeof(float); batch -= sizeof(float)) {
    float vacc = *input++;
    vacc = __builtin_wasm_max_f32(vacc, vy_min);
    vacc = __builtin_wasm_min_f32(vacc, vy_max);
    *output++ = vacc;
  }
}
