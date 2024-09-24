// Auto-generated file. Do not edit!
//   Template: src/f32-vbinary/vop-scalar.c.in
//   Generator: tools/xngen
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include "xnnpack/common.h"
#include "xnnpack/math.h"
#include "xnnpack/vbinary.h"


void xnn_f32_vmul_minmax_ukernel__wasm_u2(
    size_t batch,
    const float* input_a,
    const float* input_b,
    float* output,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input_a != NULL);
  assert(input_b != NULL);
  assert(output != NULL);

  const float voutput_min = params->scalar.min;
  const float voutput_max = params->scalar.max;

  for (; batch >= 2 * sizeof(float); batch -= 2 * sizeof(float)) {
    const float va0 = input_a[0];
    const float va1 = input_a[1];
    input_a += 2;

    const float vb0 = input_b[0];
    const float vb1 = input_b[1];
    input_b += 2;

    float vacc0 = va0 * vb0;
    float vacc1 = va1 * vb1;


    vacc0 = __builtin_wasm_max_f32(vacc0, voutput_min);
    vacc1 = __builtin_wasm_max_f32(vacc1, voutput_min);

    vacc0 = __builtin_wasm_min_f32(vacc0, voutput_max);
    vacc1 = __builtin_wasm_min_f32(vacc1, voutput_max);

    output[0] = vacc0;
    output[1] = vacc1;
    output += 2;
  }
  if XNN_UNLIKELY(batch != 0) {
    assert(batch == sizeof(float));
    const float va = *input_a;
    const float vb = *input_b;
    float vacc = va * vb;
    vacc = __builtin_wasm_max_f32(vacc, voutput_min);
    vacc = __builtin_wasm_min_f32(vacc, voutput_max);
    *output = vacc;
  }
}
