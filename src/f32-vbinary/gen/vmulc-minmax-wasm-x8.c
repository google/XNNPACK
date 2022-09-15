// Auto-generated file. Do not edit!
//   Template: src/f32-vbinary/vopc-scalar.c.in
//   Generator: tools/xngen
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <xnnpack/common.h>
#include <xnnpack/math.h>
#include <xnnpack/vbinary.h>


void xnn_f32_vmulc_minmax_ukernel__wasm_x8(
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

  const float vy_min = params->scalar.min;
  const float vy_max = params->scalar.max;

  const float vb = *input_b;
  for (; batch >= 8 * sizeof(float); batch -= 8 * sizeof(float)) {
    const float va0 = input_a[0];
    const float va1 = input_a[1];
    const float va2 = input_a[2];
    const float va3 = input_a[3];
    const float va4 = input_a[4];
    const float va5 = input_a[5];
    const float va6 = input_a[6];
    const float va7 = input_a[7];
    input_a += 8;

    float vy0 = va0 * vb;
    float vy1 = va1 * vb;
    float vy2 = va2 * vb;
    float vy3 = va3 * vb;
    float vy4 = va4 * vb;
    float vy5 = va5 * vb;
    float vy6 = va6 * vb;
    float vy7 = va7 * vb;


    vy0 = __builtin_wasm_max_f32(vy0, vy_min);
    vy1 = __builtin_wasm_max_f32(vy1, vy_min);
    vy2 = __builtin_wasm_max_f32(vy2, vy_min);
    vy3 = __builtin_wasm_max_f32(vy3, vy_min);
    vy4 = __builtin_wasm_max_f32(vy4, vy_min);
    vy5 = __builtin_wasm_max_f32(vy5, vy_min);
    vy6 = __builtin_wasm_max_f32(vy6, vy_min);
    vy7 = __builtin_wasm_max_f32(vy7, vy_min);

    vy0 = __builtin_wasm_min_f32(vy0, vy_max);
    vy1 = __builtin_wasm_min_f32(vy1, vy_max);
    vy2 = __builtin_wasm_min_f32(vy2, vy_max);
    vy3 = __builtin_wasm_min_f32(vy3, vy_max);
    vy4 = __builtin_wasm_min_f32(vy4, vy_max);
    vy5 = __builtin_wasm_min_f32(vy5, vy_max);
    vy6 = __builtin_wasm_min_f32(vy6, vy_max);
    vy7 = __builtin_wasm_min_f32(vy7, vy_max);

    output[0] = vy0;
    output[1] = vy1;
    output[2] = vy2;
    output[3] = vy3;
    output[4] = vy4;
    output[5] = vy5;
    output[6] = vy6;
    output[7] = vy7;
    output += 8;
  }
  if XNN_UNLIKELY(batch != 0) {
    do {
      const float va = *input_a++;
      float vy = va * vb;
      vy = __builtin_wasm_max_f32(vy, vy_min);
      vy = __builtin_wasm_min_f32(vy, vy_max);
      *output++ = vy;
      batch -= sizeof(float);
    } while (batch != 0);
  }
}
