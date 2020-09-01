// Auto-generated file. Do not edit!
//   Template: src/f32-vbinary/vop-scalar.c.in
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


void xnn_f32_vsub_minmax_ukernel__wasm_x8(
    size_t n,
    const float* a,
    const float* b,
    float* y,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(n != 0);
  assert(n % sizeof(float) == 0);
  assert(a != NULL);
  assert(b != NULL);
  assert(y != NULL);

  const float vy_min = params->scalar.min;
  const float vy_max = params->scalar.max;

  for (; n >= 8 * sizeof(float); n -= 8 * sizeof(float)) {
    const float va0 = a[0];
    const float va1 = a[1];
    const float va2 = a[2];
    const float va3 = a[3];
    const float va4 = a[4];
    const float va5 = a[5];
    const float va6 = a[6];
    const float va7 = a[7];
    a += 8;

    const float vb0 = b[0];
    const float vb1 = b[1];
    const float vb2 = b[2];
    const float vb3 = b[3];
    const float vb4 = b[4];
    const float vb5 = b[5];
    const float vb6 = b[6];
    const float vb7 = b[7];
    b += 8;

    float vy0 = va0 - vb0;
    float vy1 = va1 - vb1;
    float vy2 = va2 - vb2;
    float vy3 = va3 - vb3;
    float vy4 = va4 - vb4;
    float vy5 = va5 - vb5;
    float vy6 = va6 - vb6;
    float vy7 = va7 - vb7;


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

    y[0] = vy0;
    y[1] = vy1;
    y[2] = vy2;
    y[3] = vy3;
    y[4] = vy4;
    y[5] = vy5;
    y[6] = vy6;
    y[7] = vy7;
    y += 8;
  }
  if XNN_UNLIKELY(n != 0) {
    do {
      const float va = *a++;
      const float vb = *b++;
      float vy = va - vb;
      vy = __builtin_wasm_max_f32(vy, vy_min);
      vy = __builtin_wasm_min_f32(vy, vy_max);
      *y++ = vy;
      n -= sizeof(float);
    } while (n != 0);
  }
}
