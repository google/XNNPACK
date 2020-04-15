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


void xnn_f32_vsub_minmax_ukernel__wasm_x1(
    size_t n,
    const float* a,
    const float* b,
    float* y,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(n != 0);
  assert(n % sizeof(float) == 0);

  const float vy_min = params->scalar.min;
  const float vy_max = params->scalar.max;

  for (; n >= sizeof(float); n -= sizeof(float)) {
    const float va = *a++;
    const float vb = *b++;
    float vy = va - vb;
    vy = __builtin_wasm_max_f32(vy, vy_min);
    vy = __builtin_wasm_min_f32(vy, vy_max);
    *y++ = vy;
  }
}
