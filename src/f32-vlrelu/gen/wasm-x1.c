// Auto-generated file. Do not edit!
//   Template: src/f32-vlrelu/wasm.c.in
//   Generator: tools/xngen
//
// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <xnnpack/common.h>
#include <xnnpack/vunary.h>


void xnn_f32_vlrelu_ukernel__wasm_x1(
    size_t n,
    const float* x,
    float* y,
    const union xnn_f32_lrelu_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(n != 0);
  assert(n % sizeof(float) == 0);

  const float vslope = params->scalar.slope;
  const float vzero = 0.0f;

  do {
    const float vx = *x++;
    const float vnegx = __builtin_wasm_min_f32(vx, vzero);
    float vacc = vnegx * vslope;
    const float vposx = __builtin_wasm_max_f32(vx, vzero);
    vacc += vposx;
    *y++ = vacc;
    n -= sizeof(float);
  } while (n != 0);
}
