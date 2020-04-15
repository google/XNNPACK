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


void xnn_f32_vmaxc_ukernel__wasm_x2(
    size_t n,
    const float* a,
    const float* b,
    float* y,
    const union xnn_f32_default_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(n != 0);
  assert(n % sizeof(float) == 0);


  const float vb = *b;
  for (; n >= 2 * sizeof(float); n -= 2 * sizeof(float)) {
    const float va0 = a[0];
    const float va1 = a[1];
    a += 2;

    float vy0 = __builtin_wasm_max_f32(va0, vb);
    float vy1 = __builtin_wasm_max_f32(va1, vb);


    y[0] = vy0;
    y[1] = vy1;
    y += 2;
  }
  if XNN_UNLIKELY(n != 0) {
    const float va = *a;
    float vy = __builtin_wasm_max_f32(va, vb);
    *y = vy;
  }
}
