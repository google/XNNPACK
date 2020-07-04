// Auto-generated file. Do not edit!
//   Template: src/f32-vunary/psimd.c.in
//   Generator: tools/xngen
//
// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <psimd.h>

#include <xnnpack/common.h>
#include <xnnpack/math.h>
#include <xnnpack/vunary.h>


void xnn_f32_vabs_ukernel__psimd_x4(
    size_t n,
    const float* x,
    float* y,
    const union xnn_f32_abs_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_DISABLE_TSAN
{
  assert(n != 0);
  assert(n % sizeof(float) == 0);
  assert(x != NULL);
  assert(y != NULL);

  for (; n >= 4 * sizeof(float); n -= 4 * sizeof(float)) {
    const psimd_f32 vx0123 = psimd_load_f32(x);
    x += 4;

    const psimd_f32 vy0123 = psimd_abs_f32(vx0123);

    psimd_store_f32(y, vy0123);
    y += 4;
  }
  if XNN_UNLIKELY(n != 0) {
    const psimd_f32 vx = psimd_load_f32(x);
    psimd_f32 vy = psimd_abs_f32(vx);
    if (n & (2 * sizeof(float))) {
      psimd_store2_f32(y, vy);
      vy = psimd_concat_hi_f32(vy, vy);
      y += 2;
    }
    if (n & (1 * sizeof(float))) {
      psimd_store1_f32(y, vy);
    }
  }
}
