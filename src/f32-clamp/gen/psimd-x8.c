// Auto-generated file. Do not edit!
//   Template: src/f32-clamp/psimd.c.in
//   Generator: tools/xngen
//
// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <psimd.h>

#include <xnnpack/clamp.h>
#include <xnnpack/common.h>


void xnn_f32_clamp_ukernel__psimd_x8(
    size_t n,
    const float* x,
    float* y,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(n != 0);
  assert(n % sizeof(float) == 0);

  const psimd_f32 vy_min = psimd_load_splat_f32(&params->scalar.min);
  const psimd_f32 vy_max = psimd_load_splat_f32(&params->scalar.max);

  for (; n >= 8 * sizeof(float); n -= 8 * sizeof(float)) {
    psimd_f32 vacc0123 = psimd_load_f32(x);
    psimd_f32 vacc4567 = psimd_load_f32(x + 4);
    x += 8;

    vacc0123 = psimd_max_f32(vacc0123, vy_min);
    vacc4567 = psimd_max_f32(vacc4567, vy_min);

    vacc0123 = psimd_min_f32(vacc0123, vy_max);
    vacc4567 = psimd_min_f32(vacc4567, vy_max);

    psimd_store_f32(y, vacc0123);
    psimd_store_f32(y + 4, vacc4567);
    y += 8;
  }
  for (; n >= 4 * sizeof(float); n -= 4 * sizeof(float)) {
    psimd_f32 vacc = psimd_load_f32(x);
    x += 4;

    vacc = psimd_max_f32(vacc, vy_min);
    vacc = psimd_min_f32(vacc, vy_max);

    psimd_store_f32(y, vacc);
    y += 4;
  }
  if XNN_UNLIKELY(n != 0) {
    psimd_f32 vacc = psimd_load_f32(x);
    vacc = psimd_max_f32(vacc, vy_min);
    vacc = psimd_min_f32(vacc, vy_max);

    if (n & (2 * sizeof(float))) {
      psimd_store2_f32(y, vacc);
      vacc = psimd_concat_hi_f32(vacc, vacc);
      y += 2;
    }
    if (n & (1 * sizeof(float))) {
      psimd_store1_f32(y, vacc);
    }
  }
}
