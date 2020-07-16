// Auto-generated file. Do not edit!
//   Template: src/f32-vlrelu/psimd.c.in
//   Generator: tools/xngen
//
// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <psimd.h>

#include <xnnpack/common.h>
#include <xnnpack/vunary.h>


void xnn_f32_vlrelu_ukernel__psimd_x8(
    size_t n,
    const float* x,
    float* y,
    const union xnn_f32_lrelu_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_DISABLE_TSAN
{
  assert(n != 0);
  assert(n % sizeof(float) == 0);

  const psimd_f32 vslope = psimd_load_splat_f32(&params->scalar.slope);
  for (; n >= 8 * sizeof(float); n -= 8 * sizeof(float)) {
    const psimd_f32 vx0123 = psimd_load_f32(x);
    const psimd_f32 vx4567 = psimd_load_f32(x + 4);
    x += 8;

    psimd_f32 vacc0123 = psimd_mul_f32(vx0123, vslope);
    psimd_f32 vacc4567 = psimd_mul_f32(vx4567, vslope);

    vacc0123 = psimd_signblend_f32(vx0123, vacc0123, vx0123);
    vacc4567 = psimd_signblend_f32(vx4567, vacc4567, vx4567);

    psimd_store_f32(y, vacc0123);
    psimd_store_f32(y + 4, vacc4567);
    y += 8;
  }
  for (; n >= 4 * sizeof(float); n -= 4 * sizeof(float)) {
    const psimd_f32 vx = psimd_load_f32(x);
    x += 4;
    psimd_f32 vacc = psimd_mul_f32(vx, vslope);
    vacc = psimd_signblend_f32(vx, vacc, vx);
    psimd_store_f32(y, vacc);
    y += 4;
  }
  if XNN_UNLIKELY(n != 0) {
    const psimd_f32 vx = psimd_load_f32(x);
    psimd_f32 vacc = psimd_mul_f32(vx, vslope);
    vacc = psimd_signblend_f32(vx, vacc, vx);

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
