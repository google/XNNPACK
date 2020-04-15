// Auto-generated file. Do not edit!
//   Template: src/f32-hswish/psimd.c.in
//   Generator: tools/xngen
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <psimd.h>

#include <xnnpack/common.h>
#include <xnnpack/vbinary.h>


void xnn_f32_hswish_ukernel__psimd_x8(
    size_t n,
    const float* x,
    float* y,
    const union xnn_f32_hswish_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(n != 0);
  assert(n % sizeof(float) == 0);

  const psimd_f32 vsixth = psimd_load_splat_f32(&params->scalar.sixth);
  const psimd_f32 vhalf = psimd_load_splat_f32(&params->scalar.half);
  const psimd_f32 vone = psimd_load_splat_f32(&params->scalar.one);
  const psimd_f32 vzero = psimd_splat_f32(0.0f);

  for (; n >= 8 * sizeof(float); n -= 8 * sizeof(float)) {
    const psimd_f32 vx0123 = psimd_load_f32(x);
    const psimd_f32 vx4567 = psimd_load_f32(x + 4);
    x += 8;

    psimd_f32 vacc0123 = psimd_qfma_f32(vhalf, vx0123, vsixth);
    psimd_f32 vacc4567 = psimd_qfma_f32(vhalf, vx4567, vsixth);

    vacc0123 = psimd_max_f32(vacc0123, vzero);
    vacc4567 = psimd_max_f32(vacc4567, vzero);

    vacc0123 = psimd_min_f32(vacc0123, vone);
    vacc4567 = psimd_min_f32(vacc4567, vone);

    vacc0123 = psimd_mul_f32(vacc0123, vx0123);
    vacc4567 = psimd_mul_f32(vacc4567, vx4567);

    psimd_store_f32(y, vacc0123);
    psimd_store_f32(y + 4, vacc4567);
    y += 8;
  }
  for (; n >= 4 * sizeof(float); n -= 4 * sizeof(float)) {
    const psimd_f32 vx0123 = psimd_load_f32(x);
    x += 4;
    psimd_f32 vacc0123 = psimd_qfma_f32(vhalf, vx0123, vsixth);
    vacc0123 = psimd_max_f32(vacc0123, vzero);
    vacc0123 = psimd_min_f32(vacc0123, vone);
    vacc0123 = psimd_mul_f32(vacc0123, vx0123);
    psimd_store_f32(y, vacc0123);
    y += 4;
  }
  if XNN_UNLIKELY(n != 0) {
    const psimd_f32 vx0123 = psimd_load_f32(x);
    psimd_f32 vacc0123 = psimd_qfma_f32(vhalf, vx0123, vsixth);
    vacc0123 = psimd_max_f32(vacc0123, vzero);
    vacc0123 = psimd_min_f32(vacc0123, vone);
    vacc0123 = psimd_mul_f32(vacc0123, vx0123);

    if (n & (2 * sizeof(float))) {
      psimd_store2_f32(y, vacc0123);
      vacc0123 = psimd_concat_hi_f32(vacc0123, vacc0123);
      y += 2;
    }
    if (n & (1 * sizeof(float))) {
      psimd_store1_f32(y, vacc0123);
    }
  }
}
