// Auto-generated file. Do not edit!
//   Template: src/f32-vrnd/vrndd-psimd.c.in
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


void xnn_f32_vrndd_ukernel__psimd_x8(
    size_t n,
    const float* x,
    float* y,
    const union xnn_f32_rnd_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_DISABLE_TSAN
{
  assert(n != 0);
  assert(n % sizeof(float) == 0);

  const psimd_s32 vsign_mask = psimd_splat_s32(INT32_C(0x80000000));
  const psimd_f32 vmagic_number = psimd_splat_f32(0x1.000000p+23f);
  const psimd_f32 vone = psimd_splat_f32(1.0f);
  for (; n >= 8 * sizeof(float); n -= 8 * sizeof(float)) {
    const psimd_f32 vx0123 = psimd_load_f32(x);
    const psimd_f32 vx4567 = psimd_load_f32(x + 4);
    x += 8;

    const psimd_f32 vabsx0123 = psimd_andnotmask_f32(vsign_mask, vx0123);
    const psimd_f32 vabsx4567 = psimd_andnotmask_f32(vsign_mask, vx4567);

    const psimd_s32 vrndmask0123 = vsign_mask | (vabsx0123 >= vmagic_number);
    const psimd_s32 vrndmask4567 = vsign_mask | (vabsx4567 >= vmagic_number);

    const psimd_f32 vrndabsx0123 = psimd_sub_f32(psimd_add_f32(vabsx0123, vmagic_number), vmagic_number);
    const psimd_f32 vrndabsx4567 = psimd_sub_f32(psimd_add_f32(vabsx4567, vmagic_number), vmagic_number);

    const psimd_f32 vrndx0123 = psimd_blend_f32(vrndmask0123, vx0123, vrndabsx0123);
    const psimd_f32 vrndx4567 = psimd_blend_f32(vrndmask4567, vx4567, vrndabsx4567);

    const psimd_f32 vy0123 = psimd_sub_f32(vrndx0123, psimd_andmask_f32(vrndx0123 > vx0123, vone));
    const psimd_f32 vy4567 = psimd_sub_f32(vrndx4567, psimd_andmask_f32(vrndx4567 > vx4567, vone));

    psimd_store_f32(y, vy0123);
    psimd_store_f32(y + 4, vy4567);
    y += 8;
  }
  for (; n >= 4 * sizeof(float); n -= 4 * sizeof(float)) {
    const psimd_f32 vx = psimd_load_f32(x);
    x += 4;

    const psimd_f32 vabsx = psimd_andnotmask_f32(vsign_mask, vx);
    const psimd_s32 vrndmask = vsign_mask | (vabsx >= vmagic_number);
    const psimd_f32 vrndabsx = psimd_sub_f32(psimd_add_f32(vabsx, vmagic_number), vmagic_number);
    const psimd_f32 vrndx = psimd_blend_f32(vrndmask, vx, vrndabsx);
    const psimd_f32 vy = psimd_sub_f32(vrndx, psimd_andmask_f32(vrndx > vx, vone));

    psimd_store_f32(y, vy);
    y += 4;
  }
  if XNN_UNLIKELY(n != 0) {
    const psimd_f32 vx = psimd_load_f32(x);
    const psimd_f32 vabsx = psimd_andnotmask_f32(vsign_mask, vx);
    const psimd_s32 vrndmask = vsign_mask | (vabsx >= vmagic_number);
    const psimd_f32 vrndabsx = psimd_sub_f32(psimd_add_f32(vabsx, vmagic_number), vmagic_number);
    const psimd_f32 vrndx = psimd_blend_f32(vrndmask, vx, vrndabsx);
    psimd_f32 vy = psimd_sub_f32(vrndx, psimd_andmask_f32(vrndx > vx, vone));
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
