/*
 * Copyright 2019 Google LLC
 *
 * This source code is licensed under the BSD-style license found in the
 * LICENSE file in the root directory of this source tree.
 */

#include <assert.h>

#include <psimd.h>

#include <xnnpack/common.h>
#include <xnnpack/prelu.h>


void xnn_f32_prelu_ukernel_x4__psimd(
    size_t mr,
    size_t n,
    const float* x,
    size_t x_stride,
    const float* w,
    float* y,
    size_t y_stride,
    const union xnn_f32_output_params params[restrict static 1])
{
  assert(mr != 0);
  assert(mr <= 4);
  assert(n != 0);
  assert(n % sizeof(float) == 0);

  const float* x0 = x;
  const float* x1 = (const float*) ((uintptr_t) x0 + x_stride);
  if (mr < 2) {
    x1 = x0;
  }
  const float* x2 = (const float*) ((uintptr_t) x1 + x_stride);
  if (mr <= 2) {
    x2 = x1;
  }
  const float* x3 = (const float*) ((uintptr_t) x2 + x_stride);
  if (mr != 4) {
    x3 = x2;
  }

  float* y0 = y;
  float* y1 = (float*) ((uintptr_t) y0 + y_stride);
  if (mr < 2) {
    y1 = y0;
  }
  float* y2 = (float*) ((uintptr_t) y1 + y_stride);
  if (mr <= 2) {
    y2 = y1;
  }
  float* y3 = (float*) ((uintptr_t) y2 + y_stride);
  if (mr != 4) {
    y3 = y2;
  }

  const psimd_f32 vy_min = psimd_load_splat_f32(&params->scalar.min);
  const psimd_f32 vy_max = psimd_load_splat_f32(&params->scalar.max);
  for (; n >= 4 * sizeof(float); n -= 4 * sizeof(float)) {
    const psimd_f32 vw = psimd_load_f32(w);
    w += 4;
    const psimd_f32 vx0 = psimd_load_f32(x0);
    x0 += 4;
    const psimd_f32 vx1 = psimd_load_f32(x1);
    x1 += 4;
    const psimd_f32 vx2 = psimd_load_f32(x2);
    x2 += 4;
    const psimd_f32 vx3 = psimd_load_f32(x3);
    x3 += 4;

    const psimd_f32 vacc0 = psimd_signblend_f32(vx0, vx0 * vw, vx0);
    const psimd_f32 vacc1 = psimd_signblend_f32(vx1, vx1 * vw, vx1);
    const psimd_f32 vacc2 = psimd_signblend_f32(vx2, vx2 * vw, vx2);
    const psimd_f32 vacc3 = psimd_signblend_f32(vx3, vx3 * vw, vx3);

    const psimd_f32 vy0 = psimd_min_f32(psimd_max_f32(vacc0, vy_min), vy_max);
    const psimd_f32 vy1 = psimd_min_f32(psimd_max_f32(vacc1, vy_min), vy_max);
    const psimd_f32 vy2 = psimd_min_f32(psimd_max_f32(vacc2, vy_min), vy_max);
    const psimd_f32 vy3 = psimd_min_f32(psimd_max_f32(vacc3, vy_min), vy_max);

    psimd_store_f32(y0, vy0);
    y0 += 4;
    psimd_store_f32(y1, vy1);
    y1 += 4;
    psimd_store_f32(y2, vy2);
    y2 += 4;
    psimd_store_f32(y3, vy3);
    y3 += 4;
  }
  if (n != 0) {
    const psimd_f32 vw = psimd_load_f32(w);
    const psimd_f32 vx0 = psimd_load_f32(x0);
    const psimd_f32 vx1 = psimd_load_f32(x1);
    const psimd_f32 vx2 = psimd_load_f32(x2);
    const psimd_f32 vx3 = psimd_load_f32(x3);

    const psimd_f32 vacc0 = psimd_signblend_f32(vx0, vx0 * vw, vx0);
    const psimd_f32 vacc1 = psimd_signblend_f32(vx1, vx1 * vw, vx1);
    const psimd_f32 vacc2 = psimd_signblend_f32(vx2, vx2 * vw, vx2);
    const psimd_f32 vacc3 = psimd_signblend_f32(vx3, vx3 * vw, vx3);

    psimd_f32 vy0 = psimd_min_f32(psimd_max_f32(vacc0, vy_min), vy_max);
    psimd_f32 vy1 = psimd_min_f32(psimd_max_f32(vacc1, vy_min), vy_max);
    psimd_f32 vy2 = psimd_min_f32(psimd_max_f32(vacc2, vy_min), vy_max);
    psimd_f32 vy3 = psimd_min_f32(psimd_max_f32(vacc3, vy_min), vy_max);

    if (n & 2 * sizeof(float)) {
      psimd_store2_f32(y0, vy0);
      y0 += 2;
      psimd_store2_f32(y1, vy1);
      y1 += 2;
      psimd_store2_f32(y2, vy2);
      y2 += 2;
      psimd_store2_f32(y3, vy3);
      y3 += 2;

      vy0 = psimd_concat_hi_f32(vy0, vy0);
      vy1 = psimd_concat_hi_f32(vy1, vy1);
      vy2 = psimd_concat_hi_f32(vy2, vy2);
      vy3 = psimd_concat_hi_f32(vy3, vy3);
    }
    if (n & 1 * sizeof(float)) {
      psimd_store1_f32(y0, vy0);
      psimd_store1_f32(y1, vy1);
      psimd_store1_f32(y2, vy2);
      psimd_store1_f32(y3, vy3);
    }
  }
}
