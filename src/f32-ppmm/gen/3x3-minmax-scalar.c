// Auto-generated file. Do not edit!
//   Template: src/f32-ppmm/scalar.c.in
//   Generator: tools/xngen
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <xnnpack/math.h>
#include <xnnpack/ppmm.h>


void xnn_f32_ppmm_minmax_ukernel_3x3__scalar(
  size_t mr,
  size_t nc,
  size_t kc,
  const float*restrict a,
  const float*restrict w,
  float*restrict c,
  size_t cm_stride,
  size_t cn_stride,
  const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(mr != 0);
  assert(mr <= 3);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(float) == 0);

  float* c0 = c;
  float* c1 = (float*) ((uintptr_t) c0 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 2) {
    c1 = c0;
  }
  float* c2 = (float*) ((uintptr_t) c1 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 2) {
    c2 = c1;
  }

  do {
    float vacc0x0 = w[0];
    float vacc0x1 = w[1];
    float vacc0x2 = w[2];
    float vacc1x0 = vacc0x0;
    float vacc1x1 = vacc0x1;
    float vacc1x2 = vacc0x2;
    float vacc2x0 = vacc0x0;
    float vacc2x1 = vacc0x1;
    float vacc2x2 = vacc0x2;
    w += 3;

    size_t k = kc;
    do {
      const float va0 = a[0];
      const float va1 = a[1];
      const float va2 = a[2];
      a += 3;

      const float vb0 = w[0];
      const float vb1 = w[1];
      const float vb2 = w[2];
      w += 3;

      vacc0x0 += va0 * vb0;
      vacc1x0 += va1 * vb0;
      vacc2x0 += va2 * vb0;
      vacc0x1 += va0 * vb1;
      vacc1x1 += va1 * vb1;
      vacc2x1 += va2 * vb1;
      vacc0x2 += va0 * vb2;
      vacc1x2 += va1 * vb2;
      vacc2x2 += va2 * vb2;

      k -= sizeof(float);
    } while (k != 0);

    const float vmax = params->scalar.max;
    vacc0x0 = math_min_f32(vacc0x0, vmax);
    vacc1x0 = math_min_f32(vacc1x0, vmax);
    vacc2x0 = math_min_f32(vacc2x0, vmax);
    vacc0x1 = math_min_f32(vacc0x1, vmax);
    vacc1x1 = math_min_f32(vacc1x1, vmax);
    vacc2x1 = math_min_f32(vacc2x1, vmax);
    vacc0x2 = math_min_f32(vacc0x2, vmax);
    vacc1x2 = math_min_f32(vacc1x2, vmax);
    vacc2x2 = math_min_f32(vacc2x2, vmax);

    const float vmin = params->scalar.min;
    vacc0x0 = math_max_f32(vacc0x0, vmin);
    vacc1x0 = math_max_f32(vacc1x0, vmin);
    vacc2x0 = math_max_f32(vacc2x0, vmin);
    vacc0x1 = math_max_f32(vacc0x1, vmin);
    vacc1x1 = math_max_f32(vacc1x1, vmin);
    vacc2x1 = math_max_f32(vacc2x1, vmin);
    vacc0x2 = math_max_f32(vacc0x2, vmin);
    vacc1x2 = math_max_f32(vacc1x2, vmin);
    vacc2x2 = math_max_f32(vacc2x2, vmin);

    if XNN_LIKELY(nc >= 3) {
      c2[0] = vacc2x0;
      c2[1] = vacc2x1;
      c2[2] = vacc2x2;
      c1[0] = vacc1x0;
      c1[1] = vacc1x1;
      c1[2] = vacc1x2;
      c0[0] = vacc0x0;
      c0[1] = vacc0x1;
      c0[2] = vacc0x2;

      a = (const float*) ((uintptr_t) a - kc * 3);

      c2 = (float*) ((uintptr_t) c2 + cn_stride);
      c1 = (float*) ((uintptr_t) c1 + cn_stride);
      c0 = (float*) ((uintptr_t) c0 + cn_stride);

      nc -= 3;
    } else {
      if (nc & 2) {
        c2[0] = vacc2x0;
        c2[1] = vacc2x1;
        c1[0] = vacc1x0;
        c1[1] = vacc1x1;
        c0[0] = vacc0x0;
        c0[1] = vacc0x1;

        vacc2x0 = vacc2x2;
        vacc1x0 = vacc1x2;
        vacc0x0 = vacc0x2;

        c2 += 2;
        c1 += 2;
        c0 += 2;
      }
      if (nc & 1) {
        *c2 = vacc2x0;
        *c1 = vacc1x0;
        *c0 = vacc0x0;
      }

      nc = 0;
    }
  } while (nc != 0);
}
