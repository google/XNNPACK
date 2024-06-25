// Auto-generated file. Do not edit!
//   Template: src/f32-igemm/scalar.c.in
//   Generator: tools/xngen
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include "xnnpack/igemm.h"
#include "xnnpack/math.h"


void xnn_f32_igemm_minmax_ukernel_1x4__scalar(
    size_t mr,
    size_t nc,
    size_t kc,
    size_t ks,
    const float** restrict a,
    const float* restrict w,
    float* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    size_t a_offset,
    const float* zero,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(mr != 0);
  assert(mr <= 1);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(float) == 0);
  assert(ks != 0);
  assert(ks % (1 * sizeof(void*)) == 0);
  assert(a_offset % sizeof(float) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  float* c0 = c;

  const float vmin = params->scalar.min;
  const float vmax = params->scalar.max;
  do {
    float vacc00 = w[0];
    float vacc01 = w[1];
    float vacc02 = w[2];
    float vacc03 = w[3];
    w += 4;

    size_t p = ks;
    do {
      const float* restrict a0 = a[0];
      assert(a0 != NULL);
      if XNN_UNPREDICTABLE(a0 != zero) {
        a0 = (const float*) ((uintptr_t) a0 + a_offset);
      }
      a += 1;

      size_t k = kc;
      do {
        const float va0 = *a0++;

        const float vb0 = w[0];
        const float vb1 = w[1];
        const float vb2 = w[2];
        const float vb3 = w[3];
        w += 4;

        vacc00 = math_muladd_f32(va0, vb0, vacc00);
        vacc01 = math_muladd_f32(va0, vb1, vacc01);
        vacc02 = math_muladd_f32(va0, vb2, vacc02);
        vacc03 = math_muladd_f32(va0, vb3, vacc03);

        k -= sizeof(float);
      } while (k != 0);
      p -= 1 * sizeof(void*);
    } while (p != 0);

    vacc00 = math_max_f32(vacc00, vmin);
    vacc01 = math_max_f32(vacc01, vmin);
    vacc02 = math_max_f32(vacc02, vmin);
    vacc03 = math_max_f32(vacc03, vmin);

    vacc00 = math_min_f32(vacc00, vmax);
    vacc01 = math_min_f32(vacc01, vmax);
    vacc02 = math_min_f32(vacc02, vmax);
    vacc03 = math_min_f32(vacc03, vmax);

    if XNN_LIKELY(nc >= 4) {
      c0[0] = vacc00;
      c0[1] = vacc01;
      c0[2] = vacc02;
      c0[3] = vacc03;
      c0 = (float*) ((uintptr_t) c0 + cn_stride);

      a = (const float**restrict) ((uintptr_t) a - ks);
      nc -= 4;
    } else {
      if (nc & 2) {
        c0[0] = vacc00;
        c0[1] = vacc01;
        vacc00 = vacc02;
        c0 += 2;
      }
      if (nc & 1) {
        c0[0] = vacc00;
      }

      nc = 0;
    }
  } while (nc != 0);
}
