// Auto-generated file. Do not edit!
//   Template: src/f32-gemm/scalar.c.in
//   Generator: tools/xngen
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include "xnnpack/gemm.h"
#include "xnnpack/math.h"


void xnn_f32_qc8w_gemm_minmax_ukernel_1x4__scalar(
    size_t mr,
    size_t nc,
    size_t kc,
    const float* restrict a,
    size_t a_stride,
    const void* restrict w,
    float* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(mr != 0);
  assert(mr <= 1);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(float) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  const float* a0 = a;
  float* c0 = c;

  const float vmin = params->scalar.min;
  const float vmax = params->scalar.max;
  do {
    float vacc00 = ((const float*)w)[0];
    float vacc01 = ((const float*)w)[1];
    float vacc02 = ((const float*)w)[2];
    float vacc03 = ((const float*)w)[3];
    w = (const float*) w + 4;

    size_t k = kc;
    do {
      const float va0 = *a0++;

      const float vb0 = (float) ((const int8_t*) w)[0];
      const float vb1 = (float) ((const int8_t*) w)[1];
      const float vb2 = (float) ((const int8_t*) w)[2];
      const float vb3 = (float) ((const int8_t*) w)[3];
      w = (const int8_t*) w + 4;

      vacc00 = math_muladd_f32(va0, vb0, vacc00);
      vacc01 = math_muladd_f32(va0, vb1, vacc01);
      vacc02 = math_muladd_f32(va0, vb2, vacc02);
      vacc03 = math_muladd_f32(va0, vb3, vacc03);

      k -= sizeof(float);
    } while (k != 0);

    const float vscale0 = ((const float*)w)[0];
    const float vscale1 = ((const float*)w)[1];
    const float vscale2 = ((const float*)w)[2];
    const float vscale3 = ((const float*)w)[3];
    w = (const float*) w + 4;
    vacc00 *= vscale0;
    vacc01 *= vscale1;
    vacc02 *= vscale2;
    vacc03 *= vscale3;
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

      a0 = (const void*) ((uintptr_t) a0 - kc);

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
