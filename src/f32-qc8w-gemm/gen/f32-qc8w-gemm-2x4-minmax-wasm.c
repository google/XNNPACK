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


void xnn_f32_qc8w_gemm_minmax_ukernel_2x4__wasm(
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
  assert(mr <= 2);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(float) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  const float* a0 = a;
  float* c0 = c;
  const float* a1 = (const float*) ((uintptr_t) a0 + a_stride);
  float* c1 = (float*) ((uintptr_t) c0 + cm_stride);
  if XNN_UNPREDICTABLE(mr != 2) {
    a1 = a0;
    c1 = c0;
  }

  const float vmin = params->scalar.min;
  const float vmax = params->scalar.max;
  do {
    float vacc00 = ((const float*)w)[0];
    float vacc01 = ((const float*)w)[1];
    float vacc02 = ((const float*)w)[2];
    float vacc03 = ((const float*)w)[3];
    w = (const float*) w + 4;
    float vacc10 = vacc00;
    float vacc11 = vacc01;
    float vacc12 = vacc02;
    float vacc13 = vacc03;

    size_t k = kc;
    do {
      const float va0 = *a0++;
      const float va1 = *a1++;

      const float vb0 = (float) ((const int8_t*) w)[0];
      const float vb1 = (float) ((const int8_t*) w)[1];
      const float vb2 = (float) ((const int8_t*) w)[2];
      const float vb3 = (float) ((const int8_t*) w)[3];
      w = (const int8_t*) w + 4;

      vacc00 = math_muladd_f32(va0, vb0, vacc00);
      vacc01 = math_muladd_f32(va0, vb1, vacc01);
      vacc02 = math_muladd_f32(va0, vb2, vacc02);
      vacc03 = math_muladd_f32(va0, vb3, vacc03);
      vacc10 = math_muladd_f32(va1, vb0, vacc10);
      vacc11 = math_muladd_f32(va1, vb1, vacc11);
      vacc12 = math_muladd_f32(va1, vb2, vacc12);
      vacc13 = math_muladd_f32(va1, vb3, vacc13);

      k -= sizeof(float);
    } while (k != 0);

    const float vscale0 = ((const float*)w)[0];
    const float vscale1 = ((const float*)w)[1];
    const float vscale2 = ((const float*)w)[2];
    const float vscale3 = ((const float*)w)[3];
    w = (const float*) w + 4;
    vacc00 *= vscale0;
    vacc10 *= vscale0;
    vacc01 *= vscale1;
    vacc11 *= vscale1;
    vacc02 *= vscale2;
    vacc12 *= vscale2;
    vacc03 *= vscale3;
    vacc13 *= vscale3;
    vacc00 = __builtin_wasm_max_f32(vacc00, vmin);
    vacc01 = __builtin_wasm_max_f32(vacc01, vmin);
    vacc02 = __builtin_wasm_max_f32(vacc02, vmin);
    vacc03 = __builtin_wasm_max_f32(vacc03, vmin);
    vacc10 = __builtin_wasm_max_f32(vacc10, vmin);
    vacc11 = __builtin_wasm_max_f32(vacc11, vmin);
    vacc12 = __builtin_wasm_max_f32(vacc12, vmin);
    vacc13 = __builtin_wasm_max_f32(vacc13, vmin);

    vacc00 = __builtin_wasm_min_f32(vacc00, vmax);
    vacc01 = __builtin_wasm_min_f32(vacc01, vmax);
    vacc02 = __builtin_wasm_min_f32(vacc02, vmax);
    vacc03 = __builtin_wasm_min_f32(vacc03, vmax);
    vacc10 = __builtin_wasm_min_f32(vacc10, vmax);
    vacc11 = __builtin_wasm_min_f32(vacc11, vmax);
    vacc12 = __builtin_wasm_min_f32(vacc12, vmax);
    vacc13 = __builtin_wasm_min_f32(vacc13, vmax);

    if XNN_LIKELY(nc >= 4) {
      c0[0] = vacc00;
      c0[1] = vacc01;
      c0[2] = vacc02;
      c0[3] = vacc03;
      c0 = (float*) ((uintptr_t) c0 + cn_stride);
      c1[0] = vacc10;
      c1[1] = vacc11;
      c1[2] = vacc12;
      c1[3] = vacc13;
      c1 = (float*) ((uintptr_t) c1 + cn_stride);

      a0 = (const void*) ((uintptr_t) a0 - kc);
      a1 = (const void*) ((uintptr_t) a1 - kc);

      nc -= 4;
    } else {
      if (nc & 2) {
        c0[0] = vacc00;
        c0[1] = vacc01;
        vacc00 = vacc02;
        c0 += 2;
        c1[0] = vacc10;
        c1[1] = vacc11;
        vacc10 = vacc12;
        c1 += 2;
      }
      if (nc & 1) {
        c0[0] = vacc00;
        c1[0] = vacc10;
      }

      nc = 0;
    }
  } while (nc != 0);
}
