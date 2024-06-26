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


void xnn_f32_gemm_minmax_ukernel_4x4__wasm(
    size_t mr,
    size_t nc,
    size_t kc,
    const float* restrict a,
    size_t a_stride,
    const float* restrict w,
    float* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(mr != 0);
  assert(mr <= 4);
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
  if XNN_UNPREDICTABLE(mr < 2) {
    a1 = a0;
    c1 = c0;
  }
  const float* a2 = (const float*) ((uintptr_t) a1 + a_stride);
  float* c2 = (float*) ((uintptr_t) c1 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 2) {
    a2 = a1;
    c2 = c1;
  }
  const float* a3 = (const float*) ((uintptr_t) a2 + a_stride);
  float* c3 = (float*) ((uintptr_t) c2 + cm_stride);
  if XNN_UNPREDICTABLE(mr != 4) {
    a3 = a2;
    c3 = c2;
  }

  const float vmin = params->scalar.min;
  const float vmax = params->scalar.max;
  do {
    float vacc00 = w[0];
    float vacc01 = w[1];
    float vacc02 = w[2];
    float vacc03 = w[3];
    w += 4;
    float vacc10 = vacc00;
    float vacc11 = vacc01;
    float vacc12 = vacc02;
    float vacc13 = vacc03;
    float vacc20 = vacc00;
    float vacc21 = vacc01;
    float vacc22 = vacc02;
    float vacc23 = vacc03;
    float vacc30 = vacc00;
    float vacc31 = vacc01;
    float vacc32 = vacc02;
    float vacc33 = vacc03;

    size_t k = kc;
    do {
      const float va0 = *a0++;
      const float va1 = *a1++;
      const float va2 = *a2++;
      const float va3 = *a3++;

      const float vb0 = w[0];
      const float vb1 = w[1];
      const float vb2 = w[2];
      const float vb3 = w[3];
      w += 4;

      vacc00 = math_muladd_f32(va0, vb0, vacc00);
      vacc01 = math_muladd_f32(va0, vb1, vacc01);
      vacc02 = math_muladd_f32(va0, vb2, vacc02);
      vacc03 = math_muladd_f32(va0, vb3, vacc03);
      vacc10 = math_muladd_f32(va1, vb0, vacc10);
      vacc11 = math_muladd_f32(va1, vb1, vacc11);
      vacc12 = math_muladd_f32(va1, vb2, vacc12);
      vacc13 = math_muladd_f32(va1, vb3, vacc13);
      vacc20 = math_muladd_f32(va2, vb0, vacc20);
      vacc21 = math_muladd_f32(va2, vb1, vacc21);
      vacc22 = math_muladd_f32(va2, vb2, vacc22);
      vacc23 = math_muladd_f32(va2, vb3, vacc23);
      vacc30 = math_muladd_f32(va3, vb0, vacc30);
      vacc31 = math_muladd_f32(va3, vb1, vacc31);
      vacc32 = math_muladd_f32(va3, vb2, vacc32);
      vacc33 = math_muladd_f32(va3, vb3, vacc33);

      k -= sizeof(float);
    } while (k != 0);

    vacc00 = __builtin_wasm_max_f32(vacc00, vmin);
    vacc01 = __builtin_wasm_max_f32(vacc01, vmin);
    vacc02 = __builtin_wasm_max_f32(vacc02, vmin);
    vacc03 = __builtin_wasm_max_f32(vacc03, vmin);
    vacc10 = __builtin_wasm_max_f32(vacc10, vmin);
    vacc11 = __builtin_wasm_max_f32(vacc11, vmin);
    vacc12 = __builtin_wasm_max_f32(vacc12, vmin);
    vacc13 = __builtin_wasm_max_f32(vacc13, vmin);
    vacc20 = __builtin_wasm_max_f32(vacc20, vmin);
    vacc21 = __builtin_wasm_max_f32(vacc21, vmin);
    vacc22 = __builtin_wasm_max_f32(vacc22, vmin);
    vacc23 = __builtin_wasm_max_f32(vacc23, vmin);
    vacc30 = __builtin_wasm_max_f32(vacc30, vmin);
    vacc31 = __builtin_wasm_max_f32(vacc31, vmin);
    vacc32 = __builtin_wasm_max_f32(vacc32, vmin);
    vacc33 = __builtin_wasm_max_f32(vacc33, vmin);

    vacc00 = __builtin_wasm_min_f32(vacc00, vmax);
    vacc01 = __builtin_wasm_min_f32(vacc01, vmax);
    vacc02 = __builtin_wasm_min_f32(vacc02, vmax);
    vacc03 = __builtin_wasm_min_f32(vacc03, vmax);
    vacc10 = __builtin_wasm_min_f32(vacc10, vmax);
    vacc11 = __builtin_wasm_min_f32(vacc11, vmax);
    vacc12 = __builtin_wasm_min_f32(vacc12, vmax);
    vacc13 = __builtin_wasm_min_f32(vacc13, vmax);
    vacc20 = __builtin_wasm_min_f32(vacc20, vmax);
    vacc21 = __builtin_wasm_min_f32(vacc21, vmax);
    vacc22 = __builtin_wasm_min_f32(vacc22, vmax);
    vacc23 = __builtin_wasm_min_f32(vacc23, vmax);
    vacc30 = __builtin_wasm_min_f32(vacc30, vmax);
    vacc31 = __builtin_wasm_min_f32(vacc31, vmax);
    vacc32 = __builtin_wasm_min_f32(vacc32, vmax);
    vacc33 = __builtin_wasm_min_f32(vacc33, vmax);

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
      c2[0] = vacc20;
      c2[1] = vacc21;
      c2[2] = vacc22;
      c2[3] = vacc23;
      c2 = (float*) ((uintptr_t) c2 + cn_stride);
      c3[0] = vacc30;
      c3[1] = vacc31;
      c3[2] = vacc32;
      c3[3] = vacc33;
      c3 = (float*) ((uintptr_t) c3 + cn_stride);

      a0 = (const void*) ((uintptr_t) a0 - kc);
      a1 = (const void*) ((uintptr_t) a1 - kc);
      a2 = (const void*) ((uintptr_t) a2 - kc);
      a3 = (const void*) ((uintptr_t) a3 - kc);

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
        c2[0] = vacc20;
        c2[1] = vacc21;
        vacc20 = vacc22;
        c2 += 2;
        c3[0] = vacc30;
        c3[1] = vacc31;
        vacc30 = vacc32;
        c3 += 2;
      }
      if (nc & 1) {
        c0[0] = vacc00;
        c1[0] = vacc10;
        c2[0] = vacc20;
        c3[0] = vacc30;
      }

      nc = 0;
    }
  } while (nc != 0);
}
