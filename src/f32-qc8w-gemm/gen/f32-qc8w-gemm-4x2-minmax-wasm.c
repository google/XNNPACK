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
#include "xnnpack/unaligned.h"


void xnn_f32_qc8w_gemm_minmax_ukernel_4x2__wasm(
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
    float vacc00 = unaligned_indexed_load_f32(w, 0);
    float vacc01 = unaligned_indexed_load_f32(w, 1);
    w = (const float*) w + 2;
    float vacc10 = vacc00;
    float vacc11 = vacc01;
    float vacc20 = vacc00;
    float vacc21 = vacc01;
    float vacc30 = vacc00;
    float vacc31 = vacc01;

    size_t k = kc;
    do {
      const float va0 = *a0++;
      const float va1 = *a1++;
      const float va2 = *a2++;
      const float va3 = *a3++;

      const float vb0 = (float) ((const int8_t*) w)[0];
      const float vb1 = (float) ((const int8_t*) w)[1];
      w = (const int8_t*) w + 2;

      vacc00 = math_muladd_f32(va0, vb0, vacc00);
      vacc01 = math_muladd_f32(va0, vb1, vacc01);
      vacc10 = math_muladd_f32(va1, vb0, vacc10);
      vacc11 = math_muladd_f32(va1, vb1, vacc11);
      vacc20 = math_muladd_f32(va2, vb0, vacc20);
      vacc21 = math_muladd_f32(va2, vb1, vacc21);
      vacc30 = math_muladd_f32(va3, vb0, vacc30);
      vacc31 = math_muladd_f32(va3, vb1, vacc31);

      k -= sizeof(float);
    } while (k != 0);

    const float vscale0 = unaligned_indexed_load_f32(w, 0);
    const float vscale1 = unaligned_indexed_load_f32(w, 1);
    w = (const float*) w + 2;
    vacc00 *= vscale0;
    vacc10 *= vscale0;
    vacc20 *= vscale0;
    vacc30 *= vscale0;
    vacc01 *= vscale1;
    vacc11 *= vscale1;
    vacc21 *= vscale1;
    vacc31 *= vscale1;
    vacc00 = __builtin_wasm_max_f32(vacc00, vmin);
    vacc01 = __builtin_wasm_max_f32(vacc01, vmin);
    vacc10 = __builtin_wasm_max_f32(vacc10, vmin);
    vacc11 = __builtin_wasm_max_f32(vacc11, vmin);
    vacc20 = __builtin_wasm_max_f32(vacc20, vmin);
    vacc21 = __builtin_wasm_max_f32(vacc21, vmin);
    vacc30 = __builtin_wasm_max_f32(vacc30, vmin);
    vacc31 = __builtin_wasm_max_f32(vacc31, vmin);

    vacc00 = __builtin_wasm_min_f32(vacc00, vmax);
    vacc01 = __builtin_wasm_min_f32(vacc01, vmax);
    vacc10 = __builtin_wasm_min_f32(vacc10, vmax);
    vacc11 = __builtin_wasm_min_f32(vacc11, vmax);
    vacc20 = __builtin_wasm_min_f32(vacc20, vmax);
    vacc21 = __builtin_wasm_min_f32(vacc21, vmax);
    vacc30 = __builtin_wasm_min_f32(vacc30, vmax);
    vacc31 = __builtin_wasm_min_f32(vacc31, vmax);

    if XNN_LIKELY(nc >= 2) {
      c0[0] = vacc00;
      c0[1] = vacc01;
      c0 = (float*) ((uintptr_t) c0 + cn_stride);
      c1[0] = vacc10;
      c1[1] = vacc11;
      c1 = (float*) ((uintptr_t) c1 + cn_stride);
      c2[0] = vacc20;
      c2[1] = vacc21;
      c2 = (float*) ((uintptr_t) c2 + cn_stride);
      c3[0] = vacc30;
      c3[1] = vacc31;
      c3 = (float*) ((uintptr_t) c3 + cn_stride);

      a0 = (const void*) ((uintptr_t) a0 - kc);
      a1 = (const void*) ((uintptr_t) a1 - kc);
      a2 = (const void*) ((uintptr_t) a2 - kc);
      a3 = (const void*) ((uintptr_t) a3 - kc);

      nc -= 2;
    } else {
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
