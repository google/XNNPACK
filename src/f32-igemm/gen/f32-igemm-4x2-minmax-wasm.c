// Auto-generated file. Do not edit!
//   Template: src/f32-igemm/scalar.c.in
//   Generator: tools/xngen
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <xnnpack/igemm.h>
#include <xnnpack/math.h>


void xnn_f32_igemm_minmax_ukernel_4x2__wasm(
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
  assert(mr <= 4);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(float) == 0);
  assert(ks != 0);
  assert(ks % (4 * sizeof(void*)) == 0);
  assert(a_offset % sizeof(float) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  float* c0 = c;
  float* c1 = (float*) ((uintptr_t) c0 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 2) {
    c1 = c0;
  }
  float* c2 = (float*) ((uintptr_t) c1 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 2) {
    c2 = c1;
  }
  float* c3 = (float*) ((uintptr_t) c2 + cm_stride);
  if XNN_UNPREDICTABLE(mr != 4) {
    c3 = c2;
  }

  const float vmin = params->scalar.min;
  const float vmax = params->scalar.max;
  do {
    float vacc00 = w[0];
    float vacc01 = w[1];
    float vacc10 = vacc00;
    float vacc11 = vacc01;
    float vacc20 = vacc00;
    float vacc21 = vacc01;
    float vacc30 = vacc00;
    float vacc31 = vacc01;
    w += 2;

    size_t p = ks;
    do {
      const float* restrict a0 = a[0];
      assert(a0 != NULL);
      if XNN_UNPREDICTABLE(a0 != zero) {
        a0 = (const float*) ((uintptr_t) a0 + a_offset);
      }
      const float* restrict a1 = a[1];
      assert(a1 != NULL);
      if XNN_UNPREDICTABLE(a1 != zero) {
        a1 = (const float*) ((uintptr_t) a1 + a_offset);
      }
      const float* restrict a2 = a[2];
      assert(a2 != NULL);
      if XNN_UNPREDICTABLE(a2 != zero) {
        a2 = (const float*) ((uintptr_t) a2 + a_offset);
      }
      const float* restrict a3 = a[3];
      assert(a3 != NULL);
      if XNN_UNPREDICTABLE(a3 != zero) {
        a3 = (const float*) ((uintptr_t) a3 + a_offset);
      }
      a += 4;

      size_t k = kc;
      do {
        const float va0 = *a0++;
        const float va1 = *a1++;
        const float va2 = *a2++;
        const float va3 = *a3++;

        const float vb0 = w[0];
        const float vb1 = w[1];
        w += 2;

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
      p -= 4 * sizeof(void*);
    } while (p != 0);

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
      c3[0] = vacc30;
      c3[1] = vacc31;
      c3 = (float*) ((uintptr_t) c3 + cn_stride);
      c2[0] = vacc20;
      c2[1] = vacc21;
      c2 = (float*) ((uintptr_t) c2 + cn_stride);
      c1[0] = vacc10;
      c1[1] = vacc11;
      c1 = (float*) ((uintptr_t) c1 + cn_stride);
      c0[0] = vacc00;
      c0[1] = vacc01;
      c0 = (float*) ((uintptr_t) c0 + cn_stride);

      a = (const float**restrict) ((uintptr_t) a - ks);
      nc -= 2;
    } else {
      if (nc & 1) {
        c3[0] = vacc30;
        c2[0] = vacc20;
        c1[0] = vacc10;
        c0[0] = vacc00;
      }

      nc = 0;
    }
  } while (nc != 0);
}
