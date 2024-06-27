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


void xnn_f32_igemm_relu_ukernel_4x4__scalar(
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
    const union xnn_f32_relu_params params[restrict XNN_MIN_ELEMENTS(1)])
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

  do {
    float vacc00 = w[0];
    float vacc01 = w[1];
    float vacc02 = w[2];
    float vacc03 = w[3];
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
    w += 4;

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
      p -= 4 * sizeof(void*);
    } while (p != 0);

    vacc00 = math_max_f32(vacc00, 0.0f);
    vacc01 = math_max_f32(vacc01, 0.0f);
    vacc02 = math_max_f32(vacc02, 0.0f);
    vacc03 = math_max_f32(vacc03, 0.0f);
    vacc10 = math_max_f32(vacc10, 0.0f);
    vacc11 = math_max_f32(vacc11, 0.0f);
    vacc12 = math_max_f32(vacc12, 0.0f);
    vacc13 = math_max_f32(vacc13, 0.0f);
    vacc20 = math_max_f32(vacc20, 0.0f);
    vacc21 = math_max_f32(vacc21, 0.0f);
    vacc22 = math_max_f32(vacc22, 0.0f);
    vacc23 = math_max_f32(vacc23, 0.0f);
    vacc30 = math_max_f32(vacc30, 0.0f);
    vacc31 = math_max_f32(vacc31, 0.0f);
    vacc32 = math_max_f32(vacc32, 0.0f);
    vacc33 = math_max_f32(vacc33, 0.0f);

    if XNN_LIKELY(nc >= 4) {
      c3[0] = vacc30;
      c3[1] = vacc31;
      c3[2] = vacc32;
      c3[3] = vacc33;
      c3 = (float*) ((uintptr_t) c3 + cn_stride);
      c2[0] = vacc20;
      c2[1] = vacc21;
      c2[2] = vacc22;
      c2[3] = vacc23;
      c2 = (float*) ((uintptr_t) c2 + cn_stride);
      c1[0] = vacc10;
      c1[1] = vacc11;
      c1[2] = vacc12;
      c1[3] = vacc13;
      c1 = (float*) ((uintptr_t) c1 + cn_stride);
      c0[0] = vacc00;
      c0[1] = vacc01;
      c0[2] = vacc02;
      c0[3] = vacc03;
      c0 = (float*) ((uintptr_t) c0 + cn_stride);

      a = (const float**restrict) ((uintptr_t) a - ks);
      nc -= 4;
    } else {
      if (nc & 2) {
        c3[0] = vacc30;
        c3[1] = vacc31;
        vacc30 = vacc32;
        c3 += 2;
        c2[0] = vacc20;
        c2[1] = vacc21;
        vacc20 = vacc22;
        c2 += 2;
        c1[0] = vacc10;
        c1[1] = vacc11;
        vacc10 = vacc12;
        c1 += 2;
        c0[0] = vacc00;
        c0[1] = vacc01;
        vacc00 = vacc02;
        c0 += 2;
      }
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
