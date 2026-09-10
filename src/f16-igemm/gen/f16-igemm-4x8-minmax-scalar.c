// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/f16-igemm/scalar.c.in
//   Generator: tools/xngen
//
// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include "src/xnnpack/common.h"
#include "src/xnnpack/igemm.h"
#include "src/xnnpack/math.h"
#include "src/xnnpack/microparams.h"


void xnn_f16_igemm_minmax_ukernel_4x8__scalar(
    size_t mr,
    size_t nc,
    size_t kc,
    size_t ks,
    const xnn_float16** restrict a,
    const xnn_float16* restrict w,
    xnn_float16* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    size_t a_offset,
    const xnn_float16* zero,
    const struct xnn_f16_minmax_params* restrict params)
{
  assert(mr != 0);
  assert(mr <= 4);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(xnn_float16) == 0);
  assert(ks != 0);
  assert(ks % (4 * sizeof(void*)) == 0);
  assert(a_offset % sizeof(xnn_float16) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  xnn_float16* c0 = c;
  xnn_float16* c1 = (xnn_float16*) ((uintptr_t) c0 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 2) {
    c1 = c0;
  }
  xnn_float16* c2 = (xnn_float16*) ((uintptr_t) c1 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 2) {
    c2 = c1;
  }
  xnn_float16* c3 = (xnn_float16*) ((uintptr_t) c2 + cm_stride);
  if XNN_UNPREDICTABLE(mr != 4) {
    c3 = c2;
  }

  const float vmin = xnn_float16_to_float(params->scalar.min);
  const float vmax = xnn_float16_to_float(params->scalar.max);
  do {
    float vacc00 = xnn_float16_to_float(w[0]);
    float vacc01 = xnn_float16_to_float(w[1]);
    float vacc02 = xnn_float16_to_float(w[2]);
    float vacc03 = xnn_float16_to_float(w[3]);
    float vacc04 = xnn_float16_to_float(w[4]);
    float vacc05 = xnn_float16_to_float(w[5]);
    float vacc06 = xnn_float16_to_float(w[6]);
    float vacc07 = xnn_float16_to_float(w[7]);
    float vacc10 = vacc00;
    float vacc11 = vacc01;
    float vacc12 = vacc02;
    float vacc13 = vacc03;
    float vacc14 = vacc04;
    float vacc15 = vacc05;
    float vacc16 = vacc06;
    float vacc17 = vacc07;
    float vacc20 = vacc00;
    float vacc21 = vacc01;
    float vacc22 = vacc02;
    float vacc23 = vacc03;
    float vacc24 = vacc04;
    float vacc25 = vacc05;
    float vacc26 = vacc06;
    float vacc27 = vacc07;
    float vacc30 = vacc00;
    float vacc31 = vacc01;
    float vacc32 = vacc02;
    float vacc33 = vacc03;
    float vacc34 = vacc04;
    float vacc35 = vacc05;
    float vacc36 = vacc06;
    float vacc37 = vacc07;
    w += 8;

    size_t p = ks;
    do {
      const xnn_float16* restrict a0 = a[0];
      assert(a0 != NULL);
      if XNN_UNPREDICTABLE(a0 != zero) {
        a0 = (const xnn_float16*) ((uintptr_t) a0 + a_offset);
      }
      const xnn_float16* restrict a1 = a[1];
      assert(a1 != NULL);
      if XNN_UNPREDICTABLE(a1 != zero) {
        a1 = (const xnn_float16*) ((uintptr_t) a1 + a_offset);
      }
      const xnn_float16* restrict a2 = a[2];
      assert(a2 != NULL);
      if XNN_UNPREDICTABLE(a2 != zero) {
        a2 = (const xnn_float16*) ((uintptr_t) a2 + a_offset);
      }
      const xnn_float16* restrict a3 = a[3];
      assert(a3 != NULL);
      if XNN_UNPREDICTABLE(a3 != zero) {
        a3 = (const xnn_float16*) ((uintptr_t) a3 + a_offset);
      }
      a += 4;

      size_t k = kc;
      do {
        const float va0 = xnn_float16_to_float(*a0++);
        const float va1 = xnn_float16_to_float(*a1++);
        const float va2 = xnn_float16_to_float(*a2++);
        const float va3 = xnn_float16_to_float(*a3++);

        const float vb0 = xnn_float16_to_float(w[0]);
        const float vb1 = xnn_float16_to_float(w[1]);
        const float vb2 = xnn_float16_to_float(w[2]);
        const float vb3 = xnn_float16_to_float(w[3]);
        const float vb4 = xnn_float16_to_float(w[4]);
        const float vb5 = xnn_float16_to_float(w[5]);
        const float vb6 = xnn_float16_to_float(w[6]);
        const float vb7 = xnn_float16_to_float(w[7]);
        w += 8;

        vacc00 = math_muladd_f32(va0, vb0, vacc00);
        vacc01 = math_muladd_f32(va0, vb1, vacc01);
        vacc02 = math_muladd_f32(va0, vb2, vacc02);
        vacc03 = math_muladd_f32(va0, vb3, vacc03);
        vacc04 = math_muladd_f32(va0, vb4, vacc04);
        vacc05 = math_muladd_f32(va0, vb5, vacc05);
        vacc06 = math_muladd_f32(va0, vb6, vacc06);
        vacc07 = math_muladd_f32(va0, vb7, vacc07);
        vacc10 = math_muladd_f32(va1, vb0, vacc10);
        vacc11 = math_muladd_f32(va1, vb1, vacc11);
        vacc12 = math_muladd_f32(va1, vb2, vacc12);
        vacc13 = math_muladd_f32(va1, vb3, vacc13);
        vacc14 = math_muladd_f32(va1, vb4, vacc14);
        vacc15 = math_muladd_f32(va1, vb5, vacc15);
        vacc16 = math_muladd_f32(va1, vb6, vacc16);
        vacc17 = math_muladd_f32(va1, vb7, vacc17);
        vacc20 = math_muladd_f32(va2, vb0, vacc20);
        vacc21 = math_muladd_f32(va2, vb1, vacc21);
        vacc22 = math_muladd_f32(va2, vb2, vacc22);
        vacc23 = math_muladd_f32(va2, vb3, vacc23);
        vacc24 = math_muladd_f32(va2, vb4, vacc24);
        vacc25 = math_muladd_f32(va2, vb5, vacc25);
        vacc26 = math_muladd_f32(va2, vb6, vacc26);
        vacc27 = math_muladd_f32(va2, vb7, vacc27);
        vacc30 = math_muladd_f32(va3, vb0, vacc30);
        vacc31 = math_muladd_f32(va3, vb1, vacc31);
        vacc32 = math_muladd_f32(va3, vb2, vacc32);
        vacc33 = math_muladd_f32(va3, vb3, vacc33);
        vacc34 = math_muladd_f32(va3, vb4, vacc34);
        vacc35 = math_muladd_f32(va3, vb5, vacc35);
        vacc36 = math_muladd_f32(va3, vb6, vacc36);
        vacc37 = math_muladd_f32(va3, vb7, vacc37);

        k -= sizeof(xnn_float16);
      } while (k != 0);
      p -= 4 * sizeof(void*);
    } while (p != 0);

    vacc00 = math_max_f32(vacc00, vmin);
    vacc01 = math_max_f32(vacc01, vmin);
    vacc02 = math_max_f32(vacc02, vmin);
    vacc03 = math_max_f32(vacc03, vmin);
    vacc04 = math_max_f32(vacc04, vmin);
    vacc05 = math_max_f32(vacc05, vmin);
    vacc06 = math_max_f32(vacc06, vmin);
    vacc07 = math_max_f32(vacc07, vmin);
    vacc10 = math_max_f32(vacc10, vmin);
    vacc11 = math_max_f32(vacc11, vmin);
    vacc12 = math_max_f32(vacc12, vmin);
    vacc13 = math_max_f32(vacc13, vmin);
    vacc14 = math_max_f32(vacc14, vmin);
    vacc15 = math_max_f32(vacc15, vmin);
    vacc16 = math_max_f32(vacc16, vmin);
    vacc17 = math_max_f32(vacc17, vmin);
    vacc20 = math_max_f32(vacc20, vmin);
    vacc21 = math_max_f32(vacc21, vmin);
    vacc22 = math_max_f32(vacc22, vmin);
    vacc23 = math_max_f32(vacc23, vmin);
    vacc24 = math_max_f32(vacc24, vmin);
    vacc25 = math_max_f32(vacc25, vmin);
    vacc26 = math_max_f32(vacc26, vmin);
    vacc27 = math_max_f32(vacc27, vmin);
    vacc30 = math_max_f32(vacc30, vmin);
    vacc31 = math_max_f32(vacc31, vmin);
    vacc32 = math_max_f32(vacc32, vmin);
    vacc33 = math_max_f32(vacc33, vmin);
    vacc34 = math_max_f32(vacc34, vmin);
    vacc35 = math_max_f32(vacc35, vmin);
    vacc36 = math_max_f32(vacc36, vmin);
    vacc37 = math_max_f32(vacc37, vmin);

    vacc00 = math_min_f32(vacc00, vmax);
    vacc01 = math_min_f32(vacc01, vmax);
    vacc02 = math_min_f32(vacc02, vmax);
    vacc03 = math_min_f32(vacc03, vmax);
    vacc04 = math_min_f32(vacc04, vmax);
    vacc05 = math_min_f32(vacc05, vmax);
    vacc06 = math_min_f32(vacc06, vmax);
    vacc07 = math_min_f32(vacc07, vmax);
    vacc10 = math_min_f32(vacc10, vmax);
    vacc11 = math_min_f32(vacc11, vmax);
    vacc12 = math_min_f32(vacc12, vmax);
    vacc13 = math_min_f32(vacc13, vmax);
    vacc14 = math_min_f32(vacc14, vmax);
    vacc15 = math_min_f32(vacc15, vmax);
    vacc16 = math_min_f32(vacc16, vmax);
    vacc17 = math_min_f32(vacc17, vmax);
    vacc20 = math_min_f32(vacc20, vmax);
    vacc21 = math_min_f32(vacc21, vmax);
    vacc22 = math_min_f32(vacc22, vmax);
    vacc23 = math_min_f32(vacc23, vmax);
    vacc24 = math_min_f32(vacc24, vmax);
    vacc25 = math_min_f32(vacc25, vmax);
    vacc26 = math_min_f32(vacc26, vmax);
    vacc27 = math_min_f32(vacc27, vmax);
    vacc30 = math_min_f32(vacc30, vmax);
    vacc31 = math_min_f32(vacc31, vmax);
    vacc32 = math_min_f32(vacc32, vmax);
    vacc33 = math_min_f32(vacc33, vmax);
    vacc34 = math_min_f32(vacc34, vmax);
    vacc35 = math_min_f32(vacc35, vmax);
    vacc36 = math_min_f32(vacc36, vmax);
    vacc37 = math_min_f32(vacc37, vmax);
    if XNN_LIKELY(nc >= 8) {
      c3[0] = xnn_float16_from_float(vacc30);
      c3[1] = xnn_float16_from_float(vacc31);
      c3[2] = xnn_float16_from_float(vacc32);
      c3[3] = xnn_float16_from_float(vacc33);
      c3[4] = xnn_float16_from_float(vacc34);
      c3[5] = xnn_float16_from_float(vacc35);
      c3[6] = xnn_float16_from_float(vacc36);
      c3[7] = xnn_float16_from_float(vacc37);
      c3 = (xnn_float16*) ((uintptr_t) c3 + cn_stride);
      c2[0] = xnn_float16_from_float(vacc20);
      c2[1] = xnn_float16_from_float(vacc21);
      c2[2] = xnn_float16_from_float(vacc22);
      c2[3] = xnn_float16_from_float(vacc23);
      c2[4] = xnn_float16_from_float(vacc24);
      c2[5] = xnn_float16_from_float(vacc25);
      c2[6] = xnn_float16_from_float(vacc26);
      c2[7] = xnn_float16_from_float(vacc27);
      c2 = (xnn_float16*) ((uintptr_t) c2 + cn_stride);
      c1[0] = xnn_float16_from_float(vacc10);
      c1[1] = xnn_float16_from_float(vacc11);
      c1[2] = xnn_float16_from_float(vacc12);
      c1[3] = xnn_float16_from_float(vacc13);
      c1[4] = xnn_float16_from_float(vacc14);
      c1[5] = xnn_float16_from_float(vacc15);
      c1[6] = xnn_float16_from_float(vacc16);
      c1[7] = xnn_float16_from_float(vacc17);
      c1 = (xnn_float16*) ((uintptr_t) c1 + cn_stride);
      c0[0] = xnn_float16_from_float(vacc00);
      c0[1] = xnn_float16_from_float(vacc01);
      c0[2] = xnn_float16_from_float(vacc02);
      c0[3] = xnn_float16_from_float(vacc03);
      c0[4] = xnn_float16_from_float(vacc04);
      c0[5] = xnn_float16_from_float(vacc05);
      c0[6] = xnn_float16_from_float(vacc06);
      c0[7] = xnn_float16_from_float(vacc07);
      c0 = (xnn_float16*) ((uintptr_t) c0 + cn_stride);

      a = (const xnn_float16**restrict) ((uintptr_t) a - ks);
      nc -= 8;
    } else {
      if (nc & 4) {
        c3[0] = xnn_float16_from_float(vacc30);
        c3[1] = xnn_float16_from_float(vacc31);
        c3[2] = xnn_float16_from_float(vacc32);
        c3[3] = xnn_float16_from_float(vacc33);
        vacc30 = vacc34;
        vacc31 = vacc35;
        vacc32 = vacc36;
        c3 += 4;
        c2[0] = xnn_float16_from_float(vacc20);
        c2[1] = xnn_float16_from_float(vacc21);
        c2[2] = xnn_float16_from_float(vacc22);
        c2[3] = xnn_float16_from_float(vacc23);
        vacc20 = vacc24;
        vacc21 = vacc25;
        vacc22 = vacc26;
        c2 += 4;
        c1[0] = xnn_float16_from_float(vacc10);
        c1[1] = xnn_float16_from_float(vacc11);
        c1[2] = xnn_float16_from_float(vacc12);
        c1[3] = xnn_float16_from_float(vacc13);
        vacc10 = vacc14;
        vacc11 = vacc15;
        vacc12 = vacc16;
        c1 += 4;
        c0[0] = xnn_float16_from_float(vacc00);
        c0[1] = xnn_float16_from_float(vacc01);
        c0[2] = xnn_float16_from_float(vacc02);
        c0[3] = xnn_float16_from_float(vacc03);
        vacc00 = vacc04;
        vacc01 = vacc05;
        vacc02 = vacc06;
        c0 += 4;
      }
      if (nc & 2) {
        c3[0] = xnn_float16_from_float(vacc30);
        c3[1] = xnn_float16_from_float(vacc31);
        vacc30 = vacc32;
        vacc31 = vacc33;
        vacc32 = vacc34;
        vacc33 = vacc35;
        vacc34 = vacc36;
        c3 += 2;
        c2[0] = xnn_float16_from_float(vacc20);
        c2[1] = xnn_float16_from_float(vacc21);
        vacc20 = vacc22;
        vacc21 = vacc23;
        vacc22 = vacc24;
        vacc23 = vacc25;
        vacc24 = vacc26;
        c2 += 2;
        c1[0] = xnn_float16_from_float(vacc10);
        c1[1] = xnn_float16_from_float(vacc11);
        vacc10 = vacc12;
        vacc11 = vacc13;
        vacc12 = vacc14;
        vacc13 = vacc15;
        vacc14 = vacc16;
        c1 += 2;
        c0[0] = xnn_float16_from_float(vacc00);
        c0[1] = xnn_float16_from_float(vacc01);
        vacc00 = vacc02;
        vacc01 = vacc03;
        vacc02 = vacc04;
        vacc03 = vacc05;
        vacc04 = vacc06;
        c0 += 2;
      }
      if (nc & 1) {
        c3[0] = xnn_float16_from_float(vacc30);
        c2[0] = xnn_float16_from_float(vacc20);
        c1[0] = xnn_float16_from_float(vacc10);
        c0[0] = xnn_float16_from_float(vacc00);
      }

      nc = 0;
    }
  } while (nc != 0);
}
