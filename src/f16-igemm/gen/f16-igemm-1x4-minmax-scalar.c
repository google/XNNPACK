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


void xnn_f16_igemm_minmax_ukernel_1x4__scalar(
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
  assert(mr <= 1);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(xnn_float16) == 0);
  assert(ks != 0);
  assert(ks % (1 * sizeof(void*)) == 0);
  assert(a_offset % sizeof(xnn_float16) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  xnn_float16* c0 = c;

  const float vmin = xnn_float16_to_float(params->scalar.min);
  const float vmax = xnn_float16_to_float(params->scalar.max);
  do {
    float vacc00 = xnn_float16_to_float(w[0]);
    float vacc01 = xnn_float16_to_float(w[1]);
    float vacc02 = xnn_float16_to_float(w[2]);
    float vacc03 = xnn_float16_to_float(w[3]);
    w += 4;

    size_t p = ks;
    do {
      const xnn_float16* restrict a0 = a[0];
      assert(a0 != NULL);
      if XNN_UNPREDICTABLE(a0 != zero) {
        a0 = (const xnn_float16*) ((uintptr_t) a0 + a_offset);
      }
      a += 1;

      size_t k = kc;
      do {
        const float va0 = xnn_float16_to_float(*a0++);

        const float vb0 = xnn_float16_to_float(w[0]);
        const float vb1 = xnn_float16_to_float(w[1]);
        const float vb2 = xnn_float16_to_float(w[2]);
        const float vb3 = xnn_float16_to_float(w[3]);
        w += 4;

        vacc00 = math_muladd_f32(va0, vb0, vacc00);
        vacc01 = math_muladd_f32(va0, vb1, vacc01);
        vacc02 = math_muladd_f32(va0, vb2, vacc02);
        vacc03 = math_muladd_f32(va0, vb3, vacc03);

        k -= sizeof(xnn_float16);
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
      c0[0] = xnn_float16_from_float(vacc00);
      c0[1] = xnn_float16_from_float(vacc01);
      c0[2] = xnn_float16_from_float(vacc02);
      c0[3] = xnn_float16_from_float(vacc03);
      c0 = (xnn_float16*) ((uintptr_t) c0 + cn_stride);

      a = (const xnn_float16**restrict) ((uintptr_t) a - ks);
      nc -= 4;
    } else {
      if (nc & 2) {
        c0[0] = xnn_float16_from_float(vacc00);
        c0[1] = xnn_float16_from_float(vacc01);
        vacc00 = vacc02;
        c0 += 2;
      }
      if (nc & 1) {
        c0[0] = xnn_float16_from_float(vacc00);
      }

      nc = 0;
    }
  } while (nc != 0);
}
