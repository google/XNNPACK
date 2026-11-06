// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/bf16-f32-igemm/scalar.c.in
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


void xnn_bf16_f32_igemm_minmax_ukernel_1x4c2__scalar(
    size_t mr,
    size_t nc,
    size_t kc,
    size_t ks,
    const xnn_bfloat16** restrict a,
    const void* restrict w,
    float* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    size_t a_offset,
    const xnn_bfloat16* zero,
    const struct xnn_f32_minmax_params* restrict params)
{
  assert(mr != 0);
  assert(mr <= 1);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(xnn_bfloat16) == 0);
  assert(ks != 0);
  assert(ks % (1 * sizeof(void*)) == 0);
  assert(a_offset % sizeof(xnn_bfloat16) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  float* c0 = c;

  const float vmin = params->scalar.min;
  const float vmax = params->scalar.max;
  do {
    const float* wb = (const float*) w;
    float vacc0x0 = wb[0];
    float vacc0x1 = wb[1];
    float vacc0x2 = wb[2];
    float vacc0x3 = wb[3];
    const uint16_t* wk = (const uint16_t*) (wb + 4);

    size_t p = ks;
    do {
      const uint16_t* restrict a0 = (const uint16_t*) a[0];
      assert(a0 != NULL);
      if XNN_UNPREDICTABLE(a0 != (const uint16_t*) zero) {
        a0 = (const uint16_t*) ((uintptr_t) a0 + a_offset);
      }
      a += 1;

      size_t k = kc;
      for (; k >= 2 * sizeof(xnn_bfloat16); k -= 2 * sizeof(xnn_bfloat16)) {
        const float va00 = math_cvt_fp32_bf16(a0[0]);
        const float va01 = math_cvt_fp32_bf16(a0[1]);
        a0 += 2;

        const float vb00 = math_cvt_fp32_bf16(wk[0]);
        const float vb01 = math_cvt_fp32_bf16(wk[1]);
        const float vb10 = math_cvt_fp32_bf16(wk[2]);
        const float vb11 = math_cvt_fp32_bf16(wk[3]);
        const float vb20 = math_cvt_fp32_bf16(wk[4]);
        const float vb21 = math_cvt_fp32_bf16(wk[5]);
        const float vb30 = math_cvt_fp32_bf16(wk[6]);
        const float vb31 = math_cvt_fp32_bf16(wk[7]);
        wk += 8;

        vacc0x0 = math_muladd_f32(va00, vb00, vacc0x0);
        vacc0x0 = math_muladd_f32(va01, vb01, vacc0x0);
        vacc0x1 = math_muladd_f32(va00, vb10, vacc0x1);
        vacc0x1 = math_muladd_f32(va01, vb11, vacc0x1);
        vacc0x2 = math_muladd_f32(va00, vb20, vacc0x2);
        vacc0x2 = math_muladd_f32(va01, vb21, vacc0x2);
        vacc0x3 = math_muladd_f32(va00, vb30, vacc0x3);
        vacc0x3 = math_muladd_f32(va01, vb31, vacc0x3);
      }
      if XNN_UNLIKELY(k != 0) {
        const float va00 = math_cvt_fp32_bf16(a0[0]);

        const float vb00 = math_cvt_fp32_bf16(wk[0]);
        const float vb10 = math_cvt_fp32_bf16(wk[2]);
        const float vb20 = math_cvt_fp32_bf16(wk[4]);
        const float vb30 = math_cvt_fp32_bf16(wk[6]);
        wk += 8;

        vacc0x0 = math_muladd_f32(va00, vb00, vacc0x0);
        vacc0x1 = math_muladd_f32(va00, vb10, vacc0x1);
        vacc0x2 = math_muladd_f32(va00, vb20, vacc0x2);
        vacc0x3 = math_muladd_f32(va00, vb30, vacc0x3);
      }
      p -= 1 * sizeof(void*);
    } while (p != 0);
    w = (const void*) wk;

    vacc0x0 = math_min_f32(math_max_f32(vacc0x0, vmin), vmax);
    vacc0x1 = math_min_f32(math_max_f32(vacc0x1, vmin), vmax);
    vacc0x2 = math_min_f32(math_max_f32(vacc0x2, vmin), vmax);
    vacc0x3 = math_min_f32(math_max_f32(vacc0x3, vmin), vmax);

    if XNN_LIKELY(nc >= 4) {
      c0[0] = vacc0x0;
      c0[1] = vacc0x1;
      c0[2] = vacc0x2;
      c0[3] = vacc0x3;
      c0 = (float*) ((uintptr_t) c0 + cn_stride);

      a = (const xnn_bfloat16**restrict) ((uintptr_t) a - ks);
      nc -= 4;
    } else {
      if (nc & 2) {
        c0[0] = vacc0x0;
        c0[1] = vacc0x1;
        vacc0x0 = vacc0x2;
        c0 += 2;
      }
      if (nc & 1) {
        c0[0] = vacc0x0;
      }

      nc = 0;
    }
  } while (nc != 0);
}
