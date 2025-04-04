// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include "src/xnnpack/gemm.h"
#include "src/xnnpack/math.h"


void xnn_bf16_f32_gemm_minmax_ukernel_1x4c2__scalar(
    size_t mr,
    size_t nc,
    size_t kc,
    const uint16_t* restrict a,
    size_t a_stride,
    const void* restrict w,
    float* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const struct xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(mr != 0);
  assert(mr == 1);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(xnn_bfloat16) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  const uint16_t* a0 = a;
  float* c0 = c;

  const float vmin = params->scalar.min;
  const float vmax = params->scalar.max;
  do {
    const float* w_float = (const float*) w;
    float vacc00 = w_float[0];
    float vacc01 = w_float[1];
    float vacc02 = w_float[2];
    float vacc03 = w_float[3];
    w = (const void*) ((uintptr_t) w + 4 * sizeof(float));

    size_t k = kc;
    for (; k >= 2 * sizeof(xnn_bfloat16); k -= 2 * sizeof(xnn_bfloat16)) {
      const float va00 = math_cvt_fp32_bf16(*a0++);
      const float va01 = math_cvt_fp32_bf16(*a0++);


      const uint16_t* w_uint16_t = (const uint16_t*) w;
      const float vb00 = math_cvt_fp32_bf16(w_uint16_t[0]);
      const float vb01 = math_cvt_fp32_bf16(w_uint16_t[1]);
      const float vb10 = math_cvt_fp32_bf16(w_uint16_t[2]);
      const float vb11 = math_cvt_fp32_bf16(w_uint16_t[3]);
      const float vb20 = math_cvt_fp32_bf16(w_uint16_t[4]);
      const float vb21 = math_cvt_fp32_bf16(w_uint16_t[5]);
      const float vb30 = math_cvt_fp32_bf16(w_uint16_t[6]);
      const float vb31 = math_cvt_fp32_bf16(w_uint16_t[7]);
      w = (const void*) ((uintptr_t) w + 8 * sizeof(uint16_t));

      vacc00 = math_muladd_f32(va00, vb00, vacc00);
      vacc00 = math_muladd_f32(va01, vb01, vacc00);

      vacc01 = math_muladd_f32(va00, vb10, vacc01);
      vacc01 = math_muladd_f32(va01, vb11, vacc01);

      vacc02 = math_muladd_f32(va00, vb20, vacc02);
      vacc02 = math_muladd_f32(va01, vb21, vacc02);

      vacc03 = math_muladd_f32(va00, vb30, vacc03);
      vacc03 = math_muladd_f32(va01, vb31, vacc03);
    }

    if (k != 0) {
      const float va00 = math_cvt_fp32_bf16(*a0++);

      const uint16_t* w_uint16_t = (const uint16_t*) w;
      const float vb00 = math_cvt_fp32_bf16(w_uint16_t[0]);
      const float vb10 = math_cvt_fp32_bf16(w_uint16_t[2]);
      const float vb20 = math_cvt_fp32_bf16(w_uint16_t[4]);
      const float vb30 = math_cvt_fp32_bf16(w_uint16_t[6]);
      w = (const void*) ((uintptr_t) w + 8 * sizeof(uint16_t));

      vacc00 = math_muladd_f32(va00, vb00, vacc00);

      vacc01 = math_muladd_f32(va00, vb10, vacc01);

      vacc02 = math_muladd_f32(va00, vb20, vacc02);

      vacc03 = math_muladd_f32(va00, vb30, vacc03);
    }

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
      c0 += 4;

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

