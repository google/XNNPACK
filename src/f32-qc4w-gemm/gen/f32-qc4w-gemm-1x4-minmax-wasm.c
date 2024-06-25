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


void xnn_f32_qc4w_gemm_minmax_ukernel_1x4__wasm(
    size_t mr,
    size_t nc,
    size_t kc,
    const float* restrict a,
    size_t a_stride,
    const void* restrict w,
    float* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_f32_qc4w_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
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
  const int32_t vminus_kernel_zero_point = params->scalar.minus_kernel_zero_point;
  do {
    float vacc00 = ((const float*)w)[0];
    float vacc01 = ((const float*)w)[1];
    float vacc02 = ((const float*)w)[2];
    float vacc03 = ((const float*)w)[3];
    w = (const float*) w + 4;

    size_t k = kc;
    for (; k >= 2 * sizeof(float); k -= 2 * sizeof(float)) {
      const float va00 = *a0++;
      const float va01 = *a0++;

      const uint8_t vbi0 = ((const uint8_t*) w)[0];
      const uint8_t vbi1 = ((const uint8_t*) w)[1];
      const uint8_t vbi2 = ((const uint8_t*) w)[2];
      const uint8_t vbi3 = ((const uint8_t*) w)[3];
      const float vb00 = (float) ((int32_t) (vbi0 & 0xF) + vminus_kernel_zero_point);
      const float vb10 = (float) ((int32_t) (vbi1 & 0xF) + vminus_kernel_zero_point);
      const float vb20 = (float) ((int32_t) (vbi2 & 0xF) + vminus_kernel_zero_point);
      const float vb30 = (float) ((int32_t) (vbi3 & 0xF) + vminus_kernel_zero_point);
      const float vb01 = (float) ((int32_t) (vbi0 >> 4) + vminus_kernel_zero_point);
      const float vb11 = (float) ((int32_t) (vbi1 >> 4) + vminus_kernel_zero_point);
      const float vb21 = (float) ((int32_t) (vbi2 >> 4) + vminus_kernel_zero_point);
      const float vb31 = (float) ((int32_t) (vbi3 >> 4) + vminus_kernel_zero_point);
      w = (const int8_t*) w + 4;

      vacc00 = math_muladd_f32(va00, vb00, vacc00);
      vacc01 = math_muladd_f32(va00, vb10, vacc01);
      vacc02 = math_muladd_f32(va00, vb20, vacc02);
      vacc03 = math_muladd_f32(va00, vb30, vacc03);
      vacc00 = math_muladd_f32(va01, vb01, vacc00);
      vacc01 = math_muladd_f32(va01, vb11, vacc01);
      vacc02 = math_muladd_f32(va01, vb21, vacc02);
      vacc03 = math_muladd_f32(va01, vb31, vacc03);
    }
    if XNN_UNLIKELY(k != 0) {
      const float va0 = *a0++;

      const uint8_t vbi0 = ((const uint8_t*) w)[0];
      const uint8_t vbi1 = ((const uint8_t*) w)[1];
      const uint8_t vbi2 = ((const uint8_t*) w)[2];
      const uint8_t vbi3 = ((const uint8_t*) w)[3];
      const float vb0 = (float) ((int32_t) vbi0 + vminus_kernel_zero_point);
      const float vb1 = (float) ((int32_t) vbi1 + vminus_kernel_zero_point);
      const float vb2 = (float) ((int32_t) vbi2 + vminus_kernel_zero_point);
      const float vb3 = (float) ((int32_t) vbi3 + vminus_kernel_zero_point);
      w = (const int8_t*) w + 4;

      vacc00 = math_muladd_f32(va0, vb0, vacc00);
      vacc01 = math_muladd_f32(va0, vb1, vacc01);
      vacc02 = math_muladd_f32(va0, vb2, vacc02);
      vacc03 = math_muladd_f32(va0, vb3, vacc03);
    }

    const float vscale0 = ((const float*)w)[0];
    const float vscale1 = ((const float*)w)[1];
    const float vscale2 = ((const float*)w)[2];
    const float vscale3 = ((const float*)w)[3];
    w = (const float*) w + 4;
    vacc00 *= vscale0;
    vacc01 *= vscale1;
    vacc02 *= vscale2;
    vacc03 *= vscale3;
    vacc00 = __builtin_wasm_max_f32(vacc00, vmin);
    vacc01 = __builtin_wasm_max_f32(vacc01, vmin);
    vacc02 = __builtin_wasm_max_f32(vacc02, vmin);
    vacc03 = __builtin_wasm_max_f32(vacc03, vmin);

    vacc00 = __builtin_wasm_min_f32(vacc00, vmax);
    vacc01 = __builtin_wasm_min_f32(vacc01, vmax);
    vacc02 = __builtin_wasm_min_f32(vacc02, vmax);
    vacc03 = __builtin_wasm_min_f32(vacc03, vmax);

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
