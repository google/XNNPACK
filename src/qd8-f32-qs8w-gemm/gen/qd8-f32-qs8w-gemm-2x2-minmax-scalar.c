// Auto-generated file. Do not edit!
//   Template: src/qs8-gemm/scalar.c.in
//   Generator: tools/xngen
//
// Copyright 2021 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <xnnpack/gemm.h>
#include <xnnpack/math.h>
#include <xnnpack/unaligned.h>


void xnn_qd8_f32_qs8w_gemm_minmax_ukernel_2x2__scalar(
    size_t mr,
    size_t nc,
    size_t kc,
    const int8_t* restrict a,
    size_t a_stride,
    const void* restrict w,
    float* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const struct xnn_qd8_quantization_params quantization_params[restrict XNN_MIN_ELEMENTS(1)],
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(mr != 0);
  assert(mr <= 2);
  assert(nc != 0);
  assert(kc != 0);

  const int8_t* a0 = a;
  float* c0 = c;
  const int8_t* a1 = (const int8_t*) ((uintptr_t) a0 + a_stride);
  float* c1 = (float*) ((uintptr_t) c0 + cm_stride);
  if XNN_UNPREDICTABLE(mr != 2) {
    a1 = a0;
    c1 = c0;
  }

  const int32_t vzp0 = quantization_params[0].zero_point;
  const int32_t vzp1 = quantization_params[1].zero_point;
  do {
    int32_t vacc0x0 = unaligned_indexed_load_s32(w, 0);
    int32_t vacc0x1 = unaligned_indexed_load_s32(w, 1);
    int32_t vacc1x0 = vacc0x0;
    int32_t vacc1x1 = vacc0x1;
    w = (const int32_t*) w + 2;
    vacc0x0 *= vzp0;
    vacc0x1 *= vzp0;
    vacc1x0 *= vzp1;
    vacc1x1 *= vzp1;

    size_t k = kc;
    do {
      const int32_t va0 = (int32_t) *a0++;
      const int32_t va1 = (int32_t) *a1++;

      const int32_t vb0 = (int32_t) ((const int8_t*) w)[0];
      const int32_t vb1 = (int32_t) ((const int8_t*) w)[1];
      w = (const int8_t*) w + 2;

      vacc0x0 += va0 * vb0;
      vacc0x1 += va0 * vb1;
      vacc1x0 += va1 * vb0;
      vacc1x1 += va1 * vb1;

      k -= sizeof(int8_t);
    } while (k != 0);

    float vout0x0 = (float) vacc0x0;
    float vout0x1 = (float) vacc0x1;
    float vout1x0 = (float) vacc1x0;
    float vout1x1 = (float) vacc1x1;
    const float vscale0 = quantization_params[0].scale;
    const float vscale1 = quantization_params[1].scale;
    vout0x0 *= vscale0;
    vout1x0 *= vscale1;
    vout0x1 *= vscale0;
    vout1x1 *= vscale1;
    const float vbias0 = unaligned_indexed_load_f32(w, 0);
    const float vbias1 = unaligned_indexed_load_f32(w, 1);
    w = (const float*) w + 2;
    const float vmax = params->scalar.max;
    const float vmin = params->scalar.min;
    vout0x0 += vbias0;
    vout0x0 = math_max_f32(vout0x0, vmin);
    vout0x0 = math_min_f32(vout0x0, vmax);
    vout1x0 += vbias0;
    vout1x0 = math_max_f32(vout1x0, vmin);
    vout1x0 = math_min_f32(vout1x0, vmax);
    vout0x1 += vbias1;
    vout0x1 = math_max_f32(vout0x1, vmin);
    vout0x1 = math_min_f32(vout0x1, vmax);
    vout1x1 += vbias1;
    vout1x1 = math_max_f32(vout1x1, vmin);
    vout1x1 = math_min_f32(vout1x1, vmax);

    if XNN_LIKELY(nc >= 2) {
      c0[0] = vout0x0;
      c0[1] = vout0x1;
      c1[0] = vout1x0;
      c1[1] = vout1x1;

      a0 = (const int8_t*) ((uintptr_t) a0 - kc);
      a1 = (const int8_t*) ((uintptr_t) a1 - kc);

      c0 = (float*) ((uintptr_t) c0 + cn_stride);
      c1 = (float*) ((uintptr_t) c1 + cn_stride);

      nc -= 2;
    } else {
      if (nc & 1) {
        c0[0] = (float) vout0x0;
        c1[0] = (float) vout1x0;
      }

      nc = 0;
    }
  } while (nc != 0);
}
