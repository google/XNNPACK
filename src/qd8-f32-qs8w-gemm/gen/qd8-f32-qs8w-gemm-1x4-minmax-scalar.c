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


void xnn_qd8_f32_qs8w_gemm_minmax_ukernel_1x4__scalar(
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
  assert(mr <= 1);
  assert(nc != 0);
  assert(kc != 0);

  const int8_t* a0 = a;
  float* c0 = c;

  const int32_t vzp0 = quantization_params[0].zero_point;
  do {
    int32_t vacc0x0 = ((const int32_t*) w)[0];
    int32_t vacc0x1 = ((const int32_t*) w)[1];
    int32_t vacc0x2 = ((const int32_t*) w)[2];
    int32_t vacc0x3 = ((const int32_t*) w)[3];
    w = (const int32_t*) w + 4;
    vacc0x0 *= vzp0;
    vacc0x1 *= vzp0;
    vacc0x2 *= vzp0;
    vacc0x3 *= vzp0;

    size_t k = kc;
    do {
      const int32_t va0 = (int32_t) *a0++;

      const int32_t vb0 = (int32_t) ((const int8_t*) w)[0];
      const int32_t vb1 = (int32_t) ((const int8_t*) w)[1];
      const int32_t vb2 = (int32_t) ((const int8_t*) w)[2];
      const int32_t vb3 = (int32_t) ((const int8_t*) w)[3];
      w = (const int8_t*) w + 4;

      vacc0x0 += va0 * vb0;
      vacc0x1 += va0 * vb1;
      vacc0x2 += va0 * vb2;
      vacc0x3 += va0 * vb3;

      k -= sizeof(int8_t);
    } while (k != 0);

    float vout0x0 = (float) vacc0x0;
    float vout0x1 = (float) vacc0x1;
    float vout0x2 = (float) vacc0x2;
    float vout0x3 = (float) vacc0x3;
    const float vscale0 = quantization_params[0].scale;
    vout0x0 *= vscale0;
    vout0x1 *= vscale0;
    vout0x2 *= vscale0;
    vout0x3 *= vscale0;
    const float vbias0 = ((const float*) w)[0];
    const float vbias1 = ((const float*) w)[1];
    const float vbias2 = ((const float*) w)[2];
    const float vbias3 = ((const float*) w)[3];
    w = (const float*) w + 4;
    const float vmax = params->scalar.max;
    const float vmin = params->scalar.min;
    vout0x0 += vbias0;
    vout0x0 = math_max_f32(vout0x0, vmin);
    vout0x0 = math_min_f32(vout0x0, vmax);
    vout0x1 += vbias1;
    vout0x1 = math_max_f32(vout0x1, vmin);
    vout0x1 = math_min_f32(vout0x1, vmax);
    vout0x2 += vbias2;
    vout0x2 = math_max_f32(vout0x2, vmin);
    vout0x2 = math_min_f32(vout0x2, vmax);
    vout0x3 += vbias3;
    vout0x3 = math_max_f32(vout0x3, vmin);
    vout0x3 = math_min_f32(vout0x3, vmax);

    if XNN_LIKELY(nc >= 4) {
      c0[0] = vout0x0;
      c0[1] = vout0x1;
      c0[2] = vout0x2;
      c0[3] = vout0x3;

      a0 = (const int8_t*) ((uintptr_t) a0 - kc);

      c0 = (float*) ((uintptr_t) c0 + cn_stride);

      nc -= 4;
    } else {
      if (nc & 2) {
        c0[0] = (float) vout0x0;
        c0[1] = (float) vout0x1;
        vout0x0 = vout0x2;
        c0 += 2;
      }
      if (nc & 1) {
        c0[0] = (float) vout0x0;
      }

      nc = 0;
    }
  } while (nc != 0);
}
