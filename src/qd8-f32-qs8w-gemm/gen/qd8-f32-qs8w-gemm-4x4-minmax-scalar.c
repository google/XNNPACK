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


void xnn_qd8_f32_qs8w_gemm_minmax_ukernel_4x4__scalar(
    size_t mr,
    size_t nc,
    size_t kc,
    const int8_t* restrict a,
    size_t a_stride,
    const void* restrict w,
    float* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)],
    const struct xnn_qd8_quantization_params quantization_params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(mr != 0);
  assert(mr <= 4);
  assert(nc != 0);
  assert(kc != 0);

  const int8_t* a0 = a;
  float* c0 = c;
  const int8_t* a1 = (const int8_t*) ((uintptr_t) a0 + a_stride);
  float* c1 = (float*) ((uintptr_t) c0 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 2) {
    a1 = a0;
    c1 = c0;
  }
  const int8_t* a2 = (const int8_t*) ((uintptr_t) a1 + a_stride);
  float* c2 = (float*) ((uintptr_t) c1 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 2) {
    a2 = a1;
    c2 = c1;
  }
  const int8_t* a3 = (const int8_t*) ((uintptr_t) a2 + a_stride);
  float* c3 = (float*) ((uintptr_t) c2 + cm_stride);
  if XNN_UNPREDICTABLE(mr != 4) {
    a3 = a2;
    c3 = c2;
  }

  const int32_t vzp0 = quantization_params[0].zero_point;
  const int32_t vzp1 = quantization_params[1].zero_point;
  const int32_t vzp2 = quantization_params[2].zero_point;
  const int32_t vzp3 = quantization_params[3].zero_point;
  do {
    int32_t vacc0x0 = ((const int32_t*) w)[0];
    int32_t vacc0x1 = ((const int32_t*) w)[1];
    int32_t vacc0x2 = ((const int32_t*) w)[2];
    int32_t vacc0x3 = ((const int32_t*) w)[3];
    int32_t vacc1x0 = vacc0x0;
    int32_t vacc1x1 = vacc0x1;
    int32_t vacc1x2 = vacc0x2;
    int32_t vacc1x3 = vacc0x3;
    int32_t vacc2x0 = vacc0x0;
    int32_t vacc2x1 = vacc0x1;
    int32_t vacc2x2 = vacc0x2;
    int32_t vacc2x3 = vacc0x3;
    int32_t vacc3x0 = vacc0x0;
    int32_t vacc3x1 = vacc0x1;
    int32_t vacc3x2 = vacc0x2;
    int32_t vacc3x3 = vacc0x3;
    w = (const int32_t*) w + 4;
    vacc0x0 *= vzp0;
    vacc0x1 *= vzp0;
    vacc0x2 *= vzp0;
    vacc0x3 *= vzp0;
    vacc1x0 *= vzp1;
    vacc1x1 *= vzp1;
    vacc1x2 *= vzp1;
    vacc1x3 *= vzp1;
    vacc2x0 *= vzp2;
    vacc2x1 *= vzp2;
    vacc2x2 *= vzp2;
    vacc2x3 *= vzp2;
    vacc3x0 *= vzp3;
    vacc3x1 *= vzp3;
    vacc3x2 *= vzp3;
    vacc3x3 *= vzp3;

    size_t k = kc;
    do {
      const int32_t va0 = (int32_t) *a0++;
      const int32_t va1 = (int32_t) *a1++;
      const int32_t va2 = (int32_t) *a2++;
      const int32_t va3 = (int32_t) *a3++;

      const int32_t vb0 = (int32_t) ((const int8_t*) w)[0];
      const int32_t vb1 = (int32_t) ((const int8_t*) w)[1];
      const int32_t vb2 = (int32_t) ((const int8_t*) w)[2];
      const int32_t vb3 = (int32_t) ((const int8_t*) w)[3];
      w = (const int8_t*) w + 4;

      vacc0x0 += va0 * vb0;
      vacc0x1 += va0 * vb1;
      vacc0x2 += va0 * vb2;
      vacc0x3 += va0 * vb3;
      vacc1x0 += va1 * vb0;
      vacc1x1 += va1 * vb1;
      vacc1x2 += va1 * vb2;
      vacc1x3 += va1 * vb3;
      vacc2x0 += va2 * vb0;
      vacc2x1 += va2 * vb1;
      vacc2x2 += va2 * vb2;
      vacc2x3 += va2 * vb3;
      vacc3x0 += va3 * vb0;
      vacc3x1 += va3 * vb1;
      vacc3x2 += va3 * vb2;
      vacc3x3 += va3 * vb3;

      k -= sizeof(int8_t);
    } while (k != 0);

    float vout0x0 = (float) vacc0x0;
    float vout0x1 = (float) vacc0x1;
    float vout0x2 = (float) vacc0x2;
    float vout0x3 = (float) vacc0x3;
    float vout1x0 = (float) vacc1x0;
    float vout1x1 = (float) vacc1x1;
    float vout1x2 = (float) vacc1x2;
    float vout1x3 = (float) vacc1x3;
    float vout2x0 = (float) vacc2x0;
    float vout2x1 = (float) vacc2x1;
    float vout2x2 = (float) vacc2x2;
    float vout2x3 = (float) vacc2x3;
    float vout3x0 = (float) vacc3x0;
    float vout3x1 = (float) vacc3x1;
    float vout3x2 = (float) vacc3x2;
    float vout3x3 = (float) vacc3x3;
    const float vscale0 = quantization_params[0].scale;
    const float vscale1 = quantization_params[1].scale;
    const float vscale2 = quantization_params[2].scale;
    const float vscale3 = quantization_params[3].scale;
    vout0x0 *= vscale0;
    vout1x0 *= vscale1;
    vout2x0 *= vscale2;
    vout3x0 *= vscale3;
    vout0x1 *= vscale0;
    vout1x1 *= vscale1;
    vout2x1 *= vscale2;
    vout3x1 *= vscale3;
    vout0x2 *= vscale0;
    vout1x2 *= vscale1;
    vout2x2 *= vscale2;
    vout3x2 *= vscale3;
    vout0x3 *= vscale0;
    vout1x3 *= vscale1;
    vout2x3 *= vscale2;
    vout3x3 *= vscale3;
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
    vout1x0 += vbias0;
    vout1x0 = math_max_f32(vout1x0, vmin);
    vout1x0 = math_min_f32(vout1x0, vmax);
    vout2x0 += vbias0;
    vout2x0 = math_max_f32(vout2x0, vmin);
    vout2x0 = math_min_f32(vout2x0, vmax);
    vout3x0 += vbias0;
    vout3x0 = math_max_f32(vout3x0, vmin);
    vout3x0 = math_min_f32(vout3x0, vmax);
    vout0x1 += vbias1;
    vout0x1 = math_max_f32(vout0x1, vmin);
    vout0x1 = math_min_f32(vout0x1, vmax);
    vout1x1 += vbias1;
    vout1x1 = math_max_f32(vout1x1, vmin);
    vout1x1 = math_min_f32(vout1x1, vmax);
    vout2x1 += vbias1;
    vout2x1 = math_max_f32(vout2x1, vmin);
    vout2x1 = math_min_f32(vout2x1, vmax);
    vout3x1 += vbias1;
    vout3x1 = math_max_f32(vout3x1, vmin);
    vout3x1 = math_min_f32(vout3x1, vmax);
    vout0x2 += vbias2;
    vout0x2 = math_max_f32(vout0x2, vmin);
    vout0x2 = math_min_f32(vout0x2, vmax);
    vout1x2 += vbias2;
    vout1x2 = math_max_f32(vout1x2, vmin);
    vout1x2 = math_min_f32(vout1x2, vmax);
    vout2x2 += vbias2;
    vout2x2 = math_max_f32(vout2x2, vmin);
    vout2x2 = math_min_f32(vout2x2, vmax);
    vout3x2 += vbias2;
    vout3x2 = math_max_f32(vout3x2, vmin);
    vout3x2 = math_min_f32(vout3x2, vmax);
    vout0x3 += vbias3;
    vout0x3 = math_max_f32(vout0x3, vmin);
    vout0x3 = math_min_f32(vout0x3, vmax);
    vout1x3 += vbias3;
    vout1x3 = math_max_f32(vout1x3, vmin);
    vout1x3 = math_min_f32(vout1x3, vmax);
    vout2x3 += vbias3;
    vout2x3 = math_max_f32(vout2x3, vmin);
    vout2x3 = math_min_f32(vout2x3, vmax);
    vout3x3 += vbias3;
    vout3x3 = math_max_f32(vout3x3, vmin);
    vout3x3 = math_min_f32(vout3x3, vmax);

    if XNN_LIKELY(nc >= 4) {
      c3[0] = vout3x0;
      c3[1] = vout3x1;
      c3[2] = vout3x2;
      c3[3] = vout3x3;
      c2[0] = vout2x0;
      c2[1] = vout2x1;
      c2[2] = vout2x2;
      c2[3] = vout2x3;
      c1[0] = vout1x0;
      c1[1] = vout1x1;
      c1[2] = vout1x2;
      c1[3] = vout1x3;
      c0[0] = vout0x0;
      c0[1] = vout0x1;
      c0[2] = vout0x2;
      c0[3] = vout0x3;

      a0 = (const int8_t*) ((uintptr_t) a0 - kc);
      a1 = (const int8_t*) ((uintptr_t) a1 - kc);
      a2 = (const int8_t*) ((uintptr_t) a2 - kc);
      a3 = (const int8_t*) ((uintptr_t) a3 - kc);

      c0 = (float*) ((uintptr_t) c0 + cn_stride);
      c1 = (float*) ((uintptr_t) c1 + cn_stride);
      c2 = (float*) ((uintptr_t) c2 + cn_stride);
      c3 = (float*) ((uintptr_t) c3 + cn_stride);

      nc -= 4;
    } else {
      if (nc & 2) {
        c3[0] = (float) vout3x0;
        c3[1] = (float) vout3x1;
        vout3x0 = vout3x2;
        c3 += 2;
        c2[0] = (float) vout2x0;
        c2[1] = (float) vout2x1;
        vout2x0 = vout2x2;
        c2 += 2;
        c1[0] = (float) vout1x0;
        c1[1] = (float) vout1x1;
        vout1x0 = vout1x2;
        c1 += 2;
        c0[0] = (float) vout0x0;
        c0[1] = (float) vout0x1;
        vout0x0 = vout0x2;
        c0 += 2;
      }
      if (nc & 1) {
        c3[0] = (float) vout3x0;
        c2[0] = (float) vout2x0;
        c1[0] = (float) vout1x0;
        c0[0] = (float) vout0x0;
      }

      nc = 0;
    }
  } while (nc != 0);
}
