// Auto-generated file. Do not edit!
//   Template: src/qs8-gemm/scalar.c.in
//   Generator: tools/xngen
//
// Copyright 2021 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include "xnnpack/gemm.h"
#include "xnnpack/math.h"


void xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_3x4__scalar_imagic(
    size_t mr,
    size_t nc,
    size_t kc,
    const int8_t* restrict a,
    size_t a_stride,
    const void* restrict w,
    int8_t* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_qs8_qc8w_conv_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(mr != 0);
  assert(mr <= 3);
  assert(nc != 0);
  assert(kc != 0);

  const int8_t* a0 = a;
  int8_t* c0 = c;
  const int8_t* a1 = (const int8_t*) ((uintptr_t) a0 + a_stride);
  int8_t* c1 = (int8_t*) ((uintptr_t) c0 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 2) {
    a1 = a0;
    c1 = c0;
  }
  const int8_t* a2 = (const int8_t*) ((uintptr_t) a1 + a_stride);
  int8_t* c2 = (int8_t*) ((uintptr_t) c1 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 2) {
    a2 = a1;
    c2 = c1;
  }

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
    w = (const int32_t*) w + 4;

    size_t k = kc;
    do {
      const int32_t va0 = (int32_t) *a0++;
      const int32_t va1 = (int32_t) *a1++;
      const int32_t va2 = (int32_t) *a2++;

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

      k -= sizeof(int8_t);
    } while (k != 0);

    float vfpacc0x0 = (float) vacc0x0;
    float vfpacc0x1 = (float) vacc0x1;
    float vfpacc0x2 = (float) vacc0x2;
    float vfpacc0x3 = (float) vacc0x3;
    float vfpacc1x0 = (float) vacc1x0;
    float vfpacc1x1 = (float) vacc1x1;
    float vfpacc1x2 = (float) vacc1x2;
    float vfpacc1x3 = (float) vacc1x3;
    float vfpacc2x0 = (float) vacc2x0;
    float vfpacc2x1 = (float) vacc2x1;
    float vfpacc2x2 = (float) vacc2x2;
    float vfpacc2x3 = (float) vacc2x3;

    const float vscale0 = ((const float*) w)[0];
    vfpacc0x0 *= vscale0;
    vfpacc1x0 *= vscale0;
    vfpacc2x0 *= vscale0;
    const float vscale1 = ((const float*) w)[1];
    vfpacc0x1 *= vscale1;
    vfpacc1x1 *= vscale1;
    vfpacc2x1 *= vscale1;
    const float vscale2 = ((const float*) w)[2];
    vfpacc0x2 *= vscale2;
    vfpacc1x2 *= vscale2;
    vfpacc2x2 *= vscale2;
    const float vscale3 = ((const float*) w)[3];
    vfpacc0x3 *= vscale3;
    vfpacc1x3 *= vscale3;
    vfpacc2x3 *= vscale3;
    w = (const void*) ((const float*) w + 4);

    const float vmagic_bias = params->fp32_scalar_imagic.magic_bias;
    vfpacc0x0 += vmagic_bias;
    vfpacc0x1 += vmagic_bias;
    vfpacc0x2 += vmagic_bias;
    vfpacc0x3 += vmagic_bias;
    vfpacc1x0 += vmagic_bias;
    vfpacc1x1 += vmagic_bias;
    vfpacc1x2 += vmagic_bias;
    vfpacc1x3 += vmagic_bias;
    vfpacc2x0 += vmagic_bias;
    vfpacc2x1 += vmagic_bias;
    vfpacc2x2 += vmagic_bias;
    vfpacc2x3 += vmagic_bias;

    int32_t vout0x0 = (int32_t) float_as_uint32(vfpacc0x0);
    int32_t vout0x1 = (int32_t) float_as_uint32(vfpacc0x1);
    int32_t vout0x2 = (int32_t) float_as_uint32(vfpacc0x2);
    int32_t vout0x3 = (int32_t) float_as_uint32(vfpacc0x3);
    int32_t vout1x0 = (int32_t) float_as_uint32(vfpacc1x0);
    int32_t vout1x1 = (int32_t) float_as_uint32(vfpacc1x1);
    int32_t vout1x2 = (int32_t) float_as_uint32(vfpacc1x2);
    int32_t vout1x3 = (int32_t) float_as_uint32(vfpacc1x3);
    int32_t vout2x0 = (int32_t) float_as_uint32(vfpacc2x0);
    int32_t vout2x1 = (int32_t) float_as_uint32(vfpacc2x1);
    int32_t vout2x2 = (int32_t) float_as_uint32(vfpacc2x2);
    int32_t vout2x3 = (int32_t) float_as_uint32(vfpacc2x3);

    const int32_t vmagic_min = params->fp32_scalar_imagic.magic_min;
    vout0x0 = math_max_s32(vout0x0, vmagic_min);
    vout0x1 = math_max_s32(vout0x1, vmagic_min);
    vout0x2 = math_max_s32(vout0x2, vmagic_min);
    vout0x3 = math_max_s32(vout0x3, vmagic_min);
    vout1x0 = math_max_s32(vout1x0, vmagic_min);
    vout1x1 = math_max_s32(vout1x1, vmagic_min);
    vout1x2 = math_max_s32(vout1x2, vmagic_min);
    vout1x3 = math_max_s32(vout1x3, vmagic_min);
    vout2x0 = math_max_s32(vout2x0, vmagic_min);
    vout2x1 = math_max_s32(vout2x1, vmagic_min);
    vout2x2 = math_max_s32(vout2x2, vmagic_min);
    vout2x3 = math_max_s32(vout2x3, vmagic_min);

    const int32_t vmagic_max = params->fp32_scalar_imagic.magic_max;
    vout0x0 = math_min_s32(vout0x0, vmagic_max);
    vout0x1 = math_min_s32(vout0x1, vmagic_max);
    vout0x2 = math_min_s32(vout0x2, vmagic_max);
    vout0x3 = math_min_s32(vout0x3, vmagic_max);
    vout1x0 = math_min_s32(vout1x0, vmagic_max);
    vout1x1 = math_min_s32(vout1x1, vmagic_max);
    vout1x2 = math_min_s32(vout1x2, vmagic_max);
    vout1x3 = math_min_s32(vout1x3, vmagic_max);
    vout2x0 = math_min_s32(vout2x0, vmagic_max);
    vout2x1 = math_min_s32(vout2x1, vmagic_max);
    vout2x2 = math_min_s32(vout2x2, vmagic_max);
    vout2x3 = math_min_s32(vout2x3, vmagic_max);

    const int32_t vmagic_bias_less_zero_point = params->fp32_scalar_imagic.magic_bias_less_zero_point;
    vout0x0 -= vmagic_bias_less_zero_point;
    vout0x1 -= vmagic_bias_less_zero_point;
    vout0x2 -= vmagic_bias_less_zero_point;
    vout0x3 -= vmagic_bias_less_zero_point;
    vout1x0 -= vmagic_bias_less_zero_point;
    vout1x1 -= vmagic_bias_less_zero_point;
    vout1x2 -= vmagic_bias_less_zero_point;
    vout1x3 -= vmagic_bias_less_zero_point;
    vout2x0 -= vmagic_bias_less_zero_point;
    vout2x1 -= vmagic_bias_less_zero_point;
    vout2x2 -= vmagic_bias_less_zero_point;
    vout2x3 -= vmagic_bias_less_zero_point;

    if XNN_LIKELY(nc >= 4) {
      c0[0] = (int8_t) vout0x0;
      c0[1] = (int8_t) vout0x1;
      c0[2] = (int8_t) vout0x2;
      c0[3] = (int8_t) vout0x3;
      c1[0] = (int8_t) vout1x0;
      c1[1] = (int8_t) vout1x1;
      c1[2] = (int8_t) vout1x2;
      c1[3] = (int8_t) vout1x3;
      c2[0] = (int8_t) vout2x0;
      c2[1] = (int8_t) vout2x1;
      c2[2] = (int8_t) vout2x2;
      c2[3] = (int8_t) vout2x3;

      a0 = (const int8_t*) ((uintptr_t) a0 - kc);
      a1 = (const int8_t*) ((uintptr_t) a1 - kc);
      a2 = (const int8_t*) ((uintptr_t) a2 - kc);

      c0 = (int8_t*) ((uintptr_t) c0 + cn_stride);
      c1 = (int8_t*) ((uintptr_t) c1 + cn_stride);
      c2 = (int8_t*) ((uintptr_t) c2 + cn_stride);

      nc -= 4;
    } else {
      if (nc & 2) {
        c0[0] = (int8_t) vout0x0;
        c0[1] = (int8_t) vout0x1;
        vout0x0 = vout0x2;
        c0 += 2;
        c1[0] = (int8_t) vout1x0;
        c1[1] = (int8_t) vout1x1;
        vout1x0 = vout1x2;
        c1 += 2;
        c2[0] = (int8_t) vout2x0;
        c2[1] = (int8_t) vout2x1;
        vout2x0 = vout2x2;
        c2 += 2;
      }
      if (nc & 1) {
        c0[0] = (int8_t) vout0x0;
        c1[0] = (int8_t) vout1x0;
        c2[0] = (int8_t) vout2x0;
      }

      nc = 0;
    }
  } while (nc != 0);
}
