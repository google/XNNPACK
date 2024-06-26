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


void xnn_qu8_gemm_minmax_fp32_ukernel_4x4__scalar_imagic(
    size_t mr,
    size_t nc,
    size_t kc,
    const uint8_t* restrict a,
    size_t a_stride,
    const void* restrict w,
    uint8_t* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_qu8_conv_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(mr != 0);
  assert(mr <= 4);
  assert(nc != 0);
  assert(kc != 0);

  const uint8_t* a0 = a;
  uint8_t* c0 = c;
  const uint8_t* a1 = (const uint8_t*) ((uintptr_t) a0 + a_stride);
  uint8_t* c1 = (uint8_t*) ((uintptr_t) c0 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 2) {
    a1 = a0;
    c1 = c0;
  }
  const uint8_t* a2 = (const uint8_t*) ((uintptr_t) a1 + a_stride);
  uint8_t* c2 = (uint8_t*) ((uintptr_t) c1 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 2) {
    a2 = a1;
    c2 = c1;
  }
  const uint8_t* a3 = (const uint8_t*) ((uintptr_t) a2 + a_stride);
  uint8_t* c3 = (uint8_t*) ((uintptr_t) c2 + cm_stride);
  if XNN_UNPREDICTABLE(mr != 4) {
    a3 = a2;
    c3 = c2;
  }

  const int32_t vb_zero_point = params->fp32_scalar_imagic.kernel_zero_point;
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

    size_t k = kc;
    do {
      const int32_t va0 = (int32_t) (uint32_t) *a0++;
      const int32_t va1 = (int32_t) (uint32_t) *a1++;
      const int32_t va2 = (int32_t) (uint32_t) *a2++;
      const int32_t va3 = (int32_t) (uint32_t) *a3++;

      const int32_t vb0 = (int32_t) (uint32_t) ((const uint8_t*) w)[0] - vb_zero_point;
      const int32_t vb1 = (int32_t) (uint32_t) ((const uint8_t*) w)[1] - vb_zero_point;
      const int32_t vb2 = (int32_t) (uint32_t) ((const uint8_t*) w)[2] - vb_zero_point;
      const int32_t vb3 = (int32_t) (uint32_t) ((const uint8_t*) w)[3] - vb_zero_point;
      w = (const uint8_t*) w + 4;

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

      k -= sizeof(uint8_t);
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
    float vfpacc3x0 = (float) vacc3x0;
    float vfpacc3x1 = (float) vacc3x1;
    float vfpacc3x2 = (float) vacc3x2;
    float vfpacc3x3 = (float) vacc3x3;

    const float vscale = params->fp32_scalar_imagic.scale;
    vfpacc0x0 *= vscale;
    vfpacc0x1 *= vscale;
    vfpacc0x2 *= vscale;
    vfpacc0x3 *= vscale;
    vfpacc1x0 *= vscale;
    vfpacc1x1 *= vscale;
    vfpacc1x2 *= vscale;
    vfpacc1x3 *= vscale;
    vfpacc2x0 *= vscale;
    vfpacc2x1 *= vscale;
    vfpacc2x2 *= vscale;
    vfpacc2x3 *= vscale;
    vfpacc3x0 *= vscale;
    vfpacc3x1 *= vscale;
    vfpacc3x2 *= vscale;
    vfpacc3x3 *= vscale;

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
    vfpacc3x0 += vmagic_bias;
    vfpacc3x1 += vmagic_bias;
    vfpacc3x2 += vmagic_bias;
    vfpacc3x3 += vmagic_bias;

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
    int32_t vout3x0 = (int32_t) float_as_uint32(vfpacc3x0);
    int32_t vout3x1 = (int32_t) float_as_uint32(vfpacc3x1);
    int32_t vout3x2 = (int32_t) float_as_uint32(vfpacc3x2);
    int32_t vout3x3 = (int32_t) float_as_uint32(vfpacc3x3);

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
    vout3x0 = math_max_s32(vout3x0, vmagic_min);
    vout3x1 = math_max_s32(vout3x1, vmagic_min);
    vout3x2 = math_max_s32(vout3x2, vmagic_min);
    vout3x3 = math_max_s32(vout3x3, vmagic_min);

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
    vout3x0 = math_min_s32(vout3x0, vmagic_max);
    vout3x1 = math_min_s32(vout3x1, vmagic_max);
    vout3x2 = math_min_s32(vout3x2, vmagic_max);
    vout3x3 = math_min_s32(vout3x3, vmagic_max);

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
    vout3x0 -= vmagic_bias_less_zero_point;
    vout3x1 -= vmagic_bias_less_zero_point;
    vout3x2 -= vmagic_bias_less_zero_point;
    vout3x3 -= vmagic_bias_less_zero_point;

    if XNN_LIKELY(nc >= 4) {
      c0[0] = (uint8_t) vout0x0;
      c0[1] = (uint8_t) vout0x1;
      c0[2] = (uint8_t) vout0x2;
      c0[3] = (uint8_t) vout0x3;
      c1[0] = (uint8_t) vout1x0;
      c1[1] = (uint8_t) vout1x1;
      c1[2] = (uint8_t) vout1x2;
      c1[3] = (uint8_t) vout1x3;
      c2[0] = (uint8_t) vout2x0;
      c2[1] = (uint8_t) vout2x1;
      c2[2] = (uint8_t) vout2x2;
      c2[3] = (uint8_t) vout2x3;
      c3[0] = (uint8_t) vout3x0;
      c3[1] = (uint8_t) vout3x1;
      c3[2] = (uint8_t) vout3x2;
      c3[3] = (uint8_t) vout3x3;

      a0 = (const uint8_t*) ((uintptr_t) a0 - kc);
      a1 = (const uint8_t*) ((uintptr_t) a1 - kc);
      a2 = (const uint8_t*) ((uintptr_t) a2 - kc);
      a3 = (const uint8_t*) ((uintptr_t) a3 - kc);

      c0 = (uint8_t*) ((uintptr_t) c0 + cn_stride);
      c1 = (uint8_t*) ((uintptr_t) c1 + cn_stride);
      c2 = (uint8_t*) ((uintptr_t) c2 + cn_stride);
      c3 = (uint8_t*) ((uintptr_t) c3 + cn_stride);

      nc -= 4;
    } else {
      if (nc & 2) {
        c0[0] = (uint8_t) vout0x0;
        c0[1] = (uint8_t) vout0x1;
        vout0x0 = vout0x2;
        c0 += 2;
        c1[0] = (uint8_t) vout1x0;
        c1[1] = (uint8_t) vout1x1;
        vout1x0 = vout1x2;
        c1 += 2;
        c2[0] = (uint8_t) vout2x0;
        c2[1] = (uint8_t) vout2x1;
        vout2x0 = vout2x2;
        c2 += 2;
        c3[0] = (uint8_t) vout3x0;
        c3[1] = (uint8_t) vout3x1;
        vout3x0 = vout3x2;
        c3 += 2;
      }
      if (nc & 1) {
        c0[0] = (uint8_t) vout0x0;
        c1[0] = (uint8_t) vout1x0;
        c2[0] = (uint8_t) vout2x0;
        c3[0] = (uint8_t) vout3x0;
      }

      nc = 0;
    }
  } while (nc != 0);
}
