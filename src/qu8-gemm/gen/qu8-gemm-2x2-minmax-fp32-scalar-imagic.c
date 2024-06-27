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
#include "xnnpack/unaligned.h"


void xnn_qu8_gemm_minmax_fp32_ukernel_2x2__scalar_imagic(
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
  assert(mr <= 2);
  assert(nc != 0);
  assert(kc != 0);

  const uint8_t* a0 = a;
  uint8_t* c0 = c;
  const uint8_t* a1 = (const uint8_t*) ((uintptr_t) a0 + a_stride);
  uint8_t* c1 = (uint8_t*) ((uintptr_t) c0 + cm_stride);
  if XNN_UNPREDICTABLE(mr != 2) {
    a1 = a0;
    c1 = c0;
  }

  const int32_t vb_zero_point = params->fp32_scalar_imagic.kernel_zero_point;
  do {
    int32_t vacc0x0 = unaligned_indexed_load_s32(w, 0);
    int32_t vacc0x1 = unaligned_indexed_load_s32(w, 1);
    int32_t vacc1x0 = vacc0x0;
    int32_t vacc1x1 = vacc0x1;
    w = (const int32_t*) w + 2;

    size_t k = kc;
    do {
      const int32_t va0 = (int32_t) (uint32_t) *a0++;
      const int32_t va1 = (int32_t) (uint32_t) *a1++;

      const int32_t vb0 = (int32_t) (uint32_t) ((const uint8_t*) w)[0] - vb_zero_point;
      const int32_t vb1 = (int32_t) (uint32_t) ((const uint8_t*) w)[1] - vb_zero_point;
      w = (const uint8_t*) w + 2;

      vacc0x0 += va0 * vb0;
      vacc0x1 += va0 * vb1;
      vacc1x0 += va1 * vb0;
      vacc1x1 += va1 * vb1;

      k -= sizeof(uint8_t);
    } while (k != 0);

    float vfpacc0x0 = (float) vacc0x0;
    float vfpacc0x1 = (float) vacc0x1;
    float vfpacc1x0 = (float) vacc1x0;
    float vfpacc1x1 = (float) vacc1x1;

    const float vscale = params->fp32_scalar_imagic.scale;
    vfpacc0x0 *= vscale;
    vfpacc0x1 *= vscale;
    vfpacc1x0 *= vscale;
    vfpacc1x1 *= vscale;

    const float vmagic_bias = params->fp32_scalar_imagic.magic_bias;
    vfpacc0x0 += vmagic_bias;
    vfpacc0x1 += vmagic_bias;
    vfpacc1x0 += vmagic_bias;
    vfpacc1x1 += vmagic_bias;

    int32_t vout0x0 = (int32_t) float_as_uint32(vfpacc0x0);
    int32_t vout0x1 = (int32_t) float_as_uint32(vfpacc0x1);
    int32_t vout1x0 = (int32_t) float_as_uint32(vfpacc1x0);
    int32_t vout1x1 = (int32_t) float_as_uint32(vfpacc1x1);

    const int32_t vmagic_min = params->fp32_scalar_imagic.magic_min;
    vout0x0 = math_max_s32(vout0x0, vmagic_min);
    vout0x1 = math_max_s32(vout0x1, vmagic_min);
    vout1x0 = math_max_s32(vout1x0, vmagic_min);
    vout1x1 = math_max_s32(vout1x1, vmagic_min);

    const int32_t vmagic_max = params->fp32_scalar_imagic.magic_max;
    vout0x0 = math_min_s32(vout0x0, vmagic_max);
    vout0x1 = math_min_s32(vout0x1, vmagic_max);
    vout1x0 = math_min_s32(vout1x0, vmagic_max);
    vout1x1 = math_min_s32(vout1x1, vmagic_max);

    const int32_t vmagic_bias_less_zero_point = params->fp32_scalar_imagic.magic_bias_less_zero_point;
    vout0x0 -= vmagic_bias_less_zero_point;
    vout0x1 -= vmagic_bias_less_zero_point;
    vout1x0 -= vmagic_bias_less_zero_point;
    vout1x1 -= vmagic_bias_less_zero_point;

    if XNN_LIKELY(nc >= 2) {
      c0[0] = (uint8_t) vout0x0;
      c0[1] = (uint8_t) vout0x1;
      c1[0] = (uint8_t) vout1x0;
      c1[1] = (uint8_t) vout1x1;

      a0 = (const uint8_t*) ((uintptr_t) a0 - kc);
      a1 = (const uint8_t*) ((uintptr_t) a1 - kc);

      c0 = (uint8_t*) ((uintptr_t) c0 + cn_stride);
      c1 = (uint8_t*) ((uintptr_t) c1 + cn_stride);

      nc -= 2;
    } else {
      if (nc & 1) {
        c0[0] = (uint8_t) vout0x0;
        c1[0] = (uint8_t) vout1x0;
      }

      nc = 0;
    }
  } while (nc != 0);
}
