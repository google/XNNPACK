// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/qs8-igemm/scalar.c.in
//   Generator: tools/xngen
//
// Copyright 2021 Google LLC
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



void xnn_qu8_igemm_minmax_fp32_ukernel_2x4__scalar_imagic(
    size_t mr,
    size_t nc,
    size_t kc,
    size_t ks,
    const uint8_t** restrict a,
    const void* restrict w,
    uint8_t* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    size_t a_offset,
    const uint8_t* zero,
    const union xnn_qu8_conv_minmax_params* restrict params)
{
  assert(mr != 0);
  assert(mr <= 2);
  assert(nc != 0);
  assert(kc != 0);
  assert(ks != 0);
  assert(ks % (2 * sizeof(void*)) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  uint8_t* c0 = c;
  uint8_t* c1 = (uint8_t*) ((uintptr_t) c0 + cm_stride);
  if XNN_UNPREDICTABLE(mr != 2) {
    c1 = c0;
  }

  const int32_t output_min_less_zero_point = (int32_t) params->fp32_scalar.output_min - (int32_t) params->fp32_scalar.output_zero_point;
  const int32_t output_max_less_zero_point = (int32_t) params->fp32_scalar.output_max - (int32_t) params->fp32_scalar.output_zero_point;
  const float vscale = params->fp32_scalar.scale;
  const float vmagic_bias = 12582912.0f;
  const int32_t vmagic_min = (int32_t) float_as_uint32(12582912.0f + output_min_less_zero_point);
  const int32_t vmagic_max = (int32_t) float_as_uint32(12582912.0f + output_max_less_zero_point);
  const int32_t vmagic_bias_less_zero_point = INT32_C(0x4B400000) - (int32_t) params->fp32_scalar.output_zero_point;

  const int32_t vb_zero_point = params->fp32_scalar.kernel_zero_point;
  do {
    int32_t vacc0x0 = ((const int32_t*) w)[0];
    int32_t vacc0x1 = ((const int32_t*) w)[1];
    int32_t vacc0x2 = ((const int32_t*) w)[2];
    int32_t vacc0x3 = ((const int32_t*) w)[3];
    int32_t vacc1x0 = vacc0x0;
    int32_t vacc1x1 = vacc0x1;
    int32_t vacc1x2 = vacc0x2;
    int32_t vacc1x3 = vacc0x3;
    w = (const void*) ((const int32_t*) w + 4);

    size_t p = ks;
    do {
      const uint8_t* restrict a0 = a[0];
      assert(a0 != NULL);
      if XNN_UNPREDICTABLE(a0 != zero) {
        a0 = (const uint8_t*) ((uintptr_t) a0 + a_offset);
      }
      const uint8_t* restrict a1 = a[1];
      assert(a1 != NULL);
      if XNN_UNPREDICTABLE(a1 != zero) {
        a1 = (const uint8_t*) ((uintptr_t) a1 + a_offset);
      }
      a += 2;

      size_t k = kc;
      do {
        const int32_t va0 = (int32_t) (uint32_t) *a0++;
        const int32_t va1 = (int32_t) (uint32_t) *a1++;

        const int32_t vb0 = (int32_t) (uint32_t) ((const uint8_t*) w)[0] - vb_zero_point;
        const int32_t vb1 = (int32_t) (uint32_t) ((const uint8_t*) w)[1] - vb_zero_point;
        const int32_t vb2 = (int32_t) (uint32_t) ((const uint8_t*) w)[2] - vb_zero_point;
        const int32_t vb3 = (int32_t) (uint32_t) ((const uint8_t*) w)[3] - vb_zero_point;
        w = (const void*) ((const uint8_t*) w + 4);

        vacc0x0 += va0 * vb0;
        vacc0x1 += va0 * vb1;
        vacc0x2 += va0 * vb2;
        vacc0x3 += va0 * vb3;
        vacc1x0 += va1 * vb0;
        vacc1x1 += va1 * vb1;
        vacc1x2 += va1 * vb2;
        vacc1x3 += va1 * vb3;

        k -= sizeof(uint8_t);
      } while (k != 0);
      p -= 2 * sizeof(void*);
    } while (p != 0);

    float vfpacc0x0 = (float) vacc0x0;
    float vfpacc0x1 = (float) vacc0x1;
    float vfpacc0x2 = (float) vacc0x2;
    float vfpacc0x3 = (float) vacc0x3;
    float vfpacc1x0 = (float) vacc1x0;
    float vfpacc1x1 = (float) vacc1x1;
    float vfpacc1x2 = (float) vacc1x2;
    float vfpacc1x3 = (float) vacc1x3;

    vfpacc0x0 *= vscale;
    vfpacc0x1 *= vscale;
    vfpacc0x2 *= vscale;
    vfpacc0x3 *= vscale;
    vfpacc1x0 *= vscale;
    vfpacc1x1 *= vscale;
    vfpacc1x2 *= vscale;
    vfpacc1x3 *= vscale;

    vfpacc0x0 += vmagic_bias;
    vfpacc0x1 += vmagic_bias;
    vfpacc0x2 += vmagic_bias;
    vfpacc0x3 += vmagic_bias;
    vfpacc1x0 += vmagic_bias;
    vfpacc1x1 += vmagic_bias;
    vfpacc1x2 += vmagic_bias;
    vfpacc1x3 += vmagic_bias;

    int32_t vout0x0 = (int32_t) float_as_uint32(vfpacc0x0);
    int32_t vout0x1 = (int32_t) float_as_uint32(vfpacc0x1);
    int32_t vout0x2 = (int32_t) float_as_uint32(vfpacc0x2);
    int32_t vout0x3 = (int32_t) float_as_uint32(vfpacc0x3);
    int32_t vout1x0 = (int32_t) float_as_uint32(vfpacc1x0);
    int32_t vout1x1 = (int32_t) float_as_uint32(vfpacc1x1);
    int32_t vout1x2 = (int32_t) float_as_uint32(vfpacc1x2);
    int32_t vout1x3 = (int32_t) float_as_uint32(vfpacc1x3);

    vout0x0 = math_max_s32(vout0x0, vmagic_min);
    vout0x1 = math_max_s32(vout0x1, vmagic_min);
    vout0x2 = math_max_s32(vout0x2, vmagic_min);
    vout0x3 = math_max_s32(vout0x3, vmagic_min);
    vout1x0 = math_max_s32(vout1x0, vmagic_min);
    vout1x1 = math_max_s32(vout1x1, vmagic_min);
    vout1x2 = math_max_s32(vout1x2, vmagic_min);
    vout1x3 = math_max_s32(vout1x3, vmagic_min);

    vout0x0 = math_min_s32(vout0x0, vmagic_max);
    vout0x1 = math_min_s32(vout0x1, vmagic_max);
    vout0x2 = math_min_s32(vout0x2, vmagic_max);
    vout0x3 = math_min_s32(vout0x3, vmagic_max);
    vout1x0 = math_min_s32(vout1x0, vmagic_max);
    vout1x1 = math_min_s32(vout1x1, vmagic_max);
    vout1x2 = math_min_s32(vout1x2, vmagic_max);
    vout1x3 = math_min_s32(vout1x3, vmagic_max);

    vout0x0 -= vmagic_bias_less_zero_point;
    vout0x1 -= vmagic_bias_less_zero_point;
    vout0x2 -= vmagic_bias_less_zero_point;
    vout0x3 -= vmagic_bias_less_zero_point;
    vout1x0 -= vmagic_bias_less_zero_point;
    vout1x1 -= vmagic_bias_less_zero_point;
    vout1x2 -= vmagic_bias_less_zero_point;
    vout1x3 -= vmagic_bias_less_zero_point;

    if XNN_LIKELY(nc >= 4) {
      c1[0] = (uint8_t) vout1x0;
      c1[1] = (uint8_t) vout1x1;
      c1[2] = (uint8_t) vout1x2;
      c1[3] = (uint8_t) vout1x3;
      c0[0] = (uint8_t) vout0x0;
      c0[1] = (uint8_t) vout0x1;
      c0[2] = (uint8_t) vout0x2;
      c0[3] = (uint8_t) vout0x3;

      c1 = (uint8_t*) ((uintptr_t) c1 + cn_stride);
      c0 = (uint8_t*) ((uintptr_t) c0 + cn_stride);

      a = (const uint8_t**restrict) ((uintptr_t) a - ks);
      nc -= 4;
    } else {
      if (nc & 2) {
        c1[0] = (uint8_t) vout1x0;
        c1[1] = (uint8_t) vout1x1;
        vout1x0 = vout1x2;
        c1 += 2;
        c0[0] = (uint8_t) vout0x0;
        c0[1] = (uint8_t) vout0x1;
        vout0x0 = vout0x2;
        c0 += 2;
      }
      if (nc & 1) {
        c1[0] = (uint8_t) vout1x0;
        c0[0] = (uint8_t) vout0x0;
      }

      nc = 0;
    }
  } while (nc != 0);
}
