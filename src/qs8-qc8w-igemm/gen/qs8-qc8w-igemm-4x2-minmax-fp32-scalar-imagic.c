// Auto-generated file. Do not edit!
//   Template: src/qs8-igemm/scalar.c.in
//   Generator: tools/xngen
//
// Copyright 2021 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include "xnnpack/math.h"
#include "xnnpack/gemm.h"
#include "xnnpack/unaligned.h"


void xnn_qs8_qc8w_igemm_minmax_fp32_ukernel_4x2__scalar_imagic(
    size_t mr,
    size_t nc,
    size_t kc,
    size_t ks,
    const int8_t** restrict a,
    const void* restrict w,
    int8_t* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    size_t a_offset,
    const int8_t* zero,
    const union xnn_qs8_qc8w_conv_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(mr != 0);
  assert(mr <= 4);
  assert(nc != 0);
  assert(kc != 0);
  assert(ks != 0);
  assert(ks % (4 * sizeof(void*)) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  int8_t* c0 = c;
  int8_t* c1 = (int8_t*) ((uintptr_t) c0 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 2) {
    c1 = c0;
  }
  int8_t* c2 = (int8_t*) ((uintptr_t) c1 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 2) {
    c2 = c1;
  }
  int8_t* c3 = (int8_t*) ((uintptr_t) c2 + cm_stride);
  if XNN_UNPREDICTABLE(mr != 4) {
    c3 = c2;
  }

  do {
    int32_t vacc0x0 = unaligned_indexed_load_s32(w, 0);
    int32_t vacc0x1 = unaligned_indexed_load_s32(w, 1);
    int32_t vacc1x0 = vacc0x0;
    int32_t vacc1x1 = vacc0x1;
    int32_t vacc2x0 = vacc0x0;
    int32_t vacc2x1 = vacc0x1;
    int32_t vacc3x0 = vacc0x0;
    int32_t vacc3x1 = vacc0x1;
    w = (const void*) ((const int32_t*) w + 2);

    size_t p = ks;
    do {
      const int8_t* restrict a0 = a[0];
      assert(a0 != NULL);
      if XNN_UNPREDICTABLE(a0 != zero) {
        a0 = (const int8_t*) ((uintptr_t) a0 + a_offset);
      }
      const int8_t* restrict a1 = a[1];
      assert(a1 != NULL);
      if XNN_UNPREDICTABLE(a1 != zero) {
        a1 = (const int8_t*) ((uintptr_t) a1 + a_offset);
      }
      const int8_t* restrict a2 = a[2];
      assert(a2 != NULL);
      if XNN_UNPREDICTABLE(a2 != zero) {
        a2 = (const int8_t*) ((uintptr_t) a2 + a_offset);
      }
      const int8_t* restrict a3 = a[3];
      assert(a3 != NULL);
      if XNN_UNPREDICTABLE(a3 != zero) {
        a3 = (const int8_t*) ((uintptr_t) a3 + a_offset);
      }
      a += 4;

      size_t k = kc;
      do {
        const int32_t va0 = (int32_t) *a0++;
        const int32_t va1 = (int32_t) *a1++;
        const int32_t va2 = (int32_t) *a2++;
        const int32_t va3 = (int32_t) *a3++;

        const int32_t vb0 = (int32_t) ((const int8_t*) w)[0];
        const int32_t vb1 = (int32_t) ((const int8_t*) w)[1];
        w = (const void*) ((const int8_t*) w + 2);

        vacc0x0 += va0 * vb0;
        vacc0x1 += va0 * vb1;
        vacc1x0 += va1 * vb0;
        vacc1x1 += va1 * vb1;
        vacc2x0 += va2 * vb0;
        vacc2x1 += va2 * vb1;
        vacc3x0 += va3 * vb0;
        vacc3x1 += va3 * vb1;

        k -= sizeof(int8_t);
      } while (k != 0);
      p -= 4 * sizeof(void*);
    } while (p != 0);

    float vfpacc0x0 = (float) vacc0x0;
    float vfpacc0x1 = (float) vacc0x1;
    float vfpacc1x0 = (float) vacc1x0;
    float vfpacc1x1 = (float) vacc1x1;
    float vfpacc2x0 = (float) vacc2x0;
    float vfpacc2x1 = (float) vacc2x1;
    float vfpacc3x0 = (float) vacc3x0;
    float vfpacc3x1 = (float) vacc3x1;

    const float vscale0 = unaligned_indexed_load_f32(w, 0);
    vfpacc0x0 *= vscale0;
    vfpacc1x0 *= vscale0;
    vfpacc2x0 *= vscale0;
    vfpacc3x0 *= vscale0;
    const float vscale1 = unaligned_indexed_load_f32(w, 1);
    vfpacc0x1 *= vscale1;
    vfpacc1x1 *= vscale1;
    vfpacc2x1 *= vscale1;
    vfpacc3x1 *= vscale1;
    w = (const void*) ((const float*) w + 2);

    const float vmagic_bias = params->fp32_scalar_imagic.magic_bias;
    vfpacc0x0 += vmagic_bias;
    vfpacc0x1 += vmagic_bias;
    vfpacc1x0 += vmagic_bias;
    vfpacc1x1 += vmagic_bias;
    vfpacc2x0 += vmagic_bias;
    vfpacc2x1 += vmagic_bias;
    vfpacc3x0 += vmagic_bias;
    vfpacc3x1 += vmagic_bias;

    int32_t vout0x0 = (int32_t) float_as_uint32(vfpacc0x0);
    int32_t vout0x1 = (int32_t) float_as_uint32(vfpacc0x1);
    int32_t vout1x0 = (int32_t) float_as_uint32(vfpacc1x0);
    int32_t vout1x1 = (int32_t) float_as_uint32(vfpacc1x1);
    int32_t vout2x0 = (int32_t) float_as_uint32(vfpacc2x0);
    int32_t vout2x1 = (int32_t) float_as_uint32(vfpacc2x1);
    int32_t vout3x0 = (int32_t) float_as_uint32(vfpacc3x0);
    int32_t vout3x1 = (int32_t) float_as_uint32(vfpacc3x1);

    const int32_t vmagic_min = params->fp32_scalar_imagic.magic_min;
    vout0x0 = math_max_s32(vout0x0, vmagic_min);
    vout0x1 = math_max_s32(vout0x1, vmagic_min);
    vout1x0 = math_max_s32(vout1x0, vmagic_min);
    vout1x1 = math_max_s32(vout1x1, vmagic_min);
    vout2x0 = math_max_s32(vout2x0, vmagic_min);
    vout2x1 = math_max_s32(vout2x1, vmagic_min);
    vout3x0 = math_max_s32(vout3x0, vmagic_min);
    vout3x1 = math_max_s32(vout3x1, vmagic_min);

    const int32_t vmagic_max = params->fp32_scalar_imagic.magic_max;
    vout0x0 = math_min_s32(vout0x0, vmagic_max);
    vout0x1 = math_min_s32(vout0x1, vmagic_max);
    vout1x0 = math_min_s32(vout1x0, vmagic_max);
    vout1x1 = math_min_s32(vout1x1, vmagic_max);
    vout2x0 = math_min_s32(vout2x0, vmagic_max);
    vout2x1 = math_min_s32(vout2x1, vmagic_max);
    vout3x0 = math_min_s32(vout3x0, vmagic_max);
    vout3x1 = math_min_s32(vout3x1, vmagic_max);

    const int32_t vmagic_bias_less_zero_point = params->fp32_scalar_imagic.magic_bias_less_zero_point;
    vout0x0 -= vmagic_bias_less_zero_point;
    vout0x1 -= vmagic_bias_less_zero_point;
    vout1x0 -= vmagic_bias_less_zero_point;
    vout1x1 -= vmagic_bias_less_zero_point;
    vout2x0 -= vmagic_bias_less_zero_point;
    vout2x1 -= vmagic_bias_less_zero_point;
    vout3x0 -= vmagic_bias_less_zero_point;
    vout3x1 -= vmagic_bias_less_zero_point;

    if XNN_LIKELY(nc >= 2) {
      c3[0] = (int8_t) vout3x0;
      c3[1] = (int8_t) vout3x1;
      c2[0] = (int8_t) vout2x0;
      c2[1] = (int8_t) vout2x1;
      c1[0] = (int8_t) vout1x0;
      c1[1] = (int8_t) vout1x1;
      c0[0] = (int8_t) vout0x0;
      c0[1] = (int8_t) vout0x1;

      c3 = (int8_t*) ((uintptr_t) c3 + cn_stride);
      c2 = (int8_t*) ((uintptr_t) c2 + cn_stride);
      c1 = (int8_t*) ((uintptr_t) c1 + cn_stride);
      c0 = (int8_t*) ((uintptr_t) c0 + cn_stride);

      a = (const int8_t**restrict) ((uintptr_t) a - ks);
      nc -= 2;
    } else {
      if (nc & 1) {
        c3[0] = (int8_t) vout3x0;
        c2[0] = (int8_t) vout2x0;
        c1[0] = (int8_t) vout1x0;
        c0[0] = (int8_t) vout0x0;
      }

      nc = 0;
    }
  } while (nc != 0);
}
