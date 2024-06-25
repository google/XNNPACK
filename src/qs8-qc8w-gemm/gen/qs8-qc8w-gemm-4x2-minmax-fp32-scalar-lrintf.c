// Auto-generated file. Do not edit!
//   Template: src/qs8-gemm/scalar.c.in
//   Generator: tools/xngen
//
// Copyright 2021 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <math.h>

#include "xnnpack/gemm.h"
#include "xnnpack/math.h"
#include "xnnpack/unaligned.h"


void xnn_qs8_qc8w_gemm_minmax_fp32_ukernel_4x2__scalar_lrintf(
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
  assert(mr <= 4);
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
  const int8_t* a3 = (const int8_t*) ((uintptr_t) a2 + a_stride);
  int8_t* c3 = (int8_t*) ((uintptr_t) c2 + cm_stride);
  if XNN_UNPREDICTABLE(mr != 4) {
    a3 = a2;
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
    w = (const int32_t*) w + 2;

    size_t k = kc;
    do {
      const int32_t va0 = (int32_t) *a0++;
      const int32_t va1 = (int32_t) *a1++;
      const int32_t va2 = (int32_t) *a2++;
      const int32_t va3 = (int32_t) *a3++;

      const int32_t vb0 = (int32_t) ((const int8_t*) w)[0];
      const int32_t vb1 = (int32_t) ((const int8_t*) w)[1];
      w = (const int8_t*) w + 2;

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

    const float voutput_min_less_zero_point = params->fp32_scalar_lrintf.output_min_less_zero_point;
    vfpacc0x0 = math_max_f32(vfpacc0x0, voutput_min_less_zero_point);
    vfpacc0x1 = math_max_f32(vfpacc0x1, voutput_min_less_zero_point);
    vfpacc1x0 = math_max_f32(vfpacc1x0, voutput_min_less_zero_point);
    vfpacc1x1 = math_max_f32(vfpacc1x1, voutput_min_less_zero_point);
    vfpacc2x0 = math_max_f32(vfpacc2x0, voutput_min_less_zero_point);
    vfpacc2x1 = math_max_f32(vfpacc2x1, voutput_min_less_zero_point);
    vfpacc3x0 = math_max_f32(vfpacc3x0, voutput_min_less_zero_point);
    vfpacc3x1 = math_max_f32(vfpacc3x1, voutput_min_less_zero_point);

    const float voutput_max_less_zero_point = params->fp32_scalar_lrintf.output_max_less_zero_point;
    vfpacc0x0 = math_min_f32(vfpacc0x0, voutput_max_less_zero_point);
    vfpacc0x1 = math_min_f32(vfpacc0x1, voutput_max_less_zero_point);
    vfpacc1x0 = math_min_f32(vfpacc1x0, voutput_max_less_zero_point);
    vfpacc1x1 = math_min_f32(vfpacc1x1, voutput_max_less_zero_point);
    vfpacc2x0 = math_min_f32(vfpacc2x0, voutput_max_less_zero_point);
    vfpacc2x1 = math_min_f32(vfpacc2x1, voutput_max_less_zero_point);
    vfpacc3x0 = math_min_f32(vfpacc3x0, voutput_max_less_zero_point);
    vfpacc3x1 = math_min_f32(vfpacc3x1, voutput_max_less_zero_point);

    const int32_t vrndacc0x0 = (int32_t) lrintf(vfpacc0x0);
    const int32_t vrndacc0x1 = (int32_t) lrintf(vfpacc0x1);
    const int32_t vrndacc1x0 = (int32_t) lrintf(vfpacc1x0);
    const int32_t vrndacc1x1 = (int32_t) lrintf(vfpacc1x1);
    const int32_t vrndacc2x0 = (int32_t) lrintf(vfpacc2x0);
    const int32_t vrndacc2x1 = (int32_t) lrintf(vfpacc2x1);
    const int32_t vrndacc3x0 = (int32_t) lrintf(vfpacc3x0);
    const int32_t vrndacc3x1 = (int32_t) lrintf(vfpacc3x1);

    const int32_t voutput_zero_point = params->fp32_scalar_lrintf.output_zero_point;
    int32_t vout0x0 = vrndacc0x0 + voutput_zero_point;
    int32_t vout0x1 = vrndacc0x1 + voutput_zero_point;
    int32_t vout1x0 = vrndacc1x0 + voutput_zero_point;
    int32_t vout1x1 = vrndacc1x1 + voutput_zero_point;
    int32_t vout2x0 = vrndacc2x0 + voutput_zero_point;
    int32_t vout2x1 = vrndacc2x1 + voutput_zero_point;
    int32_t vout3x0 = vrndacc3x0 + voutput_zero_point;
    int32_t vout3x1 = vrndacc3x1 + voutput_zero_point;

    if XNN_LIKELY(nc >= 2) {
      c0[0] = (int8_t) vout0x0;
      c0[1] = (int8_t) vout0x1;
      c1[0] = (int8_t) vout1x0;
      c1[1] = (int8_t) vout1x1;
      c2[0] = (int8_t) vout2x0;
      c2[1] = (int8_t) vout2x1;
      c3[0] = (int8_t) vout3x0;
      c3[1] = (int8_t) vout3x1;

      a0 = (const int8_t*) ((uintptr_t) a0 - kc);
      a1 = (const int8_t*) ((uintptr_t) a1 - kc);
      a2 = (const int8_t*) ((uintptr_t) a2 - kc);
      a3 = (const int8_t*) ((uintptr_t) a3 - kc);

      c0 = (int8_t*) ((uintptr_t) c0 + cn_stride);
      c1 = (int8_t*) ((uintptr_t) c1 + cn_stride);
      c2 = (int8_t*) ((uintptr_t) c2 + cn_stride);
      c3 = (int8_t*) ((uintptr_t) c3 + cn_stride);

      nc -= 2;
    } else {
      if (nc & 1) {
        c0[0] = (int8_t) vout0x0;
        c1[0] = (int8_t) vout1x0;
        c2[0] = (int8_t) vout2x0;
        c3[0] = (int8_t) vout3x0;
      }

      nc = 0;
    }
  } while (nc != 0);
}
