// Auto-generated file. Do not edit!
//   Template: src/qs8-igemm/scalar.c.in
//   Generator: tools/xngen
//
// Copyright 2021 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <math.h>

#include <xnnpack/math.h>
#include <xnnpack/gemm.h>


void xnn_qc8_igemm_minmax_fp32_ukernel_3x2__scalar_lrint(
    size_t mr,
    size_t nc,
    size_t kc,
    size_t ks,
    const int8_t**restrict a,
    const void*restrict w,
    int8_t*restrict c,
    size_t cm_stride,
    size_t cn_stride,
    size_t a_offset,
    const int8_t* zero,
    const union xnn_qs8_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(mr != 0);
  assert(mr <= 3);
  assert(nc != 0);
  assert(kc != 0);
  assert(ks != 0);
  assert(ks % (3 * sizeof(void*)) == 0);
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

  do {
    int32_t vacc0x0 = ((const int32_t*) w)[0];
    int32_t vacc0x1 = ((const int32_t*) w)[1];
    int32_t vacc1x0 = vacc0x0;
    int32_t vacc1x1 = vacc0x1;
    int32_t vacc2x0 = vacc0x0;
    int32_t vacc2x1 = vacc0x1;
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
      a += 3;

      size_t k = kc;
      do {
        const int32_t va0 = (int32_t) *a0++;
        const int32_t va1 = (int32_t) *a1++;
        const int32_t va2 = (int32_t) *a2++;

        const int32_t vb0 = (int32_t) ((const int8_t*) w)[0];
        const int32_t vb1 = (int32_t) ((const int8_t*) w)[1];
        w = (const void*) ((const int8_t*) w + 2);

        vacc0x0 += va0 * vb0;
        vacc0x1 += va0 * vb1;
        vacc1x0 += va1 * vb0;
        vacc1x1 += va1 * vb1;
        vacc2x0 += va2 * vb0;
        vacc2x1 += va2 * vb1;

        k -= sizeof(int8_t);
      } while (k != 0);
      p -= 3 * sizeof(void*);
    } while (p != 0);

    float vfpacc0x0 = (float) vacc0x0;
    float vfpacc0x1 = (float) vacc0x1;
    float vfpacc1x0 = (float) vacc1x0;
    float vfpacc1x1 = (float) vacc1x1;
    float vfpacc2x0 = (float) vacc2x0;
    float vfpacc2x1 = (float) vacc2x1;

    const float vscale0 = ((const float*) w)[0];
    vfpacc0x0 *= vscale0;
    vfpacc1x0 *= vscale0;
    vfpacc2x0 *= vscale0;
    const float vscale1 = ((const float*) w)[1];
    vfpacc0x1 *= vscale1;
    vfpacc1x1 *= vscale1;
    vfpacc2x1 *= vscale1;
    w = (const void*) ((const float*) w + 2);

    long vrndacc0x0 = lrintf(vfpacc0x0);
    long vrndacc0x1 = lrintf(vfpacc0x1);
    long vrndacc1x0 = lrintf(vfpacc1x0);
    long vrndacc1x1 = lrintf(vfpacc1x1);
    long vrndacc2x0 = lrintf(vfpacc2x0);
    long vrndacc2x1 = lrintf(vfpacc2x1);

    const long voutput_min_less_zero_point = params->scalar_lrint.output_min_less_zero_point;
    vrndacc0x0 = XNN_UNPREDICTABLE(vrndacc0x0 < voutput_min_less_zero_point) ? voutput_min_less_zero_point : vrndacc0x0;
    vrndacc0x1 = XNN_UNPREDICTABLE(vrndacc0x1 < voutput_min_less_zero_point) ? voutput_min_less_zero_point : vrndacc0x1;
    vrndacc1x0 = XNN_UNPREDICTABLE(vrndacc1x0 < voutput_min_less_zero_point) ? voutput_min_less_zero_point : vrndacc1x0;
    vrndacc1x1 = XNN_UNPREDICTABLE(vrndacc1x1 < voutput_min_less_zero_point) ? voutput_min_less_zero_point : vrndacc1x1;
    vrndacc2x0 = XNN_UNPREDICTABLE(vrndacc2x0 < voutput_min_less_zero_point) ? voutput_min_less_zero_point : vrndacc2x0;
    vrndacc2x1 = XNN_UNPREDICTABLE(vrndacc2x1 < voutput_min_less_zero_point) ? voutput_min_less_zero_point : vrndacc2x1;

    const long voutput_max_less_zero_point = params->scalar_lrint.output_max_less_zero_point;
    vrndacc0x0 = XNN_UNPREDICTABLE(vrndacc0x0 > voutput_max_less_zero_point) ? voutput_max_less_zero_point : vrndacc0x0;
    vrndacc0x1 = XNN_UNPREDICTABLE(vrndacc0x1 > voutput_max_less_zero_point) ? voutput_max_less_zero_point : vrndacc0x1;
    vrndacc1x0 = XNN_UNPREDICTABLE(vrndacc1x0 > voutput_max_less_zero_point) ? voutput_max_less_zero_point : vrndacc1x0;
    vrndacc1x1 = XNN_UNPREDICTABLE(vrndacc1x1 > voutput_max_less_zero_point) ? voutput_max_less_zero_point : vrndacc1x1;
    vrndacc2x0 = XNN_UNPREDICTABLE(vrndacc2x0 > voutput_max_less_zero_point) ? voutput_max_less_zero_point : vrndacc2x0;
    vrndacc2x1 = XNN_UNPREDICTABLE(vrndacc2x1 > voutput_max_less_zero_point) ? voutput_max_less_zero_point : vrndacc2x1;

    const int32_t voutput_zero_point = params->scalar_lrint.output_zero_point;
    int32_t vout0x0 = (int32_t) vrndacc0x0 + voutput_zero_point;
    int32_t vout0x1 = (int32_t) vrndacc0x1 + voutput_zero_point;
    int32_t vout1x0 = (int32_t) vrndacc1x0 + voutput_zero_point;
    int32_t vout1x1 = (int32_t) vrndacc1x1 + voutput_zero_point;
    int32_t vout2x0 = (int32_t) vrndacc2x0 + voutput_zero_point;
    int32_t vout2x1 = (int32_t) vrndacc2x1 + voutput_zero_point;

    if XNN_LIKELY(nc >= 2) {
      c2[0] = (int8_t) vout2x0;
      c2[1] = (int8_t) vout2x1;
      c1[0] = (int8_t) vout1x0;
      c1[1] = (int8_t) vout1x1;
      c0[0] = (int8_t) vout0x0;
      c0[1] = (int8_t) vout0x1;

      c2 = (int8_t*) ((uintptr_t) c2 + cn_stride);
      c1 = (int8_t*) ((uintptr_t) c1 + cn_stride);
      c0 = (int8_t*) ((uintptr_t) c0 + cn_stride);

      a = (const int8_t**restrict) ((uintptr_t) a - ks);
      nc -= 2;
    } else {
      if (nc & 1) {
        c2[0] = (int8_t) vout2x0;
        c1[0] = (int8_t) vout1x0;
        c0[0] = (int8_t) vout0x0;
      }

      nc = 0;
    }
  } while (nc != 0);
}
