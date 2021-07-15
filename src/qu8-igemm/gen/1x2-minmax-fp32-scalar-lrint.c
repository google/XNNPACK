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


void xnn_qu8_igemm_minmax_fp32_ukernel_1x2__scalar_lrint(
    size_t mr,
    size_t nc,
    size_t kc,
    size_t ks,
    const uint8_t**restrict a,
    const void*restrict w,
    uint8_t*restrict c,
    size_t cm_stride,
    size_t cn_stride,
    size_t a_offset,
    const uint8_t* zero,
    const union xnn_qu8_conv_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(mr != 0);
  assert(mr <= 1);
  assert(nc != 0);
  assert(kc != 0);
  assert(ks != 0);
  assert(ks % (1 * sizeof(void*)) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  uint8_t* c0 = c;

  const int32_t vb_zero_point = params->fp32_scalar_lrint.kernel_zero_point;
  do {
    int32_t vacc0x0 = ((const int32_t*) w)[0];
    int32_t vacc0x1 = ((const int32_t*) w)[1];
    w = (const void*) ((const int32_t*) w + 2);

    size_t p = ks;
    do {
      const uint8_t* restrict a0 = a[0];
      assert(a0 != NULL);
      if XNN_UNPREDICTABLE(a0 != zero) {
        a0 = (const uint8_t*) ((uintptr_t) a0 + a_offset);
      }
      a += 1;

      size_t k = kc;
      do {
        const int32_t va0 = (int32_t) (uint32_t) *a0++;

        const int32_t vb0 = (int32_t) (uint32_t) ((const uint8_t*) w)[0] - vb_zero_point;
        const int32_t vb1 = (int32_t) (uint32_t) ((const uint8_t*) w)[1] - vb_zero_point;
        w = (const void*) ((const uint8_t*) w + 2);

        vacc0x0 += va0 * vb0;
        vacc0x1 += va0 * vb1;

        k -= sizeof(uint8_t);
      } while (k != 0);
      p -= 1 * sizeof(void*);
    } while (p != 0);

    float vfpacc0x0 = (float) vacc0x0;
    float vfpacc0x1 = (float) vacc0x1;

    const float vscale = params->fp32_scalar_lrint.scale;
    vfpacc0x0 *= vscale;
    vfpacc0x1 *= vscale;

    long vrndacc0x0 = lrintf(vfpacc0x0);
    long vrndacc0x1 = lrintf(vfpacc0x1);

    const long voutput_min_less_zero_point = params->fp32_scalar_lrint.output_min_less_zero_point;
    vrndacc0x0 = XNN_UNPREDICTABLE(vrndacc0x0 < voutput_min_less_zero_point) ? voutput_min_less_zero_point : vrndacc0x0;
    vrndacc0x1 = XNN_UNPREDICTABLE(vrndacc0x1 < voutput_min_less_zero_point) ? voutput_min_less_zero_point : vrndacc0x1;

    const long voutput_max_less_zero_point = params->fp32_scalar_lrint.output_max_less_zero_point;
    vrndacc0x0 = XNN_UNPREDICTABLE(vrndacc0x0 > voutput_max_less_zero_point) ? voutput_max_less_zero_point : vrndacc0x0;
    vrndacc0x1 = XNN_UNPREDICTABLE(vrndacc0x1 > voutput_max_less_zero_point) ? voutput_max_less_zero_point : vrndacc0x1;

    const int32_t voutput_zero_point = params->fp32_scalar_lrint.output_zero_point;
    int32_t vout0x0 = (int32_t) vrndacc0x0 + voutput_zero_point;
    int32_t vout0x1 = (int32_t) vrndacc0x1 + voutput_zero_point;

    if XNN_LIKELY(nc >= 2) {
      c0[0] = (uint8_t) vout0x0;
      c0[1] = (uint8_t) vout0x1;

      c0 = (uint8_t*) ((uintptr_t) c0 + cn_stride);

      a = (const uint8_t**restrict) ((uintptr_t) a - ks);
      nc -= 2;
    } else {
      if (nc & 1) {
        c0[0] = (uint8_t) vout0x0;
      }

      nc = 0;
    }
  } while (nc != 0);
}
