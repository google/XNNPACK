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
#include <xnnpack/unaligned.h>


void xnn_qs8_f32_gemm_minmax_ukernel_2x4__scalar(
    size_t mr,
    size_t nc,
    size_t kc,
    const int8_t* restrict a,
    size_t a_stride,
    const void* restrict w,
    float* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const float* input_scale,
    const int32_t* input_zp,
    const union xnn_qs8_conv_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(mr != 0);
  assert(mr <= 2);
  assert(nc != 0);
  assert(kc != 0);

  const int8_t* a0 = a;
  float* c0 = c;
  int32_t vzp[2];
  float vscale[2];
  vzp[0] = unaligned_indexed_load_s32(input_zp, 0);
  vscale[0] = unaligned_indexed_load_f32(input_scale, 0);
  vzp[1] = mr > 1 ? unaligned_indexed_load_s32(input_zp, 1) : vzp[0];
  vscale[1] = mr > 1 ? unaligned_indexed_load_f32(input_scale, 1) : vscale[0];
  const int8_t* a1 = (const int8_t*) ((uintptr_t) a0 + a_stride);
  float* c1 = (float*) ((uintptr_t) c0 + cm_stride);
  if XNN_UNPREDICTABLE(mr != 2) {
    a1 = a0;
    c1 = c0;
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
    w = (const void*) ((const int32_t*) w + 4);

    vacc0x0 *= vzp[0];
    vacc0x1 *= vzp[0];
    vacc0x2 *= vzp[0];
    vacc0x3 *= vzp[0];
    vacc1x0 *= vzp[1];
    vacc1x1 *= vzp[1];
    vacc1x2 *= vzp[1];
    vacc1x3 *= vzp[1];
    float vbias0 = ((const float*) w)[0];
    float vbias1 = ((const float*) w)[1];
    float vbias2 = ((const float*) w)[2];
    float vbias3 = ((const float*) w)[3];
    w = (const void*) ((const float*) w + 4);
    size_t k = kc;
    do {
      const int32_t va0 = (int32_t) *a0++;
      const int32_t va1 = (int32_t) *a1++;

      const int32_t vb0 = (int32_t) ((const int8_t*) w)[0];
      const int32_t vb1 = (int32_t) ((const int8_t*) w)[1];
      const int32_t vb2 = (int32_t) ((const int8_t*) w)[2];
      const int32_t vb3 = (int32_t) ((const int8_t*) w)[3];
      w = (const void*) ((const int8_t*) w + 4);

      vacc0x0 += va0 * vb0;
      vacc0x1 += va0 * vb1;
      vacc0x2 += va0 * vb2;
      vacc0x3 += va0 * vb3;
      vacc1x0 += va1 * vb0;
      vacc1x1 += va1 * vb1;
      vacc1x2 += va1 * vb2;
      vacc1x3 += va1 * vb3;

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
    vfpacc0x0 *= vscale[0];
    vfpacc1x0 *= vscale[1];
    vfpacc0x1 *= vscale[0];
    vfpacc1x1 *= vscale[1];
    vfpacc0x2 *= vscale[0];
    vfpacc1x2 *= vscale[1];
    vfpacc0x3 *= vscale[0];
    vfpacc1x3 *= vscale[1];
    float vout0x0 = vfpacc0x0 + vbias0;
    float vout1x0 = vfpacc1x0 + vbias0;
    float vout0x1 = vfpacc0x1 + vbias1;
    float vout1x1 = vfpacc1x1 + vbias1;
    float vout0x2 = vfpacc0x2 + vbias2;
    float vout1x2 = vfpacc1x2 + vbias2;
    float vout0x3 = vfpacc0x3 + vbias3;
    float vout1x3 = vfpacc1x3 + vbias3;

    if XNN_LIKELY(nc >= 4) {
      c0[0] = (float) vout0x0;
      c0[1] = (float) vout0x1;
      c0[2] = (float) vout0x2;
      c0[3] = (float) vout0x3;
      c1[0] = (float) vout1x0;
      c1[1] = (float) vout1x1;
      c1[2] = (float) vout1x2;
      c1[3] = (float) vout1x3;

      a0 = (const int8_t*) ((uintptr_t) a0 - kc);
      a1 = (const int8_t*) ((uintptr_t) a1 - kc);

      c0 = (float*) ((uintptr_t) c0 + cn_stride);
      c1 = (float*) ((uintptr_t) c1 + cn_stride);

      nc -= 4;
    } else {
      if (nc & 2) {
        c0[0] = (float) vout0x0;
        c0[1] = (float) vout0x1;
        vout0x0 = vout0x2;
        c0 += 2;
        c1[0] = (float) vout1x0;
        c1[1] = (float) vout1x1;
        vout1x0 = vout1x2;
        c1 += 2;
      }
      if (nc & 1) {
        c0[0] = (float) vout0x0;
        c1[0] = (float) vout1x0;
      }

      nc = 0;
    }
  } while (nc != 0);
}
