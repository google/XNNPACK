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


void xnn_qs8_f32_gemm_minmax_ukernel_1x8__scalar(
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
  assert(mr <= 1);
  assert(nc != 0);
  assert(kc != 0);

  const int8_t* a0 = a;
  float* c0 = c;
  int32_t vzp[1];
  float vscale[1];
  vzp[0] = unaligned_indexed_load_s32(input_zp, 0);
  vscale[0] = unaligned_indexed_load_f32(input_scale, 0);

  do {
    int32_t vacc0x0 = ((const int32_t*) w)[0];
    int32_t vacc0x1 = ((const int32_t*) w)[1];
    int32_t vacc0x2 = ((const int32_t*) w)[2];
    int32_t vacc0x3 = ((const int32_t*) w)[3];
    int32_t vacc0x4 = ((const int32_t*) w)[4];
    int32_t vacc0x5 = ((const int32_t*) w)[5];
    int32_t vacc0x6 = ((const int32_t*) w)[6];
    int32_t vacc0x7 = ((const int32_t*) w)[7];
    w = (const void*) ((const int32_t*) w + 8);

    vacc0x0 *= vzp[0];
    vacc0x1 *= vzp[0];
    vacc0x2 *= vzp[0];
    vacc0x3 *= vzp[0];
    vacc0x4 *= vzp[0];
    vacc0x5 *= vzp[0];
    vacc0x6 *= vzp[0];
    vacc0x7 *= vzp[0];
    float vbias0 = ((const float*) w)[0];
    float vbias1 = ((const float*) w)[1];
    float vbias2 = ((const float*) w)[2];
    float vbias3 = ((const float*) w)[3];
    float vbias4 = ((const float*) w)[4];
    float vbias5 = ((const float*) w)[5];
    float vbias6 = ((const float*) w)[6];
    float vbias7 = ((const float*) w)[7];
    w = (const void*) ((const float*) w + 8);
    size_t k = kc;
    do {
      const int32_t va0 = (int32_t) *a0++;

      const int32_t vb0 = (int32_t) ((const int8_t*) w)[0];
      const int32_t vb1 = (int32_t) ((const int8_t*) w)[1];
      const int32_t vb2 = (int32_t) ((const int8_t*) w)[2];
      const int32_t vb3 = (int32_t) ((const int8_t*) w)[3];
      const int32_t vb4 = (int32_t) ((const int8_t*) w)[4];
      const int32_t vb5 = (int32_t) ((const int8_t*) w)[5];
      const int32_t vb6 = (int32_t) ((const int8_t*) w)[6];
      const int32_t vb7 = (int32_t) ((const int8_t*) w)[7];
      w = (const void*) ((const int8_t*) w + 8);

      vacc0x0 += va0 * vb0;
      vacc0x1 += va0 * vb1;
      vacc0x2 += va0 * vb2;
      vacc0x3 += va0 * vb3;
      vacc0x4 += va0 * vb4;
      vacc0x5 += va0 * vb5;
      vacc0x6 += va0 * vb6;
      vacc0x7 += va0 * vb7;

      k -= sizeof(int8_t);
    } while (k != 0);

    float vfpacc0x0 = (float) vacc0x0;
    float vfpacc0x1 = (float) vacc0x1;
    float vfpacc0x2 = (float) vacc0x2;
    float vfpacc0x3 = (float) vacc0x3;
    float vfpacc0x4 = (float) vacc0x4;
    float vfpacc0x5 = (float) vacc0x5;
    float vfpacc0x6 = (float) vacc0x6;
    float vfpacc0x7 = (float) vacc0x7;
    vfpacc0x0 *= vscale[0];
    vfpacc0x1 *= vscale[0];
    vfpacc0x2 *= vscale[0];
    vfpacc0x3 *= vscale[0];
    vfpacc0x4 *= vscale[0];
    vfpacc0x5 *= vscale[0];
    vfpacc0x6 *= vscale[0];
    vfpacc0x7 *= vscale[0];
    float vout0x0 = vfpacc0x0 + vbias0;
    float vout0x1 = vfpacc0x1 + vbias1;
    float vout0x2 = vfpacc0x2 + vbias2;
    float vout0x3 = vfpacc0x3 + vbias3;
    float vout0x4 = vfpacc0x4 + vbias4;
    float vout0x5 = vfpacc0x5 + vbias5;
    float vout0x6 = vfpacc0x6 + vbias6;
    float vout0x7 = vfpacc0x7 + vbias7;

    if XNN_LIKELY(nc >= 8) {
      c0[0] = (float) vout0x0;
      c0[1] = (float) vout0x1;
      c0[2] = (float) vout0x2;
      c0[3] = (float) vout0x3;
      c0[4] = (float) vout0x4;
      c0[5] = (float) vout0x5;
      c0[6] = (float) vout0x6;
      c0[7] = (float) vout0x7;

      a0 = (const int8_t*) ((uintptr_t) a0 - kc);

      c0 = (float*) ((uintptr_t) c0 + cn_stride);

      nc -= 8;
    } else {
      if (nc & 4) {
        c0[0] = (float) vout0x0;
        c0[1] = (float) vout0x1;
        c0[2] = (float) vout0x2;
        c0[3] = (float) vout0x3;
        vout0x0 = vout0x4;
        vout0x1 = vout0x5;
        vout0x2 = vout0x6;
        c0 += 4;
      }
      if (nc & 2) {
        c0[0] = (float) vout0x0;
        c0[1] = (float) vout0x1;
        vout0x0 = vout0x2;
        vout0x1 = vout0x3;
        vout0x2 = vout0x4;
        vout0x3 = vout0x5;
        vout0x4 = vout0x6;
        c0 += 2;
      }
      if (nc & 1) {
        c0[0] = (float) vout0x0;
      }

      nc = 0;
    }
  } while (nc != 0);
}
