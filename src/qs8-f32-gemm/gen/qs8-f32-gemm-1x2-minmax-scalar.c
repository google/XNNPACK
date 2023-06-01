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


void xnn_qs8_f32_gemm_minmax_ukernel_1x2__scalar(
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
    int32_t vacc0x0 = unaligned_indexed_load_s32(w, 0);
    int32_t vacc0x1 = unaligned_indexed_load_s32(w, 1);
    w = (const void*) ((const int32_t*) w + 2);

    vacc0x0 *= vzp[0];
    vacc0x1 *= vzp[0];
    float vbias0 = unaligned_indexed_load_f32(w, 0);
    float vbias1 = unaligned_indexed_load_f32(w, 1);
    w = (const void*) ((const float*) w + 2);
    size_t k = kc;
    do {
      const int32_t va0 = (int32_t) *a0++;

      const int32_t vb0 = (int32_t) ((const int8_t*) w)[0];
      const int32_t vb1 = (int32_t) ((const int8_t*) w)[1];
      w = (const void*) ((const int8_t*) w + 2);

      vacc0x0 += va0 * vb0;
      vacc0x1 += va0 * vb1;

      k -= sizeof(int8_t);
    } while (k != 0);

    float vfpacc0x0 = (float) vacc0x0;
    float vfpacc0x1 = (float) vacc0x1;
    vfpacc0x0 *= vscale[0];
    vfpacc0x1 *= vscale[0];
    float vout0x0 = vfpacc0x0 + vbias0;
    float vout0x1 = vfpacc0x1 + vbias1;

    if XNN_LIKELY(nc >= 2) {
      c0[0] = (float) vout0x0;
      c0[1] = (float) vout0x1;

      a0 = (const int8_t*) ((uintptr_t) a0 - kc);

      c0 = (float*) ((uintptr_t) c0 + cn_stride);

      nc -= 2;
    } else {
      if (nc & 1) {
        c0[0] = (float) vout0x0;
      }

      nc = 0;
    }
  } while (nc != 0);
}
