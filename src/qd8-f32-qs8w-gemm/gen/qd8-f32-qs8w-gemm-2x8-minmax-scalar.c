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


void xnn_qd8_f32_qs8w_gemm_minmax_ukernel_2x8__scalar(
    size_t mr,
    size_t nc,
    size_t kc,
    const int8_t* restrict a,
    size_t a_stride,
    const void* restrict w,
    float* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)],
    const struct xnn_qd8_quantization_params quantization_params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(mr != 0);
  assert(mr <= 2);
  assert(nc != 0);
  assert(kc != 0);

  const int8_t* a0 = a;
  float* c0 = c;
  const int8_t* a1 = (const int8_t*) ((uintptr_t) a0 + a_stride);
  float* c1 = (float*) ((uintptr_t) c0 + cm_stride);
  if XNN_UNPREDICTABLE(mr != 2) {
    a1 = a0;
    c1 = c0;
  }

  do {
    const int32_t vksum0 = ((const int32_t*) w)[0];
    const int32_t vksum1 = ((const int32_t*) w)[1];
    const int32_t vksum2 = ((const int32_t*) w)[2];
    const int32_t vksum3 = ((const int32_t*) w)[3];
    const int32_t vksum4 = ((const int32_t*) w)[4];
    const int32_t vksum5 = ((const int32_t*) w)[5];
    const int32_t vksum6 = ((const int32_t*) w)[6];
    const int32_t vksum7 = ((const int32_t*) w)[7];
    const int32_t vinput_zero_point0 = quantization_params[0].zero_point;
    int32_t vacc0x0 = vksum0 * vinput_zero_point0;
    int32_t vacc0x1 = vksum1 * vinput_zero_point0;
    int32_t vacc0x2 = vksum2 * vinput_zero_point0;
    int32_t vacc0x3 = vksum3 * vinput_zero_point0;
    int32_t vacc0x4 = vksum4 * vinput_zero_point0;
    int32_t vacc0x5 = vksum5 * vinput_zero_point0;
    int32_t vacc0x6 = vksum6 * vinput_zero_point0;
    int32_t vacc0x7 = vksum7 * vinput_zero_point0;
    const int32_t vinput_zero_point1 = quantization_params[1].zero_point;
    int32_t vacc1x0 = vksum0 * vinput_zero_point1;
    int32_t vacc1x1 = vksum1 * vinput_zero_point1;
    int32_t vacc1x2 = vksum2 * vinput_zero_point1;
    int32_t vacc1x3 = vksum3 * vinput_zero_point1;
    int32_t vacc1x4 = vksum4 * vinput_zero_point1;
    int32_t vacc1x5 = vksum5 * vinput_zero_point1;
    int32_t vacc1x6 = vksum6 * vinput_zero_point1;
    int32_t vacc1x7 = vksum7 * vinput_zero_point1;
    w = (const int32_t*) w + 8;

    size_t k = kc;
    do {
      const int32_t va0 = (int32_t) *a0++;
      const int32_t va1 = (int32_t) *a1++;

      const int32_t vb0 = (int32_t) ((const int8_t*) w)[0];
      const int32_t vb1 = (int32_t) ((const int8_t*) w)[1];
      const int32_t vb2 = (int32_t) ((const int8_t*) w)[2];
      const int32_t vb3 = (int32_t) ((const int8_t*) w)[3];
      const int32_t vb4 = (int32_t) ((const int8_t*) w)[4];
      const int32_t vb5 = (int32_t) ((const int8_t*) w)[5];
      const int32_t vb6 = (int32_t) ((const int8_t*) w)[6];
      const int32_t vb7 = (int32_t) ((const int8_t*) w)[7];
      w = (const int8_t*) w + 8;

      vacc0x0 += va0 * vb0;
      vacc0x1 += va0 * vb1;
      vacc0x2 += va0 * vb2;
      vacc0x3 += va0 * vb3;
      vacc0x4 += va0 * vb4;
      vacc0x5 += va0 * vb5;
      vacc0x6 += va0 * vb6;
      vacc0x7 += va0 * vb7;
      vacc1x0 += va1 * vb0;
      vacc1x1 += va1 * vb1;
      vacc1x2 += va1 * vb2;
      vacc1x3 += va1 * vb3;
      vacc1x4 += va1 * vb4;
      vacc1x5 += va1 * vb5;
      vacc1x6 += va1 * vb6;
      vacc1x7 += va1 * vb7;

      k -= sizeof(int8_t);
    } while (k != 0);

    float vout0x0 = (float) vacc0x0;
    float vout0x1 = (float) vacc0x1;
    float vout0x2 = (float) vacc0x2;
    float vout0x3 = (float) vacc0x3;
    float vout0x4 = (float) vacc0x4;
    float vout0x5 = (float) vacc0x5;
    float vout0x6 = (float) vacc0x6;
    float vout0x7 = (float) vacc0x7;
    float vout1x0 = (float) vacc1x0;
    float vout1x1 = (float) vacc1x1;
    float vout1x2 = (float) vacc1x2;
    float vout1x3 = (float) vacc1x3;
    float vout1x4 = (float) vacc1x4;
    float vout1x5 = (float) vacc1x5;
    float vout1x6 = (float) vacc1x6;
    float vout1x7 = (float) vacc1x7;

    const float vscale0 = quantization_params[0].scale;
    const float vbias0 = ((const float*) w)[0];
    vout0x0 = math_muladd_f32(vout0x0, vscale0, vbias0);
    const float vbias1 = ((const float*) w)[1];
    vout0x1 = math_muladd_f32(vout0x1, vscale0, vbias1);
    const float vbias2 = ((const float*) w)[2];
    vout0x2 = math_muladd_f32(vout0x2, vscale0, vbias2);
    const float vbias3 = ((const float*) w)[3];
    vout0x3 = math_muladd_f32(vout0x3, vscale0, vbias3);
    const float vbias4 = ((const float*) w)[4];
    vout0x4 = math_muladd_f32(vout0x4, vscale0, vbias4);
    const float vbias5 = ((const float*) w)[5];
    vout0x5 = math_muladd_f32(vout0x5, vscale0, vbias5);
    const float vbias6 = ((const float*) w)[6];
    vout0x6 = math_muladd_f32(vout0x6, vscale0, vbias6);
    const float vbias7 = ((const float*) w)[7];
    vout0x7 = math_muladd_f32(vout0x7, vscale0, vbias7);
    const float vscale1 = quantization_params[1].scale;
    vout1x0 = math_muladd_f32(vout1x0, vscale1, vbias0);
    vout1x1 = math_muladd_f32(vout1x1, vscale1, vbias1);
    vout1x2 = math_muladd_f32(vout1x2, vscale1, vbias2);
    vout1x3 = math_muladd_f32(vout1x3, vscale1, vbias3);
    vout1x4 = math_muladd_f32(vout1x4, vscale1, vbias4);
    vout1x5 = math_muladd_f32(vout1x5, vscale1, vbias5);
    vout1x6 = math_muladd_f32(vout1x6, vscale1, vbias6);
    vout1x7 = math_muladd_f32(vout1x7, vscale1, vbias7);

    w = (const float*) w + 8;

    const float voutput_min = params->scalar.min;
    vout0x0 = math_max_f32(vout0x0, voutput_min);
    vout1x0 = math_max_f32(vout1x0, voutput_min);
    vout0x1 = math_max_f32(vout0x1, voutput_min);
    vout1x1 = math_max_f32(vout1x1, voutput_min);
    vout0x2 = math_max_f32(vout0x2, voutput_min);
    vout1x2 = math_max_f32(vout1x2, voutput_min);
    vout0x3 = math_max_f32(vout0x3, voutput_min);
    vout1x3 = math_max_f32(vout1x3, voutput_min);
    vout0x4 = math_max_f32(vout0x4, voutput_min);
    vout1x4 = math_max_f32(vout1x4, voutput_min);
    vout0x5 = math_max_f32(vout0x5, voutput_min);
    vout1x5 = math_max_f32(vout1x5, voutput_min);
    vout0x6 = math_max_f32(vout0x6, voutput_min);
    vout1x6 = math_max_f32(vout1x6, voutput_min);
    vout0x7 = math_max_f32(vout0x7, voutput_min);
    vout1x7 = math_max_f32(vout1x7, voutput_min);

    const float voutput_max = params->scalar.max;
    vout0x0 = math_min_f32(vout0x0, voutput_max);
    vout1x0 = math_min_f32(vout1x0, voutput_max);
    vout0x1 = math_min_f32(vout0x1, voutput_max);
    vout1x1 = math_min_f32(vout1x1, voutput_max);
    vout0x2 = math_min_f32(vout0x2, voutput_max);
    vout1x2 = math_min_f32(vout1x2, voutput_max);
    vout0x3 = math_min_f32(vout0x3, voutput_max);
    vout1x3 = math_min_f32(vout1x3, voutput_max);
    vout0x4 = math_min_f32(vout0x4, voutput_max);
    vout1x4 = math_min_f32(vout1x4, voutput_max);
    vout0x5 = math_min_f32(vout0x5, voutput_max);
    vout1x5 = math_min_f32(vout1x5, voutput_max);
    vout0x6 = math_min_f32(vout0x6, voutput_max);
    vout1x6 = math_min_f32(vout1x6, voutput_max);
    vout0x7 = math_min_f32(vout0x7, voutput_max);
    vout1x7 = math_min_f32(vout1x7, voutput_max);

    if XNN_LIKELY(nc >= 8) {
      c1[0] = vout1x0;
      c1[1] = vout1x1;
      c1[2] = vout1x2;
      c1[3] = vout1x3;
      c1[4] = vout1x4;
      c1[5] = vout1x5;
      c1[6] = vout1x6;
      c1[7] = vout1x7;
      c0[0] = vout0x0;
      c0[1] = vout0x1;
      c0[2] = vout0x2;
      c0[3] = vout0x3;
      c0[4] = vout0x4;
      c0[5] = vout0x5;
      c0[6] = vout0x6;
      c0[7] = vout0x7;

      a0 = (const int8_t*) ((uintptr_t) a0 - kc);
      a1 = (const int8_t*) ((uintptr_t) a1 - kc);

      c0 = (float*) ((uintptr_t) c0 + cn_stride);
      c1 = (float*) ((uintptr_t) c1 + cn_stride);

      nc -= 8;
    } else {
      if (nc & 4) {
        c1[0] = vout1x0;
        c1[1] = vout1x1;
        c1[2] = vout1x2;
        c1[3] = vout1x3;
        vout1x0 = vout1x4;
        vout1x1 = vout1x5;
        vout1x2 = vout1x6;
        c1 += 4;
        c0[0] = vout0x0;
        c0[1] = vout0x1;
        c0[2] = vout0x2;
        c0[3] = vout0x3;
        vout0x0 = vout0x4;
        vout0x1 = vout0x5;
        vout0x2 = vout0x6;
        c0 += 4;
      }
      if (nc & 2) {
        c1[0] = vout1x0;
        c1[1] = vout1x1;
        vout1x0 = vout1x2;
        vout1x1 = vout1x3;
        vout1x2 = vout1x4;
        vout1x3 = vout1x5;
        vout1x4 = vout1x6;
        c1 += 2;
        c0[0] = vout0x0;
        c0[1] = vout0x1;
        vout0x0 = vout0x2;
        vout0x1 = vout0x3;
        vout0x2 = vout0x4;
        vout0x3 = vout0x5;
        vout0x4 = vout0x6;
        c0 += 2;
      }
      if (nc & 1) {
        c1[0] = vout1x0;
        c0[0] = vout0x0;
      }

      nc = 0;
    }
  } while (nc != 0);
}
