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


void xnn_qd8_f32_qc4w_gemm_minmax_ukernel_2x8__scalar(
    size_t mr,
    size_t nc,
    size_t kc,
    const int8_t* restrict a,
    size_t a_stride,
    const void* restrict w,
    float* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_f32_qc4w_minmax_params params[restrict XNN_MIN_ELEMENTS(1)],
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

  kc = round_up_po2(kc, 2);
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
    for (; k >= 2 * sizeof(uint8_t); k -= 2 * sizeof(uint8_t)) {
      const int32_t va0c0 = (int32_t) a0[0];
      const int32_t va0c1 = (int32_t) a0[1];
      a0 += 2;
      const int32_t va1c0 = (int32_t) a1[0];
      const int32_t va1c1 = (int32_t) a1[1];
      a1 += 2;

      const uint8_t vbi0 = ((const uint8_t*) w)[0];
      const uint8_t vbi1 = ((const uint8_t*) w)[1];
      const uint8_t vbi2 = ((const uint8_t*) w)[2];
      const uint8_t vbi3 = ((const uint8_t*) w)[3];
      const uint8_t vbi4 = ((const uint8_t*) w)[4];
      const uint8_t vbi5 = ((const uint8_t*) w)[5];
      const uint8_t vbi6 = ((const uint8_t*) w)[6];
      const uint8_t vbi7 = ((const uint8_t*) w)[7];
      w = (const uint8_t*) w + 8;
      const int32_t vb0c0 = (int32_t) (int8_t) (vbi0 << 4);
      const int32_t vb0c1 = (int32_t) (int8_t) (vbi0 & 0xF0);
      const int32_t vb1c0 = (int32_t) (int8_t) (vbi1 << 4);
      const int32_t vb1c1 = (int32_t) (int8_t) (vbi1 & 0xF0);
      const int32_t vb2c0 = (int32_t) (int8_t) (vbi2 << 4);
      const int32_t vb2c1 = (int32_t) (int8_t) (vbi2 & 0xF0);
      const int32_t vb3c0 = (int32_t) (int8_t) (vbi3 << 4);
      const int32_t vb3c1 = (int32_t) (int8_t) (vbi3 & 0xF0);
      const int32_t vb4c0 = (int32_t) (int8_t) (vbi4 << 4);
      const int32_t vb4c1 = (int32_t) (int8_t) (vbi4 & 0xF0);
      const int32_t vb5c0 = (int32_t) (int8_t) (vbi5 << 4);
      const int32_t vb5c1 = (int32_t) (int8_t) (vbi5 & 0xF0);
      const int32_t vb6c0 = (int32_t) (int8_t) (vbi6 << 4);
      const int32_t vb6c1 = (int32_t) (int8_t) (vbi6 & 0xF0);
      const int32_t vb7c0 = (int32_t) (int8_t) (vbi7 << 4);
      const int32_t vb7c1 = (int32_t) (int8_t) (vbi7 & 0xF0);

      vacc0x0 += va0c0 * vb0c0;
      vacc0x1 += va0c0 * vb1c0;
      vacc0x2 += va0c0 * vb2c0;
      vacc0x3 += va0c0 * vb3c0;
      vacc0x4 += va0c0 * vb4c0;
      vacc0x5 += va0c0 * vb5c0;
      vacc0x6 += va0c0 * vb6c0;
      vacc0x7 += va0c0 * vb7c0;
      vacc1x0 += va1c0 * vb0c0;
      vacc1x1 += va1c0 * vb1c0;
      vacc1x2 += va1c0 * vb2c0;
      vacc1x3 += va1c0 * vb3c0;
      vacc1x4 += va1c0 * vb4c0;
      vacc1x5 += va1c0 * vb5c0;
      vacc1x6 += va1c0 * vb6c0;
      vacc1x7 += va1c0 * vb7c0;
      vacc0x0 += va0c1 * vb0c1;
      vacc0x1 += va0c1 * vb1c1;
      vacc0x2 += va0c1 * vb2c1;
      vacc0x3 += va0c1 * vb3c1;
      vacc0x4 += va0c1 * vb4c1;
      vacc0x5 += va0c1 * vb5c1;
      vacc0x6 += va0c1 * vb6c1;
      vacc0x7 += va0c1 * vb7c1;
      vacc1x0 += va1c1 * vb0c1;
      vacc1x1 += va1c1 * vb1c1;
      vacc1x2 += va1c1 * vb2c1;
      vacc1x3 += va1c1 * vb3c1;
      vacc1x4 += va1c1 * vb4c1;
      vacc1x5 += va1c1 * vb5c1;
      vacc1x6 += va1c1 * vb6c1;
      vacc1x7 += va1c1 * vb7c1;
    }

    float vout0x0 = (float) math_asr_s32(vacc0x0, 4);
    float vout0x1 = (float) math_asr_s32(vacc0x1, 4);
    float vout0x2 = (float) math_asr_s32(vacc0x2, 4);
    float vout0x3 = (float) math_asr_s32(vacc0x3, 4);
    float vout0x4 = (float) math_asr_s32(vacc0x4, 4);
    float vout0x5 = (float) math_asr_s32(vacc0x5, 4);
    float vout0x6 = (float) math_asr_s32(vacc0x6, 4);
    float vout0x7 = (float) math_asr_s32(vacc0x7, 4);
    float vout1x0 = (float) math_asr_s32(vacc1x0, 4);
    float vout1x1 = (float) math_asr_s32(vacc1x1, 4);
    float vout1x2 = (float) math_asr_s32(vacc1x2, 4);
    float vout1x3 = (float) math_asr_s32(vacc1x3, 4);
    float vout1x4 = (float) math_asr_s32(vacc1x4, 4);
    float vout1x5 = (float) math_asr_s32(vacc1x5, 4);
    float vout1x6 = (float) math_asr_s32(vacc1x6, 4);
    float vout1x7 = (float) math_asr_s32(vacc1x7, 4);

    const float vinput_scale0 = quantization_params[0].inv_scale;
    vout0x0 *= vinput_scale0;
    vout0x1 *= vinput_scale0;
    vout0x2 *= vinput_scale0;
    vout0x3 *= vinput_scale0;
    vout0x4 *= vinput_scale0;
    vout0x5 *= vinput_scale0;
    vout0x6 *= vinput_scale0;
    vout0x7 *= vinput_scale0;
    const float vinput_scale1 = quantization_params[1].inv_scale;
    vout1x0 *= vinput_scale1;
    vout1x1 *= vinput_scale1;
    vout1x2 *= vinput_scale1;
    vout1x3 *= vinput_scale1;
    vout1x4 *= vinput_scale1;
    vout1x5 *= vinput_scale1;
    vout1x6 *= vinput_scale1;
    vout1x7 *= vinput_scale1;

    const float vfilter_output_scale0 = ((const float*) w)[0];
    vout0x0 *= vfilter_output_scale0;
    vout1x0 *= vfilter_output_scale0;
    const float vfilter_output_scale1 = ((const float*) w)[1];
    vout0x1 *= vfilter_output_scale1;
    vout1x1 *= vfilter_output_scale1;
    const float vfilter_output_scale2 = ((const float*) w)[2];
    vout0x2 *= vfilter_output_scale2;
    vout1x2 *= vfilter_output_scale2;
    const float vfilter_output_scale3 = ((const float*) w)[3];
    vout0x3 *= vfilter_output_scale3;
    vout1x3 *= vfilter_output_scale3;
    const float vfilter_output_scale4 = ((const float*) w)[4];
    vout0x4 *= vfilter_output_scale4;
    vout1x4 *= vfilter_output_scale4;
    const float vfilter_output_scale5 = ((const float*) w)[5];
    vout0x5 *= vfilter_output_scale5;
    vout1x5 *= vfilter_output_scale5;
    const float vfilter_output_scale6 = ((const float*) w)[6];
    vout0x6 *= vfilter_output_scale6;
    vout1x6 *= vfilter_output_scale6;
    const float vfilter_output_scale7 = ((const float*) w)[7];
    vout0x7 *= vfilter_output_scale7;
    vout1x7 *= vfilter_output_scale7;

    const float vbias0 = ((const float*) w)[8];
    vout0x0 += vbias0;
    vout1x0 += vbias0;
    const float vbias1 = ((const float*) w)[9];
    vout0x1 += vbias1;
    vout1x1 += vbias1;
    const float vbias2 = ((const float*) w)[10];
    vout0x2 += vbias2;
    vout1x2 += vbias2;
    const float vbias3 = ((const float*) w)[11];
    vout0x3 += vbias3;
    vout1x3 += vbias3;
    const float vbias4 = ((const float*) w)[12];
    vout0x4 += vbias4;
    vout1x4 += vbias4;
    const float vbias5 = ((const float*) w)[13];
    vout0x5 += vbias5;
    vout1x5 += vbias5;
    const float vbias6 = ((const float*) w)[14];
    vout0x6 += vbias6;
    vout1x6 += vbias6;
    const float vbias7 = ((const float*) w)[15];
    vout0x7 += vbias7;
    vout1x7 += vbias7;

    w = (const float*) w + 16;

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
      c0[0] = vout0x0;
      c0[1] = vout0x1;
      c0[2] = vout0x2;
      c0[3] = vout0x3;
      c0[4] = vout0x4;
      c0[5] = vout0x5;
      c0[6] = vout0x6;
      c0[7] = vout0x7;
      c1[0] = vout1x0;
      c1[1] = vout1x1;
      c1[2] = vout1x2;
      c1[3] = vout1x3;
      c1[4] = vout1x4;
      c1[5] = vout1x5;
      c1[6] = vout1x6;
      c1[7] = vout1x7;

      a0 = (const int8_t*) ((uintptr_t) a0 - kc);
      a1 = (const int8_t*) ((uintptr_t) a1 - kc);

      c0 = (float*) ((uintptr_t) c0 + cn_stride);
      c1 = (float*) ((uintptr_t) c1 + cn_stride);

      nc -= 8;
    } else {
      if (nc & 4) {
        c0[0] = vout0x0;
        c0[1] = vout0x1;
        c0[2] = vout0x2;
        c0[3] = vout0x3;
        vout0x0 = vout0x4;
        vout0x1 = vout0x5;
        vout0x2 = vout0x6;
        c0 += 4;
        c1[0] = vout1x0;
        c1[1] = vout1x1;
        c1[2] = vout1x2;
        c1[3] = vout1x3;
        vout1x0 = vout1x4;
        vout1x1 = vout1x5;
        vout1x2 = vout1x6;
        c1 += 4;
      }
      if (nc & 2) {
        c0[0] = vout0x0;
        c0[1] = vout0x1;
        vout0x0 = vout0x2;
        vout0x1 = vout0x3;
        vout0x2 = vout0x4;
        vout0x3 = vout0x5;
        vout0x4 = vout0x6;
        c0 += 2;
        c1[0] = vout1x0;
        c1[1] = vout1x1;
        vout1x0 = vout1x2;
        vout1x1 = vout1x3;
        vout1x2 = vout1x4;
        vout1x3 = vout1x5;
        vout1x4 = vout1x6;
        c1 += 2;
      }
      if (nc & 1) {
        c0[0] = vout0x0;
        c1[0] = vout1x0;
      }

      nc = 0;
    }
  } while (nc != 0);
}
