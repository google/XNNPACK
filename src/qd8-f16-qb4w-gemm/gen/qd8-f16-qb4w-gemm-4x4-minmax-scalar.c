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
#include <fp16/fp16.h>


void xnn_qd8_f16_qb4w_gemm_minmax_ukernel_4x4__scalar(
    size_t mr,
    size_t nc,
    size_t kc,
    const int8_t* restrict a,
    size_t a_stride,
    const void* restrict w,
    uint16_t* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_f16_qb4w_minmax_params params[restrict XNN_MIN_ELEMENTS(1)],
    const struct xnn_qd8_quantization_params quantization_params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(mr != 0);
  assert(mr <= 4);
  assert(nc != 0);
  assert(kc != 0);
  size_t bl = params->fp16arith.blocksize;
  assert(bl <= round_up_po2(kc, 2));
  assert(bl != 0);
  assert(bl % 32 == 0);

  const int8_t* a0 = a;
  uint16_t* c0 = c;
  const int8_t* a1 = (const int8_t*) ((uintptr_t) a0 + a_stride);
  uint16_t* c1 = (uint16_t*) ((uintptr_t) c0 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 2) {
    a1 = a0;
    c1 = c0;
  }
  const int8_t* a2 = (const int8_t*) ((uintptr_t) a1 + a_stride);
  uint16_t* c2 = (uint16_t*) ((uintptr_t) c1 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 2) {
    a2 = a1;
    c2 = c1;
  }
  const int8_t* a3 = (const int8_t*) ((uintptr_t) a2 + a_stride);
  uint16_t* c3 = (uint16_t*) ((uintptr_t) c2 + cm_stride);
  if XNN_UNPREDICTABLE(mr != 4) {
    a3 = a2;
    c3 = c2;
  }

  kc = round_up_po2(kc, 2);
  do {
    const float vksum0 = ((const float*) w)[0];
    const float vksum1 = ((const float*) w)[1];
    const float vksum2 = ((const float*) w)[2];
    const float vksum3 = ((const float*) w)[3];
    const float vinput_zero_point0 = (const float) quantization_params[0].zero_point;
    float vout0x0 = vksum0 * vinput_zero_point0;
    float vout0x1 = vksum1 * vinput_zero_point0;
    float vout0x2 = vksum2 * vinput_zero_point0;
    float vout0x3 = vksum3 * vinput_zero_point0;
    const float vinput_zero_point1 = (const float) quantization_params[1].zero_point;
    float vout1x0 = vksum0 * vinput_zero_point1;
    float vout1x1 = vksum1 * vinput_zero_point1;
    float vout1x2 = vksum2 * vinput_zero_point1;
    float vout1x3 = vksum3 * vinput_zero_point1;
    const float vinput_zero_point2 = (const float) quantization_params[2].zero_point;
    float vout2x0 = vksum0 * vinput_zero_point2;
    float vout2x1 = vksum1 * vinput_zero_point2;
    float vout2x2 = vksum2 * vinput_zero_point2;
    float vout2x3 = vksum3 * vinput_zero_point2;
    const float vinput_zero_point3 = (const float) quantization_params[3].zero_point;
    float vout3x0 = vksum0 * vinput_zero_point3;
    float vout3x1 = vksum1 * vinput_zero_point3;
    float vout3x2 = vksum2 * vinput_zero_point3;
    float vout3x3 = vksum3 * vinput_zero_point3;
    w = (const float*) w + 4;

    for (size_t kb=0; kb < kc; kb += bl) {
      int32_t vacc0x0 = 0;
      int32_t vacc0x1 = 0;
      int32_t vacc0x2 = 0;
      int32_t vacc0x3 = 0;
      int32_t vacc1x0 = 0;
      int32_t vacc1x1 = 0;
      int32_t vacc1x2 = 0;
      int32_t vacc1x3 = 0;
      int32_t vacc2x0 = 0;
      int32_t vacc2x1 = 0;
      int32_t vacc2x2 = 0;
      int32_t vacc2x3 = 0;
      int32_t vacc3x0 = 0;
      int32_t vacc3x1 = 0;
      int32_t vacc3x2 = 0;
      int32_t vacc3x3 = 0;
      size_t k = bl;
      for (; k >= 2 * sizeof(uint8_t); k -= 2 * sizeof(uint8_t)) {
        const int32_t va0c0 = (int32_t) a0[0];
        const int32_t va0c1 = (int32_t) a0[1];
        a0 += 2;
        const int32_t va1c0 = (int32_t) a1[0];
        const int32_t va1c1 = (int32_t) a1[1];
        a1 += 2;
        const int32_t va2c0 = (int32_t) a2[0];
        const int32_t va2c1 = (int32_t) a2[1];
        a2 += 2;
        const int32_t va3c0 = (int32_t) a3[0];
        const int32_t va3c1 = (int32_t) a3[1];
        a3 += 2;

        const uint8_t vbi0 = ((const uint8_t*) w)[0];
        const uint8_t vbi1 = ((const uint8_t*) w)[1];
        const uint8_t vbi2 = ((const uint8_t*) w)[2];
        const uint8_t vbi3 = ((const uint8_t*) w)[3];
        w = (const uint8_t*) w + 4;
        const int32_t vb0c0 = (int32_t) (int8_t) (vbi0 << 4);
        const int32_t vb0c1 = (int32_t) (int8_t) (vbi0 & 0xF0);
        const int32_t vb1c0 = (int32_t) (int8_t) (vbi1 << 4);
        const int32_t vb1c1 = (int32_t) (int8_t) (vbi1 & 0xF0);
        const int32_t vb2c0 = (int32_t) (int8_t) (vbi2 << 4);
        const int32_t vb2c1 = (int32_t) (int8_t) (vbi2 & 0xF0);
        const int32_t vb3c0 = (int32_t) (int8_t) (vbi3 << 4);
        const int32_t vb3c1 = (int32_t) (int8_t) (vbi3 & 0xF0);

        vacc0x0 += va0c0 * vb0c0;
        vacc0x1 += va0c0 * vb1c0;
        vacc0x2 += va0c0 * vb2c0;
        vacc0x3 += va0c0 * vb3c0;
        vacc1x0 += va1c0 * vb0c0;
        vacc1x1 += va1c0 * vb1c0;
        vacc1x2 += va1c0 * vb2c0;
        vacc1x3 += va1c0 * vb3c0;
        vacc2x0 += va2c0 * vb0c0;
        vacc2x1 += va2c0 * vb1c0;
        vacc2x2 += va2c0 * vb2c0;
        vacc2x3 += va2c0 * vb3c0;
        vacc3x0 += va3c0 * vb0c0;
        vacc3x1 += va3c0 * vb1c0;
        vacc3x2 += va3c0 * vb2c0;
        vacc3x3 += va3c0 * vb3c0;
        vacc0x0 += va0c1 * vb0c1;
        vacc0x1 += va0c1 * vb1c1;
        vacc0x2 += va0c1 * vb2c1;
        vacc0x3 += va0c1 * vb3c1;
        vacc1x0 += va1c1 * vb0c1;
        vacc1x1 += va1c1 * vb1c1;
        vacc1x2 += va1c1 * vb2c1;
        vacc1x3 += va1c1 * vb3c1;
        vacc2x0 += va2c1 * vb0c1;
        vacc2x1 += va2c1 * vb1c1;
        vacc2x2 += va2c1 * vb2c1;
        vacc2x3 += va2c1 * vb3c1;
        vacc3x0 += va3c1 * vb0c1;
        vacc3x1 += va3c1 * vb1c1;
        vacc3x2 += va3c1 * vb2c1;
        vacc3x3 += va3c1 * vb3c1;
    }
    // accumulate in float
      float vf0x0 = vacc0x0;
      const float vfilter_output_scale0 = math_cvt_fp32_bf16(((const uint16_t*) w)[0]);
      float vf0x1 = vacc0x1;
      const float vfilter_output_scale1 = math_cvt_fp32_bf16(((const uint16_t*) w)[1]);
      float vf0x2 = vacc0x2;
      const float vfilter_output_scale2 = math_cvt_fp32_bf16(((const uint16_t*) w)[2]);
      float vf0x3 = vacc0x3;
      const float vfilter_output_scale3 = math_cvt_fp32_bf16(((const uint16_t*) w)[3]);
      float vf1x0 = vacc1x0;
      float vf1x1 = vacc1x1;
      float vf1x2 = vacc1x2;
      float vf1x3 = vacc1x3;
      float vf2x0 = vacc2x0;
      float vf2x1 = vacc2x1;
      float vf2x2 = vacc2x2;
      float vf2x3 = vacc2x3;
      float vf3x0 = vacc3x0;
      float vf3x1 = vacc3x1;
      float vf3x2 = vacc3x2;
      float vf3x3 = vacc3x3;

      vf0x0 *= vfilter_output_scale0;
      vout0x0 += vf0x0;
      vf0x1 *= vfilter_output_scale1;
      vout0x1 += vf0x1;
      vf0x2 *= vfilter_output_scale2;
      vout0x2 += vf0x2;
      vf0x3 *= vfilter_output_scale3;
      vout0x3 += vf0x3;
      vf1x0 *= vfilter_output_scale0;
      vout1x0 += vf1x0;
      vf1x1 *= vfilter_output_scale1;
      vout1x1 += vf1x1;
      vf1x2 *= vfilter_output_scale2;
      vout1x2 += vf1x2;
      vf1x3 *= vfilter_output_scale3;
      vout1x3 += vf1x3;
      vf2x0 *= vfilter_output_scale0;
      vout2x0 += vf2x0;
      vf2x1 *= vfilter_output_scale1;
      vout2x1 += vf2x1;
      vf2x2 *= vfilter_output_scale2;
      vout2x2 += vf2x2;
      vf2x3 *= vfilter_output_scale3;
      vout2x3 += vf2x3;
      vf3x0 *= vfilter_output_scale0;
      vout3x0 += vf3x0;
      vf3x1 *= vfilter_output_scale1;
      vout3x1 += vf3x1;
      vf3x2 *= vfilter_output_scale2;
      vout3x2 += vf3x2;
      vf3x3 *= vfilter_output_scale3;
      vout3x3 += vf3x3;
      w = (const uint16_t*) w + 4;
    }


    const float vinput_scale0 = quantization_params[0].inv_scale;
    vout0x0 *= vinput_scale0;
    vout0x1 *= vinput_scale0;
    vout0x2 *= vinput_scale0;
    vout0x3 *= vinput_scale0;
    const float vinput_scale1 = quantization_params[1].inv_scale;
    vout1x0 *= vinput_scale1;
    vout1x1 *= vinput_scale1;
    vout1x2 *= vinput_scale1;
    vout1x3 *= vinput_scale1;
    const float vinput_scale2 = quantization_params[2].inv_scale;
    vout2x0 *= vinput_scale2;
    vout2x1 *= vinput_scale2;
    vout2x2 *= vinput_scale2;
    vout2x3 *= vinput_scale2;
    const float vinput_scale3 = quantization_params[3].inv_scale;
    vout3x0 *= vinput_scale3;
    vout3x1 *= vinput_scale3;
    vout3x2 *= vinput_scale3;
    vout3x3 *= vinput_scale3;


    const float vbias0 = ((const float*) w)[0];
    vout0x0 += vbias0;
    vout1x0 += vbias0;
    vout2x0 += vbias0;
    vout3x0 += vbias0;
    const float vbias1 = ((const float*) w)[1];
    vout0x1 += vbias1;
    vout1x1 += vbias1;
    vout2x1 += vbias1;
    vout3x1 += vbias1;
    const float vbias2 = ((const float*) w)[2];
    vout0x2 += vbias2;
    vout1x2 += vbias2;
    vout2x2 += vbias2;
    vout3x2 += vbias2;
    const float vbias3 = ((const float*) w)[3];
    vout0x3 += vbias3;
    vout1x3 += vbias3;
    vout2x3 += vbias3;
    vout3x3 += vbias3;

    w = (const float*) w + 4;

    const float voutput_min = fp16_ieee_to_fp32_value(params->fp16arith.min);
    vout0x0 = math_max_f32(vout0x0, voutput_min);
    vout1x0 = math_max_f32(vout1x0, voutput_min);
    vout2x0 = math_max_f32(vout2x0, voutput_min);
    vout3x0 = math_max_f32(vout3x0, voutput_min);
    vout0x1 = math_max_f32(vout0x1, voutput_min);
    vout1x1 = math_max_f32(vout1x1, voutput_min);
    vout2x1 = math_max_f32(vout2x1, voutput_min);
    vout3x1 = math_max_f32(vout3x1, voutput_min);
    vout0x2 = math_max_f32(vout0x2, voutput_min);
    vout1x2 = math_max_f32(vout1x2, voutput_min);
    vout2x2 = math_max_f32(vout2x2, voutput_min);
    vout3x2 = math_max_f32(vout3x2, voutput_min);
    vout0x3 = math_max_f32(vout0x3, voutput_min);
    vout1x3 = math_max_f32(vout1x3, voutput_min);
    vout2x3 = math_max_f32(vout2x3, voutput_min);
    vout3x3 = math_max_f32(vout3x3, voutput_min);

    const float voutput_max = fp16_ieee_to_fp32_value(params->fp16arith.max);
    vout0x0 = math_min_f32(vout0x0, voutput_max);
    vout1x0 = math_min_f32(vout1x0, voutput_max);
    vout2x0 = math_min_f32(vout2x0, voutput_max);
    vout3x0 = math_min_f32(vout3x0, voutput_max);
    vout0x1 = math_min_f32(vout0x1, voutput_max);
    vout1x1 = math_min_f32(vout1x1, voutput_max);
    vout2x1 = math_min_f32(vout2x1, voutput_max);
    vout3x1 = math_min_f32(vout3x1, voutput_max);
    vout0x2 = math_min_f32(vout0x2, voutput_max);
    vout1x2 = math_min_f32(vout1x2, voutput_max);
    vout2x2 = math_min_f32(vout2x2, voutput_max);
    vout3x2 = math_min_f32(vout3x2, voutput_max);
    vout0x3 = math_min_f32(vout0x3, voutput_max);
    vout1x3 = math_min_f32(vout1x3, voutput_max);
    vout2x3 = math_min_f32(vout2x3, voutput_max);
    vout3x3 = math_min_f32(vout3x3, voutput_max);

    if XNN_LIKELY(nc >= 4) {
      c0[0] = fp16_ieee_from_fp32_value(vout0x0);
      c0[1] = fp16_ieee_from_fp32_value(vout0x1);
      c0[2] = fp16_ieee_from_fp32_value(vout0x2);
      c0[3] = fp16_ieee_from_fp32_value(vout0x3);
      c1[0] = fp16_ieee_from_fp32_value(vout1x0);
      c1[1] = fp16_ieee_from_fp32_value(vout1x1);
      c1[2] = fp16_ieee_from_fp32_value(vout1x2);
      c1[3] = fp16_ieee_from_fp32_value(vout1x3);
      c2[0] = fp16_ieee_from_fp32_value(vout2x0);
      c2[1] = fp16_ieee_from_fp32_value(vout2x1);
      c2[2] = fp16_ieee_from_fp32_value(vout2x2);
      c2[3] = fp16_ieee_from_fp32_value(vout2x3);
      c3[0] = fp16_ieee_from_fp32_value(vout3x0);
      c3[1] = fp16_ieee_from_fp32_value(vout3x1);
      c3[2] = fp16_ieee_from_fp32_value(vout3x2);
      c3[3] = fp16_ieee_from_fp32_value(vout3x3);

      a0 = (const int8_t*) ((uintptr_t) a0 - kc);
      a1 = (const int8_t*) ((uintptr_t) a1 - kc);
      a2 = (const int8_t*) ((uintptr_t) a2 - kc);
      a3 = (const int8_t*) ((uintptr_t) a3 - kc);

      c0 = (uint16_t*) ((uintptr_t) c0 + cn_stride);
      c1 = (uint16_t*) ((uintptr_t) c1 + cn_stride);
      c2 = (uint16_t*) ((uintptr_t) c2 + cn_stride);
      c3 = (uint16_t*) ((uintptr_t) c3 + cn_stride);

      nc -= 4;
    } else {
      if (nc & 2) {
        c0[0] = fp16_ieee_from_fp32_value(vout0x0);
        c0[1] = fp16_ieee_from_fp32_value(vout0x1);
        vout0x0 = vout0x2;
        c0 += 2;
        c1[0] = fp16_ieee_from_fp32_value(vout1x0);
        c1[1] = fp16_ieee_from_fp32_value(vout1x1);
        vout1x0 = vout1x2;
        c1 += 2;
        c2[0] = fp16_ieee_from_fp32_value(vout2x0);
        c2[1] = fp16_ieee_from_fp32_value(vout2x1);
        vout2x0 = vout2x2;
        c2 += 2;
        c3[0] = fp16_ieee_from_fp32_value(vout3x0);
        c3[1] = fp16_ieee_from_fp32_value(vout3x1);
        vout3x0 = vout3x2;
        c3 += 2;
      }
      if (nc & 1) {
        c0[0] = fp16_ieee_from_fp32_value(vout0x0);
        c1[0] = fp16_ieee_from_fp32_value(vout1x0);
        c2[0] = fp16_ieee_from_fp32_value(vout2x0);
        c3[0] = fp16_ieee_from_fp32_value(vout3x0);
      }

      nc = 0;
    }
  } while (nc != 0);
}
