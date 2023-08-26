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

void xnn_qd8_f32_qc8w_gemm_minmax_ukernel_1x4__wasm(
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
  assert(mr <= 1);
  assert(nc != 0);
  assert(kc != 0);

  const int8_t* a0 = a;
  float* c0 = c;


  do {
    const int32_t vksum0 = ((const int32_t*) w)[0];
    const int32_t vksum1 = ((const int32_t*) w)[1];
    const int32_t vksum2 = ((const int32_t*) w)[2];
    const int32_t vksum3 = ((const int32_t*) w)[3];
    const int32_t vinput_zero_point0 = quantization_params[0].zero_point;
    int32_t vacc0x0 = vksum0 * vinput_zero_point0;
    int32_t vacc0x1 = vksum1 * vinput_zero_point0;
    int32_t vacc0x2 = vksum2 * vinput_zero_point0;
    int32_t vacc0x3 = vksum3 * vinput_zero_point0;
    w = (const int32_t*) w + 4;

    size_t k = kc;
    for (; k >= 4 * sizeof(int8_t); k -= 4 * sizeof(int8_t)) {
      const int32_t va00 = (int32_t) a0[0];
      const int32_t va01 = (int32_t) a0[1];
      const int32_t va02 = (int32_t) a0[2];
      const int32_t va03 = (int32_t) a0[3];
      a0 += 4;

      const int32_t vb00 = (int32_t) ((const int8_t*) w)[0];
      const int32_t vb10 = (int32_t) ((const int8_t*) w)[1];
      const int32_t vb20 = (int32_t) ((const int8_t*) w)[2];
      const int32_t vb30 = (int32_t) ((const int8_t*) w)[3];
      const int32_t vb01 = (int32_t) ((const int8_t*) w)[4];
      const int32_t vb11 = (int32_t) ((const int8_t*) w)[5];
      const int32_t vb21 = (int32_t) ((const int8_t*) w)[6];
      const int32_t vb31 = (int32_t) ((const int8_t*) w)[7];
      const int32_t vb02 = (int32_t) ((const int8_t*) w)[8];
      const int32_t vb12 = (int32_t) ((const int8_t*) w)[9];
      const int32_t vb22 = (int32_t) ((const int8_t*) w)[10];
      const int32_t vb32 = (int32_t) ((const int8_t*) w)[11];
      const int32_t vb03 = (int32_t) ((const int8_t*) w)[12];
      const int32_t vb13 = (int32_t) ((const int8_t*) w)[13];
      const int32_t vb23 = (int32_t) ((const int8_t*) w)[14];
      const int32_t vb33 = (int32_t) ((const int8_t*) w)[15];
      w = (const int8_t*) w + 16;

      vacc0x0 += va00 * vb00;
      vacc0x1 += va00 * vb10;
      vacc0x2 += va00 * vb20;
      vacc0x3 += va00 * vb30;
      vacc0x0 += va01 * vb01;
      vacc0x1 += va01 * vb11;
      vacc0x2 += va01 * vb21;
      vacc0x3 += va01 * vb31;
      vacc0x0 += va02 * vb02;
      vacc0x1 += va02 * vb12;
      vacc0x2 += va02 * vb22;
      vacc0x3 += va02 * vb32;
      vacc0x0 += va03 * vb03;
      vacc0x1 += va03 * vb13;
      vacc0x2 += va03 * vb23;
      vacc0x3 += va03 * vb33;
    }
    if XNN_UNLIKELY(k != 0) {
      const int32_t va0 = (int32_t) *a0++;

      const int32_t vb0 = (int32_t) ((const int8_t*) w)[0];
      const int32_t vb1 = (int32_t) ((const int8_t*) w)[1];
      const int32_t vb2 = (int32_t) ((const int8_t*) w)[2];
      const int32_t vb3 = (int32_t) ((const int8_t*) w)[3];
      w = (const int8_t*) w + 4;

      vacc0x0 += va0 * vb0;
      vacc0x1 += va0 * vb1;
      vacc0x2 += va0 * vb2;
      vacc0x3 += va0 * vb3;
    }

    float vout0x0 = (float) vacc0x0;
    float vout0x1 = (float) vacc0x1;
    float vout0x2 = (float) vacc0x2;
    float vout0x3 = (float) vacc0x3;

    const float vinput_scale0 = quantization_params[0].inv_scale;
    vout0x0 *= vinput_scale0;
    vout0x1 *= vinput_scale0;
    vout0x2 *= vinput_scale0;
    vout0x3 *= vinput_scale0;

    const float vfilter_output_scale0 = ((const float*) w)[0];
    vout0x0 *= vfilter_output_scale0;
    const float vfilter_output_scale1 = ((const float*) w)[1];
    vout0x1 *= vfilter_output_scale1;
    const float vfilter_output_scale2 = ((const float*) w)[2];
    vout0x2 *= vfilter_output_scale2;
    const float vfilter_output_scale3 = ((const float*) w)[3];
    vout0x3 *= vfilter_output_scale3;

    const float vbias0 = ((const float*) w)[4];
    vout0x0 += vbias0;
    const float vbias1 = ((const float*) w)[5];
    vout0x1 += vbias1;
    const float vbias2 = ((const float*) w)[6];
    vout0x2 += vbias2;
    const float vbias3 = ((const float*) w)[7];
    vout0x3 += vbias3;

    w = (const float*) w + 8;

    const float voutput_min = params->scalar.min;
    vout0x0 = __builtin_wasm_max_f32(vout0x0, voutput_min);
    vout0x1 = __builtin_wasm_max_f32(vout0x1, voutput_min);
    vout0x2 = __builtin_wasm_max_f32(vout0x2, voutput_min);
    vout0x3 = __builtin_wasm_max_f32(vout0x3, voutput_min);

    const float voutput_max = params->scalar.max;
    vout0x0 = __builtin_wasm_min_f32(vout0x0, voutput_max);
    vout0x1 = __builtin_wasm_min_f32(vout0x1, voutput_max);
    vout0x2 = __builtin_wasm_min_f32(vout0x2, voutput_max);
    vout0x3 = __builtin_wasm_min_f32(vout0x3, voutput_max);

    if XNN_LIKELY(nc >= 4) {
      c0[0] = vout0x0;
      c0[1] = vout0x1;
      c0[2] = vout0x2;
      c0[3] = vout0x3;

      a0 = (const int8_t*) ((uintptr_t) a0 - kc);

      c0 = (float*) ((uintptr_t) c0 + cn_stride);

      nc -= 4;
    } else {
      if (nc & 2) {
        c0[0] = vout0x0;
        c0[1] = vout0x1;
        vout0x0 = vout0x2;
        c0 += 2;
      }
      if (nc & 1) {
        c0[0] = vout0x0;
      }

      nc = 0;
    }
  } while (nc != 0);
}
