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

void xnn_qs8_gemm_minmax_fp32_ukernel_3x4__wasm_fmagic(
    size_t mr,
    size_t nc,
    size_t kc,
    const int8_t* restrict a,
    size_t a_stride,
    const void* restrict w,
    int8_t* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_qs8_conv_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(mr != 0);
  assert(mr <= 3);
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


  do {
    int32_t vacc0x0 = ((const int32_t*) w)[0];
    int32_t vacc0x1 = ((const int32_t*) w)[1];
    int32_t vacc0x2 = ((const int32_t*) w)[2];
    int32_t vacc0x3 = ((const int32_t*) w)[3];
    int32_t vacc1x0 = vacc0x0;
    int32_t vacc1x1 = vacc0x1;
    int32_t vacc1x2 = vacc0x2;
    int32_t vacc1x3 = vacc0x3;
    int32_t vacc2x0 = vacc0x0;
    int32_t vacc2x1 = vacc0x1;
    int32_t vacc2x2 = vacc0x2;
    int32_t vacc2x3 = vacc0x3;
    w = (const int32_t*) w + 4;

    size_t k = kc;
    for (; k >= 4 * sizeof(int8_t); k -= 4 * sizeof(int8_t)) {
      const int32_t va00 = (int32_t) a0[0];
      const int32_t va01 = (int32_t) a0[1];
      const int32_t va02 = (int32_t) a0[2];
      const int32_t va03 = (int32_t) a0[3];
      a0 += 4;
      const int32_t va10 = (int32_t) a1[0];
      const int32_t va11 = (int32_t) a1[1];
      const int32_t va12 = (int32_t) a1[2];
      const int32_t va13 = (int32_t) a1[3];
      a1 += 4;
      const int32_t va20 = (int32_t) a2[0];
      const int32_t va21 = (int32_t) a2[1];
      const int32_t va22 = (int32_t) a2[2];
      const int32_t va23 = (int32_t) a2[3];
      a2 += 4;

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
      vacc1x0 += va10 * vb00;
      vacc1x1 += va10 * vb10;
      vacc1x2 += va10 * vb20;
      vacc1x3 += va10 * vb30;
      vacc2x0 += va20 * vb00;
      vacc2x1 += va20 * vb10;
      vacc2x2 += va20 * vb20;
      vacc2x3 += va20 * vb30;
      vacc0x0 += va01 * vb01;
      vacc0x1 += va01 * vb11;
      vacc0x2 += va01 * vb21;
      vacc0x3 += va01 * vb31;
      vacc1x0 += va11 * vb01;
      vacc1x1 += va11 * vb11;
      vacc1x2 += va11 * vb21;
      vacc1x3 += va11 * vb31;
      vacc2x0 += va21 * vb01;
      vacc2x1 += va21 * vb11;
      vacc2x2 += va21 * vb21;
      vacc2x3 += va21 * vb31;
      vacc0x0 += va02 * vb02;
      vacc0x1 += va02 * vb12;
      vacc0x2 += va02 * vb22;
      vacc0x3 += va02 * vb32;
      vacc1x0 += va12 * vb02;
      vacc1x1 += va12 * vb12;
      vacc1x2 += va12 * vb22;
      vacc1x3 += va12 * vb32;
      vacc2x0 += va22 * vb02;
      vacc2x1 += va22 * vb12;
      vacc2x2 += va22 * vb22;
      vacc2x3 += va22 * vb32;
      vacc0x0 += va03 * vb03;
      vacc0x1 += va03 * vb13;
      vacc0x2 += va03 * vb23;
      vacc0x3 += va03 * vb33;
      vacc1x0 += va13 * vb03;
      vacc1x1 += va13 * vb13;
      vacc1x2 += va13 * vb23;
      vacc1x3 += va13 * vb33;
      vacc2x0 += va23 * vb03;
      vacc2x1 += va23 * vb13;
      vacc2x2 += va23 * vb23;
      vacc2x3 += va23 * vb33;
    }
    if XNN_UNLIKELY(k != 0) {
      const int32_t va0 = (int32_t) *a0++;
      const int32_t va1 = (int32_t) *a1++;
      const int32_t va2 = (int32_t) *a2++;

      const int32_t vb0 = (int32_t) ((const int8_t*) w)[0];
      const int32_t vb1 = (int32_t) ((const int8_t*) w)[1];
      const int32_t vb2 = (int32_t) ((const int8_t*) w)[2];
      const int32_t vb3 = (int32_t) ((const int8_t*) w)[3];
      w = (const int8_t*) w + 4;

      vacc0x0 += va0 * vb0;
      vacc0x1 += va0 * vb1;
      vacc0x2 += va0 * vb2;
      vacc0x3 += va0 * vb3;
      vacc1x0 += va1 * vb0;
      vacc1x1 += va1 * vb1;
      vacc1x2 += va1 * vb2;
      vacc1x3 += va1 * vb3;
      vacc2x0 += va2 * vb0;
      vacc2x1 += va2 * vb1;
      vacc2x2 += va2 * vb2;
      vacc2x3 += va2 * vb3;
    }

    float vfpacc0x0 = (float) vacc0x0;
    float vfpacc0x1 = (float) vacc0x1;
    float vfpacc0x2 = (float) vacc0x2;
    float vfpacc0x3 = (float) vacc0x3;
    float vfpacc1x0 = (float) vacc1x0;
    float vfpacc1x1 = (float) vacc1x1;
    float vfpacc1x2 = (float) vacc1x2;
    float vfpacc1x3 = (float) vacc1x3;
    float vfpacc2x0 = (float) vacc2x0;
    float vfpacc2x1 = (float) vacc2x1;
    float vfpacc2x2 = (float) vacc2x2;
    float vfpacc2x3 = (float) vacc2x3;

    const float vscale = params->fp32_scalar_fmagic.scale;
    vfpacc0x0 *= vscale;
    vfpacc0x1 *= vscale;
    vfpacc0x2 *= vscale;
    vfpacc0x3 *= vscale;
    vfpacc1x0 *= vscale;
    vfpacc1x1 *= vscale;
    vfpacc1x2 *= vscale;
    vfpacc1x3 *= vscale;
    vfpacc2x0 *= vscale;
    vfpacc2x1 *= vscale;
    vfpacc2x2 *= vscale;
    vfpacc2x3 *= vscale;

    const float voutput_min_less_zero_point = params->fp32_scalar_fmagic.output_min_less_zero_point;
    vfpacc0x0 = __builtin_wasm_max_f32(vfpacc0x0, voutput_min_less_zero_point);
    vfpacc0x1 = __builtin_wasm_max_f32(vfpacc0x1, voutput_min_less_zero_point);
    vfpacc0x2 = __builtin_wasm_max_f32(vfpacc0x2, voutput_min_less_zero_point);
    vfpacc0x3 = __builtin_wasm_max_f32(vfpacc0x3, voutput_min_less_zero_point);
    vfpacc1x0 = __builtin_wasm_max_f32(vfpacc1x0, voutput_min_less_zero_point);
    vfpacc1x1 = __builtin_wasm_max_f32(vfpacc1x1, voutput_min_less_zero_point);
    vfpacc1x2 = __builtin_wasm_max_f32(vfpacc1x2, voutput_min_less_zero_point);
    vfpacc1x3 = __builtin_wasm_max_f32(vfpacc1x3, voutput_min_less_zero_point);
    vfpacc2x0 = __builtin_wasm_max_f32(vfpacc2x0, voutput_min_less_zero_point);
    vfpacc2x1 = __builtin_wasm_max_f32(vfpacc2x1, voutput_min_less_zero_point);
    vfpacc2x2 = __builtin_wasm_max_f32(vfpacc2x2, voutput_min_less_zero_point);
    vfpacc2x3 = __builtin_wasm_max_f32(vfpacc2x3, voutput_min_less_zero_point);

    const float voutput_max_less_zero_point = params->fp32_scalar_fmagic.output_max_less_zero_point;
    vfpacc0x0 = __builtin_wasm_min_f32(vfpacc0x0, voutput_max_less_zero_point);
    vfpacc0x1 = __builtin_wasm_min_f32(vfpacc0x1, voutput_max_less_zero_point);
    vfpacc0x2 = __builtin_wasm_min_f32(vfpacc0x2, voutput_max_less_zero_point);
    vfpacc0x3 = __builtin_wasm_min_f32(vfpacc0x3, voutput_max_less_zero_point);
    vfpacc1x0 = __builtin_wasm_min_f32(vfpacc1x0, voutput_max_less_zero_point);
    vfpacc1x1 = __builtin_wasm_min_f32(vfpacc1x1, voutput_max_less_zero_point);
    vfpacc1x2 = __builtin_wasm_min_f32(vfpacc1x2, voutput_max_less_zero_point);
    vfpacc1x3 = __builtin_wasm_min_f32(vfpacc1x3, voutput_max_less_zero_point);
    vfpacc2x0 = __builtin_wasm_min_f32(vfpacc2x0, voutput_max_less_zero_point);
    vfpacc2x1 = __builtin_wasm_min_f32(vfpacc2x1, voutput_max_less_zero_point);
    vfpacc2x2 = __builtin_wasm_min_f32(vfpacc2x2, voutput_max_less_zero_point);
    vfpacc2x3 = __builtin_wasm_min_f32(vfpacc2x3, voutput_max_less_zero_point);

    const float vmagic_bias = params->fp32_scalar_fmagic.magic_bias;
    vfpacc0x0 += vmagic_bias;
    vfpacc0x1 += vmagic_bias;
    vfpacc0x2 += vmagic_bias;
    vfpacc0x3 += vmagic_bias;
    vfpacc1x0 += vmagic_bias;
    vfpacc1x1 += vmagic_bias;
    vfpacc1x2 += vmagic_bias;
    vfpacc1x3 += vmagic_bias;
    vfpacc2x0 += vmagic_bias;
    vfpacc2x1 += vmagic_bias;
    vfpacc2x2 += vmagic_bias;
    vfpacc2x3 += vmagic_bias;

    const int32_t vmagic_bias_less_output_zero_point = params->fp32_scalar_fmagic.magic_bias_less_output_zero_point;
    int32_t vout0x0 = (int32_t) float_as_uint32(vfpacc0x0) - vmagic_bias_less_output_zero_point;
    int32_t vout0x1 = (int32_t) float_as_uint32(vfpacc0x1) - vmagic_bias_less_output_zero_point;
    int32_t vout0x2 = (int32_t) float_as_uint32(vfpacc0x2) - vmagic_bias_less_output_zero_point;
    int32_t vout0x3 = (int32_t) float_as_uint32(vfpacc0x3) - vmagic_bias_less_output_zero_point;
    int32_t vout1x0 = (int32_t) float_as_uint32(vfpacc1x0) - vmagic_bias_less_output_zero_point;
    int32_t vout1x1 = (int32_t) float_as_uint32(vfpacc1x1) - vmagic_bias_less_output_zero_point;
    int32_t vout1x2 = (int32_t) float_as_uint32(vfpacc1x2) - vmagic_bias_less_output_zero_point;
    int32_t vout1x3 = (int32_t) float_as_uint32(vfpacc1x3) - vmagic_bias_less_output_zero_point;
    int32_t vout2x0 = (int32_t) float_as_uint32(vfpacc2x0) - vmagic_bias_less_output_zero_point;
    int32_t vout2x1 = (int32_t) float_as_uint32(vfpacc2x1) - vmagic_bias_less_output_zero_point;
    int32_t vout2x2 = (int32_t) float_as_uint32(vfpacc2x2) - vmagic_bias_less_output_zero_point;
    int32_t vout2x3 = (int32_t) float_as_uint32(vfpacc2x3) - vmagic_bias_less_output_zero_point;

    if XNN_LIKELY(nc >= 4) {
      c0[0] = (int8_t) vout0x0;
      c0[1] = (int8_t) vout0x1;
      c0[2] = (int8_t) vout0x2;
      c0[3] = (int8_t) vout0x3;
      c1[0] = (int8_t) vout1x0;
      c1[1] = (int8_t) vout1x1;
      c1[2] = (int8_t) vout1x2;
      c1[3] = (int8_t) vout1x3;
      c2[0] = (int8_t) vout2x0;
      c2[1] = (int8_t) vout2x1;
      c2[2] = (int8_t) vout2x2;
      c2[3] = (int8_t) vout2x3;

      a0 = (const int8_t*) ((uintptr_t) a0 - kc);
      a1 = (const int8_t*) ((uintptr_t) a1 - kc);
      a2 = (const int8_t*) ((uintptr_t) a2 - kc);

      c0 = (int8_t*) ((uintptr_t) c0 + cn_stride);
      c1 = (int8_t*) ((uintptr_t) c1 + cn_stride);
      c2 = (int8_t*) ((uintptr_t) c2 + cn_stride);

      nc -= 4;
    } else {
      if (nc & 2) {
        c0[0] = (int8_t) vout0x0;
        c0[1] = (int8_t) vout0x1;
        vout0x0 = vout0x2;
        c0 += 2;
        c1[0] = (int8_t) vout1x0;
        c1[1] = (int8_t) vout1x1;
        vout1x0 = vout1x2;
        c1 += 2;
        c2[0] = (int8_t) vout2x0;
        c2[1] = (int8_t) vout2x1;
        vout2x0 = vout2x2;
        c2 += 2;
      }
      if (nc & 1) {
        c0[0] = (int8_t) vout0x0;
        c1[0] = (int8_t) vout1x0;
        c2[0] = (int8_t) vout2x0;
      }

      nc = 0;
    }
  } while (nc != 0);
}
