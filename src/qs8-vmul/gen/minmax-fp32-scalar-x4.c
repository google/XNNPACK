// Auto-generated file. Do not edit!
//   Template: src/qs8-vmul/scalar.c.in
//   Generator: tools/xngen
//
// Copyright 2021 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <fp16.h>

#include <xnnpack/math.h>
#include <xnnpack/vadd.h>


void xnn_qs8_vmul_minmax_fp32_ukernel__scalar_x4(
    size_t n,
    const int8_t* input_a,
    const int8_t* input_b,
    int8_t* output,
    const union xnn_qs8_mul_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_DISABLE_TSAN
{
  const int32_t va_zero_point = params->fp32_scalar.a_zero_point;
  const int32_t vb_zero_point = params->fp32_scalar.b_zero_point;
  const float vscale = params->fp32_scalar.scale;
  const float voutput_min_less_zero_point = params->fp32_scalar.output_min_less_zero_point;
  const float voutput_max_less_zero_point = params->fp32_scalar.output_max_less_zero_point;
  const float vmagic_bias = params->fp32_scalar.magic_bias;
  const int32_t vmagic_bias_less_output_zero_point = params->fp32_scalar.magic_bias_less_output_zero_point;

  for (; n >= 4 * sizeof(int8_t); n -= 4 * sizeof(int8_t)) {
    const int32_t va0 = input_a[0] - va_zero_point;
    const int32_t va1 = input_a[1] - va_zero_point;
    const int32_t va2 = input_a[2] - va_zero_point;
    const int32_t va3 = input_a[3] - va_zero_point;
    input_a += 4;

    const int32_t vb0 = input_b[0] - vb_zero_point;
    const int32_t vb1 = input_b[1] - vb_zero_point;
    const int32_t vb2 = input_b[2] - vb_zero_point;
    const int32_t vb3 = input_b[3] - vb_zero_point;
    input_b += 4;

    const int32_t vacc0 = va0 * vb0;
    const int32_t vacc1 = va1 * vb1;
    const int32_t vacc2 = va2 * vb2;
    const int32_t vacc3 = va3 * vb3;

    float vfpacc0 = (float) vacc0 * vscale;
    float vfpacc1 = (float) vacc1 * vscale;
    float vfpacc2 = (float) vacc2 * vscale;
    float vfpacc3 = (float) vacc3 * vscale;

    vfpacc0 = math_max_f32(vfpacc0, voutput_min_less_zero_point);
    vfpacc1 = math_max_f32(vfpacc1, voutput_min_less_zero_point);
    vfpacc2 = math_max_f32(vfpacc2, voutput_min_less_zero_point);
    vfpacc3 = math_max_f32(vfpacc3, voutput_min_less_zero_point);

    vfpacc0 = math_min_f32(vfpacc0, voutput_max_less_zero_point);
    vfpacc1 = math_min_f32(vfpacc1, voutput_max_less_zero_point);
    vfpacc2 = math_min_f32(vfpacc2, voutput_max_less_zero_point);
    vfpacc3 = math_min_f32(vfpacc3, voutput_max_less_zero_point);

    vfpacc0 += vmagic_bias;
    vfpacc1 += vmagic_bias;
    vfpacc2 += vmagic_bias;
    vfpacc3 += vmagic_bias;

    const int32_t vout0 = (int32_t) fp32_to_bits(vfpacc0) - vmagic_bias_less_output_zero_point;
    const int32_t vout1 = (int32_t) fp32_to_bits(vfpacc1) - vmagic_bias_less_output_zero_point;
    const int32_t vout2 = (int32_t) fp32_to_bits(vfpacc2) - vmagic_bias_less_output_zero_point;
    const int32_t vout3 = (int32_t) fp32_to_bits(vfpacc3) - vmagic_bias_less_output_zero_point;

    output[0] = (int8_t) vout0;
    output[1] = (int8_t) vout1;
    output[2] = (int8_t) vout2;
    output[3] = (int8_t) vout3;
    output += 4;
  }
  if XNN_UNLIKELY(n != 0) {
    do {
      const int32_t va = (int32_t) *input_a++ - va_zero_point;
      const int32_t vb = (int32_t) *input_b++ - vb_zero_point;
      const int32_t vacc = va * vb;

      float vfpacc = (float) vacc * vscale;
      vfpacc = math_max_f32(vfpacc, voutput_min_less_zero_point);
      vfpacc = math_min_f32(vfpacc, voutput_max_less_zero_point);
      vfpacc += vmagic_bias;
      const int32_t vout = (int32_t) fp32_to_bits(vfpacc) - vmagic_bias_less_output_zero_point;
      *output++ = (int8_t) vout;

      n -= sizeof(int8_t);
    } while (n != 0);
  }
}
