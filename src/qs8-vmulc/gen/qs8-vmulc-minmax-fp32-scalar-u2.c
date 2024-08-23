// Auto-generated file. Do not edit!
//   Template: src/qs8-vmulc/scalar.c.in
//   Generator: tools/xngen
//
// Copyright 2021 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include "xnnpack/math.h"
#include "xnnpack/vbinary.h"


void xnn_qs8_vmulc_minmax_fp32_ukernel__scalar_u2(
    size_t batch,
    const int8_t* input_a,
    const int8_t* input_b,
    int8_t* output,
    const union xnn_qs8_mul_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(int8_t) == 0);
  assert(input_a != NULL);
  assert(input_b != NULL);
  assert(output != NULL);

  const int32_t va_zero_point = params->scalar.a_zero_point;
  const float vscale = params->scalar.scale;
  const float voutput_min_less_zero_point = (int32_t) params->scalar.output_min - (int32_t) params->scalar.output_zero_point;
  const float voutput_max_less_zero_point = (int32_t) params->scalar.output_max - (int32_t) params->scalar.output_zero_point;
  const float vmagic_bias = 12582912.0f;
  const int32_t vmagic_bias_less_output_zero_point = INT32_C(0x4B400000) - (int32_t) params->scalar.output_zero_point;

  const int32_t vb = (int32_t) *input_b - params->scalar.b_zero_point;
  for (; batch >= 2 * sizeof(int8_t); batch -= 2 * sizeof(int8_t)) {
    const int32_t va0 = input_a[0] - va_zero_point;
    const int32_t va1 = input_a[1] - va_zero_point;
    input_a += 2;

    const int32_t vacc0 = va0 * vb;
    const int32_t vacc1 = va1 * vb;

    float vfpacc0 = (float) vacc0 * vscale;
    float vfpacc1 = (float) vacc1 * vscale;

    vfpacc0 = math_max_f32(vfpacc0, voutput_min_less_zero_point);
    vfpacc1 = math_max_f32(vfpacc1, voutput_min_less_zero_point);

    vfpacc0 = math_min_f32(vfpacc0, voutput_max_less_zero_point);
    vfpacc1 = math_min_f32(vfpacc1, voutput_max_less_zero_point);

    vfpacc0 += vmagic_bias;
    vfpacc1 += vmagic_bias;

    const int32_t vout0 = (int32_t) float_as_uint32(vfpacc0) - vmagic_bias_less_output_zero_point;
    const int32_t vout1 = (int32_t) float_as_uint32(vfpacc1) - vmagic_bias_less_output_zero_point;

    output[0] = (int8_t) vout0;
    output[1] = (int8_t) vout1;
    output += 2;
  }
  if XNN_UNLIKELY(batch != 0) {
    const int32_t va = (int32_t) *input_a - va_zero_point;
    const int32_t vacc = va * vb;

    float vfpacc = (float) vacc * vscale;
    vfpacc = math_max_f32(vfpacc, voutput_min_less_zero_point);
    vfpacc = math_min_f32(vfpacc, voutput_max_less_zero_point);
    vfpacc += vmagic_bias;
    const int32_t vout = (int32_t) float_as_uint32(vfpacc) - vmagic_bias_less_output_zero_point;
    *output = (int8_t) vout;
  }
}
