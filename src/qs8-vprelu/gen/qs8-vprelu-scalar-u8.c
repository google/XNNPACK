// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/qs8-vprelu/scalar.c.in
//   Generator: tools/xngen
//
// Copyright (C) 2024 Intel Corporation
//  
// Redistribution and use in source and binary forms, with or without modification,
// are permitted provided that the following conditions are met:
//  
// 1. Redistributions of source code must retain the above copyright notice,
//    this list of conditions and the following disclaimer.
// 2. Redistributions in binary form must reproduce the above copyright notice,
//    this list of conditions and the following disclaimer in the documentation
//    and/or other materials provided with the distribution.
// 3. Neither the name of the copyright holder nor the names of its contributors
//    may be used to endorse or promote products derived from this software
//    without specific prior written permission.
//  
// THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
// AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO,
// THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
// ARE DISCLAIMED.  IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS
// BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY,
// OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT
// OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR PROFITS;
// OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY,
// WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE
// OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
// EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
//  
//  
// SPDX-License-Identifier: BSD-3-Clause


#include <assert.h>
#include "src/xnnpack/math.h"
#include "src/xnnpack/vbinary.h"


void xnn_qs8_vprelu_ukernel__scalar_u8(
    size_t batch,
    const int8_t* input_a,
    const int8_t* input_b,
    int8_t* output,
    const union xnn_qs8_vprelu_scalar_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(int8_t) == 0);
  assert(input_a != NULL);
  assert(input_b != NULL);
  assert(output != NULL);

  const int32_t input_zero_point = params->scalar.input_zero_point;
  const int32_t slope_zero_point = params->scalar.slope_zero_point;
  const float vpositive_multiplier = params->scalar.positive_multiplier;
  const float vnegative_multiplier = params->scalar.negative_multiplier;                                
  const float voutput_min_less_zero_point = (int32_t) params->scalar.output_min - (int32_t) params->scalar.output_zero_point;
  const float voutput_max_less_zero_point = (int32_t) params->scalar.output_max - (int32_t) params->scalar.output_zero_point;
  const float vmagic_bias = 12582912.0f;
  const int32_t vmagic_bias_less_output_zero_point = INT32_C(0x4B400000) - (int32_t) params->scalar.output_zero_point;

  for (; batch >= 8 * sizeof(int8_t); batch -= 8 * sizeof(int8_t)) {
    const int32_t va0 = input_a[0] - input_zero_point;
    const int32_t va1 = input_a[1] - input_zero_point;
    const int32_t va2 = input_a[2] - input_zero_point;
    const int32_t va3 = input_a[3] - input_zero_point;
    const int32_t va4 = input_a[4] - input_zero_point;
    const int32_t va5 = input_a[5] - input_zero_point;
    const int32_t va6 = input_a[6] - input_zero_point;
    const int32_t va7 = input_a[7] - input_zero_point;
    input_a += 8;

    const int32_t vb0 = input_b[0] - slope_zero_point;
    const int32_t vb1 = input_b[1] - slope_zero_point;
    const int32_t vb2 = input_b[2] - slope_zero_point;
    const int32_t vb3 = input_b[3] - slope_zero_point;
    const int32_t vb4 = input_b[4] - slope_zero_point;
    const int32_t vb5 = input_b[5] - slope_zero_point;
    const int32_t vb6 = input_b[6] - slope_zero_point;
    const int32_t vb7 = input_b[7] - slope_zero_point;
    input_b += 8;

    int32_t vacc0 = XNN_UNPREDICTABLE(va0 < 0) ? va0 * vb0 : va0;
    int32_t vacc1 = XNN_UNPREDICTABLE(va1 < 0) ? va1 * vb1 : va1;
    int32_t vacc2 = XNN_UNPREDICTABLE(va2 < 0) ? va2 * vb2 : va2;
    int32_t vacc3 = XNN_UNPREDICTABLE(va3 < 0) ? va3 * vb3 : va3;
    int32_t vacc4 = XNN_UNPREDICTABLE(va4 < 0) ? va4 * vb4 : va4;
    int32_t vacc5 = XNN_UNPREDICTABLE(va5 < 0) ? va5 * vb5 : va5;
    int32_t vacc6 = XNN_UNPREDICTABLE(va6 < 0) ? va6 * vb6 : va6;
    int32_t vacc7 = XNN_UNPREDICTABLE(va7 < 0) ? va7 * vb7 : va7;

    float vscale0 = XNN_UNPREDICTABLE(va0 < 0) ? vnegative_multiplier : vpositive_multiplier;
    float vscale1 = XNN_UNPREDICTABLE(va1 < 0) ? vnegative_multiplier : vpositive_multiplier;
    float vscale2 = XNN_UNPREDICTABLE(va2 < 0) ? vnegative_multiplier : vpositive_multiplier;
    float vscale3 = XNN_UNPREDICTABLE(va3 < 0) ? vnegative_multiplier : vpositive_multiplier;
    float vscale4 = XNN_UNPREDICTABLE(va4 < 0) ? vnegative_multiplier : vpositive_multiplier;
    float vscale5 = XNN_UNPREDICTABLE(va5 < 0) ? vnegative_multiplier : vpositive_multiplier;
    float vscale6 = XNN_UNPREDICTABLE(va6 < 0) ? vnegative_multiplier : vpositive_multiplier;
    float vscale7 = XNN_UNPREDICTABLE(va7 < 0) ? vnegative_multiplier : vpositive_multiplier;

    float vfpacc0 = (float) vacc0 * vscale0;
    float vfpacc1 = (float) vacc1 * vscale1;
    float vfpacc2 = (float) vacc2 * vscale2;
    float vfpacc3 = (float) vacc3 * vscale3;
    float vfpacc4 = (float) vacc4 * vscale4;
    float vfpacc5 = (float) vacc5 * vscale5;
    float vfpacc6 = (float) vacc6 * vscale6;
    float vfpacc7 = (float) vacc7 * vscale7;

    vfpacc0 = math_max_f32(vfpacc0, voutput_min_less_zero_point);
    vfpacc1 = math_max_f32(vfpacc1, voutput_min_less_zero_point);
    vfpacc2 = math_max_f32(vfpacc2, voutput_min_less_zero_point);
    vfpacc3 = math_max_f32(vfpacc3, voutput_min_less_zero_point);
    vfpacc4 = math_max_f32(vfpacc4, voutput_min_less_zero_point);
    vfpacc5 = math_max_f32(vfpacc5, voutput_min_less_zero_point);
    vfpacc6 = math_max_f32(vfpacc6, voutput_min_less_zero_point);
    vfpacc7 = math_max_f32(vfpacc7, voutput_min_less_zero_point);

    vfpacc0 = math_min_f32(vfpacc0, voutput_max_less_zero_point);
    vfpacc1 = math_min_f32(vfpacc1, voutput_max_less_zero_point);
    vfpacc2 = math_min_f32(vfpacc2, voutput_max_less_zero_point);
    vfpacc3 = math_min_f32(vfpacc3, voutput_max_less_zero_point);
    vfpacc4 = math_min_f32(vfpacc4, voutput_max_less_zero_point);
    vfpacc5 = math_min_f32(vfpacc5, voutput_max_less_zero_point);
    vfpacc6 = math_min_f32(vfpacc6, voutput_max_less_zero_point);
    vfpacc7 = math_min_f32(vfpacc7, voutput_max_less_zero_point);

    vfpacc0 += vmagic_bias;
    vfpacc1 += vmagic_bias;
    vfpacc2 += vmagic_bias;
    vfpacc3 += vmagic_bias;
    vfpacc4 += vmagic_bias;
    vfpacc5 += vmagic_bias;
    vfpacc6 += vmagic_bias;
    vfpacc7 += vmagic_bias;

    const int32_t vout0 = (int32_t) float_as_uint32(vfpacc0) - vmagic_bias_less_output_zero_point;
    const int32_t vout1 = (int32_t) float_as_uint32(vfpacc1) - vmagic_bias_less_output_zero_point;
    const int32_t vout2 = (int32_t) float_as_uint32(vfpacc2) - vmagic_bias_less_output_zero_point;
    const int32_t vout3 = (int32_t) float_as_uint32(vfpacc3) - vmagic_bias_less_output_zero_point;
    const int32_t vout4 = (int32_t) float_as_uint32(vfpacc4) - vmagic_bias_less_output_zero_point;
    const int32_t vout5 = (int32_t) float_as_uint32(vfpacc5) - vmagic_bias_less_output_zero_point;
    const int32_t vout6 = (int32_t) float_as_uint32(vfpacc6) - vmagic_bias_less_output_zero_point;
    const int32_t vout7 = (int32_t) float_as_uint32(vfpacc7) - vmagic_bias_less_output_zero_point;

    output[0] = (int8_t) vout0;
    output[1] = (int8_t) vout1;
    output[2] = (int8_t) vout2;
    output[3] = (int8_t) vout3;
    output[4] = (int8_t) vout4;
    output[5] = (int8_t) vout5;
    output[6] = (int8_t) vout6;
    output[7] = (int8_t) vout7;
    output += 8;
  }

  if XNN_UNLIKELY(batch != 0) {
    do {
      const int32_t va = (int32_t) *input_a++ - input_zero_point;
      const int32_t vb = (int32_t) *input_b++ - slope_zero_point;
      int32_t vacc = XNN_UNPREDICTABLE(va < 0) ? va * vb : va;
      float vscale = XNN_UNPREDICTABLE(va < 0) ? vnegative_multiplier : vpositive_multiplier;
      float vfpacc = (float) vacc * vscale;
      vfpacc = math_max_f32(vfpacc, voutput_min_less_zero_point);
      vfpacc = math_min_f32(vfpacc, voutput_max_less_zero_point);
      vfpacc += vmagic_bias;
      const int32_t vout = (int32_t) float_as_uint32(vfpacc) - vmagic_bias_less_output_zero_point;
      *output++ = (int8_t) vout;
      batch -= sizeof(int8_t);
    } while (batch != 0);
  }
}
