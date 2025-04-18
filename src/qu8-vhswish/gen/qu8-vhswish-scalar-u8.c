// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/qs8-vhswish/scalar.c.in
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
#include "src/xnnpack/vunary.h"
#include <math.h>

void xnn_qu8_vhswish_ukernel__scalar_u8(
    size_t batch,
    const uint8_t* input,
    uint8_t* output,
    const union xnn_qs8_vhswish_scalar_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(uint8_t) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const float vthree=3.0;
  const float vsix=6.0;
  const float vzero=0.0;
  const float vsixth = 0x1.555556p-3f;

  const int32_t output_min=(int32_t) params->scalar.output_min;
  const int32_t output_max=(int32_t) params->scalar.output_max;
  const int32_t input_zero_point= (float) params->scalar.input_zero_point;
  const float input_scale = (float) params->scalar.input_scale;
  const float output_scale = (float) params->scalar.output_scale;
  const float voutput_min_less_zero_point = (int32_t) params->scalar.output_min - (int32_t) params->scalar.output_zero_point;
  const float voutput_max_less_zero_point = (int32_t) params->scalar.output_max - (int32_t) params->scalar.output_zero_point;
  const float vmagic_bias = 12582912.0f;
  const int32_t vmagic_bias_less_output_zero_point = INT32_C(0x4B400000) - (int32_t) params->scalar.output_zero_point;


  for (; batch >= 8 * sizeof(uint8_t); batch -= 8 * sizeof(uint8_t)) {
    int32_t va0 = ((int32_t) input[0] - input_zero_point); 
    float va_scale0= va0 * input_scale;
    int32_t va1 = ((int32_t) input[1] - input_zero_point); 
    float va_scale1= va1 * input_scale;
    int32_t va2 = ((int32_t) input[2] - input_zero_point); 
    float va_scale2= va2 * input_scale;
    int32_t va3 = ((int32_t) input[3] - input_zero_point); 
    float va_scale3= va3 * input_scale;
    int32_t va4 = ((int32_t) input[4] - input_zero_point); 
    float va_scale4= va4 * input_scale;
    int32_t va5 = ((int32_t) input[5] - input_zero_point); 
    float va_scale5= va5 * input_scale;
    int32_t va6 = ((int32_t) input[6] - input_zero_point); 
    float va_scale6= va6 * input_scale;
    int32_t va7 = ((int32_t) input[7] - input_zero_point); 
    float va_scale7= va7 * input_scale;
    input += 8;

    float vacc0 = fmin(va_scale0 + vthree, vsix);
    float vacc1 = fmin(va_scale1 + vthree, vsix);
    float vacc2 = fmin(va_scale2 + vthree, vsix);
    float vacc3 = fmin(va_scale3 + vthree, vsix);
    float vacc4 = fmin(va_scale4 + vthree, vsix);
    float vacc5 = fmin(va_scale5 + vthree, vsix);
    float vacc6 = fmin(va_scale6 + vthree, vsix);
    float vacc7 = fmin(va_scale7 + vthree, vsix);

    vacc0 = fmax(vacc0, vzero); 
    vacc1 = fmax(vacc1, vzero); 
    vacc2 = fmax(vacc2, vzero); 
    vacc3 = fmax(vacc3, vzero); 
    vacc4 = fmax(vacc4, vzero); 
    vacc5 = fmax(vacc5, vzero); 
    vacc6 = fmax(vacc6, vzero); 
    vacc7 = fmax(vacc7, vzero); 

    va_scale0 = ((va_scale0 * vsixth) * vacc0);
    va_scale1 = ((va_scale1 * vsixth) * vacc1);
    va_scale2 = ((va_scale2 * vsixth) * vacc2);
    va_scale3 = ((va_scale3 * vsixth) * vacc3);
    va_scale4 = ((va_scale4 * vsixth) * vacc4);
    va_scale5 = ((va_scale5 * vsixth) * vacc5);
    va_scale6 = ((va_scale6 * vsixth) * vacc6);
    va_scale7 = ((va_scale7 * vsixth) * vacc7);

    float vfpacc0 = (float) va_scale0 / output_scale;
    float vfpacc1 = (float) va_scale1 / output_scale;
    float vfpacc2 = (float) va_scale2 / output_scale;
    float vfpacc3 = (float) va_scale3 / output_scale;
    float vfpacc4 = (float) va_scale4 / output_scale;
    float vfpacc5 = (float) va_scale5 / output_scale;
    float vfpacc6 = (float) va_scale6 / output_scale;
    float vfpacc7 = (float) va_scale7 / output_scale;

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

    const int32_t vout0 = ((int32_t) float_as_uint32(vfpacc0) - vmagic_bias_less_output_zero_point);
    const int32_t vout1 = ((int32_t) float_as_uint32(vfpacc1) - vmagic_bias_less_output_zero_point);
    const int32_t vout2 = ((int32_t) float_as_uint32(vfpacc2) - vmagic_bias_less_output_zero_point);
    const int32_t vout3 = ((int32_t) float_as_uint32(vfpacc3) - vmagic_bias_less_output_zero_point);
    const int32_t vout4 = ((int32_t) float_as_uint32(vfpacc4) - vmagic_bias_less_output_zero_point);
    const int32_t vout5 = ((int32_t) float_as_uint32(vfpacc5) - vmagic_bias_less_output_zero_point);
    const int32_t vout6 = ((int32_t) float_as_uint32(vfpacc6) - vmagic_bias_less_output_zero_point);
    const int32_t vout7 = ((int32_t) float_as_uint32(vfpacc7) - vmagic_bias_less_output_zero_point);

    output[0] = (uint8_t) vout0;
    output[1] = (uint8_t) vout1;
    output[2] = (uint8_t) vout2;
    output[3] = (uint8_t) vout3;
    output[4] = (uint8_t) vout4;
    output[5] = (uint8_t) vout5;
    output[6] = (uint8_t) vout6;
    output[7] = (uint8_t) vout7;
    output += 8;
  }
  if XNN_UNLIKELY(batch != 0) {
    do {
      int32_t va = ((int32_t) *input++ - input_zero_point);
      float va_scale= va * input_scale;
      float vacc = fmin(va_scale + vthree, vsix);
      vacc = fmax(vacc, vzero);
      va_scale = ((va_scale * vsixth) * vacc);
      float vfpacc = (float) va_scale / output_scale;
      vfpacc = math_max_f32(vfpacc, voutput_min_less_zero_point);
      vfpacc = math_min_f32(vfpacc, voutput_max_less_zero_point);
      vfpacc += vmagic_bias;
      const int32_t vout = ((int32_t) float_as_uint32(vfpacc) - vmagic_bias_less_output_zero_point);
      *output++ = (uint8_t) vout;
      batch -= sizeof(uint8_t);
    } while (batch != 0);
  }
}
