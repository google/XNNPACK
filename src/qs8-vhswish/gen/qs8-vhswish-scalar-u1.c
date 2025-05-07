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

void xnn_qs8_vhswish_ukernel__scalar_u1(
    size_t batch,
    const int8_t* input,
    int8_t* output,
    const union xnn_qs8_vhswish_scalar_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(int8_t) == 0);
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


  do {
    int32_t va = ((int32_t) *input++ - input_zero_point);
    float va_scale = va * input_scale;
    float vacc = fmin(va_scale + vthree, vsix);
    vacc = fmax(vacc, vzero);
    va_scale = ((va_scale * vsixth) * vacc);
    float vfpacc = (float) va_scale / output_scale;
    vfpacc = math_max_f32(vfpacc, voutput_min_less_zero_point);
    vfpacc = math_min_f32(vfpacc, voutput_max_less_zero_point);
    vfpacc +=vmagic_bias;
     const int32_t vout = ((int32_t) float_as_uint32(vfpacc) - vmagic_bias_less_output_zero_point);
    *output++ = (int8_t) vout;
    batch -= sizeof(int8_t);
  } while (batch != 0);
}
