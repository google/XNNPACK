// Auto-generated file. Do not edit!
//   Template: src/f32-qs8-vcvt/scalar-fmagic.c.in
//   Generator: tools/xngen
//
// Copyright 2021 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <xnnpack/common.h>
#include <xnnpack/math.h>
#include <xnnpack/vcvt.h>

#include <fp16.h>


void xnn_f32_qu8_vcvt_ukernel__wasm_fmagic_x4(
    size_t n,
    const float* x,
    uint8_t* y,
    const union xnn_f32_qu8_cvt_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(n != 0);
  assert(n % sizeof(float) == 0);
  assert(x != NULL);
  assert(y != NULL);

  const float vscale = params->scalar_fmagic.scale;
  const float voutput_min_less_zero_point = params->scalar_fmagic.output_min_less_zero_point;
  const float voutput_max_less_zero_point = params->scalar_fmagic.output_max_less_zero_point;
  const float vmagic_bias = params->scalar_fmagic.magic_bias;
  const int32_t vmagic_bias_less_zero_point = params->scalar_fmagic.magic_bias_less_zero_point;

  for (; n >= 4 * sizeof(float); n -= 4 * sizeof(float)) {
    float vx0 = x[0];
    float vx1 = x[1];
    float vx2 = x[2];
    float vx3 = x[3];
    x += 4;

    vx0 *= vscale;
    vx1 *= vscale;
    vx2 *= vscale;
    vx3 *= vscale;

    vx0 = __builtin_wasm_max_f32(vx0, voutput_min_less_zero_point);
    vx1 = __builtin_wasm_max_f32(vx1, voutput_min_less_zero_point);
    vx2 = __builtin_wasm_max_f32(vx2, voutput_min_less_zero_point);
    vx3 = __builtin_wasm_max_f32(vx3, voutput_min_less_zero_point);

    vx0 = __builtin_wasm_min_f32(vx0, voutput_max_less_zero_point);
    vx1 = __builtin_wasm_min_f32(vx1, voutput_max_less_zero_point);
    vx2 = __builtin_wasm_min_f32(vx2, voutput_max_less_zero_point);
    vx3 = __builtin_wasm_min_f32(vx3, voutput_max_less_zero_point);

    vx0 += vmagic_bias;
    vx1 += vmagic_bias;
    vx2 += vmagic_bias;
    vx3 += vmagic_bias;

    int32_t vy0 = (int32_t) fp32_to_bits(vx0);
    int32_t vy1 = (int32_t) fp32_to_bits(vx1);
    int32_t vy2 = (int32_t) fp32_to_bits(vx2);
    int32_t vy3 = (int32_t) fp32_to_bits(vx3);

    vy0 -= vmagic_bias_less_zero_point;
    vy1 -= vmagic_bias_less_zero_point;
    vy2 -= vmagic_bias_less_zero_point;
    vy3 -= vmagic_bias_less_zero_point;

    y[0] = (uint8_t) vy0;
    y[1] = (uint8_t) vy1;
    y[2] = (uint8_t) vy2;
    y[3] = (uint8_t) vy3;
    y += 4;
  }
  if XNN_UNLIKELY(n != 0) {
    do {
      float vx = *x++;
      vx *= vscale;
      vx = __builtin_wasm_max_f32(vx, voutput_min_less_zero_point);
      vx = __builtin_wasm_min_f32(vx, voutput_max_less_zero_point);
      vx += vmagic_bias;

      int32_t vy = (int32_t) fp32_to_bits(vx);
      vy -= vmagic_bias_less_zero_point;

      *y++ = (uint8_t) vy;

      n -= sizeof(float);
    } while (n != 0);
  }
}
