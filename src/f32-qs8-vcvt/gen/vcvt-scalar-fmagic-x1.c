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


void xnn_f32_qs8_vcvt_ukernel__scalar_fmagic_x1(
    size_t n,
    const float* x,
    int8_t* y,
    const union xnn_f32_qs8_cvt_params params[restrict XNN_MIN_ELEMENTS(1)])
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

  do {
    float vx = *x++;
    vx *= vscale;
    vx = math_max_f32(vx, voutput_min_less_zero_point);
    vx = math_min_f32(vx, voutput_max_less_zero_point);
    vx += vmagic_bias;

    int32_t vy = (int32_t) fp32_to_bits(vx);
    vy -= vmagic_bias_less_zero_point;

    *y++ = (int8_t) vy;

    n -= sizeof(float);
  } while (n != 0);
}
