// Auto-generated file. Do not edit!
//   Template: src/f32-qs8-vcvt/scalar-imagic.c.in
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


void xnn_f32_qs8_vcvt_ukernel__scalar_imagic_x2(
    size_t n,
    const float* x,
    int8_t* y,
    const union xnn_f32_qs8_cvt_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(n != 0);
  assert(n % sizeof(float) == 0);
  assert(x != NULL);
  assert(y != NULL);

  const float vscale = params->scalar_imagic.scale;
  const float vmagic_bias = params->scalar_imagic.magic_bias;
  const int32_t vmagic_min = params->scalar_imagic.magic_min;
  const int32_t vmagic_max = params->scalar_imagic.magic_max;
  const int32_t vmagic_bias_less_zero_point = params->scalar_imagic.magic_bias_less_zero_point;

  for (; n >= 2 * sizeof(float); n -= 2 * sizeof(float)) {
    float vx0 = x[0];
    float vx1 = x[1];
    x += 2;

    vx0 *= vscale;
    vx1 *= vscale;

    vx0 += vmagic_bias;
    vx1 += vmagic_bias;

    int32_t vy0 = (int32_t) fp32_to_bits(vx0);
    int32_t vy1 = (int32_t) fp32_to_bits(vx1);

    vy0 = math_max_s32(vy0, vmagic_min);
    vy1 = math_max_s32(vy1, vmagic_min);

    vy0 = math_min_s32(vy0, vmagic_max);
    vy1 = math_min_s32(vy1, vmagic_max);

    vy0 -= vmagic_bias_less_zero_point;
    vy1 -= vmagic_bias_less_zero_point;

    y[0] = (int8_t) vy0;
    y[1] = (int8_t) vy1;
    y += 2;
  }
  if XNN_UNLIKELY(n != 0) {
    float vx = *x;
    vx *= vscale;
    vx += vmagic_bias;

    int32_t vy = (int32_t) fp32_to_bits(vx);
    vy = math_max_s32(vy, vmagic_min);
    vy = math_min_s32(vy, vmagic_max);
    vy -= vmagic_bias_less_zero_point;

    *y = (int8_t) vy;
  }
}
