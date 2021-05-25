// Auto-generated file. Do not edit!
//   Template: src/qs8-vadd/scalar.c.in
//   Generator: tools/xngen
//
// Copyright 2021 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <xnnpack/math.h>
#include <xnnpack/vadd.h>


void xnn_qs8_vadd_minmax_ukernel__scalar_x1(
    size_t n,
    const int8_t* input_x,
    const int8_t* input_y,
    int8_t* output,
    const union xnn_qs8_add_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_DISABLE_TSAN
{
  const int32_t vzero_point_product = params->scalar.zero_point_product;
  const int32_t vx_multiplier = params->scalar.x_multiplier;
  const int32_t vy_multiplier = params->scalar.y_multiplier;
  const uint32_t vshift = params->scalar.shift;
  const int32_t vremainder_mask = params->scalar.remainder_mask;
  const int32_t vremainder_threshold = params->scalar.remainder_threshold;
  const int32_t voutput_zero_point = params->scalar.output_zero_point;
  const int32_t voutput_min = params->scalar.output_min;
  const int32_t voutput_max = params->scalar.output_max;

  do {
    const int32_t vx = *input_x++;
    const int32_t vy = *input_y++;
    int32_t vacc = vzero_point_product + vx * vx_multiplier + vy * vy_multiplier;

    const int32_t vrem = (vacc & vremainder_mask) - (int32_t) (vacc < 0);
    int32_t vout = asr_s32(vacc, vshift) + (int32_t) (vrem > vremainder_threshold);
    vout += voutput_zero_point;
    vout = math_max_s32(vout, voutput_min);
    vout = math_min_s32(vout, voutput_max);
    *output++ = vout;

    n -= sizeof(int8_t);
  } while (n != 0);
}
