// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <xnnpack/common.h>
#include <xnnpack/scalar-utils.h>
#include <xnnpack/vadd.h>


void xnn_q8_vadd_minmax_ukernel__scalar(
    size_t n,
    const uint8_t* a,
    const uint8_t* b,
    uint8_t* y,
    const union xnn_q8_add_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(n != 0);

  const int32_t vzero_point_product = params->scalar.zero_point_product;
  const uint32_t va_multiplier = params->scalar.a_multiplier;
  const uint32_t vb_multiplier = params->scalar.b_multiplier;
  const uint32_t vshift = params->scalar.shift;
  const int32_t vremainder_mask = params->scalar.remainder_mask;
  const int32_t vremainder_threshold = params->scalar.remainder_threshold;
  const int32_t vy_zero_point = params->scalar.y_zero_point;
  const int32_t vy_max = params->scalar.y_max;
  const int32_t vy_min = params->scalar.y_min;

  do {
    const int32_t va = (int32_t) (uint32_t) *a++;
    const int32_t vb = (int32_t) (uint32_t) *b++;

    // Multiply by factors.
    const int32_t va_product = va * va_multiplier;
    const int32_t vb_product = vb * vb_multiplier;

    // Accumulate products.
    const int32_t vacc = vzero_point_product + va_product + vb_product;

    // Shift right and round.
    const int32_t vremainder = (vacc & vremainder_mask) - (int32_t) (vacc < 0);
    int32_t vy = asr_s32(vacc, vshift) + (int32_t) (vremainder > vremainder_threshold);

    // Pack, saturate, and add output zero point.
    vy += vy_zero_point;
    vy = vy < vy_min ? vy_min : vy;
    vy = vy > vy_max ? vy_max : vy;

    *y++ = vy;

    n -= sizeof(uint8_t);
  } while (n != 0);
}
