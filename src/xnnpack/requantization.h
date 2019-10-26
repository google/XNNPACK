// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#if defined(__cplusplus) && (__cplusplus >= 201103L)
  #include <cstdint>
  #include <cstddef>
  #include <cassert>
  #include <cmath>
#else
  #include <stdint.h>
  #include <stddef.h>
  #include <assert.h>
  #include <math.h>
#endif

#include <fp16.h>

#include <xnnpack/common.h>
#include <xnnpack/params.h>
#include <xnnpack/scalar-utils.h>


static inline uint8_t xnn_q31_requantize(
  int32_t n,
  union xnn_q31_requantization_params params)
{
  const int64_t product = (int64_t) n * (int64_t) params.scalar.multiplier;
  const int32_t q31product = (int32_t) (uint32_t) ((uint64_t) (product + INT64_C(0x40000000)) >> 31);
  const int32_t remainder = (q31product & params.scalar.remainder_mask) - (int32_t) (n < 0);
  n = asr_s32(q31product, params.scalar.shift) + (int32_t) (remainder > params.scalar.remainder_threshold);
  if (n < params.scalar.min_less_zero_point) {
    n = params.scalar.min_less_zero_point;
  }
  if (n > params.scalar.max_less_zero_point) {
    n = params.scalar.max_less_zero_point;
  }

  return (uint8_t) (n + params.scalar.zero_point);
}

static inline uint8_t xnn_avgpool_quantize(
  int32_t n,
  union xnn_q8_avgpool_params params)
{
  const int64_t product = (int64_t) n * (int64_t) params.scalar.multiplier;
  const int64_t adjusted_product = product - (int64_t) (n < 0);

  n = (int32_t) asr_s64(adjusted_product + params.scalar.rounding, params.scalar.right_shift);
  if (n < params.scalar.output_min_less_zero_point) {
    n = params.scalar.output_min_less_zero_point;
  }
  if (n > params.scalar.output_max_less_zero_point) {
    n = params.scalar.output_max_less_zero_point;
  }

  return (uint8_t) (n + params.scalar.output_zero_point);
}

static inline uint8_t xnn_add_quantize(
  uint8_t a, uint8_t b,
  union xnn_q8_add_params params)
{
  // Multiply by factors and accumulate products.
  int32_t acc = params.scalar.zero_point_product +
    (int32_t) ((uint32_t) a * params.scalar.a_multiplier) +
    (int32_t) ((uint32_t) b * params.scalar.b_multiplier);

  // Shift right and round.
  const int32_t rem = (acc & params.scalar.remainder_mask) - (int32_t) (acc < 0);
  acc = asr_s32(acc, params.scalar.shift) + (int32_t) (rem > params.scalar.remainder_threshold);

  // Clamp and add output zero point.
  int32_t y = acc + params.scalar.y_zero_point;
  if (y >= params.scalar.y_max) {
    y = params.scalar.y_max;
  }
  if (y <= params.scalar.y_min) {
    y = params.scalar.y_min;
  }
  return (uint8_t) y;
}
