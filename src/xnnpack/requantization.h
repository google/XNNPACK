// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <assert.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>

#include "xnnpack/common.h"
#include "xnnpack/math.h"
#include "xnnpack/microparams.h"


typedef int8_t (*xnn_qs8_requantize_fn)(
  int32_t input,
  float scale,
  int8_t output_zero_point,
  int8_t output_min,
  int8_t output_max);

typedef uint8_t (*xnn_qu8_requantize_fn)(
  int32_t input,
  float scale,
  uint8_t output_zero_point,
  uint8_t output_min,
  uint8_t output_max);

static inline int8_t xnn_qs8_requantize_fp32(
  int32_t input,
  float scale,
  int8_t zero_point,
  int8_t min,
  int8_t max)
{
  assert(scale >= 1.0f / 4294967296.0f /* 0x1.0p-32f */);
  assert(scale < 256.0f);

  const float min_less_zero_point = (float) ((int32_t) min - (int32_t) zero_point);
  const float max_less_zero_point = (float) ((int32_t) max - (int32_t) zero_point);

  float scaled_input = (float) input * scale;
  scaled_input = math_max_f32(scaled_input, min_less_zero_point);
  scaled_input = math_min_f32(scaled_input, max_less_zero_point);

  const int32_t output = (int32_t) lrintf(scaled_input) + (int32_t) zero_point;
  return (int8_t) output;
}

static inline uint8_t xnn_qu8_requantize_fp32(
  int32_t input,
  float scale,
  uint8_t zero_point,
  uint8_t min,
  uint8_t max)
{
  assert(scale >= 1.0f / 4294967296.0f /* 0x1.0p-32f */);
  assert(scale < 256.0f);

  const float min_less_zero_point = (float) ((int32_t) min - (int32_t) zero_point);
  const float max_less_zero_point = (float) ((int32_t) max - (int32_t) zero_point);

  float scaled_input = (float) input * scale;
  scaled_input = math_max_f32(scaled_input, min_less_zero_point);
  scaled_input = math_min_f32(scaled_input, max_less_zero_point);

  const int32_t output = (int32_t) lrintf(scaled_input) + (int32_t) zero_point;
  return (uint8_t) output;
}

// f32 = 2^exp * multiplier, multiplier is in [1, 2) * 2^23
struct ExpMul {
  int32_t exp;
  // 24 bits
  // multiplier_q24 is in [2^23, 2^24 - 1]
  int32_t multiplier_q24;
};

static inline struct ExpMul parse_f32(float scale) {
  assert(scale >= 0);
  uint32_t scale_bits = float_as_uint32(scale);
  const int32_t multiplier_q24 =
      (scale_bits & UINT32_C(0x007FFFFF)) | UINT32_C(0x00800000);
  int32_t exp = (scale_bits >> 23) - 127;
  struct ExpMul ret;
  ret.exp = exp;
  ret.multiplier_q24 = multiplier_q24;
  return ret;
}

// multiply_2x_high_s16 emulates X86_64 pmulhrsw.
// int16_t range is [-2^15, 2^15 - 1]
// int16_t * int16_t range is strictly included into [-2^30, 2^30 - 1],
// so the result can modeled by signed int31.
// To extract the most significant 16 bits one can shift int31 by 15.
static int16_t multiply_2x_high_s16(int16_t x, int16_t y) {
  int32_t product = (int32_t)x * (int32_t)y;
  int32_t rounding = 1 << 14;
  // This is safe from overflow since x, y are in [-2^15, 2^15 - 1],
  // therefore, x * y is in [2^-30, 2^30).
  int16_t result = (product + rounding) >> 15;
  return result;
}

static inline uint8_t clamp_s16_u8(int16_t result, uint8_t zero_point,
                                   uint8_t min, uint8_t max) {
  int16_t min16 = (int16_t)min;
  int16_t max16 = (int16_t)max;
  int16_t zero_point16 = (int16_t)zero_point;
  return math_max_s16(
      min16, math_min_s16(max16, saturating_add_s16(result, zero_point16)));
}

static inline uint8_t xnn_qu8_requantize_rndnu16(int32_t input, float scale,
                                                 uint8_t zero_point,
                                                 uint8_t min, uint8_t max) {
  assert(scale < 256.0f);
  assert(scale >= 0x1.0p-32f);

  struct ExpMul f32 = parse_f32(scale);

  int exp = f32.exp;
  assert(exp < 8);
  assert(exp >= -32);

  // multiplier_q15 is in the range [2^14, 2^15 - 1]
  int16_t multiplier_q15 =
      math_min_s32((1 << 15) - 1, math_asr_s32_rounding(f32.multiplier_q24, 9));

  // Desired product: P = input * 2^exp * multiplier_q15 * 2^-14
  // We care about the lower 8 bits of P with saturation,
  // i.e. if P >= 2^8 the answer should be 2^8 - 1.
  // To compute these 8 bits we would like to use the upper half
  // of a 16 bit x 16 bit product.
  // This is achived by a preshift of the input, depending on
  // the value of exp.
  int32_t preshifted_input = saturating_rounding_shift_left_s32(input, exp + 1);
  int16_t input16 = saturating_cast_s32_s16(preshifted_input);
  int16_t upper_half16 = multiply_2x_high_s16(input16, multiplier_q15);
  return clamp_s16_u8(upper_half16, zero_point, min, max);
}

static inline int8_t xnn_qs8_requantize_rndnu(
  int32_t input,
  float scale,
  int8_t zero_point,
  int8_t min,
  int8_t max)
{
  assert(scale < 256.0f);
  assert(scale >= 1.0f / 4294967296.0f /* 0x1.0p-32f */);

  const uint32_t scale_bits = float_as_uint32(scale);
  const int32_t multiplier = ((int32_t) scale_bits & INT32_C(0x007FFFFF)) | INT32_C(0x00800000);
  const uint32_t shift = 127 + 23 - (scale_bits >> 23);
  assert(shift >= 16);
  assert(shift < 56);

  const int64_t rounding = INT64_C(1) << (shift - 1);
  const int32_t min_less_zero_point = (int32_t) min - (int32_t) zero_point;
  const int32_t max_less_zero_point = (int32_t) max - (int32_t) zero_point;

  const int64_t abs_prescaled_input = (int64_t) input * (int64_t) multiplier;
  int32_t output = (int32_t) math_asr_s64(abs_prescaled_input + rounding, shift);
  output = math_max_s32(output, min_less_zero_point);
  output = math_min_s32(output, max_less_zero_point);
  return (int8_t) (output + (int32_t) zero_point);
}

static inline uint8_t xnn_qu8_requantize_rndnu(
  int32_t input,
  float scale,
  uint8_t zero_point,
  uint8_t min,
  uint8_t max)
{
  assert(scale < 256.0f);
  assert(scale >= 1.0f / 4294967296.0f /* 0x1.0p-32f */);

  const uint32_t scale_bits = float_as_uint32(scale);
  const int32_t multiplier = ((int32_t) scale_bits & INT32_C(0x007FFFFF)) | INT32_C(0x00800000);
  const uint32_t shift = 127 + 23 - (scale_bits >> 23);
  assert(shift >= 16);
  assert(shift < 56);

  const int64_t rounding = INT64_C(1) << (shift - 1);
  const int32_t min_less_zero_point = (int32_t) min - (int32_t) zero_point;
  const int32_t max_less_zero_point = (int32_t) max - (int32_t) zero_point;

  const int64_t abs_prescaled_input = (int64_t) input * (int64_t) multiplier;
  int32_t output = (int32_t) math_asr_s64(abs_prescaled_input + rounding, shift);
  output = math_max_s32(output, min_less_zero_point);
  output = math_min_s32(output, max_less_zero_point);
  return (uint8_t) (output + (int32_t) zero_point);
}

static inline uint8_t xnn_qu8_quantize_add(
  uint8_t a, uint8_t b,
  struct xnn_qu8_add_minmax_params params)
{
  // Multiply by factors and accumulate products.
  int32_t acc = params.scalar.bias + (int32_t) (uint32_t) a * params.scalar.a_multiplier + (int32_t) (uint32_t) b * params.scalar.b_multiplier;

  // Shift right with rounding away from zero.
  acc = math_asr_s32(acc, params.scalar.shift);

  // Clamp and add output zero point.
  acc += params.scalar.output_zero_point;
  acc = math_max_s32(acc, params.scalar.output_min);
  acc = math_min_s32(acc, params.scalar.output_max);
  return (uint8_t) acc;
}

static inline int8_t xnn_qs8_quantize_add(
  int8_t a, int8_t b,
  struct xnn_qs8_add_minmax_params params)
{
  // Multiply by factors and accumulate products.
  int32_t acc = params.scalar.bias + (int32_t) a * params.scalar.a_multiplier + (int32_t) b * params.scalar.b_multiplier;

  // Shift right with rounding away from zero.
  acc = math_asr_s32(acc, params.scalar.shift);

  // Clamp and add output zero point.
  acc += params.scalar.output_zero_point;
  acc = math_max_s32(acc, params.scalar.output_min);
  acc = math_min_s32(acc, params.scalar.output_max);
  return (int8_t) acc;
}

inline static int8_t xnn_qs8_quantize(float val, float scale, int32_t zero_point)
{
  return (int8_t) lrintf(fminf(fmaxf(val / scale + (float) zero_point, -128.0f), 127.0f));
}

inline static uint8_t xnn_qu8_quantize(float val, float scale, int32_t zero_point)
{
  return (uint8_t) lrintf(fminf(fmaxf(val / scale + (float) zero_point, 0.0f), 255.0f));
}
