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

static inline int8_t xnn_qs8_requantize_rndna(
  int32_t input,
  float scale,
  int8_t zero_point,
  int8_t min,
  int8_t max)
{
  assert(scale >= 1.0f / 4294967296.0f /* 0x1.0p-32f */);
  assert(scale < 256.0f);

  const uint32_t scale_bits = float_as_uint32(scale);
  const uint32_t multiplier = (scale_bits & UINT32_C(0x007FFFFF)) | UINT32_C(0x00800000);
  const uint32_t shift = 127 + 23 - (scale_bits >> 23);
  assert(shift >= 16);
  assert(shift < 56);

  const uint64_t rounding = UINT64_C(1) << (shift - 1);
  const int32_t min_less_zero_point = (int32_t) min - (int32_t) zero_point;
  const int32_t max_less_zero_point = (int32_t) max - (int32_t) zero_point;

  uint32_t abs_input = (uint32_t) input;
  if (input < 0) {
    abs_input = -abs_input;
  }

  const uint64_t abs_prescaled_input = (uint64_t) abs_input * (uint64_t) multiplier;
  const uint32_t abs_scaled_input = (uint32_t) ((abs_prescaled_input + rounding) >> shift);

  int32_t output = (int32_t) abs_scaled_input;
  if (input < 0) {
    output = -output;
  }

  output = math_max_s32(output, min_less_zero_point);
  output = math_min_s32(output, max_less_zero_point);
  return (int8_t) (output + (int32_t) zero_point);
}

static inline uint8_t xnn_qu8_requantize_rndna(
  int32_t input,
  float scale,
  uint8_t zero_point,
  uint8_t min,
  uint8_t max)
{
  assert(scale >= 1.0f / 4294967296.0f /* 0x1.0p-32f */);
  assert(scale < 256.0f);

  const uint32_t scale_bits = float_as_uint32(scale);
  const uint32_t multiplier = (scale_bits & UINT32_C(0x007FFFFF)) | UINT32_C(0x00800000);
  const uint32_t shift = 127 + 23 - (scale_bits >> 23);
  assert(shift >= 16);
  assert(shift < 56);

  const uint64_t rounding = UINT64_C(1) << (shift - 1);
  const int32_t min_less_zero_point = (int32_t) min - (int32_t) zero_point;
  const int32_t max_less_zero_point = (int32_t) max - (int32_t) zero_point;

  uint32_t abs_input = (uint32_t) input;
  if (input < 0) {
    abs_input = -abs_input;
  }

  const uint64_t abs_prescaled_input = (uint64_t) abs_input * (uint64_t) multiplier;
  const uint32_t abs_scaled_input = (uint32_t) ((abs_prescaled_input + rounding) >> shift);

  int32_t output = (int32_t) abs_scaled_input;
  if (input < 0) {
    output = -output;
  }

  output = math_max_s32(output, min_less_zero_point);
  output = math_min_s32(output, max_less_zero_point);
  return (uint8_t) (output + (int32_t) zero_point);
}

// f32 = 2^exp * multiplier, multiplier is in [1, 2)
struct F32 {
  int32_t exp;
  // 24 bits
  uint32_t multiplier;
};

static inline struct F32 parse_f32(float scale) {
  assert(scale >= 0);
  uint32_t scale_bits = float_as_uint32(scale);
  const uint32_t multiplier =
      (scale_bits & UINT32_C(0x007FFFFF)) | UINT32_C(0x00800000);
  int32_t exp = (scale_bits >> 23) - 127;
  struct F32 ret = {.exp = exp, .multiplier = multiplier};
  return ret;
}

static inline int16_t saturating_cast(int32_t input) {
  if (input > INT16_MAX) return INT16_MAX;
  if (input < INT16_MIN) return INT16_MIN;
  return input;
}

// upper_half_product emulates X86_64 pmulhrsw.
// int16_t range is [-2^15, 2^15 - 1]
// int16_t * int16_t range is strictly included into [-2^30, 2^30 - 1],
// so the result can modeled by signed int31.
// To extract the most significant 16 bits one can shift int31 by 15.
static int16_t upper_half_product(int16_t x, int16_t m16) {
  int32_t product = (int32_t)x * (int32_t)m16;
  int32_t rounding = 1 << 14;
  int16_t result = (product + rounding) >> 15;
  return result;
}

static inline uint8_t clamp_u8(int16_t result, uint8_t zero_point, uint8_t min,
                               uint8_t max) {
  int16_t min_less_zero_point = (int16_t)min - (int16_t)zero_point;
  int16_t max_less_zero_point = (int16_t)max - (int16_t)zero_point;
  int16_t output = math_max_s16(result, min_less_zero_point);
  output = math_min_s16(output, max_less_zero_point);
  return output + (int16_t)zero_point;
}

static inline uint8_t xnn_qu8_requantize_rndnu16(int32_t input, float scale,
                                                 uint8_t zero_point,
                                                 uint8_t min, uint8_t max) {
  assert(scale < 1.0f);
  assert(scale >= 1.0f / 4294967296.0f /* 0x1.0p-32f */);

  // multiplier represents the 24 bit mantissa.
  // i.e. scale = 2^{-exp} * (multiplier * 2^{-23})
  struct F32 f32 = parse_f32(scale);

  // scale is modelled as 2^{exp} * m16 * 2^{-14}
  int exp = f32.exp;
  assert(exp <= -1);
  assert(exp >= -32);

  // m16 is in the range [2^14, 2^15 - 1]
  int16_t m16 = f32.multiplier >> 9;

  // Desired product: P = input * 2^exp * m16 * 2^-14
  // We care about the lower 8 bits of P with saturation,
  // i.e. if P >= 2^8 the answer should be 2^8 - 1.

  // To compute these 8 bits we would like to use the upper half
  // of a 16 bit x 16 bit product.
  // This is achived by a preshift of the input, depending on
  // the value of exp.
  int right_preshift = -exp - 1;
  int32_t preshifted_input = math_asr_s32(input, right_preshift);
  int16_t input16 = saturating_cast(preshifted_input);
  int16_t upper_half16 = upper_half_product(input16, m16);

  return clamp_u8(upper_half16, zero_point, min, max);
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
