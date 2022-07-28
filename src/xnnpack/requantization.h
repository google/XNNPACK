// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <stdint.h>
#include <stddef.h>
#include <assert.h>
#include <math.h>

#include <xnnpack/common.h>
#include <xnnpack/math.h>
#include <xnnpack/microparams.h>


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
  union xnn_qu8_add_minmax_params params)
{
  // Multiply by factors and accumulate products.
  int32_t acc = params.scalar.bias + (int32_t) (uint32_t) a * params.scalar.a_multiplier + (int32_t) (uint32_t) b * params.scalar.b_multiplier;

  // Shift right with rounding away from zero.
  acc = math_asr_s32(acc, params.scalar.shift);

  // Clamp and add output zero point.
  acc = math_max_s32(acc, params.scalar.output_min_less_zero_point);
  acc = math_min_s32(acc, params.scalar.output_max_less_zero_point);
  return (int8_t) ((int32_t) acc + params.scalar.output_zero_point);
}

static inline int8_t xnn_qs8_quantize_add(
  int8_t a, int8_t b,
  union xnn_qs8_add_minmax_params params)
{
  // Multiply by factors and accumulate products.
  int32_t acc = params.scalar.bias + (int32_t) a * params.scalar.a_multiplier + (int32_t) b * params.scalar.b_multiplier;

  // Shift right with rounding away from zero.
  acc = math_asr_s32(acc, params.scalar.shift);

  // Clamp and add output zero point.
  acc = math_max_s32(acc, params.scalar.output_min_less_zero_point);
  acc = math_min_s32(acc, params.scalar.output_max_less_zero_point);
  return (int8_t) ((int32_t) acc + params.scalar.output_zero_point);
}

inline static int8_t xnn_qs8_quantize(float val, float scale, int32_t zero_point)
{
  return (int8_t) lrintf(fminf(fmaxf(val / scale + (float) zero_point, -128.0f), 127.0f));
}

inline static uint8_t xnn_qu8_quantize(float val, float scale, int32_t zero_point)
{
  return (uint8_t) lrintf(fminf(fmaxf(val / scale + (float) zero_point, 0.0f), 255.0f));
}
