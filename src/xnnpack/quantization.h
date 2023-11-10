// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <assert.h>
#include <stdint.h>
#include <math.h>

#include <fp16/fp16.h>

#include <xnnpack/math.h>
#include <xnnpack/microparams.h>

static inline struct xnn_qd8_quantization_params xnn_qd8_asymmetric_quantization_params(
    float min, float max)
{
  struct xnn_qd8_quantization_params quantization_params;
  const float qmin = INT8_MIN;
  const float qmax = INT8_MAX;
  const float rmin = math_min_f32(0.0f, min);
  const float rmax = math_max_f32(0.0f, max);
  const float scale = rmin == rmax ? 1.f : (qmax - qmin) / (rmax - rmin);
  const float descaled_min = rmin * scale;
  const float descaled_max = rmax * scale;
  const float zero_point_from_min_error = qmin + descaled_min;
  const float zero_point_from_max_error = qmax + descaled_max;
  float zero_point =
      zero_point_from_min_error + zero_point_from_max_error > 0
      ? qmin - descaled_min
      : qmax - descaled_max;
  zero_point = math_max_f32(zero_point, qmin);
  zero_point = math_min_f32(zero_point, qmax);
  assert(zero_point >= INT8_MIN);
  assert(zero_point <= INT8_MAX);
  const int8_t nudged_zero_point = (int8_t) lrintf(zero_point);
  quantization_params.inv_scale = scale;
  quantization_params.zero_point = nudged_zero_point;
  return quantization_params;
}

static inline struct xnn_qd8_quantization_params xnn_f32_qd8_asymmetric_quantization_params(
    float min, float max)
{
  struct xnn_qd8_quantization_params params = xnn_qd8_asymmetric_quantization_params(min, max);
  params.inv_scale = 1.f / params.inv_scale;
  return params;
}

static inline struct xnn_qd8_quantization_params xnn_f16_qd8_asymmetric_quantization_params(
    uint16_t min, uint16_t max, uint16_t* f16_scale)
{
  struct xnn_qd8_quantization_params params = xnn_qd8_asymmetric_quantization_params(fp16_ieee_to_fp32_value(min), fp16_ieee_to_fp32_value(max));
  *f16_scale = fp16_ieee_from_fp32_value(params.inv_scale);
  params.inv_scale = 1.f / params.inv_scale;
  return params;
}
