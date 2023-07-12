// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <assert.h>
#include <stdint.h>
#include <math.h>

#include <xnnpack/math.h>
#include <xnnpack/microparams.h>

static inline struct xnn_qd8_quantization_params xnn_f32_qd8_asymmetric_quantization_params(
    float min, float max)
{
  struct xnn_qd8_quantization_params quantization_params;
  const float qmin = INT8_MIN;
  const float qmax = INT8_MAX;
  const float rmin = math_min_f32(0.0f, min);
  const float rmax = math_max_f32(0.0f, max);
  const float scale = rmin == rmax ? 1.f : (qmax - qmin) / (rmax - rmin);
  const float rmin_scale = rmin * scale;
  const float rmax_scale = rmax * scale;
  const float zero_point_from_min_error = qmin + rmin_scale;
  const float zero_point_from_max_error = qmax + rmax_scale;
  float zero_point =
      zero_point_from_min_error + zero_point_from_max_error > 0
      ? qmin - rmin_scale
      : qmax - rmax_scale;
  zero_point = math_max_f32(zero_point, qmin);
  zero_point = math_min_f32(zero_point, qmax);
  assert(zero_point >= INT8_MIN);
  assert(zero_point <= INT8_MAX);
  const int8_t nudged_zero_point = ((int8_t) rintf(zero_point));
  quantization_params.scale = scale;
  quantization_params.zero_point = nudged_zero_point;
  return quantization_params;
}
