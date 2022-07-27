// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <math.h>
#include <stdint.h>
#include <stddef.h>

#include <xnnpack/math.h>
#include <xnnpack/requantization-stubs.h>


void xnn_qs8_requantize_fp32__scalar_fmagic(
    size_t n,
    const int32_t* input,
    float scale,
    int8_t zero_point,
    int8_t qmin,
    int8_t qmax,
    int8_t* output)
{
  assert(n % 4 == 0);
  assert(scale < 1.0f);
  assert(scale >= 0x1.0p-32f);

  const float fmin = (float) ((int32_t) qmin - (int32_t) zero_point);
  const float fmax = (float) ((int32_t) qmax - (int32_t) zero_point);
  const float fmagic = 12582912.0f;
  const int32_t imagic = INT32_C(0x4B400000) - (int32_t) zero_point;
  for (; n != 0; n -= 4) {
    const int32_t x = input[0];
    const int32_t y = input[1];
    const int32_t z = input[2];
    const int32_t w = input[3];
    input += 4;

    const float x_scaled = (float) x * scale;
    const float y_scaled = (float) y * scale;
    const float z_scaled = (float) z * scale;
    const float w_scaled = (float) w * scale;

    const float x_clamped = math_min_f32(math_max_f32(x_scaled, fmin), fmax);
    const float y_clamped = math_min_f32(math_max_f32(y_scaled, fmin), fmax);
    const float z_clamped = math_min_f32(math_max_f32(z_scaled, fmin), fmax);
    const float w_clamped = math_min_f32(math_max_f32(w_scaled, fmin), fmax);

    const int32_t x_biased = (int32_t) float_as_uint32(x_clamped + fmagic) - imagic;
    const int32_t y_biased = (int32_t) float_as_uint32(y_clamped + fmagic) - imagic;
    const int32_t z_biased = (int32_t) float_as_uint32(z_clamped + fmagic) - imagic;
    const int32_t w_biased = (int32_t) float_as_uint32(w_clamped + fmagic) - imagic;

    output[0] = (int8_t) x_biased;
    output[1] = (int8_t) y_biased;
    output[2] = (int8_t) z_biased;
    output[3] = (int8_t) w_biased;
    output += 4;
  }
}
