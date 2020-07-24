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

#include <fp16/bitcasts.h>

#include <xnnpack/requantization-stubs.h>


void xnn_qu8_requantize_fp32__scalar_lrintf(
    size_t n,
    const int32_t* input,
    float scale,
    uint8_t zero_point,
    uint8_t qmin,
    uint8_t qmax,
    uint8_t* output)
{
  assert(n % 4 == 0);
  assert(scale < 1.0f);
  assert(scale >= 0x1.0p-32f);

  const long lmin = (long) ((int32_t)(uint32_t) qmin - (int32_t)(uint32_t) zero_point);
  const long lmax = (long) ((int32_t)(uint32_t) qmax - (int32_t)(uint32_t) zero_point);
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

    const long x_rounded = lrintf(x_scaled);
    const long y_rounded = lrintf(y_scaled);
    const long z_rounded = lrintf(z_scaled);
    const long w_rounded = lrintf(w_scaled);

    const int32_t x_clamped = (int32_t)(x_rounded < lmin ? lmin : x_rounded > lmax ? lmax : x_rounded);
    const int32_t y_clamped = (int32_t)(y_rounded < lmin ? lmin : y_rounded > lmax ? lmax : y_rounded);
    const int32_t z_clamped = (int32_t)(z_rounded < lmin ? lmin : z_rounded > lmax ? lmax : z_rounded);
    const int32_t w_clamped = (int32_t)(w_rounded < lmin ? lmin : w_rounded > lmax ? lmax : w_rounded);

    const int32_t x_biased = x_clamped + (int32_t)(uint32_t) zero_point;
    const int32_t y_biased = y_clamped + (int32_t)(uint32_t) zero_point;
    const int32_t z_biased = z_clamped + (int32_t)(uint32_t) zero_point;
    const int32_t w_biased = w_clamped + (int32_t)(uint32_t) zero_point;

    output[0] = (uint8_t) x_biased;
    output[1] = (uint8_t) y_biased;
    output[2] = (uint8_t) z_biased;
    output[3] = (uint8_t) w_biased;
    output += 4;
  }
}

