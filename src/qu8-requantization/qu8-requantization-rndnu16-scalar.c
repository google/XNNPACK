// All rights reserved.
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stdint.h>
#include <stddef.h>

#include "xnnpack/math.h"
#include "xnnpack/requantization.h"
#include "xnnpack/requantization-stubs.h"

void xnn_qu8_requantize_rndnu16__scalar(
    size_t n,
    const int32_t* input,
    float scale,
    uint8_t zero_point,
    uint8_t qmin,
    uint8_t qmax,
    uint8_t* output)
{
  assert(n % 16 == 0);
  assert(scale < 256.0f);
  assert(scale >= 0x1.0p-32f);
  for (size_t i = 0; i < n; ++i) {
    *(output + i) = xnn_qu8_requantize_rndnu16(*(input + i), scale, zero_point, qmin, qmax);
  }
}
