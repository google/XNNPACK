// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <math.h>
#include <stddef.h>

#include <xnnpack/common.h>
#include <xnnpack/math-stubs.h>


void xnn_math_f32_roundz__scalar_trunc(
    size_t n,
    const float* input,
    float* output)
{
  assert(n % sizeof(float) == 0);

  for (; n != 0; n -= sizeof(float)) {
    const float vx = *input++;

    const float vy = truncf(vx);

    *output++ = vy;
  }
}
