// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <arm_neon.h>
#include <assert.h>
#include <stddef.h>

#include "xnnpack/math-stubs.h"
#include "xnnpack/math.h"


void xnn_math_f32_sqrt__neon_nr1rsqrts(
    size_t n,
    const float* input,
    float* output)
{
  assert(n % (4 * sizeof(float)) == 0);

  for (; n != 0; n -= 4 * sizeof(float)) {
    const float32x4_t vx = vld1q_f32(input); input += 4;

    // Initial approximation
    float32x4_t vrsqrtx = vrsqrteq_f32(vx);

    // Netwon-Raphson iteration: rsqrt_x <- rsqrt_x * ((3 - x * (rsqrt_x * rsqrt_x)) / 2)
    // Note: vrsqrtsq_f32(x, y) := (3 - x * y) / 2
    vrsqrtx = vmulq_f32(vrsqrtx, vrsqrtsq_f32(vx, vmulq_f32(vrsqrtx, vrsqrtx)));

    // Reconstruct sqrt(x) = rsqrt(x) * x
    const float32x4_t vy = vmulq_f32(vrsqrtx, vx);

    vst1q_f32(output, vy); output += 4;
  }
}
