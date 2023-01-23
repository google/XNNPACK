// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>

#include <arm_neon.h>

#include <xnnpack/math.h>
#include <xnnpack/math-stubs.h>


void xnn_math_f32_sqrt__neonfma_nr3fma(
    size_t n,
    const float* input,
    float* output)
{
  assert(n % (4 * sizeof(float)) == 0);

  const float32x4_t vhalf = vmovq_n_f32(0.5f);
  for (; n != 0; n -= 4 * sizeof(float)) {
    const float32x4_t vx = vld1q_f32(input); input += 4;

    // Initial approximation
    const float32x4_t vrsqrtx = vrsqrteq_f32(vx);
    float32x4_t vsqrtx = vmulq_f32(vrsqrtx, vx);
    float32x4_t vhalfrsqrtx = vmulq_f32(vrsqrtx, vhalf);

    // Netwon-Raphson iteration:
    //   residual   <- 0.5 - sqrtx * halfrsqrtx
    //   halfrsqrtx <- halfrsqrtx + halfrsqrtx * residual
    //   sqrtx      <- sqrtx + sqrtx * residual
    float32x4_t vresidual = vfmsq_f32(vhalf, vsqrtx, vhalfrsqrtx);
    vhalfrsqrtx = vfmaq_f32(vhalfrsqrtx, vresidual, vhalfrsqrtx);
    vsqrtx = vfmaq_f32(vsqrtx, vresidual, vsqrtx);

    vresidual = vfmsq_f32(vhalf, vsqrtx, vhalfrsqrtx);
    vhalfrsqrtx = vfmaq_f32(vhalfrsqrtx, vresidual, vhalfrsqrtx);
    vsqrtx = vfmaq_f32(vsqrtx, vresidual, vsqrtx);

    vresidual = vfmsq_f32(vhalf, vsqrtx, vhalfrsqrtx);
    vsqrtx = vfmaq_f32(vsqrtx, vresidual, vsqrtx);

    const float32x4_t vy = vsqrtx;

    vst1q_f32(output, vy); output += 4;
  }
}
