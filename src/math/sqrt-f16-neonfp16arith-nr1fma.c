// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>

#include <arm_neon.h>

#include <xnnpack/math.h>
#include <xnnpack/math-stubs.h>


void xnn_math_f16_sqrt__neonfp16arith_nr1fma(
    size_t n,
    const void* input,
    void* output)
{
  assert(n % (8 * sizeof(__fp16)) == 0);

  const float16x8_t vhalf = vmovq_n_f16(0.5f);
  const __fp16* i = (const __fp16*) input;
  __fp16* o = (__fp16*) output;
  for (; n != 0; n -= 8 * sizeof(__fp16)) {
    const float16x8_t vx = vld1q_f16(i); i += 8;

    // Initial approximation
    const float16x8_t vrsqrtx = vrsqrteq_f16(vx);
    float16x8_t vsqrtx = vmulq_f16(vrsqrtx, vx);
    const float16x8_t vhalfrsqrtx = vmulq_f16(vrsqrtx, vhalf);

    // Netwon-Raphson iteration:
    //   residual   <- 0.5 - sqrtx * halfrsqrtx
    //   sqrtx      <- sqrtx + sqrtx * residual
    const float16x8_t vresidual = vfmsq_f16(vhalf, vsqrtx, vhalfrsqrtx);
    vsqrtx = vfmaq_f16(vsqrtx, vresidual, vsqrtx);

    const float16x8_t vy = vsqrtx;

    vst1q_f16(o, vy); o += 8;
  }
}
