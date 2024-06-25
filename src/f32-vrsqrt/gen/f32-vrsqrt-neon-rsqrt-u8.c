// Auto-generated file. Do not edit!
//   Template: src/f32-vrsqrt/neon-rsqrt.c.in
//   Generator: tools/xngen
//
// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <math.h>

#include <arm_neon.h>

#include "xnnpack/common.h"
#include "xnnpack/vunary.h"


// In the following, we do a single Newton-Raphson step on the equation
// $x^{-2} - a$, which expands to:
//
//  $$x_{k+1} = 0.5 * x_k * (3.0 - a * x_k^2)$$
//
// So we do the following steps:
//
//  1. t0 = x_k
//  2. t1 = t0 * t0   (x_k^2)
//  3. t2 = a * t1    (a * x_k^2)
//  4. t3 = 3.0 - t2  (3.0 - a * x_k^2)
//  5. t4 = 0.5 * t0  (0.5 * x_k)
//  6. y  = t3 * t4   (0.5 * x_k * (3.0 - a * x_k^2))
//
// Where $x_k$ is the original approximation and `y` contains the improved
// approximation $x_{k+1}$.
//
// Note that Arm/NEON provides the `vrsqrtsq_f32` instruction, which does steps
// 3-5 in the above in a single instruction.
//
// Note also that the initial approximation computed by the `vrsqrteq_f32`
// instruction is only accurate to 9 bits (as opposed to 12 or 14 for x86_64),
// which requires us to do two steps of the above.


void xnn_f32_vrsqrt_ukernel__neon_rsqrt_u8(
    size_t batch,
    const float* input,
    float* output,
    const union xnn_f32_rsqrt_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  for (; batch >= 8 * sizeof(float); batch -= 8 * sizeof(float)) {
    const float32x4_t vx0123 = vld1q_f32(input); input += 4;
    const float32x4_t vx4567 = vld1q_f32(input); input += 4;

    const float32x4_t vt0_0123 = vrsqrteq_f32(vx0123);
    const float32x4_t vt0_4567 = vrsqrteq_f32(vx4567);
    const float32x4_t vt1_0123 = vmulq_f32(vt0_0123, vt0_0123);
    const float32x4_t vt1_4567 = vmulq_f32(vt0_4567, vt0_4567);
    const float32x4_t vt2_0123 = vrsqrtsq_f32(vx0123, vt1_0123);
    const float32x4_t vt2_4567 = vrsqrtsq_f32(vx4567, vt1_4567);
    const float32x4_t vt3_0123 = vmulq_f32(vt0_0123, vt2_0123);
    const float32x4_t vt3_4567 = vmulq_f32(vt0_4567, vt2_4567);
    const float32x4_t vt4_0123 = vmulq_f32(vt3_0123, vt3_0123);
    const float32x4_t vt4_4567 = vmulq_f32(vt3_4567, vt3_4567);
    const float32x4_t vt5_0123 = vrsqrtsq_f32(vx0123, vt4_0123);
    const float32x4_t vt5_4567 = vrsqrtsq_f32(vx4567, vt4_4567);
    const float32x4_t vy0123 = vmulq_f32(vt3_0123, vt5_0123);
    const float32x4_t vy4567 = vmulq_f32(vt3_4567, vt5_4567);

    vst1q_f32(output, vy0123); output += 4;
    vst1q_f32(output, vy4567); output += 4;
  }
  for (; batch >= 4 * sizeof(float); batch -= 4 * sizeof(float)) {
    const float32x4_t vx = vld1q_f32(input); input += 4;

    const float32x4_t vt0 = vrsqrteq_f32(vx);
    const float32x4_t vt1 = vmulq_f32(vt0, vt0);
    const float32x4_t vt2 = vrsqrtsq_f32(vx, vt1);
    const float32x4_t vt3 = vmulq_f32(vt0, vt2);
    const float32x4_t vt4 = vmulq_f32(vt3, vt3);
    const float32x4_t vt5 = vrsqrtsq_f32(vx, vt4);
    const float32x4_t vy = vmulq_f32(vt3, vt5);

    vst1q_f32(output, vy); output += 4;
  }
  if XNN_UNLIKELY(batch != 0) {
    const float32x4_t vx = vld1q_f32(input);
    const float32x2_t vx_lo = vget_low_f32(vx);
    const float32x2_t vx_hi = vget_high_f32(vx);

    const float32x2_t vt0 = vrsqrte_f32(vx_lo);
    const float32x2_t vt1 = vmul_f32(vt0, vt0);
    const float32x2_t vt2 = vrsqrts_f32(vx_lo, vt1);
    const float32x2_t vt3 = vmul_f32(vt0, vt2);
    const float32x2_t vt4 = vmul_f32(vt3, vt3);
    const float32x2_t vt5 = vrsqrts_f32(vx_lo, vt4);
    float32x2_t vy_lo = vmul_f32(vt3, vt5);

    if (batch & (2 * sizeof(float))) {
      vst1_f32(output, vy_lo); output += 2;

      const float32x2_t vt0 = vrsqrte_f32(vx_hi);
      const float32x2_t vt1 = vmul_f32(vt0, vt0);
      const float32x2_t vt2 = vrsqrts_f32(vx_hi, vt1);
      const float32x2_t vt3 = vmul_f32(vt0, vt2);
      const float32x2_t vt4 = vmul_f32(vt3, vt3);
      const float32x2_t vt5 = vrsqrts_f32(vx_hi, vt4);
      vy_lo = vmul_f32(vt3, vt5);
    }

    if (batch & (1 * sizeof(float))) {
      vst1_lane_f32(output, vy_lo, 0);
    }
  }
}
