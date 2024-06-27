// Auto-generated file. Do not edit!
//   Template: src/f16-vrsqrt/neonfp16arith-rsqrt.c.in
//   Generator: tools/xngen
//
// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

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
// Note that Arm/NEON provides the `vrsqrtsq_f16` instruction, which does steps
// 3-5 in the above in a single instruction.


void xnn_f16_vrsqrt_ukernel__neonfp16arith_rsqrt_u16(
    size_t batch,
    const void* input,
    void* output,
    const union xnn_f16_rsqrt_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(uint16_t) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const uint16_t* i = (const uint16_t*) input;
  uint16_t* o = (uint16_t*) output;

  for (; batch >= 16 * sizeof(uint16_t); batch -= 16 * sizeof(uint16_t)) {
    const float16x8_t vx0 = vreinterpretq_f16_u16(vld1q_u16(i)); i += 8;
    const float16x8_t vx1 = vreinterpretq_f16_u16(vld1q_u16(i)); i += 8;

    const float16x8_t vt0_0 = vrsqrteq_f16(vx0);
    const float16x8_t vt0_1 = vrsqrteq_f16(vx1);
    const float16x8_t vt1_0 = vmulq_f16(vt0_0, vt0_0);
    const float16x8_t vt1_1 = vmulq_f16(vt0_1, vt0_1);
    const float16x8_t vt2_0 = vrsqrtsq_f16(vx0, vt1_0);
    const float16x8_t vt2_1 = vrsqrtsq_f16(vx1, vt1_1);
    const float16x8_t vy0 = vmulq_f16(vt0_0, vt2_0);
    const float16x8_t vy1 = vmulq_f16(vt0_1, vt2_1);

    vst1q_u16(o, vreinterpretq_u16_f16(vy0)); o += 8;
    vst1q_u16(o, vreinterpretq_u16_f16(vy1)); o += 8;
  }
  for (; batch >= 8 * sizeof(uint16_t); batch -= 8 * sizeof(uint16_t)) {
    const float16x8_t vx = vreinterpretq_f16_u16(vld1q_u16(i)); i += 8;

    const float16x8_t vt0 = vrsqrteq_f16(vx);
    const float16x8_t vt1 = vmulq_f16(vt0, vt0);
    const float16x8_t vt2 = vrsqrtsq_f16(vx, vt1);
    const float16x8_t vy = vmulq_f16(vt0, vt2);

    vst1q_u16(o, vreinterpretq_u16_f16(vy)); o += 8;
  }
  if XNN_UNLIKELY(batch != 0) {
    const float16x8_t vx = vreinterpretq_f16_u16(vld1q_u16(i));
    const float16x4_t vx_lo = vget_low_f16(vx);
    const float16x4_t vx_hi = vget_high_f16(vx);

    const float16x4_t vt0 = vrsqrte_f16(vx_lo);
    const float16x4_t vt1 = vmul_f16(vt0, vt0);
    const float16x4_t vt2 = vrsqrts_f16(vx_lo, vt1);
    float16x4_t vy_lo = vmul_f16(vt0, vt2);

    if (batch & (4 * sizeof(uint16_t))) {
      vst1_u16(o, vreinterpret_u16_f16(vy_lo)); o += 4;

      const float16x4_t vt0 = vrsqrte_f16(vx_hi);
      const float16x4_t vt1 = vmul_f16(vt0, vt0);
      const float16x4_t vt2 = vrsqrts_f16(vx_hi, vt1);
      vy_lo = vmul_f16(vt0, vt2);
    }

    if (batch & (2 * sizeof(uint16_t))) {
      vst1_lane_u32((void*) o, vreinterpret_u32_f16(vy_lo), 0); o += 2;
      vy_lo = vext_f16(vy_lo, vy_lo, 2);
    }

    if (batch & (1 * sizeof(uint16_t))) {
      vst1_lane_u16(o, vreinterpret_u16_f16(vy_lo), 0);
    }
  }
}
