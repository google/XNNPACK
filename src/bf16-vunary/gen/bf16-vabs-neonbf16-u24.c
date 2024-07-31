// Auto-generated file. Do not edit!
//   Template: src/bf16-vunary/neon.c.in
//   Generator: tools/xngen
//
// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <arm_neon.h>

#include "xnnpack/common.h"
#include "xnnpack/vunary.h"


void xnn_bf16_vabs_ukernel__neonbf16_u24(
    size_t batch,
    const void* input,
    void* output,
    const union xnn_bf16_default_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(bfloat16_t) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const bfloat16_t* i = (const bfloat16_t*) input;
  bfloat16_t* o = (bfloat16_t*) output;
  uint16x8_t vmask = vdupq_n_u16(0x7FFF);
  for (; batch >= 24 * sizeof(bfloat16_t); batch -= 24 * sizeof(bfloat16_t)) {
    const bfloat16x8_t vx01234567 = vld1q_bf16(i); i+= 8;
    const bfloat16x8_t vx89ABCDEF = vld1q_bf16(i); i+= 8;
    const bfloat16x8_t vxGHIJKLMN = vld1q_bf16(i); i+= 8;

    const bfloat16x8_t vy01234567 = vreinterpretq_bf16_u16(vandq_u16(vreinterpretq_u16_bf16(vx01234567), vmask));
    const bfloat16x8_t vy89ABCDEF = vreinterpretq_bf16_u16(vandq_u16(vreinterpretq_u16_bf16(vx89ABCDEF), vmask));
    const bfloat16x8_t vyGHIJKLMN = vreinterpretq_bf16_u16(vandq_u16(vreinterpretq_u16_bf16(vxGHIJKLMN), vmask));

    vst1q_bf16(o, vy01234567); o+= 8;
    vst1q_bf16(o, vy89ABCDEF); o+= 8;
    vst1q_bf16(o, vyGHIJKLMN); o+= 8;
  }
  for (; batch >= 8 * sizeof(bfloat16_t); batch -= 8 * sizeof(bfloat16_t)) {
    const bfloat16x8_t vx = vld1q_bf16(i); i+= 8;
    const bfloat16x8_t vy = vreinterpretq_bf16_u16(vandq_u16(vreinterpretq_u16_bf16(vx), vmask));

    vst1q_bf16(o, vy); o += 8;
  }
  if XNN_UNLIKELY(batch != 0) {
    const bfloat16x8_t vx = vld1q_bf16(i);
    const bfloat16x8_t vy = vreinterpretq_bf16_u16(vandq_u16(vreinterpretq_u16_bf16(vx), vmask));

    bfloat16x4_t vy_lo = vget_low_bf16(vy);
    if (batch & (4 * sizeof(bfloat16_t))) {
      vst1_bf16(o, vy_lo); o += 4;
      vy_lo = vget_high_bf16(vy);
    }
    if (batch & (2 * sizeof(bfloat16_t))) {
      vst1_lane_u32((void*) o, vreinterpret_u32_bf16(vy_lo), 0); o += 2;
      vy_lo = vreinterpret_bf16_u16(vext_u16(vreinterpret_u16_bf16(vy_lo), vreinterpret_u16_bf16(vy_lo), 2));
    }
    if (batch & (1 * sizeof(bfloat16_t))) {
      vst1_lane_bf16(o, vy_lo, 0);
    }
  }
}
