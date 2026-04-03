// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/bf16-f32-vcvt/neon.c.in
//   Generator: tools/xngen
//
// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <arm_neon.h>
#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include "src/xnnpack/common.h"
#include "src/xnnpack/vcvt.h"


void xnn_bf16_f32_vcvt_ukernel__neon_u32(
    size_t batch,
    const xnn_bfloat16* input,
    float* output,
    const void* params) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(xnn_bfloat16) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const uint16_t* i = (const uint16_t*) input;
  for (; batch >= 32 * sizeof(xnn_bfloat16); batch -= 32 * sizeof(xnn_bfloat16)) {
    const uint16x8_t vbf0 = vld1q_u16(i); i += 8;
    const uint16x8_t vbf1 = vld1q_u16(i); i += 8;
    const uint16x8_t vbf2 = vld1q_u16(i); i += 8;
    const uint16x8_t vbf3 = vld1q_u16(i); i += 8;

    const float32x4_t vf0 = vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(vbf0), 16));
    const float32x4_t vf1 = vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(vbf0), 16));
    const float32x4_t vf2 = vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(vbf1), 16));
    const float32x4_t vf3 = vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(vbf1), 16));
    const float32x4_t vf4 = vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(vbf2), 16));
    const float32x4_t vf5 = vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(vbf2), 16));
    const float32x4_t vf6 = vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(vbf3), 16));
    const float32x4_t vf7 = vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(vbf3), 16));

    vst1q_f32(output, vf0); output += 4;
    vst1q_f32(output, vf1); output += 4;
    vst1q_f32(output, vf2); output += 4;
    vst1q_f32(output, vf3); output += 4;
    vst1q_f32(output, vf4); output += 4;
    vst1q_f32(output, vf5); output += 4;
    vst1q_f32(output, vf6); output += 4;
    vst1q_f32(output, vf7); output += 4;
  }
  for (; batch >= 8 * sizeof(xnn_bfloat16); batch -= 8 * sizeof(xnn_bfloat16)) {
    const uint16x8_t vbf = vld1q_u16(i); i += 8;

    const float32x4_t vf_lo = vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(vbf), 16));
    const float32x4_t vf_hi = vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(vbf), 16));

    vst1q_f32(output, vf_lo); output += 4;
    vst1q_f32(output, vf_hi); output += 4;
  }
  if XNN_UNLIKELY(batch != 0) {
    assert(batch >= 1 * sizeof(xnn_bfloat16));
    assert(batch <= 7 * sizeof(xnn_bfloat16));
    const uint16x8_t vbf = vld1q_u16(i);

    float32x4_t vf = vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(vbf), 16));
    if (batch & (4 * sizeof(xnn_bfloat16))) {
      vst1q_f32(output, vf); output += 4;
      vf = vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(vbf), 16));
    }
    float32x2_t vf_lo = vget_low_f32(vf);
    if (batch & (2 * sizeof(xnn_bfloat16))) {
      vst1_f32(output, vf_lo); output += 2;
      vf_lo = vget_high_f32(vf);
    }
    if (batch & (1 * sizeof(xnn_bfloat16))) {
      vst1_lane_f32(output, vf_lo, 0);
    }
  }
}
