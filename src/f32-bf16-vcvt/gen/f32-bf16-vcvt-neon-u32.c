// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/f32-bf16-vcvt/neon.c.in
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


void xnn_f32_bf16_vcvt_ukernel__neon_u32(
    size_t batch,
    const float* input,
    xnn_bfloat16* output,
    const void* params) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const uint32x4_t vbias = vdupq_n_u32(0x7FFFu);
  const uint32x4_t vone = vdupq_n_u32(1u);

  uint16_t* o = (uint16_t*) output;
  for (; batch >= 32 * sizeof(float); batch -= 32 * sizeof(float)) {
    const float32x4_t vf0 = vld1q_f32(input); input += 4;
    const float32x4_t vf1 = vld1q_f32(input); input += 4;
    const float32x4_t vf2 = vld1q_f32(input); input += 4;
    const float32x4_t vf3 = vld1q_f32(input); input += 4;
    const float32x4_t vf4 = vld1q_f32(input); input += 4;
    const float32x4_t vf5 = vld1q_f32(input); input += 4;
    const float32x4_t vf6 = vld1q_f32(input); input += 4;
    const float32x4_t vf7 = vld1q_f32(input); input += 4;

    const uint32x4_t vi0 = vreinterpretq_u32_f32(vf0);
    const uint32x4_t vi1 = vreinterpretq_u32_f32(vf1);
    const uint32x4_t vi2 = vreinterpretq_u32_f32(vf2);
    const uint32x4_t vi3 = vreinterpretq_u32_f32(vf3);
    const uint32x4_t vi4 = vreinterpretq_u32_f32(vf4);
    const uint32x4_t vi5 = vreinterpretq_u32_f32(vf5);
    const uint32x4_t vi6 = vreinterpretq_u32_f32(vf6);
    const uint32x4_t vi7 = vreinterpretq_u32_f32(vf7);

    const uint32x4_t vlsb0 = vandq_u32(vshrq_n_u32(vi0, 16), vone);
    const uint32x4_t vlsb1 = vandq_u32(vshrq_n_u32(vi1, 16), vone);
    const uint32x4_t vlsb2 = vandq_u32(vshrq_n_u32(vi2, 16), vone);
    const uint32x4_t vlsb3 = vandq_u32(vshrq_n_u32(vi3, 16), vone);
    const uint32x4_t vlsb4 = vandq_u32(vshrq_n_u32(vi4, 16), vone);
    const uint32x4_t vlsb5 = vandq_u32(vshrq_n_u32(vi5, 16), vone);
    const uint32x4_t vlsb6 = vandq_u32(vshrq_n_u32(vi6, 16), vone);
    const uint32x4_t vlsb7 = vandq_u32(vshrq_n_u32(vi7, 16), vone);

    const uint16x8_t vbf0 = vcombine_u16(
        vaddhn_u32(vi0, vaddq_u32(vbias, vlsb0)),
        vaddhn_u32(vi1, vaddq_u32(vbias, vlsb1)));
    const uint16x8_t vbf1 = vcombine_u16(
        vaddhn_u32(vi2, vaddq_u32(vbias, vlsb2)),
        vaddhn_u32(vi3, vaddq_u32(vbias, vlsb3)));
    const uint16x8_t vbf2 = vcombine_u16(
        vaddhn_u32(vi4, vaddq_u32(vbias, vlsb4)),
        vaddhn_u32(vi5, vaddq_u32(vbias, vlsb5)));
    const uint16x8_t vbf3 = vcombine_u16(
        vaddhn_u32(vi6, vaddq_u32(vbias, vlsb6)),
        vaddhn_u32(vi7, vaddq_u32(vbias, vlsb7)));

    vst1q_u16(o, vbf0); o += 8;
    vst1q_u16(o, vbf1); o += 8;
    vst1q_u16(o, vbf2); o += 8;
    vst1q_u16(o, vbf3); o += 8;
  }
  for (; batch >= 4 * sizeof(float); batch -= 4 * sizeof(float)) {
    const float32x4_t vf = vld1q_f32(input); input += 4;

    const uint32x4_t vi = vreinterpretq_u32_f32(vf);
    const uint32x4_t vlsb = vandq_u32(vshrq_n_u32(vi, 16), vone);
    const uint16x4_t vbf = vaddhn_u32(vi, vaddq_u32(vbias, vlsb));

    vst1_u16(o, vbf); o += 4;
  }
  if XNN_UNLIKELY(batch != 0) {
    assert(batch % sizeof(float) == 0);
    assert(batch >= 1 * sizeof(float));
    assert(batch <= 3 * sizeof(float));
    const float32x4_t vf = vld1q_f32(input);

    const uint32x4_t vi = vreinterpretq_u32_f32(vf);
    const uint32x4_t vlsb = vandq_u32(vshrq_n_u32(vi, 16), vone);
    uint16x4_t vbf = vaddhn_u32(vi, vaddq_u32(vbias, vlsb));

    if (batch & (2 * sizeof(float))) {
      vst1_lane_u32((void*) o, vreinterpret_u32_u16(vbf), 0); o += 2;
      vbf = vext_u16(vbf, vbf, 2);
    }
    if (batch & (1 * sizeof(float))) {
      vst1_lane_u16(o, vbf, 0);
    }
  }
}
