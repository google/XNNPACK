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


void xnn_f32_bf16_vcvt_ukernel__neon_u8(
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
  for (; batch >= 8 * sizeof(float); batch -= 8 * sizeof(float)) {
    const float32x4_t vf0 = vld1q_f32(input); input += 4;
    const float32x4_t vf1 = vld1q_f32(input); input += 4;

    const uint32x4_t vi0 = vreinterpretq_u32_f32(vf0);
    const uint32x4_t vi1 = vreinterpretq_u32_f32(vf1);

    const uint32x4_t vlsb0 = vandq_u32(vshrq_n_u32(vi0, 16), vone);
    const uint32x4_t vlsb1 = vandq_u32(vshrq_n_u32(vi1, 16), vone);

    const uint16x8_t vbf0 = vcombine_u16(
        vaddhn_u32(vi0, vaddq_u32(vbias, vlsb0)),
        vaddhn_u32(vi1, vaddq_u32(vbias, vlsb1)));

    vst1q_u16(o, vbf0); o += 8;
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
