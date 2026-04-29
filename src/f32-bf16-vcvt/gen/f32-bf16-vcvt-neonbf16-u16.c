// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/f32-bf16-vcvt/neonbf16.c.in
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


void xnn_f32_bf16_vcvt_ukernel__neonbf16_u16(
    size_t batch,
    const float* input,
    xnn_bfloat16* output,
    const void* params) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  uint16_t* o = (uint16_t*) output;
  for (; batch >= 16 * sizeof(float); batch -= 16 * sizeof(float)) {
    const float32x4_t vf0 = vld1q_f32(input); input += 4;
    const float32x4_t vf1 = vld1q_f32(input); input += 4;
    const float32x4_t vf2 = vld1q_f32(input); input += 4;
    const float32x4_t vf3 = vld1q_f32(input); input += 4;

    const uint16x8_t vbf0 = vreinterpretq_u16_bf16(vcombine_bf16(vcvt_bf16_f32(vf0), vcvt_bf16_f32(vf1)));
    const uint16x8_t vbf1 = vreinterpretq_u16_bf16(vcombine_bf16(vcvt_bf16_f32(vf2), vcvt_bf16_f32(vf3)));

    vst1q_u16(o, vbf0); o += 8;
    vst1q_u16(o, vbf1); o += 8;
  }
  for (; batch >= 4 * sizeof(float); batch -= 4 * sizeof(float)) {
    const float32x4_t vf = vld1q_f32(input); input += 4;

    const uint16x4_t vbf = vreinterpret_u16_bf16(vcvt_bf16_f32(vf));

    vst1_u16(o, vbf); o += 4;
  }
  if XNN_UNLIKELY(batch != 0) {
    assert(batch % sizeof(float) == 0);
    assert(batch >= 1 * sizeof(float));
    assert(batch <= 3 * sizeof(float));
    const float32x4_t vf = vld1q_f32(input);

    uint16x4_t vbf = vreinterpret_u16_bf16(vcvt_bf16_f32(vf));

    if (batch & (2 * sizeof(float))) {
      vst1_lane_u32((void*) o, vreinterpret_u32_u16(vbf), 0); o += 2;
      vbf = vext_u16(vbf, vbf, 2);
    }
    if (batch & (1 * sizeof(float))) {
      vst1_lane_u16(o, vbf, 0);
    }
  }
}
