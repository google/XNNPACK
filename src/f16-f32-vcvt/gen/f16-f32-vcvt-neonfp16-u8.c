// Auto-generated file. Do not edit!
//   Template: src/f16-f32-vcvt/neonfp16.c.in
//   Generator: tools/xngen
//
// Copyright 2021 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <arm_neon.h>

#include "xnnpack/common.h"
#include "xnnpack/vcvt.h"


void xnn_f16_f32_vcvt_ukernel__neonfp16_u8(
    size_t batch,
    const void* input,
    float* output,
    const void* params) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(uint16_t) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const uint16_t* i = (const uint16_t*) input;
  for (; batch >= 8 * sizeof(uint16_t); batch -= 8 * sizeof(uint16_t)) {
    const float16x8_t vh = vreinterpretq_f16_u16(vld1q_u16(i)); i += 8;

    const float32x4_t vf_lo = vcvt_f32_f16(vget_low_f16(vh));
    const float32x4_t vf_hi = vcvt_f32_f16(vget_high_f16(vh));

    vst1q_f32(output, vf_lo); output += 4;
    vst1q_f32(output, vf_hi); output += 4;
  }
  if XNN_UNLIKELY(batch != 0) {
    assert(batch >= 1 * sizeof(uint16_t));
    assert(batch <= 7 * sizeof(uint16_t));
    const float16x8_t vh = vreinterpretq_f16_u16(vld1q_u16(i)); i += 8;

    float32x4_t vf = vcvt_f32_f16(vget_low_f16(vh));
    if (batch & (4 * sizeof(uint16_t))) {
      vst1q_f32(output, vf); output += 4;
      vf = vcvt_f32_f16(vget_high_f16(vh));
    }
    float32x2_t vf_lo = vget_low_f32(vf);
    if (batch & (2 * sizeof(uint16_t))) {
      vst1_f32(output, vf_lo); output += 2;
      vf_lo = vget_high_f32(vf);
    }
    if (batch & (1 * sizeof(uint16_t))) {
      vst1_lane_f32(output, vf_lo, 0);
    }
  }
}
