// Auto-generated file. Do not edit!
//   Template: src/f32-vhswish/neon.c.in
//   Generator: tools/xngen
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <arm_neon.h>

#include "xnnpack/common.h"
#include "xnnpack/vunary.h"


void xnn_f32_vhswish_ukernel__neon_u4(
    size_t batch,
    const float* input,
    float* output,
    const union xnn_f32_hswish_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const float32x4_t vsixth = vdupq_n_f32(0x1.555556p-3f);
  const float32x4_t vthree = vdupq_n_f32(3.0f);
  const int32x4_t vsix = vreinterpretq_s32_f32(vdupq_n_f32(6.0f));
  const int32x4_t vzero = vdupq_n_s32(0);

  XNN_FORCE_REALIZATION(vsixth);
  XNN_FORCE_REALIZATION(vthree);
  XNN_FORCE_REALIZATION(vsix);
  // XNN_FORCE_REALIZATION(vzero);

  for (; batch >= 4 * sizeof(float); batch -= 4 * sizeof(float)) {
    float32x4_t vx = vld1q_f32(input); input += 4;
    float32x4_t vacc = vaddq_f32(vx, vthree);
    vx = vmulq_f32(vx, vsixth);
    vacc = vreinterpretq_f32_s32(vmaxq_s32(vreinterpretq_s32_f32(vacc), vzero));
    vacc = vreinterpretq_f32_s32(vminq_s32(vreinterpretq_s32_f32(vacc), vsix));
    vacc = vmulq_f32(vacc, vx);
    vst1q_f32(output, vacc); output += 4;
  }
  if XNN_UNLIKELY(batch != 0) {
    float32x4_t vx = vld1q_f32(input);
    float32x4_t vacc = vaddq_f32(vx, vthree);
    vx = vmulq_f32(vx, vsixth);
    vacc = vreinterpretq_f32_s32(vmaxq_s32(vreinterpretq_s32_f32(vacc), vzero));
    vacc = vreinterpretq_f32_s32(vminq_s32(vreinterpretq_s32_f32(vacc), vsix));
    vacc = vmulq_f32(vacc, vx);

    float32x2_t vacc_lo = vget_low_f32(vacc);
    if (batch & (2 * sizeof(float))) {
      vst1_f32(output, vacc_lo); output += 2;
      vacc_lo = vget_high_f32(vacc);
    }
    if (batch & (1 * sizeof(float))) {
      vst1_lane_f32(output, vacc_lo, 0);
    }
  }
}
