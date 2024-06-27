// Auto-generated file. Do not edit!
//   Template: src/f32-vrnd/vrndd-neon.c.in
//   Generator: tools/xngen
//
// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <arm_neon.h>

#include "xnnpack/common.h"
#include "xnnpack/math.h"
#include "xnnpack/vunary.h"


void xnn_f32_vrndd_ukernel__neon_u8(
    size_t batch,
    const float* input,
    float* output,
    const union xnn_f32_rnd_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const float32x4_t vintegral_threshold = vreinterpretq_f32_u32(vmovq_n_u32(UINT32_C(0x4B000000)));
  const uint32x4_t vone = vreinterpretq_u32_f32(vmovq_n_f32(1.0f));
  for (; batch >= 8 * sizeof(float); batch -= 8 * sizeof(float)) {
    const float32x4_t vx0123 = vld1q_f32(input); input += 4;
    const float32x4_t vx4567 = vld1q_f32(input); input += 4;

    const int32x4_t vintx0123 = vcvtq_s32_f32(vx0123);
    const int32x4_t vintx4567 = vcvtq_s32_f32(vx4567);

    uint32x4_t vrndmask0123 = vcaltq_f32(vx0123, vintegral_threshold);
    uint32x4_t vrndmask4567 = vcaltq_f32(vx4567, vintegral_threshold);

    const float32x4_t vprerndx0123 = vcvtq_f32_s32(vintx0123);
    const float32x4_t vprerndx4567 = vcvtq_f32_s32(vintx4567);

    vrndmask0123 = vbicq_u32(vrndmask0123, vmovq_n_u32(UINT32_C(0x80000000)));
    vrndmask4567 = vbicq_u32(vrndmask4567, vmovq_n_u32(UINT32_C(0x80000000)));

    const float32x4_t vrndx0123 = vbslq_f32(vrndmask0123, vprerndx0123, vx0123);
    const float32x4_t vrndx4567 = vbslq_f32(vrndmask4567, vprerndx4567, vx4567);

    const uint32x4_t vadjmask0123 = vcgtq_f32(vrndx0123, vx0123);
    const uint32x4_t vadjmask4567 = vcgtq_f32(vrndx4567, vx4567);

    const float32x4_t vadjrndx0123 = vreinterpretq_f32_u32(vandq_u32(vadjmask0123, vone));
    const float32x4_t vadjrndx4567 = vreinterpretq_f32_u32(vandq_u32(vadjmask4567, vone));

    const float32x4_t vy0123 = vsubq_f32(vrndx0123, vadjrndx0123);
    const float32x4_t vy4567 = vsubq_f32(vrndx4567, vadjrndx4567);

    vst1q_f32(output, vy0123); output += 4;
    vst1q_f32(output, vy4567); output += 4;
  }
  for (; batch >= 4 * sizeof(float); batch -= 4 * sizeof(float)) {
    const float32x4_t vx = vld1q_f32(input); input += 4;
    const int32x4_t vintx = vcvtq_s32_f32(vx);
    uint32x4_t vrndmask = vcaltq_f32(vx, vintegral_threshold);
    const float32x4_t vprerndx = vcvtq_f32_s32(vintx);
    vrndmask = vbicq_u32(vrndmask, vmovq_n_u32(UINT32_C(0x80000000)));
    const float32x4_t vrndx = vbslq_f32(vrndmask, vprerndx, vx);
    const uint32x4_t vadjmask = vcgtq_f32(vrndx, vx);
    const float32x4_t vadjrndx = vreinterpretq_f32_u32(vandq_u32(vadjmask, vone));
    const float32x4_t vy = vsubq_f32(vrndx, vadjrndx);
    vst1q_f32(output, vy); output += 4;
  }
  if XNN_UNLIKELY(batch != 0) {
    const float32x4_t vx = vld1q_f32(input);
    const int32x4_t vintx = vcvtq_s32_f32(vx);
    uint32x4_t vrndmask = vcaltq_f32(vx, vintegral_threshold);
    const float32x4_t vprerndx = vcvtq_f32_s32(vintx);
    vrndmask = vbicq_u32(vrndmask, vmovq_n_u32(UINT32_C(0x80000000)));
    const float32x4_t vrndx = vbslq_f32(vrndmask, vprerndx, vx);
    const uint32x4_t vadjmask = vcgtq_f32(vrndx, vx);
    const float32x4_t vadjrndx = vreinterpretq_f32_u32(vandq_u32(vadjmask, vone));
    const float32x4_t vy = vsubq_f32(vrndx, vadjrndx);
    float32x2_t vy_lo = vget_low_f32(vy);
    if (batch & (2 * sizeof(float))) {
      vst1_f32(output, vy_lo); output += 2;
      vy_lo = vget_high_f32(vy);
    }
    if (batch & (1 * sizeof(float))) {
      vst1_lane_f32(output, vy_lo, 0);
    }
  }
}
