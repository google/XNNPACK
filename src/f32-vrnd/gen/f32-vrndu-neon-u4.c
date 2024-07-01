// Auto-generated file. Do not edit!
//   Template: src/f32-vrnd/vrndu-neon.c.in
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


void xnn_f32_vrndu_ukernel__neon_u4(
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
  const float32x4_t vone = vmovq_n_f32(1.0f);
  for (; batch >= 4 * sizeof(float); batch -= 4 * sizeof(float)) {
    const float32x4_t vx0123 = vld1q_f32(input); input += 4;

    const int32x4_t vintx0123 = vcvtq_s32_f32(vx0123);

    uint32x4_t vrndmask0123 = vcaltq_f32(vx0123, vintegral_threshold);

    const float32x4_t vprerndx0123 = vcvtq_f32_s32(vintx0123);

    vrndmask0123 = vbicq_u32(vrndmask0123, vmovq_n_u32(UINT32_C(0x80000000)));

    const float32x4_t vrndx0123 = vbslq_f32(vrndmask0123, vprerndx0123, vx0123);

    uint32x4_t vadjmask0123 = vcgeq_f32(vrndx0123, vx0123);

    const float32x4_t vadjrndx0123 = vaddq_f32(vrndx0123, vone);

    vadjmask0123 = vorrq_u32(vadjmask0123, vmovq_n_u32(UINT32_C(0x80000000)));

    const float32x4_t vy0123 = vbslq_f32(vadjmask0123, vrndx0123, vadjrndx0123);

    vst1q_f32(output, vy0123); output += 4;
  }
  if XNN_UNLIKELY(batch != 0) {
    const float32x4_t vx = vld1q_f32(input);
    const int32x4_t vintx = vcvtq_s32_f32(vx);
    const float32x4_t vprerndx = vcvtq_f32_s32(vintx);
    uint32x4_t vrndmask = vcaltq_f32(vx, vintegral_threshold);
    vrndmask = vbicq_u32(vrndmask, vmovq_n_u32(UINT32_C(0x80000000)));
    const float32x4_t vrndx = vbslq_f32(vrndmask, vprerndx, vx);
    uint32x4_t vadjmask = vcgeq_f32(vrndx, vx);
    const float32x4_t vadjrndx = vaddq_f32(vrndx, vone);
    vadjmask = vorrq_u32(vadjmask, vmovq_n_u32(UINT32_C(0x80000000)));
    const float32x4_t vy = vbslq_f32(vadjmask, vrndx, vadjrndx);
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
