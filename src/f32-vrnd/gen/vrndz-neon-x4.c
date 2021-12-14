// Auto-generated file. Do not edit!
//   Template: src/f32-vrnd/vrndz-neon.c.in
//   Generator: tools/xngen
//
// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <arm_neon.h>

#include <xnnpack/common.h>
#include <xnnpack/math.h>
#include <xnnpack/vunary.h>


void xnn_f32_vrndz_ukernel__neon_x4(
    size_t n,
    const float* x,
    float* y,
    const union xnn_f32_rnd_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(n != 0);
  assert(n % sizeof(float) == 0);

  const float32x4_t vintegral_threshold = vreinterpretq_f32_u32(vmovq_n_u32(UINT32_C(0x4B000000)));
  for (; n >= 4 * sizeof(float); n -= 4 * sizeof(float)) {
    const float32x4_t vx0123 = vld1q_f32(x); x += 4;

    const int32x4_t vintx0123 = vcvtq_s32_f32(vx0123);

    uint32x4_t vrndmask0123 = vcaltq_f32(vx0123, vintegral_threshold);

    const float32x4_t vrndx0123 = vcvtq_f32_s32(vintx0123);

    vrndmask0123 = vbicq_u32(vrndmask0123, vmovq_n_u32(UINT32_C(0x80000000)));

    const float32x4_t vy0123 = vbslq_f32(vrndmask0123, vrndx0123, vx0123);

    vst1q_f32(y, vy0123); y += 4;
  }
  if XNN_UNLIKELY(n != 0) {
    const float32x4_t vx = vld1q_f32(x);
    const int32x4_t vintx = vcvtq_s32_f32(vx);
    uint32x4_t vrndmask = vcaltq_f32(vx, vintegral_threshold);
    const float32x4_t vrndx = vcvtq_f32_s32(vintx);
    vrndmask = vbicq_u32(vrndmask, vmovq_n_u32(UINT32_C(0x80000000)));
    const float32x4_t vy = vbslq_f32(vrndmask, vrndx, vx);
    float32x2_t vy_lo = vget_low_f32(vy);
    if (n & (2 * sizeof(float))) {
      vst1_f32(y, vy_lo); y += 2;
      vy_lo = vget_high_f32(vy);
    }
    if (n & (1 * sizeof(float))) {
      vst1_lane_f32(y, vy_lo, 0);
    }
  }
}
