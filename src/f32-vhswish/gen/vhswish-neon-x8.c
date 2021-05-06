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

#include <xnnpack/common.h>
#include <xnnpack/vunary.h>


void xnn_f32_vhswish_ukernel__neon_x8(
    size_t n,
    const float* x,
    float* y,
    const union xnn_f32_hswish_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_DISABLE_TSAN
{
  assert(n != 0);
  assert(n % sizeof(float) == 0);

  const float32x4_t vsixth = vld1q_dup_f32(&params->scalar.sixth);
  const float32x4_t vthree = vld1q_dup_f32(&params->scalar.three);
  const int32x4_t vsix = vreinterpretq_s32_f32(vld1q_dup_f32(&params->scalar.six));
  const int32x4_t vzero = vdupq_n_s32(0);

  for (; n >= 8 * sizeof(float); n -= 8 * sizeof(float)) {
    float32x4_t vx0123 = vld1q_f32(x); x += 4;
    float32x4_t vx4567 = vld1q_f32(x); x += 4;

    float32x4_t vacc0123 = vaddq_f32(vx0123, vthree);
    vx0123 = vmulq_f32(vx0123, vsixth);
    float32x4_t vacc4567 = vaddq_f32(vx4567, vthree);
    vx4567 = vmulq_f32(vx4567, vsixth);

    vacc0123 = vreinterpretq_f32_s32(vmaxq_s32(vreinterpretq_s32_f32(vacc0123), vzero));
    vacc4567 = vreinterpretq_f32_s32(vmaxq_s32(vreinterpretq_s32_f32(vacc4567), vzero));

    vacc0123 = vreinterpretq_f32_s32(vminq_s32(vreinterpretq_s32_f32(vacc0123), vsix));
    vacc4567 = vreinterpretq_f32_s32(vminq_s32(vreinterpretq_s32_f32(vacc4567), vsix));

    vacc0123 = vmulq_f32(vacc0123, vx0123);
    vacc4567 = vmulq_f32(vacc4567, vx4567);

    vst1q_f32(y, vacc0123); y += 4;
    vst1q_f32(y, vacc4567); y += 4;
  }
  for (; n >= 4 * sizeof(float); n -= 4 * sizeof(float)) {
    float32x4_t vx = vld1q_f32(x); x += 4;
    float32x4_t vacc = vaddq_f32(vx, vthree);
    vx = vmulq_f32(vx, vsixth);
    vacc = vreinterpretq_f32_s32(vmaxq_s32(vreinterpretq_s32_f32(vacc), vzero));
    vacc = vreinterpretq_f32_s32(vminq_s32(vreinterpretq_s32_f32(vacc), vsix));
    vacc = vmulq_f32(vacc, vx);
    vst1q_f32(y, vacc); y += 4;
  }
  if XNN_UNLIKELY(n != 0) {
    float32x4_t vx = vld1q_f32(x);
    float32x4_t vacc = vaddq_f32(vx, vthree);
    vx = vmulq_f32(vx, vsixth);
    vacc = vreinterpretq_f32_s32(vmaxq_s32(vreinterpretq_s32_f32(vacc), vzero));
    vacc = vreinterpretq_f32_s32(vminq_s32(vreinterpretq_s32_f32(vacc), vsix));
    vacc = vmulq_f32(vacc, vx);

    float32x2_t vacc_lo = vget_low_f32(vacc);
    if (n & (2 * sizeof(float))) {
      vst1_f32(y, vacc_lo); y += 2;
      vacc_lo = vget_high_f32(vacc);
    }
    if (n & (1 * sizeof(float))) {
      vst1_lane_f32(y, vacc_lo, 0);
    }
  }
}
