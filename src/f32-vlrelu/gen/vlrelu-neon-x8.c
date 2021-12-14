// Auto-generated file. Do not edit!
//   Template: src/f32-vlrelu/neon.c.in
//   Generator: tools/xngen
//
// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <arm_neon.h>

#include <xnnpack/common.h>
#include <xnnpack/vunary.h>


void xnn_f32_vlrelu_ukernel__neon_x8(
    size_t n,
    const float* x,
    float* y,
    const union xnn_f32_lrelu_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(n != 0);
  assert(n % sizeof(float) == 0);

  const float32x4_t vslope = vld1q_dup_f32(&params->scalar.slope);

  for (; n >= 8 * sizeof(float); n -= 8 * sizeof(float)) {
    const float32x4_t vx0123 = vld1q_f32(x); x += 4;
    const float32x4_t vx4567 = vld1q_f32(x); x += 4;

    float32x4_t vacc0123 = vmulq_f32(vx0123, vslope);
    const uint32x4_t vmask0123 = vcltq_s32(vreinterpretq_s32_f32(vx0123), vmovq_n_s32(0));
    float32x4_t vacc4567 = vmulq_f32(vx4567, vslope);
    const uint32x4_t vmask4567 = vcltq_s32(vreinterpretq_s32_f32(vx4567), vmovq_n_s32(0));

    vacc0123 = vbslq_f32(vmask0123, vacc0123, vx0123);
    vacc4567 = vbslq_f32(vmask4567, vacc4567, vx4567);

    vst1q_f32(y, vacc0123); y += 4;
    vst1q_f32(y, vacc4567); y += 4;
  }
  for (; n >= 4 * sizeof(float); n -= 4 * sizeof(float)) {
    const float32x4_t vx = vld1q_f32(x); x += 4;
    float32x4_t vacc = vmulq_f32(vx, vslope);
    const uint32x4_t vmask = vcltq_s32(vreinterpretq_s32_f32(vx), vmovq_n_s32(0));
    vacc = vbslq_f32(vmask, vacc, vx);
    vst1q_f32(y, vacc); y += 4;
  }
  if XNN_UNLIKELY(n != 0) {
    const float32x4_t vx = vld1q_f32(x);
    float32x4_t vacc = vmulq_f32(vx, vslope);
    const uint32x4_t vmask = vcltq_s32(vreinterpretq_s32_f32(vx), vmovq_n_s32(0));
    vacc = vbslq_f32(vmask, vacc, vx);

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
