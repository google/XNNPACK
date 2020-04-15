// Auto-generated file. Do not edit!
//   Template: src/f32-hswish/neon.c.in
//   Generator: tools/xngen
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <arm_neon.h>

#include <xnnpack/common.h>
#include <xnnpack/vbinary.h>


void xnn_f32_hswish_ukernel__neon_x4(
    size_t n,
    const float* x,
    float* y,
    const union xnn_f32_hswish_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(n != 0);
  assert(n % sizeof(float) == 0);

  const float32x4_t vsixth = vld1q_dup_f32(&params->scalar.sixth);
  const float32x4_t vhalf = vld1q_dup_f32(&params->scalar.half);
  const float32x4_t vone = vld1q_dup_f32(&params->scalar.one);
  const float32x4_t vzero = vdupq_n_f32(0.0f);

  for (; n >= 4 * sizeof(float); n -= 4 * sizeof(float)) {
    const float32x4_t vx0123 = vld1q_f32(x); x += 4;

    float32x4_t vacc0123 = vmlaq_f32(vhalf, vx0123, vsixth);

    vacc0123 = vmaxq_f32(vacc0123, vzero);

    vacc0123 = vminq_f32(vacc0123, vone);

    vacc0123 = vmulq_f32(vacc0123, vx0123);

    vst1q_f32(y, vacc0123); y += 4;
  }
  for (; n >= 4 * sizeof(float); n -= 4 * sizeof(float)) {
    const float32x4_t vx0123 = vld1q_f32(x); x += 4;
    float32x4_t vacc0123 = vmlaq_f32(vhalf, vx0123, vsixth);
    vacc0123 = vmaxq_f32(vacc0123, vzero);
    vacc0123 = vminq_f32(vacc0123, vone);
    vacc0123 = vmulq_f32(vacc0123, vx0123);
    vst1q_f32(y, vacc0123); y += 4;
  }
  if XNN_UNLIKELY(n != 0) {
    const float32x4_t vx0123 = vld1q_f32(x);
    float32x4_t vacc0123 = vmlaq_f32(vhalf, vx0123, vsixth);
    vacc0123 = vmaxq_f32(vacc0123, vzero);
    vacc0123 = vminq_f32(vacc0123, vone);
    vacc0123 = vmulq_f32(vacc0123, vx0123);

    float32x2_t vacc01 = vget_low_f32(vacc0123);
    if (n & (2 * sizeof(float))) {
      vst1_f32(y, vacc01); y += 2;
      vacc01 = vget_high_f32(vacc0123);
    }
    if (n & (1 * sizeof(float))) {
      vst1_lane_f32(y, vacc01, 0);
    }
  }
}
