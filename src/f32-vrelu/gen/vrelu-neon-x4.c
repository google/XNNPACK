// Auto-generated file. Do not edit!
//   Template: src/f32-vrelu/neon.c.in
//   Generator: tools/xngen
//
// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <arm_neon.h>

#include <xnnpack/vunary.h>
#include <xnnpack/common.h>


void xnn_f32_vrelu_ukernel__neon_x4(
    size_t n,
    const float* x,
    float* y,
    const union xnn_f32_relu_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_DISABLE_TSAN
{
  assert(n != 0);
  assert(n % sizeof(float) == 0);
  assert(x != NULL);
  assert(y != NULL);

  const float32x4_t vzero = vmovq_n_f32(0.0f);

  for (; n >= 4 * sizeof(float); n -= 4 * sizeof(float)) {
    float32x4_t vacc0123 = vld1q_f32(x); x += 4;

    vacc0123 = vmaxq_f32(vacc0123, vzero);

    vst1q_f32(y, vacc0123); y += 4;
  }
  if XNN_UNLIKELY(n != 0) {
    float32x4_t vacc = vld1q_f32(x);
    vacc = vmaxq_f32(vacc, vzero);

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
