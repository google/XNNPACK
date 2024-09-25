// Auto-generated file. Do not edit!
//   Template: src/f32-vbinary/vopc-neon.c.in
//   Generator: tools/xngen
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <arm_neon.h>

#include "xnnpack/common.h"
#include "xnnpack/vbinary.h"


void xnn_f32_vpreluc_ukernel__neon_u8(
    size_t batch,
    const float* input_a,
    const float* input_b,
    float* output,
    const struct xnn_f32_default_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input_a != NULL);
  assert(input_b != NULL);
  assert(output != NULL);

  const float32x4_t vb = vld1q_dup_f32(input_b);

  for (; batch >= 8 * sizeof(float); batch -= 8 * sizeof(float)) {
    const float32x4_t va0 = vld1q_f32(input_a); input_a += 4;
    const float32x4_t va1 = vld1q_f32(input_a); input_a += 4;

    float32x4_t vacc0 = vmulq_f32(va0, vb);
    float32x4_t vacc1 = vmulq_f32(va1, vb);

    const uint32x4_t vm0 = vcltq_s32(vreinterpretq_s32_f32(va0), vmovq_n_s32(0));
    const uint32x4_t vm1 = vcltq_s32(vreinterpretq_s32_f32(va1), vmovq_n_s32(0));

    vacc0 = vbslq_f32(vm0, vacc0, va0);
    vacc1 = vbslq_f32(vm1, vacc1, va1);

    vst1q_f32(output, vacc0); output += 4;
    vst1q_f32(output, vacc1); output += 4;
  }
  for (; batch >= 4 * sizeof(float); batch -= 4 * sizeof(float)) {
    const float32x4_t va = vld1q_f32(input_a); input_a += 4;

    float32x4_t vacc = vmulq_f32(va, vb);
    const uint32x4_t vm = vcltq_s32(vreinterpretq_s32_f32(va), vmovq_n_s32(0));
    vacc = vbslq_f32(vm, vacc, va);

    vst1q_f32(output, vacc); output += 4;
  }
  if XNN_UNLIKELY(batch != 0) {
    const float32x4_t va = vld1q_f32(input_a);

    float32x4_t vacc = vmulq_f32(va, vb);
    const uint32x4_t vm = vcltq_s32(vreinterpretq_s32_f32(va), vmovq_n_s32(0));
    vacc = vbslq_f32(vm, vacc, va);

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
