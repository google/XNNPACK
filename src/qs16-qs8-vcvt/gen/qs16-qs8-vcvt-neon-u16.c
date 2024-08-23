// Auto-generated file. Do not edit!
//   Template: src/qs16-qs8-vcvt/neon.c.in
//   Generator: tools/xngen
//
// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <arm_neon.h>

#include "xnnpack/common.h"
#include "xnnpack/vcvt.h"

void xnn_qs16_qs8_vcvt_ukernel__neon_u16(
    size_t batch,
    const int16_t* input,
    int8_t* output,
    const union xnn_qs16_qs8_cvt_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(int16_t) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const int32x4_t vmultiplier = vld1q_dup_s32(&params->scalar.multiplier);
  const int16x8_t voutput_zero_point = vld1q_dup_s16(&params->scalar.output_zero_point);
  for (; batch >= 16 * sizeof(int16_t); batch -= 16 * sizeof(int16_t)) {
    const int16x8_t vx0 = vld1q_s16(input); input += 8;
    const int16x8_t vx1 = vld1q_s16(input); input += 8;

    int32x4_t vacc_lo0 = vshll_n_s16(vget_low_s16(vx0), 15);
    int32x4_t vacc_hi0 = vshll_n_s16(vget_high_s16(vx0), 15);
    int32x4_t vacc_lo1 = vshll_n_s16(vget_low_s16(vx1), 15);
    int32x4_t vacc_hi1 = vshll_n_s16(vget_high_s16(vx1), 15);

    vacc_lo0 = vqrdmulhq_s32(vacc_lo0, vmultiplier);
    vacc_hi0 = vqrdmulhq_s32(vacc_hi0, vmultiplier);
    vacc_lo1 = vqrdmulhq_s32(vacc_lo1, vmultiplier);
    vacc_hi1 = vqrdmulhq_s32(vacc_hi1, vmultiplier);

    int16x8_t vacc0 = vcombine_s16(vqmovn_s32(vacc_lo0), vqmovn_s32(vacc_hi0));
    int16x8_t vacc1 = vcombine_s16(vqmovn_s32(vacc_lo1), vqmovn_s32(vacc_hi1));

    vacc0 = vqaddq_s16(vacc0, voutput_zero_point);
    vacc1 = vqaddq_s16(vacc1, voutput_zero_point);

    const int8x8_t vy0 = vqmovn_s16(vacc0);
    const int8x8_t vy1 = vqmovn_s16(vacc1);

    vst1_s8(output, vy0); output += 8;
    vst1_s8(output, vy1); output += 8;
  }
  for (; batch >= 8 * sizeof(int16_t); batch -= 8 * sizeof(int16_t)) {
    const int16x8_t vx = vld1q_s16(input); input += 8;
    int32x4_t vacc_lo = vshll_n_s16(vget_low_s16(vx), 15);
    int32x4_t vacc_hi = vshll_n_s16(vget_high_s16(vx), 15);
    vacc_lo = vqrdmulhq_s32(vacc_lo, vmultiplier);
    vacc_hi = vqrdmulhq_s32(vacc_hi, vmultiplier);
    int16x8_t vacc = vcombine_s16(vqmovn_s32(vacc_lo), vqmovn_s32(vacc_hi));
    vacc = vqaddq_s16(vacc, voutput_zero_point);
    const int8x8_t vy = vqmovn_s16(vacc);
    vst1_s8(output, vy); output += 8;
  }
  if XNN_UNLIKELY(batch != 0) {
    assert(batch >= 1 * sizeof(int16_t));
    assert(batch <= 7 * sizeof(int16_t));

    const int16x8_t vx = vld1q_s16(input);
    int32x4_t vacc_lo = vshll_n_s16(vget_low_s16(vx), 15);
    int32x4_t vacc_hi = vshll_n_s16(vget_high_s16(vx), 15);
    vacc_lo = vqrdmulhq_s32(vacc_lo, vmultiplier);
    vacc_hi = vqrdmulhq_s32(vacc_hi, vmultiplier);
    int16x8_t vacc = vcombine_s16(vqmovn_s32(vacc_lo), vqmovn_s32(vacc_hi));
    vacc = vqaddq_s16(vacc, voutput_zero_point);
    int8x8_t vy = vqmovn_s16(vacc);

    if (batch & (4 * sizeof(int16_t))) {
      vst1_lane_u32((void*) output, vreinterpret_u32_s8(vy), 0); output += 4;
      vy = vext_s8(vy, vy, 4);
    }
    if (batch & (2 * sizeof(int16_t))) {
      vst1_lane_u16((void*) output, vreinterpret_u16_s8(vy), 0); output += 2;
      vy = vext_s8(vy, vy, 2);
    }
    if (batch & (1 * sizeof(int16_t))) {
      vst1_lane_s8((void*) output, vy, 0);
    }
  }
}
