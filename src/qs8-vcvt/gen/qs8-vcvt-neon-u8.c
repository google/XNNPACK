// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/qs8-vcvt/neon.c.in
//   Generator: tools/xngen
//
// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <arm_neon.h>
#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include "src/xnnpack/common.h"
#include "src/xnnpack/microparams.h"
#include "src/xnnpack/vcvt.h"


void xnn_qs8_vcvt_ukernel__neon_u8(
    size_t batch,
    const int8_t* input,
    int8_t* output,
    const struct xnn_qs8_cvt_params* restrict params) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(int8_t) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const int32x4_t vbias = vdupq_n_s32(
      (int32_t) (((uint32_t) (int32_t) params->scalar.output_zero_point) << 16) -
      (int32_t) params->scalar.multiplier * (int32_t) params->scalar.input_zero_point +
      INT32_C(0x8000));
  const int32x4_t vmultiplier = vdupq_n_s32(params->scalar.multiplier);
  for (; batch >= 8 * sizeof(int8_t); batch -= 8 * sizeof(int8_t)) {
    const int8x8_t vx = vld1_s8(input); input += 8;
    const int16x8_t vx16 = vmovl_s8(vx);
    int32x4_t vacc_lo = vmlaq_s32(vbias, vmovl_s16(vget_low_s16(vx16)), vmultiplier);
    int32x4_t vacc_hi = vmlaq_s32(vbias, vmovl_s16(vget_high_s16(vx16)), vmultiplier);
    vacc_lo = vshrq_n_s32(vacc_lo, 16);
    vacc_hi = vshrq_n_s32(vacc_hi, 16);
    const int16x8_t vy16 = vcombine_s16(vqmovn_s32(vacc_lo), vqmovn_s32(vacc_hi));
    const int8x8_t vy = vqmovn_s16(vy16);
    vst1_s8(output, vy); output += 8;
  }
  if XNN_UNLIKELY(batch != 0) {
    assert(batch >= 1 * sizeof(int8_t));
    assert(batch <= 7 * sizeof(int8_t));

    const int8x8_t vx = vld1_s8(input);
    const int16x8_t vx16 = vmovl_s8(vx);
    int32x4_t vacc_lo = vmlaq_s32(vbias, vmovl_s16(vget_low_s16(vx16)), vmultiplier);
    int32x4_t vacc_hi = vmlaq_s32(vbias, vmovl_s16(vget_high_s16(vx16)), vmultiplier);
    vacc_lo = vshrq_n_s32(vacc_lo, 16);
    vacc_hi = vshrq_n_s32(vacc_hi, 16);
    const int16x8_t vy16 = vcombine_s16(vqmovn_s32(vacc_lo), vqmovn_s32(vacc_hi));
    int8x8_t vy = vqmovn_s16(vy16);

    if (batch & (4 * sizeof(int8_t))) {
      vst1_lane_u32((void*) output, vreinterpret_u32_s8(vy), 0); output += 4;
      vy = vext_s8(vy, vy, 4);
    }
    if (batch & (2 * sizeof(int8_t))) {
      vst1_lane_u16((void*) output, vreinterpret_u16_s8(vy), 0); output += 2;
      vy = vext_s8(vy, vy, 2);
    }
    if (batch & (1 * sizeof(int8_t))) {
      vst1_lane_s8(output, vy, 0);
    }
  }
}
