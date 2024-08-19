// Auto-generated file. Do not edit!
//   Template: src/f32-qs8-vcvt/neonv8.c.in
//   Generator: tools/xngen
//
// Copyright 2021 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <arm_neon.h>

#include "xnnpack/common.h"
#include "xnnpack/intrinsics-polyfill.h"
#include "xnnpack/vcvt.h"


void xnn_f32_qs8_vcvt_ukernel__neonv8_u16(
    size_t batch,
    const float* input,
    int8_t* output,
    const union xnn_f32_qs8_cvt_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const float32x4_t vscale = vld1q_dup_f32(&params->scalar.scale);
  const int16x8_t voutput_zero_point = vld1q_dup_s16(&params->scalar.output_zero_point);
  const int8x16_t voutput_min = vld1q_dup_s8(&params->scalar.output_min);
  const int8x16_t voutput_max = vld1q_dup_s8(&params->scalar.output_max);
  for (; batch >= 16 * sizeof(float); batch -= 16 * sizeof(float)) {
    float32x4_t vx0123 = vld1q_f32(input); input += 4;
    float32x4_t vx4567 = vld1q_f32(input); input += 4;
    float32x4_t vx89AB = vld1q_f32(input); input += 4;
    float32x4_t vxCDEF = vld1q_f32(input); input += 4;

    vx0123 = vmulq_f32(vx0123, vscale);
    vx4567 = vmulq_f32(vx4567, vscale);
    vx89AB = vmulq_f32(vx89AB, vscale);
    vxCDEF = vmulq_f32(vxCDEF, vscale);

    const int32x4_t vacc0123 = vcvtnq_s32_f32(vx0123);
    const int32x4_t vacc4567 = vcvtnq_s32_f32(vx4567);
    const int32x4_t vacc89AB = vcvtnq_s32_f32(vx89AB);
    const int32x4_t vaccCDEF = vcvtnq_s32_f32(vxCDEF);

    int16x8_t vacc01234567 = vcombine_s16(vqmovn_s32(vacc0123), vqmovn_s32(vacc4567));
    int16x8_t vacc89ABCDEF = vcombine_s16(vqmovn_s32(vacc89AB), vqmovn_s32(vaccCDEF));

    vacc01234567 = vqaddq_s16(vacc01234567, voutput_zero_point);
    vacc89ABCDEF = vqaddq_s16(vacc89ABCDEF, voutput_zero_point);

    int8x16_t vy0123456789ABCDEF = vcombine_s8(vqmovn_s16(vacc01234567), vqmovn_s16(vacc89ABCDEF));

    vy0123456789ABCDEF = vmaxq_s8(vy0123456789ABCDEF, voutput_min);

    vy0123456789ABCDEF = vminq_s8(vy0123456789ABCDEF, voutput_max);

    vst1q_s8(output, vy0123456789ABCDEF); output += 16;
  }
  for (; batch >= 8 * sizeof(float); batch -= 8 * sizeof(float)) {
    float32x4_t vx_lo = vld1q_f32(input); input += 4;
    float32x4_t vx_hi = vld1q_f32(input); input += 4;

    vx_lo = vmulq_f32(vx_lo, vscale);
    vx_hi = vmulq_f32(vx_hi, vscale);

    const int32x4_t vacc_lo = vcvtnq_s32_f32(vx_lo);
    const int32x4_t vacc_hi = vcvtnq_s32_f32(vx_hi);

    int16x8_t vacc = vcombine_s16(vqmovn_s32(vacc_lo), vqmovn_s32(vacc_hi));
    vacc = vqaddq_s16(vacc, voutput_zero_point);

    int8x8_t vy = vqmovn_s16(vacc);
    vy = vmax_s8(vy, vget_low_s8(voutput_min));
    vy = vmin_s8(vy, vget_low_s8(voutput_max));
    vst1_s8(output, vy); output += 8;
  }
  if XNN_UNLIKELY(batch != 0) {
    assert(batch >= 1 * sizeof(float));
    assert(batch <= 7 * sizeof(float));
    float32x4_t vx_lo = vld1q_f32(input);
    const float* x_hi = (const float*) ((uintptr_t) input + (batch & (4 * sizeof(float))));
    float32x4_t vx_hi = vld1q_f32(x_hi);

    vx_lo = vmulq_f32(vx_lo, vscale);
    vx_hi = vmulq_f32(vx_hi, vscale);

    const int32x4_t vacc_lo = vcvtnq_s32_f32(vx_lo);
    const int32x4_t vacc_hi = vcvtnq_s32_f32(vx_hi);

    int16x8_t vacc = vcombine_s16(vqmovn_s32(vacc_lo), vqmovn_s32(vacc_hi));
    vacc = vqaddq_s16(vacc, voutput_zero_point);

    int8x8_t vy = vqmovn_s16(vacc);
    vy = vmax_s8(vy, vget_low_s8(voutput_min));
    vy = vmin_s8(vy, vget_low_s8(voutput_max));

    if (batch & (4 * sizeof(float))) {
      vst1_lane_u32((void*) output, vreinterpret_u32_s8(vy), 0); output += 4;
      vy = vext_s8(vy, vy, 4);
    }
    if (batch & (2 * sizeof(float))) {
      vst1_lane_u16((void*) output, vreinterpret_u16_s8(vy), 0); output += 2;
      vy = vext_s8(vy, vy, 2);
    }
    if (batch & (1 * sizeof(float))) {
      vst1_lane_s8(output, vy, 0);
    }
  }
}
