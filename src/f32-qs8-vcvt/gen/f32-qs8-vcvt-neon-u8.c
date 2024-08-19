// Auto-generated file. Do not edit!
//   Template: src/f32-qs8-vcvt/neon.c.in
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


void xnn_f32_qs8_vcvt_ukernel__neon_u8(
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
  const float32x4_t vmagic_bias = vdupq_n_f32(12582912.0f);
  const int32x4_t vmagic_bias_less_zero_point = vdupq_n_s32(INT32_C(0x4B400000) - (int32_t) params->scalar.output_zero_point);
  const int8x8_t voutput_min = vld1_dup_s8(&params->scalar.output_min);
  const int8x8_t voutput_max = vld1_dup_s8(&params->scalar.output_max);
  XNN_FORCE_REALIZATION(vmagic_bias);
  for (; batch >= 8 * sizeof(float); batch -= 8 * sizeof(float)) {
    float32x4_t vx_lo = vld1q_f32(input); input += 4;
    float32x4_t vx_hi = vld1q_f32(input); input += 4;

    vx_lo = vmulq_f32(vx_lo, vscale);
    vx_hi = vmulq_f32(vx_hi, vscale);

    vx_lo = vaddq_f32(vx_lo, vmagic_bias);
    vx_hi = vaddq_f32(vx_hi, vmagic_bias);

    const int32x4_t vacc_lo = vqsubq_s32(vreinterpretq_s32_f32(vx_lo), vmagic_bias_less_zero_point);
    const int32x4_t vacc_hi = vqsubq_s32(vreinterpretq_s32_f32(vx_hi), vmagic_bias_less_zero_point);

    const int16x8_t vacc = vcombine_s16(vqmovn_s32(vacc_lo), vqmovn_s32(vacc_hi));

    int8x8_t vy = vqmovn_s16(vacc);
    vy = vmax_s8(vy, voutput_min);
    vy = vmin_s8(vy, voutput_max);
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

    vx_lo = vaddq_f32(vx_lo, vmagic_bias);
    vx_hi = vaddq_f32(vx_hi, vmagic_bias);

    const int32x4_t vacc_lo = vqsubq_s32(vreinterpretq_s32_f32(vx_lo), vmagic_bias_less_zero_point);
    const int32x4_t vacc_hi = vqsubq_s32(vreinterpretq_s32_f32(vx_hi), vmagic_bias_less_zero_point);

    const int16x8_t vacc = vcombine_s16(vqmovn_s32(vacc_lo), vqmovn_s32(vacc_hi));

    int8x8_t vy = vqmovn_s16(vacc);
    vy = vmax_s8(vy, voutput_min);
    vy = vmin_s8(vy, voutput_max);

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
