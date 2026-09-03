// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/bf16-qs8-vcvt/neonv8.c.in
//   Generator: tools/xngen
//
// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <arm_neon.h>
#include <assert.h>
#include <float.h>
#include <stddef.h>
#include <stdint.h>

#include "src/xnnpack/common.h"
#include "src/xnnpack/math.h"
#include "src/xnnpack/microparams.h"
#include "src/xnnpack/vcvt.h"


void xnn_bf16_qs8_vcvt_ukernel__neonv8_u32(
    size_t batch,
    const xnn_bfloat16* input,
    int8_t* output,
    const struct xnn_bf16_qs8_cvt_params* restrict params) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(xnn_bfloat16) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const uint16_t* i = (const uint16_t*) input;
  // Match the scalar path when the reciprocal underflows in BF16.
  const float32x4_t vscale = vdupq_n_f32(
      math_max_f32(FLT_MIN, xnn_bfloat16_to_float(params->scalar.scale)));
  const int16x8_t voutput_zero_point =
      vdupq_n_s16(params->scalar.output_zero_point);
  for (; batch >= 32 * sizeof(xnn_bfloat16); batch -= 32 * sizeof(xnn_bfloat16)) {
    const uint16x8_t vbf0 = vld1q_u16(i); i += 8;
    const uint16x8_t vbf1 = vld1q_u16(i); i += 8;
    const uint16x8_t vbf2 = vld1q_u16(i); i += 8;
    const uint16x8_t vbf3 = vld1q_u16(i); i += 8;

    float32x4_t vx0 = vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(vbf0), 16));
    float32x4_t vx1 = vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(vbf0), 16));
    float32x4_t vx2 = vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(vbf1), 16));
    float32x4_t vx3 = vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(vbf1), 16));
    float32x4_t vx4 = vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(vbf2), 16));
    float32x4_t vx5 = vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(vbf2), 16));
    float32x4_t vx6 = vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(vbf3), 16));
    float32x4_t vx7 = vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(vbf3), 16));

    vx0 = vmulq_f32(vx0, vscale);
    vx1 = vmulq_f32(vx1, vscale);
    vx2 = vmulq_f32(vx2, vscale);
    vx3 = vmulq_f32(vx3, vscale);
    vx4 = vmulq_f32(vx4, vscale);
    vx5 = vmulq_f32(vx5, vscale);
    vx6 = vmulq_f32(vx6, vscale);
    vx7 = vmulq_f32(vx7, vscale);

    const int32x4_t vacc0 = vcvtnq_s32_f32(vx0);
    const int32x4_t vacc1 = vcvtnq_s32_f32(vx1);
    const int32x4_t vacc2 = vcvtnq_s32_f32(vx2);
    const int32x4_t vacc3 = vcvtnq_s32_f32(vx3);
    const int32x4_t vacc4 = vcvtnq_s32_f32(vx4);
    const int32x4_t vacc5 = vcvtnq_s32_f32(vx5);
    const int32x4_t vacc6 = vcvtnq_s32_f32(vx6);
    const int32x4_t vacc7 = vcvtnq_s32_f32(vx7);

    int16x8_t vacc01 = vcombine_s16(vqmovn_s32(vacc0), vqmovn_s32(vacc1));
    int16x8_t vacc23 = vcombine_s16(vqmovn_s32(vacc2), vqmovn_s32(vacc3));
    int16x8_t vacc45 = vcombine_s16(vqmovn_s32(vacc4), vqmovn_s32(vacc5));
    int16x8_t vacc67 = vcombine_s16(vqmovn_s32(vacc6), vqmovn_s32(vacc7));

    vacc01 = vqaddq_s16(vacc01, voutput_zero_point);
    vacc23 = vqaddq_s16(vacc23, voutput_zero_point);
    vacc45 = vqaddq_s16(vacc45, voutput_zero_point);
    vacc67 = vqaddq_s16(vacc67, voutput_zero_point);

    const int8x16_t vy0 = vcombine_s8(vqmovn_s16(vacc01), vqmovn_s16(vacc23));
    const int8x16_t vy2 = vcombine_s8(vqmovn_s16(vacc45), vqmovn_s16(vacc67));

    vst1q_s8(output, vy0); output += 16;
    vst1q_s8(output, vy2); output += 16;
  }
  for (; batch >= 8 * sizeof(xnn_bfloat16); batch -= 8 * sizeof(xnn_bfloat16)) {
    const uint16x8_t vbf = vld1q_u16(i); i += 8;

    float32x4_t vx_lo =
        vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(vbf), 16));
    float32x4_t vx_hi =
        vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(vbf), 16));

    vx_lo = vmulq_f32(vx_lo, vscale);
    vx_hi = vmulq_f32(vx_hi, vscale);

    const int32x4_t vacc_lo = vcvtnq_s32_f32(vx_lo);
    const int32x4_t vacc_hi = vcvtnq_s32_f32(vx_hi);

    int16x8_t vacc =
        vcombine_s16(vqmovn_s32(vacc_lo), vqmovn_s32(vacc_hi));
    vacc = vqaddq_s16(vacc, voutput_zero_point);

    const int8x8_t vy = vqmovn_s16(vacc);
    vst1_s8(output, vy); output += 8;
  }
  if XNN_UNLIKELY(batch != 0) {
    assert(batch >= 1 * sizeof(xnn_bfloat16));
    assert(batch <= 7 * sizeof(xnn_bfloat16));
    const uint16x8_t vbf = vld1q_u16(i);

    float32x4_t vx_lo =
        vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(vbf), 16));
    float32x4_t vx_hi =
        vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(vbf), 16));

    vx_lo = vmulq_f32(vx_lo, vscale);
    vx_hi = vmulq_f32(vx_hi, vscale);

    const int32x4_t vacc_lo = vcvtnq_s32_f32(vx_lo);
    const int32x4_t vacc_hi = vcvtnq_s32_f32(vx_hi);

    int16x8_t vacc =
        vcombine_s16(vqmovn_s32(vacc_lo), vqmovn_s32(vacc_hi));
    vacc = vqaddq_s16(vacc, voutput_zero_point);

    int8x8_t vy = vqmovn_s16(vacc);
    if (batch & (4 * sizeof(xnn_bfloat16))) {
      vst1_lane_u32((void*) output, vreinterpret_u32_s8(vy), 0); output += 4;
      vy = vext_s8(vy, vy, 4);
    }
    if (batch & (2 * sizeof(xnn_bfloat16))) {
      vst1_lane_u16((void*) output, vreinterpret_u16_s8(vy), 0); output += 2;
      vy = vext_s8(vy, vy, 2);
    }
    if (batch & (1 * sizeof(xnn_bfloat16))) {
      vst1_lane_s8(output, vy, 0);
    }
  }
}
