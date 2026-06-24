// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/f32-qs8-vcvt/neonv8.c.in
//   Generator: tools/xngen
//
// Copyright 2021 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <arm_neon.h>
#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include "src/xnnpack/common.h"
#include "src/xnnpack/intrinsics-polyfill.h"
#include "src/xnnpack/math.h"
#include "src/xnnpack/microparams.h"
#include "src/xnnpack/vcvt.h"


void xnn_bf16_qs8_vcvt_ukernel__neonv8_u24(
    size_t batch,
    const xnn_bfloat16* input,
    int8_t* output,
    const struct xnn_bf16_qs8_cvt_params* restrict params) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(xnn_bfloat16) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const float32x4_t vscale = vdupq_n_f32(xnn_bfloat16_to_float(params->scalar.scale));
  const int16x8_t voutput_zero_point = vdupq_n_s16(params->scalar.output_zero_point);
  for (; batch >= 24 * sizeof(xnn_bfloat16); batch -= 24 * sizeof(xnn_bfloat16)) {
    const uint16x8_t vbf01234567 = vld1q_u16((const uint16_t*) input); input += 8;
    float32x4_t vx0123 = vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(vbf01234567), 16));
    float32x4_t vx4567 = vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(vbf01234567), 16));
    const uint16x8_t vbf89ABCDEF = vld1q_u16((const uint16_t*) input); input += 8;
    float32x4_t vx89AB = vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(vbf89ABCDEF), 16));
    float32x4_t vxCDEF = vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(vbf89ABCDEF), 16));
    const uint16x8_t vbfGHIJKLMN = vld1q_u16((const uint16_t*) input); input += 8;
    float32x4_t vxGHIJ = vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(vbfGHIJKLMN), 16));
    float32x4_t vxKLMN = vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(vbfGHIJKLMN), 16));

    vx0123 = vmulq_f32(vx0123, vscale);
    vx4567 = vmulq_f32(vx4567, vscale);
    vx89AB = vmulq_f32(vx89AB, vscale);
    vxCDEF = vmulq_f32(vxCDEF, vscale);
    vxGHIJ = vmulq_f32(vxGHIJ, vscale);
    vxKLMN = vmulq_f32(vxKLMN, vscale);

    const int32x4_t vacc0123 = vcvtnq_s32_f32(vx0123);
    const int32x4_t vacc4567 = vcvtnq_s32_f32(vx4567);
    const int32x4_t vacc89AB = vcvtnq_s32_f32(vx89AB);
    const int32x4_t vaccCDEF = vcvtnq_s32_f32(vxCDEF);
    const int32x4_t vaccGHIJ = vcvtnq_s32_f32(vxGHIJ);
    const int32x4_t vaccKLMN = vcvtnq_s32_f32(vxKLMN);

    int16x8_t vacc01234567 = vcombine_s16(vqmovn_s32(vacc0123), vqmovn_s32(vacc4567));
    int16x8_t vacc89ABCDEF = vcombine_s16(vqmovn_s32(vacc89AB), vqmovn_s32(vaccCDEF));
    int16x8_t vaccGHIJKLMN = vcombine_s16(vqmovn_s32(vaccGHIJ), vqmovn_s32(vaccKLMN));

    vacc01234567 = vqaddq_s16(vacc01234567, voutput_zero_point);
    vacc89ABCDEF = vqaddq_s16(vacc89ABCDEF, voutput_zero_point);
    vaccGHIJKLMN = vqaddq_s16(vaccGHIJKLMN, voutput_zero_point);

    int8x16_t vy0123456789ABCDEF = vcombine_s8(vqmovn_s16(vacc01234567), vqmovn_s16(vacc89ABCDEF));
    int8x8_t vyGHIJKLMN = vqmovn_s16(vaccGHIJKLMN);

    vst1q_s8(output, vy0123456789ABCDEF); output += 16;
    vst1_s8(output, vyGHIJKLMN); output += 8;
  }
  for (; batch >= 8 * sizeof(xnn_bfloat16); batch -= 8 * sizeof(xnn_bfloat16)) {
    const uint16x8_t vbf = vld1q_u16((const uint16_t*) input); input += 8;
    float32x4_t vx_lo = vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(vbf), 16));
    float32x4_t vx_hi = vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(vbf), 16));

    vx_lo = vmulq_f32(vx_lo, vscale);
    vx_hi = vmulq_f32(vx_hi, vscale);

    const int32x4_t vacc_lo = vcvtnq_s32_f32(vx_lo);
    const int32x4_t vacc_hi = vcvtnq_s32_f32(vx_hi);

    int16x8_t vacc = vcombine_s16(vqmovn_s32(vacc_lo), vqmovn_s32(vacc_hi));
    vacc = vqaddq_s16(vacc, voutput_zero_point);

    int8x8_t vy = vqmovn_s16(vacc);
    vst1_s8(output, vy); output += 8;
  }
  if XNN_UNLIKELY(batch != 0) {
    assert(batch >= 1 * sizeof(xnn_bfloat16));
    assert(batch <= 7 * sizeof(xnn_bfloat16));
    const uint16x8_t vbf = vld1q_u16((const uint16_t*) input);
    float32x4_t vx_lo = vreinterpretq_f32_u32(vshll_n_u16(vget_low_u16(vbf), 16));
    float32x4_t vx_hi = vreinterpretq_f32_u32(vshll_n_u16(vget_high_u16(vbf), 16));

    vx_lo = vmulq_f32(vx_lo, vscale);
    vx_hi = vmulq_f32(vx_hi, vscale);

    const int32x4_t vacc_lo = vcvtnq_s32_f32(vx_lo);
    const int32x4_t vacc_hi = vcvtnq_s32_f32(vx_hi);

    int16x8_t vacc = vcombine_s16(vqmovn_s32(vacc_lo), vqmovn_s32(vacc_hi));
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
