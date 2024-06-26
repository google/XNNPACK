// Auto-generated file. Do not edit!
//   Template: src/f16-qs8-vcvt/neonfp16arith.c.in
//   Generator: tools/xngen
//
// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <arm_neon.h>

#include "xnnpack/common.h"
#include "xnnpack/intrinsics-polyfill.h"
#include "xnnpack/vcvt.h"


void xnn_f16_qs8_vcvt_ukernel__neonfp16arith_u32(
    size_t batch,
    const void* input,
    int8_t* output,
    const union xnn_f16_qs8_cvt_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(uint16_t) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const uint16_t* i = (const uint16_t*) input;

  const float16x8_t vscale = vreinterpretq_f16_u16(vld1q_dup_u16(&params->neonfp16arith.scale));
  const int16x8_t voutput_zero_point = vld1q_dup_s16(&params->neonfp16arith.output_zero_point);
  const int8x16_t voutput_min = vld1q_dup_s8(&params->neonfp16arith.output_min);
  const int8x16_t voutput_max = vld1q_dup_s8(&params->neonfp16arith.output_max);
  for (; batch >= 32 * sizeof(uint16_t); batch -= 32 * sizeof(uint16_t)) {
    float16x8_t vx0 = vreinterpretq_f16_u16(vld1q_u16(i)); i += 8;
    float16x8_t vx8 = vreinterpretq_f16_u16(vld1q_u16(i)); i += 8;
    float16x8_t vx16 = vreinterpretq_f16_u16(vld1q_u16(i)); i += 8;
    float16x8_t vx24 = vreinterpretq_f16_u16(vld1q_u16(i)); i += 8;

    vx0 = vmulq_f16(vx0, vscale);
    vx8 = vmulq_f16(vx8, vscale);
    vx16 = vmulq_f16(vx16, vscale);
    vx24 = vmulq_f16(vx24, vscale);

    int16x8_t vacc0 = vcvtnq_s16_f16(vx0);
    int16x8_t vacc8 = vcvtnq_s16_f16(vx8);
    int16x8_t vacc16 = vcvtnq_s16_f16(vx16);
    int16x8_t vacc24 = vcvtnq_s16_f16(vx24);

    vacc0 = vqaddq_s16(vacc0, voutput_zero_point);
    vacc8 = vqaddq_s16(vacc8, voutput_zero_point);
    vacc16 = vqaddq_s16(vacc16, voutput_zero_point);
    vacc24 = vqaddq_s16(vacc24, voutput_zero_point);

    int8x16_t vy0 = vcombine_s8(vqmovn_s16(vacc0), vqmovn_s16(vacc8));
    int8x16_t vy16 = vcombine_s8(vqmovn_s16(vacc16), vqmovn_s16(vacc24));

    vy0 = vmaxq_s8(vy0, voutput_min);
    vy16 = vmaxq_s8(vy16, voutput_min);

    vy0 = vminq_s8(vy0, voutput_max);
    vy16 = vminq_s8(vy16, voutput_max);

    vst1q_s8(output, vy0); output += 16;
    vst1q_s8(output, vy16); output += 16;
  }
  for (; batch >= 8 * sizeof(uint16_t); batch -= 8 * sizeof(uint16_t)) {
    float16x8_t vx = vreinterpretq_f16_u16(vld1q_u16(i)); i += 8;

    vx = vmulq_f16(vx, vscale);

    int16x8_t vacc = vcvtnq_s16_f16(vx);

    vacc = vqaddq_s16(vacc, voutput_zero_point);

    int8x8_t vy = vqmovn_s16(vacc);
    vy = vmax_s8(vy, vget_low_s8(voutput_min));
    vy = vmin_s8(vy, vget_low_s8(voutput_max));
    vst1_s8(output, vy); output += 8;
  }
  if XNN_UNLIKELY(batch != 0) {
    assert(batch >= 1 * sizeof(uint16_t));
    assert(batch <= 7 * sizeof(uint16_t));
    float16x8_t vx = vreinterpretq_f16_u16(vld1q_u16(i));

    vx = vmulq_f16(vx, vscale);

    int16x8_t vacc = vcvtnq_s16_f16(vx);
    vacc = vqaddq_s16(vacc, voutput_zero_point);

    int8x8_t vy = vqmovn_s16(vacc);
    vy = vmax_s8(vy, vget_low_s8(voutput_min));
    vy = vmin_s8(vy, vget_low_s8(voutput_max));

    if (batch & (4 * sizeof(uint16_t))) {
      vst1_lane_u32((void*) output, vreinterpret_u32_s8(vy), 0); output += 4;
      vy = vext_s8(vy, vy, 4);
    }
    if (batch & (2 * sizeof(uint16_t))) {
      vst1_lane_u16((void*) output, vreinterpret_u16_s8(vy), 0); output += 2;
      vy = vext_s8(vy, vy, 2);
    }
    if (batch & (1 * sizeof(uint16_t))) {
      vst1_lane_s8(output, vy, 0);
    }
  }
}
