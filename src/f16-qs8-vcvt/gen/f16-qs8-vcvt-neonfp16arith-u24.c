// clang-format off
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

#include "src/xnnpack/common.h"
#include "src/xnnpack/intrinsics-polyfill.h"
#include "src/xnnpack/vcvt.h"


void xnn_f16_qs8_vcvt_ukernel__neonfp16arith_u24(
    size_t batch,
    const xnn_float16* input,
    int8_t* output,
    const struct xnn_f16_qs8_cvt_params* restrict params) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(uint16_t) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const uint16_t* i = (const uint16_t*) input;

  const float16x8_t vscale = vreinterpretq_f16_u16(vdupq_n_u16(*(const uint16_t*) &params->scalar.scale));
  const int16x8_t voutput_zero_point = vdupq_n_s16(params->scalar.output_zero_point);
  for (; batch >= 24 * sizeof(uint16_t); batch -= 24 * sizeof(uint16_t)) {
    float16x8_t vx0 = vreinterpretq_f16_u16(vld1q_u16(i)); i += 8;
    float16x8_t vx8 = vreinterpretq_f16_u16(vld1q_u16(i)); i += 8;
    float16x8_t vx16 = vreinterpretq_f16_u16(vld1q_u16(i)); i += 8;

    vx0 = vmulq_f16(vx0, vscale);
    vx8 = vmulq_f16(vx8, vscale);
    vx16 = vmulq_f16(vx16, vscale);

    int16x8_t vacc0 = vcvtnq_s16_f16(vx0);
    int16x8_t vacc8 = vcvtnq_s16_f16(vx8);
    int16x8_t vacc16 = vcvtnq_s16_f16(vx16);

    vacc0 = vqaddq_s16(vacc0, voutput_zero_point);
    vacc8 = vqaddq_s16(vacc8, voutput_zero_point);
    vacc16 = vqaddq_s16(vacc16, voutput_zero_point);

    int8x16_t vy0 = vcombine_s8(vqmovn_s16(vacc0), vqmovn_s16(vacc8));
    int8x8_t vy16 = vqmovn_s16(vacc16);

    vst1q_s8(output, vy0); output += 16;
    vst1_s8(output, vy16); output += 8;
  }
  for (; batch >= 8 * sizeof(uint16_t); batch -= 8 * sizeof(uint16_t)) {
    float16x8_t vx = vreinterpretq_f16_u16(vld1q_u16(i)); i += 8;

    vx = vmulq_f16(vx, vscale);

    int16x8_t vacc = vcvtnq_s16_f16(vx);

    vacc = vqaddq_s16(vacc, voutput_zero_point);

    int8x8_t vy = vqmovn_s16(vacc);
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
