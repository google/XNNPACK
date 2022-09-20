// Auto-generated file. Do not edit!
//   Template: src/f16-vlrelu/neonfp16arith.c.in
//   Generator: tools/xngen
//
// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <arm_neon.h>

#include <xnnpack/common.h>
#include <xnnpack/vunary.h>


void xnn_f16_vlrelu_ukernel__neonfp16arith_x16(
    size_t batch,
    const void* input,
    void* output,
    const union xnn_f16_lrelu_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(__fp16) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const float16x8_t vslope = vreinterpretq_f16_u16(vld1q_dup_u16(&params->fp16arith.slope));
  const __fp16* i = (const __fp16*) input;
  __fp16* o = (__fp16*) output;
  for (; batch >= 16 * sizeof(__fp16); batch -= 16 * sizeof(__fp16)) {
    const float16x8_t vx01234567 = vld1q_f16(i); i += 8;
    const float16x8_t vx89ABCDEF = vld1q_f16(i); i += 8;

    float16x8_t vacc01234567 = vmulq_f16(vx01234567, vslope);
    const uint16x8_t vmask01234567 = vcltq_s16(vreinterpretq_s16_f16(vx01234567), vmovq_n_s16(0));
    float16x8_t vacc89ABCDEF = vmulq_f16(vx89ABCDEF, vslope);
    const uint16x8_t vmask89ABCDEF = vcltq_s16(vreinterpretq_s16_f16(vx89ABCDEF), vmovq_n_s16(0));

    vacc01234567 = vbslq_f16(vmask01234567, vacc01234567, vx01234567);
    vacc89ABCDEF = vbslq_f16(vmask89ABCDEF, vacc89ABCDEF, vx89ABCDEF);

    vst1q_f16(o, vacc01234567); o += 8;
    vst1q_f16(o, vacc89ABCDEF); o += 8;
  }
  for (; batch >= 8 * sizeof(__fp16); batch -= 8 * sizeof(__fp16)) {
    const float16x8_t vx = vld1q_f16(i); i += 8;
    float16x8_t vacc = vmulq_f16(vx, vslope);
    const uint16x8_t vmask = vcltq_s16(vreinterpretq_s16_f16(vx), vmovq_n_s16(0));
    vacc = vbslq_f16(vmask, vacc, vx);
    vst1q_f16(o, vacc); o += 8;
  }
  if XNN_UNLIKELY(batch != 0) {
    const float16x8_t vx = vld1q_f16(i);
    float16x8_t vacc = vmulq_f16(vx, vslope);
    const uint16x8_t vmask = vcltq_s16(vreinterpretq_s16_f16(vx), vmovq_n_s16(0));
    vacc = vbslq_f16(vmask, vacc, vx);

    float16x4_t vacc_lo = vget_low_f16(vacc);
    if (batch & (4 * sizeof(__fp16))) {
      vst1_f16(o, vacc_lo); o += 4;
      vacc_lo = vget_high_f16(vacc);
    }
    if (batch & (2 * sizeof(__fp16))) {
      vst1_lane_u32((void*) o, vreinterpret_u32_f16(vacc_lo), 0); o += 2;
      vacc_lo = vext_f16(vacc_lo, vacc_lo, 2);
    }
    if (batch & (1 * sizeof(__fp16))) {
      vst1_lane_f16(o, vacc_lo, 0);
    }
  }
}
