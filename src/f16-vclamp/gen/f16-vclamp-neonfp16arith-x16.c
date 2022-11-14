// Auto-generated file. Do not edit!
//   Template: src/f16-vclamp/neonfp16arith.c.in
//   Generator: tools/xngen
//
// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <arm_neon.h>

#include <xnnpack/common.h>
#include <xnnpack/vunary.h>


void xnn_f16_vclamp_ukernel__neonfp16arith_x16(
    size_t batch,
    const void* restrict input,
    void* restrict output,
    const union xnn_f16_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(uint16_t) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const uint16_t* i = (const uint16_t*) input;
  uint16_t* o = (uint16_t*) output;

  const float16x8_t vy_min = vreinterpretq_f16_u16(vld1q_dup_u16(&params->fp16arith.min));
  const float16x8_t vy_max = vreinterpretq_f16_u16(vld1q_dup_u16(&params->fp16arith.max));

  for (; batch >= 16 * sizeof(uint16_t); batch -= 16 * sizeof(uint16_t)) {
    float16x8_t vacc01234567 = vreinterpretq_f16_u16(vld1q_u16(i)); i += 8;
    float16x8_t vacc89ABCDEF = vreinterpretq_f16_u16(vld1q_u16(i)); i += 8;

    vacc01234567 = vmaxq_f16(vacc01234567, vy_min);
    vacc89ABCDEF = vmaxq_f16(vacc89ABCDEF, vy_min);

    vacc01234567 = vminq_f16(vacc01234567, vy_max);
    vacc89ABCDEF = vminq_f16(vacc89ABCDEF, vy_max);

    vst1q_u16(o, vreinterpretq_u16_f16(vacc01234567)); o += 8;
    vst1q_u16(o, vreinterpretq_u16_f16(vacc89ABCDEF)); o += 8;
  }
  for (; batch >= 8 * sizeof(uint16_t); batch -= 8 * sizeof(uint16_t)) {
    float16x8_t vacc = vreinterpretq_f16_u16(vld1q_u16(i)); i += 8;
    vacc = vmaxq_f16(vacc, vy_min);
    vacc = vminq_f16(vacc, vy_max);
    vst1q_u16(o, vreinterpretq_u16_f16(vacc)); o += 8;
  }
  if XNN_UNLIKELY(batch != 0) {
    float16x8_t vacc = vreinterpretq_f16_u16(vld1q_u16(i));
    vacc = vmaxq_f16(vacc, vy_min);
    vacc = vminq_f16(vacc, vy_max);

    float16x4_t vacc_lo = vget_low_f16(vacc);
    if (batch & (4 * sizeof(uint16_t))) {
      vst1_u16(o, vreinterpret_u16_f16(vacc_lo)); o += 4;
      vacc_lo = vget_high_f16(vacc);
    }
    if (batch & (2 * sizeof(uint16_t))) {
      vst1_lane_u32((void*) o, vreinterpret_u32_f16(vacc_lo), 0); o += 2;
      vacc_lo = vext_f16(vacc_lo, vacc_lo, 2);
    }
    if (batch & (1 * sizeof(uint16_t))) {
      vst1_lane_f16(o, vacc_lo, 0);
    }
  }
}
