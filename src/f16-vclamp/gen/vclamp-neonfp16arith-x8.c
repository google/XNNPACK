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


void xnn_f16_vclamp_ukernel__neonfp16arith_x8(
    size_t batch,
    const void* restrict input,
    void* restrict output,
    const union xnn_f16_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(__fp16) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const __fp16* i = (const __fp16*) input;
  __fp16* o = (__fp16*) output;

  const float16x8_t vy_min = vreinterpretq_f16_u16(vld1q_dup_u16(&params->fp16arith.min));
  const float16x8_t vy_max = vreinterpretq_f16_u16(vld1q_dup_u16(&params->fp16arith.max));

  for (; batch >= 8 * sizeof(__fp16); batch -= 8 * sizeof(__fp16)) {
    float16x8_t vacc = vld1q_f16(i); i += 8;
    vacc = vmaxq_f16(vacc, vy_min);
    vacc = vminq_f16(vacc, vy_max);
    vst1q_f16(o, vacc); o += 8;
  }
  if XNN_UNLIKELY(batch != 0) {
    float16x8_t vacc = vld1q_f16(i);
    vacc = vmaxq_f16(vacc, vy_min);
    vacc = vminq_f16(vacc, vy_max);

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
