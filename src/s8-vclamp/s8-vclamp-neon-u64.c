// Copyright 2021 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <arm_neon.h>

#include <xnnpack/vunary.h>


void xnn_s8_vclamp_ukernel__neon_u64(
    size_t batch,
    const int8_t* input,
    int8_t* output,
    const union xnn_s8_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(int8_t) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const int8x16_t voutput_max = vld1q_dup_s8(&params->neon.max);
  const int8x16_t voutput_min = vld1q_dup_s8(&params->neon.min);

  for (; batch >= 64; batch -= 64) {
    int8x16_t vacc0 = vld1q_s8(input); input += 16;
    int8x16_t vacc1 = vld1q_s8(input); input += 16;
    int8x16_t vacc2 = vld1q_s8(input); input += 16;
    int8x16_t vacc3 = vld1q_s8(input); input += 16;

    vacc0 = vmaxq_s8(vacc0, voutput_min);
    vacc1 = vmaxq_s8(vacc1, voutput_min);
    vacc2 = vmaxq_s8(vacc2, voutput_min);
    vacc3 = vmaxq_s8(vacc3, voutput_min);

    vacc0 = vminq_s8(vacc0, voutput_max);
    vacc1 = vminq_s8(vacc1, voutput_max);
    vacc2 = vminq_s8(vacc2, voutput_max);
    vacc3 = vminq_s8(vacc3, voutput_max);

    vst1q_s8(output, vacc0); output += 16;
    vst1q_s8(output, vacc1); output += 16;
    vst1q_s8(output, vacc2); output += 16;
    vst1q_s8(output, vacc3); output += 16;
  }
  for (; batch >= 8; batch -= 8) {
    int8x8_t vacc = vld1_s8(input); input += 8;

    vacc = vmin_s8(vacc, vget_low_s8(voutput_max));
    vacc = vmax_s8(vacc, vget_low_s8(voutput_min));

    vst1_s8(output, vacc); output += 8;
  }
  if XNN_UNLIKELY(batch != 0) {
    int8x8_t vacc = vld1_s8(input); input += 8;

    vacc = vmin_s8(vacc, vget_low_s8(voutput_max));
    vacc = vmax_s8(vacc, vget_low_s8(voutput_min));

    if (batch & 4) {
      vst1_lane_u32((void*) output, vreinterpret_u32_s8(vacc), 0); output += 4;
      vacc = vext_s8(vacc, vacc, 4);
    }
    if (batch & 2) {
      vst1_lane_u16((void*) output, vreinterpret_u16_s8(vacc), 0); output += 2;
      vacc = vext_s8(vacc, vacc, 2);
    }
    if (batch & 1) {
      vst1_lane_s8(output, vacc, 0);
    }
  }
}
