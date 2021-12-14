// Copyright 2021 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <arm_neon.h>

#include <xnnpack/vunary.h>


void xnn_u8_vclamp_ukernel__neon_x64(
    size_t n,
    const uint8_t* x,
    uint8_t* y,
    const union xnn_u8_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(n != 0);

  const uint8x16_t voutput_max = vld1q_dup_u8(&params->neon.max);
  const uint8x16_t voutput_min = vld1q_dup_u8(&params->neon.min);

  for (; n >= 64; n -= 64) {
    uint8x16_t vacc0 = vld1q_u8(x); x += 16;
    uint8x16_t vacc1 = vld1q_u8(x); x += 16;
    uint8x16_t vacc2 = vld1q_u8(x); x += 16;
    uint8x16_t vacc3 = vld1q_u8(x); x += 16;

    vacc0 = vmaxq_u8(vacc0, voutput_min);
    vacc1 = vmaxq_u8(vacc1, voutput_min);
    vacc2 = vmaxq_u8(vacc2, voutput_min);
    vacc3 = vmaxq_u8(vacc3, voutput_min);

    vacc0 = vminq_u8(vacc0, voutput_max);
    vacc1 = vminq_u8(vacc1, voutput_max);
    vacc2 = vminq_u8(vacc2, voutput_max);
    vacc3 = vminq_u8(vacc3, voutput_max);

    vst1q_u8(y, vacc0); y += 16;
    vst1q_u8(y, vacc1); y += 16;
    vst1q_u8(y, vacc2); y += 16;
    vst1q_u8(y, vacc3); y += 16;
  }
  for (; n >= 8; n -= 8) {
    uint8x8_t vacc = vld1_u8(x); x += 8;

    vacc = vmin_u8(vacc, vget_low_u8(voutput_max));
    vacc = vmax_u8(vacc, vget_low_u8(voutput_min));

    vst1_u8(y, vacc); y += 8;
  }
  if XNN_UNLIKELY(n != 0) {
    uint8x8_t vacc = vld1_u8(x); x += 8;

    vacc = vmin_u8(vacc, vget_low_u8(voutput_max));
    vacc = vmax_u8(vacc, vget_low_u8(voutput_min));

    if (n & 4) {
      vst1_lane_u32((void*) y, vreinterpret_u32_u8(vacc), 0); y += 4;
      vacc = vext_u8(vacc, vacc, 4);
    }
    if (n & 2) {
      vst1_lane_u16((void*) y, vreinterpret_u16_u8(vacc), 0); y += 2;
      vacc = vext_u8(vacc, vacc, 2);
    }
    if (n & 1) {
      vst1_lane_u8(y, vacc, 0);
    }
  }
}
