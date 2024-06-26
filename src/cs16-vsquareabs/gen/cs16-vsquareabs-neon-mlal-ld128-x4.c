// Auto-generated file. Do not edit!
//   Template: src/cs16-vsquareabs/neon.c.in
//   Generator: tools/xngen
//
// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include <arm_neon.h>

#include "xnnpack/vsquareabs.h"


void xnn_cs16_vsquareabs_ukernel__neon_mlal_ld128_x4(
    size_t batch,
    const int16_t* input,
    uint32_t* output) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % (sizeof(int16_t) * 2) == 0);
  assert(input != NULL);
  assert(output != NULL);

  for (; batch >= 8 * sizeof(int16_t); batch -= 8 * sizeof(int16_t)) {
    const int16x4x2_t vi = vld2_s16(input); input += 8;
    int32x4_t vacc = vmull_s16(vi.val[0], vi.val[0]);
    vacc = vmlal_s16(vacc, vi.val[1], vi.val[1]);
    vst1q_u32(output, vreinterpretq_u32_s32(vacc)); output += 4;
  }
  if XNN_LIKELY(batch != 0) {
    const int16x4x2_t vi = vld2_s16(input);
    int32x4_t vacc = vmull_s16(vi.val[0], vi.val[0]);
    vacc = vmlal_s16(vacc, vi.val[1], vi.val[1]);
    uint32x2_t vacc_lo = vreinterpret_u32_s32(vget_low_s32(vacc));
    if (batch & (4 * sizeof(int16_t))) {
      vst1_u32(output, vacc_lo); output += 2;
      vacc_lo = vreinterpret_u32_s32(vget_high_s32(vacc));
    }
    if (batch & (2 * sizeof(int16_t))) {
      vst1_lane_u32(output, vacc_lo, 0);
    }
  }
}
