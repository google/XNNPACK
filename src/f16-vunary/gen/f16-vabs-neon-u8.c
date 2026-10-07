// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/f16-vunary/neon.c.in
//   Generator: tools/xngen
//
// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <arm_neon.h>
#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include "src/xnnpack/common.h"
#include "src/xnnpack/math.h"
#include "src/xnnpack/microparams.h"
#include "src/xnnpack/vunary.h"


void xnn_f16_vabs_ukernel__neon_u8(
    size_t batch,
    const xnn_float16* input,
    xnn_float16* output,
    const struct xnn_f16_default_params* restrict params) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(uint16_t) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const uint16_t* i = (const uint16_t*) input;
  uint16_t* o = (uint16_t*) output;
  const uint16x8_t vnonsign_mask = vdupq_n_u16(0x7FFF);
  for (; batch >= 8 * sizeof(uint16_t); batch -= 8 * sizeof(uint16_t)) {
    uint16x8_t vacc = vld1q_u16(i); i += 8;
    vacc = vandq_u16(vacc, vnonsign_mask);
    vst1q_u16(o, vacc); o += 8;
  }
  if XNN_UNLIKELY(batch != 0) {
    uint16x8_t vacc = vld1q_u16(i);
    vacc = vandq_u16(vacc, vnonsign_mask);
    uint16x4_t vacc_lo = vget_low_u16(vacc);
    if (batch & (4 * sizeof(uint16_t))) {
      vst1_u16(o, vacc_lo); o += 4;
      vacc_lo = vget_high_u16(vacc);
    }
    if (batch & (2 * sizeof(uint16_t))) {
      vst1_lane_u32((void*) o, vreinterpret_u32_u16(vacc_lo), 0); o += 2;
      vacc_lo = vext_u16(vacc_lo, vacc_lo, 2);
    }
    if (batch & (1 * sizeof(uint16_t))) {
      vst1_lane_u16(o, vacc_lo, 0);
    }
  }
}
