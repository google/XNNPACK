// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/f16-vrnd/neonfp16arith.c.in
//   Generator: tools/xngen
//
// Copyright 2022 Google LLC
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


void xnn_f16_vrndd_ukernel__neonfp16arith_u16(
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
  for (; batch >= 16 * sizeof(uint16_t); batch -= 16 * sizeof(uint16_t)) {
    float16x8_t vacc0 = vreinterpretq_f16_u16(vld1q_u16(i)); i += 8;
    float16x8_t vacc1 = vreinterpretq_f16_u16(vld1q_u16(i)); i += 8;

    vacc0 = vrndmq_f16(vacc0);
    vacc1 = vrndmq_f16(vacc1);

    vst1q_u16(o, vreinterpretq_u16_f16(vacc0)); o += 8;
    vst1q_u16(o, vreinterpretq_u16_f16(vacc1)); o += 8;
  }
  for (; batch >= 8 * sizeof(uint16_t); batch -= 8 * sizeof(uint16_t)) {
    float16x8_t vacc = vreinterpretq_f16_u16(vld1q_u16(i)); i += 8;
    vacc = vrndmq_f16(vacc);
    vst1q_u16(o, vreinterpretq_u16_f16(vacc)); o += 8;
  }
  if XNN_UNLIKELY(batch != 0) {
    float16x8_t vacc = vreinterpretq_f16_u16(vld1q_u16(i));
    vacc = vrndmq_f16(vacc);
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
      vst1_lane_u16(o, vreinterpret_u16_f16(vacc_lo), 0);
    }
  }
}
