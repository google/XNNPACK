// Auto-generated file. Do not edit!
//   Template: src/f16-vsqrt/neonfp16arith-sqrt.c.in
//   Generator: tools/xngen
//
// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <arm_neon.h>

#include "xnnpack/common.h"
#include "xnnpack/vunary.h"


void xnn_f16_vsqrt_ukernel__aarch64_neonfp16arith_sqrt_u8(
    size_t batch,
    const void* input,
    void* output,
    const union xnn_f16_sqrt_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(uint16_t) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const uint16_t* i = (const uint16_t*) input;
  uint16_t* o = (uint16_t*) output;
  for (; batch >= 8 * sizeof(uint16_t); batch -= 8 * sizeof(uint16_t)) {
    float16x8_t vacc = vreinterpretq_f16_u16(vld1q_u16(i)); i += 8;
    vacc = vsqrtq_f16(vacc);
    vst1q_u16(o, vreinterpretq_u16_f16(vacc)); o += 8;
  }
  if XNN_UNLIKELY(batch != 0) {
    const float16x8_t vacc = vreinterpretq_f16_u16(vld1q_u16(i));
    float16x4_t vacc_lo = vsqrt_f16(vget_low_f16(vacc));
    if (batch & (4 * sizeof(uint16_t))) {
      vst1_u16(o, vreinterpret_u16_f16(vacc_lo)); o += 4;
      vacc_lo = vsqrt_f16(vget_high_f16(vacc));
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
