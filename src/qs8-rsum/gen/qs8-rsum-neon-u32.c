// Auto-generated file. Do not edit!
//   Template: src/qs8-rsum/neon.c.in
//   Generator: tools/xngen
//
// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.


#include <assert.h>

#include <arm_neon.h>

#include "xnnpack/common.h"
#include "xnnpack/reduce.h"

void xnn_qs8_rsum_ukernel__neon_u32(
    size_t batch,
    const int8_t* input,
    int32_t* output,
    const union xnn_qs8_rsum_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(input != NULL);
  assert(output != NULL);
  assert(params != NULL);

  XNN_ALIGN(16) static const int8_t onemask_table[32] = {
    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
    0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
  };

  int32x4_t vacc0 = vmovq_n_s32(0);

  // 256 int8s may be summed into an int16 before overflowing.
  // Each register has 8 lanes and there are 1 accumulators so batch size is 2048

  for (; batch >= 2048; batch -= 2048) {
    int16x8_t vacc16_0 = vmovq_n_s16(0);
    for (size_t current_batch = 2048; current_batch > 0; current_batch -= 32) {
      const int8x16_t vt0 = vld1q_s8(input); input += 16;
      const int8x16_t vt1 = vld1q_s8(input); input += 16;
      vacc16_0 = vpadalq_s8(vacc16_0, vt0);
      vacc16_0 = vpadalq_s8(vacc16_0, vt1);
    }
    vacc0 = vpadalq_s16(vacc0, vacc16_0);
  }

  if (XNN_LIKELY(batch >= 32)) {
    assert(batch >= 1 && batch < 2048);
    int16x8_t vacc16_0 = vmovq_n_s16(0);
    for (; batch >= 32; batch -= 32) {
      const int8x16_t vt0 = vld1q_s8(input); input += 16;
      const int8x16_t vt1 = vld1q_s8(input); input += 16;
      vacc16_0 = vpadalq_s8(vacc16_0, vt0);
      vacc16_0 = vpadalq_s8(vacc16_0, vt1);
    }
    vacc0 = vpadalq_s16(vacc0, vacc16_0);
  }
  if (XNN_UNLIKELY(batch != 0)) {
    assert(batch >= 1 && batch < 2048);
    int16x8_t vacc16 = vmovq_n_s16(0);
    for (; batch >= 16; batch -= 16) {
      const int8x16_t vt = vld1q_s8(input); input += 16;
      vacc16 = vpadalq_s8(vacc16, vt);
    }
    if (XNN_UNLIKELY(batch != 0)) {
      const int8x16_t vt = vld1q_s8(input);
      const int8x16_t vonemask = vld1q_s8(&onemask_table[16 - batch]);
      const int8x16_t vtm = vmulq_s8(vt, vonemask);
      vacc16 = vpadalq_s8(vacc16, vtm);
    }
    vacc0 = vpadalq_s16(vacc0, vacc16);
  }
  #if XNN_ARCH_ARM64
    const int32_t vacc = vaddvq_s32(vacc0);
  #else
    int32x2_t vacc_lo = vadd_s32(vget_low_s32(vacc0), vget_high_s32(vacc0));
    vacc_lo = vpadd_s32(vacc_lo, vacc_lo);
    const int32_t vacc = vget_lane_s32(vacc_lo, 0);
  #endif

  *output += vacc;
}
