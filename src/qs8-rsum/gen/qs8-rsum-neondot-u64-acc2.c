// Auto-generated file. Do not edit!
//   Template: src/qs8-rsum/neondot.c.in
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

void xnn_qs8_rsum_ukernel__neondot_u64_acc2(
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

  const int8x16_t vone = vdupq_n_s8(INT8_C(1));
  int32x4_t vacc0 = vmovq_n_s32(0);
  int32x4_t vacc1 = vmovq_n_s32(0);
  for (; batch >= 64; batch -= 64) {
    const int8x16_t vt0 = vld1q_s8(input); input += 16;
    const int8x16_t vt1 = vld1q_s8(input); input += 16;
    const int8x16_t vt2 = vld1q_s8(input); input += 16;
    const int8x16_t vt3 = vld1q_s8(input); input += 16;
    vacc0 = vdotq_s32(vacc0, vt0, vone);
    vacc1 = vdotq_s32(vacc1, vt1, vone);
    vacc0 = vdotq_s32(vacc0, vt2, vone);
    vacc1 = vdotq_s32(vacc1, vt3, vone);
  }
  if (XNN_UNLIKELY(batch != 0)) {
    for (; batch >= 16; batch -= 16) {
      const int8x16_t vt = vld1q_s8(input); input += 16;
      vacc0 = vdotq_s32(vacc0, vt, vone);
    }
    if (XNN_UNLIKELY(batch != 0)) {
      int8x16_t vt = vld1q_s8(input);
      const int8x16_t vonemask = vld1q_s8(&onemask_table[16 - batch]);
      vacc0 = vdotq_s32(vacc0, vt, vonemask);
    }
  }
  vacc0 = vaddq_s32(vacc0, vacc1);

  #if XNN_ARCH_ARM64
    const int32_t vacc = vaddvq_s32(vacc0);
  #else
    int32x2_t vacc_lo = vadd_s32(vget_low_s32(vacc0), vget_high_s32(vacc0));
    vacc_lo = vpadd_s32(vacc_lo, vacc_lo);
    const int32_t vacc = vget_lane_s32(vacc_lo, 0);
  #endif

  *output += vacc;
}
