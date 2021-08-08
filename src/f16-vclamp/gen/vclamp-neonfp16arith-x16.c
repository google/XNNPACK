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
    size_t n,
    const void* restrict x_ptr,
    void* restrict y_ptr,
    const struct xnn_f16_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_DISABLE_TSAN
{
  assert(n != 0);
  assert(n % sizeof(__fp16) == 0);
  assert(x_ptr != NULL);
  assert(y_ptr != NULL);

  const __fp16* x = (const __fp16*) x_ptr;
  __fp16* y = (__fp16*) y_ptr;

  const float16x8_t vy_min = vreinterpretq_f16_u16(vld1q_dup_u16(&params->min));
  const float16x8_t vy_max = vreinterpretq_f16_u16(vld1q_dup_u16(&params->max));

  for (; n >= 16 * sizeof(__fp16); n -= 16 * sizeof(__fp16)) {
    float16x8_t vacc01234567 = vld1q_f16(x); x += 8;
    float16x8_t vacc89ABCDEF = vld1q_f16(x); x += 8;

    vacc01234567 = vmaxq_f16(vacc01234567, vy_min);
    vacc89ABCDEF = vmaxq_f16(vacc89ABCDEF, vy_min);

    vacc01234567 = vminq_f16(vacc01234567, vy_max);
    vacc89ABCDEF = vminq_f16(vacc89ABCDEF, vy_max);

    vst1q_f16(y, vacc01234567); y += 8;
    vst1q_f16(y, vacc89ABCDEF); y += 8;
  }
  for (; n >= 8 * sizeof(__fp16); n -= 8 * sizeof(__fp16)) {
    float16x8_t vacc = vld1q_f16(x); x += 8;
    vacc = vmaxq_f16(vacc, vy_min);
    vacc = vminq_f16(vacc, vy_max);
    vst1q_f16(y, vacc); y += 8;
  }
  if XNN_UNLIKELY(n != 0) {
    float16x8_t vacc = vld1q_f16(x);
    vacc = vmaxq_f16(vacc, vy_min);
    vacc = vminq_f16(vacc, vy_max);

    float16x4_t vacc_lo = vget_low_f16(vacc);
    if (n & (4 * sizeof(__fp16))) {
      vst1_f16(y, vacc_lo); y += 4;
      vacc_lo = vget_high_f16(vacc);
    }
    if (n & (2 * sizeof(__fp16))) {
      vst1_lane_u32(__builtin_assume_aligned(y, 1), vreinterpret_u32_f16(vacc_lo), 0); y += 2;
      vacc_lo = vext_f16(vacc_lo, vacc_lo, 2);
    }
    if (n & (1 * sizeof(__fp16))) {
      vst1_lane_f16(y, vacc_lo, 0);
    }
  }
}
