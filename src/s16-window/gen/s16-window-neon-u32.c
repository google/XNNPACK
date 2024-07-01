// Auto-generated file. Do not edit!
//   Template: src/s16-window/neon.c.in
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

#include "xnnpack/math.h"
#include "xnnpack/window.h"


void xnn_s16_window_ukernel__neon_u32(
    size_t rows,
    size_t channels,
    const int16_t* input,
    const int16_t* weights,
    int16_t* output,
    uint32_t shift) XNN_OOB_READS
{
  assert(rows != 0);
  assert(channels != 0);
  assert(input != NULL);
  assert(weights != NULL);
  assert(output != NULL);
  assert(shift < 32);

  const int32x4_t vshift = vdupq_n_s32(-(int32_t)shift);  // negative to shift right.

  do {
    const int16_t* w = weights;
    size_t c = channels;
    for (; c >= 32 * sizeof(int16_t); c -= 32 * sizeof(int16_t)) {
      const int16x8_t vi0 = vld1q_s16(input); input += 8;
      const int16x8_t vi1 = vld1q_s16(input); input += 8;
      const int16x8_t vi2 = vld1q_s16(input); input += 8;
      const int16x8_t vi3 = vld1q_s16(input); input += 8;

      const int16x8_t vw0 = vld1q_s16(w); w += 8;
      const int16x8_t vw1 = vld1q_s16(w); w += 8;
      const int16x8_t vw2 = vld1q_s16(w); w += 8;
      const int16x8_t vw3 = vld1q_s16(w); w += 8;

      int32x4_t vacc0_lo = vmull_s16(vget_low_s16(vi0), vget_low_s16(vw0));
      int32x4_t vacc0_hi = vmull_s16(vget_high_s16(vi0), vget_high_s16(vw0));
      int32x4_t vacc1_lo = vmull_s16(vget_low_s16(vi1), vget_low_s16(vw1));
      int32x4_t vacc1_hi = vmull_s16(vget_high_s16(vi1), vget_high_s16(vw1));
      int32x4_t vacc2_lo = vmull_s16(vget_low_s16(vi2), vget_low_s16(vw2));
      int32x4_t vacc2_hi = vmull_s16(vget_high_s16(vi2), vget_high_s16(vw2));
      int32x4_t vacc3_lo = vmull_s16(vget_low_s16(vi3), vget_low_s16(vw3));
      int32x4_t vacc3_hi = vmull_s16(vget_high_s16(vi3), vget_high_s16(vw3));

      vacc0_lo = vshlq_s32(vacc0_lo, vshift);
      vacc0_hi = vshlq_s32(vacc0_hi, vshift);
      vacc1_lo = vshlq_s32(vacc1_lo, vshift);
      vacc1_hi = vshlq_s32(vacc1_hi, vshift);
      vacc2_lo = vshlq_s32(vacc2_lo, vshift);
      vacc2_hi = vshlq_s32(vacc2_hi, vshift);
      vacc3_lo = vshlq_s32(vacc3_lo, vshift);
      vacc3_hi = vshlq_s32(vacc3_hi, vshift);

      const int16x8_t vout0 = vcombine_s16(vqmovn_s32(vacc0_lo), vqmovn_s32(vacc0_hi));
      const int16x8_t vout1 = vcombine_s16(vqmovn_s32(vacc1_lo), vqmovn_s32(vacc1_hi));
      const int16x8_t vout2 = vcombine_s16(vqmovn_s32(vacc2_lo), vqmovn_s32(vacc2_hi));
      const int16x8_t vout3 = vcombine_s16(vqmovn_s32(vacc3_lo), vqmovn_s32(vacc3_hi));

      vst1q_s16(output, vout0); output += 8;
      vst1q_s16(output, vout1); output += 8;
      vst1q_s16(output, vout2); output += 8;
      vst1q_s16(output, vout3); output += 8;
    }

    // Remainder of full vectors
    for (; c >= 8 * sizeof(int16_t); c -= 8 * sizeof(int16_t)) {
      const int16x8_t vi = vld1q_s16(input); input += 8;
      const int16x8_t vw = vld1q_s16(w); w += 8;
      int32x4_t vacc_lo = vmull_s16(vget_low_s16(vi), vget_low_s16(vw));
      int32x4_t vacc_hi = vmull_s16(vget_high_s16(vi), vget_high_s16(vw));
      vacc_lo = vshlq_s32(vacc_lo, vshift);
      vacc_hi = vshlq_s32(vacc_hi, vshift);
      const int16x8_t vout = vcombine_s16(vqmovn_s32(vacc_lo), vqmovn_s32(vacc_hi));
      vst1q_s16(output, vout); output += 8;
    }

    assert(c % 2 == 0);
    // Remainder of 1 to 7 channels
    if XNN_UNLIKELY(c != 0) {
      const int16x8_t vi = vld1q_s16(input); input = (const int16_t*) ((uintptr_t) input + c);
      const int16x8_t vw = vld1q_s16(w);
      int32x4_t vacc = vmull_s16(vget_low_s16(vi), vget_low_s16(vw));
      vacc = vshlq_s32(vacc, vshift);
      int16x4_t vout = vqmovn_s32(vacc);
      if (c & (4 * sizeof(int16_t))) {
        vst1_s16(output, vout); output += 4;
        vacc = vmull_s16(vget_high_s16(vi), vget_high_s16(vw));
        vacc = vshlq_s32(vacc, vshift);
        vout = vqmovn_s32(vacc);
      }
      if (c & (2 * sizeof(int16_t))) {
        vst1_lane_u32((void*) output, vreinterpret_u32_s16(vout), 0); output += 2;
        vout = vext_s16(vout, vout, 2);
      }
      if (c & (1 * sizeof(int16_t))) {
        vst1_lane_s16(output, vout, 0); output += 1;
      }
    }

  } while (--rows != 0);
}
