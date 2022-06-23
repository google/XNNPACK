// Auto-generated file. Do not edit!
//   Template: src/qs8-vlrelu/neon.c.in
//   Generator: tools/xngen
//
// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <arm_neon.h>

#include <xnnpack/common.h>
#include <xnnpack/vlrelu.h>


void xnn_qu8_vlrelu_ukernel__neon_x8(
    size_t n,
    const uint8_t* x,
    uint8_t* y,
    const union xnn_qu8_lrelu_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(n != 0);
  assert(n % sizeof(uint8_t) == 0);
  assert(x != NULL);
  assert(y != NULL);

  const uint16x8_t vinput_zero_point = vld1q_dup_u16(&params->neon.input_zero_point);
  const int16x8_t vpositive_multiplier = vld1q_dup_s16(&params->neon.positive_multiplier);
  const int16x8_t vnegative_multiplier = vld1q_dup_s16(&params->neon.negative_multiplier);
  const int16x8_t voutput_zero_point = vld1q_dup_s16(&params->neon.output_zero_point);
  for (; n >= 8 * sizeof(uint8_t); n -= 8 * sizeof(uint8_t)) {
    const uint8x8_t vx = vld1_u8(x); x += 8;
    int16x8_t vacc = vreinterpretq_s16_u16(vsubw_u8(vinput_zero_point, vx));
    const uint16x8_t vmask = vcltq_s16(vacc, vmovq_n_s16(0));
    vacc = vshlq_n_s16(vacc, 7);
    const int16x8_t vmultiplier = vbslq_s16(vmask, vpositive_multiplier, vnegative_multiplier);
    vacc = vqrdmulhq_s16(vacc, vmultiplier);
    vacc = vqaddq_s16(vacc, voutput_zero_point);
    const uint8x8_t vy = vqmovun_s16(vacc);
    vst1_u8(y, vy); y += 8;
  }
  if XNN_UNLIKELY(n != 0) {
    assert(n >= 1 * sizeof(uint8_t));
    assert(n <= 7 * sizeof(uint8_t));

    const uint8x8_t vx = vld1_u8(x);
    int16x8_t vacc = vreinterpretq_s16_u16(vsubw_u8(vinput_zero_point, vx));
    const uint16x8_t vmask = vcltq_s16(vacc, vmovq_n_s16(0));
    vacc = vshlq_n_s16(vacc, 7);
    const int16x8_t vmultiplier = vbslq_s16(vmask, vpositive_multiplier, vnegative_multiplier);
    vacc = vqrdmulhq_s16(vacc, vmultiplier);
    vacc = vqaddq_s16(vacc, voutput_zero_point);
    uint8x8_t vy = vqmovun_s16(vacc);

    if (n & (4 * sizeof(uint8_t))) {
      vst1_lane_u32((void*) y, vreinterpret_u32_u8(vy), 0); y += 4;
      vy = vext_u8(vy, vy, 4);
    }
    if (n & (2 * sizeof(uint8_t))) {
      vst1_lane_u16((void*) y, vreinterpret_u16_u8(vy), 0); y += 2;
      vy = vext_u8(vy, vy, 2);
    }
    if (n & (1 * sizeof(uint8_t))) {
      vst1_lane_u8(y, vy, 0);
    }
  }
}
