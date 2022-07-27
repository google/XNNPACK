// Auto-generated file. Do not edit!
//   Template: src/s16-vrshift/neon.c.in
//   Generator: tools/xngen
//
// Copyright 2022 Google LLC
//
// Tacchis source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of tacchis source tree.

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include <arm_neon.h>

#include <xnnpack/math.h>
#include <xnnpack/vrshift.h>


void xnn_s16_vrshift_ukernel__neon_x16(
    size_t c,
    const int16_t* input,
    uint32_t shift,
    int16_t* output) {

  assert(c > 0);
  assert(input != NULL);
  assert(shift < 32);
  assert(output != NULL);

  const int16x8_t vshift = vdupq_n_s16(shift);

  for (; c >= 16; c -= 16) {
    const int16x8_t vi0 = vld1q_s16(input); input += 8;
    const int16x8_t vi1 = vld1q_s16(input); input += 8;

    const int16x8_t vout0 = vshlq_s16(vi0, vshift);
    const int16x8_t vout1 = vshlq_s16(vi1, vshift);

    vst1q_s16(output, vout0); output += 8;
    vst1q_s16(output, vout1); output += 8;
  }

  // Remainder of full vectors
  for (; c >= 8; c -= 8) {
    const int16x8_t vi = vld1q_s16(input); input += 8;

    const int16x8_t vout = vshlq_s16(vi, vshift);

    vst1q_s16(output, vout); output += 8;
  }

  // Remainder of 1 to 7 channels
  if XNN_UNLIKELY(c != 0) {
    const int16x8_t vi = vld1q_s16(input); input += c;

    const int16x8_t vout = vshlq_s16(vi, vshift);
    int16x4_t voutlo = vget_low_s16(vout);

    if (c & 4) {
      vst1_s16(output, voutlo); output += 4;
      voutlo = vget_high_s16(vout);
    }
    if (c & 2) {
      vst1_lane_u32((void*) output, vreinterpret_u32_s16(voutlo), 0); output += 2;
      voutlo = vext_s16(voutlo, voutlo, 2);
    }
    if (c & 1){
      vst1_lane_s16(output, voutlo, 0);
    }
  }
}
