// Auto-generated file. Do not edit!
//   Template: src/f16-vcmul/neon.c.in
//   Generator: tools/xngen
//
// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <arm_neon.h>

#include "xnnpack/common.h"
#include "xnnpack/vbinary.h"


void xnn_f16_vcmul_ukernel__neonfp16arith_u8(
    size_t batch,
    const void* input_a,
    const void* input_b,
    void* output,
    const union xnn_f16_default_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(uint16_t) == 0);
  assert(input_a != NULL);
  assert(input_b != NULL);
  assert(output != NULL);

  const uint16_t* ar = (const uint16_t*) input_a;
  const uint16_t* ai = (const uint16_t*) ((uintptr_t) input_a + batch);
  const uint16_t* br = (const uint16_t*) input_b;
  const uint16_t* bi = (const uint16_t*) ((uintptr_t) input_b + batch);
  uint16_t* or = (uint16_t*) output;
  uint16_t* oi = (uint16_t*) ((uintptr_t) output + batch);
  for (; batch >= 8 * sizeof(uint16_t); batch -= 8 * sizeof(uint16_t)) {
    const float16x8_t var = vreinterpretq_f16_u16(vld1q_u16(ar)); ar += 8;
    const float16x8_t vai = vreinterpretq_f16_u16(vld1q_u16(ai)); ai += 8;
    const float16x8_t vbr = vreinterpretq_f16_u16(vld1q_u16(br)); br += 8;
    const float16x8_t vbi = vreinterpretq_f16_u16(vld1q_u16(bi)); bi += 8;

    float16x8_t vaccr = vmulq_f16(var, vbr);
    float16x8_t vacci = vmulq_f16(var, vbi);

    vaccr = vfmsq_f16(vaccr, vai, vbi);
    vacci = vfmaq_f16(vacci, vai, vbr);

    vst1q_u16(or, vreinterpretq_u16_f16(vaccr)); or += 8;
    vst1q_u16(oi, vreinterpretq_u16_f16(vacci)); oi += 8;
  }
  if XNN_UNLIKELY(batch != 0) {
    const float16x8_t var = vreinterpretq_f16_u16(vld1q_u16(ar)); ar += 8;
    const float16x8_t vai = vreinterpretq_f16_u16(vld1q_u16(ai)); ai += 8;
    const float16x8_t vbr = vreinterpretq_f16_u16(vld1q_u16(br)); br += 8;
    const float16x8_t vbi = vreinterpretq_f16_u16(vld1q_u16(bi)); bi += 8;

    float16x8_t vaccr = vmulq_f16(var, vbr);
    float16x8_t vacci = vmulq_f16(var, vbi);

    vaccr = vfmsq_f16(vaccr, vai, vbi);
    vacci = vfmaq_f16(vacci, vai, vbr);

    float16x4_t vaccr_lo = vget_low_f16(vaccr);
    float16x4_t vacci_lo = vget_low_f16(vacci);
    if (batch & (4 * sizeof(uint16_t))) {
      vst1_u16(or, vreinterpret_u16_f16(vaccr_lo)); or += 4;
      vst1_u16(oi, vreinterpret_u16_f16(vacci_lo)); oi += 4;
      vaccr_lo = vget_high_f16(vaccr);
      vacci_lo = vget_high_f16(vacci);
    }
    if (batch & (2 * sizeof(uint16_t))) {
      vst1_lane_u32((void*) or, vreinterpret_u32_f16(vaccr_lo), 0); or += 2;
      vst1_lane_u32((void*) oi, vreinterpret_u32_f16(vacci_lo), 0); oi += 2;
      vaccr_lo = vext_f16(vaccr_lo, vaccr_lo, 2);
      vacci_lo = vext_f16(vacci_lo, vacci_lo, 2);
    }
    if (batch & (1 * sizeof(uint16_t))) {
      vst1_lane_u16(or, vreinterpret_u16_f16(vaccr_lo), 0);
      vst1_lane_u16(oi, vreinterpret_u16_f16(vacci_lo), 0);
    }
  }
}
