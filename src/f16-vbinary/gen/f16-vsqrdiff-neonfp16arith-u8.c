// Auto-generated file. Do not edit!
//   Template: src/f16-vbinary/vop-neonfp16arith.c.in
//   Generator: tools/xngen
//
// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <arm_neon.h>

#include "xnnpack/common.h"
#include "xnnpack/vbinary.h"


void xnn_f16_vsqrdiff_ukernel__neonfp16arith_u8(
    size_t batch,
    const void* restrict input_a,
    const void* restrict input_b,
    void* restrict output,
    const union xnn_f16_default_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(uint16_t) == 0);
  assert(input_a != NULL);
  assert(input_b != NULL);
  assert(output != NULL);

  const uint16_t* a = (const uint16_t*) input_a;
  const uint16_t* b = (const uint16_t*) input_b;
  uint16_t* o = (uint16_t*) output;


  for (; batch >= 8 * sizeof(uint16_t); batch -= 8 * sizeof(uint16_t)) {
    const float16x8_t va01234567 = vreinterpretq_f16_u16(vld1q_u16(a)); a += 8;
    const float16x8_t vb01234567 = vreinterpretq_f16_u16(vld1q_u16(b)); b += 8;

    float16x8_t vy01234567 = vsubq_f16(va01234567, vb01234567);
    vy01234567 = vmulq_f16(vy01234567, vy01234567);
    vst1q_u16(o, vreinterpretq_u16_f16(vy01234567)); o += 8;
  }
  if XNN_UNLIKELY(batch != 0) {
    const float16x8_t va01234567 = vreinterpretq_f16_u16(vld1q_u16(a));
    const float16x8_t vb01234567 = vreinterpretq_f16_u16(vld1q_u16(b));

    float16x8_t vy01234567 = vsubq_f16(va01234567, vb01234567);
    vy01234567 = vmulq_f16(vy01234567, vy01234567);

    float16x4_t vy0123 = vget_low_f16(vy01234567);
    if (batch & (4 * sizeof(uint16_t))) {
      vst1_u16(o, vreinterpret_u16_f16(vy0123)); o += 4;
      vy0123 = vget_high_f16(vy01234567);
    }

    if (batch & (2 * sizeof(uint16_t))) {
      vst1_lane_u32((void*) o, vreinterpret_u32_f16(vy0123), 0); o += 2;
      vy0123 = vext_f16(vy0123, vy0123, 2);
    }

    if (batch & (1 * sizeof(uint16_t))) {
      vst1_lane_u16(o, vreinterpret_u16_f16(vy0123), 0);
    }
  }
}
