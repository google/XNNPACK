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


void xnn_f16_vdiv_minmax_ukernel__aarch64_neonfp16arith_u16(
    size_t batch,
    const void* restrict input_a,
    const void* restrict input_b,
    void* restrict output,
    const union xnn_f16_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(uint16_t) == 0);
  assert(input_a != NULL);
  assert(input_b != NULL);
  assert(output != NULL);

  const uint16_t* a = (const uint16_t*) input_a;
  const uint16_t* b = (const uint16_t*) input_b;
  uint16_t* o = (uint16_t*) output;

  const float16x8_t vy_min = vreinterpretq_f16_u16(vld1q_dup_u16(&params->fp16arith.min));
  const float16x8_t vy_max = vreinterpretq_f16_u16(vld1q_dup_u16(&params->fp16arith.max));

  for (; batch >= 16 * sizeof(uint16_t); batch -= 16 * sizeof(uint16_t)) {
    const float16x8_t va01234567 = vreinterpretq_f16_u16(vld1q_u16(a)); a += 8;
    const float16x8_t vb01234567 = vreinterpretq_f16_u16(vld1q_u16(b)); b += 8;
    const float16x8_t va456789AB = vreinterpretq_f16_u16(vld1q_u16(a)); a += 8;
    const float16x8_t vb456789AB = vreinterpretq_f16_u16(vld1q_u16(b)); b += 8;

    float16x8_t vy01234567 = vdivq_f16(va01234567, vb01234567);
    float16x8_t vy456789AB = vdivq_f16(va456789AB, vb456789AB);


    vy01234567 = vmaxq_f16(vy01234567, vy_min);
    vy456789AB = vmaxq_f16(vy456789AB, vy_min);

    vy01234567 = vminq_f16(vy01234567, vy_max);
    vy456789AB = vminq_f16(vy456789AB, vy_max);

    vst1q_u16(o, vreinterpretq_u16_f16(vy01234567)); o += 8;
    vst1q_u16(o, vreinterpretq_u16_f16(vy456789AB)); o += 8;
  }
  for (; batch >= 8 * sizeof(uint16_t); batch -= 8 * sizeof(uint16_t)) {
    const float16x8_t va01234567 = vreinterpretq_f16_u16(vld1q_u16(a)); a += 8;
    const float16x8_t vb01234567 = vreinterpretq_f16_u16(vld1q_u16(b)); b += 8;

    float16x8_t vy01234567 = vdivq_f16(va01234567, vb01234567);
    vy01234567 = vmaxq_f16(vy01234567, vy_min);
    vy01234567 = vminq_f16(vy01234567, vy_max);
    vst1q_u16(o, vreinterpretq_u16_f16(vy01234567)); o += 8;
  }
  if XNN_UNLIKELY(batch != 0) {
    const float16x8_t va01234567 = vreinterpretq_f16_u16(vld1q_u16(a));
    const float16x8_t vb01234567 = vreinterpretq_f16_u16(vld1q_u16(b));

    float16x8_t vy01234567 = vdivq_f16(va01234567, vb01234567);
    vy01234567 = vmaxq_f16(vy01234567, vy_min);
    vy01234567 = vminq_f16(vy01234567, vy_max);

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
