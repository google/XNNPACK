// Auto-generated file. Do not edit!
//   Template: src/f16-prelu/neonfp16arith.c.in
//   Generator: tools/xngen
//
// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <arm_neon.h>

#include "xnnpack/math.h"
#include "xnnpack/prelu.h"


void xnn_f16_prelu_ukernel__neonfp16arith_2x8(
    size_t rows,
    size_t channels,
    const void* restrict input,
    size_t input_stride,
    const void* restrict weights,
    void* restrict output,
    size_t output_stride) XNN_OOB_READS
{
  assert(rows != 0);
  assert(channels != 0);
  assert(channels % sizeof(uint16_t) == 0);

  const uint16_t* i0 = (const uint16_t*) input;
  uint16_t* o0 = (uint16_t*) output;
  const uint16_t* i1 = (const uint16_t*) ((uintptr_t) i0 + input_stride);
  uint16_t* o1 = (uint16_t*) ((uintptr_t) o0 + output_stride);

  const size_t input_increment = input_stride * 2 - channels;
  const size_t output_increment = output_stride * 2 - channels;

  do {
    if XNN_UNPREDICTABLE(rows < 2) {
      i1 = i0;
      o1 = o0;
    }

    const uint16_t* w = (const uint16_t*) weights;
    size_t c = channels;
    for (; c >= 8 * sizeof(uint16_t); c -= 8 * sizeof(uint16_t)) {
      const float16x8_t vw01234567 = vreinterpretq_f16_u16(vld1q_u16(w)); w += 8;

      const float16x8_t vi0x01234567 = vreinterpretq_f16_u16(vld1q_u16(i0));
      i0 += 8;
      const float16x8_t vi1x01234567 = vreinterpretq_f16_u16(vld1q_u16(i1));
      i1 += 8;

      float16x8_t vacc0x01234567 = vmulq_f16(vi0x01234567, vw01234567);
      const uint16x8_t vm0x01234567 = vcltq_s16(vreinterpretq_s16_f16(vi0x01234567), vmovq_n_s16(0));
      float16x8_t vacc1x01234567 = vmulq_f16(vi1x01234567, vw01234567);
      const uint16x8_t vm1x01234567 = vcltq_s16(vreinterpretq_s16_f16(vi1x01234567), vmovq_n_s16(0));

      vacc0x01234567 = vbslq_f16(vm0x01234567, vacc0x01234567, vi0x01234567);
      vacc1x01234567 = vbslq_f16(vm1x01234567, vacc1x01234567, vi1x01234567);

      vst1q_u16(o0, vreinterpretq_u16_f16(vacc0x01234567)); o0 += 8;
      vst1q_u16(o1, vreinterpretq_u16_f16(vacc1x01234567)); o1 += 8;
    }
    if XNN_UNLIKELY(c != 0) {
      const float16x8_t vw01234567 = vreinterpretq_f16_u16(vld1q_u16(w));

      const float16x8_t vi0x01234567 = vreinterpretq_f16_u16(vld1q_u16(i0));
      i0 = (const uint16_t*) ((uintptr_t) i0 + c);
      const float16x8_t vi1x01234567 = vreinterpretq_f16_u16(vld1q_u16(i1));
      i1 = (const uint16_t*) ((uintptr_t) i1 + c);

      float16x8_t vacc0x01234567 = vmulq_f16(vi0x01234567, vw01234567);
      const uint16x8_t vm0x01234567 = vcltq_s16(vreinterpretq_s16_f16(vi0x01234567), vmovq_n_s16(0));
      float16x8_t vacc1x01234567 = vmulq_f16(vi1x01234567, vw01234567);
      const uint16x8_t vm1x01234567 = vcltq_s16(vreinterpretq_s16_f16(vi1x01234567), vmovq_n_s16(0));

      vacc0x01234567 = vbslq_f16(vm0x01234567, vacc0x01234567, vi0x01234567);
      vacc1x01234567 = vbslq_f16(vm1x01234567, vacc1x01234567, vi1x01234567);

      float16x4_t vacc0x0123 = vget_low_f16(vacc0x01234567);
      float16x4_t vacc1x0123 = vget_low_f16(vacc1x01234567);
      if (c & (4 * sizeof(uint16_t))) {
        vst1_u16(o0, vreinterpret_u16_f16(vacc0x0123)); o0 += 4;
        vst1_u16(o1, vreinterpret_u16_f16(vacc1x0123)); o1 += 4;

        vacc0x0123 = vget_high_f16(vacc0x01234567);
        vacc1x0123 = vget_high_f16(vacc1x01234567);
      }
      if (c & (2 * sizeof(uint16_t))) {
        vst1_lane_u32((void*) o0, vreinterpret_u32_f16(vacc0x0123), 0); o0 += 2;
        vacc0x0123 = vext_f16(vacc0x0123, vacc0x0123, 2);
        vst1_lane_u32((void*) o1, vreinterpret_u32_f16(vacc1x0123), 0); o1 += 2;
        vacc1x0123 = vext_f16(vacc1x0123, vacc1x0123, 2);
      }
      if (c & (1 * sizeof(uint16_t))) {
        vst1_lane_u16(o0, vreinterpret_u16_f16(vacc0x0123), 0); o0 += 1;
        vst1_lane_u16(o1, vreinterpret_u16_f16(vacc1x0123), 0); o1 += 1;
      }
    }
    i0 = (const uint16_t*) ((uintptr_t) i0 + input_increment);
    o0 = (uint16_t*) ((uintptr_t) o0 + output_increment);
    i1 = (const uint16_t*) ((uintptr_t) i1 + input_increment);
    o1 = (uint16_t*) ((uintptr_t) o1 + output_increment);
    rows = doz(rows, 2);
  } while (rows != 0);
}
