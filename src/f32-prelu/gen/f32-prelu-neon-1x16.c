// Auto-generated file. Do not edit!
//   Template: src/f32-prelu/neon.c.in
//   Generator: tools/xngen
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <arm_neon.h>

#include "xnnpack/math.h"
#include "xnnpack/prelu.h"


void xnn_f32_prelu_ukernel__neon_1x16(
    size_t rows,
    size_t channels,
    const float* restrict input,
    size_t input_stride,
    const float* restrict weights,
    float* restrict output,
    size_t output_stride) XNN_OOB_READS
{
  assert(rows != 0);
  assert(channels != 0);
  assert(channels % sizeof(float) == 0);

  const float* i0 = input;
  float* o0 = output;

  const size_t input_increment = input_stride * 1 - channels;
  const size_t output_increment = output_stride * 1 - channels;

  do {

    const float* w = weights;
    size_t c = channels;
    for (; c >= 16 * sizeof(float); c -= 16 * sizeof(float)) {
      const float32x4_t vw0123 = vld1q_f32(w); w += 4;
      const float32x4_t vw4567 = vld1q_f32(w); w += 4;
      const float32x4_t vw89AB = vld1q_f32(w); w += 4;
      const float32x4_t vwCDEF = vld1q_f32(w); w += 4;

      const float32x4_t vi0x0123 = vld1q_f32(i0); i0 += 4;
      const float32x4_t vi0x4567 = vld1q_f32(i0); i0 += 4;
      const float32x4_t vi0x89AB = vld1q_f32(i0); i0 += 4;
      const float32x4_t vi0xCDEF = vld1q_f32(i0); i0 += 4;

      float32x4_t vacc0x0123 = vmulq_f32(vi0x0123, vw0123);
      const uint32x4_t vm0x0123 = vcltq_s32(vreinterpretq_s32_f32(vi0x0123), vmovq_n_s32(0));
      float32x4_t vacc0x4567 = vmulq_f32(vi0x4567, vw4567);
      const uint32x4_t vm0x4567 = vcltq_s32(vreinterpretq_s32_f32(vi0x4567), vmovq_n_s32(0));
      float32x4_t vacc0x89AB = vmulq_f32(vi0x89AB, vw89AB);
      const uint32x4_t vm0x89AB = vcltq_s32(vreinterpretq_s32_f32(vi0x89AB), vmovq_n_s32(0));
      float32x4_t vacc0xCDEF = vmulq_f32(vi0xCDEF, vwCDEF);
      const uint32x4_t vm0xCDEF = vcltq_s32(vreinterpretq_s32_f32(vi0xCDEF), vmovq_n_s32(0));

      vacc0x0123 = vbslq_f32(vm0x0123, vacc0x0123, vi0x0123);
      vacc0x4567 = vbslq_f32(vm0x4567, vacc0x4567, vi0x4567);
      vacc0x89AB = vbslq_f32(vm0x89AB, vacc0x89AB, vi0x89AB);
      vacc0xCDEF = vbslq_f32(vm0xCDEF, vacc0xCDEF, vi0xCDEF);

      vst1q_f32(o0, vacc0x0123); o0 += 4;
      vst1q_f32(o0, vacc0x4567); o0 += 4;
      vst1q_f32(o0, vacc0x89AB); o0 += 4;
      vst1q_f32(o0, vacc0xCDEF); o0 += 4;
    }
    for (; c >= 4 * sizeof(float); c -= 4 * sizeof(float)) {
      const float32x4_t vw0123 = vld1q_f32(w); w += 4;

      const float32x4_t vi0x0123 = vld1q_f32(i0);
      i0 += 4;

      float32x4_t vacc0x0123 = vmulq_f32(vi0x0123, vw0123);
      const uint32x4_t vm0x0123 = vcltq_s32(vreinterpretq_s32_f32(vi0x0123), vmovq_n_s32(0));

      vacc0x0123 = vbslq_f32(vm0x0123, vacc0x0123, vi0x0123);

      vst1q_f32(o0, vacc0x0123); o0 += 4;
    }
    if XNN_UNLIKELY(c != 0) {
      const float32x4_t vw0123 = vld1q_f32(w); w += 4;

      const float32x4_t vi0x0123 = vld1q_f32(i0);
      i0 = (const float*) ((uintptr_t) i0 + c);

      float32x4_t vacc0x0123 = vmulq_f32(vi0x0123, vw0123);
      const uint32x4_t vm0x0123 = vcltq_s32(vreinterpretq_s32_f32(vi0x0123), vmovq_n_s32(0));

      vacc0x0123 = vbslq_f32(vm0x0123, vacc0x0123, vi0x0123);

      float32x2_t vacc0x01 = vget_low_f32(vacc0x0123);
      if (c & (2 * sizeof(float))) {
        vst1_f32(o0, vacc0x01); o0 += 2;

        vacc0x01 = vget_high_f32(vacc0x0123);
      }
      if (c & (1 * sizeof(float))) {
        vst1_lane_f32(o0, vacc0x01, 0); o0 += 1;
      }
    }
    i0 = (const float*) ((uintptr_t) i0 + input_increment);
    o0 = (float*) ((uintptr_t) o0 + output_increment);
    rows = doz(rows, 1);
  } while (rows != 0);
}
