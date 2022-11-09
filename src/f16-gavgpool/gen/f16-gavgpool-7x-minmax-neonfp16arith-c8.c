// Auto-generated file. Do not edit!
//   Template: src/f16-gavgpool/unipass-neonfp16arith.c.in
//   Generator: tools/xngen
//
// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <arm_neon.h>

#include <xnnpack/gavgpool.h>


void xnn_f16_gavgpool_minmax_ukernel_7x__neonfp16arith_c8(
    size_t rows,
    size_t channels,
    const void* input,
    size_t input_stride,
    const void* zero,
    void* output,
    const union xnn_f16_scaleminmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(rows != 0);
  assert(rows <= 7);
  assert(channels != 0);

  const __fp16* i0 = input;
  const __fp16* i1 = (const __fp16*) ((uintptr_t) i0 + input_stride);
  if XNN_UNPREDICTABLE(rows < 2) {
    i1 = (const __fp16*) zero;
  }
  const __fp16* i2 = (const __fp16*) ((uintptr_t) i1 + input_stride);
  if XNN_UNPREDICTABLE(rows <= 2) {
    i2 = (const __fp16*) zero;
  }
  const __fp16* i3 = (const __fp16*) ((uintptr_t) i2 + input_stride);
  if XNN_UNPREDICTABLE(rows < 4) {
    i3 = (const __fp16*) zero;
  }
  const __fp16* i4 = (const __fp16*) ((uintptr_t) i3 + input_stride);
  if XNN_UNPREDICTABLE(rows <= 4) {
    i4 = (const __fp16*) zero;
  }
  const __fp16* i5 = (const __fp16*) ((uintptr_t) i4 + input_stride);
  if XNN_UNPREDICTABLE(rows < 6) {
    i5 = (const __fp16*) zero;
  }
  const __fp16* i6 = (const __fp16*) ((uintptr_t) i5 + input_stride);
  if XNN_UNPREDICTABLE(rows <= 6) {
    i6 = (const __fp16*) zero;
  }

  const float16x8_t vscale = vreinterpretq_f16_u16(vld1q_dup_u16(&params->fp16arith.scale));
  const float16x8_t vmin = vreinterpretq_f16_u16(vld1q_dup_u16(&params->fp16arith.min));
  const float16x8_t vmax = vreinterpretq_f16_u16(vld1q_dup_u16(&params->fp16arith.max));
  for (; channels >= 8; channels -= 8) {
    const float16x8_t vi0x01234567 = vld1q_f16(i0); i0 += 8;
    const float16x8_t vi1x01234567 = vld1q_f16(i1); i1 += 8;

    const float16x8_t vi2x01234567 = vld1q_f16(i2); i2 += 8;
    float16x8_t vacc01234567 = vaddq_f16(vi0x01234567, vi1x01234567);

    const float16x8_t vi3x01234567 = vld1q_f16(i3); i3 += 8;
    vacc01234567 = vaddq_f16(vacc01234567, vi2x01234567);
    const float16x8_t vi4x01234567 = vld1q_f16(i4); i4 += 8;
    vacc01234567 = vaddq_f16(vacc01234567, vi3x01234567);
    const float16x8_t vi5x01234567 = vld1q_f16(i5); i5 += 8;
    vacc01234567 = vaddq_f16(vacc01234567, vi4x01234567);
    const float16x8_t vi6x01234567 = vld1q_f16(i6); i6 += 8;
    vacc01234567 = vaddq_f16(vacc01234567, vi5x01234567);
    vacc01234567 = vaddq_f16(vacc01234567, vi6x01234567);

    vacc01234567 = vmulq_f16(vacc01234567, vscale);

    vacc01234567 = vmaxq_f16(vacc01234567, vmin);

    vacc01234567 = vminq_f16(vacc01234567, vmax);

    vst1q_f16(output, vacc01234567); output = (__fp16*) output + 8;
  }
  if XNN_UNLIKELY(channels != 0) {
    {
      const float16x8_t vi0x01234567 = vld1q_f16(i0); i0 += 8;
      const float16x8_t vi1x01234567 = vld1q_f16(i1); i1 += 8;

      const float16x8_t vi2x01234567 = vld1q_f16(i2); i2 += 8;
      float16x8_t vacc01234567 = vaddq_f16(vi0x01234567, vi1x01234567);

      const float16x8_t vi3x01234567 = vld1q_f16(i3); i3 += 8;
      vacc01234567 = vaddq_f16(vacc01234567, vi2x01234567);
      const float16x8_t vi4x01234567 = vld1q_f16(i4); i4 += 8;
      vacc01234567 = vaddq_f16(vacc01234567, vi3x01234567);
      const float16x8_t vi5x01234567 = vld1q_f16(i5); i5 += 8;
      vacc01234567 = vaddq_f16(vacc01234567, vi4x01234567);
      const float16x8_t vi6x01234567 = vld1q_f16(i6); i6 += 8;
      vacc01234567 = vaddq_f16(vacc01234567, vi5x01234567);
      vacc01234567 = vaddq_f16(vacc01234567, vi6x01234567);

      vacc01234567 = vmulq_f16(vacc01234567, vscale);
      vacc01234567 = vmaxq_f16(vacc01234567, vmin);
      vacc01234567 = vminq_f16(vacc01234567, vmax);

      float16x4_t vacc0123 = vget_low_f16(vacc01234567);
      if (channels & 4) {
        vst1_f16(output, vacc0123); output = (__fp16*) output + 4;
        vacc0123 = vget_high_f16(vacc01234567);
      }
      if (channels & 2) {
        vst1_lane_u32(output, vreinterpret_u32_f16(vacc0123), 0); output = (__fp16*) output + 2;
        vacc0123 = vext_f16(vacc0123, vacc0123, 2);
      }
      if (channels & 1) {
        vst1_lane_f16(output, vacc0123, 0); output = (__fp16*) output + 1;
      }
    }
  }
}
