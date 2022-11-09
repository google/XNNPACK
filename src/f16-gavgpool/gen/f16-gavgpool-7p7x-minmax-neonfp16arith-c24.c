// Auto-generated file. Do not edit!
//   Template: src/f16-gavgpool/multipass-neonfp16arith.c.in
//   Generator: tools/xngen
//
// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <arm_neon.h>

#include <xnnpack/gavgpool.h>
#include <xnnpack/math.h>


void xnn_f16_gavgpool_minmax_ukernel_7p7x__neonfp16arith_c24(
    size_t rows,
    size_t channels,
    const void* input,
    size_t input_stride,
    const void* zero,
    void* buffer,
    void* output,
    const union xnn_f16_scaleminmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(rows > 7);
  assert(channels != 0);

  const __fp16* i0 = input;
  const __fp16* i1 = (const __fp16*) ((uintptr_t) i0 + input_stride);
  const __fp16* i2 = (const __fp16*) ((uintptr_t) i1 + input_stride);
  const __fp16* i3 = (const __fp16*) ((uintptr_t) i2 + input_stride);
  const __fp16* i4 = (const __fp16*) ((uintptr_t) i3 + input_stride);
  const __fp16* i5 = (const __fp16*) ((uintptr_t) i4 + input_stride);
  const __fp16* i6 = (const __fp16*) ((uintptr_t) i5 + input_stride);
  const size_t input_increment = 7 * input_stride - round_up_po2(channels, 8) * sizeof(__fp16);

  __fp16* b = buffer;
  size_t c = channels;
  for (; c >= 24; c -= 24) {
    const float16x8_t vi0x01234567 = vld1q_f16(i0); i0 += 8;
    const float16x8_t vi0x89ABCDEF = vld1q_f16(i0); i0 += 8;
    const float16x8_t vi0xGHIJKLMN = vld1q_f16(i0); i0 += 8;
    const float16x8_t vi1x01234567 = vld1q_f16(i1); i1 += 8;
    const float16x8_t vi1x89ABCDEF = vld1q_f16(i1); i1 += 8;
    const float16x8_t vi1xGHIJKLMN = vld1q_f16(i1); i1 += 8;

    const float16x8_t vi2x01234567 = vld1q_f16(i2); i2 += 8;
    float16x8_t vacc01234567 = vaddq_f16(vi0x01234567, vi1x01234567);
    const float16x8_t vi2x89ABCDEF = vld1q_f16(i2); i2 += 8;
    float16x8_t vacc89ABCDEF = vaddq_f16(vi0x89ABCDEF, vi1x89ABCDEF);
    const float16x8_t vi2xGHIJKLMN = vld1q_f16(i2); i2 += 8;
    float16x8_t vaccGHIJKLMN = vaddq_f16(vi0xGHIJKLMN, vi1xGHIJKLMN);

    const float16x8_t vi3x01234567 = vld1q_f16(i3); i3 += 8;
    vacc01234567 = vaddq_f16(vacc01234567, vi2x01234567);
    const float16x8_t vi3x89ABCDEF = vld1q_f16(i3); i3 += 8;
    vacc89ABCDEF = vaddq_f16(vacc89ABCDEF, vi2x89ABCDEF);
    const float16x8_t vi3xGHIJKLMN = vld1q_f16(i3); i3 += 8;
    vaccGHIJKLMN = vaddq_f16(vaccGHIJKLMN, vi2xGHIJKLMN);
    const float16x8_t vi4x01234567 = vld1q_f16(i4); i4 += 8;
    vacc01234567 = vaddq_f16(vacc01234567, vi3x01234567);
    const float16x8_t vi4x89ABCDEF = vld1q_f16(i4); i4 += 8;
    vacc89ABCDEF = vaddq_f16(vacc89ABCDEF, vi3x89ABCDEF);
    const float16x8_t vi4xGHIJKLMN = vld1q_f16(i4); i4 += 8;
    vaccGHIJKLMN = vaddq_f16(vaccGHIJKLMN, vi3xGHIJKLMN);
    const float16x8_t vi5x01234567 = vld1q_f16(i5); i5 += 8;
    vacc01234567 = vaddq_f16(vacc01234567, vi4x01234567);
    const float16x8_t vi5x89ABCDEF = vld1q_f16(i5); i5 += 8;
    vacc89ABCDEF = vaddq_f16(vacc89ABCDEF, vi4x89ABCDEF);
    const float16x8_t vi5xGHIJKLMN = vld1q_f16(i5); i5 += 8;
    vaccGHIJKLMN = vaddq_f16(vaccGHIJKLMN, vi4xGHIJKLMN);
    const float16x8_t vi6x01234567 = vld1q_f16(i6); i6 += 8;
    vacc01234567 = vaddq_f16(vacc01234567, vi5x01234567);
    const float16x8_t vi6x89ABCDEF = vld1q_f16(i6); i6 += 8;
    vacc89ABCDEF = vaddq_f16(vacc89ABCDEF, vi5x89ABCDEF);
    const float16x8_t vi6xGHIJKLMN = vld1q_f16(i6); i6 += 8;
    vaccGHIJKLMN = vaddq_f16(vaccGHIJKLMN, vi5xGHIJKLMN);
    vacc01234567 = vaddq_f16(vacc01234567, vi6x01234567);
    vacc89ABCDEF = vaddq_f16(vacc89ABCDEF, vi6x89ABCDEF);
    vaccGHIJKLMN = vaddq_f16(vaccGHIJKLMN, vi6xGHIJKLMN);

    vst1q_f16(b, vacc01234567); b += 8;
    vst1q_f16(b, vacc89ABCDEF); b += 8;
    vst1q_f16(b, vaccGHIJKLMN); b += 8;
  }
  if XNN_UNLIKELY(c != 0) {
    do {
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

      vst1q_f16(b, vacc01234567); b += 8;

      c = doz(c, 8);
    } while (c != 0);
  }

  for (rows -= 7; rows > 7; rows -= 7) {
    i0 = (const __fp16*) ((uintptr_t) i0 + input_increment);
    i1 = (const __fp16*) ((uintptr_t) i1 + input_increment);
    i2 = (const __fp16*) ((uintptr_t) i2 + input_increment);
    i3 = (const __fp16*) ((uintptr_t) i3 + input_increment);
    i4 = (const __fp16*) ((uintptr_t) i4 + input_increment);
    i5 = (const __fp16*) ((uintptr_t) i5 + input_increment);
    i6 = (const __fp16*) ((uintptr_t) i6 + input_increment);

    __fp16* b = buffer;
    size_t c = channels;
    for (; c >= 24; c -= 24) {
      float16x8_t vacc01234567 = vld1q_f16(b);
      float16x8_t vacc89ABCDEF = vld1q_f16(b + 8);
      float16x8_t vaccGHIJKLMN = vld1q_f16(b + 16);

      const float16x8_t vi0x01234567 = vld1q_f16(i0); i0 += 8;
      const float16x8_t vi0x89ABCDEF = vld1q_f16(i0); i0 += 8;
      const float16x8_t vi0xGHIJKLMN = vld1q_f16(i0); i0 += 8;

      const float16x8_t vi1x01234567 = vld1q_f16(i1); i1 += 8;
      vacc01234567 = vaddq_f16(vacc01234567, vi0x01234567);
      const float16x8_t vi1x89ABCDEF = vld1q_f16(i1); i1 += 8;
      vacc89ABCDEF = vaddq_f16(vacc89ABCDEF, vi0x89ABCDEF);
      const float16x8_t vi1xGHIJKLMN = vld1q_f16(i1); i1 += 8;
      vaccGHIJKLMN = vaddq_f16(vaccGHIJKLMN, vi0xGHIJKLMN);
      const float16x8_t vi2x01234567 = vld1q_f16(i2); i2 += 8;
      vacc01234567 = vaddq_f16(vacc01234567, vi1x01234567);
      const float16x8_t vi2x89ABCDEF = vld1q_f16(i2); i2 += 8;
      vacc89ABCDEF = vaddq_f16(vacc89ABCDEF, vi1x89ABCDEF);
      const float16x8_t vi2xGHIJKLMN = vld1q_f16(i2); i2 += 8;
      vaccGHIJKLMN = vaddq_f16(vaccGHIJKLMN, vi1xGHIJKLMN);
      const float16x8_t vi3x01234567 = vld1q_f16(i3); i3 += 8;
      vacc01234567 = vaddq_f16(vacc01234567, vi2x01234567);
      const float16x8_t vi3x89ABCDEF = vld1q_f16(i3); i3 += 8;
      vacc89ABCDEF = vaddq_f16(vacc89ABCDEF, vi2x89ABCDEF);
      const float16x8_t vi3xGHIJKLMN = vld1q_f16(i3); i3 += 8;
      vaccGHIJKLMN = vaddq_f16(vaccGHIJKLMN, vi2xGHIJKLMN);
      const float16x8_t vi4x01234567 = vld1q_f16(i4); i4 += 8;
      vacc01234567 = vaddq_f16(vacc01234567, vi3x01234567);
      const float16x8_t vi4x89ABCDEF = vld1q_f16(i4); i4 += 8;
      vacc89ABCDEF = vaddq_f16(vacc89ABCDEF, vi3x89ABCDEF);
      const float16x8_t vi4xGHIJKLMN = vld1q_f16(i4); i4 += 8;
      vaccGHIJKLMN = vaddq_f16(vaccGHIJKLMN, vi3xGHIJKLMN);
      const float16x8_t vi5x01234567 = vld1q_f16(i5); i5 += 8;
      vacc01234567 = vaddq_f16(vacc01234567, vi4x01234567);
      const float16x8_t vi5x89ABCDEF = vld1q_f16(i5); i5 += 8;
      vacc89ABCDEF = vaddq_f16(vacc89ABCDEF, vi4x89ABCDEF);
      const float16x8_t vi5xGHIJKLMN = vld1q_f16(i5); i5 += 8;
      vaccGHIJKLMN = vaddq_f16(vaccGHIJKLMN, vi4xGHIJKLMN);
      const float16x8_t vi6x01234567 = vld1q_f16(i6); i6 += 8;
      vacc01234567 = vaddq_f16(vacc01234567, vi5x01234567);
      const float16x8_t vi6x89ABCDEF = vld1q_f16(i6); i6 += 8;
      vacc89ABCDEF = vaddq_f16(vacc89ABCDEF, vi5x89ABCDEF);
      const float16x8_t vi6xGHIJKLMN = vld1q_f16(i6); i6 += 8;
      vaccGHIJKLMN = vaddq_f16(vaccGHIJKLMN, vi5xGHIJKLMN);
      vacc01234567 = vaddq_f16(vacc01234567, vi6x01234567);
      vacc89ABCDEF = vaddq_f16(vacc89ABCDEF, vi6x89ABCDEF);
      vaccGHIJKLMN = vaddq_f16(vaccGHIJKLMN, vi6xGHIJKLMN);

      vst1q_f16(b, vacc01234567); b += 8;
      vst1q_f16(b, vacc89ABCDEF); b += 8;
      vst1q_f16(b, vaccGHIJKLMN); b += 8;
    }
    if XNN_UNLIKELY(c != 0) {
      do {
        float16x8_t vacc01234567 = vld1q_f16(b);
        const float16x8_t vi0x01234567 = vld1q_f16(i0); i0 += 8;

        const float16x8_t vi1x01234567 = vld1q_f16(i1); i1 += 8;
        vacc01234567 = vaddq_f16(vacc01234567, vi0x01234567);
        const float16x8_t vi2x01234567 = vld1q_f16(i2); i2 += 8;
        vacc01234567 = vaddq_f16(vacc01234567, vi1x01234567);
        const float16x8_t vi3x01234567 = vld1q_f16(i3); i3 += 8;
        vacc01234567 = vaddq_f16(vacc01234567, vi2x01234567);
        const float16x8_t vi4x01234567 = vld1q_f16(i4); i4 += 8;
        vacc01234567 = vaddq_f16(vacc01234567, vi3x01234567);
        const float16x8_t vi5x01234567 = vld1q_f16(i5); i5 += 8;
        vacc01234567 = vaddq_f16(vacc01234567, vi4x01234567);
        const float16x8_t vi6x01234567 = vld1q_f16(i6); i6 += 8;
        vacc01234567 = vaddq_f16(vacc01234567, vi5x01234567);
        vacc01234567 = vaddq_f16(vacc01234567, vi6x01234567);

        vst1q_f16(b, vacc01234567); b += 8;

        c = doz(c, 8);
      } while (c != 0);
    }
  }

  i0 = (const __fp16*) ((uintptr_t) i0 + input_increment);
  i1 = (const __fp16*) ((uintptr_t) i1 + input_increment);
  if XNN_UNPREDICTABLE(rows < 2) {
    i1 = (const __fp16*) zero;
  }
  i2 = (const __fp16*) ((uintptr_t) i2 + input_increment);
  if XNN_UNPREDICTABLE(rows <= 2) {
    i2 = (const __fp16*) zero;
  }
  i3 = (const __fp16*) ((uintptr_t) i3 + input_increment);
  if XNN_UNPREDICTABLE(rows < 4) {
    i3 = (const __fp16*) zero;
  }
  i4 = (const __fp16*) ((uintptr_t) i4 + input_increment);
  if XNN_UNPREDICTABLE(rows <= 4) {
    i4 = (const __fp16*) zero;
  }
  i5 = (const __fp16*) ((uintptr_t) i5 + input_increment);
  if XNN_UNPREDICTABLE(rows < 6) {
    i5 = (const __fp16*) zero;
  }
  i6 = (const __fp16*) ((uintptr_t) i6 + input_increment);
  if XNN_UNPREDICTABLE(rows <= 6) {
    i6 = (const __fp16*) zero;
  }

  const float16x8_t vscale = vreinterpretq_f16_u16(vld1q_dup_u16(&params->fp16arith.scale));
  const float16x8_t vmin = vreinterpretq_f16_u16(vld1q_dup_u16(&params->fp16arith.min));
  const float16x8_t vmax = vreinterpretq_f16_u16(vld1q_dup_u16(&params->fp16arith.max));
  for (; channels >= 24; channels -= 24) {
    float16x8_t vacc01234567 = vld1q_f16(buffer); buffer = (__fp16*) buffer + 8;
    float16x8_t vacc89ABCDEF = vld1q_f16(buffer); buffer = (__fp16*) buffer + 8;
    float16x8_t vaccGHIJKLMN = vld1q_f16(buffer); buffer = (__fp16*) buffer + 8;

    const float16x8_t vi0x01234567 = vld1q_f16(i0); i0 += 8;
    const float16x8_t vi0x89ABCDEF = vld1q_f16(i0); i0 += 8;
    const float16x8_t vi0xGHIJKLMN = vld1q_f16(i0); i0 += 8;

    const float16x8_t vi1x01234567 = vld1q_f16(i1); i1 += 8;
    vacc01234567 = vaddq_f16(vacc01234567, vi0x01234567);
    const float16x8_t vi1x89ABCDEF = vld1q_f16(i1); i1 += 8;
    vacc89ABCDEF = vaddq_f16(vacc89ABCDEF, vi0x89ABCDEF);
    const float16x8_t vi1xGHIJKLMN = vld1q_f16(i1); i1 += 8;
    vaccGHIJKLMN = vaddq_f16(vaccGHIJKLMN, vi0xGHIJKLMN);
    const float16x8_t vi2x01234567 = vld1q_f16(i2); i2 += 8;
    vacc01234567 = vaddq_f16(vacc01234567, vi1x01234567);
    const float16x8_t vi2x89ABCDEF = vld1q_f16(i2); i2 += 8;
    vacc89ABCDEF = vaddq_f16(vacc89ABCDEF, vi1x89ABCDEF);
    const float16x8_t vi2xGHIJKLMN = vld1q_f16(i2); i2 += 8;
    vaccGHIJKLMN = vaddq_f16(vaccGHIJKLMN, vi1xGHIJKLMN);
    const float16x8_t vi3x01234567 = vld1q_f16(i3); i3 += 8;
    vacc01234567 = vaddq_f16(vacc01234567, vi2x01234567);
    const float16x8_t vi3x89ABCDEF = vld1q_f16(i3); i3 += 8;
    vacc89ABCDEF = vaddq_f16(vacc89ABCDEF, vi2x89ABCDEF);
    const float16x8_t vi3xGHIJKLMN = vld1q_f16(i3); i3 += 8;
    vaccGHIJKLMN = vaddq_f16(vaccGHIJKLMN, vi2xGHIJKLMN);
    const float16x8_t vi4x01234567 = vld1q_f16(i4); i4 += 8;
    vacc01234567 = vaddq_f16(vacc01234567, vi3x01234567);
    const float16x8_t vi4x89ABCDEF = vld1q_f16(i4); i4 += 8;
    vacc89ABCDEF = vaddq_f16(vacc89ABCDEF, vi3x89ABCDEF);
    const float16x8_t vi4xGHIJKLMN = vld1q_f16(i4); i4 += 8;
    vaccGHIJKLMN = vaddq_f16(vaccGHIJKLMN, vi3xGHIJKLMN);
    const float16x8_t vi5x01234567 = vld1q_f16(i5); i5 += 8;
    vacc01234567 = vaddq_f16(vacc01234567, vi4x01234567);
    const float16x8_t vi5x89ABCDEF = vld1q_f16(i5); i5 += 8;
    vacc89ABCDEF = vaddq_f16(vacc89ABCDEF, vi4x89ABCDEF);
    const float16x8_t vi5xGHIJKLMN = vld1q_f16(i5); i5 += 8;
    vaccGHIJKLMN = vaddq_f16(vaccGHIJKLMN, vi4xGHIJKLMN);
    const float16x8_t vi6x01234567 = vld1q_f16(i6); i6 += 8;
    vacc01234567 = vaddq_f16(vacc01234567, vi5x01234567);
    const float16x8_t vi6x89ABCDEF = vld1q_f16(i6); i6 += 8;
    vacc89ABCDEF = vaddq_f16(vacc89ABCDEF, vi5x89ABCDEF);
    const float16x8_t vi6xGHIJKLMN = vld1q_f16(i6); i6 += 8;
    vaccGHIJKLMN = vaddq_f16(vaccGHIJKLMN, vi5xGHIJKLMN);
    vacc01234567 = vaddq_f16(vacc01234567, vi6x01234567);
    vacc89ABCDEF = vaddq_f16(vacc89ABCDEF, vi6x89ABCDEF);
    vaccGHIJKLMN = vaddq_f16(vaccGHIJKLMN, vi6xGHIJKLMN);

    vacc01234567 = vmulq_f16(vacc01234567, vscale);
    vacc89ABCDEF = vmulq_f16(vacc89ABCDEF, vscale);
    vaccGHIJKLMN = vmulq_f16(vaccGHIJKLMN, vscale);

    vacc01234567 = vmaxq_f16(vacc01234567, vmin);
    vacc89ABCDEF = vmaxq_f16(vacc89ABCDEF, vmin);
    vaccGHIJKLMN = vmaxq_f16(vaccGHIJKLMN, vmin);

    vacc01234567 = vminq_f16(vacc01234567, vmax);
    vacc89ABCDEF = vminq_f16(vacc89ABCDEF, vmax);
    vaccGHIJKLMN = vminq_f16(vaccGHIJKLMN, vmax);

    vst1q_f16(output, vacc01234567); output = (__fp16*) output + 8;
    vst1q_f16(output, vacc89ABCDEF); output = (__fp16*) output + 8;
    vst1q_f16(output, vaccGHIJKLMN); output = (__fp16*) output + 8;
  }
  if XNN_UNLIKELY(channels != 0) {
    do {
      float16x8_t vacc01234567 = vld1q_f16(buffer); buffer = (__fp16*) buffer + 8;

      const float16x8_t vi0x01234567 = vld1q_f16(i0); i0 += 8;
      const float16x8_t vi1x01234567 = vld1q_f16(i1); i1 += 8;
      vacc01234567 = vaddq_f16(vacc01234567, vi0x01234567);
      const float16x8_t vi2x01234567 = vld1q_f16(i2); i2 += 8;
      vacc01234567 = vaddq_f16(vacc01234567, vi1x01234567);
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

      if XNN_LIKELY(channels >= 8) {
        vst1q_f16(output, vacc01234567); output = (__fp16*) output + 8;
        channels -= 8;
      } else {
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
        channels = 0;
      }
    } while (channels != 0);
  }
}
