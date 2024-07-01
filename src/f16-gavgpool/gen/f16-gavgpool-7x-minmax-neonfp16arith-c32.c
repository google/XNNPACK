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

#include "xnnpack/gavgpool.h"


void xnn_f16_gavgpool_minmax_ukernel_7x__neonfp16arith_c32(
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

  const uint16_t* i0 = input;
  const uint16_t* i1 = (const uint16_t*) ((uintptr_t) i0 + input_stride);
  if XNN_UNPREDICTABLE(rows < 2) {
    i1 = (const uint16_t*) zero;
  }
  const uint16_t* i2 = (const uint16_t*) ((uintptr_t) i1 + input_stride);
  if XNN_UNPREDICTABLE(rows <= 2) {
    i2 = (const uint16_t*) zero;
  }
  const uint16_t* i3 = (const uint16_t*) ((uintptr_t) i2 + input_stride);
  if XNN_UNPREDICTABLE(rows < 4) {
    i3 = (const uint16_t*) zero;
  }
  const uint16_t* i4 = (const uint16_t*) ((uintptr_t) i3 + input_stride);
  if XNN_UNPREDICTABLE(rows <= 4) {
    i4 = (const uint16_t*) zero;
  }
  const uint16_t* i5 = (const uint16_t*) ((uintptr_t) i4 + input_stride);
  if XNN_UNPREDICTABLE(rows < 6) {
    i5 = (const uint16_t*) zero;
  }
  const uint16_t* i6 = (const uint16_t*) ((uintptr_t) i5 + input_stride);
  if XNN_UNPREDICTABLE(rows <= 6) {
    i6 = (const uint16_t*) zero;
  }

  const float16x8_t vscale = vreinterpretq_f16_u16(vld1q_dup_u16(&params->fp16arith.scale));
  const float16x8_t vmin = vreinterpretq_f16_u16(vld1q_dup_u16(&params->fp16arith.min));
  const float16x8_t vmax = vreinterpretq_f16_u16(vld1q_dup_u16(&params->fp16arith.max));
  for (; channels >= 32; channels -= 32) {
    const float16x8_t vi0x01234567 = vreinterpretq_f16_u16(vld1q_u16(i0)); i0 += 8;
    const float16x8_t vi0x89ABCDEF = vreinterpretq_f16_u16(vld1q_u16(i0)); i0 += 8;
    const float16x8_t vi0xGHIJKLMN = vreinterpretq_f16_u16(vld1q_u16(i0)); i0 += 8;
    const float16x8_t vi0xOPQRSTUV = vreinterpretq_f16_u16(vld1q_u16(i0)); i0 += 8;
    const float16x8_t vi1x01234567 = vreinterpretq_f16_u16(vld1q_u16(i1)); i1 += 8;
    const float16x8_t vi1x89ABCDEF = vreinterpretq_f16_u16(vld1q_u16(i1)); i1 += 8;
    const float16x8_t vi1xGHIJKLMN = vreinterpretq_f16_u16(vld1q_u16(i1)); i1 += 8;
    const float16x8_t vi1xOPQRSTUV = vreinterpretq_f16_u16(vld1q_u16(i1)); i1 += 8;

    const float16x8_t vi2x01234567 = vreinterpretq_f16_u16(vld1q_u16(i2)); i2 += 8;
    float16x8_t vacc01234567 = vaddq_f16(vi0x01234567, vi1x01234567);
    const float16x8_t vi2x89ABCDEF = vreinterpretq_f16_u16(vld1q_u16(i2)); i2 += 8;
    float16x8_t vacc89ABCDEF = vaddq_f16(vi0x89ABCDEF, vi1x89ABCDEF);
    const float16x8_t vi2xGHIJKLMN = vreinterpretq_f16_u16(vld1q_u16(i2)); i2 += 8;
    float16x8_t vaccGHIJKLMN = vaddq_f16(vi0xGHIJKLMN, vi1xGHIJKLMN);
    const float16x8_t vi2xOPQRSTUV = vreinterpretq_f16_u16(vld1q_u16(i2)); i2 += 8;
    float16x8_t vaccOPQRSTUV = vaddq_f16(vi0xOPQRSTUV, vi1xOPQRSTUV);

    const float16x8_t vi3x01234567 = vreinterpretq_f16_u16(vld1q_u16(i3)); i3 += 8;
    vacc01234567 = vaddq_f16(vacc01234567, vi2x01234567);
    const float16x8_t vi3x89ABCDEF = vreinterpretq_f16_u16(vld1q_u16(i3)); i3 += 8;
    vacc89ABCDEF = vaddq_f16(vacc89ABCDEF, vi2x89ABCDEF);
    const float16x8_t vi3xGHIJKLMN = vreinterpretq_f16_u16(vld1q_u16(i3)); i3 += 8;
    vaccGHIJKLMN = vaddq_f16(vaccGHIJKLMN, vi2xGHIJKLMN);
    const float16x8_t vi3xOPQRSTUV = vreinterpretq_f16_u16(vld1q_u16(i3)); i3 += 8;
    vaccOPQRSTUV = vaddq_f16(vaccOPQRSTUV, vi2xOPQRSTUV);
    const float16x8_t vi4x01234567 = vreinterpretq_f16_u16(vld1q_u16(i4)); i4 += 8;
    vacc01234567 = vaddq_f16(vacc01234567, vi3x01234567);
    const float16x8_t vi4x89ABCDEF = vreinterpretq_f16_u16(vld1q_u16(i4)); i4 += 8;
    vacc89ABCDEF = vaddq_f16(vacc89ABCDEF, vi3x89ABCDEF);
    const float16x8_t vi4xGHIJKLMN = vreinterpretq_f16_u16(vld1q_u16(i4)); i4 += 8;
    vaccGHIJKLMN = vaddq_f16(vaccGHIJKLMN, vi3xGHIJKLMN);
    const float16x8_t vi4xOPQRSTUV = vreinterpretq_f16_u16(vld1q_u16(i4)); i4 += 8;
    vaccOPQRSTUV = vaddq_f16(vaccOPQRSTUV, vi3xOPQRSTUV);
    const float16x8_t vi5x01234567 = vreinterpretq_f16_u16(vld1q_u16(i5)); i5 += 8;
    vacc01234567 = vaddq_f16(vacc01234567, vi4x01234567);
    const float16x8_t vi5x89ABCDEF = vreinterpretq_f16_u16(vld1q_u16(i5)); i5 += 8;
    vacc89ABCDEF = vaddq_f16(vacc89ABCDEF, vi4x89ABCDEF);
    const float16x8_t vi5xGHIJKLMN = vreinterpretq_f16_u16(vld1q_u16(i5)); i5 += 8;
    vaccGHIJKLMN = vaddq_f16(vaccGHIJKLMN, vi4xGHIJKLMN);
    const float16x8_t vi5xOPQRSTUV = vreinterpretq_f16_u16(vld1q_u16(i5)); i5 += 8;
    vaccOPQRSTUV = vaddq_f16(vaccOPQRSTUV, vi4xOPQRSTUV);
    const float16x8_t vi6x01234567 = vreinterpretq_f16_u16(vld1q_u16(i6)); i6 += 8;
    vacc01234567 = vaddq_f16(vacc01234567, vi5x01234567);
    const float16x8_t vi6x89ABCDEF = vreinterpretq_f16_u16(vld1q_u16(i6)); i6 += 8;
    vacc89ABCDEF = vaddq_f16(vacc89ABCDEF, vi5x89ABCDEF);
    const float16x8_t vi6xGHIJKLMN = vreinterpretq_f16_u16(vld1q_u16(i6)); i6 += 8;
    vaccGHIJKLMN = vaddq_f16(vaccGHIJKLMN, vi5xGHIJKLMN);
    const float16x8_t vi6xOPQRSTUV = vreinterpretq_f16_u16(vld1q_u16(i6)); i6 += 8;
    vaccOPQRSTUV = vaddq_f16(vaccOPQRSTUV, vi5xOPQRSTUV);
    vacc01234567 = vaddq_f16(vacc01234567, vi6x01234567);
    vacc89ABCDEF = vaddq_f16(vacc89ABCDEF, vi6x89ABCDEF);
    vaccGHIJKLMN = vaddq_f16(vaccGHIJKLMN, vi6xGHIJKLMN);
    vaccOPQRSTUV = vaddq_f16(vaccOPQRSTUV, vi6xOPQRSTUV);

    vacc01234567 = vmulq_f16(vacc01234567, vscale);
    vacc89ABCDEF = vmulq_f16(vacc89ABCDEF, vscale);
    vaccGHIJKLMN = vmulq_f16(vaccGHIJKLMN, vscale);
    vaccOPQRSTUV = vmulq_f16(vaccOPQRSTUV, vscale);

    vacc01234567 = vmaxq_f16(vacc01234567, vmin);
    vacc89ABCDEF = vmaxq_f16(vacc89ABCDEF, vmin);
    vaccGHIJKLMN = vmaxq_f16(vaccGHIJKLMN, vmin);
    vaccOPQRSTUV = vmaxq_f16(vaccOPQRSTUV, vmin);

    vacc01234567 = vminq_f16(vacc01234567, vmax);
    vacc89ABCDEF = vminq_f16(vacc89ABCDEF, vmax);
    vaccGHIJKLMN = vminq_f16(vaccGHIJKLMN, vmax);
    vaccOPQRSTUV = vminq_f16(vaccOPQRSTUV, vmax);

    vst1q_u16(output, vreinterpretq_u16_f16(vacc01234567)); output = (uint16_t*) output + 8;
    vst1q_u16(output, vreinterpretq_u16_f16(vacc89ABCDEF)); output = (uint16_t*) output + 8;
    vst1q_u16(output, vreinterpretq_u16_f16(vaccGHIJKLMN)); output = (uint16_t*) output + 8;
    vst1q_u16(output, vreinterpretq_u16_f16(vaccOPQRSTUV)); output = (uint16_t*) output + 8;
  }
  if XNN_UNLIKELY(channels != 0) {
    do {
      const float16x8_t vi0x01234567 = vreinterpretq_f16_u16(vld1q_u16(i0)); i0 += 8;
      const float16x8_t vi1x01234567 = vreinterpretq_f16_u16(vld1q_u16(i1)); i1 += 8;

      const float16x8_t vi2x01234567 = vreinterpretq_f16_u16(vld1q_u16(i2)); i2 += 8;
      float16x8_t vacc01234567 = vaddq_f16(vi0x01234567, vi1x01234567);

      const float16x8_t vi3x01234567 = vreinterpretq_f16_u16(vld1q_u16(i3)); i3 += 8;
      vacc01234567 = vaddq_f16(vacc01234567, vi2x01234567);
      const float16x8_t vi4x01234567 = vreinterpretq_f16_u16(vld1q_u16(i4)); i4 += 8;
      vacc01234567 = vaddq_f16(vacc01234567, vi3x01234567);
      const float16x8_t vi5x01234567 = vreinterpretq_f16_u16(vld1q_u16(i5)); i5 += 8;
      vacc01234567 = vaddq_f16(vacc01234567, vi4x01234567);
      const float16x8_t vi6x01234567 = vreinterpretq_f16_u16(vld1q_u16(i6)); i6 += 8;
      vacc01234567 = vaddq_f16(vacc01234567, vi5x01234567);
      vacc01234567 = vaddq_f16(vacc01234567, vi6x01234567);

      vacc01234567 = vmulq_f16(vacc01234567, vscale);
      vacc01234567 = vmaxq_f16(vacc01234567, vmin);
      vacc01234567 = vminq_f16(vacc01234567, vmax);

      if XNN_LIKELY(channels >= 8) {
        vst1q_u16(output, vreinterpretq_u16_f16(vacc01234567)); output = (uint16_t*) output + 8;
        channels -= 8;
      } else {
        float16x4_t vacc0123 = vget_low_f16(vacc01234567);
        if (channels & 4) {
          vst1_u16(output, vreinterpret_u16_f16(vacc0123)); output = (uint16_t*) output + 4;
          vacc0123 = vget_high_f16(vacc01234567);
        }
        if (channels & 2) {
          vst1_lane_u32(output, vreinterpret_u32_f16(vacc0123), 0); output = (uint16_t*) output + 2;
          vacc0123 = vext_f16(vacc0123, vacc0123, 2);
        }
        if (channels & 1) {
          vst1_lane_u16(output, vreinterpret_u16_f16(vacc0123), 0); output = (uint16_t*) output + 1;
        }
        channels = 0;
      }
    } while (channels != 0);
  }
}
