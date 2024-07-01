// Auto-generated file. Do not edit!
//   Template: src/qs8-gavgpool/multipass-neon.c.in
//   Generator: tools/xngen
//
// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <arm_neon.h>

#include "xnnpack/gavgpool.h"
#include "xnnpack/intrinsics-polyfill.h"
#include "xnnpack/math.h"


void xnn_qu8_gavgpool_minmax_fp32_ukernel_7p7x__neonv8_c24(
    size_t rows,
    size_t channels,
    const uint8_t* input,
    size_t input_stride,
    const uint8_t* zero,
    int32_t* buffer,
    uint8_t* output,
    const union xnn_qu8_avgpool_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(rows > 7);
  assert(channels != 0);

  const uint8_t* i0 = input;
  const uint8_t* i1 = (const uint8_t*) ((uintptr_t) i0 + input_stride);
  const uint8_t* i2 = (const uint8_t*) ((uintptr_t) i1 + input_stride);
  const uint8_t* i3 = (const uint8_t*) ((uintptr_t) i2 + input_stride);
  const uint8_t* i4 = (const uint8_t*) ((uintptr_t) i3 + input_stride);
  const uint8_t* i5 = (const uint8_t*) ((uintptr_t) i4 + input_stride);
  const uint8_t* i6 = (const uint8_t*) ((uintptr_t) i5 + input_stride);
  const size_t input_increment = 7 * input_stride - round_up_po2(channels, 8) * sizeof(uint8_t);

  const int32x4_t vinit_bias = vld1q_dup_s32(&params->fp32_neonv8.init_bias);
  int32_t* b = buffer;
  size_t c = channels;
  for (; c >= 24; c -= 24) {
    const uint8x8_t vi0x01234567 = vld1_u8(i0); i0 += 8;
    const uint8x8_t vi0x89ABCDEF = vld1_u8(i0); i0 += 8;
    const uint8x8_t vi0xGHIJKLMN = vld1_u8(i0); i0 += 8;
    const uint8x8_t vi1x01234567 = vld1_u8(i1); i1 += 8;
    const uint8x8_t vi1x89ABCDEF = vld1_u8(i1); i1 += 8;
    const uint8x8_t vi1xGHIJKLMN = vld1_u8(i1); i1 += 8;

    const uint8x8_t vi2x01234567 = vld1_u8(i2); i2 += 8;
    uint16x8_t vsum01234567 = vaddl_u8(vi0x01234567, vi1x01234567);
    const uint8x8_t vi2x89ABCDEF = vld1_u8(i2); i2 += 8;
    uint16x8_t vsum89ABCDEF = vaddl_u8(vi0x89ABCDEF, vi1x89ABCDEF);
    const uint8x8_t vi2xGHIJKLMN = vld1_u8(i2); i2 += 8;
    uint16x8_t vsumGHIJKLMN = vaddl_u8(vi0xGHIJKLMN, vi1xGHIJKLMN);

    const uint8x8_t vi3x01234567 = vld1_u8(i3); i3 += 8;
    vsum01234567 = vaddw_u8(vsum01234567, vi2x01234567);
    const uint8x8_t vi3x89ABCDEF = vld1_u8(i3); i3 += 8;
    vsum89ABCDEF = vaddw_u8(vsum89ABCDEF, vi2x89ABCDEF);
    const uint8x8_t vi3xGHIJKLMN = vld1_u8(i3); i3 += 8;
    vsumGHIJKLMN = vaddw_u8(vsumGHIJKLMN, vi2xGHIJKLMN);
    const uint8x8_t vi4x01234567 = vld1_u8(i4); i4 += 8;
    vsum01234567 = vaddw_u8(vsum01234567, vi3x01234567);
    const uint8x8_t vi4x89ABCDEF = vld1_u8(i4); i4 += 8;
    vsum89ABCDEF = vaddw_u8(vsum89ABCDEF, vi3x89ABCDEF);
    const uint8x8_t vi4xGHIJKLMN = vld1_u8(i4); i4 += 8;
    vsumGHIJKLMN = vaddw_u8(vsumGHIJKLMN, vi3xGHIJKLMN);
    const uint8x8_t vi5x01234567 = vld1_u8(i5); i5 += 8;
    vsum01234567 = vaddw_u8(vsum01234567, vi4x01234567);
    const uint8x8_t vi5x89ABCDEF = vld1_u8(i5); i5 += 8;
    vsum89ABCDEF = vaddw_u8(vsum89ABCDEF, vi4x89ABCDEF);
    const uint8x8_t vi5xGHIJKLMN = vld1_u8(i5); i5 += 8;
    vsumGHIJKLMN = vaddw_u8(vsumGHIJKLMN, vi4xGHIJKLMN);
    const uint8x8_t vi6x01234567 = vld1_u8(i6); i6 += 8;
    vsum01234567 = vaddw_u8(vsum01234567, vi5x01234567);
    const uint8x8_t vi6x89ABCDEF = vld1_u8(i6); i6 += 8;
    vsum89ABCDEF = vaddw_u8(vsum89ABCDEF, vi5x89ABCDEF);
    const uint8x8_t vi6xGHIJKLMN = vld1_u8(i6); i6 += 8;
    vsumGHIJKLMN = vaddw_u8(vsumGHIJKLMN, vi5xGHIJKLMN);
    vsum01234567 = vaddw_u8(vsum01234567, vi6x01234567);
    vsum89ABCDEF = vaddw_u8(vsum89ABCDEF, vi6x89ABCDEF);
    vsumGHIJKLMN = vaddw_u8(vsumGHIJKLMN, vi6xGHIJKLMN);

    const int32x4_t vacc0123 = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vinit_bias), vget_low_u16(vsum01234567)));
    const int32x4_t vacc4567 = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vinit_bias), vget_high_u16(vsum01234567)));
    const int32x4_t vacc89AB = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vinit_bias), vget_low_u16(vsum89ABCDEF)));
    const int32x4_t vaccCDEF = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vinit_bias), vget_high_u16(vsum89ABCDEF)));
    const int32x4_t vaccGHIJ = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vinit_bias), vget_low_u16(vsumGHIJKLMN)));
    const int32x4_t vaccKLMN = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vinit_bias), vget_high_u16(vsumGHIJKLMN)));

    vst1q_s32(b, vacc0123); b += 4;
    vst1q_s32(b, vacc4567); b += 4;
    vst1q_s32(b, vacc89AB); b += 4;
    vst1q_s32(b, vaccCDEF); b += 4;
    vst1q_s32(b, vaccGHIJ); b += 4;
    vst1q_s32(b, vaccKLMN); b += 4;
  }
  if XNN_UNLIKELY(c != 0) {
    do {
      const uint8x8_t vi0x01234567 = vld1_u8(i0); i0 += 8;
      const uint8x8_t vi1x01234567 = vld1_u8(i1); i1 += 8;
      const uint8x8_t vi2x01234567 = vld1_u8(i2); i2 += 8;
      uint16x8_t vsum01234567 = vaddl_u8(vi0x01234567, vi1x01234567);

      const uint8x8_t vi3x01234567 = vld1_u8(i3); i3 += 8;
      vsum01234567 = vaddw_u8(vsum01234567, vi2x01234567);
      const uint8x8_t vi4x01234567 = vld1_u8(i4); i4 += 8;
      vsum01234567 = vaddw_u8(vsum01234567, vi3x01234567);
      const uint8x8_t vi5x01234567 = vld1_u8(i5); i5 += 8;
      vsum01234567 = vaddw_u8(vsum01234567, vi4x01234567);
      const uint8x8_t vi6x01234567 = vld1_u8(i6); i6 += 8;
      vsum01234567 = vaddw_u8(vsum01234567, vi5x01234567);
      vsum01234567 = vaddw_u8(vsum01234567, vi6x01234567);

      const int32x4_t vacc0123 = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vinit_bias), vget_low_u16(vsum01234567)));
      const int32x4_t vacc4567 = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vinit_bias), vget_high_u16(vsum01234567)));

      vst1q_s32(b, vacc0123); b += 4;
      vst1q_s32(b, vacc4567); b += 4;

      c = doz(c, 8);
    } while (c != 0);
  }

  for (rows -= 7; rows > 7; rows -= 7) {
    i0 = (const uint8_t*) ((uintptr_t) i0 + input_increment);
    i1 = (const uint8_t*) ((uintptr_t) i1 + input_increment);
    i2 = (const uint8_t*) ((uintptr_t) i2 + input_increment);
    i3 = (const uint8_t*) ((uintptr_t) i3 + input_increment);
    i4 = (const uint8_t*) ((uintptr_t) i4 + input_increment);
    i5 = (const uint8_t*) ((uintptr_t) i5 + input_increment);
    i6 = (const uint8_t*) ((uintptr_t) i6 + input_increment);

    int32_t* b = buffer;
    size_t c = channels;
    for (; c >= 24; c -= 24) {
      const uint8x8_t vi0x01234567 = vld1_u8(i0); i0 += 8;
      const uint8x8_t vi0x89ABCDEF = vld1_u8(i0); i0 += 8;
      const uint8x8_t vi0xGHIJKLMN = vld1_u8(i0); i0 += 8;
      const uint8x8_t vi1x01234567 = vld1_u8(i1); i1 += 8;
      const uint8x8_t vi1x89ABCDEF = vld1_u8(i1); i1 += 8;
      const uint8x8_t vi1xGHIJKLMN = vld1_u8(i1); i1 += 8;

      const uint8x8_t vi2x01234567 = vld1_u8(i2); i2 += 8;
      uint16x8_t vsum01234567 = vaddl_u8(vi0x01234567, vi1x01234567);
      const uint8x8_t vi2x89ABCDEF = vld1_u8(i2); i2 += 8;
      uint16x8_t vsum89ABCDEF = vaddl_u8(vi0x89ABCDEF, vi1x89ABCDEF);
      const uint8x8_t vi2xGHIJKLMN = vld1_u8(i2); i2 += 8;
      uint16x8_t vsumGHIJKLMN = vaddl_u8(vi0xGHIJKLMN, vi1xGHIJKLMN);

      const uint8x8_t vi3x01234567 = vld1_u8(i3); i3 += 8;
      vsum01234567 = vaddw_u8(vsum01234567, vi2x01234567);
      const uint8x8_t vi3x89ABCDEF = vld1_u8(i3); i3 += 8;
      vsum89ABCDEF = vaddw_u8(vsum89ABCDEF, vi2x89ABCDEF);
      const uint8x8_t vi3xGHIJKLMN = vld1_u8(i3); i3 += 8;
      vsumGHIJKLMN = vaddw_u8(vsumGHIJKLMN, vi2xGHIJKLMN);
      const uint8x8_t vi4x01234567 = vld1_u8(i4); i4 += 8;
      vsum01234567 = vaddw_u8(vsum01234567, vi3x01234567);
      const uint8x8_t vi4x89ABCDEF = vld1_u8(i4); i4 += 8;
      vsum89ABCDEF = vaddw_u8(vsum89ABCDEF, vi3x89ABCDEF);
      const uint8x8_t vi4xGHIJKLMN = vld1_u8(i4); i4 += 8;
      vsumGHIJKLMN = vaddw_u8(vsumGHIJKLMN, vi3xGHIJKLMN);
      const uint8x8_t vi5x01234567 = vld1_u8(i5); i5 += 8;
      vsum01234567 = vaddw_u8(vsum01234567, vi4x01234567);
      const uint8x8_t vi5x89ABCDEF = vld1_u8(i5); i5 += 8;
      vsum89ABCDEF = vaddw_u8(vsum89ABCDEF, vi4x89ABCDEF);
      const uint8x8_t vi5xGHIJKLMN = vld1_u8(i5); i5 += 8;
      vsumGHIJKLMN = vaddw_u8(vsumGHIJKLMN, vi4xGHIJKLMN);
      const uint8x8_t vi6x01234567 = vld1_u8(i6); i6 += 8;
      vsum01234567 = vaddw_u8(vsum01234567, vi5x01234567);
      const uint8x8_t vi6x89ABCDEF = vld1_u8(i6); i6 += 8;
      vsum89ABCDEF = vaddw_u8(vsum89ABCDEF, vi5x89ABCDEF);
      const uint8x8_t vi6xGHIJKLMN = vld1_u8(i6); i6 += 8;
      vsumGHIJKLMN = vaddw_u8(vsumGHIJKLMN, vi5xGHIJKLMN);
      int32x4_t vacc0123 = vld1q_s32(b);
      int32x4_t vacc4567 = vld1q_s32(b + 4);
      vsum01234567 = vaddw_u8(vsum01234567, vi6x01234567);
      int32x4_t vacc89AB = vld1q_s32(b + 8);
      int32x4_t vaccCDEF = vld1q_s32(b + 12);
      vsum89ABCDEF = vaddw_u8(vsum89ABCDEF, vi6x89ABCDEF);
      int32x4_t vaccGHIJ = vld1q_s32(b + 16);
      int32x4_t vaccKLMN = vld1q_s32(b + 20);
      vsumGHIJKLMN = vaddw_u8(vsumGHIJKLMN, vi6xGHIJKLMN);

      vacc0123 = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vacc0123), vget_low_u16(vsum01234567)));
      vacc4567 = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vacc4567), vget_high_u16(vsum01234567)));
      vacc89AB = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vacc89AB), vget_low_u16(vsum89ABCDEF)));
      vaccCDEF = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vaccCDEF), vget_high_u16(vsum89ABCDEF)));
      vaccGHIJ = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vaccGHIJ), vget_low_u16(vsumGHIJKLMN)));
      vaccKLMN = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vaccKLMN), vget_high_u16(vsumGHIJKLMN)));

      vst1q_s32(b, vacc0123); b += 4;
      vst1q_s32(b, vacc4567); b += 4;
      vst1q_s32(b, vacc89AB); b += 4;
      vst1q_s32(b, vaccCDEF); b += 4;
      vst1q_s32(b, vaccGHIJ); b += 4;
      vst1q_s32(b, vaccKLMN); b += 4;
    }
    if XNN_UNLIKELY(c != 0) {
      do {
        const uint8x8_t vi0x01234567 = vld1_u8(i0); i0 += 8;
        const uint8x8_t vi1x01234567 = vld1_u8(i1); i1 += 8;
        const uint8x8_t vi2x01234567 = vld1_u8(i2); i2 += 8;
        uint16x8_t vsum01234567 = vaddl_u8(vi0x01234567, vi1x01234567);

        const uint8x8_t vi3x01234567 = vld1_u8(i3); i3 += 8;
        vsum01234567 = vaddw_u8(vsum01234567, vi2x01234567);
        const uint8x8_t vi4x01234567 = vld1_u8(i4); i4 += 8;
        vsum01234567 = vaddw_u8(vsum01234567, vi3x01234567);
        const uint8x8_t vi5x01234567 = vld1_u8(i5); i5 += 8;
        vsum01234567 = vaddw_u8(vsum01234567, vi4x01234567);
        const uint8x8_t vi6x01234567 = vld1_u8(i6); i6 += 8;
        vsum01234567 = vaddw_u8(vsum01234567, vi5x01234567);
        int32x4_t vacc0123 = vld1q_s32(b);
        int32x4_t vacc4567 = vld1q_s32(b + 4);
        vsum01234567 = vaddw_u8(vsum01234567, vi6x01234567);

        vacc0123 = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vacc0123), vget_low_u16(vsum01234567)));
        vacc4567 = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vacc4567), vget_high_u16(vsum01234567)));

        vst1q_s32(b, vacc0123); b += 4;
        vst1q_s32(b, vacc4567); b += 4;

        c = doz(c, 8);
      } while (c != 0);
    }
  }

  i0 = (const uint8_t*) ((uintptr_t) i0 + input_increment);
  i1 = (const uint8_t*) ((uintptr_t) i1 + input_increment);
  if XNN_UNPREDICTABLE(rows < 2) {
    i1 = zero;
  }
  i2 = (const uint8_t*) ((uintptr_t) i2 + input_increment);
  if XNN_UNPREDICTABLE(rows <= 2) {
    i2 = zero;
  }
  i3 = (const uint8_t*) ((uintptr_t) i3 + input_increment);
  if XNN_UNPREDICTABLE(rows < 4) {
    i3 = zero;
  }
  i4 = (const uint8_t*) ((uintptr_t) i4 + input_increment);
  if XNN_UNPREDICTABLE(rows <= 4) {
    i4 = zero;
  }
  i5 = (const uint8_t*) ((uintptr_t) i5 + input_increment);
  if XNN_UNPREDICTABLE(rows < 6) {
    i5 = zero;
  }
  i6 = (const uint8_t*) ((uintptr_t) i6 + input_increment);
  if XNN_UNPREDICTABLE(rows <= 6) {
    i6 = zero;
  }

  const float32x4_t vscale = vld1q_dup_f32(&params->fp32_neonv8.scale);
  const int16x8_t voutput_zero_point = vld1q_dup_s16(&params->fp32_neonv8.output_zero_point);
  const uint8x16_t voutput_min = vld1q_dup_u8(&params->fp32_neonv8.output_min);
  const uint8x16_t voutput_max = vld1q_dup_u8(&params->fp32_neonv8.output_max);
  for (; channels >= 24; channels -= 24) {
    const uint8x8_t vi0x01234567 = vld1_u8(i0); i0 += 8;
    const uint8x8_t vi0x89ABCDEF = vld1_u8(i0); i0 += 8;
    const uint8x8_t vi0xGHIJKLMN = vld1_u8(i0); i0 += 8;
    const uint8x8_t vi1x01234567 = vld1_u8(i1); i1 += 8;
    const uint8x8_t vi1x89ABCDEF = vld1_u8(i1); i1 += 8;
    const uint8x8_t vi1xGHIJKLMN = vld1_u8(i1); i1 += 8;

    const uint8x8_t vi2x01234567 = vld1_u8(i2); i2 += 8;
    uint16x8_t vsum01234567 = vaddl_u8(vi0x01234567, vi1x01234567);
    const uint8x8_t vi2x89ABCDEF = vld1_u8(i2); i2 += 8;
    uint16x8_t vsum89ABCDEF = vaddl_u8(vi0x89ABCDEF, vi1x89ABCDEF);
    const uint8x8_t vi2xGHIJKLMN = vld1_u8(i2); i2 += 8;
    uint16x8_t vsumGHIJKLMN = vaddl_u8(vi0xGHIJKLMN, vi1xGHIJKLMN);

    const uint8x8_t vi3x01234567 = vld1_u8(i3); i3 += 8;
    vsum01234567 = vaddw_u8(vsum01234567, vi2x01234567);
    const uint8x8_t vi3x89ABCDEF = vld1_u8(i3); i3 += 8;
    vsum89ABCDEF = vaddw_u8(vsum89ABCDEF, vi2x89ABCDEF);
    const uint8x8_t vi3xGHIJKLMN = vld1_u8(i3); i3 += 8;
    vsumGHIJKLMN = vaddw_u8(vsumGHIJKLMN, vi2xGHIJKLMN);
    const uint8x8_t vi4x01234567 = vld1_u8(i4); i4 += 8;
    vsum01234567 = vaddw_u8(vsum01234567, vi3x01234567);
    const uint8x8_t vi4x89ABCDEF = vld1_u8(i4); i4 += 8;
    vsum89ABCDEF = vaddw_u8(vsum89ABCDEF, vi3x89ABCDEF);
    const uint8x8_t vi4xGHIJKLMN = vld1_u8(i4); i4 += 8;
    vsumGHIJKLMN = vaddw_u8(vsumGHIJKLMN, vi3xGHIJKLMN);
    const uint8x8_t vi5x01234567 = vld1_u8(i5); i5 += 8;
    vsum01234567 = vaddw_u8(vsum01234567, vi4x01234567);
    const uint8x8_t vi5x89ABCDEF = vld1_u8(i5); i5 += 8;
    vsum89ABCDEF = vaddw_u8(vsum89ABCDEF, vi4x89ABCDEF);
    const uint8x8_t vi5xGHIJKLMN = vld1_u8(i5); i5 += 8;
    vsumGHIJKLMN = vaddw_u8(vsumGHIJKLMN, vi4xGHIJKLMN);
    const uint8x8_t vi6x01234567 = vld1_u8(i6); i6 += 8;
    vsum01234567 = vaddw_u8(vsum01234567, vi5x01234567);
    const uint8x8_t vi6x89ABCDEF = vld1_u8(i6); i6 += 8;
    vsum89ABCDEF = vaddw_u8(vsum89ABCDEF, vi5x89ABCDEF);
    const uint8x8_t vi6xGHIJKLMN = vld1_u8(i6); i6 += 8;
    vsumGHIJKLMN = vaddw_u8(vsumGHIJKLMN, vi5xGHIJKLMN);
    int32x4_t vacc0123 = vld1q_s32(buffer); buffer += 4;
    int32x4_t vacc4567 = vld1q_s32(buffer); buffer += 4;
    vsum01234567 = vaddw_u8(vsum01234567, vi6x01234567);
    int32x4_t vacc89AB = vld1q_s32(buffer); buffer += 4;
    int32x4_t vaccCDEF = vld1q_s32(buffer); buffer += 4;
    vsum89ABCDEF = vaddw_u8(vsum89ABCDEF, vi6x89ABCDEF);
    int32x4_t vaccGHIJ = vld1q_s32(buffer); buffer += 4;
    int32x4_t vaccKLMN = vld1q_s32(buffer); buffer += 4;
    vsumGHIJKLMN = vaddw_u8(vsumGHIJKLMN, vi6xGHIJKLMN);

    vacc0123 = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vacc0123), vget_low_u16(vsum01234567)));
    vacc4567 = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vacc4567), vget_high_u16(vsum01234567)));
    vacc89AB = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vacc89AB), vget_low_u16(vsum89ABCDEF)));
    vaccCDEF = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vaccCDEF), vget_high_u16(vsum89ABCDEF)));
    vaccGHIJ = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vaccGHIJ), vget_low_u16(vsumGHIJKLMN)));
    vaccKLMN = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vaccKLMN), vget_high_u16(vsumGHIJKLMN)));

    float32x4_t vfpacc0123 = vcvtq_f32_s32(vacc0123);
    float32x4_t vfpacc4567 = vcvtq_f32_s32(vacc4567);
    float32x4_t vfpacc89AB = vcvtq_f32_s32(vacc89AB);
    float32x4_t vfpaccCDEF = vcvtq_f32_s32(vaccCDEF);
    float32x4_t vfpaccGHIJ = vcvtq_f32_s32(vaccGHIJ);
    float32x4_t vfpaccKLMN = vcvtq_f32_s32(vaccKLMN);

    vfpacc0123 = vmulq_f32(vfpacc0123, vscale);
    vfpacc4567 = vmulq_f32(vfpacc4567, vscale);
    vfpacc89AB = vmulq_f32(vfpacc89AB, vscale);
    vfpaccCDEF = vmulq_f32(vfpaccCDEF, vscale);
    vfpaccGHIJ = vmulq_f32(vfpaccGHIJ, vscale);
    vfpaccKLMN = vmulq_f32(vfpaccKLMN, vscale);

    vacc0123 = vcvtnq_s32_f32(vfpacc0123);
    vacc4567 = vcvtnq_s32_f32(vfpacc4567);
    vacc89AB = vcvtnq_s32_f32(vfpacc89AB);
    vaccCDEF = vcvtnq_s32_f32(vfpaccCDEF);
    vaccGHIJ = vcvtnq_s32_f32(vfpaccGHIJ);
    vaccKLMN = vcvtnq_s32_f32(vfpaccKLMN);

    #if XNN_ARCH_ARM64
      int16x8_t vacc01234567 = vqmovn_high_s32(vqmovn_s32(vacc0123), vacc4567);
      int16x8_t vacc89ABCDEF = vqmovn_high_s32(vqmovn_s32(vacc89AB), vaccCDEF);
      int16x8_t vaccGHIJKLMN = vqmovn_high_s32(vqmovn_s32(vaccGHIJ), vaccKLMN);
    #else  // !XNN_ARCH_ARM64
      int16x8_t vacc01234567 = vcombine_s16(vqmovn_s32(vacc0123), vqmovn_s32(vacc4567));
      int16x8_t vacc89ABCDEF = vcombine_s16(vqmovn_s32(vacc89AB), vqmovn_s32(vaccCDEF));
      int16x8_t vaccGHIJKLMN = vcombine_s16(vqmovn_s32(vaccGHIJ), vqmovn_s32(vaccKLMN));
    #endif  // !XNN_ARCH_ARM64

    vacc01234567 = vqaddq_s16(vacc01234567, voutput_zero_point);
    vacc89ABCDEF = vqaddq_s16(vacc89ABCDEF, voutput_zero_point);
    vaccGHIJKLMN = vqaddq_s16(vaccGHIJKLMN, voutput_zero_point);

    #if XNN_ARCH_ARM64
      uint8x16_t vout0123456789ABCDEF = vqmovun_high_s16(vqmovun_s16(vacc01234567), vacc89ABCDEF);
      uint8x8_t voutGHIJKLMN = vqmovun_s16(vaccGHIJKLMN);
    #else  // !XNN_ARCH_ARM64
      uint8x16_t vout0123456789ABCDEF = vcombine_u8(vqmovun_s16(vacc01234567), vqmovun_s16(vacc89ABCDEF));
      uint8x8_t voutGHIJKLMN = vqmovun_s16(vaccGHIJKLMN);
    #endif  // !XNN_ARCH_ARM64

    vout0123456789ABCDEF = vmaxq_u8(vout0123456789ABCDEF, voutput_min);
    voutGHIJKLMN = vmax_u8(voutGHIJKLMN, vget_low_u8(voutput_min));

    vout0123456789ABCDEF = vminq_u8(vout0123456789ABCDEF, voutput_max);
    voutGHIJKLMN = vmin_u8(voutGHIJKLMN, vget_low_u8(voutput_max));

    vst1q_u8(output, vout0123456789ABCDEF); output += 16;
    vst1_u8(output, voutGHIJKLMN); output += 8;
  }
  if XNN_UNLIKELY(channels != 0) {
    do {
      const uint8x8_t vi0x01234567 = vld1_u8(i0); i0 += 8;
      const uint8x8_t vi1x01234567 = vld1_u8(i1); i1 += 8;
      const uint8x8_t vi2x01234567 = vld1_u8(i2); i2 += 8;
      uint16x8_t vsum01234567 = vaddl_u8(vi0x01234567, vi1x01234567);

      const uint8x8_t vi3x01234567 = vld1_u8(i3); i3 += 8;
      vsum01234567 = vaddw_u8(vsum01234567, vi2x01234567);
      const uint8x8_t vi4x01234567 = vld1_u8(i4); i4 += 8;
      vsum01234567 = vaddw_u8(vsum01234567, vi3x01234567);
      const uint8x8_t vi5x01234567 = vld1_u8(i5); i5 += 8;
      vsum01234567 = vaddw_u8(vsum01234567, vi4x01234567);
      const uint8x8_t vi6x01234567 = vld1_u8(i6); i6 += 8;
      vsum01234567 = vaddw_u8(vsum01234567, vi5x01234567);
      int32x4_t vacc0123 = vld1q_s32(buffer); buffer += 4;
      int32x4_t vacc4567 = vld1q_s32(buffer); buffer += 4;
      vsum01234567 = vaddw_u8(vsum01234567, vi6x01234567);

      vacc0123 = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vacc0123), vget_low_u16(vsum01234567)));
      vacc4567 = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vacc4567), vget_high_u16(vsum01234567)));

      float32x4_t vfpacc0123 = vcvtq_f32_s32(vacc0123);
      float32x4_t vfpacc4567 = vcvtq_f32_s32(vacc4567);

      vfpacc0123 = vmulq_f32(vfpacc0123, vscale);
      vfpacc4567 = vmulq_f32(vfpacc4567, vscale);

      vacc0123 = vcvtnq_s32_f32(vfpacc0123);
      vacc4567 = vcvtnq_s32_f32(vfpacc4567);

      #if XNN_ARCH_ARM64
        int16x8_t vacc01234567 = vqmovn_high_s32(vqmovn_s32(vacc0123), vacc4567);
      #else
        int16x8_t vacc01234567 = vcombine_s16(vqmovn_s32(vacc0123), vqmovn_s32(vacc4567));
      #endif
      vacc01234567 = vqaddq_s16(vacc01234567, voutput_zero_point);

      uint8x8_t vout01234567 = vqmovun_s16(vacc01234567);
      vout01234567 = vmax_u8(vout01234567, vget_low_u8(voutput_min));
      vout01234567 = vmin_u8(vout01234567, vget_low_u8(voutput_max));

      if XNN_LIKELY(channels >= 8) {
        vst1_u8(output, vout01234567); output += 8;
        channels -= 8;
      } else {
        if (channels & 4) {
          vst1_lane_u32((void*) output, vreinterpret_u32_u8(vout01234567), 0); output += 4;
          vout01234567 = vext_u8(vout01234567, vout01234567, 4);
        }
        if (channels & 2) {
          vst1_lane_u16((void*) output, vreinterpret_u16_u8(vout01234567), 0); output += 2;
          vout01234567 = vext_u8(vout01234567, vout01234567, 2);
        }
        if (channels & 1) {
          vst1_lane_u8(output, vout01234567, 0); output += 1;
        }
        channels = 0;
      }
    } while (channels != 0);
  }
}
