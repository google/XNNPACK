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
#include "xnnpack/math.h"


void xnn_qu8_gavgpool_minmax_fp32_ukernel_7p7x__neon_c32(
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

  const int32x4_t vinit_bias = vld1q_dup_s32(&params->fp32_neon.init_bias);
  int32_t* b = buffer;
  size_t c = channels;
  for (; c >= 32; c -= 32) {
    const uint8x8_t vi0x01234567 = vld1_u8(i0); i0 += 8;
    const uint8x8_t vi0x89ABCDEF = vld1_u8(i0); i0 += 8;
    const uint8x8_t vi0xGHIJKLMN = vld1_u8(i0); i0 += 8;
    const uint8x8_t vi0xOPQRSTUV = vld1_u8(i0); i0 += 8;
    const uint8x8_t vi1x01234567 = vld1_u8(i1); i1 += 8;
    const uint8x8_t vi1x89ABCDEF = vld1_u8(i1); i1 += 8;
    const uint8x8_t vi1xGHIJKLMN = vld1_u8(i1); i1 += 8;
    const uint8x8_t vi1xOPQRSTUV = vld1_u8(i1); i1 += 8;

    const uint8x8_t vi2x01234567 = vld1_u8(i2); i2 += 8;
    uint16x8_t vsum01234567 = vaddl_u8(vi0x01234567, vi1x01234567);
    const uint8x8_t vi2x89ABCDEF = vld1_u8(i2); i2 += 8;
    uint16x8_t vsum89ABCDEF = vaddl_u8(vi0x89ABCDEF, vi1x89ABCDEF);
    const uint8x8_t vi2xGHIJKLMN = vld1_u8(i2); i2 += 8;
    uint16x8_t vsumGHIJKLMN = vaddl_u8(vi0xGHIJKLMN, vi1xGHIJKLMN);
    const uint8x8_t vi2xOPQRSTUV = vld1_u8(i2); i2 += 8;
    uint16x8_t vsumOPQRSTUV = vaddl_u8(vi0xOPQRSTUV, vi1xOPQRSTUV);

    const uint8x8_t vi3x01234567 = vld1_u8(i3); i3 += 8;
    vsum01234567 = vaddw_u8(vsum01234567, vi2x01234567);
    const uint8x8_t vi3x89ABCDEF = vld1_u8(i3); i3 += 8;
    vsum89ABCDEF = vaddw_u8(vsum89ABCDEF, vi2x89ABCDEF);
    const uint8x8_t vi3xGHIJKLMN = vld1_u8(i3); i3 += 8;
    vsumGHIJKLMN = vaddw_u8(vsumGHIJKLMN, vi2xGHIJKLMN);
    const uint8x8_t vi3xOPQRSTUV = vld1_u8(i3); i3 += 8;
    vsumOPQRSTUV = vaddw_u8(vsumOPQRSTUV, vi2xOPQRSTUV);
    const uint8x8_t vi4x01234567 = vld1_u8(i4); i4 += 8;
    vsum01234567 = vaddw_u8(vsum01234567, vi3x01234567);
    const uint8x8_t vi4x89ABCDEF = vld1_u8(i4); i4 += 8;
    vsum89ABCDEF = vaddw_u8(vsum89ABCDEF, vi3x89ABCDEF);
    const uint8x8_t vi4xGHIJKLMN = vld1_u8(i4); i4 += 8;
    vsumGHIJKLMN = vaddw_u8(vsumGHIJKLMN, vi3xGHIJKLMN);
    const uint8x8_t vi4xOPQRSTUV = vld1_u8(i4); i4 += 8;
    vsumOPQRSTUV = vaddw_u8(vsumOPQRSTUV, vi3xOPQRSTUV);
    const uint8x8_t vi5x01234567 = vld1_u8(i5); i5 += 8;
    vsum01234567 = vaddw_u8(vsum01234567, vi4x01234567);
    const uint8x8_t vi5x89ABCDEF = vld1_u8(i5); i5 += 8;
    vsum89ABCDEF = vaddw_u8(vsum89ABCDEF, vi4x89ABCDEF);
    const uint8x8_t vi5xGHIJKLMN = vld1_u8(i5); i5 += 8;
    vsumGHIJKLMN = vaddw_u8(vsumGHIJKLMN, vi4xGHIJKLMN);
    const uint8x8_t vi5xOPQRSTUV = vld1_u8(i5); i5 += 8;
    vsumOPQRSTUV = vaddw_u8(vsumOPQRSTUV, vi4xOPQRSTUV);
    const uint8x8_t vi6x01234567 = vld1_u8(i6); i6 += 8;
    vsum01234567 = vaddw_u8(vsum01234567, vi5x01234567);
    const uint8x8_t vi6x89ABCDEF = vld1_u8(i6); i6 += 8;
    vsum89ABCDEF = vaddw_u8(vsum89ABCDEF, vi5x89ABCDEF);
    const uint8x8_t vi6xGHIJKLMN = vld1_u8(i6); i6 += 8;
    vsumGHIJKLMN = vaddw_u8(vsumGHIJKLMN, vi5xGHIJKLMN);
    const uint8x8_t vi6xOPQRSTUV = vld1_u8(i6); i6 += 8;
    vsumOPQRSTUV = vaddw_u8(vsumOPQRSTUV, vi5xOPQRSTUV);
    vsum01234567 = vaddw_u8(vsum01234567, vi6x01234567);
    vsum89ABCDEF = vaddw_u8(vsum89ABCDEF, vi6x89ABCDEF);
    vsumGHIJKLMN = vaddw_u8(vsumGHIJKLMN, vi6xGHIJKLMN);
    vsumOPQRSTUV = vaddw_u8(vsumOPQRSTUV, vi6xOPQRSTUV);

    const int32x4_t vacc0123 = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vinit_bias), vget_low_u16(vsum01234567)));
    const int32x4_t vacc4567 = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vinit_bias), vget_high_u16(vsum01234567)));
    const int32x4_t vacc89AB = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vinit_bias), vget_low_u16(vsum89ABCDEF)));
    const int32x4_t vaccCDEF = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vinit_bias), vget_high_u16(vsum89ABCDEF)));
    const int32x4_t vaccGHIJ = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vinit_bias), vget_low_u16(vsumGHIJKLMN)));
    const int32x4_t vaccKLMN = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vinit_bias), vget_high_u16(vsumGHIJKLMN)));
    const int32x4_t vaccOPQR = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vinit_bias), vget_low_u16(vsumOPQRSTUV)));
    const int32x4_t vaccSTUV = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vinit_bias), vget_high_u16(vsumOPQRSTUV)));

    vst1q_s32(b, vacc0123); b += 4;
    vst1q_s32(b, vacc4567); b += 4;
    vst1q_s32(b, vacc89AB); b += 4;
    vst1q_s32(b, vaccCDEF); b += 4;
    vst1q_s32(b, vaccGHIJ); b += 4;
    vst1q_s32(b, vaccKLMN); b += 4;
    vst1q_s32(b, vaccOPQR); b += 4;
    vst1q_s32(b, vaccSTUV); b += 4;
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
    for (; c >= 32; c -= 32) {
      const uint8x8_t vi0x01234567 = vld1_u8(i0); i0 += 8;
      const uint8x8_t vi0x89ABCDEF = vld1_u8(i0); i0 += 8;
      const uint8x8_t vi0xGHIJKLMN = vld1_u8(i0); i0 += 8;
      const uint8x8_t vi0xOPQRSTUV = vld1_u8(i0); i0 += 8;
      const uint8x8_t vi1x01234567 = vld1_u8(i1); i1 += 8;
      const uint8x8_t vi1x89ABCDEF = vld1_u8(i1); i1 += 8;
      const uint8x8_t vi1xGHIJKLMN = vld1_u8(i1); i1 += 8;
      const uint8x8_t vi1xOPQRSTUV = vld1_u8(i1); i1 += 8;

      const uint8x8_t vi2x01234567 = vld1_u8(i2); i2 += 8;
      uint16x8_t vsum01234567 = vaddl_u8(vi0x01234567, vi1x01234567);
      const uint8x8_t vi2x89ABCDEF = vld1_u8(i2); i2 += 8;
      uint16x8_t vsum89ABCDEF = vaddl_u8(vi0x89ABCDEF, vi1x89ABCDEF);
      const uint8x8_t vi2xGHIJKLMN = vld1_u8(i2); i2 += 8;
      uint16x8_t vsumGHIJKLMN = vaddl_u8(vi0xGHIJKLMN, vi1xGHIJKLMN);
      const uint8x8_t vi2xOPQRSTUV = vld1_u8(i2); i2 += 8;
      uint16x8_t vsumOPQRSTUV = vaddl_u8(vi0xOPQRSTUV, vi1xOPQRSTUV);

      const uint8x8_t vi3x01234567 = vld1_u8(i3); i3 += 8;
      vsum01234567 = vaddw_u8(vsum01234567, vi2x01234567);
      const uint8x8_t vi3x89ABCDEF = vld1_u8(i3); i3 += 8;
      vsum89ABCDEF = vaddw_u8(vsum89ABCDEF, vi2x89ABCDEF);
      const uint8x8_t vi3xGHIJKLMN = vld1_u8(i3); i3 += 8;
      vsumGHIJKLMN = vaddw_u8(vsumGHIJKLMN, vi2xGHIJKLMN);
      const uint8x8_t vi3xOPQRSTUV = vld1_u8(i3); i3 += 8;
      vsumOPQRSTUV = vaddw_u8(vsumOPQRSTUV, vi2xOPQRSTUV);
      const uint8x8_t vi4x01234567 = vld1_u8(i4); i4 += 8;
      vsum01234567 = vaddw_u8(vsum01234567, vi3x01234567);
      const uint8x8_t vi4x89ABCDEF = vld1_u8(i4); i4 += 8;
      vsum89ABCDEF = vaddw_u8(vsum89ABCDEF, vi3x89ABCDEF);
      const uint8x8_t vi4xGHIJKLMN = vld1_u8(i4); i4 += 8;
      vsumGHIJKLMN = vaddw_u8(vsumGHIJKLMN, vi3xGHIJKLMN);
      const uint8x8_t vi4xOPQRSTUV = vld1_u8(i4); i4 += 8;
      vsumOPQRSTUV = vaddw_u8(vsumOPQRSTUV, vi3xOPQRSTUV);
      const uint8x8_t vi5x01234567 = vld1_u8(i5); i5 += 8;
      vsum01234567 = vaddw_u8(vsum01234567, vi4x01234567);
      const uint8x8_t vi5x89ABCDEF = vld1_u8(i5); i5 += 8;
      vsum89ABCDEF = vaddw_u8(vsum89ABCDEF, vi4x89ABCDEF);
      const uint8x8_t vi5xGHIJKLMN = vld1_u8(i5); i5 += 8;
      vsumGHIJKLMN = vaddw_u8(vsumGHIJKLMN, vi4xGHIJKLMN);
      const uint8x8_t vi5xOPQRSTUV = vld1_u8(i5); i5 += 8;
      vsumOPQRSTUV = vaddw_u8(vsumOPQRSTUV, vi4xOPQRSTUV);
      const uint8x8_t vi6x01234567 = vld1_u8(i6); i6 += 8;
      vsum01234567 = vaddw_u8(vsum01234567, vi5x01234567);
      const uint8x8_t vi6x89ABCDEF = vld1_u8(i6); i6 += 8;
      vsum89ABCDEF = vaddw_u8(vsum89ABCDEF, vi5x89ABCDEF);
      const uint8x8_t vi6xGHIJKLMN = vld1_u8(i6); i6 += 8;
      vsumGHIJKLMN = vaddw_u8(vsumGHIJKLMN, vi5xGHIJKLMN);
      const uint8x8_t vi6xOPQRSTUV = vld1_u8(i6); i6 += 8;
      vsumOPQRSTUV = vaddw_u8(vsumOPQRSTUV, vi5xOPQRSTUV);
      int32x4_t vacc0123 = vld1q_s32(b);
      int32x4_t vacc4567 = vld1q_s32(b + 4);
      vsum01234567 = vaddw_u8(vsum01234567, vi6x01234567);
      int32x4_t vacc89AB = vld1q_s32(b + 8);
      int32x4_t vaccCDEF = vld1q_s32(b + 12);
      vsum89ABCDEF = vaddw_u8(vsum89ABCDEF, vi6x89ABCDEF);
      int32x4_t vaccGHIJ = vld1q_s32(b + 16);
      int32x4_t vaccKLMN = vld1q_s32(b + 20);
      vsumGHIJKLMN = vaddw_u8(vsumGHIJKLMN, vi6xGHIJKLMN);
      int32x4_t vaccOPQR = vld1q_s32(b + 24);
      int32x4_t vaccSTUV = vld1q_s32(b + 28);
      vsumOPQRSTUV = vaddw_u8(vsumOPQRSTUV, vi6xOPQRSTUV);

      vacc0123 = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vacc0123), vget_low_u16(vsum01234567)));
      vacc4567 = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vacc4567), vget_high_u16(vsum01234567)));
      vacc89AB = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vacc89AB), vget_low_u16(vsum89ABCDEF)));
      vaccCDEF = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vaccCDEF), vget_high_u16(vsum89ABCDEF)));
      vaccGHIJ = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vaccGHIJ), vget_low_u16(vsumGHIJKLMN)));
      vaccKLMN = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vaccKLMN), vget_high_u16(vsumGHIJKLMN)));
      vaccOPQR = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vaccOPQR), vget_low_u16(vsumOPQRSTUV)));
      vaccSTUV = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vaccSTUV), vget_high_u16(vsumOPQRSTUV)));

      vst1q_s32(b, vacc0123); b += 4;
      vst1q_s32(b, vacc4567); b += 4;
      vst1q_s32(b, vacc89AB); b += 4;
      vst1q_s32(b, vaccCDEF); b += 4;
      vst1q_s32(b, vaccGHIJ); b += 4;
      vst1q_s32(b, vaccKLMN); b += 4;
      vst1q_s32(b, vaccOPQR); b += 4;
      vst1q_s32(b, vaccSTUV); b += 4;
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

  const float32x4_t vscale = vld1q_dup_f32(&params->fp32_neon.scale);
  const float32x4_t vmagic_bias = vld1q_dup_f32(&params->fp32_neon.magic_bias);
  const int32x4_t vmagic_bias_less_output_zero_point = vld1q_dup_s32(&params->fp32_neon.magic_bias_less_output_zero_point);
  const uint8x16_t voutput_min = vld1q_dup_u8(&params->fp32_neon.output_min);
  const uint8x16_t voutput_max = vld1q_dup_u8(&params->fp32_neon.output_max);
  for (; channels >= 32; channels -= 32) {
    const uint8x8_t vi0x01234567 = vld1_u8(i0); i0 += 8;
    const uint8x8_t vi0x89ABCDEF = vld1_u8(i0); i0 += 8;
    const uint8x8_t vi0xGHIJKLMN = vld1_u8(i0); i0 += 8;
    const uint8x8_t vi0xOPQRSTUV = vld1_u8(i0); i0 += 8;
    const uint8x8_t vi1x01234567 = vld1_u8(i1); i1 += 8;
    const uint8x8_t vi1x89ABCDEF = vld1_u8(i1); i1 += 8;
    const uint8x8_t vi1xGHIJKLMN = vld1_u8(i1); i1 += 8;
    const uint8x8_t vi1xOPQRSTUV = vld1_u8(i1); i1 += 8;

    const uint8x8_t vi2x01234567 = vld1_u8(i2); i2 += 8;
    uint16x8_t vsum01234567 = vaddl_u8(vi0x01234567, vi1x01234567);
    const uint8x8_t vi2x89ABCDEF = vld1_u8(i2); i2 += 8;
    uint16x8_t vsum89ABCDEF = vaddl_u8(vi0x89ABCDEF, vi1x89ABCDEF);
    const uint8x8_t vi2xGHIJKLMN = vld1_u8(i2); i2 += 8;
    uint16x8_t vsumGHIJKLMN = vaddl_u8(vi0xGHIJKLMN, vi1xGHIJKLMN);
    const uint8x8_t vi2xOPQRSTUV = vld1_u8(i2); i2 += 8;
    uint16x8_t vsumOPQRSTUV = vaddl_u8(vi0xOPQRSTUV, vi1xOPQRSTUV);

    const uint8x8_t vi3x01234567 = vld1_u8(i3); i3 += 8;
    vsum01234567 = vaddw_u8(vsum01234567, vi2x01234567);
    const uint8x8_t vi3x89ABCDEF = vld1_u8(i3); i3 += 8;
    vsum89ABCDEF = vaddw_u8(vsum89ABCDEF, vi2x89ABCDEF);
    const uint8x8_t vi3xGHIJKLMN = vld1_u8(i3); i3 += 8;
    vsumGHIJKLMN = vaddw_u8(vsumGHIJKLMN, vi2xGHIJKLMN);
    const uint8x8_t vi3xOPQRSTUV = vld1_u8(i3); i3 += 8;
    vsumOPQRSTUV = vaddw_u8(vsumOPQRSTUV, vi2xOPQRSTUV);
    const uint8x8_t vi4x01234567 = vld1_u8(i4); i4 += 8;
    vsum01234567 = vaddw_u8(vsum01234567, vi3x01234567);
    const uint8x8_t vi4x89ABCDEF = vld1_u8(i4); i4 += 8;
    vsum89ABCDEF = vaddw_u8(vsum89ABCDEF, vi3x89ABCDEF);
    const uint8x8_t vi4xGHIJKLMN = vld1_u8(i4); i4 += 8;
    vsumGHIJKLMN = vaddw_u8(vsumGHIJKLMN, vi3xGHIJKLMN);
    const uint8x8_t vi4xOPQRSTUV = vld1_u8(i4); i4 += 8;
    vsumOPQRSTUV = vaddw_u8(vsumOPQRSTUV, vi3xOPQRSTUV);
    const uint8x8_t vi5x01234567 = vld1_u8(i5); i5 += 8;
    vsum01234567 = vaddw_u8(vsum01234567, vi4x01234567);
    const uint8x8_t vi5x89ABCDEF = vld1_u8(i5); i5 += 8;
    vsum89ABCDEF = vaddw_u8(vsum89ABCDEF, vi4x89ABCDEF);
    const uint8x8_t vi5xGHIJKLMN = vld1_u8(i5); i5 += 8;
    vsumGHIJKLMN = vaddw_u8(vsumGHIJKLMN, vi4xGHIJKLMN);
    const uint8x8_t vi5xOPQRSTUV = vld1_u8(i5); i5 += 8;
    vsumOPQRSTUV = vaddw_u8(vsumOPQRSTUV, vi4xOPQRSTUV);
    const uint8x8_t vi6x01234567 = vld1_u8(i6); i6 += 8;
    vsum01234567 = vaddw_u8(vsum01234567, vi5x01234567);
    const uint8x8_t vi6x89ABCDEF = vld1_u8(i6); i6 += 8;
    vsum89ABCDEF = vaddw_u8(vsum89ABCDEF, vi5x89ABCDEF);
    const uint8x8_t vi6xGHIJKLMN = vld1_u8(i6); i6 += 8;
    vsumGHIJKLMN = vaddw_u8(vsumGHIJKLMN, vi5xGHIJKLMN);
    const uint8x8_t vi6xOPQRSTUV = vld1_u8(i6); i6 += 8;
    vsumOPQRSTUV = vaddw_u8(vsumOPQRSTUV, vi5xOPQRSTUV);
    int32x4_t vacc0123 = vld1q_s32(buffer); buffer += 4;
    int32x4_t vacc4567 = vld1q_s32(buffer); buffer += 4;
    vsum01234567 = vaddw_u8(vsum01234567, vi6x01234567);
    int32x4_t vacc89AB = vld1q_s32(buffer); buffer += 4;
    int32x4_t vaccCDEF = vld1q_s32(buffer); buffer += 4;
    vsum89ABCDEF = vaddw_u8(vsum89ABCDEF, vi6x89ABCDEF);
    int32x4_t vaccGHIJ = vld1q_s32(buffer); buffer += 4;
    int32x4_t vaccKLMN = vld1q_s32(buffer); buffer += 4;
    vsumGHIJKLMN = vaddw_u8(vsumGHIJKLMN, vi6xGHIJKLMN);
    int32x4_t vaccOPQR = vld1q_s32(buffer); buffer += 4;
    int32x4_t vaccSTUV = vld1q_s32(buffer); buffer += 4;
    vsumOPQRSTUV = vaddw_u8(vsumOPQRSTUV, vi6xOPQRSTUV);

    vacc0123 = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vacc0123), vget_low_u16(vsum01234567)));
    vacc4567 = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vacc4567), vget_high_u16(vsum01234567)));
    vacc89AB = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vacc89AB), vget_low_u16(vsum89ABCDEF)));
    vaccCDEF = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vaccCDEF), vget_high_u16(vsum89ABCDEF)));
    vaccGHIJ = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vaccGHIJ), vget_low_u16(vsumGHIJKLMN)));
    vaccKLMN = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vaccKLMN), vget_high_u16(vsumGHIJKLMN)));
    vaccOPQR = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vaccOPQR), vget_low_u16(vsumOPQRSTUV)));
    vaccSTUV = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vaccSTUV), vget_high_u16(vsumOPQRSTUV)));

    float32x4_t vfpacc0123 = vcvtq_f32_s32(vacc0123);
    float32x4_t vfpacc4567 = vcvtq_f32_s32(vacc4567);
    float32x4_t vfpacc89AB = vcvtq_f32_s32(vacc89AB);
    float32x4_t vfpaccCDEF = vcvtq_f32_s32(vaccCDEF);
    float32x4_t vfpaccGHIJ = vcvtq_f32_s32(vaccGHIJ);
    float32x4_t vfpaccKLMN = vcvtq_f32_s32(vaccKLMN);
    float32x4_t vfpaccOPQR = vcvtq_f32_s32(vaccOPQR);
    float32x4_t vfpaccSTUV = vcvtq_f32_s32(vaccSTUV);

    vfpacc0123 = vmulq_f32(vfpacc0123, vscale);
    vfpacc4567 = vmulq_f32(vfpacc4567, vscale);
    vfpacc89AB = vmulq_f32(vfpacc89AB, vscale);
    vfpaccCDEF = vmulq_f32(vfpaccCDEF, vscale);
    vfpaccGHIJ = vmulq_f32(vfpaccGHIJ, vscale);
    vfpaccKLMN = vmulq_f32(vfpaccKLMN, vscale);
    vfpaccOPQR = vmulq_f32(vfpaccOPQR, vscale);
    vfpaccSTUV = vmulq_f32(vfpaccSTUV, vscale);

    vacc0123 = vreinterpretq_s32_f32(vaddq_f32(vfpacc0123, vmagic_bias));
    vacc4567 = vreinterpretq_s32_f32(vaddq_f32(vfpacc4567, vmagic_bias));
    vacc89AB = vreinterpretq_s32_f32(vaddq_f32(vfpacc89AB, vmagic_bias));
    vaccCDEF = vreinterpretq_s32_f32(vaddq_f32(vfpaccCDEF, vmagic_bias));
    vaccGHIJ = vreinterpretq_s32_f32(vaddq_f32(vfpaccGHIJ, vmagic_bias));
    vaccKLMN = vreinterpretq_s32_f32(vaddq_f32(vfpaccKLMN, vmagic_bias));
    vaccOPQR = vreinterpretq_s32_f32(vaddq_f32(vfpaccOPQR, vmagic_bias));
    vaccSTUV = vreinterpretq_s32_f32(vaddq_f32(vfpaccSTUV, vmagic_bias));

    vacc0123 = vqsubq_s32(vacc0123, vmagic_bias_less_output_zero_point);
    vacc4567 = vqsubq_s32(vacc4567, vmagic_bias_less_output_zero_point);
    vacc89AB = vqsubq_s32(vacc89AB, vmagic_bias_less_output_zero_point);
    vaccCDEF = vqsubq_s32(vaccCDEF, vmagic_bias_less_output_zero_point);
    vaccGHIJ = vqsubq_s32(vaccGHIJ, vmagic_bias_less_output_zero_point);
    vaccKLMN = vqsubq_s32(vaccKLMN, vmagic_bias_less_output_zero_point);
    vaccOPQR = vqsubq_s32(vaccOPQR, vmagic_bias_less_output_zero_point);
    vaccSTUV = vqsubq_s32(vaccSTUV, vmagic_bias_less_output_zero_point);

    #if XNN_ARCH_ARM64
      int16x8_t vacc01234567 = vqmovn_high_s32(vqmovn_s32(vacc0123), vacc4567);
      int16x8_t vacc89ABCDEF = vqmovn_high_s32(vqmovn_s32(vacc89AB), vaccCDEF);
      int16x8_t vaccGHIJKLMN = vqmovn_high_s32(vqmovn_s32(vaccGHIJ), vaccKLMN);
      int16x8_t vaccOPQRSTUV = vqmovn_high_s32(vqmovn_s32(vaccOPQR), vaccSTUV);
    #else  // !XNN_ARCH_ARM64
      int16x8_t vacc01234567 = vcombine_s16(vqmovn_s32(vacc0123), vqmovn_s32(vacc4567));
      int16x8_t vacc89ABCDEF = vcombine_s16(vqmovn_s32(vacc89AB), vqmovn_s32(vaccCDEF));
      int16x8_t vaccGHIJKLMN = vcombine_s16(vqmovn_s32(vaccGHIJ), vqmovn_s32(vaccKLMN));
      int16x8_t vaccOPQRSTUV = vcombine_s16(vqmovn_s32(vaccOPQR), vqmovn_s32(vaccSTUV));
    #endif  // !XNN_ARCH_ARM64


    #if XNN_ARCH_ARM64
      uint8x16_t vout0123456789ABCDEF = vqmovun_high_s16(vqmovun_s16(vacc01234567), vacc89ABCDEF);
      uint8x16_t voutGHIJKLMNOPQRSTUV = vqmovun_high_s16(vqmovun_s16(vaccGHIJKLMN), vaccOPQRSTUV);
    #else  // !XNN_ARCH_ARM64
      uint8x16_t vout0123456789ABCDEF = vcombine_u8(vqmovun_s16(vacc01234567), vqmovun_s16(vacc89ABCDEF));
      uint8x16_t voutGHIJKLMNOPQRSTUV = vcombine_u8(vqmovun_s16(vaccGHIJKLMN), vqmovun_s16(vaccOPQRSTUV));
    #endif  // !XNN_ARCH_ARM64

    vout0123456789ABCDEF = vmaxq_u8(vout0123456789ABCDEF, voutput_min);
    voutGHIJKLMNOPQRSTUV = vmaxq_u8(voutGHIJKLMNOPQRSTUV, voutput_min);

    vout0123456789ABCDEF = vminq_u8(vout0123456789ABCDEF, voutput_max);
    voutGHIJKLMNOPQRSTUV = vminq_u8(voutGHIJKLMNOPQRSTUV, voutput_max);

    vst1q_u8(output, vout0123456789ABCDEF); output += 16;
    vst1q_u8(output, voutGHIJKLMNOPQRSTUV); output += 16;
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

      vacc0123 = vreinterpretq_s32_f32(vaddq_f32(vfpacc0123, vmagic_bias));
      vacc4567 = vreinterpretq_s32_f32(vaddq_f32(vfpacc4567, vmagic_bias));

      vacc0123 = vqsubq_s32(vacc0123, vmagic_bias_less_output_zero_point);
      vacc4567 = vqsubq_s32(vacc4567, vmagic_bias_less_output_zero_point);

      #if XNN_ARCH_ARM64
        int16x8_t vacc01234567 = vqmovn_high_s32(vqmovn_s32(vacc0123), vacc4567);
      #else
        int16x8_t vacc01234567 = vcombine_s16(vqmovn_s32(vacc0123), vqmovn_s32(vacc4567));
      #endif

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
