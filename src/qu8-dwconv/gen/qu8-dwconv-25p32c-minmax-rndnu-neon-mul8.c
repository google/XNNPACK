// Auto-generated file. Do not edit!
//   Template: src/qu8-dwconv/unipass-neon-mul8.c.in
//   Generator: tools/xngen
//
// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <arm_neon.h>

#include "xnnpack/dwconv.h"


void xnn_qu8_dwconv_minmax_rndnu_ukernel_25p32c__neon_mul8(
    size_t channels,
    size_t output_width,
    const uint8_t** input,
    const void* weights,
    uint8_t* output,
    intptr_t input_stride,
    size_t output_increment,
    size_t input_offset,
    const uint8_t* zero,
    const union xnn_qu8_conv_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(channels != 0);
  assert(output_width != 0);

  const uint16x8_t vkernel_zero_point = vmovl_u8(vld1_dup_u8(params->rndnu_neon.kernel_zero_point));
  const int32x4_t vright_pre_shift = vld1q_dup_s32(&params->rndnu_neon.right_pre_shift);
  const int32x4_t vmultiplier = vld1q_dup_s32(&params->rndnu_neon.multiplier);
  const int32x4_t vright_post_shift = vld1q_dup_s32(&params->rndnu_neon.right_post_shift);
  const int16x8_t voutput_zero_point = vld1q_dup_s16(&params->rndnu_neon.output_zero_point);
  const uint8x16_t voutput_min = vld1q_dup_u8(&params->rndnu_neon.output_min);
  const uint8x16_t voutput_max = vld1q_dup_u8(&params->rndnu_neon.output_max);
  do {
    const uint8_t* i0 = input[0];
    assert(i0 != NULL);
    if XNN_UNPREDICTABLE(i0 != zero) {
      i0 = (const uint8_t*) ((uintptr_t) i0 + input_offset);
    }
    const uint8_t* i1 = input[1];
    assert(i1 != NULL);
    if XNN_UNPREDICTABLE(i1 != zero) {
      i1 = (const uint8_t*) ((uintptr_t) i1 + input_offset);
    }
    const uint8_t* i2 = input[2];
    assert(i2 != NULL);
    if XNN_UNPREDICTABLE(i2 != zero) {
      i2 = (const uint8_t*) ((uintptr_t) i2 + input_offset);
    }
    const uint8_t* i3 = input[3];
    assert(i3 != NULL);
    if XNN_UNPREDICTABLE(i3 != zero) {
      i3 = (const uint8_t*) ((uintptr_t) i3 + input_offset);
    }
    const uint8_t* i4 = input[4];
    assert(i4 != NULL);
    if XNN_UNPREDICTABLE(i4 != zero) {
      i4 = (const uint8_t*) ((uintptr_t) i4 + input_offset);
    }
    const uint8_t* i5 = input[5];
    assert(i5 != NULL);
    if XNN_UNPREDICTABLE(i5 != zero) {
      i5 = (const uint8_t*) ((uintptr_t) i5 + input_offset);
    }
    const uint8_t* i6 = input[6];
    assert(i6 != NULL);
    if XNN_UNPREDICTABLE(i6 != zero) {
      i6 = (const uint8_t*) ((uintptr_t) i6 + input_offset);
    }
    const uint8_t* i7 = input[7];
    assert(i7 != NULL);
    if XNN_UNPREDICTABLE(i7 != zero) {
      i7 = (const uint8_t*) ((uintptr_t) i7 + input_offset);
    }
    const uint8_t* i8 = input[8];
    assert(i8 != NULL);
    if XNN_UNPREDICTABLE(i8 != zero) {
      i8 = (const uint8_t*) ((uintptr_t) i8 + input_offset);
    }
    const uint8_t* i9 = input[9];
    assert(i9 != NULL);
    if XNN_UNPREDICTABLE(i9 != zero) {
      i9 = (const uint8_t*) ((uintptr_t) i9 + input_offset);
    }
    const uint8_t* i10 = input[10];
    assert(i10 != NULL);
    if XNN_UNPREDICTABLE(i10 != zero) {
      i10 = (const uint8_t*) ((uintptr_t) i10 + input_offset);
    }
    const uint8_t* i11 = input[11];
    assert(i11 != NULL);
    if XNN_UNPREDICTABLE(i11 != zero) {
      i11 = (const uint8_t*) ((uintptr_t) i11 + input_offset);
    }
    const uint8_t* i12 = input[12];
    assert(i12 != NULL);
    if XNN_UNPREDICTABLE(i12 != zero) {
      i12 = (const uint8_t*) ((uintptr_t) i12 + input_offset);
    }
    const uint8_t* i13 = input[13];
    assert(i13 != NULL);
    if XNN_UNPREDICTABLE(i13 != zero) {
      i13 = (const uint8_t*) ((uintptr_t) i13 + input_offset);
    }
    const uint8_t* i14 = input[14];
    assert(i14 != NULL);
    if XNN_UNPREDICTABLE(i14 != zero) {
      i14 = (const uint8_t*) ((uintptr_t) i14 + input_offset);
    }
    const uint8_t* i15 = input[15];
    assert(i15 != NULL);
    if XNN_UNPREDICTABLE(i15 != zero) {
      i15 = (const uint8_t*) ((uintptr_t) i15 + input_offset);
    }
    const uint8_t* i16 = input[16];
    assert(i16 != NULL);
    if XNN_UNPREDICTABLE(i16 != zero) {
      i16 = (const uint8_t*) ((uintptr_t) i16 + input_offset);
    }
    const uint8_t* i17 = input[17];
    assert(i17 != NULL);
    if XNN_UNPREDICTABLE(i17 != zero) {
      i17 = (const uint8_t*) ((uintptr_t) i17 + input_offset);
    }
    const uint8_t* i18 = input[18];
    assert(i18 != NULL);
    if XNN_UNPREDICTABLE(i18 != zero) {
      i18 = (const uint8_t*) ((uintptr_t) i18 + input_offset);
    }
    const uint8_t* i19 = input[19];
    assert(i19 != NULL);
    if XNN_UNPREDICTABLE(i19 != zero) {
      i19 = (const uint8_t*) ((uintptr_t) i19 + input_offset);
    }
    const uint8_t* i20 = input[20];
    assert(i20 != NULL);
    if XNN_UNPREDICTABLE(i20 != zero) {
      i20 = (const uint8_t*) ((uintptr_t) i20 + input_offset);
    }
    const uint8_t* i21 = input[21];
    assert(i21 != NULL);
    if XNN_UNPREDICTABLE(i21 != zero) {
      i21 = (const uint8_t*) ((uintptr_t) i21 + input_offset);
    }
    const uint8_t* i22 = input[22];
    assert(i22 != NULL);
    if XNN_UNPREDICTABLE(i22 != zero) {
      i22 = (const uint8_t*) ((uintptr_t) i22 + input_offset);
    }
    const uint8_t* i23 = input[23];
    assert(i23 != NULL);
    if XNN_UNPREDICTABLE(i23 != zero) {
      i23 = (const uint8_t*) ((uintptr_t) i23 + input_offset);
    }
    const uint8_t* i24 = input[24];
    assert(i24 != NULL);
    if XNN_UNPREDICTABLE(i24 != zero) {
      i24 = (const uint8_t*) ((uintptr_t) i24 + input_offset);
    }
    input = (const uint8_t**) ((uintptr_t) input + input_stride);


    size_t c = channels;
    const void* w = weights;
    for (; c >= 32; c -= 32) {
      int32x4_t vacc0123 = vld1q_s32(w); w = (const int32_t*) w + 4;
      int32x4_t vacc4567 = vld1q_s32(w); w = (const int32_t*) w + 4;
      int32x4_t vacc89AB = vld1q_s32(w); w = (const int32_t*) w + 4;
      int32x4_t vaccCDEF = vld1q_s32(w); w = (const int32_t*) w + 4;
      int32x4_t vaccGHIJ = vld1q_s32(w); w = (const int32_t*) w + 4;
      int32x4_t vaccKLMN = vld1q_s32(w); w = (const int32_t*) w + 4;
      int32x4_t vaccOPQR = vld1q_s32(w); w = (const int32_t*) w + 4;
      int32x4_t vaccSTUV = vld1q_s32(w); w = (const int32_t*) w + 4;


      const uint8x8_t vi0x01234567 = vld1_u8(i0); i0 += 8;
      const uint8x8_t vk0x01234567 = vld1_u8(w); w = (const int8_t*) w + 8;
      const uint8x8_t vi0x89ABCDEF = vld1_u8(i0); i0 += 8;
      const uint8x8_t vk0x89ABCDEF = vld1_u8(w); w = (const int8_t*) w + 8;
      const uint8x8_t vi0xGHIJKLMN = vld1_u8(i0); i0 += 8;
      const uint8x8_t vk0xGHIJKLMN = vld1_u8(w); w = (const int8_t*) w + 8;
      const uint8x8_t vi0xOPQRSTUV = vld1_u8(i0); i0 += 8;
      const uint8x8_t vk0xOPQRSTUV = vld1_u8(w); w = (const int8_t*) w + 8;

      uint16x8_t vprod01234567 = vmull_u8(vi0x01234567, vk0x01234567);
      uint16x8_t vprod89ABCDEF = vmull_u8(vi0x89ABCDEF, vk0x89ABCDEF);
      uint16x8_t vprodGHIJKLMN = vmull_u8(vi0xGHIJKLMN, vk0xGHIJKLMN);
      uint16x8_t vprodOPQRSTUV = vmull_u8(vi0xOPQRSTUV, vk0xOPQRSTUV);

      vacc0123 = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vacc0123), vget_low_u16(vprod01234567)));
      vacc4567 = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vacc4567), vget_high_u16(vprod01234567)));
      vacc89AB = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vacc89AB), vget_low_u16(vprod89ABCDEF)));
      vaccCDEF = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vaccCDEF), vget_high_u16(vprod89ABCDEF)));
      vaccGHIJ = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vaccGHIJ), vget_low_u16(vprodGHIJKLMN)));
      vaccKLMN = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vaccKLMN), vget_high_u16(vprodGHIJKLMN)));
      vaccOPQR = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vaccOPQR), vget_low_u16(vprodOPQRSTUV)));
      vaccSTUV = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vaccSTUV), vget_high_u16(vprodOPQRSTUV)));
      const uint8x8_t vi1x01234567 = vld1_u8(i1); i1 += 8;
      const uint8x8_t vk1x01234567 = vld1_u8(w); w = (const int8_t*) w + 8;
      const uint8x8_t vi1x89ABCDEF = vld1_u8(i1); i1 += 8;
      const uint8x8_t vk1x89ABCDEF = vld1_u8(w); w = (const int8_t*) w + 8;
      const uint8x8_t vi1xGHIJKLMN = vld1_u8(i1); i1 += 8;
      const uint8x8_t vk1xGHIJKLMN = vld1_u8(w); w = (const int8_t*) w + 8;
      const uint8x8_t vi1xOPQRSTUV = vld1_u8(i1); i1 += 8;
      const uint8x8_t vk1xOPQRSTUV = vld1_u8(w); w = (const int8_t*) w + 8;

      vprod01234567 = vmull_u8(vi1x01234567, vk1x01234567);
      uint16x8_t vsum01234567 = vaddl_u8(vi0x01234567, vi1x01234567);
      vprod89ABCDEF = vmull_u8(vi1x89ABCDEF, vk1x89ABCDEF);
      uint16x8_t vsum89ABCDEF = vaddl_u8(vi0x89ABCDEF, vi1x89ABCDEF);
      vprodGHIJKLMN = vmull_u8(vi1xGHIJKLMN, vk1xGHIJKLMN);
      uint16x8_t vsumGHIJKLMN = vaddl_u8(vi0xGHIJKLMN, vi1xGHIJKLMN);
      vprodOPQRSTUV = vmull_u8(vi1xOPQRSTUV, vk1xOPQRSTUV);
      uint16x8_t vsumOPQRSTUV = vaddl_u8(vi0xOPQRSTUV, vi1xOPQRSTUV);

      vacc0123 = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vacc0123), vget_low_u16(vprod01234567)));
      vacc4567 = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vacc4567), vget_high_u16(vprod01234567)));
      vacc89AB = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vacc89AB), vget_low_u16(vprod89ABCDEF)));
      vaccCDEF = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vaccCDEF), vget_high_u16(vprod89ABCDEF)));
      vaccGHIJ = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vaccGHIJ), vget_low_u16(vprodGHIJKLMN)));
      vaccKLMN = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vaccKLMN), vget_high_u16(vprodGHIJKLMN)));
      vaccOPQR = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vaccOPQR), vget_low_u16(vprodOPQRSTUV)));
      vaccSTUV = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vaccSTUV), vget_high_u16(vprodOPQRSTUV)));
      const uint8x8_t vi2x01234567 = vld1_u8(i2); i2 += 8;
      const uint8x8_t vk2x01234567 = vld1_u8(w); w = (const int8_t*) w + 8;
      const uint8x8_t vi2x89ABCDEF = vld1_u8(i2); i2 += 8;
      const uint8x8_t vk2x89ABCDEF = vld1_u8(w); w = (const int8_t*) w + 8;
      const uint8x8_t vi2xGHIJKLMN = vld1_u8(i2); i2 += 8;
      const uint8x8_t vk2xGHIJKLMN = vld1_u8(w); w = (const int8_t*) w + 8;
      const uint8x8_t vi2xOPQRSTUV = vld1_u8(i2); i2 += 8;
      const uint8x8_t vk2xOPQRSTUV = vld1_u8(w); w = (const int8_t*) w + 8;

      vprod01234567 = vmull_u8(vi2x01234567, vk2x01234567);
      vsum01234567 = vaddw_u8(vsum01234567, vi2x01234567);
      vprod89ABCDEF = vmull_u8(vi2x89ABCDEF, vk2x89ABCDEF);
      vsum89ABCDEF = vaddw_u8(vsum89ABCDEF, vi2x89ABCDEF);
      vprodGHIJKLMN = vmull_u8(vi2xGHIJKLMN, vk2xGHIJKLMN);
      vsumGHIJKLMN = vaddw_u8(vsumGHIJKLMN, vi2xGHIJKLMN);
      vprodOPQRSTUV = vmull_u8(vi2xOPQRSTUV, vk2xOPQRSTUV);
      vsumOPQRSTUV = vaddw_u8(vsumOPQRSTUV, vi2xOPQRSTUV);

      vacc0123 = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vacc0123), vget_low_u16(vprod01234567)));
      vacc4567 = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vacc4567), vget_high_u16(vprod01234567)));
      vacc89AB = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vacc89AB), vget_low_u16(vprod89ABCDEF)));
      vaccCDEF = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vaccCDEF), vget_high_u16(vprod89ABCDEF)));
      vaccGHIJ = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vaccGHIJ), vget_low_u16(vprodGHIJKLMN)));
      vaccKLMN = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vaccKLMN), vget_high_u16(vprodGHIJKLMN)));
      vaccOPQR = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vaccOPQR), vget_low_u16(vprodOPQRSTUV)));
      vaccSTUV = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vaccSTUV), vget_high_u16(vprodOPQRSTUV)));
      const uint8x8_t vi3x01234567 = vld1_u8(i3); i3 += 8;
      const uint8x8_t vk3x01234567 = vld1_u8(w); w = (const int8_t*) w + 8;
      const uint8x8_t vi3x89ABCDEF = vld1_u8(i3); i3 += 8;
      const uint8x8_t vk3x89ABCDEF = vld1_u8(w); w = (const int8_t*) w + 8;
      const uint8x8_t vi3xGHIJKLMN = vld1_u8(i3); i3 += 8;
      const uint8x8_t vk3xGHIJKLMN = vld1_u8(w); w = (const int8_t*) w + 8;
      const uint8x8_t vi3xOPQRSTUV = vld1_u8(i3); i3 += 8;
      const uint8x8_t vk3xOPQRSTUV = vld1_u8(w); w = (const int8_t*) w + 8;

      vprod01234567 = vmull_u8(vi3x01234567, vk3x01234567);
      vsum01234567 = vaddw_u8(vsum01234567, vi3x01234567);
      vprod89ABCDEF = vmull_u8(vi3x89ABCDEF, vk3x89ABCDEF);
      vsum89ABCDEF = vaddw_u8(vsum89ABCDEF, vi3x89ABCDEF);
      vprodGHIJKLMN = vmull_u8(vi3xGHIJKLMN, vk3xGHIJKLMN);
      vsumGHIJKLMN = vaddw_u8(vsumGHIJKLMN, vi3xGHIJKLMN);
      vprodOPQRSTUV = vmull_u8(vi3xOPQRSTUV, vk3xOPQRSTUV);
      vsumOPQRSTUV = vaddw_u8(vsumOPQRSTUV, vi3xOPQRSTUV);

      vacc0123 = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vacc0123), vget_low_u16(vprod01234567)));
      vacc4567 = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vacc4567), vget_high_u16(vprod01234567)));
      vacc89AB = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vacc89AB), vget_low_u16(vprod89ABCDEF)));
      vaccCDEF = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vaccCDEF), vget_high_u16(vprod89ABCDEF)));
      vaccGHIJ = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vaccGHIJ), vget_low_u16(vprodGHIJKLMN)));
      vaccKLMN = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vaccKLMN), vget_high_u16(vprodGHIJKLMN)));
      vaccOPQR = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vaccOPQR), vget_low_u16(vprodOPQRSTUV)));
      vaccSTUV = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vaccSTUV), vget_high_u16(vprodOPQRSTUV)));
      const uint8x8_t vi4x01234567 = vld1_u8(i4); i4 += 8;
      const uint8x8_t vk4x01234567 = vld1_u8(w); w = (const int8_t*) w + 8;
      const uint8x8_t vi4x89ABCDEF = vld1_u8(i4); i4 += 8;
      const uint8x8_t vk4x89ABCDEF = vld1_u8(w); w = (const int8_t*) w + 8;
      const uint8x8_t vi4xGHIJKLMN = vld1_u8(i4); i4 += 8;
      const uint8x8_t vk4xGHIJKLMN = vld1_u8(w); w = (const int8_t*) w + 8;
      const uint8x8_t vi4xOPQRSTUV = vld1_u8(i4); i4 += 8;
      const uint8x8_t vk4xOPQRSTUV = vld1_u8(w); w = (const int8_t*) w + 8;

      vprod01234567 = vmull_u8(vi4x01234567, vk4x01234567);
      vsum01234567 = vaddw_u8(vsum01234567, vi4x01234567);
      vprod89ABCDEF = vmull_u8(vi4x89ABCDEF, vk4x89ABCDEF);
      vsum89ABCDEF = vaddw_u8(vsum89ABCDEF, vi4x89ABCDEF);
      vprodGHIJKLMN = vmull_u8(vi4xGHIJKLMN, vk4xGHIJKLMN);
      vsumGHIJKLMN = vaddw_u8(vsumGHIJKLMN, vi4xGHIJKLMN);
      vprodOPQRSTUV = vmull_u8(vi4xOPQRSTUV, vk4xOPQRSTUV);
      vsumOPQRSTUV = vaddw_u8(vsumOPQRSTUV, vi4xOPQRSTUV);

      vacc0123 = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vacc0123), vget_low_u16(vprod01234567)));
      vacc4567 = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vacc4567), vget_high_u16(vprod01234567)));
      vacc89AB = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vacc89AB), vget_low_u16(vprod89ABCDEF)));
      vaccCDEF = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vaccCDEF), vget_high_u16(vprod89ABCDEF)));
      vaccGHIJ = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vaccGHIJ), vget_low_u16(vprodGHIJKLMN)));
      vaccKLMN = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vaccKLMN), vget_high_u16(vprodGHIJKLMN)));
      vaccOPQR = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vaccOPQR), vget_low_u16(vprodOPQRSTUV)));
      vaccSTUV = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vaccSTUV), vget_high_u16(vprodOPQRSTUV)));
      const uint8x8_t vi5x01234567 = vld1_u8(i5); i5 += 8;
      const uint8x8_t vk5x01234567 = vld1_u8(w); w = (const int8_t*) w + 8;
      const uint8x8_t vi5x89ABCDEF = vld1_u8(i5); i5 += 8;
      const uint8x8_t vk5x89ABCDEF = vld1_u8(w); w = (const int8_t*) w + 8;
      const uint8x8_t vi5xGHIJKLMN = vld1_u8(i5); i5 += 8;
      const uint8x8_t vk5xGHIJKLMN = vld1_u8(w); w = (const int8_t*) w + 8;
      const uint8x8_t vi5xOPQRSTUV = vld1_u8(i5); i5 += 8;
      const uint8x8_t vk5xOPQRSTUV = vld1_u8(w); w = (const int8_t*) w + 8;

      vprod01234567 = vmull_u8(vi5x01234567, vk5x01234567);
      vsum01234567 = vaddw_u8(vsum01234567, vi5x01234567);
      vprod89ABCDEF = vmull_u8(vi5x89ABCDEF, vk5x89ABCDEF);
      vsum89ABCDEF = vaddw_u8(vsum89ABCDEF, vi5x89ABCDEF);
      vprodGHIJKLMN = vmull_u8(vi5xGHIJKLMN, vk5xGHIJKLMN);
      vsumGHIJKLMN = vaddw_u8(vsumGHIJKLMN, vi5xGHIJKLMN);
      vprodOPQRSTUV = vmull_u8(vi5xOPQRSTUV, vk5xOPQRSTUV);
      vsumOPQRSTUV = vaddw_u8(vsumOPQRSTUV, vi5xOPQRSTUV);

      vacc0123 = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vacc0123), vget_low_u16(vprod01234567)));
      vacc4567 = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vacc4567), vget_high_u16(vprod01234567)));
      vacc89AB = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vacc89AB), vget_low_u16(vprod89ABCDEF)));
      vaccCDEF = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vaccCDEF), vget_high_u16(vprod89ABCDEF)));
      vaccGHIJ = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vaccGHIJ), vget_low_u16(vprodGHIJKLMN)));
      vaccKLMN = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vaccKLMN), vget_high_u16(vprodGHIJKLMN)));
      vaccOPQR = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vaccOPQR), vget_low_u16(vprodOPQRSTUV)));
      vaccSTUV = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vaccSTUV), vget_high_u16(vprodOPQRSTUV)));
      const uint8x8_t vi6x01234567 = vld1_u8(i6); i6 += 8;
      const uint8x8_t vk6x01234567 = vld1_u8(w); w = (const int8_t*) w + 8;
      const uint8x8_t vi6x89ABCDEF = vld1_u8(i6); i6 += 8;
      const uint8x8_t vk6x89ABCDEF = vld1_u8(w); w = (const int8_t*) w + 8;
      const uint8x8_t vi6xGHIJKLMN = vld1_u8(i6); i6 += 8;
      const uint8x8_t vk6xGHIJKLMN = vld1_u8(w); w = (const int8_t*) w + 8;
      const uint8x8_t vi6xOPQRSTUV = vld1_u8(i6); i6 += 8;
      const uint8x8_t vk6xOPQRSTUV = vld1_u8(w); w = (const int8_t*) w + 8;

      vprod01234567 = vmull_u8(vi6x01234567, vk6x01234567);
      vsum01234567 = vaddw_u8(vsum01234567, vi6x01234567);
      vprod89ABCDEF = vmull_u8(vi6x89ABCDEF, vk6x89ABCDEF);
      vsum89ABCDEF = vaddw_u8(vsum89ABCDEF, vi6x89ABCDEF);
      vprodGHIJKLMN = vmull_u8(vi6xGHIJKLMN, vk6xGHIJKLMN);
      vsumGHIJKLMN = vaddw_u8(vsumGHIJKLMN, vi6xGHIJKLMN);
      vprodOPQRSTUV = vmull_u8(vi6xOPQRSTUV, vk6xOPQRSTUV);
      vsumOPQRSTUV = vaddw_u8(vsumOPQRSTUV, vi6xOPQRSTUV);

      vacc0123 = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vacc0123), vget_low_u16(vprod01234567)));
      vacc4567 = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vacc4567), vget_high_u16(vprod01234567)));
      vacc89AB = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vacc89AB), vget_low_u16(vprod89ABCDEF)));
      vaccCDEF = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vaccCDEF), vget_high_u16(vprod89ABCDEF)));
      vaccGHIJ = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vaccGHIJ), vget_low_u16(vprodGHIJKLMN)));
      vaccKLMN = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vaccKLMN), vget_high_u16(vprodGHIJKLMN)));
      vaccOPQR = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vaccOPQR), vget_low_u16(vprodOPQRSTUV)));
      vaccSTUV = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vaccSTUV), vget_high_u16(vprodOPQRSTUV)));
      const uint8x8_t vi7x01234567 = vld1_u8(i7); i7 += 8;
      const uint8x8_t vk7x01234567 = vld1_u8(w); w = (const int8_t*) w + 8;
      const uint8x8_t vi7x89ABCDEF = vld1_u8(i7); i7 += 8;
      const uint8x8_t vk7x89ABCDEF = vld1_u8(w); w = (const int8_t*) w + 8;
      const uint8x8_t vi7xGHIJKLMN = vld1_u8(i7); i7 += 8;
      const uint8x8_t vk7xGHIJKLMN = vld1_u8(w); w = (const int8_t*) w + 8;
      const uint8x8_t vi7xOPQRSTUV = vld1_u8(i7); i7 += 8;
      const uint8x8_t vk7xOPQRSTUV = vld1_u8(w); w = (const int8_t*) w + 8;

      vprod01234567 = vmull_u8(vi7x01234567, vk7x01234567);
      vsum01234567 = vaddw_u8(vsum01234567, vi7x01234567);
      vprod89ABCDEF = vmull_u8(vi7x89ABCDEF, vk7x89ABCDEF);
      vsum89ABCDEF = vaddw_u8(vsum89ABCDEF, vi7x89ABCDEF);
      vprodGHIJKLMN = vmull_u8(vi7xGHIJKLMN, vk7xGHIJKLMN);
      vsumGHIJKLMN = vaddw_u8(vsumGHIJKLMN, vi7xGHIJKLMN);
      vprodOPQRSTUV = vmull_u8(vi7xOPQRSTUV, vk7xOPQRSTUV);
      vsumOPQRSTUV = vaddw_u8(vsumOPQRSTUV, vi7xOPQRSTUV);

      vacc0123 = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vacc0123), vget_low_u16(vprod01234567)));
      vacc4567 = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vacc4567), vget_high_u16(vprod01234567)));
      vacc89AB = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vacc89AB), vget_low_u16(vprod89ABCDEF)));
      vaccCDEF = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vaccCDEF), vget_high_u16(vprod89ABCDEF)));
      vaccGHIJ = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vaccGHIJ), vget_low_u16(vprodGHIJKLMN)));
      vaccKLMN = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vaccKLMN), vget_high_u16(vprodGHIJKLMN)));
      vaccOPQR = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vaccOPQR), vget_low_u16(vprodOPQRSTUV)));
      vaccSTUV = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vaccSTUV), vget_high_u16(vprodOPQRSTUV)));
      const uint8x8_t vi8x01234567 = vld1_u8(i8); i8 += 8;
      const uint8x8_t vk8x01234567 = vld1_u8(w); w = (const int8_t*) w + 8;
      const uint8x8_t vi8x89ABCDEF = vld1_u8(i8); i8 += 8;
      const uint8x8_t vk8x89ABCDEF = vld1_u8(w); w = (const int8_t*) w + 8;
      const uint8x8_t vi8xGHIJKLMN = vld1_u8(i8); i8 += 8;
      const uint8x8_t vk8xGHIJKLMN = vld1_u8(w); w = (const int8_t*) w + 8;
      const uint8x8_t vi8xOPQRSTUV = vld1_u8(i8); i8 += 8;
      const uint8x8_t vk8xOPQRSTUV = vld1_u8(w); w = (const int8_t*) w + 8;

      vprod01234567 = vmull_u8(vi8x01234567, vk8x01234567);
      vsum01234567 = vaddw_u8(vsum01234567, vi8x01234567);
      vprod89ABCDEF = vmull_u8(vi8x89ABCDEF, vk8x89ABCDEF);
      vsum89ABCDEF = vaddw_u8(vsum89ABCDEF, vi8x89ABCDEF);
      vprodGHIJKLMN = vmull_u8(vi8xGHIJKLMN, vk8xGHIJKLMN);
      vsumGHIJKLMN = vaddw_u8(vsumGHIJKLMN, vi8xGHIJKLMN);
      vprodOPQRSTUV = vmull_u8(vi8xOPQRSTUV, vk8xOPQRSTUV);
      vsumOPQRSTUV = vaddw_u8(vsumOPQRSTUV, vi8xOPQRSTUV);

      vacc0123 = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vacc0123), vget_low_u16(vprod01234567)));
      vacc4567 = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vacc4567), vget_high_u16(vprod01234567)));
      vacc89AB = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vacc89AB), vget_low_u16(vprod89ABCDEF)));
      vaccCDEF = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vaccCDEF), vget_high_u16(vprod89ABCDEF)));
      vaccGHIJ = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vaccGHIJ), vget_low_u16(vprodGHIJKLMN)));
      vaccKLMN = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vaccKLMN), vget_high_u16(vprodGHIJKLMN)));
      vaccOPQR = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vaccOPQR), vget_low_u16(vprodOPQRSTUV)));
      vaccSTUV = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vaccSTUV), vget_high_u16(vprodOPQRSTUV)));
      const uint8x8_t vi9x01234567 = vld1_u8(i9); i9 += 8;
      const uint8x8_t vk9x01234567 = vld1_u8(w); w = (const int8_t*) w + 8;
      const uint8x8_t vi9x89ABCDEF = vld1_u8(i9); i9 += 8;
      const uint8x8_t vk9x89ABCDEF = vld1_u8(w); w = (const int8_t*) w + 8;
      const uint8x8_t vi9xGHIJKLMN = vld1_u8(i9); i9 += 8;
      const uint8x8_t vk9xGHIJKLMN = vld1_u8(w); w = (const int8_t*) w + 8;
      const uint8x8_t vi9xOPQRSTUV = vld1_u8(i9); i9 += 8;
      const uint8x8_t vk9xOPQRSTUV = vld1_u8(w); w = (const int8_t*) w + 8;

      vprod01234567 = vmull_u8(vi9x01234567, vk9x01234567);
      vsum01234567 = vaddw_u8(vsum01234567, vi9x01234567);
      vprod89ABCDEF = vmull_u8(vi9x89ABCDEF, vk9x89ABCDEF);
      vsum89ABCDEF = vaddw_u8(vsum89ABCDEF, vi9x89ABCDEF);
      vprodGHIJKLMN = vmull_u8(vi9xGHIJKLMN, vk9xGHIJKLMN);
      vsumGHIJKLMN = vaddw_u8(vsumGHIJKLMN, vi9xGHIJKLMN);
      vprodOPQRSTUV = vmull_u8(vi9xOPQRSTUV, vk9xOPQRSTUV);
      vsumOPQRSTUV = vaddw_u8(vsumOPQRSTUV, vi9xOPQRSTUV);

      vacc0123 = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vacc0123), vget_low_u16(vprod01234567)));
      vacc4567 = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vacc4567), vget_high_u16(vprod01234567)));
      vacc89AB = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vacc89AB), vget_low_u16(vprod89ABCDEF)));
      vaccCDEF = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vaccCDEF), vget_high_u16(vprod89ABCDEF)));
      vaccGHIJ = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vaccGHIJ), vget_low_u16(vprodGHIJKLMN)));
      vaccKLMN = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vaccKLMN), vget_high_u16(vprodGHIJKLMN)));
      vaccOPQR = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vaccOPQR), vget_low_u16(vprodOPQRSTUV)));
      vaccSTUV = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vaccSTUV), vget_high_u16(vprodOPQRSTUV)));
      const uint8x8_t vi10x01234567 = vld1_u8(i10); i10 += 8;
      const uint8x8_t vk10x01234567 = vld1_u8(w); w = (const int8_t*) w + 8;
      const uint8x8_t vi10x89ABCDEF = vld1_u8(i10); i10 += 8;
      const uint8x8_t vk10x89ABCDEF = vld1_u8(w); w = (const int8_t*) w + 8;
      const uint8x8_t vi10xGHIJKLMN = vld1_u8(i10); i10 += 8;
      const uint8x8_t vk10xGHIJKLMN = vld1_u8(w); w = (const int8_t*) w + 8;
      const uint8x8_t vi10xOPQRSTUV = vld1_u8(i10); i10 += 8;
      const uint8x8_t vk10xOPQRSTUV = vld1_u8(w); w = (const int8_t*) w + 8;

      vprod01234567 = vmull_u8(vi10x01234567, vk10x01234567);
      vsum01234567 = vaddw_u8(vsum01234567, vi10x01234567);
      vprod89ABCDEF = vmull_u8(vi10x89ABCDEF, vk10x89ABCDEF);
      vsum89ABCDEF = vaddw_u8(vsum89ABCDEF, vi10x89ABCDEF);
      vprodGHIJKLMN = vmull_u8(vi10xGHIJKLMN, vk10xGHIJKLMN);
      vsumGHIJKLMN = vaddw_u8(vsumGHIJKLMN, vi10xGHIJKLMN);
      vprodOPQRSTUV = vmull_u8(vi10xOPQRSTUV, vk10xOPQRSTUV);
      vsumOPQRSTUV = vaddw_u8(vsumOPQRSTUV, vi10xOPQRSTUV);

      vacc0123 = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vacc0123), vget_low_u16(vprod01234567)));
      vacc4567 = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vacc4567), vget_high_u16(vprod01234567)));
      vacc89AB = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vacc89AB), vget_low_u16(vprod89ABCDEF)));
      vaccCDEF = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vaccCDEF), vget_high_u16(vprod89ABCDEF)));
      vaccGHIJ = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vaccGHIJ), vget_low_u16(vprodGHIJKLMN)));
      vaccKLMN = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vaccKLMN), vget_high_u16(vprodGHIJKLMN)));
      vaccOPQR = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vaccOPQR), vget_low_u16(vprodOPQRSTUV)));
      vaccSTUV = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vaccSTUV), vget_high_u16(vprodOPQRSTUV)));
      const uint8x8_t vi11x01234567 = vld1_u8(i11); i11 += 8;
      const uint8x8_t vk11x01234567 = vld1_u8(w); w = (const int8_t*) w + 8;
      const uint8x8_t vi11x89ABCDEF = vld1_u8(i11); i11 += 8;
      const uint8x8_t vk11x89ABCDEF = vld1_u8(w); w = (const int8_t*) w + 8;
      const uint8x8_t vi11xGHIJKLMN = vld1_u8(i11); i11 += 8;
      const uint8x8_t vk11xGHIJKLMN = vld1_u8(w); w = (const int8_t*) w + 8;
      const uint8x8_t vi11xOPQRSTUV = vld1_u8(i11); i11 += 8;
      const uint8x8_t vk11xOPQRSTUV = vld1_u8(w); w = (const int8_t*) w + 8;

      vprod01234567 = vmull_u8(vi11x01234567, vk11x01234567);
      vsum01234567 = vaddw_u8(vsum01234567, vi11x01234567);
      vprod89ABCDEF = vmull_u8(vi11x89ABCDEF, vk11x89ABCDEF);
      vsum89ABCDEF = vaddw_u8(vsum89ABCDEF, vi11x89ABCDEF);
      vprodGHIJKLMN = vmull_u8(vi11xGHIJKLMN, vk11xGHIJKLMN);
      vsumGHIJKLMN = vaddw_u8(vsumGHIJKLMN, vi11xGHIJKLMN);
      vprodOPQRSTUV = vmull_u8(vi11xOPQRSTUV, vk11xOPQRSTUV);
      vsumOPQRSTUV = vaddw_u8(vsumOPQRSTUV, vi11xOPQRSTUV);

      vacc0123 = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vacc0123), vget_low_u16(vprod01234567)));
      vacc4567 = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vacc4567), vget_high_u16(vprod01234567)));
      vacc89AB = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vacc89AB), vget_low_u16(vprod89ABCDEF)));
      vaccCDEF = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vaccCDEF), vget_high_u16(vprod89ABCDEF)));
      vaccGHIJ = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vaccGHIJ), vget_low_u16(vprodGHIJKLMN)));
      vaccKLMN = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vaccKLMN), vget_high_u16(vprodGHIJKLMN)));
      vaccOPQR = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vaccOPQR), vget_low_u16(vprodOPQRSTUV)));
      vaccSTUV = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vaccSTUV), vget_high_u16(vprodOPQRSTUV)));
      const uint8x8_t vi12x01234567 = vld1_u8(i12); i12 += 8;
      const uint8x8_t vk12x01234567 = vld1_u8(w); w = (const int8_t*) w + 8;
      const uint8x8_t vi12x89ABCDEF = vld1_u8(i12); i12 += 8;
      const uint8x8_t vk12x89ABCDEF = vld1_u8(w); w = (const int8_t*) w + 8;
      const uint8x8_t vi12xGHIJKLMN = vld1_u8(i12); i12 += 8;
      const uint8x8_t vk12xGHIJKLMN = vld1_u8(w); w = (const int8_t*) w + 8;
      const uint8x8_t vi12xOPQRSTUV = vld1_u8(i12); i12 += 8;
      const uint8x8_t vk12xOPQRSTUV = vld1_u8(w); w = (const int8_t*) w + 8;

      vprod01234567 = vmull_u8(vi12x01234567, vk12x01234567);
      vsum01234567 = vaddw_u8(vsum01234567, vi12x01234567);
      vprod89ABCDEF = vmull_u8(vi12x89ABCDEF, vk12x89ABCDEF);
      vsum89ABCDEF = vaddw_u8(vsum89ABCDEF, vi12x89ABCDEF);
      vprodGHIJKLMN = vmull_u8(vi12xGHIJKLMN, vk12xGHIJKLMN);
      vsumGHIJKLMN = vaddw_u8(vsumGHIJKLMN, vi12xGHIJKLMN);
      vprodOPQRSTUV = vmull_u8(vi12xOPQRSTUV, vk12xOPQRSTUV);
      vsumOPQRSTUV = vaddw_u8(vsumOPQRSTUV, vi12xOPQRSTUV);

      vacc0123 = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vacc0123), vget_low_u16(vprod01234567)));
      vacc4567 = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vacc4567), vget_high_u16(vprod01234567)));
      vacc89AB = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vacc89AB), vget_low_u16(vprod89ABCDEF)));
      vaccCDEF = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vaccCDEF), vget_high_u16(vprod89ABCDEF)));
      vaccGHIJ = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vaccGHIJ), vget_low_u16(vprodGHIJKLMN)));
      vaccKLMN = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vaccKLMN), vget_high_u16(vprodGHIJKLMN)));
      vaccOPQR = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vaccOPQR), vget_low_u16(vprodOPQRSTUV)));
      vaccSTUV = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vaccSTUV), vget_high_u16(vprodOPQRSTUV)));
      const uint8x8_t vi13x01234567 = vld1_u8(i13); i13 += 8;
      const uint8x8_t vk13x01234567 = vld1_u8(w); w = (const int8_t*) w + 8;
      const uint8x8_t vi13x89ABCDEF = vld1_u8(i13); i13 += 8;
      const uint8x8_t vk13x89ABCDEF = vld1_u8(w); w = (const int8_t*) w + 8;
      const uint8x8_t vi13xGHIJKLMN = vld1_u8(i13); i13 += 8;
      const uint8x8_t vk13xGHIJKLMN = vld1_u8(w); w = (const int8_t*) w + 8;
      const uint8x8_t vi13xOPQRSTUV = vld1_u8(i13); i13 += 8;
      const uint8x8_t vk13xOPQRSTUV = vld1_u8(w); w = (const int8_t*) w + 8;

      vprod01234567 = vmull_u8(vi13x01234567, vk13x01234567);
      vsum01234567 = vaddw_u8(vsum01234567, vi13x01234567);
      vprod89ABCDEF = vmull_u8(vi13x89ABCDEF, vk13x89ABCDEF);
      vsum89ABCDEF = vaddw_u8(vsum89ABCDEF, vi13x89ABCDEF);
      vprodGHIJKLMN = vmull_u8(vi13xGHIJKLMN, vk13xGHIJKLMN);
      vsumGHIJKLMN = vaddw_u8(vsumGHIJKLMN, vi13xGHIJKLMN);
      vprodOPQRSTUV = vmull_u8(vi13xOPQRSTUV, vk13xOPQRSTUV);
      vsumOPQRSTUV = vaddw_u8(vsumOPQRSTUV, vi13xOPQRSTUV);

      vacc0123 = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vacc0123), vget_low_u16(vprod01234567)));
      vacc4567 = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vacc4567), vget_high_u16(vprod01234567)));
      vacc89AB = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vacc89AB), vget_low_u16(vprod89ABCDEF)));
      vaccCDEF = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vaccCDEF), vget_high_u16(vprod89ABCDEF)));
      vaccGHIJ = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vaccGHIJ), vget_low_u16(vprodGHIJKLMN)));
      vaccKLMN = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vaccKLMN), vget_high_u16(vprodGHIJKLMN)));
      vaccOPQR = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vaccOPQR), vget_low_u16(vprodOPQRSTUV)));
      vaccSTUV = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vaccSTUV), vget_high_u16(vprodOPQRSTUV)));
      const uint8x8_t vi14x01234567 = vld1_u8(i14); i14 += 8;
      const uint8x8_t vk14x01234567 = vld1_u8(w); w = (const int8_t*) w + 8;
      const uint8x8_t vi14x89ABCDEF = vld1_u8(i14); i14 += 8;
      const uint8x8_t vk14x89ABCDEF = vld1_u8(w); w = (const int8_t*) w + 8;
      const uint8x8_t vi14xGHIJKLMN = vld1_u8(i14); i14 += 8;
      const uint8x8_t vk14xGHIJKLMN = vld1_u8(w); w = (const int8_t*) w + 8;
      const uint8x8_t vi14xOPQRSTUV = vld1_u8(i14); i14 += 8;
      const uint8x8_t vk14xOPQRSTUV = vld1_u8(w); w = (const int8_t*) w + 8;

      vprod01234567 = vmull_u8(vi14x01234567, vk14x01234567);
      vsum01234567 = vaddw_u8(vsum01234567, vi14x01234567);
      vprod89ABCDEF = vmull_u8(vi14x89ABCDEF, vk14x89ABCDEF);
      vsum89ABCDEF = vaddw_u8(vsum89ABCDEF, vi14x89ABCDEF);
      vprodGHIJKLMN = vmull_u8(vi14xGHIJKLMN, vk14xGHIJKLMN);
      vsumGHIJKLMN = vaddw_u8(vsumGHIJKLMN, vi14xGHIJKLMN);
      vprodOPQRSTUV = vmull_u8(vi14xOPQRSTUV, vk14xOPQRSTUV);
      vsumOPQRSTUV = vaddw_u8(vsumOPQRSTUV, vi14xOPQRSTUV);

      vacc0123 = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vacc0123), vget_low_u16(vprod01234567)));
      vacc4567 = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vacc4567), vget_high_u16(vprod01234567)));
      vacc89AB = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vacc89AB), vget_low_u16(vprod89ABCDEF)));
      vaccCDEF = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vaccCDEF), vget_high_u16(vprod89ABCDEF)));
      vaccGHIJ = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vaccGHIJ), vget_low_u16(vprodGHIJKLMN)));
      vaccKLMN = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vaccKLMN), vget_high_u16(vprodGHIJKLMN)));
      vaccOPQR = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vaccOPQR), vget_low_u16(vprodOPQRSTUV)));
      vaccSTUV = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vaccSTUV), vget_high_u16(vprodOPQRSTUV)));
      const uint8x8_t vi15x01234567 = vld1_u8(i15); i15 += 8;
      const uint8x8_t vk15x01234567 = vld1_u8(w); w = (const int8_t*) w + 8;
      const uint8x8_t vi15x89ABCDEF = vld1_u8(i15); i15 += 8;
      const uint8x8_t vk15x89ABCDEF = vld1_u8(w); w = (const int8_t*) w + 8;
      const uint8x8_t vi15xGHIJKLMN = vld1_u8(i15); i15 += 8;
      const uint8x8_t vk15xGHIJKLMN = vld1_u8(w); w = (const int8_t*) w + 8;
      const uint8x8_t vi15xOPQRSTUV = vld1_u8(i15); i15 += 8;
      const uint8x8_t vk15xOPQRSTUV = vld1_u8(w); w = (const int8_t*) w + 8;

      vprod01234567 = vmull_u8(vi15x01234567, vk15x01234567);
      vsum01234567 = vaddw_u8(vsum01234567, vi15x01234567);
      vprod89ABCDEF = vmull_u8(vi15x89ABCDEF, vk15x89ABCDEF);
      vsum89ABCDEF = vaddw_u8(vsum89ABCDEF, vi15x89ABCDEF);
      vprodGHIJKLMN = vmull_u8(vi15xGHIJKLMN, vk15xGHIJKLMN);
      vsumGHIJKLMN = vaddw_u8(vsumGHIJKLMN, vi15xGHIJKLMN);
      vprodOPQRSTUV = vmull_u8(vi15xOPQRSTUV, vk15xOPQRSTUV);
      vsumOPQRSTUV = vaddw_u8(vsumOPQRSTUV, vi15xOPQRSTUV);

      vacc0123 = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vacc0123), vget_low_u16(vprod01234567)));
      vacc4567 = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vacc4567), vget_high_u16(vprod01234567)));
      vacc89AB = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vacc89AB), vget_low_u16(vprod89ABCDEF)));
      vaccCDEF = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vaccCDEF), vget_high_u16(vprod89ABCDEF)));
      vaccGHIJ = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vaccGHIJ), vget_low_u16(vprodGHIJKLMN)));
      vaccKLMN = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vaccKLMN), vget_high_u16(vprodGHIJKLMN)));
      vaccOPQR = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vaccOPQR), vget_low_u16(vprodOPQRSTUV)));
      vaccSTUV = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vaccSTUV), vget_high_u16(vprodOPQRSTUV)));
      const uint8x8_t vi16x01234567 = vld1_u8(i16); i16 += 8;
      const uint8x8_t vk16x01234567 = vld1_u8(w); w = (const int8_t*) w + 8;
      const uint8x8_t vi16x89ABCDEF = vld1_u8(i16); i16 += 8;
      const uint8x8_t vk16x89ABCDEF = vld1_u8(w); w = (const int8_t*) w + 8;
      const uint8x8_t vi16xGHIJKLMN = vld1_u8(i16); i16 += 8;
      const uint8x8_t vk16xGHIJKLMN = vld1_u8(w); w = (const int8_t*) w + 8;
      const uint8x8_t vi16xOPQRSTUV = vld1_u8(i16); i16 += 8;
      const uint8x8_t vk16xOPQRSTUV = vld1_u8(w); w = (const int8_t*) w + 8;

      vprod01234567 = vmull_u8(vi16x01234567, vk16x01234567);
      vsum01234567 = vaddw_u8(vsum01234567, vi16x01234567);
      vprod89ABCDEF = vmull_u8(vi16x89ABCDEF, vk16x89ABCDEF);
      vsum89ABCDEF = vaddw_u8(vsum89ABCDEF, vi16x89ABCDEF);
      vprodGHIJKLMN = vmull_u8(vi16xGHIJKLMN, vk16xGHIJKLMN);
      vsumGHIJKLMN = vaddw_u8(vsumGHIJKLMN, vi16xGHIJKLMN);
      vprodOPQRSTUV = vmull_u8(vi16xOPQRSTUV, vk16xOPQRSTUV);
      vsumOPQRSTUV = vaddw_u8(vsumOPQRSTUV, vi16xOPQRSTUV);

      vacc0123 = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vacc0123), vget_low_u16(vprod01234567)));
      vacc4567 = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vacc4567), vget_high_u16(vprod01234567)));
      vacc89AB = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vacc89AB), vget_low_u16(vprod89ABCDEF)));
      vaccCDEF = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vaccCDEF), vget_high_u16(vprod89ABCDEF)));
      vaccGHIJ = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vaccGHIJ), vget_low_u16(vprodGHIJKLMN)));
      vaccKLMN = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vaccKLMN), vget_high_u16(vprodGHIJKLMN)));
      vaccOPQR = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vaccOPQR), vget_low_u16(vprodOPQRSTUV)));
      vaccSTUV = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vaccSTUV), vget_high_u16(vprodOPQRSTUV)));
      const uint8x8_t vi17x01234567 = vld1_u8(i17); i17 += 8;
      const uint8x8_t vk17x01234567 = vld1_u8(w); w = (const int8_t*) w + 8;
      const uint8x8_t vi17x89ABCDEF = vld1_u8(i17); i17 += 8;
      const uint8x8_t vk17x89ABCDEF = vld1_u8(w); w = (const int8_t*) w + 8;
      const uint8x8_t vi17xGHIJKLMN = vld1_u8(i17); i17 += 8;
      const uint8x8_t vk17xGHIJKLMN = vld1_u8(w); w = (const int8_t*) w + 8;
      const uint8x8_t vi17xOPQRSTUV = vld1_u8(i17); i17 += 8;
      const uint8x8_t vk17xOPQRSTUV = vld1_u8(w); w = (const int8_t*) w + 8;

      vprod01234567 = vmull_u8(vi17x01234567, vk17x01234567);
      vsum01234567 = vaddw_u8(vsum01234567, vi17x01234567);
      vprod89ABCDEF = vmull_u8(vi17x89ABCDEF, vk17x89ABCDEF);
      vsum89ABCDEF = vaddw_u8(vsum89ABCDEF, vi17x89ABCDEF);
      vprodGHIJKLMN = vmull_u8(vi17xGHIJKLMN, vk17xGHIJKLMN);
      vsumGHIJKLMN = vaddw_u8(vsumGHIJKLMN, vi17xGHIJKLMN);
      vprodOPQRSTUV = vmull_u8(vi17xOPQRSTUV, vk17xOPQRSTUV);
      vsumOPQRSTUV = vaddw_u8(vsumOPQRSTUV, vi17xOPQRSTUV);

      vacc0123 = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vacc0123), vget_low_u16(vprod01234567)));
      vacc4567 = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vacc4567), vget_high_u16(vprod01234567)));
      vacc89AB = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vacc89AB), vget_low_u16(vprod89ABCDEF)));
      vaccCDEF = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vaccCDEF), vget_high_u16(vprod89ABCDEF)));
      vaccGHIJ = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vaccGHIJ), vget_low_u16(vprodGHIJKLMN)));
      vaccKLMN = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vaccKLMN), vget_high_u16(vprodGHIJKLMN)));
      vaccOPQR = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vaccOPQR), vget_low_u16(vprodOPQRSTUV)));
      vaccSTUV = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vaccSTUV), vget_high_u16(vprodOPQRSTUV)));
      const uint8x8_t vi18x01234567 = vld1_u8(i18); i18 += 8;
      const uint8x8_t vk18x01234567 = vld1_u8(w); w = (const int8_t*) w + 8;
      const uint8x8_t vi18x89ABCDEF = vld1_u8(i18); i18 += 8;
      const uint8x8_t vk18x89ABCDEF = vld1_u8(w); w = (const int8_t*) w + 8;
      const uint8x8_t vi18xGHIJKLMN = vld1_u8(i18); i18 += 8;
      const uint8x8_t vk18xGHIJKLMN = vld1_u8(w); w = (const int8_t*) w + 8;
      const uint8x8_t vi18xOPQRSTUV = vld1_u8(i18); i18 += 8;
      const uint8x8_t vk18xOPQRSTUV = vld1_u8(w); w = (const int8_t*) w + 8;

      vprod01234567 = vmull_u8(vi18x01234567, vk18x01234567);
      vsum01234567 = vaddw_u8(vsum01234567, vi18x01234567);
      vprod89ABCDEF = vmull_u8(vi18x89ABCDEF, vk18x89ABCDEF);
      vsum89ABCDEF = vaddw_u8(vsum89ABCDEF, vi18x89ABCDEF);
      vprodGHIJKLMN = vmull_u8(vi18xGHIJKLMN, vk18xGHIJKLMN);
      vsumGHIJKLMN = vaddw_u8(vsumGHIJKLMN, vi18xGHIJKLMN);
      vprodOPQRSTUV = vmull_u8(vi18xOPQRSTUV, vk18xOPQRSTUV);
      vsumOPQRSTUV = vaddw_u8(vsumOPQRSTUV, vi18xOPQRSTUV);

      vacc0123 = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vacc0123), vget_low_u16(vprod01234567)));
      vacc4567 = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vacc4567), vget_high_u16(vprod01234567)));
      vacc89AB = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vacc89AB), vget_low_u16(vprod89ABCDEF)));
      vaccCDEF = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vaccCDEF), vget_high_u16(vprod89ABCDEF)));
      vaccGHIJ = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vaccGHIJ), vget_low_u16(vprodGHIJKLMN)));
      vaccKLMN = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vaccKLMN), vget_high_u16(vprodGHIJKLMN)));
      vaccOPQR = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vaccOPQR), vget_low_u16(vprodOPQRSTUV)));
      vaccSTUV = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vaccSTUV), vget_high_u16(vprodOPQRSTUV)));
      const uint8x8_t vi19x01234567 = vld1_u8(i19); i19 += 8;
      const uint8x8_t vk19x01234567 = vld1_u8(w); w = (const int8_t*) w + 8;
      const uint8x8_t vi19x89ABCDEF = vld1_u8(i19); i19 += 8;
      const uint8x8_t vk19x89ABCDEF = vld1_u8(w); w = (const int8_t*) w + 8;
      const uint8x8_t vi19xGHIJKLMN = vld1_u8(i19); i19 += 8;
      const uint8x8_t vk19xGHIJKLMN = vld1_u8(w); w = (const int8_t*) w + 8;
      const uint8x8_t vi19xOPQRSTUV = vld1_u8(i19); i19 += 8;
      const uint8x8_t vk19xOPQRSTUV = vld1_u8(w); w = (const int8_t*) w + 8;

      vprod01234567 = vmull_u8(vi19x01234567, vk19x01234567);
      vsum01234567 = vaddw_u8(vsum01234567, vi19x01234567);
      vprod89ABCDEF = vmull_u8(vi19x89ABCDEF, vk19x89ABCDEF);
      vsum89ABCDEF = vaddw_u8(vsum89ABCDEF, vi19x89ABCDEF);
      vprodGHIJKLMN = vmull_u8(vi19xGHIJKLMN, vk19xGHIJKLMN);
      vsumGHIJKLMN = vaddw_u8(vsumGHIJKLMN, vi19xGHIJKLMN);
      vprodOPQRSTUV = vmull_u8(vi19xOPQRSTUV, vk19xOPQRSTUV);
      vsumOPQRSTUV = vaddw_u8(vsumOPQRSTUV, vi19xOPQRSTUV);

      vacc0123 = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vacc0123), vget_low_u16(vprod01234567)));
      vacc4567 = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vacc4567), vget_high_u16(vprod01234567)));
      vacc89AB = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vacc89AB), vget_low_u16(vprod89ABCDEF)));
      vaccCDEF = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vaccCDEF), vget_high_u16(vprod89ABCDEF)));
      vaccGHIJ = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vaccGHIJ), vget_low_u16(vprodGHIJKLMN)));
      vaccKLMN = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vaccKLMN), vget_high_u16(vprodGHIJKLMN)));
      vaccOPQR = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vaccOPQR), vget_low_u16(vprodOPQRSTUV)));
      vaccSTUV = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vaccSTUV), vget_high_u16(vprodOPQRSTUV)));
      const uint8x8_t vi20x01234567 = vld1_u8(i20); i20 += 8;
      const uint8x8_t vk20x01234567 = vld1_u8(w); w = (const int8_t*) w + 8;
      const uint8x8_t vi20x89ABCDEF = vld1_u8(i20); i20 += 8;
      const uint8x8_t vk20x89ABCDEF = vld1_u8(w); w = (const int8_t*) w + 8;
      const uint8x8_t vi20xGHIJKLMN = vld1_u8(i20); i20 += 8;
      const uint8x8_t vk20xGHIJKLMN = vld1_u8(w); w = (const int8_t*) w + 8;
      const uint8x8_t vi20xOPQRSTUV = vld1_u8(i20); i20 += 8;
      const uint8x8_t vk20xOPQRSTUV = vld1_u8(w); w = (const int8_t*) w + 8;

      vprod01234567 = vmull_u8(vi20x01234567, vk20x01234567);
      vsum01234567 = vaddw_u8(vsum01234567, vi20x01234567);
      vprod89ABCDEF = vmull_u8(vi20x89ABCDEF, vk20x89ABCDEF);
      vsum89ABCDEF = vaddw_u8(vsum89ABCDEF, vi20x89ABCDEF);
      vprodGHIJKLMN = vmull_u8(vi20xGHIJKLMN, vk20xGHIJKLMN);
      vsumGHIJKLMN = vaddw_u8(vsumGHIJKLMN, vi20xGHIJKLMN);
      vprodOPQRSTUV = vmull_u8(vi20xOPQRSTUV, vk20xOPQRSTUV);
      vsumOPQRSTUV = vaddw_u8(vsumOPQRSTUV, vi20xOPQRSTUV);

      vacc0123 = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vacc0123), vget_low_u16(vprod01234567)));
      vacc4567 = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vacc4567), vget_high_u16(vprod01234567)));
      vacc89AB = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vacc89AB), vget_low_u16(vprod89ABCDEF)));
      vaccCDEF = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vaccCDEF), vget_high_u16(vprod89ABCDEF)));
      vaccGHIJ = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vaccGHIJ), vget_low_u16(vprodGHIJKLMN)));
      vaccKLMN = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vaccKLMN), vget_high_u16(vprodGHIJKLMN)));
      vaccOPQR = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vaccOPQR), vget_low_u16(vprodOPQRSTUV)));
      vaccSTUV = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vaccSTUV), vget_high_u16(vprodOPQRSTUV)));
      const uint8x8_t vi21x01234567 = vld1_u8(i21); i21 += 8;
      const uint8x8_t vk21x01234567 = vld1_u8(w); w = (const int8_t*) w + 8;
      const uint8x8_t vi21x89ABCDEF = vld1_u8(i21); i21 += 8;
      const uint8x8_t vk21x89ABCDEF = vld1_u8(w); w = (const int8_t*) w + 8;
      const uint8x8_t vi21xGHIJKLMN = vld1_u8(i21); i21 += 8;
      const uint8x8_t vk21xGHIJKLMN = vld1_u8(w); w = (const int8_t*) w + 8;
      const uint8x8_t vi21xOPQRSTUV = vld1_u8(i21); i21 += 8;
      const uint8x8_t vk21xOPQRSTUV = vld1_u8(w); w = (const int8_t*) w + 8;

      vprod01234567 = vmull_u8(vi21x01234567, vk21x01234567);
      vsum01234567 = vaddw_u8(vsum01234567, vi21x01234567);
      vprod89ABCDEF = vmull_u8(vi21x89ABCDEF, vk21x89ABCDEF);
      vsum89ABCDEF = vaddw_u8(vsum89ABCDEF, vi21x89ABCDEF);
      vprodGHIJKLMN = vmull_u8(vi21xGHIJKLMN, vk21xGHIJKLMN);
      vsumGHIJKLMN = vaddw_u8(vsumGHIJKLMN, vi21xGHIJKLMN);
      vprodOPQRSTUV = vmull_u8(vi21xOPQRSTUV, vk21xOPQRSTUV);
      vsumOPQRSTUV = vaddw_u8(vsumOPQRSTUV, vi21xOPQRSTUV);

      vacc0123 = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vacc0123), vget_low_u16(vprod01234567)));
      vacc4567 = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vacc4567), vget_high_u16(vprod01234567)));
      vacc89AB = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vacc89AB), vget_low_u16(vprod89ABCDEF)));
      vaccCDEF = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vaccCDEF), vget_high_u16(vprod89ABCDEF)));
      vaccGHIJ = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vaccGHIJ), vget_low_u16(vprodGHIJKLMN)));
      vaccKLMN = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vaccKLMN), vget_high_u16(vprodGHIJKLMN)));
      vaccOPQR = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vaccOPQR), vget_low_u16(vprodOPQRSTUV)));
      vaccSTUV = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vaccSTUV), vget_high_u16(vprodOPQRSTUV)));
      const uint8x8_t vi22x01234567 = vld1_u8(i22); i22 += 8;
      const uint8x8_t vk22x01234567 = vld1_u8(w); w = (const int8_t*) w + 8;
      const uint8x8_t vi22x89ABCDEF = vld1_u8(i22); i22 += 8;
      const uint8x8_t vk22x89ABCDEF = vld1_u8(w); w = (const int8_t*) w + 8;
      const uint8x8_t vi22xGHIJKLMN = vld1_u8(i22); i22 += 8;
      const uint8x8_t vk22xGHIJKLMN = vld1_u8(w); w = (const int8_t*) w + 8;
      const uint8x8_t vi22xOPQRSTUV = vld1_u8(i22); i22 += 8;
      const uint8x8_t vk22xOPQRSTUV = vld1_u8(w); w = (const int8_t*) w + 8;

      vprod01234567 = vmull_u8(vi22x01234567, vk22x01234567);
      vsum01234567 = vaddw_u8(vsum01234567, vi22x01234567);
      vprod89ABCDEF = vmull_u8(vi22x89ABCDEF, vk22x89ABCDEF);
      vsum89ABCDEF = vaddw_u8(vsum89ABCDEF, vi22x89ABCDEF);
      vprodGHIJKLMN = vmull_u8(vi22xGHIJKLMN, vk22xGHIJKLMN);
      vsumGHIJKLMN = vaddw_u8(vsumGHIJKLMN, vi22xGHIJKLMN);
      vprodOPQRSTUV = vmull_u8(vi22xOPQRSTUV, vk22xOPQRSTUV);
      vsumOPQRSTUV = vaddw_u8(vsumOPQRSTUV, vi22xOPQRSTUV);

      vacc0123 = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vacc0123), vget_low_u16(vprod01234567)));
      vacc4567 = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vacc4567), vget_high_u16(vprod01234567)));
      vacc89AB = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vacc89AB), vget_low_u16(vprod89ABCDEF)));
      vaccCDEF = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vaccCDEF), vget_high_u16(vprod89ABCDEF)));
      vaccGHIJ = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vaccGHIJ), vget_low_u16(vprodGHIJKLMN)));
      vaccKLMN = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vaccKLMN), vget_high_u16(vprodGHIJKLMN)));
      vaccOPQR = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vaccOPQR), vget_low_u16(vprodOPQRSTUV)));
      vaccSTUV = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vaccSTUV), vget_high_u16(vprodOPQRSTUV)));
      const uint8x8_t vi23x01234567 = vld1_u8(i23); i23 += 8;
      const uint8x8_t vk23x01234567 = vld1_u8(w); w = (const int8_t*) w + 8;
      const uint8x8_t vi23x89ABCDEF = vld1_u8(i23); i23 += 8;
      const uint8x8_t vk23x89ABCDEF = vld1_u8(w); w = (const int8_t*) w + 8;
      const uint8x8_t vi23xGHIJKLMN = vld1_u8(i23); i23 += 8;
      const uint8x8_t vk23xGHIJKLMN = vld1_u8(w); w = (const int8_t*) w + 8;
      const uint8x8_t vi23xOPQRSTUV = vld1_u8(i23); i23 += 8;
      const uint8x8_t vk23xOPQRSTUV = vld1_u8(w); w = (const int8_t*) w + 8;

      vprod01234567 = vmull_u8(vi23x01234567, vk23x01234567);
      vsum01234567 = vaddw_u8(vsum01234567, vi23x01234567);
      vprod89ABCDEF = vmull_u8(vi23x89ABCDEF, vk23x89ABCDEF);
      vsum89ABCDEF = vaddw_u8(vsum89ABCDEF, vi23x89ABCDEF);
      vprodGHIJKLMN = vmull_u8(vi23xGHIJKLMN, vk23xGHIJKLMN);
      vsumGHIJKLMN = vaddw_u8(vsumGHIJKLMN, vi23xGHIJKLMN);
      vprodOPQRSTUV = vmull_u8(vi23xOPQRSTUV, vk23xOPQRSTUV);
      vsumOPQRSTUV = vaddw_u8(vsumOPQRSTUV, vi23xOPQRSTUV);

      vacc0123 = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vacc0123), vget_low_u16(vprod01234567)));
      vacc4567 = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vacc4567), vget_high_u16(vprod01234567)));
      vacc89AB = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vacc89AB), vget_low_u16(vprod89ABCDEF)));
      vaccCDEF = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vaccCDEF), vget_high_u16(vprod89ABCDEF)));
      vaccGHIJ = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vaccGHIJ), vget_low_u16(vprodGHIJKLMN)));
      vaccKLMN = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vaccKLMN), vget_high_u16(vprodGHIJKLMN)));
      vaccOPQR = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vaccOPQR), vget_low_u16(vprodOPQRSTUV)));
      vaccSTUV = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vaccSTUV), vget_high_u16(vprodOPQRSTUV)));
      const uint8x8_t vi24x01234567 = vld1_u8(i24); i24 += 8;
      const uint8x8_t vk24x01234567 = vld1_u8(w); w = (const int8_t*) w + 8;
      const uint8x8_t vi24x89ABCDEF = vld1_u8(i24); i24 += 8;
      const uint8x8_t vk24x89ABCDEF = vld1_u8(w); w = (const int8_t*) w + 8;
      const uint8x8_t vi24xGHIJKLMN = vld1_u8(i24); i24 += 8;
      const uint8x8_t vk24xGHIJKLMN = vld1_u8(w); w = (const int8_t*) w + 8;
      const uint8x8_t vi24xOPQRSTUV = vld1_u8(i24); i24 += 8;
      const uint8x8_t vk24xOPQRSTUV = vld1_u8(w); w = (const int8_t*) w + 8;

      vprod01234567 = vmull_u8(vi24x01234567, vk24x01234567);
      vsum01234567 = vaddw_u8(vsum01234567, vi24x01234567);
      vprod89ABCDEF = vmull_u8(vi24x89ABCDEF, vk24x89ABCDEF);
      vsum89ABCDEF = vaddw_u8(vsum89ABCDEF, vi24x89ABCDEF);
      vprodGHIJKLMN = vmull_u8(vi24xGHIJKLMN, vk24xGHIJKLMN);
      vsumGHIJKLMN = vaddw_u8(vsumGHIJKLMN, vi24xGHIJKLMN);
      vprodOPQRSTUV = vmull_u8(vi24xOPQRSTUV, vk24xOPQRSTUV);
      vsumOPQRSTUV = vaddw_u8(vsumOPQRSTUV, vi24xOPQRSTUV);

      vacc0123 = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vacc0123), vget_low_u16(vprod01234567)));
      vacc4567 = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vacc4567), vget_high_u16(vprod01234567)));
      vacc89AB = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vacc89AB), vget_low_u16(vprod89ABCDEF)));
      vaccCDEF = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vaccCDEF), vget_high_u16(vprod89ABCDEF)));
      vaccGHIJ = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vaccGHIJ), vget_low_u16(vprodGHIJKLMN)));
      vaccKLMN = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vaccKLMN), vget_high_u16(vprodGHIJKLMN)));
      vaccOPQR = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vaccOPQR), vget_low_u16(vprodOPQRSTUV)));
      vaccSTUV = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vaccSTUV), vget_high_u16(vprodOPQRSTUV)));

      vacc0123 = vreinterpretq_s32_u32(vmlsl_u16(vreinterpretq_u32_s32(vacc0123), vget_low_u16(vsum01234567), vget_low_u16(vkernel_zero_point)));
      vacc4567 = vreinterpretq_s32_u32(vmlsl_u16(vreinterpretq_u32_s32(vacc4567), vget_high_u16(vsum01234567), vget_high_u16(vkernel_zero_point)));
      vacc89AB = vreinterpretq_s32_u32(vmlsl_u16(vreinterpretq_u32_s32(vacc89AB), vget_low_u16(vsum89ABCDEF), vget_low_u16(vkernel_zero_point)));
      vaccCDEF = vreinterpretq_s32_u32(vmlsl_u16(vreinterpretq_u32_s32(vaccCDEF), vget_high_u16(vsum89ABCDEF), vget_high_u16(vkernel_zero_point)));
      vaccGHIJ = vreinterpretq_s32_u32(vmlsl_u16(vreinterpretq_u32_s32(vaccGHIJ), vget_low_u16(vsumGHIJKLMN), vget_low_u16(vkernel_zero_point)));
      vaccKLMN = vreinterpretq_s32_u32(vmlsl_u16(vreinterpretq_u32_s32(vaccKLMN), vget_high_u16(vsumGHIJKLMN), vget_high_u16(vkernel_zero_point)));
      vaccOPQR = vreinterpretq_s32_u32(vmlsl_u16(vreinterpretq_u32_s32(vaccOPQR), vget_low_u16(vsumOPQRSTUV), vget_low_u16(vkernel_zero_point)));
      vaccSTUV = vreinterpretq_s32_u32(vmlsl_u16(vreinterpretq_u32_s32(vaccSTUV), vget_high_u16(vsumOPQRSTUV), vget_high_u16(vkernel_zero_point)));

      vacc0123 = vshlq_s32(vacc0123, vright_pre_shift);
      vacc4567 = vshlq_s32(vacc4567, vright_pre_shift);
      vacc89AB = vshlq_s32(vacc89AB, vright_pre_shift);
      vaccCDEF = vshlq_s32(vaccCDEF, vright_pre_shift);
      vaccGHIJ = vshlq_s32(vaccGHIJ, vright_pre_shift);
      vaccKLMN = vshlq_s32(vaccKLMN, vright_pre_shift);
      vaccOPQR = vshlq_s32(vaccOPQR, vright_pre_shift);
      vaccSTUV = vshlq_s32(vaccSTUV, vright_pre_shift);

      vacc0123 = vqdmulhq_s32(vacc0123, vmultiplier);
      vacc4567 = vqdmulhq_s32(vacc4567, vmultiplier);
      vacc89AB = vqdmulhq_s32(vacc89AB, vmultiplier);
      vaccCDEF = vqdmulhq_s32(vaccCDEF, vmultiplier);
      vaccGHIJ = vqdmulhq_s32(vaccGHIJ, vmultiplier);
      vaccKLMN = vqdmulhq_s32(vaccKLMN, vmultiplier);
      vaccOPQR = vqdmulhq_s32(vaccOPQR, vmultiplier);
      vaccSTUV = vqdmulhq_s32(vaccSTUV, vmultiplier);

      vacc0123 = vrshlq_s32(vacc0123, vright_post_shift);
      vacc4567 = vrshlq_s32(vacc4567, vright_post_shift);
      vacc89AB = vrshlq_s32(vacc89AB, vright_post_shift);
      vaccCDEF = vrshlq_s32(vaccCDEF, vright_post_shift);
      vaccGHIJ = vrshlq_s32(vaccGHIJ, vright_post_shift);
      vaccKLMN = vrshlq_s32(vaccKLMN, vright_post_shift);
      vaccOPQR = vrshlq_s32(vaccOPQR, vright_post_shift);
      vaccSTUV = vrshlq_s32(vaccSTUV, vright_post_shift);

#if XNN_ARCH_ARM64
      const int16x8_t vacc01234567 = vqaddq_s16(vqmovn_high_s32(vqmovn_s32(vacc0123), vacc4567), voutput_zero_point);
      const int16x8_t vacc89ABCDEF = vqaddq_s16(vqmovn_high_s32(vqmovn_s32(vacc89AB), vaccCDEF), voutput_zero_point);
      const int16x8_t vaccGHIJKLMN = vqaddq_s16(vqmovn_high_s32(vqmovn_s32(vaccGHIJ), vaccKLMN), voutput_zero_point);
      const int16x8_t vaccOPQRSTUV = vqaddq_s16(vqmovn_high_s32(vqmovn_s32(vaccOPQR), vaccSTUV), voutput_zero_point);

      uint8x16_t vout0123456789ABCDEF = vqmovun_high_s16(vqmovun_s16(vacc01234567), vacc89ABCDEF);
      uint8x16_t voutGHIJKLMNOPQRSTUV = vqmovun_high_s16(vqmovun_s16(vaccGHIJKLMN), vaccOPQRSTUV);
#else
      const int16x8_t vacc01234567 = vqaddq_s16(vcombine_s16(vqmovn_s32(vacc0123), vqmovn_s32(vacc4567)), voutput_zero_point);
      const int16x8_t vacc89ABCDEF = vqaddq_s16(vcombine_s16(vqmovn_s32(vacc89AB), vqmovn_s32(vaccCDEF)), voutput_zero_point);
      const int16x8_t vaccGHIJKLMN = vqaddq_s16(vcombine_s16(vqmovn_s32(vaccGHIJ), vqmovn_s32(vaccKLMN)), voutput_zero_point);
      const int16x8_t vaccOPQRSTUV = vqaddq_s16(vcombine_s16(vqmovn_s32(vaccOPQR), vqmovn_s32(vaccSTUV)), voutput_zero_point);

      uint8x16_t vout0123456789ABCDEF = vcombine_u8(vqmovun_s16(vacc01234567), vqmovun_s16(vacc89ABCDEF));
      uint8x16_t voutGHIJKLMNOPQRSTUV = vcombine_u8(vqmovun_s16(vaccGHIJKLMN), vqmovun_s16(vaccOPQRSTUV));
#endif

      vout0123456789ABCDEF = vmaxq_u8(vout0123456789ABCDEF, voutput_min);
      voutGHIJKLMNOPQRSTUV = vmaxq_u8(voutGHIJKLMNOPQRSTUV, voutput_min);

      vout0123456789ABCDEF = vminq_u8(vout0123456789ABCDEF, voutput_max);
      voutGHIJKLMNOPQRSTUV = vminq_u8(voutGHIJKLMNOPQRSTUV, voutput_max);

      vst1q_u8(output, vout0123456789ABCDEF); output += 16;
      vst1q_u8(output, voutGHIJKLMNOPQRSTUV); output += 16;
    }
    if XNN_UNLIKELY(c != 0) {
      const uint8_t* k = (const uint8_t*) ((const int32_t*) w + 32);
      do {
        int32x4_t vacc0123 = vld1q_s32(w); w = (const int32_t*) w + 4;
        int32x4_t vacc4567 = vld1q_s32(w); w = (const int32_t*) w + 4;

        const uint8x8_t vi0x01234567 = vld1_u8(i0); i0 += 8;
        const uint8x8_t vk0x01234567 = vld1_u8(k); k += 8;

        uint16x8_t vprod01234567 = vmull_u8(vi0x01234567, vk0x01234567);

        vacc0123 = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vacc0123), vget_low_u16(vprod01234567)));
        vacc4567 = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vacc4567), vget_high_u16(vprod01234567)));
        const uint8x8_t vi1x01234567 = vld1_u8(i1); i1 += 8;
        const uint8x8_t vk1x01234567 = vld1_u8((const void*) (k + 24));

        vprod01234567 = vmull_u8(vi1x01234567, vk1x01234567);
        uint16x8_t vsum01234567 = vaddl_u8(vi0x01234567, vi1x01234567);

        vacc0123 = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vacc0123), vget_low_u16(vprod01234567)));
        vacc4567 = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vacc4567), vget_high_u16(vprod01234567)));
        const uint8x8_t vi2x01234567 = vld1_u8(i2); i2 += 8;
        const uint8x8_t vk2x01234567 = vld1_u8((const void*) (k + 56));

        vprod01234567 = vmull_u8(vi2x01234567, vk2x01234567);
        vsum01234567 = vaddw_u8(vsum01234567, vi2x01234567);

        vacc0123 = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vacc0123), vget_low_u16(vprod01234567)));
        vacc4567 = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vacc4567), vget_high_u16(vprod01234567)));
        const uint8x8_t vi3x01234567 = vld1_u8(i3); i3 += 8;
        const uint8x8_t vk3x01234567 = vld1_u8((const void*) (k + 88));

        vprod01234567 = vmull_u8(vi3x01234567, vk3x01234567);
        vsum01234567 = vaddw_u8(vsum01234567, vi3x01234567);

        vacc0123 = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vacc0123), vget_low_u16(vprod01234567)));
        vacc4567 = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vacc4567), vget_high_u16(vprod01234567)));
        const uint8x8_t vi4x01234567 = vld1_u8(i4); i4 += 8;
        const uint8x8_t vk4x01234567 = vld1_u8((const void*) (k + 120));

        vprod01234567 = vmull_u8(vi4x01234567, vk4x01234567);
        vsum01234567 = vaddw_u8(vsum01234567, vi4x01234567);

        vacc0123 = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vacc0123), vget_low_u16(vprod01234567)));
        vacc4567 = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vacc4567), vget_high_u16(vprod01234567)));
        const uint8x8_t vi5x01234567 = vld1_u8(i5); i5 += 8;
        const uint8x8_t vk5x01234567 = vld1_u8((const void*) (k + 152));

        vprod01234567 = vmull_u8(vi5x01234567, vk5x01234567);
        vsum01234567 = vaddw_u8(vsum01234567, vi5x01234567);

        vacc0123 = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vacc0123), vget_low_u16(vprod01234567)));
        vacc4567 = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vacc4567), vget_high_u16(vprod01234567)));
        const uint8x8_t vi6x01234567 = vld1_u8(i6); i6 += 8;
        const uint8x8_t vk6x01234567 = vld1_u8((const void*) (k + 184));

        vprod01234567 = vmull_u8(vi6x01234567, vk6x01234567);
        vsum01234567 = vaddw_u8(vsum01234567, vi6x01234567);

        vacc0123 = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vacc0123), vget_low_u16(vprod01234567)));
        vacc4567 = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vacc4567), vget_high_u16(vprod01234567)));
        const uint8x8_t vi7x01234567 = vld1_u8(i7); i7 += 8;
        const uint8x8_t vk7x01234567 = vld1_u8((const void*) (k + 216));

        vprod01234567 = vmull_u8(vi7x01234567, vk7x01234567);
        vsum01234567 = vaddw_u8(vsum01234567, vi7x01234567);

        vacc0123 = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vacc0123), vget_low_u16(vprod01234567)));
        vacc4567 = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vacc4567), vget_high_u16(vprod01234567)));
        const uint8x8_t vi8x01234567 = vld1_u8(i8); i8 += 8;
        const uint8x8_t vk8x01234567 = vld1_u8((const void*) (k + 248));

        vprod01234567 = vmull_u8(vi8x01234567, vk8x01234567);
        vsum01234567 = vaddw_u8(vsum01234567, vi8x01234567);

        vacc0123 = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vacc0123), vget_low_u16(vprod01234567)));
        vacc4567 = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vacc4567), vget_high_u16(vprod01234567)));
        const uint8x8_t vi9x01234567 = vld1_u8(i9); i9 += 8;
        const uint8x8_t vk9x01234567 = vld1_u8((const void*) (k + 280));

        vprod01234567 = vmull_u8(vi9x01234567, vk9x01234567);
        vsum01234567 = vaddw_u8(vsum01234567, vi9x01234567);

        vacc0123 = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vacc0123), vget_low_u16(vprod01234567)));
        vacc4567 = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vacc4567), vget_high_u16(vprod01234567)));
        const uint8x8_t vi10x01234567 = vld1_u8(i10); i10 += 8;
        const uint8x8_t vk10x01234567 = vld1_u8((const void*) (k + 312));

        vprod01234567 = vmull_u8(vi10x01234567, vk10x01234567);
        vsum01234567 = vaddw_u8(vsum01234567, vi10x01234567);

        vacc0123 = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vacc0123), vget_low_u16(vprod01234567)));
        vacc4567 = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vacc4567), vget_high_u16(vprod01234567)));
        const uint8x8_t vi11x01234567 = vld1_u8(i11); i11 += 8;
        const uint8x8_t vk11x01234567 = vld1_u8((const void*) (k + 344));

        vprod01234567 = vmull_u8(vi11x01234567, vk11x01234567);
        vsum01234567 = vaddw_u8(vsum01234567, vi11x01234567);

        vacc0123 = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vacc0123), vget_low_u16(vprod01234567)));
        vacc4567 = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vacc4567), vget_high_u16(vprod01234567)));
        const uint8x8_t vi12x01234567 = vld1_u8(i12); i12 += 8;
        const uint8x8_t vk12x01234567 = vld1_u8((const void*) (k + 376));

        vprod01234567 = vmull_u8(vi12x01234567, vk12x01234567);
        vsum01234567 = vaddw_u8(vsum01234567, vi12x01234567);

        vacc0123 = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vacc0123), vget_low_u16(vprod01234567)));
        vacc4567 = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vacc4567), vget_high_u16(vprod01234567)));
        const uint8x8_t vi13x01234567 = vld1_u8(i13); i13 += 8;
        const uint8x8_t vk13x01234567 = vld1_u8((const void*) (k + 408));

        vprod01234567 = vmull_u8(vi13x01234567, vk13x01234567);
        vsum01234567 = vaddw_u8(vsum01234567, vi13x01234567);

        vacc0123 = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vacc0123), vget_low_u16(vprod01234567)));
        vacc4567 = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vacc4567), vget_high_u16(vprod01234567)));
        const uint8x8_t vi14x01234567 = vld1_u8(i14); i14 += 8;
        const uint8x8_t vk14x01234567 = vld1_u8((const void*) (k + 440));

        vprod01234567 = vmull_u8(vi14x01234567, vk14x01234567);
        vsum01234567 = vaddw_u8(vsum01234567, vi14x01234567);

        vacc0123 = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vacc0123), vget_low_u16(vprod01234567)));
        vacc4567 = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vacc4567), vget_high_u16(vprod01234567)));
        const uint8x8_t vi15x01234567 = vld1_u8(i15); i15 += 8;
        const uint8x8_t vk15x01234567 = vld1_u8((const void*) (k + 472));

        vprod01234567 = vmull_u8(vi15x01234567, vk15x01234567);
        vsum01234567 = vaddw_u8(vsum01234567, vi15x01234567);

        vacc0123 = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vacc0123), vget_low_u16(vprod01234567)));
        vacc4567 = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vacc4567), vget_high_u16(vprod01234567)));
        const uint8x8_t vi16x01234567 = vld1_u8(i16); i16 += 8;
        const uint8x8_t vk16x01234567 = vld1_u8((const void*) (k + 504));

        vprod01234567 = vmull_u8(vi16x01234567, vk16x01234567);
        vsum01234567 = vaddw_u8(vsum01234567, vi16x01234567);

        vacc0123 = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vacc0123), vget_low_u16(vprod01234567)));
        vacc4567 = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vacc4567), vget_high_u16(vprod01234567)));
        const uint8x8_t vi17x01234567 = vld1_u8(i17); i17 += 8;
        const uint8x8_t vk17x01234567 = vld1_u8((const void*) (k + 536));

        vprod01234567 = vmull_u8(vi17x01234567, vk17x01234567);
        vsum01234567 = vaddw_u8(vsum01234567, vi17x01234567);

        vacc0123 = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vacc0123), vget_low_u16(vprod01234567)));
        vacc4567 = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vacc4567), vget_high_u16(vprod01234567)));
        const uint8x8_t vi18x01234567 = vld1_u8(i18); i18 += 8;
        const uint8x8_t vk18x01234567 = vld1_u8((const void*) (k + 568));

        vprod01234567 = vmull_u8(vi18x01234567, vk18x01234567);
        vsum01234567 = vaddw_u8(vsum01234567, vi18x01234567);

        vacc0123 = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vacc0123), vget_low_u16(vprod01234567)));
        vacc4567 = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vacc4567), vget_high_u16(vprod01234567)));
        const uint8x8_t vi19x01234567 = vld1_u8(i19); i19 += 8;
        const uint8x8_t vk19x01234567 = vld1_u8((const void*) (k + 600));

        vprod01234567 = vmull_u8(vi19x01234567, vk19x01234567);
        vsum01234567 = vaddw_u8(vsum01234567, vi19x01234567);

        vacc0123 = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vacc0123), vget_low_u16(vprod01234567)));
        vacc4567 = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vacc4567), vget_high_u16(vprod01234567)));
        const uint8x8_t vi20x01234567 = vld1_u8(i20); i20 += 8;
        const uint8x8_t vk20x01234567 = vld1_u8((const void*) (k + 632));

        vprod01234567 = vmull_u8(vi20x01234567, vk20x01234567);
        vsum01234567 = vaddw_u8(vsum01234567, vi20x01234567);

        vacc0123 = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vacc0123), vget_low_u16(vprod01234567)));
        vacc4567 = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vacc4567), vget_high_u16(vprod01234567)));
        const uint8x8_t vi21x01234567 = vld1_u8(i21); i21 += 8;
        const uint8x8_t vk21x01234567 = vld1_u8((const void*) (k + 664));

        vprod01234567 = vmull_u8(vi21x01234567, vk21x01234567);
        vsum01234567 = vaddw_u8(vsum01234567, vi21x01234567);

        vacc0123 = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vacc0123), vget_low_u16(vprod01234567)));
        vacc4567 = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vacc4567), vget_high_u16(vprod01234567)));
        const uint8x8_t vi22x01234567 = vld1_u8(i22); i22 += 8;
        const uint8x8_t vk22x01234567 = vld1_u8((const void*) (k + 696));

        vprod01234567 = vmull_u8(vi22x01234567, vk22x01234567);
        vsum01234567 = vaddw_u8(vsum01234567, vi22x01234567);

        vacc0123 = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vacc0123), vget_low_u16(vprod01234567)));
        vacc4567 = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vacc4567), vget_high_u16(vprod01234567)));
        const uint8x8_t vi23x01234567 = vld1_u8(i23); i23 += 8;
        const uint8x8_t vk23x01234567 = vld1_u8((const void*) (k + 728));

        vprod01234567 = vmull_u8(vi23x01234567, vk23x01234567);
        vsum01234567 = vaddw_u8(vsum01234567, vi23x01234567);

        vacc0123 = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vacc0123), vget_low_u16(vprod01234567)));
        vacc4567 = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vacc4567), vget_high_u16(vprod01234567)));
        const uint8x8_t vi24x01234567 = vld1_u8(i24); i24 += 8;
        const uint8x8_t vk24x01234567 = vld1_u8((const void*) (k + 760));

        vprod01234567 = vmull_u8(vi24x01234567, vk24x01234567);
        vsum01234567 = vaddw_u8(vsum01234567, vi24x01234567);

        vacc0123 = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vacc0123), vget_low_u16(vprod01234567)));
        vacc4567 = vreinterpretq_s32_u32(vaddw_u16(vreinterpretq_u32_s32(vacc4567), vget_high_u16(vprod01234567)));

        vacc0123 = vreinterpretq_s32_u32(vmlsl_u16(vreinterpretq_u32_s32(vacc0123), vget_low_u16(vsum01234567), vget_low_u16(vkernel_zero_point)));
        vacc4567 = vreinterpretq_s32_u32(vmlsl_u16(vreinterpretq_u32_s32(vacc4567), vget_high_u16(vsum01234567), vget_high_u16(vkernel_zero_point)));

        vacc0123 = vrshlq_s32(vacc0123, vright_pre_shift);
        vacc4567 = vrshlq_s32(vacc4567, vright_pre_shift);

        vacc0123 = vqdmulhq_s32(vacc0123, vmultiplier);
        vacc4567 = vqdmulhq_s32(vacc4567, vmultiplier);

        vacc0123 = vrshlq_s32(vacc0123, vright_post_shift);
        vacc4567 = vrshlq_s32(vacc4567, vright_post_shift);

#if XNN_ARCH_ARM64
        const int16x8_t vacc01234567 = vqaddq_s16(vqmovn_high_s32(vqmovn_s32(vacc0123), vacc4567), voutput_zero_point);
        uint8x8_t vout01234567 = vqmovun_s16(vacc01234567);
#else
        const int16x8_t vacc01234567 = vqaddq_s16(vcombine_s16(vqmovn_s32(vacc0123), vqmovn_s32(vacc4567)), voutput_zero_point);
        uint8x8_t vout01234567 = vqmovun_s16(vacc01234567);
#endif

        vout01234567 = vmax_u8(vout01234567, vget_low_u8(voutput_min));
        vout01234567 = vmin_u8(vout01234567, vget_low_u8(voutput_max));

        if XNN_LIKELY(c >= 8) {
          vst1_u8(output, vout01234567); output += 8;
          c -= 8;
        } else {
          if (c & 4) {
            vst1_lane_u32((void*) output, vreinterpret_u32_u8(vout01234567), 0); output += 4;
            vout01234567 = vext_u8(vout01234567, vout01234567, 4);
          }
          if (c & 2) {
            vst1_lane_u16((void*) output, vreinterpret_u16_u8(vout01234567), 0); output += 2;
            vout01234567 = vext_u8(vout01234567, vout01234567, 2);
          }
          if (c & 1) {
            vst1_lane_u8(output, vout01234567, 0); output += 1;
          }
          c = 0;
        }
      } while (c != 0);
    }

    output = (uint8_t*) ((uintptr_t) output + output_increment);
  } while (--output_width != 0);
}
