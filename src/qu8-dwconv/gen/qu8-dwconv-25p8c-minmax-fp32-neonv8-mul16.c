// Auto-generated file. Do not edit!
//   Template: src/qs8-dwconv/unipass-neon-mul16.c.in
//   Generator: tools/xngen
//
// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <arm_neon.h>

#include "xnnpack/dwconv.h"
#include "xnnpack/intrinsics-polyfill.h"


void xnn_qu8_dwconv_minmax_fp32_ukernel_25p8c__neonv8_mul16(
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

  const uint8x8_t vkernel_zero_point = vld1_dup_u8(params->fp32_neonv8.kernel_zero_point);
  const float32x4_t vscale = vld1q_dup_f32(&params->fp32_neonv8.scale);
  const int16x8_t voutput_zero_point = vld1q_dup_s16(&params->fp32_neonv8.output_zero_point);
  const uint8x8_t voutput_min = vld1_dup_u8(&params->fp32_neonv8.output_min);
  const uint8x8_t voutput_max = vld1_dup_u8(&params->fp32_neonv8.output_max);
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
    for (; c >= 8; c -= 8) {
      int32x4_t vacc0123 = vld1q_s32(w); w = (const void*) ((const int32_t*) w + 4);
      int32x4_t vacc4567 = vld1q_s32(w); w = (const void*) ((const int32_t*) w + 4);


      const int16x8_t vi0x01234567 = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(i0))); i0 += 8;
      const int16x8_t vk0x01234567 = vreinterpretq_s16_u16(vsubl_u8(vld1_u8(w), vkernel_zero_point)); w = (const uint8_t*) w + 8;

      vacc0123 = vmlal_s16(vacc0123, vget_low_s16(vi0x01234567), vget_low_s16(vk0x01234567));
      vacc4567 = vmlal_s16(vacc4567, vget_high_s16(vi0x01234567), vget_high_s16(vk0x01234567));

      const int16x8_t vi1x01234567 = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(i1))); i1 += 8;
      const int16x8_t vk1x01234567 = vreinterpretq_s16_u16(vsubl_u8(vld1_u8(w), vkernel_zero_point)); w = (const uint8_t*) w + 8;

      vacc0123 = vmlal_s16(vacc0123, vget_low_s16(vi1x01234567), vget_low_s16(vk1x01234567));
      vacc4567 = vmlal_s16(vacc4567, vget_high_s16(vi1x01234567), vget_high_s16(vk1x01234567));

      const int16x8_t vi2x01234567 = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(i2))); i2 += 8;
      const int16x8_t vk2x01234567 = vreinterpretq_s16_u16(vsubl_u8(vld1_u8(w), vkernel_zero_point)); w = (const uint8_t*) w + 8;

      vacc0123 = vmlal_s16(vacc0123, vget_low_s16(vi2x01234567), vget_low_s16(vk2x01234567));
      vacc4567 = vmlal_s16(vacc4567, vget_high_s16(vi2x01234567), vget_high_s16(vk2x01234567));

      const int16x8_t vi3x01234567 = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(i3))); i3 += 8;
      const int16x8_t vk3x01234567 = vreinterpretq_s16_u16(vsubl_u8(vld1_u8(w), vkernel_zero_point)); w = (const uint8_t*) w + 8;

      vacc0123 = vmlal_s16(vacc0123, vget_low_s16(vi3x01234567), vget_low_s16(vk3x01234567));
      vacc4567 = vmlal_s16(vacc4567, vget_high_s16(vi3x01234567), vget_high_s16(vk3x01234567));

      const int16x8_t vi4x01234567 = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(i4))); i4 += 8;
      const int16x8_t vk4x01234567 = vreinterpretq_s16_u16(vsubl_u8(vld1_u8(w), vkernel_zero_point)); w = (const uint8_t*) w + 8;

      vacc0123 = vmlal_s16(vacc0123, vget_low_s16(vi4x01234567), vget_low_s16(vk4x01234567));
      vacc4567 = vmlal_s16(vacc4567, vget_high_s16(vi4x01234567), vget_high_s16(vk4x01234567));

      const int16x8_t vi5x01234567 = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(i5))); i5 += 8;
      const int16x8_t vk5x01234567 = vreinterpretq_s16_u16(vsubl_u8(vld1_u8(w), vkernel_zero_point)); w = (const uint8_t*) w + 8;

      vacc0123 = vmlal_s16(vacc0123, vget_low_s16(vi5x01234567), vget_low_s16(vk5x01234567));
      vacc4567 = vmlal_s16(vacc4567, vget_high_s16(vi5x01234567), vget_high_s16(vk5x01234567));

      const int16x8_t vi6x01234567 = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(i6))); i6 += 8;
      const int16x8_t vk6x01234567 = vreinterpretq_s16_u16(vsubl_u8(vld1_u8(w), vkernel_zero_point)); w = (const uint8_t*) w + 8;

      vacc0123 = vmlal_s16(vacc0123, vget_low_s16(vi6x01234567), vget_low_s16(vk6x01234567));
      vacc4567 = vmlal_s16(vacc4567, vget_high_s16(vi6x01234567), vget_high_s16(vk6x01234567));

      const int16x8_t vi7x01234567 = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(i7))); i7 += 8;
      const int16x8_t vk7x01234567 = vreinterpretq_s16_u16(vsubl_u8(vld1_u8(w), vkernel_zero_point)); w = (const uint8_t*) w + 8;

      vacc0123 = vmlal_s16(vacc0123, vget_low_s16(vi7x01234567), vget_low_s16(vk7x01234567));
      vacc4567 = vmlal_s16(vacc4567, vget_high_s16(vi7x01234567), vget_high_s16(vk7x01234567));

      const int16x8_t vi8x01234567 = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(i8))); i8 += 8;
      const int16x8_t vk8x01234567 = vreinterpretq_s16_u16(vsubl_u8(vld1_u8(w), vkernel_zero_point)); w = (const uint8_t*) w + 8;

      vacc0123 = vmlal_s16(vacc0123, vget_low_s16(vi8x01234567), vget_low_s16(vk8x01234567));
      vacc4567 = vmlal_s16(vacc4567, vget_high_s16(vi8x01234567), vget_high_s16(vk8x01234567));

      const int16x8_t vi9x01234567 = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(i9))); i9 += 8;
      const int16x8_t vk9x01234567 = vreinterpretq_s16_u16(vsubl_u8(vld1_u8(w), vkernel_zero_point)); w = (const uint8_t*) w + 8;

      vacc0123 = vmlal_s16(vacc0123, vget_low_s16(vi9x01234567), vget_low_s16(vk9x01234567));
      vacc4567 = vmlal_s16(vacc4567, vget_high_s16(vi9x01234567), vget_high_s16(vk9x01234567));

      const int16x8_t vi10x01234567 = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(i10))); i10 += 8;
      const int16x8_t vk10x01234567 = vreinterpretq_s16_u16(vsubl_u8(vld1_u8(w), vkernel_zero_point)); w = (const uint8_t*) w + 8;

      vacc0123 = vmlal_s16(vacc0123, vget_low_s16(vi10x01234567), vget_low_s16(vk10x01234567));
      vacc4567 = vmlal_s16(vacc4567, vget_high_s16(vi10x01234567), vget_high_s16(vk10x01234567));

      const int16x8_t vi11x01234567 = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(i11))); i11 += 8;
      const int16x8_t vk11x01234567 = vreinterpretq_s16_u16(vsubl_u8(vld1_u8(w), vkernel_zero_point)); w = (const uint8_t*) w + 8;

      vacc0123 = vmlal_s16(vacc0123, vget_low_s16(vi11x01234567), vget_low_s16(vk11x01234567));
      vacc4567 = vmlal_s16(vacc4567, vget_high_s16(vi11x01234567), vget_high_s16(vk11x01234567));

      const int16x8_t vi12x01234567 = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(i12))); i12 += 8;
      const int16x8_t vk12x01234567 = vreinterpretq_s16_u16(vsubl_u8(vld1_u8(w), vkernel_zero_point)); w = (const uint8_t*) w + 8;

      vacc0123 = vmlal_s16(vacc0123, vget_low_s16(vi12x01234567), vget_low_s16(vk12x01234567));
      vacc4567 = vmlal_s16(vacc4567, vget_high_s16(vi12x01234567), vget_high_s16(vk12x01234567));

      const int16x8_t vi13x01234567 = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(i13))); i13 += 8;
      const int16x8_t vk13x01234567 = vreinterpretq_s16_u16(vsubl_u8(vld1_u8(w), vkernel_zero_point)); w = (const uint8_t*) w + 8;

      vacc0123 = vmlal_s16(vacc0123, vget_low_s16(vi13x01234567), vget_low_s16(vk13x01234567));
      vacc4567 = vmlal_s16(vacc4567, vget_high_s16(vi13x01234567), vget_high_s16(vk13x01234567));

      const int16x8_t vi14x01234567 = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(i14))); i14 += 8;
      const int16x8_t vk14x01234567 = vreinterpretq_s16_u16(vsubl_u8(vld1_u8(w), vkernel_zero_point)); w = (const uint8_t*) w + 8;

      vacc0123 = vmlal_s16(vacc0123, vget_low_s16(vi14x01234567), vget_low_s16(vk14x01234567));
      vacc4567 = vmlal_s16(vacc4567, vget_high_s16(vi14x01234567), vget_high_s16(vk14x01234567));

      const int16x8_t vi15x01234567 = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(i15))); i15 += 8;
      const int16x8_t vk15x01234567 = vreinterpretq_s16_u16(vsubl_u8(vld1_u8(w), vkernel_zero_point)); w = (const uint8_t*) w + 8;

      vacc0123 = vmlal_s16(vacc0123, vget_low_s16(vi15x01234567), vget_low_s16(vk15x01234567));
      vacc4567 = vmlal_s16(vacc4567, vget_high_s16(vi15x01234567), vget_high_s16(vk15x01234567));

      const int16x8_t vi16x01234567 = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(i16))); i16 += 8;
      const int16x8_t vk16x01234567 = vreinterpretq_s16_u16(vsubl_u8(vld1_u8(w), vkernel_zero_point)); w = (const uint8_t*) w + 8;

      vacc0123 = vmlal_s16(vacc0123, vget_low_s16(vi16x01234567), vget_low_s16(vk16x01234567));
      vacc4567 = vmlal_s16(vacc4567, vget_high_s16(vi16x01234567), vget_high_s16(vk16x01234567));

      const int16x8_t vi17x01234567 = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(i17))); i17 += 8;
      const int16x8_t vk17x01234567 = vreinterpretq_s16_u16(vsubl_u8(vld1_u8(w), vkernel_zero_point)); w = (const uint8_t*) w + 8;

      vacc0123 = vmlal_s16(vacc0123, vget_low_s16(vi17x01234567), vget_low_s16(vk17x01234567));
      vacc4567 = vmlal_s16(vacc4567, vget_high_s16(vi17x01234567), vget_high_s16(vk17x01234567));

      const int16x8_t vi18x01234567 = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(i18))); i18 += 8;
      const int16x8_t vk18x01234567 = vreinterpretq_s16_u16(vsubl_u8(vld1_u8(w), vkernel_zero_point)); w = (const uint8_t*) w + 8;

      vacc0123 = vmlal_s16(vacc0123, vget_low_s16(vi18x01234567), vget_low_s16(vk18x01234567));
      vacc4567 = vmlal_s16(vacc4567, vget_high_s16(vi18x01234567), vget_high_s16(vk18x01234567));

      const int16x8_t vi19x01234567 = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(i19))); i19 += 8;
      const int16x8_t vk19x01234567 = vreinterpretq_s16_u16(vsubl_u8(vld1_u8(w), vkernel_zero_point)); w = (const uint8_t*) w + 8;

      vacc0123 = vmlal_s16(vacc0123, vget_low_s16(vi19x01234567), vget_low_s16(vk19x01234567));
      vacc4567 = vmlal_s16(vacc4567, vget_high_s16(vi19x01234567), vget_high_s16(vk19x01234567));

      const int16x8_t vi20x01234567 = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(i20))); i20 += 8;
      const int16x8_t vk20x01234567 = vreinterpretq_s16_u16(vsubl_u8(vld1_u8(w), vkernel_zero_point)); w = (const uint8_t*) w + 8;

      vacc0123 = vmlal_s16(vacc0123, vget_low_s16(vi20x01234567), vget_low_s16(vk20x01234567));
      vacc4567 = vmlal_s16(vacc4567, vget_high_s16(vi20x01234567), vget_high_s16(vk20x01234567));

      const int16x8_t vi21x01234567 = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(i21))); i21 += 8;
      const int16x8_t vk21x01234567 = vreinterpretq_s16_u16(vsubl_u8(vld1_u8(w), vkernel_zero_point)); w = (const uint8_t*) w + 8;

      vacc0123 = vmlal_s16(vacc0123, vget_low_s16(vi21x01234567), vget_low_s16(vk21x01234567));
      vacc4567 = vmlal_s16(vacc4567, vget_high_s16(vi21x01234567), vget_high_s16(vk21x01234567));

      const int16x8_t vi22x01234567 = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(i22))); i22 += 8;
      const int16x8_t vk22x01234567 = vreinterpretq_s16_u16(vsubl_u8(vld1_u8(w), vkernel_zero_point)); w = (const uint8_t*) w + 8;

      vacc0123 = vmlal_s16(vacc0123, vget_low_s16(vi22x01234567), vget_low_s16(vk22x01234567));
      vacc4567 = vmlal_s16(vacc4567, vget_high_s16(vi22x01234567), vget_high_s16(vk22x01234567));

      const int16x8_t vi23x01234567 = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(i23))); i23 += 8;
      const int16x8_t vk23x01234567 = vreinterpretq_s16_u16(vsubl_u8(vld1_u8(w), vkernel_zero_point)); w = (const uint8_t*) w + 8;

      vacc0123 = vmlal_s16(vacc0123, vget_low_s16(vi23x01234567), vget_low_s16(vk23x01234567));
      vacc4567 = vmlal_s16(vacc4567, vget_high_s16(vi23x01234567), vget_high_s16(vk23x01234567));

      const int16x8_t vi24x01234567 = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(i24))); i24 += 8;
      const int16x8_t vk24x01234567 = vreinterpretq_s16_u16(vsubl_u8(vld1_u8(w), vkernel_zero_point)); w = (const uint8_t*) w + 8;

      vacc0123 = vmlal_s16(vacc0123, vget_low_s16(vi24x01234567), vget_low_s16(vk24x01234567));
      vacc4567 = vmlal_s16(vacc4567, vget_high_s16(vi24x01234567), vget_high_s16(vk24x01234567));

      float32x4_t vfpacc0123 = vcvtq_f32_s32(vacc0123);
      float32x4_t vfpacc4567 = vcvtq_f32_s32(vacc4567);

      vfpacc0123 = vmulq_f32(vfpacc0123, vscale);
      vfpacc4567 = vmulq_f32(vfpacc4567, vscale);

      vacc0123 = vcvtnq_s32_f32(vfpacc0123);
      vacc4567 = vcvtnq_s32_f32(vfpacc4567);

#if XNN_ARCH_ARM64
      int16x8_t vacc01234567 = vqmovn_high_s32(vqmovn_s32(vacc0123), vacc4567);

      vacc01234567 = vqaddq_s16(vacc01234567, voutput_zero_point);

      uint8x8_t vout01234567 = vqmovun_s16(vacc01234567);
#else  // !XNN_ARCH_ARM64
      int16x8_t vacc01234567 = vcombine_s16(vqmovn_s32(vacc0123), vqmovn_s32(vacc4567));

      vacc01234567 = vqaddq_s16(vacc01234567, voutput_zero_point);

      uint8x8_t vout01234567 = vqmovun_s16(vacc01234567);
#endif  // !XNN_ARCH_ARM64

      vout01234567 = vmax_u8(vout01234567, voutput_min);

      vout01234567 = vmin_u8(vout01234567, voutput_max);

      vst1_u8(output, vout01234567); output += 8;
    }
    if XNN_UNLIKELY(c != 0) {
      {
        int32x4_t vacc0123 = vld1q_s32(w); w = (const void*) ((const int32_t*) w + 4);
        int32x4_t vacc4567 = vld1q_s32(w); w = (const void*) ((const int32_t*) w + 4);

        const int16x8_t vi0x01234567 = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(i0)));
        const int16x8_t vk0x01234567 = vreinterpretq_s16_u16(vsubl_u8(vld1_u8(w), vkernel_zero_point));

        vacc0123 = vmlal_s16(vacc0123, vget_low_s16(vi0x01234567), vget_low_s16(vk0x01234567));
        vacc4567 = vmlal_s16(vacc4567, vget_high_s16(vi0x01234567), vget_high_s16(vk0x01234567));
        const int16x8_t vi1x01234567 = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(i1)));
        const int16x8_t vk1x01234567 = vreinterpretq_s16_u16(vsubl_u8(vld1_u8((const void*) ((const uint8_t*) w + 8)), vkernel_zero_point));

        vacc0123 = vmlal_s16(vacc0123, vget_low_s16(vi1x01234567), vget_low_s16(vk1x01234567));
        vacc4567 = vmlal_s16(vacc4567, vget_high_s16(vi1x01234567), vget_high_s16(vk1x01234567));
        const int16x8_t vi2x01234567 = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(i2)));
        const int16x8_t vk2x01234567 = vreinterpretq_s16_u16(vsubl_u8(vld1_u8((const void*) ((const uint8_t*) w + 16)), vkernel_zero_point));

        vacc0123 = vmlal_s16(vacc0123, vget_low_s16(vi2x01234567), vget_low_s16(vk2x01234567));
        vacc4567 = vmlal_s16(vacc4567, vget_high_s16(vi2x01234567), vget_high_s16(vk2x01234567));
        const int16x8_t vi3x01234567 = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(i3)));
        const int16x8_t vk3x01234567 = vreinterpretq_s16_u16(vsubl_u8(vld1_u8((const void*) ((const uint8_t*) w + 24)), vkernel_zero_point));

        vacc0123 = vmlal_s16(vacc0123, vget_low_s16(vi3x01234567), vget_low_s16(vk3x01234567));
        vacc4567 = vmlal_s16(vacc4567, vget_high_s16(vi3x01234567), vget_high_s16(vk3x01234567));
        const int16x8_t vi4x01234567 = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(i4)));
        const int16x8_t vk4x01234567 = vreinterpretq_s16_u16(vsubl_u8(vld1_u8((const void*) ((const uint8_t*) w + 32)), vkernel_zero_point));

        vacc0123 = vmlal_s16(vacc0123, vget_low_s16(vi4x01234567), vget_low_s16(vk4x01234567));
        vacc4567 = vmlal_s16(vacc4567, vget_high_s16(vi4x01234567), vget_high_s16(vk4x01234567));
        const int16x8_t vi5x01234567 = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(i5)));
        const int16x8_t vk5x01234567 = vreinterpretq_s16_u16(vsubl_u8(vld1_u8((const void*) ((const uint8_t*) w + 40)), vkernel_zero_point));

        vacc0123 = vmlal_s16(vacc0123, vget_low_s16(vi5x01234567), vget_low_s16(vk5x01234567));
        vacc4567 = vmlal_s16(vacc4567, vget_high_s16(vi5x01234567), vget_high_s16(vk5x01234567));
        const int16x8_t vi6x01234567 = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(i6)));
        const int16x8_t vk6x01234567 = vreinterpretq_s16_u16(vsubl_u8(vld1_u8((const void*) ((const uint8_t*) w + 48)), vkernel_zero_point));

        vacc0123 = vmlal_s16(vacc0123, vget_low_s16(vi6x01234567), vget_low_s16(vk6x01234567));
        vacc4567 = vmlal_s16(vacc4567, vget_high_s16(vi6x01234567), vget_high_s16(vk6x01234567));
        const int16x8_t vi7x01234567 = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(i7)));
        const int16x8_t vk7x01234567 = vreinterpretq_s16_u16(vsubl_u8(vld1_u8((const void*) ((const uint8_t*) w + 56)), vkernel_zero_point));

        vacc0123 = vmlal_s16(vacc0123, vget_low_s16(vi7x01234567), vget_low_s16(vk7x01234567));
        vacc4567 = vmlal_s16(vacc4567, vget_high_s16(vi7x01234567), vget_high_s16(vk7x01234567));
        const int16x8_t vi8x01234567 = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(i8)));
        const int16x8_t vk8x01234567 = vreinterpretq_s16_u16(vsubl_u8(vld1_u8((const void*) ((const uint8_t*) w + 64)), vkernel_zero_point));

        vacc0123 = vmlal_s16(vacc0123, vget_low_s16(vi8x01234567), vget_low_s16(vk8x01234567));
        vacc4567 = vmlal_s16(vacc4567, vget_high_s16(vi8x01234567), vget_high_s16(vk8x01234567));
        const int16x8_t vi9x01234567 = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(i9)));
        const int16x8_t vk9x01234567 = vreinterpretq_s16_u16(vsubl_u8(vld1_u8((const void*) ((const uint8_t*) w + 72)), vkernel_zero_point));

        vacc0123 = vmlal_s16(vacc0123, vget_low_s16(vi9x01234567), vget_low_s16(vk9x01234567));
        vacc4567 = vmlal_s16(vacc4567, vget_high_s16(vi9x01234567), vget_high_s16(vk9x01234567));
        const int16x8_t vi10x01234567 = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(i10)));
        const int16x8_t vk10x01234567 = vreinterpretq_s16_u16(vsubl_u8(vld1_u8((const void*) ((const uint8_t*) w + 80)), vkernel_zero_point));

        vacc0123 = vmlal_s16(vacc0123, vget_low_s16(vi10x01234567), vget_low_s16(vk10x01234567));
        vacc4567 = vmlal_s16(vacc4567, vget_high_s16(vi10x01234567), vget_high_s16(vk10x01234567));
        const int16x8_t vi11x01234567 = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(i11)));
        const int16x8_t vk11x01234567 = vreinterpretq_s16_u16(vsubl_u8(vld1_u8((const void*) ((const uint8_t*) w + 88)), vkernel_zero_point));

        vacc0123 = vmlal_s16(vacc0123, vget_low_s16(vi11x01234567), vget_low_s16(vk11x01234567));
        vacc4567 = vmlal_s16(vacc4567, vget_high_s16(vi11x01234567), vget_high_s16(vk11x01234567));
        const int16x8_t vi12x01234567 = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(i12)));
        const int16x8_t vk12x01234567 = vreinterpretq_s16_u16(vsubl_u8(vld1_u8((const void*) ((const uint8_t*) w + 96)), vkernel_zero_point));

        vacc0123 = vmlal_s16(vacc0123, vget_low_s16(vi12x01234567), vget_low_s16(vk12x01234567));
        vacc4567 = vmlal_s16(vacc4567, vget_high_s16(vi12x01234567), vget_high_s16(vk12x01234567));
        const int16x8_t vi13x01234567 = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(i13)));
        const int16x8_t vk13x01234567 = vreinterpretq_s16_u16(vsubl_u8(vld1_u8((const void*) ((const uint8_t*) w + 104)), vkernel_zero_point));

        vacc0123 = vmlal_s16(vacc0123, vget_low_s16(vi13x01234567), vget_low_s16(vk13x01234567));
        vacc4567 = vmlal_s16(vacc4567, vget_high_s16(vi13x01234567), vget_high_s16(vk13x01234567));
        const int16x8_t vi14x01234567 = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(i14)));
        const int16x8_t vk14x01234567 = vreinterpretq_s16_u16(vsubl_u8(vld1_u8((const void*) ((const uint8_t*) w + 112)), vkernel_zero_point));

        vacc0123 = vmlal_s16(vacc0123, vget_low_s16(vi14x01234567), vget_low_s16(vk14x01234567));
        vacc4567 = vmlal_s16(vacc4567, vget_high_s16(vi14x01234567), vget_high_s16(vk14x01234567));
        const int16x8_t vi15x01234567 = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(i15)));
        const int16x8_t vk15x01234567 = vreinterpretq_s16_u16(vsubl_u8(vld1_u8((const void*) ((const uint8_t*) w + 120)), vkernel_zero_point));

        vacc0123 = vmlal_s16(vacc0123, vget_low_s16(vi15x01234567), vget_low_s16(vk15x01234567));
        vacc4567 = vmlal_s16(vacc4567, vget_high_s16(vi15x01234567), vget_high_s16(vk15x01234567));
        const int16x8_t vi16x01234567 = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(i16)));
        const int16x8_t vk16x01234567 = vreinterpretq_s16_u16(vsubl_u8(vld1_u8((const void*) ((const uint8_t*) w + 128)), vkernel_zero_point));

        vacc0123 = vmlal_s16(vacc0123, vget_low_s16(vi16x01234567), vget_low_s16(vk16x01234567));
        vacc4567 = vmlal_s16(vacc4567, vget_high_s16(vi16x01234567), vget_high_s16(vk16x01234567));
        const int16x8_t vi17x01234567 = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(i17)));
        const int16x8_t vk17x01234567 = vreinterpretq_s16_u16(vsubl_u8(vld1_u8((const void*) ((const uint8_t*) w + 136)), vkernel_zero_point));

        vacc0123 = vmlal_s16(vacc0123, vget_low_s16(vi17x01234567), vget_low_s16(vk17x01234567));
        vacc4567 = vmlal_s16(vacc4567, vget_high_s16(vi17x01234567), vget_high_s16(vk17x01234567));
        const int16x8_t vi18x01234567 = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(i18)));
        const int16x8_t vk18x01234567 = vreinterpretq_s16_u16(vsubl_u8(vld1_u8((const void*) ((const uint8_t*) w + 144)), vkernel_zero_point));

        vacc0123 = vmlal_s16(vacc0123, vget_low_s16(vi18x01234567), vget_low_s16(vk18x01234567));
        vacc4567 = vmlal_s16(vacc4567, vget_high_s16(vi18x01234567), vget_high_s16(vk18x01234567));
        const int16x8_t vi19x01234567 = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(i19)));
        const int16x8_t vk19x01234567 = vreinterpretq_s16_u16(vsubl_u8(vld1_u8((const void*) ((const uint8_t*) w + 152)), vkernel_zero_point));

        vacc0123 = vmlal_s16(vacc0123, vget_low_s16(vi19x01234567), vget_low_s16(vk19x01234567));
        vacc4567 = vmlal_s16(vacc4567, vget_high_s16(vi19x01234567), vget_high_s16(vk19x01234567));
        const int16x8_t vi20x01234567 = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(i20)));
        const int16x8_t vk20x01234567 = vreinterpretq_s16_u16(vsubl_u8(vld1_u8((const void*) ((const uint8_t*) w + 160)), vkernel_zero_point));

        vacc0123 = vmlal_s16(vacc0123, vget_low_s16(vi20x01234567), vget_low_s16(vk20x01234567));
        vacc4567 = vmlal_s16(vacc4567, vget_high_s16(vi20x01234567), vget_high_s16(vk20x01234567));
        const int16x8_t vi21x01234567 = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(i21)));
        const int16x8_t vk21x01234567 = vreinterpretq_s16_u16(vsubl_u8(vld1_u8((const void*) ((const uint8_t*) w + 168)), vkernel_zero_point));

        vacc0123 = vmlal_s16(vacc0123, vget_low_s16(vi21x01234567), vget_low_s16(vk21x01234567));
        vacc4567 = vmlal_s16(vacc4567, vget_high_s16(vi21x01234567), vget_high_s16(vk21x01234567));
        const int16x8_t vi22x01234567 = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(i22)));
        const int16x8_t vk22x01234567 = vreinterpretq_s16_u16(vsubl_u8(vld1_u8((const void*) ((const uint8_t*) w + 176)), vkernel_zero_point));

        vacc0123 = vmlal_s16(vacc0123, vget_low_s16(vi22x01234567), vget_low_s16(vk22x01234567));
        vacc4567 = vmlal_s16(vacc4567, vget_high_s16(vi22x01234567), vget_high_s16(vk22x01234567));
        const int16x8_t vi23x01234567 = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(i23)));
        const int16x8_t vk23x01234567 = vreinterpretq_s16_u16(vsubl_u8(vld1_u8((const void*) ((const uint8_t*) w + 184)), vkernel_zero_point));

        vacc0123 = vmlal_s16(vacc0123, vget_low_s16(vi23x01234567), vget_low_s16(vk23x01234567));
        vacc4567 = vmlal_s16(vacc4567, vget_high_s16(vi23x01234567), vget_high_s16(vk23x01234567));
        const int16x8_t vi24x01234567 = vreinterpretq_s16_u16(vmovl_u8(vld1_u8(i24)));
        const int16x8_t vk24x01234567 = vreinterpretq_s16_u16(vsubl_u8(vld1_u8((const void*) ((const uint8_t*) w + 192)), vkernel_zero_point));

        vacc0123 = vmlal_s16(vacc0123, vget_low_s16(vi24x01234567), vget_low_s16(vk24x01234567));
        vacc4567 = vmlal_s16(vacc4567, vget_high_s16(vi24x01234567), vget_high_s16(vk24x01234567));

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
        vout01234567 = vmax_u8(vout01234567, voutput_min);
        vout01234567 = vmin_u8(vout01234567, voutput_max);

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
      }
    }

    output = (uint8_t*) ((uintptr_t) output + output_increment);
  } while (--output_width != 0);
}
