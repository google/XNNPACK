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

#include <xnnpack/dwconv.h>


void xnn_qc8_dwconv_minmax_fp32_ukernel_up24x25__neon_mul16(
    size_t channels,
    size_t output_width,
    const int8_t** input,
    const void* weights,
    int8_t* output,
    size_t input_stride,
    size_t output_increment,
    size_t input_offset,
    const int8_t* zero,
    const union xnn_qs8_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_DISABLE_TSAN XNN_DISABLE_MSAN
{
  assert(channels != 0);
  assert(output_width != 0);

  const float32x4_t voutput_min_less_zero_point = vld1q_dup_f32(&params->neon_fp32.output_min_less_zero_point);
  const float32x4_t voutput_max_less_zero_point = vld1q_dup_f32(&params->neon_fp32.output_max_less_zero_point);
  const float32x4_t vmagic_bias = vld1q_dup_f32(&params->neon_fp32.magic_bias);
  const int32x4_t vmagic_bias_less_zero_point = vld1q_dup_s32(&params->neon_fp32.magic_bias_less_zero_point);
  do {
    const int8_t* i0 = input[0];
    assert(i0 != NULL);
    if XNN_UNPREDICTABLE(i0 != zero) {
      i0 = (const int8_t*) ((uintptr_t) i0 + input_offset);
    }
    const int8_t* i1 = input[1];
    assert(i1 != NULL);
    if XNN_UNPREDICTABLE(i1 != zero) {
      i1 = (const int8_t*) ((uintptr_t) i1 + input_offset);
    }
    const int8_t* i2 = input[2];
    assert(i2 != NULL);
    if XNN_UNPREDICTABLE(i2 != zero) {
      i2 = (const int8_t*) ((uintptr_t) i2 + input_offset);
    }
    const int8_t* i3 = input[3];
    assert(i3 != NULL);
    if XNN_UNPREDICTABLE(i3 != zero) {
      i3 = (const int8_t*) ((uintptr_t) i3 + input_offset);
    }
    const int8_t* i4 = input[4];
    assert(i4 != NULL);
    if XNN_UNPREDICTABLE(i4 != zero) {
      i4 = (const int8_t*) ((uintptr_t) i4 + input_offset);
    }
    const int8_t* i5 = input[5];
    assert(i5 != NULL);
    if XNN_UNPREDICTABLE(i5 != zero) {
      i5 = (const int8_t*) ((uintptr_t) i5 + input_offset);
    }
    const int8_t* i6 = input[6];
    assert(i6 != NULL);
    if XNN_UNPREDICTABLE(i6 != zero) {
      i6 = (const int8_t*) ((uintptr_t) i6 + input_offset);
    }
    const int8_t* i7 = input[7];
    assert(i7 != NULL);
    if XNN_UNPREDICTABLE(i7 != zero) {
      i7 = (const int8_t*) ((uintptr_t) i7 + input_offset);
    }
    const int8_t* i8 = input[8];
    assert(i8 != NULL);
    if XNN_UNPREDICTABLE(i8 != zero) {
      i8 = (const int8_t*) ((uintptr_t) i8 + input_offset);
    }
    const int8_t* i9 = input[9];
    assert(i9 != NULL);
    if XNN_UNPREDICTABLE(i9 != zero) {
      i9 = (const int8_t*) ((uintptr_t) i9 + input_offset);
    }
    const int8_t* i10 = input[10];
    assert(i10 != NULL);
    if XNN_UNPREDICTABLE(i10 != zero) {
      i10 = (const int8_t*) ((uintptr_t) i10 + input_offset);
    }
    const int8_t* i11 = input[11];
    assert(i11 != NULL);
    if XNN_UNPREDICTABLE(i11 != zero) {
      i11 = (const int8_t*) ((uintptr_t) i11 + input_offset);
    }
    const int8_t* i12 = input[12];
    assert(i12 != NULL);
    if XNN_UNPREDICTABLE(i12 != zero) {
      i12 = (const int8_t*) ((uintptr_t) i12 + input_offset);
    }
    const int8_t* i13 = input[13];
    assert(i13 != NULL);
    if XNN_UNPREDICTABLE(i13 != zero) {
      i13 = (const int8_t*) ((uintptr_t) i13 + input_offset);
    }
    const int8_t* i14 = input[14];
    assert(i14 != NULL);
    if XNN_UNPREDICTABLE(i14 != zero) {
      i14 = (const int8_t*) ((uintptr_t) i14 + input_offset);
    }
    const int8_t* i15 = input[15];
    assert(i15 != NULL);
    if XNN_UNPREDICTABLE(i15 != zero) {
      i15 = (const int8_t*) ((uintptr_t) i15 + input_offset);
    }
    const int8_t* i16 = input[16];
    assert(i16 != NULL);
    if XNN_UNPREDICTABLE(i16 != zero) {
      i16 = (const int8_t*) ((uintptr_t) i16 + input_offset);
    }
    const int8_t* i17 = input[17];
    assert(i17 != NULL);
    if XNN_UNPREDICTABLE(i17 != zero) {
      i17 = (const int8_t*) ((uintptr_t) i17 + input_offset);
    }
    const int8_t* i18 = input[18];
    assert(i18 != NULL);
    if XNN_UNPREDICTABLE(i18 != zero) {
      i18 = (const int8_t*) ((uintptr_t) i18 + input_offset);
    }
    const int8_t* i19 = input[19];
    assert(i19 != NULL);
    if XNN_UNPREDICTABLE(i19 != zero) {
      i19 = (const int8_t*) ((uintptr_t) i19 + input_offset);
    }
    const int8_t* i20 = input[20];
    assert(i20 != NULL);
    if XNN_UNPREDICTABLE(i20 != zero) {
      i20 = (const int8_t*) ((uintptr_t) i20 + input_offset);
    }
    const int8_t* i21 = input[21];
    assert(i21 != NULL);
    if XNN_UNPREDICTABLE(i21 != zero) {
      i21 = (const int8_t*) ((uintptr_t) i21 + input_offset);
    }
    const int8_t* i22 = input[22];
    assert(i22 != NULL);
    if XNN_UNPREDICTABLE(i22 != zero) {
      i22 = (const int8_t*) ((uintptr_t) i22 + input_offset);
    }
    const int8_t* i23 = input[23];
    assert(i23 != NULL);
    if XNN_UNPREDICTABLE(i23 != zero) {
      i23 = (const int8_t*) ((uintptr_t) i23 + input_offset);
    }
    const int8_t* i24 = input[24];
    assert(i24 != NULL);
    if XNN_UNPREDICTABLE(i24 != zero) {
      i24 = (const int8_t*) ((uintptr_t) i24 + input_offset);
    }
    input = (const int8_t**) ((uintptr_t) input + input_stride);

    size_t c = channels;
    const void* w = weights;
    for (; c >= 24; c -= 24) {
      int32x4_t vacc0123 = vld1q_s32(w); w = (const void*) ((const int32_t*) w + 4);
      int32x4_t vacc4567 = vld1q_s32(w); w = (const void*) ((const int32_t*) w + 4);
      int32x4_t vacc89AB = vld1q_s32(w); w = (const void*) ((const int32_t*) w + 4);
      int32x4_t vaccCDEF = vld1q_s32(w); w = (const void*) ((const int32_t*) w + 4);
      int32x4_t vaccGHIJ = vld1q_s32(w); w = (const void*) ((const int32_t*) w + 4);
      int32x4_t vaccKLMN = vld1q_s32(w); w = (const void*) ((const int32_t*) w + 4);


      const int16x8_t vi0x01234567 = vmovl_s8(vld1_s8(i0)); i0 += 8;
      const int16x8_t vk0x01234567 = vmovl_s8(vld1_s8(w)); w = (const void*) ((const int8_t*) w + 8);
      const int16x8_t vi0x89ABCDEF = vmovl_s8(vld1_s8(i0)); i0 += 8;
      const int16x8_t vk0x89ABCDEF = vmovl_s8(vld1_s8(w)); w = (const void*) ((const int8_t*) w + 8);
      const int16x8_t vi0xGHIJKLMN = vmovl_s8(vld1_s8(i0)); i0 += 8;
      const int16x8_t vk0xGHIJKLMN = vmovl_s8(vld1_s8(w)); w = (const void*) ((const int8_t*) w + 8);

      vacc0123 = vmlal_s16(vacc0123, vget_low_s16(vi0x01234567), vget_low_s16(vk0x01234567));
      vacc4567 = vmlal_s16(vacc4567, vget_high_s16(vi0x01234567), vget_high_s16(vk0x01234567));
      vacc89AB = vmlal_s16(vacc89AB, vget_low_s16(vi0x89ABCDEF), vget_low_s16(vk0x89ABCDEF));
      vaccCDEF = vmlal_s16(vaccCDEF, vget_high_s16(vi0x89ABCDEF), vget_high_s16(vk0x89ABCDEF));
      vaccGHIJ = vmlal_s16(vaccGHIJ, vget_low_s16(vi0xGHIJKLMN), vget_low_s16(vk0xGHIJKLMN));
      vaccKLMN = vmlal_s16(vaccKLMN, vget_high_s16(vi0xGHIJKLMN), vget_high_s16(vk0xGHIJKLMN));

      const int16x8_t vi1x01234567 = vmovl_s8(vld1_s8(i1)); i1 += 8;
      const int16x8_t vk1x01234567 = vmovl_s8(vld1_s8(w)); w = (const void*) ((const int8_t*) w + 8);
      const int16x8_t vi1x89ABCDEF = vmovl_s8(vld1_s8(i1)); i1 += 8;
      const int16x8_t vk1x89ABCDEF = vmovl_s8(vld1_s8(w)); w = (const void*) ((const int8_t*) w + 8);
      const int16x8_t vi1xGHIJKLMN = vmovl_s8(vld1_s8(i1)); i1 += 8;
      const int16x8_t vk1xGHIJKLMN = vmovl_s8(vld1_s8(w)); w = (const void*) ((const int8_t*) w + 8);

      vacc0123 = vmlal_s16(vacc0123, vget_low_s16(vi1x01234567), vget_low_s16(vk1x01234567));
      vacc4567 = vmlal_s16(vacc4567, vget_high_s16(vi1x01234567), vget_high_s16(vk1x01234567));
      vacc89AB = vmlal_s16(vacc89AB, vget_low_s16(vi1x89ABCDEF), vget_low_s16(vk1x89ABCDEF));
      vaccCDEF = vmlal_s16(vaccCDEF, vget_high_s16(vi1x89ABCDEF), vget_high_s16(vk1x89ABCDEF));
      vaccGHIJ = vmlal_s16(vaccGHIJ, vget_low_s16(vi1xGHIJKLMN), vget_low_s16(vk1xGHIJKLMN));
      vaccKLMN = vmlal_s16(vaccKLMN, vget_high_s16(vi1xGHIJKLMN), vget_high_s16(vk1xGHIJKLMN));

      const int16x8_t vi2x01234567 = vmovl_s8(vld1_s8(i2)); i2 += 8;
      const int16x8_t vk2x01234567 = vmovl_s8(vld1_s8(w)); w = (const void*) ((const int8_t*) w + 8);
      const int16x8_t vi2x89ABCDEF = vmovl_s8(vld1_s8(i2)); i2 += 8;
      const int16x8_t vk2x89ABCDEF = vmovl_s8(vld1_s8(w)); w = (const void*) ((const int8_t*) w + 8);
      const int16x8_t vi2xGHIJKLMN = vmovl_s8(vld1_s8(i2)); i2 += 8;
      const int16x8_t vk2xGHIJKLMN = vmovl_s8(vld1_s8(w)); w = (const void*) ((const int8_t*) w + 8);

      vacc0123 = vmlal_s16(vacc0123, vget_low_s16(vi2x01234567), vget_low_s16(vk2x01234567));
      vacc4567 = vmlal_s16(vacc4567, vget_high_s16(vi2x01234567), vget_high_s16(vk2x01234567));
      vacc89AB = vmlal_s16(vacc89AB, vget_low_s16(vi2x89ABCDEF), vget_low_s16(vk2x89ABCDEF));
      vaccCDEF = vmlal_s16(vaccCDEF, vget_high_s16(vi2x89ABCDEF), vget_high_s16(vk2x89ABCDEF));
      vaccGHIJ = vmlal_s16(vaccGHIJ, vget_low_s16(vi2xGHIJKLMN), vget_low_s16(vk2xGHIJKLMN));
      vaccKLMN = vmlal_s16(vaccKLMN, vget_high_s16(vi2xGHIJKLMN), vget_high_s16(vk2xGHIJKLMN));

      const int16x8_t vi3x01234567 = vmovl_s8(vld1_s8(i3)); i3 += 8;
      const int16x8_t vk3x01234567 = vmovl_s8(vld1_s8(w)); w = (const void*) ((const int8_t*) w + 8);
      const int16x8_t vi3x89ABCDEF = vmovl_s8(vld1_s8(i3)); i3 += 8;
      const int16x8_t vk3x89ABCDEF = vmovl_s8(vld1_s8(w)); w = (const void*) ((const int8_t*) w + 8);
      const int16x8_t vi3xGHIJKLMN = vmovl_s8(vld1_s8(i3)); i3 += 8;
      const int16x8_t vk3xGHIJKLMN = vmovl_s8(vld1_s8(w)); w = (const void*) ((const int8_t*) w + 8);

      vacc0123 = vmlal_s16(vacc0123, vget_low_s16(vi3x01234567), vget_low_s16(vk3x01234567));
      vacc4567 = vmlal_s16(vacc4567, vget_high_s16(vi3x01234567), vget_high_s16(vk3x01234567));
      vacc89AB = vmlal_s16(vacc89AB, vget_low_s16(vi3x89ABCDEF), vget_low_s16(vk3x89ABCDEF));
      vaccCDEF = vmlal_s16(vaccCDEF, vget_high_s16(vi3x89ABCDEF), vget_high_s16(vk3x89ABCDEF));
      vaccGHIJ = vmlal_s16(vaccGHIJ, vget_low_s16(vi3xGHIJKLMN), vget_low_s16(vk3xGHIJKLMN));
      vaccKLMN = vmlal_s16(vaccKLMN, vget_high_s16(vi3xGHIJKLMN), vget_high_s16(vk3xGHIJKLMN));

      const int16x8_t vi4x01234567 = vmovl_s8(vld1_s8(i4)); i4 += 8;
      const int16x8_t vk4x01234567 = vmovl_s8(vld1_s8(w)); w = (const void*) ((const int8_t*) w + 8);
      const int16x8_t vi4x89ABCDEF = vmovl_s8(vld1_s8(i4)); i4 += 8;
      const int16x8_t vk4x89ABCDEF = vmovl_s8(vld1_s8(w)); w = (const void*) ((const int8_t*) w + 8);
      const int16x8_t vi4xGHIJKLMN = vmovl_s8(vld1_s8(i4)); i4 += 8;
      const int16x8_t vk4xGHIJKLMN = vmovl_s8(vld1_s8(w)); w = (const void*) ((const int8_t*) w + 8);

      vacc0123 = vmlal_s16(vacc0123, vget_low_s16(vi4x01234567), vget_low_s16(vk4x01234567));
      vacc4567 = vmlal_s16(vacc4567, vget_high_s16(vi4x01234567), vget_high_s16(vk4x01234567));
      vacc89AB = vmlal_s16(vacc89AB, vget_low_s16(vi4x89ABCDEF), vget_low_s16(vk4x89ABCDEF));
      vaccCDEF = vmlal_s16(vaccCDEF, vget_high_s16(vi4x89ABCDEF), vget_high_s16(vk4x89ABCDEF));
      vaccGHIJ = vmlal_s16(vaccGHIJ, vget_low_s16(vi4xGHIJKLMN), vget_low_s16(vk4xGHIJKLMN));
      vaccKLMN = vmlal_s16(vaccKLMN, vget_high_s16(vi4xGHIJKLMN), vget_high_s16(vk4xGHIJKLMN));

      const int16x8_t vi5x01234567 = vmovl_s8(vld1_s8(i5)); i5 += 8;
      const int16x8_t vk5x01234567 = vmovl_s8(vld1_s8(w)); w = (const void*) ((const int8_t*) w + 8);
      const int16x8_t vi5x89ABCDEF = vmovl_s8(vld1_s8(i5)); i5 += 8;
      const int16x8_t vk5x89ABCDEF = vmovl_s8(vld1_s8(w)); w = (const void*) ((const int8_t*) w + 8);
      const int16x8_t vi5xGHIJKLMN = vmovl_s8(vld1_s8(i5)); i5 += 8;
      const int16x8_t vk5xGHIJKLMN = vmovl_s8(vld1_s8(w)); w = (const void*) ((const int8_t*) w + 8);

      vacc0123 = vmlal_s16(vacc0123, vget_low_s16(vi5x01234567), vget_low_s16(vk5x01234567));
      vacc4567 = vmlal_s16(vacc4567, vget_high_s16(vi5x01234567), vget_high_s16(vk5x01234567));
      vacc89AB = vmlal_s16(vacc89AB, vget_low_s16(vi5x89ABCDEF), vget_low_s16(vk5x89ABCDEF));
      vaccCDEF = vmlal_s16(vaccCDEF, vget_high_s16(vi5x89ABCDEF), vget_high_s16(vk5x89ABCDEF));
      vaccGHIJ = vmlal_s16(vaccGHIJ, vget_low_s16(vi5xGHIJKLMN), vget_low_s16(vk5xGHIJKLMN));
      vaccKLMN = vmlal_s16(vaccKLMN, vget_high_s16(vi5xGHIJKLMN), vget_high_s16(vk5xGHIJKLMN));

      const int16x8_t vi6x01234567 = vmovl_s8(vld1_s8(i6)); i6 += 8;
      const int16x8_t vk6x01234567 = vmovl_s8(vld1_s8(w)); w = (const void*) ((const int8_t*) w + 8);
      const int16x8_t vi6x89ABCDEF = vmovl_s8(vld1_s8(i6)); i6 += 8;
      const int16x8_t vk6x89ABCDEF = vmovl_s8(vld1_s8(w)); w = (const void*) ((const int8_t*) w + 8);
      const int16x8_t vi6xGHIJKLMN = vmovl_s8(vld1_s8(i6)); i6 += 8;
      const int16x8_t vk6xGHIJKLMN = vmovl_s8(vld1_s8(w)); w = (const void*) ((const int8_t*) w + 8);

      vacc0123 = vmlal_s16(vacc0123, vget_low_s16(vi6x01234567), vget_low_s16(vk6x01234567));
      vacc4567 = vmlal_s16(vacc4567, vget_high_s16(vi6x01234567), vget_high_s16(vk6x01234567));
      vacc89AB = vmlal_s16(vacc89AB, vget_low_s16(vi6x89ABCDEF), vget_low_s16(vk6x89ABCDEF));
      vaccCDEF = vmlal_s16(vaccCDEF, vget_high_s16(vi6x89ABCDEF), vget_high_s16(vk6x89ABCDEF));
      vaccGHIJ = vmlal_s16(vaccGHIJ, vget_low_s16(vi6xGHIJKLMN), vget_low_s16(vk6xGHIJKLMN));
      vaccKLMN = vmlal_s16(vaccKLMN, vget_high_s16(vi6xGHIJKLMN), vget_high_s16(vk6xGHIJKLMN));

      const int16x8_t vi7x01234567 = vmovl_s8(vld1_s8(i7)); i7 += 8;
      const int16x8_t vk7x01234567 = vmovl_s8(vld1_s8(w)); w = (const void*) ((const int8_t*) w + 8);
      const int16x8_t vi7x89ABCDEF = vmovl_s8(vld1_s8(i7)); i7 += 8;
      const int16x8_t vk7x89ABCDEF = vmovl_s8(vld1_s8(w)); w = (const void*) ((const int8_t*) w + 8);
      const int16x8_t vi7xGHIJKLMN = vmovl_s8(vld1_s8(i7)); i7 += 8;
      const int16x8_t vk7xGHIJKLMN = vmovl_s8(vld1_s8(w)); w = (const void*) ((const int8_t*) w + 8);

      vacc0123 = vmlal_s16(vacc0123, vget_low_s16(vi7x01234567), vget_low_s16(vk7x01234567));
      vacc4567 = vmlal_s16(vacc4567, vget_high_s16(vi7x01234567), vget_high_s16(vk7x01234567));
      vacc89AB = vmlal_s16(vacc89AB, vget_low_s16(vi7x89ABCDEF), vget_low_s16(vk7x89ABCDEF));
      vaccCDEF = vmlal_s16(vaccCDEF, vget_high_s16(vi7x89ABCDEF), vget_high_s16(vk7x89ABCDEF));
      vaccGHIJ = vmlal_s16(vaccGHIJ, vget_low_s16(vi7xGHIJKLMN), vget_low_s16(vk7xGHIJKLMN));
      vaccKLMN = vmlal_s16(vaccKLMN, vget_high_s16(vi7xGHIJKLMN), vget_high_s16(vk7xGHIJKLMN));

      const int16x8_t vi8x01234567 = vmovl_s8(vld1_s8(i8)); i8 += 8;
      const int16x8_t vk8x01234567 = vmovl_s8(vld1_s8(w)); w = (const void*) ((const int8_t*) w + 8);
      const int16x8_t vi8x89ABCDEF = vmovl_s8(vld1_s8(i8)); i8 += 8;
      const int16x8_t vk8x89ABCDEF = vmovl_s8(vld1_s8(w)); w = (const void*) ((const int8_t*) w + 8);
      const int16x8_t vi8xGHIJKLMN = vmovl_s8(vld1_s8(i8)); i8 += 8;
      const int16x8_t vk8xGHIJKLMN = vmovl_s8(vld1_s8(w)); w = (const void*) ((const int8_t*) w + 8);

      vacc0123 = vmlal_s16(vacc0123, vget_low_s16(vi8x01234567), vget_low_s16(vk8x01234567));
      vacc4567 = vmlal_s16(vacc4567, vget_high_s16(vi8x01234567), vget_high_s16(vk8x01234567));
      vacc89AB = vmlal_s16(vacc89AB, vget_low_s16(vi8x89ABCDEF), vget_low_s16(vk8x89ABCDEF));
      vaccCDEF = vmlal_s16(vaccCDEF, vget_high_s16(vi8x89ABCDEF), vget_high_s16(vk8x89ABCDEF));
      vaccGHIJ = vmlal_s16(vaccGHIJ, vget_low_s16(vi8xGHIJKLMN), vget_low_s16(vk8xGHIJKLMN));
      vaccKLMN = vmlal_s16(vaccKLMN, vget_high_s16(vi8xGHIJKLMN), vget_high_s16(vk8xGHIJKLMN));

      const int16x8_t vi9x01234567 = vmovl_s8(vld1_s8(i9)); i9 += 8;
      const int16x8_t vk9x01234567 = vmovl_s8(vld1_s8(w)); w = (const void*) ((const int8_t*) w + 8);
      const int16x8_t vi9x89ABCDEF = vmovl_s8(vld1_s8(i9)); i9 += 8;
      const int16x8_t vk9x89ABCDEF = vmovl_s8(vld1_s8(w)); w = (const void*) ((const int8_t*) w + 8);
      const int16x8_t vi9xGHIJKLMN = vmovl_s8(vld1_s8(i9)); i9 += 8;
      const int16x8_t vk9xGHIJKLMN = vmovl_s8(vld1_s8(w)); w = (const void*) ((const int8_t*) w + 8);

      vacc0123 = vmlal_s16(vacc0123, vget_low_s16(vi9x01234567), vget_low_s16(vk9x01234567));
      vacc4567 = vmlal_s16(vacc4567, vget_high_s16(vi9x01234567), vget_high_s16(vk9x01234567));
      vacc89AB = vmlal_s16(vacc89AB, vget_low_s16(vi9x89ABCDEF), vget_low_s16(vk9x89ABCDEF));
      vaccCDEF = vmlal_s16(vaccCDEF, vget_high_s16(vi9x89ABCDEF), vget_high_s16(vk9x89ABCDEF));
      vaccGHIJ = vmlal_s16(vaccGHIJ, vget_low_s16(vi9xGHIJKLMN), vget_low_s16(vk9xGHIJKLMN));
      vaccKLMN = vmlal_s16(vaccKLMN, vget_high_s16(vi9xGHIJKLMN), vget_high_s16(vk9xGHIJKLMN));

      const int16x8_t vi10x01234567 = vmovl_s8(vld1_s8(i10)); i10 += 8;
      const int16x8_t vk10x01234567 = vmovl_s8(vld1_s8(w)); w = (const void*) ((const int8_t*) w + 8);
      const int16x8_t vi10x89ABCDEF = vmovl_s8(vld1_s8(i10)); i10 += 8;
      const int16x8_t vk10x89ABCDEF = vmovl_s8(vld1_s8(w)); w = (const void*) ((const int8_t*) w + 8);
      const int16x8_t vi10xGHIJKLMN = vmovl_s8(vld1_s8(i10)); i10 += 8;
      const int16x8_t vk10xGHIJKLMN = vmovl_s8(vld1_s8(w)); w = (const void*) ((const int8_t*) w + 8);

      vacc0123 = vmlal_s16(vacc0123, vget_low_s16(vi10x01234567), vget_low_s16(vk10x01234567));
      vacc4567 = vmlal_s16(vacc4567, vget_high_s16(vi10x01234567), vget_high_s16(vk10x01234567));
      vacc89AB = vmlal_s16(vacc89AB, vget_low_s16(vi10x89ABCDEF), vget_low_s16(vk10x89ABCDEF));
      vaccCDEF = vmlal_s16(vaccCDEF, vget_high_s16(vi10x89ABCDEF), vget_high_s16(vk10x89ABCDEF));
      vaccGHIJ = vmlal_s16(vaccGHIJ, vget_low_s16(vi10xGHIJKLMN), vget_low_s16(vk10xGHIJKLMN));
      vaccKLMN = vmlal_s16(vaccKLMN, vget_high_s16(vi10xGHIJKLMN), vget_high_s16(vk10xGHIJKLMN));

      const int16x8_t vi11x01234567 = vmovl_s8(vld1_s8(i11)); i11 += 8;
      const int16x8_t vk11x01234567 = vmovl_s8(vld1_s8(w)); w = (const void*) ((const int8_t*) w + 8);
      const int16x8_t vi11x89ABCDEF = vmovl_s8(vld1_s8(i11)); i11 += 8;
      const int16x8_t vk11x89ABCDEF = vmovl_s8(vld1_s8(w)); w = (const void*) ((const int8_t*) w + 8);
      const int16x8_t vi11xGHIJKLMN = vmovl_s8(vld1_s8(i11)); i11 += 8;
      const int16x8_t vk11xGHIJKLMN = vmovl_s8(vld1_s8(w)); w = (const void*) ((const int8_t*) w + 8);

      vacc0123 = vmlal_s16(vacc0123, vget_low_s16(vi11x01234567), vget_low_s16(vk11x01234567));
      vacc4567 = vmlal_s16(vacc4567, vget_high_s16(vi11x01234567), vget_high_s16(vk11x01234567));
      vacc89AB = vmlal_s16(vacc89AB, vget_low_s16(vi11x89ABCDEF), vget_low_s16(vk11x89ABCDEF));
      vaccCDEF = vmlal_s16(vaccCDEF, vget_high_s16(vi11x89ABCDEF), vget_high_s16(vk11x89ABCDEF));
      vaccGHIJ = vmlal_s16(vaccGHIJ, vget_low_s16(vi11xGHIJKLMN), vget_low_s16(vk11xGHIJKLMN));
      vaccKLMN = vmlal_s16(vaccKLMN, vget_high_s16(vi11xGHIJKLMN), vget_high_s16(vk11xGHIJKLMN));

      const int16x8_t vi12x01234567 = vmovl_s8(vld1_s8(i12)); i12 += 8;
      const int16x8_t vk12x01234567 = vmovl_s8(vld1_s8(w)); w = (const void*) ((const int8_t*) w + 8);
      const int16x8_t vi12x89ABCDEF = vmovl_s8(vld1_s8(i12)); i12 += 8;
      const int16x8_t vk12x89ABCDEF = vmovl_s8(vld1_s8(w)); w = (const void*) ((const int8_t*) w + 8);
      const int16x8_t vi12xGHIJKLMN = vmovl_s8(vld1_s8(i12)); i12 += 8;
      const int16x8_t vk12xGHIJKLMN = vmovl_s8(vld1_s8(w)); w = (const void*) ((const int8_t*) w + 8);

      vacc0123 = vmlal_s16(vacc0123, vget_low_s16(vi12x01234567), vget_low_s16(vk12x01234567));
      vacc4567 = vmlal_s16(vacc4567, vget_high_s16(vi12x01234567), vget_high_s16(vk12x01234567));
      vacc89AB = vmlal_s16(vacc89AB, vget_low_s16(vi12x89ABCDEF), vget_low_s16(vk12x89ABCDEF));
      vaccCDEF = vmlal_s16(vaccCDEF, vget_high_s16(vi12x89ABCDEF), vget_high_s16(vk12x89ABCDEF));
      vaccGHIJ = vmlal_s16(vaccGHIJ, vget_low_s16(vi12xGHIJKLMN), vget_low_s16(vk12xGHIJKLMN));
      vaccKLMN = vmlal_s16(vaccKLMN, vget_high_s16(vi12xGHIJKLMN), vget_high_s16(vk12xGHIJKLMN));

      const int16x8_t vi13x01234567 = vmovl_s8(vld1_s8(i13)); i13 += 8;
      const int16x8_t vk13x01234567 = vmovl_s8(vld1_s8(w)); w = (const void*) ((const int8_t*) w + 8);
      const int16x8_t vi13x89ABCDEF = vmovl_s8(vld1_s8(i13)); i13 += 8;
      const int16x8_t vk13x89ABCDEF = vmovl_s8(vld1_s8(w)); w = (const void*) ((const int8_t*) w + 8);
      const int16x8_t vi13xGHIJKLMN = vmovl_s8(vld1_s8(i13)); i13 += 8;
      const int16x8_t vk13xGHIJKLMN = vmovl_s8(vld1_s8(w)); w = (const void*) ((const int8_t*) w + 8);

      vacc0123 = vmlal_s16(vacc0123, vget_low_s16(vi13x01234567), vget_low_s16(vk13x01234567));
      vacc4567 = vmlal_s16(vacc4567, vget_high_s16(vi13x01234567), vget_high_s16(vk13x01234567));
      vacc89AB = vmlal_s16(vacc89AB, vget_low_s16(vi13x89ABCDEF), vget_low_s16(vk13x89ABCDEF));
      vaccCDEF = vmlal_s16(vaccCDEF, vget_high_s16(vi13x89ABCDEF), vget_high_s16(vk13x89ABCDEF));
      vaccGHIJ = vmlal_s16(vaccGHIJ, vget_low_s16(vi13xGHIJKLMN), vget_low_s16(vk13xGHIJKLMN));
      vaccKLMN = vmlal_s16(vaccKLMN, vget_high_s16(vi13xGHIJKLMN), vget_high_s16(vk13xGHIJKLMN));

      const int16x8_t vi14x01234567 = vmovl_s8(vld1_s8(i14)); i14 += 8;
      const int16x8_t vk14x01234567 = vmovl_s8(vld1_s8(w)); w = (const void*) ((const int8_t*) w + 8);
      const int16x8_t vi14x89ABCDEF = vmovl_s8(vld1_s8(i14)); i14 += 8;
      const int16x8_t vk14x89ABCDEF = vmovl_s8(vld1_s8(w)); w = (const void*) ((const int8_t*) w + 8);
      const int16x8_t vi14xGHIJKLMN = vmovl_s8(vld1_s8(i14)); i14 += 8;
      const int16x8_t vk14xGHIJKLMN = vmovl_s8(vld1_s8(w)); w = (const void*) ((const int8_t*) w + 8);

      vacc0123 = vmlal_s16(vacc0123, vget_low_s16(vi14x01234567), vget_low_s16(vk14x01234567));
      vacc4567 = vmlal_s16(vacc4567, vget_high_s16(vi14x01234567), vget_high_s16(vk14x01234567));
      vacc89AB = vmlal_s16(vacc89AB, vget_low_s16(vi14x89ABCDEF), vget_low_s16(vk14x89ABCDEF));
      vaccCDEF = vmlal_s16(vaccCDEF, vget_high_s16(vi14x89ABCDEF), vget_high_s16(vk14x89ABCDEF));
      vaccGHIJ = vmlal_s16(vaccGHIJ, vget_low_s16(vi14xGHIJKLMN), vget_low_s16(vk14xGHIJKLMN));
      vaccKLMN = vmlal_s16(vaccKLMN, vget_high_s16(vi14xGHIJKLMN), vget_high_s16(vk14xGHIJKLMN));

      const int16x8_t vi15x01234567 = vmovl_s8(vld1_s8(i15)); i15 += 8;
      const int16x8_t vk15x01234567 = vmovl_s8(vld1_s8(w)); w = (const void*) ((const int8_t*) w + 8);
      const int16x8_t vi15x89ABCDEF = vmovl_s8(vld1_s8(i15)); i15 += 8;
      const int16x8_t vk15x89ABCDEF = vmovl_s8(vld1_s8(w)); w = (const void*) ((const int8_t*) w + 8);
      const int16x8_t vi15xGHIJKLMN = vmovl_s8(vld1_s8(i15)); i15 += 8;
      const int16x8_t vk15xGHIJKLMN = vmovl_s8(vld1_s8(w)); w = (const void*) ((const int8_t*) w + 8);

      vacc0123 = vmlal_s16(vacc0123, vget_low_s16(vi15x01234567), vget_low_s16(vk15x01234567));
      vacc4567 = vmlal_s16(vacc4567, vget_high_s16(vi15x01234567), vget_high_s16(vk15x01234567));
      vacc89AB = vmlal_s16(vacc89AB, vget_low_s16(vi15x89ABCDEF), vget_low_s16(vk15x89ABCDEF));
      vaccCDEF = vmlal_s16(vaccCDEF, vget_high_s16(vi15x89ABCDEF), vget_high_s16(vk15x89ABCDEF));
      vaccGHIJ = vmlal_s16(vaccGHIJ, vget_low_s16(vi15xGHIJKLMN), vget_low_s16(vk15xGHIJKLMN));
      vaccKLMN = vmlal_s16(vaccKLMN, vget_high_s16(vi15xGHIJKLMN), vget_high_s16(vk15xGHIJKLMN));

      const int16x8_t vi16x01234567 = vmovl_s8(vld1_s8(i16)); i16 += 8;
      const int16x8_t vk16x01234567 = vmovl_s8(vld1_s8(w)); w = (const void*) ((const int8_t*) w + 8);
      const int16x8_t vi16x89ABCDEF = vmovl_s8(vld1_s8(i16)); i16 += 8;
      const int16x8_t vk16x89ABCDEF = vmovl_s8(vld1_s8(w)); w = (const void*) ((const int8_t*) w + 8);
      const int16x8_t vi16xGHIJKLMN = vmovl_s8(vld1_s8(i16)); i16 += 8;
      const int16x8_t vk16xGHIJKLMN = vmovl_s8(vld1_s8(w)); w = (const void*) ((const int8_t*) w + 8);

      vacc0123 = vmlal_s16(vacc0123, vget_low_s16(vi16x01234567), vget_low_s16(vk16x01234567));
      vacc4567 = vmlal_s16(vacc4567, vget_high_s16(vi16x01234567), vget_high_s16(vk16x01234567));
      vacc89AB = vmlal_s16(vacc89AB, vget_low_s16(vi16x89ABCDEF), vget_low_s16(vk16x89ABCDEF));
      vaccCDEF = vmlal_s16(vaccCDEF, vget_high_s16(vi16x89ABCDEF), vget_high_s16(vk16x89ABCDEF));
      vaccGHIJ = vmlal_s16(vaccGHIJ, vget_low_s16(vi16xGHIJKLMN), vget_low_s16(vk16xGHIJKLMN));
      vaccKLMN = vmlal_s16(vaccKLMN, vget_high_s16(vi16xGHIJKLMN), vget_high_s16(vk16xGHIJKLMN));

      const int16x8_t vi17x01234567 = vmovl_s8(vld1_s8(i17)); i17 += 8;
      const int16x8_t vk17x01234567 = vmovl_s8(vld1_s8(w)); w = (const void*) ((const int8_t*) w + 8);
      const int16x8_t vi17x89ABCDEF = vmovl_s8(vld1_s8(i17)); i17 += 8;
      const int16x8_t vk17x89ABCDEF = vmovl_s8(vld1_s8(w)); w = (const void*) ((const int8_t*) w + 8);
      const int16x8_t vi17xGHIJKLMN = vmovl_s8(vld1_s8(i17)); i17 += 8;
      const int16x8_t vk17xGHIJKLMN = vmovl_s8(vld1_s8(w)); w = (const void*) ((const int8_t*) w + 8);

      vacc0123 = vmlal_s16(vacc0123, vget_low_s16(vi17x01234567), vget_low_s16(vk17x01234567));
      vacc4567 = vmlal_s16(vacc4567, vget_high_s16(vi17x01234567), vget_high_s16(vk17x01234567));
      vacc89AB = vmlal_s16(vacc89AB, vget_low_s16(vi17x89ABCDEF), vget_low_s16(vk17x89ABCDEF));
      vaccCDEF = vmlal_s16(vaccCDEF, vget_high_s16(vi17x89ABCDEF), vget_high_s16(vk17x89ABCDEF));
      vaccGHIJ = vmlal_s16(vaccGHIJ, vget_low_s16(vi17xGHIJKLMN), vget_low_s16(vk17xGHIJKLMN));
      vaccKLMN = vmlal_s16(vaccKLMN, vget_high_s16(vi17xGHIJKLMN), vget_high_s16(vk17xGHIJKLMN));

      const int16x8_t vi18x01234567 = vmovl_s8(vld1_s8(i18)); i18 += 8;
      const int16x8_t vk18x01234567 = vmovl_s8(vld1_s8(w)); w = (const void*) ((const int8_t*) w + 8);
      const int16x8_t vi18x89ABCDEF = vmovl_s8(vld1_s8(i18)); i18 += 8;
      const int16x8_t vk18x89ABCDEF = vmovl_s8(vld1_s8(w)); w = (const void*) ((const int8_t*) w + 8);
      const int16x8_t vi18xGHIJKLMN = vmovl_s8(vld1_s8(i18)); i18 += 8;
      const int16x8_t vk18xGHIJKLMN = vmovl_s8(vld1_s8(w)); w = (const void*) ((const int8_t*) w + 8);

      vacc0123 = vmlal_s16(vacc0123, vget_low_s16(vi18x01234567), vget_low_s16(vk18x01234567));
      vacc4567 = vmlal_s16(vacc4567, vget_high_s16(vi18x01234567), vget_high_s16(vk18x01234567));
      vacc89AB = vmlal_s16(vacc89AB, vget_low_s16(vi18x89ABCDEF), vget_low_s16(vk18x89ABCDEF));
      vaccCDEF = vmlal_s16(vaccCDEF, vget_high_s16(vi18x89ABCDEF), vget_high_s16(vk18x89ABCDEF));
      vaccGHIJ = vmlal_s16(vaccGHIJ, vget_low_s16(vi18xGHIJKLMN), vget_low_s16(vk18xGHIJKLMN));
      vaccKLMN = vmlal_s16(vaccKLMN, vget_high_s16(vi18xGHIJKLMN), vget_high_s16(vk18xGHIJKLMN));

      const int16x8_t vi19x01234567 = vmovl_s8(vld1_s8(i19)); i19 += 8;
      const int16x8_t vk19x01234567 = vmovl_s8(vld1_s8(w)); w = (const void*) ((const int8_t*) w + 8);
      const int16x8_t vi19x89ABCDEF = vmovl_s8(vld1_s8(i19)); i19 += 8;
      const int16x8_t vk19x89ABCDEF = vmovl_s8(vld1_s8(w)); w = (const void*) ((const int8_t*) w + 8);
      const int16x8_t vi19xGHIJKLMN = vmovl_s8(vld1_s8(i19)); i19 += 8;
      const int16x8_t vk19xGHIJKLMN = vmovl_s8(vld1_s8(w)); w = (const void*) ((const int8_t*) w + 8);

      vacc0123 = vmlal_s16(vacc0123, vget_low_s16(vi19x01234567), vget_low_s16(vk19x01234567));
      vacc4567 = vmlal_s16(vacc4567, vget_high_s16(vi19x01234567), vget_high_s16(vk19x01234567));
      vacc89AB = vmlal_s16(vacc89AB, vget_low_s16(vi19x89ABCDEF), vget_low_s16(vk19x89ABCDEF));
      vaccCDEF = vmlal_s16(vaccCDEF, vget_high_s16(vi19x89ABCDEF), vget_high_s16(vk19x89ABCDEF));
      vaccGHIJ = vmlal_s16(vaccGHIJ, vget_low_s16(vi19xGHIJKLMN), vget_low_s16(vk19xGHIJKLMN));
      vaccKLMN = vmlal_s16(vaccKLMN, vget_high_s16(vi19xGHIJKLMN), vget_high_s16(vk19xGHIJKLMN));

      const int16x8_t vi20x01234567 = vmovl_s8(vld1_s8(i20)); i20 += 8;
      const int16x8_t vk20x01234567 = vmovl_s8(vld1_s8(w)); w = (const void*) ((const int8_t*) w + 8);
      const int16x8_t vi20x89ABCDEF = vmovl_s8(vld1_s8(i20)); i20 += 8;
      const int16x8_t vk20x89ABCDEF = vmovl_s8(vld1_s8(w)); w = (const void*) ((const int8_t*) w + 8);
      const int16x8_t vi20xGHIJKLMN = vmovl_s8(vld1_s8(i20)); i20 += 8;
      const int16x8_t vk20xGHIJKLMN = vmovl_s8(vld1_s8(w)); w = (const void*) ((const int8_t*) w + 8);

      vacc0123 = vmlal_s16(vacc0123, vget_low_s16(vi20x01234567), vget_low_s16(vk20x01234567));
      vacc4567 = vmlal_s16(vacc4567, vget_high_s16(vi20x01234567), vget_high_s16(vk20x01234567));
      vacc89AB = vmlal_s16(vacc89AB, vget_low_s16(vi20x89ABCDEF), vget_low_s16(vk20x89ABCDEF));
      vaccCDEF = vmlal_s16(vaccCDEF, vget_high_s16(vi20x89ABCDEF), vget_high_s16(vk20x89ABCDEF));
      vaccGHIJ = vmlal_s16(vaccGHIJ, vget_low_s16(vi20xGHIJKLMN), vget_low_s16(vk20xGHIJKLMN));
      vaccKLMN = vmlal_s16(vaccKLMN, vget_high_s16(vi20xGHIJKLMN), vget_high_s16(vk20xGHIJKLMN));

      const int16x8_t vi21x01234567 = vmovl_s8(vld1_s8(i21)); i21 += 8;
      const int16x8_t vk21x01234567 = vmovl_s8(vld1_s8(w)); w = (const void*) ((const int8_t*) w + 8);
      const int16x8_t vi21x89ABCDEF = vmovl_s8(vld1_s8(i21)); i21 += 8;
      const int16x8_t vk21x89ABCDEF = vmovl_s8(vld1_s8(w)); w = (const void*) ((const int8_t*) w + 8);
      const int16x8_t vi21xGHIJKLMN = vmovl_s8(vld1_s8(i21)); i21 += 8;
      const int16x8_t vk21xGHIJKLMN = vmovl_s8(vld1_s8(w)); w = (const void*) ((const int8_t*) w + 8);

      vacc0123 = vmlal_s16(vacc0123, vget_low_s16(vi21x01234567), vget_low_s16(vk21x01234567));
      vacc4567 = vmlal_s16(vacc4567, vget_high_s16(vi21x01234567), vget_high_s16(vk21x01234567));
      vacc89AB = vmlal_s16(vacc89AB, vget_low_s16(vi21x89ABCDEF), vget_low_s16(vk21x89ABCDEF));
      vaccCDEF = vmlal_s16(vaccCDEF, vget_high_s16(vi21x89ABCDEF), vget_high_s16(vk21x89ABCDEF));
      vaccGHIJ = vmlal_s16(vaccGHIJ, vget_low_s16(vi21xGHIJKLMN), vget_low_s16(vk21xGHIJKLMN));
      vaccKLMN = vmlal_s16(vaccKLMN, vget_high_s16(vi21xGHIJKLMN), vget_high_s16(vk21xGHIJKLMN));

      const int16x8_t vi22x01234567 = vmovl_s8(vld1_s8(i22)); i22 += 8;
      const int16x8_t vk22x01234567 = vmovl_s8(vld1_s8(w)); w = (const void*) ((const int8_t*) w + 8);
      const int16x8_t vi22x89ABCDEF = vmovl_s8(vld1_s8(i22)); i22 += 8;
      const int16x8_t vk22x89ABCDEF = vmovl_s8(vld1_s8(w)); w = (const void*) ((const int8_t*) w + 8);
      const int16x8_t vi22xGHIJKLMN = vmovl_s8(vld1_s8(i22)); i22 += 8;
      const int16x8_t vk22xGHIJKLMN = vmovl_s8(vld1_s8(w)); w = (const void*) ((const int8_t*) w + 8);

      vacc0123 = vmlal_s16(vacc0123, vget_low_s16(vi22x01234567), vget_low_s16(vk22x01234567));
      vacc4567 = vmlal_s16(vacc4567, vget_high_s16(vi22x01234567), vget_high_s16(vk22x01234567));
      vacc89AB = vmlal_s16(vacc89AB, vget_low_s16(vi22x89ABCDEF), vget_low_s16(vk22x89ABCDEF));
      vaccCDEF = vmlal_s16(vaccCDEF, vget_high_s16(vi22x89ABCDEF), vget_high_s16(vk22x89ABCDEF));
      vaccGHIJ = vmlal_s16(vaccGHIJ, vget_low_s16(vi22xGHIJKLMN), vget_low_s16(vk22xGHIJKLMN));
      vaccKLMN = vmlal_s16(vaccKLMN, vget_high_s16(vi22xGHIJKLMN), vget_high_s16(vk22xGHIJKLMN));

      const int16x8_t vi23x01234567 = vmovl_s8(vld1_s8(i23)); i23 += 8;
      const int16x8_t vk23x01234567 = vmovl_s8(vld1_s8(w)); w = (const void*) ((const int8_t*) w + 8);
      const int16x8_t vi23x89ABCDEF = vmovl_s8(vld1_s8(i23)); i23 += 8;
      const int16x8_t vk23x89ABCDEF = vmovl_s8(vld1_s8(w)); w = (const void*) ((const int8_t*) w + 8);
      const int16x8_t vi23xGHIJKLMN = vmovl_s8(vld1_s8(i23)); i23 += 8;
      const int16x8_t vk23xGHIJKLMN = vmovl_s8(vld1_s8(w)); w = (const void*) ((const int8_t*) w + 8);

      vacc0123 = vmlal_s16(vacc0123, vget_low_s16(vi23x01234567), vget_low_s16(vk23x01234567));
      vacc4567 = vmlal_s16(vacc4567, vget_high_s16(vi23x01234567), vget_high_s16(vk23x01234567));
      vacc89AB = vmlal_s16(vacc89AB, vget_low_s16(vi23x89ABCDEF), vget_low_s16(vk23x89ABCDEF));
      vaccCDEF = vmlal_s16(vaccCDEF, vget_high_s16(vi23x89ABCDEF), vget_high_s16(vk23x89ABCDEF));
      vaccGHIJ = vmlal_s16(vaccGHIJ, vget_low_s16(vi23xGHIJKLMN), vget_low_s16(vk23xGHIJKLMN));
      vaccKLMN = vmlal_s16(vaccKLMN, vget_high_s16(vi23xGHIJKLMN), vget_high_s16(vk23xGHIJKLMN));

      const int16x8_t vi24x01234567 = vmovl_s8(vld1_s8(i24)); i24 += 8;
      const int16x8_t vk24x01234567 = vmovl_s8(vld1_s8(w)); w = (const void*) ((const int8_t*) w + 8);
      const int16x8_t vi24x89ABCDEF = vmovl_s8(vld1_s8(i24)); i24 += 8;
      const int16x8_t vk24x89ABCDEF = vmovl_s8(vld1_s8(w)); w = (const void*) ((const int8_t*) w + 8);
      const int16x8_t vi24xGHIJKLMN = vmovl_s8(vld1_s8(i24)); i24 += 8;
      const int16x8_t vk24xGHIJKLMN = vmovl_s8(vld1_s8(w)); w = (const void*) ((const int8_t*) w + 8);

      vacc0123 = vmlal_s16(vacc0123, vget_low_s16(vi24x01234567), vget_low_s16(vk24x01234567));
      vacc4567 = vmlal_s16(vacc4567, vget_high_s16(vi24x01234567), vget_high_s16(vk24x01234567));
      vacc89AB = vmlal_s16(vacc89AB, vget_low_s16(vi24x89ABCDEF), vget_low_s16(vk24x89ABCDEF));
      vaccCDEF = vmlal_s16(vaccCDEF, vget_high_s16(vi24x89ABCDEF), vget_high_s16(vk24x89ABCDEF));
      vaccGHIJ = vmlal_s16(vaccGHIJ, vget_low_s16(vi24xGHIJKLMN), vget_low_s16(vk24xGHIJKLMN));
      vaccKLMN = vmlal_s16(vaccKLMN, vget_high_s16(vi24xGHIJKLMN), vget_high_s16(vk24xGHIJKLMN));

      float32x4_t vfpacc0123 = vcvtq_f32_s32(vacc0123);
      float32x4_t vfpacc4567 = vcvtq_f32_s32(vacc4567);
      float32x4_t vfpacc89AB = vcvtq_f32_s32(vacc89AB);
      float32x4_t vfpaccCDEF = vcvtq_f32_s32(vaccCDEF);
      float32x4_t vfpaccGHIJ = vcvtq_f32_s32(vaccGHIJ);
      float32x4_t vfpaccKLMN = vcvtq_f32_s32(vaccKLMN);

      const float32x4_t vscale0123 = vld1q_f32((const float*) w); w = (const void*) ((const float*) w + 4);
      const float32x4_t vscale4567 = vld1q_f32((const float*) w); w = (const void*) ((const float*) w + 4);
      const float32x4_t vscale89AB = vld1q_f32((const float*) w); w = (const void*) ((const float*) w + 4);
      const float32x4_t vscaleCDEF = vld1q_f32((const float*) w); w = (const void*) ((const float*) w + 4);
      const float32x4_t vscaleGHIJ = vld1q_f32((const float*) w); w = (const void*) ((const float*) w + 4);
      const float32x4_t vscaleKLMN = vld1q_f32((const float*) w); w = (const void*) ((const float*) w + 4);

      vfpacc0123 = vmulq_f32(vfpacc0123, vscale0123);
      vfpacc4567 = vmulq_f32(vfpacc4567, vscale4567);
      vfpacc89AB = vmulq_f32(vfpacc89AB, vscale89AB);
      vfpaccCDEF = vmulq_f32(vfpaccCDEF, vscaleCDEF);
      vfpaccGHIJ = vmulq_f32(vfpaccGHIJ, vscaleGHIJ);
      vfpaccKLMN = vmulq_f32(vfpaccKLMN, vscaleKLMN);

      vfpacc0123 = vmaxq_f32(vfpacc0123, voutput_min_less_zero_point);
      vfpacc4567 = vmaxq_f32(vfpacc4567, voutput_min_less_zero_point);
      vfpacc89AB = vmaxq_f32(vfpacc89AB, voutput_min_less_zero_point);
      vfpaccCDEF = vmaxq_f32(vfpaccCDEF, voutput_min_less_zero_point);
      vfpaccGHIJ = vmaxq_f32(vfpaccGHIJ, voutput_min_less_zero_point);
      vfpaccKLMN = vmaxq_f32(vfpaccKLMN, voutput_min_less_zero_point);

      vfpacc0123 = vminq_f32(vfpacc0123, voutput_max_less_zero_point);
      vfpacc4567 = vminq_f32(vfpacc4567, voutput_max_less_zero_point);
      vfpacc89AB = vminq_f32(vfpacc89AB, voutput_max_less_zero_point);
      vfpaccCDEF = vminq_f32(vfpaccCDEF, voutput_max_less_zero_point);
      vfpaccGHIJ = vminq_f32(vfpaccGHIJ, voutput_max_less_zero_point);
      vfpaccKLMN = vminq_f32(vfpaccKLMN, voutput_max_less_zero_point);

      vacc0123 = vreinterpretq_s32_f32(vaddq_f32(vfpacc0123, vmagic_bias));
      vacc4567 = vreinterpretq_s32_f32(vaddq_f32(vfpacc4567, vmagic_bias));
      vacc89AB = vreinterpretq_s32_f32(vaddq_f32(vfpacc89AB, vmagic_bias));
      vaccCDEF = vreinterpretq_s32_f32(vaddq_f32(vfpaccCDEF, vmagic_bias));
      vaccGHIJ = vreinterpretq_s32_f32(vaddq_f32(vfpaccGHIJ, vmagic_bias));
      vaccKLMN = vreinterpretq_s32_f32(vaddq_f32(vfpaccKLMN, vmagic_bias));

      vacc0123 = vsubq_s32(vacc0123, vmagic_bias_less_zero_point);
      vacc4567 = vsubq_s32(vacc4567, vmagic_bias_less_zero_point);
      vacc89AB = vsubq_s32(vacc89AB, vmagic_bias_less_zero_point);
      vaccCDEF = vsubq_s32(vaccCDEF, vmagic_bias_less_zero_point);
      vaccGHIJ = vsubq_s32(vaccGHIJ, vmagic_bias_less_zero_point);
      vaccKLMN = vsubq_s32(vaccKLMN, vmagic_bias_less_zero_point);

#if XNN_ARCH_ARM64
      const int16x8_t vacc01234567 = vuzp1q_s16(vreinterpretq_s16_s32(vacc0123), vreinterpretq_s16_s32(vacc4567));
      const int16x8_t vacc89ABCDEF = vuzp1q_s16(vreinterpretq_s16_s32(vacc89AB), vreinterpretq_s16_s32(vaccCDEF));
      const int16x8_t vaccGHIJKLMN = vuzp1q_s16(vreinterpretq_s16_s32(vaccGHIJ), vreinterpretq_s16_s32(vaccKLMN));

      int8x16_t vout0123456789ABCDEF = vuzp1q_s8(vreinterpretq_s8_s16(vacc01234567), vreinterpretq_s8_s16(vacc89ABCDEF));
      int8x8_t voutGHIJKLMN = vmovn_s16(vaccGHIJKLMN);
#else
      const int16x8_t vacc01234567 = vcombine_s16(vmovn_s32(vacc0123), vmovn_s32(vacc4567));
      const int16x8_t vacc89ABCDEF = vcombine_s16(vmovn_s32(vacc89AB), vmovn_s32(vaccCDEF));
      const int16x8_t vaccGHIJKLMN = vcombine_s16(vmovn_s32(vaccGHIJ), vmovn_s32(vaccKLMN));

      int8x16_t vout0123456789ABCDEF = vcombine_s8(vmovn_s16(vacc01234567), vmovn_s16(vacc89ABCDEF));
      int8x8_t voutGHIJKLMN = vmovn_s16(vaccGHIJKLMN);
#endif


      vst1q_s8(output, vout0123456789ABCDEF); output += 16;
      vst1_s8(output, voutGHIJKLMN); output += 8;
    }
    if XNN_UNLIKELY(c != 0) {
      const int8_t* k = (const int8_t*) ((const int32_t*) w + 24);
      do {
        int32x4_t vacc0123 = vld1q_s32(w); w = (const void*) ((const int32_t*) w + 4);
        int32x4_t vacc4567 = vld1q_s32(w); w = (const void*) ((const int32_t*) w + 4);

        const int16x8_t vi0x01234567 = vmovl_s8(vld1_s8(i0)); i0 += 8;
        const int16x8_t vk0x01234567 = vmovl_s8(vld1_s8(k)); k += 8;

        vacc0123 = vmlal_s16(vacc0123, vget_low_s16(vi0x01234567), vget_low_s16(vk0x01234567));
        vacc4567 = vmlal_s16(vacc4567, vget_high_s16(vi0x01234567), vget_high_s16(vk0x01234567));
        const int16x8_t vi1x01234567 = vmovl_s8(vld1_s8(i1)); i1 += 8;
        const int16x8_t vk1x01234567 = vmovl_s8(vld1_s8((const void*) (k + 16)));

        vacc0123 = vmlal_s16(vacc0123, vget_low_s16(vi1x01234567), vget_low_s16(vk1x01234567));
        vacc4567 = vmlal_s16(vacc4567, vget_high_s16(vi1x01234567), vget_high_s16(vk1x01234567));
        const int16x8_t vi2x01234567 = vmovl_s8(vld1_s8(i2)); i2 += 8;
        const int16x8_t vk2x01234567 = vmovl_s8(vld1_s8((const void*) (k + 40)));

        vacc0123 = vmlal_s16(vacc0123, vget_low_s16(vi2x01234567), vget_low_s16(vk2x01234567));
        vacc4567 = vmlal_s16(vacc4567, vget_high_s16(vi2x01234567), vget_high_s16(vk2x01234567));
        const int16x8_t vi3x01234567 = vmovl_s8(vld1_s8(i3)); i3 += 8;
        const int16x8_t vk3x01234567 = vmovl_s8(vld1_s8((const void*) (k + 64)));

        vacc0123 = vmlal_s16(vacc0123, vget_low_s16(vi3x01234567), vget_low_s16(vk3x01234567));
        vacc4567 = vmlal_s16(vacc4567, vget_high_s16(vi3x01234567), vget_high_s16(vk3x01234567));
        const int16x8_t vi4x01234567 = vmovl_s8(vld1_s8(i4)); i4 += 8;
        const int16x8_t vk4x01234567 = vmovl_s8(vld1_s8((const void*) (k + 88)));

        vacc0123 = vmlal_s16(vacc0123, vget_low_s16(vi4x01234567), vget_low_s16(vk4x01234567));
        vacc4567 = vmlal_s16(vacc4567, vget_high_s16(vi4x01234567), vget_high_s16(vk4x01234567));
        const int16x8_t vi5x01234567 = vmovl_s8(vld1_s8(i5)); i5 += 8;
        const int16x8_t vk5x01234567 = vmovl_s8(vld1_s8((const void*) (k + 112)));

        vacc0123 = vmlal_s16(vacc0123, vget_low_s16(vi5x01234567), vget_low_s16(vk5x01234567));
        vacc4567 = vmlal_s16(vacc4567, vget_high_s16(vi5x01234567), vget_high_s16(vk5x01234567));
        const int16x8_t vi6x01234567 = vmovl_s8(vld1_s8(i6)); i6 += 8;
        const int16x8_t vk6x01234567 = vmovl_s8(vld1_s8((const void*) (k + 136)));

        vacc0123 = vmlal_s16(vacc0123, vget_low_s16(vi6x01234567), vget_low_s16(vk6x01234567));
        vacc4567 = vmlal_s16(vacc4567, vget_high_s16(vi6x01234567), vget_high_s16(vk6x01234567));
        const int16x8_t vi7x01234567 = vmovl_s8(vld1_s8(i7)); i7 += 8;
        const int16x8_t vk7x01234567 = vmovl_s8(vld1_s8((const void*) (k + 160)));

        vacc0123 = vmlal_s16(vacc0123, vget_low_s16(vi7x01234567), vget_low_s16(vk7x01234567));
        vacc4567 = vmlal_s16(vacc4567, vget_high_s16(vi7x01234567), vget_high_s16(vk7x01234567));
        const int16x8_t vi8x01234567 = vmovl_s8(vld1_s8(i8)); i8 += 8;
        const int16x8_t vk8x01234567 = vmovl_s8(vld1_s8((const void*) (k + 184)));

        vacc0123 = vmlal_s16(vacc0123, vget_low_s16(vi8x01234567), vget_low_s16(vk8x01234567));
        vacc4567 = vmlal_s16(vacc4567, vget_high_s16(vi8x01234567), vget_high_s16(vk8x01234567));
        const int16x8_t vi9x01234567 = vmovl_s8(vld1_s8(i9)); i9 += 8;
        const int16x8_t vk9x01234567 = vmovl_s8(vld1_s8((const void*) (k + 208)));

        vacc0123 = vmlal_s16(vacc0123, vget_low_s16(vi9x01234567), vget_low_s16(vk9x01234567));
        vacc4567 = vmlal_s16(vacc4567, vget_high_s16(vi9x01234567), vget_high_s16(vk9x01234567));
        const int16x8_t vi10x01234567 = vmovl_s8(vld1_s8(i10)); i10 += 8;
        const int16x8_t vk10x01234567 = vmovl_s8(vld1_s8((const void*) (k + 232)));

        vacc0123 = vmlal_s16(vacc0123, vget_low_s16(vi10x01234567), vget_low_s16(vk10x01234567));
        vacc4567 = vmlal_s16(vacc4567, vget_high_s16(vi10x01234567), vget_high_s16(vk10x01234567));
        const int16x8_t vi11x01234567 = vmovl_s8(vld1_s8(i11)); i11 += 8;
        const int16x8_t vk11x01234567 = vmovl_s8(vld1_s8((const void*) (k + 256)));

        vacc0123 = vmlal_s16(vacc0123, vget_low_s16(vi11x01234567), vget_low_s16(vk11x01234567));
        vacc4567 = vmlal_s16(vacc4567, vget_high_s16(vi11x01234567), vget_high_s16(vk11x01234567));
        const int16x8_t vi12x01234567 = vmovl_s8(vld1_s8(i12)); i12 += 8;
        const int16x8_t vk12x01234567 = vmovl_s8(vld1_s8((const void*) (k + 280)));

        vacc0123 = vmlal_s16(vacc0123, vget_low_s16(vi12x01234567), vget_low_s16(vk12x01234567));
        vacc4567 = vmlal_s16(vacc4567, vget_high_s16(vi12x01234567), vget_high_s16(vk12x01234567));
        const int16x8_t vi13x01234567 = vmovl_s8(vld1_s8(i13)); i13 += 8;
        const int16x8_t vk13x01234567 = vmovl_s8(vld1_s8((const void*) (k + 304)));

        vacc0123 = vmlal_s16(vacc0123, vget_low_s16(vi13x01234567), vget_low_s16(vk13x01234567));
        vacc4567 = vmlal_s16(vacc4567, vget_high_s16(vi13x01234567), vget_high_s16(vk13x01234567));
        const int16x8_t vi14x01234567 = vmovl_s8(vld1_s8(i14)); i14 += 8;
        const int16x8_t vk14x01234567 = vmovl_s8(vld1_s8((const void*) (k + 328)));

        vacc0123 = vmlal_s16(vacc0123, vget_low_s16(vi14x01234567), vget_low_s16(vk14x01234567));
        vacc4567 = vmlal_s16(vacc4567, vget_high_s16(vi14x01234567), vget_high_s16(vk14x01234567));
        const int16x8_t vi15x01234567 = vmovl_s8(vld1_s8(i15)); i15 += 8;
        const int16x8_t vk15x01234567 = vmovl_s8(vld1_s8((const void*) (k + 352)));

        vacc0123 = vmlal_s16(vacc0123, vget_low_s16(vi15x01234567), vget_low_s16(vk15x01234567));
        vacc4567 = vmlal_s16(vacc4567, vget_high_s16(vi15x01234567), vget_high_s16(vk15x01234567));
        const int16x8_t vi16x01234567 = vmovl_s8(vld1_s8(i16)); i16 += 8;
        const int16x8_t vk16x01234567 = vmovl_s8(vld1_s8((const void*) (k + 376)));

        vacc0123 = vmlal_s16(vacc0123, vget_low_s16(vi16x01234567), vget_low_s16(vk16x01234567));
        vacc4567 = vmlal_s16(vacc4567, vget_high_s16(vi16x01234567), vget_high_s16(vk16x01234567));
        const int16x8_t vi17x01234567 = vmovl_s8(vld1_s8(i17)); i17 += 8;
        const int16x8_t vk17x01234567 = vmovl_s8(vld1_s8((const void*) (k + 400)));

        vacc0123 = vmlal_s16(vacc0123, vget_low_s16(vi17x01234567), vget_low_s16(vk17x01234567));
        vacc4567 = vmlal_s16(vacc4567, vget_high_s16(vi17x01234567), vget_high_s16(vk17x01234567));
        const int16x8_t vi18x01234567 = vmovl_s8(vld1_s8(i18)); i18 += 8;
        const int16x8_t vk18x01234567 = vmovl_s8(vld1_s8((const void*) (k + 424)));

        vacc0123 = vmlal_s16(vacc0123, vget_low_s16(vi18x01234567), vget_low_s16(vk18x01234567));
        vacc4567 = vmlal_s16(vacc4567, vget_high_s16(vi18x01234567), vget_high_s16(vk18x01234567));
        const int16x8_t vi19x01234567 = vmovl_s8(vld1_s8(i19)); i19 += 8;
        const int16x8_t vk19x01234567 = vmovl_s8(vld1_s8((const void*) (k + 448)));

        vacc0123 = vmlal_s16(vacc0123, vget_low_s16(vi19x01234567), vget_low_s16(vk19x01234567));
        vacc4567 = vmlal_s16(vacc4567, vget_high_s16(vi19x01234567), vget_high_s16(vk19x01234567));
        const int16x8_t vi20x01234567 = vmovl_s8(vld1_s8(i20)); i20 += 8;
        const int16x8_t vk20x01234567 = vmovl_s8(vld1_s8((const void*) (k + 472)));

        vacc0123 = vmlal_s16(vacc0123, vget_low_s16(vi20x01234567), vget_low_s16(vk20x01234567));
        vacc4567 = vmlal_s16(vacc4567, vget_high_s16(vi20x01234567), vget_high_s16(vk20x01234567));
        const int16x8_t vi21x01234567 = vmovl_s8(vld1_s8(i21)); i21 += 8;
        const int16x8_t vk21x01234567 = vmovl_s8(vld1_s8((const void*) (k + 496)));

        vacc0123 = vmlal_s16(vacc0123, vget_low_s16(vi21x01234567), vget_low_s16(vk21x01234567));
        vacc4567 = vmlal_s16(vacc4567, vget_high_s16(vi21x01234567), vget_high_s16(vk21x01234567));
        const int16x8_t vi22x01234567 = vmovl_s8(vld1_s8(i22)); i22 += 8;
        const int16x8_t vk22x01234567 = vmovl_s8(vld1_s8((const void*) (k + 520)));

        vacc0123 = vmlal_s16(vacc0123, vget_low_s16(vi22x01234567), vget_low_s16(vk22x01234567));
        vacc4567 = vmlal_s16(vacc4567, vget_high_s16(vi22x01234567), vget_high_s16(vk22x01234567));
        const int16x8_t vi23x01234567 = vmovl_s8(vld1_s8(i23)); i23 += 8;
        const int16x8_t vk23x01234567 = vmovl_s8(vld1_s8((const void*) (k + 544)));

        vacc0123 = vmlal_s16(vacc0123, vget_low_s16(vi23x01234567), vget_low_s16(vk23x01234567));
        vacc4567 = vmlal_s16(vacc4567, vget_high_s16(vi23x01234567), vget_high_s16(vk23x01234567));
        const int16x8_t vi24x01234567 = vmovl_s8(vld1_s8(i24)); i24 += 8;
        const int16x8_t vk24x01234567 = vmovl_s8(vld1_s8((const void*) (k + 568)));

        vacc0123 = vmlal_s16(vacc0123, vget_low_s16(vi24x01234567), vget_low_s16(vk24x01234567));
        vacc4567 = vmlal_s16(vacc4567, vget_high_s16(vi24x01234567), vget_high_s16(vk24x01234567));

        float32x4_t vfpacc0123 = vcvtq_f32_s32(vacc0123);
        float32x4_t vfpacc4567 = vcvtq_f32_s32(vacc4567);

        const float32x4_t vscale0123 = vld1q_f32((const float*) ((uintptr_t) w + 16 * sizeof(int32_t) + 600 * sizeof(int8_t)));
        const float32x4_t vscale4567 = vld1q_f32((const float*) ((uintptr_t) w + 16 * sizeof(int32_t) + 600 * sizeof(int8_t) + 4 * sizeof(float)));
        vfpacc0123 = vmulq_f32(vfpacc0123, vscale0123);
        vfpacc4567 = vmulq_f32(vfpacc4567, vscale4567);

        vfpacc0123 = vmaxq_f32(vfpacc0123, voutput_min_less_zero_point);
        vfpacc4567 = vmaxq_f32(vfpacc4567, voutput_min_less_zero_point);

        vfpacc0123 = vminq_f32(vfpacc0123, voutput_max_less_zero_point);
        vfpacc4567 = vminq_f32(vfpacc4567, voutput_max_less_zero_point);

        vacc0123 = vreinterpretq_s32_f32(vaddq_f32(vfpacc0123, vmagic_bias));
        vacc4567 = vreinterpretq_s32_f32(vaddq_f32(vfpacc4567, vmagic_bias));

        vacc0123 = vsubq_s32(vacc0123, vmagic_bias_less_zero_point);
        vacc4567 = vsubq_s32(vacc4567, vmagic_bias_less_zero_point);

#if XNN_ARCH_ARM64
        const int16x8_t vacc01234567 = vuzp1q_s16(vreinterpretq_s16_s32(vacc0123), vreinterpretq_s16_s32(vacc4567));
        int8x8_t vout01234567 = vmovn_s16(vacc01234567);
#else
        const int16x8_t vacc01234567 = vcombine_s16(vmovn_s32(vacc0123), vmovn_s32(vacc4567));
        int8x8_t vout01234567 = vmovn_s16(vacc01234567);
#endif


        if XNN_LIKELY(c >= 8) {
          vst1_s8(output, vout01234567); output += 8;
          c -= 8;
        } else {
          if (c & 4) {
            vst1_lane_u32(__builtin_assume_aligned(output, 1), vreinterpret_u32_s8(vout01234567), 0); output += 4;
            vout01234567 = vext_s8(vout01234567, vout01234567, 4);
          }
          if (c & 2) {
            vst1_lane_u16(__builtin_assume_aligned(output, 1), vreinterpret_u16_s8(vout01234567), 0); output += 2;
            vout01234567 = vext_s8(vout01234567, vout01234567, 2);
          }
          if (c & 1) {
            vst1_lane_s8(output, vout01234567, 0); output += 1;
          }
          c = 0;
        }
      } while (c != 0);
    }

    output = (int8_t*) ((uintptr_t) output + output_increment);
  } while (--output_width != 0);
}
