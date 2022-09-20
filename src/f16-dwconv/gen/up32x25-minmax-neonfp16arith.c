// Auto-generated file. Do not edit!
//   Template: src/f16-dwconv/unipass-neonfp16arith.c.in
//   Generator: tools/xngen
//
// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <arm_neon.h>

#include <xnnpack/dwconv.h>


void xnn_f16_dwconv_minmax_ukernel_up32x25__neonfp16arith(
    size_t channels,
    size_t output_width,
    const void** input,
    const void* weights,
    void* output_ptr,
    size_t input_stride,
    size_t output_increment,
    size_t input_offset,
    const void* zero,
    const union xnn_f16_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(channels != 0);
  assert(output_width != 0);

  __fp16* output = (__fp16*) output_ptr;
  const float16x8_t vmax = vreinterpretq_f16_u16(vld1q_dup_u16(&params->fp16arith.max));
  const float16x8_t vmin = vreinterpretq_f16_u16(vld1q_dup_u16(&params->fp16arith.min));
  do {
    const __fp16* i0 = (const __fp16*) input[0];
    assert(i0 != NULL);
    if XNN_UNPREDICTABLE(i0 != (const __fp16*) zero) {
      i0 = (const __fp16*) ((uintptr_t) i0 + input_offset);
    }
    const __fp16* i1 = (const __fp16*) input[1];
    assert(i1 != NULL);
    if XNN_UNPREDICTABLE(i1 != (const __fp16*) zero) {
      i1 = (const __fp16*) ((uintptr_t) i1 + input_offset);
    }
    const __fp16* i2 = (const __fp16*) input[2];
    assert(i2 != NULL);
    if XNN_UNPREDICTABLE(i2 != (const __fp16*) zero) {
      i2 = (const __fp16*) ((uintptr_t) i2 + input_offset);
    }
    const __fp16* i3 = (const __fp16*) input[3];
    assert(i3 != NULL);
    if XNN_UNPREDICTABLE(i3 != (const __fp16*) zero) {
      i3 = (const __fp16*) ((uintptr_t) i3 + input_offset);
    }
    const __fp16* i4 = (const __fp16*) input[4];
    assert(i4 != NULL);
    if XNN_UNPREDICTABLE(i4 != (const __fp16*) zero) {
      i4 = (const __fp16*) ((uintptr_t) i4 + input_offset);
    }
    const __fp16* i5 = (const __fp16*) input[5];
    assert(i5 != NULL);
    if XNN_UNPREDICTABLE(i5 != (const __fp16*) zero) {
      i5 = (const __fp16*) ((uintptr_t) i5 + input_offset);
    }
    const __fp16* i6 = (const __fp16*) input[6];
    assert(i6 != NULL);
    if XNN_UNPREDICTABLE(i6 != (const __fp16*) zero) {
      i6 = (const __fp16*) ((uintptr_t) i6 + input_offset);
    }
    const __fp16* i7 = (const __fp16*) input[7];
    assert(i7 != NULL);
    if XNN_UNPREDICTABLE(i7 != (const __fp16*) zero) {
      i7 = (const __fp16*) ((uintptr_t) i7 + input_offset);
    }
    const __fp16* i8 = (const __fp16*) input[8];
    assert(i8 != NULL);
    if XNN_UNPREDICTABLE(i8 != (const __fp16*) zero) {
      i8 = (const __fp16*) ((uintptr_t) i8 + input_offset);
    }
    const __fp16* i9 = (const __fp16*) input[9];
    assert(i9 != NULL);
    if XNN_UNPREDICTABLE(i9 != (const __fp16*) zero) {
      i9 = (const __fp16*) ((uintptr_t) i9 + input_offset);
    }
    const __fp16* i10 = (const __fp16*) input[10];
    assert(i10 != NULL);
    if XNN_UNPREDICTABLE(i10 != (const __fp16*) zero) {
      i10 = (const __fp16*) ((uintptr_t) i10 + input_offset);
    }
    const __fp16* i11 = (const __fp16*) input[11];
    assert(i11 != NULL);
    if XNN_UNPREDICTABLE(i11 != (const __fp16*) zero) {
      i11 = (const __fp16*) ((uintptr_t) i11 + input_offset);
    }
    const __fp16* i12 = (const __fp16*) input[12];
    assert(i12 != NULL);
    if XNN_UNPREDICTABLE(i12 != (const __fp16*) zero) {
      i12 = (const __fp16*) ((uintptr_t) i12 + input_offset);
    }
    const __fp16* i13 = (const __fp16*) input[13];
    assert(i13 != NULL);
    if XNN_UNPREDICTABLE(i13 != (const __fp16*) zero) {
      i13 = (const __fp16*) ((uintptr_t) i13 + input_offset);
    }
    const __fp16* i14 = (const __fp16*) input[14];
    assert(i14 != NULL);
    if XNN_UNPREDICTABLE(i14 != (const __fp16*) zero) {
      i14 = (const __fp16*) ((uintptr_t) i14 + input_offset);
    }
    const __fp16* i15 = (const __fp16*) input[15];
    assert(i15 != NULL);
    if XNN_UNPREDICTABLE(i15 != (const __fp16*) zero) {
      i15 = (const __fp16*) ((uintptr_t) i15 + input_offset);
    }
    const __fp16* i16 = (const __fp16*) input[16];
    assert(i16 != NULL);
    if XNN_UNPREDICTABLE(i16 != (const __fp16*) zero) {
      i16 = (const __fp16*) ((uintptr_t) i16 + input_offset);
    }
    const __fp16* i17 = (const __fp16*) input[17];
    assert(i17 != NULL);
    if XNN_UNPREDICTABLE(i17 != (const __fp16*) zero) {
      i17 = (const __fp16*) ((uintptr_t) i17 + input_offset);
    }
    const __fp16* i18 = (const __fp16*) input[18];
    assert(i18 != NULL);
    if XNN_UNPREDICTABLE(i18 != (const __fp16*) zero) {
      i18 = (const __fp16*) ((uintptr_t) i18 + input_offset);
    }
    const __fp16* i19 = (const __fp16*) input[19];
    assert(i19 != NULL);
    if XNN_UNPREDICTABLE(i19 != (const __fp16*) zero) {
      i19 = (const __fp16*) ((uintptr_t) i19 + input_offset);
    }
    const __fp16* i20 = (const __fp16*) input[20];
    assert(i20 != NULL);
    if XNN_UNPREDICTABLE(i20 != (const __fp16*) zero) {
      i20 = (const __fp16*) ((uintptr_t) i20 + input_offset);
    }
    const __fp16* i21 = (const __fp16*) input[21];
    assert(i21 != NULL);
    if XNN_UNPREDICTABLE(i21 != (const __fp16*) zero) {
      i21 = (const __fp16*) ((uintptr_t) i21 + input_offset);
    }
    const __fp16* i22 = (const __fp16*) input[22];
    assert(i22 != NULL);
    if XNN_UNPREDICTABLE(i22 != (const __fp16*) zero) {
      i22 = (const __fp16*) ((uintptr_t) i22 + input_offset);
    }
    const __fp16* i23 = (const __fp16*) input[23];
    assert(i23 != NULL);
    if XNN_UNPREDICTABLE(i23 != (const __fp16*) zero) {
      i23 = (const __fp16*) ((uintptr_t) i23 + input_offset);
    }
    const __fp16* i24 = (const __fp16*) input[24];
    assert(i24 != NULL);
    if XNN_UNPREDICTABLE(i24 != (const __fp16*) zero) {
      i24 = (const __fp16*) ((uintptr_t) i24 + input_offset);
    }

    input = (const void**) ((uintptr_t) input + input_stride);

    size_t c = channels;
    const __fp16* w = (const __fp16*) weights;
    for (; c >= 32; c -= 32) {
      float16x8_t vacc01234567p0 = vld1q_f16(w); w += 8;
      float16x8_t vacc89ABCDEFp0 = vld1q_f16(w); w += 8;
      float16x8_t vaccGHIJKLMNp0 = vld1q_f16(w); w += 8;
      float16x8_t vaccOPQRSTUVp0 = vld1q_f16(w); w += 8;


      const float16x8_t vi0x01234567 = vld1q_f16(i0); i0 += 8;
      const float16x8_t vi0x89ABCDEF = vld1q_f16(i0); i0 += 8;
      const float16x8_t vi0xGHIJKLMN = vld1q_f16(i0); i0 += 8;
      const float16x8_t vi0xOPQRSTUV = vld1q_f16(i0); i0 += 8;
      const float16x8_t vk0x01234567 = vld1q_f16(w); w += 8;
      const float16x8_t vk0x89ABCDEF = vld1q_f16(w); w += 8;
      const float16x8_t vk0xGHIJKLMN = vld1q_f16(w); w += 8;
      const float16x8_t vk0xOPQRSTUV = vld1q_f16(w); w += 8;
      vacc01234567p0 = vfmaq_f16(vacc01234567p0, vi0x01234567, vk0x01234567);
      vacc89ABCDEFp0 = vfmaq_f16(vacc89ABCDEFp0, vi0x89ABCDEF, vk0x89ABCDEF);
      vaccGHIJKLMNp0 = vfmaq_f16(vaccGHIJKLMNp0, vi0xGHIJKLMN, vk0xGHIJKLMN);
      vaccOPQRSTUVp0 = vfmaq_f16(vaccOPQRSTUVp0, vi0xOPQRSTUV, vk0xOPQRSTUV);

      const float16x8_t vi1x01234567 = vld1q_f16(i1); i1 += 8;
      const float16x8_t vi1x89ABCDEF = vld1q_f16(i1); i1 += 8;
      const float16x8_t vi1xGHIJKLMN = vld1q_f16(i1); i1 += 8;
      const float16x8_t vi1xOPQRSTUV = vld1q_f16(i1); i1 += 8;
      const float16x8_t vk1x01234567 = vld1q_f16(w); w += 8;
      const float16x8_t vk1x89ABCDEF = vld1q_f16(w); w += 8;
      const float16x8_t vk1xGHIJKLMN = vld1q_f16(w); w += 8;
      const float16x8_t vk1xOPQRSTUV = vld1q_f16(w); w += 8;
      vacc01234567p0 = vfmaq_f16(vacc01234567p0, vi1x01234567, vk1x01234567);
      vacc89ABCDEFp0 = vfmaq_f16(vacc89ABCDEFp0, vi1x89ABCDEF, vk1x89ABCDEF);
      vaccGHIJKLMNp0 = vfmaq_f16(vaccGHIJKLMNp0, vi1xGHIJKLMN, vk1xGHIJKLMN);
      vaccOPQRSTUVp0 = vfmaq_f16(vaccOPQRSTUVp0, vi1xOPQRSTUV, vk1xOPQRSTUV);

      const float16x8_t vi2x01234567 = vld1q_f16(i2); i2 += 8;
      const float16x8_t vi2x89ABCDEF = vld1q_f16(i2); i2 += 8;
      const float16x8_t vi2xGHIJKLMN = vld1q_f16(i2); i2 += 8;
      const float16x8_t vi2xOPQRSTUV = vld1q_f16(i2); i2 += 8;
      const float16x8_t vk2x01234567 = vld1q_f16(w); w += 8;
      const float16x8_t vk2x89ABCDEF = vld1q_f16(w); w += 8;
      const float16x8_t vk2xGHIJKLMN = vld1q_f16(w); w += 8;
      const float16x8_t vk2xOPQRSTUV = vld1q_f16(w); w += 8;
      vacc01234567p0 = vfmaq_f16(vacc01234567p0, vi2x01234567, vk2x01234567);
      vacc89ABCDEFp0 = vfmaq_f16(vacc89ABCDEFp0, vi2x89ABCDEF, vk2x89ABCDEF);
      vaccGHIJKLMNp0 = vfmaq_f16(vaccGHIJKLMNp0, vi2xGHIJKLMN, vk2xGHIJKLMN);
      vaccOPQRSTUVp0 = vfmaq_f16(vaccOPQRSTUVp0, vi2xOPQRSTUV, vk2xOPQRSTUV);

      const float16x8_t vi3x01234567 = vld1q_f16(i3); i3 += 8;
      const float16x8_t vi3x89ABCDEF = vld1q_f16(i3); i3 += 8;
      const float16x8_t vi3xGHIJKLMN = vld1q_f16(i3); i3 += 8;
      const float16x8_t vi3xOPQRSTUV = vld1q_f16(i3); i3 += 8;
      const float16x8_t vk3x01234567 = vld1q_f16(w); w += 8;
      const float16x8_t vk3x89ABCDEF = vld1q_f16(w); w += 8;
      const float16x8_t vk3xGHIJKLMN = vld1q_f16(w); w += 8;
      const float16x8_t vk3xOPQRSTUV = vld1q_f16(w); w += 8;
      vacc01234567p0 = vfmaq_f16(vacc01234567p0, vi3x01234567, vk3x01234567);
      vacc89ABCDEFp0 = vfmaq_f16(vacc89ABCDEFp0, vi3x89ABCDEF, vk3x89ABCDEF);
      vaccGHIJKLMNp0 = vfmaq_f16(vaccGHIJKLMNp0, vi3xGHIJKLMN, vk3xGHIJKLMN);
      vaccOPQRSTUVp0 = vfmaq_f16(vaccOPQRSTUVp0, vi3xOPQRSTUV, vk3xOPQRSTUV);

      const float16x8_t vi4x01234567 = vld1q_f16(i4); i4 += 8;
      const float16x8_t vi4x89ABCDEF = vld1q_f16(i4); i4 += 8;
      const float16x8_t vi4xGHIJKLMN = vld1q_f16(i4); i4 += 8;
      const float16x8_t vi4xOPQRSTUV = vld1q_f16(i4); i4 += 8;
      const float16x8_t vk4x01234567 = vld1q_f16(w); w += 8;
      const float16x8_t vk4x89ABCDEF = vld1q_f16(w); w += 8;
      const float16x8_t vk4xGHIJKLMN = vld1q_f16(w); w += 8;
      const float16x8_t vk4xOPQRSTUV = vld1q_f16(w); w += 8;
      vacc01234567p0 = vfmaq_f16(vacc01234567p0, vi4x01234567, vk4x01234567);
      vacc89ABCDEFp0 = vfmaq_f16(vacc89ABCDEFp0, vi4x89ABCDEF, vk4x89ABCDEF);
      vaccGHIJKLMNp0 = vfmaq_f16(vaccGHIJKLMNp0, vi4xGHIJKLMN, vk4xGHIJKLMN);
      vaccOPQRSTUVp0 = vfmaq_f16(vaccOPQRSTUVp0, vi4xOPQRSTUV, vk4xOPQRSTUV);

      const float16x8_t vi5x01234567 = vld1q_f16(i5); i5 += 8;
      const float16x8_t vi5x89ABCDEF = vld1q_f16(i5); i5 += 8;
      const float16x8_t vi5xGHIJKLMN = vld1q_f16(i5); i5 += 8;
      const float16x8_t vi5xOPQRSTUV = vld1q_f16(i5); i5 += 8;
      const float16x8_t vk5x01234567 = vld1q_f16(w); w += 8;
      const float16x8_t vk5x89ABCDEF = vld1q_f16(w); w += 8;
      const float16x8_t vk5xGHIJKLMN = vld1q_f16(w); w += 8;
      const float16x8_t vk5xOPQRSTUV = vld1q_f16(w); w += 8;
      vacc01234567p0 = vfmaq_f16(vacc01234567p0, vi5x01234567, vk5x01234567);
      vacc89ABCDEFp0 = vfmaq_f16(vacc89ABCDEFp0, vi5x89ABCDEF, vk5x89ABCDEF);
      vaccGHIJKLMNp0 = vfmaq_f16(vaccGHIJKLMNp0, vi5xGHIJKLMN, vk5xGHIJKLMN);
      vaccOPQRSTUVp0 = vfmaq_f16(vaccOPQRSTUVp0, vi5xOPQRSTUV, vk5xOPQRSTUV);

      const float16x8_t vi6x01234567 = vld1q_f16(i6); i6 += 8;
      const float16x8_t vi6x89ABCDEF = vld1q_f16(i6); i6 += 8;
      const float16x8_t vi6xGHIJKLMN = vld1q_f16(i6); i6 += 8;
      const float16x8_t vi6xOPQRSTUV = vld1q_f16(i6); i6 += 8;
      const float16x8_t vk6x01234567 = vld1q_f16(w); w += 8;
      const float16x8_t vk6x89ABCDEF = vld1q_f16(w); w += 8;
      const float16x8_t vk6xGHIJKLMN = vld1q_f16(w); w += 8;
      const float16x8_t vk6xOPQRSTUV = vld1q_f16(w); w += 8;
      vacc01234567p0 = vfmaq_f16(vacc01234567p0, vi6x01234567, vk6x01234567);
      vacc89ABCDEFp0 = vfmaq_f16(vacc89ABCDEFp0, vi6x89ABCDEF, vk6x89ABCDEF);
      vaccGHIJKLMNp0 = vfmaq_f16(vaccGHIJKLMNp0, vi6xGHIJKLMN, vk6xGHIJKLMN);
      vaccOPQRSTUVp0 = vfmaq_f16(vaccOPQRSTUVp0, vi6xOPQRSTUV, vk6xOPQRSTUV);

      const float16x8_t vi7x01234567 = vld1q_f16(i7); i7 += 8;
      const float16x8_t vi7x89ABCDEF = vld1q_f16(i7); i7 += 8;
      const float16x8_t vi7xGHIJKLMN = vld1q_f16(i7); i7 += 8;
      const float16x8_t vi7xOPQRSTUV = vld1q_f16(i7); i7 += 8;
      const float16x8_t vk7x01234567 = vld1q_f16(w); w += 8;
      const float16x8_t vk7x89ABCDEF = vld1q_f16(w); w += 8;
      const float16x8_t vk7xGHIJKLMN = vld1q_f16(w); w += 8;
      const float16x8_t vk7xOPQRSTUV = vld1q_f16(w); w += 8;
      vacc01234567p0 = vfmaq_f16(vacc01234567p0, vi7x01234567, vk7x01234567);
      vacc89ABCDEFp0 = vfmaq_f16(vacc89ABCDEFp0, vi7x89ABCDEF, vk7x89ABCDEF);
      vaccGHIJKLMNp0 = vfmaq_f16(vaccGHIJKLMNp0, vi7xGHIJKLMN, vk7xGHIJKLMN);
      vaccOPQRSTUVp0 = vfmaq_f16(vaccOPQRSTUVp0, vi7xOPQRSTUV, vk7xOPQRSTUV);

      const float16x8_t vi8x01234567 = vld1q_f16(i8); i8 += 8;
      const float16x8_t vi8x89ABCDEF = vld1q_f16(i8); i8 += 8;
      const float16x8_t vi8xGHIJKLMN = vld1q_f16(i8); i8 += 8;
      const float16x8_t vi8xOPQRSTUV = vld1q_f16(i8); i8 += 8;
      const float16x8_t vk8x01234567 = vld1q_f16(w); w += 8;
      const float16x8_t vk8x89ABCDEF = vld1q_f16(w); w += 8;
      const float16x8_t vk8xGHIJKLMN = vld1q_f16(w); w += 8;
      const float16x8_t vk8xOPQRSTUV = vld1q_f16(w); w += 8;
      vacc01234567p0 = vfmaq_f16(vacc01234567p0, vi8x01234567, vk8x01234567);
      vacc89ABCDEFp0 = vfmaq_f16(vacc89ABCDEFp0, vi8x89ABCDEF, vk8x89ABCDEF);
      vaccGHIJKLMNp0 = vfmaq_f16(vaccGHIJKLMNp0, vi8xGHIJKLMN, vk8xGHIJKLMN);
      vaccOPQRSTUVp0 = vfmaq_f16(vaccOPQRSTUVp0, vi8xOPQRSTUV, vk8xOPQRSTUV);

      const float16x8_t vi9x01234567 = vld1q_f16(i9); i9 += 8;
      const float16x8_t vi9x89ABCDEF = vld1q_f16(i9); i9 += 8;
      const float16x8_t vi9xGHIJKLMN = vld1q_f16(i9); i9 += 8;
      const float16x8_t vi9xOPQRSTUV = vld1q_f16(i9); i9 += 8;
      const float16x8_t vk9x01234567 = vld1q_f16(w); w += 8;
      const float16x8_t vk9x89ABCDEF = vld1q_f16(w); w += 8;
      const float16x8_t vk9xGHIJKLMN = vld1q_f16(w); w += 8;
      const float16x8_t vk9xOPQRSTUV = vld1q_f16(w); w += 8;
      vacc01234567p0 = vfmaq_f16(vacc01234567p0, vi9x01234567, vk9x01234567);
      vacc89ABCDEFp0 = vfmaq_f16(vacc89ABCDEFp0, vi9x89ABCDEF, vk9x89ABCDEF);
      vaccGHIJKLMNp0 = vfmaq_f16(vaccGHIJKLMNp0, vi9xGHIJKLMN, vk9xGHIJKLMN);
      vaccOPQRSTUVp0 = vfmaq_f16(vaccOPQRSTUVp0, vi9xOPQRSTUV, vk9xOPQRSTUV);

      const float16x8_t vi10x01234567 = vld1q_f16(i10); i10 += 8;
      const float16x8_t vi10x89ABCDEF = vld1q_f16(i10); i10 += 8;
      const float16x8_t vi10xGHIJKLMN = vld1q_f16(i10); i10 += 8;
      const float16x8_t vi10xOPQRSTUV = vld1q_f16(i10); i10 += 8;
      const float16x8_t vk10x01234567 = vld1q_f16(w); w += 8;
      const float16x8_t vk10x89ABCDEF = vld1q_f16(w); w += 8;
      const float16x8_t vk10xGHIJKLMN = vld1q_f16(w); w += 8;
      const float16x8_t vk10xOPQRSTUV = vld1q_f16(w); w += 8;
      vacc01234567p0 = vfmaq_f16(vacc01234567p0, vi10x01234567, vk10x01234567);
      vacc89ABCDEFp0 = vfmaq_f16(vacc89ABCDEFp0, vi10x89ABCDEF, vk10x89ABCDEF);
      vaccGHIJKLMNp0 = vfmaq_f16(vaccGHIJKLMNp0, vi10xGHIJKLMN, vk10xGHIJKLMN);
      vaccOPQRSTUVp0 = vfmaq_f16(vaccOPQRSTUVp0, vi10xOPQRSTUV, vk10xOPQRSTUV);

      const float16x8_t vi11x01234567 = vld1q_f16(i11); i11 += 8;
      const float16x8_t vi11x89ABCDEF = vld1q_f16(i11); i11 += 8;
      const float16x8_t vi11xGHIJKLMN = vld1q_f16(i11); i11 += 8;
      const float16x8_t vi11xOPQRSTUV = vld1q_f16(i11); i11 += 8;
      const float16x8_t vk11x01234567 = vld1q_f16(w); w += 8;
      const float16x8_t vk11x89ABCDEF = vld1q_f16(w); w += 8;
      const float16x8_t vk11xGHIJKLMN = vld1q_f16(w); w += 8;
      const float16x8_t vk11xOPQRSTUV = vld1q_f16(w); w += 8;
      vacc01234567p0 = vfmaq_f16(vacc01234567p0, vi11x01234567, vk11x01234567);
      vacc89ABCDEFp0 = vfmaq_f16(vacc89ABCDEFp0, vi11x89ABCDEF, vk11x89ABCDEF);
      vaccGHIJKLMNp0 = vfmaq_f16(vaccGHIJKLMNp0, vi11xGHIJKLMN, vk11xGHIJKLMN);
      vaccOPQRSTUVp0 = vfmaq_f16(vaccOPQRSTUVp0, vi11xOPQRSTUV, vk11xOPQRSTUV);

      const float16x8_t vi12x01234567 = vld1q_f16(i12); i12 += 8;
      const float16x8_t vi12x89ABCDEF = vld1q_f16(i12); i12 += 8;
      const float16x8_t vi12xGHIJKLMN = vld1q_f16(i12); i12 += 8;
      const float16x8_t vi12xOPQRSTUV = vld1q_f16(i12); i12 += 8;
      const float16x8_t vk12x01234567 = vld1q_f16(w); w += 8;
      const float16x8_t vk12x89ABCDEF = vld1q_f16(w); w += 8;
      const float16x8_t vk12xGHIJKLMN = vld1q_f16(w); w += 8;
      const float16x8_t vk12xOPQRSTUV = vld1q_f16(w); w += 8;
      vacc01234567p0 = vfmaq_f16(vacc01234567p0, vi12x01234567, vk12x01234567);
      vacc89ABCDEFp0 = vfmaq_f16(vacc89ABCDEFp0, vi12x89ABCDEF, vk12x89ABCDEF);
      vaccGHIJKLMNp0 = vfmaq_f16(vaccGHIJKLMNp0, vi12xGHIJKLMN, vk12xGHIJKLMN);
      vaccOPQRSTUVp0 = vfmaq_f16(vaccOPQRSTUVp0, vi12xOPQRSTUV, vk12xOPQRSTUV);

      const float16x8_t vi13x01234567 = vld1q_f16(i13); i13 += 8;
      const float16x8_t vi13x89ABCDEF = vld1q_f16(i13); i13 += 8;
      const float16x8_t vi13xGHIJKLMN = vld1q_f16(i13); i13 += 8;
      const float16x8_t vi13xOPQRSTUV = vld1q_f16(i13); i13 += 8;
      const float16x8_t vk13x01234567 = vld1q_f16(w); w += 8;
      const float16x8_t vk13x89ABCDEF = vld1q_f16(w); w += 8;
      const float16x8_t vk13xGHIJKLMN = vld1q_f16(w); w += 8;
      const float16x8_t vk13xOPQRSTUV = vld1q_f16(w); w += 8;
      vacc01234567p0 = vfmaq_f16(vacc01234567p0, vi13x01234567, vk13x01234567);
      vacc89ABCDEFp0 = vfmaq_f16(vacc89ABCDEFp0, vi13x89ABCDEF, vk13x89ABCDEF);
      vaccGHIJKLMNp0 = vfmaq_f16(vaccGHIJKLMNp0, vi13xGHIJKLMN, vk13xGHIJKLMN);
      vaccOPQRSTUVp0 = vfmaq_f16(vaccOPQRSTUVp0, vi13xOPQRSTUV, vk13xOPQRSTUV);

      const float16x8_t vi14x01234567 = vld1q_f16(i14); i14 += 8;
      const float16x8_t vi14x89ABCDEF = vld1q_f16(i14); i14 += 8;
      const float16x8_t vi14xGHIJKLMN = vld1q_f16(i14); i14 += 8;
      const float16x8_t vi14xOPQRSTUV = vld1q_f16(i14); i14 += 8;
      const float16x8_t vk14x01234567 = vld1q_f16(w); w += 8;
      const float16x8_t vk14x89ABCDEF = vld1q_f16(w); w += 8;
      const float16x8_t vk14xGHIJKLMN = vld1q_f16(w); w += 8;
      const float16x8_t vk14xOPQRSTUV = vld1q_f16(w); w += 8;
      vacc01234567p0 = vfmaq_f16(vacc01234567p0, vi14x01234567, vk14x01234567);
      vacc89ABCDEFp0 = vfmaq_f16(vacc89ABCDEFp0, vi14x89ABCDEF, vk14x89ABCDEF);
      vaccGHIJKLMNp0 = vfmaq_f16(vaccGHIJKLMNp0, vi14xGHIJKLMN, vk14xGHIJKLMN);
      vaccOPQRSTUVp0 = vfmaq_f16(vaccOPQRSTUVp0, vi14xOPQRSTUV, vk14xOPQRSTUV);

      const float16x8_t vi15x01234567 = vld1q_f16(i15); i15 += 8;
      const float16x8_t vi15x89ABCDEF = vld1q_f16(i15); i15 += 8;
      const float16x8_t vi15xGHIJKLMN = vld1q_f16(i15); i15 += 8;
      const float16x8_t vi15xOPQRSTUV = vld1q_f16(i15); i15 += 8;
      const float16x8_t vk15x01234567 = vld1q_f16(w); w += 8;
      const float16x8_t vk15x89ABCDEF = vld1q_f16(w); w += 8;
      const float16x8_t vk15xGHIJKLMN = vld1q_f16(w); w += 8;
      const float16x8_t vk15xOPQRSTUV = vld1q_f16(w); w += 8;
      vacc01234567p0 = vfmaq_f16(vacc01234567p0, vi15x01234567, vk15x01234567);
      vacc89ABCDEFp0 = vfmaq_f16(vacc89ABCDEFp0, vi15x89ABCDEF, vk15x89ABCDEF);
      vaccGHIJKLMNp0 = vfmaq_f16(vaccGHIJKLMNp0, vi15xGHIJKLMN, vk15xGHIJKLMN);
      vaccOPQRSTUVp0 = vfmaq_f16(vaccOPQRSTUVp0, vi15xOPQRSTUV, vk15xOPQRSTUV);

      const float16x8_t vi16x01234567 = vld1q_f16(i16); i16 += 8;
      const float16x8_t vi16x89ABCDEF = vld1q_f16(i16); i16 += 8;
      const float16x8_t vi16xGHIJKLMN = vld1q_f16(i16); i16 += 8;
      const float16x8_t vi16xOPQRSTUV = vld1q_f16(i16); i16 += 8;
      const float16x8_t vk16x01234567 = vld1q_f16(w); w += 8;
      const float16x8_t vk16x89ABCDEF = vld1q_f16(w); w += 8;
      const float16x8_t vk16xGHIJKLMN = vld1q_f16(w); w += 8;
      const float16x8_t vk16xOPQRSTUV = vld1q_f16(w); w += 8;
      vacc01234567p0 = vfmaq_f16(vacc01234567p0, vi16x01234567, vk16x01234567);
      vacc89ABCDEFp0 = vfmaq_f16(vacc89ABCDEFp0, vi16x89ABCDEF, vk16x89ABCDEF);
      vaccGHIJKLMNp0 = vfmaq_f16(vaccGHIJKLMNp0, vi16xGHIJKLMN, vk16xGHIJKLMN);
      vaccOPQRSTUVp0 = vfmaq_f16(vaccOPQRSTUVp0, vi16xOPQRSTUV, vk16xOPQRSTUV);

      const float16x8_t vi17x01234567 = vld1q_f16(i17); i17 += 8;
      const float16x8_t vi17x89ABCDEF = vld1q_f16(i17); i17 += 8;
      const float16x8_t vi17xGHIJKLMN = vld1q_f16(i17); i17 += 8;
      const float16x8_t vi17xOPQRSTUV = vld1q_f16(i17); i17 += 8;
      const float16x8_t vk17x01234567 = vld1q_f16(w); w += 8;
      const float16x8_t vk17x89ABCDEF = vld1q_f16(w); w += 8;
      const float16x8_t vk17xGHIJKLMN = vld1q_f16(w); w += 8;
      const float16x8_t vk17xOPQRSTUV = vld1q_f16(w); w += 8;
      vacc01234567p0 = vfmaq_f16(vacc01234567p0, vi17x01234567, vk17x01234567);
      vacc89ABCDEFp0 = vfmaq_f16(vacc89ABCDEFp0, vi17x89ABCDEF, vk17x89ABCDEF);
      vaccGHIJKLMNp0 = vfmaq_f16(vaccGHIJKLMNp0, vi17xGHIJKLMN, vk17xGHIJKLMN);
      vaccOPQRSTUVp0 = vfmaq_f16(vaccOPQRSTUVp0, vi17xOPQRSTUV, vk17xOPQRSTUV);

      const float16x8_t vi18x01234567 = vld1q_f16(i18); i18 += 8;
      const float16x8_t vi18x89ABCDEF = vld1q_f16(i18); i18 += 8;
      const float16x8_t vi18xGHIJKLMN = vld1q_f16(i18); i18 += 8;
      const float16x8_t vi18xOPQRSTUV = vld1q_f16(i18); i18 += 8;
      const float16x8_t vk18x01234567 = vld1q_f16(w); w += 8;
      const float16x8_t vk18x89ABCDEF = vld1q_f16(w); w += 8;
      const float16x8_t vk18xGHIJKLMN = vld1q_f16(w); w += 8;
      const float16x8_t vk18xOPQRSTUV = vld1q_f16(w); w += 8;
      vacc01234567p0 = vfmaq_f16(vacc01234567p0, vi18x01234567, vk18x01234567);
      vacc89ABCDEFp0 = vfmaq_f16(vacc89ABCDEFp0, vi18x89ABCDEF, vk18x89ABCDEF);
      vaccGHIJKLMNp0 = vfmaq_f16(vaccGHIJKLMNp0, vi18xGHIJKLMN, vk18xGHIJKLMN);
      vaccOPQRSTUVp0 = vfmaq_f16(vaccOPQRSTUVp0, vi18xOPQRSTUV, vk18xOPQRSTUV);

      const float16x8_t vi19x01234567 = vld1q_f16(i19); i19 += 8;
      const float16x8_t vi19x89ABCDEF = vld1q_f16(i19); i19 += 8;
      const float16x8_t vi19xGHIJKLMN = vld1q_f16(i19); i19 += 8;
      const float16x8_t vi19xOPQRSTUV = vld1q_f16(i19); i19 += 8;
      const float16x8_t vk19x01234567 = vld1q_f16(w); w += 8;
      const float16x8_t vk19x89ABCDEF = vld1q_f16(w); w += 8;
      const float16x8_t vk19xGHIJKLMN = vld1q_f16(w); w += 8;
      const float16x8_t vk19xOPQRSTUV = vld1q_f16(w); w += 8;
      vacc01234567p0 = vfmaq_f16(vacc01234567p0, vi19x01234567, vk19x01234567);
      vacc89ABCDEFp0 = vfmaq_f16(vacc89ABCDEFp0, vi19x89ABCDEF, vk19x89ABCDEF);
      vaccGHIJKLMNp0 = vfmaq_f16(vaccGHIJKLMNp0, vi19xGHIJKLMN, vk19xGHIJKLMN);
      vaccOPQRSTUVp0 = vfmaq_f16(vaccOPQRSTUVp0, vi19xOPQRSTUV, vk19xOPQRSTUV);

      const float16x8_t vi20x01234567 = vld1q_f16(i20); i20 += 8;
      const float16x8_t vi20x89ABCDEF = vld1q_f16(i20); i20 += 8;
      const float16x8_t vi20xGHIJKLMN = vld1q_f16(i20); i20 += 8;
      const float16x8_t vi20xOPQRSTUV = vld1q_f16(i20); i20 += 8;
      const float16x8_t vk20x01234567 = vld1q_f16(w); w += 8;
      const float16x8_t vk20x89ABCDEF = vld1q_f16(w); w += 8;
      const float16x8_t vk20xGHIJKLMN = vld1q_f16(w); w += 8;
      const float16x8_t vk20xOPQRSTUV = vld1q_f16(w); w += 8;
      vacc01234567p0 = vfmaq_f16(vacc01234567p0, vi20x01234567, vk20x01234567);
      vacc89ABCDEFp0 = vfmaq_f16(vacc89ABCDEFp0, vi20x89ABCDEF, vk20x89ABCDEF);
      vaccGHIJKLMNp0 = vfmaq_f16(vaccGHIJKLMNp0, vi20xGHIJKLMN, vk20xGHIJKLMN);
      vaccOPQRSTUVp0 = vfmaq_f16(vaccOPQRSTUVp0, vi20xOPQRSTUV, vk20xOPQRSTUV);

      const float16x8_t vi21x01234567 = vld1q_f16(i21); i21 += 8;
      const float16x8_t vi21x89ABCDEF = vld1q_f16(i21); i21 += 8;
      const float16x8_t vi21xGHIJKLMN = vld1q_f16(i21); i21 += 8;
      const float16x8_t vi21xOPQRSTUV = vld1q_f16(i21); i21 += 8;
      const float16x8_t vk21x01234567 = vld1q_f16(w); w += 8;
      const float16x8_t vk21x89ABCDEF = vld1q_f16(w); w += 8;
      const float16x8_t vk21xGHIJKLMN = vld1q_f16(w); w += 8;
      const float16x8_t vk21xOPQRSTUV = vld1q_f16(w); w += 8;
      vacc01234567p0 = vfmaq_f16(vacc01234567p0, vi21x01234567, vk21x01234567);
      vacc89ABCDEFp0 = vfmaq_f16(vacc89ABCDEFp0, vi21x89ABCDEF, vk21x89ABCDEF);
      vaccGHIJKLMNp0 = vfmaq_f16(vaccGHIJKLMNp0, vi21xGHIJKLMN, vk21xGHIJKLMN);
      vaccOPQRSTUVp0 = vfmaq_f16(vaccOPQRSTUVp0, vi21xOPQRSTUV, vk21xOPQRSTUV);

      const float16x8_t vi22x01234567 = vld1q_f16(i22); i22 += 8;
      const float16x8_t vi22x89ABCDEF = vld1q_f16(i22); i22 += 8;
      const float16x8_t vi22xGHIJKLMN = vld1q_f16(i22); i22 += 8;
      const float16x8_t vi22xOPQRSTUV = vld1q_f16(i22); i22 += 8;
      const float16x8_t vk22x01234567 = vld1q_f16(w); w += 8;
      const float16x8_t vk22x89ABCDEF = vld1q_f16(w); w += 8;
      const float16x8_t vk22xGHIJKLMN = vld1q_f16(w); w += 8;
      const float16x8_t vk22xOPQRSTUV = vld1q_f16(w); w += 8;
      vacc01234567p0 = vfmaq_f16(vacc01234567p0, vi22x01234567, vk22x01234567);
      vacc89ABCDEFp0 = vfmaq_f16(vacc89ABCDEFp0, vi22x89ABCDEF, vk22x89ABCDEF);
      vaccGHIJKLMNp0 = vfmaq_f16(vaccGHIJKLMNp0, vi22xGHIJKLMN, vk22xGHIJKLMN);
      vaccOPQRSTUVp0 = vfmaq_f16(vaccOPQRSTUVp0, vi22xOPQRSTUV, vk22xOPQRSTUV);

      const float16x8_t vi23x01234567 = vld1q_f16(i23); i23 += 8;
      const float16x8_t vi23x89ABCDEF = vld1q_f16(i23); i23 += 8;
      const float16x8_t vi23xGHIJKLMN = vld1q_f16(i23); i23 += 8;
      const float16x8_t vi23xOPQRSTUV = vld1q_f16(i23); i23 += 8;
      const float16x8_t vk23x01234567 = vld1q_f16(w); w += 8;
      const float16x8_t vk23x89ABCDEF = vld1q_f16(w); w += 8;
      const float16x8_t vk23xGHIJKLMN = vld1q_f16(w); w += 8;
      const float16x8_t vk23xOPQRSTUV = vld1q_f16(w); w += 8;
      vacc01234567p0 = vfmaq_f16(vacc01234567p0, vi23x01234567, vk23x01234567);
      vacc89ABCDEFp0 = vfmaq_f16(vacc89ABCDEFp0, vi23x89ABCDEF, vk23x89ABCDEF);
      vaccGHIJKLMNp0 = vfmaq_f16(vaccGHIJKLMNp0, vi23xGHIJKLMN, vk23xGHIJKLMN);
      vaccOPQRSTUVp0 = vfmaq_f16(vaccOPQRSTUVp0, vi23xOPQRSTUV, vk23xOPQRSTUV);

      const float16x8_t vi24x01234567 = vld1q_f16(i24); i24 += 8;
      const float16x8_t vi24x89ABCDEF = vld1q_f16(i24); i24 += 8;
      const float16x8_t vi24xGHIJKLMN = vld1q_f16(i24); i24 += 8;
      const float16x8_t vi24xOPQRSTUV = vld1q_f16(i24); i24 += 8;
      const float16x8_t vk24x01234567 = vld1q_f16(w); w += 8;
      const float16x8_t vk24x89ABCDEF = vld1q_f16(w); w += 8;
      const float16x8_t vk24xGHIJKLMN = vld1q_f16(w); w += 8;
      const float16x8_t vk24xOPQRSTUV = vld1q_f16(w); w += 8;
      vacc01234567p0 = vfmaq_f16(vacc01234567p0, vi24x01234567, vk24x01234567);
      vacc89ABCDEFp0 = vfmaq_f16(vacc89ABCDEFp0, vi24x89ABCDEF, vk24x89ABCDEF);
      vaccGHIJKLMNp0 = vfmaq_f16(vaccGHIJKLMNp0, vi24xGHIJKLMN, vk24xGHIJKLMN);
      vaccOPQRSTUVp0 = vfmaq_f16(vaccOPQRSTUVp0, vi24xOPQRSTUV, vk24xOPQRSTUV);


      float16x8_t vacc01234567 = vmaxq_f16(vacc01234567p0, vmin);
      float16x8_t vacc89ABCDEF = vmaxq_f16(vacc89ABCDEFp0, vmin);
      float16x8_t vaccGHIJKLMN = vmaxq_f16(vaccGHIJKLMNp0, vmin);
      float16x8_t vaccOPQRSTUV = vmaxq_f16(vaccOPQRSTUVp0, vmin);
      vacc01234567 = vminq_f16(vacc01234567, vmax);
      vacc89ABCDEF = vminq_f16(vacc89ABCDEF, vmax);
      vaccGHIJKLMN = vminq_f16(vaccGHIJKLMN, vmax);
      vaccOPQRSTUV = vminq_f16(vaccOPQRSTUV, vmax);

      vst1q_f16(output, vacc01234567); output += 8;
      vst1q_f16(output, vacc89ABCDEF); output += 8;
      vst1q_f16(output, vaccGHIJKLMN); output += 8;
      vst1q_f16(output, vaccOPQRSTUV); output += 8;
    }
    for (; c >= 8; c -= 8) {
      float16x8_t vacc01234567p0 = vld1q_f16(w); w += 8;


      const float16x8_t vi0x01234567 = vld1q_f16(i0); i0 += 8;
      const float16x8_t vk0x01234567 = vld1q_f16(w + 24);
      vacc01234567p0 = vfmaq_f16(vacc01234567p0, vi0x01234567, vk0x01234567);

      const float16x8_t vi1x01234567 = vld1q_f16(i1); i1 += 8;
      const float16x8_t vk1x01234567 = vld1q_f16(w + 56);
      vacc01234567p0 = vfmaq_f16(vacc01234567p0, vi1x01234567, vk1x01234567);

      const float16x8_t vi2x01234567 = vld1q_f16(i2); i2 += 8;
      const float16x8_t vk2x01234567 = vld1q_f16(w + 88);
      vacc01234567p0 = vfmaq_f16(vacc01234567p0, vi2x01234567, vk2x01234567);

      const float16x8_t vi3x01234567 = vld1q_f16(i3); i3 += 8;
      const float16x8_t vk3x01234567 = vld1q_f16(w + 120);
      vacc01234567p0 = vfmaq_f16(vacc01234567p0, vi3x01234567, vk3x01234567);

      const float16x8_t vi4x01234567 = vld1q_f16(i4); i4 += 8;
      const float16x8_t vk4x01234567 = vld1q_f16(w + 152);
      vacc01234567p0 = vfmaq_f16(vacc01234567p0, vi4x01234567, vk4x01234567);

      const float16x8_t vi5x01234567 = vld1q_f16(i5); i5 += 8;
      const float16x8_t vk5x01234567 = vld1q_f16(w + 184);
      vacc01234567p0 = vfmaq_f16(vacc01234567p0, vi5x01234567, vk5x01234567);

      const float16x8_t vi6x01234567 = vld1q_f16(i6); i6 += 8;
      const float16x8_t vk6x01234567 = vld1q_f16(w + 216);
      vacc01234567p0 = vfmaq_f16(vacc01234567p0, vi6x01234567, vk6x01234567);

      const float16x8_t vi7x01234567 = vld1q_f16(i7); i7 += 8;
      const float16x8_t vk7x01234567 = vld1q_f16(w + 248);
      vacc01234567p0 = vfmaq_f16(vacc01234567p0, vi7x01234567, vk7x01234567);

      const float16x8_t vi8x01234567 = vld1q_f16(i8); i8 += 8;
      const float16x8_t vk8x01234567 = vld1q_f16(w + 280);
      vacc01234567p0 = vfmaq_f16(vacc01234567p0, vi8x01234567, vk8x01234567);

      const float16x8_t vi9x01234567 = vld1q_f16(i9); i9 += 8;
      const float16x8_t vk9x01234567 = vld1q_f16(w + 312);
      vacc01234567p0 = vfmaq_f16(vacc01234567p0, vi9x01234567, vk9x01234567);

      const float16x8_t vi10x01234567 = vld1q_f16(i10); i10 += 8;
      const float16x8_t vk10x01234567 = vld1q_f16(w + 344);
      vacc01234567p0 = vfmaq_f16(vacc01234567p0, vi10x01234567, vk10x01234567);

      const float16x8_t vi11x01234567 = vld1q_f16(i11); i11 += 8;
      const float16x8_t vk11x01234567 = vld1q_f16(w + 376);
      vacc01234567p0 = vfmaq_f16(vacc01234567p0, vi11x01234567, vk11x01234567);

      const float16x8_t vi12x01234567 = vld1q_f16(i12); i12 += 8;
      const float16x8_t vk12x01234567 = vld1q_f16(w + 408);
      vacc01234567p0 = vfmaq_f16(vacc01234567p0, vi12x01234567, vk12x01234567);

      const float16x8_t vi13x01234567 = vld1q_f16(i13); i13 += 8;
      const float16x8_t vk13x01234567 = vld1q_f16(w + 440);
      vacc01234567p0 = vfmaq_f16(vacc01234567p0, vi13x01234567, vk13x01234567);

      const float16x8_t vi14x01234567 = vld1q_f16(i14); i14 += 8;
      const float16x8_t vk14x01234567 = vld1q_f16(w + 472);
      vacc01234567p0 = vfmaq_f16(vacc01234567p0, vi14x01234567, vk14x01234567);

      const float16x8_t vi15x01234567 = vld1q_f16(i15); i15 += 8;
      const float16x8_t vk15x01234567 = vld1q_f16(w + 504);
      vacc01234567p0 = vfmaq_f16(vacc01234567p0, vi15x01234567, vk15x01234567);

      const float16x8_t vi16x01234567 = vld1q_f16(i16); i16 += 8;
      const float16x8_t vk16x01234567 = vld1q_f16(w + 536);
      vacc01234567p0 = vfmaq_f16(vacc01234567p0, vi16x01234567, vk16x01234567);

      const float16x8_t vi17x01234567 = vld1q_f16(i17); i17 += 8;
      const float16x8_t vk17x01234567 = vld1q_f16(w + 568);
      vacc01234567p0 = vfmaq_f16(vacc01234567p0, vi17x01234567, vk17x01234567);

      const float16x8_t vi18x01234567 = vld1q_f16(i18); i18 += 8;
      const float16x8_t vk18x01234567 = vld1q_f16(w + 600);
      vacc01234567p0 = vfmaq_f16(vacc01234567p0, vi18x01234567, vk18x01234567);

      const float16x8_t vi19x01234567 = vld1q_f16(i19); i19 += 8;
      const float16x8_t vk19x01234567 = vld1q_f16(w + 632);
      vacc01234567p0 = vfmaq_f16(vacc01234567p0, vi19x01234567, vk19x01234567);

      const float16x8_t vi20x01234567 = vld1q_f16(i20); i20 += 8;
      const float16x8_t vk20x01234567 = vld1q_f16(w + 664);
      vacc01234567p0 = vfmaq_f16(vacc01234567p0, vi20x01234567, vk20x01234567);

      const float16x8_t vi21x01234567 = vld1q_f16(i21); i21 += 8;
      const float16x8_t vk21x01234567 = vld1q_f16(w + 696);
      vacc01234567p0 = vfmaq_f16(vacc01234567p0, vi21x01234567, vk21x01234567);

      const float16x8_t vi22x01234567 = vld1q_f16(i22); i22 += 8;
      const float16x8_t vk22x01234567 = vld1q_f16(w + 728);
      vacc01234567p0 = vfmaq_f16(vacc01234567p0, vi22x01234567, vk22x01234567);

      const float16x8_t vi23x01234567 = vld1q_f16(i23); i23 += 8;
      const float16x8_t vk23x01234567 = vld1q_f16(w + 760);
      vacc01234567p0 = vfmaq_f16(vacc01234567p0, vi23x01234567, vk23x01234567);

      const float16x8_t vi24x01234567 = vld1q_f16(i24); i24 += 8;
      const float16x8_t vk24x01234567 = vld1q_f16(w + 792);
      vacc01234567p0 = vfmaq_f16(vacc01234567p0, vi24x01234567, vk24x01234567);


      float16x8_t vacc01234567 = vmaxq_f16(vacc01234567p0, vmin);
      vacc01234567 = vminq_f16(vacc01234567, vmax);

      vst1q_f16(output, vacc01234567); output += 8;
    }
    if XNN_UNLIKELY(c != 0) {
      float16x8_t vacc01234567p0 = vld1q_f16(w);


      const float16x8_t vi0x01234567 = vld1q_f16(i0);
      const float16x8_t vk0x01234567 = vld1q_f16(w + 32);
      vacc01234567p0 = vfmaq_f16(vacc01234567p0, vi0x01234567, vk0x01234567);

      const float16x8_t vi1x01234567 = vld1q_f16(i1);
      const float16x8_t vk1x01234567 = vld1q_f16(w + 64);
      vacc01234567p0 = vfmaq_f16(vacc01234567p0, vi1x01234567, vk1x01234567);

      const float16x8_t vi2x01234567 = vld1q_f16(i2);
      const float16x8_t vk2x01234567 = vld1q_f16(w + 96);
      vacc01234567p0 = vfmaq_f16(vacc01234567p0, vi2x01234567, vk2x01234567);

      const float16x8_t vi3x01234567 = vld1q_f16(i3);
      const float16x8_t vk3x01234567 = vld1q_f16(w + 128);
      vacc01234567p0 = vfmaq_f16(vacc01234567p0, vi3x01234567, vk3x01234567);

      const float16x8_t vi4x01234567 = vld1q_f16(i4);
      const float16x8_t vk4x01234567 = vld1q_f16(w + 160);
      vacc01234567p0 = vfmaq_f16(vacc01234567p0, vi4x01234567, vk4x01234567);

      const float16x8_t vi5x01234567 = vld1q_f16(i5);
      const float16x8_t vk5x01234567 = vld1q_f16(w + 192);
      vacc01234567p0 = vfmaq_f16(vacc01234567p0, vi5x01234567, vk5x01234567);

      const float16x8_t vi6x01234567 = vld1q_f16(i6);
      const float16x8_t vk6x01234567 = vld1q_f16(w + 224);
      vacc01234567p0 = vfmaq_f16(vacc01234567p0, vi6x01234567, vk6x01234567);

      const float16x8_t vi7x01234567 = vld1q_f16(i7);
      const float16x8_t vk7x01234567 = vld1q_f16(w + 256);
      vacc01234567p0 = vfmaq_f16(vacc01234567p0, vi7x01234567, vk7x01234567);

      const float16x8_t vi8x01234567 = vld1q_f16(i8);
      const float16x8_t vk8x01234567 = vld1q_f16(w + 288);
      vacc01234567p0 = vfmaq_f16(vacc01234567p0, vi8x01234567, vk8x01234567);

      const float16x8_t vi9x01234567 = vld1q_f16(i9);
      const float16x8_t vk9x01234567 = vld1q_f16(w + 320);
      vacc01234567p0 = vfmaq_f16(vacc01234567p0, vi9x01234567, vk9x01234567);

      const float16x8_t vi10x01234567 = vld1q_f16(i10);
      const float16x8_t vk10x01234567 = vld1q_f16(w + 352);
      vacc01234567p0 = vfmaq_f16(vacc01234567p0, vi10x01234567, vk10x01234567);

      const float16x8_t vi11x01234567 = vld1q_f16(i11);
      const float16x8_t vk11x01234567 = vld1q_f16(w + 384);
      vacc01234567p0 = vfmaq_f16(vacc01234567p0, vi11x01234567, vk11x01234567);

      const float16x8_t vi12x01234567 = vld1q_f16(i12);
      const float16x8_t vk12x01234567 = vld1q_f16(w + 416);
      vacc01234567p0 = vfmaq_f16(vacc01234567p0, vi12x01234567, vk12x01234567);

      const float16x8_t vi13x01234567 = vld1q_f16(i13);
      const float16x8_t vk13x01234567 = vld1q_f16(w + 448);
      vacc01234567p0 = vfmaq_f16(vacc01234567p0, vi13x01234567, vk13x01234567);

      const float16x8_t vi14x01234567 = vld1q_f16(i14);
      const float16x8_t vk14x01234567 = vld1q_f16(w + 480);
      vacc01234567p0 = vfmaq_f16(vacc01234567p0, vi14x01234567, vk14x01234567);

      const float16x8_t vi15x01234567 = vld1q_f16(i15);
      const float16x8_t vk15x01234567 = vld1q_f16(w + 512);
      vacc01234567p0 = vfmaq_f16(vacc01234567p0, vi15x01234567, vk15x01234567);

      const float16x8_t vi16x01234567 = vld1q_f16(i16);
      const float16x8_t vk16x01234567 = vld1q_f16(w + 544);
      vacc01234567p0 = vfmaq_f16(vacc01234567p0, vi16x01234567, vk16x01234567);

      const float16x8_t vi17x01234567 = vld1q_f16(i17);
      const float16x8_t vk17x01234567 = vld1q_f16(w + 576);
      vacc01234567p0 = vfmaq_f16(vacc01234567p0, vi17x01234567, vk17x01234567);

      const float16x8_t vi18x01234567 = vld1q_f16(i18);
      const float16x8_t vk18x01234567 = vld1q_f16(w + 608);
      vacc01234567p0 = vfmaq_f16(vacc01234567p0, vi18x01234567, vk18x01234567);

      const float16x8_t vi19x01234567 = vld1q_f16(i19);
      const float16x8_t vk19x01234567 = vld1q_f16(w + 640);
      vacc01234567p0 = vfmaq_f16(vacc01234567p0, vi19x01234567, vk19x01234567);

      const float16x8_t vi20x01234567 = vld1q_f16(i20);
      const float16x8_t vk20x01234567 = vld1q_f16(w + 672);
      vacc01234567p0 = vfmaq_f16(vacc01234567p0, vi20x01234567, vk20x01234567);

      const float16x8_t vi21x01234567 = vld1q_f16(i21);
      const float16x8_t vk21x01234567 = vld1q_f16(w + 704);
      vacc01234567p0 = vfmaq_f16(vacc01234567p0, vi21x01234567, vk21x01234567);

      const float16x8_t vi22x01234567 = vld1q_f16(i22);
      const float16x8_t vk22x01234567 = vld1q_f16(w + 736);
      vacc01234567p0 = vfmaq_f16(vacc01234567p0, vi22x01234567, vk22x01234567);

      const float16x8_t vi23x01234567 = vld1q_f16(i23);
      const float16x8_t vk23x01234567 = vld1q_f16(w + 768);
      vacc01234567p0 = vfmaq_f16(vacc01234567p0, vi23x01234567, vk23x01234567);

      const float16x8_t vi24x01234567 = vld1q_f16(i24);
      const float16x8_t vk24x01234567 = vld1q_f16(w + 800);
      vacc01234567p0 = vfmaq_f16(vacc01234567p0, vi24x01234567, vk24x01234567);


      float16x8_t vacc01234567 = vmaxq_f16(vacc01234567p0, vmin);
      vacc01234567 = vminq_f16(vacc01234567, vmax);

      float16x4_t vacc0123 = vget_low_f16(vacc01234567);
      if (c & 4) {
        vst1_f16(output, vacc0123); output += 4;
        vacc0123 = vget_high_f16(vacc01234567);
      }
      if (c & 2) {
        vst1_lane_u32((void*) output, vreinterpret_u32_f16(vacc0123), 0); output += 2;
        vacc0123 = vext_f16(vacc0123, vacc0123, 2);
      }
      if (c & 1) {
        vst1_lane_f16(output, vacc0123, 0); output += 1;
      }
    }

    output = (__fp16*) ((uintptr_t) output + output_increment);
  } while (--output_width != 0);
}
