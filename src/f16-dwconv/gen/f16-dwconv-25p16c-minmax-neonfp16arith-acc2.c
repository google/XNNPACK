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

#include "xnnpack/dwconv.h"


void xnn_f16_dwconv_minmax_ukernel_25p16c__neonfp16arith_acc2(
    size_t channels,
    size_t output_width,
    const void** input,
    const void* weights,
    void* output_ptr,
    intptr_t input_stride,
    size_t output_increment,
    size_t input_offset,
    const void* zero,
    const union xnn_f16_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(channels != 0);
  assert(output_width != 0);

  uint16_t* output = (uint16_t*) output_ptr;
  const float16x8_t vmin = vreinterpretq_f16_u16(vld1q_dup_u16(&params->fp16arith.min));
  const float16x8_t vmax = vreinterpretq_f16_u16(vld1q_dup_u16(&params->fp16arith.max));
  do {
    const uint16_t* i0 = (const uint16_t*) input[0];
    assert(i0 != NULL);
    if XNN_UNPREDICTABLE(i0 != (const uint16_t*) zero) {
      i0 = (const uint16_t*) ((uintptr_t) i0 + input_offset);
    }
    const uint16_t* i1 = (const uint16_t*) input[1];
    assert(i1 != NULL);
    if XNN_UNPREDICTABLE(i1 != (const uint16_t*) zero) {
      i1 = (const uint16_t*) ((uintptr_t) i1 + input_offset);
    }
    const uint16_t* i2 = (const uint16_t*) input[2];
    assert(i2 != NULL);
    if XNN_UNPREDICTABLE(i2 != (const uint16_t*) zero) {
      i2 = (const uint16_t*) ((uintptr_t) i2 + input_offset);
    }
    const uint16_t* i3 = (const uint16_t*) input[3];
    assert(i3 != NULL);
    if XNN_UNPREDICTABLE(i3 != (const uint16_t*) zero) {
      i3 = (const uint16_t*) ((uintptr_t) i3 + input_offset);
    }
    const uint16_t* i4 = (const uint16_t*) input[4];
    assert(i4 != NULL);
    if XNN_UNPREDICTABLE(i4 != (const uint16_t*) zero) {
      i4 = (const uint16_t*) ((uintptr_t) i4 + input_offset);
    }
    const uint16_t* i5 = (const uint16_t*) input[5];
    assert(i5 != NULL);
    if XNN_UNPREDICTABLE(i5 != (const uint16_t*) zero) {
      i5 = (const uint16_t*) ((uintptr_t) i5 + input_offset);
    }
    const uint16_t* i6 = (const uint16_t*) input[6];
    assert(i6 != NULL);
    if XNN_UNPREDICTABLE(i6 != (const uint16_t*) zero) {
      i6 = (const uint16_t*) ((uintptr_t) i6 + input_offset);
    }
    const uint16_t* i7 = (const uint16_t*) input[7];
    assert(i7 != NULL);
    if XNN_UNPREDICTABLE(i7 != (const uint16_t*) zero) {
      i7 = (const uint16_t*) ((uintptr_t) i7 + input_offset);
    }
    const uint16_t* i8 = (const uint16_t*) input[8];
    assert(i8 != NULL);
    if XNN_UNPREDICTABLE(i8 != (const uint16_t*) zero) {
      i8 = (const uint16_t*) ((uintptr_t) i8 + input_offset);
    }
    const uint16_t* i9 = (const uint16_t*) input[9];
    assert(i9 != NULL);
    if XNN_UNPREDICTABLE(i9 != (const uint16_t*) zero) {
      i9 = (const uint16_t*) ((uintptr_t) i9 + input_offset);
    }
    const uint16_t* i10 = (const uint16_t*) input[10];
    assert(i10 != NULL);
    if XNN_UNPREDICTABLE(i10 != (const uint16_t*) zero) {
      i10 = (const uint16_t*) ((uintptr_t) i10 + input_offset);
    }
    const uint16_t* i11 = (const uint16_t*) input[11];
    assert(i11 != NULL);
    if XNN_UNPREDICTABLE(i11 != (const uint16_t*) zero) {
      i11 = (const uint16_t*) ((uintptr_t) i11 + input_offset);
    }
    const uint16_t* i12 = (const uint16_t*) input[12];
    assert(i12 != NULL);
    if XNN_UNPREDICTABLE(i12 != (const uint16_t*) zero) {
      i12 = (const uint16_t*) ((uintptr_t) i12 + input_offset);
    }
    const uint16_t* i13 = (const uint16_t*) input[13];
    assert(i13 != NULL);
    if XNN_UNPREDICTABLE(i13 != (const uint16_t*) zero) {
      i13 = (const uint16_t*) ((uintptr_t) i13 + input_offset);
    }
    const uint16_t* i14 = (const uint16_t*) input[14];
    assert(i14 != NULL);
    if XNN_UNPREDICTABLE(i14 != (const uint16_t*) zero) {
      i14 = (const uint16_t*) ((uintptr_t) i14 + input_offset);
    }
    const uint16_t* i15 = (const uint16_t*) input[15];
    assert(i15 != NULL);
    if XNN_UNPREDICTABLE(i15 != (const uint16_t*) zero) {
      i15 = (const uint16_t*) ((uintptr_t) i15 + input_offset);
    }
    const uint16_t* i16 = (const uint16_t*) input[16];
    assert(i16 != NULL);
    if XNN_UNPREDICTABLE(i16 != (const uint16_t*) zero) {
      i16 = (const uint16_t*) ((uintptr_t) i16 + input_offset);
    }
    const uint16_t* i17 = (const uint16_t*) input[17];
    assert(i17 != NULL);
    if XNN_UNPREDICTABLE(i17 != (const uint16_t*) zero) {
      i17 = (const uint16_t*) ((uintptr_t) i17 + input_offset);
    }
    const uint16_t* i18 = (const uint16_t*) input[18];
    assert(i18 != NULL);
    if XNN_UNPREDICTABLE(i18 != (const uint16_t*) zero) {
      i18 = (const uint16_t*) ((uintptr_t) i18 + input_offset);
    }
    const uint16_t* i19 = (const uint16_t*) input[19];
    assert(i19 != NULL);
    if XNN_UNPREDICTABLE(i19 != (const uint16_t*) zero) {
      i19 = (const uint16_t*) ((uintptr_t) i19 + input_offset);
    }
    const uint16_t* i20 = (const uint16_t*) input[20];
    assert(i20 != NULL);
    if XNN_UNPREDICTABLE(i20 != (const uint16_t*) zero) {
      i20 = (const uint16_t*) ((uintptr_t) i20 + input_offset);
    }
    const uint16_t* i21 = (const uint16_t*) input[21];
    assert(i21 != NULL);
    if XNN_UNPREDICTABLE(i21 != (const uint16_t*) zero) {
      i21 = (const uint16_t*) ((uintptr_t) i21 + input_offset);
    }
    const uint16_t* i22 = (const uint16_t*) input[22];
    assert(i22 != NULL);
    if XNN_UNPREDICTABLE(i22 != (const uint16_t*) zero) {
      i22 = (const uint16_t*) ((uintptr_t) i22 + input_offset);
    }
    const uint16_t* i23 = (const uint16_t*) input[23];
    assert(i23 != NULL);
    if XNN_UNPREDICTABLE(i23 != (const uint16_t*) zero) {
      i23 = (const uint16_t*) ((uintptr_t) i23 + input_offset);
    }
    const uint16_t* i24 = (const uint16_t*) input[24];
    assert(i24 != NULL);
    if XNN_UNPREDICTABLE(i24 != (const uint16_t*) zero) {
      i24 = (const uint16_t*) ((uintptr_t) i24 + input_offset);
    }

    input = (const void**) ((uintptr_t) input + input_stride);

    size_t c = channels;
    const uint16_t* w = (const uint16_t*) weights;
    for (; c >= 16; c -= 16) {
      float16x8_t vacc01234567p0 = vreinterpretq_f16_u16(vld1q_u16(w)); w += 8;
      float16x8_t vacc89ABCDEFp0 = vreinterpretq_f16_u16(vld1q_u16(w)); w += 8;


      const float16x8_t vi0x01234567 = vreinterpretq_f16_u16(vld1q_u16(i0)); i0 += 8;
      const float16x8_t vi0x89ABCDEF = vreinterpretq_f16_u16(vld1q_u16(i0)); i0 += 8;
      const float16x8_t vk0x01234567 = vreinterpretq_f16_u16(vld1q_u16(w)); w += 8;
      const float16x8_t vk0x89ABCDEF = vreinterpretq_f16_u16(vld1q_u16(w)); w += 8;
      vacc01234567p0 = vfmaq_f16(vacc01234567p0, vi0x01234567, vk0x01234567);
      vacc89ABCDEFp0 = vfmaq_f16(vacc89ABCDEFp0, vi0x89ABCDEF, vk0x89ABCDEF);

      const float16x8_t vi1x01234567 = vreinterpretq_f16_u16(vld1q_u16(i1)); i1 += 8;
      const float16x8_t vi1x89ABCDEF = vreinterpretq_f16_u16(vld1q_u16(i1)); i1 += 8;
      const float16x8_t vk1x01234567 = vreinterpretq_f16_u16(vld1q_u16(w)); w += 8;
      const float16x8_t vk1x89ABCDEF = vreinterpretq_f16_u16(vld1q_u16(w)); w += 8;
      float16x8_t vacc01234567p1 = vmulq_f16(vi1x01234567, vk1x01234567);
      float16x8_t vacc89ABCDEFp1 = vmulq_f16(vi1x89ABCDEF, vk1x89ABCDEF);

      const float16x8_t vi2x01234567 = vreinterpretq_f16_u16(vld1q_u16(i2)); i2 += 8;
      const float16x8_t vi2x89ABCDEF = vreinterpretq_f16_u16(vld1q_u16(i2)); i2 += 8;
      const float16x8_t vk2x01234567 = vreinterpretq_f16_u16(vld1q_u16(w)); w += 8;
      const float16x8_t vk2x89ABCDEF = vreinterpretq_f16_u16(vld1q_u16(w)); w += 8;
      vacc01234567p0 = vfmaq_f16(vacc01234567p0, vi2x01234567, vk2x01234567);
      vacc89ABCDEFp0 = vfmaq_f16(vacc89ABCDEFp0, vi2x89ABCDEF, vk2x89ABCDEF);

      const float16x8_t vi3x01234567 = vreinterpretq_f16_u16(vld1q_u16(i3)); i3 += 8;
      const float16x8_t vi3x89ABCDEF = vreinterpretq_f16_u16(vld1q_u16(i3)); i3 += 8;
      const float16x8_t vk3x01234567 = vreinterpretq_f16_u16(vld1q_u16(w)); w += 8;
      const float16x8_t vk3x89ABCDEF = vreinterpretq_f16_u16(vld1q_u16(w)); w += 8;
      vacc01234567p1 = vfmaq_f16(vacc01234567p1, vi3x01234567, vk3x01234567);
      vacc89ABCDEFp1 = vfmaq_f16(vacc89ABCDEFp1, vi3x89ABCDEF, vk3x89ABCDEF);

      const float16x8_t vi4x01234567 = vreinterpretq_f16_u16(vld1q_u16(i4)); i4 += 8;
      const float16x8_t vi4x89ABCDEF = vreinterpretq_f16_u16(vld1q_u16(i4)); i4 += 8;
      const float16x8_t vk4x01234567 = vreinterpretq_f16_u16(vld1q_u16(w)); w += 8;
      const float16x8_t vk4x89ABCDEF = vreinterpretq_f16_u16(vld1q_u16(w)); w += 8;
      vacc01234567p0 = vfmaq_f16(vacc01234567p0, vi4x01234567, vk4x01234567);
      vacc89ABCDEFp0 = vfmaq_f16(vacc89ABCDEFp0, vi4x89ABCDEF, vk4x89ABCDEF);

      const float16x8_t vi5x01234567 = vreinterpretq_f16_u16(vld1q_u16(i5)); i5 += 8;
      const float16x8_t vi5x89ABCDEF = vreinterpretq_f16_u16(vld1q_u16(i5)); i5 += 8;
      const float16x8_t vk5x01234567 = vreinterpretq_f16_u16(vld1q_u16(w)); w += 8;
      const float16x8_t vk5x89ABCDEF = vreinterpretq_f16_u16(vld1q_u16(w)); w += 8;
      vacc01234567p1 = vfmaq_f16(vacc01234567p1, vi5x01234567, vk5x01234567);
      vacc89ABCDEFp1 = vfmaq_f16(vacc89ABCDEFp1, vi5x89ABCDEF, vk5x89ABCDEF);

      const float16x8_t vi6x01234567 = vreinterpretq_f16_u16(vld1q_u16(i6)); i6 += 8;
      const float16x8_t vi6x89ABCDEF = vreinterpretq_f16_u16(vld1q_u16(i6)); i6 += 8;
      const float16x8_t vk6x01234567 = vreinterpretq_f16_u16(vld1q_u16(w)); w += 8;
      const float16x8_t vk6x89ABCDEF = vreinterpretq_f16_u16(vld1q_u16(w)); w += 8;
      vacc01234567p0 = vfmaq_f16(vacc01234567p0, vi6x01234567, vk6x01234567);
      vacc89ABCDEFp0 = vfmaq_f16(vacc89ABCDEFp0, vi6x89ABCDEF, vk6x89ABCDEF);

      const float16x8_t vi7x01234567 = vreinterpretq_f16_u16(vld1q_u16(i7)); i7 += 8;
      const float16x8_t vi7x89ABCDEF = vreinterpretq_f16_u16(vld1q_u16(i7)); i7 += 8;
      const float16x8_t vk7x01234567 = vreinterpretq_f16_u16(vld1q_u16(w)); w += 8;
      const float16x8_t vk7x89ABCDEF = vreinterpretq_f16_u16(vld1q_u16(w)); w += 8;
      vacc01234567p1 = vfmaq_f16(vacc01234567p1, vi7x01234567, vk7x01234567);
      vacc89ABCDEFp1 = vfmaq_f16(vacc89ABCDEFp1, vi7x89ABCDEF, vk7x89ABCDEF);

      const float16x8_t vi8x01234567 = vreinterpretq_f16_u16(vld1q_u16(i8)); i8 += 8;
      const float16x8_t vi8x89ABCDEF = vreinterpretq_f16_u16(vld1q_u16(i8)); i8 += 8;
      const float16x8_t vk8x01234567 = vreinterpretq_f16_u16(vld1q_u16(w)); w += 8;
      const float16x8_t vk8x89ABCDEF = vreinterpretq_f16_u16(vld1q_u16(w)); w += 8;
      vacc01234567p0 = vfmaq_f16(vacc01234567p0, vi8x01234567, vk8x01234567);
      vacc89ABCDEFp0 = vfmaq_f16(vacc89ABCDEFp0, vi8x89ABCDEF, vk8x89ABCDEF);

      const float16x8_t vi9x01234567 = vreinterpretq_f16_u16(vld1q_u16(i9)); i9 += 8;
      const float16x8_t vi9x89ABCDEF = vreinterpretq_f16_u16(vld1q_u16(i9)); i9 += 8;
      const float16x8_t vk9x01234567 = vreinterpretq_f16_u16(vld1q_u16(w)); w += 8;
      const float16x8_t vk9x89ABCDEF = vreinterpretq_f16_u16(vld1q_u16(w)); w += 8;
      vacc01234567p1 = vfmaq_f16(vacc01234567p1, vi9x01234567, vk9x01234567);
      vacc89ABCDEFp1 = vfmaq_f16(vacc89ABCDEFp1, vi9x89ABCDEF, vk9x89ABCDEF);

      const float16x8_t vi10x01234567 = vreinterpretq_f16_u16(vld1q_u16(i10)); i10 += 8;
      const float16x8_t vi10x89ABCDEF = vreinterpretq_f16_u16(vld1q_u16(i10)); i10 += 8;
      const float16x8_t vk10x01234567 = vreinterpretq_f16_u16(vld1q_u16(w)); w += 8;
      const float16x8_t vk10x89ABCDEF = vreinterpretq_f16_u16(vld1q_u16(w)); w += 8;
      vacc01234567p0 = vfmaq_f16(vacc01234567p0, vi10x01234567, vk10x01234567);
      vacc89ABCDEFp0 = vfmaq_f16(vacc89ABCDEFp0, vi10x89ABCDEF, vk10x89ABCDEF);

      const float16x8_t vi11x01234567 = vreinterpretq_f16_u16(vld1q_u16(i11)); i11 += 8;
      const float16x8_t vi11x89ABCDEF = vreinterpretq_f16_u16(vld1q_u16(i11)); i11 += 8;
      const float16x8_t vk11x01234567 = vreinterpretq_f16_u16(vld1q_u16(w)); w += 8;
      const float16x8_t vk11x89ABCDEF = vreinterpretq_f16_u16(vld1q_u16(w)); w += 8;
      vacc01234567p1 = vfmaq_f16(vacc01234567p1, vi11x01234567, vk11x01234567);
      vacc89ABCDEFp1 = vfmaq_f16(vacc89ABCDEFp1, vi11x89ABCDEF, vk11x89ABCDEF);

      const float16x8_t vi12x01234567 = vreinterpretq_f16_u16(vld1q_u16(i12)); i12 += 8;
      const float16x8_t vi12x89ABCDEF = vreinterpretq_f16_u16(vld1q_u16(i12)); i12 += 8;
      const float16x8_t vk12x01234567 = vreinterpretq_f16_u16(vld1q_u16(w)); w += 8;
      const float16x8_t vk12x89ABCDEF = vreinterpretq_f16_u16(vld1q_u16(w)); w += 8;
      vacc01234567p0 = vfmaq_f16(vacc01234567p0, vi12x01234567, vk12x01234567);
      vacc89ABCDEFp0 = vfmaq_f16(vacc89ABCDEFp0, vi12x89ABCDEF, vk12x89ABCDEF);

      const float16x8_t vi13x01234567 = vreinterpretq_f16_u16(vld1q_u16(i13)); i13 += 8;
      const float16x8_t vi13x89ABCDEF = vreinterpretq_f16_u16(vld1q_u16(i13)); i13 += 8;
      const float16x8_t vk13x01234567 = vreinterpretq_f16_u16(vld1q_u16(w)); w += 8;
      const float16x8_t vk13x89ABCDEF = vreinterpretq_f16_u16(vld1q_u16(w)); w += 8;
      vacc01234567p1 = vfmaq_f16(vacc01234567p1, vi13x01234567, vk13x01234567);
      vacc89ABCDEFp1 = vfmaq_f16(vacc89ABCDEFp1, vi13x89ABCDEF, vk13x89ABCDEF);

      const float16x8_t vi14x01234567 = vreinterpretq_f16_u16(vld1q_u16(i14)); i14 += 8;
      const float16x8_t vi14x89ABCDEF = vreinterpretq_f16_u16(vld1q_u16(i14)); i14 += 8;
      const float16x8_t vk14x01234567 = vreinterpretq_f16_u16(vld1q_u16(w)); w += 8;
      const float16x8_t vk14x89ABCDEF = vreinterpretq_f16_u16(vld1q_u16(w)); w += 8;
      vacc01234567p0 = vfmaq_f16(vacc01234567p0, vi14x01234567, vk14x01234567);
      vacc89ABCDEFp0 = vfmaq_f16(vacc89ABCDEFp0, vi14x89ABCDEF, vk14x89ABCDEF);

      const float16x8_t vi15x01234567 = vreinterpretq_f16_u16(vld1q_u16(i15)); i15 += 8;
      const float16x8_t vi15x89ABCDEF = vreinterpretq_f16_u16(vld1q_u16(i15)); i15 += 8;
      const float16x8_t vk15x01234567 = vreinterpretq_f16_u16(vld1q_u16(w)); w += 8;
      const float16x8_t vk15x89ABCDEF = vreinterpretq_f16_u16(vld1q_u16(w)); w += 8;
      vacc01234567p1 = vfmaq_f16(vacc01234567p1, vi15x01234567, vk15x01234567);
      vacc89ABCDEFp1 = vfmaq_f16(vacc89ABCDEFp1, vi15x89ABCDEF, vk15x89ABCDEF);

      const float16x8_t vi16x01234567 = vreinterpretq_f16_u16(vld1q_u16(i16)); i16 += 8;
      const float16x8_t vi16x89ABCDEF = vreinterpretq_f16_u16(vld1q_u16(i16)); i16 += 8;
      const float16x8_t vk16x01234567 = vreinterpretq_f16_u16(vld1q_u16(w)); w += 8;
      const float16x8_t vk16x89ABCDEF = vreinterpretq_f16_u16(vld1q_u16(w)); w += 8;
      vacc01234567p0 = vfmaq_f16(vacc01234567p0, vi16x01234567, vk16x01234567);
      vacc89ABCDEFp0 = vfmaq_f16(vacc89ABCDEFp0, vi16x89ABCDEF, vk16x89ABCDEF);

      const float16x8_t vi17x01234567 = vreinterpretq_f16_u16(vld1q_u16(i17)); i17 += 8;
      const float16x8_t vi17x89ABCDEF = vreinterpretq_f16_u16(vld1q_u16(i17)); i17 += 8;
      const float16x8_t vk17x01234567 = vreinterpretq_f16_u16(vld1q_u16(w)); w += 8;
      const float16x8_t vk17x89ABCDEF = vreinterpretq_f16_u16(vld1q_u16(w)); w += 8;
      vacc01234567p1 = vfmaq_f16(vacc01234567p1, vi17x01234567, vk17x01234567);
      vacc89ABCDEFp1 = vfmaq_f16(vacc89ABCDEFp1, vi17x89ABCDEF, vk17x89ABCDEF);

      const float16x8_t vi18x01234567 = vreinterpretq_f16_u16(vld1q_u16(i18)); i18 += 8;
      const float16x8_t vi18x89ABCDEF = vreinterpretq_f16_u16(vld1q_u16(i18)); i18 += 8;
      const float16x8_t vk18x01234567 = vreinterpretq_f16_u16(vld1q_u16(w)); w += 8;
      const float16x8_t vk18x89ABCDEF = vreinterpretq_f16_u16(vld1q_u16(w)); w += 8;
      vacc01234567p0 = vfmaq_f16(vacc01234567p0, vi18x01234567, vk18x01234567);
      vacc89ABCDEFp0 = vfmaq_f16(vacc89ABCDEFp0, vi18x89ABCDEF, vk18x89ABCDEF);

      const float16x8_t vi19x01234567 = vreinterpretq_f16_u16(vld1q_u16(i19)); i19 += 8;
      const float16x8_t vi19x89ABCDEF = vreinterpretq_f16_u16(vld1q_u16(i19)); i19 += 8;
      const float16x8_t vk19x01234567 = vreinterpretq_f16_u16(vld1q_u16(w)); w += 8;
      const float16x8_t vk19x89ABCDEF = vreinterpretq_f16_u16(vld1q_u16(w)); w += 8;
      vacc01234567p1 = vfmaq_f16(vacc01234567p1, vi19x01234567, vk19x01234567);
      vacc89ABCDEFp1 = vfmaq_f16(vacc89ABCDEFp1, vi19x89ABCDEF, vk19x89ABCDEF);

      const float16x8_t vi20x01234567 = vreinterpretq_f16_u16(vld1q_u16(i20)); i20 += 8;
      const float16x8_t vi20x89ABCDEF = vreinterpretq_f16_u16(vld1q_u16(i20)); i20 += 8;
      const float16x8_t vk20x01234567 = vreinterpretq_f16_u16(vld1q_u16(w)); w += 8;
      const float16x8_t vk20x89ABCDEF = vreinterpretq_f16_u16(vld1q_u16(w)); w += 8;
      vacc01234567p0 = vfmaq_f16(vacc01234567p0, vi20x01234567, vk20x01234567);
      vacc89ABCDEFp0 = vfmaq_f16(vacc89ABCDEFp0, vi20x89ABCDEF, vk20x89ABCDEF);

      const float16x8_t vi21x01234567 = vreinterpretq_f16_u16(vld1q_u16(i21)); i21 += 8;
      const float16x8_t vi21x89ABCDEF = vreinterpretq_f16_u16(vld1q_u16(i21)); i21 += 8;
      const float16x8_t vk21x01234567 = vreinterpretq_f16_u16(vld1q_u16(w)); w += 8;
      const float16x8_t vk21x89ABCDEF = vreinterpretq_f16_u16(vld1q_u16(w)); w += 8;
      vacc01234567p1 = vfmaq_f16(vacc01234567p1, vi21x01234567, vk21x01234567);
      vacc89ABCDEFp1 = vfmaq_f16(vacc89ABCDEFp1, vi21x89ABCDEF, vk21x89ABCDEF);

      const float16x8_t vi22x01234567 = vreinterpretq_f16_u16(vld1q_u16(i22)); i22 += 8;
      const float16x8_t vi22x89ABCDEF = vreinterpretq_f16_u16(vld1q_u16(i22)); i22 += 8;
      const float16x8_t vk22x01234567 = vreinterpretq_f16_u16(vld1q_u16(w)); w += 8;
      const float16x8_t vk22x89ABCDEF = vreinterpretq_f16_u16(vld1q_u16(w)); w += 8;
      vacc01234567p0 = vfmaq_f16(vacc01234567p0, vi22x01234567, vk22x01234567);
      vacc89ABCDEFp0 = vfmaq_f16(vacc89ABCDEFp0, vi22x89ABCDEF, vk22x89ABCDEF);

      const float16x8_t vi23x01234567 = vreinterpretq_f16_u16(vld1q_u16(i23)); i23 += 8;
      const float16x8_t vi23x89ABCDEF = vreinterpretq_f16_u16(vld1q_u16(i23)); i23 += 8;
      const float16x8_t vk23x01234567 = vreinterpretq_f16_u16(vld1q_u16(w)); w += 8;
      const float16x8_t vk23x89ABCDEF = vreinterpretq_f16_u16(vld1q_u16(w)); w += 8;
      vacc01234567p1 = vfmaq_f16(vacc01234567p1, vi23x01234567, vk23x01234567);
      vacc89ABCDEFp1 = vfmaq_f16(vacc89ABCDEFp1, vi23x89ABCDEF, vk23x89ABCDEF);

      const float16x8_t vi24x01234567 = vreinterpretq_f16_u16(vld1q_u16(i24)); i24 += 8;
      const float16x8_t vi24x89ABCDEF = vreinterpretq_f16_u16(vld1q_u16(i24)); i24 += 8;
      const float16x8_t vk24x01234567 = vreinterpretq_f16_u16(vld1q_u16(w)); w += 8;
      const float16x8_t vk24x89ABCDEF = vreinterpretq_f16_u16(vld1q_u16(w)); w += 8;
      vacc01234567p0 = vfmaq_f16(vacc01234567p0, vi24x01234567, vk24x01234567);
      vacc89ABCDEFp0 = vfmaq_f16(vacc89ABCDEFp0, vi24x89ABCDEF, vk24x89ABCDEF);

      // Add up all accumulators to vacc0123456789ABCDEFp0
      vacc01234567p0 = vaddq_f16(vacc01234567p0, vacc01234567p1);
      vacc89ABCDEFp0 = vaddq_f16(vacc89ABCDEFp0, vacc89ABCDEFp1);

      float16x8_t vacc01234567 = vmaxq_f16(vacc01234567p0, vmin);
      float16x8_t vacc89ABCDEF = vmaxq_f16(vacc89ABCDEFp0, vmin);
      vacc01234567 = vminq_f16(vacc01234567, vmax);
      vacc89ABCDEF = vminq_f16(vacc89ABCDEF, vmax);

      vst1q_u16(output, vreinterpretq_u16_f16(vacc01234567)); output += 8;
      vst1q_u16(output, vreinterpretq_u16_f16(vacc89ABCDEF)); output += 8;
    }
    for (; c >= 8; c -= 8) {
      float16x8_t vacc01234567p0 = vreinterpretq_f16_u16(vld1q_u16(w)); w += 8;


      const float16x8_t vi0x01234567 = vreinterpretq_f16_u16(vld1q_u16(i0)); i0 += 8;
      const float16x8_t vk0x01234567 = vreinterpretq_f16_u16(vld1q_u16(w + 8));
      vacc01234567p0 = vfmaq_f16(vacc01234567p0, vi0x01234567, vk0x01234567);

      const float16x8_t vi1x01234567 = vreinterpretq_f16_u16(vld1q_u16(i1)); i1 += 8;
      const float16x8_t vk1x01234567 = vreinterpretq_f16_u16(vld1q_u16(w + 24));
      float16x8_t vacc01234567p1 = vmulq_f16(vi1x01234567, vk1x01234567);

      const float16x8_t vi2x01234567 = vreinterpretq_f16_u16(vld1q_u16(i2)); i2 += 8;
      const float16x8_t vk2x01234567 = vreinterpretq_f16_u16(vld1q_u16(w + 40));
      vacc01234567p0 = vfmaq_f16(vacc01234567p0, vi2x01234567, vk2x01234567);

      const float16x8_t vi3x01234567 = vreinterpretq_f16_u16(vld1q_u16(i3)); i3 += 8;
      const float16x8_t vk3x01234567 = vreinterpretq_f16_u16(vld1q_u16(w + 56));
      vacc01234567p1 = vfmaq_f16(vacc01234567p1, vi3x01234567, vk3x01234567);

      const float16x8_t vi4x01234567 = vreinterpretq_f16_u16(vld1q_u16(i4)); i4 += 8;
      const float16x8_t vk4x01234567 = vreinterpretq_f16_u16(vld1q_u16(w + 72));
      vacc01234567p0 = vfmaq_f16(vacc01234567p0, vi4x01234567, vk4x01234567);

      const float16x8_t vi5x01234567 = vreinterpretq_f16_u16(vld1q_u16(i5)); i5 += 8;
      const float16x8_t vk5x01234567 = vreinterpretq_f16_u16(vld1q_u16(w + 88));
      vacc01234567p1 = vfmaq_f16(vacc01234567p1, vi5x01234567, vk5x01234567);

      const float16x8_t vi6x01234567 = vreinterpretq_f16_u16(vld1q_u16(i6)); i6 += 8;
      const float16x8_t vk6x01234567 = vreinterpretq_f16_u16(vld1q_u16(w + 104));
      vacc01234567p0 = vfmaq_f16(vacc01234567p0, vi6x01234567, vk6x01234567);

      const float16x8_t vi7x01234567 = vreinterpretq_f16_u16(vld1q_u16(i7)); i7 += 8;
      const float16x8_t vk7x01234567 = vreinterpretq_f16_u16(vld1q_u16(w + 120));
      vacc01234567p1 = vfmaq_f16(vacc01234567p1, vi7x01234567, vk7x01234567);

      const float16x8_t vi8x01234567 = vreinterpretq_f16_u16(vld1q_u16(i8)); i8 += 8;
      const float16x8_t vk8x01234567 = vreinterpretq_f16_u16(vld1q_u16(w + 136));
      vacc01234567p0 = vfmaq_f16(vacc01234567p0, vi8x01234567, vk8x01234567);

      const float16x8_t vi9x01234567 = vreinterpretq_f16_u16(vld1q_u16(i9)); i9 += 8;
      const float16x8_t vk9x01234567 = vreinterpretq_f16_u16(vld1q_u16(w + 152));
      vacc01234567p1 = vfmaq_f16(vacc01234567p1, vi9x01234567, vk9x01234567);

      const float16x8_t vi10x01234567 = vreinterpretq_f16_u16(vld1q_u16(i10)); i10 += 8;
      const float16x8_t vk10x01234567 = vreinterpretq_f16_u16(vld1q_u16(w + 168));
      vacc01234567p0 = vfmaq_f16(vacc01234567p0, vi10x01234567, vk10x01234567);

      const float16x8_t vi11x01234567 = vreinterpretq_f16_u16(vld1q_u16(i11)); i11 += 8;
      const float16x8_t vk11x01234567 = vreinterpretq_f16_u16(vld1q_u16(w + 184));
      vacc01234567p1 = vfmaq_f16(vacc01234567p1, vi11x01234567, vk11x01234567);

      const float16x8_t vi12x01234567 = vreinterpretq_f16_u16(vld1q_u16(i12)); i12 += 8;
      const float16x8_t vk12x01234567 = vreinterpretq_f16_u16(vld1q_u16(w + 200));
      vacc01234567p0 = vfmaq_f16(vacc01234567p0, vi12x01234567, vk12x01234567);

      const float16x8_t vi13x01234567 = vreinterpretq_f16_u16(vld1q_u16(i13)); i13 += 8;
      const float16x8_t vk13x01234567 = vreinterpretq_f16_u16(vld1q_u16(w + 216));
      vacc01234567p1 = vfmaq_f16(vacc01234567p1, vi13x01234567, vk13x01234567);

      const float16x8_t vi14x01234567 = vreinterpretq_f16_u16(vld1q_u16(i14)); i14 += 8;
      const float16x8_t vk14x01234567 = vreinterpretq_f16_u16(vld1q_u16(w + 232));
      vacc01234567p0 = vfmaq_f16(vacc01234567p0, vi14x01234567, vk14x01234567);

      const float16x8_t vi15x01234567 = vreinterpretq_f16_u16(vld1q_u16(i15)); i15 += 8;
      const float16x8_t vk15x01234567 = vreinterpretq_f16_u16(vld1q_u16(w + 248));
      vacc01234567p1 = vfmaq_f16(vacc01234567p1, vi15x01234567, vk15x01234567);

      const float16x8_t vi16x01234567 = vreinterpretq_f16_u16(vld1q_u16(i16)); i16 += 8;
      const float16x8_t vk16x01234567 = vreinterpretq_f16_u16(vld1q_u16(w + 264));
      vacc01234567p0 = vfmaq_f16(vacc01234567p0, vi16x01234567, vk16x01234567);

      const float16x8_t vi17x01234567 = vreinterpretq_f16_u16(vld1q_u16(i17)); i17 += 8;
      const float16x8_t vk17x01234567 = vreinterpretq_f16_u16(vld1q_u16(w + 280));
      vacc01234567p1 = vfmaq_f16(vacc01234567p1, vi17x01234567, vk17x01234567);

      const float16x8_t vi18x01234567 = vreinterpretq_f16_u16(vld1q_u16(i18)); i18 += 8;
      const float16x8_t vk18x01234567 = vreinterpretq_f16_u16(vld1q_u16(w + 296));
      vacc01234567p0 = vfmaq_f16(vacc01234567p0, vi18x01234567, vk18x01234567);

      const float16x8_t vi19x01234567 = vreinterpretq_f16_u16(vld1q_u16(i19)); i19 += 8;
      const float16x8_t vk19x01234567 = vreinterpretq_f16_u16(vld1q_u16(w + 312));
      vacc01234567p1 = vfmaq_f16(vacc01234567p1, vi19x01234567, vk19x01234567);

      const float16x8_t vi20x01234567 = vreinterpretq_f16_u16(vld1q_u16(i20)); i20 += 8;
      const float16x8_t vk20x01234567 = vreinterpretq_f16_u16(vld1q_u16(w + 328));
      vacc01234567p0 = vfmaq_f16(vacc01234567p0, vi20x01234567, vk20x01234567);

      const float16x8_t vi21x01234567 = vreinterpretq_f16_u16(vld1q_u16(i21)); i21 += 8;
      const float16x8_t vk21x01234567 = vreinterpretq_f16_u16(vld1q_u16(w + 344));
      vacc01234567p1 = vfmaq_f16(vacc01234567p1, vi21x01234567, vk21x01234567);

      const float16x8_t vi22x01234567 = vreinterpretq_f16_u16(vld1q_u16(i22)); i22 += 8;
      const float16x8_t vk22x01234567 = vreinterpretq_f16_u16(vld1q_u16(w + 360));
      vacc01234567p0 = vfmaq_f16(vacc01234567p0, vi22x01234567, vk22x01234567);

      const float16x8_t vi23x01234567 = vreinterpretq_f16_u16(vld1q_u16(i23)); i23 += 8;
      const float16x8_t vk23x01234567 = vreinterpretq_f16_u16(vld1q_u16(w + 376));
      vacc01234567p1 = vfmaq_f16(vacc01234567p1, vi23x01234567, vk23x01234567);

      const float16x8_t vi24x01234567 = vreinterpretq_f16_u16(vld1q_u16(i24)); i24 += 8;
      const float16x8_t vk24x01234567 = vreinterpretq_f16_u16(vld1q_u16(w + 392));
      vacc01234567p0 = vfmaq_f16(vacc01234567p0, vi24x01234567, vk24x01234567);

      // Add up all accumulators to vacc01234567p0
      vacc01234567p0 = vaddq_f16(vacc01234567p0, vacc01234567p1);

      float16x8_t vacc01234567 = vmaxq_f16(vacc01234567p0, vmin);
      vacc01234567 = vminq_f16(vacc01234567, vmax);

      vst1q_u16(output, vreinterpretq_u16_f16(vacc01234567)); output += 8;
    }
    if XNN_UNLIKELY(c != 0) {
      float16x8_t vacc01234567p0 = vreinterpretq_f16_u16(vld1q_u16(w));


      const float16x8_t vi0x01234567 = vreinterpretq_f16_u16(vld1q_u16(i0));
      const float16x8_t vk0x01234567 = vreinterpretq_f16_u16(vld1q_u16(w + 16));
      vacc01234567p0 = vfmaq_f16(vacc01234567p0, vi0x01234567, vk0x01234567);

      const float16x8_t vi1x01234567 = vreinterpretq_f16_u16(vld1q_u16(i1));
      const float16x8_t vk1x01234567 = vreinterpretq_f16_u16(vld1q_u16(w + 32));
      float16x8_t vacc01234567p1 = vmulq_f16(vi1x01234567, vk1x01234567);

      const float16x8_t vi2x01234567 = vreinterpretq_f16_u16(vld1q_u16(i2));
      const float16x8_t vk2x01234567 = vreinterpretq_f16_u16(vld1q_u16(w + 48));
      vacc01234567p0 = vfmaq_f16(vacc01234567p0, vi2x01234567, vk2x01234567);

      const float16x8_t vi3x01234567 = vreinterpretq_f16_u16(vld1q_u16(i3));
      const float16x8_t vk3x01234567 = vreinterpretq_f16_u16(vld1q_u16(w + 64));
      vacc01234567p1 = vfmaq_f16(vacc01234567p1, vi3x01234567, vk3x01234567);

      const float16x8_t vi4x01234567 = vreinterpretq_f16_u16(vld1q_u16(i4));
      const float16x8_t vk4x01234567 = vreinterpretq_f16_u16(vld1q_u16(w + 80));
      vacc01234567p0 = vfmaq_f16(vacc01234567p0, vi4x01234567, vk4x01234567);

      const float16x8_t vi5x01234567 = vreinterpretq_f16_u16(vld1q_u16(i5));
      const float16x8_t vk5x01234567 = vreinterpretq_f16_u16(vld1q_u16(w + 96));
      vacc01234567p1 = vfmaq_f16(vacc01234567p1, vi5x01234567, vk5x01234567);

      const float16x8_t vi6x01234567 = vreinterpretq_f16_u16(vld1q_u16(i6));
      const float16x8_t vk6x01234567 = vreinterpretq_f16_u16(vld1q_u16(w + 112));
      vacc01234567p0 = vfmaq_f16(vacc01234567p0, vi6x01234567, vk6x01234567);

      const float16x8_t vi7x01234567 = vreinterpretq_f16_u16(vld1q_u16(i7));
      const float16x8_t vk7x01234567 = vreinterpretq_f16_u16(vld1q_u16(w + 128));
      vacc01234567p1 = vfmaq_f16(vacc01234567p1, vi7x01234567, vk7x01234567);

      const float16x8_t vi8x01234567 = vreinterpretq_f16_u16(vld1q_u16(i8));
      const float16x8_t vk8x01234567 = vreinterpretq_f16_u16(vld1q_u16(w + 144));
      vacc01234567p0 = vfmaq_f16(vacc01234567p0, vi8x01234567, vk8x01234567);

      const float16x8_t vi9x01234567 = vreinterpretq_f16_u16(vld1q_u16(i9));
      const float16x8_t vk9x01234567 = vreinterpretq_f16_u16(vld1q_u16(w + 160));
      vacc01234567p1 = vfmaq_f16(vacc01234567p1, vi9x01234567, vk9x01234567);

      const float16x8_t vi10x01234567 = vreinterpretq_f16_u16(vld1q_u16(i10));
      const float16x8_t vk10x01234567 = vreinterpretq_f16_u16(vld1q_u16(w + 176));
      vacc01234567p0 = vfmaq_f16(vacc01234567p0, vi10x01234567, vk10x01234567);

      const float16x8_t vi11x01234567 = vreinterpretq_f16_u16(vld1q_u16(i11));
      const float16x8_t vk11x01234567 = vreinterpretq_f16_u16(vld1q_u16(w + 192));
      vacc01234567p1 = vfmaq_f16(vacc01234567p1, vi11x01234567, vk11x01234567);

      const float16x8_t vi12x01234567 = vreinterpretq_f16_u16(vld1q_u16(i12));
      const float16x8_t vk12x01234567 = vreinterpretq_f16_u16(vld1q_u16(w + 208));
      vacc01234567p0 = vfmaq_f16(vacc01234567p0, vi12x01234567, vk12x01234567);

      const float16x8_t vi13x01234567 = vreinterpretq_f16_u16(vld1q_u16(i13));
      const float16x8_t vk13x01234567 = vreinterpretq_f16_u16(vld1q_u16(w + 224));
      vacc01234567p1 = vfmaq_f16(vacc01234567p1, vi13x01234567, vk13x01234567);

      const float16x8_t vi14x01234567 = vreinterpretq_f16_u16(vld1q_u16(i14));
      const float16x8_t vk14x01234567 = vreinterpretq_f16_u16(vld1q_u16(w + 240));
      vacc01234567p0 = vfmaq_f16(vacc01234567p0, vi14x01234567, vk14x01234567);

      const float16x8_t vi15x01234567 = vreinterpretq_f16_u16(vld1q_u16(i15));
      const float16x8_t vk15x01234567 = vreinterpretq_f16_u16(vld1q_u16(w + 256));
      vacc01234567p1 = vfmaq_f16(vacc01234567p1, vi15x01234567, vk15x01234567);

      const float16x8_t vi16x01234567 = vreinterpretq_f16_u16(vld1q_u16(i16));
      const float16x8_t vk16x01234567 = vreinterpretq_f16_u16(vld1q_u16(w + 272));
      vacc01234567p0 = vfmaq_f16(vacc01234567p0, vi16x01234567, vk16x01234567);

      const float16x8_t vi17x01234567 = vreinterpretq_f16_u16(vld1q_u16(i17));
      const float16x8_t vk17x01234567 = vreinterpretq_f16_u16(vld1q_u16(w + 288));
      vacc01234567p1 = vfmaq_f16(vacc01234567p1, vi17x01234567, vk17x01234567);

      const float16x8_t vi18x01234567 = vreinterpretq_f16_u16(vld1q_u16(i18));
      const float16x8_t vk18x01234567 = vreinterpretq_f16_u16(vld1q_u16(w + 304));
      vacc01234567p0 = vfmaq_f16(vacc01234567p0, vi18x01234567, vk18x01234567);

      const float16x8_t vi19x01234567 = vreinterpretq_f16_u16(vld1q_u16(i19));
      const float16x8_t vk19x01234567 = vreinterpretq_f16_u16(vld1q_u16(w + 320));
      vacc01234567p1 = vfmaq_f16(vacc01234567p1, vi19x01234567, vk19x01234567);

      const float16x8_t vi20x01234567 = vreinterpretq_f16_u16(vld1q_u16(i20));
      const float16x8_t vk20x01234567 = vreinterpretq_f16_u16(vld1q_u16(w + 336));
      vacc01234567p0 = vfmaq_f16(vacc01234567p0, vi20x01234567, vk20x01234567);

      const float16x8_t vi21x01234567 = vreinterpretq_f16_u16(vld1q_u16(i21));
      const float16x8_t vk21x01234567 = vreinterpretq_f16_u16(vld1q_u16(w + 352));
      vacc01234567p1 = vfmaq_f16(vacc01234567p1, vi21x01234567, vk21x01234567);

      const float16x8_t vi22x01234567 = vreinterpretq_f16_u16(vld1q_u16(i22));
      const float16x8_t vk22x01234567 = vreinterpretq_f16_u16(vld1q_u16(w + 368));
      vacc01234567p0 = vfmaq_f16(vacc01234567p0, vi22x01234567, vk22x01234567);

      const float16x8_t vi23x01234567 = vreinterpretq_f16_u16(vld1q_u16(i23));
      const float16x8_t vk23x01234567 = vreinterpretq_f16_u16(vld1q_u16(w + 384));
      vacc01234567p1 = vfmaq_f16(vacc01234567p1, vi23x01234567, vk23x01234567);

      const float16x8_t vi24x01234567 = vreinterpretq_f16_u16(vld1q_u16(i24));
      const float16x8_t vk24x01234567 = vreinterpretq_f16_u16(vld1q_u16(w + 400));
      vacc01234567p0 = vfmaq_f16(vacc01234567p0, vi24x01234567, vk24x01234567);

      // Add up all accumulators to vacc01234567p0
      vacc01234567p0 = vaddq_f16(vacc01234567p0, vacc01234567p1);

      float16x8_t vacc01234567 = vmaxq_f16(vacc01234567p0, vmin);
      vacc01234567 = vminq_f16(vacc01234567, vmax);

      float16x4_t vacc0123 = vget_low_f16(vacc01234567);
      if (c & 4) {
        vst1_u16(output, vreinterpret_u16_f16(vacc0123)); output += 4;
        vacc0123 = vget_high_f16(vacc01234567);
      }
      if (c & 2) {
        vst1_lane_u32((void*) output, vreinterpret_u32_f16(vacc0123), 0); output += 2;
        vacc0123 = vext_f16(vacc0123, vacc0123, 2);
      }
      if (c & 1) {
        vst1_lane_u16(output, vreinterpret_u16_f16(vacc0123), 0); output += 1;
      }
    }

    output = (uint16_t*) ((uintptr_t) output + output_increment);
  } while (--output_width != 0);
}
