// Auto-generated file. Do not edit!
//   Template: src/f32-dwconv/unipass-neon.c.in
//   Generator: tools/xngen
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <arm_neon.h>

#include "xnnpack/dwconv.h"


void xnn_f32_dwconv_minmax_ukernel_25p16c__neon_acc2(
    size_t channels,
    size_t output_width,
    const float** input,
    const float* weights,
    float* output,
    intptr_t input_stride,
    size_t output_increment,
    size_t input_offset,
    const float* zero,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(channels != 0);
  assert(output_width != 0);

  const float32x4_t vmax = vld1q_dup_f32(&params->scalar.max);
  const float32x4_t vmin = vld1q_dup_f32(&params->scalar.min);
  do {
    const float* i0 = input[0];
    assert(i0 != NULL);
    if XNN_UNPREDICTABLE(i0 != zero) {
      i0 = (const float*) ((uintptr_t) i0 + input_offset);
    }
    const float* i1 = input[1];
    assert(i1 != NULL);
    if XNN_UNPREDICTABLE(i1 != zero) {
      i1 = (const float*) ((uintptr_t) i1 + input_offset);
    }
    const float* i2 = input[2];
    assert(i2 != NULL);
    if XNN_UNPREDICTABLE(i2 != zero) {
      i2 = (const float*) ((uintptr_t) i2 + input_offset);
    }
    const float* i3 = input[3];
    assert(i3 != NULL);
    if XNN_UNPREDICTABLE(i3 != zero) {
      i3 = (const float*) ((uintptr_t) i3 + input_offset);
    }
    const float* i4 = input[4];
    assert(i4 != NULL);
    if XNN_UNPREDICTABLE(i4 != zero) {
      i4 = (const float*) ((uintptr_t) i4 + input_offset);
    }
    const float* i5 = input[5];
    assert(i5 != NULL);
    if XNN_UNPREDICTABLE(i5 != zero) {
      i5 = (const float*) ((uintptr_t) i5 + input_offset);
    }
    const float* i6 = input[6];
    assert(i6 != NULL);
    if XNN_UNPREDICTABLE(i6 != zero) {
      i6 = (const float*) ((uintptr_t) i6 + input_offset);
    }
    const float* i7 = input[7];
    assert(i7 != NULL);
    if XNN_UNPREDICTABLE(i7 != zero) {
      i7 = (const float*) ((uintptr_t) i7 + input_offset);
    }
    const float* i8 = input[8];
    assert(i8 != NULL);
    if XNN_UNPREDICTABLE(i8 != zero) {
      i8 = (const float*) ((uintptr_t) i8 + input_offset);
    }
    const float* i9 = input[9];
    assert(i9 != NULL);
    if XNN_UNPREDICTABLE(i9 != zero) {
      i9 = (const float*) ((uintptr_t) i9 + input_offset);
    }
    const float* i10 = input[10];
    assert(i10 != NULL);
    if XNN_UNPREDICTABLE(i10 != zero) {
      i10 = (const float*) ((uintptr_t) i10 + input_offset);
    }
    const float* i11 = input[11];
    assert(i11 != NULL);
    if XNN_UNPREDICTABLE(i11 != zero) {
      i11 = (const float*) ((uintptr_t) i11 + input_offset);
    }
    const float* i12 = input[12];
    assert(i12 != NULL);
    if XNN_UNPREDICTABLE(i12 != zero) {
      i12 = (const float*) ((uintptr_t) i12 + input_offset);
    }
    const float* i13 = input[13];
    assert(i13 != NULL);
    if XNN_UNPREDICTABLE(i13 != zero) {
      i13 = (const float*) ((uintptr_t) i13 + input_offset);
    }
    const float* i14 = input[14];
    assert(i14 != NULL);
    if XNN_UNPREDICTABLE(i14 != zero) {
      i14 = (const float*) ((uintptr_t) i14 + input_offset);
    }
    const float* i15 = input[15];
    assert(i15 != NULL);
    if XNN_UNPREDICTABLE(i15 != zero) {
      i15 = (const float*) ((uintptr_t) i15 + input_offset);
    }
    const float* i16 = input[16];
    assert(i16 != NULL);
    if XNN_UNPREDICTABLE(i16 != zero) {
      i16 = (const float*) ((uintptr_t) i16 + input_offset);
    }
    const float* i17 = input[17];
    assert(i17 != NULL);
    if XNN_UNPREDICTABLE(i17 != zero) {
      i17 = (const float*) ((uintptr_t) i17 + input_offset);
    }
    const float* i18 = input[18];
    assert(i18 != NULL);
    if XNN_UNPREDICTABLE(i18 != zero) {
      i18 = (const float*) ((uintptr_t) i18 + input_offset);
    }
    const float* i19 = input[19];
    assert(i19 != NULL);
    if XNN_UNPREDICTABLE(i19 != zero) {
      i19 = (const float*) ((uintptr_t) i19 + input_offset);
    }
    const float* i20 = input[20];
    assert(i20 != NULL);
    if XNN_UNPREDICTABLE(i20 != zero) {
      i20 = (const float*) ((uintptr_t) i20 + input_offset);
    }
    const float* i21 = input[21];
    assert(i21 != NULL);
    if XNN_UNPREDICTABLE(i21 != zero) {
      i21 = (const float*) ((uintptr_t) i21 + input_offset);
    }
    const float* i22 = input[22];
    assert(i22 != NULL);
    if XNN_UNPREDICTABLE(i22 != zero) {
      i22 = (const float*) ((uintptr_t) i22 + input_offset);
    }
    const float* i23 = input[23];
    assert(i23 != NULL);
    if XNN_UNPREDICTABLE(i23 != zero) {
      i23 = (const float*) ((uintptr_t) i23 + input_offset);
    }
    const float* i24 = input[24];
    assert(i24 != NULL);
    if XNN_UNPREDICTABLE(i24 != zero) {
      i24 = (const float*) ((uintptr_t) i24 + input_offset);
    }

    input = (const float**) ((uintptr_t) input + input_stride);

    size_t c = channels;
    const float* w = weights;
    for (; c >= 16; c -= 16) {
      float32x4_t vacc0123p0 = vld1q_f32(w); w += 4;
      float32x4_t vacc4567p0 = vld1q_f32(w); w += 4;
      float32x4_t vacc89ABp0 = vld1q_f32(w); w += 4;
      float32x4_t vaccCDEFp0 = vld1q_f32(w); w += 4;


      const float32x4_t vi0x0123 = vld1q_f32(i0); i0 += 4;
      const float32x4_t vi0x4567 = vld1q_f32(i0); i0 += 4;
      const float32x4_t vi0x89AB = vld1q_f32(i0); i0 += 4;
      const float32x4_t vi0xCDEF = vld1q_f32(i0); i0 += 4;
      const float32x4_t vk0x0123 = vld1q_f32(w); w += 4;
      const float32x4_t vk0x4567 = vld1q_f32(w); w += 4;
      const float32x4_t vk0x89AB = vld1q_f32(w); w += 4;
      const float32x4_t vk0xCDEF = vld1q_f32(w); w += 4;
      vacc0123p0 = vmlaq_f32(vacc0123p0, vi0x0123, vk0x0123);
      vacc4567p0 = vmlaq_f32(vacc4567p0, vi0x4567, vk0x4567);
      vacc89ABp0 = vmlaq_f32(vacc89ABp0, vi0x89AB, vk0x89AB);
      vaccCDEFp0 = vmlaq_f32(vaccCDEFp0, vi0xCDEF, vk0xCDEF);

      const float32x4_t vi1x0123 = vld1q_f32(i1); i1 += 4;
      const float32x4_t vi1x4567 = vld1q_f32(i1); i1 += 4;
      const float32x4_t vi1x89AB = vld1q_f32(i1); i1 += 4;
      const float32x4_t vi1xCDEF = vld1q_f32(i1); i1 += 4;
      const float32x4_t vk1x0123 = vld1q_f32(w); w += 4;
      const float32x4_t vk1x4567 = vld1q_f32(w); w += 4;
      const float32x4_t vk1x89AB = vld1q_f32(w); w += 4;
      const float32x4_t vk1xCDEF = vld1q_f32(w); w += 4;
      float32x4_t vacc0123p1 = vmulq_f32(vi1x0123, vk1x0123);
      float32x4_t vacc4567p1 = vmulq_f32(vi1x4567, vk1x4567);
      float32x4_t vacc89ABp1 = vmulq_f32(vi1x89AB, vk1x89AB);
      float32x4_t vaccCDEFp1 = vmulq_f32(vi1xCDEF, vk1xCDEF);

      const float32x4_t vi2x0123 = vld1q_f32(i2); i2 += 4;
      const float32x4_t vi2x4567 = vld1q_f32(i2); i2 += 4;
      const float32x4_t vi2x89AB = vld1q_f32(i2); i2 += 4;
      const float32x4_t vi2xCDEF = vld1q_f32(i2); i2 += 4;
      const float32x4_t vk2x0123 = vld1q_f32(w); w += 4;
      const float32x4_t vk2x4567 = vld1q_f32(w); w += 4;
      const float32x4_t vk2x89AB = vld1q_f32(w); w += 4;
      const float32x4_t vk2xCDEF = vld1q_f32(w); w += 4;
      vacc0123p0 = vmlaq_f32(vacc0123p0, vi2x0123, vk2x0123);
      vacc4567p0 = vmlaq_f32(vacc4567p0, vi2x4567, vk2x4567);
      vacc89ABp0 = vmlaq_f32(vacc89ABp0, vi2x89AB, vk2x89AB);
      vaccCDEFp0 = vmlaq_f32(vaccCDEFp0, vi2xCDEF, vk2xCDEF);

      const float32x4_t vi3x0123 = vld1q_f32(i3); i3 += 4;
      const float32x4_t vi3x4567 = vld1q_f32(i3); i3 += 4;
      const float32x4_t vi3x89AB = vld1q_f32(i3); i3 += 4;
      const float32x4_t vi3xCDEF = vld1q_f32(i3); i3 += 4;
      const float32x4_t vk3x0123 = vld1q_f32(w); w += 4;
      const float32x4_t vk3x4567 = vld1q_f32(w); w += 4;
      const float32x4_t vk3x89AB = vld1q_f32(w); w += 4;
      const float32x4_t vk3xCDEF = vld1q_f32(w); w += 4;
      vacc0123p1 = vmlaq_f32(vacc0123p1, vi3x0123, vk3x0123);
      vacc4567p1 = vmlaq_f32(vacc4567p1, vi3x4567, vk3x4567);
      vacc89ABp1 = vmlaq_f32(vacc89ABp1, vi3x89AB, vk3x89AB);
      vaccCDEFp1 = vmlaq_f32(vaccCDEFp1, vi3xCDEF, vk3xCDEF);

      const float32x4_t vi4x0123 = vld1q_f32(i4); i4 += 4;
      const float32x4_t vi4x4567 = vld1q_f32(i4); i4 += 4;
      const float32x4_t vi4x89AB = vld1q_f32(i4); i4 += 4;
      const float32x4_t vi4xCDEF = vld1q_f32(i4); i4 += 4;
      const float32x4_t vk4x0123 = vld1q_f32(w); w += 4;
      const float32x4_t vk4x4567 = vld1q_f32(w); w += 4;
      const float32x4_t vk4x89AB = vld1q_f32(w); w += 4;
      const float32x4_t vk4xCDEF = vld1q_f32(w); w += 4;
      vacc0123p0 = vmlaq_f32(vacc0123p0, vi4x0123, vk4x0123);
      vacc4567p0 = vmlaq_f32(vacc4567p0, vi4x4567, vk4x4567);
      vacc89ABp0 = vmlaq_f32(vacc89ABp0, vi4x89AB, vk4x89AB);
      vaccCDEFp0 = vmlaq_f32(vaccCDEFp0, vi4xCDEF, vk4xCDEF);

      const float32x4_t vi5x0123 = vld1q_f32(i5); i5 += 4;
      const float32x4_t vi5x4567 = vld1q_f32(i5); i5 += 4;
      const float32x4_t vi5x89AB = vld1q_f32(i5); i5 += 4;
      const float32x4_t vi5xCDEF = vld1q_f32(i5); i5 += 4;
      const float32x4_t vk5x0123 = vld1q_f32(w); w += 4;
      const float32x4_t vk5x4567 = vld1q_f32(w); w += 4;
      const float32x4_t vk5x89AB = vld1q_f32(w); w += 4;
      const float32x4_t vk5xCDEF = vld1q_f32(w); w += 4;
      vacc0123p1 = vmlaq_f32(vacc0123p1, vi5x0123, vk5x0123);
      vacc4567p1 = vmlaq_f32(vacc4567p1, vi5x4567, vk5x4567);
      vacc89ABp1 = vmlaq_f32(vacc89ABp1, vi5x89AB, vk5x89AB);
      vaccCDEFp1 = vmlaq_f32(vaccCDEFp1, vi5xCDEF, vk5xCDEF);

      const float32x4_t vi6x0123 = vld1q_f32(i6); i6 += 4;
      const float32x4_t vi6x4567 = vld1q_f32(i6); i6 += 4;
      const float32x4_t vi6x89AB = vld1q_f32(i6); i6 += 4;
      const float32x4_t vi6xCDEF = vld1q_f32(i6); i6 += 4;
      const float32x4_t vk6x0123 = vld1q_f32(w); w += 4;
      const float32x4_t vk6x4567 = vld1q_f32(w); w += 4;
      const float32x4_t vk6x89AB = vld1q_f32(w); w += 4;
      const float32x4_t vk6xCDEF = vld1q_f32(w); w += 4;
      vacc0123p0 = vmlaq_f32(vacc0123p0, vi6x0123, vk6x0123);
      vacc4567p0 = vmlaq_f32(vacc4567p0, vi6x4567, vk6x4567);
      vacc89ABp0 = vmlaq_f32(vacc89ABp0, vi6x89AB, vk6x89AB);
      vaccCDEFp0 = vmlaq_f32(vaccCDEFp0, vi6xCDEF, vk6xCDEF);

      const float32x4_t vi7x0123 = vld1q_f32(i7); i7 += 4;
      const float32x4_t vi7x4567 = vld1q_f32(i7); i7 += 4;
      const float32x4_t vi7x89AB = vld1q_f32(i7); i7 += 4;
      const float32x4_t vi7xCDEF = vld1q_f32(i7); i7 += 4;
      const float32x4_t vk7x0123 = vld1q_f32(w); w += 4;
      const float32x4_t vk7x4567 = vld1q_f32(w); w += 4;
      const float32x4_t vk7x89AB = vld1q_f32(w); w += 4;
      const float32x4_t vk7xCDEF = vld1q_f32(w); w += 4;
      vacc0123p1 = vmlaq_f32(vacc0123p1, vi7x0123, vk7x0123);
      vacc4567p1 = vmlaq_f32(vacc4567p1, vi7x4567, vk7x4567);
      vacc89ABp1 = vmlaq_f32(vacc89ABp1, vi7x89AB, vk7x89AB);
      vaccCDEFp1 = vmlaq_f32(vaccCDEFp1, vi7xCDEF, vk7xCDEF);

      const float32x4_t vi8x0123 = vld1q_f32(i8); i8 += 4;
      const float32x4_t vi8x4567 = vld1q_f32(i8); i8 += 4;
      const float32x4_t vi8x89AB = vld1q_f32(i8); i8 += 4;
      const float32x4_t vi8xCDEF = vld1q_f32(i8); i8 += 4;
      const float32x4_t vk8x0123 = vld1q_f32(w); w += 4;
      const float32x4_t vk8x4567 = vld1q_f32(w); w += 4;
      const float32x4_t vk8x89AB = vld1q_f32(w); w += 4;
      const float32x4_t vk8xCDEF = vld1q_f32(w); w += 4;
      vacc0123p0 = vmlaq_f32(vacc0123p0, vi8x0123, vk8x0123);
      vacc4567p0 = vmlaq_f32(vacc4567p0, vi8x4567, vk8x4567);
      vacc89ABp0 = vmlaq_f32(vacc89ABp0, vi8x89AB, vk8x89AB);
      vaccCDEFp0 = vmlaq_f32(vaccCDEFp0, vi8xCDEF, vk8xCDEF);

      const float32x4_t vi9x0123 = vld1q_f32(i9); i9 += 4;
      const float32x4_t vi9x4567 = vld1q_f32(i9); i9 += 4;
      const float32x4_t vi9x89AB = vld1q_f32(i9); i9 += 4;
      const float32x4_t vi9xCDEF = vld1q_f32(i9); i9 += 4;
      const float32x4_t vk9x0123 = vld1q_f32(w); w += 4;
      const float32x4_t vk9x4567 = vld1q_f32(w); w += 4;
      const float32x4_t vk9x89AB = vld1q_f32(w); w += 4;
      const float32x4_t vk9xCDEF = vld1q_f32(w); w += 4;
      vacc0123p1 = vmlaq_f32(vacc0123p1, vi9x0123, vk9x0123);
      vacc4567p1 = vmlaq_f32(vacc4567p1, vi9x4567, vk9x4567);
      vacc89ABp1 = vmlaq_f32(vacc89ABp1, vi9x89AB, vk9x89AB);
      vaccCDEFp1 = vmlaq_f32(vaccCDEFp1, vi9xCDEF, vk9xCDEF);

      const float32x4_t vi10x0123 = vld1q_f32(i10); i10 += 4;
      const float32x4_t vi10x4567 = vld1q_f32(i10); i10 += 4;
      const float32x4_t vi10x89AB = vld1q_f32(i10); i10 += 4;
      const float32x4_t vi10xCDEF = vld1q_f32(i10); i10 += 4;
      const float32x4_t vk10x0123 = vld1q_f32(w); w += 4;
      const float32x4_t vk10x4567 = vld1q_f32(w); w += 4;
      const float32x4_t vk10x89AB = vld1q_f32(w); w += 4;
      const float32x4_t vk10xCDEF = vld1q_f32(w); w += 4;
      vacc0123p0 = vmlaq_f32(vacc0123p0, vi10x0123, vk10x0123);
      vacc4567p0 = vmlaq_f32(vacc4567p0, vi10x4567, vk10x4567);
      vacc89ABp0 = vmlaq_f32(vacc89ABp0, vi10x89AB, vk10x89AB);
      vaccCDEFp0 = vmlaq_f32(vaccCDEFp0, vi10xCDEF, vk10xCDEF);

      const float32x4_t vi11x0123 = vld1q_f32(i11); i11 += 4;
      const float32x4_t vi11x4567 = vld1q_f32(i11); i11 += 4;
      const float32x4_t vi11x89AB = vld1q_f32(i11); i11 += 4;
      const float32x4_t vi11xCDEF = vld1q_f32(i11); i11 += 4;
      const float32x4_t vk11x0123 = vld1q_f32(w); w += 4;
      const float32x4_t vk11x4567 = vld1q_f32(w); w += 4;
      const float32x4_t vk11x89AB = vld1q_f32(w); w += 4;
      const float32x4_t vk11xCDEF = vld1q_f32(w); w += 4;
      vacc0123p1 = vmlaq_f32(vacc0123p1, vi11x0123, vk11x0123);
      vacc4567p1 = vmlaq_f32(vacc4567p1, vi11x4567, vk11x4567);
      vacc89ABp1 = vmlaq_f32(vacc89ABp1, vi11x89AB, vk11x89AB);
      vaccCDEFp1 = vmlaq_f32(vaccCDEFp1, vi11xCDEF, vk11xCDEF);

      const float32x4_t vi12x0123 = vld1q_f32(i12); i12 += 4;
      const float32x4_t vi12x4567 = vld1q_f32(i12); i12 += 4;
      const float32x4_t vi12x89AB = vld1q_f32(i12); i12 += 4;
      const float32x4_t vi12xCDEF = vld1q_f32(i12); i12 += 4;
      const float32x4_t vk12x0123 = vld1q_f32(w); w += 4;
      const float32x4_t vk12x4567 = vld1q_f32(w); w += 4;
      const float32x4_t vk12x89AB = vld1q_f32(w); w += 4;
      const float32x4_t vk12xCDEF = vld1q_f32(w); w += 4;
      vacc0123p0 = vmlaq_f32(vacc0123p0, vi12x0123, vk12x0123);
      vacc4567p0 = vmlaq_f32(vacc4567p0, vi12x4567, vk12x4567);
      vacc89ABp0 = vmlaq_f32(vacc89ABp0, vi12x89AB, vk12x89AB);
      vaccCDEFp0 = vmlaq_f32(vaccCDEFp0, vi12xCDEF, vk12xCDEF);

      const float32x4_t vi13x0123 = vld1q_f32(i13); i13 += 4;
      const float32x4_t vi13x4567 = vld1q_f32(i13); i13 += 4;
      const float32x4_t vi13x89AB = vld1q_f32(i13); i13 += 4;
      const float32x4_t vi13xCDEF = vld1q_f32(i13); i13 += 4;
      const float32x4_t vk13x0123 = vld1q_f32(w); w += 4;
      const float32x4_t vk13x4567 = vld1q_f32(w); w += 4;
      const float32x4_t vk13x89AB = vld1q_f32(w); w += 4;
      const float32x4_t vk13xCDEF = vld1q_f32(w); w += 4;
      vacc0123p1 = vmlaq_f32(vacc0123p1, vi13x0123, vk13x0123);
      vacc4567p1 = vmlaq_f32(vacc4567p1, vi13x4567, vk13x4567);
      vacc89ABp1 = vmlaq_f32(vacc89ABp1, vi13x89AB, vk13x89AB);
      vaccCDEFp1 = vmlaq_f32(vaccCDEFp1, vi13xCDEF, vk13xCDEF);

      const float32x4_t vi14x0123 = vld1q_f32(i14); i14 += 4;
      const float32x4_t vi14x4567 = vld1q_f32(i14); i14 += 4;
      const float32x4_t vi14x89AB = vld1q_f32(i14); i14 += 4;
      const float32x4_t vi14xCDEF = vld1q_f32(i14); i14 += 4;
      const float32x4_t vk14x0123 = vld1q_f32(w); w += 4;
      const float32x4_t vk14x4567 = vld1q_f32(w); w += 4;
      const float32x4_t vk14x89AB = vld1q_f32(w); w += 4;
      const float32x4_t vk14xCDEF = vld1q_f32(w); w += 4;
      vacc0123p0 = vmlaq_f32(vacc0123p0, vi14x0123, vk14x0123);
      vacc4567p0 = vmlaq_f32(vacc4567p0, vi14x4567, vk14x4567);
      vacc89ABp0 = vmlaq_f32(vacc89ABp0, vi14x89AB, vk14x89AB);
      vaccCDEFp0 = vmlaq_f32(vaccCDEFp0, vi14xCDEF, vk14xCDEF);

      const float32x4_t vi15x0123 = vld1q_f32(i15); i15 += 4;
      const float32x4_t vi15x4567 = vld1q_f32(i15); i15 += 4;
      const float32x4_t vi15x89AB = vld1q_f32(i15); i15 += 4;
      const float32x4_t vi15xCDEF = vld1q_f32(i15); i15 += 4;
      const float32x4_t vk15x0123 = vld1q_f32(w); w += 4;
      const float32x4_t vk15x4567 = vld1q_f32(w); w += 4;
      const float32x4_t vk15x89AB = vld1q_f32(w); w += 4;
      const float32x4_t vk15xCDEF = vld1q_f32(w); w += 4;
      vacc0123p1 = vmlaq_f32(vacc0123p1, vi15x0123, vk15x0123);
      vacc4567p1 = vmlaq_f32(vacc4567p1, vi15x4567, vk15x4567);
      vacc89ABp1 = vmlaq_f32(vacc89ABp1, vi15x89AB, vk15x89AB);
      vaccCDEFp1 = vmlaq_f32(vaccCDEFp1, vi15xCDEF, vk15xCDEF);

      const float32x4_t vi16x0123 = vld1q_f32(i16); i16 += 4;
      const float32x4_t vi16x4567 = vld1q_f32(i16); i16 += 4;
      const float32x4_t vi16x89AB = vld1q_f32(i16); i16 += 4;
      const float32x4_t vi16xCDEF = vld1q_f32(i16); i16 += 4;
      const float32x4_t vk16x0123 = vld1q_f32(w); w += 4;
      const float32x4_t vk16x4567 = vld1q_f32(w); w += 4;
      const float32x4_t vk16x89AB = vld1q_f32(w); w += 4;
      const float32x4_t vk16xCDEF = vld1q_f32(w); w += 4;
      vacc0123p0 = vmlaq_f32(vacc0123p0, vi16x0123, vk16x0123);
      vacc4567p0 = vmlaq_f32(vacc4567p0, vi16x4567, vk16x4567);
      vacc89ABp0 = vmlaq_f32(vacc89ABp0, vi16x89AB, vk16x89AB);
      vaccCDEFp0 = vmlaq_f32(vaccCDEFp0, vi16xCDEF, vk16xCDEF);

      const float32x4_t vi17x0123 = vld1q_f32(i17); i17 += 4;
      const float32x4_t vi17x4567 = vld1q_f32(i17); i17 += 4;
      const float32x4_t vi17x89AB = vld1q_f32(i17); i17 += 4;
      const float32x4_t vi17xCDEF = vld1q_f32(i17); i17 += 4;
      const float32x4_t vk17x0123 = vld1q_f32(w); w += 4;
      const float32x4_t vk17x4567 = vld1q_f32(w); w += 4;
      const float32x4_t vk17x89AB = vld1q_f32(w); w += 4;
      const float32x4_t vk17xCDEF = vld1q_f32(w); w += 4;
      vacc0123p1 = vmlaq_f32(vacc0123p1, vi17x0123, vk17x0123);
      vacc4567p1 = vmlaq_f32(vacc4567p1, vi17x4567, vk17x4567);
      vacc89ABp1 = vmlaq_f32(vacc89ABp1, vi17x89AB, vk17x89AB);
      vaccCDEFp1 = vmlaq_f32(vaccCDEFp1, vi17xCDEF, vk17xCDEF);

      const float32x4_t vi18x0123 = vld1q_f32(i18); i18 += 4;
      const float32x4_t vi18x4567 = vld1q_f32(i18); i18 += 4;
      const float32x4_t vi18x89AB = vld1q_f32(i18); i18 += 4;
      const float32x4_t vi18xCDEF = vld1q_f32(i18); i18 += 4;
      const float32x4_t vk18x0123 = vld1q_f32(w); w += 4;
      const float32x4_t vk18x4567 = vld1q_f32(w); w += 4;
      const float32x4_t vk18x89AB = vld1q_f32(w); w += 4;
      const float32x4_t vk18xCDEF = vld1q_f32(w); w += 4;
      vacc0123p0 = vmlaq_f32(vacc0123p0, vi18x0123, vk18x0123);
      vacc4567p0 = vmlaq_f32(vacc4567p0, vi18x4567, vk18x4567);
      vacc89ABp0 = vmlaq_f32(vacc89ABp0, vi18x89AB, vk18x89AB);
      vaccCDEFp0 = vmlaq_f32(vaccCDEFp0, vi18xCDEF, vk18xCDEF);

      const float32x4_t vi19x0123 = vld1q_f32(i19); i19 += 4;
      const float32x4_t vi19x4567 = vld1q_f32(i19); i19 += 4;
      const float32x4_t vi19x89AB = vld1q_f32(i19); i19 += 4;
      const float32x4_t vi19xCDEF = vld1q_f32(i19); i19 += 4;
      const float32x4_t vk19x0123 = vld1q_f32(w); w += 4;
      const float32x4_t vk19x4567 = vld1q_f32(w); w += 4;
      const float32x4_t vk19x89AB = vld1q_f32(w); w += 4;
      const float32x4_t vk19xCDEF = vld1q_f32(w); w += 4;
      vacc0123p1 = vmlaq_f32(vacc0123p1, vi19x0123, vk19x0123);
      vacc4567p1 = vmlaq_f32(vacc4567p1, vi19x4567, vk19x4567);
      vacc89ABp1 = vmlaq_f32(vacc89ABp1, vi19x89AB, vk19x89AB);
      vaccCDEFp1 = vmlaq_f32(vaccCDEFp1, vi19xCDEF, vk19xCDEF);

      const float32x4_t vi20x0123 = vld1q_f32(i20); i20 += 4;
      const float32x4_t vi20x4567 = vld1q_f32(i20); i20 += 4;
      const float32x4_t vi20x89AB = vld1q_f32(i20); i20 += 4;
      const float32x4_t vi20xCDEF = vld1q_f32(i20); i20 += 4;
      const float32x4_t vk20x0123 = vld1q_f32(w); w += 4;
      const float32x4_t vk20x4567 = vld1q_f32(w); w += 4;
      const float32x4_t vk20x89AB = vld1q_f32(w); w += 4;
      const float32x4_t vk20xCDEF = vld1q_f32(w); w += 4;
      vacc0123p0 = vmlaq_f32(vacc0123p0, vi20x0123, vk20x0123);
      vacc4567p0 = vmlaq_f32(vacc4567p0, vi20x4567, vk20x4567);
      vacc89ABp0 = vmlaq_f32(vacc89ABp0, vi20x89AB, vk20x89AB);
      vaccCDEFp0 = vmlaq_f32(vaccCDEFp0, vi20xCDEF, vk20xCDEF);

      const float32x4_t vi21x0123 = vld1q_f32(i21); i21 += 4;
      const float32x4_t vi21x4567 = vld1q_f32(i21); i21 += 4;
      const float32x4_t vi21x89AB = vld1q_f32(i21); i21 += 4;
      const float32x4_t vi21xCDEF = vld1q_f32(i21); i21 += 4;
      const float32x4_t vk21x0123 = vld1q_f32(w); w += 4;
      const float32x4_t vk21x4567 = vld1q_f32(w); w += 4;
      const float32x4_t vk21x89AB = vld1q_f32(w); w += 4;
      const float32x4_t vk21xCDEF = vld1q_f32(w); w += 4;
      vacc0123p1 = vmlaq_f32(vacc0123p1, vi21x0123, vk21x0123);
      vacc4567p1 = vmlaq_f32(vacc4567p1, vi21x4567, vk21x4567);
      vacc89ABp1 = vmlaq_f32(vacc89ABp1, vi21x89AB, vk21x89AB);
      vaccCDEFp1 = vmlaq_f32(vaccCDEFp1, vi21xCDEF, vk21xCDEF);

      const float32x4_t vi22x0123 = vld1q_f32(i22); i22 += 4;
      const float32x4_t vi22x4567 = vld1q_f32(i22); i22 += 4;
      const float32x4_t vi22x89AB = vld1q_f32(i22); i22 += 4;
      const float32x4_t vi22xCDEF = vld1q_f32(i22); i22 += 4;
      const float32x4_t vk22x0123 = vld1q_f32(w); w += 4;
      const float32x4_t vk22x4567 = vld1q_f32(w); w += 4;
      const float32x4_t vk22x89AB = vld1q_f32(w); w += 4;
      const float32x4_t vk22xCDEF = vld1q_f32(w); w += 4;
      vacc0123p0 = vmlaq_f32(vacc0123p0, vi22x0123, vk22x0123);
      vacc4567p0 = vmlaq_f32(vacc4567p0, vi22x4567, vk22x4567);
      vacc89ABp0 = vmlaq_f32(vacc89ABp0, vi22x89AB, vk22x89AB);
      vaccCDEFp0 = vmlaq_f32(vaccCDEFp0, vi22xCDEF, vk22xCDEF);

      const float32x4_t vi23x0123 = vld1q_f32(i23); i23 += 4;
      const float32x4_t vi23x4567 = vld1q_f32(i23); i23 += 4;
      const float32x4_t vi23x89AB = vld1q_f32(i23); i23 += 4;
      const float32x4_t vi23xCDEF = vld1q_f32(i23); i23 += 4;
      const float32x4_t vk23x0123 = vld1q_f32(w); w += 4;
      const float32x4_t vk23x4567 = vld1q_f32(w); w += 4;
      const float32x4_t vk23x89AB = vld1q_f32(w); w += 4;
      const float32x4_t vk23xCDEF = vld1q_f32(w); w += 4;
      vacc0123p1 = vmlaq_f32(vacc0123p1, vi23x0123, vk23x0123);
      vacc4567p1 = vmlaq_f32(vacc4567p1, vi23x4567, vk23x4567);
      vacc89ABp1 = vmlaq_f32(vacc89ABp1, vi23x89AB, vk23x89AB);
      vaccCDEFp1 = vmlaq_f32(vaccCDEFp1, vi23xCDEF, vk23xCDEF);

      const float32x4_t vi24x0123 = vld1q_f32(i24); i24 += 4;
      const float32x4_t vi24x4567 = vld1q_f32(i24); i24 += 4;
      const float32x4_t vi24x89AB = vld1q_f32(i24); i24 += 4;
      const float32x4_t vi24xCDEF = vld1q_f32(i24); i24 += 4;
      const float32x4_t vk24x0123 = vld1q_f32(w); w += 4;
      const float32x4_t vk24x4567 = vld1q_f32(w); w += 4;
      const float32x4_t vk24x89AB = vld1q_f32(w); w += 4;
      const float32x4_t vk24xCDEF = vld1q_f32(w); w += 4;
      vacc0123p0 = vmlaq_f32(vacc0123p0, vi24x0123, vk24x0123);
      vacc4567p0 = vmlaq_f32(vacc4567p0, vi24x4567, vk24x4567);
      vacc89ABp0 = vmlaq_f32(vacc89ABp0, vi24x89AB, vk24x89AB);
      vaccCDEFp0 = vmlaq_f32(vaccCDEFp0, vi24xCDEF, vk24xCDEF);

      // Add up all accumulators to vacc0123456789ABCDEFp0
      vacc0123p0 = vaddq_f32(vacc0123p0, vacc0123p1);
      vacc4567p0 = vaddq_f32(vacc4567p0, vacc4567p1);
      vacc89ABp0 = vaddq_f32(vacc89ABp0, vacc89ABp1);
      vaccCDEFp0 = vaddq_f32(vaccCDEFp0, vaccCDEFp1);

      float32x4_t vacc0123 = vmaxq_f32(vacc0123p0, vmin);
      float32x4_t vacc4567 = vmaxq_f32(vacc4567p0, vmin);
      float32x4_t vacc89AB = vmaxq_f32(vacc89ABp0, vmin);
      float32x4_t vaccCDEF = vmaxq_f32(vaccCDEFp0, vmin);
      vacc0123 = vminq_f32(vacc0123, vmax);
      vacc4567 = vminq_f32(vacc4567, vmax);
      vacc89AB = vminq_f32(vacc89AB, vmax);
      vaccCDEF = vminq_f32(vaccCDEF, vmax);

      vst1q_f32(output, vacc0123); output += 4;
      vst1q_f32(output, vacc4567); output += 4;
      vst1q_f32(output, vacc89AB); output += 4;
      vst1q_f32(output, vaccCDEF); output += 4;
    }
    for (; c >= 4; c -= 4) {
      float32x4_t vacc0123p0 = vld1q_f32(w); w += 4;


      const float32x4_t vi0x0123 = vld1q_f32(i0); i0 += 4;
      const float32x4_t vk0x0123 = vld1q_f32(w + 12);
      vacc0123p0 = vmlaq_f32(vacc0123p0, vi0x0123, vk0x0123);

      const float32x4_t vi1x0123 = vld1q_f32(i1); i1 += 4;
      const float32x4_t vk1x0123 = vld1q_f32(w + 28);
      float32x4_t vacc0123p1 = vmulq_f32(vi1x0123, vk1x0123);

      const float32x4_t vi2x0123 = vld1q_f32(i2); i2 += 4;
      const float32x4_t vk2x0123 = vld1q_f32(w + 44);
      vacc0123p0 = vmlaq_f32(vacc0123p0, vi2x0123, vk2x0123);

      const float32x4_t vi3x0123 = vld1q_f32(i3); i3 += 4;
      const float32x4_t vk3x0123 = vld1q_f32(w + 60);
      vacc0123p1 = vmlaq_f32(vacc0123p1, vi3x0123, vk3x0123);

      const float32x4_t vi4x0123 = vld1q_f32(i4); i4 += 4;
      const float32x4_t vk4x0123 = vld1q_f32(w + 76);
      vacc0123p0 = vmlaq_f32(vacc0123p0, vi4x0123, vk4x0123);

      const float32x4_t vi5x0123 = vld1q_f32(i5); i5 += 4;
      const float32x4_t vk5x0123 = vld1q_f32(w + 92);
      vacc0123p1 = vmlaq_f32(vacc0123p1, vi5x0123, vk5x0123);

      const float32x4_t vi6x0123 = vld1q_f32(i6); i6 += 4;
      const float32x4_t vk6x0123 = vld1q_f32(w + 108);
      vacc0123p0 = vmlaq_f32(vacc0123p0, vi6x0123, vk6x0123);

      const float32x4_t vi7x0123 = vld1q_f32(i7); i7 += 4;
      const float32x4_t vk7x0123 = vld1q_f32(w + 124);
      vacc0123p1 = vmlaq_f32(vacc0123p1, vi7x0123, vk7x0123);

      const float32x4_t vi8x0123 = vld1q_f32(i8); i8 += 4;
      const float32x4_t vk8x0123 = vld1q_f32(w + 140);
      vacc0123p0 = vmlaq_f32(vacc0123p0, vi8x0123, vk8x0123);

      const float32x4_t vi9x0123 = vld1q_f32(i9); i9 += 4;
      const float32x4_t vk9x0123 = vld1q_f32(w + 156);
      vacc0123p1 = vmlaq_f32(vacc0123p1, vi9x0123, vk9x0123);

      const float32x4_t vi10x0123 = vld1q_f32(i10); i10 += 4;
      const float32x4_t vk10x0123 = vld1q_f32(w + 172);
      vacc0123p0 = vmlaq_f32(vacc0123p0, vi10x0123, vk10x0123);

      const float32x4_t vi11x0123 = vld1q_f32(i11); i11 += 4;
      const float32x4_t vk11x0123 = vld1q_f32(w + 188);
      vacc0123p1 = vmlaq_f32(vacc0123p1, vi11x0123, vk11x0123);

      const float32x4_t vi12x0123 = vld1q_f32(i12); i12 += 4;
      const float32x4_t vk12x0123 = vld1q_f32(w + 204);
      vacc0123p0 = vmlaq_f32(vacc0123p0, vi12x0123, vk12x0123);

      const float32x4_t vi13x0123 = vld1q_f32(i13); i13 += 4;
      const float32x4_t vk13x0123 = vld1q_f32(w + 220);
      vacc0123p1 = vmlaq_f32(vacc0123p1, vi13x0123, vk13x0123);

      const float32x4_t vi14x0123 = vld1q_f32(i14); i14 += 4;
      const float32x4_t vk14x0123 = vld1q_f32(w + 236);
      vacc0123p0 = vmlaq_f32(vacc0123p0, vi14x0123, vk14x0123);

      const float32x4_t vi15x0123 = vld1q_f32(i15); i15 += 4;
      const float32x4_t vk15x0123 = vld1q_f32(w + 252);
      vacc0123p1 = vmlaq_f32(vacc0123p1, vi15x0123, vk15x0123);

      const float32x4_t vi16x0123 = vld1q_f32(i16); i16 += 4;
      const float32x4_t vk16x0123 = vld1q_f32(w + 268);
      vacc0123p0 = vmlaq_f32(vacc0123p0, vi16x0123, vk16x0123);

      const float32x4_t vi17x0123 = vld1q_f32(i17); i17 += 4;
      const float32x4_t vk17x0123 = vld1q_f32(w + 284);
      vacc0123p1 = vmlaq_f32(vacc0123p1, vi17x0123, vk17x0123);

      const float32x4_t vi18x0123 = vld1q_f32(i18); i18 += 4;
      const float32x4_t vk18x0123 = vld1q_f32(w + 300);
      vacc0123p0 = vmlaq_f32(vacc0123p0, vi18x0123, vk18x0123);

      const float32x4_t vi19x0123 = vld1q_f32(i19); i19 += 4;
      const float32x4_t vk19x0123 = vld1q_f32(w + 316);
      vacc0123p1 = vmlaq_f32(vacc0123p1, vi19x0123, vk19x0123);

      const float32x4_t vi20x0123 = vld1q_f32(i20); i20 += 4;
      const float32x4_t vk20x0123 = vld1q_f32(w + 332);
      vacc0123p0 = vmlaq_f32(vacc0123p0, vi20x0123, vk20x0123);

      const float32x4_t vi21x0123 = vld1q_f32(i21); i21 += 4;
      const float32x4_t vk21x0123 = vld1q_f32(w + 348);
      vacc0123p1 = vmlaq_f32(vacc0123p1, vi21x0123, vk21x0123);

      const float32x4_t vi22x0123 = vld1q_f32(i22); i22 += 4;
      const float32x4_t vk22x0123 = vld1q_f32(w + 364);
      vacc0123p0 = vmlaq_f32(vacc0123p0, vi22x0123, vk22x0123);

      const float32x4_t vi23x0123 = vld1q_f32(i23); i23 += 4;
      const float32x4_t vk23x0123 = vld1q_f32(w + 380);
      vacc0123p1 = vmlaq_f32(vacc0123p1, vi23x0123, vk23x0123);

      const float32x4_t vi24x0123 = vld1q_f32(i24); i24 += 4;
      const float32x4_t vk24x0123 = vld1q_f32(w + 396);
      vacc0123p0 = vmlaq_f32(vacc0123p0, vi24x0123, vk24x0123);

      // Add up all accumulators to vacc0123p0
      vacc0123p0 = vaddq_f32(vacc0123p0, vacc0123p1);

      float32x4_t vacc0123 = vmaxq_f32(vacc0123p0, vmin);
      vacc0123 = vminq_f32(vacc0123, vmax);

      vst1q_f32(output, vacc0123); output += 4;
    }
    if XNN_UNLIKELY(c != 0) {
      float32x4_t vacc0123p0 = vld1q_f32(w);


      const float32x4_t vi0x0123 = vld1q_f32(i0);
      const float32x4_t vk0x0123 = vld1q_f32(w + 16);
      vacc0123p0 = vmlaq_f32(vacc0123p0, vi0x0123, vk0x0123);

      const float32x4_t vi1x0123 = vld1q_f32(i1);
      const float32x4_t vk1x0123 = vld1q_f32(w + 32);
      float32x4_t vacc0123p1 = vmulq_f32(vi1x0123, vk1x0123);

      const float32x4_t vi2x0123 = vld1q_f32(i2);
      const float32x4_t vk2x0123 = vld1q_f32(w + 48);
      vacc0123p0 = vmlaq_f32(vacc0123p0, vi2x0123, vk2x0123);

      const float32x4_t vi3x0123 = vld1q_f32(i3);
      const float32x4_t vk3x0123 = vld1q_f32(w + 64);
      vacc0123p1 = vmlaq_f32(vacc0123p1, vi3x0123, vk3x0123);

      const float32x4_t vi4x0123 = vld1q_f32(i4);
      const float32x4_t vk4x0123 = vld1q_f32(w + 80);
      vacc0123p0 = vmlaq_f32(vacc0123p0, vi4x0123, vk4x0123);

      const float32x4_t vi5x0123 = vld1q_f32(i5);
      const float32x4_t vk5x0123 = vld1q_f32(w + 96);
      vacc0123p1 = vmlaq_f32(vacc0123p1, vi5x0123, vk5x0123);

      const float32x4_t vi6x0123 = vld1q_f32(i6);
      const float32x4_t vk6x0123 = vld1q_f32(w + 112);
      vacc0123p0 = vmlaq_f32(vacc0123p0, vi6x0123, vk6x0123);

      const float32x4_t vi7x0123 = vld1q_f32(i7);
      const float32x4_t vk7x0123 = vld1q_f32(w + 128);
      vacc0123p1 = vmlaq_f32(vacc0123p1, vi7x0123, vk7x0123);

      const float32x4_t vi8x0123 = vld1q_f32(i8);
      const float32x4_t vk8x0123 = vld1q_f32(w + 144);
      vacc0123p0 = vmlaq_f32(vacc0123p0, vi8x0123, vk8x0123);

      const float32x4_t vi9x0123 = vld1q_f32(i9);
      const float32x4_t vk9x0123 = vld1q_f32(w + 160);
      vacc0123p1 = vmlaq_f32(vacc0123p1, vi9x0123, vk9x0123);

      const float32x4_t vi10x0123 = vld1q_f32(i10);
      const float32x4_t vk10x0123 = vld1q_f32(w + 176);
      vacc0123p0 = vmlaq_f32(vacc0123p0, vi10x0123, vk10x0123);

      const float32x4_t vi11x0123 = vld1q_f32(i11);
      const float32x4_t vk11x0123 = vld1q_f32(w + 192);
      vacc0123p1 = vmlaq_f32(vacc0123p1, vi11x0123, vk11x0123);

      const float32x4_t vi12x0123 = vld1q_f32(i12);
      const float32x4_t vk12x0123 = vld1q_f32(w + 208);
      vacc0123p0 = vmlaq_f32(vacc0123p0, vi12x0123, vk12x0123);

      const float32x4_t vi13x0123 = vld1q_f32(i13);
      const float32x4_t vk13x0123 = vld1q_f32(w + 224);
      vacc0123p1 = vmlaq_f32(vacc0123p1, vi13x0123, vk13x0123);

      const float32x4_t vi14x0123 = vld1q_f32(i14);
      const float32x4_t vk14x0123 = vld1q_f32(w + 240);
      vacc0123p0 = vmlaq_f32(vacc0123p0, vi14x0123, vk14x0123);

      const float32x4_t vi15x0123 = vld1q_f32(i15);
      const float32x4_t vk15x0123 = vld1q_f32(w + 256);
      vacc0123p1 = vmlaq_f32(vacc0123p1, vi15x0123, vk15x0123);

      const float32x4_t vi16x0123 = vld1q_f32(i16);
      const float32x4_t vk16x0123 = vld1q_f32(w + 272);
      vacc0123p0 = vmlaq_f32(vacc0123p0, vi16x0123, vk16x0123);

      const float32x4_t vi17x0123 = vld1q_f32(i17);
      const float32x4_t vk17x0123 = vld1q_f32(w + 288);
      vacc0123p1 = vmlaq_f32(vacc0123p1, vi17x0123, vk17x0123);

      const float32x4_t vi18x0123 = vld1q_f32(i18);
      const float32x4_t vk18x0123 = vld1q_f32(w + 304);
      vacc0123p0 = vmlaq_f32(vacc0123p0, vi18x0123, vk18x0123);

      const float32x4_t vi19x0123 = vld1q_f32(i19);
      const float32x4_t vk19x0123 = vld1q_f32(w + 320);
      vacc0123p1 = vmlaq_f32(vacc0123p1, vi19x0123, vk19x0123);

      const float32x4_t vi20x0123 = vld1q_f32(i20);
      const float32x4_t vk20x0123 = vld1q_f32(w + 336);
      vacc0123p0 = vmlaq_f32(vacc0123p0, vi20x0123, vk20x0123);

      const float32x4_t vi21x0123 = vld1q_f32(i21);
      const float32x4_t vk21x0123 = vld1q_f32(w + 352);
      vacc0123p1 = vmlaq_f32(vacc0123p1, vi21x0123, vk21x0123);

      const float32x4_t vi22x0123 = vld1q_f32(i22);
      const float32x4_t vk22x0123 = vld1q_f32(w + 368);
      vacc0123p0 = vmlaq_f32(vacc0123p0, vi22x0123, vk22x0123);

      const float32x4_t vi23x0123 = vld1q_f32(i23);
      const float32x4_t vk23x0123 = vld1q_f32(w + 384);
      vacc0123p1 = vmlaq_f32(vacc0123p1, vi23x0123, vk23x0123);

      const float32x4_t vi24x0123 = vld1q_f32(i24);
      const float32x4_t vk24x0123 = vld1q_f32(w + 400);
      vacc0123p0 = vmlaq_f32(vacc0123p0, vi24x0123, vk24x0123);

      // Add up all accumulators to vacc0123p0
      vacc0123p0 = vaddq_f32(vacc0123p0, vacc0123p1);

      float32x4_t vacc0123 = vmaxq_f32(vacc0123p0, vmin);
      vacc0123 = vminq_f32(vacc0123, vmax);

      float32x2_t vacc01 = vget_low_f32(vacc0123);
      if (c & 2) {
        vst1_f32(output, vacc01); output += 2;
        vacc01 = vget_high_f32(vacc0123);
      }
      if (c & 1) {
        vst1_lane_f32(output, vacc01, 0); output += 1;
      }
    }

    output = (float*) ((uintptr_t) output + output_increment);
  } while (--output_width != 0);
}
