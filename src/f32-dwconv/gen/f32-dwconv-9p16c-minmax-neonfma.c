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


void xnn_f32_dwconv_minmax_ukernel_9p16c__neonfma(
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
      vacc0123p0 = vfmaq_f32(vacc0123p0, vi0x0123, vk0x0123);
      vacc4567p0 = vfmaq_f32(vacc4567p0, vi0x4567, vk0x4567);
      vacc89ABp0 = vfmaq_f32(vacc89ABp0, vi0x89AB, vk0x89AB);
      vaccCDEFp0 = vfmaq_f32(vaccCDEFp0, vi0xCDEF, vk0xCDEF);

      const float32x4_t vi1x0123 = vld1q_f32(i1); i1 += 4;
      const float32x4_t vi1x4567 = vld1q_f32(i1); i1 += 4;
      const float32x4_t vi1x89AB = vld1q_f32(i1); i1 += 4;
      const float32x4_t vi1xCDEF = vld1q_f32(i1); i1 += 4;
      const float32x4_t vk1x0123 = vld1q_f32(w); w += 4;
      const float32x4_t vk1x4567 = vld1q_f32(w); w += 4;
      const float32x4_t vk1x89AB = vld1q_f32(w); w += 4;
      const float32x4_t vk1xCDEF = vld1q_f32(w); w += 4;
      vacc0123p0 = vfmaq_f32(vacc0123p0, vi1x0123, vk1x0123);
      vacc4567p0 = vfmaq_f32(vacc4567p0, vi1x4567, vk1x4567);
      vacc89ABp0 = vfmaq_f32(vacc89ABp0, vi1x89AB, vk1x89AB);
      vaccCDEFp0 = vfmaq_f32(vaccCDEFp0, vi1xCDEF, vk1xCDEF);

      const float32x4_t vi2x0123 = vld1q_f32(i2); i2 += 4;
      const float32x4_t vi2x4567 = vld1q_f32(i2); i2 += 4;
      const float32x4_t vi2x89AB = vld1q_f32(i2); i2 += 4;
      const float32x4_t vi2xCDEF = vld1q_f32(i2); i2 += 4;
      const float32x4_t vk2x0123 = vld1q_f32(w); w += 4;
      const float32x4_t vk2x4567 = vld1q_f32(w); w += 4;
      const float32x4_t vk2x89AB = vld1q_f32(w); w += 4;
      const float32x4_t vk2xCDEF = vld1q_f32(w); w += 4;
      vacc0123p0 = vfmaq_f32(vacc0123p0, vi2x0123, vk2x0123);
      vacc4567p0 = vfmaq_f32(vacc4567p0, vi2x4567, vk2x4567);
      vacc89ABp0 = vfmaq_f32(vacc89ABp0, vi2x89AB, vk2x89AB);
      vaccCDEFp0 = vfmaq_f32(vaccCDEFp0, vi2xCDEF, vk2xCDEF);

      const float32x4_t vi3x0123 = vld1q_f32(i3); i3 += 4;
      const float32x4_t vi3x4567 = vld1q_f32(i3); i3 += 4;
      const float32x4_t vi3x89AB = vld1q_f32(i3); i3 += 4;
      const float32x4_t vi3xCDEF = vld1q_f32(i3); i3 += 4;
      const float32x4_t vk3x0123 = vld1q_f32(w); w += 4;
      const float32x4_t vk3x4567 = vld1q_f32(w); w += 4;
      const float32x4_t vk3x89AB = vld1q_f32(w); w += 4;
      const float32x4_t vk3xCDEF = vld1q_f32(w); w += 4;
      vacc0123p0 = vfmaq_f32(vacc0123p0, vi3x0123, vk3x0123);
      vacc4567p0 = vfmaq_f32(vacc4567p0, vi3x4567, vk3x4567);
      vacc89ABp0 = vfmaq_f32(vacc89ABp0, vi3x89AB, vk3x89AB);
      vaccCDEFp0 = vfmaq_f32(vaccCDEFp0, vi3xCDEF, vk3xCDEF);

      const float32x4_t vi4x0123 = vld1q_f32(i4); i4 += 4;
      const float32x4_t vi4x4567 = vld1q_f32(i4); i4 += 4;
      const float32x4_t vi4x89AB = vld1q_f32(i4); i4 += 4;
      const float32x4_t vi4xCDEF = vld1q_f32(i4); i4 += 4;
      const float32x4_t vk4x0123 = vld1q_f32(w); w += 4;
      const float32x4_t vk4x4567 = vld1q_f32(w); w += 4;
      const float32x4_t vk4x89AB = vld1q_f32(w); w += 4;
      const float32x4_t vk4xCDEF = vld1q_f32(w); w += 4;
      vacc0123p0 = vfmaq_f32(vacc0123p0, vi4x0123, vk4x0123);
      vacc4567p0 = vfmaq_f32(vacc4567p0, vi4x4567, vk4x4567);
      vacc89ABp0 = vfmaq_f32(vacc89ABp0, vi4x89AB, vk4x89AB);
      vaccCDEFp0 = vfmaq_f32(vaccCDEFp0, vi4xCDEF, vk4xCDEF);

      const float32x4_t vi5x0123 = vld1q_f32(i5); i5 += 4;
      const float32x4_t vi5x4567 = vld1q_f32(i5); i5 += 4;
      const float32x4_t vi5x89AB = vld1q_f32(i5); i5 += 4;
      const float32x4_t vi5xCDEF = vld1q_f32(i5); i5 += 4;
      const float32x4_t vk5x0123 = vld1q_f32(w); w += 4;
      const float32x4_t vk5x4567 = vld1q_f32(w); w += 4;
      const float32x4_t vk5x89AB = vld1q_f32(w); w += 4;
      const float32x4_t vk5xCDEF = vld1q_f32(w); w += 4;
      vacc0123p0 = vfmaq_f32(vacc0123p0, vi5x0123, vk5x0123);
      vacc4567p0 = vfmaq_f32(vacc4567p0, vi5x4567, vk5x4567);
      vacc89ABp0 = vfmaq_f32(vacc89ABp0, vi5x89AB, vk5x89AB);
      vaccCDEFp0 = vfmaq_f32(vaccCDEFp0, vi5xCDEF, vk5xCDEF);

      const float32x4_t vi6x0123 = vld1q_f32(i6); i6 += 4;
      const float32x4_t vi6x4567 = vld1q_f32(i6); i6 += 4;
      const float32x4_t vi6x89AB = vld1q_f32(i6); i6 += 4;
      const float32x4_t vi6xCDEF = vld1q_f32(i6); i6 += 4;
      const float32x4_t vk6x0123 = vld1q_f32(w); w += 4;
      const float32x4_t vk6x4567 = vld1q_f32(w); w += 4;
      const float32x4_t vk6x89AB = vld1q_f32(w); w += 4;
      const float32x4_t vk6xCDEF = vld1q_f32(w); w += 4;
      vacc0123p0 = vfmaq_f32(vacc0123p0, vi6x0123, vk6x0123);
      vacc4567p0 = vfmaq_f32(vacc4567p0, vi6x4567, vk6x4567);
      vacc89ABp0 = vfmaq_f32(vacc89ABp0, vi6x89AB, vk6x89AB);
      vaccCDEFp0 = vfmaq_f32(vaccCDEFp0, vi6xCDEF, vk6xCDEF);

      const float32x4_t vi7x0123 = vld1q_f32(i7); i7 += 4;
      const float32x4_t vi7x4567 = vld1q_f32(i7); i7 += 4;
      const float32x4_t vi7x89AB = vld1q_f32(i7); i7 += 4;
      const float32x4_t vi7xCDEF = vld1q_f32(i7); i7 += 4;
      const float32x4_t vk7x0123 = vld1q_f32(w); w += 4;
      const float32x4_t vk7x4567 = vld1q_f32(w); w += 4;
      const float32x4_t vk7x89AB = vld1q_f32(w); w += 4;
      const float32x4_t vk7xCDEF = vld1q_f32(w); w += 4;
      vacc0123p0 = vfmaq_f32(vacc0123p0, vi7x0123, vk7x0123);
      vacc4567p0 = vfmaq_f32(vacc4567p0, vi7x4567, vk7x4567);
      vacc89ABp0 = vfmaq_f32(vacc89ABp0, vi7x89AB, vk7x89AB);
      vaccCDEFp0 = vfmaq_f32(vaccCDEFp0, vi7xCDEF, vk7xCDEF);

      const float32x4_t vi8x0123 = vld1q_f32(i8); i8 += 4;
      const float32x4_t vi8x4567 = vld1q_f32(i8); i8 += 4;
      const float32x4_t vi8x89AB = vld1q_f32(i8); i8 += 4;
      const float32x4_t vi8xCDEF = vld1q_f32(i8); i8 += 4;
      const float32x4_t vk8x0123 = vld1q_f32(w); w += 4;
      const float32x4_t vk8x4567 = vld1q_f32(w); w += 4;
      const float32x4_t vk8x89AB = vld1q_f32(w); w += 4;
      const float32x4_t vk8xCDEF = vld1q_f32(w); w += 4;
      vacc0123p0 = vfmaq_f32(vacc0123p0, vi8x0123, vk8x0123);
      vacc4567p0 = vfmaq_f32(vacc4567p0, vi8x4567, vk8x4567);
      vacc89ABp0 = vfmaq_f32(vacc89ABp0, vi8x89AB, vk8x89AB);
      vaccCDEFp0 = vfmaq_f32(vaccCDEFp0, vi8xCDEF, vk8xCDEF);


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
      vacc0123p0 = vfmaq_f32(vacc0123p0, vi0x0123, vk0x0123);

      const float32x4_t vi1x0123 = vld1q_f32(i1); i1 += 4;
      const float32x4_t vk1x0123 = vld1q_f32(w + 28);
      vacc0123p0 = vfmaq_f32(vacc0123p0, vi1x0123, vk1x0123);

      const float32x4_t vi2x0123 = vld1q_f32(i2); i2 += 4;
      const float32x4_t vk2x0123 = vld1q_f32(w + 44);
      vacc0123p0 = vfmaq_f32(vacc0123p0, vi2x0123, vk2x0123);

      const float32x4_t vi3x0123 = vld1q_f32(i3); i3 += 4;
      const float32x4_t vk3x0123 = vld1q_f32(w + 60);
      vacc0123p0 = vfmaq_f32(vacc0123p0, vi3x0123, vk3x0123);

      const float32x4_t vi4x0123 = vld1q_f32(i4); i4 += 4;
      const float32x4_t vk4x0123 = vld1q_f32(w + 76);
      vacc0123p0 = vfmaq_f32(vacc0123p0, vi4x0123, vk4x0123);

      const float32x4_t vi5x0123 = vld1q_f32(i5); i5 += 4;
      const float32x4_t vk5x0123 = vld1q_f32(w + 92);
      vacc0123p0 = vfmaq_f32(vacc0123p0, vi5x0123, vk5x0123);

      const float32x4_t vi6x0123 = vld1q_f32(i6); i6 += 4;
      const float32x4_t vk6x0123 = vld1q_f32(w + 108);
      vacc0123p0 = vfmaq_f32(vacc0123p0, vi6x0123, vk6x0123);

      const float32x4_t vi7x0123 = vld1q_f32(i7); i7 += 4;
      const float32x4_t vk7x0123 = vld1q_f32(w + 124);
      vacc0123p0 = vfmaq_f32(vacc0123p0, vi7x0123, vk7x0123);

      const float32x4_t vi8x0123 = vld1q_f32(i8); i8 += 4;
      const float32x4_t vk8x0123 = vld1q_f32(w + 140);
      vacc0123p0 = vfmaq_f32(vacc0123p0, vi8x0123, vk8x0123);


      float32x4_t vacc0123 = vmaxq_f32(vacc0123p0, vmin);
      vacc0123 = vminq_f32(vacc0123, vmax);

      vst1q_f32(output, vacc0123); output += 4;
    }
    if XNN_UNLIKELY(c != 0) {
      float32x4_t vacc0123p0 = vld1q_f32(w);


      const float32x4_t vi0x0123 = vld1q_f32(i0);
      const float32x4_t vk0x0123 = vld1q_f32(w + 16);
      vacc0123p0 = vfmaq_f32(vacc0123p0, vi0x0123, vk0x0123);

      const float32x4_t vi1x0123 = vld1q_f32(i1);
      const float32x4_t vk1x0123 = vld1q_f32(w + 32);
      vacc0123p0 = vfmaq_f32(vacc0123p0, vi1x0123, vk1x0123);

      const float32x4_t vi2x0123 = vld1q_f32(i2);
      const float32x4_t vk2x0123 = vld1q_f32(w + 48);
      vacc0123p0 = vfmaq_f32(vacc0123p0, vi2x0123, vk2x0123);

      const float32x4_t vi3x0123 = vld1q_f32(i3);
      const float32x4_t vk3x0123 = vld1q_f32(w + 64);
      vacc0123p0 = vfmaq_f32(vacc0123p0, vi3x0123, vk3x0123);

      const float32x4_t vi4x0123 = vld1q_f32(i4);
      const float32x4_t vk4x0123 = vld1q_f32(w + 80);
      vacc0123p0 = vfmaq_f32(vacc0123p0, vi4x0123, vk4x0123);

      const float32x4_t vi5x0123 = vld1q_f32(i5);
      const float32x4_t vk5x0123 = vld1q_f32(w + 96);
      vacc0123p0 = vfmaq_f32(vacc0123p0, vi5x0123, vk5x0123);

      const float32x4_t vi6x0123 = vld1q_f32(i6);
      const float32x4_t vk6x0123 = vld1q_f32(w + 112);
      vacc0123p0 = vfmaq_f32(vacc0123p0, vi6x0123, vk6x0123);

      const float32x4_t vi7x0123 = vld1q_f32(i7);
      const float32x4_t vk7x0123 = vld1q_f32(w + 128);
      vacc0123p0 = vfmaq_f32(vacc0123p0, vi7x0123, vk7x0123);

      const float32x4_t vi8x0123 = vld1q_f32(i8);
      const float32x4_t vk8x0123 = vld1q_f32(w + 144);
      vacc0123p0 = vfmaq_f32(vacc0123p0, vi8x0123, vk8x0123);


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
