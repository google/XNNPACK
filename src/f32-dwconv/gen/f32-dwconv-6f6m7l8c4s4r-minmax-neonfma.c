// Auto-generated file. Do not edit!
//   Template: src/f32-dwconv/multipass-neon.c.in
//   Generator: tools/xngen
//
// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include <arm_neon.h>

#include "xnnpack/dwconv.h"
#include "xnnpack/math.h"


void xnn_f32_dwconv_minmax_ukernel_6f6m7l8c4s4r__neonfma(
    size_t channels,
    size_t output_width,
    const float** input,
    const float* weights,
    float* output,
    intptr_t input_stride,
    size_t output_increment,
    size_t input_offset,
    const float* zero,
    size_t kernel_size,
    float* buffer,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(channels != 0);
  assert(output_width != 0);
  assert(kernel_size > 6);

  const float32x4_t vmax = vld1q_dup_f32(&params->scalar.max);
  const float32x4_t vmin = vld1q_dup_f32(&params->scalar.min);
  do {
    const float* w = weights;

    // First pass to process 6 inputs.
    {
      float* b = buffer;
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
      input += 6;

      // Process c channels and write to buffer.
      size_t c = round_up_po2(channels, 4);
      for (; c >= 8; c -= 8) {
        float32x4_t vacc0123p0 = vld1q_f32(w); w += 4;
        float32x4_t vacc4567p0 = vld1q_f32(w); w += 4;


        const float32x4_t vi0x0123 = vld1q_f32(i0); i0 += 4;
        const float32x4_t vi0x4567 = vld1q_f32(i0); i0 += 4;

        const float32x4_t vk0x0123 = vld1q_f32(w); w += 4;
        const float32x4_t vk0x4567 = vld1q_f32(w); w += 4;
        vacc0123p0 = vfmaq_f32(vacc0123p0, vi0x0123, vk0x0123);
        vacc4567p0 = vfmaq_f32(vacc4567p0, vi0x4567, vk0x4567);

        const float32x4_t vi1x0123 = vld1q_f32(i1); i1 += 4;
        const float32x4_t vi1x4567 = vld1q_f32(i1); i1 += 4;

        const float32x4_t vk1x0123 = vld1q_f32(w); w += 4;
        const float32x4_t vk1x4567 = vld1q_f32(w); w += 4;
        vacc0123p0 = vfmaq_f32(vacc0123p0, vi1x0123, vk1x0123);
        vacc4567p0 = vfmaq_f32(vacc4567p0, vi1x4567, vk1x4567);

        const float32x4_t vi2x0123 = vld1q_f32(i2); i2 += 4;
        const float32x4_t vi2x4567 = vld1q_f32(i2); i2 += 4;

        const float32x4_t vk2x0123 = vld1q_f32(w); w += 4;
        const float32x4_t vk2x4567 = vld1q_f32(w); w += 4;
        vacc0123p0 = vfmaq_f32(vacc0123p0, vi2x0123, vk2x0123);
        vacc4567p0 = vfmaq_f32(vacc4567p0, vi2x4567, vk2x4567);

        const float32x4_t vi3x0123 = vld1q_f32(i3); i3 += 4;
        const float32x4_t vi3x4567 = vld1q_f32(i3); i3 += 4;

        const float32x4_t vk3x0123 = vld1q_f32(w); w += 4;
        const float32x4_t vk3x4567 = vld1q_f32(w); w += 4;
        vacc0123p0 = vfmaq_f32(vacc0123p0, vi3x0123, vk3x0123);
        vacc4567p0 = vfmaq_f32(vacc4567p0, vi3x4567, vk3x4567);

        const float32x4_t vi4x0123 = vld1q_f32(i4); i4 += 4;
        const float32x4_t vi4x4567 = vld1q_f32(i4); i4 += 4;

        const float32x4_t vk4x0123 = vld1q_f32(w); w += 4;
        const float32x4_t vk4x4567 = vld1q_f32(w); w += 4;
        vacc0123p0 = vfmaq_f32(vacc0123p0, vi4x0123, vk4x0123);
        vacc4567p0 = vfmaq_f32(vacc4567p0, vi4x4567, vk4x4567);

        const float32x4_t vi5x0123 = vld1q_f32(i5); i5 += 4;
        const float32x4_t vi5x4567 = vld1q_f32(i5); i5 += 4;

        const float32x4_t vk5x0123 = vld1q_f32(w); w += 4;
        const float32x4_t vk5x4567 = vld1q_f32(w); w += 4;
        vacc0123p0 = vfmaq_f32(vacc0123p0, vi5x0123, vk5x0123);
        vacc4567p0 = vfmaq_f32(vacc4567p0, vi5x4567, vk5x4567);


        vst1q_f32(b, vacc0123p0); b += 4;
        vst1q_f32(b, vacc4567p0); b += 4;
      }

      if (c != 0) {
        float32x4_t vacc0123p0 = vld1q_f32(w); w += 4;


        const float32x4_t vi0x0123 = vld1q_f32(i0); i0 += 4;

        const float32x4_t vk0x0123 = vld1q_f32(w); w += 4;
        vacc0123p0 = vfmaq_f32(vacc0123p0, vi0x0123, vk0x0123);

        const float32x4_t vi1x0123 = vld1q_f32(i1); i1 += 4;

        const float32x4_t vk1x0123 = vld1q_f32(w); w += 4;
        vacc0123p0 = vfmaq_f32(vacc0123p0, vi1x0123, vk1x0123);

        const float32x4_t vi2x0123 = vld1q_f32(i2); i2 += 4;

        const float32x4_t vk2x0123 = vld1q_f32(w); w += 4;
        vacc0123p0 = vfmaq_f32(vacc0123p0, vi2x0123, vk2x0123);

        const float32x4_t vi3x0123 = vld1q_f32(i3); i3 += 4;

        const float32x4_t vk3x0123 = vld1q_f32(w); w += 4;
        vacc0123p0 = vfmaq_f32(vacc0123p0, vi3x0123, vk3x0123);

        const float32x4_t vi4x0123 = vld1q_f32(i4); i4 += 4;

        const float32x4_t vk4x0123 = vld1q_f32(w); w += 4;
        vacc0123p0 = vfmaq_f32(vacc0123p0, vi4x0123, vk4x0123);

        const float32x4_t vi5x0123 = vld1q_f32(i5); i5 += 4;

        const float32x4_t vk5x0123 = vld1q_f32(w); w += 4;
        vacc0123p0 = vfmaq_f32(vacc0123p0, vi5x0123, vk5x0123);


        vst1q_f32(b, vacc0123p0); b += 4;
      }
    }

    // Middle pass to process 6 inputs in each iteration.
    for (size_t ks = kernel_size - 6; ks > 7; ks -= 6) {
      float* b = buffer;
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
      input += 6;

      size_t c = round_up_po2(channels, 4);
      for (; c >= 8; c -= 8) {
        float32x4_t vacc0123p0 = vld1q_f32(b);
        float32x4_t vacc4567p0 = vld1q_f32(b + 4);


        const float32x4_t vi0x0123 = vld1q_f32(i0); i0 += 4;
        const float32x4_t vi0x4567 = vld1q_f32(i0); i0 += 4;

        const float32x4_t vk0x0123 = vld1q_f32(w); w += 4;
        const float32x4_t vk0x4567 = vld1q_f32(w); w += 4;
        vacc0123p0 = vfmaq_f32(vacc0123p0, vi0x0123, vk0x0123);
        vacc4567p0 = vfmaq_f32(vacc4567p0, vi0x4567, vk0x4567);

        const float32x4_t vi1x0123 = vld1q_f32(i1); i1 += 4;
        const float32x4_t vi1x4567 = vld1q_f32(i1); i1 += 4;

        const float32x4_t vk1x0123 = vld1q_f32(w); w += 4;
        const float32x4_t vk1x4567 = vld1q_f32(w); w += 4;
        vacc0123p0 = vfmaq_f32(vacc0123p0, vi1x0123, vk1x0123);
        vacc4567p0 = vfmaq_f32(vacc4567p0, vi1x4567, vk1x4567);

        const float32x4_t vi2x0123 = vld1q_f32(i2); i2 += 4;
        const float32x4_t vi2x4567 = vld1q_f32(i2); i2 += 4;

        const float32x4_t vk2x0123 = vld1q_f32(w); w += 4;
        const float32x4_t vk2x4567 = vld1q_f32(w); w += 4;
        vacc0123p0 = vfmaq_f32(vacc0123p0, vi2x0123, vk2x0123);
        vacc4567p0 = vfmaq_f32(vacc4567p0, vi2x4567, vk2x4567);

        const float32x4_t vi3x0123 = vld1q_f32(i3); i3 += 4;
        const float32x4_t vi3x4567 = vld1q_f32(i3); i3 += 4;

        const float32x4_t vk3x0123 = vld1q_f32(w); w += 4;
        const float32x4_t vk3x4567 = vld1q_f32(w); w += 4;
        vacc0123p0 = vfmaq_f32(vacc0123p0, vi3x0123, vk3x0123);
        vacc4567p0 = vfmaq_f32(vacc4567p0, vi3x4567, vk3x4567);

        const float32x4_t vi4x0123 = vld1q_f32(i4); i4 += 4;
        const float32x4_t vi4x4567 = vld1q_f32(i4); i4 += 4;

        const float32x4_t vk4x0123 = vld1q_f32(w); w += 4;
        const float32x4_t vk4x4567 = vld1q_f32(w); w += 4;
        vacc0123p0 = vfmaq_f32(vacc0123p0, vi4x0123, vk4x0123);
        vacc4567p0 = vfmaq_f32(vacc4567p0, vi4x4567, vk4x4567);

        const float32x4_t vi5x0123 = vld1q_f32(i5); i5 += 4;
        const float32x4_t vi5x4567 = vld1q_f32(i5); i5 += 4;

        const float32x4_t vk5x0123 = vld1q_f32(w); w += 4;
        const float32x4_t vk5x4567 = vld1q_f32(w); w += 4;
        vacc0123p0 = vfmaq_f32(vacc0123p0, vi5x0123, vk5x0123);
        vacc4567p0 = vfmaq_f32(vacc4567p0, vi5x4567, vk5x4567);


        vst1q_f32(b, vacc0123p0); b += 4;
        vst1q_f32(b, vacc4567p0); b += 4;
      }

      if (c != 0) {
        float32x4_t vacc0123p0 = vld1q_f32(b);


        const float32x4_t vi0x0123 = vld1q_f32(i0); i0 += 4;

        const float32x4_t vk0x0123 = vld1q_f32(w); w += 4;
        vacc0123p0 = vfmaq_f32(vacc0123p0, vi0x0123, vk0x0123);

        const float32x4_t vi1x0123 = vld1q_f32(i1); i1 += 4;

        const float32x4_t vk1x0123 = vld1q_f32(w); w += 4;
        vacc0123p0 = vfmaq_f32(vacc0123p0, vi1x0123, vk1x0123);

        const float32x4_t vi2x0123 = vld1q_f32(i2); i2 += 4;

        const float32x4_t vk2x0123 = vld1q_f32(w); w += 4;
        vacc0123p0 = vfmaq_f32(vacc0123p0, vi2x0123, vk2x0123);

        const float32x4_t vi3x0123 = vld1q_f32(i3); i3 += 4;

        const float32x4_t vk3x0123 = vld1q_f32(w); w += 4;
        vacc0123p0 = vfmaq_f32(vacc0123p0, vi3x0123, vk3x0123);

        const float32x4_t vi4x0123 = vld1q_f32(i4); i4 += 4;

        const float32x4_t vk4x0123 = vld1q_f32(w); w += 4;
        vacc0123p0 = vfmaq_f32(vacc0123p0, vi4x0123, vk4x0123);

        const float32x4_t vi5x0123 = vld1q_f32(i5); i5 += 4;

        const float32x4_t vk5x0123 = vld1q_f32(w); w += 4;
        vacc0123p0 = vfmaq_f32(vacc0123p0, vi5x0123, vk5x0123);


        vst1q_f32(b, vacc0123p0); b += 4;
      }
    }

    // Last pass to process up to 7 inputs.
    {
      float* b = buffer;
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

      size_t c = channels;
      for (; c >= 8; c -= 8) {
        float32x4_t vacc0123p0 = vld1q_f32(b); b += 4;
        float32x4_t vacc4567p0 = vld1q_f32(b); b += 4;


        const float32x4_t vi0x0123 = vld1q_f32(i0); i0 += 4;
        const float32x4_t vi0x4567 = vld1q_f32(i0); i0 += 4;

        float32x4_t vk0x0123 = vld1q_f32(w); w += 4;
        float32x4_t vk0x4567 = vld1q_f32(w); w += 4;

        vacc0123p0 = vfmaq_f32(vacc0123p0, vi0x0123, vk0x0123);
        vacc4567p0 = vfmaq_f32(vacc4567p0, vi0x4567, vk0x4567);

        const float32x4_t vi1x0123 = vld1q_f32(i1); i1 += 4;
        const float32x4_t vi1x4567 = vld1q_f32(i1); i1 += 4;

        float32x4_t vk1x0123 = vld1q_f32(w); w += 4;
        float32x4_t vk1x4567 = vld1q_f32(w); w += 4;

        vacc0123p0 = vfmaq_f32(vacc0123p0, vi1x0123, vk1x0123);
        vacc4567p0 = vfmaq_f32(vacc4567p0, vi1x4567, vk1x4567);

        const float32x4_t vi2x0123 = vld1q_f32(i2); i2 += 4;
        const float32x4_t vi2x4567 = vld1q_f32(i2); i2 += 4;

        float32x4_t vk2x0123 = vld1q_f32(w); w += 4;
        float32x4_t vk2x4567 = vld1q_f32(w); w += 4;

        vacc0123p0 = vfmaq_f32(vacc0123p0, vi2x0123, vk2x0123);
        vacc4567p0 = vfmaq_f32(vacc4567p0, vi2x4567, vk2x4567);

        const float32x4_t vi3x0123 = vld1q_f32(i3); i3 += 4;
        const float32x4_t vi3x4567 = vld1q_f32(i3); i3 += 4;

        float32x4_t vk3x0123 = vld1q_f32(w); w += 4;
        float32x4_t vk3x4567 = vld1q_f32(w); w += 4;

        vacc0123p0 = vfmaq_f32(vacc0123p0, vi3x0123, vk3x0123);
        vacc4567p0 = vfmaq_f32(vacc4567p0, vi3x4567, vk3x4567);

        const float32x4_t vi4x0123 = vld1q_f32(i4); i4 += 4;
        const float32x4_t vi4x4567 = vld1q_f32(i4); i4 += 4;

        float32x4_t vk4x0123 = vld1q_f32(w); w += 4;
        float32x4_t vk4x4567 = vld1q_f32(w); w += 4;

        vacc0123p0 = vfmaq_f32(vacc0123p0, vi4x0123, vk4x0123);
        vacc4567p0 = vfmaq_f32(vacc4567p0, vi4x4567, vk4x4567);

        const float32x4_t vi5x0123 = vld1q_f32(i5); i5 += 4;
        const float32x4_t vi5x4567 = vld1q_f32(i5); i5 += 4;

        float32x4_t vk5x0123 = vld1q_f32(w); w += 4;
        float32x4_t vk5x4567 = vld1q_f32(w); w += 4;

        vacc0123p0 = vfmaq_f32(vacc0123p0, vi5x0123, vk5x0123);
        vacc4567p0 = vfmaq_f32(vacc4567p0, vi5x4567, vk5x4567);

        const float32x4_t vi6x0123 = vld1q_f32(i6); i6 += 4;
        const float32x4_t vi6x4567 = vld1q_f32(i6); i6 += 4;

        float32x4_t vk6x0123 = vld1q_f32(w); w += 4;
        float32x4_t vk6x4567 = vld1q_f32(w); w += 4;

        vacc0123p0 = vfmaq_f32(vacc0123p0, vi6x0123, vk6x0123);
        vacc4567p0 = vfmaq_f32(vacc4567p0, vi6x4567, vk6x4567);


        float32x4_t vacc0123 = vmaxq_f32(vacc0123p0, vmin);
        float32x4_t vacc4567 = vmaxq_f32(vacc4567p0, vmin);

        vacc0123 = vminq_f32(vacc0123, vmax);
        vacc4567 = vminq_f32(vacc4567, vmax);

        vst1q_f32(output, vacc0123); output += 4;
        vst1q_f32(output, vacc4567); output += 4;
      }


      for (; c >= 4; c -= 4) {
        float32x4_t vacc0123p0 = vld1q_f32(b); b += 4;


        const float32x4_t vi0x0123 = vld1q_f32(i0); i0 += 4;

        float32x4_t vk0x0123 = vld1q_f32(w); w += 4;

        vacc0123p0 = vfmaq_f32(vacc0123p0, vi0x0123, vk0x0123);

        const float32x4_t vi1x0123 = vld1q_f32(i1); i1 += 4;

        float32x4_t vk1x0123 = vld1q_f32(w); w += 4;

        vacc0123p0 = vfmaq_f32(vacc0123p0, vi1x0123, vk1x0123);

        const float32x4_t vi2x0123 = vld1q_f32(i2); i2 += 4;

        float32x4_t vk2x0123 = vld1q_f32(w); w += 4;

        vacc0123p0 = vfmaq_f32(vacc0123p0, vi2x0123, vk2x0123);

        const float32x4_t vi3x0123 = vld1q_f32(i3); i3 += 4;

        float32x4_t vk3x0123 = vld1q_f32(w); w += 4;

        vacc0123p0 = vfmaq_f32(vacc0123p0, vi3x0123, vk3x0123);

        const float32x4_t vi4x0123 = vld1q_f32(i4); i4 += 4;

        float32x4_t vk4x0123 = vld1q_f32(w); w += 4;

        vacc0123p0 = vfmaq_f32(vacc0123p0, vi4x0123, vk4x0123);

        const float32x4_t vi5x0123 = vld1q_f32(i5); i5 += 4;

        float32x4_t vk5x0123 = vld1q_f32(w); w += 4;

        vacc0123p0 = vfmaq_f32(vacc0123p0, vi5x0123, vk5x0123);

        const float32x4_t vi6x0123 = vld1q_f32(i6); i6 += 4;

        float32x4_t vk6x0123 = vld1q_f32(w); w += 4;

        vacc0123p0 = vfmaq_f32(vacc0123p0, vi6x0123, vk6x0123);



        float32x4_t vacc0123 = vmaxq_f32(vacc0123p0, vmin);

        vacc0123 = vminq_f32(vacc0123, vmax);

        vst1q_f32(output, vacc0123); output += 4;
      }

      if XNN_UNLIKELY(c != 0) {
        float32x4_t vacc0123p0 = vld1q_f32(b);

        const float32x4_t vi0x0123 = vld1q_f32(i0);
        float32x4_t vk0x0123 = vld1q_f32(w); w += 4;
        vacc0123p0 = vfmaq_f32(vacc0123p0, vi0x0123, vk0x0123);

        const float32x4_t vi1x0123 = vld1q_f32(i1);
        float32x4_t vk1x0123 = vld1q_f32(w); w += 4;
        vacc0123p0 = vfmaq_f32(vacc0123p0, vi1x0123, vk1x0123);

        const float32x4_t vi2x0123 = vld1q_f32(i2);
        float32x4_t vk2x0123 = vld1q_f32(w); w += 4;
        vacc0123p0 = vfmaq_f32(vacc0123p0, vi2x0123, vk2x0123);

        const float32x4_t vi3x0123 = vld1q_f32(i3);
        float32x4_t vk3x0123 = vld1q_f32(w); w += 4;
        vacc0123p0 = vfmaq_f32(vacc0123p0, vi3x0123, vk3x0123);

        const float32x4_t vi4x0123 = vld1q_f32(i4);
        float32x4_t vk4x0123 = vld1q_f32(w); w += 4;
        vacc0123p0 = vfmaq_f32(vacc0123p0, vi4x0123, vk4x0123);

        const float32x4_t vi5x0123 = vld1q_f32(i5);
        float32x4_t vk5x0123 = vld1q_f32(w); w += 4;
        vacc0123p0 = vfmaq_f32(vacc0123p0, vi5x0123, vk5x0123);

        const float32x4_t vi6x0123 = vld1q_f32(i6);
        float32x4_t vk6x0123 = vld1q_f32(w); w += 4;
        vacc0123p0 = vfmaq_f32(vacc0123p0, vi6x0123, vk6x0123);


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

    }
    input = (const float**) ((uintptr_t) input + input_stride);
    output = (float*) ((uintptr_t) output + output_increment);
  } while (--output_width != 0);
}
