// Auto-generated file. Do not edit!
//   Template: src/qs8-dwconv/multipass-scalar.c.in
//   Generator: tools/xngen
//
// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <math.h>

#include "xnnpack/dwconv.h"
#include "xnnpack/math.h"
#include "xnnpack/unaligned.h"


void xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_8f8m9l2c1s1r__scalar_lrintf(
    size_t channels,
    size_t output_width,
    const int8_t** input,
    const void* weights,
    int8_t* output,
    intptr_t input_stride,
    size_t output_increment,
    size_t input_offset,
    const int8_t* zero,
    size_t kernel_size,
    int32_t* buffer,
    const union xnn_qs8_qc8w_conv_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(channels != 0);
  assert(output_width != 0);
  assert(kernel_size > 8);

  const float voutput_min_less_zero_point = params->fp32_scalar_lrintf.output_min_less_zero_point;
  const float voutput_max_less_zero_point = params->fp32_scalar_lrintf.output_max_less_zero_point;
  const int32_t voutput_zero_point = params->fp32_scalar_lrintf.output_zero_point;
  do {
    const void* w = weights;

    // First pass to process 8 inputs.
    {
      int32_t* b = buffer;
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
      input += 8;

      size_t c = channels;
      for (; c >= 2; c -= 2) {
        int32_t vacc0 = unaligned_indexed_load_s32(w, 0);
        int32_t vacc1 = unaligned_indexed_load_s32(w, 1);

        const int32_t vi0x0 = (int32_t) i0[0];
        const int32_t vi0x1 = (int32_t) i0[1];
        i0 += 2;

        const int32_t vk0x0 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[0];
        const int32_t vk0x1 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[1];

        vacc0 += vi0x0 * vk0x0;
        vacc1 += vi0x1 * vk0x1;

        const int32_t vi1x0 = (int32_t) i1[0];
        const int32_t vi1x1 = (int32_t) i1[1];
        i1 += 2;

        const int32_t vk1x0 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[2];
        const int32_t vk1x1 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[3];

        vacc0 += vi1x0 * vk1x0;
        vacc1 += vi1x1 * vk1x1;

        const int32_t vi2x0 = (int32_t) i2[0];
        const int32_t vi2x1 = (int32_t) i2[1];
        i2 += 2;

        const int32_t vk2x0 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[4];
        const int32_t vk2x1 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[5];

        vacc0 += vi2x0 * vk2x0;
        vacc1 += vi2x1 * vk2x1;

        const int32_t vi3x0 = (int32_t) i3[0];
        const int32_t vi3x1 = (int32_t) i3[1];
        i3 += 2;

        const int32_t vk3x0 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[6];
        const int32_t vk3x1 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[7];

        vacc0 += vi3x0 * vk3x0;
        vacc1 += vi3x1 * vk3x1;

        const int32_t vi4x0 = (int32_t) i4[0];
        const int32_t vi4x1 = (int32_t) i4[1];
        i4 += 2;

        const int32_t vk4x0 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[8];
        const int32_t vk4x1 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[9];

        vacc0 += vi4x0 * vk4x0;
        vacc1 += vi4x1 * vk4x1;

        const int32_t vi5x0 = (int32_t) i5[0];
        const int32_t vi5x1 = (int32_t) i5[1];
        i5 += 2;

        const int32_t vk5x0 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[10];
        const int32_t vk5x1 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[11];

        vacc0 += vi5x0 * vk5x0;
        vacc1 += vi5x1 * vk5x1;

        const int32_t vi6x0 = (int32_t) i6[0];
        const int32_t vi6x1 = (int32_t) i6[1];
        i6 += 2;

        const int32_t vk6x0 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[12];
        const int32_t vk6x1 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[13];

        vacc0 += vi6x0 * vk6x0;
        vacc1 += vi6x1 * vk6x1;

        const int32_t vi7x0 = (int32_t) i7[0];
        const int32_t vi7x1 = (int32_t) i7[1];
        i7 += 2;

        const int32_t vk7x0 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[14];
        const int32_t vk7x1 = (int32_t) ((const int8_t*) ((uintptr_t) w + 2 * sizeof(int32_t)))[15];

        vacc0 += vi7x0 * vk7x0;
        vacc1 += vi7x1 * vk7x1;

        w = (const void*) ((uintptr_t) w + 2 * sizeof(int32_t) + 16 * sizeof(int8_t));
        b[0] = vacc0;
        b[1] = vacc1;
        b += 2;
      }
      if XNN_UNLIKELY(c != 0) {
        int32_t vacc = unaligned_load_s32(w);

        const int32_t vi0 = (int32_t) *i0;
        const int32_t vk0 = (int32_t) ((const int8_t*) ((uintptr_t) w + 1 * sizeof(int32_t)))[0];
        vacc += vi0 * vk0;
        const int32_t vi1 = (int32_t) *i1;
        const int32_t vk1 = (int32_t) ((const int8_t*) ((uintptr_t) w + 1 * sizeof(int32_t)))[1];
        vacc += vi1 * vk1;
        const int32_t vi2 = (int32_t) *i2;
        const int32_t vk2 = (int32_t) ((const int8_t*) ((uintptr_t) w + 1 * sizeof(int32_t)))[2];
        vacc += vi2 * vk2;
        const int32_t vi3 = (int32_t) *i3;
        const int32_t vk3 = (int32_t) ((const int8_t*) ((uintptr_t) w + 1 * sizeof(int32_t)))[3];
        vacc += vi3 * vk3;
        const int32_t vi4 = (int32_t) *i4;
        const int32_t vk4 = (int32_t) ((const int8_t*) ((uintptr_t) w + 1 * sizeof(int32_t)))[4];
        vacc += vi4 * vk4;
        const int32_t vi5 = (int32_t) *i5;
        const int32_t vk5 = (int32_t) ((const int8_t*) ((uintptr_t) w + 1 * sizeof(int32_t)))[5];
        vacc += vi5 * vk5;
        const int32_t vi6 = (int32_t) *i6;
        const int32_t vk6 = (int32_t) ((const int8_t*) ((uintptr_t) w + 1 * sizeof(int32_t)))[6];
        vacc += vi6 * vk6;
        const int32_t vi7 = (int32_t) *i7;
        const int32_t vk7 = (int32_t) ((const int8_t*) ((uintptr_t) w + 1 * sizeof(int32_t)))[7];
        vacc += vi7 * vk7;

        w = (const void*) ((uintptr_t) w + 1 * sizeof(int32_t) + 8 * sizeof(int8_t));
        *b++ = vacc;
      }
    }

    // Middle pass to process 8 inputs in each iteration.
    for (size_t ks = kernel_size - 8; ks > 9; ks -= 8) {
      int32_t* b = buffer;
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
      input += 8;

      size_t c = channels;
      for (; c >= 2; c -= 2) {
        int32_t vacc0 = b[0];
        int32_t vacc1 = b[1];

        const int32_t vi0x0 = (int32_t) i0[0];
        const int32_t vi0x1 = (int32_t) i0[1];
        i0 += 2;

        const int32_t vk0x0 = (int32_t) ((const int8_t*) w)[0];
        const int32_t vk0x1 = (int32_t) ((const int8_t*) w)[1];

        vacc0 += vi0x0 * vk0x0;
        vacc1 += vi0x1 * vk0x1;

        const int32_t vi1x0 = (int32_t) i1[0];
        const int32_t vi1x1 = (int32_t) i1[1];
        i1 += 2;

        const int32_t vk1x0 = (int32_t) ((const int8_t*) w)[2];
        const int32_t vk1x1 = (int32_t) ((const int8_t*) w)[3];

        vacc0 += vi1x0 * vk1x0;
        vacc1 += vi1x1 * vk1x1;

        const int32_t vi2x0 = (int32_t) i2[0];
        const int32_t vi2x1 = (int32_t) i2[1];
        i2 += 2;

        const int32_t vk2x0 = (int32_t) ((const int8_t*) w)[4];
        const int32_t vk2x1 = (int32_t) ((const int8_t*) w)[5];

        vacc0 += vi2x0 * vk2x0;
        vacc1 += vi2x1 * vk2x1;

        const int32_t vi3x0 = (int32_t) i3[0];
        const int32_t vi3x1 = (int32_t) i3[1];
        i3 += 2;

        const int32_t vk3x0 = (int32_t) ((const int8_t*) w)[6];
        const int32_t vk3x1 = (int32_t) ((const int8_t*) w)[7];

        vacc0 += vi3x0 * vk3x0;
        vacc1 += vi3x1 * vk3x1;

        const int32_t vi4x0 = (int32_t) i4[0];
        const int32_t vi4x1 = (int32_t) i4[1];
        i4 += 2;

        const int32_t vk4x0 = (int32_t) ((const int8_t*) w)[8];
        const int32_t vk4x1 = (int32_t) ((const int8_t*) w)[9];

        vacc0 += vi4x0 * vk4x0;
        vacc1 += vi4x1 * vk4x1;

        const int32_t vi5x0 = (int32_t) i5[0];
        const int32_t vi5x1 = (int32_t) i5[1];
        i5 += 2;

        const int32_t vk5x0 = (int32_t) ((const int8_t*) w)[10];
        const int32_t vk5x1 = (int32_t) ((const int8_t*) w)[11];

        vacc0 += vi5x0 * vk5x0;
        vacc1 += vi5x1 * vk5x1;

        const int32_t vi6x0 = (int32_t) i6[0];
        const int32_t vi6x1 = (int32_t) i6[1];
        i6 += 2;

        const int32_t vk6x0 = (int32_t) ((const int8_t*) w)[12];
        const int32_t vk6x1 = (int32_t) ((const int8_t*) w)[13];

        vacc0 += vi6x0 * vk6x0;
        vacc1 += vi6x1 * vk6x1;

        const int32_t vi7x0 = (int32_t) i7[0];
        const int32_t vi7x1 = (int32_t) i7[1];
        i7 += 2;

        const int32_t vk7x0 = (int32_t) ((const int8_t*) w)[14];
        const int32_t vk7x1 = (int32_t) ((const int8_t*) w)[15];

        vacc0 += vi7x0 * vk7x0;
        vacc1 += vi7x1 * vk7x1;

        w = (const void*) ((uintptr_t) w + 16 * sizeof(int8_t));
        b[0] = vacc0;
        b[1] = vacc1;
        b += 2;
      }
      if XNN_UNLIKELY(c != 0) {
        int32_t vacc = b[0];

        const int32_t vi0 = (int32_t) *i0;
        const int32_t vk0 = (int32_t) ((const int8_t*) w)[0];
        vacc += vi0 * vk0;
        const int32_t vi1 = (int32_t) *i1;
        const int32_t vk1 = (int32_t) ((const int8_t*) w)[1];
        vacc += vi1 * vk1;
        const int32_t vi2 = (int32_t) *i2;
        const int32_t vk2 = (int32_t) ((const int8_t*) w)[2];
        vacc += vi2 * vk2;
        const int32_t vi3 = (int32_t) *i3;
        const int32_t vk3 = (int32_t) ((const int8_t*) w)[3];
        vacc += vi3 * vk3;
        const int32_t vi4 = (int32_t) *i4;
        const int32_t vk4 = (int32_t) ((const int8_t*) w)[4];
        vacc += vi4 * vk4;
        const int32_t vi5 = (int32_t) *i5;
        const int32_t vk5 = (int32_t) ((const int8_t*) w)[5];
        vacc += vi5 * vk5;
        const int32_t vi6 = (int32_t) *i6;
        const int32_t vk6 = (int32_t) ((const int8_t*) w)[6];
        vacc += vi6 * vk6;
        const int32_t vi7 = (int32_t) *i7;
        const int32_t vk7 = (int32_t) ((const int8_t*) w)[7];
        vacc += vi7 * vk7;
        w = (const void*) ((uintptr_t) w + 8 * sizeof(int8_t));
        *b++ = vacc;
      }
    }

    // Last pass to process up to 9 inputs.
    {
      const int32_t* b = buffer;
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

      size_t c = channels;
      for (; c >= 2; c -= 2) {
        int32_t vacc0 = b[0];
        int32_t vacc1 = b[1];
        b += 2;

        const int32_t vi0x0 = (int32_t) i0[0];
        const int32_t vi0x1 = (int32_t) i0[1];
        i0 += 2;

        const int32_t vk0x0 = (int32_t) ((const int8_t*) w)[0];
        const int32_t vk0x1 = (int32_t) ((const int8_t*) w)[1];

        vacc0 += vi0x0 * vk0x0;
        vacc1 += vi0x1 * vk0x1;

        const int32_t vi1x0 = (int32_t) i1[0];
        const int32_t vi1x1 = (int32_t) i1[1];
        i1 += 2;

        const int32_t vk1x0 = (int32_t) ((const int8_t*) w)[2];
        const int32_t vk1x1 = (int32_t) ((const int8_t*) w)[3];

        vacc0 += vi1x0 * vk1x0;
        vacc1 += vi1x1 * vk1x1;

        const int32_t vi2x0 = (int32_t) i2[0];
        const int32_t vi2x1 = (int32_t) i2[1];
        i2 += 2;

        const int32_t vk2x0 = (int32_t) ((const int8_t*) w)[4];
        const int32_t vk2x1 = (int32_t) ((const int8_t*) w)[5];

        vacc0 += vi2x0 * vk2x0;
        vacc1 += vi2x1 * vk2x1;

        const int32_t vi3x0 = (int32_t) i3[0];
        const int32_t vi3x1 = (int32_t) i3[1];
        i3 += 2;

        const int32_t vk3x0 = (int32_t) ((const int8_t*) w)[6];
        const int32_t vk3x1 = (int32_t) ((const int8_t*) w)[7];

        vacc0 += vi3x0 * vk3x0;
        vacc1 += vi3x1 * vk3x1;

        const int32_t vi4x0 = (int32_t) i4[0];
        const int32_t vi4x1 = (int32_t) i4[1];
        i4 += 2;

        const int32_t vk4x0 = (int32_t) ((const int8_t*) w)[8];
        const int32_t vk4x1 = (int32_t) ((const int8_t*) w)[9];

        vacc0 += vi4x0 * vk4x0;
        vacc1 += vi4x1 * vk4x1;

        const int32_t vi5x0 = (int32_t) i5[0];
        const int32_t vi5x1 = (int32_t) i5[1];
        i5 += 2;

        const int32_t vk5x0 = (int32_t) ((const int8_t*) w)[10];
        const int32_t vk5x1 = (int32_t) ((const int8_t*) w)[11];

        vacc0 += vi5x0 * vk5x0;
        vacc1 += vi5x1 * vk5x1;

        const int32_t vi6x0 = (int32_t) i6[0];
        const int32_t vi6x1 = (int32_t) i6[1];
        i6 += 2;

        const int32_t vk6x0 = (int32_t) ((const int8_t*) w)[12];
        const int32_t vk6x1 = (int32_t) ((const int8_t*) w)[13];

        vacc0 += vi6x0 * vk6x0;
        vacc1 += vi6x1 * vk6x1;

        const int32_t vi7x0 = (int32_t) i7[0];
        const int32_t vi7x1 = (int32_t) i7[1];
        i7 += 2;

        const int32_t vk7x0 = (int32_t) ((const int8_t*) w)[14];
        const int32_t vk7x1 = (int32_t) ((const int8_t*) w)[15];

        vacc0 += vi7x0 * vk7x0;
        vacc1 += vi7x1 * vk7x1;

        const int32_t vi8x0 = (int32_t) i8[0];
        const int32_t vi8x1 = (int32_t) i8[1];
        i8 += 2;

        const int32_t vk8x0 = (int32_t) ((const int8_t*) w)[16];
        const int32_t vk8x1 = (int32_t) ((const int8_t*) w)[17];

        vacc0 += vi8x0 * vk8x0;
        vacc1 += vi8x1 * vk8x1;

        w = (const void*) ((uintptr_t) w + 18 * sizeof(int8_t));

        float vfpacc0 = (float) vacc0;
        float vfpacc1 = (float) vacc1;

        const float vscale0 = unaligned_indexed_load_f32(w, 0);
        const float vscale1 = unaligned_indexed_load_f32(w, 1);
        w = (const void*) ((const float*) w + 2);

        vfpacc0 *= vscale0;
        vfpacc1 *= vscale1;

        vfpacc0 = math_max_f32(vfpacc0, voutput_min_less_zero_point);
        vfpacc1 = math_max_f32(vfpacc1, voutput_min_less_zero_point);

        vfpacc0 = math_min_f32(vfpacc0, voutput_max_less_zero_point);
        vfpacc1 = math_min_f32(vfpacc1, voutput_max_less_zero_point);

        const int32_t vrndacc0 = (int32_t) lrintf(vfpacc0);
        const int32_t vrndacc1 = (int32_t) lrintf(vfpacc1);

        int32_t vout0 = (int32_t) vrndacc0 + voutput_zero_point;
        int32_t vout1 = (int32_t) vrndacc1 + voutput_zero_point;

        output[0] = (int8_t) vout0;
        output[1] = (int8_t) vout1;
        output += 2;
      }
      if XNN_UNLIKELY(c != 0) {
        int32_t vacc = b[0];

        const int32_t vi0 = (int32_t) *i0;
        const int32_t vk0 = (int32_t) ((const int8_t*) w)[0];
        vacc += vi0 * vk0;
        const int32_t vi1 = (int32_t) *i1;
        const int32_t vk1 = (int32_t) ((const int8_t*) w)[1];
        vacc += vi1 * vk1;
        const int32_t vi2 = (int32_t) *i2;
        const int32_t vk2 = (int32_t) ((const int8_t*) w)[2];
        vacc += vi2 * vk2;
        const int32_t vi3 = (int32_t) *i3;
        const int32_t vk3 = (int32_t) ((const int8_t*) w)[3];
        vacc += vi3 * vk3;
        const int32_t vi4 = (int32_t) *i4;
        const int32_t vk4 = (int32_t) ((const int8_t*) w)[4];
        vacc += vi4 * vk4;
        const int32_t vi5 = (int32_t) *i5;
        const int32_t vk5 = (int32_t) ((const int8_t*) w)[5];
        vacc += vi5 * vk5;
        const int32_t vi6 = (int32_t) *i6;
        const int32_t vk6 = (int32_t) ((const int8_t*) w)[6];
        vacc += vi6 * vk6;
        const int32_t vi7 = (int32_t) *i7;
        const int32_t vk7 = (int32_t) ((const int8_t*) w)[7];
        vacc += vi7 * vk7;
        const int32_t vi8 = (int32_t) *i8;
        const int32_t vk8 = (int32_t) ((const int8_t*) w)[8];
        vacc += vi8 * vk8;

        const float vscale = unaligned_load_f32((const void*) ((uintptr_t) w + 9 * sizeof(int8_t)));
        float vfpacc = (float) vacc * vscale;

        vfpacc = math_max_f32(vfpacc, voutput_min_less_zero_point);
        vfpacc = math_min_f32(vfpacc, voutput_max_less_zero_point);
        const int32_t vrndacc = (int32_t) lrintf(vfpacc);
        int32_t vout = vrndacc + voutput_zero_point;

        *output++ = (int8_t) vout;
      }
    }

    input = (const int8_t**) ((uintptr_t) input + input_stride);
    output = (int8_t*) ((uintptr_t) output + output_increment);
  } while (--output_width != 0);
}
