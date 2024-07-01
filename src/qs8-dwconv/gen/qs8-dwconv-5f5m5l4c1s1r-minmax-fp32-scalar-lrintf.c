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


void xnn_qs8_dwconv_minmax_fp32_ukernel_5f5m5l4c1s1r__scalar_lrintf(
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
    const union xnn_qs8_conv_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(channels != 0);
  assert(output_width != 0);
  assert(kernel_size > 5);

  const float vscale = params->fp32_scalar_lrintf.scale;
  const float voutput_min_less_zero_point = params->fp32_scalar_lrintf.output_min_less_zero_point;
  const float voutput_max_less_zero_point = params->fp32_scalar_lrintf.output_max_less_zero_point;
  const int32_t voutput_zero_point = params->fp32_scalar_lrintf.output_zero_point;
  do {
    const void* w = weights;

    // First pass to process 5 inputs.
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
      input += 5;

      size_t c = channels;
      for (; c >= 4; c -= 4) {
        int32_t vacc0 = ((const int32_t*) w)[0];
        int32_t vacc1 = ((const int32_t*) w)[1];
        int32_t vacc2 = ((const int32_t*) w)[2];
        int32_t vacc3 = ((const int32_t*) w)[3];

        const int32_t vi0x0 = (int32_t) i0[0];
        const int32_t vi0x1 = (int32_t) i0[1];
        const int32_t vi0x2 = (int32_t) i0[2];
        const int32_t vi0x3 = (int32_t) i0[3];
        i0 += 4;

        const int32_t vk0x0 = (int32_t) ((const int8_t*) ((uintptr_t) w + 4 * sizeof(int32_t)))[0];
        const int32_t vk0x1 = (int32_t) ((const int8_t*) ((uintptr_t) w + 4 * sizeof(int32_t)))[1];
        const int32_t vk0x2 = (int32_t) ((const int8_t*) ((uintptr_t) w + 4 * sizeof(int32_t)))[2];
        const int32_t vk0x3 = (int32_t) ((const int8_t*) ((uintptr_t) w + 4 * sizeof(int32_t)))[3];

        vacc0 += vi0x0 * vk0x0;
        vacc1 += vi0x1 * vk0x1;
        vacc2 += vi0x2 * vk0x2;
        vacc3 += vi0x3 * vk0x3;

        const int32_t vi1x0 = (int32_t) i1[0];
        const int32_t vi1x1 = (int32_t) i1[1];
        const int32_t vi1x2 = (int32_t) i1[2];
        const int32_t vi1x3 = (int32_t) i1[3];
        i1 += 4;

        const int32_t vk1x0 = (int32_t) ((const int8_t*) ((uintptr_t) w + 4 * sizeof(int32_t)))[4];
        const int32_t vk1x1 = (int32_t) ((const int8_t*) ((uintptr_t) w + 4 * sizeof(int32_t)))[5];
        const int32_t vk1x2 = (int32_t) ((const int8_t*) ((uintptr_t) w + 4 * sizeof(int32_t)))[6];
        const int32_t vk1x3 = (int32_t) ((const int8_t*) ((uintptr_t) w + 4 * sizeof(int32_t)))[7];

        vacc0 += vi1x0 * vk1x0;
        vacc1 += vi1x1 * vk1x1;
        vacc2 += vi1x2 * vk1x2;
        vacc3 += vi1x3 * vk1x3;

        const int32_t vi2x0 = (int32_t) i2[0];
        const int32_t vi2x1 = (int32_t) i2[1];
        const int32_t vi2x2 = (int32_t) i2[2];
        const int32_t vi2x3 = (int32_t) i2[3];
        i2 += 4;

        const int32_t vk2x0 = (int32_t) ((const int8_t*) ((uintptr_t) w + 4 * sizeof(int32_t)))[8];
        const int32_t vk2x1 = (int32_t) ((const int8_t*) ((uintptr_t) w + 4 * sizeof(int32_t)))[9];
        const int32_t vk2x2 = (int32_t) ((const int8_t*) ((uintptr_t) w + 4 * sizeof(int32_t)))[10];
        const int32_t vk2x3 = (int32_t) ((const int8_t*) ((uintptr_t) w + 4 * sizeof(int32_t)))[11];

        vacc0 += vi2x0 * vk2x0;
        vacc1 += vi2x1 * vk2x1;
        vacc2 += vi2x2 * vk2x2;
        vacc3 += vi2x3 * vk2x3;

        const int32_t vi3x0 = (int32_t) i3[0];
        const int32_t vi3x1 = (int32_t) i3[1];
        const int32_t vi3x2 = (int32_t) i3[2];
        const int32_t vi3x3 = (int32_t) i3[3];
        i3 += 4;

        const int32_t vk3x0 = (int32_t) ((const int8_t*) ((uintptr_t) w + 4 * sizeof(int32_t)))[12];
        const int32_t vk3x1 = (int32_t) ((const int8_t*) ((uintptr_t) w + 4 * sizeof(int32_t)))[13];
        const int32_t vk3x2 = (int32_t) ((const int8_t*) ((uintptr_t) w + 4 * sizeof(int32_t)))[14];
        const int32_t vk3x3 = (int32_t) ((const int8_t*) ((uintptr_t) w + 4 * sizeof(int32_t)))[15];

        vacc0 += vi3x0 * vk3x0;
        vacc1 += vi3x1 * vk3x1;
        vacc2 += vi3x2 * vk3x2;
        vacc3 += vi3x3 * vk3x3;

        const int32_t vi4x0 = (int32_t) i4[0];
        const int32_t vi4x1 = (int32_t) i4[1];
        const int32_t vi4x2 = (int32_t) i4[2];
        const int32_t vi4x3 = (int32_t) i4[3];
        i4 += 4;

        const int32_t vk4x0 = (int32_t) ((const int8_t*) ((uintptr_t) w + 4 * sizeof(int32_t)))[16];
        const int32_t vk4x1 = (int32_t) ((const int8_t*) ((uintptr_t) w + 4 * sizeof(int32_t)))[17];
        const int32_t vk4x2 = (int32_t) ((const int8_t*) ((uintptr_t) w + 4 * sizeof(int32_t)))[18];
        const int32_t vk4x3 = (int32_t) ((const int8_t*) ((uintptr_t) w + 4 * sizeof(int32_t)))[19];

        vacc0 += vi4x0 * vk4x0;
        vacc1 += vi4x1 * vk4x1;
        vacc2 += vi4x2 * vk4x2;
        vacc3 += vi4x3 * vk4x3;

        w = (const void*) ((uintptr_t) w + 4 * sizeof(int32_t) + 20 * sizeof(int8_t));
        b[0] = vacc0;
        b[1] = vacc1;
        b[2] = vacc2;
        b[3] = vacc3;
        b += 4;
      }
      if XNN_UNLIKELY(c != 0) {
        do {
          int32_t vacc = *((const int32_t*) w);
          const int32_t vi0 = (int32_t) *i0++;
          const int32_t vk0 = (int32_t) ((const int8_t*) ((uintptr_t) w + 1 * sizeof(int32_t)))[0];
          vacc += vi0 * vk0;
          const int32_t vi1 = (int32_t) *i1++;
          const int32_t vk1 = (int32_t) ((const int8_t*) ((uintptr_t) w + 1 * sizeof(int32_t)))[1];
          vacc += vi1 * vk1;
          const int32_t vi2 = (int32_t) *i2++;
          const int32_t vk2 = (int32_t) ((const int8_t*) ((uintptr_t) w + 1 * sizeof(int32_t)))[2];
          vacc += vi2 * vk2;
          const int32_t vi3 = (int32_t) *i3++;
          const int32_t vk3 = (int32_t) ((const int8_t*) ((uintptr_t) w + 1 * sizeof(int32_t)))[3];
          vacc += vi3 * vk3;
          const int32_t vi4 = (int32_t) *i4++;
          const int32_t vk4 = (int32_t) ((const int8_t*) ((uintptr_t) w + 1 * sizeof(int32_t)))[4];
          vacc += vi4 * vk4;

          w = (const void*) ((uintptr_t) w + 1 * sizeof(int32_t) + 5 * sizeof(int8_t));
          *b++ = vacc;
        } while (--c != 0);
      }
    }

    // Middle pass to process 5 inputs in each iteration.
    for (size_t ks = kernel_size - 5; ks > 5; ks -= 5) {
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
      input += 5;

      size_t c = channels;
      for (; c >= 4; c -= 4) {
        int32_t vacc0 = b[0];
        int32_t vacc1 = b[1];
        int32_t vacc2 = b[2];
        int32_t vacc3 = b[3];

        const int32_t vi0x0 = (int32_t) i0[0];
        const int32_t vi0x1 = (int32_t) i0[1];
        const int32_t vi0x2 = (int32_t) i0[2];
        const int32_t vi0x3 = (int32_t) i0[3];
        i0 += 4;

        const int32_t vk0x0 = (int32_t) ((const int8_t*) w)[0];
        const int32_t vk0x1 = (int32_t) ((const int8_t*) w)[1];
        const int32_t vk0x2 = (int32_t) ((const int8_t*) w)[2];
        const int32_t vk0x3 = (int32_t) ((const int8_t*) w)[3];

        vacc0 += vi0x0 * vk0x0;
        vacc1 += vi0x1 * vk0x1;
        vacc2 += vi0x2 * vk0x2;
        vacc3 += vi0x3 * vk0x3;

        const int32_t vi1x0 = (int32_t) i1[0];
        const int32_t vi1x1 = (int32_t) i1[1];
        const int32_t vi1x2 = (int32_t) i1[2];
        const int32_t vi1x3 = (int32_t) i1[3];
        i1 += 4;

        const int32_t vk1x0 = (int32_t) ((const int8_t*) w)[4];
        const int32_t vk1x1 = (int32_t) ((const int8_t*) w)[5];
        const int32_t vk1x2 = (int32_t) ((const int8_t*) w)[6];
        const int32_t vk1x3 = (int32_t) ((const int8_t*) w)[7];

        vacc0 += vi1x0 * vk1x0;
        vacc1 += vi1x1 * vk1x1;
        vacc2 += vi1x2 * vk1x2;
        vacc3 += vi1x3 * vk1x3;

        const int32_t vi2x0 = (int32_t) i2[0];
        const int32_t vi2x1 = (int32_t) i2[1];
        const int32_t vi2x2 = (int32_t) i2[2];
        const int32_t vi2x3 = (int32_t) i2[3];
        i2 += 4;

        const int32_t vk2x0 = (int32_t) ((const int8_t*) w)[8];
        const int32_t vk2x1 = (int32_t) ((const int8_t*) w)[9];
        const int32_t vk2x2 = (int32_t) ((const int8_t*) w)[10];
        const int32_t vk2x3 = (int32_t) ((const int8_t*) w)[11];

        vacc0 += vi2x0 * vk2x0;
        vacc1 += vi2x1 * vk2x1;
        vacc2 += vi2x2 * vk2x2;
        vacc3 += vi2x3 * vk2x3;

        const int32_t vi3x0 = (int32_t) i3[0];
        const int32_t vi3x1 = (int32_t) i3[1];
        const int32_t vi3x2 = (int32_t) i3[2];
        const int32_t vi3x3 = (int32_t) i3[3];
        i3 += 4;

        const int32_t vk3x0 = (int32_t) ((const int8_t*) w)[12];
        const int32_t vk3x1 = (int32_t) ((const int8_t*) w)[13];
        const int32_t vk3x2 = (int32_t) ((const int8_t*) w)[14];
        const int32_t vk3x3 = (int32_t) ((const int8_t*) w)[15];

        vacc0 += vi3x0 * vk3x0;
        vacc1 += vi3x1 * vk3x1;
        vacc2 += vi3x2 * vk3x2;
        vacc3 += vi3x3 * vk3x3;

        const int32_t vi4x0 = (int32_t) i4[0];
        const int32_t vi4x1 = (int32_t) i4[1];
        const int32_t vi4x2 = (int32_t) i4[2];
        const int32_t vi4x3 = (int32_t) i4[3];
        i4 += 4;

        const int32_t vk4x0 = (int32_t) ((const int8_t*) w)[16];
        const int32_t vk4x1 = (int32_t) ((const int8_t*) w)[17];
        const int32_t vk4x2 = (int32_t) ((const int8_t*) w)[18];
        const int32_t vk4x3 = (int32_t) ((const int8_t*) w)[19];

        vacc0 += vi4x0 * vk4x0;
        vacc1 += vi4x1 * vk4x1;
        vacc2 += vi4x2 * vk4x2;
        vacc3 += vi4x3 * vk4x3;

        w = (const void*) ((uintptr_t) w + 20 * sizeof(int8_t));
        b[0] = vacc0;
        b[1] = vacc1;
        b[2] = vacc2;
        b[3] = vacc3;
        b += 4;
      }
      if XNN_UNLIKELY(c != 0) {
        do {
          int32_t vacc = *b;
          const int32_t vi0 = (int32_t) *i0++;
          const int32_t vk0 = (int32_t) ((const int8_t*) w)[0];
          vacc += vi0 * vk0;
          const int32_t vi1 = (int32_t) *i1++;
          const int32_t vk1 = (int32_t) ((const int8_t*) w)[1];
          vacc += vi1 * vk1;
          const int32_t vi2 = (int32_t) *i2++;
          const int32_t vk2 = (int32_t) ((const int8_t*) w)[2];
          vacc += vi2 * vk2;
          const int32_t vi3 = (int32_t) *i3++;
          const int32_t vk3 = (int32_t) ((const int8_t*) w)[3];
          vacc += vi3 * vk3;
          const int32_t vi4 = (int32_t) *i4++;
          const int32_t vk4 = (int32_t) ((const int8_t*) w)[4];
          vacc += vi4 * vk4;

          w = (const void*) ((uintptr_t) w + 5 * sizeof(int8_t));
          *b++ = vacc;
        } while (--c != 0);
      }
    }

    // Last pass to process up to 5 inputs.
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

      size_t c = channels;
      for (; c >= 4; c -= 4) {
        int32_t vacc0 = b[0];
        int32_t vacc1 = b[1];
        int32_t vacc2 = b[2];
        int32_t vacc3 = b[3];
        b += 4;

        const int32_t vi0x0 = (int32_t) i0[0];
        const int32_t vi0x1 = (int32_t) i0[1];
        const int32_t vi0x2 = (int32_t) i0[2];
        const int32_t vi0x3 = (int32_t) i0[3];
        i0 += 4;

        const int32_t vk0x0 = (int32_t) ((const int8_t*) w)[0];
        const int32_t vk0x1 = (int32_t) ((const int8_t*) w)[1];
        const int32_t vk0x2 = (int32_t) ((const int8_t*) w)[2];
        const int32_t vk0x3 = (int32_t) ((const int8_t*) w)[3];

        vacc0 += vi0x0 * vk0x0;
        vacc1 += vi0x1 * vk0x1;
        vacc2 += vi0x2 * vk0x2;
        vacc3 += vi0x3 * vk0x3;

        const int32_t vi1x0 = (int32_t) i1[0];
        const int32_t vi1x1 = (int32_t) i1[1];
        const int32_t vi1x2 = (int32_t) i1[2];
        const int32_t vi1x3 = (int32_t) i1[3];
        i1 += 4;

        const int32_t vk1x0 = (int32_t) ((const int8_t*) w)[4];
        const int32_t vk1x1 = (int32_t) ((const int8_t*) w)[5];
        const int32_t vk1x2 = (int32_t) ((const int8_t*) w)[6];
        const int32_t vk1x3 = (int32_t) ((const int8_t*) w)[7];

        vacc0 += vi1x0 * vk1x0;
        vacc1 += vi1x1 * vk1x1;
        vacc2 += vi1x2 * vk1x2;
        vacc3 += vi1x3 * vk1x3;

        const int32_t vi2x0 = (int32_t) i2[0];
        const int32_t vi2x1 = (int32_t) i2[1];
        const int32_t vi2x2 = (int32_t) i2[2];
        const int32_t vi2x3 = (int32_t) i2[3];
        i2 += 4;

        const int32_t vk2x0 = (int32_t) ((const int8_t*) w)[8];
        const int32_t vk2x1 = (int32_t) ((const int8_t*) w)[9];
        const int32_t vk2x2 = (int32_t) ((const int8_t*) w)[10];
        const int32_t vk2x3 = (int32_t) ((const int8_t*) w)[11];

        vacc0 += vi2x0 * vk2x0;
        vacc1 += vi2x1 * vk2x1;
        vacc2 += vi2x2 * vk2x2;
        vacc3 += vi2x3 * vk2x3;

        const int32_t vi3x0 = (int32_t) i3[0];
        const int32_t vi3x1 = (int32_t) i3[1];
        const int32_t vi3x2 = (int32_t) i3[2];
        const int32_t vi3x3 = (int32_t) i3[3];
        i3 += 4;

        const int32_t vk3x0 = (int32_t) ((const int8_t*) w)[12];
        const int32_t vk3x1 = (int32_t) ((const int8_t*) w)[13];
        const int32_t vk3x2 = (int32_t) ((const int8_t*) w)[14];
        const int32_t vk3x3 = (int32_t) ((const int8_t*) w)[15];

        vacc0 += vi3x0 * vk3x0;
        vacc1 += vi3x1 * vk3x1;
        vacc2 += vi3x2 * vk3x2;
        vacc3 += vi3x3 * vk3x3;

        const int32_t vi4x0 = (int32_t) i4[0];
        const int32_t vi4x1 = (int32_t) i4[1];
        const int32_t vi4x2 = (int32_t) i4[2];
        const int32_t vi4x3 = (int32_t) i4[3];
        i4 += 4;

        const int32_t vk4x0 = (int32_t) ((const int8_t*) w)[16];
        const int32_t vk4x1 = (int32_t) ((const int8_t*) w)[17];
        const int32_t vk4x2 = (int32_t) ((const int8_t*) w)[18];
        const int32_t vk4x3 = (int32_t) ((const int8_t*) w)[19];

        vacc0 += vi4x0 * vk4x0;
        vacc1 += vi4x1 * vk4x1;
        vacc2 += vi4x2 * vk4x2;
        vacc3 += vi4x3 * vk4x3;

        w = (const void*) ((uintptr_t) w + 20 * sizeof(int8_t));

        float vfpacc0 = (float) vacc0;
        float vfpacc1 = (float) vacc1;
        float vfpacc2 = (float) vacc2;
        float vfpacc3 = (float) vacc3;

        vfpacc0 *= vscale;
        vfpacc1 *= vscale;
        vfpacc2 *= vscale;
        vfpacc3 *= vscale;

        vfpacc0 = math_max_f32(vfpacc0, voutput_min_less_zero_point);
        vfpacc1 = math_max_f32(vfpacc1, voutput_min_less_zero_point);
        vfpacc2 = math_max_f32(vfpacc2, voutput_min_less_zero_point);
        vfpacc3 = math_max_f32(vfpacc3, voutput_min_less_zero_point);

        vfpacc0 = math_min_f32(vfpacc0, voutput_max_less_zero_point);
        vfpacc1 = math_min_f32(vfpacc1, voutput_max_less_zero_point);
        vfpacc2 = math_min_f32(vfpacc2, voutput_max_less_zero_point);
        vfpacc3 = math_min_f32(vfpacc3, voutput_max_less_zero_point);

        const int32_t vrndacc0 = (int32_t) lrintf(vfpacc0);
        const int32_t vrndacc1 = (int32_t) lrintf(vfpacc1);
        const int32_t vrndacc2 = (int32_t) lrintf(vfpacc2);
        const int32_t vrndacc3 = (int32_t) lrintf(vfpacc3);

        int32_t vout0 = (int32_t) vrndacc0 + voutput_zero_point;
        int32_t vout1 = (int32_t) vrndacc1 + voutput_zero_point;
        int32_t vout2 = (int32_t) vrndacc2 + voutput_zero_point;
        int32_t vout3 = (int32_t) vrndacc3 + voutput_zero_point;

        output[0] = (int8_t) vout0;
        output[1] = (int8_t) vout1;
        output[2] = (int8_t) vout2;
        output[3] = (int8_t) vout3;
        output += 4;
      }
      if XNN_UNLIKELY(c != 0) {
        do {
          int32_t vacc = *b++;
          const int32_t vi0 = (int32_t) *i0++;
          const int32_t vk0 = (int32_t) ((const int8_t*) w)[0];
          vacc += vi0 * vk0;
          const int32_t vi1 = (int32_t) *i1++;
          const int32_t vk1 = (int32_t) ((const int8_t*) w)[1];
          vacc += vi1 * vk1;
          const int32_t vi2 = (int32_t) *i2++;
          const int32_t vk2 = (int32_t) ((const int8_t*) w)[2];
          vacc += vi2 * vk2;
          const int32_t vi3 = (int32_t) *i3++;
          const int32_t vk3 = (int32_t) ((const int8_t*) w)[3];
          vacc += vi3 * vk3;
          const int32_t vi4 = (int32_t) *i4++;
          const int32_t vk4 = (int32_t) ((const int8_t*) w)[4];
          vacc += vi4 * vk4;

          w = (const void*) ((uintptr_t) w + 5 * sizeof(int8_t));

          float vfpacc = (float) vacc * vscale;

          vfpacc = math_max_f32(vfpacc, voutput_min_less_zero_point);
          vfpacc = math_min_f32(vfpacc, voutput_max_less_zero_point);
          const int32_t vrndacc = (int32_t) lrintf(vfpacc);
          int32_t vout = vrndacc + voutput_zero_point;

          *output++ = (int8_t) vout;
        } while (--c != 0);
      }
    }

    input = (const int8_t**) ((uintptr_t) input + input_stride);
    output = (int8_t*) ((uintptr_t) output + output_increment);
  } while (--output_width != 0);
}
