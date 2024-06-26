// Auto-generated file. Do not edit!
//   Template: src/qs8-dwconv/multipass-scalar.c.in
//   Generator: tools/xngen
//
// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include "xnnpack/dwconv.h"
#include "xnnpack/math.h"
#include "xnnpack/unaligned.h"


void xnn_qu8_dwconv_minmax_fp32_ukernel_6f6m7l4c1s1r__scalar_imagic(
    size_t channels,
    size_t output_width,
    const uint8_t** input,
    const void* weights,
    uint8_t* output,
    intptr_t input_stride,
    size_t output_increment,
    size_t input_offset,
    const uint8_t* zero,
    size_t kernel_size,
    int32_t* buffer,
    const union xnn_qu8_conv_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(channels != 0);
  assert(output_width != 0);
  assert(kernel_size > 6);

  const float vscale = params->fp32_scalar_imagic.scale;
  const float vmagic_bias = params->fp32_scalar_imagic.magic_bias;
  const int32_t vmagic_min = params->fp32_scalar_imagic.magic_min;
  const int32_t vmagic_max = params->fp32_scalar_imagic.magic_max;
  const int32_t vmagic_bias_less_zero_point = params->fp32_scalar_imagic.magic_bias_less_zero_point;
  const int32_t vkernel_zero_point = params->fp32_scalar_imagic.kernel_zero_point;
  do {
    const void* w = weights;

    // First pass to process 6 inputs.
    {
      int32_t* b = buffer;
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
      input += 6;

      size_t c = channels;
      for (; c >= 4; c -= 4) {
        int32_t vacc0 = ((const int32_t*) w)[0];
        int32_t vacc1 = ((const int32_t*) w)[1];
        int32_t vacc2 = ((const int32_t*) w)[2];
        int32_t vacc3 = ((const int32_t*) w)[3];

        const int32_t vi0x0 = (int32_t) (uint32_t) i0[0];
        const int32_t vi0x1 = (int32_t) (uint32_t) i0[1];
        const int32_t vi0x2 = (int32_t) (uint32_t) i0[2];
        const int32_t vi0x3 = (int32_t) (uint32_t) i0[3];
        i0 += 4;

        const int32_t vk0x0 = (int32_t) (uint32_t) ((const uint8_t*) ((uintptr_t) w + 4 * sizeof(int32_t)))[0] - vkernel_zero_point;
        const int32_t vk0x1 = (int32_t) (uint32_t) ((const uint8_t*) ((uintptr_t) w + 4 * sizeof(int32_t)))[1] - vkernel_zero_point;
        const int32_t vk0x2 = (int32_t) (uint32_t) ((const uint8_t*) ((uintptr_t) w + 4 * sizeof(int32_t)))[2] - vkernel_zero_point;
        const int32_t vk0x3 = (int32_t) (uint32_t) ((const uint8_t*) ((uintptr_t) w + 4 * sizeof(int32_t)))[3] - vkernel_zero_point;

        vacc0 += vi0x0 * vk0x0;
        vacc1 += vi0x1 * vk0x1;
        vacc2 += vi0x2 * vk0x2;
        vacc3 += vi0x3 * vk0x3;

        const int32_t vi1x0 = (int32_t) (uint32_t) i1[0];
        const int32_t vi1x1 = (int32_t) (uint32_t) i1[1];
        const int32_t vi1x2 = (int32_t) (uint32_t) i1[2];
        const int32_t vi1x3 = (int32_t) (uint32_t) i1[3];
        i1 += 4;

        const int32_t vk1x0 = (int32_t) (uint32_t) ((const uint8_t*) ((uintptr_t) w + 4 * sizeof(int32_t)))[4] - vkernel_zero_point;
        const int32_t vk1x1 = (int32_t) (uint32_t) ((const uint8_t*) ((uintptr_t) w + 4 * sizeof(int32_t)))[5] - vkernel_zero_point;
        const int32_t vk1x2 = (int32_t) (uint32_t) ((const uint8_t*) ((uintptr_t) w + 4 * sizeof(int32_t)))[6] - vkernel_zero_point;
        const int32_t vk1x3 = (int32_t) (uint32_t) ((const uint8_t*) ((uintptr_t) w + 4 * sizeof(int32_t)))[7] - vkernel_zero_point;

        vacc0 += vi1x0 * vk1x0;
        vacc1 += vi1x1 * vk1x1;
        vacc2 += vi1x2 * vk1x2;
        vacc3 += vi1x3 * vk1x3;

        const int32_t vi2x0 = (int32_t) (uint32_t) i2[0];
        const int32_t vi2x1 = (int32_t) (uint32_t) i2[1];
        const int32_t vi2x2 = (int32_t) (uint32_t) i2[2];
        const int32_t vi2x3 = (int32_t) (uint32_t) i2[3];
        i2 += 4;

        const int32_t vk2x0 = (int32_t) (uint32_t) ((const uint8_t*) ((uintptr_t) w + 4 * sizeof(int32_t)))[8] - vkernel_zero_point;
        const int32_t vk2x1 = (int32_t) (uint32_t) ((const uint8_t*) ((uintptr_t) w + 4 * sizeof(int32_t)))[9] - vkernel_zero_point;
        const int32_t vk2x2 = (int32_t) (uint32_t) ((const uint8_t*) ((uintptr_t) w + 4 * sizeof(int32_t)))[10] - vkernel_zero_point;
        const int32_t vk2x3 = (int32_t) (uint32_t) ((const uint8_t*) ((uintptr_t) w + 4 * sizeof(int32_t)))[11] - vkernel_zero_point;

        vacc0 += vi2x0 * vk2x0;
        vacc1 += vi2x1 * vk2x1;
        vacc2 += vi2x2 * vk2x2;
        vacc3 += vi2x3 * vk2x3;

        const int32_t vi3x0 = (int32_t) (uint32_t) i3[0];
        const int32_t vi3x1 = (int32_t) (uint32_t) i3[1];
        const int32_t vi3x2 = (int32_t) (uint32_t) i3[2];
        const int32_t vi3x3 = (int32_t) (uint32_t) i3[3];
        i3 += 4;

        const int32_t vk3x0 = (int32_t) (uint32_t) ((const uint8_t*) ((uintptr_t) w + 4 * sizeof(int32_t)))[12] - vkernel_zero_point;
        const int32_t vk3x1 = (int32_t) (uint32_t) ((const uint8_t*) ((uintptr_t) w + 4 * sizeof(int32_t)))[13] - vkernel_zero_point;
        const int32_t vk3x2 = (int32_t) (uint32_t) ((const uint8_t*) ((uintptr_t) w + 4 * sizeof(int32_t)))[14] - vkernel_zero_point;
        const int32_t vk3x3 = (int32_t) (uint32_t) ((const uint8_t*) ((uintptr_t) w + 4 * sizeof(int32_t)))[15] - vkernel_zero_point;

        vacc0 += vi3x0 * vk3x0;
        vacc1 += vi3x1 * vk3x1;
        vacc2 += vi3x2 * vk3x2;
        vacc3 += vi3x3 * vk3x3;

        const int32_t vi4x0 = (int32_t) (uint32_t) i4[0];
        const int32_t vi4x1 = (int32_t) (uint32_t) i4[1];
        const int32_t vi4x2 = (int32_t) (uint32_t) i4[2];
        const int32_t vi4x3 = (int32_t) (uint32_t) i4[3];
        i4 += 4;

        const int32_t vk4x0 = (int32_t) (uint32_t) ((const uint8_t*) ((uintptr_t) w + 4 * sizeof(int32_t)))[16] - vkernel_zero_point;
        const int32_t vk4x1 = (int32_t) (uint32_t) ((const uint8_t*) ((uintptr_t) w + 4 * sizeof(int32_t)))[17] - vkernel_zero_point;
        const int32_t vk4x2 = (int32_t) (uint32_t) ((const uint8_t*) ((uintptr_t) w + 4 * sizeof(int32_t)))[18] - vkernel_zero_point;
        const int32_t vk4x3 = (int32_t) (uint32_t) ((const uint8_t*) ((uintptr_t) w + 4 * sizeof(int32_t)))[19] - vkernel_zero_point;

        vacc0 += vi4x0 * vk4x0;
        vacc1 += vi4x1 * vk4x1;
        vacc2 += vi4x2 * vk4x2;
        vacc3 += vi4x3 * vk4x3;

        const int32_t vi5x0 = (int32_t) (uint32_t) i5[0];
        const int32_t vi5x1 = (int32_t) (uint32_t) i5[1];
        const int32_t vi5x2 = (int32_t) (uint32_t) i5[2];
        const int32_t vi5x3 = (int32_t) (uint32_t) i5[3];
        i5 += 4;

        const int32_t vk5x0 = (int32_t) (uint32_t) ((const uint8_t*) ((uintptr_t) w + 4 * sizeof(int32_t)))[20] - vkernel_zero_point;
        const int32_t vk5x1 = (int32_t) (uint32_t) ((const uint8_t*) ((uintptr_t) w + 4 * sizeof(int32_t)))[21] - vkernel_zero_point;
        const int32_t vk5x2 = (int32_t) (uint32_t) ((const uint8_t*) ((uintptr_t) w + 4 * sizeof(int32_t)))[22] - vkernel_zero_point;
        const int32_t vk5x3 = (int32_t) (uint32_t) ((const uint8_t*) ((uintptr_t) w + 4 * sizeof(int32_t)))[23] - vkernel_zero_point;

        vacc0 += vi5x0 * vk5x0;
        vacc1 += vi5x1 * vk5x1;
        vacc2 += vi5x2 * vk5x2;
        vacc3 += vi5x3 * vk5x3;

        w = (const void*) ((uintptr_t) w + 4 * sizeof(int32_t) + 24 * sizeof(uint8_t));
        b[0] = vacc0;
        b[1] = vacc1;
        b[2] = vacc2;
        b[3] = vacc3;
        b += 4;
      }
      if XNN_UNLIKELY(c != 0) {
        do {
          int32_t vacc = *((const int32_t*) w);
          const int32_t vi0 = (int32_t) (uint32_t) *i0++;
          const int32_t vk0 = (int32_t) (uint32_t) ((const uint8_t*) ((uintptr_t) w + 1 * sizeof(int32_t)))[0] - vkernel_zero_point;
          vacc += vi0 * vk0;
          const int32_t vi1 = (int32_t) (uint32_t) *i1++;
          const int32_t vk1 = (int32_t) (uint32_t) ((const uint8_t*) ((uintptr_t) w + 1 * sizeof(int32_t)))[1] - vkernel_zero_point;
          vacc += vi1 * vk1;
          const int32_t vi2 = (int32_t) (uint32_t) *i2++;
          const int32_t vk2 = (int32_t) (uint32_t) ((const uint8_t*) ((uintptr_t) w + 1 * sizeof(int32_t)))[2] - vkernel_zero_point;
          vacc += vi2 * vk2;
          const int32_t vi3 = (int32_t) (uint32_t) *i3++;
          const int32_t vk3 = (int32_t) (uint32_t) ((const uint8_t*) ((uintptr_t) w + 1 * sizeof(int32_t)))[3] - vkernel_zero_point;
          vacc += vi3 * vk3;
          const int32_t vi4 = (int32_t) (uint32_t) *i4++;
          const int32_t vk4 = (int32_t) (uint32_t) ((const uint8_t*) ((uintptr_t) w + 1 * sizeof(int32_t)))[4] - vkernel_zero_point;
          vacc += vi4 * vk4;
          const int32_t vi5 = (int32_t) (uint32_t) *i5++;
          const int32_t vk5 = (int32_t) (uint32_t) ((const uint8_t*) ((uintptr_t) w + 1 * sizeof(int32_t)))[5] - vkernel_zero_point;
          vacc += vi5 * vk5;

          w = (const void*) ((uintptr_t) w + 1 * sizeof(int32_t) + 6 * sizeof(uint8_t));
          *b++ = vacc;
        } while (--c != 0);
      }
    }

    // Middle pass to process 6 inputs in each iteration.
    for (size_t ks = kernel_size - 6; ks > 7; ks -= 6) {
      int32_t* b = buffer;
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
      input += 6;

      size_t c = channels;
      for (; c >= 4; c -= 4) {
        int32_t vacc0 = b[0];
        int32_t vacc1 = b[1];
        int32_t vacc2 = b[2];
        int32_t vacc3 = b[3];

        const int32_t vi0x0 = (int32_t) (uint32_t) i0[0];
        const int32_t vi0x1 = (int32_t) (uint32_t) i0[1];
        const int32_t vi0x2 = (int32_t) (uint32_t) i0[2];
        const int32_t vi0x3 = (int32_t) (uint32_t) i0[3];
        i0 += 4;

        const int32_t vk0x0 = (int32_t) (uint32_t) ((const uint8_t*) w)[0] - vkernel_zero_point;
        const int32_t vk0x1 = (int32_t) (uint32_t) ((const uint8_t*) w)[1] - vkernel_zero_point;
        const int32_t vk0x2 = (int32_t) (uint32_t) ((const uint8_t*) w)[2] - vkernel_zero_point;
        const int32_t vk0x3 = (int32_t) (uint32_t) ((const uint8_t*) w)[3] - vkernel_zero_point;

        vacc0 += vi0x0 * vk0x0;
        vacc1 += vi0x1 * vk0x1;
        vacc2 += vi0x2 * vk0x2;
        vacc3 += vi0x3 * vk0x3;

        const int32_t vi1x0 = (int32_t) (uint32_t) i1[0];
        const int32_t vi1x1 = (int32_t) (uint32_t) i1[1];
        const int32_t vi1x2 = (int32_t) (uint32_t) i1[2];
        const int32_t vi1x3 = (int32_t) (uint32_t) i1[3];
        i1 += 4;

        const int32_t vk1x0 = (int32_t) (uint32_t) ((const uint8_t*) w)[4] - vkernel_zero_point;
        const int32_t vk1x1 = (int32_t) (uint32_t) ((const uint8_t*) w)[5] - vkernel_zero_point;
        const int32_t vk1x2 = (int32_t) (uint32_t) ((const uint8_t*) w)[6] - vkernel_zero_point;
        const int32_t vk1x3 = (int32_t) (uint32_t) ((const uint8_t*) w)[7] - vkernel_zero_point;

        vacc0 += vi1x0 * vk1x0;
        vacc1 += vi1x1 * vk1x1;
        vacc2 += vi1x2 * vk1x2;
        vacc3 += vi1x3 * vk1x3;

        const int32_t vi2x0 = (int32_t) (uint32_t) i2[0];
        const int32_t vi2x1 = (int32_t) (uint32_t) i2[1];
        const int32_t vi2x2 = (int32_t) (uint32_t) i2[2];
        const int32_t vi2x3 = (int32_t) (uint32_t) i2[3];
        i2 += 4;

        const int32_t vk2x0 = (int32_t) (uint32_t) ((const uint8_t*) w)[8] - vkernel_zero_point;
        const int32_t vk2x1 = (int32_t) (uint32_t) ((const uint8_t*) w)[9] - vkernel_zero_point;
        const int32_t vk2x2 = (int32_t) (uint32_t) ((const uint8_t*) w)[10] - vkernel_zero_point;
        const int32_t vk2x3 = (int32_t) (uint32_t) ((const uint8_t*) w)[11] - vkernel_zero_point;

        vacc0 += vi2x0 * vk2x0;
        vacc1 += vi2x1 * vk2x1;
        vacc2 += vi2x2 * vk2x2;
        vacc3 += vi2x3 * vk2x3;

        const int32_t vi3x0 = (int32_t) (uint32_t) i3[0];
        const int32_t vi3x1 = (int32_t) (uint32_t) i3[1];
        const int32_t vi3x2 = (int32_t) (uint32_t) i3[2];
        const int32_t vi3x3 = (int32_t) (uint32_t) i3[3];
        i3 += 4;

        const int32_t vk3x0 = (int32_t) (uint32_t) ((const uint8_t*) w)[12] - vkernel_zero_point;
        const int32_t vk3x1 = (int32_t) (uint32_t) ((const uint8_t*) w)[13] - vkernel_zero_point;
        const int32_t vk3x2 = (int32_t) (uint32_t) ((const uint8_t*) w)[14] - vkernel_zero_point;
        const int32_t vk3x3 = (int32_t) (uint32_t) ((const uint8_t*) w)[15] - vkernel_zero_point;

        vacc0 += vi3x0 * vk3x0;
        vacc1 += vi3x1 * vk3x1;
        vacc2 += vi3x2 * vk3x2;
        vacc3 += vi3x3 * vk3x3;

        const int32_t vi4x0 = (int32_t) (uint32_t) i4[0];
        const int32_t vi4x1 = (int32_t) (uint32_t) i4[1];
        const int32_t vi4x2 = (int32_t) (uint32_t) i4[2];
        const int32_t vi4x3 = (int32_t) (uint32_t) i4[3];
        i4 += 4;

        const int32_t vk4x0 = (int32_t) (uint32_t) ((const uint8_t*) w)[16] - vkernel_zero_point;
        const int32_t vk4x1 = (int32_t) (uint32_t) ((const uint8_t*) w)[17] - vkernel_zero_point;
        const int32_t vk4x2 = (int32_t) (uint32_t) ((const uint8_t*) w)[18] - vkernel_zero_point;
        const int32_t vk4x3 = (int32_t) (uint32_t) ((const uint8_t*) w)[19] - vkernel_zero_point;

        vacc0 += vi4x0 * vk4x0;
        vacc1 += vi4x1 * vk4x1;
        vacc2 += vi4x2 * vk4x2;
        vacc3 += vi4x3 * vk4x3;

        const int32_t vi5x0 = (int32_t) (uint32_t) i5[0];
        const int32_t vi5x1 = (int32_t) (uint32_t) i5[1];
        const int32_t vi5x2 = (int32_t) (uint32_t) i5[2];
        const int32_t vi5x3 = (int32_t) (uint32_t) i5[3];
        i5 += 4;

        const int32_t vk5x0 = (int32_t) (uint32_t) ((const uint8_t*) w)[20] - vkernel_zero_point;
        const int32_t vk5x1 = (int32_t) (uint32_t) ((const uint8_t*) w)[21] - vkernel_zero_point;
        const int32_t vk5x2 = (int32_t) (uint32_t) ((const uint8_t*) w)[22] - vkernel_zero_point;
        const int32_t vk5x3 = (int32_t) (uint32_t) ((const uint8_t*) w)[23] - vkernel_zero_point;

        vacc0 += vi5x0 * vk5x0;
        vacc1 += vi5x1 * vk5x1;
        vacc2 += vi5x2 * vk5x2;
        vacc3 += vi5x3 * vk5x3;

        w = (const void*) ((uintptr_t) w + 24 * sizeof(uint8_t));
        b[0] = vacc0;
        b[1] = vacc1;
        b[2] = vacc2;
        b[3] = vacc3;
        b += 4;
      }
      if XNN_UNLIKELY(c != 0) {
        do {
          int32_t vacc = *b;
          const int32_t vi0 = (int32_t) (uint32_t) *i0++;
          const int32_t vk0 = (int32_t) (uint32_t) ((const uint8_t*) w)[0] - vkernel_zero_point;
          vacc += vi0 * vk0;
          const int32_t vi1 = (int32_t) (uint32_t) *i1++;
          const int32_t vk1 = (int32_t) (uint32_t) ((const uint8_t*) w)[1] - vkernel_zero_point;
          vacc += vi1 * vk1;
          const int32_t vi2 = (int32_t) (uint32_t) *i2++;
          const int32_t vk2 = (int32_t) (uint32_t) ((const uint8_t*) w)[2] - vkernel_zero_point;
          vacc += vi2 * vk2;
          const int32_t vi3 = (int32_t) (uint32_t) *i3++;
          const int32_t vk3 = (int32_t) (uint32_t) ((const uint8_t*) w)[3] - vkernel_zero_point;
          vacc += vi3 * vk3;
          const int32_t vi4 = (int32_t) (uint32_t) *i4++;
          const int32_t vk4 = (int32_t) (uint32_t) ((const uint8_t*) w)[4] - vkernel_zero_point;
          vacc += vi4 * vk4;
          const int32_t vi5 = (int32_t) (uint32_t) *i5++;
          const int32_t vk5 = (int32_t) (uint32_t) ((const uint8_t*) w)[5] - vkernel_zero_point;
          vacc += vi5 * vk5;

          w = (const void*) ((uintptr_t) w + 6 * sizeof(uint8_t));
          *b++ = vacc;
        } while (--c != 0);
      }
    }

    // Last pass to process up to 7 inputs.
    {
      const int32_t* b = buffer;
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

      size_t c = channels;
      for (; c >= 4; c -= 4) {
        int32_t vacc0 = b[0];
        int32_t vacc1 = b[1];
        int32_t vacc2 = b[2];
        int32_t vacc3 = b[3];
        b += 4;

        const int32_t vi0x0 = (int32_t) (uint32_t) i0[0];
        const int32_t vi0x1 = (int32_t) (uint32_t) i0[1];
        const int32_t vi0x2 = (int32_t) (uint32_t) i0[2];
        const int32_t vi0x3 = (int32_t) (uint32_t) i0[3];
        i0 += 4;

        const int32_t vk0x0 = (int32_t) (uint32_t) ((const uint8_t*) w)[0] - vkernel_zero_point;
        const int32_t vk0x1 = (int32_t) (uint32_t) ((const uint8_t*) w)[1] - vkernel_zero_point;
        const int32_t vk0x2 = (int32_t) (uint32_t) ((const uint8_t*) w)[2] - vkernel_zero_point;
        const int32_t vk0x3 = (int32_t) (uint32_t) ((const uint8_t*) w)[3] - vkernel_zero_point;

        vacc0 += vi0x0 * vk0x0;
        vacc1 += vi0x1 * vk0x1;
        vacc2 += vi0x2 * vk0x2;
        vacc3 += vi0x3 * vk0x3;

        const int32_t vi1x0 = (int32_t) (uint32_t) i1[0];
        const int32_t vi1x1 = (int32_t) (uint32_t) i1[1];
        const int32_t vi1x2 = (int32_t) (uint32_t) i1[2];
        const int32_t vi1x3 = (int32_t) (uint32_t) i1[3];
        i1 += 4;

        const int32_t vk1x0 = (int32_t) (uint32_t) ((const uint8_t*) w)[4] - vkernel_zero_point;
        const int32_t vk1x1 = (int32_t) (uint32_t) ((const uint8_t*) w)[5] - vkernel_zero_point;
        const int32_t vk1x2 = (int32_t) (uint32_t) ((const uint8_t*) w)[6] - vkernel_zero_point;
        const int32_t vk1x3 = (int32_t) (uint32_t) ((const uint8_t*) w)[7] - vkernel_zero_point;

        vacc0 += vi1x0 * vk1x0;
        vacc1 += vi1x1 * vk1x1;
        vacc2 += vi1x2 * vk1x2;
        vacc3 += vi1x3 * vk1x3;

        const int32_t vi2x0 = (int32_t) (uint32_t) i2[0];
        const int32_t vi2x1 = (int32_t) (uint32_t) i2[1];
        const int32_t vi2x2 = (int32_t) (uint32_t) i2[2];
        const int32_t vi2x3 = (int32_t) (uint32_t) i2[3];
        i2 += 4;

        const int32_t vk2x0 = (int32_t) (uint32_t) ((const uint8_t*) w)[8] - vkernel_zero_point;
        const int32_t vk2x1 = (int32_t) (uint32_t) ((const uint8_t*) w)[9] - vkernel_zero_point;
        const int32_t vk2x2 = (int32_t) (uint32_t) ((const uint8_t*) w)[10] - vkernel_zero_point;
        const int32_t vk2x3 = (int32_t) (uint32_t) ((const uint8_t*) w)[11] - vkernel_zero_point;

        vacc0 += vi2x0 * vk2x0;
        vacc1 += vi2x1 * vk2x1;
        vacc2 += vi2x2 * vk2x2;
        vacc3 += vi2x3 * vk2x3;

        const int32_t vi3x0 = (int32_t) (uint32_t) i3[0];
        const int32_t vi3x1 = (int32_t) (uint32_t) i3[1];
        const int32_t vi3x2 = (int32_t) (uint32_t) i3[2];
        const int32_t vi3x3 = (int32_t) (uint32_t) i3[3];
        i3 += 4;

        const int32_t vk3x0 = (int32_t) (uint32_t) ((const uint8_t*) w)[12] - vkernel_zero_point;
        const int32_t vk3x1 = (int32_t) (uint32_t) ((const uint8_t*) w)[13] - vkernel_zero_point;
        const int32_t vk3x2 = (int32_t) (uint32_t) ((const uint8_t*) w)[14] - vkernel_zero_point;
        const int32_t vk3x3 = (int32_t) (uint32_t) ((const uint8_t*) w)[15] - vkernel_zero_point;

        vacc0 += vi3x0 * vk3x0;
        vacc1 += vi3x1 * vk3x1;
        vacc2 += vi3x2 * vk3x2;
        vacc3 += vi3x3 * vk3x3;

        const int32_t vi4x0 = (int32_t) (uint32_t) i4[0];
        const int32_t vi4x1 = (int32_t) (uint32_t) i4[1];
        const int32_t vi4x2 = (int32_t) (uint32_t) i4[2];
        const int32_t vi4x3 = (int32_t) (uint32_t) i4[3];
        i4 += 4;

        const int32_t vk4x0 = (int32_t) (uint32_t) ((const uint8_t*) w)[16] - vkernel_zero_point;
        const int32_t vk4x1 = (int32_t) (uint32_t) ((const uint8_t*) w)[17] - vkernel_zero_point;
        const int32_t vk4x2 = (int32_t) (uint32_t) ((const uint8_t*) w)[18] - vkernel_zero_point;
        const int32_t vk4x3 = (int32_t) (uint32_t) ((const uint8_t*) w)[19] - vkernel_zero_point;

        vacc0 += vi4x0 * vk4x0;
        vacc1 += vi4x1 * vk4x1;
        vacc2 += vi4x2 * vk4x2;
        vacc3 += vi4x3 * vk4x3;

        const int32_t vi5x0 = (int32_t) (uint32_t) i5[0];
        const int32_t vi5x1 = (int32_t) (uint32_t) i5[1];
        const int32_t vi5x2 = (int32_t) (uint32_t) i5[2];
        const int32_t vi5x3 = (int32_t) (uint32_t) i5[3];
        i5 += 4;

        const int32_t vk5x0 = (int32_t) (uint32_t) ((const uint8_t*) w)[20] - vkernel_zero_point;
        const int32_t vk5x1 = (int32_t) (uint32_t) ((const uint8_t*) w)[21] - vkernel_zero_point;
        const int32_t vk5x2 = (int32_t) (uint32_t) ((const uint8_t*) w)[22] - vkernel_zero_point;
        const int32_t vk5x3 = (int32_t) (uint32_t) ((const uint8_t*) w)[23] - vkernel_zero_point;

        vacc0 += vi5x0 * vk5x0;
        vacc1 += vi5x1 * vk5x1;
        vacc2 += vi5x2 * vk5x2;
        vacc3 += vi5x3 * vk5x3;

        const int32_t vi6x0 = (int32_t) (uint32_t) i6[0];
        const int32_t vi6x1 = (int32_t) (uint32_t) i6[1];
        const int32_t vi6x2 = (int32_t) (uint32_t) i6[2];
        const int32_t vi6x3 = (int32_t) (uint32_t) i6[3];
        i6 += 4;

        const int32_t vk6x0 = (int32_t) (uint32_t) ((const uint8_t*) w)[24] - vkernel_zero_point;
        const int32_t vk6x1 = (int32_t) (uint32_t) ((const uint8_t*) w)[25] - vkernel_zero_point;
        const int32_t vk6x2 = (int32_t) (uint32_t) ((const uint8_t*) w)[26] - vkernel_zero_point;
        const int32_t vk6x3 = (int32_t) (uint32_t) ((const uint8_t*) w)[27] - vkernel_zero_point;

        vacc0 += vi6x0 * vk6x0;
        vacc1 += vi6x1 * vk6x1;
        vacc2 += vi6x2 * vk6x2;
        vacc3 += vi6x3 * vk6x3;

        w = (const void*) ((uintptr_t) w + 28 * sizeof(uint8_t));

        float vfpacc0 = (float) vacc0;
        float vfpacc1 = (float) vacc1;
        float vfpacc2 = (float) vacc2;
        float vfpacc3 = (float) vacc3;

        vfpacc0 *= vscale;
        vfpacc1 *= vscale;
        vfpacc2 *= vscale;
        vfpacc3 *= vscale;

        vfpacc0 += vmagic_bias;
        vfpacc1 += vmagic_bias;
        vfpacc2 += vmagic_bias;
        vfpacc3 += vmagic_bias;

        int32_t vout0 = (int32_t) float_as_uint32(vfpacc0);
        int32_t vout1 = (int32_t) float_as_uint32(vfpacc1);
        int32_t vout2 = (int32_t) float_as_uint32(vfpacc2);
        int32_t vout3 = (int32_t) float_as_uint32(vfpacc3);

        vout0 = math_max_s32(vout0, vmagic_min);
        vout1 = math_max_s32(vout1, vmagic_min);
        vout2 = math_max_s32(vout2, vmagic_min);
        vout3 = math_max_s32(vout3, vmagic_min);

        vout0 = math_min_s32(vout0, vmagic_max);
        vout1 = math_min_s32(vout1, vmagic_max);
        vout2 = math_min_s32(vout2, vmagic_max);
        vout3 = math_min_s32(vout3, vmagic_max);

        vout0 -= vmagic_bias_less_zero_point;
        vout1 -= vmagic_bias_less_zero_point;
        vout2 -= vmagic_bias_less_zero_point;
        vout3 -= vmagic_bias_less_zero_point;

        output[0] = (uint8_t) vout0;
        output[1] = (uint8_t) vout1;
        output[2] = (uint8_t) vout2;
        output[3] = (uint8_t) vout3;
        output += 4;
      }
      if XNN_UNLIKELY(c != 0) {
        do {
          int32_t vacc = *b++;
          const int32_t vi0 = (int32_t) (uint32_t) *i0++;
          const int32_t vk0 = (int32_t) (uint32_t) ((const uint8_t*) w)[0] - vkernel_zero_point;
          vacc += vi0 * vk0;
          const int32_t vi1 = (int32_t) (uint32_t) *i1++;
          const int32_t vk1 = (int32_t) (uint32_t) ((const uint8_t*) w)[1] - vkernel_zero_point;
          vacc += vi1 * vk1;
          const int32_t vi2 = (int32_t) (uint32_t) *i2++;
          const int32_t vk2 = (int32_t) (uint32_t) ((const uint8_t*) w)[2] - vkernel_zero_point;
          vacc += vi2 * vk2;
          const int32_t vi3 = (int32_t) (uint32_t) *i3++;
          const int32_t vk3 = (int32_t) (uint32_t) ((const uint8_t*) w)[3] - vkernel_zero_point;
          vacc += vi3 * vk3;
          const int32_t vi4 = (int32_t) (uint32_t) *i4++;
          const int32_t vk4 = (int32_t) (uint32_t) ((const uint8_t*) w)[4] - vkernel_zero_point;
          vacc += vi4 * vk4;
          const int32_t vi5 = (int32_t) (uint32_t) *i5++;
          const int32_t vk5 = (int32_t) (uint32_t) ((const uint8_t*) w)[5] - vkernel_zero_point;
          vacc += vi5 * vk5;
          const int32_t vi6 = (int32_t) (uint32_t) *i6++;
          const int32_t vk6 = (int32_t) (uint32_t) ((const uint8_t*) w)[6] - vkernel_zero_point;
          vacc += vi6 * vk6;

          w = (const void*) ((uintptr_t) w + 7 * sizeof(uint8_t));

          float vfpacc = (float) vacc * vscale;

          vfpacc += vmagic_bias;
          int32_t vout = (int32_t) float_as_uint32(vfpacc);
          vout = math_max_s32(vout, vmagic_min);
          vout = math_min_s32(vout, vmagic_max);
          vout -= vmagic_bias_less_zero_point;

          *output++ = (uint8_t) vout;
        } while (--c != 0);
      }
    }

    input = (const uint8_t**) ((uintptr_t) input + input_stride);
    output = (uint8_t*) ((uintptr_t) output + output_increment);
  } while (--output_width != 0);
}
