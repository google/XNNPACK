// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <xnnpack/avgpool.h>
#include <xnnpack/math.h>


void xnn_qu8_avgpool_minmax_fp32_ukernel_9p8x__scalar_imagic_c1(
    size_t output_pixels,
    size_t kernel_elements,
    size_t channels,
    const uint8_t** input,
    size_t input_offset,
    const uint8_t* zero,
    int32_t* buffer,
    uint8_t* output,
    size_t input_increment,
    size_t output_increment,
    const union xnn_qu8_avgpool_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(output_pixels != 0);
  assert(kernel_elements > 9);
  assert(channels != 0);

  const int32_t vinit_bias = params->fp32_scalar_imagic.init_bias;
  const float vscale = params->fp32_scalar_imagic.scale;
  const float vmagic_bias = params->fp32_scalar_imagic.magic_bias;
  const int32_t vmagic_min = params->fp32_scalar_imagic.magic_min;
  const int32_t vmagic_max = params->fp32_scalar_imagic.magic_max;
  const int32_t vmagic_bias_less_zero_point = params->fp32_scalar_imagic.magic_bias_less_zero_point;
  do {
    // First pass.
    {
      const uint8_t* i0 = *input++;
      assert(i0 != NULL);
      if XNN_UNPREDICTABLE(i0 != zero) {
        i0 = (const uint8_t*) ((uintptr_t) i0 + input_offset);
      }
      const uint8_t* i1 = *input++;
      assert(i1 != NULL);
      if XNN_UNPREDICTABLE(i1 != zero) {
        i1 = (const uint8_t*) ((uintptr_t) i1 + input_offset);
      }
      const uint8_t* i2 = *input++;
      assert(i2 != NULL);
      if XNN_UNPREDICTABLE(i2 != zero) {
        i2 = (const uint8_t*) ((uintptr_t) i2 + input_offset);
      }
      const uint8_t* i3 = *input++;
      assert(i3 != NULL);
      if XNN_UNPREDICTABLE(i3 != zero) {
        i3 = (const uint8_t*) ((uintptr_t) i3 + input_offset);
      }
      const uint8_t* i4 = *input++;
      assert(i4 != NULL);
      if XNN_UNPREDICTABLE(i4 != zero) {
        i4 = (const uint8_t*) ((uintptr_t) i4 + input_offset);
      }
      const uint8_t* i5 = *input++;
      assert(i5 != NULL);
      if XNN_UNPREDICTABLE(i5 != zero) {
        i5 = (const uint8_t*) ((uintptr_t) i5 + input_offset);
      }
      const uint8_t* i6 = *input++;
      assert(i6 != NULL);
      if XNN_UNPREDICTABLE(i6 != zero) {
        i6 = (const uint8_t*) ((uintptr_t) i6 + input_offset);
      }
      const uint8_t* i7 = *input++;
      assert(i7 != NULL);
      if XNN_UNPREDICTABLE(i7 != zero) {
        i7 = (const uint8_t*) ((uintptr_t) i7 + input_offset);
      }
      const uint8_t* i8 = *input++;
      assert(i8 != NULL);
      if XNN_UNPREDICTABLE(i8 != zero) {
        i8 = (const uint8_t*) ((uintptr_t) i8 + input_offset);
      }

      int32_t* b = buffer;
      size_t c = channels;
      do {
        int32_t vacc = vinit_bias;

        const int32_t vi0 = (int32_t) (uint32_t) *i0++;
        vacc += vi0;
        const int32_t vi1 = (int32_t) (uint32_t) *i1++;
        vacc += vi1;
        const int32_t vi2 = (int32_t) (uint32_t) *i2++;
        vacc += vi2;
        const int32_t vi3 = (int32_t) (uint32_t) *i3++;
        vacc += vi3;
        const int32_t vi4 = (int32_t) (uint32_t) *i4++;
        vacc += vi4;
        const int32_t vi5 = (int32_t) (uint32_t) *i5++;
        vacc += vi5;
        const int32_t vi6 = (int32_t) (uint32_t) *i6++;
        vacc += vi6;
        const int32_t vi7 = (int32_t) (uint32_t) *i7++;
        vacc += vi7;
        const int32_t vi8 = (int32_t) (uint32_t) *i8++;
        vacc += vi8;

        *b++ = vacc;
      } while (--c != 0);
    }

    size_t k = kernel_elements;
    // Intermediate passes.
    for (k -= 9; k > 8; k -= 8) {
      const uint8_t* i0 = *input++;
      assert(i0 != NULL);
      if XNN_UNPREDICTABLE(i0 != zero) {
        i0 = (const uint8_t*) ((uintptr_t) i0 + input_offset);
      }
      const uint8_t* i1 = *input++;
      assert(i1 != NULL);
      if XNN_UNPREDICTABLE(i1 != zero) {
        i1 = (const uint8_t*) ((uintptr_t) i1 + input_offset);
      }
      const uint8_t* i2 = *input++;
      assert(i2 != NULL);
      if XNN_UNPREDICTABLE(i2 != zero) {
        i2 = (const uint8_t*) ((uintptr_t) i2 + input_offset);
      }
      const uint8_t* i3 = *input++;
      assert(i3 != NULL);
      if XNN_UNPREDICTABLE(i3 != zero) {
        i3 = (const uint8_t*) ((uintptr_t) i3 + input_offset);
      }
      const uint8_t* i4 = *input++;
      assert(i4 != NULL);
      if XNN_UNPREDICTABLE(i4 != zero) {
        i4 = (const uint8_t*) ((uintptr_t) i4 + input_offset);
      }
      const uint8_t* i5 = *input++;
      assert(i5 != NULL);
      if XNN_UNPREDICTABLE(i5 != zero) {
        i5 = (const uint8_t*) ((uintptr_t) i5 + input_offset);
      }
      const uint8_t* i6 = *input++;
      assert(i6 != NULL);
      if XNN_UNPREDICTABLE(i6 != zero) {
        i6 = (const uint8_t*) ((uintptr_t) i6 + input_offset);
      }
      const uint8_t* i7 = *input++;
      assert(i7 != NULL);
      if XNN_UNPREDICTABLE(i7 != zero) {
        i7 = (const uint8_t*) ((uintptr_t) i7 + input_offset);
      }

      int32_t* b = buffer;
      size_t c = channels;
      do {
        int32_t vacc = *b;

        const int32_t vi0 = (int32_t) (uint32_t) *i0++;
        vacc += vi0;
        const int32_t vi1 = (int32_t) (uint32_t) *i1++;
        vacc += vi1;
        const int32_t vi2 = (int32_t) (uint32_t) *i2++;
        vacc += vi2;
        const int32_t vi3 = (int32_t) (uint32_t) *i3++;
        vacc += vi3;
        const int32_t vi4 = (int32_t) (uint32_t) *i4++;
        vacc += vi4;
        const int32_t vi5 = (int32_t) (uint32_t) *i5++;
        vacc += vi5;
        const int32_t vi6 = (int32_t) (uint32_t) *i6++;
        vacc += vi6;
        const int32_t vi7 = (int32_t) (uint32_t) *i7++;
        vacc += vi7;

        *b++ = vacc;
      } while (--c != 0);
    }

    // Last pass.
    {
      const uint8_t* i0 = input[0];
      assert(i0 != NULL);
      const uint8_t* i1 = input[1];
      const uint8_t* i2 = input[2];
      const uint8_t* i3 = input[3];
      const uint8_t* i4 = input[4];
      const uint8_t* i5 = input[5];
      const uint8_t* i6 = input[6];
      const uint8_t* i7 = input[7];
      input = (const uint8_t**) ((uintptr_t) input + input_increment);
      if (k < 2) {
        i1 = zero;
      }
      assert(i1 != NULL);
      if (k <= 2) {
        i2 = zero;
      }
      assert(i2 != NULL);
      if (k < 4) {
        i3 = zero;
      }
      assert(i3 != NULL);
      if (k <= 4) {
        i4 = zero;
      }
      assert(i4 != NULL);
      if (k < 6) {
        i5 = zero;
      }
      assert(i5 != NULL);
      if (k <= 6) {
        i6 = zero;
      }
      assert(i6 != NULL);
      if (k < 8) {
        i7 = zero;
      }
      assert(i7 != NULL);
      if XNN_UNPREDICTABLE(i0 != zero) {
        i0 = (const uint8_t*) ((uintptr_t) i0 + input_offset);
      }
      if XNN_UNPREDICTABLE(i1 != zero) {
        i1 = (const uint8_t*) ((uintptr_t) i1 + input_offset);
      }
      if XNN_UNPREDICTABLE(i2 != zero) {
        i2 = (const uint8_t*) ((uintptr_t) i2 + input_offset);
      }
      if XNN_UNPREDICTABLE(i3 != zero) {
        i3 = (const uint8_t*) ((uintptr_t) i3 + input_offset);
      }
      if XNN_UNPREDICTABLE(i4 != zero) {
        i4 = (const uint8_t*) ((uintptr_t) i4 + input_offset);
      }
      if XNN_UNPREDICTABLE(i5 != zero) {
        i5 = (const uint8_t*) ((uintptr_t) i5 + input_offset);
      }
      if XNN_UNPREDICTABLE(i6 != zero) {
        i6 = (const uint8_t*) ((uintptr_t) i6 + input_offset);
      }
      if XNN_UNPREDICTABLE(i7 != zero) {
        i7 = (const uint8_t*) ((uintptr_t) i7 + input_offset);
      }

      size_t c = channels;
      int32_t* b = buffer;
      do {
        int32_t vacc = *b++;

        const int32_t vi0 = (int32_t) (uint32_t) *i0++;
        vacc += vi0;
        const int32_t vi1 = (int32_t) (uint32_t) *i1++;
        vacc += vi1;
        const int32_t vi2 = (int32_t) (uint32_t) *i2++;
        vacc += vi2;
        const int32_t vi3 = (int32_t) (uint32_t) *i3++;
        vacc += vi3;
        const int32_t vi4 = (int32_t) (uint32_t) *i4++;
        vacc += vi4;
        const int32_t vi5 = (int32_t) (uint32_t) *i5++;
        vacc += vi5;
        const int32_t vi6 = (int32_t) (uint32_t) *i6++;
        vacc += vi6;
        const int32_t vi7 = (int32_t) (uint32_t) *i7++;
        vacc += vi7;

        float vfpacc = (float) vacc * vscale;
        vfpacc += vmagic_bias;
        int32_t vout = (int32_t) float_as_uint32(vfpacc);
        vout = math_max_s32(vout, vmagic_min);
        vout = math_min_s32(vout, vmagic_max);
        vout -= vmagic_bias_less_zero_point;

        *output++ = (uint8_t) vout;
      } while (--c != 0);
    }
    output = (uint8_t*) ((uintptr_t) output + output_increment);
  } while (--output_pixels != 0);
}
