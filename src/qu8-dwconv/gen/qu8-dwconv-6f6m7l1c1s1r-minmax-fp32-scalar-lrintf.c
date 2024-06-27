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


void xnn_qu8_dwconv_minmax_fp32_ukernel_6f6m7l1c1s1r__scalar_lrintf(
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

  const float vscale = params->fp32_scalar_lrintf.scale;
  const float voutput_min_less_zero_point = params->fp32_scalar_lrintf.output_min_less_zero_point;
  const float voutput_max_less_zero_point = params->fp32_scalar_lrintf.output_max_less_zero_point;
  const int32_t voutput_zero_point = params->fp32_scalar_lrintf.output_zero_point;
  const int32_t vkernel_zero_point = params->fp32_scalar_lrintf.kernel_zero_point;
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
      do {
        int32_t vacc = unaligned_load_s32(w);
        const int32_t vi0 = (int32_t) (uint32_t) *i0++;
        const int32_t vk0 = (int32_t) (uint32_t) ((const uint8_t*) ((uintptr_t) w + sizeof(int32_t)))[0] - vkernel_zero_point;
        vacc += vi0 * vk0;
        const int32_t vi1 = (int32_t) (uint32_t) *i1++;
        const int32_t vk1 = (int32_t) (uint32_t) ((const uint8_t*) ((uintptr_t) w + sizeof(int32_t)))[1] - vkernel_zero_point;
        vacc += vi1 * vk1;
        const int32_t vi2 = (int32_t) (uint32_t) *i2++;
        const int32_t vk2 = (int32_t) (uint32_t) ((const uint8_t*) ((uintptr_t) w + sizeof(int32_t)))[2] - vkernel_zero_point;
        vacc += vi2 * vk2;
        const int32_t vi3 = (int32_t) (uint32_t) *i3++;
        const int32_t vk3 = (int32_t) (uint32_t) ((const uint8_t*) ((uintptr_t) w + sizeof(int32_t)))[3] - vkernel_zero_point;
        vacc += vi3 * vk3;
        const int32_t vi4 = (int32_t) (uint32_t) *i4++;
        const int32_t vk4 = (int32_t) (uint32_t) ((const uint8_t*) ((uintptr_t) w + sizeof(int32_t)))[4] - vkernel_zero_point;
        vacc += vi4 * vk4;
        const int32_t vi5 = (int32_t) (uint32_t) *i5++;
        const int32_t vk5 = (int32_t) (uint32_t) ((const uint8_t*) ((uintptr_t) w + sizeof(int32_t)))[5] - vkernel_zero_point;
        vacc += vi5 * vk5;

        w = (const void*) ((uintptr_t) w + sizeof(int32_t) + 6 * sizeof(uint8_t));
        *b++ = vacc;
      } while (--c != 0);
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
      do {
        int32_t vacc = unaligned_load_s32(b++);
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

        vfpacc = math_max_f32(vfpacc, voutput_min_less_zero_point);
        vfpacc = math_min_f32(vfpacc, voutput_max_less_zero_point);
        const int32_t vrndacc = (int32_t) lrintf(vfpacc);
        int32_t vout = vrndacc + voutput_zero_point;

        *output++ = (uint8_t) vout;
      } while (--c != 0);
    }

    input = (const uint8_t**) ((uintptr_t) input + input_stride);
    output = (uint8_t*) ((uintptr_t) output + output_increment);
  } while (--output_width != 0);
}
