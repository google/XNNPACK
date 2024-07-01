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


void xnn_qs8_qc8w_dwconv_minmax_fp32_ukernel_5f5m5l1c1s1r__scalar_lrintf(
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
  assert(kernel_size > 5);

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
      do {
        int32_t vacc = unaligned_load_s32(w);
        const int32_t vi0 = (int32_t) *i0++;
        const int32_t vk0 = ((const int8_t*) ((uintptr_t) w + sizeof(int32_t)))[0];
        vacc += vi0 * vk0;
        const int32_t vi1 = (int32_t) *i1++;
        const int32_t vk1 = ((const int8_t*) ((uintptr_t) w + sizeof(int32_t)))[1];
        vacc += vi1 * vk1;
        const int32_t vi2 = (int32_t) *i2++;
        const int32_t vk2 = ((const int8_t*) ((uintptr_t) w + sizeof(int32_t)))[2];
        vacc += vi2 * vk2;
        const int32_t vi3 = (int32_t) *i3++;
        const int32_t vk3 = ((const int8_t*) ((uintptr_t) w + sizeof(int32_t)))[3];
        vacc += vi3 * vk3;
        const int32_t vi4 = (int32_t) *i4++;
        const int32_t vk4 = ((const int8_t*) ((uintptr_t) w + sizeof(int32_t)))[4];
        vacc += vi4 * vk4;

        w = (const void*) ((uintptr_t) w + sizeof(int32_t) + 5 * sizeof(int8_t));
        *b++ = vacc;
      } while (--c != 0);
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
      do {
        int32_t vacc = *b;
        const int32_t vi0 = (int32_t) *i0++;
        const int32_t vk0 = ((const int8_t*) w)[0];
        vacc += vi0 * vk0;
        const int32_t vi1 = (int32_t) *i1++;
        const int32_t vk1 = ((const int8_t*) w)[1];
        vacc += vi1 * vk1;
        const int32_t vi2 = (int32_t) *i2++;
        const int32_t vk2 = ((const int8_t*) w)[2];
        vacc += vi2 * vk2;
        const int32_t vi3 = (int32_t) *i3++;
        const int32_t vk3 = ((const int8_t*) w)[3];
        vacc += vi3 * vk3;
        const int32_t vi4 = (int32_t) *i4++;
        const int32_t vk4 = ((const int8_t*) w)[4];
        vacc += vi4 * vk4;

        w = (const void*) ((uintptr_t) w + 5 * sizeof(int8_t));
        *b++ = vacc;
      } while (--c != 0);
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
      do {
        int32_t vacc = unaligned_load_s32(b++);
        const int32_t vi0 = (int32_t) *i0++;
        const int32_t vk0 = ((const int8_t*) w)[0];
        vacc += vi0 * vk0;
        const int32_t vi1 = (int32_t) *i1++;
        const int32_t vk1 = ((const int8_t*) w)[1];
        vacc += vi1 * vk1;
        const int32_t vi2 = (int32_t) *i2++;
        const int32_t vk2 = ((const int8_t*) w)[2];
        vacc += vi2 * vk2;
        const int32_t vi3 = (int32_t) *i3++;
        const int32_t vk3 = ((const int8_t*) w)[3];
        vacc += vi3 * vk3;
        const int32_t vi4 = (int32_t) *i4++;
        const int32_t vk4 = ((const int8_t*) w)[4];
        vacc += vi4 * vk4;

        w = (const void*) ((uintptr_t) w + 5 * sizeof(int8_t));

        const float vscale = unaligned_load_f32(w);
        w = (const void*) ((const float*) w + 1);
        float vfpacc = (float) vacc * vscale;

        vfpacc = math_max_f32(vfpacc, voutput_min_less_zero_point);
        vfpacc = math_min_f32(vfpacc, voutput_max_less_zero_point);
        const int32_t vrndacc = (int32_t) lrintf(vfpacc);
        int32_t vout = vrndacc + voutput_zero_point;

        *output++ = (int8_t) vout;
      } while (--c != 0);
    }

    input = (const int8_t**) ((uintptr_t) input + input_stride);
    output = (int8_t*) ((uintptr_t) output + output_increment);
  } while (--output_width != 0);
}
