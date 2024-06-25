// Auto-generated file. Do not edit!
//   Template: src/f32-dwconv/multipass-scalar.c.in
//   Generator: tools/xngen
//
// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include "xnnpack/dwconv.h"
#include "xnnpack/math.h"


void xnn_f32_dwconv_ukernel_3f3m3l1c1s1r__scalar(
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
    const union xnn_f32_default_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(channels != 0);
  assert(output_width != 0);
  assert(kernel_size > 3);

  do {
    const float* w = weights;

    // First pass to process 3 inputs.
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
      input += 3;

      // Process c channels and write to buffer.
      for (size_t c = channels; c >= 1; c -= 1) {
        float vacc0p0 = w[0];

        const float vi0 = *i0++;
        const float vk0 = w[1];
        vacc0p0 = math_muladd_f32(vi0, vk0, vacc0p0);
        const float vi1 = *i1++;
        const float vk1 = w[2];
        vacc0p0 = math_muladd_f32(vi1, vk1, vacc0p0);
        const float vi2 = *i2++;
        const float vk2 = w[3];
        vacc0p0 = math_muladd_f32(vi2, vk2, vacc0p0);

        w += 4;


        *b++ = vacc0p0;
      }
    }

    // Middle pass to process 3 inputs in each iteration.
    for (size_t ks = kernel_size - 3; ks > 3; ks -= 3) {
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
      input += 3;

      for (size_t c = channels; c >= 1; c -= 1) {
        float vacc0p0 = *b;

        const float vi0 = *i0++;
        const float vk0 = w[0];
        vacc0p0 = math_muladd_f32(vi0, vk0, vacc0p0);
        const float vi1 = *i1++;
        const float vk1 = w[1];
        vacc0p0 = math_muladd_f32(vi1, vk1, vacc0p0);
        const float vi2 = *i2++;
        const float vk2 = w[2];
        vacc0p0 = math_muladd_f32(vi2, vk2, vacc0p0);


        w += 3;
        *b++ = vacc0p0;
      }
    }

    // Last pass to process up to 3 inputs.
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

      for (size_t c = channels; c >= 1; c -= 1) {
        float vacc0p0 = *b++;

        const float vi0 = *i0++;
        const float vk0 = w[0];
        vacc0p0 = math_muladd_f32(vi0, vk0, vacc0p0);
        const float vi1 = *i1++;
        const float vk1 = w[1];
        vacc0p0 = math_muladd_f32(vi1, vk1, vacc0p0);
        const float vi2 = *i2++;
        const float vk2 = w[2];
        vacc0p0 = math_muladd_f32(vi2, vk2, vacc0p0);

        w += 3;


        *output++ = vacc0p0;
      }

    }
    input = (const float**) ((uintptr_t) input + input_stride);
    output = (float*) ((uintptr_t) output + output_increment);
  } while (--output_width != 0);
}
