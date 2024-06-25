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


void xnn_f32_dwconv_minmax_ukernel_2f2m2l4c1s1r__scalar(
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
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(channels != 0);
  assert(output_width != 0);
  assert(kernel_size > 2);

  const float vmin = params->scalar.min;
  const float vmax = params->scalar.max;
  do {
    const float* w = weights;

    // First pass to process 2 inputs.
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
      input += 2;

      // Process c channels and write to buffer.
      size_t c = round_up_po2(channels, 1);
      for (; c >= 4; c -= 4) {
        float vacc0p0 = w[0];
        float vacc1p0 = w[1];
        float vacc2p0 = w[2];
        float vacc3p0 = w[3];


        const float vi0x0 = i0[0];
        const float vi0x1 = i0[1];
        const float vi0x2 = i0[2];
        const float vi0x3 = i0[3];
        i0 += 4;

        const float vk0x0 = w[4];
        const float vk0x1 = w[5];
        const float vk0x2 = w[6];
        const float vk0x3 = w[7];
        vacc0p0 = math_muladd_f32(vi0x0, vk0x0, vacc0p0);
        vacc1p0 = math_muladd_f32(vi0x1, vk0x1, vacc1p0);
        vacc2p0 = math_muladd_f32(vi0x2, vk0x2, vacc2p0);
        vacc3p0 = math_muladd_f32(vi0x3, vk0x3, vacc3p0);

        const float vi1x0 = i1[0];
        const float vi1x1 = i1[1];
        const float vi1x2 = i1[2];
        const float vi1x3 = i1[3];
        i1 += 4;

        const float vk1x0 = w[8];
        const float vk1x1 = w[9];
        const float vk1x2 = w[10];
        const float vk1x3 = w[11];
        vacc0p0 = math_muladd_f32(vi1x0, vk1x0, vacc0p0);
        vacc1p0 = math_muladd_f32(vi1x1, vk1x1, vacc1p0);
        vacc2p0 = math_muladd_f32(vi1x2, vk1x2, vacc2p0);
        vacc3p0 = math_muladd_f32(vi1x3, vk1x3, vacc3p0);

        w += 12;


        b[0] = vacc0p0;
        b[1] = vacc1p0;
        b[2] = vacc2p0;
        b[3] = vacc3p0;
        b += 4;
      }


      for (; c != 0; c --) {
        float vacc0p0 = w[0];

        const float vi0x0 = i0[0];
        i0 += 1;

        const float vk0x0 = w[1];
        vacc0p0 = math_muladd_f32(vi0x0, vk0x0, vacc0p0);

        const float vi1x0 = i1[0];
        i1 += 1;

        const float vk1x0 = w[2];
        vacc0p0 = math_muladd_f32(vi1x0, vk1x0, vacc0p0);

        w += 3;


        b[0] = vacc0p0;
        b += 1;
      }
    }

    // Middle pass to process 2 inputs in each iteration.
    for (size_t ks = kernel_size - 2; ks > 2; ks -= 2) {
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
      input += 2;

      size_t c = round_up_po2(channels, 1);
      for (; c >= 4; c -= 4) {
        float vacc0p0 = b[0];
        float vacc1p0 = b[1];
        float vacc2p0 = b[2];
        float vacc3p0 = b[3];


        const float vi0x0 = i0[0];
        const float vi0x1 = i0[1];
        const float vi0x2 = i0[2];
        const float vi0x3 = i0[3];
        i0 += 4;

        const float vk0x0 = w[0];
        const float vk0x1 = w[1];
        const float vk0x2 = w[2];
        const float vk0x3 = w[3];
        vacc0p0 = math_muladd_f32(vi0x0, vk0x0, vacc0p0);
        vacc1p0 = math_muladd_f32(vi0x1, vk0x1, vacc1p0);
        vacc2p0 = math_muladd_f32(vi0x2, vk0x2, vacc2p0);
        vacc3p0 = math_muladd_f32(vi0x3, vk0x3, vacc3p0);

        const float vi1x0 = i1[0];
        const float vi1x1 = i1[1];
        const float vi1x2 = i1[2];
        const float vi1x3 = i1[3];
        i1 += 4;

        const float vk1x0 = w[4];
        const float vk1x1 = w[5];
        const float vk1x2 = w[6];
        const float vk1x3 = w[7];
        vacc0p0 = math_muladd_f32(vi1x0, vk1x0, vacc0p0);
        vacc1p0 = math_muladd_f32(vi1x1, vk1x1, vacc1p0);
        vacc2p0 = math_muladd_f32(vi1x2, vk1x2, vacc2p0);
        vacc3p0 = math_muladd_f32(vi1x3, vk1x3, vacc3p0);

        w += 8;


        b[0] = vacc0p0;
        b[1] = vacc1p0;
        b[2] = vacc2p0;
        b[3] = vacc3p0;
        b += 4;
      }

      for (; c != 0; c --) {
        float vacc0p0 = b[0];


        const float vi0x0 = i0[0];
        i0 += 1;

        const float vk0x0 = w[0];
        vacc0p0 = math_muladd_f32(vi0x0, vk0x0, vacc0p0);

        const float vi1x0 = i1[0];
        i1 += 1;

        const float vk1x0 = w[1];
        vacc0p0 = math_muladd_f32(vi1x0, vk1x0, vacc0p0);

        w += 2;


        b[0] = vacc0p0;
        b += 1;
      }
    }

    // Last pass to process up to 2 inputs.
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

      size_t c = channels;
      for (; c >= 4; c -= 4) {
        float vacc0p0 = b[0];
        float vacc1p0 = b[1];
        float vacc2p0 = b[2];
        float vacc3p0 = b[3];
        b += 4;


        const float vi0x0 = i0[0];
        const float vi0x1 = i0[1];
        const float vi0x2 = i0[2];
        const float vi0x3 = i0[3];
        i0 += 4;

        const float vk0x0 = w[0];
        const float vk0x1 = w[1];
        const float vk0x2 = w[2];
        const float vk0x3 = w[3];
        vacc0p0 = math_muladd_f32(vi0x0, vk0x0, vacc0p0);
        vacc1p0 = math_muladd_f32(vi0x1, vk0x1, vacc1p0);
        vacc2p0 = math_muladd_f32(vi0x2, vk0x2, vacc2p0);
        vacc3p0 = math_muladd_f32(vi0x3, vk0x3, vacc3p0);

        const float vi1x0 = i1[0];
        const float vi1x1 = i1[1];
        const float vi1x2 = i1[2];
        const float vi1x3 = i1[3];
        i1 += 4;

        const float vk1x0 = w[4];
        const float vk1x1 = w[5];
        const float vk1x2 = w[6];
        const float vk1x3 = w[7];
        vacc0p0 = math_muladd_f32(vi1x0, vk1x0, vacc0p0);
        vacc1p0 = math_muladd_f32(vi1x1, vk1x1, vacc1p0);
        vacc2p0 = math_muladd_f32(vi1x2, vk1x2, vacc2p0);
        vacc3p0 = math_muladd_f32(vi1x3, vk1x3, vacc3p0);

        w += 8;


        float vacc0 = math_max_f32(vacc0p0, vmin);
        float vacc1 = math_max_f32(vacc1p0, vmin);
        float vacc2 = math_max_f32(vacc2p0, vmin);
        float vacc3 = math_max_f32(vacc3p0, vmin);

        vacc0 = math_min_f32(vacc0, vmax);
        vacc1 = math_min_f32(vacc1, vmax);
        vacc2 = math_min_f32(vacc2, vmax);
        vacc3 = math_min_f32(vacc3, vmax);

        output[0] = vacc0;
        output[1] = vacc1;
        output[2] = vacc2;
        output[3] = vacc3;
        output += 4;
      }
      for (; c != 0; c --) {
        float vacc0p0 = *b++;

        const float vi0 = *i0++;
        const float vk0 = w[0];
        vacc0p0 = math_muladd_f32(vi0, vk0, vacc0p0);
        const float vi1 = *i1++;
        const float vk1 = w[1];
        vacc0p0 = math_muladd_f32(vi1, vk1, vacc0p0);
        w += 2;


        float vacc0 = math_max_f32(vacc0p0, vmin);
        vacc0 = math_min_f32(vacc0, vmax);
        *output++ = vacc0;
      }

    }
    input = (const float**) ((uintptr_t) input + input_stride);
    output = (float*) ((uintptr_t) output + output_increment);
  } while (--output_width != 0);
}
