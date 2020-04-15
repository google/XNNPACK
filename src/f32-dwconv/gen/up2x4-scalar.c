// Auto-generated file. Do not edit!
//   Template: src/f32-dwconv/up-scalar.c.in
//   Generator: tools/xngen
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <xnnpack/dwconv.h>
#include <xnnpack/math.h>


void xnn_f32_dwconv_ukernel_up2x4__scalar(
    size_t channels,
    size_t output_width,
    const float** input,
    const float* weights,
    float* output,
    size_t input_stride,
    size_t output_increment,
    const union xnn_f32_default_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(channels != 0);
  assert(output_width != 0);

  do {
    const float* i0 = input[0];
    assert(i0 != NULL);
    const float* i1 = input[1];
    assert(i1 != NULL);
    const float* i2 = input[2];
    assert(i2 != NULL);
    const float* i3 = input[3];
    assert(i3 != NULL);
    input = (const float**) ((uintptr_t) input + input_stride);

    size_t c = channels;
    const float* w = weights;
    for (; c >= 2; c -= 2) {
      float vacc0p0 = w[0];
      float vacc1p0 = w[1];


      const float vi0x0 = i0[0];
      const float vi0x1 = i0[1];
      i0 += 2;

      const float vk0x0 = w[2];
      vacc0p0 += vi0x0 * vk0x0;
      const float vk0x1 = w[3];
      vacc1p0 += vi0x1 * vk0x1;

      const float vi1x0 = i1[0];
      const float vi1x1 = i1[1];
      i1 += 2;

      const float vk1x0 = w[4];
      vacc0p0 += vi1x0 * vk1x0;
      const float vk1x1 = w[5];
      vacc1p0 += vi1x1 * vk1x1;

      const float vi2x0 = i2[0];
      const float vi2x1 = i2[1];
      i2 += 2;

      const float vk2x0 = w[6];
      vacc0p0 += vi2x0 * vk2x0;
      const float vk2x1 = w[7];
      vacc1p0 += vi2x1 * vk2x1;

      const float vi3x0 = i3[0];
      const float vi3x1 = i3[1];
      i3 += 2;

      const float vk3x0 = w[8];
      vacc0p0 += vi3x0 * vk3x0;
      const float vk3x1 = w[9];
      vacc1p0 += vi3x1 * vk3x1;

      w += 10;


      output[0] = vacc0p0;
      output[1] = vacc1p0;
      output += 2;
    }
    for (; c >= 1; c -= 1) {
      float vacc0p0 = *w++;

      const float vi0 = *i0++;
      const float vk0 = w[1];
      vacc0p0 += vi0 * vk0;
      const float vi1 = *i1++;
      const float vk1 = w[3];
      vacc0p0 += vi1 * vk1;
      const float vi2 = *i2++;
      const float vk2 = w[5];
      vacc0p0 += vi2 * vk2;
      const float vi3 = *i3++;
      const float vk3 = w[7];
      vacc0p0 += vi3 * vk3;


      *output++ = vacc0p0;
    }

    output = (float*) ((uintptr_t) output + output_increment);
  } while (--output_width != 0);
}
