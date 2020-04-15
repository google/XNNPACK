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


void xnn_f32_dwconv_ukernel_up1x9__wasm(
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
    const float* i4 = input[4];
    assert(i4 != NULL);
    const float* i5 = input[5];
    assert(i5 != NULL);
    const float* i6 = input[6];
    assert(i6 != NULL);
    const float* i7 = input[7];
    assert(i7 != NULL);
    const float* i8 = input[8];
    assert(i8 != NULL);
    input = (const float**) ((uintptr_t) input + input_stride);

    size_t c = channels;
    const float* w = weights;
    do {
      float vacc0p0 = w[0];

      const float vi0 = *i0++;
      const float vk0 = w[1];
      vacc0p0 += vi0 * vk0;

      const float vi1 = *i1++;
      const float vk1 = w[2];
      vacc0p0 += vi1 * vk1;

      const float vi2 = *i2++;
      const float vk2 = w[3];
      vacc0p0 += vi2 * vk2;

      const float vi3 = *i3++;
      const float vk3 = w[4];
      vacc0p0 += vi3 * vk3;

      const float vi4 = *i4++;
      const float vk4 = w[5];
      vacc0p0 += vi4 * vk4;

      const float vi5 = *i5++;
      const float vk5 = w[6];
      vacc0p0 += vi5 * vk5;

      const float vi6 = *i6++;
      const float vk6 = w[7];
      vacc0p0 += vi6 * vk6;

      const float vi7 = *i7++;
      const float vk7 = w[8];
      vacc0p0 += vi7 * vk7;

      const float vi8 = *i8++;
      const float vk8 = w[9];
      vacc0p0 += vi8 * vk8;

      w += 10;


      *output++ = vacc0p0;
    } while (--c != 0);

    output = (float*) ((uintptr_t) output + output_increment);
  } while (--output_width != 0);
}
