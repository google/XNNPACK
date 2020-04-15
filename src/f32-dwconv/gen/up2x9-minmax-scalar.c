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


void xnn_f32_dwconv_minmax_ukernel_up2x9__scalar(
    size_t channels,
    size_t output_width,
    const float** input,
    const float* weights,
    float* output,
    size_t input_stride,
    size_t output_increment,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(channels != 0);
  assert(output_width != 0);

  const float vmin = params->scalar.min;
  const float vmax = params->scalar.max;
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

      const float vi4x0 = i4[0];
      const float vi4x1 = i4[1];
      i4 += 2;

      const float vk4x0 = w[10];
      vacc0p0 += vi4x0 * vk4x0;
      const float vk4x1 = w[11];
      vacc1p0 += vi4x1 * vk4x1;

      const float vi5x0 = i5[0];
      const float vi5x1 = i5[1];
      i5 += 2;

      const float vk5x0 = w[12];
      vacc0p0 += vi5x0 * vk5x0;
      const float vk5x1 = w[13];
      vacc1p0 += vi5x1 * vk5x1;

      const float vi6x0 = i6[0];
      const float vi6x1 = i6[1];
      i6 += 2;

      const float vk6x0 = w[14];
      vacc0p0 += vi6x0 * vk6x0;
      const float vk6x1 = w[15];
      vacc1p0 += vi6x1 * vk6x1;

      const float vi7x0 = i7[0];
      const float vi7x1 = i7[1];
      i7 += 2;

      const float vk7x0 = w[16];
      vacc0p0 += vi7x0 * vk7x0;
      const float vk7x1 = w[17];
      vacc1p0 += vi7x1 * vk7x1;

      const float vi8x0 = i8[0];
      const float vi8x1 = i8[1];
      i8 += 2;

      const float vk8x0 = w[18];
      vacc0p0 += vi8x0 * vk8x0;
      const float vk8x1 = w[19];
      vacc1p0 += vi8x1 * vk8x1;

      w += 20;


      float vacc0 = math_max_f32(vacc0p0, vmin);
      float vacc1 = math_max_f32(vacc1p0, vmin);

      vacc0 = math_min_f32(vacc0, vmax);
      vacc1 = math_min_f32(vacc1, vmax);

      output[0] = vacc0;
      output[1] = vacc1;
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
      const float vi4 = *i4++;
      const float vk4 = w[9];
      vacc0p0 += vi4 * vk4;
      const float vi5 = *i5++;
      const float vk5 = w[11];
      vacc0p0 += vi5 * vk5;
      const float vi6 = *i6++;
      const float vk6 = w[13];
      vacc0p0 += vi6 * vk6;
      const float vi7 = *i7++;
      const float vk7 = w[15];
      vacc0p0 += vi7 * vk7;
      const float vi8 = *i8++;
      const float vk8 = w[17];
      vacc0p0 += vi8 * vk8;


      float vacc0 = math_max_f32(vacc0p0, vmin);
      vacc0 = math_min_f32(vacc0, vmax);
      *output++ = vacc0;
    }

    output = (float*) ((uintptr_t) output + output_increment);
  } while (--output_width != 0);
}
