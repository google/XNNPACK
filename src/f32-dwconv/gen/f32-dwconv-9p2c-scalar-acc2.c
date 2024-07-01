// Auto-generated file. Do not edit!
//   Template: src/f32-dwconv/unipass-scalar.c.in
//   Generator: tools/xngen
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include "xnnpack/dwconv.h"
#include "xnnpack/math.h"


void xnn_f32_dwconv_ukernel_9p2c__scalar_acc2(
    size_t channels,
    size_t output_width,
    const float** input,
    const float* weights,
    float* output,
    intptr_t input_stride,
    size_t output_increment,
    size_t input_offset,
    const float* zero,
    const union xnn_f32_default_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(channels != 0);
  assert(output_width != 0);

  do {
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
    const float* i3 = input[3];
    assert(i3 != NULL);
    if XNN_UNPREDICTABLE(i3 != zero) {
      i3 = (const float*) ((uintptr_t) i3 + input_offset);
    }
    const float* i4 = input[4];
    assert(i4 != NULL);
    if XNN_UNPREDICTABLE(i4 != zero) {
      i4 = (const float*) ((uintptr_t) i4 + input_offset);
    }
    const float* i5 = input[5];
    assert(i5 != NULL);
    if XNN_UNPREDICTABLE(i5 != zero) {
      i5 = (const float*) ((uintptr_t) i5 + input_offset);
    }
    const float* i6 = input[6];
    assert(i6 != NULL);
    if XNN_UNPREDICTABLE(i6 != zero) {
      i6 = (const float*) ((uintptr_t) i6 + input_offset);
    }
    const float* i7 = input[7];
    assert(i7 != NULL);
    if XNN_UNPREDICTABLE(i7 != zero) {
      i7 = (const float*) ((uintptr_t) i7 + input_offset);
    }
    const float* i8 = input[8];
    assert(i8 != NULL);
    if XNN_UNPREDICTABLE(i8 != zero) {
      i8 = (const float*) ((uintptr_t) i8 + input_offset);
    }
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
      vacc0p0 = math_muladd_f32(vi0x0, vk0x0, vacc0p0);
      const float vk0x1 = w[3];
      vacc1p0 = math_muladd_f32(vi0x1, vk0x1, vacc1p0);

      const float vi1x0 = i1[0];
      const float vi1x1 = i1[1];
      i1 += 2;

      const float vk1x0 = w[4];
      float vacc0p1 = vi1x0 * vk1x0;
      const float vk1x1 = w[5];
      float vacc1p1 = vi1x1 * vk1x1;

      const float vi2x0 = i2[0];
      const float vi2x1 = i2[1];
      i2 += 2;

      const float vk2x0 = w[6];
      vacc0p0 = math_muladd_f32(vi2x0, vk2x0, vacc0p0);
      const float vk2x1 = w[7];
      vacc1p0 = math_muladd_f32(vi2x1, vk2x1, vacc1p0);

      const float vi3x0 = i3[0];
      const float vi3x1 = i3[1];
      i3 += 2;

      const float vk3x0 = w[8];
      vacc0p1 = math_muladd_f32(vi3x0, vk3x0, vacc0p1);
      const float vk3x1 = w[9];
      vacc1p1 = math_muladd_f32(vi3x1, vk3x1, vacc1p1);

      const float vi4x0 = i4[0];
      const float vi4x1 = i4[1];
      i4 += 2;

      const float vk4x0 = w[10];
      vacc0p0 = math_muladd_f32(vi4x0, vk4x0, vacc0p0);
      const float vk4x1 = w[11];
      vacc1p0 = math_muladd_f32(vi4x1, vk4x1, vacc1p0);

      const float vi5x0 = i5[0];
      const float vi5x1 = i5[1];
      i5 += 2;

      const float vk5x0 = w[12];
      vacc0p1 = math_muladd_f32(vi5x0, vk5x0, vacc0p1);
      const float vk5x1 = w[13];
      vacc1p1 = math_muladd_f32(vi5x1, vk5x1, vacc1p1);

      const float vi6x0 = i6[0];
      const float vi6x1 = i6[1];
      i6 += 2;

      const float vk6x0 = w[14];
      vacc0p0 = math_muladd_f32(vi6x0, vk6x0, vacc0p0);
      const float vk6x1 = w[15];
      vacc1p0 = math_muladd_f32(vi6x1, vk6x1, vacc1p0);

      const float vi7x0 = i7[0];
      const float vi7x1 = i7[1];
      i7 += 2;

      const float vk7x0 = w[16];
      vacc0p1 = math_muladd_f32(vi7x0, vk7x0, vacc0p1);
      const float vk7x1 = w[17];
      vacc1p1 = math_muladd_f32(vi7x1, vk7x1, vacc1p1);

      const float vi8x0 = i8[0];
      const float vi8x1 = i8[1];
      i8 += 2;

      const float vk8x0 = w[18];
      vacc0p0 = math_muladd_f32(vi8x0, vk8x0, vacc0p0);
      const float vk8x1 = w[19];
      vacc1p0 = math_muladd_f32(vi8x1, vk8x1, vacc1p0);

      w += 20;

      // Add up all accumulators to vacc01p0
      vacc0p0 = vacc0p0 + vacc0p1;
      vacc1p0 = vacc1p0 + vacc1p1;

      output[0] = vacc0p0;
      output[1] = vacc1p0;
      output += 2;
    }
    for (; c >= 1; c -= 1) {
      float vacc0p0 = *w++;

      const float vi0 = *i0++;
      const float vk0 = w[1];
      vacc0p0 = math_muladd_f32(vi0, vk0, vacc0p0);
      const float vi1 = *i1++;
      const float vk1 = w[3];
      float vacc0p1 = vi1 * vk1;
      const float vi2 = *i2++;
      const float vk2 = w[5];
      vacc0p0 = math_muladd_f32(vi2, vk2, vacc0p0);
      const float vi3 = *i3++;
      const float vk3 = w[7];
      vacc0p1 = math_muladd_f32(vi3, vk3, vacc0p1);
      const float vi4 = *i4++;
      const float vk4 = w[9];
      vacc0p0 = math_muladd_f32(vi4, vk4, vacc0p0);
      const float vi5 = *i5++;
      const float vk5 = w[11];
      vacc0p1 = math_muladd_f32(vi5, vk5, vacc0p1);
      const float vi6 = *i6++;
      const float vk6 = w[13];
      vacc0p0 = math_muladd_f32(vi6, vk6, vacc0p0);
      const float vi7 = *i7++;
      const float vk7 = w[15];
      vacc0p1 = math_muladd_f32(vi7, vk7, vacc0p1);
      const float vi8 = *i8++;
      const float vk8 = w[17];
      vacc0p0 = math_muladd_f32(vi8, vk8, vacc0p0);

      // Add up all accumulators to vacc01p0
      vacc0p0 = vacc0p0 + vacc0p1;

      *output++ = vacc0p0;
    }

    output = (float*) ((uintptr_t) output + output_increment);
  } while (--output_width != 0);
}
