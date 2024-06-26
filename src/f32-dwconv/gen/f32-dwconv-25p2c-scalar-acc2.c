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


void xnn_f32_dwconv_ukernel_25p2c__scalar_acc2(
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
    const float* i9 = input[9];
    assert(i9 != NULL);
    if XNN_UNPREDICTABLE(i9 != zero) {
      i9 = (const float*) ((uintptr_t) i9 + input_offset);
    }
    const float* i10 = input[10];
    assert(i10 != NULL);
    if XNN_UNPREDICTABLE(i10 != zero) {
      i10 = (const float*) ((uintptr_t) i10 + input_offset);
    }
    const float* i11 = input[11];
    assert(i11 != NULL);
    if XNN_UNPREDICTABLE(i11 != zero) {
      i11 = (const float*) ((uintptr_t) i11 + input_offset);
    }
    const float* i12 = input[12];
    assert(i12 != NULL);
    if XNN_UNPREDICTABLE(i12 != zero) {
      i12 = (const float*) ((uintptr_t) i12 + input_offset);
    }
    const float* i13 = input[13];
    assert(i13 != NULL);
    if XNN_UNPREDICTABLE(i13 != zero) {
      i13 = (const float*) ((uintptr_t) i13 + input_offset);
    }
    const float* i14 = input[14];
    assert(i14 != NULL);
    if XNN_UNPREDICTABLE(i14 != zero) {
      i14 = (const float*) ((uintptr_t) i14 + input_offset);
    }
    const float* i15 = input[15];
    assert(i15 != NULL);
    if XNN_UNPREDICTABLE(i15 != zero) {
      i15 = (const float*) ((uintptr_t) i15 + input_offset);
    }
    const float* i16 = input[16];
    assert(i16 != NULL);
    if XNN_UNPREDICTABLE(i16 != zero) {
      i16 = (const float*) ((uintptr_t) i16 + input_offset);
    }
    const float* i17 = input[17];
    assert(i17 != NULL);
    if XNN_UNPREDICTABLE(i17 != zero) {
      i17 = (const float*) ((uintptr_t) i17 + input_offset);
    }
    const float* i18 = input[18];
    assert(i18 != NULL);
    if XNN_UNPREDICTABLE(i18 != zero) {
      i18 = (const float*) ((uintptr_t) i18 + input_offset);
    }
    const float* i19 = input[19];
    assert(i19 != NULL);
    if XNN_UNPREDICTABLE(i19 != zero) {
      i19 = (const float*) ((uintptr_t) i19 + input_offset);
    }
    const float* i20 = input[20];
    assert(i20 != NULL);
    if XNN_UNPREDICTABLE(i20 != zero) {
      i20 = (const float*) ((uintptr_t) i20 + input_offset);
    }
    const float* i21 = input[21];
    assert(i21 != NULL);
    if XNN_UNPREDICTABLE(i21 != zero) {
      i21 = (const float*) ((uintptr_t) i21 + input_offset);
    }
    const float* i22 = input[22];
    assert(i22 != NULL);
    if XNN_UNPREDICTABLE(i22 != zero) {
      i22 = (const float*) ((uintptr_t) i22 + input_offset);
    }
    const float* i23 = input[23];
    assert(i23 != NULL);
    if XNN_UNPREDICTABLE(i23 != zero) {
      i23 = (const float*) ((uintptr_t) i23 + input_offset);
    }
    const float* i24 = input[24];
    assert(i24 != NULL);
    if XNN_UNPREDICTABLE(i24 != zero) {
      i24 = (const float*) ((uintptr_t) i24 + input_offset);
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

      const float vi9x0 = i9[0];
      const float vi9x1 = i9[1];
      i9 += 2;

      const float vk9x0 = w[20];
      vacc0p1 = math_muladd_f32(vi9x0, vk9x0, vacc0p1);
      const float vk9x1 = w[21];
      vacc1p1 = math_muladd_f32(vi9x1, vk9x1, vacc1p1);

      const float vi10x0 = i10[0];
      const float vi10x1 = i10[1];
      i10 += 2;

      const float vk10x0 = w[22];
      vacc0p0 = math_muladd_f32(vi10x0, vk10x0, vacc0p0);
      const float vk10x1 = w[23];
      vacc1p0 = math_muladd_f32(vi10x1, vk10x1, vacc1p0);

      const float vi11x0 = i11[0];
      const float vi11x1 = i11[1];
      i11 += 2;

      const float vk11x0 = w[24];
      vacc0p1 = math_muladd_f32(vi11x0, vk11x0, vacc0p1);
      const float vk11x1 = w[25];
      vacc1p1 = math_muladd_f32(vi11x1, vk11x1, vacc1p1);

      const float vi12x0 = i12[0];
      const float vi12x1 = i12[1];
      i12 += 2;

      const float vk12x0 = w[26];
      vacc0p0 = math_muladd_f32(vi12x0, vk12x0, vacc0p0);
      const float vk12x1 = w[27];
      vacc1p0 = math_muladd_f32(vi12x1, vk12x1, vacc1p0);

      const float vi13x0 = i13[0];
      const float vi13x1 = i13[1];
      i13 += 2;

      const float vk13x0 = w[28];
      vacc0p1 = math_muladd_f32(vi13x0, vk13x0, vacc0p1);
      const float vk13x1 = w[29];
      vacc1p1 = math_muladd_f32(vi13x1, vk13x1, vacc1p1);

      const float vi14x0 = i14[0];
      const float vi14x1 = i14[1];
      i14 += 2;

      const float vk14x0 = w[30];
      vacc0p0 = math_muladd_f32(vi14x0, vk14x0, vacc0p0);
      const float vk14x1 = w[31];
      vacc1p0 = math_muladd_f32(vi14x1, vk14x1, vacc1p0);

      const float vi15x0 = i15[0];
      const float vi15x1 = i15[1];
      i15 += 2;

      const float vk15x0 = w[32];
      vacc0p1 = math_muladd_f32(vi15x0, vk15x0, vacc0p1);
      const float vk15x1 = w[33];
      vacc1p1 = math_muladd_f32(vi15x1, vk15x1, vacc1p1);

      const float vi16x0 = i16[0];
      const float vi16x1 = i16[1];
      i16 += 2;

      const float vk16x0 = w[34];
      vacc0p0 = math_muladd_f32(vi16x0, vk16x0, vacc0p0);
      const float vk16x1 = w[35];
      vacc1p0 = math_muladd_f32(vi16x1, vk16x1, vacc1p0);

      const float vi17x0 = i17[0];
      const float vi17x1 = i17[1];
      i17 += 2;

      const float vk17x0 = w[36];
      vacc0p1 = math_muladd_f32(vi17x0, vk17x0, vacc0p1);
      const float vk17x1 = w[37];
      vacc1p1 = math_muladd_f32(vi17x1, vk17x1, vacc1p1);

      const float vi18x0 = i18[0];
      const float vi18x1 = i18[1];
      i18 += 2;

      const float vk18x0 = w[38];
      vacc0p0 = math_muladd_f32(vi18x0, vk18x0, vacc0p0);
      const float vk18x1 = w[39];
      vacc1p0 = math_muladd_f32(vi18x1, vk18x1, vacc1p0);

      const float vi19x0 = i19[0];
      const float vi19x1 = i19[1];
      i19 += 2;

      const float vk19x0 = w[40];
      vacc0p1 = math_muladd_f32(vi19x0, vk19x0, vacc0p1);
      const float vk19x1 = w[41];
      vacc1p1 = math_muladd_f32(vi19x1, vk19x1, vacc1p1);

      const float vi20x0 = i20[0];
      const float vi20x1 = i20[1];
      i20 += 2;

      const float vk20x0 = w[42];
      vacc0p0 = math_muladd_f32(vi20x0, vk20x0, vacc0p0);
      const float vk20x1 = w[43];
      vacc1p0 = math_muladd_f32(vi20x1, vk20x1, vacc1p0);

      const float vi21x0 = i21[0];
      const float vi21x1 = i21[1];
      i21 += 2;

      const float vk21x0 = w[44];
      vacc0p1 = math_muladd_f32(vi21x0, vk21x0, vacc0p1);
      const float vk21x1 = w[45];
      vacc1p1 = math_muladd_f32(vi21x1, vk21x1, vacc1p1);

      const float vi22x0 = i22[0];
      const float vi22x1 = i22[1];
      i22 += 2;

      const float vk22x0 = w[46];
      vacc0p0 = math_muladd_f32(vi22x0, vk22x0, vacc0p0);
      const float vk22x1 = w[47];
      vacc1p0 = math_muladd_f32(vi22x1, vk22x1, vacc1p0);

      const float vi23x0 = i23[0];
      const float vi23x1 = i23[1];
      i23 += 2;

      const float vk23x0 = w[48];
      vacc0p1 = math_muladd_f32(vi23x0, vk23x0, vacc0p1);
      const float vk23x1 = w[49];
      vacc1p1 = math_muladd_f32(vi23x1, vk23x1, vacc1p1);

      const float vi24x0 = i24[0];
      const float vi24x1 = i24[1];
      i24 += 2;

      const float vk24x0 = w[50];
      vacc0p0 = math_muladd_f32(vi24x0, vk24x0, vacc0p0);
      const float vk24x1 = w[51];
      vacc1p0 = math_muladd_f32(vi24x1, vk24x1, vacc1p0);

      w += 52;

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
      const float vi9 = *i9++;
      const float vk9 = w[19];
      vacc0p1 = math_muladd_f32(vi9, vk9, vacc0p1);
      const float vi10 = *i10++;
      const float vk10 = w[21];
      vacc0p0 = math_muladd_f32(vi10, vk10, vacc0p0);
      const float vi11 = *i11++;
      const float vk11 = w[23];
      vacc0p1 = math_muladd_f32(vi11, vk11, vacc0p1);
      const float vi12 = *i12++;
      const float vk12 = w[25];
      vacc0p0 = math_muladd_f32(vi12, vk12, vacc0p0);
      const float vi13 = *i13++;
      const float vk13 = w[27];
      vacc0p1 = math_muladd_f32(vi13, vk13, vacc0p1);
      const float vi14 = *i14++;
      const float vk14 = w[29];
      vacc0p0 = math_muladd_f32(vi14, vk14, vacc0p0);
      const float vi15 = *i15++;
      const float vk15 = w[31];
      vacc0p1 = math_muladd_f32(vi15, vk15, vacc0p1);
      const float vi16 = *i16++;
      const float vk16 = w[33];
      vacc0p0 = math_muladd_f32(vi16, vk16, vacc0p0);
      const float vi17 = *i17++;
      const float vk17 = w[35];
      vacc0p1 = math_muladd_f32(vi17, vk17, vacc0p1);
      const float vi18 = *i18++;
      const float vk18 = w[37];
      vacc0p0 = math_muladd_f32(vi18, vk18, vacc0p0);
      const float vi19 = *i19++;
      const float vk19 = w[39];
      vacc0p1 = math_muladd_f32(vi19, vk19, vacc0p1);
      const float vi20 = *i20++;
      const float vk20 = w[41];
      vacc0p0 = math_muladd_f32(vi20, vk20, vacc0p0);
      const float vi21 = *i21++;
      const float vk21 = w[43];
      vacc0p1 = math_muladd_f32(vi21, vk21, vacc0p1);
      const float vi22 = *i22++;
      const float vk22 = w[45];
      vacc0p0 = math_muladd_f32(vi22, vk22, vacc0p0);
      const float vi23 = *i23++;
      const float vk23 = w[47];
      vacc0p1 = math_muladd_f32(vi23, vk23, vacc0p1);
      const float vi24 = *i24++;
      const float vk24 = w[49];
      vacc0p0 = math_muladd_f32(vi24, vk24, vacc0p0);

      // Add up all accumulators to vacc01p0
      vacc0p0 = vacc0p0 + vacc0p1;

      *output++ = vacc0p0;
    }

    output = (float*) ((uintptr_t) output + output_increment);
  } while (--output_width != 0);
}
