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


void xnn_f32_dwconv_minmax_ukernel_25p1c__wasm(
    size_t channels,
    size_t output_width,
    const float** input,
    const float* weights,
    float* output,
    intptr_t input_stride,
    size_t output_increment,
    size_t input_offset,
    const float* zero,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(channels != 0);
  assert(output_width != 0);

  const float vmin = params->scalar.min;
  const float vmax = params->scalar.max;
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
    do {
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

      const float vi3 = *i3++;
      const float vk3 = w[4];
      vacc0p0 = math_muladd_f32(vi3, vk3, vacc0p0);

      const float vi4 = *i4++;
      const float vk4 = w[5];
      vacc0p0 = math_muladd_f32(vi4, vk4, vacc0p0);

      const float vi5 = *i5++;
      const float vk5 = w[6];
      vacc0p0 = math_muladd_f32(vi5, vk5, vacc0p0);

      const float vi6 = *i6++;
      const float vk6 = w[7];
      vacc0p0 = math_muladd_f32(vi6, vk6, vacc0p0);

      const float vi7 = *i7++;
      const float vk7 = w[8];
      vacc0p0 = math_muladd_f32(vi7, vk7, vacc0p0);

      const float vi8 = *i8++;
      const float vk8 = w[9];
      vacc0p0 = math_muladd_f32(vi8, vk8, vacc0p0);

      const float vi9 = *i9++;
      const float vk9 = w[10];
      vacc0p0 = math_muladd_f32(vi9, vk9, vacc0p0);

      const float vi10 = *i10++;
      const float vk10 = w[11];
      vacc0p0 = math_muladd_f32(vi10, vk10, vacc0p0);

      const float vi11 = *i11++;
      const float vk11 = w[12];
      vacc0p0 = math_muladd_f32(vi11, vk11, vacc0p0);

      const float vi12 = *i12++;
      const float vk12 = w[13];
      vacc0p0 = math_muladd_f32(vi12, vk12, vacc0p0);

      const float vi13 = *i13++;
      const float vk13 = w[14];
      vacc0p0 = math_muladd_f32(vi13, vk13, vacc0p0);

      const float vi14 = *i14++;
      const float vk14 = w[15];
      vacc0p0 = math_muladd_f32(vi14, vk14, vacc0p0);

      const float vi15 = *i15++;
      const float vk15 = w[16];
      vacc0p0 = math_muladd_f32(vi15, vk15, vacc0p0);

      const float vi16 = *i16++;
      const float vk16 = w[17];
      vacc0p0 = math_muladd_f32(vi16, vk16, vacc0p0);

      const float vi17 = *i17++;
      const float vk17 = w[18];
      vacc0p0 = math_muladd_f32(vi17, vk17, vacc0p0);

      const float vi18 = *i18++;
      const float vk18 = w[19];
      vacc0p0 = math_muladd_f32(vi18, vk18, vacc0p0);

      const float vi19 = *i19++;
      const float vk19 = w[20];
      vacc0p0 = math_muladd_f32(vi19, vk19, vacc0p0);

      const float vi20 = *i20++;
      const float vk20 = w[21];
      vacc0p0 = math_muladd_f32(vi20, vk20, vacc0p0);

      const float vi21 = *i21++;
      const float vk21 = w[22];
      vacc0p0 = math_muladd_f32(vi21, vk21, vacc0p0);

      const float vi22 = *i22++;
      const float vk22 = w[23];
      vacc0p0 = math_muladd_f32(vi22, vk22, vacc0p0);

      const float vi23 = *i23++;
      const float vk23 = w[24];
      vacc0p0 = math_muladd_f32(vi23, vk23, vacc0p0);

      const float vi24 = *i24++;
      const float vk24 = w[25];
      vacc0p0 = math_muladd_f32(vi24, vk24, vacc0p0);

      w += 26;


      float vacc0 = __builtin_wasm_max_f32(vacc0p0, vmin);
      vacc0 = __builtin_wasm_min_f32(vacc0, vmax);
      *output++ = vacc0;
    } while (--c != 0);

    output = (float*) ((uintptr_t) output + output_increment);
  } while (--output_width != 0);
}
