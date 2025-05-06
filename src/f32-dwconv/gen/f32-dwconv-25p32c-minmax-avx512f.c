// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/f32-dwconv/simd.c.in
//   Generator: tools/xngen
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include "src/xnnpack/simd/f32-avx512f.h"

#include "src/xnnpack/dwconv.h"


void xnn_f32_dwconv_minmax_ukernel_25p32c__avx512f(
    size_t channels,
    size_t output_width,
    const float** input,
    const float* weights,
    float* output,
    intptr_t input_stride,
    size_t output_increment,
    size_t input_offset,
    size_t input_pixel_stride,
    const float* zero,
    const struct xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(channels != 0);
  assert(output_width != 0);

  const xnn_simd_f32_t vmin = xnn_set1_f32(params->scalar.min);
  const xnn_simd_f32_t vmax = xnn_set1_f32(params->scalar.max);
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
    for (; c >= 32; c -= 32) {
      xnn_simd_f32_t vacc0p0 = xnn_load_f32(w + 0);
      xnn_simd_f32_t vacc16p0 = xnn_load_f32(w + 16);


      const xnn_simd_f32_t vi0x0 = xnn_loadu_f32(i0 + 0);
      const xnn_simd_f32_t vi0x16 = xnn_loadu_f32(i0 + 16);
      i0 += 32;

      const xnn_simd_f32_t vk0x0 = xnn_load_f32(w + 32);
      const xnn_simd_f32_t vk0x16 = xnn_load_f32(w + 48);
      vacc0p0 = xnn_fmadd_f32(vi0x0, vk0x0, vacc0p0);
      vacc16p0 = xnn_fmadd_f32(vi0x16, vk0x16, vacc16p0);

      const xnn_simd_f32_t vi1x0 = xnn_loadu_f32(i1 + 0);
      const xnn_simd_f32_t vi1x16 = xnn_loadu_f32(i1 + 16);
      i1 += 32;

      const xnn_simd_f32_t vk1x0 = xnn_load_f32(w + 64);
      const xnn_simd_f32_t vk1x16 = xnn_load_f32(w + 80);
      vacc0p0 = xnn_fmadd_f32(vi1x0, vk1x0, vacc0p0);
      vacc16p0 = xnn_fmadd_f32(vi1x16, vk1x16, vacc16p0);

      const xnn_simd_f32_t vi2x0 = xnn_loadu_f32(i2 + 0);
      const xnn_simd_f32_t vi2x16 = xnn_loadu_f32(i2 + 16);
      i2 += 32;

      const xnn_simd_f32_t vk2x0 = xnn_load_f32(w + 96);
      const xnn_simd_f32_t vk2x16 = xnn_load_f32(w + 112);
      vacc0p0 = xnn_fmadd_f32(vi2x0, vk2x0, vacc0p0);
      vacc16p0 = xnn_fmadd_f32(vi2x16, vk2x16, vacc16p0);

      const xnn_simd_f32_t vi3x0 = xnn_loadu_f32(i3 + 0);
      const xnn_simd_f32_t vi3x16 = xnn_loadu_f32(i3 + 16);
      i3 += 32;

      const xnn_simd_f32_t vk3x0 = xnn_load_f32(w + 128);
      const xnn_simd_f32_t vk3x16 = xnn_load_f32(w + 144);
      vacc0p0 = xnn_fmadd_f32(vi3x0, vk3x0, vacc0p0);
      vacc16p0 = xnn_fmadd_f32(vi3x16, vk3x16, vacc16p0);

      const xnn_simd_f32_t vi4x0 = xnn_loadu_f32(i4 + 0);
      const xnn_simd_f32_t vi4x16 = xnn_loadu_f32(i4 + 16);
      i4 += 32;

      const xnn_simd_f32_t vk4x0 = xnn_load_f32(w + 160);
      const xnn_simd_f32_t vk4x16 = xnn_load_f32(w + 176);
      vacc0p0 = xnn_fmadd_f32(vi4x0, vk4x0, vacc0p0);
      vacc16p0 = xnn_fmadd_f32(vi4x16, vk4x16, vacc16p0);

      const xnn_simd_f32_t vi5x0 = xnn_loadu_f32(i5 + 0);
      const xnn_simd_f32_t vi5x16 = xnn_loadu_f32(i5 + 16);
      i5 += 32;

      const xnn_simd_f32_t vk5x0 = xnn_load_f32(w + 192);
      const xnn_simd_f32_t vk5x16 = xnn_load_f32(w + 208);
      vacc0p0 = xnn_fmadd_f32(vi5x0, vk5x0, vacc0p0);
      vacc16p0 = xnn_fmadd_f32(vi5x16, vk5x16, vacc16p0);

      const xnn_simd_f32_t vi6x0 = xnn_loadu_f32(i6 + 0);
      const xnn_simd_f32_t vi6x16 = xnn_loadu_f32(i6 + 16);
      i6 += 32;

      const xnn_simd_f32_t vk6x0 = xnn_load_f32(w + 224);
      const xnn_simd_f32_t vk6x16 = xnn_load_f32(w + 240);
      vacc0p0 = xnn_fmadd_f32(vi6x0, vk6x0, vacc0p0);
      vacc16p0 = xnn_fmadd_f32(vi6x16, vk6x16, vacc16p0);

      const xnn_simd_f32_t vi7x0 = xnn_loadu_f32(i7 + 0);
      const xnn_simd_f32_t vi7x16 = xnn_loadu_f32(i7 + 16);
      i7 += 32;

      const xnn_simd_f32_t vk7x0 = xnn_load_f32(w + 256);
      const xnn_simd_f32_t vk7x16 = xnn_load_f32(w + 272);
      vacc0p0 = xnn_fmadd_f32(vi7x0, vk7x0, vacc0p0);
      vacc16p0 = xnn_fmadd_f32(vi7x16, vk7x16, vacc16p0);

      const xnn_simd_f32_t vi8x0 = xnn_loadu_f32(i8 + 0);
      const xnn_simd_f32_t vi8x16 = xnn_loadu_f32(i8 + 16);
      i8 += 32;

      const xnn_simd_f32_t vk8x0 = xnn_load_f32(w + 288);
      const xnn_simd_f32_t vk8x16 = xnn_load_f32(w + 304);
      vacc0p0 = xnn_fmadd_f32(vi8x0, vk8x0, vacc0p0);
      vacc16p0 = xnn_fmadd_f32(vi8x16, vk8x16, vacc16p0);

      const xnn_simd_f32_t vi9x0 = xnn_loadu_f32(i9 + 0);
      const xnn_simd_f32_t vi9x16 = xnn_loadu_f32(i9 + 16);
      i9 += 32;

      const xnn_simd_f32_t vk9x0 = xnn_load_f32(w + 320);
      const xnn_simd_f32_t vk9x16 = xnn_load_f32(w + 336);
      vacc0p0 = xnn_fmadd_f32(vi9x0, vk9x0, vacc0p0);
      vacc16p0 = xnn_fmadd_f32(vi9x16, vk9x16, vacc16p0);

      const xnn_simd_f32_t vi10x0 = xnn_loadu_f32(i10 + 0);
      const xnn_simd_f32_t vi10x16 = xnn_loadu_f32(i10 + 16);
      i10 += 32;

      const xnn_simd_f32_t vk10x0 = xnn_load_f32(w + 352);
      const xnn_simd_f32_t vk10x16 = xnn_load_f32(w + 368);
      vacc0p0 = xnn_fmadd_f32(vi10x0, vk10x0, vacc0p0);
      vacc16p0 = xnn_fmadd_f32(vi10x16, vk10x16, vacc16p0);

      const xnn_simd_f32_t vi11x0 = xnn_loadu_f32(i11 + 0);
      const xnn_simd_f32_t vi11x16 = xnn_loadu_f32(i11 + 16);
      i11 += 32;

      const xnn_simd_f32_t vk11x0 = xnn_load_f32(w + 384);
      const xnn_simd_f32_t vk11x16 = xnn_load_f32(w + 400);
      vacc0p0 = xnn_fmadd_f32(vi11x0, vk11x0, vacc0p0);
      vacc16p0 = xnn_fmadd_f32(vi11x16, vk11x16, vacc16p0);

      const xnn_simd_f32_t vi12x0 = xnn_loadu_f32(i12 + 0);
      const xnn_simd_f32_t vi12x16 = xnn_loadu_f32(i12 + 16);
      i12 += 32;

      const xnn_simd_f32_t vk12x0 = xnn_load_f32(w + 416);
      const xnn_simd_f32_t vk12x16 = xnn_load_f32(w + 432);
      vacc0p0 = xnn_fmadd_f32(vi12x0, vk12x0, vacc0p0);
      vacc16p0 = xnn_fmadd_f32(vi12x16, vk12x16, vacc16p0);

      const xnn_simd_f32_t vi13x0 = xnn_loadu_f32(i13 + 0);
      const xnn_simd_f32_t vi13x16 = xnn_loadu_f32(i13 + 16);
      i13 += 32;

      const xnn_simd_f32_t vk13x0 = xnn_load_f32(w + 448);
      const xnn_simd_f32_t vk13x16 = xnn_load_f32(w + 464);
      vacc0p0 = xnn_fmadd_f32(vi13x0, vk13x0, vacc0p0);
      vacc16p0 = xnn_fmadd_f32(vi13x16, vk13x16, vacc16p0);

      const xnn_simd_f32_t vi14x0 = xnn_loadu_f32(i14 + 0);
      const xnn_simd_f32_t vi14x16 = xnn_loadu_f32(i14 + 16);
      i14 += 32;

      const xnn_simd_f32_t vk14x0 = xnn_load_f32(w + 480);
      const xnn_simd_f32_t vk14x16 = xnn_load_f32(w + 496);
      vacc0p0 = xnn_fmadd_f32(vi14x0, vk14x0, vacc0p0);
      vacc16p0 = xnn_fmadd_f32(vi14x16, vk14x16, vacc16p0);

      const xnn_simd_f32_t vi15x0 = xnn_loadu_f32(i15 + 0);
      const xnn_simd_f32_t vi15x16 = xnn_loadu_f32(i15 + 16);
      i15 += 32;

      const xnn_simd_f32_t vk15x0 = xnn_load_f32(w + 512);
      const xnn_simd_f32_t vk15x16 = xnn_load_f32(w + 528);
      vacc0p0 = xnn_fmadd_f32(vi15x0, vk15x0, vacc0p0);
      vacc16p0 = xnn_fmadd_f32(vi15x16, vk15x16, vacc16p0);

      const xnn_simd_f32_t vi16x0 = xnn_loadu_f32(i16 + 0);
      const xnn_simd_f32_t vi16x16 = xnn_loadu_f32(i16 + 16);
      i16 += 32;

      const xnn_simd_f32_t vk16x0 = xnn_load_f32(w + 544);
      const xnn_simd_f32_t vk16x16 = xnn_load_f32(w + 560);
      vacc0p0 = xnn_fmadd_f32(vi16x0, vk16x0, vacc0p0);
      vacc16p0 = xnn_fmadd_f32(vi16x16, vk16x16, vacc16p0);

      const xnn_simd_f32_t vi17x0 = xnn_loadu_f32(i17 + 0);
      const xnn_simd_f32_t vi17x16 = xnn_loadu_f32(i17 + 16);
      i17 += 32;

      const xnn_simd_f32_t vk17x0 = xnn_load_f32(w + 576);
      const xnn_simd_f32_t vk17x16 = xnn_load_f32(w + 592);
      vacc0p0 = xnn_fmadd_f32(vi17x0, vk17x0, vacc0p0);
      vacc16p0 = xnn_fmadd_f32(vi17x16, vk17x16, vacc16p0);

      const xnn_simd_f32_t vi18x0 = xnn_loadu_f32(i18 + 0);
      const xnn_simd_f32_t vi18x16 = xnn_loadu_f32(i18 + 16);
      i18 += 32;

      const xnn_simd_f32_t vk18x0 = xnn_load_f32(w + 608);
      const xnn_simd_f32_t vk18x16 = xnn_load_f32(w + 624);
      vacc0p0 = xnn_fmadd_f32(vi18x0, vk18x0, vacc0p0);
      vacc16p0 = xnn_fmadd_f32(vi18x16, vk18x16, vacc16p0);

      const xnn_simd_f32_t vi19x0 = xnn_loadu_f32(i19 + 0);
      const xnn_simd_f32_t vi19x16 = xnn_loadu_f32(i19 + 16);
      i19 += 32;

      const xnn_simd_f32_t vk19x0 = xnn_load_f32(w + 640);
      const xnn_simd_f32_t vk19x16 = xnn_load_f32(w + 656);
      vacc0p0 = xnn_fmadd_f32(vi19x0, vk19x0, vacc0p0);
      vacc16p0 = xnn_fmadd_f32(vi19x16, vk19x16, vacc16p0);

      const xnn_simd_f32_t vi20x0 = xnn_loadu_f32(i20 + 0);
      const xnn_simd_f32_t vi20x16 = xnn_loadu_f32(i20 + 16);
      i20 += 32;

      const xnn_simd_f32_t vk20x0 = xnn_load_f32(w + 672);
      const xnn_simd_f32_t vk20x16 = xnn_load_f32(w + 688);
      vacc0p0 = xnn_fmadd_f32(vi20x0, vk20x0, vacc0p0);
      vacc16p0 = xnn_fmadd_f32(vi20x16, vk20x16, vacc16p0);

      const xnn_simd_f32_t vi21x0 = xnn_loadu_f32(i21 + 0);
      const xnn_simd_f32_t vi21x16 = xnn_loadu_f32(i21 + 16);
      i21 += 32;

      const xnn_simd_f32_t vk21x0 = xnn_load_f32(w + 704);
      const xnn_simd_f32_t vk21x16 = xnn_load_f32(w + 720);
      vacc0p0 = xnn_fmadd_f32(vi21x0, vk21x0, vacc0p0);
      vacc16p0 = xnn_fmadd_f32(vi21x16, vk21x16, vacc16p0);

      const xnn_simd_f32_t vi22x0 = xnn_loadu_f32(i22 + 0);
      const xnn_simd_f32_t vi22x16 = xnn_loadu_f32(i22 + 16);
      i22 += 32;

      const xnn_simd_f32_t vk22x0 = xnn_load_f32(w + 736);
      const xnn_simd_f32_t vk22x16 = xnn_load_f32(w + 752);
      vacc0p0 = xnn_fmadd_f32(vi22x0, vk22x0, vacc0p0);
      vacc16p0 = xnn_fmadd_f32(vi22x16, vk22x16, vacc16p0);

      const xnn_simd_f32_t vi23x0 = xnn_loadu_f32(i23 + 0);
      const xnn_simd_f32_t vi23x16 = xnn_loadu_f32(i23 + 16);
      i23 += 32;

      const xnn_simd_f32_t vk23x0 = xnn_load_f32(w + 768);
      const xnn_simd_f32_t vk23x16 = xnn_load_f32(w + 784);
      vacc0p0 = xnn_fmadd_f32(vi23x0, vk23x0, vacc0p0);
      vacc16p0 = xnn_fmadd_f32(vi23x16, vk23x16, vacc16p0);

      const xnn_simd_f32_t vi24x0 = xnn_loadu_f32(i24 + 0);
      const xnn_simd_f32_t vi24x16 = xnn_loadu_f32(i24 + 16);
      i24 += 32;

      const xnn_simd_f32_t vk24x0 = xnn_load_f32(w + 800);
      const xnn_simd_f32_t vk24x16 = xnn_load_f32(w + 816);
      vacc0p0 = xnn_fmadd_f32(vi24x0, vk24x0, vacc0p0);
      vacc16p0 = xnn_fmadd_f32(vi24x16, vk24x16, vacc16p0);

      w += 832;


      xnn_simd_f32_t vacc0 = xnn_max_f32(vmin, vacc0p0);
      xnn_simd_f32_t vacc16 = xnn_max_f32(vmin, vacc16p0);
      vacc0 = xnn_min_f32(vmax, vacc0);
      vacc16 = xnn_min_f32(vmax, vacc16);

      xnn_storeu_f32(output + 0, vacc0);
      xnn_storeu_f32(output + 16, vacc16);
      output += 32;
    }
    for (; c >= 16; c -= 16) {
      xnn_simd_f32_t vacc0p0 = xnn_load_f32(w);

      const xnn_simd_f32_t vi0x0 = xnn_loadu_f32(i0);
      i0 += 16;

      const xnn_simd_f32_t vk0x0 = xnn_load_f32(w + 32);
      vacc0p0 = xnn_fmadd_f32(vi0x0, vk0x0, vacc0p0);

      const xnn_simd_f32_t vi1x0 = xnn_loadu_f32(i1);
      i1 += 16;

      const xnn_simd_f32_t vk1x0 = xnn_load_f32(w + 64);
      vacc0p0 = xnn_fmadd_f32(vi1x0, vk1x0, vacc0p0);

      const xnn_simd_f32_t vi2x0 = xnn_loadu_f32(i2);
      i2 += 16;

      const xnn_simd_f32_t vk2x0 = xnn_load_f32(w + 96);
      vacc0p0 = xnn_fmadd_f32(vi2x0, vk2x0, vacc0p0);

      const xnn_simd_f32_t vi3x0 = xnn_loadu_f32(i3);
      i3 += 16;

      const xnn_simd_f32_t vk3x0 = xnn_load_f32(w + 128);
      vacc0p0 = xnn_fmadd_f32(vi3x0, vk3x0, vacc0p0);

      const xnn_simd_f32_t vi4x0 = xnn_loadu_f32(i4);
      i4 += 16;

      const xnn_simd_f32_t vk4x0 = xnn_load_f32(w + 160);
      vacc0p0 = xnn_fmadd_f32(vi4x0, vk4x0, vacc0p0);

      const xnn_simd_f32_t vi5x0 = xnn_loadu_f32(i5);
      i5 += 16;

      const xnn_simd_f32_t vk5x0 = xnn_load_f32(w + 192);
      vacc0p0 = xnn_fmadd_f32(vi5x0, vk5x0, vacc0p0);

      const xnn_simd_f32_t vi6x0 = xnn_loadu_f32(i6);
      i6 += 16;

      const xnn_simd_f32_t vk6x0 = xnn_load_f32(w + 224);
      vacc0p0 = xnn_fmadd_f32(vi6x0, vk6x0, vacc0p0);

      const xnn_simd_f32_t vi7x0 = xnn_loadu_f32(i7);
      i7 += 16;

      const xnn_simd_f32_t vk7x0 = xnn_load_f32(w + 256);
      vacc0p0 = xnn_fmadd_f32(vi7x0, vk7x0, vacc0p0);

      const xnn_simd_f32_t vi8x0 = xnn_loadu_f32(i8);
      i8 += 16;

      const xnn_simd_f32_t vk8x0 = xnn_load_f32(w + 288);
      vacc0p0 = xnn_fmadd_f32(vi8x0, vk8x0, vacc0p0);

      const xnn_simd_f32_t vi9x0 = xnn_loadu_f32(i9);
      i9 += 16;

      const xnn_simd_f32_t vk9x0 = xnn_load_f32(w + 320);
      vacc0p0 = xnn_fmadd_f32(vi9x0, vk9x0, vacc0p0);

      const xnn_simd_f32_t vi10x0 = xnn_loadu_f32(i10);
      i10 += 16;

      const xnn_simd_f32_t vk10x0 = xnn_load_f32(w + 352);
      vacc0p0 = xnn_fmadd_f32(vi10x0, vk10x0, vacc0p0);

      const xnn_simd_f32_t vi11x0 = xnn_loadu_f32(i11);
      i11 += 16;

      const xnn_simd_f32_t vk11x0 = xnn_load_f32(w + 384);
      vacc0p0 = xnn_fmadd_f32(vi11x0, vk11x0, vacc0p0);

      const xnn_simd_f32_t vi12x0 = xnn_loadu_f32(i12);
      i12 += 16;

      const xnn_simd_f32_t vk12x0 = xnn_load_f32(w + 416);
      vacc0p0 = xnn_fmadd_f32(vi12x0, vk12x0, vacc0p0);

      const xnn_simd_f32_t vi13x0 = xnn_loadu_f32(i13);
      i13 += 16;

      const xnn_simd_f32_t vk13x0 = xnn_load_f32(w + 448);
      vacc0p0 = xnn_fmadd_f32(vi13x0, vk13x0, vacc0p0);

      const xnn_simd_f32_t vi14x0 = xnn_loadu_f32(i14);
      i14 += 16;

      const xnn_simd_f32_t vk14x0 = xnn_load_f32(w + 480);
      vacc0p0 = xnn_fmadd_f32(vi14x0, vk14x0, vacc0p0);

      const xnn_simd_f32_t vi15x0 = xnn_loadu_f32(i15);
      i15 += 16;

      const xnn_simd_f32_t vk15x0 = xnn_load_f32(w + 512);
      vacc0p0 = xnn_fmadd_f32(vi15x0, vk15x0, vacc0p0);

      const xnn_simd_f32_t vi16x0 = xnn_loadu_f32(i16);
      i16 += 16;

      const xnn_simd_f32_t vk16x0 = xnn_load_f32(w + 544);
      vacc0p0 = xnn_fmadd_f32(vi16x0, vk16x0, vacc0p0);

      const xnn_simd_f32_t vi17x0 = xnn_loadu_f32(i17);
      i17 += 16;

      const xnn_simd_f32_t vk17x0 = xnn_load_f32(w + 576);
      vacc0p0 = xnn_fmadd_f32(vi17x0, vk17x0, vacc0p0);

      const xnn_simd_f32_t vi18x0 = xnn_loadu_f32(i18);
      i18 += 16;

      const xnn_simd_f32_t vk18x0 = xnn_load_f32(w + 608);
      vacc0p0 = xnn_fmadd_f32(vi18x0, vk18x0, vacc0p0);

      const xnn_simd_f32_t vi19x0 = xnn_loadu_f32(i19);
      i19 += 16;

      const xnn_simd_f32_t vk19x0 = xnn_load_f32(w + 640);
      vacc0p0 = xnn_fmadd_f32(vi19x0, vk19x0, vacc0p0);

      const xnn_simd_f32_t vi20x0 = xnn_loadu_f32(i20);
      i20 += 16;

      const xnn_simd_f32_t vk20x0 = xnn_load_f32(w + 672);
      vacc0p0 = xnn_fmadd_f32(vi20x0, vk20x0, vacc0p0);

      const xnn_simd_f32_t vi21x0 = xnn_loadu_f32(i21);
      i21 += 16;

      const xnn_simd_f32_t vk21x0 = xnn_load_f32(w + 704);
      vacc0p0 = xnn_fmadd_f32(vi21x0, vk21x0, vacc0p0);

      const xnn_simd_f32_t vi22x0 = xnn_loadu_f32(i22);
      i22 += 16;

      const xnn_simd_f32_t vk22x0 = xnn_load_f32(w + 736);
      vacc0p0 = xnn_fmadd_f32(vi22x0, vk22x0, vacc0p0);

      const xnn_simd_f32_t vi23x0 = xnn_loadu_f32(i23);
      i23 += 16;

      const xnn_simd_f32_t vk23x0 = xnn_load_f32(w + 768);
      vacc0p0 = xnn_fmadd_f32(vi23x0, vk23x0, vacc0p0);

      const xnn_simd_f32_t vi24x0 = xnn_loadu_f32(i24);
      i24 += 16;

      const xnn_simd_f32_t vk24x0 = xnn_load_f32(w + 800);
      vacc0p0 = xnn_fmadd_f32(vi24x0, vk24x0, vacc0p0);

      w += 16;


      xnn_simd_f32_t vacc0 = xnn_max_f32(vmin, vacc0p0);
      vacc0 = xnn_min_f32(vmax, vacc0);

      xnn_storeu_f32(output, vacc0);
      output += 16;
    }
    if XNN_UNLIKELY(c != 0) {
      xnn_simd_f32_t vacc0p0 = xnn_load_tail_f32(w, c);

      const xnn_simd_f32_t vi0x0 = xnn_load_tail_f32(i0, c);
      const xnn_simd_f32_t vk0x0 = xnn_load_tail_f32(w + 32, c);
      vacc0p0 = xnn_fmadd_f32(vi0x0, vk0x0, vacc0p0);

      const xnn_simd_f32_t vi1x0 = xnn_load_tail_f32(i1, c);
      const xnn_simd_f32_t vk1x0 = xnn_load_tail_f32(w + 64, c);
      vacc0p0 = xnn_fmadd_f32(vi1x0, vk1x0, vacc0p0);

      const xnn_simd_f32_t vi2x0 = xnn_load_tail_f32(i2, c);
      const xnn_simd_f32_t vk2x0 = xnn_load_tail_f32(w + 96, c);
      vacc0p0 = xnn_fmadd_f32(vi2x0, vk2x0, vacc0p0);

      const xnn_simd_f32_t vi3x0 = xnn_load_tail_f32(i3, c);
      const xnn_simd_f32_t vk3x0 = xnn_load_tail_f32(w + 128, c);
      vacc0p0 = xnn_fmadd_f32(vi3x0, vk3x0, vacc0p0);

      const xnn_simd_f32_t vi4x0 = xnn_load_tail_f32(i4, c);
      const xnn_simd_f32_t vk4x0 = xnn_load_tail_f32(w + 160, c);
      vacc0p0 = xnn_fmadd_f32(vi4x0, vk4x0, vacc0p0);

      const xnn_simd_f32_t vi5x0 = xnn_load_tail_f32(i5, c);
      const xnn_simd_f32_t vk5x0 = xnn_load_tail_f32(w + 192, c);
      vacc0p0 = xnn_fmadd_f32(vi5x0, vk5x0, vacc0p0);

      const xnn_simd_f32_t vi6x0 = xnn_load_tail_f32(i6, c);
      const xnn_simd_f32_t vk6x0 = xnn_load_tail_f32(w + 224, c);
      vacc0p0 = xnn_fmadd_f32(vi6x0, vk6x0, vacc0p0);

      const xnn_simd_f32_t vi7x0 = xnn_load_tail_f32(i7, c);
      const xnn_simd_f32_t vk7x0 = xnn_load_tail_f32(w + 256, c);
      vacc0p0 = xnn_fmadd_f32(vi7x0, vk7x0, vacc0p0);

      const xnn_simd_f32_t vi8x0 = xnn_load_tail_f32(i8, c);
      const xnn_simd_f32_t vk8x0 = xnn_load_tail_f32(w + 288, c);
      vacc0p0 = xnn_fmadd_f32(vi8x0, vk8x0, vacc0p0);

      const xnn_simd_f32_t vi9x0 = xnn_load_tail_f32(i9, c);
      const xnn_simd_f32_t vk9x0 = xnn_load_tail_f32(w + 320, c);
      vacc0p0 = xnn_fmadd_f32(vi9x0, vk9x0, vacc0p0);

      const xnn_simd_f32_t vi10x0 = xnn_load_tail_f32(i10, c);
      const xnn_simd_f32_t vk10x0 = xnn_load_tail_f32(w + 352, c);
      vacc0p0 = xnn_fmadd_f32(vi10x0, vk10x0, vacc0p0);

      const xnn_simd_f32_t vi11x0 = xnn_load_tail_f32(i11, c);
      const xnn_simd_f32_t vk11x0 = xnn_load_tail_f32(w + 384, c);
      vacc0p0 = xnn_fmadd_f32(vi11x0, vk11x0, vacc0p0);

      const xnn_simd_f32_t vi12x0 = xnn_load_tail_f32(i12, c);
      const xnn_simd_f32_t vk12x0 = xnn_load_tail_f32(w + 416, c);
      vacc0p0 = xnn_fmadd_f32(vi12x0, vk12x0, vacc0p0);

      const xnn_simd_f32_t vi13x0 = xnn_load_tail_f32(i13, c);
      const xnn_simd_f32_t vk13x0 = xnn_load_tail_f32(w + 448, c);
      vacc0p0 = xnn_fmadd_f32(vi13x0, vk13x0, vacc0p0);

      const xnn_simd_f32_t vi14x0 = xnn_load_tail_f32(i14, c);
      const xnn_simd_f32_t vk14x0 = xnn_load_tail_f32(w + 480, c);
      vacc0p0 = xnn_fmadd_f32(vi14x0, vk14x0, vacc0p0);

      const xnn_simd_f32_t vi15x0 = xnn_load_tail_f32(i15, c);
      const xnn_simd_f32_t vk15x0 = xnn_load_tail_f32(w + 512, c);
      vacc0p0 = xnn_fmadd_f32(vi15x0, vk15x0, vacc0p0);

      const xnn_simd_f32_t vi16x0 = xnn_load_tail_f32(i16, c);
      const xnn_simd_f32_t vk16x0 = xnn_load_tail_f32(w + 544, c);
      vacc0p0 = xnn_fmadd_f32(vi16x0, vk16x0, vacc0p0);

      const xnn_simd_f32_t vi17x0 = xnn_load_tail_f32(i17, c);
      const xnn_simd_f32_t vk17x0 = xnn_load_tail_f32(w + 576, c);
      vacc0p0 = xnn_fmadd_f32(vi17x0, vk17x0, vacc0p0);

      const xnn_simd_f32_t vi18x0 = xnn_load_tail_f32(i18, c);
      const xnn_simd_f32_t vk18x0 = xnn_load_tail_f32(w + 608, c);
      vacc0p0 = xnn_fmadd_f32(vi18x0, vk18x0, vacc0p0);

      const xnn_simd_f32_t vi19x0 = xnn_load_tail_f32(i19, c);
      const xnn_simd_f32_t vk19x0 = xnn_load_tail_f32(w + 640, c);
      vacc0p0 = xnn_fmadd_f32(vi19x0, vk19x0, vacc0p0);

      const xnn_simd_f32_t vi20x0 = xnn_load_tail_f32(i20, c);
      const xnn_simd_f32_t vk20x0 = xnn_load_tail_f32(w + 672, c);
      vacc0p0 = xnn_fmadd_f32(vi20x0, vk20x0, vacc0p0);

      const xnn_simd_f32_t vi21x0 = xnn_load_tail_f32(i21, c);
      const xnn_simd_f32_t vk21x0 = xnn_load_tail_f32(w + 704, c);
      vacc0p0 = xnn_fmadd_f32(vi21x0, vk21x0, vacc0p0);

      const xnn_simd_f32_t vi22x0 = xnn_load_tail_f32(i22, c);
      const xnn_simd_f32_t vk22x0 = xnn_load_tail_f32(w + 736, c);
      vacc0p0 = xnn_fmadd_f32(vi22x0, vk22x0, vacc0p0);

      const xnn_simd_f32_t vi23x0 = xnn_load_tail_f32(i23, c);
      const xnn_simd_f32_t vk23x0 = xnn_load_tail_f32(w + 768, c);
      vacc0p0 = xnn_fmadd_f32(vi23x0, vk23x0, vacc0p0);

      const xnn_simd_f32_t vi24x0 = xnn_load_tail_f32(i24, c);
      const xnn_simd_f32_t vk24x0 = xnn_load_tail_f32(w + 800, c);
      vacc0p0 = xnn_fmadd_f32(vi24x0, vk24x0, vacc0p0);


      xnn_simd_f32_t vacc0 = xnn_max_f32(vmin, vacc0p0);
      vacc0 = xnn_min_f32(vmax, vacc0);

      xnn_store_tail_f32(output, vacc0, c);
      output += c;
    }

    input_offset += input_pixel_stride;
    output = (float*) ((uintptr_t) output + output_increment);
  } while (--output_width != 0);
}
