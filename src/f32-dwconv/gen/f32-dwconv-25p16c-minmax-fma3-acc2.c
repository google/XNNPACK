// Auto-generated file. Do not edit!
//   Template: src/f32-dwconv/unipass-avx.c.in
//   Generator: tools/xngen
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <immintrin.h>

#include "xnnpack/dwconv.h"


void xnn_f32_dwconv_minmax_ukernel_25p16c__fma3_acc2(
    size_t channels,
    size_t output_width,
    const float** input,
    const float* weights,
    float* output,
    intptr_t input_stride,
    size_t output_increment,
    size_t input_offset,
    const float* zero,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(channels != 0);
  assert(output_width != 0);

  static const int32_t mask_table[14] = {-1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0};

  const __m256 vmin = _mm256_set1_ps(params->avx.min);
  const __m256 vmax = _mm256_set1_ps(params->avx.max);
  XNN_FORCE_REALIZATION(vmin);
  XNN_FORCE_REALIZATION(vmax);
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
    for (; c >= 16; c -= 16) {
      __m256 vacc01234567p0 = _mm256_load_ps(w);
      __m256 vacc89ABCDEFp0 = _mm256_load_ps(w + 8);


      const __m256 vi0x01234567 = _mm256_loadu_ps(i0);
      const __m256 vi0x89ABCDEF = _mm256_loadu_ps(i0 + 8);
      i0 += 16;

      const __m256 vk0x01234567 = _mm256_load_ps(w + 16);
      const __m256 vk0x89ABCDEF = _mm256_load_ps(w + 24);
      vacc01234567p0 = _mm256_fmadd_ps(vi0x01234567, vk0x01234567, vacc01234567p0);
      vacc89ABCDEFp0 = _mm256_fmadd_ps(vi0x89ABCDEF, vk0x89ABCDEF, vacc89ABCDEFp0);

      const __m256 vi1x01234567 = _mm256_loadu_ps(i1);
      const __m256 vi1x89ABCDEF = _mm256_loadu_ps(i1 + 8);
      i1 += 16;

      const __m256 vk1x01234567 = _mm256_load_ps(w + 32);
      const __m256 vk1x89ABCDEF = _mm256_load_ps(w + 40);
      __m256 vacc01234567p1 = _mm256_mul_ps(vi1x01234567, vk1x01234567);
      __m256 vacc89ABCDEFp1 = _mm256_mul_ps(vi1x89ABCDEF, vk1x89ABCDEF);

      const __m256 vi2x01234567 = _mm256_loadu_ps(i2);
      const __m256 vi2x89ABCDEF = _mm256_loadu_ps(i2 + 8);
      i2 += 16;

      const __m256 vk2x01234567 = _mm256_load_ps(w + 48);
      const __m256 vk2x89ABCDEF = _mm256_load_ps(w + 56);
      vacc01234567p0 = _mm256_fmadd_ps(vi2x01234567, vk2x01234567, vacc01234567p0);
      vacc89ABCDEFp0 = _mm256_fmadd_ps(vi2x89ABCDEF, vk2x89ABCDEF, vacc89ABCDEFp0);

      const __m256 vi3x01234567 = _mm256_loadu_ps(i3);
      const __m256 vi3x89ABCDEF = _mm256_loadu_ps(i3 + 8);
      i3 += 16;

      const __m256 vk3x01234567 = _mm256_load_ps(w + 64);
      const __m256 vk3x89ABCDEF = _mm256_load_ps(w + 72);
      vacc01234567p1 = _mm256_fmadd_ps(vi3x01234567, vk3x01234567, vacc01234567p1);
      vacc89ABCDEFp1 = _mm256_fmadd_ps(vi3x89ABCDEF, vk3x89ABCDEF, vacc89ABCDEFp1);

      const __m256 vi4x01234567 = _mm256_loadu_ps(i4);
      const __m256 vi4x89ABCDEF = _mm256_loadu_ps(i4 + 8);
      i4 += 16;

      const __m256 vk4x01234567 = _mm256_load_ps(w + 80);
      const __m256 vk4x89ABCDEF = _mm256_load_ps(w + 88);
      vacc01234567p0 = _mm256_fmadd_ps(vi4x01234567, vk4x01234567, vacc01234567p0);
      vacc89ABCDEFp0 = _mm256_fmadd_ps(vi4x89ABCDEF, vk4x89ABCDEF, vacc89ABCDEFp0);

      const __m256 vi5x01234567 = _mm256_loadu_ps(i5);
      const __m256 vi5x89ABCDEF = _mm256_loadu_ps(i5 + 8);
      i5 += 16;

      const __m256 vk5x01234567 = _mm256_load_ps(w + 96);
      const __m256 vk5x89ABCDEF = _mm256_load_ps(w + 104);
      vacc01234567p1 = _mm256_fmadd_ps(vi5x01234567, vk5x01234567, vacc01234567p1);
      vacc89ABCDEFp1 = _mm256_fmadd_ps(vi5x89ABCDEF, vk5x89ABCDEF, vacc89ABCDEFp1);

      const __m256 vi6x01234567 = _mm256_loadu_ps(i6);
      const __m256 vi6x89ABCDEF = _mm256_loadu_ps(i6 + 8);
      i6 += 16;

      const __m256 vk6x01234567 = _mm256_load_ps(w + 112);
      const __m256 vk6x89ABCDEF = _mm256_load_ps(w + 120);
      vacc01234567p0 = _mm256_fmadd_ps(vi6x01234567, vk6x01234567, vacc01234567p0);
      vacc89ABCDEFp0 = _mm256_fmadd_ps(vi6x89ABCDEF, vk6x89ABCDEF, vacc89ABCDEFp0);

      const __m256 vi7x01234567 = _mm256_loadu_ps(i7);
      const __m256 vi7x89ABCDEF = _mm256_loadu_ps(i7 + 8);
      i7 += 16;

      const __m256 vk7x01234567 = _mm256_load_ps(w + 128);
      const __m256 vk7x89ABCDEF = _mm256_load_ps(w + 136);
      vacc01234567p1 = _mm256_fmadd_ps(vi7x01234567, vk7x01234567, vacc01234567p1);
      vacc89ABCDEFp1 = _mm256_fmadd_ps(vi7x89ABCDEF, vk7x89ABCDEF, vacc89ABCDEFp1);

      const __m256 vi8x01234567 = _mm256_loadu_ps(i8);
      const __m256 vi8x89ABCDEF = _mm256_loadu_ps(i8 + 8);
      i8 += 16;

      const __m256 vk8x01234567 = _mm256_load_ps(w + 144);
      const __m256 vk8x89ABCDEF = _mm256_load_ps(w + 152);
      vacc01234567p0 = _mm256_fmadd_ps(vi8x01234567, vk8x01234567, vacc01234567p0);
      vacc89ABCDEFp0 = _mm256_fmadd_ps(vi8x89ABCDEF, vk8x89ABCDEF, vacc89ABCDEFp0);

      const __m256 vi9x01234567 = _mm256_loadu_ps(i9);
      const __m256 vi9x89ABCDEF = _mm256_loadu_ps(i9 + 8);
      i9 += 16;

      const __m256 vk9x01234567 = _mm256_load_ps(w + 160);
      const __m256 vk9x89ABCDEF = _mm256_load_ps(w + 168);
      vacc01234567p1 = _mm256_fmadd_ps(vi9x01234567, vk9x01234567, vacc01234567p1);
      vacc89ABCDEFp1 = _mm256_fmadd_ps(vi9x89ABCDEF, vk9x89ABCDEF, vacc89ABCDEFp1);

      const __m256 vi10x01234567 = _mm256_loadu_ps(i10);
      const __m256 vi10x89ABCDEF = _mm256_loadu_ps(i10 + 8);
      i10 += 16;

      const __m256 vk10x01234567 = _mm256_load_ps(w + 176);
      const __m256 vk10x89ABCDEF = _mm256_load_ps(w + 184);
      vacc01234567p0 = _mm256_fmadd_ps(vi10x01234567, vk10x01234567, vacc01234567p0);
      vacc89ABCDEFp0 = _mm256_fmadd_ps(vi10x89ABCDEF, vk10x89ABCDEF, vacc89ABCDEFp0);

      const __m256 vi11x01234567 = _mm256_loadu_ps(i11);
      const __m256 vi11x89ABCDEF = _mm256_loadu_ps(i11 + 8);
      i11 += 16;

      const __m256 vk11x01234567 = _mm256_load_ps(w + 192);
      const __m256 vk11x89ABCDEF = _mm256_load_ps(w + 200);
      vacc01234567p1 = _mm256_fmadd_ps(vi11x01234567, vk11x01234567, vacc01234567p1);
      vacc89ABCDEFp1 = _mm256_fmadd_ps(vi11x89ABCDEF, vk11x89ABCDEF, vacc89ABCDEFp1);

      const __m256 vi12x01234567 = _mm256_loadu_ps(i12);
      const __m256 vi12x89ABCDEF = _mm256_loadu_ps(i12 + 8);
      i12 += 16;

      const __m256 vk12x01234567 = _mm256_load_ps(w + 208);
      const __m256 vk12x89ABCDEF = _mm256_load_ps(w + 216);
      vacc01234567p0 = _mm256_fmadd_ps(vi12x01234567, vk12x01234567, vacc01234567p0);
      vacc89ABCDEFp0 = _mm256_fmadd_ps(vi12x89ABCDEF, vk12x89ABCDEF, vacc89ABCDEFp0);

      const __m256 vi13x01234567 = _mm256_loadu_ps(i13);
      const __m256 vi13x89ABCDEF = _mm256_loadu_ps(i13 + 8);
      i13 += 16;

      const __m256 vk13x01234567 = _mm256_load_ps(w + 224);
      const __m256 vk13x89ABCDEF = _mm256_load_ps(w + 232);
      vacc01234567p1 = _mm256_fmadd_ps(vi13x01234567, vk13x01234567, vacc01234567p1);
      vacc89ABCDEFp1 = _mm256_fmadd_ps(vi13x89ABCDEF, vk13x89ABCDEF, vacc89ABCDEFp1);

      const __m256 vi14x01234567 = _mm256_loadu_ps(i14);
      const __m256 vi14x89ABCDEF = _mm256_loadu_ps(i14 + 8);
      i14 += 16;

      const __m256 vk14x01234567 = _mm256_load_ps(w + 240);
      const __m256 vk14x89ABCDEF = _mm256_load_ps(w + 248);
      vacc01234567p0 = _mm256_fmadd_ps(vi14x01234567, vk14x01234567, vacc01234567p0);
      vacc89ABCDEFp0 = _mm256_fmadd_ps(vi14x89ABCDEF, vk14x89ABCDEF, vacc89ABCDEFp0);

      const __m256 vi15x01234567 = _mm256_loadu_ps(i15);
      const __m256 vi15x89ABCDEF = _mm256_loadu_ps(i15 + 8);
      i15 += 16;

      const __m256 vk15x01234567 = _mm256_load_ps(w + 256);
      const __m256 vk15x89ABCDEF = _mm256_load_ps(w + 264);
      vacc01234567p1 = _mm256_fmadd_ps(vi15x01234567, vk15x01234567, vacc01234567p1);
      vacc89ABCDEFp1 = _mm256_fmadd_ps(vi15x89ABCDEF, vk15x89ABCDEF, vacc89ABCDEFp1);

      const __m256 vi16x01234567 = _mm256_loadu_ps(i16);
      const __m256 vi16x89ABCDEF = _mm256_loadu_ps(i16 + 8);
      i16 += 16;

      const __m256 vk16x01234567 = _mm256_load_ps(w + 272);
      const __m256 vk16x89ABCDEF = _mm256_load_ps(w + 280);
      vacc01234567p0 = _mm256_fmadd_ps(vi16x01234567, vk16x01234567, vacc01234567p0);
      vacc89ABCDEFp0 = _mm256_fmadd_ps(vi16x89ABCDEF, vk16x89ABCDEF, vacc89ABCDEFp0);

      const __m256 vi17x01234567 = _mm256_loadu_ps(i17);
      const __m256 vi17x89ABCDEF = _mm256_loadu_ps(i17 + 8);
      i17 += 16;

      const __m256 vk17x01234567 = _mm256_load_ps(w + 288);
      const __m256 vk17x89ABCDEF = _mm256_load_ps(w + 296);
      vacc01234567p1 = _mm256_fmadd_ps(vi17x01234567, vk17x01234567, vacc01234567p1);
      vacc89ABCDEFp1 = _mm256_fmadd_ps(vi17x89ABCDEF, vk17x89ABCDEF, vacc89ABCDEFp1);

      const __m256 vi18x01234567 = _mm256_loadu_ps(i18);
      const __m256 vi18x89ABCDEF = _mm256_loadu_ps(i18 + 8);
      i18 += 16;

      const __m256 vk18x01234567 = _mm256_load_ps(w + 304);
      const __m256 vk18x89ABCDEF = _mm256_load_ps(w + 312);
      vacc01234567p0 = _mm256_fmadd_ps(vi18x01234567, vk18x01234567, vacc01234567p0);
      vacc89ABCDEFp0 = _mm256_fmadd_ps(vi18x89ABCDEF, vk18x89ABCDEF, vacc89ABCDEFp0);

      const __m256 vi19x01234567 = _mm256_loadu_ps(i19);
      const __m256 vi19x89ABCDEF = _mm256_loadu_ps(i19 + 8);
      i19 += 16;

      const __m256 vk19x01234567 = _mm256_load_ps(w + 320);
      const __m256 vk19x89ABCDEF = _mm256_load_ps(w + 328);
      vacc01234567p1 = _mm256_fmadd_ps(vi19x01234567, vk19x01234567, vacc01234567p1);
      vacc89ABCDEFp1 = _mm256_fmadd_ps(vi19x89ABCDEF, vk19x89ABCDEF, vacc89ABCDEFp1);

      const __m256 vi20x01234567 = _mm256_loadu_ps(i20);
      const __m256 vi20x89ABCDEF = _mm256_loadu_ps(i20 + 8);
      i20 += 16;

      const __m256 vk20x01234567 = _mm256_load_ps(w + 336);
      const __m256 vk20x89ABCDEF = _mm256_load_ps(w + 344);
      vacc01234567p0 = _mm256_fmadd_ps(vi20x01234567, vk20x01234567, vacc01234567p0);
      vacc89ABCDEFp0 = _mm256_fmadd_ps(vi20x89ABCDEF, vk20x89ABCDEF, vacc89ABCDEFp0);

      const __m256 vi21x01234567 = _mm256_loadu_ps(i21);
      const __m256 vi21x89ABCDEF = _mm256_loadu_ps(i21 + 8);
      i21 += 16;

      const __m256 vk21x01234567 = _mm256_load_ps(w + 352);
      const __m256 vk21x89ABCDEF = _mm256_load_ps(w + 360);
      vacc01234567p1 = _mm256_fmadd_ps(vi21x01234567, vk21x01234567, vacc01234567p1);
      vacc89ABCDEFp1 = _mm256_fmadd_ps(vi21x89ABCDEF, vk21x89ABCDEF, vacc89ABCDEFp1);

      const __m256 vi22x01234567 = _mm256_loadu_ps(i22);
      const __m256 vi22x89ABCDEF = _mm256_loadu_ps(i22 + 8);
      i22 += 16;

      const __m256 vk22x01234567 = _mm256_load_ps(w + 368);
      const __m256 vk22x89ABCDEF = _mm256_load_ps(w + 376);
      vacc01234567p0 = _mm256_fmadd_ps(vi22x01234567, vk22x01234567, vacc01234567p0);
      vacc89ABCDEFp0 = _mm256_fmadd_ps(vi22x89ABCDEF, vk22x89ABCDEF, vacc89ABCDEFp0);

      const __m256 vi23x01234567 = _mm256_loadu_ps(i23);
      const __m256 vi23x89ABCDEF = _mm256_loadu_ps(i23 + 8);
      i23 += 16;

      const __m256 vk23x01234567 = _mm256_load_ps(w + 384);
      const __m256 vk23x89ABCDEF = _mm256_load_ps(w + 392);
      vacc01234567p1 = _mm256_fmadd_ps(vi23x01234567, vk23x01234567, vacc01234567p1);
      vacc89ABCDEFp1 = _mm256_fmadd_ps(vi23x89ABCDEF, vk23x89ABCDEF, vacc89ABCDEFp1);

      const __m256 vi24x01234567 = _mm256_loadu_ps(i24);
      const __m256 vi24x89ABCDEF = _mm256_loadu_ps(i24 + 8);
      i24 += 16;

      const __m256 vk24x01234567 = _mm256_load_ps(w + 400);
      const __m256 vk24x89ABCDEF = _mm256_load_ps(w + 408);
      vacc01234567p0 = _mm256_fmadd_ps(vi24x01234567, vk24x01234567, vacc01234567p0);
      vacc89ABCDEFp0 = _mm256_fmadd_ps(vi24x89ABCDEF, vk24x89ABCDEF, vacc89ABCDEFp0);

      w += 416;

      // Add up all accumulators to vacc0123456789ABCDEFp0
      vacc01234567p0 = _mm256_add_ps(vacc01234567p0, vacc01234567p1);
      vacc89ABCDEFp0 = _mm256_add_ps(vacc89ABCDEFp0, vacc89ABCDEFp1);

      __m256 vacc01234567 = _mm256_max_ps(vmin, vacc01234567p0);
      __m256 vacc89ABCDEF = _mm256_max_ps(vmin, vacc89ABCDEFp0);
      vacc01234567 = _mm256_min_ps(vmax, vacc01234567);
      vacc89ABCDEF = _mm256_min_ps(vmax, vacc89ABCDEF);

      _mm256_storeu_ps(output, vacc01234567);
      _mm256_storeu_ps(output + 8, vacc89ABCDEF);
      output += 16;
    }
    for (; c >= 8; c -= 8) {
      __m256 vacc01234567p0 = _mm256_load_ps(w);

      const __m256 vi0x01234567 = _mm256_loadu_ps(i0);
      i0 += 8;

      const __m256 vk0x01234567 = _mm256_load_ps(w + 16);
      vacc01234567p0 = _mm256_fmadd_ps(vi0x01234567, vk0x01234567, vacc01234567p0);

      const __m256 vi1x01234567 = _mm256_loadu_ps(i1);
      i1 += 8;

      const __m256 vk1x01234567 = _mm256_load_ps(w + 32);
      __m256 vacc01234567p1 = _mm256_mul_ps(vi1x01234567, vk1x01234567);

      const __m256 vi2x01234567 = _mm256_loadu_ps(i2);
      i2 += 8;

      const __m256 vk2x01234567 = _mm256_load_ps(w + 48);
      vacc01234567p0 = _mm256_fmadd_ps(vi2x01234567, vk2x01234567, vacc01234567p0);

      const __m256 vi3x01234567 = _mm256_loadu_ps(i3);
      i3 += 8;

      const __m256 vk3x01234567 = _mm256_load_ps(w + 64);
      vacc01234567p1 = _mm256_fmadd_ps(vi3x01234567, vk3x01234567, vacc01234567p1);

      const __m256 vi4x01234567 = _mm256_loadu_ps(i4);
      i4 += 8;

      const __m256 vk4x01234567 = _mm256_load_ps(w + 80);
      vacc01234567p0 = _mm256_fmadd_ps(vi4x01234567, vk4x01234567, vacc01234567p0);

      const __m256 vi5x01234567 = _mm256_loadu_ps(i5);
      i5 += 8;

      const __m256 vk5x01234567 = _mm256_load_ps(w + 96);
      vacc01234567p1 = _mm256_fmadd_ps(vi5x01234567, vk5x01234567, vacc01234567p1);

      const __m256 vi6x01234567 = _mm256_loadu_ps(i6);
      i6 += 8;

      const __m256 vk6x01234567 = _mm256_load_ps(w + 112);
      vacc01234567p0 = _mm256_fmadd_ps(vi6x01234567, vk6x01234567, vacc01234567p0);

      const __m256 vi7x01234567 = _mm256_loadu_ps(i7);
      i7 += 8;

      const __m256 vk7x01234567 = _mm256_load_ps(w + 128);
      vacc01234567p1 = _mm256_fmadd_ps(vi7x01234567, vk7x01234567, vacc01234567p1);

      const __m256 vi8x01234567 = _mm256_loadu_ps(i8);
      i8 += 8;

      const __m256 vk8x01234567 = _mm256_load_ps(w + 144);
      vacc01234567p0 = _mm256_fmadd_ps(vi8x01234567, vk8x01234567, vacc01234567p0);

      const __m256 vi9x01234567 = _mm256_loadu_ps(i9);
      i9 += 8;

      const __m256 vk9x01234567 = _mm256_load_ps(w + 160);
      vacc01234567p1 = _mm256_fmadd_ps(vi9x01234567, vk9x01234567, vacc01234567p1);

      const __m256 vi10x01234567 = _mm256_loadu_ps(i10);
      i10 += 8;

      const __m256 vk10x01234567 = _mm256_load_ps(w + 176);
      vacc01234567p0 = _mm256_fmadd_ps(vi10x01234567, vk10x01234567, vacc01234567p0);

      const __m256 vi11x01234567 = _mm256_loadu_ps(i11);
      i11 += 8;

      const __m256 vk11x01234567 = _mm256_load_ps(w + 192);
      vacc01234567p1 = _mm256_fmadd_ps(vi11x01234567, vk11x01234567, vacc01234567p1);

      const __m256 vi12x01234567 = _mm256_loadu_ps(i12);
      i12 += 8;

      const __m256 vk12x01234567 = _mm256_load_ps(w + 208);
      vacc01234567p0 = _mm256_fmadd_ps(vi12x01234567, vk12x01234567, vacc01234567p0);

      const __m256 vi13x01234567 = _mm256_loadu_ps(i13);
      i13 += 8;

      const __m256 vk13x01234567 = _mm256_load_ps(w + 224);
      vacc01234567p1 = _mm256_fmadd_ps(vi13x01234567, vk13x01234567, vacc01234567p1);

      const __m256 vi14x01234567 = _mm256_loadu_ps(i14);
      i14 += 8;

      const __m256 vk14x01234567 = _mm256_load_ps(w + 240);
      vacc01234567p0 = _mm256_fmadd_ps(vi14x01234567, vk14x01234567, vacc01234567p0);

      const __m256 vi15x01234567 = _mm256_loadu_ps(i15);
      i15 += 8;

      const __m256 vk15x01234567 = _mm256_load_ps(w + 256);
      vacc01234567p1 = _mm256_fmadd_ps(vi15x01234567, vk15x01234567, vacc01234567p1);

      const __m256 vi16x01234567 = _mm256_loadu_ps(i16);
      i16 += 8;

      const __m256 vk16x01234567 = _mm256_load_ps(w + 272);
      vacc01234567p0 = _mm256_fmadd_ps(vi16x01234567, vk16x01234567, vacc01234567p0);

      const __m256 vi17x01234567 = _mm256_loadu_ps(i17);
      i17 += 8;

      const __m256 vk17x01234567 = _mm256_load_ps(w + 288);
      vacc01234567p1 = _mm256_fmadd_ps(vi17x01234567, vk17x01234567, vacc01234567p1);

      const __m256 vi18x01234567 = _mm256_loadu_ps(i18);
      i18 += 8;

      const __m256 vk18x01234567 = _mm256_load_ps(w + 304);
      vacc01234567p0 = _mm256_fmadd_ps(vi18x01234567, vk18x01234567, vacc01234567p0);

      const __m256 vi19x01234567 = _mm256_loadu_ps(i19);
      i19 += 8;

      const __m256 vk19x01234567 = _mm256_load_ps(w + 320);
      vacc01234567p1 = _mm256_fmadd_ps(vi19x01234567, vk19x01234567, vacc01234567p1);

      const __m256 vi20x01234567 = _mm256_loadu_ps(i20);
      i20 += 8;

      const __m256 vk20x01234567 = _mm256_load_ps(w + 336);
      vacc01234567p0 = _mm256_fmadd_ps(vi20x01234567, vk20x01234567, vacc01234567p0);

      const __m256 vi21x01234567 = _mm256_loadu_ps(i21);
      i21 += 8;

      const __m256 vk21x01234567 = _mm256_load_ps(w + 352);
      vacc01234567p1 = _mm256_fmadd_ps(vi21x01234567, vk21x01234567, vacc01234567p1);

      const __m256 vi22x01234567 = _mm256_loadu_ps(i22);
      i22 += 8;

      const __m256 vk22x01234567 = _mm256_load_ps(w + 368);
      vacc01234567p0 = _mm256_fmadd_ps(vi22x01234567, vk22x01234567, vacc01234567p0);

      const __m256 vi23x01234567 = _mm256_loadu_ps(i23);
      i23 += 8;

      const __m256 vk23x01234567 = _mm256_load_ps(w + 384);
      vacc01234567p1 = _mm256_fmadd_ps(vi23x01234567, vk23x01234567, vacc01234567p1);

      const __m256 vi24x01234567 = _mm256_loadu_ps(i24);
      i24 += 8;

      const __m256 vk24x01234567 = _mm256_load_ps(w + 400);
      vacc01234567p0 = _mm256_fmadd_ps(vi24x01234567, vk24x01234567, vacc01234567p0);

      w += 8;

      // Add up all accumulators to vacc01234567p0
      vacc01234567p0 = _mm256_add_ps(vacc01234567p0, vacc01234567p1);

      __m256 vacc01234567 = _mm256_max_ps(vmin, vacc01234567p0);
      vacc01234567 = _mm256_min_ps(vmax, vacc01234567);

      _mm256_storeu_ps(output, vacc01234567);
      output += 8;
    }
    if XNN_UNLIKELY(c != 0) {
      assert(c >= 1);
      assert(c <= 7);
      const __m256i vmask = _mm256_loadu_si256((const __m256i*) &mask_table[7 - c]);

      __m256 vacc01234567p0 = _mm256_load_ps(w);

      const __m256 vi0x01234567 = _mm256_maskload_ps(i0, vmask);
      const __m256 vk0x01234567 = _mm256_load_ps(w + 16);
      vacc01234567p0 = _mm256_fmadd_ps(vi0x01234567, vk0x01234567, vacc01234567p0);

      const __m256 vi1x01234567 = _mm256_maskload_ps(i1, vmask);
      const __m256 vk1x01234567 = _mm256_load_ps(w + 32);
      __m256 vacc01234567p1 = _mm256_mul_ps(vi1x01234567, vk1x01234567);

      const __m256 vi2x01234567 = _mm256_maskload_ps(i2, vmask);
      const __m256 vk2x01234567 = _mm256_load_ps(w + 48);
      vacc01234567p0 = _mm256_fmadd_ps(vi2x01234567, vk2x01234567, vacc01234567p0);

      const __m256 vi3x01234567 = _mm256_maskload_ps(i3, vmask);
      const __m256 vk3x01234567 = _mm256_load_ps(w + 64);
      vacc01234567p1 = _mm256_fmadd_ps(vi3x01234567, vk3x01234567, vacc01234567p1);

      const __m256 vi4x01234567 = _mm256_maskload_ps(i4, vmask);
      const __m256 vk4x01234567 = _mm256_load_ps(w + 80);
      vacc01234567p0 = _mm256_fmadd_ps(vi4x01234567, vk4x01234567, vacc01234567p0);

      const __m256 vi5x01234567 = _mm256_maskload_ps(i5, vmask);
      const __m256 vk5x01234567 = _mm256_load_ps(w + 96);
      vacc01234567p1 = _mm256_fmadd_ps(vi5x01234567, vk5x01234567, vacc01234567p1);

      const __m256 vi6x01234567 = _mm256_maskload_ps(i6, vmask);
      const __m256 vk6x01234567 = _mm256_load_ps(w + 112);
      vacc01234567p0 = _mm256_fmadd_ps(vi6x01234567, vk6x01234567, vacc01234567p0);

      const __m256 vi7x01234567 = _mm256_maskload_ps(i7, vmask);
      const __m256 vk7x01234567 = _mm256_load_ps(w + 128);
      vacc01234567p1 = _mm256_fmadd_ps(vi7x01234567, vk7x01234567, vacc01234567p1);

      const __m256 vi8x01234567 = _mm256_maskload_ps(i8, vmask);
      const __m256 vk8x01234567 = _mm256_load_ps(w + 144);
      vacc01234567p0 = _mm256_fmadd_ps(vi8x01234567, vk8x01234567, vacc01234567p0);

      const __m256 vi9x01234567 = _mm256_maskload_ps(i9, vmask);
      const __m256 vk9x01234567 = _mm256_load_ps(w + 160);
      vacc01234567p1 = _mm256_fmadd_ps(vi9x01234567, vk9x01234567, vacc01234567p1);

      const __m256 vi10x01234567 = _mm256_maskload_ps(i10, vmask);
      const __m256 vk10x01234567 = _mm256_load_ps(w + 176);
      vacc01234567p0 = _mm256_fmadd_ps(vi10x01234567, vk10x01234567, vacc01234567p0);

      const __m256 vi11x01234567 = _mm256_maskload_ps(i11, vmask);
      const __m256 vk11x01234567 = _mm256_load_ps(w + 192);
      vacc01234567p1 = _mm256_fmadd_ps(vi11x01234567, vk11x01234567, vacc01234567p1);

      const __m256 vi12x01234567 = _mm256_maskload_ps(i12, vmask);
      const __m256 vk12x01234567 = _mm256_load_ps(w + 208);
      vacc01234567p0 = _mm256_fmadd_ps(vi12x01234567, vk12x01234567, vacc01234567p0);

      const __m256 vi13x01234567 = _mm256_maskload_ps(i13, vmask);
      const __m256 vk13x01234567 = _mm256_load_ps(w + 224);
      vacc01234567p1 = _mm256_fmadd_ps(vi13x01234567, vk13x01234567, vacc01234567p1);

      const __m256 vi14x01234567 = _mm256_maskload_ps(i14, vmask);
      const __m256 vk14x01234567 = _mm256_load_ps(w + 240);
      vacc01234567p0 = _mm256_fmadd_ps(vi14x01234567, vk14x01234567, vacc01234567p0);

      const __m256 vi15x01234567 = _mm256_maskload_ps(i15, vmask);
      const __m256 vk15x01234567 = _mm256_load_ps(w + 256);
      vacc01234567p1 = _mm256_fmadd_ps(vi15x01234567, vk15x01234567, vacc01234567p1);

      const __m256 vi16x01234567 = _mm256_maskload_ps(i16, vmask);
      const __m256 vk16x01234567 = _mm256_load_ps(w + 272);
      vacc01234567p0 = _mm256_fmadd_ps(vi16x01234567, vk16x01234567, vacc01234567p0);

      const __m256 vi17x01234567 = _mm256_maskload_ps(i17, vmask);
      const __m256 vk17x01234567 = _mm256_load_ps(w + 288);
      vacc01234567p1 = _mm256_fmadd_ps(vi17x01234567, vk17x01234567, vacc01234567p1);

      const __m256 vi18x01234567 = _mm256_maskload_ps(i18, vmask);
      const __m256 vk18x01234567 = _mm256_load_ps(w + 304);
      vacc01234567p0 = _mm256_fmadd_ps(vi18x01234567, vk18x01234567, vacc01234567p0);

      const __m256 vi19x01234567 = _mm256_maskload_ps(i19, vmask);
      const __m256 vk19x01234567 = _mm256_load_ps(w + 320);
      vacc01234567p1 = _mm256_fmadd_ps(vi19x01234567, vk19x01234567, vacc01234567p1);

      const __m256 vi20x01234567 = _mm256_maskload_ps(i20, vmask);
      const __m256 vk20x01234567 = _mm256_load_ps(w + 336);
      vacc01234567p0 = _mm256_fmadd_ps(vi20x01234567, vk20x01234567, vacc01234567p0);

      const __m256 vi21x01234567 = _mm256_maskload_ps(i21, vmask);
      const __m256 vk21x01234567 = _mm256_load_ps(w + 352);
      vacc01234567p1 = _mm256_fmadd_ps(vi21x01234567, vk21x01234567, vacc01234567p1);

      const __m256 vi22x01234567 = _mm256_maskload_ps(i22, vmask);
      const __m256 vk22x01234567 = _mm256_load_ps(w + 368);
      vacc01234567p0 = _mm256_fmadd_ps(vi22x01234567, vk22x01234567, vacc01234567p0);

      const __m256 vi23x01234567 = _mm256_maskload_ps(i23, vmask);
      const __m256 vk23x01234567 = _mm256_load_ps(w + 384);
      vacc01234567p1 = _mm256_fmadd_ps(vi23x01234567, vk23x01234567, vacc01234567p1);

      const __m256 vi24x01234567 = _mm256_maskload_ps(i24, vmask);
      const __m256 vk24x01234567 = _mm256_load_ps(w + 400);
      vacc01234567p0 = _mm256_fmadd_ps(vi24x01234567, vk24x01234567, vacc01234567p0);

      // Add up all accumulators to vacc01234567p0
      vacc01234567p0 = _mm256_add_ps(vacc01234567p0, vacc01234567p1);

      __m256 vacc01234567 = _mm256_max_ps(vmin, vacc01234567p0);
      vacc01234567 = _mm256_min_ps(vmax, vacc01234567);

      __m128 vacc0123 = _mm256_castps256_ps128(vacc01234567);
      if (c & 4) {
        _mm_storeu_ps(output, vacc0123);
        vacc0123 = _mm256_extractf128_ps(vacc01234567, 1);
        output += 4;
      }
      if (c & 2) {
        _mm_storel_pi((__m64*) output, vacc0123);
        vacc0123 = _mm_movehl_ps(vacc0123, vacc0123);
        output += 2;
      }
      if (c & 1) {
        _mm_store_ss(output, vacc0123);
        output += 1;
      }
    }

    output = (float*) ((uintptr_t) output + output_increment);
  } while (--output_width != 0);
}
