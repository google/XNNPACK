// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/f16-dwconv/unipass.c.in
//   Generator: tools/xngen
//
// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include "src/xnnpack/common.h"
#include "src/xnnpack/dwconv.h"
#include "src/xnnpack/microparams.h"
#include "src/xnnpack/simd/f16-wasmrelaxedsimd.h"


void xnn_f16_dwconv_minmax_ukernel_25p8c__wasmrelaxedsimd_acc2(
    size_t channels,
    size_t output_width,
    const xnn_float16** input,
    const xnn_float16* weights,
    xnn_float16* output,
    intptr_t input_stride,
    size_t output_increment,
    size_t input_offset,
    size_t input_pixel_stride,
    const xnn_float16* zero,
    const struct xnn_f16_minmax_params* restrict params)
{
  assert(channels != 0);
  assert(output_width != 0);
  assert(xnn_simd_size_f16 == 8);

  const xnn_simd_f16_t vmin = xnn_set1_f16(params->scalar.min);
  const xnn_simd_f16_t vmax = xnn_set1_f16(params->scalar.max);
  do {
    const xnn_float16* i0 = input[0];
    assert(i0 != NULL);
    if XNN_UNPREDICTABLE(i0 != zero) {
      i0 = (const xnn_float16*) ((uintptr_t) i0 + input_offset);
    }
    const xnn_float16* i1 = input[1];
    assert(i1 != NULL);
    if XNN_UNPREDICTABLE(i1 != zero) {
      i1 = (const xnn_float16*) ((uintptr_t) i1 + input_offset);
    }
    const xnn_float16* i2 = input[2];
    assert(i2 != NULL);
    if XNN_UNPREDICTABLE(i2 != zero) {
      i2 = (const xnn_float16*) ((uintptr_t) i2 + input_offset);
    }
    const xnn_float16* i3 = input[3];
    assert(i3 != NULL);
    if XNN_UNPREDICTABLE(i3 != zero) {
      i3 = (const xnn_float16*) ((uintptr_t) i3 + input_offset);
    }
    const xnn_float16* i4 = input[4];
    assert(i4 != NULL);
    if XNN_UNPREDICTABLE(i4 != zero) {
      i4 = (const xnn_float16*) ((uintptr_t) i4 + input_offset);
    }
    const xnn_float16* i5 = input[5];
    assert(i5 != NULL);
    if XNN_UNPREDICTABLE(i5 != zero) {
      i5 = (const xnn_float16*) ((uintptr_t) i5 + input_offset);
    }
    const xnn_float16* i6 = input[6];
    assert(i6 != NULL);
    if XNN_UNPREDICTABLE(i6 != zero) {
      i6 = (const xnn_float16*) ((uintptr_t) i6 + input_offset);
    }
    const xnn_float16* i7 = input[7];
    assert(i7 != NULL);
    if XNN_UNPREDICTABLE(i7 != zero) {
      i7 = (const xnn_float16*) ((uintptr_t) i7 + input_offset);
    }
    const xnn_float16* i8 = input[8];
    assert(i8 != NULL);
    if XNN_UNPREDICTABLE(i8 != zero) {
      i8 = (const xnn_float16*) ((uintptr_t) i8 + input_offset);
    }
    const xnn_float16* i9 = input[9];
    assert(i9 != NULL);
    if XNN_UNPREDICTABLE(i9 != zero) {
      i9 = (const xnn_float16*) ((uintptr_t) i9 + input_offset);
    }
    const xnn_float16* i10 = input[10];
    assert(i10 != NULL);
    if XNN_UNPREDICTABLE(i10 != zero) {
      i10 = (const xnn_float16*) ((uintptr_t) i10 + input_offset);
    }
    const xnn_float16* i11 = input[11];
    assert(i11 != NULL);
    if XNN_UNPREDICTABLE(i11 != zero) {
      i11 = (const xnn_float16*) ((uintptr_t) i11 + input_offset);
    }
    const xnn_float16* i12 = input[12];
    assert(i12 != NULL);
    if XNN_UNPREDICTABLE(i12 != zero) {
      i12 = (const xnn_float16*) ((uintptr_t) i12 + input_offset);
    }
    const xnn_float16* i13 = input[13];
    assert(i13 != NULL);
    if XNN_UNPREDICTABLE(i13 != zero) {
      i13 = (const xnn_float16*) ((uintptr_t) i13 + input_offset);
    }
    const xnn_float16* i14 = input[14];
    assert(i14 != NULL);
    if XNN_UNPREDICTABLE(i14 != zero) {
      i14 = (const xnn_float16*) ((uintptr_t) i14 + input_offset);
    }
    const xnn_float16* i15 = input[15];
    assert(i15 != NULL);
    if XNN_UNPREDICTABLE(i15 != zero) {
      i15 = (const xnn_float16*) ((uintptr_t) i15 + input_offset);
    }
    const xnn_float16* i16 = input[16];
    assert(i16 != NULL);
    if XNN_UNPREDICTABLE(i16 != zero) {
      i16 = (const xnn_float16*) ((uintptr_t) i16 + input_offset);
    }
    const xnn_float16* i17 = input[17];
    assert(i17 != NULL);
    if XNN_UNPREDICTABLE(i17 != zero) {
      i17 = (const xnn_float16*) ((uintptr_t) i17 + input_offset);
    }
    const xnn_float16* i18 = input[18];
    assert(i18 != NULL);
    if XNN_UNPREDICTABLE(i18 != zero) {
      i18 = (const xnn_float16*) ((uintptr_t) i18 + input_offset);
    }
    const xnn_float16* i19 = input[19];
    assert(i19 != NULL);
    if XNN_UNPREDICTABLE(i19 != zero) {
      i19 = (const xnn_float16*) ((uintptr_t) i19 + input_offset);
    }
    const xnn_float16* i20 = input[20];
    assert(i20 != NULL);
    if XNN_UNPREDICTABLE(i20 != zero) {
      i20 = (const xnn_float16*) ((uintptr_t) i20 + input_offset);
    }
    const xnn_float16* i21 = input[21];
    assert(i21 != NULL);
    if XNN_UNPREDICTABLE(i21 != zero) {
      i21 = (const xnn_float16*) ((uintptr_t) i21 + input_offset);
    }
    const xnn_float16* i22 = input[22];
    assert(i22 != NULL);
    if XNN_UNPREDICTABLE(i22 != zero) {
      i22 = (const xnn_float16*) ((uintptr_t) i22 + input_offset);
    }
    const xnn_float16* i23 = input[23];
    assert(i23 != NULL);
    if XNN_UNPREDICTABLE(i23 != zero) {
      i23 = (const xnn_float16*) ((uintptr_t) i23 + input_offset);
    }
    const xnn_float16* i24 = input[24];
    assert(i24 != NULL);
    if XNN_UNPREDICTABLE(i24 != zero) {
      i24 = (const xnn_float16*) ((uintptr_t) i24 + input_offset);
    }
    input = (const xnn_float16**) ((uintptr_t) input + input_stride);

    size_t c = channels;
    const xnn_float16* w = weights;
    for (; c >= 8; c -= 8) {
      xnn_simd_f16_t vacc01234567p0 = xnn_loadu_f16(w + 0);

      const xnn_simd_f16_t vi0x01234567 = xnn_loadu_f16(i0 + 0);
      const xnn_simd_f16_t vk0x01234567 = xnn_loadu_f16(w + 8);
      vacc01234567p0 = xnn_fmadd_f16(vi0x01234567, vk0x01234567, vacc01234567p0);
      i0 += 8;
      const xnn_simd_f16_t vi1x01234567 = xnn_loadu_f16(i1 + 0);
      const xnn_simd_f16_t vk1x01234567 = xnn_loadu_f16(w + 16);
      xnn_simd_f16_t vacc01234567p1 = xnn_mul_f16(vi1x01234567, vk1x01234567);
      i1 += 8;
      const xnn_simd_f16_t vi2x01234567 = xnn_loadu_f16(i2 + 0);
      const xnn_simd_f16_t vk2x01234567 = xnn_loadu_f16(w + 24);
      vacc01234567p0 = xnn_fmadd_f16(vi2x01234567, vk2x01234567, vacc01234567p0);
      i2 += 8;
      const xnn_simd_f16_t vi3x01234567 = xnn_loadu_f16(i3 + 0);
      const xnn_simd_f16_t vk3x01234567 = xnn_loadu_f16(w + 32);
      vacc01234567p1 = xnn_fmadd_f16(vi3x01234567, vk3x01234567, vacc01234567p1);
      i3 += 8;
      const xnn_simd_f16_t vi4x01234567 = xnn_loadu_f16(i4 + 0);
      const xnn_simd_f16_t vk4x01234567 = xnn_loadu_f16(w + 40);
      vacc01234567p0 = xnn_fmadd_f16(vi4x01234567, vk4x01234567, vacc01234567p0);
      i4 += 8;
      const xnn_simd_f16_t vi5x01234567 = xnn_loadu_f16(i5 + 0);
      const xnn_simd_f16_t vk5x01234567 = xnn_loadu_f16(w + 48);
      vacc01234567p1 = xnn_fmadd_f16(vi5x01234567, vk5x01234567, vacc01234567p1);
      i5 += 8;
      const xnn_simd_f16_t vi6x01234567 = xnn_loadu_f16(i6 + 0);
      const xnn_simd_f16_t vk6x01234567 = xnn_loadu_f16(w + 56);
      vacc01234567p0 = xnn_fmadd_f16(vi6x01234567, vk6x01234567, vacc01234567p0);
      i6 += 8;
      const xnn_simd_f16_t vi7x01234567 = xnn_loadu_f16(i7 + 0);
      const xnn_simd_f16_t vk7x01234567 = xnn_loadu_f16(w + 64);
      vacc01234567p1 = xnn_fmadd_f16(vi7x01234567, vk7x01234567, vacc01234567p1);
      i7 += 8;
      const xnn_simd_f16_t vi8x01234567 = xnn_loadu_f16(i8 + 0);
      const xnn_simd_f16_t vk8x01234567 = xnn_loadu_f16(w + 72);
      vacc01234567p0 = xnn_fmadd_f16(vi8x01234567, vk8x01234567, vacc01234567p0);
      i8 += 8;
      const xnn_simd_f16_t vi9x01234567 = xnn_loadu_f16(i9 + 0);
      const xnn_simd_f16_t vk9x01234567 = xnn_loadu_f16(w + 80);
      vacc01234567p1 = xnn_fmadd_f16(vi9x01234567, vk9x01234567, vacc01234567p1);
      i9 += 8;
      const xnn_simd_f16_t vi10x01234567 = xnn_loadu_f16(i10 + 0);
      const xnn_simd_f16_t vk10x01234567 = xnn_loadu_f16(w + 88);
      vacc01234567p0 = xnn_fmadd_f16(vi10x01234567, vk10x01234567, vacc01234567p0);
      i10 += 8;
      const xnn_simd_f16_t vi11x01234567 = xnn_loadu_f16(i11 + 0);
      const xnn_simd_f16_t vk11x01234567 = xnn_loadu_f16(w + 96);
      vacc01234567p1 = xnn_fmadd_f16(vi11x01234567, vk11x01234567, vacc01234567p1);
      i11 += 8;
      const xnn_simd_f16_t vi12x01234567 = xnn_loadu_f16(i12 + 0);
      const xnn_simd_f16_t vk12x01234567 = xnn_loadu_f16(w + 104);
      vacc01234567p0 = xnn_fmadd_f16(vi12x01234567, vk12x01234567, vacc01234567p0);
      i12 += 8;
      const xnn_simd_f16_t vi13x01234567 = xnn_loadu_f16(i13 + 0);
      const xnn_simd_f16_t vk13x01234567 = xnn_loadu_f16(w + 112);
      vacc01234567p1 = xnn_fmadd_f16(vi13x01234567, vk13x01234567, vacc01234567p1);
      i13 += 8;
      const xnn_simd_f16_t vi14x01234567 = xnn_loadu_f16(i14 + 0);
      const xnn_simd_f16_t vk14x01234567 = xnn_loadu_f16(w + 120);
      vacc01234567p0 = xnn_fmadd_f16(vi14x01234567, vk14x01234567, vacc01234567p0);
      i14 += 8;
      const xnn_simd_f16_t vi15x01234567 = xnn_loadu_f16(i15 + 0);
      const xnn_simd_f16_t vk15x01234567 = xnn_loadu_f16(w + 128);
      vacc01234567p1 = xnn_fmadd_f16(vi15x01234567, vk15x01234567, vacc01234567p1);
      i15 += 8;
      const xnn_simd_f16_t vi16x01234567 = xnn_loadu_f16(i16 + 0);
      const xnn_simd_f16_t vk16x01234567 = xnn_loadu_f16(w + 136);
      vacc01234567p0 = xnn_fmadd_f16(vi16x01234567, vk16x01234567, vacc01234567p0);
      i16 += 8;
      const xnn_simd_f16_t vi17x01234567 = xnn_loadu_f16(i17 + 0);
      const xnn_simd_f16_t vk17x01234567 = xnn_loadu_f16(w + 144);
      vacc01234567p1 = xnn_fmadd_f16(vi17x01234567, vk17x01234567, vacc01234567p1);
      i17 += 8;
      const xnn_simd_f16_t vi18x01234567 = xnn_loadu_f16(i18 + 0);
      const xnn_simd_f16_t vk18x01234567 = xnn_loadu_f16(w + 152);
      vacc01234567p0 = xnn_fmadd_f16(vi18x01234567, vk18x01234567, vacc01234567p0);
      i18 += 8;
      const xnn_simd_f16_t vi19x01234567 = xnn_loadu_f16(i19 + 0);
      const xnn_simd_f16_t vk19x01234567 = xnn_loadu_f16(w + 160);
      vacc01234567p1 = xnn_fmadd_f16(vi19x01234567, vk19x01234567, vacc01234567p1);
      i19 += 8;
      const xnn_simd_f16_t vi20x01234567 = xnn_loadu_f16(i20 + 0);
      const xnn_simd_f16_t vk20x01234567 = xnn_loadu_f16(w + 168);
      vacc01234567p0 = xnn_fmadd_f16(vi20x01234567, vk20x01234567, vacc01234567p0);
      i20 += 8;
      const xnn_simd_f16_t vi21x01234567 = xnn_loadu_f16(i21 + 0);
      const xnn_simd_f16_t vk21x01234567 = xnn_loadu_f16(w + 176);
      vacc01234567p1 = xnn_fmadd_f16(vi21x01234567, vk21x01234567, vacc01234567p1);
      i21 += 8;
      const xnn_simd_f16_t vi22x01234567 = xnn_loadu_f16(i22 + 0);
      const xnn_simd_f16_t vk22x01234567 = xnn_loadu_f16(w + 184);
      vacc01234567p0 = xnn_fmadd_f16(vi22x01234567, vk22x01234567, vacc01234567p0);
      i22 += 8;
      const xnn_simd_f16_t vi23x01234567 = xnn_loadu_f16(i23 + 0);
      const xnn_simd_f16_t vk23x01234567 = xnn_loadu_f16(w + 192);
      vacc01234567p1 = xnn_fmadd_f16(vi23x01234567, vk23x01234567, vacc01234567p1);
      i23 += 8;
      const xnn_simd_f16_t vi24x01234567 = xnn_loadu_f16(i24 + 0);
      const xnn_simd_f16_t vk24x01234567 = xnn_loadu_f16(w + 200);
      vacc01234567p0 = xnn_fmadd_f16(vi24x01234567, vk24x01234567, vacc01234567p0);
      i24 += 8;

      w += 208;

      vacc01234567p0 = xnn_add_f16(vacc01234567p0, vacc01234567p1);

      xnn_simd_f16_t vacc01234567 = xnn_max_f16(vacc01234567p0, vmin);
      vacc01234567 = xnn_min_f16(vacc01234567, vmax);
      xnn_storeu_f16(output + 0, vacc01234567);
      output += 8;
    }
    if XNN_UNLIKELY(c != 0) {
      xnn_simd_f16_t vacc01234567p0 = xnn_loadu_f16(w);

      const xnn_simd_f16_t vi0x01234567 = xnn_load_tail_f16(i0, c);
      const xnn_simd_f16_t vk0x01234567 = xnn_loadu_f16(w + 8);
      vacc01234567p0 = xnn_fmadd_f16(vi0x01234567, vk0x01234567, vacc01234567p0);
      const xnn_simd_f16_t vi1x01234567 = xnn_load_tail_f16(i1, c);
      const xnn_simd_f16_t vk1x01234567 = xnn_loadu_f16(w + 16);
      xnn_simd_f16_t vacc01234567p1 = xnn_mul_f16(vi1x01234567, vk1x01234567);
      const xnn_simd_f16_t vi2x01234567 = xnn_load_tail_f16(i2, c);
      const xnn_simd_f16_t vk2x01234567 = xnn_loadu_f16(w + 24);
      vacc01234567p0 = xnn_fmadd_f16(vi2x01234567, vk2x01234567, vacc01234567p0);
      const xnn_simd_f16_t vi3x01234567 = xnn_load_tail_f16(i3, c);
      const xnn_simd_f16_t vk3x01234567 = xnn_loadu_f16(w + 32);
      vacc01234567p1 = xnn_fmadd_f16(vi3x01234567, vk3x01234567, vacc01234567p1);
      const xnn_simd_f16_t vi4x01234567 = xnn_load_tail_f16(i4, c);
      const xnn_simd_f16_t vk4x01234567 = xnn_loadu_f16(w + 40);
      vacc01234567p0 = xnn_fmadd_f16(vi4x01234567, vk4x01234567, vacc01234567p0);
      const xnn_simd_f16_t vi5x01234567 = xnn_load_tail_f16(i5, c);
      const xnn_simd_f16_t vk5x01234567 = xnn_loadu_f16(w + 48);
      vacc01234567p1 = xnn_fmadd_f16(vi5x01234567, vk5x01234567, vacc01234567p1);
      const xnn_simd_f16_t vi6x01234567 = xnn_load_tail_f16(i6, c);
      const xnn_simd_f16_t vk6x01234567 = xnn_loadu_f16(w + 56);
      vacc01234567p0 = xnn_fmadd_f16(vi6x01234567, vk6x01234567, vacc01234567p0);
      const xnn_simd_f16_t vi7x01234567 = xnn_load_tail_f16(i7, c);
      const xnn_simd_f16_t vk7x01234567 = xnn_loadu_f16(w + 64);
      vacc01234567p1 = xnn_fmadd_f16(vi7x01234567, vk7x01234567, vacc01234567p1);
      const xnn_simd_f16_t vi8x01234567 = xnn_load_tail_f16(i8, c);
      const xnn_simd_f16_t vk8x01234567 = xnn_loadu_f16(w + 72);
      vacc01234567p0 = xnn_fmadd_f16(vi8x01234567, vk8x01234567, vacc01234567p0);
      const xnn_simd_f16_t vi9x01234567 = xnn_load_tail_f16(i9, c);
      const xnn_simd_f16_t vk9x01234567 = xnn_loadu_f16(w + 80);
      vacc01234567p1 = xnn_fmadd_f16(vi9x01234567, vk9x01234567, vacc01234567p1);
      const xnn_simd_f16_t vi10x01234567 = xnn_load_tail_f16(i10, c);
      const xnn_simd_f16_t vk10x01234567 = xnn_loadu_f16(w + 88);
      vacc01234567p0 = xnn_fmadd_f16(vi10x01234567, vk10x01234567, vacc01234567p0);
      const xnn_simd_f16_t vi11x01234567 = xnn_load_tail_f16(i11, c);
      const xnn_simd_f16_t vk11x01234567 = xnn_loadu_f16(w + 96);
      vacc01234567p1 = xnn_fmadd_f16(vi11x01234567, vk11x01234567, vacc01234567p1);
      const xnn_simd_f16_t vi12x01234567 = xnn_load_tail_f16(i12, c);
      const xnn_simd_f16_t vk12x01234567 = xnn_loadu_f16(w + 104);
      vacc01234567p0 = xnn_fmadd_f16(vi12x01234567, vk12x01234567, vacc01234567p0);
      const xnn_simd_f16_t vi13x01234567 = xnn_load_tail_f16(i13, c);
      const xnn_simd_f16_t vk13x01234567 = xnn_loadu_f16(w + 112);
      vacc01234567p1 = xnn_fmadd_f16(vi13x01234567, vk13x01234567, vacc01234567p1);
      const xnn_simd_f16_t vi14x01234567 = xnn_load_tail_f16(i14, c);
      const xnn_simd_f16_t vk14x01234567 = xnn_loadu_f16(w + 120);
      vacc01234567p0 = xnn_fmadd_f16(vi14x01234567, vk14x01234567, vacc01234567p0);
      const xnn_simd_f16_t vi15x01234567 = xnn_load_tail_f16(i15, c);
      const xnn_simd_f16_t vk15x01234567 = xnn_loadu_f16(w + 128);
      vacc01234567p1 = xnn_fmadd_f16(vi15x01234567, vk15x01234567, vacc01234567p1);
      const xnn_simd_f16_t vi16x01234567 = xnn_load_tail_f16(i16, c);
      const xnn_simd_f16_t vk16x01234567 = xnn_loadu_f16(w + 136);
      vacc01234567p0 = xnn_fmadd_f16(vi16x01234567, vk16x01234567, vacc01234567p0);
      const xnn_simd_f16_t vi17x01234567 = xnn_load_tail_f16(i17, c);
      const xnn_simd_f16_t vk17x01234567 = xnn_loadu_f16(w + 144);
      vacc01234567p1 = xnn_fmadd_f16(vi17x01234567, vk17x01234567, vacc01234567p1);
      const xnn_simd_f16_t vi18x01234567 = xnn_load_tail_f16(i18, c);
      const xnn_simd_f16_t vk18x01234567 = xnn_loadu_f16(w + 152);
      vacc01234567p0 = xnn_fmadd_f16(vi18x01234567, vk18x01234567, vacc01234567p0);
      const xnn_simd_f16_t vi19x01234567 = xnn_load_tail_f16(i19, c);
      const xnn_simd_f16_t vk19x01234567 = xnn_loadu_f16(w + 160);
      vacc01234567p1 = xnn_fmadd_f16(vi19x01234567, vk19x01234567, vacc01234567p1);
      const xnn_simd_f16_t vi20x01234567 = xnn_load_tail_f16(i20, c);
      const xnn_simd_f16_t vk20x01234567 = xnn_loadu_f16(w + 168);
      vacc01234567p0 = xnn_fmadd_f16(vi20x01234567, vk20x01234567, vacc01234567p0);
      const xnn_simd_f16_t vi21x01234567 = xnn_load_tail_f16(i21, c);
      const xnn_simd_f16_t vk21x01234567 = xnn_loadu_f16(w + 176);
      vacc01234567p1 = xnn_fmadd_f16(vi21x01234567, vk21x01234567, vacc01234567p1);
      const xnn_simd_f16_t vi22x01234567 = xnn_load_tail_f16(i22, c);
      const xnn_simd_f16_t vk22x01234567 = xnn_loadu_f16(w + 184);
      vacc01234567p0 = xnn_fmadd_f16(vi22x01234567, vk22x01234567, vacc01234567p0);
      const xnn_simd_f16_t vi23x01234567 = xnn_load_tail_f16(i23, c);
      const xnn_simd_f16_t vk23x01234567 = xnn_loadu_f16(w + 192);
      vacc01234567p1 = xnn_fmadd_f16(vi23x01234567, vk23x01234567, vacc01234567p1);
      const xnn_simd_f16_t vi24x01234567 = xnn_load_tail_f16(i24, c);
      const xnn_simd_f16_t vk24x01234567 = xnn_loadu_f16(w + 200);
      vacc01234567p0 = xnn_fmadd_f16(vi24x01234567, vk24x01234567, vacc01234567p0);

      vacc01234567p0 = xnn_add_f16(vacc01234567p0, vacc01234567p1);

      xnn_simd_f16_t vacc01234567 = xnn_max_f16(vacc01234567p0, vmin);
      vacc01234567 = xnn_min_f16(vacc01234567, vmax);
      xnn_store_tail_f16(output, vacc01234567, c);
      output += c;
    }

    input_offset += input_pixel_stride;
    output = (xnn_float16*) ((uintptr_t) output + output_increment);
  } while (--output_width != 0);
}
