// Auto-generated file. Do not edit!
//   Template: src/f32-dwconv/unipass-wasmsimd.c.in
//   Generator: tools/xngen
//
// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <wasm_simd128.h>

#include "xnnpack/dwconv.h"


void xnn_f32_dwconv_minmax_ukernel_25p4c__wasmrelaxedsimd_fma(
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

  const v128_t vmin = wasm_v128_load32_splat(&params->wasmsimd.min);
  const v128_t vmax = wasm_v128_load32_splat(&params->wasmsimd.max);
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
    for (; c >= 4; c -= 4) {
      v128_t vacc0123p0 = wasm_v128_load(w);


      const v128_t vi0x0123 = wasm_v128_load(i0);
      i0 += 4;

      const v128_t vk0x0123 = wasm_v128_load(w + 4);
      vacc0123p0 = wasm_f32x4_relaxed_madd(vi0x0123, vk0x0123, vacc0123p0);

      const v128_t vi1x0123 = wasm_v128_load(i1);
      i1 += 4;

      const v128_t vk1x0123 = wasm_v128_load(w + 8);
      vacc0123p0 = wasm_f32x4_relaxed_madd(vi1x0123, vk1x0123, vacc0123p0);

      const v128_t vi2x0123 = wasm_v128_load(i2);
      i2 += 4;

      const v128_t vk2x0123 = wasm_v128_load(w + 12);
      vacc0123p0 = wasm_f32x4_relaxed_madd(vi2x0123, vk2x0123, vacc0123p0);

      const v128_t vi3x0123 = wasm_v128_load(i3);
      i3 += 4;

      const v128_t vk3x0123 = wasm_v128_load(w + 16);
      vacc0123p0 = wasm_f32x4_relaxed_madd(vi3x0123, vk3x0123, vacc0123p0);

      const v128_t vi4x0123 = wasm_v128_load(i4);
      i4 += 4;

      const v128_t vk4x0123 = wasm_v128_load(w + 20);
      vacc0123p0 = wasm_f32x4_relaxed_madd(vi4x0123, vk4x0123, vacc0123p0);

      const v128_t vi5x0123 = wasm_v128_load(i5);
      i5 += 4;

      const v128_t vk5x0123 = wasm_v128_load(w + 24);
      vacc0123p0 = wasm_f32x4_relaxed_madd(vi5x0123, vk5x0123, vacc0123p0);

      const v128_t vi6x0123 = wasm_v128_load(i6);
      i6 += 4;

      const v128_t vk6x0123 = wasm_v128_load(w + 28);
      vacc0123p0 = wasm_f32x4_relaxed_madd(vi6x0123, vk6x0123, vacc0123p0);

      const v128_t vi7x0123 = wasm_v128_load(i7);
      i7 += 4;

      const v128_t vk7x0123 = wasm_v128_load(w + 32);
      vacc0123p0 = wasm_f32x4_relaxed_madd(vi7x0123, vk7x0123, vacc0123p0);

      const v128_t vi8x0123 = wasm_v128_load(i8);
      i8 += 4;

      const v128_t vk8x0123 = wasm_v128_load(w + 36);
      vacc0123p0 = wasm_f32x4_relaxed_madd(vi8x0123, vk8x0123, vacc0123p0);

      const v128_t vi9x0123 = wasm_v128_load(i9);
      i9 += 4;

      const v128_t vk9x0123 = wasm_v128_load(w + 40);
      vacc0123p0 = wasm_f32x4_relaxed_madd(vi9x0123, vk9x0123, vacc0123p0);

      const v128_t vi10x0123 = wasm_v128_load(i10);
      i10 += 4;

      const v128_t vk10x0123 = wasm_v128_load(w + 44);
      vacc0123p0 = wasm_f32x4_relaxed_madd(vi10x0123, vk10x0123, vacc0123p0);

      const v128_t vi11x0123 = wasm_v128_load(i11);
      i11 += 4;

      const v128_t vk11x0123 = wasm_v128_load(w + 48);
      vacc0123p0 = wasm_f32x4_relaxed_madd(vi11x0123, vk11x0123, vacc0123p0);

      const v128_t vi12x0123 = wasm_v128_load(i12);
      i12 += 4;

      const v128_t vk12x0123 = wasm_v128_load(w + 52);
      vacc0123p0 = wasm_f32x4_relaxed_madd(vi12x0123, vk12x0123, vacc0123p0);

      const v128_t vi13x0123 = wasm_v128_load(i13);
      i13 += 4;

      const v128_t vk13x0123 = wasm_v128_load(w + 56);
      vacc0123p0 = wasm_f32x4_relaxed_madd(vi13x0123, vk13x0123, vacc0123p0);

      const v128_t vi14x0123 = wasm_v128_load(i14);
      i14 += 4;

      const v128_t vk14x0123 = wasm_v128_load(w + 60);
      vacc0123p0 = wasm_f32x4_relaxed_madd(vi14x0123, vk14x0123, vacc0123p0);

      const v128_t vi15x0123 = wasm_v128_load(i15);
      i15 += 4;

      const v128_t vk15x0123 = wasm_v128_load(w + 64);
      vacc0123p0 = wasm_f32x4_relaxed_madd(vi15x0123, vk15x0123, vacc0123p0);

      const v128_t vi16x0123 = wasm_v128_load(i16);
      i16 += 4;

      const v128_t vk16x0123 = wasm_v128_load(w + 68);
      vacc0123p0 = wasm_f32x4_relaxed_madd(vi16x0123, vk16x0123, vacc0123p0);

      const v128_t vi17x0123 = wasm_v128_load(i17);
      i17 += 4;

      const v128_t vk17x0123 = wasm_v128_load(w + 72);
      vacc0123p0 = wasm_f32x4_relaxed_madd(vi17x0123, vk17x0123, vacc0123p0);

      const v128_t vi18x0123 = wasm_v128_load(i18);
      i18 += 4;

      const v128_t vk18x0123 = wasm_v128_load(w + 76);
      vacc0123p0 = wasm_f32x4_relaxed_madd(vi18x0123, vk18x0123, vacc0123p0);

      const v128_t vi19x0123 = wasm_v128_load(i19);
      i19 += 4;

      const v128_t vk19x0123 = wasm_v128_load(w + 80);
      vacc0123p0 = wasm_f32x4_relaxed_madd(vi19x0123, vk19x0123, vacc0123p0);

      const v128_t vi20x0123 = wasm_v128_load(i20);
      i20 += 4;

      const v128_t vk20x0123 = wasm_v128_load(w + 84);
      vacc0123p0 = wasm_f32x4_relaxed_madd(vi20x0123, vk20x0123, vacc0123p0);

      const v128_t vi21x0123 = wasm_v128_load(i21);
      i21 += 4;

      const v128_t vk21x0123 = wasm_v128_load(w + 88);
      vacc0123p0 = wasm_f32x4_relaxed_madd(vi21x0123, vk21x0123, vacc0123p0);

      const v128_t vi22x0123 = wasm_v128_load(i22);
      i22 += 4;

      const v128_t vk22x0123 = wasm_v128_load(w + 92);
      vacc0123p0 = wasm_f32x4_relaxed_madd(vi22x0123, vk22x0123, vacc0123p0);

      const v128_t vi23x0123 = wasm_v128_load(i23);
      i23 += 4;

      const v128_t vk23x0123 = wasm_v128_load(w + 96);
      vacc0123p0 = wasm_f32x4_relaxed_madd(vi23x0123, vk23x0123, vacc0123p0);

      const v128_t vi24x0123 = wasm_v128_load(i24);
      i24 += 4;

      const v128_t vk24x0123 = wasm_v128_load(w + 100);
      vacc0123p0 = wasm_f32x4_relaxed_madd(vi24x0123, vk24x0123, vacc0123p0);

      w += 104;


      v128_t vacc0123 = wasm_f32x4_relaxed_max(vmin, vacc0123p0);

      vacc0123 = wasm_f32x4_relaxed_min(vmax, vacc0123);

      wasm_v128_store(output, vacc0123);
      output += 4;
    }
    if XNN_UNLIKELY(c != 0) {
      v128_t vacc0123p0 = wasm_v128_load(w);

      const v128_t vi0x0123 = wasm_v128_load(i0);
      const v128_t vk0x0123 = wasm_v128_load(w + 4);
      vacc0123p0 = wasm_f32x4_relaxed_madd(vi0x0123, vk0x0123, vacc0123p0);

      const v128_t vi1x0123 = wasm_v128_load(i1);
      const v128_t vk1x0123 = wasm_v128_load(w + 8);
      vacc0123p0 = wasm_f32x4_relaxed_madd(vi1x0123, vk1x0123, vacc0123p0);

      const v128_t vi2x0123 = wasm_v128_load(i2);
      const v128_t vk2x0123 = wasm_v128_load(w + 12);
      vacc0123p0 = wasm_f32x4_relaxed_madd(vi2x0123, vk2x0123, vacc0123p0);

      const v128_t vi3x0123 = wasm_v128_load(i3);
      const v128_t vk3x0123 = wasm_v128_load(w + 16);
      vacc0123p0 = wasm_f32x4_relaxed_madd(vi3x0123, vk3x0123, vacc0123p0);

      const v128_t vi4x0123 = wasm_v128_load(i4);
      const v128_t vk4x0123 = wasm_v128_load(w + 20);
      vacc0123p0 = wasm_f32x4_relaxed_madd(vi4x0123, vk4x0123, vacc0123p0);

      const v128_t vi5x0123 = wasm_v128_load(i5);
      const v128_t vk5x0123 = wasm_v128_load(w + 24);
      vacc0123p0 = wasm_f32x4_relaxed_madd(vi5x0123, vk5x0123, vacc0123p0);

      const v128_t vi6x0123 = wasm_v128_load(i6);
      const v128_t vk6x0123 = wasm_v128_load(w + 28);
      vacc0123p0 = wasm_f32x4_relaxed_madd(vi6x0123, vk6x0123, vacc0123p0);

      const v128_t vi7x0123 = wasm_v128_load(i7);
      const v128_t vk7x0123 = wasm_v128_load(w + 32);
      vacc0123p0 = wasm_f32x4_relaxed_madd(vi7x0123, vk7x0123, vacc0123p0);

      const v128_t vi8x0123 = wasm_v128_load(i8);
      const v128_t vk8x0123 = wasm_v128_load(w + 36);
      vacc0123p0 = wasm_f32x4_relaxed_madd(vi8x0123, vk8x0123, vacc0123p0);

      const v128_t vi9x0123 = wasm_v128_load(i9);
      const v128_t vk9x0123 = wasm_v128_load(w + 40);
      vacc0123p0 = wasm_f32x4_relaxed_madd(vi9x0123, vk9x0123, vacc0123p0);

      const v128_t vi10x0123 = wasm_v128_load(i10);
      const v128_t vk10x0123 = wasm_v128_load(w + 44);
      vacc0123p0 = wasm_f32x4_relaxed_madd(vi10x0123, vk10x0123, vacc0123p0);

      const v128_t vi11x0123 = wasm_v128_load(i11);
      const v128_t vk11x0123 = wasm_v128_load(w + 48);
      vacc0123p0 = wasm_f32x4_relaxed_madd(vi11x0123, vk11x0123, vacc0123p0);

      const v128_t vi12x0123 = wasm_v128_load(i12);
      const v128_t vk12x0123 = wasm_v128_load(w + 52);
      vacc0123p0 = wasm_f32x4_relaxed_madd(vi12x0123, vk12x0123, vacc0123p0);

      const v128_t vi13x0123 = wasm_v128_load(i13);
      const v128_t vk13x0123 = wasm_v128_load(w + 56);
      vacc0123p0 = wasm_f32x4_relaxed_madd(vi13x0123, vk13x0123, vacc0123p0);

      const v128_t vi14x0123 = wasm_v128_load(i14);
      const v128_t vk14x0123 = wasm_v128_load(w + 60);
      vacc0123p0 = wasm_f32x4_relaxed_madd(vi14x0123, vk14x0123, vacc0123p0);

      const v128_t vi15x0123 = wasm_v128_load(i15);
      const v128_t vk15x0123 = wasm_v128_load(w + 64);
      vacc0123p0 = wasm_f32x4_relaxed_madd(vi15x0123, vk15x0123, vacc0123p0);

      const v128_t vi16x0123 = wasm_v128_load(i16);
      const v128_t vk16x0123 = wasm_v128_load(w + 68);
      vacc0123p0 = wasm_f32x4_relaxed_madd(vi16x0123, vk16x0123, vacc0123p0);

      const v128_t vi17x0123 = wasm_v128_load(i17);
      const v128_t vk17x0123 = wasm_v128_load(w + 72);
      vacc0123p0 = wasm_f32x4_relaxed_madd(vi17x0123, vk17x0123, vacc0123p0);

      const v128_t vi18x0123 = wasm_v128_load(i18);
      const v128_t vk18x0123 = wasm_v128_load(w + 76);
      vacc0123p0 = wasm_f32x4_relaxed_madd(vi18x0123, vk18x0123, vacc0123p0);

      const v128_t vi19x0123 = wasm_v128_load(i19);
      const v128_t vk19x0123 = wasm_v128_load(w + 80);
      vacc0123p0 = wasm_f32x4_relaxed_madd(vi19x0123, vk19x0123, vacc0123p0);

      const v128_t vi20x0123 = wasm_v128_load(i20);
      const v128_t vk20x0123 = wasm_v128_load(w + 84);
      vacc0123p0 = wasm_f32x4_relaxed_madd(vi20x0123, vk20x0123, vacc0123p0);

      const v128_t vi21x0123 = wasm_v128_load(i21);
      const v128_t vk21x0123 = wasm_v128_load(w + 88);
      vacc0123p0 = wasm_f32x4_relaxed_madd(vi21x0123, vk21x0123, vacc0123p0);

      const v128_t vi22x0123 = wasm_v128_load(i22);
      const v128_t vk22x0123 = wasm_v128_load(w + 92);
      vacc0123p0 = wasm_f32x4_relaxed_madd(vi22x0123, vk22x0123, vacc0123p0);

      const v128_t vi23x0123 = wasm_v128_load(i23);
      const v128_t vk23x0123 = wasm_v128_load(w + 96);
      vacc0123p0 = wasm_f32x4_relaxed_madd(vi23x0123, vk23x0123, vacc0123p0);

      const v128_t vi24x0123 = wasm_v128_load(i24);
      const v128_t vk24x0123 = wasm_v128_load(w + 100);
      vacc0123p0 = wasm_f32x4_relaxed_madd(vi24x0123, vk24x0123, vacc0123p0);


      v128_t vacc0123 = wasm_f32x4_relaxed_max(vmin, vacc0123p0);
      vacc0123 = wasm_f32x4_relaxed_min(vmax, vacc0123);

      if (c & 2) {
        wasm_v128_store64_lane(output, vacc0123, 0);
        vacc0123 = wasm_v64x2_shuffle(vacc0123, vacc0123, 1, 1);
        output += 2;
      }
      if (c & 1) {
        wasm_v128_store32_lane(output, vacc0123, 0);
        output += 1;
      }
    }

    output = (float*) ((uintptr_t) output + output_increment);
  } while (--output_width != 0);
}
