// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/f16-dwconv/unipass-f16-f32acc.c.in
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
#include "src/xnnpack/simd/f32-wasmrelaxedsimd.h"
#undef XNN_SIMD_HAS_NATIVE_FMA
#include "src/xnnpack/simd/f16-wasmrelaxedsimd.h"


static XNN_INLINE xnn_simd_f32_t xnn_loadu_f16_f32(const xnn_float16* input) {
  return xnn_cvt_f32_f16(xnn_loadu_f16(input));
}

static XNN_INLINE xnn_simd_f32_t xnn_load_tail_f16_f32(
    const xnn_float16* input, size_t elements)
{
  return xnn_cvt_f32_f16(xnn_load_tail_f16(input, elements));
}

static XNN_INLINE void xnn_store_f32_f16(
    xnn_float16* output, xnn_simd_f32_t value)
{
  xnn_store_tail_f16(output, xnn_cvt_f16_f32(value), xnn_simd_size_f32);
}

static XNN_INLINE void xnn_store_tail_f32_f16(
    xnn_float16* output, xnn_simd_f32_t value, size_t elements)
{
  xnn_store_tail_f16(output, xnn_cvt_f16_f32(value), elements);
}


void xnn_f16_f32acc_dwconv_minmax_ukernel_25p8c__wasmrelaxedsimd_acc2(
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
    const struct xnn_f16_minmax_params* restrict params) XNN_OOB_READS
{
  assert(channels != 0);
  assert(output_width != 0);
  assert(xnn_simd_size_f32 == 4);

  const xnn_simd_f32_t vmin = xnn_set1_f32(xnn_float16_to_float(params->scalar.min));
  const xnn_simd_f32_t vmax = xnn_set1_f32(xnn_float16_to_float(params->scalar.max));
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
      xnn_simd_f32_t vacc0123p0 = xnn_loadu_f16_f32(w + 0);
      xnn_simd_f32_t vacc4567p0 = xnn_loadu_f16_f32(w + 4);

      const xnn_simd_f32_t vi0x0123 = xnn_loadu_f16_f32(i0 + 0);
      const xnn_simd_f32_t vk0x0123 = xnn_loadu_f16_f32(w + 8);
      vacc0123p0 = xnn_fmadd_f32(vi0x0123, vk0x0123, vacc0123p0);
      const xnn_simd_f32_t vi0x4567 = xnn_loadu_f16_f32(i0 + 4);
      const xnn_simd_f32_t vk0x4567 = xnn_loadu_f16_f32(w + 12);
      vacc4567p0 = xnn_fmadd_f32(vi0x4567, vk0x4567, vacc4567p0);
      i0 += 8;
      const xnn_simd_f32_t vi1x0123 = xnn_loadu_f16_f32(i1 + 0);
      const xnn_simd_f32_t vk1x0123 = xnn_loadu_f16_f32(w + 16);
      xnn_simd_f32_t vacc0123p1 = xnn_mul_f32(vi1x0123, vk1x0123);
      const xnn_simd_f32_t vi1x4567 = xnn_loadu_f16_f32(i1 + 4);
      const xnn_simd_f32_t vk1x4567 = xnn_loadu_f16_f32(w + 20);
      xnn_simd_f32_t vacc4567p1 = xnn_mul_f32(vi1x4567, vk1x4567);
      i1 += 8;
      const xnn_simd_f32_t vi2x0123 = xnn_loadu_f16_f32(i2 + 0);
      const xnn_simd_f32_t vk2x0123 = xnn_loadu_f16_f32(w + 24);
      vacc0123p0 = xnn_fmadd_f32(vi2x0123, vk2x0123, vacc0123p0);
      const xnn_simd_f32_t vi2x4567 = xnn_loadu_f16_f32(i2 + 4);
      const xnn_simd_f32_t vk2x4567 = xnn_loadu_f16_f32(w + 28);
      vacc4567p0 = xnn_fmadd_f32(vi2x4567, vk2x4567, vacc4567p0);
      i2 += 8;
      const xnn_simd_f32_t vi3x0123 = xnn_loadu_f16_f32(i3 + 0);
      const xnn_simd_f32_t vk3x0123 = xnn_loadu_f16_f32(w + 32);
      vacc0123p1 = xnn_fmadd_f32(vi3x0123, vk3x0123, vacc0123p1);
      const xnn_simd_f32_t vi3x4567 = xnn_loadu_f16_f32(i3 + 4);
      const xnn_simd_f32_t vk3x4567 = xnn_loadu_f16_f32(w + 36);
      vacc4567p1 = xnn_fmadd_f32(vi3x4567, vk3x4567, vacc4567p1);
      i3 += 8;
      const xnn_simd_f32_t vi4x0123 = xnn_loadu_f16_f32(i4 + 0);
      const xnn_simd_f32_t vk4x0123 = xnn_loadu_f16_f32(w + 40);
      vacc0123p0 = xnn_fmadd_f32(vi4x0123, vk4x0123, vacc0123p0);
      const xnn_simd_f32_t vi4x4567 = xnn_loadu_f16_f32(i4 + 4);
      const xnn_simd_f32_t vk4x4567 = xnn_loadu_f16_f32(w + 44);
      vacc4567p0 = xnn_fmadd_f32(vi4x4567, vk4x4567, vacc4567p0);
      i4 += 8;
      const xnn_simd_f32_t vi5x0123 = xnn_loadu_f16_f32(i5 + 0);
      const xnn_simd_f32_t vk5x0123 = xnn_loadu_f16_f32(w + 48);
      vacc0123p1 = xnn_fmadd_f32(vi5x0123, vk5x0123, vacc0123p1);
      const xnn_simd_f32_t vi5x4567 = xnn_loadu_f16_f32(i5 + 4);
      const xnn_simd_f32_t vk5x4567 = xnn_loadu_f16_f32(w + 52);
      vacc4567p1 = xnn_fmadd_f32(vi5x4567, vk5x4567, vacc4567p1);
      i5 += 8;
      const xnn_simd_f32_t vi6x0123 = xnn_loadu_f16_f32(i6 + 0);
      const xnn_simd_f32_t vk6x0123 = xnn_loadu_f16_f32(w + 56);
      vacc0123p0 = xnn_fmadd_f32(vi6x0123, vk6x0123, vacc0123p0);
      const xnn_simd_f32_t vi6x4567 = xnn_loadu_f16_f32(i6 + 4);
      const xnn_simd_f32_t vk6x4567 = xnn_loadu_f16_f32(w + 60);
      vacc4567p0 = xnn_fmadd_f32(vi6x4567, vk6x4567, vacc4567p0);
      i6 += 8;
      const xnn_simd_f32_t vi7x0123 = xnn_loadu_f16_f32(i7 + 0);
      const xnn_simd_f32_t vk7x0123 = xnn_loadu_f16_f32(w + 64);
      vacc0123p1 = xnn_fmadd_f32(vi7x0123, vk7x0123, vacc0123p1);
      const xnn_simd_f32_t vi7x4567 = xnn_loadu_f16_f32(i7 + 4);
      const xnn_simd_f32_t vk7x4567 = xnn_loadu_f16_f32(w + 68);
      vacc4567p1 = xnn_fmadd_f32(vi7x4567, vk7x4567, vacc4567p1);
      i7 += 8;
      const xnn_simd_f32_t vi8x0123 = xnn_loadu_f16_f32(i8 + 0);
      const xnn_simd_f32_t vk8x0123 = xnn_loadu_f16_f32(w + 72);
      vacc0123p0 = xnn_fmadd_f32(vi8x0123, vk8x0123, vacc0123p0);
      const xnn_simd_f32_t vi8x4567 = xnn_loadu_f16_f32(i8 + 4);
      const xnn_simd_f32_t vk8x4567 = xnn_loadu_f16_f32(w + 76);
      vacc4567p0 = xnn_fmadd_f32(vi8x4567, vk8x4567, vacc4567p0);
      i8 += 8;
      const xnn_simd_f32_t vi9x0123 = xnn_loadu_f16_f32(i9 + 0);
      const xnn_simd_f32_t vk9x0123 = xnn_loadu_f16_f32(w + 80);
      vacc0123p1 = xnn_fmadd_f32(vi9x0123, vk9x0123, vacc0123p1);
      const xnn_simd_f32_t vi9x4567 = xnn_loadu_f16_f32(i9 + 4);
      const xnn_simd_f32_t vk9x4567 = xnn_loadu_f16_f32(w + 84);
      vacc4567p1 = xnn_fmadd_f32(vi9x4567, vk9x4567, vacc4567p1);
      i9 += 8;
      const xnn_simd_f32_t vi10x0123 = xnn_loadu_f16_f32(i10 + 0);
      const xnn_simd_f32_t vk10x0123 = xnn_loadu_f16_f32(w + 88);
      vacc0123p0 = xnn_fmadd_f32(vi10x0123, vk10x0123, vacc0123p0);
      const xnn_simd_f32_t vi10x4567 = xnn_loadu_f16_f32(i10 + 4);
      const xnn_simd_f32_t vk10x4567 = xnn_loadu_f16_f32(w + 92);
      vacc4567p0 = xnn_fmadd_f32(vi10x4567, vk10x4567, vacc4567p0);
      i10 += 8;
      const xnn_simd_f32_t vi11x0123 = xnn_loadu_f16_f32(i11 + 0);
      const xnn_simd_f32_t vk11x0123 = xnn_loadu_f16_f32(w + 96);
      vacc0123p1 = xnn_fmadd_f32(vi11x0123, vk11x0123, vacc0123p1);
      const xnn_simd_f32_t vi11x4567 = xnn_loadu_f16_f32(i11 + 4);
      const xnn_simd_f32_t vk11x4567 = xnn_loadu_f16_f32(w + 100);
      vacc4567p1 = xnn_fmadd_f32(vi11x4567, vk11x4567, vacc4567p1);
      i11 += 8;
      const xnn_simd_f32_t vi12x0123 = xnn_loadu_f16_f32(i12 + 0);
      const xnn_simd_f32_t vk12x0123 = xnn_loadu_f16_f32(w + 104);
      vacc0123p0 = xnn_fmadd_f32(vi12x0123, vk12x0123, vacc0123p0);
      const xnn_simd_f32_t vi12x4567 = xnn_loadu_f16_f32(i12 + 4);
      const xnn_simd_f32_t vk12x4567 = xnn_loadu_f16_f32(w + 108);
      vacc4567p0 = xnn_fmadd_f32(vi12x4567, vk12x4567, vacc4567p0);
      i12 += 8;
      const xnn_simd_f32_t vi13x0123 = xnn_loadu_f16_f32(i13 + 0);
      const xnn_simd_f32_t vk13x0123 = xnn_loadu_f16_f32(w + 112);
      vacc0123p1 = xnn_fmadd_f32(vi13x0123, vk13x0123, vacc0123p1);
      const xnn_simd_f32_t vi13x4567 = xnn_loadu_f16_f32(i13 + 4);
      const xnn_simd_f32_t vk13x4567 = xnn_loadu_f16_f32(w + 116);
      vacc4567p1 = xnn_fmadd_f32(vi13x4567, vk13x4567, vacc4567p1);
      i13 += 8;
      const xnn_simd_f32_t vi14x0123 = xnn_loadu_f16_f32(i14 + 0);
      const xnn_simd_f32_t vk14x0123 = xnn_loadu_f16_f32(w + 120);
      vacc0123p0 = xnn_fmadd_f32(vi14x0123, vk14x0123, vacc0123p0);
      const xnn_simd_f32_t vi14x4567 = xnn_loadu_f16_f32(i14 + 4);
      const xnn_simd_f32_t vk14x4567 = xnn_loadu_f16_f32(w + 124);
      vacc4567p0 = xnn_fmadd_f32(vi14x4567, vk14x4567, vacc4567p0);
      i14 += 8;
      const xnn_simd_f32_t vi15x0123 = xnn_loadu_f16_f32(i15 + 0);
      const xnn_simd_f32_t vk15x0123 = xnn_loadu_f16_f32(w + 128);
      vacc0123p1 = xnn_fmadd_f32(vi15x0123, vk15x0123, vacc0123p1);
      const xnn_simd_f32_t vi15x4567 = xnn_loadu_f16_f32(i15 + 4);
      const xnn_simd_f32_t vk15x4567 = xnn_loadu_f16_f32(w + 132);
      vacc4567p1 = xnn_fmadd_f32(vi15x4567, vk15x4567, vacc4567p1);
      i15 += 8;
      const xnn_simd_f32_t vi16x0123 = xnn_loadu_f16_f32(i16 + 0);
      const xnn_simd_f32_t vk16x0123 = xnn_loadu_f16_f32(w + 136);
      vacc0123p0 = xnn_fmadd_f32(vi16x0123, vk16x0123, vacc0123p0);
      const xnn_simd_f32_t vi16x4567 = xnn_loadu_f16_f32(i16 + 4);
      const xnn_simd_f32_t vk16x4567 = xnn_loadu_f16_f32(w + 140);
      vacc4567p0 = xnn_fmadd_f32(vi16x4567, vk16x4567, vacc4567p0);
      i16 += 8;
      const xnn_simd_f32_t vi17x0123 = xnn_loadu_f16_f32(i17 + 0);
      const xnn_simd_f32_t vk17x0123 = xnn_loadu_f16_f32(w + 144);
      vacc0123p1 = xnn_fmadd_f32(vi17x0123, vk17x0123, vacc0123p1);
      const xnn_simd_f32_t vi17x4567 = xnn_loadu_f16_f32(i17 + 4);
      const xnn_simd_f32_t vk17x4567 = xnn_loadu_f16_f32(w + 148);
      vacc4567p1 = xnn_fmadd_f32(vi17x4567, vk17x4567, vacc4567p1);
      i17 += 8;
      const xnn_simd_f32_t vi18x0123 = xnn_loadu_f16_f32(i18 + 0);
      const xnn_simd_f32_t vk18x0123 = xnn_loadu_f16_f32(w + 152);
      vacc0123p0 = xnn_fmadd_f32(vi18x0123, vk18x0123, vacc0123p0);
      const xnn_simd_f32_t vi18x4567 = xnn_loadu_f16_f32(i18 + 4);
      const xnn_simd_f32_t vk18x4567 = xnn_loadu_f16_f32(w + 156);
      vacc4567p0 = xnn_fmadd_f32(vi18x4567, vk18x4567, vacc4567p0);
      i18 += 8;
      const xnn_simd_f32_t vi19x0123 = xnn_loadu_f16_f32(i19 + 0);
      const xnn_simd_f32_t vk19x0123 = xnn_loadu_f16_f32(w + 160);
      vacc0123p1 = xnn_fmadd_f32(vi19x0123, vk19x0123, vacc0123p1);
      const xnn_simd_f32_t vi19x4567 = xnn_loadu_f16_f32(i19 + 4);
      const xnn_simd_f32_t vk19x4567 = xnn_loadu_f16_f32(w + 164);
      vacc4567p1 = xnn_fmadd_f32(vi19x4567, vk19x4567, vacc4567p1);
      i19 += 8;
      const xnn_simd_f32_t vi20x0123 = xnn_loadu_f16_f32(i20 + 0);
      const xnn_simd_f32_t vk20x0123 = xnn_loadu_f16_f32(w + 168);
      vacc0123p0 = xnn_fmadd_f32(vi20x0123, vk20x0123, vacc0123p0);
      const xnn_simd_f32_t vi20x4567 = xnn_loadu_f16_f32(i20 + 4);
      const xnn_simd_f32_t vk20x4567 = xnn_loadu_f16_f32(w + 172);
      vacc4567p0 = xnn_fmadd_f32(vi20x4567, vk20x4567, vacc4567p0);
      i20 += 8;
      const xnn_simd_f32_t vi21x0123 = xnn_loadu_f16_f32(i21 + 0);
      const xnn_simd_f32_t vk21x0123 = xnn_loadu_f16_f32(w + 176);
      vacc0123p1 = xnn_fmadd_f32(vi21x0123, vk21x0123, vacc0123p1);
      const xnn_simd_f32_t vi21x4567 = xnn_loadu_f16_f32(i21 + 4);
      const xnn_simd_f32_t vk21x4567 = xnn_loadu_f16_f32(w + 180);
      vacc4567p1 = xnn_fmadd_f32(vi21x4567, vk21x4567, vacc4567p1);
      i21 += 8;
      const xnn_simd_f32_t vi22x0123 = xnn_loadu_f16_f32(i22 + 0);
      const xnn_simd_f32_t vk22x0123 = xnn_loadu_f16_f32(w + 184);
      vacc0123p0 = xnn_fmadd_f32(vi22x0123, vk22x0123, vacc0123p0);
      const xnn_simd_f32_t vi22x4567 = xnn_loadu_f16_f32(i22 + 4);
      const xnn_simd_f32_t vk22x4567 = xnn_loadu_f16_f32(w + 188);
      vacc4567p0 = xnn_fmadd_f32(vi22x4567, vk22x4567, vacc4567p0);
      i22 += 8;
      const xnn_simd_f32_t vi23x0123 = xnn_loadu_f16_f32(i23 + 0);
      const xnn_simd_f32_t vk23x0123 = xnn_loadu_f16_f32(w + 192);
      vacc0123p1 = xnn_fmadd_f32(vi23x0123, vk23x0123, vacc0123p1);
      const xnn_simd_f32_t vi23x4567 = xnn_loadu_f16_f32(i23 + 4);
      const xnn_simd_f32_t vk23x4567 = xnn_loadu_f16_f32(w + 196);
      vacc4567p1 = xnn_fmadd_f32(vi23x4567, vk23x4567, vacc4567p1);
      i23 += 8;
      const xnn_simd_f32_t vi24x0123 = xnn_loadu_f16_f32(i24 + 0);
      const xnn_simd_f32_t vk24x0123 = xnn_loadu_f16_f32(w + 200);
      vacc0123p0 = xnn_fmadd_f32(vi24x0123, vk24x0123, vacc0123p0);
      const xnn_simd_f32_t vi24x4567 = xnn_loadu_f16_f32(i24 + 4);
      const xnn_simd_f32_t vk24x4567 = xnn_loadu_f16_f32(w + 204);
      vacc4567p0 = xnn_fmadd_f32(vi24x4567, vk24x4567, vacc4567p0);
      i24 += 8;

      w += 208;

      vacc0123p0 = xnn_add_f32(vacc0123p0, vacc0123p1);
      vacc4567p0 = xnn_add_f32(vacc4567p0, vacc4567p1);

      xnn_simd_f32_t vacc0123 = xnn_max_f32(vacc0123p0, vmin);
      vacc0123 = xnn_min_f32(vacc0123, vmax);
      xnn_store_f32_f16(output + 0, vacc0123);
      xnn_simd_f32_t vacc4567 = xnn_max_f32(vacc4567p0, vmin);
      vacc4567 = xnn_min_f32(vacc4567, vmax);
      xnn_store_f32_f16(output + 4, vacc4567);
      output += 8;
    }
    for (; c >= 4; c -= 4) {
      xnn_simd_f32_t vacc0123p0 = xnn_loadu_f16_f32(w);

      const xnn_simd_f32_t vi0x0123 = xnn_loadu_f16_f32(i0);
      const xnn_simd_f32_t vk0x0123 = xnn_loadu_f16_f32(w + 8);
      vacc0123p0 = xnn_fmadd_f32(vi0x0123, vk0x0123, vacc0123p0);
      i0 += 4;
      const xnn_simd_f32_t vi1x0123 = xnn_loadu_f16_f32(i1);
      const xnn_simd_f32_t vk1x0123 = xnn_loadu_f16_f32(w + 16);
      xnn_simd_f32_t vacc0123p1 = xnn_mul_f32(vi1x0123, vk1x0123);
      i1 += 4;
      const xnn_simd_f32_t vi2x0123 = xnn_loadu_f16_f32(i2);
      const xnn_simd_f32_t vk2x0123 = xnn_loadu_f16_f32(w + 24);
      vacc0123p0 = xnn_fmadd_f32(vi2x0123, vk2x0123, vacc0123p0);
      i2 += 4;
      const xnn_simd_f32_t vi3x0123 = xnn_loadu_f16_f32(i3);
      const xnn_simd_f32_t vk3x0123 = xnn_loadu_f16_f32(w + 32);
      vacc0123p1 = xnn_fmadd_f32(vi3x0123, vk3x0123, vacc0123p1);
      i3 += 4;
      const xnn_simd_f32_t vi4x0123 = xnn_loadu_f16_f32(i4);
      const xnn_simd_f32_t vk4x0123 = xnn_loadu_f16_f32(w + 40);
      vacc0123p0 = xnn_fmadd_f32(vi4x0123, vk4x0123, vacc0123p0);
      i4 += 4;
      const xnn_simd_f32_t vi5x0123 = xnn_loadu_f16_f32(i5);
      const xnn_simd_f32_t vk5x0123 = xnn_loadu_f16_f32(w + 48);
      vacc0123p1 = xnn_fmadd_f32(vi5x0123, vk5x0123, vacc0123p1);
      i5 += 4;
      const xnn_simd_f32_t vi6x0123 = xnn_loadu_f16_f32(i6);
      const xnn_simd_f32_t vk6x0123 = xnn_loadu_f16_f32(w + 56);
      vacc0123p0 = xnn_fmadd_f32(vi6x0123, vk6x0123, vacc0123p0);
      i6 += 4;
      const xnn_simd_f32_t vi7x0123 = xnn_loadu_f16_f32(i7);
      const xnn_simd_f32_t vk7x0123 = xnn_loadu_f16_f32(w + 64);
      vacc0123p1 = xnn_fmadd_f32(vi7x0123, vk7x0123, vacc0123p1);
      i7 += 4;
      const xnn_simd_f32_t vi8x0123 = xnn_loadu_f16_f32(i8);
      const xnn_simd_f32_t vk8x0123 = xnn_loadu_f16_f32(w + 72);
      vacc0123p0 = xnn_fmadd_f32(vi8x0123, vk8x0123, vacc0123p0);
      i8 += 4;
      const xnn_simd_f32_t vi9x0123 = xnn_loadu_f16_f32(i9);
      const xnn_simd_f32_t vk9x0123 = xnn_loadu_f16_f32(w + 80);
      vacc0123p1 = xnn_fmadd_f32(vi9x0123, vk9x0123, vacc0123p1);
      i9 += 4;
      const xnn_simd_f32_t vi10x0123 = xnn_loadu_f16_f32(i10);
      const xnn_simd_f32_t vk10x0123 = xnn_loadu_f16_f32(w + 88);
      vacc0123p0 = xnn_fmadd_f32(vi10x0123, vk10x0123, vacc0123p0);
      i10 += 4;
      const xnn_simd_f32_t vi11x0123 = xnn_loadu_f16_f32(i11);
      const xnn_simd_f32_t vk11x0123 = xnn_loadu_f16_f32(w + 96);
      vacc0123p1 = xnn_fmadd_f32(vi11x0123, vk11x0123, vacc0123p1);
      i11 += 4;
      const xnn_simd_f32_t vi12x0123 = xnn_loadu_f16_f32(i12);
      const xnn_simd_f32_t vk12x0123 = xnn_loadu_f16_f32(w + 104);
      vacc0123p0 = xnn_fmadd_f32(vi12x0123, vk12x0123, vacc0123p0);
      i12 += 4;
      const xnn_simd_f32_t vi13x0123 = xnn_loadu_f16_f32(i13);
      const xnn_simd_f32_t vk13x0123 = xnn_loadu_f16_f32(w + 112);
      vacc0123p1 = xnn_fmadd_f32(vi13x0123, vk13x0123, vacc0123p1);
      i13 += 4;
      const xnn_simd_f32_t vi14x0123 = xnn_loadu_f16_f32(i14);
      const xnn_simd_f32_t vk14x0123 = xnn_loadu_f16_f32(w + 120);
      vacc0123p0 = xnn_fmadd_f32(vi14x0123, vk14x0123, vacc0123p0);
      i14 += 4;
      const xnn_simd_f32_t vi15x0123 = xnn_loadu_f16_f32(i15);
      const xnn_simd_f32_t vk15x0123 = xnn_loadu_f16_f32(w + 128);
      vacc0123p1 = xnn_fmadd_f32(vi15x0123, vk15x0123, vacc0123p1);
      i15 += 4;
      const xnn_simd_f32_t vi16x0123 = xnn_loadu_f16_f32(i16);
      const xnn_simd_f32_t vk16x0123 = xnn_loadu_f16_f32(w + 136);
      vacc0123p0 = xnn_fmadd_f32(vi16x0123, vk16x0123, vacc0123p0);
      i16 += 4;
      const xnn_simd_f32_t vi17x0123 = xnn_loadu_f16_f32(i17);
      const xnn_simd_f32_t vk17x0123 = xnn_loadu_f16_f32(w + 144);
      vacc0123p1 = xnn_fmadd_f32(vi17x0123, vk17x0123, vacc0123p1);
      i17 += 4;
      const xnn_simd_f32_t vi18x0123 = xnn_loadu_f16_f32(i18);
      const xnn_simd_f32_t vk18x0123 = xnn_loadu_f16_f32(w + 152);
      vacc0123p0 = xnn_fmadd_f32(vi18x0123, vk18x0123, vacc0123p0);
      i18 += 4;
      const xnn_simd_f32_t vi19x0123 = xnn_loadu_f16_f32(i19);
      const xnn_simd_f32_t vk19x0123 = xnn_loadu_f16_f32(w + 160);
      vacc0123p1 = xnn_fmadd_f32(vi19x0123, vk19x0123, vacc0123p1);
      i19 += 4;
      const xnn_simd_f32_t vi20x0123 = xnn_loadu_f16_f32(i20);
      const xnn_simd_f32_t vk20x0123 = xnn_loadu_f16_f32(w + 168);
      vacc0123p0 = xnn_fmadd_f32(vi20x0123, vk20x0123, vacc0123p0);
      i20 += 4;
      const xnn_simd_f32_t vi21x0123 = xnn_loadu_f16_f32(i21);
      const xnn_simd_f32_t vk21x0123 = xnn_loadu_f16_f32(w + 176);
      vacc0123p1 = xnn_fmadd_f32(vi21x0123, vk21x0123, vacc0123p1);
      i21 += 4;
      const xnn_simd_f32_t vi22x0123 = xnn_loadu_f16_f32(i22);
      const xnn_simd_f32_t vk22x0123 = xnn_loadu_f16_f32(w + 184);
      vacc0123p0 = xnn_fmadd_f32(vi22x0123, vk22x0123, vacc0123p0);
      i22 += 4;
      const xnn_simd_f32_t vi23x0123 = xnn_loadu_f16_f32(i23);
      const xnn_simd_f32_t vk23x0123 = xnn_loadu_f16_f32(w + 192);
      vacc0123p1 = xnn_fmadd_f32(vi23x0123, vk23x0123, vacc0123p1);
      i23 += 4;
      const xnn_simd_f32_t vi24x0123 = xnn_loadu_f16_f32(i24);
      const xnn_simd_f32_t vk24x0123 = xnn_loadu_f16_f32(w + 200);
      vacc0123p0 = xnn_fmadd_f32(vi24x0123, vk24x0123, vacc0123p0);
      i24 += 4;

      w += 4;

      vacc0123p0 = xnn_add_f32(vacc0123p0, vacc0123p1);

      xnn_simd_f32_t vacc0123 = xnn_max_f32(vacc0123p0, vmin);
      vacc0123 = xnn_min_f32(vacc0123, vmax);
      xnn_store_f32_f16(output, vacc0123);
      output += 4;
    }
    if XNN_UNLIKELY(c != 0) {
      xnn_simd_f32_t vacc0123p0 = xnn_loadu_f16_f32(w);

      const xnn_simd_f32_t vi0x0123 = xnn_load_tail_f16_f32(i0, c);
      const xnn_simd_f32_t vk0x0123 = xnn_loadu_f16_f32(w + 8);
      vacc0123p0 = xnn_fmadd_f32(vi0x0123, vk0x0123, vacc0123p0);
      const xnn_simd_f32_t vi1x0123 = xnn_load_tail_f16_f32(i1, c);
      const xnn_simd_f32_t vk1x0123 = xnn_loadu_f16_f32(w + 16);
      xnn_simd_f32_t vacc0123p1 = xnn_mul_f32(vi1x0123, vk1x0123);
      const xnn_simd_f32_t vi2x0123 = xnn_load_tail_f16_f32(i2, c);
      const xnn_simd_f32_t vk2x0123 = xnn_loadu_f16_f32(w + 24);
      vacc0123p0 = xnn_fmadd_f32(vi2x0123, vk2x0123, vacc0123p0);
      const xnn_simd_f32_t vi3x0123 = xnn_load_tail_f16_f32(i3, c);
      const xnn_simd_f32_t vk3x0123 = xnn_loadu_f16_f32(w + 32);
      vacc0123p1 = xnn_fmadd_f32(vi3x0123, vk3x0123, vacc0123p1);
      const xnn_simd_f32_t vi4x0123 = xnn_load_tail_f16_f32(i4, c);
      const xnn_simd_f32_t vk4x0123 = xnn_loadu_f16_f32(w + 40);
      vacc0123p0 = xnn_fmadd_f32(vi4x0123, vk4x0123, vacc0123p0);
      const xnn_simd_f32_t vi5x0123 = xnn_load_tail_f16_f32(i5, c);
      const xnn_simd_f32_t vk5x0123 = xnn_loadu_f16_f32(w + 48);
      vacc0123p1 = xnn_fmadd_f32(vi5x0123, vk5x0123, vacc0123p1);
      const xnn_simd_f32_t vi6x0123 = xnn_load_tail_f16_f32(i6, c);
      const xnn_simd_f32_t vk6x0123 = xnn_loadu_f16_f32(w + 56);
      vacc0123p0 = xnn_fmadd_f32(vi6x0123, vk6x0123, vacc0123p0);
      const xnn_simd_f32_t vi7x0123 = xnn_load_tail_f16_f32(i7, c);
      const xnn_simd_f32_t vk7x0123 = xnn_loadu_f16_f32(w + 64);
      vacc0123p1 = xnn_fmadd_f32(vi7x0123, vk7x0123, vacc0123p1);
      const xnn_simd_f32_t vi8x0123 = xnn_load_tail_f16_f32(i8, c);
      const xnn_simd_f32_t vk8x0123 = xnn_loadu_f16_f32(w + 72);
      vacc0123p0 = xnn_fmadd_f32(vi8x0123, vk8x0123, vacc0123p0);
      const xnn_simd_f32_t vi9x0123 = xnn_load_tail_f16_f32(i9, c);
      const xnn_simd_f32_t vk9x0123 = xnn_loadu_f16_f32(w + 80);
      vacc0123p1 = xnn_fmadd_f32(vi9x0123, vk9x0123, vacc0123p1);
      const xnn_simd_f32_t vi10x0123 = xnn_load_tail_f16_f32(i10, c);
      const xnn_simd_f32_t vk10x0123 = xnn_loadu_f16_f32(w + 88);
      vacc0123p0 = xnn_fmadd_f32(vi10x0123, vk10x0123, vacc0123p0);
      const xnn_simd_f32_t vi11x0123 = xnn_load_tail_f16_f32(i11, c);
      const xnn_simd_f32_t vk11x0123 = xnn_loadu_f16_f32(w + 96);
      vacc0123p1 = xnn_fmadd_f32(vi11x0123, vk11x0123, vacc0123p1);
      const xnn_simd_f32_t vi12x0123 = xnn_load_tail_f16_f32(i12, c);
      const xnn_simd_f32_t vk12x0123 = xnn_loadu_f16_f32(w + 104);
      vacc0123p0 = xnn_fmadd_f32(vi12x0123, vk12x0123, vacc0123p0);
      const xnn_simd_f32_t vi13x0123 = xnn_load_tail_f16_f32(i13, c);
      const xnn_simd_f32_t vk13x0123 = xnn_loadu_f16_f32(w + 112);
      vacc0123p1 = xnn_fmadd_f32(vi13x0123, vk13x0123, vacc0123p1);
      const xnn_simd_f32_t vi14x0123 = xnn_load_tail_f16_f32(i14, c);
      const xnn_simd_f32_t vk14x0123 = xnn_loadu_f16_f32(w + 120);
      vacc0123p0 = xnn_fmadd_f32(vi14x0123, vk14x0123, vacc0123p0);
      const xnn_simd_f32_t vi15x0123 = xnn_load_tail_f16_f32(i15, c);
      const xnn_simd_f32_t vk15x0123 = xnn_loadu_f16_f32(w + 128);
      vacc0123p1 = xnn_fmadd_f32(vi15x0123, vk15x0123, vacc0123p1);
      const xnn_simd_f32_t vi16x0123 = xnn_load_tail_f16_f32(i16, c);
      const xnn_simd_f32_t vk16x0123 = xnn_loadu_f16_f32(w + 136);
      vacc0123p0 = xnn_fmadd_f32(vi16x0123, vk16x0123, vacc0123p0);
      const xnn_simd_f32_t vi17x0123 = xnn_load_tail_f16_f32(i17, c);
      const xnn_simd_f32_t vk17x0123 = xnn_loadu_f16_f32(w + 144);
      vacc0123p1 = xnn_fmadd_f32(vi17x0123, vk17x0123, vacc0123p1);
      const xnn_simd_f32_t vi18x0123 = xnn_load_tail_f16_f32(i18, c);
      const xnn_simd_f32_t vk18x0123 = xnn_loadu_f16_f32(w + 152);
      vacc0123p0 = xnn_fmadd_f32(vi18x0123, vk18x0123, vacc0123p0);
      const xnn_simd_f32_t vi19x0123 = xnn_load_tail_f16_f32(i19, c);
      const xnn_simd_f32_t vk19x0123 = xnn_loadu_f16_f32(w + 160);
      vacc0123p1 = xnn_fmadd_f32(vi19x0123, vk19x0123, vacc0123p1);
      const xnn_simd_f32_t vi20x0123 = xnn_load_tail_f16_f32(i20, c);
      const xnn_simd_f32_t vk20x0123 = xnn_loadu_f16_f32(w + 168);
      vacc0123p0 = xnn_fmadd_f32(vi20x0123, vk20x0123, vacc0123p0);
      const xnn_simd_f32_t vi21x0123 = xnn_load_tail_f16_f32(i21, c);
      const xnn_simd_f32_t vk21x0123 = xnn_loadu_f16_f32(w + 176);
      vacc0123p1 = xnn_fmadd_f32(vi21x0123, vk21x0123, vacc0123p1);
      const xnn_simd_f32_t vi22x0123 = xnn_load_tail_f16_f32(i22, c);
      const xnn_simd_f32_t vk22x0123 = xnn_loadu_f16_f32(w + 184);
      vacc0123p0 = xnn_fmadd_f32(vi22x0123, vk22x0123, vacc0123p0);
      const xnn_simd_f32_t vi23x0123 = xnn_load_tail_f16_f32(i23, c);
      const xnn_simd_f32_t vk23x0123 = xnn_loadu_f16_f32(w + 192);
      vacc0123p1 = xnn_fmadd_f32(vi23x0123, vk23x0123, vacc0123p1);
      const xnn_simd_f32_t vi24x0123 = xnn_load_tail_f16_f32(i24, c);
      const xnn_simd_f32_t vk24x0123 = xnn_loadu_f16_f32(w + 200);
      vacc0123p0 = xnn_fmadd_f32(vi24x0123, vk24x0123, vacc0123p0);

      vacc0123p0 = xnn_add_f32(vacc0123p0, vacc0123p1);

      xnn_simd_f32_t vacc0123 = xnn_max_f32(vacc0123p0, vmin);
      vacc0123 = xnn_min_f32(vacc0123, vmax);
      xnn_store_tail_f32_f16(output, vacc0123, c);
      output += c;
    }

    input_offset += input_pixel_stride;
    output = (xnn_float16*) ((uintptr_t) output + output_increment);
  } while (--output_width != 0);
}
