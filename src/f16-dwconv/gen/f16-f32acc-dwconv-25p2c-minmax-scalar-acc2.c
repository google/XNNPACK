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
#include "src/xnnpack/simd/f32-scalar.h"
#include "src/xnnpack/simd/f16-scalar.h"


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


void xnn_f16_f32acc_dwconv_minmax_ukernel_25p2c__scalar_acc2(
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
  assert(xnn_simd_size_f32 == 1);

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
    for (; c >= 2; c -= 2) {
      xnn_simd_f32_t vacc0p0 = xnn_loadu_f16_f32(w + 0);
      xnn_simd_f32_t vacc1p0 = xnn_loadu_f16_f32(w + 1);

      const xnn_simd_f32_t vi0x0 = xnn_loadu_f16_f32(i0 + 0);
      const xnn_simd_f32_t vk0x0 = xnn_loadu_f16_f32(w + 2);
      vacc0p0 = xnn_fmadd_f32(vi0x0, vk0x0, vacc0p0);
      const xnn_simd_f32_t vi0x1 = xnn_loadu_f16_f32(i0 + 1);
      const xnn_simd_f32_t vk0x1 = xnn_loadu_f16_f32(w + 3);
      vacc1p0 = xnn_fmadd_f32(vi0x1, vk0x1, vacc1p0);
      i0 += 2;
      const xnn_simd_f32_t vi1x0 = xnn_loadu_f16_f32(i1 + 0);
      const xnn_simd_f32_t vk1x0 = xnn_loadu_f16_f32(w + 4);
      xnn_simd_f32_t vacc0p1 = xnn_mul_f32(vi1x0, vk1x0);
      const xnn_simd_f32_t vi1x1 = xnn_loadu_f16_f32(i1 + 1);
      const xnn_simd_f32_t vk1x1 = xnn_loadu_f16_f32(w + 5);
      xnn_simd_f32_t vacc1p1 = xnn_mul_f32(vi1x1, vk1x1);
      i1 += 2;
      const xnn_simd_f32_t vi2x0 = xnn_loadu_f16_f32(i2 + 0);
      const xnn_simd_f32_t vk2x0 = xnn_loadu_f16_f32(w + 6);
      vacc0p0 = xnn_fmadd_f32(vi2x0, vk2x0, vacc0p0);
      const xnn_simd_f32_t vi2x1 = xnn_loadu_f16_f32(i2 + 1);
      const xnn_simd_f32_t vk2x1 = xnn_loadu_f16_f32(w + 7);
      vacc1p0 = xnn_fmadd_f32(vi2x1, vk2x1, vacc1p0);
      i2 += 2;
      const xnn_simd_f32_t vi3x0 = xnn_loadu_f16_f32(i3 + 0);
      const xnn_simd_f32_t vk3x0 = xnn_loadu_f16_f32(w + 8);
      vacc0p1 = xnn_fmadd_f32(vi3x0, vk3x0, vacc0p1);
      const xnn_simd_f32_t vi3x1 = xnn_loadu_f16_f32(i3 + 1);
      const xnn_simd_f32_t vk3x1 = xnn_loadu_f16_f32(w + 9);
      vacc1p1 = xnn_fmadd_f32(vi3x1, vk3x1, vacc1p1);
      i3 += 2;
      const xnn_simd_f32_t vi4x0 = xnn_loadu_f16_f32(i4 + 0);
      const xnn_simd_f32_t vk4x0 = xnn_loadu_f16_f32(w + 10);
      vacc0p0 = xnn_fmadd_f32(vi4x0, vk4x0, vacc0p0);
      const xnn_simd_f32_t vi4x1 = xnn_loadu_f16_f32(i4 + 1);
      const xnn_simd_f32_t vk4x1 = xnn_loadu_f16_f32(w + 11);
      vacc1p0 = xnn_fmadd_f32(vi4x1, vk4x1, vacc1p0);
      i4 += 2;
      const xnn_simd_f32_t vi5x0 = xnn_loadu_f16_f32(i5 + 0);
      const xnn_simd_f32_t vk5x0 = xnn_loadu_f16_f32(w + 12);
      vacc0p1 = xnn_fmadd_f32(vi5x0, vk5x0, vacc0p1);
      const xnn_simd_f32_t vi5x1 = xnn_loadu_f16_f32(i5 + 1);
      const xnn_simd_f32_t vk5x1 = xnn_loadu_f16_f32(w + 13);
      vacc1p1 = xnn_fmadd_f32(vi5x1, vk5x1, vacc1p1);
      i5 += 2;
      const xnn_simd_f32_t vi6x0 = xnn_loadu_f16_f32(i6 + 0);
      const xnn_simd_f32_t vk6x0 = xnn_loadu_f16_f32(w + 14);
      vacc0p0 = xnn_fmadd_f32(vi6x0, vk6x0, vacc0p0);
      const xnn_simd_f32_t vi6x1 = xnn_loadu_f16_f32(i6 + 1);
      const xnn_simd_f32_t vk6x1 = xnn_loadu_f16_f32(w + 15);
      vacc1p0 = xnn_fmadd_f32(vi6x1, vk6x1, vacc1p0);
      i6 += 2;
      const xnn_simd_f32_t vi7x0 = xnn_loadu_f16_f32(i7 + 0);
      const xnn_simd_f32_t vk7x0 = xnn_loadu_f16_f32(w + 16);
      vacc0p1 = xnn_fmadd_f32(vi7x0, vk7x0, vacc0p1);
      const xnn_simd_f32_t vi7x1 = xnn_loadu_f16_f32(i7 + 1);
      const xnn_simd_f32_t vk7x1 = xnn_loadu_f16_f32(w + 17);
      vacc1p1 = xnn_fmadd_f32(vi7x1, vk7x1, vacc1p1);
      i7 += 2;
      const xnn_simd_f32_t vi8x0 = xnn_loadu_f16_f32(i8 + 0);
      const xnn_simd_f32_t vk8x0 = xnn_loadu_f16_f32(w + 18);
      vacc0p0 = xnn_fmadd_f32(vi8x0, vk8x0, vacc0p0);
      const xnn_simd_f32_t vi8x1 = xnn_loadu_f16_f32(i8 + 1);
      const xnn_simd_f32_t vk8x1 = xnn_loadu_f16_f32(w + 19);
      vacc1p0 = xnn_fmadd_f32(vi8x1, vk8x1, vacc1p0);
      i8 += 2;
      const xnn_simd_f32_t vi9x0 = xnn_loadu_f16_f32(i9 + 0);
      const xnn_simd_f32_t vk9x0 = xnn_loadu_f16_f32(w + 20);
      vacc0p1 = xnn_fmadd_f32(vi9x0, vk9x0, vacc0p1);
      const xnn_simd_f32_t vi9x1 = xnn_loadu_f16_f32(i9 + 1);
      const xnn_simd_f32_t vk9x1 = xnn_loadu_f16_f32(w + 21);
      vacc1p1 = xnn_fmadd_f32(vi9x1, vk9x1, vacc1p1);
      i9 += 2;
      const xnn_simd_f32_t vi10x0 = xnn_loadu_f16_f32(i10 + 0);
      const xnn_simd_f32_t vk10x0 = xnn_loadu_f16_f32(w + 22);
      vacc0p0 = xnn_fmadd_f32(vi10x0, vk10x0, vacc0p0);
      const xnn_simd_f32_t vi10x1 = xnn_loadu_f16_f32(i10 + 1);
      const xnn_simd_f32_t vk10x1 = xnn_loadu_f16_f32(w + 23);
      vacc1p0 = xnn_fmadd_f32(vi10x1, vk10x1, vacc1p0);
      i10 += 2;
      const xnn_simd_f32_t vi11x0 = xnn_loadu_f16_f32(i11 + 0);
      const xnn_simd_f32_t vk11x0 = xnn_loadu_f16_f32(w + 24);
      vacc0p1 = xnn_fmadd_f32(vi11x0, vk11x0, vacc0p1);
      const xnn_simd_f32_t vi11x1 = xnn_loadu_f16_f32(i11 + 1);
      const xnn_simd_f32_t vk11x1 = xnn_loadu_f16_f32(w + 25);
      vacc1p1 = xnn_fmadd_f32(vi11x1, vk11x1, vacc1p1);
      i11 += 2;
      const xnn_simd_f32_t vi12x0 = xnn_loadu_f16_f32(i12 + 0);
      const xnn_simd_f32_t vk12x0 = xnn_loadu_f16_f32(w + 26);
      vacc0p0 = xnn_fmadd_f32(vi12x0, vk12x0, vacc0p0);
      const xnn_simd_f32_t vi12x1 = xnn_loadu_f16_f32(i12 + 1);
      const xnn_simd_f32_t vk12x1 = xnn_loadu_f16_f32(w + 27);
      vacc1p0 = xnn_fmadd_f32(vi12x1, vk12x1, vacc1p0);
      i12 += 2;
      const xnn_simd_f32_t vi13x0 = xnn_loadu_f16_f32(i13 + 0);
      const xnn_simd_f32_t vk13x0 = xnn_loadu_f16_f32(w + 28);
      vacc0p1 = xnn_fmadd_f32(vi13x0, vk13x0, vacc0p1);
      const xnn_simd_f32_t vi13x1 = xnn_loadu_f16_f32(i13 + 1);
      const xnn_simd_f32_t vk13x1 = xnn_loadu_f16_f32(w + 29);
      vacc1p1 = xnn_fmadd_f32(vi13x1, vk13x1, vacc1p1);
      i13 += 2;
      const xnn_simd_f32_t vi14x0 = xnn_loadu_f16_f32(i14 + 0);
      const xnn_simd_f32_t vk14x0 = xnn_loadu_f16_f32(w + 30);
      vacc0p0 = xnn_fmadd_f32(vi14x0, vk14x0, vacc0p0);
      const xnn_simd_f32_t vi14x1 = xnn_loadu_f16_f32(i14 + 1);
      const xnn_simd_f32_t vk14x1 = xnn_loadu_f16_f32(w + 31);
      vacc1p0 = xnn_fmadd_f32(vi14x1, vk14x1, vacc1p0);
      i14 += 2;
      const xnn_simd_f32_t vi15x0 = xnn_loadu_f16_f32(i15 + 0);
      const xnn_simd_f32_t vk15x0 = xnn_loadu_f16_f32(w + 32);
      vacc0p1 = xnn_fmadd_f32(vi15x0, vk15x0, vacc0p1);
      const xnn_simd_f32_t vi15x1 = xnn_loadu_f16_f32(i15 + 1);
      const xnn_simd_f32_t vk15x1 = xnn_loadu_f16_f32(w + 33);
      vacc1p1 = xnn_fmadd_f32(vi15x1, vk15x1, vacc1p1);
      i15 += 2;
      const xnn_simd_f32_t vi16x0 = xnn_loadu_f16_f32(i16 + 0);
      const xnn_simd_f32_t vk16x0 = xnn_loadu_f16_f32(w + 34);
      vacc0p0 = xnn_fmadd_f32(vi16x0, vk16x0, vacc0p0);
      const xnn_simd_f32_t vi16x1 = xnn_loadu_f16_f32(i16 + 1);
      const xnn_simd_f32_t vk16x1 = xnn_loadu_f16_f32(w + 35);
      vacc1p0 = xnn_fmadd_f32(vi16x1, vk16x1, vacc1p0);
      i16 += 2;
      const xnn_simd_f32_t vi17x0 = xnn_loadu_f16_f32(i17 + 0);
      const xnn_simd_f32_t vk17x0 = xnn_loadu_f16_f32(w + 36);
      vacc0p1 = xnn_fmadd_f32(vi17x0, vk17x0, vacc0p1);
      const xnn_simd_f32_t vi17x1 = xnn_loadu_f16_f32(i17 + 1);
      const xnn_simd_f32_t vk17x1 = xnn_loadu_f16_f32(w + 37);
      vacc1p1 = xnn_fmadd_f32(vi17x1, vk17x1, vacc1p1);
      i17 += 2;
      const xnn_simd_f32_t vi18x0 = xnn_loadu_f16_f32(i18 + 0);
      const xnn_simd_f32_t vk18x0 = xnn_loadu_f16_f32(w + 38);
      vacc0p0 = xnn_fmadd_f32(vi18x0, vk18x0, vacc0p0);
      const xnn_simd_f32_t vi18x1 = xnn_loadu_f16_f32(i18 + 1);
      const xnn_simd_f32_t vk18x1 = xnn_loadu_f16_f32(w + 39);
      vacc1p0 = xnn_fmadd_f32(vi18x1, vk18x1, vacc1p0);
      i18 += 2;
      const xnn_simd_f32_t vi19x0 = xnn_loadu_f16_f32(i19 + 0);
      const xnn_simd_f32_t vk19x0 = xnn_loadu_f16_f32(w + 40);
      vacc0p1 = xnn_fmadd_f32(vi19x0, vk19x0, vacc0p1);
      const xnn_simd_f32_t vi19x1 = xnn_loadu_f16_f32(i19 + 1);
      const xnn_simd_f32_t vk19x1 = xnn_loadu_f16_f32(w + 41);
      vacc1p1 = xnn_fmadd_f32(vi19x1, vk19x1, vacc1p1);
      i19 += 2;
      const xnn_simd_f32_t vi20x0 = xnn_loadu_f16_f32(i20 + 0);
      const xnn_simd_f32_t vk20x0 = xnn_loadu_f16_f32(w + 42);
      vacc0p0 = xnn_fmadd_f32(vi20x0, vk20x0, vacc0p0);
      const xnn_simd_f32_t vi20x1 = xnn_loadu_f16_f32(i20 + 1);
      const xnn_simd_f32_t vk20x1 = xnn_loadu_f16_f32(w + 43);
      vacc1p0 = xnn_fmadd_f32(vi20x1, vk20x1, vacc1p0);
      i20 += 2;
      const xnn_simd_f32_t vi21x0 = xnn_loadu_f16_f32(i21 + 0);
      const xnn_simd_f32_t vk21x0 = xnn_loadu_f16_f32(w + 44);
      vacc0p1 = xnn_fmadd_f32(vi21x0, vk21x0, vacc0p1);
      const xnn_simd_f32_t vi21x1 = xnn_loadu_f16_f32(i21 + 1);
      const xnn_simd_f32_t vk21x1 = xnn_loadu_f16_f32(w + 45);
      vacc1p1 = xnn_fmadd_f32(vi21x1, vk21x1, vacc1p1);
      i21 += 2;
      const xnn_simd_f32_t vi22x0 = xnn_loadu_f16_f32(i22 + 0);
      const xnn_simd_f32_t vk22x0 = xnn_loadu_f16_f32(w + 46);
      vacc0p0 = xnn_fmadd_f32(vi22x0, vk22x0, vacc0p0);
      const xnn_simd_f32_t vi22x1 = xnn_loadu_f16_f32(i22 + 1);
      const xnn_simd_f32_t vk22x1 = xnn_loadu_f16_f32(w + 47);
      vacc1p0 = xnn_fmadd_f32(vi22x1, vk22x1, vacc1p0);
      i22 += 2;
      const xnn_simd_f32_t vi23x0 = xnn_loadu_f16_f32(i23 + 0);
      const xnn_simd_f32_t vk23x0 = xnn_loadu_f16_f32(w + 48);
      vacc0p1 = xnn_fmadd_f32(vi23x0, vk23x0, vacc0p1);
      const xnn_simd_f32_t vi23x1 = xnn_loadu_f16_f32(i23 + 1);
      const xnn_simd_f32_t vk23x1 = xnn_loadu_f16_f32(w + 49);
      vacc1p1 = xnn_fmadd_f32(vi23x1, vk23x1, vacc1p1);
      i23 += 2;
      const xnn_simd_f32_t vi24x0 = xnn_loadu_f16_f32(i24 + 0);
      const xnn_simd_f32_t vk24x0 = xnn_loadu_f16_f32(w + 50);
      vacc0p0 = xnn_fmadd_f32(vi24x0, vk24x0, vacc0p0);
      const xnn_simd_f32_t vi24x1 = xnn_loadu_f16_f32(i24 + 1);
      const xnn_simd_f32_t vk24x1 = xnn_loadu_f16_f32(w + 51);
      vacc1p0 = xnn_fmadd_f32(vi24x1, vk24x1, vacc1p0);
      i24 += 2;

      w += 52;

      vacc0p0 = xnn_add_f32(vacc0p0, vacc0p1);
      vacc1p0 = xnn_add_f32(vacc1p0, vacc1p1);

      xnn_simd_f32_t vacc0 = xnn_max_f32(vacc0p0, vmin);
      vacc0 = xnn_min_f32(vacc0, vmax);
      xnn_store_f32_f16(output + 0, vacc0);
      xnn_simd_f32_t vacc1 = xnn_max_f32(vacc1p0, vmin);
      vacc1 = xnn_min_f32(vacc1, vmax);
      xnn_store_f32_f16(output + 1, vacc1);
      output += 2;
    }
    for (; c >= 1; c -= 1) {
      xnn_simd_f32_t vacc0p0 = xnn_loadu_f16_f32(w);

      const xnn_simd_f32_t vi0x0 = xnn_loadu_f16_f32(i0);
      const xnn_simd_f32_t vk0x0 = xnn_loadu_f16_f32(w + 2);
      vacc0p0 = xnn_fmadd_f32(vi0x0, vk0x0, vacc0p0);
      i0 += 1;
      const xnn_simd_f32_t vi1x0 = xnn_loadu_f16_f32(i1);
      const xnn_simd_f32_t vk1x0 = xnn_loadu_f16_f32(w + 4);
      xnn_simd_f32_t vacc0p1 = xnn_mul_f32(vi1x0, vk1x0);
      i1 += 1;
      const xnn_simd_f32_t vi2x0 = xnn_loadu_f16_f32(i2);
      const xnn_simd_f32_t vk2x0 = xnn_loadu_f16_f32(w + 6);
      vacc0p0 = xnn_fmadd_f32(vi2x0, vk2x0, vacc0p0);
      i2 += 1;
      const xnn_simd_f32_t vi3x0 = xnn_loadu_f16_f32(i3);
      const xnn_simd_f32_t vk3x0 = xnn_loadu_f16_f32(w + 8);
      vacc0p1 = xnn_fmadd_f32(vi3x0, vk3x0, vacc0p1);
      i3 += 1;
      const xnn_simd_f32_t vi4x0 = xnn_loadu_f16_f32(i4);
      const xnn_simd_f32_t vk4x0 = xnn_loadu_f16_f32(w + 10);
      vacc0p0 = xnn_fmadd_f32(vi4x0, vk4x0, vacc0p0);
      i4 += 1;
      const xnn_simd_f32_t vi5x0 = xnn_loadu_f16_f32(i5);
      const xnn_simd_f32_t vk5x0 = xnn_loadu_f16_f32(w + 12);
      vacc0p1 = xnn_fmadd_f32(vi5x0, vk5x0, vacc0p1);
      i5 += 1;
      const xnn_simd_f32_t vi6x0 = xnn_loadu_f16_f32(i6);
      const xnn_simd_f32_t vk6x0 = xnn_loadu_f16_f32(w + 14);
      vacc0p0 = xnn_fmadd_f32(vi6x0, vk6x0, vacc0p0);
      i6 += 1;
      const xnn_simd_f32_t vi7x0 = xnn_loadu_f16_f32(i7);
      const xnn_simd_f32_t vk7x0 = xnn_loadu_f16_f32(w + 16);
      vacc0p1 = xnn_fmadd_f32(vi7x0, vk7x0, vacc0p1);
      i7 += 1;
      const xnn_simd_f32_t vi8x0 = xnn_loadu_f16_f32(i8);
      const xnn_simd_f32_t vk8x0 = xnn_loadu_f16_f32(w + 18);
      vacc0p0 = xnn_fmadd_f32(vi8x0, vk8x0, vacc0p0);
      i8 += 1;
      const xnn_simd_f32_t vi9x0 = xnn_loadu_f16_f32(i9);
      const xnn_simd_f32_t vk9x0 = xnn_loadu_f16_f32(w + 20);
      vacc0p1 = xnn_fmadd_f32(vi9x0, vk9x0, vacc0p1);
      i9 += 1;
      const xnn_simd_f32_t vi10x0 = xnn_loadu_f16_f32(i10);
      const xnn_simd_f32_t vk10x0 = xnn_loadu_f16_f32(w + 22);
      vacc0p0 = xnn_fmadd_f32(vi10x0, vk10x0, vacc0p0);
      i10 += 1;
      const xnn_simd_f32_t vi11x0 = xnn_loadu_f16_f32(i11);
      const xnn_simd_f32_t vk11x0 = xnn_loadu_f16_f32(w + 24);
      vacc0p1 = xnn_fmadd_f32(vi11x0, vk11x0, vacc0p1);
      i11 += 1;
      const xnn_simd_f32_t vi12x0 = xnn_loadu_f16_f32(i12);
      const xnn_simd_f32_t vk12x0 = xnn_loadu_f16_f32(w + 26);
      vacc0p0 = xnn_fmadd_f32(vi12x0, vk12x0, vacc0p0);
      i12 += 1;
      const xnn_simd_f32_t vi13x0 = xnn_loadu_f16_f32(i13);
      const xnn_simd_f32_t vk13x0 = xnn_loadu_f16_f32(w + 28);
      vacc0p1 = xnn_fmadd_f32(vi13x0, vk13x0, vacc0p1);
      i13 += 1;
      const xnn_simd_f32_t vi14x0 = xnn_loadu_f16_f32(i14);
      const xnn_simd_f32_t vk14x0 = xnn_loadu_f16_f32(w + 30);
      vacc0p0 = xnn_fmadd_f32(vi14x0, vk14x0, vacc0p0);
      i14 += 1;
      const xnn_simd_f32_t vi15x0 = xnn_loadu_f16_f32(i15);
      const xnn_simd_f32_t vk15x0 = xnn_loadu_f16_f32(w + 32);
      vacc0p1 = xnn_fmadd_f32(vi15x0, vk15x0, vacc0p1);
      i15 += 1;
      const xnn_simd_f32_t vi16x0 = xnn_loadu_f16_f32(i16);
      const xnn_simd_f32_t vk16x0 = xnn_loadu_f16_f32(w + 34);
      vacc0p0 = xnn_fmadd_f32(vi16x0, vk16x0, vacc0p0);
      i16 += 1;
      const xnn_simd_f32_t vi17x0 = xnn_loadu_f16_f32(i17);
      const xnn_simd_f32_t vk17x0 = xnn_loadu_f16_f32(w + 36);
      vacc0p1 = xnn_fmadd_f32(vi17x0, vk17x0, vacc0p1);
      i17 += 1;
      const xnn_simd_f32_t vi18x0 = xnn_loadu_f16_f32(i18);
      const xnn_simd_f32_t vk18x0 = xnn_loadu_f16_f32(w + 38);
      vacc0p0 = xnn_fmadd_f32(vi18x0, vk18x0, vacc0p0);
      i18 += 1;
      const xnn_simd_f32_t vi19x0 = xnn_loadu_f16_f32(i19);
      const xnn_simd_f32_t vk19x0 = xnn_loadu_f16_f32(w + 40);
      vacc0p1 = xnn_fmadd_f32(vi19x0, vk19x0, vacc0p1);
      i19 += 1;
      const xnn_simd_f32_t vi20x0 = xnn_loadu_f16_f32(i20);
      const xnn_simd_f32_t vk20x0 = xnn_loadu_f16_f32(w + 42);
      vacc0p0 = xnn_fmadd_f32(vi20x0, vk20x0, vacc0p0);
      i20 += 1;
      const xnn_simd_f32_t vi21x0 = xnn_loadu_f16_f32(i21);
      const xnn_simd_f32_t vk21x0 = xnn_loadu_f16_f32(w + 44);
      vacc0p1 = xnn_fmadd_f32(vi21x0, vk21x0, vacc0p1);
      i21 += 1;
      const xnn_simd_f32_t vi22x0 = xnn_loadu_f16_f32(i22);
      const xnn_simd_f32_t vk22x0 = xnn_loadu_f16_f32(w + 46);
      vacc0p0 = xnn_fmadd_f32(vi22x0, vk22x0, vacc0p0);
      i22 += 1;
      const xnn_simd_f32_t vi23x0 = xnn_loadu_f16_f32(i23);
      const xnn_simd_f32_t vk23x0 = xnn_loadu_f16_f32(w + 48);
      vacc0p1 = xnn_fmadd_f32(vi23x0, vk23x0, vacc0p1);
      i23 += 1;
      const xnn_simd_f32_t vi24x0 = xnn_loadu_f16_f32(i24);
      const xnn_simd_f32_t vk24x0 = xnn_loadu_f16_f32(w + 50);
      vacc0p0 = xnn_fmadd_f32(vi24x0, vk24x0, vacc0p0);
      i24 += 1;

      w += 1;

      vacc0p0 = xnn_add_f32(vacc0p0, vacc0p1);

      xnn_simd_f32_t vacc0 = xnn_max_f32(vacc0p0, vmin);
      vacc0 = xnn_min_f32(vacc0, vmax);
      xnn_store_f32_f16(output, vacc0);
      output += 1;
    }
    if XNN_UNLIKELY(c != 0) {
      xnn_simd_f32_t vacc0p0 = xnn_loadu_f16_f32(w);

      const xnn_simd_f32_t vi0x0 = xnn_load_tail_f16_f32(i0, c);
      const xnn_simd_f32_t vk0x0 = xnn_loadu_f16_f32(w + 2);
      vacc0p0 = xnn_fmadd_f32(vi0x0, vk0x0, vacc0p0);
      const xnn_simd_f32_t vi1x0 = xnn_load_tail_f16_f32(i1, c);
      const xnn_simd_f32_t vk1x0 = xnn_loadu_f16_f32(w + 4);
      xnn_simd_f32_t vacc0p1 = xnn_mul_f32(vi1x0, vk1x0);
      const xnn_simd_f32_t vi2x0 = xnn_load_tail_f16_f32(i2, c);
      const xnn_simd_f32_t vk2x0 = xnn_loadu_f16_f32(w + 6);
      vacc0p0 = xnn_fmadd_f32(vi2x0, vk2x0, vacc0p0);
      const xnn_simd_f32_t vi3x0 = xnn_load_tail_f16_f32(i3, c);
      const xnn_simd_f32_t vk3x0 = xnn_loadu_f16_f32(w + 8);
      vacc0p1 = xnn_fmadd_f32(vi3x0, vk3x0, vacc0p1);
      const xnn_simd_f32_t vi4x0 = xnn_load_tail_f16_f32(i4, c);
      const xnn_simd_f32_t vk4x0 = xnn_loadu_f16_f32(w + 10);
      vacc0p0 = xnn_fmadd_f32(vi4x0, vk4x0, vacc0p0);
      const xnn_simd_f32_t vi5x0 = xnn_load_tail_f16_f32(i5, c);
      const xnn_simd_f32_t vk5x0 = xnn_loadu_f16_f32(w + 12);
      vacc0p1 = xnn_fmadd_f32(vi5x0, vk5x0, vacc0p1);
      const xnn_simd_f32_t vi6x0 = xnn_load_tail_f16_f32(i6, c);
      const xnn_simd_f32_t vk6x0 = xnn_loadu_f16_f32(w + 14);
      vacc0p0 = xnn_fmadd_f32(vi6x0, vk6x0, vacc0p0);
      const xnn_simd_f32_t vi7x0 = xnn_load_tail_f16_f32(i7, c);
      const xnn_simd_f32_t vk7x0 = xnn_loadu_f16_f32(w + 16);
      vacc0p1 = xnn_fmadd_f32(vi7x0, vk7x0, vacc0p1);
      const xnn_simd_f32_t vi8x0 = xnn_load_tail_f16_f32(i8, c);
      const xnn_simd_f32_t vk8x0 = xnn_loadu_f16_f32(w + 18);
      vacc0p0 = xnn_fmadd_f32(vi8x0, vk8x0, vacc0p0);
      const xnn_simd_f32_t vi9x0 = xnn_load_tail_f16_f32(i9, c);
      const xnn_simd_f32_t vk9x0 = xnn_loadu_f16_f32(w + 20);
      vacc0p1 = xnn_fmadd_f32(vi9x0, vk9x0, vacc0p1);
      const xnn_simd_f32_t vi10x0 = xnn_load_tail_f16_f32(i10, c);
      const xnn_simd_f32_t vk10x0 = xnn_loadu_f16_f32(w + 22);
      vacc0p0 = xnn_fmadd_f32(vi10x0, vk10x0, vacc0p0);
      const xnn_simd_f32_t vi11x0 = xnn_load_tail_f16_f32(i11, c);
      const xnn_simd_f32_t vk11x0 = xnn_loadu_f16_f32(w + 24);
      vacc0p1 = xnn_fmadd_f32(vi11x0, vk11x0, vacc0p1);
      const xnn_simd_f32_t vi12x0 = xnn_load_tail_f16_f32(i12, c);
      const xnn_simd_f32_t vk12x0 = xnn_loadu_f16_f32(w + 26);
      vacc0p0 = xnn_fmadd_f32(vi12x0, vk12x0, vacc0p0);
      const xnn_simd_f32_t vi13x0 = xnn_load_tail_f16_f32(i13, c);
      const xnn_simd_f32_t vk13x0 = xnn_loadu_f16_f32(w + 28);
      vacc0p1 = xnn_fmadd_f32(vi13x0, vk13x0, vacc0p1);
      const xnn_simd_f32_t vi14x0 = xnn_load_tail_f16_f32(i14, c);
      const xnn_simd_f32_t vk14x0 = xnn_loadu_f16_f32(w + 30);
      vacc0p0 = xnn_fmadd_f32(vi14x0, vk14x0, vacc0p0);
      const xnn_simd_f32_t vi15x0 = xnn_load_tail_f16_f32(i15, c);
      const xnn_simd_f32_t vk15x0 = xnn_loadu_f16_f32(w + 32);
      vacc0p1 = xnn_fmadd_f32(vi15x0, vk15x0, vacc0p1);
      const xnn_simd_f32_t vi16x0 = xnn_load_tail_f16_f32(i16, c);
      const xnn_simd_f32_t vk16x0 = xnn_loadu_f16_f32(w + 34);
      vacc0p0 = xnn_fmadd_f32(vi16x0, vk16x0, vacc0p0);
      const xnn_simd_f32_t vi17x0 = xnn_load_tail_f16_f32(i17, c);
      const xnn_simd_f32_t vk17x0 = xnn_loadu_f16_f32(w + 36);
      vacc0p1 = xnn_fmadd_f32(vi17x0, vk17x0, vacc0p1);
      const xnn_simd_f32_t vi18x0 = xnn_load_tail_f16_f32(i18, c);
      const xnn_simd_f32_t vk18x0 = xnn_loadu_f16_f32(w + 38);
      vacc0p0 = xnn_fmadd_f32(vi18x0, vk18x0, vacc0p0);
      const xnn_simd_f32_t vi19x0 = xnn_load_tail_f16_f32(i19, c);
      const xnn_simd_f32_t vk19x0 = xnn_loadu_f16_f32(w + 40);
      vacc0p1 = xnn_fmadd_f32(vi19x0, vk19x0, vacc0p1);
      const xnn_simd_f32_t vi20x0 = xnn_load_tail_f16_f32(i20, c);
      const xnn_simd_f32_t vk20x0 = xnn_loadu_f16_f32(w + 42);
      vacc0p0 = xnn_fmadd_f32(vi20x0, vk20x0, vacc0p0);
      const xnn_simd_f32_t vi21x0 = xnn_load_tail_f16_f32(i21, c);
      const xnn_simd_f32_t vk21x0 = xnn_loadu_f16_f32(w + 44);
      vacc0p1 = xnn_fmadd_f32(vi21x0, vk21x0, vacc0p1);
      const xnn_simd_f32_t vi22x0 = xnn_load_tail_f16_f32(i22, c);
      const xnn_simd_f32_t vk22x0 = xnn_loadu_f16_f32(w + 46);
      vacc0p0 = xnn_fmadd_f32(vi22x0, vk22x0, vacc0p0);
      const xnn_simd_f32_t vi23x0 = xnn_load_tail_f16_f32(i23, c);
      const xnn_simd_f32_t vk23x0 = xnn_loadu_f16_f32(w + 48);
      vacc0p1 = xnn_fmadd_f32(vi23x0, vk23x0, vacc0p1);
      const xnn_simd_f32_t vi24x0 = xnn_load_tail_f16_f32(i24, c);
      const xnn_simd_f32_t vk24x0 = xnn_loadu_f16_f32(w + 50);
      vacc0p0 = xnn_fmadd_f32(vi24x0, vk24x0, vacc0p0);

      vacc0p0 = xnn_add_f32(vacc0p0, vacc0p1);

      xnn_simd_f32_t vacc0 = xnn_max_f32(vacc0p0, vmin);
      vacc0 = xnn_min_f32(vacc0, vmax);
      xnn_store_tail_f32_f16(output, vacc0, c);
      output += c;
    }

    input_offset += input_pixel_stride;
    output = (xnn_float16*) ((uintptr_t) output + output_increment);
  } while (--output_width != 0);
}
