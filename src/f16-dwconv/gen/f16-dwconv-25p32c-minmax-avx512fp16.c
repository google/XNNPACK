// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/f16-dwconv/unipass-avx512fp16.c.in
//   Generator: tools/xngen
//
// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include <immintrin.h>

#include "src/xnnpack/common.h"
#include "src/xnnpack/dwconv.h"
#include "src/xnnpack/math.h"
#include "src/xnnpack/microparams.h"


void xnn_f16_dwconv_minmax_ukernel_25p32c__avx512fp16(
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

#if defined(__AVX512FP16__)
  const __m512h vmin = _mm512_castsi512_ph(_mm512_set1_epi16(*(const uint16_t*) &params->scalar.min));
  const __m512h vmax = _mm512_castsi512_ph(_mm512_set1_epi16(*(const uint16_t*) &params->scalar.max));

  uint16_t* o = (uint16_t*) output;
  do {
    const uint16_t* i0 = (const uint16_t*) input[0];
    assert(i0 != NULL);
    if XNN_UNPREDICTABLE(i0 != (const uint16_t*) zero) {
      i0 = (const uint16_t*) ((uintptr_t) i0 + input_offset);
    }
    const uint16_t* i1 = (const uint16_t*) input[1];
    assert(i1 != NULL);
    if XNN_UNPREDICTABLE(i1 != (const uint16_t*) zero) {
      i1 = (const uint16_t*) ((uintptr_t) i1 + input_offset);
    }
    const uint16_t* i2 = (const uint16_t*) input[2];
    assert(i2 != NULL);
    if XNN_UNPREDICTABLE(i2 != (const uint16_t*) zero) {
      i2 = (const uint16_t*) ((uintptr_t) i2 + input_offset);
    }
    const uint16_t* i3 = (const uint16_t*) input[3];
    assert(i3 != NULL);
    if XNN_UNPREDICTABLE(i3 != (const uint16_t*) zero) {
      i3 = (const uint16_t*) ((uintptr_t) i3 + input_offset);
    }
    const uint16_t* i4 = (const uint16_t*) input[4];
    assert(i4 != NULL);
    if XNN_UNPREDICTABLE(i4 != (const uint16_t*) zero) {
      i4 = (const uint16_t*) ((uintptr_t) i4 + input_offset);
    }
    const uint16_t* i5 = (const uint16_t*) input[5];
    assert(i5 != NULL);
    if XNN_UNPREDICTABLE(i5 != (const uint16_t*) zero) {
      i5 = (const uint16_t*) ((uintptr_t) i5 + input_offset);
    }
    const uint16_t* i6 = (const uint16_t*) input[6];
    assert(i6 != NULL);
    if XNN_UNPREDICTABLE(i6 != (const uint16_t*) zero) {
      i6 = (const uint16_t*) ((uintptr_t) i6 + input_offset);
    }
    const uint16_t* i7 = (const uint16_t*) input[7];
    assert(i7 != NULL);
    if XNN_UNPREDICTABLE(i7 != (const uint16_t*) zero) {
      i7 = (const uint16_t*) ((uintptr_t) i7 + input_offset);
    }
    const uint16_t* i8 = (const uint16_t*) input[8];
    assert(i8 != NULL);
    if XNN_UNPREDICTABLE(i8 != (const uint16_t*) zero) {
      i8 = (const uint16_t*) ((uintptr_t) i8 + input_offset);
    }
    const uint16_t* i9 = (const uint16_t*) input[9];
    assert(i9 != NULL);
    if XNN_UNPREDICTABLE(i9 != (const uint16_t*) zero) {
      i9 = (const uint16_t*) ((uintptr_t) i9 + input_offset);
    }
    const uint16_t* i10 = (const uint16_t*) input[10];
    assert(i10 != NULL);
    if XNN_UNPREDICTABLE(i10 != (const uint16_t*) zero) {
      i10 = (const uint16_t*) ((uintptr_t) i10 + input_offset);
    }
    const uint16_t* i11 = (const uint16_t*) input[11];
    assert(i11 != NULL);
    if XNN_UNPREDICTABLE(i11 != (const uint16_t*) zero) {
      i11 = (const uint16_t*) ((uintptr_t) i11 + input_offset);
    }
    const uint16_t* i12 = (const uint16_t*) input[12];
    assert(i12 != NULL);
    if XNN_UNPREDICTABLE(i12 != (const uint16_t*) zero) {
      i12 = (const uint16_t*) ((uintptr_t) i12 + input_offset);
    }
    const uint16_t* i13 = (const uint16_t*) input[13];
    assert(i13 != NULL);
    if XNN_UNPREDICTABLE(i13 != (const uint16_t*) zero) {
      i13 = (const uint16_t*) ((uintptr_t) i13 + input_offset);
    }
    const uint16_t* i14 = (const uint16_t*) input[14];
    assert(i14 != NULL);
    if XNN_UNPREDICTABLE(i14 != (const uint16_t*) zero) {
      i14 = (const uint16_t*) ((uintptr_t) i14 + input_offset);
    }
    const uint16_t* i15 = (const uint16_t*) input[15];
    assert(i15 != NULL);
    if XNN_UNPREDICTABLE(i15 != (const uint16_t*) zero) {
      i15 = (const uint16_t*) ((uintptr_t) i15 + input_offset);
    }
    const uint16_t* i16 = (const uint16_t*) input[16];
    assert(i16 != NULL);
    if XNN_UNPREDICTABLE(i16 != (const uint16_t*) zero) {
      i16 = (const uint16_t*) ((uintptr_t) i16 + input_offset);
    }
    const uint16_t* i17 = (const uint16_t*) input[17];
    assert(i17 != NULL);
    if XNN_UNPREDICTABLE(i17 != (const uint16_t*) zero) {
      i17 = (const uint16_t*) ((uintptr_t) i17 + input_offset);
    }
    const uint16_t* i18 = (const uint16_t*) input[18];
    assert(i18 != NULL);
    if XNN_UNPREDICTABLE(i18 != (const uint16_t*) zero) {
      i18 = (const uint16_t*) ((uintptr_t) i18 + input_offset);
    }
    const uint16_t* i19 = (const uint16_t*) input[19];
    assert(i19 != NULL);
    if XNN_UNPREDICTABLE(i19 != (const uint16_t*) zero) {
      i19 = (const uint16_t*) ((uintptr_t) i19 + input_offset);
    }
    const uint16_t* i20 = (const uint16_t*) input[20];
    assert(i20 != NULL);
    if XNN_UNPREDICTABLE(i20 != (const uint16_t*) zero) {
      i20 = (const uint16_t*) ((uintptr_t) i20 + input_offset);
    }
    const uint16_t* i21 = (const uint16_t*) input[21];
    assert(i21 != NULL);
    if XNN_UNPREDICTABLE(i21 != (const uint16_t*) zero) {
      i21 = (const uint16_t*) ((uintptr_t) i21 + input_offset);
    }
    const uint16_t* i22 = (const uint16_t*) input[22];
    assert(i22 != NULL);
    if XNN_UNPREDICTABLE(i22 != (const uint16_t*) zero) {
      i22 = (const uint16_t*) ((uintptr_t) i22 + input_offset);
    }
    const uint16_t* i23 = (const uint16_t*) input[23];
    assert(i23 != NULL);
    if XNN_UNPREDICTABLE(i23 != (const uint16_t*) zero) {
      i23 = (const uint16_t*) ((uintptr_t) i23 + input_offset);
    }
    const uint16_t* i24 = (const uint16_t*) input[24];
    assert(i24 != NULL);
    if XNN_UNPREDICTABLE(i24 != (const uint16_t*) zero) {
      i24 = (const uint16_t*) ((uintptr_t) i24 + input_offset);
    }
    input = (const xnn_float16**) ((uintptr_t) input + input_stride);

    size_t c = channels;
    const uint16_t* w = (const uint16_t*) weights;
    for (; c >= 32; c -= 32) {
      __m512h vacc0p0 = _mm512_loadu_ph(w + 0);


      const __m512h vi0x0 = _mm512_loadu_ph(i0 + 0);
      i0 += 32;

      const __m512h vk0x0 = _mm512_loadu_ph(w + 32);
      vacc0p0 = _mm512_fmadd_ph(vi0x0, vk0x0, vacc0p0);

      const __m512h vi1x0 = _mm512_loadu_ph(i1 + 0);
      i1 += 32;

      const __m512h vk1x0 = _mm512_loadu_ph(w + 64);
      vacc0p0 = _mm512_fmadd_ph(vi1x0, vk1x0, vacc0p0);

      const __m512h vi2x0 = _mm512_loadu_ph(i2 + 0);
      i2 += 32;

      const __m512h vk2x0 = _mm512_loadu_ph(w + 96);
      vacc0p0 = _mm512_fmadd_ph(vi2x0, vk2x0, vacc0p0);

      const __m512h vi3x0 = _mm512_loadu_ph(i3 + 0);
      i3 += 32;

      const __m512h vk3x0 = _mm512_loadu_ph(w + 128);
      vacc0p0 = _mm512_fmadd_ph(vi3x0, vk3x0, vacc0p0);

      const __m512h vi4x0 = _mm512_loadu_ph(i4 + 0);
      i4 += 32;

      const __m512h vk4x0 = _mm512_loadu_ph(w + 160);
      vacc0p0 = _mm512_fmadd_ph(vi4x0, vk4x0, vacc0p0);

      const __m512h vi5x0 = _mm512_loadu_ph(i5 + 0);
      i5 += 32;

      const __m512h vk5x0 = _mm512_loadu_ph(w + 192);
      vacc0p0 = _mm512_fmadd_ph(vi5x0, vk5x0, vacc0p0);

      const __m512h vi6x0 = _mm512_loadu_ph(i6 + 0);
      i6 += 32;

      const __m512h vk6x0 = _mm512_loadu_ph(w + 224);
      vacc0p0 = _mm512_fmadd_ph(vi6x0, vk6x0, vacc0p0);

      const __m512h vi7x0 = _mm512_loadu_ph(i7 + 0);
      i7 += 32;

      const __m512h vk7x0 = _mm512_loadu_ph(w + 256);
      vacc0p0 = _mm512_fmadd_ph(vi7x0, vk7x0, vacc0p0);

      const __m512h vi8x0 = _mm512_loadu_ph(i8 + 0);
      i8 += 32;

      const __m512h vk8x0 = _mm512_loadu_ph(w + 288);
      vacc0p0 = _mm512_fmadd_ph(vi8x0, vk8x0, vacc0p0);

      const __m512h vi9x0 = _mm512_loadu_ph(i9 + 0);
      i9 += 32;

      const __m512h vk9x0 = _mm512_loadu_ph(w + 320);
      vacc0p0 = _mm512_fmadd_ph(vi9x0, vk9x0, vacc0p0);

      const __m512h vi10x0 = _mm512_loadu_ph(i10 + 0);
      i10 += 32;

      const __m512h vk10x0 = _mm512_loadu_ph(w + 352);
      vacc0p0 = _mm512_fmadd_ph(vi10x0, vk10x0, vacc0p0);

      const __m512h vi11x0 = _mm512_loadu_ph(i11 + 0);
      i11 += 32;

      const __m512h vk11x0 = _mm512_loadu_ph(w + 384);
      vacc0p0 = _mm512_fmadd_ph(vi11x0, vk11x0, vacc0p0);

      const __m512h vi12x0 = _mm512_loadu_ph(i12 + 0);
      i12 += 32;

      const __m512h vk12x0 = _mm512_loadu_ph(w + 416);
      vacc0p0 = _mm512_fmadd_ph(vi12x0, vk12x0, vacc0p0);

      const __m512h vi13x0 = _mm512_loadu_ph(i13 + 0);
      i13 += 32;

      const __m512h vk13x0 = _mm512_loadu_ph(w + 448);
      vacc0p0 = _mm512_fmadd_ph(vi13x0, vk13x0, vacc0p0);

      const __m512h vi14x0 = _mm512_loadu_ph(i14 + 0);
      i14 += 32;

      const __m512h vk14x0 = _mm512_loadu_ph(w + 480);
      vacc0p0 = _mm512_fmadd_ph(vi14x0, vk14x0, vacc0p0);

      const __m512h vi15x0 = _mm512_loadu_ph(i15 + 0);
      i15 += 32;

      const __m512h vk15x0 = _mm512_loadu_ph(w + 512);
      vacc0p0 = _mm512_fmadd_ph(vi15x0, vk15x0, vacc0p0);

      const __m512h vi16x0 = _mm512_loadu_ph(i16 + 0);
      i16 += 32;

      const __m512h vk16x0 = _mm512_loadu_ph(w + 544);
      vacc0p0 = _mm512_fmadd_ph(vi16x0, vk16x0, vacc0p0);

      const __m512h vi17x0 = _mm512_loadu_ph(i17 + 0);
      i17 += 32;

      const __m512h vk17x0 = _mm512_loadu_ph(w + 576);
      vacc0p0 = _mm512_fmadd_ph(vi17x0, vk17x0, vacc0p0);

      const __m512h vi18x0 = _mm512_loadu_ph(i18 + 0);
      i18 += 32;

      const __m512h vk18x0 = _mm512_loadu_ph(w + 608);
      vacc0p0 = _mm512_fmadd_ph(vi18x0, vk18x0, vacc0p0);

      const __m512h vi19x0 = _mm512_loadu_ph(i19 + 0);
      i19 += 32;

      const __m512h vk19x0 = _mm512_loadu_ph(w + 640);
      vacc0p0 = _mm512_fmadd_ph(vi19x0, vk19x0, vacc0p0);

      const __m512h vi20x0 = _mm512_loadu_ph(i20 + 0);
      i20 += 32;

      const __m512h vk20x0 = _mm512_loadu_ph(w + 672);
      vacc0p0 = _mm512_fmadd_ph(vi20x0, vk20x0, vacc0p0);

      const __m512h vi21x0 = _mm512_loadu_ph(i21 + 0);
      i21 += 32;

      const __m512h vk21x0 = _mm512_loadu_ph(w + 704);
      vacc0p0 = _mm512_fmadd_ph(vi21x0, vk21x0, vacc0p0);

      const __m512h vi22x0 = _mm512_loadu_ph(i22 + 0);
      i22 += 32;

      const __m512h vk22x0 = _mm512_loadu_ph(w + 736);
      vacc0p0 = _mm512_fmadd_ph(vi22x0, vk22x0, vacc0p0);

      const __m512h vi23x0 = _mm512_loadu_ph(i23 + 0);
      i23 += 32;

      const __m512h vk23x0 = _mm512_loadu_ph(w + 768);
      vacc0p0 = _mm512_fmadd_ph(vi23x0, vk23x0, vacc0p0);

      const __m512h vi24x0 = _mm512_loadu_ph(i24 + 0);
      i24 += 32;

      const __m512h vk24x0 = _mm512_loadu_ph(w + 800);
      vacc0p0 = _mm512_fmadd_ph(vi24x0, vk24x0, vacc0p0);

      w += 832;


      __m512h vacc0 = _mm512_max_ph(vmin, vacc0p0);
      vacc0 = _mm512_min_ph(vmax, vacc0);

      _mm512_storeu_ph(o + 0, vacc0);
      o += 32;
    }
    if XNN_UNLIKELY(c != 0) {
      assert(c >= 1);
      assert(c < 32);
      const __mmask32 vmask = _cvtu32_mask32((UINT32_C(1) << c) - UINT32_C(1));

      __m512h vacc0p0 = _mm512_castsi512_ph(_mm512_maskz_loadu_epi16(vmask, w));


      const __m512h vi0x0 = _mm512_castsi512_ph(_mm512_maskz_loadu_epi16(vmask, i0));

      const __m512h vk0x0 = _mm512_castsi512_ph(_mm512_maskz_loadu_epi16(vmask, w + 32));
      vacc0p0 = _mm512_fmadd_ph(vi0x0, vk0x0, vacc0p0);

      const __m512h vi1x0 = _mm512_castsi512_ph(_mm512_maskz_loadu_epi16(vmask, i1));

      const __m512h vk1x0 = _mm512_castsi512_ph(_mm512_maskz_loadu_epi16(vmask, w + 64));
      vacc0p0 = _mm512_fmadd_ph(vi1x0, vk1x0, vacc0p0);

      const __m512h vi2x0 = _mm512_castsi512_ph(_mm512_maskz_loadu_epi16(vmask, i2));

      const __m512h vk2x0 = _mm512_castsi512_ph(_mm512_maskz_loadu_epi16(vmask, w + 96));
      vacc0p0 = _mm512_fmadd_ph(vi2x0, vk2x0, vacc0p0);

      const __m512h vi3x0 = _mm512_castsi512_ph(_mm512_maskz_loadu_epi16(vmask, i3));

      const __m512h vk3x0 = _mm512_castsi512_ph(_mm512_maskz_loadu_epi16(vmask, w + 128));
      vacc0p0 = _mm512_fmadd_ph(vi3x0, vk3x0, vacc0p0);

      const __m512h vi4x0 = _mm512_castsi512_ph(_mm512_maskz_loadu_epi16(vmask, i4));

      const __m512h vk4x0 = _mm512_castsi512_ph(_mm512_maskz_loadu_epi16(vmask, w + 160));
      vacc0p0 = _mm512_fmadd_ph(vi4x0, vk4x0, vacc0p0);

      const __m512h vi5x0 = _mm512_castsi512_ph(_mm512_maskz_loadu_epi16(vmask, i5));

      const __m512h vk5x0 = _mm512_castsi512_ph(_mm512_maskz_loadu_epi16(vmask, w + 192));
      vacc0p0 = _mm512_fmadd_ph(vi5x0, vk5x0, vacc0p0);

      const __m512h vi6x0 = _mm512_castsi512_ph(_mm512_maskz_loadu_epi16(vmask, i6));

      const __m512h vk6x0 = _mm512_castsi512_ph(_mm512_maskz_loadu_epi16(vmask, w + 224));
      vacc0p0 = _mm512_fmadd_ph(vi6x0, vk6x0, vacc0p0);

      const __m512h vi7x0 = _mm512_castsi512_ph(_mm512_maskz_loadu_epi16(vmask, i7));

      const __m512h vk7x0 = _mm512_castsi512_ph(_mm512_maskz_loadu_epi16(vmask, w + 256));
      vacc0p0 = _mm512_fmadd_ph(vi7x0, vk7x0, vacc0p0);

      const __m512h vi8x0 = _mm512_castsi512_ph(_mm512_maskz_loadu_epi16(vmask, i8));

      const __m512h vk8x0 = _mm512_castsi512_ph(_mm512_maskz_loadu_epi16(vmask, w + 288));
      vacc0p0 = _mm512_fmadd_ph(vi8x0, vk8x0, vacc0p0);

      const __m512h vi9x0 = _mm512_castsi512_ph(_mm512_maskz_loadu_epi16(vmask, i9));

      const __m512h vk9x0 = _mm512_castsi512_ph(_mm512_maskz_loadu_epi16(vmask, w + 320));
      vacc0p0 = _mm512_fmadd_ph(vi9x0, vk9x0, vacc0p0);

      const __m512h vi10x0 = _mm512_castsi512_ph(_mm512_maskz_loadu_epi16(vmask, i10));

      const __m512h vk10x0 = _mm512_castsi512_ph(_mm512_maskz_loadu_epi16(vmask, w + 352));
      vacc0p0 = _mm512_fmadd_ph(vi10x0, vk10x0, vacc0p0);

      const __m512h vi11x0 = _mm512_castsi512_ph(_mm512_maskz_loadu_epi16(vmask, i11));

      const __m512h vk11x0 = _mm512_castsi512_ph(_mm512_maskz_loadu_epi16(vmask, w + 384));
      vacc0p0 = _mm512_fmadd_ph(vi11x0, vk11x0, vacc0p0);

      const __m512h vi12x0 = _mm512_castsi512_ph(_mm512_maskz_loadu_epi16(vmask, i12));

      const __m512h vk12x0 = _mm512_castsi512_ph(_mm512_maskz_loadu_epi16(vmask, w + 416));
      vacc0p0 = _mm512_fmadd_ph(vi12x0, vk12x0, vacc0p0);

      const __m512h vi13x0 = _mm512_castsi512_ph(_mm512_maskz_loadu_epi16(vmask, i13));

      const __m512h vk13x0 = _mm512_castsi512_ph(_mm512_maskz_loadu_epi16(vmask, w + 448));
      vacc0p0 = _mm512_fmadd_ph(vi13x0, vk13x0, vacc0p0);

      const __m512h vi14x0 = _mm512_castsi512_ph(_mm512_maskz_loadu_epi16(vmask, i14));

      const __m512h vk14x0 = _mm512_castsi512_ph(_mm512_maskz_loadu_epi16(vmask, w + 480));
      vacc0p0 = _mm512_fmadd_ph(vi14x0, vk14x0, vacc0p0);

      const __m512h vi15x0 = _mm512_castsi512_ph(_mm512_maskz_loadu_epi16(vmask, i15));

      const __m512h vk15x0 = _mm512_castsi512_ph(_mm512_maskz_loadu_epi16(vmask, w + 512));
      vacc0p0 = _mm512_fmadd_ph(vi15x0, vk15x0, vacc0p0);

      const __m512h vi16x0 = _mm512_castsi512_ph(_mm512_maskz_loadu_epi16(vmask, i16));

      const __m512h vk16x0 = _mm512_castsi512_ph(_mm512_maskz_loadu_epi16(vmask, w + 544));
      vacc0p0 = _mm512_fmadd_ph(vi16x0, vk16x0, vacc0p0);

      const __m512h vi17x0 = _mm512_castsi512_ph(_mm512_maskz_loadu_epi16(vmask, i17));

      const __m512h vk17x0 = _mm512_castsi512_ph(_mm512_maskz_loadu_epi16(vmask, w + 576));
      vacc0p0 = _mm512_fmadd_ph(vi17x0, vk17x0, vacc0p0);

      const __m512h vi18x0 = _mm512_castsi512_ph(_mm512_maskz_loadu_epi16(vmask, i18));

      const __m512h vk18x0 = _mm512_castsi512_ph(_mm512_maskz_loadu_epi16(vmask, w + 608));
      vacc0p0 = _mm512_fmadd_ph(vi18x0, vk18x0, vacc0p0);

      const __m512h vi19x0 = _mm512_castsi512_ph(_mm512_maskz_loadu_epi16(vmask, i19));

      const __m512h vk19x0 = _mm512_castsi512_ph(_mm512_maskz_loadu_epi16(vmask, w + 640));
      vacc0p0 = _mm512_fmadd_ph(vi19x0, vk19x0, vacc0p0);

      const __m512h vi20x0 = _mm512_castsi512_ph(_mm512_maskz_loadu_epi16(vmask, i20));

      const __m512h vk20x0 = _mm512_castsi512_ph(_mm512_maskz_loadu_epi16(vmask, w + 672));
      vacc0p0 = _mm512_fmadd_ph(vi20x0, vk20x0, vacc0p0);

      const __m512h vi21x0 = _mm512_castsi512_ph(_mm512_maskz_loadu_epi16(vmask, i21));

      const __m512h vk21x0 = _mm512_castsi512_ph(_mm512_maskz_loadu_epi16(vmask, w + 704));
      vacc0p0 = _mm512_fmadd_ph(vi21x0, vk21x0, vacc0p0);

      const __m512h vi22x0 = _mm512_castsi512_ph(_mm512_maskz_loadu_epi16(vmask, i22));

      const __m512h vk22x0 = _mm512_castsi512_ph(_mm512_maskz_loadu_epi16(vmask, w + 736));
      vacc0p0 = _mm512_fmadd_ph(vi22x0, vk22x0, vacc0p0);

      const __m512h vi23x0 = _mm512_castsi512_ph(_mm512_maskz_loadu_epi16(vmask, i23));

      const __m512h vk23x0 = _mm512_castsi512_ph(_mm512_maskz_loadu_epi16(vmask, w + 768));
      vacc0p0 = _mm512_fmadd_ph(vi23x0, vk23x0, vacc0p0);

      const __m512h vi24x0 = _mm512_castsi512_ph(_mm512_maskz_loadu_epi16(vmask, i24));

      const __m512h vk24x0 = _mm512_castsi512_ph(_mm512_maskz_loadu_epi16(vmask, w + 800));
      vacc0p0 = _mm512_fmadd_ph(vi24x0, vk24x0, vacc0p0);


      __m512h vacc0 = _mm512_max_ph(vmin, vacc0p0);
      vacc0 = _mm512_min_ph(vmax, vacc0);

      _mm512_mask_storeu_epi16(o, vmask, _mm512_castph_si512(vacc0));
      o += c;
    }

    input_offset += input_pixel_stride;
    o = (uint16_t*) ((uintptr_t) o + output_increment);
  } while (--output_width != 0);
#endif  // defined(__AVX512FP16__)
}
