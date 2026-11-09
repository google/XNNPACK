// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/bf16-f32-dwconv/unipass-simd.c.in
//   Generator: tools/xngen
//
// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <immintrin.h>
#include <stddef.h>
#include <stdint.h>

#include "src/xnnpack/common.h"
#include "src/xnnpack/dwconv.h"
#include "src/xnnpack/intrinsics-polyfill.h"
#include "src/xnnpack/microparams.h"


static XNN_INLINE __m512 load_bf16_as_f32(const uint16_t* ptr) {
  return _mm512_castsi512_ps(_mm512_slli_epi32(
      _mm512_cvtepu16_epi32(_mm256_loadu_si256((const __m256i*) ptr)), 16));
}

static XNN_INLINE __m512 maskz_load_bf16_as_f32(__mmask16 mask,
                                                const uint16_t* ptr) {
  return _mm512_castsi512_ps(_mm512_slli_epi32(
      _mm512_cvtepu16_epi32(_mm256_maskz_loadu_epi16(mask, ptr)), 16));
}

// Packed weights layout, per block of 32 channels:
//   float bias[32];
//   uint16_t kernel[25][32];  // bf16
void xnn_bf16_f32_dwconv_minmax_ukernel_25p32c__avx512skx_acc2(
    size_t channels,
    size_t output_width,
    const xnn_bfloat16** input,
    const void* weights,
    float* output,
    intptr_t input_stride,
    size_t output_increment,
    size_t input_offset,
    size_t input_pixel_stride,
    const xnn_bfloat16* zero,
    const struct xnn_f32_minmax_params* restrict params)
{
  assert(channels != 0);
  assert(output_width != 0);

  const __m512 vmin = _mm512_set1_ps(params->scalar.min);
  const __m512 vmax = _mm512_set1_ps(params->scalar.max);
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
    input = (const xnn_bfloat16**) ((uintptr_t) input + input_stride);

    size_t c = channels;
    const float* wb = (const float*) weights;
    for (; c >= 32; c -= 32) {
      const uint16_t* wk = (const uint16_t*) (wb + 32);
      __m512 vacc0p0 = _mm512_loadu_ps(wb + 0);
      __m512 vacc16p0 = _mm512_loadu_ps(wb + 16);


      const __m512 vi0x0 = load_bf16_as_f32(i0 + 0);
      const __m512 vi0x16 = load_bf16_as_f32(i0 + 16);
      i0 += 32;

      const __m512 vk0x0 = load_bf16_as_f32(wk + 0);
      const __m512 vk0x16 = load_bf16_as_f32(wk + 16);
      vacc0p0 = _mm512_fmadd_ps(vi0x0, vk0x0, vacc0p0);
      vacc16p0 = _mm512_fmadd_ps(vi0x16, vk0x16, vacc16p0);

      const __m512 vi1x0 = load_bf16_as_f32(i1 + 0);
      const __m512 vi1x16 = load_bf16_as_f32(i1 + 16);
      i1 += 32;

      const __m512 vk1x0 = load_bf16_as_f32(wk + 32);
      const __m512 vk1x16 = load_bf16_as_f32(wk + 48);
      __m512 vacc0p1 = _mm512_mul_ps(vi1x0, vk1x0);
      __m512 vacc16p1 = _mm512_mul_ps(vi1x16, vk1x16);

      const __m512 vi2x0 = load_bf16_as_f32(i2 + 0);
      const __m512 vi2x16 = load_bf16_as_f32(i2 + 16);
      i2 += 32;

      const __m512 vk2x0 = load_bf16_as_f32(wk + 64);
      const __m512 vk2x16 = load_bf16_as_f32(wk + 80);
      vacc0p0 = _mm512_fmadd_ps(vi2x0, vk2x0, vacc0p0);
      vacc16p0 = _mm512_fmadd_ps(vi2x16, vk2x16, vacc16p0);

      const __m512 vi3x0 = load_bf16_as_f32(i3 + 0);
      const __m512 vi3x16 = load_bf16_as_f32(i3 + 16);
      i3 += 32;

      const __m512 vk3x0 = load_bf16_as_f32(wk + 96);
      const __m512 vk3x16 = load_bf16_as_f32(wk + 112);
      vacc0p1 = _mm512_fmadd_ps(vi3x0, vk3x0, vacc0p1);
      vacc16p1 = _mm512_fmadd_ps(vi3x16, vk3x16, vacc16p1);

      const __m512 vi4x0 = load_bf16_as_f32(i4 + 0);
      const __m512 vi4x16 = load_bf16_as_f32(i4 + 16);
      i4 += 32;

      const __m512 vk4x0 = load_bf16_as_f32(wk + 128);
      const __m512 vk4x16 = load_bf16_as_f32(wk + 144);
      vacc0p0 = _mm512_fmadd_ps(vi4x0, vk4x0, vacc0p0);
      vacc16p0 = _mm512_fmadd_ps(vi4x16, vk4x16, vacc16p0);

      const __m512 vi5x0 = load_bf16_as_f32(i5 + 0);
      const __m512 vi5x16 = load_bf16_as_f32(i5 + 16);
      i5 += 32;

      const __m512 vk5x0 = load_bf16_as_f32(wk + 160);
      const __m512 vk5x16 = load_bf16_as_f32(wk + 176);
      vacc0p1 = _mm512_fmadd_ps(vi5x0, vk5x0, vacc0p1);
      vacc16p1 = _mm512_fmadd_ps(vi5x16, vk5x16, vacc16p1);

      const __m512 vi6x0 = load_bf16_as_f32(i6 + 0);
      const __m512 vi6x16 = load_bf16_as_f32(i6 + 16);
      i6 += 32;

      const __m512 vk6x0 = load_bf16_as_f32(wk + 192);
      const __m512 vk6x16 = load_bf16_as_f32(wk + 208);
      vacc0p0 = _mm512_fmadd_ps(vi6x0, vk6x0, vacc0p0);
      vacc16p0 = _mm512_fmadd_ps(vi6x16, vk6x16, vacc16p0);

      const __m512 vi7x0 = load_bf16_as_f32(i7 + 0);
      const __m512 vi7x16 = load_bf16_as_f32(i7 + 16);
      i7 += 32;

      const __m512 vk7x0 = load_bf16_as_f32(wk + 224);
      const __m512 vk7x16 = load_bf16_as_f32(wk + 240);
      vacc0p1 = _mm512_fmadd_ps(vi7x0, vk7x0, vacc0p1);
      vacc16p1 = _mm512_fmadd_ps(vi7x16, vk7x16, vacc16p1);

      const __m512 vi8x0 = load_bf16_as_f32(i8 + 0);
      const __m512 vi8x16 = load_bf16_as_f32(i8 + 16);
      i8 += 32;

      const __m512 vk8x0 = load_bf16_as_f32(wk + 256);
      const __m512 vk8x16 = load_bf16_as_f32(wk + 272);
      vacc0p0 = _mm512_fmadd_ps(vi8x0, vk8x0, vacc0p0);
      vacc16p0 = _mm512_fmadd_ps(vi8x16, vk8x16, vacc16p0);

      const __m512 vi9x0 = load_bf16_as_f32(i9 + 0);
      const __m512 vi9x16 = load_bf16_as_f32(i9 + 16);
      i9 += 32;

      const __m512 vk9x0 = load_bf16_as_f32(wk + 288);
      const __m512 vk9x16 = load_bf16_as_f32(wk + 304);
      vacc0p1 = _mm512_fmadd_ps(vi9x0, vk9x0, vacc0p1);
      vacc16p1 = _mm512_fmadd_ps(vi9x16, vk9x16, vacc16p1);

      const __m512 vi10x0 = load_bf16_as_f32(i10 + 0);
      const __m512 vi10x16 = load_bf16_as_f32(i10 + 16);
      i10 += 32;

      const __m512 vk10x0 = load_bf16_as_f32(wk + 320);
      const __m512 vk10x16 = load_bf16_as_f32(wk + 336);
      vacc0p0 = _mm512_fmadd_ps(vi10x0, vk10x0, vacc0p0);
      vacc16p0 = _mm512_fmadd_ps(vi10x16, vk10x16, vacc16p0);

      const __m512 vi11x0 = load_bf16_as_f32(i11 + 0);
      const __m512 vi11x16 = load_bf16_as_f32(i11 + 16);
      i11 += 32;

      const __m512 vk11x0 = load_bf16_as_f32(wk + 352);
      const __m512 vk11x16 = load_bf16_as_f32(wk + 368);
      vacc0p1 = _mm512_fmadd_ps(vi11x0, vk11x0, vacc0p1);
      vacc16p1 = _mm512_fmadd_ps(vi11x16, vk11x16, vacc16p1);

      const __m512 vi12x0 = load_bf16_as_f32(i12 + 0);
      const __m512 vi12x16 = load_bf16_as_f32(i12 + 16);
      i12 += 32;

      const __m512 vk12x0 = load_bf16_as_f32(wk + 384);
      const __m512 vk12x16 = load_bf16_as_f32(wk + 400);
      vacc0p0 = _mm512_fmadd_ps(vi12x0, vk12x0, vacc0p0);
      vacc16p0 = _mm512_fmadd_ps(vi12x16, vk12x16, vacc16p0);

      const __m512 vi13x0 = load_bf16_as_f32(i13 + 0);
      const __m512 vi13x16 = load_bf16_as_f32(i13 + 16);
      i13 += 32;

      const __m512 vk13x0 = load_bf16_as_f32(wk + 416);
      const __m512 vk13x16 = load_bf16_as_f32(wk + 432);
      vacc0p1 = _mm512_fmadd_ps(vi13x0, vk13x0, vacc0p1);
      vacc16p1 = _mm512_fmadd_ps(vi13x16, vk13x16, vacc16p1);

      const __m512 vi14x0 = load_bf16_as_f32(i14 + 0);
      const __m512 vi14x16 = load_bf16_as_f32(i14 + 16);
      i14 += 32;

      const __m512 vk14x0 = load_bf16_as_f32(wk + 448);
      const __m512 vk14x16 = load_bf16_as_f32(wk + 464);
      vacc0p0 = _mm512_fmadd_ps(vi14x0, vk14x0, vacc0p0);
      vacc16p0 = _mm512_fmadd_ps(vi14x16, vk14x16, vacc16p0);

      const __m512 vi15x0 = load_bf16_as_f32(i15 + 0);
      const __m512 vi15x16 = load_bf16_as_f32(i15 + 16);
      i15 += 32;

      const __m512 vk15x0 = load_bf16_as_f32(wk + 480);
      const __m512 vk15x16 = load_bf16_as_f32(wk + 496);
      vacc0p1 = _mm512_fmadd_ps(vi15x0, vk15x0, vacc0p1);
      vacc16p1 = _mm512_fmadd_ps(vi15x16, vk15x16, vacc16p1);

      const __m512 vi16x0 = load_bf16_as_f32(i16 + 0);
      const __m512 vi16x16 = load_bf16_as_f32(i16 + 16);
      i16 += 32;

      const __m512 vk16x0 = load_bf16_as_f32(wk + 512);
      const __m512 vk16x16 = load_bf16_as_f32(wk + 528);
      vacc0p0 = _mm512_fmadd_ps(vi16x0, vk16x0, vacc0p0);
      vacc16p0 = _mm512_fmadd_ps(vi16x16, vk16x16, vacc16p0);

      const __m512 vi17x0 = load_bf16_as_f32(i17 + 0);
      const __m512 vi17x16 = load_bf16_as_f32(i17 + 16);
      i17 += 32;

      const __m512 vk17x0 = load_bf16_as_f32(wk + 544);
      const __m512 vk17x16 = load_bf16_as_f32(wk + 560);
      vacc0p1 = _mm512_fmadd_ps(vi17x0, vk17x0, vacc0p1);
      vacc16p1 = _mm512_fmadd_ps(vi17x16, vk17x16, vacc16p1);

      const __m512 vi18x0 = load_bf16_as_f32(i18 + 0);
      const __m512 vi18x16 = load_bf16_as_f32(i18 + 16);
      i18 += 32;

      const __m512 vk18x0 = load_bf16_as_f32(wk + 576);
      const __m512 vk18x16 = load_bf16_as_f32(wk + 592);
      vacc0p0 = _mm512_fmadd_ps(vi18x0, vk18x0, vacc0p0);
      vacc16p0 = _mm512_fmadd_ps(vi18x16, vk18x16, vacc16p0);

      const __m512 vi19x0 = load_bf16_as_f32(i19 + 0);
      const __m512 vi19x16 = load_bf16_as_f32(i19 + 16);
      i19 += 32;

      const __m512 vk19x0 = load_bf16_as_f32(wk + 608);
      const __m512 vk19x16 = load_bf16_as_f32(wk + 624);
      vacc0p1 = _mm512_fmadd_ps(vi19x0, vk19x0, vacc0p1);
      vacc16p1 = _mm512_fmadd_ps(vi19x16, vk19x16, vacc16p1);

      const __m512 vi20x0 = load_bf16_as_f32(i20 + 0);
      const __m512 vi20x16 = load_bf16_as_f32(i20 + 16);
      i20 += 32;

      const __m512 vk20x0 = load_bf16_as_f32(wk + 640);
      const __m512 vk20x16 = load_bf16_as_f32(wk + 656);
      vacc0p0 = _mm512_fmadd_ps(vi20x0, vk20x0, vacc0p0);
      vacc16p0 = _mm512_fmadd_ps(vi20x16, vk20x16, vacc16p0);

      const __m512 vi21x0 = load_bf16_as_f32(i21 + 0);
      const __m512 vi21x16 = load_bf16_as_f32(i21 + 16);
      i21 += 32;

      const __m512 vk21x0 = load_bf16_as_f32(wk + 672);
      const __m512 vk21x16 = load_bf16_as_f32(wk + 688);
      vacc0p1 = _mm512_fmadd_ps(vi21x0, vk21x0, vacc0p1);
      vacc16p1 = _mm512_fmadd_ps(vi21x16, vk21x16, vacc16p1);

      const __m512 vi22x0 = load_bf16_as_f32(i22 + 0);
      const __m512 vi22x16 = load_bf16_as_f32(i22 + 16);
      i22 += 32;

      const __m512 vk22x0 = load_bf16_as_f32(wk + 704);
      const __m512 vk22x16 = load_bf16_as_f32(wk + 720);
      vacc0p0 = _mm512_fmadd_ps(vi22x0, vk22x0, vacc0p0);
      vacc16p0 = _mm512_fmadd_ps(vi22x16, vk22x16, vacc16p0);

      const __m512 vi23x0 = load_bf16_as_f32(i23 + 0);
      const __m512 vi23x16 = load_bf16_as_f32(i23 + 16);
      i23 += 32;

      const __m512 vk23x0 = load_bf16_as_f32(wk + 736);
      const __m512 vk23x16 = load_bf16_as_f32(wk + 752);
      vacc0p1 = _mm512_fmadd_ps(vi23x0, vk23x0, vacc0p1);
      vacc16p1 = _mm512_fmadd_ps(vi23x16, vk23x16, vacc16p1);

      const __m512 vi24x0 = load_bf16_as_f32(i24 + 0);
      const __m512 vi24x16 = load_bf16_as_f32(i24 + 16);
      i24 += 32;

      const __m512 vk24x0 = load_bf16_as_f32(wk + 768);
      const __m512 vk24x16 = load_bf16_as_f32(wk + 784);
      vacc0p0 = _mm512_fmadd_ps(vi24x0, vk24x0, vacc0p0);
      vacc16p0 = _mm512_fmadd_ps(vi24x16, vk24x16, vacc16p0);

      wb = (const float*) (wk + 800);

      vacc0p0 = _mm512_add_ps(vacc0p0, vacc0p1);
      vacc16p0 = _mm512_add_ps(vacc16p0, vacc16p1);

      __m512 vacc0 = _mm512_max_ps(vmin, vacc0p0);
      __m512 vacc16 = _mm512_max_ps(vmin, vacc16p0);
      vacc0 = _mm512_min_ps(vmax, vacc0);
      vacc16 = _mm512_min_ps(vmax, vacc16);

      _mm512_storeu_ps(output + 0, vacc0);
      _mm512_storeu_ps(output + 16, vacc16);
      output += 32;
    }
    if XNN_UNLIKELY(c != 0) {
      const uint16_t* wk = (const uint16_t*) (wb + 32);
      for (; c >= 16; c -= 16) {
        __m512 vacc0p0 = _mm512_loadu_ps(wb);

        const __m512 vi0x0 = load_bf16_as_f32(i0);
        i0 += 16;

        const __m512 vk0x0 = load_bf16_as_f32(wk + 0);
        vacc0p0 = _mm512_fmadd_ps(vi0x0, vk0x0, vacc0p0);

        const __m512 vi1x0 = load_bf16_as_f32(i1);
        i1 += 16;

        const __m512 vk1x0 = load_bf16_as_f32(wk + 32);
        __m512 vacc0p1 = _mm512_mul_ps(vi1x0, vk1x0);

        const __m512 vi2x0 = load_bf16_as_f32(i2);
        i2 += 16;

        const __m512 vk2x0 = load_bf16_as_f32(wk + 64);
        vacc0p0 = _mm512_fmadd_ps(vi2x0, vk2x0, vacc0p0);

        const __m512 vi3x0 = load_bf16_as_f32(i3);
        i3 += 16;

        const __m512 vk3x0 = load_bf16_as_f32(wk + 96);
        vacc0p1 = _mm512_fmadd_ps(vi3x0, vk3x0, vacc0p1);

        const __m512 vi4x0 = load_bf16_as_f32(i4);
        i4 += 16;

        const __m512 vk4x0 = load_bf16_as_f32(wk + 128);
        vacc0p0 = _mm512_fmadd_ps(vi4x0, vk4x0, vacc0p0);

        const __m512 vi5x0 = load_bf16_as_f32(i5);
        i5 += 16;

        const __m512 vk5x0 = load_bf16_as_f32(wk + 160);
        vacc0p1 = _mm512_fmadd_ps(vi5x0, vk5x0, vacc0p1);

        const __m512 vi6x0 = load_bf16_as_f32(i6);
        i6 += 16;

        const __m512 vk6x0 = load_bf16_as_f32(wk + 192);
        vacc0p0 = _mm512_fmadd_ps(vi6x0, vk6x0, vacc0p0);

        const __m512 vi7x0 = load_bf16_as_f32(i7);
        i7 += 16;

        const __m512 vk7x0 = load_bf16_as_f32(wk + 224);
        vacc0p1 = _mm512_fmadd_ps(vi7x0, vk7x0, vacc0p1);

        const __m512 vi8x0 = load_bf16_as_f32(i8);
        i8 += 16;

        const __m512 vk8x0 = load_bf16_as_f32(wk + 256);
        vacc0p0 = _mm512_fmadd_ps(vi8x0, vk8x0, vacc0p0);

        const __m512 vi9x0 = load_bf16_as_f32(i9);
        i9 += 16;

        const __m512 vk9x0 = load_bf16_as_f32(wk + 288);
        vacc0p1 = _mm512_fmadd_ps(vi9x0, vk9x0, vacc0p1);

        const __m512 vi10x0 = load_bf16_as_f32(i10);
        i10 += 16;

        const __m512 vk10x0 = load_bf16_as_f32(wk + 320);
        vacc0p0 = _mm512_fmadd_ps(vi10x0, vk10x0, vacc0p0);

        const __m512 vi11x0 = load_bf16_as_f32(i11);
        i11 += 16;

        const __m512 vk11x0 = load_bf16_as_f32(wk + 352);
        vacc0p1 = _mm512_fmadd_ps(vi11x0, vk11x0, vacc0p1);

        const __m512 vi12x0 = load_bf16_as_f32(i12);
        i12 += 16;

        const __m512 vk12x0 = load_bf16_as_f32(wk + 384);
        vacc0p0 = _mm512_fmadd_ps(vi12x0, vk12x0, vacc0p0);

        const __m512 vi13x0 = load_bf16_as_f32(i13);
        i13 += 16;

        const __m512 vk13x0 = load_bf16_as_f32(wk + 416);
        vacc0p1 = _mm512_fmadd_ps(vi13x0, vk13x0, vacc0p1);

        const __m512 vi14x0 = load_bf16_as_f32(i14);
        i14 += 16;

        const __m512 vk14x0 = load_bf16_as_f32(wk + 448);
        vacc0p0 = _mm512_fmadd_ps(vi14x0, vk14x0, vacc0p0);

        const __m512 vi15x0 = load_bf16_as_f32(i15);
        i15 += 16;

        const __m512 vk15x0 = load_bf16_as_f32(wk + 480);
        vacc0p1 = _mm512_fmadd_ps(vi15x0, vk15x0, vacc0p1);

        const __m512 vi16x0 = load_bf16_as_f32(i16);
        i16 += 16;

        const __m512 vk16x0 = load_bf16_as_f32(wk + 512);
        vacc0p0 = _mm512_fmadd_ps(vi16x0, vk16x0, vacc0p0);

        const __m512 vi17x0 = load_bf16_as_f32(i17);
        i17 += 16;

        const __m512 vk17x0 = load_bf16_as_f32(wk + 544);
        vacc0p1 = _mm512_fmadd_ps(vi17x0, vk17x0, vacc0p1);

        const __m512 vi18x0 = load_bf16_as_f32(i18);
        i18 += 16;

        const __m512 vk18x0 = load_bf16_as_f32(wk + 576);
        vacc0p0 = _mm512_fmadd_ps(vi18x0, vk18x0, vacc0p0);

        const __m512 vi19x0 = load_bf16_as_f32(i19);
        i19 += 16;

        const __m512 vk19x0 = load_bf16_as_f32(wk + 608);
        vacc0p1 = _mm512_fmadd_ps(vi19x0, vk19x0, vacc0p1);

        const __m512 vi20x0 = load_bf16_as_f32(i20);
        i20 += 16;

        const __m512 vk20x0 = load_bf16_as_f32(wk + 640);
        vacc0p0 = _mm512_fmadd_ps(vi20x0, vk20x0, vacc0p0);

        const __m512 vi21x0 = load_bf16_as_f32(i21);
        i21 += 16;

        const __m512 vk21x0 = load_bf16_as_f32(wk + 672);
        vacc0p1 = _mm512_fmadd_ps(vi21x0, vk21x0, vacc0p1);

        const __m512 vi22x0 = load_bf16_as_f32(i22);
        i22 += 16;

        const __m512 vk22x0 = load_bf16_as_f32(wk + 704);
        vacc0p0 = _mm512_fmadd_ps(vi22x0, vk22x0, vacc0p0);

        const __m512 vi23x0 = load_bf16_as_f32(i23);
        i23 += 16;

        const __m512 vk23x0 = load_bf16_as_f32(wk + 736);
        vacc0p1 = _mm512_fmadd_ps(vi23x0, vk23x0, vacc0p1);

        const __m512 vi24x0 = load_bf16_as_f32(i24);
        i24 += 16;

        const __m512 vk24x0 = load_bf16_as_f32(wk + 768);
        vacc0p0 = _mm512_fmadd_ps(vi24x0, vk24x0, vacc0p0);

        wb += 16;
        wk += 16;

        vacc0p0 = _mm512_add_ps(vacc0p0, vacc0p1);

        __m512 vacc0 = _mm512_max_ps(vmin, vacc0p0);
        vacc0 = _mm512_min_ps(vmax, vacc0);

        _mm512_storeu_ps(output, vacc0);
        output += 16;
      }
      if XNN_LIKELY(c != 0) {
        const __mmask16 vmask = _cvtu32_mask16((uint32_t) ((UINT32_C(1) << c) - 1));
        __m512 vacc0p0 = _mm512_maskz_loadu_ps(vmask, wb);

        const __m512 vi0x0 = maskz_load_bf16_as_f32(vmask, i0);
        const __m512 vk0x0 = maskz_load_bf16_as_f32(vmask, wk + 0);
        vacc0p0 = _mm512_fmadd_ps(vi0x0, vk0x0, vacc0p0);

        const __m512 vi1x0 = maskz_load_bf16_as_f32(vmask, i1);
        const __m512 vk1x0 = maskz_load_bf16_as_f32(vmask, wk + 32);
        __m512 vacc0p1 = _mm512_mul_ps(vi1x0, vk1x0);

        const __m512 vi2x0 = maskz_load_bf16_as_f32(vmask, i2);
        const __m512 vk2x0 = maskz_load_bf16_as_f32(vmask, wk + 64);
        vacc0p0 = _mm512_fmadd_ps(vi2x0, vk2x0, vacc0p0);

        const __m512 vi3x0 = maskz_load_bf16_as_f32(vmask, i3);
        const __m512 vk3x0 = maskz_load_bf16_as_f32(vmask, wk + 96);
        vacc0p1 = _mm512_fmadd_ps(vi3x0, vk3x0, vacc0p1);

        const __m512 vi4x0 = maskz_load_bf16_as_f32(vmask, i4);
        const __m512 vk4x0 = maskz_load_bf16_as_f32(vmask, wk + 128);
        vacc0p0 = _mm512_fmadd_ps(vi4x0, vk4x0, vacc0p0);

        const __m512 vi5x0 = maskz_load_bf16_as_f32(vmask, i5);
        const __m512 vk5x0 = maskz_load_bf16_as_f32(vmask, wk + 160);
        vacc0p1 = _mm512_fmadd_ps(vi5x0, vk5x0, vacc0p1);

        const __m512 vi6x0 = maskz_load_bf16_as_f32(vmask, i6);
        const __m512 vk6x0 = maskz_load_bf16_as_f32(vmask, wk + 192);
        vacc0p0 = _mm512_fmadd_ps(vi6x0, vk6x0, vacc0p0);

        const __m512 vi7x0 = maskz_load_bf16_as_f32(vmask, i7);
        const __m512 vk7x0 = maskz_load_bf16_as_f32(vmask, wk + 224);
        vacc0p1 = _mm512_fmadd_ps(vi7x0, vk7x0, vacc0p1);

        const __m512 vi8x0 = maskz_load_bf16_as_f32(vmask, i8);
        const __m512 vk8x0 = maskz_load_bf16_as_f32(vmask, wk + 256);
        vacc0p0 = _mm512_fmadd_ps(vi8x0, vk8x0, vacc0p0);

        const __m512 vi9x0 = maskz_load_bf16_as_f32(vmask, i9);
        const __m512 vk9x0 = maskz_load_bf16_as_f32(vmask, wk + 288);
        vacc0p1 = _mm512_fmadd_ps(vi9x0, vk9x0, vacc0p1);

        const __m512 vi10x0 = maskz_load_bf16_as_f32(vmask, i10);
        const __m512 vk10x0 = maskz_load_bf16_as_f32(vmask, wk + 320);
        vacc0p0 = _mm512_fmadd_ps(vi10x0, vk10x0, vacc0p0);

        const __m512 vi11x0 = maskz_load_bf16_as_f32(vmask, i11);
        const __m512 vk11x0 = maskz_load_bf16_as_f32(vmask, wk + 352);
        vacc0p1 = _mm512_fmadd_ps(vi11x0, vk11x0, vacc0p1);

        const __m512 vi12x0 = maskz_load_bf16_as_f32(vmask, i12);
        const __m512 vk12x0 = maskz_load_bf16_as_f32(vmask, wk + 384);
        vacc0p0 = _mm512_fmadd_ps(vi12x0, vk12x0, vacc0p0);

        const __m512 vi13x0 = maskz_load_bf16_as_f32(vmask, i13);
        const __m512 vk13x0 = maskz_load_bf16_as_f32(vmask, wk + 416);
        vacc0p1 = _mm512_fmadd_ps(vi13x0, vk13x0, vacc0p1);

        const __m512 vi14x0 = maskz_load_bf16_as_f32(vmask, i14);
        const __m512 vk14x0 = maskz_load_bf16_as_f32(vmask, wk + 448);
        vacc0p0 = _mm512_fmadd_ps(vi14x0, vk14x0, vacc0p0);

        const __m512 vi15x0 = maskz_load_bf16_as_f32(vmask, i15);
        const __m512 vk15x0 = maskz_load_bf16_as_f32(vmask, wk + 480);
        vacc0p1 = _mm512_fmadd_ps(vi15x0, vk15x0, vacc0p1);

        const __m512 vi16x0 = maskz_load_bf16_as_f32(vmask, i16);
        const __m512 vk16x0 = maskz_load_bf16_as_f32(vmask, wk + 512);
        vacc0p0 = _mm512_fmadd_ps(vi16x0, vk16x0, vacc0p0);

        const __m512 vi17x0 = maskz_load_bf16_as_f32(vmask, i17);
        const __m512 vk17x0 = maskz_load_bf16_as_f32(vmask, wk + 544);
        vacc0p1 = _mm512_fmadd_ps(vi17x0, vk17x0, vacc0p1);

        const __m512 vi18x0 = maskz_load_bf16_as_f32(vmask, i18);
        const __m512 vk18x0 = maskz_load_bf16_as_f32(vmask, wk + 576);
        vacc0p0 = _mm512_fmadd_ps(vi18x0, vk18x0, vacc0p0);

        const __m512 vi19x0 = maskz_load_bf16_as_f32(vmask, i19);
        const __m512 vk19x0 = maskz_load_bf16_as_f32(vmask, wk + 608);
        vacc0p1 = _mm512_fmadd_ps(vi19x0, vk19x0, vacc0p1);

        const __m512 vi20x0 = maskz_load_bf16_as_f32(vmask, i20);
        const __m512 vk20x0 = maskz_load_bf16_as_f32(vmask, wk + 640);
        vacc0p0 = _mm512_fmadd_ps(vi20x0, vk20x0, vacc0p0);

        const __m512 vi21x0 = maskz_load_bf16_as_f32(vmask, i21);
        const __m512 vk21x0 = maskz_load_bf16_as_f32(vmask, wk + 672);
        vacc0p1 = _mm512_fmadd_ps(vi21x0, vk21x0, vacc0p1);

        const __m512 vi22x0 = maskz_load_bf16_as_f32(vmask, i22);
        const __m512 vk22x0 = maskz_load_bf16_as_f32(vmask, wk + 704);
        vacc0p0 = _mm512_fmadd_ps(vi22x0, vk22x0, vacc0p0);

        const __m512 vi23x0 = maskz_load_bf16_as_f32(vmask, i23);
        const __m512 vk23x0 = maskz_load_bf16_as_f32(vmask, wk + 736);
        vacc0p1 = _mm512_fmadd_ps(vi23x0, vk23x0, vacc0p1);

        const __m512 vi24x0 = maskz_load_bf16_as_f32(vmask, i24);
        const __m512 vk24x0 = maskz_load_bf16_as_f32(vmask, wk + 768);
        vacc0p0 = _mm512_fmadd_ps(vi24x0, vk24x0, vacc0p0);

        vacc0p0 = _mm512_add_ps(vacc0p0, vacc0p1);

        __m512 vacc0 = _mm512_max_ps(vmin, vacc0p0);
        vacc0 = _mm512_min_ps(vmax, vacc0);

        _mm512_mask_storeu_ps(output, vmask, vacc0);
        output += c;
      }
    }

    input_offset += input_pixel_stride;
    output = (float*) ((uintptr_t) output + output_increment);
  } while (--output_width != 0);
}
