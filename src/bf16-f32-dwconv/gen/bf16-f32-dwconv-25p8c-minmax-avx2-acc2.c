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


static XNN_INLINE __m256 load_bf16_as_f32(const uint16_t* ptr) {
  return _mm256_castsi256_ps(_mm256_slli_epi32(
      _mm256_cvtepu16_epi32(_mm_loadu_si128((const __m128i*) ptr)), 16));
}

// Packed weights layout, per block of 8 channels:
//   float bias[8];
//   uint16_t kernel[25][8];  // bf16
void xnn_bf16_f32_dwconv_minmax_ukernel_25p8c__avx2_acc2(
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
    const struct xnn_f32_minmax_params* restrict params) XNN_OOB_READS
{
  assert(channels != 0);
  assert(output_width != 0);

  const __m256 vmin = _mm256_set1_ps(params->scalar.min);
  const __m256 vmax = _mm256_set1_ps(params->scalar.max);
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
    for (; c >= 8; c -= 8) {
      const uint16_t* wk = (const uint16_t*) (wb + 8);
      __m256 vacc0p0 = _mm256_loadu_ps(wb + 0);


      const __m256 vi0x0 = load_bf16_as_f32(i0 + 0);
      i0 += 8;

      const __m256 vk0x0 = load_bf16_as_f32(wk + 0);
      vacc0p0 = _mm256_fmadd_ps(vi0x0, vk0x0, vacc0p0);

      const __m256 vi1x0 = load_bf16_as_f32(i1 + 0);
      i1 += 8;

      const __m256 vk1x0 = load_bf16_as_f32(wk + 8);
      __m256 vacc0p1 = _mm256_mul_ps(vi1x0, vk1x0);

      const __m256 vi2x0 = load_bf16_as_f32(i2 + 0);
      i2 += 8;

      const __m256 vk2x0 = load_bf16_as_f32(wk + 16);
      vacc0p0 = _mm256_fmadd_ps(vi2x0, vk2x0, vacc0p0);

      const __m256 vi3x0 = load_bf16_as_f32(i3 + 0);
      i3 += 8;

      const __m256 vk3x0 = load_bf16_as_f32(wk + 24);
      vacc0p1 = _mm256_fmadd_ps(vi3x0, vk3x0, vacc0p1);

      const __m256 vi4x0 = load_bf16_as_f32(i4 + 0);
      i4 += 8;

      const __m256 vk4x0 = load_bf16_as_f32(wk + 32);
      vacc0p0 = _mm256_fmadd_ps(vi4x0, vk4x0, vacc0p0);

      const __m256 vi5x0 = load_bf16_as_f32(i5 + 0);
      i5 += 8;

      const __m256 vk5x0 = load_bf16_as_f32(wk + 40);
      vacc0p1 = _mm256_fmadd_ps(vi5x0, vk5x0, vacc0p1);

      const __m256 vi6x0 = load_bf16_as_f32(i6 + 0);
      i6 += 8;

      const __m256 vk6x0 = load_bf16_as_f32(wk + 48);
      vacc0p0 = _mm256_fmadd_ps(vi6x0, vk6x0, vacc0p0);

      const __m256 vi7x0 = load_bf16_as_f32(i7 + 0);
      i7 += 8;

      const __m256 vk7x0 = load_bf16_as_f32(wk + 56);
      vacc0p1 = _mm256_fmadd_ps(vi7x0, vk7x0, vacc0p1);

      const __m256 vi8x0 = load_bf16_as_f32(i8 + 0);
      i8 += 8;

      const __m256 vk8x0 = load_bf16_as_f32(wk + 64);
      vacc0p0 = _mm256_fmadd_ps(vi8x0, vk8x0, vacc0p0);

      const __m256 vi9x0 = load_bf16_as_f32(i9 + 0);
      i9 += 8;

      const __m256 vk9x0 = load_bf16_as_f32(wk + 72);
      vacc0p1 = _mm256_fmadd_ps(vi9x0, vk9x0, vacc0p1);

      const __m256 vi10x0 = load_bf16_as_f32(i10 + 0);
      i10 += 8;

      const __m256 vk10x0 = load_bf16_as_f32(wk + 80);
      vacc0p0 = _mm256_fmadd_ps(vi10x0, vk10x0, vacc0p0);

      const __m256 vi11x0 = load_bf16_as_f32(i11 + 0);
      i11 += 8;

      const __m256 vk11x0 = load_bf16_as_f32(wk + 88);
      vacc0p1 = _mm256_fmadd_ps(vi11x0, vk11x0, vacc0p1);

      const __m256 vi12x0 = load_bf16_as_f32(i12 + 0);
      i12 += 8;

      const __m256 vk12x0 = load_bf16_as_f32(wk + 96);
      vacc0p0 = _mm256_fmadd_ps(vi12x0, vk12x0, vacc0p0);

      const __m256 vi13x0 = load_bf16_as_f32(i13 + 0);
      i13 += 8;

      const __m256 vk13x0 = load_bf16_as_f32(wk + 104);
      vacc0p1 = _mm256_fmadd_ps(vi13x0, vk13x0, vacc0p1);

      const __m256 vi14x0 = load_bf16_as_f32(i14 + 0);
      i14 += 8;

      const __m256 vk14x0 = load_bf16_as_f32(wk + 112);
      vacc0p0 = _mm256_fmadd_ps(vi14x0, vk14x0, vacc0p0);

      const __m256 vi15x0 = load_bf16_as_f32(i15 + 0);
      i15 += 8;

      const __m256 vk15x0 = load_bf16_as_f32(wk + 120);
      vacc0p1 = _mm256_fmadd_ps(vi15x0, vk15x0, vacc0p1);

      const __m256 vi16x0 = load_bf16_as_f32(i16 + 0);
      i16 += 8;

      const __m256 vk16x0 = load_bf16_as_f32(wk + 128);
      vacc0p0 = _mm256_fmadd_ps(vi16x0, vk16x0, vacc0p0);

      const __m256 vi17x0 = load_bf16_as_f32(i17 + 0);
      i17 += 8;

      const __m256 vk17x0 = load_bf16_as_f32(wk + 136);
      vacc0p1 = _mm256_fmadd_ps(vi17x0, vk17x0, vacc0p1);

      const __m256 vi18x0 = load_bf16_as_f32(i18 + 0);
      i18 += 8;

      const __m256 vk18x0 = load_bf16_as_f32(wk + 144);
      vacc0p0 = _mm256_fmadd_ps(vi18x0, vk18x0, vacc0p0);

      const __m256 vi19x0 = load_bf16_as_f32(i19 + 0);
      i19 += 8;

      const __m256 vk19x0 = load_bf16_as_f32(wk + 152);
      vacc0p1 = _mm256_fmadd_ps(vi19x0, vk19x0, vacc0p1);

      const __m256 vi20x0 = load_bf16_as_f32(i20 + 0);
      i20 += 8;

      const __m256 vk20x0 = load_bf16_as_f32(wk + 160);
      vacc0p0 = _mm256_fmadd_ps(vi20x0, vk20x0, vacc0p0);

      const __m256 vi21x0 = load_bf16_as_f32(i21 + 0);
      i21 += 8;

      const __m256 vk21x0 = load_bf16_as_f32(wk + 168);
      vacc0p1 = _mm256_fmadd_ps(vi21x0, vk21x0, vacc0p1);

      const __m256 vi22x0 = load_bf16_as_f32(i22 + 0);
      i22 += 8;

      const __m256 vk22x0 = load_bf16_as_f32(wk + 176);
      vacc0p0 = _mm256_fmadd_ps(vi22x0, vk22x0, vacc0p0);

      const __m256 vi23x0 = load_bf16_as_f32(i23 + 0);
      i23 += 8;

      const __m256 vk23x0 = load_bf16_as_f32(wk + 184);
      vacc0p1 = _mm256_fmadd_ps(vi23x0, vk23x0, vacc0p1);

      const __m256 vi24x0 = load_bf16_as_f32(i24 + 0);
      i24 += 8;

      const __m256 vk24x0 = load_bf16_as_f32(wk + 192);
      vacc0p0 = _mm256_fmadd_ps(vi24x0, vk24x0, vacc0p0);

      wb = (const float*) (wk + 200);

      vacc0p0 = _mm256_add_ps(vacc0p0, vacc0p1);

      __m256 vacc0 = _mm256_max_ps(vmin, vacc0p0);
      vacc0 = _mm256_min_ps(vmax, vacc0);

      _mm256_storeu_ps(output + 0, vacc0);
      output += 8;
    }
    if XNN_UNLIKELY(c != 0) {
      const uint16_t* wk = (const uint16_t*) (wb + 8);
      if XNN_LIKELY(c != 0) {
        // Reads past the last channel are covered by XNN_EXTRA_BYTES on the
        // inputs and by the channel tile padding of the packed weights.
        __m256 vacc0p0 = _mm256_loadu_ps(wb);

        const __m256 vi0x0 = load_bf16_as_f32(i0);
        const __m256 vk0x0 = load_bf16_as_f32(wk + 0);
        vacc0p0 = _mm256_fmadd_ps(vi0x0, vk0x0, vacc0p0);

        const __m256 vi1x0 = load_bf16_as_f32(i1);
        const __m256 vk1x0 = load_bf16_as_f32(wk + 8);
        __m256 vacc0p1 = _mm256_mul_ps(vi1x0, vk1x0);

        const __m256 vi2x0 = load_bf16_as_f32(i2);
        const __m256 vk2x0 = load_bf16_as_f32(wk + 16);
        vacc0p0 = _mm256_fmadd_ps(vi2x0, vk2x0, vacc0p0);

        const __m256 vi3x0 = load_bf16_as_f32(i3);
        const __m256 vk3x0 = load_bf16_as_f32(wk + 24);
        vacc0p1 = _mm256_fmadd_ps(vi3x0, vk3x0, vacc0p1);

        const __m256 vi4x0 = load_bf16_as_f32(i4);
        const __m256 vk4x0 = load_bf16_as_f32(wk + 32);
        vacc0p0 = _mm256_fmadd_ps(vi4x0, vk4x0, vacc0p0);

        const __m256 vi5x0 = load_bf16_as_f32(i5);
        const __m256 vk5x0 = load_bf16_as_f32(wk + 40);
        vacc0p1 = _mm256_fmadd_ps(vi5x0, vk5x0, vacc0p1);

        const __m256 vi6x0 = load_bf16_as_f32(i6);
        const __m256 vk6x0 = load_bf16_as_f32(wk + 48);
        vacc0p0 = _mm256_fmadd_ps(vi6x0, vk6x0, vacc0p0);

        const __m256 vi7x0 = load_bf16_as_f32(i7);
        const __m256 vk7x0 = load_bf16_as_f32(wk + 56);
        vacc0p1 = _mm256_fmadd_ps(vi7x0, vk7x0, vacc0p1);

        const __m256 vi8x0 = load_bf16_as_f32(i8);
        const __m256 vk8x0 = load_bf16_as_f32(wk + 64);
        vacc0p0 = _mm256_fmadd_ps(vi8x0, vk8x0, vacc0p0);

        const __m256 vi9x0 = load_bf16_as_f32(i9);
        const __m256 vk9x0 = load_bf16_as_f32(wk + 72);
        vacc0p1 = _mm256_fmadd_ps(vi9x0, vk9x0, vacc0p1);

        const __m256 vi10x0 = load_bf16_as_f32(i10);
        const __m256 vk10x0 = load_bf16_as_f32(wk + 80);
        vacc0p0 = _mm256_fmadd_ps(vi10x0, vk10x0, vacc0p0);

        const __m256 vi11x0 = load_bf16_as_f32(i11);
        const __m256 vk11x0 = load_bf16_as_f32(wk + 88);
        vacc0p1 = _mm256_fmadd_ps(vi11x0, vk11x0, vacc0p1);

        const __m256 vi12x0 = load_bf16_as_f32(i12);
        const __m256 vk12x0 = load_bf16_as_f32(wk + 96);
        vacc0p0 = _mm256_fmadd_ps(vi12x0, vk12x0, vacc0p0);

        const __m256 vi13x0 = load_bf16_as_f32(i13);
        const __m256 vk13x0 = load_bf16_as_f32(wk + 104);
        vacc0p1 = _mm256_fmadd_ps(vi13x0, vk13x0, vacc0p1);

        const __m256 vi14x0 = load_bf16_as_f32(i14);
        const __m256 vk14x0 = load_bf16_as_f32(wk + 112);
        vacc0p0 = _mm256_fmadd_ps(vi14x0, vk14x0, vacc0p0);

        const __m256 vi15x0 = load_bf16_as_f32(i15);
        const __m256 vk15x0 = load_bf16_as_f32(wk + 120);
        vacc0p1 = _mm256_fmadd_ps(vi15x0, vk15x0, vacc0p1);

        const __m256 vi16x0 = load_bf16_as_f32(i16);
        const __m256 vk16x0 = load_bf16_as_f32(wk + 128);
        vacc0p0 = _mm256_fmadd_ps(vi16x0, vk16x0, vacc0p0);

        const __m256 vi17x0 = load_bf16_as_f32(i17);
        const __m256 vk17x0 = load_bf16_as_f32(wk + 136);
        vacc0p1 = _mm256_fmadd_ps(vi17x0, vk17x0, vacc0p1);

        const __m256 vi18x0 = load_bf16_as_f32(i18);
        const __m256 vk18x0 = load_bf16_as_f32(wk + 144);
        vacc0p0 = _mm256_fmadd_ps(vi18x0, vk18x0, vacc0p0);

        const __m256 vi19x0 = load_bf16_as_f32(i19);
        const __m256 vk19x0 = load_bf16_as_f32(wk + 152);
        vacc0p1 = _mm256_fmadd_ps(vi19x0, vk19x0, vacc0p1);

        const __m256 vi20x0 = load_bf16_as_f32(i20);
        const __m256 vk20x0 = load_bf16_as_f32(wk + 160);
        vacc0p0 = _mm256_fmadd_ps(vi20x0, vk20x0, vacc0p0);

        const __m256 vi21x0 = load_bf16_as_f32(i21);
        const __m256 vk21x0 = load_bf16_as_f32(wk + 168);
        vacc0p1 = _mm256_fmadd_ps(vi21x0, vk21x0, vacc0p1);

        const __m256 vi22x0 = load_bf16_as_f32(i22);
        const __m256 vk22x0 = load_bf16_as_f32(wk + 176);
        vacc0p0 = _mm256_fmadd_ps(vi22x0, vk22x0, vacc0p0);

        const __m256 vi23x0 = load_bf16_as_f32(i23);
        const __m256 vk23x0 = load_bf16_as_f32(wk + 184);
        vacc0p1 = _mm256_fmadd_ps(vi23x0, vk23x0, vacc0p1);

        const __m256 vi24x0 = load_bf16_as_f32(i24);
        const __m256 vk24x0 = load_bf16_as_f32(wk + 192);
        vacc0p0 = _mm256_fmadd_ps(vi24x0, vk24x0, vacc0p0);

        vacc0p0 = _mm256_add_ps(vacc0p0, vacc0p1);

        __m256 vacc0 = _mm256_max_ps(vmin, vacc0p0);
        vacc0 = _mm256_min_ps(vmax, vacc0);

        __m128 vacc0_lo = _mm256_castps256_ps128(vacc0);
        if (c & 4) {
          _mm_storeu_ps(output, vacc0_lo);
          vacc0_lo = _mm256_extractf128_ps(vacc0, 1);
          output += 4;
        }
        if (c & 2) {
          _mm_storel_pi((__m64*) output, vacc0_lo);
          vacc0_lo = _mm_movehl_ps(vacc0_lo, vacc0_lo);
          output += 2;
        }
        if (c & 1) {
          _mm_store_ss(output, vacc0_lo);
          output += 1;
        }
      }
    }

    input_offset += input_pixel_stride;
    output = (float*) ((uintptr_t) output + output_increment);
  } while (--output_width != 0);
}
