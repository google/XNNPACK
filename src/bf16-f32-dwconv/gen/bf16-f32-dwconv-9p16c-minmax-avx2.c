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

// Packed weights layout, per block of 16 channels:
//   float bias[16];
//   uint16_t kernel[9][16];  // bf16
void xnn_bf16_f32_dwconv_minmax_ukernel_9p16c__avx2(
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
    input = (const xnn_bfloat16**) ((uintptr_t) input + input_stride);

    size_t c = channels;
    const float* wb = (const float*) weights;
    for (; c >= 16; c -= 16) {
      const uint16_t* wk = (const uint16_t*) (wb + 16);
      __m256 vacc0p0 = _mm256_loadu_ps(wb + 0);
      __m256 vacc8p0 = _mm256_loadu_ps(wb + 8);


      const __m256 vi0x0 = load_bf16_as_f32(i0 + 0);
      const __m256 vi0x8 = load_bf16_as_f32(i0 + 8);
      i0 += 16;

      const __m256 vk0x0 = load_bf16_as_f32(wk + 0);
      const __m256 vk0x8 = load_bf16_as_f32(wk + 8);
      vacc0p0 = _mm256_fmadd_ps(vi0x0, vk0x0, vacc0p0);
      vacc8p0 = _mm256_fmadd_ps(vi0x8, vk0x8, vacc8p0);

      const __m256 vi1x0 = load_bf16_as_f32(i1 + 0);
      const __m256 vi1x8 = load_bf16_as_f32(i1 + 8);
      i1 += 16;

      const __m256 vk1x0 = load_bf16_as_f32(wk + 16);
      const __m256 vk1x8 = load_bf16_as_f32(wk + 24);
      vacc0p0 = _mm256_fmadd_ps(vi1x0, vk1x0, vacc0p0);
      vacc8p0 = _mm256_fmadd_ps(vi1x8, vk1x8, vacc8p0);

      const __m256 vi2x0 = load_bf16_as_f32(i2 + 0);
      const __m256 vi2x8 = load_bf16_as_f32(i2 + 8);
      i2 += 16;

      const __m256 vk2x0 = load_bf16_as_f32(wk + 32);
      const __m256 vk2x8 = load_bf16_as_f32(wk + 40);
      vacc0p0 = _mm256_fmadd_ps(vi2x0, vk2x0, vacc0p0);
      vacc8p0 = _mm256_fmadd_ps(vi2x8, vk2x8, vacc8p0);

      const __m256 vi3x0 = load_bf16_as_f32(i3 + 0);
      const __m256 vi3x8 = load_bf16_as_f32(i3 + 8);
      i3 += 16;

      const __m256 vk3x0 = load_bf16_as_f32(wk + 48);
      const __m256 vk3x8 = load_bf16_as_f32(wk + 56);
      vacc0p0 = _mm256_fmadd_ps(vi3x0, vk3x0, vacc0p0);
      vacc8p0 = _mm256_fmadd_ps(vi3x8, vk3x8, vacc8p0);

      const __m256 vi4x0 = load_bf16_as_f32(i4 + 0);
      const __m256 vi4x8 = load_bf16_as_f32(i4 + 8);
      i4 += 16;

      const __m256 vk4x0 = load_bf16_as_f32(wk + 64);
      const __m256 vk4x8 = load_bf16_as_f32(wk + 72);
      vacc0p0 = _mm256_fmadd_ps(vi4x0, vk4x0, vacc0p0);
      vacc8p0 = _mm256_fmadd_ps(vi4x8, vk4x8, vacc8p0);

      const __m256 vi5x0 = load_bf16_as_f32(i5 + 0);
      const __m256 vi5x8 = load_bf16_as_f32(i5 + 8);
      i5 += 16;

      const __m256 vk5x0 = load_bf16_as_f32(wk + 80);
      const __m256 vk5x8 = load_bf16_as_f32(wk + 88);
      vacc0p0 = _mm256_fmadd_ps(vi5x0, vk5x0, vacc0p0);
      vacc8p0 = _mm256_fmadd_ps(vi5x8, vk5x8, vacc8p0);

      const __m256 vi6x0 = load_bf16_as_f32(i6 + 0);
      const __m256 vi6x8 = load_bf16_as_f32(i6 + 8);
      i6 += 16;

      const __m256 vk6x0 = load_bf16_as_f32(wk + 96);
      const __m256 vk6x8 = load_bf16_as_f32(wk + 104);
      vacc0p0 = _mm256_fmadd_ps(vi6x0, vk6x0, vacc0p0);
      vacc8p0 = _mm256_fmadd_ps(vi6x8, vk6x8, vacc8p0);

      const __m256 vi7x0 = load_bf16_as_f32(i7 + 0);
      const __m256 vi7x8 = load_bf16_as_f32(i7 + 8);
      i7 += 16;

      const __m256 vk7x0 = load_bf16_as_f32(wk + 112);
      const __m256 vk7x8 = load_bf16_as_f32(wk + 120);
      vacc0p0 = _mm256_fmadd_ps(vi7x0, vk7x0, vacc0p0);
      vacc8p0 = _mm256_fmadd_ps(vi7x8, vk7x8, vacc8p0);

      const __m256 vi8x0 = load_bf16_as_f32(i8 + 0);
      const __m256 vi8x8 = load_bf16_as_f32(i8 + 8);
      i8 += 16;

      const __m256 vk8x0 = load_bf16_as_f32(wk + 128);
      const __m256 vk8x8 = load_bf16_as_f32(wk + 136);
      vacc0p0 = _mm256_fmadd_ps(vi8x0, vk8x0, vacc0p0);
      vacc8p0 = _mm256_fmadd_ps(vi8x8, vk8x8, vacc8p0);

      wb = (const float*) (wk + 144);


      __m256 vacc0 = _mm256_max_ps(vmin, vacc0p0);
      __m256 vacc8 = _mm256_max_ps(vmin, vacc8p0);
      vacc0 = _mm256_min_ps(vmax, vacc0);
      vacc8 = _mm256_min_ps(vmax, vacc8);

      _mm256_storeu_ps(output + 0, vacc0);
      _mm256_storeu_ps(output + 8, vacc8);
      output += 16;
    }
    if XNN_UNLIKELY(c != 0) {
      const uint16_t* wk = (const uint16_t*) (wb + 16);
      for (; c >= 8; c -= 8) {
        __m256 vacc0p0 = _mm256_loadu_ps(wb);

        const __m256 vi0x0 = load_bf16_as_f32(i0);
        i0 += 8;

        const __m256 vk0x0 = load_bf16_as_f32(wk + 0);
        vacc0p0 = _mm256_fmadd_ps(vi0x0, vk0x0, vacc0p0);

        const __m256 vi1x0 = load_bf16_as_f32(i1);
        i1 += 8;

        const __m256 vk1x0 = load_bf16_as_f32(wk + 16);
        vacc0p0 = _mm256_fmadd_ps(vi1x0, vk1x0, vacc0p0);

        const __m256 vi2x0 = load_bf16_as_f32(i2);
        i2 += 8;

        const __m256 vk2x0 = load_bf16_as_f32(wk + 32);
        vacc0p0 = _mm256_fmadd_ps(vi2x0, vk2x0, vacc0p0);

        const __m256 vi3x0 = load_bf16_as_f32(i3);
        i3 += 8;

        const __m256 vk3x0 = load_bf16_as_f32(wk + 48);
        vacc0p0 = _mm256_fmadd_ps(vi3x0, vk3x0, vacc0p0);

        const __m256 vi4x0 = load_bf16_as_f32(i4);
        i4 += 8;

        const __m256 vk4x0 = load_bf16_as_f32(wk + 64);
        vacc0p0 = _mm256_fmadd_ps(vi4x0, vk4x0, vacc0p0);

        const __m256 vi5x0 = load_bf16_as_f32(i5);
        i5 += 8;

        const __m256 vk5x0 = load_bf16_as_f32(wk + 80);
        vacc0p0 = _mm256_fmadd_ps(vi5x0, vk5x0, vacc0p0);

        const __m256 vi6x0 = load_bf16_as_f32(i6);
        i6 += 8;

        const __m256 vk6x0 = load_bf16_as_f32(wk + 96);
        vacc0p0 = _mm256_fmadd_ps(vi6x0, vk6x0, vacc0p0);

        const __m256 vi7x0 = load_bf16_as_f32(i7);
        i7 += 8;

        const __m256 vk7x0 = load_bf16_as_f32(wk + 112);
        vacc0p0 = _mm256_fmadd_ps(vi7x0, vk7x0, vacc0p0);

        const __m256 vi8x0 = load_bf16_as_f32(i8);
        i8 += 8;

        const __m256 vk8x0 = load_bf16_as_f32(wk + 128);
        vacc0p0 = _mm256_fmadd_ps(vi8x0, vk8x0, vacc0p0);

        wb += 8;
        wk += 8;


        __m256 vacc0 = _mm256_max_ps(vmin, vacc0p0);
        vacc0 = _mm256_min_ps(vmax, vacc0);

        _mm256_storeu_ps(output, vacc0);
        output += 8;
      }
      if XNN_LIKELY(c != 0) {
        // Reads past the last channel are covered by XNN_EXTRA_BYTES on the
        // inputs and by the channel tile padding of the packed weights.
        __m256 vacc0p0 = _mm256_loadu_ps(wb);

        const __m256 vi0x0 = load_bf16_as_f32(i0);
        const __m256 vk0x0 = load_bf16_as_f32(wk + 0);
        vacc0p0 = _mm256_fmadd_ps(vi0x0, vk0x0, vacc0p0);

        const __m256 vi1x0 = load_bf16_as_f32(i1);
        const __m256 vk1x0 = load_bf16_as_f32(wk + 16);
        vacc0p0 = _mm256_fmadd_ps(vi1x0, vk1x0, vacc0p0);

        const __m256 vi2x0 = load_bf16_as_f32(i2);
        const __m256 vk2x0 = load_bf16_as_f32(wk + 32);
        vacc0p0 = _mm256_fmadd_ps(vi2x0, vk2x0, vacc0p0);

        const __m256 vi3x0 = load_bf16_as_f32(i3);
        const __m256 vk3x0 = load_bf16_as_f32(wk + 48);
        vacc0p0 = _mm256_fmadd_ps(vi3x0, vk3x0, vacc0p0);

        const __m256 vi4x0 = load_bf16_as_f32(i4);
        const __m256 vk4x0 = load_bf16_as_f32(wk + 64);
        vacc0p0 = _mm256_fmadd_ps(vi4x0, vk4x0, vacc0p0);

        const __m256 vi5x0 = load_bf16_as_f32(i5);
        const __m256 vk5x0 = load_bf16_as_f32(wk + 80);
        vacc0p0 = _mm256_fmadd_ps(vi5x0, vk5x0, vacc0p0);

        const __m256 vi6x0 = load_bf16_as_f32(i6);
        const __m256 vk6x0 = load_bf16_as_f32(wk + 96);
        vacc0p0 = _mm256_fmadd_ps(vi6x0, vk6x0, vacc0p0);

        const __m256 vi7x0 = load_bf16_as_f32(i7);
        const __m256 vk7x0 = load_bf16_as_f32(wk + 112);
        vacc0p0 = _mm256_fmadd_ps(vi7x0, vk7x0, vacc0p0);

        const __m256 vi8x0 = load_bf16_as_f32(i8);
        const __m256 vk8x0 = load_bf16_as_f32(wk + 128);
        vacc0p0 = _mm256_fmadd_ps(vi8x0, vk8x0, vacc0p0);


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
