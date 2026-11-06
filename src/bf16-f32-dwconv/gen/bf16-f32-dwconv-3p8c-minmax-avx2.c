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
//   uint16_t kernel[3][8];  // bf16
void xnn_bf16_f32_dwconv_minmax_ukernel_3p8c__avx2(
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
      vacc0p0 = _mm256_fmadd_ps(vi1x0, vk1x0, vacc0p0);

      const __m256 vi2x0 = load_bf16_as_f32(i2 + 0);
      i2 += 8;

      const __m256 vk2x0 = load_bf16_as_f32(wk + 16);
      vacc0p0 = _mm256_fmadd_ps(vi2x0, vk2x0, vacc0p0);

      wb = (const float*) (wk + 24);


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
        vacc0p0 = _mm256_fmadd_ps(vi1x0, vk1x0, vacc0p0);

        const __m256 vi2x0 = load_bf16_as_f32(i2);
        const __m256 vk2x0 = load_bf16_as_f32(wk + 16);
        vacc0p0 = _mm256_fmadd_ps(vi2x0, vk2x0, vacc0p0);


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
