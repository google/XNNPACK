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

// Packed weights layout, per block of 16 channels:
//   float bias[16];
//   uint16_t kernel[9][16];  // bf16
void xnn_bf16_f32_dwconv_minmax_ukernel_9p16c__avx512skx(
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
    input = (const xnn_bfloat16**) ((uintptr_t) input + input_stride);

    size_t c = channels;
    const float* wb = (const float*) weights;
    for (; c >= 16; c -= 16) {
      const uint16_t* wk = (const uint16_t*) (wb + 16);
      __m512 vacc0p0 = _mm512_loadu_ps(wb + 0);


      const __m512 vi0x0 = load_bf16_as_f32(i0 + 0);
      i0 += 16;

      const __m512 vk0x0 = load_bf16_as_f32(wk + 0);
      vacc0p0 = _mm512_fmadd_ps(vi0x0, vk0x0, vacc0p0);

      const __m512 vi1x0 = load_bf16_as_f32(i1 + 0);
      i1 += 16;

      const __m512 vk1x0 = load_bf16_as_f32(wk + 16);
      vacc0p0 = _mm512_fmadd_ps(vi1x0, vk1x0, vacc0p0);

      const __m512 vi2x0 = load_bf16_as_f32(i2 + 0);
      i2 += 16;

      const __m512 vk2x0 = load_bf16_as_f32(wk + 32);
      vacc0p0 = _mm512_fmadd_ps(vi2x0, vk2x0, vacc0p0);

      const __m512 vi3x0 = load_bf16_as_f32(i3 + 0);
      i3 += 16;

      const __m512 vk3x0 = load_bf16_as_f32(wk + 48);
      vacc0p0 = _mm512_fmadd_ps(vi3x0, vk3x0, vacc0p0);

      const __m512 vi4x0 = load_bf16_as_f32(i4 + 0);
      i4 += 16;

      const __m512 vk4x0 = load_bf16_as_f32(wk + 64);
      vacc0p0 = _mm512_fmadd_ps(vi4x0, vk4x0, vacc0p0);

      const __m512 vi5x0 = load_bf16_as_f32(i5 + 0);
      i5 += 16;

      const __m512 vk5x0 = load_bf16_as_f32(wk + 80);
      vacc0p0 = _mm512_fmadd_ps(vi5x0, vk5x0, vacc0p0);

      const __m512 vi6x0 = load_bf16_as_f32(i6 + 0);
      i6 += 16;

      const __m512 vk6x0 = load_bf16_as_f32(wk + 96);
      vacc0p0 = _mm512_fmadd_ps(vi6x0, vk6x0, vacc0p0);

      const __m512 vi7x0 = load_bf16_as_f32(i7 + 0);
      i7 += 16;

      const __m512 vk7x0 = load_bf16_as_f32(wk + 112);
      vacc0p0 = _mm512_fmadd_ps(vi7x0, vk7x0, vacc0p0);

      const __m512 vi8x0 = load_bf16_as_f32(i8 + 0);
      i8 += 16;

      const __m512 vk8x0 = load_bf16_as_f32(wk + 128);
      vacc0p0 = _mm512_fmadd_ps(vi8x0, vk8x0, vacc0p0);

      wb = (const float*) (wk + 144);


      __m512 vacc0 = _mm512_max_ps(vmin, vacc0p0);
      vacc0 = _mm512_min_ps(vmax, vacc0);

      _mm512_storeu_ps(output + 0, vacc0);
      output += 16;
    }
    if XNN_UNLIKELY(c != 0) {
      const uint16_t* wk = (const uint16_t*) (wb + 16);
      if XNN_LIKELY(c != 0) {
        const __mmask16 vmask = _cvtu32_mask16((uint32_t) ((UINT32_C(1) << c) - 1));
        __m512 vacc0p0 = _mm512_maskz_loadu_ps(vmask, wb);

        const __m512 vi0x0 = maskz_load_bf16_as_f32(vmask, i0);
        const __m512 vk0x0 = maskz_load_bf16_as_f32(vmask, wk + 0);
        vacc0p0 = _mm512_fmadd_ps(vi0x0, vk0x0, vacc0p0);

        const __m512 vi1x0 = maskz_load_bf16_as_f32(vmask, i1);
        const __m512 vk1x0 = maskz_load_bf16_as_f32(vmask, wk + 16);
        vacc0p0 = _mm512_fmadd_ps(vi1x0, vk1x0, vacc0p0);

        const __m512 vi2x0 = maskz_load_bf16_as_f32(vmask, i2);
        const __m512 vk2x0 = maskz_load_bf16_as_f32(vmask, wk + 32);
        vacc0p0 = _mm512_fmadd_ps(vi2x0, vk2x0, vacc0p0);

        const __m512 vi3x0 = maskz_load_bf16_as_f32(vmask, i3);
        const __m512 vk3x0 = maskz_load_bf16_as_f32(vmask, wk + 48);
        vacc0p0 = _mm512_fmadd_ps(vi3x0, vk3x0, vacc0p0);

        const __m512 vi4x0 = maskz_load_bf16_as_f32(vmask, i4);
        const __m512 vk4x0 = maskz_load_bf16_as_f32(vmask, wk + 64);
        vacc0p0 = _mm512_fmadd_ps(vi4x0, vk4x0, vacc0p0);

        const __m512 vi5x0 = maskz_load_bf16_as_f32(vmask, i5);
        const __m512 vk5x0 = maskz_load_bf16_as_f32(vmask, wk + 80);
        vacc0p0 = _mm512_fmadd_ps(vi5x0, vk5x0, vacc0p0);

        const __m512 vi6x0 = maskz_load_bf16_as_f32(vmask, i6);
        const __m512 vk6x0 = maskz_load_bf16_as_f32(vmask, wk + 96);
        vacc0p0 = _mm512_fmadd_ps(vi6x0, vk6x0, vacc0p0);

        const __m512 vi7x0 = maskz_load_bf16_as_f32(vmask, i7);
        const __m512 vk7x0 = maskz_load_bf16_as_f32(vmask, wk + 112);
        vacc0p0 = _mm512_fmadd_ps(vi7x0, vk7x0, vacc0p0);

        const __m512 vi8x0 = maskz_load_bf16_as_f32(vmask, i8);
        const __m512 vk8x0 = maskz_load_bf16_as_f32(vmask, wk + 128);
        vacc0p0 = _mm512_fmadd_ps(vi8x0, vk8x0, vacc0p0);


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
