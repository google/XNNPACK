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


void xnn_f16_dwconv_minmax_ukernel_9p32c__avx512fp16_acc2(
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
      __m512h vacc0p1 = _mm512_mul_ph(vi1x0, vk1x0);

      const __m512h vi2x0 = _mm512_loadu_ph(i2 + 0);
      i2 += 32;

      const __m512h vk2x0 = _mm512_loadu_ph(w + 96);
      vacc0p0 = _mm512_fmadd_ph(vi2x0, vk2x0, vacc0p0);

      const __m512h vi3x0 = _mm512_loadu_ph(i3 + 0);
      i3 += 32;

      const __m512h vk3x0 = _mm512_loadu_ph(w + 128);
      vacc0p1 = _mm512_fmadd_ph(vi3x0, vk3x0, vacc0p1);

      const __m512h vi4x0 = _mm512_loadu_ph(i4 + 0);
      i4 += 32;

      const __m512h vk4x0 = _mm512_loadu_ph(w + 160);
      vacc0p0 = _mm512_fmadd_ph(vi4x0, vk4x0, vacc0p0);

      const __m512h vi5x0 = _mm512_loadu_ph(i5 + 0);
      i5 += 32;

      const __m512h vk5x0 = _mm512_loadu_ph(w + 192);
      vacc0p1 = _mm512_fmadd_ph(vi5x0, vk5x0, vacc0p1);

      const __m512h vi6x0 = _mm512_loadu_ph(i6 + 0);
      i6 += 32;

      const __m512h vk6x0 = _mm512_loadu_ph(w + 224);
      vacc0p0 = _mm512_fmadd_ph(vi6x0, vk6x0, vacc0p0);

      const __m512h vi7x0 = _mm512_loadu_ph(i7 + 0);
      i7 += 32;

      const __m512h vk7x0 = _mm512_loadu_ph(w + 256);
      vacc0p1 = _mm512_fmadd_ph(vi7x0, vk7x0, vacc0p1);

      const __m512h vi8x0 = _mm512_loadu_ph(i8 + 0);
      i8 += 32;

      const __m512h vk8x0 = _mm512_loadu_ph(w + 288);
      vacc0p0 = _mm512_fmadd_ph(vi8x0, vk8x0, vacc0p0);

      w += 320;

      // Add up all accumulators to the p0 accumulators.
      vacc0p0 = _mm512_add_ph(vacc0p0, vacc0p1);

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
      __m512h vacc0p1 = _mm512_mul_ph(vi1x0, vk1x0);

      const __m512h vi2x0 = _mm512_castsi512_ph(_mm512_maskz_loadu_epi16(vmask, i2));

      const __m512h vk2x0 = _mm512_castsi512_ph(_mm512_maskz_loadu_epi16(vmask, w + 96));
      vacc0p0 = _mm512_fmadd_ph(vi2x0, vk2x0, vacc0p0);

      const __m512h vi3x0 = _mm512_castsi512_ph(_mm512_maskz_loadu_epi16(vmask, i3));

      const __m512h vk3x0 = _mm512_castsi512_ph(_mm512_maskz_loadu_epi16(vmask, w + 128));
      vacc0p1 = _mm512_fmadd_ph(vi3x0, vk3x0, vacc0p1);

      const __m512h vi4x0 = _mm512_castsi512_ph(_mm512_maskz_loadu_epi16(vmask, i4));

      const __m512h vk4x0 = _mm512_castsi512_ph(_mm512_maskz_loadu_epi16(vmask, w + 160));
      vacc0p0 = _mm512_fmadd_ph(vi4x0, vk4x0, vacc0p0);

      const __m512h vi5x0 = _mm512_castsi512_ph(_mm512_maskz_loadu_epi16(vmask, i5));

      const __m512h vk5x0 = _mm512_castsi512_ph(_mm512_maskz_loadu_epi16(vmask, w + 192));
      vacc0p1 = _mm512_fmadd_ph(vi5x0, vk5x0, vacc0p1);

      const __m512h vi6x0 = _mm512_castsi512_ph(_mm512_maskz_loadu_epi16(vmask, i6));

      const __m512h vk6x0 = _mm512_castsi512_ph(_mm512_maskz_loadu_epi16(vmask, w + 224));
      vacc0p0 = _mm512_fmadd_ph(vi6x0, vk6x0, vacc0p0);

      const __m512h vi7x0 = _mm512_castsi512_ph(_mm512_maskz_loadu_epi16(vmask, i7));

      const __m512h vk7x0 = _mm512_castsi512_ph(_mm512_maskz_loadu_epi16(vmask, w + 256));
      vacc0p1 = _mm512_fmadd_ph(vi7x0, vk7x0, vacc0p1);

      const __m512h vi8x0 = _mm512_castsi512_ph(_mm512_maskz_loadu_epi16(vmask, i8));

      const __m512h vk8x0 = _mm512_castsi512_ph(_mm512_maskz_loadu_epi16(vmask, w + 288));
      vacc0p0 = _mm512_fmadd_ph(vi8x0, vk8x0, vacc0p0);

      // Add up all accumulators to vacc0p0
      vacc0p0 = _mm512_add_ph(vacc0p0, vacc0p1);

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
