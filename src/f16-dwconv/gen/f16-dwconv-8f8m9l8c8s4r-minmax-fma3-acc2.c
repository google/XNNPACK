// Auto-generated file. Do not edit!
//   Template: src/f16-dwconv/multipass-fma3.c.in
//   Generator: tools/xngen
//
// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include <immintrin.h>

#include "xnnpack/dwconv.h"
#include "xnnpack/intrinsics-polyfill.h"
#include "xnnpack/math.h"


void xnn_f16_dwconv_minmax_ukernel_8f8m9l8c8s4r__fma3_acc2(
    size_t channels,
    size_t output_width,
    const xnn_float16** input,
    const xnn_float16* weights,
    xnn_float16* output,
    intptr_t input_stride,
    size_t output_increment,
    size_t input_offset,
    const xnn_float16* zero,
    size_t kernel_size,
    xnn_float16* buffer,
    const union xnn_f16_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(channels != 0);
  assert(output_width != 0);
  assert(kernel_size > 8);

  const __m256 vmin = _mm256_cvtph_ps(_mm_set1_epi16(*(const uint16_t*) &params->scalar.min));
  const __m256 vmax = _mm256_cvtph_ps(_mm_set1_epi16(*(const uint16_t*) &params->scalar.max));
  XNN_FORCE_REALIZATION(vmin);
  XNN_FORCE_REALIZATION(vmax);

  do {
    const xnn_float16* w = weights;

    // First pass to process 8 inputs.
    {
      uint16_t* b = (uint16_t*) buffer;
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
      input += 8;

      // Process c channels and write to buffer.
      size_t c = round_up_po2(channels, 4);

      for (; c >= 8; c -= 8) {
        __m256 vacc01234567p0 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) w));


        const __m256 vi0x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i0)));
        i0 += 8;

        const __m256 vk0x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 8)));
        vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi0x01234567, vk0x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));

        const __m256 vi1x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i1)));
        i1 += 8;

        const __m256 vk1x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 16)));
        __m256 vacc01234567p1 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_mul_ps(vi1x01234567, vk1x01234567), _MM_FROUND_TO_NEAREST_INT));

        const __m256 vi2x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i2)));
        i2 += 8;

        const __m256 vk2x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 24)));
        vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi2x01234567, vk2x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));

        const __m256 vi3x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i3)));
        i3 += 8;

        const __m256 vk3x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 32)));
        vacc01234567p1 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi3x01234567, vk3x01234567, vacc01234567p1), _MM_FROUND_TO_NEAREST_INT));

        const __m256 vi4x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i4)));
        i4 += 8;

        const __m256 vk4x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 40)));
        vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi4x01234567, vk4x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));

        const __m256 vi5x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i5)));
        i5 += 8;

        const __m256 vk5x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 48)));
        vacc01234567p1 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi5x01234567, vk5x01234567, vacc01234567p1), _MM_FROUND_TO_NEAREST_INT));

        const __m256 vi6x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i6)));
        i6 += 8;

        const __m256 vk6x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 56)));
        vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi6x01234567, vk6x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));

        const __m256 vi7x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i7)));
        i7 += 8;

        const __m256 vk7x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 64)));
        vacc01234567p1 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi7x01234567, vk7x01234567, vacc01234567p1), _MM_FROUND_TO_NEAREST_INT));

        w += 72;

        // Add up all accumulators to vacc01234567p0
        vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_add_ps(vacc01234567p0, vacc01234567p1), _MM_FROUND_TO_NEAREST_INT));

        _mm_store_si128((__m128i*) b, _mm256_cvtps_ph(vacc01234567p0, _MM_FROUND_TO_NEAREST_INT));
        b += 8;
      }

      if (c != 0) {
        assert(c >= 1);
        assert(c <= 7);
        __m256 vacc01234567p0 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) w));


        const __m256 vi0x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i0));

        const __m256 vk0x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 8)));
        vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi0x01234567, vk0x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));

        const __m256 vi1x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i1));

        const __m256 vk1x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 16)));
        __m256 vacc01234567p1 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_mul_ps(vi1x01234567, vk1x01234567), _MM_FROUND_TO_NEAREST_INT));

        const __m256 vi2x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i2));

        const __m256 vk2x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 24)));
        vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi2x01234567, vk2x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));

        const __m256 vi3x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i3));

        const __m256 vk3x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 32)));
        vacc01234567p1 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi3x01234567, vk3x01234567, vacc01234567p1), _MM_FROUND_TO_NEAREST_INT));

        const __m256 vi4x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i4));

        const __m256 vk4x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 40)));
        vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi4x01234567, vk4x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));

        const __m256 vi5x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i5));

        const __m256 vk5x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 48)));
        vacc01234567p1 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi5x01234567, vk5x01234567, vacc01234567p1), _MM_FROUND_TO_NEAREST_INT));

        const __m256 vi6x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i6));

        const __m256 vk6x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 56)));
        vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi6x01234567, vk6x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));

        const __m256 vi7x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i7));

        const __m256 vk7x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 64)));
        vacc01234567p1 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi7x01234567, vk7x01234567, vacc01234567p1), _MM_FROUND_TO_NEAREST_INT));

        w += 72;

        // Add up all accumulators to vacc01234567p0
        vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_add_ps(vacc01234567p0, vacc01234567p1), _MM_FROUND_TO_NEAREST_INT));

        _mm_store_si128((__m128i*) b, _mm256_cvtps_ph(vacc01234567p0, _MM_FROUND_TO_NEAREST_INT));
      }
    }

    // Middle pass to process 8 inputs in each iteration.
    for (size_t ks = kernel_size - 8; ks > 9; ks -= 8) {
      uint16_t* b = (uint16_t*) buffer;
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
      input += 8;

      size_t c = round_up_po2(channels, 4);

      for (; c >= 8; c -= 8) {
        __m256 vacc01234567p0 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (b)));


        const __m256 vi0x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i0)));
        i0 += 8;

        const __m256 vk0x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w)));
        vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi0x01234567, vk0x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));

        const __m256 vi1x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i1)));
        i1 += 8;

        const __m256 vk1x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 8)));
        __m256 vacc01234567p1 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_mul_ps(vi1x01234567, vk1x01234567), _MM_FROUND_TO_NEAREST_INT));

        const __m256 vi2x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i2)));
        i2 += 8;

        const __m256 vk2x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 16)));
        vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi2x01234567, vk2x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));

        const __m256 vi3x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i3)));
        i3 += 8;

        const __m256 vk3x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 24)));
        vacc01234567p1 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi3x01234567, vk3x01234567, vacc01234567p1), _MM_FROUND_TO_NEAREST_INT));

        const __m256 vi4x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i4)));
        i4 += 8;

        const __m256 vk4x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 32)));
        vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi4x01234567, vk4x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));

        const __m256 vi5x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i5)));
        i5 += 8;

        const __m256 vk5x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 40)));
        vacc01234567p1 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi5x01234567, vk5x01234567, vacc01234567p1), _MM_FROUND_TO_NEAREST_INT));

        const __m256 vi6x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i6)));
        i6 += 8;

        const __m256 vk6x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 48)));
        vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi6x01234567, vk6x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));

        const __m256 vi7x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i7)));
        i7 += 8;

        const __m256 vk7x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 56)));
        vacc01234567p1 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi7x01234567, vk7x01234567, vacc01234567p1), _MM_FROUND_TO_NEAREST_INT));

        w += 64;

        // Add up all accumulators to vacc01234567p0
        vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_add_ps(vacc01234567p0, vacc01234567p1), _MM_FROUND_TO_NEAREST_INT));

        _mm_store_si128((__m128i*) b, _mm256_cvtps_ph(vacc01234567p0, _MM_FROUND_TO_NEAREST_INT));
        b += 8;
      }

      if (c != 0) {
        assert(c >= 1);
        assert(c <= 7);
        __m256 vacc01234567p0 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (b)));


        const __m256 vi0x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i0));

        const __m256 vk0x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w)));
        vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi0x01234567, vk0x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));

        const __m256 vi1x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i1));

        const __m256 vk1x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 8)));
        __m256 vacc01234567p1 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_mul_ps(vi1x01234567, vk1x01234567), _MM_FROUND_TO_NEAREST_INT));

        const __m256 vi2x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i2));

        const __m256 vk2x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 16)));
        vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi2x01234567, vk2x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));

        const __m256 vi3x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i3));

        const __m256 vk3x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 24)));
        vacc01234567p1 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi3x01234567, vk3x01234567, vacc01234567p1), _MM_FROUND_TO_NEAREST_INT));

        const __m256 vi4x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i4));

        const __m256 vk4x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 32)));
        vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi4x01234567, vk4x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));

        const __m256 vi5x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i5));

        const __m256 vk5x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 40)));
        vacc01234567p1 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi5x01234567, vk5x01234567, vacc01234567p1), _MM_FROUND_TO_NEAREST_INT));

        const __m256 vi6x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i6));

        const __m256 vk6x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 48)));
        vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi6x01234567, vk6x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));

        const __m256 vi7x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i7));

        const __m256 vk7x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 56)));
        vacc01234567p1 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi7x01234567, vk7x01234567, vacc01234567p1), _MM_FROUND_TO_NEAREST_INT));

        w += 64;

        // Add up all accumulators to vacc01234567p0
        vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_add_ps(vacc01234567p0, vacc01234567p1), _MM_FROUND_TO_NEAREST_INT));

        _mm_store_si128((__m128i*) b, _mm256_cvtps_ph(vacc01234567p0, _MM_FROUND_TO_NEAREST_INT));
      }
    }

    // Last pass to process up to 9 inputs.
    {
      uint16_t* b = (uint16_t*) buffer;
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

      size_t c = channels;


      for (; c >= 8; c -= 8) {
        __m256 vacc01234567p0 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (b)));
        b += 8;


        const __m256 vi0x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i0)));
        i0 += 8;

        __m256 vk0x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w)));

        vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi0x01234567, vk0x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));

        const __m256 vi1x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i1)));
        i1 += 8;

        __m256 vk1x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 8)));

        __m256 vacc01234567p1 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_mul_ps(vi1x01234567, vk1x01234567), _MM_FROUND_TO_NEAREST_INT));

        const __m256 vi2x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i2)));
        i2 += 8;

        __m256 vk2x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 16)));

        vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi2x01234567, vk2x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));

        const __m256 vi3x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i3)));
        i3 += 8;

        __m256 vk3x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 24)));

        vacc01234567p1 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi3x01234567, vk3x01234567, vacc01234567p1), _MM_FROUND_TO_NEAREST_INT));

        const __m256 vi4x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i4)));
        i4 += 8;

        __m256 vk4x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 32)));

        vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi4x01234567, vk4x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));

        const __m256 vi5x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i5)));
        i5 += 8;

        __m256 vk5x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 40)));

        vacc01234567p1 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi5x01234567, vk5x01234567, vacc01234567p1), _MM_FROUND_TO_NEAREST_INT));

        const __m256 vi6x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i6)));
        i6 += 8;

        __m256 vk6x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 48)));

        vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi6x01234567, vk6x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));

        const __m256 vi7x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i7)));
        i7 += 8;

        __m256 vk7x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 56)));

        vacc01234567p1 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi7x01234567, vk7x01234567, vacc01234567p1), _MM_FROUND_TO_NEAREST_INT));

        const __m256 vi8x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i8)));
        i8 += 8;

        __m256 vk8x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 64)));

        vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi8x01234567, vk8x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));

        w += 72;


        // Add up all accumulators to vacc01234567p0
        vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_add_ps(vacc01234567p0, vacc01234567p1), _MM_FROUND_TO_NEAREST_INT));

        __m256 vacc01234567 = _mm256_max_ps(vacc01234567p0, vmin);

        vacc01234567 = _mm256_min_ps(vacc01234567, vmax);

        _mm_storeu_si128((__m128i*) output, _mm256_cvtps_ph(vacc01234567, _MM_FROUND_TO_NEAREST_INT));
        output = (xnn_float16*) output + 8;
      }

      if XNN_UNLIKELY(c != 0) {
        assert(c >= 1);
        assert(c <= 7);
        __m256 vacc01234567p0 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (b)));

        const __m256 vi0x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i0));
        __m256 vk0x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w)));
        vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi0x01234567, vk0x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));

        const __m256 vi1x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i1));
        __m256 vk1x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 8)));
        __m256 vacc01234567p1 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_mul_ps(vi1x01234567, vk1x01234567), _MM_FROUND_TO_NEAREST_INT));

        const __m256 vi2x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i2));
        __m256 vk2x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 16)));
        vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi2x01234567, vk2x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));

        const __m256 vi3x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i3));
        __m256 vk3x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 24)));
        vacc01234567p1 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi3x01234567, vk3x01234567, vacc01234567p1), _MM_FROUND_TO_NEAREST_INT));

        const __m256 vi4x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i4));
        __m256 vk4x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 32)));
        vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi4x01234567, vk4x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));

        const __m256 vi5x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i5));
        __m256 vk5x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 40)));
        vacc01234567p1 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi5x01234567, vk5x01234567, vacc01234567p1), _MM_FROUND_TO_NEAREST_INT));

        const __m256 vi6x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i6));
        __m256 vk6x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 48)));
        vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi6x01234567, vk6x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));

        const __m256 vi7x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i7));
        __m256 vk7x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 56)));
        vacc01234567p1 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi7x01234567, vk7x01234567, vacc01234567p1), _MM_FROUND_TO_NEAREST_INT));

        const __m256 vi8x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i8));
        __m256 vk8x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 64)));
        vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi8x01234567, vk8x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));

        // Add up all accumulators to vacc01234567p0
        vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_add_ps(vacc01234567p0, vacc01234567p1), _MM_FROUND_TO_NEAREST_INT));

        __m256 vacc01234567 = _mm256_max_ps(vacc01234567p0, vmin);
        vacc01234567 = _mm256_min_ps(vacc01234567, vmax);

        __m128i vh01234567 = _mm256_cvtps_ph(vacc01234567, _MM_FROUND_TO_NEAREST_INT);
        if (c & 4) {
          _mm_storel_epi64((__m128i*) output, vh01234567);
          vh01234567 = _mm_unpackhi_epi64(vh01234567, vh01234567);
          output = (xnn_float16*) output + 4;
        }
        if (c & 2) {
          _mm_storeu_si32(output, vh01234567);
          vh01234567 = _mm_srli_epi64(vh01234567, 32);
          output = (xnn_float16*) output + 2;
        }
        if (c & 1) {
          *((uint16_t*) output) = (uint16_t) _mm_extract_epi16(vh01234567, 0);
          output = (xnn_float16*) output + 1;
        }
      }

    }
    input = (const xnn_float16**) ((uintptr_t) input + input_stride);
    output = (xnn_float16*) ((uintptr_t) output + output_increment);
  } while (--output_width != 0);
}
