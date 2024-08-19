// Copyright 2021 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Generator: tools/update-microkernels.py -a

#include <assert.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#ifndef M_LN2
#define M_LN2 0.69314718055994531
#endif  // M_LN2

#include <immintrin.h>

#include "xnnpack/common.h"
#include "xnnpack/dwconv.h"
#include "xnnpack/gemm.h"
#include "xnnpack/ibilinear.h"
#include "xnnpack/igemm.h"
#include "xnnpack/intrinsics-polyfill.h"
#include "xnnpack/math.h"
#include "xnnpack/microparams.h"
#include "xnnpack/prefetch.h"
#include "xnnpack/simd/f32-fma3.h"
#include "xnnpack/unaligned.h"
#include "xnnpack/vmulcaddc.h"
#include "xnnpack/vunary.h"


void xnn_f16_dwconv_minmax_ukernel_25p8c__fma3_acc2(
    size_t channels,
    size_t output_width,
    const void** input,
    const void* weights,
    void* output,
    intptr_t input_stride,
    size_t output_increment,
    size_t input_offset,
    const void* zero,
    const union xnn_f16_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(channels != 0);
  assert(output_width != 0);

  const __m256 vmax = _mm256_set1_ps(params->avx.max);
  const __m256 vmin = _mm256_set1_ps(params->avx.min);
  XNN_FORCE_REALIZATION(vmin);
  XNN_FORCE_REALIZATION(vmax);

  uint16_t* o = (uint16_t*) output;
  do {
    const uint16_t* i0 = input[0];
    assert(i0 != NULL);
    if XNN_UNPREDICTABLE(i0 != zero) {
      i0 = (const uint16_t*) ((uintptr_t) i0 + input_offset);
    }
    const uint16_t* i1 = input[1];
    assert(i1 != NULL);
    if XNN_UNPREDICTABLE(i1 != zero) {
      i1 = (const uint16_t*) ((uintptr_t) i1 + input_offset);
    }
    const uint16_t* i2 = input[2];
    assert(i2 != NULL);
    if XNN_UNPREDICTABLE(i2 != zero) {
      i2 = (const uint16_t*) ((uintptr_t) i2 + input_offset);
    }
    const uint16_t* i3 = input[3];
    assert(i3 != NULL);
    if XNN_UNPREDICTABLE(i3 != zero) {
      i3 = (const uint16_t*) ((uintptr_t) i3 + input_offset);
    }
    const uint16_t* i4 = input[4];
    assert(i4 != NULL);
    if XNN_UNPREDICTABLE(i4 != zero) {
      i4 = (const uint16_t*) ((uintptr_t) i4 + input_offset);
    }
    const uint16_t* i5 = input[5];
    assert(i5 != NULL);
    if XNN_UNPREDICTABLE(i5 != zero) {
      i5 = (const uint16_t*) ((uintptr_t) i5 + input_offset);
    }
    const uint16_t* i6 = input[6];
    assert(i6 != NULL);
    if XNN_UNPREDICTABLE(i6 != zero) {
      i6 = (const uint16_t*) ((uintptr_t) i6 + input_offset);
    }
    const uint16_t* i7 = input[7];
    assert(i7 != NULL);
    if XNN_UNPREDICTABLE(i7 != zero) {
      i7 = (const uint16_t*) ((uintptr_t) i7 + input_offset);
    }
    const uint16_t* i8 = input[8];
    assert(i8 != NULL);
    if XNN_UNPREDICTABLE(i8 != zero) {
      i8 = (const uint16_t*) ((uintptr_t) i8 + input_offset);
    }
    const uint16_t* i9 = input[9];
    assert(i9 != NULL);
    if XNN_UNPREDICTABLE(i9 != zero) {
      i9 = (const uint16_t*) ((uintptr_t) i9 + input_offset);
    }
    const uint16_t* i10 = input[10];
    assert(i10 != NULL);
    if XNN_UNPREDICTABLE(i10 != zero) {
      i10 = (const uint16_t*) ((uintptr_t) i10 + input_offset);
    }
    const uint16_t* i11 = input[11];
    assert(i11 != NULL);
    if XNN_UNPREDICTABLE(i11 != zero) {
      i11 = (const uint16_t*) ((uintptr_t) i11 + input_offset);
    }
    const uint16_t* i12 = input[12];
    assert(i12 != NULL);
    if XNN_UNPREDICTABLE(i12 != zero) {
      i12 = (const uint16_t*) ((uintptr_t) i12 + input_offset);
    }
    const uint16_t* i13 = input[13];
    assert(i13 != NULL);
    if XNN_UNPREDICTABLE(i13 != zero) {
      i13 = (const uint16_t*) ((uintptr_t) i13 + input_offset);
    }
    const uint16_t* i14 = input[14];
    assert(i14 != NULL);
    if XNN_UNPREDICTABLE(i14 != zero) {
      i14 = (const uint16_t*) ((uintptr_t) i14 + input_offset);
    }
    const uint16_t* i15 = input[15];
    assert(i15 != NULL);
    if XNN_UNPREDICTABLE(i15 != zero) {
      i15 = (const uint16_t*) ((uintptr_t) i15 + input_offset);
    }
    const uint16_t* i16 = input[16];
    assert(i16 != NULL);
    if XNN_UNPREDICTABLE(i16 != zero) {
      i16 = (const uint16_t*) ((uintptr_t) i16 + input_offset);
    }
    const uint16_t* i17 = input[17];
    assert(i17 != NULL);
    if XNN_UNPREDICTABLE(i17 != zero) {
      i17 = (const uint16_t*) ((uintptr_t) i17 + input_offset);
    }
    const uint16_t* i18 = input[18];
    assert(i18 != NULL);
    if XNN_UNPREDICTABLE(i18 != zero) {
      i18 = (const uint16_t*) ((uintptr_t) i18 + input_offset);
    }
    const uint16_t* i19 = input[19];
    assert(i19 != NULL);
    if XNN_UNPREDICTABLE(i19 != zero) {
      i19 = (const uint16_t*) ((uintptr_t) i19 + input_offset);
    }
    const uint16_t* i20 = input[20];
    assert(i20 != NULL);
    if XNN_UNPREDICTABLE(i20 != zero) {
      i20 = (const uint16_t*) ((uintptr_t) i20 + input_offset);
    }
    const uint16_t* i21 = input[21];
    assert(i21 != NULL);
    if XNN_UNPREDICTABLE(i21 != zero) {
      i21 = (const uint16_t*) ((uintptr_t) i21 + input_offset);
    }
    const uint16_t* i22 = input[22];
    assert(i22 != NULL);
    if XNN_UNPREDICTABLE(i22 != zero) {
      i22 = (const uint16_t*) ((uintptr_t) i22 + input_offset);
    }
    const uint16_t* i23 = input[23];
    assert(i23 != NULL);
    if XNN_UNPREDICTABLE(i23 != zero) {
      i23 = (const uint16_t*) ((uintptr_t) i23 + input_offset);
    }
    const uint16_t* i24 = input[24];
    assert(i24 != NULL);
    if XNN_UNPREDICTABLE(i24 != zero) {
      i24 = (const uint16_t*) ((uintptr_t) i24 + input_offset);
    }
    input = (const void**) ((uintptr_t) input + input_stride);

    size_t c = channels;
    const uint16_t* w = weights;
    for (; c >= 8; c -= 8) {
      __m256 vacc01234567p0 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) w));


      const __m256 vi0x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i0));
      i0 += 8;

      const __m256 vk0x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (w + 8)));
      vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi0x01234567, vk0x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));

      const __m256 vi1x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i1));
      i1 += 8;

      const __m256 vk1x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (w + 16)));
      __m256 vacc01234567p1 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_mul_ps(vi1x01234567, vk1x01234567), _MM_FROUND_TO_NEAREST_INT));

      const __m256 vi2x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i2));
      i2 += 8;

      const __m256 vk2x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (w + 24)));
      vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi2x01234567, vk2x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));

      const __m256 vi3x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i3));
      i3 += 8;

      const __m256 vk3x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (w + 32)));
      vacc01234567p1 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi3x01234567, vk3x01234567, vacc01234567p1), _MM_FROUND_TO_NEAREST_INT));

      const __m256 vi4x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i4));
      i4 += 8;

      const __m256 vk4x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (w + 40)));
      vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi4x01234567, vk4x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));

      const __m256 vi5x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i5));
      i5 += 8;

      const __m256 vk5x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (w + 48)));
      vacc01234567p1 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi5x01234567, vk5x01234567, vacc01234567p1), _MM_FROUND_TO_NEAREST_INT));

      const __m256 vi6x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i6));
      i6 += 8;

      const __m256 vk6x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (w + 56)));
      vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi6x01234567, vk6x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));

      const __m256 vi7x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i7));
      i7 += 8;

      const __m256 vk7x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (w + 64)));
      vacc01234567p1 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi7x01234567, vk7x01234567, vacc01234567p1), _MM_FROUND_TO_NEAREST_INT));

      const __m256 vi8x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i8));
      i8 += 8;

      const __m256 vk8x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (w + 72)));
      vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi8x01234567, vk8x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));

      const __m256 vi9x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i9));
      i9 += 8;

      const __m256 vk9x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (w + 80)));
      vacc01234567p1 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi9x01234567, vk9x01234567, vacc01234567p1), _MM_FROUND_TO_NEAREST_INT));

      const __m256 vi10x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i10));
      i10 += 8;

      const __m256 vk10x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (w + 88)));
      vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi10x01234567, vk10x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));

      const __m256 vi11x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i11));
      i11 += 8;

      const __m256 vk11x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (w + 96)));
      vacc01234567p1 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi11x01234567, vk11x01234567, vacc01234567p1), _MM_FROUND_TO_NEAREST_INT));

      const __m256 vi12x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i12));
      i12 += 8;

      const __m256 vk12x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (w + 104)));
      vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi12x01234567, vk12x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));

      const __m256 vi13x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i13));
      i13 += 8;

      const __m256 vk13x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (w + 112)));
      vacc01234567p1 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi13x01234567, vk13x01234567, vacc01234567p1), _MM_FROUND_TO_NEAREST_INT));

      const __m256 vi14x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i14));
      i14 += 8;

      const __m256 vk14x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (w + 120)));
      vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi14x01234567, vk14x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));

      const __m256 vi15x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i15));
      i15 += 8;

      const __m256 vk15x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (w + 128)));
      vacc01234567p1 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi15x01234567, vk15x01234567, vacc01234567p1), _MM_FROUND_TO_NEAREST_INT));

      const __m256 vi16x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i16));
      i16 += 8;

      const __m256 vk16x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (w + 136)));
      vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi16x01234567, vk16x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));

      const __m256 vi17x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i17));
      i17 += 8;

      const __m256 vk17x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (w + 144)));
      vacc01234567p1 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi17x01234567, vk17x01234567, vacc01234567p1), _MM_FROUND_TO_NEAREST_INT));

      const __m256 vi18x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i18));
      i18 += 8;

      const __m256 vk18x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (w + 152)));
      vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi18x01234567, vk18x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));

      const __m256 vi19x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i19));
      i19 += 8;

      const __m256 vk19x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (w + 160)));
      vacc01234567p1 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi19x01234567, vk19x01234567, vacc01234567p1), _MM_FROUND_TO_NEAREST_INT));

      const __m256 vi20x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i20));
      i20 += 8;

      const __m256 vk20x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (w + 168)));
      vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi20x01234567, vk20x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));

      const __m256 vi21x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i21));
      i21 += 8;

      const __m256 vk21x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (w + 176)));
      vacc01234567p1 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi21x01234567, vk21x01234567, vacc01234567p1), _MM_FROUND_TO_NEAREST_INT));

      const __m256 vi22x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i22));
      i22 += 8;

      const __m256 vk22x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (w + 184)));
      vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi22x01234567, vk22x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));

      const __m256 vi23x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i23));
      i23 += 8;

      const __m256 vk23x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (w + 192)));
      vacc01234567p1 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi23x01234567, vk23x01234567, vacc01234567p1), _MM_FROUND_TO_NEAREST_INT));

      const __m256 vi24x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i24));
      i24 += 8;

      const __m256 vk24x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (w + 200)));
      vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi24x01234567, vk24x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));

      w += 208;

      // Add up all accumulators to vacc01234567p0
      vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_add_ps(vacc01234567p0, vacc01234567p1), _MM_FROUND_TO_NEAREST_INT));

      __m256 vacc01234567 = _mm256_max_ps(vacc01234567p0, vmin);
      vacc01234567 = _mm256_min_ps(vacc01234567, vmax);

      _mm_storeu_si128((__m128i*) o, _mm256_cvtps_ph(vacc01234567, _MM_FROUND_TO_NEAREST_INT));
      o += 8;
    }
    if XNN_UNLIKELY(c != 0) {
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

      const __m256 vi8x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i8));

      const __m256 vk8x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 72)));
      vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi8x01234567, vk8x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));

      const __m256 vi9x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i9));

      const __m256 vk9x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 80)));
      vacc01234567p1 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi9x01234567, vk9x01234567, vacc01234567p1), _MM_FROUND_TO_NEAREST_INT));

      const __m256 vi10x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i10));

      const __m256 vk10x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 88)));
      vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi10x01234567, vk10x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));

      const __m256 vi11x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i11));

      const __m256 vk11x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 96)));
      vacc01234567p1 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi11x01234567, vk11x01234567, vacc01234567p1), _MM_FROUND_TO_NEAREST_INT));

      const __m256 vi12x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i12));

      const __m256 vk12x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 104)));
      vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi12x01234567, vk12x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));

      const __m256 vi13x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i13));

      const __m256 vk13x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 112)));
      vacc01234567p1 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi13x01234567, vk13x01234567, vacc01234567p1), _MM_FROUND_TO_NEAREST_INT));

      const __m256 vi14x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i14));

      const __m256 vk14x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 120)));
      vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi14x01234567, vk14x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));

      const __m256 vi15x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i15));

      const __m256 vk15x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 128)));
      vacc01234567p1 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi15x01234567, vk15x01234567, vacc01234567p1), _MM_FROUND_TO_NEAREST_INT));

      const __m256 vi16x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i16));

      const __m256 vk16x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 136)));
      vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi16x01234567, vk16x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));

      const __m256 vi17x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i17));

      const __m256 vk17x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 144)));
      vacc01234567p1 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi17x01234567, vk17x01234567, vacc01234567p1), _MM_FROUND_TO_NEAREST_INT));

      const __m256 vi18x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i18));

      const __m256 vk18x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 152)));
      vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi18x01234567, vk18x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));

      const __m256 vi19x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i19));

      const __m256 vk19x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 160)));
      vacc01234567p1 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi19x01234567, vk19x01234567, vacc01234567p1), _MM_FROUND_TO_NEAREST_INT));

      const __m256 vi20x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i20));

      const __m256 vk20x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 168)));
      vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi20x01234567, vk20x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));

      const __m256 vi21x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i21));

      const __m256 vk21x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 176)));
      vacc01234567p1 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi21x01234567, vk21x01234567, vacc01234567p1), _MM_FROUND_TO_NEAREST_INT));

      const __m256 vi22x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i22));

      const __m256 vk22x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 184)));
      vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi22x01234567, vk22x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));

      const __m256 vi23x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i23));

      const __m256 vk23x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 192)));
      vacc01234567p1 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi23x01234567, vk23x01234567, vacc01234567p1), _MM_FROUND_TO_NEAREST_INT));

      const __m256 vi24x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i24));

      const __m256 vk24x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 200)));
      vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi24x01234567, vk24x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));

      // Add up all accumulators to vacc01234567p0
      vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_add_ps(vacc01234567p0, vacc01234567p1), _MM_FROUND_TO_NEAREST_INT));

      __m256 vacc01234567 = _mm256_max_ps(vacc01234567p0, vmin);
      vacc01234567 = _mm256_min_ps(vacc01234567, vmax);

      __m128i vh01234567 = _mm256_cvtps_ph(vacc01234567, _MM_FROUND_TO_NEAREST_INT);
      if (c & 4) {
        _mm_storel_epi64((__m128i*) o, vh01234567);
        vh01234567 = _mm_unpackhi_epi64(vh01234567, vh01234567);
        o += 4;
      }
      if (c & 2) {
        _mm_storeu_si32(o, vh01234567);
        vh01234567 = _mm_srli_epi64(vh01234567, 32);
        o += 2;
      }
      if (c & 1) {
        *o = (uint16_t) _mm_extract_epi16(vh01234567, 0);
        o += 1;
      }
    }

    o = (uint16_t*) ((uintptr_t) o + output_increment);
  } while (--output_width != 0);
}

void xnn_f16_dwconv_minmax_ukernel_3p16c__fma3(
    size_t channels,
    size_t output_width,
    const void** input,
    const void* weights,
    void* output,
    intptr_t input_stride,
    size_t output_increment,
    size_t input_offset,
    const void* zero,
    const union xnn_f16_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(channels != 0);
  assert(output_width != 0);

  const __m256 vmax = _mm256_set1_ps(params->avx.max);
  const __m256 vmin = _mm256_set1_ps(params->avx.min);
  XNN_FORCE_REALIZATION(vmin);
  XNN_FORCE_REALIZATION(vmax);

  uint16_t* o = (uint16_t*) output;
  do {
    const uint16_t* i0 = input[0];
    assert(i0 != NULL);
    if XNN_UNPREDICTABLE(i0 != zero) {
      i0 = (const uint16_t*) ((uintptr_t) i0 + input_offset);
    }
    const uint16_t* i1 = input[1];
    assert(i1 != NULL);
    if XNN_UNPREDICTABLE(i1 != zero) {
      i1 = (const uint16_t*) ((uintptr_t) i1 + input_offset);
    }
    const uint16_t* i2 = input[2];
    assert(i2 != NULL);
    if XNN_UNPREDICTABLE(i2 != zero) {
      i2 = (const uint16_t*) ((uintptr_t) i2 + input_offset);
    }
    input = (const void**) ((uintptr_t) input + input_stride);

    size_t c = channels;
    const uint16_t* w = weights;
    for (; c >= 16; c -= 16) {
      __m256 vacc01234567p0 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) w));
      __m256 vacc89ABCDEFp0 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 8)));


      const __m256 vi0x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i0));
      const __m256 vi0x89ABCDEF = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i0 + 8)));
      i0 += 16;

      const __m256 vk0x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (w + 16)));
      const __m256 vk0x89ABCDEF = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (w + 24)));
      vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi0x01234567, vk0x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));
      vacc89ABCDEFp0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi0x89ABCDEF, vk0x89ABCDEF, vacc89ABCDEFp0), _MM_FROUND_TO_NEAREST_INT));

      const __m256 vi1x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i1));
      const __m256 vi1x89ABCDEF = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i1 + 8)));
      i1 += 16;

      const __m256 vk1x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (w + 32)));
      const __m256 vk1x89ABCDEF = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (w + 40)));
      vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi1x01234567, vk1x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));
      vacc89ABCDEFp0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi1x89ABCDEF, vk1x89ABCDEF, vacc89ABCDEFp0), _MM_FROUND_TO_NEAREST_INT));

      const __m256 vi2x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i2));
      const __m256 vi2x89ABCDEF = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i2 + 8)));
      i2 += 16;

      const __m256 vk2x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (w + 48)));
      const __m256 vk2x89ABCDEF = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (w + 56)));
      vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi2x01234567, vk2x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));
      vacc89ABCDEFp0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi2x89ABCDEF, vk2x89ABCDEF, vacc89ABCDEFp0), _MM_FROUND_TO_NEAREST_INT));

      w += 64;


      __m256 vacc01234567 = _mm256_max_ps(vacc01234567p0, vmin);
      __m256 vacc89ABCDEF = _mm256_max_ps(vacc89ABCDEFp0, vmin);
      vacc01234567 = _mm256_min_ps(vacc01234567, vmax);
      vacc89ABCDEF = _mm256_min_ps(vacc89ABCDEF, vmax);

      _mm_storeu_si128((__m128i*) o, _mm256_cvtps_ph(vacc01234567, _MM_FROUND_TO_NEAREST_INT));
      _mm_storeu_si128((__m128i*) (o + 8), _mm256_cvtps_ph(vacc89ABCDEF, _MM_FROUND_TO_NEAREST_INT));
      o += 16;
    }
    for (; c >= 8; c -= 8) {
      __m256 vacc01234567p0 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) w));

      const __m256 vi0x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i0));
      i0 += 8;

      const __m256 vk0x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 16)));
      vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi0x01234567, vk0x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));

      const __m256 vi1x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i1));
      i1 += 8;

      const __m256 vk1x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 32)));
      vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi1x01234567, vk1x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));

      const __m256 vi2x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i2));
      i2 += 8;

      const __m256 vk2x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 48)));
      vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi2x01234567, vk2x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));

      w += 8;


      __m256 vacc01234567 = _mm256_max_ps(vacc01234567p0, vmin);
      vacc01234567 = _mm256_min_ps(vacc01234567, vmax);

      _mm_storeu_si128((__m128i*) o, _mm256_cvtps_ph(vacc01234567, _MM_FROUND_TO_NEAREST_INT));
      o += 8;
    }
    if XNN_UNLIKELY(c != 0) {
      assert(c >= 1);
      assert(c <= 7);

      __m256 vacc01234567p0 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) w));

      const __m256 vi0x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i0));

      const __m256 vk0x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 16)));
      vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi0x01234567, vk0x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));

      const __m256 vi1x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i1));

      const __m256 vk1x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 32)));
      vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi1x01234567, vk1x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));

      const __m256 vi2x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i2));

      const __m256 vk2x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 48)));
      vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi2x01234567, vk2x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));


      __m256 vacc01234567 = _mm256_max_ps(vacc01234567p0, vmin);
      vacc01234567 = _mm256_min_ps(vacc01234567, vmax);

      __m128i vh01234567 = _mm256_cvtps_ph(vacc01234567, _MM_FROUND_TO_NEAREST_INT);
      if (c & 4) {
        _mm_storel_epi64((__m128i*) o, vh01234567);
        vh01234567 = _mm_unpackhi_epi64(vh01234567, vh01234567);
        o += 4;
      }
      if (c & 2) {
        _mm_storeu_si32(o, vh01234567);
        vh01234567 = _mm_srli_epi64(vh01234567, 32);
        o += 2;
      }
      if (c & 1) {
        *o = (uint16_t) _mm_extract_epi16(vh01234567, 0);
        o += 1;
      }
    }

    o = (uint16_t*) ((uintptr_t) o + output_increment);
  } while (--output_width != 0);
}

void xnn_f16_dwconv_minmax_ukernel_4p16c__fma3(
    size_t channels,
    size_t output_width,
    const void** input,
    const void* weights,
    void* output,
    intptr_t input_stride,
    size_t output_increment,
    size_t input_offset,
    const void* zero,
    const union xnn_f16_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(channels != 0);
  assert(output_width != 0);

  const __m256 vmax = _mm256_set1_ps(params->avx.max);
  const __m256 vmin = _mm256_set1_ps(params->avx.min);
  XNN_FORCE_REALIZATION(vmin);
  XNN_FORCE_REALIZATION(vmax);

  uint16_t* o = (uint16_t*) output;
  do {
    const uint16_t* i0 = input[0];
    assert(i0 != NULL);
    if XNN_UNPREDICTABLE(i0 != zero) {
      i0 = (const uint16_t*) ((uintptr_t) i0 + input_offset);
    }
    const uint16_t* i1 = input[1];
    assert(i1 != NULL);
    if XNN_UNPREDICTABLE(i1 != zero) {
      i1 = (const uint16_t*) ((uintptr_t) i1 + input_offset);
    }
    const uint16_t* i2 = input[2];
    assert(i2 != NULL);
    if XNN_UNPREDICTABLE(i2 != zero) {
      i2 = (const uint16_t*) ((uintptr_t) i2 + input_offset);
    }
    const uint16_t* i3 = input[3];
    assert(i3 != NULL);
    if XNN_UNPREDICTABLE(i3 != zero) {
      i3 = (const uint16_t*) ((uintptr_t) i3 + input_offset);
    }
    input = (const void**) ((uintptr_t) input + input_stride);

    size_t c = channels;
    const uint16_t* w = weights;
    for (; c >= 16; c -= 16) {
      __m256 vacc01234567p0 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) w));
      __m256 vacc89ABCDEFp0 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 8)));


      const __m256 vi0x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i0));
      const __m256 vi0x89ABCDEF = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i0 + 8)));
      i0 += 16;

      const __m256 vk0x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (w + 16)));
      const __m256 vk0x89ABCDEF = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (w + 24)));
      vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi0x01234567, vk0x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));
      vacc89ABCDEFp0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi0x89ABCDEF, vk0x89ABCDEF, vacc89ABCDEFp0), _MM_FROUND_TO_NEAREST_INT));

      const __m256 vi1x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i1));
      const __m256 vi1x89ABCDEF = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i1 + 8)));
      i1 += 16;

      const __m256 vk1x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (w + 32)));
      const __m256 vk1x89ABCDEF = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (w + 40)));
      vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi1x01234567, vk1x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));
      vacc89ABCDEFp0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi1x89ABCDEF, vk1x89ABCDEF, vacc89ABCDEFp0), _MM_FROUND_TO_NEAREST_INT));

      const __m256 vi2x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i2));
      const __m256 vi2x89ABCDEF = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i2 + 8)));
      i2 += 16;

      const __m256 vk2x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (w + 48)));
      const __m256 vk2x89ABCDEF = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (w + 56)));
      vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi2x01234567, vk2x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));
      vacc89ABCDEFp0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi2x89ABCDEF, vk2x89ABCDEF, vacc89ABCDEFp0), _MM_FROUND_TO_NEAREST_INT));

      const __m256 vi3x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i3));
      const __m256 vi3x89ABCDEF = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i3 + 8)));
      i3 += 16;

      const __m256 vk3x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (w + 64)));
      const __m256 vk3x89ABCDEF = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (w + 72)));
      vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi3x01234567, vk3x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));
      vacc89ABCDEFp0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi3x89ABCDEF, vk3x89ABCDEF, vacc89ABCDEFp0), _MM_FROUND_TO_NEAREST_INT));

      w += 80;


      __m256 vacc01234567 = _mm256_max_ps(vacc01234567p0, vmin);
      __m256 vacc89ABCDEF = _mm256_max_ps(vacc89ABCDEFp0, vmin);
      vacc01234567 = _mm256_min_ps(vacc01234567, vmax);
      vacc89ABCDEF = _mm256_min_ps(vacc89ABCDEF, vmax);

      _mm_storeu_si128((__m128i*) o, _mm256_cvtps_ph(vacc01234567, _MM_FROUND_TO_NEAREST_INT));
      _mm_storeu_si128((__m128i*) (o + 8), _mm256_cvtps_ph(vacc89ABCDEF, _MM_FROUND_TO_NEAREST_INT));
      o += 16;
    }
    for (; c >= 8; c -= 8) {
      __m256 vacc01234567p0 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) w));

      const __m256 vi0x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i0));
      i0 += 8;

      const __m256 vk0x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 16)));
      vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi0x01234567, vk0x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));

      const __m256 vi1x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i1));
      i1 += 8;

      const __m256 vk1x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 32)));
      vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi1x01234567, vk1x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));

      const __m256 vi2x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i2));
      i2 += 8;

      const __m256 vk2x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 48)));
      vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi2x01234567, vk2x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));

      const __m256 vi3x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i3));
      i3 += 8;

      const __m256 vk3x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 64)));
      vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi3x01234567, vk3x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));

      w += 8;


      __m256 vacc01234567 = _mm256_max_ps(vacc01234567p0, vmin);
      vacc01234567 = _mm256_min_ps(vacc01234567, vmax);

      _mm_storeu_si128((__m128i*) o, _mm256_cvtps_ph(vacc01234567, _MM_FROUND_TO_NEAREST_INT));
      o += 8;
    }
    if XNN_UNLIKELY(c != 0) {
      assert(c >= 1);
      assert(c <= 7);

      __m256 vacc01234567p0 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) w));

      const __m256 vi0x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i0));

      const __m256 vk0x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 16)));
      vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi0x01234567, vk0x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));

      const __m256 vi1x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i1));

      const __m256 vk1x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 32)));
      vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi1x01234567, vk1x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));

      const __m256 vi2x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i2));

      const __m256 vk2x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 48)));
      vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi2x01234567, vk2x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));

      const __m256 vi3x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i3));

      const __m256 vk3x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 64)));
      vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi3x01234567, vk3x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));


      __m256 vacc01234567 = _mm256_max_ps(vacc01234567p0, vmin);
      vacc01234567 = _mm256_min_ps(vacc01234567, vmax);

      __m128i vh01234567 = _mm256_cvtps_ph(vacc01234567, _MM_FROUND_TO_NEAREST_INT);
      if (c & 4) {
        _mm_storel_epi64((__m128i*) o, vh01234567);
        vh01234567 = _mm_unpackhi_epi64(vh01234567, vh01234567);
        o += 4;
      }
      if (c & 2) {
        _mm_storeu_si32(o, vh01234567);
        vh01234567 = _mm_srli_epi64(vh01234567, 32);
        o += 2;
      }
      if (c & 1) {
        *o = (uint16_t) _mm_extract_epi16(vh01234567, 0);
        o += 1;
      }
    }

    o = (uint16_t*) ((uintptr_t) o + output_increment);
  } while (--output_width != 0);
}

void xnn_f16_dwconv_minmax_ukernel_9p16c__fma3(
    size_t channels,
    size_t output_width,
    const void** input,
    const void* weights,
    void* output,
    intptr_t input_stride,
    size_t output_increment,
    size_t input_offset,
    const void* zero,
    const union xnn_f16_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(channels != 0);
  assert(output_width != 0);

  const __m256 vmax = _mm256_set1_ps(params->avx.max);
  const __m256 vmin = _mm256_set1_ps(params->avx.min);
  XNN_FORCE_REALIZATION(vmin);
  XNN_FORCE_REALIZATION(vmax);

  uint16_t* o = (uint16_t*) output;
  do {
    const uint16_t* i0 = input[0];
    assert(i0 != NULL);
    if XNN_UNPREDICTABLE(i0 != zero) {
      i0 = (const uint16_t*) ((uintptr_t) i0 + input_offset);
    }
    const uint16_t* i1 = input[1];
    assert(i1 != NULL);
    if XNN_UNPREDICTABLE(i1 != zero) {
      i1 = (const uint16_t*) ((uintptr_t) i1 + input_offset);
    }
    const uint16_t* i2 = input[2];
    assert(i2 != NULL);
    if XNN_UNPREDICTABLE(i2 != zero) {
      i2 = (const uint16_t*) ((uintptr_t) i2 + input_offset);
    }
    const uint16_t* i3 = input[3];
    assert(i3 != NULL);
    if XNN_UNPREDICTABLE(i3 != zero) {
      i3 = (const uint16_t*) ((uintptr_t) i3 + input_offset);
    }
    const uint16_t* i4 = input[4];
    assert(i4 != NULL);
    if XNN_UNPREDICTABLE(i4 != zero) {
      i4 = (const uint16_t*) ((uintptr_t) i4 + input_offset);
    }
    const uint16_t* i5 = input[5];
    assert(i5 != NULL);
    if XNN_UNPREDICTABLE(i5 != zero) {
      i5 = (const uint16_t*) ((uintptr_t) i5 + input_offset);
    }
    const uint16_t* i6 = input[6];
    assert(i6 != NULL);
    if XNN_UNPREDICTABLE(i6 != zero) {
      i6 = (const uint16_t*) ((uintptr_t) i6 + input_offset);
    }
    const uint16_t* i7 = input[7];
    assert(i7 != NULL);
    if XNN_UNPREDICTABLE(i7 != zero) {
      i7 = (const uint16_t*) ((uintptr_t) i7 + input_offset);
    }
    const uint16_t* i8 = input[8];
    assert(i8 != NULL);
    if XNN_UNPREDICTABLE(i8 != zero) {
      i8 = (const uint16_t*) ((uintptr_t) i8 + input_offset);
    }
    input = (const void**) ((uintptr_t) input + input_stride);

    size_t c = channels;
    const uint16_t* w = weights;
    for (; c >= 16; c -= 16) {
      __m256 vacc01234567p0 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) w));
      __m256 vacc89ABCDEFp0 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 8)));


      const __m256 vi0x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i0));
      const __m256 vi0x89ABCDEF = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i0 + 8)));
      i0 += 16;

      const __m256 vk0x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (w + 16)));
      const __m256 vk0x89ABCDEF = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (w + 24)));
      vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi0x01234567, vk0x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));
      vacc89ABCDEFp0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi0x89ABCDEF, vk0x89ABCDEF, vacc89ABCDEFp0), _MM_FROUND_TO_NEAREST_INT));

      const __m256 vi1x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i1));
      const __m256 vi1x89ABCDEF = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i1 + 8)));
      i1 += 16;

      const __m256 vk1x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (w + 32)));
      const __m256 vk1x89ABCDEF = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (w + 40)));
      vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi1x01234567, vk1x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));
      vacc89ABCDEFp0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi1x89ABCDEF, vk1x89ABCDEF, vacc89ABCDEFp0), _MM_FROUND_TO_NEAREST_INT));

      const __m256 vi2x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i2));
      const __m256 vi2x89ABCDEF = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i2 + 8)));
      i2 += 16;

      const __m256 vk2x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (w + 48)));
      const __m256 vk2x89ABCDEF = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (w + 56)));
      vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi2x01234567, vk2x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));
      vacc89ABCDEFp0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi2x89ABCDEF, vk2x89ABCDEF, vacc89ABCDEFp0), _MM_FROUND_TO_NEAREST_INT));

      const __m256 vi3x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i3));
      const __m256 vi3x89ABCDEF = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i3 + 8)));
      i3 += 16;

      const __m256 vk3x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (w + 64)));
      const __m256 vk3x89ABCDEF = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (w + 72)));
      vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi3x01234567, vk3x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));
      vacc89ABCDEFp0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi3x89ABCDEF, vk3x89ABCDEF, vacc89ABCDEFp0), _MM_FROUND_TO_NEAREST_INT));

      const __m256 vi4x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i4));
      const __m256 vi4x89ABCDEF = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i4 + 8)));
      i4 += 16;

      const __m256 vk4x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (w + 80)));
      const __m256 vk4x89ABCDEF = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (w + 88)));
      vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi4x01234567, vk4x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));
      vacc89ABCDEFp0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi4x89ABCDEF, vk4x89ABCDEF, vacc89ABCDEFp0), _MM_FROUND_TO_NEAREST_INT));

      const __m256 vi5x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i5));
      const __m256 vi5x89ABCDEF = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i5 + 8)));
      i5 += 16;

      const __m256 vk5x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (w + 96)));
      const __m256 vk5x89ABCDEF = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (w + 104)));
      vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi5x01234567, vk5x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));
      vacc89ABCDEFp0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi5x89ABCDEF, vk5x89ABCDEF, vacc89ABCDEFp0), _MM_FROUND_TO_NEAREST_INT));

      const __m256 vi6x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i6));
      const __m256 vi6x89ABCDEF = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i6 + 8)));
      i6 += 16;

      const __m256 vk6x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (w + 112)));
      const __m256 vk6x89ABCDEF = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (w + 120)));
      vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi6x01234567, vk6x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));
      vacc89ABCDEFp0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi6x89ABCDEF, vk6x89ABCDEF, vacc89ABCDEFp0), _MM_FROUND_TO_NEAREST_INT));

      const __m256 vi7x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i7));
      const __m256 vi7x89ABCDEF = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i7 + 8)));
      i7 += 16;

      const __m256 vk7x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (w + 128)));
      const __m256 vk7x89ABCDEF = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (w + 136)));
      vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi7x01234567, vk7x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));
      vacc89ABCDEFp0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi7x89ABCDEF, vk7x89ABCDEF, vacc89ABCDEFp0), _MM_FROUND_TO_NEAREST_INT));

      const __m256 vi8x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i8));
      const __m256 vi8x89ABCDEF = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i8 + 8)));
      i8 += 16;

      const __m256 vk8x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (w + 144)));
      const __m256 vk8x89ABCDEF = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (w + 152)));
      vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi8x01234567, vk8x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));
      vacc89ABCDEFp0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi8x89ABCDEF, vk8x89ABCDEF, vacc89ABCDEFp0), _MM_FROUND_TO_NEAREST_INT));

      w += 160;


      __m256 vacc01234567 = _mm256_max_ps(vacc01234567p0, vmin);
      __m256 vacc89ABCDEF = _mm256_max_ps(vacc89ABCDEFp0, vmin);
      vacc01234567 = _mm256_min_ps(vacc01234567, vmax);
      vacc89ABCDEF = _mm256_min_ps(vacc89ABCDEF, vmax);

      _mm_storeu_si128((__m128i*) o, _mm256_cvtps_ph(vacc01234567, _MM_FROUND_TO_NEAREST_INT));
      _mm_storeu_si128((__m128i*) (o + 8), _mm256_cvtps_ph(vacc89ABCDEF, _MM_FROUND_TO_NEAREST_INT));
      o += 16;
    }
    for (; c >= 8; c -= 8) {
      __m256 vacc01234567p0 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) w));

      const __m256 vi0x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i0));
      i0 += 8;

      const __m256 vk0x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 16)));
      vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi0x01234567, vk0x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));

      const __m256 vi1x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i1));
      i1 += 8;

      const __m256 vk1x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 32)));
      vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi1x01234567, vk1x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));

      const __m256 vi2x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i2));
      i2 += 8;

      const __m256 vk2x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 48)));
      vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi2x01234567, vk2x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));

      const __m256 vi3x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i3));
      i3 += 8;

      const __m256 vk3x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 64)));
      vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi3x01234567, vk3x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));

      const __m256 vi4x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i4));
      i4 += 8;

      const __m256 vk4x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 80)));
      vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi4x01234567, vk4x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));

      const __m256 vi5x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i5));
      i5 += 8;

      const __m256 vk5x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 96)));
      vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi5x01234567, vk5x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));

      const __m256 vi6x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i6));
      i6 += 8;

      const __m256 vk6x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 112)));
      vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi6x01234567, vk6x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));

      const __m256 vi7x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i7));
      i7 += 8;

      const __m256 vk7x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 128)));
      vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi7x01234567, vk7x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));

      const __m256 vi8x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i8));
      i8 += 8;

      const __m256 vk8x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 144)));
      vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi8x01234567, vk8x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));

      w += 8;


      __m256 vacc01234567 = _mm256_max_ps(vacc01234567p0, vmin);
      vacc01234567 = _mm256_min_ps(vacc01234567, vmax);

      _mm_storeu_si128((__m128i*) o, _mm256_cvtps_ph(vacc01234567, _MM_FROUND_TO_NEAREST_INT));
      o += 8;
    }
    if XNN_UNLIKELY(c != 0) {
      assert(c >= 1);
      assert(c <= 7);

      __m256 vacc01234567p0 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) w));

      const __m256 vi0x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i0));

      const __m256 vk0x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 16)));
      vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi0x01234567, vk0x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));

      const __m256 vi1x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i1));

      const __m256 vk1x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 32)));
      vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi1x01234567, vk1x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));

      const __m256 vi2x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i2));

      const __m256 vk2x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 48)));
      vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi2x01234567, vk2x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));

      const __m256 vi3x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i3));

      const __m256 vk3x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 64)));
      vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi3x01234567, vk3x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));

      const __m256 vi4x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i4));

      const __m256 vk4x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 80)));
      vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi4x01234567, vk4x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));

      const __m256 vi5x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i5));

      const __m256 vk5x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 96)));
      vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi5x01234567, vk5x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));

      const __m256 vi6x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i6));

      const __m256 vk6x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 112)));
      vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi6x01234567, vk6x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));

      const __m256 vi7x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i7));

      const __m256 vk7x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 128)));
      vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi7x01234567, vk7x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));

      const __m256 vi8x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i8));

      const __m256 vk8x01234567 = _mm256_cvtph_ps(_mm_load_si128((const __m128i*) (w + 144)));
      vacc01234567p0 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vi8x01234567, vk8x01234567, vacc01234567p0), _MM_FROUND_TO_NEAREST_INT));


      __m256 vacc01234567 = _mm256_max_ps(vacc01234567p0, vmin);
      vacc01234567 = _mm256_min_ps(vacc01234567, vmax);

      __m128i vh01234567 = _mm256_cvtps_ph(vacc01234567, _MM_FROUND_TO_NEAREST_INT);
      if (c & 4) {
        _mm_storel_epi64((__m128i*) o, vh01234567);
        vh01234567 = _mm_unpackhi_epi64(vh01234567, vh01234567);
        o += 4;
      }
      if (c & 2) {
        _mm_storeu_si32(o, vh01234567);
        vh01234567 = _mm_srli_epi64(vh01234567, 32);
        o += 2;
      }
      if (c & 1) {
        *o = (uint16_t) _mm_extract_epi16(vh01234567, 0);
        o += 1;
      }
    }

    o = (uint16_t*) ((uintptr_t) o + output_increment);
  } while (--output_width != 0);
}

void xnn_f16_ibilinear_ukernel__fma3_c8(
    size_t output_pixels,
    size_t channels,
    const void** restrict input,
    size_t input_offset,
    const void* restrict weights,
    void* restrict output,
    size_t output_increment) XNN_OOB_READS
{
  assert(output_pixels != 0);
  assert(channels != 0);
  assert(channels % sizeof(uint16_t) == 0);

  uint16_t* o = (uint16_t*) output;
  do {
    const uint16_t* i0 = (const uint16_t*) ((uintptr_t) input[0] + input_offset);
    const uint16_t* i1 = (const uint16_t*) ((uintptr_t) input[1] + input_offset);
    const uint16_t* i2 = (const uint16_t*) ((uintptr_t) input[2] + input_offset);
    const uint16_t* i3 = (const uint16_t*) ((uintptr_t) input[3] + input_offset);
    input += 4;

    const __m256 valphahv = _mm256_cvtph_ps(_mm_castps_si128(_mm_broadcast_ss(weights)));
    const __m256 valphah = _mm256_permute_ps(valphahv, _MM_SHUFFLE(2, 0, 2, 0));
    const __m256 valphav = _mm256_permute_ps(valphahv, _MM_SHUFFLE(3, 1, 3, 1));
    weights = (const uint16_t*) weights + 2;

    size_t c = channels;
    for (; c >= 8 * sizeof(uint16_t); c -= 8 * sizeof(uint16_t)) {
      const __m256 vtl = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i0));
      i0 += 8;
      const __m256 vtr = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i1));
      i1 += 8;
      const __m256 vbl = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i2));
      i2 += 8;
      const __m256 vbr = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i3));
      i3 += 8;

      const __m256 vtd = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_sub_ps(vtr, vtl), _MM_FROUND_TO_NEAREST_INT));
      const __m256 vbd = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_sub_ps(vbr, vbl), _MM_FROUND_TO_NEAREST_INT));

      const __m256 vt = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vtd, valphah, vtl), _MM_FROUND_TO_NEAREST_INT));
      const __m256 vb = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vbd, valphah, vbl), _MM_FROUND_TO_NEAREST_INT));

      const __m256 vd = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_sub_ps(vb, vt), _MM_FROUND_TO_NEAREST_INT));

      const __m128i vo = _mm256_cvtps_ph(_mm256_fmadd_ps(vd, valphav, vt), _MM_FROUND_TO_NEAREST_INT);

      _mm_storeu_si128((__m128i*) o, vo);
      o += 8;
    }
    if XNN_UNLIKELY(c != 0) {
      const __m256 vtl = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i0));
      i0 += 8;
      const __m256 vtr = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i1));
      i1 += 8;
      const __m256 vbl = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i2));
      i2 += 8;
      const __m256 vbr = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i3));
      i3 += 8;

      const __m256 vtd = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_sub_ps(vtr, vtl), _MM_FROUND_TO_NEAREST_INT));
      const __m256 vbd = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_sub_ps(vbr, vbl), _MM_FROUND_TO_NEAREST_INT));

      const __m256 vt = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vtd, valphah, vtl), _MM_FROUND_TO_NEAREST_INT));
      const __m256 vb = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vbd, valphah, vbl), _MM_FROUND_TO_NEAREST_INT));

      const __m256 vd = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_sub_ps(vb, vt), _MM_FROUND_TO_NEAREST_INT));

      __m128i vo = _mm256_cvtps_ph(_mm256_fmadd_ps(vd, valphav, vt), _MM_FROUND_TO_NEAREST_INT);
      if (c & (4 * sizeof(uint16_t))) {
        _mm_storel_epi64((__m128i*) o, vo);
        vo = _mm_unpackhi_epi64(vo, vo);
        o += 4;
      }
      if (c & (2 * sizeof(uint16_t))) {
        _mm_storeu_si32(o, vo);
        vo = _mm_srli_epi64(vo, 32);
        o += 2;
      }
      if (c & (1 * sizeof(uint16_t))) {
        *o = (uint16_t) _mm_extract_epi16(vo, 0);
        o += 1;
      }
    }

    o = (uint16_t*) ((uintptr_t) o + output_increment);
  } while (--output_pixels != 0);
}

void xnn_f16_vmulcaddc_minmax_ukernel_c8__fma3_2x(
    size_t rows,
    size_t channels,
    const void* restrict input,
    size_t input_stride,
    const void* restrict weights,
    void* restrict output,
    size_t output_stride,
    const union xnn_f16_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(rows != 0);
  assert(channels != 0);
  assert(channels % sizeof(uint16_t) == 0);

  const uint16_t* i0 = (const uint16_t*) input;
  uint16_t* o0 = (uint16_t*) output;
  const uint16_t* i1 = (const uint16_t*) ((uintptr_t) i0 + input_stride);
  uint16_t* o1 = (uint16_t*) ((uintptr_t) o0 + output_stride);

  const size_t input_increment = input_stride * 2 - channels;
  const size_t output_increment = output_stride * 2 - channels;

  const __m256 vmin = _mm256_set1_ps(params->avx.min);
  const __m256 vmax = _mm256_set1_ps(params->avx.max);
  XNN_FORCE_REALIZATION(vmin);
  XNN_FORCE_REALIZATION(vmax);
  do {
    if XNN_UNPREDICTABLE(rows < 2) {
      i1 = i0;
      o1 = o0;
    }

    const uint16_t* w = (const uint16_t*) weights;
    size_t c = channels;
    for (; c >= 8 * sizeof(uint16_t); c -= 8 * sizeof(uint16_t)) {
      const __m256 vscale = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) w));

      __m256 vacc0 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i0));
      i0 += 8;
      __m256 vacc1 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i1));
      i1 += 8;

      const __m256 vbias = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (w + 8)));
      w += 16;

      vacc0 = _mm256_fmadd_ps(vacc0, vscale, vbias);
      vacc1 = _mm256_fmadd_ps(vacc1, vscale, vbias);

      vacc0 = _mm256_max_ps(vacc0, vmin);
      vacc1 = _mm256_max_ps(vacc1, vmin);

      vacc0 = _mm256_min_ps(vacc0, vmax);
      vacc1 = _mm256_min_ps(vacc1, vmax);

      _mm_storeu_si128((__m128i*) o0, _mm256_cvtps_ph(vacc0, _MM_FROUND_TO_NEAREST_INT));
      o0 += 8;
      _mm_storeu_si128((__m128i*) o1, _mm256_cvtps_ph(vacc1, _MM_FROUND_TO_NEAREST_INT));
      o1 += 8;
    }
    if XNN_UNLIKELY(c != 0) {
      const __m256 vscale = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) w));

      __m256 vacc0 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i0));
      i0 = (const uint16_t*) ((uintptr_t) i0 + c);
      __m256 vacc1 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i1));
      i1 = (const uint16_t*) ((uintptr_t) i1 + c);

      const __m256 vbias = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (w + 8)));

      vacc0 = _mm256_fmadd_ps(vacc0, vscale, vbias);
      vacc1 = _mm256_fmadd_ps(vacc1, vscale, vbias);

      vacc0 = _mm256_max_ps(vacc0, vmin);
      vacc1 = _mm256_max_ps(vacc1, vmin);

      vacc0 = _mm256_min_ps(vacc0, vmax);
      vacc1 = _mm256_min_ps(vacc1, vmax);

      __m128i vh0 = _mm256_cvtps_ph(vacc0, _MM_FROUND_TO_NEAREST_INT);
      __m128i vh1 = _mm256_cvtps_ph(vacc1, _MM_FROUND_TO_NEAREST_INT);

      if (c & (4 * sizeof(uint16_t))) {
        _mm_storel_epi64((__m128i*) o0, vh0);
        _mm_storel_epi64((__m128i*) o1, vh1);

        vh0 = _mm_unpackhi_epi64(vh0, vh0);
        vh1 = _mm_unpackhi_epi64(vh1, vh1);

        o0 += 4;
        o1 += 4;
      }
      if (c & (2 * sizeof(uint16_t))) {
        _mm_storeu_si32(o0, vh0);
        _mm_storeu_si32(o1, vh1);

        vh0 = _mm_srli_epi64(vh0, 32);
        vh1 = _mm_srli_epi64(vh1, 32);

        o0 += 2;
        o1 += 2;
      }
      if (c & (1 * sizeof(uint16_t))) {
        *o0 = (uint16_t) _mm_extract_epi16(vh0, 0);
        *o1 = (uint16_t) _mm_extract_epi16(vh1, 0);

        o0 += 1;
        o1 += 1;
      }
    }
    i0 = (const uint16_t*) ((uintptr_t) i0 + input_increment);
    o0 = (uint16_t*) ((uintptr_t) o0 + output_increment);
    i1 = (const uint16_t*) ((uintptr_t) i1 + input_increment);
    o1 = (uint16_t*) ((uintptr_t) o1 + output_increment);
    rows = doz(rows, 2);
  } while (rows != 0);
}

void xnn_f16_vtanh_ukernel__fma3_polynomial_p19h9t2_u32(
    size_t batch,
    const void* input,
    void* output,
    const union xnn_f16_tanh_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(uint16_t) == 0);
  assert(input != NULL);
  assert(output != NULL);


  const __m256 vneg_sat_cutoff = _mm256_load_ps(params->avx_polynomial_p19h9t2.neg_sat_cutoff);
  const __m256 vpos_sat_cutoff = _mm256_load_ps(params->avx_polynomial_p19h9t2.pos_sat_cutoff);
  const __m256 vc19 = _mm256_load_ps(params->avx_polynomial_p19h9t2.c19);
  const __m256 vc17 = _mm256_load_ps(params->avx_polynomial_p19h9t2.c17);
  const __m256 vc15 = _mm256_load_ps(params->avx_polynomial_p19h9t2.c15);
  const __m256 vc13 = _mm256_load_ps(params->avx_polynomial_p19h9t2.c13);
  const __m256 vc11 = _mm256_load_ps(params->avx_polynomial_p19h9t2.c11);
  const __m256 vc9 = _mm256_load_ps(params->avx_polynomial_p19h9t2.c9);
  const __m256 vc7 = _mm256_load_ps(params->avx_polynomial_p19h9t2.c7);
  const __m256 vc5 = _mm256_load_ps(params->avx_polynomial_p19h9t2.c5);
  const __m256 vc3 = _mm256_load_ps(params->avx_polynomial_p19h9t2.c3);

  const uint16_t* i = (const uint16_t*) input;
  uint16_t* o = (uint16_t*) output;
  for (; batch >= 32 * sizeof(uint16_t); batch -= 32 * sizeof(uint16_t)) {
    __m256 vx0 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i));
    __m256 vx1 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i + 8)));
    __m256 vx2 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i + 16)));
    __m256 vx3 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i + 24)));
    i += 32;

    vx0 = _mm256_max_ps(vneg_sat_cutoff, vx0);
    vx1 = _mm256_max_ps(vneg_sat_cutoff, vx1);
    vx2 = _mm256_max_ps(vneg_sat_cutoff, vx2);
    vx3 = _mm256_max_ps(vneg_sat_cutoff, vx3);
    vx0 = _mm256_min_ps(vpos_sat_cutoff, vx0);
    vx1 = _mm256_min_ps(vpos_sat_cutoff, vx1);
    vx2 = _mm256_min_ps(vpos_sat_cutoff, vx2);
    vx3 = _mm256_min_ps(vpos_sat_cutoff, vx3);

    const __m256 vt0 = _mm256_mul_ps(vx0, vx0);
    const __m256 vt1 = _mm256_mul_ps(vx1, vx1);
    const __m256 vt2 = _mm256_mul_ps(vx2, vx2);
    const __m256 vt3 = _mm256_mul_ps(vx3, vx3);

    __m256 vp0 = vc19;
    __m256 vp1 = vc19;
    __m256 vp2 = vc19;
    __m256 vp3 = vc19;
    vp0 = _mm256_fmadd_ps(vp0, vt0, vc17);
    vp1 = _mm256_fmadd_ps(vp1, vt1, vc17);
    vp2 = _mm256_fmadd_ps(vp2, vt2, vc17);
    vp3 = _mm256_fmadd_ps(vp3, vt3, vc17);
    vp0 = _mm256_fmadd_ps(vp0, vt0, vc15);
    vp1 = _mm256_fmadd_ps(vp1, vt1, vc15);
    vp2 = _mm256_fmadd_ps(vp2, vt2, vc15);
    vp3 = _mm256_fmadd_ps(vp3, vt3, vc15);
    vp0 = _mm256_fmadd_ps(vp0, vt0, vc13);
    vp1 = _mm256_fmadd_ps(vp1, vt1, vc13);
    vp2 = _mm256_fmadd_ps(vp2, vt2, vc13);
    vp3 = _mm256_fmadd_ps(vp3, vt3, vc13);
    vp0 = _mm256_fmadd_ps(vp0, vt0, vc11);
    vp1 = _mm256_fmadd_ps(vp1, vt1, vc11);
    vp2 = _mm256_fmadd_ps(vp2, vt2, vc11);
    vp3 = _mm256_fmadd_ps(vp3, vt3, vc11);
    vp0 = _mm256_fmadd_ps(vp0, vt0, vc9);
    vp1 = _mm256_fmadd_ps(vp1, vt1, vc9);
    vp2 = _mm256_fmadd_ps(vp2, vt2, vc9);
    vp3 = _mm256_fmadd_ps(vp3, vt3, vc9);
    vp0 = _mm256_fmadd_ps(vp0, vt0, vc7);
    vp1 = _mm256_fmadd_ps(vp1, vt1, vc7);
    vp2 = _mm256_fmadd_ps(vp2, vt2, vc7);
    vp3 = _mm256_fmadd_ps(vp3, vt3, vc7);
    vp0 = _mm256_fmadd_ps(vp0, vt0, vc5);
    vp1 = _mm256_fmadd_ps(vp1, vt1, vc5);
    vp2 = _mm256_fmadd_ps(vp2, vt2, vc5);
    vp3 = _mm256_fmadd_ps(vp3, vt3, vc5);
    vp0 = _mm256_fmadd_ps(vp0, vt0, vc3);
    vp1 = _mm256_fmadd_ps(vp1, vt1, vc3);
    vp2 = _mm256_fmadd_ps(vp2, vt2, vc3);
    vp3 = _mm256_fmadd_ps(vp3, vt3, vc3);

    const __m256 vxt0 = _mm256_mul_ps(vx0, vt0);
    const __m256 vxt1 = _mm256_mul_ps(vx1, vt1);
    const __m256 vxt2 = _mm256_mul_ps(vx2, vt2);
    const __m256 vxt3 = _mm256_mul_ps(vx3, vt3);
    const __m256 vy0 = _mm256_fmadd_ps(vp0, vxt0, vx0);
    const __m256 vy1 = _mm256_fmadd_ps(vp1, vxt1, vx1);
    const __m256 vy2 = _mm256_fmadd_ps(vp2, vxt2, vx2);
    const __m256 vy3 = _mm256_fmadd_ps(vp3, vxt3, vx3);

    _mm_storeu_si128((__m128i*) o, _mm256_cvtps_ph(vy0, _MM_FROUND_TO_NEAREST_INT));
    _mm_storeu_si128((__m128i*) (o + 8), _mm256_cvtps_ph(vy1, _MM_FROUND_TO_NEAREST_INT));
    _mm_storeu_si128((__m128i*) (o + 16), _mm256_cvtps_ph(vy2, _MM_FROUND_TO_NEAREST_INT));
    _mm_storeu_si128((__m128i*) (o + 24), _mm256_cvtps_ph(vy3, _MM_FROUND_TO_NEAREST_INT));
    o += 32;
  }
  for (; batch >= 8 * sizeof(uint16_t); batch -= 8 * sizeof(uint16_t)) {
    __m256 vx = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i));
    i += 8;

    vx = _mm256_max_ps(vneg_sat_cutoff, vx);
    vx = _mm256_min_ps(vpos_sat_cutoff, vx);

    const __m256 vt = _mm256_mul_ps(vx, vx);

    __m256 vp = vc19;
    vp = _mm256_fmadd_ps(vp, vt, vc17);
    vp = _mm256_fmadd_ps(vp, vt, vc15);
    vp = _mm256_fmadd_ps(vp, vt, vc13);
    vp = _mm256_fmadd_ps(vp, vt, vc11);
    vp = _mm256_fmadd_ps(vp, vt, vc9);
    vp = _mm256_fmadd_ps(vp, vt, vc7);
    vp = _mm256_fmadd_ps(vp, vt, vc5);
    vp = _mm256_fmadd_ps(vp, vt, vc3);

    const __m256 vxt = _mm256_mul_ps(vx, vt);
    const __m256 vy = _mm256_fmadd_ps(vp, vxt, vx);

    _mm_storeu_si128((__m128i*) o, _mm256_cvtps_ph(vy, _MM_FROUND_TO_NEAREST_INT));
    o += 8;
  }
  if (batch != 0) {
    __m256 vx = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i));

    vx = _mm256_max_ps(vneg_sat_cutoff, vx);
    vx = _mm256_min_ps(vpos_sat_cutoff, vx);

    const __m256 vt = _mm256_mul_ps(vx, vx);

    __m256 vp = vc19;
    vp = _mm256_fmadd_ps(vp, vt, vc17);
    vp = _mm256_fmadd_ps(vp, vt, vc15);
    vp = _mm256_fmadd_ps(vp, vt, vc13);
    vp = _mm256_fmadd_ps(vp, vt, vc11);
    vp = _mm256_fmadd_ps(vp, vt, vc9);
    vp = _mm256_fmadd_ps(vp, vt, vc7);
    vp = _mm256_fmadd_ps(vp, vt, vc5);
    vp = _mm256_fmadd_ps(vp, vt, vc3);

    const __m256 vxt = _mm256_mul_ps(vx, vt);
    const __m256 vy = _mm256_fmadd_ps(vp, vxt, vx);

    __m128i vh = _mm256_cvtps_ph(vy, _MM_FROUND_TO_NEAREST_INT);

    if (batch & (4 * sizeof(uint16_t))) {
      _mm_storel_epi64((__m128i*) o, vh);
      vh = _mm_unpackhi_epi64(vh, vh);
      o += 4;
    }
    if (batch & (2 * sizeof(uint16_t))) {
      _mm_storeu_si32(o, vh);
      vh = _mm_srli_epi64(vh, 32);
      o += 2;
    }
    if (batch & (1 * sizeof(uint16_t))) {
      *o = (uint16_t) _mm_extract_epi16(vh, 0);
    }
  }
}

void xnn_f32_dwconv_minmax_ukernel_25p8c__fma3(
    size_t channels,
    size_t output_width,
    const float** input,
    const float* weights,
    float* output,
    intptr_t input_stride,
    size_t output_increment,
    size_t input_offset,
    const float* zero,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(channels != 0);
  assert(output_width != 0);

  static const int32_t mask_table[14] = {-1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0};

  const __m256 vmin = _mm256_set1_ps(params->avx.min);
  const __m256 vmax = _mm256_set1_ps(params->avx.max);
  XNN_FORCE_REALIZATION(vmin);
  XNN_FORCE_REALIZATION(vmax);
  do {
    const float* i0 = input[0];
    assert(i0 != NULL);
    if XNN_UNPREDICTABLE(i0 != zero) {
      i0 = (const float*) ((uintptr_t) i0 + input_offset);
    }
    const float* i1 = input[1];
    assert(i1 != NULL);
    if XNN_UNPREDICTABLE(i1 != zero) {
      i1 = (const float*) ((uintptr_t) i1 + input_offset);
    }
    const float* i2 = input[2];
    assert(i2 != NULL);
    if XNN_UNPREDICTABLE(i2 != zero) {
      i2 = (const float*) ((uintptr_t) i2 + input_offset);
    }
    const float* i3 = input[3];
    assert(i3 != NULL);
    if XNN_UNPREDICTABLE(i3 != zero) {
      i3 = (const float*) ((uintptr_t) i3 + input_offset);
    }
    const float* i4 = input[4];
    assert(i4 != NULL);
    if XNN_UNPREDICTABLE(i4 != zero) {
      i4 = (const float*) ((uintptr_t) i4 + input_offset);
    }
    const float* i5 = input[5];
    assert(i5 != NULL);
    if XNN_UNPREDICTABLE(i5 != zero) {
      i5 = (const float*) ((uintptr_t) i5 + input_offset);
    }
    const float* i6 = input[6];
    assert(i6 != NULL);
    if XNN_UNPREDICTABLE(i6 != zero) {
      i6 = (const float*) ((uintptr_t) i6 + input_offset);
    }
    const float* i7 = input[7];
    assert(i7 != NULL);
    if XNN_UNPREDICTABLE(i7 != zero) {
      i7 = (const float*) ((uintptr_t) i7 + input_offset);
    }
    const float* i8 = input[8];
    assert(i8 != NULL);
    if XNN_UNPREDICTABLE(i8 != zero) {
      i8 = (const float*) ((uintptr_t) i8 + input_offset);
    }
    const float* i9 = input[9];
    assert(i9 != NULL);
    if XNN_UNPREDICTABLE(i9 != zero) {
      i9 = (const float*) ((uintptr_t) i9 + input_offset);
    }
    const float* i10 = input[10];
    assert(i10 != NULL);
    if XNN_UNPREDICTABLE(i10 != zero) {
      i10 = (const float*) ((uintptr_t) i10 + input_offset);
    }
    const float* i11 = input[11];
    assert(i11 != NULL);
    if XNN_UNPREDICTABLE(i11 != zero) {
      i11 = (const float*) ((uintptr_t) i11 + input_offset);
    }
    const float* i12 = input[12];
    assert(i12 != NULL);
    if XNN_UNPREDICTABLE(i12 != zero) {
      i12 = (const float*) ((uintptr_t) i12 + input_offset);
    }
    const float* i13 = input[13];
    assert(i13 != NULL);
    if XNN_UNPREDICTABLE(i13 != zero) {
      i13 = (const float*) ((uintptr_t) i13 + input_offset);
    }
    const float* i14 = input[14];
    assert(i14 != NULL);
    if XNN_UNPREDICTABLE(i14 != zero) {
      i14 = (const float*) ((uintptr_t) i14 + input_offset);
    }
    const float* i15 = input[15];
    assert(i15 != NULL);
    if XNN_UNPREDICTABLE(i15 != zero) {
      i15 = (const float*) ((uintptr_t) i15 + input_offset);
    }
    const float* i16 = input[16];
    assert(i16 != NULL);
    if XNN_UNPREDICTABLE(i16 != zero) {
      i16 = (const float*) ((uintptr_t) i16 + input_offset);
    }
    const float* i17 = input[17];
    assert(i17 != NULL);
    if XNN_UNPREDICTABLE(i17 != zero) {
      i17 = (const float*) ((uintptr_t) i17 + input_offset);
    }
    const float* i18 = input[18];
    assert(i18 != NULL);
    if XNN_UNPREDICTABLE(i18 != zero) {
      i18 = (const float*) ((uintptr_t) i18 + input_offset);
    }
    const float* i19 = input[19];
    assert(i19 != NULL);
    if XNN_UNPREDICTABLE(i19 != zero) {
      i19 = (const float*) ((uintptr_t) i19 + input_offset);
    }
    const float* i20 = input[20];
    assert(i20 != NULL);
    if XNN_UNPREDICTABLE(i20 != zero) {
      i20 = (const float*) ((uintptr_t) i20 + input_offset);
    }
    const float* i21 = input[21];
    assert(i21 != NULL);
    if XNN_UNPREDICTABLE(i21 != zero) {
      i21 = (const float*) ((uintptr_t) i21 + input_offset);
    }
    const float* i22 = input[22];
    assert(i22 != NULL);
    if XNN_UNPREDICTABLE(i22 != zero) {
      i22 = (const float*) ((uintptr_t) i22 + input_offset);
    }
    const float* i23 = input[23];
    assert(i23 != NULL);
    if XNN_UNPREDICTABLE(i23 != zero) {
      i23 = (const float*) ((uintptr_t) i23 + input_offset);
    }
    const float* i24 = input[24];
    assert(i24 != NULL);
    if XNN_UNPREDICTABLE(i24 != zero) {
      i24 = (const float*) ((uintptr_t) i24 + input_offset);
    }
    input = (const float**) ((uintptr_t) input + input_stride);

    size_t c = channels;
    const float* w = weights;
    for (; c >= 8; c -= 8) {
      __m256 vacc01234567p0 = _mm256_load_ps(w);


      const __m256 vi0x01234567 = _mm256_loadu_ps(i0);
      i0 += 8;

      const __m256 vk0x01234567 = _mm256_load_ps(w + 8);
      vacc01234567p0 = _mm256_fmadd_ps(vi0x01234567, vk0x01234567, vacc01234567p0);

      const __m256 vi1x01234567 = _mm256_loadu_ps(i1);
      i1 += 8;

      const __m256 vk1x01234567 = _mm256_load_ps(w + 16);
      vacc01234567p0 = _mm256_fmadd_ps(vi1x01234567, vk1x01234567, vacc01234567p0);

      const __m256 vi2x01234567 = _mm256_loadu_ps(i2);
      i2 += 8;

      const __m256 vk2x01234567 = _mm256_load_ps(w + 24);
      vacc01234567p0 = _mm256_fmadd_ps(vi2x01234567, vk2x01234567, vacc01234567p0);

      const __m256 vi3x01234567 = _mm256_loadu_ps(i3);
      i3 += 8;

      const __m256 vk3x01234567 = _mm256_load_ps(w + 32);
      vacc01234567p0 = _mm256_fmadd_ps(vi3x01234567, vk3x01234567, vacc01234567p0);

      const __m256 vi4x01234567 = _mm256_loadu_ps(i4);
      i4 += 8;

      const __m256 vk4x01234567 = _mm256_load_ps(w + 40);
      vacc01234567p0 = _mm256_fmadd_ps(vi4x01234567, vk4x01234567, vacc01234567p0);

      const __m256 vi5x01234567 = _mm256_loadu_ps(i5);
      i5 += 8;

      const __m256 vk5x01234567 = _mm256_load_ps(w + 48);
      vacc01234567p0 = _mm256_fmadd_ps(vi5x01234567, vk5x01234567, vacc01234567p0);

      const __m256 vi6x01234567 = _mm256_loadu_ps(i6);
      i6 += 8;

      const __m256 vk6x01234567 = _mm256_load_ps(w + 56);
      vacc01234567p0 = _mm256_fmadd_ps(vi6x01234567, vk6x01234567, vacc01234567p0);

      const __m256 vi7x01234567 = _mm256_loadu_ps(i7);
      i7 += 8;

      const __m256 vk7x01234567 = _mm256_load_ps(w + 64);
      vacc01234567p0 = _mm256_fmadd_ps(vi7x01234567, vk7x01234567, vacc01234567p0);

      const __m256 vi8x01234567 = _mm256_loadu_ps(i8);
      i8 += 8;

      const __m256 vk8x01234567 = _mm256_load_ps(w + 72);
      vacc01234567p0 = _mm256_fmadd_ps(vi8x01234567, vk8x01234567, vacc01234567p0);

      const __m256 vi9x01234567 = _mm256_loadu_ps(i9);
      i9 += 8;

      const __m256 vk9x01234567 = _mm256_load_ps(w + 80);
      vacc01234567p0 = _mm256_fmadd_ps(vi9x01234567, vk9x01234567, vacc01234567p0);

      const __m256 vi10x01234567 = _mm256_loadu_ps(i10);
      i10 += 8;

      const __m256 vk10x01234567 = _mm256_load_ps(w + 88);
      vacc01234567p0 = _mm256_fmadd_ps(vi10x01234567, vk10x01234567, vacc01234567p0);

      const __m256 vi11x01234567 = _mm256_loadu_ps(i11);
      i11 += 8;

      const __m256 vk11x01234567 = _mm256_load_ps(w + 96);
      vacc01234567p0 = _mm256_fmadd_ps(vi11x01234567, vk11x01234567, vacc01234567p0);

      const __m256 vi12x01234567 = _mm256_loadu_ps(i12);
      i12 += 8;

      const __m256 vk12x01234567 = _mm256_load_ps(w + 104);
      vacc01234567p0 = _mm256_fmadd_ps(vi12x01234567, vk12x01234567, vacc01234567p0);

      const __m256 vi13x01234567 = _mm256_loadu_ps(i13);
      i13 += 8;

      const __m256 vk13x01234567 = _mm256_load_ps(w + 112);
      vacc01234567p0 = _mm256_fmadd_ps(vi13x01234567, vk13x01234567, vacc01234567p0);

      const __m256 vi14x01234567 = _mm256_loadu_ps(i14);
      i14 += 8;

      const __m256 vk14x01234567 = _mm256_load_ps(w + 120);
      vacc01234567p0 = _mm256_fmadd_ps(vi14x01234567, vk14x01234567, vacc01234567p0);

      const __m256 vi15x01234567 = _mm256_loadu_ps(i15);
      i15 += 8;

      const __m256 vk15x01234567 = _mm256_load_ps(w + 128);
      vacc01234567p0 = _mm256_fmadd_ps(vi15x01234567, vk15x01234567, vacc01234567p0);

      const __m256 vi16x01234567 = _mm256_loadu_ps(i16);
      i16 += 8;

      const __m256 vk16x01234567 = _mm256_load_ps(w + 136);
      vacc01234567p0 = _mm256_fmadd_ps(vi16x01234567, vk16x01234567, vacc01234567p0);

      const __m256 vi17x01234567 = _mm256_loadu_ps(i17);
      i17 += 8;

      const __m256 vk17x01234567 = _mm256_load_ps(w + 144);
      vacc01234567p0 = _mm256_fmadd_ps(vi17x01234567, vk17x01234567, vacc01234567p0);

      const __m256 vi18x01234567 = _mm256_loadu_ps(i18);
      i18 += 8;

      const __m256 vk18x01234567 = _mm256_load_ps(w + 152);
      vacc01234567p0 = _mm256_fmadd_ps(vi18x01234567, vk18x01234567, vacc01234567p0);

      const __m256 vi19x01234567 = _mm256_loadu_ps(i19);
      i19 += 8;

      const __m256 vk19x01234567 = _mm256_load_ps(w + 160);
      vacc01234567p0 = _mm256_fmadd_ps(vi19x01234567, vk19x01234567, vacc01234567p0);

      const __m256 vi20x01234567 = _mm256_loadu_ps(i20);
      i20 += 8;

      const __m256 vk20x01234567 = _mm256_load_ps(w + 168);
      vacc01234567p0 = _mm256_fmadd_ps(vi20x01234567, vk20x01234567, vacc01234567p0);

      const __m256 vi21x01234567 = _mm256_loadu_ps(i21);
      i21 += 8;

      const __m256 vk21x01234567 = _mm256_load_ps(w + 176);
      vacc01234567p0 = _mm256_fmadd_ps(vi21x01234567, vk21x01234567, vacc01234567p0);

      const __m256 vi22x01234567 = _mm256_loadu_ps(i22);
      i22 += 8;

      const __m256 vk22x01234567 = _mm256_load_ps(w + 184);
      vacc01234567p0 = _mm256_fmadd_ps(vi22x01234567, vk22x01234567, vacc01234567p0);

      const __m256 vi23x01234567 = _mm256_loadu_ps(i23);
      i23 += 8;

      const __m256 vk23x01234567 = _mm256_load_ps(w + 192);
      vacc01234567p0 = _mm256_fmadd_ps(vi23x01234567, vk23x01234567, vacc01234567p0);

      const __m256 vi24x01234567 = _mm256_loadu_ps(i24);
      i24 += 8;

      const __m256 vk24x01234567 = _mm256_load_ps(w + 200);
      vacc01234567p0 = _mm256_fmadd_ps(vi24x01234567, vk24x01234567, vacc01234567p0);

      w += 208;


      __m256 vacc01234567 = _mm256_max_ps(vmin, vacc01234567p0);
      vacc01234567 = _mm256_min_ps(vmax, vacc01234567);

      _mm256_storeu_ps(output, vacc01234567);
      output += 8;
    }
    if XNN_UNLIKELY(c != 0) {
      assert(c >= 1);
      assert(c <= 7);
      const __m256i vmask = _mm256_loadu_si256((const __m256i*) &mask_table[7 - c]);

      __m256 vacc01234567p0 = _mm256_load_ps(w);

      const __m256 vi0x01234567 = _mm256_maskload_ps(i0, vmask);
      const __m256 vk0x01234567 = _mm256_load_ps(w + 8);
      vacc01234567p0 = _mm256_fmadd_ps(vi0x01234567, vk0x01234567, vacc01234567p0);

      const __m256 vi1x01234567 = _mm256_maskload_ps(i1, vmask);
      const __m256 vk1x01234567 = _mm256_load_ps(w + 16);
      vacc01234567p0 = _mm256_fmadd_ps(vi1x01234567, vk1x01234567, vacc01234567p0);

      const __m256 vi2x01234567 = _mm256_maskload_ps(i2, vmask);
      const __m256 vk2x01234567 = _mm256_load_ps(w + 24);
      vacc01234567p0 = _mm256_fmadd_ps(vi2x01234567, vk2x01234567, vacc01234567p0);

      const __m256 vi3x01234567 = _mm256_maskload_ps(i3, vmask);
      const __m256 vk3x01234567 = _mm256_load_ps(w + 32);
      vacc01234567p0 = _mm256_fmadd_ps(vi3x01234567, vk3x01234567, vacc01234567p0);

      const __m256 vi4x01234567 = _mm256_maskload_ps(i4, vmask);
      const __m256 vk4x01234567 = _mm256_load_ps(w + 40);
      vacc01234567p0 = _mm256_fmadd_ps(vi4x01234567, vk4x01234567, vacc01234567p0);

      const __m256 vi5x01234567 = _mm256_maskload_ps(i5, vmask);
      const __m256 vk5x01234567 = _mm256_load_ps(w + 48);
      vacc01234567p0 = _mm256_fmadd_ps(vi5x01234567, vk5x01234567, vacc01234567p0);

      const __m256 vi6x01234567 = _mm256_maskload_ps(i6, vmask);
      const __m256 vk6x01234567 = _mm256_load_ps(w + 56);
      vacc01234567p0 = _mm256_fmadd_ps(vi6x01234567, vk6x01234567, vacc01234567p0);

      const __m256 vi7x01234567 = _mm256_maskload_ps(i7, vmask);
      const __m256 vk7x01234567 = _mm256_load_ps(w + 64);
      vacc01234567p0 = _mm256_fmadd_ps(vi7x01234567, vk7x01234567, vacc01234567p0);

      const __m256 vi8x01234567 = _mm256_maskload_ps(i8, vmask);
      const __m256 vk8x01234567 = _mm256_load_ps(w + 72);
      vacc01234567p0 = _mm256_fmadd_ps(vi8x01234567, vk8x01234567, vacc01234567p0);

      const __m256 vi9x01234567 = _mm256_maskload_ps(i9, vmask);
      const __m256 vk9x01234567 = _mm256_load_ps(w + 80);
      vacc01234567p0 = _mm256_fmadd_ps(vi9x01234567, vk9x01234567, vacc01234567p0);

      const __m256 vi10x01234567 = _mm256_maskload_ps(i10, vmask);
      const __m256 vk10x01234567 = _mm256_load_ps(w + 88);
      vacc01234567p0 = _mm256_fmadd_ps(vi10x01234567, vk10x01234567, vacc01234567p0);

      const __m256 vi11x01234567 = _mm256_maskload_ps(i11, vmask);
      const __m256 vk11x01234567 = _mm256_load_ps(w + 96);
      vacc01234567p0 = _mm256_fmadd_ps(vi11x01234567, vk11x01234567, vacc01234567p0);

      const __m256 vi12x01234567 = _mm256_maskload_ps(i12, vmask);
      const __m256 vk12x01234567 = _mm256_load_ps(w + 104);
      vacc01234567p0 = _mm256_fmadd_ps(vi12x01234567, vk12x01234567, vacc01234567p0);

      const __m256 vi13x01234567 = _mm256_maskload_ps(i13, vmask);
      const __m256 vk13x01234567 = _mm256_load_ps(w + 112);
      vacc01234567p0 = _mm256_fmadd_ps(vi13x01234567, vk13x01234567, vacc01234567p0);

      const __m256 vi14x01234567 = _mm256_maskload_ps(i14, vmask);
      const __m256 vk14x01234567 = _mm256_load_ps(w + 120);
      vacc01234567p0 = _mm256_fmadd_ps(vi14x01234567, vk14x01234567, vacc01234567p0);

      const __m256 vi15x01234567 = _mm256_maskload_ps(i15, vmask);
      const __m256 vk15x01234567 = _mm256_load_ps(w + 128);
      vacc01234567p0 = _mm256_fmadd_ps(vi15x01234567, vk15x01234567, vacc01234567p0);

      const __m256 vi16x01234567 = _mm256_maskload_ps(i16, vmask);
      const __m256 vk16x01234567 = _mm256_load_ps(w + 136);
      vacc01234567p0 = _mm256_fmadd_ps(vi16x01234567, vk16x01234567, vacc01234567p0);

      const __m256 vi17x01234567 = _mm256_maskload_ps(i17, vmask);
      const __m256 vk17x01234567 = _mm256_load_ps(w + 144);
      vacc01234567p0 = _mm256_fmadd_ps(vi17x01234567, vk17x01234567, vacc01234567p0);

      const __m256 vi18x01234567 = _mm256_maskload_ps(i18, vmask);
      const __m256 vk18x01234567 = _mm256_load_ps(w + 152);
      vacc01234567p0 = _mm256_fmadd_ps(vi18x01234567, vk18x01234567, vacc01234567p0);

      const __m256 vi19x01234567 = _mm256_maskload_ps(i19, vmask);
      const __m256 vk19x01234567 = _mm256_load_ps(w + 160);
      vacc01234567p0 = _mm256_fmadd_ps(vi19x01234567, vk19x01234567, vacc01234567p0);

      const __m256 vi20x01234567 = _mm256_maskload_ps(i20, vmask);
      const __m256 vk20x01234567 = _mm256_load_ps(w + 168);
      vacc01234567p0 = _mm256_fmadd_ps(vi20x01234567, vk20x01234567, vacc01234567p0);

      const __m256 vi21x01234567 = _mm256_maskload_ps(i21, vmask);
      const __m256 vk21x01234567 = _mm256_load_ps(w + 176);
      vacc01234567p0 = _mm256_fmadd_ps(vi21x01234567, vk21x01234567, vacc01234567p0);

      const __m256 vi22x01234567 = _mm256_maskload_ps(i22, vmask);
      const __m256 vk22x01234567 = _mm256_load_ps(w + 184);
      vacc01234567p0 = _mm256_fmadd_ps(vi22x01234567, vk22x01234567, vacc01234567p0);

      const __m256 vi23x01234567 = _mm256_maskload_ps(i23, vmask);
      const __m256 vk23x01234567 = _mm256_load_ps(w + 192);
      vacc01234567p0 = _mm256_fmadd_ps(vi23x01234567, vk23x01234567, vacc01234567p0);

      const __m256 vi24x01234567 = _mm256_maskload_ps(i24, vmask);
      const __m256 vk24x01234567 = _mm256_load_ps(w + 200);
      vacc01234567p0 = _mm256_fmadd_ps(vi24x01234567, vk24x01234567, vacc01234567p0);


      __m256 vacc01234567 = _mm256_max_ps(vmin, vacc01234567p0);
      vacc01234567 = _mm256_min_ps(vmax, vacc01234567);

      __m128 vacc0123 = _mm256_castps256_ps128(vacc01234567);
      if (c & 4) {
        _mm_storeu_ps(output, vacc0123);
        vacc0123 = _mm256_extractf128_ps(vacc01234567, 1);
        output += 4;
      }
      if (c & 2) {
        _mm_storel_pi((__m64*) output, vacc0123);
        vacc0123 = _mm_movehl_ps(vacc0123, vacc0123);
        output += 2;
      }
      if (c & 1) {
        _mm_store_ss(output, vacc0123);
        output += 1;
      }
    }

    output = (float*) ((uintptr_t) output + output_increment);
  } while (--output_width != 0);
}

void xnn_f32_dwconv_minmax_ukernel_3p16c__fma3(
    size_t channels,
    size_t output_width,
    const float** input,
    const float* weights,
    float* output,
    intptr_t input_stride,
    size_t output_increment,
    size_t input_offset,
    const float* zero,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(channels != 0);
  assert(output_width != 0);

  static const int32_t mask_table[14] = {-1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0};

  const __m256 vmin = _mm256_set1_ps(params->avx.min);
  const __m256 vmax = _mm256_set1_ps(params->avx.max);
  XNN_FORCE_REALIZATION(vmin);
  XNN_FORCE_REALIZATION(vmax);
  do {
    const float* i0 = input[0];
    assert(i0 != NULL);
    if XNN_UNPREDICTABLE(i0 != zero) {
      i0 = (const float*) ((uintptr_t) i0 + input_offset);
    }
    const float* i1 = input[1];
    assert(i1 != NULL);
    if XNN_UNPREDICTABLE(i1 != zero) {
      i1 = (const float*) ((uintptr_t) i1 + input_offset);
    }
    const float* i2 = input[2];
    assert(i2 != NULL);
    if XNN_UNPREDICTABLE(i2 != zero) {
      i2 = (const float*) ((uintptr_t) i2 + input_offset);
    }
    input = (const float**) ((uintptr_t) input + input_stride);

    size_t c = channels;
    const float* w = weights;
    for (; c >= 16; c -= 16) {
      __m256 vacc01234567p0 = _mm256_load_ps(w);
      __m256 vacc89ABCDEFp0 = _mm256_load_ps(w + 8);


      const __m256 vi0x01234567 = _mm256_loadu_ps(i0);
      const __m256 vi0x89ABCDEF = _mm256_loadu_ps(i0 + 8);
      i0 += 16;

      const __m256 vk0x01234567 = _mm256_load_ps(w + 16);
      const __m256 vk0x89ABCDEF = _mm256_load_ps(w + 24);
      vacc01234567p0 = _mm256_fmadd_ps(vi0x01234567, vk0x01234567, vacc01234567p0);
      vacc89ABCDEFp0 = _mm256_fmadd_ps(vi0x89ABCDEF, vk0x89ABCDEF, vacc89ABCDEFp0);

      const __m256 vi1x01234567 = _mm256_loadu_ps(i1);
      const __m256 vi1x89ABCDEF = _mm256_loadu_ps(i1 + 8);
      i1 += 16;

      const __m256 vk1x01234567 = _mm256_load_ps(w + 32);
      const __m256 vk1x89ABCDEF = _mm256_load_ps(w + 40);
      vacc01234567p0 = _mm256_fmadd_ps(vi1x01234567, vk1x01234567, vacc01234567p0);
      vacc89ABCDEFp0 = _mm256_fmadd_ps(vi1x89ABCDEF, vk1x89ABCDEF, vacc89ABCDEFp0);

      const __m256 vi2x01234567 = _mm256_loadu_ps(i2);
      const __m256 vi2x89ABCDEF = _mm256_loadu_ps(i2 + 8);
      i2 += 16;

      const __m256 vk2x01234567 = _mm256_load_ps(w + 48);
      const __m256 vk2x89ABCDEF = _mm256_load_ps(w + 56);
      vacc01234567p0 = _mm256_fmadd_ps(vi2x01234567, vk2x01234567, vacc01234567p0);
      vacc89ABCDEFp0 = _mm256_fmadd_ps(vi2x89ABCDEF, vk2x89ABCDEF, vacc89ABCDEFp0);

      w += 64;


      __m256 vacc01234567 = _mm256_max_ps(vmin, vacc01234567p0);
      __m256 vacc89ABCDEF = _mm256_max_ps(vmin, vacc89ABCDEFp0);
      vacc01234567 = _mm256_min_ps(vmax, vacc01234567);
      vacc89ABCDEF = _mm256_min_ps(vmax, vacc89ABCDEF);

      _mm256_storeu_ps(output, vacc01234567);
      _mm256_storeu_ps(output + 8, vacc89ABCDEF);
      output += 16;
    }
    for (; c >= 8; c -= 8) {
      __m256 vacc01234567p0 = _mm256_load_ps(w);

      const __m256 vi0x01234567 = _mm256_loadu_ps(i0);
      i0 += 8;

      const __m256 vk0x01234567 = _mm256_load_ps(w + 16);
      vacc01234567p0 = _mm256_fmadd_ps(vi0x01234567, vk0x01234567, vacc01234567p0);

      const __m256 vi1x01234567 = _mm256_loadu_ps(i1);
      i1 += 8;

      const __m256 vk1x01234567 = _mm256_load_ps(w + 32);
      vacc01234567p0 = _mm256_fmadd_ps(vi1x01234567, vk1x01234567, vacc01234567p0);

      const __m256 vi2x01234567 = _mm256_loadu_ps(i2);
      i2 += 8;

      const __m256 vk2x01234567 = _mm256_load_ps(w + 48);
      vacc01234567p0 = _mm256_fmadd_ps(vi2x01234567, vk2x01234567, vacc01234567p0);

      w += 8;


      __m256 vacc01234567 = _mm256_max_ps(vmin, vacc01234567p0);
      vacc01234567 = _mm256_min_ps(vmax, vacc01234567);

      _mm256_storeu_ps(output, vacc01234567);
      output += 8;
    }
    if XNN_UNLIKELY(c != 0) {
      assert(c >= 1);
      assert(c <= 7);
      const __m256i vmask = _mm256_loadu_si256((const __m256i*) &mask_table[7 - c]);

      __m256 vacc01234567p0 = _mm256_load_ps(w);

      const __m256 vi0x01234567 = _mm256_maskload_ps(i0, vmask);
      const __m256 vk0x01234567 = _mm256_load_ps(w + 16);
      vacc01234567p0 = _mm256_fmadd_ps(vi0x01234567, vk0x01234567, vacc01234567p0);

      const __m256 vi1x01234567 = _mm256_maskload_ps(i1, vmask);
      const __m256 vk1x01234567 = _mm256_load_ps(w + 32);
      vacc01234567p0 = _mm256_fmadd_ps(vi1x01234567, vk1x01234567, vacc01234567p0);

      const __m256 vi2x01234567 = _mm256_maskload_ps(i2, vmask);
      const __m256 vk2x01234567 = _mm256_load_ps(w + 48);
      vacc01234567p0 = _mm256_fmadd_ps(vi2x01234567, vk2x01234567, vacc01234567p0);


      __m256 vacc01234567 = _mm256_max_ps(vmin, vacc01234567p0);
      vacc01234567 = _mm256_min_ps(vmax, vacc01234567);

      __m128 vacc0123 = _mm256_castps256_ps128(vacc01234567);
      if (c & 4) {
        _mm_storeu_ps(output, vacc0123);
        vacc0123 = _mm256_extractf128_ps(vacc01234567, 1);
        output += 4;
      }
      if (c & 2) {
        _mm_storel_pi((__m64*) output, vacc0123);
        vacc0123 = _mm_movehl_ps(vacc0123, vacc0123);
        output += 2;
      }
      if (c & 1) {
        _mm_store_ss(output, vacc0123);
        output += 1;
      }
    }

    output = (float*) ((uintptr_t) output + output_increment);
  } while (--output_width != 0);
}

void xnn_f32_dwconv_minmax_ukernel_4p16c__fma3(
    size_t channels,
    size_t output_width,
    const float** input,
    const float* weights,
    float* output,
    intptr_t input_stride,
    size_t output_increment,
    size_t input_offset,
    const float* zero,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(channels != 0);
  assert(output_width != 0);

  static const int32_t mask_table[14] = {-1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0};

  const __m256 vmin = _mm256_set1_ps(params->avx.min);
  const __m256 vmax = _mm256_set1_ps(params->avx.max);
  XNN_FORCE_REALIZATION(vmin);
  XNN_FORCE_REALIZATION(vmax);
  do {
    const float* i0 = input[0];
    assert(i0 != NULL);
    if XNN_UNPREDICTABLE(i0 != zero) {
      i0 = (const float*) ((uintptr_t) i0 + input_offset);
    }
    const float* i1 = input[1];
    assert(i1 != NULL);
    if XNN_UNPREDICTABLE(i1 != zero) {
      i1 = (const float*) ((uintptr_t) i1 + input_offset);
    }
    const float* i2 = input[2];
    assert(i2 != NULL);
    if XNN_UNPREDICTABLE(i2 != zero) {
      i2 = (const float*) ((uintptr_t) i2 + input_offset);
    }
    const float* i3 = input[3];
    assert(i3 != NULL);
    if XNN_UNPREDICTABLE(i3 != zero) {
      i3 = (const float*) ((uintptr_t) i3 + input_offset);
    }
    input = (const float**) ((uintptr_t) input + input_stride);

    size_t c = channels;
    const float* w = weights;
    for (; c >= 16; c -= 16) {
      __m256 vacc01234567p0 = _mm256_load_ps(w);
      __m256 vacc89ABCDEFp0 = _mm256_load_ps(w + 8);


      const __m256 vi0x01234567 = _mm256_loadu_ps(i0);
      const __m256 vi0x89ABCDEF = _mm256_loadu_ps(i0 + 8);
      i0 += 16;

      const __m256 vk0x01234567 = _mm256_load_ps(w + 16);
      const __m256 vk0x89ABCDEF = _mm256_load_ps(w + 24);
      vacc01234567p0 = _mm256_fmadd_ps(vi0x01234567, vk0x01234567, vacc01234567p0);
      vacc89ABCDEFp0 = _mm256_fmadd_ps(vi0x89ABCDEF, vk0x89ABCDEF, vacc89ABCDEFp0);

      const __m256 vi1x01234567 = _mm256_loadu_ps(i1);
      const __m256 vi1x89ABCDEF = _mm256_loadu_ps(i1 + 8);
      i1 += 16;

      const __m256 vk1x01234567 = _mm256_load_ps(w + 32);
      const __m256 vk1x89ABCDEF = _mm256_load_ps(w + 40);
      vacc01234567p0 = _mm256_fmadd_ps(vi1x01234567, vk1x01234567, vacc01234567p0);
      vacc89ABCDEFp0 = _mm256_fmadd_ps(vi1x89ABCDEF, vk1x89ABCDEF, vacc89ABCDEFp0);

      const __m256 vi2x01234567 = _mm256_loadu_ps(i2);
      const __m256 vi2x89ABCDEF = _mm256_loadu_ps(i2 + 8);
      i2 += 16;

      const __m256 vk2x01234567 = _mm256_load_ps(w + 48);
      const __m256 vk2x89ABCDEF = _mm256_load_ps(w + 56);
      vacc01234567p0 = _mm256_fmadd_ps(vi2x01234567, vk2x01234567, vacc01234567p0);
      vacc89ABCDEFp0 = _mm256_fmadd_ps(vi2x89ABCDEF, vk2x89ABCDEF, vacc89ABCDEFp0);

      const __m256 vi3x01234567 = _mm256_loadu_ps(i3);
      const __m256 vi3x89ABCDEF = _mm256_loadu_ps(i3 + 8);
      i3 += 16;

      const __m256 vk3x01234567 = _mm256_load_ps(w + 64);
      const __m256 vk3x89ABCDEF = _mm256_load_ps(w + 72);
      vacc01234567p0 = _mm256_fmadd_ps(vi3x01234567, vk3x01234567, vacc01234567p0);
      vacc89ABCDEFp0 = _mm256_fmadd_ps(vi3x89ABCDEF, vk3x89ABCDEF, vacc89ABCDEFp0);

      w += 80;


      __m256 vacc01234567 = _mm256_max_ps(vmin, vacc01234567p0);
      __m256 vacc89ABCDEF = _mm256_max_ps(vmin, vacc89ABCDEFp0);
      vacc01234567 = _mm256_min_ps(vmax, vacc01234567);
      vacc89ABCDEF = _mm256_min_ps(vmax, vacc89ABCDEF);

      _mm256_storeu_ps(output, vacc01234567);
      _mm256_storeu_ps(output + 8, vacc89ABCDEF);
      output += 16;
    }
    for (; c >= 8; c -= 8) {
      __m256 vacc01234567p0 = _mm256_load_ps(w);

      const __m256 vi0x01234567 = _mm256_loadu_ps(i0);
      i0 += 8;

      const __m256 vk0x01234567 = _mm256_load_ps(w + 16);
      vacc01234567p0 = _mm256_fmadd_ps(vi0x01234567, vk0x01234567, vacc01234567p0);

      const __m256 vi1x01234567 = _mm256_loadu_ps(i1);
      i1 += 8;

      const __m256 vk1x01234567 = _mm256_load_ps(w + 32);
      vacc01234567p0 = _mm256_fmadd_ps(vi1x01234567, vk1x01234567, vacc01234567p0);

      const __m256 vi2x01234567 = _mm256_loadu_ps(i2);
      i2 += 8;

      const __m256 vk2x01234567 = _mm256_load_ps(w + 48);
      vacc01234567p0 = _mm256_fmadd_ps(vi2x01234567, vk2x01234567, vacc01234567p0);

      const __m256 vi3x01234567 = _mm256_loadu_ps(i3);
      i3 += 8;

      const __m256 vk3x01234567 = _mm256_load_ps(w + 64);
      vacc01234567p0 = _mm256_fmadd_ps(vi3x01234567, vk3x01234567, vacc01234567p0);

      w += 8;


      __m256 vacc01234567 = _mm256_max_ps(vmin, vacc01234567p0);
      vacc01234567 = _mm256_min_ps(vmax, vacc01234567);

      _mm256_storeu_ps(output, vacc01234567);
      output += 8;
    }
    if XNN_UNLIKELY(c != 0) {
      assert(c >= 1);
      assert(c <= 7);
      const __m256i vmask = _mm256_loadu_si256((const __m256i*) &mask_table[7 - c]);

      __m256 vacc01234567p0 = _mm256_load_ps(w);

      const __m256 vi0x01234567 = _mm256_maskload_ps(i0, vmask);
      const __m256 vk0x01234567 = _mm256_load_ps(w + 16);
      vacc01234567p0 = _mm256_fmadd_ps(vi0x01234567, vk0x01234567, vacc01234567p0);

      const __m256 vi1x01234567 = _mm256_maskload_ps(i1, vmask);
      const __m256 vk1x01234567 = _mm256_load_ps(w + 32);
      vacc01234567p0 = _mm256_fmadd_ps(vi1x01234567, vk1x01234567, vacc01234567p0);

      const __m256 vi2x01234567 = _mm256_maskload_ps(i2, vmask);
      const __m256 vk2x01234567 = _mm256_load_ps(w + 48);
      vacc01234567p0 = _mm256_fmadd_ps(vi2x01234567, vk2x01234567, vacc01234567p0);

      const __m256 vi3x01234567 = _mm256_maskload_ps(i3, vmask);
      const __m256 vk3x01234567 = _mm256_load_ps(w + 64);
      vacc01234567p0 = _mm256_fmadd_ps(vi3x01234567, vk3x01234567, vacc01234567p0);


      __m256 vacc01234567 = _mm256_max_ps(vmin, vacc01234567p0);
      vacc01234567 = _mm256_min_ps(vmax, vacc01234567);

      __m128 vacc0123 = _mm256_castps256_ps128(vacc01234567);
      if (c & 4) {
        _mm_storeu_ps(output, vacc0123);
        vacc0123 = _mm256_extractf128_ps(vacc01234567, 1);
        output += 4;
      }
      if (c & 2) {
        _mm_storel_pi((__m64*) output, vacc0123);
        vacc0123 = _mm_movehl_ps(vacc0123, vacc0123);
        output += 2;
      }
      if (c & 1) {
        _mm_store_ss(output, vacc0123);
        output += 1;
      }
    }

    output = (float*) ((uintptr_t) output + output_increment);
  } while (--output_width != 0);
}

void xnn_f32_dwconv_minmax_ukernel_5f5m5l8c8s4r__fma3(
    size_t channels,
    size_t output_width,
    const float** input,
    const float* weights,
    float* output,
    intptr_t input_stride,
    size_t output_increment,
    size_t input_offset,
    const float* zero,
    size_t kernel_size,
    float* buffer,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(channels != 0);
  assert(output_width != 0);
  assert(kernel_size > 5);

  static const int32_t mask_table[14] = {-1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0};

  const __m256 vmin = _mm256_set1_ps(params->avx.min);
  const __m256 vmax = _mm256_set1_ps(params->avx.max);
  XNN_FORCE_REALIZATION(vmin);
  XNN_FORCE_REALIZATION(vmax);
  do {
    const float* w = weights;

    // First pass to process 5 inputs.
    {
      float* b = buffer;
      const float* i0 = input[0];
      assert(i0 != NULL);
      if XNN_UNPREDICTABLE(i0 != zero) {
        i0 = (const float*) ((uintptr_t) i0 + input_offset);
      }
      const float* i1 = input[1];
      assert(i1 != NULL);
      if XNN_UNPREDICTABLE(i1 != zero) {
        i1 = (const float*) ((uintptr_t) i1 + input_offset);
      }
      const float* i2 = input[2];
      assert(i2 != NULL);
      if XNN_UNPREDICTABLE(i2 != zero) {
        i2 = (const float*) ((uintptr_t) i2 + input_offset);
      }
      const float* i3 = input[3];
      assert(i3 != NULL);
      if XNN_UNPREDICTABLE(i3 != zero) {
        i3 = (const float*) ((uintptr_t) i3 + input_offset);
      }
      const float* i4 = input[4];
      assert(i4 != NULL);
      if XNN_UNPREDICTABLE(i4 != zero) {
        i4 = (const float*) ((uintptr_t) i4 + input_offset);
      }
      input += 5;

      // Process c channels and write to buffer.
      size_t c = round_up_po2(channels, 4);

      for (; c >= 8; c -= 8) {
        __m256 vacc01234567p0 = _mm256_load_ps(w);


        const __m256 vi0x01234567 = _mm256_loadu_ps(i0);
        i0 += 8;

        const __m256 vk0x01234567 = _mm256_load_ps(w + 8);
        vacc01234567p0 = _mm256_fmadd_ps(vi0x01234567, vk0x01234567, vacc01234567p0);

        const __m256 vi1x01234567 = _mm256_loadu_ps(i1);
        i1 += 8;

        const __m256 vk1x01234567 = _mm256_load_ps(w + 16);
        vacc01234567p0 = _mm256_fmadd_ps(vi1x01234567, vk1x01234567, vacc01234567p0);

        const __m256 vi2x01234567 = _mm256_loadu_ps(i2);
        i2 += 8;

        const __m256 vk2x01234567 = _mm256_load_ps(w + 24);
        vacc01234567p0 = _mm256_fmadd_ps(vi2x01234567, vk2x01234567, vacc01234567p0);

        const __m256 vi3x01234567 = _mm256_loadu_ps(i3);
        i3 += 8;

        const __m256 vk3x01234567 = _mm256_load_ps(w + 32);
        vacc01234567p0 = _mm256_fmadd_ps(vi3x01234567, vk3x01234567, vacc01234567p0);

        const __m256 vi4x01234567 = _mm256_loadu_ps(i4);
        i4 += 8;

        const __m256 vk4x01234567 = _mm256_load_ps(w + 40);
        vacc01234567p0 = _mm256_fmadd_ps(vi4x01234567, vk4x01234567, vacc01234567p0);

        w += 48;


        _mm256_store_ps(b, vacc01234567p0);
        b += 8;
      }

      if (c != 0) {
        assert(c >= 1);
        assert(c <= 7);
        const __m256i vmask = _mm256_loadu_si256((const __m256i*) &mask_table[7 - c]);
        __m256 vacc01234567p0 = _mm256_load_ps(w);


        const __m256 vi0x01234567 = _mm256_maskload_ps(i0, vmask);

        const __m256 vk0x01234567 = _mm256_load_ps(w + 8);
        vacc01234567p0 = _mm256_fmadd_ps(vi0x01234567, vk0x01234567, vacc01234567p0);

        const __m256 vi1x01234567 = _mm256_maskload_ps(i1, vmask);

        const __m256 vk1x01234567 = _mm256_load_ps(w + 16);
        vacc01234567p0 = _mm256_fmadd_ps(vi1x01234567, vk1x01234567, vacc01234567p0);

        const __m256 vi2x01234567 = _mm256_maskload_ps(i2, vmask);

        const __m256 vk2x01234567 = _mm256_load_ps(w + 24);
        vacc01234567p0 = _mm256_fmadd_ps(vi2x01234567, vk2x01234567, vacc01234567p0);

        const __m256 vi3x01234567 = _mm256_maskload_ps(i3, vmask);

        const __m256 vk3x01234567 = _mm256_load_ps(w + 32);
        vacc01234567p0 = _mm256_fmadd_ps(vi3x01234567, vk3x01234567, vacc01234567p0);

        const __m256 vi4x01234567 = _mm256_maskload_ps(i4, vmask);

        const __m256 vk4x01234567 = _mm256_load_ps(w + 40);
        vacc01234567p0 = _mm256_fmadd_ps(vi4x01234567, vk4x01234567, vacc01234567p0);

        w += 48;


        _mm256_store_ps(b, vacc01234567p0);
      }
    }

    // Middle pass to process 5 inputs in each iteration.
    for (size_t ks = kernel_size - 5; ks > 5; ks -= 5) {
      float* b = buffer;
      const float* i0 = input[0];
      assert(i0 != NULL);
      if XNN_UNPREDICTABLE(i0 != zero) {
        i0 = (const float*) ((uintptr_t) i0 + input_offset);
      }
      const float* i1 = input[1];
      assert(i1 != NULL);
      if XNN_UNPREDICTABLE(i1 != zero) {
        i1 = (const float*) ((uintptr_t) i1 + input_offset);
      }
      const float* i2 = input[2];
      assert(i2 != NULL);
      if XNN_UNPREDICTABLE(i2 != zero) {
        i2 = (const float*) ((uintptr_t) i2 + input_offset);
      }
      const float* i3 = input[3];
      assert(i3 != NULL);
      if XNN_UNPREDICTABLE(i3 != zero) {
        i3 = (const float*) ((uintptr_t) i3 + input_offset);
      }
      const float* i4 = input[4];
      assert(i4 != NULL);
      if XNN_UNPREDICTABLE(i4 != zero) {
        i4 = (const float*) ((uintptr_t) i4 + input_offset);
      }
      input += 5;

      size_t c = round_up_po2(channels, 4);

      for (; c >= 8; c -= 8) {
        __m256 vacc01234567p0 = _mm256_load_ps(b);


        const __m256 vi0x01234567 = _mm256_loadu_ps(i0);
        i0 += 8;

        const __m256 vk0x01234567 = _mm256_load_ps(w);
        vacc01234567p0 = _mm256_fmadd_ps(vi0x01234567, vk0x01234567, vacc01234567p0);

        const __m256 vi1x01234567 = _mm256_loadu_ps(i1);
        i1 += 8;

        const __m256 vk1x01234567 = _mm256_load_ps(w + 8);
        vacc01234567p0 = _mm256_fmadd_ps(vi1x01234567, vk1x01234567, vacc01234567p0);

        const __m256 vi2x01234567 = _mm256_loadu_ps(i2);
        i2 += 8;

        const __m256 vk2x01234567 = _mm256_load_ps(w + 16);
        vacc01234567p0 = _mm256_fmadd_ps(vi2x01234567, vk2x01234567, vacc01234567p0);

        const __m256 vi3x01234567 = _mm256_loadu_ps(i3);
        i3 += 8;

        const __m256 vk3x01234567 = _mm256_load_ps(w + 24);
        vacc01234567p0 = _mm256_fmadd_ps(vi3x01234567, vk3x01234567, vacc01234567p0);

        const __m256 vi4x01234567 = _mm256_loadu_ps(i4);
        i4 += 8;

        const __m256 vk4x01234567 = _mm256_load_ps(w + 32);
        vacc01234567p0 = _mm256_fmadd_ps(vi4x01234567, vk4x01234567, vacc01234567p0);

        w += 40;


        _mm256_store_ps(b, vacc01234567p0);
        b += 8;
      }

      if (c != 0) {
        assert(c >= 1);
        assert(c <= 7);
        const __m256i vmask = _mm256_loadu_si256((const __m256i*) &mask_table[7 - c]);
        __m256 vacc01234567p0 = _mm256_load_ps(b);


        const __m256 vi0x01234567 = _mm256_maskload_ps(i0, vmask);

        const __m256 vk0x01234567 = _mm256_load_ps(w);
        vacc01234567p0 = _mm256_fmadd_ps(vi0x01234567, vk0x01234567, vacc01234567p0);

        const __m256 vi1x01234567 = _mm256_maskload_ps(i1, vmask);

        const __m256 vk1x01234567 = _mm256_load_ps(w + 8);
        vacc01234567p0 = _mm256_fmadd_ps(vi1x01234567, vk1x01234567, vacc01234567p0);

        const __m256 vi2x01234567 = _mm256_maskload_ps(i2, vmask);

        const __m256 vk2x01234567 = _mm256_load_ps(w + 16);
        vacc01234567p0 = _mm256_fmadd_ps(vi2x01234567, vk2x01234567, vacc01234567p0);

        const __m256 vi3x01234567 = _mm256_maskload_ps(i3, vmask);

        const __m256 vk3x01234567 = _mm256_load_ps(w + 24);
        vacc01234567p0 = _mm256_fmadd_ps(vi3x01234567, vk3x01234567, vacc01234567p0);

        const __m256 vi4x01234567 = _mm256_maskload_ps(i4, vmask);

        const __m256 vk4x01234567 = _mm256_load_ps(w + 32);
        vacc01234567p0 = _mm256_fmadd_ps(vi4x01234567, vk4x01234567, vacc01234567p0);

        w += 40;


        _mm256_store_ps(b, vacc01234567p0);
      }
    }

    // Last pass to process up to 5 inputs.
    {
      float* b = buffer;
      const float* i0 = input[0];
      assert(i0 != NULL);
      if XNN_UNPREDICTABLE(i0 != zero) {
        i0 = (const float*) ((uintptr_t) i0 + input_offset);
      }
      const float* i1 = input[1];
      assert(i1 != NULL);
      if XNN_UNPREDICTABLE(i1 != zero) {
        i1 = (const float*) ((uintptr_t) i1 + input_offset);
      }
      const float* i2 = input[2];
      assert(i2 != NULL);
      if XNN_UNPREDICTABLE(i2 != zero) {
        i2 = (const float*) ((uintptr_t) i2 + input_offset);
      }
      const float* i3 = input[3];
      assert(i3 != NULL);
      if XNN_UNPREDICTABLE(i3 != zero) {
        i3 = (const float*) ((uintptr_t) i3 + input_offset);
      }
      const float* i4 = input[4];
      assert(i4 != NULL);
      if XNN_UNPREDICTABLE(i4 != zero) {
        i4 = (const float*) ((uintptr_t) i4 + input_offset);
      }

      size_t c = channels;


      for (; c >= 8; c -= 8) {
        __m256 vacc01234567p0 = _mm256_load_ps(b);
        b += 8;


        const __m256 vi0x01234567 = _mm256_loadu_ps(i0);
        i0 += 8;

        __m256 vk0x01234567 = _mm256_load_ps(w);

        vacc01234567p0 = _mm256_fmadd_ps(vi0x01234567, vk0x01234567, vacc01234567p0);

        const __m256 vi1x01234567 = _mm256_loadu_ps(i1);
        i1 += 8;

        __m256 vk1x01234567 = _mm256_load_ps(w + 8);

        vacc01234567p0 = _mm256_fmadd_ps(vi1x01234567, vk1x01234567, vacc01234567p0);

        const __m256 vi2x01234567 = _mm256_loadu_ps(i2);
        i2 += 8;

        __m256 vk2x01234567 = _mm256_load_ps(w + 16);

        vacc01234567p0 = _mm256_fmadd_ps(vi2x01234567, vk2x01234567, vacc01234567p0);

        const __m256 vi3x01234567 = _mm256_loadu_ps(i3);
        i3 += 8;

        __m256 vk3x01234567 = _mm256_load_ps(w + 24);

        vacc01234567p0 = _mm256_fmadd_ps(vi3x01234567, vk3x01234567, vacc01234567p0);

        const __m256 vi4x01234567 = _mm256_loadu_ps(i4);
        i4 += 8;

        __m256 vk4x01234567 = _mm256_load_ps(w + 32);

        vacc01234567p0 = _mm256_fmadd_ps(vi4x01234567, vk4x01234567, vacc01234567p0);

        w += 40;



        __m256 vacc01234567 = _mm256_max_ps(vmin, vacc01234567p0);

        vacc01234567 = _mm256_min_ps(vmax, vacc01234567);

        _mm256_storeu_ps(output, vacc01234567);
        output += 8;
      }

      if XNN_UNLIKELY(c != 0) {
        assert(c >= 1);
        assert(c <= 7);
        __m256 vacc01234567p0 = _mm256_load_ps(b);
        const __m256i vmask = _mm256_loadu_si256((const __m256i*) &mask_table[7 - c]);

        const __m256 vi0x01234567 = _mm256_maskload_ps(i0, vmask);
        __m256 vk0x01234567 = _mm256_load_ps(w);
        vacc01234567p0 = _mm256_fmadd_ps(vi0x01234567, vk0x01234567, vacc01234567p0);

        const __m256 vi1x01234567 = _mm256_maskload_ps(i1, vmask);
        __m256 vk1x01234567 = _mm256_load_ps(w + 8);
        vacc01234567p0 = _mm256_fmadd_ps(vi1x01234567, vk1x01234567, vacc01234567p0);

        const __m256 vi2x01234567 = _mm256_maskload_ps(i2, vmask);
        __m256 vk2x01234567 = _mm256_load_ps(w + 16);
        vacc01234567p0 = _mm256_fmadd_ps(vi2x01234567, vk2x01234567, vacc01234567p0);

        const __m256 vi3x01234567 = _mm256_maskload_ps(i3, vmask);
        __m256 vk3x01234567 = _mm256_load_ps(w + 24);
        vacc01234567p0 = _mm256_fmadd_ps(vi3x01234567, vk3x01234567, vacc01234567p0);

        const __m256 vi4x01234567 = _mm256_maskload_ps(i4, vmask);
        __m256 vk4x01234567 = _mm256_load_ps(w + 32);
        vacc01234567p0 = _mm256_fmadd_ps(vi4x01234567, vk4x01234567, vacc01234567p0);


        __m256 vacc01234567 = _mm256_max_ps(vmin, vacc01234567p0);
        vacc01234567 = _mm256_min_ps(vmax, vacc01234567);

        __m128 vacc0123 = _mm256_castps256_ps128(vacc01234567);
        if (c & 4) {
          _mm_storeu_ps(output, vacc0123);
          vacc0123 = _mm256_extractf128_ps(vacc01234567, 1);
          output += 4;
        }
        if (c & 2) {
          _mm_storel_pi((__m64*) output, vacc0123);
          vacc0123 = _mm_movehl_ps(vacc0123, vacc0123);
          output += 2;
        }
        if (c & 1) {
          _mm_store_ss(output, vacc0123);
          output += 1;
        }
      }

    }
    input = (const float**) ((uintptr_t) input + input_stride);
    output = (float*) ((uintptr_t) output + output_increment);
  } while (--output_width != 0);
}

void xnn_f32_dwconv_minmax_ukernel_9p16c__fma3(
    size_t channels,
    size_t output_width,
    const float** input,
    const float* weights,
    float* output,
    intptr_t input_stride,
    size_t output_increment,
    size_t input_offset,
    const float* zero,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(channels != 0);
  assert(output_width != 0);

  static const int32_t mask_table[14] = {-1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0};

  const __m256 vmin = _mm256_set1_ps(params->avx.min);
  const __m256 vmax = _mm256_set1_ps(params->avx.max);
  XNN_FORCE_REALIZATION(vmin);
  XNN_FORCE_REALIZATION(vmax);
  do {
    const float* i0 = input[0];
    assert(i0 != NULL);
    if XNN_UNPREDICTABLE(i0 != zero) {
      i0 = (const float*) ((uintptr_t) i0 + input_offset);
    }
    const float* i1 = input[1];
    assert(i1 != NULL);
    if XNN_UNPREDICTABLE(i1 != zero) {
      i1 = (const float*) ((uintptr_t) i1 + input_offset);
    }
    const float* i2 = input[2];
    assert(i2 != NULL);
    if XNN_UNPREDICTABLE(i2 != zero) {
      i2 = (const float*) ((uintptr_t) i2 + input_offset);
    }
    const float* i3 = input[3];
    assert(i3 != NULL);
    if XNN_UNPREDICTABLE(i3 != zero) {
      i3 = (const float*) ((uintptr_t) i3 + input_offset);
    }
    const float* i4 = input[4];
    assert(i4 != NULL);
    if XNN_UNPREDICTABLE(i4 != zero) {
      i4 = (const float*) ((uintptr_t) i4 + input_offset);
    }
    const float* i5 = input[5];
    assert(i5 != NULL);
    if XNN_UNPREDICTABLE(i5 != zero) {
      i5 = (const float*) ((uintptr_t) i5 + input_offset);
    }
    const float* i6 = input[6];
    assert(i6 != NULL);
    if XNN_UNPREDICTABLE(i6 != zero) {
      i6 = (const float*) ((uintptr_t) i6 + input_offset);
    }
    const float* i7 = input[7];
    assert(i7 != NULL);
    if XNN_UNPREDICTABLE(i7 != zero) {
      i7 = (const float*) ((uintptr_t) i7 + input_offset);
    }
    const float* i8 = input[8];
    assert(i8 != NULL);
    if XNN_UNPREDICTABLE(i8 != zero) {
      i8 = (const float*) ((uintptr_t) i8 + input_offset);
    }
    input = (const float**) ((uintptr_t) input + input_stride);

    size_t c = channels;
    const float* w = weights;
    for (; c >= 16; c -= 16) {
      __m256 vacc01234567p0 = _mm256_load_ps(w);
      __m256 vacc89ABCDEFp0 = _mm256_load_ps(w + 8);


      const __m256 vi0x01234567 = _mm256_loadu_ps(i0);
      const __m256 vi0x89ABCDEF = _mm256_loadu_ps(i0 + 8);
      i0 += 16;

      const __m256 vk0x01234567 = _mm256_load_ps(w + 16);
      const __m256 vk0x89ABCDEF = _mm256_load_ps(w + 24);
      vacc01234567p0 = _mm256_fmadd_ps(vi0x01234567, vk0x01234567, vacc01234567p0);
      vacc89ABCDEFp0 = _mm256_fmadd_ps(vi0x89ABCDEF, vk0x89ABCDEF, vacc89ABCDEFp0);

      const __m256 vi1x01234567 = _mm256_loadu_ps(i1);
      const __m256 vi1x89ABCDEF = _mm256_loadu_ps(i1 + 8);
      i1 += 16;

      const __m256 vk1x01234567 = _mm256_load_ps(w + 32);
      const __m256 vk1x89ABCDEF = _mm256_load_ps(w + 40);
      vacc01234567p0 = _mm256_fmadd_ps(vi1x01234567, vk1x01234567, vacc01234567p0);
      vacc89ABCDEFp0 = _mm256_fmadd_ps(vi1x89ABCDEF, vk1x89ABCDEF, vacc89ABCDEFp0);

      const __m256 vi2x01234567 = _mm256_loadu_ps(i2);
      const __m256 vi2x89ABCDEF = _mm256_loadu_ps(i2 + 8);
      i2 += 16;

      const __m256 vk2x01234567 = _mm256_load_ps(w + 48);
      const __m256 vk2x89ABCDEF = _mm256_load_ps(w + 56);
      vacc01234567p0 = _mm256_fmadd_ps(vi2x01234567, vk2x01234567, vacc01234567p0);
      vacc89ABCDEFp0 = _mm256_fmadd_ps(vi2x89ABCDEF, vk2x89ABCDEF, vacc89ABCDEFp0);

      const __m256 vi3x01234567 = _mm256_loadu_ps(i3);
      const __m256 vi3x89ABCDEF = _mm256_loadu_ps(i3 + 8);
      i3 += 16;

      const __m256 vk3x01234567 = _mm256_load_ps(w + 64);
      const __m256 vk3x89ABCDEF = _mm256_load_ps(w + 72);
      vacc01234567p0 = _mm256_fmadd_ps(vi3x01234567, vk3x01234567, vacc01234567p0);
      vacc89ABCDEFp0 = _mm256_fmadd_ps(vi3x89ABCDEF, vk3x89ABCDEF, vacc89ABCDEFp0);

      const __m256 vi4x01234567 = _mm256_loadu_ps(i4);
      const __m256 vi4x89ABCDEF = _mm256_loadu_ps(i4 + 8);
      i4 += 16;

      const __m256 vk4x01234567 = _mm256_load_ps(w + 80);
      const __m256 vk4x89ABCDEF = _mm256_load_ps(w + 88);
      vacc01234567p0 = _mm256_fmadd_ps(vi4x01234567, vk4x01234567, vacc01234567p0);
      vacc89ABCDEFp0 = _mm256_fmadd_ps(vi4x89ABCDEF, vk4x89ABCDEF, vacc89ABCDEFp0);

      const __m256 vi5x01234567 = _mm256_loadu_ps(i5);
      const __m256 vi5x89ABCDEF = _mm256_loadu_ps(i5 + 8);
      i5 += 16;

      const __m256 vk5x01234567 = _mm256_load_ps(w + 96);
      const __m256 vk5x89ABCDEF = _mm256_load_ps(w + 104);
      vacc01234567p0 = _mm256_fmadd_ps(vi5x01234567, vk5x01234567, vacc01234567p0);
      vacc89ABCDEFp0 = _mm256_fmadd_ps(vi5x89ABCDEF, vk5x89ABCDEF, vacc89ABCDEFp0);

      const __m256 vi6x01234567 = _mm256_loadu_ps(i6);
      const __m256 vi6x89ABCDEF = _mm256_loadu_ps(i6 + 8);
      i6 += 16;

      const __m256 vk6x01234567 = _mm256_load_ps(w + 112);
      const __m256 vk6x89ABCDEF = _mm256_load_ps(w + 120);
      vacc01234567p0 = _mm256_fmadd_ps(vi6x01234567, vk6x01234567, vacc01234567p0);
      vacc89ABCDEFp0 = _mm256_fmadd_ps(vi6x89ABCDEF, vk6x89ABCDEF, vacc89ABCDEFp0);

      const __m256 vi7x01234567 = _mm256_loadu_ps(i7);
      const __m256 vi7x89ABCDEF = _mm256_loadu_ps(i7 + 8);
      i7 += 16;

      const __m256 vk7x01234567 = _mm256_load_ps(w + 128);
      const __m256 vk7x89ABCDEF = _mm256_load_ps(w + 136);
      vacc01234567p0 = _mm256_fmadd_ps(vi7x01234567, vk7x01234567, vacc01234567p0);
      vacc89ABCDEFp0 = _mm256_fmadd_ps(vi7x89ABCDEF, vk7x89ABCDEF, vacc89ABCDEFp0);

      const __m256 vi8x01234567 = _mm256_loadu_ps(i8);
      const __m256 vi8x89ABCDEF = _mm256_loadu_ps(i8 + 8);
      i8 += 16;

      const __m256 vk8x01234567 = _mm256_load_ps(w + 144);
      const __m256 vk8x89ABCDEF = _mm256_load_ps(w + 152);
      vacc01234567p0 = _mm256_fmadd_ps(vi8x01234567, vk8x01234567, vacc01234567p0);
      vacc89ABCDEFp0 = _mm256_fmadd_ps(vi8x89ABCDEF, vk8x89ABCDEF, vacc89ABCDEFp0);

      w += 160;


      __m256 vacc01234567 = _mm256_max_ps(vmin, vacc01234567p0);
      __m256 vacc89ABCDEF = _mm256_max_ps(vmin, vacc89ABCDEFp0);
      vacc01234567 = _mm256_min_ps(vmax, vacc01234567);
      vacc89ABCDEF = _mm256_min_ps(vmax, vacc89ABCDEF);

      _mm256_storeu_ps(output, vacc01234567);
      _mm256_storeu_ps(output + 8, vacc89ABCDEF);
      output += 16;
    }
    for (; c >= 8; c -= 8) {
      __m256 vacc01234567p0 = _mm256_load_ps(w);

      const __m256 vi0x01234567 = _mm256_loadu_ps(i0);
      i0 += 8;

      const __m256 vk0x01234567 = _mm256_load_ps(w + 16);
      vacc01234567p0 = _mm256_fmadd_ps(vi0x01234567, vk0x01234567, vacc01234567p0);

      const __m256 vi1x01234567 = _mm256_loadu_ps(i1);
      i1 += 8;

      const __m256 vk1x01234567 = _mm256_load_ps(w + 32);
      vacc01234567p0 = _mm256_fmadd_ps(vi1x01234567, vk1x01234567, vacc01234567p0);

      const __m256 vi2x01234567 = _mm256_loadu_ps(i2);
      i2 += 8;

      const __m256 vk2x01234567 = _mm256_load_ps(w + 48);
      vacc01234567p0 = _mm256_fmadd_ps(vi2x01234567, vk2x01234567, vacc01234567p0);

      const __m256 vi3x01234567 = _mm256_loadu_ps(i3);
      i3 += 8;

      const __m256 vk3x01234567 = _mm256_load_ps(w + 64);
      vacc01234567p0 = _mm256_fmadd_ps(vi3x01234567, vk3x01234567, vacc01234567p0);

      const __m256 vi4x01234567 = _mm256_loadu_ps(i4);
      i4 += 8;

      const __m256 vk4x01234567 = _mm256_load_ps(w + 80);
      vacc01234567p0 = _mm256_fmadd_ps(vi4x01234567, vk4x01234567, vacc01234567p0);

      const __m256 vi5x01234567 = _mm256_loadu_ps(i5);
      i5 += 8;

      const __m256 vk5x01234567 = _mm256_load_ps(w + 96);
      vacc01234567p0 = _mm256_fmadd_ps(vi5x01234567, vk5x01234567, vacc01234567p0);

      const __m256 vi6x01234567 = _mm256_loadu_ps(i6);
      i6 += 8;

      const __m256 vk6x01234567 = _mm256_load_ps(w + 112);
      vacc01234567p0 = _mm256_fmadd_ps(vi6x01234567, vk6x01234567, vacc01234567p0);

      const __m256 vi7x01234567 = _mm256_loadu_ps(i7);
      i7 += 8;

      const __m256 vk7x01234567 = _mm256_load_ps(w + 128);
      vacc01234567p0 = _mm256_fmadd_ps(vi7x01234567, vk7x01234567, vacc01234567p0);

      const __m256 vi8x01234567 = _mm256_loadu_ps(i8);
      i8 += 8;

      const __m256 vk8x01234567 = _mm256_load_ps(w + 144);
      vacc01234567p0 = _mm256_fmadd_ps(vi8x01234567, vk8x01234567, vacc01234567p0);

      w += 8;


      __m256 vacc01234567 = _mm256_max_ps(vmin, vacc01234567p0);
      vacc01234567 = _mm256_min_ps(vmax, vacc01234567);

      _mm256_storeu_ps(output, vacc01234567);
      output += 8;
    }
    if XNN_UNLIKELY(c != 0) {
      assert(c >= 1);
      assert(c <= 7);
      const __m256i vmask = _mm256_loadu_si256((const __m256i*) &mask_table[7 - c]);

      __m256 vacc01234567p0 = _mm256_load_ps(w);

      const __m256 vi0x01234567 = _mm256_maskload_ps(i0, vmask);
      const __m256 vk0x01234567 = _mm256_load_ps(w + 16);
      vacc01234567p0 = _mm256_fmadd_ps(vi0x01234567, vk0x01234567, vacc01234567p0);

      const __m256 vi1x01234567 = _mm256_maskload_ps(i1, vmask);
      const __m256 vk1x01234567 = _mm256_load_ps(w + 32);
      vacc01234567p0 = _mm256_fmadd_ps(vi1x01234567, vk1x01234567, vacc01234567p0);

      const __m256 vi2x01234567 = _mm256_maskload_ps(i2, vmask);
      const __m256 vk2x01234567 = _mm256_load_ps(w + 48);
      vacc01234567p0 = _mm256_fmadd_ps(vi2x01234567, vk2x01234567, vacc01234567p0);

      const __m256 vi3x01234567 = _mm256_maskload_ps(i3, vmask);
      const __m256 vk3x01234567 = _mm256_load_ps(w + 64);
      vacc01234567p0 = _mm256_fmadd_ps(vi3x01234567, vk3x01234567, vacc01234567p0);

      const __m256 vi4x01234567 = _mm256_maskload_ps(i4, vmask);
      const __m256 vk4x01234567 = _mm256_load_ps(w + 80);
      vacc01234567p0 = _mm256_fmadd_ps(vi4x01234567, vk4x01234567, vacc01234567p0);

      const __m256 vi5x01234567 = _mm256_maskload_ps(i5, vmask);
      const __m256 vk5x01234567 = _mm256_load_ps(w + 96);
      vacc01234567p0 = _mm256_fmadd_ps(vi5x01234567, vk5x01234567, vacc01234567p0);

      const __m256 vi6x01234567 = _mm256_maskload_ps(i6, vmask);
      const __m256 vk6x01234567 = _mm256_load_ps(w + 112);
      vacc01234567p0 = _mm256_fmadd_ps(vi6x01234567, vk6x01234567, vacc01234567p0);

      const __m256 vi7x01234567 = _mm256_maskload_ps(i7, vmask);
      const __m256 vk7x01234567 = _mm256_load_ps(w + 128);
      vacc01234567p0 = _mm256_fmadd_ps(vi7x01234567, vk7x01234567, vacc01234567p0);

      const __m256 vi8x01234567 = _mm256_maskload_ps(i8, vmask);
      const __m256 vk8x01234567 = _mm256_load_ps(w + 144);
      vacc01234567p0 = _mm256_fmadd_ps(vi8x01234567, vk8x01234567, vacc01234567p0);


      __m256 vacc01234567 = _mm256_max_ps(vmin, vacc01234567p0);
      vacc01234567 = _mm256_min_ps(vmax, vacc01234567);

      __m128 vacc0123 = _mm256_castps256_ps128(vacc01234567);
      if (c & 4) {
        _mm_storeu_ps(output, vacc0123);
        vacc0123 = _mm256_extractf128_ps(vacc01234567, 1);
        output += 4;
      }
      if (c & 2) {
        _mm_storel_pi((__m64*) output, vacc0123);
        vacc0123 = _mm_movehl_ps(vacc0123, vacc0123);
        output += 2;
      }
      if (c & 1) {
        _mm_store_ss(output, vacc0123);
        output += 1;
      }
    }

    output = (float*) ((uintptr_t) output + output_increment);
  } while (--output_width != 0);
}

void xnn_f32_gemm_minmax_ukernel_1x16__fma3_broadcast(
    size_t mr,
    size_t nc,
    size_t kc,
    const float* restrict a,
    size_t a_stride,
    const float* restrict w,
    float* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(mr != 0);
  assert(mr <= 1);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(float) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  const float* a0 = a;
  float* c0 = c;

  const __m256 vmin = _mm256_set1_ps(params->avx.min);
  const __m256 vmax = _mm256_set1_ps(params->avx.max);
  XNN_FORCE_REALIZATION(vmin);
  XNN_FORCE_REALIZATION(vmax);

  do {
    __m256 vacc0x01234567 = _mm256_load_ps(w + 0);
    __m256 vacc0x89ABCDEF = _mm256_load_ps(w + 8);
    w += 16;

    size_t k = kc;
    do {
      const __m256 va0 = _mm256_broadcast_ss(a0);
      a0 += 1;

      const __m256 vb01234567 = _mm256_load_ps(w);
      const __m256 vb89ABCDEF = _mm256_load_ps(w + 8);
      w += 16;

      vacc0x01234567 = _mm256_fmadd_ps(va0, vb01234567, vacc0x01234567);
      vacc0x89ABCDEF = _mm256_fmadd_ps(va0, vb89ABCDEF, vacc0x89ABCDEF);

      k -= sizeof(float);
    } while (k != 0);

    vacc0x01234567 = _mm256_max_ps(vmin, vacc0x01234567);
    vacc0x89ABCDEF = _mm256_max_ps(vmin, vacc0x89ABCDEF);

    vacc0x01234567 = _mm256_min_ps(vmax, vacc0x01234567);
    vacc0x89ABCDEF = _mm256_min_ps(vmax, vacc0x89ABCDEF);

    if XNN_LIKELY(nc >= 16) {
      _mm256_storeu_ps(c0, vacc0x01234567);
      _mm256_storeu_ps(c0 + 8, vacc0x89ABCDEF);
      c0 = (float*) ((uintptr_t) c0 + cn_stride);

      a0 = (const float*) ((uintptr_t) a0 - kc);

      nc -= 16;
    } else {
      if (nc & 8) {
        _mm256_storeu_ps(c0, vacc0x01234567);

        vacc0x01234567 = vacc0x89ABCDEF;

        c0 += 8;
      }
      __m128 vacc0x0123 = _mm256_castps256_ps128(vacc0x01234567);
      if (nc & 4) {
        _mm_storeu_ps(c0, vacc0x0123);

        vacc0x0123 = _mm256_extractf128_ps(vacc0x01234567, 1);

        c0 += 4;
      }
      if (nc & 2) {
        _mm_storel_pi((__m64*) c0, vacc0x0123);

        vacc0x0123 = _mm_movehl_ps(vacc0x0123, vacc0x0123);

        c0 += 2;
      }
      if (nc & 1) {
        _mm_store_ss(c0, vacc0x0123);
      }

      nc = 0;
    }
  } while (nc != 0);
}

void xnn_f32_gemm_minmax_ukernel_1x16s4__fma3_broadcast(
    size_t mr,
    size_t nc,
    size_t kc,
    const float* restrict a,
    size_t a_stride,
    const float* restrict w,
    float* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(mr != 0);
  assert(mr <= 1);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(float) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  const float* a0 = a;
  float* c0 = c;

  const __m256 vmin = _mm256_set1_ps(params->avx.min);
  const __m256 vmax = _mm256_set1_ps(params->avx.max);
  XNN_FORCE_REALIZATION(vmin);
  XNN_FORCE_REALIZATION(vmax);

  do {
    __m256 vacc0x01234567 = _mm256_load_ps(w + 0);
    __m256 vacc0x89ABCDEF = _mm256_load_ps(w + 8);
    w += 16;

    size_t k = kc;
    while (k >= 4 * sizeof(float)) {
      __m256 va0 = _mm256_broadcast_ps((const __m128*) a0);
      a0 += 4;

      const __m256 vb01234567c0 = _mm256_load_ps(w + 0);
      const __m256 vb89ABCDEFc0 = _mm256_load_ps(w + 8);

      vacc0x01234567 = _mm256_fmadd_ps(va0, vb01234567c0, vacc0x01234567);
      vacc0x89ABCDEF = _mm256_fmadd_ps(va0, vb89ABCDEFc0, vacc0x89ABCDEF);

      va0 = _mm256_permute_ps(va0, _MM_SHUFFLE(0, 3, 2, 1));
      const __m256 vb01234567c1 = _mm256_load_ps(w + 16);
      const __m256 vb89ABCDEFc1 = _mm256_load_ps(w + 24);

      vacc0x01234567 = _mm256_fmadd_ps(va0, vb01234567c1, vacc0x01234567);
      vacc0x89ABCDEF = _mm256_fmadd_ps(va0, vb89ABCDEFc1, vacc0x89ABCDEF);

      va0 = _mm256_permute_ps(va0, _MM_SHUFFLE(0, 3, 2, 1));
      const __m256 vb01234567c2 = _mm256_load_ps(w + 32);
      const __m256 vb89ABCDEFc2 = _mm256_load_ps(w + 40);

      vacc0x01234567 = _mm256_fmadd_ps(va0, vb01234567c2, vacc0x01234567);
      vacc0x89ABCDEF = _mm256_fmadd_ps(va0, vb89ABCDEFc2, vacc0x89ABCDEF);

      va0 = _mm256_permute_ps(va0, _MM_SHUFFLE(0, 3, 2, 1));
      const __m256 vb01234567c3 = _mm256_load_ps(w + 48);
      const __m256 vb89ABCDEFc3 = _mm256_load_ps(w + 56);

      vacc0x01234567 = _mm256_fmadd_ps(va0, vb01234567c3, vacc0x01234567);
      vacc0x89ABCDEF = _mm256_fmadd_ps(va0, vb89ABCDEFc3, vacc0x89ABCDEF);


      w += 64;
      k -= 4 * sizeof(float);
    }
    if XNN_UNLIKELY(k != 0) {
      __m256 va0 = _mm256_broadcast_ps((const __m128*) a0);
      a0 = (const float*) ((uintptr_t) a0 + k);

      const __m256 vzero = _mm256_setzero_ps();
      const __m256 vb01234567c0 = _mm256_load_ps(w + 0);
      const __m256 vb89ABCDEFc0 = _mm256_load_ps(w + 8);

      vacc0x01234567 = _mm256_fmadd_ps(_mm256_and_ps(va0, _mm256_cmp_ps(vb01234567c0, vzero, _CMP_NEQ_OQ)), vb01234567c0, vacc0x01234567);
      vacc0x89ABCDEF = _mm256_fmadd_ps(_mm256_and_ps(va0, _mm256_cmp_ps(vb89ABCDEFc0, vzero, _CMP_NEQ_OQ)), vb89ABCDEFc0, vacc0x89ABCDEF);

      va0 = _mm256_permute_ps(va0, _MM_SHUFFLE(0, 3, 2, 1));
      const __m256 vb01234567c1 = _mm256_load_ps(w + 16);
      const __m256 vb89ABCDEFc1 = _mm256_load_ps(w + 24);

      vacc0x01234567 = _mm256_fmadd_ps(_mm256_and_ps(va0, _mm256_cmp_ps(vb01234567c1, vzero, _CMP_NEQ_OQ)), vb01234567c1, vacc0x01234567);
      vacc0x89ABCDEF = _mm256_fmadd_ps(_mm256_and_ps(va0, _mm256_cmp_ps(vb89ABCDEFc1, vzero, _CMP_NEQ_OQ)), vb89ABCDEFc1, vacc0x89ABCDEF);

      va0 = _mm256_permute_ps(va0, _MM_SHUFFLE(0, 3, 2, 1));
      const __m256 vb01234567c2 = _mm256_load_ps(w + 32);
      const __m256 vb89ABCDEFc2 = _mm256_load_ps(w + 40);

      vacc0x01234567 = _mm256_fmadd_ps(_mm256_and_ps(va0, _mm256_cmp_ps(vb01234567c2, vzero, _CMP_NEQ_OQ)), vb01234567c2, vacc0x01234567);
      vacc0x89ABCDEF = _mm256_fmadd_ps(_mm256_and_ps(va0, _mm256_cmp_ps(vb89ABCDEFc2, vzero, _CMP_NEQ_OQ)), vb89ABCDEFc2, vacc0x89ABCDEF);

      va0 = _mm256_permute_ps(va0, _MM_SHUFFLE(0, 3, 2, 1));
      const __m256 vb01234567c3 = _mm256_load_ps(w + 48);
      const __m256 vb89ABCDEFc3 = _mm256_load_ps(w + 56);

      vacc0x01234567 = _mm256_fmadd_ps(_mm256_and_ps(va0, _mm256_cmp_ps(vb01234567c3, vzero, _CMP_NEQ_OQ)), vb01234567c3, vacc0x01234567);
      vacc0x89ABCDEF = _mm256_fmadd_ps(_mm256_and_ps(va0, _mm256_cmp_ps(vb89ABCDEFc3, vzero, _CMP_NEQ_OQ)), vb89ABCDEFc3, vacc0x89ABCDEF);


      w += 64;
    }

    vacc0x01234567 = _mm256_max_ps(vmin, vacc0x01234567);
    vacc0x89ABCDEF = _mm256_max_ps(vmin, vacc0x89ABCDEF);

    vacc0x01234567 = _mm256_min_ps(vmax, vacc0x01234567);
    vacc0x89ABCDEF = _mm256_min_ps(vmax, vacc0x89ABCDEF);

    if XNN_LIKELY(nc >= 16) {
      _mm256_storeu_ps(c0, vacc0x01234567);
      _mm256_storeu_ps(c0 + 8, vacc0x89ABCDEF);
      c0 = (float*) ((uintptr_t) c0 + cn_stride);

      a0 = (const float*) ((uintptr_t) a0 - kc);

      nc -= 16;
    } else {
      if (nc & 8) {
        _mm256_storeu_ps(c0, vacc0x01234567);

        vacc0x01234567 = vacc0x89ABCDEF;

        c0 += 8;
      }
      __m128 vacc0x0123 = _mm256_castps256_ps128(vacc0x01234567);
      if (nc & 4) {
        _mm_storeu_ps(c0, vacc0x0123);

        vacc0x0123 = _mm256_extractf128_ps(vacc0x01234567, 1);

        c0 += 4;
      }
      if (nc & 2) {
        _mm_storel_pi((__m64*) c0, vacc0x0123);

        vacc0x0123 = _mm_movehl_ps(vacc0x0123, vacc0x0123);

        c0 += 2;
      }
      if (nc & 1) {
        _mm_store_ss(c0, vacc0x0123);
      }

      nc = 0;
    }
  } while (nc != 0);
}

void xnn_f32_gemm_minmax_ukernel_4x16s4__fma3_broadcast(
    size_t mr,
    size_t nc,
    size_t kc,
    const float* restrict a,
    size_t a_stride,
    const float* restrict w,
    float* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(mr != 0);
  assert(mr <= 4);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(float) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  const float* a0 = a;
  float* c0 = c;
  const float* a1 = (const float*) ((uintptr_t) a0 + a_stride);
  float* c1 = (float*) ((uintptr_t) c0 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 2) {
    a1 = a0;
    c1 = c0;
  }
  const float* a2 = (const float*) ((uintptr_t) a1 + a_stride);
  float* c2 = (float*) ((uintptr_t) c1 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 2) {
    a2 = a1;
    c2 = c1;
  }
  const float* a3 = (const float*) ((uintptr_t) a2 + a_stride);
  float* c3 = (float*) ((uintptr_t) c2 + cm_stride);
  if XNN_UNPREDICTABLE(mr != 4) {
    a3 = a2;
    c3 = c2;
  }

  const __m256 vmin = _mm256_set1_ps(params->avx.min);
  const __m256 vmax = _mm256_set1_ps(params->avx.max);
  XNN_FORCE_REALIZATION(vmin);
  XNN_FORCE_REALIZATION(vmax);

  do {
    __m256 vacc0x01234567 = _mm256_load_ps(w + 0);
    __m256 vacc0x89ABCDEF = _mm256_load_ps(w + 8);
    __m256 vacc1x01234567 = vacc0x01234567;
    __m256 vacc1x89ABCDEF = vacc0x89ABCDEF;
    __m256 vacc2x01234567 = vacc0x01234567;
    __m256 vacc2x89ABCDEF = vacc0x89ABCDEF;
    __m256 vacc3x01234567 = vacc0x01234567;
    __m256 vacc3x89ABCDEF = vacc0x89ABCDEF;
    w += 16;

    size_t k = kc;
    while (k >= 4 * sizeof(float)) {
      __m256 va0 = _mm256_broadcast_ps((const __m128*) a0);
      a0 += 4;
      __m256 va1 = _mm256_broadcast_ps((const __m128*) a1);
      a1 += 4;
      __m256 va2 = _mm256_broadcast_ps((const __m128*) a2);
      a2 += 4;
      __m256 va3 = _mm256_broadcast_ps((const __m128*) a3);
      a3 += 4;

      const __m256 vb01234567c0 = _mm256_load_ps(w + 0);
      const __m256 vb89ABCDEFc0 = _mm256_load_ps(w + 8);

      vacc0x01234567 = _mm256_fmadd_ps(va0, vb01234567c0, vacc0x01234567);
      vacc1x01234567 = _mm256_fmadd_ps(va1, vb01234567c0, vacc1x01234567);
      vacc2x01234567 = _mm256_fmadd_ps(va2, vb01234567c0, vacc2x01234567);
      vacc3x01234567 = _mm256_fmadd_ps(va3, vb01234567c0, vacc3x01234567);
      vacc0x89ABCDEF = _mm256_fmadd_ps(va0, vb89ABCDEFc0, vacc0x89ABCDEF);
      vacc1x89ABCDEF = _mm256_fmadd_ps(va1, vb89ABCDEFc0, vacc1x89ABCDEF);
      vacc2x89ABCDEF = _mm256_fmadd_ps(va2, vb89ABCDEFc0, vacc2x89ABCDEF);
      vacc3x89ABCDEF = _mm256_fmadd_ps(va3, vb89ABCDEFc0, vacc3x89ABCDEF);

      va0 = _mm256_permute_ps(va0, _MM_SHUFFLE(0, 3, 2, 1));
      va1 = _mm256_permute_ps(va1, _MM_SHUFFLE(0, 3, 2, 1));
      va2 = _mm256_permute_ps(va2, _MM_SHUFFLE(0, 3, 2, 1));
      va3 = _mm256_permute_ps(va3, _MM_SHUFFLE(0, 3, 2, 1));
      const __m256 vb01234567c1 = _mm256_load_ps(w + 16);
      const __m256 vb89ABCDEFc1 = _mm256_load_ps(w + 24);

      vacc0x01234567 = _mm256_fmadd_ps(va0, vb01234567c1, vacc0x01234567);
      vacc1x01234567 = _mm256_fmadd_ps(va1, vb01234567c1, vacc1x01234567);
      vacc2x01234567 = _mm256_fmadd_ps(va2, vb01234567c1, vacc2x01234567);
      vacc3x01234567 = _mm256_fmadd_ps(va3, vb01234567c1, vacc3x01234567);
      vacc0x89ABCDEF = _mm256_fmadd_ps(va0, vb89ABCDEFc1, vacc0x89ABCDEF);
      vacc1x89ABCDEF = _mm256_fmadd_ps(va1, vb89ABCDEFc1, vacc1x89ABCDEF);
      vacc2x89ABCDEF = _mm256_fmadd_ps(va2, vb89ABCDEFc1, vacc2x89ABCDEF);
      vacc3x89ABCDEF = _mm256_fmadd_ps(va3, vb89ABCDEFc1, vacc3x89ABCDEF);

      va0 = _mm256_permute_ps(va0, _MM_SHUFFLE(0, 3, 2, 1));
      va1 = _mm256_permute_ps(va1, _MM_SHUFFLE(0, 3, 2, 1));
      va2 = _mm256_permute_ps(va2, _MM_SHUFFLE(0, 3, 2, 1));
      va3 = _mm256_permute_ps(va3, _MM_SHUFFLE(0, 3, 2, 1));
      const __m256 vb01234567c2 = _mm256_load_ps(w + 32);
      const __m256 vb89ABCDEFc2 = _mm256_load_ps(w + 40);

      vacc0x01234567 = _mm256_fmadd_ps(va0, vb01234567c2, vacc0x01234567);
      vacc1x01234567 = _mm256_fmadd_ps(va1, vb01234567c2, vacc1x01234567);
      vacc2x01234567 = _mm256_fmadd_ps(va2, vb01234567c2, vacc2x01234567);
      vacc3x01234567 = _mm256_fmadd_ps(va3, vb01234567c2, vacc3x01234567);
      vacc0x89ABCDEF = _mm256_fmadd_ps(va0, vb89ABCDEFc2, vacc0x89ABCDEF);
      vacc1x89ABCDEF = _mm256_fmadd_ps(va1, vb89ABCDEFc2, vacc1x89ABCDEF);
      vacc2x89ABCDEF = _mm256_fmadd_ps(va2, vb89ABCDEFc2, vacc2x89ABCDEF);
      vacc3x89ABCDEF = _mm256_fmadd_ps(va3, vb89ABCDEFc2, vacc3x89ABCDEF);

      va0 = _mm256_permute_ps(va0, _MM_SHUFFLE(0, 3, 2, 1));
      va1 = _mm256_permute_ps(va1, _MM_SHUFFLE(0, 3, 2, 1));
      va2 = _mm256_permute_ps(va2, _MM_SHUFFLE(0, 3, 2, 1));
      va3 = _mm256_permute_ps(va3, _MM_SHUFFLE(0, 3, 2, 1));
      const __m256 vb01234567c3 = _mm256_load_ps(w + 48);
      const __m256 vb89ABCDEFc3 = _mm256_load_ps(w + 56);

      vacc0x01234567 = _mm256_fmadd_ps(va0, vb01234567c3, vacc0x01234567);
      vacc1x01234567 = _mm256_fmadd_ps(va1, vb01234567c3, vacc1x01234567);
      vacc2x01234567 = _mm256_fmadd_ps(va2, vb01234567c3, vacc2x01234567);
      vacc3x01234567 = _mm256_fmadd_ps(va3, vb01234567c3, vacc3x01234567);
      vacc0x89ABCDEF = _mm256_fmadd_ps(va0, vb89ABCDEFc3, vacc0x89ABCDEF);
      vacc1x89ABCDEF = _mm256_fmadd_ps(va1, vb89ABCDEFc3, vacc1x89ABCDEF);
      vacc2x89ABCDEF = _mm256_fmadd_ps(va2, vb89ABCDEFc3, vacc2x89ABCDEF);
      vacc3x89ABCDEF = _mm256_fmadd_ps(va3, vb89ABCDEFc3, vacc3x89ABCDEF);


      w += 64;
      k -= 4 * sizeof(float);
    }
    if XNN_UNLIKELY(k != 0) {
      __m256 va0 = _mm256_broadcast_ps((const __m128*) a0);
      a0 = (const float*) ((uintptr_t) a0 + k);
      __m256 va1 = _mm256_broadcast_ps((const __m128*) a1);
      a1 = (const float*) ((uintptr_t) a1 + k);
      __m256 va2 = _mm256_broadcast_ps((const __m128*) a2);
      a2 = (const float*) ((uintptr_t) a2 + k);
      __m256 va3 = _mm256_broadcast_ps((const __m128*) a3);
      a3 = (const float*) ((uintptr_t) a3 + k);

      const __m256 vzero = _mm256_setzero_ps();
      const __m256 vb01234567c0 = _mm256_load_ps(w + 0);
      const __m256 vb89ABCDEFc0 = _mm256_load_ps(w + 8);

      vacc0x01234567 = _mm256_fmadd_ps(_mm256_and_ps(va0, _mm256_cmp_ps(vb01234567c0, vzero, _CMP_NEQ_OQ)), vb01234567c0, vacc0x01234567);
      vacc1x01234567 = _mm256_fmadd_ps(_mm256_and_ps(va1, _mm256_cmp_ps(vb01234567c0, vzero, _CMP_NEQ_OQ)), vb01234567c0, vacc1x01234567);
      vacc2x01234567 = _mm256_fmadd_ps(_mm256_and_ps(va2, _mm256_cmp_ps(vb01234567c0, vzero, _CMP_NEQ_OQ)), vb01234567c0, vacc2x01234567);
      vacc3x01234567 = _mm256_fmadd_ps(_mm256_and_ps(va3, _mm256_cmp_ps(vb01234567c0, vzero, _CMP_NEQ_OQ)), vb01234567c0, vacc3x01234567);
      vacc0x89ABCDEF = _mm256_fmadd_ps(_mm256_and_ps(va0, _mm256_cmp_ps(vb89ABCDEFc0, vzero, _CMP_NEQ_OQ)), vb89ABCDEFc0, vacc0x89ABCDEF);
      vacc1x89ABCDEF = _mm256_fmadd_ps(_mm256_and_ps(va1, _mm256_cmp_ps(vb89ABCDEFc0, vzero, _CMP_NEQ_OQ)), vb89ABCDEFc0, vacc1x89ABCDEF);
      vacc2x89ABCDEF = _mm256_fmadd_ps(_mm256_and_ps(va2, _mm256_cmp_ps(vb89ABCDEFc0, vzero, _CMP_NEQ_OQ)), vb89ABCDEFc0, vacc2x89ABCDEF);
      vacc3x89ABCDEF = _mm256_fmadd_ps(_mm256_and_ps(va3, _mm256_cmp_ps(vb89ABCDEFc0, vzero, _CMP_NEQ_OQ)), vb89ABCDEFc0, vacc3x89ABCDEF);

      va0 = _mm256_permute_ps(va0, _MM_SHUFFLE(0, 3, 2, 1));
      va1 = _mm256_permute_ps(va1, _MM_SHUFFLE(0, 3, 2, 1));
      va2 = _mm256_permute_ps(va2, _MM_SHUFFLE(0, 3, 2, 1));
      va3 = _mm256_permute_ps(va3, _MM_SHUFFLE(0, 3, 2, 1));
      const __m256 vb01234567c1 = _mm256_load_ps(w + 16);
      const __m256 vb89ABCDEFc1 = _mm256_load_ps(w + 24);

      vacc0x01234567 = _mm256_fmadd_ps(_mm256_and_ps(va0, _mm256_cmp_ps(vb01234567c1, vzero, _CMP_NEQ_OQ)), vb01234567c1, vacc0x01234567);
      vacc1x01234567 = _mm256_fmadd_ps(_mm256_and_ps(va1, _mm256_cmp_ps(vb01234567c1, vzero, _CMP_NEQ_OQ)), vb01234567c1, vacc1x01234567);
      vacc2x01234567 = _mm256_fmadd_ps(_mm256_and_ps(va2, _mm256_cmp_ps(vb01234567c1, vzero, _CMP_NEQ_OQ)), vb01234567c1, vacc2x01234567);
      vacc3x01234567 = _mm256_fmadd_ps(_mm256_and_ps(va3, _mm256_cmp_ps(vb01234567c1, vzero, _CMP_NEQ_OQ)), vb01234567c1, vacc3x01234567);
      vacc0x89ABCDEF = _mm256_fmadd_ps(_mm256_and_ps(va0, _mm256_cmp_ps(vb89ABCDEFc1, vzero, _CMP_NEQ_OQ)), vb89ABCDEFc1, vacc0x89ABCDEF);
      vacc1x89ABCDEF = _mm256_fmadd_ps(_mm256_and_ps(va1, _mm256_cmp_ps(vb89ABCDEFc1, vzero, _CMP_NEQ_OQ)), vb89ABCDEFc1, vacc1x89ABCDEF);
      vacc2x89ABCDEF = _mm256_fmadd_ps(_mm256_and_ps(va2, _mm256_cmp_ps(vb89ABCDEFc1, vzero, _CMP_NEQ_OQ)), vb89ABCDEFc1, vacc2x89ABCDEF);
      vacc3x89ABCDEF = _mm256_fmadd_ps(_mm256_and_ps(va3, _mm256_cmp_ps(vb89ABCDEFc1, vzero, _CMP_NEQ_OQ)), vb89ABCDEFc1, vacc3x89ABCDEF);

      va0 = _mm256_permute_ps(va0, _MM_SHUFFLE(0, 3, 2, 1));
      va1 = _mm256_permute_ps(va1, _MM_SHUFFLE(0, 3, 2, 1));
      va2 = _mm256_permute_ps(va2, _MM_SHUFFLE(0, 3, 2, 1));
      va3 = _mm256_permute_ps(va3, _MM_SHUFFLE(0, 3, 2, 1));
      const __m256 vb01234567c2 = _mm256_load_ps(w + 32);
      const __m256 vb89ABCDEFc2 = _mm256_load_ps(w + 40);

      vacc0x01234567 = _mm256_fmadd_ps(_mm256_and_ps(va0, _mm256_cmp_ps(vb01234567c2, vzero, _CMP_NEQ_OQ)), vb01234567c2, vacc0x01234567);
      vacc1x01234567 = _mm256_fmadd_ps(_mm256_and_ps(va1, _mm256_cmp_ps(vb01234567c2, vzero, _CMP_NEQ_OQ)), vb01234567c2, vacc1x01234567);
      vacc2x01234567 = _mm256_fmadd_ps(_mm256_and_ps(va2, _mm256_cmp_ps(vb01234567c2, vzero, _CMP_NEQ_OQ)), vb01234567c2, vacc2x01234567);
      vacc3x01234567 = _mm256_fmadd_ps(_mm256_and_ps(va3, _mm256_cmp_ps(vb01234567c2, vzero, _CMP_NEQ_OQ)), vb01234567c2, vacc3x01234567);
      vacc0x89ABCDEF = _mm256_fmadd_ps(_mm256_and_ps(va0, _mm256_cmp_ps(vb89ABCDEFc2, vzero, _CMP_NEQ_OQ)), vb89ABCDEFc2, vacc0x89ABCDEF);
      vacc1x89ABCDEF = _mm256_fmadd_ps(_mm256_and_ps(va1, _mm256_cmp_ps(vb89ABCDEFc2, vzero, _CMP_NEQ_OQ)), vb89ABCDEFc2, vacc1x89ABCDEF);
      vacc2x89ABCDEF = _mm256_fmadd_ps(_mm256_and_ps(va2, _mm256_cmp_ps(vb89ABCDEFc2, vzero, _CMP_NEQ_OQ)), vb89ABCDEFc2, vacc2x89ABCDEF);
      vacc3x89ABCDEF = _mm256_fmadd_ps(_mm256_and_ps(va3, _mm256_cmp_ps(vb89ABCDEFc2, vzero, _CMP_NEQ_OQ)), vb89ABCDEFc2, vacc3x89ABCDEF);

      va0 = _mm256_permute_ps(va0, _MM_SHUFFLE(0, 3, 2, 1));
      va1 = _mm256_permute_ps(va1, _MM_SHUFFLE(0, 3, 2, 1));
      va2 = _mm256_permute_ps(va2, _MM_SHUFFLE(0, 3, 2, 1));
      va3 = _mm256_permute_ps(va3, _MM_SHUFFLE(0, 3, 2, 1));
      const __m256 vb01234567c3 = _mm256_load_ps(w + 48);
      const __m256 vb89ABCDEFc3 = _mm256_load_ps(w + 56);

      vacc0x01234567 = _mm256_fmadd_ps(_mm256_and_ps(va0, _mm256_cmp_ps(vb01234567c3, vzero, _CMP_NEQ_OQ)), vb01234567c3, vacc0x01234567);
      vacc1x01234567 = _mm256_fmadd_ps(_mm256_and_ps(va1, _mm256_cmp_ps(vb01234567c3, vzero, _CMP_NEQ_OQ)), vb01234567c3, vacc1x01234567);
      vacc2x01234567 = _mm256_fmadd_ps(_mm256_and_ps(va2, _mm256_cmp_ps(vb01234567c3, vzero, _CMP_NEQ_OQ)), vb01234567c3, vacc2x01234567);
      vacc3x01234567 = _mm256_fmadd_ps(_mm256_and_ps(va3, _mm256_cmp_ps(vb01234567c3, vzero, _CMP_NEQ_OQ)), vb01234567c3, vacc3x01234567);
      vacc0x89ABCDEF = _mm256_fmadd_ps(_mm256_and_ps(va0, _mm256_cmp_ps(vb89ABCDEFc3, vzero, _CMP_NEQ_OQ)), vb89ABCDEFc3, vacc0x89ABCDEF);
      vacc1x89ABCDEF = _mm256_fmadd_ps(_mm256_and_ps(va1, _mm256_cmp_ps(vb89ABCDEFc3, vzero, _CMP_NEQ_OQ)), vb89ABCDEFc3, vacc1x89ABCDEF);
      vacc2x89ABCDEF = _mm256_fmadd_ps(_mm256_and_ps(va2, _mm256_cmp_ps(vb89ABCDEFc3, vzero, _CMP_NEQ_OQ)), vb89ABCDEFc3, vacc2x89ABCDEF);
      vacc3x89ABCDEF = _mm256_fmadd_ps(_mm256_and_ps(va3, _mm256_cmp_ps(vb89ABCDEFc3, vzero, _CMP_NEQ_OQ)), vb89ABCDEFc3, vacc3x89ABCDEF);


      w += 64;
    }

    vacc0x01234567 = _mm256_max_ps(vmin, vacc0x01234567);
    vacc1x01234567 = _mm256_max_ps(vmin, vacc1x01234567);
    vacc2x01234567 = _mm256_max_ps(vmin, vacc2x01234567);
    vacc3x01234567 = _mm256_max_ps(vmin, vacc3x01234567);
    vacc0x89ABCDEF = _mm256_max_ps(vmin, vacc0x89ABCDEF);
    vacc1x89ABCDEF = _mm256_max_ps(vmin, vacc1x89ABCDEF);
    vacc2x89ABCDEF = _mm256_max_ps(vmin, vacc2x89ABCDEF);
    vacc3x89ABCDEF = _mm256_max_ps(vmin, vacc3x89ABCDEF);

    vacc0x01234567 = _mm256_min_ps(vmax, vacc0x01234567);
    vacc1x01234567 = _mm256_min_ps(vmax, vacc1x01234567);
    vacc2x01234567 = _mm256_min_ps(vmax, vacc2x01234567);
    vacc3x01234567 = _mm256_min_ps(vmax, vacc3x01234567);
    vacc0x89ABCDEF = _mm256_min_ps(vmax, vacc0x89ABCDEF);
    vacc1x89ABCDEF = _mm256_min_ps(vmax, vacc1x89ABCDEF);
    vacc2x89ABCDEF = _mm256_min_ps(vmax, vacc2x89ABCDEF);
    vacc3x89ABCDEF = _mm256_min_ps(vmax, vacc3x89ABCDEF);

    if XNN_LIKELY(nc >= 16) {
      _mm256_storeu_ps(c0, vacc0x01234567);
      _mm256_storeu_ps(c0 + 8, vacc0x89ABCDEF);
      c0 = (float*) ((uintptr_t) c0 + cn_stride);
      _mm256_storeu_ps(c1, vacc1x01234567);
      _mm256_storeu_ps(c1 + 8, vacc1x89ABCDEF);
      c1 = (float*) ((uintptr_t) c1 + cn_stride);
      _mm256_storeu_ps(c2, vacc2x01234567);
      _mm256_storeu_ps(c2 + 8, vacc2x89ABCDEF);
      c2 = (float*) ((uintptr_t) c2 + cn_stride);
      _mm256_storeu_ps(c3, vacc3x01234567);
      _mm256_storeu_ps(c3 + 8, vacc3x89ABCDEF);
      c3 = (float*) ((uintptr_t) c3 + cn_stride);

      a0 = (const float*) ((uintptr_t) a0 - kc);
      a1 = (const float*) ((uintptr_t) a1 - kc);
      a2 = (const float*) ((uintptr_t) a2 - kc);
      a3 = (const float*) ((uintptr_t) a3 - kc);

      nc -= 16;
    } else {
      if (nc & 8) {
        _mm256_storeu_ps(c0, vacc0x01234567);
        _mm256_storeu_ps(c1, vacc1x01234567);
        _mm256_storeu_ps(c2, vacc2x01234567);
        _mm256_storeu_ps(c3, vacc3x01234567);

        vacc0x01234567 = vacc0x89ABCDEF;
        vacc1x01234567 = vacc1x89ABCDEF;
        vacc2x01234567 = vacc2x89ABCDEF;
        vacc3x01234567 = vacc3x89ABCDEF;

        c0 += 8;
        c1 += 8;
        c2 += 8;
        c3 += 8;
      }
      __m128 vacc0x0123 = _mm256_castps256_ps128(vacc0x01234567);
      __m128 vacc1x0123 = _mm256_castps256_ps128(vacc1x01234567);
      __m128 vacc2x0123 = _mm256_castps256_ps128(vacc2x01234567);
      __m128 vacc3x0123 = _mm256_castps256_ps128(vacc3x01234567);
      if (nc & 4) {
        _mm_storeu_ps(c0, vacc0x0123);
        _mm_storeu_ps(c1, vacc1x0123);
        _mm_storeu_ps(c2, vacc2x0123);
        _mm_storeu_ps(c3, vacc3x0123);

        vacc0x0123 = _mm256_extractf128_ps(vacc0x01234567, 1);
        vacc1x0123 = _mm256_extractf128_ps(vacc1x01234567, 1);
        vacc2x0123 = _mm256_extractf128_ps(vacc2x01234567, 1);
        vacc3x0123 = _mm256_extractf128_ps(vacc3x01234567, 1);

        c0 += 4;
        c1 += 4;
        c2 += 4;
        c3 += 4;
      }
      if (nc & 2) {
        _mm_storel_pi((__m64*) c0, vacc0x0123);
        _mm_storel_pi((__m64*) c1, vacc1x0123);
        _mm_storel_pi((__m64*) c2, vacc2x0123);
        _mm_storel_pi((__m64*) c3, vacc3x0123);

        vacc0x0123 = _mm_movehl_ps(vacc0x0123, vacc0x0123);
        vacc1x0123 = _mm_movehl_ps(vacc1x0123, vacc1x0123);
        vacc2x0123 = _mm_movehl_ps(vacc2x0123, vacc2x0123);
        vacc3x0123 = _mm_movehl_ps(vacc3x0123, vacc3x0123);

        c0 += 2;
        c1 += 2;
        c2 += 2;
        c3 += 2;
      }
      if (nc & 1) {
        _mm_store_ss(c0, vacc0x0123);
        _mm_store_ss(c1, vacc1x0123);
        _mm_store_ss(c2, vacc2x0123);
        _mm_store_ss(c3, vacc3x0123);
      }

      nc = 0;
    }
  } while (nc != 0);
}

void xnn_f32_gemm_minmax_ukernel_5x16__fma3_broadcast(
    size_t mr,
    size_t nc,
    size_t kc,
    const float* restrict a,
    size_t a_stride,
    const float* restrict w,
    float* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(mr != 0);
  assert(mr <= 5);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(float) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  const float* a0 = a;
  float* c0 = c;
  const float* a1 = (const float*) ((uintptr_t) a0 + a_stride);
  float* c1 = (float*) ((uintptr_t) c0 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 2) {
    a1 = a0;
    c1 = c0;
  }
  const float* a2 = (const float*) ((uintptr_t) a1 + a_stride);
  float* c2 = (float*) ((uintptr_t) c1 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 2) {
    a2 = a1;
    c2 = c1;
  }
  const float* a3 = (const float*) ((uintptr_t) a2 + a_stride);
  float* c3 = (float*) ((uintptr_t) c2 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 4) {
    a3 = a2;
    c3 = c2;
  }
  const float* a4 = (const float*) ((uintptr_t) a3 + a_stride);
  float* c4 = (float*) ((uintptr_t) c3 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 4) {
    a4 = a3;
    c4 = c3;
  }

  const __m256 vmin = _mm256_set1_ps(params->avx.min);
  const __m256 vmax = _mm256_set1_ps(params->avx.max);
  XNN_FORCE_REALIZATION(vmin);
  XNN_FORCE_REALIZATION(vmax);

  do {
    __m256 vacc0x01234567 = _mm256_load_ps(w + 0);
    __m256 vacc0x89ABCDEF = _mm256_load_ps(w + 8);
    __m256 vacc1x01234567 = vacc0x01234567;
    __m256 vacc1x89ABCDEF = vacc0x89ABCDEF;
    __m256 vacc2x01234567 = vacc0x01234567;
    __m256 vacc2x89ABCDEF = vacc0x89ABCDEF;
    __m256 vacc3x01234567 = vacc0x01234567;
    __m256 vacc3x89ABCDEF = vacc0x89ABCDEF;
    __m256 vacc4x01234567 = vacc0x01234567;
    __m256 vacc4x89ABCDEF = vacc0x89ABCDEF;
    w += 16;

    size_t k = kc;
    do {
      const __m256 va0 = _mm256_broadcast_ss(a0);
      a0 += 1;
      const __m256 va1 = _mm256_broadcast_ss(a1);
      a1 += 1;
      const __m256 va2 = _mm256_broadcast_ss(a2);
      a2 += 1;
      const __m256 va3 = _mm256_broadcast_ss(a3);
      a3 += 1;
      const __m256 va4 = _mm256_broadcast_ss(a4);
      a4 += 1;

      const __m256 vb01234567 = _mm256_load_ps(w);
      const __m256 vb89ABCDEF = _mm256_load_ps(w + 8);
      w += 16;

      vacc0x01234567 = _mm256_fmadd_ps(va0, vb01234567, vacc0x01234567);
      vacc1x01234567 = _mm256_fmadd_ps(va1, vb01234567, vacc1x01234567);
      vacc2x01234567 = _mm256_fmadd_ps(va2, vb01234567, vacc2x01234567);
      vacc3x01234567 = _mm256_fmadd_ps(va3, vb01234567, vacc3x01234567);
      vacc4x01234567 = _mm256_fmadd_ps(va4, vb01234567, vacc4x01234567);
      vacc0x89ABCDEF = _mm256_fmadd_ps(va0, vb89ABCDEF, vacc0x89ABCDEF);
      vacc1x89ABCDEF = _mm256_fmadd_ps(va1, vb89ABCDEF, vacc1x89ABCDEF);
      vacc2x89ABCDEF = _mm256_fmadd_ps(va2, vb89ABCDEF, vacc2x89ABCDEF);
      vacc3x89ABCDEF = _mm256_fmadd_ps(va3, vb89ABCDEF, vacc3x89ABCDEF);
      vacc4x89ABCDEF = _mm256_fmadd_ps(va4, vb89ABCDEF, vacc4x89ABCDEF);

      k -= sizeof(float);
    } while (k != 0);

    vacc0x01234567 = _mm256_max_ps(vmin, vacc0x01234567);
    vacc1x01234567 = _mm256_max_ps(vmin, vacc1x01234567);
    vacc2x01234567 = _mm256_max_ps(vmin, vacc2x01234567);
    vacc3x01234567 = _mm256_max_ps(vmin, vacc3x01234567);
    vacc4x01234567 = _mm256_max_ps(vmin, vacc4x01234567);
    vacc0x89ABCDEF = _mm256_max_ps(vmin, vacc0x89ABCDEF);
    vacc1x89ABCDEF = _mm256_max_ps(vmin, vacc1x89ABCDEF);
    vacc2x89ABCDEF = _mm256_max_ps(vmin, vacc2x89ABCDEF);
    vacc3x89ABCDEF = _mm256_max_ps(vmin, vacc3x89ABCDEF);
    vacc4x89ABCDEF = _mm256_max_ps(vmin, vacc4x89ABCDEF);

    vacc0x01234567 = _mm256_min_ps(vmax, vacc0x01234567);
    vacc1x01234567 = _mm256_min_ps(vmax, vacc1x01234567);
    vacc2x01234567 = _mm256_min_ps(vmax, vacc2x01234567);
    vacc3x01234567 = _mm256_min_ps(vmax, vacc3x01234567);
    vacc4x01234567 = _mm256_min_ps(vmax, vacc4x01234567);
    vacc0x89ABCDEF = _mm256_min_ps(vmax, vacc0x89ABCDEF);
    vacc1x89ABCDEF = _mm256_min_ps(vmax, vacc1x89ABCDEF);
    vacc2x89ABCDEF = _mm256_min_ps(vmax, vacc2x89ABCDEF);
    vacc3x89ABCDEF = _mm256_min_ps(vmax, vacc3x89ABCDEF);
    vacc4x89ABCDEF = _mm256_min_ps(vmax, vacc4x89ABCDEF);

    if XNN_LIKELY(nc >= 16) {
      _mm256_storeu_ps(c0, vacc0x01234567);
      _mm256_storeu_ps(c0 + 8, vacc0x89ABCDEF);
      c0 = (float*) ((uintptr_t) c0 + cn_stride);
      _mm256_storeu_ps(c1, vacc1x01234567);
      _mm256_storeu_ps(c1 + 8, vacc1x89ABCDEF);
      c1 = (float*) ((uintptr_t) c1 + cn_stride);
      _mm256_storeu_ps(c2, vacc2x01234567);
      _mm256_storeu_ps(c2 + 8, vacc2x89ABCDEF);
      c2 = (float*) ((uintptr_t) c2 + cn_stride);
      _mm256_storeu_ps(c3, vacc3x01234567);
      _mm256_storeu_ps(c3 + 8, vacc3x89ABCDEF);
      c3 = (float*) ((uintptr_t) c3 + cn_stride);
      _mm256_storeu_ps(c4, vacc4x01234567);
      _mm256_storeu_ps(c4 + 8, vacc4x89ABCDEF);
      c4 = (float*) ((uintptr_t) c4 + cn_stride);

      a0 = (const float*) ((uintptr_t) a0 - kc);
      a1 = (const float*) ((uintptr_t) a1 - kc);
      a2 = (const float*) ((uintptr_t) a2 - kc);
      a3 = (const float*) ((uintptr_t) a3 - kc);
      a4 = (const float*) ((uintptr_t) a4 - kc);

      nc -= 16;
    } else {
      if (nc & 8) {
        _mm256_storeu_ps(c0, vacc0x01234567);
        _mm256_storeu_ps(c1, vacc1x01234567);
        _mm256_storeu_ps(c2, vacc2x01234567);
        _mm256_storeu_ps(c3, vacc3x01234567);
        _mm256_storeu_ps(c4, vacc4x01234567);

        vacc0x01234567 = vacc0x89ABCDEF;
        vacc1x01234567 = vacc1x89ABCDEF;
        vacc2x01234567 = vacc2x89ABCDEF;
        vacc3x01234567 = vacc3x89ABCDEF;
        vacc4x01234567 = vacc4x89ABCDEF;

        c0 += 8;
        c1 += 8;
        c2 += 8;
        c3 += 8;
        c4 += 8;
      }
      __m128 vacc0x0123 = _mm256_castps256_ps128(vacc0x01234567);
      __m128 vacc1x0123 = _mm256_castps256_ps128(vacc1x01234567);
      __m128 vacc2x0123 = _mm256_castps256_ps128(vacc2x01234567);
      __m128 vacc3x0123 = _mm256_castps256_ps128(vacc3x01234567);
      __m128 vacc4x0123 = _mm256_castps256_ps128(vacc4x01234567);
      if (nc & 4) {
        _mm_storeu_ps(c0, vacc0x0123);
        _mm_storeu_ps(c1, vacc1x0123);
        _mm_storeu_ps(c2, vacc2x0123);
        _mm_storeu_ps(c3, vacc3x0123);
        _mm_storeu_ps(c4, vacc4x0123);

        vacc0x0123 = _mm256_extractf128_ps(vacc0x01234567, 1);
        vacc1x0123 = _mm256_extractf128_ps(vacc1x01234567, 1);
        vacc2x0123 = _mm256_extractf128_ps(vacc2x01234567, 1);
        vacc3x0123 = _mm256_extractf128_ps(vacc3x01234567, 1);
        vacc4x0123 = _mm256_extractf128_ps(vacc4x01234567, 1);

        c0 += 4;
        c1 += 4;
        c2 += 4;
        c3 += 4;
        c4 += 4;
      }
      if (nc & 2) {
        _mm_storel_pi((__m64*) c0, vacc0x0123);
        _mm_storel_pi((__m64*) c1, vacc1x0123);
        _mm_storel_pi((__m64*) c2, vacc2x0123);
        _mm_storel_pi((__m64*) c3, vacc3x0123);
        _mm_storel_pi((__m64*) c4, vacc4x0123);

        vacc0x0123 = _mm_movehl_ps(vacc0x0123, vacc0x0123);
        vacc1x0123 = _mm_movehl_ps(vacc1x0123, vacc1x0123);
        vacc2x0123 = _mm_movehl_ps(vacc2x0123, vacc2x0123);
        vacc3x0123 = _mm_movehl_ps(vacc3x0123, vacc3x0123);
        vacc4x0123 = _mm_movehl_ps(vacc4x0123, vacc4x0123);

        c0 += 2;
        c1 += 2;
        c2 += 2;
        c3 += 2;
        c4 += 2;
      }
      if (nc & 1) {
        _mm_store_ss(c0, vacc0x0123);
        _mm_store_ss(c1, vacc1x0123);
        _mm_store_ss(c2, vacc2x0123);
        _mm_store_ss(c3, vacc3x0123);
        _mm_store_ss(c4, vacc4x0123);
      }

      nc = 0;
    }
  } while (nc != 0);
}

void xnn_f32_igemm_minmax_ukernel_1x16__fma3_broadcast(
    size_t mr,
    size_t nc,
    size_t kc,
    size_t ks,
    const float** restrict a,
    const float* restrict w,
    float* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    size_t a_offset,
    const float* zero,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(mr != 0);
  assert(mr <= 1);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(float) == 0);
  assert(ks != 0);
  assert(ks % (1 * sizeof(void*)) == 0);
  assert(a_offset % sizeof(float) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  float* c0 = c;

  const __m256 vmax = _mm256_set1_ps(params->avx.max);
  const __m256 vmin = _mm256_set1_ps(params->avx.min);
  XNN_FORCE_REALIZATION(vmin);
  XNN_FORCE_REALIZATION(vmax);

  do {
    __m256 vacc0x01234567 = _mm256_load_ps(w);
    __m256 vacc0x89ABCDEF = _mm256_load_ps(w + 8);
    w += 16;

    size_t p = ks;
    do {
      const float* restrict a0 = a[0];
      assert(a0 != NULL);
      if XNN_UNPREDICTABLE(a0 != zero) {
        a0 = (const float*) ((uintptr_t) a0 + a_offset);
      }
      a += 1;

      size_t k = kc;
      do {
        const __m256 vb01234567 = _mm256_load_ps(w);
        const __m256 vb89ABCDEF = _mm256_load_ps(w + 8);
        w += 16;

        const __m256 va0 = _mm256_broadcast_ss(a0);
        a0 += 1;

        vacc0x01234567 = _mm256_fmadd_ps(va0, vb01234567, vacc0x01234567);
        vacc0x89ABCDEF = _mm256_fmadd_ps(va0, vb89ABCDEF, vacc0x89ABCDEF);
        k -= sizeof(float);
      } while (k != 0);
      p -= 1 * sizeof(void*);
    } while (p != 0);

    vacc0x01234567 = _mm256_max_ps(vmin, vacc0x01234567);
    vacc0x89ABCDEF = _mm256_max_ps(vmin, vacc0x89ABCDEF);

    vacc0x01234567 = _mm256_min_ps(vmax, vacc0x01234567);
    vacc0x89ABCDEF = _mm256_min_ps(vmax, vacc0x89ABCDEF);

    if XNN_LIKELY(nc >= 16) {
      _mm256_storeu_ps(c0, vacc0x01234567);
      _mm256_storeu_ps(c0 + 8, vacc0x89ABCDEF);
      c0 = (float*) ((uintptr_t) c0 + cn_stride);

      a = (const float**restrict) ((uintptr_t) a - ks);
      nc -= 16;
    } else {
      if (nc & 8) {
        _mm256_storeu_ps(c0, vacc0x01234567);

        vacc0x01234567 = vacc0x89ABCDEF;

        c0 += 8;
      }
      __m128 vacc0x0123 = _mm256_castps256_ps128(vacc0x01234567);
      if (nc & 4) {
        _mm_storeu_ps(c0, vacc0x0123);

        vacc0x0123 = _mm256_extractf128_ps(vacc0x01234567, 1);

        c0 += 4;
      }
      if (nc & 2) {
        _mm_storel_pi((__m64*) c0, vacc0x0123);

        vacc0x0123 = _mm_movehl_ps(vacc0x0123, vacc0x0123);

        c0 += 2;
      }
      if (nc & 1) {
        _mm_store_ss(c0, vacc0x0123);
      }

      nc = 0;
    }
  } while (nc != 0);
}

void xnn_f32_igemm_minmax_ukernel_1x16s4__fma3_broadcast(
    size_t mr,
    size_t nc,
    size_t kc,
    size_t ks,
    const float** restrict a,
    const float* restrict w,
    float* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    size_t a_offset,
    const float* zero,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(mr != 0);
  assert(mr <= 1);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(float) == 0);
  assert(ks != 0);
  assert(ks % (1 * sizeof(void*)) == 0);
  assert(a_offset % sizeof(float) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  float* c0 = c;

  const __m256 vmax = _mm256_set1_ps(params->avx.max);
  const __m256 vmin = _mm256_set1_ps(params->avx.min);
  XNN_FORCE_REALIZATION(vmin);
  XNN_FORCE_REALIZATION(vmax);

  do {
    __m256 vacc0x01234567 = _mm256_load_ps(w);
    __m256 vacc0x89ABCDEF = _mm256_load_ps(w + 8);
    w += 16;

    size_t p = ks;
    do {
      const float* restrict a0 = a[0];
      assert(a0 != NULL);
      if XNN_UNPREDICTABLE(a0 != zero) {
        a0 = (const float*) ((uintptr_t) a0 + a_offset);
      }
      a += 1;

      size_t k = kc;
      while (k >= 4 * sizeof(float)) {
        __m256 va0 = _mm256_broadcast_ps((const __m128*) a0);
        a0 += 4;


        const __m256 vb01234567c0 = _mm256_load_ps(w + 0);
        const __m256 vb89ABCDEFc0 = _mm256_load_ps(w + 8);

        vacc0x01234567 = _mm256_fmadd_ps(va0, vb01234567c0, vacc0x01234567);
        vacc0x89ABCDEF = _mm256_fmadd_ps(va0, vb89ABCDEFc0, vacc0x89ABCDEF);

        va0 = _mm256_permute_ps(va0, _MM_SHUFFLE(0, 3, 2, 1));

        const __m256 vb01234567c1 = _mm256_load_ps(w + 16);
        const __m256 vb89ABCDEFc1 = _mm256_load_ps(w + 24);

        vacc0x01234567 = _mm256_fmadd_ps(va0, vb01234567c1, vacc0x01234567);
        vacc0x89ABCDEF = _mm256_fmadd_ps(va0, vb89ABCDEFc1, vacc0x89ABCDEF);

        va0 = _mm256_permute_ps(va0, _MM_SHUFFLE(0, 3, 2, 1));

        const __m256 vb01234567c2 = _mm256_load_ps(w + 32);
        const __m256 vb89ABCDEFc2 = _mm256_load_ps(w + 40);

        vacc0x01234567 = _mm256_fmadd_ps(va0, vb01234567c2, vacc0x01234567);
        vacc0x89ABCDEF = _mm256_fmadd_ps(va0, vb89ABCDEFc2, vacc0x89ABCDEF);

        va0 = _mm256_permute_ps(va0, _MM_SHUFFLE(0, 3, 2, 1));

        const __m256 vb01234567c3 = _mm256_load_ps(w + 48);
        const __m256 vb89ABCDEFc3 = _mm256_load_ps(w + 56);

        vacc0x01234567 = _mm256_fmadd_ps(va0, vb01234567c3, vacc0x01234567);
        vacc0x89ABCDEF = _mm256_fmadd_ps(va0, vb89ABCDEFc3, vacc0x89ABCDEF);


        w += 64;
        k -= 4 * sizeof(float);
      }
      if XNN_UNLIKELY(k != 0) {
        __m256 va0 = _mm256_broadcast_ps((const __m128*) a0);
        a0 = (const float*) ((uintptr_t) a0 + k);

        const __m256 vzero = _mm256_setzero_ps();

        const __m256 vb01234567c0 = _mm256_load_ps(w + 0);
        const __m256 vb89ABCDEFc0 = _mm256_load_ps(w + 8);

        vacc0x01234567 = _mm256_fmadd_ps(_mm256_and_ps(va0, _mm256_cmp_ps(vb01234567c0, vzero, _CMP_NEQ_OQ)), vb01234567c0, vacc0x01234567);
        vacc0x89ABCDEF = _mm256_fmadd_ps(_mm256_and_ps(va0, _mm256_cmp_ps(vb89ABCDEFc0, vzero, _CMP_NEQ_OQ)), vb89ABCDEFc0, vacc0x89ABCDEF);

        va0 = _mm256_permute_ps(va0, _MM_SHUFFLE(0, 3, 2, 1));

        const __m256 vb01234567c1 = _mm256_load_ps(w + 16);
        const __m256 vb89ABCDEFc1 = _mm256_load_ps(w + 24);

        vacc0x01234567 = _mm256_fmadd_ps(_mm256_and_ps(va0, _mm256_cmp_ps(vb01234567c1, vzero, _CMP_NEQ_OQ)), vb01234567c1, vacc0x01234567);
        vacc0x89ABCDEF = _mm256_fmadd_ps(_mm256_and_ps(va0, _mm256_cmp_ps(vb89ABCDEFc1, vzero, _CMP_NEQ_OQ)), vb89ABCDEFc1, vacc0x89ABCDEF);

        va0 = _mm256_permute_ps(va0, _MM_SHUFFLE(0, 3, 2, 1));

        const __m256 vb01234567c2 = _mm256_load_ps(w + 32);
        const __m256 vb89ABCDEFc2 = _mm256_load_ps(w + 40);

        vacc0x01234567 = _mm256_fmadd_ps(_mm256_and_ps(va0, _mm256_cmp_ps(vb01234567c2, vzero, _CMP_NEQ_OQ)), vb01234567c2, vacc0x01234567);
        vacc0x89ABCDEF = _mm256_fmadd_ps(_mm256_and_ps(va0, _mm256_cmp_ps(vb89ABCDEFc2, vzero, _CMP_NEQ_OQ)), vb89ABCDEFc2, vacc0x89ABCDEF);

        va0 = _mm256_permute_ps(va0, _MM_SHUFFLE(0, 3, 2, 1));

        const __m256 vb01234567c3 = _mm256_load_ps(w + 48);
        const __m256 vb89ABCDEFc3 = _mm256_load_ps(w + 56);

        vacc0x01234567 = _mm256_fmadd_ps(_mm256_and_ps(va0, _mm256_cmp_ps(vb01234567c3, vzero, _CMP_NEQ_OQ)), vb01234567c3, vacc0x01234567);
        vacc0x89ABCDEF = _mm256_fmadd_ps(_mm256_and_ps(va0, _mm256_cmp_ps(vb89ABCDEFc3, vzero, _CMP_NEQ_OQ)), vb89ABCDEFc3, vacc0x89ABCDEF);


        w += 64;
      }
      p -= 1 * sizeof(void*);
    } while (p != 0);

    vacc0x01234567 = _mm256_max_ps(vmin, vacc0x01234567);
    vacc0x89ABCDEF = _mm256_max_ps(vmin, vacc0x89ABCDEF);

    vacc0x01234567 = _mm256_min_ps(vmax, vacc0x01234567);
    vacc0x89ABCDEF = _mm256_min_ps(vmax, vacc0x89ABCDEF);

    if XNN_LIKELY(nc >= 16) {
      _mm256_storeu_ps(c0, vacc0x01234567);
      _mm256_storeu_ps(c0 + 8, vacc0x89ABCDEF);
      c0 = (float*) ((uintptr_t) c0 + cn_stride);

      a = (const float**restrict) ((uintptr_t) a - ks);
      nc -= 16;
    } else {
      if (nc & 8) {
        _mm256_storeu_ps(c0, vacc0x01234567);

        vacc0x01234567 = vacc0x89ABCDEF;

        c0 += 8;
      }
      __m128 vacc0x0123 = _mm256_castps256_ps128(vacc0x01234567);
      if (nc & 4) {
        _mm_storeu_ps(c0, vacc0x0123);

        vacc0x0123 = _mm256_extractf128_ps(vacc0x01234567, 1);

        c0 += 4;
      }
      if (nc & 2) {
        _mm_storel_pi((__m64*) c0, vacc0x0123);

        vacc0x0123 = _mm_movehl_ps(vacc0x0123, vacc0x0123);

        c0 += 2;
      }
      if (nc & 1) {
        _mm_store_ss(c0, vacc0x0123);
      }

      nc = 0;
    }
  } while (nc != 0);
}

void xnn_f32_igemm_minmax_ukernel_4x16s4__fma3_broadcast(
    size_t mr,
    size_t nc,
    size_t kc,
    size_t ks,
    const float** restrict a,
    const float* restrict w,
    float* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    size_t a_offset,
    const float* zero,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(mr != 0);
  assert(mr <= 4);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(float) == 0);
  assert(ks != 0);
  assert(ks % (4 * sizeof(void*)) == 0);
  assert(a_offset % sizeof(float) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  float* c0 = c;
  float* c1 = (float*) ((uintptr_t) c0 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 2) {
    c1 = c0;
  }
  float* c2 = (float*) ((uintptr_t) c1 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 2) {
    c2 = c1;
  }
  float* c3 = (float*) ((uintptr_t) c2 + cm_stride);
  if XNN_UNPREDICTABLE(mr != 4) {
    c3 = c2;
  }

  const __m256 vmax = _mm256_set1_ps(params->avx.max);
  const __m256 vmin = _mm256_set1_ps(params->avx.min);
  XNN_FORCE_REALIZATION(vmin);
  XNN_FORCE_REALIZATION(vmax);

  do {
    __m256 vacc0x01234567 = _mm256_load_ps(w);
    __m256 vacc0x89ABCDEF = _mm256_load_ps(w + 8);
    __m256 vacc1x01234567 = vacc0x01234567;
    __m256 vacc1x89ABCDEF = vacc0x89ABCDEF;
    __m256 vacc2x01234567 = vacc0x01234567;
    __m256 vacc2x89ABCDEF = vacc0x89ABCDEF;
    __m256 vacc3x01234567 = vacc0x01234567;
    __m256 vacc3x89ABCDEF = vacc0x89ABCDEF;
    w += 16;

    size_t p = ks;
    do {
      const float* restrict a0 = a[0];
      assert(a0 != NULL);
      if XNN_UNPREDICTABLE(a0 != zero) {
        a0 = (const float*) ((uintptr_t) a0 + a_offset);
      }
      const float* restrict a1 = a[1];
      assert(a1 != NULL);
      if XNN_UNPREDICTABLE(a1 != zero) {
        a1 = (const float*) ((uintptr_t) a1 + a_offset);
      }
      const float* restrict a2 = a[2];
      assert(a2 != NULL);
      if XNN_UNPREDICTABLE(a2 != zero) {
        a2 = (const float*) ((uintptr_t) a2 + a_offset);
      }
      const float* restrict a3 = a[3];
      assert(a3 != NULL);
      if XNN_UNPREDICTABLE(a3 != zero) {
        a3 = (const float*) ((uintptr_t) a3 + a_offset);
      }
      a += 4;

      size_t k = kc;
      while (k >= 4 * sizeof(float)) {
        __m256 va0 = _mm256_broadcast_ps((const __m128*) a0);
        a0 += 4;
        __m256 va1 = _mm256_broadcast_ps((const __m128*) a1);
        a1 += 4;
        __m256 va2 = _mm256_broadcast_ps((const __m128*) a2);
        a2 += 4;
        __m256 va3 = _mm256_broadcast_ps((const __m128*) a3);
        a3 += 4;


        const __m256 vb01234567c0 = _mm256_load_ps(w + 0);
        const __m256 vb89ABCDEFc0 = _mm256_load_ps(w + 8);

        vacc0x01234567 = _mm256_fmadd_ps(va0, vb01234567c0, vacc0x01234567);
        vacc1x01234567 = _mm256_fmadd_ps(va1, vb01234567c0, vacc1x01234567);
        vacc2x01234567 = _mm256_fmadd_ps(va2, vb01234567c0, vacc2x01234567);
        vacc3x01234567 = _mm256_fmadd_ps(va3, vb01234567c0, vacc3x01234567);
        vacc0x89ABCDEF = _mm256_fmadd_ps(va0, vb89ABCDEFc0, vacc0x89ABCDEF);
        vacc1x89ABCDEF = _mm256_fmadd_ps(va1, vb89ABCDEFc0, vacc1x89ABCDEF);
        vacc2x89ABCDEF = _mm256_fmadd_ps(va2, vb89ABCDEFc0, vacc2x89ABCDEF);
        vacc3x89ABCDEF = _mm256_fmadd_ps(va3, vb89ABCDEFc0, vacc3x89ABCDEF);

        va0 = _mm256_permute_ps(va0, _MM_SHUFFLE(0, 3, 2, 1));
        va1 = _mm256_permute_ps(va1, _MM_SHUFFLE(0, 3, 2, 1));
        va2 = _mm256_permute_ps(va2, _MM_SHUFFLE(0, 3, 2, 1));
        va3 = _mm256_permute_ps(va3, _MM_SHUFFLE(0, 3, 2, 1));

        const __m256 vb01234567c1 = _mm256_load_ps(w + 16);
        const __m256 vb89ABCDEFc1 = _mm256_load_ps(w + 24);

        vacc0x01234567 = _mm256_fmadd_ps(va0, vb01234567c1, vacc0x01234567);
        vacc1x01234567 = _mm256_fmadd_ps(va1, vb01234567c1, vacc1x01234567);
        vacc2x01234567 = _mm256_fmadd_ps(va2, vb01234567c1, vacc2x01234567);
        vacc3x01234567 = _mm256_fmadd_ps(va3, vb01234567c1, vacc3x01234567);
        vacc0x89ABCDEF = _mm256_fmadd_ps(va0, vb89ABCDEFc1, vacc0x89ABCDEF);
        vacc1x89ABCDEF = _mm256_fmadd_ps(va1, vb89ABCDEFc1, vacc1x89ABCDEF);
        vacc2x89ABCDEF = _mm256_fmadd_ps(va2, vb89ABCDEFc1, vacc2x89ABCDEF);
        vacc3x89ABCDEF = _mm256_fmadd_ps(va3, vb89ABCDEFc1, vacc3x89ABCDEF);

        va0 = _mm256_permute_ps(va0, _MM_SHUFFLE(0, 3, 2, 1));
        va1 = _mm256_permute_ps(va1, _MM_SHUFFLE(0, 3, 2, 1));
        va2 = _mm256_permute_ps(va2, _MM_SHUFFLE(0, 3, 2, 1));
        va3 = _mm256_permute_ps(va3, _MM_SHUFFLE(0, 3, 2, 1));

        const __m256 vb01234567c2 = _mm256_load_ps(w + 32);
        const __m256 vb89ABCDEFc2 = _mm256_load_ps(w + 40);

        vacc0x01234567 = _mm256_fmadd_ps(va0, vb01234567c2, vacc0x01234567);
        vacc1x01234567 = _mm256_fmadd_ps(va1, vb01234567c2, vacc1x01234567);
        vacc2x01234567 = _mm256_fmadd_ps(va2, vb01234567c2, vacc2x01234567);
        vacc3x01234567 = _mm256_fmadd_ps(va3, vb01234567c2, vacc3x01234567);
        vacc0x89ABCDEF = _mm256_fmadd_ps(va0, vb89ABCDEFc2, vacc0x89ABCDEF);
        vacc1x89ABCDEF = _mm256_fmadd_ps(va1, vb89ABCDEFc2, vacc1x89ABCDEF);
        vacc2x89ABCDEF = _mm256_fmadd_ps(va2, vb89ABCDEFc2, vacc2x89ABCDEF);
        vacc3x89ABCDEF = _mm256_fmadd_ps(va3, vb89ABCDEFc2, vacc3x89ABCDEF);

        va0 = _mm256_permute_ps(va0, _MM_SHUFFLE(0, 3, 2, 1));
        va1 = _mm256_permute_ps(va1, _MM_SHUFFLE(0, 3, 2, 1));
        va2 = _mm256_permute_ps(va2, _MM_SHUFFLE(0, 3, 2, 1));
        va3 = _mm256_permute_ps(va3, _MM_SHUFFLE(0, 3, 2, 1));

        const __m256 vb01234567c3 = _mm256_load_ps(w + 48);
        const __m256 vb89ABCDEFc3 = _mm256_load_ps(w + 56);

        vacc0x01234567 = _mm256_fmadd_ps(va0, vb01234567c3, vacc0x01234567);
        vacc1x01234567 = _mm256_fmadd_ps(va1, vb01234567c3, vacc1x01234567);
        vacc2x01234567 = _mm256_fmadd_ps(va2, vb01234567c3, vacc2x01234567);
        vacc3x01234567 = _mm256_fmadd_ps(va3, vb01234567c3, vacc3x01234567);
        vacc0x89ABCDEF = _mm256_fmadd_ps(va0, vb89ABCDEFc3, vacc0x89ABCDEF);
        vacc1x89ABCDEF = _mm256_fmadd_ps(va1, vb89ABCDEFc3, vacc1x89ABCDEF);
        vacc2x89ABCDEF = _mm256_fmadd_ps(va2, vb89ABCDEFc3, vacc2x89ABCDEF);
        vacc3x89ABCDEF = _mm256_fmadd_ps(va3, vb89ABCDEFc3, vacc3x89ABCDEF);


        w += 64;
        k -= 4 * sizeof(float);
      }
      if XNN_UNLIKELY(k != 0) {
        __m256 va0 = _mm256_broadcast_ps((const __m128*) a0);
        a0 = (const float*) ((uintptr_t) a0 + k);
        __m256 va1 = _mm256_broadcast_ps((const __m128*) a1);
        a1 = (const float*) ((uintptr_t) a1 + k);
        __m256 va2 = _mm256_broadcast_ps((const __m128*) a2);
        a2 = (const float*) ((uintptr_t) a2 + k);
        __m256 va3 = _mm256_broadcast_ps((const __m128*) a3);
        a3 = (const float*) ((uintptr_t) a3 + k);

        const __m256 vzero = _mm256_setzero_ps();

        const __m256 vb01234567c0 = _mm256_load_ps(w + 0);
        const __m256 vb89ABCDEFc0 = _mm256_load_ps(w + 8);

        vacc0x01234567 = _mm256_fmadd_ps(_mm256_and_ps(va0, _mm256_cmp_ps(vb01234567c0, vzero, _CMP_NEQ_OQ)), vb01234567c0, vacc0x01234567);
        vacc1x01234567 = _mm256_fmadd_ps(_mm256_and_ps(va1, _mm256_cmp_ps(vb01234567c0, vzero, _CMP_NEQ_OQ)), vb01234567c0, vacc1x01234567);
        vacc2x01234567 = _mm256_fmadd_ps(_mm256_and_ps(va2, _mm256_cmp_ps(vb01234567c0, vzero, _CMP_NEQ_OQ)), vb01234567c0, vacc2x01234567);
        vacc3x01234567 = _mm256_fmadd_ps(_mm256_and_ps(va3, _mm256_cmp_ps(vb01234567c0, vzero, _CMP_NEQ_OQ)), vb01234567c0, vacc3x01234567);
        vacc0x89ABCDEF = _mm256_fmadd_ps(_mm256_and_ps(va0, _mm256_cmp_ps(vb89ABCDEFc0, vzero, _CMP_NEQ_OQ)), vb89ABCDEFc0, vacc0x89ABCDEF);
        vacc1x89ABCDEF = _mm256_fmadd_ps(_mm256_and_ps(va1, _mm256_cmp_ps(vb89ABCDEFc0, vzero, _CMP_NEQ_OQ)), vb89ABCDEFc0, vacc1x89ABCDEF);
        vacc2x89ABCDEF = _mm256_fmadd_ps(_mm256_and_ps(va2, _mm256_cmp_ps(vb89ABCDEFc0, vzero, _CMP_NEQ_OQ)), vb89ABCDEFc0, vacc2x89ABCDEF);
        vacc3x89ABCDEF = _mm256_fmadd_ps(_mm256_and_ps(va3, _mm256_cmp_ps(vb89ABCDEFc0, vzero, _CMP_NEQ_OQ)), vb89ABCDEFc0, vacc3x89ABCDEF);

        va0 = _mm256_permute_ps(va0, _MM_SHUFFLE(0, 3, 2, 1));
        va1 = _mm256_permute_ps(va1, _MM_SHUFFLE(0, 3, 2, 1));
        va2 = _mm256_permute_ps(va2, _MM_SHUFFLE(0, 3, 2, 1));
        va3 = _mm256_permute_ps(va3, _MM_SHUFFLE(0, 3, 2, 1));

        const __m256 vb01234567c1 = _mm256_load_ps(w + 16);
        const __m256 vb89ABCDEFc1 = _mm256_load_ps(w + 24);

        vacc0x01234567 = _mm256_fmadd_ps(_mm256_and_ps(va0, _mm256_cmp_ps(vb01234567c1, vzero, _CMP_NEQ_OQ)), vb01234567c1, vacc0x01234567);
        vacc1x01234567 = _mm256_fmadd_ps(_mm256_and_ps(va1, _mm256_cmp_ps(vb01234567c1, vzero, _CMP_NEQ_OQ)), vb01234567c1, vacc1x01234567);
        vacc2x01234567 = _mm256_fmadd_ps(_mm256_and_ps(va2, _mm256_cmp_ps(vb01234567c1, vzero, _CMP_NEQ_OQ)), vb01234567c1, vacc2x01234567);
        vacc3x01234567 = _mm256_fmadd_ps(_mm256_and_ps(va3, _mm256_cmp_ps(vb01234567c1, vzero, _CMP_NEQ_OQ)), vb01234567c1, vacc3x01234567);
        vacc0x89ABCDEF = _mm256_fmadd_ps(_mm256_and_ps(va0, _mm256_cmp_ps(vb89ABCDEFc1, vzero, _CMP_NEQ_OQ)), vb89ABCDEFc1, vacc0x89ABCDEF);
        vacc1x89ABCDEF = _mm256_fmadd_ps(_mm256_and_ps(va1, _mm256_cmp_ps(vb89ABCDEFc1, vzero, _CMP_NEQ_OQ)), vb89ABCDEFc1, vacc1x89ABCDEF);
        vacc2x89ABCDEF = _mm256_fmadd_ps(_mm256_and_ps(va2, _mm256_cmp_ps(vb89ABCDEFc1, vzero, _CMP_NEQ_OQ)), vb89ABCDEFc1, vacc2x89ABCDEF);
        vacc3x89ABCDEF = _mm256_fmadd_ps(_mm256_and_ps(va3, _mm256_cmp_ps(vb89ABCDEFc1, vzero, _CMP_NEQ_OQ)), vb89ABCDEFc1, vacc3x89ABCDEF);

        va0 = _mm256_permute_ps(va0, _MM_SHUFFLE(0, 3, 2, 1));
        va1 = _mm256_permute_ps(va1, _MM_SHUFFLE(0, 3, 2, 1));
        va2 = _mm256_permute_ps(va2, _MM_SHUFFLE(0, 3, 2, 1));
        va3 = _mm256_permute_ps(va3, _MM_SHUFFLE(0, 3, 2, 1));

        const __m256 vb01234567c2 = _mm256_load_ps(w + 32);
        const __m256 vb89ABCDEFc2 = _mm256_load_ps(w + 40);

        vacc0x01234567 = _mm256_fmadd_ps(_mm256_and_ps(va0, _mm256_cmp_ps(vb01234567c2, vzero, _CMP_NEQ_OQ)), vb01234567c2, vacc0x01234567);
        vacc1x01234567 = _mm256_fmadd_ps(_mm256_and_ps(va1, _mm256_cmp_ps(vb01234567c2, vzero, _CMP_NEQ_OQ)), vb01234567c2, vacc1x01234567);
        vacc2x01234567 = _mm256_fmadd_ps(_mm256_and_ps(va2, _mm256_cmp_ps(vb01234567c2, vzero, _CMP_NEQ_OQ)), vb01234567c2, vacc2x01234567);
        vacc3x01234567 = _mm256_fmadd_ps(_mm256_and_ps(va3, _mm256_cmp_ps(vb01234567c2, vzero, _CMP_NEQ_OQ)), vb01234567c2, vacc3x01234567);
        vacc0x89ABCDEF = _mm256_fmadd_ps(_mm256_and_ps(va0, _mm256_cmp_ps(vb89ABCDEFc2, vzero, _CMP_NEQ_OQ)), vb89ABCDEFc2, vacc0x89ABCDEF);
        vacc1x89ABCDEF = _mm256_fmadd_ps(_mm256_and_ps(va1, _mm256_cmp_ps(vb89ABCDEFc2, vzero, _CMP_NEQ_OQ)), vb89ABCDEFc2, vacc1x89ABCDEF);
        vacc2x89ABCDEF = _mm256_fmadd_ps(_mm256_and_ps(va2, _mm256_cmp_ps(vb89ABCDEFc2, vzero, _CMP_NEQ_OQ)), vb89ABCDEFc2, vacc2x89ABCDEF);
        vacc3x89ABCDEF = _mm256_fmadd_ps(_mm256_and_ps(va3, _mm256_cmp_ps(vb89ABCDEFc2, vzero, _CMP_NEQ_OQ)), vb89ABCDEFc2, vacc3x89ABCDEF);

        va0 = _mm256_permute_ps(va0, _MM_SHUFFLE(0, 3, 2, 1));
        va1 = _mm256_permute_ps(va1, _MM_SHUFFLE(0, 3, 2, 1));
        va2 = _mm256_permute_ps(va2, _MM_SHUFFLE(0, 3, 2, 1));
        va3 = _mm256_permute_ps(va3, _MM_SHUFFLE(0, 3, 2, 1));

        const __m256 vb01234567c3 = _mm256_load_ps(w + 48);
        const __m256 vb89ABCDEFc3 = _mm256_load_ps(w + 56);

        vacc0x01234567 = _mm256_fmadd_ps(_mm256_and_ps(va0, _mm256_cmp_ps(vb01234567c3, vzero, _CMP_NEQ_OQ)), vb01234567c3, vacc0x01234567);
        vacc1x01234567 = _mm256_fmadd_ps(_mm256_and_ps(va1, _mm256_cmp_ps(vb01234567c3, vzero, _CMP_NEQ_OQ)), vb01234567c3, vacc1x01234567);
        vacc2x01234567 = _mm256_fmadd_ps(_mm256_and_ps(va2, _mm256_cmp_ps(vb01234567c3, vzero, _CMP_NEQ_OQ)), vb01234567c3, vacc2x01234567);
        vacc3x01234567 = _mm256_fmadd_ps(_mm256_and_ps(va3, _mm256_cmp_ps(vb01234567c3, vzero, _CMP_NEQ_OQ)), vb01234567c3, vacc3x01234567);
        vacc0x89ABCDEF = _mm256_fmadd_ps(_mm256_and_ps(va0, _mm256_cmp_ps(vb89ABCDEFc3, vzero, _CMP_NEQ_OQ)), vb89ABCDEFc3, vacc0x89ABCDEF);
        vacc1x89ABCDEF = _mm256_fmadd_ps(_mm256_and_ps(va1, _mm256_cmp_ps(vb89ABCDEFc3, vzero, _CMP_NEQ_OQ)), vb89ABCDEFc3, vacc1x89ABCDEF);
        vacc2x89ABCDEF = _mm256_fmadd_ps(_mm256_and_ps(va2, _mm256_cmp_ps(vb89ABCDEFc3, vzero, _CMP_NEQ_OQ)), vb89ABCDEFc3, vacc2x89ABCDEF);
        vacc3x89ABCDEF = _mm256_fmadd_ps(_mm256_and_ps(va3, _mm256_cmp_ps(vb89ABCDEFc3, vzero, _CMP_NEQ_OQ)), vb89ABCDEFc3, vacc3x89ABCDEF);


        w += 64;
      }
      p -= 4 * sizeof(void*);
    } while (p != 0);

    vacc0x01234567 = _mm256_max_ps(vmin, vacc0x01234567);
    vacc1x01234567 = _mm256_max_ps(vmin, vacc1x01234567);
    vacc2x01234567 = _mm256_max_ps(vmin, vacc2x01234567);
    vacc3x01234567 = _mm256_max_ps(vmin, vacc3x01234567);
    vacc0x89ABCDEF = _mm256_max_ps(vmin, vacc0x89ABCDEF);
    vacc1x89ABCDEF = _mm256_max_ps(vmin, vacc1x89ABCDEF);
    vacc2x89ABCDEF = _mm256_max_ps(vmin, vacc2x89ABCDEF);
    vacc3x89ABCDEF = _mm256_max_ps(vmin, vacc3x89ABCDEF);

    vacc0x01234567 = _mm256_min_ps(vmax, vacc0x01234567);
    vacc1x01234567 = _mm256_min_ps(vmax, vacc1x01234567);
    vacc2x01234567 = _mm256_min_ps(vmax, vacc2x01234567);
    vacc3x01234567 = _mm256_min_ps(vmax, vacc3x01234567);
    vacc0x89ABCDEF = _mm256_min_ps(vmax, vacc0x89ABCDEF);
    vacc1x89ABCDEF = _mm256_min_ps(vmax, vacc1x89ABCDEF);
    vacc2x89ABCDEF = _mm256_min_ps(vmax, vacc2x89ABCDEF);
    vacc3x89ABCDEF = _mm256_min_ps(vmax, vacc3x89ABCDEF);

    if XNN_LIKELY(nc >= 16) {
      _mm256_storeu_ps(c3, vacc3x01234567);
      _mm256_storeu_ps(c3 + 8, vacc3x89ABCDEF);
      c3 = (float*) ((uintptr_t) c3 + cn_stride);
      _mm256_storeu_ps(c2, vacc2x01234567);
      _mm256_storeu_ps(c2 + 8, vacc2x89ABCDEF);
      c2 = (float*) ((uintptr_t) c2 + cn_stride);
      _mm256_storeu_ps(c1, vacc1x01234567);
      _mm256_storeu_ps(c1 + 8, vacc1x89ABCDEF);
      c1 = (float*) ((uintptr_t) c1 + cn_stride);
      _mm256_storeu_ps(c0, vacc0x01234567);
      _mm256_storeu_ps(c0 + 8, vacc0x89ABCDEF);
      c0 = (float*) ((uintptr_t) c0 + cn_stride);

      a = (const float**restrict) ((uintptr_t) a - ks);
      nc -= 16;
    } else {
      if (nc & 8) {
        _mm256_storeu_ps(c3, vacc3x01234567);
        _mm256_storeu_ps(c2, vacc2x01234567);
        _mm256_storeu_ps(c1, vacc1x01234567);
        _mm256_storeu_ps(c0, vacc0x01234567);

        vacc3x01234567 = vacc3x89ABCDEF;
        vacc2x01234567 = vacc2x89ABCDEF;
        vacc1x01234567 = vacc1x89ABCDEF;
        vacc0x01234567 = vacc0x89ABCDEF;

        c3 += 8;
        c2 += 8;
        c1 += 8;
        c0 += 8;
      }
      __m128 vacc3x0123 = _mm256_castps256_ps128(vacc3x01234567);
      __m128 vacc2x0123 = _mm256_castps256_ps128(vacc2x01234567);
      __m128 vacc1x0123 = _mm256_castps256_ps128(vacc1x01234567);
      __m128 vacc0x0123 = _mm256_castps256_ps128(vacc0x01234567);
      if (nc & 4) {
        _mm_storeu_ps(c3, vacc3x0123);
        _mm_storeu_ps(c2, vacc2x0123);
        _mm_storeu_ps(c1, vacc1x0123);
        _mm_storeu_ps(c0, vacc0x0123);

        vacc3x0123 = _mm256_extractf128_ps(vacc3x01234567, 1);
        vacc2x0123 = _mm256_extractf128_ps(vacc2x01234567, 1);
        vacc1x0123 = _mm256_extractf128_ps(vacc1x01234567, 1);
        vacc0x0123 = _mm256_extractf128_ps(vacc0x01234567, 1);

        c3 += 4;
        c2 += 4;
        c1 += 4;
        c0 += 4;
      }
      if (nc & 2) {
        _mm_storel_pi((__m64*) c3, vacc3x0123);
        _mm_storel_pi((__m64*) c2, vacc2x0123);
        _mm_storel_pi((__m64*) c1, vacc1x0123);
        _mm_storel_pi((__m64*) c0, vacc0x0123);

        vacc3x0123 = _mm_movehl_ps(vacc3x0123, vacc3x0123);
        vacc2x0123 = _mm_movehl_ps(vacc2x0123, vacc2x0123);
        vacc1x0123 = _mm_movehl_ps(vacc1x0123, vacc1x0123);
        vacc0x0123 = _mm_movehl_ps(vacc0x0123, vacc0x0123);

        c3 += 2;
        c2 += 2;
        c1 += 2;
        c0 += 2;
      }
      if (nc & 1) {
        _mm_store_ss(c3, vacc3x0123);
        _mm_store_ss(c2, vacc2x0123);
        _mm_store_ss(c1, vacc1x0123);
        _mm_store_ss(c0, vacc0x0123);
      }

      nc = 0;
    }
  } while (nc != 0);
}

void xnn_f32_igemm_minmax_ukernel_5x16__fma3_broadcast_prfm(
    size_t mr,
    size_t nc,
    size_t kc,
    size_t ks,
    const float** restrict a,
    const float* restrict w,
    float* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    size_t a_offset,
    const float* zero,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(mr != 0);
  assert(mr <= 5);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(float) == 0);
  assert(ks != 0);
  assert(ks % (5 * sizeof(void*)) == 0);
  assert(a_offset % sizeof(float) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  float* c0 = c;
  float* c1 = (float*) ((uintptr_t) c0 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 2) {
    c1 = c0;
  }
  float* c2 = (float*) ((uintptr_t) c1 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 2) {
    c2 = c1;
  }
  float* c3 = (float*) ((uintptr_t) c2 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 4) {
    c3 = c2;
  }
  float* c4 = (float*) ((uintptr_t) c3 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 4) {
    c4 = c3;
  }

  const __m256 vmax = _mm256_set1_ps(params->avx.max);
  const __m256 vmin = _mm256_set1_ps(params->avx.min);
  XNN_FORCE_REALIZATION(vmin);
  XNN_FORCE_REALIZATION(vmax);

  do {
    __m256 vacc0x01234567 = _mm256_load_ps(w);
    __m256 vacc0x89ABCDEF = _mm256_load_ps(w + 8);
    __m256 vacc1x01234567 = vacc0x01234567;
    __m256 vacc1x89ABCDEF = vacc0x89ABCDEF;
    __m256 vacc2x01234567 = vacc0x01234567;
    __m256 vacc2x89ABCDEF = vacc0x89ABCDEF;
    __m256 vacc3x01234567 = vacc0x01234567;
    __m256 vacc3x89ABCDEF = vacc0x89ABCDEF;
    __m256 vacc4x01234567 = vacc0x01234567;
    __m256 vacc4x89ABCDEF = vacc0x89ABCDEF;
    w += 16;

    size_t p = ks;
    do {
      const float* restrict a0 = a[0];
      assert(a0 != NULL);
      if XNN_UNPREDICTABLE(a0 != zero) {
        a0 = (const float*) ((uintptr_t) a0 + a_offset);
      }
      const float* restrict a1 = a[1];
      assert(a1 != NULL);
      if XNN_UNPREDICTABLE(a1 != zero) {
        a1 = (const float*) ((uintptr_t) a1 + a_offset);
      }
      const float* restrict a2 = a[2];
      assert(a2 != NULL);
      if XNN_UNPREDICTABLE(a2 != zero) {
        a2 = (const float*) ((uintptr_t) a2 + a_offset);
      }
      const float* restrict a3 = a[3];
      assert(a3 != NULL);
      if XNN_UNPREDICTABLE(a3 != zero) {
        a3 = (const float*) ((uintptr_t) a3 + a_offset);
      }
      const float* restrict a4 = a[4];
      assert(a4 != NULL);
      if XNN_UNPREDICTABLE(a4 != zero) {
        a4 = (const float*) ((uintptr_t) a4 + a_offset);
      }
      a += 5;

      size_t k = kc;
      do {
        const __m256 vb01234567 = _mm256_load_ps(w);
        const __m256 vb89ABCDEF = _mm256_load_ps(w + 8);
        w += 16;
        xnn_prefetch_to_l1(w + 224);

        const __m256 va0 = _mm256_broadcast_ss(a0);
        a0 += 1;
        const __m256 va1 = _mm256_broadcast_ss(a1);
        a1 += 1;
        const __m256 va2 = _mm256_broadcast_ss(a2);
        a2 += 1;
        const __m256 va3 = _mm256_broadcast_ss(a3);
        a3 += 1;
        const __m256 va4 = _mm256_broadcast_ss(a4);
        a4 += 1;

        vacc0x01234567 = _mm256_fmadd_ps(va0, vb01234567, vacc0x01234567);
        vacc0x89ABCDEF = _mm256_fmadd_ps(va0, vb89ABCDEF, vacc0x89ABCDEF);
        vacc1x01234567 = _mm256_fmadd_ps(va1, vb01234567, vacc1x01234567);
        vacc1x89ABCDEF = _mm256_fmadd_ps(va1, vb89ABCDEF, vacc1x89ABCDEF);
        vacc2x01234567 = _mm256_fmadd_ps(va2, vb01234567, vacc2x01234567);
        vacc2x89ABCDEF = _mm256_fmadd_ps(va2, vb89ABCDEF, vacc2x89ABCDEF);
        vacc3x01234567 = _mm256_fmadd_ps(va3, vb01234567, vacc3x01234567);
        vacc3x89ABCDEF = _mm256_fmadd_ps(va3, vb89ABCDEF, vacc3x89ABCDEF);
        vacc4x01234567 = _mm256_fmadd_ps(va4, vb01234567, vacc4x01234567);
        vacc4x89ABCDEF = _mm256_fmadd_ps(va4, vb89ABCDEF, vacc4x89ABCDEF);
        k -= sizeof(float);
      } while (k != 0);
      p -= 5 * sizeof(void*);
    } while (p != 0);

    vacc0x01234567 = _mm256_max_ps(vmin, vacc0x01234567);
    vacc1x01234567 = _mm256_max_ps(vmin, vacc1x01234567);
    vacc2x01234567 = _mm256_max_ps(vmin, vacc2x01234567);
    vacc3x01234567 = _mm256_max_ps(vmin, vacc3x01234567);
    vacc4x01234567 = _mm256_max_ps(vmin, vacc4x01234567);
    vacc0x89ABCDEF = _mm256_max_ps(vmin, vacc0x89ABCDEF);
    vacc1x89ABCDEF = _mm256_max_ps(vmin, vacc1x89ABCDEF);
    vacc2x89ABCDEF = _mm256_max_ps(vmin, vacc2x89ABCDEF);
    vacc3x89ABCDEF = _mm256_max_ps(vmin, vacc3x89ABCDEF);
    vacc4x89ABCDEF = _mm256_max_ps(vmin, vacc4x89ABCDEF);

    vacc0x01234567 = _mm256_min_ps(vmax, vacc0x01234567);
    vacc1x01234567 = _mm256_min_ps(vmax, vacc1x01234567);
    vacc2x01234567 = _mm256_min_ps(vmax, vacc2x01234567);
    vacc3x01234567 = _mm256_min_ps(vmax, vacc3x01234567);
    vacc4x01234567 = _mm256_min_ps(vmax, vacc4x01234567);
    vacc0x89ABCDEF = _mm256_min_ps(vmax, vacc0x89ABCDEF);
    vacc1x89ABCDEF = _mm256_min_ps(vmax, vacc1x89ABCDEF);
    vacc2x89ABCDEF = _mm256_min_ps(vmax, vacc2x89ABCDEF);
    vacc3x89ABCDEF = _mm256_min_ps(vmax, vacc3x89ABCDEF);
    vacc4x89ABCDEF = _mm256_min_ps(vmax, vacc4x89ABCDEF);

    if XNN_LIKELY(nc >= 16) {
      _mm256_storeu_ps(c4, vacc4x01234567);
      _mm256_storeu_ps(c4 + 8, vacc4x89ABCDEF);
      c4 = (float*) ((uintptr_t) c4 + cn_stride);
      _mm256_storeu_ps(c3, vacc3x01234567);
      _mm256_storeu_ps(c3 + 8, vacc3x89ABCDEF);
      c3 = (float*) ((uintptr_t) c3 + cn_stride);
      _mm256_storeu_ps(c2, vacc2x01234567);
      _mm256_storeu_ps(c2 + 8, vacc2x89ABCDEF);
      c2 = (float*) ((uintptr_t) c2 + cn_stride);
      _mm256_storeu_ps(c1, vacc1x01234567);
      _mm256_storeu_ps(c1 + 8, vacc1x89ABCDEF);
      c1 = (float*) ((uintptr_t) c1 + cn_stride);
      _mm256_storeu_ps(c0, vacc0x01234567);
      _mm256_storeu_ps(c0 + 8, vacc0x89ABCDEF);
      c0 = (float*) ((uintptr_t) c0 + cn_stride);

      a = (const float**restrict) ((uintptr_t) a - ks);
      nc -= 16;
    } else {
      if (nc & 8) {
        _mm256_storeu_ps(c4, vacc4x01234567);
        _mm256_storeu_ps(c3, vacc3x01234567);
        _mm256_storeu_ps(c2, vacc2x01234567);
        _mm256_storeu_ps(c1, vacc1x01234567);
        _mm256_storeu_ps(c0, vacc0x01234567);

        vacc4x01234567 = vacc4x89ABCDEF;
        vacc3x01234567 = vacc3x89ABCDEF;
        vacc2x01234567 = vacc2x89ABCDEF;
        vacc1x01234567 = vacc1x89ABCDEF;
        vacc0x01234567 = vacc0x89ABCDEF;

        c4 += 8;
        c3 += 8;
        c2 += 8;
        c1 += 8;
        c0 += 8;
      }
      __m128 vacc4x0123 = _mm256_castps256_ps128(vacc4x01234567);
      __m128 vacc3x0123 = _mm256_castps256_ps128(vacc3x01234567);
      __m128 vacc2x0123 = _mm256_castps256_ps128(vacc2x01234567);
      __m128 vacc1x0123 = _mm256_castps256_ps128(vacc1x01234567);
      __m128 vacc0x0123 = _mm256_castps256_ps128(vacc0x01234567);
      if (nc & 4) {
        _mm_storeu_ps(c4, vacc4x0123);
        _mm_storeu_ps(c3, vacc3x0123);
        _mm_storeu_ps(c2, vacc2x0123);
        _mm_storeu_ps(c1, vacc1x0123);
        _mm_storeu_ps(c0, vacc0x0123);

        vacc4x0123 = _mm256_extractf128_ps(vacc4x01234567, 1);
        vacc3x0123 = _mm256_extractf128_ps(vacc3x01234567, 1);
        vacc2x0123 = _mm256_extractf128_ps(vacc2x01234567, 1);
        vacc1x0123 = _mm256_extractf128_ps(vacc1x01234567, 1);
        vacc0x0123 = _mm256_extractf128_ps(vacc0x01234567, 1);

        c4 += 4;
        c3 += 4;
        c2 += 4;
        c1 += 4;
        c0 += 4;
      }
      if (nc & 2) {
        _mm_storel_pi((__m64*) c4, vacc4x0123);
        _mm_storel_pi((__m64*) c3, vacc3x0123);
        _mm_storel_pi((__m64*) c2, vacc2x0123);
        _mm_storel_pi((__m64*) c1, vacc1x0123);
        _mm_storel_pi((__m64*) c0, vacc0x0123);

        vacc4x0123 = _mm_movehl_ps(vacc4x0123, vacc4x0123);
        vacc3x0123 = _mm_movehl_ps(vacc3x0123, vacc3x0123);
        vacc2x0123 = _mm_movehl_ps(vacc2x0123, vacc2x0123);
        vacc1x0123 = _mm_movehl_ps(vacc1x0123, vacc1x0123);
        vacc0x0123 = _mm_movehl_ps(vacc0x0123, vacc0x0123);

        c4 += 2;
        c3 += 2;
        c2 += 2;
        c1 += 2;
        c0 += 2;
      }
      if (nc & 1) {
        _mm_store_ss(c4, vacc4x0123);
        _mm_store_ss(c3, vacc3x0123);
        _mm_store_ss(c2, vacc2x0123);
        _mm_store_ss(c1, vacc1x0123);
        _mm_store_ss(c0, vacc0x0123);
      }

      nc = 0;
    }
  } while (nc != 0);
}

void xnn_f32_qc4w_gemm_minmax_ukernel_1x16__fma3_broadcast(
    size_t mr,
    size_t nc,
    size_t kc,
    const float* restrict a,
    size_t a_stride,
    const void* restrict w,
    float* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_f32_qc4w_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(mr != 0);
  assert(mr <= 1);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(float) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  const float* a0 = a;
  float* c0 = c;
  const __m256i vmagic_bias_c0 = _mm256_load_si256((const __m256i*) params->avx.magic_bias_c0);
  const __m256i vmagic_bias_c1 = _mm256_load_si256((const __m256i*) params->avx.magic_bias_c1);
  const __m256 vmagic_bias_plus_kernel_zero_point_c0 = _mm256_load_ps(params->avx.magic_bias_plus_kernel_zero_point_c0);
  const __m256 vmagic_bias_plus_kernel_zero_point_c1 = _mm256_load_ps(params->avx.magic_bias_plus_kernel_zero_point_c1);

  const __m256 vmax = _mm256_set1_ps(params->avx.max);
  const __m256 vmin = _mm256_set1_ps(params->avx.min);
  XNN_FORCE_REALIZATION(vmin);
  XNN_FORCE_REALIZATION(vmax);

  do {
    __m256 vacc0x01234567 = _mm256_loadu_ps((const float*) w + 0);
    __m256 vacc0x89ABCDEF = _mm256_loadu_ps((const float*) w + 8);
    w = (const float*) w + 16;

    size_t k = kc;
    for (; k >= 2 * sizeof(float); k -= sizeof(float) * 2) {
      const __m256 va0c0 = _mm256_broadcast_ss(a0);
      a0 += 1;
      const __m256 va0c1 = _mm256_broadcast_ss(a0);
      a0 += 1;

      const __m128i vbi0123c01 = _mm_cvtepu8_epi32(_mm_cvtsi32_si128((int) unaligned_load_u32((const uint8_t*) w)));
      const __m128i vbi4567c01 = _mm_cvtepu8_epi32(_mm_cvtsi32_si128((int) unaligned_load_u32((const uint8_t*) w + 4)));
      const __m128i vbi89ABc01 = _mm_cvtepu8_epi32(_mm_cvtsi32_si128((int) unaligned_load_u32((const uint8_t*) w + 8)));
      const __m128i vbiCDEFc01 = _mm_cvtepu8_epi32(_mm_cvtsi32_si128((int) unaligned_load_u32((const uint8_t*) w + 12)));
      const __m256 vbi01234567c01 = _mm256_insertf128_ps(_mm256_castsi256_ps(_mm256_castsi128_si256(vbi0123c01)), _mm_castsi128_ps(vbi4567c01), 1);
      const __m256 vbi89ABCDEFc01 = _mm256_insertf128_ps(_mm256_castsi256_ps(_mm256_castsi128_si256(vbi89ABc01)), _mm_castsi128_ps(vbiCDEFc01), 1);
      w = (const uint8_t*) w + 16;

      const __m256 vbm01234567c0 = _mm256_or_ps(vbi01234567c01, _mm256_castsi256_ps(vmagic_bias_c0));
      const __m256 vbm89ABCDEFc0 = _mm256_or_ps(vbi89ABCDEFc01, _mm256_castsi256_ps(vmagic_bias_c0));
      const __m256 vbm01234567c1 = _mm256_or_ps(vbi01234567c01, _mm256_castsi256_ps(vmagic_bias_c1));
      const __m256 vbm89ABCDEFc1 = _mm256_or_ps(vbi89ABCDEFc01, _mm256_castsi256_ps(vmagic_bias_c1));

      const __m256 vb01234567c0 = _mm256_sub_ps(vbm01234567c0, vmagic_bias_plus_kernel_zero_point_c0);
      const __m256 vb89ABCDEFc0 = _mm256_sub_ps(vbm89ABCDEFc0, vmagic_bias_plus_kernel_zero_point_c0);
      const __m256 vb01234567c1 = _mm256_sub_ps(vbm01234567c1, vmagic_bias_plus_kernel_zero_point_c1);
      const __m256 vb89ABCDEFc1 = _mm256_sub_ps(vbm89ABCDEFc1, vmagic_bias_plus_kernel_zero_point_c1);

      vacc0x01234567 = _mm256_fmadd_ps(va0c0, vb01234567c0, vacc0x01234567);
      vacc0x89ABCDEF = _mm256_fmadd_ps(va0c0, vb89ABCDEFc0, vacc0x89ABCDEF);
      vacc0x01234567 = _mm256_fmadd_ps(va0c1, vb01234567c1, vacc0x01234567);
      vacc0x89ABCDEF = _mm256_fmadd_ps(va0c1, vb89ABCDEFc1, vacc0x89ABCDEF);
    }

    if XNN_UNLIKELY(k != 0) {
      const __m256 va0 = _mm256_broadcast_ss(a0);
      a0 += 1;

      const __m128i vbi0123 = _mm_cvtepu8_epi32(_mm_cvtsi32_si128((int) unaligned_load_u32((const uint8_t*) w)));
      const __m128i vbi4567 = _mm_cvtepu8_epi32(_mm_cvtsi32_si128((int) unaligned_load_u32((const uint8_t*) w + 4)));
      const __m128i vbi89AB = _mm_cvtepu8_epi32(_mm_cvtsi32_si128((int) unaligned_load_u32((const uint8_t*) w + 8)));
      const __m128i vbiCDEF = _mm_cvtepu8_epi32(_mm_cvtsi32_si128((int) unaligned_load_u32((const uint8_t*) w + 12)));
      const __m256 vbi01234567 = _mm256_insertf128_ps(_mm256_castsi256_ps(_mm256_castsi128_si256(vbi0123)), _mm_castsi128_ps(vbi4567), 1);
      const __m256 vbi89ABCDEF = _mm256_insertf128_ps(_mm256_castsi256_ps(_mm256_castsi128_si256(vbi89AB)), _mm_castsi128_ps(vbiCDEF), 1);
      w = (const uint8_t*) w + 16;

      const __m256 vbm01234567 = _mm256_or_ps(vbi01234567, _mm256_castsi256_ps(vmagic_bias_c0));
      const __m256 vbm89ABCDEF = _mm256_or_ps(vbi89ABCDEF, _mm256_castsi256_ps(vmagic_bias_c0));

      const __m256 vb01234567 = _mm256_sub_ps(vbm01234567, vmagic_bias_plus_kernel_zero_point_c0);
      const __m256 vb89ABCDEF = _mm256_sub_ps(vbm89ABCDEF, vmagic_bias_plus_kernel_zero_point_c0);

      vacc0x01234567 = _mm256_fmadd_ps(va0, vb01234567, vacc0x01234567);
      vacc0x89ABCDEF = _mm256_fmadd_ps(va0, vb89ABCDEF, vacc0x89ABCDEF);
    }

    const __m256 vscale01234567 = _mm256_loadu_ps((const float*) w + 0);
    vacc0x01234567 = _mm256_mul_ps(vacc0x01234567, vscale01234567);
    const __m256 vscale89ABCDEF = _mm256_loadu_ps((const float*) w + 8);
    vacc0x89ABCDEF = _mm256_mul_ps(vacc0x89ABCDEF, vscale89ABCDEF);
    w = (const float*) w + 16;
    vacc0x01234567 = _mm256_max_ps(vmin, vacc0x01234567);
    vacc0x89ABCDEF = _mm256_max_ps(vmin, vacc0x89ABCDEF);

    vacc0x01234567 = _mm256_min_ps(vmax, vacc0x01234567);
    vacc0x89ABCDEF = _mm256_min_ps(vmax, vacc0x89ABCDEF);

    if XNN_LIKELY(nc >= 16) {
      _mm256_storeu_ps(c0, vacc0x01234567);
      _mm256_storeu_ps(c0 + 8, vacc0x89ABCDEF);
      c0 = (float*) ((uintptr_t) c0 + cn_stride);

      a0 = (const float*) ((uintptr_t) a0 - kc);

      nc -= 16;
    } else {
      if (nc & 8) {
        _mm256_storeu_ps(c0, vacc0x01234567);

        vacc0x01234567 = vacc0x89ABCDEF;

        c0 += 8;
      }
      __m128 vacc0x0123 = _mm256_castps256_ps128(vacc0x01234567);
      if (nc & 4) {
        _mm_storeu_ps(c0, vacc0x0123);

        vacc0x0123 = _mm256_extractf128_ps(vacc0x01234567, 1);

        c0 += 4;
      }
      if (nc & 2) {
        _mm_storel_pi((__m64*) c0, vacc0x0123);

        vacc0x0123 = _mm_movehl_ps(vacc0x0123, vacc0x0123);

        c0 += 2;
      }
      if (nc & 1) {
        _mm_store_ss(c0, vacc0x0123);
      }

      nc = 0;
    }
  } while (nc != 0);
}

void xnn_f32_qc4w_gemm_minmax_ukernel_3x16__fma3_broadcast(
    size_t mr,
    size_t nc,
    size_t kc,
    const float* restrict a,
    size_t a_stride,
    const void* restrict w,
    float* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_f32_qc4w_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(mr != 0);
  assert(mr <= 3);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(float) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  const float* a0 = a;
  float* c0 = c;
  const float* a1 = (const float*) ((uintptr_t) a0 + a_stride);
  float* c1 = (float*) ((uintptr_t) c0 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 2) {
    a1 = a0;
    c1 = c0;
  }
  const float* a2 = (const float*) ((uintptr_t) a1 + a_stride);
  float* c2 = (float*) ((uintptr_t) c1 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 2) {
    a2 = a1;
    c2 = c1;
  }
  const __m256i vmagic_bias_c0 = _mm256_load_si256((const __m256i*) params->avx.magic_bias_c0);
  const __m256i vmagic_bias_c1 = _mm256_load_si256((const __m256i*) params->avx.magic_bias_c1);
  const __m256 vmagic_bias_plus_kernel_zero_point_c0 = _mm256_load_ps(params->avx.magic_bias_plus_kernel_zero_point_c0);
  const __m256 vmagic_bias_plus_kernel_zero_point_c1 = _mm256_load_ps(params->avx.magic_bias_plus_kernel_zero_point_c1);

  const __m256 vmax = _mm256_set1_ps(params->avx.max);
  const __m256 vmin = _mm256_set1_ps(params->avx.min);
  XNN_FORCE_REALIZATION(vmin);
  XNN_FORCE_REALIZATION(vmax);

  do {
    __m256 vacc0x01234567 = _mm256_loadu_ps((const float*) w + 0);
    __m256 vacc0x89ABCDEF = _mm256_loadu_ps((const float*) w + 8);
    __m256 vacc1x01234567 = vacc0x01234567;
    __m256 vacc1x89ABCDEF = vacc0x89ABCDEF;
    __m256 vacc2x01234567 = vacc0x01234567;
    __m256 vacc2x89ABCDEF = vacc0x89ABCDEF;
    w = (const float*) w + 16;

    size_t k = kc;
    for (; k >= 2 * sizeof(float); k -= sizeof(float) * 2) {
      const __m256 va0c0 = _mm256_broadcast_ss(a0);
      a0 += 1;
      const __m256 va1c0 = _mm256_broadcast_ss(a1);
      a1 += 1;
      const __m256 va2c0 = _mm256_broadcast_ss(a2);
      a2 += 1;
      const __m256 va0c1 = _mm256_broadcast_ss(a0);
      a0 += 1;
      const __m256 va1c1 = _mm256_broadcast_ss(a1);
      a1 += 1;
      const __m256 va2c1 = _mm256_broadcast_ss(a2);
      a2 += 1;

      const __m128i vbi0123c01 = _mm_cvtepu8_epi32(_mm_cvtsi32_si128((int) unaligned_load_u32((const uint8_t*) w)));
      const __m128i vbi4567c01 = _mm_cvtepu8_epi32(_mm_cvtsi32_si128((int) unaligned_load_u32((const uint8_t*) w + 4)));
      const __m128i vbi89ABc01 = _mm_cvtepu8_epi32(_mm_cvtsi32_si128((int) unaligned_load_u32((const uint8_t*) w + 8)));
      const __m128i vbiCDEFc01 = _mm_cvtepu8_epi32(_mm_cvtsi32_si128((int) unaligned_load_u32((const uint8_t*) w + 12)));
      const __m256 vbi01234567c01 = _mm256_insertf128_ps(_mm256_castsi256_ps(_mm256_castsi128_si256(vbi0123c01)), _mm_castsi128_ps(vbi4567c01), 1);
      const __m256 vbi89ABCDEFc01 = _mm256_insertf128_ps(_mm256_castsi256_ps(_mm256_castsi128_si256(vbi89ABc01)), _mm_castsi128_ps(vbiCDEFc01), 1);
      w = (const uint8_t*) w + 16;

      const __m256 vbm01234567c0 = _mm256_or_ps(vbi01234567c01, _mm256_castsi256_ps(vmagic_bias_c0));
      const __m256 vbm89ABCDEFc0 = _mm256_or_ps(vbi89ABCDEFc01, _mm256_castsi256_ps(vmagic_bias_c0));
      const __m256 vbm01234567c1 = _mm256_or_ps(vbi01234567c01, _mm256_castsi256_ps(vmagic_bias_c1));
      const __m256 vbm89ABCDEFc1 = _mm256_or_ps(vbi89ABCDEFc01, _mm256_castsi256_ps(vmagic_bias_c1));

      const __m256 vb01234567c0 = _mm256_sub_ps(vbm01234567c0, vmagic_bias_plus_kernel_zero_point_c0);
      const __m256 vb89ABCDEFc0 = _mm256_sub_ps(vbm89ABCDEFc0, vmagic_bias_plus_kernel_zero_point_c0);
      const __m256 vb01234567c1 = _mm256_sub_ps(vbm01234567c1, vmagic_bias_plus_kernel_zero_point_c1);
      const __m256 vb89ABCDEFc1 = _mm256_sub_ps(vbm89ABCDEFc1, vmagic_bias_plus_kernel_zero_point_c1);

      vacc0x01234567 = _mm256_fmadd_ps(va0c0, vb01234567c0, vacc0x01234567);
      vacc1x01234567 = _mm256_fmadd_ps(va1c0, vb01234567c0, vacc1x01234567);
      vacc2x01234567 = _mm256_fmadd_ps(va2c0, vb01234567c0, vacc2x01234567);
      vacc0x89ABCDEF = _mm256_fmadd_ps(va0c0, vb89ABCDEFc0, vacc0x89ABCDEF);
      vacc1x89ABCDEF = _mm256_fmadd_ps(va1c0, vb89ABCDEFc0, vacc1x89ABCDEF);
      vacc2x89ABCDEF = _mm256_fmadd_ps(va2c0, vb89ABCDEFc0, vacc2x89ABCDEF);
      vacc0x01234567 = _mm256_fmadd_ps(va0c1, vb01234567c1, vacc0x01234567);
      vacc1x01234567 = _mm256_fmadd_ps(va1c1, vb01234567c1, vacc1x01234567);
      vacc2x01234567 = _mm256_fmadd_ps(va2c1, vb01234567c1, vacc2x01234567);
      vacc0x89ABCDEF = _mm256_fmadd_ps(va0c1, vb89ABCDEFc1, vacc0x89ABCDEF);
      vacc1x89ABCDEF = _mm256_fmadd_ps(va1c1, vb89ABCDEFc1, vacc1x89ABCDEF);
      vacc2x89ABCDEF = _mm256_fmadd_ps(va2c1, vb89ABCDEFc1, vacc2x89ABCDEF);
    }

    if XNN_UNLIKELY(k != 0) {
      const __m256 va0 = _mm256_broadcast_ss(a0);
      a0 += 1;
      const __m256 va1 = _mm256_broadcast_ss(a1);
      a1 += 1;
      const __m256 va2 = _mm256_broadcast_ss(a2);
      a2 += 1;

      const __m128i vbi0123 = _mm_cvtepu8_epi32(_mm_cvtsi32_si128((int) unaligned_load_u32((const uint8_t*) w)));
      const __m128i vbi4567 = _mm_cvtepu8_epi32(_mm_cvtsi32_si128((int) unaligned_load_u32((const uint8_t*) w + 4)));
      const __m128i vbi89AB = _mm_cvtepu8_epi32(_mm_cvtsi32_si128((int) unaligned_load_u32((const uint8_t*) w + 8)));
      const __m128i vbiCDEF = _mm_cvtepu8_epi32(_mm_cvtsi32_si128((int) unaligned_load_u32((const uint8_t*) w + 12)));
      const __m256 vbi01234567 = _mm256_insertf128_ps(_mm256_castsi256_ps(_mm256_castsi128_si256(vbi0123)), _mm_castsi128_ps(vbi4567), 1);
      const __m256 vbi89ABCDEF = _mm256_insertf128_ps(_mm256_castsi256_ps(_mm256_castsi128_si256(vbi89AB)), _mm_castsi128_ps(vbiCDEF), 1);
      w = (const uint8_t*) w + 16;

      const __m256 vbm01234567 = _mm256_or_ps(vbi01234567, _mm256_castsi256_ps(vmagic_bias_c0));
      const __m256 vbm89ABCDEF = _mm256_or_ps(vbi89ABCDEF, _mm256_castsi256_ps(vmagic_bias_c0));

      const __m256 vb01234567 = _mm256_sub_ps(vbm01234567, vmagic_bias_plus_kernel_zero_point_c0);
      const __m256 vb89ABCDEF = _mm256_sub_ps(vbm89ABCDEF, vmagic_bias_plus_kernel_zero_point_c0);

      vacc0x01234567 = _mm256_fmadd_ps(va0, vb01234567, vacc0x01234567);
      vacc1x01234567 = _mm256_fmadd_ps(va1, vb01234567, vacc1x01234567);
      vacc2x01234567 = _mm256_fmadd_ps(va2, vb01234567, vacc2x01234567);
      vacc0x89ABCDEF = _mm256_fmadd_ps(va0, vb89ABCDEF, vacc0x89ABCDEF);
      vacc1x89ABCDEF = _mm256_fmadd_ps(va1, vb89ABCDEF, vacc1x89ABCDEF);
      vacc2x89ABCDEF = _mm256_fmadd_ps(va2, vb89ABCDEF, vacc2x89ABCDEF);
    }

    const __m256 vscale01234567 = _mm256_loadu_ps((const float*) w + 0);
    vacc0x01234567 = _mm256_mul_ps(vacc0x01234567, vscale01234567);
    vacc1x01234567 = _mm256_mul_ps(vacc1x01234567, vscale01234567);
    vacc2x01234567 = _mm256_mul_ps(vacc2x01234567, vscale01234567);
    const __m256 vscale89ABCDEF = _mm256_loadu_ps((const float*) w + 8);
    vacc0x89ABCDEF = _mm256_mul_ps(vacc0x89ABCDEF, vscale89ABCDEF);
    vacc1x89ABCDEF = _mm256_mul_ps(vacc1x89ABCDEF, vscale89ABCDEF);
    vacc2x89ABCDEF = _mm256_mul_ps(vacc2x89ABCDEF, vscale89ABCDEF);
    w = (const float*) w + 16;
    vacc0x01234567 = _mm256_max_ps(vmin, vacc0x01234567);
    vacc1x01234567 = _mm256_max_ps(vmin, vacc1x01234567);
    vacc2x01234567 = _mm256_max_ps(vmin, vacc2x01234567);
    vacc0x89ABCDEF = _mm256_max_ps(vmin, vacc0x89ABCDEF);
    vacc1x89ABCDEF = _mm256_max_ps(vmin, vacc1x89ABCDEF);
    vacc2x89ABCDEF = _mm256_max_ps(vmin, vacc2x89ABCDEF);

    vacc0x01234567 = _mm256_min_ps(vmax, vacc0x01234567);
    vacc1x01234567 = _mm256_min_ps(vmax, vacc1x01234567);
    vacc2x01234567 = _mm256_min_ps(vmax, vacc2x01234567);
    vacc0x89ABCDEF = _mm256_min_ps(vmax, vacc0x89ABCDEF);
    vacc1x89ABCDEF = _mm256_min_ps(vmax, vacc1x89ABCDEF);
    vacc2x89ABCDEF = _mm256_min_ps(vmax, vacc2x89ABCDEF);

    if XNN_LIKELY(nc >= 16) {
      _mm256_storeu_ps(c2, vacc2x01234567);
      _mm256_storeu_ps(c2 + 8, vacc2x89ABCDEF);
      c2 = (float*) ((uintptr_t) c2 + cn_stride);
      _mm256_storeu_ps(c1, vacc1x01234567);
      _mm256_storeu_ps(c1 + 8, vacc1x89ABCDEF);
      c1 = (float*) ((uintptr_t) c1 + cn_stride);
      _mm256_storeu_ps(c0, vacc0x01234567);
      _mm256_storeu_ps(c0 + 8, vacc0x89ABCDEF);
      c0 = (float*) ((uintptr_t) c0 + cn_stride);

      a2 = (const float*) ((uintptr_t) a2 - kc);
      a1 = (const float*) ((uintptr_t) a1 - kc);
      a0 = (const float*) ((uintptr_t) a0 - kc);

      nc -= 16;
    } else {
      if (nc & 8) {
        _mm256_storeu_ps(c2, vacc2x01234567);
        _mm256_storeu_ps(c1, vacc1x01234567);
        _mm256_storeu_ps(c0, vacc0x01234567);

        vacc2x01234567 = vacc2x89ABCDEF;
        vacc1x01234567 = vacc1x89ABCDEF;
        vacc0x01234567 = vacc0x89ABCDEF;

        c2 += 8;
        c1 += 8;
        c0 += 8;
      }
      __m128 vacc2x0123 = _mm256_castps256_ps128(vacc2x01234567);
      __m128 vacc1x0123 = _mm256_castps256_ps128(vacc1x01234567);
      __m128 vacc0x0123 = _mm256_castps256_ps128(vacc0x01234567);
      if (nc & 4) {
        _mm_storeu_ps(c2, vacc2x0123);
        _mm_storeu_ps(c1, vacc1x0123);
        _mm_storeu_ps(c0, vacc0x0123);

        vacc2x0123 = _mm256_extractf128_ps(vacc2x01234567, 1);
        vacc1x0123 = _mm256_extractf128_ps(vacc1x01234567, 1);
        vacc0x0123 = _mm256_extractf128_ps(vacc0x01234567, 1);

        c2 += 4;
        c1 += 4;
        c0 += 4;
      }
      if (nc & 2) {
        _mm_storel_pi((__m64*) c2, vacc2x0123);
        _mm_storel_pi((__m64*) c1, vacc1x0123);
        _mm_storel_pi((__m64*) c0, vacc0x0123);

        vacc2x0123 = _mm_movehl_ps(vacc2x0123, vacc2x0123);
        vacc1x0123 = _mm_movehl_ps(vacc1x0123, vacc1x0123);
        vacc0x0123 = _mm_movehl_ps(vacc0x0123, vacc0x0123);

        c2 += 2;
        c1 += 2;
        c0 += 2;
      }
      if (nc & 1) {
        _mm_store_ss(c2, vacc2x0123);
        _mm_store_ss(c1, vacc1x0123);
        _mm_store_ss(c0, vacc0x0123);
      }

      nc = 0;
    }
  } while (nc != 0);
}

void xnn_f32_qc8w_gemm_minmax_ukernel_1x16__fma3_broadcast(
    size_t mr,
    size_t nc,
    size_t kc,
    const float* restrict a,
    size_t a_stride,
    const void* restrict w,
    float* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(mr != 0);
  assert(mr <= 1);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(float) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  const float* a0 = a;
  float* c0 = c;

  const __m256 vmin = _mm256_set1_ps(params->avx.min);
  const __m256 vmax = _mm256_set1_ps(params->avx.max);
  XNN_FORCE_REALIZATION(vmin);
  XNN_FORCE_REALIZATION(vmax);

  do {
    __m256 vacc0x01234567 = _mm256_loadu_ps((const float*) w + 0);
    __m256 vacc0x89ABCDEF = _mm256_loadu_ps((const float*) w + 8);
    w = (const float*) w + 16;

    size_t k = kc;
    do {
      const __m256 va0 = _mm256_broadcast_ss(a0);
      a0 += 1;

      const __m128i vbi0123 = _mm_cvtepi8_epi32(_mm_cvtsi32_si128((int) unaligned_load_u32((const int8_t*) w)));
      const __m128i vbi4567 = _mm_cvtepi8_epi32(_mm_cvtsi32_si128((int) unaligned_load_u32((const int8_t*) w + 4)));
      const __m128i vbi89AB = _mm_cvtepi8_epi32(_mm_cvtsi32_si128((int) unaligned_load_u32((const int8_t*) w + 8)));
      const __m128i vbiCDEF = _mm_cvtepi8_epi32(_mm_cvtsi32_si128((int) unaligned_load_u32((const int8_t*) w + 12)));
      const __m256i vbi01234567 = _mm256_castps_si256(_mm256_insertf128_ps(_mm256_castsi256_ps(_mm256_castsi128_si256(vbi0123)), _mm_castsi128_ps(vbi4567), 1));
      const __m256i vbi89ABCDEF = _mm256_castps_si256(_mm256_insertf128_ps(_mm256_castsi256_ps(_mm256_castsi128_si256(vbi89AB)), _mm_castsi128_ps(vbiCDEF), 1));
      w = (const int8_t*) w + 16;
      const __m256 vb01234567 = _mm256_cvtepi32_ps(vbi01234567);
      const __m256 vb89ABCDEF = _mm256_cvtepi32_ps(vbi89ABCDEF);

      vacc0x01234567 = _mm256_fmadd_ps(va0, vb01234567, vacc0x01234567);
      vacc0x89ABCDEF = _mm256_fmadd_ps(va0, vb89ABCDEF, vacc0x89ABCDEF);

      k -= sizeof(float);
    } while (k != 0);

    const __m256 vscale01234567 = _mm256_loadu_ps((const float*) w + 0);
    vacc0x01234567 = _mm256_mul_ps(vacc0x01234567, vscale01234567);
    const __m256 vscale89ABCDEF = _mm256_loadu_ps((const float*) w + 8);
    vacc0x89ABCDEF = _mm256_mul_ps(vacc0x89ABCDEF, vscale89ABCDEF);
    w = (const float*) w + 16;
    vacc0x01234567 = _mm256_max_ps(vmin, vacc0x01234567);
    vacc0x89ABCDEF = _mm256_max_ps(vmin, vacc0x89ABCDEF);

    vacc0x01234567 = _mm256_min_ps(vmax, vacc0x01234567);
    vacc0x89ABCDEF = _mm256_min_ps(vmax, vacc0x89ABCDEF);

    if XNN_LIKELY(nc >= 16) {
      _mm256_storeu_ps(c0, vacc0x01234567);
      _mm256_storeu_ps(c0 + 8, vacc0x89ABCDEF);
      c0 = (float*) ((uintptr_t) c0 + cn_stride);

      a0 = (const float*) ((uintptr_t) a0 - kc);

      nc -= 16;
    } else {
      if (nc & 8) {
        _mm256_storeu_ps(c0, vacc0x01234567);

        vacc0x01234567 = vacc0x89ABCDEF;

        c0 += 8;
      }
      __m128 vacc0x0123 = _mm256_castps256_ps128(vacc0x01234567);
      if (nc & 4) {
        _mm_storeu_ps(c0, vacc0x0123);

        vacc0x0123 = _mm256_extractf128_ps(vacc0x01234567, 1);

        c0 += 4;
      }
      if (nc & 2) {
        _mm_storel_pi((__m64*) c0, vacc0x0123);

        vacc0x0123 = _mm_movehl_ps(vacc0x0123, vacc0x0123);

        c0 += 2;
      }
      if (nc & 1) {
        _mm_store_ss(c0, vacc0x0123);
      }

      nc = 0;
    }
  } while (nc != 0);
}

void xnn_f32_qc8w_gemm_minmax_ukernel_5x16__fma3_broadcast(
    size_t mr,
    size_t nc,
    size_t kc,
    const float* restrict a,
    size_t a_stride,
    const void* restrict w,
    float* restrict c,
    size_t cm_stride,
    size_t cn_stride,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(mr != 0);
  assert(mr <= 5);
  assert(nc != 0);
  assert(kc != 0);
  assert(kc % sizeof(float) == 0);
  assert(a != NULL);
  assert(w != NULL);
  assert(c != NULL);

  const float* a0 = a;
  float* c0 = c;
  const float* a1 = (const float*) ((uintptr_t) a0 + a_stride);
  float* c1 = (float*) ((uintptr_t) c0 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 2) {
    a1 = a0;
    c1 = c0;
  }
  const float* a2 = (const float*) ((uintptr_t) a1 + a_stride);
  float* c2 = (float*) ((uintptr_t) c1 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 2) {
    a2 = a1;
    c2 = c1;
  }
  const float* a3 = (const float*) ((uintptr_t) a2 + a_stride);
  float* c3 = (float*) ((uintptr_t) c2 + cm_stride);
  if XNN_UNPREDICTABLE(mr < 4) {
    a3 = a2;
    c3 = c2;
  }
  const float* a4 = (const float*) ((uintptr_t) a3 + a_stride);
  float* c4 = (float*) ((uintptr_t) c3 + cm_stride);
  if XNN_UNPREDICTABLE(mr <= 4) {
    a4 = a3;
    c4 = c3;
  }

  const __m256 vmin = _mm256_set1_ps(params->avx.min);
  const __m256 vmax = _mm256_set1_ps(params->avx.max);
  XNN_FORCE_REALIZATION(vmin);
  XNN_FORCE_REALIZATION(vmax);

  do {
    __m256 vacc0x01234567 = _mm256_loadu_ps((const float*) w + 0);
    __m256 vacc0x89ABCDEF = _mm256_loadu_ps((const float*) w + 8);
    __m256 vacc1x01234567 = vacc0x01234567;
    __m256 vacc1x89ABCDEF = vacc0x89ABCDEF;
    __m256 vacc2x01234567 = vacc0x01234567;
    __m256 vacc2x89ABCDEF = vacc0x89ABCDEF;
    __m256 vacc3x01234567 = vacc0x01234567;
    __m256 vacc3x89ABCDEF = vacc0x89ABCDEF;
    __m256 vacc4x01234567 = vacc0x01234567;
    __m256 vacc4x89ABCDEF = vacc0x89ABCDEF;
    w = (const float*) w + 16;

    size_t k = kc;
    do {
      const __m256 va0 = _mm256_broadcast_ss(a0);
      a0 += 1;
      const __m256 va1 = _mm256_broadcast_ss(a1);
      a1 += 1;
      const __m256 va2 = _mm256_broadcast_ss(a2);
      a2 += 1;
      const __m256 va3 = _mm256_broadcast_ss(a3);
      a3 += 1;
      const __m256 va4 = _mm256_broadcast_ss(a4);
      a4 += 1;

      const __m128i vbi0123 = _mm_cvtepi8_epi32(_mm_cvtsi32_si128((int) unaligned_load_u32((const int8_t*) w)));
      const __m128i vbi4567 = _mm_cvtepi8_epi32(_mm_cvtsi32_si128((int) unaligned_load_u32((const int8_t*) w + 4)));
      const __m128i vbi89AB = _mm_cvtepi8_epi32(_mm_cvtsi32_si128((int) unaligned_load_u32((const int8_t*) w + 8)));
      const __m128i vbiCDEF = _mm_cvtepi8_epi32(_mm_cvtsi32_si128((int) unaligned_load_u32((const int8_t*) w + 12)));
      const __m256i vbi01234567 = _mm256_castps_si256(_mm256_insertf128_ps(_mm256_castsi256_ps(_mm256_castsi128_si256(vbi0123)), _mm_castsi128_ps(vbi4567), 1));
      const __m256i vbi89ABCDEF = _mm256_castps_si256(_mm256_insertf128_ps(_mm256_castsi256_ps(_mm256_castsi128_si256(vbi89AB)), _mm_castsi128_ps(vbiCDEF), 1));
      w = (const int8_t*) w + 16;
      const __m256 vb01234567 = _mm256_cvtepi32_ps(vbi01234567);
      const __m256 vb89ABCDEF = _mm256_cvtepi32_ps(vbi89ABCDEF);

      vacc0x01234567 = _mm256_fmadd_ps(va0, vb01234567, vacc0x01234567);
      vacc1x01234567 = _mm256_fmadd_ps(va1, vb01234567, vacc1x01234567);
      vacc2x01234567 = _mm256_fmadd_ps(va2, vb01234567, vacc2x01234567);
      vacc3x01234567 = _mm256_fmadd_ps(va3, vb01234567, vacc3x01234567);
      vacc4x01234567 = _mm256_fmadd_ps(va4, vb01234567, vacc4x01234567);
      vacc0x89ABCDEF = _mm256_fmadd_ps(va0, vb89ABCDEF, vacc0x89ABCDEF);
      vacc1x89ABCDEF = _mm256_fmadd_ps(va1, vb89ABCDEF, vacc1x89ABCDEF);
      vacc2x89ABCDEF = _mm256_fmadd_ps(va2, vb89ABCDEF, vacc2x89ABCDEF);
      vacc3x89ABCDEF = _mm256_fmadd_ps(va3, vb89ABCDEF, vacc3x89ABCDEF);
      vacc4x89ABCDEF = _mm256_fmadd_ps(va4, vb89ABCDEF, vacc4x89ABCDEF);

      k -= sizeof(float);
    } while (k != 0);

    const __m256 vscale01234567 = _mm256_loadu_ps((const float*) w + 0);
    vacc0x01234567 = _mm256_mul_ps(vacc0x01234567, vscale01234567);
    vacc1x01234567 = _mm256_mul_ps(vacc1x01234567, vscale01234567);
    vacc2x01234567 = _mm256_mul_ps(vacc2x01234567, vscale01234567);
    vacc3x01234567 = _mm256_mul_ps(vacc3x01234567, vscale01234567);
    vacc4x01234567 = _mm256_mul_ps(vacc4x01234567, vscale01234567);
    const __m256 vscale89ABCDEF = _mm256_loadu_ps((const float*) w + 8);
    vacc0x89ABCDEF = _mm256_mul_ps(vacc0x89ABCDEF, vscale89ABCDEF);
    vacc1x89ABCDEF = _mm256_mul_ps(vacc1x89ABCDEF, vscale89ABCDEF);
    vacc2x89ABCDEF = _mm256_mul_ps(vacc2x89ABCDEF, vscale89ABCDEF);
    vacc3x89ABCDEF = _mm256_mul_ps(vacc3x89ABCDEF, vscale89ABCDEF);
    vacc4x89ABCDEF = _mm256_mul_ps(vacc4x89ABCDEF, vscale89ABCDEF);
    w = (const float*) w + 16;
    vacc0x01234567 = _mm256_max_ps(vmin, vacc0x01234567);
    vacc1x01234567 = _mm256_max_ps(vmin, vacc1x01234567);
    vacc2x01234567 = _mm256_max_ps(vmin, vacc2x01234567);
    vacc3x01234567 = _mm256_max_ps(vmin, vacc3x01234567);
    vacc4x01234567 = _mm256_max_ps(vmin, vacc4x01234567);
    vacc0x89ABCDEF = _mm256_max_ps(vmin, vacc0x89ABCDEF);
    vacc1x89ABCDEF = _mm256_max_ps(vmin, vacc1x89ABCDEF);
    vacc2x89ABCDEF = _mm256_max_ps(vmin, vacc2x89ABCDEF);
    vacc3x89ABCDEF = _mm256_max_ps(vmin, vacc3x89ABCDEF);
    vacc4x89ABCDEF = _mm256_max_ps(vmin, vacc4x89ABCDEF);

    vacc0x01234567 = _mm256_min_ps(vmax, vacc0x01234567);
    vacc1x01234567 = _mm256_min_ps(vmax, vacc1x01234567);
    vacc2x01234567 = _mm256_min_ps(vmax, vacc2x01234567);
    vacc3x01234567 = _mm256_min_ps(vmax, vacc3x01234567);
    vacc4x01234567 = _mm256_min_ps(vmax, vacc4x01234567);
    vacc0x89ABCDEF = _mm256_min_ps(vmax, vacc0x89ABCDEF);
    vacc1x89ABCDEF = _mm256_min_ps(vmax, vacc1x89ABCDEF);
    vacc2x89ABCDEF = _mm256_min_ps(vmax, vacc2x89ABCDEF);
    vacc3x89ABCDEF = _mm256_min_ps(vmax, vacc3x89ABCDEF);
    vacc4x89ABCDEF = _mm256_min_ps(vmax, vacc4x89ABCDEF);

    if XNN_LIKELY(nc >= 16) {
      _mm256_storeu_ps(c0, vacc0x01234567);
      _mm256_storeu_ps(c0 + 8, vacc0x89ABCDEF);
      c0 = (float*) ((uintptr_t) c0 + cn_stride);
      _mm256_storeu_ps(c1, vacc1x01234567);
      _mm256_storeu_ps(c1 + 8, vacc1x89ABCDEF);
      c1 = (float*) ((uintptr_t) c1 + cn_stride);
      _mm256_storeu_ps(c2, vacc2x01234567);
      _mm256_storeu_ps(c2 + 8, vacc2x89ABCDEF);
      c2 = (float*) ((uintptr_t) c2 + cn_stride);
      _mm256_storeu_ps(c3, vacc3x01234567);
      _mm256_storeu_ps(c3 + 8, vacc3x89ABCDEF);
      c3 = (float*) ((uintptr_t) c3 + cn_stride);
      _mm256_storeu_ps(c4, vacc4x01234567);
      _mm256_storeu_ps(c4 + 8, vacc4x89ABCDEF);
      c4 = (float*) ((uintptr_t) c4 + cn_stride);

      a0 = (const float*) ((uintptr_t) a0 - kc);
      a1 = (const float*) ((uintptr_t) a1 - kc);
      a2 = (const float*) ((uintptr_t) a2 - kc);
      a3 = (const float*) ((uintptr_t) a3 - kc);
      a4 = (const float*) ((uintptr_t) a4 - kc);

      nc -= 16;
    } else {
      if (nc & 8) {
        _mm256_storeu_ps(c0, vacc0x01234567);
        _mm256_storeu_ps(c1, vacc1x01234567);
        _mm256_storeu_ps(c2, vacc2x01234567);
        _mm256_storeu_ps(c3, vacc3x01234567);
        _mm256_storeu_ps(c4, vacc4x01234567);

        vacc0x01234567 = vacc0x89ABCDEF;
        vacc1x01234567 = vacc1x89ABCDEF;
        vacc2x01234567 = vacc2x89ABCDEF;
        vacc3x01234567 = vacc3x89ABCDEF;
        vacc4x01234567 = vacc4x89ABCDEF;

        c0 += 8;
        c1 += 8;
        c2 += 8;
        c3 += 8;
        c4 += 8;
      }
      __m128 vacc0x0123 = _mm256_castps256_ps128(vacc0x01234567);
      __m128 vacc1x0123 = _mm256_castps256_ps128(vacc1x01234567);
      __m128 vacc2x0123 = _mm256_castps256_ps128(vacc2x01234567);
      __m128 vacc3x0123 = _mm256_castps256_ps128(vacc3x01234567);
      __m128 vacc4x0123 = _mm256_castps256_ps128(vacc4x01234567);
      if (nc & 4) {
        _mm_storeu_ps(c0, vacc0x0123);
        _mm_storeu_ps(c1, vacc1x0123);
        _mm_storeu_ps(c2, vacc2x0123);
        _mm_storeu_ps(c3, vacc3x0123);
        _mm_storeu_ps(c4, vacc4x0123);

        vacc0x0123 = _mm256_extractf128_ps(vacc0x01234567, 1);
        vacc1x0123 = _mm256_extractf128_ps(vacc1x01234567, 1);
        vacc2x0123 = _mm256_extractf128_ps(vacc2x01234567, 1);
        vacc3x0123 = _mm256_extractf128_ps(vacc3x01234567, 1);
        vacc4x0123 = _mm256_extractf128_ps(vacc4x01234567, 1);

        c0 += 4;
        c1 += 4;
        c2 += 4;
        c3 += 4;
        c4 += 4;
      }
      if (nc & 2) {
        _mm_storel_pi((__m64*) c0, vacc0x0123);
        _mm_storel_pi((__m64*) c1, vacc1x0123);
        _mm_storel_pi((__m64*) c2, vacc2x0123);
        _mm_storel_pi((__m64*) c3, vacc3x0123);
        _mm_storel_pi((__m64*) c4, vacc4x0123);

        vacc0x0123 = _mm_movehl_ps(vacc0x0123, vacc0x0123);
        vacc1x0123 = _mm_movehl_ps(vacc1x0123, vacc1x0123);
        vacc2x0123 = _mm_movehl_ps(vacc2x0123, vacc2x0123);
        vacc3x0123 = _mm_movehl_ps(vacc3x0123, vacc3x0123);
        vacc4x0123 = _mm_movehl_ps(vacc4x0123, vacc4x0123);

        c0 += 2;
        c1 += 2;
        c2 += 2;
        c3 += 2;
        c4 += 2;
      }
      if (nc & 1) {
        _mm_store_ss(c0, vacc0x0123);
        _mm_store_ss(c1, vacc1x0123);
        _mm_store_ss(c2, vacc2x0123);
        _mm_store_ss(c3, vacc3x0123);
        _mm_store_ss(c4, vacc4x0123);
      }

      nc = 0;
    }
  } while (nc != 0);
}

void xnn_f32_vhswish_ukernel__fma3_u16(
    size_t batch,
    const float* input,
    float* output,
    const union xnn_f32_hswish_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  static const int32_t mask_table[14] = {-1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0};

  const __m256 vsixth = _mm256_set1_ps(0x1.555556p-3f);
  const __m256 vhalf = _mm256_set1_ps(0.5f);
  const __m256 vone = _mm256_set1_ps(1.0f);
  const __m256 vzero = _mm256_setzero_ps();

  XNN_FORCE_REALIZATION(vsixth);
  XNN_FORCE_REALIZATION(vhalf);
  XNN_FORCE_REALIZATION(vone);
  // XNN_FORCE_REALIZATION(vzero);

  for (; batch >= 16 * sizeof(float); batch -= 16 * sizeof(float)) {
    const __m256 vx01234567 = _mm256_loadu_ps(input);
    const __m256 vx89ABCDEF = _mm256_loadu_ps(input + 8);
    input += 16;

    __m256 vacc01234567 = _mm256_fmadd_ps(vx01234567, vsixth, vhalf);
    __m256 vacc89ABCDEF = _mm256_fmadd_ps(vx89ABCDEF, vsixth, vhalf);

    vacc01234567 = _mm256_max_ps(vacc01234567, vzero);
    vacc89ABCDEF = _mm256_max_ps(vacc89ABCDEF, vzero);

    vacc01234567 = _mm256_min_ps(vacc01234567, vone);
    vacc89ABCDEF = _mm256_min_ps(vacc89ABCDEF, vone);

    vacc01234567 = _mm256_mul_ps(vacc01234567, vx01234567);
    vacc89ABCDEF = _mm256_mul_ps(vacc89ABCDEF, vx89ABCDEF);

    _mm256_storeu_ps(output, vacc01234567);
    _mm256_storeu_ps(output + 8, vacc89ABCDEF);
    output += 16;
  }
  for (; batch >= 8 * sizeof(float); batch -= 8 * sizeof(float)) {
    const __m256 vx = _mm256_loadu_ps(input);
    input += 8;
    __m256 vacc = _mm256_fmadd_ps(vx, vsixth, vhalf);
    vacc = _mm256_max_ps(vacc, vzero);
    vacc = _mm256_min_ps(vacc, vone);
    vacc = _mm256_mul_ps(vacc, vx);
    _mm256_storeu_ps(output, vacc);
    output += 8;
  }
  if XNN_UNLIKELY(batch != 0) {
    assert(batch >= 1 * sizeof(float));
    assert(batch <= 7 * sizeof(float));
    const __m256i vmask = _mm256_loadu_si256((const __m256i*) ((uintptr_t) &mask_table[7] - batch));

    const __m256 vx = _mm256_maskload_ps(input, vmask);
    __m256 vacc = _mm256_fmadd_ps(vx, vsixth, vhalf);
    vacc = _mm256_max_ps(vacc, vzero);
    vacc = _mm256_min_ps(vacc, vone);
    vacc = _mm256_mul_ps(vacc, vx);

    __m128 vacc_lo = _mm256_castps256_ps128(vacc);
    if (batch & (4 * sizeof(float))) {
      _mm_storeu_ps(output, vacc_lo);
      vacc_lo = _mm256_extractf128_ps(vacc, 1);
      output += 4;
    }
    if (batch & (2 * sizeof(float))) {
      _mm_storel_pi((__m64*) output, vacc_lo);
      vacc_lo = _mm_movehl_ps(vacc_lo, vacc_lo);
      output += 2;
    }
    if (batch & (1 * sizeof(float))) {
      _mm_store_ss(output, vacc_lo);
    }
  }
}

void xnn_f32_vrsqrt_ukernel__fma3_rsqrt_u16(
    size_t batch,
    const float* input,
    float* output,
    const union xnn_f32_rsqrt_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  static const int32_t mask_table[14] = {-1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0};

  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  // Constants for the Newton-Raphson iteration.
  const __m256 vthree = _mm256_set1_ps(3.0f);
  const __m256 vneg_half = _mm256_set1_ps(-0.5f);

  for (; batch >= 16 * sizeof(float); batch -= 16 * sizeof(float)) {
    const __m256 vx0 = _mm256_loadu_ps(input);
    const __m256 vx1 = _mm256_loadu_ps(input + 8);
    input += 16;

    // Generate the initial 12-bit approximation.
    const __m256 vt0_0 = _mm256_rsqrt_ps(vx0);
    const __m256 vt0_1 = _mm256_rsqrt_ps(vx1);

    // Do a single Newton-Raphson step as described above.
    const __m256 vt1_0 = _mm256_mul_ps(vt0_0, vt0_0);
    const __m256 vt1_1 = _mm256_mul_ps(vt0_1, vt0_1);
    const __m256 vt3_0 = _mm256_fmsub_ps(vx0, vt1_0, vthree);
    const __m256 vt3_1 = _mm256_fmsub_ps(vx1, vt1_1, vthree);
    const __m256 vt4_0 = _mm256_mul_ps(vneg_half, vt0_0);
    const __m256 vt4_1 = _mm256_mul_ps(vneg_half, vt0_1);
    const __m256 vy0 = _mm256_mul_ps(vt3_0, vt4_0);
    const __m256 vy1 = _mm256_mul_ps(vt3_1, vt4_1);

    // Store the results.
    _mm256_storeu_ps(output, vy0);
    _mm256_storeu_ps(output + 8, vy1);
    output += 16;
  }
  for (; batch >= 8 * sizeof(float); batch -= 8 * sizeof(float)) {
    const __m256 vx = _mm256_loadu_ps(input);
    input += 8;

    // Generate the initial 12-bit approximation.
    const __m256 vt0 = _mm256_rsqrt_ps(vx);

    // Do a single Newton-Raphson step as described above.
    const __m256 vt1 = _mm256_mul_ps(vt0, vt0);
    const __m256 vt3 = _mm256_fmsub_ps(vx, vt1, vthree);
    const __m256 vt4 = _mm256_mul_ps(vneg_half, vt0);
    const __m256 vy = _mm256_mul_ps(vt3, vt4);

    _mm256_storeu_ps(output, vy);
    output += 8;
  }
  if XNN_UNLIKELY(batch != 0) {
    assert(batch >= 1 * sizeof(float));
    assert(batch <= 7 * sizeof(float));
    const __m256i vmask = _mm256_loadu_si256((const __m256i*)((uintptr_t) &mask_table[7] - batch));

    const __m256 vx = _mm256_maskload_ps(input, vmask);

    // Generate the initial 12-bit approximation.
    const __m256 vt0 = _mm256_rsqrt_ps(vx);

    // Do a single Newton-Raphson step as described above.
    const __m256 vt1 = _mm256_mul_ps(vt0, vt0);
    const __m256 vt3 = _mm256_fmsub_ps(vx, vt1, vthree);
    const __m256 vt4 = _mm256_mul_ps(vneg_half, vt0);
    __m256 vy = _mm256_mul_ps(vt3, vt4);

    __m128 vy_lo = _mm256_castps256_ps128(vy);
    if (batch & (4 * sizeof(float))) {
      _mm_storeu_ps(output, vy_lo);
      vy_lo = _mm256_extractf128_ps(vy, 1);
      output += 4;
    }
    if (batch & (2 * sizeof(float))) {
      _mm_storel_pi((__m64*) output, vy_lo);
      vy_lo = _mm_movehl_ps(vy_lo, vy_lo);
      output += 2;
    }
    if (batch & (1 * sizeof(float))) {
      _mm_store_ss(output, vy_lo);
    }
  }
}

void xnn_f32_vsqrt_ukernel__fma3_rsqrt_u16(
    size_t batch, const float* input, float* output,
    const union xnn_f32_sqrt_params params[restrict XNN_MIN_ELEMENTS(1)])
    XNN_OOB_READS {
  static const int32_t mask_table[14] = {-1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0};

  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  // Constants for the Newton-Raphson iteration.
  const __m256 vthree = _mm256_set1_ps(3.0f);
  const __m256 vneg_half = _mm256_set1_ps(-0.5f);

  for (; batch >= 16 * sizeof(float); batch -= 16 * sizeof(float)) {
    const __m256 vx0 = _mm256_loadu_ps(input);
    const __m256 vx1 = _mm256_loadu_ps(input + 8);
    input += 16;

    // Create a mask of the +/-0 inputs, which will be flushed to zero later.
    const __m256 vinf_mask_0 = _mm256_cmp_ps(vx0, _mm256_setzero_ps(), _CMP_EQ_OQ);
    const __m256 vinf_mask_1 = _mm256_cmp_ps(vx1, _mm256_setzero_ps(), _CMP_EQ_OQ);

    // Generate the initial 12-bit approximation.
    const __m256 vt0_0 = _mm256_rsqrt_ps(vx0);
    const __m256 vt0_1 = _mm256_rsqrt_ps(vx1);

    // Do a single Newton-Raphson step as described above.
    const __m256 vt1_0 = _mm256_mul_ps(vt0_0, vt0_0);
    const __m256 vt1_1 = _mm256_mul_ps(vt0_1, vt0_1);
    const __m256 vt3_0 = _mm256_fmsub_ps(vx0, vt1_0, vthree);
    const __m256 vt3_1 = _mm256_fmsub_ps(vx1, vt1_1, vthree);
    const __m256 vt4_0 = _mm256_mul_ps(vneg_half, vt0_0);
    const __m256 vt4_1 = _mm256_mul_ps(vneg_half, vt0_1);
    const __m256 vt5_0 = _mm256_mul_ps(vt3_0, vt4_0);
    const __m256 vt5_1 = _mm256_mul_ps(vt3_1, vt4_1);
    const __m256 vt6_0 = _mm256_andnot_ps(vinf_mask_0, vt5_0);
    const __m256 vt6_1 = _mm256_andnot_ps(vinf_mask_1, vt5_1);
    const __m256 vy0 = _mm256_mul_ps(vx0, vt6_0);
    const __m256 vy1 = _mm256_mul_ps(vx1, vt6_1);

    // Store the results.
    _mm256_storeu_ps(output, vy0);
    _mm256_storeu_ps(output + 8, vy1);
    output += 16;
  }
  for (; batch >= 8 * sizeof(float); batch -= 8 * sizeof(float)) {
    const __m256 vx = _mm256_loadu_ps(input);
    input += 8;

    // Create a mask of the +/-0 inputs, which will be flushed to zero later.
    const __m256 vinf_mask = _mm256_cmp_ps(vx, _mm256_setzero_ps(), _CMP_EQ_OQ);

    // Generate the initial 12-bit approximation.
    const __m256 vt0 = _mm256_rsqrt_ps(vx);

    // Do a single Newton-Raphson step as described above.
    const __m256 vt1 = _mm256_mul_ps(vt0, vt0);
    const __m256 vt3 = _mm256_fmsub_ps(vx, vt1, vthree);
    const __m256 vt4 = _mm256_mul_ps(vneg_half, vt0);
    const __m256 vt5 = _mm256_mul_ps(vt3, vt4);
    const __m256 vt6 = _mm256_andnot_ps(vinf_mask, vt5);
    const __m256 vy = _mm256_mul_ps(vx, vt6);

    _mm256_storeu_ps(output, vy);
    output += 8;
  }
  if XNN_UNLIKELY(batch != 0) {
    assert(batch >= 1 * sizeof(float));
    assert(batch <= 7 * sizeof(float));
    const __m256i vmask = _mm256_loadu_si256(
        (const __m256i*)((uintptr_t)&mask_table[7] - batch));

    const __m256 vx = _mm256_maskload_ps(input, vmask);

    // Create a mask of the +/-0 inputs, which will be flushed to zero later.
    const __m256 vinf_mask = _mm256_cmp_ps(vx, _mm256_setzero_ps(), _CMP_EQ_OQ);

    // Generate the initial 12-bit approximation.
    const __m256 vt0 = _mm256_rsqrt_ps(vx);

    // Do a single Newton-Raphson step as described above.
    const __m256 vt1 = _mm256_mul_ps(vt0, vt0);
    const __m256 vt3 = _mm256_fmsub_ps(vx, vt1, vthree);
    const __m256 vt4 = _mm256_mul_ps(vneg_half, vt0);
    const __m256 vt5 = _mm256_mul_ps(vt3, vt4);
    const __m256 vt6 = _mm256_andnot_ps(vinf_mask, vt5);
    __m256 vy = _mm256_mul_ps(vx, vt6);

    __m128 vy_lo = _mm256_castps256_ps128(vy);
    if (batch & (4 * sizeof(float))) {
      _mm_storeu_ps(output, vy_lo);
      vy_lo = _mm256_extractf128_ps(vy, 1);
      output += 4;
    }
    if (batch & (2 * sizeof(float))) {
      _mm_storel_pi((__m64*) output, vy_lo);
      vy_lo = _mm_movehl_ps(vy_lo, vy_lo);
      output += 2;
    }
    if (batch & (1 * sizeof(float))) {
      _mm_store_ss(output, vy_lo);
    }
  }
}

void xnn_f32_vgelu_ukernel__fma3_rational_12_10_div_u16(
    size_t batch,
    const float* input,
    float* output,
    const union xnn_f32_default_params unused_params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);
  assert(xnn_simd_size_f32 == 8);

  // Cap the inputs to this value as `erf(x/sqrt(2))` will always be `+/-1.0f`
  // beyond this point. This value is chosen as the first floating point
  // number as of which the interpolation returns +/-1.0f.
  #if XNN_SIMD_HAS_NATIVE_FMA || (XNN_ARCH_RISCV && XNN_ENABLE_RISCV_VECTOR)
    XNN_SIMD_CONST_F32(vmax_abs_x, 5.1638283730e+00f);
  #else
    XNN_SIMD_CONST_F32(vmax_abs_x, 5.1158981323e+00f);
  #endif  // XNN_SIMD_HAS_NATIVE_FMA

  // The monomial coefficients of the numerator polynomial (odd).
  XNN_SIMD_CONST_F32(valpha_1, 7.9788452387e-01f);
  XNN_SIMD_CONST_F32(valpha_3, 6.6972173750e-02f);
  XNN_SIMD_CONST_F32(valpha_5, 9.3065137044e-03f);
  XNN_SIMD_CONST_F32(valpha_7, 3.2973114867e-04f);
  XNN_SIMD_CONST_F32(valpha_9, 1.2609783880e-05f);
  XNN_SIMD_CONST_F32(valpha_11, 4.5835321316e-08f);

  // The monomial coefficients of the denominator polynomial (even).
  // XNN_SIMD_CONST_F32(vbeta_0, 1.0f);
  XNN_SIMD_CONST_F32(vbeta_2, 2.5060352683e-01f);
  XNN_SIMD_CONST_F32(vbeta_4, 2.8431978077e-02f);
  XNN_SIMD_CONST_F32(vbeta_6, 1.8622842617e-03f);
  XNN_SIMD_CONST_F32(vbeta_8, 7.2267655923e-05f);
  XNN_SIMD_CONST_F32(vbeta_10, 1.1988805682e-06f);

  XNN_SIMD_CONST_F32(vone, 1.0f);
  XNN_SIMD_CONST_F32(vhalf, 0.5f);

  for (; batch >= 16 * sizeof(float); batch -= 16 * sizeof(float)) {
    const xnn_simd_f32_t vx_orig_0 = xnn_loadu_f32(input);
    const xnn_simd_f32_t vx_orig_1 = xnn_loadu_f32(input + 1 * xnn_simd_size_f32);
    input += 16;

    // Clamp the inputs to the interpolation range.
    xnn_simd_f32_t vx_0 = xnn_min_f32(vmax_abs_x, vx_orig_0);
    xnn_simd_f32_t vx_1 = xnn_min_f32(vmax_abs_x, vx_orig_1);
    vx_0 = xnn_max_f32(xnn_neg_f32(vmax_abs_x), vx_0);
    vx_1 = xnn_max_f32(xnn_neg_f32(vmax_abs_x), vx_1);

    // Since the polynomials are odd/even, we need x^2.
    const xnn_simd_f32_t vx2_0 = xnn_mul_f32(vx_0, vx_0);
    const xnn_simd_f32_t vx2_1 = xnn_mul_f32(vx_1, vx_1);

    // Evaluate the numerator polynomial p.
    xnn_simd_f32_t vp_0 = xnn_fmadd_f32(vx2_0, valpha_11, valpha_9);
    xnn_simd_f32_t vp_1 = xnn_fmadd_f32(vx2_1, valpha_11, valpha_9);
    vp_0 = xnn_fmadd_f32(vx2_0, vp_0, valpha_7);
    vp_1 = xnn_fmadd_f32(vx2_1, vp_1, valpha_7);
    vp_0 = xnn_fmadd_f32(vx2_0, vp_0, valpha_5);
    vp_1 = xnn_fmadd_f32(vx2_1, vp_1, valpha_5);
    vp_0 = xnn_fmadd_f32(vx2_0, vp_0, valpha_3);
    vp_1 = xnn_fmadd_f32(vx2_1, vp_1, valpha_3);
    vp_0 = xnn_fmadd_f32(vx2_0, vp_0, valpha_1);
    vp_1 = xnn_fmadd_f32(vx2_1, vp_1, valpha_1);
    vp_0 = xnn_mul_f32(vx_0, vp_0);
    vp_1 = xnn_mul_f32(vx_1, vp_1);

    // Evaluate the denominator polynomial q.
    xnn_simd_f32_t vq_0 = xnn_fmadd_f32(vx2_0, vbeta_10, vbeta_8);
    xnn_simd_f32_t vq_1 = xnn_fmadd_f32(vx2_1, vbeta_10, vbeta_8);
    vq_0 = xnn_fmadd_f32(vx2_0, vq_0, vbeta_6);
    vq_1 = xnn_fmadd_f32(vx2_1, vq_1, vbeta_6);
    vq_0 = xnn_fmadd_f32(vx2_0, vq_0, vbeta_4);
    vq_1 = xnn_fmadd_f32(vx2_1, vq_1, vbeta_4);
    vq_0 = xnn_fmadd_f32(vx2_0, vq_0, vbeta_2);
    vq_1 = xnn_fmadd_f32(vx2_1, vq_1, vbeta_2);
    vq_0 = xnn_fmadd_f32(vx2_0, vq_0, vone);
    vq_1 = xnn_fmadd_f32(vx2_1, vq_1, vone);

    // Divide the numerator by the denominator.
    const xnn_simd_f32_t verf_0 = xnn_div_f32(vp_0, vq_0);
    const xnn_simd_f32_t verf_1 = xnn_div_f32(vp_1, vq_1);

    // Add one to the rational interpolant, and multiply by 0.5 times the
    // original input.
    const xnn_simd_f32_t vy_0 = xnn_mul_f32(xnn_mul_f32(vx_orig_0, vhalf),
                                        xnn_add_f32(verf_0, vone));
    const xnn_simd_f32_t vy_1 = xnn_mul_f32(xnn_mul_f32(vx_orig_1, vhalf),
                                        xnn_add_f32(verf_1, vone));

    xnn_storeu_f32(output, vy_0);
    xnn_storeu_f32(output + 1 * xnn_simd_size_f32, vy_1);
    output += 16;
  }
  for (; batch >= xnn_simd_bytes_f32; batch -= xnn_simd_bytes_f32) {
    const xnn_simd_f32_t vx_orig = xnn_loadu_f32(input);
    input += xnn_simd_size_f32;

    // Clamp the inputs to the interpolation range.
    xnn_simd_f32_t vx = xnn_min_f32(vmax_abs_x, vx_orig);
    vx = xnn_max_f32(xnn_neg_f32(vmax_abs_x), vx);

    // Since the polynomials are odd/even, we need x^2.
    const xnn_simd_f32_t vx2 = xnn_mul_f32(vx, vx);

    // Evaluate the numerator polynomial p.
    xnn_simd_f32_t vp = xnn_fmadd_f32(vx2, valpha_11, valpha_9);
    vp = xnn_fmadd_f32(vx2, vp, valpha_7);
    vp = xnn_fmadd_f32(vx2, vp, valpha_5);
    vp = xnn_fmadd_f32(vx2, vp, valpha_3);
    vp = xnn_fmadd_f32(vx2, vp, valpha_1);
    vp = xnn_mul_f32(vx, vp);

    // Evaluate the denominator polynomial q.
    xnn_simd_f32_t vq = xnn_fmadd_f32(vx2, vbeta_10, vbeta_8);
    vq = xnn_fmadd_f32(vx2, vq, vbeta_6);
    vq = xnn_fmadd_f32(vx2, vq, vbeta_4);
    vq = xnn_fmadd_f32(vx2, vq, vbeta_2);
    vq = xnn_fmadd_f32(vx2, vq, vone);

    // Divide the numerator by the denominator and add one
    const xnn_simd_f32_t verf =  xnn_div_f32(vp, vq);

    // Add one to the rational interpolant, and multiply by 0.5 times the
    // original input.
    const xnn_simd_f32_t vy = xnn_mul_f32(xnn_mul_f32(vx_orig, vhalf),
                                          xnn_add_f32(verf, vone));

    xnn_storeu_f32(output, vy);
    output += xnn_simd_size_f32;
  }
  if XNN_UNLIKELY(batch != 0) {
    xnn_simd_f32_t vx_orig = xnn_load_tail_f32(input, batch >> XNN_LOG2_SIZEOF_FLOAT);

  // See above for comments.
  xnn_simd_f32_t vx = xnn_min_f32(vmax_abs_x, vx_orig);
  vx = xnn_max_f32(xnn_neg_f32(vmax_abs_x), vx);
  const xnn_simd_f32_t vx2 = xnn_mul_f32(vx, vx);
  xnn_simd_f32_t vp = xnn_fmadd_f32(vx2, valpha_11, valpha_9);
  vp = xnn_fmadd_f32(vx2, vp, valpha_7);
  vp = xnn_fmadd_f32(vx2, vp, valpha_5);
  vp = xnn_fmadd_f32(vx2, vp, valpha_3);
  vp = xnn_fmadd_f32(vx2, vp, valpha_1);
  vp = xnn_mul_f32(vx, vp);
  xnn_simd_f32_t vq = xnn_fmadd_f32(vx2, vbeta_10, vbeta_8);
  vq = xnn_fmadd_f32(vx2, vq, vbeta_6);
  vq = xnn_fmadd_f32(vx2, vq, vbeta_4);
  vq = xnn_fmadd_f32(vx2, vq, vbeta_2);
  vq = xnn_fmadd_f32(vx2, vq, vone);
  const xnn_simd_f32_t verf =  xnn_div_f32(vp, vq);
  const xnn_simd_f32_t vy = xnn_mul_f32(xnn_mul_f32(vx_orig, vhalf),
                                        xnn_add_f32(verf, vone));

    xnn_store_tail_f32(output, vy, batch >> XNN_LOG2_SIZEOF_FLOAT);
  }
}

static XNN_INLINE xnn_simd_f32_t xnn_signed_getexp_f32(xnn_simd_f32_t a) {
  // The bits of IEE754 single-precision floating-point format are:
  //
  //   s | e e e e e e e e | m m m m m m m m m m m m m m m m m m m m m m m
  //
  // We start by masking out the sign and exponent and shifting it 8 bits to the
  // right arithmetically, i.e. extending by the leftmost sign bit:
  //
  //   s | s s s s s s s s | e e e e e e e e 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
  //
  // These bits are then `or`-ed with `256.0f`, which has a biased exponent of
  // `135` and all mantissa bit set to zero. This is equivalent to adding the
  // biased integer exponent to `256.0`:
  //
  //   0 | 1 0 0 0 0 1 1 1 | e e e e e e e e 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
  //
  // We can re-extract the exponent as a `float` value by subtracting `256.0`
  // plus the exponent bias `127.0`, i.e. `383.0`.
  //
  // Note that if the sign bit is `1`, we end up with the floating point bits:
  //
  //   1 | 1 1 1 1 1 1 1 1 | e e e e e e e e 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0
  //
  // Which is `-NaN` if the exponent is non-zero, and `-Inf` if the exponent is
  // zero (e.g. the input was `0.0f` or denormal).

  // Some useful constants.
  XNN_SIMD_CONST_F32(sign_mask, -0.0f);
  XNN_SIMD_CONST_U32(sign_and_exp_mask, 0xff800000);
  XNN_SIMD_CONST_F32(bias_256, 256.0f);
  XNN_SIMD_CONST_F32(bias_383, 383.0f);

  // If `a` is `0.0f`, flip its sign bit so that we return `-Inf`.
  a = xnn_or_f32(xnn_and_f32(xnn_cmpeq_f32(a, xnn_zero_f32()), sign_mask), a);

  // Extract the exponent and shift the exponent to the most significant bits of
  // the mantissa.
  const xnn_simd_f32_t exp =
      xnn_sra_f32(xnn_and_f32(a, sign_and_exp_mask), 8);

  // Add the shifted exponent to `256.0f` by copying its bits to the mantissa,
  // then subtract out `383.0f`, i.e. the original `256.0f` plus the `127`
  // exponent bias, resulting in the unbiased exponent.
  return xnn_sub_f32(xnn_or_f32(bias_256, exp), bias_383);
}

void xnn_f32_vlog_ukernel__fma3_rational_3_3_div_u16(
    size_t batch,
    const float* input,
    float* output,
    const union xnn_f32_default_params unused_params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);
  assert(xnn_simd_size_f32 == 8);

  // Some useful constants.
  XNN_SIMD_CONST_F32(vone, 1.0f);
  XNN_SIMD_CONST_F32(vln2, M_LN2);
  XNN_SIMD_CONST_U32(vmantissa_bits_mask, 0x007FFFFFUL);

  // Note that these two values are not _exactly_ `(float)M_SQRT2` and
  // `(float)M_SQRT1_2`, but are instead chosen such that their product is
  // exactly `1.0f` when evaluated in `float` precision.
  XNN_SIMD_CONST_F32(vsqrt2, 1.4142134190e+00);
  XNN_SIMD_CONST_F32(vsqrt1_2, 7.0710688829e-01);

  // The monomial coefficients of the numerator polynomial.
  // XNN_SIMD_CONST_F32(valpha_0, 0.0f);
  // XNN_SIMD_CONST_F32(valpha_1, 1.0f);
  // XNN_SIMD_CONST_F32(valpha_2, 1.0f);
  XNN_SIMD_CONST_F32(valpha_3, 1.824996918440e-01);

  // The monomial coefficients of the denominator polynomial.
  // XNN_SIMD_CONST_F32(vbeta_0, 1.0f);
  XNN_SIMD_CONST_F32(vbeta_1, 1.5f);
  XNN_SIMD_CONST_F32(vbeta_2, 0.599170029163);
  XNN_SIMD_CONST_F32(vbeta_3, 0.049584995955);


  for (; batch >= 16 * sizeof(float); batch -= 16 * sizeof(float)) {
    xnn_simd_f32_t vx_0 = xnn_loadu_f32(input);
    xnn_simd_f32_t vx_1 = xnn_loadu_f32(input + 1 * xnn_simd_size_f32);
    input += 16;

    // Scale `x` with `sqrt(2)` so that the exponent is rounded up.
    vx_0 = xnn_mul_f32(vx_0, vsqrt2);
    vx_1 = xnn_mul_f32(vx_1, vsqrt2);

    // Extract the exponent.
    const xnn_simd_f32_t vexp_0 = xnn_signed_getexp_f32(vx_0);
    const xnn_simd_f32_t vexp_1 = xnn_signed_getexp_f32(vx_1);

    // Normalize `x` to an exponent of zero.
    vx_0 = xnn_or_f32(xnn_and_f32(vx_0, vmantissa_bits_mask), vone);
    vx_1 = xnn_or_f32(xnn_and_f32(vx_1, vmantissa_bits_mask), vone);

    // Scale `x` back with `1/sqrt(2)` to move its range from `[1.0, 2.0)` to
    // `[sqrt(1/2), sqrt(2))`, and further subtract `1.0` so that it is around
    // zero, i.e. `[sqrt(1/2) - 1, sqrt(2) - 1)`, or `[−0.29289, 0.4142136)`.
    vx_0 = xnn_sub_f32(xnn_mul_f32(vx_0, vsqrt1_2), vone);
    vx_1 = xnn_sub_f32(xnn_mul_f32(vx_1, vsqrt1_2), vone);

    // In the following, we use a 3/2-degree rational polynomial to
    // approximate the (shifted) `log(x + 1)` on the (shifted) interval
    // `[sqrt(1/2) - 1, sqrt(2) - 1)`. The shifted interval is chosen so that
    // `f(0) = 0`.

    // Evaluate the numerator polynomial p.
    xnn_simd_f32_t vp_0 = xnn_fmadd_f32(vx_0, valpha_3, vone);
    xnn_simd_f32_t vp_1 = xnn_fmadd_f32(vx_1, valpha_3, vone);
    vp_0 = xnn_fmadd_f32(vx_0, vp_0, vone);
    vp_1 = xnn_fmadd_f32(vx_1, vp_1, vone);
    vp_0 = xnn_mul_f32(vx_0, vp_0);
    vp_1 = xnn_mul_f32(vx_1, vp_1);

    // Evaluate the denominator polynomial q.
    xnn_simd_f32_t vq_0 = xnn_fmadd_f32(vx_0, vbeta_3, vbeta_2);
    xnn_simd_f32_t vq_1 = xnn_fmadd_f32(vx_1, vbeta_3, vbeta_2);
    vq_0 = xnn_fmadd_f32(vx_0, vq_0, vbeta_1);
    vq_1 = xnn_fmadd_f32(vx_1, vq_1, vbeta_1);
    vq_0 = xnn_fmadd_f32(vx_0, vq_0, vone);
    vq_1 = xnn_fmadd_f32(vx_1, vq_1, vone);

    // Divide the numerator by the denominator.
    xnn_simd_f32_t vy_0 = xnn_div_f32(vp_0, vq_0);
    xnn_simd_f32_t vy_1 = xnn_div_f32(vp_1, vq_1);

    // Put it all together, i.e. `log(x) = `log(2)*exp + y`.
    vy_0 = xnn_fmadd_f32(vexp_0, vln2, vy_0);
    vy_1 = xnn_fmadd_f32(vexp_1, vln2, vy_1);

    xnn_storeu_f32(output, vy_0);
    xnn_storeu_f32(output + 1 * xnn_simd_size_f32, vy_1);
    output += 16;
  }
  for (; batch >= xnn_simd_bytes_f32; batch -= xnn_simd_bytes_f32) {
    xnn_simd_f32_t vx = xnn_loadu_f32(input);
    input += xnn_simd_size_f32;

    // Scale `x` with `sqrt(2)` so that the exponent is rounded up.
    vx = xnn_mul_f32(vx, vsqrt2);

    // Extract the exponent.
    const xnn_simd_f32_t vexp = xnn_signed_getexp_f32(vx);

    // Normalize `x` to an exponent of zero.
    vx = xnn_or_f32(xnn_and_f32(vx, vmantissa_bits_mask), vone);

    // Scale `x` back with `1/sqrt(2)` to move its range from `[1.0, 2.0)` to
    // `[sqrt(1/2), sqrt(2))`, and further subtract `1.0` so that it is around
    // zero, i.e. `[sqrt(1/2) - 1, sqrt(2) - 1)`, or `[−0.29289, 0.4142136)`.
    vx = xnn_sub_f32(xnn_mul_f32(vx, vsqrt1_2), vone);

    // In the following, we use a 3/2-degree rational polynomial to
    // approximate the (shifted) `log(x + 1)` on the (shifted) interval
    // `[sqrt(1/2) - 1, sqrt(2) - 1)`. The shifted interval is chosen so that
    // `f(0) = 0`.

    // Evaluate the numerator polynomial p.
    xnn_simd_f32_t vp = xnn_fmadd_f32(vx, valpha_3, vone);
    vp = xnn_fmadd_f32(vx, vp, vone);
    vp = xnn_mul_f32(vx, vp);

    // Evaluate the denominator polynomial q.
    xnn_simd_f32_t vq = xnn_fmadd_f32(vx, vbeta_3, vbeta_2);
    vq = xnn_fmadd_f32(vx, vq, vbeta_1);
    vq = xnn_fmadd_f32(vx, vq, vone);

    // Divide the numerator by the denominator.
    xnn_simd_f32_t vy =  xnn_div_f32(vp, vq);

    // Put it all together, i.e. `log(x) = `log(2)*exp + y`.
    vy = xnn_fmadd_f32(vexp, vln2, vy);

    xnn_storeu_f32(output, vy);
    output += xnn_simd_size_f32;
  }
  if XNN_UNLIKELY(batch != 0) {
    xnn_simd_f32_t vx = xnn_load_tail_f32(input, batch >> XNN_LOG2_SIZEOF_FLOAT);

    // See the loop above for comments.
    vx = xnn_mul_f32(vx, vsqrt2);
    const xnn_simd_f32_t vexp = xnn_signed_getexp_f32(vx);
    vx = xnn_or_f32(xnn_and_f32(vx, vmantissa_bits_mask), vone);
    vx = xnn_sub_f32(xnn_mul_f32(vx, vsqrt1_2), vone);
    xnn_simd_f32_t vp = xnn_fmadd_f32(vx, valpha_3, vone);
    vp = xnn_fmadd_f32(vx, vp, vone);
    vp = xnn_mul_f32(vx, vp);
    xnn_simd_f32_t vq = xnn_fmadd_f32(vx, vbeta_3, vbeta_2);
    vq = xnn_fmadd_f32(vx, vq, vbeta_1);
    vq = xnn_fmadd_f32(vx, vq, vone);
    xnn_simd_f32_t vy =  xnn_div_f32(vp, vq);
    vy = xnn_fmadd_f32(vexp, vln2, vy);

    xnn_store_tail_f32(output, vy, batch >> XNN_LOG2_SIZEOF_FLOAT);
  }
}

void xnn_f32_vtanh_ukernel__fma3_rational_9_6_div_u16(
    size_t batch,
    const float* input,
    float* output,
    const union xnn_f32_tanh_params unused_params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);
  assert(xnn_simd_size_f32 == 8);

  // Cap the inputs to this value as `tanh(x)` will always be `+/-1.0f` beyond
  // this point. This value is chosen as the first floating point number as of
  // which the interpolation returns 1.0f.
  #if XNN_SIMD_HAS_NATIVE_FMA
    XNN_SIMD_CONST_F32(vmax_x, 7.646893501282f);
    XNN_SIMD_CONST_F32(vmin_x, -7.646893501282f);
  #else
    XNN_SIMD_CONST_F32(vmax_x, 7.623543739319f);
    XNN_SIMD_CONST_F32(vmin_x, -7.623543739319f);
  #endif  // XNN_SIMD_HAS_NATIVE_FMA

  // The monomial coefficients of the numerator polynomial (odd).
  XNN_SIMD_CONST_F32(valpha_1, -9.022999554873e-03f);
  XNN_SIMD_CONST_F32(valpha_3, -1.146968104877e-03f);
  XNN_SIMD_CONST_F32(valpha_5, -2.432360815874e-05f);
  XNN_SIMD_CONST_F32(valpha_7, -6.458659385089e-08f);
  XNN_SIMD_CONST_F32(valpha_9, 5.535878699892e-11f);

  // The monomial coefficients of the denominator polynomial (even).
  XNN_SIMD_CONST_F32(vbeta_0, -9.023001417518e-03f);
  XNN_SIMD_CONST_F32(vbeta_2, -4.154618829489e-03f);
  XNN_SIMD_CONST_F32(vbeta_4, -2.061512641376e-04f);
  XNN_SIMD_CONST_F32(vbeta_6, -1.774490101525e-06f);


  for (; batch >= 16 * sizeof(float); batch -= 16 * sizeof(float)) {
    xnn_simd_f32_t vx_0 = xnn_loadu_f32(input);
    xnn_simd_f32_t vx_1 = xnn_loadu_f32(input + 1 * xnn_simd_size_f32);
    input += 16;

    // Clamp the inputs to the interpolation range.
    vx_0 = xnn_min_f32(vmax_x, vx_0);
    vx_1 = xnn_min_f32(vmax_x, vx_1);
    vx_0 = xnn_max_f32(vmin_x, vx_0);
    vx_1 = xnn_max_f32(vmin_x, vx_1);

    // Since the polynomials are odd/even, we need x^2.
    const xnn_simd_f32_t vx2_0 = xnn_mul_f32(vx_0, vx_0);
    const xnn_simd_f32_t vx2_1 = xnn_mul_f32(vx_1, vx_1);

    // Evaluate the numerator polynomial p.
    xnn_simd_f32_t vp_0 = xnn_fmadd_f32(vx2_0, valpha_9, valpha_7);
    xnn_simd_f32_t vp_1 = xnn_fmadd_f32(vx2_1, valpha_9, valpha_7);
    vp_0 = xnn_fmadd_f32(vx2_0, vp_0, valpha_5);
    vp_1 = xnn_fmadd_f32(vx2_1, vp_1, valpha_5);
    vp_0 = xnn_fmadd_f32(vx2_0, vp_0, valpha_3);
    vp_1 = xnn_fmadd_f32(vx2_1, vp_1, valpha_3);
    vp_0 = xnn_fmadd_f32(vx2_0, vp_0, valpha_1);
    vp_1 = xnn_fmadd_f32(vx2_1, vp_1, valpha_1);
    vp_0 = xnn_mul_f32(vx_0, vp_0);
    vp_1 = xnn_mul_f32(vx_1, vp_1);

    // Evaluate the denominator polynomial q.
    xnn_simd_f32_t vq_0 = xnn_fmadd_f32(vx2_0, vbeta_6, vbeta_4);
    xnn_simd_f32_t vq_1 = xnn_fmadd_f32(vx2_1, vbeta_6, vbeta_4);
    vq_0 = xnn_fmadd_f32(vx2_0, vq_0, vbeta_2);
    vq_1 = xnn_fmadd_f32(vx2_1, vq_1, vbeta_2);
    vq_0 = xnn_fmadd_f32(vx2_0, vq_0, vbeta_0);
    vq_1 = xnn_fmadd_f32(vx2_1, vq_1, vbeta_0);

    // Divide the numerator by the denominator.
    const xnn_simd_f32_t vy_0 = xnn_div_f32(vp_0, vq_0);
    const xnn_simd_f32_t vy_1 = xnn_div_f32(vp_1, vq_1);

    xnn_storeu_f32(output, vy_0);
    xnn_storeu_f32(output + 1 * xnn_simd_size_f32, vy_1);
    output += 16;
  }
  for (; batch >= xnn_simd_bytes_f32; batch -= xnn_simd_bytes_f32) {
    xnn_simd_f32_t vx = xnn_loadu_f32(input);
    input += xnn_simd_size_f32;

    // Clamp the inputs to the interpolation range.
    vx = xnn_min_f32(vmax_x, vx);
    vx = xnn_max_f32(vmin_x, vx);

    // Since the polynomials are odd/even, we need x^2.
    const xnn_simd_f32_t vx2 = xnn_mul_f32(vx, vx);

    // Evaluate the numerator polynomial p.
    xnn_simd_f32_t vp = xnn_fmadd_f32(vx2, valpha_9, valpha_7);
    vp = xnn_fmadd_f32(vx2, vp, valpha_5);
    vp = xnn_fmadd_f32(vx2, vp, valpha_3);
    vp = xnn_fmadd_f32(vx2, vp, valpha_1);
    vp = xnn_mul_f32(vx, vp);

    // Evaluate the denominator polynomial q.
    xnn_simd_f32_t vq = xnn_fmadd_f32(vx2, vbeta_6, vbeta_4);
    vq = xnn_fmadd_f32(vx2, vq, vbeta_2);
    vq = xnn_fmadd_f32(vx2, vq, vbeta_0);

    // Divide the numerator by the denominator.
    const xnn_simd_f32_t vy =  xnn_div_f32(vp, vq);

    xnn_storeu_f32(output, vy);
    output += xnn_simd_size_f32;
  }
  if XNN_UNLIKELY(batch != 0) {
    xnn_simd_f32_t vx = xnn_load_tail_f32(input, batch >> XNN_LOG2_SIZEOF_FLOAT);

    // Clamp the inputs to the interpolation range.
    vx = xnn_min_f32(vmax_x, vx);
    vx = xnn_max_f32(vmin_x, vx);

    // Since the polynomials are odd/even, we need x^2.
    const xnn_simd_f32_t vx2 = xnn_mul_f32(vx, vx);

    // Evaluate the numerator polynomial p.
    xnn_simd_f32_t vp = xnn_fmadd_f32(vx2, valpha_9, valpha_7);
    vp = xnn_fmadd_f32(vx2, vp, valpha_5);
    vp = xnn_fmadd_f32(vx2, vp, valpha_3);
    vp = xnn_fmadd_f32(vx2, vp, valpha_1);
    vp = xnn_mul_f32(vx, vp);

    // Evaluate the denominator polynomial q.
    xnn_simd_f32_t vq = xnn_fmadd_f32(vx2, vbeta_6, vbeta_4);
    vq = xnn_fmadd_f32(vx2, vq, vbeta_2);
    vq = xnn_fmadd_f32(vx2, vq, vbeta_0);

    // Divide the numerator by the denominator.
    const xnn_simd_f32_t vy =  xnn_div_f32(vp, vq);

    xnn_store_tail_f32(output, vy, batch >> XNN_LOG2_SIZEOF_FLOAT);
  }
}
