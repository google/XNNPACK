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

#include <immintrin.h>

#include "xnnpack/avgpool.h"
#include "xnnpack/common.h"
#include "xnnpack/gavgpool.h"
#include "xnnpack/intrinsics-polyfill.h"
#include "xnnpack/math.h"
#include "xnnpack/maxpool.h"
#include "xnnpack/microparams.h"
#include "xnnpack/prelu.h"
#include "xnnpack/reduce.h"
#include "xnnpack/unaligned.h"
#include "xnnpack/vbinary.h"
#include "xnnpack/vcvt.h"
#include "xnnpack/vunary.h"


void xnn_f16_avgpool_minmax_ukernel_9p8x__f16c_c8(
    size_t output_pixels,
    size_t kernel_elements,
    size_t channels,
    const void** input,
    size_t input_offset,
    const void* zero,
    void* buffer,
    void* output,
    size_t input_increment,
    size_t output_increment,
    const union xnn_f16_scaleminmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(output_pixels != 0);
  assert(kernel_elements > 9);
  assert(channels != 0);

  const __m256 vscale = _mm256_set1_ps(params->avx.scale);
  const __m256 vmin = _mm256_set1_ps(params->avx.min);
  const __m256 vmax = _mm256_set1_ps(params->avx.max);
  XNN_FORCE_REALIZATION(vscale);
  XNN_FORCE_REALIZATION(vmin);
  XNN_FORCE_REALIZATION(vmax);

  uint16_t* o = (uint16_t*) output;
  do {
    {
      const uint16_t* i0 = *input++;
      assert(i0 != NULL);
      if XNN_UNPREDICTABLE(i0 != zero) {
        i0 = (const uint16_t*) ((uintptr_t) i0 + input_offset);
      }
      const uint16_t* i1 = *input++;
      assert(i1 != NULL);
      if XNN_UNPREDICTABLE(i1 != zero) {
        i1 = (const uint16_t*) ((uintptr_t) i1 + input_offset);
      }
      const uint16_t* i2 = *input++;
      assert(i2 != NULL);
      if XNN_UNPREDICTABLE(i2 != zero) {
        i2 = (const uint16_t*) ((uintptr_t) i2 + input_offset);
      }
      const uint16_t* i3 = *input++;
      assert(i3 != NULL);
      if XNN_UNPREDICTABLE(i3 != zero) {
        i3 = (const uint16_t*) ((uintptr_t) i3 + input_offset);
      }
      const uint16_t* i4 = *input++;
      assert(i4 != NULL);
      if XNN_UNPREDICTABLE(i4 != zero) {
        i4 = (const uint16_t*) ((uintptr_t) i4 + input_offset);
      }
      const uint16_t* i5 = *input++;
      assert(i5 != NULL);
      if XNN_UNPREDICTABLE(i5 != zero) {
        i5 = (const uint16_t*) ((uintptr_t) i5 + input_offset);
      }
      const uint16_t* i6 = *input++;
      assert(i6 != NULL);
      if XNN_UNPREDICTABLE(i6 != zero) {
        i6 = (const uint16_t*) ((uintptr_t) i6 + input_offset);
      }
      const uint16_t* i7 = *input++;
      assert(i7 != NULL);
      if XNN_UNPREDICTABLE(i7 != zero) {
        i7 = (const uint16_t*) ((uintptr_t) i7 + input_offset);
      }
      const uint16_t* i8 = *input++;
      assert(i8 != NULL);
      if XNN_UNPREDICTABLE(i8 != zero) {
        i8 = (const uint16_t*) ((uintptr_t) i8 + input_offset);
      }

      uint16_t* b = (uint16_t*) buffer;
      for (size_t c = 0; c < channels; c += 8) {
        const __m256 vi0 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i0));
        i0 += 8;
        const __m256 vi1 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i1));
        i1 += 8;
        const __m256 vi2 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i2));
        i2 += 8;
        const __m256 vi3 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i3));
        i3 += 8;
        const __m256 vi4 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i4));
        i4 += 8;
        const __m256 vi5 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i5));
        i5 += 8;
        const __m256 vi6 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i6));
        i6 += 8;
        const __m256 vi7 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i7));
        i7 += 8;
        const __m256 vi8 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i8));
        i8 += 8;

        const __m256 vsum01 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_add_ps(vi0, vi1), _MM_FROUND_TO_NEAREST_INT));
        const __m256 vsum23 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_add_ps(vi2, vi3), _MM_FROUND_TO_NEAREST_INT));
        const __m256 vsum45 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_add_ps(vi4, vi5), _MM_FROUND_TO_NEAREST_INT));
        const __m256 vsum67 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_add_ps(vi6, vi7), _MM_FROUND_TO_NEAREST_INT));
        const __m256 vsum018 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_add_ps(vsum01, vi8), _MM_FROUND_TO_NEAREST_INT));
        const __m256 vsum2345 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_add_ps(vsum23, vsum45), _MM_FROUND_TO_NEAREST_INT));
        const __m256 vsum01678 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_add_ps(vsum018, vsum67), _MM_FROUND_TO_NEAREST_INT));
        const __m128i vsum = _mm256_cvtps_ph(_mm256_add_ps(vsum2345, vsum01678), _MM_FROUND_TO_NEAREST_INT);

        _mm_storeu_si128((__m128i*) b, vsum);
        b += 8;
      }
    }

    size_t k = kernel_elements;
    for (k -= 9; k > 8; k -= 8) {
      const uint16_t* i0 = (const uint16_t*) *input++;
      assert(i0 != NULL);
      if XNN_UNPREDICTABLE(i0 != zero) {
        i0 = (const uint16_t*) ((uintptr_t) i0 + input_offset);
      }
      const uint16_t* i1 = (const uint16_t*) *input++;
      assert(i1 != NULL);
      if XNN_UNPREDICTABLE(i1 != zero) {
        i1 = (const uint16_t*) ((uintptr_t) i1 + input_offset);
      }
      const uint16_t* i2 = (const uint16_t*) *input++;
      assert(i2 != NULL);
      if XNN_UNPREDICTABLE(i2 != zero) {
        i2 = (const uint16_t*) ((uintptr_t) i2 + input_offset);
      }
      const uint16_t* i3 = (const uint16_t*) *input++;
      assert(i3 != NULL);
      if XNN_UNPREDICTABLE(i3 != zero) {
        i3 = (const uint16_t*) ((uintptr_t) i3 + input_offset);
      }
      const uint16_t* i4 = (const uint16_t*) *input++;
      assert(i4 != NULL);
      if XNN_UNPREDICTABLE(i4 != zero) {
        i4 = (const uint16_t*) ((uintptr_t) i4 + input_offset);
      }
      const uint16_t* i5 = (const uint16_t*) *input++;
      assert(i5 != NULL);
      if XNN_UNPREDICTABLE(i5 != zero) {
        i5 = (const uint16_t*) ((uintptr_t) i5 + input_offset);
      }
      const uint16_t* i6 = (const uint16_t*) *input++;
      assert(i6 != NULL);
      if XNN_UNPREDICTABLE(i6 != zero) {
        i6 = (const uint16_t*) ((uintptr_t) i6 + input_offset);
      }
      const uint16_t* i7 = (const uint16_t*) *input++;
      assert(i7 != NULL);
      if XNN_UNPREDICTABLE(i7 != zero) {
        i7 = (const uint16_t*) ((uintptr_t) i7 + input_offset);
      }

      uint16_t* b = (uint16_t*) buffer;
      for (size_t c = 0; c < channels; c += 8) {
        const __m256 vi0 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i0));
        i0 += 8;
        const __m256 vi1 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i1));
        i1 += 8;
        const __m256 vi2 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i2));
        i2 += 8;
        const __m256 vi3 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i3));
        i3 += 8;
        const __m256 vi4 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i4));
        i4 += 8;
        const __m256 vi5 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i5));
        i5 += 8;
        const __m256 vi6 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i6));
        i6 += 8;
        const __m256 vi7 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i7));
        i7 += 8;
        const __m256 vacc = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) b));

        const __m256 vsum01 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_add_ps(vi0, vi1), _MM_FROUND_TO_NEAREST_INT));
        const __m256 vsum23 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_add_ps(vi2, vi3), _MM_FROUND_TO_NEAREST_INT));
        const __m256 vsum45 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_add_ps(vi4, vi5), _MM_FROUND_TO_NEAREST_INT));
        const __m256 vsum67 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_add_ps(vi6, vi7), _MM_FROUND_TO_NEAREST_INT));
        const __m256 vsum01a = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_add_ps(vsum01, vacc), _MM_FROUND_TO_NEAREST_INT));
        const __m256 vsum2345 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_add_ps(vsum23, vsum45), _MM_FROUND_TO_NEAREST_INT));
        const __m256 vsum0167a = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_add_ps(vsum01a, vsum67), _MM_FROUND_TO_NEAREST_INT));
        const __m128i vsum = _mm256_cvtps_ph(_mm256_add_ps(vsum2345, vsum0167a), _MM_FROUND_TO_NEAREST_INT);

        _mm_storeu_si128((__m128i*) b, vsum);
        b += 8;
      }
    }

    assert(k >= 1);
    {
      const uint16_t* i0 = (const uint16_t*) input[0];
      assert(i0 != NULL);
      const uint16_t* i1 = (const uint16_t*) input[1];
      const uint16_t* i2 = (const uint16_t*) input[2];
      const uint16_t* i3 = (const uint16_t*) input[3];
      const uint16_t* i4 = (const uint16_t*) input[4];
      const uint16_t* i5 = (const uint16_t*) input[5];
      const uint16_t* i6 = (const uint16_t*) input[6];
      const uint16_t* i7 = (const uint16_t*) input[7];
      input = (const void**) ((uintptr_t) input + input_increment);
      if (k < 2) {
        i1 = (const uint16_t*) zero;
      }
      assert(i1 != NULL);
      if (k <= 2) {
        i2 = (const uint16_t*) zero;
      }
      assert(i2 != NULL);
      if (k < 4) {
        i3 = (const uint16_t*) zero;
      }
      assert(i3 != NULL);
      if (k <= 4) {
        i4 = (const uint16_t*) zero;
      }
      assert(i4 != NULL);
      if (k < 6) {
        i5 = (const uint16_t*) zero;
      }
      assert(i5 != NULL);
      if (k <= 6) {
        i6 = (const uint16_t*) zero;
      }
      assert(i6 != NULL);
      if (k < 8) {
        i7 = (const uint16_t*) zero;
      }
      assert(i7 != NULL);
      if XNN_UNPREDICTABLE(i0 != zero) {
        i0 = (const uint16_t*) ((uintptr_t) i0 + input_offset);
      }
      if XNN_UNPREDICTABLE(i1 != zero) {
        i1 = (const uint16_t*) ((uintptr_t) i1 + input_offset);
      }
      if XNN_UNPREDICTABLE(i2 != zero) {
        i2 = (const uint16_t*) ((uintptr_t) i2 + input_offset);
      }
      if XNN_UNPREDICTABLE(i3 != zero) {
        i3 = (const uint16_t*) ((uintptr_t) i3 + input_offset);
      }
      if XNN_UNPREDICTABLE(i4 != zero) {
        i4 = (const uint16_t*) ((uintptr_t) i4 + input_offset);
      }
      if XNN_UNPREDICTABLE(i5 != zero) {
        i5 = (const uint16_t*) ((uintptr_t) i5 + input_offset);
      }
      if XNN_UNPREDICTABLE(i6 != zero) {
        i6 = (const uint16_t*) ((uintptr_t) i6 + input_offset);
      }
      if XNN_UNPREDICTABLE(i7 != zero) {
        i7 = (const uint16_t*) ((uintptr_t) i7 + input_offset);
      }

      size_t c = channels;
      uint16_t* b = (uint16_t*) buffer;
      while (c >= 8) {
        const __m256 vi0 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i0));
        i0 += 8;
        const __m256 vi1 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i1));
        i1 += 8;
        const __m256 vi2 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i2));
        i2 += 8;
        const __m256 vi3 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i3));
        i3 += 8;
        const __m256 vi4 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i4));
        i4 += 8;
        const __m256 vi5 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i5));
        i5 += 8;
        const __m256 vi6 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i6));
        i6 += 8;
        const __m256 vi7 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i7));
        i7 += 8;
        const __m256 vacc = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) b));
        b += 8;

        const __m256 vsum01 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_add_ps(vi0, vi1), _MM_FROUND_TO_NEAREST_INT));
        const __m256 vsum23 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_add_ps(vi2, vi3), _MM_FROUND_TO_NEAREST_INT));
        const __m256 vsum45 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_add_ps(vi4, vi5), _MM_FROUND_TO_NEAREST_INT));
        const __m256 vsum67 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_add_ps(vi6, vi7), _MM_FROUND_TO_NEAREST_INT));
        const __m256 vsum01a = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_add_ps(vsum01, vacc), _MM_FROUND_TO_NEAREST_INT));
        const __m256 vsum2345 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_add_ps(vsum23, vsum45), _MM_FROUND_TO_NEAREST_INT));
        const __m256 vsum0167a = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_add_ps(vsum01a, vsum67), _MM_FROUND_TO_NEAREST_INT));
        const __m256 vsum = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_add_ps(vsum2345, vsum0167a), _MM_FROUND_TO_NEAREST_INT));

        __m256 vout = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_mul_ps(vsum, vscale), _MM_FROUND_TO_NEAREST_INT));
        vout = _mm256_max_ps(vout, vmin);
        vout = _mm256_min_ps(vout, vmax);

        _mm_storeu_si128((__m128i*) o, _mm256_cvtps_ph(vout, _MM_FROUND_TO_NEAREST_INT));
        o += 8;

        c -= 8;
      }
      if (c != 0) {
        const __m256 vi0 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i0));
        const __m256 vi1 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i1));
        const __m256 vi2 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i2));
        const __m256 vi3 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i3));
        const __m256 vi4 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i4));
        const __m256 vi5 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i5));
        const __m256 vi6 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i6));
        const __m256 vi7 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i7));
        const __m256 vacc = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) b));

        const __m256 vsum01 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_add_ps(vi0, vi1), _MM_FROUND_TO_NEAREST_INT));
        const __m256 vsum23 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_add_ps(vi2, vi3), _MM_FROUND_TO_NEAREST_INT));
        const __m256 vsum45 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_add_ps(vi4, vi5), _MM_FROUND_TO_NEAREST_INT));
        const __m256 vsum67 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_add_ps(vi6, vi7), _MM_FROUND_TO_NEAREST_INT));
        const __m256 vsum01a = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_add_ps(vsum01, vacc), _MM_FROUND_TO_NEAREST_INT));
        const __m256 vsum2345 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_add_ps(vsum23, vsum45), _MM_FROUND_TO_NEAREST_INT));
        const __m256 vsum0167a = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_add_ps(vsum01a, vsum67), _MM_FROUND_TO_NEAREST_INT));
        const __m256 vsum = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_add_ps(vsum2345, vsum0167a), _MM_FROUND_TO_NEAREST_INT));

        __m256 vout = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_mul_ps(vsum, vscale), _MM_FROUND_TO_NEAREST_INT));
        vout = _mm256_max_ps(vout, vmin);
        vout = _mm256_min_ps(vout, vmax);

        __m128i vh = _mm256_cvtps_ph(vout, _MM_FROUND_TO_NEAREST_INT);
        if (c & 4) {
          _mm_storel_epi64((__m128i*) o, vh);
          vh = _mm_unpackhi_epi64(vh, vh);
          o += 4;
        }
        if (c & 2) {
          _mm_storeu_si32(o, vh);
          vh = _mm_srli_epi64(vh, 32);
          o += 2;
        }
        if (c & 1) {
          *o = (uint16_t) _mm_extract_epi16(vh, 0);
          o += 1;
        }
      }
    }
    o = (uint16_t*) ((uintptr_t) o + output_increment);
  } while (--output_pixels != 0);
}

void xnn_f16_avgpool_minmax_ukernel_9x__f16c_c8(
    size_t output_pixels,
    size_t kernel_elements,
    size_t channels,
    const void** input,
    size_t input_offset,
    const void* zero,
    void* output,
    size_t input_increment,
    size_t output_increment,
    const union xnn_f16_scaleminmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(output_pixels != 0);
  assert(kernel_elements != 0);
  assert(kernel_elements <= 9);
  assert(channels != 0);

  const __m256 vscale = _mm256_set1_ps(params->avx.scale);
  const __m256 vmin = _mm256_set1_ps(params->avx.min);
  const __m256 vmax = _mm256_set1_ps(params->avx.max);
  XNN_FORCE_REALIZATION(vscale);
  XNN_FORCE_REALIZATION(vmin);
  XNN_FORCE_REALIZATION(vmax);

  uint16_t* o = (uint16_t*) output;
  do {
    const uint16_t* i0 = (const uint16_t*) input[0];
    assert(i0 != NULL);
    const uint16_t* i1 = (const uint16_t*) input[1];
    const uint16_t* i2 = (const uint16_t*) input[2];
    const uint16_t* i3 = (const uint16_t*) input[3];
    const uint16_t* i4 = (const uint16_t*) input[4];
    const uint16_t* i5 = (const uint16_t*) input[5];
    const uint16_t* i6 = (const uint16_t*) input[6];
    const uint16_t* i7 = (const uint16_t*) input[7];
    const uint16_t* i8 = (const uint16_t*) input[8];
    input = (const void**) ((uintptr_t) input + input_increment);
    if (kernel_elements < 2) {
      i1 = (const uint16_t*) zero;
    }
    assert(i1 != NULL);
    if (kernel_elements <= 2) {
      i2 = (const uint16_t*) zero;
    }
    assert(i2 != NULL);
    if (kernel_elements < 4) {
      i3 = (const uint16_t*) zero;
    }
    assert(i3 != NULL);
    if (kernel_elements <= 4) {
      i4 = (const uint16_t*) zero;
    }
    assert(i4 != NULL);
    if (kernel_elements < 6) {
      i5 = (const uint16_t*) zero;
    }
    assert(i5 != NULL);
    if (kernel_elements <= 6) {
      i6 = (const uint16_t*) zero;
    }
    assert(i6 != NULL);
    if (kernel_elements < 8) {
      i7 = (const uint16_t*) zero;
    }
    assert(i7 != NULL);
    if (kernel_elements <= 8) {
      i8 = (const uint16_t*) zero;
    }
    assert(i8 != NULL);
    if XNN_UNPREDICTABLE(i0 != zero) {
      i0 = (const uint16_t*) ((uintptr_t) i0 + input_offset);
    }
    if XNN_UNPREDICTABLE(i1 != zero) {
      i1 = (const uint16_t*) ((uintptr_t) i1 + input_offset);
    }
    if XNN_UNPREDICTABLE(i2 != zero) {
      i2 = (const uint16_t*) ((uintptr_t) i2 + input_offset);
    }
    if XNN_UNPREDICTABLE(i3 != zero) {
      i3 = (const uint16_t*) ((uintptr_t) i3 + input_offset);
    }
    if XNN_UNPREDICTABLE(i4 != zero) {
      i4 = (const uint16_t*) ((uintptr_t) i4 + input_offset);
    }
    if XNN_UNPREDICTABLE(i5 != zero) {
      i5 = (const uint16_t*) ((uintptr_t) i5 + input_offset);
    }
    if XNN_UNPREDICTABLE(i6 != zero) {
      i6 = (const uint16_t*) ((uintptr_t) i6 + input_offset);
    }
    if XNN_UNPREDICTABLE(i7 != zero) {
      i7 = (const uint16_t*) ((uintptr_t) i7 + input_offset);
    }
    if XNN_UNPREDICTABLE(i8 != zero) {
      i8 = (const uint16_t*) ((uintptr_t) i8 + input_offset);
    }

    size_t c = channels;
    while (c >= 8) {
      const __m256 vi0 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i0));
      i0 += 8;
      const __m256 vi1 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i1));
      i1 += 8;
      const __m256 vi2 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i2));
      i2 += 8;
      const __m256 vi3 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i3));
      i3 += 8;
      const __m256 vi4 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i4));
      i4 += 8;
      const __m256 vi5 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i5));
      i5 += 8;
      const __m256 vi6 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i6));
      i6 += 8;
      const __m256 vi7 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i7));
      i7 += 8;
      const __m256 vi8 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i8));
      i8 += 8;

      const __m256 vsum01 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_add_ps(vi0, vi1), _MM_FROUND_TO_NEAREST_INT));
      const __m256 vsum23 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_add_ps(vi2, vi3), _MM_FROUND_TO_NEAREST_INT));
      const __m256 vsum45 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_add_ps(vi4, vi5), _MM_FROUND_TO_NEAREST_INT));
      const __m256 vsum67 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_add_ps(vi6, vi7), _MM_FROUND_TO_NEAREST_INT));
      const __m256 vsum018 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_add_ps(vsum01, vi8), _MM_FROUND_TO_NEAREST_INT));
      const __m256 vsum2345 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_add_ps(vsum23, vsum45), _MM_FROUND_TO_NEAREST_INT));
      const __m256 vsum01678 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_add_ps(vsum018, vsum67), _MM_FROUND_TO_NEAREST_INT));
      const __m256 vsum = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_add_ps(vsum2345, vsum01678), _MM_FROUND_TO_NEAREST_INT));

      __m256 vout = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_mul_ps(vsum, vscale), _MM_FROUND_TO_NEAREST_INT));
      vout = _mm256_max_ps(vout, vmin);
      vout = _mm256_min_ps(vout, vmax);

      _mm_storeu_si128((__m128i*) o, _mm256_cvtps_ph(vout, _MM_FROUND_TO_NEAREST_INT));
      o += 8;

      c -= 8;
    }
    if (c != 0) {
      const __m256 vi0 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i0));
      const __m256 vi1 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i1));
      const __m256 vi2 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i2));
      const __m256 vi3 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i3));
      const __m256 vi4 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i4));
      const __m256 vi5 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i5));
      const __m256 vi6 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i6));
      const __m256 vi7 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i7));
      const __m256 vi8 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i8));

      const __m256 vsum01 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_add_ps(vi0, vi1), _MM_FROUND_TO_NEAREST_INT));
      const __m256 vsum23 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_add_ps(vi2, vi3), _MM_FROUND_TO_NEAREST_INT));
      const __m256 vsum45 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_add_ps(vi4, vi5), _MM_FROUND_TO_NEAREST_INT));
      const __m256 vsum67 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_add_ps(vi6, vi7), _MM_FROUND_TO_NEAREST_INT));
      const __m256 vsum018 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_add_ps(vsum01, vi8), _MM_FROUND_TO_NEAREST_INT));
      const __m256 vsum2345 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_add_ps(vsum23, vsum45), _MM_FROUND_TO_NEAREST_INT));
      const __m256 vsum01678 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_add_ps(vsum018, vsum67), _MM_FROUND_TO_NEAREST_INT));
      const __m256 vsum = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_add_ps(vsum2345, vsum01678), _MM_FROUND_TO_NEAREST_INT));

      __m256 vout = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_mul_ps(vsum, vscale), _MM_FROUND_TO_NEAREST_INT));
      vout = _mm256_max_ps(vout, vmin);
      vout = _mm256_min_ps(vout, vmax);

      __m128i vh = _mm256_cvtps_ph(vout, _MM_FROUND_TO_NEAREST_INT);
      if (c & 4) {
        _mm_storel_epi64((__m128i*) o, vh);
        vh = _mm_unpackhi_epi64(vh, vh);
        o += 4;
      }
      if (c & 2) {
        _mm_storeu_si32(o, vh);
        vh = _mm_srli_epi64(vh, 32);
        o += 2;
      }
      if (c & 1) {
        *o = (uint16_t) _mm_extract_epi16(vh, 0);
        o += 1;
      }
    }
    o = (uint16_t*) ((uintptr_t) o + output_increment);
  } while (--output_pixels != 0);
}

void xnn_f16_f32_vcvt_ukernel__f16c_u16(
    size_t batch,
    const void* input,
    float* output,
    const void* params) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(uint16_t) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const uint16_t* i = (const uint16_t*) input;
  for (; batch >= 16 * sizeof(uint16_t); batch -= 16 * sizeof(uint16_t)) {
    const __m256 vacc0 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i));
    const __m256 vacc1 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i + 8)));
    i += 16;

    _mm256_storeu_ps(output, vacc0);
    _mm256_storeu_ps(output + 8, vacc1);
    output += 16;
  }
  for (; batch >= 8 * sizeof(uint16_t); batch -= 8 * sizeof(uint16_t)) {
    const __m256 vacc = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i));
    i += 8;

    _mm256_storeu_ps(output, vacc);
    output += 8;
  }
  if XNN_UNLIKELY(batch != 0) {
    assert(batch >= 1 * sizeof(uint16_t));
    assert(batch <= 7 * sizeof(uint16_t));
    const __m256 vacc = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i));

    __m128 vacc_lo = _mm256_castps256_ps128(vacc);
    if (batch & (4 * sizeof(uint16_t))) {
      _mm_storeu_ps(output, vacc_lo);
      vacc_lo = _mm256_extractf128_ps(vacc, 1);
      output += 4;
    }
    if (batch & (2 * sizeof(uint16_t))) {
      _mm_storel_pi((__m64*) output, vacc_lo);
      vacc_lo = _mm_movehl_ps(vacc_lo, vacc_lo);
      output += 2;
    }
    if (batch & (1 * sizeof(uint16_t))) {
      _mm_store_ss(output, vacc_lo);
    }
  }
}

void xnn_f16_f32acc_rdsum_ukernel_7p7x__f16c_c32(
    size_t rows,
    size_t channels,
    const void* input,
    size_t input_stride,
    const void* zero,
    float* output,
    const union xnn_f16_f32acc_scale_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(rows != 0);
  assert(channels != 0);
  assert(input != NULL);
  assert(output != NULL);

  const __m256 vscale = _mm256_set1_ps(params->scale);

  size_t input_increment = 7 * input_stride;
  for (; channels >= 32; channels -= 32) {
    const uint16_t* i0 = input;
    const uint16_t* i1 = (const uint16_t*) ((uintptr_t) input + 1 * input_stride);
    const uint16_t* i2 = (const uint16_t*) ((uintptr_t) input + 2 * input_stride);
    const uint16_t* i3 = (const uint16_t*) ((uintptr_t) input + 3 * input_stride);
    const uint16_t* i4 = (const uint16_t*) ((uintptr_t) input + 4 * input_stride);
    const uint16_t* i5 = (const uint16_t*) ((uintptr_t) input + 5 * input_stride);
    const uint16_t* i6 = (const uint16_t*) ((uintptr_t) input + 6 * input_stride);

    __m256 vacc0 = _mm256_setzero_ps();
    __m256 vacc1 = _mm256_setzero_ps();
    __m256 vacc2 = _mm256_setzero_ps();
    __m256 vacc3 = _mm256_setzero_ps();

    for (int r = rows; r > 0; r -= 7) {
      if XNN_UNPREDICTABLE(r < 2) {
        i1 = zero;
      }
      if XNN_UNPREDICTABLE(r <= 2) {
        i2 = zero;
      }
      if XNN_UNPREDICTABLE(r < 4) {
        i3 = zero;
      }
      if XNN_UNPREDICTABLE(r <= 4) {
        i4 = zero;
      }
      if XNN_UNPREDICTABLE(r < 6) {
        i5 = zero;
      }
      if XNN_UNPREDICTABLE(r <= 6) {
        i6 = zero;
      }
      __m256 vin0;
      __m256 vin1;
      __m256 vin2;
      __m256 vin3;
      vin0 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (&i0[0])));
      vin1 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (&i0[8])));
      vin2 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (&i0[16])));
      vin3 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (&i0[24])));
      vacc0 = _mm256_add_ps(vin0, vacc0);
      vacc1 = _mm256_add_ps(vin1, vacc1);
      vacc2 = _mm256_add_ps(vin2, vacc2);
      vacc3 = _mm256_add_ps(vin3, vacc3);
      vin0 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (&i1[0])));
      vin1 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (&i1[8])));
      vin2 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (&i1[16])));
      vin3 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (&i1[24])));
      vacc0 = _mm256_add_ps(vin0, vacc0);
      vacc1 = _mm256_add_ps(vin1, vacc1);
      vacc2 = _mm256_add_ps(vin2, vacc2);
      vacc3 = _mm256_add_ps(vin3, vacc3);
      vin0 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (&i2[0])));
      vin1 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (&i2[8])));
      vin2 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (&i2[16])));
      vin3 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (&i2[24])));
      vacc0 = _mm256_add_ps(vin0, vacc0);
      vacc1 = _mm256_add_ps(vin1, vacc1);
      vacc2 = _mm256_add_ps(vin2, vacc2);
      vacc3 = _mm256_add_ps(vin3, vacc3);
      vin0 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (&i3[0])));
      vin1 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (&i3[8])));
      vin2 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (&i3[16])));
      vin3 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (&i3[24])));
      vacc0 = _mm256_add_ps(vin0, vacc0);
      vacc1 = _mm256_add_ps(vin1, vacc1);
      vacc2 = _mm256_add_ps(vin2, vacc2);
      vacc3 = _mm256_add_ps(vin3, vacc3);
      vin0 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (&i4[0])));
      vin1 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (&i4[8])));
      vin2 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (&i4[16])));
      vin3 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (&i4[24])));
      vacc0 = _mm256_add_ps(vin0, vacc0);
      vacc1 = _mm256_add_ps(vin1, vacc1);
      vacc2 = _mm256_add_ps(vin2, vacc2);
      vacc3 = _mm256_add_ps(vin3, vacc3);
      vin0 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (&i5[0])));
      vin1 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (&i5[8])));
      vin2 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (&i5[16])));
      vin3 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (&i5[24])));
      vacc0 = _mm256_add_ps(vin0, vacc0);
      vacc1 = _mm256_add_ps(vin1, vacc1);
      vacc2 = _mm256_add_ps(vin2, vacc2);
      vacc3 = _mm256_add_ps(vin3, vacc3);
      vin0 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (&i6[0])));
      vin1 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (&i6[8])));
      vin2 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (&i6[16])));
      vin3 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (&i6[24])));
      vacc0 = _mm256_add_ps(vin0, vacc0);
      vacc1 = _mm256_add_ps(vin1, vacc1);
      vacc2 = _mm256_add_ps(vin2, vacc2);
      vacc3 = _mm256_add_ps(vin3, vacc3);
      i0 = (const uint16_t*) ((uintptr_t) i0 + input_increment);
      i1 = (const uint16_t*) ((uintptr_t) i1 + input_increment);
      i2 = (const uint16_t*) ((uintptr_t) i2 + input_increment);
      i3 = (const uint16_t*) ((uintptr_t) i3 + input_increment);
      i4 = (const uint16_t*) ((uintptr_t) i4 + input_increment);
      i5 = (const uint16_t*) ((uintptr_t) i5 + input_increment);
      i6 = (const uint16_t*) ((uintptr_t) i6 + input_increment);
    }
    vacc0 = _mm256_mul_ps(vacc0, vscale);
    vacc1 = _mm256_mul_ps(vacc1, vscale);
    vacc2 = _mm256_mul_ps(vacc2, vscale);
    vacc3 = _mm256_mul_ps(vacc3, vscale);

    const float* o = output;
    __m256 vo0 = _mm256_loadu_ps(o); o = (const void*) ((uintptr_t) o + 8 * sizeof(float));
    __m256 vo1 = _mm256_loadu_ps(o); o = (const void*) ((uintptr_t) o + 8 * sizeof(float));
    __m256 vo2 = _mm256_loadu_ps(o); o = (const void*) ((uintptr_t) o + 8 * sizeof(float));
    __m256 vo3 = _mm256_loadu_ps(o); o = (const void*) ((uintptr_t) o + 8 * sizeof(float));
    vacc0 = _mm256_add_ps(vo0, vacc0);
    vacc1 = _mm256_add_ps(vo1, vacc1);
    vacc2 = _mm256_add_ps(vo2, vacc2);
    vacc3 = _mm256_add_ps(vo3, vacc3);
    _mm256_storeu_ps(output, vacc0); output = (void*) ((uintptr_t) output + 8 * sizeof(float));
    _mm256_storeu_ps(output, vacc1); output = (void*) ((uintptr_t) output + 8 * sizeof(float));
    _mm256_storeu_ps(output, vacc2); output = (void*) ((uintptr_t) output + 8 * sizeof(float));
    _mm256_storeu_ps(output, vacc3); output = (void*) ((uintptr_t) output + 8 * sizeof(float));

    input = (const uint16_t*) ((uintptr_t) input + 32 * sizeof(uint16_t));
  }
  if (channels != 0) {
    input_increment = 7 * input_stride;
    const uint16_t* i0 = input;
    const uint16_t* i1 = (const uint16_t*) ((uintptr_t) input + 1 * input_stride);
    const uint16_t* i2 = (const uint16_t*) ((uintptr_t) input + 2 * input_stride);
    const uint16_t* i3 = (const uint16_t*) ((uintptr_t) input + 3 * input_stride);
    const uint16_t* i4 = (const uint16_t*) ((uintptr_t) input + 4 * input_stride);
    const uint16_t* i5 = (const uint16_t*) ((uintptr_t) input + 5 * input_stride);
    const uint16_t* i6 = (const uint16_t*) ((uintptr_t) input + 6 * input_stride);
    __m256 vacc[4];
    vacc[0] = _mm256_setzero_ps();
    vacc[1] = _mm256_setzero_ps();
    vacc[2] = _mm256_setzero_ps();
    vacc[3] = _mm256_setzero_ps();

    const size_t num_full_chunks = channels >> 3;
    const size_t num_chunks = round_up_po2(channels, 8) >> 3;
    const size_t remainder = channels & 0x7;
    for (int r = rows; r > 0; r -= 7) {
      if XNN_UNPREDICTABLE(r < 2) {
        i1 = zero;
      }
      if XNN_UNPREDICTABLE(r <= 2) {
        i2 = zero;
      }
      if XNN_UNPREDICTABLE(r < 4) {
        i3 = zero;
      }
      if XNN_UNPREDICTABLE(r <= 4) {
        i4 = zero;
      }
      if XNN_UNPREDICTABLE(r < 6) {
        i5 = zero;
      }
      if XNN_UNPREDICTABLE(r <= 6) {
        i6 = zero;
      }
      for (int i = 0; i < num_full_chunks; ++i) {
        vacc[i] = _mm256_add_ps(_mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) &i0[i*8])), vacc[i]);
        vacc[i] = _mm256_add_ps(_mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) &i1[i*8])), vacc[i]);
        vacc[i] = _mm256_add_ps(_mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) &i2[i*8])), vacc[i]);
        vacc[i] = _mm256_add_ps(_mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) &i3[i*8])), vacc[i]);
        vacc[i] = _mm256_add_ps(_mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) &i4[i*8])), vacc[i]);
        vacc[i] = _mm256_add_ps(_mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) &i5[i*8])), vacc[i]);
        vacc[i] = _mm256_add_ps(_mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) &i6[i*8])), vacc[i]);
      }

      if (remainder) {
        vacc[num_full_chunks] = _mm256_add_ps(vacc[num_full_chunks],  _mm256_cvtph_ps(_mm_loadu_si128((const  __m128i*) &i0[num_full_chunks*8])));
        vacc[num_full_chunks] = _mm256_add_ps(vacc[num_full_chunks],  _mm256_cvtph_ps(_mm_loadu_si128((const  __m128i*) &i1[num_full_chunks*8])));
        vacc[num_full_chunks] = _mm256_add_ps(vacc[num_full_chunks],  _mm256_cvtph_ps(_mm_loadu_si128((const  __m128i*) &i2[num_full_chunks*8])));
        vacc[num_full_chunks] = _mm256_add_ps(vacc[num_full_chunks],  _mm256_cvtph_ps(_mm_loadu_si128((const  __m128i*) &i3[num_full_chunks*8])));
        vacc[num_full_chunks] = _mm256_add_ps(vacc[num_full_chunks],  _mm256_cvtph_ps(_mm_loadu_si128((const  __m128i*) &i4[num_full_chunks*8])));
        vacc[num_full_chunks] = _mm256_add_ps(vacc[num_full_chunks],  _mm256_cvtph_ps(_mm_loadu_si128((const  __m128i*) &i5[num_full_chunks*8])));
        vacc[num_full_chunks] = _mm256_add_ps(vacc[num_full_chunks],  _mm256_cvtph_ps(_mm_loadu_si128((const  __m128i*) &i6[num_full_chunks*8])));
      }
      i0 = (const uint16_t*) ((uintptr_t) i0 + input_increment);
      i1 = (const uint16_t*) ((uintptr_t) i1 + input_increment);
      i2 = (const uint16_t*) ((uintptr_t) i2 + input_increment);
      i3 = (const uint16_t*) ((uintptr_t) i3 + input_increment);
      i4 = (const uint16_t*) ((uintptr_t) i4 + input_increment);
      i5 = (const uint16_t*) ((uintptr_t) i5 + input_increment);
      i6 = (const uint16_t*) ((uintptr_t) i6 + input_increment);
    }
    for (size_t i = 0; i < num_chunks; ++i) {
      vacc[i] = _mm256_mul_ps(vacc[i], vscale);
    }

    __m256 vo[4];
    const float* o = output;
    for (int i = 0; i < num_full_chunks; ++i) {
      vo[i] = _mm256_loadu_ps(o); o = (const void*) ((uintptr_t) o + 8 * sizeof(float));
    }
    for (int i = 0; i < num_full_chunks; ++i) {
      vacc[i] = _mm256_add_ps(vo[i], vacc[i]);
    }
    for (int i = 0; i < num_full_chunks; ++i) {
      _mm256_storeu_ps(output, vacc[i]); output = (void*) ((uintptr_t) output + 8 * sizeof(float));
    }
    if (remainder) {
      __m256 vout = vacc[num_full_chunks];
      __m128 vout_low = _mm256_castps256_ps128(vout);
      if (channels & 4) {
        __m128 vo =  _mm_loadu_ps(output);
        vo = _mm_add_ps(vout_low, vo);
        _mm_storeu_ps(output, vo);
        vout_low  = _mm256_castps256_ps128(_mm256_permute2f128_ps(vout, vout, 1));
        output = (void*) ((uintptr_t) output + 4 * sizeof(float));
      }
      if (channels & 2) {
        __m128 vo =  _mm_castsi128_ps(_mm_loadl_epi64((__m128i*) output));
        vo = _mm_add_ps(vout_low, vo);
        _mm_storel_pi((__m64*) output, vo);
        vout_low = _mm_movehl_ps(vout_low, vout_low);
        output = (void*) ((uintptr_t) output + 2 * sizeof(float));
      }
      if (channels & 1) {
        __m128 vo = _mm_castsi128_ps(_mm_cvtsi32_si128(unaligned_load_s32(output)));
        vo = _mm_add_ps(vout_low, vo);
        _mm_store_ss(output, vo);
      }
    }
  }
}

void xnn_f16_f32acc_rsum_ukernel__f16c_u32_acc4(
    size_t batch,
    const void* input,
    float* output,
    const union xnn_f16_f32acc_scale_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  static const int32_t mask_table[14] = {-1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0};

  assert(batch != 0);
  assert(batch % sizeof(uint16_t) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const uint16_t* i = (const uint16_t*) input;
  __m256 vacc0 = _mm256_setzero_ps();
  __m256 vacc1 = _mm256_setzero_ps();
  __m256 vacc2 = _mm256_setzero_ps();
  __m256 vacc3 = _mm256_setzero_ps();
  for (; batch >= 32 * sizeof(uint16_t); batch -= 32 * sizeof(uint16_t)) {
    const __m256 vt0 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i));
    const __m256 vt1 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i + 8)));
    const __m256 vt2 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i + 16)));
    const __m256 vt3 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i + 24)));
    i += 32;

    vacc0 = _mm256_add_ps(vacc0, vt0);
    vacc1 = _mm256_add_ps(vacc1, vt1);
    vacc2 = _mm256_add_ps(vacc2, vt2);
    vacc3 = _mm256_add_ps(vacc3, vt3);
  }
  vacc0 = _mm256_add_ps(vacc0, vacc1);
  vacc2 = _mm256_add_ps(vacc2, vacc3);
  vacc0 = _mm256_add_ps(vacc0, vacc2);
  for (; batch >= 8 * sizeof(uint16_t); batch -= 8 * sizeof(uint16_t)) {
    const __m256 vt = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i));
    i += 8;

    vacc0 = _mm256_add_ps(vacc0, vt);
  }
  if XNN_UNLIKELY(batch != 0) {
    assert(batch >= 1 * sizeof(uint16_t));
    assert(batch <= 7 * sizeof(uint16_t));
    const __m128i vmask = _mm_loadu_si128((const __m128i*) ((uintptr_t) &mask_table[7] - batch));
    const __m128i vh = _mm_castps_si128(_mm_maskload_ps((const float*) i, vmask));
    const __m256 vt = _mm256_cvtph_ps(vh);
    vacc0 = _mm256_add_ps(vacc0, vt);
    i = (const void*) ((uintptr_t) i + batch);
    if (batch & (1 * sizeof(uint16_t))) {
      const __m128i vh = _mm_insert_epi16(_mm_setzero_si128(), (int) unaligned_load_u16(i - 1), 0);
      const __m256 vt = _mm256_zextps128_ps256(_mm_cvtph_ps(vh));
      vacc0 = _mm256_add_ps(vacc0, vt);
    }
  }
  __m128 vacc = _mm_add_ps(_mm256_castps256_ps128(vacc0), _mm256_extractf128_ps(vacc0, 1));
  vacc = _mm_add_ps(vacc, _mm_movehl_ps(vacc, vacc));
  vacc = _mm_add_ss(vacc, _mm_movehdup_ps(vacc));
  vacc = _mm_mul_ss(vacc, _mm_load_ss(&params->scale));

  float vout = _mm_cvtss_f32(vacc);
  *output += vout;
}

void xnn_f16_gavgpool_minmax_ukernel_7p7x__f16c_c8(
    size_t rows,
    size_t channels,
    const void* input,
    size_t input_stride,
    const void* zero,
    void* buffer,
    void* output,
    const union xnn_f16_scaleminmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(rows > 7);
  assert(channels != 0);

  const uint16_t* i0 = input;
  const uint16_t* i1 = (const uint16_t*) ((uintptr_t) i0 + input_stride);
  const uint16_t* i2 = (const uint16_t*) ((uintptr_t) i1 + input_stride);
  const uint16_t* i3 = (const uint16_t*) ((uintptr_t) i2 + input_stride);
  const uint16_t* i4 = (const uint16_t*) ((uintptr_t) i3 + input_stride);
  const uint16_t* i5 = (const uint16_t*) ((uintptr_t) i4 + input_stride);
  const uint16_t* i6 = (const uint16_t*) ((uintptr_t) i5 + input_stride);
  const size_t input_increment = 7 * input_stride - round_up_po2(channels, 8) * sizeof(uint16_t);

  uint16_t* b = buffer;
  size_t c = channels;
  for (; c != 0; c = doz(c, 8)) {
    const __m256 vi0x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i0)); i0 += 8;
    const __m256 vi1x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i1)); i1 += 8;

    const __m256 vi2x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i2)); i2 += 8;
    __m128i vacc01234567 = _mm256_cvtps_ph(_mm256_add_ps(vi0x01234567, vi1x01234567), _MM_FROUND_TO_NEAREST_INT);

    const __m256 vi3x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i3)); i3 += 8;
    vacc01234567 = _mm256_cvtps_ph(_mm256_add_ps(_mm256_cvtph_ps(vacc01234567), vi2x01234567), _MM_FROUND_TO_NEAREST_INT);
    const __m256 vi4x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i4)); i4 += 8;
    vacc01234567 = _mm256_cvtps_ph(_mm256_add_ps(_mm256_cvtph_ps(vacc01234567), vi3x01234567), _MM_FROUND_TO_NEAREST_INT);
    const __m256 vi5x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i5)); i5 += 8;
    vacc01234567 = _mm256_cvtps_ph(_mm256_add_ps(_mm256_cvtph_ps(vacc01234567), vi4x01234567), _MM_FROUND_TO_NEAREST_INT);
    const __m256 vi6x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i6)); i6 += 8;
    vacc01234567 = _mm256_cvtps_ph(_mm256_add_ps(_mm256_cvtph_ps(vacc01234567), vi5x01234567), _MM_FROUND_TO_NEAREST_INT);
    vacc01234567 = _mm256_cvtps_ph(_mm256_add_ps(_mm256_cvtph_ps(vacc01234567), vi6x01234567), _MM_FROUND_TO_NEAREST_INT);

    _mm_store_si128((__m128i*) b, vacc01234567); b += 8;
  }

  for (rows -= 7; rows > 7; rows -= 7) {
    i0 = (const uint16_t*) ((uintptr_t) i0 + input_increment);
    i1 = (const uint16_t*) ((uintptr_t) i1 + input_increment);
    i2 = (const uint16_t*) ((uintptr_t) i2 + input_increment);
    i3 = (const uint16_t*) ((uintptr_t) i3 + input_increment);
    i4 = (const uint16_t*) ((uintptr_t) i4 + input_increment);
    i5 = (const uint16_t*) ((uintptr_t) i5 + input_increment);
    i6 = (const uint16_t*) ((uintptr_t) i6 + input_increment);

    uint16_t* b = buffer;
    size_t c = channels;
    for (; c != 0; c = doz(c, 8)) {
      __m128i vacc01234567 = _mm_loadu_si128((const __m128i*) b);

      const __m256 vi0x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i0)); i0 += 8;

      const __m256 vi1x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i1)); i1 += 8;
      vacc01234567 = _mm256_cvtps_ph(_mm256_add_ps(_mm256_cvtph_ps(vacc01234567), vi0x01234567), _MM_FROUND_TO_NEAREST_INT);
      const __m256 vi2x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i2)); i2 += 8;
      vacc01234567 = _mm256_cvtps_ph(_mm256_add_ps(_mm256_cvtph_ps(vacc01234567), vi1x01234567), _MM_FROUND_TO_NEAREST_INT);
      const __m256 vi3x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i3)); i3 += 8;
      vacc01234567 = _mm256_cvtps_ph(_mm256_add_ps(_mm256_cvtph_ps(vacc01234567), vi2x01234567), _MM_FROUND_TO_NEAREST_INT);
      const __m256 vi4x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i4)); i4 += 8;
      vacc01234567 = _mm256_cvtps_ph(_mm256_add_ps(_mm256_cvtph_ps(vacc01234567), vi3x01234567), _MM_FROUND_TO_NEAREST_INT);
      const __m256 vi5x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i5)); i5 += 8;
      vacc01234567 = _mm256_cvtps_ph(_mm256_add_ps(_mm256_cvtph_ps(vacc01234567), vi4x01234567), _MM_FROUND_TO_NEAREST_INT);
      const __m256 vi6x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i6)); i6 += 8;
      vacc01234567 = _mm256_cvtps_ph(_mm256_add_ps(_mm256_cvtph_ps(vacc01234567), vi5x01234567), _MM_FROUND_TO_NEAREST_INT);
      vacc01234567 = _mm256_cvtps_ph(_mm256_add_ps(_mm256_cvtph_ps(vacc01234567), vi6x01234567), _MM_FROUND_TO_NEAREST_INT);

      _mm_store_si128((__m128i*) b, vacc01234567); b += 8;
    }
  }

  i0 = (const uint16_t*) ((uintptr_t) i0 + input_increment);
  i1 = (const uint16_t*) ((uintptr_t) i1 + input_increment);
  if XNN_UNPREDICTABLE(rows < 2) {
    i1 = (const uint16_t*) zero;
  }
  i2 = (const uint16_t*) ((uintptr_t) i2 + input_increment);
  if XNN_UNPREDICTABLE(rows <= 2) {
    i2 = (const uint16_t*) zero;
  }
  i3 = (const uint16_t*) ((uintptr_t) i3 + input_increment);
  if XNN_UNPREDICTABLE(rows < 4) {
    i3 = (const uint16_t*) zero;
  }
  i4 = (const uint16_t*) ((uintptr_t) i4 + input_increment);
  if XNN_UNPREDICTABLE(rows <= 4) {
    i4 = (const uint16_t*) zero;
  }
  i5 = (const uint16_t*) ((uintptr_t) i5 + input_increment);
  if XNN_UNPREDICTABLE(rows < 6) {
    i5 = (const uint16_t*) zero;
  }
  i6 = (const uint16_t*) ((uintptr_t) i6 + input_increment);
  if XNN_UNPREDICTABLE(rows <= 6) {
    i6 = (const uint16_t*) zero;
  }
  uint16_t* o = (uint16_t*) output;

  const __m256 vscale = _mm256_set1_ps(params->avx.scale);
  const __m256 vmin = _mm256_set1_ps(params->avx.min);
  const __m256 vmax = _mm256_set1_ps(params->avx.max);
  XNN_FORCE_REALIZATION(vscale);
  XNN_FORCE_REALIZATION(vmin);
  XNN_FORCE_REALIZATION(vmax);
  for (; channels >= 8; channels -= 8) {
    __m128i vacc01234567 = _mm_loadu_si128((const __m128i*) buffer); buffer = (uint16_t*) buffer + 8;

    const __m256 vi0x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i0)); i0 += 8;

    const __m256 vi1x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i1)); i1 += 8;
    vacc01234567 = _mm256_cvtps_ph(_mm256_add_ps(_mm256_cvtph_ps(vacc01234567), vi0x01234567), _MM_FROUND_TO_NEAREST_INT);
    const __m256 vi2x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i2)); i2 += 8;
    vacc01234567 = _mm256_cvtps_ph(_mm256_add_ps(_mm256_cvtph_ps(vacc01234567), vi1x01234567), _MM_FROUND_TO_NEAREST_INT);
    const __m256 vi3x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i3)); i3 += 8;
    vacc01234567 = _mm256_cvtps_ph(_mm256_add_ps(_mm256_cvtph_ps(vacc01234567), vi2x01234567), _MM_FROUND_TO_NEAREST_INT);
    const __m256 vi4x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i4)); i4 += 8;
    vacc01234567 = _mm256_cvtps_ph(_mm256_add_ps(_mm256_cvtph_ps(vacc01234567), vi3x01234567), _MM_FROUND_TO_NEAREST_INT);
    const __m256 vi5x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i5)); i5 += 8;
    vacc01234567 = _mm256_cvtps_ph(_mm256_add_ps(_mm256_cvtph_ps(vacc01234567), vi4x01234567), _MM_FROUND_TO_NEAREST_INT);
    const __m256 vi6x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i6)); i6 += 8;
    vacc01234567 = _mm256_cvtps_ph(_mm256_add_ps(_mm256_cvtph_ps(vacc01234567), vi5x01234567), _MM_FROUND_TO_NEAREST_INT);
    vacc01234567 = _mm256_cvtps_ph(_mm256_add_ps(_mm256_cvtph_ps(vacc01234567), vi6x01234567), _MM_FROUND_TO_NEAREST_INT);

    vacc01234567 = _mm256_cvtps_ph(_mm256_mul_ps(_mm256_cvtph_ps(vacc01234567), vscale), _MM_FROUND_TO_NEAREST_INT);

    __m256 vout01234567 = _mm256_max_ps(_mm256_cvtph_ps(vacc01234567), vmin);

    vout01234567 = _mm256_min_ps(vout01234567, vmax);

    _mm_storeu_si128((__m128i*) o, _mm256_cvtps_ph(vout01234567, _MM_FROUND_TO_NEAREST_INT));
    o += 8;
  }
  if XNN_UNLIKELY(channels != 0) {
    {
      __m128i vacc01234567 = _mm_loadu_si128((const __m128i*) buffer); buffer = (uint16_t*) buffer + 8;

      const __m256 vi0x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i0)); i0 += 8;
      const __m256 vi1x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i1)); i1 += 8;
      vacc01234567 = _mm256_cvtps_ph(_mm256_add_ps(_mm256_cvtph_ps(vacc01234567), vi0x01234567), _MM_FROUND_TO_NEAREST_INT);
      const __m256 vi2x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i2)); i2 += 8;
      vacc01234567 = _mm256_cvtps_ph(_mm256_add_ps(_mm256_cvtph_ps(vacc01234567), vi1x01234567), _MM_FROUND_TO_NEAREST_INT);
      const __m256 vi3x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i3)); i3 += 8;
      vacc01234567 = _mm256_cvtps_ph(_mm256_add_ps(_mm256_cvtph_ps(vacc01234567), vi2x01234567), _MM_FROUND_TO_NEAREST_INT);
      const __m256 vi4x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i4)); i4 += 8;
      vacc01234567 = _mm256_cvtps_ph(_mm256_add_ps(_mm256_cvtph_ps(vacc01234567), vi3x01234567), _MM_FROUND_TO_NEAREST_INT);
      const __m256 vi5x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i5)); i5 += 8;
      vacc01234567 = _mm256_cvtps_ph(_mm256_add_ps(_mm256_cvtph_ps(vacc01234567), vi4x01234567), _MM_FROUND_TO_NEAREST_INT);
      const __m256 vi6x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i6)); i6 += 8;
      vacc01234567 = _mm256_cvtps_ph(_mm256_add_ps(_mm256_cvtph_ps(vacc01234567), vi5x01234567), _MM_FROUND_TO_NEAREST_INT);
      vacc01234567 = _mm256_cvtps_ph(_mm256_add_ps(_mm256_cvtph_ps(vacc01234567), vi6x01234567), _MM_FROUND_TO_NEAREST_INT);

      vacc01234567 = _mm256_cvtps_ph(_mm256_mul_ps(_mm256_cvtph_ps(vacc01234567), vscale), _MM_FROUND_TO_NEAREST_INT);
      __m256 vout01234567 = _mm256_max_ps(_mm256_cvtph_ps(vacc01234567), vmin);
      vout01234567 = _mm256_min_ps(vout01234567, vmax);

      __m128i vh01234567 = _mm256_cvtps_ph(vout01234567, _MM_FROUND_TO_NEAREST_INT);
      if (channels & 4) {
        _mm_storel_epi64((__m128i*) o, vh01234567);
        o += 4;
        vh01234567 = _mm_unpackhi_epi64(vh01234567, vh01234567);
      }
      if (channels & 2) {
        _mm_storeu_si32(o, vh01234567);
        o += 2;
        vh01234567 = _mm_srli_epi64(vh01234567, 32);
      }
      if (channels & 1) {
        *o = (uint16_t) _mm_extract_epi16(vh01234567, 0);
      }
    }
  }
}

void xnn_f16_gavgpool_minmax_ukernel_7x__f16c_c8(
    size_t rows,
    size_t channels,
    const void* input,
    size_t input_stride,
    const void* zero,
    void* output,
    const union xnn_f16_scaleminmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(rows != 0);
  assert(rows <= 7);
  assert(channels != 0);

  const uint16_t* i0 = input;
  const uint16_t* i1 = (const uint16_t*) ((uintptr_t) i0 + input_stride);
  if XNN_UNPREDICTABLE(rows < 2) {
    i1 = (const uint16_t*) zero;
  }
  const uint16_t* i2 = (const uint16_t*) ((uintptr_t) i1 + input_stride);
  if XNN_UNPREDICTABLE(rows <= 2) {
    i2 = (const uint16_t*) zero;
  }
  const uint16_t* i3 = (const uint16_t*) ((uintptr_t) i2 + input_stride);
  if XNN_UNPREDICTABLE(rows < 4) {
    i3 = (const uint16_t*) zero;
  }
  const uint16_t* i4 = (const uint16_t*) ((uintptr_t) i3 + input_stride);
  if XNN_UNPREDICTABLE(rows <= 4) {
    i4 = (const uint16_t*) zero;
  }
  const uint16_t* i5 = (const uint16_t*) ((uintptr_t) i4 + input_stride);
  if XNN_UNPREDICTABLE(rows < 6) {
    i5 = (const uint16_t*) zero;
  }
  const uint16_t* i6 = (const uint16_t*) ((uintptr_t) i5 + input_stride);
  if XNN_UNPREDICTABLE(rows <= 6) {
    i6 = (const uint16_t*) zero;
  }
  uint16_t* o = (uint16_t*) output;

  const __m256 vscale = _mm256_set1_ps(params->avx.scale);
  const __m256 vmin = _mm256_set1_ps(params->avx.min);
  const __m256 vmax = _mm256_set1_ps(params->avx.max);
  XNN_FORCE_REALIZATION(vscale);
  XNN_FORCE_REALIZATION(vmin);
  XNN_FORCE_REALIZATION(vmax);
  for (; channels >= 8; channels -= 8) {
    const __m256 vi0x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i0));
    i0 += 8;
    const __m256 vi1x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i1));
    i1 += 8;

    const __m256 vi2x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i2));
    __m128i vacc01234567 = _mm256_cvtps_ph(_mm256_add_ps(vi0x01234567, vi1x01234567), _MM_FROUND_TO_NEAREST_INT);
    i2 += 8;

    const __m256 vi3x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i3));
    i3 += 8;
    vacc01234567 = _mm256_cvtps_ph(_mm256_add_ps(_mm256_cvtph_ps(vacc01234567), vi2x01234567), _MM_FROUND_TO_NEAREST_INT);
    const __m256 vi4x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i4));
    i4 += 8;
    vacc01234567 = _mm256_cvtps_ph(_mm256_add_ps(_mm256_cvtph_ps(vacc01234567), vi3x01234567), _MM_FROUND_TO_NEAREST_INT);
    const __m256 vi5x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i5));
    i5 += 8;
    vacc01234567 = _mm256_cvtps_ph(_mm256_add_ps(_mm256_cvtph_ps(vacc01234567), vi4x01234567), _MM_FROUND_TO_NEAREST_INT);
    const __m256 vi6x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i6));
    i6 += 8;
    vacc01234567 = _mm256_cvtps_ph(_mm256_add_ps(_mm256_cvtph_ps(vacc01234567), vi5x01234567), _MM_FROUND_TO_NEAREST_INT);
    vacc01234567 = _mm256_cvtps_ph(_mm256_add_ps(_mm256_cvtph_ps(vacc01234567), vi6x01234567), _MM_FROUND_TO_NEAREST_INT);

    vacc01234567 = _mm256_cvtps_ph(_mm256_mul_ps(_mm256_cvtph_ps(vacc01234567), vscale), _MM_FROUND_TO_NEAREST_INT);

    __m256 vout01234567 = _mm256_max_ps(_mm256_cvtph_ps(vacc01234567), vmin);

    vout01234567 = _mm256_min_ps(vout01234567, vmax);

    _mm_storeu_si128((__m128i*) o, _mm256_cvtps_ph(vout01234567, _MM_FROUND_TO_NEAREST_INT));
    o += 8;
  }
  if XNN_UNLIKELY(channels != 0) {
    {
      const __m256 vi0x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i0));
      const __m256 vi1x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i1));

      const __m256 vi2x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i2));
      __m128i vacc01234567 = _mm256_cvtps_ph(_mm256_add_ps(vi0x01234567, vi1x01234567), _MM_FROUND_TO_NEAREST_INT);

      const __m256 vi3x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i3));
      vacc01234567 = _mm256_cvtps_ph(_mm256_add_ps(_mm256_cvtph_ps(vacc01234567), vi2x01234567), _MM_FROUND_TO_NEAREST_INT);
      const __m256 vi4x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i4));
      vacc01234567 = _mm256_cvtps_ph(_mm256_add_ps(_mm256_cvtph_ps(vacc01234567), vi3x01234567), _MM_FROUND_TO_NEAREST_INT);
      const __m256 vi5x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i5));
      vacc01234567 = _mm256_cvtps_ph(_mm256_add_ps(_mm256_cvtph_ps(vacc01234567), vi4x01234567), _MM_FROUND_TO_NEAREST_INT);
      const __m256 vi6x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i6));
      vacc01234567 = _mm256_cvtps_ph(_mm256_add_ps(_mm256_cvtph_ps(vacc01234567), vi5x01234567), _MM_FROUND_TO_NEAREST_INT);
      vacc01234567 = _mm256_cvtps_ph(_mm256_add_ps(_mm256_cvtph_ps(vacc01234567), vi6x01234567), _MM_FROUND_TO_NEAREST_INT);

      vacc01234567 = _mm256_cvtps_ph(_mm256_mul_ps(_mm256_cvtph_ps(vacc01234567), vscale), _MM_FROUND_TO_NEAREST_INT);
      __m256 vout01234567 = _mm256_max_ps(_mm256_cvtph_ps(vacc01234567), vmin);
      vout01234567 = _mm256_min_ps(vout01234567, vmax);

      __m128i vh01234567 = _mm256_cvtps_ph(vout01234567, _MM_FROUND_TO_NEAREST_INT);
      if (channels & 4) {
        _mm_storel_epi64((__m128i*) o, vh01234567);
        o += 4;
        vh01234567 = _mm_unpackhi_epi64(vh01234567, vh01234567);
      }
      if (channels & 2) {
        _mm_storeu_si32(o, vh01234567);
        o += 2;
        vh01234567 = _mm_srli_epi64(vh01234567, 32);
      }
      if (channels & 1) {
        *o = (uint16_t) _mm_extract_epi16(vh01234567, 0);
      }
    }
  }
}

void xnn_f16_maxpool_minmax_ukernel_9p8x__f16c_c8(
    size_t output_pixels,
    size_t kernel_elements,
    size_t channels,
    const void** input,
    size_t input_offset,
    void* output,
    size_t input_increment,
    size_t output_increment,
    const union xnn_f16_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(output_pixels != 0);
  assert(kernel_elements != 0);
  assert(channels != 0);

  const __m256 voutput_min = _mm256_set1_ps(params->avx.min);
  const __m256 voutput_max = _mm256_set1_ps(params->avx.max);
  XNN_FORCE_REALIZATION(voutput_min);
  XNN_FORCE_REALIZATION(voutput_max);
  do {
    uint16_t* o = output;
    {
      const uint16_t* i0 = *input++;
      const uint16_t* i1 = *input++;
      const uint16_t* i2 = *input++;
      const uint16_t* i3 = *input++;
      const uint16_t* i4 = *input++;
      const uint16_t* i5 = *input++;
      const uint16_t* i6 = *input++;
      const uint16_t* i7 = *input++;
      const uint16_t* i8 = *input++;
      i0 = (const uint16_t*) ((uintptr_t) i0 + input_offset);
      i1 = (const uint16_t*) ((uintptr_t) i1 + input_offset);
      i2 = (const uint16_t*) ((uintptr_t) i2 + input_offset);
      i3 = (const uint16_t*) ((uintptr_t) i3 + input_offset);
      i4 = (const uint16_t*) ((uintptr_t) i4 + input_offset);
      i5 = (const uint16_t*) ((uintptr_t) i5 + input_offset);
      i6 = (const uint16_t*) ((uintptr_t) i6 + input_offset);
      i7 = (const uint16_t*) ((uintptr_t) i7 + input_offset);
      i8 = (const uint16_t*) ((uintptr_t) i8 + input_offset);
      if (kernel_elements < 2) {
        i1 = i0;
      }
      if (kernel_elements <= 2) {
        i2 = i0;
      }
      if (kernel_elements < 4) {
        i3 = i0;
      }
      if (kernel_elements <= 4) {
        i4 = i0;
      }
      if (kernel_elements < 6) {
        i5 = i0;
      }
      if (kernel_elements <= 6) {
        i6 = i0;
      }
      if (kernel_elements < 8) {
        i7 = i0;
      }
      if (kernel_elements <= 8) {
        i8 = i0;
      }

      size_t c = channels;
      for (; c >= 8; c -= 8) {
        const __m256 vi0 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i0));
        i0 += 8;
        const __m256 vi1 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i1));
        i1 += 8;
        const __m256 vi2 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i2));
        i2 += 8;
        const __m256 vi3 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i3));
        i3 += 8;
        const __m256 vi4 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i4));
        i4 += 8;
        const __m256 vi5 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i5));
        i5 += 8;
        const __m256 vi6 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i6));
        i6 += 8;
        const __m256 vi7 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i7));
        i7 += 8;
        const __m256 vi8 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i8));
        i8 += 8;

        const __m256 vmax018 = _mm256_max_ps(_mm256_max_ps(vi0, vi1), vi8);
        const __m256 vmax23 = _mm256_max_ps(vi2, vi3);
        const __m256 vmax45 = _mm256_max_ps(vi4, vi5);
        const __m256 vmax67 = _mm256_max_ps(vi6, vi7);

        const __m256 vmax2345 = _mm256_max_ps(vmax23, vmax45);
        const __m256 vmax01678 = _mm256_max_ps(vmax018, vmax67);
        const __m256 vmax = _mm256_max_ps(vmax2345, vmax01678);
        const __m256 vout = _mm256_max_ps(_mm256_min_ps(vmax, voutput_max), voutput_min);

        _mm_storeu_si128((__m128i*) o, _mm256_cvtps_ph(vout, _MM_FROUND_TO_NEAREST_INT));
        o += 8;
      }
      if (c != 0) {
        const __m256 vi0 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i0));
        i0 += 8;
        const __m256 vi1 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i1));
        i1 += 8;
        const __m256 vi2 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i2));
        i2 += 8;
        const __m256 vi3 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i3));
        i3 += 8;
        const __m256 vi4 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i4));
        i4 += 8;
        const __m256 vi5 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i5));
        i5 += 8;
        const __m256 vi6 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i6));
        i6 += 8;
        const __m256 vi7 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i7));
        i7 += 8;
        const __m256 vi8 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i8));
        i8 += 8;

        const __m256 vmax018 = _mm256_max_ps(_mm256_max_ps(vi0, vi1), vi8);
        const __m256 vmax23 = _mm256_max_ps(vi2, vi3);
        const __m256 vmax45 = _mm256_max_ps(vi4, vi5);
        const __m256 vmax67 = _mm256_max_ps(vi6, vi7);

        const __m256 vmax2345 = _mm256_max_ps(vmax23, vmax45);
        const __m256 vmax01678 = _mm256_max_ps(vmax018, vmax67);
        const __m256 vmax = _mm256_max_ps(vmax2345, vmax01678);
        __m256 vout = _mm256_max_ps(_mm256_min_ps(vmax, voutput_max), voutput_min);

        __m128i vh = _mm256_cvtps_ph(vout, _MM_FROUND_TO_NEAREST_INT);
        if (c & 4) {
          _mm_storel_epi64((__m128i*) o, vh);
          vh = _mm_unpackhi_epi64(vh, vh);
          o += 4;
        }
        if (c & 2) {
          _mm_storeu_si32(o, vh);
          vh = _mm_srli_epi64(vh, 32);
          o += 2;
        }
        if (c & 1) {
          *o = _mm_extract_epi16(vh, 0);
          o += 1;
        }
      }
    }

    for (ptrdiff_t k = (ptrdiff_t) kernel_elements - 9; k > 0; k -= 8) {
      const uint16_t* i0 = *input++;
      const uint16_t* i1 = *input++;
      const uint16_t* i2 = *input++;
      const uint16_t* i3 = *input++;
      const uint16_t* i4 = *input++;
      const uint16_t* i5 = *input++;
      const uint16_t* i6 = *input++;
      const uint16_t* i7 = *input++;
      i0 = (const uint16_t*) ((uintptr_t) i0 + input_offset);
      i1 = (const uint16_t*) ((uintptr_t) i1 + input_offset);
      i2 = (const uint16_t*) ((uintptr_t) i2 + input_offset);
      i3 = (const uint16_t*) ((uintptr_t) i3 + input_offset);
      i4 = (const uint16_t*) ((uintptr_t) i4 + input_offset);
      i5 = (const uint16_t*) ((uintptr_t) i5 + input_offset);
      i6 = (const uint16_t*) ((uintptr_t) i6 + input_offset);
      i7 = (const uint16_t*) ((uintptr_t) i7 + input_offset);
      if (k < 2) {
        i1 = i0;
      }
      if (k <= 2) {
        i2 = i0;
      }
      if (k < 4) {
        i3 = i0;
      }
      if (k <= 4) {
        i4 = i0;
      }
      if (k < 6) {
        i5 = i0;
      }
      if (k <= 6) {
        i6 = i0;
      }
      if (k < 8) {
        i7 = i0;
      }

      o = output;
      size_t c = channels;
      for (; c >= 8; c -= 8) {
        const __m256 vi0 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i0));
        i0 += 8;
        const __m256 vi1 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i1));
        i1 += 8;
        const __m256 vi2 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i2));
        i2 += 8;
        const __m256 vi3 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i3));
        i3 += 8;
        const __m256 vi4 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i4));
        i4 += 8;
        const __m256 vi5 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i5));
        i5 += 8;
        const __m256 vi6 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i6));
        i6 += 8;
        const __m256 vi7 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i7));
        i7 += 8;
        const __m256 vo = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) o));

        const __m256 vmax01 = _mm256_max_ps(_mm256_max_ps(vi0, vi1), vo);
        const __m256 vmax23 = _mm256_max_ps(vi2, vi3);
        const __m256 vmax45 = _mm256_max_ps(vi4, vi5);
        const __m256 vmax67 = _mm256_max_ps(vi6, vi7);

        const __m256 vmax2345 = _mm256_max_ps(vmax23, vmax45);
        const __m256 vmax0167 = _mm256_max_ps(vmax01, vmax67);
        const __m256 vmax = _mm256_max_ps(vmax2345, vmax0167);
        const __m256 vout = _mm256_max_ps(_mm256_min_ps(vmax, voutput_max), voutput_min);

        _mm_storeu_si128((__m128i*) o, _mm256_cvtps_ph(vout, _MM_FROUND_TO_NEAREST_INT));
        o += 8;
      }
      if (c != 0) {
        const __m256 vi0 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i0));
        const __m256 vi1 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i1));
        const __m256 vi2 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i2));
        const __m256 vi3 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i3));
        const __m256 vi4 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i4));
        const __m256 vi5 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i5));
        const __m256 vi6 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i6));
        const __m256 vi7 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i7));
        const __m256 vo = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) o));

        const __m256 vmax01 = _mm256_max_ps(_mm256_max_ps(vi0, vi1), vo);
        const __m256 vmax23 = _mm256_max_ps(vi2, vi3);
        const __m256 vmax45 = _mm256_max_ps(vi4, vi5);
        const __m256 vmax67 = _mm256_max_ps(vi6, vi7);

        const __m256 vmax2345 = _mm256_max_ps(vmax23, vmax45);
        const __m256 vmax0167 = _mm256_max_ps(vmax01, vmax67);
        const __m256 vmax = _mm256_max_ps(vmax2345, vmax0167);
        __m256 vout = _mm256_max_ps(_mm256_min_ps(vmax, voutput_max), voutput_min);

        __m128i vh = _mm256_cvtps_ph(vout, _MM_FROUND_TO_NEAREST_INT);
        if (c & 4) {
          _mm_storel_epi64((__m128i*) o, vh);
          vh = _mm_unpackhi_epi64(vh, vh);
          o += 4;
        }
        if (c & 2) {
          _mm_storeu_si32(o, vh);
          vh = _mm_srli_epi64(vh, 32);
          o += 2;
        }
        if (c & 1) {
          *o = _mm_extract_epi16(vh, 0);
          o += 1;
        }
      }
    }
    input = (const void**) ((uintptr_t) input + input_increment);
    output = (uint16_t*) ((uintptr_t) o + output_increment);
  } while (--output_pixels != 0);
}

void xnn_f16_prelu_ukernel__f16c_2x16(
    size_t rows,
    size_t channels,
    const void* restrict input,
    size_t input_stride,
    const void* restrict weights,
    void* restrict output,
    size_t output_stride) XNN_OOB_READS
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

  do {
    if XNN_UNPREDICTABLE(rows < 2) {
      i1 = i0;
      o1 = o0;
    }

    const uint16_t* w = (const uint16_t*) weights;
    size_t c = channels;
    for (; c >= 16 * sizeof(uint16_t); c -= 16 * sizeof(uint16_t)) {
      const __m256 vw01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) w));
      const __m256 vw89ABCDEF = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (w + 8)));
      w += 16;

      const __m256 vi0x001234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i0));
      const __m256 vi0x089ABCDEF = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i0 + 8)));
      i0 += 16;
      const __m256 vi1x001234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i1));
      const __m256 vi1x089ABCDEF = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i1 + 8)));
      i1 += 16;

      __m256 vacc0x001234567 = _mm256_mul_ps(vi0x001234567, vw01234567);
      __m256 vacc0x089ABCDEF = _mm256_mul_ps(vi0x089ABCDEF, vw89ABCDEF);
      __m256 vacc1x001234567 = _mm256_mul_ps(vi1x001234567, vw01234567);
      __m256 vacc1x089ABCDEF = _mm256_mul_ps(vi1x089ABCDEF, vw89ABCDEF);

      vacc0x001234567 = _mm256_blendv_ps(vi0x001234567, vacc0x001234567, vi0x001234567);
      vacc0x089ABCDEF = _mm256_blendv_ps(vi0x089ABCDEF, vacc0x089ABCDEF, vi0x089ABCDEF);
      vacc1x001234567 = _mm256_blendv_ps(vi1x001234567, vacc1x001234567, vi1x001234567);
      vacc1x089ABCDEF = _mm256_blendv_ps(vi1x089ABCDEF, vacc1x089ABCDEF, vi1x089ABCDEF);

      _mm_storeu_si128((__m128i*) o0, _mm256_cvtps_ph(vacc0x089ABCDEF, _MM_FROUND_TO_NEAREST_INT));
      _mm_storeu_si128((__m128i*) (o0 + 0), _mm256_cvtps_ph(vacc0x001234567, _MM_FROUND_TO_NEAREST_INT));
      _mm_storeu_si128((__m128i*) (o0 + 8), _mm256_cvtps_ph(vacc0x089ABCDEF, _MM_FROUND_TO_NEAREST_INT));
      o0 += 16;
      _mm_storeu_si128((__m128i*) o1, _mm256_cvtps_ph(vacc1x089ABCDEF, _MM_FROUND_TO_NEAREST_INT));
      _mm_storeu_si128((__m128i*) (o1 + 0), _mm256_cvtps_ph(vacc1x001234567, _MM_FROUND_TO_NEAREST_INT));
      _mm_storeu_si128((__m128i*) (o1 + 8), _mm256_cvtps_ph(vacc1x089ABCDEF, _MM_FROUND_TO_NEAREST_INT));
      o1 += 16;
    }
    for (; c >= 8 * sizeof(uint16_t); c -= 8 * sizeof(uint16_t)) {
      const __m256 vw01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) w));
      w += 8;

      const __m256 vi0x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i0));
      i0 += 8;
      const __m256 vi1x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i1));
      i1 += 8;

      __m256 vacc0x01234567 = _mm256_mul_ps(vi0x01234567, vw01234567);
      __m256 vacc1x01234567 = _mm256_mul_ps(vi1x01234567, vw01234567);

      vacc0x01234567 = _mm256_blendv_ps(vi0x01234567, vacc0x01234567, vi0x01234567);
      vacc1x01234567 = _mm256_blendv_ps(vi1x01234567, vacc1x01234567, vi1x01234567);

      _mm_storeu_si128((__m128i*) o0, _mm256_cvtps_ph(vacc0x01234567, _MM_FROUND_TO_NEAREST_INT));
      o0 += 8;
      _mm_storeu_si128((__m128i*) o1, _mm256_cvtps_ph(vacc1x01234567, _MM_FROUND_TO_NEAREST_INT));
      o1 += 8;
    }
    if XNN_UNLIKELY(c != 0) {
      const __m256 vw01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) w));

      const __m256 vi0x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i0));
      i0 = (const uint16_t*) ((uintptr_t) i0 + c);
      const __m256 vi1x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i1));
      i1 = (const uint16_t*) ((uintptr_t) i1 + c);

      __m256 vacc0x01234567 = _mm256_mul_ps(vi0x01234567, vw01234567);
      __m256 vacc1x01234567 = _mm256_mul_ps(vi1x01234567, vw01234567);

      vacc0x01234567 = _mm256_blendv_ps(vi0x01234567, vacc0x01234567, vi0x01234567);
      vacc1x01234567 = _mm256_blendv_ps(vi1x01234567, vacc1x01234567, vi1x01234567);

      __m128i vh0x01234567 = _mm256_cvtps_ph(vacc0x01234567, _MM_FROUND_TO_NEAREST_INT);
      __m128i vh1x01234567 = _mm256_cvtps_ph(vacc1x01234567, _MM_FROUND_TO_NEAREST_INT);
      if (c & (4 * sizeof(uint16_t))) {
        _mm_storel_epi64((__m128i*) o0, vh0x01234567);
        _mm_storel_epi64((__m128i*) o1, vh1x01234567);

        vh0x01234567 = _mm_unpackhi_epi64(vh0x01234567, vh0x01234567);
        vh1x01234567 = _mm_unpackhi_epi64(vh1x01234567, vh1x01234567);

        o0 += 4;
        o1 += 4;
      }
      if (c & (2 * sizeof(uint16_t))) {
        _mm_storeu_si32(o0, vh0x01234567);
        _mm_storeu_si32(o1, vh1x01234567);

        vh0x01234567 = _mm_srli_epi64(vh0x01234567, 32);
        vh1x01234567 = _mm_srli_epi64(vh1x01234567, 32);

        o0 += 2;
        o1 += 2;
      }
      if (c & (1 * sizeof(uint16_t))) {
        *o0 = (uint16_t) _mm_extract_epi16(vh0x01234567, 0);
        *o1 = (uint16_t) _mm_extract_epi16(vh1x01234567, 0);

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

void xnn_f16_rmax_ukernel__f16c_u32(
    size_t batch,
    const void* input,
    void* output,
    const union xnn_f16_default_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(uint16_t) == 0);
  assert(input != 0);
  assert(output != 0);

  const uint16_t* i = (const uint16_t*) input;
  __m128i vmax_init = _mm_shufflelo_epi16(_mm_loadl_epi64((const __m128i*) i), _MM_SHUFFLE(0, 0, 0, 0));
  vmax_init = _mm_unpacklo_epi64(vmax_init, vmax_init);
  __m256 vmax0 = _mm256_cvtph_ps(vmax_init);
  __m256 vmax1 = vmax0;
  __m256 vmax2 = vmax0;
  __m256 vmax3 = vmax0;
  for (; batch >= 32 * sizeof(uint16_t); batch -= 32 * sizeof(uint16_t)) {
    const __m256 vx0 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i));
    const __m256 vx1 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i + 8)));
    const __m256 vx2 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i + 16)));
    const __m256 vx3 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i + 24)));
    i += 32;

    vmax0 = _mm256_max_ps(vmax0, vx0);
    vmax1 = _mm256_max_ps(vmax1, vx1);
    vmax2 = _mm256_max_ps(vmax2, vx2);
    vmax3 = _mm256_max_ps(vmax3, vx3);
  }
  __m256 vmax = _mm256_max_ps(_mm256_max_ps(vmax0, vmax1), _mm256_max_ps(vmax2, vmax3));
  for (; batch >= 8 * sizeof(uint16_t); batch -= 8 * sizeof(uint16_t)) {
    const __m256 vx = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i));
    i += 8;
    vmax = _mm256_max_ps(vmax, vx);
  }
  __m128 vmax_lo = _mm_max_ps(_mm256_castps256_ps128(vmax), _mm256_extractf128_ps(vmax, 1));
  if XNN_UNLIKELY(batch != 0) {
    const __m256 vx = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i));
    __m128 vx_lo = _mm256_castps256_ps128(vx);
    if (batch & (4 * sizeof(uint16_t))) {
      vmax_lo = _mm_max_ps(vmax_lo, vx_lo);
      vx_lo = _mm256_extractf128_ps(vx, 1);
    }
    if (batch & (2 * sizeof(uint16_t))) {
      vmax_lo = _mm_blend_ps(_mm_max_ps(vmax_lo, vx_lo), vmax_lo, 0xC);
      vx_lo = _mm_movehl_ps(vx_lo, vx_lo);
    }
    if (batch & (1 * sizeof(uint16_t))) {
      vmax_lo = _mm_max_ss(vmax_lo, vx_lo);
    }
  }
  vmax_lo = _mm_max_ps(vmax_lo, _mm_movehl_ps(vmax_lo, vmax_lo));
  vmax_lo = _mm_max_ss(vmax_lo, _mm_movehdup_ps(vmax_lo));
  *((uint16_t*) output) = (uint16_t) _mm_extract_epi16(_mm_cvtps_ph(vmax_lo, _MM_FROUND_TO_NEAREST_INT), 0);
}

void xnn_f16_vadd_minmax_ukernel__f16c_u16(
    size_t batch,
    const void* restrict input_a,
    const void* restrict input_b,
    void* restrict output,
    const union xnn_f16_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(uint16_t) == 0);
  assert(input_a != NULL);
  assert(input_b != NULL);
  assert(output != NULL);

  const uint16_t* a = (const uint16_t*) input_a;
  const uint16_t* b = (const uint16_t*) input_b;
  uint16_t* o = (uint16_t*) output;

  const __m256 vy_min = _mm256_set1_ps(params->avx.min);
  const __m256 vy_max = _mm256_set1_ps(params->avx.max);
  XNN_FORCE_REALIZATION(vy_min);
  XNN_FORCE_REALIZATION(vy_max);

  for (; batch >= 16 * sizeof(uint16_t); batch -= 16 * sizeof(uint16_t)) {
    const __m256 va01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) a));
    const __m256 vb01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) b));
    const __m256 va456789AB = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (a + 8)));
    const __m256 vb456789AB = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (b + 8)));
    a += 16;
    b += 16;

    __m256 vy01234567 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_add_ps(va01234567, vb01234567), _MM_FROUND_TO_NEAREST_INT));
    __m256 vy456789AB = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_add_ps(va456789AB, vb456789AB), _MM_FROUND_TO_NEAREST_INT));


    vy01234567 = _mm256_max_ps(vy01234567, vy_min);
    vy456789AB = _mm256_max_ps(vy456789AB, vy_min);

    vy01234567 = _mm256_min_ps(vy01234567, vy_max);
    vy456789AB = _mm256_min_ps(vy456789AB, vy_max);

    _mm_storeu_si128((__m128i*) o, _mm256_cvtps_ph(vy01234567, _MM_FROUND_TO_NEAREST_INT));
    _mm_storeu_si128((__m128i*) (o + 8), _mm256_cvtps_ph(vy456789AB, _MM_FROUND_TO_NEAREST_INT));
    o += 16;
  }
  for (; batch >= 8 * sizeof(uint16_t); batch -= 8 * sizeof(uint16_t)) {
    const __m256 va = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) a));
    const __m256 vb = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) b));
    a += 8;
    b += 8;

    __m256 vy = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_add_ps(va, vb), _MM_FROUND_TO_NEAREST_INT));

    vy = _mm256_max_ps(vy, vy_min);
    vy = _mm256_min_ps(vy, vy_max);

    _mm_storeu_si128((__m128i*) o, _mm256_cvtps_ph(vy, _MM_FROUND_TO_NEAREST_INT));
    o += 8;
  }
  if XNN_UNLIKELY(batch != 0) {
    const __m256 va = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) a));
    const __m256 vb = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) b));

    __m256 vy = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_add_ps(va, vb), _MM_FROUND_TO_NEAREST_INT));

    vy = _mm256_max_ps(vy, vy_min);
    vy = _mm256_min_ps(vy, vy_max);

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

void xnn_f16_vaddc_minmax_ukernel__f16c_u16(
    size_t batch,
    const void* restrict input_a,
    const void* restrict input_b,
    void* restrict output,
    const union xnn_f16_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(uint16_t) == 0);
  assert(input_a != NULL);
  assert(input_b != NULL);
  assert(output != NULL);

  const uint16_t* a = (const uint16_t*) input_a;
  const uint16_t* b = (const uint16_t*) input_b;
  uint16_t* o = (uint16_t*) output;

  const __m256 vy_min = _mm256_set1_ps(params->avx.min);
  const __m256 vy_max = _mm256_set1_ps(params->avx.max);
  XNN_FORCE_REALIZATION(vy_min);
  XNN_FORCE_REALIZATION(vy_max);

  const __m256 vb = _mm256_cvtph_ps(_mm_set1_epi16((short) *b));
  for (; batch >= 16 * sizeof(uint16_t); batch -= 16 * sizeof(uint16_t)) {
    const __m256 va01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) a));
    const __m256 va456789AB = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (a + 8)));
    a += 16;

    __m256 vy01234567 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_add_ps(va01234567, vb), _MM_FROUND_TO_NEAREST_INT));
    __m256 vy456789AB = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_add_ps(va456789AB, vb), _MM_FROUND_TO_NEAREST_INT));


    vy01234567 = _mm256_max_ps(vy01234567, vy_min);
    vy456789AB = _mm256_max_ps(vy456789AB, vy_min);

    vy01234567 = _mm256_min_ps(vy01234567, vy_max);
    vy456789AB = _mm256_min_ps(vy456789AB, vy_max);

    _mm_storeu_si128((__m128i*) o, _mm256_cvtps_ph(vy01234567, _MM_FROUND_TO_NEAREST_INT));
    _mm_storeu_si128((__m128i*) (o + 8), _mm256_cvtps_ph(vy456789AB, _MM_FROUND_TO_NEAREST_INT));
    o += 16;
  }
  for (; batch >= 8 * sizeof(uint16_t); batch -= 8 * sizeof(uint16_t)) {
    const __m256 va = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) a));
    a += 8;

    __m256 vy = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_add_ps(va, vb), _MM_FROUND_TO_NEAREST_INT));

    vy = _mm256_max_ps(vy, vy_min);
    vy = _mm256_min_ps(vy, vy_max);

    _mm_storeu_si128((__m128i*) o, _mm256_cvtps_ph(vy, _MM_FROUND_TO_NEAREST_INT));
    o += 8;
  }
  if XNN_UNLIKELY(batch != 0) {
    const __m256 va = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) a));

    __m256 vy = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_add_ps(va, vb), _MM_FROUND_TO_NEAREST_INT));

    vy = _mm256_max_ps(vy, vy_min);
    vy = _mm256_min_ps(vy, vy_max);

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

void xnn_f16_vdiv_minmax_ukernel__f16c_u8(
    size_t batch,
    const void* restrict input_a,
    const void* restrict input_b,
    void* restrict output,
    const union xnn_f16_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(uint16_t) == 0);
  assert(input_a != NULL);
  assert(input_b != NULL);
  assert(output != NULL);

  const uint16_t* a = (const uint16_t*) input_a;
  const uint16_t* b = (const uint16_t*) input_b;
  uint16_t* o = (uint16_t*) output;

  const __m256 vy_min = _mm256_set1_ps(params->avx.min);
  const __m256 vy_max = _mm256_set1_ps(params->avx.max);
  XNN_FORCE_REALIZATION(vy_min);
  XNN_FORCE_REALIZATION(vy_max);

  for (; batch >= 8 * sizeof(uint16_t); batch -= 8 * sizeof(uint16_t)) {
    const __m256 va = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) a));
    const __m256 vb = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) b));
    a += 8;
    b += 8;

    __m256 vy = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_div_ps(va, vb), _MM_FROUND_TO_NEAREST_INT));

    vy = _mm256_max_ps(vy, vy_min);
    vy = _mm256_min_ps(vy, vy_max);

    _mm_storeu_si128((__m128i*) o, _mm256_cvtps_ph(vy, _MM_FROUND_TO_NEAREST_INT));
    o += 8;
  }
  if XNN_UNLIKELY(batch != 0) {
    const __m256 va = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) a));
    const __m256 vb = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) b));

    __m256 vy = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_div_ps(va, vb), _MM_FROUND_TO_NEAREST_INT));

    vy = _mm256_max_ps(vy, vy_min);
    vy = _mm256_min_ps(vy, vy_max);

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

void xnn_f16_vdivc_minmax_ukernel__f16c_u8(
    size_t batch,
    const void* restrict input_a,
    const void* restrict input_b,
    void* restrict output,
    const union xnn_f16_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(uint16_t) == 0);
  assert(input_a != NULL);
  assert(input_b != NULL);
  assert(output != NULL);

  const uint16_t* a = (const uint16_t*) input_a;
  const uint16_t* b = (const uint16_t*) input_b;
  uint16_t* o = (uint16_t*) output;

  const __m256 vy_min = _mm256_set1_ps(params->avx.min);
  const __m256 vy_max = _mm256_set1_ps(params->avx.max);
  XNN_FORCE_REALIZATION(vy_min);
  XNN_FORCE_REALIZATION(vy_max);

  const __m256 vb = _mm256_cvtph_ps(_mm_set1_epi16((short) *b));
  for (; batch >= 8 * sizeof(uint16_t); batch -= 8 * sizeof(uint16_t)) {
    const __m256 va = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) a));
    a += 8;

    __m256 vy = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_div_ps(va, vb), _MM_FROUND_TO_NEAREST_INT));

    vy = _mm256_max_ps(vy, vy_min);
    vy = _mm256_min_ps(vy, vy_max);

    _mm_storeu_si128((__m128i*) o, _mm256_cvtps_ph(vy, _MM_FROUND_TO_NEAREST_INT));
    o += 8;
  }
  if XNN_UNLIKELY(batch != 0) {
    const __m256 va = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) a));

    __m256 vy = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_div_ps(va, vb), _MM_FROUND_TO_NEAREST_INT));

    vy = _mm256_max_ps(vy, vy_min);
    vy = _mm256_min_ps(vy, vy_max);

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

void xnn_f16_vmax_ukernel__f16c_u16(
    size_t batch,
    const void* restrict input_a,
    const void* restrict input_b,
    void* restrict output,
    const union xnn_f16_default_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(uint16_t) == 0);
  assert(input_a != NULL);
  assert(input_b != NULL);
  assert(output != NULL);

  const uint16_t* a = (const uint16_t*) input_a;
  const uint16_t* b = (const uint16_t*) input_b;
  uint16_t* o = (uint16_t*) output;


  for (; batch >= 16 * sizeof(uint16_t); batch -= 16 * sizeof(uint16_t)) {
    const __m256 va01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) a));
    const __m256 vb01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) b));
    const __m256 va456789AB = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (a + 8)));
    const __m256 vb456789AB = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (b + 8)));
    a += 16;
    b += 16;

    __m256 vy01234567 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_max_ps(va01234567, vb01234567), _MM_FROUND_TO_NEAREST_INT));
    __m256 vy456789AB = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_max_ps(va456789AB, vb456789AB), _MM_FROUND_TO_NEAREST_INT));



    _mm_storeu_si128((__m128i*) o, _mm256_cvtps_ph(vy01234567, _MM_FROUND_TO_NEAREST_INT));
    _mm_storeu_si128((__m128i*) (o + 8), _mm256_cvtps_ph(vy456789AB, _MM_FROUND_TO_NEAREST_INT));
    o += 16;
  }
  for (; batch >= 8 * sizeof(uint16_t); batch -= 8 * sizeof(uint16_t)) {
    const __m256 va = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) a));
    const __m256 vb = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) b));
    a += 8;
    b += 8;

    __m256 vy = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_max_ps(va, vb), _MM_FROUND_TO_NEAREST_INT));


    _mm_storeu_si128((__m128i*) o, _mm256_cvtps_ph(vy, _MM_FROUND_TO_NEAREST_INT));
    o += 8;
  }
  if XNN_UNLIKELY(batch != 0) {
    const __m256 va = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) a));
    const __m256 vb = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) b));

    __m256 vy = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_max_ps(va, vb), _MM_FROUND_TO_NEAREST_INT));


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

void xnn_f16_vmaxc_ukernel__f16c_u16(
    size_t batch,
    const void* restrict input_a,
    const void* restrict input_b,
    void* restrict output,
    const union xnn_f16_default_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(uint16_t) == 0);
  assert(input_a != NULL);
  assert(input_b != NULL);
  assert(output != NULL);

  const uint16_t* a = (const uint16_t*) input_a;
  const uint16_t* b = (const uint16_t*) input_b;
  uint16_t* o = (uint16_t*) output;


  const __m256 vb = _mm256_cvtph_ps(_mm_set1_epi16((short) *b));
  for (; batch >= 16 * sizeof(uint16_t); batch -= 16 * sizeof(uint16_t)) {
    const __m256 va01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) a));
    const __m256 va456789AB = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (a + 8)));
    a += 16;

    __m256 vy01234567 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_max_ps(va01234567, vb), _MM_FROUND_TO_NEAREST_INT));
    __m256 vy456789AB = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_max_ps(va456789AB, vb), _MM_FROUND_TO_NEAREST_INT));



    _mm_storeu_si128((__m128i*) o, _mm256_cvtps_ph(vy01234567, _MM_FROUND_TO_NEAREST_INT));
    _mm_storeu_si128((__m128i*) (o + 8), _mm256_cvtps_ph(vy456789AB, _MM_FROUND_TO_NEAREST_INT));
    o += 16;
  }
  for (; batch >= 8 * sizeof(uint16_t); batch -= 8 * sizeof(uint16_t)) {
    const __m256 va = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) a));
    a += 8;

    __m256 vy = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_max_ps(va, vb), _MM_FROUND_TO_NEAREST_INT));


    _mm_storeu_si128((__m128i*) o, _mm256_cvtps_ph(vy, _MM_FROUND_TO_NEAREST_INT));
    o += 8;
  }
  if XNN_UNLIKELY(batch != 0) {
    const __m256 va = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) a));

    __m256 vy = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_max_ps(va, vb), _MM_FROUND_TO_NEAREST_INT));


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

void xnn_f16_vmin_ukernel__f16c_u16(
    size_t batch,
    const void* restrict input_a,
    const void* restrict input_b,
    void* restrict output,
    const union xnn_f16_default_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(uint16_t) == 0);
  assert(input_a != NULL);
  assert(input_b != NULL);
  assert(output != NULL);

  const uint16_t* a = (const uint16_t*) input_a;
  const uint16_t* b = (const uint16_t*) input_b;
  uint16_t* o = (uint16_t*) output;


  for (; batch >= 16 * sizeof(uint16_t); batch -= 16 * sizeof(uint16_t)) {
    const __m256 va01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) a));
    const __m256 vb01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) b));
    const __m256 va456789AB = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (a + 8)));
    const __m256 vb456789AB = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (b + 8)));
    a += 16;
    b += 16;

    __m256 vy01234567 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_min_ps(va01234567, vb01234567), _MM_FROUND_TO_NEAREST_INT));
    __m256 vy456789AB = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_min_ps(va456789AB, vb456789AB), _MM_FROUND_TO_NEAREST_INT));



    _mm_storeu_si128((__m128i*) o, _mm256_cvtps_ph(vy01234567, _MM_FROUND_TO_NEAREST_INT));
    _mm_storeu_si128((__m128i*) (o + 8), _mm256_cvtps_ph(vy456789AB, _MM_FROUND_TO_NEAREST_INT));
    o += 16;
  }
  for (; batch >= 8 * sizeof(uint16_t); batch -= 8 * sizeof(uint16_t)) {
    const __m256 va = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) a));
    const __m256 vb = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) b));
    a += 8;
    b += 8;

    __m256 vy = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_min_ps(va, vb), _MM_FROUND_TO_NEAREST_INT));


    _mm_storeu_si128((__m128i*) o, _mm256_cvtps_ph(vy, _MM_FROUND_TO_NEAREST_INT));
    o += 8;
  }
  if XNN_UNLIKELY(batch != 0) {
    const __m256 va = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) a));
    const __m256 vb = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) b));

    __m256 vy = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_min_ps(va, vb), _MM_FROUND_TO_NEAREST_INT));


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

void xnn_f16_vminc_ukernel__f16c_u16(
    size_t batch,
    const void* restrict input_a,
    const void* restrict input_b,
    void* restrict output,
    const union xnn_f16_default_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(uint16_t) == 0);
  assert(input_a != NULL);
  assert(input_b != NULL);
  assert(output != NULL);

  const uint16_t* a = (const uint16_t*) input_a;
  const uint16_t* b = (const uint16_t*) input_b;
  uint16_t* o = (uint16_t*) output;


  const __m256 vb = _mm256_cvtph_ps(_mm_set1_epi16((short) *b));
  for (; batch >= 16 * sizeof(uint16_t); batch -= 16 * sizeof(uint16_t)) {
    const __m256 va01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) a));
    const __m256 va456789AB = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (a + 8)));
    a += 16;

    __m256 vy01234567 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_min_ps(va01234567, vb), _MM_FROUND_TO_NEAREST_INT));
    __m256 vy456789AB = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_min_ps(va456789AB, vb), _MM_FROUND_TO_NEAREST_INT));



    _mm_storeu_si128((__m128i*) o, _mm256_cvtps_ph(vy01234567, _MM_FROUND_TO_NEAREST_INT));
    _mm_storeu_si128((__m128i*) (o + 8), _mm256_cvtps_ph(vy456789AB, _MM_FROUND_TO_NEAREST_INT));
    o += 16;
  }
  for (; batch >= 8 * sizeof(uint16_t); batch -= 8 * sizeof(uint16_t)) {
    const __m256 va = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) a));
    a += 8;

    __m256 vy = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_min_ps(va, vb), _MM_FROUND_TO_NEAREST_INT));


    _mm_storeu_si128((__m128i*) o, _mm256_cvtps_ph(vy, _MM_FROUND_TO_NEAREST_INT));
    o += 8;
  }
  if XNN_UNLIKELY(batch != 0) {
    const __m256 va = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) a));

    __m256 vy = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_min_ps(va, vb), _MM_FROUND_TO_NEAREST_INT));


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

void xnn_f16_vmul_minmax_ukernel__f16c_u16(
    size_t batch,
    const void* restrict input_a,
    const void* restrict input_b,
    void* restrict output,
    const union xnn_f16_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(uint16_t) == 0);
  assert(input_a != NULL);
  assert(input_b != NULL);
  assert(output != NULL);

  const uint16_t* a = (const uint16_t*) input_a;
  const uint16_t* b = (const uint16_t*) input_b;
  uint16_t* o = (uint16_t*) output;

  const __m256 vy_min = _mm256_set1_ps(params->avx.min);
  const __m256 vy_max = _mm256_set1_ps(params->avx.max);
  XNN_FORCE_REALIZATION(vy_min);
  XNN_FORCE_REALIZATION(vy_max);

  for (; batch >= 16 * sizeof(uint16_t); batch -= 16 * sizeof(uint16_t)) {
    const __m256 va01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) a));
    const __m256 vb01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) b));
    const __m256 va456789AB = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (a + 8)));
    const __m256 vb456789AB = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (b + 8)));
    a += 16;
    b += 16;

    __m256 vy01234567 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_mul_ps(va01234567, vb01234567), _MM_FROUND_TO_NEAREST_INT));
    __m256 vy456789AB = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_mul_ps(va456789AB, vb456789AB), _MM_FROUND_TO_NEAREST_INT));


    vy01234567 = _mm256_max_ps(vy01234567, vy_min);
    vy456789AB = _mm256_max_ps(vy456789AB, vy_min);

    vy01234567 = _mm256_min_ps(vy01234567, vy_max);
    vy456789AB = _mm256_min_ps(vy456789AB, vy_max);

    _mm_storeu_si128((__m128i*) o, _mm256_cvtps_ph(vy01234567, _MM_FROUND_TO_NEAREST_INT));
    _mm_storeu_si128((__m128i*) (o + 8), _mm256_cvtps_ph(vy456789AB, _MM_FROUND_TO_NEAREST_INT));
    o += 16;
  }
  for (; batch >= 8 * sizeof(uint16_t); batch -= 8 * sizeof(uint16_t)) {
    const __m256 va = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) a));
    const __m256 vb = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) b));
    a += 8;
    b += 8;

    __m256 vy = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_mul_ps(va, vb), _MM_FROUND_TO_NEAREST_INT));

    vy = _mm256_max_ps(vy, vy_min);
    vy = _mm256_min_ps(vy, vy_max);

    _mm_storeu_si128((__m128i*) o, _mm256_cvtps_ph(vy, _MM_FROUND_TO_NEAREST_INT));
    o += 8;
  }
  if XNN_UNLIKELY(batch != 0) {
    const __m256 va = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) a));
    const __m256 vb = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) b));

    __m256 vy = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_mul_ps(va, vb), _MM_FROUND_TO_NEAREST_INT));

    vy = _mm256_max_ps(vy, vy_min);
    vy = _mm256_min_ps(vy, vy_max);

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

void xnn_f16_vmulc_minmax_ukernel__f16c_u16(
    size_t batch,
    const void* restrict input_a,
    const void* restrict input_b,
    void* restrict output,
    const union xnn_f16_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(uint16_t) == 0);
  assert(input_a != NULL);
  assert(input_b != NULL);
  assert(output != NULL);

  const uint16_t* a = (const uint16_t*) input_a;
  const uint16_t* b = (const uint16_t*) input_b;
  uint16_t* o = (uint16_t*) output;

  const __m256 vy_min = _mm256_set1_ps(params->avx.min);
  const __m256 vy_max = _mm256_set1_ps(params->avx.max);
  XNN_FORCE_REALIZATION(vy_min);
  XNN_FORCE_REALIZATION(vy_max);

  const __m256 vb = _mm256_cvtph_ps(_mm_set1_epi16((short) *b));
  for (; batch >= 16 * sizeof(uint16_t); batch -= 16 * sizeof(uint16_t)) {
    const __m256 va01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) a));
    const __m256 va456789AB = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (a + 8)));
    a += 16;

    __m256 vy01234567 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_mul_ps(va01234567, vb), _MM_FROUND_TO_NEAREST_INT));
    __m256 vy456789AB = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_mul_ps(va456789AB, vb), _MM_FROUND_TO_NEAREST_INT));


    vy01234567 = _mm256_max_ps(vy01234567, vy_min);
    vy456789AB = _mm256_max_ps(vy456789AB, vy_min);

    vy01234567 = _mm256_min_ps(vy01234567, vy_max);
    vy456789AB = _mm256_min_ps(vy456789AB, vy_max);

    _mm_storeu_si128((__m128i*) o, _mm256_cvtps_ph(vy01234567, _MM_FROUND_TO_NEAREST_INT));
    _mm_storeu_si128((__m128i*) (o + 8), _mm256_cvtps_ph(vy456789AB, _MM_FROUND_TO_NEAREST_INT));
    o += 16;
  }
  for (; batch >= 8 * sizeof(uint16_t); batch -= 8 * sizeof(uint16_t)) {
    const __m256 va = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) a));
    a += 8;

    __m256 vy = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_mul_ps(va, vb), _MM_FROUND_TO_NEAREST_INT));

    vy = _mm256_max_ps(vy, vy_min);
    vy = _mm256_min_ps(vy, vy_max);

    _mm_storeu_si128((__m128i*) o, _mm256_cvtps_ph(vy, _MM_FROUND_TO_NEAREST_INT));
    o += 8;
  }
  if XNN_UNLIKELY(batch != 0) {
    const __m256 va = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) a));

    __m256 vy = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_mul_ps(va, vb), _MM_FROUND_TO_NEAREST_INT));

    vy = _mm256_max_ps(vy, vy_min);
    vy = _mm256_min_ps(vy, vy_max);

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

void xnn_f16_vrdivc_minmax_ukernel__f16c_u8(
    size_t batch,
    const void* restrict input_a,
    const void* restrict input_b,
    void* restrict output,
    const union xnn_f16_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(uint16_t) == 0);
  assert(input_a != NULL);
  assert(input_b != NULL);
  assert(output != NULL);

  const uint16_t* a = (const uint16_t*) input_a;
  const uint16_t* b = (const uint16_t*) input_b;
  uint16_t* o = (uint16_t*) output;

  const __m256 vy_min = _mm256_set1_ps(params->avx.min);
  const __m256 vy_max = _mm256_set1_ps(params->avx.max);
  XNN_FORCE_REALIZATION(vy_min);
  XNN_FORCE_REALIZATION(vy_max);

  const __m256 vb = _mm256_cvtph_ps(_mm_set1_epi16((short) *b));
  for (; batch >= 8 * sizeof(uint16_t); batch -= 8 * sizeof(uint16_t)) {
    const __m256 va = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) a));
    a += 8;

    __m256 vy = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_div_ps(vb, va), _MM_FROUND_TO_NEAREST_INT));

    vy = _mm256_max_ps(vy, vy_min);
    vy = _mm256_min_ps(vy, vy_max);

    _mm_storeu_si128((__m128i*) o, _mm256_cvtps_ph(vy, _MM_FROUND_TO_NEAREST_INT));
    o += 8;
  }
  if XNN_UNLIKELY(batch != 0) {
    const __m256 va = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) a));

    __m256 vy = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_div_ps(vb, va), _MM_FROUND_TO_NEAREST_INT));

    vy = _mm256_max_ps(vy, vy_min);
    vy = _mm256_min_ps(vy, vy_max);

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

void xnn_f16_vrsubc_minmax_ukernel__f16c_u16(
    size_t batch,
    const void* restrict input_a,
    const void* restrict input_b,
    void* restrict output,
    const union xnn_f16_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(uint16_t) == 0);
  assert(input_a != NULL);
  assert(input_b != NULL);
  assert(output != NULL);

  const uint16_t* a = (const uint16_t*) input_a;
  const uint16_t* b = (const uint16_t*) input_b;
  uint16_t* o = (uint16_t*) output;

  const __m256 vy_min = _mm256_set1_ps(params->avx.min);
  const __m256 vy_max = _mm256_set1_ps(params->avx.max);
  XNN_FORCE_REALIZATION(vy_min);
  XNN_FORCE_REALIZATION(vy_max);

  const __m256 vb = _mm256_cvtph_ps(_mm_set1_epi16((short) *b));
  for (; batch >= 16 * sizeof(uint16_t); batch -= 16 * sizeof(uint16_t)) {
    const __m256 va01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) a));
    const __m256 va456789AB = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (a + 8)));
    a += 16;

    __m256 vy01234567 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_sub_ps(vb, va01234567), _MM_FROUND_TO_NEAREST_INT));
    __m256 vy456789AB = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_sub_ps(vb, va456789AB), _MM_FROUND_TO_NEAREST_INT));


    vy01234567 = _mm256_max_ps(vy01234567, vy_min);
    vy456789AB = _mm256_max_ps(vy456789AB, vy_min);

    vy01234567 = _mm256_min_ps(vy01234567, vy_max);
    vy456789AB = _mm256_min_ps(vy456789AB, vy_max);

    _mm_storeu_si128((__m128i*) o, _mm256_cvtps_ph(vy01234567, _MM_FROUND_TO_NEAREST_INT));
    _mm_storeu_si128((__m128i*) (o + 8), _mm256_cvtps_ph(vy456789AB, _MM_FROUND_TO_NEAREST_INT));
    o += 16;
  }
  for (; batch >= 8 * sizeof(uint16_t); batch -= 8 * sizeof(uint16_t)) {
    const __m256 va = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) a));
    a += 8;

    __m256 vy = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_sub_ps(vb, va), _MM_FROUND_TO_NEAREST_INT));

    vy = _mm256_max_ps(vy, vy_min);
    vy = _mm256_min_ps(vy, vy_max);

    _mm_storeu_si128((__m128i*) o, _mm256_cvtps_ph(vy, _MM_FROUND_TO_NEAREST_INT));
    o += 8;
  }
  if XNN_UNLIKELY(batch != 0) {
    const __m256 va = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) a));

    __m256 vy = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_sub_ps(vb, va), _MM_FROUND_TO_NEAREST_INT));

    vy = _mm256_max_ps(vy, vy_min);
    vy = _mm256_min_ps(vy, vy_max);

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

void xnn_f16_vsqrdiff_ukernel__f16c_u16(
    size_t batch,
    const void* restrict input_a,
    const void* restrict input_b,
    void* restrict output,
    const union xnn_f16_default_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(uint16_t) == 0);
  assert(input_a != NULL);
  assert(input_b != NULL);
  assert(output != NULL);

  const uint16_t* a = (const uint16_t*) input_a;
  const uint16_t* b = (const uint16_t*) input_b;
  uint16_t* o = (uint16_t*) output;


  for (; batch >= 16 * sizeof(uint16_t); batch -= 16 * sizeof(uint16_t)) {
    const __m256 va01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) a));
    const __m256 vb01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) b));
    const __m256 va456789AB = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (a + 8)));
    const __m256 vb456789AB = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (b + 8)));
    a += 16;
    b += 16;

    __m256 vy01234567 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_sub_ps(va01234567, vb01234567), _MM_FROUND_TO_NEAREST_INT));
    __m256 vy456789AB = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_sub_ps(va456789AB, vb456789AB), _MM_FROUND_TO_NEAREST_INT));

    vy01234567 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_mul_ps(vy01234567, vy01234567), _MM_FROUND_TO_NEAREST_INT));
    vy456789AB = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_mul_ps(vy456789AB, vy456789AB), _MM_FROUND_TO_NEAREST_INT));


    _mm_storeu_si128((__m128i*) o, _mm256_cvtps_ph(vy01234567, _MM_FROUND_TO_NEAREST_INT));
    _mm_storeu_si128((__m128i*) (o + 8), _mm256_cvtps_ph(vy456789AB, _MM_FROUND_TO_NEAREST_INT));
    o += 16;
  }
  for (; batch >= 8 * sizeof(uint16_t); batch -= 8 * sizeof(uint16_t)) {
    const __m256 va = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) a));
    const __m256 vb = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) b));
    a += 8;
    b += 8;

    __m256 vy = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_sub_ps(va, vb), _MM_FROUND_TO_NEAREST_INT));
    vy = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_mul_ps(vy, vy), _MM_FROUND_TO_NEAREST_INT));


    _mm_storeu_si128((__m128i*) o, _mm256_cvtps_ph(vy, _MM_FROUND_TO_NEAREST_INT));
    o += 8;
  }
  if XNN_UNLIKELY(batch != 0) {
    const __m256 va = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) a));
    const __m256 vb = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) b));

    __m256 vy = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_sub_ps(va, vb), _MM_FROUND_TO_NEAREST_INT));
    vy = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_mul_ps(vy, vy), _MM_FROUND_TO_NEAREST_INT));


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

void xnn_f16_vsqrdiffc_ukernel__f16c_u16(
    size_t batch,
    const void* restrict input_a,
    const void* restrict input_b,
    void* restrict output,
    const union xnn_f16_default_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(uint16_t) == 0);
  assert(input_a != NULL);
  assert(input_b != NULL);
  assert(output != NULL);

  const uint16_t* a = (const uint16_t*) input_a;
  const uint16_t* b = (const uint16_t*) input_b;
  uint16_t* o = (uint16_t*) output;


  const __m256 vb = _mm256_cvtph_ps(_mm_set1_epi16((short) *b));
  for (; batch >= 16 * sizeof(uint16_t); batch -= 16 * sizeof(uint16_t)) {
    const __m256 va01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) a));
    const __m256 va456789AB = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (a + 8)));
    a += 16;

    __m256 vy01234567 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_sub_ps(va01234567, vb), _MM_FROUND_TO_NEAREST_INT));
    __m256 vy456789AB = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_sub_ps(va456789AB, vb), _MM_FROUND_TO_NEAREST_INT));

    vy01234567 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_mul_ps(vy01234567, vy01234567), _MM_FROUND_TO_NEAREST_INT));
    vy456789AB = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_mul_ps(vy456789AB, vy456789AB), _MM_FROUND_TO_NEAREST_INT));


    _mm_storeu_si128((__m128i*) o, _mm256_cvtps_ph(vy01234567, _MM_FROUND_TO_NEAREST_INT));
    _mm_storeu_si128((__m128i*) (o + 8), _mm256_cvtps_ph(vy456789AB, _MM_FROUND_TO_NEAREST_INT));
    o += 16;
  }
  for (; batch >= 8 * sizeof(uint16_t); batch -= 8 * sizeof(uint16_t)) {
    const __m256 va = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) a));
    a += 8;

    __m256 vy = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_sub_ps(va, vb), _MM_FROUND_TO_NEAREST_INT));
    vy = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_mul_ps(vy, vy), _MM_FROUND_TO_NEAREST_INT));


    _mm_storeu_si128((__m128i*) o, _mm256_cvtps_ph(vy, _MM_FROUND_TO_NEAREST_INT));
    o += 8;
  }
  if XNN_UNLIKELY(batch != 0) {
    const __m256 va = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) a));

    __m256 vy = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_sub_ps(va, vb), _MM_FROUND_TO_NEAREST_INT));
    vy = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_mul_ps(vy, vy), _MM_FROUND_TO_NEAREST_INT));


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

void xnn_f16_vsub_minmax_ukernel__f16c_u16(
    size_t batch,
    const void* restrict input_a,
    const void* restrict input_b,
    void* restrict output,
    const union xnn_f16_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(uint16_t) == 0);
  assert(input_a != NULL);
  assert(input_b != NULL);
  assert(output != NULL);

  const uint16_t* a = (const uint16_t*) input_a;
  const uint16_t* b = (const uint16_t*) input_b;
  uint16_t* o = (uint16_t*) output;

  const __m256 vy_min = _mm256_set1_ps(params->avx.min);
  const __m256 vy_max = _mm256_set1_ps(params->avx.max);
  XNN_FORCE_REALIZATION(vy_min);
  XNN_FORCE_REALIZATION(vy_max);

  for (; batch >= 16 * sizeof(uint16_t); batch -= 16 * sizeof(uint16_t)) {
    const __m256 va01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) a));
    const __m256 vb01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) b));
    const __m256 va456789AB = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (a + 8)));
    const __m256 vb456789AB = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (b + 8)));
    a += 16;
    b += 16;

    __m256 vy01234567 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_sub_ps(va01234567, vb01234567), _MM_FROUND_TO_NEAREST_INT));
    __m256 vy456789AB = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_sub_ps(va456789AB, vb456789AB), _MM_FROUND_TO_NEAREST_INT));


    vy01234567 = _mm256_max_ps(vy01234567, vy_min);
    vy456789AB = _mm256_max_ps(vy456789AB, vy_min);

    vy01234567 = _mm256_min_ps(vy01234567, vy_max);
    vy456789AB = _mm256_min_ps(vy456789AB, vy_max);

    _mm_storeu_si128((__m128i*) o, _mm256_cvtps_ph(vy01234567, _MM_FROUND_TO_NEAREST_INT));
    _mm_storeu_si128((__m128i*) (o + 8), _mm256_cvtps_ph(vy456789AB, _MM_FROUND_TO_NEAREST_INT));
    o += 16;
  }
  for (; batch >= 8 * sizeof(uint16_t); batch -= 8 * sizeof(uint16_t)) {
    const __m256 va = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) a));
    const __m256 vb = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) b));
    a += 8;
    b += 8;

    __m256 vy = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_sub_ps(va, vb), _MM_FROUND_TO_NEAREST_INT));

    vy = _mm256_max_ps(vy, vy_min);
    vy = _mm256_min_ps(vy, vy_max);

    _mm_storeu_si128((__m128i*) o, _mm256_cvtps_ph(vy, _MM_FROUND_TO_NEAREST_INT));
    o += 8;
  }
  if XNN_UNLIKELY(batch != 0) {
    const __m256 va = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) a));
    const __m256 vb = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) b));

    __m256 vy = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_sub_ps(va, vb), _MM_FROUND_TO_NEAREST_INT));

    vy = _mm256_max_ps(vy, vy_min);
    vy = _mm256_min_ps(vy, vy_max);

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

void xnn_f16_vsubc_minmax_ukernel__f16c_u16(
    size_t batch,
    const void* restrict input_a,
    const void* restrict input_b,
    void* restrict output,
    const union xnn_f16_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(uint16_t) == 0);
  assert(input_a != NULL);
  assert(input_b != NULL);
  assert(output != NULL);

  const uint16_t* a = (const uint16_t*) input_a;
  const uint16_t* b = (const uint16_t*) input_b;
  uint16_t* o = (uint16_t*) output;

  const __m256 vy_min = _mm256_set1_ps(params->avx.min);
  const __m256 vy_max = _mm256_set1_ps(params->avx.max);
  XNN_FORCE_REALIZATION(vy_min);
  XNN_FORCE_REALIZATION(vy_max);

  const __m256 vb = _mm256_cvtph_ps(_mm_set1_epi16((short) *b));
  for (; batch >= 16 * sizeof(uint16_t); batch -= 16 * sizeof(uint16_t)) {
    const __m256 va01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) a));
    const __m256 va456789AB = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (a + 8)));
    a += 16;

    __m256 vy01234567 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_sub_ps(va01234567, vb), _MM_FROUND_TO_NEAREST_INT));
    __m256 vy456789AB = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_sub_ps(va456789AB, vb), _MM_FROUND_TO_NEAREST_INT));


    vy01234567 = _mm256_max_ps(vy01234567, vy_min);
    vy456789AB = _mm256_max_ps(vy456789AB, vy_min);

    vy01234567 = _mm256_min_ps(vy01234567, vy_max);
    vy456789AB = _mm256_min_ps(vy456789AB, vy_max);

    _mm_storeu_si128((__m128i*) o, _mm256_cvtps_ph(vy01234567, _MM_FROUND_TO_NEAREST_INT));
    _mm_storeu_si128((__m128i*) (o + 8), _mm256_cvtps_ph(vy456789AB, _MM_FROUND_TO_NEAREST_INT));
    o += 16;
  }
  for (; batch >= 8 * sizeof(uint16_t); batch -= 8 * sizeof(uint16_t)) {
    const __m256 va = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) a));
    a += 8;

    __m256 vy = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_sub_ps(va, vb), _MM_FROUND_TO_NEAREST_INT));

    vy = _mm256_max_ps(vy, vy_min);
    vy = _mm256_min_ps(vy, vy_max);

    _mm_storeu_si128((__m128i*) o, _mm256_cvtps_ph(vy, _MM_FROUND_TO_NEAREST_INT));
    o += 8;
  }
  if XNN_UNLIKELY(batch != 0) {
    const __m256 va = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) a));

    __m256 vy = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_sub_ps(va, vb), _MM_FROUND_TO_NEAREST_INT));

    vy = _mm256_max_ps(vy, vy_min);
    vy = _mm256_min_ps(vy, vy_max);

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

void xnn_f16_vclamp_ukernel__f16c_u16(
    size_t batch,
    const void* restrict input,
    void* restrict output,
    const union xnn_f16_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(uint16_t) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const uint16_t* i = (const uint16_t*) input;
  uint16_t* o = (uint16_t*) output;

  const __m256 vy_min = _mm256_set1_ps(params->avx.min);
  const __m256 vy_max = _mm256_set1_ps(params->avx.max);

  for (; batch >= 16 * sizeof(uint16_t); batch -= 16 * sizeof(uint16_t)) {
    __m256 vacc01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i));
    __m256 vacc89ABCDEF = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i + 8)));
    i += 16;

    vacc01234567 = _mm256_max_ps(vacc01234567, vy_min);
    vacc89ABCDEF = _mm256_max_ps(vacc89ABCDEF, vy_min);

    vacc01234567 = _mm256_min_ps(vacc01234567, vy_max);
    vacc89ABCDEF = _mm256_min_ps(vacc89ABCDEF, vy_max);

    _mm_storeu_si128((__m128i*) o, _mm256_cvtps_ph(vacc01234567, _MM_FROUND_TO_NEAREST_INT));
    _mm_storeu_si128((__m128i*) (o + 8), _mm256_cvtps_ph(vacc89ABCDEF, _MM_FROUND_TO_NEAREST_INT));
    o += 16;
  }
  for (; batch >= 8 * sizeof(uint16_t); batch -= 8 * sizeof(uint16_t)) {
    __m256 vacc = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i));
    i += 8;
    vacc = _mm256_max_ps(vacc, vy_min);
    vacc = _mm256_min_ps(vacc, vy_max);
    _mm_storeu_si128((__m128i*) o, _mm256_cvtps_ph(vacc, _MM_FROUND_TO_NEAREST_INT));
    o += 8;
  }
  if XNN_UNLIKELY(batch != 0) {
    __m256 vacc = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i));
    vacc = _mm256_max_ps(vacc, vy_min);
    vacc = _mm256_min_ps(vacc, vy_max);

    __m128i vh = _mm256_cvtps_ph(vacc, _MM_FROUND_TO_NEAREST_INT);
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
      *o = _mm_extract_epi16(vh, 0);
    }
  }
}

void xnn_f16_vhswish_ukernel__f16c_u16(
    size_t batch,
    const void* restrict input,
    void* restrict output,
    const union xnn_f16_hswish_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(uint16_t) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const uint16_t* i = (const uint16_t*) input;
  uint16_t* o = (uint16_t*) output;

  const __m256 vsixth = _mm256_set1_ps(0x1.554000p-3f);
  const __m256 vthree = _mm256_set1_ps(3.0f);
  const __m128i vsix = _mm_set1_epi16(UINT16_C(0x4600));
  const __m128i vzero = _mm_setzero_si128();

  XNN_FORCE_REALIZATION(vsixth);
  XNN_FORCE_REALIZATION(vthree);
  XNN_FORCE_REALIZATION(vsix);
  // XNN_FORCE_REALIZATION(vzero);

  for (; batch >= 16 * sizeof(uint16_t); batch -= 16 * sizeof(uint16_t)) {
    __m256 vx01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i));
    __m256 vx89ABCDEF = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i + 8)));
    i += 16;

    __m128i vacc01234567 = _mm256_cvtps_ph(_mm256_add_ps(vx01234567, vthree), _MM_FROUND_TO_NEAREST_INT);
    vx01234567 = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_mul_ps(vx01234567, vsixth), _MM_FROUND_TO_NEAREST_INT));
    __m128i vacc89ABCDEF = _mm256_cvtps_ph(_mm256_add_ps(vx89ABCDEF, vthree), _MM_FROUND_TO_NEAREST_INT);
    vx89ABCDEF = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_mul_ps(vx89ABCDEF, vsixth), _MM_FROUND_TO_NEAREST_INT));

    vacc01234567 = _mm_max_epi16(vacc01234567, vzero);
    vacc89ABCDEF = _mm_max_epi16(vacc89ABCDEF, vzero);

    vacc01234567 = _mm_min_epi16(vacc01234567, vsix);
    vacc89ABCDEF = _mm_min_epi16(vacc89ABCDEF, vsix);

    vacc01234567 = _mm256_cvtps_ph(_mm256_mul_ps(_mm256_cvtph_ps(vacc01234567), vx01234567), _MM_FROUND_TO_NEAREST_INT);
    vacc89ABCDEF = _mm256_cvtps_ph(_mm256_mul_ps(_mm256_cvtph_ps(vacc89ABCDEF), vx89ABCDEF), _MM_FROUND_TO_NEAREST_INT);

    _mm_storeu_si128((__m128i*) o, vacc01234567);
    _mm_storeu_si128((__m128i*) (o + 8), vacc89ABCDEF);
    o += 16;
  }
  for (; batch >= 8 * sizeof(uint16_t); batch -= 8 * sizeof(uint16_t)) {
    __m256 vx = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i));
    i += 8;
    __m128i vacc = _mm256_cvtps_ph(_mm256_add_ps(vx, vthree), _MM_FROUND_TO_NEAREST_INT);
    vx = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_mul_ps(vx, vsixth), _MM_FROUND_TO_NEAREST_INT));
    vacc = _mm_max_epi16(vacc, vzero);
    vacc = _mm_min_epi16(vacc, vsix);
    vacc = _mm256_cvtps_ph(_mm256_mul_ps(_mm256_cvtph_ps(vacc), vx), _MM_FROUND_TO_NEAREST_INT);
    _mm_storeu_si128((__m128i*) o, vacc);
    o += 8;
  }
  if XNN_UNLIKELY(batch != 0) {
    __m256 vx = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i));
    __m128i vacc = _mm256_cvtps_ph(_mm256_add_ps(vx, vthree), _MM_FROUND_TO_NEAREST_INT);
    vx = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_mul_ps(vx, vsixth), _MM_FROUND_TO_NEAREST_INT));
    vacc = _mm_max_epi16(vacc, vzero);
    vacc = _mm_min_epi16(vacc, vsix);
    vacc = _mm256_cvtps_ph(_mm256_mul_ps(_mm256_cvtph_ps(vacc), vx), _MM_FROUND_TO_NEAREST_INT);

    if (batch & (4 * sizeof(uint16_t))) {
      _mm_storel_epi64((__m128i*) o, vacc);
      vacc = _mm_unpackhi_epi64(vacc, vacc);
      o += 4;
    }
    if (batch & (2 * sizeof(uint16_t))) {
      _mm_storeu_si32(o, vacc);
      vacc = _mm_srli_epi64(vacc, 32);
      o += 2;
    }
    if (batch & (1 * sizeof(uint16_t))) {
      *o = (uint16_t) _mm_extract_epi16(vacc, 0);
    }
  }
}

void xnn_f16_vlrelu_ukernel__f16c_u16(
    size_t batch,
    const void* input,
    void* output,
    const union xnn_f16_lrelu_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(uint16_t) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const __m256 vslope = _mm256_set1_ps(params->avx.slope);
  const uint16_t* i = (const uint16_t*) input;
  uint16_t* o = (uint16_t*) output;
  for (; batch >= 16 * sizeof(uint16_t); batch -= 16 * sizeof(uint16_t)) {
    const __m256 vx01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i));
    const __m256 vx89ABCDEF = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i + 8)));
    i += 16;

    __m256 vacc01234567 = _mm256_mul_ps(vx01234567, vslope);
    __m256 vacc89ABCDEF = _mm256_mul_ps(vx89ABCDEF, vslope);

    vacc01234567 = _mm256_blendv_ps(vx01234567, vacc01234567, vx01234567);
    vacc89ABCDEF = _mm256_blendv_ps(vx89ABCDEF, vacc89ABCDEF, vx89ABCDEF);

    _mm_storeu_si128((__m128i*) o, _mm256_cvtps_ph(vacc01234567, _MM_FROUND_TO_NEAREST_INT));
    _mm_storeu_si128((__m128i*) (o + 8), _mm256_cvtps_ph(vacc89ABCDEF, _MM_FROUND_TO_NEAREST_INT));
    o += 16;
  }
  for (; batch >= 8 * sizeof(uint16_t); batch -= 8 * sizeof(uint16_t)) {
    const __m256 vx = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i));
    i += 8;

    __m256 vacc = _mm256_mul_ps(vx, vslope);
    vacc = _mm256_blendv_ps(vx, vacc, vx);

    _mm_storeu_si128((__m128i*) o, _mm256_cvtps_ph(vacc, _MM_FROUND_TO_NEAREST_INT));
    o += 8;
  }
  if XNN_UNLIKELY(batch != 0) {
    const __m256 vx = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i));

    __m256 vacc = _mm256_mul_ps(vx, vslope);
    vacc = _mm256_blendv_ps(vx, vacc, vx);

    __m128i vh = _mm256_cvtps_ph(vacc, _MM_FROUND_TO_NEAREST_INT);
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
      *o = _mm_extract_epi16(vh, 0);
    }
  }
}

void xnn_f16_vrndd_ukernel__f16c_u16(
    size_t batch,
    const void* input,
    void* output,
    const union xnn_f16_rnd_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(uint16_t) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const uint16_t* i = (const uint16_t*) input;
  uint16_t* o = (uint16_t*) output;
  for (; batch >= 16 * sizeof(uint16_t); batch -= 16 * sizeof(uint16_t)) {
    __m256 vacc0 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i));
    __m256 vacc1 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i + 8)));
    i += 16;

    vacc0 = _mm256_round_ps(vacc0, _MM_FROUND_TO_NEG_INF | _MM_FROUND_TO_NEAREST_INT);
    vacc1 = _mm256_round_ps(vacc1, _MM_FROUND_TO_NEG_INF | _MM_FROUND_TO_NEAREST_INT);

    _mm_storeu_si128((__m128i*) o, _mm256_cvtps_ph(vacc0, _MM_FROUND_TO_NEAREST_INT));
    _mm_storeu_si128((__m128i*) (o + 8), _mm256_cvtps_ph(vacc1, _MM_FROUND_TO_NEAREST_INT));
    o += 16;
  }
  for (; batch >= 8 * sizeof(uint16_t); batch -= 8 * sizeof(uint16_t)) {
    __m256 vacc = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i));
    i += 8;

    vacc = _mm256_round_ps(vacc, _MM_FROUND_TO_NEG_INF | _MM_FROUND_TO_NEAREST_INT);

    _mm_storeu_si128((__m128i*) o, _mm256_cvtps_ph(vacc, _MM_FROUND_TO_NEAREST_INT));
    o += 8;
  }
  if XNN_UNLIKELY(batch != 0) {
    assert(batch >= 1 * sizeof(uint16_t));
    assert(batch <= 7 * sizeof(uint16_t));
    __m256 vacc = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i));
    vacc = _mm256_round_ps(vacc, _MM_FROUND_TO_NEG_INF | _MM_FROUND_TO_NEAREST_INT);
    __m128i vh = _mm256_cvtps_ph(vacc, _MM_FROUND_TO_NEAREST_INT);
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

void xnn_f16_vrndne_ukernel__f16c_u16(
    size_t batch,
    const void* input,
    void* output,
    const union xnn_f16_rnd_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(uint16_t) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const uint16_t* i = (const uint16_t*) input;
  uint16_t* o = (uint16_t*) output;
  for (; batch >= 16 * sizeof(uint16_t); batch -= 16 * sizeof(uint16_t)) {
    __m256 vacc0 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i));
    __m256 vacc1 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i + 8)));
    i += 16;

    vacc0 = _mm256_round_ps(vacc0, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_TO_NEAREST_INT);
    vacc1 = _mm256_round_ps(vacc1, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_TO_NEAREST_INT);

    _mm_storeu_si128((__m128i*) o, _mm256_cvtps_ph(vacc0, _MM_FROUND_TO_NEAREST_INT));
    _mm_storeu_si128((__m128i*) (o + 8), _mm256_cvtps_ph(vacc1, _MM_FROUND_TO_NEAREST_INT));
    o += 16;
  }
  for (; batch >= 8 * sizeof(uint16_t); batch -= 8 * sizeof(uint16_t)) {
    __m256 vacc = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i));
    i += 8;

    vacc = _mm256_round_ps(vacc, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_TO_NEAREST_INT);

    _mm_storeu_si128((__m128i*) o, _mm256_cvtps_ph(vacc, _MM_FROUND_TO_NEAREST_INT));
    o += 8;
  }
  if XNN_UNLIKELY(batch != 0) {
    assert(batch >= 1 * sizeof(uint16_t));
    assert(batch <= 7 * sizeof(uint16_t));
    __m256 vacc = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i));
    vacc = _mm256_round_ps(vacc, _MM_FROUND_TO_NEAREST_INT | _MM_FROUND_TO_NEAREST_INT);
    __m128i vh = _mm256_cvtps_ph(vacc, _MM_FROUND_TO_NEAREST_INT);
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

void xnn_f16_vrndu_ukernel__f16c_u16(
    size_t batch,
    const void* input,
    void* output,
    const union xnn_f16_rnd_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(uint16_t) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const uint16_t* i = (const uint16_t*) input;
  uint16_t* o = (uint16_t*) output;
  for (; batch >= 16 * sizeof(uint16_t); batch -= 16 * sizeof(uint16_t)) {
    __m256 vacc0 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i));
    __m256 vacc1 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i + 8)));
    i += 16;

    vacc0 = _mm256_round_ps(vacc0, _MM_FROUND_TO_POS_INF | _MM_FROUND_TO_NEAREST_INT);
    vacc1 = _mm256_round_ps(vacc1, _MM_FROUND_TO_POS_INF | _MM_FROUND_TO_NEAREST_INT);

    _mm_storeu_si128((__m128i*) o, _mm256_cvtps_ph(vacc0, _MM_FROUND_TO_NEAREST_INT));
    _mm_storeu_si128((__m128i*) (o + 8), _mm256_cvtps_ph(vacc1, _MM_FROUND_TO_NEAREST_INT));
    o += 16;
  }
  for (; batch >= 8 * sizeof(uint16_t); batch -= 8 * sizeof(uint16_t)) {
    __m256 vacc = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i));
    i += 8;

    vacc = _mm256_round_ps(vacc, _MM_FROUND_TO_POS_INF | _MM_FROUND_TO_NEAREST_INT);

    _mm_storeu_si128((__m128i*) o, _mm256_cvtps_ph(vacc, _MM_FROUND_TO_NEAREST_INT));
    o += 8;
  }
  if XNN_UNLIKELY(batch != 0) {
    assert(batch >= 1 * sizeof(uint16_t));
    assert(batch <= 7 * sizeof(uint16_t));
    __m256 vacc = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i));
    vacc = _mm256_round_ps(vacc, _MM_FROUND_TO_POS_INF | _MM_FROUND_TO_NEAREST_INT);
    __m128i vh = _mm256_cvtps_ph(vacc, _MM_FROUND_TO_NEAREST_INT);
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

void xnn_f16_vrndz_ukernel__f16c_u16(
    size_t batch,
    const void* input,
    void* output,
    const union xnn_f16_rnd_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(uint16_t) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const uint16_t* i = (const uint16_t*) input;
  uint16_t* o = (uint16_t*) output;
  for (; batch >= 16 * sizeof(uint16_t); batch -= 16 * sizeof(uint16_t)) {
    __m256 vacc0 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i));
    __m256 vacc1 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i + 8)));
    i += 16;

    vacc0 = _mm256_round_ps(vacc0, _MM_FROUND_TO_ZERO | _MM_FROUND_TO_NEAREST_INT);
    vacc1 = _mm256_round_ps(vacc1, _MM_FROUND_TO_ZERO | _MM_FROUND_TO_NEAREST_INT);

    _mm_storeu_si128((__m128i*) o, _mm256_cvtps_ph(vacc0, _MM_FROUND_TO_NEAREST_INT));
    _mm_storeu_si128((__m128i*) (o + 8), _mm256_cvtps_ph(vacc1, _MM_FROUND_TO_NEAREST_INT));
    o += 16;
  }
  for (; batch >= 8 * sizeof(uint16_t); batch -= 8 * sizeof(uint16_t)) {
    __m256 vacc = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i));
    i += 8;

    vacc = _mm256_round_ps(vacc, _MM_FROUND_TO_ZERO | _MM_FROUND_TO_NEAREST_INT);

    _mm_storeu_si128((__m128i*) o, _mm256_cvtps_ph(vacc, _MM_FROUND_TO_NEAREST_INT));
    o += 8;
  }
  if XNN_UNLIKELY(batch != 0) {
    assert(batch >= 1 * sizeof(uint16_t));
    assert(batch <= 7 * sizeof(uint16_t));
    __m256 vacc = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i));
    vacc = _mm256_round_ps(vacc, _MM_FROUND_TO_ZERO | _MM_FROUND_TO_NEAREST_INT);
    __m128i vh = _mm256_cvtps_ph(vacc, _MM_FROUND_TO_NEAREST_INT);
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

void xnn_f16_vrsqrt_ukernel__f16c_rsqrt_u32(
    size_t batch,
    const void* input,
    void* output,
    const union xnn_f16_rsqrt_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(uint16_t) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const uint16_t* i = (const uint16_t*) input;
  uint16_t* o = (uint16_t*) output;
  for (; batch >= 32 * sizeof(uint16_t); batch -= 32 * sizeof(uint16_t)) {
    __m256 vacc0 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i));
    __m256 vacc1 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i + 8)));
    __m256 vacc2 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i + 16)));
    __m256 vacc3 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i + 24)));
    i += 32;

    vacc0 = _mm256_rsqrt_ps(vacc0);
    vacc1 = _mm256_rsqrt_ps(vacc1);
    vacc2 = _mm256_rsqrt_ps(vacc2);
    vacc3 = _mm256_rsqrt_ps(vacc3);

    _mm_storeu_si128((__m128i*) o, _mm256_cvtps_ph(vacc0, _MM_FROUND_TO_NEAREST_INT));
    _mm_storeu_si128((__m128i*) (o + 8), _mm256_cvtps_ph(vacc1, _MM_FROUND_TO_NEAREST_INT));
    _mm_storeu_si128((__m128i*) (o + 16), _mm256_cvtps_ph(vacc2, _MM_FROUND_TO_NEAREST_INT));
    _mm_storeu_si128((__m128i*) (o + 24), _mm256_cvtps_ph(vacc3, _MM_FROUND_TO_NEAREST_INT));
    o += 32;
  }
  for (; batch >= 8 * sizeof(uint16_t); batch -= 8 * sizeof(uint16_t)) {
    __m256 vacc = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i));
    i += 8;
    vacc = _mm256_rsqrt_ps(vacc);
    _mm_storeu_si128((__m128i*) o, _mm256_cvtps_ph(vacc, _MM_FROUND_TO_NEAREST_INT));
    o += 8;
  }
  if XNN_UNLIKELY(batch != 0) {
    __m256 vacc = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i));
    vacc = _mm256_rsqrt_ps(vacc);
    __m128i vh = _mm256_cvtps_ph(vacc, _MM_FROUND_TO_NEAREST_INT);
    if (batch & (4 * sizeof(uint16_t))) {
      _mm_storel_epi64((__m128i*) o, vh);
      o += 4;
      vh = _mm_unpackhi_epi64(vh, vh);
    }
    if (batch & (2 * sizeof(uint16_t))) {
      _mm_storeu_si32(o, vh);
      o += 2;
      vh = _mm_srli_epi64(vh, 32);
    }
    if (batch & (1 * sizeof(uint16_t))) {
      *o = (uint16_t) _mm_extract_epi16(vh, 0);
    }
  }
}

void xnn_f16_vsqrt_ukernel__f16c_rsqrt_u32(
    size_t batch,
    const void* input,
    void* output,
    const union xnn_f16_sqrt_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(uint16_t) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const uint16_t* i = (const uint16_t*) input;
  uint16_t* o = (uint16_t*) output;
  const __m256 vinf = _mm256_set1_ps(INFINITY);
  for (; batch >= 32 * sizeof(uint16_t); batch -= 32 * sizeof(uint16_t)) {
    __m256 vacc0 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i));
    __m256 vacc1 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i + 8)));
    __m256 vacc2 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i + 16)));
    __m256 vacc3 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i + 24)));
    i += 32;

    const __m256 vt0_0 = _mm256_rsqrt_ps(vacc0);
    const __m256 vt0_1 = _mm256_rsqrt_ps(vacc1);
    const __m256 vt0_2 = _mm256_rsqrt_ps(vacc2);
    const __m256 vt0_3 = _mm256_rsqrt_ps(vacc3);
    const __m256 vt1_0 = _mm256_cmp_ps(vt0_0, vinf, _CMP_LT_OQ);
    const __m256 vt1_1 = _mm256_cmp_ps(vt0_1, vinf, _CMP_LT_OQ);
    const __m256 vt1_2 = _mm256_cmp_ps(vt0_2, vinf, _CMP_LT_OQ);
    const __m256 vt1_3 = _mm256_cmp_ps(vt0_3, vinf, _CMP_LT_OQ);
    const __m256 vt2_0 = _mm256_and_ps(vt0_0, vt1_0);
    const __m256 vt2_1 = _mm256_and_ps(vt0_1, vt1_1);
    const __m256 vt2_2 = _mm256_and_ps(vt0_2, vt1_2);
    const __m256 vt2_3 = _mm256_and_ps(vt0_3, vt1_3);
    vacc0 = _mm256_mul_ps(vacc0, vt2_0);
    vacc1 = _mm256_mul_ps(vacc1, vt2_1);
    vacc2 = _mm256_mul_ps(vacc2, vt2_2);
    vacc3 = _mm256_mul_ps(vacc3, vt2_3);

    _mm_storeu_si128((__m128i*) o, _mm256_cvtps_ph(vacc0, _MM_FROUND_TO_NEAREST_INT));
    _mm_storeu_si128((__m128i*) (o + 8), _mm256_cvtps_ph(vacc1, _MM_FROUND_TO_NEAREST_INT));
    _mm_storeu_si128((__m128i*) (o + 16), _mm256_cvtps_ph(vacc2, _MM_FROUND_TO_NEAREST_INT));
    _mm_storeu_si128((__m128i*) (o + 24), _mm256_cvtps_ph(vacc3, _MM_FROUND_TO_NEAREST_INT));
    o += 32;
  }
  for (; batch >= 8 * sizeof(uint16_t); batch -= 8 * sizeof(uint16_t)) {
    __m256 vacc = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i));
    i += 8;
    const __m256 vt0 = _mm256_rsqrt_ps(vacc);
    const __m256 vt1 = _mm256_cmp_ps(vt0, vinf, _CMP_LT_OQ);
    const __m256 vt2 = _mm256_and_ps(vt0, vt1);
    vacc = _mm256_mul_ps(vacc, vt2);
    _mm_storeu_si128((__m128i*) o, _mm256_cvtps_ph(vacc, _MM_FROUND_TO_NEAREST_INT));
    o += 8;
  }
  if XNN_UNLIKELY(batch != 0) {
    __m256 vacc = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i));
    const __m256 vt0 = _mm256_rsqrt_ps(vacc);
    const __m256 vt1 = _mm256_cmp_ps(vt0, vinf, _CMP_LT_OQ);
    const __m256 vt2 = _mm256_and_ps(vt0, vt1);
    vacc = _mm256_mul_ps(vacc, vt2);
    __m128i vh = _mm256_cvtps_ph(vacc, _MM_FROUND_TO_NEAREST_INT);
    if (batch & (4 * sizeof(uint16_t))) {
      _mm_storel_epi64((__m128i*) o, vh);
      o += 4;
      vh = _mm_unpackhi_epi64(vh, vh);
    }
    if (batch & (2 * sizeof(uint16_t))) {
      _mm_storeu_si32(o, vh);
      o += 2;
      vh = _mm_srli_epi64(vh, 32);
    }
    if (batch & (1 * sizeof(uint16_t))) {
      *o = (uint16_t) _mm_extract_epi16(vh, 0);
    }
  }
}

void xnn_f16_vtanh_ukernel__f16c_expm1minus_rr1_p3h2ts_rcp_u72(
    size_t batch,
    const void* input,
    void* output,
    const union xnn_f16_tanh_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(uint16_t) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const __m128i vsign_mask = _mm_load_si128((const __m128i*) params->avx_expm1minus_rr1_p3h2.sign_mask);
  const __m256 vsat_cutoff = _mm256_load_ps(params->avx_expm1minus_rr1_p3h2.sat_cutoff);
  const __m256 vlog2e = _mm256_load_ps(params->avx_expm1minus_rr1_p3h2.log2e);
  const __m256 vmagic_bias = _mm256_load_ps(params->avx_expm1minus_rr1_p3h2.magic_bias);
  const __m256 vminus_ln2 = _mm256_load_ps(params->avx_expm1minus_rr1_p3h2.minus_ln2);
  const __m256 vc3 = _mm256_load_ps(params->avx_expm1minus_rr1_p3h2.c3);
  const __m256 vc2 = _mm256_load_ps(params->avx_expm1minus_rr1_p3h2.c2);
  const __m256 vtwo = _mm256_load_ps(params->avx_expm1minus_rr1_p3h2.two);
  const __m256 vminus_one = _mm256_load_ps(params->avx_expm1minus_rr1_p3h2.minus_one);

  const uint16_t* i = (const uint16_t*) input;
  uint16_t* o = (uint16_t*) output;
  for (; batch >= 72 * sizeof(uint16_t); batch -= 72 * sizeof(uint16_t)) {
    const __m128i vx0 = _mm_loadu_si128((const __m128i*) i);
    const __m128i vx1 = _mm_loadu_si128((const __m128i*) (i + 8));
    const __m128i vx2 = _mm_loadu_si128((const __m128i*) (i + 16));
    const __m128i vx3 = _mm_loadu_si128((const __m128i*) (i + 24));
    const __m128i vx4 = _mm_loadu_si128((const __m128i*) (i + 32));
    const __m128i vx5 = _mm_loadu_si128((const __m128i*) (i + 40));
    const __m128i vx6 = _mm_loadu_si128((const __m128i*) (i + 48));
    const __m128i vx7 = _mm_loadu_si128((const __m128i*) (i + 56));
    const __m128i vx8 = _mm_loadu_si128((const __m128i*) (i + 64));
    i += 72;

    const __m128i vabsx0 = _mm_or_si128(vx0, vsign_mask);
    const __m128i vabsx1 = _mm_or_si128(vx1, vsign_mask);
    const __m128i vabsx2 = _mm_or_si128(vx2, vsign_mask);
    const __m128i vabsx3 = _mm_or_si128(vx3, vsign_mask);
    const __m128i vabsx4 = _mm_or_si128(vx4, vsign_mask);
    const __m128i vabsx5 = _mm_or_si128(vx5, vsign_mask);
    const __m128i vabsx6 = _mm_or_si128(vx6, vsign_mask);
    const __m128i vabsx7 = _mm_or_si128(vx7, vsign_mask);
    const __m128i vabsx8 = _mm_or_si128(vx8, vsign_mask);

    __m256 vz0 = _mm256_cvtph_ps(vabsx0);
    const __m128i vinvsignx0 = _mm_xor_si128(vx0, vabsx0);
    __m256 vz1 = _mm256_cvtph_ps(vabsx1);
    const __m128i vinvsignx1 = _mm_xor_si128(vx1, vabsx1);
    __m256 vz2 = _mm256_cvtph_ps(vabsx2);
    const __m128i vinvsignx2 = _mm_xor_si128(vx2, vabsx2);
    __m256 vz3 = _mm256_cvtph_ps(vabsx3);
    const __m128i vinvsignx3 = _mm_xor_si128(vx3, vabsx3);
    __m256 vz4 = _mm256_cvtph_ps(vabsx4);
    const __m128i vinvsignx4 = _mm_xor_si128(vx4, vabsx4);
    __m256 vz5 = _mm256_cvtph_ps(vabsx5);
    const __m128i vinvsignx5 = _mm_xor_si128(vx5, vabsx5);
    __m256 vz6 = _mm256_cvtph_ps(vabsx6);
    const __m128i vinvsignx6 = _mm_xor_si128(vx6, vabsx6);
    __m256 vz7 = _mm256_cvtph_ps(vabsx7);
    const __m128i vinvsignx7 = _mm_xor_si128(vx7, vabsx7);
    __m256 vz8 = _mm256_cvtph_ps(vabsx8);
    const __m128i vinvsignx8 = _mm_xor_si128(vx8, vabsx8);

    const __m256 vm0 = _mm256_cmp_ps(vz0, vsat_cutoff, _CMP_LE_OS);
    __m256 vn0 = _mm256_add_ps(_mm256_mul_ps(vz0, vlog2e), vmagic_bias);
    const __m256 vm1 = _mm256_cmp_ps(vz1, vsat_cutoff, _CMP_LE_OS);
    __m256 vn1 = _mm256_add_ps(_mm256_mul_ps(vz1, vlog2e), vmagic_bias);
    const __m256 vm2 = _mm256_cmp_ps(vz2, vsat_cutoff, _CMP_LE_OS);
    __m256 vn2 = _mm256_add_ps(_mm256_mul_ps(vz2, vlog2e), vmagic_bias);
    const __m256 vm3 = _mm256_cmp_ps(vz3, vsat_cutoff, _CMP_LE_OS);
    __m256 vn3 = _mm256_add_ps(_mm256_mul_ps(vz3, vlog2e), vmagic_bias);
    const __m256 vm4 = _mm256_cmp_ps(vz4, vsat_cutoff, _CMP_LE_OS);
    __m256 vn4 = _mm256_add_ps(_mm256_mul_ps(vz4, vlog2e), vmagic_bias);
    const __m256 vm5 = _mm256_cmp_ps(vz5, vsat_cutoff, _CMP_LE_OS);
    __m256 vn5 = _mm256_add_ps(_mm256_mul_ps(vz5, vlog2e), vmagic_bias);
    const __m256 vm6 = _mm256_cmp_ps(vz6, vsat_cutoff, _CMP_LE_OS);
    __m256 vn6 = _mm256_add_ps(_mm256_mul_ps(vz6, vlog2e), vmagic_bias);
    const __m256 vm7 = _mm256_cmp_ps(vz7, vsat_cutoff, _CMP_LE_OS);
    __m256 vn7 = _mm256_add_ps(_mm256_mul_ps(vz7, vlog2e), vmagic_bias);
    const __m256 vm8 = _mm256_cmp_ps(vz8, vsat_cutoff, _CMP_LE_OS);
    __m256 vn8 = _mm256_add_ps(_mm256_mul_ps(vz8, vlog2e), vmagic_bias);

    const __m128 vn0_hi = _mm256_extractf128_ps(vn0, 1);
    __m256 vs0 = _mm256_castps128_ps256(_mm_castsi128_ps(_mm_slli_epi32(_mm_castps_si128(_mm256_castps256_ps128(vn0)), 23)));
    vn0 = _mm256_sub_ps(vn0, vmagic_bias);
    const __m128 vn1_hi = _mm256_extractf128_ps(vn1, 1);
    __m256 vs1 = _mm256_castps128_ps256(_mm_castsi128_ps(_mm_slli_epi32(_mm_castps_si128(_mm256_castps256_ps128(vn1)), 23)));
    vn1 = _mm256_sub_ps(vn1, vmagic_bias);
    const __m128 vn2_hi = _mm256_extractf128_ps(vn2, 1);
    __m256 vs2 = _mm256_castps128_ps256(_mm_castsi128_ps(_mm_slli_epi32(_mm_castps_si128(_mm256_castps256_ps128(vn2)), 23)));
    vn2 = _mm256_sub_ps(vn2, vmagic_bias);
    const __m128 vn3_hi = _mm256_extractf128_ps(vn3, 1);
    __m256 vs3 = _mm256_castps128_ps256(_mm_castsi128_ps(_mm_slli_epi32(_mm_castps_si128(_mm256_castps256_ps128(vn3)), 23)));
    vn3 = _mm256_sub_ps(vn3, vmagic_bias);
    const __m128 vn4_hi = _mm256_extractf128_ps(vn4, 1);
    __m256 vs4 = _mm256_castps128_ps256(_mm_castsi128_ps(_mm_slli_epi32(_mm_castps_si128(_mm256_castps256_ps128(vn4)), 23)));
    vn4 = _mm256_sub_ps(vn4, vmagic_bias);
    const __m128 vn5_hi = _mm256_extractf128_ps(vn5, 1);
    __m256 vs5 = _mm256_castps128_ps256(_mm_castsi128_ps(_mm_slli_epi32(_mm_castps_si128(_mm256_castps256_ps128(vn5)), 23)));
    vn5 = _mm256_sub_ps(vn5, vmagic_bias);
    const __m128 vn6_hi = _mm256_extractf128_ps(vn6, 1);
    __m256 vs6 = _mm256_castps128_ps256(_mm_castsi128_ps(_mm_slli_epi32(_mm_castps_si128(_mm256_castps256_ps128(vn6)), 23)));
    vn6 = _mm256_sub_ps(vn6, vmagic_bias);
    const __m128 vn7_hi = _mm256_extractf128_ps(vn7, 1);
    __m256 vs7 = _mm256_castps128_ps256(_mm_castsi128_ps(_mm_slli_epi32(_mm_castps_si128(_mm256_castps256_ps128(vn7)), 23)));
    vn7 = _mm256_sub_ps(vn7, vmagic_bias);
    const __m128 vn8_hi = _mm256_extractf128_ps(vn8, 1);
    __m256 vs8 = _mm256_castps128_ps256(_mm_castsi128_ps(_mm_slli_epi32(_mm_castps_si128(_mm256_castps256_ps128(vn8)), 23)));
    vn8 = _mm256_sub_ps(vn8, vmagic_bias);

    const __m128 vs0_hi = _mm_castsi128_ps(_mm_slli_epi32(_mm_castps_si128(vn0_hi), 23));
    const __m128 vs1_hi = _mm_castsi128_ps(_mm_slli_epi32(_mm_castps_si128(vn1_hi), 23));
    const __m128 vs2_hi = _mm_castsi128_ps(_mm_slli_epi32(_mm_castps_si128(vn2_hi), 23));
    const __m128 vs3_hi = _mm_castsi128_ps(_mm_slli_epi32(_mm_castps_si128(vn3_hi), 23));
    const __m128 vs4_hi = _mm_castsi128_ps(_mm_slli_epi32(_mm_castps_si128(vn4_hi), 23));
    const __m128 vs5_hi = _mm_castsi128_ps(_mm_slli_epi32(_mm_castps_si128(vn5_hi), 23));
    const __m128 vs6_hi = _mm_castsi128_ps(_mm_slli_epi32(_mm_castps_si128(vn6_hi), 23));
    const __m128 vs7_hi = _mm_castsi128_ps(_mm_slli_epi32(_mm_castps_si128(vn7_hi), 23));
    const __m128 vs8_hi = _mm_castsi128_ps(_mm_slli_epi32(_mm_castps_si128(vn8_hi), 23));

    vs0 = _mm256_insertf128_ps(vs0, vs0_hi, 1);
    vs1 = _mm256_insertf128_ps(vs1, vs1_hi, 1);
    vs2 = _mm256_insertf128_ps(vs2, vs2_hi, 1);
    vs3 = _mm256_insertf128_ps(vs3, vs3_hi, 1);
    vs4 = _mm256_insertf128_ps(vs4, vs4_hi, 1);
    vs5 = _mm256_insertf128_ps(vs5, vs5_hi, 1);
    vs6 = _mm256_insertf128_ps(vs6, vs6_hi, 1);
    vs7 = _mm256_insertf128_ps(vs7, vs7_hi, 1);
    vs8 = _mm256_insertf128_ps(vs8, vs8_hi, 1);

    const __m256 vt0 = _mm256_add_ps(_mm256_mul_ps(vn0, vminus_ln2), vz0);
    const __m256 vt1 = _mm256_add_ps(_mm256_mul_ps(vn1, vminus_ln2), vz1);
    const __m256 vt2 = _mm256_add_ps(_mm256_mul_ps(vn2, vminus_ln2), vz2);
    const __m256 vt3 = _mm256_add_ps(_mm256_mul_ps(vn3, vminus_ln2), vz3);
    const __m256 vt4 = _mm256_add_ps(_mm256_mul_ps(vn4, vminus_ln2), vz4);
    const __m256 vt5 = _mm256_add_ps(_mm256_mul_ps(vn5, vminus_ln2), vz5);
    const __m256 vt6 = _mm256_add_ps(_mm256_mul_ps(vn6, vminus_ln2), vz6);
    const __m256 vt7 = _mm256_add_ps(_mm256_mul_ps(vn7, vminus_ln2), vz7);
    const __m256 vt8 = _mm256_add_ps(_mm256_mul_ps(vn8, vminus_ln2), vz8);

    __m256 vp0 = _mm256_add_ps(_mm256_mul_ps(vc3, vt0), vc2);
    __m256 vp1 = _mm256_add_ps(_mm256_mul_ps(vc3, vt1), vc2);
    __m256 vp2 = _mm256_add_ps(_mm256_mul_ps(vc3, vt2), vc2);
    __m256 vp3 = _mm256_add_ps(_mm256_mul_ps(vc3, vt3), vc2);
    __m256 vp4 = _mm256_add_ps(_mm256_mul_ps(vc3, vt4), vc2);
    __m256 vp5 = _mm256_add_ps(_mm256_mul_ps(vc3, vt5), vc2);
    __m256 vp6 = _mm256_add_ps(_mm256_mul_ps(vc3, vt6), vc2);
    __m256 vp7 = _mm256_add_ps(_mm256_mul_ps(vc3, vt7), vc2);
    __m256 vp8 = _mm256_add_ps(_mm256_mul_ps(vc3, vt8), vc2);
    vp0 = _mm256_add_ps(_mm256_mul_ps(vp0, vt0), vtwo);
    vp1 = _mm256_add_ps(_mm256_mul_ps(vp1, vt1), vtwo);
    vp2 = _mm256_add_ps(_mm256_mul_ps(vp2, vt2), vtwo);
    vp3 = _mm256_add_ps(_mm256_mul_ps(vp3, vt3), vtwo);
    vp4 = _mm256_add_ps(_mm256_mul_ps(vp4, vt4), vtwo);
    vp5 = _mm256_add_ps(_mm256_mul_ps(vp5, vt5), vtwo);
    vp6 = _mm256_add_ps(_mm256_mul_ps(vp6, vt6), vtwo);
    vp7 = _mm256_add_ps(_mm256_mul_ps(vp7, vt7), vtwo);
    vp8 = _mm256_add_ps(_mm256_mul_ps(vp8, vt8), vtwo);

    const __m256 vts0 = _mm256_mul_ps(vt0, vs0);
    const __m256 vsmo0 = _mm256_add_ps(vs0, vminus_one);
    const __m256 vts1 = _mm256_mul_ps(vt1, vs1);
    const __m256 vsmo1 = _mm256_add_ps(vs1, vminus_one);
    const __m256 vts2 = _mm256_mul_ps(vt2, vs2);
    const __m256 vsmo2 = _mm256_add_ps(vs2, vminus_one);
    const __m256 vts3 = _mm256_mul_ps(vt3, vs3);
    const __m256 vsmo3 = _mm256_add_ps(vs3, vminus_one);
    const __m256 vts4 = _mm256_mul_ps(vt4, vs4);
    const __m256 vsmo4 = _mm256_add_ps(vs4, vminus_one);
    const __m256 vts5 = _mm256_mul_ps(vt5, vs5);
    const __m256 vsmo5 = _mm256_add_ps(vs5, vminus_one);
    const __m256 vts6 = _mm256_mul_ps(vt6, vs6);
    const __m256 vsmo6 = _mm256_add_ps(vs6, vminus_one);
    const __m256 vts7 = _mm256_mul_ps(vt7, vs7);
    const __m256 vsmo7 = _mm256_add_ps(vs7, vminus_one);
    const __m256 vts8 = _mm256_mul_ps(vt8, vs8);
    const __m256 vsmo8 = _mm256_add_ps(vs8, vminus_one);
    const __m256 vemo0 = _mm256_add_ps(_mm256_mul_ps(vp0, vts0), vsmo0);
    const __m256 vemo1 = _mm256_add_ps(_mm256_mul_ps(vp1, vts1), vsmo1);
    const __m256 vemo2 = _mm256_add_ps(_mm256_mul_ps(vp2, vts2), vsmo2);
    const __m256 vemo3 = _mm256_add_ps(_mm256_mul_ps(vp3, vts3), vsmo3);
    const __m256 vemo4 = _mm256_add_ps(_mm256_mul_ps(vp4, vts4), vsmo4);
    const __m256 vemo5 = _mm256_add_ps(_mm256_mul_ps(vp5, vts5), vsmo5);
    const __m256 vemo6 = _mm256_add_ps(_mm256_mul_ps(vp6, vts6), vsmo6);
    const __m256 vemo7 = _mm256_add_ps(_mm256_mul_ps(vp7, vts7), vsmo7);
    const __m256 vemo8 = _mm256_add_ps(_mm256_mul_ps(vp8, vts8), vsmo8);

    const __m256 vepo0 = _mm256_add_ps(vemo0, vtwo);
    const __m256 vepo1 = _mm256_add_ps(vemo1, vtwo);
    const __m256 vepo2 = _mm256_add_ps(vemo2, vtwo);
    const __m256 vepo3 = _mm256_add_ps(vemo3, vtwo);
    const __m256 vepo4 = _mm256_add_ps(vemo4, vtwo);
    const __m256 vepo5 = _mm256_add_ps(vemo5, vtwo);
    const __m256 vepo6 = _mm256_add_ps(vemo6, vtwo);
    const __m256 vepo7 = _mm256_add_ps(vemo7, vtwo);
    const __m256 vepo8 = _mm256_add_ps(vemo8, vtwo);

    __m256 vrepo0 = _mm256_rcp_ps(vepo0);
    __m256 vrepo1 = _mm256_rcp_ps(vepo1);
    __m256 vrepo2 = _mm256_rcp_ps(vepo2);
    __m256 vrepo3 = _mm256_rcp_ps(vepo3);
    __m256 vrepo4 = _mm256_rcp_ps(vepo4);
    __m256 vrepo5 = _mm256_rcp_ps(vepo5);
    __m256 vrepo6 = _mm256_rcp_ps(vepo6);
    __m256 vrepo7 = _mm256_rcp_ps(vepo7);
    __m256 vrepo8 = _mm256_rcp_ps(vepo8);

    __m256 vy0 = _mm256_mul_ps(vemo0, vrepo0);
    __m256 vy1 = _mm256_mul_ps(vemo1, vrepo1);
    __m256 vy2 = _mm256_mul_ps(vemo2, vrepo2);
    __m256 vy3 = _mm256_mul_ps(vemo3, vrepo3);
    __m256 vy4 = _mm256_mul_ps(vemo4, vrepo4);
    __m256 vy5 = _mm256_mul_ps(vemo5, vrepo5);
    __m256 vy6 = _mm256_mul_ps(vemo6, vrepo6);
    __m256 vy7 = _mm256_mul_ps(vemo7, vrepo7);
    __m256 vy8 = _mm256_mul_ps(vemo8, vrepo8);

    vy0 = _mm256_blendv_ps(vy0, vminus_one, vm0);
    vy1 = _mm256_blendv_ps(vy1, vminus_one, vm1);
    vy2 = _mm256_blendv_ps(vy2, vminus_one, vm2);
    vy3 = _mm256_blendv_ps(vy3, vminus_one, vm3);
    vy4 = _mm256_blendv_ps(vy4, vminus_one, vm4);
    vy5 = _mm256_blendv_ps(vy5, vminus_one, vm5);
    vy6 = _mm256_blendv_ps(vy6, vminus_one, vm6);
    vy7 = _mm256_blendv_ps(vy7, vminus_one, vm7);
    vy8 = _mm256_blendv_ps(vy8, vminus_one, vm8);

    __m128i vh0 = _mm256_cvtps_ph(vy0, _MM_FROUND_TO_NEAREST_INT);
    __m128i vh1 = _mm256_cvtps_ph(vy1, _MM_FROUND_TO_NEAREST_INT);
    __m128i vh2 = _mm256_cvtps_ph(vy2, _MM_FROUND_TO_NEAREST_INT);
    __m128i vh3 = _mm256_cvtps_ph(vy3, _MM_FROUND_TO_NEAREST_INT);
    __m128i vh4 = _mm256_cvtps_ph(vy4, _MM_FROUND_TO_NEAREST_INT);
    __m128i vh5 = _mm256_cvtps_ph(vy5, _MM_FROUND_TO_NEAREST_INT);
    __m128i vh6 = _mm256_cvtps_ph(vy6, _MM_FROUND_TO_NEAREST_INT);
    __m128i vh7 = _mm256_cvtps_ph(vy7, _MM_FROUND_TO_NEAREST_INT);
    __m128i vh8 = _mm256_cvtps_ph(vy8, _MM_FROUND_TO_NEAREST_INT);
    vh0 = _mm_xor_si128(vh0, vinvsignx0);
    vh1 = _mm_xor_si128(vh1, vinvsignx1);
    vh2 = _mm_xor_si128(vh2, vinvsignx2);
    vh3 = _mm_xor_si128(vh3, vinvsignx3);
    vh4 = _mm_xor_si128(vh4, vinvsignx4);
    vh5 = _mm_xor_si128(vh5, vinvsignx5);
    vh6 = _mm_xor_si128(vh6, vinvsignx6);
    vh7 = _mm_xor_si128(vh7, vinvsignx7);
    vh8 = _mm_xor_si128(vh8, vinvsignx8);

    _mm_storeu_si128((__m128i*) o, vh0);
    _mm_storeu_si128((__m128i*) (o + 8), vh1);
    _mm_storeu_si128((__m128i*) (o + 16), vh2);
    _mm_storeu_si128((__m128i*) (o + 24), vh3);
    _mm_storeu_si128((__m128i*) (o + 32), vh4);
    _mm_storeu_si128((__m128i*) (o + 40), vh5);
    _mm_storeu_si128((__m128i*) (o + 48), vh6);
    _mm_storeu_si128((__m128i*) (o + 56), vh7);
    _mm_storeu_si128((__m128i*) (o + 64), vh8);
    o += 72;
  }
  for (; batch >= 8 * sizeof(uint16_t); batch -= 8 * sizeof(uint16_t)) {
    const __m128i vx = _mm_loadu_si128((const __m128i*) i);
    i += 8;

    const __m128i vabsx = _mm_or_si128(vx, vsign_mask);
    __m256 vz = _mm256_cvtph_ps(vabsx);

    const __m128i vinvsignx = _mm_xor_si128(vx, vabsx);
    const __m256 vm = _mm256_cmp_ps(vz, vsat_cutoff, _CMP_LE_OS);

    __m256 vn = _mm256_add_ps(_mm256_mul_ps(vz, vlog2e), vmagic_bias);

    const __m128 vn_hi = _mm256_extractf128_ps(vn, 1);
    __m256 vs = _mm256_castps128_ps256(_mm_castsi128_ps(_mm_slli_epi32(_mm_castps_si128(_mm256_castps256_ps128(vn)), 23)));
    const __m128 vs_hi = _mm_castsi128_ps(_mm_slli_epi32(_mm_castps_si128(vn_hi), 23));
    vs = _mm256_insertf128_ps(vs, vs_hi, 1);

    vn = _mm256_sub_ps(vn, vmagic_bias);

    const __m256 vt = _mm256_add_ps(_mm256_mul_ps(vn, vminus_ln2), vz);

    __m256 vp = _mm256_add_ps(_mm256_mul_ps(vc3, vt), vc2);
    vp = _mm256_add_ps(_mm256_mul_ps(vp, vt), vtwo);

    const __m256 vts = _mm256_mul_ps(vt, vs);
    const __m256 vsmo = _mm256_add_ps(vs, vminus_one);
    const __m256 vemo = _mm256_add_ps(_mm256_mul_ps(vp, vts), vsmo);

    const __m256 vepo = _mm256_add_ps(vemo, vtwo);

    __m256 vrepo = _mm256_rcp_ps(vepo);

    __m256 vy = _mm256_mul_ps(vemo, vrepo);

    vy = _mm256_blendv_ps(vy, vminus_one, vm);

    __m128i vh = _mm256_cvtps_ph(vy, _MM_FROUND_TO_NEAREST_INT);
    vh = _mm_xor_si128(vh, vinvsignx);

    _mm_storeu_si128((__m128i*) o, vh);
    o += 8;
  }
  if (batch != 0) {
    const __m128i vx = _mm_loadu_si128((const __m128i*) i);

    const __m128i vabsx = _mm_or_si128(vx, vsign_mask);
    __m256 vz = _mm256_cvtph_ps(vabsx);

    const __m128i vinvsignx = _mm_xor_si128(vx, vabsx);
    const __m256 vm = _mm256_cmp_ps(vz, vsat_cutoff, _CMP_LE_OS);

    __m256 vn = _mm256_add_ps(_mm256_mul_ps(vz, vlog2e), vmagic_bias);

    const __m128 vn_hi = _mm256_extractf128_ps(vn, 1);
    __m256 vs = _mm256_castps128_ps256(_mm_castsi128_ps(_mm_slli_epi32(_mm_castps_si128(_mm256_castps256_ps128(vn)), 23)));
    const __m128 vs_hi = _mm_castsi128_ps(_mm_slli_epi32(_mm_castps_si128(vn_hi), 23));
    vs = _mm256_insertf128_ps(vs, vs_hi, 1);

    vn = _mm256_sub_ps(vn, vmagic_bias);

    const __m256 vt = _mm256_add_ps(_mm256_mul_ps(vn, vminus_ln2), vz);

    __m256 vp = _mm256_add_ps(_mm256_mul_ps(vc3, vt), vc2);
    vp = _mm256_add_ps(_mm256_mul_ps(vp, vt), vtwo);

    const __m256 vts = _mm256_mul_ps(vt, vs);
    const __m256 vsmo = _mm256_add_ps(vs, vminus_one);
    const __m256 vemo = _mm256_add_ps(_mm256_mul_ps(vp, vts), vsmo);

    const __m256 vepo = _mm256_add_ps(vemo, vtwo);

    __m256 vrepo = _mm256_rcp_ps(vepo);

    __m256 vy = _mm256_mul_ps(vemo, vrepo);

    vy = _mm256_blendv_ps(vy, vminus_one, vm);

    __m128i vh = _mm256_cvtps_ph(vy, _MM_FROUND_TO_NEAREST_INT);
    vh = _mm_xor_si128(vh, vinvsignx);

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

void xnn_f16_vsqr_ukernel__f16c_u16(
    size_t batch,
    const void* input,
    void* output,
    const union xnn_f16_default_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(uint16_t) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const uint16_t* i = (const uint16_t*) input;
  uint16_t* o = (uint16_t*) output;
  for (; batch >= 16 * sizeof(uint16_t); batch -= 16 * sizeof(uint16_t)) {
    __m256 vacc0 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i));
    __m256 vacc1 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i + 8)));
    i += 16;

    vacc0 = _mm256_mul_ps(vacc0, vacc0);
    vacc1 = _mm256_mul_ps(vacc1, vacc1);

    _mm_storeu_si128((__m128i*) o, _mm256_cvtps_ph(vacc0, _MM_FROUND_TO_NEAREST_INT));
    _mm_storeu_si128((__m128i*) (o + 8), _mm256_cvtps_ph(vacc1, _MM_FROUND_TO_NEAREST_INT));
    o += 16;
  }
  for (; batch >= 8 * sizeof(uint16_t); batch -= 8 * sizeof(uint16_t)) {
    __m256 vacc = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i));
    i += 8;
    vacc = _mm256_mul_ps(vacc, vacc);
    _mm_storeu_si128((__m128i*) o, _mm256_cvtps_ph(vacc, _MM_FROUND_TO_NEAREST_INT));
    o += 8;
  }
  if XNN_UNLIKELY(batch != 0) {
    __m256 vacc = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i));
    vacc = _mm256_mul_ps(vacc, vacc);
    __m128i vh = _mm256_cvtps_ph(vacc, _MM_FROUND_TO_NEAREST_INT);
    if (batch & (4 * sizeof(uint16_t))) {
      _mm_storel_epi64((__m128i*) o, vh);
      o += 4;
      vh = _mm_unpackhi_epi64(vh, vh);
    }
    if (batch & (2 * sizeof(uint16_t))) {
      _mm_storeu_si32(o, vh);
      o += 2;
      vh = _mm_srli_epi64(vh, 32);
    }
    if (batch & (1 * sizeof(uint16_t))) {
      *o = (uint16_t) _mm_extract_epi16(vh, 0);
    }
  }
}

void xnn_f32_f16_vcvt_ukernel__f16c_u16(
    size_t batch,
    const float* input,
    void* output,
    const void* params)
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  static const uint32_t mask_table[14] = {-1, -1, -1, -1, -1, -1, -1, 0, 0, 0, 0, 0, 0, 0};

  uint16_t* o = (uint16_t*) output;
  for (; batch >= 16 * sizeof(float); batch -= 16 * sizeof(float)) {
    const __m256 vf0 = _mm256_loadu_ps(input);
    const __m256 vf1 = _mm256_loadu_ps(input + 8);
    input += 16;

    _mm_storeu_si128((__m128i*) o, _mm256_cvtps_ph(vf0, _MM_FROUND_TO_NEAREST_INT));
    _mm_storeu_si128((__m128i*) (o + 8), _mm256_cvtps_ph(vf1, _MM_FROUND_TO_NEAREST_INT));
    o += 16;
  }
  for (; batch >= 8 * sizeof(float); batch -= 8 * sizeof(float)) {
    const __m256 vf = _mm256_loadu_ps(input);
    input += 8;

    _mm_storeu_si128((__m128i*) o, _mm256_cvtps_ph(vf, _MM_FROUND_TO_NEAREST_INT));
    o += 8;
  }
  if XNN_UNLIKELY(batch != 0) {
    assert(batch >= 1 * sizeof(float));
    assert(batch <= 7 * sizeof(float));
    const __m256i vmask = _mm256_loadu_si256((const __m256i*) ((uintptr_t) &mask_table[7] - batch));

    const __m256 vf = _mm256_maskload_ps(input, vmask);

    __m128 vf_lo = _mm256_castps256_ps128(vf);
    if (batch & (4 * sizeof(float))) {
      _mm_storel_epi64((__m128i*) o, _mm_cvtps_ph(vf_lo, _MM_FROUND_TO_NEAREST_INT));
      vf_lo = _mm256_extractf128_ps(vf, 1);
      o += 4;
    }
    __m128i vh = _mm_cvtps_ph(vf_lo, _MM_FROUND_TO_NEAREST_INT);
    if (batch & (2 * sizeof(float))) {
      _mm_storeu_si32(o, vh);
      vh = _mm_srli_epi64(vh, 32);
      o += 2;
    }
    if (batch & (1 * sizeof(float))) {
      *((uint16_t*) o) = _mm_extract_epi16(vh, 0);
    }
  }
}
