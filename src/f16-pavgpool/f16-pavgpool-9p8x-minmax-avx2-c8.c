// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <immintrin.h>

#include <stddef.h>
#include <stdint.h>

#include "xnnpack/common.h"
#include "xnnpack/intrinsics-polyfill.h"
#include "xnnpack/microparams.h"
#include "xnnpack/pavgpool.h"


void xnn_f16_pavgpool_minmax_ukernel_9p8x__avx2_c8(
    size_t output_pixels,
    size_t kernel_elements,
    size_t channels,
    const xnn_float16** input,
    size_t input_offset,
    const xnn_float16* zero,
    const xnn_float16* multiplier,
    xnn_float16* buffer,
    xnn_float16* output,
    size_t input_increment,
    size_t output_increment,
    const struct xnn_f16_scaleminmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(output_pixels != 0);
  assert(kernel_elements > 9);
  assert(channels != 0);

  const __m256 voutput_min = _mm256_cvtph_ps(_mm_set1_epi16(*(const uint16_t*) &params->scalar.min));
  const __m256 voutput_max = _mm256_cvtph_ps(_mm_set1_epi16(*(const uint16_t*) &params->scalar.max));
  XNN_FORCE_REALIZATION(voutput_min);
  XNN_FORCE_REALIZATION(voutput_max);

  uint16_t* o = (uint16_t*) output;
  do {
    {
      const uint16_t* i0 = (const uint16_t*) *input++;
      assert(i0 != NULL);
      if XNN_UNPREDICTABLE(i0 != (const uint16_t*) zero) {
        i0 = (const uint16_t*) ((uintptr_t) i0 + input_offset);
      }
      const uint16_t* i1 = (const uint16_t*) *input++;
      assert(i1 != NULL);
      if XNN_UNPREDICTABLE(i1 != (const uint16_t*) zero) {
        i1 = (const uint16_t*) ((uintptr_t) i1 + input_offset);
      }
      const uint16_t* i2 = (const uint16_t*) *input++;
      assert(i2 != NULL);
      if XNN_UNPREDICTABLE(i2 != (const uint16_t*) zero) {
        i2 = (const uint16_t*) ((uintptr_t) i2 + input_offset);
      }
      const uint16_t* i3 = (const uint16_t*) *input++;
      assert(i3 != NULL);
      if XNN_UNPREDICTABLE(i3 != (const uint16_t*) zero) {
        i3 = (const uint16_t*) ((uintptr_t) i3 + input_offset);
      }
      const uint16_t* i4 = (const uint16_t*) *input++;
      assert(i4 != NULL);
      if XNN_UNPREDICTABLE(i4 != (const uint16_t*) zero) {
        i4 = (const uint16_t*) ((uintptr_t) i4 + input_offset);
      }
      const uint16_t* i5 = (const uint16_t*) *input++;
      assert(i5 != NULL);
      if XNN_UNPREDICTABLE(i5 != (const uint16_t*) zero) {
        i5 = (const uint16_t*) ((uintptr_t) i5 + input_offset);
      }
      const uint16_t* i6 = (const uint16_t*) *input++;
      assert(i6 != NULL);
      if XNN_UNPREDICTABLE(i6 != (const uint16_t*) zero) {
        i6 = (const uint16_t*) ((uintptr_t) i6 + input_offset);
      }
      const uint16_t* i7 = (const uint16_t*) *input++;
      assert(i7 != NULL);
      if XNN_UNPREDICTABLE(i7 != (const uint16_t*) zero) {
        i7 = (const uint16_t*) ((uintptr_t) i7 + input_offset);
      }
      const uint16_t* i8 = (const uint16_t*) *input++;
      assert(i8 != NULL);
      if XNN_UNPREDICTABLE(i8 != (const uint16_t*) zero) {
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
        const __m256 vsum = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_add_ps(vsum2345, vsum01678), _MM_FROUND_TO_NEAREST_INT));

        _mm_storeu_si128((__m128i*) b, _mm256_cvtps_ph(vsum, _MM_FROUND_TO_NEAREST_INT));
        b += 8;
      }
    }

    size_t k = kernel_elements;
    for (k -= 9; k > 8; k -= 8) {
      const uint16_t* i0 = (const uint16_t*) *input++;
      assert(i0 != NULL);
      if XNN_UNPREDICTABLE(i0 != (const uint16_t*) zero) {
        i0 = (const uint16_t*) ((uintptr_t) i0 + input_offset);
      }
      const uint16_t* i1 = (const uint16_t*) *input++;
      assert(i1 != NULL);
      if XNN_UNPREDICTABLE(i1 != (const uint16_t*) zero) {
        i1 = (const uint16_t*) ((uintptr_t) i1 + input_offset);
      }
      const uint16_t* i2 = (const uint16_t*) *input++;
      assert(i2 != NULL);
      if XNN_UNPREDICTABLE(i2 != (const uint16_t*) zero) {
        i2 = (const uint16_t*) ((uintptr_t) i2 + input_offset);
      }
      const uint16_t* i3 = (const uint16_t*) *input++;
      assert(i3 != NULL);
      if XNN_UNPREDICTABLE(i3 != (const uint16_t*) zero) {
        i3 = (const uint16_t*) ((uintptr_t) i3 + input_offset);
      }
      const uint16_t* i4 = (const uint16_t*) *input++;
      assert(i4 != NULL);
      if XNN_UNPREDICTABLE(i4 != (const uint16_t*) zero) {
        i4 = (const uint16_t*) ((uintptr_t) i4 + input_offset);
      }
      const uint16_t* i5 = (const uint16_t*) *input++;
      assert(i5 != NULL);
      if XNN_UNPREDICTABLE(i5 != (const uint16_t*) zero) {
        i5 = (const uint16_t*) ((uintptr_t) i5 + input_offset);
      }
      const uint16_t* i6 = (const uint16_t*) *input++;
      assert(i6 != NULL);
      if XNN_UNPREDICTABLE(i6 != (const uint16_t*) zero) {
        i6 = (const uint16_t*) ((uintptr_t) i6 + input_offset);
      }
      const uint16_t* i7 = (const uint16_t*) *input++;
      assert(i7 != NULL);
      if XNN_UNPREDICTABLE(i7 != (const uint16_t*) zero) {
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
        const __m256 vsum = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_add_ps(vsum2345, vsum0167a), _MM_FROUND_TO_NEAREST_INT));

        _mm_storeu_si128((__m128i*) b, _mm256_cvtps_ph(vsum, _MM_FROUND_TO_NEAREST_INT));
        b += 8;
      }
    }

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
      input = (const xnn_float16**) ((uintptr_t) input + input_increment);
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
      if XNN_UNPREDICTABLE(i0 != (const uint16_t*) zero) {
        i0 = (const uint16_t*) ((uintptr_t) i0 + input_offset);
      }
      if XNN_UNPREDICTABLE(i1 != (const uint16_t*) zero) {
        i1 = (const uint16_t*) ((uintptr_t) i1 + input_offset);
      }
      if XNN_UNPREDICTABLE(i2 != (const uint16_t*) zero) {
        i2 = (const uint16_t*) ((uintptr_t) i2 + input_offset);
      }
      if XNN_UNPREDICTABLE(i3 != (const uint16_t*) zero) {
        i3 = (const uint16_t*) ((uintptr_t) i3 + input_offset);
      }
      if XNN_UNPREDICTABLE(i4 != (const uint16_t*) zero) {
        i4 = (const uint16_t*) ((uintptr_t) i4 + input_offset);
      }
      if XNN_UNPREDICTABLE(i5 != (const uint16_t*) zero) {
        i5 = (const uint16_t*) ((uintptr_t) i5 + input_offset);
      }
      if XNN_UNPREDICTABLE(i6 != (const uint16_t*) zero) {
        i6 = (const uint16_t*) ((uintptr_t) i6 + input_offset);
      }
      if XNN_UNPREDICTABLE(i7 != (const uint16_t*) zero) {
        i7 = (const uint16_t*) ((uintptr_t) i7 + input_offset);
      }

      const __m256 vmultiplier = _mm256_cvtph_ps(_mm_set1_epi16((short) *((const uint16_t*) multiplier)));
      multiplier = (const xnn_float16*) multiplier + 1;

      size_t c = channels;
      const uint16_t* b = (const uint16_t*) buffer;
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

        __m256 vout = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_mul_ps(vsum, vmultiplier), _MM_FROUND_TO_NEAREST_INT));
        vout = _mm256_max_ps(vout, voutput_min);
        vout = _mm256_min_ps(vout, voutput_max);

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

        __m256 vout = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_mul_ps(vsum, vmultiplier), _MM_FROUND_TO_NEAREST_INT));
        vout = _mm256_max_ps(vout, voutput_min);
        vout = _mm256_min_ps(vout, voutput_max);

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
