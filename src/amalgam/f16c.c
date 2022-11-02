// Copyright 2021 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <immintrin.h>

#include <xnnpack/avgpool.h>
#include <xnnpack/common.h>
#include <xnnpack/gavgpool.h>
#include <xnnpack/intrinsics-polyfill.h>
#include <xnnpack/math.h>
#include <xnnpack/maxpool.h>
#include <xnnpack/prelu.h>
#include <xnnpack/rmax.h>
#include <xnnpack/vbinary.h>
#include <xnnpack/vcvt.h>
#include <xnnpack/vunary.h>


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

  const __m256 vscale = _mm256_load_ps(params->avx.scale);
  const __m256 vmin = _mm256_load_ps(params->avx.min);
  const __m256 vmax = _mm256_load_ps(params->avx.max);

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

  const __m256 vscale = _mm256_load_ps(params->avx.scale);
  const __m256 vmin = _mm256_load_ps(params->avx.min);
  const __m256 vmax = _mm256_load_ps(params->avx.max);

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

void xnn_f16_f32_vcvt_ukernel__f16c_x16(
    size_t batch,
    const void* input,
    float* output,
    const union xnn_f16_f32_cvt_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
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

  const __m256 vscale = _mm256_load_ps(params->avx.scale);
  const __m256 vmin = _mm256_load_ps(params->avx.min);
  const __m256 vmax = _mm256_load_ps(params->avx.max);
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

  const __m256 vscale = _mm256_load_ps(params->avx.scale);
  const __m256 vmin = _mm256_load_ps(params->avx.min);
  const __m256 vmax = _mm256_load_ps(params->avx.max);
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

  const __m256 voutput_min = _mm256_load_ps(params->avx.min);
  const __m256 voutput_max = _mm256_load_ps(params->avx.max);
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

void xnn_f16_rmax_ukernel__f16c(
    size_t batch,
    const void* input,
    void* output) XNN_OOB_READS
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

void xnn_f16_vadd_minmax_ukernel__f16c_x16(
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

  const __m256 vy_min = _mm256_load_ps(params->avx.min);
  const __m256 vy_max = _mm256_load_ps(params->avx.max);

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

void xnn_f16_vaddc_minmax_ukernel__f16c_x16(
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

  const __m256 vy_min = _mm256_load_ps(params->avx.min);
  const __m256 vy_max = _mm256_load_ps(params->avx.max);

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

void xnn_f16_vdiv_minmax_ukernel__f16c_x8(
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

  const __m256 vy_min = _mm256_load_ps(params->avx.min);
  const __m256 vy_max = _mm256_load_ps(params->avx.max);

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

void xnn_f16_vdivc_minmax_ukernel__f16c_x8(
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

  const __m256 vy_min = _mm256_load_ps(params->avx.min);
  const __m256 vy_max = _mm256_load_ps(params->avx.max);

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

void xnn_f16_vmax_ukernel__f16c_x16(
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

void xnn_f16_vmaxc_ukernel__f16c_x16(
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

void xnn_f16_vmin_ukernel__f16c_x16(
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

void xnn_f16_vminc_ukernel__f16c_x16(
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

void xnn_f16_vmul_minmax_ukernel__f16c_x16(
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

  const __m256 vy_min = _mm256_load_ps(params->avx.min);
  const __m256 vy_max = _mm256_load_ps(params->avx.max);

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

void xnn_f16_vmulc_minmax_ukernel__f16c_x16(
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

  const __m256 vy_min = _mm256_load_ps(params->avx.min);
  const __m256 vy_max = _mm256_load_ps(params->avx.max);

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

void xnn_f16_vrdivc_minmax_ukernel__f16c_x8(
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

  const __m256 vy_min = _mm256_load_ps(params->avx.min);
  const __m256 vy_max = _mm256_load_ps(params->avx.max);

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

void xnn_f16_vrsubc_minmax_ukernel__f16c_x16(
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

  const __m256 vy_min = _mm256_load_ps(params->avx.min);
  const __m256 vy_max = _mm256_load_ps(params->avx.max);

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

void xnn_f16_vsqrdiff_ukernel__f16c_x16(
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

void xnn_f16_vsqrdiffc_ukernel__f16c_x16(
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

void xnn_f16_vsub_minmax_ukernel__f16c_x16(
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

  const __m256 vy_min = _mm256_load_ps(params->avx.min);
  const __m256 vy_max = _mm256_load_ps(params->avx.max);

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

void xnn_f16_vsubc_minmax_ukernel__f16c_x16(
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

  const __m256 vy_min = _mm256_load_ps(params->avx.min);
  const __m256 vy_max = _mm256_load_ps(params->avx.max);

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

void xnn_f16_vclamp_ukernel__f16c_x16(
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

  const __m256 vy_min = _mm256_load_ps(params->avx.min);
  const __m256 vy_max = _mm256_load_ps(params->avx.max);

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

void xnn_f16_vhswish_ukernel__f16c_x16(
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

  const __m256 vsixth = _mm256_load_ps(params->avx.sixth);
  const __m256 vthree = _mm256_load_ps(params->avx.three);
  const __m128i vsix = _mm_load_si128((const __m128i*) params->avx.six);
  const __m128i vzero = _mm_setzero_si128();

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

void xnn_f16_vlrelu_ukernel__f16c_x16(
    size_t batch,
    const void* input,
    void* output,
    const union xnn_f16_lrelu_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(uint16_t) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const __m256 vslope = _mm256_load_ps(params->avx.slope);
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

void xnn_f16_vrndd_ukernel__f16c_x16(
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

void xnn_f16_vrndne_ukernel__f16c_x16(
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

void xnn_f16_vrndu_ukernel__f16c_x16(
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

void xnn_f16_vrndz_ukernel__f16c_x16(
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

void xnn_f16_vsqrt_ukernel__f16c_sqrt_x8(
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
  for (; batch >= 8 * sizeof(uint16_t); batch -= 8 * sizeof(uint16_t)) {
    __m256 vacc = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i));
    i += 8;
    vacc = _mm256_sqrt_ps(vacc);
    _mm_storeu_si128((__m128i*) o, _mm256_cvtps_ph(vacc, _MM_FROUND_TO_NEAREST_INT));
    o += 8;
  }
  if XNN_UNLIKELY(batch != 0) {
    __m256 vacc = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i));
    vacc = _mm256_sqrt_ps(vacc);
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

void xnn_f16_vsqr_ukernel__f16c_x16(
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

void xnn_f32_f16_vcvt_ukernel__f16c_x16(
    size_t batch,
    const float* input,
    void* output,
    const union xnn_f32_f16_cvt_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

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
    const __m256i vmask = _mm256_loadu_si256((const __m256i*) ((uintptr_t) &params->f16c.mask_table[7] - batch));

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
