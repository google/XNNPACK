// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <immintrin.h>

#include "xnnpack/common.h"
#include "xnnpack/intrinsics-polyfill.h"
#include "xnnpack/maxpool.h"


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
