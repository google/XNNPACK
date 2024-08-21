// Auto-generated file. Do not edit!
//   Template: src/f16-vmulcaddc/fma3.c.in
//   Generator: tools/xngen
//
// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <immintrin.h>

#include "xnnpack/intrinsics-polyfill.h"
#include "xnnpack/math.h"
#include "xnnpack/vmulcaddc.h"


void xnn_f16_vmulcaddc_minmax_ukernel_c16__fma3_2x(
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
    for (; c >= 16 * sizeof(uint16_t); c -= 16 * sizeof(uint16_t)) {
      const __m256 vscale01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) w));
      const __m256 vscale89ABCDEF = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (w + 8)));

      __m256 vacc0x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i0));
      __m256 vacc0x89ABCDEF = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i0 + 8)));
      i0 += 16;
      __m256 vacc1x01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i1));
      __m256 vacc1x89ABCDEF = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i1 + 8)));
      i1 += 16;

      const __m256 vbias01234567 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (w + 16)));
      const __m256 vbias89ABCDEF = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (w + 24)));
      w += 32;

      vacc0x01234567 = _mm256_fmadd_ps(vacc0x01234567, vscale01234567, vbias01234567);
      vacc0x89ABCDEF = _mm256_fmadd_ps(vacc0x89ABCDEF, vscale89ABCDEF, vbias89ABCDEF);
      vacc1x01234567 = _mm256_fmadd_ps(vacc1x01234567, vscale01234567, vbias01234567);
      vacc1x89ABCDEF = _mm256_fmadd_ps(vacc1x89ABCDEF, vscale89ABCDEF, vbias89ABCDEF);

      vacc0x01234567 = _mm256_max_ps(vacc0x01234567, vmin);
      vacc0x89ABCDEF = _mm256_max_ps(vacc0x89ABCDEF, vmin);
      vacc1x01234567 = _mm256_max_ps(vacc1x01234567, vmin);
      vacc1x89ABCDEF = _mm256_max_ps(vacc1x89ABCDEF, vmin);

      vacc0x01234567 = _mm256_min_ps(vacc0x01234567, vmax);
      vacc0x89ABCDEF = _mm256_min_ps(vacc0x89ABCDEF, vmax);
      vacc1x01234567 = _mm256_min_ps(vacc1x01234567, vmax);
      vacc1x89ABCDEF = _mm256_min_ps(vacc1x89ABCDEF, vmax);

      _mm_storeu_si128((__m128i*) o0, _mm256_cvtps_ph(vacc0x01234567, _MM_FROUND_TO_NEAREST_INT));
      _mm_storeu_si128((__m128i*) (o0 + 8), _mm256_cvtps_ph(vacc0x89ABCDEF, _MM_FROUND_TO_NEAREST_INT));
      o0 += 16;
      _mm_storeu_si128((__m128i*) o1, _mm256_cvtps_ph(vacc1x01234567, _MM_FROUND_TO_NEAREST_INT));
      _mm_storeu_si128((__m128i*) (o1 + 8), _mm256_cvtps_ph(vacc1x89ABCDEF, _MM_FROUND_TO_NEAREST_INT));
      o1 += 16;
    }
    for (; c >= 8 * sizeof(uint16_t); c -= 8 * sizeof(uint16_t)) {
      const __m256 vscale = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) w));

      __m256 vacc0 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i0));
      i0 += 8;
      __m256 vacc1 = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i1));
      i1 += 8;

      const __m256 vbias = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (w + 16)));
      w += 8;

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

      const __m256 vbias = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (w + 16)));

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
