// Auto-generated file. Do not edit!
//   Template: src/f16-ibilinear/fma3.c.in
//   Generator: tools/xngen
//
// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <immintrin.h>

#include "xnnpack/common.h"
#include "xnnpack/ibilinear.h"
#include "xnnpack/intrinsics-polyfill.h"


void xnn_f16_ibilinear_ukernel__fma3_c16(
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
    for (; c >= 16 * sizeof(uint16_t); c -= 16 * sizeof(uint16_t)) {
      const __m256 vtl456789AB = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i0));
      const __m256 vtr456789AB = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i1));
      const __m256 vbl456789AB = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i2));
      const __m256 vbr456789AB = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) i3));
      const __m256 vtlCDEFGHIJ = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i0 + 8)));
      const __m256 vtrCDEFGHIJ = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i1 + 8)));
      const __m256 vblCDEFGHIJ = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i2 + 8)));
      const __m256 vbrCDEFGHIJ = _mm256_cvtph_ps(_mm_loadu_si128((const __m128i*) (i3 + 8)));
      i0 += 16;
      i1 += 16;
      i2 += 16;
      i3 += 16;

      const __m256 vtd456789AB = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_sub_ps(vtr456789AB, vtl456789AB), _MM_FROUND_TO_NEAREST_INT));
      const __m256 vbd456789AB = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_sub_ps(vbr456789AB, vbl456789AB), _MM_FROUND_TO_NEAREST_INT));
      const __m256 vtdCDEFGHIJ = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_sub_ps(vtrCDEFGHIJ, vtlCDEFGHIJ), _MM_FROUND_TO_NEAREST_INT));
      const __m256 vbdCDEFGHIJ = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_sub_ps(vbrCDEFGHIJ, vblCDEFGHIJ), _MM_FROUND_TO_NEAREST_INT));

      const __m256 vt456789AB = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vtd456789AB, valphah, vtl456789AB), _MM_FROUND_TO_NEAREST_INT));
      const __m256 vb456789AB = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vbd456789AB, valphah, vbl456789AB), _MM_FROUND_TO_NEAREST_INT));
      const __m256 vtCDEFGHIJ = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vtdCDEFGHIJ, valphah, vtlCDEFGHIJ), _MM_FROUND_TO_NEAREST_INT));
      const __m256 vbCDEFGHIJ = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_fmadd_ps(vbdCDEFGHIJ, valphah, vblCDEFGHIJ), _MM_FROUND_TO_NEAREST_INT));

      const __m256 vd456789AB = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_sub_ps(vb456789AB, vt456789AB), _MM_FROUND_TO_NEAREST_INT));
      const __m256 vdCDEFGHIJ = _mm256_cvtph_ps(_mm256_cvtps_ph(_mm256_sub_ps(vbCDEFGHIJ, vtCDEFGHIJ), _MM_FROUND_TO_NEAREST_INT));

      const __m128i vo456789AB = _mm256_cvtps_ph(_mm256_fmadd_ps(vd456789AB, valphav, vt456789AB), _MM_FROUND_TO_NEAREST_INT);
      const __m128i voCDEFGHIJ = _mm256_cvtps_ph(_mm256_fmadd_ps(vdCDEFGHIJ, valphav, vtCDEFGHIJ), _MM_FROUND_TO_NEAREST_INT);

      _mm_storeu_si128((__m128i*) o, vo456789AB);
      _mm_storeu_si128((__m128i*) (o + 8), voCDEFGHIJ);
      o += 16;
    }
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
