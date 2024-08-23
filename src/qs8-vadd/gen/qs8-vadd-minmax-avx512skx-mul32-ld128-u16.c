// Auto-generated file. Do not edit!
//   Template: src/qs8-vadd/avx512skx-mul32-ld128.c.in
//   Generator: tools/xngen
//
// Copyright 2021 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <immintrin.h>

#include "xnnpack/common.h"
#include "xnnpack/intrinsics-polyfill.h"
#include "xnnpack/vbinary.h"


void xnn_qs8_vadd_minmax_ukernel__avx512skx_mul32_ld128_u16(
    size_t batch,
    const int8_t* input_a,
    const int8_t* input_b,
    int8_t* output,
    const union xnn_qs8_add_minmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(int8_t) == 0);
  assert(input_a != NULL);
  assert(input_b != NULL);
  assert(output != NULL);

  const __m512i vbias = _mm512_set1_epi32(params->scalar.bias);
  const __m512i va_multiplier = _mm512_set1_epi32(params->scalar.a_multiplier);
  const __m512i vb_multiplier = _mm512_set1_epi32(params->scalar.b_multiplier);
  const __m128i vshift = _mm_set1_epi64x(params->scalar.shift);
  const __m256i voutput_zero_point = _mm256_set1_epi16(params->scalar.output_zero_point);
  const __m128i voutput_min = _mm_set1_epi8(params->scalar.output_min);
  const __m128i voutput_max = _mm_set1_epi8(params->scalar.output_max);

  XNN_FORCE_REALIZATION(vbias);
  XNN_FORCE_REALIZATION(va_multiplier);
  XNN_FORCE_REALIZATION(vb_multiplier);
  XNN_FORCE_REALIZATION(vshift);
  XNN_FORCE_REALIZATION(voutput_zero_point);
  XNN_FORCE_REALIZATION(voutput_min);
  XNN_FORCE_REALIZATION(voutput_max);

  for (; batch >= 16 * sizeof(int8_t); batch -= 16 * sizeof(int8_t)) {
    const __m512i va0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) input_a));
    const __m512i vb0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_loadu_si128((const __m128i*) input_b));
    input_a += 16;
    input_b += 16;

    __m512i vacc0123456789ABCDEF = _mm512_add_epi32(vbias, _mm512_mullo_epi32(va0123456789ABCDEF, va_multiplier));

    vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vb0123456789ABCDEF, vb_multiplier));

    vacc0123456789ABCDEF = _mm512_sra_epi32(vacc0123456789ABCDEF, vshift);

    __m256i vout012389AB4567CDEF = _mm256_adds_epi16(_mm256_packs_epi32(_mm512_castsi512_si256(vacc0123456789ABCDEF), _mm512_extracti32x8_epi32(vacc0123456789ABCDEF, 1)), voutput_zero_point);

    __m128i vout0123456789ABCDEF = _mm_shuffle_epi32(_mm_packs_epi16(_mm256_castsi256_si128(vout012389AB4567CDEF), _mm256_extracti128_si256(vout012389AB4567CDEF, 1)), _MM_SHUFFLE(3, 1, 2, 0));

    vout0123456789ABCDEF = _mm_max_epi8(vout0123456789ABCDEF, voutput_min);

    vout0123456789ABCDEF = _mm_min_epi8(vout0123456789ABCDEF, voutput_max);

    _mm_storeu_si128((__m128i*) output, vout0123456789ABCDEF);
    output += 16;
  }
  if XNN_UNLIKELY(batch != 0) {
    {
      const __mmask16 vmask = _cvtu32_mask16((uint32_t) ((UINT32_C(1) << batch) - UINT32_C(1)));
      const __m512i va0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_maskz_loadu_epi8(vmask, input_a));
      const __m512i vb0123456789ABCDEF = _mm512_cvtepi8_epi32(_mm_maskz_loadu_epi8(vmask, input_b));

      __m512i vacc0123456789ABCDEF = _mm512_add_epi32(vbias, _mm512_mullo_epi32(va0123456789ABCDEF, va_multiplier));

      vacc0123456789ABCDEF = _mm512_add_epi32(vacc0123456789ABCDEF, _mm512_mullo_epi32(vb0123456789ABCDEF, vb_multiplier));

      vacc0123456789ABCDEF = _mm512_sra_epi32(vacc0123456789ABCDEF, vshift);

      __m256i vout012389AB4567CDEF = _mm256_adds_epi16(_mm256_packs_epi32(_mm512_castsi512_si256(vacc0123456789ABCDEF), _mm512_extracti32x8_epi32(vacc0123456789ABCDEF, 1)), voutput_zero_point);
      __m128i vout0123456789ABCDEF = _mm_shuffle_epi32(_mm_packs_epi16(_mm256_castsi256_si128(vout012389AB4567CDEF), _mm256_extracti128_si256(vout012389AB4567CDEF, 1)), _MM_SHUFFLE(3, 1, 2, 0));
      vout0123456789ABCDEF = _mm_max_epi8(vout0123456789ABCDEF, voutput_min);
      vout0123456789ABCDEF = _mm_min_epi8(vout0123456789ABCDEF, voutput_max);

      _mm_mask_storeu_epi8(output, vmask, vout0123456789ABCDEF);
    }
  }
}
