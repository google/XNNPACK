// Auto-generated file. Do not edit!
//   Template: src/f32-qs8-vcvt/avx512skx.c.in
//   Generator: tools/xngen
//
// Copyright 2021 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <immintrin.h>

#include <xnnpack/common.h>
#include <xnnpack/intrinsics-polyfill.h>
#include <xnnpack/vcvt.h>


void xnn_f32_qu8_vcvt_ukernel__avx512skx_u32(
    size_t batch,
    const float* input,
    uint8_t* output,
    const union xnn_f32_qu8_cvt_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const __m512 vscale = _mm512_load_ps(params->avx512.scale);
  const __m512 voutput_max_less_zero_point = _mm512_load_ps(params->avx512.output_max_less_zero_point);
  const __m512i voutput_zero_point = _mm512_load_si512(params->avx512.output_zero_point);
  const __m256i vshuffle256_mask = _mm256_load_si256((const __m256i*) params->avx512.shuffle256_mask);
  const __m256i voutput_min = _mm256_load_si256((const __m256i*) params->avx512.output_min);
  for (; batch >= 32 * sizeof(float); batch -= 32 * sizeof(float)) {
    __m512 vx0123 = _mm512_loadu_ps(input);
    __m512 vx4567 = _mm512_loadu_ps(input + 16);
    input += 32;

    vx0123 = _mm512_mul_ps(vx0123, vscale);
    vx4567 = _mm512_mul_ps(vx4567, vscale);

    vx0123 = _mm512_min_ps(vx0123, voutput_max_less_zero_point);
    vx4567 = _mm512_min_ps(vx4567, voutput_max_less_zero_point);

    const __m512i vacc0123 = _mm512_cvtps_epi32(vx0123);
    const __m512i vacc4567 = _mm512_cvtps_epi32(vx4567);

    __m512i vacc04152637 = _mm512_packs_epi32(vacc0123, vacc4567);

    vacc04152637 = _mm512_adds_epi16(vacc04152637, voutput_zero_point);

    __m256i vy04261537 = _mm256_packus_epi16(_mm512_castsi512_si256(vacc04152637), _mm512_extracti32x8_epi32(vacc04152637, 1));

    vy04261537 = _mm256_max_epu8(vy04261537, voutput_min);

    const __m256i vy01234567 = _mm256_permutevar8x32_epi32(vy04261537, vshuffle256_mask);

    _mm256_storeu_si256((__m256i*) output, vy01234567);
    output += 32;
  }
  for (; batch >= 16 * sizeof(float); batch -= 16 * sizeof(float)) {
    __m512 vx0123 = _mm512_loadu_ps(input);
    vx0123 = _mm512_mul_ps(vx0123, vscale);
    vx0123 = _mm512_min_ps(vx0123, voutput_max_less_zero_point);
    input += 16;

    const __m512i vacc0123 = _mm512_cvtps_epi32(vx0123);

    __m256i vacc0213 = _mm256_packs_epi32(_mm512_castsi512_si256(vacc0123), _mm512_extracti32x8_epi32(vacc0123, 1));
    vacc0213 = _mm256_adds_epi16(vacc0213, _mm512_castsi512_si256(voutput_zero_point));
    const __m128i vy0213 = _mm_packus_epi16(_mm256_castsi256_si128(vacc0213), _mm256_extracti128_si256(vacc0213, 1));
    __m128i vy0123 = _mm_shuffle_epi32(vy0213, _MM_SHUFFLE(3, 1, 2, 0));
    vy0123 = _mm_max_epu8(vy0123, _mm256_castsi256_si128(voutput_min));

    _mm_storeu_si128((__m128i*) output, vy0123);
    output += 16;
  }
  if XNN_UNLIKELY(batch != 0) {
    assert(batch >= 1 * sizeof(float));
    assert(batch <= 15 * sizeof(float));

    // Prepare mask for valid elements (depends on batch).
    batch >>= XNN_LOG2_SIZEOF_FLOAT;
    const __mmask16 vmask = _cvtu32_mask16((uint32_t) ((UINT32_C(1) << batch) - UINT32_C(1)));

    __m512 vx0123 = _mm512_maskz_loadu_ps(vmask, input);
    vx0123 = _mm512_mul_ps(vx0123, vscale);
    vx0123 = _mm512_min_ps(vx0123, voutput_max_less_zero_point);

    const __m512i vacc0123 = _mm512_cvtps_epi32(vx0123);

    __m256i vacc0213 = _mm256_packs_epi32(_mm512_castsi512_si256(vacc0123), _mm512_extracti32x8_epi32(vacc0123, 1));
    vacc0213 = _mm256_adds_epi16(vacc0213, _mm512_castsi512_si256(voutput_zero_point));
    const __m128i vy0213 = _mm_packus_epi16(_mm256_castsi256_si128(vacc0213), _mm256_extracti128_si256(vacc0213, 1));
    __m128i vy0123 = _mm_shuffle_epi32(vy0213, _MM_SHUFFLE(3, 1, 2, 0));
    vy0123 = _mm_max_epu8(vy0123, _mm256_castsi256_si128(voutput_min));

    _mm_mask_storeu_epi8(output, vmask, vy0123);
  }
}
