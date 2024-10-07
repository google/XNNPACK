// Auto-generated file. Do not edit!
//   Template: src/qs8-prelu/avx2.c.in
//   Generator: tools/xngen
//

#include <assert.h>

#include <immintrin.h>

#include "xnnpack/common.h"
#include "xnnpack/intrinsics-polyfill.h"
#include "xnnpack/math.h"
#include "xnnpack/prelu.h"


void xnn_qs8_prelu_ukernel__avx2_2x16(
    size_t rows,
    size_t channels,
    const int8_t* restrict input,
    size_t input_stride,
    const void* restrict weights,
    int8_t* restrict output,
    size_t output_stride,
    const struct xnn_qs8_prelu_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(rows != 0);
  assert(channels != 0);
  assert(channels % sizeof(int8_t) == 0);
  assert(input != NULL);
  assert(output != NULL);
  
  const int8_t* i0 = (const int8_t*) input;
  int8_t* o0 = (int8_t*) output;
  const __m256i vinput_zero_point = _mm256_set1_epi16(params->scalar.input_zero_point);
  const __m256i vpositive_multiplier = _mm256_set1_epi16(params->scalar.positive_multiplier);
  const __m256i voutput_zero_point = _mm256_set1_epi16(params->scalar.output_zero_point);

  const int8_t* i1 = (const int8_t*) ((uintptr_t) i0 + input_stride);
  int8_t* o1 = (int8_t*) ((uintptr_t) o0 + output_stride);

  const size_t input_increment = input_stride * 2 - channels;
  const size_t output_increment = output_stride * 2 - channels;

  do {
    if XNN_UNPREDICTABLE(rows < 2) {
      i1 = i0;
      o1 = o0;
    }

    const int16_t* w = weights;
    size_t c = channels;

    for (; c >= 16 * sizeof(int8_t); c -= 16 * sizeof(int8_t)) {
        __m256i vacc0 = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) i0));
        i0 += 16;
        __m256i vacc1 = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) i1));
        i1 += 16;

        const __m256i vnegative_multiplier = _mm256_loadu_si256((const __m256i*) w);
        w += 16;

        __m256i vmultiplier0 = _mm256_cmpgt_epi16(vacc0, vinput_zero_point);
        vacc0 = _mm256_sub_epi16(vinput_zero_point, vacc0);
        __m256i vmultiplier1 = _mm256_cmpgt_epi16(vacc1, vinput_zero_point);
        vacc1 = _mm256_sub_epi16(vinput_zero_point, vacc1);

        vmultiplier0 = _mm256_blendv_epi8(vnegative_multiplier, vpositive_multiplier, vmultiplier0);
        vacc0 = _mm256_slli_epi16(vacc0, 7);
        vmultiplier1 = _mm256_blendv_epi8(vnegative_multiplier, vpositive_multiplier, vmultiplier1);
        vacc1 = _mm256_slli_epi16(vacc1, 7);

        vacc0 = _mm256_mulhrs_epi16(vacc0, vmultiplier0);
        vacc1 = _mm256_mulhrs_epi16(vacc1, vmultiplier1);

        vacc0 = _mm256_adds_epi16(vacc0, voutput_zero_point);
        vacc1 = _mm256_adds_epi16(vacc1, voutput_zero_point);

        const __m128i vacc0_hi = _mm256_extracti128_si256(vacc0, 1);
        const __m128i vy0 = _mm_packs_epi16(_mm256_castsi256_si128(vacc0), vacc0_hi);
        _mm_storeu_si128((__m128i*) o0, vy0);
        o0 += 16;
        const __m128i vacc1_hi = _mm256_extracti128_si256(vacc1, 1);
        const __m128i vy1 = _mm_packs_epi16(_mm256_castsi256_si128(vacc1), vacc1_hi);
        _mm_storeu_si128((__m128i*) o1, vy1);
        o1 += 16;
    }
    if XNN_UNLIKELY(c != 0) {
        assert(c >= 1 * sizeof(int8_t));
        assert(c <= 15 * sizeof(int8_t));

        __m256i vacc0 = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) i0));
        __m256i vacc1 = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) i1));

        const __m256i vnegative_multiplier = _mm256_loadu_si256((const __m256i*) w);

        __m256i vmultiplier0 = _mm256_cmpgt_epi16(vacc0, vinput_zero_point);
        vacc0 = _mm256_sub_epi16(vinput_zero_point, vacc0);
        __m256i vmultiplier1 = _mm256_cmpgt_epi16(vacc1, vinput_zero_point);
        vacc1 = _mm256_sub_epi16(vinput_zero_point, vacc1);

        vmultiplier0 = _mm256_blendv_epi8(vnegative_multiplier, vpositive_multiplier, vmultiplier0);
        vacc0 = _mm256_slli_epi16(vacc0, 7);
        vmultiplier1 = _mm256_blendv_epi8(vnegative_multiplier, vpositive_multiplier, vmultiplier1);
        vacc1 = _mm256_slli_epi16(vacc1, 7);

        vacc0 = _mm256_mulhrs_epi16(vacc0, vmultiplier0);
        vacc1 = _mm256_mulhrs_epi16(vacc1, vmultiplier1);

        vacc0 = _mm256_adds_epi16(vacc0, voutput_zero_point);
        vacc1 = _mm256_adds_epi16(vacc1, voutput_zero_point);

        const __m128i vacc0_hi = _mm256_extracti128_si256(vacc0, 1);
        __m128i vy0 = _mm_packs_epi16(_mm256_castsi256_si128(vacc0), vacc0_hi);
        const __m128i vacc1_hi = _mm256_extracti128_si256(vacc1, 1);
        __m128i vy1 = _mm_packs_epi16(_mm256_castsi256_si128(vacc1), vacc1_hi);

        if (c & (8 * sizeof(int8_t))) {
            _mm_storel_epi64((__m128i*) o0, vy0);
            vy0 = _mm_unpackhi_epi64(vy0, vy0);
            o0 += 8;
            _mm_storel_epi64((__m128i*) o1, vy1);
            vy1 = _mm_unpackhi_epi64(vy1, vy1);
            o1 += 8;
        }
        if (c & (4 * sizeof(int8_t))) {
            _mm_storeu_si32(o0, vy0);
            vy0 = _mm_srli_epi64(vy0, 32);
            o0 += 4;
            _mm_storeu_si32(o1, vy1);
            vy1 = _mm_srli_epi64(vy1, 32);
            o1 += 4;
        }
        if (c & (2 * sizeof(int8_t))) {
            _mm_storeu_si16(o0, vy0);
            vy0 = _mm_srli_epi32(vy0, 16);
            o0 += 2;
            _mm_storeu_si16(o1, vy1);
            vy1 = _mm_srli_epi32(vy1, 16);
            o1 += 2;
        }

        if (c & (1 * sizeof(int8_t))) {
            *o0 = (int8_t) _mm_extract_epi8(vy0, 0);
            o0 += 1;
            *o1 = (int8_t) _mm_extract_epi8(vy1, 0);
            o1 += 1;
        }
    }
    i0 = (const int8_t*) ((uintptr_t) i0 + input_increment);
    o0 = (int8_t*) ((uintptr_t) o0 + output_increment);
    i1 = (const int8_t*) ((uintptr_t) i1 + input_increment);
    o1 = (int8_t*) ((uintptr_t) o1 + output_increment);
    rows = doz(rows, 2);
  }while (rows != 0);
}
