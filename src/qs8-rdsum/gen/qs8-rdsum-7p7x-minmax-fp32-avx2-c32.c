// Auto-generated file. Do not edit!
//   Template: src/qs8-rdsum/avx2.c.in
//   Generator: tools/xngen
//
// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <math.h>

#include <immintrin.h>

#include "xnnpack/unaligned.h"
#include "xnnpack/common.h"
#include "xnnpack/reduce.h"
#include "xnnpack/math.h"


void xnn_qs8_rdsum_ukernel_7p7x__avx2_c32(
    size_t rows,
    size_t channels,
    const int8_t* input,
    size_t input_stride,
    const int8_t* zero,
    int32_t* output,
    const union xnn_qs8_rsum_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(rows != 0);
  assert(channels != 0);
  assert(input != NULL);
  assert(output != NULL);

  size_t input_increment = 7 * input_stride;
  for (; channels >= 32; channels -= 32) {
    const int8_t* i0 = input;
    const int8_t* i1 = (const int8_t*) ((uintptr_t) input + 1 * input_stride);
    const int8_t* i2 = (const int8_t*) ((uintptr_t) input + 2 * input_stride);
    const int8_t* i3 = (const int8_t*) ((uintptr_t) input + 3 * input_stride);
    const int8_t* i4 = (const int8_t*) ((uintptr_t) input + 4 * input_stride);
    const int8_t* i5 = (const int8_t*) ((uintptr_t) input + 5 * input_stride);
    const int8_t* i6 = (const int8_t*) ((uintptr_t) input + 6 * input_stride);

    __m256i vacc01234567 = _mm256_setzero_si256();
    __m256i vacc89ABCDEF = _mm256_setzero_si256();
    __m256i vaccGHIJKLMN = _mm256_setzero_si256();
    __m256i vaccOPQRSTUV = _mm256_setzero_si256();

    // 256 int8s may be summed into an int16 before overflowing
    // To prevent handling the tails of the inner 256 loop, we round 256 down to
    // the nearest integer multiple of ACCUMULATORS.
    int r = rows;
    while (r > 0) {
      __m256i vacc16_0123456789ABCDEF = _mm256_setzero_si256();
      __m256i vacc16_GHIJKLMNOPQRSTUV = _mm256_setzero_si256();
      for (int current_batch = min(r, 252); current_batch > 0; current_batch -= 7) {
        if XNN_UNPREDICTABLE(current_batch < 2) {
          i1 = zero;
        }
        if XNN_UNPREDICTABLE(current_batch <= 2) {
          i2 = zero;
        }
        if XNN_UNPREDICTABLE(current_batch < 4) {
          i3 = zero;
        }
        if XNN_UNPREDICTABLE(current_batch <= 4) {
          i4 = zero;
        }
        if XNN_UNPREDICTABLE(current_batch < 6) {
          i5 = zero;
        }
        if XNN_UNPREDICTABLE(current_batch <= 6) {
          i6 = zero;
        }
        __m256i vin0123456789ABCDEF;
        __m256i vinGHIJKLMNOPQRSTUV;
        vin0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) &i0[0]));
        vinGHIJKLMNOPQRSTUV = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) &i0[16]));
        vacc16_0123456789ABCDEF = _mm256_add_epi16(vacc16_0123456789ABCDEF, vin0123456789ABCDEF);
        vacc16_GHIJKLMNOPQRSTUV = _mm256_add_epi16(vacc16_GHIJKLMNOPQRSTUV, vinGHIJKLMNOPQRSTUV);
        vin0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) &i1[0]));
        vinGHIJKLMNOPQRSTUV = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) &i1[16]));
        vacc16_0123456789ABCDEF = _mm256_add_epi16(vacc16_0123456789ABCDEF, vin0123456789ABCDEF);
        vacc16_GHIJKLMNOPQRSTUV = _mm256_add_epi16(vacc16_GHIJKLMNOPQRSTUV, vinGHIJKLMNOPQRSTUV);
        vin0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) &i2[0]));
        vinGHIJKLMNOPQRSTUV = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) &i2[16]));
        vacc16_0123456789ABCDEF = _mm256_add_epi16(vacc16_0123456789ABCDEF, vin0123456789ABCDEF);
        vacc16_GHIJKLMNOPQRSTUV = _mm256_add_epi16(vacc16_GHIJKLMNOPQRSTUV, vinGHIJKLMNOPQRSTUV);
        vin0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) &i3[0]));
        vinGHIJKLMNOPQRSTUV = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) &i3[16]));
        vacc16_0123456789ABCDEF = _mm256_add_epi16(vacc16_0123456789ABCDEF, vin0123456789ABCDEF);
        vacc16_GHIJKLMNOPQRSTUV = _mm256_add_epi16(vacc16_GHIJKLMNOPQRSTUV, vinGHIJKLMNOPQRSTUV);
        vin0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) &i4[0]));
        vinGHIJKLMNOPQRSTUV = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) &i4[16]));
        vacc16_0123456789ABCDEF = _mm256_add_epi16(vacc16_0123456789ABCDEF, vin0123456789ABCDEF);
        vacc16_GHIJKLMNOPQRSTUV = _mm256_add_epi16(vacc16_GHIJKLMNOPQRSTUV, vinGHIJKLMNOPQRSTUV);
        vin0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) &i5[0]));
        vinGHIJKLMNOPQRSTUV = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) &i5[16]));
        vacc16_0123456789ABCDEF = _mm256_add_epi16(vacc16_0123456789ABCDEF, vin0123456789ABCDEF);
        vacc16_GHIJKLMNOPQRSTUV = _mm256_add_epi16(vacc16_GHIJKLMNOPQRSTUV, vinGHIJKLMNOPQRSTUV);
        vin0123456789ABCDEF = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) &i6[0]));
        vinGHIJKLMNOPQRSTUV = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) &i6[16]));
        vacc16_0123456789ABCDEF = _mm256_add_epi16(vacc16_0123456789ABCDEF, vin0123456789ABCDEF);
        vacc16_GHIJKLMNOPQRSTUV = _mm256_add_epi16(vacc16_GHIJKLMNOPQRSTUV, vinGHIJKLMNOPQRSTUV);
        i0 = (const int8_t*) ((uintptr_t) i0 + input_increment);
        i1 = (const int8_t*) ((uintptr_t) i1 + input_increment);
        i2 = (const int8_t*) ((uintptr_t) i2 + input_increment);
        i3 = (const int8_t*) ((uintptr_t) i3 + input_increment);
        i4 = (const int8_t*) ((uintptr_t) i4 + input_increment);
        i5 = (const int8_t*) ((uintptr_t) i5 + input_increment);
        i6 = (const int8_t*) ((uintptr_t) i6 + input_increment);
      }
      vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_cvtepi16_epi32(_mm256_castsi256_si128(vacc16_0123456789ABCDEF)));
      vacc89ABCDEF = _mm256_add_epi32(vacc89ABCDEF, _mm256_cvtepi16_epi32(_mm256_extractf128_si256(vacc16_0123456789ABCDEF, 1)));
      vaccGHIJKLMN = _mm256_add_epi32(vaccGHIJKLMN, _mm256_cvtepi16_epi32(_mm256_castsi256_si128(vacc16_GHIJKLMNOPQRSTUV)));
      vaccOPQRSTUV = _mm256_add_epi32(vaccOPQRSTUV, _mm256_cvtepi16_epi32(_mm256_extractf128_si256(vacc16_GHIJKLMNOPQRSTUV, 1)));
      r = doz(r, 252);
    }

    const int32_t* o = output;
    __m256i vo01234567 = _mm256_loadu_si256((const __m256i*) o); o += 8;
    __m256i vo89ABCDEF = _mm256_loadu_si256((const __m256i*) o); o += 8;
    __m256i voGHIJKLMN = _mm256_loadu_si256((const __m256i*) o); o += 8;
    __m256i voOPQRSTUV = _mm256_loadu_si256((const __m256i*) o); o += 8;
    vo01234567 = _mm256_add_epi32(vo01234567, vacc01234567);
    vo89ABCDEF = _mm256_add_epi32(vo89ABCDEF, vacc89ABCDEF);
    voGHIJKLMN = _mm256_add_epi32(voGHIJKLMN, vaccGHIJKLMN);
    voOPQRSTUV = _mm256_add_epi32(voOPQRSTUV, vaccOPQRSTUV);
    _mm256_storeu_si256((__m256i*) output, vo01234567); output += 8;
    _mm256_storeu_si256((__m256i*) output, vo89ABCDEF); output += 8;
    _mm256_storeu_si256((__m256i*) output, voGHIJKLMN); output += 8;
    _mm256_storeu_si256((__m256i*) output, voOPQRSTUV); output += 8;

    input = (const int8_t*) ((uintptr_t) input + 32 * sizeof(int8_t));
  }
  if (channels != 0) {
    input_increment = 7 * input_stride;
    // 256 int8s may be summed into an int16 before overflowing.
    do {
      int num_batches = floor((rows + 251) / 252);
      int r = rows;
      const int8_t* i0 = input;
      const int8_t* i1 = (const int8_t*) ((uintptr_t) input + 1 * input_stride);
      const int8_t* i2 = (const int8_t*) ((uintptr_t) input + 2 * input_stride);
      const int8_t* i3 = (const int8_t*) ((uintptr_t) input + 3 * input_stride);
      const int8_t* i4 = (const int8_t*) ((uintptr_t) input + 4 * input_stride);
      const int8_t* i5 = (const int8_t*) ((uintptr_t) input + 5 * input_stride);
      const int8_t* i6 = (const int8_t*) ((uintptr_t) input + 6 * input_stride);

      __m256i vacc01234567 = _mm256_setzero_si256();
      __m256i vacc89ABCDEF = _mm256_setzero_si256();

      for (; num_batches > 0; --num_batches) {
        __m256i vacc16_0123456789ABCDEF = _mm256_setzero_si256();
        for (int current_batch = min(r, 252); current_batch > 0; current_batch -= 7) {
          if XNN_UNPREDICTABLE(current_batch < 2) {
            i1 = zero;
          }
          if XNN_UNPREDICTABLE(current_batch <= 2) {
            i2 = zero;
          }
          if XNN_UNPREDICTABLE(current_batch < 4) {
            i3 = zero;
          }
          if XNN_UNPREDICTABLE(current_batch <= 4) {
            i4 = zero;
          }
          if XNN_UNPREDICTABLE(current_batch < 6) {
            i5 = zero;
          }
          if XNN_UNPREDICTABLE(current_batch <= 6) {
            i6 = zero;
          }

          __m256i vin0 = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*)&i0[0]));
          __m256i vin1 = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*)&i1[0]));
          __m256i vin2 = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*)&i2[0]));
          __m256i vin3 = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*)&i3[0]));
          __m256i vin4 = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*)&i4[0]));
          __m256i vin5 = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*)&i5[0]));
          __m256i vin6 = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*)&i6[0]));
          vacc16_0123456789ABCDEF = _mm256_add_epi16(vacc16_0123456789ABCDEF, vin0);
          vacc16_0123456789ABCDEF = _mm256_add_epi16(vacc16_0123456789ABCDEF, vin1);
          vacc16_0123456789ABCDEF = _mm256_add_epi16(vacc16_0123456789ABCDEF, vin2);
          vacc16_0123456789ABCDEF = _mm256_add_epi16(vacc16_0123456789ABCDEF, vin3);
          vacc16_0123456789ABCDEF = _mm256_add_epi16(vacc16_0123456789ABCDEF, vin4);
          vacc16_0123456789ABCDEF = _mm256_add_epi16(vacc16_0123456789ABCDEF, vin5);
          vacc16_0123456789ABCDEF = _mm256_add_epi16(vacc16_0123456789ABCDEF, vin6);
          i0 = (const int8_t*) ((uintptr_t) i0 + input_increment);
          i1 = (const int8_t*) ((uintptr_t) i1 + input_increment);
          i2 = (const int8_t*) ((uintptr_t) i2 + input_increment);
          i3 = (const int8_t*) ((uintptr_t) i3 + input_increment);
          i4 = (const int8_t*) ((uintptr_t) i4 + input_increment);
          i5 = (const int8_t*) ((uintptr_t) i5 + input_increment);
          i6 = (const int8_t*) ((uintptr_t) i6 + input_increment);
        }
        vacc01234567 = _mm256_add_epi32(vacc01234567, _mm256_cvtepi16_epi32(_mm256_castsi256_si128(vacc16_0123456789ABCDEF)));
        vacc89ABCDEF = _mm256_add_epi32(vacc89ABCDEF, _mm256_cvtepi16_epi32(_mm256_extractf128_si256(vacc16_0123456789ABCDEF, 1)));
        r = doz(r, 252);
      }

      if XNN_LIKELY(channels >= 8) {
        __m256i vo01234567 = _mm256_loadu_si256((const __m256i*) output);
        vo01234567 = _mm256_add_epi32(vo01234567, vacc01234567);
        _mm256_storeu_si256((__m256i*) output, vo01234567); output += 8;
        channels -= 8;
        input = (const int8_t*) ((uintptr_t) input + 8 * sizeof(int8_t));
      } else {
        __m128i vacc0123 = _mm256_castsi256_si128(vacc01234567);
        if (channels & 4) {
          __m128i vo0123 = _mm_loadu_si128((const __m128i*) output);
          vo0123 = _mm_add_epi32(vo0123, vacc0123);
          _mm_storeu_si128((__m128i*) output, vo0123); output += 4;
          vacc0123 = _mm256_extractf128_si256(vacc01234567, 1);
        }
        if (channels & 2) {
          __m128i vo01 = _mm_loadl_epi64((const __m128i*) output);
          vo01 = _mm_add_epi32(vo01, vacc0123);
          _mm_storel_epi64((__m128i*) output, vo01); output += 2;
          vacc0123 = _mm_srli_si128(vacc0123, 8);
        }
        if (channels & 1) {
          *output += _mm_cvtsi128_si32(vacc0123);
        }
        channels = 0;
      }
    } while (channels != 0);
  }
}
