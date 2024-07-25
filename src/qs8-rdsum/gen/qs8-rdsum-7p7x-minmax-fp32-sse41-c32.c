// Auto-generated file. Do not edit!
//   Template: src/qs8-rdsum/sse41.c.in
//   Generator: tools/xngen
//
// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <math.h>

#include <immintrin.h>

#include "xnnpack/common.h"
#include "xnnpack/intrinsics-polyfill.h"
#include "xnnpack/math.h"
#include "xnnpack/reduce.h"
#include "xnnpack/unaligned.h"


void xnn_qs8_rdsum_ukernel_7p7x__sse41_c32(
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

    __m128i vacc0123 = _mm_setzero_si128();
    __m128i vacc4567 = _mm_setzero_si128();
    __m128i vacc89AB = _mm_setzero_si128();
    __m128i vaccCDEF = _mm_setzero_si128();
    __m128i vaccGHIJ = _mm_setzero_si128();
    __m128i vaccKLMN = _mm_setzero_si128();
    __m128i vaccOPQR = _mm_setzero_si128();
    __m128i vaccSTUV = _mm_setzero_si128();

    // 256 int8s may be summed into an int16 before overflowing
    // To prevent handling the tails of the inner 256 loop, we round 256 down to
    // the nearest integer multiple of ACCUMULATORS.
    int r = rows;
    while (r > 0) {
      __m128i vacc16_01234567 = _mm_setzero_si128();
      __m128i vacc16_89ABCDEF = _mm_setzero_si128();
      __m128i vacc16_GHIJKLMN = _mm_setzero_si128();
      __m128i vacc16_OPQRSTUV = _mm_setzero_si128();
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
        __m128i vin01234567;
        __m128i vin89ABCDEF;
        __m128i vinGHIJKLMN;
        __m128i vinOPQRSTUV;
        vin01234567 = _mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*)&i0[0]));
        vin89ABCDEF = _mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*)&i0[8]));
        vinGHIJKLMN = _mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*)&i0[16]));
        vinOPQRSTUV = _mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*)&i0[24]));
        vacc16_01234567 = _mm_add_epi16(vacc16_01234567, vin01234567);
        vacc16_89ABCDEF = _mm_add_epi16(vacc16_89ABCDEF, vin89ABCDEF);
        vacc16_GHIJKLMN = _mm_add_epi16(vacc16_GHIJKLMN, vinGHIJKLMN);
        vacc16_OPQRSTUV = _mm_add_epi16(vacc16_OPQRSTUV, vinOPQRSTUV);
        vin01234567 = _mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*)&i1[0]));
        vin89ABCDEF = _mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*)&i1[8]));
        vinGHIJKLMN = _mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*)&i1[16]));
        vinOPQRSTUV = _mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*)&i1[24]));
        vacc16_01234567 = _mm_add_epi16(vacc16_01234567, vin01234567);
        vacc16_89ABCDEF = _mm_add_epi16(vacc16_89ABCDEF, vin89ABCDEF);
        vacc16_GHIJKLMN = _mm_add_epi16(vacc16_GHIJKLMN, vinGHIJKLMN);
        vacc16_OPQRSTUV = _mm_add_epi16(vacc16_OPQRSTUV, vinOPQRSTUV);
        vin01234567 = _mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*)&i2[0]));
        vin89ABCDEF = _mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*)&i2[8]));
        vinGHIJKLMN = _mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*)&i2[16]));
        vinOPQRSTUV = _mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*)&i2[24]));
        vacc16_01234567 = _mm_add_epi16(vacc16_01234567, vin01234567);
        vacc16_89ABCDEF = _mm_add_epi16(vacc16_89ABCDEF, vin89ABCDEF);
        vacc16_GHIJKLMN = _mm_add_epi16(vacc16_GHIJKLMN, vinGHIJKLMN);
        vacc16_OPQRSTUV = _mm_add_epi16(vacc16_OPQRSTUV, vinOPQRSTUV);
        vin01234567 = _mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*)&i3[0]));
        vin89ABCDEF = _mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*)&i3[8]));
        vinGHIJKLMN = _mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*)&i3[16]));
        vinOPQRSTUV = _mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*)&i3[24]));
        vacc16_01234567 = _mm_add_epi16(vacc16_01234567, vin01234567);
        vacc16_89ABCDEF = _mm_add_epi16(vacc16_89ABCDEF, vin89ABCDEF);
        vacc16_GHIJKLMN = _mm_add_epi16(vacc16_GHIJKLMN, vinGHIJKLMN);
        vacc16_OPQRSTUV = _mm_add_epi16(vacc16_OPQRSTUV, vinOPQRSTUV);
        vin01234567 = _mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*)&i4[0]));
        vin89ABCDEF = _mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*)&i4[8]));
        vinGHIJKLMN = _mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*)&i4[16]));
        vinOPQRSTUV = _mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*)&i4[24]));
        vacc16_01234567 = _mm_add_epi16(vacc16_01234567, vin01234567);
        vacc16_89ABCDEF = _mm_add_epi16(vacc16_89ABCDEF, vin89ABCDEF);
        vacc16_GHIJKLMN = _mm_add_epi16(vacc16_GHIJKLMN, vinGHIJKLMN);
        vacc16_OPQRSTUV = _mm_add_epi16(vacc16_OPQRSTUV, vinOPQRSTUV);
        vin01234567 = _mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*)&i5[0]));
        vin89ABCDEF = _mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*)&i5[8]));
        vinGHIJKLMN = _mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*)&i5[16]));
        vinOPQRSTUV = _mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*)&i5[24]));
        vacc16_01234567 = _mm_add_epi16(vacc16_01234567, vin01234567);
        vacc16_89ABCDEF = _mm_add_epi16(vacc16_89ABCDEF, vin89ABCDEF);
        vacc16_GHIJKLMN = _mm_add_epi16(vacc16_GHIJKLMN, vinGHIJKLMN);
        vacc16_OPQRSTUV = _mm_add_epi16(vacc16_OPQRSTUV, vinOPQRSTUV);
        vin01234567 = _mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*)&i6[0]));
        vin89ABCDEF = _mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*)&i6[8]));
        vinGHIJKLMN = _mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*)&i6[16]));
        vinOPQRSTUV = _mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*)&i6[24]));
        vacc16_01234567 = _mm_add_epi16(vacc16_01234567, vin01234567);
        vacc16_89ABCDEF = _mm_add_epi16(vacc16_89ABCDEF, vin89ABCDEF);
        vacc16_GHIJKLMN = _mm_add_epi16(vacc16_GHIJKLMN, vinGHIJKLMN);
        vacc16_OPQRSTUV = _mm_add_epi16(vacc16_OPQRSTUV, vinOPQRSTUV);
        i0 = (const int8_t*) ((uintptr_t) i0 + input_increment);
        i1 = (const int8_t*) ((uintptr_t) i1 + input_increment);
        i2 = (const int8_t*) ((uintptr_t) i2 + input_increment);
        i3 = (const int8_t*) ((uintptr_t) i3 + input_increment);
        i4 = (const int8_t*) ((uintptr_t) i4 + input_increment);
        i5 = (const int8_t*) ((uintptr_t) i5 + input_increment);
        i6 = (const int8_t*) ((uintptr_t) i6 + input_increment);
      }
      vacc0123 = _mm_add_epi32(vacc0123, _mm_cvtepi16_epi32(vacc16_01234567));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_cvtepi16_epi32(_mm_srli_si128(vacc16_01234567, 8)));
      vacc89AB = _mm_add_epi32(vacc89AB, _mm_cvtepi16_epi32(vacc16_89ABCDEF));
      vaccCDEF = _mm_add_epi32(vaccCDEF, _mm_cvtepi16_epi32(_mm_srli_si128(vacc16_89ABCDEF, 8)));
      vaccGHIJ = _mm_add_epi32(vaccGHIJ, _mm_cvtepi16_epi32(vacc16_GHIJKLMN));
      vaccKLMN = _mm_add_epi32(vaccKLMN, _mm_cvtepi16_epi32(_mm_srli_si128(vacc16_GHIJKLMN, 8)));
      vaccOPQR = _mm_add_epi32(vaccOPQR, _mm_cvtepi16_epi32(vacc16_OPQRSTUV));
      vaccSTUV = _mm_add_epi32(vaccSTUV, _mm_cvtepi16_epi32(_mm_srli_si128(vacc16_OPQRSTUV, 8)));
      r = doz(r, 252);
    }

    __m128i vo0123 = _mm_loadu_si128((const __m128i*) ((uintptr_t) output + 0 * sizeof(int32_t)));
    __m128i vo4567 = _mm_loadu_si128((const __m128i*) ((uintptr_t) output + 4 * sizeof(int32_t)));
    __m128i vo89AB = _mm_loadu_si128((const __m128i*) ((uintptr_t) output + 8 * sizeof(int32_t)));
    __m128i voCDEF = _mm_loadu_si128((const __m128i*) ((uintptr_t) output + 12 * sizeof(int32_t)));
    __m128i voGHIJ = _mm_loadu_si128((const __m128i*) ((uintptr_t) output + 16 * sizeof(int32_t)));
    __m128i voKLMN = _mm_loadu_si128((const __m128i*) ((uintptr_t) output + 20 * sizeof(int32_t)));
    __m128i voOPQR = _mm_loadu_si128((const __m128i*) ((uintptr_t) output + 24 * sizeof(int32_t)));
    __m128i voSTUV = _mm_loadu_si128((const __m128i*) ((uintptr_t) output + 28 * sizeof(int32_t)));
    vo0123 = _mm_add_epi32(vo0123, vacc0123);
    vo4567 = _mm_add_epi32(vo4567, vacc4567);
    vo89AB = _mm_add_epi32(vo89AB, vacc89AB);
    voCDEF = _mm_add_epi32(voCDEF, vaccCDEF);
    voGHIJ = _mm_add_epi32(voGHIJ, vaccGHIJ);
    voKLMN = _mm_add_epi32(voKLMN, vaccKLMN);
    voOPQR = _mm_add_epi32(voOPQR, vaccOPQR);
    voSTUV = _mm_add_epi32(voSTUV, vaccSTUV);
    _mm_storeu_si128((__m128i*) output, vo0123); output += 4;
    _mm_storeu_si128((__m128i*) output, vo4567); output += 4;
    _mm_storeu_si128((__m128i*) output, vo89AB); output += 4;
    _mm_storeu_si128((__m128i*) output, voCDEF); output += 4;
    _mm_storeu_si128((__m128i*) output, voGHIJ); output += 4;
    _mm_storeu_si128((__m128i*) output, voKLMN); output += 4;
    _mm_storeu_si128((__m128i*) output, voOPQR); output += 4;
    _mm_storeu_si128((__m128i*) output, voSTUV); output += 4;

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

      __m128i vacc0123 = _mm_setzero_si128();
      __m128i vacc4567 = _mm_setzero_si128();

      for (; num_batches > 0; --num_batches) {
        __m128i vacc16_01234567 = _mm_setzero_si128();
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

          __m128i vin0 = _mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*)&i0[0]));
          __m128i vin1 = _mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*)&i1[0]));
          __m128i vin2 = _mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*)&i2[0]));
          __m128i vin3 = _mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*)&i3[0]));
          __m128i vin4 = _mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*)&i4[0]));
          __m128i vin5 = _mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*)&i5[0]));
          __m128i vin6 = _mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*)&i6[0]));
          vacc16_01234567 = _mm_add_epi16(vacc16_01234567, vin0);
          vacc16_01234567 = _mm_add_epi16(vacc16_01234567, vin1);
          vacc16_01234567 = _mm_add_epi16(vacc16_01234567, vin2);
          vacc16_01234567 = _mm_add_epi16(vacc16_01234567, vin3);
          vacc16_01234567 = _mm_add_epi16(vacc16_01234567, vin4);
          vacc16_01234567 = _mm_add_epi16(vacc16_01234567, vin5);
          vacc16_01234567 = _mm_add_epi16(vacc16_01234567, vin6);
          i0 = (const int8_t*) ((uintptr_t) i0 + input_increment);
          i1 = (const int8_t*) ((uintptr_t) i1 + input_increment);
          i2 = (const int8_t*) ((uintptr_t) i2 + input_increment);
          i3 = (const int8_t*) ((uintptr_t) i3 + input_increment);
          i4 = (const int8_t*) ((uintptr_t) i4 + input_increment);
          i5 = (const int8_t*) ((uintptr_t) i5 + input_increment);
          i6 = (const int8_t*) ((uintptr_t) i6 + input_increment);
        }
        vacc0123 = _mm_add_epi32(vacc0123, _mm_cvtepi16_epi32(vacc16_01234567));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_cvtepi16_epi32(_mm_srli_si128(vacc16_01234567, 8)));
        r = doz(r, 252);
      }

      if XNN_LIKELY(channels >= 8) {
        __m128i vo0123 = _mm_loadu_si128((const __m128i*) output);
        __m128i vo4567 = _mm_loadu_si128((const __m128i*) ((uintptr_t) output + 4 * sizeof(int32_t)));
        vo0123 = _mm_add_epi32(vo0123, vacc0123);
        vo4567 = _mm_add_epi32(vo4567, vacc4567);
        _mm_storeu_si128((__m128i*) output, vo0123); output += 4;
        _mm_storeu_si128((__m128i*) output, vo4567); output += 4;
        channels -= 8;
        input = (const int8_t*) ((uintptr_t) input + 8 * sizeof(int8_t));
      } else {
        if (channels & 4) {
          __m128i vo0123 = _mm_loadu_si128((const __m128i*) output);
          vo0123 = _mm_add_epi32(vo0123, vacc0123);
          _mm_storeu_si128((__m128i*) output, vo0123); output += 4;
          vacc0123 = vacc4567;
        }
        if (channels & 2) {
          __m128i vo01 = _mm_loadl_epi64((const __m128i*) output);
          vo01 = _mm_add_epi32(vo01, vacc0123);
          _mm_storel_epi64((__m128i*) output, vo01); output += 2;
          vacc0123 = _mm_srli_si128(vacc0123, 8);
        }
        if (channels & 1) {
          __m128i vo0 = _mm_cvtsi32_si128(unaligned_load_s32(output));
          vo0 = _mm_add_epi32(vo0, vacc0123);
          _mm_storeu_si32(output, vo0);
        }
        channels = 0;
      }
    } while (channels != 0);
  }
}
