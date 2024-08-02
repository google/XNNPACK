// Auto-generated file. Do not edit!
//   Template: src/qs8-rdsum/avx512skx.c.in
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


void xnn_qs8_rdsum_ukernel_7p7x__avx512skx_c128(
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
  for (; channels >= 128; channels -= 128) {
    const int8_t* i0 = input;
    const int8_t* i1 = (const int8_t*) ((uintptr_t) input + 1 * input_stride);
    const int8_t* i2 = (const int8_t*) ((uintptr_t) input + 2 * input_stride);
    const int8_t* i3 = (const int8_t*) ((uintptr_t) input + 3 * input_stride);
    const int8_t* i4 = (const int8_t*) ((uintptr_t) input + 4 * input_stride);
    const int8_t* i5 = (const int8_t*) ((uintptr_t) input + 5 * input_stride);
    const int8_t* i6 = (const int8_t*) ((uintptr_t) input + 6 * input_stride);

    __m512i vacc0_16 = _mm512_setzero_si512();
    __m512i vacc16_32 = _mm512_setzero_si512();
    __m512i vacc32_48 = _mm512_setzero_si512();
    __m512i vacc48_64 = _mm512_setzero_si512();
    __m512i vacc64_80 = _mm512_setzero_si512();
    __m512i vacc80_96 = _mm512_setzero_si512();
    __m512i vacc96_112 = _mm512_setzero_si512();
    __m512i vacc112_128 = _mm512_setzero_si512();

    // 256 int8s may be summed into an int16 before overflowing
    // To prevent handling the tails of the inner 256 loop, we round 256 down to
    // the nearest integer multiple of ACCUMULATORS.
    int num_batches = floor((rows + 251) / 252);
    int r = rows;
    for (; num_batches > 0; --num_batches) {
      __m512i v16acc_0_32 = _mm512_setzero_si512();
      __m512i v16acc_32_64 = _mm512_setzero_si512();
      __m512i v16acc_64_96 = _mm512_setzero_si512();
      __m512i v16acc_96_128 = _mm512_setzero_si512();
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
        __m512i vin0_32;
        __m512i vin32_64;
        __m512i vin64_96;
        __m512i vin96_128;
        vin0_32 = _mm512_cvtepi8_epi16(_mm256_loadu_si256((const __m256i*) &i0[0]));
        vin32_64 = _mm512_cvtepi8_epi16(_mm256_loadu_si256((const __m256i*) &i0[32]));
        vin64_96 = _mm512_cvtepi8_epi16(_mm256_loadu_si256((const __m256i*) &i0[64]));
        vin96_128 = _mm512_cvtepi8_epi16(_mm256_loadu_si256((const __m256i*) &i0[96]));
        v16acc_0_32 = _mm512_add_epi16(v16acc_0_32, vin0_32);
        v16acc_32_64 = _mm512_add_epi16(v16acc_32_64, vin32_64);
        v16acc_64_96 = _mm512_add_epi16(v16acc_64_96, vin64_96);
        v16acc_96_128 = _mm512_add_epi16(v16acc_96_128, vin96_128);
        vin0_32 = _mm512_cvtepi8_epi16(_mm256_loadu_si256((const __m256i*) &i1[0]));
        vin32_64 = _mm512_cvtepi8_epi16(_mm256_loadu_si256((const __m256i*) &i1[32]));
        vin64_96 = _mm512_cvtepi8_epi16(_mm256_loadu_si256((const __m256i*) &i1[64]));
        vin96_128 = _mm512_cvtepi8_epi16(_mm256_loadu_si256((const __m256i*) &i1[96]));
        v16acc_0_32 = _mm512_add_epi16(v16acc_0_32, vin0_32);
        v16acc_32_64 = _mm512_add_epi16(v16acc_32_64, vin32_64);
        v16acc_64_96 = _mm512_add_epi16(v16acc_64_96, vin64_96);
        v16acc_96_128 = _mm512_add_epi16(v16acc_96_128, vin96_128);
        vin0_32 = _mm512_cvtepi8_epi16(_mm256_loadu_si256((const __m256i*) &i2[0]));
        vin32_64 = _mm512_cvtepi8_epi16(_mm256_loadu_si256((const __m256i*) &i2[32]));
        vin64_96 = _mm512_cvtepi8_epi16(_mm256_loadu_si256((const __m256i*) &i2[64]));
        vin96_128 = _mm512_cvtepi8_epi16(_mm256_loadu_si256((const __m256i*) &i2[96]));
        v16acc_0_32 = _mm512_add_epi16(v16acc_0_32, vin0_32);
        v16acc_32_64 = _mm512_add_epi16(v16acc_32_64, vin32_64);
        v16acc_64_96 = _mm512_add_epi16(v16acc_64_96, vin64_96);
        v16acc_96_128 = _mm512_add_epi16(v16acc_96_128, vin96_128);
        vin0_32 = _mm512_cvtepi8_epi16(_mm256_loadu_si256((const __m256i*) &i3[0]));
        vin32_64 = _mm512_cvtepi8_epi16(_mm256_loadu_si256((const __m256i*) &i3[32]));
        vin64_96 = _mm512_cvtepi8_epi16(_mm256_loadu_si256((const __m256i*) &i3[64]));
        vin96_128 = _mm512_cvtepi8_epi16(_mm256_loadu_si256((const __m256i*) &i3[96]));
        v16acc_0_32 = _mm512_add_epi16(v16acc_0_32, vin0_32);
        v16acc_32_64 = _mm512_add_epi16(v16acc_32_64, vin32_64);
        v16acc_64_96 = _mm512_add_epi16(v16acc_64_96, vin64_96);
        v16acc_96_128 = _mm512_add_epi16(v16acc_96_128, vin96_128);
        vin0_32 = _mm512_cvtepi8_epi16(_mm256_loadu_si256((const __m256i*) &i4[0]));
        vin32_64 = _mm512_cvtepi8_epi16(_mm256_loadu_si256((const __m256i*) &i4[32]));
        vin64_96 = _mm512_cvtepi8_epi16(_mm256_loadu_si256((const __m256i*) &i4[64]));
        vin96_128 = _mm512_cvtepi8_epi16(_mm256_loadu_si256((const __m256i*) &i4[96]));
        v16acc_0_32 = _mm512_add_epi16(v16acc_0_32, vin0_32);
        v16acc_32_64 = _mm512_add_epi16(v16acc_32_64, vin32_64);
        v16acc_64_96 = _mm512_add_epi16(v16acc_64_96, vin64_96);
        v16acc_96_128 = _mm512_add_epi16(v16acc_96_128, vin96_128);
        vin0_32 = _mm512_cvtepi8_epi16(_mm256_loadu_si256((const __m256i*) &i5[0]));
        vin32_64 = _mm512_cvtepi8_epi16(_mm256_loadu_si256((const __m256i*) &i5[32]));
        vin64_96 = _mm512_cvtepi8_epi16(_mm256_loadu_si256((const __m256i*) &i5[64]));
        vin96_128 = _mm512_cvtepi8_epi16(_mm256_loadu_si256((const __m256i*) &i5[96]));
        v16acc_0_32 = _mm512_add_epi16(v16acc_0_32, vin0_32);
        v16acc_32_64 = _mm512_add_epi16(v16acc_32_64, vin32_64);
        v16acc_64_96 = _mm512_add_epi16(v16acc_64_96, vin64_96);
        v16acc_96_128 = _mm512_add_epi16(v16acc_96_128, vin96_128);
        vin0_32 = _mm512_cvtepi8_epi16(_mm256_loadu_si256((const __m256i*) &i6[0]));
        vin32_64 = _mm512_cvtepi8_epi16(_mm256_loadu_si256((const __m256i*) &i6[32]));
        vin64_96 = _mm512_cvtepi8_epi16(_mm256_loadu_si256((const __m256i*) &i6[64]));
        vin96_128 = _mm512_cvtepi8_epi16(_mm256_loadu_si256((const __m256i*) &i6[96]));
        v16acc_0_32 = _mm512_add_epi16(v16acc_0_32, vin0_32);
        v16acc_32_64 = _mm512_add_epi16(v16acc_32_64, vin32_64);
        v16acc_64_96 = _mm512_add_epi16(v16acc_64_96, vin64_96);
        v16acc_96_128 = _mm512_add_epi16(v16acc_96_128, vin96_128);
        i0 = (const int8_t*) ((uintptr_t) i0 + input_increment);
        i1 = (const int8_t*) ((uintptr_t) i1 + input_increment);
        i2 = (const int8_t*) ((uintptr_t) i2 + input_increment);
        i3 = (const int8_t*) ((uintptr_t) i3 + input_increment);
        i4 = (const int8_t*) ((uintptr_t) i4 + input_increment);
        i5 = (const int8_t*) ((uintptr_t) i5 + input_increment);
        i6 = (const int8_t*) ((uintptr_t) i6 + input_increment);
      }
      vacc0_16 = _mm512_add_epi32(vacc0_16, _mm512_cvtepi16_epi32(_mm512_castsi512_si256(v16acc_0_32)));
      vacc16_32 = _mm512_add_epi32(vacc16_32, _mm512_cvtepi16_epi32(_mm512_extracti32x8_epi32(v16acc_0_32, 1)));
      vacc32_48 = _mm512_add_epi32(vacc32_48, _mm512_cvtepi16_epi32(_mm512_castsi512_si256(v16acc_32_64)));
      vacc48_64 = _mm512_add_epi32(vacc48_64, _mm512_cvtepi16_epi32(_mm512_extracti32x8_epi32(v16acc_32_64, 1)));
      vacc64_80 = _mm512_add_epi32(vacc64_80, _mm512_cvtepi16_epi32(_mm512_castsi512_si256(v16acc_64_96)));
      vacc80_96 = _mm512_add_epi32(vacc80_96, _mm512_cvtepi16_epi32(_mm512_extracti32x8_epi32(v16acc_64_96, 1)));
      vacc96_112 = _mm512_add_epi32(vacc96_112, _mm512_cvtepi16_epi32(_mm512_castsi512_si256(v16acc_96_128)));
      vacc112_128 = _mm512_add_epi32(vacc112_128, _mm512_cvtepi16_epi32(_mm512_extracti32x8_epi32(v16acc_96_128, 1)));
      r = doz(r, 252);
    }

    const int32_t* o = output;
    __m512i vo0_16 = _mm512_loadu_si512((const __m512i*) o); o += 16;
    __m512i vo16_32 = _mm512_loadu_si512((const __m512i*) o); o += 16;
    __m512i vo32_48 = _mm512_loadu_si512((const __m512i*) o); o += 16;
    __m512i vo48_64 = _mm512_loadu_si512((const __m512i*) o); o += 16;
    __m512i vo64_80 = _mm512_loadu_si512((const __m512i*) o); o += 16;
    __m512i vo80_96 = _mm512_loadu_si512((const __m512i*) o); o += 16;
    __m512i vo96_112 = _mm512_loadu_si512((const __m512i*) o); o += 16;
    __m512i vo112_128 = _mm512_loadu_si512((const __m512i*) o); o += 16;
    vo0_16 = _mm512_add_epi32(vacc0_16, vo0_16);
    vo16_32 = _mm512_add_epi32(vacc16_32, vo16_32);
    vo32_48 = _mm512_add_epi32(vacc32_48, vo32_48);
    vo48_64 = _mm512_add_epi32(vacc48_64, vo48_64);
    vo64_80 = _mm512_add_epi32(vacc64_80, vo64_80);
    vo80_96 = _mm512_add_epi32(vacc80_96, vo80_96);
    vo96_112 = _mm512_add_epi32(vacc96_112, vo96_112);
    vo112_128 = _mm512_add_epi32(vacc112_128, vo112_128);
    _mm512_storeu_si512((__m512i*) output, vo0_16); output += 16;
    _mm512_storeu_si512((__m512i*) output, vo16_32); output += 16;
    _mm512_storeu_si512((__m512i*) output, vo32_48); output += 16;
    _mm512_storeu_si512((__m512i*) output, vo48_64); output += 16;
    _mm512_storeu_si512((__m512i*) output, vo64_80); output += 16;
    _mm512_storeu_si512((__m512i*) output, vo80_96); output += 16;
    _mm512_storeu_si512((__m512i*) output, vo96_112); output += 16;
    _mm512_storeu_si512((__m512i*) output, vo112_128); output += 16;

    input = (const int8_t*) ((uintptr_t) input + 128 * sizeof(int8_t));
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

      __m512i vacc0_16 = _mm512_setzero_si512();
      __m512i v16acc_32 = _mm512_setzero_si512();

      const size_t shift = channels < 32 ? channels : 32;
      const __mmask32 vmask = _cvtu32_mask32((uint32_t) ((UINT64_C(1) << shift) - UINT64_C(1)));
      for (; num_batches > 0; --num_batches) {
        __m512i v16acc_0_32 = _mm512_setzero_si512();
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

          __m512i vin0 = _mm512_cvtepi8_epi16(_mm256_maskz_loadu_epi8(vmask, (const __m256i*)&i0[0]));
          __m512i vin1 = _mm512_cvtepi8_epi16(_mm256_maskz_loadu_epi8(vmask, (const __m256i*)&i1[0]));
          __m512i vin2 = _mm512_cvtepi8_epi16(_mm256_maskz_loadu_epi8(vmask, (const __m256i*)&i2[0]));
          __m512i vin3 = _mm512_cvtepi8_epi16(_mm256_maskz_loadu_epi8(vmask, (const __m256i*)&i3[0]));
          __m512i vin4 = _mm512_cvtepi8_epi16(_mm256_maskz_loadu_epi8(vmask, (const __m256i*)&i4[0]));
          __m512i vin5 = _mm512_cvtepi8_epi16(_mm256_maskz_loadu_epi8(vmask, (const __m256i*)&i5[0]));
          __m512i vin6 = _mm512_cvtepi8_epi16(_mm256_maskz_loadu_epi8(vmask, (const __m256i*)&i6[0]));
          v16acc_0_32 = _mm512_add_epi16(v16acc_0_32, vin0);
          v16acc_0_32 = _mm512_add_epi16(v16acc_0_32, vin1);
          v16acc_0_32 = _mm512_add_epi16(v16acc_0_32, vin2);
          v16acc_0_32 = _mm512_add_epi16(v16acc_0_32, vin3);
          v16acc_0_32 = _mm512_add_epi16(v16acc_0_32, vin4);
          v16acc_0_32 = _mm512_add_epi16(v16acc_0_32, vin5);
          v16acc_0_32 = _mm512_add_epi16(v16acc_0_32, vin6);
          i0 = (const int8_t*) ((uintptr_t) i0 + input_increment);
          i1 = (const int8_t*) ((uintptr_t) i1 + input_increment);
          i2 = (const int8_t*) ((uintptr_t) i2 + input_increment);
          i3 = (const int8_t*) ((uintptr_t) i3 + input_increment);
          i4 = (const int8_t*) ((uintptr_t) i4 + input_increment);
          i5 = (const int8_t*) ((uintptr_t) i5 + input_increment);
          i6 = (const int8_t*) ((uintptr_t) i6 + input_increment);
        }
        vacc0_16 = _mm512_add_epi32(vacc0_16, _mm512_cvtepi16_epi32(_mm512_castsi512_si256(v16acc_0_32)));
        v16acc_32 = _mm512_add_epi32(v16acc_32, _mm512_cvtepi16_epi32(_mm512_extracti32x8_epi32(v16acc_0_32, 1)));
        r = doz(r, 252);
      }

      if XNN_LIKELY(channels >= 32) {
        __m512i vo0_16 = _mm512_loadu_epi32(output);
        __m512i vo16_32 = _mm512_loadu_epi32(output + 16);
        vo0_16 = _mm512_add_epi32(vo0_16, vacc0_16);
        vo16_32 = _mm512_add_epi32(vo16_32, v16acc_32);
        _mm512_storeu_si512((__m512i*) output, vo0_16); output += 16;
        _mm512_storeu_si512((__m512i*) output, vo16_32); output += 16;
        channels -= 32;
        input = (const int8_t*) ((uintptr_t) input + 32 * sizeof(int8_t));
      } else {
        if (channels & 16) {
          __m512i vo0_16 = _mm512_loadu_epi32(output);
          vo0_16 = _mm512_add_epi32(vo0_16, vacc0_16);
          _mm512_storeu_si512((__m512i*) output, vo0_16); output += 16;
          vacc0_16 = v16acc_32;
        }
        __m256i vacc0_8 = _mm512_castsi512_si256(vacc0_16);
        if (channels & 8) {
          __m256i vo0_8 = _mm256_loadu_si256((const __m256i*) output);
          vo0_8 = _mm256_add_epi32(vo0_8, vacc0_8);
          _mm256_storeu_si256((__m256i*) output, vo0_8); output += 8;
          vacc0_8 = _mm512_extracti32x8_epi32(vacc0_16, 1);
        }
        if (channels & 4) {
          __m128i vo0_4 = _mm_loadu_si128((const __m128i*) output);
          vo0_4 = _mm_add_epi32(vo0_4, _mm256_castsi256_si128(vacc0_8));
          _mm_storeu_si128((__m128i*) output, vo0_4); output += 4;
          vacc0_8 = _mm256_castsi128_si256(_mm256_extractf128_si256(vacc0_8, 1));
        }
        if (channels & 2) {
          __m128i vo0_2 = _mm_loadl_epi64((const __m128i*) output);
          vo0_2 = _mm_add_epi32(vo0_2, _mm256_castsi256_si128(vacc0_8));
          _mm_storel_epi64((__m128i*) output, vo0_2); output += 2;
          vacc0_8 = _mm256_srli_si256(vacc0_8, 8);
        }
        if (channels & 1) {
          *output += _mm_cvtsi128_si32(_mm256_castsi256_si128(vacc0_8));
        }
        channels = 0;
      }
    } while (channels != 0);
  }
}
