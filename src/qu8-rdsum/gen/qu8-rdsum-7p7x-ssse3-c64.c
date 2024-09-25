// Auto-generated file. Do not edit!
//   Template: src/qu8-rdsum/ssse3.c.in
//   Generator: tools/xngen
//
// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
#include <assert.h>
#include <math.h>

#include <tmmintrin.h>

#include "xnnpack/common.h"
#include "xnnpack/intrinsics-polyfill.h"
#include "xnnpack/math.h"
#include "xnnpack/reduce.h"
#include "xnnpack/unaligned.h"


void xnn_qu8_rdsum_ukernel_7p7x__ssse3_c64(
    size_t rows,
    size_t channels,
    const uint8_t* input,
    size_t input_stride,
    const uint8_t* zero,
    uint32_t* output,
    const struct xnn_qs8_rsum_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(rows != 0);
  assert(channels != 0);
  assert(input != NULL);
  assert(output != NULL);

  size_t input_increment = 7 * input_stride;
  for (; channels >= 64; channels -= 64) {
    const uint8_t* i0 = input;
    const uint8_t* i1 = (const uint8_t*) ((uintptr_t) input + 1 * input_stride);
    const uint8_t* i2 = (const uint8_t*) ((uintptr_t) input + 2 * input_stride);
    const uint8_t* i3 = (const uint8_t*) ((uintptr_t) input + 3 * input_stride);
    const uint8_t* i4 = (const uint8_t*) ((uintptr_t) input + 4 * input_stride);
    const uint8_t* i5 = (const uint8_t*) ((uintptr_t) input + 5 * input_stride);
    const uint8_t* i6 = (const uint8_t*) ((uintptr_t) input + 6 * input_stride);

    __m128i vacc0 = _mm_setzero_si128();
    __m128i vacc4 = _mm_setzero_si128();
    __m128i vacc8 = _mm_setzero_si128();
    __m128i vacc12 = _mm_setzero_si128();
    __m128i vacc16 = _mm_setzero_si128();
    __m128i vacc20 = _mm_setzero_si128();
    __m128i vacc24 = _mm_setzero_si128();
    __m128i vacc28 = _mm_setzero_si128();
    __m128i vacc32 = _mm_setzero_si128();
    __m128i vacc36 = _mm_setzero_si128();
    __m128i vacc40 = _mm_setzero_si128();
    __m128i vacc44 = _mm_setzero_si128();
    __m128i vacc48 = _mm_setzero_si128();
    __m128i vacc52 = _mm_setzero_si128();
    __m128i vacc56 = _mm_setzero_si128();
    __m128i vacc60 = _mm_setzero_si128();

    // 256 uint8s may be summed into an uint16 before overflowing
    // To prevent handling the tails of the inner 256 loop, we round 256 down to
    // the nearest integer multiple of ACCUMULATORS.
    int r = rows;
    __m128i vone = _mm_set1_epi8(1);

    while (r > 0) {
      __m128i vacc16_0 = _mm_setzero_si128();
      __m128i vacc16_8 = _mm_setzero_si128();
      __m128i vacc16_16 = _mm_setzero_si128();
      __m128i vacc16_24 = _mm_setzero_si128();
      __m128i vacc16_32 = _mm_setzero_si128();
      __m128i vacc16_40 = _mm_setzero_si128();
      __m128i vacc16_48 = _mm_setzero_si128();
      __m128i vacc16_56 = _mm_setzero_si128();
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

        __m128i vin_lo;
        __m128i vin_hi;
        __m128i vin0;
        __m128i vin1;
        __m128i vin2;
        __m128i vin3;
        __m128i vin4;
        __m128i vin5;
        __m128i vin6;

        vin0 = _mm_loadu_si128((const __m128i*)&i0[0]);
        vin1 = _mm_loadu_si128((const __m128i*)&i1[0]);
        vin2 = _mm_loadu_si128((const __m128i*)&i2[0]);
        vin3 = _mm_loadu_si128((const __m128i*)&i3[0]);
        vin4 = _mm_loadu_si128((const __m128i*)&i4[0]);
        vin5 = _mm_loadu_si128((const __m128i*)&i5[0]);
        vin6 = _mm_loadu_si128((const __m128i*)&i6[0]);
        vin_lo = _mm_unpacklo_epi8(vin0, vin1);
        vin_hi = _mm_unpackhi_epi8(vin0, vin1);
        vin_lo = _mm_maddubs_epi16(vin_lo, vone);
        vin_hi = _mm_maddubs_epi16(vin_hi, vone);
        vacc16_0 = _mm_add_epi16(vacc16_0, vin_lo);
        vacc16_8 = _mm_add_epi16(vacc16_8, vin_hi);
        vin_lo = _mm_unpacklo_epi8(vin2, vin3);
        vin_hi = _mm_unpackhi_epi8(vin2, vin3);
        vin_lo = _mm_maddubs_epi16(vin_lo, vone);
        vin_hi = _mm_maddubs_epi16(vin_hi, vone);
        vacc16_0 = _mm_add_epi16(vacc16_0, vin_lo);
        vacc16_8 = _mm_add_epi16(vacc16_8, vin_hi);
        vin_lo = _mm_unpacklo_epi8(vin4, vin5);
        vin_hi = _mm_unpackhi_epi8(vin4, vin5);
        vin_lo = _mm_maddubs_epi16(vin_lo, vone);
        vin_hi = _mm_maddubs_epi16(vin_hi, vone);
        vacc16_0 = _mm_add_epi16(vacc16_0, vin_lo);
        vacc16_8 = _mm_add_epi16(vacc16_8, vin_hi);

        vin_lo = _mm_unpacklo_epi8(vin6, _mm_setzero_si128());
        vin_hi = _mm_unpackhi_epi8(vin6, _mm_setzero_si128());
        vacc16_0 = _mm_add_epi16(vacc16_0, vin_lo);
        vacc16_8 = _mm_add_epi16(vacc16_8, vin_hi);
        vin0 = _mm_loadu_si128((const __m128i*)&i0[16]);
        vin1 = _mm_loadu_si128((const __m128i*)&i1[16]);
        vin2 = _mm_loadu_si128((const __m128i*)&i2[16]);
        vin3 = _mm_loadu_si128((const __m128i*)&i3[16]);
        vin4 = _mm_loadu_si128((const __m128i*)&i4[16]);
        vin5 = _mm_loadu_si128((const __m128i*)&i5[16]);
        vin6 = _mm_loadu_si128((const __m128i*)&i6[16]);
        vin_lo = _mm_unpacklo_epi8(vin0, vin1);
        vin_hi = _mm_unpackhi_epi8(vin0, vin1);
        vin_lo = _mm_maddubs_epi16(vin_lo, vone);
        vin_hi = _mm_maddubs_epi16(vin_hi, vone);
        vacc16_16 = _mm_add_epi16(vacc16_16, vin_lo);
        vacc16_24 = _mm_add_epi16(vacc16_24, vin_hi);
        vin_lo = _mm_unpacklo_epi8(vin2, vin3);
        vin_hi = _mm_unpackhi_epi8(vin2, vin3);
        vin_lo = _mm_maddubs_epi16(vin_lo, vone);
        vin_hi = _mm_maddubs_epi16(vin_hi, vone);
        vacc16_16 = _mm_add_epi16(vacc16_16, vin_lo);
        vacc16_24 = _mm_add_epi16(vacc16_24, vin_hi);
        vin_lo = _mm_unpacklo_epi8(vin4, vin5);
        vin_hi = _mm_unpackhi_epi8(vin4, vin5);
        vin_lo = _mm_maddubs_epi16(vin_lo, vone);
        vin_hi = _mm_maddubs_epi16(vin_hi, vone);
        vacc16_16 = _mm_add_epi16(vacc16_16, vin_lo);
        vacc16_24 = _mm_add_epi16(vacc16_24, vin_hi);

        vin_lo = _mm_unpacklo_epi8(vin6, _mm_setzero_si128());
        vin_hi = _mm_unpackhi_epi8(vin6, _mm_setzero_si128());
        vacc16_16 = _mm_add_epi16(vacc16_16, vin_lo);
        vacc16_24 = _mm_add_epi16(vacc16_24, vin_hi);
        vin0 = _mm_loadu_si128((const __m128i*)&i0[32]);
        vin1 = _mm_loadu_si128((const __m128i*)&i1[32]);
        vin2 = _mm_loadu_si128((const __m128i*)&i2[32]);
        vin3 = _mm_loadu_si128((const __m128i*)&i3[32]);
        vin4 = _mm_loadu_si128((const __m128i*)&i4[32]);
        vin5 = _mm_loadu_si128((const __m128i*)&i5[32]);
        vin6 = _mm_loadu_si128((const __m128i*)&i6[32]);
        vin_lo = _mm_unpacklo_epi8(vin0, vin1);
        vin_hi = _mm_unpackhi_epi8(vin0, vin1);
        vin_lo = _mm_maddubs_epi16(vin_lo, vone);
        vin_hi = _mm_maddubs_epi16(vin_hi, vone);
        vacc16_32 = _mm_add_epi16(vacc16_32, vin_lo);
        vacc16_40 = _mm_add_epi16(vacc16_40, vin_hi);
        vin_lo = _mm_unpacklo_epi8(vin2, vin3);
        vin_hi = _mm_unpackhi_epi8(vin2, vin3);
        vin_lo = _mm_maddubs_epi16(vin_lo, vone);
        vin_hi = _mm_maddubs_epi16(vin_hi, vone);
        vacc16_32 = _mm_add_epi16(vacc16_32, vin_lo);
        vacc16_40 = _mm_add_epi16(vacc16_40, vin_hi);
        vin_lo = _mm_unpacklo_epi8(vin4, vin5);
        vin_hi = _mm_unpackhi_epi8(vin4, vin5);
        vin_lo = _mm_maddubs_epi16(vin_lo, vone);
        vin_hi = _mm_maddubs_epi16(vin_hi, vone);
        vacc16_32 = _mm_add_epi16(vacc16_32, vin_lo);
        vacc16_40 = _mm_add_epi16(vacc16_40, vin_hi);

        vin_lo = _mm_unpacklo_epi8(vin6, _mm_setzero_si128());
        vin_hi = _mm_unpackhi_epi8(vin6, _mm_setzero_si128());
        vacc16_32 = _mm_add_epi16(vacc16_32, vin_lo);
        vacc16_40 = _mm_add_epi16(vacc16_40, vin_hi);
        vin0 = _mm_loadu_si128((const __m128i*)&i0[48]);
        vin1 = _mm_loadu_si128((const __m128i*)&i1[48]);
        vin2 = _mm_loadu_si128((const __m128i*)&i2[48]);
        vin3 = _mm_loadu_si128((const __m128i*)&i3[48]);
        vin4 = _mm_loadu_si128((const __m128i*)&i4[48]);
        vin5 = _mm_loadu_si128((const __m128i*)&i5[48]);
        vin6 = _mm_loadu_si128((const __m128i*)&i6[48]);
        vin_lo = _mm_unpacklo_epi8(vin0, vin1);
        vin_hi = _mm_unpackhi_epi8(vin0, vin1);
        vin_lo = _mm_maddubs_epi16(vin_lo, vone);
        vin_hi = _mm_maddubs_epi16(vin_hi, vone);
        vacc16_48 = _mm_add_epi16(vacc16_48, vin_lo);
        vacc16_56 = _mm_add_epi16(vacc16_56, vin_hi);
        vin_lo = _mm_unpacklo_epi8(vin2, vin3);
        vin_hi = _mm_unpackhi_epi8(vin2, vin3);
        vin_lo = _mm_maddubs_epi16(vin_lo, vone);
        vin_hi = _mm_maddubs_epi16(vin_hi, vone);
        vacc16_48 = _mm_add_epi16(vacc16_48, vin_lo);
        vacc16_56 = _mm_add_epi16(vacc16_56, vin_hi);
        vin_lo = _mm_unpacklo_epi8(vin4, vin5);
        vin_hi = _mm_unpackhi_epi8(vin4, vin5);
        vin_lo = _mm_maddubs_epi16(vin_lo, vone);
        vin_hi = _mm_maddubs_epi16(vin_hi, vone);
        vacc16_48 = _mm_add_epi16(vacc16_48, vin_lo);
        vacc16_56 = _mm_add_epi16(vacc16_56, vin_hi);

        vin_lo = _mm_unpacklo_epi8(vin6, _mm_setzero_si128());
        vin_hi = _mm_unpackhi_epi8(vin6, _mm_setzero_si128());
        vacc16_48 = _mm_add_epi16(vacc16_48, vin_lo);
        vacc16_56 = _mm_add_epi16(vacc16_56, vin_hi);

        i0 = (const uint8_t*) ((uintptr_t) i0 + input_increment);
        i1 = (const uint8_t*) ((uintptr_t) i1 + input_increment);
        i2 = (const uint8_t*) ((uintptr_t) i2 + input_increment);
        i3 = (const uint8_t*) ((uintptr_t) i3 + input_increment);
        i4 = (const uint8_t*) ((uintptr_t) i4 + input_increment);
        i5 = (const uint8_t*) ((uintptr_t) i5 + input_increment);
        i6 = (const uint8_t*) ((uintptr_t) i6 + input_increment);
      }
      vacc0 = _mm_add_epi32(vacc0, _mm_unpacklo_epi16(vacc16_0, _mm_setzero_si128()));
      vacc4 = _mm_add_epi32(vacc4, _mm_unpacklo_epi16(_mm_srli_si128(vacc16_0, 8), _mm_setzero_si128()));
      vacc8 = _mm_add_epi32(vacc8, _mm_unpacklo_epi16(vacc16_8, _mm_setzero_si128()));
      vacc12 = _mm_add_epi32(vacc12, _mm_unpacklo_epi16(_mm_srli_si128(vacc16_8, 8), _mm_setzero_si128()));
      vacc16 = _mm_add_epi32(vacc16, _mm_unpacklo_epi16(vacc16_16, _mm_setzero_si128()));
      vacc20 = _mm_add_epi32(vacc20, _mm_unpacklo_epi16(_mm_srli_si128(vacc16_16, 8), _mm_setzero_si128()));
      vacc24 = _mm_add_epi32(vacc24, _mm_unpacklo_epi16(vacc16_24, _mm_setzero_si128()));
      vacc28 = _mm_add_epi32(vacc28, _mm_unpacklo_epi16(_mm_srli_si128(vacc16_24, 8), _mm_setzero_si128()));
      vacc32 = _mm_add_epi32(vacc32, _mm_unpacklo_epi16(vacc16_32, _mm_setzero_si128()));
      vacc36 = _mm_add_epi32(vacc36, _mm_unpacklo_epi16(_mm_srli_si128(vacc16_32, 8), _mm_setzero_si128()));
      vacc40 = _mm_add_epi32(vacc40, _mm_unpacklo_epi16(vacc16_40, _mm_setzero_si128()));
      vacc44 = _mm_add_epi32(vacc44, _mm_unpacklo_epi16(_mm_srli_si128(vacc16_40, 8), _mm_setzero_si128()));
      vacc48 = _mm_add_epi32(vacc48, _mm_unpacklo_epi16(vacc16_48, _mm_setzero_si128()));
      vacc52 = _mm_add_epi32(vacc52, _mm_unpacklo_epi16(_mm_srli_si128(vacc16_48, 8), _mm_setzero_si128()));
      vacc56 = _mm_add_epi32(vacc56, _mm_unpacklo_epi16(vacc16_56, _mm_setzero_si128()));
      vacc60 = _mm_add_epi32(vacc60, _mm_unpacklo_epi16(_mm_srli_si128(vacc16_56, 8), _mm_setzero_si128()));
      r = doz(r, 252);
    }

    __m128i vo0 = _mm_loadu_si128((const __m128i*) ((uintptr_t) output + 0 * sizeof(uint32_t)));
    __m128i vo4 = _mm_loadu_si128((const __m128i*) ((uintptr_t) output + 4 * sizeof(uint32_t)));
    __m128i vo8 = _mm_loadu_si128((const __m128i*) ((uintptr_t) output + 8 * sizeof(uint32_t)));
    __m128i vo12 = _mm_loadu_si128((const __m128i*) ((uintptr_t) output + 12 * sizeof(uint32_t)));
    __m128i vo16 = _mm_loadu_si128((const __m128i*) ((uintptr_t) output + 16 * sizeof(uint32_t)));
    __m128i vo20 = _mm_loadu_si128((const __m128i*) ((uintptr_t) output + 20 * sizeof(uint32_t)));
    __m128i vo24 = _mm_loadu_si128((const __m128i*) ((uintptr_t) output + 24 * sizeof(uint32_t)));
    __m128i vo28 = _mm_loadu_si128((const __m128i*) ((uintptr_t) output + 28 * sizeof(uint32_t)));
    __m128i vo32 = _mm_loadu_si128((const __m128i*) ((uintptr_t) output + 32 * sizeof(uint32_t)));
    __m128i vo36 = _mm_loadu_si128((const __m128i*) ((uintptr_t) output + 36 * sizeof(uint32_t)));
    __m128i vo40 = _mm_loadu_si128((const __m128i*) ((uintptr_t) output + 40 * sizeof(uint32_t)));
    __m128i vo44 = _mm_loadu_si128((const __m128i*) ((uintptr_t) output + 44 * sizeof(uint32_t)));
    __m128i vo48 = _mm_loadu_si128((const __m128i*) ((uintptr_t) output + 48 * sizeof(uint32_t)));
    __m128i vo52 = _mm_loadu_si128((const __m128i*) ((uintptr_t) output + 52 * sizeof(uint32_t)));
    __m128i vo56 = _mm_loadu_si128((const __m128i*) ((uintptr_t) output + 56 * sizeof(uint32_t)));
    __m128i vo60 = _mm_loadu_si128((const __m128i*) ((uintptr_t) output + 60 * sizeof(uint32_t)));
    vo0 = _mm_add_epi32(vo0, vacc0);
    vo4 = _mm_add_epi32(vo4, vacc4);
    vo8 = _mm_add_epi32(vo8, vacc8);
    vo12 = _mm_add_epi32(vo12, vacc12);
    vo16 = _mm_add_epi32(vo16, vacc16);
    vo20 = _mm_add_epi32(vo20, vacc20);
    vo24 = _mm_add_epi32(vo24, vacc24);
    vo28 = _mm_add_epi32(vo28, vacc28);
    vo32 = _mm_add_epi32(vo32, vacc32);
    vo36 = _mm_add_epi32(vo36, vacc36);
    vo40 = _mm_add_epi32(vo40, vacc40);
    vo44 = _mm_add_epi32(vo44, vacc44);
    vo48 = _mm_add_epi32(vo48, vacc48);
    vo52 = _mm_add_epi32(vo52, vacc52);
    vo56 = _mm_add_epi32(vo56, vacc56);
    vo60 = _mm_add_epi32(vo60, vacc60);
    _mm_storeu_si128((__m128i*) output, vo0); output += 4;
    _mm_storeu_si128((__m128i*) output, vo4); output += 4;
    _mm_storeu_si128((__m128i*) output, vo8); output += 4;
    _mm_storeu_si128((__m128i*) output, vo12); output += 4;
    _mm_storeu_si128((__m128i*) output, vo16); output += 4;
    _mm_storeu_si128((__m128i*) output, vo20); output += 4;
    _mm_storeu_si128((__m128i*) output, vo24); output += 4;
    _mm_storeu_si128((__m128i*) output, vo28); output += 4;
    _mm_storeu_si128((__m128i*) output, vo32); output += 4;
    _mm_storeu_si128((__m128i*) output, vo36); output += 4;
    _mm_storeu_si128((__m128i*) output, vo40); output += 4;
    _mm_storeu_si128((__m128i*) output, vo44); output += 4;
    _mm_storeu_si128((__m128i*) output, vo48); output += 4;
    _mm_storeu_si128((__m128i*) output, vo52); output += 4;
    _mm_storeu_si128((__m128i*) output, vo56); output += 4;
    _mm_storeu_si128((__m128i*) output, vo60); output += 4;

    input = (const uint8_t*) ((uintptr_t) input + 64 * sizeof(uint8_t));
  }
  if (channels != 0) {
    input_increment = 7 * input_stride;
    // 256 uint8s may be summed into an uint16 before overflowing.
    do {
      int num_batches = floor((rows + 251) / 252);
      int r = rows;
      const uint8_t* i0 = input;
      const uint8_t* i1 = (const uint8_t*) ((uintptr_t) input + 1 * input_stride);
      const uint8_t* i2 = (const uint8_t*) ((uintptr_t) input + 2 * input_stride);
      const uint8_t* i3 = (const uint8_t*) ((uintptr_t) input + 3 * input_stride);
      const uint8_t* i4 = (const uint8_t*) ((uintptr_t) input + 4 * input_stride);
      const uint8_t* i5 = (const uint8_t*) ((uintptr_t) input + 5 * input_stride);
      const uint8_t* i6 = (const uint8_t*) ((uintptr_t) input + 6 * input_stride);

      __m128i vacc0123 = _mm_setzero_si128();
      __m128i vacc4567 = _mm_setzero_si128();
      __m128i vone = _mm_set1_epi8(1);

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

          __m128i vin_lo;
          __m128i vin_hi;
          __m128i vin0 = _mm_loadl_epi64((const __m128i*)&i0[0]);
          __m128i vin1 = _mm_loadl_epi64((const __m128i*)&i1[0]);
          __m128i vin2 = _mm_loadl_epi64((const __m128i*)&i2[0]);
          __m128i vin3 = _mm_loadl_epi64((const __m128i*)&i3[0]);
          __m128i vin4 = _mm_loadl_epi64((const __m128i*)&i4[0]);
          __m128i vin5 = _mm_loadl_epi64((const __m128i*)&i5[0]);
          __m128i vin6 = _mm_loadl_epi64((const __m128i*)&i6[0]);
          vin_lo = _mm_unpacklo_epi8(vin0, vin1);
          vin_hi = _mm_unpackhi_epi8(vin0, vin1);
          vin_lo = _mm_maddubs_epi16(vin_lo, vone);
          vin_hi = _mm_maddubs_epi16(vin_hi, vone);
          vacc16_01234567 = _mm_add_epi16(vacc16_01234567, vin_lo);
          vacc16_01234567 = _mm_add_epi16(vacc16_01234567, vin_hi);
          vin_lo = _mm_unpacklo_epi8(vin2, vin3);
          vin_hi = _mm_unpackhi_epi8(vin2, vin3);
          vin_lo = _mm_maddubs_epi16(vin_lo, vone);
          vin_hi = _mm_maddubs_epi16(vin_hi, vone);
          vacc16_01234567 = _mm_add_epi16(vacc16_01234567, vin_lo);
          vacc16_01234567 = _mm_add_epi16(vacc16_01234567, vin_hi);
          vin_lo = _mm_unpacklo_epi8(vin4, vin5);
          vin_hi = _mm_unpackhi_epi8(vin4, vin5);
          vin_lo = _mm_maddubs_epi16(vin_lo, vone);
          vin_hi = _mm_maddubs_epi16(vin_hi, vone);
          vacc16_01234567 = _mm_add_epi16(vacc16_01234567, vin_lo);
          vacc16_01234567 = _mm_add_epi16(vacc16_01234567, vin_hi);

          vin_lo = _mm_unpacklo_epi8(vin6, _mm_setzero_si128());
          vin_hi = _mm_unpackhi_epi8(vin6, _mm_setzero_si128());
          vacc16_01234567 = _mm_add_epi16(vacc16_01234567, vin_lo);
          vacc16_01234567 = _mm_add_epi16(vacc16_01234567, vin_hi);

          i0 = (const uint8_t*) ((uintptr_t) i0 + input_increment);
          i1 = (const uint8_t*) ((uintptr_t) i1 + input_increment);
          i2 = (const uint8_t*) ((uintptr_t) i2 + input_increment);
          i3 = (const uint8_t*) ((uintptr_t) i3 + input_increment);
          i4 = (const uint8_t*) ((uintptr_t) i4 + input_increment);
          i5 = (const uint8_t*) ((uintptr_t) i5 + input_increment);
          i6 = (const uint8_t*) ((uintptr_t) i6 + input_increment);
        }
        vacc0123 = _mm_add_epi32(vacc0123, _mm_unpacklo_epi16(vacc16_01234567, _mm_setzero_si128()));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_unpacklo_epi16(_mm_srli_si128(vacc16_01234567, 8), _mm_setzero_si128()));
        r = doz(r, 252);
      }

      if XNN_LIKELY(channels >= 8) {
        __m128i vo0123 = _mm_loadu_si128((const __m128i*) output);
        __m128i vo4567 = _mm_loadu_si128((const __m128i*) ((uintptr_t) output + 4 * sizeof(uint32_t)));
        vo0123 = _mm_add_epi32(vo0123, vacc0123);
        vo4567 = _mm_add_epi32(vo4567, vacc4567);
        _mm_storeu_si128((__m128i*) output, vo0123); output += 4;
        _mm_storeu_si128((__m128i*) output, vo4567); output += 4;
        channels -= 8;
        input = (const uint8_t*) ((uintptr_t) input + 8 * sizeof(uint8_t));
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
          __m128i vo0 = _mm_cvtsi32_si128(unaligned_load_u32(output));
          vo0 = _mm_add_epi32(vo0, vacc0123);
          _mm_storeu_si32(output, vo0);
        }
        channels = 0;
      }
    } while (channels != 0);
  }
}
