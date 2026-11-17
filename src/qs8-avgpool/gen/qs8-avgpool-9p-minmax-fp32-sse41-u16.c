// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/qs8-avgpool/sse41.c.in
//   Generator: tools/xngen
//
// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <smmintrin.h>
#include <stddef.h>
#include <stdint.h>

#include "src/xnnpack/avgpool.h"
#include "src/xnnpack/common.h"
#include "src/xnnpack/intrinsics-polyfill.h"
#include "src/xnnpack/math.h"
#include "src/xnnpack/microparams.h"
#include "src/xnnpack/unaligned.h"


void xnn_qs8_avgpool_minmax_fp32_ukernel_9p__sse41_u16(
    size_t output_pixels,
    size_t kernel_elements,
    size_t channels,
    const int8_t** input,
    size_t input_offset,
    size_t input_pixel_stride,
    const int8_t* zero,
    const float* multiplier,
    int8_t* output,
    size_t input_increment,
    size_t output_increment,
    const struct xnn_qs8_avgpool_minmax_params* restrict params) XNN_OOB_READS
{
  assert(output_pixels != 0);
  assert(kernel_elements != 0);
  assert(channels != 0);

  const __m128i vinit_bias = _mm_set1_epi32(params->scalar.init_bias);
  __m128 vscale = _mm_set1_ps(params->scalar.scale);
  const __m128 voutput_max_less_zero_point = _mm_set1_ps(
      (float) ((int32_t) params->scalar.output_max - (int32_t) params->scalar.output_zero_point));
  const __m128i voutput_zero_point = _mm_set1_epi16(params->scalar.output_zero_point);
  const __m128i voutput_min = _mm_set1_epi8((char) params->scalar.output_min);

  XNN_FORCE_REALIZATION(vinit_bias);
  XNN_FORCE_REALIZATION(voutput_max_less_zero_point);
  XNN_FORCE_REALIZATION(voutput_zero_point);
  XNN_FORCE_REALIZATION(voutput_min);

  do {
    if (multiplier != NULL) {
      vscale = _mm_set1_ps(*multiplier++);
    }

    int8_t* o = output;
    size_t c_offset = 0;
    size_t c = channels;
    for (; c >= 16; c -= 16) {
      const int8_t** i = input;
      size_t k = kernel_elements;
      __m128i vacc0123 = _mm_setzero_si128();
      __m128i vacc4567 = _mm_setzero_si128();
      __m128i vacc89AB = _mm_setzero_si128();
      __m128i vaccCDEF = _mm_setzero_si128();
      do {
        vacc0123 = _mm_add_epi32(vacc0123, vinit_bias);
        vacc4567 = _mm_add_epi32(vacc4567, vinit_bias);
        vacc89AB = _mm_add_epi32(vacc89AB, vinit_bias);
        vaccCDEF = _mm_add_epi32(vaccCDEF, vinit_bias);

        const int8_t* i0 = 0 < k ? *i++ : zero;
        assert(i0 != NULL);
        if XNN_UNPREDICTABLE(i0 != zero) {
          i0 = (const int8_t*) ((uintptr_t) i0 + input_offset);
        }
        i0 += c_offset;
        const __m128i vi0_lo = _mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*) i0));
        const __m128i vi0_hi = _mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*) (i0 + 8)));
        const int8_t* i1 = 1 < k ? *i++ : zero;
        assert(i1 != NULL);
        if XNN_UNPREDICTABLE(i1 != zero) {
          i1 = (const int8_t*) ((uintptr_t) i1 + input_offset);
        }
        i1 += c_offset;
        const __m128i vi1_lo = _mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*) i1));
        const __m128i vi1_hi = _mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*) (i1 + 8)));
        const int8_t* i2 = 2 < k ? *i++ : zero;
        assert(i2 != NULL);
        if XNN_UNPREDICTABLE(i2 != zero) {
          i2 = (const int8_t*) ((uintptr_t) i2 + input_offset);
        }
        i2 += c_offset;
        const __m128i vi2_lo = _mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*) i2));
        const __m128i vi2_hi = _mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*) (i2 + 8)));
        const int8_t* i3 = 3 < k ? *i++ : zero;
        assert(i3 != NULL);
        if XNN_UNPREDICTABLE(i3 != zero) {
          i3 = (const int8_t*) ((uintptr_t) i3 + input_offset);
        }
        i3 += c_offset;
        const __m128i vi3_lo = _mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*) i3));
        const __m128i vi3_hi = _mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*) (i3 + 8)));
        const int8_t* i4 = 4 < k ? *i++ : zero;
        assert(i4 != NULL);
        if XNN_UNPREDICTABLE(i4 != zero) {
          i4 = (const int8_t*) ((uintptr_t) i4 + input_offset);
        }
        i4 += c_offset;
        const __m128i vi4_lo = _mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*) i4));
        const __m128i vi4_hi = _mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*) (i4 + 8)));
        const int8_t* i5 = 5 < k ? *i++ : zero;
        assert(i5 != NULL);
        if XNN_UNPREDICTABLE(i5 != zero) {
          i5 = (const int8_t*) ((uintptr_t) i5 + input_offset);
        }
        i5 += c_offset;
        const __m128i vi5_lo = _mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*) i5));
        const __m128i vi5_hi = _mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*) (i5 + 8)));
        const int8_t* i6 = 6 < k ? *i++ : zero;
        assert(i6 != NULL);
        if XNN_UNPREDICTABLE(i6 != zero) {
          i6 = (const int8_t*) ((uintptr_t) i6 + input_offset);
        }
        i6 += c_offset;
        const __m128i vi6_lo = _mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*) i6));
        const __m128i vi6_hi = _mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*) (i6 + 8)));
        const int8_t* i7 = 7 < k ? *i++ : zero;
        assert(i7 != NULL);
        if XNN_UNPREDICTABLE(i7 != zero) {
          i7 = (const int8_t*) ((uintptr_t) i7 + input_offset);
        }
        i7 += c_offset;
        const __m128i vi7_lo = _mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*) i7));
        const __m128i vi7_hi = _mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*) (i7 + 8)));
        const int8_t* i8 = 8 < k ? *i++ : zero;
        assert(i8 != NULL);
        if XNN_UNPREDICTABLE(i8 != zero) {
          i8 = (const int8_t*) ((uintptr_t) i8 + input_offset);
        }
        i8 += c_offset;
        const __m128i vi8_lo = _mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*) i8));
        const __m128i vi8_hi = _mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*) (i8 + 8)));

        const __m128i vsum018_lo = _mm_add_epi16(_mm_add_epi16(vi0_lo, vi1_lo), vi8_lo);
        const __m128i vsum23_lo = _mm_add_epi16(vi2_lo, vi3_lo);
        const __m128i vsum45_lo = _mm_add_epi16(vi4_lo, vi5_lo);
        const __m128i vsum67_lo = _mm_add_epi16(vi6_lo, vi7_lo);
        const __m128i vsum_lo = _mm_add_epi16(_mm_add_epi16(vsum23_lo, vsum45_lo), _mm_add_epi16(vsum018_lo, vsum67_lo));

        const __m128i vsum018_hi = _mm_add_epi16(_mm_add_epi16(vi0_hi, vi1_hi), vi8_hi);
        const __m128i vsum23_hi = _mm_add_epi16(vi2_hi, vi3_hi);
        const __m128i vsum45_hi = _mm_add_epi16(vi4_hi, vi5_hi);
        const __m128i vsum67_hi = _mm_add_epi16(vi6_hi, vi7_hi);
        const __m128i vsum_hi = _mm_add_epi16(_mm_add_epi16(vsum23_hi, vsum45_hi), _mm_add_epi16(vsum018_hi, vsum67_hi));

        vacc0123 = _mm_add_epi32(vacc0123, _mm_cvtepi16_epi32(vsum_lo));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_cvtepi16_epi32(_mm_srli_si128(vsum_lo, 8)));
        vacc89AB = _mm_add_epi32(vacc89AB, _mm_cvtepi16_epi32(vsum_hi));
        vaccCDEF = _mm_add_epi32(vaccCDEF, _mm_cvtepi16_epi32(_mm_srli_si128(vsum_hi, 8)));

        k = doz(k, 9);
      } while (k != 0);

      __m128 vscaled0123 = _mm_min_ps(_mm_mul_ps(_mm_cvtepi32_ps(vacc0123), vscale), voutput_max_less_zero_point);
      __m128 vscaled4567 = _mm_min_ps(_mm_mul_ps(_mm_cvtepi32_ps(vacc4567), vscale), voutput_max_less_zero_point);
      __m128 vscaled89AB = _mm_min_ps(_mm_mul_ps(_mm_cvtepi32_ps(vacc89AB), vscale), voutput_max_less_zero_point);
      __m128 vscaledCDEF = _mm_min_ps(_mm_mul_ps(_mm_cvtepi32_ps(vaccCDEF), vscale), voutput_max_less_zero_point);

      vacc0123 = _mm_cvtps_epi32(vscaled0123);
      vacc4567 = _mm_cvtps_epi32(vscaled4567);
      vacc89AB = _mm_cvtps_epi32(vscaled89AB);
      vaccCDEF = _mm_cvtps_epi32(vscaledCDEF);

      const __m128i vout01234567 = _mm_adds_epi16(_mm_packs_epi32(vacc0123, vacc4567), voutput_zero_point);
      const __m128i vout89ABCDEF = _mm_adds_epi16(_mm_packs_epi32(vacc89AB, vaccCDEF), voutput_zero_point);

      __m128i vout = _mm_packs_epi16(vout01234567, vout89ABCDEF);
      vout = _mm_max_epi8(vout, voutput_min);

      _mm_storeu_si128((__m128i*) o, vout);
      o += 16;
      c_offset += 16;
    }
    if XNN_UNLIKELY(c != 0) {
      do {
        const int8_t** i = input;
        size_t k = kernel_elements;
        __m128i vacc0123 = _mm_setzero_si128();
        __m128i vacc4567 = _mm_setzero_si128();
        do {
          vacc0123 = _mm_add_epi32(vacc0123, vinit_bias);
          vacc4567 = _mm_add_epi32(vacc4567, vinit_bias);

          const int8_t* i0 = 0 < k ? *i++ : zero;
          assert(i0 != NULL);
          if XNN_UNPREDICTABLE(i0 != zero) {
            i0 = (const int8_t*) ((uintptr_t) i0 + input_offset);
          }
          i0 += c_offset;
          const __m128i vi0 = _mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*) i0));
          const int8_t* i1 = 1 < k ? *i++ : zero;
          assert(i1 != NULL);
          if XNN_UNPREDICTABLE(i1 != zero) {
            i1 = (const int8_t*) ((uintptr_t) i1 + input_offset);
          }
          i1 += c_offset;
          const __m128i vi1 = _mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*) i1));
          const int8_t* i2 = 2 < k ? *i++ : zero;
          assert(i2 != NULL);
          if XNN_UNPREDICTABLE(i2 != zero) {
            i2 = (const int8_t*) ((uintptr_t) i2 + input_offset);
          }
          i2 += c_offset;
          const __m128i vi2 = _mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*) i2));
          const int8_t* i3 = 3 < k ? *i++ : zero;
          assert(i3 != NULL);
          if XNN_UNPREDICTABLE(i3 != zero) {
            i3 = (const int8_t*) ((uintptr_t) i3 + input_offset);
          }
          i3 += c_offset;
          const __m128i vi3 = _mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*) i3));
          const int8_t* i4 = 4 < k ? *i++ : zero;
          assert(i4 != NULL);
          if XNN_UNPREDICTABLE(i4 != zero) {
            i4 = (const int8_t*) ((uintptr_t) i4 + input_offset);
          }
          i4 += c_offset;
          const __m128i vi4 = _mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*) i4));
          const int8_t* i5 = 5 < k ? *i++ : zero;
          assert(i5 != NULL);
          if XNN_UNPREDICTABLE(i5 != zero) {
            i5 = (const int8_t*) ((uintptr_t) i5 + input_offset);
          }
          i5 += c_offset;
          const __m128i vi5 = _mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*) i5));
          const int8_t* i6 = 6 < k ? *i++ : zero;
          assert(i6 != NULL);
          if XNN_UNPREDICTABLE(i6 != zero) {
            i6 = (const int8_t*) ((uintptr_t) i6 + input_offset);
          }
          i6 += c_offset;
          const __m128i vi6 = _mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*) i6));
          const int8_t* i7 = 7 < k ? *i++ : zero;
          assert(i7 != NULL);
          if XNN_UNPREDICTABLE(i7 != zero) {
            i7 = (const int8_t*) ((uintptr_t) i7 + input_offset);
          }
          i7 += c_offset;
          const __m128i vi7 = _mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*) i7));
          const int8_t* i8 = 8 < k ? *i++ : zero;
          assert(i8 != NULL);
          if XNN_UNPREDICTABLE(i8 != zero) {
            i8 = (const int8_t*) ((uintptr_t) i8 + input_offset);
          }
          i8 += c_offset;
          const __m128i vi8 = _mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*) i8));

          const __m128i vsum018 = _mm_add_epi16(_mm_add_epi16(vi0, vi1), vi8);
          const __m128i vsum23 = _mm_add_epi16(vi2, vi3);
          const __m128i vsum45 = _mm_add_epi16(vi4, vi5);
          const __m128i vsum67 = _mm_add_epi16(vi6, vi7);
          const __m128i vsum = _mm_add_epi16(_mm_add_epi16(vsum23, vsum45), _mm_add_epi16(vsum018, vsum67));

          vacc0123 = _mm_add_epi32(vacc0123, _mm_cvtepi16_epi32(vsum));
          vacc4567 = _mm_add_epi32(vacc4567, _mm_cvtepi16_epi32(_mm_srli_si128(vsum, 8)));

          k = doz(k, 9);
        } while (k != 0);

        __m128 vscaled0123 = _mm_min_ps(_mm_mul_ps(_mm_cvtepi32_ps(vacc0123), vscale), voutput_max_less_zero_point);
        __m128 vscaled4567 = _mm_min_ps(_mm_mul_ps(_mm_cvtepi32_ps(vacc4567), vscale), voutput_max_less_zero_point);

        vacc0123 = _mm_cvtps_epi32(vscaled0123);
        vacc4567 = _mm_cvtps_epi32(vscaled4567);

        const __m128i vout01234567 = _mm_adds_epi16(_mm_packs_epi32(vacc0123, vacc4567), voutput_zero_point);
        __m128i vout = _mm_packs_epi16(vout01234567, vout01234567);
        vout = _mm_max_epi8(vout, voutput_min);

        if XNN_LIKELY(c >= 8) {
          _mm_storel_epi64((__m128i*) o, vout);
          o += 8;
          c_offset += 8;
          c -= 8;
        } else {
          if (c & 4) {
            unaligned_store_u32(o, (uint32_t) _mm_cvtsi128_si32(vout));
            vout = _mm_srli_epi64(vout, 32);
            o += 4;
          }
          if (c & 2) {
            unaligned_store_u16(o, (uint16_t) _mm_extract_epi16(vout, 0));
            vout = _mm_srli_epi32(vout, 16);
            o += 2;
          }
          if (c & 1) {
            *o = (int8_t) _mm_extract_epi8(vout, 0);
          }
          c = 0;
        }
      } while (c != 0);
    }

    input = (const int8_t**) ((uintptr_t) input + input_increment);
    input_offset += input_pixel_stride;
    output = (int8_t*) ((uintptr_t) output + output_increment);
  } while (--output_pixels != 0);
}
