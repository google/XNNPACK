// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/qs8-avgpool/avx2.c.in
//   Generator: tools/xngen
//
// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <immintrin.h>
#include <stddef.h>
#include <stdint.h>

#include "src/xnnpack/avgpool.h"
#include "src/xnnpack/common.h"
#include "src/xnnpack/intrinsics-polyfill.h"
#include "src/xnnpack/math.h"
#include "src/xnnpack/microparams.h"
#include "src/xnnpack/unaligned.h"


void xnn_qs8_avgpool_minmax_fp32_ukernel_9p__avx2_u32(
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

  const __m256i vinit_bias = _mm256_set1_epi32(params->scalar.init_bias);
  __m256 vscale = _mm256_set1_ps(params->scalar.scale);
  const __m256 voutput_max_less_zero_point = _mm256_set1_ps(
      (float) ((int32_t) params->scalar.output_max - (int32_t) params->scalar.output_zero_point));
  const __m256i voutput_zero_point = _mm256_set1_epi16(params->scalar.output_zero_point);
  const __m256i voutput_min = _mm256_set1_epi8((char) params->scalar.output_min);
  const __m256i vpermute_mask = _mm256_set_epi32(7, 3, 6, 2, 5, 1, 4, 0);

  XNN_FORCE_REALIZATION(vinit_bias);
  XNN_FORCE_REALIZATION(voutput_max_less_zero_point);
  XNN_FORCE_REALIZATION(voutput_zero_point);
  XNN_FORCE_REALIZATION(voutput_min);
  XNN_FORCE_REALIZATION(vpermute_mask);

  do {
    if (multiplier != NULL) {
      vscale = _mm256_set1_ps(*multiplier++);
    }

    int8_t* o = output;
    size_t c_offset = 0;
    size_t c = channels;
    for (; c >= 32; c -= 32) {
      const int8_t** i = input;
      size_t k = kernel_elements;
      __m256i vacc0 = _mm256_setzero_si256();
      __m256i vacc1 = _mm256_setzero_si256();
      __m256i vacc2 = _mm256_setzero_si256();
      __m256i vacc3 = _mm256_setzero_si256();
      do {
        vacc0 = _mm256_add_epi32(vacc0, vinit_bias);
        vacc1 = _mm256_add_epi32(vacc1, vinit_bias);
        vacc2 = _mm256_add_epi32(vacc2, vinit_bias);
        vacc3 = _mm256_add_epi32(vacc3, vinit_bias);

        const int8_t* i0 = 0 < k ? *i++ : zero;
        assert(i0 != NULL);
        if XNN_UNPREDICTABLE(i0 != zero) {
          i0 = (const int8_t*) ((uintptr_t) i0 + input_offset);
        }
        i0 += c_offset;
        const __m256i vi0_lo = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) i0));
        const __m256i vi0_hi = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) (i0 + 16)));
        const int8_t* i1 = 1 < k ? *i++ : zero;
        assert(i1 != NULL);
        if XNN_UNPREDICTABLE(i1 != zero) {
          i1 = (const int8_t*) ((uintptr_t) i1 + input_offset);
        }
        i1 += c_offset;
        const __m256i vi1_lo = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) i1));
        const __m256i vi1_hi = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) (i1 + 16)));
        const int8_t* i2 = 2 < k ? *i++ : zero;
        assert(i2 != NULL);
        if XNN_UNPREDICTABLE(i2 != zero) {
          i2 = (const int8_t*) ((uintptr_t) i2 + input_offset);
        }
        i2 += c_offset;
        const __m256i vi2_lo = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) i2));
        const __m256i vi2_hi = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) (i2 + 16)));
        const int8_t* i3 = 3 < k ? *i++ : zero;
        assert(i3 != NULL);
        if XNN_UNPREDICTABLE(i3 != zero) {
          i3 = (const int8_t*) ((uintptr_t) i3 + input_offset);
        }
        i3 += c_offset;
        const __m256i vi3_lo = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) i3));
        const __m256i vi3_hi = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) (i3 + 16)));
        const int8_t* i4 = 4 < k ? *i++ : zero;
        assert(i4 != NULL);
        if XNN_UNPREDICTABLE(i4 != zero) {
          i4 = (const int8_t*) ((uintptr_t) i4 + input_offset);
        }
        i4 += c_offset;
        const __m256i vi4_lo = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) i4));
        const __m256i vi4_hi = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) (i4 + 16)));
        const int8_t* i5 = 5 < k ? *i++ : zero;
        assert(i5 != NULL);
        if XNN_UNPREDICTABLE(i5 != zero) {
          i5 = (const int8_t*) ((uintptr_t) i5 + input_offset);
        }
        i5 += c_offset;
        const __m256i vi5_lo = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) i5));
        const __m256i vi5_hi = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) (i5 + 16)));
        const int8_t* i6 = 6 < k ? *i++ : zero;
        assert(i6 != NULL);
        if XNN_UNPREDICTABLE(i6 != zero) {
          i6 = (const int8_t*) ((uintptr_t) i6 + input_offset);
        }
        i6 += c_offset;
        const __m256i vi6_lo = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) i6));
        const __m256i vi6_hi = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) (i6 + 16)));
        const int8_t* i7 = 7 < k ? *i++ : zero;
        assert(i7 != NULL);
        if XNN_UNPREDICTABLE(i7 != zero) {
          i7 = (const int8_t*) ((uintptr_t) i7 + input_offset);
        }
        i7 += c_offset;
        const __m256i vi7_lo = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) i7));
        const __m256i vi7_hi = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) (i7 + 16)));
        const int8_t* i8 = 8 < k ? *i++ : zero;
        assert(i8 != NULL);
        if XNN_UNPREDICTABLE(i8 != zero) {
          i8 = (const int8_t*) ((uintptr_t) i8 + input_offset);
        }
        i8 += c_offset;
        const __m256i vi8_lo = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) i8));
        const __m256i vi8_hi = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) (i8 + 16)));

        const __m256i vsum018_lo = _mm256_add_epi16(_mm256_add_epi16(vi0_lo, vi1_lo), vi8_lo);
        const __m256i vsum23_lo = _mm256_add_epi16(vi2_lo, vi3_lo);
        const __m256i vsum45_lo = _mm256_add_epi16(vi4_lo, vi5_lo);
        const __m256i vsum67_lo = _mm256_add_epi16(vi6_lo, vi7_lo);
        const __m256i vsum_lo = _mm256_add_epi16(_mm256_add_epi16(vsum23_lo, vsum45_lo), _mm256_add_epi16(vsum018_lo, vsum67_lo));

        const __m256i vsum018_hi = _mm256_add_epi16(_mm256_add_epi16(vi0_hi, vi1_hi), vi8_hi);
        const __m256i vsum23_hi = _mm256_add_epi16(vi2_hi, vi3_hi);
        const __m256i vsum45_hi = _mm256_add_epi16(vi4_hi, vi5_hi);
        const __m256i vsum67_hi = _mm256_add_epi16(vi6_hi, vi7_hi);
        const __m256i vsum_hi = _mm256_add_epi16(_mm256_add_epi16(vsum23_hi, vsum45_hi), _mm256_add_epi16(vsum018_hi, vsum67_hi));

        vacc0 = _mm256_add_epi32(vacc0, _mm256_cvtepi16_epi32(_mm256_castsi256_si128(vsum_lo)));
        vacc1 = _mm256_add_epi32(vacc1, _mm256_cvtepi16_epi32(_mm256_extracti128_si256(vsum_lo, 1)));
        vacc2 = _mm256_add_epi32(vacc2, _mm256_cvtepi16_epi32(_mm256_castsi256_si128(vsum_hi)));
        vacc3 = _mm256_add_epi32(vacc3, _mm256_cvtepi16_epi32(_mm256_extracti128_si256(vsum_hi, 1)));

        k = doz(k, 9);
      } while (k != 0);

      __m256 vscaled0 = _mm256_min_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(vacc0), vscale), voutput_max_less_zero_point);
      __m256 vscaled1 = _mm256_min_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(vacc1), vscale), voutput_max_less_zero_point);
      __m256 vscaled2 = _mm256_min_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(vacc2), vscale), voutput_max_less_zero_point);
      __m256 vscaled3 = _mm256_min_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(vacc3), vscale), voutput_max_less_zero_point);

      vacc0 = _mm256_cvtps_epi32(vscaled0);
      vacc1 = _mm256_cvtps_epi32(vscaled1);
      vacc2 = _mm256_cvtps_epi32(vscaled2);
      vacc3 = _mm256_cvtps_epi32(vscaled3);

      const __m256i vout01 = _mm256_adds_epi16(_mm256_packs_epi32(vacc0, vacc1), voutput_zero_point);
      const __m256i vout23 = _mm256_adds_epi16(_mm256_packs_epi32(vacc2, vacc3), voutput_zero_point);

      __m256i vout = _mm256_packs_epi16(vout01, vout23);
      vout = _mm256_permutevar8x32_epi32(vout, vpermute_mask);
      vout = _mm256_max_epi8(vout, voutput_min);

      _mm256_storeu_si256((__m256i*) o, vout);
      o += 32;
      c_offset += 32;
    }
    if XNN_UNLIKELY(c != 0) {
      do {
        const int8_t** i = input;
        size_t k = kernel_elements;
        __m256i vacc0 = _mm256_setzero_si256();
        __m256i vacc1 = _mm256_setzero_si256();
        do {
          vacc0 = _mm256_add_epi32(vacc0, vinit_bias);
          vacc1 = _mm256_add_epi32(vacc1, vinit_bias);

          const int8_t* i0 = 0 < k ? *i++ : zero;
          assert(i0 != NULL);
          if XNN_UNPREDICTABLE(i0 != zero) {
            i0 = (const int8_t*) ((uintptr_t) i0 + input_offset);
          }
          i0 += c_offset;
          const __m256i vi0 = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) i0));
          const int8_t* i1 = 1 < k ? *i++ : zero;
          assert(i1 != NULL);
          if XNN_UNPREDICTABLE(i1 != zero) {
            i1 = (const int8_t*) ((uintptr_t) i1 + input_offset);
          }
          i1 += c_offset;
          const __m256i vi1 = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) i1));
          const int8_t* i2 = 2 < k ? *i++ : zero;
          assert(i2 != NULL);
          if XNN_UNPREDICTABLE(i2 != zero) {
            i2 = (const int8_t*) ((uintptr_t) i2 + input_offset);
          }
          i2 += c_offset;
          const __m256i vi2 = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) i2));
          const int8_t* i3 = 3 < k ? *i++ : zero;
          assert(i3 != NULL);
          if XNN_UNPREDICTABLE(i3 != zero) {
            i3 = (const int8_t*) ((uintptr_t) i3 + input_offset);
          }
          i3 += c_offset;
          const __m256i vi3 = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) i3));
          const int8_t* i4 = 4 < k ? *i++ : zero;
          assert(i4 != NULL);
          if XNN_UNPREDICTABLE(i4 != zero) {
            i4 = (const int8_t*) ((uintptr_t) i4 + input_offset);
          }
          i4 += c_offset;
          const __m256i vi4 = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) i4));
          const int8_t* i5 = 5 < k ? *i++ : zero;
          assert(i5 != NULL);
          if XNN_UNPREDICTABLE(i5 != zero) {
            i5 = (const int8_t*) ((uintptr_t) i5 + input_offset);
          }
          i5 += c_offset;
          const __m256i vi5 = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) i5));
          const int8_t* i6 = 6 < k ? *i++ : zero;
          assert(i6 != NULL);
          if XNN_UNPREDICTABLE(i6 != zero) {
            i6 = (const int8_t*) ((uintptr_t) i6 + input_offset);
          }
          i6 += c_offset;
          const __m256i vi6 = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) i6));
          const int8_t* i7 = 7 < k ? *i++ : zero;
          assert(i7 != NULL);
          if XNN_UNPREDICTABLE(i7 != zero) {
            i7 = (const int8_t*) ((uintptr_t) i7 + input_offset);
          }
          i7 += c_offset;
          const __m256i vi7 = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) i7));
          const int8_t* i8 = 8 < k ? *i++ : zero;
          assert(i8 != NULL);
          if XNN_UNPREDICTABLE(i8 != zero) {
            i8 = (const int8_t*) ((uintptr_t) i8 + input_offset);
          }
          i8 += c_offset;
          const __m256i vi8 = _mm256_cvtepi8_epi16(_mm_loadu_si128((const __m128i*) i8));

          const __m256i vsum018 = _mm256_add_epi16(_mm256_add_epi16(vi0, vi1), vi8);
          const __m256i vsum23 = _mm256_add_epi16(vi2, vi3);
          const __m256i vsum45 = _mm256_add_epi16(vi4, vi5);
          const __m256i vsum67 = _mm256_add_epi16(vi6, vi7);
          const __m256i vsum = _mm256_add_epi16(_mm256_add_epi16(vsum23, vsum45), _mm256_add_epi16(vsum018, vsum67));

          vacc0 = _mm256_add_epi32(vacc0, _mm256_cvtepi16_epi32(_mm256_castsi256_si128(vsum)));
          vacc1 = _mm256_add_epi32(vacc1, _mm256_cvtepi16_epi32(_mm256_extracti128_si256(vsum, 1)));

          k = doz(k, 9);
        } while (k != 0);

        __m256 vscaled0 = _mm256_min_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(vacc0), vscale), voutput_max_less_zero_point);
        __m256 vscaled1 = _mm256_min_ps(_mm256_mul_ps(_mm256_cvtepi32_ps(vacc1), vscale), voutput_max_less_zero_point);

        vacc0 = _mm256_cvtps_epi32(vscaled0);
        vacc1 = _mm256_cvtps_epi32(vscaled1);

        const __m256i vout01 = _mm256_adds_epi16(_mm256_packs_epi32(vacc0, vacc1), voutput_zero_point);
        __m128i vout = _mm_shuffle_epi32(
            _mm_packs_epi16(_mm256_castsi256_si128(vout01), _mm256_extracti128_si256(vout01, 1)),
            _MM_SHUFFLE(3, 1, 2, 0));
        vout = _mm_max_epi8(vout, _mm256_castsi256_si128(voutput_min));

        if XNN_LIKELY(c >= 16) {
          _mm_storeu_si128((__m128i*) o, vout);
          o += 16;
          c_offset += 16;
          c -= 16;
        } else {
          if (c & 8) {
            _mm_storel_epi64((__m128i*) o, vout);
            vout = _mm_srli_si128(vout, 8);
            o += 8;
          }
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
