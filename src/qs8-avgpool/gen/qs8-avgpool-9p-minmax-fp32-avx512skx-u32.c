// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/qs8-avgpool/avx512skx.c.in
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


void xnn_qs8_avgpool_minmax_fp32_ukernel_9p__avx512skx_u32(
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

  const __m512i vinit_bias = _mm512_set1_epi32(params->scalar.init_bias);
  __m512 vscale = _mm512_set1_ps(params->scalar.scale);
  const __m512 voutput_max_less_zero_point = _mm512_set1_ps(
      (float) ((int32_t) params->scalar.output_max - (int32_t) params->scalar.output_zero_point));
  const __m512i voutput_zero_point = _mm512_set1_epi16(params->scalar.output_zero_point);
  const __m256i voutput_min = _mm256_set1_epi8((char) params->scalar.output_min);
  const __m256i vpermute_mask = _mm256_set_epi32(7, 3, 5, 1, 6, 2, 4, 0);

  do {
    if (multiplier != NULL) {
      vscale = _mm512_set1_ps(*multiplier++);
    }

    int8_t* o = output;
    size_t c_offset = 0;
    size_t c = channels;
    for (; c >= 32; c -= 32) {
      const int8_t** i = input;
      size_t k = kernel_elements;
      __m512i vacc0 = _mm512_setzero_si512();
      __m512i vacc1 = _mm512_setzero_si512();
      do {
        vacc0 = _mm512_add_epi32(vacc0, vinit_bias);
        vacc1 = _mm512_add_epi32(vacc1, vinit_bias);

        const int8_t* i0 = 0 < k ? *i++ : zero;
        assert(i0 != NULL);
        if XNN_UNPREDICTABLE(i0 != zero) {
          i0 = (const int8_t*) ((uintptr_t) i0 + input_offset);
        }
        i0 += c_offset;
        const __m512i vi0 = _mm512_cvtepi8_epi16(_mm256_loadu_si256((const __m256i*) i0));
        const int8_t* i1 = 1 < k ? *i++ : zero;
        assert(i1 != NULL);
        if XNN_UNPREDICTABLE(i1 != zero) {
          i1 = (const int8_t*) ((uintptr_t) i1 + input_offset);
        }
        i1 += c_offset;
        const __m512i vi1 = _mm512_cvtepi8_epi16(_mm256_loadu_si256((const __m256i*) i1));
        const int8_t* i2 = 2 < k ? *i++ : zero;
        assert(i2 != NULL);
        if XNN_UNPREDICTABLE(i2 != zero) {
          i2 = (const int8_t*) ((uintptr_t) i2 + input_offset);
        }
        i2 += c_offset;
        const __m512i vi2 = _mm512_cvtepi8_epi16(_mm256_loadu_si256((const __m256i*) i2));
        const int8_t* i3 = 3 < k ? *i++ : zero;
        assert(i3 != NULL);
        if XNN_UNPREDICTABLE(i3 != zero) {
          i3 = (const int8_t*) ((uintptr_t) i3 + input_offset);
        }
        i3 += c_offset;
        const __m512i vi3 = _mm512_cvtepi8_epi16(_mm256_loadu_si256((const __m256i*) i3));
        const int8_t* i4 = 4 < k ? *i++ : zero;
        assert(i4 != NULL);
        if XNN_UNPREDICTABLE(i4 != zero) {
          i4 = (const int8_t*) ((uintptr_t) i4 + input_offset);
        }
        i4 += c_offset;
        const __m512i vi4 = _mm512_cvtepi8_epi16(_mm256_loadu_si256((const __m256i*) i4));
        const int8_t* i5 = 5 < k ? *i++ : zero;
        assert(i5 != NULL);
        if XNN_UNPREDICTABLE(i5 != zero) {
          i5 = (const int8_t*) ((uintptr_t) i5 + input_offset);
        }
        i5 += c_offset;
        const __m512i vi5 = _mm512_cvtepi8_epi16(_mm256_loadu_si256((const __m256i*) i5));
        const int8_t* i6 = 6 < k ? *i++ : zero;
        assert(i6 != NULL);
        if XNN_UNPREDICTABLE(i6 != zero) {
          i6 = (const int8_t*) ((uintptr_t) i6 + input_offset);
        }
        i6 += c_offset;
        const __m512i vi6 = _mm512_cvtepi8_epi16(_mm256_loadu_si256((const __m256i*) i6));
        const int8_t* i7 = 7 < k ? *i++ : zero;
        assert(i7 != NULL);
        if XNN_UNPREDICTABLE(i7 != zero) {
          i7 = (const int8_t*) ((uintptr_t) i7 + input_offset);
        }
        i7 += c_offset;
        const __m512i vi7 = _mm512_cvtepi8_epi16(_mm256_loadu_si256((const __m256i*) i7));
        const int8_t* i8 = 8 < k ? *i++ : zero;
        assert(i8 != NULL);
        if XNN_UNPREDICTABLE(i8 != zero) {
          i8 = (const int8_t*) ((uintptr_t) i8 + input_offset);
        }
        i8 += c_offset;
        const __m512i vi8 = _mm512_cvtepi8_epi16(_mm256_loadu_si256((const __m256i*) i8));

        const __m512i vsum018 = _mm512_add_epi16(_mm512_add_epi16(vi0, vi1), vi8);
        const __m512i vsum23 = _mm512_add_epi16(vi2, vi3);
        const __m512i vsum45 = _mm512_add_epi16(vi4, vi5);
        const __m512i vsum67 = _mm512_add_epi16(vi6, vi7);
        const __m512i vsum = _mm512_add_epi16(_mm512_add_epi16(vsum23, vsum45), _mm512_add_epi16(vsum018, vsum67));

        vacc0 = _mm512_add_epi32(vacc0, _mm512_cvtepi16_epi32(_mm512_castsi512_si256(vsum)));
        vacc1 = _mm512_add_epi32(vacc1, _mm512_cvtepi16_epi32(_mm512_extracti32x8_epi32(vsum, 1)));

        k = doz(k, 9);
      } while (k != 0);

      __m512 vscaled0 = _mm512_min_ps(_mm512_mul_ps(_mm512_cvtepi32_ps(vacc0), vscale), voutput_max_less_zero_point);
      __m512 vscaled1 = _mm512_min_ps(_mm512_mul_ps(_mm512_cvtepi32_ps(vacc1), vscale), voutput_max_less_zero_point);

      vacc0 = _mm512_cvtps_epi32(vscaled0);
      vacc1 = _mm512_cvtps_epi32(vscaled1);

      const __m512i vout16 = _mm512_adds_epi16(_mm512_packs_epi32(vacc0, vacc1), voutput_zero_point);
      const __m256i vout16_lo = _mm512_castsi512_si256(vout16);
      const __m256i vout16_hi = _mm512_extracti32x8_epi32(vout16, 1);
      __m256i vout = _mm256_packs_epi16(vout16_lo, vout16_hi);
      vout = _mm256_permutevar8x32_epi32(vout, vpermute_mask);
      vout = _mm256_max_epi8(vout, voutput_min);

      _mm256_storeu_si256((__m256i*) o, vout);
      o += 32;
      c_offset += 32;
    }
    if XNN_UNLIKELY(c != 0) {
      const __mmask16 vmask = _cvtu32_mask16((uint32_t) ((UINT32_C(1) << (c & 15)) - UINT32_C(1)));
      do {
        const int8_t** i = input;
        size_t k = kernel_elements;
        __m512i vacc0 = _mm512_setzero_si512();
        do {
          vacc0 = _mm512_add_epi32(vacc0, vinit_bias);

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

          vacc0 = _mm512_add_epi32(vacc0, _mm512_cvtepi16_epi32(vsum));

          k = doz(k, 9);
        } while (k != 0);

        __m512 vscaled0 = _mm512_min_ps(_mm512_mul_ps(_mm512_cvtepi32_ps(vacc0), vscale), voutput_max_less_zero_point);
        vacc0 = _mm512_cvtps_epi32(vscaled0);

        const __m256i vout16 = _mm256_adds_epi16(
            _mm256_packs_epi32(_mm512_castsi512_si256(vacc0), _mm512_extracti32x8_epi32(vacc0, 1)),
            _mm512_castsi512_si256(voutput_zero_point));
        __m128i vout = _mm_shuffle_epi32(
            _mm_packs_epi16(_mm256_castsi256_si128(vout16), _mm256_extracti128_si256(vout16, 1)),
            _MM_SHUFFLE(3, 1, 2, 0));
        vout = _mm_max_epi8(vout, _mm256_castsi256_si128(voutput_min));

        if XNN_LIKELY(c >= 16) {
          _mm_storeu_si128((__m128i*) o, vout);
          o += 16;
          c_offset += 16;
          c -= 16;
        } else {
          _mm_mask_storeu_epi8(o, vmask, vout);
          o = (int8_t*) ((uintptr_t) o + c);
          c = 0;
        }
      } while (c != 0);
    }

    input = (const int8_t**) ((uintptr_t) input + input_increment);
    input_offset += input_pixel_stride;
    output = (int8_t*) ((uintptr_t) output + output_increment);
  } while (--output_pixels != 0);
}
