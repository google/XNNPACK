// Auto-generated file. Do not edit!
//   Template: src/qs8-dwconv/unipass-sse-mul32.c.in
//   Generator: tools/xngen
//
// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <immintrin.h>

#include <xnnpack/dwconv.h>
#include <xnnpack/intrinsics-polyfill.h>


void xnn_qs8_dwconv_minmax_gemmlowp_ukernel_up8x9__sse41_mul32(
    size_t channels,
    size_t output_width,
    const int8_t** input,
    const void* weights,
    int8_t* output,
    size_t input_stride,
    size_t output_increment,
    size_t input_offset,
    const int8_t* zero,
    const union xnn_qs8_conv_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_DISABLE_TSAN XNN_DISABLE_MSAN
{
  assert(channels != 0);
  assert(output_width != 0);

  do {
    const int8_t* i0 = input[0];
    assert(i0 != NULL);
    if XNN_UNPREDICTABLE(i0 != zero) {
      i0 = (const int8_t*) ((uintptr_t) i0 + input_offset);
    }
    const int8_t* i1 = input[1];
    assert(i1 != NULL);
    if XNN_UNPREDICTABLE(i1 != zero) {
      i1 = (const int8_t*) ((uintptr_t) i1 + input_offset);
    }
    const int8_t* i2 = input[2];
    assert(i2 != NULL);
    if XNN_UNPREDICTABLE(i2 != zero) {
      i2 = (const int8_t*) ((uintptr_t) i2 + input_offset);
    }
    const int8_t* i3 = input[3];
    assert(i3 != NULL);
    if XNN_UNPREDICTABLE(i3 != zero) {
      i3 = (const int8_t*) ((uintptr_t) i3 + input_offset);
    }
    const int8_t* i4 = input[4];
    assert(i4 != NULL);
    if XNN_UNPREDICTABLE(i4 != zero) {
      i4 = (const int8_t*) ((uintptr_t) i4 + input_offset);
    }
    const int8_t* i5 = input[5];
    assert(i5 != NULL);
    if XNN_UNPREDICTABLE(i5 != zero) {
      i5 = (const int8_t*) ((uintptr_t) i5 + input_offset);
    }
    const int8_t* i6 = input[6];
    assert(i6 != NULL);
    if XNN_UNPREDICTABLE(i6 != zero) {
      i6 = (const int8_t*) ((uintptr_t) i6 + input_offset);
    }
    const int8_t* i7 = input[7];
    assert(i7 != NULL);
    if XNN_UNPREDICTABLE(i7 != zero) {
      i7 = (const int8_t*) ((uintptr_t) i7 + input_offset);
    }
    const int8_t* i8 = input[8];
    assert(i8 != NULL);
    if XNN_UNPREDICTABLE(i8 != zero) {
      i8 = (const int8_t*) ((uintptr_t) i8 + input_offset);
    }
    input = (const int8_t**) ((uintptr_t) input + input_stride);

    size_t c = channels;
    const void* w = weights;
    for (; c >= 8; c -= 8) {
      __m128i vacc0123 = _mm_loadu_si128((const __m128i*) w);
      __m128i vacc4567 = _mm_loadu_si128((const __m128i*) ((const int32_t*) w + 4));


      const __m128i vi0x0123 = _mm_cvtepi8_epi32(_mm_loadu_si32(i0));
      const __m128i vk0x0123 = _mm_cvtepi8_epi32(_mm_loadu_si32((const void*) ((uintptr_t) w + 8 * sizeof(int32_t) + 0 * sizeof(int8_t))));
      const __m128i vi0x4567 = _mm_cvtepi8_epi32(_mm_loadu_si32(i0 + 4));
      const __m128i vk0x4567 = _mm_cvtepi8_epi32(_mm_loadu_si32((const void*) ((uintptr_t) w + 8 * sizeof(int32_t) + 4 * sizeof(int8_t))));
      i0 += 8;

      vacc0123 = _mm_add_epi32(vacc0123, _mm_mullo_epi32(vi0x0123, vk0x0123));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_mullo_epi32(vi0x4567, vk0x4567));

      const __m128i vi1x0123 = _mm_cvtepi8_epi32(_mm_loadu_si32(i1));
      const __m128i vk1x0123 = _mm_cvtepi8_epi32(_mm_loadu_si32((const void*) ((uintptr_t) w + 8 * sizeof(int32_t) + 8 * sizeof(int8_t))));
      const __m128i vi1x4567 = _mm_cvtepi8_epi32(_mm_loadu_si32(i1 + 4));
      const __m128i vk1x4567 = _mm_cvtepi8_epi32(_mm_loadu_si32((const void*) ((uintptr_t) w + 8 * sizeof(int32_t) + 12 * sizeof(int8_t))));
      i1 += 8;

      vacc0123 = _mm_add_epi32(vacc0123, _mm_mullo_epi32(vi1x0123, vk1x0123));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_mullo_epi32(vi1x4567, vk1x4567));

      const __m128i vi2x0123 = _mm_cvtepi8_epi32(_mm_loadu_si32(i2));
      const __m128i vk2x0123 = _mm_cvtepi8_epi32(_mm_loadu_si32((const void*) ((uintptr_t) w + 8 * sizeof(int32_t) + 16 * sizeof(int8_t))));
      const __m128i vi2x4567 = _mm_cvtepi8_epi32(_mm_loadu_si32(i2 + 4));
      const __m128i vk2x4567 = _mm_cvtepi8_epi32(_mm_loadu_si32((const void*) ((uintptr_t) w + 8 * sizeof(int32_t) + 20 * sizeof(int8_t))));
      i2 += 8;

      vacc0123 = _mm_add_epi32(vacc0123, _mm_mullo_epi32(vi2x0123, vk2x0123));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_mullo_epi32(vi2x4567, vk2x4567));

      const __m128i vi3x0123 = _mm_cvtepi8_epi32(_mm_loadu_si32(i3));
      const __m128i vk3x0123 = _mm_cvtepi8_epi32(_mm_loadu_si32((const void*) ((uintptr_t) w + 8 * sizeof(int32_t) + 24 * sizeof(int8_t))));
      const __m128i vi3x4567 = _mm_cvtepi8_epi32(_mm_loadu_si32(i3 + 4));
      const __m128i vk3x4567 = _mm_cvtepi8_epi32(_mm_loadu_si32((const void*) ((uintptr_t) w + 8 * sizeof(int32_t) + 28 * sizeof(int8_t))));
      i3 += 8;

      vacc0123 = _mm_add_epi32(vacc0123, _mm_mullo_epi32(vi3x0123, vk3x0123));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_mullo_epi32(vi3x4567, vk3x4567));

      const __m128i vi4x0123 = _mm_cvtepi8_epi32(_mm_loadu_si32(i4));
      const __m128i vk4x0123 = _mm_cvtepi8_epi32(_mm_loadu_si32((const void*) ((uintptr_t) w + 8 * sizeof(int32_t) + 32 * sizeof(int8_t))));
      const __m128i vi4x4567 = _mm_cvtepi8_epi32(_mm_loadu_si32(i4 + 4));
      const __m128i vk4x4567 = _mm_cvtepi8_epi32(_mm_loadu_si32((const void*) ((uintptr_t) w + 8 * sizeof(int32_t) + 36 * sizeof(int8_t))));
      i4 += 8;

      vacc0123 = _mm_add_epi32(vacc0123, _mm_mullo_epi32(vi4x0123, vk4x0123));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_mullo_epi32(vi4x4567, vk4x4567));

      const __m128i vi5x0123 = _mm_cvtepi8_epi32(_mm_loadu_si32(i5));
      const __m128i vk5x0123 = _mm_cvtepi8_epi32(_mm_loadu_si32((const void*) ((uintptr_t) w + 8 * sizeof(int32_t) + 40 * sizeof(int8_t))));
      const __m128i vi5x4567 = _mm_cvtepi8_epi32(_mm_loadu_si32(i5 + 4));
      const __m128i vk5x4567 = _mm_cvtepi8_epi32(_mm_loadu_si32((const void*) ((uintptr_t) w + 8 * sizeof(int32_t) + 44 * sizeof(int8_t))));
      i5 += 8;

      vacc0123 = _mm_add_epi32(vacc0123, _mm_mullo_epi32(vi5x0123, vk5x0123));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_mullo_epi32(vi5x4567, vk5x4567));

      const __m128i vi6x0123 = _mm_cvtepi8_epi32(_mm_loadu_si32(i6));
      const __m128i vk6x0123 = _mm_cvtepi8_epi32(_mm_loadu_si32((const void*) ((uintptr_t) w + 8 * sizeof(int32_t) + 48 * sizeof(int8_t))));
      const __m128i vi6x4567 = _mm_cvtepi8_epi32(_mm_loadu_si32(i6 + 4));
      const __m128i vk6x4567 = _mm_cvtepi8_epi32(_mm_loadu_si32((const void*) ((uintptr_t) w + 8 * sizeof(int32_t) + 52 * sizeof(int8_t))));
      i6 += 8;

      vacc0123 = _mm_add_epi32(vacc0123, _mm_mullo_epi32(vi6x0123, vk6x0123));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_mullo_epi32(vi6x4567, vk6x4567));

      const __m128i vi7x0123 = _mm_cvtepi8_epi32(_mm_loadu_si32(i7));
      const __m128i vk7x0123 = _mm_cvtepi8_epi32(_mm_loadu_si32((const void*) ((uintptr_t) w + 8 * sizeof(int32_t) + 56 * sizeof(int8_t))));
      const __m128i vi7x4567 = _mm_cvtepi8_epi32(_mm_loadu_si32(i7 + 4));
      const __m128i vk7x4567 = _mm_cvtepi8_epi32(_mm_loadu_si32((const void*) ((uintptr_t) w + 8 * sizeof(int32_t) + 60 * sizeof(int8_t))));
      i7 += 8;

      vacc0123 = _mm_add_epi32(vacc0123, _mm_mullo_epi32(vi7x0123, vk7x0123));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_mullo_epi32(vi7x4567, vk7x4567));

      const __m128i vi8x0123 = _mm_cvtepi8_epi32(_mm_loadu_si32(i8));
      const __m128i vk8x0123 = _mm_cvtepi8_epi32(_mm_loadu_si32((const void*) ((uintptr_t) w + 8 * sizeof(int32_t) + 64 * sizeof(int8_t))));
      const __m128i vi8x4567 = _mm_cvtepi8_epi32(_mm_loadu_si32(i8 + 4));
      const __m128i vk8x4567 = _mm_cvtepi8_epi32(_mm_loadu_si32((const void*) ((uintptr_t) w + 8 * sizeof(int32_t) + 68 * sizeof(int8_t))));
      i8 += 8;

      vacc0123 = _mm_add_epi32(vacc0123, _mm_mullo_epi32(vi8x0123, vk8x0123));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_mullo_epi32(vi8x4567, vk8x4567));

      w = (const void*) ((uintptr_t) w + 8 * sizeof(int32_t) + 72 * sizeof(int8_t));

      const __m128i vmultiplier = _mm_load_si128((const __m128i*) params->gemmlowp_sse4.multiplier);
      const __m128i vrounding = _mm_load_si128((const __m128i*) params->gemmlowp_sse4.rounding);

      const __m128i vacc13 = _mm_shuffle_epi32(vacc0123, _MM_SHUFFLE(3, 3, 1, 1));
      const __m128i vprod02 = _mm_add_epi64(_mm_mul_epi32(vacc0123, vmultiplier), vrounding);
      const __m128i vprod13 = _mm_add_epi64(_mm_mul_epi32(vacc13, vmultiplier), vrounding);
      const __m128i vacc57 = _mm_shuffle_epi32(vacc4567, _MM_SHUFFLE(3, 3, 1, 1));
      const __m128i vprod46 = _mm_add_epi64(_mm_mul_epi32(vacc4567, vmultiplier), vrounding);
      const __m128i vprod57 = _mm_add_epi64(_mm_mul_epi32(vacc57, vmultiplier), vrounding);

      const __m128i vq31prod02 = _mm_srli_epi64(vprod02, 31);
      const __m128i vq31prod13 = _mm_add_epi64(vprod13, vprod13);
      const __m128i vq31prod46 = _mm_srli_epi64(vprod46, 31);
      const __m128i vq31prod57 = _mm_add_epi64(vprod57, vprod57);

      const __m128i vq31prod0123 = _mm_blend_epi16(vq31prod02, vq31prod13, 0xCC);
      const __m128i vq31prod4567 = _mm_blend_epi16(vq31prod46, vq31prod57, 0xCC);

      const __m128i vremainder_mask = _mm_load_si128((const __m128i*) params->gemmlowp_sse4.remainder_mask);
      const __m128i vrem0123 =
        _mm_add_epi32(_mm_and_si128(vq31prod0123, vremainder_mask), _mm_cmpgt_epi32(_mm_setzero_si128(), vq31prod0123));
      const __m128i vrem4567 =
        _mm_add_epi32(_mm_and_si128(vq31prod4567, vremainder_mask), _mm_cmpgt_epi32(_mm_setzero_si128(), vq31prod4567));

      const __m128i vremainder_threshold = _mm_load_si128((const __m128i*) params->gemmlowp_sse4.remainder_threshold);
      const __m128i vshift = _mm_loadl_epi64((const __m128i*) params->gemmlowp_sse4.shift);
      vacc0123 =
        _mm_sub_epi32(_mm_sra_epi32(vq31prod0123, vshift), _mm_cmpgt_epi32(vrem0123, vremainder_threshold));
      vacc4567 =
        _mm_sub_epi32(_mm_sra_epi32(vq31prod4567, vshift), _mm_cmpgt_epi32(vrem4567, vremainder_threshold));

      const __m128i voutput_zero_point = _mm_load_si128((const __m128i*) params->gemmlowp_sse4.output_zero_point);
      __m128i vout01234567 = _mm_adds_epi16(_mm_packs_epi32(vacc0123, vacc4567), voutput_zero_point);

      const __m128i voutput_min = _mm_load_si128((const __m128i*) params->gemmlowp_sse4.output_min);
      const __m128i voutput_max = _mm_load_si128((const __m128i*) params->gemmlowp_sse4.output_max);
      __m128i vout0123456701234567 = _mm_packs_epi16(vout01234567, vout01234567);
      vout0123456701234567 = _mm_max_epi8(vout0123456701234567, voutput_min);
      vout0123456701234567 = _mm_min_epi8(vout0123456701234567, voutput_max);

      _mm_storel_epi64((__m128i*) output, vout0123456701234567);
      output += 8;
    }
    if XNN_UNLIKELY(c != 0) {
      const int8_t* k = (const int8_t*) ((const int32_t*) w + 8);
      do {
        __m128i vacc0123 = _mm_loadu_si128((const __m128i*) w);

        const __m128i vi0x0123 = _mm_cvtepi8_epi32(_mm_loadu_si32(i0));
        const __m128i vk0x0123 = _mm_cvtepi8_epi32(_mm_loadu_si32(k));
        i0 += 4;

        vacc0123 = _mm_add_epi32(vacc0123, _mm_mullo_epi32(vi0x0123, vk0x0123));
        const __m128i vi1x0123 = _mm_cvtepi8_epi32(_mm_loadu_si32(i1));
        const __m128i vk1x0123 = _mm_cvtepi8_epi32(_mm_loadu_si32((const void*) (k + 8)));
        i1 += 4;

        vacc0123 = _mm_add_epi32(vacc0123, _mm_mullo_epi32(vi1x0123, vk1x0123));
        const __m128i vi2x0123 = _mm_cvtepi8_epi32(_mm_loadu_si32(i2));
        const __m128i vk2x0123 = _mm_cvtepi8_epi32(_mm_loadu_si32((const void*) (k + 16)));
        i2 += 4;

        vacc0123 = _mm_add_epi32(vacc0123, _mm_mullo_epi32(vi2x0123, vk2x0123));
        const __m128i vi3x0123 = _mm_cvtepi8_epi32(_mm_loadu_si32(i3));
        const __m128i vk3x0123 = _mm_cvtepi8_epi32(_mm_loadu_si32((const void*) (k + 24)));
        i3 += 4;

        vacc0123 = _mm_add_epi32(vacc0123, _mm_mullo_epi32(vi3x0123, vk3x0123));
        const __m128i vi4x0123 = _mm_cvtepi8_epi32(_mm_loadu_si32(i4));
        const __m128i vk4x0123 = _mm_cvtepi8_epi32(_mm_loadu_si32((const void*) (k + 32)));
        i4 += 4;

        vacc0123 = _mm_add_epi32(vacc0123, _mm_mullo_epi32(vi4x0123, vk4x0123));
        const __m128i vi5x0123 = _mm_cvtepi8_epi32(_mm_loadu_si32(i5));
        const __m128i vk5x0123 = _mm_cvtepi8_epi32(_mm_loadu_si32((const void*) (k + 40)));
        i5 += 4;

        vacc0123 = _mm_add_epi32(vacc0123, _mm_mullo_epi32(vi5x0123, vk5x0123));
        const __m128i vi6x0123 = _mm_cvtepi8_epi32(_mm_loadu_si32(i6));
        const __m128i vk6x0123 = _mm_cvtepi8_epi32(_mm_loadu_si32((const void*) (k + 48)));
        i6 += 4;

        vacc0123 = _mm_add_epi32(vacc0123, _mm_mullo_epi32(vi6x0123, vk6x0123));
        const __m128i vi7x0123 = _mm_cvtepi8_epi32(_mm_loadu_si32(i7));
        const __m128i vk7x0123 = _mm_cvtepi8_epi32(_mm_loadu_si32((const void*) (k + 56)));
        i7 += 4;

        vacc0123 = _mm_add_epi32(vacc0123, _mm_mullo_epi32(vi7x0123, vk7x0123));
        const __m128i vi8x0123 = _mm_cvtepi8_epi32(_mm_loadu_si32(i8));
        const __m128i vk8x0123 = _mm_cvtepi8_epi32(_mm_loadu_si32((const void*) (k + 64)));
        i8 += 4;

        vacc0123 = _mm_add_epi32(vacc0123, _mm_mullo_epi32(vi8x0123, vk8x0123));

        k += 4;

        const __m128i vmultiplier = _mm_load_si128((const __m128i*) params->gemmlowp_sse4.multiplier);
        const __m128i vrounding = _mm_load_si128((const __m128i*) params->gemmlowp_sse4.rounding);

        const __m128i vacc13 = _mm_shuffle_epi32(vacc0123, _MM_SHUFFLE(3, 3, 1, 1));

        const __m128i vprod02 = _mm_add_epi64(_mm_mul_epi32(vacc0123, vmultiplier), vrounding);
        const __m128i vprod13 = _mm_add_epi64(_mm_mul_epi32(vacc13, vmultiplier), vrounding);

        const __m128i vq31prod02 = _mm_srli_epi64(vprod02, 31);
        const __m128i vq31prod13 = _mm_add_epi64(vprod13, vprod13);

        const __m128i vq31prod0123 = _mm_blend_epi16(vq31prod02, vq31prod13, 0xCC);

        const __m128i vremainder_mask = _mm_load_si128((const __m128i*) params->gemmlowp_sse4.remainder_mask);
        const __m128i vrem0123 =
          _mm_add_epi32(_mm_and_si128(vq31prod0123, vremainder_mask), _mm_cmpgt_epi32(_mm_setzero_si128(), vq31prod0123));

        const __m128i vremainder_threshold = _mm_load_si128((const __m128i*) params->gemmlowp_sse4.remainder_threshold);
        const __m128i vshift = _mm_loadl_epi64((const __m128i*) params->gemmlowp_sse4.shift);
        vacc0123 =
          _mm_sub_epi32(_mm_sra_epi32(vq31prod0123, vshift), _mm_cmpgt_epi32(vrem0123, vremainder_threshold));

        w = (const void*) ((const int32_t*) w + 4);

        const __m128i voutput_zero_point = _mm_load_si128((const __m128i*) params->gemmlowp_sse4.output_zero_point);
        __m128i vout0123 = _mm_adds_epi16(_mm_packs_epi32(vacc0123, vacc0123), voutput_zero_point);

        vout0123 = _mm_packs_epi16(vout0123, vout0123);
        vout0123 = _mm_max_epi8(vout0123, _mm_load_si128((const __m128i*) params->gemmlowp_sse4.output_min));
        vout0123 = _mm_min_epi8(vout0123, _mm_load_si128((const __m128i*) params->gemmlowp_sse4.output_max));

        if XNN_LIKELY(c >= 4) {
          _mm_storeu_si32(output, vout0123);
          output += 4;
          c -= 4;
        } else {
          if (c & 2) {
            *((uint16_t*) output) = (uint16_t) _mm_extract_epi16(vout0123, 0);
            vout0123 = _mm_srli_epi32(vout0123, 16);
            output += 2;
          }
          if (c & 1) {
            *output = (int8_t) _mm_extract_epi8(vout0123, 0);
            output += 1;
          }
          c = 0;
        }
      } while (c != 0);
    }

    output = (int8_t*) ((uintptr_t) output + output_increment);
  } while (--output_width != 0);
}
