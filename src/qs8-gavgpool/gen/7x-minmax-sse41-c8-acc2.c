// Auto-generated file. Do not edit!
//   Template: src/qs8-gavgpool/unipass-sse.c.in
//   Generator: tools/xngen
//
// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <smmintrin.h>

#include <xnnpack/gavgpool.h>


void xnn_qs8_gavgpool_minmax_ukernel_7x__sse41_c8_acc2(
    size_t rows,
    size_t channels,
    const int8_t* input,
    size_t input_stride,
    const int8_t* zero,
    int8_t* output,
    const union xnn_qs8_avgpool_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_DISABLE_TSAN
{
  assert(rows != 0);
  assert(rows <= 7);
  assert(channels != 0);

  const int8_t* i0 = input;
  const int8_t* i1 = (const int8_t*) ((uintptr_t) i0 + input_stride);
  if XNN_UNPREDICTABLE(rows < 2) {
    i1 = zero;
  }
  const int8_t* i2 = (const int8_t*) ((uintptr_t) i1 + input_stride);
  if XNN_UNPREDICTABLE(rows <= 2) {
    i2 = zero;
  }
  const int8_t* i3 = (const int8_t*) ((uintptr_t) i2 + input_stride);
  if XNN_UNPREDICTABLE(rows < 4) {
    i3 = zero;
  }
  const int8_t* i4 = (const int8_t*) ((uintptr_t) i3 + input_stride);
  if XNN_UNPREDICTABLE(rows <= 4) {
    i4 = zero;
  }
  const int8_t* i5 = (const int8_t*) ((uintptr_t) i4 + input_stride);
  if XNN_UNPREDICTABLE(rows < 6) {
    i5 = zero;
  }
  const int8_t* i6 = (const int8_t*) ((uintptr_t) i5 + input_stride);
  if XNN_UNPREDICTABLE(rows <= 6) {
    i6 = zero;
  }

  const __m128i vbias = _mm_load_si128((const __m128i*) params->sse2.bias);
  const __m128i vmultiplier = _mm_load_si128((const __m128i*) params->sse2.multiplier);
  const __m128i vrounding = _mm_load_si128((const __m128i*) params->sse2.rounding);
  const __m128i vshift = _mm_loadl_epi64((const __m128i*) params->sse2.shift);
  while (channels >= 8) {
    const __m128i vxi0x01234567 = _mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*) i0));
    i0 += 8;
    const __m128i vxi1x01234567 = _mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*) i1));
    i1 += 8;
    const __m128i vxi2x01234567 = _mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*) i2));
    i2 += 8;
    const __m128i vxi3x01234567 = _mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*) i3));
    i3 += 8;
    const __m128i vxi4x01234567 = _mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*) i4));
    i4 += 8;
    const __m128i vxi5x01234567 = _mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*) i5));
    i5 += 8;
    const __m128i vxi6x01234567 = _mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*) i6));
    i6 += 8;


    __m128i vacc0x01234567 = _mm_add_epi16(vxi0x01234567, vxi1x01234567);
    __m128i vacc1x01234567 = _mm_add_epi16(vxi2x01234567, vxi3x01234567);

    vacc0x01234567 = _mm_add_epi16(vacc0x01234567, vxi4x01234567);
    vacc1x01234567 = _mm_add_epi16(vacc1x01234567, vxi5x01234567);
    vacc0x01234567 = _mm_add_epi16(vacc0x01234567, vxi6x01234567);

    // Add up all accumulators to vacc0x01234567
    vacc0x01234567 = _mm_add_epi16(vacc0x01234567, vacc1x01234567);

    const __m128i vacc0123 = _mm_add_epi32(vbias, _mm_cvtepi16_epi32(vacc0x01234567));
    const __m128i vacc4567 = _mm_add_epi32(vbias, _mm_unpackhi_epi16(vacc0x01234567, _mm_cmpgt_epi16(_mm_setzero_si128(), vacc0x01234567)));

    const __m128i vabsacc0123 = _mm_abs_epi32(vacc0123);
    const __m128i vabsacc4567 = _mm_abs_epi32(vacc4567);

    const __m128i vabsacc13 = _mm_shuffle_epi32(vabsacc0123, _MM_SHUFFLE(3, 3, 1, 1));
    const __m128i vabsacc57 = _mm_shuffle_epi32(vabsacc4567, _MM_SHUFFLE(3, 3, 1, 1));

    const __m128i vabsprod02 = _mm_mul_epu32(vabsacc0123, vmultiplier);
    const __m128i vabsprod13 = _mm_mul_epu32(vabsacc13, vmultiplier);
    const __m128i vabsprod46 = _mm_mul_epu32(vabsacc4567, vmultiplier);
    const __m128i vabsprod57 = _mm_mul_epu32(vabsacc57, vmultiplier);

    const __m128i vabsout02 = _mm_srl_epi64(_mm_add_epi64(vabsprod02, vrounding), vshift);
    const __m128i vabsout13 = _mm_srl_epi64(_mm_add_epi64(vabsprod13, vrounding), vshift);
    const __m128i vabsout46 = _mm_srl_epi64(_mm_add_epi64(vabsprod46, vrounding), vshift);
    const __m128i vabsout57 = _mm_srl_epi64(_mm_add_epi64(vabsprod57, vrounding), vshift);

    const __m128i vabsout0123 = _mm_blend_epi16(vabsout02, _mm_shuffle_epi32(vabsout13, _MM_SHUFFLE(2, 2, 0, 0)), 0xCC);
    const __m128i vabsout4567 = _mm_blend_epi16(vabsout46, _mm_shuffle_epi32(vabsout57, _MM_SHUFFLE(2, 2, 0, 0)), 0xCC);

    const __m128i vout0123 = _mm_sign_epi32(vabsout0123, vacc0123);
    const __m128i vout4567 = _mm_sign_epi32(vabsout4567, vacc4567);

    const __m128i voutput_zero_point = _mm_load_si128((const __m128i*) params->sse2.output_zero_point);
    __m128i vout01234567 = _mm_adds_epi16(_mm_packs_epi32(vout0123, vout4567), voutput_zero_point);

    const __m128i voutput_min = _mm_load_si128((const __m128i*) params->sse2.output_min);
    const __m128i voutput_max = _mm_load_si128((const __m128i*) params->sse2.output_max);
    vout01234567 = _mm_min_epi16(_mm_max_epi16(vout01234567, voutput_min), voutput_max);

    __m128i vout0123456701234567 = _mm_packs_epi16(vout01234567, vout01234567);

    _mm_storel_epi64((__m128i*) output, vout0123456701234567);
    output += 8;

    channels -= 8;
  }
  if XNN_UNLIKELY(channels != 0) {
    {
      const __m128i vxi0x01234567 = _mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*) i0));
      i0 += 8;
      const __m128i vxi1x01234567 = _mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*) i1));
      i1 += 8;
      const __m128i vxi2x01234567 = _mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*) i2));
      i2 += 8;
      const __m128i vxi3x01234567 = _mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*) i3));
      i3 += 8;
      const __m128i vxi4x01234567 = _mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*) i4));
      i4 += 8;
      const __m128i vxi5x01234567 = _mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*) i5));
      i5 += 8;
      const __m128i vxi6x01234567 = _mm_cvtepi8_epi16(_mm_loadl_epi64((const __m128i*) i6));
      i6 += 8;


      __m128i vacc0x01234567 = _mm_add_epi16(vxi0x01234567, vxi1x01234567);
      __m128i vacc1x01234567 = _mm_add_epi16(vxi2x01234567, vxi3x01234567);

      vacc0x01234567 = _mm_add_epi16(vacc0x01234567, vxi4x01234567);
      vacc1x01234567 = _mm_add_epi16(vacc1x01234567, vxi5x01234567);
      vacc0x01234567 = _mm_add_epi16(vacc0x01234567, vxi6x01234567);

      // Add up all accumulators to vacc0x01234567
      vacc0x01234567 = _mm_add_epi16(vacc0x01234567, vacc1x01234567);

      const __m128i vacc0123 = _mm_add_epi32(vbias, _mm_cvtepi16_epi32(vacc0x01234567));
      const __m128i vacc4567 = _mm_add_epi32(vbias, _mm_unpackhi_epi16(vacc0x01234567, _mm_cmpgt_epi16(_mm_setzero_si128(), vacc0x01234567)));

      const __m128i vabsacc0123 = _mm_abs_epi32(vacc0123);
      const __m128i vabsacc4567 = _mm_abs_epi32(vacc4567);

      const __m128i vabsacc13 = _mm_shuffle_epi32(vabsacc0123, _MM_SHUFFLE(3, 3, 1, 1));
      const __m128i vabsacc57 = _mm_shuffle_epi32(vabsacc4567, _MM_SHUFFLE(3, 3, 1, 1));

      const __m128i vabsprod02 = _mm_mul_epu32(vabsacc0123, vmultiplier);
      const __m128i vabsprod13 = _mm_mul_epu32(vabsacc13, vmultiplier);
      const __m128i vabsprod46 = _mm_mul_epu32(vabsacc4567, vmultiplier);
      const __m128i vabsprod57 = _mm_mul_epu32(vabsacc57, vmultiplier);

      const __m128i vabsout02 = _mm_srl_epi64(_mm_add_epi64(vabsprod02, vrounding), vshift);
      const __m128i vabsout13 = _mm_srl_epi64(_mm_add_epi64(vabsprod13, vrounding), vshift);
      const __m128i vabsout46 = _mm_srl_epi64(_mm_add_epi64(vabsprod46, vrounding), vshift);
      const __m128i vabsout57 = _mm_srl_epi64(_mm_add_epi64(vabsprod57, vrounding), vshift);

      const __m128i vabsout0123 = _mm_blend_epi16(vabsout02, _mm_shuffle_epi32(vabsout13, _MM_SHUFFLE(2, 2, 0, 0)), 0xCC);
      const __m128i vabsout4567 = _mm_blend_epi16(vabsout46, _mm_shuffle_epi32(vabsout57, _MM_SHUFFLE(2, 2, 0, 0)), 0xCC);

      const __m128i vout0123 = _mm_sign_epi32(vabsout0123, vacc0123);
      const __m128i vout4567 = _mm_sign_epi32(vabsout4567, vacc4567);

      const __m128i voutput_zero_point = _mm_load_si128((const __m128i*) params->sse2.output_zero_point);
      __m128i vout01234567 = _mm_adds_epi16(_mm_packs_epi32(vout0123, vout4567), voutput_zero_point);

      const __m128i voutput_min = _mm_load_si128((const __m128i*) params->sse2.output_min);
      const __m128i voutput_max = _mm_load_si128((const __m128i*) params->sse2.output_max);
      vout01234567 = _mm_min_epi16(_mm_max_epi16(vout01234567, voutput_min), voutput_max);

      __m128i vout0123456701234567 = _mm_packs_epi16(vout01234567, vout01234567);

      if (channels & 4) {
        *((uint32_t*) output) = (uint32_t) _mm_cvtsi128_si32(vout0123456701234567);
        vout0123456701234567 = _mm_srli_epi64(vout0123456701234567, 32);
        output += 4;
      }
      if (channels & 2) {
        *((uint16_t*) output) = (uint16_t) _mm_extract_epi16(vout0123456701234567, 0);
        vout0123456701234567 = _mm_srli_epi32(vout0123456701234567, 16);
        output += 2;
      }
      if (channels & 1) {
        *output = (int8_t) _mm_extract_epi8(vout0123456701234567, 0);
      }
    }
  }
}
