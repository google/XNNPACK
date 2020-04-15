// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <emmintrin.h>

#include <xnnpack/avgpool.h>


void xnn_q8_avgpool_minmax_ukernel_9p8x__sse2_c8(
    size_t output_pixels,
    size_t kernel_elements,
    size_t channels,
    const uint8_t** input,
    size_t input_offset,
    const uint8_t* zero,
    int32_t* buffer,
    uint8_t* output,
    size_t input_increment,
    size_t output_increment,
    const union xnn_q8_avgpool_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(output_pixels != 0);
  assert(kernel_elements > 9);
  assert(channels != 0);

  const __m128i vbias = _mm_load_si128((const __m128i*) &params->sse2.bias);
  const __m128i vzero = _mm_setzero_si128();
  const __m128i vmultiplier = _mm_load_si128((const __m128i*) params->sse2.multiplier);
  const __m128i vrounding = _mm_load_si128((const __m128i*) params->sse2.rounding);
  const __m128i vright_shift = _mm_loadl_epi64((const __m128i*) params->sse2.right_shift);

  do {
    {
      const uint8_t* i0 = *input++;
      assert(i0 != NULL);
      if XNN_UNPREDICTABLE(i0 != zero) {
        i0 = (const uint8_t*) ((uintptr_t) i0 + input_offset);
      }
      const uint8_t* i1 = *input++;
      assert(i1 != NULL);
      if XNN_UNPREDICTABLE(i1 != zero) {
        i1 = (const uint8_t*) ((uintptr_t) i1 + input_offset);
      }
      const uint8_t* i2 = *input++;
      assert(i2 != NULL);
      if XNN_UNPREDICTABLE(i2 != zero) {
        i2 = (const uint8_t*) ((uintptr_t) i2 + input_offset);
      }
      const uint8_t* i3 = *input++;
      assert(i3 != NULL);
      if XNN_UNPREDICTABLE(i3 != zero) {
        i3 = (const uint8_t*) ((uintptr_t) i3 + input_offset);
      }
      const uint8_t* i4 = *input++;
      assert(i4 != NULL);
      if XNN_UNPREDICTABLE(i4 != zero) {
        i4 = (const uint8_t*) ((uintptr_t) i4 + input_offset);
      }
      const uint8_t* i5 = *input++;
      assert(i5 != NULL);
      if XNN_UNPREDICTABLE(i5 != zero) {
        i5 = (const uint8_t*) ((uintptr_t) i5 + input_offset);
      }
      const uint8_t* i6 = *input++;
      assert(i6 != NULL);
      if XNN_UNPREDICTABLE(i6 != zero) {
        i6 = (const uint8_t*) ((uintptr_t) i6 + input_offset);
      }
      const uint8_t* i7 = *input++;
      assert(i7 != NULL);
      if XNN_UNPREDICTABLE(i7 != zero) {
        i7 = (const uint8_t*) ((uintptr_t) i7 + input_offset);
      }
      const uint8_t* i8 = *input++;
      assert(i8 != NULL);
      if XNN_UNPREDICTABLE(i8 != zero) {
        i8 = (const uint8_t*) ((uintptr_t) i8 + input_offset);
      }

      int32_t* b = buffer;
      for (size_t c = 0; c < channels; c += 8) {
        const __m128i vi0 = _mm_loadl_epi64((const __m128i*) i0); i0 += 8;
        const __m128i vi1 = _mm_loadl_epi64((const __m128i*) i1); i1 += 8;
        const __m128i vi2 = _mm_loadl_epi64((const __m128i*) i2); i2 += 8;
        const __m128i vi3 = _mm_loadl_epi64((const __m128i*) i3); i3 += 8;
        const __m128i vi4 = _mm_loadl_epi64((const __m128i*) i4); i4 += 8;
        const __m128i vi5 = _mm_loadl_epi64((const __m128i*) i5); i5 += 8;
        const __m128i vi6 = _mm_loadl_epi64((const __m128i*) i6); i6 += 8;
        const __m128i vi7 = _mm_loadl_epi64((const __m128i*) i7); i7 += 8;
        const __m128i vi8 = _mm_loadl_epi64((const __m128i*) i8); i8 += 8;

        const __m128i vxi0 = _mm_unpacklo_epi8(vi0, vzero);
        const __m128i vxi1 = _mm_unpacklo_epi8(vi1, vzero);
        const __m128i vxi2 = _mm_unpacklo_epi8(vi2, vzero);
        const __m128i vxi3 = _mm_unpacklo_epi8(vi3, vzero);
        const __m128i vxi4 = _mm_unpacklo_epi8(vi4, vzero);
        const __m128i vxi5 = _mm_unpacklo_epi8(vi5, vzero);
        const __m128i vxi6 = _mm_unpacklo_epi8(vi6, vzero);
        const __m128i vxi7 = _mm_unpacklo_epi8(vi7, vzero);
        const __m128i vxi8 = _mm_unpacklo_epi8(vi8, vzero);

        const __m128i vsum018 = _mm_add_epi16(_mm_add_epi16(vxi0, vxi1), vxi8);
        const __m128i vsum23 = _mm_add_epi16(vxi2, vxi3);
        const __m128i vsum45 = _mm_add_epi16(vxi4, vxi5);
        const __m128i vsum67 = _mm_add_epi16(vxi6, vxi7);

        const __m128i vsum2345 = _mm_add_epi16(vsum23, vsum45);
        const __m128i vsum01678 = _mm_add_epi16(vsum018, vsum67);
        const __m128i vsum = _mm_add_epi16(vsum2345, vsum01678);

        const __m128i vacc_lo = _mm_add_epi32(vbias, _mm_unpacklo_epi16(vsum, vzero));
        const __m128i vacc_hi = _mm_add_epi32(vbias, _mm_unpackhi_epi16(vsum, vzero));

        _mm_store_si128((__m128i*) b, vacc_lo);
        _mm_store_si128((__m128i*) b + 1, vacc_hi);
        b += 8;
      }
    }

    size_t k = kernel_elements;
    for (k -= 9; k > 8; k -= 8) {
      const uint8_t* i0 = *input++;
      assert(i0 != NULL);
      if XNN_UNPREDICTABLE(i0 != zero) {
        i0 = (const uint8_t*) ((uintptr_t) i0 + input_offset);
      }
      const uint8_t* i1 = *input++;
      assert(i1 != NULL);
      if XNN_UNPREDICTABLE(i1 != zero) {
        i1 = (const uint8_t*) ((uintptr_t) i1 + input_offset);
      }
      const uint8_t* i2 = *input++;
      assert(i2 != NULL);
      if XNN_UNPREDICTABLE(i2 != zero) {
        i2 = (const uint8_t*) ((uintptr_t) i2 + input_offset);
      }
      const uint8_t* i3 = *input++;
      assert(i3 != NULL);
      if XNN_UNPREDICTABLE(i3 != zero) {
        i3 = (const uint8_t*) ((uintptr_t) i3 + input_offset);
      }
      const uint8_t* i4 = *input++;
      assert(i4 != NULL);
      if XNN_UNPREDICTABLE(i4 != zero) {
        i4 = (const uint8_t*) ((uintptr_t) i4 + input_offset);
      }
      const uint8_t* i5 = *input++;
      assert(i5 != NULL);
      if XNN_UNPREDICTABLE(i5 != zero) {
        i5 = (const uint8_t*) ((uintptr_t) i5 + input_offset);
      }
      const uint8_t* i6 = *input++;
      assert(i6 != NULL);
      if XNN_UNPREDICTABLE(i6 != zero) {
        i6 = (const uint8_t*) ((uintptr_t) i6 + input_offset);
      }
      const uint8_t* i7 = *input++;
      assert(i7 != NULL);
      if XNN_UNPREDICTABLE(i7 != zero) {
        i7 = (const uint8_t*) ((uintptr_t) i7 + input_offset);
      }

      int32_t* b = buffer;
      for (size_t c = 0; c < channels; c += 8) {
        const __m128i vi0 = _mm_loadl_epi64((const __m128i*) i0); i0 += 8;
        const __m128i vi1 = _mm_loadl_epi64((const __m128i*) i1); i1 += 8;
        const __m128i vi2 = _mm_loadl_epi64((const __m128i*) i2); i2 += 8;
        const __m128i vi3 = _mm_loadl_epi64((const __m128i*) i3); i3 += 8;
        const __m128i vi4 = _mm_loadl_epi64((const __m128i*) i4); i4 += 8;
        const __m128i vi5 = _mm_loadl_epi64((const __m128i*) i5); i5 += 8;
        const __m128i vi6 = _mm_loadl_epi64((const __m128i*) i6); i6 += 8;
        const __m128i vi7 = _mm_loadl_epi64((const __m128i*) i7); i7 += 8;
        __m128i vacc_lo = _mm_load_si128((const __m128i*) b);
        __m128i vacc_hi = _mm_load_si128((const __m128i*) b + 1);

        const __m128i vxi0 = _mm_unpacklo_epi8(vi0, vzero);
        const __m128i vxi1 = _mm_unpacklo_epi8(vi1, vzero);
        const __m128i vxi2 = _mm_unpacklo_epi8(vi2, vzero);
        const __m128i vxi3 = _mm_unpacklo_epi8(vi3, vzero);
        const __m128i vxi4 = _mm_unpacklo_epi8(vi4, vzero);
        const __m128i vxi5 = _mm_unpacklo_epi8(vi5, vzero);
        const __m128i vxi6 = _mm_unpacklo_epi8(vi6, vzero);
        const __m128i vxi7 = _mm_unpacklo_epi8(vi7, vzero);

        const __m128i vsum01 = _mm_add_epi16(vxi0, vxi1);
        const __m128i vsum23 = _mm_add_epi16(vxi2, vxi3);
        const __m128i vsum45 = _mm_add_epi16(vxi4, vxi5);
        const __m128i vsum67 = _mm_add_epi16(vxi6, vxi7);

        const __m128i vsum0123 = _mm_add_epi16(vsum01, vsum23);
        const __m128i vsum4567 = _mm_add_epi16(vsum45, vsum67);
        const __m128i vsum = _mm_add_epi16(vsum0123, vsum4567);

        vacc_lo = _mm_add_epi32(vacc_lo, _mm_unpacklo_epi16(vsum, vzero));
        vacc_hi = _mm_add_epi32(vacc_hi, _mm_unpackhi_epi16(vsum, vzero));

        _mm_store_si128((__m128i*) b, vacc_lo);
        _mm_store_si128((__m128i*) b + 1, vacc_hi);
        b += 8;
      }
    }

    {
      const uint8_t* i0 = input[0];
      assert(i0 != NULL);
      const uint8_t* i1 = input[1];
      const uint8_t* i2 = input[2];
      const uint8_t* i3 = input[3];
      const uint8_t* i4 = input[4];
      const uint8_t* i5 = input[5];
      const uint8_t* i6 = input[6];
      const uint8_t* i7 = input[7];
      input = (const uint8_t**) ((uintptr_t) input + input_increment);
      if (k < 2) {
        i1 = zero;
      }
      assert(i1 != NULL);
      if (k <= 2) {
        i2 = zero;
      }
      assert(i2 != NULL);
      if (k < 4) {
        i3 = zero;
      }
      assert(i3 != NULL);
      if (k <= 4) {
        i4 = zero;
      }
      assert(i4 != NULL);
      if (k < 6) {
        i5 = zero;
      }
      assert(i5 != NULL);
      if (k <= 6) {
        i6 = zero;
      }
      assert(i6 != NULL);
      if (k < 8) {
        i7 = zero;
      }
      assert(i7 != NULL);
      if XNN_UNPREDICTABLE(i0 != zero) {
        i0 = (const uint8_t*) ((uintptr_t) i0 + input_offset);
      }
      if XNN_UNPREDICTABLE(i1 != zero) {
        i1 = (const uint8_t*) ((uintptr_t) i1 + input_offset);
      }
      if XNN_UNPREDICTABLE(i2 != zero) {
        i2 = (const uint8_t*) ((uintptr_t) i2 + input_offset);
      }
      if XNN_UNPREDICTABLE(i3 != zero) {
        i3 = (const uint8_t*) ((uintptr_t) i3 + input_offset);
      }
      if XNN_UNPREDICTABLE(i4 != zero) {
        i4 = (const uint8_t*) ((uintptr_t) i4 + input_offset);
      }
      if XNN_UNPREDICTABLE(i5 != zero) {
        i5 = (const uint8_t*) ((uintptr_t) i5 + input_offset);
      }
      if XNN_UNPREDICTABLE(i6 != zero) {
        i6 = (const uint8_t*) ((uintptr_t) i6 + input_offset);
      }
      if XNN_UNPREDICTABLE(i7 != zero) {
        i7 = (const uint8_t*) ((uintptr_t) i7 + input_offset);
      }

      size_t c = channels;
      int32_t* b = buffer;
      while (c >= 8) {
        const __m128i vi0 = _mm_loadl_epi64((const __m128i*) i0); i0 += 8;
        const __m128i vi1 = _mm_loadl_epi64((const __m128i*) i1); i1 += 8;
        const __m128i vi2 = _mm_loadl_epi64((const __m128i*) i2); i2 += 8;
        const __m128i vi3 = _mm_loadl_epi64((const __m128i*) i3); i3 += 8;
        const __m128i vi4 = _mm_loadl_epi64((const __m128i*) i4); i4 += 8;
        const __m128i vi5 = _mm_loadl_epi64((const __m128i*) i5); i5 += 8;
        const __m128i vi6 = _mm_loadl_epi64((const __m128i*) i6); i6 += 8;
        const __m128i vi7 = _mm_loadl_epi64((const __m128i*) i7); i7 += 8;
        __m128i vacc_lo = _mm_load_si128((const __m128i*) b);
        __m128i vacc_hi = _mm_load_si128((const __m128i*) b + 1);
        b += 8;

        const __m128i vxi0 = _mm_unpacklo_epi8(vi0, vzero);
        const __m128i vxi1 = _mm_unpacklo_epi8(vi1, vzero);
        const __m128i vxi2 = _mm_unpacklo_epi8(vi2, vzero);
        const __m128i vxi3 = _mm_unpacklo_epi8(vi3, vzero);
        const __m128i vxi4 = _mm_unpacklo_epi8(vi4, vzero);
        const __m128i vxi5 = _mm_unpacklo_epi8(vi5, vzero);
        const __m128i vxi6 = _mm_unpacklo_epi8(vi6, vzero);
        const __m128i vxi7 = _mm_unpacklo_epi8(vi7, vzero);

        const __m128i vsum01 = _mm_add_epi16(vxi0, vxi1);
        const __m128i vsum23 = _mm_add_epi16(vxi2, vxi3);
        const __m128i vsum45 = _mm_add_epi16(vxi4, vxi5);
        const __m128i vsum67 = _mm_add_epi16(vxi6, vxi7);

        const __m128i vsum0123 = _mm_add_epi16(vsum01, vsum23);
        const __m128i vsum4567 = _mm_add_epi16(vsum45, vsum67);
        const __m128i vsum = _mm_add_epi16(vsum0123, vsum4567);

        vacc_lo = _mm_add_epi32(vacc_lo, _mm_unpacklo_epi16(vsum, vzero));
        vacc_hi = _mm_add_epi32(vacc_hi, _mm_unpackhi_epi16(vsum, vzero));

        const __m128i vneg_mask_lo = _mm_cmpgt_epi32(_mm_setzero_si128(), vacc_lo);
        const __m128i vneg_mask_hi = _mm_cmpgt_epi32(_mm_setzero_si128(), vacc_hi);

        const __m128i vabs_lo0123 = _mm_sub_epi32(_mm_xor_si128(vacc_lo, vneg_mask_lo), vneg_mask_lo);
        const __m128i vabs_hi0123 = _mm_sub_epi32(_mm_xor_si128(vacc_hi, vneg_mask_hi), vneg_mask_hi);

        const __m128i vabs_lo1032 = _mm_shuffle_epi32(vabs_lo0123, _MM_SHUFFLE(2, 3, 0, 1));
        const __m128i vabs_hi1032 = _mm_shuffle_epi32(vabs_hi0123, _MM_SHUFFLE(2, 3, 0, 1));

        const __m128i vabsmul_lo02 = _mm_mul_epu32(vabs_lo0123, vmultiplier);
        const __m128i vabsmul_hi02 = _mm_mul_epu32(vabs_hi0123, vmultiplier);

        const __m128i vabsmul_lo13 = _mm_mul_epu32(vabs_lo1032, vmultiplier);
        const __m128i vabsmul_hi13 = _mm_mul_epu32(vabs_hi1032, vmultiplier);

        const __m128i vabs_scaled_lo02 = _mm_srl_epi64(_mm_add_epi64(vabsmul_lo02, vrounding), vright_shift);
        const __m128i vabs_scaled_lo13 = _mm_srl_epi64(_mm_add_epi64(vabsmul_lo13, vrounding), vright_shift);
        const __m128i vabs_scaled_hi02 = _mm_srl_epi64(_mm_add_epi64(vabsmul_hi02, vrounding), vright_shift);
        const __m128i vabs_scaled_hi13 = _mm_srl_epi64(_mm_add_epi64(vabsmul_hi13, vrounding), vright_shift);

        const __m128i vabs_scaled_lo0213 = _mm_castps_si128(
            _mm_shuffle_ps(_mm_castsi128_ps(vabs_scaled_lo02), _mm_castsi128_ps(vabs_scaled_lo13), _MM_SHUFFLE(2, 0, 2, 0)));
        const __m128i vabs_scaled_hi0213 = _mm_castps_si128(
            _mm_shuffle_ps(_mm_castsi128_ps(vabs_scaled_hi02), _mm_castsi128_ps(vabs_scaled_hi13), _MM_SHUFFLE(2, 0, 2, 0)));

        const __m128i vabs_scaled_lo = _mm_shuffle_epi32(vabs_scaled_lo0213, _MM_SHUFFLE(3, 1, 2, 0));
        const __m128i vabs_scaled_hi = _mm_shuffle_epi32(vabs_scaled_hi0213, _MM_SHUFFLE(3, 1, 2, 0));

        const __m128i vscaled_lo = _mm_sub_epi32(_mm_xor_si128(vabs_scaled_lo, vneg_mask_lo), vneg_mask_lo);
        const __m128i vscaled_hi = _mm_sub_epi32(_mm_xor_si128(vabs_scaled_hi, vneg_mask_hi), vneg_mask_hi);

        __m128i vout = _mm_packs_epi32(vscaled_lo, vscaled_hi);
        vout = _mm_adds_epi16(vout, _mm_load_si128((const __m128i*) &params->sse2.output_zero_point));
        vout = _mm_packus_epi16(vout, vout);
        vout = _mm_min_epu8(vout, _mm_load_si128((const __m128i*) &params->sse2.output_max));
        vout = _mm_max_epu8(vout, _mm_load_si128((const __m128i*) &params->sse2.output_min));

        _mm_storel_epi64((__m128i*) output, vout);
        output += 8;

        c -= 8;
      }
      if (c != 0) {
        const __m128i vi0 = _mm_loadl_epi64((const __m128i*) i0);
        const __m128i vi1 = _mm_loadl_epi64((const __m128i*) i1);
        const __m128i vi2 = _mm_loadl_epi64((const __m128i*) i2);
        const __m128i vi3 = _mm_loadl_epi64((const __m128i*) i3);
        const __m128i vi4 = _mm_loadl_epi64((const __m128i*) i4);
        const __m128i vi5 = _mm_loadl_epi64((const __m128i*) i5);
        const __m128i vi6 = _mm_loadl_epi64((const __m128i*) i6);
        const __m128i vi7 = _mm_loadl_epi64((const __m128i*) i7);
        __m128i vacc_lo = _mm_load_si128((const __m128i*) b);
        __m128i vacc_hi = _mm_load_si128((const __m128i*) b + 1);

        const __m128i vxi0 = _mm_unpacklo_epi8(vi0, vzero);
        const __m128i vxi1 = _mm_unpacklo_epi8(vi1, vzero);
        const __m128i vxi2 = _mm_unpacklo_epi8(vi2, vzero);
        const __m128i vxi3 = _mm_unpacklo_epi8(vi3, vzero);
        const __m128i vxi4 = _mm_unpacklo_epi8(vi4, vzero);
        const __m128i vxi5 = _mm_unpacklo_epi8(vi5, vzero);
        const __m128i vxi6 = _mm_unpacklo_epi8(vi6, vzero);
        const __m128i vxi7 = _mm_unpacklo_epi8(vi7, vzero);

        const __m128i vsum01 = _mm_add_epi16(vxi0, vxi1);
        const __m128i vsum23 = _mm_add_epi16(vxi2, vxi3);
        const __m128i vsum45 = _mm_add_epi16(vxi4, vxi5);
        const __m128i vsum67 = _mm_add_epi16(vxi6, vxi7);

        const __m128i vsum0123 = _mm_add_epi16(vsum01, vsum23);
        const __m128i vsum4567 = _mm_add_epi16(vsum45, vsum67);
        const __m128i vsum = _mm_add_epi16(vsum0123, vsum4567);

        vacc_lo = _mm_add_epi32(vacc_lo, _mm_unpacklo_epi16(vsum, vzero));
        vacc_hi = _mm_add_epi32(vacc_hi, _mm_unpackhi_epi16(vsum, vzero));

        const __m128i vneg_mask_lo = _mm_cmpgt_epi32(_mm_setzero_si128(), vacc_lo);
        const __m128i vneg_mask_hi = _mm_cmpgt_epi32(_mm_setzero_si128(), vacc_hi);

        const __m128i vabs_lo0123 = _mm_sub_epi32(_mm_xor_si128(vacc_lo, vneg_mask_lo), vneg_mask_lo);
        const __m128i vabs_hi0123 = _mm_sub_epi32(_mm_xor_si128(vacc_hi, vneg_mask_hi), vneg_mask_hi);

        const __m128i vabs_lo1032 = _mm_shuffle_epi32(vabs_lo0123, _MM_SHUFFLE(2, 3, 0, 1));
        const __m128i vabs_hi1032 = _mm_shuffle_epi32(vabs_hi0123, _MM_SHUFFLE(2, 3, 0, 1));

        const __m128i vabsmul_lo02 = _mm_mul_epu32(vabs_lo0123, vmultiplier);
        const __m128i vabsmul_hi02 = _mm_mul_epu32(vabs_hi0123, vmultiplier);

        const __m128i vabsmul_lo13 = _mm_mul_epu32(vabs_lo1032, vmultiplier);
        const __m128i vabsmul_hi13 = _mm_mul_epu32(vabs_hi1032, vmultiplier);

        const __m128i vabs_scaled_lo02 = _mm_srl_epi64(_mm_add_epi64(vabsmul_lo02, vrounding), vright_shift);
        const __m128i vabs_scaled_lo13 = _mm_srl_epi64(_mm_add_epi64(vabsmul_lo13, vrounding), vright_shift);
        const __m128i vabs_scaled_hi02 = _mm_srl_epi64(_mm_add_epi64(vabsmul_hi02, vrounding), vright_shift);
        const __m128i vabs_scaled_hi13 = _mm_srl_epi64(_mm_add_epi64(vabsmul_hi13, vrounding), vright_shift);

        const __m128i vabs_scaled_lo0213 = _mm_castps_si128(
            _mm_shuffle_ps(_mm_castsi128_ps(vabs_scaled_lo02), _mm_castsi128_ps(vabs_scaled_lo13), _MM_SHUFFLE(2, 0, 2, 0)));
        const __m128i vabs_scaled_hi0213 = _mm_castps_si128(
            _mm_shuffle_ps(_mm_castsi128_ps(vabs_scaled_hi02), _mm_castsi128_ps(vabs_scaled_hi13), _MM_SHUFFLE(2, 0, 2, 0)));

        const __m128i vabs_scaled_lo = _mm_shuffle_epi32(vabs_scaled_lo0213, _MM_SHUFFLE(3, 1, 2, 0));
        const __m128i vabs_scaled_hi = _mm_shuffle_epi32(vabs_scaled_hi0213, _MM_SHUFFLE(3, 1, 2, 0));

        const __m128i vscaled_lo = _mm_sub_epi32(_mm_xor_si128(vabs_scaled_lo, vneg_mask_lo), vneg_mask_lo);
        const __m128i vscaled_hi = _mm_sub_epi32(_mm_xor_si128(vabs_scaled_hi, vneg_mask_hi), vneg_mask_hi);

        __m128i vout = _mm_packs_epi32(vscaled_lo, vscaled_hi);
        vout = _mm_adds_epi16(vout, _mm_load_si128((const __m128i*) &params->sse2.output_zero_point));
        vout = _mm_packus_epi16(vout, vout);
        vout = _mm_min_epu8(vout, _mm_load_si128((const __m128i*) &params->sse2.output_max));
        vout = _mm_max_epu8(vout, _mm_load_si128((const __m128i*) &params->sse2.output_min));

        if (c & 4) {
          *((uint32_t*) output) = (uint32_t) _mm_cvtsi128_si32(vout);
          output += 4;
          vout = _mm_srli_epi64(vout, 32);
        }
        if (c & 2) {
          *((uint16_t*) output) = (uint16_t) _mm_extract_epi16(vout, 0);
          output += 2;
          vout = _mm_srli_epi32(vout, 16);
        }
        if (c & 1) {
          *((uint8_t*) output) = (uint8_t) _mm_cvtsi128_si32(vout);
          output += 1;
        }
      }
    }
    output = (uint8_t*) ((uintptr_t) output + output_increment);
  } while (--output_pixels != 0);
}
