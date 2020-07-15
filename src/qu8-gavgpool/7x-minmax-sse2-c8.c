// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <emmintrin.h>

#include <xnnpack/gavgpool.h>


void xnn_qu8_gavgpool_minmax_ukernel_7x__sse2_c8(
    size_t rows,
    size_t channels,
    const uint8_t* input,
    size_t input_stride,
    const uint8_t* zero,
    uint8_t* output,
    const union xnn_qu8_avgpool_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_DISABLE_TSAN
{
  assert(rows != 0);
  assert(rows <= 7);
  assert(channels != 0);

  const uint8_t* i0 = input;
  const uint8_t* i1 = (const uint8_t*) ((uintptr_t) i0 + input_stride);
  if (rows < 2) {
    i1 = zero;
  }
  const uint8_t* i2 = (const uint8_t*) ((uintptr_t) i1 + input_stride);
  if (rows <= 2) {
    i2 = zero;
  }
  const uint8_t* i3 = (const uint8_t*) ((uintptr_t) i2 + input_stride);
  if (rows < 4) {
    i3 = zero;
  }
  const uint8_t* i4 = (const uint8_t*) ((uintptr_t) i3 + input_stride);
  if (rows <= 4) {
    i4 = zero;
  }
  const uint8_t* i5 = (const uint8_t*) ((uintptr_t) i4 + input_stride);
  if (rows < 6) {
    i5 = zero;
  }
  const uint8_t* i6 = (const uint8_t*) ((uintptr_t) i5 + input_stride);
  if (rows <= 6) {
    i6 = zero;
  }

  const __m128i vbias = _mm_load_si128((const __m128i*) &params->sse2.bias);
  const __m128i vzero = _mm_setzero_si128();
  const __m128i vmultiplier = _mm_load_si128((const __m128i*) params->sse2.multiplier);
  const __m128i vrounding = _mm_load_si128((const __m128i*) params->sse2.rounding);
  const __m128i vright_shift = _mm_loadl_epi64((const __m128i*) params->sse2.right_shift);

  while (channels >= 8) {
    const __m128i vi0 = _mm_loadl_epi64((const __m128i*) i0); i0 += 8;
    const __m128i vi1 = _mm_loadl_epi64((const __m128i*) i1); i1 += 8;
    const __m128i vi2 = _mm_loadl_epi64((const __m128i*) i2); i2 += 8;
    const __m128i vi3 = _mm_loadl_epi64((const __m128i*) i3); i3 += 8;
    const __m128i vi4 = _mm_loadl_epi64((const __m128i*) i4); i4 += 8;
    const __m128i vi5 = _mm_loadl_epi64((const __m128i*) i5); i5 += 8;
    const __m128i vi6 = _mm_loadl_epi64((const __m128i*) i6); i6 += 8;

    const __m128i vxi0 = _mm_unpacklo_epi8(vi0, vzero);
    const __m128i vxi1 = _mm_unpacklo_epi8(vi1, vzero);
    const __m128i vxi2 = _mm_unpacklo_epi8(vi2, vzero);
    const __m128i vxi3 = _mm_unpacklo_epi8(vi3, vzero);
    const __m128i vxi4 = _mm_unpacklo_epi8(vi4, vzero);
    const __m128i vxi5 = _mm_unpacklo_epi8(vi5, vzero);
    const __m128i vxi6 = _mm_unpacklo_epi8(vi6, vzero);

    const __m128i vsum01 = _mm_add_epi16(vxi0, vxi1);
    const __m128i vsum23 = _mm_add_epi16(vxi2, vxi3);
    const __m128i vsum45 = _mm_add_epi16(vxi4, vxi5);

    const __m128i vsum016 = _mm_add_epi16(vsum01, vxi6);
    const __m128i vsum2345 = _mm_add_epi16(vsum23, vsum45);
    const __m128i vsum = _mm_add_epi16(vsum016, vsum2345);

    __m128i vacc_lo = _mm_add_epi32(vbias, _mm_unpacklo_epi16(vsum, vzero));
    __m128i vacc_hi = _mm_add_epi32(vbias, _mm_unpackhi_epi16(vsum, vzero));

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
    vout = _mm_adds_epi16(vout, _mm_load_si128((const __m128i*) params->sse2.output_zero_point));
    vout = _mm_packus_epi16(vout, vout);
    vout = _mm_min_epu8(vout, _mm_load_si128((const __m128i*) params->sse2.output_max));
    vout = _mm_max_epu8(vout, _mm_load_si128((const __m128i*) params->sse2.output_min));

    _mm_storel_epi64((__m128i*) output, vout); output += 8;

    channels -= 8;
  }
  if (channels != 0) {
    const __m128i vi0 = _mm_loadl_epi64((const __m128i*) i0);
    const __m128i vi1 = _mm_loadl_epi64((const __m128i*) i1);
    const __m128i vi2 = _mm_loadl_epi64((const __m128i*) i2);
    const __m128i vi3 = _mm_loadl_epi64((const __m128i*) i3);
    const __m128i vi4 = _mm_loadl_epi64((const __m128i*) i4);
    const __m128i vi5 = _mm_loadl_epi64((const __m128i*) i5);
    const __m128i vi6 = _mm_loadl_epi64((const __m128i*) i6);

    const __m128i vxi0 = _mm_unpacklo_epi8(vi0, vzero);
    const __m128i vxi1 = _mm_unpacklo_epi8(vi1, vzero);
    const __m128i vxi2 = _mm_unpacklo_epi8(vi2, vzero);
    const __m128i vxi3 = _mm_unpacklo_epi8(vi3, vzero);
    const __m128i vxi4 = _mm_unpacklo_epi8(vi4, vzero);
    const __m128i vxi5 = _mm_unpacklo_epi8(vi5, vzero);
    const __m128i vxi6 = _mm_unpacklo_epi8(vi6, vzero);

    const __m128i vsum01 = _mm_add_epi16(vxi0, vxi1);
    const __m128i vsum23 = _mm_add_epi16(vxi2, vxi3);
    const __m128i vsum45 = _mm_add_epi16(vxi4, vxi5);

    const __m128i vsum016 = _mm_add_epi16(vsum01, vxi6);
    const __m128i vsum2345 = _mm_add_epi16(vsum23, vsum45);
    const __m128i vsum = _mm_add_epi16(vsum016, vsum2345);

    __m128i vacc_lo = _mm_add_epi32(vbias, _mm_unpacklo_epi16(vsum, vzero));
    __m128i vacc_hi = _mm_add_epi32(vbias, _mm_unpackhi_epi16(vsum, vzero));

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
    vout = _mm_adds_epi16(vout, _mm_load_si128((const __m128i*) params->sse2.output_zero_point));
    vout = _mm_packus_epi16(vout, vout);
    vout = _mm_min_epu8(vout, _mm_load_si128((const __m128i*) params->sse2.output_max));
    vout = _mm_max_epu8(vout, _mm_load_si128((const __m128i*) params->sse2.output_min));

    if (channels & 4) {
      *((uint32_t*) output) = (uint32_t) _mm_cvtsi128_si32(vout);
      output += 4;
      vout = _mm_srli_epi64(vout, 32);
    }
    if (channels & 2) {
      *((uint16_t*) output) = (uint16_t) _mm_extract_epi16(vout, 0);
      output += 2;
      vout = _mm_srli_epi32(vout, 16);
    }
    if (channels & 1) {
      *((uint8_t*) output) = (uint8_t) _mm_cvtsi128_si32(vout);
    }
  }
}
