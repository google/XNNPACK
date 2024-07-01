// Auto-generated file. Do not edit!
//   Template: src/qs8-gavgpool/multipass-sse4.c.in
//   Generator: tools/xngen
//
// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <smmintrin.h>

#include "xnnpack/gavgpool.h"
#include "xnnpack/math.h"
#include "xnnpack/unaligned.h"


void xnn_qu8_gavgpool_minmax_fp32_ukernel_7p7x__sse41_c8(
    size_t rows,
    size_t channels,
    const uint8_t* input,
    size_t input_stride,
    const uint8_t* zero,
    int32_t* buffer,
    uint8_t* output,
    const union xnn_qu8_avgpool_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(rows > 7);
  assert(channels != 0);

  const uint8_t* i0 = input;
  const uint8_t* i1 = (const uint8_t*) ((uintptr_t) i0 + input_stride);
  const uint8_t* i2 = (const uint8_t*) ((uintptr_t) i1 + input_stride);
  const uint8_t* i3 = (const uint8_t*) ((uintptr_t) i2 + input_stride);
  const uint8_t* i4 = (const uint8_t*) ((uintptr_t) i3 + input_stride);
  const uint8_t* i5 = (const uint8_t*) ((uintptr_t) i4 + input_stride);
  const uint8_t* i6 = (const uint8_t*) ((uintptr_t) i5 + input_stride);
  const size_t input_increment = 7 * input_stride - round_up_po2(channels, 8) * sizeof(uint8_t);

  const __m128i vinit_bias = _mm_load_si128((const __m128i*) params->fp32_sse4.init_bias);
  int32_t* b = buffer;
  size_t c = channels;
  for (; c != 0; c = doz(c, 8)) {
    const __m128i vxi0x01234567 = _mm_cvtepu8_epi16(_mm_loadl_epi64((const __m128i*) i0));
    i0 += 8;
    const __m128i vxi1x01234567 = _mm_cvtepu8_epi16(_mm_loadl_epi64((const __m128i*) i1));
    i1 += 8;

    __m128i vacc01234567 = _mm_add_epi16(vxi0x01234567, vxi1x01234567);
    const __m128i vxi2x01234567 = _mm_cvtepu8_epi16(_mm_loadl_epi64((const __m128i*) i2));
    i2 += 8;

    vacc01234567 = _mm_add_epi16(vacc01234567, vxi2x01234567);
    const __m128i vxi3x01234567 = _mm_cvtepu8_epi16(_mm_loadl_epi64((const __m128i*) i3));
    i3 += 8;
    vacc01234567 = _mm_add_epi16(vacc01234567, vxi3x01234567);
    const __m128i vxi4x01234567 = _mm_cvtepu8_epi16(_mm_loadl_epi64((const __m128i*) i4));
    i4 += 8;
    vacc01234567 = _mm_add_epi16(vacc01234567, vxi4x01234567);
    const __m128i vxi5x01234567 = _mm_cvtepu8_epi16(_mm_loadl_epi64((const __m128i*) i5));
    i5 += 8;
    vacc01234567 = _mm_add_epi16(vacc01234567, vxi5x01234567);
    const __m128i vxi6x01234567 = _mm_cvtepu8_epi16(_mm_loadl_epi64((const __m128i*) i6));
    i6 += 8;

    vacc01234567 = _mm_add_epi16(vacc01234567, vxi6x01234567);

    const __m128i vzero = _mm_setzero_si128();
    __m128i vacc0123 = _mm_cvtepu16_epi32(vacc01234567);
    __m128i vacc4567 = _mm_unpackhi_epi16(vacc01234567, vzero);

    vacc0123 = _mm_add_epi32(vacc0123, vinit_bias);
    vacc4567 = _mm_add_epi32(vacc4567, vinit_bias);

    _mm_store_si128((__m128i*) b, vacc0123);
    _mm_store_si128((__m128i*) (b + 4), vacc4567);
    b += 8;
  }

  for (rows -= 7; rows > 7; rows -= 7) {
    i0 = (const uint8_t*) ((uintptr_t) i0 + input_increment);
    i1 = (const uint8_t*) ((uintptr_t) i1 + input_increment);
    i2 = (const uint8_t*) ((uintptr_t) i2 + input_increment);
    i3 = (const uint8_t*) ((uintptr_t) i3 + input_increment);
    i4 = (const uint8_t*) ((uintptr_t) i4 + input_increment);
    i5 = (const uint8_t*) ((uintptr_t) i5 + input_increment);
    i6 = (const uint8_t*) ((uintptr_t) i6 + input_increment);

    int32_t* b = buffer;
    size_t c = channels;
    for (; c != 0; c = doz(c, 8)) {
      const __m128i vxi0x01234567 = _mm_cvtepu8_epi16(_mm_loadl_epi64((const __m128i*) i0));
      i0 += 8;
      const __m128i vxi1x01234567 = _mm_cvtepu8_epi16(_mm_loadl_epi64((const __m128i*) i1));
      i1 += 8;

      __m128i vacc01234567 = _mm_add_epi16(vxi0x01234567, vxi1x01234567);
      const __m128i vxi2x01234567 = _mm_cvtepu8_epi16(_mm_loadl_epi64((const __m128i*) i2));
      i2 += 8;

      vacc01234567 = _mm_add_epi16(vacc01234567, vxi2x01234567);
      const __m128i vxi3x01234567 = _mm_cvtepu8_epi16(_mm_loadl_epi64((const __m128i*) i3));
      i3 += 8;
      vacc01234567 = _mm_add_epi16(vacc01234567, vxi3x01234567);
      const __m128i vxi4x01234567 = _mm_cvtepu8_epi16(_mm_loadl_epi64((const __m128i*) i4));
      i4 += 8;
      vacc01234567 = _mm_add_epi16(vacc01234567, vxi4x01234567);
      const __m128i vxi5x01234567 = _mm_cvtepu8_epi16(_mm_loadl_epi64((const __m128i*) i5));
      i5 += 8;
      vacc01234567 = _mm_add_epi16(vacc01234567, vxi5x01234567);
      const __m128i vxi6x01234567 = _mm_cvtepu8_epi16(_mm_loadl_epi64((const __m128i*) i6));
      i6 += 8;

      vacc01234567 = _mm_add_epi16(vacc01234567, vxi6x01234567);

      const __m128i vzero = _mm_setzero_si128();
      __m128i vacc0123 = _mm_cvtepu16_epi32(vacc01234567);
      __m128i vacc4567 = _mm_unpackhi_epi16(vacc01234567, vzero);

      vacc0123 = _mm_add_epi32(vacc0123, _mm_load_si128((const __m128i*) b));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_load_si128((const __m128i*) (b + 4)));

      _mm_store_si128((__m128i*) b, vacc0123);
      _mm_store_si128((__m128i*) (b + 4), vacc4567);
      b += 8;
    }
  }

  i0 = (const uint8_t*) ((uintptr_t) i0 + input_increment);
  i1 = (const uint8_t*) ((uintptr_t) i1 + input_increment);
  if XNN_UNPREDICTABLE(rows < 2) {
    i1 = zero;
  }
  i2 = (const uint8_t*) ((uintptr_t) i2 + input_increment);
  if XNN_UNPREDICTABLE(rows <= 2) {
    i2 = zero;
  }
  i3 = (const uint8_t*) ((uintptr_t) i3 + input_increment);
  if XNN_UNPREDICTABLE(rows < 4) {
    i3 = zero;
  }
  i4 = (const uint8_t*) ((uintptr_t) i4 + input_increment);
  if XNN_UNPREDICTABLE(rows <= 4) {
    i4 = zero;
  }
  i5 = (const uint8_t*) ((uintptr_t) i5 + input_increment);
  if XNN_UNPREDICTABLE(rows < 6) {
    i5 = zero;
  }
  i6 = (const uint8_t*) ((uintptr_t) i6 + input_increment);
  if XNN_UNPREDICTABLE(rows <= 6) {
    i6 = zero;
  }

  const __m128 vscale = _mm_load_ps(params->fp32_sse4.scale);
  const __m128 voutput_max_less_zero_point = _mm_load_ps(params->fp32_sse4.output_max_less_zero_point);
  const __m128i voutput_zero_point = _mm_load_si128((const __m128i*) params->fp32_sse4.output_zero_point);
  const __m128i voutput_min = _mm_load_si128((const __m128i*) params->fp32_sse4.output_min);
  for (; channels >= 8; channels -= 8) {
    const __m128i vxi0x01234567 = _mm_cvtepu8_epi16(_mm_loadl_epi64((const __m128i*) i0));
    i0 += 8;
    const __m128i vxi1x01234567 = _mm_cvtepu8_epi16(_mm_loadl_epi64((const __m128i*) i1));
    i1 += 8;

    __m128i vacc01234567 = _mm_add_epi16(vxi0x01234567, vxi1x01234567);
    const __m128i vxi2x01234567 = _mm_cvtepu8_epi16(_mm_loadl_epi64((const __m128i*) i2));
    i2 += 8;

    vacc01234567 = _mm_add_epi16(vacc01234567, vxi2x01234567);
    const __m128i vxi3x01234567 = _mm_cvtepu8_epi16(_mm_loadl_epi64((const __m128i*) i3));
    i3 += 8;
    vacc01234567 = _mm_add_epi16(vacc01234567, vxi3x01234567);
    const __m128i vxi4x01234567 = _mm_cvtepu8_epi16(_mm_loadl_epi64((const __m128i*) i4));
    i4 += 8;
    vacc01234567 = _mm_add_epi16(vacc01234567, vxi4x01234567);
    const __m128i vxi5x01234567 = _mm_cvtepu8_epi16(_mm_loadl_epi64((const __m128i*) i5));
    i5 += 8;
    vacc01234567 = _mm_add_epi16(vacc01234567, vxi5x01234567);
    const __m128i vxi6x01234567 = _mm_cvtepu8_epi16(_mm_loadl_epi64((const __m128i*) i6));
    i6 += 8;

    vacc01234567 = _mm_add_epi16(vacc01234567, vxi6x01234567);

    const __m128i vzero = _mm_setzero_si128();
    __m128i vacc0123 = _mm_cvtepu16_epi32(vacc01234567);
    __m128i vacc4567 = _mm_unpackhi_epi16(vacc01234567, vzero);

    vacc0123 = _mm_add_epi32(vacc0123, _mm_load_si128((const __m128i*) buffer));
    vacc4567 = _mm_add_epi32(vacc4567, _mm_load_si128((const __m128i*) (buffer + 4)));
    buffer += 8;

    __m128 vfpacc0123 = _mm_cvtepi32_ps(vacc0123);
    __m128 vfpacc4567 = _mm_cvtepi32_ps(vacc4567);

    vfpacc0123 = _mm_mul_ps(vfpacc0123, vscale);
    vfpacc4567 = _mm_mul_ps(vfpacc4567, vscale);

    vfpacc0123 = _mm_min_ps(vfpacc0123, voutput_max_less_zero_point);
    vfpacc4567 = _mm_min_ps(vfpacc4567, voutput_max_less_zero_point);

    vacc0123 = _mm_cvtps_epi32(vfpacc0123);
    vacc4567 = _mm_cvtps_epi32(vfpacc4567);

    __m128i vout01234567 = _mm_adds_epi16(_mm_packs_epi32(vacc0123, vacc4567), voutput_zero_point);

    __m128i vout0123456701234567 = _mm_packus_epi16(vout01234567, vout01234567);

    vout0123456701234567 = _mm_max_epu8(vout0123456701234567, voutput_min);

    _mm_storel_epi64((__m128i*) output, vout0123456701234567);
    output += 8;
  }
  if XNN_UNLIKELY(channels != 0) {
    {
      const __m128i vxi0x01234567 = _mm_cvtepu8_epi16(_mm_loadl_epi64((const __m128i*) i0));
      i0 += 8;
      const __m128i vxi1x01234567 = _mm_cvtepu8_epi16(_mm_loadl_epi64((const __m128i*) i1));
      i1 += 8;

      __m128i vacc01234567 = _mm_add_epi16(vxi0x01234567, vxi1x01234567);
      const __m128i vxi2x01234567 = _mm_cvtepu8_epi16(_mm_loadl_epi64((const __m128i*) i2));
      i2 += 8;

      vacc01234567 = _mm_add_epi16(vacc01234567, vxi2x01234567);
      const __m128i vxi3x01234567 = _mm_cvtepu8_epi16(_mm_loadl_epi64((const __m128i*) i3));
      i3 += 8;
      vacc01234567 = _mm_add_epi16(vacc01234567, vxi3x01234567);
      const __m128i vxi4x01234567 = _mm_cvtepu8_epi16(_mm_loadl_epi64((const __m128i*) i4));
      i4 += 8;
      vacc01234567 = _mm_add_epi16(vacc01234567, vxi4x01234567);
      const __m128i vxi5x01234567 = _mm_cvtepu8_epi16(_mm_loadl_epi64((const __m128i*) i5));
      i5 += 8;
      vacc01234567 = _mm_add_epi16(vacc01234567, vxi5x01234567);
      const __m128i vxi6x01234567 = _mm_cvtepu8_epi16(_mm_loadl_epi64((const __m128i*) i6));
      i6 += 8;

      vacc01234567 = _mm_add_epi16(vacc01234567, vxi6x01234567);

      __m128i vacc0123 = _mm_cvtepu16_epi32(vacc01234567);
      __m128i vacc4567 = _mm_unpackhi_epi16(vacc01234567, _mm_setzero_si128());

      vacc0123 = _mm_add_epi32(vacc0123, _mm_load_si128((const __m128i*) buffer));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_load_si128((const __m128i*) (buffer + 4)));
      buffer += 8;

      __m128 vfpacc0123 = _mm_cvtepi32_ps(vacc0123);
      __m128 vfpacc4567 = _mm_cvtepi32_ps(vacc4567);

      vfpacc0123 = _mm_mul_ps(vfpacc0123, vscale);
      vfpacc4567 = _mm_mul_ps(vfpacc4567, vscale);

      vfpacc0123 = _mm_min_ps(vfpacc0123, voutput_max_less_zero_point);
      vfpacc4567 = _mm_min_ps(vfpacc4567, voutput_max_less_zero_point);

      vacc0123 = _mm_cvtps_epi32(vfpacc0123);
      vacc4567 = _mm_cvtps_epi32(vfpacc4567);

      __m128i vout01234567 = _mm_adds_epi16(_mm_packs_epi32(vacc0123, vacc4567), voutput_zero_point);

      __m128i vout0123456701234567 = _mm_packus_epi16(vout01234567, vout01234567);
      vout0123456701234567 = _mm_max_epu8(vout0123456701234567, voutput_min);

      if (channels & 4) {
        unaligned_store_u32(output, (uint32_t) _mm_cvtsi128_si32(vout0123456701234567));
        vout0123456701234567 = _mm_srli_epi64(vout0123456701234567, 32);
        output += 4;
      }
      if (channels & 2) {
        unaligned_store_u16(output, (uint16_t) _mm_extract_epi16(vout0123456701234567, 0));
        vout0123456701234567 = _mm_srli_epi32(vout0123456701234567, 16);
        output += 2;
      }
      if (channels & 1) {
        *output = (uint8_t) _mm_extract_epi8(vout0123456701234567, 0);
      }
    }
  }
}
