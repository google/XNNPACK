// Auto-generated file. Do not edit!
//   Template: src/qs8-gavgpool/unipass-sse2.c.in
//   Generator: tools/xngen
//
// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <emmintrin.h>

#include "xnnpack/gavgpool.h"
#include "xnnpack/unaligned.h"


void xnn_qs8_gavgpool_minmax_fp32_ukernel_7x__sse2_c8(
    size_t rows,
    size_t channels,
    const int8_t* input,
    size_t input_stride,
    const int8_t* zero,
    int8_t* output,
    const union xnn_qs8_avgpool_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
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

  const __m128i vinit_bias = _mm_load_si128((const __m128i*) params->fp32_sse2.init_bias);
  const __m128 vscale = _mm_load_ps(params->fp32_sse2.scale);
  const __m128 voutput_max_less_zero_point = _mm_load_ps(params->fp32_sse2.output_max_less_zero_point);
  const __m128i voutput_zero_point = _mm_load_si128((const __m128i*) params->fp32_sse2.output_zero_point);
  const __m128i voutput_min = _mm_load_si128((const __m128i*) params->fp32_sse2.output_min);
  for (; channels >= 8; channels -= 8) {

    const __m128i vi0x01234567 = _mm_loadl_epi64((const __m128i*) i0);
    i0 += 8;

    const __m128i vxi0x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vi0x01234567, vi0x01234567), 8);
    const __m128i vi1x01234567 = _mm_loadl_epi64((const __m128i*) i1);
    i1 += 8;

    const __m128i vxi1x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vi1x01234567, vi1x01234567), 8);
    const __m128i vi2x01234567 = _mm_loadl_epi64((const __m128i*) i2);
    i2 += 8;

    __m128i vacc01234567 = _mm_add_epi16(vxi0x01234567, vxi1x01234567);
    const __m128i vxi2x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vi2x01234567, vi2x01234567), 8);
    const __m128i vi3x01234567 = _mm_loadl_epi64((const __m128i*) i3);
    i3 += 8;

    vacc01234567 = _mm_add_epi16(vacc01234567, vxi2x01234567);
    const __m128i vxi3x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vi3x01234567, vi3x01234567), 8);
    const __m128i vi4x01234567 = _mm_loadl_epi64((const __m128i*) i4);
    i4 += 8;

    vacc01234567 = _mm_add_epi16(vacc01234567, vxi3x01234567);
    const __m128i vxi4x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vi4x01234567, vi4x01234567), 8);
    const __m128i vi5x01234567 = _mm_loadl_epi64((const __m128i*) i5);
    i5 += 8;

    vacc01234567 = _mm_add_epi16(vacc01234567, vxi4x01234567);
    const __m128i vxi5x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vi5x01234567, vi5x01234567), 8);
    const __m128i vi6x01234567 = _mm_loadl_epi64((const __m128i*) i6);
    i6 += 8;

    vacc01234567 = _mm_add_epi16(vacc01234567, vxi5x01234567);
    const __m128i vxi6x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vi6x01234567, vi6x01234567), 8);

    vacc01234567 = _mm_add_epi16(vacc01234567, vxi6x01234567);

    const __m128i vsgnacc01234567 = _mm_cmpgt_epi16(_mm_setzero_si128(), vacc01234567);
    __m128i vacc0123 = _mm_unpacklo_epi16(vacc01234567, vsgnacc01234567);
    __m128i vacc4567 = _mm_unpackhi_epi16(vacc01234567, vsgnacc01234567);

    vacc0123 = _mm_add_epi32(vacc0123, vinit_bias);
    vacc4567 = _mm_add_epi32(vacc4567, vinit_bias);

    __m128 vfpacc0123 = _mm_cvtepi32_ps(vacc0123);
    __m128 vfpacc4567 = _mm_cvtepi32_ps(vacc4567);

    vfpacc0123 = _mm_mul_ps(vfpacc0123, vscale);
    vfpacc4567 = _mm_mul_ps(vfpacc4567, vscale);

    vfpacc0123 = _mm_min_ps(vfpacc0123, voutput_max_less_zero_point);
    vfpacc4567 = _mm_min_ps(vfpacc4567, voutput_max_less_zero_point);

    vacc0123 = _mm_cvtps_epi32(vfpacc0123);
    vacc4567 = _mm_cvtps_epi32(vfpacc4567);

    __m128i vout01234567 = _mm_adds_epi16(_mm_packs_epi32(vacc0123, vacc4567), voutput_zero_point);

    vout01234567 = _mm_max_epi16(vout01234567, voutput_min);

    __m128i vout0123456701234567 = _mm_packs_epi16(vout01234567, vout01234567);


    _mm_storel_epi64((__m128i*) output, vout0123456701234567);
    output += 8;
  }
  if XNN_UNLIKELY(channels != 0) {
    {

      const __m128i vi0x01234567 = _mm_loadl_epi64((const __m128i*) i0);
      i0 += 8;

      const __m128i vi1x01234567 = _mm_loadl_epi64((const __m128i*) i1);
      i1 += 8;

      const __m128i vxi0x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vi0x01234567, vi0x01234567), 8);
      const __m128i vi2x01234567 = _mm_loadl_epi64((const __m128i*) i2);
      i2 += 8;

      const __m128i vxi1x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vi1x01234567, vi1x01234567), 8);
      const __m128i vi3x01234567 = _mm_loadl_epi64((const __m128i*) i3);
      i3 += 8;

      __m128i vacc01234567 = _mm_add_epi16(vxi0x01234567, vxi1x01234567);
      const __m128i vxi2x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vi2x01234567, vi2x01234567), 8);
      const __m128i vi4x01234567 = _mm_loadl_epi64((const __m128i*) i4);
      i4 += 8;

      vacc01234567 = _mm_add_epi16(vacc01234567, vxi2x01234567);
      const __m128i vxi3x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vi3x01234567, vi3x01234567), 8);
      const __m128i vi5x01234567 = _mm_loadl_epi64((const __m128i*) i5);
      i5 += 8;

      vacc01234567 = _mm_add_epi16(vacc01234567, vxi3x01234567);
      const __m128i vxi4x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vi4x01234567, vi4x01234567), 8);
      const __m128i vi6x01234567 = _mm_loadl_epi64((const __m128i*) i6);
      i6 += 8;

      vacc01234567 = _mm_add_epi16(vacc01234567, vxi4x01234567);
      const __m128i vxi5x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vi5x01234567, vi5x01234567), 8);

      vacc01234567 = _mm_add_epi16(vacc01234567, vxi5x01234567);
      const __m128i vxi6x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vi6x01234567, vi6x01234567), 8);

      vacc01234567 = _mm_add_epi16(vacc01234567, vxi6x01234567);

      const __m128i vsgnacc01234567 = _mm_cmpgt_epi16(_mm_setzero_si128(), vacc01234567);
      __m128i vacc0123 = _mm_unpacklo_epi16(vacc01234567, vsgnacc01234567);
      __m128i vacc4567 = _mm_unpackhi_epi16(vacc01234567, vsgnacc01234567);

      vacc0123 = _mm_add_epi32(vacc0123, vinit_bias);
      vacc4567 = _mm_add_epi32(vacc4567, vinit_bias);

      __m128 vfpacc0123 = _mm_cvtepi32_ps(vacc0123);
      __m128 vfpacc4567 = _mm_cvtepi32_ps(vacc4567);

      vfpacc0123 = _mm_mul_ps(vfpacc0123, vscale);
      vfpacc4567 = _mm_mul_ps(vfpacc4567, vscale);

      vfpacc0123 = _mm_min_ps(vfpacc0123, voutput_max_less_zero_point);
      vfpacc4567 = _mm_min_ps(vfpacc4567, voutput_max_less_zero_point);

      vacc0123 = _mm_cvtps_epi32(vfpacc0123);
      vacc4567 = _mm_cvtps_epi32(vfpacc4567);

      __m128i vout01234567 = _mm_adds_epi16(_mm_packs_epi32(vacc0123, vacc4567), voutput_zero_point);
      vout01234567 = _mm_max_epi16(vout01234567, voutput_min);

      __m128i vout0123456701234567 = _mm_packs_epi16(vout01234567, vout01234567);

      if (channels & 4) {
        unaligned_store_u32(output, (uint32_t) _mm_cvtsi128_si32(vout0123456701234567));
        vout0123456701234567 = _mm_srli_epi64(vout0123456701234567, 32);
        output += 4;
      }
      uint32_t vout0123 = (uint32_t) _mm_cvtsi128_si32(vout0123456701234567);
      if (channels & 2) {
        unaligned_store_u16(output, (uint16_t) vout0123);
        vout0123 >>= 16;
        output += 2;
      }
      if (channels & 1) {
        *output = (int8_t) vout0123;
      }
    }
  }
}
