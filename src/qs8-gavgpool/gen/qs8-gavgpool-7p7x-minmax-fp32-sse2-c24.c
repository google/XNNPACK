// Auto-generated file. Do not edit!
//   Template: src/qs8-gavgpool/multipass-sse2.c.in
//   Generator: tools/xngen
//
// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <emmintrin.h>

#include "xnnpack/gavgpool.h"
#include "xnnpack/math.h"
#include "xnnpack/unaligned.h"


void xnn_qs8_gavgpool_minmax_fp32_ukernel_7p7x__sse2_c24(
    size_t rows,
    size_t channels,
    const int8_t* input,
    size_t input_stride,
    const int8_t* zero,
    int32_t* buffer,
    int8_t* output,
    const union xnn_qs8_avgpool_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(rows > 7);
  assert(channels != 0);

  const int8_t* i0 = input;
  const int8_t* i1 = (const int8_t*) ((uintptr_t) i0 + input_stride);
  const int8_t* i2 = (const int8_t*) ((uintptr_t) i1 + input_stride);
  const int8_t* i3 = (const int8_t*) ((uintptr_t) i2 + input_stride);
  const int8_t* i4 = (const int8_t*) ((uintptr_t) i3 + input_stride);
  const int8_t* i5 = (const int8_t*) ((uintptr_t) i4 + input_stride);
  const int8_t* i6 = (const int8_t*) ((uintptr_t) i5 + input_stride);
  const size_t input_increment = 7 * input_stride - round_up_po2(channels, 8) * sizeof(int8_t);

  const __m128i vinit_bias = _mm_load_si128((const __m128i*) params->fp32_sse2.init_bias);
  int32_t* b = buffer;
  size_t c = channels;
  for (; c >= 24; c -= 24) {

    const __m128i vi0x01234567 = _mm_loadl_epi64((const __m128i*) i0);
    const __m128i vi0x89ABCDEF = _mm_loadl_epi64((const __m128i*) (i0 + 8));
    const __m128i vi0xGHIJKLMN = _mm_loadl_epi64((const __m128i*) (i0 + 16));
    i0 += 24;

    const __m128i vxi0x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vi0x01234567, vi0x01234567), 8);
    const __m128i vi1x01234567 = _mm_loadl_epi64((const __m128i*) i1);
    const __m128i vxi0x89ABCDEF = _mm_srai_epi16(_mm_unpacklo_epi8(vi0x89ABCDEF, vi0x89ABCDEF), 8);
    const __m128i vi1x89ABCDEF = _mm_loadl_epi64((const __m128i*) (i1 + 8));
    const __m128i vxi0xGHIJKLMN = _mm_srai_epi16(_mm_unpacklo_epi8(vi0xGHIJKLMN, vi0xGHIJKLMN), 8);
    const __m128i vi1xGHIJKLMN = _mm_loadl_epi64((const __m128i*) (i1 + 16));
    i1 += 24;

    const __m128i vxi1x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vi1x01234567, vi1x01234567), 8);
    const __m128i vi2x01234567 = _mm_loadl_epi64((const __m128i*) i2);
    const __m128i vxi1x89ABCDEF = _mm_srai_epi16(_mm_unpacklo_epi8(vi1x89ABCDEF, vi1x89ABCDEF), 8);
    const __m128i vi2x89ABCDEF = _mm_loadl_epi64((const __m128i*) (i2 + 8));
    const __m128i vxi1xGHIJKLMN = _mm_srai_epi16(_mm_unpacklo_epi8(vi1xGHIJKLMN, vi1xGHIJKLMN), 8);
    const __m128i vi2xGHIJKLMN = _mm_loadl_epi64((const __m128i*) (i2 + 16));
    i2 += 24;

    __m128i vacc01234567 = _mm_add_epi16(vxi0x01234567, vxi1x01234567);
    const __m128i vxi2x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vi2x01234567, vi2x01234567), 8);
    const __m128i vi3x01234567 = _mm_loadl_epi64((const __m128i*) i3);
    __m128i vacc89ABCDEF = _mm_add_epi16(vxi0x89ABCDEF, vxi1x89ABCDEF);
    const __m128i vxi2x89ABCDEF = _mm_srai_epi16(_mm_unpacklo_epi8(vi2x89ABCDEF, vi2x89ABCDEF), 8);
    const __m128i vi3x89ABCDEF = _mm_loadl_epi64((const __m128i*) (i3 + 8));
    __m128i vaccGHIJKLMN = _mm_add_epi16(vxi0xGHIJKLMN, vxi1xGHIJKLMN);
    const __m128i vxi2xGHIJKLMN = _mm_srai_epi16(_mm_unpacklo_epi8(vi2xGHIJKLMN, vi2xGHIJKLMN), 8);
    const __m128i vi3xGHIJKLMN = _mm_loadl_epi64((const __m128i*) (i3 + 16));
    i3 += 24;

    vacc01234567 = _mm_add_epi16(vacc01234567, vxi2x01234567);
    const __m128i vxi3x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vi3x01234567, vi3x01234567), 8);
    const __m128i vi4x01234567 = _mm_loadl_epi64((const __m128i*) i4);
    vacc89ABCDEF = _mm_add_epi16(vacc89ABCDEF, vxi2x89ABCDEF);
    const __m128i vxi3x89ABCDEF = _mm_srai_epi16(_mm_unpacklo_epi8(vi3x89ABCDEF, vi3x89ABCDEF), 8);
    const __m128i vi4x89ABCDEF = _mm_loadl_epi64((const __m128i*) (i4 + 8));
    vaccGHIJKLMN = _mm_add_epi16(vaccGHIJKLMN, vxi2xGHIJKLMN);
    const __m128i vxi3xGHIJKLMN = _mm_srai_epi16(_mm_unpacklo_epi8(vi3xGHIJKLMN, vi3xGHIJKLMN), 8);
    const __m128i vi4xGHIJKLMN = _mm_loadl_epi64((const __m128i*) (i4 + 16));
    i4 += 24;

    vacc01234567 = _mm_add_epi16(vacc01234567, vxi3x01234567);
    const __m128i vxi4x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vi4x01234567, vi4x01234567), 8);
    const __m128i vi5x01234567 = _mm_loadl_epi64((const __m128i*) i5);
    vacc89ABCDEF = _mm_add_epi16(vacc89ABCDEF, vxi3x89ABCDEF);
    const __m128i vxi4x89ABCDEF = _mm_srai_epi16(_mm_unpacklo_epi8(vi4x89ABCDEF, vi4x89ABCDEF), 8);
    const __m128i vi5x89ABCDEF = _mm_loadl_epi64((const __m128i*) (i5 + 8));
    vaccGHIJKLMN = _mm_add_epi16(vaccGHIJKLMN, vxi3xGHIJKLMN);
    const __m128i vxi4xGHIJKLMN = _mm_srai_epi16(_mm_unpacklo_epi8(vi4xGHIJKLMN, vi4xGHIJKLMN), 8);
    const __m128i vi5xGHIJKLMN = _mm_loadl_epi64((const __m128i*) (i5 + 16));
    i5 += 24;

    vacc01234567 = _mm_add_epi16(vacc01234567, vxi4x01234567);
    const __m128i vxi5x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vi5x01234567, vi5x01234567), 8);
    const __m128i vi6x01234567 = _mm_loadl_epi64((const __m128i*) i6);
    vacc89ABCDEF = _mm_add_epi16(vacc89ABCDEF, vxi4x89ABCDEF);
    const __m128i vxi5x89ABCDEF = _mm_srai_epi16(_mm_unpacklo_epi8(vi5x89ABCDEF, vi5x89ABCDEF), 8);
    const __m128i vi6x89ABCDEF = _mm_loadl_epi64((const __m128i*) (i6 + 8));
    vaccGHIJKLMN = _mm_add_epi16(vaccGHIJKLMN, vxi4xGHIJKLMN);
    const __m128i vxi5xGHIJKLMN = _mm_srai_epi16(_mm_unpacklo_epi8(vi5xGHIJKLMN, vi5xGHIJKLMN), 8);
    const __m128i vi6xGHIJKLMN = _mm_loadl_epi64((const __m128i*) (i6 + 16));
    i6 += 24;

    vacc01234567 = _mm_add_epi16(vacc01234567, vxi5x01234567);
    const __m128i vxi6x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vi6x01234567, vi6x01234567), 8);
    vacc89ABCDEF = _mm_add_epi16(vacc89ABCDEF, vxi5x89ABCDEF);
    const __m128i vxi6x89ABCDEF = _mm_srai_epi16(_mm_unpacklo_epi8(vi6x89ABCDEF, vi6x89ABCDEF), 8);
    vaccGHIJKLMN = _mm_add_epi16(vaccGHIJKLMN, vxi5xGHIJKLMN);
    const __m128i vxi6xGHIJKLMN = _mm_srai_epi16(_mm_unpacklo_epi8(vi6xGHIJKLMN, vi6xGHIJKLMN), 8);

    vacc01234567 = _mm_add_epi16(vacc01234567, vxi6x01234567);
    vacc89ABCDEF = _mm_add_epi16(vacc89ABCDEF, vxi6x89ABCDEF);
    vaccGHIJKLMN = _mm_add_epi16(vaccGHIJKLMN, vxi6xGHIJKLMN);

    const __m128i vsgnacc01234567 = _mm_cmpgt_epi16(_mm_setzero_si128(), vacc01234567);
    __m128i vacc0123 = _mm_unpacklo_epi16(vacc01234567, vsgnacc01234567);
    __m128i vacc4567 = _mm_unpackhi_epi16(vacc01234567, vsgnacc01234567);
    const __m128i vsgnacc89ABCDEF = _mm_cmpgt_epi16(_mm_setzero_si128(), vacc89ABCDEF);
    __m128i vacc89AB = _mm_unpacklo_epi16(vacc89ABCDEF, vsgnacc89ABCDEF);
    __m128i vaccCDEF = _mm_unpackhi_epi16(vacc89ABCDEF, vsgnacc89ABCDEF);
    const __m128i vsgnaccGHIJKLMN = _mm_cmpgt_epi16(_mm_setzero_si128(), vaccGHIJKLMN);
    __m128i vaccGHIJ = _mm_unpacklo_epi16(vaccGHIJKLMN, vsgnaccGHIJKLMN);
    __m128i vaccKLMN = _mm_unpackhi_epi16(vaccGHIJKLMN, vsgnaccGHIJKLMN);

    vacc0123 = _mm_add_epi32(vacc0123, vinit_bias);
    vacc4567 = _mm_add_epi32(vacc4567, vinit_bias);
    vacc89AB = _mm_add_epi32(vacc89AB, vinit_bias);
    vaccCDEF = _mm_add_epi32(vaccCDEF, vinit_bias);
    vaccGHIJ = _mm_add_epi32(vaccGHIJ, vinit_bias);
    vaccKLMN = _mm_add_epi32(vaccKLMN, vinit_bias);

    _mm_store_si128((__m128i*) b, vacc0123);
    _mm_store_si128((__m128i*) (b + 4), vacc4567);
    _mm_store_si128((__m128i*) (b + 8), vacc89AB);
    _mm_store_si128((__m128i*) (b + 12), vaccCDEF);
    _mm_store_si128((__m128i*) (b + 16), vaccGHIJ);
    _mm_store_si128((__m128i*) (b + 20), vaccKLMN);
    b += 24;
  }
  if XNN_UNLIKELY(c != 0) {
    do {

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

      _mm_store_si128((__m128i*) b, vacc0123);
      _mm_store_si128((__m128i*) (b + 4), vacc4567);
      b += 8;

      c = doz(c, 8);
    } while (c != 0);
  }

  for (rows -= 7; rows > 7; rows -= 7) {
    i0 = (const int8_t*) ((uintptr_t) i0 + input_increment);
    i1 = (const int8_t*) ((uintptr_t) i1 + input_increment);
    i2 = (const int8_t*) ((uintptr_t) i2 + input_increment);
    i3 = (const int8_t*) ((uintptr_t) i3 + input_increment);
    i4 = (const int8_t*) ((uintptr_t) i4 + input_increment);
    i5 = (const int8_t*) ((uintptr_t) i5 + input_increment);
    i6 = (const int8_t*) ((uintptr_t) i6 + input_increment);

    int32_t* b = buffer;
    size_t c = channels;
    for (; c >= 24; c -= 24) {

      const __m128i vi0x01234567 = _mm_loadl_epi64((const __m128i*) i0);
      const __m128i vi0x89ABCDEF = _mm_loadl_epi64((const __m128i*) (i0 + 8));
      const __m128i vi0xGHIJKLMN = _mm_loadl_epi64((const __m128i*) (i0 + 16));
      i0 += 24;

      const __m128i vxi0x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vi0x01234567, vi0x01234567), 8);
      const __m128i vi1x01234567 = _mm_loadl_epi64((const __m128i*) i1);
      const __m128i vxi0x89ABCDEF = _mm_srai_epi16(_mm_unpacklo_epi8(vi0x89ABCDEF, vi0x89ABCDEF), 8);
      const __m128i vi1x89ABCDEF = _mm_loadl_epi64((const __m128i*) (i1 + 8));
      const __m128i vxi0xGHIJKLMN = _mm_srai_epi16(_mm_unpacklo_epi8(vi0xGHIJKLMN, vi0xGHIJKLMN), 8);
      const __m128i vi1xGHIJKLMN = _mm_loadl_epi64((const __m128i*) (i1 + 16));
      i1 += 24;

      const __m128i vxi1x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vi1x01234567, vi1x01234567), 8);
      const __m128i vi2x01234567 = _mm_loadl_epi64((const __m128i*) i2);
      const __m128i vxi1x89ABCDEF = _mm_srai_epi16(_mm_unpacklo_epi8(vi1x89ABCDEF, vi1x89ABCDEF), 8);
      const __m128i vi2x89ABCDEF = _mm_loadl_epi64((const __m128i*) (i2 + 8));
      const __m128i vxi1xGHIJKLMN = _mm_srai_epi16(_mm_unpacklo_epi8(vi1xGHIJKLMN, vi1xGHIJKLMN), 8);
      const __m128i vi2xGHIJKLMN = _mm_loadl_epi64((const __m128i*) (i2 + 16));
      i2 += 24;

      __m128i vacc01234567 = _mm_add_epi16(vxi0x01234567, vxi1x01234567);
      const __m128i vxi2x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vi2x01234567, vi2x01234567), 8);
      const __m128i vi3x01234567 = _mm_loadl_epi64((const __m128i*) i3);
      __m128i vacc89ABCDEF = _mm_add_epi16(vxi0x89ABCDEF, vxi1x89ABCDEF);
      const __m128i vxi2x89ABCDEF = _mm_srai_epi16(_mm_unpacklo_epi8(vi2x89ABCDEF, vi2x89ABCDEF), 8);
      const __m128i vi3x89ABCDEF = _mm_loadl_epi64((const __m128i*) (i3 + 8));
      __m128i vaccGHIJKLMN = _mm_add_epi16(vxi0xGHIJKLMN, vxi1xGHIJKLMN);
      const __m128i vxi2xGHIJKLMN = _mm_srai_epi16(_mm_unpacklo_epi8(vi2xGHIJKLMN, vi2xGHIJKLMN), 8);
      const __m128i vi3xGHIJKLMN = _mm_loadl_epi64((const __m128i*) (i3 + 16));
      i3 += 24;

      vacc01234567 = _mm_add_epi16(vacc01234567, vxi2x01234567);
      const __m128i vxi3x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vi3x01234567, vi3x01234567), 8);
      const __m128i vi4x01234567 = _mm_loadl_epi64((const __m128i*) i4);
      vacc89ABCDEF = _mm_add_epi16(vacc89ABCDEF, vxi2x89ABCDEF);
      const __m128i vxi3x89ABCDEF = _mm_srai_epi16(_mm_unpacklo_epi8(vi3x89ABCDEF, vi3x89ABCDEF), 8);
      const __m128i vi4x89ABCDEF = _mm_loadl_epi64((const __m128i*) (i4 + 8));
      vaccGHIJKLMN = _mm_add_epi16(vaccGHIJKLMN, vxi2xGHIJKLMN);
      const __m128i vxi3xGHIJKLMN = _mm_srai_epi16(_mm_unpacklo_epi8(vi3xGHIJKLMN, vi3xGHIJKLMN), 8);
      const __m128i vi4xGHIJKLMN = _mm_loadl_epi64((const __m128i*) (i4 + 16));
      i4 += 24;

      vacc01234567 = _mm_add_epi16(vacc01234567, vxi3x01234567);
      const __m128i vxi4x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vi4x01234567, vi4x01234567), 8);
      const __m128i vi5x01234567 = _mm_loadl_epi64((const __m128i*) i5);
      vacc89ABCDEF = _mm_add_epi16(vacc89ABCDEF, vxi3x89ABCDEF);
      const __m128i vxi4x89ABCDEF = _mm_srai_epi16(_mm_unpacklo_epi8(vi4x89ABCDEF, vi4x89ABCDEF), 8);
      const __m128i vi5x89ABCDEF = _mm_loadl_epi64((const __m128i*) (i5 + 8));
      vaccGHIJKLMN = _mm_add_epi16(vaccGHIJKLMN, vxi3xGHIJKLMN);
      const __m128i vxi4xGHIJKLMN = _mm_srai_epi16(_mm_unpacklo_epi8(vi4xGHIJKLMN, vi4xGHIJKLMN), 8);
      const __m128i vi5xGHIJKLMN = _mm_loadl_epi64((const __m128i*) (i5 + 16));
      i5 += 24;

      vacc01234567 = _mm_add_epi16(vacc01234567, vxi4x01234567);
      const __m128i vxi5x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vi5x01234567, vi5x01234567), 8);
      const __m128i vi6x01234567 = _mm_loadl_epi64((const __m128i*) i6);
      vacc89ABCDEF = _mm_add_epi16(vacc89ABCDEF, vxi4x89ABCDEF);
      const __m128i vxi5x89ABCDEF = _mm_srai_epi16(_mm_unpacklo_epi8(vi5x89ABCDEF, vi5x89ABCDEF), 8);
      const __m128i vi6x89ABCDEF = _mm_loadl_epi64((const __m128i*) (i6 + 8));
      vaccGHIJKLMN = _mm_add_epi16(vaccGHIJKLMN, vxi4xGHIJKLMN);
      const __m128i vxi5xGHIJKLMN = _mm_srai_epi16(_mm_unpacklo_epi8(vi5xGHIJKLMN, vi5xGHIJKLMN), 8);
      const __m128i vi6xGHIJKLMN = _mm_loadl_epi64((const __m128i*) (i6 + 16));
      i6 += 24;

      vacc01234567 = _mm_add_epi16(vacc01234567, vxi5x01234567);
      const __m128i vxi6x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vi6x01234567, vi6x01234567), 8);
      vacc89ABCDEF = _mm_add_epi16(vacc89ABCDEF, vxi5x89ABCDEF);
      const __m128i vxi6x89ABCDEF = _mm_srai_epi16(_mm_unpacklo_epi8(vi6x89ABCDEF, vi6x89ABCDEF), 8);
      vaccGHIJKLMN = _mm_add_epi16(vaccGHIJKLMN, vxi5xGHIJKLMN);
      const __m128i vxi6xGHIJKLMN = _mm_srai_epi16(_mm_unpacklo_epi8(vi6xGHIJKLMN, vi6xGHIJKLMN), 8);

      vacc01234567 = _mm_add_epi16(vacc01234567, vxi6x01234567);
      vacc89ABCDEF = _mm_add_epi16(vacc89ABCDEF, vxi6x89ABCDEF);
      vaccGHIJKLMN = _mm_add_epi16(vaccGHIJKLMN, vxi6xGHIJKLMN);

      const __m128i vsgnacc01234567 = _mm_cmpgt_epi16(_mm_setzero_si128(), vacc01234567);
      __m128i vacc0123 = _mm_unpacklo_epi16(vacc01234567, vsgnacc01234567);
      __m128i vacc4567 = _mm_unpackhi_epi16(vacc01234567, vsgnacc01234567);
      const __m128i vsgnacc89ABCDEF = _mm_cmpgt_epi16(_mm_setzero_si128(), vacc89ABCDEF);
      __m128i vacc89AB = _mm_unpacklo_epi16(vacc89ABCDEF, vsgnacc89ABCDEF);
      __m128i vaccCDEF = _mm_unpackhi_epi16(vacc89ABCDEF, vsgnacc89ABCDEF);
      const __m128i vsgnaccGHIJKLMN = _mm_cmpgt_epi16(_mm_setzero_si128(), vaccGHIJKLMN);
      __m128i vaccGHIJ = _mm_unpacklo_epi16(vaccGHIJKLMN, vsgnaccGHIJKLMN);
      __m128i vaccKLMN = _mm_unpackhi_epi16(vaccGHIJKLMN, vsgnaccGHIJKLMN);

      vacc0123 = _mm_add_epi32(vacc0123, _mm_load_si128((const __m128i*) b));
      vacc4567 = _mm_add_epi32(vacc4567, _mm_load_si128((const __m128i*) (b + 4)));
      vacc89AB = _mm_add_epi32(vacc89AB, _mm_load_si128((const __m128i*) (b + 8)));
      vaccCDEF = _mm_add_epi32(vaccCDEF, _mm_load_si128((const __m128i*) (b + 12)));
      vaccGHIJ = _mm_add_epi32(vaccGHIJ, _mm_load_si128((const __m128i*) (b + 16)));
      vaccKLMN = _mm_add_epi32(vaccKLMN, _mm_load_si128((const __m128i*) (b + 20)));

      _mm_store_si128((__m128i*) b, vacc0123);
      _mm_store_si128((__m128i*) (b + 4), vacc4567);
      _mm_store_si128((__m128i*) (b + 8), vacc89AB);
      _mm_store_si128((__m128i*) (b + 12), vaccCDEF);
      _mm_store_si128((__m128i*) (b + 16), vaccGHIJ);
      _mm_store_si128((__m128i*) (b + 20), vaccKLMN);
      b += 24;
    }
    if XNN_UNLIKELY(c != 0) {
      do {

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

        vacc0123 = _mm_add_epi32(vacc0123, _mm_load_si128((const __m128i*) b));
        vacc4567 = _mm_add_epi32(vacc4567, _mm_load_si128((const __m128i*) (b + 4)));

        _mm_store_si128((__m128i*) b, vacc0123);
        _mm_store_si128((__m128i*) (b + 4), vacc4567);
        b += 8;

        c = doz(c, 8);
      } while (c != 0);
    }
  }

  i0 = (const int8_t*) ((uintptr_t) i0 + input_increment);
  i1 = (const int8_t*) ((uintptr_t) i1 + input_increment);
  if XNN_UNPREDICTABLE(rows < 2) {
    i1 = zero;
  }
  i2 = (const int8_t*) ((uintptr_t) i2 + input_increment);
  if XNN_UNPREDICTABLE(rows <= 2) {
    i2 = zero;
  }
  i3 = (const int8_t*) ((uintptr_t) i3 + input_increment);
  if XNN_UNPREDICTABLE(rows < 4) {
    i3 = zero;
  }
  i4 = (const int8_t*) ((uintptr_t) i4 + input_increment);
  if XNN_UNPREDICTABLE(rows <= 4) {
    i4 = zero;
  }
  i5 = (const int8_t*) ((uintptr_t) i5 + input_increment);
  if XNN_UNPREDICTABLE(rows < 6) {
    i5 = zero;
  }
  i6 = (const int8_t*) ((uintptr_t) i6 + input_increment);
  if XNN_UNPREDICTABLE(rows <= 6) {
    i6 = zero;
  }

  const __m128 vscale = _mm_load_ps(params->fp32_sse2.scale);
  const __m128 voutput_max_less_zero_point = _mm_load_ps(params->fp32_sse2.output_max_less_zero_point);
  const __m128i voutput_zero_point = _mm_load_si128((const __m128i*) params->fp32_sse2.output_zero_point);
  const __m128i voutput_min = _mm_load_si128((const __m128i*) params->fp32_sse2.output_min);
  for (; channels >= 24; channels -= 24) {

    const __m128i vi0x01234567 = _mm_loadl_epi64((const __m128i*) i0);
    const __m128i vi0x89ABCDEF = _mm_loadl_epi64((const __m128i*) (i0 + 8));
    const __m128i vi0xGHIJKLMN = _mm_loadl_epi64((const __m128i*) (i0 + 16));
    i0 += 24;

    const __m128i vxi0x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vi0x01234567, vi0x01234567), 8);
    const __m128i vi1x01234567 = _mm_loadl_epi64((const __m128i*) i1);
    const __m128i vxi0x89ABCDEF = _mm_srai_epi16(_mm_unpacklo_epi8(vi0x89ABCDEF, vi0x89ABCDEF), 8);
    const __m128i vi1x89ABCDEF = _mm_loadl_epi64((const __m128i*) (i1 + 8));
    const __m128i vxi0xGHIJKLMN = _mm_srai_epi16(_mm_unpacklo_epi8(vi0xGHIJKLMN, vi0xGHIJKLMN), 8);
    const __m128i vi1xGHIJKLMN = _mm_loadl_epi64((const __m128i*) (i1 + 16));
    i1 += 24;

    const __m128i vxi1x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vi1x01234567, vi1x01234567), 8);
    const __m128i vi2x01234567 = _mm_loadl_epi64((const __m128i*) i2);
    const __m128i vxi1x89ABCDEF = _mm_srai_epi16(_mm_unpacklo_epi8(vi1x89ABCDEF, vi1x89ABCDEF), 8);
    const __m128i vi2x89ABCDEF = _mm_loadl_epi64((const __m128i*) (i2 + 8));
    const __m128i vxi1xGHIJKLMN = _mm_srai_epi16(_mm_unpacklo_epi8(vi1xGHIJKLMN, vi1xGHIJKLMN), 8);
    const __m128i vi2xGHIJKLMN = _mm_loadl_epi64((const __m128i*) (i2 + 16));
    i2 += 24;

    __m128i vacc01234567 = _mm_add_epi16(vxi0x01234567, vxi1x01234567);
    const __m128i vxi2x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vi2x01234567, vi2x01234567), 8);
    const __m128i vi3x01234567 = _mm_loadl_epi64((const __m128i*) i3);
    __m128i vacc89ABCDEF = _mm_add_epi16(vxi0x89ABCDEF, vxi1x89ABCDEF);
    const __m128i vxi2x89ABCDEF = _mm_srai_epi16(_mm_unpacklo_epi8(vi2x89ABCDEF, vi2x89ABCDEF), 8);
    const __m128i vi3x89ABCDEF = _mm_loadl_epi64((const __m128i*) (i3 + 8));
    __m128i vaccGHIJKLMN = _mm_add_epi16(vxi0xGHIJKLMN, vxi1xGHIJKLMN);
    const __m128i vxi2xGHIJKLMN = _mm_srai_epi16(_mm_unpacklo_epi8(vi2xGHIJKLMN, vi2xGHIJKLMN), 8);
    const __m128i vi3xGHIJKLMN = _mm_loadl_epi64((const __m128i*) (i3 + 16));
    i3 += 24;

    vacc01234567 = _mm_add_epi16(vacc01234567, vxi2x01234567);
    const __m128i vxi3x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vi3x01234567, vi3x01234567), 8);
    const __m128i vi4x01234567 = _mm_loadl_epi64((const __m128i*) i4);
    vacc89ABCDEF = _mm_add_epi16(vacc89ABCDEF, vxi2x89ABCDEF);
    const __m128i vxi3x89ABCDEF = _mm_srai_epi16(_mm_unpacklo_epi8(vi3x89ABCDEF, vi3x89ABCDEF), 8);
    const __m128i vi4x89ABCDEF = _mm_loadl_epi64((const __m128i*) (i4 + 8));
    vaccGHIJKLMN = _mm_add_epi16(vaccGHIJKLMN, vxi2xGHIJKLMN);
    const __m128i vxi3xGHIJKLMN = _mm_srai_epi16(_mm_unpacklo_epi8(vi3xGHIJKLMN, vi3xGHIJKLMN), 8);
    const __m128i vi4xGHIJKLMN = _mm_loadl_epi64((const __m128i*) (i4 + 16));
    i4 += 24;

    vacc01234567 = _mm_add_epi16(vacc01234567, vxi3x01234567);
    const __m128i vxi4x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vi4x01234567, vi4x01234567), 8);
    const __m128i vi5x01234567 = _mm_loadl_epi64((const __m128i*) i5);
    vacc89ABCDEF = _mm_add_epi16(vacc89ABCDEF, vxi3x89ABCDEF);
    const __m128i vxi4x89ABCDEF = _mm_srai_epi16(_mm_unpacklo_epi8(vi4x89ABCDEF, vi4x89ABCDEF), 8);
    const __m128i vi5x89ABCDEF = _mm_loadl_epi64((const __m128i*) (i5 + 8));
    vaccGHIJKLMN = _mm_add_epi16(vaccGHIJKLMN, vxi3xGHIJKLMN);
    const __m128i vxi4xGHIJKLMN = _mm_srai_epi16(_mm_unpacklo_epi8(vi4xGHIJKLMN, vi4xGHIJKLMN), 8);
    const __m128i vi5xGHIJKLMN = _mm_loadl_epi64((const __m128i*) (i5 + 16));
    i5 += 24;

    vacc01234567 = _mm_add_epi16(vacc01234567, vxi4x01234567);
    const __m128i vxi5x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vi5x01234567, vi5x01234567), 8);
    const __m128i vi6x01234567 = _mm_loadl_epi64((const __m128i*) i6);
    vacc89ABCDEF = _mm_add_epi16(vacc89ABCDEF, vxi4x89ABCDEF);
    const __m128i vxi5x89ABCDEF = _mm_srai_epi16(_mm_unpacklo_epi8(vi5x89ABCDEF, vi5x89ABCDEF), 8);
    const __m128i vi6x89ABCDEF = _mm_loadl_epi64((const __m128i*) (i6 + 8));
    vaccGHIJKLMN = _mm_add_epi16(vaccGHIJKLMN, vxi4xGHIJKLMN);
    const __m128i vxi5xGHIJKLMN = _mm_srai_epi16(_mm_unpacklo_epi8(vi5xGHIJKLMN, vi5xGHIJKLMN), 8);
    const __m128i vi6xGHIJKLMN = _mm_loadl_epi64((const __m128i*) (i6 + 16));
    i6 += 24;

    vacc01234567 = _mm_add_epi16(vacc01234567, vxi5x01234567);
    const __m128i vxi6x01234567 = _mm_srai_epi16(_mm_unpacklo_epi8(vi6x01234567, vi6x01234567), 8);
    vacc89ABCDEF = _mm_add_epi16(vacc89ABCDEF, vxi5x89ABCDEF);
    const __m128i vxi6x89ABCDEF = _mm_srai_epi16(_mm_unpacklo_epi8(vi6x89ABCDEF, vi6x89ABCDEF), 8);
    vaccGHIJKLMN = _mm_add_epi16(vaccGHIJKLMN, vxi5xGHIJKLMN);
    const __m128i vxi6xGHIJKLMN = _mm_srai_epi16(_mm_unpacklo_epi8(vi6xGHIJKLMN, vi6xGHIJKLMN), 8);

    vacc01234567 = _mm_add_epi16(vacc01234567, vxi6x01234567);
    vacc89ABCDEF = _mm_add_epi16(vacc89ABCDEF, vxi6x89ABCDEF);
    vaccGHIJKLMN = _mm_add_epi16(vaccGHIJKLMN, vxi6xGHIJKLMN);

    const __m128i vsgnacc01234567 = _mm_cmpgt_epi16(_mm_setzero_si128(), vacc01234567);
    __m128i vacc0123 = _mm_unpacklo_epi16(vacc01234567, vsgnacc01234567);
    __m128i vacc4567 = _mm_unpackhi_epi16(vacc01234567, vsgnacc01234567);
    const __m128i vsgnacc89ABCDEF = _mm_cmpgt_epi16(_mm_setzero_si128(), vacc89ABCDEF);
    __m128i vacc89AB = _mm_unpacklo_epi16(vacc89ABCDEF, vsgnacc89ABCDEF);
    __m128i vaccCDEF = _mm_unpackhi_epi16(vacc89ABCDEF, vsgnacc89ABCDEF);
    const __m128i vsgnaccGHIJKLMN = _mm_cmpgt_epi16(_mm_setzero_si128(), vaccGHIJKLMN);
    __m128i vaccGHIJ = _mm_unpacklo_epi16(vaccGHIJKLMN, vsgnaccGHIJKLMN);
    __m128i vaccKLMN = _mm_unpackhi_epi16(vaccGHIJKLMN, vsgnaccGHIJKLMN);

    vacc0123 = _mm_add_epi32(vacc0123, _mm_load_si128((const __m128i*) buffer));
    vacc4567 = _mm_add_epi32(vacc4567, _mm_load_si128((const __m128i*) (buffer + 4)));
    vacc89AB = _mm_add_epi32(vacc89AB, _mm_load_si128((const __m128i*) (buffer + 8)));
    vaccCDEF = _mm_add_epi32(vaccCDEF, _mm_load_si128((const __m128i*) (buffer + 12)));
    vaccGHIJ = _mm_add_epi32(vaccGHIJ, _mm_load_si128((const __m128i*) (buffer + 16)));
    vaccKLMN = _mm_add_epi32(vaccKLMN, _mm_load_si128((const __m128i*) (buffer + 20)));
    buffer += 24;

    __m128 vfpacc0123 = _mm_cvtepi32_ps(vacc0123);
    __m128 vfpacc4567 = _mm_cvtepi32_ps(vacc4567);
    __m128 vfpacc89AB = _mm_cvtepi32_ps(vacc89AB);
    __m128 vfpaccCDEF = _mm_cvtepi32_ps(vaccCDEF);
    __m128 vfpaccGHIJ = _mm_cvtepi32_ps(vaccGHIJ);
    __m128 vfpaccKLMN = _mm_cvtepi32_ps(vaccKLMN);

    vfpacc0123 = _mm_mul_ps(vfpacc0123, vscale);
    vfpacc4567 = _mm_mul_ps(vfpacc4567, vscale);
    vfpacc89AB = _mm_mul_ps(vfpacc89AB, vscale);
    vfpaccCDEF = _mm_mul_ps(vfpaccCDEF, vscale);
    vfpaccGHIJ = _mm_mul_ps(vfpaccGHIJ, vscale);
    vfpaccKLMN = _mm_mul_ps(vfpaccKLMN, vscale);

    vfpacc0123 = _mm_min_ps(vfpacc0123, voutput_max_less_zero_point);
    vfpacc4567 = _mm_min_ps(vfpacc4567, voutput_max_less_zero_point);
    vfpacc89AB = _mm_min_ps(vfpacc89AB, voutput_max_less_zero_point);
    vfpaccCDEF = _mm_min_ps(vfpaccCDEF, voutput_max_less_zero_point);
    vfpaccGHIJ = _mm_min_ps(vfpaccGHIJ, voutput_max_less_zero_point);
    vfpaccKLMN = _mm_min_ps(vfpaccKLMN, voutput_max_less_zero_point);

    vacc0123 = _mm_cvtps_epi32(vfpacc0123);
    vacc4567 = _mm_cvtps_epi32(vfpacc4567);
    vacc89AB = _mm_cvtps_epi32(vfpacc89AB);
    vaccCDEF = _mm_cvtps_epi32(vfpaccCDEF);
    vaccGHIJ = _mm_cvtps_epi32(vfpaccGHIJ);
    vaccKLMN = _mm_cvtps_epi32(vfpaccKLMN);

    __m128i vout01234567 = _mm_adds_epi16(_mm_packs_epi32(vacc0123, vacc4567), voutput_zero_point);
    __m128i vout89ABCDEF = _mm_adds_epi16(_mm_packs_epi32(vacc89AB, vaccCDEF), voutput_zero_point);
    __m128i voutGHIJKLMN = _mm_adds_epi16(_mm_packs_epi32(vaccGHIJ, vaccKLMN), voutput_zero_point);

    vout01234567 = _mm_max_epi16(vout01234567, voutput_min);
    vout89ABCDEF = _mm_max_epi16(vout89ABCDEF, voutput_min);
    voutGHIJKLMN = _mm_max_epi16(voutGHIJKLMN, voutput_min);

    __m128i vout0123456789ABCDEF = _mm_packs_epi16(vout01234567, vout89ABCDEF);
    __m128i voutGHIJKLMNGHIJKLMN = _mm_packs_epi16(voutGHIJKLMN, voutGHIJKLMN);


    _mm_storeu_si128((__m128i*) output, vout0123456789ABCDEF);
    _mm_storel_epi64((__m128i*) (output + 16), voutGHIJKLMNGHIJKLMN);
    output += 24;
  }
  if XNN_UNLIKELY(channels != 0) {
    do {

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
      vout01234567 = _mm_max_epi16(vout01234567, voutput_min);

      __m128i vout0123456701234567 = _mm_packs_epi16(vout01234567, vout01234567);

      if XNN_LIKELY(channels >= 8) {
        _mm_storel_epi64((__m128i*) output, vout0123456701234567);
        output += 8;
        channels -= 8;
      } else {
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
          output += 1;
        }
        channels = 0;
      }
    } while (channels != 0);
  }
}
