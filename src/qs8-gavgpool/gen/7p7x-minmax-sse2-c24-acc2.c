// Auto-generated file. Do not edit!
//   Template: src/qs8-gavgpool/multipass-sse.c.in
//   Generator: tools/xngen
//
// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <emmintrin.h>

#include <xnnpack/gavgpool.h>
#include <xnnpack/math.h>


void xnn_qs8_gavgpool_minmax_ukernel_7p7x__sse2_c24_acc2(
    size_t rows,
    size_t channels,
    const int8_t* input,
    size_t input_stride,
    const int8_t* zero,
    int32_t* buffer,
    int8_t* output,
    const union xnn_qs8_avgpool_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_DISABLE_TSAN
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
  const size_t input_increment = 7 * input_stride - round_up_po2(channels, 8);

  const __m128i vbias = _mm_load_si128((const __m128i*) params->sse2.bias);
  int32_t* b = buffer;
  size_t c = channels;
  for (; c >= 24; c -= 24) {
    const __m128i vi0x01234567 = _mm_loadl_epi64((const __m128i*) i0);
    const __m128i vi0x89ABCDEF = _mm_loadl_epi64((const __m128i*) (i0 + 8));
    const __m128i vi0xGHIJKLMN = _mm_loadl_epi64((const __m128i*) (i0 + 16));
    i0 += 24;
    const __m128i vi1x01234567 = _mm_loadl_epi64((const __m128i*) i1);
    const __m128i vi1x89ABCDEF = _mm_loadl_epi64((const __m128i*) (i1 + 8));
    const __m128i vi1xGHIJKLMN = _mm_loadl_epi64((const __m128i*) (i1 + 16));
    i1 += 24;
    const __m128i vi2x01234567 = _mm_loadl_epi64((const __m128i*) i2);
    const __m128i vi2x89ABCDEF = _mm_loadl_epi64((const __m128i*) (i2 + 8));
    const __m128i vi2xGHIJKLMN = _mm_loadl_epi64((const __m128i*) (i2 + 16));
    i2 += 24;
    const __m128i vi3x01234567 = _mm_loadl_epi64((const __m128i*) i3);
    const __m128i vi3x89ABCDEF = _mm_loadl_epi64((const __m128i*) (i3 + 8));
    const __m128i vi3xGHIJKLMN = _mm_loadl_epi64((const __m128i*) (i3 + 16));
    i3 += 24;
    const __m128i vi4x01234567 = _mm_loadl_epi64((const __m128i*) i4);
    const __m128i vi4x89ABCDEF = _mm_loadl_epi64((const __m128i*) (i4 + 8));
    const __m128i vi4xGHIJKLMN = _mm_loadl_epi64((const __m128i*) (i4 + 16));
    i4 += 24;
    const __m128i vi5x01234567 = _mm_loadl_epi64((const __m128i*) i5);
    const __m128i vi5x89ABCDEF = _mm_loadl_epi64((const __m128i*) (i5 + 8));
    const __m128i vi5xGHIJKLMN = _mm_loadl_epi64((const __m128i*) (i5 + 16));
    i5 += 24;
    const __m128i vi6x01234567 = _mm_loadl_epi64((const __m128i*) i6);
    const __m128i vi6x89ABCDEF = _mm_loadl_epi64((const __m128i*) (i6 + 8));
    const __m128i vi6xGHIJKLMN = _mm_loadl_epi64((const __m128i*) (i6 + 16));
    i6 += 24;

    const __m128i vxi0x01234567 = _mm_unpacklo_epi8(vi0x01234567, _mm_cmpgt_epi8(_mm_setzero_si128(), vi0x01234567));
    const __m128i vxi0x89ABCDEF = _mm_unpacklo_epi8(vi0x89ABCDEF, _mm_cmpgt_epi8(_mm_setzero_si128(), vi0x89ABCDEF));
    const __m128i vxi0xGHIJKLMN = _mm_unpacklo_epi8(vi0xGHIJKLMN, _mm_cmpgt_epi8(_mm_setzero_si128(), vi0xGHIJKLMN));
    const __m128i vxi1x01234567 = _mm_unpacklo_epi8(vi1x01234567, _mm_cmpgt_epi8(_mm_setzero_si128(), vi1x01234567));
    const __m128i vxi1x89ABCDEF = _mm_unpacklo_epi8(vi1x89ABCDEF, _mm_cmpgt_epi8(_mm_setzero_si128(), vi1x89ABCDEF));
    const __m128i vxi1xGHIJKLMN = _mm_unpacklo_epi8(vi1xGHIJKLMN, _mm_cmpgt_epi8(_mm_setzero_si128(), vi1xGHIJKLMN));
    const __m128i vxi2x01234567 = _mm_unpacklo_epi8(vi2x01234567, _mm_cmpgt_epi8(_mm_setzero_si128(), vi2x01234567));
    const __m128i vxi2x89ABCDEF = _mm_unpacklo_epi8(vi2x89ABCDEF, _mm_cmpgt_epi8(_mm_setzero_si128(), vi2x89ABCDEF));
    const __m128i vxi2xGHIJKLMN = _mm_unpacklo_epi8(vi2xGHIJKLMN, _mm_cmpgt_epi8(_mm_setzero_si128(), vi2xGHIJKLMN));
    const __m128i vxi3x01234567 = _mm_unpacklo_epi8(vi3x01234567, _mm_cmpgt_epi8(_mm_setzero_si128(), vi3x01234567));
    const __m128i vxi3x89ABCDEF = _mm_unpacklo_epi8(vi3x89ABCDEF, _mm_cmpgt_epi8(_mm_setzero_si128(), vi3x89ABCDEF));
    const __m128i vxi3xGHIJKLMN = _mm_unpacklo_epi8(vi3xGHIJKLMN, _mm_cmpgt_epi8(_mm_setzero_si128(), vi3xGHIJKLMN));
    const __m128i vxi4x01234567 = _mm_unpacklo_epi8(vi4x01234567, _mm_cmpgt_epi8(_mm_setzero_si128(), vi4x01234567));
    const __m128i vxi4x89ABCDEF = _mm_unpacklo_epi8(vi4x89ABCDEF, _mm_cmpgt_epi8(_mm_setzero_si128(), vi4x89ABCDEF));
    const __m128i vxi4xGHIJKLMN = _mm_unpacklo_epi8(vi4xGHIJKLMN, _mm_cmpgt_epi8(_mm_setzero_si128(), vi4xGHIJKLMN));
    const __m128i vxi5x01234567 = _mm_unpacklo_epi8(vi5x01234567, _mm_cmpgt_epi8(_mm_setzero_si128(), vi5x01234567));
    const __m128i vxi5x89ABCDEF = _mm_unpacklo_epi8(vi5x89ABCDEF, _mm_cmpgt_epi8(_mm_setzero_si128(), vi5x89ABCDEF));
    const __m128i vxi5xGHIJKLMN = _mm_unpacklo_epi8(vi5xGHIJKLMN, _mm_cmpgt_epi8(_mm_setzero_si128(), vi5xGHIJKLMN));
    const __m128i vxi6x01234567 = _mm_unpacklo_epi8(vi6x01234567, _mm_cmpgt_epi8(_mm_setzero_si128(), vi6x01234567));
    const __m128i vxi6x89ABCDEF = _mm_unpacklo_epi8(vi6x89ABCDEF, _mm_cmpgt_epi8(_mm_setzero_si128(), vi6x89ABCDEF));
    const __m128i vxi6xGHIJKLMN = _mm_unpacklo_epi8(vi6xGHIJKLMN, _mm_cmpgt_epi8(_mm_setzero_si128(), vi6xGHIJKLMN));

    __m128i vacc0x01234567 = _mm_add_epi16(vxi0x01234567, vxi1x01234567);
    __m128i vacc0x89ABCDEF = _mm_add_epi16(vxi0x89ABCDEF, vxi1x89ABCDEF);
    __m128i vacc0xGHIJKLMN = _mm_add_epi16(vxi0xGHIJKLMN, vxi1xGHIJKLMN);
    __m128i vacc1x01234567 = _mm_add_epi16(vxi2x01234567, vxi3x01234567);
    __m128i vacc1x89ABCDEF = _mm_add_epi16(vxi2x89ABCDEF, vxi3x89ABCDEF);
    __m128i vacc1xGHIJKLMN = _mm_add_epi16(vxi2xGHIJKLMN, vxi3xGHIJKLMN);

    vacc0x01234567 = _mm_add_epi16(vacc0x01234567, vxi4x01234567);
    vacc0x89ABCDEF = _mm_add_epi16(vacc0x89ABCDEF, vxi4x89ABCDEF);
    vacc0xGHIJKLMN = _mm_add_epi16(vacc0xGHIJKLMN, vxi4xGHIJKLMN);
    vacc1x01234567 = _mm_add_epi16(vacc1x01234567, vxi5x01234567);
    vacc1x89ABCDEF = _mm_add_epi16(vacc1x89ABCDEF, vxi5x89ABCDEF);
    vacc1xGHIJKLMN = _mm_add_epi16(vacc1xGHIJKLMN, vxi5xGHIJKLMN);
    vacc0x01234567 = _mm_add_epi16(vacc0x01234567, vxi6x01234567);
    vacc0x89ABCDEF = _mm_add_epi16(vacc0x89ABCDEF, vxi6x89ABCDEF);
    vacc0xGHIJKLMN = _mm_add_epi16(vacc0xGHIJKLMN, vxi6xGHIJKLMN);

    // Add up all accumulators to vacc0x0123456789ABCDEFGHIJKLMN
    vacc0x01234567 = _mm_add_epi16(vacc0x01234567, vacc1x01234567);
    vacc0x89ABCDEF = _mm_add_epi16(vacc0x89ABCDEF, vacc1x89ABCDEF);
    vacc0xGHIJKLMN = _mm_add_epi16(vacc0xGHIJKLMN, vacc1xGHIJKLMN);

    const __m128i vsgnacc0x01234567 = _mm_cmpgt_epi16(_mm_setzero_si128(), vacc0x01234567);
    const __m128i vacc0123 = _mm_add_epi32(vbias, _mm_unpacklo_epi16(vacc0x01234567, vsgnacc0x01234567));
    const __m128i vacc4567 = _mm_add_epi32(vbias, _mm_unpackhi_epi16(vacc0x01234567, vsgnacc0x01234567));
    const __m128i vsgnacc0x89ABCDEF = _mm_cmpgt_epi16(_mm_setzero_si128(), vacc0x89ABCDEF);
    const __m128i vacc89AB = _mm_add_epi32(vbias, _mm_unpacklo_epi16(vacc0x89ABCDEF, vsgnacc0x89ABCDEF));
    const __m128i vaccCDEF = _mm_add_epi32(vbias, _mm_unpackhi_epi16(vacc0x89ABCDEF, vsgnacc0x89ABCDEF));
    const __m128i vsgnacc0xGHIJKLMN = _mm_cmpgt_epi16(_mm_setzero_si128(), vacc0xGHIJKLMN);
    const __m128i vaccGHIJ = _mm_add_epi32(vbias, _mm_unpacklo_epi16(vacc0xGHIJKLMN, vsgnacc0xGHIJKLMN));
    const __m128i vaccKLMN = _mm_add_epi32(vbias, _mm_unpackhi_epi16(vacc0xGHIJKLMN, vsgnacc0xGHIJKLMN));

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
      const __m128i vi2x01234567 = _mm_loadl_epi64((const __m128i*) i2);
      i2 += 8;
      const __m128i vi3x01234567 = _mm_loadl_epi64((const __m128i*) i3);
      i3 += 8;
      const __m128i vi4x01234567 = _mm_loadl_epi64((const __m128i*) i4);
      i4 += 8;
      const __m128i vi5x01234567 = _mm_loadl_epi64((const __m128i*) i5);
      i5 += 8;
      const __m128i vi6x01234567 = _mm_loadl_epi64((const __m128i*) i6);
      i6 += 8;

      const __m128i vxi0x01234567 = _mm_unpacklo_epi8(vi0x01234567, _mm_cmpgt_epi8(_mm_setzero_si128(), vi0x01234567));
      const __m128i vxi1x01234567 = _mm_unpacklo_epi8(vi1x01234567, _mm_cmpgt_epi8(_mm_setzero_si128(), vi1x01234567));
      const __m128i vxi2x01234567 = _mm_unpacklo_epi8(vi2x01234567, _mm_cmpgt_epi8(_mm_setzero_si128(), vi2x01234567));
      const __m128i vxi3x01234567 = _mm_unpacklo_epi8(vi3x01234567, _mm_cmpgt_epi8(_mm_setzero_si128(), vi3x01234567));
      const __m128i vxi4x01234567 = _mm_unpacklo_epi8(vi4x01234567, _mm_cmpgt_epi8(_mm_setzero_si128(), vi4x01234567));
      const __m128i vxi5x01234567 = _mm_unpacklo_epi8(vi5x01234567, _mm_cmpgt_epi8(_mm_setzero_si128(), vi5x01234567));
      const __m128i vxi6x01234567 = _mm_unpacklo_epi8(vi6x01234567, _mm_cmpgt_epi8(_mm_setzero_si128(), vi6x01234567));

      __m128i vacc0x01234567 = _mm_add_epi16(vxi0x01234567, vxi1x01234567);
      __m128i vacc1x01234567 = _mm_add_epi16(vxi2x01234567, vxi3x01234567);

      vacc0x01234567 = _mm_add_epi16(vacc0x01234567, vxi4x01234567);
      vacc1x01234567 = _mm_add_epi16(vacc1x01234567, vxi5x01234567);
      vacc0x01234567 = _mm_add_epi16(vacc0x01234567, vxi6x01234567);

      // Add up all accumulators to vacc0x01234567
      vacc0x01234567 = _mm_add_epi16(vacc0x01234567, vacc1x01234567);

      const __m128i vsgnacc0x01234567 = _mm_cmpgt_epi16(_mm_setzero_si128(), vacc0x01234567);
      const __m128i vacc0123 = _mm_add_epi32(vbias, _mm_unpacklo_epi16(vacc0x01234567, vsgnacc0x01234567));
      const __m128i vacc4567 = _mm_add_epi32(vbias, _mm_unpackhi_epi16(vacc0x01234567, vsgnacc0x01234567));

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
      const __m128i vi1x01234567 = _mm_loadl_epi64((const __m128i*) i1);
      const __m128i vi1x89ABCDEF = _mm_loadl_epi64((const __m128i*) (i1 + 8));
      const __m128i vi1xGHIJKLMN = _mm_loadl_epi64((const __m128i*) (i1 + 16));
      i1 += 24;
      const __m128i vi2x01234567 = _mm_loadl_epi64((const __m128i*) i2);
      const __m128i vi2x89ABCDEF = _mm_loadl_epi64((const __m128i*) (i2 + 8));
      const __m128i vi2xGHIJKLMN = _mm_loadl_epi64((const __m128i*) (i2 + 16));
      i2 += 24;
      const __m128i vi3x01234567 = _mm_loadl_epi64((const __m128i*) i3);
      const __m128i vi3x89ABCDEF = _mm_loadl_epi64((const __m128i*) (i3 + 8));
      const __m128i vi3xGHIJKLMN = _mm_loadl_epi64((const __m128i*) (i3 + 16));
      i3 += 24;
      const __m128i vi4x01234567 = _mm_loadl_epi64((const __m128i*) i4);
      const __m128i vi4x89ABCDEF = _mm_loadl_epi64((const __m128i*) (i4 + 8));
      const __m128i vi4xGHIJKLMN = _mm_loadl_epi64((const __m128i*) (i4 + 16));
      i4 += 24;
      const __m128i vi5x01234567 = _mm_loadl_epi64((const __m128i*) i5);
      const __m128i vi5x89ABCDEF = _mm_loadl_epi64((const __m128i*) (i5 + 8));
      const __m128i vi5xGHIJKLMN = _mm_loadl_epi64((const __m128i*) (i5 + 16));
      i5 += 24;
      const __m128i vi6x01234567 = _mm_loadl_epi64((const __m128i*) i6);
      const __m128i vi6x89ABCDEF = _mm_loadl_epi64((const __m128i*) (i6 + 8));
      const __m128i vi6xGHIJKLMN = _mm_loadl_epi64((const __m128i*) (i6 + 16));
      i6 += 24;

      const __m128i vxi0x01234567 = _mm_unpacklo_epi8(vi0x01234567, _mm_cmpgt_epi8(_mm_setzero_si128(), vi0x01234567));
      const __m128i vxi0x89ABCDEF = _mm_unpacklo_epi8(vi0x89ABCDEF, _mm_cmpgt_epi8(_mm_setzero_si128(), vi0x89ABCDEF));
      const __m128i vxi0xGHIJKLMN = _mm_unpacklo_epi8(vi0xGHIJKLMN, _mm_cmpgt_epi8(_mm_setzero_si128(), vi0xGHIJKLMN));
      const __m128i vxi1x01234567 = _mm_unpacklo_epi8(vi1x01234567, _mm_cmpgt_epi8(_mm_setzero_si128(), vi1x01234567));
      const __m128i vxi1x89ABCDEF = _mm_unpacklo_epi8(vi1x89ABCDEF, _mm_cmpgt_epi8(_mm_setzero_si128(), vi1x89ABCDEF));
      const __m128i vxi1xGHIJKLMN = _mm_unpacklo_epi8(vi1xGHIJKLMN, _mm_cmpgt_epi8(_mm_setzero_si128(), vi1xGHIJKLMN));
      const __m128i vxi2x01234567 = _mm_unpacklo_epi8(vi2x01234567, _mm_cmpgt_epi8(_mm_setzero_si128(), vi2x01234567));
      const __m128i vxi2x89ABCDEF = _mm_unpacklo_epi8(vi2x89ABCDEF, _mm_cmpgt_epi8(_mm_setzero_si128(), vi2x89ABCDEF));
      const __m128i vxi2xGHIJKLMN = _mm_unpacklo_epi8(vi2xGHIJKLMN, _mm_cmpgt_epi8(_mm_setzero_si128(), vi2xGHIJKLMN));
      const __m128i vxi3x01234567 = _mm_unpacklo_epi8(vi3x01234567, _mm_cmpgt_epi8(_mm_setzero_si128(), vi3x01234567));
      const __m128i vxi3x89ABCDEF = _mm_unpacklo_epi8(vi3x89ABCDEF, _mm_cmpgt_epi8(_mm_setzero_si128(), vi3x89ABCDEF));
      const __m128i vxi3xGHIJKLMN = _mm_unpacklo_epi8(vi3xGHIJKLMN, _mm_cmpgt_epi8(_mm_setzero_si128(), vi3xGHIJKLMN));
      const __m128i vxi4x01234567 = _mm_unpacklo_epi8(vi4x01234567, _mm_cmpgt_epi8(_mm_setzero_si128(), vi4x01234567));
      const __m128i vxi4x89ABCDEF = _mm_unpacklo_epi8(vi4x89ABCDEF, _mm_cmpgt_epi8(_mm_setzero_si128(), vi4x89ABCDEF));
      const __m128i vxi4xGHIJKLMN = _mm_unpacklo_epi8(vi4xGHIJKLMN, _mm_cmpgt_epi8(_mm_setzero_si128(), vi4xGHIJKLMN));
      const __m128i vxi5x01234567 = _mm_unpacklo_epi8(vi5x01234567, _mm_cmpgt_epi8(_mm_setzero_si128(), vi5x01234567));
      const __m128i vxi5x89ABCDEF = _mm_unpacklo_epi8(vi5x89ABCDEF, _mm_cmpgt_epi8(_mm_setzero_si128(), vi5x89ABCDEF));
      const __m128i vxi5xGHIJKLMN = _mm_unpacklo_epi8(vi5xGHIJKLMN, _mm_cmpgt_epi8(_mm_setzero_si128(), vi5xGHIJKLMN));
      const __m128i vxi6x01234567 = _mm_unpacklo_epi8(vi6x01234567, _mm_cmpgt_epi8(_mm_setzero_si128(), vi6x01234567));
      const __m128i vxi6x89ABCDEF = _mm_unpacklo_epi8(vi6x89ABCDEF, _mm_cmpgt_epi8(_mm_setzero_si128(), vi6x89ABCDEF));
      const __m128i vxi6xGHIJKLMN = _mm_unpacklo_epi8(vi6xGHIJKLMN, _mm_cmpgt_epi8(_mm_setzero_si128(), vi6xGHIJKLMN));

      __m128i vacc0x01234567 = _mm_add_epi16(vxi0x01234567, vxi1x01234567);
      __m128i vacc0x89ABCDEF = _mm_add_epi16(vxi0x89ABCDEF, vxi1x89ABCDEF);
      __m128i vacc0xGHIJKLMN = _mm_add_epi16(vxi0xGHIJKLMN, vxi1xGHIJKLMN);
      __m128i vacc1x01234567 = _mm_add_epi16(vxi2x01234567, vxi3x01234567);
      __m128i vacc1x89ABCDEF = _mm_add_epi16(vxi2x89ABCDEF, vxi3x89ABCDEF);
      __m128i vacc1xGHIJKLMN = _mm_add_epi16(vxi2xGHIJKLMN, vxi3xGHIJKLMN);

      vacc0x01234567 = _mm_add_epi16(vacc0x01234567, vxi4x01234567);
      vacc0x89ABCDEF = _mm_add_epi16(vacc0x89ABCDEF, vxi4x89ABCDEF);
      vacc0xGHIJKLMN = _mm_add_epi16(vacc0xGHIJKLMN, vxi4xGHIJKLMN);
      vacc1x01234567 = _mm_add_epi16(vacc1x01234567, vxi5x01234567);
      vacc1x89ABCDEF = _mm_add_epi16(vacc1x89ABCDEF, vxi5x89ABCDEF);
      vacc1xGHIJKLMN = _mm_add_epi16(vacc1xGHIJKLMN, vxi5xGHIJKLMN);
      vacc0x01234567 = _mm_add_epi16(vacc0x01234567, vxi6x01234567);
      vacc0x89ABCDEF = _mm_add_epi16(vacc0x89ABCDEF, vxi6x89ABCDEF);
      vacc0xGHIJKLMN = _mm_add_epi16(vacc0xGHIJKLMN, vxi6xGHIJKLMN);

      // Add up all accumulators to vacc0x0123456789ABCDEFGHIJKLMN
      vacc0x01234567 = _mm_add_epi16(vacc0x01234567, vacc1x01234567);
      vacc0x89ABCDEF = _mm_add_epi16(vacc0x89ABCDEF, vacc1x89ABCDEF);
      vacc0xGHIJKLMN = _mm_add_epi16(vacc0xGHIJKLMN, vacc1xGHIJKLMN);

      const __m128i vsgnacc0x01234567 = _mm_cmpgt_epi16(_mm_setzero_si128(), vacc0x01234567);
      const __m128i vacc0123 = _mm_add_epi32(_mm_unpacklo_epi16(vacc0x01234567, vsgnacc0x01234567), _mm_load_si128((const __m128i*) (b + 0)));
      const __m128i vacc4567 = _mm_add_epi32(_mm_unpackhi_epi16(vacc0x01234567, vsgnacc0x01234567), _mm_load_si128((const __m128i*) (b + 4)));
      const __m128i vsgnacc0x89ABCDEF = _mm_cmpgt_epi16(_mm_setzero_si128(), vacc0x89ABCDEF);
      const __m128i vacc89AB = _mm_add_epi32(_mm_unpacklo_epi16(vacc0x89ABCDEF, vsgnacc0x89ABCDEF), _mm_load_si128((const __m128i*) (b + 8)));
      const __m128i vaccCDEF = _mm_add_epi32(_mm_unpackhi_epi16(vacc0x89ABCDEF, vsgnacc0x89ABCDEF), _mm_load_si128((const __m128i*) (b + 12)));
      const __m128i vsgnacc0xGHIJKLMN = _mm_cmpgt_epi16(_mm_setzero_si128(), vacc0xGHIJKLMN);
      const __m128i vaccGHIJ = _mm_add_epi32(_mm_unpacklo_epi16(vacc0xGHIJKLMN, vsgnacc0xGHIJKLMN), _mm_load_si128((const __m128i*) (b + 16)));
      const __m128i vaccKLMN = _mm_add_epi32(_mm_unpackhi_epi16(vacc0xGHIJKLMN, vsgnacc0xGHIJKLMN), _mm_load_si128((const __m128i*) (b + 20)));

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
        const __m128i vi2x01234567 = _mm_loadl_epi64((const __m128i*) i2);
        i2 += 8;
        const __m128i vi3x01234567 = _mm_loadl_epi64((const __m128i*) i3);
        i3 += 8;
        const __m128i vi4x01234567 = _mm_loadl_epi64((const __m128i*) i4);
        i4 += 8;
        const __m128i vi5x01234567 = _mm_loadl_epi64((const __m128i*) i5);
        i5 += 8;
        const __m128i vi6x01234567 = _mm_loadl_epi64((const __m128i*) i6);
        i6 += 8;

        const __m128i vxi0x01234567 = _mm_unpacklo_epi8(vi0x01234567, _mm_cmpgt_epi8(_mm_setzero_si128(), vi0x01234567));
        const __m128i vxi1x01234567 = _mm_unpacklo_epi8(vi1x01234567, _mm_cmpgt_epi8(_mm_setzero_si128(), vi1x01234567));
        const __m128i vxi2x01234567 = _mm_unpacklo_epi8(vi2x01234567, _mm_cmpgt_epi8(_mm_setzero_si128(), vi2x01234567));
        const __m128i vxi3x01234567 = _mm_unpacklo_epi8(vi3x01234567, _mm_cmpgt_epi8(_mm_setzero_si128(), vi3x01234567));
        const __m128i vxi4x01234567 = _mm_unpacklo_epi8(vi4x01234567, _mm_cmpgt_epi8(_mm_setzero_si128(), vi4x01234567));
        const __m128i vxi5x01234567 = _mm_unpacklo_epi8(vi5x01234567, _mm_cmpgt_epi8(_mm_setzero_si128(), vi5x01234567));
        const __m128i vxi6x01234567 = _mm_unpacklo_epi8(vi6x01234567, _mm_cmpgt_epi8(_mm_setzero_si128(), vi6x01234567));

        __m128i vacc0x01234567 = _mm_add_epi16(vxi0x01234567, vxi1x01234567);
        __m128i vacc1x01234567 = _mm_add_epi16(vxi2x01234567, vxi3x01234567);

        vacc0x01234567 = _mm_add_epi16(vacc0x01234567, vxi4x01234567);
        vacc1x01234567 = _mm_add_epi16(vacc1x01234567, vxi5x01234567);
        vacc0x01234567 = _mm_add_epi16(vacc0x01234567, vxi6x01234567);

        // Add up all accumulators to vacc0x01234567
        vacc0x01234567 = _mm_add_epi16(vacc0x01234567, vacc1x01234567);

        const __m128i vsgnacc0x01234567 = _mm_cmpgt_epi16(_mm_setzero_si128(), vacc0x01234567);
        const __m128i vacc0123 = _mm_add_epi32(_mm_unpacklo_epi16(vacc0x01234567, vsgnacc0x01234567), _mm_load_si128((const __m128i*) b));
        const __m128i vacc4567 = _mm_add_epi32(_mm_unpackhi_epi16(vacc0x01234567, vsgnacc0x01234567), _mm_load_si128((const __m128i*) (b + 4)));

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

  const __m128i vmultiplier = _mm_load_si128((const __m128i*) params->sse2.multiplier);
  const __m128i vrounding = _mm_load_si128((const __m128i*) params->sse2.rounding);
  const __m128i vshift = _mm_loadl_epi64((const __m128i*) params->sse2.shift);
  while (channels >= 24) {
    const __m128i vi0x01234567 = _mm_loadl_epi64((const __m128i*) i0);
    const __m128i vi0x89ABCDEF = _mm_loadl_epi64((const __m128i*) (i0 + 8));
    const __m128i vi0xGHIJKLMN = _mm_loadl_epi64((const __m128i*) (i0 + 16));
    i0 += 24;
    const __m128i vi1x01234567 = _mm_loadl_epi64((const __m128i*) i1);
    const __m128i vi1x89ABCDEF = _mm_loadl_epi64((const __m128i*) (i1 + 8));
    const __m128i vi1xGHIJKLMN = _mm_loadl_epi64((const __m128i*) (i1 + 16));
    i1 += 24;
    const __m128i vi2x01234567 = _mm_loadl_epi64((const __m128i*) i2);
    const __m128i vi2x89ABCDEF = _mm_loadl_epi64((const __m128i*) (i2 + 8));
    const __m128i vi2xGHIJKLMN = _mm_loadl_epi64((const __m128i*) (i2 + 16));
    i2 += 24;
    const __m128i vi3x01234567 = _mm_loadl_epi64((const __m128i*) i3);
    const __m128i vi3x89ABCDEF = _mm_loadl_epi64((const __m128i*) (i3 + 8));
    const __m128i vi3xGHIJKLMN = _mm_loadl_epi64((const __m128i*) (i3 + 16));
    i3 += 24;
    const __m128i vi4x01234567 = _mm_loadl_epi64((const __m128i*) i4);
    const __m128i vi4x89ABCDEF = _mm_loadl_epi64((const __m128i*) (i4 + 8));
    const __m128i vi4xGHIJKLMN = _mm_loadl_epi64((const __m128i*) (i4 + 16));
    i4 += 24;
    const __m128i vi5x01234567 = _mm_loadl_epi64((const __m128i*) i5);
    const __m128i vi5x89ABCDEF = _mm_loadl_epi64((const __m128i*) (i5 + 8));
    const __m128i vi5xGHIJKLMN = _mm_loadl_epi64((const __m128i*) (i5 + 16));
    i5 += 24;
    const __m128i vi6x01234567 = _mm_loadl_epi64((const __m128i*) i6);
    const __m128i vi6x89ABCDEF = _mm_loadl_epi64((const __m128i*) (i6 + 8));
    const __m128i vi6xGHIJKLMN = _mm_loadl_epi64((const __m128i*) (i6 + 16));
    i6 += 24;

    const __m128i vxi0x01234567 = _mm_unpacklo_epi8(vi0x01234567, _mm_cmpgt_epi8(_mm_setzero_si128(), vi0x01234567));
    const __m128i vxi0x89ABCDEF = _mm_unpacklo_epi8(vi0x89ABCDEF, _mm_cmpgt_epi8(_mm_setzero_si128(), vi0x89ABCDEF));
    const __m128i vxi0xGHIJKLMN = _mm_unpacklo_epi8(vi0xGHIJKLMN, _mm_cmpgt_epi8(_mm_setzero_si128(), vi0xGHIJKLMN));
    const __m128i vxi1x01234567 = _mm_unpacklo_epi8(vi1x01234567, _mm_cmpgt_epi8(_mm_setzero_si128(), vi1x01234567));
    const __m128i vxi1x89ABCDEF = _mm_unpacklo_epi8(vi1x89ABCDEF, _mm_cmpgt_epi8(_mm_setzero_si128(), vi1x89ABCDEF));
    const __m128i vxi1xGHIJKLMN = _mm_unpacklo_epi8(vi1xGHIJKLMN, _mm_cmpgt_epi8(_mm_setzero_si128(), vi1xGHIJKLMN));
    const __m128i vxi2x01234567 = _mm_unpacklo_epi8(vi2x01234567, _mm_cmpgt_epi8(_mm_setzero_si128(), vi2x01234567));
    const __m128i vxi2x89ABCDEF = _mm_unpacklo_epi8(vi2x89ABCDEF, _mm_cmpgt_epi8(_mm_setzero_si128(), vi2x89ABCDEF));
    const __m128i vxi2xGHIJKLMN = _mm_unpacklo_epi8(vi2xGHIJKLMN, _mm_cmpgt_epi8(_mm_setzero_si128(), vi2xGHIJKLMN));
    const __m128i vxi3x01234567 = _mm_unpacklo_epi8(vi3x01234567, _mm_cmpgt_epi8(_mm_setzero_si128(), vi3x01234567));
    const __m128i vxi3x89ABCDEF = _mm_unpacklo_epi8(vi3x89ABCDEF, _mm_cmpgt_epi8(_mm_setzero_si128(), vi3x89ABCDEF));
    const __m128i vxi3xGHIJKLMN = _mm_unpacklo_epi8(vi3xGHIJKLMN, _mm_cmpgt_epi8(_mm_setzero_si128(), vi3xGHIJKLMN));
    const __m128i vxi4x01234567 = _mm_unpacklo_epi8(vi4x01234567, _mm_cmpgt_epi8(_mm_setzero_si128(), vi4x01234567));
    const __m128i vxi4x89ABCDEF = _mm_unpacklo_epi8(vi4x89ABCDEF, _mm_cmpgt_epi8(_mm_setzero_si128(), vi4x89ABCDEF));
    const __m128i vxi4xGHIJKLMN = _mm_unpacklo_epi8(vi4xGHIJKLMN, _mm_cmpgt_epi8(_mm_setzero_si128(), vi4xGHIJKLMN));
    const __m128i vxi5x01234567 = _mm_unpacklo_epi8(vi5x01234567, _mm_cmpgt_epi8(_mm_setzero_si128(), vi5x01234567));
    const __m128i vxi5x89ABCDEF = _mm_unpacklo_epi8(vi5x89ABCDEF, _mm_cmpgt_epi8(_mm_setzero_si128(), vi5x89ABCDEF));
    const __m128i vxi5xGHIJKLMN = _mm_unpacklo_epi8(vi5xGHIJKLMN, _mm_cmpgt_epi8(_mm_setzero_si128(), vi5xGHIJKLMN));
    const __m128i vxi6x01234567 = _mm_unpacklo_epi8(vi6x01234567, _mm_cmpgt_epi8(_mm_setzero_si128(), vi6x01234567));
    const __m128i vxi6x89ABCDEF = _mm_unpacklo_epi8(vi6x89ABCDEF, _mm_cmpgt_epi8(_mm_setzero_si128(), vi6x89ABCDEF));
    const __m128i vxi6xGHIJKLMN = _mm_unpacklo_epi8(vi6xGHIJKLMN, _mm_cmpgt_epi8(_mm_setzero_si128(), vi6xGHIJKLMN));

    __m128i vacc0x01234567 = _mm_add_epi16(vxi0x01234567, vxi1x01234567);
    __m128i vacc0x89ABCDEF = _mm_add_epi16(vxi0x89ABCDEF, vxi1x89ABCDEF);
    __m128i vacc0xGHIJKLMN = _mm_add_epi16(vxi0xGHIJKLMN, vxi1xGHIJKLMN);
    __m128i vacc1x01234567 = _mm_add_epi16(vxi2x01234567, vxi3x01234567);
    __m128i vacc1x89ABCDEF = _mm_add_epi16(vxi2x89ABCDEF, vxi3x89ABCDEF);
    __m128i vacc1xGHIJKLMN = _mm_add_epi16(vxi2xGHIJKLMN, vxi3xGHIJKLMN);

    vacc0x01234567 = _mm_add_epi16(vacc0x01234567, vxi4x01234567);
    vacc0x89ABCDEF = _mm_add_epi16(vacc0x89ABCDEF, vxi4x89ABCDEF);
    vacc0xGHIJKLMN = _mm_add_epi16(vacc0xGHIJKLMN, vxi4xGHIJKLMN);
    vacc1x01234567 = _mm_add_epi16(vacc1x01234567, vxi5x01234567);
    vacc1x89ABCDEF = _mm_add_epi16(vacc1x89ABCDEF, vxi5x89ABCDEF);
    vacc1xGHIJKLMN = _mm_add_epi16(vacc1xGHIJKLMN, vxi5xGHIJKLMN);
    vacc0x01234567 = _mm_add_epi16(vacc0x01234567, vxi6x01234567);
    vacc0x89ABCDEF = _mm_add_epi16(vacc0x89ABCDEF, vxi6x89ABCDEF);
    vacc0xGHIJKLMN = _mm_add_epi16(vacc0xGHIJKLMN, vxi6xGHIJKLMN);

    // Add up all accumulators to vacc0x0123456789ABCDEFGHIJKLMN
    vacc0x01234567 = _mm_add_epi16(vacc0x01234567, vacc1x01234567);
    vacc0x89ABCDEF = _mm_add_epi16(vacc0x89ABCDEF, vacc1x89ABCDEF);
    vacc0xGHIJKLMN = _mm_add_epi16(vacc0xGHIJKLMN, vacc1xGHIJKLMN);

    const __m128i vsgnacc0x01234567 = _mm_cmpgt_epi16(_mm_setzero_si128(), vacc0x01234567);
    const __m128i vacc0123 = _mm_add_epi32(_mm_unpacklo_epi16(vacc0x01234567, vsgnacc0x01234567), _mm_load_si128((const __m128i*) (buffer + 0)));
    const __m128i vacc4567 = _mm_add_epi32(_mm_unpackhi_epi16(vacc0x01234567, vsgnacc0x01234567), _mm_load_si128((const __m128i*) (buffer + 4)));
    const __m128i vsgnacc0x89ABCDEF = _mm_cmpgt_epi16(_mm_setzero_si128(), vacc0x89ABCDEF);
    const __m128i vacc89AB = _mm_add_epi32(_mm_unpacklo_epi16(vacc0x89ABCDEF, vsgnacc0x89ABCDEF), _mm_load_si128((const __m128i*) (buffer + 8)));
    const __m128i vaccCDEF = _mm_add_epi32(_mm_unpackhi_epi16(vacc0x89ABCDEF, vsgnacc0x89ABCDEF), _mm_load_si128((const __m128i*) (buffer + 12)));
    const __m128i vsgnacc0xGHIJKLMN = _mm_cmpgt_epi16(_mm_setzero_si128(), vacc0xGHIJKLMN);
    const __m128i vaccGHIJ = _mm_add_epi32(_mm_unpacklo_epi16(vacc0xGHIJKLMN, vsgnacc0xGHIJKLMN), _mm_load_si128((const __m128i*) (buffer + 16)));
    const __m128i vaccKLMN = _mm_add_epi32(_mm_unpackhi_epi16(vacc0xGHIJKLMN, vsgnacc0xGHIJKLMN), _mm_load_si128((const __m128i*) (buffer + 20)));
    buffer += 24;

    const __m128i vsgnacc0123 = _mm_cmpgt_epi32(_mm_setzero_si128(), vacc0123);
    const __m128i vsgnacc4567 = _mm_cmpgt_epi32(_mm_setzero_si128(), vacc4567);
    const __m128i vsgnacc89AB = _mm_cmpgt_epi32(_mm_setzero_si128(), vacc89AB);
    const __m128i vsgnaccCDEF = _mm_cmpgt_epi32(_mm_setzero_si128(), vaccCDEF);
    const __m128i vsgnaccGHIJ = _mm_cmpgt_epi32(_mm_setzero_si128(), vaccGHIJ);
    const __m128i vsgnaccKLMN = _mm_cmpgt_epi32(_mm_setzero_si128(), vaccKLMN);

    const __m128i vabsacc0123 = _mm_sub_epi32(_mm_xor_si128(vacc0123, vsgnacc0123), vsgnacc0123);
    const __m128i vabsacc4567 = _mm_sub_epi32(_mm_xor_si128(vacc4567, vsgnacc4567), vsgnacc4567);
    const __m128i vabsacc89AB = _mm_sub_epi32(_mm_xor_si128(vacc89AB, vsgnacc89AB), vsgnacc89AB);
    const __m128i vabsaccCDEF = _mm_sub_epi32(_mm_xor_si128(vaccCDEF, vsgnaccCDEF), vsgnaccCDEF);
    const __m128i vabsaccGHIJ = _mm_sub_epi32(_mm_xor_si128(vaccGHIJ, vsgnaccGHIJ), vsgnaccGHIJ);
    const __m128i vabsaccKLMN = _mm_sub_epi32(_mm_xor_si128(vaccKLMN, vsgnaccKLMN), vsgnaccKLMN);

    const __m128i vabsacc13 = _mm_shuffle_epi32(vabsacc0123, _MM_SHUFFLE(3, 3, 1, 1));
    const __m128i vabsacc57 = _mm_shuffle_epi32(vabsacc4567, _MM_SHUFFLE(3, 3, 1, 1));
    const __m128i vabsacc9B = _mm_shuffle_epi32(vabsacc89AB, _MM_SHUFFLE(3, 3, 1, 1));
    const __m128i vabsaccDF = _mm_shuffle_epi32(vabsaccCDEF, _MM_SHUFFLE(3, 3, 1, 1));
    const __m128i vabsaccHJ = _mm_shuffle_epi32(vabsaccGHIJ, _MM_SHUFFLE(3, 3, 1, 1));
    const __m128i vabsaccLN = _mm_shuffle_epi32(vabsaccKLMN, _MM_SHUFFLE(3, 3, 1, 1));

    const __m128i vabsprod02 = _mm_mul_epu32(vabsacc0123, vmultiplier);
    const __m128i vabsprod13 = _mm_mul_epu32(vabsacc13, vmultiplier);
    const __m128i vabsprod46 = _mm_mul_epu32(vabsacc4567, vmultiplier);
    const __m128i vabsprod57 = _mm_mul_epu32(vabsacc57, vmultiplier);
    const __m128i vabsprod8A = _mm_mul_epu32(vabsacc89AB, vmultiplier);
    const __m128i vabsprod9B = _mm_mul_epu32(vabsacc9B, vmultiplier);
    const __m128i vabsprodCE = _mm_mul_epu32(vabsaccCDEF, vmultiplier);
    const __m128i vabsprodDF = _mm_mul_epu32(vabsaccDF, vmultiplier);
    const __m128i vabsprodGI = _mm_mul_epu32(vabsaccGHIJ, vmultiplier);
    const __m128i vabsprodHJ = _mm_mul_epu32(vabsaccHJ, vmultiplier);
    const __m128i vabsprodKM = _mm_mul_epu32(vabsaccKLMN, vmultiplier);
    const __m128i vabsprodLN = _mm_mul_epu32(vabsaccLN, vmultiplier);

    const __m128i vabsout02 = _mm_srl_epi64(_mm_add_epi64(vabsprod02, vrounding), vshift);
    const __m128i vabsout13 = _mm_srl_epi64(_mm_add_epi64(vabsprod13, vrounding), vshift);
    const __m128i vabsout46 = _mm_srl_epi64(_mm_add_epi64(vabsprod46, vrounding), vshift);
    const __m128i vabsout57 = _mm_srl_epi64(_mm_add_epi64(vabsprod57, vrounding), vshift);
    const __m128i vabsout8A = _mm_srl_epi64(_mm_add_epi64(vabsprod8A, vrounding), vshift);
    const __m128i vabsout9B = _mm_srl_epi64(_mm_add_epi64(vabsprod9B, vrounding), vshift);
    const __m128i vabsoutCE = _mm_srl_epi64(_mm_add_epi64(vabsprodCE, vrounding), vshift);
    const __m128i vabsoutDF = _mm_srl_epi64(_mm_add_epi64(vabsprodDF, vrounding), vshift);
    const __m128i vabsoutGI = _mm_srl_epi64(_mm_add_epi64(vabsprodGI, vrounding), vshift);
    const __m128i vabsoutHJ = _mm_srl_epi64(_mm_add_epi64(vabsprodHJ, vrounding), vshift);
    const __m128i vabsoutKM = _mm_srl_epi64(_mm_add_epi64(vabsprodKM, vrounding), vshift);
    const __m128i vabsoutLN = _mm_srl_epi64(_mm_add_epi64(vabsprodLN, vrounding), vshift);

    const __m128i vabsout0213 = _mm_castps_si128(
        _mm_shuffle_ps(_mm_castsi128_ps(vabsout02), _mm_castsi128_ps(vabsout13), _MM_SHUFFLE(2, 0, 2, 0)));
    const __m128i vabsout4657 = _mm_castps_si128(
        _mm_shuffle_ps(_mm_castsi128_ps(vabsout46), _mm_castsi128_ps(vabsout57), _MM_SHUFFLE(2, 0, 2, 0)));
    const __m128i vabsout8A9B = _mm_castps_si128(
        _mm_shuffle_ps(_mm_castsi128_ps(vabsout8A), _mm_castsi128_ps(vabsout9B), _MM_SHUFFLE(2, 0, 2, 0)));
    const __m128i vabsoutCEDF = _mm_castps_si128(
        _mm_shuffle_ps(_mm_castsi128_ps(vabsoutCE), _mm_castsi128_ps(vabsoutDF), _MM_SHUFFLE(2, 0, 2, 0)));
    const __m128i vabsoutGIHJ = _mm_castps_si128(
        _mm_shuffle_ps(_mm_castsi128_ps(vabsoutGI), _mm_castsi128_ps(vabsoutHJ), _MM_SHUFFLE(2, 0, 2, 0)));
    const __m128i vabsoutKMLN = _mm_castps_si128(
        _mm_shuffle_ps(_mm_castsi128_ps(vabsoutKM), _mm_castsi128_ps(vabsoutLN), _MM_SHUFFLE(2, 0, 2, 0)));

    const __m128i vabsout0123 = _mm_shuffle_epi32(vabsout0213, _MM_SHUFFLE(3, 1, 2, 0));
    const __m128i vabsout4567 = _mm_shuffle_epi32(vabsout4657, _MM_SHUFFLE(3, 1, 2, 0));
    const __m128i vabsout89AB = _mm_shuffle_epi32(vabsout8A9B, _MM_SHUFFLE(3, 1, 2, 0));
    const __m128i vabsoutCDEF = _mm_shuffle_epi32(vabsoutCEDF, _MM_SHUFFLE(3, 1, 2, 0));
    const __m128i vabsoutGHIJ = _mm_shuffle_epi32(vabsoutGIHJ, _MM_SHUFFLE(3, 1, 2, 0));
    const __m128i vabsoutKLMN = _mm_shuffle_epi32(vabsoutKMLN, _MM_SHUFFLE(3, 1, 2, 0));

    const __m128i vout0123 = _mm_sub_epi32(_mm_xor_si128(vabsout0123, vsgnacc0123), vsgnacc0123);
    const __m128i vout4567 = _mm_sub_epi32(_mm_xor_si128(vabsout4567, vsgnacc4567), vsgnacc4567);
    const __m128i vout89AB = _mm_sub_epi32(_mm_xor_si128(vabsout89AB, vsgnacc89AB), vsgnacc89AB);
    const __m128i voutCDEF = _mm_sub_epi32(_mm_xor_si128(vabsoutCDEF, vsgnaccCDEF), vsgnaccCDEF);
    const __m128i voutGHIJ = _mm_sub_epi32(_mm_xor_si128(vabsoutGHIJ, vsgnaccGHIJ), vsgnaccGHIJ);
    const __m128i voutKLMN = _mm_sub_epi32(_mm_xor_si128(vabsoutKLMN, vsgnaccKLMN), vsgnaccKLMN);

    const __m128i voutput_zero_point = _mm_load_si128((const __m128i*) params->sse2.output_zero_point);
    __m128i vout01234567 = _mm_adds_epi16(_mm_packs_epi32(vout0123, vout4567), voutput_zero_point);
    __m128i vout89ABCDEF = _mm_adds_epi16(_mm_packs_epi32(vout89AB, voutCDEF), voutput_zero_point);
    __m128i voutGHIJKLMN = _mm_adds_epi16(_mm_packs_epi32(voutGHIJ, voutKLMN), voutput_zero_point);

    const __m128i voutput_min = _mm_load_si128((const __m128i*) params->sse2.output_min);
    const __m128i voutput_max = _mm_load_si128((const __m128i*) params->sse2.output_max);
    vout01234567 = _mm_min_epi16(_mm_max_epi16(vout01234567, voutput_min), voutput_max);
    vout89ABCDEF = _mm_min_epi16(_mm_max_epi16(vout89ABCDEF, voutput_min), voutput_max);
    voutGHIJKLMN = _mm_min_epi16(_mm_max_epi16(voutGHIJKLMN, voutput_min), voutput_max);

    __m128i vout0123456789ABCDEF = _mm_packs_epi16(vout01234567, vout89ABCDEF);
    __m128i voutGHIJKLMNGHIJKLMN = _mm_packs_epi16(voutGHIJKLMN, voutGHIJKLMN);

    _mm_storeu_si128((__m128i*) output, vout0123456789ABCDEF);
    _mm_storel_epi64((__m128i*) (output + 16), voutGHIJKLMNGHIJKLMN);
    output += 24;

    channels -= 24;
  }
  if XNN_UNLIKELY(channels != 0) {
    do {
      const __m128i vi0x01234567 = _mm_loadl_epi64((const __m128i*) i0);
      i0 += 8;
      const __m128i vi1x01234567 = _mm_loadl_epi64((const __m128i*) i1);
      i1 += 8;
      const __m128i vi2x01234567 = _mm_loadl_epi64((const __m128i*) i2);
      i2 += 8;
      const __m128i vi3x01234567 = _mm_loadl_epi64((const __m128i*) i3);
      i3 += 8;
      const __m128i vi4x01234567 = _mm_loadl_epi64((const __m128i*) i4);
      i4 += 8;
      const __m128i vi5x01234567 = _mm_loadl_epi64((const __m128i*) i5);
      i5 += 8;
      const __m128i vi6x01234567 = _mm_loadl_epi64((const __m128i*) i6);
      i6 += 8;

      const __m128i vxi0x01234567 = _mm_unpacklo_epi8(vi0x01234567, _mm_cmpgt_epi8(_mm_setzero_si128(), vi0x01234567));
      const __m128i vxi1x01234567 = _mm_unpacklo_epi8(vi1x01234567, _mm_cmpgt_epi8(_mm_setzero_si128(), vi1x01234567));
      const __m128i vxi2x01234567 = _mm_unpacklo_epi8(vi2x01234567, _mm_cmpgt_epi8(_mm_setzero_si128(), vi2x01234567));
      const __m128i vxi3x01234567 = _mm_unpacklo_epi8(vi3x01234567, _mm_cmpgt_epi8(_mm_setzero_si128(), vi3x01234567));
      const __m128i vxi4x01234567 = _mm_unpacklo_epi8(vi4x01234567, _mm_cmpgt_epi8(_mm_setzero_si128(), vi4x01234567));
      const __m128i vxi5x01234567 = _mm_unpacklo_epi8(vi5x01234567, _mm_cmpgt_epi8(_mm_setzero_si128(), vi5x01234567));
      const __m128i vxi6x01234567 = _mm_unpacklo_epi8(vi6x01234567, _mm_cmpgt_epi8(_mm_setzero_si128(), vi6x01234567));

      __m128i vacc0x01234567 = _mm_add_epi16(vxi0x01234567, vxi1x01234567);
      __m128i vacc1x01234567 = _mm_add_epi16(vxi2x01234567, vxi3x01234567);

      vacc0x01234567 = _mm_add_epi16(vacc0x01234567, vxi4x01234567);
      vacc1x01234567 = _mm_add_epi16(vacc1x01234567, vxi5x01234567);
      vacc0x01234567 = _mm_add_epi16(vacc0x01234567, vxi6x01234567);

      // Add up all accumulators to vacc0x01234567
      vacc0x01234567 = _mm_add_epi16(vacc0x01234567, vacc1x01234567);

      const __m128i vsgnacc0x01234567 = _mm_cmpgt_epi16(_mm_setzero_si128(), vacc0x01234567);
      const __m128i vacc0123 = _mm_add_epi32(_mm_unpacklo_epi16(vacc0x01234567, vsgnacc0x01234567), _mm_load_si128((const __m128i*) buffer));
      const __m128i vacc4567 = _mm_add_epi32(_mm_unpackhi_epi16(vacc0x01234567, vsgnacc0x01234567), _mm_load_si128((const __m128i*) (buffer + 4)));
      buffer += 8;

      const __m128i vsgnacc0123 = _mm_cmpgt_epi32(_mm_setzero_si128(), vacc0123);
      const __m128i vsgnacc4567 = _mm_cmpgt_epi32(_mm_setzero_si128(), vacc4567);

      const __m128i vabsacc0123 = _mm_sub_epi32(_mm_xor_si128(vacc0123, vsgnacc0123), vsgnacc0123);
      const __m128i vabsacc4567 = _mm_sub_epi32(_mm_xor_si128(vacc4567, vsgnacc4567), vsgnacc4567);

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

      const __m128i vabsout0213 = _mm_castps_si128(
          _mm_shuffle_ps(_mm_castsi128_ps(vabsout02), _mm_castsi128_ps(vabsout13), _MM_SHUFFLE(2, 0, 2, 0)));
      const __m128i vabsout4657 = _mm_castps_si128(
          _mm_shuffle_ps(_mm_castsi128_ps(vabsout46), _mm_castsi128_ps(vabsout57), _MM_SHUFFLE(2, 0, 2, 0)));

      const __m128i vabsout0123 = _mm_shuffle_epi32(vabsout0213, _MM_SHUFFLE(3, 1, 2, 0));
      const __m128i vabsout4567 = _mm_shuffle_epi32(vabsout4657, _MM_SHUFFLE(3, 1, 2, 0));

      const __m128i vout0123 = _mm_sub_epi32(_mm_xor_si128(vabsout0123, vsgnacc0123), vsgnacc0123);
      const __m128i vout4567 = _mm_sub_epi32(_mm_xor_si128(vabsout4567, vsgnacc4567), vsgnacc4567);

      const __m128i voutput_zero_point = _mm_load_si128((const __m128i*) params->sse2.output_zero_point);
      __m128i vout01234567 = _mm_adds_epi16(_mm_packs_epi32(vout0123, vout4567), voutput_zero_point);

      const __m128i voutput_min = _mm_load_si128((const __m128i*) params->sse2.output_min);
      const __m128i voutput_max = _mm_load_si128((const __m128i*) params->sse2.output_max);
      vout01234567 = _mm_min_epi16(_mm_max_epi16(vout01234567, voutput_min), voutput_max);

      __m128i vout0123456701234567 = _mm_packs_epi16(vout01234567, vout01234567);

      if XNN_LIKELY(channels >= 8) {
        _mm_storel_epi64((__m128i*) output, vout0123456701234567);
        output += 8;
        channels -= 8;
      } else {
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
          *output = (int32_t) _mm_cvtsi128_si32(vout0123456701234567);
          output += 1;
        }
        channels = 0;
      }
    } while (channels != 0);
  }
}
