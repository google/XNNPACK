// Auto-generated file. Do not edit!
//   Template: src/s8-ibilinear/sse.c.in
//   Generator: tools/xngen
//
// Copyright 2021 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <emmintrin.h>

#include "xnnpack/common.h"
#include "xnnpack/ibilinear.h"
#include "xnnpack/unaligned.h"


void xnn_u8_ibilinear_ukernel__sse2_c8(
    size_t output_pixels,
    size_t channels,
    const uint8_t** restrict input,
    size_t input_offset,
    const int16_t* restrict weights,
    uint8_t* restrict output,
    size_t output_increment) XNN_OOB_READS
{
  assert(output_pixels != 0);
  assert(channels != 0);

  do {
    const uint8_t* i0 = (const uint8_t*) ((uintptr_t) input[0] + input_offset);
    const uint8_t* i1 = (const uint8_t*) ((uintptr_t) input[1] + input_offset);
    const uint8_t* i2 = (const uint8_t*) ((uintptr_t) input[2] + input_offset);
    const uint8_t* i3 = (const uint8_t*) ((uintptr_t) input[3] + input_offset);
    input += 4;

    const __m128i valpha = _mm_cvtsi32_si128(*((const int*) weights));
    weights += 2;
    __m128i valphah = _mm_shufflelo_epi16(valpha, _MM_SHUFFLE(0, 0, 0, 0));
    valphah = _mm_unpacklo_epi64(valphah, valphah);
    __m128i valphav = _mm_shufflelo_epi16(valpha, _MM_SHUFFLE(1, 1, 1, 1));
    valphav = _mm_unpacklo_epi64(valphav, valphav);

    valphah = _mm_xor_si128(valphah, _mm_set1_epi32(0xFFFF0000));
    valphah = _mm_add_epi16(valphah, _mm_set1_epi32(0x08010000));

    const __m128i vrounding = _mm_set1_epi32(0x00200000);

    size_t c = channels;
    for (; c >= 8 * sizeof(uint8_t); c -= 8 * sizeof(uint8_t)) {
      __m128i vtl01234567 = _mm_loadl_epi64((const __m128i*) i0);
      i0 += 8;
      __m128i vtr01234567 = _mm_loadl_epi64((const __m128i*) i1);
      i1 += 8;
      __m128i vbl01234567 = _mm_loadl_epi64((const __m128i*) i2);
      i2 += 8;
      __m128i vbr01234567 = _mm_loadl_epi64((const __m128i*) i3);
      i3 += 8;

      __m128i vzero = _mm_setzero_si128();
      vtl01234567 = _mm_unpacklo_epi8(vtl01234567, vzero);
      vtr01234567 = _mm_unpacklo_epi8(vtr01234567, vzero);
      vbl01234567 = _mm_unpacklo_epi8(vbl01234567, vzero);
      vbr01234567 = _mm_unpacklo_epi8(vbr01234567, vzero);

      const __m128i vdr01234567 = _mm_sub_epi16(vbr01234567, vtr01234567);
      const __m128i vt0123 = _mm_madd_epi16(_mm_unpacklo_epi16(vtr01234567, vtl01234567), valphah);
      const __m128i vdl01234567 = _mm_sub_epi16(vbl01234567, vtl01234567);
      const __m128i vt4567 = _mm_madd_epi16(_mm_unpackhi_epi16(vtr01234567, vtl01234567), valphah);

      const __m128i vd0123 = _mm_madd_epi16(_mm_unpacklo_epi16(vdr01234567, vdl01234567), valphah);
      const __m128i vd4567 = _mm_madd_epi16(_mm_unpackhi_epi16(vdr01234567, vdl01234567), valphah);

      __m128i vacc0123 = _mm_slli_epi32(_mm_mulhi_epu16(vd0123, valphav), 16);
      __m128i vacc4567 = _mm_slli_epi32(_mm_mulhi_epu16(vd4567, valphav), 16);

      vacc0123 = _mm_add_epi16(_mm_mullo_epi16(vd0123, valphav), vacc0123);
      vacc4567 = _mm_add_epi16(_mm_mullo_epi16(vd4567, valphav), vacc4567);

      vacc0123 = _mm_add_epi32(_mm_slli_epi32(vt0123, 11), vacc0123);
      vacc4567 = _mm_add_epi32(_mm_slli_epi32(vt4567, 11), vacc4567);

      vacc0123 = _mm_srli_epi32(_mm_add_epi16(vacc0123, vrounding), 22);
      vacc4567 = _mm_srli_epi32(_mm_add_epi16(vacc4567, vrounding), 22);

      const __m128i vacc01234567 = _mm_packs_epi32(vacc0123, vacc4567);

      const __m128i vo01234567 = _mm_packus_epi16(vacc01234567, vacc01234567);

      _mm_storel_epi64((__m128i*) output, vo01234567);
      output += 8;
    }
    if XNN_UNLIKELY(c != 0) {
      __m128i vtl01234567 = _mm_loadl_epi64((const __m128i*) i0);
      __m128i vtr01234567 = _mm_loadl_epi64((const __m128i*) i1);
      __m128i vbl01234567 = _mm_loadl_epi64((const __m128i*) i2);
      __m128i vbr01234567 = _mm_loadl_epi64((const __m128i*) i3);

      __m128i vzero = _mm_setzero_si128();
      vtl01234567 = _mm_unpacklo_epi8(vtl01234567, vzero);
      vtr01234567 = _mm_unpacklo_epi8(vtr01234567, vzero);
      vbl01234567 = _mm_unpacklo_epi8(vbl01234567, vzero);
      vbr01234567 = _mm_unpacklo_epi8(vbr01234567, vzero);

      const __m128i vdr01234567 = _mm_sub_epi16(vbr01234567, vtr01234567);
      const __m128i vt0123 = _mm_madd_epi16(_mm_unpacklo_epi16(vtr01234567, vtl01234567), valphah);
      const __m128i vdl01234567 = _mm_sub_epi16(vbl01234567, vtl01234567);
      const __m128i vt4567 = _mm_madd_epi16(_mm_unpackhi_epi16(vtr01234567, vtl01234567), valphah);

      const __m128i vd0123 = _mm_madd_epi16(_mm_unpacklo_epi16(vdr01234567, vdl01234567), valphah);
      const __m128i vd4567 = _mm_madd_epi16(_mm_unpackhi_epi16(vdr01234567, vdl01234567), valphah);

      __m128i vacc0123 = _mm_slli_epi32(_mm_mulhi_epu16(vd0123, valphav), 16);
      __m128i vacc4567 = _mm_slli_epi32(_mm_mulhi_epu16(vd4567, valphav), 16);

      vacc0123 = _mm_add_epi16(_mm_mullo_epi16(vd0123, valphav), vacc0123);
      vacc4567 = _mm_add_epi16(_mm_mullo_epi16(vd4567, valphav), vacc4567);

      vacc0123 = _mm_add_epi32(_mm_slli_epi32(vt0123, 11), vacc0123);
      vacc4567 = _mm_add_epi32(_mm_slli_epi32(vt4567, 11), vacc4567);

      vacc0123 = _mm_srli_epi32(_mm_add_epi16(vacc0123, vrounding), 22);
      vacc4567 = _mm_srli_epi32(_mm_add_epi16(vacc4567, vrounding), 22);

      const __m128i vacc01234567 = _mm_packs_epi32(vacc0123, vacc4567);

      __m128i vo01234567 = _mm_packus_epi16(vacc01234567, vacc01234567);

      if (c & (4 * sizeof(uint8_t))) {
        unaligned_store_u32(output, (uint32_t) _mm_cvtsi128_si32(vo01234567));
        output += 4;
        vo01234567 = _mm_srli_epi64(vo01234567, 32);
      }
      uint32_t vo0123 = (uint32_t) _mm_cvtsi128_si32(vo01234567);
      if (c & (2 * sizeof(uint8_t))) {
        unaligned_store_u16(output, (uint16_t) vo0123);
        output += 2;
        vo0123 >>= 16;
      }
      if (c & (1 * sizeof(uint8_t))) {
        *output++ = (uint8_t) vo0123;
      }
    }

    output = (uint8_t*) ((uintptr_t) output + output_increment);
  } while (--output_pixels != 0);
}
