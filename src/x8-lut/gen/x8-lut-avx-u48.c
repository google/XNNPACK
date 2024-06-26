// Auto-generated file. Do not edit!
//   Template: src/x8-lut/ssse3.c.in
//   Generator: tools/xngen
//
// Copyright 2021 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <immintrin.h>

#include "xnnpack/common.h"
#include "xnnpack/intrinsics-polyfill.h"
#include "xnnpack/lut.h"


void xnn_x8_lut_ukernel__avx_u48(
    size_t batch,
    const uint8_t* input,
    uint8_t* output,
    const uint8_t table[restrict XNN_MIN_ELEMENTS(256)])
{
  assert(batch != 0);
  assert(batch % sizeof(uint8_t) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const __m128i vt0 = _mm_load_si128((const __m128i*) table);
  const __m128i vt1 = _mm_load_si128((const __m128i*) (table + 16));
  const __m128i vt2 = _mm_load_si128((const __m128i*) (table + 32));
  const __m128i vt3 = _mm_load_si128((const __m128i*) (table + 48));
  const __m128i vt4 = _mm_load_si128((const __m128i*) (table + 64));
  const __m128i vt5 = _mm_load_si128((const __m128i*) (table + 80));
  const __m128i vt6 = _mm_load_si128((const __m128i*) (table + 96));
  const __m128i vt7 = _mm_load_si128((const __m128i*) (table + 112));
  const __m128i vt8 = _mm_load_si128((const __m128i*) (table + 128));
  const __m128i vt9 = _mm_load_si128((const __m128i*) (table + 144));
  const __m128i vtA = _mm_load_si128((const __m128i*) (table + 160));
  const __m128i vtB = _mm_load_si128((const __m128i*) (table + 176));
  const __m128i vtC = _mm_load_si128((const __m128i*) (table + 192));
  const __m128i vtD = _mm_load_si128((const __m128i*) (table + 208));
  const __m128i vtE = _mm_load_si128((const __m128i*) (table + 224));
  const __m128i vtF = _mm_load_si128((const __m128i*) (table + 240));

  const __m128i vtable0 = vt0;
  const __m128i vtable1 = _mm_xor_si128(vt0, vt1);
  const __m128i vtable2 = _mm_xor_si128(vt1, vt2);
  const __m128i vtable3 = _mm_xor_si128(vt2, vt3);
  const __m128i vtable4 = _mm_xor_si128(vt3, vt4);
  const __m128i vtable5 = _mm_xor_si128(vt4, vt5);
  const __m128i vtable6 = _mm_xor_si128(vt5, vt6);
  const __m128i vtable7 = _mm_xor_si128(vt6, vt7);
  const __m128i vtable8 = _mm_xor_si128(_mm_xor_si128(vt7, vt8), vtable0);
  const __m128i vtable9 = _mm_xor_si128(_mm_xor_si128(vt8, vt9), vtable1);
  const __m128i vtableA = _mm_xor_si128(_mm_xor_si128(vt9, vtA), vtable2);
  const __m128i vtableB = _mm_xor_si128(_mm_xor_si128(vtA, vtB), vtable3);
  const __m128i vtableC = _mm_xor_si128(_mm_xor_si128(vtB, vtC), vtable4);
  const __m128i vtableD = _mm_xor_si128(_mm_xor_si128(vtC, vtD), vtable5);
  const __m128i vtableE = _mm_xor_si128(_mm_xor_si128(vtD, vtE), vtable6);
  const __m128i vtableF = _mm_xor_si128(_mm_xor_si128(vtE, vtF), vtable7);

  const __m128i voffset = _mm_set1_epi8(16);
  for (; batch >= 48 * sizeof(uint8_t); batch -= 48 * sizeof(uint8_t)) {
    __m128i vx0 = _mm_loadu_si128((const __m128i*) input);
    __m128i vx1 = _mm_loadu_si128((const __m128i*) (input + 16));
    __m128i vx2 = _mm_loadu_si128((const __m128i*) (input + 32));
    input += 48;

    __m128i vy0 = _mm_shuffle_epi8(vtable0, vx0);
    __m128i vy1 = _mm_shuffle_epi8(vtable0, vx1);
    __m128i vy2 = _mm_shuffle_epi8(vtable0, vx2);

    vx0 = _mm_sub_epi8(vx0, voffset);
    vx1 = _mm_sub_epi8(vx1, voffset);
    vx2 = _mm_sub_epi8(vx2, voffset);
    vy0 = _mm_xor_si128(vy0, _mm_shuffle_epi8(vtable1, vx0));
    vy1 = _mm_xor_si128(vy1, _mm_shuffle_epi8(vtable1, vx1));
    vy2 = _mm_xor_si128(vy2, _mm_shuffle_epi8(vtable1, vx2));
    vx0 = _mm_sub_epi8(vx0, voffset);
    vx1 = _mm_sub_epi8(vx1, voffset);
    vx2 = _mm_sub_epi8(vx2, voffset);
    vy0 = _mm_xor_si128(vy0, _mm_shuffle_epi8(vtable2, vx0));
    vy1 = _mm_xor_si128(vy1, _mm_shuffle_epi8(vtable2, vx1));
    vy2 = _mm_xor_si128(vy2, _mm_shuffle_epi8(vtable2, vx2));
    vx0 = _mm_sub_epi8(vx0, voffset);
    vx1 = _mm_sub_epi8(vx1, voffset);
    vx2 = _mm_sub_epi8(vx2, voffset);
    vy0 = _mm_xor_si128(vy0, _mm_shuffle_epi8(vtable3, vx0));
    vy1 = _mm_xor_si128(vy1, _mm_shuffle_epi8(vtable3, vx1));
    vy2 = _mm_xor_si128(vy2, _mm_shuffle_epi8(vtable3, vx2));
    vx0 = _mm_sub_epi8(vx0, voffset);
    vx1 = _mm_sub_epi8(vx1, voffset);
    vx2 = _mm_sub_epi8(vx2, voffset);
    vy0 = _mm_xor_si128(vy0, _mm_shuffle_epi8(vtable4, vx0));
    vy1 = _mm_xor_si128(vy1, _mm_shuffle_epi8(vtable4, vx1));
    vy2 = _mm_xor_si128(vy2, _mm_shuffle_epi8(vtable4, vx2));
    vx0 = _mm_sub_epi8(vx0, voffset);
    vx1 = _mm_sub_epi8(vx1, voffset);
    vx2 = _mm_sub_epi8(vx2, voffset);
    vy0 = _mm_xor_si128(vy0, _mm_shuffle_epi8(vtable5, vx0));
    vy1 = _mm_xor_si128(vy1, _mm_shuffle_epi8(vtable5, vx1));
    vy2 = _mm_xor_si128(vy2, _mm_shuffle_epi8(vtable5, vx2));
    vx0 = _mm_sub_epi8(vx0, voffset);
    vx1 = _mm_sub_epi8(vx1, voffset);
    vx2 = _mm_sub_epi8(vx2, voffset);
    vy0 = _mm_xor_si128(vy0, _mm_shuffle_epi8(vtable6, vx0));
    vy1 = _mm_xor_si128(vy1, _mm_shuffle_epi8(vtable6, vx1));
    vy2 = _mm_xor_si128(vy2, _mm_shuffle_epi8(vtable6, vx2));
    vx0 = _mm_sub_epi8(vx0, voffset);
    vx1 = _mm_sub_epi8(vx1, voffset);
    vx2 = _mm_sub_epi8(vx2, voffset);
    vy0 = _mm_xor_si128(vy0, _mm_shuffle_epi8(vtable7, vx0));
    vy1 = _mm_xor_si128(vy1, _mm_shuffle_epi8(vtable7, vx1));
    vy2 = _mm_xor_si128(vy2, _mm_shuffle_epi8(vtable7, vx2));
    vx0 = _mm_sub_epi8(vx0, voffset);
    vx1 = _mm_sub_epi8(vx1, voffset);
    vx2 = _mm_sub_epi8(vx2, voffset);
    vy0 = _mm_xor_si128(vy0, _mm_shuffle_epi8(vtable8, vx0));
    vy1 = _mm_xor_si128(vy1, _mm_shuffle_epi8(vtable8, vx1));
    vy2 = _mm_xor_si128(vy2, _mm_shuffle_epi8(vtable8, vx2));

    vx0 = _mm_subs_epi8(vx0, voffset);
    vx1 = _mm_subs_epi8(vx1, voffset);
    vx2 = _mm_subs_epi8(vx2, voffset);
    vy0 = _mm_xor_si128(vy0, _mm_shuffle_epi8(vtable9, vx0));
    vy1 = _mm_xor_si128(vy1, _mm_shuffle_epi8(vtable9, vx1));
    vy2 = _mm_xor_si128(vy2, _mm_shuffle_epi8(vtable9, vx2));
    vx0 = _mm_subs_epi8(vx0, voffset);
    vx1 = _mm_subs_epi8(vx1, voffset);
    vx2 = _mm_subs_epi8(vx2, voffset);
    vy0 = _mm_xor_si128(vy0, _mm_shuffle_epi8(vtableA, vx0));
    vy1 = _mm_xor_si128(vy1, _mm_shuffle_epi8(vtableA, vx1));
    vy2 = _mm_xor_si128(vy2, _mm_shuffle_epi8(vtableA, vx2));
    vx0 = _mm_subs_epi8(vx0, voffset);
    vx1 = _mm_subs_epi8(vx1, voffset);
    vx2 = _mm_subs_epi8(vx2, voffset);
    vy0 = _mm_xor_si128(vy0, _mm_shuffle_epi8(vtableB, vx0));
    vy1 = _mm_xor_si128(vy1, _mm_shuffle_epi8(vtableB, vx1));
    vy2 = _mm_xor_si128(vy2, _mm_shuffle_epi8(vtableB, vx2));
    vx0 = _mm_subs_epi8(vx0, voffset);
    vx1 = _mm_subs_epi8(vx1, voffset);
    vx2 = _mm_subs_epi8(vx2, voffset);
    vy0 = _mm_xor_si128(vy0, _mm_shuffle_epi8(vtableC, vx0));
    vy1 = _mm_xor_si128(vy1, _mm_shuffle_epi8(vtableC, vx1));
    vy2 = _mm_xor_si128(vy2, _mm_shuffle_epi8(vtableC, vx2));
    vx0 = _mm_subs_epi8(vx0, voffset);
    vx1 = _mm_subs_epi8(vx1, voffset);
    vx2 = _mm_subs_epi8(vx2, voffset);
    vy0 = _mm_xor_si128(vy0, _mm_shuffle_epi8(vtableD, vx0));
    vy1 = _mm_xor_si128(vy1, _mm_shuffle_epi8(vtableD, vx1));
    vy2 = _mm_xor_si128(vy2, _mm_shuffle_epi8(vtableD, vx2));
    vx0 = _mm_subs_epi8(vx0, voffset);
    vx1 = _mm_subs_epi8(vx1, voffset);
    vx2 = _mm_subs_epi8(vx2, voffset);
    vy0 = _mm_xor_si128(vy0, _mm_shuffle_epi8(vtableE, vx0));
    vy1 = _mm_xor_si128(vy1, _mm_shuffle_epi8(vtableE, vx1));
    vy2 = _mm_xor_si128(vy2, _mm_shuffle_epi8(vtableE, vx2));
    vx0 = _mm_subs_epi8(vx0, voffset);
    vx1 = _mm_subs_epi8(vx1, voffset);
    vx2 = _mm_subs_epi8(vx2, voffset);
    vy0 = _mm_xor_si128(vy0, _mm_shuffle_epi8(vtableF, vx0));
    vy1 = _mm_xor_si128(vy1, _mm_shuffle_epi8(vtableF, vx1));
    vy2 = _mm_xor_si128(vy2, _mm_shuffle_epi8(vtableF, vx2));

    _mm_storeu_si128((__m128i*) output, vy0);
    _mm_storeu_si128((__m128i*) (output + 16), vy1);
    _mm_storeu_si128((__m128i*) (output + 32), vy2);
    output += 48;
  }
  for (; batch >= 16 * sizeof(uint8_t); batch -= 16 * sizeof(uint8_t)) {
    __m128i vx = _mm_loadu_si128((const __m128i*) input);
    input += 16;

    __m128i vy = _mm_shuffle_epi8(vtable0, vx);

    vx = _mm_sub_epi8(vx, voffset);
    vy = _mm_xor_si128(vy, _mm_shuffle_epi8(vtable1, vx));
    vx = _mm_sub_epi8(vx, voffset);
    vy = _mm_xor_si128(vy, _mm_shuffle_epi8(vtable2, vx));
    vx = _mm_sub_epi8(vx, voffset);
    vy = _mm_xor_si128(vy, _mm_shuffle_epi8(vtable3, vx));
    vx = _mm_sub_epi8(vx, voffset);
    vy = _mm_xor_si128(vy, _mm_shuffle_epi8(vtable4, vx));
    vx = _mm_sub_epi8(vx, voffset);
    vy = _mm_xor_si128(vy, _mm_shuffle_epi8(vtable5, vx));
    vx = _mm_sub_epi8(vx, voffset);
    vy = _mm_xor_si128(vy, _mm_shuffle_epi8(vtable6, vx));
    vx = _mm_sub_epi8(vx, voffset);
    vy = _mm_xor_si128(vy, _mm_shuffle_epi8(vtable7, vx));
    vx = _mm_sub_epi8(vx, voffset);
    vy = _mm_xor_si128(vy, _mm_shuffle_epi8(vtable8, vx));

    vx = _mm_subs_epi8(vx, voffset);
    vy = _mm_xor_si128(vy, _mm_shuffle_epi8(vtable9, vx));
    vx = _mm_subs_epi8(vx, voffset);
    vy = _mm_xor_si128(vy, _mm_shuffle_epi8(vtableA, vx));
    vx = _mm_subs_epi8(vx, voffset);
    vy = _mm_xor_si128(vy, _mm_shuffle_epi8(vtableB, vx));
    vx = _mm_subs_epi8(vx, voffset);
    vy = _mm_xor_si128(vy, _mm_shuffle_epi8(vtableC, vx));
    vx = _mm_subs_epi8(vx, voffset);
    vy = _mm_xor_si128(vy, _mm_shuffle_epi8(vtableD, vx));
    vx = _mm_subs_epi8(vx, voffset);
    vy = _mm_xor_si128(vy, _mm_shuffle_epi8(vtableE, vx));
    vx = _mm_subs_epi8(vx, voffset);
    vy = _mm_xor_si128(vy, _mm_shuffle_epi8(vtableF, vx));

    _mm_storeu_si128((__m128i*) output, vy);
    output += 16;
  }
  if XNN_UNLIKELY(batch != 0) {
    __m128i vx = _mm_loadu_si128((const __m128i*) input);

    __m128i vy = _mm_shuffle_epi8(vtable0, vx);

    vx = _mm_sub_epi8(vx, voffset);
    vy = _mm_xor_si128(vy, _mm_shuffle_epi8(vtable1, vx));
    vx = _mm_sub_epi8(vx, voffset);
    vy = _mm_xor_si128(vy, _mm_shuffle_epi8(vtable2, vx));
    vx = _mm_sub_epi8(vx, voffset);
    vy = _mm_xor_si128(vy, _mm_shuffle_epi8(vtable3, vx));
    vx = _mm_sub_epi8(vx, voffset);
    vy = _mm_xor_si128(vy, _mm_shuffle_epi8(vtable4, vx));
    vx = _mm_sub_epi8(vx, voffset);
    vy = _mm_xor_si128(vy, _mm_shuffle_epi8(vtable5, vx));
    vx = _mm_sub_epi8(vx, voffset);
    vy = _mm_xor_si128(vy, _mm_shuffle_epi8(vtable6, vx));
    vx = _mm_sub_epi8(vx, voffset);
    vy = _mm_xor_si128(vy, _mm_shuffle_epi8(vtable7, vx));
    vx = _mm_sub_epi8(vx, voffset);
    vy = _mm_xor_si128(vy, _mm_shuffle_epi8(vtable8, vx));

    vx = _mm_subs_epi8(vx, voffset);
    vy = _mm_xor_si128(vy, _mm_shuffle_epi8(vtable9, vx));
    vx = _mm_subs_epi8(vx, voffset);
    vy = _mm_xor_si128(vy, _mm_shuffle_epi8(vtableA, vx));
    vx = _mm_subs_epi8(vx, voffset);
    vy = _mm_xor_si128(vy, _mm_shuffle_epi8(vtableB, vx));
    vx = _mm_subs_epi8(vx, voffset);
    vy = _mm_xor_si128(vy, _mm_shuffle_epi8(vtableC, vx));
    vx = _mm_subs_epi8(vx, voffset);
    vy = _mm_xor_si128(vy, _mm_shuffle_epi8(vtableD, vx));
    vx = _mm_subs_epi8(vx, voffset);
    vy = _mm_xor_si128(vy, _mm_shuffle_epi8(vtableE, vx));
    vx = _mm_subs_epi8(vx, voffset);
    vy = _mm_xor_si128(vy, _mm_shuffle_epi8(vtableF, vx));

    if (batch & (8 * sizeof(uint8_t))) {
      _mm_storel_epi64((__m128i*) output, vy);
      vy = _mm_unpackhi_epi64(vy, vy);
      output += 8;
    }
    if (batch & (4 * sizeof(uint8_t))) {
      _mm_storeu_si32(output, vy);
      vy = _mm_srli_epi64(vy, 32);
      output += 4;
    }
    if (batch & (2 * sizeof(uint8_t))) {
      _mm_storeu_si16(output, vy);
      vy = _mm_srli_epi32(vy, 16);
      output += 2;
    }
    if (batch & (1 * sizeof(uint8_t))) {
      *output = (uint8_t) _mm_extract_epi8(vy, 0);
    }
  }
}
