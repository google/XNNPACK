// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/f16-dwconv2d-chw/3x3p1-avx512fp16.c.in
//   Generator: tools/xngen
//
// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include <immintrin.h>

#include "src/xnnpack/common.h"
#include "src/xnnpack/dwconv.h"
#include "src/xnnpack/math.h"
#include "src/xnnpack/microparams.h"


void xnn_f16_dwconv2d_chw_ukernel_3x3p1__avx512fp16_2x32(
    size_t input_height,
    size_t input_width,
    const xnn_float16* input,
    const xnn_float16* weights,
    const xnn_float16* zero,
    xnn_float16* output,
    uint32_t padding_top,
    const struct xnn_f16_minmax_params* restrict params) XNN_OOB_READS
{
  assert(input_height != 0);
  assert(input_width != 0);
  assert(input_width % sizeof(uint16_t) == 0);
  assert(padding_top == 1);

  const __m512h vmin = _mm512_castsi512_ph(_mm512_set1_epi16(*(const uint16_t*) &params->scalar.min));
  const __m512h vmax = _mm512_castsi512_ph(_mm512_set1_epi16(*(const uint16_t*) &params->scalar.max));

  const __m512i vidx_prev = _mm512_set_epi16(
      30, 29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15,
      14, 13, 12, 11, 10,  9,  8,  7,  6,  5,  4,  3,  2,  1,  0, 63);

  const __m512i vidx_next = _mm512_set_epi16(
      0, 63, 62, 61, 60, 59, 58, 57, 56, 55, 54, 53, 52, 51, 50, 49,
      48, 47, 46, 45, 44, 43, 42, 41, 40, 39, 38, 37, 36, 35, 34, 33);

  const __m512i vidx_next2 = _mm512_set_epi16(
      1, 0, 63, 62, 61, 60, 59, 58, 57, 56, 55, 54, 53, 52, 51, 50,
      49, 48, 47, 46, 45, 44, 43, 42, 41, 40, 39, 38, 37, 36, 35, 34);

  const __m512i vidx_prev2 = _mm512_set_epi16(
      29, 28, 27, 26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14,
      13, 12, 11, 10,  9,  8,  7,  6,  5,  4,  3,  2,  1,  0, 63, 62);

  const __m512i vidx_even = _mm512_set_epi16(
      62, 60, 58, 56, 54, 52, 50, 48, 46, 44, 42, 40, 38, 36, 34, 32,
      30, 28, 26, 24, 22, 20, 18, 16, 14, 12, 10,  8,  6,  4,  2,  0);

  const __m512i vidx_odd = _mm512_set_epi16(
      63, 61, 59, 57, 55, 53, 51, 49, 47, 45, 43, 41, 39, 37, 35, 33,
      31, 29, 27, 25, 23, 21, 19, 17, 15, 13, 11,  9,  7,  5,  3,  1);

  (void) vidx_prev;
  (void) vidx_next;
  (void) vidx_next2;
  (void) vidx_prev2;
  (void) vidx_even;
  (void) vidx_odd;





  const __m512h vbias = _mm512_castsi512_ph(_mm512_set1_epi16(*(const uint16_t*) weights));
  const __m512h vk00 = _mm512_castsi512_ph(_mm512_set1_epi16(*(const uint16_t*) (weights + 1)));
  const __m512h vk01 = _mm512_castsi512_ph(_mm512_set1_epi16(*(const uint16_t*) (weights + 2)));
  const __m512h vk02 = _mm512_castsi512_ph(_mm512_set1_epi16(*(const uint16_t*) (weights + 3)));
  const __m512h vk10 = _mm512_castsi512_ph(_mm512_set1_epi16(*(const uint16_t*) (weights + 4)));
  const __m512h vk11 = _mm512_castsi512_ph(_mm512_set1_epi16(*(const uint16_t*) (weights + 5)));
  const __m512h vk12 = _mm512_castsi512_ph(_mm512_set1_epi16(*(const uint16_t*) (weights + 6)));
  const __m512h vk20 = _mm512_castsi512_ph(_mm512_set1_epi16(*(const uint16_t*) (weights + 7)));
  const __m512h vk21 = _mm512_castsi512_ph(_mm512_set1_epi16(*(const uint16_t*) (weights + 8)));
  const __m512h vk22 = _mm512_castsi512_ph(_mm512_set1_epi16(*(const uint16_t*) (weights + 9)));

  const size_t input_decrement = round_up_po2(input_width, 32 * sizeof(uint16_t));

  const uint16_t* i0 = (const uint16_t*) zero;
  const uint16_t* i1 = (const uint16_t*) input;
  const uint16_t* i2 = (const uint16_t*) ((uintptr_t) i1 + input_width);
  const uint16_t* i3 = (const uint16_t*) ((uintptr_t) i2 + input_width);

  uint16_t* o0 = (uint16_t*) output;
  uint16_t* o1 = (uint16_t*) ((uintptr_t) o0 + input_width);

  size_t output_height = input_height;
  do {
    if XNN_UNPREDICTABLE(output_height < 2) {
      i2 = (const uint16_t*) zero;
      o1 = o0;
    }
    if XNN_UNPREDICTABLE(output_height < 3) {
      i3 = (const uint16_t*) zero;
    }

    __m512h vi0x_prev = _mm512_setzero_ph();
    __m512h vi1x_prev = _mm512_setzero_ph();
    __m512h vi2x_prev = _mm512_setzero_ph();
    __m512h vi3x_prev = _mm512_setzero_ph();

    __m512h vi0x_curr = _mm512_castsi512_ph(_mm512_loadu_epi16(i0));
    i0 += 32;
    __m512h vi1x_curr = _mm512_castsi512_ph(_mm512_loadu_epi16(i1));
    i1 += 32;
    __m512h vi2x_curr = _mm512_castsi512_ph(_mm512_loadu_epi16(i2));
    i2 += 32;
    __m512h vi3x_curr = _mm512_castsi512_ph(_mm512_loadu_epi16(i3));
    i3 += 32;

    size_t w = input_width;
    for (; w > 32 * sizeof(uint16_t); w -= 32 * sizeof(uint16_t)) {
      const __m512h vi0x_next = _mm512_castsi512_ph(_mm512_loadu_epi16(i0));
      i0 += 32;
      const __m512h vi1x_next = _mm512_castsi512_ph(_mm512_loadu_epi16(i1));
      i1 += 32;
      const __m512h vi2x_next = _mm512_castsi512_ph(_mm512_loadu_epi16(i2));
      i2 += 32;
      const __m512h vi3x_next = _mm512_castsi512_ph(_mm512_loadu_epi16(i3));
      i3 += 32;

      __m512h vo0p0 = _mm512_fmadd_ph(vi0x_curr, vk01, vbias);
      __m512h vo1p0 = _mm512_fmadd_ph(vi1x_curr, vk01, vbias);
      vo0p0 = _mm512_fmadd_ph(vi1x_curr, vk11, vo0p0);
      vo1p0 = _mm512_fmadd_ph(vi2x_curr, vk11, vo1p0);
      vo0p0 = _mm512_fmadd_ph(vi2x_curr, vk21, vo0p0);
      vo1p0 = _mm512_fmadd_ph(vi3x_curr, vk21, vo1p0);

      const __m512h vi0x_left = _mm512_castsi512_ph(_mm512_permutex2var_epi16(
          _mm512_castph_si512(vi0x_curr), vidx_prev, _mm512_castph_si512(vi0x_prev)));
      const __m512h vi1x_left = _mm512_castsi512_ph(_mm512_permutex2var_epi16(
          _mm512_castph_si512(vi1x_curr), vidx_prev, _mm512_castph_si512(vi1x_prev)));
      const __m512h vi2x_left = _mm512_castsi512_ph(_mm512_permutex2var_epi16(
          _mm512_castph_si512(vi2x_curr), vidx_prev, _mm512_castph_si512(vi2x_prev)));
      const __m512h vi3x_left = _mm512_castsi512_ph(_mm512_permutex2var_epi16(
          _mm512_castph_si512(vi3x_curr), vidx_prev, _mm512_castph_si512(vi3x_prev)));

      vo0p0 = _mm512_fmadd_ph(vi0x_left, vk00, vo0p0);
      vo1p0 = _mm512_fmadd_ph(vi1x_left, vk00, vo1p0);
      vo0p0 = _mm512_fmadd_ph(vi1x_left, vk10, vo0p0);
      vo1p0 = _mm512_fmadd_ph(vi2x_left, vk10, vo1p0);
      vo0p0 = _mm512_fmadd_ph(vi2x_left, vk20, vo0p0);
      vo1p0 = _mm512_fmadd_ph(vi3x_left, vk20, vo1p0);

      vi0x_prev = vi0x_curr;
      vi1x_prev = vi1x_curr;
      vi2x_prev = vi2x_curr;
      vi3x_prev = vi3x_curr;

      const __m512h vi0x_right = _mm512_castsi512_ph(_mm512_permutex2var_epi16(
          _mm512_castph_si512(vi0x_next), vidx_next, _mm512_castph_si512(vi0x_curr)));
      const __m512h vi1x_right = _mm512_castsi512_ph(_mm512_permutex2var_epi16(
          _mm512_castph_si512(vi1x_next), vidx_next, _mm512_castph_si512(vi1x_curr)));
      const __m512h vi2x_right = _mm512_castsi512_ph(_mm512_permutex2var_epi16(
          _mm512_castph_si512(vi2x_next), vidx_next, _mm512_castph_si512(vi2x_curr)));
      const __m512h vi3x_right = _mm512_castsi512_ph(_mm512_permutex2var_epi16(
          _mm512_castph_si512(vi3x_next), vidx_next, _mm512_castph_si512(vi3x_curr)));

      vo0p0 = _mm512_fmadd_ph(vi0x_right, vk02, vo0p0);
      vo1p0 = _mm512_fmadd_ph(vi1x_right, vk02, vo1p0);
      vo0p0 = _mm512_fmadd_ph(vi1x_right, vk12, vo0p0);
      vo1p0 = _mm512_fmadd_ph(vi2x_right, vk12, vo1p0);
      vo0p0 = _mm512_fmadd_ph(vi2x_right, vk22, vo0p0);
      vo1p0 = _mm512_fmadd_ph(vi3x_right, vk22, vo1p0);

      vi0x_curr = vi0x_next;
      vi1x_curr = vi1x_next;
      vi2x_curr = vi2x_next;
      vi3x_curr = vi3x_next;


      __m512h vo0 = _mm512_max_ph(vo0p0, vmin);
      __m512h vo1 = _mm512_max_ph(vo1p0, vmin);

      vo0 = _mm512_min_ph(vo0, vmax);
      vo1 = _mm512_min_ph(vo1, vmax);

      _mm512_storeu_epi16(o1, _mm512_castph_si512(vo1));
      o1 += 32;
      _mm512_storeu_epi16(o0, _mm512_castph_si512(vo0));
      o0 += 32;
    }
    // Always process the last block of 1..32 pixels.
    assert(w >= 1 * sizeof(uint16_t));
    assert(w <= 32 * sizeof(uint16_t));
    {
      __mmask32 vmask = _cvtu32_mask32((uint32_t)(((uint64_t)1 << (w / sizeof(uint16_t))) - 1));
      vi0x_curr = _mm512_castsi512_ph(_mm512_maskz_mov_epi16(vmask, _mm512_castph_si512(vi0x_curr)));
      vi1x_curr = _mm512_castsi512_ph(_mm512_maskz_mov_epi16(vmask, _mm512_castph_si512(vi1x_curr)));
      vi2x_curr = _mm512_castsi512_ph(_mm512_maskz_mov_epi16(vmask, _mm512_castph_si512(vi2x_curr)));
      vi3x_curr = _mm512_castsi512_ph(_mm512_maskz_mov_epi16(vmask, _mm512_castph_si512(vi3x_curr)));

      __m512h vo0p0 = _mm512_fmadd_ph(vi0x_curr, vk01, vbias);
      __m512h vo1p0 = _mm512_fmadd_ph(vi1x_curr, vk01, vbias);
      vo0p0 = _mm512_fmadd_ph(vi1x_curr, vk11, vo0p0);
      vo1p0 = _mm512_fmadd_ph(vi2x_curr, vk11, vo1p0);
      vo0p0 = _mm512_fmadd_ph(vi2x_curr, vk21, vo0p0);
      vo1p0 = _mm512_fmadd_ph(vi3x_curr, vk21, vo1p0);

      const __m512h vi0x_left = _mm512_castsi512_ph(_mm512_permutex2var_epi16(
          _mm512_castph_si512(vi0x_curr), vidx_prev, _mm512_castph_si512(vi0x_prev)));
      const __m512h vi1x_left = _mm512_castsi512_ph(_mm512_permutex2var_epi16(
          _mm512_castph_si512(vi1x_curr), vidx_prev, _mm512_castph_si512(vi1x_prev)));
      const __m512h vi2x_left = _mm512_castsi512_ph(_mm512_permutex2var_epi16(
          _mm512_castph_si512(vi2x_curr), vidx_prev, _mm512_castph_si512(vi2x_prev)));
      const __m512h vi3x_left = _mm512_castsi512_ph(_mm512_permutex2var_epi16(
          _mm512_castph_si512(vi3x_curr), vidx_prev, _mm512_castph_si512(vi3x_prev)));

      vo0p0 = _mm512_fmadd_ph(vi0x_left, vk00, vo0p0);
      vo1p0 = _mm512_fmadd_ph(vi1x_left, vk00, vo1p0);
      vo0p0 = _mm512_fmadd_ph(vi1x_left, vk10, vo0p0);
      vo1p0 = _mm512_fmadd_ph(vi2x_left, vk10, vo1p0);
      vo0p0 = _mm512_fmadd_ph(vi2x_left, vk20, vo0p0);
      vo1p0 = _mm512_fmadd_ph(vi3x_left, vk20, vo1p0);

      const __m512h vzero = _mm512_setzero_ph();
      const __m512h vi0x_right = _mm512_castsi512_ph(_mm512_permutex2var_epi16(
          _mm512_castph_si512(vzero), vidx_next, _mm512_castph_si512(vi0x_curr)));
      const __m512h vi1x_right = _mm512_castsi512_ph(_mm512_permutex2var_epi16(
          _mm512_castph_si512(vzero), vidx_next, _mm512_castph_si512(vi1x_curr)));
      const __m512h vi2x_right = _mm512_castsi512_ph(_mm512_permutex2var_epi16(
          _mm512_castph_si512(vzero), vidx_next, _mm512_castph_si512(vi2x_curr)));
      const __m512h vi3x_right = _mm512_castsi512_ph(_mm512_permutex2var_epi16(
          _mm512_castph_si512(vzero), vidx_next, _mm512_castph_si512(vi3x_curr)));

      vo0p0 = _mm512_fmadd_ph(vi0x_right, vk02, vo0p0);
      vo1p0 = _mm512_fmadd_ph(vi1x_right, vk02, vo1p0);
      vo0p0 = _mm512_fmadd_ph(vi1x_right, vk12, vo0p0);
      vo1p0 = _mm512_fmadd_ph(vi2x_right, vk12, vo1p0);
      vo0p0 = _mm512_fmadd_ph(vi2x_right, vk22, vo0p0);
      vo1p0 = _mm512_fmadd_ph(vi3x_right, vk22, vo1p0);


      __m512h vo0 = _mm512_max_ph(vo0p0, vmin);
      __m512h vo1 = _mm512_max_ph(vo1p0, vmin);

      vo0 = _mm512_min_ph(vo0, vmax);
      vo1 = _mm512_min_ph(vo1, vmax);

      if XNN_LIKELY(w == 32 * sizeof(uint16_t)) {
        _mm512_storeu_epi16(o1, _mm512_castph_si512(vo1));
        o1 += 32;
        _mm512_storeu_epi16(o0, _mm512_castph_si512(vo0));
        o0 += 32;
      } else {
        _mm512_mask_storeu_epi16(o1, vmask, _mm512_castph_si512(vo1));
        o1 = (uint16_t*) ((uintptr_t) o1 + w);
        _mm512_mask_storeu_epi16(o0, vmask, _mm512_castph_si512(vo0));
        o0 = (uint16_t*) ((uintptr_t) o0 + w);
      }
    }

    i0 = (const uint16_t*) ((uintptr_t) i2 - input_decrement);
    i1 = (const uint16_t*) ((uintptr_t) i3 - input_decrement);
    i2 = (const uint16_t*) ((uintptr_t) i1 + input_width);
    i3 = (const uint16_t*) ((uintptr_t) i2 + input_width);

    o0 = o1;
    o1 = (uint16_t*) ((uintptr_t) o0 + input_width);

    output_height = doz(output_height, 2);
  } while (output_height != 0);
}
