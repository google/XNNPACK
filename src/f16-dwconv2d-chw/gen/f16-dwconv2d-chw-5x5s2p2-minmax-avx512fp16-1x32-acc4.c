// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/f16-dwconv2d-chw/5x5s2p2-avx512fp16.c.in
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


void xnn_f16_dwconv2d_chw_ukernel_5x5s2p2__avx512fp16_1x32_acc4(
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
  assert(padding_top >= 1);
  assert(padding_top <= 2);

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
  const __m512h vk03 = _mm512_castsi512_ph(_mm512_set1_epi16(*(const uint16_t*) (weights + 4)));
  const __m512h vk04 = _mm512_castsi512_ph(_mm512_set1_epi16(*(const uint16_t*) (weights + 5)));
  const __m512h vk10 = _mm512_castsi512_ph(_mm512_set1_epi16(*(const uint16_t*) (weights + 6)));
  const __m512h vk11 = _mm512_castsi512_ph(_mm512_set1_epi16(*(const uint16_t*) (weights + 7)));
  const __m512h vk12 = _mm512_castsi512_ph(_mm512_set1_epi16(*(const uint16_t*) (weights + 8)));
  const __m512h vk13 = _mm512_castsi512_ph(_mm512_set1_epi16(*(const uint16_t*) (weights + 9)));
  const __m512h vk14 = _mm512_castsi512_ph(_mm512_set1_epi16(*(const uint16_t*) (weights + 10)));
  const __m512h vk20 = _mm512_castsi512_ph(_mm512_set1_epi16(*(const uint16_t*) (weights + 11)));
  const __m512h vk21 = _mm512_castsi512_ph(_mm512_set1_epi16(*(const uint16_t*) (weights + 12)));
  const __m512h vk22 = _mm512_castsi512_ph(_mm512_set1_epi16(*(const uint16_t*) (weights + 13)));
  const __m512h vk23 = _mm512_castsi512_ph(_mm512_set1_epi16(*(const uint16_t*) (weights + 14)));
  const __m512h vk24 = _mm512_castsi512_ph(_mm512_set1_epi16(*(const uint16_t*) (weights + 15)));
  const __m512h vk30 = _mm512_castsi512_ph(_mm512_set1_epi16(*(const uint16_t*) (weights + 16)));
  const __m512h vk31 = _mm512_castsi512_ph(_mm512_set1_epi16(*(const uint16_t*) (weights + 17)));
  const __m512h vk32 = _mm512_castsi512_ph(_mm512_set1_epi16(*(const uint16_t*) (weights + 18)));
  const __m512h vk33 = _mm512_castsi512_ph(_mm512_set1_epi16(*(const uint16_t*) (weights + 19)));
  const __m512h vk34 = _mm512_castsi512_ph(_mm512_set1_epi16(*(const uint16_t*) (weights + 20)));
  const __m512h vk40 = _mm512_castsi512_ph(_mm512_set1_epi16(*(const uint16_t*) (weights + 21)));
  const __m512h vk41 = _mm512_castsi512_ph(_mm512_set1_epi16(*(const uint16_t*) (weights + 22)));
  const __m512h vk42 = _mm512_castsi512_ph(_mm512_set1_epi16(*(const uint16_t*) (weights + 23)));
  const __m512h vk43 = _mm512_castsi512_ph(_mm512_set1_epi16(*(const uint16_t*) (weights + 24)));
  const __m512h vk44 = _mm512_castsi512_ph(_mm512_set1_epi16(*(const uint16_t*) (weights + 25)));

  const uint32_t padding_top_less_1 = padding_top - 1;
  const size_t input_decrement = round_up_po2(input_width, 64 * sizeof(uint16_t));

  const uint16_t* i0 = (const uint16_t*) zero;
  const uint16_t* i1 = (const uint16_t*) ((uintptr_t) input - ((-padding_top_less_1) & input_width));
  const uint16_t* i2 = (const uint16_t*) ((uintptr_t) i1 + input_width);
  if XNN_UNPREDICTABLE(padding_top_less_1 != 0) {
    i1 = (const uint16_t*) zero;
  }
  const uint16_t* i3 = (const uint16_t*) ((uintptr_t) i2 + input_width);
  const uint16_t* i4 = (const uint16_t*) ((uintptr_t) i3 + input_width);


  uint16_t* o0 = (uint16_t*) output;

  size_t padded_input_height = input_height + (padding_top_less_1 + 1) + 2 /* padding bottom */;
  size_t output_height = (padded_input_height - 5 /* kernel size */ + 2 /* subsampling */) / 2;
  do {
    if XNN_UNPREDICTABLE(padded_input_height < 6) {
      i3 = (const uint16_t*) zero;
    }
    if XNN_UNPREDICTABLE(padded_input_height < 7) {
      i4 = (const uint16_t*) zero;
    }

    __m512h vi0x_prev_odd = _mm512_setzero_ph();
    __m512h vi1x_prev_odd = _mm512_setzero_ph();
    __m512h vi2x_prev_odd = _mm512_setzero_ph();
    __m512h vi3x_prev_odd = _mm512_setzero_ph();
    __m512h vi4x_prev_odd = _mm512_setzero_ph();

    __m512h vi0x_prev_even = _mm512_setzero_ph();
    __m512h vi1x_prev_even = _mm512_setzero_ph();
    __m512h vi2x_prev_even = _mm512_setzero_ph();
    __m512h vi3x_prev_even = _mm512_setzero_ph();
    __m512h vi4x_prev_even = _mm512_setzero_ph();

    size_t w = input_width;
    for (; w > 64 * sizeof(uint16_t); w -= 64 * sizeof(uint16_t)) {
      const __m512h vi0x_curr0 = _mm512_castsi512_ph(_mm512_loadu_epi16(i0));
      const __m512h vi0x_curr1 = _mm512_castsi512_ph(_mm512_loadu_epi16(i0 + 32));
      i0 += 64;
      const __m512h vi1x_curr0 = _mm512_castsi512_ph(_mm512_loadu_epi16(i1));
      const __m512h vi1x_curr1 = _mm512_castsi512_ph(_mm512_loadu_epi16(i1 + 32));
      i1 += 64;
      const __m512h vi2x_curr0 = _mm512_castsi512_ph(_mm512_loadu_epi16(i2));
      const __m512h vi2x_curr1 = _mm512_castsi512_ph(_mm512_loadu_epi16(i2 + 32));
      i2 += 64;
      const __m512h vi3x_curr0 = _mm512_castsi512_ph(_mm512_loadu_epi16(i3));
      const __m512h vi3x_curr1 = _mm512_castsi512_ph(_mm512_loadu_epi16(i3 + 32));
      i3 += 64;
      const __m512h vi4x_curr0 = _mm512_castsi512_ph(_mm512_loadu_epi16(i4));
      const __m512h vi4x_curr1 = _mm512_castsi512_ph(_mm512_loadu_epi16(i4 + 32));
      i4 += 64;

      __m512h vi0x_even = _mm512_castsi512_ph(_mm512_permutex2var_epi16(_mm512_castph_si512(vi0x_curr0), vidx_even, _mm512_castph_si512(vi0x_curr1)));
      __m512h vi0x_odd = _mm512_castsi512_ph(_mm512_permutex2var_epi16(_mm512_castph_si512(vi0x_curr0), vidx_odd, _mm512_castph_si512(vi0x_curr1)));
      __m512h vi1x_even = _mm512_castsi512_ph(_mm512_permutex2var_epi16(_mm512_castph_si512(vi1x_curr0), vidx_even, _mm512_castph_si512(vi1x_curr1)));
      __m512h vi1x_odd = _mm512_castsi512_ph(_mm512_permutex2var_epi16(_mm512_castph_si512(vi1x_curr0), vidx_odd, _mm512_castph_si512(vi1x_curr1)));
      __m512h vi2x_even = _mm512_castsi512_ph(_mm512_permutex2var_epi16(_mm512_castph_si512(vi2x_curr0), vidx_even, _mm512_castph_si512(vi2x_curr1)));
      __m512h vi2x_odd = _mm512_castsi512_ph(_mm512_permutex2var_epi16(_mm512_castph_si512(vi2x_curr0), vidx_odd, _mm512_castph_si512(vi2x_curr1)));
      __m512h vi3x_even = _mm512_castsi512_ph(_mm512_permutex2var_epi16(_mm512_castph_si512(vi3x_curr0), vidx_even, _mm512_castph_si512(vi3x_curr1)));
      __m512h vi3x_odd = _mm512_castsi512_ph(_mm512_permutex2var_epi16(_mm512_castph_si512(vi3x_curr0), vidx_odd, _mm512_castph_si512(vi3x_curr1)));
      __m512h vi4x_even = _mm512_castsi512_ph(_mm512_permutex2var_epi16(_mm512_castph_si512(vi4x_curr0), vidx_even, _mm512_castph_si512(vi4x_curr1)));
      __m512h vi4x_odd = _mm512_castsi512_ph(_mm512_permutex2var_epi16(_mm512_castph_si512(vi4x_curr0), vidx_odd, _mm512_castph_si512(vi4x_curr1)));

      // Center
      __m512h vo0p0 = _mm512_fmadd_ph(vi0x_even, vk02, vbias);
      __m512h vo0p1 = _mm512_mul_ph(vi1x_even, vk12);
      __m512h vo0p2 = _mm512_mul_ph(vi2x_even, vk22);
      __m512h vo0p3 = _mm512_mul_ph(vi3x_even, vk32);
      vo0p0 = _mm512_fmadd_ph(vi4x_even, vk42, vo0p0);

      // Right 1 (Left shift odd by 1, or just move odd right by 0, actually even and odd interleaved)
      // Original 5x5s2: Center is 2.
      // Left 1 is vk1, left 2 is vk0. Right 1 is vk3, right 2 is vk4.
      // odd elements are at offsets +1. So odd elements ARE Right 1!
      // wait, `vi4x_odd` is exactly the odd elements.
      vo0p1 = _mm512_fmadd_ph(vi0x_odd, vk03, vo0p1);
      vo0p2 = _mm512_fmadd_ph(vi1x_odd, vk13, vo0p2);
      vo0p3 = _mm512_fmadd_ph(vi2x_odd, vk23, vo0p3);
      vo0p0 = _mm512_fmadd_ph(vi3x_odd, vk33, vo0p0);
      vo0p1 = _mm512_fmadd_ph(vi4x_odd, vk43, vo0p1);

      // Left 1 (Right shift odd by 1 element, from previous odd)
      const __m512h vi0x_odd_prev = _mm512_castsi512_ph(_mm512_permutex2var_epi16(
        _mm512_castph_si512(vi0x_odd), vidx_prev, _mm512_castph_si512(vi0x_prev_odd)));
      const __m512h vi1x_odd_prev = _mm512_castsi512_ph(_mm512_permutex2var_epi16(
        _mm512_castph_si512(vi1x_odd), vidx_prev, _mm512_castph_si512(vi1x_prev_odd)));
      const __m512h vi2x_odd_prev = _mm512_castsi512_ph(_mm512_permutex2var_epi16(
        _mm512_castph_si512(vi2x_odd), vidx_prev, _mm512_castph_si512(vi2x_prev_odd)));
      const __m512h vi3x_odd_prev = _mm512_castsi512_ph(_mm512_permutex2var_epi16(
        _mm512_castph_si512(vi3x_odd), vidx_prev, _mm512_castph_si512(vi3x_prev_odd)));
      const __m512h vi4x_odd_prev = _mm512_castsi512_ph(_mm512_permutex2var_epi16(
        _mm512_castph_si512(vi4x_odd), vidx_prev, _mm512_castph_si512(vi4x_prev_odd)));

      vo0p2 = _mm512_fmadd_ph(vi0x_odd_prev, vk01, vo0p2);
      vo0p3 = _mm512_fmadd_ph(vi1x_odd_prev, vk11, vo0p3);
      vo0p0 = _mm512_fmadd_ph(vi2x_odd_prev, vk21, vo0p0);
      vo0p1 = _mm512_fmadd_ph(vi3x_odd_prev, vk31, vo0p1);
      vo0p2 = _mm512_fmadd_ph(vi4x_odd_prev, vk41, vo0p2);

      // Left 2 (Right shift even by 1 element, from previous even)
      const __m512h vi0x_even_prev = _mm512_castsi512_ph(_mm512_permutex2var_epi16(
          _mm512_castph_si512(vi0x_even), vidx_prev, _mm512_castph_si512(vi0x_prev_even)));
      const __m512h vi1x_even_prev = _mm512_castsi512_ph(_mm512_permutex2var_epi16(
          _mm512_castph_si512(vi1x_even), vidx_prev, _mm512_castph_si512(vi1x_prev_even)));
      const __m512h vi2x_even_prev = _mm512_castsi512_ph(_mm512_permutex2var_epi16(
          _mm512_castph_si512(vi2x_even), vidx_prev, _mm512_castph_si512(vi2x_prev_even)));
      const __m512h vi3x_even_prev = _mm512_castsi512_ph(_mm512_permutex2var_epi16(
          _mm512_castph_si512(vi3x_even), vidx_prev, _mm512_castph_si512(vi3x_prev_even)));
      const __m512h vi4x_even_prev = _mm512_castsi512_ph(_mm512_permutex2var_epi16(
          _mm512_castph_si512(vi4x_even), vidx_prev, _mm512_castph_si512(vi4x_prev_even)));

      vo0p3 = _mm512_fmadd_ph(vi0x_even_prev, vk00, vo0p3);
      vo0p0 = _mm512_fmadd_ph(vi1x_even_prev, vk10, vo0p0);
      vo0p1 = _mm512_fmadd_ph(vi2x_even_prev, vk20, vo0p1);
      vo0p2 = _mm512_fmadd_ph(vi3x_even_prev, vk30, vo0p2);
      vo0p3 = _mm512_fmadd_ph(vi4x_even_prev, vk40, vo0p3);

      // Right 2 (Left shift even by 1 element, from next even)
      // For this, we just need next even's first element, so we can't do it before next even is loaded, 
      // but wait! We can just use `vidx_next` on `vi0x_even` with the next loop iteration, OR we load the next 2 bytes early.
      // In the SSE template, they read from memory directly: `const __m128 vi0xGHIJ = _mm_loadu_ps(i0); ... vi0xGIKM = _mm_shuffle_ps ...`
      // For AVX512, since we read `i0`, the easiest way to get "Right 2" is to read `i0 + 2` directly!
      // But wait! Right 2 is `vi4x_even` shifted left by 1. 
      // Oh, `i0` is already advanced by 64. So `i0` points to the next elements.
      const __m512h vi0x_next_curr0 = _mm512_castsi512_ph(_mm512_loadu_epi16(i0)); // only need the first element
      // We only need the very first element of `vi0x_next_curr0`.
        const __m512h vi0x_even_next = _mm512_castsi512_ph(_mm512_permutex2var_epi16(
          _mm512_castph_si512(vi0x_next_curr0), vidx_next, _mm512_castph_si512(vi0x_even)));
      const __m512h vi1x_next_curr0 = _mm512_castsi512_ph(_mm512_loadu_epi16(i1)); // only need the first element
      // We only need the very first element of `vi1x_next_curr0`.
        const __m512h vi1x_even_next = _mm512_castsi512_ph(_mm512_permutex2var_epi16(
          _mm512_castph_si512(vi1x_next_curr0), vidx_next, _mm512_castph_si512(vi1x_even)));
      const __m512h vi2x_next_curr0 = _mm512_castsi512_ph(_mm512_loadu_epi16(i2)); // only need the first element
      // We only need the very first element of `vi2x_next_curr0`.
        const __m512h vi2x_even_next = _mm512_castsi512_ph(_mm512_permutex2var_epi16(
          _mm512_castph_si512(vi2x_next_curr0), vidx_next, _mm512_castph_si512(vi2x_even)));
      const __m512h vi3x_next_curr0 = _mm512_castsi512_ph(_mm512_loadu_epi16(i3)); // only need the first element
      // We only need the very first element of `vi3x_next_curr0`.
        const __m512h vi3x_even_next = _mm512_castsi512_ph(_mm512_permutex2var_epi16(
          _mm512_castph_si512(vi3x_next_curr0), vidx_next, _mm512_castph_si512(vi3x_even)));
      const __m512h vi4x_next_curr0 = _mm512_castsi512_ph(_mm512_loadu_epi16(i4)); // only need the first element
      // We only need the very first element of `vi4x_next_curr0`.
        const __m512h vi4x_even_next = _mm512_castsi512_ph(_mm512_permutex2var_epi16(
          _mm512_castph_si512(vi4x_next_curr0), vidx_next, _mm512_castph_si512(vi4x_even)));

      vo0p0 = _mm512_fmadd_ph(vi0x_even_next, vk04, vo0p0);
      vo0p1 = _mm512_fmadd_ph(vi1x_even_next, vk14, vo0p1);
      vo0p2 = _mm512_fmadd_ph(vi2x_even_next, vk24, vo0p2);
      vo0p3 = _mm512_fmadd_ph(vi3x_even_next, vk34, vo0p3);
      vo0p0 = _mm512_fmadd_ph(vi4x_even_next, vk44, vo0p0);

      vi0x_prev_even = vi0x_even;
      vi0x_prev_odd = vi0x_odd;
      vi1x_prev_even = vi1x_even;
      vi1x_prev_odd = vi1x_odd;
      vi2x_prev_even = vi2x_even;
      vi2x_prev_odd = vi2x_odd;
      vi3x_prev_even = vi3x_even;
      vi3x_prev_odd = vi3x_odd;
      vi4x_prev_even = vi4x_even;
      vi4x_prev_odd = vi4x_odd;

      vo0p0 = _mm512_add_ph(vo0p0, vo0p1);
      vo0p2 = _mm512_add_ph(vo0p2, vo0p3);
      vo0p0 = _mm512_add_ph(vo0p0, vo0p2);

      __m512h vo0 = _mm512_max_ph(vo0p0, vmin);

      vo0 = _mm512_min_ph(vo0, vmax);

      _mm512_storeu_epi16(o0, _mm512_castph_si512(vo0));
      o0 += 32;
    }

    assert(w >= 1 * sizeof(uint16_t));
    assert(w <= 64 * sizeof(uint16_t));
    {
      __mmask64 vmask_total = _cvtu64_mask64(w == 64 * sizeof(uint16_t) ? ~0ULL : (((uint64_t)1 << (w / sizeof(uint16_t))) - 1));
      __mmask32 vmask_lo = (__mmask32)vmask_total;
      __mmask32 vmask_hi = (__mmask32)(vmask_total >> 32);

      const __m512h vi0x_curr0 = _mm512_castsi512_ph(_mm512_maskz_loadu_epi16(vmask_lo, i0));
      const __m512h vi0x_curr1 = _mm512_castsi512_ph(_mm512_maskz_loadu_epi16(vmask_hi, i0 + 32));
      const __m512h vi1x_curr0 = _mm512_castsi512_ph(_mm512_maskz_loadu_epi16(vmask_lo, i1));
      const __m512h vi1x_curr1 = _mm512_castsi512_ph(_mm512_maskz_loadu_epi16(vmask_hi, i1 + 32));
      const __m512h vi2x_curr0 = _mm512_castsi512_ph(_mm512_maskz_loadu_epi16(vmask_lo, i2));
      const __m512h vi2x_curr1 = _mm512_castsi512_ph(_mm512_maskz_loadu_epi16(vmask_hi, i2 + 32));
      const __m512h vi3x_curr0 = _mm512_castsi512_ph(_mm512_maskz_loadu_epi16(vmask_lo, i3));
      const __m512h vi3x_curr1 = _mm512_castsi512_ph(_mm512_maskz_loadu_epi16(vmask_hi, i3 + 32));
      const __m512h vi4x_curr0 = _mm512_castsi512_ph(_mm512_maskz_loadu_epi16(vmask_lo, i4));
      const __m512h vi4x_curr1 = _mm512_castsi512_ph(_mm512_maskz_loadu_epi16(vmask_hi, i4 + 32));

      __m512h vi0x_even = _mm512_castsi512_ph(_mm512_permutex2var_epi16(_mm512_castph_si512(vi0x_curr0), vidx_even, _mm512_castph_si512(vi0x_curr1)));
      __m512h vi0x_odd = _mm512_castsi512_ph(_mm512_permutex2var_epi16(_mm512_castph_si512(vi0x_curr0), vidx_odd, _mm512_castph_si512(vi0x_curr1)));
      __m512h vi1x_even = _mm512_castsi512_ph(_mm512_permutex2var_epi16(_mm512_castph_si512(vi1x_curr0), vidx_even, _mm512_castph_si512(vi1x_curr1)));
      __m512h vi1x_odd = _mm512_castsi512_ph(_mm512_permutex2var_epi16(_mm512_castph_si512(vi1x_curr0), vidx_odd, _mm512_castph_si512(vi1x_curr1)));
      __m512h vi2x_even = _mm512_castsi512_ph(_mm512_permutex2var_epi16(_mm512_castph_si512(vi2x_curr0), vidx_even, _mm512_castph_si512(vi2x_curr1)));
      __m512h vi2x_odd = _mm512_castsi512_ph(_mm512_permutex2var_epi16(_mm512_castph_si512(vi2x_curr0), vidx_odd, _mm512_castph_si512(vi2x_curr1)));
      __m512h vi3x_even = _mm512_castsi512_ph(_mm512_permutex2var_epi16(_mm512_castph_si512(vi3x_curr0), vidx_even, _mm512_castph_si512(vi3x_curr1)));
      __m512h vi3x_odd = _mm512_castsi512_ph(_mm512_permutex2var_epi16(_mm512_castph_si512(vi3x_curr0), vidx_odd, _mm512_castph_si512(vi3x_curr1)));
      __m512h vi4x_even = _mm512_castsi512_ph(_mm512_permutex2var_epi16(_mm512_castph_si512(vi4x_curr0), vidx_even, _mm512_castph_si512(vi4x_curr1)));
      __m512h vi4x_odd = _mm512_castsi512_ph(_mm512_permutex2var_epi16(_mm512_castph_si512(vi4x_curr0), vidx_odd, _mm512_castph_si512(vi4x_curr1)));

      __m512h vo0p0 = _mm512_fmadd_ph(vi0x_even, vk02, vbias);
      __m512h vo0p1 = _mm512_mul_ph(vi1x_even, vk12);
      __m512h vo0p2 = _mm512_mul_ph(vi2x_even, vk22);
      __m512h vo0p3 = _mm512_mul_ph(vi3x_even, vk32);
      vo0p0 = _mm512_fmadd_ph(vi4x_even, vk42, vo0p0);

      vo0p1 = _mm512_fmadd_ph(vi0x_odd, vk03, vo0p1);
      vo0p2 = _mm512_fmadd_ph(vi1x_odd, vk13, vo0p2);
      vo0p3 = _mm512_fmadd_ph(vi2x_odd, vk23, vo0p3);
      vo0p0 = _mm512_fmadd_ph(vi3x_odd, vk33, vo0p0);
      vo0p1 = _mm512_fmadd_ph(vi4x_odd, vk43, vo0p1);

      const __m512h vi0x_odd_prev = _mm512_castsi512_ph(_mm512_permutex2var_epi16(
        _mm512_castph_si512(vi0x_odd), vidx_prev, _mm512_castph_si512(vi0x_prev_odd)));
      const __m512h vi1x_odd_prev = _mm512_castsi512_ph(_mm512_permutex2var_epi16(
        _mm512_castph_si512(vi1x_odd), vidx_prev, _mm512_castph_si512(vi1x_prev_odd)));
      const __m512h vi2x_odd_prev = _mm512_castsi512_ph(_mm512_permutex2var_epi16(
        _mm512_castph_si512(vi2x_odd), vidx_prev, _mm512_castph_si512(vi2x_prev_odd)));
      const __m512h vi3x_odd_prev = _mm512_castsi512_ph(_mm512_permutex2var_epi16(
        _mm512_castph_si512(vi3x_odd), vidx_prev, _mm512_castph_si512(vi3x_prev_odd)));
      const __m512h vi4x_odd_prev = _mm512_castsi512_ph(_mm512_permutex2var_epi16(
        _mm512_castph_si512(vi4x_odd), vidx_prev, _mm512_castph_si512(vi4x_prev_odd)));

      vo0p2 = _mm512_fmadd_ph(vi0x_odd_prev, vk01, vo0p2);
      vo0p3 = _mm512_fmadd_ph(vi1x_odd_prev, vk11, vo0p3);
      vo0p0 = _mm512_fmadd_ph(vi2x_odd_prev, vk21, vo0p0);
      vo0p1 = _mm512_fmadd_ph(vi3x_odd_prev, vk31, vo0p1);
      vo0p2 = _mm512_fmadd_ph(vi4x_odd_prev, vk41, vo0p2);

      const __m512h vi0x_even_prev = _mm512_castsi512_ph(_mm512_permutex2var_epi16(
          _mm512_castph_si512(vi0x_even), vidx_prev, _mm512_castph_si512(vi0x_prev_even)));
      const __m512h vi1x_even_prev = _mm512_castsi512_ph(_mm512_permutex2var_epi16(
          _mm512_castph_si512(vi1x_even), vidx_prev, _mm512_castph_si512(vi1x_prev_even)));
      const __m512h vi2x_even_prev = _mm512_castsi512_ph(_mm512_permutex2var_epi16(
          _mm512_castph_si512(vi2x_even), vidx_prev, _mm512_castph_si512(vi2x_prev_even)));
      const __m512h vi3x_even_prev = _mm512_castsi512_ph(_mm512_permutex2var_epi16(
          _mm512_castph_si512(vi3x_even), vidx_prev, _mm512_castph_si512(vi3x_prev_even)));
      const __m512h vi4x_even_prev = _mm512_castsi512_ph(_mm512_permutex2var_epi16(
          _mm512_castph_si512(vi4x_even), vidx_prev, _mm512_castph_si512(vi4x_prev_even)));

      vo0p3 = _mm512_fmadd_ph(vi0x_even_prev, vk00, vo0p3);
      vo0p0 = _mm512_fmadd_ph(vi1x_even_prev, vk10, vo0p0);
      vo0p1 = _mm512_fmadd_ph(vi2x_even_prev, vk20, vo0p1);
      vo0p2 = _mm512_fmadd_ph(vi3x_even_prev, vk30, vo0p2);
      vo0p3 = _mm512_fmadd_ph(vi4x_even_prev, vk40, vo0p3);

      const __m512h vzero = _mm512_setzero_ph();

      const __m512h vi0x_even_next = _mm512_castsi512_ph(_mm512_permutex2var_epi16(
        _mm512_castph_si512(vzero), vidx_next, _mm512_castph_si512(vi0x_even)));
      const __m512h vi1x_even_next = _mm512_castsi512_ph(_mm512_permutex2var_epi16(
        _mm512_castph_si512(vzero), vidx_next, _mm512_castph_si512(vi1x_even)));
      const __m512h vi2x_even_next = _mm512_castsi512_ph(_mm512_permutex2var_epi16(
        _mm512_castph_si512(vzero), vidx_next, _mm512_castph_si512(vi2x_even)));
      const __m512h vi3x_even_next = _mm512_castsi512_ph(_mm512_permutex2var_epi16(
        _mm512_castph_si512(vzero), vidx_next, _mm512_castph_si512(vi3x_even)));
      const __m512h vi4x_even_next = _mm512_castsi512_ph(_mm512_permutex2var_epi16(
        _mm512_castph_si512(vzero), vidx_next, _mm512_castph_si512(vi4x_even)));

      vo0p0 = _mm512_fmadd_ph(vi0x_even_next, vk04, vo0p0);
      vo0p1 = _mm512_fmadd_ph(vi1x_even_next, vk14, vo0p1);
      vo0p2 = _mm512_fmadd_ph(vi2x_even_next, vk24, vo0p2);
      vo0p3 = _mm512_fmadd_ph(vi3x_even_next, vk34, vo0p3);
      vo0p0 = _mm512_fmadd_ph(vi4x_even_next, vk44, vo0p0);

      vo0p0 = _mm512_add_ph(vo0p0, vo0p1);
      vo0p2 = _mm512_add_ph(vo0p2, vo0p3);
      vo0p0 = _mm512_add_ph(vo0p0, vo0p2);

      __m512h vo0 = _mm512_max_ph(vo0p0, vmin);

      vo0 = _mm512_min_ph(vo0, vmax);

      size_t w_tmp = (w + 1 * sizeof(uint16_t)) / (2 * sizeof(uint16_t));
      if XNN_LIKELY(w_tmp == 32) {
        _mm512_storeu_epi16(o0, _mm512_castph_si512(vo0));
        o0 += 32;
      } else {
        __mmask32 vmask_out = _cvtu32_mask32((uint32_t)(((uint64_t)1 << w_tmp) - 1));
        _mm512_mask_storeu_epi16(o0, vmask_out, _mm512_castph_si512(vo0));
        o0 = (uint16_t*) ((uintptr_t) o0 + w_tmp * sizeof(uint16_t));
      }
    }

    i0 = (const uint16_t*) ((uintptr_t) i2 - input_decrement);
    i1 = (const uint16_t*) ((uintptr_t) i3 - input_decrement);
    i2 = (const uint16_t*) ((uintptr_t) i4 - input_decrement);
    i3 = (const uint16_t*) ((uintptr_t) i2 + input_width);
    i4 = (const uint16_t*) ((uintptr_t) i3 + input_width);


    output_height -= 1;
    padded_input_height -= 2;
  } while (output_height != 0);
}
