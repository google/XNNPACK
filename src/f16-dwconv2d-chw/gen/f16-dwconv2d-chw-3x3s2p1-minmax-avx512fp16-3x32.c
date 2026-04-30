// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/f16-dwconv2d-chw/3x3s2p1-avx512fp16.c.in
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


void xnn_f16_dwconv2d_chw_ukernel_3x3s2p1__avx512fp16_3x32(
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
  assert(padding_top <= 1);

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

  const size_t input_decrement = round_down_po2(input_width, 32 /* SIMD output width */ * 2 /* subsampling */ * sizeof(uint16_t));
  const size_t output_width = round_down_po2((input_width + (2 /* padding */ - 3 /* kernel size */ + 2 /* subsampling */) * sizeof(uint16_t)) / 2, sizeof(uint16_t));

  const uint16_t* i0 = (const uint16_t*) ((uintptr_t) input - ((-padding_top) & input_width));
  const uint16_t* i1 = (const uint16_t*) ((uintptr_t) i0 + input_width);
  if XNN_UNPREDICTABLE(padding_top != 0) {
    i0 = (const uint16_t*) zero;
  }
  const uint16_t* i2 = (const uint16_t*) ((uintptr_t) i1 + input_width);
  const uint16_t* i3 = (const uint16_t*) ((uintptr_t) i2 + input_width);
  const uint16_t* i4 = (const uint16_t*) ((uintptr_t) i3 + input_width);
  const uint16_t* i5 = (const uint16_t*) ((uintptr_t) i4 + input_width);
  const uint16_t* i6 = (const uint16_t*) ((uintptr_t) i5 + input_width);

  uint16_t* o0 = (uint16_t*) output;
  uint16_t* o1 = (uint16_t*) ((uintptr_t) o0 + output_width);
  uint16_t* o2 = (uint16_t*) ((uintptr_t) o1 + output_width);

  size_t padded_input_height = input_height + padding_top + 1 /* padding bottom */;
  size_t output_height = (padded_input_height - 3 /* kernel size */ + 2 /* subsampling */) / 2;
  do {
    if XNN_UNPREDICTABLE(padded_input_height < 4) {
      i2 = (const uint16_t*) zero;
    }
    if XNN_UNPREDICTABLE(padded_input_height < 5) {
      i3 = (const uint16_t*) zero;
      o1 = o0;
    }
    if XNN_UNPREDICTABLE(padded_input_height < 6) {
      i4 = (const uint16_t*) zero;
    }
    if XNN_UNPREDICTABLE(padded_input_height < 7) {
      i5 = (const uint16_t*) zero;
      o2 = o1;
    }
    if XNN_UNPREDICTABLE(padded_input_height < 8) {
      i6 = (const uint16_t*) zero;
    }

    __m512h vi0x_prev_odd = _mm512_setzero_ph();
    __m512h vi1x_prev_odd = _mm512_setzero_ph();
    __m512h vi2x_prev_odd = _mm512_setzero_ph();
    __m512h vi3x_prev_odd = _mm512_setzero_ph();
    __m512h vi4x_prev_odd = _mm512_setzero_ph();
    __m512h vi5x_prev_odd = _mm512_setzero_ph();
    __m512h vi6x_prev_odd = _mm512_setzero_ph();

    size_t w = input_width;
    for (; w >= 64 * sizeof(uint16_t); w -= 64 * sizeof(uint16_t)) {
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
      const __m512h vi5x_curr0 = _mm512_castsi512_ph(_mm512_loadu_epi16(i5));
      const __m512h vi5x_curr1 = _mm512_castsi512_ph(_mm512_loadu_epi16(i5 + 32));
      i5 += 64;
      const __m512h vi6x_curr0 = _mm512_castsi512_ph(_mm512_loadu_epi16(i6));
      const __m512h vi6x_curr1 = _mm512_castsi512_ph(_mm512_loadu_epi16(i6 + 32));
      i6 += 64;

      const __m512h vi0x_even = _mm512_castsi512_ph(_mm512_permutex2var_epi16(_mm512_castph_si512(vi0x_curr0), vidx_even, _mm512_castph_si512(vi0x_curr1)));
      const __m512h vi0x_odd = _mm512_castsi512_ph(_mm512_permutex2var_epi16(_mm512_castph_si512(vi0x_curr0), vidx_odd, _mm512_castph_si512(vi0x_curr1)));
      const __m512h vi1x_even = _mm512_castsi512_ph(_mm512_permutex2var_epi16(_mm512_castph_si512(vi1x_curr0), vidx_even, _mm512_castph_si512(vi1x_curr1)));
      const __m512h vi1x_odd = _mm512_castsi512_ph(_mm512_permutex2var_epi16(_mm512_castph_si512(vi1x_curr0), vidx_odd, _mm512_castph_si512(vi1x_curr1)));
      const __m512h vi2x_even = _mm512_castsi512_ph(_mm512_permutex2var_epi16(_mm512_castph_si512(vi2x_curr0), vidx_even, _mm512_castph_si512(vi2x_curr1)));
      const __m512h vi2x_odd = _mm512_castsi512_ph(_mm512_permutex2var_epi16(_mm512_castph_si512(vi2x_curr0), vidx_odd, _mm512_castph_si512(vi2x_curr1)));
      const __m512h vi3x_even = _mm512_castsi512_ph(_mm512_permutex2var_epi16(_mm512_castph_si512(vi3x_curr0), vidx_even, _mm512_castph_si512(vi3x_curr1)));
      const __m512h vi3x_odd = _mm512_castsi512_ph(_mm512_permutex2var_epi16(_mm512_castph_si512(vi3x_curr0), vidx_odd, _mm512_castph_si512(vi3x_curr1)));
      const __m512h vi4x_even = _mm512_castsi512_ph(_mm512_permutex2var_epi16(_mm512_castph_si512(vi4x_curr0), vidx_even, _mm512_castph_si512(vi4x_curr1)));
      const __m512h vi4x_odd = _mm512_castsi512_ph(_mm512_permutex2var_epi16(_mm512_castph_si512(vi4x_curr0), vidx_odd, _mm512_castph_si512(vi4x_curr1)));
      const __m512h vi5x_even = _mm512_castsi512_ph(_mm512_permutex2var_epi16(_mm512_castph_si512(vi5x_curr0), vidx_even, _mm512_castph_si512(vi5x_curr1)));
      const __m512h vi5x_odd = _mm512_castsi512_ph(_mm512_permutex2var_epi16(_mm512_castph_si512(vi5x_curr0), vidx_odd, _mm512_castph_si512(vi5x_curr1)));
      const __m512h vi6x_even = _mm512_castsi512_ph(_mm512_permutex2var_epi16(_mm512_castph_si512(vi6x_curr0), vidx_even, _mm512_castph_si512(vi6x_curr1)));
      const __m512h vi6x_odd = _mm512_castsi512_ph(_mm512_permutex2var_epi16(_mm512_castph_si512(vi6x_curr0), vidx_odd, _mm512_castph_si512(vi6x_curr1)));

      __m512h vo0p0 = _mm512_fmadd_ph(vi0x_even, vk01, vbias);
      __m512h vo1p0 = _mm512_fmadd_ph(vi2x_even, vk01, vbias);
      __m512h vo2p0 = _mm512_fmadd_ph(vi4x_even, vk01, vbias);
      vo0p0 = _mm512_fmadd_ph(vi1x_even, vk11, vo0p0);
      vo1p0 = _mm512_fmadd_ph(vi3x_even, vk11, vo1p0);
      vo2p0 = _mm512_fmadd_ph(vi5x_even, vk11, vo2p0);
      vo0p0 = _mm512_fmadd_ph(vi2x_even, vk21, vo0p0);
      vo1p0 = _mm512_fmadd_ph(vi4x_even, vk21, vo1p0);
      vo2p0 = _mm512_fmadd_ph(vi6x_even, vk21, vo2p0);

      const __m512h vi0x_odd_next = _mm512_castsi512_ph(_mm512_permutex2var_epi16(
        _mm512_castph_si512(vi0x_odd), vidx_next, _mm512_castph_si512(vi0x_odd)));
      const __m512h vi1x_odd_next = _mm512_castsi512_ph(_mm512_permutex2var_epi16(
        _mm512_castph_si512(vi1x_odd), vidx_next, _mm512_castph_si512(vi1x_odd)));
      const __m512h vi2x_odd_next = _mm512_castsi512_ph(_mm512_permutex2var_epi16(
        _mm512_castph_si512(vi2x_odd), vidx_next, _mm512_castph_si512(vi2x_odd)));
      const __m512h vi3x_odd_next = _mm512_castsi512_ph(_mm512_permutex2var_epi16(
        _mm512_castph_si512(vi3x_odd), vidx_next, _mm512_castph_si512(vi3x_odd)));
      const __m512h vi4x_odd_next = _mm512_castsi512_ph(_mm512_permutex2var_epi16(
        _mm512_castph_si512(vi4x_odd), vidx_next, _mm512_castph_si512(vi4x_odd)));
      const __m512h vi5x_odd_next = _mm512_castsi512_ph(_mm512_permutex2var_epi16(
        _mm512_castph_si512(vi5x_odd), vidx_next, _mm512_castph_si512(vi5x_odd)));
      const __m512h vi6x_odd_next = _mm512_castsi512_ph(_mm512_permutex2var_epi16(
        _mm512_castph_si512(vi6x_odd), vidx_next, _mm512_castph_si512(vi6x_odd)));

      vo0p0 = _mm512_fmadd_ph(vi0x_odd_next, vk02, vo0p0);
      vo1p0 = _mm512_fmadd_ph(vi2x_odd_next, vk02, vo1p0);
      vo2p0 = _mm512_fmadd_ph(vi4x_odd_next, vk02, vo2p0);
      vo0p0 = _mm512_fmadd_ph(vi1x_odd_next, vk12, vo0p0);
      vo1p0 = _mm512_fmadd_ph(vi3x_odd_next, vk12, vo1p0);
      vo2p0 = _mm512_fmadd_ph(vi5x_odd_next, vk12, vo2p0);
      vo0p0 = _mm512_fmadd_ph(vi2x_odd_next, vk22, vo0p0);
      vo1p0 = _mm512_fmadd_ph(vi4x_odd_next, vk22, vo1p0);
      vo2p0 = _mm512_fmadd_ph(vi6x_odd_next, vk22, vo2p0);

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
      const __m512h vi5x_odd_prev = _mm512_castsi512_ph(_mm512_permutex2var_epi16(
        _mm512_castph_si512(vi5x_odd), vidx_prev, _mm512_castph_si512(vi5x_prev_odd)));
      const __m512h vi6x_odd_prev = _mm512_castsi512_ph(_mm512_permutex2var_epi16(
        _mm512_castph_si512(vi6x_odd), vidx_prev, _mm512_castph_si512(vi6x_prev_odd)));

      vi0x_prev_odd = vi0x_odd;
      vi1x_prev_odd = vi1x_odd;
      vi2x_prev_odd = vi2x_odd;
      vi3x_prev_odd = vi3x_odd;
      vi4x_prev_odd = vi4x_odd;
      vi5x_prev_odd = vi5x_odd;
      vi6x_prev_odd = vi6x_odd;

      vo0p0 = _mm512_fmadd_ph(vi0x_odd_prev, vk00, vo0p0);
      vo1p0 = _mm512_fmadd_ph(vi2x_odd_prev, vk00, vo1p0);
      vo2p0 = _mm512_fmadd_ph(vi4x_odd_prev, vk00, vo2p0);
      vo0p0 = _mm512_fmadd_ph(vi1x_odd_prev, vk10, vo0p0);
      vo1p0 = _mm512_fmadd_ph(vi3x_odd_prev, vk10, vo1p0);
      vo2p0 = _mm512_fmadd_ph(vi5x_odd_prev, vk10, vo2p0);
      vo0p0 = _mm512_fmadd_ph(vi2x_odd_prev, vk20, vo0p0);
      vo1p0 = _mm512_fmadd_ph(vi4x_odd_prev, vk20, vo1p0);
      vo2p0 = _mm512_fmadd_ph(vi6x_odd_prev, vk20, vo2p0);


      __m512h vo0 = _mm512_max_ph(vo0p0, vmin);
      __m512h vo1 = _mm512_max_ph(vo1p0, vmin);
      __m512h vo2 = _mm512_max_ph(vo2p0, vmin);

      vo0 = _mm512_min_ph(vo0, vmax);
      vo1 = _mm512_min_ph(vo1, vmax);
      vo2 = _mm512_min_ph(vo2, vmax);

      _mm512_storeu_epi16(o2, _mm512_castph_si512(vo2));
      o2 += 32;
      _mm512_storeu_epi16(o1, _mm512_castph_si512(vo1));
      o1 += 32;
      _mm512_storeu_epi16(o0, _mm512_castph_si512(vo0));
      o0 += 32;
    }
    // Potentially process the last block of 0..63 pixels.
    assert(w < 64 * sizeof(uint16_t));
    if XNN_LIKELY(w != 0) {
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
      const __m512h vi5x_curr0 = _mm512_castsi512_ph(_mm512_maskz_loadu_epi16(vmask_lo, i5));
      const __m512h vi5x_curr1 = _mm512_castsi512_ph(_mm512_maskz_loadu_epi16(vmask_hi, i5 + 32));
      const __m512h vi6x_curr0 = _mm512_castsi512_ph(_mm512_maskz_loadu_epi16(vmask_lo, i6));
      const __m512h vi6x_curr1 = _mm512_castsi512_ph(_mm512_maskz_loadu_epi16(vmask_hi, i6 + 32));

      const __m512h vi0x_even = _mm512_castsi512_ph(_mm512_permutex2var_epi16(_mm512_castph_si512(vi0x_curr0), vidx_even, _mm512_castph_si512(vi0x_curr1)));
      const __m512h vi0x_odd = _mm512_castsi512_ph(_mm512_permutex2var_epi16(_mm512_castph_si512(vi0x_curr0), vidx_odd, _mm512_castph_si512(vi0x_curr1)));
      const __m512h vi1x_even = _mm512_castsi512_ph(_mm512_permutex2var_epi16(_mm512_castph_si512(vi1x_curr0), vidx_even, _mm512_castph_si512(vi1x_curr1)));
      const __m512h vi1x_odd = _mm512_castsi512_ph(_mm512_permutex2var_epi16(_mm512_castph_si512(vi1x_curr0), vidx_odd, _mm512_castph_si512(vi1x_curr1)));
      const __m512h vi2x_even = _mm512_castsi512_ph(_mm512_permutex2var_epi16(_mm512_castph_si512(vi2x_curr0), vidx_even, _mm512_castph_si512(vi2x_curr1)));
      const __m512h vi2x_odd = _mm512_castsi512_ph(_mm512_permutex2var_epi16(_mm512_castph_si512(vi2x_curr0), vidx_odd, _mm512_castph_si512(vi2x_curr1)));
      const __m512h vi3x_even = _mm512_castsi512_ph(_mm512_permutex2var_epi16(_mm512_castph_si512(vi3x_curr0), vidx_even, _mm512_castph_si512(vi3x_curr1)));
      const __m512h vi3x_odd = _mm512_castsi512_ph(_mm512_permutex2var_epi16(_mm512_castph_si512(vi3x_curr0), vidx_odd, _mm512_castph_si512(vi3x_curr1)));
      const __m512h vi4x_even = _mm512_castsi512_ph(_mm512_permutex2var_epi16(_mm512_castph_si512(vi4x_curr0), vidx_even, _mm512_castph_si512(vi4x_curr1)));
      const __m512h vi4x_odd = _mm512_castsi512_ph(_mm512_permutex2var_epi16(_mm512_castph_si512(vi4x_curr0), vidx_odd, _mm512_castph_si512(vi4x_curr1)));
      const __m512h vi5x_even = _mm512_castsi512_ph(_mm512_permutex2var_epi16(_mm512_castph_si512(vi5x_curr0), vidx_even, _mm512_castph_si512(vi5x_curr1)));
      const __m512h vi5x_odd = _mm512_castsi512_ph(_mm512_permutex2var_epi16(_mm512_castph_si512(vi5x_curr0), vidx_odd, _mm512_castph_si512(vi5x_curr1)));
      const __m512h vi6x_even = _mm512_castsi512_ph(_mm512_permutex2var_epi16(_mm512_castph_si512(vi6x_curr0), vidx_even, _mm512_castph_si512(vi6x_curr1)));
      const __m512h vi6x_odd = _mm512_castsi512_ph(_mm512_permutex2var_epi16(_mm512_castph_si512(vi6x_curr0), vidx_odd, _mm512_castph_si512(vi6x_curr1)));

      __m512h vo0p0 = _mm512_fmadd_ph(vi0x_even, vk01, vbias);
      __m512h vo1p0 = _mm512_fmadd_ph(vi2x_even, vk01, vbias);
      __m512h vo2p0 = _mm512_fmadd_ph(vi4x_even, vk01, vbias);
      vo0p0 = _mm512_fmadd_ph(vi1x_even, vk11, vo0p0);
      vo1p0 = _mm512_fmadd_ph(vi3x_even, vk11, vo1p0);
      vo2p0 = _mm512_fmadd_ph(vi5x_even, vk11, vo2p0);
      vo0p0 = _mm512_fmadd_ph(vi2x_even, vk21, vo0p0);
      vo1p0 = _mm512_fmadd_ph(vi4x_even, vk21, vo1p0);
      vo2p0 = _mm512_fmadd_ph(vi6x_even, vk21, vo2p0);

      const __m512h vi0x_odd_next = _mm512_castsi512_ph(_mm512_permutex2var_epi16(
        _mm512_castph_si512(vi0x_odd), vidx_next, _mm512_castph_si512(vi0x_odd)));
      const __m512h vi1x_odd_next = _mm512_castsi512_ph(_mm512_permutex2var_epi16(
        _mm512_castph_si512(vi1x_odd), vidx_next, _mm512_castph_si512(vi1x_odd)));
      const __m512h vi2x_odd_next = _mm512_castsi512_ph(_mm512_permutex2var_epi16(
        _mm512_castph_si512(vi2x_odd), vidx_next, _mm512_castph_si512(vi2x_odd)));
      const __m512h vi3x_odd_next = _mm512_castsi512_ph(_mm512_permutex2var_epi16(
        _mm512_castph_si512(vi3x_odd), vidx_next, _mm512_castph_si512(vi3x_odd)));
      const __m512h vi4x_odd_next = _mm512_castsi512_ph(_mm512_permutex2var_epi16(
        _mm512_castph_si512(vi4x_odd), vidx_next, _mm512_castph_si512(vi4x_odd)));
      const __m512h vi5x_odd_next = _mm512_castsi512_ph(_mm512_permutex2var_epi16(
        _mm512_castph_si512(vi5x_odd), vidx_next, _mm512_castph_si512(vi5x_odd)));
      const __m512h vi6x_odd_next = _mm512_castsi512_ph(_mm512_permutex2var_epi16(
        _mm512_castph_si512(vi6x_odd), vidx_next, _mm512_castph_si512(vi6x_odd)));

      vo0p0 = _mm512_fmadd_ph(vi0x_odd_next, vk02, vo0p0);
      vo1p0 = _mm512_fmadd_ph(vi2x_odd_next, vk02, vo1p0);
      vo2p0 = _mm512_fmadd_ph(vi4x_odd_next, vk02, vo2p0);
      vo0p0 = _mm512_fmadd_ph(vi1x_odd_next, vk12, vo0p0);
      vo1p0 = _mm512_fmadd_ph(vi3x_odd_next, vk12, vo1p0);
      vo2p0 = _mm512_fmadd_ph(vi5x_odd_next, vk12, vo2p0);
      vo0p0 = _mm512_fmadd_ph(vi2x_odd_next, vk22, vo0p0);
      vo1p0 = _mm512_fmadd_ph(vi4x_odd_next, vk22, vo1p0);
      vo2p0 = _mm512_fmadd_ph(vi6x_odd_next, vk22, vo2p0);

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
      const __m512h vi5x_odd_prev = _mm512_castsi512_ph(_mm512_permutex2var_epi16(
        _mm512_castph_si512(vi5x_odd), vidx_prev, _mm512_castph_si512(vi5x_prev_odd)));
      const __m512h vi6x_odd_prev = _mm512_castsi512_ph(_mm512_permutex2var_epi16(
        _mm512_castph_si512(vi6x_odd), vidx_prev, _mm512_castph_si512(vi6x_prev_odd)));

      vi0x_prev_odd = vi0x_odd;
      vi1x_prev_odd = vi1x_odd;
      vi2x_prev_odd = vi2x_odd;
      vi3x_prev_odd = vi3x_odd;
      vi4x_prev_odd = vi4x_odd;
      vi5x_prev_odd = vi5x_odd;
      vi6x_prev_odd = vi6x_odd;

      vo0p0 = _mm512_fmadd_ph(vi0x_odd_prev, vk00, vo0p0);
      vo1p0 = _mm512_fmadd_ph(vi2x_odd_prev, vk00, vo1p0);
      vo2p0 = _mm512_fmadd_ph(vi4x_odd_prev, vk00, vo2p0);
      vo0p0 = _mm512_fmadd_ph(vi1x_odd_prev, vk10, vo0p0);
      vo1p0 = _mm512_fmadd_ph(vi3x_odd_prev, vk10, vo1p0);
      vo2p0 = _mm512_fmadd_ph(vi5x_odd_prev, vk10, vo2p0);
      vo0p0 = _mm512_fmadd_ph(vi2x_odd_prev, vk20, vo0p0);
      vo1p0 = _mm512_fmadd_ph(vi4x_odd_prev, vk20, vo1p0);
      vo2p0 = _mm512_fmadd_ph(vi6x_odd_prev, vk20, vo2p0);


      __m512h vo0 = _mm512_max_ph(vo0p0, vmin);
      __m512h vo1 = _mm512_max_ph(vo1p0, vmin);
      __m512h vo2 = _mm512_max_ph(vo2p0, vmin);

      vo0 = _mm512_min_ph(vo0, vmax);
      vo1 = _mm512_min_ph(vo1, vmax);
      vo2 = _mm512_min_ph(vo2, vmax);

      size_t w_tmp = (w + 1 * sizeof(uint16_t)) / (2 * sizeof(uint16_t));
      if XNN_LIKELY(w_tmp == 32) {
        _mm512_storeu_epi16(o2, _mm512_castph_si512(vo2));
        o2 += 32;
        _mm512_storeu_epi16(o1, _mm512_castph_si512(vo1));
        o1 += 32;
        _mm512_storeu_epi16(o0, _mm512_castph_si512(vo0));
        o0 += 32;
      } else {
        __mmask32 vmask_out = _cvtu32_mask32((uint32_t)(((uint64_t)1 << w_tmp) - 1));
        _mm512_mask_storeu_epi16(o2, vmask_out, _mm512_castph_si512(vo2));
        o2 = (uint16_t*) ((uintptr_t) o2 + w_tmp * sizeof(uint16_t));
        _mm512_mask_storeu_epi16(o1, vmask_out, _mm512_castph_si512(vo1));
        o1 = (uint16_t*) ((uintptr_t) o1 + w_tmp * sizeof(uint16_t));
        _mm512_mask_storeu_epi16(o0, vmask_out, _mm512_castph_si512(vo0));
        o0 = (uint16_t*) ((uintptr_t) o0 + w_tmp * sizeof(uint16_t));
      }
    }

    i0 = (const uint16_t*) ((uintptr_t) i6 - input_decrement);
    i1 = (const uint16_t*) ((uintptr_t) i0 + input_width);
    i2 = (const uint16_t*) ((uintptr_t) i1 + input_width);
    i3 = (const uint16_t*) ((uintptr_t) i2 + input_width);
    i4 = (const uint16_t*) ((uintptr_t) i3 + input_width);
    i5 = (const uint16_t*) ((uintptr_t) i4 + input_width);
    i6 = (const uint16_t*) ((uintptr_t) i5 + input_width);

    o0 = o2;
    o1 = (uint16_t*) ((uintptr_t) o0 + output_width);
    o2 = (uint16_t*) ((uintptr_t) o1 + output_width);

    output_height = doz(output_height, 3);
    padded_input_height = doz(padded_input_height, 6);
  } while (output_height != 0);
}
