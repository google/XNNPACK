// Auto-generated file. Do not edit!
//   Template: src/f32-dwconv2d-chw/3x3p1-ssse3.c.in
//   Generator: tools/xngen
//
// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <tmmintrin.h>

#include "xnnpack/dwconv.h"
#include "xnnpack/math.h"


void xnn_f32_dwconv2d_chw_ukernel_3x3p1__ssse3_1x4(
    size_t input_height,
    size_t input_width,
    const float* input,
    const float* weights,
    const float* zero,
    float* output,
    uint32_t padding_top,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(input_height != 0);
  assert(input_width != 0);
  assert(input_width % sizeof(float) == 0);
  assert(padding_top == 1);

  const __m128 vmin = _mm_set1_ps(params->scalar.min);
  const __m128 vmax = _mm_set1_ps(params->scalar.max);
  XNN_FORCE_REALIZATION(vmin);
  XNN_FORCE_REALIZATION(vmax);

  static const int32_t mask_table[7] = {-1, -1, -1, -1, 0, 0, 0};
  const __m128 vmask = _mm_loadu_ps((const float*) &mask_table[3 - (((input_width >> 2) - 1) & 3)]);

  const __m128 vbias = _mm_load1_ps(weights);
  const __m128 vk00 = _mm_load1_ps(weights + 1);
  const __m128 vk01 = _mm_load1_ps(weights + 2);
  const __m128 vk02 = _mm_load1_ps(weights + 3);
  const __m128 vk10 = _mm_load1_ps(weights + 4);
  const __m128 vk11 = _mm_load1_ps(weights + 5);
  const __m128 vk12 = _mm_load1_ps(weights + 6);
  const __m128 vk20 = _mm_load1_ps(weights + 7);
  const __m128 vk21 = _mm_load1_ps(weights + 8);
  const __m128 vk22 = _mm_load1_ps(weights + 9);

  const size_t input_decrement = round_up_po2(input_width, 4 * sizeof(float));

  const float* i0 = zero;
  const float* i1 = input;
  const float* i2 = (const float*) ((uintptr_t) i1 + input_width);

  float* o0 = output;

  size_t output_height = input_height;
  do {
    if XNN_UNPREDICTABLE(output_height < 2) {
      i2 = zero;
    }

    __m128 vi0x0123 = _mm_setzero_ps();
    __m128 vi1x0123 = _mm_setzero_ps();
    __m128 vi2x0123 = _mm_setzero_ps();

    __m128 vi0x4567 = _mm_loadu_ps(i0);
    i0 += 4;
    __m128 vi1x4567 = _mm_loadu_ps(i1);
    i1 += 4;
    __m128 vi2x4567 = _mm_loadu_ps(i2);
    i2 += 4;

    size_t w = input_width;
    for (; w > 4 * sizeof(float); w -= 4 * sizeof(float)) {
      const __m128 vi0x89AB = _mm_loadu_ps(i0);
      i0 += 4;
      const __m128 vi1x89AB = _mm_loadu_ps(i1);
      i1 += 4;
      const __m128 vi2x89AB = _mm_loadu_ps(i2);
      i2 += 4;

      __m128 vo0p0 = _mm_add_ps(vbias, _mm_mul_ps(vi0x4567, vk01));
      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi1x4567, vk11));
      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi2x4567, vk21));

      const __m128 vi0x3456 = _mm_castsi128_ps(_mm_alignr_epi8(_mm_castps_si128(vi0x4567), _mm_castps_si128(vi0x0123), 12));
      const __m128 vi1x3456 = _mm_castsi128_ps(_mm_alignr_epi8(_mm_castps_si128(vi1x4567), _mm_castps_si128(vi1x0123), 12));
      const __m128 vi2x3456 = _mm_castsi128_ps(_mm_alignr_epi8(_mm_castps_si128(vi2x4567), _mm_castps_si128(vi2x0123), 12));

      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi0x3456, vk00));
      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi1x3456, vk10));
      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi2x3456, vk20));

      vi0x0123 = vi0x4567;
      vi1x0123 = vi1x4567;
      vi2x0123 = vi2x4567;

      const __m128 vi0x5678 = _mm_castsi128_ps(_mm_alignr_epi8(_mm_castps_si128(vi0x89AB), _mm_castps_si128(vi0x4567), 4));
      const __m128 vi1x5678 = _mm_castsi128_ps(_mm_alignr_epi8(_mm_castps_si128(vi1x89AB), _mm_castps_si128(vi1x4567), 4));
      const __m128 vi2x5678 = _mm_castsi128_ps(_mm_alignr_epi8(_mm_castps_si128(vi2x89AB), _mm_castps_si128(vi2x4567), 4));

      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi0x5678, vk02));
      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi1x5678, vk12));
      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi2x5678, vk22));

      vi0x4567 = vi0x89AB;
      vi1x4567 = vi1x89AB;
      vi2x4567 = vi2x89AB;


      __m128 vo0 = _mm_max_ps(vo0p0, vmin);

      vo0 = _mm_min_ps(vo0, vmax);

      _mm_storeu_ps(o0, vo0);
      o0 += 4;
    }
    // Always process the last block of 1..4 pixels.
    assert(w >= 1 * sizeof(float));
    assert(w <= 4 * sizeof(float));
    {
      vi0x4567 = _mm_and_ps(vmask, vi0x4567);
      vi1x4567 = _mm_and_ps(vmask, vi1x4567);
      vi2x4567 = _mm_and_ps(vmask, vi2x4567);

      __m128 vo0p0 = _mm_add_ps(vbias, _mm_mul_ps(vi0x4567, vk01));
      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi1x4567, vk11));
      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi2x4567, vk21));

      const __m128 vi0x3456 = _mm_castsi128_ps(_mm_alignr_epi8(_mm_castps_si128(vi0x4567), _mm_castps_si128(vi0x0123), 12));
      const __m128 vi1x3456 = _mm_castsi128_ps(_mm_alignr_epi8(_mm_castps_si128(vi1x4567), _mm_castps_si128(vi1x0123), 12));
      const __m128 vi2x3456 = _mm_castsi128_ps(_mm_alignr_epi8(_mm_castps_si128(vi2x4567), _mm_castps_si128(vi2x0123), 12));

      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi0x3456, vk00));
      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi1x3456, vk10));
      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi2x3456, vk20));

      const __m128i vzero = _mm_setzero_si128();
      const __m128 vi0x5678 = _mm_castsi128_ps(_mm_alignr_epi8(vzero, _mm_castps_si128(vi0x4567), 4));
      const __m128 vi1x5678 = _mm_castsi128_ps(_mm_alignr_epi8(vzero, _mm_castps_si128(vi1x4567), 4));
      const __m128 vi2x5678 = _mm_castsi128_ps(_mm_alignr_epi8(vzero, _mm_castps_si128(vi2x4567), 4));

      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi0x5678, vk02));
      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi1x5678, vk12));
      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi2x5678, vk22));


      __m128 vo0 = _mm_max_ps(vo0p0, vmin);

      vo0 = _mm_min_ps(vo0, vmax);

      if XNN_LIKELY(w == 4 * sizeof(float)) {
        _mm_storeu_ps(o0, vo0);
        o0 += 4;
      } else {
        if (w & (2 * sizeof(float))) {
          _mm_storel_pi((__m64*) o0, vo0);
          o0 += 2;

          vo0 = _mm_movehl_ps(vo0, vo0);
        }
        if (w & (1 * sizeof(float))) {
          _mm_store_ss(o0, vo0);
          o0 += 1;
        }
      }
    }

    i0 = (const float*) ((uintptr_t) i1 - input_decrement);
    i1 = (const float*) ((uintptr_t) i2 - input_decrement);
    i2 = (const float*) ((uintptr_t) i1 + input_width);


  } while (--output_height != 0);
}
