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


void xnn_f32_dwconv2d_chw_ukernel_3x3p1__ssse3_6x4(
    size_t input_height,
    size_t input_width,
    const float* input,
    const float* weights,
    const float* zero,
    float* output,
    uint32_t padding_top,
    const union xnn_f32_chw_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(input_height != 0);
  assert(input_width != 0);
  assert(input_width % sizeof(float) == 0);
  assert(padding_top == 1);

  const __m128 vmask = _mm_load_ps((const float*) params->sse_stride1.mask);
  const __m128 vmax = _mm_set1_ps(params->sse_stride1.max);
  const __m128 vmin = _mm_set1_ps(params->sse_stride1.min);
  XNN_FORCE_REALIZATION(vmin);
  XNN_FORCE_REALIZATION(vmax);

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
  const float* i3 = (const float*) ((uintptr_t) i2 + input_width);
  const float* i4 = (const float*) ((uintptr_t) i3 + input_width);
  const float* i5 = (const float*) ((uintptr_t) i4 + input_width);
  const float* i6 = (const float*) ((uintptr_t) i5 + input_width);
  const float* i7 = (const float*) ((uintptr_t) i6 + input_width);

  float* o0 = output;
  float* o1 = (float*) ((uintptr_t) o0 + input_width);
  float* o2 = (float*) ((uintptr_t) o1 + input_width);
  float* o3 = (float*) ((uintptr_t) o2 + input_width);
  float* o4 = (float*) ((uintptr_t) o3 + input_width);
  float* o5 = (float*) ((uintptr_t) o4 + input_width);

  size_t output_height = input_height;
  do {
    if XNN_UNPREDICTABLE(output_height < 2) {
      i2 = zero;
      o1 = o0;
    }
    if XNN_UNPREDICTABLE(output_height < 3) {
      i3 = zero;
      o2 = o1;
    }
    if XNN_UNPREDICTABLE(output_height < 4) {
      i4 = zero;
      o3 = o2;
    }
    if XNN_UNPREDICTABLE(output_height < 5) {
      i5 = zero;
      o4 = o3;
    }
    if XNN_UNPREDICTABLE(output_height < 6) {
      i6 = zero;
      o5 = o4;
    }
    if XNN_UNPREDICTABLE(output_height < 7) {
      i7 = zero;
    }

    __m128 vi0x0123 = _mm_setzero_ps();
    __m128 vi1x0123 = _mm_setzero_ps();
    __m128 vi2x0123 = _mm_setzero_ps();
    __m128 vi3x0123 = _mm_setzero_ps();
    __m128 vi4x0123 = _mm_setzero_ps();
    __m128 vi5x0123 = _mm_setzero_ps();
    __m128 vi6x0123 = _mm_setzero_ps();
    __m128 vi7x0123 = _mm_setzero_ps();

    __m128 vi0x4567 = _mm_loadu_ps(i0);
    i0 += 4;
    __m128 vi1x4567 = _mm_loadu_ps(i1);
    i1 += 4;
    __m128 vi2x4567 = _mm_loadu_ps(i2);
    i2 += 4;
    __m128 vi3x4567 = _mm_loadu_ps(i3);
    i3 += 4;
    __m128 vi4x4567 = _mm_loadu_ps(i4);
    i4 += 4;
    __m128 vi5x4567 = _mm_loadu_ps(i5);
    i5 += 4;
    __m128 vi6x4567 = _mm_loadu_ps(i6);
    i6 += 4;
    __m128 vi7x4567 = _mm_loadu_ps(i7);
    i7 += 4;

    size_t w = input_width;
    for (; w > 4 * sizeof(float); w -= 4 * sizeof(float)) {
      const __m128 vi0x89AB = _mm_loadu_ps(i0);
      i0 += 4;
      const __m128 vi1x89AB = _mm_loadu_ps(i1);
      i1 += 4;
      const __m128 vi2x89AB = _mm_loadu_ps(i2);
      i2 += 4;
      const __m128 vi3x89AB = _mm_loadu_ps(i3);
      i3 += 4;
      const __m128 vi4x89AB = _mm_loadu_ps(i4);
      i4 += 4;
      const __m128 vi5x89AB = _mm_loadu_ps(i5);
      i5 += 4;
      const __m128 vi6x89AB = _mm_loadu_ps(i6);
      i6 += 4;
      const __m128 vi7x89AB = _mm_loadu_ps(i7);
      i7 += 4;

      __m128 vo0p0 = _mm_add_ps(vbias, _mm_mul_ps(vi0x4567, vk01));
      __m128 vo1p0 = _mm_add_ps(vbias, _mm_mul_ps(vi1x4567, vk01));
      __m128 vo2p0 = _mm_add_ps(vbias, _mm_mul_ps(vi2x4567, vk01));
      __m128 vo3p0 = _mm_add_ps(vbias, _mm_mul_ps(vi3x4567, vk01));
      __m128 vo4p0 = _mm_add_ps(vbias, _mm_mul_ps(vi4x4567, vk01));
      __m128 vo5p0 = _mm_add_ps(vbias, _mm_mul_ps(vi5x4567, vk01));
      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi1x4567, vk11));
      vo1p0 = _mm_add_ps(vo1p0, _mm_mul_ps(vi2x4567, vk11));
      vo2p0 = _mm_add_ps(vo2p0, _mm_mul_ps(vi3x4567, vk11));
      vo3p0 = _mm_add_ps(vo3p0, _mm_mul_ps(vi4x4567, vk11));
      vo4p0 = _mm_add_ps(vo4p0, _mm_mul_ps(vi5x4567, vk11));
      vo5p0 = _mm_add_ps(vo5p0, _mm_mul_ps(vi6x4567, vk11));
      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi2x4567, vk21));
      vo1p0 = _mm_add_ps(vo1p0, _mm_mul_ps(vi3x4567, vk21));
      vo2p0 = _mm_add_ps(vo2p0, _mm_mul_ps(vi4x4567, vk21));
      vo3p0 = _mm_add_ps(vo3p0, _mm_mul_ps(vi5x4567, vk21));
      vo4p0 = _mm_add_ps(vo4p0, _mm_mul_ps(vi6x4567, vk21));
      vo5p0 = _mm_add_ps(vo5p0, _mm_mul_ps(vi7x4567, vk21));

      const __m128 vi0x3456 = _mm_castsi128_ps(_mm_alignr_epi8(_mm_castps_si128(vi0x4567), _mm_castps_si128(vi0x0123), 12));
      const __m128 vi1x3456 = _mm_castsi128_ps(_mm_alignr_epi8(_mm_castps_si128(vi1x4567), _mm_castps_si128(vi1x0123), 12));
      const __m128 vi2x3456 = _mm_castsi128_ps(_mm_alignr_epi8(_mm_castps_si128(vi2x4567), _mm_castps_si128(vi2x0123), 12));
      const __m128 vi3x3456 = _mm_castsi128_ps(_mm_alignr_epi8(_mm_castps_si128(vi3x4567), _mm_castps_si128(vi3x0123), 12));
      const __m128 vi4x3456 = _mm_castsi128_ps(_mm_alignr_epi8(_mm_castps_si128(vi4x4567), _mm_castps_si128(vi4x0123), 12));
      const __m128 vi5x3456 = _mm_castsi128_ps(_mm_alignr_epi8(_mm_castps_si128(vi5x4567), _mm_castps_si128(vi5x0123), 12));
      const __m128 vi6x3456 = _mm_castsi128_ps(_mm_alignr_epi8(_mm_castps_si128(vi6x4567), _mm_castps_si128(vi6x0123), 12));
      const __m128 vi7x3456 = _mm_castsi128_ps(_mm_alignr_epi8(_mm_castps_si128(vi7x4567), _mm_castps_si128(vi7x0123), 12));

      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi0x3456, vk00));
      vo1p0 = _mm_add_ps(vo1p0, _mm_mul_ps(vi1x3456, vk00));
      vo2p0 = _mm_add_ps(vo2p0, _mm_mul_ps(vi2x3456, vk00));
      vo3p0 = _mm_add_ps(vo3p0, _mm_mul_ps(vi3x3456, vk00));
      vo4p0 = _mm_add_ps(vo4p0, _mm_mul_ps(vi4x3456, vk00));
      vo5p0 = _mm_add_ps(vo5p0, _mm_mul_ps(vi5x3456, vk00));
      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi1x3456, vk10));
      vo1p0 = _mm_add_ps(vo1p0, _mm_mul_ps(vi2x3456, vk10));
      vo2p0 = _mm_add_ps(vo2p0, _mm_mul_ps(vi3x3456, vk10));
      vo3p0 = _mm_add_ps(vo3p0, _mm_mul_ps(vi4x3456, vk10));
      vo4p0 = _mm_add_ps(vo4p0, _mm_mul_ps(vi5x3456, vk10));
      vo5p0 = _mm_add_ps(vo5p0, _mm_mul_ps(vi6x3456, vk10));
      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi2x3456, vk20));
      vo1p0 = _mm_add_ps(vo1p0, _mm_mul_ps(vi3x3456, vk20));
      vo2p0 = _mm_add_ps(vo2p0, _mm_mul_ps(vi4x3456, vk20));
      vo3p0 = _mm_add_ps(vo3p0, _mm_mul_ps(vi5x3456, vk20));
      vo4p0 = _mm_add_ps(vo4p0, _mm_mul_ps(vi6x3456, vk20));
      vo5p0 = _mm_add_ps(vo5p0, _mm_mul_ps(vi7x3456, vk20));

      vi0x0123 = vi0x4567;
      vi1x0123 = vi1x4567;
      vi2x0123 = vi2x4567;
      vi3x0123 = vi3x4567;
      vi4x0123 = vi4x4567;
      vi5x0123 = vi5x4567;
      vi6x0123 = vi6x4567;
      vi7x0123 = vi7x4567;

      const __m128 vi0x5678 = _mm_castsi128_ps(_mm_alignr_epi8(_mm_castps_si128(vi0x89AB), _mm_castps_si128(vi0x4567), 4));
      const __m128 vi1x5678 = _mm_castsi128_ps(_mm_alignr_epi8(_mm_castps_si128(vi1x89AB), _mm_castps_si128(vi1x4567), 4));
      const __m128 vi2x5678 = _mm_castsi128_ps(_mm_alignr_epi8(_mm_castps_si128(vi2x89AB), _mm_castps_si128(vi2x4567), 4));
      const __m128 vi3x5678 = _mm_castsi128_ps(_mm_alignr_epi8(_mm_castps_si128(vi3x89AB), _mm_castps_si128(vi3x4567), 4));
      const __m128 vi4x5678 = _mm_castsi128_ps(_mm_alignr_epi8(_mm_castps_si128(vi4x89AB), _mm_castps_si128(vi4x4567), 4));
      const __m128 vi5x5678 = _mm_castsi128_ps(_mm_alignr_epi8(_mm_castps_si128(vi5x89AB), _mm_castps_si128(vi5x4567), 4));
      const __m128 vi6x5678 = _mm_castsi128_ps(_mm_alignr_epi8(_mm_castps_si128(vi6x89AB), _mm_castps_si128(vi6x4567), 4));
      const __m128 vi7x5678 = _mm_castsi128_ps(_mm_alignr_epi8(_mm_castps_si128(vi7x89AB), _mm_castps_si128(vi7x4567), 4));

      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi0x5678, vk02));
      vo1p0 = _mm_add_ps(vo1p0, _mm_mul_ps(vi1x5678, vk02));
      vo2p0 = _mm_add_ps(vo2p0, _mm_mul_ps(vi2x5678, vk02));
      vo3p0 = _mm_add_ps(vo3p0, _mm_mul_ps(vi3x5678, vk02));
      vo4p0 = _mm_add_ps(vo4p0, _mm_mul_ps(vi4x5678, vk02));
      vo5p0 = _mm_add_ps(vo5p0, _mm_mul_ps(vi5x5678, vk02));
      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi1x5678, vk12));
      vo1p0 = _mm_add_ps(vo1p0, _mm_mul_ps(vi2x5678, vk12));
      vo2p0 = _mm_add_ps(vo2p0, _mm_mul_ps(vi3x5678, vk12));
      vo3p0 = _mm_add_ps(vo3p0, _mm_mul_ps(vi4x5678, vk12));
      vo4p0 = _mm_add_ps(vo4p0, _mm_mul_ps(vi5x5678, vk12));
      vo5p0 = _mm_add_ps(vo5p0, _mm_mul_ps(vi6x5678, vk12));
      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi2x5678, vk22));
      vo1p0 = _mm_add_ps(vo1p0, _mm_mul_ps(vi3x5678, vk22));
      vo2p0 = _mm_add_ps(vo2p0, _mm_mul_ps(vi4x5678, vk22));
      vo3p0 = _mm_add_ps(vo3p0, _mm_mul_ps(vi5x5678, vk22));
      vo4p0 = _mm_add_ps(vo4p0, _mm_mul_ps(vi6x5678, vk22));
      vo5p0 = _mm_add_ps(vo5p0, _mm_mul_ps(vi7x5678, vk22));

      vi0x4567 = vi0x89AB;
      vi1x4567 = vi1x89AB;
      vi2x4567 = vi2x89AB;
      vi3x4567 = vi3x89AB;
      vi4x4567 = vi4x89AB;
      vi5x4567 = vi5x89AB;
      vi6x4567 = vi6x89AB;
      vi7x4567 = vi7x89AB;


      __m128 vo0 = _mm_max_ps(vo0p0, vmin);
      __m128 vo1 = _mm_max_ps(vo1p0, vmin);
      __m128 vo2 = _mm_max_ps(vo2p0, vmin);
      __m128 vo3 = _mm_max_ps(vo3p0, vmin);
      __m128 vo4 = _mm_max_ps(vo4p0, vmin);
      __m128 vo5 = _mm_max_ps(vo5p0, vmin);

      vo0 = _mm_min_ps(vo0, vmax);
      vo1 = _mm_min_ps(vo1, vmax);
      vo2 = _mm_min_ps(vo2, vmax);
      vo3 = _mm_min_ps(vo3, vmax);
      vo4 = _mm_min_ps(vo4, vmax);
      vo5 = _mm_min_ps(vo5, vmax);

      _mm_storeu_ps(o5, vo5);
      o5 += 4;
      _mm_storeu_ps(o4, vo4);
      o4 += 4;
      _mm_storeu_ps(o3, vo3);
      o3 += 4;
      _mm_storeu_ps(o2, vo2);
      o2 += 4;
      _mm_storeu_ps(o1, vo1);
      o1 += 4;
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
      vi3x4567 = _mm_and_ps(vmask, vi3x4567);
      vi4x4567 = _mm_and_ps(vmask, vi4x4567);
      vi5x4567 = _mm_and_ps(vmask, vi5x4567);
      vi6x4567 = _mm_and_ps(vmask, vi6x4567);
      vi7x4567 = _mm_and_ps(vmask, vi7x4567);

      __m128 vo0p0 = _mm_add_ps(vbias, _mm_mul_ps(vi0x4567, vk01));
      __m128 vo1p0 = _mm_add_ps(vbias, _mm_mul_ps(vi1x4567, vk01));
      __m128 vo2p0 = _mm_add_ps(vbias, _mm_mul_ps(vi2x4567, vk01));
      __m128 vo3p0 = _mm_add_ps(vbias, _mm_mul_ps(vi3x4567, vk01));
      __m128 vo4p0 = _mm_add_ps(vbias, _mm_mul_ps(vi4x4567, vk01));
      __m128 vo5p0 = _mm_add_ps(vbias, _mm_mul_ps(vi5x4567, vk01));
      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi1x4567, vk11));
      vo1p0 = _mm_add_ps(vo1p0, _mm_mul_ps(vi2x4567, vk11));
      vo2p0 = _mm_add_ps(vo2p0, _mm_mul_ps(vi3x4567, vk11));
      vo3p0 = _mm_add_ps(vo3p0, _mm_mul_ps(vi4x4567, vk11));
      vo4p0 = _mm_add_ps(vo4p0, _mm_mul_ps(vi5x4567, vk11));
      vo5p0 = _mm_add_ps(vo5p0, _mm_mul_ps(vi6x4567, vk11));
      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi2x4567, vk21));
      vo1p0 = _mm_add_ps(vo1p0, _mm_mul_ps(vi3x4567, vk21));
      vo2p0 = _mm_add_ps(vo2p0, _mm_mul_ps(vi4x4567, vk21));
      vo3p0 = _mm_add_ps(vo3p0, _mm_mul_ps(vi5x4567, vk21));
      vo4p0 = _mm_add_ps(vo4p0, _mm_mul_ps(vi6x4567, vk21));
      vo5p0 = _mm_add_ps(vo5p0, _mm_mul_ps(vi7x4567, vk21));

      const __m128 vi0x3456 = _mm_castsi128_ps(_mm_alignr_epi8(_mm_castps_si128(vi0x4567), _mm_castps_si128(vi0x0123), 12));
      const __m128 vi1x3456 = _mm_castsi128_ps(_mm_alignr_epi8(_mm_castps_si128(vi1x4567), _mm_castps_si128(vi1x0123), 12));
      const __m128 vi2x3456 = _mm_castsi128_ps(_mm_alignr_epi8(_mm_castps_si128(vi2x4567), _mm_castps_si128(vi2x0123), 12));
      const __m128 vi3x3456 = _mm_castsi128_ps(_mm_alignr_epi8(_mm_castps_si128(vi3x4567), _mm_castps_si128(vi3x0123), 12));
      const __m128 vi4x3456 = _mm_castsi128_ps(_mm_alignr_epi8(_mm_castps_si128(vi4x4567), _mm_castps_si128(vi4x0123), 12));
      const __m128 vi5x3456 = _mm_castsi128_ps(_mm_alignr_epi8(_mm_castps_si128(vi5x4567), _mm_castps_si128(vi5x0123), 12));
      const __m128 vi6x3456 = _mm_castsi128_ps(_mm_alignr_epi8(_mm_castps_si128(vi6x4567), _mm_castps_si128(vi6x0123), 12));
      const __m128 vi7x3456 = _mm_castsi128_ps(_mm_alignr_epi8(_mm_castps_si128(vi7x4567), _mm_castps_si128(vi7x0123), 12));

      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi0x3456, vk00));
      vo1p0 = _mm_add_ps(vo1p0, _mm_mul_ps(vi1x3456, vk00));
      vo2p0 = _mm_add_ps(vo2p0, _mm_mul_ps(vi2x3456, vk00));
      vo3p0 = _mm_add_ps(vo3p0, _mm_mul_ps(vi3x3456, vk00));
      vo4p0 = _mm_add_ps(vo4p0, _mm_mul_ps(vi4x3456, vk00));
      vo5p0 = _mm_add_ps(vo5p0, _mm_mul_ps(vi5x3456, vk00));
      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi1x3456, vk10));
      vo1p0 = _mm_add_ps(vo1p0, _mm_mul_ps(vi2x3456, vk10));
      vo2p0 = _mm_add_ps(vo2p0, _mm_mul_ps(vi3x3456, vk10));
      vo3p0 = _mm_add_ps(vo3p0, _mm_mul_ps(vi4x3456, vk10));
      vo4p0 = _mm_add_ps(vo4p0, _mm_mul_ps(vi5x3456, vk10));
      vo5p0 = _mm_add_ps(vo5p0, _mm_mul_ps(vi6x3456, vk10));
      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi2x3456, vk20));
      vo1p0 = _mm_add_ps(vo1p0, _mm_mul_ps(vi3x3456, vk20));
      vo2p0 = _mm_add_ps(vo2p0, _mm_mul_ps(vi4x3456, vk20));
      vo3p0 = _mm_add_ps(vo3p0, _mm_mul_ps(vi5x3456, vk20));
      vo4p0 = _mm_add_ps(vo4p0, _mm_mul_ps(vi6x3456, vk20));
      vo5p0 = _mm_add_ps(vo5p0, _mm_mul_ps(vi7x3456, vk20));

      const __m128i vzero = _mm_setzero_si128();
      const __m128 vi0x5678 = _mm_castsi128_ps(_mm_alignr_epi8(vzero, _mm_castps_si128(vi0x4567), 4));
      const __m128 vi1x5678 = _mm_castsi128_ps(_mm_alignr_epi8(vzero, _mm_castps_si128(vi1x4567), 4));
      const __m128 vi2x5678 = _mm_castsi128_ps(_mm_alignr_epi8(vzero, _mm_castps_si128(vi2x4567), 4));
      const __m128 vi3x5678 = _mm_castsi128_ps(_mm_alignr_epi8(vzero, _mm_castps_si128(vi3x4567), 4));
      const __m128 vi4x5678 = _mm_castsi128_ps(_mm_alignr_epi8(vzero, _mm_castps_si128(vi4x4567), 4));
      const __m128 vi5x5678 = _mm_castsi128_ps(_mm_alignr_epi8(vzero, _mm_castps_si128(vi5x4567), 4));
      const __m128 vi6x5678 = _mm_castsi128_ps(_mm_alignr_epi8(vzero, _mm_castps_si128(vi6x4567), 4));
      const __m128 vi7x5678 = _mm_castsi128_ps(_mm_alignr_epi8(vzero, _mm_castps_si128(vi7x4567), 4));

      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi0x5678, vk02));
      vo1p0 = _mm_add_ps(vo1p0, _mm_mul_ps(vi1x5678, vk02));
      vo2p0 = _mm_add_ps(vo2p0, _mm_mul_ps(vi2x5678, vk02));
      vo3p0 = _mm_add_ps(vo3p0, _mm_mul_ps(vi3x5678, vk02));
      vo4p0 = _mm_add_ps(vo4p0, _mm_mul_ps(vi4x5678, vk02));
      vo5p0 = _mm_add_ps(vo5p0, _mm_mul_ps(vi5x5678, vk02));
      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi1x5678, vk12));
      vo1p0 = _mm_add_ps(vo1p0, _mm_mul_ps(vi2x5678, vk12));
      vo2p0 = _mm_add_ps(vo2p0, _mm_mul_ps(vi3x5678, vk12));
      vo3p0 = _mm_add_ps(vo3p0, _mm_mul_ps(vi4x5678, vk12));
      vo4p0 = _mm_add_ps(vo4p0, _mm_mul_ps(vi5x5678, vk12));
      vo5p0 = _mm_add_ps(vo5p0, _mm_mul_ps(vi6x5678, vk12));
      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi2x5678, vk22));
      vo1p0 = _mm_add_ps(vo1p0, _mm_mul_ps(vi3x5678, vk22));
      vo2p0 = _mm_add_ps(vo2p0, _mm_mul_ps(vi4x5678, vk22));
      vo3p0 = _mm_add_ps(vo3p0, _mm_mul_ps(vi5x5678, vk22));
      vo4p0 = _mm_add_ps(vo4p0, _mm_mul_ps(vi6x5678, vk22));
      vo5p0 = _mm_add_ps(vo5p0, _mm_mul_ps(vi7x5678, vk22));


      __m128 vo0 = _mm_max_ps(vo0p0, vmin);
      __m128 vo1 = _mm_max_ps(vo1p0, vmin);
      __m128 vo2 = _mm_max_ps(vo2p0, vmin);
      __m128 vo3 = _mm_max_ps(vo3p0, vmin);
      __m128 vo4 = _mm_max_ps(vo4p0, vmin);
      __m128 vo5 = _mm_max_ps(vo5p0, vmin);

      vo0 = _mm_min_ps(vo0, vmax);
      vo1 = _mm_min_ps(vo1, vmax);
      vo2 = _mm_min_ps(vo2, vmax);
      vo3 = _mm_min_ps(vo3, vmax);
      vo4 = _mm_min_ps(vo4, vmax);
      vo5 = _mm_min_ps(vo5, vmax);

      if XNN_LIKELY(w == 4 * sizeof(float)) {
        _mm_storeu_ps(o5, vo5);
        o5 += 4;
        _mm_storeu_ps(o4, vo4);
        o4 += 4;
        _mm_storeu_ps(o3, vo3);
        o3 += 4;
        _mm_storeu_ps(o2, vo2);
        o2 += 4;
        _mm_storeu_ps(o1, vo1);
        o1 += 4;
        _mm_storeu_ps(o0, vo0);
        o0 += 4;
      } else {
        if (w & (2 * sizeof(float))) {
          _mm_storel_pi((__m64*) o5, vo5);
          o5 += 2;
          _mm_storel_pi((__m64*) o4, vo4);
          o4 += 2;
          _mm_storel_pi((__m64*) o3, vo3);
          o3 += 2;
          _mm_storel_pi((__m64*) o2, vo2);
          o2 += 2;
          _mm_storel_pi((__m64*) o1, vo1);
          o1 += 2;
          _mm_storel_pi((__m64*) o0, vo0);
          o0 += 2;

          vo0 = _mm_movehl_ps(vo0, vo0);
          vo1 = _mm_movehl_ps(vo1, vo1);
          vo2 = _mm_movehl_ps(vo2, vo2);
          vo3 = _mm_movehl_ps(vo3, vo3);
          vo4 = _mm_movehl_ps(vo4, vo4);
          vo5 = _mm_movehl_ps(vo5, vo5);
        }
        if (w & (1 * sizeof(float))) {
          _mm_store_ss(o5, vo5);
          o5 += 1;
          _mm_store_ss(o4, vo4);
          o4 += 1;
          _mm_store_ss(o3, vo3);
          o3 += 1;
          _mm_store_ss(o2, vo2);
          o2 += 1;
          _mm_store_ss(o1, vo1);
          o1 += 1;
          _mm_store_ss(o0, vo0);
          o0 += 1;
        }
      }
    }

    i0 = (const float*) ((uintptr_t) i6 - input_decrement);
    i1 = (const float*) ((uintptr_t) i7 - input_decrement);
    i2 = (const float*) ((uintptr_t) i1 + input_width);
    i3 = (const float*) ((uintptr_t) i2 + input_width);
    i4 = (const float*) ((uintptr_t) i3 + input_width);
    i5 = (const float*) ((uintptr_t) i4 + input_width);
    i6 = (const float*) ((uintptr_t) i5 + input_width);
    i7 = (const float*) ((uintptr_t) i6 + input_width);

    o0 = o5;
    o1 = (float*) ((uintptr_t) o0 + input_width);
    o2 = (float*) ((uintptr_t) o1 + input_width);
    o3 = (float*) ((uintptr_t) o2 + input_width);
    o4 = (float*) ((uintptr_t) o3 + input_width);
    o5 = (float*) ((uintptr_t) o4 + input_width);

    output_height = doz(output_height, 6);
  } while (output_height != 0);
}
