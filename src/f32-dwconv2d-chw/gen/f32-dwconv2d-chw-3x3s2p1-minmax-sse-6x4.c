// Auto-generated file. Do not edit!
//   Template: src/f32-dwconv2d-chw/3x3s2p1-sse.c.in
//   Generator: tools/xngen
//
// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <xmmintrin.h>

#include <xnnpack/dwconv.h>
#include <xnnpack/math.h>


void xnn_f32_dwconv2d_chw_ukernel_3x3s2p1__sse_6x4(
    size_t input_height,
    size_t input_width,
    const float* input,
    const float* weights,
    const float* zero,
    float* output,
    uint32_t padding_top,
    const union xnn_f32_chw_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(input_height != 0);
  assert(input_width != 0);
  assert(input_width % sizeof(float) == 0);
  assert(padding_top >= 0);
  assert(padding_top <= 1);

  const __m128 vmask_even = _mm_load_ps((const float*) params->sse.mask_even);
  const __m128 vmask_odd  = _mm_load_ps((const float*) params->sse.mask_odd);
  const __m128 vmax = _mm_load_ps(params->sse.max);
  const __m128 vmin = _mm_load_ps(params->sse.min);

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

  const size_t input_decrement = round_down_po2(input_width, 4 /* SIMD output width */ * 2 /* subsampling */ * sizeof(float));
  const size_t output_width = round_down_po2((input_width + (2 /* padding */ - 3 /* kernel size */ + 2 /* subsampling */) * sizeof(float)) / 2, sizeof(float));

  const float* i0 = (const float*) ((uintptr_t) input - ((-padding_top) & input_width));
  const float* i1 = (const float*) ((uintptr_t) i0 + input_width);
  if XNN_UNPREDICTABLE(padding_top != 0) {
    i0 = zero;
  }
  const float* i2 = (const float*) ((uintptr_t) i1 + input_width);
  const float* i3 = (const float*) ((uintptr_t) i2 + input_width);
  const float* i4 = (const float*) ((uintptr_t) i3 + input_width);
  const float* i5 = (const float*) ((uintptr_t) i4 + input_width);
  const float* i6 = (const float*) ((uintptr_t) i5 + input_width);
  const float* i7 = (const float*) ((uintptr_t) i6 + input_width);
  const float* i8 = (const float*) ((uintptr_t) i7 + input_width);
  const float* i9 = (const float*) ((uintptr_t) i8 + input_width);
  const float* i10 = (const float*) ((uintptr_t) i9 + input_width);
  const float* i11 = (const float*) ((uintptr_t) i10 + input_width);
  const float* i12 = (const float*) ((uintptr_t) i11 + input_width);

  float* o0 = output;
  float* o1 = (float*) ((uintptr_t) o0 + output_width);
  float* o2 = (float*) ((uintptr_t) o1 + output_width);
  float* o3 = (float*) ((uintptr_t) o2 + output_width);
  float* o4 = (float*) ((uintptr_t) o3 + output_width);
  float* o5 = (float*) ((uintptr_t) o4 + output_width);

  size_t padded_input_height = input_height + padding_top + 1 /* padding bottom */;
  size_t output_height = (padded_input_height - 3 /* kernel size */ + 2 /* subsampling */) / 2;
  do {
    if XNN_UNPREDICTABLE(padded_input_height < 4) {
      i2 = zero;
    }
    if XNN_UNPREDICTABLE(padded_input_height < 5) {
      i3 = zero;
      o1 = o0;
    }
    if XNN_UNPREDICTABLE(padded_input_height < 6) {
      i4 = zero;
    }
    if XNN_UNPREDICTABLE(padded_input_height < 7) {
      i5 = zero;
      o2 = o1;
    }
    if XNN_UNPREDICTABLE(padded_input_height < 8) {
      i6 = zero;
    }
    if XNN_UNPREDICTABLE(padded_input_height < 9) {
      i7 = zero;
      o3 = o2;
    }
    if XNN_UNPREDICTABLE(padded_input_height < 10) {
      i8 = zero;
    }
    if XNN_UNPREDICTABLE(padded_input_height < 11) {
      i9 = zero;
      o4 = o3;
    }
    if XNN_UNPREDICTABLE(padded_input_height < 12) {
      i10 = zero;
    }
    if XNN_UNPREDICTABLE(padded_input_height < 13) {
      i11 = zero;
      o5 = o4;
    }
    if XNN_UNPREDICTABLE(padded_input_height < 14) {
      i12 = zero;
    }

    __m128 vi0x7531 = _mm_setzero_ps();
    __m128 vi1x7531 = _mm_setzero_ps();
    __m128 vi2x7531 = _mm_setzero_ps();
    __m128 vi3x7531 = _mm_setzero_ps();
    __m128 vi4x7531 = _mm_setzero_ps();
    __m128 vi5x7531 = _mm_setzero_ps();
    __m128 vi6x7531 = _mm_setzero_ps();
    __m128 vi7x7531 = _mm_setzero_ps();
    __m128 vi8x7531 = _mm_setzero_ps();
    __m128 vi9x7531 = _mm_setzero_ps();
    __m128 vi10x7531 = _mm_setzero_ps();
    __m128 vi11x7531 = _mm_setzero_ps();
    __m128 vi12x7531 = _mm_setzero_ps();

    size_t w = input_width;
    for (; w >= 8 * sizeof(float); w -= 8 * sizeof(float)) {
      const __m128 vi0x89AB = _mm_loadu_ps(i0);
      const __m128 vi0xCDEF = _mm_loadu_ps(i0 + 4);
      i0 += 8;
      const __m128 vi1x89AB = _mm_loadu_ps(i1);
      const __m128 vi1xCDEF = _mm_loadu_ps(i1 + 4);
      i1 += 8;
      const __m128 vi2x89AB = _mm_loadu_ps(i2);
      const __m128 vi2xCDEF = _mm_loadu_ps(i2 + 4);
      i2 += 8;
      const __m128 vi3x89AB = _mm_loadu_ps(i3);
      const __m128 vi3xCDEF = _mm_loadu_ps(i3 + 4);
      i3 += 8;
      const __m128 vi4x89AB = _mm_loadu_ps(i4);
      const __m128 vi4xCDEF = _mm_loadu_ps(i4 + 4);
      i4 += 8;
      const __m128 vi5x89AB = _mm_loadu_ps(i5);
      const __m128 vi5xCDEF = _mm_loadu_ps(i5 + 4);
      i5 += 8;
      const __m128 vi6x89AB = _mm_loadu_ps(i6);
      const __m128 vi6xCDEF = _mm_loadu_ps(i6 + 4);
      i6 += 8;
      const __m128 vi7x89AB = _mm_loadu_ps(i7);
      const __m128 vi7xCDEF = _mm_loadu_ps(i7 + 4);
      i7 += 8;
      const __m128 vi8x89AB = _mm_loadu_ps(i8);
      const __m128 vi8xCDEF = _mm_loadu_ps(i8 + 4);
      i8 += 8;
      const __m128 vi9x89AB = _mm_loadu_ps(i9);
      const __m128 vi9xCDEF = _mm_loadu_ps(i9 + 4);
      i9 += 8;
      const __m128 vi10x89AB = _mm_loadu_ps(i10);
      const __m128 vi10xCDEF = _mm_loadu_ps(i10 + 4);
      i10 += 8;
      const __m128 vi11x89AB = _mm_loadu_ps(i11);
      const __m128 vi11xCDEF = _mm_loadu_ps(i11 + 4);
      i11 += 8;
      const __m128 vi12x89AB = _mm_loadu_ps(i12);
      const __m128 vi12xCDEF = _mm_loadu_ps(i12 + 4);
      i12 += 8;

      const __m128 vi0x8ACE = _mm_shuffle_ps(vi0x89AB, vi0xCDEF, _MM_SHUFFLE(2, 0, 2, 0));
      const __m128 vi0x9BDF = _mm_shuffle_ps(vi0x89AB, vi0xCDEF, _MM_SHUFFLE(3, 1, 3, 1));
      const __m128 vi1x8ACE = _mm_shuffle_ps(vi1x89AB, vi1xCDEF, _MM_SHUFFLE(2, 0, 2, 0));
      const __m128 vi1x9BDF = _mm_shuffle_ps(vi1x89AB, vi1xCDEF, _MM_SHUFFLE(3, 1, 3, 1));
      const __m128 vi2x8ACE = _mm_shuffle_ps(vi2x89AB, vi2xCDEF, _MM_SHUFFLE(2, 0, 2, 0));
      const __m128 vi2x9BDF = _mm_shuffle_ps(vi2x89AB, vi2xCDEF, _MM_SHUFFLE(3, 1, 3, 1));
      const __m128 vi3x8ACE = _mm_shuffle_ps(vi3x89AB, vi3xCDEF, _MM_SHUFFLE(2, 0, 2, 0));
      const __m128 vi3x9BDF = _mm_shuffle_ps(vi3x89AB, vi3xCDEF, _MM_SHUFFLE(3, 1, 3, 1));
      const __m128 vi4x8ACE = _mm_shuffle_ps(vi4x89AB, vi4xCDEF, _MM_SHUFFLE(2, 0, 2, 0));
      const __m128 vi4x9BDF = _mm_shuffle_ps(vi4x89AB, vi4xCDEF, _MM_SHUFFLE(3, 1, 3, 1));
      const __m128 vi5x8ACE = _mm_shuffle_ps(vi5x89AB, vi5xCDEF, _MM_SHUFFLE(2, 0, 2, 0));
      const __m128 vi5x9BDF = _mm_shuffle_ps(vi5x89AB, vi5xCDEF, _MM_SHUFFLE(3, 1, 3, 1));
      const __m128 vi6x8ACE = _mm_shuffle_ps(vi6x89AB, vi6xCDEF, _MM_SHUFFLE(2, 0, 2, 0));
      const __m128 vi6x9BDF = _mm_shuffle_ps(vi6x89AB, vi6xCDEF, _MM_SHUFFLE(3, 1, 3, 1));
      const __m128 vi7x8ACE = _mm_shuffle_ps(vi7x89AB, vi7xCDEF, _MM_SHUFFLE(2, 0, 2, 0));
      const __m128 vi7x9BDF = _mm_shuffle_ps(vi7x89AB, vi7xCDEF, _MM_SHUFFLE(3, 1, 3, 1));
      const __m128 vi8x8ACE = _mm_shuffle_ps(vi8x89AB, vi8xCDEF, _MM_SHUFFLE(2, 0, 2, 0));
      const __m128 vi8x9BDF = _mm_shuffle_ps(vi8x89AB, vi8xCDEF, _MM_SHUFFLE(3, 1, 3, 1));
      const __m128 vi9x8ACE = _mm_shuffle_ps(vi9x89AB, vi9xCDEF, _MM_SHUFFLE(2, 0, 2, 0));
      const __m128 vi9x9BDF = _mm_shuffle_ps(vi9x89AB, vi9xCDEF, _MM_SHUFFLE(3, 1, 3, 1));
      const __m128 vi10x8ACE = _mm_shuffle_ps(vi10x89AB, vi10xCDEF, _MM_SHUFFLE(2, 0, 2, 0));
      const __m128 vi10x9BDF = _mm_shuffle_ps(vi10x89AB, vi10xCDEF, _MM_SHUFFLE(3, 1, 3, 1));
      const __m128 vi11x8ACE = _mm_shuffle_ps(vi11x89AB, vi11xCDEF, _MM_SHUFFLE(2, 0, 2, 0));
      const __m128 vi11x9BDF = _mm_shuffle_ps(vi11x89AB, vi11xCDEF, _MM_SHUFFLE(3, 1, 3, 1));
      const __m128 vi12x8ACE = _mm_shuffle_ps(vi12x89AB, vi12xCDEF, _MM_SHUFFLE(2, 0, 2, 0));
      const __m128 vi12x9BDF = _mm_shuffle_ps(vi12x89AB, vi12xCDEF, _MM_SHUFFLE(3, 1, 3, 1));

      __m128 vo0p0 = _mm_add_ps(vbias, _mm_mul_ps(vi0x8ACE, vk01));
      __m128 vo1p0 = _mm_add_ps(vbias, _mm_mul_ps(vi2x8ACE, vk01));
      __m128 vo2p0 = _mm_add_ps(vbias, _mm_mul_ps(vi4x8ACE, vk01));
      __m128 vo3p0 = _mm_add_ps(vbias, _mm_mul_ps(vi6x8ACE, vk01));
      __m128 vo4p0 = _mm_add_ps(vbias, _mm_mul_ps(vi8x8ACE, vk01));
      __m128 vo5p0 = _mm_add_ps(vbias, _mm_mul_ps(vi10x8ACE, vk01));
      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi1x8ACE, vk11));
      vo1p0 = _mm_add_ps(vo1p0, _mm_mul_ps(vi3x8ACE, vk11));
      vo2p0 = _mm_add_ps(vo2p0, _mm_mul_ps(vi5x8ACE, vk11));
      vo3p0 = _mm_add_ps(vo3p0, _mm_mul_ps(vi7x8ACE, vk11));
      vo4p0 = _mm_add_ps(vo4p0, _mm_mul_ps(vi9x8ACE, vk11));
      vo5p0 = _mm_add_ps(vo5p0, _mm_mul_ps(vi11x8ACE, vk11));
      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi2x8ACE, vk21));
      vo1p0 = _mm_add_ps(vo1p0, _mm_mul_ps(vi4x8ACE, vk21));
      vo2p0 = _mm_add_ps(vo2p0, _mm_mul_ps(vi6x8ACE, vk21));
      vo3p0 = _mm_add_ps(vo3p0, _mm_mul_ps(vi8x8ACE, vk21));
      vo4p0 = _mm_add_ps(vo4p0, _mm_mul_ps(vi10x8ACE, vk21));
      vo5p0 = _mm_add_ps(vo5p0, _mm_mul_ps(vi12x8ACE, vk21));

      const __m128 vi0xF9BD = _mm_shuffle_ps(vi0x9BDF, vi0x9BDF, _MM_SHUFFLE(2, 1, 0, 3));
      const __m128 vi1xF9BD = _mm_shuffle_ps(vi1x9BDF, vi1x9BDF, _MM_SHUFFLE(2, 1, 0, 3));
      const __m128 vi2xF9BD = _mm_shuffle_ps(vi2x9BDF, vi2x9BDF, _MM_SHUFFLE(2, 1, 0, 3));
      const __m128 vi3xF9BD = _mm_shuffle_ps(vi3x9BDF, vi3x9BDF, _MM_SHUFFLE(2, 1, 0, 3));
      const __m128 vi4xF9BD = _mm_shuffle_ps(vi4x9BDF, vi4x9BDF, _MM_SHUFFLE(2, 1, 0, 3));
      const __m128 vi5xF9BD = _mm_shuffle_ps(vi5x9BDF, vi5x9BDF, _MM_SHUFFLE(2, 1, 0, 3));
      const __m128 vi6xF9BD = _mm_shuffle_ps(vi6x9BDF, vi6x9BDF, _MM_SHUFFLE(2, 1, 0, 3));
      const __m128 vi7xF9BD = _mm_shuffle_ps(vi7x9BDF, vi7x9BDF, _MM_SHUFFLE(2, 1, 0, 3));
      const __m128 vi8xF9BD = _mm_shuffle_ps(vi8x9BDF, vi8x9BDF, _MM_SHUFFLE(2, 1, 0, 3));
      const __m128 vi9xF9BD = _mm_shuffle_ps(vi9x9BDF, vi9x9BDF, _MM_SHUFFLE(2, 1, 0, 3));
      const __m128 vi10xF9BD = _mm_shuffle_ps(vi10x9BDF, vi10x9BDF, _MM_SHUFFLE(2, 1, 0, 3));
      const __m128 vi11xF9BD = _mm_shuffle_ps(vi11x9BDF, vi11x9BDF, _MM_SHUFFLE(2, 1, 0, 3));
      const __m128 vi12xF9BD = _mm_shuffle_ps(vi12x9BDF, vi12x9BDF, _MM_SHUFFLE(2, 1, 0, 3));

      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi0x9BDF, vk02));
      vo1p0 = _mm_add_ps(vo1p0, _mm_mul_ps(vi2x9BDF, vk02));
      vo2p0 = _mm_add_ps(vo2p0, _mm_mul_ps(vi4x9BDF, vk02));
      vo3p0 = _mm_add_ps(vo3p0, _mm_mul_ps(vi6x9BDF, vk02));
      vo4p0 = _mm_add_ps(vo4p0, _mm_mul_ps(vi8x9BDF, vk02));
      vo5p0 = _mm_add_ps(vo5p0, _mm_mul_ps(vi10x9BDF, vk02));
      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi1x9BDF, vk12));
      vo1p0 = _mm_add_ps(vo1p0, _mm_mul_ps(vi3x9BDF, vk12));
      vo2p0 = _mm_add_ps(vo2p0, _mm_mul_ps(vi5x9BDF, vk12));
      vo3p0 = _mm_add_ps(vo3p0, _mm_mul_ps(vi7x9BDF, vk12));
      vo4p0 = _mm_add_ps(vo4p0, _mm_mul_ps(vi9x9BDF, vk12));
      vo5p0 = _mm_add_ps(vo5p0, _mm_mul_ps(vi11x9BDF, vk12));
      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi2x9BDF, vk22));
      vo1p0 = _mm_add_ps(vo1p0, _mm_mul_ps(vi4x9BDF, vk22));
      vo2p0 = _mm_add_ps(vo2p0, _mm_mul_ps(vi6x9BDF, vk22));
      vo3p0 = _mm_add_ps(vo3p0, _mm_mul_ps(vi8x9BDF, vk22));
      vo4p0 = _mm_add_ps(vo4p0, _mm_mul_ps(vi10x9BDF, vk22));
      vo5p0 = _mm_add_ps(vo5p0, _mm_mul_ps(vi12x9BDF, vk22));

      const __m128 vi0x7BDF = _mm_move_ss(vi0xF9BD, vi0x7531);
      const __m128 vi1x7BDF = _mm_move_ss(vi1xF9BD, vi1x7531);
      const __m128 vi2x7BDF = _mm_move_ss(vi2xF9BD, vi2x7531);
      const __m128 vi3x7BDF = _mm_move_ss(vi3xF9BD, vi3x7531);
      const __m128 vi4x7BDF = _mm_move_ss(vi4xF9BD, vi4x7531);
      const __m128 vi5x7BDF = _mm_move_ss(vi5xF9BD, vi5x7531);
      const __m128 vi6x7BDF = _mm_move_ss(vi6xF9BD, vi6x7531);
      const __m128 vi7x7BDF = _mm_move_ss(vi7xF9BD, vi7x7531);
      const __m128 vi8x7BDF = _mm_move_ss(vi8xF9BD, vi8x7531);
      const __m128 vi9x7BDF = _mm_move_ss(vi9xF9BD, vi9x7531);
      const __m128 vi10x7BDF = _mm_move_ss(vi10xF9BD, vi10x7531);
      const __m128 vi11x7BDF = _mm_move_ss(vi11xF9BD, vi11x7531);
      const __m128 vi12x7BDF = _mm_move_ss(vi12xF9BD, vi12x7531);

      vi0x7531 = vi0xF9BD;
      vi1x7531 = vi1xF9BD;
      vi2x7531 = vi2xF9BD;
      vi3x7531 = vi3xF9BD;
      vi4x7531 = vi4xF9BD;
      vi5x7531 = vi5xF9BD;
      vi6x7531 = vi6xF9BD;
      vi7x7531 = vi7xF9BD;
      vi8x7531 = vi8xF9BD;
      vi9x7531 = vi9xF9BD;
      vi10x7531 = vi10xF9BD;
      vi11x7531 = vi11xF9BD;
      vi12x7531 = vi12xF9BD;

      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi0x7BDF, vk00));
      vo1p0 = _mm_add_ps(vo1p0, _mm_mul_ps(vi2x7BDF, vk00));
      vo2p0 = _mm_add_ps(vo2p0, _mm_mul_ps(vi4x7BDF, vk00));
      vo3p0 = _mm_add_ps(vo3p0, _mm_mul_ps(vi6x7BDF, vk00));
      vo4p0 = _mm_add_ps(vo4p0, _mm_mul_ps(vi8x7BDF, vk00));
      vo5p0 = _mm_add_ps(vo5p0, _mm_mul_ps(vi10x7BDF, vk00));
      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi1x7BDF, vk10));
      vo1p0 = _mm_add_ps(vo1p0, _mm_mul_ps(vi3x7BDF, vk10));
      vo2p0 = _mm_add_ps(vo2p0, _mm_mul_ps(vi5x7BDF, vk10));
      vo3p0 = _mm_add_ps(vo3p0, _mm_mul_ps(vi7x7BDF, vk10));
      vo4p0 = _mm_add_ps(vo4p0, _mm_mul_ps(vi9x7BDF, vk10));
      vo5p0 = _mm_add_ps(vo5p0, _mm_mul_ps(vi11x7BDF, vk10));
      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi2x7BDF, vk20));
      vo1p0 = _mm_add_ps(vo1p0, _mm_mul_ps(vi4x7BDF, vk20));
      vo2p0 = _mm_add_ps(vo2p0, _mm_mul_ps(vi6x7BDF, vk20));
      vo3p0 = _mm_add_ps(vo3p0, _mm_mul_ps(vi8x7BDF, vk20));
      vo4p0 = _mm_add_ps(vo4p0, _mm_mul_ps(vi10x7BDF, vk20));
      vo5p0 = _mm_add_ps(vo5p0, _mm_mul_ps(vi12x7BDF, vk20));


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
    // Potentially process the last block of 0..7 pixels.
    assert(w < 8 * sizeof(float));
    if XNN_LIKELY(w != 0) {
      const __m128 vi0x89AB = _mm_loadu_ps(i0);
      const __m128 vi0xCDEF = _mm_loadu_ps(i0 + 4);
      const __m128 vi1x89AB = _mm_loadu_ps(i1);
      const __m128 vi1xCDEF = _mm_loadu_ps(i1 + 4);
      const __m128 vi2x89AB = _mm_loadu_ps(i2);
      const __m128 vi2xCDEF = _mm_loadu_ps(i2 + 4);
      const __m128 vi3x89AB = _mm_loadu_ps(i3);
      const __m128 vi3xCDEF = _mm_loadu_ps(i3 + 4);
      const __m128 vi4x89AB = _mm_loadu_ps(i4);
      const __m128 vi4xCDEF = _mm_loadu_ps(i4 + 4);
      const __m128 vi5x89AB = _mm_loadu_ps(i5);
      const __m128 vi5xCDEF = _mm_loadu_ps(i5 + 4);
      const __m128 vi6x89AB = _mm_loadu_ps(i6);
      const __m128 vi6xCDEF = _mm_loadu_ps(i6 + 4);
      const __m128 vi7x89AB = _mm_loadu_ps(i7);
      const __m128 vi7xCDEF = _mm_loadu_ps(i7 + 4);
      const __m128 vi8x89AB = _mm_loadu_ps(i8);
      const __m128 vi8xCDEF = _mm_loadu_ps(i8 + 4);
      const __m128 vi9x89AB = _mm_loadu_ps(i9);
      const __m128 vi9xCDEF = _mm_loadu_ps(i9 + 4);
      const __m128 vi10x89AB = _mm_loadu_ps(i10);
      const __m128 vi10xCDEF = _mm_loadu_ps(i10 + 4);
      const __m128 vi11x89AB = _mm_loadu_ps(i11);
      const __m128 vi11xCDEF = _mm_loadu_ps(i11 + 4);
      const __m128 vi12x89AB = _mm_loadu_ps(i12);
      const __m128 vi12xCDEF = _mm_loadu_ps(i12 + 4);

      const __m128 vi0x8ACE = _mm_and_ps(vmask_even, _mm_shuffle_ps(vi0x89AB, vi0xCDEF, _MM_SHUFFLE(2, 0, 2, 0)));
      const __m128 vi0x9BDF = _mm_and_ps(vmask_odd,  _mm_shuffle_ps(vi0x89AB, vi0xCDEF, _MM_SHUFFLE(3, 1, 3, 1)));
      const __m128 vi1x8ACE = _mm_and_ps(vmask_even, _mm_shuffle_ps(vi1x89AB, vi1xCDEF, _MM_SHUFFLE(2, 0, 2, 0)));
      const __m128 vi1x9BDF = _mm_and_ps(vmask_odd,  _mm_shuffle_ps(vi1x89AB, vi1xCDEF, _MM_SHUFFLE(3, 1, 3, 1)));
      const __m128 vi2x8ACE = _mm_and_ps(vmask_even, _mm_shuffle_ps(vi2x89AB, vi2xCDEF, _MM_SHUFFLE(2, 0, 2, 0)));
      const __m128 vi2x9BDF = _mm_and_ps(vmask_odd,  _mm_shuffle_ps(vi2x89AB, vi2xCDEF, _MM_SHUFFLE(3, 1, 3, 1)));
      const __m128 vi3x8ACE = _mm_and_ps(vmask_even, _mm_shuffle_ps(vi3x89AB, vi3xCDEF, _MM_SHUFFLE(2, 0, 2, 0)));
      const __m128 vi3x9BDF = _mm_and_ps(vmask_odd,  _mm_shuffle_ps(vi3x89AB, vi3xCDEF, _MM_SHUFFLE(3, 1, 3, 1)));
      const __m128 vi4x8ACE = _mm_and_ps(vmask_even, _mm_shuffle_ps(vi4x89AB, vi4xCDEF, _MM_SHUFFLE(2, 0, 2, 0)));
      const __m128 vi4x9BDF = _mm_and_ps(vmask_odd,  _mm_shuffle_ps(vi4x89AB, vi4xCDEF, _MM_SHUFFLE(3, 1, 3, 1)));
      const __m128 vi5x8ACE = _mm_and_ps(vmask_even, _mm_shuffle_ps(vi5x89AB, vi5xCDEF, _MM_SHUFFLE(2, 0, 2, 0)));
      const __m128 vi5x9BDF = _mm_and_ps(vmask_odd,  _mm_shuffle_ps(vi5x89AB, vi5xCDEF, _MM_SHUFFLE(3, 1, 3, 1)));
      const __m128 vi6x8ACE = _mm_and_ps(vmask_even, _mm_shuffle_ps(vi6x89AB, vi6xCDEF, _MM_SHUFFLE(2, 0, 2, 0)));
      const __m128 vi6x9BDF = _mm_and_ps(vmask_odd,  _mm_shuffle_ps(vi6x89AB, vi6xCDEF, _MM_SHUFFLE(3, 1, 3, 1)));
      const __m128 vi7x8ACE = _mm_and_ps(vmask_even, _mm_shuffle_ps(vi7x89AB, vi7xCDEF, _MM_SHUFFLE(2, 0, 2, 0)));
      const __m128 vi7x9BDF = _mm_and_ps(vmask_odd,  _mm_shuffle_ps(vi7x89AB, vi7xCDEF, _MM_SHUFFLE(3, 1, 3, 1)));
      const __m128 vi8x8ACE = _mm_and_ps(vmask_even, _mm_shuffle_ps(vi8x89AB, vi8xCDEF, _MM_SHUFFLE(2, 0, 2, 0)));
      const __m128 vi8x9BDF = _mm_and_ps(vmask_odd,  _mm_shuffle_ps(vi8x89AB, vi8xCDEF, _MM_SHUFFLE(3, 1, 3, 1)));
      const __m128 vi9x8ACE = _mm_and_ps(vmask_even, _mm_shuffle_ps(vi9x89AB, vi9xCDEF, _MM_SHUFFLE(2, 0, 2, 0)));
      const __m128 vi9x9BDF = _mm_and_ps(vmask_odd,  _mm_shuffle_ps(vi9x89AB, vi9xCDEF, _MM_SHUFFLE(3, 1, 3, 1)));
      const __m128 vi10x8ACE = _mm_and_ps(vmask_even, _mm_shuffle_ps(vi10x89AB, vi10xCDEF, _MM_SHUFFLE(2, 0, 2, 0)));
      const __m128 vi10x9BDF = _mm_and_ps(vmask_odd,  _mm_shuffle_ps(vi10x89AB, vi10xCDEF, _MM_SHUFFLE(3, 1, 3, 1)));
      const __m128 vi11x8ACE = _mm_and_ps(vmask_even, _mm_shuffle_ps(vi11x89AB, vi11xCDEF, _MM_SHUFFLE(2, 0, 2, 0)));
      const __m128 vi11x9BDF = _mm_and_ps(vmask_odd,  _mm_shuffle_ps(vi11x89AB, vi11xCDEF, _MM_SHUFFLE(3, 1, 3, 1)));
      const __m128 vi12x8ACE = _mm_and_ps(vmask_even, _mm_shuffle_ps(vi12x89AB, vi12xCDEF, _MM_SHUFFLE(2, 0, 2, 0)));
      const __m128 vi12x9BDF = _mm_and_ps(vmask_odd,  _mm_shuffle_ps(vi12x89AB, vi12xCDEF, _MM_SHUFFLE(3, 1, 3, 1)));

      __m128 vo0p0 = _mm_add_ps(vbias, _mm_mul_ps(vi0x8ACE, vk01));
      __m128 vo1p0 = _mm_add_ps(vbias, _mm_mul_ps(vi2x8ACE, vk01));
      __m128 vo2p0 = _mm_add_ps(vbias, _mm_mul_ps(vi4x8ACE, vk01));
      __m128 vo3p0 = _mm_add_ps(vbias, _mm_mul_ps(vi6x8ACE, vk01));
      __m128 vo4p0 = _mm_add_ps(vbias, _mm_mul_ps(vi8x8ACE, vk01));
      __m128 vo5p0 = _mm_add_ps(vbias, _mm_mul_ps(vi10x8ACE, vk01));
      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi1x8ACE, vk11));
      vo1p0 = _mm_add_ps(vo1p0, _mm_mul_ps(vi3x8ACE, vk11));
      vo2p0 = _mm_add_ps(vo2p0, _mm_mul_ps(vi5x8ACE, vk11));
      vo3p0 = _mm_add_ps(vo3p0, _mm_mul_ps(vi7x8ACE, vk11));
      vo4p0 = _mm_add_ps(vo4p0, _mm_mul_ps(vi9x8ACE, vk11));
      vo5p0 = _mm_add_ps(vo5p0, _mm_mul_ps(vi11x8ACE, vk11));
      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi2x8ACE, vk21));
      vo1p0 = _mm_add_ps(vo1p0, _mm_mul_ps(vi4x8ACE, vk21));
      vo2p0 = _mm_add_ps(vo2p0, _mm_mul_ps(vi6x8ACE, vk21));
      vo3p0 = _mm_add_ps(vo3p0, _mm_mul_ps(vi8x8ACE, vk21));
      vo4p0 = _mm_add_ps(vo4p0, _mm_mul_ps(vi10x8ACE, vk21));
      vo5p0 = _mm_add_ps(vo5p0, _mm_mul_ps(vi12x8ACE, vk21));

      const __m128 vi0xF9BD = _mm_shuffle_ps(vi0x9BDF, vi0x9BDF, _MM_SHUFFLE(2, 1, 0, 3));
      const __m128 vi1xF9BD = _mm_shuffle_ps(vi1x9BDF, vi1x9BDF, _MM_SHUFFLE(2, 1, 0, 3));
      const __m128 vi2xF9BD = _mm_shuffle_ps(vi2x9BDF, vi2x9BDF, _MM_SHUFFLE(2, 1, 0, 3));
      const __m128 vi3xF9BD = _mm_shuffle_ps(vi3x9BDF, vi3x9BDF, _MM_SHUFFLE(2, 1, 0, 3));
      const __m128 vi4xF9BD = _mm_shuffle_ps(vi4x9BDF, vi4x9BDF, _MM_SHUFFLE(2, 1, 0, 3));
      const __m128 vi5xF9BD = _mm_shuffle_ps(vi5x9BDF, vi5x9BDF, _MM_SHUFFLE(2, 1, 0, 3));
      const __m128 vi6xF9BD = _mm_shuffle_ps(vi6x9BDF, vi6x9BDF, _MM_SHUFFLE(2, 1, 0, 3));
      const __m128 vi7xF9BD = _mm_shuffle_ps(vi7x9BDF, vi7x9BDF, _MM_SHUFFLE(2, 1, 0, 3));
      const __m128 vi8xF9BD = _mm_shuffle_ps(vi8x9BDF, vi8x9BDF, _MM_SHUFFLE(2, 1, 0, 3));
      const __m128 vi9xF9BD = _mm_shuffle_ps(vi9x9BDF, vi9x9BDF, _MM_SHUFFLE(2, 1, 0, 3));
      const __m128 vi10xF9BD = _mm_shuffle_ps(vi10x9BDF, vi10x9BDF, _MM_SHUFFLE(2, 1, 0, 3));
      const __m128 vi11xF9BD = _mm_shuffle_ps(vi11x9BDF, vi11x9BDF, _MM_SHUFFLE(2, 1, 0, 3));
      const __m128 vi12xF9BD = _mm_shuffle_ps(vi12x9BDF, vi12x9BDF, _MM_SHUFFLE(2, 1, 0, 3));

      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi0x9BDF, vk02));
      vo1p0 = _mm_add_ps(vo1p0, _mm_mul_ps(vi2x9BDF, vk02));
      vo2p0 = _mm_add_ps(vo2p0, _mm_mul_ps(vi4x9BDF, vk02));
      vo3p0 = _mm_add_ps(vo3p0, _mm_mul_ps(vi6x9BDF, vk02));
      vo4p0 = _mm_add_ps(vo4p0, _mm_mul_ps(vi8x9BDF, vk02));
      vo5p0 = _mm_add_ps(vo5p0, _mm_mul_ps(vi10x9BDF, vk02));
      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi1x9BDF, vk12));
      vo1p0 = _mm_add_ps(vo1p0, _mm_mul_ps(vi3x9BDF, vk12));
      vo2p0 = _mm_add_ps(vo2p0, _mm_mul_ps(vi5x9BDF, vk12));
      vo3p0 = _mm_add_ps(vo3p0, _mm_mul_ps(vi7x9BDF, vk12));
      vo4p0 = _mm_add_ps(vo4p0, _mm_mul_ps(vi9x9BDF, vk12));
      vo5p0 = _mm_add_ps(vo5p0, _mm_mul_ps(vi11x9BDF, vk12));
      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi2x9BDF, vk22));
      vo1p0 = _mm_add_ps(vo1p0, _mm_mul_ps(vi4x9BDF, vk22));
      vo2p0 = _mm_add_ps(vo2p0, _mm_mul_ps(vi6x9BDF, vk22));
      vo3p0 = _mm_add_ps(vo3p0, _mm_mul_ps(vi8x9BDF, vk22));
      vo4p0 = _mm_add_ps(vo4p0, _mm_mul_ps(vi10x9BDF, vk22));
      vo5p0 = _mm_add_ps(vo5p0, _mm_mul_ps(vi12x9BDF, vk22));

      const __m128 vi0x7BDF = _mm_move_ss(vi0xF9BD, vi0x7531);
      const __m128 vi1x7BDF = _mm_move_ss(vi1xF9BD, vi1x7531);
      const __m128 vi2x7BDF = _mm_move_ss(vi2xF9BD, vi2x7531);
      const __m128 vi3x7BDF = _mm_move_ss(vi3xF9BD, vi3x7531);
      const __m128 vi4x7BDF = _mm_move_ss(vi4xF9BD, vi4x7531);
      const __m128 vi5x7BDF = _mm_move_ss(vi5xF9BD, vi5x7531);
      const __m128 vi6x7BDF = _mm_move_ss(vi6xF9BD, vi6x7531);
      const __m128 vi7x7BDF = _mm_move_ss(vi7xF9BD, vi7x7531);
      const __m128 vi8x7BDF = _mm_move_ss(vi8xF9BD, vi8x7531);
      const __m128 vi9x7BDF = _mm_move_ss(vi9xF9BD, vi9x7531);
      const __m128 vi10x7BDF = _mm_move_ss(vi10xF9BD, vi10x7531);
      const __m128 vi11x7BDF = _mm_move_ss(vi11xF9BD, vi11x7531);
      const __m128 vi12x7BDF = _mm_move_ss(vi12xF9BD, vi12x7531);

      vi0x7531 = vi0xF9BD;
      vi1x7531 = vi1xF9BD;
      vi2x7531 = vi2xF9BD;
      vi3x7531 = vi3xF9BD;
      vi4x7531 = vi4xF9BD;
      vi5x7531 = vi5xF9BD;
      vi6x7531 = vi6xF9BD;
      vi7x7531 = vi7xF9BD;
      vi8x7531 = vi8xF9BD;
      vi9x7531 = vi9xF9BD;
      vi10x7531 = vi10xF9BD;
      vi11x7531 = vi11xF9BD;
      vi12x7531 = vi12xF9BD;

      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi0x7BDF, vk00));
      vo1p0 = _mm_add_ps(vo1p0, _mm_mul_ps(vi2x7BDF, vk00));
      vo2p0 = _mm_add_ps(vo2p0, _mm_mul_ps(vi4x7BDF, vk00));
      vo3p0 = _mm_add_ps(vo3p0, _mm_mul_ps(vi6x7BDF, vk00));
      vo4p0 = _mm_add_ps(vo4p0, _mm_mul_ps(vi8x7BDF, vk00));
      vo5p0 = _mm_add_ps(vo5p0, _mm_mul_ps(vi10x7BDF, vk00));
      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi1x7BDF, vk10));
      vo1p0 = _mm_add_ps(vo1p0, _mm_mul_ps(vi3x7BDF, vk10));
      vo2p0 = _mm_add_ps(vo2p0, _mm_mul_ps(vi5x7BDF, vk10));
      vo3p0 = _mm_add_ps(vo3p0, _mm_mul_ps(vi7x7BDF, vk10));
      vo4p0 = _mm_add_ps(vo4p0, _mm_mul_ps(vi9x7BDF, vk10));
      vo5p0 = _mm_add_ps(vo5p0, _mm_mul_ps(vi11x7BDF, vk10));
      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi2x7BDF, vk20));
      vo1p0 = _mm_add_ps(vo1p0, _mm_mul_ps(vi4x7BDF, vk20));
      vo2p0 = _mm_add_ps(vo2p0, _mm_mul_ps(vi6x7BDF, vk20));
      vo3p0 = _mm_add_ps(vo3p0, _mm_mul_ps(vi8x7BDF, vk20));
      vo4p0 = _mm_add_ps(vo4p0, _mm_mul_ps(vi10x7BDF, vk20));
      vo5p0 = _mm_add_ps(vo5p0, _mm_mul_ps(vi12x7BDF, vk20));


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

      if (w == 7 * sizeof(float)) {
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
        w += 1 * sizeof(float);
        if (w & (4 * sizeof(float))) {
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
        if (w & (2 * sizeof(float))) {
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

    i0 = (const float*) ((uintptr_t) i12 - input_decrement);
    i1 = (const float*) ((uintptr_t) i0 + input_width);
    i2 = (const float*) ((uintptr_t) i1 + input_width);
    i3 = (const float*) ((uintptr_t) i2 + input_width);
    i4 = (const float*) ((uintptr_t) i3 + input_width);
    i5 = (const float*) ((uintptr_t) i4 + input_width);
    i6 = (const float*) ((uintptr_t) i5 + input_width);
    i7 = (const float*) ((uintptr_t) i6 + input_width);
    i8 = (const float*) ((uintptr_t) i7 + input_width);
    i9 = (const float*) ((uintptr_t) i8 + input_width);
    i10 = (const float*) ((uintptr_t) i9 + input_width);
    i11 = (const float*) ((uintptr_t) i10 + input_width);
    i12 = (const float*) ((uintptr_t) i11 + input_width);

    o0 = o5;
    o1 = (float*) ((uintptr_t) o0 + output_width);
    o2 = (float*) ((uintptr_t) o1 + output_width);
    o3 = (float*) ((uintptr_t) o2 + output_width);
    o4 = (float*) ((uintptr_t) o3 + output_width);
    o5 = (float*) ((uintptr_t) o4 + output_width);

    output_height = doz(output_height, 6);
    padded_input_height = doz(padded_input_height, 12);
  } while (output_height != 0);
}
