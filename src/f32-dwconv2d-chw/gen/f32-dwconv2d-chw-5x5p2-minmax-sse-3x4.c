// Auto-generated file. Do not edit!
//   Template: src/f32-dwconv2d-chw/5x5p2-sse.c.in
//   Generator: tools/xngen
//
// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <xmmintrin.h>

#include "xnnpack/dwconv.h"
#include "xnnpack/math.h"


void xnn_f32_dwconv2d_chw_ukernel_5x5p2__sse_3x4(
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
  assert(padding_top == 2);

  const __m128 vmask = _mm_load_ps((const float*) params->sse_stride1.mask);
  const __m128 vmax = _mm_set1_ps(params->sse_stride1.max);
  const __m128 vmin = _mm_set1_ps(params->sse_stride1.min);
  XNN_FORCE_REALIZATION(vmin);
  XNN_FORCE_REALIZATION(vmax);

  const __m128 vbias = _mm_load1_ps(weights);
  const __m128 vk00 = _mm_load1_ps(weights + 1);
  const __m128 vk01 = _mm_load1_ps(weights + 2);
  const __m128 vk02 = _mm_load1_ps(weights + 3);
  const __m128 vk03 = _mm_load1_ps(weights + 4);
  const __m128 vk04 = _mm_load1_ps(weights + 5);
  const __m128 vk10 = _mm_load1_ps(weights + 6);
  const __m128 vk11 = _mm_load1_ps(weights + 7);
  const __m128 vk12 = _mm_load1_ps(weights + 8);
  const __m128 vk13 = _mm_load1_ps(weights + 9);
  const __m128 vk14 = _mm_load1_ps(weights + 10);
  const __m128 vk20 = _mm_load1_ps(weights + 11);
  const __m128 vk21 = _mm_load1_ps(weights + 12);
  const __m128 vk22 = _mm_load1_ps(weights + 13);
  const __m128 vk23 = _mm_load1_ps(weights + 14);
  const __m128 vk24 = _mm_load1_ps(weights + 15);
  const __m128 vk30 = _mm_load1_ps(weights + 16);
  const __m128 vk31 = _mm_load1_ps(weights + 17);
  const __m128 vk32 = _mm_load1_ps(weights + 18);
  const __m128 vk33 = _mm_load1_ps(weights + 19);
  const __m128 vk34 = _mm_load1_ps(weights + 20);
  const __m128 vk40 = _mm_load1_ps(weights + 21);
  const __m128 vk41 = _mm_load1_ps(weights + 22);
  const __m128 vk42 = _mm_load1_ps(weights + 23);
  const __m128 vk43 = _mm_load1_ps(weights + 24);
  const __m128 vk44 = _mm_load1_ps(weights + 25);

  const size_t input_decrement = round_up_po2(input_width, 4 * sizeof(float));

  const float* i0 = zero;
  const float* i1 = zero;
  const float* i2 = input;
  const float* i3 = (const float*) ((uintptr_t) i2 + input_width);
  const float* i4 = (const float*) ((uintptr_t) i3 + input_width);
  const float* i5 = (const float*) ((uintptr_t) i4 + input_width);
  const float* i6 = (const float*) ((uintptr_t) i5 + input_width);

  float* o0 = output;
  float* o1 = (float*) ((uintptr_t) o0 + input_width);
  float* o2 = (float*) ((uintptr_t) o1 + input_width);

  size_t output_height = input_height;
  do {
    if XNN_UNPREDICTABLE(output_height < 2) {
      i3 = zero;
      o1 = o0;
    }
    if XNN_UNPREDICTABLE(output_height < 3) {
      i4 = zero;
      o2 = o1;
    }
    if XNN_UNPREDICTABLE(output_height < 4) {
      i5 = zero;
    }
    if XNN_UNPREDICTABLE(output_height < 5) {
      i6 = zero;
    }

    __m128 vi0x3012 = _mm_setzero_ps();
    __m128 vi1x3012 = _mm_setzero_ps();
    __m128 vi2x3012 = _mm_setzero_ps();
    __m128 vi3x3012 = _mm_setzero_ps();
    __m128 vi4x3012 = _mm_setzero_ps();
    __m128 vi5x3012 = _mm_setzero_ps();
    __m128 vi6x3012 = _mm_setzero_ps();

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

    size_t w = input_width;
    for (; w > 8 * sizeof(float); w -= 4 * sizeof(float)) {
      __m128 vo0p0 = _mm_add_ps(vbias, _mm_mul_ps(vi0x4567, vk02));
      __m128 vo1p0 = _mm_add_ps(vbias, _mm_mul_ps(vi1x4567, vk02));
      __m128 vo2p0 = _mm_add_ps(vbias, _mm_mul_ps(vi2x4567, vk02));
      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi1x4567, vk12));
      vo1p0 = _mm_add_ps(vo1p0, _mm_mul_ps(vi2x4567, vk12));
      vo2p0 = _mm_add_ps(vo2p0, _mm_mul_ps(vi3x4567, vk12));
      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi2x4567, vk22));
      vo1p0 = _mm_add_ps(vo1p0, _mm_mul_ps(vi3x4567, vk22));
      vo2p0 = _mm_add_ps(vo2p0, _mm_mul_ps(vi4x4567, vk22));
      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi3x4567, vk32));
      vo1p0 = _mm_add_ps(vo1p0, _mm_mul_ps(vi4x4567, vk32));
      vo2p0 = _mm_add_ps(vo2p0, _mm_mul_ps(vi5x4567, vk32));
      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi4x4567, vk42));
      vo1p0 = _mm_add_ps(vo1p0, _mm_mul_ps(vi5x4567, vk42));
      vo2p0 = _mm_add_ps(vo2p0, _mm_mul_ps(vi6x4567, vk42));

      const __m128 vi0x7456 = _mm_shuffle_ps(vi0x4567, vi0x4567, _MM_SHUFFLE(2, 1, 0, 3));
      const __m128 vi1x7456 = _mm_shuffle_ps(vi1x4567, vi1x4567, _MM_SHUFFLE(2, 1, 0, 3));
      const __m128 vi2x7456 = _mm_shuffle_ps(vi2x4567, vi2x4567, _MM_SHUFFLE(2, 1, 0, 3));
      const __m128 vi3x7456 = _mm_shuffle_ps(vi3x4567, vi3x4567, _MM_SHUFFLE(2, 1, 0, 3));
      const __m128 vi4x7456 = _mm_shuffle_ps(vi4x4567, vi4x4567, _MM_SHUFFLE(2, 1, 0, 3));
      const __m128 vi5x7456 = _mm_shuffle_ps(vi5x4567, vi5x4567, _MM_SHUFFLE(2, 1, 0, 3));
      const __m128 vi6x7456 = _mm_shuffle_ps(vi6x4567, vi6x4567, _MM_SHUFFLE(2, 1, 0, 3));

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

      const __m128 vi0x3456 = _mm_move_ss(vi0x7456, vi0x3012);
      const __m128 vi1x3456 = _mm_move_ss(vi1x7456, vi1x3012);
      const __m128 vi2x3456 = _mm_move_ss(vi2x7456, vi2x3012);
      const __m128 vi3x3456 = _mm_move_ss(vi3x7456, vi3x3012);
      const __m128 vi4x3456 = _mm_move_ss(vi4x7456, vi4x3012);
      const __m128 vi5x3456 = _mm_move_ss(vi5x7456, vi5x3012);
      const __m128 vi6x3456 = _mm_move_ss(vi6x7456, vi6x3012);

      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi0x3456, vk01));
      vo1p0 = _mm_add_ps(vo1p0, _mm_mul_ps(vi1x3456, vk01));
      vo2p0 = _mm_add_ps(vo2p0, _mm_mul_ps(vi2x3456, vk01));
      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi1x3456, vk11));
      vo1p0 = _mm_add_ps(vo1p0, _mm_mul_ps(vi2x3456, vk11));
      vo2p0 = _mm_add_ps(vo2p0, _mm_mul_ps(vi3x3456, vk11));
      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi2x3456, vk21));
      vo1p0 = _mm_add_ps(vo1p0, _mm_mul_ps(vi3x3456, vk21));
      vo2p0 = _mm_add_ps(vo2p0, _mm_mul_ps(vi4x3456, vk21));
      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi3x3456, vk31));
      vo1p0 = _mm_add_ps(vo1p0, _mm_mul_ps(vi4x3456, vk31));
      vo2p0 = _mm_add_ps(vo2p0, _mm_mul_ps(vi5x3456, vk31));
      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi4x3456, vk41));
      vo1p0 = _mm_add_ps(vo1p0, _mm_mul_ps(vi5x3456, vk41));
      vo2p0 = _mm_add_ps(vo2p0, _mm_mul_ps(vi6x3456, vk41));

      const __m128 vi0x2345 = _mm_shuffle_ps(vi0x3012, vi0x7456, _MM_SHUFFLE(2, 1, 0, 3));
      vi0x3012 = vi0x7456;
      const __m128 vi1x2345 = _mm_shuffle_ps(vi1x3012, vi1x7456, _MM_SHUFFLE(2, 1, 0, 3));
      vi1x3012 = vi1x7456;
      const __m128 vi2x2345 = _mm_shuffle_ps(vi2x3012, vi2x7456, _MM_SHUFFLE(2, 1, 0, 3));
      vi2x3012 = vi2x7456;
      const __m128 vi3x2345 = _mm_shuffle_ps(vi3x3012, vi3x7456, _MM_SHUFFLE(2, 1, 0, 3));
      vi3x3012 = vi3x7456;
      const __m128 vi4x2345 = _mm_shuffle_ps(vi4x3012, vi4x7456, _MM_SHUFFLE(2, 1, 0, 3));
      vi4x3012 = vi4x7456;
      const __m128 vi5x2345 = _mm_shuffle_ps(vi5x3012, vi5x7456, _MM_SHUFFLE(2, 1, 0, 3));
      vi5x3012 = vi5x7456;
      const __m128 vi6x2345 = _mm_shuffle_ps(vi6x3012, vi6x7456, _MM_SHUFFLE(2, 1, 0, 3));
      vi6x3012 = vi6x7456;

      const __m128 vi0x8567 = _mm_move_ss(vi0x4567, vi0x89AB);
      vi0x4567 = vi0x89AB;
      const __m128 vi1x8567 = _mm_move_ss(vi1x4567, vi1x89AB);
      vi1x4567 = vi1x89AB;
      const __m128 vi2x8567 = _mm_move_ss(vi2x4567, vi2x89AB);
      vi2x4567 = vi2x89AB;
      const __m128 vi3x8567 = _mm_move_ss(vi3x4567, vi3x89AB);
      vi3x4567 = vi3x89AB;
      const __m128 vi4x8567 = _mm_move_ss(vi4x4567, vi4x89AB);
      vi4x4567 = vi4x89AB;
      const __m128 vi5x8567 = _mm_move_ss(vi5x4567, vi5x89AB);
      vi5x4567 = vi5x89AB;
      const __m128 vi6x8567 = _mm_move_ss(vi6x4567, vi6x89AB);
      vi6x4567 = vi6x89AB;

      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi0x2345, vk00));
      vo1p0 = _mm_add_ps(vo1p0, _mm_mul_ps(vi1x2345, vk00));
      vo2p0 = _mm_add_ps(vo2p0, _mm_mul_ps(vi2x2345, vk00));
      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi1x2345, vk10));
      vo1p0 = _mm_add_ps(vo1p0, _mm_mul_ps(vi2x2345, vk10));
      vo2p0 = _mm_add_ps(vo2p0, _mm_mul_ps(vi3x2345, vk10));
      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi2x2345, vk20));
      vo1p0 = _mm_add_ps(vo1p0, _mm_mul_ps(vi3x2345, vk20));
      vo2p0 = _mm_add_ps(vo2p0, _mm_mul_ps(vi4x2345, vk20));
      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi3x2345, vk30));
      vo1p0 = _mm_add_ps(vo1p0, _mm_mul_ps(vi4x2345, vk30));
      vo2p0 = _mm_add_ps(vo2p0, _mm_mul_ps(vi5x2345, vk30));
      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi4x2345, vk40));
      vo1p0 = _mm_add_ps(vo1p0, _mm_mul_ps(vi5x2345, vk40));
      vo2p0 = _mm_add_ps(vo2p0, _mm_mul_ps(vi6x2345, vk40));

      const __m128 vi0x5678 = _mm_shuffle_ps(vi0x8567, vi0x8567, _MM_SHUFFLE(0, 3, 2, 1));
      const __m128 vi1x5678 = _mm_shuffle_ps(vi1x8567, vi1x8567, _MM_SHUFFLE(0, 3, 2, 1));
      const __m128 vi2x5678 = _mm_shuffle_ps(vi2x8567, vi2x8567, _MM_SHUFFLE(0, 3, 2, 1));
      const __m128 vi3x5678 = _mm_shuffle_ps(vi3x8567, vi3x8567, _MM_SHUFFLE(0, 3, 2, 1));
      const __m128 vi4x5678 = _mm_shuffle_ps(vi4x8567, vi4x8567, _MM_SHUFFLE(0, 3, 2, 1));
      const __m128 vi5x5678 = _mm_shuffle_ps(vi5x8567, vi5x8567, _MM_SHUFFLE(0, 3, 2, 1));
      const __m128 vi6x5678 = _mm_shuffle_ps(vi6x8567, vi6x8567, _MM_SHUFFLE(0, 3, 2, 1));

      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi0x5678, vk03));
      vo1p0 = _mm_add_ps(vo1p0, _mm_mul_ps(vi1x5678, vk03));
      vo2p0 = _mm_add_ps(vo2p0, _mm_mul_ps(vi2x5678, vk03));
      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi1x5678, vk13));
      vo1p0 = _mm_add_ps(vo1p0, _mm_mul_ps(vi2x5678, vk13));
      vo2p0 = _mm_add_ps(vo2p0, _mm_mul_ps(vi3x5678, vk13));
      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi2x5678, vk23));
      vo1p0 = _mm_add_ps(vo1p0, _mm_mul_ps(vi3x5678, vk23));
      vo2p0 = _mm_add_ps(vo2p0, _mm_mul_ps(vi4x5678, vk23));
      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi3x5678, vk33));
      vo1p0 = _mm_add_ps(vo1p0, _mm_mul_ps(vi4x5678, vk33));
      vo2p0 = _mm_add_ps(vo2p0, _mm_mul_ps(vi5x5678, vk33));
      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi4x5678, vk43));
      vo1p0 = _mm_add_ps(vo1p0, _mm_mul_ps(vi5x5678, vk43));
      vo2p0 = _mm_add_ps(vo2p0, _mm_mul_ps(vi6x5678, vk43));

      const __m128 vi0x6789 = _mm_shuffle_ps(vi0x5678, vi0x89AB, _MM_SHUFFLE(1, 0, 2, 1));
      const __m128 vi1x6789 = _mm_shuffle_ps(vi1x5678, vi1x89AB, _MM_SHUFFLE(1, 0, 2, 1));
      const __m128 vi2x6789 = _mm_shuffle_ps(vi2x5678, vi2x89AB, _MM_SHUFFLE(1, 0, 2, 1));
      const __m128 vi3x6789 = _mm_shuffle_ps(vi3x5678, vi3x89AB, _MM_SHUFFLE(1, 0, 2, 1));
      const __m128 vi4x6789 = _mm_shuffle_ps(vi4x5678, vi4x89AB, _MM_SHUFFLE(1, 0, 2, 1));
      const __m128 vi5x6789 = _mm_shuffle_ps(vi5x5678, vi5x89AB, _MM_SHUFFLE(1, 0, 2, 1));
      const __m128 vi6x6789 = _mm_shuffle_ps(vi6x5678, vi6x89AB, _MM_SHUFFLE(1, 0, 2, 1));

      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi0x6789, vk04));
      vo1p0 = _mm_add_ps(vo1p0, _mm_mul_ps(vi1x6789, vk04));
      vo2p0 = _mm_add_ps(vo2p0, _mm_mul_ps(vi2x6789, vk04));
      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi1x6789, vk14));
      vo1p0 = _mm_add_ps(vo1p0, _mm_mul_ps(vi2x6789, vk14));
      vo2p0 = _mm_add_ps(vo2p0, _mm_mul_ps(vi3x6789, vk14));
      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi2x6789, vk24));
      vo1p0 = _mm_add_ps(vo1p0, _mm_mul_ps(vi3x6789, vk24));
      vo2p0 = _mm_add_ps(vo2p0, _mm_mul_ps(vi4x6789, vk24));
      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi3x6789, vk34));
      vo1p0 = _mm_add_ps(vo1p0, _mm_mul_ps(vi4x6789, vk34));
      vo2p0 = _mm_add_ps(vo2p0, _mm_mul_ps(vi5x6789, vk34));
      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi4x6789, vk44));
      vo1p0 = _mm_add_ps(vo1p0, _mm_mul_ps(vi5x6789, vk44));
      vo2p0 = _mm_add_ps(vo2p0, _mm_mul_ps(vi6x6789, vk44));


      __m128 vo0 = _mm_max_ps(vo0p0, vmin);
      __m128 vo1 = _mm_max_ps(vo1p0, vmin);
      __m128 vo2 = _mm_max_ps(vo2p0, vmin);

      vo0 = _mm_min_ps(vo0, vmax);
      vo1 = _mm_min_ps(vo1, vmax);
      vo2 = _mm_min_ps(vo2, vmax);

      _mm_storeu_ps(o2, vo2);
      o2 += 4;
      _mm_storeu_ps(o1, vo1);
      o1 += 4;
      _mm_storeu_ps(o0, vo0);
      o0 += 4;
    }
    // Always process the last block of 5..8 pixels.
    if XNN_LIKELY(w > 4 * sizeof(float)) {
      __m128 vo0p0 = _mm_add_ps(vbias, _mm_mul_ps(vi0x4567, vk02));
      __m128 vo1p0 = _mm_add_ps(vbias, _mm_mul_ps(vi1x4567, vk02));
      __m128 vo2p0 = _mm_add_ps(vbias, _mm_mul_ps(vi2x4567, vk02));
      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi1x4567, vk12));
      vo1p0 = _mm_add_ps(vo1p0, _mm_mul_ps(vi2x4567, vk12));
      vo2p0 = _mm_add_ps(vo2p0, _mm_mul_ps(vi3x4567, vk12));
      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi2x4567, vk22));
      vo1p0 = _mm_add_ps(vo1p0, _mm_mul_ps(vi3x4567, vk22));
      vo2p0 = _mm_add_ps(vo2p0, _mm_mul_ps(vi4x4567, vk22));
      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi3x4567, vk32));
      vo1p0 = _mm_add_ps(vo1p0, _mm_mul_ps(vi4x4567, vk32));
      vo2p0 = _mm_add_ps(vo2p0, _mm_mul_ps(vi5x4567, vk32));
      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi4x4567, vk42));
      vo1p0 = _mm_add_ps(vo1p0, _mm_mul_ps(vi5x4567, vk42));
      vo2p0 = _mm_add_ps(vo2p0, _mm_mul_ps(vi6x4567, vk42));

      const __m128 vi0x7456 = _mm_shuffle_ps(vi0x4567, vi0x4567, _MM_SHUFFLE(2, 1, 0, 3));
      const __m128 vi1x7456 = _mm_shuffle_ps(vi1x4567, vi1x4567, _MM_SHUFFLE(2, 1, 0, 3));
      const __m128 vi2x7456 = _mm_shuffle_ps(vi2x4567, vi2x4567, _MM_SHUFFLE(2, 1, 0, 3));
      const __m128 vi3x7456 = _mm_shuffle_ps(vi3x4567, vi3x4567, _MM_SHUFFLE(2, 1, 0, 3));
      const __m128 vi4x7456 = _mm_shuffle_ps(vi4x4567, vi4x4567, _MM_SHUFFLE(2, 1, 0, 3));
      const __m128 vi5x7456 = _mm_shuffle_ps(vi5x4567, vi5x4567, _MM_SHUFFLE(2, 1, 0, 3));
      const __m128 vi6x7456 = _mm_shuffle_ps(vi6x4567, vi6x4567, _MM_SHUFFLE(2, 1, 0, 3));

      const __m128 vi0x89AB = _mm_and_ps(_mm_loadu_ps(i0), vmask);
      i0 += 4;
      const __m128 vi1x89AB = _mm_and_ps(_mm_loadu_ps(i1), vmask);
      i1 += 4;
      const __m128 vi2x89AB = _mm_and_ps(_mm_loadu_ps(i2), vmask);
      i2 += 4;
      const __m128 vi3x89AB = _mm_and_ps(_mm_loadu_ps(i3), vmask);
      i3 += 4;
      const __m128 vi4x89AB = _mm_and_ps(_mm_loadu_ps(i4), vmask);
      i4 += 4;
      const __m128 vi5x89AB = _mm_and_ps(_mm_loadu_ps(i5), vmask);
      i5 += 4;
      const __m128 vi6x89AB = _mm_and_ps(_mm_loadu_ps(i6), vmask);
      i6 += 4;

      const __m128 vi0x3456 = _mm_move_ss(vi0x7456, vi0x3012);
      const __m128 vi1x3456 = _mm_move_ss(vi1x7456, vi1x3012);
      const __m128 vi2x3456 = _mm_move_ss(vi2x7456, vi2x3012);
      const __m128 vi3x3456 = _mm_move_ss(vi3x7456, vi3x3012);
      const __m128 vi4x3456 = _mm_move_ss(vi4x7456, vi4x3012);
      const __m128 vi5x3456 = _mm_move_ss(vi5x7456, vi5x3012);
      const __m128 vi6x3456 = _mm_move_ss(vi6x7456, vi6x3012);

      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi0x3456, vk01));
      vo1p0 = _mm_add_ps(vo1p0, _mm_mul_ps(vi1x3456, vk01));
      vo2p0 = _mm_add_ps(vo2p0, _mm_mul_ps(vi2x3456, vk01));
      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi1x3456, vk11));
      vo1p0 = _mm_add_ps(vo1p0, _mm_mul_ps(vi2x3456, vk11));
      vo2p0 = _mm_add_ps(vo2p0, _mm_mul_ps(vi3x3456, vk11));
      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi2x3456, vk21));
      vo1p0 = _mm_add_ps(vo1p0, _mm_mul_ps(vi3x3456, vk21));
      vo2p0 = _mm_add_ps(vo2p0, _mm_mul_ps(vi4x3456, vk21));
      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi3x3456, vk31));
      vo1p0 = _mm_add_ps(vo1p0, _mm_mul_ps(vi4x3456, vk31));
      vo2p0 = _mm_add_ps(vo2p0, _mm_mul_ps(vi5x3456, vk31));
      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi4x3456, vk41));
      vo1p0 = _mm_add_ps(vo1p0, _mm_mul_ps(vi5x3456, vk41));
      vo2p0 = _mm_add_ps(vo2p0, _mm_mul_ps(vi6x3456, vk41));

      const __m128 vi0x2345 = _mm_shuffle_ps(vi0x3012, vi0x7456, _MM_SHUFFLE(2, 1, 0, 3));
      vi0x3012 = vi0x7456;
      const __m128 vi1x2345 = _mm_shuffle_ps(vi1x3012, vi1x7456, _MM_SHUFFLE(2, 1, 0, 3));
      vi1x3012 = vi1x7456;
      const __m128 vi2x2345 = _mm_shuffle_ps(vi2x3012, vi2x7456, _MM_SHUFFLE(2, 1, 0, 3));
      vi2x3012 = vi2x7456;
      const __m128 vi3x2345 = _mm_shuffle_ps(vi3x3012, vi3x7456, _MM_SHUFFLE(2, 1, 0, 3));
      vi3x3012 = vi3x7456;
      const __m128 vi4x2345 = _mm_shuffle_ps(vi4x3012, vi4x7456, _MM_SHUFFLE(2, 1, 0, 3));
      vi4x3012 = vi4x7456;
      const __m128 vi5x2345 = _mm_shuffle_ps(vi5x3012, vi5x7456, _MM_SHUFFLE(2, 1, 0, 3));
      vi5x3012 = vi5x7456;
      const __m128 vi6x2345 = _mm_shuffle_ps(vi6x3012, vi6x7456, _MM_SHUFFLE(2, 1, 0, 3));
      vi6x3012 = vi6x7456;

      const __m128 vi0x8567 = _mm_move_ss(vi0x4567, vi0x89AB);
      vi0x4567 = vi0x89AB;
      const __m128 vi1x8567 = _mm_move_ss(vi1x4567, vi1x89AB);
      vi1x4567 = vi1x89AB;
      const __m128 vi2x8567 = _mm_move_ss(vi2x4567, vi2x89AB);
      vi2x4567 = vi2x89AB;
      const __m128 vi3x8567 = _mm_move_ss(vi3x4567, vi3x89AB);
      vi3x4567 = vi3x89AB;
      const __m128 vi4x8567 = _mm_move_ss(vi4x4567, vi4x89AB);
      vi4x4567 = vi4x89AB;
      const __m128 vi5x8567 = _mm_move_ss(vi5x4567, vi5x89AB);
      vi5x4567 = vi5x89AB;
      const __m128 vi6x8567 = _mm_move_ss(vi6x4567, vi6x89AB);
      vi6x4567 = vi6x89AB;

      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi0x2345, vk00));
      vo1p0 = _mm_add_ps(vo1p0, _mm_mul_ps(vi1x2345, vk00));
      vo2p0 = _mm_add_ps(vo2p0, _mm_mul_ps(vi2x2345, vk00));
      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi1x2345, vk10));
      vo1p0 = _mm_add_ps(vo1p0, _mm_mul_ps(vi2x2345, vk10));
      vo2p0 = _mm_add_ps(vo2p0, _mm_mul_ps(vi3x2345, vk10));
      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi2x2345, vk20));
      vo1p0 = _mm_add_ps(vo1p0, _mm_mul_ps(vi3x2345, vk20));
      vo2p0 = _mm_add_ps(vo2p0, _mm_mul_ps(vi4x2345, vk20));
      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi3x2345, vk30));
      vo1p0 = _mm_add_ps(vo1p0, _mm_mul_ps(vi4x2345, vk30));
      vo2p0 = _mm_add_ps(vo2p0, _mm_mul_ps(vi5x2345, vk30));
      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi4x2345, vk40));
      vo1p0 = _mm_add_ps(vo1p0, _mm_mul_ps(vi5x2345, vk40));
      vo2p0 = _mm_add_ps(vo2p0, _mm_mul_ps(vi6x2345, vk40));

      const __m128 vi0x5678 = _mm_shuffle_ps(vi0x8567, vi0x8567, _MM_SHUFFLE(0, 3, 2, 1));
      const __m128 vi1x5678 = _mm_shuffle_ps(vi1x8567, vi1x8567, _MM_SHUFFLE(0, 3, 2, 1));
      const __m128 vi2x5678 = _mm_shuffle_ps(vi2x8567, vi2x8567, _MM_SHUFFLE(0, 3, 2, 1));
      const __m128 vi3x5678 = _mm_shuffle_ps(vi3x8567, vi3x8567, _MM_SHUFFLE(0, 3, 2, 1));
      const __m128 vi4x5678 = _mm_shuffle_ps(vi4x8567, vi4x8567, _MM_SHUFFLE(0, 3, 2, 1));
      const __m128 vi5x5678 = _mm_shuffle_ps(vi5x8567, vi5x8567, _MM_SHUFFLE(0, 3, 2, 1));
      const __m128 vi6x5678 = _mm_shuffle_ps(vi6x8567, vi6x8567, _MM_SHUFFLE(0, 3, 2, 1));

      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi0x5678, vk03));
      vo1p0 = _mm_add_ps(vo1p0, _mm_mul_ps(vi1x5678, vk03));
      vo2p0 = _mm_add_ps(vo2p0, _mm_mul_ps(vi2x5678, vk03));
      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi1x5678, vk13));
      vo1p0 = _mm_add_ps(vo1p0, _mm_mul_ps(vi2x5678, vk13));
      vo2p0 = _mm_add_ps(vo2p0, _mm_mul_ps(vi3x5678, vk13));
      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi2x5678, vk23));
      vo1p0 = _mm_add_ps(vo1p0, _mm_mul_ps(vi3x5678, vk23));
      vo2p0 = _mm_add_ps(vo2p0, _mm_mul_ps(vi4x5678, vk23));
      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi3x5678, vk33));
      vo1p0 = _mm_add_ps(vo1p0, _mm_mul_ps(vi4x5678, vk33));
      vo2p0 = _mm_add_ps(vo2p0, _mm_mul_ps(vi5x5678, vk33));
      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi4x5678, vk43));
      vo1p0 = _mm_add_ps(vo1p0, _mm_mul_ps(vi5x5678, vk43));
      vo2p0 = _mm_add_ps(vo2p0, _mm_mul_ps(vi6x5678, vk43));

      const __m128 vi0x6789 = _mm_shuffle_ps(vi0x5678, vi0x89AB, _MM_SHUFFLE(1, 0, 2, 1));
      const __m128 vi1x6789 = _mm_shuffle_ps(vi1x5678, vi1x89AB, _MM_SHUFFLE(1, 0, 2, 1));
      const __m128 vi2x6789 = _mm_shuffle_ps(vi2x5678, vi2x89AB, _MM_SHUFFLE(1, 0, 2, 1));
      const __m128 vi3x6789 = _mm_shuffle_ps(vi3x5678, vi3x89AB, _MM_SHUFFLE(1, 0, 2, 1));
      const __m128 vi4x6789 = _mm_shuffle_ps(vi4x5678, vi4x89AB, _MM_SHUFFLE(1, 0, 2, 1));
      const __m128 vi5x6789 = _mm_shuffle_ps(vi5x5678, vi5x89AB, _MM_SHUFFLE(1, 0, 2, 1));
      const __m128 vi6x6789 = _mm_shuffle_ps(vi6x5678, vi6x89AB, _MM_SHUFFLE(1, 0, 2, 1));

      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi0x6789, vk04));
      vo1p0 = _mm_add_ps(vo1p0, _mm_mul_ps(vi1x6789, vk04));
      vo2p0 = _mm_add_ps(vo2p0, _mm_mul_ps(vi2x6789, vk04));
      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi1x6789, vk14));
      vo1p0 = _mm_add_ps(vo1p0, _mm_mul_ps(vi2x6789, vk14));
      vo2p0 = _mm_add_ps(vo2p0, _mm_mul_ps(vi3x6789, vk14));
      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi2x6789, vk24));
      vo1p0 = _mm_add_ps(vo1p0, _mm_mul_ps(vi3x6789, vk24));
      vo2p0 = _mm_add_ps(vo2p0, _mm_mul_ps(vi4x6789, vk24));
      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi3x6789, vk34));
      vo1p0 = _mm_add_ps(vo1p0, _mm_mul_ps(vi4x6789, vk34));
      vo2p0 = _mm_add_ps(vo2p0, _mm_mul_ps(vi5x6789, vk34));
      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi4x6789, vk44));
      vo1p0 = _mm_add_ps(vo1p0, _mm_mul_ps(vi5x6789, vk44));
      vo2p0 = _mm_add_ps(vo2p0, _mm_mul_ps(vi6x6789, vk44));


      __m128 vo0 = _mm_max_ps(vo0p0, vmin);
      __m128 vo1 = _mm_max_ps(vo1p0, vmin);
      __m128 vo2 = _mm_max_ps(vo2p0, vmin);

      vo0 = _mm_min_ps(vo0, vmax);
      vo1 = _mm_min_ps(vo1, vmax);
      vo2 = _mm_min_ps(vo2, vmax);

      _mm_storeu_ps(o2, vo2);
      o2 += 4;
      _mm_storeu_ps(o1, vo1);
      o1 += 4;
      _mm_storeu_ps(o0, vo0);
      o0 += 4;

      w -= 4 * sizeof(float);
    }
    assert(w >= 1 * sizeof(float));
    assert(w <= 4 * sizeof(float));
    {
      vi0x4567 = _mm_and_ps(vi0x4567, vmask);
      vi1x4567 = _mm_and_ps(vi1x4567, vmask);
      vi2x4567 = _mm_and_ps(vi2x4567, vmask);
      vi3x4567 = _mm_and_ps(vi3x4567, vmask);
      vi4x4567 = _mm_and_ps(vi4x4567, vmask);
      vi5x4567 = _mm_and_ps(vi5x4567, vmask);
      vi6x4567 = _mm_and_ps(vi6x4567, vmask);

      __m128 vo0p0 = _mm_add_ps(vbias, _mm_mul_ps(vi0x4567, vk02));
      __m128 vo1p0 = _mm_add_ps(vbias, _mm_mul_ps(vi1x4567, vk02));
      __m128 vo2p0 = _mm_add_ps(vbias, _mm_mul_ps(vi2x4567, vk02));
      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi1x4567, vk12));
      vo1p0 = _mm_add_ps(vo1p0, _mm_mul_ps(vi2x4567, vk12));
      vo2p0 = _mm_add_ps(vo2p0, _mm_mul_ps(vi3x4567, vk12));
      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi2x4567, vk22));
      vo1p0 = _mm_add_ps(vo1p0, _mm_mul_ps(vi3x4567, vk22));
      vo2p0 = _mm_add_ps(vo2p0, _mm_mul_ps(vi4x4567, vk22));
      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi3x4567, vk32));
      vo1p0 = _mm_add_ps(vo1p0, _mm_mul_ps(vi4x4567, vk32));
      vo2p0 = _mm_add_ps(vo2p0, _mm_mul_ps(vi5x4567, vk32));
      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi4x4567, vk42));
      vo1p0 = _mm_add_ps(vo1p0, _mm_mul_ps(vi5x4567, vk42));
      vo2p0 = _mm_add_ps(vo2p0, _mm_mul_ps(vi6x4567, vk42));

      const __m128 vi0x7456 = _mm_shuffle_ps(vi0x4567, vi0x4567, _MM_SHUFFLE(2, 1, 0, 3));
      const __m128 vi1x7456 = _mm_shuffle_ps(vi1x4567, vi1x4567, _MM_SHUFFLE(2, 1, 0, 3));
      const __m128 vi2x7456 = _mm_shuffle_ps(vi2x4567, vi2x4567, _MM_SHUFFLE(2, 1, 0, 3));
      const __m128 vi3x7456 = _mm_shuffle_ps(vi3x4567, vi3x4567, _MM_SHUFFLE(2, 1, 0, 3));
      const __m128 vi4x7456 = _mm_shuffle_ps(vi4x4567, vi4x4567, _MM_SHUFFLE(2, 1, 0, 3));
      const __m128 vi5x7456 = _mm_shuffle_ps(vi5x4567, vi5x4567, _MM_SHUFFLE(2, 1, 0, 3));
      const __m128 vi6x7456 = _mm_shuffle_ps(vi6x4567, vi6x4567, _MM_SHUFFLE(2, 1, 0, 3));

      const __m128 vi0x3456 = _mm_move_ss(vi0x7456, vi0x3012);
      const __m128 vi1x3456 = _mm_move_ss(vi1x7456, vi1x3012);
      const __m128 vi2x3456 = _mm_move_ss(vi2x7456, vi2x3012);
      const __m128 vi3x3456 = _mm_move_ss(vi3x7456, vi3x3012);
      const __m128 vi4x3456 = _mm_move_ss(vi4x7456, vi4x3012);
      const __m128 vi5x3456 = _mm_move_ss(vi5x7456, vi5x3012);
      const __m128 vi6x3456 = _mm_move_ss(vi6x7456, vi6x3012);

      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi0x3456, vk01));
      vo1p0 = _mm_add_ps(vo1p0, _mm_mul_ps(vi1x3456, vk01));
      vo2p0 = _mm_add_ps(vo2p0, _mm_mul_ps(vi2x3456, vk01));
      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi1x3456, vk11));
      vo1p0 = _mm_add_ps(vo1p0, _mm_mul_ps(vi2x3456, vk11));
      vo2p0 = _mm_add_ps(vo2p0, _mm_mul_ps(vi3x3456, vk11));
      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi2x3456, vk21));
      vo1p0 = _mm_add_ps(vo1p0, _mm_mul_ps(vi3x3456, vk21));
      vo2p0 = _mm_add_ps(vo2p0, _mm_mul_ps(vi4x3456, vk21));
      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi3x3456, vk31));
      vo1p0 = _mm_add_ps(vo1p0, _mm_mul_ps(vi4x3456, vk31));
      vo2p0 = _mm_add_ps(vo2p0, _mm_mul_ps(vi5x3456, vk31));
      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi4x3456, vk41));
      vo1p0 = _mm_add_ps(vo1p0, _mm_mul_ps(vi5x3456, vk41));
      vo2p0 = _mm_add_ps(vo2p0, _mm_mul_ps(vi6x3456, vk41));

      const __m128 vi0x2345 = _mm_shuffle_ps(vi0x3012, vi0x7456, _MM_SHUFFLE(2, 1, 0, 3));
      const __m128 vi1x2345 = _mm_shuffle_ps(vi1x3012, vi1x7456, _MM_SHUFFLE(2, 1, 0, 3));
      const __m128 vi2x2345 = _mm_shuffle_ps(vi2x3012, vi2x7456, _MM_SHUFFLE(2, 1, 0, 3));
      const __m128 vi3x2345 = _mm_shuffle_ps(vi3x3012, vi3x7456, _MM_SHUFFLE(2, 1, 0, 3));
      const __m128 vi4x2345 = _mm_shuffle_ps(vi4x3012, vi4x7456, _MM_SHUFFLE(2, 1, 0, 3));
      const __m128 vi5x2345 = _mm_shuffle_ps(vi5x3012, vi5x7456, _MM_SHUFFLE(2, 1, 0, 3));
      const __m128 vi6x2345 = _mm_shuffle_ps(vi6x3012, vi6x7456, _MM_SHUFFLE(2, 1, 0, 3));

      const __m128 vzero = _mm_setzero_ps();
      const __m128 vi0x8567 = _mm_move_ss(vi0x4567, vzero);
      const __m128 vi1x8567 = _mm_move_ss(vi1x4567, vzero);
      const __m128 vi2x8567 = _mm_move_ss(vi2x4567, vzero);
      const __m128 vi3x8567 = _mm_move_ss(vi3x4567, vzero);
      const __m128 vi4x8567 = _mm_move_ss(vi4x4567, vzero);
      const __m128 vi5x8567 = _mm_move_ss(vi5x4567, vzero);
      const __m128 vi6x8567 = _mm_move_ss(vi6x4567, vzero);

      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi0x2345, vk00));
      vo1p0 = _mm_add_ps(vo1p0, _mm_mul_ps(vi1x2345, vk00));
      vo2p0 = _mm_add_ps(vo2p0, _mm_mul_ps(vi2x2345, vk00));
      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi1x2345, vk10));
      vo1p0 = _mm_add_ps(vo1p0, _mm_mul_ps(vi2x2345, vk10));
      vo2p0 = _mm_add_ps(vo2p0, _mm_mul_ps(vi3x2345, vk10));
      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi2x2345, vk20));
      vo1p0 = _mm_add_ps(vo1p0, _mm_mul_ps(vi3x2345, vk20));
      vo2p0 = _mm_add_ps(vo2p0, _mm_mul_ps(vi4x2345, vk20));
      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi3x2345, vk30));
      vo1p0 = _mm_add_ps(vo1p0, _mm_mul_ps(vi4x2345, vk30));
      vo2p0 = _mm_add_ps(vo2p0, _mm_mul_ps(vi5x2345, vk30));
      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi4x2345, vk40));
      vo1p0 = _mm_add_ps(vo1p0, _mm_mul_ps(vi5x2345, vk40));
      vo2p0 = _mm_add_ps(vo2p0, _mm_mul_ps(vi6x2345, vk40));

      const __m128 vi0x5678 = _mm_shuffle_ps(vi0x8567, vi0x8567, _MM_SHUFFLE(0, 3, 2, 1));
      const __m128 vi1x5678 = _mm_shuffle_ps(vi1x8567, vi1x8567, _MM_SHUFFLE(0, 3, 2, 1));
      const __m128 vi2x5678 = _mm_shuffle_ps(vi2x8567, vi2x8567, _MM_SHUFFLE(0, 3, 2, 1));
      const __m128 vi3x5678 = _mm_shuffle_ps(vi3x8567, vi3x8567, _MM_SHUFFLE(0, 3, 2, 1));
      const __m128 vi4x5678 = _mm_shuffle_ps(vi4x8567, vi4x8567, _MM_SHUFFLE(0, 3, 2, 1));
      const __m128 vi5x5678 = _mm_shuffle_ps(vi5x8567, vi5x8567, _MM_SHUFFLE(0, 3, 2, 1));
      const __m128 vi6x5678 = _mm_shuffle_ps(vi6x8567, vi6x8567, _MM_SHUFFLE(0, 3, 2, 1));

      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi0x5678, vk03));
      vo1p0 = _mm_add_ps(vo1p0, _mm_mul_ps(vi1x5678, vk03));
      vo2p0 = _mm_add_ps(vo2p0, _mm_mul_ps(vi2x5678, vk03));
      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi1x5678, vk13));
      vo1p0 = _mm_add_ps(vo1p0, _mm_mul_ps(vi2x5678, vk13));
      vo2p0 = _mm_add_ps(vo2p0, _mm_mul_ps(vi3x5678, vk13));
      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi2x5678, vk23));
      vo1p0 = _mm_add_ps(vo1p0, _mm_mul_ps(vi3x5678, vk23));
      vo2p0 = _mm_add_ps(vo2p0, _mm_mul_ps(vi4x5678, vk23));
      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi3x5678, vk33));
      vo1p0 = _mm_add_ps(vo1p0, _mm_mul_ps(vi4x5678, vk33));
      vo2p0 = _mm_add_ps(vo2p0, _mm_mul_ps(vi5x5678, vk33));
      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi4x5678, vk43));
      vo1p0 = _mm_add_ps(vo1p0, _mm_mul_ps(vi5x5678, vk43));
      vo2p0 = _mm_add_ps(vo2p0, _mm_mul_ps(vi6x5678, vk43));

      const __m128 vi0x6789 = _mm_shuffle_ps(vi0x5678, vzero, _MM_SHUFFLE(1, 0, 2, 1));
      const __m128 vi1x6789 = _mm_shuffle_ps(vi1x5678, vzero, _MM_SHUFFLE(1, 0, 2, 1));
      const __m128 vi2x6789 = _mm_shuffle_ps(vi2x5678, vzero, _MM_SHUFFLE(1, 0, 2, 1));
      const __m128 vi3x6789 = _mm_shuffle_ps(vi3x5678, vzero, _MM_SHUFFLE(1, 0, 2, 1));
      const __m128 vi4x6789 = _mm_shuffle_ps(vi4x5678, vzero, _MM_SHUFFLE(1, 0, 2, 1));
      const __m128 vi5x6789 = _mm_shuffle_ps(vi5x5678, vzero, _MM_SHUFFLE(1, 0, 2, 1));
      const __m128 vi6x6789 = _mm_shuffle_ps(vi6x5678, vzero, _MM_SHUFFLE(1, 0, 2, 1));

      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi0x6789, vk04));
      vo1p0 = _mm_add_ps(vo1p0, _mm_mul_ps(vi1x6789, vk04));
      vo2p0 = _mm_add_ps(vo2p0, _mm_mul_ps(vi2x6789, vk04));
      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi1x6789, vk14));
      vo1p0 = _mm_add_ps(vo1p0, _mm_mul_ps(vi2x6789, vk14));
      vo2p0 = _mm_add_ps(vo2p0, _mm_mul_ps(vi3x6789, vk14));
      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi2x6789, vk24));
      vo1p0 = _mm_add_ps(vo1p0, _mm_mul_ps(vi3x6789, vk24));
      vo2p0 = _mm_add_ps(vo2p0, _mm_mul_ps(vi4x6789, vk24));
      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi3x6789, vk34));
      vo1p0 = _mm_add_ps(vo1p0, _mm_mul_ps(vi4x6789, vk34));
      vo2p0 = _mm_add_ps(vo2p0, _mm_mul_ps(vi5x6789, vk34));
      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi4x6789, vk44));
      vo1p0 = _mm_add_ps(vo1p0, _mm_mul_ps(vi5x6789, vk44));
      vo2p0 = _mm_add_ps(vo2p0, _mm_mul_ps(vi6x6789, vk44));


      __m128 vo0 = _mm_max_ps(vo0p0, vmin);
      __m128 vo1 = _mm_max_ps(vo1p0, vmin);
      __m128 vo2 = _mm_max_ps(vo2p0, vmin);

      vo0 = _mm_min_ps(vo0, vmax);
      vo1 = _mm_min_ps(vo1, vmax);
      vo2 = _mm_min_ps(vo2, vmax);

      if XNN_LIKELY(w & (4 * sizeof(float))) {
        _mm_storeu_ps(o2, vo2);
        o2 += 4;
        _mm_storeu_ps(o1, vo1);
        o1 += 4;
        _mm_storeu_ps(o0, vo0);
        o0 += 4;
      } else {
        if (w & (2 * sizeof(float))) {
          _mm_storel_pi((__m64*) o2, vo2);
          o2 += 2;
          _mm_storel_pi((__m64*) o1, vo1);
          o1 += 2;
          _mm_storel_pi((__m64*) o0, vo0);
          o0 += 2;

          vo0 = _mm_movehl_ps(vo0, vo0);
          vo1 = _mm_movehl_ps(vo1, vo1);
          vo2 = _mm_movehl_ps(vo2, vo2);
        }
        if (w & (1 * sizeof(float))) {
          _mm_store_ss(o2, vo2);
          o2 += 1;
          _mm_store_ss(o1, vo1);
          o1 += 1;
          _mm_store_ss(o0, vo0);
          o0 += 1;
        }
      }
    }

    i0 = (const float*) ((uintptr_t) i3 - input_decrement);
    i1 = (const float*) ((uintptr_t) i4 - input_decrement);
    i2 = (const float*) ((uintptr_t) i1 + input_width);
    i3 = (const float*) ((uintptr_t) i2 + input_width);
    i4 = (const float*) ((uintptr_t) i3 + input_width);
    i5 = (const float*) ((uintptr_t) i4 + input_width);
    i6 = (const float*) ((uintptr_t) i5 + input_width);

    o0 = o2;
    o1 = (float*) ((uintptr_t) o0 + input_width);
    o2 = (float*) ((uintptr_t) o1 + input_width);

    output_height = doz(output_height, 3);
  } while (output_height != 0);
}
