// Auto-generated file. Do not edit!
//   Template: src/f32-dwconv2d-chw/5x5s2p2-sse.c.in
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


void xnn_f32_dwconv2d_chw_ukernel_5x5s2p2__sse_2x4_acc2(
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
  assert(padding_top >= 1);
  assert(padding_top <= 2);

  const __m128 vmask_even = _mm_load_ps((const float*) params->sse_stride2.mask_even);
  const __m128 vmask_odd  = _mm_load_ps((const float*) params->sse_stride2.mask_odd);
  const __m128 vmax = _mm_set1_ps(params->sse_stride2.max);
  const __m128 vmin = _mm_set1_ps(params->sse_stride2.min);
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

  const uint32_t padding_top_less_1 = padding_top - 1;
  const size_t input_decrement = round_up_po2(input_width, 8 * sizeof(float));

  const float* i0 = zero;
  const float* i1 = (const float*) ((uintptr_t) input - ((-padding_top_less_1) & input_width));
  const float* i2 = (const float*) ((uintptr_t) i1 + input_width);
  if XNN_UNPREDICTABLE(padding_top_less_1 != 0) {
    i1 = zero;
  }
  const float* i3 = (const float*) ((uintptr_t) i2 + input_width);
  const float* i4 = (const float*) ((uintptr_t) i3 + input_width);
  const float* i5 = (const float*) ((uintptr_t) i4 + input_width);
  const float* i6 = (const float*) ((uintptr_t) i5 + input_width);

  const size_t output_width = round_down_po2((input_width + (2 /* padding */ - 3 /* kernel size */ + 2 /* subsampling */) * sizeof(float)) / 2, sizeof(float));

  float* o0 = output;
  float* o1 = (float*) ((uintptr_t) o0 + output_width);

  size_t padded_input_height = input_height + (padding_top_less_1 + 1) + 2 /* padding bottom */;
  size_t output_height = (padded_input_height - 5 /* kernel size */ + 2 /* subsampling */) / 2;
  do {
    if XNN_UNPREDICTABLE(padded_input_height < 6) {
      i3 = zero;
    }
    if XNN_UNPREDICTABLE(padded_input_height < 7) {
      i4 = zero;
      o1 = o0;
    }
    if XNN_UNPREDICTABLE(padded_input_height < 8) {
      i5 = zero;
    }
    if XNN_UNPREDICTABLE(padded_input_height < 9) {
      i6 = zero;
    }

    __m128 vi0x6024 = _mm_setzero_ps();
    __m128 vi1x6024 = _mm_setzero_ps();
    __m128 vi2x6024 = _mm_setzero_ps();
    __m128 vi3x6024 = _mm_setzero_ps();
    __m128 vi4x6024 = _mm_setzero_ps();
    __m128 vi5x6024 = _mm_setzero_ps();
    __m128 vi6x6024 = _mm_setzero_ps();

    __m128 vi0x7135 = _mm_setzero_ps();
    __m128 vi1x7135 = _mm_setzero_ps();
    __m128 vi2x7135 = _mm_setzero_ps();
    __m128 vi3x7135 = _mm_setzero_ps();
    __m128 vi4x7135 = _mm_setzero_ps();
    __m128 vi5x7135 = _mm_setzero_ps();
    __m128 vi6x7135 = _mm_setzero_ps();

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

    __m128 vi0x8ACE = _mm_shuffle_ps(vi0x89AB, vi0xCDEF, _MM_SHUFFLE(2, 0, 2, 0));
    __m128 vi0x9BDF = _mm_shuffle_ps(vi0x89AB, vi0xCDEF, _MM_SHUFFLE(3, 1, 3, 1));
    __m128 vi1x8ACE = _mm_shuffle_ps(vi1x89AB, vi1xCDEF, _MM_SHUFFLE(2, 0, 2, 0));
    __m128 vi1x9BDF = _mm_shuffle_ps(vi1x89AB, vi1xCDEF, _MM_SHUFFLE(3, 1, 3, 1));
    __m128 vi2x8ACE = _mm_shuffle_ps(vi2x89AB, vi2xCDEF, _MM_SHUFFLE(2, 0, 2, 0));
    __m128 vi2x9BDF = _mm_shuffle_ps(vi2x89AB, vi2xCDEF, _MM_SHUFFLE(3, 1, 3, 1));
    __m128 vi3x8ACE = _mm_shuffle_ps(vi3x89AB, vi3xCDEF, _MM_SHUFFLE(2, 0, 2, 0));
    __m128 vi3x9BDF = _mm_shuffle_ps(vi3x89AB, vi3xCDEF, _MM_SHUFFLE(3, 1, 3, 1));
    __m128 vi4x8ACE = _mm_shuffle_ps(vi4x89AB, vi4xCDEF, _MM_SHUFFLE(2, 0, 2, 0));
    __m128 vi4x9BDF = _mm_shuffle_ps(vi4x89AB, vi4xCDEF, _MM_SHUFFLE(3, 1, 3, 1));
    __m128 vi5x8ACE = _mm_shuffle_ps(vi5x89AB, vi5xCDEF, _MM_SHUFFLE(2, 0, 2, 0));
    __m128 vi5x9BDF = _mm_shuffle_ps(vi5x89AB, vi5xCDEF, _MM_SHUFFLE(3, 1, 3, 1));
    __m128 vi6x8ACE = _mm_shuffle_ps(vi6x89AB, vi6xCDEF, _MM_SHUFFLE(2, 0, 2, 0));
    __m128 vi6x9BDF = _mm_shuffle_ps(vi6x89AB, vi6xCDEF, _MM_SHUFFLE(3, 1, 3, 1));

    size_t w = input_width;
    for (; w > 8 * sizeof(float); w -= 8 * sizeof(float)) {
      __m128 vo0p0 = _mm_add_ps(vbias, _mm_mul_ps(vi0x8ACE, vk02));
      __m128 vo1p0 = _mm_add_ps(vbias, _mm_mul_ps(vi2x8ACE, vk02));
      __m128 vo0p1 = _mm_mul_ps(vi1x8ACE, vk12);
      __m128 vo1p1 = _mm_mul_ps(vi3x8ACE, vk12);
      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi2x8ACE, vk22));
      vo1p0 = _mm_add_ps(vo1p0, _mm_mul_ps(vi4x8ACE, vk22));
      vo0p1 = _mm_add_ps(vo0p1, _mm_mul_ps(vi3x8ACE, vk32));
      vo1p1 = _mm_add_ps(vo1p1, _mm_mul_ps(vi5x8ACE, vk32));
      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi4x8ACE, vk42));
      vo1p0 = _mm_add_ps(vo1p0, _mm_mul_ps(vi6x8ACE, vk42));

      const __m128 vi0xE8AC = _mm_shuffle_ps(vi0x8ACE, vi0x8ACE, _MM_SHUFFLE(2, 1, 0, 3));
      const __m128 vi1xE8AC = _mm_shuffle_ps(vi1x8ACE, vi1x8ACE, _MM_SHUFFLE(2, 1, 0, 3));
      const __m128 vi2xE8AC = _mm_shuffle_ps(vi2x8ACE, vi2x8ACE, _MM_SHUFFLE(2, 1, 0, 3));
      const __m128 vi3xE8AC = _mm_shuffle_ps(vi3x8ACE, vi3x8ACE, _MM_SHUFFLE(2, 1, 0, 3));
      const __m128 vi4xE8AC = _mm_shuffle_ps(vi4x8ACE, vi4x8ACE, _MM_SHUFFLE(2, 1, 0, 3));
      const __m128 vi5xE8AC = _mm_shuffle_ps(vi5x8ACE, vi5x8ACE, _MM_SHUFFLE(2, 1, 0, 3));
      const __m128 vi6xE8AC = _mm_shuffle_ps(vi6x8ACE, vi6x8ACE, _MM_SHUFFLE(2, 1, 0, 3));

      vo0p1 = _mm_add_ps(vo0p1, _mm_mul_ps(vi0x9BDF, vk03));
      vo1p1 = _mm_add_ps(vo1p1, _mm_mul_ps(vi2x9BDF, vk03));
      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi1x9BDF, vk13));
      vo1p0 = _mm_add_ps(vo1p0, _mm_mul_ps(vi3x9BDF, vk13));
      vo0p1 = _mm_add_ps(vo0p1, _mm_mul_ps(vi2x9BDF, vk23));
      vo1p1 = _mm_add_ps(vo1p1, _mm_mul_ps(vi4x9BDF, vk23));
      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi3x9BDF, vk33));
      vo1p0 = _mm_add_ps(vo1p0, _mm_mul_ps(vi5x9BDF, vk33));
      vo0p1 = _mm_add_ps(vo0p1, _mm_mul_ps(vi4x9BDF, vk43));
      vo1p1 = _mm_add_ps(vo1p1, _mm_mul_ps(vi6x9BDF, vk43));

      const __m128 vi0x68AC = _mm_move_ss(vi0xE8AC, vi0x6024);
      vi0x6024 = vi0xE8AC;
      const __m128 vi1x68AC = _mm_move_ss(vi1xE8AC, vi1x6024);
      vi1x6024 = vi1xE8AC;
      const __m128 vi2x68AC = _mm_move_ss(vi2xE8AC, vi2x6024);
      vi2x6024 = vi2xE8AC;
      const __m128 vi3x68AC = _mm_move_ss(vi3xE8AC, vi3x6024);
      vi3x6024 = vi3xE8AC;
      const __m128 vi4x68AC = _mm_move_ss(vi4xE8AC, vi4x6024);
      vi4x6024 = vi4xE8AC;
      const __m128 vi5x68AC = _mm_move_ss(vi5xE8AC, vi5x6024);
      vi5x6024 = vi5xE8AC;
      const __m128 vi6x68AC = _mm_move_ss(vi6xE8AC, vi6x6024);
      vi6x6024 = vi6xE8AC;

      const __m128 vi0xF9BD = _mm_shuffle_ps(vi0x9BDF, vi0x9BDF, _MM_SHUFFLE(2, 1, 0, 3));
      const __m128 vi1xF9BD = _mm_shuffle_ps(vi1x9BDF, vi1x9BDF, _MM_SHUFFLE(2, 1, 0, 3));
      const __m128 vi2xF9BD = _mm_shuffle_ps(vi2x9BDF, vi2x9BDF, _MM_SHUFFLE(2, 1, 0, 3));
      const __m128 vi3xF9BD = _mm_shuffle_ps(vi3x9BDF, vi3x9BDF, _MM_SHUFFLE(2, 1, 0, 3));
      const __m128 vi4xF9BD = _mm_shuffle_ps(vi4x9BDF, vi4x9BDF, _MM_SHUFFLE(2, 1, 0, 3));
      const __m128 vi5xF9BD = _mm_shuffle_ps(vi5x9BDF, vi5x9BDF, _MM_SHUFFLE(2, 1, 0, 3));
      const __m128 vi6xF9BD = _mm_shuffle_ps(vi6x9BDF, vi6x9BDF, _MM_SHUFFLE(2, 1, 0, 3));

      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi0x68AC, vk00));
      vo1p0 = _mm_add_ps(vo1p0, _mm_mul_ps(vi2x68AC, vk00));
      vo0p1 = _mm_add_ps(vo0p1, _mm_mul_ps(vi1x68AC, vk10));
      vo1p1 = _mm_add_ps(vo1p1, _mm_mul_ps(vi3x68AC, vk10));
      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi2x68AC, vk20));
      vo1p0 = _mm_add_ps(vo1p0, _mm_mul_ps(vi4x68AC, vk20));
      vo0p1 = _mm_add_ps(vo0p1, _mm_mul_ps(vi3x68AC, vk30));
      vo1p1 = _mm_add_ps(vo1p1, _mm_mul_ps(vi5x68AC, vk30));
      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi4x68AC, vk40));
      vo1p0 = _mm_add_ps(vo1p0, _mm_mul_ps(vi6x68AC, vk40));

      const __m128 vi0xGHIJ = _mm_loadu_ps(i0);
      const __m128 vi0xKLMN = _mm_loadu_ps(i0 + 4);
      i0 += 8;
      const __m128 vi1xGHIJ = _mm_loadu_ps(i1);
      const __m128 vi1xKLMN = _mm_loadu_ps(i1 + 4);
      i1 += 8;
      const __m128 vi2xGHIJ = _mm_loadu_ps(i2);
      const __m128 vi2xKLMN = _mm_loadu_ps(i2 + 4);
      i2 += 8;
      const __m128 vi3xGHIJ = _mm_loadu_ps(i3);
      const __m128 vi3xKLMN = _mm_loadu_ps(i3 + 4);
      i3 += 8;
      const __m128 vi4xGHIJ = _mm_loadu_ps(i4);
      const __m128 vi4xKLMN = _mm_loadu_ps(i4 + 4);
      i4 += 8;
      const __m128 vi5xGHIJ = _mm_loadu_ps(i5);
      const __m128 vi5xKLMN = _mm_loadu_ps(i5 + 4);
      i5 += 8;
      const __m128 vi6xGHIJ = _mm_loadu_ps(i6);
      const __m128 vi6xKLMN = _mm_loadu_ps(i6 + 4);
      i6 += 8;

      const __m128 vi0x79BD = _mm_move_ss(vi0xF9BD, vi0x7135);
      vi0x7135 = vi0xF9BD;
      const __m128 vi1x79BD = _mm_move_ss(vi1xF9BD, vi1x7135);
      vi1x7135 = vi1xF9BD;
      const __m128 vi2x79BD = _mm_move_ss(vi2xF9BD, vi2x7135);
      vi2x7135 = vi2xF9BD;
      const __m128 vi3x79BD = _mm_move_ss(vi3xF9BD, vi3x7135);
      vi3x7135 = vi3xF9BD;
      const __m128 vi4x79BD = _mm_move_ss(vi4xF9BD, vi4x7135);
      vi4x7135 = vi4xF9BD;
      const __m128 vi5x79BD = _mm_move_ss(vi5xF9BD, vi5x7135);
      vi5x7135 = vi5xF9BD;
      const __m128 vi6x79BD = _mm_move_ss(vi6xF9BD, vi6x7135);
      vi6x7135 = vi6xF9BD;

      const __m128 vi0xGIKM = _mm_shuffle_ps(vi0xGHIJ, vi0xKLMN, _MM_SHUFFLE(2, 0, 2, 0));
      const __m128 vi0xHJLN = _mm_shuffle_ps(vi0xGHIJ, vi0xKLMN, _MM_SHUFFLE(3, 1, 3, 1));
      vi0x9BDF = vi0xHJLN;
      const __m128 vi1xGIKM = _mm_shuffle_ps(vi1xGHIJ, vi1xKLMN, _MM_SHUFFLE(2, 0, 2, 0));
      const __m128 vi1xHJLN = _mm_shuffle_ps(vi1xGHIJ, vi1xKLMN, _MM_SHUFFLE(3, 1, 3, 1));
      vi1x9BDF = vi1xHJLN;
      const __m128 vi2xGIKM = _mm_shuffle_ps(vi2xGHIJ, vi2xKLMN, _MM_SHUFFLE(2, 0, 2, 0));
      const __m128 vi2xHJLN = _mm_shuffle_ps(vi2xGHIJ, vi2xKLMN, _MM_SHUFFLE(3, 1, 3, 1));
      vi2x9BDF = vi2xHJLN;
      const __m128 vi3xGIKM = _mm_shuffle_ps(vi3xGHIJ, vi3xKLMN, _MM_SHUFFLE(2, 0, 2, 0));
      const __m128 vi3xHJLN = _mm_shuffle_ps(vi3xGHIJ, vi3xKLMN, _MM_SHUFFLE(3, 1, 3, 1));
      vi3x9BDF = vi3xHJLN;
      const __m128 vi4xGIKM = _mm_shuffle_ps(vi4xGHIJ, vi4xKLMN, _MM_SHUFFLE(2, 0, 2, 0));
      const __m128 vi4xHJLN = _mm_shuffle_ps(vi4xGHIJ, vi4xKLMN, _MM_SHUFFLE(3, 1, 3, 1));
      vi4x9BDF = vi4xHJLN;
      const __m128 vi5xGIKM = _mm_shuffle_ps(vi5xGHIJ, vi5xKLMN, _MM_SHUFFLE(2, 0, 2, 0));
      const __m128 vi5xHJLN = _mm_shuffle_ps(vi5xGHIJ, vi5xKLMN, _MM_SHUFFLE(3, 1, 3, 1));
      vi5x9BDF = vi5xHJLN;
      const __m128 vi6xGIKM = _mm_shuffle_ps(vi6xGHIJ, vi6xKLMN, _MM_SHUFFLE(2, 0, 2, 0));
      const __m128 vi6xHJLN = _mm_shuffle_ps(vi6xGHIJ, vi6xKLMN, _MM_SHUFFLE(3, 1, 3, 1));
      vi6x9BDF = vi6xHJLN;

      vo0p1 = _mm_add_ps(vo0p1, _mm_mul_ps(vi0x79BD, vk01));
      vo1p1 = _mm_add_ps(vo1p1, _mm_mul_ps(vi2x79BD, vk01));
      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi1x79BD, vk11));
      vo1p0 = _mm_add_ps(vo1p0, _mm_mul_ps(vi3x79BD, vk11));
      vo0p1 = _mm_add_ps(vo0p1, _mm_mul_ps(vi2x79BD, vk21));
      vo1p1 = _mm_add_ps(vo1p1, _mm_mul_ps(vi4x79BD, vk21));
      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi3x79BD, vk31));
      vo1p0 = _mm_add_ps(vo1p0, _mm_mul_ps(vi5x79BD, vk31));
      vo0p1 = _mm_add_ps(vo0p1, _mm_mul_ps(vi4x79BD, vk41));
      vo1p1 = _mm_add_ps(vo1p1, _mm_mul_ps(vi6x79BD, vk41));

      const __m128 vi0xGACE = _mm_move_ss(vi0x8ACE, vi0xGIKM);
      vi0x8ACE = vi0xGIKM;
      const __m128 vi1xGACE = _mm_move_ss(vi1x8ACE, vi1xGIKM);
      vi1x8ACE = vi1xGIKM;
      const __m128 vi2xGACE = _mm_move_ss(vi2x8ACE, vi2xGIKM);
      vi2x8ACE = vi2xGIKM;
      const __m128 vi3xGACE = _mm_move_ss(vi3x8ACE, vi3xGIKM);
      vi3x8ACE = vi3xGIKM;
      const __m128 vi4xGACE = _mm_move_ss(vi4x8ACE, vi4xGIKM);
      vi4x8ACE = vi4xGIKM;
      const __m128 vi5xGACE = _mm_move_ss(vi5x8ACE, vi5xGIKM);
      vi5x8ACE = vi5xGIKM;
      const __m128 vi6xGACE = _mm_move_ss(vi6x8ACE, vi6xGIKM);
      vi6x8ACE = vi6xGIKM;

      const __m128 vi0xACEG = _mm_shuffle_ps(vi0xGACE, vi0xGACE, _MM_SHUFFLE(0, 3, 2, 1));
      const __m128 vi1xACEG = _mm_shuffle_ps(vi1xGACE, vi1xGACE, _MM_SHUFFLE(0, 3, 2, 1));
      const __m128 vi2xACEG = _mm_shuffle_ps(vi2xGACE, vi2xGACE, _MM_SHUFFLE(0, 3, 2, 1));
      const __m128 vi3xACEG = _mm_shuffle_ps(vi3xGACE, vi3xGACE, _MM_SHUFFLE(0, 3, 2, 1));
      const __m128 vi4xACEG = _mm_shuffle_ps(vi4xGACE, vi4xGACE, _MM_SHUFFLE(0, 3, 2, 1));
      const __m128 vi5xACEG = _mm_shuffle_ps(vi5xGACE, vi5xGACE, _MM_SHUFFLE(0, 3, 2, 1));
      const __m128 vi6xACEG = _mm_shuffle_ps(vi6xGACE, vi6xGACE, _MM_SHUFFLE(0, 3, 2, 1));

      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi0xACEG, vk04));
      vo1p0 = _mm_add_ps(vo1p0, _mm_mul_ps(vi2xACEG, vk04));
      vo0p1 = _mm_add_ps(vo0p1, _mm_mul_ps(vi1xACEG, vk14));
      vo1p1 = _mm_add_ps(vo1p1, _mm_mul_ps(vi3xACEG, vk14));
      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi2xACEG, vk24));
      vo1p0 = _mm_add_ps(vo1p0, _mm_mul_ps(vi4xACEG, vk24));
      vo0p1 = _mm_add_ps(vo0p1, _mm_mul_ps(vi3xACEG, vk34));
      vo1p1 = _mm_add_ps(vo1p1, _mm_mul_ps(vi5xACEG, vk34));
      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi4xACEG, vk44));
      vo1p0 = _mm_add_ps(vo1p0, _mm_mul_ps(vi6xACEG, vk44));

      vo0p0 = _mm_add_ps(vo0p0, vo0p1);
      vo1p0 = _mm_add_ps(vo1p0, vo1p1);

      __m128 vo0 = _mm_max_ps(vo0p0, vmin);
      __m128 vo1 = _mm_max_ps(vo1p0, vmin);

      vo0 = _mm_min_ps(vo0, vmax);
      vo1 = _mm_min_ps(vo1, vmax);

      _mm_storeu_ps(o1, vo1);
      o1 += 4;
      _mm_storeu_ps(o0, vo0);
      o0 += 4;
    }
    // Last block has 1-8 pixels to process.
    assert(w <= 8 * sizeof(float));
    assert(w >= 1 * sizeof(float));
    {
      vi0x8ACE = _mm_and_ps(vi0x8ACE, vmask_even);
      vi0x9BDF = _mm_and_ps(vi0x9BDF, vmask_odd);
      vi1x8ACE = _mm_and_ps(vi1x8ACE, vmask_even);
      vi1x9BDF = _mm_and_ps(vi1x9BDF, vmask_odd);
      vi2x8ACE = _mm_and_ps(vi2x8ACE, vmask_even);
      vi2x9BDF = _mm_and_ps(vi2x9BDF, vmask_odd);
      vi3x8ACE = _mm_and_ps(vi3x8ACE, vmask_even);
      vi3x9BDF = _mm_and_ps(vi3x9BDF, vmask_odd);
      vi4x8ACE = _mm_and_ps(vi4x8ACE, vmask_even);
      vi4x9BDF = _mm_and_ps(vi4x9BDF, vmask_odd);
      vi5x8ACE = _mm_and_ps(vi5x8ACE, vmask_even);
      vi5x9BDF = _mm_and_ps(vi5x9BDF, vmask_odd);
      vi6x8ACE = _mm_and_ps(vi6x8ACE, vmask_even);
      vi6x9BDF = _mm_and_ps(vi6x9BDF, vmask_odd);

      __m128 vo0p0 = _mm_add_ps(vbias, _mm_mul_ps(vi0x8ACE, vk02));
      __m128 vo1p0 = _mm_add_ps(vbias, _mm_mul_ps(vi2x8ACE, vk02));
      __m128 vo0p1 = _mm_mul_ps(vi1x8ACE, vk12);
      __m128 vo1p1 = _mm_mul_ps(vi3x8ACE, vk12);
      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi2x8ACE, vk22));
      vo1p0 = _mm_add_ps(vo1p0, _mm_mul_ps(vi4x8ACE, vk22));
      vo0p1 = _mm_add_ps(vo0p1, _mm_mul_ps(vi3x8ACE, vk32));
      vo1p1 = _mm_add_ps(vo1p1, _mm_mul_ps(vi5x8ACE, vk32));
      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi4x8ACE, vk42));
      vo1p0 = _mm_add_ps(vo1p0, _mm_mul_ps(vi6x8ACE, vk42));

      const __m128 vi0xE8AC = _mm_shuffle_ps(vi0x8ACE, vi0x8ACE, _MM_SHUFFLE(2, 1, 0, 3));
      const __m128 vi1xE8AC = _mm_shuffle_ps(vi1x8ACE, vi1x8ACE, _MM_SHUFFLE(2, 1, 0, 3));
      const __m128 vi2xE8AC = _mm_shuffle_ps(vi2x8ACE, vi2x8ACE, _MM_SHUFFLE(2, 1, 0, 3));
      const __m128 vi3xE8AC = _mm_shuffle_ps(vi3x8ACE, vi3x8ACE, _MM_SHUFFLE(2, 1, 0, 3));
      const __m128 vi4xE8AC = _mm_shuffle_ps(vi4x8ACE, vi4x8ACE, _MM_SHUFFLE(2, 1, 0, 3));
      const __m128 vi5xE8AC = _mm_shuffle_ps(vi5x8ACE, vi5x8ACE, _MM_SHUFFLE(2, 1, 0, 3));
      const __m128 vi6xE8AC = _mm_shuffle_ps(vi6x8ACE, vi6x8ACE, _MM_SHUFFLE(2, 1, 0, 3));

      vo0p1 = _mm_add_ps(vo0p1, _mm_mul_ps(vi0x9BDF, vk03));
      vo1p1 = _mm_add_ps(vo1p1, _mm_mul_ps(vi2x9BDF, vk03));
      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi1x9BDF, vk13));
      vo1p0 = _mm_add_ps(vo1p0, _mm_mul_ps(vi3x9BDF, vk13));
      vo0p1 = _mm_add_ps(vo0p1, _mm_mul_ps(vi2x9BDF, vk23));
      vo1p1 = _mm_add_ps(vo1p1, _mm_mul_ps(vi4x9BDF, vk23));
      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi3x9BDF, vk33));
      vo1p0 = _mm_add_ps(vo1p0, _mm_mul_ps(vi5x9BDF, vk33));
      vo0p1 = _mm_add_ps(vo0p1, _mm_mul_ps(vi4x9BDF, vk43));
      vo1p1 = _mm_add_ps(vo1p1, _mm_mul_ps(vi6x9BDF, vk43));

      const __m128 vi0x68AC = _mm_move_ss(vi0xE8AC, vi0x6024);
      const __m128 vi1x68AC = _mm_move_ss(vi1xE8AC, vi1x6024);
      const __m128 vi2x68AC = _mm_move_ss(vi2xE8AC, vi2x6024);
      const __m128 vi3x68AC = _mm_move_ss(vi3xE8AC, vi3x6024);
      const __m128 vi4x68AC = _mm_move_ss(vi4xE8AC, vi4x6024);
      const __m128 vi5x68AC = _mm_move_ss(vi5xE8AC, vi5x6024);
      const __m128 vi6x68AC = _mm_move_ss(vi6xE8AC, vi6x6024);

      const __m128 vi0xF9BD = _mm_shuffle_ps(vi0x9BDF, vi0x9BDF, _MM_SHUFFLE(2, 1, 0, 3));
      const __m128 vi1xF9BD = _mm_shuffle_ps(vi1x9BDF, vi1x9BDF, _MM_SHUFFLE(2, 1, 0, 3));
      const __m128 vi2xF9BD = _mm_shuffle_ps(vi2x9BDF, vi2x9BDF, _MM_SHUFFLE(2, 1, 0, 3));
      const __m128 vi3xF9BD = _mm_shuffle_ps(vi3x9BDF, vi3x9BDF, _MM_SHUFFLE(2, 1, 0, 3));
      const __m128 vi4xF9BD = _mm_shuffle_ps(vi4x9BDF, vi4x9BDF, _MM_SHUFFLE(2, 1, 0, 3));
      const __m128 vi5xF9BD = _mm_shuffle_ps(vi5x9BDF, vi5x9BDF, _MM_SHUFFLE(2, 1, 0, 3));
      const __m128 vi6xF9BD = _mm_shuffle_ps(vi6x9BDF, vi6x9BDF, _MM_SHUFFLE(2, 1, 0, 3));

      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi0x68AC, vk00));
      vo1p0 = _mm_add_ps(vo1p0, _mm_mul_ps(vi2x68AC, vk00));
      vo0p1 = _mm_add_ps(vo0p1, _mm_mul_ps(vi1x68AC, vk10));
      vo1p1 = _mm_add_ps(vo1p1, _mm_mul_ps(vi3x68AC, vk10));
      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi2x68AC, vk20));
      vo1p0 = _mm_add_ps(vo1p0, _mm_mul_ps(vi4x68AC, vk20));
      vo0p1 = _mm_add_ps(vo0p1, _mm_mul_ps(vi3x68AC, vk30));
      vo1p1 = _mm_add_ps(vo1p1, _mm_mul_ps(vi5x68AC, vk30));
      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi4x68AC, vk40));
      vo1p0 = _mm_add_ps(vo1p0, _mm_mul_ps(vi6x68AC, vk40));

      const __m128 vi0x79BD = _mm_move_ss(vi0xF9BD, vi0x7135);
      const __m128 vi1x79BD = _mm_move_ss(vi1xF9BD, vi1x7135);
      const __m128 vi2x79BD = _mm_move_ss(vi2xF9BD, vi2x7135);
      const __m128 vi3x79BD = _mm_move_ss(vi3xF9BD, vi3x7135);
      const __m128 vi4x79BD = _mm_move_ss(vi4xF9BD, vi4x7135);
      const __m128 vi5x79BD = _mm_move_ss(vi5xF9BD, vi5x7135);
      const __m128 vi6x79BD = _mm_move_ss(vi6xF9BD, vi6x7135);

      vo0p1 = _mm_add_ps(vo0p1, _mm_mul_ps(vi0x79BD, vk01));
      vo1p1 = _mm_add_ps(vo1p1, _mm_mul_ps(vi2x79BD, vk01));
      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi1x79BD, vk11));
      vo1p0 = _mm_add_ps(vo1p0, _mm_mul_ps(vi3x79BD, vk11));
      vo0p1 = _mm_add_ps(vo0p1, _mm_mul_ps(vi2x79BD, vk21));
      vo1p1 = _mm_add_ps(vo1p1, _mm_mul_ps(vi4x79BD, vk21));
      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi3x79BD, vk31));
      vo1p0 = _mm_add_ps(vo1p0, _mm_mul_ps(vi5x79BD, vk31));
      vo0p1 = _mm_add_ps(vo0p1, _mm_mul_ps(vi4x79BD, vk41));
      vo1p1 = _mm_add_ps(vo1p1, _mm_mul_ps(vi6x79BD, vk41));

      const __m128 vzero = _mm_setzero_ps();
      const __m128 vi0xGACE = _mm_move_ss(vi0x8ACE, vzero);
      const __m128 vi1xGACE = _mm_move_ss(vi1x8ACE, vzero);
      const __m128 vi2xGACE = _mm_move_ss(vi2x8ACE, vzero);
      const __m128 vi3xGACE = _mm_move_ss(vi3x8ACE, vzero);
      const __m128 vi4xGACE = _mm_move_ss(vi4x8ACE, vzero);
      const __m128 vi5xGACE = _mm_move_ss(vi5x8ACE, vzero);
      const __m128 vi6xGACE = _mm_move_ss(vi6x8ACE, vzero);

      const __m128 vi0xACEG = _mm_shuffle_ps(vi0xGACE, vi0xGACE, _MM_SHUFFLE(0, 3, 2, 1));
      const __m128 vi1xACEG = _mm_shuffle_ps(vi1xGACE, vi1xGACE, _MM_SHUFFLE(0, 3, 2, 1));
      const __m128 vi2xACEG = _mm_shuffle_ps(vi2xGACE, vi2xGACE, _MM_SHUFFLE(0, 3, 2, 1));
      const __m128 vi3xACEG = _mm_shuffle_ps(vi3xGACE, vi3xGACE, _MM_SHUFFLE(0, 3, 2, 1));
      const __m128 vi4xACEG = _mm_shuffle_ps(vi4xGACE, vi4xGACE, _MM_SHUFFLE(0, 3, 2, 1));
      const __m128 vi5xACEG = _mm_shuffle_ps(vi5xGACE, vi5xGACE, _MM_SHUFFLE(0, 3, 2, 1));
      const __m128 vi6xACEG = _mm_shuffle_ps(vi6xGACE, vi6xGACE, _MM_SHUFFLE(0, 3, 2, 1));

      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi0xACEG, vk04));
      vo1p0 = _mm_add_ps(vo1p0, _mm_mul_ps(vi2xACEG, vk04));
      vo0p1 = _mm_add_ps(vo0p1, _mm_mul_ps(vi1xACEG, vk14));
      vo1p1 = _mm_add_ps(vo1p1, _mm_mul_ps(vi3xACEG, vk14));
      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi2xACEG, vk24));
      vo1p0 = _mm_add_ps(vo1p0, _mm_mul_ps(vi4xACEG, vk24));
      vo0p1 = _mm_add_ps(vo0p1, _mm_mul_ps(vi3xACEG, vk34));
      vo1p1 = _mm_add_ps(vo1p1, _mm_mul_ps(vi5xACEG, vk34));
      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi4xACEG, vk44));
      vo1p0 = _mm_add_ps(vo1p0, _mm_mul_ps(vi6xACEG, vk44));

      vo0p0 = _mm_add_ps(vo0p0, vo0p1);
      vo1p0 = _mm_add_ps(vo1p0, vo1p1);

      __m128 vo0 = _mm_max_ps(vo0p0, vmin);
      __m128 vo1 = _mm_max_ps(vo1p0, vmin);

      vo0 = _mm_min_ps(vo0, vmax);
      vo1 = _mm_min_ps(vo1, vmax);

      size_t w_tmp = (w + 1 * sizeof(float)) / (2 * sizeof(float));
      if XNN_LIKELY(w_tmp >= 4) {
        _mm_storeu_ps(o1, vo1);
        o1 += 4;
        _mm_storeu_ps(o0, vo0);
        o0 += 4;
      } else {
        if (w_tmp & 2) {
          _mm_storel_pi((__m64*) o1, vo1);
          o1 += 2;
          _mm_storel_pi((__m64*) o0, vo0);
          o0 += 2;

          vo0 = _mm_movehl_ps(vo0, vo0);
          vo1 = _mm_movehl_ps(vo1, vo1);
        }
        if (w_tmp & 1) {
          _mm_store_ss(o1, vo1);
          o1 += 1;
          _mm_store_ss(o0, vo0);
          o0 += 1;
        }
      }
    }

    i0 = (const float*) ((uintptr_t) i4 - input_decrement);
    i1 = (const float*) ((uintptr_t) i5 - input_decrement);
    i2 = (const float*) ((uintptr_t) i6 - input_decrement);
    i3 = (const float*) ((uintptr_t) i2 + input_width);
    i4 = (const float*) ((uintptr_t) i3 + input_width);
    i5 = (const float*) ((uintptr_t) i4 + input_width);
    i6 = (const float*) ((uintptr_t) i5 + input_width);

    o0 = o1;
    o1 = (float*) ((uintptr_t) o0 + output_width);

    output_height = doz(output_height, 2);
    padded_input_height = doz(padded_input_height, 4);
  } while (output_height != 0);
}
