// Auto-generated file. Do not edit!
//   Template: src/f32-dwconv2d-chw/3x3p1-sse.c.in
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


void xnn_f32_dwconv2d_chw_ukernel_3x3p1__sse_6x4(
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

    // vi0x3012 = ( vi02, vi01, vi{M}0, vi{M}3 )
    __m128 vi0x3012 = _mm_setzero_ps();
    // vi1x3012 = ( vi12, vi11, vi{M}0, vi{M}3 )
    __m128 vi1x3012 = _mm_setzero_ps();
    // vi2x3012 = ( vi22, vi21, vi{M}0, vi{M}3 )
    __m128 vi2x3012 = _mm_setzero_ps();
    // vi3x3012 = ( vi32, vi31, vi{M}0, vi{M}3 )
    __m128 vi3x3012 = _mm_setzero_ps();
    // vi4x3012 = ( vi42, vi41, vi{M}0, vi{M}3 )
    __m128 vi4x3012 = _mm_setzero_ps();
    // vi5x3012 = ( vi52, vi51, vi{M}0, vi{M}3 )
    __m128 vi5x3012 = _mm_setzero_ps();
    // vi6x3012 = ( vi62, vi61, vi{M}0, vi{M}3 )
    __m128 vi6x3012 = _mm_setzero_ps();
    // vi7x3012 = ( vi72, vi71, vi{M}0, vi{M}3 )
    __m128 vi7x3012 = _mm_setzero_ps();

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
      // vi0x89AB = ( vi0B, vi0A, vi09, vi08 )
      const __m128 vi0x89AB = _mm_loadu_ps(i0);
      i0 += 4;
      // vi1x89AB = ( vi1B, vi1A, vi19, vi18 )
      const __m128 vi1x89AB = _mm_loadu_ps(i1);
      i1 += 4;
      // vi2x89AB = ( vi2B, vi2A, vi29, vi28 )
      const __m128 vi2x89AB = _mm_loadu_ps(i2);
      i2 += 4;
      // vi3x89AB = ( vi3B, vi3A, vi39, vi38 )
      const __m128 vi3x89AB = _mm_loadu_ps(i3);
      i3 += 4;
      // vi4x89AB = ( vi4B, vi4A, vi49, vi48 )
      const __m128 vi4x89AB = _mm_loadu_ps(i4);
      i4 += 4;
      // vi5x89AB = ( vi5B, vi5A, vi59, vi58 )
      const __m128 vi5x89AB = _mm_loadu_ps(i5);
      i5 += 4;
      // vi6x89AB = ( vi6B, vi6A, vi69, vi68 )
      const __m128 vi6x89AB = _mm_loadu_ps(i6);
      i6 += 4;
      // vi7x89AB = ( vi7B, vi7A, vi79, vi78 )
      const __m128 vi7x89AB = _mm_loadu_ps(i7);
      i7 += 4;

      // vi0x7456 = ( vi06, vi05, vi04, vi07 )
      const __m128 vi0x7456 = _mm_shuffle_ps(vi0x4567, vi0x4567, _MM_SHUFFLE(2, 1, 0, 3));
      // vi1x7456 = ( vi16, vi15, vi14, vi17 )
      const __m128 vi1x7456 = _mm_shuffle_ps(vi1x4567, vi1x4567, _MM_SHUFFLE(2, 1, 0, 3));
      // vi2x7456 = ( vi26, vi25, vi24, vi27 )
      const __m128 vi2x7456 = _mm_shuffle_ps(vi2x4567, vi2x4567, _MM_SHUFFLE(2, 1, 0, 3));
      // vi3x7456 = ( vi36, vi35, vi34, vi37 )
      const __m128 vi3x7456 = _mm_shuffle_ps(vi3x4567, vi3x4567, _MM_SHUFFLE(2, 1, 0, 3));
      // vi4x7456 = ( vi46, vi45, vi44, vi47 )
      const __m128 vi4x7456 = _mm_shuffle_ps(vi4x4567, vi4x4567, _MM_SHUFFLE(2, 1, 0, 3));
      // vi5x7456 = ( vi56, vi55, vi54, vi57 )
      const __m128 vi5x7456 = _mm_shuffle_ps(vi5x4567, vi5x4567, _MM_SHUFFLE(2, 1, 0, 3));
      // vi6x7456 = ( vi66, vi65, vi64, vi67 )
      const __m128 vi6x7456 = _mm_shuffle_ps(vi6x4567, vi6x4567, _MM_SHUFFLE(2, 1, 0, 3));
      // vi7x7456 = ( vi76, vi75, vi74, vi77 )
      const __m128 vi7x7456 = _mm_shuffle_ps(vi7x4567, vi7x4567, _MM_SHUFFLE(2, 1, 0, 3));

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

      // vi0x3456 = ( vi06, vi05, vi04, vi03 )
      const __m128 vi0x3456 = _mm_move_ss(vi0x7456, vi0x3012);
      // vi1x3456 = ( vi16, vi15, vi14, vi13 )
      const __m128 vi1x3456 = _mm_move_ss(vi1x7456, vi1x3012);
      // vi2x3456 = ( vi26, vi25, vi24, vi23 )
      const __m128 vi2x3456 = _mm_move_ss(vi2x7456, vi2x3012);
      // vi3x3456 = ( vi36, vi35, vi34, vi33 )
      const __m128 vi3x3456 = _mm_move_ss(vi3x7456, vi3x3012);
      // vi4x3456 = ( vi46, vi45, vi44, vi43 )
      const __m128 vi4x3456 = _mm_move_ss(vi4x7456, vi4x3012);
      // vi5x3456 = ( vi56, vi55, vi54, vi53 )
      const __m128 vi5x3456 = _mm_move_ss(vi5x7456, vi5x3012);
      // vi6x3456 = ( vi66, vi65, vi64, vi63 )
      const __m128 vi6x3456 = _mm_move_ss(vi6x7456, vi6x3012);
      // vi7x3456 = ( vi76, vi75, vi74, vi73 )
      const __m128 vi7x3456 = _mm_move_ss(vi7x7456, vi7x3012);

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

      vi0x3012 = vi0x7456;
      vi1x3012 = vi1x7456;
      vi2x3012 = vi2x7456;
      vi3x3012 = vi3x7456;
      vi4x3012 = vi4x7456;
      vi5x3012 = vi5x7456;
      vi6x3012 = vi6x7456;
      vi7x3012 = vi7x7456;

      // vi0x8567 = ( vi07, vi06, vi05, vi08 )
      const __m128 vi0x8567 = _mm_move_ss(vi0x4567, vi0x89AB);
      // vi1x8567 = ( vi17, vi16, vi15, vi18 )
      const __m128 vi1x8567 = _mm_move_ss(vi1x4567, vi1x89AB);
      // vi2x8567 = ( vi27, vi26, vi25, vi28 )
      const __m128 vi2x8567 = _mm_move_ss(vi2x4567, vi2x89AB);
      // vi3x8567 = ( vi37, vi36, vi35, vi38 )
      const __m128 vi3x8567 = _mm_move_ss(vi3x4567, vi3x89AB);
      // vi4x8567 = ( vi47, vi46, vi45, vi48 )
      const __m128 vi4x8567 = _mm_move_ss(vi4x4567, vi4x89AB);
      // vi5x8567 = ( vi57, vi56, vi55, vi58 )
      const __m128 vi5x8567 = _mm_move_ss(vi5x4567, vi5x89AB);
      // vi6x8567 = ( vi67, vi66, vi65, vi68 )
      const __m128 vi6x8567 = _mm_move_ss(vi6x4567, vi6x89AB);
      // vi7x8567 = ( vi77, vi76, vi75, vi78 )
      const __m128 vi7x8567 = _mm_move_ss(vi7x4567, vi7x89AB);

      // vi0x5678 = ( vi08, vi07, vi06, vi05 )
      const __m128 vi0x5678 = _mm_shuffle_ps(vi0x8567, vi0x8567, _MM_SHUFFLE(0, 3, 2, 1));
      // vi1x5678 = ( vi18, vi17, vi16, vi15 )
      const __m128 vi1x5678 = _mm_shuffle_ps(vi1x8567, vi1x8567, _MM_SHUFFLE(0, 3, 2, 1));
      // vi2x5678 = ( vi28, vi27, vi26, vi25 )
      const __m128 vi2x5678 = _mm_shuffle_ps(vi2x8567, vi2x8567, _MM_SHUFFLE(0, 3, 2, 1));
      // vi3x5678 = ( vi38, vi37, vi36, vi35 )
      const __m128 vi3x5678 = _mm_shuffle_ps(vi3x8567, vi3x8567, _MM_SHUFFLE(0, 3, 2, 1));
      // vi4x5678 = ( vi48, vi47, vi46, vi45 )
      const __m128 vi4x5678 = _mm_shuffle_ps(vi4x8567, vi4x8567, _MM_SHUFFLE(0, 3, 2, 1));
      // vi5x5678 = ( vi58, vi57, vi56, vi55 )
      const __m128 vi5x5678 = _mm_shuffle_ps(vi5x8567, vi5x8567, _MM_SHUFFLE(0, 3, 2, 1));
      // vi6x5678 = ( vi68, vi67, vi66, vi65 )
      const __m128 vi6x5678 = _mm_shuffle_ps(vi6x8567, vi6x8567, _MM_SHUFFLE(0, 3, 2, 1));
      // vi7x5678 = ( vi78, vi77, vi76, vi75 )
      const __m128 vi7x5678 = _mm_shuffle_ps(vi7x8567, vi7x8567, _MM_SHUFFLE(0, 3, 2, 1));

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

      // vi0x7456 = ( vi06, vi05, vi04, vi07 )
      const __m128 vi0x7456 = _mm_shuffle_ps(vi0x4567, vi0x4567, _MM_SHUFFLE(2, 1, 0, 3));
      // vi1x7456 = ( vi16, vi15, vi14, vi17 )
      const __m128 vi1x7456 = _mm_shuffle_ps(vi1x4567, vi1x4567, _MM_SHUFFLE(2, 1, 0, 3));
      // vi2x7456 = ( vi26, vi25, vi24, vi27 )
      const __m128 vi2x7456 = _mm_shuffle_ps(vi2x4567, vi2x4567, _MM_SHUFFLE(2, 1, 0, 3));
      // vi3x7456 = ( vi36, vi35, vi34, vi37 )
      const __m128 vi3x7456 = _mm_shuffle_ps(vi3x4567, vi3x4567, _MM_SHUFFLE(2, 1, 0, 3));
      // vi4x7456 = ( vi46, vi45, vi44, vi47 )
      const __m128 vi4x7456 = _mm_shuffle_ps(vi4x4567, vi4x4567, _MM_SHUFFLE(2, 1, 0, 3));
      // vi5x7456 = ( vi56, vi55, vi54, vi57 )
      const __m128 vi5x7456 = _mm_shuffle_ps(vi5x4567, vi5x4567, _MM_SHUFFLE(2, 1, 0, 3));
      // vi6x7456 = ( vi66, vi65, vi64, vi67 )
      const __m128 vi6x7456 = _mm_shuffle_ps(vi6x4567, vi6x4567, _MM_SHUFFLE(2, 1, 0, 3));
      // vi7x7456 = ( vi76, vi75, vi74, vi77 )
      const __m128 vi7x7456 = _mm_shuffle_ps(vi7x4567, vi7x4567, _MM_SHUFFLE(2, 1, 0, 3));

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

      // vi0x3456 = ( vi06, vi05, vi04, vi03 )
      const __m128 vi0x3456 = _mm_move_ss(vi0x7456, vi0x3012);
      // vi1x3456 = ( vi16, vi15, vi14, vi13 )
      const __m128 vi1x3456 = _mm_move_ss(vi1x7456, vi1x3012);
      // vi2x3456 = ( vi26, vi25, vi24, vi23 )
      const __m128 vi2x3456 = _mm_move_ss(vi2x7456, vi2x3012);
      // vi3x3456 = ( vi36, vi35, vi34, vi33 )
      const __m128 vi3x3456 = _mm_move_ss(vi3x7456, vi3x3012);
      // vi4x3456 = ( vi46, vi45, vi44, vi43 )
      const __m128 vi4x3456 = _mm_move_ss(vi4x7456, vi4x3012);
      // vi5x3456 = ( vi56, vi55, vi54, vi53 )
      const __m128 vi5x3456 = _mm_move_ss(vi5x7456, vi5x3012);
      // vi6x3456 = ( vi66, vi65, vi64, vi63 )
      const __m128 vi6x3456 = _mm_move_ss(vi6x7456, vi6x3012);
      // vi7x3456 = ( vi76, vi75, vi74, vi73 )
      const __m128 vi7x3456 = _mm_move_ss(vi7x7456, vi7x3012);

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

      const __m128 vzero = _mm_setzero_ps();
      // vi0x8567 = ( vi07, vi06, vi05, 0.0 )
      const __m128 vi0x8567 = _mm_move_ss(vi0x4567, vzero);
      // vi1x8567 = ( vi17, vi16, vi15, 0.0 )
      const __m128 vi1x8567 = _mm_move_ss(vi1x4567, vzero);
      // vi2x8567 = ( vi27, vi26, vi25, 0.0 )
      const __m128 vi2x8567 = _mm_move_ss(vi2x4567, vzero);
      // vi3x8567 = ( vi37, vi36, vi35, 0.0 )
      const __m128 vi3x8567 = _mm_move_ss(vi3x4567, vzero);
      // vi4x8567 = ( vi47, vi46, vi45, 0.0 )
      const __m128 vi4x8567 = _mm_move_ss(vi4x4567, vzero);
      // vi5x8567 = ( vi57, vi56, vi55, 0.0 )
      const __m128 vi5x8567 = _mm_move_ss(vi5x4567, vzero);
      // vi6x8567 = ( vi67, vi66, vi65, 0.0 )
      const __m128 vi6x8567 = _mm_move_ss(vi6x4567, vzero);
      // vi7x8567 = ( vi77, vi76, vi75, 0.0 )
      const __m128 vi7x8567 = _mm_move_ss(vi7x4567, vzero);

      // vi0x5678 = ( vi08, vi07, vi06, vi05 )
      const __m128 vi0x5678 = _mm_shuffle_ps(vi0x8567, vi0x8567, _MM_SHUFFLE(0, 3, 2, 1));
      // vi1x5678 = ( vi18, vi17, vi16, vi15 )
      const __m128 vi1x5678 = _mm_shuffle_ps(vi1x8567, vi1x8567, _MM_SHUFFLE(0, 3, 2, 1));
      // vi2x5678 = ( vi28, vi27, vi26, vi25 )
      const __m128 vi2x5678 = _mm_shuffle_ps(vi2x8567, vi2x8567, _MM_SHUFFLE(0, 3, 2, 1));
      // vi3x5678 = ( vi38, vi37, vi36, vi35 )
      const __m128 vi3x5678 = _mm_shuffle_ps(vi3x8567, vi3x8567, _MM_SHUFFLE(0, 3, 2, 1));
      // vi4x5678 = ( vi48, vi47, vi46, vi45 )
      const __m128 vi4x5678 = _mm_shuffle_ps(vi4x8567, vi4x8567, _MM_SHUFFLE(0, 3, 2, 1));
      // vi5x5678 = ( vi58, vi57, vi56, vi55 )
      const __m128 vi5x5678 = _mm_shuffle_ps(vi5x8567, vi5x8567, _MM_SHUFFLE(0, 3, 2, 1));
      // vi6x5678 = ( vi68, vi67, vi66, vi65 )
      const __m128 vi6x5678 = _mm_shuffle_ps(vi6x8567, vi6x8567, _MM_SHUFFLE(0, 3, 2, 1));
      // vi7x5678 = ( vi78, vi77, vi76, vi75 )
      const __m128 vi7x5678 = _mm_shuffle_ps(vi7x8567, vi7x8567, _MM_SHUFFLE(0, 3, 2, 1));

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
