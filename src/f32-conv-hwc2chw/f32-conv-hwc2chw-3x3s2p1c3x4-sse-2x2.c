// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <xmmintrin.h>

#include "xnnpack/conv.h"
#include "xnnpack/math.h"


void xnn_f32_conv_hwc2chw_ukernel_3x3s2p1c3x4__sse_2x2(
    size_t input_height,
    size_t input_width,
    size_t output_y_start,
    size_t output_y_end,
    const float* input,
    const float* zero,
    const float* weights,
    float* output,
    size_t input_padding_top,
    size_t output_channels,
    size_t output_height_stride,
    size_t output_channel_stride,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(input_width != 0);
  assert(output_y_end > output_y_start);
  assert(input_padding_top <= 1);
  assert(output_channels != 0);

  const size_t input_height_stride = input_width * 3 /* channels */ * sizeof(float);
  const size_t input_width_increment = round_down_po2(input_width, 4) * 3 /* channels */ * sizeof(float);
  const size_t output_width = (input_width + 1) / 2;
  const size_t output_channel_increment = output_channel_stride * 4 - output_width * sizeof(float);

  // Adjustment for padding processed below
  const float* i0 = (const float*) ((uintptr_t) input + input_height_stride * (output_y_start * 2 - input_padding_top));
  const float* i1 = (const float*) ((uintptr_t) i0 + input_height_stride);
  const float* i2 = (const float*) ((uintptr_t) i1 + input_height_stride);
  const float* i3 = (const float*) ((uintptr_t) i2 + input_height_stride);
  const float* i4 = (const float*) ((uintptr_t) i3 + input_height_stride);
  float* output0 = (float*) ((uintptr_t) output + output_height_stride * output_y_start);
  float* output1 = (float*) ((uintptr_t) output0 + output_height_stride);

  if XNN_UNPREDICTABLE(output_y_start < input_padding_top) {
    i0 = zero;
  }

  const __m128 vmin = _mm_set1_ps(params->sse.min);
  const __m128 vmax = _mm_set1_ps(params->sse.max);
  XNN_FORCE_REALIZATION(vmin);
  XNN_FORCE_REALIZATION(vmax);

  for (size_t output_y = output_y_start; output_y < output_y_end; output_y += 2) {
    const size_t input_y2 = output_y * 2 + 2 - input_padding_top;
    const size_t input_y4 = input_y2 + 2;
    if XNN_UNPREDICTABLE(input_y2 >= input_height) {
      i2 = zero;
    }
    if XNN_UNPREDICTABLE(input_y4 > input_height) {
      i3 = zero;
    }
    if XNN_UNPREDICTABLE(input_y4 >= input_height) {
      i4 = zero;
    }
    if XNN_UNPREDICTABLE(output_y + 2 > output_y_end) {
      output1 = output0;
    }

    const float* w = weights;
    size_t c = output_channels;
    float* o0c0 = output0;
    float* o1c0 = output1;
    float* o0c1 = (float*) ((uintptr_t) o0c0 + output_channel_stride);
    float* o1c1 = (float*) ((uintptr_t) o1c0 + output_channel_stride);
    float* o0c2 = (float*) ((uintptr_t) o0c1 + output_channel_stride);
    float* o1c2 = (float*) ((uintptr_t) o1c1 + output_channel_stride);
    float* o0c3 = (float*) ((uintptr_t) o0c2 + output_channel_stride);
    float* o1c3 = (float*) ((uintptr_t) o1c2 + output_channel_stride);
    do {
      if XNN_UNPREDICTABLE(c < 2) {
        o0c1 = o0c0;
        o1c1 = o1c0;
      }
      if XNN_UNPREDICTABLE(c <= 2) {
        o0c2 = o0c1;
        o1c2 = o1c1;
      }
      if XNN_UNPREDICTABLE(c < 4) {
        o0c3 = o0c2;
        o1c3 = o1c2;
      }

      // viMx0 = ( iM0c2, iM0c1, iM0c0, --- )
      __m128 vi0x0 = _mm_setzero_ps();
      __m128 vi1x0 = _mm_setzero_ps();
      __m128 vi2x0 = _mm_setzero_ps();
      __m128 vi3x0 = _mm_setzero_ps();
      __m128 vi4x0 = _mm_setzero_ps();

      size_t iw = input_width;
      for (; iw >= 4; iw -= 4) {
        __m128 vo0x0 = _mm_load_ps(w);
        __m128 vo1x0 = vo0x0;
        __m128 vo0x1 = vo0x0;
        __m128 vo1x1 = vo0x0;

        const __m128 vk00c0 = _mm_load_ps(w + 4);

        // viMx1 = ( iM2c0, iM1c2, iM1c1, iM1c0 )
        const __m128 vi0x1 = _mm_loadu_ps(i0); i0 += 4;
        const __m128 vi1x1 = _mm_loadu_ps(i1); i1 += 4;
        const __m128 vi2x1 = _mm_loadu_ps(i2); i2 += 4;
        const __m128 vi3x1 = _mm_loadu_ps(i3); i3 += 4;
        const __m128 vi4x1 = _mm_loadu_ps(i4); i4 += 4;

        vo0x0 = _mm_add_ps(vo0x0, _mm_mul_ps(vk00c0, _mm_shuffle_ps(vi0x0, vi0x0, _MM_SHUFFLE(1, 1, 1, 1))));
        vo1x0 = _mm_add_ps(vo1x0, _mm_mul_ps(vk00c0, _mm_shuffle_ps(vi2x0, vi2x0, _MM_SHUFFLE(1, 1, 1, 1))));
        vo0x1 = _mm_add_ps(vo0x1, _mm_mul_ps(vk00c0, _mm_shuffle_ps(vi0x1, vi0x1, _MM_SHUFFLE(3, 3, 3, 3))));
        vo1x1 = _mm_add_ps(vo1x1, _mm_mul_ps(vk00c0, _mm_shuffle_ps(vi2x1, vi2x1, _MM_SHUFFLE(3, 3, 3, 3))));

        const __m128 vk10c0 = _mm_load_ps(w + 8);

        vo0x0 = _mm_add_ps(vo0x0, _mm_mul_ps(vk10c0, _mm_shuffle_ps(vi1x0, vi1x0, _MM_SHUFFLE(1, 1, 1, 1))));
        vo1x0 = _mm_add_ps(vo1x0, _mm_mul_ps(vk10c0, _mm_shuffle_ps(vi3x0, vi3x0, _MM_SHUFFLE(1, 1, 1, 1))));
        vo0x1 = _mm_add_ps(vo0x1, _mm_mul_ps(vk10c0, _mm_shuffle_ps(vi1x1, vi1x1, _MM_SHUFFLE(3, 3, 3, 3))));
        vo1x1 = _mm_add_ps(vo1x1, _mm_mul_ps(vk10c0, _mm_shuffle_ps(vi3x1, vi3x1, _MM_SHUFFLE(3, 3, 3, 3))));

        const __m128 vk20c0 = _mm_load_ps(w + 12);

        vo0x0 = _mm_add_ps(vo0x0, _mm_mul_ps(vk20c0, _mm_shuffle_ps(vi2x0, vi2x0, _MM_SHUFFLE(1, 1, 1, 1))));
        vo1x0 = _mm_add_ps(vo1x0, _mm_mul_ps(vk20c0, _mm_shuffle_ps(vi4x0, vi4x0, _MM_SHUFFLE(1, 1, 1, 1))));
        vo0x1 = _mm_add_ps(vo0x1, _mm_mul_ps(vk20c0, _mm_shuffle_ps(vi2x1, vi2x1, _MM_SHUFFLE(3, 3, 3, 3))));
        vo1x1 = _mm_add_ps(vo1x1, _mm_mul_ps(vk20c0, _mm_shuffle_ps(vi4x1, vi4x1, _MM_SHUFFLE(3, 3, 3, 3))));

        const __m128 vk00c1 = _mm_load_ps(w + 16);

        // viMx2 = ( iM3c1, iM3c0, iM2c2, iM2c1 )
        const __m128 vi0x2 = _mm_loadu_ps(i0); i0 += 4;
        const __m128 vi1x2 = _mm_loadu_ps(i1); i1 += 4;
        const __m128 vi2x2 = _mm_loadu_ps(i2); i2 += 4;
        const __m128 vi3x2 = _mm_loadu_ps(i3); i3 += 4;
        const __m128 vi4x2 = _mm_loadu_ps(i4); i4 += 4;

        vo0x0 = _mm_add_ps(vo0x0, _mm_mul_ps(vk00c1, _mm_shuffle_ps(vi0x0, vi0x0, _MM_SHUFFLE(2, 2, 2, 2))));
        vo1x0 = _mm_add_ps(vo1x0, _mm_mul_ps(vk00c1, _mm_shuffle_ps(vi2x0, vi2x0, _MM_SHUFFLE(2, 2, 2, 2))));
        vo0x1 = _mm_add_ps(vo0x1, _mm_mul_ps(vk00c1, _mm_shuffle_ps(vi0x2, vi0x2, _MM_SHUFFLE(0, 0, 0, 0))));
        vo1x1 = _mm_add_ps(vo1x1, _mm_mul_ps(vk00c1, _mm_shuffle_ps(vi2x2, vi2x2, _MM_SHUFFLE(0, 0, 0, 0))));

        const __m128 vk10c1 = _mm_load_ps(w + 20);

        vo0x0 = _mm_add_ps(vo0x0, _mm_mul_ps(vk10c1, _mm_shuffle_ps(vi1x0, vi1x0, _MM_SHUFFLE(2, 2, 2, 2))));
        vo1x0 = _mm_add_ps(vo1x0, _mm_mul_ps(vk10c1, _mm_shuffle_ps(vi3x0, vi3x0, _MM_SHUFFLE(2, 2, 2, 2))));
        vo0x1 = _mm_add_ps(vo0x1, _mm_mul_ps(vk10c1, _mm_shuffle_ps(vi1x2, vi1x2, _MM_SHUFFLE(0, 0, 0, 0))));
        vo1x1 = _mm_add_ps(vo1x1, _mm_mul_ps(vk10c1, _mm_shuffle_ps(vi3x2, vi3x2, _MM_SHUFFLE(0, 0, 0, 0))));

        const __m128 vk20c1 = _mm_load_ps(w + 24);

        vo0x0 = _mm_add_ps(vo0x0, _mm_mul_ps(vk20c1, _mm_shuffle_ps(vi2x0, vi2x0, _MM_SHUFFLE(2, 2, 2, 2))));
        vo1x0 = _mm_add_ps(vo1x0, _mm_mul_ps(vk20c1, _mm_shuffle_ps(vi4x0, vi4x0, _MM_SHUFFLE(2, 2, 2, 2))));
        vo0x1 = _mm_add_ps(vo0x1, _mm_mul_ps(vk20c1, _mm_shuffle_ps(vi2x2, vi2x2, _MM_SHUFFLE(0, 0, 0, 0))));
        vo1x1 = _mm_add_ps(vo1x1, _mm_mul_ps(vk20c1, _mm_shuffle_ps(vi4x2, vi4x2, _MM_SHUFFLE(0, 0, 0, 0))));

        const __m128 vk00c2 = _mm_load_ps(w + 28);

        vo0x0 = _mm_add_ps(vo0x0, _mm_mul_ps(vk00c2, _mm_shuffle_ps(vi0x0, vi0x0, _MM_SHUFFLE(3, 3, 3, 3))));
        vo1x0 = _mm_add_ps(vo1x0, _mm_mul_ps(vk00c2, _mm_shuffle_ps(vi2x0, vi2x0, _MM_SHUFFLE(3, 3, 3, 3))));
        vo0x1 = _mm_add_ps(vo0x1, _mm_mul_ps(vk00c2, _mm_shuffle_ps(vi0x2, vi0x2, _MM_SHUFFLE(1, 1, 1, 1))));
        vo1x1 = _mm_add_ps(vo1x1, _mm_mul_ps(vk00c2, _mm_shuffle_ps(vi2x2, vi2x2, _MM_SHUFFLE(1, 1, 1, 1))));

        const __m128 vk10c2 = _mm_load_ps(w + 32);

        vo0x0 = _mm_add_ps(vo0x0, _mm_mul_ps(vk10c2, _mm_shuffle_ps(vi1x0, vi1x0, _MM_SHUFFLE(3, 3, 3, 3))));
        vo1x0 = _mm_add_ps(vo1x0, _mm_mul_ps(vk10c2, _mm_shuffle_ps(vi3x0, vi3x0, _MM_SHUFFLE(3, 3, 3, 3))));
        vo0x1 = _mm_add_ps(vo0x1, _mm_mul_ps(vk10c2, _mm_shuffle_ps(vi1x2, vi1x2, _MM_SHUFFLE(1, 1, 1, 1))));
        vo1x1 = _mm_add_ps(vo1x1, _mm_mul_ps(vk10c2, _mm_shuffle_ps(vi3x2, vi3x2, _MM_SHUFFLE(1, 1, 1, 1))));

        const __m128 vk20c2 = _mm_load_ps(w + 36);

        vo0x0 = _mm_add_ps(vo0x0, _mm_mul_ps(vk20c2, _mm_shuffle_ps(vi2x0, vi2x0, _MM_SHUFFLE(3, 3, 3, 3))));
        vo1x0 = _mm_add_ps(vo1x0, _mm_mul_ps(vk20c2, _mm_shuffle_ps(vi4x0, vi4x0, _MM_SHUFFLE(3, 3, 3, 3))));
        vo0x1 = _mm_add_ps(vo0x1, _mm_mul_ps(vk20c2, _mm_shuffle_ps(vi2x2, vi2x2, _MM_SHUFFLE(1, 1, 1, 1))));
        vo1x1 = _mm_add_ps(vo1x1, _mm_mul_ps(vk20c2, _mm_shuffle_ps(vi4x2, vi4x2, _MM_SHUFFLE(1, 1, 1, 1))));

        const __m128 vk01c0 = _mm_load_ps(w + 40);

        vo0x0 = _mm_add_ps(vo0x0, _mm_mul_ps(vk01c0, _mm_shuffle_ps(vi0x1, vi0x1, _MM_SHUFFLE(0, 0, 0, 0))));
        vo1x0 = _mm_add_ps(vo1x0, _mm_mul_ps(vk01c0, _mm_shuffle_ps(vi2x1, vi2x1, _MM_SHUFFLE(0, 0, 0, 0))));
        vo0x1 = _mm_add_ps(vo0x1, _mm_mul_ps(vk01c0, _mm_shuffle_ps(vi0x2, vi0x2, _MM_SHUFFLE(2, 2, 2, 2))));
        vo1x1 = _mm_add_ps(vo1x1, _mm_mul_ps(vk01c0, _mm_shuffle_ps(vi2x2, vi2x2, _MM_SHUFFLE(2, 2, 2, 2))));

        const __m128 vk11c0 = _mm_load_ps(w + 44);

        vo0x0 = _mm_add_ps(vo0x0, _mm_mul_ps(vk11c0, _mm_shuffle_ps(vi1x1, vi1x1, _MM_SHUFFLE(0, 0, 0, 0))));
        vo1x0 = _mm_add_ps(vo1x0, _mm_mul_ps(vk11c0, _mm_shuffle_ps(vi3x1, vi3x1, _MM_SHUFFLE(0, 0, 0, 0))));
        vo0x1 = _mm_add_ps(vo0x1, _mm_mul_ps(vk11c0, _mm_shuffle_ps(vi1x2, vi1x2, _MM_SHUFFLE(2, 2, 2, 2))));
        vo1x1 = _mm_add_ps(vo1x1, _mm_mul_ps(vk11c0, _mm_shuffle_ps(vi3x2, vi3x2, _MM_SHUFFLE(2, 2, 2, 2))));

        const __m128 vk21c0 = _mm_load_ps(w + 48);

        vo0x0 = _mm_add_ps(vo0x0, _mm_mul_ps(vk21c0, _mm_shuffle_ps(vi2x1, vi2x1, _MM_SHUFFLE(0, 0, 0, 0))));
        vo1x0 = _mm_add_ps(vo1x0, _mm_mul_ps(vk21c0, _mm_shuffle_ps(vi4x1, vi4x1, _MM_SHUFFLE(0, 0, 0, 0))));
        vo0x1 = _mm_add_ps(vo0x1, _mm_mul_ps(vk21c0, _mm_shuffle_ps(vi2x2, vi2x2, _MM_SHUFFLE(2, 2, 2, 2))));
        vo1x1 = _mm_add_ps(vo1x1, _mm_mul_ps(vk21c0, _mm_shuffle_ps(vi4x2, vi4x2, _MM_SHUFFLE(2, 2, 2, 2))));

        const __m128 vk01c1 = _mm_load_ps(w + 52);

        vo0x0 = _mm_add_ps(vo0x0, _mm_mul_ps(vk01c1, _mm_shuffle_ps(vi0x1, vi0x1, _MM_SHUFFLE(1, 1, 1, 1))));
        vo1x0 = _mm_add_ps(vo1x0, _mm_mul_ps(vk01c1, _mm_shuffle_ps(vi2x1, vi2x1, _MM_SHUFFLE(1, 1, 1, 1))));
        vo0x1 = _mm_add_ps(vo0x1, _mm_mul_ps(vk01c1, _mm_shuffle_ps(vi0x2, vi0x2, _MM_SHUFFLE(3, 3, 3, 3))));
        vo1x1 = _mm_add_ps(vo1x1, _mm_mul_ps(vk01c1, _mm_shuffle_ps(vi2x2, vi2x2, _MM_SHUFFLE(3, 3, 3, 3))));

        const __m128 vk11c1 = _mm_load_ps(w + 56);

        vo0x0 = _mm_add_ps(vo0x0, _mm_mul_ps(vk11c1, _mm_shuffle_ps(vi1x1, vi1x1, _MM_SHUFFLE(1, 1, 1, 1))));
        vo1x0 = _mm_add_ps(vo1x0, _mm_mul_ps(vk11c1, _mm_shuffle_ps(vi3x1, vi3x1, _MM_SHUFFLE(1, 1, 1, 1))));
        vo0x1 = _mm_add_ps(vo0x1, _mm_mul_ps(vk11c1, _mm_shuffle_ps(vi1x2, vi1x2, _MM_SHUFFLE(3, 3, 3, 3))));
        vo1x1 = _mm_add_ps(vo1x1, _mm_mul_ps(vk11c1, _mm_shuffle_ps(vi3x2, vi3x2, _MM_SHUFFLE(3, 3, 3, 3))));

        const __m128 vk21c1 = _mm_load_ps(w + 60);

        vo0x0 = _mm_add_ps(vo0x0, _mm_mul_ps(vk21c1, _mm_shuffle_ps(vi2x1, vi2x1, _MM_SHUFFLE(1, 1, 1, 1))));
        vo1x0 = _mm_add_ps(vo1x0, _mm_mul_ps(vk21c1, _mm_shuffle_ps(vi4x1, vi4x1, _MM_SHUFFLE(1, 1, 1, 1))));
        vo0x1 = _mm_add_ps(vo0x1, _mm_mul_ps(vk21c1, _mm_shuffle_ps(vi2x2, vi2x2, _MM_SHUFFLE(3, 3, 3, 3))));
        vo1x1 = _mm_add_ps(vo1x1, _mm_mul_ps(vk21c1, _mm_shuffle_ps(vi4x2, vi4x2, _MM_SHUFFLE(3, 3, 3, 3))));

        const __m128 vk01c2 = _mm_load_ps(w + 64);

        // viMx3 = ( iM4c2, iM4c1, iM4c0, iM3c2 )
        const __m128 vi0x3 = _mm_loadu_ps(i0); i0 += 4;
        const __m128 vi1x3 = _mm_loadu_ps(i1); i1 += 4;
        const __m128 vi2x3 = _mm_loadu_ps(i2); i2 += 4;
        const __m128 vi3x3 = _mm_loadu_ps(i3); i3 += 4;
        const __m128 vi4x3 = _mm_loadu_ps(i4); i4 += 4;

        vo0x0 = _mm_add_ps(vo0x0, _mm_mul_ps(vk01c2, _mm_shuffle_ps(vi0x1, vi0x1, _MM_SHUFFLE(2, 2, 2, 2))));
        vo1x0 = _mm_add_ps(vo1x0, _mm_mul_ps(vk01c2, _mm_shuffle_ps(vi2x1, vi2x1, _MM_SHUFFLE(2, 2, 2, 2))));
        vo0x1 = _mm_add_ps(vo0x1, _mm_mul_ps(vk01c2, _mm_shuffle_ps(vi0x3, vi0x3, _MM_SHUFFLE(0, 0, 0, 0))));
        vo1x1 = _mm_add_ps(vo1x1, _mm_mul_ps(vk01c2, _mm_shuffle_ps(vi2x3, vi2x3, _MM_SHUFFLE(0, 0, 0, 0))));

        const __m128 vk11c2 = _mm_load_ps(w + 68);

        vo0x0 = _mm_add_ps(vo0x0, _mm_mul_ps(vk11c2, _mm_shuffle_ps(vi1x1, vi1x1, _MM_SHUFFLE(2, 2, 2, 2))));
        vo1x0 = _mm_add_ps(vo1x0, _mm_mul_ps(vk11c2, _mm_shuffle_ps(vi3x1, vi3x1, _MM_SHUFFLE(2, 2, 2, 2))));
        vo0x1 = _mm_add_ps(vo0x1, _mm_mul_ps(vk11c2, _mm_shuffle_ps(vi1x3, vi1x3, _MM_SHUFFLE(0, 0, 0, 0))));
        vo1x1 = _mm_add_ps(vo1x1, _mm_mul_ps(vk11c2, _mm_shuffle_ps(vi3x3, vi3x3, _MM_SHUFFLE(0, 0, 0, 0))));

        const __m128 vk21c2 = _mm_load_ps(w + 72);

        vo0x0 = _mm_add_ps(vo0x0, _mm_mul_ps(vk21c2, _mm_shuffle_ps(vi2x1, vi2x1, _MM_SHUFFLE(2, 2, 2, 2))));
        vo1x0 = _mm_add_ps(vo1x0, _mm_mul_ps(vk21c2, _mm_shuffle_ps(vi4x1, vi4x1, _MM_SHUFFLE(2, 2, 2, 2))));
        vo0x1 = _mm_add_ps(vo0x1, _mm_mul_ps(vk21c2, _mm_shuffle_ps(vi2x3, vi2x3, _MM_SHUFFLE(0, 0, 0, 0))));
        vo1x1 = _mm_add_ps(vo1x1, _mm_mul_ps(vk21c2, _mm_shuffle_ps(vi4x3, vi4x3, _MM_SHUFFLE(0, 0, 0, 0))));

        const __m128 vk02c0 = _mm_load_ps(w + 76);

        vo0x0 = _mm_add_ps(vo0x0, _mm_mul_ps(vk02c0, _mm_shuffle_ps(vi0x1, vi0x1, _MM_SHUFFLE(3, 3, 3, 3))));
        vo1x0 = _mm_add_ps(vo1x0, _mm_mul_ps(vk02c0, _mm_shuffle_ps(vi2x1, vi2x1, _MM_SHUFFLE(3, 3, 3, 3))));
        vo0x1 = _mm_add_ps(vo0x1, _mm_mul_ps(vk02c0, _mm_shuffle_ps(vi0x3, vi0x3, _MM_SHUFFLE(1, 1, 1, 1))));
        vo1x1 = _mm_add_ps(vo1x1, _mm_mul_ps(vk02c0, _mm_shuffle_ps(vi2x3, vi2x3, _MM_SHUFFLE(1, 1, 1, 1))));

        const __m128 vk12c0 = _mm_load_ps(w + 80);

        vo0x0 = _mm_add_ps(vo0x0, _mm_mul_ps(vk12c0, _mm_shuffle_ps(vi1x1, vi1x1, _MM_SHUFFLE(3, 3, 3, 3))));
        vo1x0 = _mm_add_ps(vo1x0, _mm_mul_ps(vk12c0, _mm_shuffle_ps(vi3x1, vi3x1, _MM_SHUFFLE(3, 3, 3, 3))));
        vo0x1 = _mm_add_ps(vo0x1, _mm_mul_ps(vk12c0, _mm_shuffle_ps(vi1x3, vi1x3, _MM_SHUFFLE(1, 1, 1, 1))));
        vo1x1 = _mm_add_ps(vo1x1, _mm_mul_ps(vk12c0, _mm_shuffle_ps(vi3x3, vi3x3, _MM_SHUFFLE(1, 1, 1, 1))));

        const __m128 vk22c0 = _mm_load_ps(w + 84);

        vo0x0 = _mm_add_ps(vo0x0, _mm_mul_ps(vk22c0, _mm_shuffle_ps(vi2x1, vi2x1, _MM_SHUFFLE(3, 3, 3, 3))));
        vo1x0 = _mm_add_ps(vo1x0, _mm_mul_ps(vk22c0, _mm_shuffle_ps(vi4x1, vi4x1, _MM_SHUFFLE(3, 3, 3, 3))));
        vo0x1 = _mm_add_ps(vo0x1, _mm_mul_ps(vk22c0, _mm_shuffle_ps(vi2x3, vi2x3, _MM_SHUFFLE(1, 1, 1, 1))));
        vo1x1 = _mm_add_ps(vo1x1, _mm_mul_ps(vk22c0, _mm_shuffle_ps(vi4x3, vi4x3, _MM_SHUFFLE(1, 1, 1, 1))));

        const __m128 vk02c1 = _mm_load_ps(w + 88);

        vo0x0 = _mm_add_ps(vo0x0, _mm_mul_ps(vk02c1, _mm_shuffle_ps(vi0x2, vi0x2, _MM_SHUFFLE(0, 0, 0, 0))));
        vo1x0 = _mm_add_ps(vo1x0, _mm_mul_ps(vk02c1, _mm_shuffle_ps(vi2x2, vi2x2, _MM_SHUFFLE(0, 0, 0, 0))));
        vo0x1 = _mm_add_ps(vo0x1, _mm_mul_ps(vk02c1, _mm_shuffle_ps(vi0x3, vi0x3, _MM_SHUFFLE(2, 2, 2, 2))));
        vo1x1 = _mm_add_ps(vo1x1, _mm_mul_ps(vk02c1, _mm_shuffle_ps(vi2x3, vi2x3, _MM_SHUFFLE(2, 2, 2, 2))));

        const __m128 vk12c1 = _mm_load_ps(w + 92);

        vo0x0 = _mm_add_ps(vo0x0, _mm_mul_ps(vk12c1, _mm_shuffle_ps(vi1x2, vi1x2, _MM_SHUFFLE(0, 0, 0, 0))));
        vo1x0 = _mm_add_ps(vo1x0, _mm_mul_ps(vk12c1, _mm_shuffle_ps(vi3x2, vi3x2, _MM_SHUFFLE(0, 0, 0, 0))));
        vo0x1 = _mm_add_ps(vo0x1, _mm_mul_ps(vk12c1, _mm_shuffle_ps(vi1x3, vi1x3, _MM_SHUFFLE(2, 2, 2, 2))));
        vo1x1 = _mm_add_ps(vo1x1, _mm_mul_ps(vk12c1, _mm_shuffle_ps(vi3x3, vi3x3, _MM_SHUFFLE(2, 2, 2, 2))));

        const __m128 vk22c1 = _mm_load_ps(w + 96);

        vo0x0 = _mm_add_ps(vo0x0, _mm_mul_ps(vk22c1, _mm_shuffle_ps(vi2x2, vi2x2, _MM_SHUFFLE(0, 0, 0, 0))));
        vo1x0 = _mm_add_ps(vo1x0, _mm_mul_ps(vk22c1, _mm_shuffle_ps(vi4x2, vi4x2, _MM_SHUFFLE(0, 0, 0, 0))));
        vo0x1 = _mm_add_ps(vo0x1, _mm_mul_ps(vk22c1, _mm_shuffle_ps(vi2x3, vi2x3, _MM_SHUFFLE(2, 2, 2, 2))));
        vo1x1 = _mm_add_ps(vo1x1, _mm_mul_ps(vk22c1, _mm_shuffle_ps(vi4x3, vi4x3, _MM_SHUFFLE(2, 2, 2, 2))));

        const __m128 vk02c2 = _mm_load_ps(w + 100);

        vo0x0 = _mm_add_ps(vo0x0, _mm_mul_ps(vk02c2, _mm_shuffle_ps(vi0x2, vi0x2, _MM_SHUFFLE(1, 1, 1, 1))));
        vo1x0 = _mm_add_ps(vo1x0, _mm_mul_ps(vk02c2, _mm_shuffle_ps(vi2x2, vi2x2, _MM_SHUFFLE(1, 1, 1, 1))));
        vo0x1 = _mm_add_ps(vo0x1, _mm_mul_ps(vk02c2, _mm_shuffle_ps(vi0x3, vi0x3, _MM_SHUFFLE(3, 3, 3, 3))));
        vo1x1 = _mm_add_ps(vo1x1, _mm_mul_ps(vk02c2, _mm_shuffle_ps(vi2x3, vi2x3, _MM_SHUFFLE(3, 3, 3, 3))));

        const __m128 vk12c2 = _mm_load_ps(w + 104);

        vo0x0 = _mm_add_ps(vo0x0, _mm_mul_ps(vk12c2, _mm_shuffle_ps(vi1x2, vi1x2, _MM_SHUFFLE(1, 1, 1, 1))));
        vo1x0 = _mm_add_ps(vo1x0, _mm_mul_ps(vk12c2, _mm_shuffle_ps(vi3x2, vi3x2, _MM_SHUFFLE(1, 1, 1, 1))));
        vo0x1 = _mm_add_ps(vo0x1, _mm_mul_ps(vk12c2, _mm_shuffle_ps(vi1x3, vi1x3, _MM_SHUFFLE(3, 3, 3, 3))));
        vo1x1 = _mm_add_ps(vo1x1, _mm_mul_ps(vk12c2, _mm_shuffle_ps(vi3x3, vi3x3, _MM_SHUFFLE(3, 3, 3, 3))));

        const __m128 vk22c2 = _mm_load_ps(w + 108);

        vo0x0 = _mm_add_ps(vo0x0, _mm_mul_ps(vk22c2, _mm_shuffle_ps(vi2x2, vi2x2, _MM_SHUFFLE(1, 1, 1, 1))));
        vo1x0 = _mm_add_ps(vo1x0, _mm_mul_ps(vk22c2, _mm_shuffle_ps(vi4x2, vi4x2, _MM_SHUFFLE(1, 1, 1, 1))));
        vo0x1 = _mm_add_ps(vo0x1, _mm_mul_ps(vk22c2, _mm_shuffle_ps(vi2x3, vi2x3, _MM_SHUFFLE(3, 3, 3, 3))));
        vo1x1 = _mm_add_ps(vo1x1, _mm_mul_ps(vk22c2, _mm_shuffle_ps(vi4x3, vi4x3, _MM_SHUFFLE(3, 3, 3, 3))));

        vi0x0 = vi0x3;
        vi1x0 = vi1x3;
        vi2x0 = vi2x3;
        vi3x0 = vi3x3;
        vi4x0 = vi4x3;

        vo0x0 = _mm_max_ps(vo0x0, vmin);
        vo1x0 = _mm_max_ps(vo1x0, vmin);
        vo0x1 = _mm_max_ps(vo0x1, vmin);
        vo1x1 = _mm_max_ps(vo1x1, vmin);

        vo0x0 = _mm_min_ps(vo0x0, vmax);
        vo1x0 = _mm_min_ps(vo1x0, vmax);
        vo0x1 = _mm_min_ps(vo0x1, vmax);
        vo1x1 = _mm_min_ps(vo1x1, vmax);

        const __m128 vo0c01 = _mm_unpacklo_ps(vo0x0, vo0x1);
        const __m128 vo0c23 = _mm_unpackhi_ps(vo0x0, vo0x1);
        const __m128 vo1c01 = _mm_unpacklo_ps(vo1x0, vo1x1);
        const __m128 vo1c23 = _mm_unpackhi_ps(vo1x0, vo1x1);

        // Always 2+ output width elements remaining
        _mm_storel_pi((__m64 *)o1c0, vo1c01); o1c0 += 2;
        _mm_storel_pi((__m64 *)o1c1, _mm_shuffle_ps(vo1c01, vo1c01, _MM_SHUFFLE(3, 2, 3, 2))); o1c1 += 2;
        _mm_storel_pi((__m64 *)o1c2, vo1c23); o1c2 += 2;
        _mm_storel_pi((__m64 *)o1c3, _mm_shuffle_ps(vo1c23, vo1c23, _MM_SHUFFLE(3, 2, 3, 2))); o1c3 += 2;

        _mm_storel_pi((__m64 *)o0c0, vo0c01); o0c0 += 2;
        _mm_storel_pi((__m64 *)o0c1, _mm_shuffle_ps(vo0c01, vo0c01, _MM_SHUFFLE(3, 2, 3, 2))); o0c1 += 2;
        _mm_storel_pi((__m64 *)o0c2, vo0c23); o0c2 += 2;
        _mm_storel_pi((__m64 *)o0c3, _mm_shuffle_ps(vo0c23, vo0c23, _MM_SHUFFLE(3, 2, 3, 2))); o0c3 += 2;
      }
      assert(iw < 4);
      if XNN_UNLIKELY(iw != 0) {
        __m128 vo0x0 = _mm_load_ps(w);
        __m128 vo1x0 = vo0x0;
        __m128 vo0x1 = vo0x0;
        __m128 vo1x1 = vo0x0;

        const __m128 vk00c0 = _mm_load_ps(w + 4);

        // viMx1 = ( iM2c0, iM1c2, iM1c1, iM1c0 )
        __m128 vi0x1 = _mm_loadu_ps(i0);
        __m128 vi1x1 = _mm_loadu_ps(i1);
        __m128 vi2x1 = _mm_loadu_ps(i2);
        __m128 vi3x1 = _mm_loadu_ps(i3);
        __m128 vi4x1 = _mm_loadu_ps(i4);

        vo0x0 = _mm_add_ps(vo0x0, _mm_mul_ps(vk00c0, _mm_shuffle_ps(vi0x0, vi0x0, _MM_SHUFFLE(1, 1, 1, 1))));
        vo1x0 = _mm_add_ps(vo1x0, _mm_mul_ps(vk00c0, _mm_shuffle_ps(vi2x0, vi2x0, _MM_SHUFFLE(1, 1, 1, 1))));
        if (iw > 2) {
          vo0x1 = _mm_add_ps(vo0x1, _mm_mul_ps(vk00c0, _mm_shuffle_ps(vi0x1, vi0x1, _MM_SHUFFLE(3, 3, 3, 3))));
          vo1x1 = _mm_add_ps(vo1x1, _mm_mul_ps(vk00c0, _mm_shuffle_ps(vi2x1, vi2x1, _MM_SHUFFLE(3, 3, 3, 3))));
        }

        const __m128 vk10c0 = _mm_load_ps(w + 8);

        vo0x0 = _mm_add_ps(vo0x0, _mm_mul_ps(vk10c0, _mm_shuffle_ps(vi1x0, vi1x0, _MM_SHUFFLE(1, 1, 1, 1))));
        vo1x0 = _mm_add_ps(vo1x0, _mm_mul_ps(vk10c0, _mm_shuffle_ps(vi3x0, vi3x0, _MM_SHUFFLE(1, 1, 1, 1))));
        if (iw > 2) {
          vo0x1 = _mm_add_ps(vo0x1, _mm_mul_ps(vk10c0, _mm_shuffle_ps(vi1x1, vi1x1, _MM_SHUFFLE(3, 3, 3, 3))));
          vo1x1 = _mm_add_ps(vo1x1, _mm_mul_ps(vk10c0, _mm_shuffle_ps(vi3x1, vi3x1, _MM_SHUFFLE(3, 3, 3, 3))));
        }

        const __m128 vk20c0 = _mm_load_ps(w + 12);

        vo0x0 = _mm_add_ps(vo0x0, _mm_mul_ps(vk20c0, _mm_shuffle_ps(vi2x0, vi2x0, _MM_SHUFFLE(1, 1, 1, 1))));
        vo1x0 = _mm_add_ps(vo1x0, _mm_mul_ps(vk20c0, _mm_shuffle_ps(vi4x0, vi4x0, _MM_SHUFFLE(1, 1, 1, 1))));
        if (iw > 2) {
          vo0x1 = _mm_add_ps(vo0x1, _mm_mul_ps(vk20c0, _mm_shuffle_ps(vi2x1, vi2x1, _MM_SHUFFLE(3, 3, 3, 3))));
          vo1x1 = _mm_add_ps(vo1x1, _mm_mul_ps(vk20c0, _mm_shuffle_ps(vi4x1, vi4x1, _MM_SHUFFLE(3, 3, 3, 3))));
        }

        const __m128 vk00c1 = _mm_load_ps(w + 16);

        __m128 vi0x2 = _mm_setzero_ps();
        __m128 vi1x2 = _mm_setzero_ps();
        __m128 vi2x2 = _mm_setzero_ps();
        __m128 vi3x2 = _mm_setzero_ps();
        __m128 vi4x2 = _mm_setzero_ps();
        if (iw >= 2) {
          // viMx2 = ( iM3c1, iM3c0, iM2c2, iM2c1 )
          vi0x2 = _mm_loadu_ps(i0 + 4);
          vi1x2 = _mm_loadu_ps(i1 + 4);
          vi2x2 = _mm_loadu_ps(i2 + 4);
          vi3x2 = _mm_loadu_ps(i3 + 4);
          vi4x2 = _mm_loadu_ps(i4 + 4);
        }

        vo0x0 = _mm_add_ps(vo0x0, _mm_mul_ps(vk00c1, _mm_shuffle_ps(vi0x0, vi0x0, _MM_SHUFFLE(2, 2, 2, 2))));
        vo1x0 = _mm_add_ps(vo1x0, _mm_mul_ps(vk00c1, _mm_shuffle_ps(vi2x0, vi2x0, _MM_SHUFFLE(2, 2, 2, 2))));
        vo0x1 = _mm_add_ps(vo0x1, _mm_mul_ps(vk00c1, _mm_shuffle_ps(vi0x2, vi0x2, _MM_SHUFFLE(0, 0, 0, 0))));
        vo1x1 = _mm_add_ps(vo1x1, _mm_mul_ps(vk00c1, _mm_shuffle_ps(vi2x2, vi2x2, _MM_SHUFFLE(0, 0, 0, 0))));

        const __m128 vk10c1 = _mm_load_ps(w + 20);

        vo0x0 = _mm_add_ps(vo0x0, _mm_mul_ps(vk10c1, _mm_shuffle_ps(vi1x0, vi1x0, _MM_SHUFFLE(2, 2, 2, 2))));
        vo1x0 = _mm_add_ps(vo1x0, _mm_mul_ps(vk10c1, _mm_shuffle_ps(vi3x0, vi3x0, _MM_SHUFFLE(2, 2, 2, 2))));
        vo0x1 = _mm_add_ps(vo0x1, _mm_mul_ps(vk10c1, _mm_shuffle_ps(vi1x2, vi1x2, _MM_SHUFFLE(0, 0, 0, 0))));
        vo1x1 = _mm_add_ps(vo1x1, _mm_mul_ps(vk10c1, _mm_shuffle_ps(vi3x2, vi3x2, _MM_SHUFFLE(0, 0, 0, 0))));

        const __m128 vk20c1 = _mm_load_ps(w + 24);

        vo0x0 = _mm_add_ps(vo0x0, _mm_mul_ps(vk20c1, _mm_shuffle_ps(vi2x0, vi2x0, _MM_SHUFFLE(2, 2, 2, 2))));
        vo1x0 = _mm_add_ps(vo1x0, _mm_mul_ps(vk20c1, _mm_shuffle_ps(vi4x0, vi4x0, _MM_SHUFFLE(2, 2, 2, 2))));
        vo0x1 = _mm_add_ps(vo0x1, _mm_mul_ps(vk20c1, _mm_shuffle_ps(vi2x2, vi2x2, _MM_SHUFFLE(0, 0, 0, 0))));
        vo1x1 = _mm_add_ps(vo1x1, _mm_mul_ps(vk20c1, _mm_shuffle_ps(vi4x2, vi4x2, _MM_SHUFFLE(0, 0, 0, 0))));

        const __m128 vk00c2 = _mm_load_ps(w + 28);

        vo0x0 = _mm_add_ps(vo0x0, _mm_mul_ps(vk00c2, _mm_shuffle_ps(vi0x0, vi0x0, _MM_SHUFFLE(3, 3, 3, 3))));
        vo1x0 = _mm_add_ps(vo1x0, _mm_mul_ps(vk00c2, _mm_shuffle_ps(vi2x0, vi2x0, _MM_SHUFFLE(3, 3, 3, 3))));
        vo0x1 = _mm_add_ps(vo0x1, _mm_mul_ps(vk00c2, _mm_shuffle_ps(vi0x2, vi0x2, _MM_SHUFFLE(1, 1, 1, 1))));
        vo1x1 = _mm_add_ps(vo1x1, _mm_mul_ps(vk00c2, _mm_shuffle_ps(vi2x2, vi2x2, _MM_SHUFFLE(1, 1, 1, 1))));

        const __m128 vk10c2 = _mm_load_ps(w + 32);

        vo0x0 = _mm_add_ps(vo0x0, _mm_mul_ps(vk10c2, _mm_shuffle_ps(vi1x0, vi1x0, _MM_SHUFFLE(3, 3, 3, 3))));
        vo1x0 = _mm_add_ps(vo1x0, _mm_mul_ps(vk10c2, _mm_shuffle_ps(vi3x0, vi3x0, _MM_SHUFFLE(3, 3, 3, 3))));
        vo0x1 = _mm_add_ps(vo0x1, _mm_mul_ps(vk10c2, _mm_shuffle_ps(vi1x2, vi1x2, _MM_SHUFFLE(1, 1, 1, 1))));
        vo1x1 = _mm_add_ps(vo1x1, _mm_mul_ps(vk10c2, _mm_shuffle_ps(vi3x2, vi3x2, _MM_SHUFFLE(1, 1, 1, 1))));

        const __m128 vk20c2 = _mm_load_ps(w + 36);

        vo0x0 = _mm_add_ps(vo0x0, _mm_mul_ps(vk20c2, _mm_shuffle_ps(vi2x0, vi2x0, _MM_SHUFFLE(3, 3, 3, 3))));
        vo1x0 = _mm_add_ps(vo1x0, _mm_mul_ps(vk20c2, _mm_shuffle_ps(vi4x0, vi4x0, _MM_SHUFFLE(3, 3, 3, 3))));
        vo0x1 = _mm_add_ps(vo0x1, _mm_mul_ps(vk20c2, _mm_shuffle_ps(vi2x2, vi2x2, _MM_SHUFFLE(1, 1, 1, 1))));
        vo1x1 = _mm_add_ps(vo1x1, _mm_mul_ps(vk20c2, _mm_shuffle_ps(vi4x2, vi4x2, _MM_SHUFFLE(1, 1, 1, 1))));

        const __m128 vk01c0 = _mm_load_ps(w + 40);

        vo0x0 = _mm_add_ps(vo0x0, _mm_mul_ps(vk01c0, _mm_shuffle_ps(vi0x1, vi0x1, _MM_SHUFFLE(0, 0, 0, 0))));
        vo1x0 = _mm_add_ps(vo1x0, _mm_mul_ps(vk01c0, _mm_shuffle_ps(vi2x1, vi2x1, _MM_SHUFFLE(0, 0, 0, 0))));
        if (iw > 2) {
          vo0x1 = _mm_add_ps(vo0x1, _mm_mul_ps(vk01c0, _mm_shuffle_ps(vi0x2, vi0x2, _MM_SHUFFLE(2, 2, 2, 2))));
          vo1x1 = _mm_add_ps(vo1x1, _mm_mul_ps(vk01c0, _mm_shuffle_ps(vi2x2, vi2x2, _MM_SHUFFLE(2, 2, 2, 2))));
        }

        const __m128 vk11c0 = _mm_load_ps(w + 44);

        vo0x0 = _mm_add_ps(vo0x0, _mm_mul_ps(vk11c0, _mm_shuffle_ps(vi1x1, vi1x1, _MM_SHUFFLE(0, 0, 0, 0))));
        vo1x0 = _mm_add_ps(vo1x0, _mm_mul_ps(vk11c0, _mm_shuffle_ps(vi3x1, vi3x1, _MM_SHUFFLE(0, 0, 0, 0))));
        if (iw > 2) {
          vo0x1 = _mm_add_ps(vo0x1, _mm_mul_ps(vk11c0, _mm_shuffle_ps(vi1x2, vi1x2, _MM_SHUFFLE(2, 2, 2, 2))));
          vo1x1 = _mm_add_ps(vo1x1, _mm_mul_ps(vk11c0, _mm_shuffle_ps(vi3x2, vi3x2, _MM_SHUFFLE(2, 2, 2, 2))));
        }

        const __m128 vk21c0 = _mm_load_ps(w + 48);

        vo0x0 = _mm_add_ps(vo0x0, _mm_mul_ps(vk21c0, _mm_shuffle_ps(vi2x1, vi2x1, _MM_SHUFFLE(0, 0, 0, 0))));
        vo1x0 = _mm_add_ps(vo1x0, _mm_mul_ps(vk21c0, _mm_shuffle_ps(vi4x1, vi4x1, _MM_SHUFFLE(0, 0, 0, 0))));
        if (iw > 2) {
          vo0x1 = _mm_add_ps(vo0x1, _mm_mul_ps(vk21c0, _mm_shuffle_ps(vi2x2, vi2x2, _MM_SHUFFLE(2, 2, 2, 2))));
          vo1x1 = _mm_add_ps(vo1x1, _mm_mul_ps(vk21c0, _mm_shuffle_ps(vi4x2, vi4x2, _MM_SHUFFLE(2, 2, 2, 2))));
        }

        const __m128 vk01c1 = _mm_load_ps(w + 52);

        vo0x0 = _mm_add_ps(vo0x0, _mm_mul_ps(vk01c1, _mm_shuffle_ps(vi0x1, vi0x1, _MM_SHUFFLE(1, 1, 1, 1))));
        vo1x0 = _mm_add_ps(vo1x0, _mm_mul_ps(vk01c1, _mm_shuffle_ps(vi2x1, vi2x1, _MM_SHUFFLE(1, 1, 1, 1))));
        if (iw > 2) {
          vo0x1 = _mm_add_ps(vo0x1, _mm_mul_ps(vk01c1, _mm_shuffle_ps(vi0x2, vi0x2, _MM_SHUFFLE(3, 3, 3, 3))));
          vo1x1 = _mm_add_ps(vo1x1, _mm_mul_ps(vk01c1, _mm_shuffle_ps(vi2x2, vi2x2, _MM_SHUFFLE(3, 3, 3, 3))));
        }

        const __m128 vk11c1 = _mm_load_ps(w + 56);

        vo0x0 = _mm_add_ps(vo0x0, _mm_mul_ps(vk11c1, _mm_shuffle_ps(vi1x1, vi1x1, _MM_SHUFFLE(1, 1, 1, 1))));
        vo1x0 = _mm_add_ps(vo1x0, _mm_mul_ps(vk11c1, _mm_shuffle_ps(vi3x1, vi3x1, _MM_SHUFFLE(1, 1, 1, 1))));
        if (iw > 2) {
          vo0x1 = _mm_add_ps(vo0x1, _mm_mul_ps(vk11c1, _mm_shuffle_ps(vi1x2, vi1x2, _MM_SHUFFLE(3, 3, 3, 3))));
          vo1x1 = _mm_add_ps(vo1x1, _mm_mul_ps(vk11c1, _mm_shuffle_ps(vi3x2, vi3x2, _MM_SHUFFLE(3, 3, 3, 3))));
        }

        const __m128 vk21c1 = _mm_load_ps(w + 60);

        vo0x0 = _mm_add_ps(vo0x0, _mm_mul_ps(vk21c1, _mm_shuffle_ps(vi2x1, vi2x1, _MM_SHUFFLE(1, 1, 1, 1))));
        vo1x0 = _mm_add_ps(vo1x0, _mm_mul_ps(vk21c1, _mm_shuffle_ps(vi4x1, vi4x1, _MM_SHUFFLE(1, 1, 1, 1))));
        if (iw > 2) {
          vo0x1 = _mm_add_ps(vo0x1, _mm_mul_ps(vk21c1, _mm_shuffle_ps(vi2x2, vi2x2, _MM_SHUFFLE(3, 3, 3, 3))));
          vo1x1 = _mm_add_ps(vo1x1, _mm_mul_ps(vk21c1, _mm_shuffle_ps(vi4x2, vi4x2, _MM_SHUFFLE(3, 3, 3, 3))));
        }

        const __m128 vk01c2 = _mm_load_ps(w + 64);

        __m128 vi0x3 = _mm_setzero_ps();
        __m128 vi1x3 = _mm_setzero_ps();
        __m128 vi2x3 = _mm_setzero_ps();
        __m128 vi3x3 = _mm_setzero_ps();
        __m128 vi4x3 = _mm_setzero_ps();
        if (iw > 2) {
          // viMx3 = ( 0.0, 0.0, 0.0, iM3c2 )
          vi0x3 = _mm_load_ss(i0 + 8);
          vi1x3 = _mm_load_ss(i1 + 8);
          vi2x3 = _mm_load_ss(i2 + 8);
          vi3x3 = _mm_load_ss(i3 + 8);
          vi4x3 = _mm_load_ss(i4 + 8);
        }

        vo0x0 = _mm_add_ps(vo0x0, _mm_mul_ps(vk01c2, _mm_shuffle_ps(vi0x1, vi0x1, _MM_SHUFFLE(2, 2, 2, 2))));
        vo1x0 = _mm_add_ps(vo1x0, _mm_mul_ps(vk01c2, _mm_shuffle_ps(vi2x1, vi2x1, _MM_SHUFFLE(2, 2, 2, 2))));
        vo0x1 = _mm_add_ps(vo0x1, _mm_mul_ps(vk01c2, _mm_shuffle_ps(vi0x3, vi0x3, _MM_SHUFFLE(0, 0, 0, 0))));
        vo1x1 = _mm_add_ps(vo1x1, _mm_mul_ps(vk01c2, _mm_shuffle_ps(vi2x3, vi2x3, _MM_SHUFFLE(0, 0, 0, 0))));

        const __m128 vk11c2 = _mm_load_ps(w + 68);

        vo0x0 = _mm_add_ps(vo0x0, _mm_mul_ps(vk11c2, _mm_shuffle_ps(vi1x1, vi1x1, _MM_SHUFFLE(2, 2, 2, 2))));
        vo1x0 = _mm_add_ps(vo1x0, _mm_mul_ps(vk11c2, _mm_shuffle_ps(vi3x1, vi3x1, _MM_SHUFFLE(2, 2, 2, 2))));
        vo0x1 = _mm_add_ps(vo0x1, _mm_mul_ps(vk11c2, _mm_shuffle_ps(vi1x3, vi1x3, _MM_SHUFFLE(0, 0, 0, 0))));
        vo1x1 = _mm_add_ps(vo1x1, _mm_mul_ps(vk11c2, _mm_shuffle_ps(vi3x3, vi3x3, _MM_SHUFFLE(0, 0, 0, 0))));

        const __m128 vk21c2 = _mm_load_ps(w + 72);

        vo0x0 = _mm_add_ps(vo0x0, _mm_mul_ps(vk21c2, _mm_shuffle_ps(vi2x1, vi2x1, _MM_SHUFFLE(2, 2, 2, 2))));
        vo1x0 = _mm_add_ps(vo1x0, _mm_mul_ps(vk21c2, _mm_shuffle_ps(vi4x1, vi4x1, _MM_SHUFFLE(2, 2, 2, 2))));
        vo0x1 = _mm_add_ps(vo0x1, _mm_mul_ps(vk21c2, _mm_shuffle_ps(vi2x3, vi2x3, _MM_SHUFFLE(0, 0, 0, 0))));
        vo1x1 = _mm_add_ps(vo1x1, _mm_mul_ps(vk21c2, _mm_shuffle_ps(vi4x3, vi4x3, _MM_SHUFFLE(0, 0, 0, 0))));

        if (iw >= 2) {
          const __m128 vk02c0 = _mm_load_ps(w + 76);

          vo0x0 = _mm_add_ps(vo0x0, _mm_mul_ps(vk02c0, _mm_shuffle_ps(vi0x1, vi0x1, _MM_SHUFFLE(3, 3, 3, 3))));
          vo1x0 = _mm_add_ps(vo1x0, _mm_mul_ps(vk02c0, _mm_shuffle_ps(vi2x1, vi2x1, _MM_SHUFFLE(3, 3, 3, 3))));

          const __m128 vk12c0 = _mm_load_ps(w + 80);

          vo0x0 = _mm_add_ps(vo0x0, _mm_mul_ps(vk12c0, _mm_shuffle_ps(vi1x1, vi1x1, _MM_SHUFFLE(3, 3, 3, 3))));
          vo1x0 = _mm_add_ps(vo1x0, _mm_mul_ps(vk12c0, _mm_shuffle_ps(vi3x1, vi3x1, _MM_SHUFFLE(3, 3, 3, 3))));

          const __m128 vk22c0 = _mm_load_ps(w + 84);

          vo0x0 = _mm_add_ps(vo0x0, _mm_mul_ps(vk22c0, _mm_shuffle_ps(vi2x1, vi2x1, _MM_SHUFFLE(3, 3, 3, 3))));
          vo1x0 = _mm_add_ps(vo1x0, _mm_mul_ps(vk22c0, _mm_shuffle_ps(vi4x1, vi4x1, _MM_SHUFFLE(3, 3, 3, 3))));

          const __m128 vk02c1 = _mm_load_ps(w + 88);

          vo0x0 = _mm_add_ps(vo0x0, _mm_mul_ps(vk02c1, _mm_shuffle_ps(vi0x2, vi0x2, _MM_SHUFFLE(0, 0, 0, 0))));
          vo1x0 = _mm_add_ps(vo1x0, _mm_mul_ps(vk02c1, _mm_shuffle_ps(vi2x2, vi2x2, _MM_SHUFFLE(0, 0, 0, 0))));

          const __m128 vk12c1 = _mm_load_ps(w + 92);

          vo0x0 = _mm_add_ps(vo0x0, _mm_mul_ps(vk12c1, _mm_shuffle_ps(vi1x2, vi1x2, _MM_SHUFFLE(0, 0, 0, 0))));
          vo1x0 = _mm_add_ps(vo1x0, _mm_mul_ps(vk12c1, _mm_shuffle_ps(vi3x2, vi3x2, _MM_SHUFFLE(0, 0, 0, 0))));

          const __m128 vk22c1 = _mm_load_ps(w + 96);

          vo0x0 = _mm_add_ps(vo0x0, _mm_mul_ps(vk22c1, _mm_shuffle_ps(vi2x2, vi2x2, _MM_SHUFFLE(0, 0, 0, 0))));
          vo1x0 = _mm_add_ps(vo1x0, _mm_mul_ps(vk22c1, _mm_shuffle_ps(vi4x2, vi4x2, _MM_SHUFFLE(0, 0, 0, 0))));

          const __m128 vk02c2 = _mm_load_ps(w + 100);

          vo0x0 = _mm_add_ps(vo0x0, _mm_mul_ps(vk02c2, _mm_shuffle_ps(vi0x2, vi0x2, _MM_SHUFFLE(1, 1, 1, 1))));
          vo1x0 = _mm_add_ps(vo1x0, _mm_mul_ps(vk02c2, _mm_shuffle_ps(vi2x2, vi2x2, _MM_SHUFFLE(1, 1, 1, 1))));

          const __m128 vk12c2 = _mm_load_ps(w + 104);

          vo0x0 = _mm_add_ps(vo0x0, _mm_mul_ps(vk12c2, _mm_shuffle_ps(vi1x2, vi1x2, _MM_SHUFFLE(1, 1, 1, 1))));
          vo1x0 = _mm_add_ps(vo1x0, _mm_mul_ps(vk12c2, _mm_shuffle_ps(vi3x2, vi3x2, _MM_SHUFFLE(1, 1, 1, 1))));

          const __m128 vk22c2 = _mm_load_ps(w + 108);

          vo0x0 = _mm_add_ps(vo0x0, _mm_mul_ps(vk22c2, _mm_shuffle_ps(vi2x2, vi2x2, _MM_SHUFFLE(1, 1, 1, 1))));
          vo1x0 = _mm_add_ps(vo1x0, _mm_mul_ps(vk22c2, _mm_shuffle_ps(vi4x2, vi4x2, _MM_SHUFFLE(1, 1, 1, 1))));
        }

        vo0x0 = _mm_max_ps(vo0x0, vmin);
        vo1x0 = _mm_max_ps(vo1x0, vmin);
        vo0x1 = _mm_max_ps(vo0x1, vmin);
        vo1x1 = _mm_max_ps(vo1x1, vmin);

        vo0x0 = _mm_min_ps(vo0x0, vmax);
        vo1x0 = _mm_min_ps(vo1x0, vmax);
        vo0x1 = _mm_min_ps(vo0x1, vmax);
        vo1x1 = _mm_min_ps(vo1x1, vmax);

        if (iw == 3) {
          // Exactly 2 output width elements remaining
          const __m128 vo0c01 = _mm_unpacklo_ps(vo0x0, vo0x1);
          const __m128 vo0c23 = _mm_unpackhi_ps(vo0x0, vo0x1);
          const __m128 vo1c01 = _mm_unpacklo_ps(vo1x0, vo1x1);
          const __m128 vo1c23 = _mm_unpackhi_ps(vo1x0, vo1x1);

          _mm_storel_pi((__m64 *)o1c0, vo1c01); o1c0 += 2;
          _mm_storel_pi((__m64 *)o1c1, _mm_shuffle_ps(vo1c01, vo1c01, _MM_SHUFFLE(3, 2, 3, 2))); o1c1 += 2;
          _mm_storel_pi((__m64 *)o1c2, vo1c23); o1c2 += 2;
          _mm_storel_pi((__m64 *)o1c3, _mm_shuffle_ps(vo1c23, vo1c23, _MM_SHUFFLE(3, 2, 3, 2))); o1c3 += 2;

          _mm_storel_pi((__m64 *)o0c0, vo0c01); o0c0 += 2;
          _mm_storel_pi((__m64 *)o0c1, _mm_shuffle_ps(vo0c01, vo0c01, _MM_SHUFFLE(3, 2, 3, 2))); o0c1 += 2;
          _mm_storel_pi((__m64 *)o0c2, vo0c23); o0c2 += 2;
          _mm_storel_pi((__m64 *)o0c3, _mm_shuffle_ps(vo0c23, vo0c23, _MM_SHUFFLE(3, 2, 3, 2))); o0c3 += 2;
        } else {
          // Exactly 1 output width element remaining

          _mm_store_ss(o1c0, _mm_shuffle_ps(vo1x0, vo1x0, _MM_SHUFFLE(0, 0, 0, 0))); o1c0 += 1;
          _mm_store_ss(o1c1, _mm_shuffle_ps(vo1x0, vo1x0, _MM_SHUFFLE(1, 1, 1, 1))); o1c1 += 1;
          _mm_store_ss(o1c2, _mm_shuffle_ps(vo1x0, vo1x0, _MM_SHUFFLE(2, 2, 2, 2))); o1c2 += 1;
          _mm_store_ss(o1c3, _mm_shuffle_ps(vo1x0, vo1x0, _MM_SHUFFLE(3, 3, 3, 3))); o1c3 += 1;

          _mm_store_ss(o0c0, _mm_shuffle_ps(vo0x0, vo0x0, _MM_SHUFFLE(0, 0, 0, 0))); o0c0 += 1;
          _mm_store_ss(o0c1, _mm_shuffle_ps(vo0x0, vo0x0, _MM_SHUFFLE(1, 1, 1, 1))); o0c1 += 1;
          _mm_store_ss(o0c2, _mm_shuffle_ps(vo0x0, vo0x0, _MM_SHUFFLE(2, 2, 2, 2))); o0c2 += 1;
          _mm_store_ss(o0c3, _mm_shuffle_ps(vo0x0, vo0x0, _MM_SHUFFLE(3, 3, 3, 3))); o0c3 += 1;
        }
      }
      // Move output pointers back to the position of the first pixel in a row,
      // and forward to the next block of output channels.
      o0c0 = (float*) ((uintptr_t) o0c0 + output_channel_increment);
      o0c1 = (float*) ((uintptr_t) o0c1 + output_channel_increment);
      o0c2 = (float*) ((uintptr_t) o0c2 + output_channel_increment);
      o0c3 = (float*) ((uintptr_t) o0c3 + output_channel_increment);
      o1c0 = (float*) ((uintptr_t) o1c0 + output_channel_increment);
      o1c1 = (float*) ((uintptr_t) o1c1 + output_channel_increment);
      o1c2 = (float*) ((uintptr_t) o1c2 + output_channel_increment);
      o1c3 = (float*) ((uintptr_t) o1c3 + output_channel_increment);
      // Revert input pointers to the position of the first pixel in a row
      i0 = (const float*) ((uintptr_t) i0 - input_width_increment);
      i1 = (const float*) ((uintptr_t) i1 - input_width_increment);
      i2 = (const float*) ((uintptr_t) i2 - input_width_increment);
      i3 = (const float*) ((uintptr_t) i3 - input_width_increment);
      i4 = (const float*) ((uintptr_t) i4 - input_width_increment);
      // Move to the block of weights for the next 4 output channels
      w += 112;
      c = doz(c, 4);
    } while (c != 0);
    // Move output pointers forward to the next two rows
    output0 = (float*) ((uintptr_t) output1 + output_height_stride);
    output1 = (float*) ((uintptr_t) output0 + output_height_stride);
    // Move input pointers forward to the next four rows
    i0 = i4;
    i1 = (const float*) ((uintptr_t) i0 + input_height_stride);
    i2 = (const float*) ((uintptr_t) i1 + input_height_stride);
    i3 = (const float*) ((uintptr_t) i2 + input_height_stride);
    i4 = (const float*) ((uintptr_t) i3 + input_height_stride);
  }
}
