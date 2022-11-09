// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <xmmintrin.h>

#include <xnnpack/conv.h>
#include <xnnpack/math.h>


void xnn_f32_conv_hwc2chw_ukernel_3x3s2p1c3x4__sse_1x1(
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
  const size_t input_width_decrement = round_down_po2(input_width, 2) * 3 /* channels */ * sizeof(float);
  const size_t output_width = (input_width + 1) / 2;
  const size_t output_channel_increment = output_channel_stride * 4 - output_width * sizeof(float);

  // Adjustment for padding processed below
  const float* i0 = (const float*) ((uintptr_t) input + input_height_stride * (output_y_start * 2 - input_padding_top));
  const float* i1 = (const float*) ((uintptr_t) i0 + input_height_stride);
  const float* i2 = (const float*) ((uintptr_t) i1 + input_height_stride);
  float* output0 = (float*) ((uintptr_t) output + output_height_stride * output_y_start);

  if XNN_UNPREDICTABLE(output_y_start < input_padding_top) {
    i0 = zero;
  }

  const __m128 vmin = _mm_load_ps(params->sse.min);
  const __m128 vmax = _mm_load_ps(params->sse.max);

  for (size_t output_y = output_y_start; output_y < output_y_end; output_y += 1) {
    const size_t input_y2 = output_y * 2 + 2 - input_padding_top;
    if XNN_UNPREDICTABLE(input_y2 >= input_height) {
      i2 = zero;
    }

    const float* w = weights;
    size_t c = output_channels;
    float* o0c0 = output0;
    float* o0c1 = (float*) ((uintptr_t) o0c0 + output_channel_stride);
    float* o0c2 = (float*) ((uintptr_t) o0c1 + output_channel_stride);
    float* o0c3 = (float*) ((uintptr_t) o0c2 + output_channel_stride);
    do {
      if XNN_UNPREDICTABLE(c < 2) {
        o0c1 = o0c0;
      }
      if XNN_UNPREDICTABLE(c <= 2) {
        o0c2 = o0c1;
      }
      if XNN_UNPREDICTABLE(c < 4) {
        o0c3 = o0c2;
      }

      // Left edge padding
      __m128 vi00c0 = _mm_setzero_ps();
      __m128 vi00c1 = _mm_setzero_ps();
      __m128 vi00c2 = _mm_setzero_ps();
      __m128 vi10c0 = _mm_setzero_ps();
      __m128 vi10c1 = _mm_setzero_ps();
      __m128 vi10c2 = _mm_setzero_ps();
      __m128 vi20c0 = _mm_setzero_ps();
      __m128 vi20c1 = _mm_setzero_ps();
      __m128 vi20c2 = _mm_setzero_ps();

      size_t iw = input_width;
      for (; iw >= 2; iw -= 2) {
        __m128 voc0123 = _mm_loadu_ps(w);

        const __m128 vk00c0x0123 = _mm_load_ps(w + 4);
        voc0123 = _mm_add_ps(voc0123, _mm_mul_ps(vk00c0x0123, vi00c0));

        const __m128 vk10c0x0123 = _mm_load_ps(w + 8);
        voc0123 = _mm_add_ps(voc0123, _mm_mul_ps(vk10c0x0123, vi10c0));

        const __m128 vk20c0x0123 = _mm_load_ps(w + 12);
        voc0123 = _mm_add_ps(voc0123, _mm_mul_ps(vk20c0x0123, vi20c0));

        const __m128 vk00c1x0123 = _mm_load_ps(w + 16);
        voc0123 = _mm_add_ps(voc0123, _mm_mul_ps(vk00c1x0123, vi00c1));

        const __m128 vk10c1x0123 = _mm_load_ps(w + 20);
        voc0123 = _mm_add_ps(voc0123, _mm_mul_ps(vk10c1x0123, vi10c1));

        const __m128 vk20c1x0123 = _mm_load_ps(w + 24);
        voc0123 = _mm_add_ps(voc0123, _mm_mul_ps(vk20c1x0123, vi20c1));

        const __m128 vk00c2x0123 = _mm_load_ps(w + 28);
        voc0123 = _mm_add_ps(voc0123, _mm_mul_ps(vk00c2x0123, vi00c2));

        const __m128 vk10c2x0123 = _mm_load_ps(w + 32);
        voc0123 = _mm_add_ps(voc0123, _mm_mul_ps(vk10c2x0123, vi10c2));

        const __m128 vk20c2x0123 = _mm_load_ps(w + 36);
        voc0123 = _mm_add_ps(voc0123, _mm_mul_ps(vk20c2x0123, vi20c2));

        const __m128 vk01c0x0123 = _mm_load_ps(w + 40);
        const __m128 vi01c0 = _mm_load1_ps(i0);
        voc0123 = _mm_add_ps(voc0123, _mm_mul_ps(vk01c0x0123, vi01c0));

        const __m128 vk11c0x0123 = _mm_load_ps(w + 44);
        const __m128 vi11c0 = _mm_load1_ps(i1);
        voc0123 = _mm_add_ps(voc0123, _mm_mul_ps(vk11c0x0123, vi11c0));

        const __m128 vk21c0x0123 = _mm_load_ps(w + 48);
        const __m128 vi21c0 = _mm_load1_ps(i2);
        voc0123 = _mm_add_ps(voc0123, _mm_mul_ps(vk21c0x0123, vi21c0));

        const __m128 vk01c1x0123 = _mm_load_ps(w + 52);
        const __m128 vi01c1 = _mm_load1_ps(i0 + 1);
        voc0123 = _mm_add_ps(voc0123, _mm_mul_ps(vk01c1x0123, vi01c1));

        const __m128 vk11c1x0123 = _mm_load_ps(w + 56);
        const __m128 vi11c1 = _mm_load1_ps(i1 + 1);
        voc0123 = _mm_add_ps(voc0123, _mm_mul_ps(vk11c1x0123, vi11c1));

        const __m128 vk21c1x0123 = _mm_load_ps(w + 60);
        const __m128 vi21c1 = _mm_load1_ps(i2 + 1);
        voc0123 = _mm_add_ps(voc0123, _mm_mul_ps(vk21c1x0123, vi21c1));

        const __m128 vk01c2x0123 = _mm_load_ps(w + 64);
        const __m128 vi01c2 = _mm_load1_ps(i0 + 2);
        voc0123 = _mm_add_ps(voc0123, _mm_mul_ps(vk01c2x0123, vi01c2));

        const __m128 vk11c2x0123 = _mm_load_ps(w + 68);
        const __m128 vi11c2 = _mm_load1_ps(i1 + 2);
        voc0123 = _mm_add_ps(voc0123, _mm_mul_ps(vk11c2x0123, vi11c2));

        const __m128 vk21c2x0123 = _mm_load_ps(w + 72);
        const __m128 vi21c2 = _mm_load1_ps(i2 + 2);
        voc0123 = _mm_add_ps(voc0123, _mm_mul_ps(vk21c2x0123, vi21c2));

        const __m128 vk02c0x0123 = _mm_load_ps(w + 76);
        const __m128 vi02c0 = _mm_load1_ps(i0 + 3);
        voc0123 = _mm_add_ps(voc0123, _mm_mul_ps(vk02c0x0123, vi02c0));

        const __m128 vk12c0x0123 = _mm_load_ps(w + 80);
        const __m128 vi12c0 = _mm_load1_ps(i1 + 3);
        voc0123 = _mm_add_ps(voc0123, _mm_mul_ps(vk12c0x0123, vi12c0));

        const __m128 vk22c0x0123 = _mm_load_ps(w + 84);
        const __m128 vi22c0 = _mm_load1_ps(i2 + 3);
        voc0123 = _mm_add_ps(voc0123, _mm_mul_ps(vk22c0x0123, vi22c0));

        vi00c0 = vi02c0;
        vi10c0 = vi12c0;
        vi20c0 = vi22c0;

        const __m128 vk02c1x0123 = _mm_load_ps(w + 88);
        const __m128 vi02c1 = _mm_load1_ps(i0 + 4);
        voc0123 = _mm_add_ps(voc0123, _mm_mul_ps(vk02c1x0123, vi02c1));

        const __m128 vk12c1x0123 = _mm_load_ps(w + 92);
        const __m128 vi12c1 = _mm_load1_ps(i1 + 4);
        voc0123 = _mm_add_ps(voc0123, _mm_mul_ps(vk12c1x0123, vi12c1));

        const __m128 vk22c1x0123 = _mm_load_ps(w + 96);
        const __m128 vi22c1 = _mm_load1_ps(i2 + 4);
        voc0123 = _mm_add_ps(voc0123, _mm_mul_ps(vk22c1x0123, vi22c1));

        vi00c1 = vi02c1;
        vi10c1 = vi12c1;
        vi20c1 = vi22c1;

        const __m128 vk02c2x0123 = _mm_load_ps(w + 100);
        const __m128 vi02c2 = _mm_load1_ps(i0 + 5);
        voc0123 = _mm_add_ps(voc0123, _mm_mul_ps(vk02c2x0123, vi02c2));

        const __m128 vk12c2x0123 = _mm_load_ps(w + 104);
        const __m128 vi12c2 = _mm_load1_ps(i1 + 5);
        voc0123 = _mm_add_ps(voc0123, _mm_mul_ps(vk12c2x0123, vi12c2));

        const __m128 vk22c2x0123 = _mm_load_ps(w + 108);
        const __m128 vi22c2 = _mm_load1_ps(i2 + 5);
        voc0123 = _mm_add_ps(voc0123, _mm_mul_ps(vk22c2x0123, vi22c2));

        vi00c2 = vi02c2;
        vi10c2 = vi12c2;
        vi20c2 = vi22c2;

        voc0123 = _mm_min_ps(voc0123, vmax);
        voc0123 = _mm_max_ps(voc0123, vmin);

        _mm_store_ss(o0c0, voc0123); o0c0++;
        _mm_store_ss(o0c1, _mm_shuffle_ps(voc0123, voc0123, 1)); o0c1++;
        _mm_store_ss(o0c2, _mm_shuffle_ps(voc0123, voc0123, 2)); o0c2++;
        _mm_store_ss(o0c3, _mm_shuffle_ps(voc0123, voc0123, 3)); o0c3++;

        i0 += 6;
        i1 += 6;
        i2 += 6;
      }
      assert(iw < 2);
      if XNN_UNLIKELY(iw != 0) {
        __m128 voc0123 = _mm_load_ps(w);

        const __m128 vk00c0x0123 = _mm_load_ps(w + 4);
        voc0123 = _mm_add_ps(voc0123, _mm_mul_ps(vk00c0x0123, vi00c0));

        const __m128 vk10c0x0123 = _mm_load_ps(w + 8);
        voc0123 = _mm_add_ps(voc0123, _mm_mul_ps(vk10c0x0123, vi10c0));

        const __m128 vk20c0x0123 = _mm_load_ps(w + 12);
        voc0123 = _mm_add_ps(voc0123, _mm_mul_ps(vk20c0x0123, vi20c0));

        const __m128 vk00c1x0123 = _mm_load_ps(w + 16);
        voc0123 = _mm_add_ps(voc0123, _mm_mul_ps(vk00c1x0123, vi00c1));

        const __m128 vk10c1x0123 = _mm_load_ps(w + 20);
        voc0123 = _mm_add_ps(voc0123, _mm_mul_ps(vk10c1x0123, vi10c1));

        const __m128 vk20c1x0123 = _mm_load_ps(w + 24);
        voc0123 = _mm_add_ps(voc0123, _mm_mul_ps(vk20c1x0123, vi20c1));

        const __m128 vk00c2x0123 = _mm_load_ps(w + 28);
        voc0123 = _mm_add_ps(voc0123, _mm_mul_ps(vk00c2x0123, vi00c2));

        const __m128 vk10c2x0123 = _mm_load_ps(w + 32);
        voc0123 = _mm_add_ps(voc0123, _mm_mul_ps(vk10c2x0123, vi10c2));

        const __m128 vk20c2x0123 = _mm_load_ps(w + 36);
        voc0123 = _mm_add_ps(voc0123, _mm_mul_ps(vk20c2x0123, vi20c2));

        const __m128 vk01c0x0123 = _mm_load_ps(w + 40);
        const __m128 vi01c0 = _mm_load1_ps(i0);
        voc0123 = _mm_add_ps(voc0123, _mm_mul_ps(vk01c0x0123, vi01c0));

        const __m128 vk11c0x0123 = _mm_load_ps(w + 44);
        const __m128 vi11c0 = _mm_load1_ps(i1);
        voc0123 = _mm_add_ps(voc0123, _mm_mul_ps(vk11c0x0123, vi11c0));

        const __m128 vk21c0x0123 = _mm_load_ps(w + 48);
        const __m128 vi21c0 = _mm_load1_ps(i2);
        voc0123 = _mm_add_ps(voc0123, _mm_mul_ps(vk21c0x0123, vi21c0));

        const __m128 vk01c1x0123 = _mm_load_ps(w + 52);
        const __m128 vi01c1 = _mm_load1_ps(i0 + 1);
        voc0123 = _mm_add_ps(voc0123, _mm_mul_ps(vk01c1x0123, vi01c1));

        const __m128 vk11c1x0123 = _mm_load_ps(w + 56);
        const __m128 vi11c1 = _mm_load1_ps(i1 + 1);
        voc0123 = _mm_add_ps(voc0123, _mm_mul_ps(vk11c1x0123, vi11c1));

        const __m128 vk21c1x0123 = _mm_load_ps(w + 60);
        const __m128 vi21c1 = _mm_load1_ps(i2 + 1);
        voc0123 = _mm_add_ps(voc0123, _mm_mul_ps(vk21c1x0123, vi21c1));

        const __m128 vk01c2x0123 = _mm_load_ps(w + 64);
        const __m128 vi01c2 = _mm_load1_ps(i0 + 2);
        voc0123 = _mm_add_ps(voc0123, _mm_mul_ps(vk01c2x0123, vi01c2));

        const __m128 vk11c2x0123 = _mm_load_ps(w + 68);
        const __m128 vi11c2 = _mm_load1_ps(i1 + 2);
        voc0123 = _mm_add_ps(voc0123, _mm_mul_ps(vk11c2x0123, vi11c2));

        const __m128 vk21c2x0123 = _mm_load_ps(w + 72);
        const __m128 vi21c2 = _mm_load1_ps(i2 + 2);
        voc0123 = _mm_add_ps(voc0123, _mm_mul_ps(vk21c2x0123, vi21c2));

        voc0123 = _mm_min_ps(voc0123, vmax);
        voc0123 = _mm_max_ps(voc0123, vmin);

        _mm_store_ss(o0c0, voc0123); o0c0++;
        _mm_store_ss(o0c1, _mm_shuffle_ps(voc0123, voc0123, 1)); o0c1++;
        _mm_store_ss(o0c2, _mm_shuffle_ps(voc0123, voc0123, 2)); o0c2++;
        _mm_store_ss(o0c3, _mm_shuffle_ps(voc0123, voc0123, 3)); o0c3++;
      }
      // Move output pointers back to the position of the first pixel in a row,
      // and forward to the next block of output channels.
      o0c0 = (float*) ((uintptr_t) o0c0 + output_channel_increment);
      o0c1 = (float*) ((uintptr_t) o0c1 + output_channel_increment);
      o0c2 = (float*) ((uintptr_t) o0c2 + output_channel_increment);
      o0c3 = (float*) ((uintptr_t) o0c3 + output_channel_increment);
      // Revert input pointers to the position of the first pixel in a row
      i0 = (const float*) ((uintptr_t) i0 - input_width_decrement);
      i1 = (const float*) ((uintptr_t) i1 - input_width_decrement);
      i2 = (const float*) ((uintptr_t) i2 - input_width_decrement);
      // Move to the block of weights for the next 4 output channels
      w += 112;
      c = doz(c, 4);
    } while (c != 0);
    // Move output pointers forward to the next row
    output0 = (float*) ((uintptr_t) output0 + output_height_stride);
    // Move input pointers forward to the next row
    i0 = i2;
    i1 = (const float*) ((uintptr_t) i0 + input_height_stride);
    i2 = (const float*) ((uintptr_t) i1 + input_height_stride);
  }
}
