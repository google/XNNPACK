// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <xmmintrin.h>

#include <xnnpack/dwconv.h>
#include <xnnpack/math.h>


void xnn_f32_dwconv_chw_ukernel_3x3s2p1__sse(
    size_t input_height,
    size_t input_width,
    const float* input,
    const float* weights,
    const float* zero,
    float* output,
    uint32_t padding_top,
    size_t input_tuple_stride,
    size_t output_tuple_stride,
    size_t input_width_stride,
    size_t output_width_stride,
    const union xnn_f32_chw_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(input_height!= 0);
  assert(input_width != 0);
  assert(padding_top >= 0 && padding_top <= 1);

  const size_t padded_input_height = input_height + padding_top + 1 /* padding_bottom */;
  const size_t output_height = (padded_input_height - 3) / 2 + 1;

  const __m128 vmask_even = _mm_load_ps((const float*) params->sse.mask_even);
  const __m128 vmask_odd  = _mm_load_ps((const float*) params->sse.mask_odd);
  const __m128 vmax = _mm_load_ps(params->sse.max);
  const __m128 vmin = _mm_load_ps(params->sse.min);

  const size_t input_width_decrement_single = input_width / 8  * input_tuple_stride * 2;
  const size_t input_width_increment = input_width_stride * 2 - input_width_decrement_single;
  const size_t output_width_increment = output_width_stride - input_width / 8 * output_tuple_stride;

  const float* i0;
  const float* i1;
  const float* i2;

  if (padding_top == 0) {
    i0 = input;
    i1 = (const float*) ((uintptr_t) i0 + input_width_stride);
    i2 = (const float*) ((uintptr_t) i1 + input_width_stride);
    if (input_height <= 2) {
      i2 = zero;
    }
    if (input_height == 1) {
      i1 = zero;
    }
  } else {
    i0 = zero;
    i1 = input;
    i2 = (const float*) ((uintptr_t) i1 + input_width_stride);
    if (input_height == 1) {
      i2 = zero;
    }
  }

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

  size_t m = output_height;
  do {
    __m128 vi0x7531 = _mm_setzero_ps();
    __m128 vi1x7531 = _mm_setzero_ps();
    __m128 vi2x7531 = _mm_setzero_ps();

    size_t k = input_width;
    for (; k >= 8; k -= 8) {
      __m128 vo8ACEp0 = vbias;

      const __m128 vi0x89AB = _mm_loadu_ps(i0);
      i0 = (const float*) ((uintptr_t) i0 + input_tuple_stride);
      const __m128 vi1x89AB = _mm_loadu_ps(i1);
      i1 = (const float*) ((uintptr_t) i1 + input_tuple_stride);
      const __m128 vi2x89AB = _mm_loadu_ps(i2);
      i2 = (const float*) ((uintptr_t) i2 + input_tuple_stride);

      const __m128 vi0xCDEF = _mm_loadu_ps(i0);
      i0 = (const float*) ((uintptr_t) i0 + input_tuple_stride);
      const __m128 vi1xCDEF = _mm_loadu_ps(i1);
      i1 = (const float*) ((uintptr_t) i1 + input_tuple_stride);
      const __m128 vi2xCDEF = _mm_loadu_ps(i2);
      i2 = (const float*) ((uintptr_t) i2 + input_tuple_stride);

      const __m128 vi0x8ACE = _mm_shuffle_ps(vi0x89AB, vi0xCDEF, _MM_SHUFFLE(2, 0, 2, 0));
      const __m128 vi0x9BDF = _mm_shuffle_ps(vi0x89AB, vi0xCDEF, _MM_SHUFFLE(3, 1, 3, 1));
      const __m128 vi1x8ACE = _mm_shuffle_ps(vi1x89AB, vi1xCDEF, _MM_SHUFFLE(2, 0, 2, 0));
      const __m128 vi1x9BDF = _mm_shuffle_ps(vi1x89AB, vi1xCDEF, _MM_SHUFFLE(3, 1, 3, 1));
      const __m128 vi2x8ACE = _mm_shuffle_ps(vi2x89AB, vi2xCDEF, _MM_SHUFFLE(2, 0, 2, 0));
      const __m128 vi2x9BDF = _mm_shuffle_ps(vi2x89AB, vi2xCDEF, _MM_SHUFFLE(3, 1, 3, 1));

      vo8ACEp0 = _mm_add_ps(vo8ACEp0, _mm_mul_ps(vi0x8ACE, vk01));
      __m128 vo8ACEp1 = _mm_mul_ps(vi1x8ACE, vk11);
      __m128 vo8ACEp2 = _mm_mul_ps(vi2x8ACE, vk21);

      const __m128 vi0xF9BD = _mm_shuffle_ps(vi0x9BDF, vi0x9BDF, _MM_SHUFFLE(2, 1, 0, 3));
      const __m128 vi1xF9BD = _mm_shuffle_ps(vi1x9BDF, vi1x9BDF, _MM_SHUFFLE(2, 1, 0, 3));
      const __m128 vi2xF9BD = _mm_shuffle_ps(vi2x9BDF, vi2x9BDF, _MM_SHUFFLE(2, 1, 0, 3));

      vo8ACEp0 = _mm_add_ps(vo8ACEp0, _mm_mul_ps(vi0x9BDF, vk02));
      vo8ACEp1 = _mm_add_ps(vo8ACEp1, _mm_mul_ps(vi1x9BDF, vk12));
      vo8ACEp2 = _mm_add_ps(vo8ACEp2, _mm_mul_ps(vi2x9BDF, vk22));

      const __m128 vi0x7BDF = _mm_move_ss(vi0xF9BD, vi0x7531);
      const __m128 vi1x7BDF = _mm_move_ss(vi1xF9BD, vi1x7531);
      const __m128 vi2x7BDF = _mm_move_ss(vi2xF9BD, vi2x7531);

      vi0x7531 = vi0xF9BD;
      vi1x7531 = vi1xF9BD;
      vi2x7531 = vi2xF9BD;

      vo8ACEp0 = _mm_add_ps(vo8ACEp0, _mm_mul_ps(vi0x7BDF, vk00));
      vo8ACEp1 = _mm_add_ps(vo8ACEp1, _mm_mul_ps(vi1x7BDF, vk10));
      vo8ACEp2 = _mm_add_ps(vo8ACEp2, _mm_mul_ps(vi2x7BDF, vk20));

      __m128 vo = _mm_add_ps(vo8ACEp0, vo8ACEp1);
      vo = _mm_add_ps(vo, vo8ACEp2);

      vo = _mm_max_ps(vo, vmin);
      vo = _mm_min_ps(vo, vmax);

      _mm_storeu_ps(output, vo);
      output = (float*) ((uintptr_t) output + output_tuple_stride);
    }
    // Last block has 0-7 pixels to process.
    assert(k < 8);
    if XNN_LIKELY(k != 0) {
      __m128 vo8ACEp0 = vbias;

      const __m128 vi0x89AB = _mm_loadu_ps(i0);
      const __m128 vi1x89AB = _mm_loadu_ps(i1);
      const __m128 vi2x89AB = _mm_loadu_ps(i2);

      const __m128 vi0xCDEF = _mm_loadu_ps((const float*) ((uintptr_t) i0 + input_tuple_stride));
      const __m128 vi1xCDEF = _mm_loadu_ps((const float*) ((uintptr_t) i1 + input_tuple_stride));
      const __m128 vi2xCDEF = _mm_loadu_ps((const float*) ((uintptr_t) i2 + input_tuple_stride));

      const __m128 vi0x8ACE = _mm_and_ps(vmask_even, _mm_shuffle_ps(vi0x89AB, vi0xCDEF, _MM_SHUFFLE(2, 0, 2, 0)));
      const __m128 vi0x9BDF = _mm_and_ps(vmask_odd,  _mm_shuffle_ps(vi0x89AB, vi0xCDEF, _MM_SHUFFLE(3, 1, 3, 1)));
      const __m128 vi1x8ACE = _mm_and_ps(vmask_even, _mm_shuffle_ps(vi1x89AB, vi1xCDEF, _MM_SHUFFLE(2, 0, 2, 0)));
      const __m128 vi1x9BDF = _mm_and_ps(vmask_odd,  _mm_shuffle_ps(vi1x89AB, vi1xCDEF, _MM_SHUFFLE(3, 1, 3, 1)));
      const __m128 vi2x8ACE = _mm_and_ps(vmask_even, _mm_shuffle_ps(vi2x89AB, vi2xCDEF, _MM_SHUFFLE(2, 0, 2, 0)));
      const __m128 vi2x9BDF = _mm_and_ps(vmask_odd,  _mm_shuffle_ps(vi2x89AB, vi2xCDEF, _MM_SHUFFLE(3, 1, 3, 1)));

      vo8ACEp0 = _mm_add_ps(vo8ACEp0, _mm_mul_ps(vi0x8ACE, vk01));
      __m128 vo8ACEp1 = _mm_mul_ps(vi1x8ACE, vk11);
      __m128 vo8ACEp2 = _mm_mul_ps(vi2x8ACE, vk21);

      const __m128 vi0xF9BD = _mm_shuffle_ps(vi0x9BDF, vi0x9BDF, _MM_SHUFFLE(2, 1, 0, 3));
      const __m128 vi1xF9BD = _mm_shuffle_ps(vi1x9BDF, vi1x9BDF, _MM_SHUFFLE(2, 1, 0, 3));
      const __m128 vi2xF9BD = _mm_shuffle_ps(vi2x9BDF, vi2x9BDF, _MM_SHUFFLE(2, 1, 0, 3));

      vo8ACEp0 = _mm_add_ps(vo8ACEp0, _mm_mul_ps(vi0x9BDF, vk02));
      vo8ACEp1 = _mm_add_ps(vo8ACEp1, _mm_mul_ps(vi1x9BDF, vk12));
      vo8ACEp2 = _mm_add_ps(vo8ACEp2, _mm_mul_ps(vi2x9BDF, vk22));

      const __m128 vi0x7BDF = _mm_move_ss(vi0xF9BD, vi0x7531);
      const __m128 vi1x7BDF = _mm_move_ss(vi1xF9BD, vi1x7531);
      const __m128 vi2x7BDF = _mm_move_ss(vi2xF9BD, vi2x7531);

      vo8ACEp0 = _mm_add_ps(vo8ACEp0, _mm_mul_ps(vi0x7BDF, vk00));
      vo8ACEp1 = _mm_add_ps(vo8ACEp1, _mm_mul_ps(vi1x7BDF, vk10));
      vo8ACEp2 = _mm_add_ps(vo8ACEp2, _mm_mul_ps(vi2x7BDF, vk20));

      __m128 vo = _mm_add_ps(vo8ACEp0, vo8ACEp1);
      vo = _mm_add_ps(vo, vo8ACEp2);

      vo = _mm_max_ps(vo, vmin);
      vo = _mm_min_ps(vo, vmax);

      if (k == 7) {
        _mm_storeu_ps(output, vo);
      } else {
        float* output_lo = output;
        k += 1;
        if (k & 4) {
          _mm_storel_pi((__m64*) output_lo, vo);
          output_lo += 2;
          vo = _mm_movehl_ps(vo, vo);
        }
        if (k & 2) {
          _mm_store_ss(output_lo, vo);
        }
      }
    }

    i0 = (const float*) ((uintptr_t) i2 - input_width_decrement_single);
    i1 = (const float*) ((uintptr_t) i1 + input_width_increment);
    i2 = (const float*) ((uintptr_t) i2 + input_width_increment);
    output = (float*) ((uintptr_t) output + output_width_increment);
    m -= 1;
    if (m == 1 && padding_top == input_height % 2) {
      // to mimic the following code with only one if, we do some small
      // shenanigans...
      // if (padding_top == 0 && input_height % 2 == 0) {
      //   i2 = zero;
      // } else if (padding_top == 1 && input_height % 2 == 1) {
      //   i2 = zero;
      // }
      i2 = zero;
    }
  } while (m != 0);
}
