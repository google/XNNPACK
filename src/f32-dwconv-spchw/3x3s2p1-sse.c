// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <xmmintrin.h>

#include <xnnpack/dwconv.h>
#include <xnnpack/math.h>


void xnn_f32_dwconv_spchw_ukernel_3x3s2p1__sse(
    size_t m,
    size_t n,
    const float* input,
    const float* weights,
    float* output,
    size_t input_tuple_stride,
    size_t output_tuple_stride,
    size_t input_width_stride,
    size_t output_width_stride,
    const union xnn_f32_spchw_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(n != 0);

  const __m128 vmask_even = _mm_load_ps((const float*) params->sse.mask_even);
  const __m128 vmask_odd  = _mm_load_ps((const float*) params->sse.mask_odd);
  const __m128 vmax = _mm_load_ps(params->sse.max);
  const __m128 vmin = _mm_load_ps(params->sse.min);

  const size_t input_width_increment = input_width_stride * 2 - n / 8 * input_tuple_stride * 2;
  const size_t output_width_increment = output_width_stride - n / 8 * output_tuple_stride;

  // No vertical padding.
  const float* i0 = input;
  const float* i1 = (const float*) ((uintptr_t) i0 + input_width_stride);
  const float* i2 = (const float*) ((uintptr_t) i1 + input_width_stride);

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

  do {
    __m128 vi0x7531 = _mm_setzero_ps();
    __m128 vi1x7531 = _mm_setzero_ps();
    __m128 vi2x7531 = _mm_setzero_ps();

    size_t k = n;
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

    i0 = (const float*) ((uintptr_t) i0 + input_width_increment);
    i1 = (const float*) ((uintptr_t) i1 + input_width_increment);
    i2 = (const float*) ((uintptr_t) i2 + input_width_increment);
    output = (float*) ((uintptr_t) output + output_width_increment);
  } while (--m != 0);
}
