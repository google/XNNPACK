// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <xmmintrin.h>

#include <xnnpack/dwconv.h>
#include <xnnpack/math.h>


void xnn_f32_dwconv_spchw_ukernel_3x3p1__sse(
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

  const __m128 vmask = _mm_load_ps((const float*) params->sse.mask);
  const __m128 vmax = _mm_load_ps(params->sse.max);
  const __m128 vmin = _mm_load_ps(params->sse.min);

  const size_t input_width_increment = input_width_stride - round_up_po2(n, 4) / 4 * input_tuple_stride;
  const size_t output_width_increment = output_width_stride - (n - 1) / 4 * output_tuple_stride;

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
    // vi0x3012 = ( vi02, vi01, vi00, vi03 )
    __m128 vi0x3012 = _mm_setzero_ps();
    // vi1x3012 = ( vi12, vi11, vi10, vi13 )
    __m128 vi1x3012 = _mm_setzero_ps();
    // vi2x3012 = ( vi22, vi21, vi20, vi13 )
    __m128 vi2x3012 = _mm_setzero_ps();
    // vi0x4567 = ( vi07, vi06, vi05, vi04 )
    __m128 vi0x4567 = _mm_loadu_ps(i0);
    i0 = (const float*) ((uintptr_t) i0 + input_tuple_stride);
    // vi1x4567 = ( vi17, vi16, vi15, vi14 )
    __m128 vi1x4567 = _mm_loadu_ps(i1);
    i1 = (const float*) ((uintptr_t) i1 + input_tuple_stride);
    // vi2x4567 = ( vi27, vi26, vi25, vi24 )
    __m128 vi2x4567 = _mm_loadu_ps(i2);
    i2 = (const float*) ((uintptr_t) i2 + input_tuple_stride);

    size_t k = n;
    for (; k > 4; k -= 4) {
      __m128 vo4567p0 = vbias;

      // vi0x89AB = ( vi0B, vi0A, vi09, vi08 )
      const __m128 vi0x89AB = _mm_loadu_ps(i0);
      i0 = (const float*) ((uintptr_t) i0 + input_tuple_stride);
      // vi1x89AB = ( vi1B, vi0A, vi09, vi08 )
      const __m128 vi1x89AB = _mm_loadu_ps(i1);
      i1 = (const float*) ((uintptr_t) i1 + input_tuple_stride);
      // vi2x89AB = ( vi2B, vi0A, vi09, vi08 )
      const __m128 vi2x89AB = _mm_loadu_ps(i2);
      i2 = (const float*) ((uintptr_t) i2 + input_tuple_stride);

      // vi0x7456 = ( vi06, vi05, vi04, vi07 )
      const __m128 vi0x7456 = _mm_shuffle_ps(vi0x4567, vi0x4567, _MM_SHUFFLE(2, 1, 0, 3));
      // vi1x7456 = ( vi16, vi15, vi14, vi17 )
      const __m128 vi1x7456 = _mm_shuffle_ps(vi1x4567, vi1x4567, _MM_SHUFFLE(2, 1, 0, 3));
      // vi2x7456 = ( vi26, vi25, vi24, vi27 )
      const __m128 vi2x7456 = _mm_shuffle_ps(vi2x4567, vi2x4567, _MM_SHUFFLE(2, 1, 0, 3));

      vo4567p0 = _mm_add_ps(vo4567p0, _mm_mul_ps(vi0x4567, vk01));
      __m128 vo4567p1 = _mm_mul_ps(vi1x4567, vk11);
      __m128 vo4567p2 = _mm_mul_ps(vi2x4567, vk21);

      // vi0x3456 = ( vi06, vi05, vi04, vi03 )
      const __m128 vi0x3456 = _mm_move_ss(vi0x7456, vi0x3012);
      // vi1x3456 = ( vi16, vi15, vi14, vi13 )
      const __m128 vi1x3456 = _mm_move_ss(vi1x7456, vi1x3012);
      // vi2x3456 = ( vi26, vi25, vi24, vi23 )
      const __m128 vi2x3456 = _mm_move_ss(vi2x7456, vi2x3012);

      vo4567p0 = _mm_add_ps(vo4567p0, _mm_mul_ps(vi0x3456, vk00));
      vo4567p1 = _mm_add_ps(vo4567p1, _mm_mul_ps(vi1x3456, vk10));
      vo4567p2 = _mm_add_ps(vo4567p2, _mm_mul_ps(vi2x3456, vk20));

      vi0x3012 = vi0x7456;
      vi1x3012 = vi1x7456;
      vi2x3012 = vi2x7456;

      // vi0x8567 = ( vi07, vi06, vi05, vi08 )
      const __m128 vi0x8567 = _mm_move_ss(vi0x4567, vi0x89AB);
      // vi1x8567 = ( vi17, vi16, vi15, vi18 )
      const __m128 vi1x8567 = _mm_move_ss(vi1x4567, vi1x89AB);
      // vi2x8567 = ( vi27, vi26, vi25, vi28 )
      const __m128 vi2x8567 = _mm_move_ss(vi2x4567, vi2x89AB);

      // vi0x5678 = ( vi08, vi07, vi06, vi05 )
      const __m128 vi0x5678 = _mm_shuffle_ps(vi0x8567, vi0x8567, _MM_SHUFFLE(0, 3, 2, 1));
      // vi1x5678 = ( vi18, vi17, vi16, vi15 )
      const __m128 vi1x5678 = _mm_shuffle_ps(vi1x8567, vi1x8567, _MM_SHUFFLE(0, 3, 2, 1));
      // vi2x5678 = ( vi28, vi27, vi26, vi25 )
      const __m128 vi2x5678 = _mm_shuffle_ps(vi2x8567, vi2x8567, _MM_SHUFFLE(0, 3, 2, 1));

      vo4567p0 = _mm_add_ps(vo4567p0, _mm_mul_ps(vi0x5678, vk02));
      vo4567p1 = _mm_add_ps(vo4567p1, _mm_mul_ps(vi1x5678, vk12));
      vo4567p2 = _mm_add_ps(vo4567p2, _mm_mul_ps(vi2x5678, vk22));

      vi0x4567 = vi0x89AB;
      vi1x4567 = vi1x89AB;
      vi2x4567 = vi2x89AB;

      __m128 vo = _mm_add_ps(vo4567p0, vo4567p1);
      vo = _mm_add_ps(vo, vo4567p2);

      vo = _mm_max_ps(vo, vmin);
      vo = _mm_min_ps(vo, vmax);

      _mm_storeu_ps(output, vo);
      output = (float*) ((uintptr_t) output + output_tuple_stride);
    }
    // Always process the last block of 1..4 pixels.
    assert(k >= 1);
    assert(k <= 4);
    {
      __m128 vo4567p0 = vbias;

      vi0x4567 = _mm_and_ps(vmask, vi0x4567);
      vi1x4567 = _mm_and_ps(vmask, vi1x4567);
      vi2x4567 = _mm_and_ps(vmask, vi2x4567);

      // vi0x7456 = ( vi06, vi05, vi04, vi07 )
      const __m128 vi0x7456 = _mm_shuffle_ps(vi0x4567, vi0x4567, _MM_SHUFFLE(2, 1, 0, 3));
      // vi1x7456 = ( vi16, vi15, vi14, vi17 )
      const __m128 vi1x7456 = _mm_shuffle_ps(vi1x4567, vi1x4567, _MM_SHUFFLE(2, 1, 0, 3));
      // vi2x7456 = ( vi26, vi25, vi24, vi27 )
      const __m128 vi2x7456 = _mm_shuffle_ps(vi2x4567, vi2x4567, _MM_SHUFFLE(2, 1, 0, 3));

      vo4567p0 = _mm_add_ps(vo4567p0, _mm_mul_ps(vi0x4567, vk01));
      __m128 vo4567p1 = _mm_mul_ps(vi1x4567, vk11);
      __m128 vo4567p2 = _mm_mul_ps(vi2x4567, vk21);

      // vi0x3456 = ( vi06, vi05, vi04, vi03 )
      const __m128 vi0x3456 = _mm_move_ss(vi0x7456, vi0x3012);
      // vi1x3456 = ( vi16, vi15, vi14, vi13 )
      const __m128 vi1x3456 = _mm_move_ss(vi1x7456, vi1x3012);
      // vi2x3456 = ( vi26, vi25, vi24, vi23 )
      const __m128 vi2x3456 = _mm_move_ss(vi2x7456, vi2x3012);

      vo4567p0 = _mm_add_ps(vo4567p0, _mm_mul_ps(vi0x3456, vk00));
      vo4567p1 = _mm_add_ps(vo4567p1, _mm_mul_ps(vi1x3456, vk10));
      vo4567p2 = _mm_add_ps(vo4567p2, _mm_mul_ps(vi2x3456, vk20));

      const __m128 vzero = _mm_setzero_ps();
      // vi0x8567 = ( vi07, vi06, vi05, 0.0 )
      const __m128 vi0x8567 = _mm_move_ss(vi0x4567, vzero);
      // vi1x8567 = ( vi17, vi16, vi15, 0.0 )
      const __m128 vi1x8567 = _mm_move_ss(vi1x4567, vzero);
      // vi2x8567 = ( vi27, vi26, vi25, 0.0 )
      const __m128 vi2x8567 = _mm_move_ss(vi2x4567, vzero);

      // vi0x5678 = ( vi08, vi07, vi06, vi05 )
      const __m128 vi0x5678 = _mm_shuffle_ps(vi0x8567, vi0x8567, _MM_SHUFFLE(0, 3, 2, 1));
      // vi1x5678 = ( vi18, vi17, vi16, vi15 )
      const __m128 vi1x5678 = _mm_shuffle_ps(vi1x8567, vi1x8567, _MM_SHUFFLE(0, 3, 2, 1));
      // vi2x5678 = ( vi28, vi27, vi26, vi25 )
      const __m128 vi2x5678 = _mm_shuffle_ps(vi2x8567, vi2x8567, _MM_SHUFFLE(0, 3, 2, 1));

      vo4567p0 = _mm_add_ps(vo4567p0, _mm_mul_ps(vi0x5678, vk02));
      vo4567p1 = _mm_add_ps(vo4567p1, _mm_mul_ps(vi1x5678, vk12));
      vo4567p2 = _mm_add_ps(vo4567p2, _mm_mul_ps(vi2x5678, vk22));

      __m128 vo = _mm_add_ps(vo4567p0, vo4567p1);
      vo = _mm_add_ps(vo, vo4567p2);

      vo = _mm_max_ps(vo, vmin);
      vo = _mm_min_ps(vo, vmax);

      if XNN_LIKELY(k == 4) {
        _mm_storeu_ps(output, vo);
      } else {
        float* output_lo = output;
        if (k & 2) {
          _mm_storel_pi((__m64*) output_lo, vo);
          output_lo += 2;
          vo = _mm_movehl_ps(vo, vo);
        }
        if (k & 1) {
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
