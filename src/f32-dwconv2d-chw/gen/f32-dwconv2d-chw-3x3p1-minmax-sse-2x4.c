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


void xnn_f32_dwconv2d_chw_ukernel_3x3p1__sse_2x4(
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

  float* o0 = output;
  float* o1 = (float*) ((uintptr_t) o0 + input_width);

  size_t output_height = input_height;
  do {
    if XNN_UNPREDICTABLE(output_height < 2) {
      i2 = zero;
      o1 = o0;
    }
    if XNN_UNPREDICTABLE(output_height < 3) {
      i3 = zero;
    }

    // vi0x3012 = ( vi02, vi01, vi{M}0, vi{M}3 )
    __m128 vi0x3012 = _mm_setzero_ps();
    // vi1x3012 = ( vi12, vi11, vi{M}0, vi{M}3 )
    __m128 vi1x3012 = _mm_setzero_ps();
    // vi2x3012 = ( vi22, vi21, vi{M}0, vi{M}3 )
    __m128 vi2x3012 = _mm_setzero_ps();
    // vi3x3012 = ( vi32, vi31, vi{M}0, vi{M}3 )
    __m128 vi3x3012 = _mm_setzero_ps();

    __m128 vi0x4567 = _mm_loadu_ps(i0);
    i0 += 4;
    __m128 vi1x4567 = _mm_loadu_ps(i1);
    i1 += 4;
    __m128 vi2x4567 = _mm_loadu_ps(i2);
    i2 += 4;
    __m128 vi3x4567 = _mm_loadu_ps(i3);
    i3 += 4;

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

      // vi0x7456 = ( vi06, vi05, vi04, vi07 )
      const __m128 vi0x7456 = _mm_shuffle_ps(vi0x4567, vi0x4567, _MM_SHUFFLE(2, 1, 0, 3));
      // vi1x7456 = ( vi16, vi15, vi14, vi17 )
      const __m128 vi1x7456 = _mm_shuffle_ps(vi1x4567, vi1x4567, _MM_SHUFFLE(2, 1, 0, 3));
      // vi2x7456 = ( vi26, vi25, vi24, vi27 )
      const __m128 vi2x7456 = _mm_shuffle_ps(vi2x4567, vi2x4567, _MM_SHUFFLE(2, 1, 0, 3));
      // vi3x7456 = ( vi36, vi35, vi34, vi37 )
      const __m128 vi3x7456 = _mm_shuffle_ps(vi3x4567, vi3x4567, _MM_SHUFFLE(2, 1, 0, 3));

      __m128 vo0p0 = _mm_add_ps(vbias, _mm_mul_ps(vi0x4567, vk01));
      __m128 vo1p0 = _mm_add_ps(vbias, _mm_mul_ps(vi1x4567, vk01));
      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi1x4567, vk11));
      vo1p0 = _mm_add_ps(vo1p0, _mm_mul_ps(vi2x4567, vk11));
      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi2x4567, vk21));
      vo1p0 = _mm_add_ps(vo1p0, _mm_mul_ps(vi3x4567, vk21));

      // vi0x3456 = ( vi06, vi05, vi04, vi03 )
      const __m128 vi0x3456 = _mm_move_ss(vi0x7456, vi0x3012);
      // vi1x3456 = ( vi16, vi15, vi14, vi13 )
      const __m128 vi1x3456 = _mm_move_ss(vi1x7456, vi1x3012);
      // vi2x3456 = ( vi26, vi25, vi24, vi23 )
      const __m128 vi2x3456 = _mm_move_ss(vi2x7456, vi2x3012);
      // vi3x3456 = ( vi36, vi35, vi34, vi33 )
      const __m128 vi3x3456 = _mm_move_ss(vi3x7456, vi3x3012);

      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi0x3456, vk00));
      vo1p0 = _mm_add_ps(vo1p0, _mm_mul_ps(vi1x3456, vk00));
      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi1x3456, vk10));
      vo1p0 = _mm_add_ps(vo1p0, _mm_mul_ps(vi2x3456, vk10));
      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi2x3456, vk20));
      vo1p0 = _mm_add_ps(vo1p0, _mm_mul_ps(vi3x3456, vk20));

      vi0x3012 = vi0x7456;
      vi1x3012 = vi1x7456;
      vi2x3012 = vi2x7456;
      vi3x3012 = vi3x7456;

      // vi0x8567 = ( vi07, vi06, vi05, vi08 )
      const __m128 vi0x8567 = _mm_move_ss(vi0x4567, vi0x89AB);
      // vi1x8567 = ( vi17, vi16, vi15, vi18 )
      const __m128 vi1x8567 = _mm_move_ss(vi1x4567, vi1x89AB);
      // vi2x8567 = ( vi27, vi26, vi25, vi28 )
      const __m128 vi2x8567 = _mm_move_ss(vi2x4567, vi2x89AB);
      // vi3x8567 = ( vi37, vi36, vi35, vi38 )
      const __m128 vi3x8567 = _mm_move_ss(vi3x4567, vi3x89AB);

      // vi0x5678 = ( vi08, vi07, vi06, vi05 )
      const __m128 vi0x5678 = _mm_shuffle_ps(vi0x8567, vi0x8567, _MM_SHUFFLE(0, 3, 2, 1));
      // vi1x5678 = ( vi18, vi17, vi16, vi15 )
      const __m128 vi1x5678 = _mm_shuffle_ps(vi1x8567, vi1x8567, _MM_SHUFFLE(0, 3, 2, 1));
      // vi2x5678 = ( vi28, vi27, vi26, vi25 )
      const __m128 vi2x5678 = _mm_shuffle_ps(vi2x8567, vi2x8567, _MM_SHUFFLE(0, 3, 2, 1));
      // vi3x5678 = ( vi38, vi37, vi36, vi35 )
      const __m128 vi3x5678 = _mm_shuffle_ps(vi3x8567, vi3x8567, _MM_SHUFFLE(0, 3, 2, 1));

      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi0x5678, vk02));
      vo1p0 = _mm_add_ps(vo1p0, _mm_mul_ps(vi1x5678, vk02));
      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi1x5678, vk12));
      vo1p0 = _mm_add_ps(vo1p0, _mm_mul_ps(vi2x5678, vk12));
      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi2x5678, vk22));
      vo1p0 = _mm_add_ps(vo1p0, _mm_mul_ps(vi3x5678, vk22));

      vi0x4567 = vi0x89AB;
      vi1x4567 = vi1x89AB;
      vi2x4567 = vi2x89AB;
      vi3x4567 = vi3x89AB;


      __m128 vo0 = _mm_max_ps(vo0p0, vmin);
      __m128 vo1 = _mm_max_ps(vo1p0, vmin);

      vo0 = _mm_min_ps(vo0, vmax);
      vo1 = _mm_min_ps(vo1, vmax);

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

      // vi0x7456 = ( vi06, vi05, vi04, vi07 )
      const __m128 vi0x7456 = _mm_shuffle_ps(vi0x4567, vi0x4567, _MM_SHUFFLE(2, 1, 0, 3));
      // vi1x7456 = ( vi16, vi15, vi14, vi17 )
      const __m128 vi1x7456 = _mm_shuffle_ps(vi1x4567, vi1x4567, _MM_SHUFFLE(2, 1, 0, 3));
      // vi2x7456 = ( vi26, vi25, vi24, vi27 )
      const __m128 vi2x7456 = _mm_shuffle_ps(vi2x4567, vi2x4567, _MM_SHUFFLE(2, 1, 0, 3));
      // vi3x7456 = ( vi36, vi35, vi34, vi37 )
      const __m128 vi3x7456 = _mm_shuffle_ps(vi3x4567, vi3x4567, _MM_SHUFFLE(2, 1, 0, 3));

      __m128 vo0p0 = _mm_add_ps(vbias, _mm_mul_ps(vi0x4567, vk01));
      __m128 vo1p0 = _mm_add_ps(vbias, _mm_mul_ps(vi1x4567, vk01));
      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi1x4567, vk11));
      vo1p0 = _mm_add_ps(vo1p0, _mm_mul_ps(vi2x4567, vk11));
      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi2x4567, vk21));
      vo1p0 = _mm_add_ps(vo1p0, _mm_mul_ps(vi3x4567, vk21));

      // vi0x3456 = ( vi06, vi05, vi04, vi03 )
      const __m128 vi0x3456 = _mm_move_ss(vi0x7456, vi0x3012);
      // vi1x3456 = ( vi16, vi15, vi14, vi13 )
      const __m128 vi1x3456 = _mm_move_ss(vi1x7456, vi1x3012);
      // vi2x3456 = ( vi26, vi25, vi24, vi23 )
      const __m128 vi2x3456 = _mm_move_ss(vi2x7456, vi2x3012);
      // vi3x3456 = ( vi36, vi35, vi34, vi33 )
      const __m128 vi3x3456 = _mm_move_ss(vi3x7456, vi3x3012);

      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi0x3456, vk00));
      vo1p0 = _mm_add_ps(vo1p0, _mm_mul_ps(vi1x3456, vk00));
      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi1x3456, vk10));
      vo1p0 = _mm_add_ps(vo1p0, _mm_mul_ps(vi2x3456, vk10));
      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi2x3456, vk20));
      vo1p0 = _mm_add_ps(vo1p0, _mm_mul_ps(vi3x3456, vk20));

      const __m128 vzero = _mm_setzero_ps();
      // vi0x8567 = ( vi07, vi06, vi05, 0.0 )
      const __m128 vi0x8567 = _mm_move_ss(vi0x4567, vzero);
      // vi1x8567 = ( vi17, vi16, vi15, 0.0 )
      const __m128 vi1x8567 = _mm_move_ss(vi1x4567, vzero);
      // vi2x8567 = ( vi27, vi26, vi25, 0.0 )
      const __m128 vi2x8567 = _mm_move_ss(vi2x4567, vzero);
      // vi3x8567 = ( vi37, vi36, vi35, 0.0 )
      const __m128 vi3x8567 = _mm_move_ss(vi3x4567, vzero);

      // vi0x5678 = ( vi08, vi07, vi06, vi05 )
      const __m128 vi0x5678 = _mm_shuffle_ps(vi0x8567, vi0x8567, _MM_SHUFFLE(0, 3, 2, 1));
      // vi1x5678 = ( vi18, vi17, vi16, vi15 )
      const __m128 vi1x5678 = _mm_shuffle_ps(vi1x8567, vi1x8567, _MM_SHUFFLE(0, 3, 2, 1));
      // vi2x5678 = ( vi28, vi27, vi26, vi25 )
      const __m128 vi2x5678 = _mm_shuffle_ps(vi2x8567, vi2x8567, _MM_SHUFFLE(0, 3, 2, 1));
      // vi3x5678 = ( vi38, vi37, vi36, vi35 )
      const __m128 vi3x5678 = _mm_shuffle_ps(vi3x8567, vi3x8567, _MM_SHUFFLE(0, 3, 2, 1));

      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi0x5678, vk02));
      vo1p0 = _mm_add_ps(vo1p0, _mm_mul_ps(vi1x5678, vk02));
      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi1x5678, vk12));
      vo1p0 = _mm_add_ps(vo1p0, _mm_mul_ps(vi2x5678, vk12));
      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi2x5678, vk22));
      vo1p0 = _mm_add_ps(vo1p0, _mm_mul_ps(vi3x5678, vk22));


      __m128 vo0 = _mm_max_ps(vo0p0, vmin);
      __m128 vo1 = _mm_max_ps(vo1p0, vmin);

      vo0 = _mm_min_ps(vo0, vmax);
      vo1 = _mm_min_ps(vo1, vmax);

      if XNN_LIKELY(w == 4 * sizeof(float)) {
        _mm_storeu_ps(o1, vo1);
        o1 += 4;
        _mm_storeu_ps(o0, vo0);
        o0 += 4;
      } else {
        if (w & (2 * sizeof(float))) {
          _mm_storel_pi((__m64*) o1, vo1);
          o1 += 2;
          _mm_storel_pi((__m64*) o0, vo0);
          o0 += 2;

          vo0 = _mm_movehl_ps(vo0, vo0);
          vo1 = _mm_movehl_ps(vo1, vo1);
        }
        if (w & (1 * sizeof(float))) {
          _mm_store_ss(o1, vo1);
          o1 += 1;
          _mm_store_ss(o0, vo0);
          o0 += 1;
        }
      }
    }

    i0 = (const float*) ((uintptr_t) i2 - input_decrement);
    i1 = (const float*) ((uintptr_t) i3 - input_decrement);
    i2 = (const float*) ((uintptr_t) i1 + input_width);
    i3 = (const float*) ((uintptr_t) i2 + input_width);

    o0 = o1;
    o1 = (float*) ((uintptr_t) o0 + input_width);

    output_height = doz(output_height, 2);
  } while (output_height != 0);
}
