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

#include "xnnpack/dwconv.h"
#include "xnnpack/math.h"


void xnn_f32_dwconv2d_chw_ukernel_3x3s2p1__sse_1x4_acc2(
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
  assert(padding_top >= 0);
  assert(padding_top <= 1);

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
  const __m128 vk10 = _mm_load1_ps(weights + 4);
  const __m128 vk11 = _mm_load1_ps(weights + 5);
  const __m128 vk12 = _mm_load1_ps(weights + 6);
  const __m128 vk20 = _mm_load1_ps(weights + 7);
  const __m128 vk21 = _mm_load1_ps(weights + 8);
  const __m128 vk22 = _mm_load1_ps(weights + 9);

  const size_t input_decrement = round_down_po2(input_width, 4 /* SIMD output width */ * 2 /* subsampling */ * sizeof(float));

  const float* i0 = (const float*) ((uintptr_t) input - ((-padding_top) & input_width));
  const float* i1 = (const float*) ((uintptr_t) i0 + input_width);
  if XNN_UNPREDICTABLE(padding_top != 0) {
    i0 = zero;
  }
  const float* i2 = (const float*) ((uintptr_t) i1 + input_width);

  float* o0 = output;

  size_t padded_input_height = input_height + padding_top + 1 /* padding bottom */;
  size_t output_height = (padded_input_height - 3 /* kernel size */ + 2 /* subsampling */) / 2;
  do {
    if XNN_UNPREDICTABLE(padded_input_height < 4) {
      i2 = zero;
    }

    __m128 vi0x7531 = _mm_setzero_ps();
    __m128 vi1x7531 = _mm_setzero_ps();
    __m128 vi2x7531 = _mm_setzero_ps();

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

      const __m128 vi0x8ACE = _mm_shuffle_ps(vi0x89AB, vi0xCDEF, _MM_SHUFFLE(2, 0, 2, 0));
      const __m128 vi0x9BDF = _mm_shuffle_ps(vi0x89AB, vi0xCDEF, _MM_SHUFFLE(3, 1, 3, 1));
      const __m128 vi1x8ACE = _mm_shuffle_ps(vi1x89AB, vi1xCDEF, _MM_SHUFFLE(2, 0, 2, 0));
      const __m128 vi1x9BDF = _mm_shuffle_ps(vi1x89AB, vi1xCDEF, _MM_SHUFFLE(3, 1, 3, 1));
      const __m128 vi2x8ACE = _mm_shuffle_ps(vi2x89AB, vi2xCDEF, _MM_SHUFFLE(2, 0, 2, 0));
      const __m128 vi2x9BDF = _mm_shuffle_ps(vi2x89AB, vi2xCDEF, _MM_SHUFFLE(3, 1, 3, 1));

      __m128 vo0p0 = _mm_add_ps(vbias, _mm_mul_ps(vi0x8ACE, vk01));
      __m128 vo0p1 = _mm_mul_ps(vi1x8ACE, vk11);
      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi2x8ACE, vk21));

      const __m128 vi0xF9BD = _mm_shuffle_ps(vi0x9BDF, vi0x9BDF, _MM_SHUFFLE(2, 1, 0, 3));
      const __m128 vi1xF9BD = _mm_shuffle_ps(vi1x9BDF, vi1x9BDF, _MM_SHUFFLE(2, 1, 0, 3));
      const __m128 vi2xF9BD = _mm_shuffle_ps(vi2x9BDF, vi2x9BDF, _MM_SHUFFLE(2, 1, 0, 3));

      vo0p1 = _mm_add_ps(vo0p1, _mm_mul_ps(vi0x9BDF, vk02));
      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi1x9BDF, vk12));
      vo0p1 = _mm_add_ps(vo0p1, _mm_mul_ps(vi2x9BDF, vk22));

      const __m128 vi0x79BD = _mm_move_ss(vi0xF9BD, vi0x7531);
      const __m128 vi1x79BD = _mm_move_ss(vi1xF9BD, vi1x7531);
      const __m128 vi2x79BD = _mm_move_ss(vi2xF9BD, vi2x7531);

      vi0x7531 = vi0xF9BD;
      vi1x7531 = vi1xF9BD;
      vi2x7531 = vi2xF9BD;

      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi0x79BD, vk00));
      vo0p1 = _mm_add_ps(vo0p1, _mm_mul_ps(vi1x79BD, vk10));
      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi2x79BD, vk20));

      vo0p0 = _mm_add_ps(vo0p0, vo0p1);

      __m128 vo0 = _mm_max_ps(vo0p0, vmin);

      vo0 = _mm_min_ps(vo0, vmax);

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

      const __m128 vi0x8ACE = _mm_and_ps(vmask_even, _mm_shuffle_ps(vi0x89AB, vi0xCDEF, _MM_SHUFFLE(2, 0, 2, 0)));
      const __m128 vi0x9BDF = _mm_and_ps(vmask_odd,  _mm_shuffle_ps(vi0x89AB, vi0xCDEF, _MM_SHUFFLE(3, 1, 3, 1)));
      const __m128 vi1x8ACE = _mm_and_ps(vmask_even, _mm_shuffle_ps(vi1x89AB, vi1xCDEF, _MM_SHUFFLE(2, 0, 2, 0)));
      const __m128 vi1x9BDF = _mm_and_ps(vmask_odd,  _mm_shuffle_ps(vi1x89AB, vi1xCDEF, _MM_SHUFFLE(3, 1, 3, 1)));
      const __m128 vi2x8ACE = _mm_and_ps(vmask_even, _mm_shuffle_ps(vi2x89AB, vi2xCDEF, _MM_SHUFFLE(2, 0, 2, 0)));
      const __m128 vi2x9BDF = _mm_and_ps(vmask_odd,  _mm_shuffle_ps(vi2x89AB, vi2xCDEF, _MM_SHUFFLE(3, 1, 3, 1)));

      __m128 vo0p0 = _mm_add_ps(vbias, _mm_mul_ps(vi0x8ACE, vk01));
      __m128 vo0p1 = _mm_mul_ps(vi1x8ACE, vk11);
      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi2x8ACE, vk21));

      const __m128 vi0xF9BD = _mm_shuffle_ps(vi0x9BDF, vi0x9BDF, _MM_SHUFFLE(2, 1, 0, 3));
      const __m128 vi1xF9BD = _mm_shuffle_ps(vi1x9BDF, vi1x9BDF, _MM_SHUFFLE(2, 1, 0, 3));
      const __m128 vi2xF9BD = _mm_shuffle_ps(vi2x9BDF, vi2x9BDF, _MM_SHUFFLE(2, 1, 0, 3));

      vo0p1 = _mm_add_ps(vo0p1, _mm_mul_ps(vi0x9BDF, vk02));
      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi1x9BDF, vk12));
      vo0p1 = _mm_add_ps(vo0p1, _mm_mul_ps(vi2x9BDF, vk22));

      const __m128 vi0x79BD = _mm_move_ss(vi0xF9BD, vi0x7531);
      const __m128 vi1x79BD = _mm_move_ss(vi1xF9BD, vi1x7531);
      const __m128 vi2x79BD = _mm_move_ss(vi2xF9BD, vi2x7531);

      vi0x7531 = vi0xF9BD;
      vi1x7531 = vi1xF9BD;
      vi2x7531 = vi2xF9BD;

      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi0x79BD, vk00));
      vo0p1 = _mm_add_ps(vo0p1, _mm_mul_ps(vi1x79BD, vk10));
      vo0p0 = _mm_add_ps(vo0p0, _mm_mul_ps(vi2x79BD, vk20));

      vo0p0 = _mm_add_ps(vo0p0, vo0p1);

      __m128 vo0 = _mm_max_ps(vo0p0, vmin);

      vo0 = _mm_min_ps(vo0, vmax);

      if (w == 7 * sizeof(float)) {
        _mm_storeu_ps(o0, vo0);
        o0 += 4;
      } else {
        w += 1 * sizeof(float);
        if (w & (4 * sizeof(float))) {
          _mm_storel_pi((__m64*) o0, vo0);
          o0 += 2;

          vo0 = _mm_movehl_ps(vo0, vo0);
        }
        if (w & (2 * sizeof(float))) {
          _mm_store_ss(o0, vo0);
          o0 += 1;
        }
      }
    }

    i0 = (const float*) ((uintptr_t) i2 - input_decrement);
    i1 = (const float*) ((uintptr_t) i0 + input_width);
    i2 = (const float*) ((uintptr_t) i1 + input_width);


    output_height -= 1;
    padded_input_height -= 2;
  } while (output_height != 0);
}
