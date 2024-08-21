// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <xmmintrin.h>

#include "xnnpack/maxpool.h"


void xnn_f32_maxpool_minmax_ukernel_9p8x__sse_c4(
    size_t output_pixels,
    size_t kernel_elements,
    size_t channels,
    const float** input,
    size_t input_offset,
    float* output,
    size_t input_increment,
    size_t output_increment,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(output_pixels != 0);
  assert(kernel_elements != 0);
  assert(channels != 0);

  const __m128 voutput_max = _mm_set1_ps(params->sse.max);
  const __m128 voutput_min = _mm_set1_ps(params->sse.min);
  XNN_FORCE_REALIZATION(voutput_max);
  XNN_FORCE_REALIZATION(voutput_min);

  do {
    float* o = output;
    {
      const float* i0 = *input++;
      const float* i1 = *input++;
      const float* i2 = *input++;
      const float* i3 = *input++;
      const float* i4 = *input++;
      const float* i5 = *input++;
      const float* i6 = *input++;
      const float* i7 = *input++;
      const float* i8 = *input++;
      i0 = (const float*) ((uintptr_t) i0 + input_offset);
      i1 = (const float*) ((uintptr_t) i1 + input_offset);
      i2 = (const float*) ((uintptr_t) i2 + input_offset);
      i3 = (const float*) ((uintptr_t) i3 + input_offset);
      i4 = (const float*) ((uintptr_t) i4 + input_offset);
      i5 = (const float*) ((uintptr_t) i5 + input_offset);
      i6 = (const float*) ((uintptr_t) i6 + input_offset);
      i7 = (const float*) ((uintptr_t) i7 + input_offset);
      i8 = (const float*) ((uintptr_t) i8 + input_offset);
      if (kernel_elements < 2) {
        i1 = i0;
      }
      if (kernel_elements <= 2) {
        i2 = i0;
      }
      if (kernel_elements < 4) {
        i3 = i0;
      }
      if (kernel_elements <= 4) {
        i4 = i0;
      }
      if (kernel_elements < 6) {
        i5 = i0;
      }
      if (kernel_elements <= 6) {
        i6 = i0;
      }
      if (kernel_elements < 8) {
        i7 = i0;
      }
      if (kernel_elements <= 8) {
        i8 = i0;
      }

      size_t c = channels;
      for (; c >= 4; c -= 4) {
        const __m128 vi0 = _mm_loadu_ps(i0);
        i0 += 4;
        const __m128 vi1 = _mm_loadu_ps(i1);
        i1 += 4;
        const __m128 vi2 = _mm_loadu_ps(i2);
        i2 += 4;
        const __m128 vi3 = _mm_loadu_ps(i3);
        i3 += 4;
        const __m128 vi4 = _mm_loadu_ps(i4);
        i4 += 4;
        const __m128 vi5 = _mm_loadu_ps(i5);
        i5 += 4;
        const __m128 vi6 = _mm_loadu_ps(i6);
        i6 += 4;
        const __m128 vi7 = _mm_loadu_ps(i7);
        i7 += 4;
        const __m128 vi8 = _mm_loadu_ps(i8);
        i8 += 4;

        const __m128 vmax018 = _mm_max_ps(_mm_max_ps(vi0, vi1), vi8);
        const __m128 vmax23 = _mm_max_ps(vi2, vi3);
        const __m128 vmax45 = _mm_max_ps(vi4, vi5);
        const __m128 vmax67 = _mm_max_ps(vi6, vi7);

        const __m128 vmax2345 = _mm_max_ps(vmax23, vmax45);
        const __m128 vmax01678 = _mm_max_ps(vmax018, vmax67);
        const __m128 vmax = _mm_max_ps(vmax2345, vmax01678);
        const __m128 vout = _mm_max_ps(_mm_min_ps(vmax, voutput_max), voutput_min);

        _mm_storeu_ps(o, vout);
        o += 4;
      }
      if (c != 0) {
        const __m128 vi0 = _mm_loadu_ps(i0);
        i0 += 4;
        const __m128 vi1 = _mm_loadu_ps(i1);
        i1 += 4;
        const __m128 vi2 = _mm_loadu_ps(i2);
        i2 += 4;
        const __m128 vi3 = _mm_loadu_ps(i3);
        i3 += 4;
        const __m128 vi4 = _mm_loadu_ps(i4);
        i4 += 4;
        const __m128 vi5 = _mm_loadu_ps(i5);
        i5 += 4;
        const __m128 vi6 = _mm_loadu_ps(i6);
        i6 += 4;
        const __m128 vi7 = _mm_loadu_ps(i7);
        i7 += 4;
        const __m128 vi8 = _mm_loadu_ps(i8);
        i8 += 4;

        const __m128 vmax018 = _mm_max_ps(_mm_max_ps(vi0, vi1), vi8);
        const __m128 vmax23 = _mm_max_ps(vi2, vi3);
        const __m128 vmax45 = _mm_max_ps(vi4, vi5);
        const __m128 vmax67 = _mm_max_ps(vi6, vi7);

        const __m128 vmax2345 = _mm_max_ps(vmax23, vmax45);
        const __m128 vmax01678 = _mm_max_ps(vmax018, vmax67);
        const __m128 vmax = _mm_max_ps(vmax2345, vmax01678);
        __m128 vout = _mm_max_ps(_mm_min_ps(vmax, voutput_max), voutput_min);

        if (c & 2) {
          _mm_storel_pi((__m64*) o, vout);
          o += 2;
          vout = _mm_movehl_ps(vout, vout);
        }
        if (c & 1) {
          _mm_store_ss(o, vout);
          o += 1;
        }
      }
    }

    for (ptrdiff_t k = (ptrdiff_t) kernel_elements - 9; k > 0; k -= 8) {
      const float* i0 = *input++;
      const float* i1 = *input++;
      const float* i2 = *input++;
      const float* i3 = *input++;
      const float* i4 = *input++;
      const float* i5 = *input++;
      const float* i6 = *input++;
      const float* i7 = *input++;
      i0 = (const float*) ((uintptr_t) i0 + input_offset);
      i1 = (const float*) ((uintptr_t) i1 + input_offset);
      i2 = (const float*) ((uintptr_t) i2 + input_offset);
      i3 = (const float*) ((uintptr_t) i3 + input_offset);
      i4 = (const float*) ((uintptr_t) i4 + input_offset);
      i5 = (const float*) ((uintptr_t) i5 + input_offset);
      i6 = (const float*) ((uintptr_t) i6 + input_offset);
      i7 = (const float*) ((uintptr_t) i7 + input_offset);
      if (k < 2) {
        i1 = i0;
      }
      if (k <= 2) {
        i2 = i0;
      }
      if (k < 4) {
        i3 = i0;
      }
      if (k <= 4) {
        i4 = i0;
      }
      if (k < 6) {
        i5 = i0;
      }
      if (k <= 6) {
        i6 = i0;
      }
      if (k < 8) {
        i7 = i0;
      }

      o = output;
      size_t c = channels;
      for (; c >= 4; c -= 4) {
        const __m128 vi0 = _mm_loadu_ps(i0);
        i0 += 4;
        const __m128 vi1 = _mm_loadu_ps(i1);
        i1 += 4;
        const __m128 vi2 = _mm_loadu_ps(i2);
        i2 += 4;
        const __m128 vi3 = _mm_loadu_ps(i3);
        i3 += 4;
        const __m128 vi4 = _mm_loadu_ps(i4);
        i4 += 4;
        const __m128 vi5 = _mm_loadu_ps(i5);
        i5 += 4;
        const __m128 vi6 = _mm_loadu_ps(i6);
        i6 += 4;
        const __m128 vi7 = _mm_loadu_ps(i7);
        i7 += 4;
        const __m128 vo = _mm_loadu_ps(o);

        const __m128 vmax01 = _mm_max_ps(_mm_max_ps(vi0, vi1), vo);
        const __m128 vmax23 = _mm_max_ps(vi2, vi3);
        const __m128 vmax45 = _mm_max_ps(vi4, vi5);
        const __m128 vmax67 = _mm_max_ps(vi6, vi7);

        const __m128 vmax2345 = _mm_max_ps(vmax23, vmax45);
        const __m128 vmax0167 = _mm_max_ps(vmax01, vmax67);
        const __m128 vmax = _mm_max_ps(vmax2345, vmax0167);
        const __m128 vout = _mm_max_ps(_mm_min_ps(vmax, voutput_max), voutput_min);

        _mm_storeu_ps(o, vout);
        o += 4;
      }
      if (c != 0) {
        const __m128 vi0 = _mm_loadu_ps(i0);
        const __m128 vi1 = _mm_loadu_ps(i1);
        const __m128 vi2 = _mm_loadu_ps(i2);
        const __m128 vi3 = _mm_loadu_ps(i3);
        const __m128 vi4 = _mm_loadu_ps(i4);
        const __m128 vi5 = _mm_loadu_ps(i5);
        const __m128 vi6 = _mm_loadu_ps(i6);
        const __m128 vi7 = _mm_loadu_ps(i7);
        const __m128 vo = _mm_loadu_ps(o);

        const __m128 vmax01 = _mm_max_ps(_mm_max_ps(vi0, vi1), vo);
        const __m128 vmax23 = _mm_max_ps(vi2, vi3);
        const __m128 vmax45 = _mm_max_ps(vi4, vi5);
        const __m128 vmax67 = _mm_max_ps(vi6, vi7);

        const __m128 vmax2345 = _mm_max_ps(vmax23, vmax45);
        const __m128 vmax0167 = _mm_max_ps(vmax01, vmax67);
        const __m128 vmax = _mm_max_ps(vmax2345, vmax0167);
        __m128 vout = _mm_max_ps(_mm_min_ps(vmax, voutput_max), voutput_min);

        if (c & 2) {
          _mm_storel_pi((__m64*) o, vout);
          o += 2;
          vout = _mm_movehl_ps(vout, vout);
        }
        if (c & 1) {
          _mm_store_ss(o, vout);
          o += 1;
        }
      }
    }
    input = (const float**) ((uintptr_t) input + input_increment);
    output = (float*) ((uintptr_t) o + output_increment);
  } while (--output_pixels != 0);
}
