// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <xmmintrin.h>

#include <xnnpack/maxpool.h>


void xnn_f32_maxpool_ukernel_9p8q__sse(
    size_t n,
    size_t ks,
    size_t kc,
    const float** input,
    float* output,
    size_t input_increment,
    size_t output_increment,
    const union xnn_f32_output_params params[restrict static 1])
{
  assert(n != 0);
  assert(ks != 0);
  assert(kc != 0);

  const __m128 voutput_max = _mm_load_ps(params->sse.max);
  const __m128 voutput_min = _mm_load_ps(params->sse.min);
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
      if (ks < 2) {
        i1 = i0;
      }
      if (ks <= 2) {
        i2 = i0;
      }
      if (ks < 4) {
        i3 = i0;
      }
      if (ks <= 4) {
        i4 = i0;
      }
      if (ks < 6) {
        i5 = i0;
      }
      if (ks <= 6) {
        i6 = i0;
      }
      if (ks < 8) {
        i7 = i0;
      }
      if (ks <= 8) {
        i8 = i0;
      }

      size_t k = kc;
      for (; k >= 4; k -= 4) {
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
      if (k != 0) {
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

        if (k & 2) {
          _mm_storel_pi((__m64*) o, vout);
          o += 2;
          vout = _mm_movehl_ps(vout, vout);
        }
        if (k & 1) {
          _mm_store_ss(o, vout);
          o += 1;
        }
      }
    }

    for (ptrdiff_t m = (ptrdiff_t) ks - 9; m > 0; m -= 8) {
      const float* i0 = *input++;
      const float* i1 = *input++;
      const float* i2 = *input++;
      const float* i3 = *input++;
      const float* i4 = *input++;
      const float* i5 = *input++;
      const float* i6 = *input++;
      const float* i7 = *input++;
      if (m < 2) {
        i1 = i0;
      }
      if (m <= 2) {
        i2 = i0;
      }
      if (m < 4) {
        i3 = i0;
      }
      if (m <= 4) {
        i4 = i0;
      }
      if (m < 6) {
        i5 = i0;
      }
      if (m <= 6) {
        i6 = i0;
      }
      if (m < 8) {
        i7 = i0;
      }

      o = output;
      size_t k = kc;
      for (; k >= 4; k -= 4) {
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
      if (k != 0) {
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

        if (k & 2) {
          _mm_storel_pi((__m64*) o, vout);
          o += 2;
          vout = _mm_movehl_ps(vout, vout);
        }
        if (k & 1) {
          _mm_store_ss(o, vout);
          o += 1;
        }
      }
    }
    input = (const float**) ((uintptr_t) input + input_increment);
    output = (float*) ((uintptr_t) o + output_increment);
  } while (--n != 0);
}
