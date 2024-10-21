// Auto-generated file. Do not edit!
//   Template: src/f32-prelu/sse.c.in
//   Generator: tools/xngen
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <xmmintrin.h>

#include "xnnpack/math.h"
#include "xnnpack/prelu.h"


void xnn_f32_prelu_ukernel__sse_2x8(
    size_t rows,
    size_t channels,
    const float* restrict input,
    size_t input_stride,
    const float* restrict weights,
    float* restrict output,
    size_t output_stride) XNN_OOB_READS
{
  assert(rows != 0);
  assert(channels != 0);
  assert(channels % sizeof(float) == 0);

  const float* i0 = input;
  float* o0 = output;
  const float* i1 = (const float*) ((uintptr_t) i0 + input_stride);
  float* o1 = (float*) ((uintptr_t) o0 + output_stride);

  const size_t input_increment = input_stride * 2 - channels;
  const size_t output_increment = output_stride * 2 - channels;

  const __m128 vzero = _mm_setzero_ps();
  do {
    if XNN_UNPREDICTABLE(rows < 2) {
      i1 = i0;
      o1 = o0;
    }

    const float* w = weights;
    size_t c = channels;
    for (; c >= 8 * sizeof(float); c -= 8 * sizeof(float)) {
      const __m128 vw0123 = _mm_load_ps(w);
      const __m128 vw4567 = _mm_load_ps(w + 4);
      w += 8;

      __m128 vi0x0123 = _mm_loadu_ps(i0);
      __m128 vi0x4567 = _mm_loadu_ps(i0 + 4);
      i0 += 8;
      __m128 vi1x0123 = _mm_loadu_ps(i1);
      __m128 vi1x4567 = _mm_loadu_ps(i1 + 4);
      i1 += 8;

      __m128 vacc0x0123 = _mm_max_ps(_mm_setzero_ps(), vi0x0123);
      vi0x0123 = _mm_min_ps(vi0x0123, vzero);
      __m128 vacc0x4567 = _mm_max_ps(_mm_setzero_ps(), vi0x4567);
      vi0x4567 = _mm_min_ps(vi0x4567, vzero);
      __m128 vacc1x0123 = _mm_max_ps(_mm_setzero_ps(), vi1x0123);
      vi1x0123 = _mm_min_ps(vi1x0123, vzero);
      __m128 vacc1x4567 = _mm_max_ps(_mm_setzero_ps(), vi1x4567);
      vi1x4567 = _mm_min_ps(vi1x4567, vzero);

      vacc0x0123 = _mm_add_ps(vacc0x0123, _mm_mul_ps(vi0x0123, vw0123));
      vacc0x4567 = _mm_add_ps(vacc0x4567, _mm_mul_ps(vi0x4567, vw4567));
      vacc1x0123 = _mm_add_ps(vacc1x0123, _mm_mul_ps(vi1x0123, vw0123));
      vacc1x4567 = _mm_add_ps(vacc1x4567, _mm_mul_ps(vi1x4567, vw4567));

      _mm_storeu_ps(o0, vacc0x0123);
      _mm_storeu_ps(o0 + 4, vacc0x4567);
      o0 += 8;
      _mm_storeu_ps(o1, vacc1x0123);
      _mm_storeu_ps(o1 + 4, vacc1x4567);
      o1 += 8;
    }
    for (; c >= 4 * sizeof(float); c -= 4 * sizeof(float)) {
      const __m128 vw0123 = _mm_load_ps(w);
      w += 4;

      __m128 vi0x0123 = _mm_loadu_ps(i0);
      i0 += 4;
      __m128 vi1x0123 = _mm_loadu_ps(i1);
      i1 += 4;

      __m128 vacc0x0123 = _mm_max_ps(_mm_setzero_ps(), vi0x0123);
      vi0x0123 = _mm_min_ps(vi0x0123, vzero);
      __m128 vacc1x0123 = _mm_max_ps(_mm_setzero_ps(), vi1x0123);
      vi1x0123 = _mm_min_ps(vi1x0123, vzero);

      vacc0x0123 = _mm_add_ps(vacc0x0123, _mm_mul_ps(vi0x0123, vw0123));
      vacc1x0123 = _mm_add_ps(vacc1x0123, _mm_mul_ps(vi1x0123, vw0123));

      _mm_storeu_ps(o0, vacc0x0123);
      o0 += 4;
      _mm_storeu_ps(o1, vacc1x0123);
      o1 += 4;
    }
    if XNN_UNLIKELY(c != 0) {
      const __m128 vw0123 = _mm_load_ps(w);
      w = (const float*) ((uintptr_t) w + c);

      __m128 vi0x0123 = _mm_loadu_ps(i0);
      i0 = (const float*) ((uintptr_t) i0 + c);
      __m128 vi1x0123 = _mm_loadu_ps(i1);
      i1 = (const float*) ((uintptr_t) i1 + c);

      __m128 vacc0x0123 = _mm_max_ps(_mm_setzero_ps(), vi0x0123);
      vi0x0123 = _mm_min_ps(vi0x0123, vzero);
      __m128 vacc1x0123 = _mm_max_ps(_mm_setzero_ps(), vi1x0123);
      vi1x0123 = _mm_min_ps(vi1x0123, vzero);

      vacc0x0123 = _mm_add_ps(vacc0x0123, _mm_mul_ps(vi0x0123, vw0123));
      vacc1x0123 = _mm_add_ps(vacc1x0123, _mm_mul_ps(vi1x0123, vw0123));

      if (c & (2 * sizeof(float))) {
        _mm_storel_pi((__m64*) o0, vacc0x0123);
        _mm_storel_pi((__m64*) o1, vacc1x0123);

        vacc0x0123 = _mm_movehl_ps(vacc0x0123, vacc0x0123);
        vacc1x0123 = _mm_movehl_ps(vacc1x0123, vacc1x0123);

        o0 += 2;
        o1 += 2;
      }
      if (c & (1 * sizeof(float))) {
        _mm_store_ss(o0, vacc0x0123);
        _mm_store_ss(o1, vacc1x0123);

        o0 += 1;
        o1 += 1;
      }
    }
    i0 = (const float*) ((uintptr_t) i0 + input_increment);
    o0 = (float*) ((uintptr_t) o0 + output_increment);
    i1 = (const float*) ((uintptr_t) i1 + input_increment);
    o1 = (float*) ((uintptr_t) o1 + output_increment);
    rows = doz(rows, 2);
  } while (rows != 0);
}
