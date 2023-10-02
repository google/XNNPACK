// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <xmmintrin.h>

#include <xnnpack/gavgpool.h>
#include <xnnpack/math.h>


void xnn_f32_gavgpool_cw_ukernel__sse_u4(
    size_t elements,
    size_t channels,
    const float* input,
    float* output,
    const union xnn_f32_gavgpool_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(elements != 0);
  assert(elements % sizeof(float) == 0);
  assert(channels != 0);

  const float* i0 = input;
  const float* i1 = (const float*) ((uintptr_t) i0 + elements);
  const float* i2 = (const float*) ((uintptr_t) i1 + elements);
  const float* i3 = (const float*) ((uintptr_t) i2 + elements);

  const __m128 vmask = _mm_load_ps((const float*) params->sse.mask);
  const __m128 vmultiplier = _mm_load_ps(params->sse.multiplier);
  const __m128 voutput_min = _mm_load_ps(params->sse.output_min);
  const __m128 voutput_max = _mm_load_ps(params->sse.output_max);

  while (channels >= 4) {
    __m128 vsum0 = _mm_setzero_ps();
    __m128 vsum1 = _mm_setzero_ps();
    __m128 vsum2 = _mm_setzero_ps();
    __m128 vsum3 = _mm_setzero_ps();
    size_t n = elements;
    while (n >= 4 * sizeof(float)) {
      const __m128 vi0 = _mm_loadu_ps(i0);
      i0 += 4;
      const __m128 vi1 = _mm_loadu_ps(i1);
      i1 += 4;
      const __m128 vi2 = _mm_loadu_ps(i2);
      i2 += 4;
      const __m128 vi3 = _mm_loadu_ps(i3);
      i3 += 4;

      vsum0 = _mm_add_ps(vsum0, vi0);
      vsum1 = _mm_add_ps(vsum1, vi1);
      vsum2 = _mm_add_ps(vsum2, vi2);
      vsum3 = _mm_add_ps(vsum3, vi3);
      n -= 4 * sizeof(float);
    }

    if XNN_UNLIKELY(n != 0) {
      const __m128 vi0 = _mm_and_ps(_mm_loadu_ps(i0), vmask);
      i0 = (const float*) ((uintptr_t) i0 + n);
      const __m128 vi1 = _mm_and_ps(_mm_loadu_ps(i1), vmask);
      i1 = (const float*) ((uintptr_t) i1 + n);
      const __m128 vi2 = _mm_and_ps(_mm_loadu_ps(i2), vmask);
      i2 = (const float*) ((uintptr_t) i2 + n);
      const __m128 vi3 = _mm_and_ps(_mm_loadu_ps(i3), vmask);
      i3 = (const float*) ((uintptr_t) i3 + n);

      vsum0 = _mm_add_ps(vsum0, vi0);
      vsum1 = _mm_add_ps(vsum1, vi1);
      vsum2 = _mm_add_ps(vsum2, vi2);
      vsum3 = _mm_add_ps(vsum3, vi3);
    }

    // Having exactly 4 rows makes this work out nicely as we end up with
    // the 4 totals in 4 different lanes of the same vector.
    const __m128 vsum01 = _mm_add_ps(_mm_unpacklo_ps(vsum0, vsum1), _mm_unpackhi_ps(vsum0, vsum1));
    const __m128 vsum23 = _mm_add_ps(_mm_unpacklo_ps(vsum2, vsum3), _mm_unpackhi_ps(vsum2, vsum3));
    const __m128 vsum = _mm_add_ps(_mm_movelh_ps(vsum01, vsum23), _mm_movehl_ps(vsum23, vsum01));
    __m128 vout = _mm_mul_ps(vsum, vmultiplier);

    vout = _mm_max_ps(vout, voutput_min);
    vout = _mm_min_ps(vout, voutput_max);

    _mm_storeu_ps(output, vout);
    output += 4;
    i0 = i3;
    i1 = (const float*) ((uintptr_t) i0 + elements);
    i2 = (const float*) ((uintptr_t) i1 + elements);
    i3 = (const float*) ((uintptr_t) i2 + elements);
    channels -= 4;
  }

  while (channels != 0) {
    __m128 vsum = _mm_setzero_ps();
    size_t n = elements;
    while (n >= 4 * sizeof(float)) {
      const __m128 vi0 = _mm_loadu_ps(i0);
      i0 += 4;
      vsum = _mm_add_ps(vsum, vi0);
      n -= 4 * sizeof(float);
    }

    if XNN_UNLIKELY(n != 0) {
      __m128 vi0 = _mm_and_ps(_mm_loadu_ps(i0), vmask);
      i0 = (const float*) ((uintptr_t) i0 + n);
      vsum = _mm_add_ps(vsum, vi0);
    }

    vsum = _mm_add_ps(vsum, _mm_movehl_ps(vsum, vsum));
    vsum = _mm_add_ss(vsum, _mm_shuffle_ps(vsum, vsum, _MM_SHUFFLE(3, 2, 1, 1)));

    __m128 vout = _mm_mul_ss(vsum, vmultiplier);

    vout = _mm_max_ss(vout, voutput_min);
    vout = _mm_min_ss(vout, voutput_max);

    _mm_store_ss(output, vout);
    output += 1;
    channels -= 1;
  }
}
