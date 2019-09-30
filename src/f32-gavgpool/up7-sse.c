// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <xmmintrin.h>

#include <xnnpack/gavgpool.h>


void xnn_f32_gavgpool_ukernel_up7__sse(
    size_t m,
    size_t n,
    const float* input,
    size_t input_stride,
    const float* zero,
    float* output,
    const union xnn_f32_avgpool_params params[restrict static 1])
{
  assert(m != 0);
  assert(m <= 7);
  assert(n != 0);

  const float* i0 = input;
  const float* i1 = (const float*) ((uintptr_t) i0 + input_stride);
  if (m < 2) {
    i1 = zero;
  }
  const float* i2 = (const float*) ((uintptr_t) i1 + input_stride);
  if (m <= 2) {
    i2 = zero;
  }
  const float* i3 = (const float*) ((uintptr_t) i2 + input_stride);
  if (m < 4) {
    i3 = zero;
  }
  const float* i4 = (const float*) ((uintptr_t) i3 + input_stride);
  if (m <= 4) {
    i4 = zero;
  }
  const float* i5 = (const float*) ((uintptr_t) i4 + input_stride);
  if (m < 6) {
    i5 = zero;
  }
  const float* i6 = (const float*) ((uintptr_t) i5 + input_stride);
  if (m <= 6) {
    i6 = zero;
  }
  const __m128 vmultiplier = _mm_load_ps(params->sse2.multiplier);
  const __m128 voutput_min = _mm_load_ps(params->sse2.output_min);
  const __m128 voutput_max = _mm_load_ps(params->sse2.output_max);

  while (n >= 4) {
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

    const __m128 vsum01 = _mm_add_ps(vi0, vi1);
    const __m128 vsum23 = _mm_add_ps(vi2, vi3);
    const __m128 vsum45 = _mm_add_ps(vi4, vi5);

    const __m128 vsum016 = _mm_add_ps(vsum01, vi6);
    const __m128 vsum2345 = _mm_add_ps(vsum23, vsum45);

    const __m128 vsum = _mm_add_ps(vsum016, vsum2345);

    __m128 vout = _mm_mul_ps(vsum, vmultiplier);
    vout = _mm_max_ps(vout, voutput_min);
    vout = _mm_min_ps(vout, voutput_max);

    _mm_storeu_ps(output, vout);
    output += 4;

    n -= 4;
  }
  if (n != 0) {
    const __m128 vi0 = _mm_loadu_ps(i0);
    const __m128 vi1 = _mm_loadu_ps(i1);
    const __m128 vi2 = _mm_loadu_ps(i2);
    const __m128 vi3 = _mm_loadu_ps(i3);
    const __m128 vi4 = _mm_loadu_ps(i4);
    const __m128 vi5 = _mm_loadu_ps(i5);
    const __m128 vi6 = _mm_loadu_ps(i6);

    const __m128 vsum01 = _mm_add_ps(vi0, vi1);
    const __m128 vsum23 = _mm_add_ps(vi2, vi3);
    const __m128 vsum45 = _mm_add_ps(vi4, vi5);

    const __m128 vsum016 = _mm_add_ps(vsum01, vi6);
    const __m128 vsum2345 = _mm_add_ps(vsum23, vsum45);

    const __m128 vsum = _mm_add_ps(vsum016, vsum2345);

    __m128 vout = _mm_mul_ps(vsum, vmultiplier);
    vout = _mm_max_ps(vout, voutput_min);
    vout = _mm_min_ps(vout, voutput_max);

    if (n & 2) {
      _mm_storel_pi((__m64*) output, vout);
      vout = _mm_movehl_ps(vout, vout);
      output += 2;
    }
    if (n & 1) {
      _mm_store_ss(output, vout);
    }
  }
}
