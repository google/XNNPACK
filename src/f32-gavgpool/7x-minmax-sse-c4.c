// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <xmmintrin.h>

#include <xnnpack/gavgpool.h>


void xnn_f32_gavgpool_minmax_ukernel_7x__sse_c4(
    size_t rows,
    size_t channels,
    const float* input,
    size_t input_stride,
    const float* zero,
    float* output,
    const union xnn_f32_scaleminmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(rows != 0);
  assert(rows <= 7);
  assert(channels != 0);

  const float* i0 = input;
  const float* i1 = (const float*) ((uintptr_t) i0 + input_stride);
  if (rows < 2) {
    i1 = zero;
  }
  const float* i2 = (const float*) ((uintptr_t) i1 + input_stride);
  if (rows <= 2) {
    i2 = zero;
  }
  const float* i3 = (const float*) ((uintptr_t) i2 + input_stride);
  if (rows < 4) {
    i3 = zero;
  }
  const float* i4 = (const float*) ((uintptr_t) i3 + input_stride);
  if (rows <= 4) {
    i4 = zero;
  }
  const float* i5 = (const float*) ((uintptr_t) i4 + input_stride);
  if (rows < 6) {
    i5 = zero;
  }
  const float* i6 = (const float*) ((uintptr_t) i5 + input_stride);
  if (rows <= 6) {
    i6 = zero;
  }
  const __m128 vscale = _mm_load_ps(params->sse2.scale);
  const __m128 vmin = _mm_load_ps(params->sse2.min);
  const __m128 vmax = _mm_load_ps(params->sse2.max);

  while (channels >= 4) {
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

    __m128 vout = _mm_mul_ps(vsum, vscale);
    vout = _mm_max_ps(vout, vmin);
    vout = _mm_min_ps(vout, vmax);

    _mm_storeu_ps(output, vout);
    output += 4;

    channels -= 4;
  }
  if (channels != 0) {
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

    __m128 vout = _mm_mul_ps(vsum, vscale);
    vout = _mm_max_ps(vout, vmin);
    vout = _mm_min_ps(vout, vmax);

    if (channels & 2) {
      _mm_storel_pi((__m64*) output, vout);
      vout = _mm_movehl_ps(vout, vout);
      output += 2;
    }
    if (channels & 1) {
      _mm_store_ss(output, vout);
    }
  }
}
