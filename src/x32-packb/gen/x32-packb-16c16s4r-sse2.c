// Auto-generated file. Do not edit!
//   Template: src/x32-packb/sse2.c.in
//   Generator: tools/xngen
//
// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>
#include <xmmintrin.h>

#include <xnnpack/math.h>
#include <xnnpack/packb.h>

void xnn_x32_packb_gemm_ukernel_16c16s4r__sse2(
  size_t groups,
  size_t channels,
  const uint32_t* bias,
  uint32_t* packed_weights,
  size_t channel_tile_stride,
  size_t channel_subtile_stride,
  const union xnn_x32_packb_params* params)
{
  assert(groups != 0);
  assert(channels != 0);
  assert(packed_weights != NULL);

  float* w = (float*) packed_weights;
  const float* b = (const float*) bias;

  do {
    size_t c = channels;

    // Channel tile loop.
    for (; c >= 16; c -= 16) {
      const __m128 vb0123 = _mm_loadu_ps(b);
      const __m128 vb4567 = _mm_loadu_ps(b + 4);
      const __m128 vb89AB = _mm_loadu_ps(b + 8);
      const __m128 vbCDEF = _mm_loadu_ps(b + 12);
      b += 16;

      _mm_store_ps(w, vb0123);
      _mm_store_ps(w + 4, vb4567);
      _mm_store_ps(w + 8, vb89AB);
      _mm_store_ps(w + 12, vbCDEF);
      w = (float*) ((uintptr_t) w + channel_tile_stride);
    }

    float* prev_w = w;
    for (; c >= 4; c -= 4) {
      const __m128 vb0123 = _mm_loadu_ps(b);
      b += 4;
      _mm_store_ps(w, vb0123);
      w += 4;
    }

    if (c & 2) {
      __m128 vb0123 = _mm_castsi128_ps(_mm_loadl_epi64((const __m128i*) b));
      b += 2;
      _mm_storel_pi((__m64*) w, vb0123);
      w += 2;
    }

    if (c & 1) {
      __m128 vb0123 = _mm_load_ss(b);
      b++;
      _mm_store_ss(w, vb0123);
      w++;
    }

    w = (float*) ((uintptr_t) prev_w + channel_tile_stride);
  } while (--groups != 0);
}
