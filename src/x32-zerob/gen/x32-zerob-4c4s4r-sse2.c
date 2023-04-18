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

void xnn_x32_zerob_gemm_ukernel_4c4s4r__sse2(
  size_t groups,
  size_t channels,
  uint32_t* packed_weights,
  size_t channel_tile_stride,
  size_t channel_subtile_stride,
  const union xnn_x32_packb_params* params)
{
  assert(groups != 0);
  assert(channels != 0);
  assert(packed_weights != NULL);

  float* w = (float*) packed_weights;
  __m128 vzero = _mm_setzero_ps();

  do {
    size_t c = channels;

    // Channel tile loop.
    for (; c >= 4; c -= 4) {
      _mm_store_ps(w, vzero);
      w = (float*) ((uintptr_t) w + channel_tile_stride);
    }

    float* prev_w = w;
    for (; c >= 4; c -= 4) {
      _mm_store_ps(w, vzero);
      w += 4;
    }

    if (c != 0) {
      _mm_store_ps(w, vzero);
    }

    w = (float*) ((uintptr_t) prev_w + channel_tile_stride);
  } while (--groups != 0);
}
