// Copyright 2021 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <xmmintrin.h>

#include <assert.h>

#include "xnnpack/common.h"
#include "xnnpack/math.h"
#include "xnnpack/transpose.h"

void xnn_x32_transposec_ukernel__4x4_sse(
    const uint32_t* input,
    uint32_t* output,
    size_t input_stride,
    size_t output_stride,
    size_t block_width,
    size_t block_height) XNN_OOB_READS
{
  assert(block_width == 1 || output_stride >= block_height * sizeof(uint32_t));
  assert(block_height == 1 || input_stride >= block_width * sizeof(uint32_t));

  const size_t tile_height = 4;
  const size_t tile_width = 4;
  const size_t tile_wbytes = tile_width * sizeof(float);
  const size_t input_vreset = tile_wbytes - round_down_po2(block_height, tile_height) * input_stride;
  const size_t output_vreset = tile_height * output_stride - round_down_po2(block_height, 2) * sizeof(uint32_t);
  const size_t input_offset = tile_height * input_stride;

  const float* i0 = (const float*) input;
  const float* i1 = (const float*) ((uintptr_t) i0 + input_stride);
  const float* i2 = (const float*) ((uintptr_t) i1 + input_stride);
  const float* i3 = (const float*) ((uintptr_t) i2 + input_stride);

  float* o0 = (float*) output;
  float* o1 = (float*) ((uintptr_t) o0 + output_stride);
  float* o2 = (float*) ((uintptr_t) o1 + output_stride);
  float* o3 = (float*) ((uintptr_t) o2 + output_stride);

  do {
    if XNN_UNPREDICTABLE(block_width < 2) {
      o1 = o0;
    }
    if XNN_UNPREDICTABLE(block_width <= 2) {
      o2 = o0;
    }
    if XNN_UNPREDICTABLE(block_width < 4) {
      o3 = o0;
    }
    size_t bh = block_height;
    for (; bh >= 4; bh -= 4) {
      __m128 v0 = _mm_loadu_ps(i0);
      i0 = (const float*) ((uintptr_t) i0 + input_offset);
      __m128 v1 = _mm_loadu_ps(i1);
      i1 = (const float*) ((uintptr_t) i1 + input_offset);
      __m128 v2 = _mm_loadu_ps(i2);
      i2 = (const float*) ((uintptr_t) i2 + input_offset);
      __m128 v3 = _mm_loadu_ps(i3);
      i3 = (const float*) ((uintptr_t) i3 + input_offset);

      _MM_TRANSPOSE4_PS(v0, v1, v2, v3);

      _mm_storeu_ps(o3, v3);
      o3 = (float*) ((uintptr_t) o3 + tile_wbytes);
      _mm_storeu_ps(o2, v2);
      o2 = (float*) ((uintptr_t) o2 + tile_wbytes);
      _mm_storeu_ps(o1, v1);
      o1 = (float*) ((uintptr_t) o1 + tile_wbytes);
      _mm_storeu_ps(o0, v0);
      o0 = (float*) ((uintptr_t) o0 + tile_wbytes);
    }

    if (bh != 0) {
      if XNN_UNPREDICTABLE(bh <= 2) {
        i2 = i0;
      }
      if XNN_UNPREDICTABLE(bh < 2) {
        i1 = i0;
      }
      __m128 v0 = _mm_loadu_ps(i0);
      __m128 v1 = _mm_loadu_ps(i1);
      __m128 v2 = _mm_loadu_ps(i2);
      __m128 v3 = _mm_setzero_ps();

      _MM_TRANSPOSE4_PS(v0, v1, v2, v3);

      if (bh & 2) {
        _mm_storel_pi((__m64*) o3, v3);
        o3 += 2;
        _mm_storel_pi((__m64*) o2, v2);
        o2 += 2;
        _mm_storel_pi((__m64*) o1, v1);
        o1 += 2;
        _mm_storel_pi((__m64*) o0, v0);
        o0 += 2;
        v0 = _mm_movehl_ps(v0, v0);
        v1 = _mm_movehl_ps(v1, v1);
        v2 = _mm_movehl_ps(v2, v2);
        v3 = _mm_movehl_ps(v3, v3);
      }
      if (bh & 1) {
        _mm_store_ss(o3, v3);
        _mm_store_ss(o2, v2);
        _mm_store_ss(o1, v1);
        _mm_store_ss(o0, v0);
      }
    }
    i0 = (const float*) ((uintptr_t) i0 + input_vreset);
    i1 = (const float*) ((uintptr_t) i0 + input_stride);
    i2 = (const float*) ((uintptr_t) i1 + input_stride);
    i3 = (const float*) ((uintptr_t) i2 + input_stride);
    o0 = (float*) ((uintptr_t) o0 + output_vreset);
    o1 = (float*) ((uintptr_t) o1 + output_vreset);
    o2 = (float*) ((uintptr_t) o2 + output_vreset);
    o3 = (float*) ((uintptr_t) o3 + output_vreset);
    block_width = doz(block_width, tile_width);
  } while (block_width != 0);
}
