// Copyright 2021 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <stdio.h>
#include <xmmintrin.h>

#include <assert.h>

#include <xnnpack/common.h>
#include <xnnpack/transpose.h>

void xnn_x32_transpose_ukernel__4x4_sse(
    const uint32_t *input,
    uint32_t * output,
    size_t input_stride,
    size_t output_stride,
    size_t block_width,
    size_t block_height){
  const size_t ukernel_size = 4;
  const size_t ukernel_bytes = ukernel_size * sizeof(uint32_t);
  assert(block_height >= ukernel_size);
  assert(block_width >= ukernel_bytes);
  assert(output_stride >= block_height * sizeof(uint32_t));
  assert(input_stride >= block_width);
  size_t w_size = block_width;
  const size_t input_reset = ukernel_bytes - block_height * input_stride;
  const size_t output_col_reset = ukernel_bytes - 3 * output_stride;
  const size_t output_row_reset = ukernel_size * output_stride - block_height * sizeof(uint32_t);
  const float *input_f = (const float*)input;
  float *output_f = (float*)output;
  size_t h_size = block_height;
  for (; w_size >= ukernel_bytes; w_size -= ukernel_bytes) {
    h_size = block_height;
inner_loop:
    for (; h_size >= ukernel_size; h_size -= ukernel_size) {
      __m128 v0 = _mm_loadu_ps(input_f);
      input_f = (float*) ((uintptr_t) input_f + input_stride);
      __m128 v1 = _mm_loadu_ps(input_f);
      input_f = (float*) ((uintptr_t) input_f + input_stride);
      __m128 v2 = _mm_loadu_ps(input_f);
      input_f = (float*) ((uintptr_t) input_f + input_stride);
      __m128 v3 = _mm_loadu_ps(input_f);
      input_f = (float*) ((uintptr_t) input_f + input_stride);
      _MM_TRANSPOSE4_PS(v0, v1, v2, v3);
      _mm_storeu_ps(output_f, v0);
      output_f = (float*) ((uintptr_t) output_f + output_stride);
      _mm_storeu_ps(output_f, v1);
      output_f = (float*) ((uintptr_t) output_f + output_stride);
      _mm_storeu_ps(output_f, v2);
      output_f = (float*) ((uintptr_t) output_f + output_stride);
      _mm_storeu_ps(output_f, v3);
      output_f = (float*) ((uintptr_t) output_f + output_col_reset);
    }
    if XNN_UNLIKELY(h_size != 0) {
      int diff = h_size - ukernel_size;
      input_f = (float*) ((uintptr_t) input_f + diff * input_stride);
      output_f = (float*) ((uintptr_t) output_f + diff * sizeof(uint32_t));
      h_size = ukernel_size;
      goto inner_loop;
    }
    input_f = (float*) ((uintptr_t) input_f + input_reset);
    output_f = (float*) ((uintptr_t) output_f + output_row_reset);
  }
  if XNN_UNLIKELY(w_size != 0) {
    // Shift input and output pointers back.
    const size_t h_address_increment = ukernel_bytes - w_size;
    input_f = (float*) ((uintptr_t) input_f -  h_address_increment);
    output_f = (float*) ((uintptr_t) output_f -  h_address_increment * (output_stride >> 2));// sizeof(uint32_t))
    h_size = block_height;
    w_size = ukernel_bytes;
    goto inner_loop;
  }
}
