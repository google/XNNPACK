// Copyright 2021 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <string.h>

#include <xnnpack/common.h>
#include <xnnpack/math.h>
#include <xnnpack/transpose.h>

void xnn_xx_transposev_ukernel__1x1_scalar_memcpy(
    const void* input,
    void* output,
    size_t input_row_stride,
    size_t output_row_stride,
    size_t input_element_stride,
    size_t output_element_stride,
    size_t element_size,
    size_t block_width,
    size_t block_height)
{
  const size_t input_reset = input_element_stride - block_height * input_row_stride;
  const size_t output_reset = output_row_stride - block_height * output_element_stride;

  const void* i = (const void*) input;
  void* o = (void*) output;

  do {
    size_t bh = block_height;
    for (; bh >= 1; bh -= 1) {
      memcpy(o, i, element_size);
      i = (const void*) ((uintptr_t) i + input_row_stride);
      o = (void*) ((uintptr_t) o + output_element_stride);
    }

    i = (const void*) ((uintptr_t) i + input_reset);
    o = (void*) ((uintptr_t) o + output_reset);
    block_width -= 1;
  } while (block_width != 0);
}
