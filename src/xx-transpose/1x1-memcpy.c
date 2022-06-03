// Copyright 2021 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <string.h>

#include <xnnpack/common.h>
#include <xnnpack/math.h>
#include <xnnpack/transpose.h>

void xnn_xx_transposev_ukernel__1x1_memcpy(
    const void* input,
    void* output,
    size_t input_stride,
    size_t output_stride,
    size_t element_size,
    size_t input_inner_size,
    size_t output_inner_size,
    size_t block_width,
    size_t block_height)
{
  const size_t tile_height = 1;
  const size_t tile_width = 1;
  const size_t tile_wbytes = tile_width * input_inner_size;
  const size_t input_reset = tile_wbytes - block_height * input_stride;
  const size_t output_reset = tile_width * output_stride - block_height * output_inner_size;
  const size_t input_offset = tile_height * input_stride;

  const void* i = (const void*) input;
  void* o = (void*) output;

  do {
    size_t bh = block_height;
    for (; bh >= 1; bh -= 1) {
      memcpy(o, i, element_size);
      i = (const void*) ((uintptr_t) i + input_offset);
      o = (void*) ((uintptr_t) o + output_inner_size);
    }

    i = (const void*) ((uintptr_t) i + input_reset);
    o = (void*) ((uintptr_t) o + output_reset);
    block_width -= 1;
  } while (block_width != 0);
}
