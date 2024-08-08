// Auto-generated file. Do not edit!
//   Template: src/x32-transposec/scalar.c.in
//   Generator: tools/xngen
//
// Copyright 2021 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.


#include <assert.h>

#include "xnnpack/common.h"
#include "xnnpack/math.h"
#include "xnnpack/transpose.h"

void xnn_x64_transposec_ukernel__1x2_scalar_int(
    const uint64_t *input,
    uint64_t * output,
    size_t input_stride,
    size_t output_stride,
    size_t block_width,
    size_t block_height) XNN_OOB_READS
{
  assert(block_width == 1 || output_stride >= block_height * sizeof(int64_t));
  assert(block_height == 1 || input_stride >= block_width * sizeof(int64_t));

  const size_t tile_height = 1;
  const size_t tile_width = 2;
  const size_t tile_wbytes = tile_width * sizeof(int64_t);
  const size_t input_reset = tile_wbytes - block_height * input_stride;
  const size_t output_reset = tile_width * output_stride - block_height * sizeof(int64_t);
  const size_t input_offset = tile_height * input_stride;

  const int64_t* i0 = (const int64_t*) input;

  int64_t* o0 = (int64_t*) output;
  int64_t* o1 = (int64_t*) ((uintptr_t) o0 + output_stride);

  do {
    if XNN_UNPREDICTABLE(block_width < 2) {
      o1 = o0;
    }
    size_t bh = block_height;
    for (; bh >= 1; bh -= 1) {
      *o1++ = i0[1];
      *o0++ = i0[0];
      i0 = (const int64_t*) ((uintptr_t) i0 + input_offset);
    }

    i0 = (const int64_t*) ((uintptr_t) i0 + input_reset);
    o0 = (int64_t*) ((uintptr_t) o0 + output_reset);
    o1 = (int64_t*) ((uintptr_t) o1 + output_reset);
    block_width = doz(block_width, tile_width);
  } while (block_width != 0);
}
