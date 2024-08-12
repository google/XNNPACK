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

void xnn_x8_transposec_ukernel__2x1_scalar_int(
    const uint8_t *input,
    uint8_t * output,
    size_t input_stride,
    size_t output_stride,
    size_t block_width,
    size_t block_height) XNN_OOB_READS
{
  assert(block_width == 1 || output_stride >= block_height * sizeof(int8_t));
  assert(block_height == 1 || input_stride >= block_width * sizeof(int8_t));

  const size_t tile_height = 2;
  const size_t tile_width = 1;
  const size_t tile_wbytes = tile_width * sizeof(int8_t);
  const size_t input_reset = tile_wbytes - round_down_po2(block_height, tile_height) * input_stride;
  const size_t output_reset = tile_width * output_stride - round_down_po2(block_height, 2) * sizeof(int8_t);
  const size_t input_offset = tile_height * input_stride;

  const int8_t* i0 = (const int8_t*) input;
  const int8_t* i1 = (const int8_t*) ((uintptr_t) i0 + input_stride);

  int8_t* o0 = (int8_t*) output;

  do {
    size_t bh = block_height;
    for (; bh >= 2; bh -= 2) {
      *o0++ = i0[0];
      *o0++ = i1[0];
      i0 = (const int8_t*) ((uintptr_t) i0 + input_offset);
      i1 = (const int8_t*) ((uintptr_t) i1 + input_offset);
    }
    if (bh & 1) {
      o0[0] = i0[0];
    }

    i0 = (const int8_t*) ((uintptr_t) i0 + input_reset);
    i1 = (const int8_t*) ((uintptr_t) i0 + input_stride);
    o0 = (int8_t*) ((uintptr_t) o0 + output_reset);
    block_width = doz(block_width, tile_width);
  } while (block_width != 0);
}
