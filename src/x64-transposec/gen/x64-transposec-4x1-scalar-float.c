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

void xnn_x64_transposec_ukernel__4x1_scalar_float(
    const uint64_t *input,
    uint64_t * output,
    size_t input_stride,
    size_t output_stride,
    size_t block_width,
    size_t block_height) XNN_OOB_READS
{
  assert(block_width == 1 || output_stride >= block_height * sizeof(double));
  assert(block_height == 1 || input_stride >= block_width * sizeof(double));

  const size_t tile_height = 4;
  const size_t tile_width = 1;
  const size_t tile_wbytes = tile_width * sizeof(double);
  const size_t input_reset = tile_wbytes - round_down_po2(block_height, tile_height) * input_stride;
  const size_t output_reset = tile_width * output_stride - round_down_po2(block_height, 2) * sizeof(double);
  const size_t input_offset = tile_height * input_stride;

  const double* i0 = (const double*) input;
  const double* i1 = (const double*) ((uintptr_t) i0 + input_stride);
  const double* i2 = (const double*) ((uintptr_t) i1 + input_stride);
  const double* i3 = (const double*) ((uintptr_t) i2 + input_stride);

  double* o0 = (double*) output;

  do {
    size_t bh = block_height;
    for (; bh >= 4; bh -= 4) {
      *o0++ = i0[0];
      *o0++ = i1[0];
      *o0++ = i2[0];
      *o0++ = i3[0];
      i0 = (const double*) ((uintptr_t) i0 + input_offset);
      i1 = (const double*) ((uintptr_t) i1 + input_offset);
      i2 = (const double*) ((uintptr_t) i2 + input_offset);
      i3 = (const double*) ((uintptr_t) i3 + input_offset);
    }
    const double* i = i0;
    if (bh & 2) {
      o0[0] = i0[0];
      o0[1] = i1[0];
      o0 += 2;
      i = i2;
    }
    if (bh & 1) {
      o0[0] = i[0];
    }

    i0 = (const double*) ((uintptr_t) i0 + input_reset);
    i1 = (const double*) ((uintptr_t) i0 + input_stride);
    i2 = (const double*) ((uintptr_t) i1 + input_stride);
    i3 = (const double*) ((uintptr_t) i2 + input_stride);
    o0 = (double*) ((uintptr_t) o0 + output_reset);
    block_width = doz(block_width, tile_width);
  } while (block_width != 0);
}
