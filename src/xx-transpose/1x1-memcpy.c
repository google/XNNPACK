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
    size_t block_height)
{
  const void* i = (const void*) input;
  void* o = (void*) output;

  do {
    memcpy(o, i, element_size);
    i = (const void*) ((uintptr_t) i + input_stride);
    o = (void*) ((uintptr_t) o + output_stride);
    block_height -= 1;
  } while (block_height != 0);
}
