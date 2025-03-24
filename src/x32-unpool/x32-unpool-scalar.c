// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include "src/xnnpack/unpool.h"


void xnn_x32_unpool_ukernel__scalar(
    size_t kernel_elements,
    size_t channels,
    uint32_t fill,
    const uint32_t* input,
    const uint32_t* index,
    uint32_t** output,
    size_t output_offset)
{
  // Pre-initialize outputs with constant.
  uint32_t** os = output;
  do {
    uint32_t* o = (uint32_t*) ((uintptr_t) *os++ + output_offset);
    size_t c = channels;
    do {
      *o++ = fill;
    } while (--c != 0);
  } while (--kernel_elements != 0);

  // Copy indexed elements to output.
  size_t offset = output_offset;
  do {
    const uint32_t i = *index++;
    *((uint32_t*) ((uintptr_t) output[i] + offset)) = *input++;
    offset += sizeof(uint32_t);
  } while (--channels != 0);
}
