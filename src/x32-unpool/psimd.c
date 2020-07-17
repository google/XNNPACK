// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <psimd.h>

#include <xnnpack/unpool.h>


void xnn_x32_unpool_ukernel__psimd(
    size_t kernel_elements,
    size_t channels,
    uint32_t fill,
    const uint32_t* input,
    const uint32_t* index,
    uint32_t** output)
{
  // Pre-initialize outputs with constant.
  const psimd_u32 vfill = psimd_splat_u32(fill);
  uint32_t** os = output;
  do {
    uint32_t* o = *os++;
    size_t c = channels;
    for (; c >= 4; c -= 4) {
      psimd_store_u32(o, vfill);
      o += 4;
    }
    if (c != 0) {
      if (c & 2) {
        psimd_store2_u32(o, vfill);
        o += 2;
      }
      if (c & 1) {
        psimd_store1_u32(o, vfill);
      }
    }
  } while (--kernel_elements != 0);

  // Copy indexed elements to output.
  size_t offset = 0;
  do {
    const uint32_t i = *index++;
    *((uint32_t*) ((uintptr_t) output[i] + offset)) = *input++;
    offset += sizeof(uint32_t);
  } while (--channels != 0);
}
