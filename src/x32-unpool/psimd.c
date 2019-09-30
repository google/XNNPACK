// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <psimd.h>

#include <xnnpack/pad.h>


void xnn_x32_unpool_ukernel__psimd(
    size_t p,
    size_t c,
    uint32_t f,
    const uint32_t* input,
    const uint32_t* index,
    uint32_t** output)
{
  // Pre-initialize outputs with constant.
  const psimd_u32 vf = psimd_splat_u32(f);
  uint32_t** os = output;
  do {
    uint32_t* o = *os++;
    size_t k = c;
    for (; k >= 4; k -= 4) {
      psimd_store_u32(o, vf);
      o += 4;
    }
    if (k != 0) {
      if (k & 2) {
        psimd_store2_u32(o, vf);
        o += 2;
      }
      if (k & 1) {
        psimd_store1_u32(o, vf);
      }
    }
  } while (--p != 0);

  // Copy indexed elements to output.
  size_t offset = 0;
  do {
    const uint32_t i = *index++;
    *((uint32_t*) ((uintptr_t) output[i] + offset)) = *input++;
    offset += sizeof(uint32_t);
  } while (--c != 0);
}
