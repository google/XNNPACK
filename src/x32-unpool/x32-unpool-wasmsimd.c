// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <wasm_simd128.h>

#include "xnnpack/unpool.h"


void xnn_x32_unpool_ukernel__wasmsimd(
    size_t kernel_elements,
    size_t channels,
    uint32_t fill,
    const uint32_t* input,
    const uint32_t* index,
    uint32_t** output)
{
  // Pre-initialize outputs with constant.
  const v128_t vfill = wasm_i32x4_splat(fill);
  uint32_t** os = output;
  do {
    float* o = (float*) *os++;
    size_t c = channels;
    for (; c >= 4; c -= 4) {
      wasm_v128_store(o, vfill);
      o += 4;
    }
    if (c != 0) {
      if (c & 2) {
        wasm_v128_store64_lane(o, vfill, 0);
        o += 2;
      }
      if (c & 1) {
        wasm_v128_store32_lane(o, vfill, 0);
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
