// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <wasm_simd128.h>

#include "xnnpack/fill.h"
#include "xnnpack/unaligned.h"


void xnn_xx_fill_ukernel__wasmsimd_u64(
    size_t rows,
    size_t channels,
    void* output,
    size_t output_stride,
    const uint32_t fill_pattern)
{
  assert(rows != 0);
  assert(channels != 0);

  const size_t output_increment = output_stride - channels;

  const v128_t vfill_pattern = wasm_i32x4_splat(fill_pattern);
  do {
    size_t c = channels;
    for (; c >= 64 * sizeof(uint8_t); c -= 64 * sizeof(uint8_t)) {
      wasm_v128_store(output, vfill_pattern);
      wasm_v128_store((uint8_t*) output + 16, vfill_pattern);
      wasm_v128_store((uint8_t*) output + 32, vfill_pattern);
      wasm_v128_store((uint8_t*) output + 48, vfill_pattern);
      output = ((uint8_t*) output + 64);
    }
    for (; c >= 16 * sizeof(uint8_t); c -= 16 * sizeof(uint8_t)) {
      wasm_v128_store(output, vfill_pattern);
      output = ((uint8_t*) output + 16);
    }
    if XNN_UNLIKELY(c != 0) {
      if XNN_LIKELY(c & (8 * sizeof(uint8_t))) {
        wasm_v128_store64_lane(output, vfill_pattern, 0);
        output = ((uint8_t*) output + 8);
      }
      if XNN_LIKELY(c & (4 * sizeof(uint8_t))) {
        unaligned_store_u32(output, fill_pattern);
        output = ((uint8_t*) output + 4);
      }
      uint32_t vfill_subpattern = fill_pattern;
      if XNN_LIKELY(c & (2 * sizeof(uint8_t))) {
        unaligned_store_u16(output, (uint16_t) vfill_subpattern);
        vfill_subpattern >>= 16;
        output = ((uint8_t*) output + 2);
      }
      if XNN_LIKELY(c & (1 * sizeof(uint8_t))) {
        *((uint8_t*) output) = (uint8_t) vfill_subpattern;
        output = ((uint8_t*) output + 1);
      }
    }
    output = (void*) ((uintptr_t) output + output_increment);
  } while (--rows != 0);
}
