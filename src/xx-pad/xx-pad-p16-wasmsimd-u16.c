// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <wasm_simd128.h>

#include "xnnpack/pad.h"
#include "xnnpack/unaligned.h"


void xnn_xx_pad_ukernel_p16__wasmsimd_u16(
    size_t rows,
    size_t channels,
    size_t pre_padding,
    size_t post_padding,
    const void* input,
    size_t input_stride,
    void* output,
    size_t output_stride,
    const uint32_t fill_pattern) XNN_OOB_READS
{
  const size_t input_increment = input_stride - channels;
  const size_t output_increment = output_stride - (pre_padding + channels + post_padding);

  const v128_t vfill_pattern = wasm_i32x4_splat((int32_t) fill_pattern);
  do {
    // Pre-pad input channels.
    size_t l = pre_padding;
    if XNN_LIKELY(l != 0) {
      for (; l >= 16 * sizeof(uint8_t); l -= 16 * sizeof(uint8_t)) {
        wasm_v128_store(output, vfill_pattern);
        output = (uint8_t*) output + 16;
      }
      if (l & (8 * sizeof(uint8_t))) {
        wasm_v128_store64_lane(output, vfill_pattern, 0);
        output = (uint8_t*) output + 8;
      }
      if (l & (4 * sizeof(uint8_t))) {
        unaligned_store_u32(output, fill_pattern);
        output = (uint8_t*) output + 4;
      }
      uint32_t vfill_subpattern = fill_pattern;
      if (l & (2 * sizeof(uint8_t))) {
        unaligned_store_u16(output, (uint16_t) vfill_subpattern);
        vfill_subpattern >>= 16;
        output = (uint8_t*) output + 2;
      }
      if (l & (1 * sizeof(uint8_t))) {
        *((uint8_t*) output) = (uint8_t) vfill_subpattern;
        output = (uint8_t*) output + 1;
      }
    }

    // Copy input channels.
    size_t c = channels;
    for (; c >= 16 * sizeof(uint8_t); c -= 16 * sizeof(uint8_t)) {
      const v128_t vdata = wasm_v128_load(input);
      input = (const uint8_t*) input + 16;

      wasm_v128_store(output, vdata);
      output = (uint8_t*) output + 16;
    }
    if XNN_UNLIKELY(c != 0) {
      v128_t vdata = wasm_v128_load(input);
      input = (const void*) ((uintptr_t) input + c);
      if (c & (8 * sizeof(uint8_t))) {
        wasm_v128_store64_lane(output, vdata, 0);
        vdata = wasm_v64x2_shuffle(vdata, vdata, 1, 1);
        output = (uint8_t*) output + 8;
      }
      if (c & (4 * sizeof(uint8_t))) {
        wasm_v128_store32_lane(output, vdata, 0);
        vdata = wasm_u64x2_shr(vdata, 32);
        output = (uint8_t*) output + 4;
      }
      if (c & (2 * sizeof(uint8_t))) {
        wasm_v128_store16_lane(output, vdata, 0);
        vdata = wasm_u32x4_shr(vdata, 16);
        output = (uint8_t*) output + 2;
      }
      if (c & (1 * sizeof(uint8_t))) {
        wasm_v128_store8_lane(output, vdata, 0);
        output = (uint8_t*) output + 1;
      }
    }

    // Post-pad input channels.
    size_t r = post_padding;
    if XNN_LIKELY(r != 0) {
      for (; r >= 16 * sizeof(uint8_t); r -= 16 * sizeof(uint8_t)) {
        wasm_v128_store(output, vfill_pattern);
        output = (uint8_t*) output + 16;
      }
      if (r & (8 * sizeof(uint8_t))) {
        wasm_v128_store64_lane(output, vfill_pattern, 0);
        output = (uint8_t*) output + 8;
      }
      if (r & (4 * sizeof(uint8_t))) {
        unaligned_store_u32(output, fill_pattern);
        output = (uint8_t*) output + 4;
      }
      uint32_t vfill_subpattern = fill_pattern;
      if (r & (2 * sizeof(uint8_t))) {
        unaligned_store_u16(output, (uint16_t) vfill_subpattern);
        vfill_subpattern >>= 16;
        output = (uint8_t*) output + 2;
      }
      if (r & (1 * sizeof(uint8_t))) {
        *((uint8_t*) output) = (uint8_t) vfill_subpattern;
        output = (uint8_t*) output + 1;
      }
    }

    input = (const void*) ((uintptr_t) input + input_increment);
    output = (void*) ((uintptr_t) output + output_increment);
  } while (--rows != 0);
}
