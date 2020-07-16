// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <wasm_simd128.h>

#include <xnnpack/fill.h>


void xnn_x32_fill_ukernel__wasmsimd(
    size_t rows,
    size_t channels,
    uint32_t* output,
    size_t output_stride,
    const uint32_t* fill_value)
{
  assert(rows != 0);
  assert(channels != 0);
  assert(channels % sizeof(uint32_t) == 0);
  assert(fill_value != NULL);

  const size_t output_increment = output_stride - channels;

  const v128_t vfill = wasm_v32x4_load_splat(fill_value);
  do {
    size_t c = channels;
    for (; c >= 16 * sizeof(uint32_t); c -= 16 * sizeof(uint32_t)) {
      wasm_v128_store(output, vfill);
      wasm_v128_store(output + 4, vfill);
      wasm_v128_store(output + 8, vfill);
      wasm_v128_store(output + 12, vfill);
      output += 16;
    }
    for (; c >= 4 * sizeof(uint32_t); c -= 4 * sizeof(uint32_t)) {
      wasm_v128_store(output, vfill);
      output += 4;
    }
    if XNN_UNLIKELY(c != 0) {
      if XNN_LIKELY(c & (2 * sizeof(uint32_t))) {
        *((double*) output) = wasm_f64x2_extract_lane(vfill, 0);
        output += 2;
      }
      if XNN_LIKELY(c & (1 * sizeof(uint32_t))) {
        *((float*) output) = wasm_f32x4_extract_lane(vfill, 0);
        output += 1;
      }
    }
    output = (void*) ((uintptr_t) output + output_increment);
  } while (--rows != 0);
}
