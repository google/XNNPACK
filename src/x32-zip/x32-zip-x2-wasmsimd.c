// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <wasm_simd128.h>

#include "xnnpack/zip.h"


void xnn_x32_zip_x2_ukernel__wasmsimd(
    size_t n,
    const uint32_t* input,
    uint32_t* output)
{
  assert(n != 0);
  assert(n % sizeof(uint32_t) == 0);

  const float* x = (const float*) input;
  const float* y = (const float*) ((uintptr_t) x + n);
  float* o = (float*) output;

  while (n >= 4 * sizeof(uint32_t)) {
    const v128_t vx = wasm_v128_load(x);
    x += 4;
    const v128_t vy = wasm_v128_load(y);
    y += 4;
    const v128_t vxy_lo = wasm_v32x4_shuffle(vx, vy, 0, 4, 1, 5);
    const v128_t vxy_hi = wasm_v32x4_shuffle(vx, vy, 2, 6, 3, 7);
    wasm_v128_store(o, vxy_lo);
    wasm_v128_store(o + 4, vxy_hi);
    o += 8;
    n -= 4 * sizeof(uint32_t);
  }
  if XNN_UNLIKELY(n != 0) {
    if (n & (2 * sizeof(uint32_t))) {
      const double vx = *((const double*) x);
      x += 2;
      const double vy = *((const double*) y);
      y += 2;
      const v128_t vxy = wasm_f64x2_make(vx, vy);
      wasm_v128_store(o, wasm_v32x4_shuffle(vxy, vxy, 0, 2, 1, 3));
      o += 4;
    }
    if (n & (1 * sizeof(uint32_t))) {
      const float vx = *x;
      const float vy = *y;
      o[0] = vx;
      o[1] = vy;
    }
  }
}
