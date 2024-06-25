// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <wasm_simd128.h>

#include "xnnpack/zip.h"


void xnn_x32_zip_x4_ukernel__wasmsimd(
    size_t n,
    const uint32_t* input,
    uint32_t* output)
{
  assert(n != 0);
  assert(n % sizeof(uint32_t) == 0);

  const float* x = (const float*) input;
  const float* y = (const float*) ((uintptr_t) x + n);
  const float* z = (const float*) ((uintptr_t) y + n);
  const float* w = (const float*) ((uintptr_t) z + n);
  float* o = (float*) output;

  while (n >= 4 * sizeof(uint32_t)) {
    const v128_t vx = wasm_v128_load(x);
    x += 4;
    const v128_t vy = wasm_v128_load(y);
    y += 4;
    const v128_t vz = wasm_v128_load(z);
    z += 4;
    const v128_t vw = wasm_v128_load(w);
    w += 4;

    const v128_t vxy_lo = wasm_v32x4_shuffle(vx, vy, 0, 4, 1, 5);
    const v128_t vxy_hi = wasm_v32x4_shuffle(vx, vy, 2, 6, 3, 7);
    const v128_t vzw_lo = wasm_v32x4_shuffle(vz, vw, 0, 4, 1, 5);
    const v128_t vzw_hi = wasm_v32x4_shuffle(vz, vw, 2, 6, 3, 7);

    const v128_t vxyzw0 = wasm_v32x4_shuffle(vxy_lo, vzw_lo, 0, 1, 4, 5);
    const v128_t vxyzw1 = wasm_v32x4_shuffle(vxy_lo, vzw_lo, 2, 3, 6, 7);
    const v128_t vxyzw2 = wasm_v32x4_shuffle(vxy_hi, vzw_hi, 0, 1, 4, 5);
    const v128_t vxyzw3 = wasm_v32x4_shuffle(vxy_hi, vzw_hi, 2, 3, 6, 7);

    wasm_v128_store(o, vxyzw0);
    wasm_v128_store(o + 4, vxyzw1);
    wasm_v128_store(o + 8, vxyzw2);
    wasm_v128_store(o + 12, vxyzw3);
    o += 16;
    n -= 4 * sizeof(uint32_t);
  }
  if XNN_UNLIKELY(n != 0) {
    if (n & (2 * sizeof(uint32_t))) {
      const double vx = *((const double*) x);
      x += 2;
      const double vy = *((const double*) y);
      y += 2;
      const double vz = *((const double*) z);
      z += 2;
      const double vw = *((const double*) w);
      w += 2;

      const v128_t vxy = wasm_f64x2_make(vx, vy);
      const v128_t vzw = wasm_f64x2_make(vz, vw);

      const v128_t vxyzw_lo = wasm_v32x4_shuffle(vxy, vzw, 0, 2, 4, 6);
      const v128_t vxyzw_hi = wasm_v32x4_shuffle(vxy, vzw, 1, 3, 5, 7);

      wasm_v128_store(o, vxyzw_lo);
      wasm_v128_store(o + 4, vxyzw_hi);
      o += 8;
    }
    if (n & (1 * sizeof(uint32_t))) {
      const float vx = *x;
      const float vy = *y;
      const float vz = *z;
      const float vw = *w;
      o[0] = vx;
      o[1] = vy;
      o[2] = vz;
      o[3] = vw;
    }
  }
}
