// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <wasm_simd128.h>

#include "xnnpack/zip.h"


void xnn_x32_zip_xm_ukernel__wasmsimd(
    size_t n,
    size_t m,
    const uint32_t* input,
    uint32_t* output)
{
  assert(n != 0);
  assert(n % sizeof(uint32_t) == 0);
  assert(m >= 4);

  const float* w = (const float*) input;
  float* o = (float*) output;
  const size_t group_increment = m * 4;
  const size_t input_increment = n * 3;
  const size_t output_increment = 4 * sizeof(uint32_t) - m * n;
  const float* last_input = (const float*) ((uintptr_t) input + n * (m - 1));
  float* last_output = (float*) ((uintptr_t) output + (m * 4 - 4 * sizeof(uint32_t)));

  for (size_t i = 0; i < m; i += 4) {
    w = (const float*) ((uintptr_t) w + input_increment);
    if (w >= last_input) {
      w = last_input;
    }
    const float* z = (const float*) ((uintptr_t) w - n);
    const float* y = (const float*) ((uintptr_t) z - n);
    const float* x = (const float*) ((uintptr_t) y - n);

    size_t k = n;
    while (k >= 4 * sizeof(uint32_t)) {
      const v128_t vx = wasm_v128_load((const v128_t*) x);
      x += 4;
      const v128_t vy = wasm_v128_load((const v128_t*) y);
      y += 4;
      const v128_t vz = wasm_v128_load((const v128_t*) z);
      z += 4;
      const v128_t vw = wasm_v128_load((const v128_t*) w);
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
      o = (float*) ((uintptr_t) o + group_increment);

      wasm_v128_store(o, vxyzw1);
      o = (float*) ((uintptr_t) o + group_increment);

      wasm_v128_store(o, vxyzw2);
      o = (float*) ((uintptr_t) o + group_increment);

      wasm_v128_store(o, vxyzw3);
      o = (float*) ((uintptr_t) o + group_increment);

      k -= 4 * sizeof(uint32_t);
    }
    if XNN_UNLIKELY(k != 0) {
      if (k & (2 * sizeof(uint32_t))) {
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
        o = (float*) ((uintptr_t) o + group_increment);

        wasm_v128_store(o, vxyzw_hi);
        o = (float*) ((uintptr_t) o + group_increment);
      }
      if (k & (1 * sizeof(uint32_t))) {
        const float vx = *x;
        const float vy = *y;
        const float vz = *z;
        const float vw = *w++;

        o[0] = vx;
        o[1] = vy;
        o[2] = vz;
        o[3] = vw;
        o = (float*) ((uintptr_t) o + group_increment);
      }
    }
    o = (float*) ((uintptr_t) o + output_increment);
    if (o > last_output) {
      o = last_output;
    }
  }
}
