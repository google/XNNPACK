// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <wasm_simd128.h>

#include <xnnpack/rmax.h>


void xnn_f32_rmax_ukernel__wasmsimd_arm(
    size_t n,
    const float* x,
    float* y)
{
  assert(n != 0);
  assert(n % sizeof(float) == 0);

  v128_t vmax0 = wasm_v128_load32_splat(x);
  v128_t vmax1 = vmax0;
  v128_t vmax2 = vmax0;
  v128_t vmax3 = vmax0;
  for (; n >= 16 * sizeof(float); n -= 16 * sizeof(float)) {
    const v128_t vx0 = wasm_v128_load(x);
    const v128_t vx1 = wasm_v128_load(x + 4);
    const v128_t vx2 = wasm_v128_load(x + 8);
    const v128_t vx3 = wasm_v128_load(x + 12);
    x += 16;

    vmax0 = wasm_f32x4_max(vmax0, vx0);
    vmax1 = wasm_f32x4_max(vmax1, vx1);
    vmax2 = wasm_f32x4_max(vmax2, vx2);
    vmax3 = wasm_f32x4_max(vmax3, vx3);
  }
  v128_t vmax0123 = wasm_f32x4_max(wasm_f32x4_max(vmax0, vmax1), wasm_f32x4_max(vmax2, vmax3));
  for (; n >= 4 * sizeof(float); n -= 4 * sizeof(float)) {
    const v128_t vx = wasm_v128_load(x);
    vmax0123 = wasm_f32x4_max(vmax0123, vx);
    x += 4;
  }
  vmax0123 = wasm_f32x4_max(vmax0123, wasm_v32x4_shuffle(vmax0123, vmax0123, 2, 3, 0, 1));
  float vmax = __builtin_wasm_max_f32(wasm_f32x4_extract_lane(vmax0123, 0), wasm_f32x4_extract_lane(vmax0123, 1));
  if XNN_UNLIKELY(n != 0) {
    do {
      const float vx = *x++;
      vmax = __builtin_wasm_max_f32(vx, vmax);
      n -= 4;
    } while (n != 0);
  }
  *y = vmax;
}
