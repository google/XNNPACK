// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <wasm_simd128.h>

#include <xnnpack/math.h>
#include <xnnpack/rmax.h>


void xnn_f32_rmax_ukernel__wasmsimd_x86(
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

    const v128_t vlt0 = wasm_f32x4_lt(vx0, vmax0);
    const v128_t vlt1 = wasm_f32x4_lt(vx1, vmax1);
    const v128_t vlt2 = wasm_f32x4_lt(vx2, vmax2);
    const v128_t vlt3 = wasm_f32x4_lt(vx3, vmax3);

    vmax0 = wasm_v128_bitselect(vmax0, vx0, vlt0);
    vmax1 = wasm_v128_bitselect(vmax1, vx1, vlt1);
    vmax2 = wasm_v128_bitselect(vmax2, vx2, vlt2);
    vmax3 = wasm_v128_bitselect(vmax3, vx3, vlt3);
  }
  const v128_t vlt01 = wasm_f32x4_lt(vmax0, vmax1);
  const v128_t vlt23 = wasm_f32x4_lt(vmax2, vmax3);
  const v128_t vmax01 = wasm_v128_bitselect(vmax1, vmax0, vlt01);
  const v128_t vmax23 = wasm_v128_bitselect(vmax3, vmax2, vlt23);
  v128_t vmax0123 = wasm_v128_bitselect(vmax23, vmax01, wasm_f32x4_lt(vmax01, vmax23));
  for (; n >= 4 * sizeof(float); n -= 4 * sizeof(float)) {
    const v128_t vx = wasm_v128_load(x);
    vmax0123 = wasm_v128_bitselect(vmax0123, vx, wasm_f32x4_lt(vx, vmax0123));
    x += 4;
  }
  const v128_t vmax2301 = wasm_v32x4_shuffle(vmax0123, vmax0123, 2, 3, 0, 1);
  vmax0123 = wasm_v128_bitselect(vmax2301, vmax0123, wasm_f32x4_lt(vmax0123, vmax2301));
  float vmax = math_max_f32(wasm_f32x4_extract_lane(vmax0123, 0), wasm_f32x4_extract_lane(vmax0123, 1));
  if XNN_UNLIKELY(n != 0) {
    do {
      const float vx = *x++;
      vmax = math_max_f32(vx, vmax);
      n -= 4;
    } while (n != 0);
  }
  *y = vmax;
}
