// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <wasm_simd128.h>

#include <xnnpack/math.h>
#include <xnnpack/rmax.h>


void xnn_f32_rmax_ukernel__wasmsimd_x86(
    size_t batch,
    const float* input,
    float* output)
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  v128_t vmax0 = wasm_v128_load32_splat(input);
  v128_t vmax1 = vmax0;
  v128_t vmax2 = vmax0;
  v128_t vmax3 = vmax0;
  for (; batch >= 16 * sizeof(float); batch -= 16 * sizeof(float)) {
    const v128_t vx0 = wasm_v128_load(input);
    const v128_t vx1 = wasm_v128_load(input + 4);
    const v128_t vx2 = wasm_v128_load(input + 8);
    const v128_t vx3 = wasm_v128_load(input + 12);
    input += 16;

    vmax0 = wasm_f32x4_pmax(vx0, vmax0);
    vmax1 = wasm_f32x4_pmax(vx1, vmax1);
    vmax2 = wasm_f32x4_pmax(vx2, vmax2);
    vmax3 = wasm_f32x4_pmax(vx3, vmax3);
  }
  const v128_t vmax01 = wasm_f32x4_pmax(vmax1, vmax0);
  const v128_t vmax23 = wasm_f32x4_pmax(vmax3, vmax2);
  v128_t vmax0123 = wasm_f32x4_pmax(vmax23, vmax01);
  for (; batch >= 4 * sizeof(float); batch -= 4 * sizeof(float)) {
    const v128_t vx = wasm_v128_load(input);
    vmax0123 = wasm_f32x4_pmax(vx, vmax0123);
    input += 4;
  }
  const v128_t vmax2301 = wasm_v32x4_shuffle(vmax0123, vmax0123, 2, 3, 0, 1);
  vmax0123 = wasm_f32x4_pmax(vmax2301, vmax0123);
  float vmax = math_max_f32(wasm_f32x4_extract_lane(vmax0123, 0), wasm_f32x4_extract_lane(vmax0123, 1));
  if XNN_UNLIKELY(batch != 0) {
    do {
      const float vx = *input++;
      vmax = math_max_f32(vx, vmax);
      batch -= sizeof(float);
    } while (batch != 0);
  }
  *output = vmax;
}
