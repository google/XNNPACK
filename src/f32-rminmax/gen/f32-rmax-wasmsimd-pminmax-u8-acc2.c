// Auto-generated file. Do not edit!
//   Template: src/f32-rminmax/wasmsimd.c.in
//   Generator: tools/xngen
//
// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <wasm_simd128.h>

#include "xnnpack/common.h"
#include "xnnpack/reduce.h"


void xnn_f32_rmax_ukernel__wasmsimd_pminmax_u8_acc2(
    size_t batch,
    const float* input,
    float* output,
    const union xnn_f32_default_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  v128_t vmax0 = wasm_v128_load32_splat(input);
  v128_t vmax1 = vmax0;
  for (; batch >= 8 * sizeof(float); batch -= 8 * sizeof(float)) {
    const v128_t vt0 = wasm_v128_load(input);
    const v128_t vt1 = wasm_v128_load(input + 4);
    input += 8;

    vmax0 = wasm_f32x4_pmax(vmax0, vt0);
    vmax1 = wasm_f32x4_pmax(vmax1, vt1);
  }
  vmax0 = wasm_f32x4_pmax(vmax0, vmax1);
  for (; batch >= 4 * sizeof(float); batch -= 4 * sizeof(float)) {
    const v128_t vt = wasm_v128_load(input);
    input += 4;

    vmax0 = wasm_f32x4_pmax(vmax0, vt);
  }
  vmax0 = wasm_f32x4_pmax(vmax0, wasm_v64x2_shuffle(vmax0, vmax0, 1, 1));
  if XNN_UNLIKELY(batch & (2 * sizeof(float))) {
    const v128_t vt = wasm_v128_load64_zero(input);
    input += 2;
    vmax0 = wasm_f32x4_pmax(vmax0, vt);
  }
  vmax0 = wasm_f32x4_pmax(vmax0, wasm_v32x4_shuffle(vmax0, vmax0, 1, 1, 1, 1));
  if XNN_UNLIKELY(batch & (1 * sizeof(float))) {
    const v128_t vt = wasm_v128_load32_zero(input);
    vmax0 = wasm_f32x4_pmax(vmax0, vt);
  }
  wasm_v128_store32_lane(output, vmax0, 0);
}
