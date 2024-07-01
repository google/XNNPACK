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


void xnn_f32_rmin_ukernel__wasmsimd_pminmax_u12_acc3(
    size_t batch,
    const float* input,
    float* output,
    const union xnn_f32_default_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  v128_t vmin0 = wasm_v128_load32_splat(input);
  v128_t vmin1 = vmin0;
  v128_t vmin2 = vmin0;
  for (; batch >= 12 * sizeof(float); batch -= 12 * sizeof(float)) {
    const v128_t vt0 = wasm_v128_load(input);
    const v128_t vt1 = wasm_v128_load(input + 4);
    const v128_t vt2 = wasm_v128_load(input + 8);
    input += 12;

    vmin0 = wasm_f32x4_pmin(vmin0, vt0);
    vmin1 = wasm_f32x4_pmin(vmin1, vt1);
    vmin2 = wasm_f32x4_pmin(vmin2, vt2);
  }
  vmin0 = wasm_f32x4_pmin(vmin0, vmin1);
  vmin0 = wasm_f32x4_pmin(vmin0, vmin2);
  for (; batch >= 4 * sizeof(float); batch -= 4 * sizeof(float)) {
    const v128_t vt = wasm_v128_load(input);
    input += 4;

    vmin0 = wasm_f32x4_pmin(vmin0, vt);
  }
  vmin0 = wasm_f32x4_pmin(vmin0, wasm_v64x2_shuffle(vmin0, vmin0, 1, 1));
  if XNN_UNLIKELY(batch & (2 * sizeof(float))) {
    const v128_t vt = wasm_v128_load64_zero(input);
    input += 2;
    vmin0 = wasm_f32x4_pmin(vmin0, vt);
  }
  vmin0 = wasm_f32x4_pmin(vmin0, wasm_v32x4_shuffle(vmin0, vmin0, 1, 1, 1, 1));
  if XNN_UNLIKELY(batch & (1 * sizeof(float))) {
    const v128_t vt = wasm_v128_load32_zero(input);
    vmin0 = wasm_f32x4_pmin(vmin0, vt);
  }
  wasm_v128_store32_lane(output, vmin0, 0);
}
