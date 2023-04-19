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

#include <xnnpack/common.h>
#include <xnnpack/reduce.h>


void xnn_f32_rmax_ukernel__wasmsimd_pminmax_x16_acc4(
    size_t batch,
    const float* input,
    float* output,
    const union xnn_f32_default_params* params)
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  v128_t vacc0 = wasm_v128_load32_splat(input);
  v128_t vacc1 = vacc0;
  v128_t vacc2 = vacc0;
  v128_t vacc3 = vacc0;
  for (; batch >= 16 * sizeof(float); batch -= 16 * sizeof(float)) {
    const v128_t vt0 = wasm_v128_load(input);
    const v128_t vt1 = wasm_v128_load(input + 4);
    const v128_t vt2 = wasm_v128_load(input + 8);
    const v128_t vt3 = wasm_v128_load(input + 12);
    input += 16;

    vacc0 = wasm_f32x4_pmax(vacc0, vt0);
    vacc1 = wasm_f32x4_pmax(vacc1, vt1);
    vacc2 = wasm_f32x4_pmax(vacc2, vt2);
    vacc3 = wasm_f32x4_pmax(vacc3, vt3);
  }
  vacc0 = wasm_f32x4_pmax(vacc0, vacc1);
  vacc2 = wasm_f32x4_pmax(vacc2, vacc3);
  vacc0 = wasm_f32x4_pmax(vacc0, vacc2);
  for (; batch >= 4 * sizeof(float); batch -= 4 * sizeof(float)) {
    const v128_t vt = wasm_v128_load(input);
    input += 4;

    vacc0 = wasm_f32x4_pmax(vacc0, vt);
  }
  vacc0 = wasm_f32x4_pmax(vacc0, wasm_v64x2_shuffle(vacc0, vacc0, 1, 1));
  if XNN_UNLIKELY(batch & (2 * sizeof(float))) {
    const v128_t vt = wasm_v128_load64_zero(input);
    input += 2;
    vacc0 = wasm_f32x4_pmax(vacc0, vt);
  }
  vacc0 = wasm_f32x4_pmax(vacc0, wasm_v32x4_shuffle(vacc0, vacc0, 1, 1, 1, 1));
  if XNN_UNLIKELY(batch & (1 * sizeof(float))) {
    const v128_t vt = wasm_v128_load32_zero(input);
    vacc0 = wasm_f32x4_pmax(vacc0, vt);
  }
  wasm_v128_store32_lane(output, vacc0, 0);
}
