// Auto-generated file. Do not edit!
//   Template: src/f32-rsum/wasmsimd.c.in
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


void xnn_f32_rsum_ukernel__wasmsimd_u16_acc4(
    size_t batch,
    const float* input,
    float* output,
    const struct xnn_f32_scale_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  v128_t vacc0 = wasm_f32x4_const_splat(0.0f);
  v128_t vacc1 = wasm_f32x4_const_splat(0.0f);
  v128_t vacc2 = wasm_f32x4_const_splat(0.0f);
  v128_t vacc3 = wasm_f32x4_const_splat(0.0f);
  for (; batch >= 16 * sizeof(float); batch -= 16 * sizeof(float)) {
    const v128_t vt0 = wasm_v128_load(input);
    const v128_t vt1 = wasm_v128_load(input + 4);
    const v128_t vt2 = wasm_v128_load(input + 8);
    const v128_t vt3 = wasm_v128_load(input + 12);
    input += 16;

    vacc0 = wasm_f32x4_add(vacc0, vt0);
    vacc1 = wasm_f32x4_add(vacc1, vt1);
    vacc2 = wasm_f32x4_add(vacc2, vt2);
    vacc3 = wasm_f32x4_add(vacc3, vt3);
  }
  vacc0 = wasm_f32x4_add(vacc0, vacc1);
  vacc2 = wasm_f32x4_add(vacc2, vacc3);
  vacc0 = wasm_f32x4_add(vacc0, vacc2);
  for (; batch >= 4 * sizeof(float); batch -= 4 * sizeof(float)) {
    const v128_t vt = wasm_v128_load(input);
    input += 4;

    vacc0 = wasm_f32x4_add(vacc0, vt);
  }
  vacc0 = wasm_f32x4_add(vacc0, wasm_v64x2_shuffle(vacc0, vacc0, 1, 1));
  if XNN_UNLIKELY(batch & (2 * sizeof(float))) {
    const v128_t vt = wasm_v128_load64_zero(input);
    input += 2;
    vacc0 = wasm_f32x4_add(vacc0, vt);
  }
  vacc0 = wasm_f32x4_add(vacc0, wasm_v32x4_shuffle(vacc0, vacc0, 1, 1, 1, 1));
  if XNN_UNLIKELY(batch & (1 * sizeof(float))) {
    const v128_t vt = wasm_v128_load32_zero(input);
    vacc0 = wasm_f32x4_add(vacc0, vt);
  }
  const v128_t vscale = wasm_v128_load32_zero(&params->scalar.scale);
  vacc0 = wasm_f32x4_mul(vacc0, vscale);
  *output += wasm_f32x4_extract_lane(vacc0, 0);
}
