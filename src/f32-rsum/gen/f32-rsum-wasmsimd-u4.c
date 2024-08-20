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


void xnn_f32_rsum_ukernel__wasmsimd_u4(
    size_t batch,
    const float* input,
    float* output,
    const union xnn_f32_scaleminmax_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  v128_t vacc0 = wasm_f32x4_const_splat(0.0f);
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
  const v128_t vmin = wasm_v128_load32_zero(&params->scalar.min);
  const v128_t vmax = wasm_v128_load32_zero(&params->scalar.max);
  vacc0 = wasm_f32x4_mul(vacc0, vscale);
  vacc0 = wasm_f32x4_max(vacc0, vmin);
  vacc0 = wasm_f32x4_min(vacc0, vmax);
  *output += wasm_f32x4_extract_lane(vacc0, 0);
}
