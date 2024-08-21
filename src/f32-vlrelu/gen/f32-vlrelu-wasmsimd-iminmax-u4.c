// Auto-generated file. Do not edit!
//   Template: src/f32-vlrelu/wasmsimd-iminmax.c.in
//   Generator: tools/xngen
//
// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <wasm_simd128.h>

#include "xnnpack/common.h"
#include "xnnpack/vunary.h"


void xnn_f32_vlrelu_ukernel__wasmsimd_iminmax_u4(
    size_t batch,
    const float* input,
    float* output,
    const union xnn_f32_lrelu_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const v128_t vslope = wasm_v128_load32_splat(&params->scalar.slope);
  const v128_t vzero = wasm_i32x4_const_splat(0);
  for (; batch >= 4 * sizeof(float); batch -= 4 * sizeof(float)) {
    v128_t vx = wasm_v128_load(input);
    input += 4;
    v128_t vacc = wasm_i32x4_max(vx, vzero);
    vx = wasm_i32x4_min(vx, vzero);
    vacc = wasm_f32x4_add(wasm_f32x4_mul(vx, vslope), vacc);
    wasm_v128_store(output, vacc);
    output += 4;
  }
  if XNN_UNLIKELY(batch != 0) {
    v128_t vx = wasm_v128_load(input);
    v128_t vacc = wasm_i32x4_max(vx, vzero);
    vx = wasm_i32x4_min(vx, vzero);
    vacc = wasm_f32x4_add(wasm_f32x4_mul(vx, vslope), vacc);

    if (batch & (2 * sizeof(float))) {
      wasm_v128_store64_lane(output, vacc, 0);
      vacc = wasm_v64x2_shuffle(vacc, vacc, 1, 1);
      output += 2;
    }
    if (batch & (1 * sizeof(float))) {
      wasm_v128_store32_lane(output, vacc, 0);
    }
  }
}
