// Auto-generated file. Do not edit!
//   Template: src/f32-vrelu/wasmsimd.c.in
//   Generator: tools/xngen
//
// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <wasm_simd128.h>

#include "xnnpack/vunary.h"
#include "xnnpack/common.h"


void xnn_f32_vrelu_ukernel__wasmsimd_u16(
    size_t batch,
    const float* input,
    float* output,
    const union xnn_f32_relu_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const v128_t vzero = wasm_i32x4_const_splat(0);

  for (; batch >= 16 * sizeof(float); batch -= 16 * sizeof(float)) {
    v128_t vacc0123 = wasm_v128_load(input);
    v128_t vacc4567 = wasm_v128_load(input + 4);
    v128_t vacc89AB = wasm_v128_load(input + 8);
    v128_t vaccCDEF = wasm_v128_load(input + 12);
    input += 16;

    vacc0123 = wasm_i32x4_max(vacc0123, vzero);
    vacc4567 = wasm_i32x4_max(vacc4567, vzero);
    vacc89AB = wasm_i32x4_max(vacc89AB, vzero);
    vaccCDEF = wasm_i32x4_max(vaccCDEF, vzero);

    wasm_v128_store(output, vacc0123);
    wasm_v128_store(output + 4, vacc4567);
    wasm_v128_store(output + 8, vacc89AB);
    wasm_v128_store(output + 12, vaccCDEF);
    output += 16;
  }
  for (; batch >= 4 * sizeof(float); batch -= 4 * sizeof(float)) {
    v128_t vacc = wasm_v128_load(input);
    input += 4;

    vacc = wasm_i32x4_max(vacc, vzero);

    wasm_v128_store(output, vacc);
    output += 4;
  }
  if XNN_UNLIKELY(batch != 0) {
    v128_t vacc = wasm_v128_load(input);

    vacc = wasm_i32x4_max(vacc, vzero);

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
