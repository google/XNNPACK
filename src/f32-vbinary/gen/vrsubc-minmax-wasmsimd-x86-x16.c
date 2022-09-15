// Auto-generated file. Do not edit!
//   Template: src/f32-vbinary/vopc-wasmsimd.c.in
//   Generator: tools/xngen
//
// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include <wasm_simd128.h>

#include <xnnpack/common.h>
#include <xnnpack/vbinary.h>


void xnn_f32_vrsubc_minmax_ukernel__wasmsimd_x86_x16(
    size_t batch,
    const float* input_a,
    const float* input_b,
    float* output,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input_a != NULL);
  assert(input_b != NULL);
  assert(output != NULL);

  const v128_t vy_min = wasm_v128_load64_splat(params->wasmsimd.min);
  const v128_t vy_max = wasm_v128_load64_splat(params->wasmsimd.max);
  const v128_t vb = wasm_v128_load32_splat(input_b);
  for (; batch >= 16 * sizeof(float); batch -= 16 * sizeof(float)) {
    const v128_t va0123 = wasm_v128_load(input_a);
    const v128_t va4567 = wasm_v128_load(input_a + 4);
    const v128_t va89AB = wasm_v128_load(input_a + 8);
    const v128_t vaCDEF = wasm_v128_load(input_a + 12);
    input_a += 16;

    v128_t vy0123 = wasm_f32x4_sub(vb, va0123);
    v128_t vy4567 = wasm_f32x4_sub(vb, va4567);
    v128_t vy89AB = wasm_f32x4_sub(vb, va89AB);
    v128_t vyCDEF = wasm_f32x4_sub(vb, vaCDEF);


    vy0123 = wasm_f32x4_pmax(vy_min, vy0123);
    vy4567 = wasm_f32x4_pmax(vy_min, vy4567);
    vy89AB = wasm_f32x4_pmax(vy_min, vy89AB);
    vyCDEF = wasm_f32x4_pmax(vy_min, vyCDEF);

    vy0123 = wasm_f32x4_pmin(vy_max, vy0123);
    vy4567 = wasm_f32x4_pmin(vy_max, vy4567);
    vy89AB = wasm_f32x4_pmin(vy_max, vy89AB);
    vyCDEF = wasm_f32x4_pmin(vy_max, vyCDEF);

    wasm_v128_store(output, vy0123);
    wasm_v128_store(output + 4, vy4567);
    wasm_v128_store(output + 8, vy89AB);
    wasm_v128_store(output + 12, vyCDEF);
    output += 16;
  }
  for (; batch >= 4 * sizeof(float); batch -= 4 * sizeof(float)) {
    const v128_t va = wasm_v128_load(input_a);
    input_a += 4;

    v128_t vy = wasm_f32x4_sub(vb, va);

    vy = wasm_f32x4_pmax(vy_min, vy);
    vy = wasm_f32x4_pmin(vy_max, vy);

    wasm_v128_store(output, vy);
    output += 4;
  }
  if XNN_UNLIKELY(batch != 0) {
    const v128_t va = wasm_v128_load(input_a);

    v128_t vy = wasm_f32x4_sub(vb, va);

    vy = wasm_f32x4_pmax(vy_min, vy);
    vy = wasm_f32x4_pmin(vy_max, vy);

    if (batch & (2 * sizeof(float))) {
      *((double*) output) = wasm_f64x2_extract_lane(vy, 0);
      vy = wasm_v32x4_shuffle(vy, vy, 2, 3, 2, 3);
      output += 2;
    }
    if (batch & (1 * sizeof(float))) {
      *output = wasm_f32x4_extract_lane(vy, 0);
    }
  }
}
