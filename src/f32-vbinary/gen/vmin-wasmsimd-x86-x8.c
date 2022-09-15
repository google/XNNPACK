// Auto-generated file. Do not edit!
//   Template: src/f32-vbinary/vop-wasmsimd.c.in
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


void xnn_f32_vmin_ukernel__wasmsimd_x86_x8(
    size_t batch,
    const float* input_a,
    const float* input_b,
    float* output,
    const union xnn_f32_default_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input_a != NULL);
  assert(input_b != NULL);
  assert(output != NULL);


  for (; batch >= 8 * sizeof(float); batch -= 8 * sizeof(float)) {
    const v128_t va0123 = wasm_v128_load(input_a);
    const v128_t va4567 = wasm_v128_load(input_a + 4);
    input_a += 8;

    const v128_t vb0123 = wasm_v128_load(input_b);
    const v128_t vb4567 = wasm_v128_load(input_b + 4);
    input_b += 8;

    const v128_t vm0123 = wasm_f32x4_lt(va0123, vb0123);
    const v128_t vm4567 = wasm_f32x4_lt(va4567, vb4567);

    v128_t vy0123 = wasm_v128_bitselect(va0123, vb0123, vm0123);
    v128_t vy4567 = wasm_v128_bitselect(va4567, vb4567, vm4567);


    wasm_v128_store(output, vy0123);
    wasm_v128_store(output + 4, vy4567);
    output += 8;
  }
  for (; batch >= 4 * sizeof(float); batch -= 4 * sizeof(float)) {
    const v128_t va = wasm_v128_load(input_a);
    input_a += 4;

    const v128_t vb = wasm_v128_load(input_b);
    input_b += 4;

    const v128_t vm = wasm_f32x4_lt(va, vb);
    v128_t vy = wasm_v128_bitselect(va, vb, vm);


    wasm_v128_store(output, vy);
    output += 4;
  }
  if XNN_UNLIKELY(batch != 0) {
    const v128_t va = wasm_v128_load(input_a);
    const v128_t vb = wasm_v128_load(input_b);

    const v128_t vm = wasm_f32x4_lt(va, vb);
    v128_t vy = wasm_v128_bitselect(va, vb, vm);


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
