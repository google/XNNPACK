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


void xnn_f32_vsubc_minmax_ukernel__wasmsimd_arm_x4(
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
  for (; batch >= 4 * sizeof(float); batch -= 4 * sizeof(float)) {
    const v128_t va0123 = wasm_v128_load(input_a);
    input_a += 4;

    v128_t vy0123 = wasm_f32x4_sub(va0123, vb);


    vy0123 = wasm_f32x4_max(vy0123, vy_min);

    vy0123 = wasm_f32x4_min(vy0123, vy_max);

    wasm_v128_store(output, vy0123);
    output += 4;
  }
  if XNN_UNLIKELY(batch != 0) {
    const v128_t va = wasm_v128_load(input_a);

    v128_t vy = wasm_f32x4_sub(va, vb);

    vy = wasm_f32x4_max(vy, vy_min);
    vy = wasm_f32x4_min(vy, vy_max);

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
