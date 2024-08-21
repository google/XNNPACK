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

#include "xnnpack/common.h"
#include "xnnpack/vbinary.h"


void xnn_f32_vaddc_minmax_ukernel__wasmsimd_arm_u8(
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

  const v128_t voutput_min = wasm_v128_load32_splat(&params->wasmsimd.min);
  const v128_t voutput_max = wasm_v128_load32_splat(&params->wasmsimd.max);
  XNN_FORCE_REALIZATION(voutput_min);
  XNN_FORCE_REALIZATION(voutput_max);
  const v128_t vb = wasm_v128_load32_splat(input_b);

  for (; batch >= 8 * sizeof(float); batch -= 8 * sizeof(float)) {
    const v128_t va0 = wasm_v128_load(input_a);
    const v128_t va1 = wasm_v128_load(input_a + 4);
    input_a += 8;

    v128_t vy0 = wasm_f32x4_add(va0, vb);
    v128_t vy1 = wasm_f32x4_add(va1, vb);


    vy0 = wasm_f32x4_max(vy0, voutput_min);
    vy1 = wasm_f32x4_max(vy1, voutput_min);

    vy0 = wasm_f32x4_min(vy0, voutput_max);
    vy1 = wasm_f32x4_min(vy1, voutput_max);

    wasm_v128_store(output, vy0);
    wasm_v128_store(output + 4, vy1);
    output += 8;
  }
  for (; batch >= 4 * sizeof(float); batch -= 4 * sizeof(float)) {
    const v128_t va = wasm_v128_load(input_a);
    input_a += 4;

    v128_t vy = wasm_f32x4_add(va, vb);

    vy = wasm_f32x4_max(vy, voutput_min);
    vy = wasm_f32x4_min(vy, voutput_max);

    wasm_v128_store(output, vy);
    output += 4;
  }
  if XNN_UNLIKELY(batch != 0) {
    const v128_t va = wasm_v128_load(input_a);

    v128_t vy = wasm_f32x4_add(va, vb);

    vy = wasm_f32x4_max(vy, voutput_min);
    vy = wasm_f32x4_min(vy, voutput_max);

    if (batch & (2 * sizeof(float))) {
      wasm_v128_store64_lane(output, vy, 0);
      vy = wasm_v64x2_shuffle(vy, vy, 1, 1);
      output += 2;
    }
    if (batch & (1 * sizeof(float))) {
      wasm_v128_store32_lane(output, vy, 0);
    }
  }
}
