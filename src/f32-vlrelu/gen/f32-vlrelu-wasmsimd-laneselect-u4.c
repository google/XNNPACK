// Auto-generated file. Do not edit!
//   Template: src/f32-vlrelu/wasmsimd-laneselect.c.in
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


void xnn_f32_vlrelu_ukernel__wasmsimd_laneselect_u4(
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
  for (; batch >= 4 * sizeof(float); batch -= 4 * sizeof(float)) {
    const v128_t vx = wasm_v128_load(input);
    input += 4;
    v128_t vacc = wasm_f32x4_mul(vx, vslope);
    const v128_t vmask = wasm_i32x4_shr(vx, 31);
    vacc = wasm_v128_bitselect(vacc, vx, vmask);
    wasm_v128_store(output, vacc);
    output += 4;
  }
  if XNN_UNLIKELY(batch != 0) {
    const v128_t vx = wasm_v128_load(input);
    v128_t vacc = wasm_f32x4_mul(vx, vslope);
    const v128_t vmask = wasm_i32x4_shr(vx, 31);
    vacc = wasm_v128_bitselect(vacc, vx, vmask);

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
