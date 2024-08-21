// Auto-generated file. Do not edit!
//   Template: src/f32-vclamp/wasmsimd.c.in
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


void xnn_f32_vclamp_ukernel__wasmsimd_x86_u4(
    size_t batch,
    const float* input,
    float* output,
    const union xnn_f32_minmax_params params[restrict XNN_MIN_ELEMENTS(1)]) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const v128_t vy_min = wasm_v128_load32_splat(&params->wasmsimd.min);
  const v128_t vy_max = wasm_v128_load32_splat(&params->wasmsimd.max);
  XNN_FORCE_REALIZATION(vy_min);
  XNN_FORCE_REALIZATION(vy_max);

  for (; batch >= 4 * sizeof(float); batch -= 4 * sizeof(float)) {
    v128_t vacc = wasm_v128_load(input);
    input += 4;

    vacc = wasm_f32x4_pmax(vy_min, vacc);
    vacc = wasm_f32x4_pmin(vy_max, vacc);

    wasm_v128_store(output, vacc);
    output += 4;
  }
  if XNN_UNLIKELY(batch != 0) {
    v128_t vacc = wasm_v128_load(input);

    vacc = wasm_f32x4_pmax(vy_min, vacc);
    vacc = wasm_f32x4_pmin(vy_max, vacc);

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
