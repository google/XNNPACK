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

#include "xnnpack/common.h"
#include "xnnpack/vbinary.h"


void xnn_f32_vsub_minmax_ukernel__wasmsimd_arm_u4(
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

  for (; batch >= 4 * sizeof(float); batch -= 4 * sizeof(float)) {
    const v128_t va = wasm_v128_load(input_a);
    input_a += 4;

    const v128_t vb = wasm_v128_load(input_b);
    input_b += 4;

    v128_t vacc = wasm_f32x4_sub(va, vb);

    vacc = wasm_f32x4_max(vacc, voutput_min);
    vacc = wasm_f32x4_min(vacc, voutput_max);

    wasm_v128_store(output, vacc);
    output += 4;
  }
  if XNN_UNLIKELY(batch != 0) {
    const v128_t va = wasm_v128_load(input_a);
    const v128_t vb = wasm_v128_load(input_b);

    v128_t vacc = wasm_f32x4_sub(va, vb);

    vacc = wasm_f32x4_max(vacc, voutput_min);
    vacc = wasm_f32x4_min(vacc, voutput_max);

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
