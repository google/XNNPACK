// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/bf16-f32-vcvt/hvx.c.in
//   Generator: tools/xngen
//
// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include "src/xnnpack/common.h"
#include "src/xnnpack/simd/f32-hvx.h"
#include "src/xnnpack/vcvt.h"


void xnn_bf16_f32_vcvt_ukernel__hvx_u32(
    size_t batch,
    const xnn_bfloat16* input,
    float* output,
    const void* params) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(xnn_bfloat16) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const HVX_VectorPair vzero = Q6_W_vzero();
  for (; batch >= 32 * sizeof(xnn_bfloat16); batch -= 32 * sizeof(xnn_bfloat16)) {
    const HVX_Vector vx = *((const HVX_UVector*) input);
    input += 32;

    const xnn_simd_f32_t vy = Q6_V_lo_W(Q6_Ww_vunpackoor_WwVh(vzero, vx));

    xnn_storeu_f32(output, vy);
    output += 32;
  }
  if XNN_UNLIKELY(batch != 0) {
    assert(batch >= 1 * sizeof(xnn_bfloat16));
    assert(batch <= 31 * sizeof(xnn_bfloat16));

    const size_t num_elements = batch >> 1;
    const HVX_Vector vx = *((const HVX_UVector*) input);
    const xnn_simd_f32_t vy = Q6_V_lo_W(Q6_Ww_vunpackoor_WwVh(vzero, vx));
    xnn_store_tail_f32(output, vy, num_elements);
  }
}
