// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/f32-vlrelu/hvx.c.in
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
#include "src/xnnpack/math.h"
#include "src/xnnpack/microparams.h"
#include "src/xnnpack/simd/f32-hvx.h"
#include "src/xnnpack/vunary.h"


void xnn_f32_vlrelu_ukernel__hvx_u32(
    size_t batch,
    const float* input,
    float* output,
    const struct xnn_f32_lrelu_params* restrict params) XNN_OOB_READS
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  const xnn_simd_f32_t vslope = xnn_set1_f32(params->scalar.slope);
  const xnn_simd_f32_t vzero = xnn_zero_f32();

  for (; batch >= 32 * sizeof(float); batch -= 32 * sizeof(float)) {
    const xnn_simd_f32_t vx = xnn_loadu_f32(input);
    input += 32;

    xnn_simd_f32_t vacc = xnn_mul_f32(vx, vslope);
    const HVX_VectorPred vm = Q6_Q_vcmp_gt_VwVw(vzero, vx);
    vacc = Q6_V_vmux_QVV(vm, vacc, vx);

    xnn_storeu_f32(output, vacc);
    output += 32;
  }
  if XNN_UNLIKELY(batch != 0) {
    assert(batch >= 1 * sizeof(float));
    assert(batch <= 31 * sizeof(float));

    const size_t num_elements = batch >> XNN_LOG2_SIZEOF_FLOAT;
    const xnn_simd_f32_t vx = xnn_load_tail_f32(input, num_elements);

    xnn_simd_f32_t vacc = xnn_mul_f32(vx, vslope);
    const HVX_VectorPred vm = Q6_Q_vcmp_gt_VwVw(vzero, vx);
    vacc = Q6_V_vmux_QVV(vm, vacc, vx);

    xnn_store_tail_f32(output, vacc, num_elements);
  }
}
