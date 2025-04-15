// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/f32-rminmax/simd.c.in
//   Generator: tools/xngen
//
// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>

#include "src/xnnpack/common.h"
#include "src/xnnpack/reduce.h"

#include "src/xnnpack/simd/f32-hvx.h"


void xnn_f32_rmax_ukernel__hvx_u32(
    size_t batch,
    const float* input,
    float* output,
    const struct xnn_f32_default_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  xnn_simd_f32_t vmax0 = xnn_set1_f32(output[0]);
  for (; batch >= 32 * sizeof(float); batch -= 32 * sizeof(float)) {
    const xnn_simd_f32_t vt = xnn_loadu_f32(input);
    input += 32;

    vmax0 = xnn_max_f32(vmax0, vt);
  }

  if XNN_UNLIKELY(batch) {
    const xnn_simd_f32_t vt = xnn_load_tail_f32(input, batch >> XNN_LOG2_SIZEOF_FLOAT);
    HVX_VectorPred mask = Q6_Q_vsetq_R(batch);

    vmax0 = xnn_max_f32(vmax0, Q6_V_vmux_QVV(mask, vt, vmax0));
  }

  const float vmax = xnn_reduce_max_f32(vmax0);

  output[0] = vmax;
}
