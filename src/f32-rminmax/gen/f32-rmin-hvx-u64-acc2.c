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


void xnn_f32_rmin_ukernel__hvx_u64_acc2(
    size_t batch,
    const float* input,
    float* output,
    const struct xnn_f32_default_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  xnn_simd_f32_t vmin0 = xnn_set1_f32(output[0]);
  xnn_simd_f32_t vmin1 = vmin0;
  for (; batch >= 64 * sizeof(float); batch -= 64 * sizeof(float)) {
    const xnn_simd_f32_t vt0 = xnn_loadu_f32(input + 0);
    const xnn_simd_f32_t vt1 = xnn_loadu_f32(input + 32);
    input += 64;

    vmin0 = xnn_min_f32(vmin0, vt0);
    vmin1 = xnn_min_f32(vmin1, vt1);
  }
  vmin0 = xnn_min_f32(vmin0, vmin1);
  for (; batch >= 32 * sizeof(float); batch -= 32 * sizeof(float)) {
    const xnn_simd_f32_t vt = xnn_loadu_f32(input);
    input += 32;

    vmin0 = xnn_min_f32(vmin0, vt);
  }

  if XNN_UNLIKELY(batch) {
    const xnn_simd_f32_t vt = xnn_load_tail_f32(input, batch >> XNN_LOG2_SIZEOF_FLOAT);
    HVX_VectorPred mask = Q6_Q_vsetq_R(batch);

    vmin0 = xnn_min_f32(vmin0, Q6_V_vmux_QVV(mask, vt, vmin0));
  }

  const float vmin = xnn_reduce_min_f32(vmin0);

  output[0] = vmin;
}
