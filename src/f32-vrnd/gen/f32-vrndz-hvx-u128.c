// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/f32-vrnd/simd.c.in
//   Generator: tools/xngen
//
// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include "src/xnnpack/common.h"
#include "src/xnnpack/microparams.h"
#include "src/xnnpack/simd/f32-hvx.h"
#include "src/xnnpack/vunary.h"


void xnn_f32_vrndz_ukernel__hvx_u128(
    size_t batch,
    const float* input,
    float* output,
    const struct xnn_f32_default_params* restrict params)
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  for (; batch >= 128 * sizeof(float); batch -= 128 * sizeof(float)) {
    const xnn_simd_f32_t vx0 = xnn_loadu_f32(input); input += 32;
    const xnn_simd_f32_t vx1 = xnn_loadu_f32(input); input += 32;
    const xnn_simd_f32_t vx2 = xnn_loadu_f32(input); input += 32;
    const xnn_simd_f32_t vx3 = xnn_loadu_f32(input); input += 32;

    const xnn_simd_f32_t vy0 = xnn_trunc_f32(vx0);
    const xnn_simd_f32_t vy1 = xnn_trunc_f32(vx1);
    const xnn_simd_f32_t vy2 = xnn_trunc_f32(vx2);
    const xnn_simd_f32_t vy3 = xnn_trunc_f32(vx3);

    xnn_storeu_f32(output, vy0); output += 32;
    xnn_storeu_f32(output, vy1); output += 32;
    xnn_storeu_f32(output, vy2); output += 32;
    xnn_storeu_f32(output, vy3); output += 32;
  }
  for (; batch >= 32 * sizeof(float); batch -= 32 * sizeof(float)) {
    const xnn_simd_f32_t vx = xnn_loadu_f32(input); input += 32;
    const xnn_simd_f32_t vy = xnn_trunc_f32(vx);
    xnn_storeu_f32(output, vy); output += 32;
  }
  if XNN_UNLIKELY(batch != 0) {
    const xnn_simd_f32_t vx = xnn_load_tail_f32(input, batch >> XNN_LOG2_SIZEOF_FLOAT);
    xnn_simd_f32_t vy = xnn_trunc_f32(vx);
    xnn_store_tail_f32(output, vy, batch >> XNN_LOG2_SIZEOF_FLOAT);
  }
}
