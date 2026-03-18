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
#include <stddef.h>
#include <stdint.h>

#include "src/xnnpack/common.h"
#include "src/xnnpack/microparams.h"
#include "src/xnnpack/reduce.h"
#include "src/xnnpack/simd/f32-scalar.h"


static XNN_INLINE void load_tail_reduce_minmax_f32(
  float* max, xnn_simd_f32_t vmax,
  const float* input, size_t num_elements
) {
  assert(num_elements < xnn_simd_size_f32);
  for (; num_elements != 0; num_elements--) {
    const xnn_simd_f32_t vt = xnn_set1_f32(input[0]);
    input += 1;

    vmax = xnn_max_f32(vmax, vt);
  }

  *max = xnn_reduce_max_f32(vmax);
}

void xnn_f32_rmax_ukernel__scalar_u1(
    size_t batch,
    const float* input,
    float* output,
    const struct xnn_f32_default_params* restrict params)
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  xnn_simd_f32_t vmax0 = xnn_set1_f32(output[0]);
  for (; batch >= 1 * sizeof(float); batch -= 1 * sizeof(float)) {
    const xnn_simd_f32_t vt = xnn_loadu_f32(input);
    input += 1;

    vmax0 = xnn_max_f32(vmax0, vt);
  }

  load_tail_reduce_minmax_f32(
    &output[0], vmax0,
    input, batch >> XNN_LOG2_SIZEOF_FLOAT
  );

}
