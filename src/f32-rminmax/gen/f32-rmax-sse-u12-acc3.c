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

#include "src/xnnpack/simd/f32-sse2.h"


static XNN_INLINE void load_tail_reduce_minmax_f32(
  float* max, xnn_simd_f32_t vmax,
  const float* input, size_t num_elements
) {
  assert(num_elements < xnn_simd_size_f32);
  for (; num_elements > 0; num_elements -= 1) {
    const __m128 vt = _mm_load_ss(input);
    input += 1;

    vmax = _mm_max_ss(vmax, vt);
  }

  *max = xnn_reduce_max_f32(vmax);
}

void xnn_f32_rmax_ukernel__sse_u12_acc3(
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
  xnn_simd_f32_t vmax1 = vmax0;
  xnn_simd_f32_t vmax2 = vmax0;
  for (; batch >= 12 * sizeof(float); batch -= 12 * sizeof(float)) {
    const xnn_simd_f32_t vt0 = xnn_loadu_f32(input + 0);
    const xnn_simd_f32_t vt1 = xnn_loadu_f32(input + 4);
    const xnn_simd_f32_t vt2 = xnn_loadu_f32(input + 8);
    input += 12;

    vmax0 = xnn_max_f32(vmax0, vt0);
    vmax1 = xnn_max_f32(vmax1, vt1);
    vmax2 = xnn_max_f32(vmax2, vt2);
  }
  vmax0 = xnn_max_f32(vmax0, vmax1);
  vmax0 = xnn_max_f32(vmax0, vmax2);
  for (; batch >= 4 * sizeof(float); batch -= 4 * sizeof(float)) {
    const xnn_simd_f32_t vt = xnn_loadu_f32(input);
    input += 4;

    vmax0 = xnn_max_f32(vmax0, vt);
  }

  load_tail_reduce_minmax_f32(
    &output[0], vmax0,
    input, batch >> XNN_LOG2_SIZEOF_FLOAT
  );

}
