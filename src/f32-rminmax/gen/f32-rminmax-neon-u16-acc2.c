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
#include "src/xnnpack/simd/f32-neon.h"


static XNN_INLINE void load_tail_reduce_minmax_f32(
  float* min, xnn_simd_f32_t vmin,
  float* max, xnn_simd_f32_t vmax,
  const float* input, size_t num_elements
) {
  assert(num_elements < xnn_simd_size_f32);
  float32x2_t result_min =
      vmin_f32(vget_low_f32(vmin), vget_high_f32(vmin));
  float32x2_t result_max =
      vmax_f32(vget_low_f32(vmax), vget_high_f32(vmax));
  if XNN_UNLIKELY (num_elements & 2) {
    const float32x2_t vt = vld1_f32(input);
    input += 2;

    result_min = vmin_f32(result_min, vt);
    result_max = vmax_f32(result_max, vt);
  }

  result_min = vpmin_f32(result_min, result_min);
  result_max = vpmax_f32(result_max, result_max);
  if XNN_UNLIKELY (num_elements & 1) {
    const float32x2_t vt = vld1_dup_f32(input);
    result_min = vmin_f32(result_min, vt);
    result_max = vmax_f32(result_max, vt);
  }

  *min = vget_lane_f32(result_min, 0);
  *max = vget_lane_f32(result_max, 0);
}

void xnn_f32_rminmax_ukernel__neon_u16_acc2(
    size_t batch,
    const float* input,
    float* output,
    const struct xnn_f32_default_params* restrict params)
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  xnn_simd_f32_t vmin0 = xnn_set1_f32(output[0]);
  xnn_simd_f32_t vmax0 = xnn_set1_f32(output[1]);
  xnn_simd_f32_t vmin1 = vmin0;
  xnn_simd_f32_t vmax1 = vmax0;
  for (; batch >= 16 * sizeof(float); batch -= 16 * sizeof(float)) {
    const xnn_simd_f32_t vt0 = xnn_loadu_f32(input + 0);
    const xnn_simd_f32_t vt1 = xnn_loadu_f32(input + 4);
    const xnn_simd_f32_t vt2 = xnn_loadu_f32(input + 8);
    const xnn_simd_f32_t vt3 = xnn_loadu_f32(input + 12);
    input += 16;

    vmin0 = xnn_min_f32(vmin0, vt0);
    vmax0 = xnn_max_f32(vmax0, vt0);
    vmin1 = xnn_min_f32(vmin1, vt1);
    vmax1 = xnn_max_f32(vmax1, vt1);
    vmin0 = xnn_min_f32(vmin0, vt2);
    vmax0 = xnn_max_f32(vmax0, vt2);
    vmin1 = xnn_min_f32(vmin1, vt3);
    vmax1 = xnn_max_f32(vmax1, vt3);
  }
  vmin0 = xnn_min_f32(vmin0, vmin1);
  vmax0 = xnn_max_f32(vmax0, vmax1);
  for (; batch >= 4 * sizeof(float); batch -= 4 * sizeof(float)) {
    const xnn_simd_f32_t vt = xnn_loadu_f32(input);
    input += 4;

    vmin0 = xnn_min_f32(vmin0, vt);
    vmax0 = xnn_max_f32(vmax0, vt);
  }

  load_tail_reduce_minmax_f32(
    &output[0], vmin0,
    &output[1], vmax0,
    input, batch >> XNN_LOG2_SIZEOF_FLOAT
  );

}
