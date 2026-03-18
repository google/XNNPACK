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
#include "src/xnnpack/simd/f32-sse2.h"


static XNN_INLINE void load_tail_reduce_minmax_f32(
  float* min, xnn_simd_f32_t vmin,
  const float* input, size_t num_elements
) {
  assert(num_elements < xnn_simd_size_f32);
  for (; num_elements > 0; num_elements -= 1) {
    const __m128 vt = _mm_load_ss(input);
    input += 1;

    vmin = _mm_min_ss(vmin, vt);
  }

  *min = xnn_reduce_min_f32(vmin);
}

void xnn_f32_rmin_ukernel__sse_u4(
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
  for (; batch >= 4 * sizeof(float); batch -= 4 * sizeof(float)) {
    const xnn_simd_f32_t vt = xnn_loadu_f32(input);
    input += 4;

    vmin0 = xnn_min_f32(vmin0, vt);
  }

  load_tail_reduce_minmax_f32(
    &output[0], vmin0,
    input, batch >> XNN_LOG2_SIZEOF_FLOAT
  );

}
