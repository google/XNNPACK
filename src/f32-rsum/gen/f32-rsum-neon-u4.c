// clang-format off
// Auto-generated file. Do not edit!
//   Template: src/f32-rsum/simd.c.in
//   Generator: tools/xngen
//
// Copyright 2023 Google LLC
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

static XNN_INLINE float load_tail_reduce_add_f32(xnn_simd_f32_t acc,
                                                 const float* input,
                                                 size_t num_elements) {
  assert(num_elements < xnn_simd_size_f32);
  float32x2_t result =
      vadd_f32(vget_low_f32(acc), vget_high_f32(acc));
  if XNN_UNLIKELY (num_elements & 2) {
    const float32x2_t vt = vld1_f32(input);
    input += 2;
    result = vadd_f32(result, vt);
  }
  result = vpadd_f32(result, result);
  if XNN_UNLIKELY (num_elements & 1) {
    const float32x2_t vt = vld1_dup_f32(input);
    result = vadd_f32(result, vt);
  }
  return vget_lane_f32(result, 0);
}

void xnn_f32_rsum_ukernel__neon_u4(
    size_t batch,
    const float* input,
    float* output,
    const struct xnn_f32_scale_params* restrict params)
{
  assert(batch != 0);
  assert(batch % sizeof(float) == 0);
  assert(input != NULL);
  assert(output != NULL);

  xnn_simd_f32_t vacc0 = xnn_zero_f32();
  for (; batch >= 4 * sizeof(float); batch -= 4 * sizeof(float)) {
    const xnn_simd_f32_t vt0 = xnn_loadu_f32(input);
    input += 4;

    vacc0 = xnn_add_f32(vacc0, vt0);
  }
  const float vscale = params->scalar.scale;
  float vresult = load_tail_reduce_add_f32(vacc0, input, batch >> XNN_LOG2_SIZEOF_FLOAT);
  *output += vresult * vscale;
}
