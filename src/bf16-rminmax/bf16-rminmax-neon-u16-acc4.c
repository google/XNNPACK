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

#include "src/xnnpack/simd/f32-neon.h"
#include "src/xnnpack/simd/s16-neon.h"

void xnn_bf16_rminmax_ukernel__neon_u16_acc4(
    size_t batch,
    const uint16_t* input,
    uint16_t* output,
    const struct xnn_f32_default_params params[restrict XNN_MIN_ELEMENTS(1)])
{
  assert(batch != 0);
  assert(batch % sizeof(uint16_t) == 0);
  assert(input != NULL);
  assert(output != NULL);

  float32x4_t vmin0 = vreinterpretq_f32_u32(vshll_n_u16(vld1_dup_u16(output), 16));
  float32x4_t vmax0 = vreinterpretq_f32_u32(vshll_n_u16(vld1_dup_u16((uint16_t*)((uintptr_t) output + 1 * sizeof(uint16_t))), 16));
  float32x4_t vmin1 = vmin0;
  float32x4_t vmax1 = vmax0;
  float32x4_t vmin2 = vmin0;
  float32x4_t vmax2 = vmax0;
  float32x4_t vmin3 = vmin0;
  float32x4_t vmax3 = vmax0;
  for (; batch >= 16 * sizeof(uint16_t); batch -= 16 * sizeof(uint16_t)) {
    const float32x4_t vt0 = vreinterpretq_f32_u32(vshll_n_u16(vld1_u16(input), 16)); input += 4;
    const float32x4_t vt1 = vreinterpretq_f32_u32(vshll_n_u16(vld1_u16(input), 16)); input += 4;
    const float32x4_t vt2 = vreinterpretq_f32_u32(vshll_n_u16(vld1_u16(input), 16)); input += 4;
    const float32x4_t vt3 = vreinterpretq_f32_u32(vshll_n_u16(vld1_u16(input), 16)); input += 4;

    vmin0 = vminq_f32(vmin0, vt0);
    vmax0 = vmaxq_f32(vmax0, vt0);
    vmin1 = vminq_f32(vmin1, vt1);
    vmax1 = vmaxq_f32(vmax1, vt1);
    vmin2 = vminq_f32(vmin2, vt2);
    vmax2 = vmaxq_f32(vmax2, vt2);
    vmin3 = vminq_f32(vmin3, vt3);
    vmax3 = vmaxq_f32(vmax3, vt3);
  }
  vmin0 = vminq_f32(vmin0, vmin1);
  vmax0 = vmaxq_f32(vmax0, vmax1);
  vmin2 = vminq_f32(vmin2, vmin3);
  vmax2 = vmaxq_f32(vmax2, vmax3);
  vmin0 = vminq_f32(vmin0, vmin2);
  vmax0 = vmaxq_f32(vmax0, vmax2);
  for (; batch >= 4 * sizeof(uint16_t); batch -= 4 * sizeof(uint16_t)) {
    const float32x4_t vt = vreinterpretq_f32_u32(vshll_n_u16(vld1_u16(input), 16)); input += 4;
    vmin0 = vminq_f32(vmin0, vt);
    vmax0 = vmaxq_f32(vmax0, vt);
  }
  float32x2_t result_min = vmin_f32(vget_low_f32(vmin0), vget_high_f32(vmin0));
  float32x2_t result_max = vmax_f32(vget_low_f32(vmax0), vget_high_f32(vmax0));
  float32x4_t vt = vreinterpretq_f32_u32(vshll_n_u16(vld1_u16(input), 16));
  if (batch & (2 * sizeof(uint16_t))) {
    result_max = vmax_f32(result_max, vget_low_f32(vt));
    result_min = vmin_f32(result_min, vget_low_f32(vt));
    vt = vcombine_f32(vget_high_f32(vt), vget_low_f32(vt));
  }
  result_max = vpmax_f32(result_max, result_max);
  result_min = vpmin_f32(result_min, result_min);
  if (batch & (1 * sizeof(uint16_t))) {
    result_max = vmax_f32(result_max, vdup_lane_f32(vget_low_f32(vt), 0));
    result_min = vmin_f32(result_min, vdup_lane_f32(vget_low_f32(vt), 0));
  }
  output[0] = math_cvt_bf16_fp32(vget_lane_f32(result_min, 0));
  output[1] = math_cvt_bf16_fp32(vget_lane_f32(result_max, 0));
}
