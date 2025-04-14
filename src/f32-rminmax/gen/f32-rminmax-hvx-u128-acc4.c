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


static XNN_INLINE float xnn_reduce_max_f32(xnn_simd_f32_t v) {
  HVX_VectorPair vsum_pair = Q6_W_vshuff_VVR(v, v, 64);
  v = Q6_Vsf_vmax_VsfVsf(Q6_V_lo_W(vsum_pair), Q6_V_hi_W(vsum_pair));

  vsum_pair = Q6_W_vshuff_VVR(v, v, 32);
  v = Q6_Vsf_vmax_VsfVsf(Q6_V_lo_W(vsum_pair), Q6_V_hi_W(vsum_pair));

  vsum_pair = Q6_W_vshuff_VVR(v, v, 16);
  v = Q6_Vsf_vmax_VsfVsf(Q6_V_lo_W(vsum_pair), Q6_V_hi_W(vsum_pair));

  vsum_pair = Q6_W_vshuff_VVR(v, v, 8);
  v = Q6_Vsf_vmax_VsfVsf(Q6_V_lo_W(vsum_pair), Q6_V_hi_W(vsum_pair));

  vsum_pair = Q6_W_vshuff_VVR(v, v, 4);
  v = Q6_Vsf_vmax_VsfVsf(Q6_V_lo_W(vsum_pair), Q6_V_hi_W(vsum_pair));

  return *((float*)&v);
}

static XNN_INLINE float xnn_reduce_min_f32(xnn_simd_f32_t v) {
  HVX_VectorPair vsum_pair = Q6_W_vshuff_VVR(v, v, 64);
  v = Q6_Vsf_vmin_VsfVsf(Q6_V_lo_W(vsum_pair), Q6_V_hi_W(vsum_pair));

  vsum_pair = Q6_W_vshuff_VVR(v, v, 32);
  v = Q6_Vsf_vmin_VsfVsf(Q6_V_lo_W(vsum_pair), Q6_V_hi_W(vsum_pair));

  vsum_pair = Q6_W_vshuff_VVR(v, v, 16);
  v = Q6_Vsf_vmin_VsfVsf(Q6_V_lo_W(vsum_pair), Q6_V_hi_W(vsum_pair));

  vsum_pair = Q6_W_vshuff_VVR(v, v, 8);
  v = Q6_Vsf_vmin_VsfVsf(Q6_V_lo_W(vsum_pair), Q6_V_hi_W(vsum_pair));

  vsum_pair = Q6_W_vshuff_VVR(v, v, 4);
  v = Q6_Vsf_vmin_VsfVsf(Q6_V_lo_W(vsum_pair), Q6_V_hi_W(vsum_pair));

  return *((float*)&v);
}


void xnn_f32_rminmax_ukernel__hvx_u128_acc4(
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
  xnn_simd_f32_t vmax0 = xnn_set1_f32(output[1]);
  xnn_simd_f32_t vmin1 = vmin0;
  xnn_simd_f32_t vmax1 = vmax0;
  xnn_simd_f32_t vmin2 = vmin0;
  xnn_simd_f32_t vmax2 = vmax0;
  xnn_simd_f32_t vmin3 = vmin0;
  xnn_simd_f32_t vmax3 = vmax0;
  for (; batch >= 128 * sizeof(float); batch -= 128 * sizeof(float)) {
    const xnn_simd_f32_t vt0 = xnn_loadu_f32(input);
    const xnn_simd_f32_t vt1 = xnn_loadu_f32(input + 32);
    const xnn_simd_f32_t vt2 = xnn_loadu_f32(input + 64);
    const xnn_simd_f32_t vt3 = xnn_loadu_f32(input + 96);
    input += 128;

    vmin0 = xnn_min_f32(vmin0, vt0);
    vmax0 = xnn_max_f32(vmax0, vt0);
    vmin1 = xnn_min_f32(vmin1, vt1);
    vmax1 = xnn_max_f32(vmax1, vt1);
    vmin2 = xnn_min_f32(vmin2, vt2);
    vmax2 = xnn_max_f32(vmax2, vt2);
    vmin3 = xnn_min_f32(vmin3, vt3);
    vmax3 = xnn_max_f32(vmax3, vt3);
  }
  vmin0 = xnn_min_f32(vmin0, vmin1);
  vmax0 = xnn_max_f32(vmax0, vmax1);
  vmin2 = xnn_min_f32(vmin2, vmin3);
  vmax2 = xnn_max_f32(vmax2, vmax3);
  vmin0 = xnn_min_f32(vmin0, vmin2);
  vmax0 = xnn_max_f32(vmax0, vmax2);
  for (; batch >= 32 * sizeof(float); batch -= 32 * sizeof(float)) {
    const xnn_simd_f32_t vt = xnn_loadu_f32(input);
    input += 32;

    vmin0 = xnn_min_f32(vmin0, vt);
    vmax0 = xnn_max_f32(vmax0, vt);
  }

  for (; batch != 0; batch -= sizeof(float)) {
    const xnn_simd_f32_t vt = xnn_set1_f32(input[0]);
    input += 1;

    vmin0 = xnn_min_f32(vmin0, vt);
    vmax0 = xnn_max_f32(vmax0, vt);
  }

  const float vmin = xnn_reduce_min_f32(vmin0);
  const float vmax = xnn_reduce_max_f32(vmax0);

  output[0] = vmin;
  output[1] = vmax;
}
