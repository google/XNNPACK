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


void xnn_f32_rmin_ukernel__hvx_u32(
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
  for (; batch >= 32 * sizeof(float); batch -= 32 * sizeof(float)) {
    const xnn_simd_f32_t vt = xnn_loadu_f32(input);
    input += 32;

    vmin0 = xnn_min_f32(vmin0, vt);
  }

  for (; batch != 0; batch -= sizeof(float)) {
    const xnn_simd_f32_t vt = xnn_set1_f32(input[0]);
    input += 1;

    vmin0 = xnn_min_f32(vmin0, vt);
  }

  const float vmin = xnn_reduce_min_f32(vmin0);

  output[0] = vmin;
}
