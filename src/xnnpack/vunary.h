// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <stddef.h>
#include <stdint.h>

#include <xnnpack/params.h>
#include <xnnpack/common.h>

#ifdef __cplusplus
extern "C" {
#endif


#define DECLARE_F32_VUNARY_UKERNEL_FUNCTION(fn_name) \
  XNN_INTERNAL void fn_name(                         \
      size_t n,                                      \
      const float* x,                                \
      float* y,                                      \
      const void* params);

DECLARE_F32_VUNARY_UKERNEL_FUNCTION(xnn_f32_sigmoid_ukernel__neonfma_rr1_p5_div_x4)
DECLARE_F32_VUNARY_UKERNEL_FUNCTION(xnn_f32_sigmoid_ukernel__neonfma_rr1_p5_div_x8)
DECLARE_F32_VUNARY_UKERNEL_FUNCTION(xnn_f32_sigmoid_ukernel__neonfma_rr1_p5_div_x12)
DECLARE_F32_VUNARY_UKERNEL_FUNCTION(xnn_f32_sigmoid_ukernel__neonfma_rr1_p5_div_x16)
DECLARE_F32_VUNARY_UKERNEL_FUNCTION(xnn_f32_sigmoid_ukernel__neonfma_rr1_p5_div_x20)
DECLARE_F32_VUNARY_UKERNEL_FUNCTION(xnn_f32_sigmoid_ukernel__neonfma_rr1_p5_div_x24)

DECLARE_F32_VUNARY_UKERNEL_FUNCTION(xnn_f32_sigmoid_ukernel__neonfma_rr1_p5_nr2fma_x4)
DECLARE_F32_VUNARY_UKERNEL_FUNCTION(xnn_f32_sigmoid_ukernel__neonfma_rr1_p5_nr2fma_x8)
DECLARE_F32_VUNARY_UKERNEL_FUNCTION(xnn_f32_sigmoid_ukernel__neonfma_rr1_p5_nr2fma_x12)
DECLARE_F32_VUNARY_UKERNEL_FUNCTION(xnn_f32_sigmoid_ukernel__neonfma_rr1_p5_nr2fma_x16)
DECLARE_F32_VUNARY_UKERNEL_FUNCTION(xnn_f32_sigmoid_ukernel__neonfma_rr1_p5_nr2fma_x20)
DECLARE_F32_VUNARY_UKERNEL_FUNCTION(xnn_f32_sigmoid_ukernel__neonfma_rr1_p5_nr2fma_x24)

DECLARE_F32_VUNARY_UKERNEL_FUNCTION(xnn_f32_sigmoid_ukernel__neonfma_rr1_p5_nr1recps1fma_x4)
DECLARE_F32_VUNARY_UKERNEL_FUNCTION(xnn_f32_sigmoid_ukernel__neonfma_rr1_p5_nr1recps1fma_x8)
DECLARE_F32_VUNARY_UKERNEL_FUNCTION(xnn_f32_sigmoid_ukernel__neonfma_rr1_p5_nr1recps1fma_x12)
DECLARE_F32_VUNARY_UKERNEL_FUNCTION(xnn_f32_sigmoid_ukernel__neonfma_rr1_p5_nr1recps1fma_x16)
DECLARE_F32_VUNARY_UKERNEL_FUNCTION(xnn_f32_sigmoid_ukernel__neonfma_rr1_p5_nr1recps1fma_x20)
DECLARE_F32_VUNARY_UKERNEL_FUNCTION(xnn_f32_sigmoid_ukernel__neonfma_rr1_p5_nr1recps1fma_x24)

DECLARE_F32_VUNARY_UKERNEL_FUNCTION(xnn_f32_sigmoid_ukernel__neonfma_rr1_p5_nr2recps_x4)
DECLARE_F32_VUNARY_UKERNEL_FUNCTION(xnn_f32_sigmoid_ukernel__neonfma_rr1_p5_nr2recps_x8)
DECLARE_F32_VUNARY_UKERNEL_FUNCTION(xnn_f32_sigmoid_ukernel__neonfma_rr1_p5_nr2recps_x12)
DECLARE_F32_VUNARY_UKERNEL_FUNCTION(xnn_f32_sigmoid_ukernel__neonfma_rr1_p5_nr2recps_x16)
DECLARE_F32_VUNARY_UKERNEL_FUNCTION(xnn_f32_sigmoid_ukernel__neonfma_rr1_p5_nr2recps_x20)
DECLARE_F32_VUNARY_UKERNEL_FUNCTION(xnn_f32_sigmoid_ukernel__neonfma_rr1_p5_nr2recps_x24)

DECLARE_F32_VUNARY_UKERNEL_FUNCTION(xnn_f32_sigmoid_ukernel__neon_rr2_p5_nr2recps_x4)
DECLARE_F32_VUNARY_UKERNEL_FUNCTION(xnn_f32_sigmoid_ukernel__neon_rr2_p5_nr2recps_x8)
DECLARE_F32_VUNARY_UKERNEL_FUNCTION(xnn_f32_sigmoid_ukernel__neon_rr2_p5_nr2recps_x12)
DECLARE_F32_VUNARY_UKERNEL_FUNCTION(xnn_f32_sigmoid_ukernel__neon_rr2_p5_nr2recps_x16)
DECLARE_F32_VUNARY_UKERNEL_FUNCTION(xnn_f32_sigmoid_ukernel__neon_rr2_p5_nr2recps_x20)
DECLARE_F32_VUNARY_UKERNEL_FUNCTION(xnn_f32_sigmoid_ukernel__neon_rr2_p5_nr2recps_x24)

DECLARE_F32_VUNARY_UKERNEL_FUNCTION(xnn_f32_sigmoid_ukernel__neonfma_rr1_lut64_p2_div_x4)
DECLARE_F32_VUNARY_UKERNEL_FUNCTION(xnn_f32_sigmoid_ukernel__neonfma_rr1_lut64_p2_div_x8)
DECLARE_F32_VUNARY_UKERNEL_FUNCTION(xnn_f32_sigmoid_ukernel__neonfma_rr1_lut64_p2_div_x12)
DECLARE_F32_VUNARY_UKERNEL_FUNCTION(xnn_f32_sigmoid_ukernel__neonfma_rr1_lut64_p2_div_x16)
DECLARE_F32_VUNARY_UKERNEL_FUNCTION(xnn_f32_sigmoid_ukernel__neonfma_rr1_lut64_p2_div_x20)
DECLARE_F32_VUNARY_UKERNEL_FUNCTION(xnn_f32_sigmoid_ukernel__neonfma_rr1_lut64_p2_div_x24)

DECLARE_F32_VUNARY_UKERNEL_FUNCTION(xnn_f32_sigmoid_ukernel__neonfma_rr1_lut64_p2_nr2fma_x4)
DECLARE_F32_VUNARY_UKERNEL_FUNCTION(xnn_f32_sigmoid_ukernel__neonfma_rr1_lut64_p2_nr2fma_x8)
DECLARE_F32_VUNARY_UKERNEL_FUNCTION(xnn_f32_sigmoid_ukernel__neonfma_rr1_lut64_p2_nr2fma_x12)
DECLARE_F32_VUNARY_UKERNEL_FUNCTION(xnn_f32_sigmoid_ukernel__neonfma_rr1_lut64_p2_nr2fma_x16)
DECLARE_F32_VUNARY_UKERNEL_FUNCTION(xnn_f32_sigmoid_ukernel__neonfma_rr1_lut64_p2_nr2fma_x20)
DECLARE_F32_VUNARY_UKERNEL_FUNCTION(xnn_f32_sigmoid_ukernel__neonfma_rr1_lut64_p2_nr2fma_x24)

DECLARE_F32_VUNARY_UKERNEL_FUNCTION(xnn_f32_sigmoid_ukernel__neonfma_rr1_lut64_p2_nr1recps1fma_x4)
DECLARE_F32_VUNARY_UKERNEL_FUNCTION(xnn_f32_sigmoid_ukernel__neonfma_rr1_lut64_p2_nr1recps1fma_x8)
DECLARE_F32_VUNARY_UKERNEL_FUNCTION(xnn_f32_sigmoid_ukernel__neonfma_rr1_lut64_p2_nr1recps1fma_x12)
DECLARE_F32_VUNARY_UKERNEL_FUNCTION(xnn_f32_sigmoid_ukernel__neonfma_rr1_lut64_p2_nr1recps1fma_x16)
DECLARE_F32_VUNARY_UKERNEL_FUNCTION(xnn_f32_sigmoid_ukernel__neonfma_rr1_lut64_p2_nr1recps1fma_x20)
DECLARE_F32_VUNARY_UKERNEL_FUNCTION(xnn_f32_sigmoid_ukernel__neonfma_rr1_lut64_p2_nr1recps1fma_x24)

DECLARE_F32_VUNARY_UKERNEL_FUNCTION(xnn_f32_sigmoid_ukernel__neonfma_rr1_lut64_p2_nr2recps_x4)
DECLARE_F32_VUNARY_UKERNEL_FUNCTION(xnn_f32_sigmoid_ukernel__neonfma_rr1_lut64_p2_nr2recps_x8)
DECLARE_F32_VUNARY_UKERNEL_FUNCTION(xnn_f32_sigmoid_ukernel__neonfma_rr1_lut64_p2_nr2recps_x12)
DECLARE_F32_VUNARY_UKERNEL_FUNCTION(xnn_f32_sigmoid_ukernel__neonfma_rr1_lut64_p2_nr2recps_x16)
DECLARE_F32_VUNARY_UKERNEL_FUNCTION(xnn_f32_sigmoid_ukernel__neonfma_rr1_lut64_p2_nr2recps_x20)
DECLARE_F32_VUNARY_UKERNEL_FUNCTION(xnn_f32_sigmoid_ukernel__neonfma_rr1_lut64_p2_nr2recps_x24)

DECLARE_F32_VUNARY_UKERNEL_FUNCTION(xnn_f32_sigmoid_ukernel__neon_rr2_lut64_p2_nr2recps_x4)
DECLARE_F32_VUNARY_UKERNEL_FUNCTION(xnn_f32_sigmoid_ukernel__neon_rr2_lut64_p2_nr2recps_x8)
DECLARE_F32_VUNARY_UKERNEL_FUNCTION(xnn_f32_sigmoid_ukernel__neon_rr2_lut64_p2_nr2recps_x12)
DECLARE_F32_VUNARY_UKERNEL_FUNCTION(xnn_f32_sigmoid_ukernel__neon_rr2_lut64_p2_nr2recps_x16)
DECLARE_F32_VUNARY_UKERNEL_FUNCTION(xnn_f32_sigmoid_ukernel__neon_rr2_lut64_p2_nr2recps_x20)
DECLARE_F32_VUNARY_UKERNEL_FUNCTION(xnn_f32_sigmoid_ukernel__neon_rr2_lut64_p2_nr2recps_x24)

DECLARE_F32_VUNARY_UKERNEL_FUNCTION(xnn_f32_sigmoid_ukernel__neonfma_rr1_lut2048_p1_div_x4)
DECLARE_F32_VUNARY_UKERNEL_FUNCTION(xnn_f32_sigmoid_ukernel__neonfma_rr1_lut2048_p1_div_x8)
DECLARE_F32_VUNARY_UKERNEL_FUNCTION(xnn_f32_sigmoid_ukernel__neonfma_rr1_lut2048_p1_div_x12)
DECLARE_F32_VUNARY_UKERNEL_FUNCTION(xnn_f32_sigmoid_ukernel__neonfma_rr1_lut2048_p1_div_x16)
DECLARE_F32_VUNARY_UKERNEL_FUNCTION(xnn_f32_sigmoid_ukernel__neonfma_rr1_lut2048_p1_div_x20)
DECLARE_F32_VUNARY_UKERNEL_FUNCTION(xnn_f32_sigmoid_ukernel__neonfma_rr1_lut2048_p1_div_x24)

DECLARE_F32_VUNARY_UKERNEL_FUNCTION(xnn_f32_sigmoid_ukernel__neonfma_rr1_lut2048_p1_nr2fma_x4)
DECLARE_F32_VUNARY_UKERNEL_FUNCTION(xnn_f32_sigmoid_ukernel__neonfma_rr1_lut2048_p1_nr2fma_x8)
DECLARE_F32_VUNARY_UKERNEL_FUNCTION(xnn_f32_sigmoid_ukernel__neonfma_rr1_lut2048_p1_nr2fma_x12)
DECLARE_F32_VUNARY_UKERNEL_FUNCTION(xnn_f32_sigmoid_ukernel__neonfma_rr1_lut2048_p1_nr2fma_x16)
DECLARE_F32_VUNARY_UKERNEL_FUNCTION(xnn_f32_sigmoid_ukernel__neonfma_rr1_lut2048_p1_nr2fma_x20)
DECLARE_F32_VUNARY_UKERNEL_FUNCTION(xnn_f32_sigmoid_ukernel__neonfma_rr1_lut2048_p1_nr2fma_x24)

DECLARE_F32_VUNARY_UKERNEL_FUNCTION(xnn_f32_sigmoid_ukernel__neonfma_rr1_lut2048_p1_nr1recps1fma_x4)
DECLARE_F32_VUNARY_UKERNEL_FUNCTION(xnn_f32_sigmoid_ukernel__neonfma_rr1_lut2048_p1_nr1recps1fma_x8)
DECLARE_F32_VUNARY_UKERNEL_FUNCTION(xnn_f32_sigmoid_ukernel__neonfma_rr1_lut2048_p1_nr1recps1fma_x12)
DECLARE_F32_VUNARY_UKERNEL_FUNCTION(xnn_f32_sigmoid_ukernel__neonfma_rr1_lut2048_p1_nr1recps1fma_x16)
DECLARE_F32_VUNARY_UKERNEL_FUNCTION(xnn_f32_sigmoid_ukernel__neonfma_rr1_lut2048_p1_nr1recps1fma_x20)
DECLARE_F32_VUNARY_UKERNEL_FUNCTION(xnn_f32_sigmoid_ukernel__neonfma_rr1_lut2048_p1_nr1recps1fma_x24)

DECLARE_F32_VUNARY_UKERNEL_FUNCTION(xnn_f32_sigmoid_ukernel__neonfma_rr1_lut2048_p1_nr2recps_x4)
DECLARE_F32_VUNARY_UKERNEL_FUNCTION(xnn_f32_sigmoid_ukernel__neonfma_rr1_lut2048_p1_nr2recps_x8)
DECLARE_F32_VUNARY_UKERNEL_FUNCTION(xnn_f32_sigmoid_ukernel__neonfma_rr1_lut2048_p1_nr2recps_x12)
DECLARE_F32_VUNARY_UKERNEL_FUNCTION(xnn_f32_sigmoid_ukernel__neonfma_rr1_lut2048_p1_nr2recps_x16)
DECLARE_F32_VUNARY_UKERNEL_FUNCTION(xnn_f32_sigmoid_ukernel__neonfma_rr1_lut2048_p1_nr2recps_x20)
DECLARE_F32_VUNARY_UKERNEL_FUNCTION(xnn_f32_sigmoid_ukernel__neonfma_rr1_lut2048_p1_nr2recps_x24)

DECLARE_F32_VUNARY_UKERNEL_FUNCTION(xnn_f32_sigmoid_ukernel__neon_rr2_lut2048_p1_nr2recps_x4)
DECLARE_F32_VUNARY_UKERNEL_FUNCTION(xnn_f32_sigmoid_ukernel__neon_rr2_lut2048_p1_nr2recps_x8)
DECLARE_F32_VUNARY_UKERNEL_FUNCTION(xnn_f32_sigmoid_ukernel__neon_rr2_lut2048_p1_nr2recps_x12)
DECLARE_F32_VUNARY_UKERNEL_FUNCTION(xnn_f32_sigmoid_ukernel__neon_rr2_lut2048_p1_nr2recps_x16)
DECLARE_F32_VUNARY_UKERNEL_FUNCTION(xnn_f32_sigmoid_ukernel__neon_rr2_lut2048_p1_nr2recps_x20)
DECLARE_F32_VUNARY_UKERNEL_FUNCTION(xnn_f32_sigmoid_ukernel__neon_rr2_lut2048_p1_nr2recps_x24)

DECLARE_F32_VUNARY_UKERNEL_FUNCTION(xnn_f32_sigmoid_ukernel__neon_frac_p9_p10_nr1recps_x16)

DECLARE_F32_VUNARY_UKERNEL_FUNCTION(xnn_f32_sigmoid_ukernel__sse2_p5_div_x4)
DECLARE_F32_VUNARY_UKERNEL_FUNCTION(xnn_f32_sigmoid_ukernel__sse2_p5_div_x8)
DECLARE_F32_VUNARY_UKERNEL_FUNCTION(xnn_f32_sigmoid_ukernel__sse2_p5_div_x12)
DECLARE_F32_VUNARY_UKERNEL_FUNCTION(xnn_f32_sigmoid_ukernel__sse2_p5_div_x16)
DECLARE_F32_VUNARY_UKERNEL_FUNCTION(xnn_f32_sigmoid_ukernel__sse2_p5_div_x20)
DECLARE_F32_VUNARY_UKERNEL_FUNCTION(xnn_f32_sigmoid_ukernel__sse2_p5_div_x24)

DECLARE_F32_VUNARY_UKERNEL_FUNCTION(xnn_f32_sigmoid_ukernel__sse41_p5_div_x4)
DECLARE_F32_VUNARY_UKERNEL_FUNCTION(xnn_f32_sigmoid_ukernel__sse41_p5_div_x8)
DECLARE_F32_VUNARY_UKERNEL_FUNCTION(xnn_f32_sigmoid_ukernel__sse41_p5_div_x12)
DECLARE_F32_VUNARY_UKERNEL_FUNCTION(xnn_f32_sigmoid_ukernel__sse41_p5_div_x16)
DECLARE_F32_VUNARY_UKERNEL_FUNCTION(xnn_f32_sigmoid_ukernel__sse41_p5_div_x20)
DECLARE_F32_VUNARY_UKERNEL_FUNCTION(xnn_f32_sigmoid_ukernel__sse41_p5_div_x24)

DECLARE_F32_VUNARY_UKERNEL_FUNCTION(xnn_f32_sigmoid_ukernel__avx2_rr1_p5_div_x8)
DECLARE_F32_VUNARY_UKERNEL_FUNCTION(xnn_f32_sigmoid_ukernel__avx2_rr1_p5_div_x16)
DECLARE_F32_VUNARY_UKERNEL_FUNCTION(xnn_f32_sigmoid_ukernel__avx2_rr1_p5_div_x24)
DECLARE_F32_VUNARY_UKERNEL_FUNCTION(xnn_f32_sigmoid_ukernel__avx2_rr1_p5_div_x32)
DECLARE_F32_VUNARY_UKERNEL_FUNCTION(xnn_f32_sigmoid_ukernel__avx2_rr1_p5_div_x40)
DECLARE_F32_VUNARY_UKERNEL_FUNCTION(xnn_f32_sigmoid_ukernel__avx2_rr1_p5_div_x48)
DECLARE_F32_VUNARY_UKERNEL_FUNCTION(xnn_f32_sigmoid_ukernel__avx2_rr1_p5_div_x56)
DECLARE_F32_VUNARY_UKERNEL_FUNCTION(xnn_f32_sigmoid_ukernel__avx2_rr1_p5_div_x64)
DECLARE_F32_VUNARY_UKERNEL_FUNCTION(xnn_f32_sigmoid_ukernel__avx2_rr1_p5_div_x72)
DECLARE_F32_VUNARY_UKERNEL_FUNCTION(xnn_f32_sigmoid_ukernel__avx2_rr1_p5_div_x80)

DECLARE_F32_VUNARY_UKERNEL_FUNCTION(xnn_f32_sigmoid_ukernel__avx2_rr1_p5_nr1fma_x8)
DECLARE_F32_VUNARY_UKERNEL_FUNCTION(xnn_f32_sigmoid_ukernel__avx2_rr1_p5_nr1fma_x16)
DECLARE_F32_VUNARY_UKERNEL_FUNCTION(xnn_f32_sigmoid_ukernel__avx2_rr1_p5_nr1fma_x24)
DECLARE_F32_VUNARY_UKERNEL_FUNCTION(xnn_f32_sigmoid_ukernel__avx2_rr1_p5_nr1fma_x32)
DECLARE_F32_VUNARY_UKERNEL_FUNCTION(xnn_f32_sigmoid_ukernel__avx2_rr1_p5_nr1fma_x40)
DECLARE_F32_VUNARY_UKERNEL_FUNCTION(xnn_f32_sigmoid_ukernel__avx2_rr1_p5_nr1fma_x48)
DECLARE_F32_VUNARY_UKERNEL_FUNCTION(xnn_f32_sigmoid_ukernel__avx2_rr1_p5_nr1fma_x56)
DECLARE_F32_VUNARY_UKERNEL_FUNCTION(xnn_f32_sigmoid_ukernel__avx2_rr1_p5_nr1fma_x64)
DECLARE_F32_VUNARY_UKERNEL_FUNCTION(xnn_f32_sigmoid_ukernel__avx2_rr1_p5_nr1fma_x72)
DECLARE_F32_VUNARY_UKERNEL_FUNCTION(xnn_f32_sigmoid_ukernel__avx2_rr1_p5_nr1fma_x80)

DECLARE_F32_VUNARY_UKERNEL_FUNCTION(xnn_f32_sigmoid_ukernel__avx2_rr1_p5_nr2fma_x8)
DECLARE_F32_VUNARY_UKERNEL_FUNCTION(xnn_f32_sigmoid_ukernel__avx2_rr1_p5_nr2fma_x16)
DECLARE_F32_VUNARY_UKERNEL_FUNCTION(xnn_f32_sigmoid_ukernel__avx2_rr1_p5_nr2fma_x24)
DECLARE_F32_VUNARY_UKERNEL_FUNCTION(xnn_f32_sigmoid_ukernel__avx2_rr1_p5_nr2fma_x32)
DECLARE_F32_VUNARY_UKERNEL_FUNCTION(xnn_f32_sigmoid_ukernel__avx2_rr1_p5_nr2fma_x40)
DECLARE_F32_VUNARY_UKERNEL_FUNCTION(xnn_f32_sigmoid_ukernel__avx2_rr1_p5_nr2fma_x48)
DECLARE_F32_VUNARY_UKERNEL_FUNCTION(xnn_f32_sigmoid_ukernel__avx2_rr1_p5_nr2fma_x56)
DECLARE_F32_VUNARY_UKERNEL_FUNCTION(xnn_f32_sigmoid_ukernel__avx2_rr1_p5_nr2fma_x64)
DECLARE_F32_VUNARY_UKERNEL_FUNCTION(xnn_f32_sigmoid_ukernel__avx2_rr1_p5_nr2fma_x72)
DECLARE_F32_VUNARY_UKERNEL_FUNCTION(xnn_f32_sigmoid_ukernel__avx2_rr1_p5_nr2fma_x80)

DECLARE_F32_VUNARY_UKERNEL_FUNCTION(xnn_f32_sigmoid_ukernel__psimd_p5_div_x4)
DECLARE_F32_VUNARY_UKERNEL_FUNCTION(xnn_f32_sigmoid_ukernel__psimd_p5_div_x8)
DECLARE_F32_VUNARY_UKERNEL_FUNCTION(xnn_f32_sigmoid_ukernel__psimd_p5_div_x12)
DECLARE_F32_VUNARY_UKERNEL_FUNCTION(xnn_f32_sigmoid_ukernel__psimd_p5_div_x16)
DECLARE_F32_VUNARY_UKERNEL_FUNCTION(xnn_f32_sigmoid_ukernel__psimd_p5_div_x20)
DECLARE_F32_VUNARY_UKERNEL_FUNCTION(xnn_f32_sigmoid_ukernel__psimd_p5_div_x24)

DECLARE_F32_VUNARY_UKERNEL_FUNCTION(xnn_f32_sigmoid_ukernel__scalar_lut2048_p1_div_x1)
DECLARE_F32_VUNARY_UKERNEL_FUNCTION(xnn_f32_sigmoid_ukernel__scalar_lut2048_p1_div_x2)
DECLARE_F32_VUNARY_UKERNEL_FUNCTION(xnn_f32_sigmoid_ukernel__scalar_lut2048_p1_div_x4)

DECLARE_F32_VUNARY_UKERNEL_FUNCTION(xnn_f32_sigmoid_ukernel__scalar_lut64_p2_div_x1)
DECLARE_F32_VUNARY_UKERNEL_FUNCTION(xnn_f32_sigmoid_ukernel__scalar_lut64_p2_div_x2)
DECLARE_F32_VUNARY_UKERNEL_FUNCTION(xnn_f32_sigmoid_ukernel__scalar_lut64_p2_div_x4)

DECLARE_F32_VUNARY_UKERNEL_FUNCTION(xnn_f32_sigmoid_ukernel__scalar_p5_div_x1)
DECLARE_F32_VUNARY_UKERNEL_FUNCTION(xnn_f32_sigmoid_ukernel__scalar_p5_div_x2)
DECLARE_F32_VUNARY_UKERNEL_FUNCTION(xnn_f32_sigmoid_ukernel__scalar_p5_div_x4)

DECLARE_F32_VUNARY_UKERNEL_FUNCTION(xnn_f32_vsqr_ukernel__neon_x4)
DECLARE_F32_VUNARY_UKERNEL_FUNCTION(xnn_f32_vsqr_ukernel__neon_x8)

DECLARE_F32_VUNARY_UKERNEL_FUNCTION(xnn_f32_vsqr_ukernel__sse_x4)
DECLARE_F32_VUNARY_UKERNEL_FUNCTION(xnn_f32_vsqr_ukernel__sse_x8)

DECLARE_F32_VUNARY_UKERNEL_FUNCTION(xnn_f32_vsqr_ukernel__avx_x8)
DECLARE_F32_VUNARY_UKERNEL_FUNCTION(xnn_f32_vsqr_ukernel__avx_x16)

DECLARE_F32_VUNARY_UKERNEL_FUNCTION(xnn_f32_vsqr_ukernel__avx512f_x16)
DECLARE_F32_VUNARY_UKERNEL_FUNCTION(xnn_f32_vsqr_ukernel__avx512f_x32)

DECLARE_F32_VUNARY_UKERNEL_FUNCTION(xnn_f32_vsqr_ukernel__psimd_x4)
DECLARE_F32_VUNARY_UKERNEL_FUNCTION(xnn_f32_vsqr_ukernel__psimd_x8)

DECLARE_F32_VUNARY_UKERNEL_FUNCTION(xnn_f32_vsqr_ukernel__wasmsimd_x4)
DECLARE_F32_VUNARY_UKERNEL_FUNCTION(xnn_f32_vsqr_ukernel__wasmsimd_x8)

DECLARE_F32_VUNARY_UKERNEL_FUNCTION(xnn_f32_vsqr_ukernel__scalar_x1)
DECLARE_F32_VUNARY_UKERNEL_FUNCTION(xnn_f32_vsqr_ukernel__scalar_x2)
DECLARE_F32_VUNARY_UKERNEL_FUNCTION(xnn_f32_vsqr_ukernel__scalar_x4)


#define DECLARE_F32_VSQRT_UKERNEL_FUNCTION(fn_name) \
  XNN_INTERNAL void fn_name(                        \
      size_t n,                                     \
      const float* x,                               \
      float* y,                                     \
      const union xnn_f32_sqrt_params* params);

DECLARE_F32_VSQRT_UKERNEL_FUNCTION(xnn_f32_vsqrt_ukernel__neon_sqrt_x4)
DECLARE_F32_VSQRT_UKERNEL_FUNCTION(xnn_f32_vsqrt_ukernel__neon_sqrt_x8)

DECLARE_F32_VSQRT_UKERNEL_FUNCTION(xnn_f32_vsqrt_ukernel__neonfma_nr1rsqrts1fma1adj_x4)
DECLARE_F32_VSQRT_UKERNEL_FUNCTION(xnn_f32_vsqrt_ukernel__neonfma_nr1rsqrts1fma1adj_x8)
DECLARE_F32_VSQRT_UKERNEL_FUNCTION(xnn_f32_vsqrt_ukernel__neonfma_nr1rsqrts1fma1adj_x12)
DECLARE_F32_VSQRT_UKERNEL_FUNCTION(xnn_f32_vsqrt_ukernel__neonfma_nr1rsqrts1fma1adj_x16)
DECLARE_F32_VSQRT_UKERNEL_FUNCTION(xnn_f32_vsqrt_ukernel__neonfma_nr1rsqrts1fma1adj_x20)
DECLARE_F32_VSQRT_UKERNEL_FUNCTION(xnn_f32_vsqrt_ukernel__neonfma_nr1rsqrts1fma1adj_x24)
DECLARE_F32_VSQRT_UKERNEL_FUNCTION(xnn_f32_vsqrt_ukernel__neonfma_nr1rsqrts1fma1adj_x28)
DECLARE_F32_VSQRT_UKERNEL_FUNCTION(xnn_f32_vsqrt_ukernel__neonfma_nr1rsqrts1fma1adj_x32)
DECLARE_F32_VSQRT_UKERNEL_FUNCTION(xnn_f32_vsqrt_ukernel__neonfma_nr1rsqrts1fma1adj_x36)
DECLARE_F32_VSQRT_UKERNEL_FUNCTION(xnn_f32_vsqrt_ukernel__neonfma_nr1rsqrts1fma1adj_x40)

DECLARE_F32_VSQRT_UKERNEL_FUNCTION(xnn_f32_vsqrt_ukernel__neonfma_nr2fma1adj_x4)
DECLARE_F32_VSQRT_UKERNEL_FUNCTION(xnn_f32_vsqrt_ukernel__neonfma_nr2fma1adj_x8)
DECLARE_F32_VSQRT_UKERNEL_FUNCTION(xnn_f32_vsqrt_ukernel__neonfma_nr2fma1adj_x12)
DECLARE_F32_VSQRT_UKERNEL_FUNCTION(xnn_f32_vsqrt_ukernel__neonfma_nr2fma1adj_x16)
DECLARE_F32_VSQRT_UKERNEL_FUNCTION(xnn_f32_vsqrt_ukernel__neonfma_nr2fma1adj_x20)
DECLARE_F32_VSQRT_UKERNEL_FUNCTION(xnn_f32_vsqrt_ukernel__neonfma_nr2fma1adj_x24)
DECLARE_F32_VSQRT_UKERNEL_FUNCTION(xnn_f32_vsqrt_ukernel__neonfma_nr2fma1adj_x28)
DECLARE_F32_VSQRT_UKERNEL_FUNCTION(xnn_f32_vsqrt_ukernel__neonfma_nr2fma1adj_x32)
DECLARE_F32_VSQRT_UKERNEL_FUNCTION(xnn_f32_vsqrt_ukernel__neonfma_nr2fma1adj_x36)
DECLARE_F32_VSQRT_UKERNEL_FUNCTION(xnn_f32_vsqrt_ukernel__neonfma_nr2fma1adj_x40)

DECLARE_F32_VSQRT_UKERNEL_FUNCTION(xnn_f32_vsqrt_ukernel__sse_sqrt_x4)
DECLARE_F32_VSQRT_UKERNEL_FUNCTION(xnn_f32_vsqrt_ukernel__sse_sqrt_x8)

DECLARE_F32_VSQRT_UKERNEL_FUNCTION(xnn_f32_vsqrt_ukernel__avx_sqrt_x8)
DECLARE_F32_VSQRT_UKERNEL_FUNCTION(xnn_f32_vsqrt_ukernel__avx_sqrt_x16)

DECLARE_F32_VSQRT_UKERNEL_FUNCTION(xnn_f32_vsqrt_ukernel__fma3_nr1fma1adj_x8)
DECLARE_F32_VSQRT_UKERNEL_FUNCTION(xnn_f32_vsqrt_ukernel__fma3_nr1fma1adj_x16)
DECLARE_F32_VSQRT_UKERNEL_FUNCTION(xnn_f32_vsqrt_ukernel__fma3_nr1fma1adj_x24)
DECLARE_F32_VSQRT_UKERNEL_FUNCTION(xnn_f32_vsqrt_ukernel__fma3_nr1fma1adj_x32)
DECLARE_F32_VSQRT_UKERNEL_FUNCTION(xnn_f32_vsqrt_ukernel__fma3_nr1fma1adj_x40)
DECLARE_F32_VSQRT_UKERNEL_FUNCTION(xnn_f32_vsqrt_ukernel__fma3_nr1fma1adj_x48)
DECLARE_F32_VSQRT_UKERNEL_FUNCTION(xnn_f32_vsqrt_ukernel__fma3_nr1fma1adj_x56)
DECLARE_F32_VSQRT_UKERNEL_FUNCTION(xnn_f32_vsqrt_ukernel__fma3_nr1fma1adj_x64)

DECLARE_F32_VSQRT_UKERNEL_FUNCTION(xnn_f32_vsqrt_ukernel__avx512f_nr1fma1adj_x16)
DECLARE_F32_VSQRT_UKERNEL_FUNCTION(xnn_f32_vsqrt_ukernel__avx512f_nr1fma1adj_x32)
DECLARE_F32_VSQRT_UKERNEL_FUNCTION(xnn_f32_vsqrt_ukernel__avx512f_nr1fma1adj_x48)
DECLARE_F32_VSQRT_UKERNEL_FUNCTION(xnn_f32_vsqrt_ukernel__avx512f_nr1fma1adj_x64)
DECLARE_F32_VSQRT_UKERNEL_FUNCTION(xnn_f32_vsqrt_ukernel__avx512f_nr1fma1adj_x80)
DECLARE_F32_VSQRT_UKERNEL_FUNCTION(xnn_f32_vsqrt_ukernel__avx512f_nr1fma1adj_x96)
DECLARE_F32_VSQRT_UKERNEL_FUNCTION(xnn_f32_vsqrt_ukernel__avx512f_nr1fma1adj_x112)
DECLARE_F32_VSQRT_UKERNEL_FUNCTION(xnn_f32_vsqrt_ukernel__avx512f_nr1fma1adj_x128)

DECLARE_F32_VSQRT_UKERNEL_FUNCTION(xnn_f32_vsqrt_ukernel__wasmsimd_sqrt_x4)
DECLARE_F32_VSQRT_UKERNEL_FUNCTION(xnn_f32_vsqrt_ukernel__wasmsimd_sqrt_x8)

DECLARE_F32_VSQRT_UKERNEL_FUNCTION(xnn_f32_vsqrt_ukernel__scalar_sqrt_x1)
DECLARE_F32_VSQRT_UKERNEL_FUNCTION(xnn_f32_vsqrt_ukernel__scalar_sqrt_x2)
DECLARE_F32_VSQRT_UKERNEL_FUNCTION(xnn_f32_vsqrt_ukernel__scalar_sqrt_x4)


#define DECLARE_F32_VABS_UKERNEL_FUNCTION(fn_name) \
  XNN_INTERNAL void fn_name(                       \
      size_t n,                                    \
      const float* x,                              \
      float* y,                                    \
      const union xnn_f32_abs_params* params);


DECLARE_F32_VABS_UKERNEL_FUNCTION(xnn_f32_vabs_ukernel__neon_x4)
DECLARE_F32_VABS_UKERNEL_FUNCTION(xnn_f32_vabs_ukernel__neon_x8)

DECLARE_F32_VABS_UKERNEL_FUNCTION(xnn_f32_vabs_ukernel__sse_x4)
DECLARE_F32_VABS_UKERNEL_FUNCTION(xnn_f32_vabs_ukernel__sse_x8)

DECLARE_F32_VABS_UKERNEL_FUNCTION(xnn_f32_vabs_ukernel__avx_x8)
DECLARE_F32_VABS_UKERNEL_FUNCTION(xnn_f32_vabs_ukernel__avx_x16)

DECLARE_F32_VABS_UKERNEL_FUNCTION(xnn_f32_vabs_ukernel__avx512f_x16)
DECLARE_F32_VABS_UKERNEL_FUNCTION(xnn_f32_vabs_ukernel__avx512f_x32)

DECLARE_F32_VABS_UKERNEL_FUNCTION(xnn_f32_vabs_ukernel__psimd_x4)
DECLARE_F32_VABS_UKERNEL_FUNCTION(xnn_f32_vabs_ukernel__psimd_x8)

DECLARE_F32_VABS_UKERNEL_FUNCTION(xnn_f32_vabs_ukernel__wasmsimd_x4)
DECLARE_F32_VABS_UKERNEL_FUNCTION(xnn_f32_vabs_ukernel__wasmsimd_x8)

DECLARE_F32_VABS_UKERNEL_FUNCTION(xnn_f32_vabs_ukernel__scalar_x1)
DECLARE_F32_VABS_UKERNEL_FUNCTION(xnn_f32_vabs_ukernel__scalar_x2)
DECLARE_F32_VABS_UKERNEL_FUNCTION(xnn_f32_vabs_ukernel__scalar_x4)


#define DECLARE_F32_VLRELU_UKERNEL_FUNCTION(fn_name) \
  XNN_INTERNAL void fn_name(                         \
      size_t n,                                      \
      const float* x,                                \
      float* y,                                      \
      const union xnn_f32_lrelu_params* params);


DECLARE_F32_VLRELU_UKERNEL_FUNCTION(xnn_f32_vlrelu_ukernel__neon_x4)
DECLARE_F32_VLRELU_UKERNEL_FUNCTION(xnn_f32_vlrelu_ukernel__neon_x8)

DECLARE_F32_VLRELU_UKERNEL_FUNCTION(xnn_f32_vlrelu_ukernel__sse_x4)
DECLARE_F32_VLRELU_UKERNEL_FUNCTION(xnn_f32_vlrelu_ukernel__sse_x8)

DECLARE_F32_VLRELU_UKERNEL_FUNCTION(xnn_f32_vlrelu_ukernel__sse2_x4)
DECLARE_F32_VLRELU_UKERNEL_FUNCTION(xnn_f32_vlrelu_ukernel__sse2_x8)

DECLARE_F32_VLRELU_UKERNEL_FUNCTION(xnn_f32_vlrelu_ukernel__sse41_x4)
DECLARE_F32_VLRELU_UKERNEL_FUNCTION(xnn_f32_vlrelu_ukernel__sse41_x8)

DECLARE_F32_VLRELU_UKERNEL_FUNCTION(xnn_f32_vlrelu_ukernel__avx_x8)
DECLARE_F32_VLRELU_UKERNEL_FUNCTION(xnn_f32_vlrelu_ukernel__avx_x16)

DECLARE_F32_VLRELU_UKERNEL_FUNCTION(xnn_f32_vlrelu_ukernel__avx512f_x16)
DECLARE_F32_VLRELU_UKERNEL_FUNCTION(xnn_f32_vlrelu_ukernel__avx512f_x32)

DECLARE_F32_VLRELU_UKERNEL_FUNCTION(xnn_f32_vlrelu_ukernel__psimd_x4)
DECLARE_F32_VLRELU_UKERNEL_FUNCTION(xnn_f32_vlrelu_ukernel__psimd_x8)

DECLARE_F32_VLRELU_UKERNEL_FUNCTION(xnn_f32_vlrelu_ukernel__wasmsimd_bitselect_x4)
DECLARE_F32_VLRELU_UKERNEL_FUNCTION(xnn_f32_vlrelu_ukernel__wasmsimd_bitselect_x8)

DECLARE_F32_VLRELU_UKERNEL_FUNCTION(xnn_f32_vlrelu_ukernel__wasmsimd_minmax_x4)
DECLARE_F32_VLRELU_UKERNEL_FUNCTION(xnn_f32_vlrelu_ukernel__wasmsimd_minmax_x8)

DECLARE_F32_VLRELU_UKERNEL_FUNCTION(xnn_f32_vlrelu_ukernel__wasm_x1)
DECLARE_F32_VLRELU_UKERNEL_FUNCTION(xnn_f32_vlrelu_ukernel__wasm_x2)
DECLARE_F32_VLRELU_UKERNEL_FUNCTION(xnn_f32_vlrelu_ukernel__wasm_x4)

DECLARE_F32_VLRELU_UKERNEL_FUNCTION(xnn_f32_vlrelu_ukernel__scalar_x1)
DECLARE_F32_VLRELU_UKERNEL_FUNCTION(xnn_f32_vlrelu_ukernel__scalar_x2)
DECLARE_F32_VLRELU_UKERNEL_FUNCTION(xnn_f32_vlrelu_ukernel__scalar_x4)


#define DECLARE_F32_VNEG_UKERNEL_FUNCTION(fn_name) \
  XNN_INTERNAL void fn_name(                       \
      size_t n,                                    \
      const float* x,                              \
      float* y,                                    \
      const union xnn_f32_neg_params* params);


DECLARE_F32_VNEG_UKERNEL_FUNCTION(xnn_f32_vneg_ukernel__neon_x4)
DECLARE_F32_VNEG_UKERNEL_FUNCTION(xnn_f32_vneg_ukernel__neon_x8)

DECLARE_F32_VNEG_UKERNEL_FUNCTION(xnn_f32_vneg_ukernel__sse_x4)
DECLARE_F32_VNEG_UKERNEL_FUNCTION(xnn_f32_vneg_ukernel__sse_x8)

DECLARE_F32_VNEG_UKERNEL_FUNCTION(xnn_f32_vneg_ukernel__avx_x8)
DECLARE_F32_VNEG_UKERNEL_FUNCTION(xnn_f32_vneg_ukernel__avx_x16)

DECLARE_F32_VNEG_UKERNEL_FUNCTION(xnn_f32_vneg_ukernel__avx512f_x16)
DECLARE_F32_VNEG_UKERNEL_FUNCTION(xnn_f32_vneg_ukernel__avx512f_x32)

DECLARE_F32_VNEG_UKERNEL_FUNCTION(xnn_f32_vneg_ukernel__psimd_x4)
DECLARE_F32_VNEG_UKERNEL_FUNCTION(xnn_f32_vneg_ukernel__psimd_x8)

DECLARE_F32_VNEG_UKERNEL_FUNCTION(xnn_f32_vneg_ukernel__wasmsimd_x4)
DECLARE_F32_VNEG_UKERNEL_FUNCTION(xnn_f32_vneg_ukernel__wasmsimd_x8)

DECLARE_F32_VNEG_UKERNEL_FUNCTION(xnn_f32_vneg_ukernel__scalar_x1)
DECLARE_F32_VNEG_UKERNEL_FUNCTION(xnn_f32_vneg_ukernel__scalar_x2)
DECLARE_F32_VNEG_UKERNEL_FUNCTION(xnn_f32_vneg_ukernel__scalar_x4)


#define DECLARE_F32_VRND_UKERNEL_FUNCTION(fn_name) \
  XNN_INTERNAL void fn_name(                       \
      size_t n,                                    \
      const float* x,                              \
      float* y,                                    \
      const union xnn_f32_rnd_params* params);

DECLARE_F32_VRND_UKERNEL_FUNCTION(xnn_f32_vrndne_ukernel__neon_x4)
DECLARE_F32_VRND_UKERNEL_FUNCTION(xnn_f32_vrndne_ukernel__neon_x8)
DECLARE_F32_VRND_UKERNEL_FUNCTION(xnn_f32_vrndne_ukernel__neonv8_x4)
DECLARE_F32_VRND_UKERNEL_FUNCTION(xnn_f32_vrndne_ukernel__neonv8_x8)
DECLARE_F32_VRND_UKERNEL_FUNCTION(xnn_f32_vrndne_ukernel__sse2_x4)
DECLARE_F32_VRND_UKERNEL_FUNCTION(xnn_f32_vrndne_ukernel__sse2_x8)
DECLARE_F32_VRND_UKERNEL_FUNCTION(xnn_f32_vrndne_ukernel__sse41_x4)
DECLARE_F32_VRND_UKERNEL_FUNCTION(xnn_f32_vrndne_ukernel__sse41_x8)
DECLARE_F32_VRND_UKERNEL_FUNCTION(xnn_f32_vrndne_ukernel__avx_x8)
DECLARE_F32_VRND_UKERNEL_FUNCTION(xnn_f32_vrndne_ukernel__avx_x16)
DECLARE_F32_VRND_UKERNEL_FUNCTION(xnn_f32_vrndne_ukernel__avx512f_x16)
DECLARE_F32_VRND_UKERNEL_FUNCTION(xnn_f32_vrndne_ukernel__avx512f_x32)
DECLARE_F32_VRND_UKERNEL_FUNCTION(xnn_f32_vrndne_ukernel__psimd_x4)
DECLARE_F32_VRND_UKERNEL_FUNCTION(xnn_f32_vrndne_ukernel__psimd_x8)
DECLARE_F32_VRND_UKERNEL_FUNCTION(xnn_f32_vrndne_ukernel__wasmsimd_x4)
DECLARE_F32_VRND_UKERNEL_FUNCTION(xnn_f32_vrndne_ukernel__wasmsimd_x8)
DECLARE_F32_VRND_UKERNEL_FUNCTION(xnn_f32_vrndne_ukernel__scalar_libm_x1)
DECLARE_F32_VRND_UKERNEL_FUNCTION(xnn_f32_vrndne_ukernel__scalar_libm_x2)
DECLARE_F32_VRND_UKERNEL_FUNCTION(xnn_f32_vrndne_ukernel__scalar_libm_x4)

DECLARE_F32_VRND_UKERNEL_FUNCTION(xnn_f32_vrndz_ukernel__neon_x4)
DECLARE_F32_VRND_UKERNEL_FUNCTION(xnn_f32_vrndz_ukernel__neon_x8)
DECLARE_F32_VRND_UKERNEL_FUNCTION(xnn_f32_vrndz_ukernel__neonv8_x4)
DECLARE_F32_VRND_UKERNEL_FUNCTION(xnn_f32_vrndz_ukernel__neonv8_x8)
DECLARE_F32_VRND_UKERNEL_FUNCTION(xnn_f32_vrndz_ukernel__sse2_x4)
DECLARE_F32_VRND_UKERNEL_FUNCTION(xnn_f32_vrndz_ukernel__sse2_x8)
DECLARE_F32_VRND_UKERNEL_FUNCTION(xnn_f32_vrndz_ukernel__sse41_x4)
DECLARE_F32_VRND_UKERNEL_FUNCTION(xnn_f32_vrndz_ukernel__sse41_x8)
DECLARE_F32_VRND_UKERNEL_FUNCTION(xnn_f32_vrndz_ukernel__avx_x8)
DECLARE_F32_VRND_UKERNEL_FUNCTION(xnn_f32_vrndz_ukernel__avx_x16)
DECLARE_F32_VRND_UKERNEL_FUNCTION(xnn_f32_vrndz_ukernel__avx512f_x16)
DECLARE_F32_VRND_UKERNEL_FUNCTION(xnn_f32_vrndz_ukernel__avx512f_x32)
DECLARE_F32_VRND_UKERNEL_FUNCTION(xnn_f32_vrndz_ukernel__psimd_x4)
DECLARE_F32_VRND_UKERNEL_FUNCTION(xnn_f32_vrndz_ukernel__psimd_x8)
DECLARE_F32_VRND_UKERNEL_FUNCTION(xnn_f32_vrndz_ukernel__wasmsimd_x4)
DECLARE_F32_VRND_UKERNEL_FUNCTION(xnn_f32_vrndz_ukernel__wasmsimd_x8)
DECLARE_F32_VRND_UKERNEL_FUNCTION(xnn_f32_vrndz_ukernel__scalar_libm_x1)
DECLARE_F32_VRND_UKERNEL_FUNCTION(xnn_f32_vrndz_ukernel__scalar_libm_x2)
DECLARE_F32_VRND_UKERNEL_FUNCTION(xnn_f32_vrndz_ukernel__scalar_libm_x4)

DECLARE_F32_VRND_UKERNEL_FUNCTION(xnn_f32_vrndu_ukernel__neon_x4)
DECLARE_F32_VRND_UKERNEL_FUNCTION(xnn_f32_vrndu_ukernel__neon_x8)
DECLARE_F32_VRND_UKERNEL_FUNCTION(xnn_f32_vrndu_ukernel__neonv8_x4)
DECLARE_F32_VRND_UKERNEL_FUNCTION(xnn_f32_vrndu_ukernel__neonv8_x8)
DECLARE_F32_VRND_UKERNEL_FUNCTION(xnn_f32_vrndu_ukernel__sse2_x4)
DECLARE_F32_VRND_UKERNEL_FUNCTION(xnn_f32_vrndu_ukernel__sse2_x8)
DECLARE_F32_VRND_UKERNEL_FUNCTION(xnn_f32_vrndu_ukernel__sse41_x4)
DECLARE_F32_VRND_UKERNEL_FUNCTION(xnn_f32_vrndu_ukernel__sse41_x8)
DECLARE_F32_VRND_UKERNEL_FUNCTION(xnn_f32_vrndu_ukernel__avx_x8)
DECLARE_F32_VRND_UKERNEL_FUNCTION(xnn_f32_vrndu_ukernel__avx_x16)
DECLARE_F32_VRND_UKERNEL_FUNCTION(xnn_f32_vrndu_ukernel__avx512f_x16)
DECLARE_F32_VRND_UKERNEL_FUNCTION(xnn_f32_vrndu_ukernel__avx512f_x32)
DECLARE_F32_VRND_UKERNEL_FUNCTION(xnn_f32_vrndu_ukernel__psimd_x4)
DECLARE_F32_VRND_UKERNEL_FUNCTION(xnn_f32_vrndu_ukernel__psimd_x8)
DECLARE_F32_VRND_UKERNEL_FUNCTION(xnn_f32_vrndu_ukernel__wasmsimd_x4)
DECLARE_F32_VRND_UKERNEL_FUNCTION(xnn_f32_vrndu_ukernel__wasmsimd_x8)
DECLARE_F32_VRND_UKERNEL_FUNCTION(xnn_f32_vrndu_ukernel__scalar_libm_x1)
DECLARE_F32_VRND_UKERNEL_FUNCTION(xnn_f32_vrndu_ukernel__scalar_libm_x2)
DECLARE_F32_VRND_UKERNEL_FUNCTION(xnn_f32_vrndu_ukernel__scalar_libm_x4)

DECLARE_F32_VRND_UKERNEL_FUNCTION(xnn_f32_vrndd_ukernel__neon_x4)
DECLARE_F32_VRND_UKERNEL_FUNCTION(xnn_f32_vrndd_ukernel__neon_x8)
DECLARE_F32_VRND_UKERNEL_FUNCTION(xnn_f32_vrndd_ukernel__neonv8_x4)
DECLARE_F32_VRND_UKERNEL_FUNCTION(xnn_f32_vrndd_ukernel__neonv8_x8)
DECLARE_F32_VRND_UKERNEL_FUNCTION(xnn_f32_vrndd_ukernel__sse2_x4)
DECLARE_F32_VRND_UKERNEL_FUNCTION(xnn_f32_vrndd_ukernel__sse2_x8)
DECLARE_F32_VRND_UKERNEL_FUNCTION(xnn_f32_vrndd_ukernel__sse41_x4)
DECLARE_F32_VRND_UKERNEL_FUNCTION(xnn_f32_vrndd_ukernel__sse41_x8)
DECLARE_F32_VRND_UKERNEL_FUNCTION(xnn_f32_vrndd_ukernel__avx_x8)
DECLARE_F32_VRND_UKERNEL_FUNCTION(xnn_f32_vrndd_ukernel__avx_x16)
DECLARE_F32_VRND_UKERNEL_FUNCTION(xnn_f32_vrndd_ukernel__avx512f_x16)
DECLARE_F32_VRND_UKERNEL_FUNCTION(xnn_f32_vrndd_ukernel__avx512f_x32)
DECLARE_F32_VRND_UKERNEL_FUNCTION(xnn_f32_vrndd_ukernel__psimd_x4)
DECLARE_F32_VRND_UKERNEL_FUNCTION(xnn_f32_vrndd_ukernel__psimd_x8)
DECLARE_F32_VRND_UKERNEL_FUNCTION(xnn_f32_vrndd_ukernel__wasmsimd_x4)
DECLARE_F32_VRND_UKERNEL_FUNCTION(xnn_f32_vrndd_ukernel__wasmsimd_x8)
DECLARE_F32_VRND_UKERNEL_FUNCTION(xnn_f32_vrndd_ukernel__scalar_libm_x1)
DECLARE_F32_VRND_UKERNEL_FUNCTION(xnn_f32_vrndd_ukernel__scalar_libm_x2)
DECLARE_F32_VRND_UKERNEL_FUNCTION(xnn_f32_vrndd_ukernel__scalar_libm_x4)

#define DECLARE_F16_CLAMP_UKERNEL_FUNCTION(fn_name)   \
  XNN_INTERNAL void fn_name(                          \
      size_t n,                                       \
      const void* x,                                  \
      void* y,                                        \
      const struct xnn_f16_minmax_params* params);

DECLARE_F16_CLAMP_UKERNEL_FUNCTION(xnn_f16_clamp_ukernel__neonfp16arith_x8)
DECLARE_F16_CLAMP_UKERNEL_FUNCTION(xnn_f16_clamp_ukernel__neonfp16arith_x16)

#define DECLARE_F32_CLAMP_UKERNEL_FUNCTION(fn_name)   \
  XNN_INTERNAL void fn_name(                          \
      size_t n,                                       \
      const float* x,                                 \
      float* y,                                       \
      const union xnn_f32_minmax_params* params);

DECLARE_F32_CLAMP_UKERNEL_FUNCTION(xnn_f32_clamp_ukernel__neon_x4)
DECLARE_F32_CLAMP_UKERNEL_FUNCTION(xnn_f32_clamp_ukernel__neon_x8)
DECLARE_F32_CLAMP_UKERNEL_FUNCTION(xnn_f32_clamp_ukernel__sse_x4)
DECLARE_F32_CLAMP_UKERNEL_FUNCTION(xnn_f32_clamp_ukernel__sse_x8)
DECLARE_F32_CLAMP_UKERNEL_FUNCTION(xnn_f32_clamp_ukernel__avx_x8)
DECLARE_F32_CLAMP_UKERNEL_FUNCTION(xnn_f32_clamp_ukernel__avx_x16)
DECLARE_F32_CLAMP_UKERNEL_FUNCTION(xnn_f32_clamp_ukernel__avx512f_x16)
DECLARE_F32_CLAMP_UKERNEL_FUNCTION(xnn_f32_clamp_ukernel__avx512f_x32)
DECLARE_F32_CLAMP_UKERNEL_FUNCTION(xnn_f32_clamp_ukernel__wasmsimd_arm_x4)
DECLARE_F32_CLAMP_UKERNEL_FUNCTION(xnn_f32_clamp_ukernel__wasmsimd_arm_x8)
DECLARE_F32_CLAMP_UKERNEL_FUNCTION(xnn_f32_clamp_ukernel__wasmsimd_x86_x4)
DECLARE_F32_CLAMP_UKERNEL_FUNCTION(xnn_f32_clamp_ukernel__wasmsimd_x86_x8)
DECLARE_F32_CLAMP_UKERNEL_FUNCTION(xnn_f32_clamp_ukernel__wasm_x1)
DECLARE_F32_CLAMP_UKERNEL_FUNCTION(xnn_f32_clamp_ukernel__wasm_x2)
DECLARE_F32_CLAMP_UKERNEL_FUNCTION(xnn_f32_clamp_ukernel__wasm_x4)
DECLARE_F32_CLAMP_UKERNEL_FUNCTION(xnn_f32_clamp_ukernel__scalar_x1)
DECLARE_F32_CLAMP_UKERNEL_FUNCTION(xnn_f32_clamp_ukernel__scalar_x2)
DECLARE_F32_CLAMP_UKERNEL_FUNCTION(xnn_f32_clamp_ukernel__scalar_x4)

#define DECLARE_U8_CLAMP_UKERNEL_FUNCTION(fn_name) \
  XNN_INTERNAL void fn_name(                       \
      size_t n,                                    \
      const uint8_t* x,                            \
      uint8_t* y,                                  \
      const union xnn_u8_minmax_params* params);

DECLARE_U8_CLAMP_UKERNEL_FUNCTION(xnn_u8_clamp_ukernel__neon_x64)
DECLARE_U8_CLAMP_UKERNEL_FUNCTION(xnn_u8_clamp_ukernel__sse2_x64)
DECLARE_U8_CLAMP_UKERNEL_FUNCTION(xnn_u8_clamp_ukernel__scalar_x4)

#define DECLARE_F16_RELU_UKERNEL_FUNCTION(fn_name)   \
  XNN_INTERNAL void fn_name(                         \
      size_t n,                                      \
      const void* x,                                 \
      void* y,                                       \
      const struct xnn_f16_relu_params* params);

DECLARE_F16_RELU_UKERNEL_FUNCTION(xnn_f16_relu_ukernel__neonfp16arith_x8)
DECLARE_F16_RELU_UKERNEL_FUNCTION(xnn_f16_relu_ukernel__neonfp16arith_x16)

#define DECLARE_F32_RELU_UKERNEL_FUNCTION(fn_name)   \
  XNN_INTERNAL void fn_name(                         \
      size_t n,                                      \
      const float* x,                                \
      float* y,                                      \
      const union xnn_f32_relu_params* params);

DECLARE_F32_RELU_UKERNEL_FUNCTION(xnn_f32_relu_ukernel__neon_x4)
DECLARE_F32_RELU_UKERNEL_FUNCTION(xnn_f32_relu_ukernel__neon_x8)
DECLARE_F32_RELU_UKERNEL_FUNCTION(xnn_f32_relu_ukernel__sse_x4)
DECLARE_F32_RELU_UKERNEL_FUNCTION(xnn_f32_relu_ukernel__sse_x8)
DECLARE_F32_RELU_UKERNEL_FUNCTION(xnn_f32_relu_ukernel__avx_x8)
DECLARE_F32_RELU_UKERNEL_FUNCTION(xnn_f32_relu_ukernel__avx_x16)
DECLARE_F32_RELU_UKERNEL_FUNCTION(xnn_f32_relu_ukernel__avx512f_x16)
DECLARE_F32_RELU_UKERNEL_FUNCTION(xnn_f32_relu_ukernel__avx512f_x32)
DECLARE_F32_RELU_UKERNEL_FUNCTION(xnn_f32_relu_ukernel__wasmsimd_x4)
DECLARE_F32_RELU_UKERNEL_FUNCTION(xnn_f32_relu_ukernel__wasmsimd_x8)
DECLARE_F32_RELU_UKERNEL_FUNCTION(xnn_f32_relu_ukernel__wasm_x1)
DECLARE_F32_RELU_UKERNEL_FUNCTION(xnn_f32_relu_ukernel__wasm_x2)
DECLARE_F32_RELU_UKERNEL_FUNCTION(xnn_f32_relu_ukernel__wasm_x4)
DECLARE_F32_RELU_UKERNEL_FUNCTION(xnn_f32_relu_ukernel__scalar_x1)
DECLARE_F32_RELU_UKERNEL_FUNCTION(xnn_f32_relu_ukernel__scalar_x2)
DECLARE_F32_RELU_UKERNEL_FUNCTION(xnn_f32_relu_ukernel__scalar_x4)

#ifdef __cplusplus
}  // extern "C"
#endif
