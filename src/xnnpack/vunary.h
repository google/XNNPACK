// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <stddef.h>
#include <stdint.h>

#include <xnnpack/common.h>
#include <xnnpack/microparams.h>

#ifdef __cplusplus
extern "C" {
#endif


#define DECLARE_F16_VABS_UKERNEL_FUNCTION(fn_name) \
  XNN_INTERNAL void fn_name(                       \
      size_t n,                                    \
      const void* x,                               \
      void* y,                                     \
      const union xnn_f16_abs_params* params);

DECLARE_F16_VABS_UKERNEL_FUNCTION(xnn_f16_vabs_ukernel__neonfp16arith_x8)
DECLARE_F16_VABS_UKERNEL_FUNCTION(xnn_f16_vabs_ukernel__neonfp16arith_x16)

DECLARE_F16_VABS_UKERNEL_FUNCTION(xnn_f16_vabs_ukernel__sse2_x8)
DECLARE_F16_VABS_UKERNEL_FUNCTION(xnn_f16_vabs_ukernel__sse2_x16)


#define DECLARE_F16_VCLAMP_UKERNEL_FUNCTION(fn_name) \
  XNN_INTERNAL void fn_name(                         \
      size_t n,                                      \
      const void* x,                                 \
      void* y,                                       \
      const union xnn_f16_minmax_params* params);

DECLARE_F16_VCLAMP_UKERNEL_FUNCTION(xnn_f16_vclamp_ukernel__neonfp16arith_x8)
DECLARE_F16_VCLAMP_UKERNEL_FUNCTION(xnn_f16_vclamp_ukernel__neonfp16arith_x16)

DECLARE_F16_VCLAMP_UKERNEL_FUNCTION(xnn_f16_vclamp_ukernel__f16c_x8)
DECLARE_F16_VCLAMP_UKERNEL_FUNCTION(xnn_f16_vclamp_ukernel__f16c_x16)


#define DECLARE_F16_VELU_UKERNEL_FUNCTION(fn_name) \
  XNN_INTERNAL void fn_name(                       \
      size_t n,                                    \
      const void* x,                               \
      void* y,                                     \
      const union xnn_f16_elu_params* params);

DECLARE_F16_VELU_UKERNEL_FUNCTION(xnn_f16_velu_ukernel__neonfp16arith_rr1_p3_x8)
DECLARE_F16_VELU_UKERNEL_FUNCTION(xnn_f16_velu_ukernel__neonfp16arith_rr1_p3_x16)

DECLARE_F16_VELU_UKERNEL_FUNCTION(xnn_f16_velu_ukernel__avx2_rr1_p3_x8)
DECLARE_F16_VELU_UKERNEL_FUNCTION(xnn_f16_velu_ukernel__avx2_rr1_p3_x16)


#define DECLARE_F16_VHSWISH_UKERNEL_FUNCTION(fn_name) \
  XNN_INTERNAL void fn_name(                          \
      size_t n,                                       \
      const void* x,                                  \
      void* y,                                        \
      const union xnn_f16_hswish_params* params);

DECLARE_F16_VHSWISH_UKERNEL_FUNCTION(xnn_f16_vhswish_ukernel__neonfp16arith_x8)
DECLARE_F16_VHSWISH_UKERNEL_FUNCTION(xnn_f16_vhswish_ukernel__neonfp16arith_x16)

DECLARE_F16_VHSWISH_UKERNEL_FUNCTION(xnn_f16_vhswish_ukernel__f16c_x8)
DECLARE_F16_VHSWISH_UKERNEL_FUNCTION(xnn_f16_vhswish_ukernel__f16c_x16)


#define DECLARE_F16_VNEG_UKERNEL_FUNCTION(fn_name) \
  XNN_INTERNAL void fn_name(                       \
      size_t n,                                    \
      const void* x,                               \
      void* y,                                     \
      const union xnn_f16_neg_params* params);


DECLARE_F16_VNEG_UKERNEL_FUNCTION(xnn_f16_vneg_ukernel__neonfp16arith_x8)
DECLARE_F16_VNEG_UKERNEL_FUNCTION(xnn_f16_vneg_ukernel__neonfp16arith_x16)

DECLARE_F16_VNEG_UKERNEL_FUNCTION(xnn_f16_vneg_ukernel__sse2_x8)
DECLARE_F16_VNEG_UKERNEL_FUNCTION(xnn_f16_vneg_ukernel__sse2_x16)


#define DECLARE_F16_VRND_UKERNEL_FUNCTION(fn_name) \
  XNN_INTERNAL void fn_name(                       \
      size_t n,                                    \
      const void* x,                               \
      void* y,                                     \
      const union xnn_f16_rnd_params* params);

DECLARE_F16_VRND_UKERNEL_FUNCTION(xnn_f16_vrndne_ukernel__f16c_x8)
DECLARE_F16_VRND_UKERNEL_FUNCTION(xnn_f16_vrndne_ukernel__f16c_x16)
DECLARE_F16_VRND_UKERNEL_FUNCTION(xnn_f16_vrndne_ukernel__neonfp16arith_x8)
DECLARE_F16_VRND_UKERNEL_FUNCTION(xnn_f16_vrndne_ukernel__neonfp16arith_x16)

DECLARE_F16_VRND_UKERNEL_FUNCTION(xnn_f16_vrndz_ukernel__f16c_x8)
DECLARE_F16_VRND_UKERNEL_FUNCTION(xnn_f16_vrndz_ukernel__f16c_x16)
DECLARE_F16_VRND_UKERNEL_FUNCTION(xnn_f16_vrndz_ukernel__neonfp16arith_x8)
DECLARE_F16_VRND_UKERNEL_FUNCTION(xnn_f16_vrndz_ukernel__neonfp16arith_x16)

DECLARE_F16_VRND_UKERNEL_FUNCTION(xnn_f16_vrndu_ukernel__f16c_x8)
DECLARE_F16_VRND_UKERNEL_FUNCTION(xnn_f16_vrndu_ukernel__f16c_x16)
DECLARE_F16_VRND_UKERNEL_FUNCTION(xnn_f16_vrndu_ukernel__neonfp16arith_x8)
DECLARE_F16_VRND_UKERNEL_FUNCTION(xnn_f16_vrndu_ukernel__neonfp16arith_x16)

DECLARE_F16_VRND_UKERNEL_FUNCTION(xnn_f16_vrndd_ukernel__f16c_x8)
DECLARE_F16_VRND_UKERNEL_FUNCTION(xnn_f16_vrndd_ukernel__f16c_x16)
DECLARE_F16_VRND_UKERNEL_FUNCTION(xnn_f16_vrndd_ukernel__neonfp16arith_x8)
DECLARE_F16_VRND_UKERNEL_FUNCTION(xnn_f16_vrndd_ukernel__neonfp16arith_x16)


#define DECLARE_F16_VSIGMOID_UKERNEL_FUNCTION(fn_name) \
  XNN_INTERNAL void fn_name(                           \
      size_t batch,                                    \
      const void* input,                               \
      void* output,                                    \
      const union xnn_f16_sigmoid_params* params);

DECLARE_F16_VSIGMOID_UKERNEL_FUNCTION(xnn_f16_vsigmoid_ukernel__avx2_rr1_p2_div_x8)
DECLARE_F16_VSIGMOID_UKERNEL_FUNCTION(xnn_f16_vsigmoid_ukernel__avx2_rr1_p2_div_x16)
DECLARE_F16_VSIGMOID_UKERNEL_FUNCTION(xnn_f16_vsigmoid_ukernel__avx2_rr1_p2_div_x24)
DECLARE_F16_VSIGMOID_UKERNEL_FUNCTION(xnn_f16_vsigmoid_ukernel__avx2_rr1_p2_div_x32)
DECLARE_F16_VSIGMOID_UKERNEL_FUNCTION(xnn_f16_vsigmoid_ukernel__avx2_rr1_p2_div_x40)
DECLARE_F16_VSIGMOID_UKERNEL_FUNCTION(xnn_f16_vsigmoid_ukernel__avx2_rr1_p2_div_x48)
DECLARE_F16_VSIGMOID_UKERNEL_FUNCTION(xnn_f16_vsigmoid_ukernel__avx2_rr1_p2_div_x56)
DECLARE_F16_VSIGMOID_UKERNEL_FUNCTION(xnn_f16_vsigmoid_ukernel__avx2_rr1_p2_div_x64)

DECLARE_F16_VSIGMOID_UKERNEL_FUNCTION(xnn_f16_vsigmoid_ukernel__avx2_rr1_p2_rcp_x8)
DECLARE_F16_VSIGMOID_UKERNEL_FUNCTION(xnn_f16_vsigmoid_ukernel__avx2_rr1_p2_rcp_x16)
DECLARE_F16_VSIGMOID_UKERNEL_FUNCTION(xnn_f16_vsigmoid_ukernel__avx2_rr1_p2_rcp_x24)
DECLARE_F16_VSIGMOID_UKERNEL_FUNCTION(xnn_f16_vsigmoid_ukernel__avx2_rr1_p2_rcp_x32)
DECLARE_F16_VSIGMOID_UKERNEL_FUNCTION(xnn_f16_vsigmoid_ukernel__avx2_rr1_p2_rcp_x40)
DECLARE_F16_VSIGMOID_UKERNEL_FUNCTION(xnn_f16_vsigmoid_ukernel__avx2_rr1_p2_rcp_x48)
DECLARE_F16_VSIGMOID_UKERNEL_FUNCTION(xnn_f16_vsigmoid_ukernel__avx2_rr1_p2_rcp_x56)
DECLARE_F16_VSIGMOID_UKERNEL_FUNCTION(xnn_f16_vsigmoid_ukernel__avx2_rr1_p2_rcp_x64)

DECLARE_F16_VSIGMOID_UKERNEL_FUNCTION(xnn_f16_vsigmoid_ukernel__aarch64_neonfp16arith_rr2_p2_div_x8)
DECLARE_F16_VSIGMOID_UKERNEL_FUNCTION(xnn_f16_vsigmoid_ukernel__aarch64_neonfp16arith_rr2_p2_div_x16)
DECLARE_F16_VSIGMOID_UKERNEL_FUNCTION(xnn_f16_vsigmoid_ukernel__aarch64_neonfp16arith_rr2_p2_div_x24)
DECLARE_F16_VSIGMOID_UKERNEL_FUNCTION(xnn_f16_vsigmoid_ukernel__aarch64_neonfp16arith_rr2_p2_div_x32)
DECLARE_F16_VSIGMOID_UKERNEL_FUNCTION(xnn_f16_vsigmoid_ukernel__aarch64_neonfp16arith_rr2_p2_div_x40)
DECLARE_F16_VSIGMOID_UKERNEL_FUNCTION(xnn_f16_vsigmoid_ukernel__aarch64_neonfp16arith_rr2_p2_div_x48)
DECLARE_F16_VSIGMOID_UKERNEL_FUNCTION(xnn_f16_vsigmoid_ukernel__aarch64_neonfp16arith_rr2_p2_div_x56)
DECLARE_F16_VSIGMOID_UKERNEL_FUNCTION(xnn_f16_vsigmoid_ukernel__aarch64_neonfp16arith_rr2_p2_div_x64)

DECLARE_F16_VSIGMOID_UKERNEL_FUNCTION(xnn_f16_vsigmoid_ukernel__neonfp16arith_rr2_p2_nr1fma_x8)
DECLARE_F16_VSIGMOID_UKERNEL_FUNCTION(xnn_f16_vsigmoid_ukernel__neonfp16arith_rr2_p2_nr1fma_x16)
DECLARE_F16_VSIGMOID_UKERNEL_FUNCTION(xnn_f16_vsigmoid_ukernel__neonfp16arith_rr2_p2_nr1fma_x24)
DECLARE_F16_VSIGMOID_UKERNEL_FUNCTION(xnn_f16_vsigmoid_ukernel__neonfp16arith_rr2_p2_nr1fma_x32)
DECLARE_F16_VSIGMOID_UKERNEL_FUNCTION(xnn_f16_vsigmoid_ukernel__neonfp16arith_rr2_p2_nr1fma_x40)
DECLARE_F16_VSIGMOID_UKERNEL_FUNCTION(xnn_f16_vsigmoid_ukernel__neonfp16arith_rr2_p2_nr1fma_x48)
DECLARE_F16_VSIGMOID_UKERNEL_FUNCTION(xnn_f16_vsigmoid_ukernel__neonfp16arith_rr2_p2_nr1fma_x56)
DECLARE_F16_VSIGMOID_UKERNEL_FUNCTION(xnn_f16_vsigmoid_ukernel__neonfp16arith_rr2_p2_nr1fma_x64)

DECLARE_F16_VSIGMOID_UKERNEL_FUNCTION(xnn_f16_vsigmoid_ukernel__neonfp16arith_rr2_p2_nr1recps_x8)
DECLARE_F16_VSIGMOID_UKERNEL_FUNCTION(xnn_f16_vsigmoid_ukernel__neonfp16arith_rr2_p2_nr1recps_x16)
DECLARE_F16_VSIGMOID_UKERNEL_FUNCTION(xnn_f16_vsigmoid_ukernel__neonfp16arith_rr2_p2_nr1recps_x24)
DECLARE_F16_VSIGMOID_UKERNEL_FUNCTION(xnn_f16_vsigmoid_ukernel__neonfp16arith_rr2_p2_nr1recps_x32)
DECLARE_F16_VSIGMOID_UKERNEL_FUNCTION(xnn_f16_vsigmoid_ukernel__neonfp16arith_rr2_p2_nr1recps_x40)
DECLARE_F16_VSIGMOID_UKERNEL_FUNCTION(xnn_f16_vsigmoid_ukernel__neonfp16arith_rr2_p2_nr1recps_x48)
DECLARE_F16_VSIGMOID_UKERNEL_FUNCTION(xnn_f16_vsigmoid_ukernel__neonfp16arith_rr2_p2_nr1recps_x56)
DECLARE_F16_VSIGMOID_UKERNEL_FUNCTION(xnn_f16_vsigmoid_ukernel__neonfp16arith_rr2_p2_nr1recps_x64)


#define DECLARE_F16_VSQR_UKERNEL_FUNCTION(fn_name) \
  XNN_INTERNAL void fn_name(                       \
      size_t n,                                    \
      const void* x,                               \
      void* y,                                     \
      const union xnn_f16_default_params* params);

DECLARE_F16_VSQR_UKERNEL_FUNCTION(xnn_f16_vsqr_ukernel__neonfp16arith_x8)
DECLARE_F16_VSQR_UKERNEL_FUNCTION(xnn_f16_vsqr_ukernel__neonfp16arith_x16)

DECLARE_F16_VSQR_UKERNEL_FUNCTION(xnn_f16_vsqr_ukernel__f16c_x8)
DECLARE_F16_VSQR_UKERNEL_FUNCTION(xnn_f16_vsqr_ukernel__f16c_x16)


#define DECLARE_F16_VSQRT_UKERNEL_FUNCTION(fn_name) \
  XNN_INTERNAL void fn_name(                        \
      size_t n,                                     \
      const void* x,                                \
      void* y,                                      \
      const union xnn_f16_sqrt_params* params);

DECLARE_F16_VSQRT_UKERNEL_FUNCTION(xnn_f16_vsqrt_ukernel__fp16arith_sqrt_x1)
DECLARE_F16_VSQRT_UKERNEL_FUNCTION(xnn_f16_vsqrt_ukernel__fp16arith_sqrt_x2)
DECLARE_F16_VSQRT_UKERNEL_FUNCTION(xnn_f16_vsqrt_ukernel__fp16arith_sqrt_x4)

DECLARE_F16_VSQRT_UKERNEL_FUNCTION(xnn_f16_vsqrt_ukernel__neonfp16arith_nr1fma1adj_x8)
DECLARE_F16_VSQRT_UKERNEL_FUNCTION(xnn_f16_vsqrt_ukernel__neonfp16arith_nr1fma1adj_x16)
DECLARE_F16_VSQRT_UKERNEL_FUNCTION(xnn_f16_vsqrt_ukernel__neonfp16arith_nr1fma1adj_x24)
DECLARE_F16_VSQRT_UKERNEL_FUNCTION(xnn_f16_vsqrt_ukernel__neonfp16arith_nr1fma1adj_x32)

DECLARE_F16_VSQRT_UKERNEL_FUNCTION(xnn_f16_vsqrt_ukernel__aarch64_neonfp16arith_sqrt_x8)
DECLARE_F16_VSQRT_UKERNEL_FUNCTION(xnn_f16_vsqrt_ukernel__aarch64_neonfp16arith_sqrt_x16)

DECLARE_F16_VSQRT_UKERNEL_FUNCTION(xnn_f16_vsqrt_ukernel__f16c_sqrt_x8)
DECLARE_F16_VSQRT_UKERNEL_FUNCTION(xnn_f16_vsqrt_ukernel__f16c_sqrt_x16)


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

DECLARE_F32_VABS_UKERNEL_FUNCTION(xnn_f32_vabs_ukernel__wasmsimd_x4)
DECLARE_F32_VABS_UKERNEL_FUNCTION(xnn_f32_vabs_ukernel__wasmsimd_x8)

DECLARE_F32_VABS_UKERNEL_FUNCTION(xnn_f32_vabs_ukernel__scalar_x1)
DECLARE_F32_VABS_UKERNEL_FUNCTION(xnn_f32_vabs_ukernel__scalar_x2)
DECLARE_F32_VABS_UKERNEL_FUNCTION(xnn_f32_vabs_ukernel__scalar_x4)


#define DECLARE_F32_VCLAMP_UKERNEL_FUNCTION(fn_name) \
  XNN_INTERNAL void fn_name(                         \
      size_t n,                                      \
      const float* x,                                \
      float* y,                                      \
      const union xnn_f32_minmax_params* params);

DECLARE_F32_VCLAMP_UKERNEL_FUNCTION(xnn_f32_vclamp_ukernel__avx_x8)
DECLARE_F32_VCLAMP_UKERNEL_FUNCTION(xnn_f32_vclamp_ukernel__avx_x16)
DECLARE_F32_VCLAMP_UKERNEL_FUNCTION(xnn_f32_vclamp_ukernel__avx512f_x16)
DECLARE_F32_VCLAMP_UKERNEL_FUNCTION(xnn_f32_vclamp_ukernel__avx512f_x32)
DECLARE_F32_VCLAMP_UKERNEL_FUNCTION(xnn_f32_vclamp_ukernel__neon_x4)
DECLARE_F32_VCLAMP_UKERNEL_FUNCTION(xnn_f32_vclamp_ukernel__neon_x8)
DECLARE_F32_VCLAMP_UKERNEL_FUNCTION(xnn_f32_vclamp_ukernel__scalar_x1)
DECLARE_F32_VCLAMP_UKERNEL_FUNCTION(xnn_f32_vclamp_ukernel__scalar_x2)
DECLARE_F32_VCLAMP_UKERNEL_FUNCTION(xnn_f32_vclamp_ukernel__scalar_x4)
DECLARE_F32_VCLAMP_UKERNEL_FUNCTION(xnn_f32_vclamp_ukernel__sse_x4)
DECLARE_F32_VCLAMP_UKERNEL_FUNCTION(xnn_f32_vclamp_ukernel__sse_x8)
DECLARE_F32_VCLAMP_UKERNEL_FUNCTION(xnn_f32_vclamp_ukernel__wasm_x1)
DECLARE_F32_VCLAMP_UKERNEL_FUNCTION(xnn_f32_vclamp_ukernel__wasm_x2)
DECLARE_F32_VCLAMP_UKERNEL_FUNCTION(xnn_f32_vclamp_ukernel__wasm_x4)
DECLARE_F32_VCLAMP_UKERNEL_FUNCTION(xnn_f32_vclamp_ukernel__wasmsimd_arm_x4)
DECLARE_F32_VCLAMP_UKERNEL_FUNCTION(xnn_f32_vclamp_ukernel__wasmsimd_arm_x8)
DECLARE_F32_VCLAMP_UKERNEL_FUNCTION(xnn_f32_vclamp_ukernel__wasmsimd_x86_x4)
DECLARE_F32_VCLAMP_UKERNEL_FUNCTION(xnn_f32_vclamp_ukernel__wasmsimd_x86_x8)


#define DECLARE_F32_VELU_UKERNEL_FUNCTION(fn_name) \
  XNN_INTERNAL void fn_name(                       \
      size_t n,                                    \
      const float* x,                              \
      float* y,                                    \
      const union xnn_f32_elu_params* params);

DECLARE_F32_VELU_UKERNEL_FUNCTION(xnn_f32_velu_ukernel__neon_rr2_lut16_p3_x4)
DECLARE_F32_VELU_UKERNEL_FUNCTION(xnn_f32_velu_ukernel__neon_rr2_lut16_p3_x8)
DECLARE_F32_VELU_UKERNEL_FUNCTION(xnn_f32_velu_ukernel__neon_rr2_lut16_p3_x12)
DECLARE_F32_VELU_UKERNEL_FUNCTION(xnn_f32_velu_ukernel__neon_rr2_lut16_p3_x16)
DECLARE_F32_VELU_UKERNEL_FUNCTION(xnn_f32_velu_ukernel__neon_rr2_lut16_p3_x20)
DECLARE_F32_VELU_UKERNEL_FUNCTION(xnn_f32_velu_ukernel__neon_rr2_lut16_p3_x24)

DECLARE_F32_VELU_UKERNEL_FUNCTION(xnn_f32_velu_ukernel__neon_rr2_p6_x4)
DECLARE_F32_VELU_UKERNEL_FUNCTION(xnn_f32_velu_ukernel__neon_rr2_p6_x8)
DECLARE_F32_VELU_UKERNEL_FUNCTION(xnn_f32_velu_ukernel__neon_rr2_p6_x12)
DECLARE_F32_VELU_UKERNEL_FUNCTION(xnn_f32_velu_ukernel__neon_rr2_p6_x16)
DECLARE_F32_VELU_UKERNEL_FUNCTION(xnn_f32_velu_ukernel__neon_rr2_p6_x20)
DECLARE_F32_VELU_UKERNEL_FUNCTION(xnn_f32_velu_ukernel__neon_rr2_p6_x24)

DECLARE_F32_VELU_UKERNEL_FUNCTION(xnn_f32_velu_ukernel__neonfma_rr1_lut16_p3_x4)
DECLARE_F32_VELU_UKERNEL_FUNCTION(xnn_f32_velu_ukernel__neonfma_rr1_lut16_p3_x8)
DECLARE_F32_VELU_UKERNEL_FUNCTION(xnn_f32_velu_ukernel__neonfma_rr1_lut16_p3_x12)
DECLARE_F32_VELU_UKERNEL_FUNCTION(xnn_f32_velu_ukernel__neonfma_rr1_lut16_p3_x16)
DECLARE_F32_VELU_UKERNEL_FUNCTION(xnn_f32_velu_ukernel__neonfma_rr1_lut16_p3_x20)
DECLARE_F32_VELU_UKERNEL_FUNCTION(xnn_f32_velu_ukernel__neonfma_rr1_lut16_p3_x24)

DECLARE_F32_VELU_UKERNEL_FUNCTION(xnn_f32_velu_ukernel__neonfma_rr1_p6_x4)
DECLARE_F32_VELU_UKERNEL_FUNCTION(xnn_f32_velu_ukernel__neonfma_rr1_p6_x8)
DECLARE_F32_VELU_UKERNEL_FUNCTION(xnn_f32_velu_ukernel__neonfma_rr1_p6_x12)
DECLARE_F32_VELU_UKERNEL_FUNCTION(xnn_f32_velu_ukernel__neonfma_rr1_p6_x16)
DECLARE_F32_VELU_UKERNEL_FUNCTION(xnn_f32_velu_ukernel__neonfma_rr1_p6_x20)
DECLARE_F32_VELU_UKERNEL_FUNCTION(xnn_f32_velu_ukernel__neonfma_rr1_p6_x24)

DECLARE_F32_VELU_UKERNEL_FUNCTION(xnn_f32_velu_ukernel__sse2_rr2_lut16_p3_x4)
DECLARE_F32_VELU_UKERNEL_FUNCTION(xnn_f32_velu_ukernel__sse2_rr2_lut16_p3_x8)
DECLARE_F32_VELU_UKERNEL_FUNCTION(xnn_f32_velu_ukernel__sse2_rr2_lut16_p3_x12)
DECLARE_F32_VELU_UKERNEL_FUNCTION(xnn_f32_velu_ukernel__sse2_rr2_lut16_p3_x16)
DECLARE_F32_VELU_UKERNEL_FUNCTION(xnn_f32_velu_ukernel__sse2_rr2_lut16_p3_x20)
DECLARE_F32_VELU_UKERNEL_FUNCTION(xnn_f32_velu_ukernel__sse2_rr2_lut16_p3_x24)

DECLARE_F32_VELU_UKERNEL_FUNCTION(xnn_f32_velu_ukernel__sse2_rr2_p6_x4)
DECLARE_F32_VELU_UKERNEL_FUNCTION(xnn_f32_velu_ukernel__sse2_rr2_p6_x8)
DECLARE_F32_VELU_UKERNEL_FUNCTION(xnn_f32_velu_ukernel__sse2_rr2_p6_x12)
DECLARE_F32_VELU_UKERNEL_FUNCTION(xnn_f32_velu_ukernel__sse2_rr2_p6_x16)
DECLARE_F32_VELU_UKERNEL_FUNCTION(xnn_f32_velu_ukernel__sse2_rr2_p6_x20)
DECLARE_F32_VELU_UKERNEL_FUNCTION(xnn_f32_velu_ukernel__sse2_rr2_p6_x24)

DECLARE_F32_VELU_UKERNEL_FUNCTION(xnn_f32_velu_ukernel__sse41_rr2_lut16_p3_x4)
DECLARE_F32_VELU_UKERNEL_FUNCTION(xnn_f32_velu_ukernel__sse41_rr2_lut16_p3_x8)
DECLARE_F32_VELU_UKERNEL_FUNCTION(xnn_f32_velu_ukernel__sse41_rr2_lut16_p3_x12)
DECLARE_F32_VELU_UKERNEL_FUNCTION(xnn_f32_velu_ukernel__sse41_rr2_lut16_p3_x16)
DECLARE_F32_VELU_UKERNEL_FUNCTION(xnn_f32_velu_ukernel__sse41_rr2_lut16_p3_x20)
DECLARE_F32_VELU_UKERNEL_FUNCTION(xnn_f32_velu_ukernel__sse41_rr2_lut16_p3_x24)

DECLARE_F32_VELU_UKERNEL_FUNCTION(xnn_f32_velu_ukernel__sse41_rr2_p6_x4)
DECLARE_F32_VELU_UKERNEL_FUNCTION(xnn_f32_velu_ukernel__sse41_rr2_p6_x8)
DECLARE_F32_VELU_UKERNEL_FUNCTION(xnn_f32_velu_ukernel__sse41_rr2_p6_x12)
DECLARE_F32_VELU_UKERNEL_FUNCTION(xnn_f32_velu_ukernel__sse41_rr2_p6_x16)
DECLARE_F32_VELU_UKERNEL_FUNCTION(xnn_f32_velu_ukernel__sse41_rr2_p6_x20)
DECLARE_F32_VELU_UKERNEL_FUNCTION(xnn_f32_velu_ukernel__sse41_rr2_p6_x24)

DECLARE_F32_VELU_UKERNEL_FUNCTION(xnn_f32_velu_ukernel__avx_rr2_lut4_p4_perm_x8)
DECLARE_F32_VELU_UKERNEL_FUNCTION(xnn_f32_velu_ukernel__avx_rr2_lut4_p4_perm_x16)
DECLARE_F32_VELU_UKERNEL_FUNCTION(xnn_f32_velu_ukernel__avx_rr2_lut4_p4_perm_x24)
DECLARE_F32_VELU_UKERNEL_FUNCTION(xnn_f32_velu_ukernel__avx_rr2_lut4_p4_perm_x32)
DECLARE_F32_VELU_UKERNEL_FUNCTION(xnn_f32_velu_ukernel__avx_rr2_lut4_p4_perm_x40)
DECLARE_F32_VELU_UKERNEL_FUNCTION(xnn_f32_velu_ukernel__avx_rr2_lut4_p4_perm_x48)

DECLARE_F32_VELU_UKERNEL_FUNCTION(xnn_f32_velu_ukernel__avx_rr2_lut16_p3_x8)
DECLARE_F32_VELU_UKERNEL_FUNCTION(xnn_f32_velu_ukernel__avx_rr2_lut16_p3_x16)
DECLARE_F32_VELU_UKERNEL_FUNCTION(xnn_f32_velu_ukernel__avx_rr2_lut16_p3_x24)
DECLARE_F32_VELU_UKERNEL_FUNCTION(xnn_f32_velu_ukernel__avx_rr2_lut16_p3_x32)
DECLARE_F32_VELU_UKERNEL_FUNCTION(xnn_f32_velu_ukernel__avx_rr2_lut16_p3_x40)
DECLARE_F32_VELU_UKERNEL_FUNCTION(xnn_f32_velu_ukernel__avx_rr2_lut16_p3_x48)

DECLARE_F32_VELU_UKERNEL_FUNCTION(xnn_f32_velu_ukernel__avx_rr2_p6_x8)
DECLARE_F32_VELU_UKERNEL_FUNCTION(xnn_f32_velu_ukernel__avx_rr2_p6_x16)
DECLARE_F32_VELU_UKERNEL_FUNCTION(xnn_f32_velu_ukernel__avx_rr2_p6_x24)
DECLARE_F32_VELU_UKERNEL_FUNCTION(xnn_f32_velu_ukernel__avx_rr2_p6_x32)
DECLARE_F32_VELU_UKERNEL_FUNCTION(xnn_f32_velu_ukernel__avx_rr2_p6_x40)
DECLARE_F32_VELU_UKERNEL_FUNCTION(xnn_f32_velu_ukernel__avx_rr2_p6_x48)

DECLARE_F32_VELU_UKERNEL_FUNCTION(xnn_f32_velu_ukernel__avx2_rr1_lut4_p4_perm_x8)
DECLARE_F32_VELU_UKERNEL_FUNCTION(xnn_f32_velu_ukernel__avx2_rr1_lut4_p4_perm_x16)
DECLARE_F32_VELU_UKERNEL_FUNCTION(xnn_f32_velu_ukernel__avx2_rr1_lut4_p4_perm_x24)
DECLARE_F32_VELU_UKERNEL_FUNCTION(xnn_f32_velu_ukernel__avx2_rr1_lut4_p4_perm_x32)
DECLARE_F32_VELU_UKERNEL_FUNCTION(xnn_f32_velu_ukernel__avx2_rr1_lut4_p4_perm_x40)
DECLARE_F32_VELU_UKERNEL_FUNCTION(xnn_f32_velu_ukernel__avx2_rr1_lut4_p4_perm_x48)
DECLARE_F32_VELU_UKERNEL_FUNCTION(xnn_f32_velu_ukernel__avx2_rr1_lut4_p4_perm_x56)
DECLARE_F32_VELU_UKERNEL_FUNCTION(xnn_f32_velu_ukernel__avx2_rr1_lut4_p4_perm_x64)
DECLARE_F32_VELU_UKERNEL_FUNCTION(xnn_f32_velu_ukernel__avx2_rr1_lut4_p4_perm_x72)
DECLARE_F32_VELU_UKERNEL_FUNCTION(xnn_f32_velu_ukernel__avx2_rr1_lut4_p4_perm_x80)

DECLARE_F32_VELU_UKERNEL_FUNCTION(xnn_f32_velu_ukernel__avx2_rr1_lut8_p4_perm_x8)
DECLARE_F32_VELU_UKERNEL_FUNCTION(xnn_f32_velu_ukernel__avx2_rr1_lut8_p4_perm_x16)
DECLARE_F32_VELU_UKERNEL_FUNCTION(xnn_f32_velu_ukernel__avx2_rr1_lut8_p4_perm_x24)
DECLARE_F32_VELU_UKERNEL_FUNCTION(xnn_f32_velu_ukernel__avx2_rr1_lut8_p4_perm_x32)
DECLARE_F32_VELU_UKERNEL_FUNCTION(xnn_f32_velu_ukernel__avx2_rr1_lut8_p4_perm_x40)
DECLARE_F32_VELU_UKERNEL_FUNCTION(xnn_f32_velu_ukernel__avx2_rr1_lut8_p4_perm_x48)
DECLARE_F32_VELU_UKERNEL_FUNCTION(xnn_f32_velu_ukernel__avx2_rr1_lut8_p4_perm_x56)
DECLARE_F32_VELU_UKERNEL_FUNCTION(xnn_f32_velu_ukernel__avx2_rr1_lut8_p4_perm_x64)
DECLARE_F32_VELU_UKERNEL_FUNCTION(xnn_f32_velu_ukernel__avx2_rr1_lut8_p4_perm_x72)
DECLARE_F32_VELU_UKERNEL_FUNCTION(xnn_f32_velu_ukernel__avx2_rr1_lut8_p4_perm_x80)

DECLARE_F32_VELU_UKERNEL_FUNCTION(xnn_f32_velu_ukernel__avx2_rr1_lut16_p3_gather_x8)
DECLARE_F32_VELU_UKERNEL_FUNCTION(xnn_f32_velu_ukernel__avx2_rr1_lut16_p3_gather_x16)
DECLARE_F32_VELU_UKERNEL_FUNCTION(xnn_f32_velu_ukernel__avx2_rr1_lut16_p3_gather_x24)
DECLARE_F32_VELU_UKERNEL_FUNCTION(xnn_f32_velu_ukernel__avx2_rr1_lut16_p3_gather_x32)
DECLARE_F32_VELU_UKERNEL_FUNCTION(xnn_f32_velu_ukernel__avx2_rr1_lut16_p3_gather_x40)
DECLARE_F32_VELU_UKERNEL_FUNCTION(xnn_f32_velu_ukernel__avx2_rr1_lut16_p3_gather_x48)
DECLARE_F32_VELU_UKERNEL_FUNCTION(xnn_f32_velu_ukernel__avx2_rr1_lut16_p3_gather_x56)
DECLARE_F32_VELU_UKERNEL_FUNCTION(xnn_f32_velu_ukernel__avx2_rr1_lut16_p3_gather_x64)
DECLARE_F32_VELU_UKERNEL_FUNCTION(xnn_f32_velu_ukernel__avx2_rr1_lut16_p3_gather_x72)
DECLARE_F32_VELU_UKERNEL_FUNCTION(xnn_f32_velu_ukernel__avx2_rr1_lut16_p3_gather_x80)

DECLARE_F32_VELU_UKERNEL_FUNCTION(xnn_f32_velu_ukernel__avx2_rr1_p6_x8)
DECLARE_F32_VELU_UKERNEL_FUNCTION(xnn_f32_velu_ukernel__avx2_rr1_p6_x16)
DECLARE_F32_VELU_UKERNEL_FUNCTION(xnn_f32_velu_ukernel__avx2_rr1_p6_x24)
DECLARE_F32_VELU_UKERNEL_FUNCTION(xnn_f32_velu_ukernel__avx2_rr1_p6_x32)
DECLARE_F32_VELU_UKERNEL_FUNCTION(xnn_f32_velu_ukernel__avx2_rr1_p6_x40)
DECLARE_F32_VELU_UKERNEL_FUNCTION(xnn_f32_velu_ukernel__avx2_rr1_p6_x48)
DECLARE_F32_VELU_UKERNEL_FUNCTION(xnn_f32_velu_ukernel__avx2_rr1_p6_x56)
DECLARE_F32_VELU_UKERNEL_FUNCTION(xnn_f32_velu_ukernel__avx2_rr1_p6_x64)
DECLARE_F32_VELU_UKERNEL_FUNCTION(xnn_f32_velu_ukernel__avx2_rr1_p6_x72)
DECLARE_F32_VELU_UKERNEL_FUNCTION(xnn_f32_velu_ukernel__avx2_rr1_p6_x80)

DECLARE_F32_VELU_UKERNEL_FUNCTION(xnn_f32_velu_ukernel__avx512f_rr1_lut16_p3_perm_x16)
DECLARE_F32_VELU_UKERNEL_FUNCTION(xnn_f32_velu_ukernel__avx512f_rr1_lut16_p3_perm_x32)
DECLARE_F32_VELU_UKERNEL_FUNCTION(xnn_f32_velu_ukernel__avx512f_rr1_lut16_p3_perm_x48)
DECLARE_F32_VELU_UKERNEL_FUNCTION(xnn_f32_velu_ukernel__avx512f_rr1_lut16_p3_perm_x64)
DECLARE_F32_VELU_UKERNEL_FUNCTION(xnn_f32_velu_ukernel__avx512f_rr1_lut16_p3_perm_x80)
DECLARE_F32_VELU_UKERNEL_FUNCTION(xnn_f32_velu_ukernel__avx512f_rr1_lut16_p3_perm_x96)
DECLARE_F32_VELU_UKERNEL_FUNCTION(xnn_f32_velu_ukernel__avx512f_rr1_lut16_p3_perm_x112)
DECLARE_F32_VELU_UKERNEL_FUNCTION(xnn_f32_velu_ukernel__avx512f_rr1_lut16_p3_perm_x128)

DECLARE_F32_VELU_UKERNEL_FUNCTION(xnn_f32_velu_ukernel__avx512f_rr1_p6_x16)
DECLARE_F32_VELU_UKERNEL_FUNCTION(xnn_f32_velu_ukernel__avx512f_rr1_p6_x32)
DECLARE_F32_VELU_UKERNEL_FUNCTION(xnn_f32_velu_ukernel__avx512f_rr1_p6_x48)
DECLARE_F32_VELU_UKERNEL_FUNCTION(xnn_f32_velu_ukernel__avx512f_rr1_p6_x64)
DECLARE_F32_VELU_UKERNEL_FUNCTION(xnn_f32_velu_ukernel__avx512f_rr1_p6_x80)
DECLARE_F32_VELU_UKERNEL_FUNCTION(xnn_f32_velu_ukernel__avx512f_rr1_p6_x96)
DECLARE_F32_VELU_UKERNEL_FUNCTION(xnn_f32_velu_ukernel__avx512f_rr1_p6_x112)
DECLARE_F32_VELU_UKERNEL_FUNCTION(xnn_f32_velu_ukernel__avx512f_rr1_p6_x128)

DECLARE_F32_VELU_UKERNEL_FUNCTION(xnn_f32_velu_ukernel__wasmsimd_arm_rr2_lut16_p3_x4)
DECLARE_F32_VELU_UKERNEL_FUNCTION(xnn_f32_velu_ukernel__wasmsimd_arm_rr2_lut16_p3_x8)
DECLARE_F32_VELU_UKERNEL_FUNCTION(xnn_f32_velu_ukernel__wasmsimd_arm_rr2_lut16_p3_x12)
DECLARE_F32_VELU_UKERNEL_FUNCTION(xnn_f32_velu_ukernel__wasmsimd_arm_rr2_lut16_p3_x16)
DECLARE_F32_VELU_UKERNEL_FUNCTION(xnn_f32_velu_ukernel__wasmsimd_arm_rr2_lut16_p3_x20)
DECLARE_F32_VELU_UKERNEL_FUNCTION(xnn_f32_velu_ukernel__wasmsimd_arm_rr2_lut16_p3_x24)

DECLARE_F32_VELU_UKERNEL_FUNCTION(xnn_f32_velu_ukernel__wasmsimd_x86_rr2_lut16_p3_x4)
DECLARE_F32_VELU_UKERNEL_FUNCTION(xnn_f32_velu_ukernel__wasmsimd_x86_rr2_lut16_p3_x8)
DECLARE_F32_VELU_UKERNEL_FUNCTION(xnn_f32_velu_ukernel__wasmsimd_x86_rr2_lut16_p3_x12)
DECLARE_F32_VELU_UKERNEL_FUNCTION(xnn_f32_velu_ukernel__wasmsimd_x86_rr2_lut16_p3_x16)
DECLARE_F32_VELU_UKERNEL_FUNCTION(xnn_f32_velu_ukernel__wasmsimd_x86_rr2_lut16_p3_x20)
DECLARE_F32_VELU_UKERNEL_FUNCTION(xnn_f32_velu_ukernel__wasmsimd_x86_rr2_lut16_p3_x24)

DECLARE_F32_VELU_UKERNEL_FUNCTION(xnn_f32_velu_ukernel__wasmrelaxedsimd_rr2_lut16_p3_x4)
DECLARE_F32_VELU_UKERNEL_FUNCTION(xnn_f32_velu_ukernel__wasmrelaxedsimd_rr2_lut16_p3_x8)
DECLARE_F32_VELU_UKERNEL_FUNCTION(xnn_f32_velu_ukernel__wasmrelaxedsimd_rr2_lut16_p3_x12)
DECLARE_F32_VELU_UKERNEL_FUNCTION(xnn_f32_velu_ukernel__wasmrelaxedsimd_rr2_lut16_p3_x16)
DECLARE_F32_VELU_UKERNEL_FUNCTION(xnn_f32_velu_ukernel__wasmrelaxedsimd_rr2_lut16_p3_x20)
DECLARE_F32_VELU_UKERNEL_FUNCTION(xnn_f32_velu_ukernel__wasmrelaxedsimd_rr2_lut16_p3_x24)

DECLARE_F32_VELU_UKERNEL_FUNCTION(xnn_f32_velu_ukernel__wasmrelaxedsimd_fma_rr2_lut16_p3_x4)
DECLARE_F32_VELU_UKERNEL_FUNCTION(xnn_f32_velu_ukernel__wasmrelaxedsimd_fma_rr2_lut16_p3_x8)
DECLARE_F32_VELU_UKERNEL_FUNCTION(xnn_f32_velu_ukernel__wasmrelaxedsimd_fma_rr2_lut16_p3_x12)
DECLARE_F32_VELU_UKERNEL_FUNCTION(xnn_f32_velu_ukernel__wasmrelaxedsimd_fma_rr2_lut16_p3_x16)
DECLARE_F32_VELU_UKERNEL_FUNCTION(xnn_f32_velu_ukernel__wasmrelaxedsimd_fma_rr2_lut16_p3_x20)
DECLARE_F32_VELU_UKERNEL_FUNCTION(xnn_f32_velu_ukernel__wasmrelaxedsimd_fma_rr2_lut16_p3_x24)

DECLARE_F32_VELU_UKERNEL_FUNCTION(xnn_f32_velu_ukernel__wasmsimd_arm_rr2_p6_x4)
DECLARE_F32_VELU_UKERNEL_FUNCTION(xnn_f32_velu_ukernel__wasmsimd_arm_rr2_p6_x8)
DECLARE_F32_VELU_UKERNEL_FUNCTION(xnn_f32_velu_ukernel__wasmsimd_arm_rr2_p6_x12)
DECLARE_F32_VELU_UKERNEL_FUNCTION(xnn_f32_velu_ukernel__wasmsimd_arm_rr2_p6_x16)
DECLARE_F32_VELU_UKERNEL_FUNCTION(xnn_f32_velu_ukernel__wasmsimd_arm_rr2_p6_x20)
DECLARE_F32_VELU_UKERNEL_FUNCTION(xnn_f32_velu_ukernel__wasmsimd_arm_rr2_p6_x24)

DECLARE_F32_VELU_UKERNEL_FUNCTION(xnn_f32_velu_ukernel__wasmsimd_x86_rr2_p6_x4)
DECLARE_F32_VELU_UKERNEL_FUNCTION(xnn_f32_velu_ukernel__wasmsimd_x86_rr2_p6_x8)
DECLARE_F32_VELU_UKERNEL_FUNCTION(xnn_f32_velu_ukernel__wasmsimd_x86_rr2_p6_x12)
DECLARE_F32_VELU_UKERNEL_FUNCTION(xnn_f32_velu_ukernel__wasmsimd_x86_rr2_p6_x16)
DECLARE_F32_VELU_UKERNEL_FUNCTION(xnn_f32_velu_ukernel__wasmsimd_x86_rr2_p6_x20)
DECLARE_F32_VELU_UKERNEL_FUNCTION(xnn_f32_velu_ukernel__wasmsimd_x86_rr2_p6_x24)

DECLARE_F32_VELU_UKERNEL_FUNCTION(xnn_f32_velu_ukernel__wasmrelaxedsimd_rr2_p6_x4)
DECLARE_F32_VELU_UKERNEL_FUNCTION(xnn_f32_velu_ukernel__wasmrelaxedsimd_rr2_p6_x8)
DECLARE_F32_VELU_UKERNEL_FUNCTION(xnn_f32_velu_ukernel__wasmrelaxedsimd_rr2_p6_x12)
DECLARE_F32_VELU_UKERNEL_FUNCTION(xnn_f32_velu_ukernel__wasmrelaxedsimd_rr2_p6_x16)
DECLARE_F32_VELU_UKERNEL_FUNCTION(xnn_f32_velu_ukernel__wasmrelaxedsimd_rr2_p6_x20)
DECLARE_F32_VELU_UKERNEL_FUNCTION(xnn_f32_velu_ukernel__wasmrelaxedsimd_rr2_p6_x24)

DECLARE_F32_VELU_UKERNEL_FUNCTION(xnn_f32_velu_ukernel__wasmrelaxedsimd_fma_rr2_p6_x4)
DECLARE_F32_VELU_UKERNEL_FUNCTION(xnn_f32_velu_ukernel__wasmrelaxedsimd_fma_rr2_p6_x8)
DECLARE_F32_VELU_UKERNEL_FUNCTION(xnn_f32_velu_ukernel__wasmrelaxedsimd_fma_rr2_p6_x12)
DECLARE_F32_VELU_UKERNEL_FUNCTION(xnn_f32_velu_ukernel__wasmrelaxedsimd_fma_rr2_p6_x16)
DECLARE_F32_VELU_UKERNEL_FUNCTION(xnn_f32_velu_ukernel__wasmrelaxedsimd_fma_rr2_p6_x20)
DECLARE_F32_VELU_UKERNEL_FUNCTION(xnn_f32_velu_ukernel__wasmrelaxedsimd_fma_rr2_p6_x24)

DECLARE_F32_VELU_UKERNEL_FUNCTION(xnn_f32_velu_ukernel__wasm_rr2_lut16_p3_x1)
DECLARE_F32_VELU_UKERNEL_FUNCTION(xnn_f32_velu_ukernel__wasm_rr2_lut16_p3_x2)
DECLARE_F32_VELU_UKERNEL_FUNCTION(xnn_f32_velu_ukernel__wasm_rr2_lut16_p3_x3)
DECLARE_F32_VELU_UKERNEL_FUNCTION(xnn_f32_velu_ukernel__wasm_rr2_lut16_p3_x4)
DECLARE_F32_VELU_UKERNEL_FUNCTION(xnn_f32_velu_ukernel__wasm_rr2_lut16_p3_x5)
DECLARE_F32_VELU_UKERNEL_FUNCTION(xnn_f32_velu_ukernel__wasm_rr2_lut16_p3_x6)

DECLARE_F32_VELU_UKERNEL_FUNCTION(xnn_f32_velu_ukernel__wasm_rr2_p6_x1)
DECLARE_F32_VELU_UKERNEL_FUNCTION(xnn_f32_velu_ukernel__wasm_rr2_p6_x2)
DECLARE_F32_VELU_UKERNEL_FUNCTION(xnn_f32_velu_ukernel__wasm_rr2_p6_x3)
DECLARE_F32_VELU_UKERNEL_FUNCTION(xnn_f32_velu_ukernel__wasm_rr2_p6_x4)
DECLARE_F32_VELU_UKERNEL_FUNCTION(xnn_f32_velu_ukernel__wasm_rr2_p6_x5)
DECLARE_F32_VELU_UKERNEL_FUNCTION(xnn_f32_velu_ukernel__wasm_rr2_p6_x6)

DECLARE_F32_VELU_UKERNEL_FUNCTION(xnn_f32_velu_ukernel__scalar_rr2_lut16_p3_x1)
DECLARE_F32_VELU_UKERNEL_FUNCTION(xnn_f32_velu_ukernel__scalar_rr2_lut16_p3_x2)
DECLARE_F32_VELU_UKERNEL_FUNCTION(xnn_f32_velu_ukernel__scalar_rr2_lut16_p3_x3)
DECLARE_F32_VELU_UKERNEL_FUNCTION(xnn_f32_velu_ukernel__scalar_rr2_lut16_p3_x4)
DECLARE_F32_VELU_UKERNEL_FUNCTION(xnn_f32_velu_ukernel__scalar_rr2_lut16_p3_x5)
DECLARE_F32_VELU_UKERNEL_FUNCTION(xnn_f32_velu_ukernel__scalar_rr2_lut16_p3_x6)

DECLARE_F32_VELU_UKERNEL_FUNCTION(xnn_f32_velu_ukernel__scalar_rr2_p6_x1)
DECLARE_F32_VELU_UKERNEL_FUNCTION(xnn_f32_velu_ukernel__scalar_rr2_p6_x2)
DECLARE_F32_VELU_UKERNEL_FUNCTION(xnn_f32_velu_ukernel__scalar_rr2_p6_x3)
DECLARE_F32_VELU_UKERNEL_FUNCTION(xnn_f32_velu_ukernel__scalar_rr2_p6_x4)
DECLARE_F32_VELU_UKERNEL_FUNCTION(xnn_f32_velu_ukernel__scalar_rr2_p6_x5)
DECLARE_F32_VELU_UKERNEL_FUNCTION(xnn_f32_velu_ukernel__scalar_rr2_p6_x6)


#define DECLARE_F32_VHSWISH_UKERNEL_FUNCTION(fn_name) \
  XNN_INTERNAL void fn_name(                          \
      size_t n,                                       \
      const float* x,                                 \
      float* y,                                       \
      const union xnn_f32_hswish_params* params);

DECLARE_F32_VHSWISH_UKERNEL_FUNCTION(xnn_f32_vhswish_ukernel__neon_x4)
DECLARE_F32_VHSWISH_UKERNEL_FUNCTION(xnn_f32_vhswish_ukernel__neon_x8)
DECLARE_F32_VHSWISH_UKERNEL_FUNCTION(xnn_f32_vhswish_ukernel__neon_x16)

DECLARE_F32_VHSWISH_UKERNEL_FUNCTION(xnn_f32_vhswish_ukernel__sse_x4)
DECLARE_F32_VHSWISH_UKERNEL_FUNCTION(xnn_f32_vhswish_ukernel__sse_x8)

DECLARE_F32_VHSWISH_UKERNEL_FUNCTION(xnn_f32_vhswish_ukernel__avx_x8)
DECLARE_F32_VHSWISH_UKERNEL_FUNCTION(xnn_f32_vhswish_ukernel__avx_x16)

DECLARE_F32_VHSWISH_UKERNEL_FUNCTION(xnn_f32_vhswish_ukernel__fma3_x8)
DECLARE_F32_VHSWISH_UKERNEL_FUNCTION(xnn_f32_vhswish_ukernel__fma3_x16)

DECLARE_F32_VHSWISH_UKERNEL_FUNCTION(xnn_f32_vhswish_ukernel__avx512f_x16)
DECLARE_F32_VHSWISH_UKERNEL_FUNCTION(xnn_f32_vhswish_ukernel__avx512f_x32)

DECLARE_F32_VHSWISH_UKERNEL_FUNCTION(xnn_f32_vhswish_ukernel__wasmsimd_x4)
DECLARE_F32_VHSWISH_UKERNEL_FUNCTION(xnn_f32_vhswish_ukernel__wasmsimd_x8)
DECLARE_F32_VHSWISH_UKERNEL_FUNCTION(xnn_f32_vhswish_ukernel__wasmsimd_x16)

DECLARE_F32_VHSWISH_UKERNEL_FUNCTION(xnn_f32_vhswish_ukernel__wasm_x1)
DECLARE_F32_VHSWISH_UKERNEL_FUNCTION(xnn_f32_vhswish_ukernel__wasm_x2)
DECLARE_F32_VHSWISH_UKERNEL_FUNCTION(xnn_f32_vhswish_ukernel__wasm_x4)

DECLARE_F32_VHSWISH_UKERNEL_FUNCTION(xnn_f32_vhswish_ukernel__scalar_x1)
DECLARE_F32_VHSWISH_UKERNEL_FUNCTION(xnn_f32_vhswish_ukernel__scalar_x2)
DECLARE_F32_VHSWISH_UKERNEL_FUNCTION(xnn_f32_vhswish_ukernel__scalar_x4)


#define DECLARE_F16_VLRELU_UKERNEL_FUNCTION(fn_name) \
  XNN_INTERNAL void fn_name(                         \
      size_t n,                                      \
      const void* x,                                 \
      void* y,                                       \
      const union xnn_f16_lrelu_params* params);

DECLARE_F16_VLRELU_UKERNEL_FUNCTION(xnn_f16_vlrelu_ukernel__neonfp16arith_x8)
DECLARE_F16_VLRELU_UKERNEL_FUNCTION(xnn_f16_vlrelu_ukernel__neonfp16arith_x16)

DECLARE_F16_VLRELU_UKERNEL_FUNCTION(xnn_f16_vlrelu_ukernel__f16c_x8)
DECLARE_F16_VLRELU_UKERNEL_FUNCTION(xnn_f16_vlrelu_ukernel__f16c_x16)


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

DECLARE_F32_VLRELU_UKERNEL_FUNCTION(xnn_f32_vlrelu_ukernel__wasmsimd_laneselect_x4)
DECLARE_F32_VLRELU_UKERNEL_FUNCTION(xnn_f32_vlrelu_ukernel__wasmsimd_laneselect_x8)

DECLARE_F32_VLRELU_UKERNEL_FUNCTION(xnn_f32_vlrelu_ukernel__wasmsimd_iminmax_x4)
DECLARE_F32_VLRELU_UKERNEL_FUNCTION(xnn_f32_vlrelu_ukernel__wasmsimd_iminmax_x8)

DECLARE_F32_VLRELU_UKERNEL_FUNCTION(xnn_f32_vlrelu_ukernel__wasmrelaxedsimd_laneselect_x4)
DECLARE_F32_VLRELU_UKERNEL_FUNCTION(xnn_f32_vlrelu_ukernel__wasmrelaxedsimd_laneselect_x8)

DECLARE_F32_VLRELU_UKERNEL_FUNCTION(xnn_f32_vlrelu_ukernel__wasmrelaxedsimd_iminmax_x4)
DECLARE_F32_VLRELU_UKERNEL_FUNCTION(xnn_f32_vlrelu_ukernel__wasmrelaxedsimd_iminmax_x8)

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

DECLARE_F32_VNEG_UKERNEL_FUNCTION(xnn_f32_vneg_ukernel__wasmsimd_x4)
DECLARE_F32_VNEG_UKERNEL_FUNCTION(xnn_f32_vneg_ukernel__wasmsimd_x8)

DECLARE_F32_VNEG_UKERNEL_FUNCTION(xnn_f32_vneg_ukernel__scalar_x1)
DECLARE_F32_VNEG_UKERNEL_FUNCTION(xnn_f32_vneg_ukernel__scalar_x2)
DECLARE_F32_VNEG_UKERNEL_FUNCTION(xnn_f32_vneg_ukernel__scalar_x4)


#define DECLARE_F32_VRELU_UKERNEL_FUNCTION(fn_name) \
  XNN_INTERNAL void fn_name(                        \
      size_t n,                                     \
      const float* x,                               \
      float* y,                                     \
      const union xnn_f32_relu_params* params);

DECLARE_F32_VRELU_UKERNEL_FUNCTION(xnn_f32_vrelu_ukernel__avx_x8)
DECLARE_F32_VRELU_UKERNEL_FUNCTION(xnn_f32_vrelu_ukernel__avx_x16)
DECLARE_F32_VRELU_UKERNEL_FUNCTION(xnn_f32_vrelu_ukernel__avx512f_x16)
DECLARE_F32_VRELU_UKERNEL_FUNCTION(xnn_f32_vrelu_ukernel__avx512f_x32)
DECLARE_F32_VRELU_UKERNEL_FUNCTION(xnn_f32_vrelu_ukernel__neon_x4)
DECLARE_F32_VRELU_UKERNEL_FUNCTION(xnn_f32_vrelu_ukernel__neon_x8)
DECLARE_F32_VRELU_UKERNEL_FUNCTION(xnn_f32_vrelu_ukernel__scalar_x1)
DECLARE_F32_VRELU_UKERNEL_FUNCTION(xnn_f32_vrelu_ukernel__scalar_x2)
DECLARE_F32_VRELU_UKERNEL_FUNCTION(xnn_f32_vrelu_ukernel__scalar_x4)
DECLARE_F32_VRELU_UKERNEL_FUNCTION(xnn_f32_vrelu_ukernel__scalar_x8)
DECLARE_F32_VRELU_UKERNEL_FUNCTION(xnn_f32_vrelu_ukernel__sse_x4)
DECLARE_F32_VRELU_UKERNEL_FUNCTION(xnn_f32_vrelu_ukernel__sse_x8)
DECLARE_F32_VRELU_UKERNEL_FUNCTION(xnn_f32_vrelu_ukernel__wasm_x1)
DECLARE_F32_VRELU_UKERNEL_FUNCTION(xnn_f32_vrelu_ukernel__wasm_x2)
DECLARE_F32_VRELU_UKERNEL_FUNCTION(xnn_f32_vrelu_ukernel__wasm_x4)
DECLARE_F32_VRELU_UKERNEL_FUNCTION(xnn_f32_vrelu_ukernel__wasm_x8)
DECLARE_F32_VRELU_UKERNEL_FUNCTION(xnn_f32_vrelu_ukernel__wasmsimd_x4)
DECLARE_F32_VRELU_UKERNEL_FUNCTION(xnn_f32_vrelu_ukernel__wasmsimd_x8)
DECLARE_F32_VRELU_UKERNEL_FUNCTION(xnn_f32_vrelu_ukernel__wasmsimd_x16)
DECLARE_F32_VRELU_UKERNEL_FUNCTION(xnn_f32_vrelu_ukernel__wasm32_shr_x1)
DECLARE_F32_VRELU_UKERNEL_FUNCTION(xnn_f32_vrelu_ukernel__wasm32_shr_x2)
DECLARE_F32_VRELU_UKERNEL_FUNCTION(xnn_f32_vrelu_ukernel__wasm32_shr_x4)


#define DECLARE_F32_VRND_UKERNEL_FUNCTION(fn_name) \
  XNN_INTERNAL void fn_name(                       \
      size_t n,                                    \
      const float* x,                              \
      float* y,                                    \
      const union xnn_f32_rnd_params* params);

DECLARE_F32_VRND_UKERNEL_FUNCTION(xnn_f32_vrndne_ukernel__avx_x8)
DECLARE_F32_VRND_UKERNEL_FUNCTION(xnn_f32_vrndne_ukernel__avx_x16)
DECLARE_F32_VRND_UKERNEL_FUNCTION(xnn_f32_vrndne_ukernel__avx512f_x16)
DECLARE_F32_VRND_UKERNEL_FUNCTION(xnn_f32_vrndne_ukernel__avx512f_x32)
DECLARE_F32_VRND_UKERNEL_FUNCTION(xnn_f32_vrndne_ukernel__neon_x4)
DECLARE_F32_VRND_UKERNEL_FUNCTION(xnn_f32_vrndne_ukernel__neon_x8)
DECLARE_F32_VRND_UKERNEL_FUNCTION(xnn_f32_vrndne_ukernel__neonv8_x4)
DECLARE_F32_VRND_UKERNEL_FUNCTION(xnn_f32_vrndne_ukernel__neonv8_x8)
DECLARE_F32_VRND_UKERNEL_FUNCTION(xnn_f32_vrndne_ukernel__scalar_libm_x1)
DECLARE_F32_VRND_UKERNEL_FUNCTION(xnn_f32_vrndne_ukernel__scalar_libm_x2)
DECLARE_F32_VRND_UKERNEL_FUNCTION(xnn_f32_vrndne_ukernel__scalar_libm_x4)
DECLARE_F32_VRND_UKERNEL_FUNCTION(xnn_f32_vrndne_ukernel__sse2_x4)
DECLARE_F32_VRND_UKERNEL_FUNCTION(xnn_f32_vrndne_ukernel__sse2_x8)
DECLARE_F32_VRND_UKERNEL_FUNCTION(xnn_f32_vrndne_ukernel__sse41_x4)
DECLARE_F32_VRND_UKERNEL_FUNCTION(xnn_f32_vrndne_ukernel__sse41_x8)
DECLARE_F32_VRND_UKERNEL_FUNCTION(xnn_f32_vrndne_ukernel__wasmsimd_x4)
DECLARE_F32_VRND_UKERNEL_FUNCTION(xnn_f32_vrndne_ukernel__wasmsimd_x8)

DECLARE_F32_VRND_UKERNEL_FUNCTION(xnn_f32_vrndz_ukernel__avx_x8)
DECLARE_F32_VRND_UKERNEL_FUNCTION(xnn_f32_vrndz_ukernel__avx_x16)
DECLARE_F32_VRND_UKERNEL_FUNCTION(xnn_f32_vrndz_ukernel__avx512f_x16)
DECLARE_F32_VRND_UKERNEL_FUNCTION(xnn_f32_vrndz_ukernel__avx512f_x32)
DECLARE_F32_VRND_UKERNEL_FUNCTION(xnn_f32_vrndz_ukernel__neon_x4)
DECLARE_F32_VRND_UKERNEL_FUNCTION(xnn_f32_vrndz_ukernel__neon_x8)
DECLARE_F32_VRND_UKERNEL_FUNCTION(xnn_f32_vrndz_ukernel__neonv8_x4)
DECLARE_F32_VRND_UKERNEL_FUNCTION(xnn_f32_vrndz_ukernel__neonv8_x8)
DECLARE_F32_VRND_UKERNEL_FUNCTION(xnn_f32_vrndz_ukernel__scalar_libm_x1)
DECLARE_F32_VRND_UKERNEL_FUNCTION(xnn_f32_vrndz_ukernel__scalar_libm_x2)
DECLARE_F32_VRND_UKERNEL_FUNCTION(xnn_f32_vrndz_ukernel__scalar_libm_x4)
DECLARE_F32_VRND_UKERNEL_FUNCTION(xnn_f32_vrndz_ukernel__sse2_x4)
DECLARE_F32_VRND_UKERNEL_FUNCTION(xnn_f32_vrndz_ukernel__sse2_x8)
DECLARE_F32_VRND_UKERNEL_FUNCTION(xnn_f32_vrndz_ukernel__sse41_x4)
DECLARE_F32_VRND_UKERNEL_FUNCTION(xnn_f32_vrndz_ukernel__sse41_x8)
DECLARE_F32_VRND_UKERNEL_FUNCTION(xnn_f32_vrndz_ukernel__wasmsimd_x4)
DECLARE_F32_VRND_UKERNEL_FUNCTION(xnn_f32_vrndz_ukernel__wasmsimd_x8)

DECLARE_F32_VRND_UKERNEL_FUNCTION(xnn_f32_vrndu_ukernel__avx_x8)
DECLARE_F32_VRND_UKERNEL_FUNCTION(xnn_f32_vrndu_ukernel__avx_x16)
DECLARE_F32_VRND_UKERNEL_FUNCTION(xnn_f32_vrndu_ukernel__avx512f_x16)
DECLARE_F32_VRND_UKERNEL_FUNCTION(xnn_f32_vrndu_ukernel__avx512f_x32)
DECLARE_F32_VRND_UKERNEL_FUNCTION(xnn_f32_vrndu_ukernel__neon_x4)
DECLARE_F32_VRND_UKERNEL_FUNCTION(xnn_f32_vrndu_ukernel__neon_x8)
DECLARE_F32_VRND_UKERNEL_FUNCTION(xnn_f32_vrndu_ukernel__neonv8_x4)
DECLARE_F32_VRND_UKERNEL_FUNCTION(xnn_f32_vrndu_ukernel__neonv8_x8)
DECLARE_F32_VRND_UKERNEL_FUNCTION(xnn_f32_vrndu_ukernel__scalar_libm_x1)
DECLARE_F32_VRND_UKERNEL_FUNCTION(xnn_f32_vrndu_ukernel__scalar_libm_x2)
DECLARE_F32_VRND_UKERNEL_FUNCTION(xnn_f32_vrndu_ukernel__scalar_libm_x4)
DECLARE_F32_VRND_UKERNEL_FUNCTION(xnn_f32_vrndu_ukernel__sse2_x4)
DECLARE_F32_VRND_UKERNEL_FUNCTION(xnn_f32_vrndu_ukernel__sse2_x8)
DECLARE_F32_VRND_UKERNEL_FUNCTION(xnn_f32_vrndu_ukernel__sse41_x4)
DECLARE_F32_VRND_UKERNEL_FUNCTION(xnn_f32_vrndu_ukernel__sse41_x8)
DECLARE_F32_VRND_UKERNEL_FUNCTION(xnn_f32_vrndu_ukernel__wasmsimd_x4)
DECLARE_F32_VRND_UKERNEL_FUNCTION(xnn_f32_vrndu_ukernel__wasmsimd_x8)

DECLARE_F32_VRND_UKERNEL_FUNCTION(xnn_f32_vrndd_ukernel__avx_x8)
DECLARE_F32_VRND_UKERNEL_FUNCTION(xnn_f32_vrndd_ukernel__avx_x16)
DECLARE_F32_VRND_UKERNEL_FUNCTION(xnn_f32_vrndd_ukernel__avx512f_x16)
DECLARE_F32_VRND_UKERNEL_FUNCTION(xnn_f32_vrndd_ukernel__avx512f_x32)
DECLARE_F32_VRND_UKERNEL_FUNCTION(xnn_f32_vrndd_ukernel__neon_x4)
DECLARE_F32_VRND_UKERNEL_FUNCTION(xnn_f32_vrndd_ukernel__neon_x8)
DECLARE_F32_VRND_UKERNEL_FUNCTION(xnn_f32_vrndd_ukernel__neonv8_x4)
DECLARE_F32_VRND_UKERNEL_FUNCTION(xnn_f32_vrndd_ukernel__neonv8_x8)
DECLARE_F32_VRND_UKERNEL_FUNCTION(xnn_f32_vrndd_ukernel__scalar_libm_x1)
DECLARE_F32_VRND_UKERNEL_FUNCTION(xnn_f32_vrndd_ukernel__scalar_libm_x2)
DECLARE_F32_VRND_UKERNEL_FUNCTION(xnn_f32_vrndd_ukernel__scalar_libm_x4)
DECLARE_F32_VRND_UKERNEL_FUNCTION(xnn_f32_vrndd_ukernel__sse2_x4)
DECLARE_F32_VRND_UKERNEL_FUNCTION(xnn_f32_vrndd_ukernel__sse2_x8)
DECLARE_F32_VRND_UKERNEL_FUNCTION(xnn_f32_vrndd_ukernel__sse41_x4)
DECLARE_F32_VRND_UKERNEL_FUNCTION(xnn_f32_vrndd_ukernel__sse41_x8)
DECLARE_F32_VRND_UKERNEL_FUNCTION(xnn_f32_vrndd_ukernel__wasmsimd_x4)
DECLARE_F32_VRND_UKERNEL_FUNCTION(xnn_f32_vrndd_ukernel__wasmsimd_x8)


#define DECLARE_F32_VSQRT_UKERNEL_FUNCTION(fn_name) \
  XNN_INTERNAL void fn_name(                        \
      size_t n,                                     \
      const float* x,                               \
      float* y,                                     \
      const union xnn_f32_sqrt_params* params);

DECLARE_F32_VSQRT_UKERNEL_FUNCTION(xnn_f32_vsqrt_ukernel__aarch64_neon_sqrt_x4)
DECLARE_F32_VSQRT_UKERNEL_FUNCTION(xnn_f32_vsqrt_ukernel__aarch64_neon_sqrt_x8)

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


#define DECLARE_F32_VSIGMOID_UKERNEL_FUNCTION(fn_name) \
  XNN_INTERNAL void fn_name(                           \
      size_t n,                                        \
      const float* x,                                  \
      float* y,                                        \
      const union xnn_f32_sigmoid_params* params);

DECLARE_F32_VSIGMOID_UKERNEL_FUNCTION(xnn_f32_vsigmoid_ukernel__aarch64_neonfma_rr1_p5_div_x4)
DECLARE_F32_VSIGMOID_UKERNEL_FUNCTION(xnn_f32_vsigmoid_ukernel__aarch64_neonfma_rr1_p5_div_x8)
DECLARE_F32_VSIGMOID_UKERNEL_FUNCTION(xnn_f32_vsigmoid_ukernel__aarch64_neonfma_rr1_p5_div_x12)
DECLARE_F32_VSIGMOID_UKERNEL_FUNCTION(xnn_f32_vsigmoid_ukernel__aarch64_neonfma_rr1_p5_div_x16)
DECLARE_F32_VSIGMOID_UKERNEL_FUNCTION(xnn_f32_vsigmoid_ukernel__aarch64_neonfma_rr1_p5_div_x20)
DECLARE_F32_VSIGMOID_UKERNEL_FUNCTION(xnn_f32_vsigmoid_ukernel__aarch64_neonfma_rr1_p5_div_x24)

DECLARE_F32_VSIGMOID_UKERNEL_FUNCTION(xnn_f32_vsigmoid_ukernel__neonfma_rr1_p5_nr2fma_x4)
DECLARE_F32_VSIGMOID_UKERNEL_FUNCTION(xnn_f32_vsigmoid_ukernel__neonfma_rr1_p5_nr2fma_x8)
DECLARE_F32_VSIGMOID_UKERNEL_FUNCTION(xnn_f32_vsigmoid_ukernel__neonfma_rr1_p5_nr2fma_x12)
DECLARE_F32_VSIGMOID_UKERNEL_FUNCTION(xnn_f32_vsigmoid_ukernel__neonfma_rr1_p5_nr2fma_x16)
DECLARE_F32_VSIGMOID_UKERNEL_FUNCTION(xnn_f32_vsigmoid_ukernel__neonfma_rr1_p5_nr2fma_x20)
DECLARE_F32_VSIGMOID_UKERNEL_FUNCTION(xnn_f32_vsigmoid_ukernel__neonfma_rr1_p5_nr2fma_x24)

DECLARE_F32_VSIGMOID_UKERNEL_FUNCTION(xnn_f32_vsigmoid_ukernel__neonfma_rr1_p5_nr1recps1fma_x4)
DECLARE_F32_VSIGMOID_UKERNEL_FUNCTION(xnn_f32_vsigmoid_ukernel__neonfma_rr1_p5_nr1recps1fma_x8)
DECLARE_F32_VSIGMOID_UKERNEL_FUNCTION(xnn_f32_vsigmoid_ukernel__neonfma_rr1_p5_nr1recps1fma_x12)
DECLARE_F32_VSIGMOID_UKERNEL_FUNCTION(xnn_f32_vsigmoid_ukernel__neonfma_rr1_p5_nr1recps1fma_x16)
DECLARE_F32_VSIGMOID_UKERNEL_FUNCTION(xnn_f32_vsigmoid_ukernel__neonfma_rr1_p5_nr1recps1fma_x20)
DECLARE_F32_VSIGMOID_UKERNEL_FUNCTION(xnn_f32_vsigmoid_ukernel__neonfma_rr1_p5_nr1recps1fma_x24)

DECLARE_F32_VSIGMOID_UKERNEL_FUNCTION(xnn_f32_vsigmoid_ukernel__neonfma_rr1_p5_nr2recps_x4)
DECLARE_F32_VSIGMOID_UKERNEL_FUNCTION(xnn_f32_vsigmoid_ukernel__neonfma_rr1_p5_nr2recps_x8)
DECLARE_F32_VSIGMOID_UKERNEL_FUNCTION(xnn_f32_vsigmoid_ukernel__neonfma_rr1_p5_nr2recps_x12)
DECLARE_F32_VSIGMOID_UKERNEL_FUNCTION(xnn_f32_vsigmoid_ukernel__neonfma_rr1_p5_nr2recps_x16)
DECLARE_F32_VSIGMOID_UKERNEL_FUNCTION(xnn_f32_vsigmoid_ukernel__neonfma_rr1_p5_nr2recps_x20)
DECLARE_F32_VSIGMOID_UKERNEL_FUNCTION(xnn_f32_vsigmoid_ukernel__neonfma_rr1_p5_nr2recps_x24)

DECLARE_F32_VSIGMOID_UKERNEL_FUNCTION(xnn_f32_vsigmoid_ukernel__neon_rr2_p5_nr2recps_x4)
DECLARE_F32_VSIGMOID_UKERNEL_FUNCTION(xnn_f32_vsigmoid_ukernel__neon_rr2_p5_nr2recps_x8)
DECLARE_F32_VSIGMOID_UKERNEL_FUNCTION(xnn_f32_vsigmoid_ukernel__neon_rr2_p5_nr2recps_x12)
DECLARE_F32_VSIGMOID_UKERNEL_FUNCTION(xnn_f32_vsigmoid_ukernel__neon_rr2_p5_nr2recps_x16)
DECLARE_F32_VSIGMOID_UKERNEL_FUNCTION(xnn_f32_vsigmoid_ukernel__neon_rr2_p5_nr2recps_x20)
DECLARE_F32_VSIGMOID_UKERNEL_FUNCTION(xnn_f32_vsigmoid_ukernel__neon_rr2_p5_nr2recps_x24)

DECLARE_F32_VSIGMOID_UKERNEL_FUNCTION(xnn_f32_vsigmoid_ukernel__aarch64_neonfma_rr1_lut64_p2_div_x4)
DECLARE_F32_VSIGMOID_UKERNEL_FUNCTION(xnn_f32_vsigmoid_ukernel__aarch64_neonfma_rr1_lut64_p2_div_x8)
DECLARE_F32_VSIGMOID_UKERNEL_FUNCTION(xnn_f32_vsigmoid_ukernel__aarch64_neonfma_rr1_lut64_p2_div_x12)
DECLARE_F32_VSIGMOID_UKERNEL_FUNCTION(xnn_f32_vsigmoid_ukernel__aarch64_neonfma_rr1_lut64_p2_div_x16)
DECLARE_F32_VSIGMOID_UKERNEL_FUNCTION(xnn_f32_vsigmoid_ukernel__aarch64_neonfma_rr1_lut64_p2_div_x20)
DECLARE_F32_VSIGMOID_UKERNEL_FUNCTION(xnn_f32_vsigmoid_ukernel__aarch64_neonfma_rr1_lut64_p2_div_x24)

DECLARE_F32_VSIGMOID_UKERNEL_FUNCTION(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut64_p2_nr2fma_x4)
DECLARE_F32_VSIGMOID_UKERNEL_FUNCTION(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut64_p2_nr2fma_x8)
DECLARE_F32_VSIGMOID_UKERNEL_FUNCTION(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut64_p2_nr2fma_x12)
DECLARE_F32_VSIGMOID_UKERNEL_FUNCTION(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut64_p2_nr2fma_x16)
DECLARE_F32_VSIGMOID_UKERNEL_FUNCTION(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut64_p2_nr2fma_x20)
DECLARE_F32_VSIGMOID_UKERNEL_FUNCTION(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut64_p2_nr2fma_x24)

DECLARE_F32_VSIGMOID_UKERNEL_FUNCTION(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut64_p2_nr1recps1fma_x4)
DECLARE_F32_VSIGMOID_UKERNEL_FUNCTION(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut64_p2_nr1recps1fma_x8)
DECLARE_F32_VSIGMOID_UKERNEL_FUNCTION(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut64_p2_nr1recps1fma_x12)
DECLARE_F32_VSIGMOID_UKERNEL_FUNCTION(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut64_p2_nr1recps1fma_x16)
DECLARE_F32_VSIGMOID_UKERNEL_FUNCTION(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut64_p2_nr1recps1fma_x20)
DECLARE_F32_VSIGMOID_UKERNEL_FUNCTION(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut64_p2_nr1recps1fma_x24)

DECLARE_F32_VSIGMOID_UKERNEL_FUNCTION(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut64_p2_nr2recps_x4)
DECLARE_F32_VSIGMOID_UKERNEL_FUNCTION(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut64_p2_nr2recps_x8)
DECLARE_F32_VSIGMOID_UKERNEL_FUNCTION(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut64_p2_nr2recps_x12)
DECLARE_F32_VSIGMOID_UKERNEL_FUNCTION(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut64_p2_nr2recps_x16)
DECLARE_F32_VSIGMOID_UKERNEL_FUNCTION(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut64_p2_nr2recps_x20)
DECLARE_F32_VSIGMOID_UKERNEL_FUNCTION(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut64_p2_nr2recps_x24)

DECLARE_F32_VSIGMOID_UKERNEL_FUNCTION(xnn_f32_vsigmoid_ukernel__neon_rr2_lut64_p2_nr2recps_x4)
DECLARE_F32_VSIGMOID_UKERNEL_FUNCTION(xnn_f32_vsigmoid_ukernel__neon_rr2_lut64_p2_nr2recps_x8)
DECLARE_F32_VSIGMOID_UKERNEL_FUNCTION(xnn_f32_vsigmoid_ukernel__neon_rr2_lut64_p2_nr2recps_x12)
DECLARE_F32_VSIGMOID_UKERNEL_FUNCTION(xnn_f32_vsigmoid_ukernel__neon_rr2_lut64_p2_nr2recps_x16)
DECLARE_F32_VSIGMOID_UKERNEL_FUNCTION(xnn_f32_vsigmoid_ukernel__neon_rr2_lut64_p2_nr2recps_x20)
DECLARE_F32_VSIGMOID_UKERNEL_FUNCTION(xnn_f32_vsigmoid_ukernel__neon_rr2_lut64_p2_nr2recps_x24)

DECLARE_F32_VSIGMOID_UKERNEL_FUNCTION(xnn_f32_vsigmoid_ukernel__aarch64_neonfma_rr1_lut2048_p1_div_x4)
DECLARE_F32_VSIGMOID_UKERNEL_FUNCTION(xnn_f32_vsigmoid_ukernel__aarch64_neonfma_rr1_lut2048_p1_div_x8)
DECLARE_F32_VSIGMOID_UKERNEL_FUNCTION(xnn_f32_vsigmoid_ukernel__aarch64_neonfma_rr1_lut2048_p1_div_x12)
DECLARE_F32_VSIGMOID_UKERNEL_FUNCTION(xnn_f32_vsigmoid_ukernel__aarch64_neonfma_rr1_lut2048_p1_div_x16)
DECLARE_F32_VSIGMOID_UKERNEL_FUNCTION(xnn_f32_vsigmoid_ukernel__aarch64_neonfma_rr1_lut2048_p1_div_x20)
DECLARE_F32_VSIGMOID_UKERNEL_FUNCTION(xnn_f32_vsigmoid_ukernel__aarch64_neonfma_rr1_lut2048_p1_div_x24)

DECLARE_F32_VSIGMOID_UKERNEL_FUNCTION(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut2048_p1_nr2fma_x4)
DECLARE_F32_VSIGMOID_UKERNEL_FUNCTION(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut2048_p1_nr2fma_x8)
DECLARE_F32_VSIGMOID_UKERNEL_FUNCTION(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut2048_p1_nr2fma_x12)
DECLARE_F32_VSIGMOID_UKERNEL_FUNCTION(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut2048_p1_nr2fma_x16)
DECLARE_F32_VSIGMOID_UKERNEL_FUNCTION(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut2048_p1_nr2fma_x20)
DECLARE_F32_VSIGMOID_UKERNEL_FUNCTION(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut2048_p1_nr2fma_x24)

DECLARE_F32_VSIGMOID_UKERNEL_FUNCTION(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut2048_p1_nr1recps1fma_x4)
DECLARE_F32_VSIGMOID_UKERNEL_FUNCTION(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut2048_p1_nr1recps1fma_x8)
DECLARE_F32_VSIGMOID_UKERNEL_FUNCTION(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut2048_p1_nr1recps1fma_x12)
DECLARE_F32_VSIGMOID_UKERNEL_FUNCTION(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut2048_p1_nr1recps1fma_x16)
DECLARE_F32_VSIGMOID_UKERNEL_FUNCTION(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut2048_p1_nr1recps1fma_x20)
DECLARE_F32_VSIGMOID_UKERNEL_FUNCTION(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut2048_p1_nr1recps1fma_x24)

DECLARE_F32_VSIGMOID_UKERNEL_FUNCTION(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut2048_p1_nr2recps_x4)
DECLARE_F32_VSIGMOID_UKERNEL_FUNCTION(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut2048_p1_nr2recps_x8)
DECLARE_F32_VSIGMOID_UKERNEL_FUNCTION(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut2048_p1_nr2recps_x12)
DECLARE_F32_VSIGMOID_UKERNEL_FUNCTION(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut2048_p1_nr2recps_x16)
DECLARE_F32_VSIGMOID_UKERNEL_FUNCTION(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut2048_p1_nr2recps_x20)
DECLARE_F32_VSIGMOID_UKERNEL_FUNCTION(xnn_f32_vsigmoid_ukernel__neonfma_rr1_lut2048_p1_nr2recps_x24)

DECLARE_F32_VSIGMOID_UKERNEL_FUNCTION(xnn_f32_vsigmoid_ukernel__neon_rr2_lut2048_p1_nr2recps_x4)
DECLARE_F32_VSIGMOID_UKERNEL_FUNCTION(xnn_f32_vsigmoid_ukernel__neon_rr2_lut2048_p1_nr2recps_x8)
DECLARE_F32_VSIGMOID_UKERNEL_FUNCTION(xnn_f32_vsigmoid_ukernel__neon_rr2_lut2048_p1_nr2recps_x12)
DECLARE_F32_VSIGMOID_UKERNEL_FUNCTION(xnn_f32_vsigmoid_ukernel__neon_rr2_lut2048_p1_nr2recps_x16)
DECLARE_F32_VSIGMOID_UKERNEL_FUNCTION(xnn_f32_vsigmoid_ukernel__neon_rr2_lut2048_p1_nr2recps_x20)
DECLARE_F32_VSIGMOID_UKERNEL_FUNCTION(xnn_f32_vsigmoid_ukernel__neon_rr2_lut2048_p1_nr2recps_x24)

DECLARE_F32_VSIGMOID_UKERNEL_FUNCTION(xnn_f32_vsigmoid_ukernel__sse2_rr2_p5_div_x4)
DECLARE_F32_VSIGMOID_UKERNEL_FUNCTION(xnn_f32_vsigmoid_ukernel__sse2_rr2_p5_div_x8)
DECLARE_F32_VSIGMOID_UKERNEL_FUNCTION(xnn_f32_vsigmoid_ukernel__sse2_rr2_p5_div_x12)
DECLARE_F32_VSIGMOID_UKERNEL_FUNCTION(xnn_f32_vsigmoid_ukernel__sse2_rr2_p5_div_x16)
DECLARE_F32_VSIGMOID_UKERNEL_FUNCTION(xnn_f32_vsigmoid_ukernel__sse2_rr2_p5_div_x20)
DECLARE_F32_VSIGMOID_UKERNEL_FUNCTION(xnn_f32_vsigmoid_ukernel__sse2_rr2_p5_div_x24)

DECLARE_F32_VSIGMOID_UKERNEL_FUNCTION(xnn_f32_vsigmoid_ukernel__sse41_rr2_p5_div_x4)
DECLARE_F32_VSIGMOID_UKERNEL_FUNCTION(xnn_f32_vsigmoid_ukernel__sse41_rr2_p5_div_x8)
DECLARE_F32_VSIGMOID_UKERNEL_FUNCTION(xnn_f32_vsigmoid_ukernel__sse41_rr2_p5_div_x12)
DECLARE_F32_VSIGMOID_UKERNEL_FUNCTION(xnn_f32_vsigmoid_ukernel__sse41_rr2_p5_div_x16)
DECLARE_F32_VSIGMOID_UKERNEL_FUNCTION(xnn_f32_vsigmoid_ukernel__sse41_rr2_p5_div_x20)
DECLARE_F32_VSIGMOID_UKERNEL_FUNCTION(xnn_f32_vsigmoid_ukernel__sse41_rr2_p5_div_x24)

DECLARE_F32_VSIGMOID_UKERNEL_FUNCTION(xnn_f32_vsigmoid_ukernel__sse2_rr2_lut64_p2_div_x4)
DECLARE_F32_VSIGMOID_UKERNEL_FUNCTION(xnn_f32_vsigmoid_ukernel__sse2_rr2_lut64_p2_div_x8)
DECLARE_F32_VSIGMOID_UKERNEL_FUNCTION(xnn_f32_vsigmoid_ukernel__sse2_rr2_lut64_p2_div_x12)
DECLARE_F32_VSIGMOID_UKERNEL_FUNCTION(xnn_f32_vsigmoid_ukernel__sse2_rr2_lut64_p2_div_x16)
DECLARE_F32_VSIGMOID_UKERNEL_FUNCTION(xnn_f32_vsigmoid_ukernel__sse2_rr2_lut64_p2_div_x20)
DECLARE_F32_VSIGMOID_UKERNEL_FUNCTION(xnn_f32_vsigmoid_ukernel__sse2_rr2_lut64_p2_div_x24)

DECLARE_F32_VSIGMOID_UKERNEL_FUNCTION(xnn_f32_vsigmoid_ukernel__sse41_rr2_lut64_p2_div_x4)
DECLARE_F32_VSIGMOID_UKERNEL_FUNCTION(xnn_f32_vsigmoid_ukernel__sse41_rr2_lut64_p2_div_x8)
DECLARE_F32_VSIGMOID_UKERNEL_FUNCTION(xnn_f32_vsigmoid_ukernel__sse41_rr2_lut64_p2_div_x12)
DECLARE_F32_VSIGMOID_UKERNEL_FUNCTION(xnn_f32_vsigmoid_ukernel__sse41_rr2_lut64_p2_div_x16)
DECLARE_F32_VSIGMOID_UKERNEL_FUNCTION(xnn_f32_vsigmoid_ukernel__sse41_rr2_lut64_p2_div_x20)
DECLARE_F32_VSIGMOID_UKERNEL_FUNCTION(xnn_f32_vsigmoid_ukernel__sse41_rr2_lut64_p2_div_x24)

DECLARE_F32_VSIGMOID_UKERNEL_FUNCTION(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_div_x8)
DECLARE_F32_VSIGMOID_UKERNEL_FUNCTION(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_div_x16)
DECLARE_F32_VSIGMOID_UKERNEL_FUNCTION(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_div_x24)
DECLARE_F32_VSIGMOID_UKERNEL_FUNCTION(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_div_x32)
DECLARE_F32_VSIGMOID_UKERNEL_FUNCTION(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_div_x40)
DECLARE_F32_VSIGMOID_UKERNEL_FUNCTION(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_div_x48)
DECLARE_F32_VSIGMOID_UKERNEL_FUNCTION(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_div_x56)
DECLARE_F32_VSIGMOID_UKERNEL_FUNCTION(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_div_x64)
DECLARE_F32_VSIGMOID_UKERNEL_FUNCTION(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_div_x72)
DECLARE_F32_VSIGMOID_UKERNEL_FUNCTION(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_div_x80)

DECLARE_F32_VSIGMOID_UKERNEL_FUNCTION(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_nr2_x8)
DECLARE_F32_VSIGMOID_UKERNEL_FUNCTION(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_nr2_x16)
DECLARE_F32_VSIGMOID_UKERNEL_FUNCTION(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_nr2_x24)
DECLARE_F32_VSIGMOID_UKERNEL_FUNCTION(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_nr2_x32)
DECLARE_F32_VSIGMOID_UKERNEL_FUNCTION(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_nr2_x40)
DECLARE_F32_VSIGMOID_UKERNEL_FUNCTION(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_nr2_x48)
DECLARE_F32_VSIGMOID_UKERNEL_FUNCTION(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_nr2_x56)
DECLARE_F32_VSIGMOID_UKERNEL_FUNCTION(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_nr2_x64)
DECLARE_F32_VSIGMOID_UKERNEL_FUNCTION(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_nr2_x72)
DECLARE_F32_VSIGMOID_UKERNEL_FUNCTION(xnn_f32_vsigmoid_ukernel__avx_rr2_p5_nr2_x80)

DECLARE_F32_VSIGMOID_UKERNEL_FUNCTION(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_div_x8)
DECLARE_F32_VSIGMOID_UKERNEL_FUNCTION(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_div_x16)
DECLARE_F32_VSIGMOID_UKERNEL_FUNCTION(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_div_x24)
DECLARE_F32_VSIGMOID_UKERNEL_FUNCTION(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_div_x32)
DECLARE_F32_VSIGMOID_UKERNEL_FUNCTION(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_div_x40)
DECLARE_F32_VSIGMOID_UKERNEL_FUNCTION(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_div_x48)
DECLARE_F32_VSIGMOID_UKERNEL_FUNCTION(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_div_x56)
DECLARE_F32_VSIGMOID_UKERNEL_FUNCTION(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_div_x64)
DECLARE_F32_VSIGMOID_UKERNEL_FUNCTION(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_div_x72)
DECLARE_F32_VSIGMOID_UKERNEL_FUNCTION(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_div_x80)

DECLARE_F32_VSIGMOID_UKERNEL_FUNCTION(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr1fma_x8)
DECLARE_F32_VSIGMOID_UKERNEL_FUNCTION(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr1fma_x16)
DECLARE_F32_VSIGMOID_UKERNEL_FUNCTION(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr1fma_x24)
DECLARE_F32_VSIGMOID_UKERNEL_FUNCTION(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr1fma_x32)
DECLARE_F32_VSIGMOID_UKERNEL_FUNCTION(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr1fma_x40)
DECLARE_F32_VSIGMOID_UKERNEL_FUNCTION(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr1fma_x48)
DECLARE_F32_VSIGMOID_UKERNEL_FUNCTION(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr1fma_x56)
DECLARE_F32_VSIGMOID_UKERNEL_FUNCTION(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr1fma_x64)
DECLARE_F32_VSIGMOID_UKERNEL_FUNCTION(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr1fma_x72)
DECLARE_F32_VSIGMOID_UKERNEL_FUNCTION(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr1fma_x80)

DECLARE_F32_VSIGMOID_UKERNEL_FUNCTION(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr2fma_x8)
DECLARE_F32_VSIGMOID_UKERNEL_FUNCTION(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr2fma_x16)
DECLARE_F32_VSIGMOID_UKERNEL_FUNCTION(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr2fma_x24)
DECLARE_F32_VSIGMOID_UKERNEL_FUNCTION(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr2fma_x32)
DECLARE_F32_VSIGMOID_UKERNEL_FUNCTION(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr2fma_x40)
DECLARE_F32_VSIGMOID_UKERNEL_FUNCTION(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr2fma_x48)
DECLARE_F32_VSIGMOID_UKERNEL_FUNCTION(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr2fma_x56)
DECLARE_F32_VSIGMOID_UKERNEL_FUNCTION(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr2fma_x64)
DECLARE_F32_VSIGMOID_UKERNEL_FUNCTION(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr2fma_x72)
DECLARE_F32_VSIGMOID_UKERNEL_FUNCTION(xnn_f32_vsigmoid_ukernel__avx2_rr1_p5_nr2fma_x80)

DECLARE_F32_VSIGMOID_UKERNEL_FUNCTION(xnn_f32_vsigmoid_ukernel__avx512f_rr1_p5_scalef_div_x16)
DECLARE_F32_VSIGMOID_UKERNEL_FUNCTION(xnn_f32_vsigmoid_ukernel__avx512f_rr1_p5_scalef_div_x32)
DECLARE_F32_VSIGMOID_UKERNEL_FUNCTION(xnn_f32_vsigmoid_ukernel__avx512f_rr1_p5_scalef_div_x48)
DECLARE_F32_VSIGMOID_UKERNEL_FUNCTION(xnn_f32_vsigmoid_ukernel__avx512f_rr1_p5_scalef_div_x64)
DECLARE_F32_VSIGMOID_UKERNEL_FUNCTION(xnn_f32_vsigmoid_ukernel__avx512f_rr1_p5_scalef_div_x80)
DECLARE_F32_VSIGMOID_UKERNEL_FUNCTION(xnn_f32_vsigmoid_ukernel__avx512f_rr1_p5_scalef_div_x96)
DECLARE_F32_VSIGMOID_UKERNEL_FUNCTION(xnn_f32_vsigmoid_ukernel__avx512f_rr1_p5_scalef_div_x112)
DECLARE_F32_VSIGMOID_UKERNEL_FUNCTION(xnn_f32_vsigmoid_ukernel__avx512f_rr1_p5_scalef_div_x128)

DECLARE_F32_VSIGMOID_UKERNEL_FUNCTION(xnn_f32_vsigmoid_ukernel__avx512f_rr1_p5_scalef_nr1fma_x16)
DECLARE_F32_VSIGMOID_UKERNEL_FUNCTION(xnn_f32_vsigmoid_ukernel__avx512f_rr1_p5_scalef_nr1fma_x32)
DECLARE_F32_VSIGMOID_UKERNEL_FUNCTION(xnn_f32_vsigmoid_ukernel__avx512f_rr1_p5_scalef_nr1fma_x48)
DECLARE_F32_VSIGMOID_UKERNEL_FUNCTION(xnn_f32_vsigmoid_ukernel__avx512f_rr1_p5_scalef_nr1fma_x64)
DECLARE_F32_VSIGMOID_UKERNEL_FUNCTION(xnn_f32_vsigmoid_ukernel__avx512f_rr1_p5_scalef_nr1fma_x80)
DECLARE_F32_VSIGMOID_UKERNEL_FUNCTION(xnn_f32_vsigmoid_ukernel__avx512f_rr1_p5_scalef_nr1fma_x96)
DECLARE_F32_VSIGMOID_UKERNEL_FUNCTION(xnn_f32_vsigmoid_ukernel__avx512f_rr1_p5_scalef_nr1fma_x112)
DECLARE_F32_VSIGMOID_UKERNEL_FUNCTION(xnn_f32_vsigmoid_ukernel__avx512f_rr1_p5_scalef_nr1fma_x128)

DECLARE_F32_VSIGMOID_UKERNEL_FUNCTION(xnn_f32_vsigmoid_ukernel__avx512f_rr1_lut16_p3_perm_scalef_div_x16)
DECLARE_F32_VSIGMOID_UKERNEL_FUNCTION(xnn_f32_vsigmoid_ukernel__avx512f_rr1_lut16_p3_perm_scalef_div_x32)
DECLARE_F32_VSIGMOID_UKERNEL_FUNCTION(xnn_f32_vsigmoid_ukernel__avx512f_rr1_lut16_p3_perm_scalef_div_x48)
DECLARE_F32_VSIGMOID_UKERNEL_FUNCTION(xnn_f32_vsigmoid_ukernel__avx512f_rr1_lut16_p3_perm_scalef_div_x64)
DECLARE_F32_VSIGMOID_UKERNEL_FUNCTION(xnn_f32_vsigmoid_ukernel__avx512f_rr1_lut16_p3_perm_scalef_div_x80)
DECLARE_F32_VSIGMOID_UKERNEL_FUNCTION(xnn_f32_vsigmoid_ukernel__avx512f_rr1_lut16_p3_perm_scalef_div_x96)
DECLARE_F32_VSIGMOID_UKERNEL_FUNCTION(xnn_f32_vsigmoid_ukernel__avx512f_rr1_lut16_p3_perm_scalef_div_x112)
DECLARE_F32_VSIGMOID_UKERNEL_FUNCTION(xnn_f32_vsigmoid_ukernel__avx512f_rr1_lut16_p3_perm_scalef_div_x128)

DECLARE_F32_VSIGMOID_UKERNEL_FUNCTION(xnn_f32_vsigmoid_ukernel__avx512f_rr1_lut16_p3_perm_scalef_nr1fma_x16)
DECLARE_F32_VSIGMOID_UKERNEL_FUNCTION(xnn_f32_vsigmoid_ukernel__avx512f_rr1_lut16_p3_perm_scalef_nr1fma_x32)
DECLARE_F32_VSIGMOID_UKERNEL_FUNCTION(xnn_f32_vsigmoid_ukernel__avx512f_rr1_lut16_p3_perm_scalef_nr1fma_x48)
DECLARE_F32_VSIGMOID_UKERNEL_FUNCTION(xnn_f32_vsigmoid_ukernel__avx512f_rr1_lut16_p3_perm_scalef_nr1fma_x64)
DECLARE_F32_VSIGMOID_UKERNEL_FUNCTION(xnn_f32_vsigmoid_ukernel__avx512f_rr1_lut16_p3_perm_scalef_nr1fma_x80)
DECLARE_F32_VSIGMOID_UKERNEL_FUNCTION(xnn_f32_vsigmoid_ukernel__avx512f_rr1_lut16_p3_perm_scalef_nr1fma_x96)
DECLARE_F32_VSIGMOID_UKERNEL_FUNCTION(xnn_f32_vsigmoid_ukernel__avx512f_rr1_lut16_p3_perm_scalef_nr1fma_x112)
DECLARE_F32_VSIGMOID_UKERNEL_FUNCTION(xnn_f32_vsigmoid_ukernel__avx512f_rr1_lut16_p3_perm_scalef_nr1fma_x128)

DECLARE_F32_VSIGMOID_UKERNEL_FUNCTION(xnn_f32_vsigmoid_ukernel__avx512f_rr2_lut32_p2_perm2_scalef_div_x16)
DECLARE_F32_VSIGMOID_UKERNEL_FUNCTION(xnn_f32_vsigmoid_ukernel__avx512f_rr2_lut32_p2_perm2_scalef_div_x32)
DECLARE_F32_VSIGMOID_UKERNEL_FUNCTION(xnn_f32_vsigmoid_ukernel__avx512f_rr2_lut32_p2_perm2_scalef_div_x48)
DECLARE_F32_VSIGMOID_UKERNEL_FUNCTION(xnn_f32_vsigmoid_ukernel__avx512f_rr2_lut32_p2_perm2_scalef_div_x64)
DECLARE_F32_VSIGMOID_UKERNEL_FUNCTION(xnn_f32_vsigmoid_ukernel__avx512f_rr2_lut32_p2_perm2_scalef_div_x80)
DECLARE_F32_VSIGMOID_UKERNEL_FUNCTION(xnn_f32_vsigmoid_ukernel__avx512f_rr2_lut32_p2_perm2_scalef_div_x96)
DECLARE_F32_VSIGMOID_UKERNEL_FUNCTION(xnn_f32_vsigmoid_ukernel__avx512f_rr2_lut32_p2_perm2_scalef_div_x112)
DECLARE_F32_VSIGMOID_UKERNEL_FUNCTION(xnn_f32_vsigmoid_ukernel__avx512f_rr2_lut32_p2_perm2_scalef_div_x128)

DECLARE_F32_VSIGMOID_UKERNEL_FUNCTION(xnn_f32_vsigmoid_ukernel__avx512f_rr2_lut32_p2_perm2_scalef_nr1fma_x16)
DECLARE_F32_VSIGMOID_UKERNEL_FUNCTION(xnn_f32_vsigmoid_ukernel__avx512f_rr2_lut32_p2_perm2_scalef_nr1fma_x32)
DECLARE_F32_VSIGMOID_UKERNEL_FUNCTION(xnn_f32_vsigmoid_ukernel__avx512f_rr2_lut32_p2_perm2_scalef_nr1fma_x48)
DECLARE_F32_VSIGMOID_UKERNEL_FUNCTION(xnn_f32_vsigmoid_ukernel__avx512f_rr2_lut32_p2_perm2_scalef_nr1fma_x64)
DECLARE_F32_VSIGMOID_UKERNEL_FUNCTION(xnn_f32_vsigmoid_ukernel__avx512f_rr2_lut32_p2_perm2_scalef_nr1fma_x80)
DECLARE_F32_VSIGMOID_UKERNEL_FUNCTION(xnn_f32_vsigmoid_ukernel__avx512f_rr2_lut32_p2_perm2_scalef_nr1fma_x96)
DECLARE_F32_VSIGMOID_UKERNEL_FUNCTION(xnn_f32_vsigmoid_ukernel__avx512f_rr2_lut32_p2_perm2_scalef_nr1fma_x112)
DECLARE_F32_VSIGMOID_UKERNEL_FUNCTION(xnn_f32_vsigmoid_ukernel__avx512f_rr2_lut32_p2_perm2_scalef_nr1fma_x128)

DECLARE_F32_VSIGMOID_UKERNEL_FUNCTION(xnn_f32_vsigmoid_ukernel__wasmsimd_rr2_p5_div_x4)
DECLARE_F32_VSIGMOID_UKERNEL_FUNCTION(xnn_f32_vsigmoid_ukernel__wasmsimd_rr2_p5_div_x8)
DECLARE_F32_VSIGMOID_UKERNEL_FUNCTION(xnn_f32_vsigmoid_ukernel__wasmsimd_rr2_p5_div_x12)
DECLARE_F32_VSIGMOID_UKERNEL_FUNCTION(xnn_f32_vsigmoid_ukernel__wasmsimd_rr2_p5_div_x16)
DECLARE_F32_VSIGMOID_UKERNEL_FUNCTION(xnn_f32_vsigmoid_ukernel__wasmsimd_rr2_p5_div_x20)
DECLARE_F32_VSIGMOID_UKERNEL_FUNCTION(xnn_f32_vsigmoid_ukernel__wasmsimd_rr2_p5_div_x24)

DECLARE_F32_VSIGMOID_UKERNEL_FUNCTION(xnn_f32_vsigmoid_ukernel__wasmsimd_rr2_lut64_p2_div_x4)
DECLARE_F32_VSIGMOID_UKERNEL_FUNCTION(xnn_f32_vsigmoid_ukernel__wasmsimd_rr2_lut64_p2_div_x8)
DECLARE_F32_VSIGMOID_UKERNEL_FUNCTION(xnn_f32_vsigmoid_ukernel__wasmsimd_rr2_lut64_p2_div_x12)
DECLARE_F32_VSIGMOID_UKERNEL_FUNCTION(xnn_f32_vsigmoid_ukernel__wasmsimd_rr2_lut64_p2_div_x16)
DECLARE_F32_VSIGMOID_UKERNEL_FUNCTION(xnn_f32_vsigmoid_ukernel__wasmsimd_rr2_lut64_p2_div_x20)
DECLARE_F32_VSIGMOID_UKERNEL_FUNCTION(xnn_f32_vsigmoid_ukernel__wasmsimd_rr2_lut64_p2_div_x24)

DECLARE_F32_VSIGMOID_UKERNEL_FUNCTION(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_rr2_p5_div_x4)
DECLARE_F32_VSIGMOID_UKERNEL_FUNCTION(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_rr2_p5_div_x8)
DECLARE_F32_VSIGMOID_UKERNEL_FUNCTION(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_rr2_p5_div_x12)
DECLARE_F32_VSIGMOID_UKERNEL_FUNCTION(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_rr2_p5_div_x16)
DECLARE_F32_VSIGMOID_UKERNEL_FUNCTION(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_rr2_p5_div_x20)
DECLARE_F32_VSIGMOID_UKERNEL_FUNCTION(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_rr2_p5_div_x24)

DECLARE_F32_VSIGMOID_UKERNEL_FUNCTION(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_rr2_lut64_p2_div_x4)
DECLARE_F32_VSIGMOID_UKERNEL_FUNCTION(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_rr2_lut64_p2_div_x8)
DECLARE_F32_VSIGMOID_UKERNEL_FUNCTION(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_rr2_lut64_p2_div_x12)
DECLARE_F32_VSIGMOID_UKERNEL_FUNCTION(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_rr2_lut64_p2_div_x16)
DECLARE_F32_VSIGMOID_UKERNEL_FUNCTION(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_rr2_lut64_p2_div_x20)
DECLARE_F32_VSIGMOID_UKERNEL_FUNCTION(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_rr2_lut64_p2_div_x24)

DECLARE_F32_VSIGMOID_UKERNEL_FUNCTION(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_fma_rr2_p5_div_x4)
DECLARE_F32_VSIGMOID_UKERNEL_FUNCTION(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_fma_rr2_p5_div_x8)
DECLARE_F32_VSIGMOID_UKERNEL_FUNCTION(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_fma_rr2_p5_div_x12)
DECLARE_F32_VSIGMOID_UKERNEL_FUNCTION(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_fma_rr2_p5_div_x16)
DECLARE_F32_VSIGMOID_UKERNEL_FUNCTION(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_fma_rr2_p5_div_x20)
DECLARE_F32_VSIGMOID_UKERNEL_FUNCTION(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_fma_rr2_p5_div_x24)

DECLARE_F32_VSIGMOID_UKERNEL_FUNCTION(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_fma_rr2_lut64_p2_div_x4)
DECLARE_F32_VSIGMOID_UKERNEL_FUNCTION(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_fma_rr2_lut64_p2_div_x8)
DECLARE_F32_VSIGMOID_UKERNEL_FUNCTION(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_fma_rr2_lut64_p2_div_x12)
DECLARE_F32_VSIGMOID_UKERNEL_FUNCTION(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_fma_rr2_lut64_p2_div_x16)
DECLARE_F32_VSIGMOID_UKERNEL_FUNCTION(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_fma_rr2_lut64_p2_div_x20)
DECLARE_F32_VSIGMOID_UKERNEL_FUNCTION(xnn_f32_vsigmoid_ukernel__wasmrelaxedsimd_fma_rr2_lut64_p2_div_x24)

DECLARE_F32_VSIGMOID_UKERNEL_FUNCTION(xnn_f32_vsigmoid_ukernel__scalar_rr2_lut2048_p1_div_x1)
DECLARE_F32_VSIGMOID_UKERNEL_FUNCTION(xnn_f32_vsigmoid_ukernel__scalar_rr2_lut2048_p1_div_x2)
DECLARE_F32_VSIGMOID_UKERNEL_FUNCTION(xnn_f32_vsigmoid_ukernel__scalar_rr2_lut2048_p1_div_x4)

DECLARE_F32_VSIGMOID_UKERNEL_FUNCTION(xnn_f32_vsigmoid_ukernel__scalar_rr2_lut64_p2_div_x1)
DECLARE_F32_VSIGMOID_UKERNEL_FUNCTION(xnn_f32_vsigmoid_ukernel__scalar_rr2_lut64_p2_div_x2)
DECLARE_F32_VSIGMOID_UKERNEL_FUNCTION(xnn_f32_vsigmoid_ukernel__scalar_rr2_lut64_p2_div_x4)

DECLARE_F32_VSIGMOID_UKERNEL_FUNCTION(xnn_f32_vsigmoid_ukernel__scalar_rr2_p5_div_x1)
DECLARE_F32_VSIGMOID_UKERNEL_FUNCTION(xnn_f32_vsigmoid_ukernel__scalar_rr2_p5_div_x2)
DECLARE_F32_VSIGMOID_UKERNEL_FUNCTION(xnn_f32_vsigmoid_ukernel__scalar_rr2_p5_div_x4)


#define DECLARE_F32_VSQR_UKERNEL_FUNCTION(fn_name) \
  XNN_INTERNAL void fn_name(                       \
      size_t n,                                    \
      const float* x,                              \
      float* y,                                    \
      const union xnn_f32_default_params* params);

DECLARE_F32_VSQR_UKERNEL_FUNCTION(xnn_f32_vsqr_ukernel__neon_x4)
DECLARE_F32_VSQR_UKERNEL_FUNCTION(xnn_f32_vsqr_ukernel__neon_x8)

DECLARE_F32_VSQR_UKERNEL_FUNCTION(xnn_f32_vsqr_ukernel__sse_x4)
DECLARE_F32_VSQR_UKERNEL_FUNCTION(xnn_f32_vsqr_ukernel__sse_x8)

DECLARE_F32_VSQR_UKERNEL_FUNCTION(xnn_f32_vsqr_ukernel__avx_x8)
DECLARE_F32_VSQR_UKERNEL_FUNCTION(xnn_f32_vsqr_ukernel__avx_x16)

DECLARE_F32_VSQR_UKERNEL_FUNCTION(xnn_f32_vsqr_ukernel__avx512f_x16)
DECLARE_F32_VSQR_UKERNEL_FUNCTION(xnn_f32_vsqr_ukernel__avx512f_x32)

DECLARE_F32_VSQR_UKERNEL_FUNCTION(xnn_f32_vsqr_ukernel__wasmsimd_x4)
DECLARE_F32_VSQR_UKERNEL_FUNCTION(xnn_f32_vsqr_ukernel__wasmsimd_x8)

DECLARE_F32_VSQR_UKERNEL_FUNCTION(xnn_f32_vsqr_ukernel__scalar_x1)
DECLARE_F32_VSQR_UKERNEL_FUNCTION(xnn_f32_vsqr_ukernel__scalar_x2)
DECLARE_F32_VSQR_UKERNEL_FUNCTION(xnn_f32_vsqr_ukernel__scalar_x4)


#define DECLARE_S8_VCLAMP_UKERNEL_FUNCTION(fn_name) \
  XNN_INTERNAL void fn_name(                        \
      size_t n,                                     \
      const int8_t* x,                              \
      int8_t* y,                                    \
      const union xnn_s8_minmax_params* params);

DECLARE_S8_VCLAMP_UKERNEL_FUNCTION(xnn_s8_vclamp_ukernel__neon_x64)
DECLARE_S8_VCLAMP_UKERNEL_FUNCTION(xnn_s8_vclamp_ukernel__scalar_x4)
DECLARE_S8_VCLAMP_UKERNEL_FUNCTION(xnn_s8_vclamp_ukernel__sse2_x64)
DECLARE_S8_VCLAMP_UKERNEL_FUNCTION(xnn_s8_vclamp_ukernel__sse41_x64)
DECLARE_S8_VCLAMP_UKERNEL_FUNCTION(xnn_s8_vclamp_ukernel__wasmsimd_x64)


#define DECLARE_U8_VCLAMP_UKERNEL_FUNCTION(fn_name) \
  XNN_INTERNAL void fn_name(                        \
      size_t n,                                     \
      const uint8_t* x,                             \
      uint8_t* y,                                   \
      const union xnn_u8_minmax_params* params);

DECLARE_U8_VCLAMP_UKERNEL_FUNCTION(xnn_u8_vclamp_ukernel__neon_x64)
DECLARE_U8_VCLAMP_UKERNEL_FUNCTION(xnn_u8_vclamp_ukernel__scalar_x4)
DECLARE_U8_VCLAMP_UKERNEL_FUNCTION(xnn_u8_vclamp_ukernel__sse2_x64)
DECLARE_U8_VCLAMP_UKERNEL_FUNCTION(xnn_u8_vclamp_ukernel__wasmsimd_x64)


#define DECLARE_U64_U32_VSQRTSHIFT_UKERNEL_FUNCTION(fn_name) \
  XNN_INTERNAL void fn_name(                                 \
      size_t n,                                              \
      const uint64_t* x,                                     \
      uint32_t* y,                                           \
      uint32_t shift);

DECLARE_U64_U32_VSQRTSHIFT_UKERNEL_FUNCTION(xnn_u64_u32_vsqrtshift_ukernel__scalar_cvtu32_sqrt_cvtu32f64_x1)


#define DECLARE_XX_VUNARY_UKERNEL_FUNCTION(fn_name) \
  XNN_INTERNAL void fn_name(                        \
      size_t size,                                  \
      const void* input,                            \
      void* output,                                 \
      const void* params);

DECLARE_XX_VUNARY_UKERNEL_FUNCTION(xnn_xx_copy_ukernel__scalar_memcpy)

#ifdef __cplusplus
}  // extern "C"
#endif
