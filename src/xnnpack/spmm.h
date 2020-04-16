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


#define DECLARE_F32_SPMM_MINMAX_UKERNEL_FUNCTION(fn_name) \
  XNN_INTERNAL void fn_name(                              \
    uint32_t m,                                           \
    uint32_t n,                                           \
    const float* a,                                       \
    const float* w,                                       \
    const int32_t* dmap,                                  \
    const uint32_t* nmap,                                 \
    float* c,                                             \
    const union xnn_f32_minmax_params* params);

DECLARE_F32_SPMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_spmm_minmax_ukernel_12x1__neonfma)
DECLARE_F32_SPMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_spmm_minmax_ukernel_12x2__neonfma)
DECLARE_F32_SPMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_spmm_minmax_ukernel_12x4__neonfma)
DECLARE_F32_SPMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_spmm_minmax_ukernel_16x1__neonfma)
DECLARE_F32_SPMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_spmm_minmax_ukernel_16x2__neonfma)
DECLARE_F32_SPMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_spmm_minmax_ukernel_16x4__neonfma)
DECLARE_F32_SPMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_spmm_minmax_ukernel_16x1__neonfma_pipelined)
DECLARE_F32_SPMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_spmm_minmax_ukernel_16x1__neonfma_unroll2)
DECLARE_F32_SPMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_spmm_minmax_ukernel_1x1__scalar)
DECLARE_F32_SPMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_spmm_minmax_ukernel_1x1__scalar_pipelined)
DECLARE_F32_SPMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_spmm_minmax_ukernel_2x1__scalar)
DECLARE_F32_SPMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_spmm_minmax_ukernel_2x1__scalar_pipelined)
DECLARE_F32_SPMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_spmm_minmax_ukernel_4x1__neonfma)
DECLARE_F32_SPMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_spmm_minmax_ukernel_4x2__neonfma)
DECLARE_F32_SPMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_spmm_minmax_ukernel_4x4__neonfma)
DECLARE_F32_SPMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_spmm_minmax_ukernel_4x1__neonfma_pipelined)
DECLARE_F32_SPMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_spmm_minmax_ukernel_4x1__neonfma_unroll2)
DECLARE_F32_SPMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_spmm_minmax_ukernel_4x1__scalar)
DECLARE_F32_SPMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_spmm_minmax_ukernel_4x1__scalar_pipelined)
DECLARE_F32_SPMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_spmm_minmax_ukernel_4x1__sse)
DECLARE_F32_SPMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_spmm_minmax_ukernel_8x1__neonfma)
DECLARE_F32_SPMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_spmm_minmax_ukernel_8x2__neonfma)
DECLARE_F32_SPMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_spmm_minmax_ukernel_8x4__neonfma)
DECLARE_F32_SPMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_spmm_minmax_ukernel_8x1__neonfma_pipelined)
DECLARE_F32_SPMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_spmm_minmax_ukernel_8x1__neonfma_unroll2)
DECLARE_F32_SPMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_spmm_minmax_ukernel_8x1__scalar)
DECLARE_F32_SPMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_spmm_minmax_ukernel_8x2__scalar)
DECLARE_F32_SPMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_spmm_minmax_ukernel_8x4__scalar)
DECLARE_F32_SPMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_spmm_minmax_ukernel_8x1__scalar_pipelined)
DECLARE_F32_SPMM_MINMAX_UKERNEL_FUNCTION(xnn_f32_spmm_minmax_ukernel_8x1__sse)

#define DECLARE_F16_SPMM_MINMAX_UKERNEL_FUNCTION(fn_name) \
  XNN_INTERNAL void fn_name(                              \
    uint32_t m,                                           \
    uint32_t n,                                           \
    const void* a,                                        \
    const void* w,                                        \
    const int32_t* dmap,                                  \
    const uint32_t* nmap,                                 \
    void* c,                                              \
    const struct xnn_f16_scaleminmax_params* params);

DECLARE_F16_SPMM_MINMAX_UKERNEL_FUNCTION(xnn_f16_spmm_minmax_ukernel_8x1__neonfp16arith)
DECLARE_F16_SPMM_MINMAX_UKERNEL_FUNCTION(xnn_f16_spmm_minmax_ukernel_8x1__neonfp16arith_unroll2)
DECLARE_F16_SPMM_MINMAX_UKERNEL_FUNCTION(xnn_f16_spmm_minmax_ukernel_16x1__neonfp16arith)
DECLARE_F16_SPMM_MINMAX_UKERNEL_FUNCTION(xnn_f16_spmm_minmax_ukernel_16x1__neonfp16arith_unroll2)
DECLARE_F16_SPMM_MINMAX_UKERNEL_FUNCTION(xnn_f16_spmm_minmax_ukernel_24x1__neonfp16arith)
DECLARE_F16_SPMM_MINMAX_UKERNEL_FUNCTION(xnn_f16_spmm_minmax_ukernel_24x1__neonfp16arith_unroll2)
DECLARE_F16_SPMM_MINMAX_UKERNEL_FUNCTION(xnn_f16_spmm_minmax_ukernel_32x1__neonfp16arith)
DECLARE_F16_SPMM_MINMAX_UKERNEL_FUNCTION(xnn_f16_spmm_minmax_ukernel_32x1__neonfp16arith_unroll2)

#ifdef __cplusplus
}  // extern "C"
#endif
