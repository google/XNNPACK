// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
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

#define DECLARE_F32_DWCONV_UNIPASS_UKERNEL_FUNCTION(fn_name) \
  XNN_INTERNAL void fn_name(                                 \
    size_t channels,                                         \
    size_t output_width,                                     \
    const float** input,                                     \
    const float* weights,                                    \
    float* output,                                           \
    size_t input_stride,                                     \
    size_t output_increment,                                 \
    const union xnn_f32_default_params* params);

#define DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(fn_name) \
  XNN_INTERNAL void fn_name(                                        \
    size_t channels,                                                \
    size_t output_width,                                            \
    const float** input,                                            \
    const float* weights,                                           \
    float* output,                                                  \
    size_t input_stride,                                            \
    size_t output_increment,                                        \
    const union xnn_f32_minmax_params* params);

DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up4x4__neon)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up4x4__neon_acc2)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up8x4__neon)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up8x4__neon_acc2)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up4x4__neonfma)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up4x4__neonfma_acc2)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up8x4__neonfma)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up8x4__neonfma_acc2)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up4x4__psimd)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up4x4__psimd_acc2)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up8x4__psimd)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up8x4__psimd_acc2)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up4x4__sse)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up4x4__sse_acc2)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up8x4__sse)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up8x4__sse_acc2)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up8x4__avx)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up8x4__avx_acc2)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up16x4__avx)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up16x4__avx_acc2)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up8x4__fma3)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up8x4__fma3_acc2)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up16x4__fma3)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up16x4__fma3_acc2)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up16x4__avx512f)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up16x4__avx512f_acc2)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up32x4__avx512f)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up32x4__avx512f_acc2)

DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up4x9__neon)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up4x9__neon_acc2)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up8x9__neon)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up8x9__neon_acc2)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up4x9__neonfma)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up4x9__neonfma_acc2)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up8x9__neonfma)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up8x9__neonfma_acc2)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up4x9__aarch64_neonfma)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up4x9__aarch64_neonfma_cortex_a55)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up4x9__psimd)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up4x9__psimd_acc2)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up8x9__psimd)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up8x9__psimd_acc2)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up4x9__sse)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up4x9__sse_acc2)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up8x9__sse)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up8x9__sse_acc2)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up8x9__avx)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up8x9__avx_acc2)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up16x9__avx)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up16x9__avx_acc2)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up8x9__fma3)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up8x9__fma3_acc2)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up16x9__fma3)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up16x9__fma3_acc2)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up16x9__avx512f)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up16x9__avx512f_acc2)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up32x9__avx512f)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up32x9__avx512f_acc2)

DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up4x25__neon)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up4x25__neon_acc2)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up8x25__neon)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up8x25__neon_acc2)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up4x25__neonfma)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up4x25__neonfma_acc2)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up8x25__neonfma)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up8x25__neonfma_acc2)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up4x25__psimd)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up4x25__psimd_acc2)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up8x25__psimd)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up8x25__psimd_acc2)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up4x25__sse)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up4x25__sse_acc2)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up8x25__sse)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up8x25__sse_acc2)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up8x25__avx)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up8x25__avx_acc2)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up16x25__avx)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up16x25__avx_acc2)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up8x25__fma3)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up8x25__fma3_acc2)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up16x25__fma3)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up16x25__fma3_acc2)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up16x25__avx512f)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up16x25__avx512f_acc2)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up32x25__avx512f)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up32x25__avx512f_acc2)

DECLARE_F32_DWCONV_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_ukernel_up1x4__wasm)
DECLARE_F32_DWCONV_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_ukernel_up1x4__wasm_acc2)
DECLARE_F32_DWCONV_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_ukernel_up2x4__wasm)
DECLARE_F32_DWCONV_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_ukernel_up2x4__wasm_acc2)

DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up1x4__wasm)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up1x4__wasm_acc2)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up2x4__wasm)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up2x4__wasm_acc2)

DECLARE_F32_DWCONV_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_ukernel_up1x9__wasm)
DECLARE_F32_DWCONV_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_ukernel_up1x9__wasm_acc2)
DECLARE_F32_DWCONV_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_ukernel_up2x9__wasm)
DECLARE_F32_DWCONV_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_ukernel_up2x9__wasm_acc2)

DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up1x9__wasm)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up1x9__wasm_acc2)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up2x9__wasm)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up2x9__wasm_acc2)

DECLARE_F32_DWCONV_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_ukernel_up1x25__wasm)
DECLARE_F32_DWCONV_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_ukernel_up1x25__wasm_acc2)
DECLARE_F32_DWCONV_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_ukernel_up2x25__wasm)
DECLARE_F32_DWCONV_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_ukernel_up2x25__wasm_acc2)

DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up1x25__wasm)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up1x25__wasm_acc2)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up2x25__wasm)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up2x25__wasm_acc2)

DECLARE_F32_DWCONV_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_ukernel_up1x4__scalar)
DECLARE_F32_DWCONV_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_ukernel_up1x4__scalar_acc2)
DECLARE_F32_DWCONV_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_ukernel_up2x4__scalar)
DECLARE_F32_DWCONV_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_ukernel_up2x4__scalar_acc2)

DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up1x4__scalar)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up1x4__scalar_acc2)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up2x4__scalar)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up2x4__scalar_acc2)

DECLARE_F32_DWCONV_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_ukernel_up1x9__scalar)
DECLARE_F32_DWCONV_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_ukernel_up1x9__scalar_acc2)
DECLARE_F32_DWCONV_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_ukernel_up2x9__scalar)
DECLARE_F32_DWCONV_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_ukernel_up2x9__scalar_acc2)

DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up1x9__scalar)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up1x9__scalar_acc2)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up2x9__scalar)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up2x9__scalar_acc2)

DECLARE_F32_DWCONV_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_ukernel_up1x25__scalar)
DECLARE_F32_DWCONV_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_ukernel_up1x25__scalar_acc2)
DECLARE_F32_DWCONV_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_ukernel_up2x25__scalar)
DECLARE_F32_DWCONV_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_ukernel_up2x25__scalar_acc2)

DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up1x25__scalar)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up1x25__scalar_acc2)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up2x25__scalar)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up2x25__scalar_acc2)


#define DECLARE_Q8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(fn_name) \
  XNN_INTERNAL void fn_name(                                       \
    size_t channels,                                               \
    size_t output_width,                                           \
    const uint8_t** input,                                         \
    const void* weights,                                           \
    uint8_t* output,                                               \
    size_t input_stride,                                           \
    size_t output_increment,                                       \
    const union xnn_q8_gemm_params* params);

DECLARE_Q8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_q8_dwconv_minmax_ukernel_up1x9__scalar)
DECLARE_Q8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_q8_dwconv_minmax_ukernel_up8x9__aarch32_neon)
DECLARE_Q8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_q8_dwconv_minmax_ukernel_up8x9__neon)
DECLARE_Q8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_q8_dwconv_minmax_ukernel_up8x9__sse2)


#define DECLARE_F32_DWCONV_SPCHW_UKERNEL_FUNCTION(fn_name) \
  XNN_INTERNAL void fn_name(                               \
    size_t m,                                              \
    size_t n,                                              \
    const float* input,                                    \
    const float* weights,                                  \
    float* output,                                         \
    size_t input_tuple_stride,                             \
    size_t output_tuple_stride,                            \
    size_t input_height_stride,                            \
    size_t output_height_stride,                           \
    const union xnn_f32_spchw_params* params);

DECLARE_F32_DWCONV_SPCHW_UKERNEL_FUNCTION(xnn_f32_dwconv_spchw_ukernel_3x3p1__scalar)
DECLARE_F32_DWCONV_SPCHW_UKERNEL_FUNCTION(xnn_f32_dwconv_spchw_ukernel_5x5p2__scalar)
DECLARE_F32_DWCONV_SPCHW_UKERNEL_FUNCTION(xnn_f32_dwconv_spchw_ukernel_3x3s2p1__scalar)
DECLARE_F32_DWCONV_SPCHW_UKERNEL_FUNCTION(xnn_f32_dwconv_spchw_ukernel_5x5s2p2__scalar)
DECLARE_F32_DWCONV_SPCHW_UKERNEL_FUNCTION(xnn_f32_dwconv_spchw_ukernel_3x3p1__neonfma)
DECLARE_F32_DWCONV_SPCHW_UKERNEL_FUNCTION(xnn_f32_dwconv_spchw_ukernel_5x5p2__neonfma)
DECLARE_F32_DWCONV_SPCHW_UKERNEL_FUNCTION(xnn_f32_dwconv_spchw_ukernel_3x3p1__sse)
DECLARE_F32_DWCONV_SPCHW_UKERNEL_FUNCTION(xnn_f32_dwconv_spchw_ukernel_3x3s2p1__neonfma)
DECLARE_F32_DWCONV_SPCHW_UKERNEL_FUNCTION(xnn_f32_dwconv_spchw_ukernel_5x5s2p2__neonfma)
DECLARE_F32_DWCONV_SPCHW_UKERNEL_FUNCTION(xnn_f32_dwconv_spchw_ukernel_3x3s2p1__sse)


#ifdef __cplusplus
}  // extern "C"
#endif
