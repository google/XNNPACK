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

#include <xnnpack/common.h>
#include <xnnpack/microparams.h>

#ifdef __cplusplus
extern "C" {
#endif


#define DECLARE_F32_DWCONV_UNIPASS_UKERNEL_FUNCTION(fn_name) \
  XNN_INTERNAL void fn_name(                                 \
    size_t channels,                                         \
    size_t output_width,                                     \
    size_t kernel_elements,                                  \
    const float** input,                                     \
    const float* weights,                                    \
    float* output,                                           \
    size_t input_stride,                                     \
    size_t output_increment,                                 \
    size_t input_offset,                                     \
    const float* zero,                                       \
    const union xnn_f32_default_params* params);

#define DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(fn_name) \
  XNN_INTERNAL void fn_name(                                        \
    size_t channels,                                                \
    size_t output_width,                                            \
    size_t kernel_elements,                                         \
    const float** input,                                            \
    const float* weights,                                           \
    float* output,                                                  \
    size_t input_stride,                                            \
    size_t output_increment,                                        \
    size_t input_offset,                                            \
    const float* zero,                                              \
    const union xnn_f32_minmax_params* params);

DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up4x3__neon)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up4x3__neon_acc2)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up4x3__neonfma)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up4x3__neonfma_acc2)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up4x3__sse)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up4x3__sse_acc2)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up8x3__avx)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up8x3__avx_acc2)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up8x3__fma3)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up8x3__fma3_acc2)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up8x3__neon)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up8x3__neon_acc2)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up8x3__neonfma)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up8x3__neonfma_acc2)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up8x3__sse)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up8x3__sse_acc2)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up16x3__avx)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up16x3__avx_acc2)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up16x3__avx512f)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up16x3__avx512f_acc2)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up16x3__fma3)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up16x3__fma3_acc2)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up16x3__neon)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up16x3__neon_acc2)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up16x3__neonfma)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up16x3__neonfma_acc2)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up32x3__avx512f)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up32x3__avx512f_acc2)

DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up4x4__neon)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up4x4__neon_acc2)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up4x4__neonfma)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up4x4__neonfma_acc2)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up4x4__sse)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up4x4__sse_acc2)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up8x4__avx)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up8x4__avx_acc2)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up8x4__fma3)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up8x4__fma3_acc2)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up8x4__neon)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up8x4__neon_acc2)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up8x4__neonfma)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up8x4__neonfma_acc2)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up8x4__sse)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up8x4__sse_acc2)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up16x4__avx)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up16x4__avx_acc2)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up16x4__avx512f)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up16x4__avx512f_acc2)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up16x4__fma3)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up16x4__fma3_acc2)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up16x4__neon)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up16x4__neon_acc2)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up16x4__neonfma)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up16x4__neonfma_acc2)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up32x4__avx512f)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up32x4__avx512f_acc2)

DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up4x9__aarch64_neonfma)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up4x9__aarch64_neonfma_cortex_a55)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up4x9__neon)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up4x9__neon_acc2)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up4x9__neonfma)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up4x9__neonfma_acc2)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up4x9__sse)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up4x9__sse_acc2)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up8x9__avx)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up8x9__avx_acc2)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up8x9__fma3)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up8x9__fma3_acc2)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up8x9__neon)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up8x9__neon_acc2)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up8x9__neonfma)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up8x9__neonfma_acc2)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up8x9__sse)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up8x9__sse_acc2)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up16x9__avx)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up16x9__avx_acc2)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up16x9__avx512f)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up16x9__avx512f_acc2)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up16x9__fma3)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up16x9__fma3_acc2)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up16x9__neon)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up16x9__neon_acc2)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up16x9__neonfma)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up16x9__neonfma_acc2)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up32x9__avx512f)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up32x9__avx512f_acc2)

DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up4x25__neon)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up4x25__neon_acc2)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up4x25__neonfma)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up4x25__neonfma_acc2)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up4x25__sse)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up4x25__sse_acc2)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up8x25__avx)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up8x25__avx_acc2)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up8x25__fma3)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up8x25__fma3_acc2)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up8x25__neon)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up8x25__neon_acc2)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up8x25__neonfma)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up8x25__neonfma_acc2)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up8x25__sse)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up8x25__sse_acc2)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up16x25__avx)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up16x25__avx_acc2)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up16x25__avx512f)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up16x25__avx512f_acc2)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up16x25__fma3)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up16x25__fma3_acc2)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up16x25__neon)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up16x25__neon_acc2)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up16x25__neonfma)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up16x25__neonfma_acc2)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up32x25__avx512f)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up32x25__avx512f_acc2)

DECLARE_F32_DWCONV_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_ukernel_up4x3__wasmrelaxedsimd_fma)
DECLARE_F32_DWCONV_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_ukernel_up4x3__wasmsimd)
DECLARE_F32_DWCONV_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_ukernel_up8x3__wasmrelaxedsimd_fma)
DECLARE_F32_DWCONV_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_ukernel_up8x3__wasmsimd)

DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up4x3__wasmrelaxedsimd)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up4x3__wasmrelaxedsimd_acc2)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up4x3__wasmrelaxedsimd_fma)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up4x3__wasmrelaxedsimd_fma_acc2)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up4x3__wasmsimd_arm)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up4x3__wasmsimd_arm_acc2)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up4x3__wasmsimd_x86)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up4x3__wasmsimd_x86_acc2)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up8x3__wasmrelaxedsimd)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up8x3__wasmrelaxedsimd_acc2)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up8x3__wasmrelaxedsimd_fma)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up8x3__wasmrelaxedsimd_fma_acc2)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up8x3__wasmsimd_arm)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up8x3__wasmsimd_arm_acc2)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up8x3__wasmsimd_x86)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up8x3__wasmsimd_x86_acc2)

DECLARE_F32_DWCONV_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_ukernel_up4x4__wasmrelaxedsimd_fma)
DECLARE_F32_DWCONV_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_ukernel_up4x4__wasmsimd)
DECLARE_F32_DWCONV_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_ukernel_up8x4__wasmrelaxedsimd_fma)
DECLARE_F32_DWCONV_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_ukernel_up8x4__wasmsimd)

DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up4x4__wasmrelaxedsimd)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up4x4__wasmrelaxedsimd_acc2)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up4x4__wasmrelaxedsimd_fma)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up4x4__wasmrelaxedsimd_fma_acc2)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up4x4__wasmsimd_arm)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up4x4__wasmsimd_arm_acc2)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up4x4__wasmsimd_x86)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up4x4__wasmsimd_x86_acc2)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up8x4__wasmrelaxedsimd)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up8x4__wasmrelaxedsimd_acc2)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up8x4__wasmrelaxedsimd_fma)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up8x4__wasmrelaxedsimd_fma_acc2)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up8x4__wasmsimd_arm)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up8x4__wasmsimd_arm_acc2)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up8x4__wasmsimd_x86)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up8x4__wasmsimd_x86_acc2)

DECLARE_F32_DWCONV_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_ukernel_up4x9__wasmrelaxedsimd_fma)
DECLARE_F32_DWCONV_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_ukernel_up4x9__wasmsimd)
DECLARE_F32_DWCONV_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_ukernel_up4x9__wasmsimd_acc2)
DECLARE_F32_DWCONV_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_ukernel_up8x9__wasmrelaxedsimd_fma)
DECLARE_F32_DWCONV_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_ukernel_up8x9__wasmsimd)
DECLARE_F32_DWCONV_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_ukernel_up8x9__wasmsimd_acc2)

DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up4x9__wasmrelaxedsimd)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up4x9__wasmrelaxedsimd_acc2)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up4x9__wasmrelaxedsimd_fma)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up4x9__wasmrelaxedsimd_fma_acc2)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up4x9__wasmsimd_arm)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up4x9__wasmsimd_arm_acc2)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up4x9__wasmsimd_x86)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up4x9__wasmsimd_x86_acc2)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up8x9__wasmrelaxedsimd)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up8x9__wasmrelaxedsimd_acc2)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up8x9__wasmrelaxedsimd_fma)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up8x9__wasmrelaxedsimd_fma_acc2)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up8x9__wasmsimd_arm)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up8x9__wasmsimd_arm_acc2)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up8x9__wasmsimd_x86)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up8x9__wasmsimd_x86_acc2)

DECLARE_F32_DWCONV_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_ukernel_up4x25__wasmrelaxedsimd_fma)
DECLARE_F32_DWCONV_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_ukernel_up4x25__wasmsimd)
DECLARE_F32_DWCONV_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_ukernel_up8x25__wasmrelaxedsimd_fma)
DECLARE_F32_DWCONV_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_ukernel_up8x25__wasmsimd)

DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up4x25__wasmrelaxedsimd)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up4x25__wasmrelaxedsimd_acc2)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up4x25__wasmrelaxedsimd_fma)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up4x25__wasmrelaxedsimd_fma_acc2)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up4x25__wasmsimd_arm)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up4x25__wasmsimd_arm_acc2)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up4x25__wasmsimd_x86)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up4x25__wasmsimd_x86_acc2)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up8x25__wasmrelaxedsimd)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up8x25__wasmrelaxedsimd_acc2)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up8x25__wasmrelaxedsimd_fma)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up8x25__wasmrelaxedsimd_fma_acc2)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up8x25__wasmsimd_arm)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up8x25__wasmsimd_arm_acc2)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up8x25__wasmsimd_x86)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up8x25__wasmsimd_x86_acc2)

DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up1x3__wasm)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up1x3__wasm_acc2)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up2x3__wasm)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up2x3__wasm_acc2)

DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up1x4__wasm)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up1x4__wasm_acc2)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up2x4__wasm)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up2x4__wasm_acc2)

DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up1x9__wasm)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up1x9__wasm_acc2)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up2x9__wasm)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up2x9__wasm_acc2)

DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up1x25__wasm)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up1x25__wasm_acc2)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up2x25__wasm)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up2x25__wasm_acc2)

DECLARE_F32_DWCONV_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_ukernel_up1x3__scalar)
DECLARE_F32_DWCONV_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_ukernel_up1x3__scalar_acc2)
DECLARE_F32_DWCONV_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_ukernel_up2x3__scalar)
DECLARE_F32_DWCONV_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_ukernel_up2x3__scalar_acc2)

DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up1x3__scalar)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up1x3__scalar_acc2)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up2x3__scalar)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up2x3__scalar_acc2)

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


#define DECLARE_F16_DWCONV_UNIPASS_UKERNEL_FUNCTION(fn_name) \
  XNN_INTERNAL void fn_name(                                 \
    size_t channels,                                         \
    size_t output_width,                                     \
    size_t kernel_elements,                                  \
    const void** input,                                      \
    const void* weights,                                     \
    void* output,                                            \
    size_t input_stride,                                     \
    size_t output_increment,                                 \
    size_t input_offset,                                     \
    const void* zero,                                        \
    const union xnn_f16_default_params* params);

#define DECLARE_F16_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(fn_name) \
  XNN_INTERNAL void fn_name(                                        \
    size_t channels,                                                \
    size_t output_width,                                            \
    size_t kernel_elements,                                         \
    const void** input,                                             \
    const void* weights,                                            \
    void* output,                                                   \
    size_t input_stride,                                            \
    size_t output_increment,                                        \
    size_t input_offset,                                            \
    const void* zero,                                               \
    const union xnn_f16_minmax_params* params);

DECLARE_F16_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f16_dwconv_minmax_ukernel_up8x3__neonfp16arith)
DECLARE_F16_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f16_dwconv_minmax_ukernel_up8x3__neonfp16arith_acc2)
DECLARE_F16_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f16_dwconv_minmax_ukernel_up8x4__neonfp16arith)
DECLARE_F16_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f16_dwconv_minmax_ukernel_up8x4__neonfp16arith_acc2)
DECLARE_F16_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f16_dwconv_minmax_ukernel_up8x9__neonfp16arith)
DECLARE_F16_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f16_dwconv_minmax_ukernel_up8x9__neonfp16arith_acc2)
DECLARE_F16_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f16_dwconv_minmax_ukernel_up8x25__neonfp16arith)
DECLARE_F16_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f16_dwconv_minmax_ukernel_up8x25__neonfp16arith_acc2)
DECLARE_F16_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f16_dwconv_minmax_ukernel_up16x3__neonfp16arith)
DECLARE_F16_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f16_dwconv_minmax_ukernel_up16x3__neonfp16arith_acc2)
DECLARE_F16_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f16_dwconv_minmax_ukernel_up16x4__neonfp16arith)
DECLARE_F16_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f16_dwconv_minmax_ukernel_up16x4__neonfp16arith_acc2)
DECLARE_F16_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f16_dwconv_minmax_ukernel_up16x9__neonfp16arith)
DECLARE_F16_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f16_dwconv_minmax_ukernel_up16x9__neonfp16arith_acc2)
DECLARE_F16_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f16_dwconv_minmax_ukernel_up16x25__neonfp16arith)
DECLARE_F16_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f16_dwconv_minmax_ukernel_up16x25__neonfp16arith_acc2)
DECLARE_F16_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f16_dwconv_minmax_ukernel_up32x3__neonfp16arith)
DECLARE_F16_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f16_dwconv_minmax_ukernel_up32x3__neonfp16arith_acc2)
DECLARE_F16_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f16_dwconv_minmax_ukernel_up32x4__neonfp16arith)
DECLARE_F16_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f16_dwconv_minmax_ukernel_up32x4__neonfp16arith_acc2)
DECLARE_F16_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f16_dwconv_minmax_ukernel_up32x9__neonfp16arith)
DECLARE_F16_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f16_dwconv_minmax_ukernel_up32x9__neonfp16arith_acc2)
DECLARE_F16_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f16_dwconv_minmax_ukernel_up32x25__neonfp16arith)
DECLARE_F16_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f16_dwconv_minmax_ukernel_up32x25__neonfp16arith_acc2)

DECLARE_F16_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f16_dwconv_minmax_ukernel_up8x3__fma3)
DECLARE_F16_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f16_dwconv_minmax_ukernel_up8x3__fma3_acc2)
DECLARE_F16_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f16_dwconv_minmax_ukernel_up8x4__fma3)
DECLARE_F16_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f16_dwconv_minmax_ukernel_up8x4__fma3_acc2)
DECLARE_F16_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f16_dwconv_minmax_ukernel_up8x9__fma3)
DECLARE_F16_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f16_dwconv_minmax_ukernel_up8x9__fma3_acc2)
DECLARE_F16_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f16_dwconv_minmax_ukernel_up8x25__fma3)
DECLARE_F16_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f16_dwconv_minmax_ukernel_up8x25__fma3_acc2)
DECLARE_F16_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f16_dwconv_minmax_ukernel_up16x3__fma3)
DECLARE_F16_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f16_dwconv_minmax_ukernel_up16x3__fma3_acc2)
DECLARE_F16_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f16_dwconv_minmax_ukernel_up16x4__fma3)
DECLARE_F16_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f16_dwconv_minmax_ukernel_up16x4__fma3_acc2)
DECLARE_F16_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f16_dwconv_minmax_ukernel_up16x9__fma3)
DECLARE_F16_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f16_dwconv_minmax_ukernel_up16x9__fma3_acc2)
DECLARE_F16_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f16_dwconv_minmax_ukernel_up16x25__fma3)
DECLARE_F16_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f16_dwconv_minmax_ukernel_up16x25__fma3_acc2)
DECLARE_F16_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f16_dwconv_minmax_ukernel_up32x3__fma3)
DECLARE_F16_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f16_dwconv_minmax_ukernel_up32x3__fma3_acc2)
DECLARE_F16_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f16_dwconv_minmax_ukernel_up32x4__fma3)
DECLARE_F16_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f16_dwconv_minmax_ukernel_up32x4__fma3_acc2)
DECLARE_F16_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f16_dwconv_minmax_ukernel_up32x9__fma3)
DECLARE_F16_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f16_dwconv_minmax_ukernel_up32x9__fma3_acc2)
DECLARE_F16_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f16_dwconv_minmax_ukernel_up32x25__fma3)
DECLARE_F16_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f16_dwconv_minmax_ukernel_up32x25__fma3_acc2)


#define DECLARE_QU8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(fn_name) \
  XNN_INTERNAL void fn_name(                                        \
    size_t channels,                                                \
    size_t output_width,                                            \
    size_t kernel_elements,                                         \
    const uint8_t** input,                                          \
    const void* weights,                                            \
    uint8_t* output,                                                \
    size_t input_stride,                                            \
    size_t output_increment,                                        \
    size_t input_offset,                                            \
    const uint8_t* zero,                                            \
    const union xnn_qu8_conv_minmax_params* params);

DECLARE_QU8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qu8_dwconv_minmax_fp32_ukernel_up8x9__neon_mul16)
DECLARE_QU8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qu8_dwconv_minmax_fp32_ukernel_up16x9__neon_mul16)
DECLARE_QU8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qu8_dwconv_minmax_fp32_ukernel_up24x9__neon_mul16)
DECLARE_QU8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qu8_dwconv_minmax_fp32_ukernel_up32x9__neon_mul16)

DECLARE_QU8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qu8_dwconv_minmax_fp32_ukernel_up8x9__neonv8_mul16)
DECLARE_QU8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qu8_dwconv_minmax_fp32_ukernel_up16x9__neonv8_mul16)
DECLARE_QU8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qu8_dwconv_minmax_fp32_ukernel_up24x9__neonv8_mul16)
DECLARE_QU8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qu8_dwconv_minmax_fp32_ukernel_up32x9__neonv8_mul16)

DECLARE_QU8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qu8_dwconv_minmax_rndnu_ukernel_up8x9__neon_mul16)
DECLARE_QU8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qu8_dwconv_minmax_rndnu_ukernel_up16x9__neon_mul16)
DECLARE_QU8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qu8_dwconv_minmax_rndnu_ukernel_up24x9__neon_mul16)
DECLARE_QU8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qu8_dwconv_minmax_rndnu_ukernel_up32x9__neon_mul16)

DECLARE_QU8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qu8_dwconv_minmax_rndnu_ukernel_up8x9__neon_mul8)
DECLARE_QU8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qu8_dwconv_minmax_rndnu_ukernel_up16x9__neon_mul8)
DECLARE_QU8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qu8_dwconv_minmax_rndnu_ukernel_up24x9__neon_mul8)
DECLARE_QU8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qu8_dwconv_minmax_rndnu_ukernel_up32x9__neon_mul8)

DECLARE_QU8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qu8_dwconv_minmax_fp32_ukernel_up8x9__sse2_mul16)
DECLARE_QU8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qu8_dwconv_minmax_fp32_ukernel_up16x9__sse2_mul16)

DECLARE_QU8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qu8_dwconv_minmax_fp32_ukernel_up8x9__sse41_mul16)
DECLARE_QU8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qu8_dwconv_minmax_fp32_ukernel_up16x9__sse41_mul16)

DECLARE_QU8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qu8_dwconv_minmax_fp32_ukernel_up8x9__avx_mul16)
DECLARE_QU8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qu8_dwconv_minmax_fp32_ukernel_up16x9__avx_mul16)

DECLARE_QU8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qu8_dwconv_minmax_fp32_ukernel_up8x25__sse2_mul16)
DECLARE_QU8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qu8_dwconv_minmax_fp32_ukernel_up16x25__sse2_mul16)

DECLARE_QU8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qu8_dwconv_minmax_fp32_ukernel_up8x25__sse41_mul16)
DECLARE_QU8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qu8_dwconv_minmax_fp32_ukernel_up16x25__sse41_mul16)

DECLARE_QU8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qu8_dwconv_minmax_fp32_ukernel_up8x25__avx_mul16)
DECLARE_QU8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qu8_dwconv_minmax_fp32_ukernel_up16x25__avx_mul16)

DECLARE_QU8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qu8_dwconv_minmax_fp32_ukernel_up8x9__sse41_mul32)
DECLARE_QU8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qu8_dwconv_minmax_fp32_ukernel_up16x9__sse41_mul32)

DECLARE_QU8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qu8_dwconv_minmax_fp32_ukernel_up8x9__avx_mul32)
DECLARE_QU8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qu8_dwconv_minmax_fp32_ukernel_up16x9__avx_mul32)

DECLARE_QU8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qu8_dwconv_minmax_fp32_ukernel_up8x9__xop_mul32)
DECLARE_QU8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qu8_dwconv_minmax_fp32_ukernel_up16x9__xop_mul32)

DECLARE_QU8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qu8_dwconv_minmax_fp32_ukernel_up8x9__avx2_mul32)
DECLARE_QU8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qu8_dwconv_minmax_fp32_ukernel_up16x9__avx2_mul32)
DECLARE_QU8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qu8_dwconv_minmax_fp32_ukernel_up32x9__avx2_mul32)

DECLARE_QU8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qu8_dwconv_minmax_fp32_ukernel_up16x9__avx512skx_mul32)
DECLARE_QU8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qu8_dwconv_minmax_fp32_ukernel_up32x9__avx512skx_mul32)

DECLARE_QU8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qu8_dwconv_minmax_fp32_ukernel_up8x9__wasmsimd_mul16)
DECLARE_QU8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qu8_dwconv_minmax_fp32_ukernel_up16x9__wasmsimd_mul16)
DECLARE_QU8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qu8_dwconv_minmax_fp32_ukernel_up24x9__wasmsimd_mul16)

DECLARE_QU8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qu8_dwconv_minmax_fp32_ukernel_up1x9__wasm_fmagic)
DECLARE_QU8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qu8_dwconv_minmax_fp32_ukernel_up2x9__wasm_fmagic)
DECLARE_QU8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qu8_dwconv_minmax_fp32_ukernel_up4x9__wasm_fmagic)

DECLARE_QU8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qu8_dwconv_minmax_fp32_ukernel_up1x9__scalar_fmagic)
DECLARE_QU8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qu8_dwconv_minmax_fp32_ukernel_up2x9__scalar_fmagic)
DECLARE_QU8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qu8_dwconv_minmax_fp32_ukernel_up4x9__scalar_fmagic)

DECLARE_QU8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qu8_dwconv_minmax_fp32_ukernel_up1x9__scalar_imagic)
DECLARE_QU8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qu8_dwconv_minmax_fp32_ukernel_up2x9__scalar_imagic)
DECLARE_QU8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qu8_dwconv_minmax_fp32_ukernel_up4x9__scalar_imagic)

DECLARE_QU8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qu8_dwconv_minmax_fp32_ukernel_up1x9__scalar_lrintf)
DECLARE_QU8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qu8_dwconv_minmax_fp32_ukernel_up2x9__scalar_lrintf)
DECLARE_QU8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qu8_dwconv_minmax_fp32_ukernel_up4x9__scalar_lrintf)

DECLARE_QU8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qu8_dwconv_minmax_fp32_ukernel_up8x25__neon_mul16)
DECLARE_QU8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qu8_dwconv_minmax_fp32_ukernel_up16x25__neon_mul16)
DECLARE_QU8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qu8_dwconv_minmax_fp32_ukernel_up24x25__neon_mul16)
DECLARE_QU8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qu8_dwconv_minmax_fp32_ukernel_up32x25__neon_mul16)

DECLARE_QU8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qu8_dwconv_minmax_fp32_ukernel_up8x25__neonv8_mul16)
DECLARE_QU8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qu8_dwconv_minmax_fp32_ukernel_up16x25__neonv8_mul16)
DECLARE_QU8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qu8_dwconv_minmax_fp32_ukernel_up24x25__neonv8_mul16)
DECLARE_QU8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qu8_dwconv_minmax_fp32_ukernel_up32x25__neonv8_mul16)

DECLARE_QU8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qu8_dwconv_minmax_rndnu_ukernel_up8x25__neon_mul16)
DECLARE_QU8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qu8_dwconv_minmax_rndnu_ukernel_up16x25__neon_mul16)
DECLARE_QU8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qu8_dwconv_minmax_rndnu_ukernel_up24x25__neon_mul16)
DECLARE_QU8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qu8_dwconv_minmax_rndnu_ukernel_up32x25__neon_mul16)

DECLARE_QU8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qu8_dwconv_minmax_rndnu_ukernel_up8x25__neon_mul8)
DECLARE_QU8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qu8_dwconv_minmax_rndnu_ukernel_up16x25__neon_mul8)
DECLARE_QU8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qu8_dwconv_minmax_rndnu_ukernel_up24x25__neon_mul8)
DECLARE_QU8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qu8_dwconv_minmax_rndnu_ukernel_up32x25__neon_mul8)

DECLARE_QU8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qu8_dwconv_minmax_fp32_ukernel_up8x25__sse41_mul32)
DECLARE_QU8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qu8_dwconv_minmax_fp32_ukernel_up16x25__sse41_mul32)

DECLARE_QU8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qu8_dwconv_minmax_fp32_ukernel_up8x25__avx_mul32)
DECLARE_QU8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qu8_dwconv_minmax_fp32_ukernel_up16x25__avx_mul32)

DECLARE_QU8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qu8_dwconv_minmax_fp32_ukernel_up8x25__xop_mul32)
DECLARE_QU8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qu8_dwconv_minmax_fp32_ukernel_up16x25__xop_mul32)

DECLARE_QU8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qu8_dwconv_minmax_fp32_ukernel_up8x25__avx2_mul32)
DECLARE_QU8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qu8_dwconv_minmax_fp32_ukernel_up16x25__avx2_mul32)
DECLARE_QU8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qu8_dwconv_minmax_fp32_ukernel_up32x25__avx2_mul32)

DECLARE_QU8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qu8_dwconv_minmax_fp32_ukernel_up16x25__avx512skx_mul32)
DECLARE_QU8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qu8_dwconv_minmax_fp32_ukernel_up32x25__avx512skx_mul32)

DECLARE_QU8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qu8_dwconv_minmax_fp32_ukernel_up8x25__wasmsimd_mul16)
DECLARE_QU8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qu8_dwconv_minmax_fp32_ukernel_up16x25__wasmsimd_mul16)
DECLARE_QU8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qu8_dwconv_minmax_fp32_ukernel_up24x25__wasmsimd_mul16)

DECLARE_QU8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qu8_dwconv_minmax_fp32_ukernel_up1x25__wasm_fmagic)
DECLARE_QU8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qu8_dwconv_minmax_fp32_ukernel_up2x25__wasm_fmagic)
DECLARE_QU8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qu8_dwconv_minmax_fp32_ukernel_up4x25__wasm_fmagic)

DECLARE_QU8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qu8_dwconv_minmax_fp32_ukernel_up1x25__scalar_fmagic)
DECLARE_QU8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qu8_dwconv_minmax_fp32_ukernel_up2x25__scalar_fmagic)
DECLARE_QU8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qu8_dwconv_minmax_fp32_ukernel_up4x25__scalar_fmagic)

DECLARE_QU8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qu8_dwconv_minmax_fp32_ukernel_up1x25__scalar_imagic)
DECLARE_QU8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qu8_dwconv_minmax_fp32_ukernel_up2x25__scalar_imagic)
DECLARE_QU8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qu8_dwconv_minmax_fp32_ukernel_up4x25__scalar_imagic)

DECLARE_QU8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qu8_dwconv_minmax_fp32_ukernel_up1x25__scalar_lrintf)
DECLARE_QU8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qu8_dwconv_minmax_fp32_ukernel_up2x25__scalar_lrintf)
DECLARE_QU8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qu8_dwconv_minmax_fp32_ukernel_up4x25__scalar_lrintf)


#define DECLARE_QS8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(fn_name) \
  XNN_INTERNAL void fn_name(                                        \
    size_t channels,                                                \
    size_t output_width,                                            \
    size_t kernel_elements,                                         \
    const int8_t** input,                                           \
    const void* weights,                                            \
    int8_t* output,                                                 \
    size_t input_stride,                                            \
    size_t output_increment,                                        \
    size_t input_offset,                                            \
    const int8_t* zero,                                             \
    const union xnn_qs8_conv_minmax_params* params);

DECLARE_QS8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qs8_dwconv_minmax_rndnu_ukernel_up8x9__neon_mul8_ld64)
DECLARE_QS8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qs8_dwconv_minmax_rndnu_ukernel_up16x9__neon_mul8_ld64)
DECLARE_QS8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qs8_dwconv_minmax_rndnu_ukernel_up16x9__neon_mul8_ld128)

DECLARE_QS8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qs8_dwconv_minmax_rndnu_ukernel_up8x9__neon_mla8_ld64)
DECLARE_QS8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qs8_dwconv_minmax_rndnu_ukernel_up16x9__neon_mla8_ld64)
DECLARE_QS8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qs8_dwconv_minmax_rndnu_ukernel_up16x9__neon_mla8_ld128)

DECLARE_QS8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qs8_dwconv_minmax_rndnu_ukernel_up8x25__neon_mul8_ld64)
DECLARE_QS8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qs8_dwconv_minmax_rndnu_ukernel_up16x25__neon_mul8_ld64)
DECLARE_QS8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qs8_dwconv_minmax_rndnu_ukernel_up16x25__neon_mul8_ld128)

DECLARE_QS8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qs8_dwconv_minmax_rndnu_ukernel_up8x25__neon_mla8_ld64)
DECLARE_QS8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qs8_dwconv_minmax_rndnu_ukernel_up16x25__neon_mla8_ld64)
DECLARE_QS8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qs8_dwconv_minmax_rndnu_ukernel_up16x25__neon_mla8_ld128)

DECLARE_QS8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qs8_dwconv_minmax_fp32_ukernel_up8x9__neon_mul16)
DECLARE_QS8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qs8_dwconv_minmax_fp32_ukernel_up16x9__neon_mul16)
DECLARE_QS8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qs8_dwconv_minmax_fp32_ukernel_up24x9__neon_mul16)
DECLARE_QS8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qs8_dwconv_minmax_fp32_ukernel_up32x9__neon_mul16)

DECLARE_QS8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qs8_dwconv_minmax_fp32_ukernel_up8x9__neonv8_mul16)
DECLARE_QS8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qs8_dwconv_minmax_fp32_ukernel_up16x9__neonv8_mul16)
DECLARE_QS8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qs8_dwconv_minmax_fp32_ukernel_up24x9__neonv8_mul16)
DECLARE_QS8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qs8_dwconv_minmax_fp32_ukernel_up32x9__neonv8_mul16)

DECLARE_QS8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qs8_dwconv_minmax_rndnu_ukernel_up8x9__neon_mul16)
DECLARE_QS8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qs8_dwconv_minmax_rndnu_ukernel_up16x9__neon_mul16)
DECLARE_QS8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qs8_dwconv_minmax_rndnu_ukernel_up24x9__neon_mul16)
DECLARE_QS8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qs8_dwconv_minmax_rndnu_ukernel_up32x9__neon_mul16)

DECLARE_QS8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qs8_dwconv_minmax_fp32_ukernel_up8x9__sse2_mul16)
DECLARE_QS8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qs8_dwconv_minmax_fp32_ukernel_up16x9__sse2_mul16)
DECLARE_QS8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qs8_dwconv_minmax_fp32_ukernel_up24x9__sse2_mul16)

DECLARE_QS8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qs8_dwconv_minmax_fp32_ukernel_up8x9__sse2_mul16_add16)
DECLARE_QS8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qs8_dwconv_minmax_fp32_ukernel_up16x9__sse2_mul16_add16)

DECLARE_QS8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qs8_dwconv_minmax_fp32_ukernel_up8x9__sse41_mul16)
DECLARE_QS8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qs8_dwconv_minmax_fp32_ukernel_up16x9__sse41_mul16)
DECLARE_QS8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qs8_dwconv_minmax_fp32_ukernel_up24x9__sse41_mul16)

DECLARE_QS8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qs8_dwconv_minmax_fp32_ukernel_up8x9__sse41_mul16_add16)
DECLARE_QS8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qs8_dwconv_minmax_fp32_ukernel_up16x9__sse41_mul16_add16)

DECLARE_QS8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qs8_dwconv_minmax_fp32_ukernel_up8x9__avx_mul16)
DECLARE_QS8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qs8_dwconv_minmax_fp32_ukernel_up16x9__avx_mul16)
DECLARE_QS8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qs8_dwconv_minmax_fp32_ukernel_up24x9__avx_mul16)

DECLARE_QS8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qs8_dwconv_minmax_fp32_ukernel_up8x9__avx_mul16_add16)
DECLARE_QS8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qs8_dwconv_minmax_fp32_ukernel_up16x9__avx_mul16_add16)

DECLARE_QS8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qs8_dwconv_minmax_fp32_ukernel_up8x9__xop_mul16_add16)
DECLARE_QS8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qs8_dwconv_minmax_fp32_ukernel_up16x9__xop_mul16_add16)

DECLARE_QS8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qs8_dwconv_minmax_fp32_ukernel_up16x9__avx2_mul16_vpmovsx)
DECLARE_QS8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qs8_dwconv_minmax_fp32_ukernel_up32x9__avx2_mul16_vpmovsx)

DECLARE_QS8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qs8_dwconv_minmax_fp32_ukernel_up16x9__avx2_mul16_vpunpck)
DECLARE_QS8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qs8_dwconv_minmax_fp32_ukernel_up32x9__avx2_mul16_vpunpck)

DECLARE_QS8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qs8_dwconv_minmax_fp32_ukernel_up16x9__avx2_mul16_add16_vpunpck)
DECLARE_QS8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qs8_dwconv_minmax_fp32_ukernel_up32x9__avx2_mul16_add16_vpunpck)

DECLARE_QS8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qs8_dwconv_minmax_fp32_ukernel_up8x9__sse41_mul32)
DECLARE_QS8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qs8_dwconv_minmax_fp32_ukernel_up16x9__sse41_mul32)
DECLARE_QS8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qs8_dwconv_minmax_fp32_ukernel_up24x9__sse41_mul32)

DECLARE_QS8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qs8_dwconv_minmax_fp32_ukernel_up8x9__avx_mul32)
DECLARE_QS8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qs8_dwconv_minmax_fp32_ukernel_up16x9__avx_mul32)
DECLARE_QS8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qs8_dwconv_minmax_fp32_ukernel_up24x9__avx_mul32)

DECLARE_QS8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qs8_dwconv_minmax_fp32_ukernel_up8x9__xop_mul32)
DECLARE_QS8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qs8_dwconv_minmax_fp32_ukernel_up16x9__xop_mul32)
DECLARE_QS8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qs8_dwconv_minmax_fp32_ukernel_up24x9__xop_mul32)

DECLARE_QS8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qs8_dwconv_minmax_fp32_ukernel_up8x9__avx2_mul32)
DECLARE_QS8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qs8_dwconv_minmax_fp32_ukernel_up16x9__avx2_mul32)
DECLARE_QS8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qs8_dwconv_minmax_fp32_ukernel_up24x9__avx2_mul32)
DECLARE_QS8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qs8_dwconv_minmax_fp32_ukernel_up32x9__avx2_mul32)

DECLARE_QS8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qs8_dwconv_minmax_fp32_ukernel_up16x9__avx512skx_mul32)
DECLARE_QS8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qs8_dwconv_minmax_fp32_ukernel_up32x9__avx512skx_mul32)

DECLARE_QS8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qs8_dwconv_minmax_fp32_ukernel_up8x9__wasmsimd_mul16)
DECLARE_QS8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qs8_dwconv_minmax_fp32_ukernel_up16x9__wasmsimd_mul16)
DECLARE_QS8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qs8_dwconv_minmax_fp32_ukernel_up24x9__wasmsimd_mul16)

DECLARE_QS8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qs8_dwconv_minmax_fp32_ukernel_up8x9__wasmsimd_mul16_add16)
DECLARE_QS8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qs8_dwconv_minmax_fp32_ukernel_up16x9__wasmsimd_mul16_add16)
DECLARE_QS8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qs8_dwconv_minmax_fp32_ukernel_up24x9__wasmsimd_mul16_add16)

DECLARE_QS8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qs8_dwconv_minmax_fp32_ukernel_up1x9__wasm_fmagic)
DECLARE_QS8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qs8_dwconv_minmax_fp32_ukernel_up2x9__wasm_fmagic)
DECLARE_QS8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qs8_dwconv_minmax_fp32_ukernel_up4x9__wasm_fmagic)

DECLARE_QS8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qs8_dwconv_minmax_fp32_ukernel_up1x9__scalar_fmagic)
DECLARE_QS8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qs8_dwconv_minmax_fp32_ukernel_up2x9__scalar_fmagic)
DECLARE_QS8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qs8_dwconv_minmax_fp32_ukernel_up4x9__scalar_fmagic)

DECLARE_QS8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qs8_dwconv_minmax_fp32_ukernel_up1x9__scalar_imagic)
DECLARE_QS8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qs8_dwconv_minmax_fp32_ukernel_up2x9__scalar_imagic)
DECLARE_QS8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qs8_dwconv_minmax_fp32_ukernel_up4x9__scalar_imagic)

DECLARE_QS8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qs8_dwconv_minmax_fp32_ukernel_up1x9__scalar_lrintf)
DECLARE_QS8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qs8_dwconv_minmax_fp32_ukernel_up2x9__scalar_lrintf)
DECLARE_QS8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qs8_dwconv_minmax_fp32_ukernel_up4x9__scalar_lrintf)

DECLARE_QS8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qs8_dwconv_minmax_fp32_ukernel_up8x25__neon_mul16)
DECLARE_QS8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qs8_dwconv_minmax_fp32_ukernel_up16x25__neon_mul16)
DECLARE_QS8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qs8_dwconv_minmax_fp32_ukernel_up24x25__neon_mul16)
DECLARE_QS8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qs8_dwconv_minmax_fp32_ukernel_up32x25__neon_mul16)

DECLARE_QS8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qs8_dwconv_minmax_fp32_ukernel_up8x25__neonv8_mul16)
DECLARE_QS8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qs8_dwconv_minmax_fp32_ukernel_up16x25__neonv8_mul16)
DECLARE_QS8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qs8_dwconv_minmax_fp32_ukernel_up24x25__neonv8_mul16)
DECLARE_QS8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qs8_dwconv_minmax_fp32_ukernel_up32x25__neonv8_mul16)

DECLARE_QS8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qs8_dwconv_minmax_rndnu_ukernel_up8x25__neon_mul16)
DECLARE_QS8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qs8_dwconv_minmax_rndnu_ukernel_up16x25__neon_mul16)
DECLARE_QS8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qs8_dwconv_minmax_rndnu_ukernel_up24x25__neon_mul16)
DECLARE_QS8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qs8_dwconv_minmax_rndnu_ukernel_up32x25__neon_mul16)

DECLARE_QS8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qs8_dwconv_minmax_fp32_ukernel_up8x25__sse2_mul16)
DECLARE_QS8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qs8_dwconv_minmax_fp32_ukernel_up16x25__sse2_mul16)
DECLARE_QS8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qs8_dwconv_minmax_fp32_ukernel_up24x25__sse2_mul16)

DECLARE_QS8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qs8_dwconv_minmax_fp32_ukernel_up8x25__sse2_mul16_add16)
DECLARE_QS8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qs8_dwconv_minmax_fp32_ukernel_up16x25__sse2_mul16_add16)

DECLARE_QS8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qs8_dwconv_minmax_fp32_ukernel_up8x25__sse41_mul16)
DECLARE_QS8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qs8_dwconv_minmax_fp32_ukernel_up16x25__sse41_mul16)
DECLARE_QS8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qs8_dwconv_minmax_fp32_ukernel_up24x25__sse41_mul16)

DECLARE_QS8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qs8_dwconv_minmax_fp32_ukernel_up8x25__sse41_mul16_add16)
DECLARE_QS8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qs8_dwconv_minmax_fp32_ukernel_up16x25__sse41_mul16_add16)

DECLARE_QS8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qs8_dwconv_minmax_fp32_ukernel_up8x25__avx_mul16)
DECLARE_QS8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qs8_dwconv_minmax_fp32_ukernel_up16x25__avx_mul16)
DECLARE_QS8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qs8_dwconv_minmax_fp32_ukernel_up24x25__avx_mul16)

DECLARE_QS8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qs8_dwconv_minmax_fp32_ukernel_up8x25__avx_mul16_add16)
DECLARE_QS8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qs8_dwconv_minmax_fp32_ukernel_up16x25__avx_mul16_add16)

DECLARE_QS8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qs8_dwconv_minmax_fp32_ukernel_up8x25__xop_mul16_add16)
DECLARE_QS8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qs8_dwconv_minmax_fp32_ukernel_up16x25__xop_mul16_add16)

DECLARE_QS8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qs8_dwconv_minmax_fp32_ukernel_up16x25__avx2_mul16_vpmovsx)
DECLARE_QS8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qs8_dwconv_minmax_fp32_ukernel_up32x25__avx2_mul16_vpmovsx)

DECLARE_QS8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qs8_dwconv_minmax_fp32_ukernel_up16x25__avx2_mul16_vpunpck)
DECLARE_QS8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qs8_dwconv_minmax_fp32_ukernel_up32x25__avx2_mul16_vpunpck)

DECLARE_QS8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qs8_dwconv_minmax_fp32_ukernel_up16x25__avx2_mul16_add16_vpunpck)
DECLARE_QS8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qs8_dwconv_minmax_fp32_ukernel_up32x25__avx2_mul16_add16_vpunpck)

DECLARE_QS8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qs8_dwconv_minmax_fp32_ukernel_up8x25__sse41_mul32)
DECLARE_QS8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qs8_dwconv_minmax_fp32_ukernel_up16x25__sse41_mul32)
DECLARE_QS8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qs8_dwconv_minmax_fp32_ukernel_up24x25__sse41_mul32)

DECLARE_QS8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qs8_dwconv_minmax_fp32_ukernel_up8x25__avx_mul32)
DECLARE_QS8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qs8_dwconv_minmax_fp32_ukernel_up16x25__avx_mul32)
DECLARE_QS8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qs8_dwconv_minmax_fp32_ukernel_up24x25__avx_mul32)

DECLARE_QS8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qs8_dwconv_minmax_fp32_ukernel_up8x25__xop_mul32)
DECLARE_QS8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qs8_dwconv_minmax_fp32_ukernel_up16x25__xop_mul32)
DECLARE_QS8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qs8_dwconv_minmax_fp32_ukernel_up24x25__xop_mul32)

DECLARE_QS8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qs8_dwconv_minmax_fp32_ukernel_up8x25__avx2_mul32)
DECLARE_QS8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qs8_dwconv_minmax_fp32_ukernel_up16x25__avx2_mul32)
DECLARE_QS8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qs8_dwconv_minmax_fp32_ukernel_up24x25__avx2_mul32)
DECLARE_QS8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qs8_dwconv_minmax_fp32_ukernel_up32x25__avx2_mul32)

DECLARE_QS8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qs8_dwconv_minmax_fp32_ukernel_up16x25__avx512skx_mul32)
DECLARE_QS8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qs8_dwconv_minmax_fp32_ukernel_up32x25__avx512skx_mul32)

DECLARE_QS8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qs8_dwconv_minmax_fp32_ukernel_up8x25__wasmsimd_mul16)
DECLARE_QS8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qs8_dwconv_minmax_fp32_ukernel_up16x25__wasmsimd_mul16)
DECLARE_QS8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qs8_dwconv_minmax_fp32_ukernel_up24x25__wasmsimd_mul16)

DECLARE_QS8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qs8_dwconv_minmax_fp32_ukernel_up8x25__wasmsimd_mul16_add16)
DECLARE_QS8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qs8_dwconv_minmax_fp32_ukernel_up16x25__wasmsimd_mul16_add16)
DECLARE_QS8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qs8_dwconv_minmax_fp32_ukernel_up24x25__wasmsimd_mul16_add16)

DECLARE_QS8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qs8_dwconv_minmax_fp32_ukernel_up1x25__wasm_fmagic)
DECLARE_QS8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qs8_dwconv_minmax_fp32_ukernel_up2x25__wasm_fmagic)
DECLARE_QS8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qs8_dwconv_minmax_fp32_ukernel_up4x25__wasm_fmagic)

DECLARE_QS8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qs8_dwconv_minmax_fp32_ukernel_up1x25__scalar_fmagic)
DECLARE_QS8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qs8_dwconv_minmax_fp32_ukernel_up2x25__scalar_fmagic)
DECLARE_QS8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qs8_dwconv_minmax_fp32_ukernel_up4x25__scalar_fmagic)

DECLARE_QS8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qs8_dwconv_minmax_fp32_ukernel_up1x25__scalar_imagic)
DECLARE_QS8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qs8_dwconv_minmax_fp32_ukernel_up2x25__scalar_imagic)
DECLARE_QS8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qs8_dwconv_minmax_fp32_ukernel_up4x25__scalar_imagic)

DECLARE_QS8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qs8_dwconv_minmax_fp32_ukernel_up1x25__scalar_lrintf)
DECLARE_QS8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qs8_dwconv_minmax_fp32_ukernel_up2x25__scalar_lrintf)
DECLARE_QS8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qs8_dwconv_minmax_fp32_ukernel_up4x25__scalar_lrintf)


#define DECLARE_QC8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(fn_name) \
  XNN_INTERNAL void fn_name(                                        \
    size_t channels,                                                \
    size_t output_width,                                            \
    size_t kernel_elements,                                         \
    const int8_t** input,                                           \
    const void* weights,                                            \
    int8_t* output,                                                 \
    size_t input_stride,                                            \
    size_t output_increment,                                        \
    size_t input_offset,                                            \
    const int8_t* zero,                                             \
    const union xnn_qs8_minmax_params* params);

DECLARE_QC8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qc8_dwconv_minmax_fp32_ukernel_up8x3__aarch32_neonv8_mla8_cortex_a35)
DECLARE_QC8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qc8_dwconv_minmax_fp32_ukernel_up16x3__aarch32_neonv8_mla8_cortex_a35)

DECLARE_QC8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qc8_dwconv_minmax_fp32_ukernel_up8x3__neon_mla8_ld64)
DECLARE_QC8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qc8_dwconv_minmax_fp32_ukernel_up16x3__neon_mla8_ld64)
DECLARE_QC8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qc8_dwconv_minmax_fp32_ukernel_up16x3__neon_mla8_ld128)

DECLARE_QC8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qc8_dwconv_minmax_fp32_ukernel_up8x3__neonv8_mla8_ld64)
DECLARE_QC8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qc8_dwconv_minmax_fp32_ukernel_up16x3__neonv8_mla8_ld64)
DECLARE_QC8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qc8_dwconv_minmax_fp32_ukernel_up16x3__neonv8_mla8_ld128)

DECLARE_QC8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qc8_dwconv_minmax_fp32_ukernel_up8x9__neon_mul8_ld64)
DECLARE_QC8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qc8_dwconv_minmax_fp32_ukernel_up16x9__neon_mul8_ld64)
DECLARE_QC8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qc8_dwconv_minmax_fp32_ukernel_up16x9__neon_mul8_ld128)

DECLARE_QC8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qc8_dwconv_minmax_fp32_ukernel_up8x9__neon_mla8_ld64)
DECLARE_QC8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qc8_dwconv_minmax_fp32_ukernel_up16x9__neon_mla8_ld64)
DECLARE_QC8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qc8_dwconv_minmax_fp32_ukernel_up16x9__neon_mla8_ld128)

DECLARE_QC8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qc8_dwconv_minmax_fp32_ukernel_up8x9__neonv8_mul8_ld64)
DECLARE_QC8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qc8_dwconv_minmax_fp32_ukernel_up16x9__neonv8_mul8_ld64)
DECLARE_QC8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qc8_dwconv_minmax_fp32_ukernel_up16x9__neonv8_mul8_ld128)

DECLARE_QC8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qc8_dwconv_minmax_fp32_ukernel_up8x9__neonv8_mla8_ld64)
DECLARE_QC8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qc8_dwconv_minmax_fp32_ukernel_up16x9__neonv8_mla8_ld64)
DECLARE_QC8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qc8_dwconv_minmax_fp32_ukernel_up16x9__neonv8_mla8_ld128)

DECLARE_QC8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qc8_dwconv_minmax_fp32_ukernel_up8x25__neon_mul8_ld64)
DECLARE_QC8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qc8_dwconv_minmax_fp32_ukernel_up16x25__neon_mul8_ld64)
DECLARE_QC8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qc8_dwconv_minmax_fp32_ukernel_up16x25__neon_mul8_ld128)

DECLARE_QC8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qc8_dwconv_minmax_fp32_ukernel_up8x25__neon_mla8_ld64)
DECLARE_QC8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qc8_dwconv_minmax_fp32_ukernel_up16x25__neon_mla8_ld64)
DECLARE_QC8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qc8_dwconv_minmax_fp32_ukernel_up16x25__neon_mla8_ld128)

DECLARE_QC8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qc8_dwconv_minmax_fp32_ukernel_up8x25__neonv8_mul8_ld64)
DECLARE_QC8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qc8_dwconv_minmax_fp32_ukernel_up16x25__neonv8_mul8_ld64)
DECLARE_QC8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qc8_dwconv_minmax_fp32_ukernel_up16x25__neonv8_mul8_ld128)

DECLARE_QC8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qc8_dwconv_minmax_fp32_ukernel_up8x25__neonv8_mla8_ld64)
DECLARE_QC8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qc8_dwconv_minmax_fp32_ukernel_up16x25__neonv8_mla8_ld64)
DECLARE_QC8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qc8_dwconv_minmax_fp32_ukernel_up16x25__neonv8_mla8_ld128)

DECLARE_QC8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qc8_dwconv_minmax_fp32_ukernel_up8x9__neon_mul16)
DECLARE_QC8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qc8_dwconv_minmax_fp32_ukernel_up16x9__neon_mul16)
DECLARE_QC8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qc8_dwconv_minmax_fp32_ukernel_up24x9__neon_mul16)
DECLARE_QC8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qc8_dwconv_minmax_fp32_ukernel_up32x9__neon_mul16)

DECLARE_QC8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qc8_dwconv_minmax_fp32_ukernel_up8x9__neonv8_mul16)
DECLARE_QC8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qc8_dwconv_minmax_fp32_ukernel_up16x9__neonv8_mul16)
DECLARE_QC8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qc8_dwconv_minmax_fp32_ukernel_up24x9__neonv8_mul16)
DECLARE_QC8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qc8_dwconv_minmax_fp32_ukernel_up32x9__neonv8_mul16)

DECLARE_QC8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qc8_dwconv_minmax_fp32_ukernel_up8x25__neon_mul16)
DECLARE_QC8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qc8_dwconv_minmax_fp32_ukernel_up16x25__neon_mul16)
DECLARE_QC8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qc8_dwconv_minmax_fp32_ukernel_up24x25__neon_mul16)
DECLARE_QC8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qc8_dwconv_minmax_fp32_ukernel_up32x25__neon_mul16)

DECLARE_QC8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qc8_dwconv_minmax_fp32_ukernel_up8x25__neonv8_mul16)
DECLARE_QC8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qc8_dwconv_minmax_fp32_ukernel_up16x25__neonv8_mul16)
DECLARE_QC8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qc8_dwconv_minmax_fp32_ukernel_up24x25__neonv8_mul16)
DECLARE_QC8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qc8_dwconv_minmax_fp32_ukernel_up32x25__neonv8_mul16)

DECLARE_QC8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qc8_dwconv_minmax_fp32_ukernel_up8x3__sse2_mul16)
DECLARE_QC8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qc8_dwconv_minmax_fp32_ukernel_up8x3__sse41_mul16)
DECLARE_QC8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qc8_dwconv_minmax_fp32_ukernel_up16x3__avx_mul16_add16)
DECLARE_QC8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qc8_dwconv_minmax_fp32_ukernel_up16x3__avx2_mul32)
DECLARE_QC8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qc8_dwconv_minmax_fp32_ukernel_up16x3__xop_mul16_add16)
DECLARE_QC8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qc8_dwconv_minmax_fp32_ukernel_up32x3__avx512skx_mul32)

DECLARE_QC8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qc8_dwconv_minmax_fp32_ukernel_up16x3__wasmsimd_mul16_add16)

DECLARE_QC8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qc8_dwconv_minmax_fp32_ukernel_up2x3__scalar_imagic)
DECLARE_QC8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qc8_dwconv_minmax_fp32_ukernel_up2x3__scalar_lrintf)
DECLARE_QC8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qc8_dwconv_minmax_fp32_ukernel_up2x3__wasm_fmagic)

DECLARE_QC8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qc8_dwconv_minmax_fp32_ukernel_up8x9__sse2_mul16)
DECLARE_QC8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qc8_dwconv_minmax_fp32_ukernel_up16x9__sse2_mul16)
DECLARE_QC8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qc8_dwconv_minmax_fp32_ukernel_up24x9__sse2_mul16)

DECLARE_QC8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qc8_dwconv_minmax_fp32_ukernel_up8x9__sse2_mul16_add16)
DECLARE_QC8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qc8_dwconv_minmax_fp32_ukernel_up16x9__sse2_mul16_add16)

DECLARE_QC8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qc8_dwconv_minmax_fp32_ukernel_up8x9__sse41_mul16)
DECLARE_QC8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qc8_dwconv_minmax_fp32_ukernel_up16x9__sse41_mul16)
DECLARE_QC8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qc8_dwconv_minmax_fp32_ukernel_up24x9__sse41_mul16)

DECLARE_QC8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qc8_dwconv_minmax_fp32_ukernel_up8x9__sse41_mul16_add16)
DECLARE_QC8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qc8_dwconv_minmax_fp32_ukernel_up16x9__sse41_mul16_add16)

DECLARE_QC8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qc8_dwconv_minmax_fp32_ukernel_up8x9__avx_mul16)
DECLARE_QC8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qc8_dwconv_minmax_fp32_ukernel_up16x9__avx_mul16)
DECLARE_QC8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qc8_dwconv_minmax_fp32_ukernel_up24x9__avx_mul16)

DECLARE_QC8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qc8_dwconv_minmax_fp32_ukernel_up8x9__avx_mul16_add16)
DECLARE_QC8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qc8_dwconv_minmax_fp32_ukernel_up16x9__avx_mul16_add16)

DECLARE_QC8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qc8_dwconv_minmax_fp32_ukernel_up8x9__xop_mul16_add16)
DECLARE_QC8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qc8_dwconv_minmax_fp32_ukernel_up16x9__xop_mul16_add16)

DECLARE_QC8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qc8_dwconv_minmax_fp32_ukernel_up16x9__avx2_mul16_vpmovsx)
DECLARE_QC8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qc8_dwconv_minmax_fp32_ukernel_up32x9__avx2_mul16_vpmovsx)

DECLARE_QC8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qc8_dwconv_minmax_fp32_ukernel_up16x9__avx2_mul16_vpunpck)
DECLARE_QC8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qc8_dwconv_minmax_fp32_ukernel_up32x9__avx2_mul16_vpunpck)

DECLARE_QC8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qc8_dwconv_minmax_fp32_ukernel_up16x9__avx2_mul16_add16_vpunpck)
DECLARE_QC8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qc8_dwconv_minmax_fp32_ukernel_up32x9__avx2_mul16_add16_vpunpck)

DECLARE_QC8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qc8_dwconv_minmax_fp32_ukernel_up8x9__sse41_mul32)
DECLARE_QC8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qc8_dwconv_minmax_fp32_ukernel_up16x9__sse41_mul32)
DECLARE_QC8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qc8_dwconv_minmax_fp32_ukernel_up24x9__sse41_mul32)

DECLARE_QC8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qc8_dwconv_minmax_fp32_ukernel_up8x9__avx_mul32)
DECLARE_QC8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qc8_dwconv_minmax_fp32_ukernel_up16x9__avx_mul32)
DECLARE_QC8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qc8_dwconv_minmax_fp32_ukernel_up24x9__avx_mul32)

DECLARE_QC8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qc8_dwconv_minmax_fp32_ukernel_up8x9__xop_mul32)
DECLARE_QC8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qc8_dwconv_minmax_fp32_ukernel_up16x9__xop_mul32)
DECLARE_QC8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qc8_dwconv_minmax_fp32_ukernel_up24x9__xop_mul32)

DECLARE_QC8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qc8_dwconv_minmax_fp32_ukernel_up8x9__avx2_mul32)
DECLARE_QC8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qc8_dwconv_minmax_fp32_ukernel_up16x9__avx2_mul32)
DECLARE_QC8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qc8_dwconv_minmax_fp32_ukernel_up24x9__avx2_mul32)
DECLARE_QC8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qc8_dwconv_minmax_fp32_ukernel_up32x9__avx2_mul32)

DECLARE_QC8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qc8_dwconv_minmax_fp32_ukernel_up16x9__avx512skx_mul32)
DECLARE_QC8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qc8_dwconv_minmax_fp32_ukernel_up32x9__avx512skx_mul32)

DECLARE_QC8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qc8_dwconv_minmax_fp32_ukernel_up8x9__wasmsimd_mul16)
DECLARE_QC8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qc8_dwconv_minmax_fp32_ukernel_up16x9__wasmsimd_mul16)
DECLARE_QC8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qc8_dwconv_minmax_fp32_ukernel_up24x9__wasmsimd_mul16)

DECLARE_QC8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qc8_dwconv_minmax_fp32_ukernel_up8x9__wasmsimd_mul16_add16)
DECLARE_QC8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qc8_dwconv_minmax_fp32_ukernel_up16x9__wasmsimd_mul16_add16)
DECLARE_QC8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qc8_dwconv_minmax_fp32_ukernel_up24x9__wasmsimd_mul16_add16)

DECLARE_QC8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qc8_dwconv_minmax_fp32_ukernel_up1x9__wasm_fmagic)
DECLARE_QC8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qc8_dwconv_minmax_fp32_ukernel_up2x9__wasm_fmagic)
DECLARE_QC8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qc8_dwconv_minmax_fp32_ukernel_up4x9__wasm_fmagic)

DECLARE_QC8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qc8_dwconv_minmax_fp32_ukernel_up1x9__scalar_fmagic)
DECLARE_QC8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qc8_dwconv_minmax_fp32_ukernel_up2x9__scalar_fmagic)
DECLARE_QC8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qc8_dwconv_minmax_fp32_ukernel_up4x9__scalar_fmagic)

DECLARE_QC8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qc8_dwconv_minmax_fp32_ukernel_up1x9__scalar_imagic)
DECLARE_QC8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qc8_dwconv_minmax_fp32_ukernel_up2x9__scalar_imagic)
DECLARE_QC8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qc8_dwconv_minmax_fp32_ukernel_up4x9__scalar_imagic)

DECLARE_QC8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qc8_dwconv_minmax_fp32_ukernel_up1x9__scalar_lrintf)
DECLARE_QC8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qc8_dwconv_minmax_fp32_ukernel_up2x9__scalar_lrintf)
DECLARE_QC8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qc8_dwconv_minmax_fp32_ukernel_up4x9__scalar_lrintf)

DECLARE_QC8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qc8_dwconv_minmax_fp32_ukernel_up8x25__sse2_mul16)
DECLARE_QC8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qc8_dwconv_minmax_fp32_ukernel_up16x25__sse2_mul16)
DECLARE_QC8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qc8_dwconv_minmax_fp32_ukernel_up24x25__sse2_mul16)

DECLARE_QC8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qc8_dwconv_minmax_fp32_ukernel_up8x25__sse2_mul16_add16)
DECLARE_QC8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qc8_dwconv_minmax_fp32_ukernel_up16x25__sse2_mul16_add16)

DECLARE_QC8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qc8_dwconv_minmax_fp32_ukernel_up8x25__sse41_mul16)
DECLARE_QC8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qc8_dwconv_minmax_fp32_ukernel_up16x25__sse41_mul16)
DECLARE_QC8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qc8_dwconv_minmax_fp32_ukernel_up24x25__sse41_mul16)

DECLARE_QC8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qc8_dwconv_minmax_fp32_ukernel_up8x25__sse41_mul16_add16)
DECLARE_QC8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qc8_dwconv_minmax_fp32_ukernel_up16x25__sse41_mul16_add16)

DECLARE_QC8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qc8_dwconv_minmax_fp32_ukernel_up8x25__avx_mul16)
DECLARE_QC8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qc8_dwconv_minmax_fp32_ukernel_up16x25__avx_mul16)
DECLARE_QC8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qc8_dwconv_minmax_fp32_ukernel_up24x25__avx_mul16)

DECLARE_QC8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qc8_dwconv_minmax_fp32_ukernel_up8x25__avx_mul16_add16)
DECLARE_QC8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qc8_dwconv_minmax_fp32_ukernel_up16x25__avx_mul16_add16)

DECLARE_QC8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qc8_dwconv_minmax_fp32_ukernel_up8x25__xop_mul16_add16)
DECLARE_QC8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qc8_dwconv_minmax_fp32_ukernel_up16x25__xop_mul16_add16)

DECLARE_QC8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qc8_dwconv_minmax_fp32_ukernel_up16x25__avx2_mul16_vpmovsx)
DECLARE_QC8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qc8_dwconv_minmax_fp32_ukernel_up32x25__avx2_mul16_vpmovsx)

DECLARE_QC8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qc8_dwconv_minmax_fp32_ukernel_up16x25__avx2_mul16_vpunpck)
DECLARE_QC8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qc8_dwconv_minmax_fp32_ukernel_up32x25__avx2_mul16_vpunpck)

DECLARE_QC8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qc8_dwconv_minmax_fp32_ukernel_up16x25__avx2_mul16_add16_vpunpck)
DECLARE_QC8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qc8_dwconv_minmax_fp32_ukernel_up32x25__avx2_mul16_add16_vpunpck)

DECLARE_QC8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qc8_dwconv_minmax_fp32_ukernel_up8x25__sse41_mul32)
DECLARE_QC8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qc8_dwconv_minmax_fp32_ukernel_up16x25__sse41_mul32)
DECLARE_QC8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qc8_dwconv_minmax_fp32_ukernel_up24x25__sse41_mul32)

DECLARE_QC8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qc8_dwconv_minmax_fp32_ukernel_up8x25__avx_mul32)
DECLARE_QC8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qc8_dwconv_minmax_fp32_ukernel_up16x25__avx_mul32)
DECLARE_QC8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qc8_dwconv_minmax_fp32_ukernel_up24x25__avx_mul32)

DECLARE_QC8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qc8_dwconv_minmax_fp32_ukernel_up8x25__xop_mul32)
DECLARE_QC8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qc8_dwconv_minmax_fp32_ukernel_up16x25__xop_mul32)
DECLARE_QC8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qc8_dwconv_minmax_fp32_ukernel_up24x25__xop_mul32)

DECLARE_QC8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qc8_dwconv_minmax_fp32_ukernel_up8x25__avx2_mul32)
DECLARE_QC8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qc8_dwconv_minmax_fp32_ukernel_up16x25__avx2_mul32)
DECLARE_QC8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qc8_dwconv_minmax_fp32_ukernel_up24x25__avx2_mul32)
DECLARE_QC8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qc8_dwconv_minmax_fp32_ukernel_up32x25__avx2_mul32)

DECLARE_QC8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qc8_dwconv_minmax_fp32_ukernel_up16x25__avx512skx_mul32)
DECLARE_QC8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qc8_dwconv_minmax_fp32_ukernel_up32x25__avx512skx_mul32)

DECLARE_QC8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qc8_dwconv_minmax_fp32_ukernel_up8x25__wasmsimd_mul16)
DECLARE_QC8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qc8_dwconv_minmax_fp32_ukernel_up16x25__wasmsimd_mul16)
DECLARE_QC8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qc8_dwconv_minmax_fp32_ukernel_up24x25__wasmsimd_mul16)

DECLARE_QC8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qc8_dwconv_minmax_fp32_ukernel_up8x25__wasmsimd_mul16_add16)
DECLARE_QC8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qc8_dwconv_minmax_fp32_ukernel_up16x25__wasmsimd_mul16_add16)
DECLARE_QC8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qc8_dwconv_minmax_fp32_ukernel_up24x25__wasmsimd_mul16_add16)

DECLARE_QC8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qc8_dwconv_minmax_fp32_ukernel_up1x25__wasm_fmagic)
DECLARE_QC8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qc8_dwconv_minmax_fp32_ukernel_up2x25__wasm_fmagic)
DECLARE_QC8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qc8_dwconv_minmax_fp32_ukernel_up4x25__wasm_fmagic)

DECLARE_QC8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qc8_dwconv_minmax_fp32_ukernel_up1x25__scalar_fmagic)
DECLARE_QC8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qc8_dwconv_minmax_fp32_ukernel_up2x25__scalar_fmagic)
DECLARE_QC8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qc8_dwconv_minmax_fp32_ukernel_up4x25__scalar_fmagic)

DECLARE_QC8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qc8_dwconv_minmax_fp32_ukernel_up1x25__scalar_imagic)
DECLARE_QC8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qc8_dwconv_minmax_fp32_ukernel_up2x25__scalar_imagic)
DECLARE_QC8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qc8_dwconv_minmax_fp32_ukernel_up4x25__scalar_imagic)

DECLARE_QC8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qc8_dwconv_minmax_fp32_ukernel_up1x25__scalar_lrintf)
DECLARE_QC8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qc8_dwconv_minmax_fp32_ukernel_up2x25__scalar_lrintf)
DECLARE_QC8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qc8_dwconv_minmax_fp32_ukernel_up4x25__scalar_lrintf)


#define DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(fn_name) \
  XNN_INTERNAL void fn_name(                                      \
    size_t input_height,                                          \
    size_t input_width,                                           \
    const float* input,                                           \
    const float* weights,                                         \
    const float* zero,                                            \
    float* output,                                                \
    uint32_t padding_top,                                         \
    const union xnn_f32_chw_params* params);

DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_3x3p1__neon_1x4)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_3x3p1__neon_1x4_acc2)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_3x3p1__neon_1x4_acc3)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_3x3p1__neon_1x4_acc4)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_3x3p1__neon_2x4)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_3x3p1__neon_2x4_acc2)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_3x3p1__neon_3x4)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_3x3p1__neon_4x4)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_3x3p1__neon_5x4)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_3x3p1__neon_6x4)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_3x3p1__neonfma_1x4)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_3x3p1__neonfma_1x4_acc2)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_3x3p1__neonfma_1x4_acc3)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_3x3p1__neonfma_1x4_acc4)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_3x3p1__neonfma_2x4)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_3x3p1__neonfma_2x4_acc2)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_3x3p1__neonfma_3x4)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_3x3p1__neonfma_4x4)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_3x3p1__neonfma_5x4)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_3x3p1__neonfma_6x4)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_3x3p1__scalar_1x1)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_3x3p1__scalar_1x1_acc2)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_3x3p1__scalar_1x1_acc3)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_3x3p1__scalar_1x1_acc4)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_3x3p1__scalar_2x1)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_3x3p1__scalar_2x1_acc2)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_3x3p1__scalar_3x1)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_3x3p1__scalar_4x1)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_3x3p1__scalar_5x1)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_3x3p1__scalar_6x1)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_3x3p1__sse_1x4)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_3x3p1__sse_1x4_acc2)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_3x3p1__sse_1x4_acc3)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_3x3p1__sse_1x4_acc4)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_3x3p1__sse_2x4)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_3x3p1__sse_2x4_acc2)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_3x3p1__sse_3x4)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_3x3p1__sse_4x4)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_3x3p1__sse_5x4)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_3x3p1__sse_6x4)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_3x3p1__ssse3_1x4)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_3x3p1__ssse3_1x4_acc2)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_3x3p1__ssse3_1x4_acc3)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_3x3p1__ssse3_1x4_acc4)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_3x3p1__ssse3_2x4)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_3x3p1__ssse3_2x4_acc2)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_3x3p1__ssse3_3x4)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_3x3p1__ssse3_4x4)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_3x3p1__ssse3_5x4)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_3x3p1__ssse3_6x4)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_3x3p1__wasmsimd_arm_loadsplat_1x4)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_3x3p1__wasmsimd_arm_loadsplat_1x4_acc2)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_3x3p1__wasmsimd_arm_loadsplat_1x4_acc3)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_3x3p1__wasmsimd_arm_loadsplat_1x4_acc4)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_3x3p1__wasmsimd_arm_loadsplat_2x4)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_3x3p1__wasmsimd_arm_loadsplat_2x4_acc2)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_3x3p1__wasmsimd_arm_loadsplat_3x4)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_3x3p1__wasmsimd_arm_loadsplat_4x4)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_3x3p1__wasmsimd_arm_loadsplat_5x4)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_3x3p1__wasmsimd_arm_loadsplat_6x4)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_3x3p1__wasmsimd_arm_splat_1x4)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_3x3p1__wasmsimd_arm_splat_1x4_acc2)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_3x3p1__wasmsimd_arm_splat_1x4_acc3)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_3x3p1__wasmsimd_arm_splat_1x4_acc4)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_3x3p1__wasmsimd_arm_splat_2x4)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_3x3p1__wasmsimd_arm_splat_2x4_acc2)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_3x3p1__wasmsimd_arm_splat_3x4)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_3x3p1__wasmsimd_arm_splat_4x4)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_3x3p1__wasmsimd_arm_splat_5x4)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_3x3p1__wasmsimd_arm_splat_6x4)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_3x3p1__wasmsimd_x86_loadsplat_1x4)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_3x3p1__wasmsimd_x86_loadsplat_1x4_acc2)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_3x3p1__wasmsimd_x86_loadsplat_1x4_acc3)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_3x3p1__wasmsimd_x86_loadsplat_1x4_acc4)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_3x3p1__wasmsimd_x86_loadsplat_2x4)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_3x3p1__wasmsimd_x86_loadsplat_2x4_acc2)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_3x3p1__wasmsimd_x86_loadsplat_3x4)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_3x3p1__wasmsimd_x86_loadsplat_4x4)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_3x3p1__wasmsimd_x86_loadsplat_5x4)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_3x3p1__wasmsimd_x86_loadsplat_6x4)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_3x3p1__wasmsimd_x86_splat_1x4)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_3x3p1__wasmsimd_x86_splat_1x4_acc2)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_3x3p1__wasmsimd_x86_splat_1x4_acc3)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_3x3p1__wasmsimd_x86_splat_1x4_acc4)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_3x3p1__wasmsimd_x86_splat_2x4)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_3x3p1__wasmsimd_x86_splat_2x4_acc2)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_3x3p1__wasmsimd_x86_splat_3x4)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_3x3p1__wasmsimd_x86_splat_4x4)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_3x3p1__wasmsimd_x86_splat_5x4)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_3x3p1__wasmsimd_x86_splat_6x4)

DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_3x3s2p1__neon_1x4)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_3x3s2p1__neon_1x4_acc2)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_3x3s2p1__neon_1x4_acc3)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_3x3s2p1__neon_1x4_acc4)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_3x3s2p1__neon_2x4)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_3x3s2p1__neon_2x4_acc2)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_3x3s2p1__neon_3x4)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_3x3s2p1__neon_4x4)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_3x3s2p1__neonfma_1x4)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_3x3s2p1__neonfma_1x4_acc2)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_3x3s2p1__neonfma_1x4_acc3)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_3x3s2p1__neonfma_1x4_acc4)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_3x3s2p1__neonfma_2x4)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_3x3s2p1__neonfma_2x4_acc2)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_3x3s2p1__neonfma_3x4)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_3x3s2p1__neonfma_4x4)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_3x3s2p1__neonfma_5x4)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_3x3s2p1__neonfma_6x4)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_3x3s2p1__scalar_1x1)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_3x3s2p1__scalar_1x1_acc2)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_3x3s2p1__scalar_1x1_acc3)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_3x3s2p1__scalar_1x1_acc4)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_3x3s2p1__scalar_2x1)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_3x3s2p1__scalar_2x1_acc2)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_3x3s2p1__scalar_3x1)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_3x3s2p1__scalar_4x1)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_3x3s2p1__sse_1x4)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_3x3s2p1__sse_1x4_acc2)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_3x3s2p1__sse_1x4_acc3)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_3x3s2p1__sse_1x4_acc4)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_3x3s2p1__sse_2x4)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_3x3s2p1__sse_2x4_acc2)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_3x3s2p1__sse_3x4)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_3x3s2p1__sse_4x4)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_3x3s2p1__wasmsimd_arm_loadsplat_1x4)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_3x3s2p1__wasmsimd_arm_loadsplat_1x4_acc2)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_3x3s2p1__wasmsimd_arm_loadsplat_1x4_acc3)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_3x3s2p1__wasmsimd_arm_loadsplat_1x4_acc4)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_3x3s2p1__wasmsimd_arm_loadsplat_2x4)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_3x3s2p1__wasmsimd_arm_loadsplat_2x4_acc2)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_3x3s2p1__wasmsimd_arm_loadsplat_3x4)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_3x3s2p1__wasmsimd_arm_loadsplat_4x4)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_3x3s2p1__wasmsimd_arm_splat_1x4)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_3x3s2p1__wasmsimd_arm_splat_1x4_acc2)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_3x3s2p1__wasmsimd_arm_splat_1x4_acc3)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_3x3s2p1__wasmsimd_arm_splat_1x4_acc4)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_3x3s2p1__wasmsimd_arm_splat_2x4)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_3x3s2p1__wasmsimd_arm_splat_2x4_acc2)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_3x3s2p1__wasmsimd_arm_splat_3x4)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_3x3s2p1__wasmsimd_arm_splat_4x4)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_3x3s2p1__wasmsimd_x86_loadsplat_1x4)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_3x3s2p1__wasmsimd_x86_loadsplat_1x4_acc2)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_3x3s2p1__wasmsimd_x86_loadsplat_1x4_acc3)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_3x3s2p1__wasmsimd_x86_loadsplat_1x4_acc4)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_3x3s2p1__wasmsimd_x86_loadsplat_2x4)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_3x3s2p1__wasmsimd_x86_loadsplat_2x4_acc2)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_3x3s2p1__wasmsimd_x86_loadsplat_3x4)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_3x3s2p1__wasmsimd_x86_loadsplat_4x4)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_3x3s2p1__wasmsimd_x86_splat_1x4)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_3x3s2p1__wasmsimd_x86_splat_1x4_acc2)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_3x3s2p1__wasmsimd_x86_splat_1x4_acc3)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_3x3s2p1__wasmsimd_x86_splat_1x4_acc4)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_3x3s2p1__wasmsimd_x86_splat_2x4)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_3x3s2p1__wasmsimd_x86_splat_2x4_acc2)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_3x3s2p1__wasmsimd_x86_splat_3x4)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_3x3s2p1__wasmsimd_x86_splat_4x4)

DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_5x5p2__neon_1x4)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_5x5p2__neon_1x4_acc2)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_5x5p2__neon_1x4_acc3)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_5x5p2__neon_1x4_acc4)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_5x5p2__neon_1x4_acc5)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_5x5p2__neon_2x4)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_5x5p2__neon_2x4_acc2)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_5x5p2__neon_2x4_acc3)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_5x5p2__neon_3x4)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_5x5p2__neon_3x4_acc2)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_5x5p2__neon_4x4)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_5x5p2__neon_4x4_acc2)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_5x5p2__neon_5x4)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_5x5p2__neonfma_1x4)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_5x5p2__neonfma_1x4_acc2)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_5x5p2__neonfma_1x4_acc3)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_5x5p2__neonfma_1x4_acc4)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_5x5p2__neonfma_1x4_acc5)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_5x5p2__neonfma_2x4)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_5x5p2__neonfma_2x4_acc2)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_5x5p2__neonfma_2x4_acc3)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_5x5p2__neonfma_3x4)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_5x5p2__neonfma_3x4_acc2)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_5x5p2__neonfma_4x4)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_5x5p2__neonfma_4x4_acc2)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_5x5p2__neonfma_5x4)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_5x5p2__scalar_1x1)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_5x5p2__scalar_1x1_acc2)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_5x5p2__scalar_1x1_acc3)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_5x5p2__scalar_1x1_acc4)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_5x5p2__scalar_1x1_acc5)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_5x5p2__scalar_2x1)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_5x5p2__scalar_2x1_acc2)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_5x5p2__scalar_2x1_acc3)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_5x5p2__scalar_3x1)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_5x5p2__scalar_3x1_acc2)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_5x5p2__sse_1x4)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_5x5p2__sse_1x4_acc2)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_5x5p2__sse_1x4_acc3)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_5x5p2__sse_1x4_acc4)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_5x5p2__sse_1x4_acc5)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_5x5p2__sse_2x4)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_5x5p2__sse_2x4_acc2)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_5x5p2__sse_2x4_acc3)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_5x5p2__sse_3x4)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_5x5p2__sse_3x4_acc2)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_5x5p2__sse_4x4)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_5x5p2__sse_4x4_acc2)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_5x5p2__sse_5x4)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_5x5p2__wasmsimd_arm_loadsplat_1x4)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_5x5p2__wasmsimd_arm_loadsplat_1x4_acc2)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_5x5p2__wasmsimd_arm_loadsplat_1x4_acc3)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_5x5p2__wasmsimd_arm_loadsplat_1x4_acc4)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_5x5p2__wasmsimd_arm_loadsplat_1x4_acc5)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_5x5p2__wasmsimd_arm_loadsplat_2x4)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_5x5p2__wasmsimd_arm_loadsplat_2x4_acc2)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_5x5p2__wasmsimd_arm_loadsplat_2x4_acc3)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_5x5p2__wasmsimd_arm_loadsplat_3x4)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_5x5p2__wasmsimd_arm_loadsplat_3x4_acc2)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_5x5p2__wasmsimd_arm_loadsplat_4x4)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_5x5p2__wasmsimd_arm_loadsplat_4x4_acc2)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_5x5p2__wasmsimd_arm_loadsplat_5x4)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_5x5p2__wasmsimd_arm_splat_1x4)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_5x5p2__wasmsimd_arm_splat_1x4_acc2)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_5x5p2__wasmsimd_arm_splat_1x4_acc3)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_5x5p2__wasmsimd_arm_splat_1x4_acc4)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_5x5p2__wasmsimd_arm_splat_1x4_acc5)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_5x5p2__wasmsimd_arm_splat_2x4)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_5x5p2__wasmsimd_arm_splat_2x4_acc2)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_5x5p2__wasmsimd_arm_splat_2x4_acc3)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_5x5p2__wasmsimd_arm_splat_3x4)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_5x5p2__wasmsimd_arm_splat_3x4_acc2)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_5x5p2__wasmsimd_arm_splat_4x4)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_5x5p2__wasmsimd_arm_splat_4x4_acc2)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_5x5p2__wasmsimd_arm_splat_5x4)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_5x5p2__wasmsimd_x86_loadsplat_1x4)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_5x5p2__wasmsimd_x86_loadsplat_1x4_acc2)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_5x5p2__wasmsimd_x86_loadsplat_1x4_acc3)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_5x5p2__wasmsimd_x86_loadsplat_1x4_acc4)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_5x5p2__wasmsimd_x86_loadsplat_1x4_acc5)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_5x5p2__wasmsimd_x86_loadsplat_2x4)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_5x5p2__wasmsimd_x86_loadsplat_2x4_acc2)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_5x5p2__wasmsimd_x86_loadsplat_2x4_acc3)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_5x5p2__wasmsimd_x86_loadsplat_3x4)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_5x5p2__wasmsimd_x86_loadsplat_3x4_acc2)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_5x5p2__wasmsimd_x86_loadsplat_4x4)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_5x5p2__wasmsimd_x86_loadsplat_4x4_acc2)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_5x5p2__wasmsimd_x86_loadsplat_5x4)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_5x5p2__wasmsimd_x86_splat_1x4)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_5x5p2__wasmsimd_x86_splat_1x4_acc2)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_5x5p2__wasmsimd_x86_splat_1x4_acc3)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_5x5p2__wasmsimd_x86_splat_1x4_acc4)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_5x5p2__wasmsimd_x86_splat_1x4_acc5)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_5x5p2__wasmsimd_x86_splat_2x4)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_5x5p2__wasmsimd_x86_splat_2x4_acc2)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_5x5p2__wasmsimd_x86_splat_2x4_acc3)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_5x5p2__wasmsimd_x86_splat_3x4)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_5x5p2__wasmsimd_x86_splat_3x4_acc2)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_5x5p2__wasmsimd_x86_splat_4x4)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_5x5p2__wasmsimd_x86_splat_4x4_acc2)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_5x5p2__wasmsimd_x86_splat_5x4)

DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_5x5s2p2__neon_1x4)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_5x5s2p2__neon_1x4_acc2)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_5x5s2p2__neon_1x4_acc3)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_5x5s2p2__neon_1x4_acc4)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_5x5s2p2__neon_1x4_acc5)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_5x5s2p2__neon_2x4)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_5x5s2p2__neon_2x4_acc2)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_5x5s2p2__neon_2x4_acc3)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_5x5s2p2__neon_3x4)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_5x5s2p2__neon_3x4_acc2)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_5x5s2p2__neonfma_1x4)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_5x5s2p2__neonfma_1x4_acc2)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_5x5s2p2__neonfma_1x4_acc3)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_5x5s2p2__neonfma_1x4_acc4)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_5x5s2p2__neonfma_1x4_acc5)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_5x5s2p2__neonfma_2x4)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_5x5s2p2__neonfma_2x4_acc2)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_5x5s2p2__neonfma_2x4_acc3)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_5x5s2p2__neonfma_3x4)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_5x5s2p2__neonfma_3x4_acc2)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_5x5s2p2__scalar_1x1)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_5x5s2p2__scalar_1x1_acc2)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_5x5s2p2__scalar_1x1_acc3)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_5x5s2p2__scalar_1x1_acc4)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_5x5s2p2__scalar_1x1_acc5)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_5x5s2p2__scalar_2x1)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_5x5s2p2__scalar_2x1_acc2)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_5x5s2p2__scalar_2x1_acc3)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_5x5s2p2__scalar_3x1)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_5x5s2p2__scalar_3x1_acc2)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_5x5s2p2__sse_1x4)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_5x5s2p2__sse_1x4_acc2)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_5x5s2p2__sse_1x4_acc3)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_5x5s2p2__sse_1x4_acc4)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_5x5s2p2__sse_1x4_acc5)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_5x5s2p2__sse_2x4)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_5x5s2p2__sse_2x4_acc2)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_5x5s2p2__sse_2x4_acc3)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_5x5s2p2__sse_3x4)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_5x5s2p2__sse_3x4_acc2)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_5x5s2p2__wasmsimd_arm_loadsplat_1x4)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_5x5s2p2__wasmsimd_arm_loadsplat_1x4_acc2)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_5x5s2p2__wasmsimd_arm_loadsplat_1x4_acc3)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_5x5s2p2__wasmsimd_arm_loadsplat_1x4_acc4)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_5x5s2p2__wasmsimd_arm_loadsplat_1x4_acc5)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_5x5s2p2__wasmsimd_arm_loadsplat_2x4)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_5x5s2p2__wasmsimd_arm_loadsplat_2x4_acc2)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_5x5s2p2__wasmsimd_arm_loadsplat_2x4_acc3)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_5x5s2p2__wasmsimd_arm_loadsplat_3x4)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_5x5s2p2__wasmsimd_arm_loadsplat_3x4_acc2)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_5x5s2p2__wasmsimd_arm_splat_1x4)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_5x5s2p2__wasmsimd_arm_splat_1x4_acc2)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_5x5s2p2__wasmsimd_arm_splat_1x4_acc3)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_5x5s2p2__wasmsimd_arm_splat_1x4_acc4)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_5x5s2p2__wasmsimd_arm_splat_1x4_acc5)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_5x5s2p2__wasmsimd_arm_splat_2x4)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_5x5s2p2__wasmsimd_arm_splat_2x4_acc2)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_5x5s2p2__wasmsimd_arm_splat_2x4_acc3)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_5x5s2p2__wasmsimd_arm_splat_3x4)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_5x5s2p2__wasmsimd_arm_splat_3x4_acc2)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_5x5s2p2__wasmsimd_x86_loadsplat_1x4)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_5x5s2p2__wasmsimd_x86_loadsplat_1x4_acc2)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_5x5s2p2__wasmsimd_x86_loadsplat_1x4_acc3)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_5x5s2p2__wasmsimd_x86_loadsplat_1x4_acc4)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_5x5s2p2__wasmsimd_x86_loadsplat_1x4_acc5)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_5x5s2p2__wasmsimd_x86_loadsplat_2x4)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_5x5s2p2__wasmsimd_x86_loadsplat_2x4_acc2)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_5x5s2p2__wasmsimd_x86_loadsplat_2x4_acc3)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_5x5s2p2__wasmsimd_x86_loadsplat_3x4)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_5x5s2p2__wasmsimd_x86_loadsplat_3x4_acc2)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_5x5s2p2__wasmsimd_x86_splat_1x4)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_5x5s2p2__wasmsimd_x86_splat_1x4_acc2)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_5x5s2p2__wasmsimd_x86_splat_1x4_acc3)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_5x5s2p2__wasmsimd_x86_splat_1x4_acc4)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_5x5s2p2__wasmsimd_x86_splat_1x4_acc5)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_5x5s2p2__wasmsimd_x86_splat_2x4)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_5x5s2p2__wasmsimd_x86_splat_2x4_acc2)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_5x5s2p2__wasmsimd_x86_splat_2x4_acc3)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_5x5s2p2__wasmsimd_x86_splat_3x4)
DECLARE_F32_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f32_dwconv2d_chw_ukernel_5x5s2p2__wasmsimd_x86_splat_3x4_acc2)


#define DECLARE_F16_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(fn_name) \
  XNN_INTERNAL void fn_name(                                      \
    size_t input_height,                                          \
    size_t input_width,                                           \
    const void* input,                                            \
    const void* weights,                                          \
    const void* zero,                                             \
    void* output,                                                 \
    uint32_t padding_top,                                         \
    const union xnn_f16_chw_params* params);

DECLARE_F16_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f16_dwconv2d_chw_ukernel_3x3p1__neonfp16arith_1x8)
DECLARE_F16_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f16_dwconv2d_chw_ukernel_3x3p1__neonfp16arith_1x8_acc2)
DECLARE_F16_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f16_dwconv2d_chw_ukernel_3x3p1__neonfp16arith_1x8_acc3)
DECLARE_F16_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f16_dwconv2d_chw_ukernel_3x3p1__neonfp16arith_1x8_acc4)
DECLARE_F16_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f16_dwconv2d_chw_ukernel_3x3p1__neonfp16arith_2x8)
DECLARE_F16_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f16_dwconv2d_chw_ukernel_3x3p1__neonfp16arith_2x8_acc2)
DECLARE_F16_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f16_dwconv2d_chw_ukernel_3x3p1__neonfp16arith_3x8)
DECLARE_F16_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f16_dwconv2d_chw_ukernel_3x3p1__neonfp16arith_4x8)
DECLARE_F16_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f16_dwconv2d_chw_ukernel_3x3p1__neonfp16arith_5x8)
DECLARE_F16_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f16_dwconv2d_chw_ukernel_3x3p1__neonfp16arith_6x8)

DECLARE_F16_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f16_dwconv2d_chw_ukernel_3x3s2p1__neonfp16arith_1x4)
DECLARE_F16_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f16_dwconv2d_chw_ukernel_3x3s2p1__neonfp16arith_1x4_acc2)
DECLARE_F16_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f16_dwconv2d_chw_ukernel_3x3s2p1__neonfp16arith_1x4_acc3)
DECLARE_F16_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f16_dwconv2d_chw_ukernel_3x3s2p1__neonfp16arith_1x4_acc4)
DECLARE_F16_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f16_dwconv2d_chw_ukernel_3x3s2p1__neonfp16arith_2x4)
DECLARE_F16_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f16_dwconv2d_chw_ukernel_3x3s2p1__neonfp16arith_2x4_acc2)
DECLARE_F16_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f16_dwconv2d_chw_ukernel_3x3s2p1__neonfp16arith_3x4)
DECLARE_F16_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f16_dwconv2d_chw_ukernel_3x3s2p1__neonfp16arith_4x4)

DECLARE_F16_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f16_dwconv2d_chw_ukernel_5x5p2__neonfp16arith_1x4)
DECLARE_F16_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f16_dwconv2d_chw_ukernel_5x5p2__neonfp16arith_1x4_acc2)
DECLARE_F16_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f16_dwconv2d_chw_ukernel_5x5p2__neonfp16arith_1x4_acc3)
DECLARE_F16_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f16_dwconv2d_chw_ukernel_5x5p2__neonfp16arith_1x4_acc4)
DECLARE_F16_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f16_dwconv2d_chw_ukernel_5x5p2__neonfp16arith_1x4_acc5)
DECLARE_F16_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f16_dwconv2d_chw_ukernel_5x5p2__neonfp16arith_2x4)
DECLARE_F16_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f16_dwconv2d_chw_ukernel_5x5p2__neonfp16arith_2x4_acc2)
DECLARE_F16_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f16_dwconv2d_chw_ukernel_5x5p2__neonfp16arith_2x4_acc3)
DECLARE_F16_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f16_dwconv2d_chw_ukernel_5x5p2__neonfp16arith_3x4)
DECLARE_F16_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f16_dwconv2d_chw_ukernel_5x5p2__neonfp16arith_3x4_acc2)
DECLARE_F16_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f16_dwconv2d_chw_ukernel_5x5p2__neonfp16arith_4x4)
DECLARE_F16_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f16_dwconv2d_chw_ukernel_5x5p2__neonfp16arith_4x4_acc2)
DECLARE_F16_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f16_dwconv2d_chw_ukernel_5x5p2__neonfp16arith_5x4)

DECLARE_F16_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f16_dwconv2d_chw_ukernel_5x5s2p2__neonfp16arith_1x4)
DECLARE_F16_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f16_dwconv2d_chw_ukernel_5x5s2p2__neonfp16arith_1x4_acc2)
DECLARE_F16_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f16_dwconv2d_chw_ukernel_5x5s2p2__neonfp16arith_1x4_acc3)
DECLARE_F16_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f16_dwconv2d_chw_ukernel_5x5s2p2__neonfp16arith_1x4_acc4)
DECLARE_F16_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f16_dwconv2d_chw_ukernel_5x5s2p2__neonfp16arith_1x4_acc5)
DECLARE_F16_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f16_dwconv2d_chw_ukernel_5x5s2p2__neonfp16arith_2x4)
DECLARE_F16_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f16_dwconv2d_chw_ukernel_5x5s2p2__neonfp16arith_2x4_acc2)
DECLARE_F16_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f16_dwconv2d_chw_ukernel_5x5s2p2__neonfp16arith_2x4_acc3)
DECLARE_F16_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f16_dwconv2d_chw_ukernel_5x5s2p2__neonfp16arith_3x4)
DECLARE_F16_DWCONV2D_CHW_MINMAX_UKERNEL_FUNCTION(xnn_f16_dwconv2d_chw_ukernel_5x5s2p2__neonfp16arith_3x4_acc2)


#ifdef __cplusplus
}  // extern "C"
#endif
