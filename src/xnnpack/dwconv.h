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
    size_t input_offset,                                     \
    const float* zero,                                       \
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
    size_t input_offset,                                            \
    const float* zero,                                              \
    const union xnn_f32_minmax_params* params);

DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up4x4__neon)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up4x4__neon_acc2)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up8x4__neon)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up8x4__neon_acc2)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up4x4__neonfma)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up4x4__neonfma_acc2)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up8x4__neonfma)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up8x4__neonfma_acc2)
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

DECLARE_F32_DWCONV_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_ukernel_up4x4__wasmsimd)
DECLARE_F32_DWCONV_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_ukernel_up8x4__wasmsimd)

DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up4x4__wasmsimd_arm)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up4x4__wasmsimd_acc2_arm)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up8x4__wasmsimd_arm)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up8x4__wasmsimd_acc2_arm)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up4x4__wasmsimd_x86)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up4x4__wasmsimd_acc2_x86)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up8x4__wasmsimd_x86)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up8x4__wasmsimd_acc2_x86)

DECLARE_F32_DWCONV_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_ukernel_up4x9__wasmsimd)
DECLARE_F32_DWCONV_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_ukernel_up8x9__wasmsimd)

DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up4x9__wasmsimd_arm)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up4x9__wasmsimd_acc2_arm)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up8x9__wasmsimd_arm)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up8x9__wasmsimd_acc2_arm)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up4x9__wasmsimd_x86)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up4x9__wasmsimd_acc2_x86)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up8x9__wasmsimd_x86)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up8x9__wasmsimd_acc2_x86)

DECLARE_F32_DWCONV_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_ukernel_up4x25__wasmsimd)
DECLARE_F32_DWCONV_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_ukernel_up8x25__wasmsimd)

DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up4x25__wasmsimd_arm)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up4x25__wasmsimd_acc2_arm)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up8x25__wasmsimd_arm)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up8x25__wasmsimd_acc2_arm)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up4x25__wasmsimd_x86)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up4x25__wasmsimd_acc2_x86)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up8x25__wasmsimd_x86)
DECLARE_F32_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f32_dwconv_minmax_ukernel_up8x25__wasmsimd_acc2_x86)

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

#define DECLARE_F16_DWCONV_UNIPASS_UKERNEL_FUNCTION(fn_name) \
  XNN_INTERNAL void fn_name(                                 \
    size_t channels,                                         \
    size_t output_width,                                     \
    const void** input,                                      \
    const void* weights,                                     \
    void* output,                                            \
    size_t input_stride,                                     \
    size_t output_increment,                                 \
    size_t input_offset,                                     \
    const void* zero,                                        \
    const struct xnn_f16_default_params* params);

#define DECLARE_F16_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(fn_name) \
  XNN_INTERNAL void fn_name(                                        \
    size_t channels,                                                \
    size_t output_width,                                            \
    const void** input,                                             \
    const void* weights,                                            \
    void* output,                                                   \
    size_t input_stride,                                            \
    size_t output_increment,                                        \
    size_t input_offset,                                            \
    const void* zero,                                               \
    const struct xnn_f16_minmax_params* params);

DECLARE_F16_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f16_dwconv_minmax_ukernel_up8x4__neonfp16arith)
DECLARE_F16_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f16_dwconv_minmax_ukernel_up8x4__neonfp16arith_acc2)
DECLARE_F16_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f16_dwconv_minmax_ukernel_up16x4__neonfp16arith)
DECLARE_F16_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f16_dwconv_minmax_ukernel_up16x4__neonfp16arith_acc2)
DECLARE_F16_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f16_dwconv_minmax_ukernel_up8x9__neonfp16arith)
DECLARE_F16_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f16_dwconv_minmax_ukernel_up8x9__neonfp16arith_acc2)
DECLARE_F16_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f16_dwconv_minmax_ukernel_up16x9__neonfp16arith)
DECLARE_F16_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f16_dwconv_minmax_ukernel_up16x9__neonfp16arith_acc2)
DECLARE_F16_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f16_dwconv_minmax_ukernel_up8x25__neonfp16arith)
DECLARE_F16_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f16_dwconv_minmax_ukernel_up8x25__neonfp16arith_acc2)
DECLARE_F16_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f16_dwconv_minmax_ukernel_up16x25__neonfp16arith)
DECLARE_F16_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_f16_dwconv_minmax_ukernel_up16x25__neonfp16arith_acc2)

#define DECLARE_QU8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(fn_name) \
  XNN_INTERNAL void fn_name(                                        \
    size_t channels,                                                \
    size_t output_width,                                            \
    const uint8_t** input,                                          \
    const void* weights,                                            \
    uint8_t* output,                                                \
    size_t input_stride,                                            \
    size_t output_increment,                                        \
    size_t input_offset,                                            \
    const uint8_t* zero,                                            \
    const union xnn_qu8_gemm_params* params);

DECLARE_QU8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qu8_dwconv_minmax_ukernel_up1x9__scalar)
DECLARE_QU8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qu8_dwconv_minmax_ukernel_up8x9__neon)
DECLARE_QU8_DWCONV_MINMAX_UNIPASS_UKERNEL_FUNCTION(xnn_qu8_dwconv_minmax_ukernel_up8x9__sse2)


#define DECLARE_F32_DWCONV_CHW_UKERNEL_FUNCTION(fn_name) \
  XNN_INTERNAL void fn_name(                             \
    size_t input_height,                                 \
    size_t input_width,                                  \
    const float* input,                                  \
    const float* weights,                                \
    const float* zero,                                   \
    float* output,                                       \
    uint32_t padding_top,                                \
    size_t input_tuple_stride,                           \
    size_t output_tuple_stride,                          \
    size_t input_height_stride,                          \
    size_t output_height_stride,                         \
    const union xnn_f32_chw_params* params);

DECLARE_F32_DWCONV_CHW_UKERNEL_FUNCTION(xnn_f32_dwconv_chw_ukernel_3x3p1__scalar)
DECLARE_F32_DWCONV_CHW_UKERNEL_FUNCTION(xnn_f32_dwconv_chw_ukernel_5x5p2__scalar)
DECLARE_F32_DWCONV_CHW_UKERNEL_FUNCTION(xnn_f32_dwconv_chw_ukernel_3x3s2p1__scalar)
DECLARE_F32_DWCONV_CHW_UKERNEL_FUNCTION(xnn_f32_dwconv_chw_ukernel_5x5s2p2__scalar)
DECLARE_F32_DWCONV_CHW_UKERNEL_FUNCTION(xnn_f32_dwconv_chw_ukernel_3x3p1__psimd)
DECLARE_F32_DWCONV_CHW_UKERNEL_FUNCTION(xnn_f32_dwconv_chw_ukernel_3x3s2p1__psimd)
DECLARE_F32_DWCONV_CHW_UKERNEL_FUNCTION(xnn_f32_dwconv_chw_ukernel_5x5p2__psimd)
DECLARE_F32_DWCONV_CHW_UKERNEL_FUNCTION(xnn_f32_dwconv_chw_ukernel_5x5s2p2__psimd)
DECLARE_F32_DWCONV_CHW_UKERNEL_FUNCTION(xnn_f32_dwconv_chw_ukernel_3x3p1__neonfma)
DECLARE_F32_DWCONV_CHW_UKERNEL_FUNCTION(xnn_f32_dwconv_chw_ukernel_3x3s2p1__neonfma)
DECLARE_F32_DWCONV_CHW_UKERNEL_FUNCTION(xnn_f32_dwconv_chw_ukernel_5x5p2__neonfma)
DECLARE_F32_DWCONV_CHW_UKERNEL_FUNCTION(xnn_f32_dwconv_chw_ukernel_5x5s2p2__neonfma)
DECLARE_F32_DWCONV_CHW_UKERNEL_FUNCTION(xnn_f32_dwconv_chw_ukernel_3x3p1__sse)
DECLARE_F32_DWCONV_CHW_UKERNEL_FUNCTION(xnn_f32_dwconv_chw_ukernel_3x3s2p1__sse)


#ifdef __cplusplus
}  // extern "C"
#endif
