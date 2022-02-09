// Copyright 2021 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <stddef.h>
#include <stdint.h>
#include <xnnpack/common.h>
#include <xnnpack/params.h>

#ifdef __cplusplus
extern "C" {
#endif

#define DECLARE_X64_TRANSPOSE_UKERNEL_FUNCTION(fn_name) \
  XNN_INTERNAL void fn_name(const uint64_t* input,      \
                            uint64_t* output,           \
                            size_t input_stride,        \
                            size_t output_stride,       \
                            size_t block_width,         \
                            size_t block_height);

DECLARE_X64_TRANSPOSE_UKERNEL_FUNCTION(xnn_x64_transpose_ukernel__1x2_scalar_float)
DECLARE_X64_TRANSPOSE_UKERNEL_FUNCTION(xnn_x64_transpose_ukernel__1x2_scalar_int)
DECLARE_X64_TRANSPOSE_UKERNEL_FUNCTION(xnn_x64_transpose_ukernel__2x1_scalar_float)
DECLARE_X64_TRANSPOSE_UKERNEL_FUNCTION(xnn_x64_transpose_ukernel__2x1_scalar_int)
DECLARE_X64_TRANSPOSE_UKERNEL_FUNCTION(xnn_x64_transpose_ukernel__2x2_multi_mov_sse2)
DECLARE_X64_TRANSPOSE_UKERNEL_FUNCTION(xnn_x64_transpose_ukernel__2x2_multi_multi_sse2)
DECLARE_X64_TRANSPOSE_UKERNEL_FUNCTION(xnn_x64_transpose_ukernel__2x2_multi_switch_sse2)
DECLARE_X64_TRANSPOSE_UKERNEL_FUNCTION(xnn_x64_transpose_ukernel__2x2_reuse_mov_sse2)
DECLARE_X64_TRANSPOSE_UKERNEL_FUNCTION(xnn_x64_transpose_ukernel__2x2_reuse_multi_sse2)
DECLARE_X64_TRANSPOSE_UKERNEL_FUNCTION(xnn_x64_transpose_ukernel__2x2_reuse_switch_sse2)
DECLARE_X64_TRANSPOSE_UKERNEL_FUNCTION(xnn_x64_transpose_ukernel__2x2_scalar_float)
DECLARE_X64_TRANSPOSE_UKERNEL_FUNCTION(xnn_x64_transpose_ukernel__2x2_scalar_int)
DECLARE_X64_TRANSPOSE_UKERNEL_FUNCTION(xnn_x64_transpose_ukernel__4x1_scalar_float)
DECLARE_X64_TRANSPOSE_UKERNEL_FUNCTION(xnn_x64_transpose_ukernel__4x1_scalar_int)
DECLARE_X64_TRANSPOSE_UKERNEL_FUNCTION(xnn_x64_transpose_ukernel__4x2_scalar_float)
DECLARE_X64_TRANSPOSE_UKERNEL_FUNCTION(xnn_x64_transpose_ukernel__4x2_scalar_int)

#define DECLARE_X32_TRANSPOSE_UKERNEL_FUNCTION(fn_name) \
  XNN_INTERNAL void fn_name(const uint32_t* input,      \
                            uint32_t* output,           \
                            size_t input_stride,        \
                            size_t output_stride,       \
                            size_t block_width,         \
                            size_t block_height);

DECLARE_X32_TRANSPOSE_UKERNEL_FUNCTION(xnn_x32_transpose_ukernel__1x2_scalar_float)
DECLARE_X32_TRANSPOSE_UKERNEL_FUNCTION(xnn_x32_transpose_ukernel__1x2_scalar_int)
DECLARE_X32_TRANSPOSE_UKERNEL_FUNCTION(xnn_x32_transpose_ukernel__1x4_scalar_float)
DECLARE_X32_TRANSPOSE_UKERNEL_FUNCTION(xnn_x32_transpose_ukernel__1x4_scalar_int)
DECLARE_X32_TRANSPOSE_UKERNEL_FUNCTION(xnn_x32_transpose_ukernel__2x1_scalar_float)
DECLARE_X32_TRANSPOSE_UKERNEL_FUNCTION(xnn_x32_transpose_ukernel__2x1_scalar_int)
DECLARE_X32_TRANSPOSE_UKERNEL_FUNCTION(xnn_x32_transpose_ukernel__2x2_scalar_float)
DECLARE_X32_TRANSPOSE_UKERNEL_FUNCTION(xnn_x32_transpose_ukernel__2x2_scalar_int)
DECLARE_X32_TRANSPOSE_UKERNEL_FUNCTION(xnn_x32_transpose_ukernel__2x4_scalar_float)
DECLARE_X32_TRANSPOSE_UKERNEL_FUNCTION(xnn_x32_transpose_ukernel__2x4_scalar_int)
DECLARE_X32_TRANSPOSE_UKERNEL_FUNCTION(xnn_x32_transpose_ukernel__4x1_scalar_float)
DECLARE_X32_TRANSPOSE_UKERNEL_FUNCTION(xnn_x32_transpose_ukernel__4x1_scalar_int)
DECLARE_X32_TRANSPOSE_UKERNEL_FUNCTION(xnn_x32_transpose_ukernel__4x2_scalar_float)
DECLARE_X32_TRANSPOSE_UKERNEL_FUNCTION(xnn_x32_transpose_ukernel__4x2_scalar_int)
DECLARE_X32_TRANSPOSE_UKERNEL_FUNCTION(xnn_x32_transpose_ukernel__4x4_aarch64_neon_tbl)
DECLARE_X32_TRANSPOSE_UKERNEL_FUNCTION(xnn_x32_transpose_ukernel__4x4_multi_dec_zip_neon)
DECLARE_X32_TRANSPOSE_UKERNEL_FUNCTION(xnn_x32_transpose_ukernel__4x4_multi_mov_sse2)
DECLARE_X32_TRANSPOSE_UKERNEL_FUNCTION(xnn_x32_transpose_ukernel__4x4_multi_mov_zip_neon)
DECLARE_X32_TRANSPOSE_UKERNEL_FUNCTION(xnn_x32_transpose_ukernel__4x4_multi_multi_sse2)
DECLARE_X32_TRANSPOSE_UKERNEL_FUNCTION(xnn_x32_transpose_ukernel__4x4_multi_multi_zip_neon)
DECLARE_X32_TRANSPOSE_UKERNEL_FUNCTION(xnn_x32_transpose_ukernel__4x4_multi_switch_sse2)
DECLARE_X32_TRANSPOSE_UKERNEL_FUNCTION(xnn_x32_transpose_ukernel__4x4_multi_switch_zip_neon)
DECLARE_X32_TRANSPOSE_UKERNEL_FUNCTION(xnn_x32_transpose_ukernel__4x4_reuse_dec_zip_neon)
DECLARE_X32_TRANSPOSE_UKERNEL_FUNCTION(xnn_x32_transpose_ukernel__4x4_reuse_mov_sse2)
DECLARE_X32_TRANSPOSE_UKERNEL_FUNCTION(xnn_x32_transpose_ukernel__4x4_reuse_mov_zip_neon)
DECLARE_X32_TRANSPOSE_UKERNEL_FUNCTION(xnn_x32_transpose_ukernel__4x4_reuse_multi_sse2)
DECLARE_X32_TRANSPOSE_UKERNEL_FUNCTION(xnn_x32_transpose_ukernel__4x4_reuse_multi_zip_neon)
DECLARE_X32_TRANSPOSE_UKERNEL_FUNCTION(xnn_x32_transpose_ukernel__4x4_reuse_switch_sse2)
DECLARE_X32_TRANSPOSE_UKERNEL_FUNCTION(xnn_x32_transpose_ukernel__4x4_reuse_switch_zip_neon)
DECLARE_X32_TRANSPOSE_UKERNEL_FUNCTION(xnn_x32_transpose_ukernel__4x4_scalar_float)
DECLARE_X32_TRANSPOSE_UKERNEL_FUNCTION(xnn_x32_transpose_ukernel__4x4_scalar_int)
DECLARE_X32_TRANSPOSE_UKERNEL_FUNCTION(xnn_x32_transpose_ukernel__4x4_sse)
DECLARE_X32_TRANSPOSE_UKERNEL_FUNCTION(xnn_x32_transpose_ukernel__4x4_wasmsimd)

#define DECLARE_X16_TRANSPOSE_UKERNEL_FUNCTION(fn_name) \
  XNN_INTERNAL void fn_name(const uint16_t* input,      \
                            uint16_t* output,           \
                            size_t input_stride,        \
                            size_t output_stride,       \
                            size_t block_width,         \
                            size_t block_height);

DECLARE_X16_TRANSPOSE_UKERNEL_FUNCTION(xnn_x16_transpose_ukernel__1x2_scalar_int)
DECLARE_X16_TRANSPOSE_UKERNEL_FUNCTION(xnn_x16_transpose_ukernel__1x4_scalar_int)
DECLARE_X16_TRANSPOSE_UKERNEL_FUNCTION(xnn_x16_transpose_ukernel__2x1_scalar_int)
DECLARE_X16_TRANSPOSE_UKERNEL_FUNCTION(xnn_x16_transpose_ukernel__2x2_scalar_int)
DECLARE_X16_TRANSPOSE_UKERNEL_FUNCTION(xnn_x16_transpose_ukernel__2x4_scalar_int)
DECLARE_X16_TRANSPOSE_UKERNEL_FUNCTION(xnn_x16_transpose_ukernel__4x1_scalar_int)
DECLARE_X16_TRANSPOSE_UKERNEL_FUNCTION(xnn_x16_transpose_ukernel__4x2_scalar_int)
DECLARE_X16_TRANSPOSE_UKERNEL_FUNCTION(xnn_x16_transpose_ukernel__4x4_scalar_int)
DECLARE_X16_TRANSPOSE_UKERNEL_FUNCTION(xnn_x16_transpose_ukernel__4x8_sse2)
DECLARE_X16_TRANSPOSE_UKERNEL_FUNCTION(xnn_x16_transpose_ukernel__8x8_multi_dec_zip_neon)
DECLARE_X16_TRANSPOSE_UKERNEL_FUNCTION(xnn_x16_transpose_ukernel__8x8_multi_mov_sse2)
DECLARE_X16_TRANSPOSE_UKERNEL_FUNCTION(xnn_x16_transpose_ukernel__8x8_multi_mov_zip_neon)
DECLARE_X16_TRANSPOSE_UKERNEL_FUNCTION(xnn_x16_transpose_ukernel__8x8_multi_switch_sse2)
DECLARE_X16_TRANSPOSE_UKERNEL_FUNCTION(xnn_x16_transpose_ukernel__8x8_multi_switch_zip_neon)
DECLARE_X16_TRANSPOSE_UKERNEL_FUNCTION(xnn_x16_transpose_ukernel__8x8_reuse_dec_zip_neon)
DECLARE_X16_TRANSPOSE_UKERNEL_FUNCTION(xnn_x16_transpose_ukernel__8x8_reuse_mov_sse2)
DECLARE_X16_TRANSPOSE_UKERNEL_FUNCTION(xnn_x16_transpose_ukernel__8x8_reuse_mov_zip_neon)
DECLARE_X16_TRANSPOSE_UKERNEL_FUNCTION(xnn_x16_transpose_ukernel__8x8_reuse_multi_sse2)
DECLARE_X16_TRANSPOSE_UKERNEL_FUNCTION(xnn_x16_transpose_ukernel__8x8_reuse_multi_zip_neon)
DECLARE_X16_TRANSPOSE_UKERNEL_FUNCTION(xnn_x16_transpose_ukernel__8x8_reuse_switch_sse2)
DECLARE_X16_TRANSPOSE_UKERNEL_FUNCTION(xnn_x16_transpose_ukernel__8x8_reuse_switch_zip_neon)

#define DECLARE_X8_TRANSPOSE_UKERNEL_FUNCTION(fn_name)  \
  XNN_INTERNAL void fn_name(const uint8_t* input,       \
                            uint8_t* output,            \
                            size_t input_stride,        \
                            size_t output_stride,       \
                            size_t block_width,         \
                            size_t block_height);

DECLARE_X8_TRANSPOSE_UKERNEL_FUNCTION(xnn_x8_transpose_ukernel__1x2_scalar_int)
DECLARE_X8_TRANSPOSE_UKERNEL_FUNCTION(xnn_x8_transpose_ukernel__1x4_scalar_int)
DECLARE_X8_TRANSPOSE_UKERNEL_FUNCTION(xnn_x8_transpose_ukernel__2x1_scalar_int)
DECLARE_X8_TRANSPOSE_UKERNEL_FUNCTION(xnn_x8_transpose_ukernel__2x2_scalar_int)
DECLARE_X8_TRANSPOSE_UKERNEL_FUNCTION(xnn_x8_transpose_ukernel__2x4_scalar_int)
DECLARE_X8_TRANSPOSE_UKERNEL_FUNCTION(xnn_x8_transpose_ukernel__4x1_scalar_int)
DECLARE_X8_TRANSPOSE_UKERNEL_FUNCTION(xnn_x8_transpose_ukernel__4x2_scalar_int)
DECLARE_X8_TRANSPOSE_UKERNEL_FUNCTION(xnn_x8_transpose_ukernel__4x4_scalar_int)
DECLARE_X8_TRANSPOSE_UKERNEL_FUNCTION(xnn_x8_transpose_ukernel__16x16_reuse_dec_zip_neon)
DECLARE_X8_TRANSPOSE_UKERNEL_FUNCTION(xnn_x8_transpose_ukernel__16x16_reuse_mov_sse2)
DECLARE_X8_TRANSPOSE_UKERNEL_FUNCTION(xnn_x8_transpose_ukernel__16x16_reuse_mov_zip_neon)
DECLARE_X8_TRANSPOSE_UKERNEL_FUNCTION(xnn_x8_transpose_ukernel__16x16_reuse_switch_sse2)
DECLARE_X8_TRANSPOSE_UKERNEL_FUNCTION(xnn_x8_transpose_ukernel__16x16_reuse_switch_zip_neon)

#ifdef __cplusplus
}  // extern "C"
#endif
