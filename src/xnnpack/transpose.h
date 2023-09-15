// Copyright 2021 Google LLC
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

#define DECLARE_XX_TRANSPOSEV_UKERNEL_FUNCTION(fn_name) \
  XNN_INTERNAL void fn_name(                            \
      const void* input,                                \
      void* output,                                     \
      size_t input_row_stride,                          \
      size_t output_row_stride,                         \
      size_t input_element_stride,                      \
      size_t output_element_stride,                     \
      size_t element_size,                              \
      size_t block_width,                               \
      size_t block_height);

DECLARE_XX_TRANSPOSEV_UKERNEL_FUNCTION(xnn_xx_transposev_ukernel__1x1_scalar_memcpy)

#define DECLARE_X64_TRANSPOSEC_UKERNEL_FUNCTION(fn_name) \
  XNN_INTERNAL void fn_name(                             \
      const uint64_t* input,                             \
      uint64_t* output,                                  \
      size_t input_stride,                               \
      size_t output_stride,                              \
      size_t block_width,                                \
      size_t block_height,                               \
      const union xnn_x64_transpose_params params[XNN_RESTRICT XNN_MIN_ELEMENTS(1)]);

DECLARE_X64_TRANSPOSEC_UKERNEL_FUNCTION(xnn_x64_transposec_ukernel__1x2_scalar_float)
DECLARE_X64_TRANSPOSEC_UKERNEL_FUNCTION(xnn_x64_transposec_ukernel__1x2_scalar_int)
DECLARE_X64_TRANSPOSEC_UKERNEL_FUNCTION(xnn_x64_transposec_ukernel__2x1_scalar_float)
DECLARE_X64_TRANSPOSEC_UKERNEL_FUNCTION(xnn_x64_transposec_ukernel__2x1_scalar_int)
DECLARE_X64_TRANSPOSEC_UKERNEL_FUNCTION(xnn_x64_transposec_ukernel__2x2_multi_dec_zip_neon)
DECLARE_X64_TRANSPOSEC_UKERNEL_FUNCTION(xnn_x64_transposec_ukernel__2x2_multi_mov_sse2)
DECLARE_X64_TRANSPOSEC_UKERNEL_FUNCTION(xnn_x64_transposec_ukernel__2x2_multi_mov_zip_neon)
DECLARE_X64_TRANSPOSEC_UKERNEL_FUNCTION(xnn_x64_transposec_ukernel__2x2_multi_multi_sse2)
DECLARE_X64_TRANSPOSEC_UKERNEL_FUNCTION(xnn_x64_transposec_ukernel__2x2_multi_multi_zip_neon)
DECLARE_X64_TRANSPOSEC_UKERNEL_FUNCTION(xnn_x64_transposec_ukernel__2x2_multi_switch_sse2)
DECLARE_X64_TRANSPOSEC_UKERNEL_FUNCTION(xnn_x64_transposec_ukernel__2x2_multi_switch_zip_neon)
DECLARE_X64_TRANSPOSEC_UKERNEL_FUNCTION(xnn_x64_transposec_ukernel__2x2_reuse_dec_zip_neon)
DECLARE_X64_TRANSPOSEC_UKERNEL_FUNCTION(xnn_x64_transposec_ukernel__2x2_reuse_mov_sse2)
DECLARE_X64_TRANSPOSEC_UKERNEL_FUNCTION(xnn_x64_transposec_ukernel__2x2_reuse_mov_zip_neon)
DECLARE_X64_TRANSPOSEC_UKERNEL_FUNCTION(xnn_x64_transposec_ukernel__2x2_reuse_multi_sse2)
DECLARE_X64_TRANSPOSEC_UKERNEL_FUNCTION(xnn_x64_transposec_ukernel__2x2_reuse_multi_zip_neon)
DECLARE_X64_TRANSPOSEC_UKERNEL_FUNCTION(xnn_x64_transposec_ukernel__2x2_reuse_switch_sse2)
DECLARE_X64_TRANSPOSEC_UKERNEL_FUNCTION(xnn_x64_transposec_ukernel__2x2_reuse_switch_zip_neon)
DECLARE_X64_TRANSPOSEC_UKERNEL_FUNCTION(xnn_x64_transposec_ukernel__2x2_scalar_float)
DECLARE_X64_TRANSPOSEC_UKERNEL_FUNCTION(xnn_x64_transposec_ukernel__2x2_scalar_int)
DECLARE_X64_TRANSPOSEC_UKERNEL_FUNCTION(xnn_x64_transposec_ukernel__4x1_scalar_float)
DECLARE_X64_TRANSPOSEC_UKERNEL_FUNCTION(xnn_x64_transposec_ukernel__4x1_scalar_int)
DECLARE_X64_TRANSPOSEC_UKERNEL_FUNCTION(xnn_x64_transposec_ukernel__4x2_scalar_float)
DECLARE_X64_TRANSPOSEC_UKERNEL_FUNCTION(xnn_x64_transposec_ukernel__4x2_scalar_int)
DECLARE_X64_TRANSPOSEC_UKERNEL_FUNCTION(xnn_x64_transposec_ukernel__4x4_multi_mov_avx)
DECLARE_X64_TRANSPOSEC_UKERNEL_FUNCTION(xnn_x64_transposec_ukernel__4x4_multi_multi_avx)
DECLARE_X64_TRANSPOSEC_UKERNEL_FUNCTION(xnn_x64_transposec_ukernel__4x4_multi_switch_avx)
DECLARE_X64_TRANSPOSEC_UKERNEL_FUNCTION(xnn_x64_transposec_ukernel__4x4_reuse_mov_avx)
DECLARE_X64_TRANSPOSEC_UKERNEL_FUNCTION(xnn_x64_transposec_ukernel__4x4_reuse_multi_avx)
DECLARE_X64_TRANSPOSEC_UKERNEL_FUNCTION(xnn_x64_transposec_ukernel__4x4_reuse_switch_avx)

#define DECLARE_X32_TRANSPOSEC_UKERNEL_FUNCTION(fn_name) \
  XNN_INTERNAL void fn_name(                             \
      const uint32_t* input,                             \
      uint32_t* output,                                  \
      size_t input_stride,                               \
      size_t output_stride,                              \
      size_t block_width,                                \
      size_t block_height,                               \
      const union xnn_x32_transpose_params params[XNN_RESTRICT XNN_MIN_ELEMENTS(1)]);

DECLARE_X32_TRANSPOSEC_UKERNEL_FUNCTION(xnn_x32_transposec_ukernel__1x2_scalar_float)
DECLARE_X32_TRANSPOSEC_UKERNEL_FUNCTION(xnn_x32_transposec_ukernel__1x2_scalar_int)
DECLARE_X32_TRANSPOSEC_UKERNEL_FUNCTION(xnn_x32_transposec_ukernel__1x4_scalar_float)
DECLARE_X32_TRANSPOSEC_UKERNEL_FUNCTION(xnn_x32_transposec_ukernel__1x4_scalar_int)
DECLARE_X32_TRANSPOSEC_UKERNEL_FUNCTION(xnn_x32_transposec_ukernel__2x1_scalar_float)
DECLARE_X32_TRANSPOSEC_UKERNEL_FUNCTION(xnn_x32_transposec_ukernel__2x1_scalar_int)
DECLARE_X32_TRANSPOSEC_UKERNEL_FUNCTION(xnn_x32_transposec_ukernel__2x2_multi_dec_zip_neon)
DECLARE_X32_TRANSPOSEC_UKERNEL_FUNCTION(xnn_x32_transposec_ukernel__2x2_multi_mov_zip_neon)
DECLARE_X32_TRANSPOSEC_UKERNEL_FUNCTION(xnn_x32_transposec_ukernel__2x2_multi_multi_zip_neon)
DECLARE_X32_TRANSPOSEC_UKERNEL_FUNCTION(xnn_x32_transposec_ukernel__2x2_multi_switch_zip_neon)
DECLARE_X32_TRANSPOSEC_UKERNEL_FUNCTION(xnn_x32_transposec_ukernel__2x2_reuse_dec_zip_neon)
DECLARE_X32_TRANSPOSEC_UKERNEL_FUNCTION(xnn_x32_transposec_ukernel__2x2_reuse_mov_zip_neon)
DECLARE_X32_TRANSPOSEC_UKERNEL_FUNCTION(xnn_x32_transposec_ukernel__2x2_reuse_multi_zip_neon)
DECLARE_X32_TRANSPOSEC_UKERNEL_FUNCTION(xnn_x32_transposec_ukernel__2x2_reuse_switch_zip_neon)
DECLARE_X32_TRANSPOSEC_UKERNEL_FUNCTION(xnn_x32_transposec_ukernel__2x2_scalar_float)
DECLARE_X32_TRANSPOSEC_UKERNEL_FUNCTION(xnn_x32_transposec_ukernel__2x2_scalar_int)
DECLARE_X32_TRANSPOSEC_UKERNEL_FUNCTION(xnn_x32_transposec_ukernel__2x4_scalar_float)
DECLARE_X32_TRANSPOSEC_UKERNEL_FUNCTION(xnn_x32_transposec_ukernel__2x4_scalar_int)
DECLARE_X32_TRANSPOSEC_UKERNEL_FUNCTION(xnn_x32_transposec_ukernel__4x1_scalar_float)
DECLARE_X32_TRANSPOSEC_UKERNEL_FUNCTION(xnn_x32_transposec_ukernel__4x1_scalar_int)
DECLARE_X32_TRANSPOSEC_UKERNEL_FUNCTION(xnn_x32_transposec_ukernel__4x2_scalar_float)
DECLARE_X32_TRANSPOSEC_UKERNEL_FUNCTION(xnn_x32_transposec_ukernel__4x2_scalar_int)
DECLARE_X32_TRANSPOSEC_UKERNEL_FUNCTION(xnn_x32_transposec_ukernel__4x4_aarch64_neon_tbl128)
DECLARE_X32_TRANSPOSEC_UKERNEL_FUNCTION(xnn_x32_transposec_ukernel__4x4_multi_dec_zip_neon)
DECLARE_X32_TRANSPOSEC_UKERNEL_FUNCTION(xnn_x32_transposec_ukernel__4x4_multi_mov_sse2)
DECLARE_X32_TRANSPOSEC_UKERNEL_FUNCTION(xnn_x32_transposec_ukernel__4x4_multi_mov_wasmsimd)
DECLARE_X32_TRANSPOSEC_UKERNEL_FUNCTION(xnn_x32_transposec_ukernel__4x4_multi_mov_zip_neon)
DECLARE_X32_TRANSPOSEC_UKERNEL_FUNCTION(xnn_x32_transposec_ukernel__4x4_multi_multi_sse2)
DECLARE_X32_TRANSPOSEC_UKERNEL_FUNCTION(xnn_x32_transposec_ukernel__4x4_multi_multi_wasmsimd)
DECLARE_X32_TRANSPOSEC_UKERNEL_FUNCTION(xnn_x32_transposec_ukernel__4x4_multi_multi_zip_neon)
DECLARE_X32_TRANSPOSEC_UKERNEL_FUNCTION(xnn_x32_transposec_ukernel__4x4_multi_switch_sse2)
DECLARE_X32_TRANSPOSEC_UKERNEL_FUNCTION(xnn_x32_transposec_ukernel__4x4_multi_switch_wasmsimd)
DECLARE_X32_TRANSPOSEC_UKERNEL_FUNCTION(xnn_x32_transposec_ukernel__4x4_multi_switch_zip_neon)
DECLARE_X32_TRANSPOSEC_UKERNEL_FUNCTION(xnn_x32_transposec_ukernel__4x4_reuse_dec_zip_neon)
DECLARE_X32_TRANSPOSEC_UKERNEL_FUNCTION(xnn_x32_transposec_ukernel__4x4_reuse_mov_sse2)
DECLARE_X32_TRANSPOSEC_UKERNEL_FUNCTION(xnn_x32_transposec_ukernel__4x4_reuse_mov_wasmsimd)
DECLARE_X32_TRANSPOSEC_UKERNEL_FUNCTION(xnn_x32_transposec_ukernel__4x4_reuse_mov_zip_neon)
DECLARE_X32_TRANSPOSEC_UKERNEL_FUNCTION(xnn_x32_transposec_ukernel__4x4_reuse_multi_sse2)
DECLARE_X32_TRANSPOSEC_UKERNEL_FUNCTION(xnn_x32_transposec_ukernel__4x4_reuse_multi_wasmsimd)
DECLARE_X32_TRANSPOSEC_UKERNEL_FUNCTION(xnn_x32_transposec_ukernel__4x4_reuse_multi_zip_neon)
DECLARE_X32_TRANSPOSEC_UKERNEL_FUNCTION(xnn_x32_transposec_ukernel__4x4_reuse_switch_sse2)
DECLARE_X32_TRANSPOSEC_UKERNEL_FUNCTION(xnn_x32_transposec_ukernel__4x4_reuse_switch_wasmsimd)
DECLARE_X32_TRANSPOSEC_UKERNEL_FUNCTION(xnn_x32_transposec_ukernel__4x4_reuse_switch_zip_neon)
DECLARE_X32_TRANSPOSEC_UKERNEL_FUNCTION(xnn_x32_transposec_ukernel__4x4_scalar_float)
DECLARE_X32_TRANSPOSEC_UKERNEL_FUNCTION(xnn_x32_transposec_ukernel__4x4_scalar_int)
DECLARE_X32_TRANSPOSEC_UKERNEL_FUNCTION(xnn_x32_transposec_ukernel__4x4_sse)
DECLARE_X32_TRANSPOSEC_UKERNEL_FUNCTION(xnn_x32_transposec_ukernel__8x8_multi_mov_avx)
DECLARE_X32_TRANSPOSEC_UKERNEL_FUNCTION(xnn_x32_transposec_ukernel__8x8_multi_switch_avx)
DECLARE_X32_TRANSPOSEC_UKERNEL_FUNCTION(xnn_x32_transposec_ukernel__8x8_reuse_mov_avx)
DECLARE_X32_TRANSPOSEC_UKERNEL_FUNCTION(xnn_x32_transposec_ukernel__8x8_reuse_multi_avx)
DECLARE_X32_TRANSPOSEC_UKERNEL_FUNCTION(xnn_x32_transposec_ukernel__8x8_reuse_switch_avx)

#define DECLARE_X24_TRANSPOSEC_UKERNEL_FUNCTION(fn_name) \
  XNN_INTERNAL void fn_name(                             \
      const void* input,                                 \
      void* output,                                      \
      size_t input_stride,                               \
      size_t output_stride,                              \
      size_t block_width,                                \
      size_t block_height,                               \
      const union xnn_x24_transpose_params params[XNN_RESTRICT XNN_MIN_ELEMENTS(1)]);

DECLARE_X24_TRANSPOSEC_UKERNEL_FUNCTION(xnn_x24_transposec_ukernel__1x2_scalar)
DECLARE_X24_TRANSPOSEC_UKERNEL_FUNCTION(xnn_x24_transposec_ukernel__1x4_scalar)
DECLARE_X24_TRANSPOSEC_UKERNEL_FUNCTION(xnn_x24_transposec_ukernel__2x1_scalar)
DECLARE_X24_TRANSPOSEC_UKERNEL_FUNCTION(xnn_x24_transposec_ukernel__2x2_neon_tbl64)
DECLARE_X24_TRANSPOSEC_UKERNEL_FUNCTION(xnn_x24_transposec_ukernel__2x2_scalar)
DECLARE_X24_TRANSPOSEC_UKERNEL_FUNCTION(xnn_x24_transposec_ukernel__2x4_scalar)
DECLARE_X24_TRANSPOSEC_UKERNEL_FUNCTION(xnn_x24_transposec_ukernel__4x1_scalar)
DECLARE_X24_TRANSPOSEC_UKERNEL_FUNCTION(xnn_x24_transposec_ukernel__4x2_scalar)
DECLARE_X24_TRANSPOSEC_UKERNEL_FUNCTION(xnn_x24_transposec_ukernel__4x4_aarch64_neon_tbl128)
DECLARE_X24_TRANSPOSEC_UKERNEL_FUNCTION(xnn_x24_transposec_ukernel__4x4_scalar)
DECLARE_X24_TRANSPOSEC_UKERNEL_FUNCTION(xnn_x24_transposec_ukernel__4x4_ssse3)

#define DECLARE_X16_TRANSPOSEC_UKERNEL_FUNCTION(fn_name) \
  XNN_INTERNAL void fn_name(                             \
      const uint16_t* input,                             \
      uint16_t* output,                                  \
      size_t input_stride,                               \
      size_t output_stride,                              \
      size_t block_width,                                \
      size_t block_height,                               \
      const union xnn_x16_transpose_params params[XNN_RESTRICT XNN_MIN_ELEMENTS(1)]);

DECLARE_X16_TRANSPOSEC_UKERNEL_FUNCTION(xnn_x16_transposec_ukernel__1x2_scalar_int)
DECLARE_X16_TRANSPOSEC_UKERNEL_FUNCTION(xnn_x16_transposec_ukernel__1x4_scalar_int)
DECLARE_X16_TRANSPOSEC_UKERNEL_FUNCTION(xnn_x16_transposec_ukernel__2x1_scalar_int)
DECLARE_X16_TRANSPOSEC_UKERNEL_FUNCTION(xnn_x16_transposec_ukernel__2x2_scalar_int)
DECLARE_X16_TRANSPOSEC_UKERNEL_FUNCTION(xnn_x16_transposec_ukernel__2x4_scalar_int)
DECLARE_X16_TRANSPOSEC_UKERNEL_FUNCTION(xnn_x16_transposec_ukernel__4x1_scalar_int)
DECLARE_X16_TRANSPOSEC_UKERNEL_FUNCTION(xnn_x16_transposec_ukernel__4x2_scalar_int)
DECLARE_X16_TRANSPOSEC_UKERNEL_FUNCTION(xnn_x16_transposec_ukernel__4x4_multi_dec_zip_neon)
DECLARE_X16_TRANSPOSEC_UKERNEL_FUNCTION(xnn_x16_transposec_ukernel__4x4_multi_mov_zip_neon)
DECLARE_X16_TRANSPOSEC_UKERNEL_FUNCTION(xnn_x16_transposec_ukernel__4x4_multi_multi_zip_neon)
DECLARE_X16_TRANSPOSEC_UKERNEL_FUNCTION(xnn_x16_transposec_ukernel__4x4_multi_switch_zip_neon)
DECLARE_X16_TRANSPOSEC_UKERNEL_FUNCTION(xnn_x16_transposec_ukernel__4x4_reuse_dec_zip_neon)
DECLARE_X16_TRANSPOSEC_UKERNEL_FUNCTION(xnn_x16_transposec_ukernel__4x4_reuse_mov_zip_neon)
DECLARE_X16_TRANSPOSEC_UKERNEL_FUNCTION(xnn_x16_transposec_ukernel__4x4_reuse_multi_zip_neon)
DECLARE_X16_TRANSPOSEC_UKERNEL_FUNCTION(xnn_x16_transposec_ukernel__4x4_reuse_switch_zip_neon)
DECLARE_X16_TRANSPOSEC_UKERNEL_FUNCTION(xnn_x16_transposec_ukernel__4x4_scalar_int)
DECLARE_X16_TRANSPOSEC_UKERNEL_FUNCTION(xnn_x16_transposec_ukernel__4x8_sse2)
DECLARE_X16_TRANSPOSEC_UKERNEL_FUNCTION(xnn_x16_transposec_ukernel__8x8_multi_dec_zip_neon)
DECLARE_X16_TRANSPOSEC_UKERNEL_FUNCTION(xnn_x16_transposec_ukernel__8x8_multi_mov_sse2)
DECLARE_X16_TRANSPOSEC_UKERNEL_FUNCTION(xnn_x16_transposec_ukernel__8x8_multi_mov_wasmsimd)
DECLARE_X16_TRANSPOSEC_UKERNEL_FUNCTION(xnn_x16_transposec_ukernel__8x8_multi_mov_zip_neon)
DECLARE_X16_TRANSPOSEC_UKERNEL_FUNCTION(xnn_x16_transposec_ukernel__8x8_multi_switch_sse2)
DECLARE_X16_TRANSPOSEC_UKERNEL_FUNCTION(xnn_x16_transposec_ukernel__8x8_multi_switch_wasmsimd)
DECLARE_X16_TRANSPOSEC_UKERNEL_FUNCTION(xnn_x16_transposec_ukernel__8x8_multi_switch_zip_neon)
DECLARE_X16_TRANSPOSEC_UKERNEL_FUNCTION(xnn_x16_transposec_ukernel__8x8_reuse_dec_zip_neon)
DECLARE_X16_TRANSPOSEC_UKERNEL_FUNCTION(xnn_x16_transposec_ukernel__8x8_reuse_mov_sse2)
DECLARE_X16_TRANSPOSEC_UKERNEL_FUNCTION(xnn_x16_transposec_ukernel__8x8_reuse_mov_wasmsimd)
DECLARE_X16_TRANSPOSEC_UKERNEL_FUNCTION(xnn_x16_transposec_ukernel__8x8_reuse_mov_zip_neon)
DECLARE_X16_TRANSPOSEC_UKERNEL_FUNCTION(xnn_x16_transposec_ukernel__8x8_reuse_multi_sse2)
DECLARE_X16_TRANSPOSEC_UKERNEL_FUNCTION(xnn_x16_transposec_ukernel__8x8_reuse_multi_wasmsimd)
DECLARE_X16_TRANSPOSEC_UKERNEL_FUNCTION(xnn_x16_transposec_ukernel__8x8_reuse_multi_zip_neon)
DECLARE_X16_TRANSPOSEC_UKERNEL_FUNCTION(xnn_x16_transposec_ukernel__8x8_reuse_switch_sse2)
DECLARE_X16_TRANSPOSEC_UKERNEL_FUNCTION(xnn_x16_transposec_ukernel__8x8_reuse_switch_wasmsimd)
DECLARE_X16_TRANSPOSEC_UKERNEL_FUNCTION(xnn_x16_transposec_ukernel__8x8_reuse_switch_zip_neon)
DECLARE_X16_TRANSPOSEC_UKERNEL_FUNCTION(xnn_x16_transposec_ukernel__16x16_reuse_mov_avx2)
DECLARE_X16_TRANSPOSEC_UKERNEL_FUNCTION(xnn_x16_transposec_ukernel__16x16_reuse_switch_avx2)

#define DECLARE_X8_TRANSPOSEC_UKERNEL_FUNCTION(fn_name) \
  XNN_INTERNAL void fn_name(                            \
      const uint8_t* input,                             \
      uint8_t* output,                                  \
      size_t input_stride,                              \
      size_t output_stride,                             \
      size_t block_width,                               \
      size_t block_height,                              \
      const union xnn_x8_transpose_params params[XNN_RESTRICT XNN_MIN_ELEMENTS(1)]);

DECLARE_X8_TRANSPOSEC_UKERNEL_FUNCTION(xnn_x8_transposec_ukernel__1x2_scalar_int)
DECLARE_X8_TRANSPOSEC_UKERNEL_FUNCTION(xnn_x8_transposec_ukernel__1x4_scalar_int)
DECLARE_X8_TRANSPOSEC_UKERNEL_FUNCTION(xnn_x8_transposec_ukernel__2x1_scalar_int)
DECLARE_X8_TRANSPOSEC_UKERNEL_FUNCTION(xnn_x8_transposec_ukernel__2x2_scalar_int)
DECLARE_X8_TRANSPOSEC_UKERNEL_FUNCTION(xnn_x8_transposec_ukernel__2x4_scalar_int)
DECLARE_X8_TRANSPOSEC_UKERNEL_FUNCTION(xnn_x8_transposec_ukernel__4x1_scalar_int)
DECLARE_X8_TRANSPOSEC_UKERNEL_FUNCTION(xnn_x8_transposec_ukernel__4x2_scalar_int)
DECLARE_X8_TRANSPOSEC_UKERNEL_FUNCTION(xnn_x8_transposec_ukernel__4x4_scalar_int)
DECLARE_X8_TRANSPOSEC_UKERNEL_FUNCTION(xnn_x8_transposec_ukernel__8x8_multi_dec_zip_neon)
DECLARE_X8_TRANSPOSEC_UKERNEL_FUNCTION(xnn_x8_transposec_ukernel__8x8_multi_mov_zip_neon)
DECLARE_X8_TRANSPOSEC_UKERNEL_FUNCTION(xnn_x8_transposec_ukernel__8x8_multi_switch_zip_neon)
DECLARE_X8_TRANSPOSEC_UKERNEL_FUNCTION(xnn_x8_transposec_ukernel__8x8_reuse_dec_zip_neon)
DECLARE_X8_TRANSPOSEC_UKERNEL_FUNCTION(xnn_x8_transposec_ukernel__8x8_reuse_mov_zip_neon)
DECLARE_X8_TRANSPOSEC_UKERNEL_FUNCTION(xnn_x8_transposec_ukernel__8x8_reuse_multi_zip_neon)
DECLARE_X8_TRANSPOSEC_UKERNEL_FUNCTION(xnn_x8_transposec_ukernel__8x8_reuse_switch_zip_neon)
DECLARE_X8_TRANSPOSEC_UKERNEL_FUNCTION(xnn_x8_transposec_ukernel__16x16_reuse_dec_zip_neon)
DECLARE_X8_TRANSPOSEC_UKERNEL_FUNCTION(xnn_x8_transposec_ukernel__16x16_reuse_mov_sse2)
DECLARE_X8_TRANSPOSEC_UKERNEL_FUNCTION(xnn_x8_transposec_ukernel__16x16_reuse_mov_wasmsimd)
DECLARE_X8_TRANSPOSEC_UKERNEL_FUNCTION(xnn_x8_transposec_ukernel__16x16_reuse_mov_zip_neon)
DECLARE_X8_TRANSPOSEC_UKERNEL_FUNCTION(xnn_x8_transposec_ukernel__16x16_reuse_switch_sse2)
DECLARE_X8_TRANSPOSEC_UKERNEL_FUNCTION(xnn_x8_transposec_ukernel__16x16_reuse_switch_wasmsimd)
DECLARE_X8_TRANSPOSEC_UKERNEL_FUNCTION(xnn_x8_transposec_ukernel__16x16_reuse_switch_zip_neon)
DECLARE_X8_TRANSPOSEC_UKERNEL_FUNCTION(xnn_x8_transposec_ukernel__32x32_reuse_mov_avx2)
DECLARE_X8_TRANSPOSEC_UKERNEL_FUNCTION(xnn_x8_transposec_ukernel__32x32_reuse_switch_avx2)

#ifdef __cplusplus
}  // extern "C"
#endif
