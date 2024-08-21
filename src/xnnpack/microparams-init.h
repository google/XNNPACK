// Copyright 2021 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <stddef.h>
#include <stdint.h>

#include "xnnpack/common.h"
#include "xnnpack/microparams.h"


#ifdef __cplusplus
extern "C" {
#endif


#define DECLARE_INIT_QS8_QC8W_CONV_MINMAX_PARAMS_FUNCTION(fn_name)     \
  XNN_INTERNAL size_t fn_name(                                         \
    union xnn_qs8_qc8w_conv_minmax_params params[XNN_MIN_ELEMENTS(1)], \
    int8_t output_zero_point,                                          \
    int8_t output_min,                                                 \
    int8_t output_max);

DECLARE_INIT_QS8_QC8W_CONV_MINMAX_PARAMS_FUNCTION(xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_fmagic_params)
DECLARE_INIT_QS8_QC8W_CONV_MINMAX_PARAMS_FUNCTION(xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_imagic_params)
DECLARE_INIT_QS8_QC8W_CONV_MINMAX_PARAMS_FUNCTION(xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_lrintf_params)
#if XNN_ARCH_ARM
  DECLARE_INIT_QS8_QC8W_CONV_MINMAX_PARAMS_FUNCTION(xnn_init_qs8_qc8w_conv_minmax_fp32_armsimd32_params)
#endif  // XNN_ARCH_ARM
#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  DECLARE_INIT_QS8_QC8W_CONV_MINMAX_PARAMS_FUNCTION(xnn_init_qs8_qc8w_conv_minmax_fp32_neon_params)
  DECLARE_INIT_QS8_QC8W_CONV_MINMAX_PARAMS_FUNCTION(xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64
#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  DECLARE_INIT_QS8_QC8W_CONV_MINMAX_PARAMS_FUNCTION(xnn_init_qs8_qc8w_conv_minmax_fp32_sse2_params)
  DECLARE_INIT_QS8_QC8W_CONV_MINMAX_PARAMS_FUNCTION(xnn_init_qs8_qc8w_conv_minmax_fp32_sse4_params)
  DECLARE_INIT_QS8_QC8W_CONV_MINMAX_PARAMS_FUNCTION(xnn_init_qs8_qc8w_conv_minmax_fp32_avx2_params)
  DECLARE_INIT_QS8_QC8W_CONV_MINMAX_PARAMS_FUNCTION(xnn_init_qs8_qc8w_conv_minmax_fp32_avx512_params)
  DECLARE_INIT_QS8_QC8W_CONV_MINMAX_PARAMS_FUNCTION(xnn_init_qs8_qc8w_conv_minmax_fp32_avx512vnni_params)
  DECLARE_INIT_QS8_QC8W_CONV_MINMAX_PARAMS_FUNCTION(xnn_init_qs8_qc8w_conv_minmax_fp32_avxvnni_params)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64
#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  DECLARE_INIT_QS8_QC8W_CONV_MINMAX_PARAMS_FUNCTION(xnn_init_qs8_qc8w_conv_minmax_fp32_wasmsimd_params)
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#define DECLARE_INIT_QS8_CONV_MINMAX_PARAMS_FUNCTION(fn_name)     \
  XNN_INTERNAL size_t fn_name(                                    \
    union xnn_qs8_conv_minmax_params params[XNN_MIN_ELEMENTS(1)], \
    float scale,                                                  \
    int8_t output_zero_point,                                     \
    int8_t output_min,                                            \
    int8_t output_max);

DECLARE_INIT_QS8_CONV_MINMAX_PARAMS_FUNCTION(xnn_init_qs8_conv_minmax_fp32_scalar_fmagic_params)
DECLARE_INIT_QS8_CONV_MINMAX_PARAMS_FUNCTION(xnn_init_qs8_conv_minmax_fp32_scalar_imagic_params)
DECLARE_INIT_QS8_CONV_MINMAX_PARAMS_FUNCTION(xnn_init_qs8_conv_minmax_fp32_scalar_lrintf_params)
DECLARE_INIT_QS8_CONV_MINMAX_PARAMS_FUNCTION(xnn_init_qs8_conv_minmax_rndnu_scalar_params)
#if XNN_ARCH_ARM
  DECLARE_INIT_QS8_CONV_MINMAX_PARAMS_FUNCTION(xnn_init_qs8_conv_minmax_fp32_armsimd32_params)
#endif  // XNN_ARCH_ARM
#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  DECLARE_INIT_QS8_CONV_MINMAX_PARAMS_FUNCTION(xnn_init_qs8_conv_minmax_fp32_neon_params)
  DECLARE_INIT_QS8_CONV_MINMAX_PARAMS_FUNCTION(xnn_init_qs8_conv_minmax_fp32_neonv8_params)
  DECLARE_INIT_QS8_CONV_MINMAX_PARAMS_FUNCTION(xnn_init_qs8_conv_minmax_rndnu_neon_params)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64
#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  DECLARE_INIT_QS8_CONV_MINMAX_PARAMS_FUNCTION(xnn_init_qs8_conv_minmax_fp32_sse2_params)
  DECLARE_INIT_QS8_CONV_MINMAX_PARAMS_FUNCTION(xnn_init_qs8_conv_minmax_fp32_sse4_params)
  DECLARE_INIT_QS8_CONV_MINMAX_PARAMS_FUNCTION(xnn_init_qs8_conv_minmax_fp32_avx2_params)
  DECLARE_INIT_QS8_CONV_MINMAX_PARAMS_FUNCTION(xnn_init_qs8_conv_minmax_fp32_avx512_params)
  DECLARE_INIT_QS8_CONV_MINMAX_PARAMS_FUNCTION(xnn_init_qs8_conv_minmax_fp32_avx512vnni_params)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64
#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  DECLARE_INIT_QS8_CONV_MINMAX_PARAMS_FUNCTION(xnn_init_qs8_conv_minmax_fp32_wasmsimd_params)
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#define DECLARE_INIT_QU8_CONV_MINMAX_PARAMS_FUNCTION(fn_name)     \
  XNN_INTERNAL size_t fn_name(                                    \
    union xnn_qu8_conv_minmax_params params[XNN_MIN_ELEMENTS(1)], \
    uint8_t kernel_zero_point,                                    \
    float scale,                                                  \
    uint8_t output_zero_point,                                    \
    uint8_t output_min,                                           \
    uint8_t output_max);

DECLARE_INIT_QU8_CONV_MINMAX_PARAMS_FUNCTION(xnn_init_qu8_conv_minmax_fp32_scalar_fmagic_params)
DECLARE_INIT_QU8_CONV_MINMAX_PARAMS_FUNCTION(xnn_init_qu8_conv_minmax_fp32_scalar_imagic_params)
DECLARE_INIT_QU8_CONV_MINMAX_PARAMS_FUNCTION(xnn_init_qu8_conv_minmax_fp32_scalar_lrintf_params)
DECLARE_INIT_QU8_CONV_MINMAX_PARAMS_FUNCTION(xnn_init_qu8_conv_minmax_rndnu_scalar_params)
#if XNN_ARCH_ARM
  DECLARE_INIT_QU8_CONV_MINMAX_PARAMS_FUNCTION(xnn_init_qu8_conv_minmax_fp32_armsimd32_params)
#endif  // XNN_ARCH_ARM
#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  DECLARE_INIT_QU8_CONV_MINMAX_PARAMS_FUNCTION(xnn_init_qu8_conv_minmax_fp32_neon_params)
  DECLARE_INIT_QU8_CONV_MINMAX_PARAMS_FUNCTION(xnn_init_qu8_conv_minmax_fp32_neonv8_params)
  DECLARE_INIT_QU8_CONV_MINMAX_PARAMS_FUNCTION(xnn_init_qu8_conv_minmax_rndnu_neon_params)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64
#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  DECLARE_INIT_QU8_CONV_MINMAX_PARAMS_FUNCTION(xnn_init_qu8_conv_minmax_fp32_sse2_params)
  DECLARE_INIT_QU8_CONV_MINMAX_PARAMS_FUNCTION(xnn_init_qu8_conv_minmax_fp32_avx2_params)
  DECLARE_INIT_QU8_CONV_MINMAX_PARAMS_FUNCTION(xnn_init_qu8_conv_minmax_fp32_avx512_params)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64
#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  DECLARE_INIT_QU8_CONV_MINMAX_PARAMS_FUNCTION(xnn_init_qu8_conv_minmax_fp32_wasmsimd_params)
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


XNN_INTERNAL void xnn_init_qs8_qc8w_scale_fp32_params(
  size_t channels,
  size_t channels_tile,
  size_t channels_subtile,
  size_t stride,
  size_t substride,
  // How much offset to subtract from packed_w pointer when moving from channels_tile to channels_subtile.
  size_t stride_offset,
  const float scale[XNN_MIN_ELEMENTS(1)],
  void* packed_w);

XNN_INTERNAL void xnn_init_blockwise_scale_fp32_params(
  size_t channels,
  size_t channels_tile,
  size_t channels_subtile,
  size_t stride,
  size_t substride,
  size_t num_blocks,
  size_t block_stride,
  // How much offset to subtract from packed_w pointer when moving from channels_tile to channels_subtile.
  size_t stride_offset,
  const float scale[XNN_MIN_ELEMENTS(1)],
  void* packed_w);

XNN_INTERNAL void xnn_init_blockwise_scale_bf16_params(
  size_t channels,
  size_t channels_tile,
  size_t channels_subtile,
  size_t stride,
  size_t substride,
  size_t num_blocks,
  size_t block_stride,
  // How much offset to subtract from packed_w pointer when moving from channels_tile to channels_subtile.
  size_t stride_offset,
  const uint16_t scale[XNN_MIN_ELEMENTS(1)],
  void* packed_w);

XNN_INTERNAL void xnn_init_qs8_to_qs8_qc8w_scale_fp32_params(
  size_t channels,
  size_t channels_tile,
  size_t channels_subtile,
  size_t stride,
  size_t substride,
  // How much offset to subtract from packed_w pointer when moving from channels_tile to channels_subtile.
  size_t stride_offset,
  const float scale[XNN_MIN_ELEMENTS(1)],
  void* packed_w);

#define DECLARE_INIT_QS8_AVGPOOL_PARAMS_FUNCTION(fn_name)            \
  XNN_INTERNAL size_t fn_name(                                       \
    union xnn_qs8_avgpool_minmax_params params[XNN_MIN_ELEMENTS(1)], \
    int32_t bias,                                                    \
    float scale,                                                     \
    int8_t output_zero_point,                                        \
    int8_t output_min,                                               \
    int8_t output_max);

DECLARE_INIT_QS8_AVGPOOL_PARAMS_FUNCTION(xnn_init_qs8_avgpool_minmax_fp32_scalar_fmagic_params)
DECLARE_INIT_QS8_AVGPOOL_PARAMS_FUNCTION(xnn_init_qs8_avgpool_minmax_fp32_scalar_imagic_params)
DECLARE_INIT_QS8_AVGPOOL_PARAMS_FUNCTION(xnn_init_qs8_avgpool_minmax_fp32_scalar_lrintf_params)
#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  DECLARE_INIT_QS8_AVGPOOL_PARAMS_FUNCTION(xnn_init_qs8_avgpool_minmax_fp32_neon_params)
  DECLARE_INIT_QS8_AVGPOOL_PARAMS_FUNCTION(xnn_init_qs8_avgpool_minmax_fp32_neonv8_params)
  DECLARE_INIT_QS8_AVGPOOL_PARAMS_FUNCTION(xnn_init_qs8_avgpool_minmax_rndnu_neon_params)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64
#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  DECLARE_INIT_QS8_AVGPOOL_PARAMS_FUNCTION(xnn_init_qs8_avgpool_minmax_fp32_sse2_params)
  DECLARE_INIT_QS8_AVGPOOL_PARAMS_FUNCTION(xnn_init_qs8_avgpool_minmax_fp32_ssse3_params)
  DECLARE_INIT_QS8_AVGPOOL_PARAMS_FUNCTION(xnn_init_qs8_avgpool_minmax_fp32_sse4_params)
  DECLARE_INIT_QS8_AVGPOOL_PARAMS_FUNCTION(xnn_init_qs8_avgpool_minmax_fp32_avx2_params)
  DECLARE_INIT_QS8_AVGPOOL_PARAMS_FUNCTION(xnn_init_qs8_avgpool_minmax_fp32_avx512_params)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64
#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  DECLARE_INIT_QS8_AVGPOOL_PARAMS_FUNCTION(xnn_init_qs8_avgpool_minmax_fp32_wasmsimd_params)
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#define DECLARE_UPDATE_QS8_AVGPOOL_PARAMS_FUNCTION(fn_name)          \
  XNN_INTERNAL void fn_name(                                         \
    union xnn_qs8_avgpool_minmax_params params[XNN_MIN_ELEMENTS(1)], \
    int32_t bias,                                                    \
    float scale);

DECLARE_UPDATE_QS8_AVGPOOL_PARAMS_FUNCTION(xnn_update_qs8_avgpool_minmax_fp32_scalar_fmagic_params)
DECLARE_UPDATE_QS8_AVGPOOL_PARAMS_FUNCTION(xnn_update_qs8_avgpool_minmax_fp32_scalar_imagic_params)
DECLARE_UPDATE_QS8_AVGPOOL_PARAMS_FUNCTION(xnn_update_qs8_avgpool_minmax_fp32_scalar_lrintf_params)
#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  DECLARE_UPDATE_QS8_AVGPOOL_PARAMS_FUNCTION(xnn_update_qs8_avgpool_minmax_fp32_neon_params)
  DECLARE_UPDATE_QS8_AVGPOOL_PARAMS_FUNCTION(xnn_update_qs8_avgpool_minmax_fp32_neonv8_params)
  DECLARE_UPDATE_QS8_AVGPOOL_PARAMS_FUNCTION(xnn_update_qs8_avgpool_minmax_rndnu_neon_params)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64
#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  DECLARE_UPDATE_QS8_AVGPOOL_PARAMS_FUNCTION(xnn_update_qs8_avgpool_minmax_fp32_sse2_params)
  DECLARE_UPDATE_QS8_AVGPOOL_PARAMS_FUNCTION(xnn_update_qs8_avgpool_minmax_fp32_sse4_params)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64
#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  DECLARE_UPDATE_QS8_AVGPOOL_PARAMS_FUNCTION(xnn_update_qs8_avgpool_minmax_fp32_wasmsimd_params)
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#define DECLARE_INIT_QU8_AVGPOOL_PARAMS_FUNCTION(fn_name)            \
  XNN_INTERNAL size_t fn_name(                                       \
    union xnn_qu8_avgpool_minmax_params params[XNN_MIN_ELEMENTS(1)], \
    int32_t bias,                                                    \
    float scale,                                                     \
    uint8_t output_zero_point,                                       \
    uint8_t output_min,                                              \
    uint8_t output_max);

DECLARE_INIT_QU8_AVGPOOL_PARAMS_FUNCTION(xnn_init_qu8_avgpool_minmax_fp32_scalar_fmagic_params)
DECLARE_INIT_QU8_AVGPOOL_PARAMS_FUNCTION(xnn_init_qu8_avgpool_minmax_fp32_scalar_imagic_params)
DECLARE_INIT_QU8_AVGPOOL_PARAMS_FUNCTION(xnn_init_qu8_avgpool_minmax_fp32_scalar_lrintf_params)
#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  DECLARE_INIT_QU8_AVGPOOL_PARAMS_FUNCTION(xnn_init_qu8_avgpool_minmax_fp32_neon_params)
  DECLARE_INIT_QU8_AVGPOOL_PARAMS_FUNCTION(xnn_init_qu8_avgpool_minmax_fp32_neonv8_params)
  DECLARE_INIT_QU8_AVGPOOL_PARAMS_FUNCTION(xnn_init_qu8_avgpool_minmax_rndnu_neon_params)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64
#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  DECLARE_INIT_QU8_AVGPOOL_PARAMS_FUNCTION(xnn_init_qu8_avgpool_minmax_fp32_sse2_params)
  DECLARE_INIT_QU8_AVGPOOL_PARAMS_FUNCTION(xnn_init_qu8_avgpool_minmax_fp32_sse4_params)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64
#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  DECLARE_INIT_QU8_AVGPOOL_PARAMS_FUNCTION(xnn_init_qu8_avgpool_minmax_fp32_wasmsimd_params)
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#define DECLARE_UPDATE_QU8_AVGPOOL_PARAMS_FUNCTION(fn_name)          \
  XNN_INTERNAL void fn_name(                                         \
    union xnn_qu8_avgpool_minmax_params params[XNN_MIN_ELEMENTS(1)], \
    int32_t bias,                                                    \
    float scale);

DECLARE_UPDATE_QU8_AVGPOOL_PARAMS_FUNCTION(xnn_update_qu8_avgpool_minmax_fp32_scalar_fmagic_params)
DECLARE_UPDATE_QU8_AVGPOOL_PARAMS_FUNCTION(xnn_update_qu8_avgpool_minmax_fp32_scalar_imagic_params)
DECLARE_UPDATE_QU8_AVGPOOL_PARAMS_FUNCTION(xnn_update_qu8_avgpool_minmax_fp32_scalar_lrintf_params)
#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  DECLARE_UPDATE_QU8_AVGPOOL_PARAMS_FUNCTION(xnn_update_qu8_avgpool_minmax_fp32_neon_params)
  DECLARE_UPDATE_QU8_AVGPOOL_PARAMS_FUNCTION(xnn_update_qu8_avgpool_minmax_fp32_neonv8_params)
  DECLARE_UPDATE_QU8_AVGPOOL_PARAMS_FUNCTION(xnn_update_qu8_avgpool_minmax_rndnu_neon_params)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64
#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  DECLARE_UPDATE_QU8_AVGPOOL_PARAMS_FUNCTION(xnn_update_qu8_avgpool_minmax_fp32_sse2_params)
  DECLARE_UPDATE_QU8_AVGPOOL_PARAMS_FUNCTION(xnn_update_qu8_avgpool_minmax_fp32_sse4_params)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64
#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  DECLARE_UPDATE_QU8_AVGPOOL_PARAMS_FUNCTION(xnn_update_qu8_avgpool_minmax_fp32_wasmsimd_params)
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#define DECLARE_INIT_F16_SCALE_PARAMS_FUNCTION(fn_name)     \
  XNN_INTERNAL size_t fn_name(                              \
    union xnn_f16_scale_params params[XNN_MIN_ELEMENTS(1)], \
    uint16_t scale);

  DECLARE_INIT_F16_SCALE_PARAMS_FUNCTION(xnn_init_f16_scale_fp16arith_params)


#define DECLARE_INIT_F16_F32ACC_SCALE_PARAMS_FUNCTION(fn_name)     \
  XNN_INTERNAL size_t fn_name(                                     \
    union xnn_f16_f32acc_scale_params params[XNN_MIN_ELEMENTS(1)], \
    float scale);

DECLARE_INIT_F16_F32ACC_SCALE_PARAMS_FUNCTION(xnn_init_f16_f32acc_scale_scalar_params)


#define DECLARE_INIT_F32_SCALE_PARAMS_FUNCTION(fn_name)     \
  XNN_INTERNAL size_t fn_name(                              \
    union xnn_f32_scale_params params[XNN_MIN_ELEMENTS(1)], \
    float scale);

DECLARE_INIT_F32_SCALE_PARAMS_FUNCTION(xnn_init_f32_scale_scalar_params)


#define DECLARE_INIT_F16_SCALEMINMAX_PARAMS_FUNCTION(fn_name)     \
  XNN_INTERNAL size_t fn_name(                                    \
    union xnn_f16_scaleminmax_params params[XNN_MIN_ELEMENTS(1)], \
    uint16_t scale,                                               \
    uint16_t min,                                                 \
    uint16_t max);

#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  DECLARE_INIT_F16_SCALEMINMAX_PARAMS_FUNCTION(xnn_init_f16_scaleminmax_fp16arith_params)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64
#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  DECLARE_INIT_F16_SCALEMINMAX_PARAMS_FUNCTION(xnn_init_f16_scaleminmax_avx_params)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#define DECLARE_UPDATE_F16_SCALEMINMAX_PARAMS_FUNCTION(fn_name)   \
  XNN_INTERNAL void fn_name(                                      \
    union xnn_f16_scaleminmax_params params[XNN_MIN_ELEMENTS(1)], \
    uint16_t scale);

#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  DECLARE_UPDATE_F16_SCALEMINMAX_PARAMS_FUNCTION(xnn_update_f16_scaleminmax_fp16arith_params)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64
#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  DECLARE_UPDATE_F16_SCALEMINMAX_PARAMS_FUNCTION(xnn_update_f16_scaleminmax_avx_params)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#define DECLARE_INIT_F32_SCALEMINMAX_PARAMS_FUNCTION(fn_name)     \
  XNN_INTERNAL size_t fn_name(                                    \
    union xnn_f32_scaleminmax_params params[XNN_MIN_ELEMENTS(1)], \
    float scale,                                                  \
    float min,                                                    \
    float max);

DECLARE_INIT_F32_SCALEMINMAX_PARAMS_FUNCTION(xnn_init_f32_scaleminmax_scalar_params)
#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  DECLARE_INIT_F32_SCALEMINMAX_PARAMS_FUNCTION(xnn_init_f32_scaleminmax_sse_params)
  DECLARE_INIT_F32_SCALEMINMAX_PARAMS_FUNCTION(xnn_init_f32_scaleminmax_avx_params)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#define DECLARE_UPDATE_F32_SCALEMINMAX_PARAMS_FUNCTION(fn_name)   \
  XNN_INTERNAL void fn_name(                                      \
    union xnn_f32_scaleminmax_params params[XNN_MIN_ELEMENTS(1)], \
    float scale);

DECLARE_UPDATE_F32_SCALEMINMAX_PARAMS_FUNCTION(xnn_update_f32_scaleminmax_scalar_params)
#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  DECLARE_UPDATE_F32_SCALEMINMAX_PARAMS_FUNCTION(xnn_update_f32_scaleminmax_sse_params)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


XNN_INTERNAL size_t xnn_init_f16_gavgpool_neonfp16arith_params(
  union xnn_f16_gavgpool_params params[XNN_MIN_ELEMENTS(1)],
  uint16_t multiplier,
  uint16_t output_min,
  uint16_t output_max,
  uint32_t width);

#define DECLARE_INIT_F32_GAVGPOOL_PARAMS_FUNCITON(fn_name)     \
  XNN_INTERNAL size_t fn_name(                                 \
    union xnn_f32_gavgpool_params params[XNN_MIN_ELEMENTS(1)], \
    float multiplier,                                          \
    float output_min,                                          \
    float output_max,                                          \
    uint32_t width);

DECLARE_INIT_F32_GAVGPOOL_PARAMS_FUNCITON(xnn_init_f32_gavgpool_scalar_params);
#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  DECLARE_INIT_F32_GAVGPOOL_PARAMS_FUNCITON(xnn_init_f32_gavgpool_neon_params);
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64
#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  DECLARE_INIT_F32_GAVGPOOL_PARAMS_FUNCITON(xnn_init_f32_gavgpool_sse_params);
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

#if XNN_ARCH_ARM || XNN_ARCH_ARM64
XNN_INTERNAL void xnn_update_f16_gavgpool_neonfp16arith_params(
  union xnn_f16_gavgpool_params params[XNN_MIN_ELEMENTS(1)],
  uint16_t multiplier,
  uint32_t width);
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64

XNN_INTERNAL void xnn_update_f32_gavgpool_params(
  union xnn_f32_gavgpool_params params[XNN_MIN_ELEMENTS(1)],
  float multiplier,
  uint32_t width);

#define DECLARE_INIT_BF16_MINMAX_PARAMS_FUNCTION(fn_name)     \
  XNN_INTERNAL size_t fn_name(                                \
    union xnn_bf16_minmax_params params[XNN_MIN_ELEMENTS(1)], \
    uint16_t output_min,                                      \
    uint16_t output_max);

DECLARE_INIT_BF16_MINMAX_PARAMS_FUNCTION(xnn_init_bf16_minmax_scalar_params)


#define DECLARE_INIT_F16_MINMAX_PARAMS_FUNCTION(fn_name)     \
  XNN_INTERNAL size_t fn_name(                               \
    union xnn_f16_minmax_params params[XNN_MIN_ELEMENTS(1)], \
    uint16_t output_min,                                     \
    uint16_t output_max);

  DECLARE_INIT_F16_MINMAX_PARAMS_FUNCTION(xnn_init_f16_minmax_fp16arith_params)
#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  DECLARE_INIT_F16_MINMAX_PARAMS_FUNCTION(xnn_init_f16_minmax_avx_params)
  DECLARE_INIT_F16_MINMAX_PARAMS_FUNCTION(xnn_init_f16_minmax_avxvnni_params)
  DECLARE_INIT_F16_MINMAX_PARAMS_FUNCTION(xnn_init_f16_minmax_scalar_params)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#define DECLARE_INIT_F32_MINMAX_PARAMS_FUNCTION(fn_name)     \
  XNN_INTERNAL size_t fn_name(                               \
    union xnn_f32_minmax_params params[XNN_MIN_ELEMENTS(1)], \
    float output_min,                                        \
    float output_max);

DECLARE_INIT_F32_MINMAX_PARAMS_FUNCTION(xnn_init_f32_minmax_scalar_params)
#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  DECLARE_INIT_F32_MINMAX_PARAMS_FUNCTION(xnn_init_f32_minmax_sse_params)
  DECLARE_INIT_F32_MINMAX_PARAMS_FUNCTION(xnn_init_f32_minmax_avx_params)
  DECLARE_INIT_F32_MINMAX_PARAMS_FUNCTION(xnn_init_f32_minmax_avx512vnni_params)
  DECLARE_INIT_F32_MINMAX_PARAMS_FUNCTION(xnn_init_f32_minmax_avxvnni_params)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64
#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  DECLARE_INIT_F32_MINMAX_PARAMS_FUNCTION(xnn_init_f32_minmax_wasmsimd_params)
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
#if XNN_ARCH_HEXAGON
  DECLARE_INIT_F32_MINMAX_PARAMS_FUNCTION(xnn_init_f32_minmax_hvx_params)
#endif

#define DECLARE_INIT_F16_QC4W_MINMAX_PARAMS_FUNCTION(fn_name)     \
  XNN_INTERNAL size_t fn_name(                                    \
    union xnn_f16_qc4w_minmax_params params[XNN_MIN_ELEMENTS(1)], \
    uint16_t output_min,                                          \
    uint16_t output_max,                                          \
    uint8_t kernel_zero_point);

DECLARE_INIT_F16_QC4W_MINMAX_PARAMS_FUNCTION(xnn_init_f16_qc4w_minmax_scalar_params)

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  DECLARE_INIT_F16_QC4W_MINMAX_PARAMS_FUNCTION(xnn_init_f16_qc4w_minmax_avx_params)
  DECLARE_INIT_F16_QC4W_MINMAX_PARAMS_FUNCTION(xnn_init_f16_qc4w_minmax_avxvnni_params)
  DECLARE_INIT_F16_QC4W_MINMAX_PARAMS_FUNCTION(xnn_init_f16_qc4w_minmax_avxvnni_madd_params)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#define DECLARE_INIT_F16_QB4W_MINMAX_PARAMS_FUNCTION(fn_name)     \
  XNN_INTERNAL size_t fn_name(                                    \
    union xnn_f16_qb4w_minmax_params params[XNN_MIN_ELEMENTS(1)], \
    uint16_t output_min,                                          \
    uint16_t output_max,                                          \
    uint8_t kernel_zero_point,                                    \
    size_t blocksize);

DECLARE_INIT_F16_QB4W_MINMAX_PARAMS_FUNCTION(xnn_init_f16_qb4w_minmax_scalar_params)

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  DECLARE_INIT_F16_QB4W_MINMAX_PARAMS_FUNCTION(xnn_init_f16_qb4w_minmax_avx_params)
  DECLARE_INIT_F16_QB4W_MINMAX_PARAMS_FUNCTION(xnn_init_f16_qb4w_minmax_avxvnni_params)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#define DECLARE_INIT_F32_QC4W_MINMAX_PARAMS_FUNCTION(fn_name)     \
  XNN_INTERNAL size_t fn_name(                                    \
    union xnn_f32_qc4w_minmax_params params[XNN_MIN_ELEMENTS(1)], \
    float output_min,                                             \
    float output_max,                                             \
    uint8_t kernel_zero_point);

DECLARE_INIT_F32_QC4W_MINMAX_PARAMS_FUNCTION(xnn_init_f32_qc4w_minmax_scalar_params)
#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  DECLARE_INIT_F32_QC4W_MINMAX_PARAMS_FUNCTION(xnn_init_f32_qc4w_minmax_sse_params)
  DECLARE_INIT_F32_QC4W_MINMAX_PARAMS_FUNCTION(xnn_init_f32_qc4w_minmax_avx_params)
  DECLARE_INIT_F32_QC4W_MINMAX_PARAMS_FUNCTION(xnn_init_f32_qc4w_minmax_avx512_params)
  DECLARE_INIT_F32_QC4W_MINMAX_PARAMS_FUNCTION(xnn_init_f32_qc4w_minmax_avx512vnni_params)
  DECLARE_INIT_F32_QC4W_MINMAX_PARAMS_FUNCTION(xnn_init_f32_qc4w_minmax_avx512vnni_madd_params)
  DECLARE_INIT_F32_QC4W_MINMAX_PARAMS_FUNCTION(xnn_init_f32_qc4w_minmax_avxvnni_params)
  DECLARE_INIT_F32_QC4W_MINMAX_PARAMS_FUNCTION(xnn_init_f32_qc4w_minmax_avxvnni_madd_params)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64
#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  DECLARE_INIT_F32_QC4W_MINMAX_PARAMS_FUNCTION(xnn_init_f32_qc4w_minmax_wasmsimd_params)
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD

#define DECLARE_INIT_F32_QB4W_MINMAX_PARAMS_FUNCTION(fn_name)     \
  XNN_INTERNAL size_t fn_name(                                    \
    union xnn_f32_qb4w_minmax_params params[XNN_MIN_ELEMENTS(1)], \
    float output_min,                                             \
    float output_max,                                             \
    uint8_t kernel_zero_point,                                    \
    size_t blocksize);

DECLARE_INIT_F32_QB4W_MINMAX_PARAMS_FUNCTION(xnn_init_f32_qb4w_minmax_scalar_params)
#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  DECLARE_INIT_F32_QB4W_MINMAX_PARAMS_FUNCTION(xnn_init_f32_qb4w_minmax_sse_params)
  DECLARE_INIT_F32_QB4W_MINMAX_PARAMS_FUNCTION(xnn_init_f32_qb4w_minmax_avx_params)
  DECLARE_INIT_F32_QB4W_MINMAX_PARAMS_FUNCTION(xnn_init_f32_qb4w_minmax_avx512_params)
  DECLARE_INIT_F32_QB4W_MINMAX_PARAMS_FUNCTION(xnn_init_f32_qb4w_minmax_avx512vnni_params)
  DECLARE_INIT_F32_QB4W_MINMAX_PARAMS_FUNCTION(xnn_init_f32_qb4w_minmax_avxvnni_params)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64
#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  DECLARE_INIT_F32_QB4W_MINMAX_PARAMS_FUNCTION(xnn_init_f32_qb4w_minmax_wasmsimd_params)
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD

#define DECLARE_INIT_QS8_HSWISH_PARAMS_FUNCTION(fn_name)       \
  XNN_INTERNAL size_t fn_name(                                 \
    union xnn_qs8_hswish_params params[XNN_MIN_ELEMENTS(1)],   \
    int16_t input_zero_point,                                  \
    int16_t output_zero_point,                                 \
    float input_scale,                                         \
    float output_scale);

DECLARE_INIT_QS8_HSWISH_PARAMS_FUNCTION(xnn_init_qs8_hswish_scalar_params)
#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  DECLARE_INIT_QS8_HSWISH_PARAMS_FUNCTION(xnn_init_qs8_hswish_sse2_params)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64
#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  DECLARE_INIT_QS8_HSWISH_PARAMS_FUNCTION(xnn_init_qs8_hswish_scalar_params)
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD

#define DECLARE_INIT_QU8_HSWISH_PARAMS_FUNCTION(fn_name)       \
  XNN_INTERNAL size_t fn_name(                                 \
    union xnn_qu8_hswish_params params[XNN_MIN_ELEMENTS(1)],   \
    int16_t input_zero_point,                                  \
    int16_t output_zero_point,                                 \
    float input_scale,                                         \
    float output_scale);

DECLARE_INIT_QU8_HSWISH_PARAMS_FUNCTION(xnn_init_qu8_hswish_scalar_params)
#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  DECLARE_INIT_QU8_HSWISH_PARAMS_FUNCTION(xnn_init_qu8_hswish_sse2_params)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#define DECLARE_INIT_F16_TANH_PARAMS_FUNCTION(fn_name)      \
  XNN_INTERNAL size_t fn_name(                              \
    union xnn_f16_tanh_params params[XNN_MIN_ELEMENTS(1)]);

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  DECLARE_INIT_F16_TANH_PARAMS_FUNCTION(xnn_init_f16_tanh_avx_expm1minus_rr1_p3h2_params)
  DECLARE_INIT_F16_TANH_PARAMS_FUNCTION(xnn_init_f16_tanh_avx_polynomial_p19h9t2_params)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#define DECLARE_INIT_F32_TANH_PARAMS_FUNCTION(fn_name)      \
  XNN_INTERNAL size_t fn_name(                              \
    union xnn_f32_tanh_params params[XNN_MIN_ELEMENTS(1)]);

DECLARE_INIT_F32_TANH_PARAMS_FUNCTION(xnn_init_f32_tanh_scalar_expm1minus_rr1_lut8_p4h3_params)
DECLARE_INIT_F32_TANH_PARAMS_FUNCTION(xnn_init_f32_tanh_scalar_expm1minus_rr1_p6h5_params)
#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  DECLARE_INIT_F32_TANH_PARAMS_FUNCTION(xnn_init_f32_tanh_wasmsimd_expm1minus_rr1_lut8_p4h3_abs_params)
  DECLARE_INIT_F32_TANH_PARAMS_FUNCTION(xnn_init_f32_tanh_wasmsimd_expm1minus_rr1_p6h5_abs_params)
  DECLARE_INIT_F32_TANH_PARAMS_FUNCTION(xnn_init_f32_tanh_wasmsimd_expm1minus_rr1_lut8_p4h3_nabs_params)
  DECLARE_INIT_F32_TANH_PARAMS_FUNCTION(xnn_init_f32_tanh_wasmsimd_expm1minus_rr1_p6h5_nabs_params)
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  DECLARE_INIT_F32_TANH_PARAMS_FUNCTION(xnn_init_f32_tanh_neon_expm1minus_rr1_lut8_p4h3_params)
  DECLARE_INIT_F32_TANH_PARAMS_FUNCTION(xnn_init_f32_tanh_neon_expm1minus_rr1_p6h5_params)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64

#define DECLARE_INIT_bf16_default_params_FUNCTION(fn_name)      \
  XNN_INTERNAL size_t fn_name(                             \
    union xnn_bf16_default_params params[XNN_MIN_ELEMENTS(1)]);

#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  DECLARE_INIT_bf16_default_params_FUNCTION(xnn_init_bf16_abs_neon_params)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64

#define DECLARE_INIT_F16_ABS_PARAMS_FUNCTION(fn_name)      \
  XNN_INTERNAL size_t fn_name(                             \
    union xnn_f16_default_params params[XNN_MIN_ELEMENTS(1)]);

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  DECLARE_INIT_F16_ABS_PARAMS_FUNCTION(xnn_init_f16_abs_sse_params)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

#define DECLARE_INIT_F16_ELU_PARAMS_FUNCTION(fn_name)     \
  XNN_INTERNAL size_t fn_name(                            \
    union xnn_f16_elu_params params[XNN_MIN_ELEMENTS(1)], \
    uint16_t prescale,                                    \
    uint16_t alpha,                                       \
    uint16_t beta);

DECLARE_INIT_F16_ELU_PARAMS_FUNCTION(xnn_init_f16_elu_scalar_params)
#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  DECLARE_INIT_F16_ELU_PARAMS_FUNCTION(xnn_init_f16_elu_avx2_params)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#define DECLARE_INIT_F32_ELU_PARAMS_FUNCTION(fn_name)     \
  XNN_INTERNAL size_t fn_name(                            \
    union xnn_f32_elu_params params[XNN_MIN_ELEMENTS(1)], \
    float prescale,                                       \
    float alpha,                                          \
    float beta);

DECLARE_INIT_F32_ELU_PARAMS_FUNCTION(xnn_init_f32_elu_scalar_params)


#define DECLARE_INIT_F16_LRELU_PARAMS_FUNCTION(fn_name)     \
  XNN_INTERNAL size_t fn_name(                              \
    union xnn_f16_lrelu_params params[XNN_MIN_ELEMENTS(1)], \
    uint16_t slope);

#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  DECLARE_INIT_F16_LRELU_PARAMS_FUNCTION(xnn_init_f16_lrelu_fp16arith_params)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64
#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  DECLARE_INIT_F16_LRELU_PARAMS_FUNCTION(xnn_init_f16_lrelu_avx_params)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#define DECLARE_INIT_F32_LRELU_PARAMS_FUNCTION(fn_name)     \
  XNN_INTERNAL size_t fn_name(                              \
    union xnn_f32_lrelu_params params[XNN_MIN_ELEMENTS(1)], \
    float slope);

DECLARE_INIT_F32_LRELU_PARAMS_FUNCTION(xnn_init_f32_lrelu_scalar_params)


#define DECLARE_INIT_QS8_LRELU_PARAMS_FUNCTION(fn_name)     \
  XNN_INTERNAL size_t fn_name(                              \
    union xnn_qs8_lrelu_params params[XNN_MIN_ELEMENTS(1)], \
    float positive_scale,                                   \
    float negative_scale,                                   \
    int8_t input_zero_point,                                \
    int8_t output_zero_point);

DECLARE_INIT_QS8_LRELU_PARAMS_FUNCTION(xnn_init_qs8_lrelu_scalar_params)


#define DECLARE_INIT_QU8_LRELU_PARAMS_FUNCTION(fn_name)     \
  XNN_INTERNAL size_t fn_name(                              \
    union xnn_qu8_lrelu_params params[XNN_MIN_ELEMENTS(1)], \
    float positive_scale,                                   \
    float negative_scale,                                   \
    uint8_t input_zero_point,                               \
    uint8_t output_zero_point);

DECLARE_INIT_QU8_LRELU_PARAMS_FUNCTION(xnn_init_qu8_lrelu_scalar_params)


#define DECLARE_INIT_F32_SQRT_PARAMS_FUNCTION(fn_name)      \
  XNN_INTERNAL size_t fn_name(                              \
    union xnn_f32_sqrt_params params[XNN_MIN_ELEMENTS(1)]);

#define DECLARE_INIT_F32_RSQRT_PARAMS_FUNCTION(fn_name)      \
  XNN_INTERNAL size_t fn_name(                              \
    union xnn_f32_rsqrt_params params[XNN_MIN_ELEMENTS(1)]);

#define DECLARE_INIT_F16_CHW_PARAMS_FUNCTION(fn_name)     \
  XNN_INTERNAL size_t fn_name(                            \
    union xnn_f16_chw_params params[XNN_MIN_ELEMENTS(1)], \
    uint32_t width,                                       \
    uint16_t output_min,                                  \
    uint16_t output_max);

#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  DECLARE_INIT_F16_CHW_PARAMS_FUNCTION(xnn_init_f16_chw_neonfp16arith_stride1_params)
  DECLARE_INIT_F16_CHW_PARAMS_FUNCTION(xnn_init_f16_chw_neonfp16arith_stride2_params)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#define DECLARE_INIT_F32_CHW_PARAMS_FUNCTION(fn_name)     \
  XNN_INTERNAL size_t fn_name(                            \
    union xnn_f32_chw_params params[XNN_MIN_ELEMENTS(1)], \
    uint32_t width,                                       \
    float output_min,                                     \
    float output_max);

DECLARE_INIT_F32_CHW_PARAMS_FUNCTION(xnn_init_f32_chw_scalar_params)
#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  DECLARE_INIT_F32_CHW_PARAMS_FUNCTION(xnn_init_f32_chw_neon_stride1_params)
  DECLARE_INIT_F32_CHW_PARAMS_FUNCTION(xnn_init_f32_chw_neon_stride2_params)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64
#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  DECLARE_INIT_F32_CHW_PARAMS_FUNCTION(xnn_init_f32_chw_sse_stride1_params)
  DECLARE_INIT_F32_CHW_PARAMS_FUNCTION(xnn_init_f32_chw_sse_stride2_params)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64
#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  DECLARE_INIT_F32_CHW_PARAMS_FUNCTION(xnn_init_f32_chw_wasmsimd_stride1_params)
  DECLARE_INIT_F32_CHW_PARAMS_FUNCTION(xnn_init_f32_chw_wasmsimd_stride2_params)
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#define DECLARE_UPDATE_F16_CHW_PARAMS_FUNCTION(fn_name)   \
  XNN_INTERNAL void fn_name(                              \
    union xnn_f16_chw_params params[XNN_MIN_ELEMENTS(1)], \
    uint32_t width);

#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  DECLARE_UPDATE_F16_CHW_PARAMS_FUNCTION(xnn_update_f16_chw_neonfp16arith_stride1_params)
  DECLARE_UPDATE_F16_CHW_PARAMS_FUNCTION(xnn_update_f16_chw_neonfp16arith_stride2_params)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#define DECLARE_UPDATE_F32_CHW_PARAMS_FUNCTION(fn_name)   \
  XNN_INTERNAL void fn_name(                              \
    union xnn_f32_chw_params params[XNN_MIN_ELEMENTS(1)], \
    uint32_t width);

#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  DECLARE_UPDATE_F32_CHW_PARAMS_FUNCTION(xnn_update_f32_chw_neon_stride1_params)
  DECLARE_UPDATE_F32_CHW_PARAMS_FUNCTION(xnn_update_f32_chw_neon_stride2_params)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64
#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  DECLARE_UPDATE_F32_CHW_PARAMS_FUNCTION(xnn_update_f32_chw_sse_stride1_params)
  DECLARE_UPDATE_F32_CHW_PARAMS_FUNCTION(xnn_update_f32_chw_sse_stride2_params)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64
#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  DECLARE_UPDATE_F32_CHW_PARAMS_FUNCTION(xnn_update_f32_chw_wasmsimd_stride1_params)
  DECLARE_UPDATE_F32_CHW_PARAMS_FUNCTION(xnn_update_f32_chw_wasmsimd_stride2_params)
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#define DECLARE_INIT_S8_MINMAX_PARAMS_FUNCTION(fn_name)     \
  XNN_INTERNAL size_t fn_name(                              \
    union xnn_s8_minmax_params params[XNN_MIN_ELEMENTS(1)], \
    int8_t output_min,                                      \
    int8_t output_max);

DECLARE_INIT_S8_MINMAX_PARAMS_FUNCTION(xnn_init_s8_minmax_scalar_params)


#define DECLARE_INIT_U8_MINMAX_PARAMS_FUNCTION(fn_name)      \
  XNN_INTERNAL size_t fn_name(                               \
    union xnn_u8_minmax_params params[XNN_MIN_ELEMENTS(1)],  \
    uint8_t output_min,                                      \
    uint8_t output_max);

DECLARE_INIT_U8_MINMAX_PARAMS_FUNCTION(xnn_init_u8_minmax_scalar_params)


#define DECLARE_INIT_QS8_ADD_MINMAX_PARAMS_FUNCTION(fn_name)        \
  XNN_INTERNAL size_t fn_name(                                      \
    union xnn_qs8_add_minmax_params params[XNN_MIN_ELEMENTS(1)], \
    int8_t x_zero_point,                                            \
    int8_t y_zero_point,                                            \
    int8_t output_zero_point,                                       \
    float x_output_scale,                                           \
    float y_output_scale,                                           \
    int8_t output_min,                                              \
    int8_t output_max);

DECLARE_INIT_QS8_ADD_MINMAX_PARAMS_FUNCTION(xnn_init_qs8_add_minmax_scalar_params)
#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  DECLARE_INIT_QS8_ADD_MINMAX_PARAMS_FUNCTION(xnn_init_qs8_add_minmax_scalar_params)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64
#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  DECLARE_INIT_QS8_ADD_MINMAX_PARAMS_FUNCTION(xnn_init_qs8_add_minmax_scalar_params)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64
#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  DECLARE_INIT_QS8_ADD_MINMAX_PARAMS_FUNCTION(xnn_init_qs8_add_minmax_scalar_params)
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
#if XNN_ARCH_HEXAGON
  DECLARE_INIT_QS8_ADD_MINMAX_PARAMS_FUNCTION(xnn_init_qs8_add_minmax_scalar_params)
#endif // XNN_ARCH_HEXAGON

#define DECLARE_INIT_QU8_ADD_MINMAX_PARAMS_FUNCTION(fn_name)        \
  XNN_INTERNAL size_t fn_name(                                      \
    union xnn_qu8_add_minmax_params params[XNN_MIN_ELEMENTS(1)], \
    uint8_t x_zero_point,                                           \
    uint8_t y_zero_point,                                           \
    uint8_t output_zero_point,                                      \
    float x_output_scale,                                           \
    float y_output_scale,                                           \
    uint8_t output_min,                                             \
    uint8_t output_max);

DECLARE_INIT_QU8_ADD_MINMAX_PARAMS_FUNCTION(xnn_init_qu8_add_minmax_scalar_params)
#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  DECLARE_INIT_QU8_ADD_MINMAX_PARAMS_FUNCTION(xnn_init_qu8_add_minmax_scalar_params)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64
#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  DECLARE_INIT_QU8_ADD_MINMAX_PARAMS_FUNCTION(xnn_init_qu8_add_minmax_scalar_params)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64
#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  DECLARE_INIT_QU8_ADD_MINMAX_PARAMS_FUNCTION(xnn_init_qu8_add_minmax_scalar_params)
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#define DECLARE_INIT_QS8_MUL_MINMAX_PARAMS_FUNCTION(fn_name)     \
  XNN_INTERNAL size_t fn_name(                                   \
    union xnn_qs8_mul_minmax_params params[XNN_MIN_ELEMENTS(1)], \
    int8_t a_zero_point,                                         \
    int8_t b_zero_point,                                         \
    int8_t output_zero_point,                                    \
    float product_output_scale,                                  \
    int8_t output_min,                                           \
    int8_t output_max);

DECLARE_INIT_QS8_MUL_MINMAX_PARAMS_FUNCTION(xnn_init_qs8_mul_minmax_scalar_params)
#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  DECLARE_INIT_QS8_MUL_MINMAX_PARAMS_FUNCTION(xnn_init_qs8_mul_minmax_scalar_params)
  DECLARE_INIT_QS8_MUL_MINMAX_PARAMS_FUNCTION(xnn_init_qs8_mul_minmax_scalar_params)
  DECLARE_INIT_QS8_MUL_MINMAX_PARAMS_FUNCTION(xnn_init_qs8_mul_minmax_rndnu_neon_params)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64
#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  DECLARE_INIT_QS8_MUL_MINMAX_PARAMS_FUNCTION(xnn_init_qs8_mul_minmax_scalar_params)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64
#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  DECLARE_INIT_QS8_MUL_MINMAX_PARAMS_FUNCTION(xnn_init_qs8_mul_minmax_scalar_params)
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#define DECLARE_INIT_QU8_MUL_MINMAX_PARAMS_FUNCTION(fn_name)     \
  XNN_INTERNAL size_t fn_name(                                   \
    union xnn_qu8_mul_minmax_params params[XNN_MIN_ELEMENTS(1)], \
    uint8_t a_zero_point,                                        \
    uint8_t b_zero_point,                                        \
    uint8_t output_zero_point,                                   \
    float product_output_scale,                                  \
    uint8_t output_min,                                          \
    uint8_t output_max);

DECLARE_INIT_QU8_MUL_MINMAX_PARAMS_FUNCTION(xnn_init_qu8_mul_minmax_scalar_params)
#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  DECLARE_INIT_QU8_MUL_MINMAX_PARAMS_FUNCTION(xnn_init_qu8_mul_minmax_scalar_params)
  DECLARE_INIT_QU8_MUL_MINMAX_PARAMS_FUNCTION(xnn_init_qu8_mul_minmax_scalar_params)
  DECLARE_INIT_QU8_MUL_MINMAX_PARAMS_FUNCTION(xnn_init_qu8_mul_minmax_rndnu_neon_params)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64
#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  DECLARE_INIT_QU8_MUL_MINMAX_PARAMS_FUNCTION(xnn_init_qu8_mul_minmax_scalar_params)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64
#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  DECLARE_INIT_QU8_MUL_MINMAX_PARAMS_FUNCTION(xnn_init_qu8_mul_minmax_scalar_params)
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#define DECLARE_INIT_F16_QS8_CVT_PARAMS_FUNCTION(fn_name)     \
  XNN_INTERNAL size_t fn_name(                                \
    union xnn_f16_qs8_cvt_params params[XNN_MIN_ELEMENTS(1)], \
    uint16_t scale,                                           \
    int8_t zero_point,                                        \
    int8_t output_min,                                        \
    int8_t output_max);

#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  DECLARE_INIT_F16_QS8_CVT_PARAMS_FUNCTION(xnn_init_f16_qs8_cvt_neonfp16arith_params)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64

DECLARE_INIT_F16_QS8_CVT_PARAMS_FUNCTION(xnn_init_f16_qs8_cvt_scalar_params)
DECLARE_INIT_F16_QS8_CVT_PARAMS_FUNCTION(xnn_init_f16_qs8_cvt_scalar_params)
        //
#define DECLARE_INIT_F32_QS8_CVT_PARAMS_FUNCTION(fn_name)     \
  XNN_INTERNAL size_t fn_name(                                \
    union xnn_f32_qs8_cvt_params params[XNN_MIN_ELEMENTS(1)], \
    float scale,                                              \
    int8_t zero_point,                                        \
    int8_t output_min,                                        \
    int8_t output_max);

DECLARE_INIT_F32_QS8_CVT_PARAMS_FUNCTION(xnn_init_f32_qs8_cvt_scalar_params)
#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  DECLARE_INIT_F32_QS8_CVT_PARAMS_FUNCTION(xnn_init_f32_qs8_cvt_scalar_params)
  DECLARE_INIT_F32_QS8_CVT_PARAMS_FUNCTION(xnn_init_f32_qs8_cvt_scalar_params)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64
#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  DECLARE_INIT_F32_QS8_CVT_PARAMS_FUNCTION(xnn_init_f32_qs8_cvt_scalar_params)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64
#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  DECLARE_INIT_F32_QS8_CVT_PARAMS_FUNCTION(xnn_init_f32_qs8_cvt_scalar_params)
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
#if XNN_ARCH_HEXAGON
  DECLARE_INIT_F32_QS8_CVT_PARAMS_FUNCTION(xnn_init_f32_qs8_cvt_scalar_params)
#endif  // XNN_ARCH_HEXAGON


#define DECLARE_INIT_F32_QU8_CVT_PARAMS_FUNCTION(fn_name)     \
  XNN_INTERNAL size_t fn_name(                                \
    union xnn_f32_qu8_cvt_params params[XNN_MIN_ELEMENTS(1)], \
    float scale,                                              \
    uint8_t zero_point,                                       \
    uint8_t output_min,                                       \
    uint8_t output_max);

DECLARE_INIT_F32_QU8_CVT_PARAMS_FUNCTION(xnn_init_f32_qu8_cvt_scalar_params)
#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  DECLARE_INIT_F32_QU8_CVT_PARAMS_FUNCTION(xnn_init_f32_qu8_cvt_scalar_params)
  DECLARE_INIT_F32_QU8_CVT_PARAMS_FUNCTION(xnn_init_f32_qu8_cvt_scalar_params)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64
#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  DECLARE_INIT_F32_QU8_CVT_PARAMS_FUNCTION(xnn_init_f32_qu8_cvt_scalar_params)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64
#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  DECLARE_INIT_F32_QU8_CVT_PARAMS_FUNCTION(xnn_init_f32_qu8_cvt_scalar_params)
  DECLARE_INIT_F32_QU8_CVT_PARAMS_FUNCTION(xnn_init_f32_qu8_cvt_scalar_params)
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#define DECLARE_INIT_QS8_CVT_PARAMS_FUNCTION(fn_name)     \
  XNN_INTERNAL size_t fn_name(                            \
    union xnn_qs8_cvt_params params[XNN_MIN_ELEMENTS(1)], \
    float input_output_scale,                             \
    int8_t input_zero_point,                              \
    int8_t output_zero_point);

DECLARE_INIT_QS8_CVT_PARAMS_FUNCTION(xnn_init_qs8_cvt_scalar_params)

#define DECLARE_INIT_QS16_QS8_CVT_PARAMS_FUNCTION(fn_name)     \
  XNN_INTERNAL size_t fn_name(                                 \
    union xnn_qs16_qs8_cvt_params params[XNN_MIN_ELEMENTS(1)], \
    float input_output_scale,                                  \
    int8_t output_zero_point);

DECLARE_INIT_QS16_QS8_CVT_PARAMS_FUNCTION(xnn_init_qs16_qs8_cvt_scalar_params)

#define DECLARE_INIT_QS8_F16_CVT_PARAMS_FUNCTION(fn_name)     \
  XNN_INTERNAL size_t fn_name(                                \
    union xnn_qs8_f16_cvt_params params[XNN_MIN_ELEMENTS(1)], \
    uint16_t scale,                                           \
    int8_t zero_point);

#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  DECLARE_INIT_QS8_F16_CVT_PARAMS_FUNCTION(xnn_init_qs8_f16_cvt_neonfp16arith_params)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64
#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  DECLARE_INIT_QS8_F16_CVT_PARAMS_FUNCTION(xnn_init_qs8_f16_cvt_avx_params)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

#define DECLARE_INIT_QS8_F32_CVT_PARAMS_FUNCTION(fn_name)     \
  XNN_INTERNAL size_t fn_name(                                \
    union xnn_qs8_f32_cvt_params params[XNN_MIN_ELEMENTS(1)], \
    float scale,                                              \
    int8_t zero_point);

DECLARE_INIT_QS8_F32_CVT_PARAMS_FUNCTION(xnn_init_qs8_f32_cvt_scalar_params)

#define DECLARE_INIT_QU8_CVT_PARAMS_FUNCTION(fn_name)     \
  XNN_INTERNAL size_t fn_name(                            \
    union xnn_qu8_cvt_params params[XNN_MIN_ELEMENTS(1)], \
    float input_output_scale,                             \
    uint8_t input_zero_point,                             \
    uint8_t output_zero_point);

DECLARE_INIT_QU8_CVT_PARAMS_FUNCTION(xnn_init_qu8_cvt_scalar_params)

#define DECLARE_INIT_QU8_F32_CVT_PARAMS_FUNCTION(fn_name)     \
  XNN_INTERNAL size_t fn_name(                                \
    union xnn_qu8_f32_cvt_params params[XNN_MIN_ELEMENTS(1)], \
    float scale,                                              \
    uint8_t zero_point);

DECLARE_INIT_QU8_F32_CVT_PARAMS_FUNCTION(xnn_init_qu8_f32_cvt_scalar_params)


#ifdef __cplusplus
}  // extern "C"
#endif
