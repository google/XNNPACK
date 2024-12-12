// Copyright 2021 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <stddef.h>
#include <stdint.h>

#include "xnnpack.h"
#include "xnnpack/common.h"
#include "xnnpack/math.h"
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

DECLARE_INIT_QS8_QC8W_CONV_MINMAX_PARAMS_FUNCTION(xnn_init_qs8_qc8w_conv_minmax_fp32_scalar_params)
#if XNN_ARCH_ARM
  DECLARE_INIT_QS8_QC8W_CONV_MINMAX_PARAMS_FUNCTION(xnn_init_qs8_qc8w_conv_minmax_fp32_armsimd32_params)
#endif  // XNN_ARCH_ARM
#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  DECLARE_INIT_QS8_QC8W_CONV_MINMAX_PARAMS_FUNCTION(xnn_init_qs8_qc8w_conv_minmax_fp32_neon_params)
  DECLARE_INIT_QS8_QC8W_CONV_MINMAX_PARAMS_FUNCTION(xnn_init_qs8_qc8w_conv_minmax_fp32_neonv8_params)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#define DECLARE_INIT_QS8_CONV_MINMAX_PARAMS_FUNCTION(fn_name)     \
  XNN_INTERNAL size_t fn_name(                                    \
    union xnn_qs8_conv_minmax_params params[XNN_MIN_ELEMENTS(1)], \
    float scale,                                                  \
    int8_t output_zero_point,                                     \
    int8_t output_min,                                            \
    int8_t output_max);

DECLARE_INIT_QS8_CONV_MINMAX_PARAMS_FUNCTION(xnn_init_qs8_conv_minmax_fp32_scalar_params)
DECLARE_INIT_QS8_CONV_MINMAX_PARAMS_FUNCTION(xnn_init_qs8_conv_minmax_rndnu_scalar_params)
#if XNN_ARCH_ARM
  DECLARE_INIT_QS8_CONV_MINMAX_PARAMS_FUNCTION(xnn_init_qs8_conv_minmax_fp32_armsimd32_params)
#endif  // XNN_ARCH_ARM
#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  DECLARE_INIT_QS8_CONV_MINMAX_PARAMS_FUNCTION(xnn_init_qs8_conv_minmax_fp32_neon_params)
  DECLARE_INIT_QS8_CONV_MINMAX_PARAMS_FUNCTION(xnn_init_qs8_conv_minmax_fp32_neonv8_params)
  DECLARE_INIT_QS8_CONV_MINMAX_PARAMS_FUNCTION(xnn_init_qs8_conv_minmax_rndnu_neon_params)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64



#define DECLARE_INIT_QU8_CONV_MINMAX_PARAMS_FUNCTION(fn_name)     \
  XNN_INTERNAL size_t fn_name(                                    \
    union xnn_qu8_conv_minmax_params params[XNN_MIN_ELEMENTS(1)], \
    uint8_t kernel_zero_point,                                    \
    float scale,                                                  \
    uint8_t output_zero_point,                                    \
    uint8_t output_min,                                           \
    uint8_t output_max);

DECLARE_INIT_QU8_CONV_MINMAX_PARAMS_FUNCTION(xnn_init_qu8_conv_minmax_fp32_scalar_params)
DECLARE_INIT_QU8_CONV_MINMAX_PARAMS_FUNCTION(xnn_init_qu8_conv_minmax_rndnu_scalar_params)
DECLARE_INIT_QU8_CONV_MINMAX_PARAMS_FUNCTION(xnn_init_qu8_conv_minmax_rndnu16_scalar_params)
#if XNN_ARCH_ARM
  DECLARE_INIT_QU8_CONV_MINMAX_PARAMS_FUNCTION(xnn_init_qu8_conv_minmax_fp32_armsimd32_params)
#endif  // XNN_ARCH_ARM
#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  DECLARE_INIT_QU8_CONV_MINMAX_PARAMS_FUNCTION(xnn_init_qu8_conv_minmax_fp32_neon_params)
  DECLARE_INIT_QU8_CONV_MINMAX_PARAMS_FUNCTION(xnn_init_qu8_conv_minmax_fp32_neonv8_params)
  DECLARE_INIT_QU8_CONV_MINMAX_PARAMS_FUNCTION(xnn_init_qu8_conv_minmax_rndnu_neon_params)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


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
  const xnn_bfloat16 scale[XNN_MIN_ELEMENTS(1)],
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

XNN_INTERNAL size_t xnn_init_qu8_avgpool_minmax_fp32_scalar_params(
  struct xnn_qu8_avgpool_minmax_params params[XNN_MIN_ELEMENTS(1)],
  int32_t bias,
  float scale,
  uint8_t output_zero_point,
  uint8_t output_min,
  uint8_t output_max);

XNN_INTERNAL void xnn_update_qu8_avgpool_minmax_fp32_scalar_params(
  struct xnn_qu8_avgpool_minmax_params params[XNN_MIN_ELEMENTS(1)],
  int32_t bias,
  float scale);

XNN_INTERNAL size_t xnn_init_f16_scale_scalar_params(
  struct xnn_f16_scale_params params[XNN_MIN_ELEMENTS(1)],
  xnn_float16 scale);

XNN_INTERNAL size_t xnn_init_f16_f32acc_scale_scalar_params(
  struct xnn_f16_f32acc_scale_params params[XNN_MIN_ELEMENTS(1)],
  float scale);

XNN_INTERNAL size_t xnn_init_f32_scale_scalar_params(
  struct xnn_f32_scale_params params[XNN_MIN_ELEMENTS(1)],
  float scale);

#define DECLARE_INIT_REDUCE_PARAMS_FUNCTION(fn_name)            \
  XNN_INTERNAL size_t fn_name(                                  \
      struct xnn_reduce_params params[XNN_MIN_ELEMENTS(1)],     \
      const struct xnn_quantization_params* input_quantization, \
      const struct xnn_quantization_params* output_quantization);

DECLARE_INIT_REDUCE_PARAMS_FUNCTION(xnn_init_qs8_reduce_scalar_params);
DECLARE_INIT_REDUCE_PARAMS_FUNCTION(xnn_init_qu8_reduce_scalar_params);

#define DECLARE_UPDATE_REDUCE_PARAMS_FUNCTION(fn_name)          \
  XNN_INTERNAL size_t fn_name(                                  \
      struct xnn_reduce_params params[XNN_MIN_ELEMENTS(1)],     \
      float scale);

DECLARE_UPDATE_REDUCE_PARAMS_FUNCTION(xnn_update_f32_reduce_scalar_params);
DECLARE_UPDATE_REDUCE_PARAMS_FUNCTION(xnn_update_qs8_reduce_scalar_params);
DECLARE_UPDATE_REDUCE_PARAMS_FUNCTION(xnn_update_qu8_reduce_scalar_params);

XNN_INTERNAL size_t xnn_init_f16_scaleminmax_scalar_params(
  struct xnn_f16_scaleminmax_params params[XNN_MIN_ELEMENTS(1)],
  xnn_float16 scale,
  xnn_float16 min,
  xnn_float16 max);

XNN_INTERNAL void xnn_update_f16_scaleminmax_scalar_params(
  struct xnn_f16_scaleminmax_params params[XNN_MIN_ELEMENTS(1)],
  xnn_float16 scale);

XNN_INTERNAL size_t xnn_init_f32_scaleminmax_scalar_params(
  struct xnn_f32_scaleminmax_params params[XNN_MIN_ELEMENTS(1)],
  float scale,
  float min,
  float max);

XNN_INTERNAL void xnn_update_f32_scaleminmax_scalar_params(
  struct xnn_f32_scaleminmax_params params[XNN_MIN_ELEMENTS(1)],
  float scale);

XNN_INTERNAL size_t xnn_init_s8_minmax_scalar_params(
  struct xnn_s8_minmax_params params[XNN_MIN_ELEMENTS(1)],
  int8_t output_min,
  int8_t output_max);

XNN_INTERNAL size_t xnn_init_u8_minmax_scalar_params(
  struct xnn_u8_minmax_params params[XNN_MIN_ELEMENTS(1)],
  uint8_t output_min,
  uint8_t output_max);

XNN_INTERNAL size_t xnn_init_bf16_minmax_scalar_params(
  struct xnn_bf16_minmax_params params[XNN_MIN_ELEMENTS(1)],
  xnn_bfloat16 output_min,
  xnn_bfloat16 output_max);


XNN_INTERNAL size_t xnn_init_f16_minmax_scalar_params(
  union xnn_f16_minmax_params params[XNN_MIN_ELEMENTS(1)],
  xnn_float16 output_min,
  xnn_float16 output_max);

XNN_INTERNAL size_t xnn_init_f32_minmax_scalar_params(
  union xnn_f32_minmax_params params[XNN_MIN_ELEMENTS(1)],
  float output_min,
  float output_max);


XNN_INTERNAL size_t xnn_init_f16_qc4w_minmax_scalar_params(
  struct xnn_f16_qc4w_minmax_params params[XNN_MIN_ELEMENTS(1)],
  xnn_float16 output_min,
  xnn_float16 output_max,
  uint8_t kernel_zero_point);

XNN_INTERNAL size_t xnn_init_f16_qb4w_minmax_scalar_params(
  struct xnn_f16_qb4w_minmax_params params[XNN_MIN_ELEMENTS(1)],
  xnn_float16 output_min,
  xnn_float16 output_max,
  uint8_t kernel_zero_point,
  size_t blocksize);

XNN_INTERNAL size_t xnn_init_f32_qc4w_minmax_scalar_params(
  struct xnn_f32_qc4w_minmax_params params[XNN_MIN_ELEMENTS(1)],
  float output_min,
  float output_max,
  uint8_t kernel_zero_point);

XNN_INTERNAL size_t xnn_init_f32_qb4w_minmax_scalar_params(
  struct xnn_f32_qb4w_minmax_params params[XNN_MIN_ELEMENTS(1)],
  float output_min,
  float output_max,
  uint8_t kernel_zero_point,
  size_t blocksize);

#define DECLARE_INIT_UNARY_MICROPARAMS_FUNCTION(fn_name)       \
  XNN_INTERNAL size_t fn_name(                                 \
    union xnn_unary_uparams* microparams,                      \
    const union xnn_unary_params* op_params,                   \
    const struct xnn_quantization_params* input_quantization,  \
    const struct xnn_quantization_params* output_quantization);

DECLARE_INIT_UNARY_MICROPARAMS_FUNCTION(xnn_init_f16_elu_scalar_params);
DECLARE_INIT_UNARY_MICROPARAMS_FUNCTION(xnn_init_f32_elu_scalar_params);
DECLARE_INIT_UNARY_MICROPARAMS_FUNCTION(xnn_init_f16_lrelu_scalar_params);
DECLARE_INIT_UNARY_MICROPARAMS_FUNCTION(xnn_init_f32_lrelu_scalar_params);
DECLARE_INIT_UNARY_MICROPARAMS_FUNCTION(xnn_init_qs8_lrelu_scalar_params);
DECLARE_INIT_UNARY_MICROPARAMS_FUNCTION(xnn_init_qu8_lrelu_scalar_params);
DECLARE_INIT_UNARY_MICROPARAMS_FUNCTION(xnn_init_f16_clamp_scalar_params);
DECLARE_INIT_UNARY_MICROPARAMS_FUNCTION(xnn_init_f32_clamp_scalar_params);
DECLARE_INIT_UNARY_MICROPARAMS_FUNCTION(xnn_init_qs8_clamp_scalar_params);
DECLARE_INIT_UNARY_MICROPARAMS_FUNCTION(xnn_init_qu8_clamp_scalar_params);

DECLARE_INIT_UNARY_MICROPARAMS_FUNCTION(xnn_init_f16_qs8_cvt_scalar_params);
DECLARE_INIT_UNARY_MICROPARAMS_FUNCTION(xnn_init_f16_qu8_cvt_scalar_params);
DECLARE_INIT_UNARY_MICROPARAMS_FUNCTION(xnn_init_f32_qs8_cvt_scalar_params);
DECLARE_INIT_UNARY_MICROPARAMS_FUNCTION(xnn_init_f32_qu8_cvt_scalar_params);
DECLARE_INIT_UNARY_MICROPARAMS_FUNCTION(xnn_init_qs8_cvt_scalar_params);
DECLARE_INIT_UNARY_MICROPARAMS_FUNCTION(xnn_init_qs16_qs8_cvt_scalar_params);
DECLARE_INIT_UNARY_MICROPARAMS_FUNCTION(xnn_init_qs8_f16_cvt_scalar_params);
DECLARE_INIT_UNARY_MICROPARAMS_FUNCTION(xnn_init_qs8_f32_cvt_scalar_params);
DECLARE_INIT_UNARY_MICROPARAMS_FUNCTION(xnn_init_qu8_cvt_scalar_params);
DECLARE_INIT_UNARY_MICROPARAMS_FUNCTION(xnn_init_qu8_f32_cvt_scalar_params);

XNN_INTERNAL size_t xnn_init_qs8_add_minmax_scalar_params(
    struct xnn_qs8_add_minmax_params uparams[XNN_MIN_ELEMENTS(1)],
    const struct xnn_quantization_params* a_quantization,
    const struct xnn_quantization_params* b_quantization,
    const struct xnn_quantization_params* output_quantization);

XNN_INTERNAL size_t xnn_init_qu8_add_minmax_scalar_params(
    struct xnn_qu8_add_minmax_params uparams[XNN_MIN_ELEMENTS(1)],
    const struct xnn_quantization_params* a_quantization,
    const struct xnn_quantization_params* b_quantization,
    const struct xnn_quantization_params* output_quantization);

#define DECLARE_INIT_QS8_MUL_MINMAX_PARAMS_FUNCTION(fn_name)        \
  XNN_INTERNAL size_t fn_name(                                      \
      union xnn_qs8_mul_minmax_params uparams[XNN_MIN_ELEMENTS(1)], \
      const struct xnn_quantization_params* a_quantization,         \
      const struct xnn_quantization_params* b_quantization,         \
      const struct xnn_quantization_params* output_quantization);

DECLARE_INIT_QS8_MUL_MINMAX_PARAMS_FUNCTION(xnn_init_qs8_mul_minmax_scalar_params)
#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  DECLARE_INIT_QS8_MUL_MINMAX_PARAMS_FUNCTION(xnn_init_qs8_mul_minmax_rndnu_neon_params)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64

#define DECLARE_INIT_QU8_MUL_MINMAX_PARAMS_FUNCTION(fn_name)        \
  XNN_INTERNAL size_t fn_name(                                      \
      union xnn_qu8_mul_minmax_params uparams[XNN_MIN_ELEMENTS(1)], \
      const struct xnn_quantization_params* a_quantization,         \
      const struct xnn_quantization_params* b_quantization,         \
      const struct xnn_quantization_params* output_quantization);

  DECLARE_INIT_QU8_MUL_MINMAX_PARAMS_FUNCTION(
      xnn_init_qu8_mul_minmax_scalar_params)
#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  DECLARE_INIT_QU8_MUL_MINMAX_PARAMS_FUNCTION(xnn_init_qu8_mul_minmax_rndnu_neon_params)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#ifdef __cplusplus
}  // extern "C"
#endif
