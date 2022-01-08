// Copyright 2021 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <stdint.h>
#include <stddef.h>

#include <xnnpack/common.h>
#include <xnnpack/params.h>


#ifdef __cplusplus
extern "C" {
#endif


#define DECLARE_INIT_QU8_CONV_MINMAX_PARAMS_FUNCTION(fn_name)     \
  XNN_INTERNAL void fn_name(                                      \
    union xnn_qu8_conv_minmax_params params[XNN_MIN_ELEMENTS(1)], \
    uint8_t kernel_zero_point,                                    \
    float scale,                                                  \
    uint8_t output_zero_point,                                    \
    uint8_t output_min,                                           \
    uint8_t output_max);

DECLARE_INIT_QU8_CONV_MINMAX_PARAMS_FUNCTION(xnn_init_qu8_conv_minmax_fp32_scalar_fmagic_params)
DECLARE_INIT_QU8_CONV_MINMAX_PARAMS_FUNCTION(xnn_init_qu8_conv_minmax_fp32_scalar_imagic_params)
DECLARE_INIT_QU8_CONV_MINMAX_PARAMS_FUNCTION(xnn_init_qu8_conv_minmax_fp32_scalar_lrintf_params)

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


#define DECLARE_INIT_QS8_CONV_MINMAX_PARAMS_FUNCTION(fn_name)     \
  XNN_INTERNAL void fn_name(                                      \
    union xnn_qs8_conv_minmax_params params[XNN_MIN_ELEMENTS(1)], \
    float scale,                                                  \
    int8_t output_zero_point,                                     \
    int8_t output_min,                                            \
    int8_t output_max);

DECLARE_INIT_QS8_CONV_MINMAX_PARAMS_FUNCTION(xnn_init_qs8_conv_minmax_fp32_scalar_fmagic_params)
DECLARE_INIT_QS8_CONV_MINMAX_PARAMS_FUNCTION(xnn_init_qs8_conv_minmax_fp32_scalar_imagic_params)
DECLARE_INIT_QS8_CONV_MINMAX_PARAMS_FUNCTION(xnn_init_qs8_conv_minmax_fp32_scalar_lrintf_params)

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
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  DECLARE_INIT_QS8_CONV_MINMAX_PARAMS_FUNCTION(xnn_init_qs8_conv_minmax_fp32_wasmsimd_params)
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


XNN_INTERNAL void xnn_init_qc8_scale_fp32_params(
  size_t channels,
  size_t channels_tile,
  size_t stride,
  const float scale[XNN_MIN_ELEMENTS(1)],
  void* packed_w);


#define DECLARE_INIT_QS8_MINMAX_PARAMS_FUNCTION(fn_name)     \
  XNN_INTERNAL void fn_name(                                 \
    union xnn_qs8_minmax_params params[XNN_MIN_ELEMENTS(1)], \
    int8_t output_zero_point,                                \
    int8_t output_min,                                       \
    int8_t output_max);

DECLARE_INIT_QS8_MINMAX_PARAMS_FUNCTION(xnn_init_qs8_minmax_scalar_fmagic_params)
DECLARE_INIT_QS8_MINMAX_PARAMS_FUNCTION(xnn_init_qs8_minmax_scalar_imagic_params)
DECLARE_INIT_QS8_MINMAX_PARAMS_FUNCTION(xnn_init_qs8_minmax_scalar_lrintf_params)

#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  DECLARE_INIT_QS8_MINMAX_PARAMS_FUNCTION(xnn_init_qs8_minmax_neon_params)
  DECLARE_INIT_QS8_MINMAX_PARAMS_FUNCTION(xnn_init_qs8_minmax_neonv8_params)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  DECLARE_INIT_QS8_MINMAX_PARAMS_FUNCTION(xnn_init_qs8_minmax_sse2_params)
  DECLARE_INIT_QS8_MINMAX_PARAMS_FUNCTION(xnn_init_qs8_minmax_sse4_params)
  DECLARE_INIT_QS8_MINMAX_PARAMS_FUNCTION(xnn_init_qs8_minmax_avx2_params)
  DECLARE_INIT_QS8_MINMAX_PARAMS_FUNCTION(xnn_init_qs8_minmax_avx512_params)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  DECLARE_INIT_QS8_MINMAX_PARAMS_FUNCTION(xnn_init_qs8_minmax_wasmsimd_params)
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


XNN_INTERNAL void xnn_init_qu8_avgpool_minmax_params(
  union xnn_qu8_avgpool_minmax_params params[XNN_MIN_ELEMENTS(1)],
  int32_t bias,
  float scale,
  uint8_t output_zero_point,
  uint8_t output_min,
  uint8_t output_max);

XNN_INTERNAL void xnn_init_qu8_avgpool_minmax_scalar_params(
  union xnn_qu8_avgpool_minmax_params params[XNN_MIN_ELEMENTS(1)],
  int32_t bias,
  float scale,
  uint8_t output_zero_point,
  uint8_t output_min,
  uint8_t output_max);

XNN_INTERNAL void xnn_update_qu8_avgpool_minmax_params(
  union xnn_qu8_avgpool_minmax_params* params,
  int32_t bias,
  float scale);

XNN_INTERNAL void xnn_init_qs8_avgpool_minmax_params(
  union xnn_qs8_avgpool_minmax_params params[XNN_MIN_ELEMENTS(1)],
  int32_t bias,
  float scale,
  int8_t output_zero_point,
  int8_t output_min,
  int8_t output_max);

XNN_INTERNAL void xnn_init_qs8_avgpool_minmax_scalar_params(
  union xnn_qs8_avgpool_minmax_params params[XNN_MIN_ELEMENTS(1)],
  int32_t bias,
  float scale,
  int8_t output_zero_point,
  int8_t output_min,
  int8_t output_max);

XNN_INTERNAL void xnn_update_qs8_avgpool_minmax_params(
  union xnn_qs8_avgpool_minmax_params* params,
  int32_t bias,
  float scale);

XNN_INTERNAL void xnn_update_f16_scaleminmax_params(
  union xnn_f16_scaleminmax_params* params,
  uint16_t scale);

XNN_INTERNAL void xnn_update_f32_scaleminmax_params(
  union xnn_f32_scaleminmax_params* params,
  float scale);

#define DECLARE_INIT_F16_SCALEMINMAX_PARAMS_FUNCTION(fn_name)     \
  XNN_INTERNAL void fn_name(                                      \
    union xnn_f16_scaleminmax_params params[XNN_MIN_ELEMENTS(1)], \
    uint16_t scale,                                               \
    uint16_t min,                                                 \
    uint16_t max);

// TODO(maratek): remove once all operators are updated to function pointers
DECLARE_INIT_F16_SCALEMINMAX_PARAMS_FUNCTION(xnn_init_f16_scaleminmax_neon_params)

#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  DECLARE_INIT_F16_SCALEMINMAX_PARAMS_FUNCTION(xnn_init_f16_scaleminmax_neon_params)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  DECLARE_INIT_F16_SCALEMINMAX_PARAMS_FUNCTION(xnn_init_f16_scaleminmax_avx_params)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


XNN_INTERNAL void xnn_init_f32_scaleminmax_scalar_params(
  union xnn_f32_scaleminmax_params params[XNN_MIN_ELEMENTS(1)],
  float scale,
  float min,
  float max);

XNN_INTERNAL void xnn_init_f32_scaleminmax_params(
  union xnn_f32_scaleminmax_params params[XNN_MIN_ELEMENTS(1)],
  float scale,
  float min,
  float max);

XNN_INTERNAL void xnn_init_f32_gavgpool_params(
  union xnn_f32_gavgpool_params params[XNN_MIN_ELEMENTS(1)],
  float multiplier,
  float output_min,
  float output_max,
  uint32_t width);

XNN_INTERNAL void xnn_update_f32_gavgpool_params(
  union xnn_f32_gavgpool_params* params,
  float multiplier,
  uint32_t width);

XNN_INTERNAL void xnn_init_scalar_f32_gavgpool_params(
  union xnn_f32_gavgpool_params params[XNN_MIN_ELEMENTS(1)],
  float multiplier,
  float output_min,
  float output_max,
  uint32_t width);

XNN_INTERNAL void xnn_init_f16_minmax_params(
  union xnn_f16_minmax_params params[XNN_MIN_ELEMENTS(1)],
  uint16_t min,
  uint16_t max);


#define DECLARE_INIT_F32_DEFAULT_PARAMS_FUNCTION(fn_name)      \
  XNN_INTERNAL void fn_name(                                   \
    union xnn_f32_default_params params[XNN_MIN_ELEMENTS(1)]);

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  DECLARE_INIT_F32_DEFAULT_PARAMS_FUNCTION(xnn_init_f32_default_avx_params)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#define DECLARE_INIT_F32_MINMAX_PARAMS_FUNCTION(fn_name)     \
  XNN_INTERNAL void fn_name(                                 \
    union xnn_f32_minmax_params params[XNN_MIN_ELEMENTS(1)], \
    float output_min,                                        \
    float output_max);

DECLARE_INIT_F32_MINMAX_PARAMS_FUNCTION(xnn_init_f32_minmax_params)

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  DECLARE_INIT_F32_MINMAX_PARAMS_FUNCTION(xnn_init_f32_minmax_sse_params)
  DECLARE_INIT_F32_MINMAX_PARAMS_FUNCTION(xnn_init_f32_minmax_avx_params)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  DECLARE_INIT_F32_MINMAX_PARAMS_FUNCTION(xnn_init_f32_minmax_wasmsimd_params)
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD

DECLARE_INIT_F32_MINMAX_PARAMS_FUNCTION(xnn_init_f32_minmax_scalar_params)


XNN_INTERNAL void xnn_init_f16_hswish_params(
  union xnn_f16_hswish_params params[XNN_MIN_ELEMENTS(1)]);


#define DECLARE_INIT_F32_HSWISH_PARAMS_FUNCTION(fn_name) \
  XNN_INTERNAL void fn_name(                             \
    union xnn_f32_hswish_params params[XNN_MIN_ELEMENTS(1)]);

DECLARE_INIT_F32_HSWISH_PARAMS_FUNCTION(xnn_init_f32_hswish_scalar_params)

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  DECLARE_INIT_F32_HSWISH_PARAMS_FUNCTION(xnn_init_f32_hswish_sse_params)
  DECLARE_INIT_F32_HSWISH_PARAMS_FUNCTION(xnn_init_f32_hswish_avx_params)
  DECLARE_INIT_F32_HSWISH_PARAMS_FUNCTION(xnn_init_f32_hswish_avx512_params)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  DECLARE_INIT_F32_HSWISH_PARAMS_FUNCTION(xnn_init_f32_hswish_wasmsimd_params)
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#define DECLARE_INIT_F32_SIGMOID_PARAMS_FUNCTION(fn_name)      \
  XNN_INTERNAL void fn_name(                                   \
    union xnn_f32_sigmoid_params params[XNN_MIN_ELEMENTS(1)]);

DECLARE_INIT_F32_SIGMOID_PARAMS_FUNCTION(xnn_init_f32_sigmoid_scalar_rr2_lut2048_p1_params)
DECLARE_INIT_F32_SIGMOID_PARAMS_FUNCTION(xnn_init_f32_sigmoid_scalar_rr2_lut64_p2_params)
DECLARE_INIT_F32_SIGMOID_PARAMS_FUNCTION(xnn_init_f32_sigmoid_scalar_rr2_p5_params)

#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  DECLARE_INIT_F32_SIGMOID_PARAMS_FUNCTION(xnn_init_f32_sigmoid_neon_rr2_lut2048_p1_params)
  DECLARE_INIT_F32_SIGMOID_PARAMS_FUNCTION(xnn_init_f32_sigmoid_neon_rr2_lut64_p2_params)
  DECLARE_INIT_F32_SIGMOID_PARAMS_FUNCTION(xnn_init_f32_sigmoid_neon_rr2_p5_params)
  DECLARE_INIT_F32_SIGMOID_PARAMS_FUNCTION(xnn_init_f32_sigmoid_neonfma_rr1_lut2048_p1_params)
  DECLARE_INIT_F32_SIGMOID_PARAMS_FUNCTION(xnn_init_f32_sigmoid_neonfma_rr1_lut64_p2_params)
  DECLARE_INIT_F32_SIGMOID_PARAMS_FUNCTION(xnn_init_f32_sigmoid_neonfma_rr1_p5_params)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  DECLARE_INIT_F32_SIGMOID_PARAMS_FUNCTION(xnn_init_f32_sigmoid_sse2_rr2_lut64_p2_params)
  DECLARE_INIT_F32_SIGMOID_PARAMS_FUNCTION(xnn_init_f32_sigmoid_sse2_rr2_p5_params)
  DECLARE_INIT_F32_SIGMOID_PARAMS_FUNCTION(xnn_init_f32_sigmoid_avx_rr2_p5_params)
  DECLARE_INIT_F32_SIGMOID_PARAMS_FUNCTION(xnn_init_f32_sigmoid_avx2_rr1_p5_params)
  DECLARE_INIT_F32_SIGMOID_PARAMS_FUNCTION(xnn_init_f32_sigmoid_avx512_rr1_lut16_p3_params)
  DECLARE_INIT_F32_SIGMOID_PARAMS_FUNCTION(xnn_init_f32_sigmoid_avx512_rr2_lut32_p2_params)
  DECLARE_INIT_F32_SIGMOID_PARAMS_FUNCTION(xnn_init_f32_sigmoid_avx512_rr1_p5_params)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  DECLARE_INIT_F32_SIGMOID_PARAMS_FUNCTION(xnn_init_f32_sigmoid_wasmsimd_rr2_lut64_p2_params)
  DECLARE_INIT_F32_SIGMOID_PARAMS_FUNCTION(xnn_init_f32_sigmoid_wasmsimd_rr2_p5_params)
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#define DECLARE_INIT_F32_ABS_PARAMS_FUNCTION(fn_name)      \
  XNN_INTERNAL void fn_name(                               \
    union xnn_f32_abs_params params[XNN_MIN_ELEMENTS(1)]);

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  DECLARE_INIT_F32_ABS_PARAMS_FUNCTION(xnn_init_f32_abs_sse_params)
  DECLARE_INIT_F32_ABS_PARAMS_FUNCTION(xnn_init_f32_abs_avx_params)
  DECLARE_INIT_F32_ABS_PARAMS_FUNCTION(xnn_init_f32_abs_avx512_params)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  DECLARE_INIT_F32_ABS_PARAMS_FUNCTION(xnn_init_f32_abs_wasmsimd_params)
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#define DECLARE_INIT_F32_NEG_PARAMS_FUNCTION(fn_name)      \
  XNN_INTERNAL void fn_name(                               \
    union xnn_f32_neg_params params[XNN_MIN_ELEMENTS(1)]);

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  DECLARE_INIT_F32_NEG_PARAMS_FUNCTION(xnn_init_f32_neg_sse_params)
  DECLARE_INIT_F32_NEG_PARAMS_FUNCTION(xnn_init_f32_neg_avx_params)
  DECLARE_INIT_F32_NEG_PARAMS_FUNCTION(xnn_init_f32_neg_avx512_params)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  DECLARE_INIT_F32_NEG_PARAMS_FUNCTION(xnn_init_f32_neg_wasmsimd_params)
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#define DECLARE_INIT_F32_RND_PARAMS_FUNCTION(fn_name)      \
  XNN_INTERNAL void fn_name(                               \
    union xnn_f32_rnd_params params[XNN_MIN_ELEMENTS(1)]);

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  DECLARE_INIT_F32_RND_PARAMS_FUNCTION(xnn_init_f32_rnd_sse2_params)
  DECLARE_INIT_F32_RND_PARAMS_FUNCTION(xnn_init_f32_rnd_avx_params)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  DECLARE_INIT_F32_RND_PARAMS_FUNCTION(xnn_init_f32_rnd_wasmsimd_params)
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#define DECLARE_INIT_F32_ELU_PARAMS_FUNCTION(fn_name)     \
  XNN_INTERNAL void fn_name(                              \
    union xnn_f32_elu_params params[XNN_MIN_ELEMENTS(1)], \
    float prescale,                                       \
    float alpha,                                          \
    float beta);

DECLARE_INIT_F32_ELU_PARAMS_FUNCTION(xnn_init_f32_elu_scalar_rr2_lut16_p3_params)
DECLARE_INIT_F32_ELU_PARAMS_FUNCTION(xnn_init_f32_elu_scalar_rr2_p6_params)

#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  DECLARE_INIT_F32_ELU_PARAMS_FUNCTION(xnn_init_f32_elu_neon_rr2_lut16_p3_params)
  DECLARE_INIT_F32_ELU_PARAMS_FUNCTION(xnn_init_f32_elu_neon_rr2_p6_params)
  DECLARE_INIT_F32_ELU_PARAMS_FUNCTION(xnn_init_f32_elu_neonfma_rr1_lut16_p3_params)
  DECLARE_INIT_F32_ELU_PARAMS_FUNCTION(xnn_init_f32_elu_neonfma_rr1_p6_params)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  DECLARE_INIT_F32_ELU_PARAMS_FUNCTION(xnn_init_f32_elu_sse2_rr2_lut16_p3_params)
  DECLARE_INIT_F32_ELU_PARAMS_FUNCTION(xnn_init_f32_elu_sse2_rr2_p6_params)
  DECLARE_INIT_F32_ELU_PARAMS_FUNCTION(xnn_init_f32_elu_avx_rr2_lut16_p3_params)
  DECLARE_INIT_F32_ELU_PARAMS_FUNCTION(xnn_init_f32_elu_avx_rr2_lut4_p4_params)
  DECLARE_INIT_F32_ELU_PARAMS_FUNCTION(xnn_init_f32_elu_avx_rr2_p6_params)
  DECLARE_INIT_F32_ELU_PARAMS_FUNCTION(xnn_init_f32_elu_avx2_rr1_lut16_p3_params)
  DECLARE_INIT_F32_ELU_PARAMS_FUNCTION(xnn_init_f32_elu_avx2_rr1_lut8_p4_params)
  DECLARE_INIT_F32_ELU_PARAMS_FUNCTION(xnn_init_f32_elu_avx2_rr1_lut4_p4_params)
  DECLARE_INIT_F32_ELU_PARAMS_FUNCTION(xnn_init_f32_elu_avx2_rr1_p6_params)
  DECLARE_INIT_F32_ELU_PARAMS_FUNCTION(xnn_init_f32_elu_avx512_rr1_lut16_p3_params)
  DECLARE_INIT_F32_ELU_PARAMS_FUNCTION(xnn_init_f32_elu_avx512_rr1_p6_params)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  DECLARE_INIT_F32_ELU_PARAMS_FUNCTION(xnn_init_f32_elu_wasmsimd_rr2_lut16_p3_params)
  DECLARE_INIT_F32_ELU_PARAMS_FUNCTION(xnn_init_f32_elu_wasmsimd_rr2_p6_params)
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#define DECLARE_INIT_F32_EXPMINUS_PARAMS_FUNCTION(fn_name)      \
  XNN_INTERNAL void fn_name(                                   \
    union xnn_f32_expminus_params params[XNN_MIN_ELEMENTS(1)]);

DECLARE_INIT_F32_EXPMINUS_PARAMS_FUNCTION(xnn_init_f32_expminus_scalar_rr2_lut64_p2_params)
DECLARE_INIT_F32_EXPMINUS_PARAMS_FUNCTION(xnn_init_f32_expminus_scalar_rr2_p5_params)

#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  DECLARE_INIT_F32_EXPMINUS_PARAMS_FUNCTION(xnn_init_f32_expminus_neon_rr2_lut64_p2_params)
  DECLARE_INIT_F32_EXPMINUS_PARAMS_FUNCTION(xnn_init_f32_expminus_neon_rr2_p5_params)
  DECLARE_INIT_F32_EXPMINUS_PARAMS_FUNCTION(xnn_init_f32_expminus_neonfma_rr1_lut64_p2_params)
  DECLARE_INIT_F32_EXPMINUS_PARAMS_FUNCTION(xnn_init_f32_expminus_neonfma_rr1_p5_params)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  DECLARE_INIT_F32_EXPMINUS_PARAMS_FUNCTION(xnn_init_f32_expminus_sse2_rr2_p5_params)
  DECLARE_INIT_F32_EXPMINUS_PARAMS_FUNCTION(xnn_init_f32_expminus_avx2_rr1_p5_params)
  DECLARE_INIT_F32_EXPMINUS_PARAMS_FUNCTION(xnn_init_f32_expminus_avx512_rr1_p5_params)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  DECLARE_INIT_F32_EXPMINUS_PARAMS_FUNCTION(xnn_init_f32_expminus_wasmsimd_rr2_p5_params)
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#define DECLARE_INIT_F32_LRELU_PARAMS_FUNCTION(fn_name)     \
  XNN_INTERNAL void fn_name(                                \
    union xnn_f32_lrelu_params params[XNN_MIN_ELEMENTS(1)], \
    float slope);

DECLARE_INIT_F32_LRELU_PARAMS_FUNCTION(xnn_init_f32_lrelu_scalar_params)

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  DECLARE_INIT_F32_LRELU_PARAMS_FUNCTION(xnn_init_f32_lrelu_sse_params)
  DECLARE_INIT_F32_LRELU_PARAMS_FUNCTION(xnn_init_f32_lrelu_avx_params)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  DECLARE_INIT_F32_LRELU_PARAMS_FUNCTION(xnn_init_f32_lrelu_wasmsimd_params)
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#define DECLARE_INIT_F32_SQRT_PARAMS_FUNCTION(fn_name)      \
  XNN_INTERNAL void fn_name(                                \
    union xnn_f32_sqrt_params params[XNN_MIN_ELEMENTS(1)]);

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  DECLARE_INIT_F32_SQRT_PARAMS_FUNCTION(xnn_init_f32_sqrt_avx_params)
  DECLARE_INIT_F32_SQRT_PARAMS_FUNCTION(xnn_init_f32_sqrt_fma_params)
  DECLARE_INIT_F32_SQRT_PARAMS_FUNCTION(xnn_init_f32_sqrt_avx512_params)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


XNN_INTERNAL void xnn_init_f32_chw_params(
  union xnn_f32_chw_params params[XNN_MIN_ELEMENTS(1)],
  uint32_t width,
  float output_min,
  float output_max);

XNN_INTERNAL void xnn_update_f32_chw_params(
  union xnn_f32_chw_params* params,
  uint32_t width);

XNN_INTERNAL void xnn_init_scalar_f32_chw_params(
  union xnn_f32_chw_params params[XNN_MIN_ELEMENTS(1)],
  uint32_t width,
  float output_min,
  float output_max);


#define DECLARE_INIT_S8_MINMAX_PARAMS_FUNCTION(fn_name)     \
  XNN_INTERNAL void fn_name(                                \
    union xnn_s8_minmax_params params[XNN_MIN_ELEMENTS(1)], \
    int8_t output_min,                                      \
    int8_t output_max);

DECLARE_INIT_S8_MINMAX_PARAMS_FUNCTION(xnn_init_s8_minmax_scalar_params)

#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  DECLARE_INIT_S8_MINMAX_PARAMS_FUNCTION(xnn_init_s8_minmax_neon_params)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  DECLARE_INIT_S8_MINMAX_PARAMS_FUNCTION(xnn_init_s8_minmax_sse2_params)
  DECLARE_INIT_S8_MINMAX_PARAMS_FUNCTION(xnn_init_s8_minmax_sse4_params)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  DECLARE_INIT_S8_MINMAX_PARAMS_FUNCTION(xnn_init_s8_minmax_wasmsimd_params)
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#define DECLARE_INIT_U8_MINMAX_PARAMS_FUNCTION(fn_name)      \
  XNN_INTERNAL void fn_name(                                 \
    union xnn_u8_minmax_params params[XNN_MIN_ELEMENTS(1)],  \
    uint8_t output_min,                                      \
    uint8_t output_max);

DECLARE_INIT_U8_MINMAX_PARAMS_FUNCTION(xnn_init_u8_minmax_params)
DECLARE_INIT_U8_MINMAX_PARAMS_FUNCTION(xnn_init_u8_minmax_scalar_params)

#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  DECLARE_INIT_U8_MINMAX_PARAMS_FUNCTION(xnn_init_u8_minmax_neon_params)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  DECLARE_INIT_U8_MINMAX_PARAMS_FUNCTION(xnn_init_u8_minmax_sse2_params)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  DECLARE_INIT_U8_MINMAX_PARAMS_FUNCTION(xnn_init_u8_minmax_wasmsimd_params)
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#define DECLARE_INIT_QU8_ADD_MINMAX_PARAMS_FUNCTION(fn_name)        \
  XNN_INTERNAL void fn_name(                                        \
    union xnn_qu8_addsub_minmax_params params[XNN_MIN_ELEMENTS(1)], \
    uint8_t x_zero_point,                                           \
    uint8_t y_zero_point,                                           \
    uint8_t output_zero_point,                                      \
    float x_output_scale,                                           \
    float y_output_scale,                                           \
    uint8_t output_min,                                             \
    uint8_t output_max);

DECLARE_INIT_QU8_ADD_MINMAX_PARAMS_FUNCTION(xnn_init_qu8_add_minmax_scalar_params)

#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  DECLARE_INIT_QU8_ADD_MINMAX_PARAMS_FUNCTION(xnn_init_qu8_add_minmax_neon_params)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  DECLARE_INIT_QU8_ADD_MINMAX_PARAMS_FUNCTION(xnn_init_qu8_add_minmax_sse2_params)
  DECLARE_INIT_QU8_ADD_MINMAX_PARAMS_FUNCTION(xnn_init_qu8_add_minmax_sse4_params)
  DECLARE_INIT_QU8_ADD_MINMAX_PARAMS_FUNCTION(xnn_init_qu8_add_minmax_avx2_params)
  DECLARE_INIT_QU8_ADD_MINMAX_PARAMS_FUNCTION(xnn_init_qu8_add_minmax_avx512_params)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  DECLARE_INIT_QU8_ADD_MINMAX_PARAMS_FUNCTION(xnn_init_qu8_add_minmax_wasmsimd_params)
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#define DECLARE_INIT_QS8_ADD_MINMAX_PARAMS_FUNCTION(fn_name)        \
  XNN_INTERNAL void fn_name(                                        \
    union xnn_qs8_addsub_minmax_params params[XNN_MIN_ELEMENTS(1)], \
    int8_t x_zero_point,                                            \
    int8_t y_zero_point,                                            \
    int8_t output_zero_point,                                       \
    float x_output_scale,                                           \
    float y_output_scale,                                           \
    int8_t output_min,                                              \
    int8_t output_max);

DECLARE_INIT_QS8_ADD_MINMAX_PARAMS_FUNCTION(xnn_init_qs8_add_minmax_scalar_params)

#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  DECLARE_INIT_QS8_ADD_MINMAX_PARAMS_FUNCTION(xnn_init_qs8_add_minmax_neon_params)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  DECLARE_INIT_QS8_ADD_MINMAX_PARAMS_FUNCTION(xnn_init_qs8_add_minmax_sse2_params)
  DECLARE_INIT_QS8_ADD_MINMAX_PARAMS_FUNCTION(xnn_init_qs8_add_minmax_sse4_mul16_params)
  DECLARE_INIT_QS8_ADD_MINMAX_PARAMS_FUNCTION(xnn_init_qs8_add_minmax_sse4_mul32_params)
  DECLARE_INIT_QS8_ADD_MINMAX_PARAMS_FUNCTION(xnn_init_qs8_add_minmax_avx2_params)
  DECLARE_INIT_QS8_ADD_MINMAX_PARAMS_FUNCTION(xnn_init_qs8_add_minmax_avx512_params)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  DECLARE_INIT_QS8_ADD_MINMAX_PARAMS_FUNCTION(xnn_init_qs8_add_minmax_wasmsimd_params)
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#define DECLARE_INIT_QU8_MUL_MINMAX_PARAMS_FUNCTION(fn_name)     \
  XNN_INTERNAL void fn_name(                                     \
    union xnn_qu8_mul_minmax_params params[XNN_MIN_ELEMENTS(1)], \
    uint8_t a_zero_point,                                        \
    uint8_t b_zero_point,                                        \
    uint8_t output_zero_point,                                   \
    float product_output_scale,                                  \
    uint8_t output_min,                                          \
    uint8_t output_max);

DECLARE_INIT_QU8_MUL_MINMAX_PARAMS_FUNCTION(xnn_init_qu8_mul_minmax_fp32_scalar_params)

#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  DECLARE_INIT_QU8_MUL_MINMAX_PARAMS_FUNCTION(xnn_init_qu8_mul_minmax_fp32_neon_params)
  DECLARE_INIT_QU8_MUL_MINMAX_PARAMS_FUNCTION(xnn_init_qu8_mul_minmax_fp32_neonv8_params)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  DECLARE_INIT_QU8_MUL_MINMAX_PARAMS_FUNCTION(xnn_init_qu8_mul_minmax_fp32_sse2_params)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  DECLARE_INIT_QU8_MUL_MINMAX_PARAMS_FUNCTION(xnn_init_qu8_mul_minmax_fp32_wasmsimd_params)
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#define DECLARE_INIT_QS8_MUL_MINMAX_PARAMS_FUNCTION(fn_name)     \
  XNN_INTERNAL void fn_name(                                     \
    union xnn_qs8_mul_minmax_params params[XNN_MIN_ELEMENTS(1)], \
    int8_t a_zero_point,                                         \
    int8_t b_zero_point,                                         \
    int8_t output_zero_point,                                    \
    float product_output_scale,                                  \
    int8_t output_min,                                           \
    int8_t output_max);

DECLARE_INIT_QS8_MUL_MINMAX_PARAMS_FUNCTION(xnn_init_qs8_mul_minmax_fp32_scalar_params)

#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  DECLARE_INIT_QS8_MUL_MINMAX_PARAMS_FUNCTION(xnn_init_qs8_mul_minmax_fp32_neon_params)
  DECLARE_INIT_QS8_MUL_MINMAX_PARAMS_FUNCTION(xnn_init_qs8_mul_minmax_fp32_neonv8_params)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  DECLARE_INIT_QS8_MUL_MINMAX_PARAMS_FUNCTION(xnn_init_qs8_mul_minmax_fp32_sse2_params)
  DECLARE_INIT_QS8_MUL_MINMAX_PARAMS_FUNCTION(xnn_init_qs8_mul_minmax_fp32_sse4_params)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  DECLARE_INIT_QS8_MUL_MINMAX_PARAMS_FUNCTION(xnn_init_qs8_mul_minmax_fp32_wasmsimd_params)
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#define DECLARE_INIT_F16_F32_CVT_PARAMS_FUNCTION(fn_name)      \
  XNN_INTERNAL void fn_name(                                   \
    union xnn_f16_f32_cvt_params params[XNN_MIN_ELEMENTS(1)]);

DECLARE_INIT_F16_F32_CVT_PARAMS_FUNCTION(xnn_init_f16_f32_cvt_scalar_params)

#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  DECLARE_INIT_F16_F32_CVT_PARAMS_FUNCTION(xnn_init_f16_f32_cvt_neon_params)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  DECLARE_INIT_F16_F32_CVT_PARAMS_FUNCTION(xnn_init_f16_f32_cvt_sse_int16_params)
  DECLARE_INIT_F16_F32_CVT_PARAMS_FUNCTION(xnn_init_f16_f32_cvt_sse_int32_params)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  DECLARE_INIT_F16_F32_CVT_PARAMS_FUNCTION(xnn_init_f16_f32_cvt_wasmsimd_int16_params)
  DECLARE_INIT_F16_F32_CVT_PARAMS_FUNCTION(xnn_init_f16_f32_cvt_wasmsimd_int32_params)
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#define DECLARE_INIT_F32_F16_CVT_PARAMS_FUNCTION(fn_name)      \
  XNN_INTERNAL void fn_name(                                   \
    union xnn_f32_f16_cvt_params params[XNN_MIN_ELEMENTS(1)]);

DECLARE_INIT_F32_F16_CVT_PARAMS_FUNCTION(xnn_init_f32_f16_cvt_scalar_bitcast_params)
DECLARE_INIT_F32_F16_CVT_PARAMS_FUNCTION(xnn_init_f32_f16_cvt_scalar_fabsf_params)

#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  DECLARE_INIT_F32_F16_CVT_PARAMS_FUNCTION(xnn_init_f32_f16_cvt_neon_params)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  DECLARE_INIT_F32_F16_CVT_PARAMS_FUNCTION(xnn_init_f32_f16_cvt_sse2_params)
  DECLARE_INIT_F32_F16_CVT_PARAMS_FUNCTION(xnn_init_f32_f16_cvt_f16c_params)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  DECLARE_INIT_F32_F16_CVT_PARAMS_FUNCTION(xnn_init_f32_f16_cvt_wasmsimd_params)
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#define DECLARE_INIT_F32_QS8_CVT_PARAMS_FUNCTION(fn_name)     \
  XNN_INTERNAL void fn_name(                                  \
    union xnn_f32_qs8_cvt_params params[XNN_MIN_ELEMENTS(1)], \
    float scale,                                              \
    int8_t zero_point,                                        \
    int8_t output_min,                                        \
    int8_t output_max);

DECLARE_INIT_F32_QS8_CVT_PARAMS_FUNCTION(xnn_init_f32_qs8_cvt_scalar_fmagic_params)
DECLARE_INIT_F32_QS8_CVT_PARAMS_FUNCTION(xnn_init_f32_qs8_cvt_scalar_imagic_params)
DECLARE_INIT_F32_QS8_CVT_PARAMS_FUNCTION(xnn_init_f32_qs8_cvt_scalar_lrintf_params)

#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  DECLARE_INIT_F32_QS8_CVT_PARAMS_FUNCTION(xnn_init_f32_qs8_cvt_neon_params)
  DECLARE_INIT_F32_QS8_CVT_PARAMS_FUNCTION(xnn_init_f32_qs8_cvt_neonv8_params)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  DECLARE_INIT_F32_QS8_CVT_PARAMS_FUNCTION(xnn_init_f32_qs8_cvt_sse2_params)
  DECLARE_INIT_F32_QS8_CVT_PARAMS_FUNCTION(xnn_init_f32_qs8_cvt_sse4_params)
  DECLARE_INIT_F32_QS8_CVT_PARAMS_FUNCTION(xnn_init_f32_qs8_cvt_avx_params)
  DECLARE_INIT_F32_QS8_CVT_PARAMS_FUNCTION(xnn_init_f32_qs8_cvt_avx2_params)
  DECLARE_INIT_F32_QS8_CVT_PARAMS_FUNCTION(xnn_init_f32_qs8_cvt_avx512_params)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  DECLARE_INIT_F32_QS8_CVT_PARAMS_FUNCTION(xnn_init_f32_qs8_cvt_wasmsimd_cvt_params)
  DECLARE_INIT_F32_QS8_CVT_PARAMS_FUNCTION(xnn_init_f32_qs8_cvt_wasmsimd_magic_params)
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#define DECLARE_INIT_F32_QU8_CVT_PARAMS_FUNCTION(fn_name)     \
  XNN_INTERNAL void fn_name(                                  \
    union xnn_f32_qu8_cvt_params params[XNN_MIN_ELEMENTS(1)], \
    float scale,                                              \
    uint8_t zero_point,                                       \
    uint8_t output_min,                                       \
    uint8_t output_max);

DECLARE_INIT_F32_QU8_CVT_PARAMS_FUNCTION(xnn_init_f32_qu8_cvt_scalar_fmagic_params)
DECLARE_INIT_F32_QU8_CVT_PARAMS_FUNCTION(xnn_init_f32_qu8_cvt_scalar_imagic_params)
DECLARE_INIT_F32_QU8_CVT_PARAMS_FUNCTION(xnn_init_f32_qu8_cvt_scalar_lrintf_params)

#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  DECLARE_INIT_F32_QU8_CVT_PARAMS_FUNCTION(xnn_init_f32_qu8_cvt_neon_params)
  DECLARE_INIT_F32_QU8_CVT_PARAMS_FUNCTION(xnn_init_f32_qu8_cvt_neonv8_params)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  DECLARE_INIT_F32_QU8_CVT_PARAMS_FUNCTION(xnn_init_f32_qu8_cvt_sse2_params)
  DECLARE_INIT_F32_QU8_CVT_PARAMS_FUNCTION(xnn_init_f32_qu8_cvt_avx_params)
  DECLARE_INIT_F32_QU8_CVT_PARAMS_FUNCTION(xnn_init_f32_qu8_cvt_avx2_params)
  DECLARE_INIT_F32_QU8_CVT_PARAMS_FUNCTION(xnn_init_f32_qu8_cvt_avx512_params)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  DECLARE_INIT_F32_QU8_CVT_PARAMS_FUNCTION(xnn_init_f32_qu8_cvt_wasmsimd_cvt_params)
  DECLARE_INIT_F32_QU8_CVT_PARAMS_FUNCTION(xnn_init_f32_qu8_cvt_wasmsimd_magic_params)
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#define DECLARE_INIT_QS8_F32_CVT_PARAMS_FUNCTION(fn_name)     \
  XNN_INTERNAL void fn_name(                                  \
    union xnn_qs8_f32_cvt_params params[XNN_MIN_ELEMENTS(1)], \
    float scale,                                              \
    int8_t zero_point);

DECLARE_INIT_QS8_F32_CVT_PARAMS_FUNCTION(xnn_init_qs8_f32_cvt_scalar_params)

#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  DECLARE_INIT_QS8_F32_CVT_PARAMS_FUNCTION(xnn_init_qs8_f32_cvt_neon_params)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  DECLARE_INIT_QS8_F32_CVT_PARAMS_FUNCTION(xnn_init_qs8_f32_cvt_sse2_params)
  DECLARE_INIT_QS8_F32_CVT_PARAMS_FUNCTION(xnn_init_qs8_f32_cvt_sse4_params)
  DECLARE_INIT_QS8_F32_CVT_PARAMS_FUNCTION(xnn_init_qs8_f32_cvt_avx_params)
  DECLARE_INIT_QS8_F32_CVT_PARAMS_FUNCTION(xnn_init_qs8_f32_cvt_avx512_params)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  DECLARE_INIT_QS8_F32_CVT_PARAMS_FUNCTION(xnn_init_qs8_f32_cvt_wasmsimd_params)
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#define DECLARE_INIT_QU8_F32_CVT_PARAMS_FUNCTION(fn_name)     \
  XNN_INTERNAL void fn_name(                                  \
    union xnn_qu8_f32_cvt_params params[XNN_MIN_ELEMENTS(1)], \
    float scale,                                              \
    uint8_t zero_point);

DECLARE_INIT_QU8_F32_CVT_PARAMS_FUNCTION(xnn_init_qu8_f32_cvt_scalar_params)

#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  DECLARE_INIT_QU8_F32_CVT_PARAMS_FUNCTION(xnn_init_qu8_f32_cvt_neon_params)
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  DECLARE_INIT_QU8_F32_CVT_PARAMS_FUNCTION(xnn_init_qu8_f32_cvt_sse2_params)
  DECLARE_INIT_QU8_F32_CVT_PARAMS_FUNCTION(xnn_init_qu8_f32_cvt_sse4_params)
  DECLARE_INIT_QU8_F32_CVT_PARAMS_FUNCTION(xnn_init_qu8_f32_cvt_avx_params)
  DECLARE_INIT_QU8_F32_CVT_PARAMS_FUNCTION(xnn_init_qu8_f32_cvt_avx512_params)
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64

#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  DECLARE_INIT_QU8_F32_CVT_PARAMS_FUNCTION(xnn_init_qu8_f32_cvt_wasmsimd_params)
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD

#ifdef __cplusplus
}  // extern "C"
#endif
