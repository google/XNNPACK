// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <stdbool.h>
#include <stddef.h>

#include "xnnpack/common.h"
#include "xnnpack/config-types.h"
#include "xnnpack/hardware-config.h"
#include "xnnpack/microfnptr.h"
#include "xnnpack/microparams.h"

#ifdef __cplusplus
extern "C" {
#endif

XNN_INTERNAL const struct xnn_x8_lut_config* xnn_init_x8_lut_config();

XNN_INTERNAL const struct xnn_transpose_config* xnn_init_transpose_config();

XNN_INTERNAL const struct xnn_cmul_config* xnn_init_f16_cmul_config();
XNN_INTERNAL const struct xnn_cmul_config* xnn_init_f32_cmul_config();

XNN_INTERNAL const struct xnn_binary_elementwise_config* xnn_init_f16_vadd_config();
XNN_INTERNAL const struct xnn_binary_elementwise_config* xnn_init_f16_vdiv_config();
XNN_INTERNAL const struct xnn_binary_elementwise_config* xnn_init_f16_vmax_config();
XNN_INTERNAL const struct xnn_binary_elementwise_config* xnn_init_f16_vmin_config();
XNN_INTERNAL const struct xnn_binary_elementwise_config* xnn_init_f16_vmul_config();
XNN_INTERNAL const struct xnn_binary_elementwise_config* xnn_init_f16_vsub_config();
XNN_INTERNAL const struct xnn_binary_elementwise_config* xnn_init_f16_vsqrdiff_config();
XNN_INTERNAL const struct xnn_binary_elementwise_config* xnn_init_f32_vadd_config();
XNN_INTERNAL const struct xnn_binary_elementwise_config* xnn_init_f32_vcopysign_config();
XNN_INTERNAL const struct xnn_binary_elementwise_config* xnn_init_s32_vmul_config();
XNN_INTERNAL const struct xnn_binary_elementwise_config* xnn_init_f32_vdiv_config();
XNN_INTERNAL const struct xnn_binary_elementwise_config* xnn_init_f32_vmax_config();
XNN_INTERNAL const struct xnn_binary_elementwise_config* xnn_init_f32_vmin_config();
XNN_INTERNAL const struct xnn_binary_elementwise_config* xnn_init_f32_vmul_config();
XNN_INTERNAL const struct xnn_binary_elementwise_config* xnn_init_f32_vsub_config();
XNN_INTERNAL const struct xnn_binary_elementwise_config* xnn_init_f32_vsqrdiff_config();
XNN_INTERNAL const struct xnn_binary_elementwise_config* xnn_init_qs8_vadd_config();
XNN_INTERNAL const struct xnn_binary_elementwise_config* xnn_init_qs8_vmul_config();
XNN_INTERNAL const struct xnn_binary_elementwise_config* xnn_init_qu8_vadd_config();
XNN_INTERNAL const struct xnn_binary_elementwise_config* xnn_init_qu8_vmul_config();

XNN_INTERNAL const struct xnn_unary_elementwise_config* xnn_init_f16_abs_config();
XNN_INTERNAL const struct xnn_unary_elementwise_config* xnn_init_f16_clamp_config();
XNN_INTERNAL const struct xnn_unary_elementwise_config* xnn_init_f16_elu_config();
XNN_INTERNAL const struct xnn_unary_elementwise_config* xnn_init_f16_hswish_config();
XNN_INTERNAL const struct xnn_unary_elementwise_config* xnn_init_f16_lrelu_config();
XNN_INTERNAL const struct xnn_unary_elementwise_config* xnn_init_f16_neg_config();
XNN_INTERNAL const struct xnn_unary_elementwise_config* xnn_init_f16_relu_config();
XNN_INTERNAL const struct xnn_unary_elementwise_config* xnn_init_f16_rndd_config();
XNN_INTERNAL const struct xnn_unary_elementwise_config* xnn_init_f16_rndne_config();
XNN_INTERNAL const struct xnn_unary_elementwise_config* xnn_init_f16_rndu_config();
XNN_INTERNAL const struct xnn_unary_elementwise_config* xnn_init_f16_rndz_config();
XNN_INTERNAL const struct xnn_unary_elementwise_config* xnn_init_f16_rsqrt_config();
XNN_INTERNAL const struct xnn_unary_elementwise_config* xnn_init_f16_sigmoid_config();
XNN_INTERNAL const struct xnn_unary_elementwise_config* xnn_init_f16_sqr_config();
XNN_INTERNAL const struct xnn_unary_elementwise_config* xnn_init_f16_sqrt_config();
XNN_INTERNAL const struct xnn_unary_elementwise_config* xnn_init_f16_tanh_config();
XNN_INTERNAL const struct xnn_unary_elementwise_config* xnn_init_f16_to_f32_cvt_config();
XNN_INTERNAL const struct xnn_unary_elementwise_config* xnn_init_f16_to_qs8_cvt_config();
XNN_INTERNAL const struct xnn_unary_elementwise_config* xnn_init_f32_abs_config();
XNN_INTERNAL const struct xnn_unary_elementwise_config* xnn_init_f32_clamp_config();
XNN_INTERNAL const struct xnn_unary_elementwise_config* xnn_init_f32_elu_config();
XNN_INTERNAL const struct xnn_unary_elementwise_config* xnn_init_f32_exp_config();
XNN_INTERNAL const struct xnn_unary_elementwise_config* xnn_init_f32_gelu_config();
XNN_INTERNAL const struct xnn_unary_elementwise_config* xnn_init_f32_hswish_config();
XNN_INTERNAL const struct xnn_unary_elementwise_config* xnn_init_f32_log_config();
XNN_INTERNAL const struct xnn_unary_elementwise_config* xnn_init_f32_lrelu_config();
XNN_INTERNAL const struct xnn_unary_elementwise_config* xnn_init_f32_neg_config();
XNN_INTERNAL const struct xnn_unary_elementwise_config* xnn_init_f32_relu_config();
XNN_INTERNAL const struct xnn_unary_elementwise_config* xnn_init_f32_rndd_config();
XNN_INTERNAL const struct xnn_unary_elementwise_config* xnn_init_f32_rndne_config();
XNN_INTERNAL const struct xnn_unary_elementwise_config* xnn_init_f32_rndu_config();
XNN_INTERNAL const struct xnn_unary_elementwise_config* xnn_init_f32_rndz_config();
XNN_INTERNAL const struct xnn_unary_elementwise_config* xnn_init_f32_rsqrt_config();
XNN_INTERNAL const struct xnn_unary_elementwise_config* xnn_init_f32_sigmoid_config();
XNN_INTERNAL const struct xnn_unary_elementwise_config* xnn_init_f32_sqr_config();
XNN_INTERNAL const struct xnn_unary_elementwise_config* xnn_init_f32_sqrt_config();
XNN_INTERNAL const struct xnn_unary_elementwise_config* xnn_init_f32_tanh_config();
XNN_INTERNAL const struct xnn_unary_elementwise_config* xnn_init_f32_to_f16_cvt_config();
XNN_INTERNAL const struct xnn_unary_elementwise_config*
xnn_init_f32_to_qp8_cvt_config();
XNN_INTERNAL const struct xnn_unary_elementwise_config* xnn_init_f32_to_qs8_cvt_config();
XNN_INTERNAL const struct xnn_unary_elementwise_config* xnn_init_f32_to_qu8_cvt_config();
XNN_INTERNAL const struct xnn_unary_elementwise_config* xnn_init_qs8_cvt_config();
XNN_INTERNAL const struct xnn_unary_elementwise_config* xnn_init_qs8_hswish_config();
XNN_INTERNAL const struct xnn_unary_elementwise_config* xnn_init_qs8_lrelu_config();
XNN_INTERNAL const struct xnn_unary_elementwise_config* xnn_init_qs8_to_f16_cvt_config();
XNN_INTERNAL const struct xnn_unary_elementwise_config* xnn_init_qs8_to_f32_cvt_config();
XNN_INTERNAL const struct xnn_unary_elementwise_config* xnn_init_qs16_to_qs8_cvt_config();
XNN_INTERNAL const struct xnn_unary_elementwise_config* xnn_init_qu8_cvt_config();
XNN_INTERNAL const struct xnn_unary_elementwise_config* xnn_init_qu8_hswish_config();
XNN_INTERNAL const struct xnn_unary_elementwise_config* xnn_init_qu8_lrelu_config();
XNN_INTERNAL const struct xnn_unary_elementwise_config* xnn_init_qu8_to_f32_cvt_config();
XNN_INTERNAL const struct xnn_unary_elementwise_config* xnn_init_s8_clamp_config();
XNN_INTERNAL const struct xnn_unary_elementwise_config* xnn_init_u8_clamp_config();
XNN_INTERNAL const struct xnn_unary_elementwise_config* xnn_init_xx_copy_config();

XNN_INTERNAL const struct xnn_reduce_config* xnn_init_f16_f32acc_rsum_config();
XNN_INTERNAL const struct xnn_reduce_config* xnn_init_f16_f32acc_rdsum_config();
XNN_INTERNAL const struct xnn_reduce_config* xnn_init_f16_rminmax_config();
XNN_INTERNAL const struct xnn_reduce_config* xnn_init_f32_rminmax_config();
XNN_INTERNAL const struct xnn_reduce_config* xnn_init_f32_rsum_config();
XNN_INTERNAL const struct xnn_reduce_config* xnn_init_f32_rdsum_config();

XNN_INTERNAL const struct xnn_xx_fill_config* xnn_init_xx_fill_config();

XNN_INTERNAL const struct xnn_xx_pad_config* xnn_init_xx_pad_config();

XNN_INTERNAL const struct xnn_avgpool_config* xnn_init_f16_avgpool_config();
XNN_INTERNAL const struct xnn_avgpool_config* xnn_init_f32_avgpool_config();
XNN_INTERNAL const struct xnn_avgpool_config* xnn_init_qu8_avgpool_config();

XNN_INTERNAL const struct xnn_pavgpool_config* xnn_init_f16_pavgpool_config();
XNN_INTERNAL const struct xnn_pavgpool_config* xnn_init_f32_pavgpool_config();

XNN_INTERNAL const struct xnn_gavgpool_config* xnn_init_f16_gavgpool_config();
XNN_INTERNAL const struct xnn_gavgpool_config* xnn_init_f32_gavgpool_config();
XNN_INTERNAL const struct xnn_gavgpool_config* xnn_init_qs8_gavgpool_config();
XNN_INTERNAL const struct xnn_gavgpool_config* xnn_init_qu8_gavgpool_config();

XNN_INTERNAL const struct xnn_gavgpool_cw_config* xnn_init_f16_gavgpool_cw_config();
XNN_INTERNAL const struct xnn_gavgpool_cw_config* xnn_init_f32_gavgpool_cw_config();

#define XNN_MAX_F16_DWCONV_UKERNELS 4
#define XNN_MAX_F32_DWCONV_UKERNELS 4
#define XNN_MAX_QC8_DWCONV_UKERNELS 3
#define XNN_MAX_QS8_DWCONV_UKERNELS 2
#define XNN_MAX_QU8_DWCONV_UKERNELS 2

XNN_INTERNAL struct xnn_dwconv_config* xnn_init_f16_dwconv_config();
XNN_INTERNAL struct xnn_dwconv_config* xnn_init_f32_dwconv_config();
XNN_INTERNAL struct xnn_dwconv_config* xnn_init_qs8_qc8w_dwconv_config();
XNN_INTERNAL struct xnn_dwconv_config* xnn_init_qs8_dwconv_config();
XNN_INTERNAL struct xnn_dwconv_config* xnn_init_qu8_dwconv_config();

// Bilinear interpolation (2D).
XNN_INTERNAL const struct xnn_ibilinear_config* xnn_init_f16_ibilinear_config();
XNN_INTERNAL const struct xnn_ibilinear_config* xnn_init_f32_ibilinear_config();
XNN_INTERNAL const struct xnn_ibilinear_config* xnn_init_s8_ibilinear_config();
XNN_INTERNAL const struct xnn_ibilinear_config* xnn_init_u8_ibilinear_config();

// Bilinear interpolation (2D) in CHW layout.
XNN_INTERNAL const struct xnn_ibilinear_chw_config* xnn_init_f16_ibilinear_chw_config();
XNN_INTERNAL const struct xnn_ibilinear_chw_config* xnn_init_f32_ibilinear_chw_config();

XNN_INTERNAL const struct xnn_prelu_config* xnn_init_f16_prelu_config();
XNN_INTERNAL const struct xnn_prelu_config* xnn_init_f32_prelu_config();

static inline struct xnn_hmp_dqgemm_ukernel xnn_init_hmp_dqgemm_ukernel(
    xnn_dqgemm_ukernel_fn function) {
  struct xnn_hmp_dqgemm_ukernel ukernel = {{ function }};
  for (size_t i = 1; i < XNN_MAX_UARCH_TYPES; i++) {
    ukernel.function[i] = function;
  }
  return ukernel;
}

static inline struct xnn_hmp_dqgemm_bl_ukernel xnn_init_hmp_dqgemm_bl_ukernel(
    xnn_dqgemm_bl_ukernel_fn function) {
  struct xnn_hmp_dqgemm_bl_ukernel ukernel;// = {{ function }};
  for (size_t i = 0; i < XNN_MAX_UARCH_TYPES; i++) {
    ukernel.function[i] = function;
  }
  return ukernel;
}

static inline struct xnn_hmp_dqigemm_ukernel xnn_init_hmp_dqigemm_ukernel(
    xnn_dqigemm_ukernel_fn function) {
  struct xnn_hmp_dqigemm_ukernel ukernel = {{function}};
  for (size_t i = 1; i < XNN_MAX_UARCH_TYPES; i++) {
    ukernel.function[i] = function;
  }
  return ukernel;
}

static inline struct xnn_hmp_qp8gemm_ukernel xnn_init_hmp_qp8gemm_ukernel(
    xnn_qp8_f32_qc4w_gemm_minmax_ukernel_fn function) {
  struct xnn_hmp_qp8gemm_ukernel ukernel = {{function}};
  for (size_t i = 1; i < XNN_MAX_UARCH_TYPES; i++) {
    ukernel.function[i] = function;
  }
  return ukernel;
}

static inline struct xnn_hmp_gemm_ukernel xnn_init_hmp_gemm_ukernel(xnn_gemm_ukernel_fn function) {
  struct xnn_hmp_gemm_ukernel ukernel = {{ function }};
  for (size_t i = 1; i < XNN_MAX_UARCH_TYPES; i++) {
    ukernel.function[i] = function;
  }
  return ukernel;
}

static inline bool xnn_is_hmp_gemm_ukernel(struct xnn_hmp_gemm_ukernel ukernel) {
#if XNN_MAX_UARCH_TYPES == 1
  return false;
#else
  uintptr_t default_fn = (uintptr_t) ukernel.function[XNN_UARCH_DEFAULT];
  uintptr_t difference = 0;
  for (size_t i = 1; i < XNN_MAX_UARCH_TYPES; i++) {
    difference |= (default_fn ^ (uintptr_t) ukernel.function[i]);
  }
  return difference != 0;
#endif
}

static inline struct xnn_hmp_igemm_ukernel xnn_init_hmp_igemm_ukernel(xnn_igemm_ukernel_fn function) {
  struct xnn_hmp_igemm_ukernel ukernel = {{ function }};
  for (size_t i = 1; i < XNN_MAX_UARCH_TYPES; i++) {
    ukernel.function[i] = function;
  }
  return ukernel;
}

static inline bool xnn_is_hmp_igemm_ukernel(struct xnn_hmp_igemm_ukernel ukernel) {
#if XNN_MAX_UARCH_TYPES == 1
  return false;
#else
  uintptr_t default_fn = (uintptr_t) ukernel.function[XNN_UARCH_DEFAULT];
  uintptr_t difference = 0;
  for (size_t i = 1; i < XNN_MAX_UARCH_TYPES; i++) {
    difference |= (default_fn ^ (uintptr_t) ukernel.function[i]);
  }
  return difference != 0;
#endif
}

XNN_INTERNAL struct xnn_gemm_config* xnn_init_f16_gemm_config();
XNN_INTERNAL struct xnn_gemm_config* xnn_init_f32_gemm_config();
XNN_INTERNAL struct xnn_gemm_config* xnn_init_f32_gemm_nr2_config();
XNN_INTERNAL struct xnn_gemm_config* xnn_init_f32_qc8w_gemm_config();
XNN_INTERNAL struct xnn_gemm_config* xnn_init_f32_qc4w_gemm_config();
XNN_INTERNAL struct xnn_gemm_config* xnn_init_qd8_f16_qb4w_gemm_config();
XNN_INTERNAL struct xnn_gemm_config* xnn_init_qd8_f16_qc4w_gemm_config();
XNN_INTERNAL struct xnn_gemm_config* xnn_init_qd8_f16_qc8w_gemm_config();
XNN_INTERNAL struct xnn_gemm_config* xnn_init_qd8_f32_qb4w_gemm_config();
XNN_INTERNAL struct xnn_gemm_config* xnn_init_qd8_f32_qc4w_gemm_config();
XNN_INTERNAL struct xnn_gemm_config* xnn_init_qd8_f32_qc8w_gemm_config();
XNN_INTERNAL struct xnn_gemm_config* xnn_init_qp8_f32_qc4w_gemm_config();
XNN_INTERNAL struct xnn_gemm_config* xnn_init_qs8_qc8w_gemm_config();
XNN_INTERNAL struct xnn_gemm_config* xnn_init_qu8_gemm_config();

XNN_INTERNAL const struct xnn_maxpool_config* xnn_init_f16_maxpool_config();
XNN_INTERNAL const struct xnn_maxpool_config* xnn_init_f32_maxpool_config();
XNN_INTERNAL const struct xnn_maxpool_config* xnn_init_s8_maxpool_config();
XNN_INTERNAL const struct xnn_maxpool_config* xnn_init_u8_maxpool_config();

XNN_INTERNAL const struct xnn_zip_config* xnn_init_x8_zip_config();
XNN_INTERNAL const struct xnn_zip_config* xnn_init_x32_zip_config();

XNN_INTERNAL const struct xnn_rmax_config* xnn_init_f16_rmax_config();
XNN_INTERNAL const struct xnn_rmax_config* xnn_init_f32_rmax_config();
XNN_INTERNAL const struct xnn_rmax_config* xnn_init_u8_rmax_config();

// Sparse Matrix-Dense Matrix Multiplication (NR=1 block).
XNN_INTERNAL const struct xnn_spmm_config* xnn_init_f16_spmm_config();
XNN_INTERNAL const struct xnn_spmm_config* xnn_init_f32_spmm_config();
// Sparse Matrix-Dense Matrix Multiplication (NR=2 block).
XNN_INTERNAL const struct xnn_spmm_config* xnn_init_f32_spmm2_config();
// Sparse Matrix-Dense Matrix Multiplication (NR=4 block).
XNN_INTERNAL const struct xnn_spmm_config* xnn_init_f32_spmm4_config();

XNN_INTERNAL const struct xnn_dwconv2d_chw_config* xnn_init_f16_dwconv2d_chw_config();
XNN_INTERNAL const struct xnn_dwconv2d_chw_config* xnn_init_f32_dwconv2d_chw_config();

// Direct 3x3 stride-2 Convolution with 3 input channels and HWC->CHW layout conversion.
XNN_INTERNAL const struct xnn_conv_hwc2chw_config* xnn_init_f16_conv_hwc2chw_3x3c3s2_config();
XNN_INTERNAL const struct xnn_conv_hwc2chw_config* xnn_init_f32_conv_hwc2chw_3x3c3s2_config();

XNN_INTERNAL const struct xnn_vmulcaddc_config* xnn_init_f16_vmulcaddc_config();
XNN_INTERNAL const struct xnn_vmulcaddc_config* xnn_init_f32_vmulcaddc_config();

XNN_INTERNAL const struct xnn_raddstoreexpminusmax_config* xnn_init_f16_raddstoreexpminusmax_config();
XNN_INTERNAL const struct xnn_raddstoreexpminusmax_config* xnn_init_f32_raddstoreexpminusmax_config();

#define XNN_MAX_F32_ARGMAXPOOL_UKERNELS 3

XNN_INTERNAL const struct xnn_argmaxpool_config* xnn_init_f32_argmaxpool_config();

XNN_INTERNAL const struct xnn_lut32norm_config* xnn_init_u8_lut32norm_config();

XNN_INTERNAL const struct xnn_unpool_config* xnn_init_x32_unpool_config();

#ifdef __cplusplus
}  // extern "C"
#endif
