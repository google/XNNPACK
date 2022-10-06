// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include <xnnpack.h>
#include <xnnpack/common.h>
#include <xnnpack/microfnptr.h>
#include <xnnpack/microparams.h>

struct xnn_hmp_gemm_ukernel {
  xnn_gemm_ukernel_function function[XNN_MAX_UARCH_TYPES];
#if XNN_PLATFORM_JIT
  size_t generated_code_offset[XNN_MAX_UARCH_TYPES];
#endif  // XNN_PLATFORM_JIT
};

static inline struct xnn_hmp_gemm_ukernel xnn_init_hmp_gemm_ukernel(xnn_gemm_ukernel_function function) {
  struct xnn_hmp_gemm_ukernel ukernel = {{ function }};
  for (size_t i = 1; i < XNN_MAX_UARCH_TYPES; i++) {
    ukernel.function[i] = function;
#if XNN_PLATFORM_JIT
    ukernel.generated_code_offset[i] = SIZE_MAX;
#endif  // XNN_PLATFORM_JIT
  }
  return ukernel;
}

static inline bool xnn_is_hmp_gemm_ukernel(struct xnn_hmp_gemm_ukernel ukernel) {
#if XNN_MAX_UARCH_TYPES == 1
  return false;
#else
  uintptr_t default_function = (uintptr_t) ukernel.function[XNN_UARCH_DEFAULT];
  uintptr_t difference = 0;
  for (size_t i = 1; i < XNN_MAX_UARCH_TYPES; i++) {
    difference |= (default_function ^ (uintptr_t) ukernel.function[i]);
  }
  return difference != 0;
#endif
}

struct xnn_hmp_igemm_ukernel {
  xnn_igemm_ukernel_function function[XNN_MAX_UARCH_TYPES];
#if XNN_PLATFORM_JIT
  size_t generated_code_offset[XNN_MAX_UARCH_TYPES];
#endif  // XNN_PLATFORM_JIT
};

static inline struct xnn_hmp_igemm_ukernel xnn_init_hmp_igemm_ukernel(xnn_igemm_ukernel_function function) {
  struct xnn_hmp_igemm_ukernel ukernel = {{ function }};
  for (size_t i = 1; i < XNN_MAX_UARCH_TYPES; i++) {
    ukernel.function[i] = function;
#if XNN_PLATFORM_JIT
    ukernel.generated_code_offset[i] = SIZE_MAX;
#endif  // XNN_PLATFORM_JIT
  }
  return ukernel;
}

static inline bool xnn_is_hmp_igemm_ukernel(struct xnn_hmp_igemm_ukernel ukernel) {
#if XNN_MAX_UARCH_TYPES == 1
  return false;
#else
  uintptr_t default_function = (uintptr_t) ukernel.function[XNN_UARCH_DEFAULT];
  uintptr_t difference = 0;
  for (size_t i = 1; i < XNN_MAX_UARCH_TYPES; i++) {
    difference |= (default_function ^ (uintptr_t) ukernel.function[i]);
  }
  return difference != 0;
#endif
}

// Largest GEMM/IGEMM MR used in init.c is 7 (x86 AVX512).
// Largest GEMM/IGEMM MR is 8 in e2e benchmarks.
#define XNN_MAX_MR 8

struct gemm_fused_ukernels {
  struct xnn_hmp_gemm_ukernel gemm[XNN_MAX_MR];
  struct xnn_hmp_igemm_ukernel igemm[XNN_MAX_MR];
};

struct transpose_parameters {
  union {
    xnn_transposec_ukernel_function const_size_ukernel;
    xnn_transposev_ukernel_function variable_size_ukernel;
  };
  // Maximum number of elements to process per ukernel call.
  uint8_t tile_size;
};

#if XNN_PLATFORM_JIT
struct xnn_hmp_gemm_codegen {
  xnn_jit_gemm_code_generator_function function[XNN_MAX_UARCH_TYPES];
};

static inline struct xnn_hmp_gemm_codegen xnn_init_hmp_gemm_codegen(xnn_jit_gemm_code_generator_function function) {
  struct xnn_hmp_gemm_codegen ukernel = {{ function }};
  for (size_t i = 1; i < XNN_MAX_UARCH_TYPES; i++) {
    ukernel.function[i] = function;
  }
  return ukernel;
}

static inline bool xnn_is_hmp_gemm_codegen(struct xnn_hmp_gemm_codegen ukernel) {
#if XNN_MAX_UARCH_TYPES == 1
  return false;
#else
  uintptr_t default_function = (uintptr_t) ukernel.function[XNN_UARCH_DEFAULT];
  uintptr_t difference = 0;
  for (size_t i = 1; i < XNN_MAX_UARCH_TYPES; i++) {
    difference |= (default_function ^ (uintptr_t) ukernel.function[i]);
  }
  return difference != 0;
#endif
}

struct xnn_hmp_igemm_codegen {
  xnn_jit_igemm_code_generator_function function[XNN_MAX_UARCH_TYPES];
};

static inline struct xnn_hmp_igemm_codegen xnn_init_hmp_igemm_codegen(xnn_jit_igemm_code_generator_function function) {
  struct xnn_hmp_igemm_codegen ukernel = {{ function }};
  for (size_t i = 1; i < XNN_MAX_UARCH_TYPES; i++) {
    ukernel.function[i] = function;
  }
  return ukernel;
}

static inline bool xnn_is_hmp_igemm_codegen(struct xnn_hmp_igemm_codegen ukernel) {
#if XNN_MAX_UARCH_TYPES == 1
  return false;
#else
  uintptr_t default_function = (uintptr_t) ukernel.function[XNN_UARCH_DEFAULT];
  uintptr_t difference = 0;
  for (size_t i = 1; i < XNN_MAX_UARCH_TYPES; i++) {
    difference |= (default_function ^ (uintptr_t) ukernel.function[i]);
  }
  return difference != 0;
#endif
}

struct gemm_codegens {
  struct xnn_hmp_gemm_codegen gemm;
  struct xnn_hmp_igemm_codegen igemm;
  // Optional JIT GEMM and IGEMM micro-kernels with MR=1 and the same NR and KR parameters.
  struct xnn_hmp_gemm_codegen gemm1;
  struct xnn_hmp_igemm_codegen igemm1;
};
#endif  // XNN_PLATFORM_JIT

struct gemm_parameters {
  struct gemm_fused_ukernels minmax;
  struct gemm_fused_ukernels relu;
  struct gemm_fused_ukernels linear;
#if XNN_PLATFORM_JIT
  struct gemm_codegens generator;
#endif  // XNN_PLATFORM_JIT
  union {
    xnn_init_qc8_conv_minmax_params_fn qc8;
    xnn_init_qs8_conv_minmax_params_fn qs8;
    xnn_init_qu8_conv_minmax_params_fn qu8;
    xnn_init_f16_minmax_params_fn f16;
    xnn_init_f32_minmax_params_fn f32;
  } init;
  uint8_t mr;
  uint8_t nr;
  uint8_t log2_kr;
  uint8_t log2_sr;
};

struct vunary_parameters {
  xnn_vunary_ukernel_function ukernel;
  union {
    xnn_init_f16_f32_cvt_params_fn f16_f32_cvt;
    xnn_init_f16_abs_params_fn f16_abs;
    xnn_init_f16_elu_params_fn f16_elu;
    xnn_init_f16_hswish_params_fn f16_hswish;
    xnn_init_f16_lrelu_params_fn f16_lrelu;
    xnn_init_f16_neg_params_fn f16_neg;
    xnn_init_f16_minmax_params_fn f16_minmax;
    xnn_init_f16_sigmoid_params_fn f16_sigmoid;
    xnn_init_f16_sqrt_params_fn f16_sqrt;
    xnn_init_f32_abs_params_fn f32_abs;
    xnn_init_f32_default_params_fn f32_default;
    xnn_init_f32_elu_params_fn f32_elu;
    xnn_init_f32_f16_cvt_params_fn f32_f16_cvt;
    xnn_init_f32_hswish_params_fn f32_hswish;
    xnn_init_f32_lrelu_params_fn f32_lrelu;
    xnn_init_f32_minmax_params_fn f32_minmax;
    xnn_init_f32_neg_params_fn f32_neg;
    xnn_init_f32_qs8_cvt_params_fn f32_qs8_cvt;
    xnn_init_f32_qu8_cvt_params_fn f32_qu8_cvt;
    xnn_init_f32_rnd_params_fn f32_rnd;
    xnn_init_f32_sigmoid_params_fn f32_sigmoid;
    xnn_init_f32_sqrt_params_fn f32_sqrt;
    xnn_init_qs8_cvt_params_fn qs8_cvt;
    xnn_init_qs8_f32_cvt_params_fn qs8_f32_cvt;
    xnn_init_qs8_lrelu_params_fn qs8_lrelu;
    xnn_init_qu8_cvt_params_fn qu8_cvt;
    xnn_init_qu8_f32_cvt_params_fn qu8_f32_cvt;
    xnn_init_qu8_lrelu_params_fn qu8_lrelu;
    xnn_init_s8_minmax_params_fn s8_minmax;
    xnn_init_u8_minmax_params_fn u8_minmax;
  } init;
  // Number of elements in a tile.
  // For best efficiency, micro-kernel must process a multiple of this number of elements in each call.
  uint8_t element_tile;
};

struct vbinary_fused_ukernels {
  xnn_vbinary_ukernel_function op_ukernel;
  xnn_vbinary_ukernel_function opc_ukernel;
  xnn_vbinary_ukernel_function ropc_ukernel;
};

struct vbinary_parameters {
  struct vbinary_fused_ukernels minmax;
  struct vbinary_fused_ukernels linear;
  union {
    xnn_init_f16_minmax_params_fn f16_minmax;
    xnn_init_f32_default_params_fn f32_default;
    xnn_init_f32_minmax_params_fn f32_minmax;
    xnn_init_qs8_add_minmax_params_fn qs8_add;
    xnn_init_qs8_mul_minmax_params_fn qs8_mul;
    xnn_init_qu8_add_minmax_params_fn qu8_add;
    xnn_init_qu8_mul_minmax_params_fn qu8_mul;
  } init;
  // Number of elements in a tile.
  // For best efficiency, micro-kernel must process a multiple of this number of elements in each call.
  uint8_t element_tile;
};

struct spmm_parameters {
  xnn_spmm_ukernel_function ukernel;
  // Number of M-dimension elements in a tile.
  // Corresponds to a block of pixels in 1x1 Convolution and a block of batch size in Fully Connected operator.
  uint8_t mr;
  // Number of N-dimension elements in a tile.
  // Corresponds to a block of output channels/features in 1x1 Convolution and Fully Connected operator.
  uint8_t nr;
};

struct conv_hwc2chw_parameters {
  xnn_conv_hwc2chw_ukernel_function ukernel_with_symm_padding;
  // Number of output channels in a tile.
  // This parameter must be passed as is to weight packing function.
  uint8_t output_channel_tile;
  // Number of output height pixels in a tile.
  // For best efficiency, micro-kernel must produce a multiple of this number of rows in each call.
  uint8_t output_height_tile;
  // Number of output width pixels in a tile.
  uint8_t output_width_tile;
};

struct dwconv2d_chw_parameters {
  xnn_dwconv2d_chw_ukernel_function ukernel;
  // Number of output width pixels in a tile.
  uint8_t output_width_tile;
  // Number of output height pixels in a tile.
  // For best efficiency, micro-kernel must produce a multiple of this number of rows in each call.
  uint8_t output_height_tile;
};

struct gavgpool_cw_parameters {
  xnn_gavgpool_cw_ukernel_function ukernel;
  // Number of channels in a tile.
  // For best efficiency, micro-kernel must process a multiple of this number of channels in each call.
  uint8_t channel_tile;
};

union dwconv_fused_ukernels {
  xnn_dwconv_unipass_ukernel_function unipass;
  xnn_dwconv_multipass_ukernel_function multipass;
};

struct dwconv_parameters {
  union dwconv_fused_ukernels minmax;
  union dwconv_fused_ukernels linear;
  union {
    xnn_init_qc8_conv_minmax_params_fn qc8;
    xnn_init_qs8_conv_minmax_params_fn qs8;
    xnn_init_qu8_conv_minmax_params_fn qu8;
    xnn_init_f16_minmax_params_fn f16;
    xnn_init_f32_minmax_params_fn f32;
  } init;
  uint8_t channel_tile;
  uint8_t primary_tile;
  uint8_t incremental_tile;
};

struct gavgpool_parameters {
  xnn_gavgpool_unipass_ukernel_function unipass;
  xnn_gavgpool_multipass_ukernel_function multipass;
  union {
    xnn_init_f16_scaleminmax_params_fn f16;
    xnn_init_f32_scaleminmax_params_fn f32;
    xnn_init_qs8_avgpool_minmax_params_fn qs8;
    xnn_init_qu8_avgpool_minmax_params_fn qu8;
  } init;
  union {
    xnn_update_f16_scaleminmax_params_fn f16;
    xnn_update_f32_scaleminmax_params_fn f32;
    xnn_update_qs8_avgpool_minmax_params_fn qs8;
    xnn_update_qu8_avgpool_minmax_params_fn qu8;
  } update;
  // Number of rows in a tile.
  // For best efficiency, micro-kernel must produce a multiple of this number of rows in each call.
  uint16_t row_tile;
  // Number of channels in a tile.
  // For best efficiency, micro-kernel must process a multiple of this number of channels in each call.
  uint16_t channel_tile;
};

struct avgpool_parameters {
  xnn_avgpool_unipass_ukernel_function unipass;
  xnn_avgpool_multipass_ukernel_function multipass;
  union {
    xnn_init_f16_scaleminmax_params_fn f16;
    xnn_init_f32_scaleminmax_params_fn f32;
    xnn_init_qu8_avgpool_minmax_params_fn qu8;
  } init;
  // Number of rows in a primary tile.
  // Unipass micro-kernel must be called with this number of rows, or fewer.
  // Multipass micro-kernel must be called with more than this number of rows.
  uint8_t primary_tile;
  // Number of rows in an incremental tile.
  // For best efficiency, multipass micro-kernel must process the number of rows in the primary tile plus a multiple
  // of this number of rows in each call. This number has no meaning for the unipass micro-kernel.
  uint8_t incremental_tile;
  // Number of channels in a tile.
  // For best efficiency, micro-kernel must process a multiple of this number of channels in each call.
  uint16_t channel_tile;
};

struct pavgpool_parameters {
  xnn_pavgpool_unipass_ukernel_function unipass;
  xnn_pavgpool_multipass_ukernel_function multipass;
  union {
    xnn_init_f16_minmax_params_fn f16;
    xnn_init_f32_minmax_params_fn f32;
  } init;
  // Number of rows in a primary tile.
  // Unipass micro-kernel must be called with this number of rows, or fewer.
  // Multipass micro-kernel must be called with more than this number of rows.
  uint8_t primary_tile;
  // Number of rows in an incremental tile.
  // For best efficiency, multipass micro-kernel must process the number of rows in the primary tile plus a multiple
  // of this number of rows in each call. This number has no meaning for the unipass micro-kernel.
  uint8_t incremental_tile;
  // Number of channels in a tile.
  // For best efficiency, micro-kernel must process a multiple of this number of channels in each call.
  uint16_t channel_tile;
};

struct argmaxpool_parameters {
  union {
    xnn_argmaxpool_unipass_ukernel_function up;
    xnn_argmaxpool_multipass_ukernel_function mp;
  };
  uint8_t mr;
  uint8_t qr;
};

struct maxpool_parameters {
  xnn_maxpool_ukernel_function ukernel;
  union {
    xnn_init_s8_minmax_params_fn s8;
    xnn_init_u8_minmax_params_fn u8;
    xnn_init_f32_minmax_params_fn f32;
    xnn_init_f16_minmax_params_fn f16;
  } init;
  uint8_t mr;
  uint8_t qr;
};

struct ibilinear_parameters {
  xnn_ibilinear_ukernel_function ukernel;
  // Number of output pixels in a tile.
  // For best efficiency, micro-kernel must produce a multiple of this number of pixels in each call.
  uint8_t pixel_tile;
  // Number of channels in a tile.
  // For best efficiency, micro-kernel must process a multiple of this number of channels in each call.
  uint8_t channel_tile;
};

struct ibilinear_chw_parameters {
  xnn_ibilinear_chw_ukernel_function ukernel;
  // Number of output pixels in a tile.
  // For best efficiency, micro-kernel must produce a multiple of this number of pixels in each call.
  uint8_t pixel_tile;
  // Number of channels in a tile.
  // For best efficiency, micro-kernel must process a multiple of this number of channels in each call.
  uint8_t channel_tile;
};

struct zip_parameters {
  xnn_zipc_ukernel_function x2;
  xnn_zipc_ukernel_function x3;
  xnn_zipc_ukernel_function x4;
  xnn_zipv_ukernel_function xm;
};

struct prelu_parameters {
  xnn_prelu_ukernel_function ukernel;
  uint16_t row_tile;
  uint16_t channel_tile;
};

struct raddstoreexpminusmax_parameters {
  xnn_raddstoreexpminusmax_ukernel_function ukernel;
  union {
    xnn_init_f16_expminus_params_fn f16;
    xnn_init_f32_expminus_params_fn f32;
  } init;
  // Number of elements in a tile.
  // For best efficiency, micro-kernel must process a multiple of this number of elements in each call.
  uint8_t element_tile;
};

struct fill_parameters {
  xnn_fill_ukernel_function ukernel;
  // Number of rows of inputs processed in one tile.
  // For best efficiency, micro-kernel must produce a multiple of this number of rows in each call.
  uint8_t row_tile;
};

struct pad_parameters {
  xnn_pad_ukernel_function ukernel;
  // Number of rows of inputs processed in one tile.
  // For best efficiency, micro-kernel must produce a multiple of this number of rows in each call.
  uint8_t row_tile;
};

struct vmulcaddc_parameters {
  xnn_vmulcaddc_ukernel_function ukernel;
  union {
    xnn_init_f16_minmax_params_fn f16;
    xnn_init_f32_minmax_params_fn f32;
  } init;
  uint8_t channel_tile;
  uint8_t row_tile;
};

#define XNN_MAX_QC8_DWCONV_UKERNELS 3
#define XNN_MAX_QS8_DWCONV_UKERNELS 2
#define XNN_MAX_QU8_DWCONV_UKERNELS 2
#define XNN_MAX_F16_DWCONV_UKERNELS 4
#define XNN_MAX_F32_DWCONV_UKERNELS 4
#define XNN_MAX_F32_ARGMAXPOOL_UKERNELS 3

// Indicates that XNNPACK as a whole has initialized.
// This does not guarantee that any particular microkernels are available.
#define XNN_INIT_FLAG_XNNPACK    0x00000001
// Indicates that F32 XNNPACK microkernels are available for use.
#define XNN_INIT_FLAG_F32        0x00000002
// Indicates that X32 XNNPACK microkernels are available for use.
#define XNN_INIT_FLAG_X32        0x00000004
// Indicates that F16 XNNPACK microkernels are available for use.
#define XNN_INIT_FLAG_F16        0x00000008
// Indicates that F16 XNNPACK microkernels are natively supported by the hardware.
#define XNN_INIT_FLAG_F16_NATIVE 0x00000010
// Indicates that X16 XNNPACK microkernels are available for use.
#define XNN_INIT_FLAG_X16        0x00000020
// Indicates that QC8 XNNPACK microkernels are available for use.
#define XNN_INIT_FLAG_QC8        0x00000040
// Indicates that QS8 XNNPACK microkernels are available for use.
#define XNN_INIT_FLAG_QS8        0x00000080
// Indicates that QU8 XNNPACK microkernels are available for use.
#define XNN_INIT_FLAG_QU8        0x00000100
// Indicates that S8 XNNPACK microkernels are available for use.
#define XNN_INIT_FLAG_S8         0x00000200
// Indicates that U8 XNNPACK microkernels are available for use.
#define XNN_INIT_FLAG_U8         0x00000400
// Indicates that X8 XNNPACK microkernels are available for use.
#define XNN_INIT_FLAG_X8         0x00000800
// Indicates that XX XNNPACK microkernels are available for use.
#define XNN_INIT_FLAG_XX         0x00001000
// Indicates that VCVT XNNPACK microkernels are available for use.
#define XNN_INIT_FLAG_VCVT       0x00002000
// Indicates that CHW XNNPACK microkernels are optimized for the host platform.
#define XNN_INIT_FLAG_CHW_OPT    0x00004000

struct xnn_parameters {
  // Bitwise combination of XNN_INIT_FLAG_* flags
  uint32_t init_flags;
  struct xnn_allocator allocator;
  struct {
    struct gemm_parameters gemm;
    struct dwconv_parameters dwconv[XNN_MAX_QC8_DWCONV_UKERNELS];
  } qc8;
  struct {
    struct gemm_parameters gemm;
    struct dwconv_parameters dwconv[XNN_MAX_QS8_DWCONV_UKERNELS];
    struct gavgpool_parameters gavgpool;
    struct vbinary_parameters vadd;
    struct vbinary_parameters vmul;
    struct vunary_parameters lrelu;
  } qs8;
  struct {
    struct gemm_parameters gemm;
    struct dwconv_parameters dwconv[XNN_MAX_QU8_DWCONV_UKERNELS];
    struct avgpool_parameters avgpool;
    struct gavgpool_parameters gavgpool;
    struct vbinary_parameters vadd;
    struct vbinary_parameters vmul;
    struct vunary_parameters lrelu;
  } qu8;
  struct {
    struct vunary_parameters clamp;
    // Bilinear interpolation (2D).
    struct ibilinear_parameters ibilinear;
    struct maxpool_parameters maxpool;
  } s8;
  struct {
    struct vunary_parameters clamp;
    // Bilinear interpolation (2D).
    struct ibilinear_parameters ibilinear;
    struct maxpool_parameters maxpool;
    xnn_u8_lut32norm_ukernel_function lut32norm;
    xnn_u8_rmax_ukernel_function rmax;
  } u8;
  struct {
    xnn_x8_lut_ukernel_function lut;
    struct zip_parameters zip;
    struct transpose_parameters transpose;
  } x8;
  struct {
    struct transpose_parameters transpose;
  } x16;
  struct {
    struct gemm_parameters gemm;
    struct gemm_parameters gemm2;
    struct dwconv_parameters dwconv[XNN_MAX_F16_DWCONV_UKERNELS];
    struct avgpool_parameters avgpool;
    struct pavgpool_parameters pavgpool;
    struct gavgpool_parameters gavgpool;
    struct maxpool_parameters maxpool;
    // Bilinear interpolation (2D).
    struct ibilinear_parameters ibilinear;
    struct vunary_parameters abs;
    struct vunary_parameters clamp;
    struct vunary_parameters elu;
    struct vunary_parameters hswish;
    struct vunary_parameters lrelu;
    struct vunary_parameters neg;
    struct vunary_parameters rndne;
    struct vunary_parameters rndz;
    struct vunary_parameters rndu;
    struct vunary_parameters rndd;
    struct vunary_parameters sigmoid;
    struct vunary_parameters sqr;
    struct vunary_parameters sqrt;
    struct prelu_parameters prelu;
    struct vbinary_parameters vadd;
    struct vbinary_parameters vdiv;
    struct vbinary_parameters vmax;
    struct vbinary_parameters vmin;
    struct vbinary_parameters vmul;
    struct vbinary_parameters vsub;
    struct vbinary_parameters vsqrdiff;
    struct vmulcaddc_parameters vmulcaddc;
    struct raddstoreexpminusmax_parameters raddstoreexpminusmax;
    xnn_rmax_ukernel_function rmax;
    // Sparse Matrix-Dense Matrix Multiplication (NR=1 block).
    struct spmm_parameters spmm;
    // Direct 3x3 stride-2 Convolution with 3 input channels and HWC->CHW layout conversion.
    struct conv_hwc2chw_parameters conv_hwc2chw_3x3c3s2;
    // Direct 3x3 stride-1 Convolution with padding 1 on left and right in CHW layout.
    struct dwconv2d_chw_parameters dwconv2d_chw_3x3;
    // Direct 3x3 stride-2 Convolution with padding 1 on left and right in CHW layout.
    struct dwconv2d_chw_parameters dwconv2d_chw_3x3s2;
    // Direct 5x5 stride-1 Convolution with padding 2 on left and right in CHW layout.
    struct dwconv2d_chw_parameters dwconv2d_chw_5x5;
    // Direct 5x5 stride-2 Convolution with padding 2 on left and right in CHW layout.
    struct dwconv2d_chw_parameters dwconv2d_chw_5x5s2;
    // Global Average Pooling in CW layout.
    struct gavgpool_cw_parameters gavgpool_cw;
    // Bilinear interpolation (2D) in CHW layout.
    struct ibilinear_chw_parameters ibilinear_chw;
  } f16;
  struct {
    struct gemm_parameters gemm;
    struct gemm_parameters gemm2;
    struct dwconv_parameters dwconv[XNN_MAX_F32_DWCONV_UKERNELS];
    struct avgpool_parameters avgpool;
    struct pavgpool_parameters pavgpool;
    struct gavgpool_parameters gavgpool;
    struct maxpool_parameters maxpool;
    struct argmaxpool_parameters argmaxpool[XNN_MAX_F32_ARGMAXPOOL_UKERNELS];
    // Bilinear interpolation (2D).
    struct ibilinear_parameters ibilinear;
    struct vunary_parameters abs;
    struct vunary_parameters clamp;
    struct vunary_parameters elu;
    struct vunary_parameters hswish;
    struct vunary_parameters lrelu;
    struct vunary_parameters neg;
    struct vunary_parameters relu;
    struct vunary_parameters rndne;
    struct vunary_parameters rndz;
    struct vunary_parameters rndu;
    struct vunary_parameters rndd;
    struct vunary_parameters sigmoid;
    struct vunary_parameters sqr;
    struct vunary_parameters sqrt;
    struct prelu_parameters prelu;
    struct vbinary_parameters vadd;
    struct vbinary_parameters vdiv;
    struct vbinary_parameters vmax;
    struct vbinary_parameters vmin;
    struct vbinary_parameters vmul;
    struct vbinary_parameters vsub;
    struct vbinary_parameters vsqrdiff;
    struct vmulcaddc_parameters vmulcaddc;
    struct raddstoreexpminusmax_parameters raddstoreexpminusmax;
    xnn_rmax_ukernel_function rmax;
    // Sparse Matrix-Dense Matrix Multiplication (NR=1 block).
    struct spmm_parameters spmm;
    // Sparse Matrix-Dense Matrix Multiplication (NR=2 block).
    struct spmm_parameters spmm2;
    // Sparse Matrix-Dense Matrix Multiplication (NR=4 block).
    struct spmm_parameters spmm4;
    // Direct 3x3 stride-2 Convolution with 3 input channels and HWC->CHW layout conversion.
    struct conv_hwc2chw_parameters conv_hwc2chw_3x3c3s2;
    // Direct 3x3 stride-1 Convolution with padding 1 on left and right in CHW layout.
    struct dwconv2d_chw_parameters dwconv2d_chw_3x3;
    // Direct 3x3 stride-2 Convolution with padding 1 on left and right in CHW layout.
    struct dwconv2d_chw_parameters dwconv2d_chw_3x3s2;
    // Direct 5x5 stride-1 Convolution with padding 2 on left and right in CHW layout.
    struct dwconv2d_chw_parameters dwconv2d_chw_5x5;
    // Direct 5x5 stride-2 Convolution with padding 2 on left and right in CHW layout.
    struct dwconv2d_chw_parameters dwconv2d_chw_5x5s2;
    // Global Average Pooling in CW layout.
    struct gavgpool_cw_parameters gavgpool_cw;
    // Bilinear interpolation (2D) in CHW layout.
    struct ibilinear_chw_parameters ibilinear_chw;
  } f32;
  struct {
    struct vunary_parameters f16_to_f32;
    struct vunary_parameters f32_to_f16;
    struct vunary_parameters f32_to_qs8;
    struct vunary_parameters f32_to_qu8;
    struct vunary_parameters qs8;
    struct vunary_parameters qs8_to_f32;
    struct vunary_parameters qu8;
    struct vunary_parameters qu8_to_f32;
  } vcvt;
  struct {
    xnn_unpool_ukernel_function unpool;
    struct zip_parameters zip;
    struct transpose_parameters transpose;
  } x32;
  struct {
    xnn_vunary_ukernel_function copy;
    struct fill_parameters fill;
    struct pad_parameters pad;
    struct transpose_parameters transpose;
  } xx;
};

#ifdef __cplusplus
extern "C" XNN_INTERNAL struct xnn_parameters xnn_params;
#else
extern XNN_INTERNAL struct xnn_parameters xnn_params;
#endif
