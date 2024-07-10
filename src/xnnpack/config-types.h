// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "xnnpack/common.h"
#include "xnnpack/microfnptr.h"
#include "xnnpack/microparams.h"

#ifdef __cplusplus
extern "C" {
#endif

struct xnn_x8_lut_config {
  xnn_x8_lut_ukernel_fn microkernel;
  // Number of elements in a tile.
  // For best efficiency, micro-kernel must process a multiple of this number of elements in each call.
  size_t tile_size;
};

struct xnn_transpose_subconfig {
  union {
    xnn_transposec_ukernel_fn const_size_ukernel;
    xnn_transposev_ukernel_fn variable_size_ukernel;
  };
  union {
    xnn_init_x8_transpose_params_fn x8;
    xnn_init_x16_transpose_params_fn x16;
    xnn_init_x24_transpose_params_fn x24;
    xnn_init_x32_transpose_params_fn x32;
    xnn_init_x64_transpose_params_fn x64;
  } init;
  // Maximum number of elements to process per ukernel call.
  size_t tile_size;
};

struct xnn_transpose_config {
  struct xnn_transpose_subconfig x8;
  struct xnn_transpose_subconfig x16;
  struct xnn_transpose_subconfig x24;
  struct xnn_transpose_subconfig x32;
  struct xnn_transpose_subconfig x64;
  struct xnn_transpose_subconfig xx;
  xnn_vunary_ukernel_fn copy;
};

struct xnn_cmul_config {
  xnn_vbinary_ukernel_fn ukernel;
  union {
    xnn_init_f32_default_params_fn f32_default;
  } init;
  // Number of elements in a tile.
  // For best efficiency, micro-kernel must process a multiple of this number of elements in each call.
  size_t element_tile;
};

struct xnn_binary_elementwise_subconfig {
  xnn_vbinary_ukernel_fn op_ukernel;
  xnn_vbinary_ukernel_fn opc_ukernel;
  xnn_vbinary_ukernel_fn ropc_ukernel;
  // Number of elements in a tile.
  // For best efficiency, micro-kernel must process a multiple of this number of elements in each call.
  size_t element_tile;
};

struct xnn_binary_elementwise_config {
  struct xnn_binary_elementwise_subconfig minmax;
  struct xnn_binary_elementwise_subconfig linear;
  union {
    xnn_init_f16_minmax_params_fn f16_minmax;
    xnn_init_f32_default_params_fn f32_default;
    xnn_init_f32_minmax_params_fn f32_minmax;
    xnn_init_qs8_add_minmax_params_fn qs8_add;
    xnn_init_qs8_mul_minmax_params_fn qs8_mul;
    xnn_init_qu8_add_minmax_params_fn qu8_add;
    xnn_init_qu8_mul_minmax_params_fn qu8_mul;
  } init;
};

struct xnn_unary_elementwise_config {
  xnn_vunary_ukernel_fn ukernel;
  union {
    xnn_init_f16_f32_cvt_params_fn f16_f32_cvt;
    xnn_init_f16_qs8_cvt_params_fn f16_qs8_cvt;
    xnn_init_f16_abs_params_fn f16_abs;
    xnn_init_f16_elu_params_fn f16_elu;
    xnn_init_f16_hswish_params_fn f16_hswish;
    xnn_init_f16_lrelu_params_fn f16_lrelu;
    xnn_init_f16_neg_params_fn f16_neg;
    xnn_init_f16_minmax_params_fn f16_minmax;
    xnn_init_f16_rsqrt_params_fn f16_rsqrt;
    xnn_init_f16_sigmoid_params_fn f16_sigmoid;
    xnn_init_f16_sqrt_params_fn f16_sqrt;
    xnn_init_f16_tanh_params_fn f16_tanh;
    xnn_init_f32_default_params_fn f32_default;
    xnn_init_f32_elu_params_fn f32_elu;
    xnn_init_f32_exp_params_fn f32_exp;
    xnn_init_f32_f16_cvt_params_fn f32_f16_cvt;
    xnn_init_f32_hswish_params_fn f32_hswish;
    xnn_init_f32_log_params_fn f32_log;
    xnn_init_f32_lrelu_params_fn f32_lrelu;
    xnn_init_f32_minmax_params_fn f32_minmax;
    xnn_init_f32_qs8_cvt_params_fn f32_qs8_cvt;
    xnn_init_f32_qu8_cvt_params_fn f32_qu8_cvt;
    xnn_init_f32_rnd_params_fn f32_rnd;
    xnn_init_f32_rsqrt_params_fn f32_rsqrt;
    xnn_init_f32_sigmoid_params_fn f32_sigmoid;
    xnn_init_f32_sqrt_params_fn f32_sqrt;
    xnn_init_f32_tanh_params_fn f32_tanh;
    xnn_init_qs8_cvt_params_fn qs8_cvt;
    xnn_init_qs8_f16_cvt_params_fn qs8_f16_cvt;
    xnn_init_qs8_f32_cvt_params_fn qs8_f32_cvt;
    xnn_init_qs8_hswish_params_fn qs8_hswish;
    xnn_init_qs8_lrelu_params_fn qs8_lrelu;
    xnn_init_qs16_qs8_cvt_params_fn qs16_qs8_cvt;
    xnn_init_qu8_cvt_params_fn qu8_cvt;
    xnn_init_qu8_f32_cvt_params_fn qu8_f32_cvt;
    xnn_init_qu8_hswish_params_fn qu8_hswish;
    xnn_init_qu8_lrelu_params_fn qu8_lrelu;
    xnn_init_s8_minmax_params_fn s8_minmax;
    xnn_init_u8_minmax_params_fn u8_minmax;
  } init;
  // Number of elements in a tile.
  // For best efficiency, micro-kernel must process a multiple of this number of
  // elements in each call.
  uint8_t element_tile;
};

struct xnn_reduce_config {
  xnn_reduce_ukernel_fn ukernel;
  xnn_rdsum_ukernel_fn rd_ukernel;
  union {
    xnn_init_f16_f32acc_scale_params_fn f16_f32acc_scale;
    xnn_init_f16_default_params_fn f16_default;
    xnn_init_f32_default_params_fn f32_default;
    xnn_init_f32_scale_params_fn f32_scale;
  } init;
  // Number of elements in a tile.
  // For best efficiency, micro-kernel must process a multiple of this number of
  // elements in each call.
  size_t element_tile;
};

struct xnn_xx_fill_config {
  xnn_fill_ukernel_fn ukernel;
  // Number of rows of inputs processed in one tile.
  // For best efficiency, micro-kernel must process a multiple of this number of rows in each call.
  uint8_t row_tile;
};

struct xnn_xx_pad_config {
  xnn_pad_ukernel_fn ukernel;
  // Number of rows of inputs processed in one tile.
  // For best efficiency, micro-kernel must process a multiple of this number of rows in each call.
  uint8_t row_tile;
};

struct xnn_avgpool_config {
  xnn_avgpool_unipass_ukernel_fn unipass;
  xnn_avgpool_multipass_ukernel_fn multipass;
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

struct xnn_pavgpool_config {
  xnn_pavgpool_unipass_ukernel_fn unipass;
  xnn_pavgpool_multipass_ukernel_fn multipass;
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

struct xnn_gavgpool_config {
  xnn_gavgpool_unipass_ukernel_fn unipass;
  xnn_gavgpool_multipass_ukernel_fn multipass;
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
  // For best efficiency, micro-kernel must process a multiple of this number of rows in each call.
  uint16_t row_tile;
  // Number of channels in a tile.
  // For best efficiency, micro-kernel must process a multiple of this number of channels in each call.
  uint16_t channel_tile;
};

struct xnn_gavgpool_cw_config {
  xnn_gavgpool_cw_ukernel_fn ukernel;
  union {
    xnn_init_f16_gavgpool_neon_params_fn f16;
    xnn_init_f32_gavgpool_params_fn f32;
  } init;
  union {
    xnn_update_f16_gavgpool_neonfp16arith_params_fn f16;
    xnn_update_f32_gavgpool_params_fn f32;
  } update;

  // Number of input pixels in a tile.
  // For best efficiency, micro-kernel must process a multiple of this number of pixels in each call.
  uint8_t pixel_tile;
  // Channel tile is always 1.
};

union xnn_dwconv_ukernel {
  xnn_dwconv_unipass_ukernel_fn unipass;
  xnn_dwconv_multipass_ukernel_fn multipass;
};

struct xnn_dwconv_config {
  union xnn_dwconv_ukernel minmax;
  union xnn_dwconv_ukernel linear;
  union {
    xnn_init_qs8_conv_minmax_params_fn qs8;
    xnn_init_qs8_qc8w_conv_minmax_params_fn qs8_qc8w;
    xnn_init_qu8_conv_minmax_params_fn qu8;
    xnn_init_f16_minmax_params_fn f16;
    xnn_init_f32_minmax_params_fn f32;
  } init;
  // Number of channels in a tile.
  uint8_t channel_tile;
  // Number of channels in a subtile. This must be less-than-equal channel_tile. After processing channel_tile, the
  // remainder is processed in tiles of channel_subtile.
  uint8_t channel_subtile;
  // How much to round channels by to get more optimal tiling.
  uint8_t channel_round;
  // Number of elements in the tile. For multipass, this is the tile size for first pass.
  uint8_t primary_tile;
  // Tile size for middle pass. Middle pass can be run multiple times. Will be zero for unipass, non-zero and not
  // greater than last_tile for multipass.
  uint8_t middle_tile;
  // Tile size for last pass. Will be zero for unipass, non-zero for multipass.
  uint8_t last_tile;
};

struct xnn_ibilinear_config {
  xnn_ibilinear_ukernel_fn ukernel;
  // Number of output pixels in a tile.
  // For best efficiency, micro-kernel must process a multiple of this number of pixels in each call.
  uint8_t pixel_tile;
  // Number of channels in a tile.
  // For best efficiency, micro-kernel must process a multiple of this number of channels in each call.
  uint8_t channel_tile;
};

// Bilinear interpolation (2D).

struct xnn_ibilinear_chw_config {
  xnn_ibilinear_chw_ukernel_fn ukernel;
  // Number of output pixels in a tile.
  // For best efficiency, micro-kernel must process a multiple of this number of pixels in each call.
  uint8_t pixel_tile;
  // Number of channels in a tile.
  // For best efficiency, micro-kernel must process a multiple of this number of channels in each call.
  uint8_t channel_tile;
};

// Bilinear interpolation (2D) in CHW layout.

struct xnn_prelu_config {
  xnn_prelu_ukernel_fn ukernel;
  // Number of rows in a tile.
  // For best efficiency, micro-kernel must process a multiple of this number of rows in each call.
  uint16_t row_tile;
  // Number of channels in a tile.
  // For best efficiency, micro-kernel must process a multiple of this number of channels in each call.
  uint16_t channel_tile;
};

struct xnn_gemm_config {
  struct gemm_fused_ukernels minmax;
  struct gemm_fused_ukernels relu;
  struct gemm_fused_ukernels linear;
#if XNN_PLATFORM_JIT
  struct gemm_codegens generator;
#endif  // XNN_PLATFORM_JIT
  union {
    xnn_init_f16_minmax_params_fn f16;
    xnn_init_f32_minmax_params_fn f32;
    xnn_init_f16_qc4w_minmax_params_fn f16_qc4w;
    xnn_init_f32_qc4w_minmax_params_fn f32_qc4w;
    xnn_init_qs8_conv_minmax_params_fn qs8;
    xnn_init_qs8_qc8w_conv_minmax_params_fn qs8_qc8w;
    xnn_init_qu8_conv_minmax_params_fn qu8;
  } init;
  // TODO(b/346765736): Replace all uses of packing functions with this.
  xnn_pack_weights_and_biases_fn pack_weights_and_biases;
  xnn_packed_stride_weights_and_biases_fn packed_stride_weights_and_biases;
  // Deprecated. Use pack_weights_and_biases instead.
  xnn_packw_gemm_gio_ukernel_fn pack_gemm_gio;
  // Deprecated. Use pack_weights_and_biases instead.
  xnn_packw_gemm_goi_ukernel_fn pack_gemm_goi;
  xnn_pack_conv_goki_w_fn pack_igemm_goki;
  xnn_pack_conv_kgo_w_fn pack_igemm_kgo;
  xnn_pack_deconv_goki_w_fn pack_deconv_goki;
  uint8_t mr;
  uint8_t nr;
  uint8_t log2_kr;
  uint8_t log2_sr;
  uint8_t planes;  // number of 4 bit planes (1 for legacy, 2 for unzip)
};

struct xnn_maxpool_config {
  xnn_maxpool_ukernel_fn ukernel;
  union {
    xnn_init_s8_minmax_params_fn s8;
    xnn_init_u8_minmax_params_fn u8;
    xnn_init_f32_minmax_params_fn f32;
    xnn_init_f16_minmax_params_fn f16;
  } init;
  // Number of elements in a tile for the first pass.
  uint8_t first_pass_tile_size;
  // Number of elements in a tile for the remainder pass. If the pooling size is less than or equals to
  // first_pass_tile_size, remainder passes are not run. We run as many remainder passes as required to cover the entire
  // pooling window.
  uint8_t remainder_pass_tile_size;
};

struct xnn_zip_config {
  xnn_zipc_ukernel_fn x2;
  xnn_zipc_ukernel_fn x3;
  xnn_zipc_ukernel_fn x4;
  xnn_zipv_ukernel_fn xm;
};

struct xnn_rmax_config {
  xnn_rmax_ukernel_fn ukernel;
  union {
    xnn_init_f32_default_params_fn f32;
    xnn_init_f16_default_params_fn f16;
  } init;
};

struct xnn_spmm_config {
  xnn_spmm_ukernel_fn ukernel;
  union {
    xnn_init_f16_minmax_params_fn f16;
    xnn_init_f32_minmax_params_fn f32;
  } init;
  // Number of M-dimension elements in a tile.
  // Corresponds to a block of pixels in 1x1 Convolution and a block of batch size in Fully Connected operator.
  uint8_t mr;
  // Number of N-dimension elements in a tile.
  // Corresponds to a block of output channels/features in 1x1 Convolution and Fully Connected operator.
  uint8_t nr;
};

struct xnn_dwconv2d_chw_parameters {
  xnn_dwconv2d_chw_ukernel_fn ukernel;
  union {
    xnn_init_f16_chw_params_fn f16;
    xnn_init_f32_chw_params_fn f32;
  } init;
  union {
    xnn_update_f16_chw_params_fn f16;
    xnn_update_f32_chw_params_fn f32;
  } update;
  // Number of output width pixels in a tile.
  uint8_t output_width_tile;
  // Number of output height pixels in a tile.
  // For best efficiency, micro-kernel must produce a multiple of this number of rows in each call.
  uint8_t output_height_tile;
};

struct xnn_dwconv2d_chw_config {
  // Direct 3x3 stride-1 Convolution with padding 1 on left and right in CHW layout.
  struct xnn_dwconv2d_chw_parameters dwconv2d_chw_3x3;
  // Direct 3x3 stride-2 Convolution with padding 1 on left and right in CHW layout.
  struct xnn_dwconv2d_chw_parameters dwconv2d_chw_3x3s2;
  // Direct 5x5 stride-1 Convolution with padding 2 on left and right in CHW layout.
  struct xnn_dwconv2d_chw_parameters dwconv2d_chw_5x5;
  // Direct 5x5 stride-2 Convolution with padding 2 on left and right in CHW layout.
  struct xnn_dwconv2d_chw_parameters dwconv2d_chw_5x5s2;
};

struct xnn_conv_hwc2chw_config {
  xnn_conv_hwc2chw_ukernel_fn ukernel_with_symm_padding;
  union {
    xnn_init_f16_minmax_params_fn f16;
    xnn_init_f32_minmax_params_fn f32;
  } init;
  // Number of output channels in a tile.
  // This parameter must be passed as is to weight packing function.
  uint8_t output_channel_tile;
  // Number of output height pixels in a tile.
  // For best efficiency, micro-kernel must produce a multiple of this number of rows in each call.
  uint8_t output_height_tile;
  // Number of output width pixels in a tile.
  uint8_t output_width_tile;
};

struct xnn_vmulcaddc_config {
  xnn_vmulcaddc_ukernel_fn ukernel;
  union {
    xnn_init_f16_minmax_params_fn f16;
    xnn_init_f32_minmax_params_fn f32;
  } init;
  // Number of channels in a tile.
  // For best efficiency, micro-kernel must process a multiple of this number of channels in each call.
  uint8_t channel_tile;
  // Number of rows of inputs processed in one tile.
  // For best efficiency, micro-kernel must process a multiple of this number of rows in each call.
  uint8_t row_tile;
};

struct xnn_raddstoreexpminusmax_config {
  xnn_raddstoreexpminusmax_ukernel_fn ukernel;
  union {
    xnn_init_f16_expminus_params_fn f16;
    xnn_init_f32_expminus_params_fn f32;
  } init;
  // Number of elements in a tile.
  // For best efficiency, micro-kernel must process a multiple of this number of elements in each call.
  uint8_t element_tile;
};

struct xnn_argmaxpool_config {
  union {
    xnn_argmaxpool_unipass_ukernel_fn up;
    xnn_argmaxpool_multipass_ukernel_fn mp;
  };
  // // Number of elements in a tile for the first pass.
  uint8_t first_pass_tile_size;
  // Number of elements in a tile for the remainder pass. If the pooling size is less than or equals to
  // first_pass_tile_size, remainder passes are not run. We run as many remainder passes as required to cover the entire
  // pooling window.
  uint8_t remainder_pass_tile_size;
};

struct xnn_lut32norm_config {
  xnn_u8_lut32norm_ukernel_fn lut32norm;
};

struct xnn_unpool_config {
  xnn_unpool_ukernel_fn unpool;
};

#ifdef __cplusplus
}  // extern "C"
#endif
