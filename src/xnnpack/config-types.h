// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>

#include "src/xnnpack/hardware-config.h"
#include "src/xnnpack/microfnptr.h"

#ifdef __cplusplus
extern "C" {
#endif

struct xnn_x8_lut_config {
  xnn_x8_lut_ukernel_fn microkernel;
};

struct xnn_transpose_subconfig {
  union {
    xnn_transposec_ukernel_fn const_size_ukernel;
    xnn_transposev_ukernel_fn variable_size_ukernel;
  };
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
};

struct xnn_binary_elementwise_config {
  xnn_vbinary_ukernel_fn op_ukernel;
  xnn_vbinary_ukernel_fn opc_ukernel;
  xnn_vbinary_ukernel_fn ropc_ukernel;
  xnn_init_binary_params_fn init;
  // Number of elements in a tile.
  // For best efficiency, micro-kernel must process a multiple of this number of
  // elements in each call.
  size_t element_tile;
};

struct xnn_unary_elementwise_config {
  xnn_vunary_ukernel_fn ukernel;
  xnn_init_unary_uparams_fn init;
};

struct xnn_reduce_config {
  xnn_reduce_ukernel_fn ukernel;
  xnn_reduce_discontiguous_ukernel_fn rd_ukernel;
  uint32_t identity_value;
  union {
    xnn_init_reduce_params_fn reduce;
    xnn_init_f32_default_params_fn f32;
    xnn_init_f16_default_params_fn f16;
  } init;
  xnn_update_reduce_params_fn update;
};

struct xnn_xx_fill_config {
  xnn_fill_ukernel_fn ukernel;
};

struct xnn_xx_pad_config {
  xnn_pad_ukernel_fn ukernel;
};

struct xnn_avgpool_config {
  xnn_avgpool_ukernel_fn ukernel;
  union {
    xnn_init_f16_scaleminmax_params_fn f16;
    xnn_init_f32_scaleminmax_params_fn f32;
  } init;
  // Number of rows in a primary tile.
  // TODO: Only used by tests, it should be removed.
  uint8_t primary_tile;
  // Number of channels in a tile.
  // For best efficiency, micro-kernel must process a multiple of this number of
  // channels in each call.
  uint16_t channel_tile;
};

struct xnn_pack_lh_config {
  xnn_pack_lh_ukernel_fn ukernel;
  xnn_pack_lh_size_fn size_fn;
  xnn_pack_lh_offset_fn offset_fn;
};

struct xnn_dwconv_config {
  xnn_dwconv_ukernel_fn minmax;
  xnn_dwconv_ukernel_fn linear;
  union {
    xnn_init_qs8_conv_minmax_params_fn qs8;
    xnn_init_qs8_qc8w_conv_minmax_params_fn qs8_qc8w;
    xnn_init_qu8_conv_minmax_params_fn qu8;
    xnn_init_f16_minmax_params_fn f16;
    xnn_init_f32_minmax_params_fn f32;
  } init;
  // Number of channels in a tile.
  uint32_t channel_tile;
  // Number of elements in the tile.
  uint8_t primary_tile;
};

// Bilinear interpolation (2D).

struct xnn_ibilinear_config {
  xnn_ibilinear_ukernel_fn ukernel;
  // Number of output pixels in a tile.
  // For best efficiency, micro-kernel must process a multiple of this number of
  // pixels in each call.
  uint8_t pixel_tile;
  size_t log2_data_element_size;
  size_t log2_weight_element_size;
  xnn_indirection_init_resize_bilinear2d_hwc_fn indirection_init;
};

// Bilinear interpolation (2D) in CHW layout.

struct xnn_ibilinear_chw_config {
  xnn_ibilinear_chw_ukernel_fn ukernel;
  // Number of channels in a tile.
  // For best efficiency, micro-kernel must process a multiple of this number of
  // channels in each call.
  uint8_t channel_tile;
  size_t log2_data_element_size;
  size_t log2_weight_element_size;
  xnn_indirection_init_resize_bilinear2d_chw_fn indirection_init;
};

struct xnn_gemm_config {
  struct gemm_fused_ukernels minmax;
  struct gemm_fused_ukernels relu;
  struct gemm_fused_ukernels linear;
  union {
    xnn_init_f16_minmax_params_fn f16;
    xnn_init_f32_minmax_params_fn f32;
    xnn_init_f16_qc4w_minmax_params_fn f16_qc4w;
    xnn_init_f16_qb4w_minmax_params_fn f16_qb4w;
    xnn_init_f32_qc4w_minmax_params_fn f32_qc4w;
    xnn_init_f32_qb4w_minmax_params_fn f32_qb4w;
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
  // TODO(b/346765736): Use pack_weights_and_biases instead.
  xnn_packw_gemm_goi_bl_ukernel_fn pack_gemm_goi_bl;
  xnn_pack_conv_goki_w_fn pack_igemm_goki;
  xnn_pack_conv_kgo_w_fn pack_igemm_kgo;
  xnn_pack_deconv_goki_w_fn pack_deconv_goki;
  uint8_t mr;
  uint8_t nr;
  uint8_t log2_kr;
  uint8_t log2_sr;
  uint8_t planes;     // number of 4 bit planes (1 for legacy, 2 for unzip)
  uint8_t mr_packed;  // `mr` value used for packed left-hand operands.
  enum xnn_arch_flags arch;
};

struct xnn_maxpool_config {
  xnn_maxpool_ukernel_fn ukernel;
  union {
    xnn_init_s8_minmax_params_fn s8;
    xnn_init_u8_minmax_params_fn u8;
    xnn_init_f32_minmax_params_fn f32;
    xnn_init_f16_minmax_params_fn f16;
  } init;
};

struct xnn_zip_config {
  xnn_zipc_ukernel_fn x2;
  xnn_zipc_ukernel_fn x3;
  xnn_zipc_ukernel_fn x4;
  xnn_zipv_ukernel_fn xm;
};

struct xnn_spmm_config {
  xnn_spmm_ukernel_fn ukernel;
  union {
    xnn_init_f16_minmax_params_fn f16;
    xnn_init_f32_minmax_params_fn f32;
  } init;
  // Number of M-dimension elements in a tile.
  // Corresponds to a block of pixels in 1x1 Convolution and a block of batch
  // size in Fully Connected operator.
  uint8_t mr;
  // Number of N-dimension elements in a tile.
  // Corresponds to a block of output channels/features in 1x1 Convolution and
  // Fully Connected operator.
  uint8_t nr;
};

struct xnn_dwconv2d_chw_parameters {
  xnn_dwconv2d_chw_ukernel_fn ukernel;
  union {
    xnn_init_f16_minmax_params_fn f16;
    xnn_init_f32_minmax_params_fn f32;
  } init;
  // Number of output width pixels in a tile.
  uint8_t output_width_tile;
};

struct xnn_dwconv2d_chw_config {
  // Direct 3x3 stride-1 Convolution with padding 1 on left and right in CHW
  // layout.
  struct xnn_dwconv2d_chw_parameters dwconv2d_chw_3x3;
  // Direct 3x3 stride-2 Convolution with padding 1 on left and right in CHW
  // layout.
  struct xnn_dwconv2d_chw_parameters dwconv2d_chw_3x3s2;
  // Direct 5x5 stride-1 Convolution with padding 2 on left and right in CHW
  // layout.
  struct xnn_dwconv2d_chw_parameters dwconv2d_chw_5x5;
  // Direct 5x5 stride-2 Convolution with padding 2 on left and right in CHW
  // layout.
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
  // For best efficiency, micro-kernel must produce a multiple of this number of
  // rows in each call.
  uint8_t output_height_tile;
};

struct xnn_vmulcaddc_config {
  xnn_vmulcaddc_ukernel_fn ukernel;
  union {
    xnn_init_f16_minmax_params_fn f16;
    xnn_init_f32_minmax_params_fn f32;
  } init;
  // Number of channels in a tile.
  // For best efficiency, micro-kernel must process a multiple of this number of
  // channels in each call.
  uint8_t channel_tile;
  // Number of rows of inputs processed in one tile.
  // For best efficiency, micro-kernel must process a multiple of this number of
  // rows in each call.
  uint8_t row_tile;
};

struct xnn_raddstoreexpminusmax_config {
  xnn_raddstoreexpminusmax_ukernel_fn ukernel;
  union {
    xnn_init_f16_expminus_params_fn f16;
    xnn_init_f32_expminus_params_fn f32;
  } init;
};

struct xnn_argmaxpool_config {
  xnn_argmaxpool_unipass_ukernel_fn ukernel;
  // Number of elements in a tile for the first pass.
  uint8_t primary_tile;
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
