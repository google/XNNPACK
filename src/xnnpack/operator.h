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

#include <pthreadpool.h>

#include <xnnpack/params.h>
#include <xnnpack/compute.h>


enum xnn_ukernel_type {
  xnn_ukernel_type_none = 0,
  xnn_ukernel_type_add,
  xnn_ukernel_type_argmax_pooling,
  xnn_ukernel_type_average_pooling,
  xnn_ukernel_type_binary_elementwise,
  xnn_ukernel_type_channel_shuffle,
  xnn_ukernel_type_conv2d_hwc2chw,
  xnn_ukernel_type_dwconv,
  xnn_ukernel_type_gemm,
  xnn_ukernel_type_global_average_pooling,
  xnn_ukernel_type_igemm,
  xnn_ukernel_type_lut,
  xnn_ukernel_type_max_pooling,
  xnn_ukernel_type_pad,
  xnn_ukernel_type_pixelwise_average_pooling,
  xnn_ukernel_type_prelu,
  xnn_ukernel_type_softmax,
  xnn_ukernel_type_spmm,
  xnn_ukernel_type_subconv2d,
  xnn_ukernel_type_unary_elementwise,
  xnn_ukernel_type_unpooling,
  xnn_ukernel_type_vmulcaddc,
};

enum xnn_operator_type {
  xnn_operator_type_invalid = 0,
  xnn_operator_type_abs_nc_f32,
  xnn_operator_type_add_nd_f16,
  xnn_operator_type_add_nd_f32,
  xnn_operator_type_argmax_pooling_nhwc_f32,
  xnn_operator_type_average_pooling_nhwc_f32,
  xnn_operator_type_average_pooling_nhwc_qu8,
  xnn_operator_type_bankers_rounding_nc_f32,
  xnn_operator_type_channel_shuffle_nc_x32,
  xnn_operator_type_channel_shuffle_nc_x8,
  xnn_operator_type_clamp_nc_f32,
  xnn_operator_type_clamp_nc_u8,
  xnn_operator_type_ceiling_nc_f32,
  xnn_operator_type_constant_pad_nd_x32,
  xnn_operator_type_convolution_nchw_f32,
  xnn_operator_type_convolution_nhwc_f16,
  xnn_operator_type_convolution_nhwc_f32,
  xnn_operator_type_convolution_nhwc_qu8,
  xnn_operator_type_copy_nc_x32,
  xnn_operator_type_deconvolution_nhwc_f32,
  xnn_operator_type_deconvolution_nhwc_qu8,
  xnn_operator_type_divide_nd_f32,
  xnn_operator_type_fully_connected_nc_f32,
  xnn_operator_type_fully_connected_nc_qu8,
  xnn_operator_type_floor_nc_f32,
  xnn_operator_type_global_average_pooling_nwc_f16,
  xnn_operator_type_global_average_pooling_nwc_f32,
  xnn_operator_type_global_average_pooling_nwc_qu8,
  xnn_operator_type_global_average_pooling_ncw_f32,
  xnn_operator_type_hardswish_nc_f32,
  xnn_operator_type_leaky_relu_nc_f32,
  xnn_operator_type_leaky_relu_nc_qu8,
  xnn_operator_type_max_pooling_nhwc_f32,
  xnn_operator_type_max_pooling_nhwc_u8,
  xnn_operator_type_maximum_nd_f32,
  xnn_operator_type_minimum_nd_f32,
  xnn_operator_type_multiply_nd_f32,
  xnn_operator_type_negate_nc_f32,
  xnn_operator_type_prelu_nc_f32,
  xnn_operator_type_resize_bilinear_nhwc_f32,
  xnn_operator_type_sigmoid_nc_f32,
  xnn_operator_type_sigmoid_nc_qu8,
  xnn_operator_type_softmax_nc_f32,
  xnn_operator_type_softmax_nc_qu8,
  xnn_operator_type_square_nc_f32,
  xnn_operator_type_square_root_nc_f32,
  xnn_operator_type_squared_difference_nd_f32,
  xnn_operator_type_subtract_nd_f32,
  xnn_operator_type_truncation_nc_f32,
  xnn_operator_type_unpooling_nhwc_x32,
};

struct xnn_ukernel_conv2d {
  union {
    xnn_conv_hwc2chw_ukernel_function hwc2chw_function;
    xnn_conv_hwc_ukernel_function hwc_function;
  };
  uint8_t output_height_tile;
  uint8_t output_channel_tile;
};

struct xnn_ukernel_dwconv {
  union {
    xnn_dwconv_unipass_ukernel_function unipass_function;
    xnn_dwconv_multipass_ukernel_function multipass_function;
  };
  uint8_t primary_tile;
  uint8_t incremental_tile;
};

// Direct 2D Depthwise Convolution
struct xnn_ukernel_dwconv2d {
  union {
    xnn_dwconv_chw_ukernel_function chw_function;
  };
  uint8_t input_width_tile;
  uint8_t output_width_tile;
};

struct xnn_ukernel_gemm {
  struct xnn_hmp_gemm_ukernel general_case;
  struct xnn_hmp_gemm_ukernel mr1_case;
  uint8_t mr;
  uint8_t nr;
  uint8_t kr;
};

struct xnn_ukernel_igemm {
  struct xnn_hmp_igemm_ukernel general_case;
  struct xnn_hmp_igemm_ukernel mr1_case;
  struct xnn_hmp_gemm_ukernel gemm_case;
  uint8_t mr;
  uint8_t nr;
  uint8_t kr;
};

struct xnn_ukernel_spmm {
  xnn_spmm_ukernel_function function;
  uint8_t mr;
};

struct xnn_ukernel_vmulcaddc {
  xnn_vmulcaddc_ukernel_function function;
  uint8_t mr;
};

struct xnn_ukernel_vbinary {
  xnn_vbinary_ukernel_function function;
};

struct xnn_ukernel {
  enum xnn_ukernel_type type;
  union {
    struct xnn_ukernel_conv2d conv2d;
    struct xnn_ukernel_dwconv dwconv;
    struct xnn_ukernel_dwconv2d dwconv2d;
    struct xnn_ukernel_gemm gemm;
    struct xnn_ukernel_igemm igemm;
    struct xnn_ukernel_spmm spmm;
    struct xnn_ukernel_vmulcaddc vmulcaddc;
    struct xnn_ukernel_vbinary vbinary;
  };
};

enum xnn_run_state {
  xnn_run_state_invalid = 0,
  xnn_run_state_ready,
  xnn_run_state_skip,
};

struct subconvolution_params {
  void* weights;
  size_t w_stride;
  const void** indirection_buffer;
  void* output;
  size_t slice_width;
  size_t slice_height;
  size_t indirection_y_stride;
  size_t indirection_x_stride;
  // scaled_kernel_size := kernel_size * mr * sizeof(void*).
  size_t scaled_kernel_size;
};

struct xnn_operator {
  size_t batch_size;
  uint32_t padding_top;
  uint32_t padding_right;
  uint32_t padding_bottom;
  uint32_t padding_left;
  uint32_t kernel_height;
  uint32_t kernel_width;
  uint32_t stride_height;
  uint32_t stride_width;
  uint32_t dilation_height;
  uint32_t dilation_width;
  uint32_t groups;
  size_t group_channels;
  size_t group_input_channels;
  size_t group_output_channels;
  size_t channels;

  size_t pad_before_channels;
  size_t pad_after_channels;
  uint32_t pad_value;

  size_t input_height;
  size_t input_width;
  size_t input_pixel_stride;
  const void* input;
  const void** indirection_buffer;

  size_t input2_pixel_stride;
  const void* input2;

  size_t output_height;
  size_t output_width;
  size_t output_pixel_stride;
  void* output;

  void* packed_weights;
  // Total number of non-zero kernel elements when weights use sparse representation.
  size_t num_nonzero_values;
  // Total number of non-zero kernel blocks when weights use sparse representation.
  size_t num_nonzero_blocks;
  // Total number of output channel blocks when weights use sparse representation.
  size_t num_output_channel_blocks;
  // Input channel corresponding to the first non-zero kernel element.
  size_t first_input_channel;

  float input_scale;
  float output_scale;
  uint8_t input_zero_point;
  uint8_t kernel_zero_point;
  uint8_t output_zero_point;
  uint8_t output_min;
  uint8_t output_max;

  size_t valid_batch_size;
  size_t last_input_height;
  size_t last_input_width;
  const void* last_input;
  size_t last_output_height;
  size_t last_output_width;
  void* last_output;

  void* zero_buffer;
  void* lookup_table;
  void* pixelwise_buffer;
  struct subconvolution_params* subconvolution_buffer;
  uint32_t flags;

  union {
    union xnn_f32_abs_params f32_abs;
    union xnn_f32_lrelu_params f32_lrelu;
    union xnn_f32_neg_params f32_neg;
    union xnn_f32_rnd_params f32_rnd;
    // Parameters for Global Average Pooling in CHW layout
    union xnn_f32_gavgpool_params f32_gavgpool;
    union xnn_f32_hswish_params f32_hswish;
    struct {
      struct xnn_f16_minmax_params f16_minmax;
      struct xnn_f16_scaleminmax_params f16_scaleminmax;
    };
    // Pixelwise Average Pooling normally use f32_minmax_params, but also initialize
    // f32_scaleminmax_params in case it needs to switch to Global Average Pooling operation.
    struct {
      union xnn_f32_minmax_params f32_minmax;
      union xnn_f32_scaleminmax_params f32_scaleminmax;
    };
    union xnn_f32_chw_params f32_chw;
    union xnn_qu8_add_params qu8_add;
    union xnn_qu8_gemm_params qu8_gemm;
    // Average Pooling normally use qu8_avgpool_params, but also initialize qu8_gavgpool_params in case it needs to switch
    // to Global Average Pooling operation.
    struct {
      union xnn_qu8_avgpool_params qu8_avgpool;
      union xnn_qu8_avgpool_params qu8_gavgpool;
    };
    union xnn_u8_minmax_params u8_minmax;
  } params;
  enum xnn_operator_type type;
  struct xnn_ukernel ukernel;

  struct compute_parameters compute;
  struct compute_parameters compute2;
  union {
    struct argmax_pooling_context argmax_pooling;
    struct average_pooling_context average_pooling;
    struct channel_shuffle_context channel_shuffle;
    struct conv2d_context conv2d;
    struct dwconv2d_context dwconv2d;
    struct dwconv_context dwconv;
    struct elementwise_binary_context elementwise_binary;
    struct gemm_context gemm;
    struct global_average_pooling_nwc_context global_average_pooling_nwc;
    struct global_average_pooling_ncw_context global_average_pooling_ncw;
    struct igemm_context igemm;
    struct lut_contiguous_context lut_contiguous;
    struct lut_strided_context lut_strided;
    struct max_pooling_context max_pooling;
    struct pad_context pad;
    struct pixelwise_average_pooling_context pixelwise_average_pooling;
    struct prelu_context prelu;
    struct resize_bilinear_context resize_bilinear;
    struct spmm_context spmm;
    struct subconv_context subconv;
    struct subgemm_context subgemm;
    struct f32_three_pass_softmax_context f32_three_pass_softmax;
    struct u8_softmax_context u8_softmax;
    struct univector_contiguous_context univector_contiguous;
    struct univector_strided_context univector_strided;
    struct unpooling_context unpooling;
    struct vmulcaddc_context vmulcaddc;
  } context;

  enum xnn_run_state state;
};
