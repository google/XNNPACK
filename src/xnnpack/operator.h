// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#ifndef THIRD_PARTY_XNNPACK_SRC_XNNPACK_OPERATOR_H_
#define THIRD_PARTY_XNNPACK_SRC_XNNPACK_OPERATOR_H_

#include <stddef.h>
#include <stdint.h>

#include "include/xnnpack.h"
#include "src/xnnpack/common.h"
#include "src/xnnpack/compute.h"
#include "src/xnnpack/config-types.h"
#include "src/xnnpack/microfnptr.h"
#include "src/xnnpack/microkernel-type.h"
#include "src/xnnpack/microparams.h"
#include "src/xnnpack/node-type.h"
#include "src/xnnpack/operator-type.h"
#include <pthreadpool.h>

// Maximum number of pthreadpool parallelization invocations per operator.
#define XNN_MAX_COMPUTE_INVOCATIONS 3

#ifdef __cplusplus
extern "C" {
#endif

typedef struct xnn_operator* xnn_operator_t;

struct xnn_ukernel_conv2d {
  union {
    xnn_conv_hwc2chw_ukernel_fn hwc2chw_fn;
    xnn_conv_hwc_ukernel_fn hwc_fn;
  };
  uint8_t output_height_tile;
  uint8_t output_channel_tile;
};

struct xnn_ukernel_dwconv {
  xnn_dwconv_ukernel_fn ukernel;
  uint32_t channel_tile;
  uint8_t primary_tile;
};

// Direct 2D Depthwise Convolution
struct xnn_ukernel_dwconv2d {
  union {
    xnn_dwconv2d_chw_ukernel_fn chw_fn;
  };
  uint8_t output_width_tile;
};

struct xnn_ukernel_gemm {
  struct xnn_hmp_gemm_ukernel gemm_cases[XNN_MAX_MR];
  // Attention operator uses both types of packing.
  xnn_packw_gemm_goi_ukernel_fn packw_gemm_goi;
  xnn_packw_gemm_gio_ukernel_fn packw_gemm_gio;
  uint8_t mr;
  uint8_t mr_packed;
  uint8_t nr;
  uint8_t kr;
  uint8_t sr;
  uint8_t kp;
};

struct xnn_ukernel_igemm {
  struct xnn_hmp_igemm_ukernel igemm_cases[XNN_MAX_MR];
  struct xnn_hmp_gemm_ukernel gemm_cases[XNN_MAX_MR];
  uint8_t mr;
  uint8_t nr;
  uint8_t kr;
  uint8_t sr;
};

struct xnn_ukernel_spmm {
  xnn_spmm_ukernel_fn function;
  uint8_t mr;
};

struct xnn_ukernel_vmulcaddc {
  xnn_vmulcaddc_ukernel_fn function;
  uint8_t mr;
  uint8_t channel_tile;
};

struct xnn_ukernel_vbinary {
  xnn_vbinary_ukernel_fn op_fn;
  xnn_vbinary_ukernel_fn opc_fn;
  xnn_vbinary_ukernel_fn ropc_fn;
};

struct xnn_ukernel_vunary {
  xnn_vunary_ukernel_fn function;
};

struct xnn_ukernel {
  enum xnn_microkernel_type type;
  // Used by subconv2d whether it is a GEMM or IGEMM.
  enum xnn_microkernel_type subtype;
  union {
    struct xnn_ukernel_conv2d conv2d;
    struct xnn_ukernel_dwconv dwconv;
    struct xnn_ukernel_dwconv2d dwconv2d;
    struct {
      struct xnn_ukernel_gemm gemm;
      struct xnn_ukernel_gemm gemm_nr2;
    };
    struct xnn_ukernel_igemm igemm;
    struct xnn_ukernel_spmm spmm;
    struct xnn_ukernel_vmulcaddc vmulcaddc;
    struct xnn_ukernel_vbinary vbinary;
    struct xnn_ukernel_vunary vunary;
  };
};

// Valid state transitions:
// - xnn_run_state_invalid -> xnn_run_state_skip
// - xnn_run_state_invalid -> xnn_run_state_ready
// - xnn_run_state_invalid -> xnn_run_state_needs_setup -> xnn_run_state_ready
enum xnn_run_state {
  // When an operator is first created, it starts off in invalid state, it needs
  // to be setup, or reshape + setup.
  xnn_run_state_invalid = 0,
  // Operator is ready to be run.
  xnn_run_state_ready,
  // Operator doesn't need to be run.
  xnn_run_state_skip,
  // Operator has been reshaped, but not setup yet, pointers are not set.
  xnn_run_state_needs_setup,
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

  uint32_t pad_value;

  size_t input_height;
  size_t input_width;
  size_t input_pixel_stride;
  const void* input;
  const void** indirection_buffer;

  size_t output_height;
  size_t output_width;
  size_t output_pixel_stride;
  void* output;
  const void* quantization_params;

  size_t k_block_size;

  union {
    // Pointer to allocated packed weights. Use this if weights_cache is NULL.
    void* pointer;
    // Offset into the weights cache where the packed weights are. Only valid if
    // weights_cache is not NULL.
    size_t offset;
  } packed_weights;
  // Stride between each set of packed weights.
  size_t weights_stride;
  // Total number of non-zero kernel elements when weights use sparse
  // representation.
  size_t num_nonzero_values;
  // Total number of non-zero kernel blocks when weights use sparse
  // representation.
  size_t num_nonzero_blocks;
  // Total number of output channel blocks when weights use sparse
  // representation.
  size_t num_output_channel_blocks;
  // Input channel corresponding to the first non-zero kernel element.
  size_t first_input_channel;

  float input_scale;
  float output_scale;

  size_t valid_batch_size;
  size_t last_input_height;
  size_t last_input_width;
  size_t last_input_channels;
  const void* last_input;
  size_t last_output_height;
  size_t last_output_width;
  void* last_output;
  uint32_t last_mr;

  uint32_t block_size;

  void* zero_buffer;
  void** zero_buffers;
  size_t zero_size;
  void* lookup_table;
  void* pixelwise_buffer;
  struct subconvolution_params* subconvolution_buffer;
  uint32_t flags;

  union {
    struct {
      uint32_t log2_element_size;
      enum xnn_binary_operator op_type;
    } binary_elementwise;
    struct {
      uint8_t num_nonbatch_dims;
      uint8_t log2_input_size;
      uint8_t log2_output_size;
      enum xnn_unary_operator op_type;
    } unary_elementwise;
    struct {
      uint32_t log2_data_element_size;
      uint32_t log2_accumulator_element_size;
      uint32_t identity_value;
    } reduce;
    struct {
      enum xnn_node_type subtype;
    } copy;
  };

  union {
    union xnn_binary_uparams binary;
    union xnn_unary_uparams unary;
    struct xnn_f16_default_params f16_default;
    struct xnn_f32_default_params f32_default;
    struct xnn_f16_minmax_params f16_minmax;
    struct xnn_f16_scaleminmax_params f16_scaleminmax;
    struct xnn_reduce_params reduce;
    struct xnn_f32_minmax_params f32_minmax;
    struct xnn_f32_scaleminmax_params f32_scaleminmax;
    struct xnn_f32_scale_params f32_scale;
    struct xnn_f16_minmax_params f16_chw;
    struct xnn_f32_minmax_params f32_chw;
    struct xnn_f32_qb4w_minmax_params f32_qb4w_minmax;
    struct xnn_f32_qc4w_minmax_params f32_qc4w_minmax;
    union xnn_qs8_conv_minmax_params qs8_conv_minmax;
    union xnn_qs8_qc8w_conv_minmax_params qs8_qc8w_conv_minmax;
    union xnn_qu8_conv_minmax_params qu8_conv_minmax;
    struct xnn_s8_minmax_params s8_minmax;
    struct xnn_s32_default_params s32_default;
    struct xnn_u8_minmax_params u8_minmax;
  } params;
  // Second set of params. Operators like Dynamic Fully Connected only decides
  // on the specific config to use during reshape, so it needs to keep two sets
  // of params around. Configs can have different initialization functions. We
  // also use this to store parameters to binary operators. For most such
  // operators, this is a copy of params, but params need to be swapped for
  // commutative ops with per-operand params.
  union {
    union xnn_binary_uparams binary;
    union xnn_unary_uparams unary;
    struct xnn_f16_default_params f16_default;
    struct xnn_f32_minmax_params f32_minmax;
    struct xnn_f32_default_params f32_default;
    struct xnn_s8_minmax_params s8_minmax;
    struct xnn_u8_minmax_params u8_minmax;
  } params2;
  enum xnn_operator_type type;
  struct xnn_ukernel ukernel;

  union {
    const struct xnn_argmaxpool_config* argmaxpool_config;
    struct {
      const struct xnn_avgpool_config* avgpool_config;
      const struct xnn_reduce_config* discontiguous_reduce_config;
      const struct xnn_reduce_config* contiguous_reduce_config;
      const struct xnn_unary_elementwise_config* cvt_config;
    };
    const struct xnn_ibilinear_chw_config* ibilinear_chw_config;
    const struct xnn_ibilinear_config* ibilinear_config;
    struct {
      union {
        // For QU8.
        const struct xnn_lut32norm_config* lut32norm_config;
        // For F16 and F32.
        struct {
          const struct xnn_raddstoreexpminusmax_config*
              raddstoreexpminusmax_config;
          const struct xnn_binary_elementwise_config* vmul_config;
        };
      };
    };  // For softmax operator.
    const struct xnn_maxpool_config* maxpool_config;
    const struct xnn_unpool_config* unpool_config;
    const struct xnn_zip_config* zip_config;
    struct {
      const struct xnn_xx_fill_config* fill_config;
      const struct xnn_xx_pad_config* pad_config;
    };  // For constant pad operator.
    const struct xnn_x8_lut_config* lut_config;
    const struct xnn_cmul_config* cmul_config;
    const struct xnn_transpose_config* transpose_config;
    const struct xnn_binary_elementwise_config* binary_elementwise_config;
    struct {
      const struct xnn_unary_elementwise_config* unary_elementwise_config;
      const struct xnn_gemm_config*
          gemm_config;  // For dynamic quantization convert operator.
    };  // For unary elementwise operators.
    struct {
      const struct xnn_reduce_config* rmax_config;
      const struct xnn_raddstoreexpminusmax_config* raddstoreexpminusmax_config;
      const struct xnn_binary_elementwise_config* vadd_config;
      const struct xnn_binary_elementwise_config* vmul_config;
      const struct xnn_unary_elementwise_config* vtanh_config;
      const struct xnn_binary_elementwise_config* vprelu_config;
      enum xnn_attention_logits_cap_type cap_type;
      struct xnn_attention_logits_cap_tanh_params cap_params;
    } attention;  // For attention operator.
    const struct xnn_pack_lh_config* pack_lh_config;
  };

  struct compute_parameters compute[XNN_MAX_COMPUTE_INVOCATIONS];
  union {
    struct argmax_pooling_context argmax_pooling;
    struct average_pooling_context average_pooling;
    struct conv2d_context conv2d;
    struct dwconv2d_context dwconv2d;
    struct {
      struct dwconv_context dwconv;
      struct dwconv_indirection_init_context dwconv_indirection_init;
    } dwconv;
    struct elementwise_binary_context elementwise_binary;
    // PACKW GEMM GOI + GEMM are used together in Dynamic Fully Connected.
    struct {
      union {
        struct gemm_context gemm;
      } gemm;
      struct packw_gemm_goi_context packw_gemm_goi;
      struct packw_gemm_gio_context packw_gemm_gio;
      bool const_weights;
    } gemm;
    struct {
      struct igemm_context igemm;
      struct conv2d_igemm_indirection_init_context
          conv2d_igemm_indirection_init;
    } igemm;
    struct lut_contiguous_context lut_contiguous;
    struct lut_strided_context lut_strided;
    struct max_pooling_context max_pooling;
    struct pad_context pad;
    struct reduce_context reduce;
    struct {
      struct resize_bilinear_context resize_bilinear;
      struct resize_bilinear_nhwc_indirection_init_context
          resize_nhwc_indirection_init;
    };
    struct resize_bilinear_chw_context resize_bilinear_chw;
    struct slice_context slice;
    struct spmm_context spmm;
    struct subconv_context subconv;
    struct subgemm_context subgemm;
    struct transpose_context transpose;
    struct floating_point_softmax_context floating_point_softmax;
    struct u8_softmax_context u8_softmax;
    struct f16_qd8_convert_context f16_qd8_convert;
    struct f32_qd8_convert_context f32_qd8_convert;
    struct f32_qp8_convert_context f32_qp8_convert;
    struct univector_contiguous_context univector_contiguous;
    struct univector_strided_context univector_strided;
    struct unpooling_context unpooling;
    struct vmulcaddc_context vmulcaddc;
    struct rope_context rope;
    struct pack_lh_context pack_lh;
  } context;

  struct xnn_code_cache* code_cache;
  xnn_weights_cache_t weights_cache;
  enum xnn_run_state state;
};

XNN_INTERNAL enum xnn_status xnn_run_operator_with_index(
    xnn_operator_t op, size_t opdata_index, size_t operator_object_index,
    pthreadpool_t threadpool);

XNN_INTERNAL enum xnn_operator_type xnn_reduce_operator_to_operator_type(
    enum xnn_reduce_operator type);

enum xnn_status xnn_run_operator(
  xnn_operator_t op,
  pthreadpool_t threadpool);

enum xnn_status xnn_delete_operator(
  xnn_operator_t op);

/// Operator API:
/// - create operator will create and populate a xnn_operator_t
/// - reshape operator will update fields in xnn_operator_t with shape/dimensions and parallelization information
/// - setup operator will update pointers to input and outputs
/// Each supported operator must have a create, reshape, and setup function. (Optionally a run function.)
/// Operators listed below are in alphabetical order by operator name; within each operator, we sort alphabetically by
/// data layout and type. We also group create, reshape, setup (and optionally run) functions of each operator together.

enum xnn_status xnn_create_binary_elementwise_nd(
  enum xnn_binary_operator type,
  enum xnn_datatype datatype,
  const struct xnn_quantization_params* input1_quantization,
  const struct xnn_quantization_params* input2_quantization,
  const struct xnn_quantization_params* output_quantization,
  uint32_t flags,
  xnn_operator_t* binary_op_out);

enum xnn_status xnn_reshape_binary_elementwise_nd(
  xnn_operator_t binary_op,
  size_t num_input1_dims,
  const size_t* input1_shape,
  size_t num_input2_dims,
  const size_t* input2_shape,
  pthreadpool_t threadpool);

enum xnn_status xnn_setup_binary_elementwise_nd(
  xnn_operator_t binary_op,
  const void* input1,
  const void* input2,
  void* output);

enum xnn_status xnn_run_binary_elementwise_nd(
  enum xnn_binary_operator type,
  enum xnn_datatype datatype,
  const struct xnn_quantization_params* input1_quantization,
  const struct xnn_quantization_params* input2_quantization,
  const struct xnn_quantization_params* output_quantization,
  uint32_t flags,
  size_t num_input1_dims,
  const size_t* input1_shape,
  size_t num_input2_dims,
  const size_t* input2_shape,
  const void* input1,
  const void* input2,
  void* output,
  pthreadpool_t threadpool);

enum xnn_status xnn_create_unary_elementwise_nc(
  enum xnn_unary_operator op_type,
  enum xnn_datatype input_datatype,
  enum xnn_datatype output_datatype,
  const union xnn_unary_params* params,
  const struct xnn_quantization_params* input_quantization,
  const struct xnn_quantization_params* output_quantization,
  uint32_t flags,
  xnn_operator_t* op_out);

enum xnn_status xnn_reshape_unary_elementwise_nc(
  xnn_operator_t op,
  size_t batch_size,
  size_t channels,
  size_t input_stride,
  size_t output_stride,
  pthreadpool_t threadpool);

enum xnn_status xnn_setup_unary_elementwise_nc(
  xnn_operator_t op,
  const void* input,
  void* output);

enum xnn_status xnn_run_unary_elementwise_nc(
  // create parameters
  enum xnn_unary_operator op_type,
  enum xnn_datatype input_datatype,
  enum xnn_datatype output_datatype,
  const union xnn_unary_params* params,
  const struct xnn_quantization_params* input_quantization,
  const struct xnn_quantization_params* output_quantization,
  uint32_t flags,
  // reshape parameters
  size_t batch_size,
  size_t channels,
  size_t input_stride,
  size_t output_stride,
  pthreadpool_t threadpool,
  // setup parameters
  const void* input,
  void* output);

enum xnn_status xnn_create_argmax_pooling2d_nhwc_f32(
  uint32_t input_padding_top,
  uint32_t input_padding_right,
  uint32_t input_padding_bottom,
  uint32_t input_padding_left,
  uint32_t pooling_height,
  uint32_t pooling_width,
  uint32_t flags,
  xnn_operator_t* argmax_pooling_op_out);

enum xnn_status xnn_reshape_argmax_pooling2d_nhwc_f32(
  xnn_operator_t argmax_pooling_op,
  size_t batch_size,
  size_t input_height,
  size_t input_width,
  size_t channels,
  size_t input_pixel_stride,
  size_t output_pixel_stride,
  size_t* output_height_out,
  size_t* output_width_out,
  pthreadpool_t threadpool);

enum xnn_status xnn_setup_argmax_pooling2d_nhwc_f32(
  xnn_operator_t argmax_pooling_op,
  const float* input,
  float* output,
  uint32_t* index);

enum xnn_status xnn_create_average_pooling2d_nhwc_f16(
  uint32_t input_padding_top,
  uint32_t input_padding_right,
  uint32_t input_padding_bottom,
  uint32_t input_padding_left,
  uint32_t pooling_height,
  uint32_t pooling_width,
  uint32_t stride_height,
  uint32_t stride_width,
  float output_min,
  float output_max,
  uint32_t flags,
  xnn_operator_t* average_pooling_op_out);

enum xnn_status xnn_reshape_average_pooling2d_nhwc_f16(
  xnn_operator_t average_pooling_op,
  size_t batch_size,
  size_t input_height,
  size_t input_width,
  size_t channels,
  size_t input_pixel_stride,
  size_t output_pixel_stride,
  size_t* output_height_out,
  size_t* output_width_out,
  pthreadpool_t threadpool);

enum xnn_status xnn_setup_average_pooling2d_nhwc_f16(
  xnn_operator_t average_pooling_op,
  const void* input,
  void* output);

enum xnn_status xnn_create_average_pooling2d_nhwc_f32(
  uint32_t input_padding_top,
  uint32_t input_padding_right,
  uint32_t input_padding_bottom,
  uint32_t input_padding_left,
  uint32_t pooling_height,
  uint32_t pooling_width,
  uint32_t stride_height,
  uint32_t stride_width,
  float output_min,
  float output_max,
  uint32_t flags,
  xnn_operator_t* average_pooling_op_out);

enum xnn_status xnn_reshape_average_pooling2d_nhwc_f32(
  xnn_operator_t average_pooling_op,
  size_t batch_size,
  size_t input_height,
  size_t input_width,
  size_t channels,
  size_t input_pixel_stride,
  size_t output_pixel_stride,
  size_t* output_height_out,
  size_t* output_width_out,
  pthreadpool_t threadpool);

enum xnn_status xnn_setup_average_pooling2d_nhwc_f32(
  xnn_operator_t average_pooling_op,
  const float* input,
  float* output);

enum xnn_status xnn_create_batch_matrix_multiply_nc_f16(
  uint32_t flags,
  xnn_operator_t* batch_matrix_multiply_op);

enum xnn_status xnn_create_batch_matrix_multiply_nc_bf16_f32(
    uint32_t flags, xnn_operator_t* batch_matrix_multiply_op);

enum xnn_status xnn_create_batch_matrix_multiply_nc_f16_const_weights(
    size_t batch_size_b, size_t k, size_t n, const void* data_b, uint32_t flags,
    xnn_operator_t* batch_matrix_multiply_op);

enum xnn_status xnn_reshape_batch_matrix_multiply_nc_f16(
    xnn_operator_t batch_matrix_multiply_op, size_t num_batch_dims,
    const size_t* batch_dims_a, const size_t* batch_dims_b, size_t m, size_t k,
    size_t n, size_t* workspace_size, size_t* workspace_alignment,
    pthreadpool_t threadpool);

enum xnn_status xnn_setup_batch_matrix_multiply_nc_f16(
    xnn_operator_t batch_matrix_multiply_op, void* workspace,
    const void* input_a, const void* input_b, void* output);

enum xnn_status xnn_reshape_batch_matrix_multiply_nc_bf16_f32(
    xnn_operator_t batch_matrix_multiply_op, size_t num_batch_dims,
    const size_t* batch_dims_a, const size_t* batch_dims_b, size_t m, size_t k,
    size_t n, size_t* workspace_size, size_t* workspace_alignment,
    pthreadpool_t threadpool);

enum xnn_status xnn_setup_batch_matrix_multiply_nc_bf16_f32(
    xnn_operator_t batch_matrix_multiply_op, void* workspace,
    const void* input_a, const void* input_b, void* output);

enum xnn_status xnn_create_batch_matrix_multiply_nc_f32(
    uint32_t flags, xnn_operator_t* batch_matrix_multiply_op);

enum xnn_status xnn_create_batch_matrix_multiply_nc_f32_const_weights(
    size_t batch_size_b, size_t k, size_t n, const float* data_b,
    uint32_t flags, xnn_operator_t* batch_matrix_multiply_op);

enum xnn_status xnn_reshape_batch_matrix_multiply_nc_f32(
    xnn_operator_t batch_matrix_multiply_op, size_t num_batch_dims,
    const size_t* batch_dims_a, const size_t* batch_dims_b, size_t m, size_t k,
    size_t n, size_t* workspace_size, size_t* workspace_alignment,
    pthreadpool_t threadpool);

enum xnn_status xnn_setup_batch_matrix_multiply_nc_f32(
    xnn_operator_t batch_matrix_multiply_op, void* workspace,
    const float* input_a, const float* input_b, float* output);

enum xnn_status xnn_create_batch_matrix_multiply_nc_qd8_f32_qc8w(
    size_t batch_size_b, size_t k, size_t n, const int8_t* data_b,
    const float* scale_b, uint32_t flags,
    xnn_operator_t* batch_matrix_multiply_op);

enum xnn_status xnn_reshape_batch_matrix_multiply_nc_qd8_f32_qc8w(
    xnn_operator_t batch_matrix_multiply_op, size_t num_batch_dims,
    const size_t* batch_dims_a, const size_t* batch_dims_b, size_t m, size_t k,
    size_t n, pthreadpool_t threadpool);

enum xnn_status xnn_setup_batch_matrix_multiply_nc_qd8_f32_qc8w(
    xnn_operator_t batch_matrix_multiply_op, const int8_t* input_a,
    const struct xnn_quantization_params* quantization_params,
    float* output);

enum xnn_status xnn_create_constant_pad_nd_x8(
  const void* padding_value,
  uint32_t flags,
  xnn_operator_t* constant_pad_op_out);

enum xnn_status xnn_reshape_constant_pad_nd_x8(
  xnn_operator_t constant_pad_op,
  size_t num_dims,
  const size_t* input_shape,
  const size_t* pre_padding,
  const size_t* post_padding,
  pthreadpool_t threadpool);

enum xnn_status xnn_setup_constant_pad_nd_x8(
  xnn_operator_t constant_pad_op,
  const void* input,
  void* output);

enum xnn_status xnn_run_constant_pad_nd_x8(
  uint32_t flags,
  size_t num_dims,
  const size_t* input_shape,
  const size_t* pre_paddings,
  const size_t* post_paddings,
  const void* input,
  void* output,
  const void* padding_value,
  pthreadpool_t threadpool);

enum xnn_status xnn_create_constant_pad_nd_x16(
  const void* padding_value,
  uint32_t flags,
  xnn_operator_t* constant_pad_op_out);

enum xnn_status xnn_reshape_constant_pad_nd_x16(
  xnn_operator_t constant_pad_op,
  size_t num_dims,
  const size_t* input_shape,
  const size_t* pre_padding,
  const size_t* post_padding,
  pthreadpool_t threadpool);

enum xnn_status xnn_setup_constant_pad_nd_x16(
  xnn_operator_t constant_pad_op,
  const void* input,
  void* output);

enum xnn_status xnn_run_constant_pad_nd_x16(
  uint32_t flags,
  size_t num_dims,
  const size_t* input_shape,
  const size_t* pre_paddings,
  const size_t* post_paddings,
  const void* input,
  void* output,
  const void* padding_value,
  pthreadpool_t threadpool);

enum xnn_status xnn_create_constant_pad_nd_x32(
  const void* padding_value,
  uint32_t flags,
  xnn_operator_t* constant_pad_op_out);

enum xnn_status xnn_reshape_constant_pad_nd_x32(
  xnn_operator_t constant_pad_op,
  size_t num_dims,
  const size_t* input_shape,
  const size_t* pre_padding,
  const size_t* post_padding,
  pthreadpool_t threadpool);

enum xnn_status xnn_setup_constant_pad_nd_x32(
  xnn_operator_t constant_pad_op,
  const void* input,
  void* output);

enum xnn_status xnn_run_constant_pad_nd_x32(
  uint32_t flags,
  size_t num_dims,
  const size_t* input_shape,
  const size_t* pre_paddings,
  const size_t* post_paddings,
  const void* input,
  void* output,
  const void* padding_value,
  pthreadpool_t threadpool);

enum xnn_status xnn_create_convert_nc_f16_qd8(
  uint32_t flags,
  xnn_operator_t* convert_op_out);

enum xnn_status xnn_reshape_convert_nc_f16_qd8(
  xnn_operator_t convert_op,
  size_t batch_size,
  size_t channels,
  size_t input_stride,
  size_t output_stride,
  pthreadpool_t threadpool);

// quantization_params must be padded with at least XNN_EXTRA_QUANTIZATION_PARAMS entries.
enum xnn_status xnn_setup_convert_nc_f16_qd8(
  xnn_operator_t convert_op,
  const void* input,
  int8_t* output,
  struct xnn_quantization_params* quantization_params);

enum xnn_status xnn_create_convert_nc_f32_qd8(
  uint32_t flags,
  xnn_operator_t* convert_op_out);

enum xnn_status xnn_reshape_convert_nc_f32_qd8(
  xnn_operator_t convert_op,
  size_t batch_size,
  size_t channels,
  size_t input_stride,
  size_t output_stride,
  pthreadpool_t threadpool);

// quantization_params must be padded with at least XNN_EXTRA_QUANTIZATION_PARAMS entries.
enum xnn_status xnn_setup_convert_nc_f32_qd8(
  xnn_operator_t convert_op,
  const float* input,
  int8_t* output,
  struct xnn_quantization_params* quantization_params);

XNN_DEPRECATED enum xnn_status xnn_run_convert_nc_f32_f16(
  size_t channels,
  size_t input_stride,
  size_t output_stride,
  size_t batch_size,
  const float* input,
  void* output,
  uint32_t flags,
  pthreadpool_t threadpool);

enum xnn_status xnn_create_convolution2d_nchw_f16(
  uint32_t input_padding_top,
  uint32_t input_padding_right,
  uint32_t input_padding_bottom,
  uint32_t input_padding_left,
  uint32_t kernel_height,
  uint32_t kernel_width,
  uint32_t subsampling_height,
  uint32_t subsampling_width,
  uint32_t dilation_height,
  uint32_t dilation_width,
  uint32_t groups,
  size_t group_input_channels,
  size_t group_output_channels,
  size_t input_channel_stride,
  size_t output_channel_stride,
  const void* kernel,
  const void* bias,
  float output_min,
  float output_max,
  uint32_t flags,
  xnn_code_cache_t code_cache,
  xnn_weights_cache_t weights_cache,
  xnn_operator_t* convolution_op_out);

enum xnn_status xnn_reshape_convolution2d_nchw_f16(
  xnn_operator_t convolution_op,
  size_t batch_size,
  size_t input_height,
  size_t input_width,
  size_t* output_height_out,
  size_t* output_width_out,
  pthreadpool_t threadpool);

enum xnn_status xnn_setup_convolution2d_nchw_f16(
  xnn_operator_t convolution_op,
  const void* input,
  void* output);

enum xnn_status xnn_create_convolution2d_nchw_f32(
  uint32_t input_padding_top,
  uint32_t input_padding_right,
  uint32_t input_padding_bottom,
  uint32_t input_padding_left,
  uint32_t kernel_height,
  uint32_t kernel_width,
  uint32_t subsampling_height,
  uint32_t subsampling_width,
  uint32_t dilation_height,
  uint32_t dilation_width,
  uint32_t groups,
  size_t group_input_channels,
  size_t group_output_channels,
  size_t input_channel_stride,
  size_t output_channel_stride,
  const float* kernel,
  const float* bias,
  float output_min,
  float output_max,
  uint32_t flags,
  xnn_code_cache_t code_cache,
  xnn_weights_cache_t weights_cache,
  xnn_operator_t* convolution_op_out);

enum xnn_status xnn_reshape_convolution2d_nchw_f32(
  xnn_operator_t convolution_op,
  size_t batch_size,
  size_t input_height,
  size_t input_width,
  size_t* output_height_out,
  size_t* output_width_out,
  pthreadpool_t threadpool);

enum xnn_status xnn_setup_convolution2d_nchw_f32(
  xnn_operator_t convolution_op,
  const float* input,
  float* output);

enum xnn_status xnn_create_convolution2d_nhwc_f16(
  uint32_t input_padding_top,
  uint32_t input_padding_right,
  uint32_t input_padding_bottom,
  uint32_t input_padding_left,
  uint32_t kernel_height,
  uint32_t kernel_width,
  uint32_t subsampling_height,
  uint32_t subsampling_width,
  uint32_t dilation_height,
  uint32_t dilation_width,
  uint32_t groups,
  size_t group_input_channels,
  size_t group_output_channels,
  size_t input_channel_stride,
  size_t output_channel_stride,
  const void* kernel,
  const void* bias,
  float output_min,
  float output_max,
  uint32_t flags,
  xnn_code_cache_t code_cache,
  xnn_weights_cache_t weights_cache,
  xnn_operator_t* convolution_op_out);

enum xnn_status xnn_reshape_convolution2d_nhwc_f16(
  xnn_operator_t convolution_op,
  size_t batch_size,
  size_t input_height,
  size_t input_width,
  size_t* workspace_size,
  size_t* workspace_alignment,
  size_t* output_height_out,
  size_t* output_width_out,
  pthreadpool_t threadpool);

enum xnn_status xnn_setup_convolution2d_nhwc_f16(
  xnn_operator_t convolution_op,
  void* workspace,
  const void* input,
  void* output);

enum xnn_status xnn_create_convolution2d_nhwc_f32(
  uint32_t input_padding_top,
  uint32_t input_padding_right,
  uint32_t input_padding_bottom,
  uint32_t input_padding_left,
  uint32_t kernel_height,
  uint32_t kernel_width,
  uint32_t subsampling_height,
  uint32_t subsampling_width,
  uint32_t dilation_height,
  uint32_t dilation_width,
  uint32_t groups,
  size_t group_input_channels,
  size_t group_output_channels,
  size_t input_channel_stride,
  size_t output_channel_stride,
  const float* kernel,
  const float* bias,
  float output_min,
  float output_max,
  uint32_t flags,
  xnn_code_cache_t code_cache,
  xnn_weights_cache_t weights_cache,
  xnn_operator_t* convolution_op_out);

enum xnn_status xnn_create_convolution2d_nhwc_f32_f16(
  uint32_t input_padding_top,
  uint32_t input_padding_right,
  uint32_t input_padding_bottom,
  uint32_t input_padding_left,
  uint32_t kernel_height,
  uint32_t kernel_width,
  uint32_t subsampling_height,
  uint32_t subsampling_width,
  uint32_t dilation_height,
  uint32_t dilation_width,
  uint32_t groups,
  size_t group_input_channels,
  size_t group_output_channels,
  size_t input_channel_stride,
  size_t output_channel_stride,
  const void* kernel,
  const void* bias,
  float output_min,
  float output_max,
  uint32_t flags,
  xnn_code_cache_t code_cache,
  xnn_weights_cache_t weights_cache,
  xnn_operator_t* convolution_op_out);

// Forward declare.
struct xnn_post_operation;

/// Deprecated
enum xnn_status xnn_create_fused_convolution2d_nhwc_f32(
    uint32_t input_padding_top,
    uint32_t input_padding_right,
    uint32_t input_padding_bottom,
    uint32_t input_padding_left,
    uint32_t kernel_height,
    uint32_t kernel_width,
    uint32_t subsampling_height,
    uint32_t subsampling_width,
    uint32_t dilation_height,
    uint32_t dilation_width,
    uint32_t groups,
    size_t group_input_channels,
    size_t group_output_channels,
    size_t input_channel_stride,
    size_t output_channel_stride,
    const float* kernel,
    const float* bias,
    size_t num_post_operations,
    struct xnn_post_operation* post_operations,
    uint32_t flags,
    xnn_code_cache_t code_cache,
    xnn_weights_cache_t weights_cache,
    xnn_operator_t* convolution_op_out);

enum xnn_status xnn_reshape_convolution2d_nhwc_f32(
  xnn_operator_t convolution_op,
  size_t batch_size,
  size_t input_height,
  size_t input_width,
  size_t* workspace_size,
  size_t* workspace_alignment,
  size_t* output_height_out,
  size_t* output_width_out,
  pthreadpool_t threadpool);

enum xnn_status xnn_setup_convolution2d_nhwc_f32(
  xnn_operator_t convolution_op,
  void* workspace,
  const float* input,
  float* output);

enum xnn_status xnn_create_convolution2d_nhwc_qd8_f16_qc8w(
    uint32_t input_padding_top, uint32_t input_padding_right,
    uint32_t input_padding_bottom, uint32_t input_padding_left,
    uint32_t kernel_height, uint32_t kernel_width, uint32_t subsampling_height,
    uint32_t subsampling_width, uint32_t dilation_height,
    uint32_t dilation_width, uint32_t groups, size_t group_input_channels,
    size_t group_output_channels, size_t input_channel_stride,
    size_t output_channel_stride, const float* kernel_scale,
    const int8_t* kernel, const float* bias, float output_min, float output_max,
    uint32_t flags, xnn_code_cache_t code_cache,
    xnn_weights_cache_t weights_cache, xnn_operator_t* convolution_op_out);

enum xnn_status xnn_create_convolution2d_nhwc_qd8_f32_qc8w(
    uint32_t input_padding_top, uint32_t input_padding_right,
    uint32_t input_padding_bottom, uint32_t input_padding_left,
    uint32_t kernel_height, uint32_t kernel_width, uint32_t subsampling_height,
    uint32_t subsampling_width, uint32_t dilation_height,
    uint32_t dilation_width, uint32_t groups, size_t group_input_channels,
    size_t group_output_channels, size_t input_channel_stride,
    size_t output_channel_stride, const float* kernel_scale,
    const int8_t* kernel, const float* bias, float output_min, float output_max,
    uint32_t flags, xnn_code_cache_t code_cache,
    xnn_weights_cache_t weights_cache, xnn_operator_t* convolution_op_out);

enum xnn_status xnn_create_convolution2d_nhwc_qs8(
  uint32_t input_padding_top,
  uint32_t input_padding_right,
  uint32_t input_padding_bottom,
  uint32_t input_padding_left,
  uint32_t kernel_height,
  uint32_t kernel_width,
  uint32_t subsampling_height,
  uint32_t subsampling_width,
  uint32_t dilation_height,
  uint32_t dilation_width,
  uint32_t groups,
  size_t group_input_channels,
  size_t group_output_channels,
  size_t input_channel_stride,
  size_t output_channel_stride,
  int8_t input_zero_point,
  float input_scale,
  float kernel_scale,
  const int8_t* kernel,
  const int32_t* bias,
  int8_t output_zero_point,
  float output_scale,
  int8_t output_min,
  int8_t output_max,
  uint32_t flags,
  xnn_code_cache_t code_cache,
  xnn_weights_cache_t weights_cache,
  xnn_operator_t* convolution_op_out);

enum xnn_status xnn_reshape_convolution2d_nhwc_qd8_f16_qc8w(
    xnn_operator_t convolution_op, size_t batch_size, size_t input_height,
    size_t input_width, size_t* workspace_size, size_t* workspace_alignment,
    size_t* output_height_out, size_t* output_width_out,
    pthreadpool_t threadpool);

enum xnn_status xnn_reshape_convolution2d_nhwc_qd8_f32_qc8w(
    xnn_operator_t convolution_op, size_t batch_size, size_t input_height,
    size_t input_width, size_t* workspace_size, size_t* workspace_alignment,
    size_t* output_height_out, size_t* output_width_out,
    pthreadpool_t threadpool);

enum xnn_status xnn_reshape_convolution2d_nhwc_qs8(
  xnn_operator_t convolution_op,
  size_t batch_size,
  size_t input_height,
  size_t input_width,
  size_t* workspace_size,
  size_t* workspace_alignment,
  size_t* output_height_out,
  size_t* output_width_out,
  pthreadpool_t threadpool);

enum xnn_status xnn_setup_convolution2d_nhwc_qd8_f16_qc8w(
    xnn_operator_t convolution_op, void* workspace, const int8_t* input,
    void* output,
    const struct xnn_quantization_params* quantization_params);

enum xnn_status xnn_setup_convolution2d_nhwc_qd8_f32_qc8w(
    xnn_operator_t convolution_op, void* workspace, const int8_t* input,
    float* output,
    const struct xnn_quantization_params* quantization_params);

enum xnn_status xnn_setup_convolution2d_nhwc_qs8(
  xnn_operator_t convolution_op,
  void* workspace,
  const int8_t* input,
  int8_t* output);

enum xnn_status xnn_create_convolution2d_nhwc_qs8_qc8w(
  uint32_t input_padding_top,
  uint32_t input_padding_right,
  uint32_t input_padding_bottom,
  uint32_t input_padding_left,
  uint32_t kernel_height,
  uint32_t kernel_width,
  uint32_t subsampling_height,
  uint32_t subsampling_width,
  uint32_t dilation_height,
  uint32_t dilation_width,
  uint32_t groups,
  size_t group_input_channels,
  size_t group_output_channels,
  size_t input_channel_stride,
  size_t output_channel_stride,
  int8_t input_zero_point,
  float input_scale,
  const float* kernel_scale,
  const int8_t* kernel,
  const int32_t* bias,
  int8_t output_zero_point,
  float output_scale,
  int8_t output_min,
  int8_t output_max,
  uint32_t flags,
  xnn_code_cache_t code_cache,
  xnn_weights_cache_t weights_cache,
  xnn_operator_t* convolution_op_out);

enum xnn_status xnn_reshape_convolution2d_nhwc_qs8_qc8w(
  xnn_operator_t convolution_op,
  size_t batch_size,
  size_t input_height,
  size_t input_width,
  size_t* workspace_size,
  size_t* workspace_alignment,
  size_t* output_height_out,
  size_t* output_width_out,
  pthreadpool_t threadpool);

enum xnn_status xnn_setup_convolution2d_nhwc_qs8_qc8w(
  xnn_operator_t convolution_op,
  void* workspace,
  const int8_t* input,
  int8_t* output);

enum xnn_status xnn_create_convolution2d_nhwc_qu8(
  uint32_t input_padding_top,
  uint32_t input_padding_right,
  uint32_t input_padding_bottom,
  uint32_t input_padding_left,
  uint32_t kernel_height,
  uint32_t kernel_width,
  uint32_t subsampling_height,
  uint32_t subsampling_width,
  uint32_t dilation_height,
  uint32_t dilation_width,
  uint32_t groups,
  size_t group_input_channels,
  size_t group_output_channels,
  size_t input_channel_stride,
  size_t output_channel_stride,
  uint8_t input_zero_point,
  float input_scale,
  uint8_t kernel_zero_point,
  float kernel_scale,
  const uint8_t* kernel,
  const int32_t* bias,
  uint8_t output_zero_point,
  float output_scale,
  uint8_t output_min,
  uint8_t output_max,
  uint32_t flags,
  xnn_code_cache_t code_cache,
  xnn_weights_cache_t weights_cache,
  xnn_operator_t* convolution_op_out);

enum xnn_status xnn_reshape_convolution2d_nhwc_qu8(
  xnn_operator_t convolution_op,
  size_t batch_size,
  size_t input_height,
  size_t input_width,
  size_t* workspace_size,
  size_t* workspace_alignment,
  size_t* output_height_out,
  size_t* output_width_out,
  pthreadpool_t threadpool);

enum xnn_status xnn_setup_convolution2d_nhwc_qu8(
  xnn_operator_t convolution_op,
  void* workspace,
  const uint8_t* input,
  uint8_t* output);

enum xnn_status xnn_create_copy_nc_x8(
  uint32_t flags,
  xnn_operator_t* copy_op_out);

enum xnn_status xnn_reshape_copy_nc_x8(
  xnn_operator_t copy_op,
  size_t batch_size,
  size_t channels,
  size_t input_stride,
  size_t output_stride,
  pthreadpool_t threadpool);

enum xnn_status xnn_setup_copy_nc_x8(
  xnn_operator_t copy_op,
  const void* input,
  void* output);

enum xnn_status xnn_create_copy_nc_x16(
  uint32_t flags,
  xnn_operator_t* copy_op_out);

enum xnn_status xnn_reshape_copy_nc_x16(
  xnn_operator_t copy_op,
  size_t batch_size,
  size_t channels,
  size_t input_stride,
  size_t output_stride,
  pthreadpool_t threadpool);

enum xnn_status xnn_setup_copy_nc_x16(
  xnn_operator_t copy_op,
  const void* input,
  void* output);

enum xnn_status xnn_create_copy_nc_x32(
  uint32_t flags,
  xnn_operator_t* copy_op_out);

enum xnn_status xnn_reshape_copy_nc_x32(
  xnn_operator_t copy_op,
  size_t batch_size,
  size_t channels,
  size_t input_stride,
  size_t output_stride,
  pthreadpool_t threadpool);

enum xnn_status xnn_setup_copy_nc_x32(
  xnn_operator_t copy_op,
  const void* input,
  void* output);

enum xnn_status xnn_run_copy_nc_x32(
  size_t channels,
  size_t input_stride,
  size_t output_stride,
  size_t batch_size,
  const uint32_t* input,
  uint32_t* output,
  uint32_t flags,
  pthreadpool_t threadpool);

enum xnn_status xnn_create_deconvolution2d_nhwc_f16(
  uint32_t output_padding_top,
  uint32_t output_padding_right,
  uint32_t output_padding_bottom,
  uint32_t output_padding_left,
  uint32_t kernel_height,
  uint32_t kernel_width,
  uint32_t stride_height,
  uint32_t stride_width,
  uint32_t dilation_height,
  uint32_t dilation_width,
  uint32_t groups,
  size_t group_input_channels,
  size_t group_output_channels,
  size_t input_pixel_stride,
  size_t output_pixel_stride,
  const void* kernel,
  const void* bias,
  float output_min,
  float output_max,
  uint32_t flags,
  xnn_code_cache_t code_cache,
  xnn_weights_cache_t weights_cache,
  xnn_operator_t* deconvolution_op_out);

enum xnn_status xnn_reshape_deconvolution2d_nhwc_f16(
  xnn_operator_t deconvolution_op,
  size_t batch_size,
  size_t input_height,
  size_t input_width,
  uint32_t adjustment_height,
  uint32_t adjustment_width,
  size_t* output_height_out,
  size_t* output_width_out,
  pthreadpool_t threadpool);

enum xnn_status xnn_setup_deconvolution2d_nhwc_f16(
  xnn_operator_t deconvolution_op,
  const void* input,
  void* output);

enum xnn_status xnn_create_deconvolution2d_nhwc_f32(
  uint32_t output_padding_top,
  uint32_t output_padding_right,
  uint32_t output_padding_bottom,
  uint32_t output_padding_left,
  uint32_t kernel_height,
  uint32_t kernel_width,
  uint32_t stride_height,
  uint32_t stride_width,
  uint32_t dilation_height,
  uint32_t dilation_width,
  uint32_t groups,
  size_t group_input_channels,
  size_t group_output_channels,
  size_t input_pixel_stride,
  size_t output_pixel_stride,
  const float* kernel,
  const float* bias,
  float output_min,
  float output_max,
  uint32_t flags,
  xnn_code_cache_t code_cache,
  xnn_weights_cache_t weights_cache,
  xnn_operator_t* deconvolution_op_out);

enum xnn_status xnn_create_deconvolution2d_nhwc_f32_f16(
  uint32_t output_padding_top,
  uint32_t output_padding_right,
  uint32_t output_padding_bottom,
  uint32_t output_padding_left,
  uint32_t kernel_height,
  uint32_t kernel_width,
  uint32_t stride_height,
  uint32_t stride_width,
  uint32_t dilation_height,
  uint32_t dilation_width,
  uint32_t groups,
  size_t group_input_channels,
  size_t group_output_channels,
  size_t input_pixel_stride,
  size_t output_pixel_stride,
  const void* kernel,
  const void* bias,
  float output_min,
  float output_max,
  uint32_t flags,
  xnn_code_cache_t code_cache,
  xnn_weights_cache_t weights_cache,
  xnn_operator_t* deconvolution_op_out);

enum xnn_status xnn_reshape_deconvolution2d_nhwc_f32(
  xnn_operator_t deconvolution_op,
  size_t batch_size,
  size_t input_height,
  size_t input_width,
  uint32_t adjustment_height,
  uint32_t adjustment_width,
  size_t* output_height_out,
  size_t* output_width_out,
  pthreadpool_t threadpool);

enum xnn_status xnn_setup_deconvolution2d_nhwc_f32(
  xnn_operator_t deconvolution_op,
  const float* input,
  float* output);

enum xnn_status xnn_create_deconvolution2d_nhwc_qd8_f32_qc8w(
  uint32_t output_padding_top,
  uint32_t output_padding_right,
  uint32_t output_padding_bottom,
  uint32_t output_padding_left,
  uint32_t kernel_height,
  uint32_t kernel_width,
  uint32_t stride_height,
  uint32_t stride_width,
  uint32_t dilation_height,
  uint32_t dilation_width,
  uint32_t groups,
  size_t group_input_channels,
  size_t group_output_channels,
  size_t input_pixel_stride,
  size_t output_pixel_stride,
  const float* kernel_scale,
  const int8_t* kernel,
  const float* bias,
  float output_min,
  float output_max,
  uint32_t flags,
  xnn_code_cache_t code_cache,
  xnn_weights_cache_t weights_cache,
  xnn_operator_t* deconvolution_op_out);

enum xnn_status xnn_reshape_deconvolution2d_nhwc_qd8_f32_qc8w(
  xnn_operator_t deconvolution_op,
  size_t batch_size,
  size_t input_height,
  size_t input_width,
  uint32_t adjustment_height,
  uint32_t adjustment_width,
  size_t* output_height_out,
  size_t* output_width_out,
  pthreadpool_t threadpool);

enum xnn_status xnn_setup_deconvolution2d_nhwc_qd8_f32_qc8w(
  xnn_operator_t deconvolution_op,
  const int8_t* input,
  float* output,
  const struct xnn_quantization_params* quantization_params);

enum xnn_status xnn_create_deconvolution2d_nhwc_qs8(
  uint32_t output_padding_top,
  uint32_t output_padding_right,
  uint32_t output_padding_bottom,
  uint32_t output_padding_left,
  uint32_t kernel_height,
  uint32_t kernel_width,
  uint32_t stride_height,
  uint32_t stride_width,
  uint32_t dilation_height,
  uint32_t dilation_width,
  uint32_t groups,
  size_t group_input_channels,
  size_t group_output_channels,
  size_t input_pixel_stride,
  size_t output_pixel_stride,
  int8_t input_zero_point,
  float input_scale,
  float kernel_scale,
  const int8_t* kernel,
  const int32_t* bias,
  int8_t output_zero_point,
  float output_scale,
  int8_t output_min,
  int8_t output_max,
  uint32_t flags,
  xnn_code_cache_t code_cache,
  xnn_weights_cache_t weights_cache,
  xnn_operator_t* deconvolution_op_out);

enum xnn_status xnn_reshape_deconvolution2d_nhwc_qs8(
  xnn_operator_t deconvolution_op,
  size_t batch_size,
  size_t input_height,
  size_t input_width,
  uint32_t adjustment_height,
  uint32_t adjustment_width,
  size_t* output_height_out,
  size_t* output_width_out,
  pthreadpool_t threadpool);

enum xnn_status xnn_setup_deconvolution2d_nhwc_qs8(
  xnn_operator_t deconvolution_op,
  const int8_t* input,
  int8_t* output);

enum xnn_status xnn_create_deconvolution2d_nhwc_qs8_qc8w(
  uint32_t output_padding_top,
  uint32_t output_padding_right,
  uint32_t output_padding_bottom,
  uint32_t output_padding_left,
  uint32_t kernel_height,
  uint32_t kernel_width,
  uint32_t stride_height,
  uint32_t stride_width,
  uint32_t dilation_height,
  uint32_t dilation_width,
  uint32_t groups,
  size_t group_input_channels,
  size_t group_output_channels,
  size_t input_pixel_stride,
  size_t output_pixel_stride,
  int8_t input_zero_point,
  float input_scale,
  const float* kernel_scale,
  const int8_t* kernel,
  const int32_t* bias,
  int8_t output_zero_point,
  float output_scale,
  int8_t output_min,
  int8_t output_max,
  uint32_t flags,
  xnn_code_cache_t code_cache,
  xnn_weights_cache_t weights_cache,
  xnn_operator_t* deconvolution_op_out);

enum xnn_status xnn_reshape_deconvolution2d_nhwc_qs8_qc8w(
  xnn_operator_t deconvolution_op,
  size_t batch_size,
  size_t input_height,
  size_t input_width,
  uint32_t adjustment_height,
  uint32_t adjustment_width,
  size_t* output_height_out,
  size_t* output_width_out,
  pthreadpool_t threadpool);

enum xnn_status xnn_setup_deconvolution2d_nhwc_qs8_qc8w(
  xnn_operator_t deconvolution_op,
  const int8_t* input,
  int8_t* output);

enum xnn_status xnn_create_deconvolution2d_nhwc_qu8(
  uint32_t output_padding_top,
  uint32_t output_padding_right,
  uint32_t output_padding_bottom,
  uint32_t output_padding_left,
  uint32_t kernel_height,
  uint32_t kernel_width,
  uint32_t stride_height,
  uint32_t stride_width,
  uint32_t dilation_height,
  uint32_t dilation_width,
  uint32_t groups,
  size_t group_input_channels,
  size_t group_output_channels,
  size_t input_pixel_stride,
  size_t output_pixel_stride,
  uint8_t input_zero_point,
  float input_scale,
  uint8_t kernel_zero_point,
  float kernel_scale,
  const uint8_t* kernel,
  const int32_t* bias,
  uint8_t output_zero_point,
  float output_scale,
  uint8_t output_min,
  uint8_t output_max,
  uint32_t flags,
  xnn_code_cache_t code_cache,
  xnn_weights_cache_t weights_cache,
  xnn_operator_t* deconvolution_op_out);

enum xnn_status xnn_reshape_deconvolution2d_nhwc_qu8(
  xnn_operator_t deconvolution_op,
  size_t batch_size,
  size_t input_height,
  size_t input_width,
  uint32_t adjustment_height,
  uint32_t adjustment_width,
  size_t* output_height_out,
  size_t* output_width_out,
  pthreadpool_t threadpool);

enum xnn_status xnn_setup_deconvolution2d_nhwc_qu8(
  xnn_operator_t deconvolution_op,
  const uint8_t* input,
  uint8_t* output);

enum xnn_status xnn_create_depth_to_space_nchw2nhwc_x16(
  uint32_t block_size,
  uint32_t flags,
  xnn_operator_t* depth_to_space_op_out);

enum xnn_status xnn_reshape_depth_to_space_nchw2nhwc_x16(
  xnn_operator_t depth_to_space_op,
  size_t batch_size,
  size_t input_height,
  size_t input_width,
  size_t input_channels,
  size_t* output_height_out,
  size_t* output_width_out,
  size_t* output_channels_out,
  pthreadpool_t threadpool);

enum xnn_status xnn_setup_depth_to_space_nchw2nhwc_x16(
  xnn_operator_t depth_to_space_op,
  const void* input,
  void* output);

enum xnn_status xnn_create_depth_to_space_nchw2nhwc_x32(
  uint32_t block_size,
  uint32_t flags,
  xnn_operator_t* depth_to_space_op_out);

enum xnn_status xnn_reshape_depth_to_space_nchw2nhwc_x32(
  xnn_operator_t depth_to_space_op,
  size_t batch_size,
  size_t input_height,
  size_t input_width,
  size_t input_channels,
  size_t* output_height_out,
  size_t* output_width_out,
  size_t* output_channels_out,
  pthreadpool_t threadpool);

enum xnn_status xnn_setup_depth_to_space_nchw2nhwc_x32(
  xnn_operator_t depth_to_space_op,
  const void* input,
  void* output);

enum xnn_status xnn_create_depth_to_space_nhwc_x8(
  uint32_t block_size,
  uint32_t flags,
  xnn_operator_t* depth_to_space_op_out);

enum xnn_status xnn_reshape_depth_to_space_nhwc_x8(
  xnn_operator_t depth_to_space_op,
  size_t batch_size,
  size_t input_height,
  size_t input_width,
  size_t input_channels,
  size_t* output_height_out,
  size_t* output_width_out,
  size_t* output_channels_out,
  pthreadpool_t threadpool);

enum xnn_status xnn_setup_depth_to_space_nhwc_x8(
  xnn_operator_t depth_to_space_op,
  const void* input,
  void* output);

enum xnn_status xnn_create_depth_to_space_nhwc_x16(
  uint32_t block_size,
  uint32_t flags,
  xnn_operator_t* depth_to_space_op_out);

enum xnn_status xnn_reshape_depth_to_space_nhwc_x16(
  xnn_operator_t depth_to_space_op,
  size_t batch_size,
  size_t input_height,
  size_t input_width,
  size_t input_channels,
  size_t* output_height_out,
  size_t* output_width_out,
  size_t* output_channels_out,
  pthreadpool_t threadpool);

enum xnn_status xnn_setup_depth_to_space_nhwc_x16(
  xnn_operator_t depth_to_space_op,
  const void* input,
  void* output);

enum xnn_status xnn_create_depth_to_space_nhwc_x32(
  uint32_t block_size,
  uint32_t flags,
  xnn_operator_t* depth_to_space_op_out);

enum xnn_status xnn_reshape_depth_to_space_nhwc_x32(
  xnn_operator_t depth_to_space_op,
  size_t batch_size,
  size_t input_height,
  size_t input_width,
  size_t input_channels,
  size_t* output_height_out,
  size_t* output_width_out,
  size_t* output_channels_out,
  pthreadpool_t threadpool);

enum xnn_status xnn_setup_depth_to_space_nhwc_x32(
  xnn_operator_t depth_to_space_op,
  const void* input,
  void* output);

enum xnn_status xnn_create_dynamic_fully_connected_nc_f16(
  float output_min,
  float output_max,
  uint32_t flags,
  xnn_operator_t* dynamic_fully_connected_op_out);

enum xnn_status xnn_reshape_dynamic_fully_connected_nc_f16(
  xnn_operator_t dynamic_fully_connected_op,
  size_t batch_size,
  size_t input_channels,
  size_t output_channels,
  size_t input_stride,
  size_t output_stride,
  size_t* workspace_size,
  size_t* workspace_alignment,
  pthreadpool_t threadpool);

enum xnn_status xnn_setup_dynamic_fully_connected_nc_f16(
  xnn_operator_t dynamic_fully_connected_op,
  void* workspace,
  const void* input,
  const void* kernel,
  const void* bias,
  void* output);

enum xnn_status xnn_create_dynamic_fully_connected_nc_f32(
  float output_min,
  float output_max,
  uint32_t flags,
  xnn_operator_t* dynamic_fully_connected_op_out);

enum xnn_status xnn_reshape_dynamic_fully_connected_nc_f32(
  xnn_operator_t dynamic_fully_connected_op,
  size_t batch_size,
  size_t input_channels,
  size_t output_channels,
  size_t input_stride,
  size_t output_stride,
  size_t* workspace_size,
  size_t* workspace_alignment,
  pthreadpool_t threadpool);

enum xnn_status xnn_setup_dynamic_fully_connected_nc_f32(
  xnn_operator_t dynamic_fully_connected_op,
  void* workspace,
  const float* input,
  const float* kernel,
  const float* bias,
  float* output);

enum xnn_status xnn_create_fully_connected_nc_f16(
  size_t input_channels,
  size_t output_channels,
  size_t input_stride,
  size_t output_stride,
  const void* kernel,
  const void* bias,
  float output_min,
  float output_max,
  uint32_t flags,
  xnn_code_cache_t code_cache,
  xnn_weights_cache_t weights_cache,
  xnn_operator_t* fully_connected_op_out);

enum xnn_status xnn_reshape_fully_connected_nc_f16(
  xnn_operator_t fully_connected_op,
  size_t batch_size,
  pthreadpool_t threadpool);

enum xnn_status xnn_setup_fully_connected_nc_f16(
  xnn_operator_t fully_connected_op,
  const void* input,
  void* output);

enum xnn_status xnn_create_fully_connected_nc_f32_f16(
  size_t input_channels,
  size_t output_channels,
  size_t input_stride,
  size_t output_stride,
  const void* kernel,
  const void* bias,
  float output_min,
  float output_max,
  uint32_t flags,
  xnn_code_cache_t code_cache,
  xnn_weights_cache_t weights_cache,
  xnn_operator_t* fully_connected_op_out);

enum xnn_status xnn_create_fully_connected_nc_bf16_f32(
  size_t input_channels,
  size_t output_channels,
  size_t input_stride,
  size_t output_stride,
  const void* kernel,
  const float* bias,
  float output_min,
  float output_max,
  uint32_t flags,
  xnn_code_cache_t code_cache,
  xnn_weights_cache_t weights_cache,
  xnn_operator_t* fully_connected_op_out);

enum xnn_status xnn_create_fully_connected_nc_f32(
  size_t input_channels,
  size_t output_channels,
  size_t input_stride,
  size_t output_stride,
  const float* kernel,
  const float* bias,
  float output_min,
  float output_max,
  uint32_t flags,
  xnn_code_cache_t code_cache,
  xnn_weights_cache_t weights_cache,
  xnn_operator_t* fully_connected_op_out);

enum xnn_status xnn_reshape_fully_connected_nc_f32_f16(
  xnn_operator_t fully_connected_op,
  size_t batch_size,
  pthreadpool_t threadpool);

enum xnn_status xnn_reshape_fully_connected_nc_bf16_f32(
  xnn_operator_t fully_connected_op,
  size_t batch_size,
  pthreadpool_t threadpool);

enum xnn_status xnn_reshape_fully_connected_nc_f32(
  xnn_operator_t fully_connected_op,
  size_t batch_size,
  pthreadpool_t threadpool);

enum xnn_status xnn_setup_fully_connected_nc_f32_f16(
  xnn_operator_t fully_connected_op,
  const float* input,
  float* output);

enum xnn_status xnn_setup_fully_connected_nc_bf16_f32(
  xnn_operator_t fully_connected_op,
  const void* input,
  float* output);

enum xnn_status xnn_setup_fully_connected_nc_f32(
  xnn_operator_t fully_connected_op,
  const float* input,
  float* output);

enum xnn_status xnn_create_fully_connected_nc_f32_qc4w(
  size_t input_channels,
  size_t output_channels,
  size_t input_stride,
  size_t output_stride,
  uint8_t kernel_zero_point,
  const float* kernel_scale,
  const uint8_t* kernel,
  const float* bias,
  float output_min,
  float output_max,
  uint32_t flags,
  xnn_code_cache_t code_cache,
  xnn_weights_cache_t weights_cache,
  xnn_operator_t* fully_connected_op_out);

enum xnn_status xnn_reshape_fully_connected_nc_f32_qc4w(
  xnn_operator_t fully_connected_op,
  size_t batch_size,
  pthreadpool_t threadpool);

enum xnn_status xnn_setup_fully_connected_nc_f32_qc4w(
  xnn_operator_t fully_connected_op,
  const float* input,
  float* output);

enum xnn_status xnn_create_fully_connected_nc_f32_qc8w(
  size_t input_channels,
  size_t output_channels,
  size_t input_stride,
  size_t output_stride,
  const float* kernel_scale,
  const int8_t* kernel,
  const float* bias,
  float output_min,
  float output_max,
  uint32_t flags,
  xnn_code_cache_t code_cache,
  xnn_weights_cache_t weights_cache,
  xnn_operator_t* fully_connected_op_out);

enum xnn_status xnn_reshape_fully_connected_nc_f32_qc8w(
  xnn_operator_t fully_connected_op,
  size_t batch_size,
  pthreadpool_t threadpool);

enum xnn_status xnn_setup_fully_connected_nc_f32_qc8w(
  xnn_operator_t fully_connected_op,
  const float* input,
  float* output);

enum xnn_status xnn_create_fully_connected_nc_qd8_f16_qc4w(
  size_t input_channels,
  size_t output_channels,
  size_t input_stride,
  size_t output_stride,
  uint8_t kernel_zero_point,
  const float* kernel_scale,
  const void* kernel,
  const float* bias,
  float output_min,
  float output_max,
  uint32_t flags,
  xnn_code_cache_t code_cache,
  xnn_weights_cache_t weights_cache,
  xnn_operator_t* fully_connected_op_out);

enum xnn_status xnn_setup_fully_connected_nc_qd8_f16_qc4w(
  xnn_operator_t fully_connected_op,
  const int8_t* input,
  void* output,
  const struct xnn_quantization_params* quantization_params);

enum xnn_status xnn_reshape_fully_connected_nc_qd8_f16_qc4w(
  xnn_operator_t fully_connected_op,
  size_t batch_size,
  pthreadpool_t threadpool);

enum xnn_status xnn_create_fully_connected_nc_qd8_f16_qb4w(
    size_t input_channels,
    size_t output_channels,
    size_t input_stride,
    size_t output_stride,
    size_t block_size,
    uint8_t kernel_zero_point,
    const uint16_t* kernel_scale,
    const void* kernel,
    const float* bias,
    float output_min,
    float output_max,
    uint32_t flags,
    xnn_code_cache_t code_cache,
    xnn_weights_cache_t weights_cache,
    xnn_operator_t* fully_connected_op_out);

enum xnn_status xnn_reshape_fully_connected_nc_qd8_f16_qb4w(
    xnn_operator_t fully_connected_op,
    size_t batch_size,
    pthreadpool_t threadpool);

enum xnn_status xnn_setup_fully_connected_nc_qd8_f16_qb4w(
    xnn_operator_t fully_connected_op,
    const int8_t* input,
    void* output,
    const struct xnn_quantization_params* quantization_params);

enum xnn_status xnn_create_fully_connected_nc_qd8_f32_qc4w(
  size_t input_channels,
  size_t output_channels,
  size_t input_stride,
  size_t output_stride,
  uint8_t kernel_zero_point,
  const float* kernel_scale,
  const void* kernel,
  const float* bias,
  float output_min,
  float output_max,
  uint32_t flags,
  xnn_code_cache_t code_cache,
  xnn_weights_cache_t weights_cache,
  xnn_operator_t* fully_connected_op_out);

enum xnn_status xnn_setup_fully_connected_nc_qd8_f32_qc4w(
  xnn_operator_t fully_connected_op,
  const int8_t* input,
  float* output,
  const struct xnn_quantization_params* quantization_params);

enum xnn_status xnn_reshape_fully_connected_nc_qd8_f32_qc4w(
  xnn_operator_t fully_connected_op,
  size_t batch_size,
  pthreadpool_t threadpool);

enum xnn_status xnn_create_fully_connected_nc_qd8_f32_qb4w(
  size_t input_channels,
  size_t output_channels,
  size_t input_stride,
  size_t output_stride,
  size_t block_size,
  uint8_t kernel_zero_point,
  const uint16_t* kernel_scale,
  const void* kernel,
  const float* bias,
  float output_min,
  float output_max,
  uint32_t flags,
  xnn_code_cache_t code_cache,
  xnn_weights_cache_t weights_cache,
  xnn_operator_t* fully_connected_op_out);

enum xnn_status xnn_reshape_fully_connected_nc_qd8_f32_qb4w(
  xnn_operator_t fully_connected_op,
  size_t batch_size,
  pthreadpool_t threadpool);

enum xnn_status xnn_setup_fully_connected_nc_qd8_f32_qb4w(
  xnn_operator_t fully_connected_op,
  const int8_t* input,
  float* output,
  const struct xnn_quantization_params* quantization_params);

enum xnn_status xnn_create_fully_connected_nc_qd8_f16_qc8w(
  size_t input_channels,
  size_t output_channels,
  size_t input_stride,
  size_t output_stride,
  const float* kernel_scale,
  const int8_t* kernel,
  const float* bias,
  float output_min,
  float output_max,
  uint32_t flags,
  xnn_code_cache_t code_cache,
  xnn_weights_cache_t weights_cache,
  xnn_operator_t* fully_connected_op_out);

enum xnn_status xnn_setup_fully_connected_nc_qd8_f16_qc8w(
  xnn_operator_t fully_connected_op,
  const int8_t* input,
  void* output,
  const struct xnn_quantization_params* quantization_params);

enum xnn_status xnn_reshape_fully_connected_nc_qd8_f16_qc8w(
  xnn_operator_t fully_connected_op,
  size_t batch_size,
  pthreadpool_t threadpool);

enum xnn_status xnn_create_fully_connected_nc_qd8_f32_qc8w(
  size_t input_channels,
  size_t output_channels,
  size_t input_stride,
  size_t output_stride,
  const float* kernel_scale,
  const int8_t* kernel,
  const float* bias,
  float output_min,
  float output_max,
  uint32_t flags,
  xnn_code_cache_t code_cache,
  xnn_weights_cache_t weights_cache,
  xnn_operator_t* fully_connected_op_out);

enum xnn_status xnn_setup_fully_connected_nc_qd8_f32_qc8w(
  xnn_operator_t fully_connected_op,
  const int8_t* input,
  float* output,
  const struct xnn_quantization_params* quantization_params);

enum xnn_status xnn_reshape_fully_connected_nc_qd8_f32_qc8w(
  xnn_operator_t fully_connected_op,
  size_t batch_size,
  pthreadpool_t threadpool);

enum xnn_status xnn_create_fully_connected_nc_qs8(
  size_t input_channels,
  size_t output_channels,
  size_t input_stride,
  size_t output_stride,
  int8_t input_zero_point,
  float input_scale,
  float kernel_scale,
  const int8_t* kernel,
  const int32_t* bias,
  int8_t output_zero_point,
  float output_scale,
  int8_t output_min,
  int8_t output_max,
  uint32_t flags,
  xnn_code_cache_t code_cache,
  xnn_weights_cache_t weights_cache,
  xnn_operator_t* fully_connected_op_out);

enum xnn_status xnn_reshape_fully_connected_nc_qs8(
  xnn_operator_t fully_connected_op,
  size_t batch_size,
  pthreadpool_t threadpool);

enum xnn_status xnn_setup_fully_connected_nc_qs8(
  xnn_operator_t fully_connected_op,
  const int8_t* input,
  int8_t* output);

enum xnn_status xnn_create_fully_connected_nc_qs8_qc4w(
  size_t input_channels,
  size_t output_channels,
  size_t input_stride,
  size_t output_stride,
  int8_t input_zero_point,
  float input_scale,
  uint8_t kernel_zero_point,
  const float* kernel_scale,
  const void* kernel,
  const int32_t* bias,
  int8_t output_zero_point,
  float output_scale,
  int8_t output_min,
  int8_t output_max,
  uint32_t flags,
  xnn_code_cache_t code_cache,
  xnn_weights_cache_t weights_cache,
  xnn_operator_t* fully_connected_op_out);

enum xnn_status xnn_reshape_fully_connected_nc_qs8_qc4w(
  xnn_operator_t fully_connected_op,
  size_t batch_size,
  pthreadpool_t threadpool);

enum xnn_status xnn_setup_fully_connected_nc_qs8_qc4w(
  xnn_operator_t fully_connected_op,
  const int8_t* input,
  int8_t* output);

enum xnn_status xnn_create_fully_connected_nc_qs8_qc8w(
  size_t input_channels,
  size_t output_channels,
  size_t input_stride,
  size_t output_stride,
  int8_t input_zero_point,
  float input_scale,
  const float* kernel_scale,
  const int8_t* kernel,
  const int32_t* bias,
  int8_t output_zero_point,
  float output_scale,
  int8_t output_min,
  int8_t output_max,
  uint32_t flags,
  xnn_code_cache_t code_cache,
  xnn_weights_cache_t weights_cache,
  xnn_operator_t* fully_connected_op_out);

enum xnn_status xnn_reshape_fully_connected_nc_qs8_qc8w(
  xnn_operator_t fully_connected_op,
  size_t batch_size,
  pthreadpool_t threadpool);

enum xnn_status xnn_setup_fully_connected_nc_qs8_qc8w(
  xnn_operator_t fully_connected_op,
  const int8_t* input,
  int8_t* output);

enum xnn_status xnn_create_fully_connected_nc_qu8(
  size_t input_channels,
  size_t output_channels,
  size_t input_stride,
  size_t output_stride,
  uint8_t input_zero_point,
  float input_scale,
  uint8_t kernel_zero_point,
  float kernel_scale,
  const uint8_t* kernel,
  const int32_t* bias,
  uint8_t output_zero_point,
  float output_scale,
  uint8_t output_min,
  uint8_t output_max,
  uint32_t flags,
  xnn_code_cache_t code_cache,
  xnn_weights_cache_t weights_cache,
  xnn_operator_t* fully_connected_op_out);

enum xnn_status xnn_reshape_fully_connected_nc_qu8(
  xnn_operator_t fully_connected_op,
  size_t batch_size,
  pthreadpool_t threadpool);

enum xnn_status xnn_setup_fully_connected_nc_qu8(
  xnn_operator_t fully_connected_op,
  const uint8_t* input,
  uint8_t* output);


enum xnn_status xnn_create_max_pooling2d_nhwc_f16(
  uint32_t input_padding_top,
  uint32_t input_padding_right,
  uint32_t input_padding_bottom,
  uint32_t input_padding_left,
  uint32_t pooling_height,
  uint32_t pooling_width,
  uint32_t stride_height,
  uint32_t stride_width,
  uint32_t dilation_height,
  uint32_t dilation_width,
  float output_min,
  float output_max,
  uint32_t flags,
  xnn_operator_t* max_pooling_op_out);

enum xnn_status xnn_reshape_max_pooling2d_nhwc_f16(
  xnn_operator_t max_pooling_op,
  size_t batch_size,
  size_t input_height,
  size_t input_width,
  size_t channels,
  size_t input_pixel_stride,
  size_t output_pixel_stride,
  size_t* output_height_out,
  size_t* output_width_out,
  pthreadpool_t threadpool);

enum xnn_status xnn_setup_max_pooling2d_nhwc_f16(
  xnn_operator_t max_pooling_op,
  const void* input,
  void* output);

enum xnn_status xnn_create_max_pooling2d_nhwc_f32(
  uint32_t input_padding_top,
  uint32_t input_padding_right,
  uint32_t input_padding_bottom,
  uint32_t input_padding_left,
  uint32_t pooling_height,
  uint32_t pooling_width,
  uint32_t stride_height,
  uint32_t stride_width,
  uint32_t dilation_height,
  uint32_t dilation_width,
  float output_min,
  float output_max,
  uint32_t flags,
  xnn_operator_t* max_pooling_op_out);

enum xnn_status xnn_reshape_max_pooling2d_nhwc_f32(
  xnn_operator_t max_pooling_op,
  size_t batch_size,
  size_t input_height,
  size_t input_width,
  size_t channels,
  size_t input_pixel_stride,
  size_t output_pixel_stride,
  size_t* output_height_out,
  size_t* output_width_out,
  pthreadpool_t threadpool);

enum xnn_status xnn_setup_max_pooling2d_nhwc_f32(
  xnn_operator_t max_pooling_op,
  const float* input,
  float* output);

enum xnn_status xnn_create_max_pooling2d_nhwc_s8(
  uint32_t input_padding_top,
  uint32_t input_padding_right,
  uint32_t input_padding_bottom,
  uint32_t input_padding_left,
  uint32_t pooling_height,
  uint32_t pooling_width,
  uint32_t stride_height,
  uint32_t stride_width,
  uint32_t dilation_height,
  uint32_t dilation_width,
  int8_t output_min,
  int8_t output_max,
  uint32_t flags,
  xnn_operator_t* max_pooling_op_out);

enum xnn_status xnn_reshape_max_pooling2d_nhwc_s8(
  xnn_operator_t max_pooling_op,
  size_t batch_size,
  size_t input_height,
  size_t input_width,
  size_t channels,
  size_t input_pixel_stride,
  size_t output_pixel_stride,
  size_t* output_height_out,
  size_t* output_width_out,
  pthreadpool_t threadpool);

enum xnn_status xnn_setup_max_pooling2d_nhwc_s8(
  xnn_operator_t max_pooling_op,
  const int8_t* input,
  int8_t* output);

enum xnn_status xnn_create_max_pooling2d_nhwc_u8(
  uint32_t input_padding_top,
  uint32_t input_padding_right,
  uint32_t input_padding_bottom,
  uint32_t input_padding_left,
  uint32_t pooling_height,
  uint32_t pooling_width,
  uint32_t stride_height,
  uint32_t stride_width,
  uint32_t dilation_height,
  uint32_t dilation_width,
  uint8_t output_min,
  uint8_t output_max,
  uint32_t flags,
  xnn_operator_t* max_pooling_op_out);

enum xnn_status xnn_reshape_max_pooling2d_nhwc_u8(
  xnn_operator_t max_pooling_op,
  size_t batch_size,
  size_t input_height,
  size_t input_width,
  size_t channels,
  size_t input_pixel_stride,
  size_t output_pixel_stride,
  size_t* output_height_out,
  size_t* output_width_out,
  pthreadpool_t threadpool);

enum xnn_status xnn_setup_max_pooling2d_nhwc_u8(
  xnn_operator_t max_pooling_op,
  const uint8_t* input,
  uint8_t* output);

enum xnn_status xnn_create_reduce_nd(
  enum xnn_reduce_operator reduce_operator_type,
  enum xnn_datatype datatype,
  const struct xnn_quantization_params* input_quantization,
  const struct xnn_quantization_params* output_quantization,
  uint32_t flags,
  xnn_operator_t* reduce_op_out);

enum xnn_status xnn_reshape_reduce_nd(  //
    xnn_operator_t reduce_op,           //
    size_t num_reduction_axes,          //
    const int64_t* reduction_axes,      //
    size_t num_input_dims,              //
    const size_t* input_shape,          //
    size_t* workspace_size,             //
    size_t* workspace_alignment,        //
    pthreadpool_t threadpool);

enum xnn_status xnn_setup_reduce_nd(
    xnn_operator_t reduce_op,
    void* workspace,
    const void* input,
    void* output);

enum xnn_status xnn_create_resize_bilinear2d_nchw(
  enum xnn_datatype datatype,
  size_t output_height,
  size_t output_width,
  uint32_t flags,
  xnn_operator_t* resize_op_out);

enum xnn_status xnn_reshape_resize_bilinear2d_nchw(
  xnn_operator_t resize_op,
  size_t batch_size,
  size_t input_height,
  size_t input_width,
  size_t channels,
  size_t input_pixel_stride,
  size_t output_pixel_stride,
  pthreadpool_t threadpool);

enum xnn_status xnn_setup_resize_bilinear2d_nchw(
  xnn_operator_t resize_op,
  const void* input,
  void* output);

enum xnn_status xnn_create_resize_bilinear2d_nhwc(
  enum xnn_datatype datatype,
  size_t output_height,
  size_t output_width,
  uint32_t flags,
  xnn_operator_t* resize_op_out);

enum xnn_status xnn_reshape_resize_bilinear2d_nhwc(
  xnn_operator_t resize_op,
  size_t batch_size,
  size_t input_height,
  size_t input_width,
  size_t channels,
  size_t input_pixel_stride,
  size_t output_pixel_stride,
  size_t* workspace_size,
  size_t* workspace_alignment,
  pthreadpool_t threadpool);

enum xnn_status xnn_setup_resize_bilinear2d_nhwc(
  xnn_operator_t resize_op,
  void* workspace,
  const void* input,
  void* output);

enum xnn_status xnn_create_rope_nthc_f16(
  uint32_t flags,
  xnn_operator_t* rope_op_out);

enum xnn_status xnn_reshape_rope_nthc_f16(
  xnn_operator_t rope_op,
  size_t batch_size,
  size_t tokens,
  size_t heads,
  size_t channels,
  pthreadpool_t threadpool);

enum xnn_status xnn_setup_rope_nthc_f16(
  xnn_operator_t rope_op,
  const void* input,
  const void* weights,
  void* output);

enum xnn_status xnn_create_rope_nthc_f32(
  uint32_t flags,
  xnn_operator_t* rope_op_out);

enum xnn_status xnn_reshape_rope_nthc_f32(
  xnn_operator_t rope_op,
  size_t batch_size,
  size_t tokens,
  size_t heads,
  size_t channels,
  pthreadpool_t threadpool);

enum xnn_status xnn_setup_rope_nthc_f32(
  xnn_operator_t rope_op,
  const float* input,
  const float* weights,
  float* output);

enum xnn_status xnn_create_slice_nd_x16(
  uint32_t flags,
  xnn_operator_t* slice_op_out);

enum xnn_status xnn_reshape_slice_nd_x16(
  xnn_operator_t slice_op,
  size_t num_dims,
  const size_t* input_shape,
  const size_t* offsets,
  const size_t* sizes,
  pthreadpool_t threadpool);

enum xnn_status xnn_setup_slice_nd_x16(
  xnn_operator_t slice_op,
  const void* input,
  void* output);

enum xnn_status xnn_create_slice_nd_x32(
  uint32_t flags,
  xnn_operator_t* slice_op_out);

enum xnn_status xnn_reshape_slice_nd_x32(
  xnn_operator_t slice_op,
  size_t num_dims,
  const size_t* input_shape,
  const size_t* offsets,
  const size_t* sizes,
  pthreadpool_t threadpool);

enum xnn_status xnn_setup_slice_nd_x32(
  xnn_operator_t slice_op,
  const void* input,
  void* output);

enum xnn_status xnn_run_slice_nd_x32(
  size_t num_dims,
  const size_t* input_shape,
  const size_t* offsets,
  const size_t* sizes,
  const void* input,
  void* output,
  uint32_t flags,
  pthreadpool_t threadpool);

enum xnn_status xnn_create_softmax_nc_f16(
  uint32_t flags,
  xnn_operator_t* softmax_op_out);

enum xnn_status xnn_reshape_softmax_nc_f16(
  xnn_operator_t softmax_op,
  size_t channels,
  size_t input_stride,
  size_t output_stride,
  size_t batch_size,
  pthreadpool_t threadpool);

enum xnn_status xnn_setup_softmax_nc_f16(
  xnn_operator_t softmax_op,
  const void* input,
  void* output);

enum xnn_status xnn_create_softmax_nc_f32(
  uint32_t flags,
  xnn_operator_t* softmax_op_out);

enum xnn_status xnn_reshape_softmax_nc_f32(
  xnn_operator_t softmax_op,
  size_t channels,
  size_t input_stride,
  size_t output_stride,
  size_t batch_size,
  pthreadpool_t threadpool);

enum xnn_status xnn_setup_softmax_nc_f32(
  xnn_operator_t softmax_op,
  const float* input,
  float* output);

enum xnn_status xnn_create_softmax_nc_qu8(
  float input_scale,
  uint8_t output_zero_point,
  float output_scale,
  uint32_t flags,
  xnn_operator_t* softmax_op_out);

enum xnn_status xnn_reshape_softmax_nc_qu8(
  xnn_operator_t softmax_op,
  size_t channels,
  size_t input_stride,
  size_t output_stride,
  size_t batch_size,
  pthreadpool_t threadpool);

enum xnn_status xnn_setup_softmax_nc_qu8(
  xnn_operator_t softmax_op,
  const uint8_t* input,
  uint8_t* output);

enum xnn_status xnn_create_space_to_depth_nhwc_x16(
  uint32_t block_size,
  uint32_t flags,
  xnn_operator_t* space_to_depth_op_out);

enum xnn_status xnn_reshape_space_to_depth_nhwc_x16(
  xnn_operator_t space_to_depth_op,
  size_t batch_size,
  size_t input_height,
  size_t input_width,
  size_t input_channels,
  size_t* output_height_out,
  size_t* output_width_out,
  size_t* output_channels_out,
  pthreadpool_t threadpool);

enum xnn_status xnn_setup_space_to_depth_nhwc_x16(
  xnn_operator_t space_to_depth_op,
  const void* input,
  void* output);

enum xnn_status xnn_create_space_to_depth_nhwc_x32(
  uint32_t block_size,
  uint32_t flags,
  xnn_operator_t* space_to_depth_op_out);

enum xnn_status xnn_reshape_space_to_depth_nhwc_x32(
  xnn_operator_t space_to_depth_op,
  size_t batch_size,
  size_t input_height,
  size_t input_width,
  size_t input_channels,
  size_t* output_height_out,
  size_t* output_width_out,
  size_t* output_channels_out,
  pthreadpool_t threadpool);

enum xnn_status xnn_setup_space_to_depth_nhwc_x32(
  xnn_operator_t space_to_depth_op,
  const void* input,
  void* output);

enum xnn_status xnn_create_transpose_nd_x8(
  uint32_t flags,
  xnn_operator_t* transpose_op_out);

enum xnn_status xnn_reshape_transpose_nd_x8(
  xnn_operator_t transpose_op,
  size_t num_dims,
  const size_t* input_shape,
  const size_t* output_perm,
  pthreadpool_t threadpool);

enum xnn_status xnn_setup_transpose_nd_x8(
  xnn_operator_t transpose_op,
  const void* input,
  void* output);

enum xnn_status xnn_run_transpose_nd_x8(
  const void* input,
  void* output,
  size_t num_dims,
  const size_t* input_shape,
  const size_t* output_perm,
  uint32_t flags,
  pthreadpool_t threadpool);

enum xnn_status xnn_create_transpose_nd_x16(
  uint32_t flags,
  xnn_operator_t* transpose_op_out);

enum xnn_status xnn_reshape_transpose_nd_x16(
  xnn_operator_t transpose_op,
  size_t num_dims,
  const size_t* input_shape,
  const size_t* output_perm,
  pthreadpool_t threadpool);

enum xnn_status xnn_setup_transpose_nd_x16(
  xnn_operator_t transpose_op,
  const void* input,
  void* output);

enum xnn_status xnn_run_transpose_nd_x16(
  const void* input,
  void* output,
  size_t num_dims,
  const size_t* input_shape,
  const size_t* output_perm,
  uint32_t flags,
  pthreadpool_t threadpool);

enum xnn_status xnn_create_transpose_nd_x32(
  uint32_t flags,
  xnn_operator_t* transpose_op_out);

enum xnn_status xnn_reshape_transpose_nd_x32(
  xnn_operator_t transpose_op,
  size_t num_dims,
  const size_t* input_shape,
  const size_t* output_perm,
  pthreadpool_t threadpool);

enum xnn_status xnn_setup_transpose_nd_x32(
  xnn_operator_t transpose_op,
  const void* input,
  void* output);

enum xnn_status xnn_run_transpose_nd_x32(
  const void* input,
  void* output,
  size_t num_dims,
  const size_t* input_shape,
  const size_t* output_perm,
  uint32_t flags,
  pthreadpool_t threadpool);

enum xnn_status xnn_create_transpose_nd_x64(
  uint32_t flags,
  xnn_operator_t* transpose_op_out);

enum xnn_status xnn_reshape_transpose_nd_x64(
  xnn_operator_t transpose_op,
  size_t num_dims,
  const size_t* input_shape,
  const size_t* output_perm,
  pthreadpool_t threadpool);

enum xnn_status xnn_setup_transpose_nd_x64(
  xnn_operator_t transpose_op,
  const void* input,
  void* output);

enum xnn_status xnn_run_transpose_nd_x64(
  const void* input,
  void* output,
  size_t num_dims,
  const size_t* input_shape,
  const size_t* output_perm,
  uint32_t flags,
  pthreadpool_t threadpool);

enum xnn_status xnn_create_unpooling2d_nhwc_x32(
  uint32_t input_padding_top,
  uint32_t input_padding_right,
  uint32_t input_padding_bottom,
  uint32_t input_padding_left,
  uint32_t pooling_height,
  uint32_t pooling_width,
  uint32_t flags,
  xnn_operator_t* unpooling_op_out);

enum xnn_status xnn_reshape_unpooling2d_nhwc_x32(
  xnn_operator_t unpooling_op,
  size_t batch_size,
  size_t input_height,
  size_t input_width,
  size_t channels,
  size_t input_pixel_stride,
  size_t output_pixel_stride,
  size_t* output_height_out,
  size_t* output_width_out,
  pthreadpool_t threadpool);

enum xnn_status xnn_setup_unpooling2d_nhwc_x32(
  xnn_operator_t unpooling_op,
  const void* input,
  const uint32_t* index,
  void* output);

enum xnn_status xnn_create_slice_nd_x8(
  uint32_t flags,
  xnn_operator_t* slice_op_out);

enum xnn_status xnn_reshape_slice_nd_x8(
  xnn_operator_t slice_op,
  size_t num_dims,
  const size_t* input_shape,
  const size_t* offsets,
  const size_t* sizes,
  pthreadpool_t threadpool);

enum xnn_status xnn_setup_slice_nd_x8(
  xnn_operator_t slice_op,
  const void* input,
  void* output);

enum xnn_status xnn_create_space_to_depth_nhwc_x8(
  uint32_t block_size,
  uint32_t flags,
  xnn_operator_t* space_to_depth_op_out);

enum xnn_status xnn_reshape_space_to_depth_nhwc_x8(
  xnn_operator_t space_to_depth_op,
  size_t batch_size,
  size_t input_height,
  size_t input_width,
  size_t input_channels,
  size_t* output_height_out,
  size_t* output_width_out,
  size_t* output_channels_out,
  pthreadpool_t threadpool);

enum xnn_status xnn_setup_space_to_depth_nhwc_x8(
  xnn_operator_t space_to_depth_op,
  const void* input,
  void* output);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // THIRD_PARTY_XNNPACK_SRC_XNNPACK_OPERATOR_H_
