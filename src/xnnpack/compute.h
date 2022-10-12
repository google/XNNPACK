// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once


#include <stddef.h>
#include <stdint.h>

#include <xnnpack.h>
#include <xnnpack/common.h>
#include <xnnpack/math.h>
#include <xnnpack/params.h>


enum xnn_parallelization_type {
  xnn_parallelization_type_invalid = 0,
  xnn_parallelization_type_1d,
  xnn_parallelization_type_1d_tile_1d,
  xnn_parallelization_type_2d,
  xnn_parallelization_type_2d_tile_1d,
  xnn_parallelization_type_2d_tile_2d,
  xnn_parallelization_type_3d,
  xnn_parallelization_type_3d_tile_2d,
  xnn_parallelization_type_4d,
  xnn_parallelization_type_4d_tile_2d,
  xnn_parallelization_type_5d,
  xnn_parallelization_type_5d_tile_2d,
  xnn_parallelization_type_6d_tile_2d,
#if XNN_MAX_UARCH_TYPES > 1
  xnn_parallelization_type_2d_tile_2d_with_uarch,
  xnn_parallelization_type_3d_tile_2d_with_uarch,
  xnn_parallelization_type_4d_tile_2d_with_uarch,
#endif  // XNN_MAX_UARCH_TYPES > 1
};

struct compute_parameters {
  enum xnn_parallelization_type type;
  union {
    pthreadpool_task_1d_t task_1d;
    pthreadpool_task_1d_tile_1d_t task_1d_tile_1d;
    pthreadpool_task_2d_t task_2d;
    pthreadpool_task_2d_tile_1d_t task_2d_tile_1d;
    pthreadpool_task_2d_tile_2d_t task_2d_tile_2d;
    pthreadpool_task_3d_t task_3d;
    pthreadpool_task_3d_tile_2d_t task_3d_tile_2d;
    pthreadpool_task_4d_t task_4d;
    pthreadpool_task_4d_tile_2d_t task_4d_tile_2d;
    pthreadpool_task_5d_t task_5d;
    pthreadpool_task_5d_tile_2d_t task_5d_tile_2d;
    pthreadpool_task_6d_tile_2d_t task_6d_tile_2d;
#if XNN_MAX_UARCH_TYPES > 1
    pthreadpool_task_2d_tile_2d_with_id_t task_2d_tile_2d_with_id;
    pthreadpool_task_3d_tile_2d_with_id_t task_3d_tile_2d_with_id;
    pthreadpool_task_4d_tile_2d_with_id_t task_4d_tile_2d_with_id;
#endif  // XNN_MAX_UARCH_TYPES > 1
  };
  size_t range[6];
  size_t tile[2];
};

struct transpose_context {
  const void* x;
  void* y;
  union {
    xnn_transposec_ukernel_function const_size_ukernel;
    xnn_transposev_ukernel_function variable_size_ukernel;
  };
  union {
    size_t element_size;
    size_t log2_element_size;
  };
  size_t input_stride[XNN_MAX_TENSOR_DIMS];
  size_t output_stride[XNN_MAX_TENSOR_DIMS];
};

XNN_PRIVATE void xnn_compute_transposec_2d(
    const struct transpose_context* context,
    size_t i,
    size_t j,
    size_t tile_i,
    size_t tile_j);

XNN_PRIVATE void xnn_compute_transposec_3d(
    const struct transpose_context* context,
    size_t i,
    size_t j,
    size_t k,
    size_t tile_j,
    size_t tile_k);

XNN_PRIVATE void xnn_compute_transposec_4d(
    const struct transpose_context* context,
    size_t i,
    size_t j,
    size_t k,
    size_t l,
    size_t tile_k,
    size_t tile_l);

XNN_PRIVATE void xnn_compute_transposec_5d(
    const struct transpose_context* context,
    size_t i,
    size_t j,
    size_t k,
    size_t l,
    size_t m,
    size_t tile_l,
    size_t tile_m);

XNN_PRIVATE void xnn_compute_transposec_6d(
    const struct transpose_context* context,
    size_t i,
    size_t j,
    size_t k,
    size_t l,
    size_t m,
    size_t n,
    size_t tile_m,
    size_t tile_n);

XNN_PRIVATE void xnn_compute_transposev_2d(
    const struct transpose_context* context,
    size_t i,
    size_t j,
    size_t tile_i,
    size_t tile_j);

XNN_PRIVATE void xnn_compute_transposev_3d(
    const struct transpose_context* context,
    size_t i,
    size_t j,
    size_t k,
    size_t tile_j,
    size_t tile_k);

XNN_PRIVATE void xnn_compute_transposev_4d(
    const struct transpose_context* context,
    size_t i,
    size_t j,
    size_t k,
    size_t l,
    size_t tile_k,
    size_t tile_l);

XNN_PRIVATE void xnn_compute_transposev_5d(
    const struct transpose_context* context,
    size_t i,
    size_t j,
    size_t k,
    size_t l,
    size_t m,
    size_t tile_l,
    size_t tile_m);

XNN_PRIVATE void xnn_compute_transposev_6d(
    const struct transpose_context* context,
    size_t i,
    size_t j,
    size_t k,
    size_t l,
    size_t m,
    size_t n,
    size_t tile_m,
    size_t tile_n);

struct gemm_context {
  size_t k_scaled;
  const void* a;
  size_t a_stride;
  const void* packed_w;
  size_t w_stride;
  size_t wg_stride;
  void* c;
  size_t cm_stride;
  size_t cn_stride;
  size_t cg_stride;
  uint32_t log2_csize;
  struct xnn_hmp_gemm_ukernel ukernel;
  void* fused_params;
  union {
    union xnn_qs8_conv_minmax_params qs8;
    union xnn_qu8_conv_minmax_params qu8;
    union xnn_f16_scaleminmax_params f16;
    union xnn_f32_minmax_params f32;
  } params;
};

#ifndef __cplusplus
  XNN_PRIVATE void xnn_compute_grouped_gemm(
      const struct gemm_context context[restrict XNN_MIN_ELEMENTS(1)],
      size_t group_index,
      size_t mr_block_start,
      size_t nr_block_start,
      size_t mr_block_size,
      size_t nr_block_size);

  XNN_PRIVATE void xnn_compute_gemm(
      const struct gemm_context context[restrict XNN_MIN_ELEMENTS(1)],
      size_t mr_block_start,
      size_t nr_block_start,
      size_t mr_block_size,
      size_t nr_block_size);

  #if XNN_MAX_UARCH_TYPES > 1
    XNN_PRIVATE void xnn_compute_hmp_grouped_gemm(
        const struct gemm_context context[restrict XNN_MIN_ELEMENTS(1)],
        uint32_t uarch_index,
        size_t group_index,
        size_t mr_block_start,
        size_t nr_block_start,
        size_t mr_block_size,
        size_t nr_block_size);

    XNN_PRIVATE void xnn_compute_hmp_gemm(
        const struct gemm_context context[restrict XNN_MIN_ELEMENTS(1)],
        uint32_t uarch_index,
        size_t mr_block_start,
        size_t nr_block_start,
        size_t mr_block_size,
        size_t nr_block_size);
  #endif  // XNN_MAX_UARCH_TYPES > 1
#endif

// Context for Sparse Matrix-Dense Matrix Multiplication.
// C [MxN] := A [MxK] * B [KxN] + bias [N]
// A and C are dense matrices with row-major storage, B is a sparse matrix.
struct spmm_context {
  // N dimension of the B and C matrices.
  // Corresponds to number of output channels in 1x1 convolution.
  size_t n;
  // M dimension of the A and C matrices, pre-scaled by sizeof(element size).
  // Corresponds to the stride, in bytes, between adjacent rows of C matrix.
  size_t scaled_m;
  // Input matrix A.
  const void* input;
  // Packed bias elements and non-zero filter elements.
  const void* nonzero_weights;
  // Input pointer increments, in bytes, after each processed non-zero weight.
  const int32_t* input_increments;
  // Number of non-zero filter elements per each N (output channel) dimension.
  const uint32_t* output_channel_nonzeros;
  // Output matrix C.
  void* output;
  // Stride, in bytes, between matrices A corresponding to different images in batched 1x1 Convolution
  size_t batched_input_stride;
  // Stride, in bytes, between matrices C corresponding to different images in batched 1x1 Convolution
  size_t batched_output_stride;
  // Micro-kernel function pointer.
  xnn_spmm_ukernel_function ukernel;
  // Output activation parameters.
  union {
    union xnn_f32_minmax_params f32;
  } params;
};

#ifndef __cplusplus
  XNN_PRIVATE void xnn_compute_spmm(
    const struct spmm_context context[restrict XNN_MIN_ELEMENTS(1)],
    size_t batch_index,
    size_t mr_block_start,
    size_t mr_block_size);
#endif

struct igemm_context {
  size_t ks;
  size_t ks_scaled;
  size_t kc;
  size_t w_stride;
  const void** indirect_a;
  size_t a_offset;
  void* zero;
  const void* packed_w;
  void* c;
  size_t cm_stride;
  size_t cn_stride;
  size_t ga_stride;
  size_t gw_stride;
  size_t gc_stride;
  size_t ba_stride;
  size_t bc_stride;
  uint32_t log2_csize;
  struct xnn_hmp_igemm_ukernel ukernel;
  union {
    union xnn_qs8_conv_minmax_params qs8;
    union xnn_qu8_conv_minmax_params qu8;
    union xnn_f16_scaleminmax_params f16;
    union xnn_f32_minmax_params f32;
  } params;
};

#ifndef __cplusplus
  XNN_PRIVATE void xnn_compute_grouped_igemm(
      const struct igemm_context context[restrict XNN_MIN_ELEMENTS(1)],
      size_t group_index,
      size_t mr_block_start,
      size_t nr_block_start,
      size_t mr_block_size,
      size_t nr_block_size);

  XNN_PRIVATE void xnn_compute_grouped_batch_igemm(
      const struct igemm_context context[restrict XNN_MIN_ELEMENTS(1)],
      size_t batch_index,
      size_t group_index,
      size_t mr_block_start,
      size_t nr_block_start,
      size_t mr_block_size,
      size_t nr_block_size);

  XNN_PRIVATE void xnn_compute_igemm(
      const struct igemm_context context[restrict XNN_MIN_ELEMENTS(1)],
      size_t mr_block_start,
      size_t nr_block_start,
      size_t mr_block_size,
      size_t nr_block_size);

  XNN_PRIVATE void xnn_compute_batch_igemm(
      const struct igemm_context context[restrict XNN_MIN_ELEMENTS(1)],
      size_t batch_index,
      size_t mr_block_start,
      size_t nr_block_start,
      size_t mr_block_size,
      size_t nr_block_size);

  #if XNN_MAX_UARCH_TYPES > 1
    XNN_PRIVATE void xnn_compute_hmp_grouped_igemm(
        const struct igemm_context context[restrict XNN_MIN_ELEMENTS(1)],
        uint32_t uarch_index,
        size_t group_index,
        size_t mr_block_start,
        size_t nr_block_start,
        size_t mr_block_size,
        size_t nr_block_size);

    XNN_PRIVATE void xnn_compute_hmp_grouped_batch_igemm(
        const struct igemm_context context[restrict XNN_MIN_ELEMENTS(1)],
        uint32_t uarch_index,
        size_t batch_index,
        size_t group_index,
        size_t mr_block_start,
        size_t nr_block_start,
        size_t mr_block_size,
        size_t nr_block_size);

    XNN_PRIVATE void xnn_compute_hmp_igemm(
        const struct igemm_context context[restrict XNN_MIN_ELEMENTS(1)],
        uint32_t uarch_index,
        size_t mr_block_start,
        size_t nr_block_start,
        size_t mr_block_size,
        size_t nr_block_size);

    XNN_PRIVATE void xnn_compute_batch_hmp_igemm(
        const struct igemm_context context[restrict XNN_MIN_ELEMENTS(1)],
        uint32_t uarch_index,
        size_t batch_index,
        size_t mr_block_start,
        size_t nr_block_start,
        size_t mr_block_size,
        size_t nr_block_size);
  #endif  // XNN_MAX_UARCH_TYPES > 1
#endif

struct subgemm_context {
  const struct subconvolution_params* subconvolution_params;
  size_t kc;
  const void* a;
  size_t ax_stride;
  size_t ay_stride;
  size_t cx_stride;
  size_t cy_stride;
  size_t cn_stride;
  size_t ga_stride;
  size_t gw_stride;
  size_t gc_stride;
  size_t ba_stride;
  size_t bc_stride;
  uint32_t log2_csize;
  struct xnn_hmp_gemm_ukernel ukernel;
  union {
    union xnn_qs8_conv_minmax_params qs8;
    union xnn_qu8_conv_minmax_params qu8;
    union xnn_f16_scaleminmax_params f16;
    union xnn_f32_minmax_params f32;
  } params;
};

#ifndef __cplusplus
  XNN_PRIVATE void xnn_compute_grouped_subgemm2d(
      const struct subgemm_context context[restrict XNN_MIN_ELEMENTS(1)],
      size_t batch_index,
      size_t group_index,
      size_t subkernel_index,
      size_t slice_y,
      size_t slice_x_start,
      size_t nr_block_start,
      size_t slice_x_max,
      size_t nr_block_size);

  XNN_PRIVATE void xnn_compute_subgemm2d(
      const struct subgemm_context context[restrict XNN_MIN_ELEMENTS(1)],
      size_t batch_index,
      size_t subkernel_index,
      size_t slice_y,
      size_t slice_x_start,
      size_t nr_block_start,
      size_t slice_x_max,
      size_t nr_block_size);
#endif

struct subconv_context {
  const struct subconvolution_params* subconvolution_params;
  size_t kc;
  size_t a_offset;
  void* zero;
  size_t cx_stride;
  size_t cy_stride;
  size_t cn_stride;
  size_t ga_stride;
  size_t gw_stride;
  size_t gc_stride;
  size_t ba_stride;
  size_t bc_stride;
  uint32_t log2_csize;
  struct xnn_hmp_igemm_ukernel ukernel;
  union {
    union xnn_qs8_conv_minmax_params qs8;
    union xnn_qu8_conv_minmax_params qu8;
    union xnn_f16_scaleminmax_params f16;
    union xnn_f32_minmax_params f32;
  } params;
};

#ifndef __cplusplus
  XNN_PRIVATE void xnn_compute_grouped_subconv2d(
      const struct subconv_context context[restrict XNN_MIN_ELEMENTS(1)],
      size_t batch_index,
      size_t group_index,
      size_t subkernel_index,
      size_t slice_y,
      size_t slice_x_start,
      size_t nr_block_start,
      size_t slice_x_max,
      size_t nr_block_size);

  XNN_PRIVATE void xnn_compute_subconv2d(
      const struct subconv_context context[restrict XNN_MIN_ELEMENTS(1)],
      size_t batch_index,
      size_t subkernel_index,
      size_t slice_y,
      size_t slice_x_start,
      size_t nr_block_start,
      size_t slice_x_max,
      size_t nr_block_size);
#endif

struct conv2d_context {
  size_t input_height;
  size_t input_width;
  const void* input;
  size_t input_batch_stride;
  const void* zero;
  const void* packed_weights;
  void* output;
  size_t output_batch_stride;
  size_t input_padding_top;
  size_t output_channels;
  size_t output_height_stride;
  size_t output_channel_stride;
  union {
    xnn_conv_hwc2chw_ukernel_function hwc2chw_ukernel;
  };
  union {
    union xnn_f32_minmax_params f32;
  } params;
};

#ifndef __cplusplus
  XNN_PRIVATE void xnn_compute_conv2d_hwc2chw(
      const struct conv2d_context context[restrict XNN_MIN_ELEMENTS(1)],
      size_t batch_index,
      size_t output_y_start,
      size_t output_y_slice);
#endif

struct dwconv_context {
  const void** indirect_input;
  size_t indirect_input_width_stride;
  size_t indirect_input_height_stride;
  size_t input_offset;
  size_t input_batch_stride;
  const void* packed_weights;
  void* output;
  size_t output_batch_stride;
  size_t output_height_stride;
  size_t output_width;
  size_t groups;
  const void* zero;
  size_t output_increment;
  union {
    union xnn_qs8_conv_minmax_params qs8;
    union xnn_qu8_conv_minmax_params qu8;
    union xnn_f16_minmax_params f16;
    union xnn_f32_minmax_params f32;
  } params;
  union {
    xnn_dwconv_unipass_ukernel_function unipass_ukernel;
  };
};

#ifndef __cplusplus
  XNN_PRIVATE void xnn_compute_dwconv_unipass(
      const struct dwconv_context context[restrict XNN_MIN_ELEMENTS(1)],
      size_t batch_index,
      size_t output_y);
#endif

struct dwconv2d_context {
  size_t input_height;
  size_t input_width;
  const void* input;
  const void* zero;
  uint32_t input_padding_top;
  size_t input_channel_stride;
  size_t input_batch_stride;
  const void* packed_weights;
  size_t weights_channel_stride;
  void* output;
  size_t output_channel_stride;
  size_t output_batch_stride;
  union {
    union xnn_f32_chw_params f32;
  } params;
  union {
    xnn_dwconv2d_chw_ukernel_function chw_ukernel;
  };
};

#ifndef __cplusplus
  XNN_PRIVATE void xnn_compute_dwconv2d_chw(
      const struct dwconv2d_context context[restrict XNN_MIN_ELEMENTS(1)],
      size_t batch_index,
      size_t channel);
#endif

struct max_pooling_context {
  const void** indirect_input;
  size_t indirect_input_height_stride;
  size_t input_offset;
  size_t input_batch_stride;
  void* output;
  size_t output_batch_stride;
  size_t output_height_stride;
  size_t output_width;
  size_t pooling_size;
  size_t channels;
  size_t input_increment;
  size_t output_increment;
  union {
    union xnn_u8_minmax_params u8;
    union xnn_f32_minmax_params f32;
  } params;
  xnn_maxpool_ukernel_function ukernel;
};

#ifndef __cplusplus
  XNN_PRIVATE void xnn_compute_max_pooling(
      const struct max_pooling_context context[restrict XNN_MIN_ELEMENTS(1)],
      size_t batch_index,
      size_t output_y);
#endif

struct unpooling_context {
  const void* input;
  size_t input_height_stride;
  size_t input_width_stride;
  const uint32_t* index;
  size_t index_height_stride;
  size_t index_width_stride;
  const void** indirect_output;
  size_t indirect_output_height_stride;
  size_t indirect_output_width_stride;
  size_t pooling_size;
  size_t channels;
  uint32_t fill_value;
  xnn_unpool_ukernel_function ukernel;
};

#ifndef __cplusplus
  XNN_PRIVATE void xnn_compute_unpooling(
      const struct unpooling_context context[restrict XNN_MIN_ELEMENTS(1)],
      size_t input_y,
      size_t input_x);
#endif

struct argmax_pooling_context {
  const void** indirect_input;
  size_t indirect_input_height_stride;
  size_t input_offset;
  size_t input_batch_stride;
  void* output;
  size_t output_batch_stride;
  size_t output_height_stride;
  size_t output_width;
  uint32_t* index;
  size_t index_batch_stride;
  size_t index_height_stride;
  size_t pooling_size;
  size_t channels;
  size_t input_increment;
  size_t output_increment;
  union {
    xnn_argmaxpool_unipass_ukernel_function unipass_ukernel;
    xnn_argmaxpool_multipass_ukernel_function multipass_ukernel;
  };
};

#ifndef __cplusplus
  XNN_PRIVATE void xnn_compute_argmax_pooling_unipass(
      const struct argmax_pooling_context context[restrict XNN_MIN_ELEMENTS(1)],
      size_t batch_index,
      size_t output_y);

  XNN_PRIVATE void xnn_compute_argmax_pooling_multipass(
      const struct argmax_pooling_context context[restrict XNN_MIN_ELEMENTS(1)],
      size_t batch_index,
      size_t output_y);
#endif

struct average_pooling_context {
  const void** indirect_input;
  size_t indirect_input_height_stride;
  size_t input_offset;
  size_t input_batch_stride;
  void* output;
  size_t output_batch_stride;
  size_t output_height_stride;
  size_t output_width;
  size_t pooling_size;
  size_t channels;
  const void* zero;
  size_t input_increment;
  size_t output_increment;
  union {
    union xnn_f16_scaleminmax_params f16;
    union xnn_f32_scaleminmax_params f32;
    union xnn_qu8_avgpool_minmax_params qu8;
  } params;
  union {
    xnn_avgpool_unipass_ukernel_function unipass_ukernel;
    xnn_avgpool_multipass_ukernel_function multipass_ukernel;
  };
};

#ifndef __cplusplus
  XNN_PRIVATE void xnn_compute_average_pooling_unipass(
      const struct average_pooling_context context[restrict XNN_MIN_ELEMENTS(1)],
      size_t batch_index,
      size_t output_y);

  XNN_PRIVATE void xnn_compute_average_pooling_multipass(
      const struct average_pooling_context context[restrict XNN_MIN_ELEMENTS(1)],
      size_t batch_index,
      size_t output_y);
#endif

struct pixelwise_average_pooling_context {
  const void** indirect_input;
  size_t indirect_input_height_stride;
  size_t input_offset;
  size_t input_batch_stride;
  const void* pixelwise_buffer;
  size_t pixelwise_buffer_height_stride;
  void* output;
  size_t output_batch_stride;
  size_t output_height_stride;
  size_t output_width;
  size_t pooling_size;
  size_t channels;
  const void* zero;
  size_t input_increment;
  size_t output_increment;
  union {
    union xnn_f16_minmax_params f16;
    union xnn_f32_minmax_params f32;
    union xnn_u8_minmax_params u8;
  } params;
  union {
    xnn_pavgpool_unipass_ukernel_function unipass_ukernel;
    xnn_pavgpool_multipass_ukernel_function multipass_ukernel;
  };
};

#ifndef __cplusplus
  XNN_PRIVATE void xnn_compute_pixelwise_average_pooling_unipass(
      const struct pixelwise_average_pooling_context context[restrict XNN_MIN_ELEMENTS(1)],
      size_t batch_index,
      size_t output_y);

  XNN_PRIVATE void xnn_compute_pixelwise_average_pooling_multipass(
      const struct pixelwise_average_pooling_context context[restrict XNN_MIN_ELEMENTS(1)],
      size_t batch_index,
      size_t output_y);
#endif

struct global_average_pooling_nwc_context {
  const void* input;
  const void* zero;
  size_t input_pixel_stride;
  size_t input_batch_stride;
  size_t input_elements;
  size_t channels;
  void* output;
  size_t output_batch_stride;
  union {
    union xnn_qs8_avgpool_minmax_params qs8;
    union xnn_qu8_avgpool_minmax_params qu8;
    union xnn_f16_scaleminmax_params f16;
    union xnn_f32_scaleminmax_params f32;
  } params;
  union {
    xnn_gavgpool_unipass_ukernel_function unipass_ukernel;
    xnn_gavgpool_multipass_ukernel_function multipass_ukernel;
  };
};

#ifndef __cplusplus
  XNN_PRIVATE void xnn_compute_global_average_pooling_nwc_unipass(
      const struct global_average_pooling_nwc_context context[restrict XNN_MIN_ELEMENTS(1)],
      size_t batch_index);

  XNN_PRIVATE void xnn_compute_global_average_pooling_nwc_multipass(
      const struct global_average_pooling_nwc_context context[restrict XNN_MIN_ELEMENTS(1)],
      size_t batch_index);
#endif

struct global_average_pooling_ncw_context {
  size_t input_elements;
  const void* input;
  size_t input_channel_stride;
  size_t input_batch_stride;
  void* output;
  size_t output_channel_stride;
  size_t output_batch_stride;
  xnn_gavgpool_cw_ukernel_function ukernel;
  union {
    union xnn_f32_gavgpool_params f32;
  } params;
};

#ifndef __cplusplus
  XNN_PRIVATE void xnn_compute_global_average_pooling_ncw(
      const struct global_average_pooling_ncw_context context[restrict XNN_MIN_ELEMENTS(1)],
      size_t batch_index,
      size_t channels_start,
      size_t channels_slice);
#endif

struct resize_bilinear_context {
  // Number of channels multiplied by sizeof(input element).
  size_t scaled_channels;
  // Indirection buffer with pointers related to rows of input pixels.
  const void** indirect_input;
  // Offset, in bytes, to be added to pointers in indirection buffer.
  size_t input_offset;
  // Stride, in bytes, between images of consecutive batches in the input.
  size_t input_batch_stride;
  // Packed pairs of (x, y) linear interpolation coefficients.
  const void* packed_weights;
  // Pointer to the output tensor.
  void* output;
  // Stride, in bytes, between adjacent pixels in the output.
  size_t output_pixel_stride;
  // Stride, in bytes, between images of consecutive batches in the output.
  size_t output_batch_stride;
  // log2(sizeof(weight element)).
  uint32_t log2_wsize;
  // Pointer to BILINEAR micro-kernel function.
  xnn_ibilinear_ukernel_function ukernel;
};

struct resize_bilinear_chw_context {
  // Number of pixels per output image plane.
  size_t output_pixels;
  // Number of channels multiplied by sizeof(input element).
  size_t channels;
  // Stride, in bytes, between adjacent channels in the input.
  size_t input_channel_stride;
  // Indirection buffer with pointers related to rows of input pixels.
  const void** indirect_input;
  // Offset, in bytes, to be added to pointers in indirection buffer.
  size_t input_offset;
  // Stride, in bytes, between images of consecutive batches in the input.
  size_t input_batch_stride;
  // Packed pairs of (x, y) linear interpolation coefficients.
  const void* packed_weights;
  // Pointer to the output tensor.
  void* output;
  // Stride, in bytes, between images of consecutive batches in the output.
  size_t output_batch_stride;
  // Stride, in bytes, between consecutive channels of an output image.
  size_t output_channel_stride;
  // Pointer to BILINEAR micro-kernel function.
  xnn_ibilinear_chw_ukernel_function ukernel;
};

#ifndef __cplusplus
  XNN_PRIVATE void xnn_compute_resize_bilinear(
      const struct resize_bilinear_context context[restrict XNN_MIN_ELEMENTS(1)],
      size_t batch_index,
      size_t pixel_start,
      size_t pixel_range);
  XNN_PRIVATE void xnn_compute_resize_bilinear_chw(
    const struct resize_bilinear_chw_context context[restrict XNN_MIN_ELEMENTS(1)],
    size_t batch_index,
    size_t pixel_start,
    size_t pixel_range);
#endif

struct elementwise_binary_context {
  const void* a;
  size_t a_stride[XNN_MAX_TENSOR_DIMS - 1];
  const void* b;
  size_t b_stride[XNN_MAX_TENSOR_DIMS - 1];
  void* y;
  size_t y_stride[XNN_MAX_TENSOR_DIMS - 1];
  size_t elements;
  union {
    union xnn_qs8_add_minmax_params qs8_addsub;
    union xnn_qu8_add_minmax_params qu8_addsub;
    union xnn_qs8_mul_minmax_params qs8_mul;
    union xnn_qu8_mul_minmax_params qu8_mul;
    union xnn_f16_minmax_params f16;
    union xnn_f32_minmax_params f32;
  } params;
  xnn_vbinary_ukernel_function ukernel;
};

#ifndef __cplusplus
  XNN_PRIVATE void xnn_compute_elementwise_binary_1d(
      const struct elementwise_binary_context context[restrict XNN_MIN_ELEMENTS(1)],
      size_t i);
  XNN_PRIVATE void xnn_compute_elementwise_binary_2d(
      const struct elementwise_binary_context context[restrict XNN_MIN_ELEMENTS(1)],
      size_t i, size_t j);
  XNN_PRIVATE void xnn_compute_elementwise_binary_3d(
      const struct elementwise_binary_context context[restrict XNN_MIN_ELEMENTS(1)],
      size_t i, size_t j, size_t k);
  XNN_PRIVATE void xnn_compute_elementwise_binary_4d(
      const struct elementwise_binary_context context[restrict XNN_MIN_ELEMENTS(1)],
      size_t i, size_t j, size_t k, size_t l);
  XNN_PRIVATE void xnn_compute_elementwise_binary_5d(
      const struct elementwise_binary_context context[restrict XNN_MIN_ELEMENTS(1)],
      size_t i, size_t j, size_t k, size_t l, size_t m);
#endif

struct channel_shuffle_context {
  const void* x;
  size_t x_stride;
  void* y;
  size_t y_stride;
  size_t n;
  size_t m;
  union {
    xnn_zipc_ukernel_function fixed_ukernel;
    xnn_zipv_ukernel_function variable_ukernel;
  };
};

#ifndef __cplusplus
  XNN_PRIVATE void xnn_compute_channel_shuffle_fixed(
      const struct channel_shuffle_context context[restrict XNN_MIN_ELEMENTS(1)],
      size_t index);

  XNN_PRIVATE void xnn_compute_channel_shuffle_variable(
      const struct channel_shuffle_context context[restrict XNN_MIN_ELEMENTS(1)],
      size_t index);
#endif

struct lut_strided_context {
  size_t n;
  const void* x;
  size_t x_stride;
  const void* t;
  void* y;
  size_t y_stride;
  xnn_x8_lut_ukernel_function ukernel;
};

#ifndef __cplusplus
  XNN_PRIVATE void xnn_compute_lut_strided(
      const struct lut_strided_context context[restrict XNN_MIN_ELEMENTS(1)],
      size_t batch_index);
#endif

struct lut_contiguous_context {
  const void* x;
  size_t x_stride;
  const void* t;
  void* y;
  size_t y_stride;
  xnn_x8_lut_ukernel_function ukernel;
};

#ifndef __cplusplus
  XNN_PRIVATE void xnn_compute_lut_contiguous(
      const struct lut_contiguous_context context[restrict XNN_MIN_ELEMENTS(1)],
      size_t offset,
      size_t size);
#endif

struct univector_strided_context {
  size_t n;
  const void* x;
  size_t x_stride;
  void* y;
  size_t y_stride;
  xnn_vunary_ukernel_function ukernel;
  union {
    union xnn_f16_abs_params f16_abs;
    union xnn_f16_default_params f16_default;
    union xnn_f16_f32_cvt_params f16_f32_cvt;
    union xnn_f16_hswish_params f16_hswish;
    union xnn_f16_lrelu_params f16_lrelu;
    union xnn_f16_minmax_params f16_minmax;
    union xnn_f16_neg_params f16_neg;
    union xnn_f16_sigmoid_params f16_sigmoid;
    union xnn_f32_abs_params f32_abs;
    union xnn_f32_default_params f32_default;
    union xnn_f32_elu_params f32_elu;
    union xnn_f32_f16_cvt_params f32_f16_cvt;
    union xnn_f32_hswish_params f32_hswish;
    union xnn_f32_lrelu_params f32_lrelu;
    union xnn_f32_minmax_params f32_minmax;
    union xnn_f32_neg_params f32_neg;
    union xnn_f32_qs8_cvt_params f32_qs8_cvt;
    union xnn_f32_qu8_cvt_params f32_qu8_cvt;
    union xnn_f32_rnd_params f32_rnd;
    union xnn_f32_sigmoid_params f32_sigmoid;
    union xnn_f32_sqrt_params f32_sqrt;
    union xnn_qs8_cvt_params qs8_cvt;
    union xnn_qs8_f32_cvt_params qs8_f32_cvt;
    union xnn_qs8_lrelu_params qs8_lrelu;
    union xnn_qu8_cvt_params qu8_cvt;
    union xnn_qu8_f32_cvt_params qu8_f32_cvt;
    union xnn_qu8_lrelu_params qu8_lrelu;
    union xnn_s8_minmax_params s8_minmax;
    union xnn_u8_minmax_params u8_minmax;
  } params;
};

#ifndef __cplusplus
  XNN_PRIVATE void xnn_compute_univector_strided(
      const struct univector_strided_context context[restrict XNN_MIN_ELEMENTS(1)],
      size_t batch_index,
      size_t batch_range);
#endif

struct univector_contiguous_context {
  const void* x;
  void* y;
  uint16_t log2_xsize;
  uint16_t log2_ysize;
  xnn_vunary_ukernel_function ukernel;
  union {
    union xnn_f16_abs_params f16_abs;
    union xnn_f16_default_params f16_default;
    union xnn_f16_f32_cvt_params f16_f32_cvt;
    union xnn_f16_hswish_params f16_hswish;
    union xnn_f16_lrelu_params f16_lrelu;
    union xnn_f16_minmax_params f16_minmax;
    union xnn_f16_neg_params f16_neg;
    union xnn_f16_sigmoid_params f16_sigmoid;
    union xnn_f32_abs_params f32_abs;
    union xnn_f32_default_params f32_default;
    union xnn_f32_elu_params f32_elu;
    union xnn_f32_f16_cvt_params f32_f16_cvt;
    union xnn_f32_hswish_params f32_hswish;
    union xnn_f32_lrelu_params f32_lrelu;
    union xnn_f32_minmax_params f32_minmax;
    union xnn_f32_neg_params f32_neg;
    union xnn_f32_qs8_cvt_params f32_qs8_cvt;
    union xnn_f32_qu8_cvt_params f32_qu8_cvt;
    union xnn_f32_rnd_params f32_rnd;
    union xnn_f32_sigmoid_params f32_sigmoid;
    union xnn_f32_sqrt_params f32_sqrt;
    union xnn_qs8_cvt_params qs8_cvt;
    union xnn_qs8_f32_cvt_params qs8_f32_cvt;
    union xnn_qs8_lrelu_params qs8_lrelu;
    union xnn_qu8_cvt_params qu8_cvt;
    union xnn_qu8_f32_cvt_params qu8_f32_cvt;
    union xnn_qu8_lrelu_params qu8_lrelu;
    union xnn_s8_minmax_params s8_minmax;
    union xnn_u8_minmax_params u8_minmax;
  } params;
};

#ifndef __cplusplus
  XNN_PRIVATE void xnn_compute_univector_contiguous(
      const struct univector_contiguous_context context[restrict XNN_MIN_ELEMENTS(1)],
      size_t offset,
      size_t size);
#endif

struct prelu_context {
  size_t n;
  const void* x;
  size_t x_stride;
  const void* w;
  void* y;
  size_t y_stride;
  xnn_prelu_ukernel_function ukernel;
};

#ifndef __cplusplus
  XNN_PRIVATE void xnn_compute_prelu(
      const struct prelu_context context[restrict XNN_MIN_ELEMENTS(1)],
      size_t batch_start,
      size_t batch_range);
#endif

struct vmulcaddc_context {
  size_t n;
  const void* x;
  size_t x_stride;
  const void* w;
  void* y;
  size_t y_stride;
  xnn_vmulcaddc_ukernel_function ukernel;
  union {
    union xnn_f16_minmax_params f16;
    union xnn_f32_minmax_params f32;
  } params;
};

#ifndef __cplusplus
  XNN_PRIVATE void xnn_compute_vmulcaddc(
      const struct vmulcaddc_context context[restrict XNN_MIN_ELEMENTS(1)],
      size_t batch_start,
      size_t batch_size);
#endif

struct pad_context {
  const void* input;
  size_t input_stride[XNN_MAX_TENSOR_DIMS - 1];
  void* output;
  size_t output_stride[XNN_MAX_TENSOR_DIMS - 1];
  size_t pre_paddings[XNN_MAX_TENSOR_DIMS];
  size_t post_paddings[1];
  size_t input_size[XNN_MAX_TENSOR_DIMS];
  size_t output_size[1];
  uint32_t padding_value;
  xnn_pad_ukernel_function pad_ukernel;
  xnn_fill_ukernel_function fill_ukernel;
};

#ifndef __cplusplus
  XNN_PRIVATE void xnn_compute_pad_5d(
      const struct pad_context context[restrict XNN_MIN_ELEMENTS(1)],
      size_t i, size_t j, size_t k, size_t l, size_t m);
#endif

struct slice_context {
  const void* input;
  size_t input_stride[XNN_MAX_TENSOR_DIMS - 1];
  void* output;
  size_t output_stride[XNN_MAX_TENSOR_DIMS - 1];
  size_t offsets[XNN_MAX_TENSOR_DIMS];
  size_t contiguous_size;
  xnn_vunary_ukernel_function ukernel;
};

#ifndef __cplusplus
  XNN_PRIVATE void xnn_compute_slice_1d(
      const struct slice_context context[restrict XNN_MIN_ELEMENTS(1)],
      size_t i);
  XNN_PRIVATE void xnn_compute_slice_2d(
      const struct slice_context context[restrict XNN_MIN_ELEMENTS(1)],
      size_t i, size_t j);
  XNN_PRIVATE void xnn_compute_slice_3d(
      const struct slice_context context[restrict XNN_MIN_ELEMENTS(1)],
      size_t i, size_t j, size_t k);
  XNN_PRIVATE void xnn_compute_slice_4d(
      const struct slice_context context[restrict XNN_MIN_ELEMENTS(1)],
      size_t i, size_t j, size_t k, size_t l);
  XNN_PRIVATE void xnn_compute_slice_5d(
      const struct slice_context context[restrict XNN_MIN_ELEMENTS(1)],
      size_t i, size_t j, size_t k, size_t l, size_t m);
#endif

struct u8_softmax_context {
  size_t n;
  const uint8_t* x;
  size_t x_stride;
  const uint32_t* t;
  uint8_t* y;
  size_t y_stride;
  xnn_u8_rmax_ukernel_function rmax_ukernel;
  xnn_u8_lut32norm_ukernel_function lut_norm_ukernel;
};

#ifndef __cplusplus
  XNN_PRIVATE void xnn_compute_u8_softmax(
      const struct u8_softmax_context context[restrict XNN_MIN_ELEMENTS(1)],
      size_t batch_index);
#endif

typedef void (*xnn_compute_reciprocal_function)(const void* input, void* output);

struct floating_point_softmax_context {
  size_t n;
  const void* x;
  size_t x_stride;
  void* y;
  size_t y_stride;
  xnn_rmax_ukernel_function rmax_ukernel;
  xnn_raddstoreexpminusmax_ukernel_function raddstoreexpminusmax_ukernel;
  xnn_compute_reciprocal_function compute_reciprocal;
  xnn_vbinary_ukernel_function vmulc_ukernel;
  union {
    union xnn_f16_minmax_params f16;
    union xnn_f32_minmax_params f32;
  } minmax_params;
  union {
    union xnn_f16_expminus_params f16;
    union xnn_f32_expminus_params f32;
  } expminus_params;
};

#ifndef __cplusplus
  XNN_PRIVATE void xnn_compute_floating_point_softmax(
      const struct floating_point_softmax_context context[restrict XNN_MIN_ELEMENTS(1)],
      size_t batch_index);
#endif
