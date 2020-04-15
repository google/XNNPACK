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
  xnn_parallelization_type_3d_tile_2d,
  xnn_parallelization_type_4d_tile_2d,
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
    pthreadpool_task_3d_tile_2d_t task_3d_tile_2d;
    pthreadpool_task_4d_tile_2d_t task_4d_tile_2d;
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
  union {
    union xnn_q8_gemm_params q8;
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
  // Input matrix A.
  const void* a;
  // Packed bias elements and non-zero filter elements.
  const void* packed_weights;
  // Input pointer increments, in bytes, after each processed non-zero weight.
  const int32_t* input_increments;
  // Number of non-zero filter elements per each N (output channel) dimension.
  const uint32_t* output_channel_nonzeros;
  // Output matrix C.
  void* c;
  // Stride, in bytes, between matrices A corresponding to different images in batched 1x1 Convolution
  size_t batched_a_stride;
  // Stride, in bytes, between matrices C corresponding to different images in batched 1x1 Convolution
  size_t batched_c_stride;
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
    union xnn_q8_gemm_params q8;
    union xnn_f32_minmax_params f32;
  } params;
};

#ifndef __cplusplus
  XNN_PRIVATE void xnn_compute_grouped_igemm(
      const struct igemm_context context[restrict XNN_MIN_ELEMENTS(1)],
      size_t batch_index,
      size_t group_index,
      size_t mr_block_start,
      size_t nr_block_start,
      size_t mr_block_size,
      size_t nr_block_size);

  XNN_PRIVATE void xnn_compute_igemm(
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
        size_t batch_index,
        size_t group_index,
        size_t mr_block_start,
        size_t nr_block_start,
        size_t mr_block_size,
        size_t nr_block_size);

    XNN_PRIVATE void xnn_compute_hmp_igemm(
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
    union xnn_q8_gemm_params q8;
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
    union xnn_q8_gemm_params q8;
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

struct dconv2d_context {
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
    xnn_conv_hwc2spchw_ukernel_function hwc2spchw_ukernel;
  };
  union {
    union xnn_f32_minmax_params f32;
  } params;
};

#ifndef __cplusplus
  XNN_PRIVATE void xnn_compute_dconv2d_hwc2spchw(
      const struct dconv2d_context context[restrict XNN_MIN_ELEMENTS(1)],
      size_t batch_index,
      size_t output_y_start,
      size_t output_y_slice);
#endif

struct dwconv_context {
  size_t groups;
  const void** indirection_buffer;
  size_t indirection_buffer_row_stride;
  size_t indirection_buffer_col_stride;
  const void* packed_weights;
  void* output;
  size_t output_width;
  size_t output_row_stride;
  size_t output_col_increment;
  union {
    union xnn_q8_gemm_params q8;
    union xnn_f32_minmax_params f32;
  } params;
  union {
    xnn_dwconv_unipass_ukernel_function unipass_ukernel;
  };
};

#ifndef __cplusplus
  XNN_PRIVATE void xnn_compute_dwconv_unipass(
      const struct dwconv_context context[restrict XNN_MIN_ELEMENTS(1)],
      size_t output_y);
#endif

struct dwconv2d_context {
  size_t output_height;
  size_t input_width;
  const void* input;
  size_t input_channel_stride;
  size_t input_batch_stride;
  const void* packed_weights;
  size_t weights_channel_stride;
  void* output;
  size_t output_channel_stride;
  size_t output_batch_stride;
  size_t input_tuple_stride;
  size_t output_tuple_stride;
  size_t input_pixel_stride;
  size_t output_pixel_stride;
  union {
    union xnn_f32_spchw_params f32;
  } params;
  union {
    xnn_dwconv_spchw_ukernel_function spchw_ukernel;
  };
};

#ifndef __cplusplus
  XNN_PRIVATE void xnn_compute_dwconv2d_spchw(
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
  void** indirect_output;
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
    union xnn_f32_minmax_params f32;
  } params;
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
    union xnn_q8_avgpool_params q8;
    union xnn_f32_scaleminmax_params f32;
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
    union xnn_u8_minmax_params u8;
    union xnn_f32_minmax_params f32;
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
    union xnn_q8_avgpool_params q8;
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
  xnn_gavgpool_spchw_ukernel_function ukernel;
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

#ifndef __cplusplus
  XNN_PRIVATE void xnn_compute_resize_bilinear(
      const struct resize_bilinear_context context[restrict XNN_MIN_ELEMENTS(1)],
      size_t batch_index,
      size_t pixel_start,
      size_t pixel_range);
#endif

struct add_strided_context {
  size_t n;
  const void* a;
  size_t a_stride;
  const void* b;
  size_t b_stride;
  const void* y;
  size_t y_stride;
  union {
    union xnn_q8_add_params q8;
    union xnn_f32_minmax_params f32;
  } params;
  xnn_vadd_ukernel_function ukernel;
};

#ifndef __cplusplus
  XNN_PRIVATE void xnn_compute_add_strided(
      const struct add_strided_context context[restrict XNN_MIN_ELEMENTS(1)],
      size_t batch_index,
      size_t batch_range);
#endif

struct add_contiguous_context {
  const void* a;
  const void* b;
  void* y;
  union {
    union xnn_q8_add_params q8;
    union xnn_f32_minmax_params f32;
  } params;
  xnn_vadd_ukernel_function ukernel;
};

#ifndef __cplusplus
  XNN_PRIVATE void xnn_compute_add_contiguous(
      const struct add_contiguous_context context[restrict XNN_MIN_ELEMENTS(1)],
      size_t offset,
      size_t size);
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
    union xnn_q8_add_params q8;
    union xnn_f32_minmax_params f32;
  } params;
  xnn_vbinary_ukernel_function ukernel;
};

#ifndef __cplusplus
  XNN_PRIVATE void xnn_compute_elementwise_binary_5d(
      const struct elementwise_binary_context context[restrict XNN_MIN_ELEMENTS(1)],
      size_t i, size_t j, size_t k, size_t l, size_t m, size_t l_range, size_t m_range);
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
  xnn_univector_ukernel_function ukernel;
  union {
    union xnn_u8_minmax_params u8_output;
    union xnn_f32_minmax_params f32_output;
    union xnn_f32_hswish_params f32_hswish;
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
  size_t x_stride;
  void* y;
  size_t y_stride;
  xnn_univector_ukernel_function ukernel;
  union {
    union xnn_u8_minmax_params u8_output;
    union xnn_f32_minmax_params f32_output;
    union xnn_f32_hswish_params f32_hswish;
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
    union xnn_f32_minmax_params f32;
  } params;
};

#ifndef __cplusplus
  XNN_PRIVATE void xnn_compute_vmulcaddc(
      const struct vmulcaddc_context context[restrict XNN_MIN_ELEMENTS(1)],
      size_t batch_start,
      size_t batch_size);
#endif

struct channel_pad_context {
  size_t n;
  size_t l;
  size_t r;
  uint32_t c;
  const void* x;
  size_t x_stride;
  void* y;
  size_t y_stride;
  xnn_pad_ukernel_function ukernel;
};

#ifndef __cplusplus
  XNN_PRIVATE void xnn_compute_channel_pad(
      const struct channel_pad_context context[restrict XNN_MIN_ELEMENTS(1)],
      size_t batch_start,
      size_t batch_range);
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

struct f32_three_pass_softmax_context {
  size_t n;
  const void* x;
  size_t x_stride;
  void* y;
  size_t y_stride;
  xnn_f32_rmax_ukernel_function rmax_ukernel;
  xnn_f32_raddstoreexpminusmax_ukernel_function raddstoreexpminusmax_ukernel;
  xnn_vbinary_ukernel_function vmulc_ukernel;
  union xnn_f32_minmax_params params;
};

#ifndef __cplusplus
  XNN_PRIVATE void xnn_compute_f32_three_pass_softmax(
      const struct f32_three_pass_softmax_context context[restrict XNN_MIN_ELEMENTS(1)],
      size_t batch_index);
#endif
