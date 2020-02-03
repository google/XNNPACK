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

#include <pthreadpool.h>

#ifdef __cplusplus
extern "C" {
#endif

/// The number of bytes XNNPACK may read beyond array bounds.
/// The caller must allocate at this this many extra bytes after the tensor data passed to XNNPACK.
///
/// Note: XNNPACK reads, but never writes beyond array bounds.
#define XNN_EXTRA_BYTES 16

/// Maximum number of dimensions in tensor shape.
#define XNN_MAX_TENSOR_DIMS 6

/// The convolution operator represents a depthwise convolution, and use HWGo layout for filters.
#define XNN_FLAG_DEPTHWISE_CONVOLUTION 0x00000001

/// Assume transposed weights in a fully connected operator.
#define XNN_FLAG_TRANSPOSE_WEIGHTS 0x00000001

/// The operator assumes NHWC layout for the input, regardless of the output layout.
#define XNN_FLAG_INPUT_NHWC 0x00000002

/// Match "SAME" padding in TensorFlow. Exact padding values are computed dynamically depending on input size.
#define XNN_FLAG_TENSORFLOW_SAME_PADDING 0x00000004

/// Match behaviour of TensorFlow 1.x.
#define XNN_FLAG_TENSORFLOW_LEGACY_MODE 0x00000004

/// Align corners of input and output images in resize operations.
#define XNN_FLAG_ALIGN_CORNERS 0x00000008

/// Status code for any XNNPACK function call.
enum xnn_status {
  /// The call succeeded, and all output arguments now contain valid data.
  xnn_status_success = 0,
  xnn_status_uninitialized = 1,
  xnn_status_invalid_parameter = 2,
  xnn_status_invalid_state = 3,
  xnn_status_unsupported_parameter = 4,
  xnn_status_unsupported_hardware = 5,
  xnn_status_out_of_memory = 6,
};

struct xnn_allocator {
  /// User-specified pointer that will be passed as-is to all functions in this structure.
  void* context;
  /// Pointer to a function to be called for general memory allocation.
  ///
  /// @param context - The user-specified pointer from xnn_allocator structure.
  /// @param size - The size of the memory block to allocate, in bytes.
  ///
  /// @returns Pointer to the allocated memory block of at least @ref size bytes.
  ///          If allocation fails, the function must return NULL.
  void* (*allocate)(void* context, size_t size);
  /// Pointer to a function to be called for general memory re-allocation, i.e. to increase or shrink a previously
  /// allocated memory block. The content of the old memory block is copied to the new memory block.
  ///
  /// @param context - The user-specified pointer from xnn_allocator structure.
  /// @param pointer - Pointer to a memory block allocated by @ref allocate or @ref reallocate functions. Can be NULL.
  ///                  If the pointer is NULL, the @ref reallocate call is equivalent to an @ref allocate call.
  /// @param size - The new size of the memory block to allocate, in bytes.
  ///
  /// @returns Pointer to the newly allocated memory block of at least @ref size bytes with the content of the previous
  ///          memory block.
  ///          If allocation fails, the function must return NULL, but must not release the previous memory block.
  void* (*reallocate)(void* context, void* pointer, size_t size);
  /// Pointer to a function to be called for general memory de-allocation.
  ///
  /// @param context - The user-specified pointer from xnn_allocator structure.
  /// @param pointer - Pointer to a memory block allocated by @ref allocate or @ref reallocate functions. Can be NULL.
  ///                  If the pointer is NULL, the @ref deallocate call is a no-op.
  void (*deallocate)(void* context, void* pointer);
  /// Pointer to a function to be called for aligned memory allocation.
  ///
  /// @param context - The user-specified pointer from xnn_allocator structure.
  /// @param alignment - The alignment of the memory block to allocate, in bytes. Alignment is always a power-of-2.
  /// @param size - The size of the memory block to allocate, in bytes.
  ///
  /// @returns Pointer to the allocated memory block of at least @ref size bytes.
  ///          If allocation fails, the function must return NULL.
  void* (*aligned_allocate)(void* context, size_t alignment, size_t size);
  /// Pointer to a function to be called for aligned memory de-allocation.
  ///
  /// @param context - The user-specified pointer from xnn_allocator structure.
  /// @param pointer - Pointer to a memory block allocated by @ref aligned_allocate function. Can be NULL.
  ///                  If the pointer is NULL, the @ref aligned_deallocate call is a no-op.
  void (*aligned_deallocate)(void* context, void* pointer);
};

/// Initialize XNNPACK library.
///
/// XNNPACK must be successfully initialized before use.
/// During initialization, XNNPACK populates internal structures depending on host processor. It can be time-consuming.
///
/// @param[in] allocator - structure with function pointers to be use for memory allocation and de-allocation.
///                        If this argument is NULL, system-provided memory management functions (e.g. malloc/free)
///                        will be used.
///
/// @retval xnn_status_success - XNNPACK is succesfully initialized and ready to use.
/// @retval xnn_status_out_of_memory - initialization failed due to out-of-memory condition.
/// @retval xnn_status_unsupported_hardware - initialization failed because the host processor does not satisfy the
///                                           minimum hardware requirements for XNNPACK. E.g. this may happen on x86
///                                           processors without SSE2 extension, or on 32-bit ARM processors without
///                                           the NEON SIMD extension.
enum xnn_status xnn_initialize(const struct xnn_allocator* allocator);

/// Deinitialize XNNPACK library.
///
/// To avoid memory and resource leaks, users must call xnn_deinitialize once for each successful xnn_initialize call.
///
/// @retval xnn_status_success - deinitialization call succeeded.
enum xnn_status xnn_deinitialize(void);

typedef struct xnn_subgraph* xnn_subgraph_t;

enum xnn_status xnn_create_subgraph(
  uint32_t external_value_ids,
  uint32_t flags,
  xnn_subgraph_t* subgraph_out);

enum xnn_status xnn_delete_subgraph(
  xnn_subgraph_t subgraph);

#define XNN_VALUE_FLAG_EXTERNAL_INPUT  0x00000001
#define XNN_VALUE_FLAG_EXTERNAL_OUTPUT 0x00000002

#define XNN_INVALID_VALUE_ID UINT32_MAX

enum xnn_datatype {
  xnn_datatype_invalid = 0,
  xnn_datatype_fp32 = 1,
  xnn_datatype_fp16 = 2,
};

/// Define a tensor-type Value and add it to a subgraph.
///
/// @param datatype - type of tensor elements.
/// @param num_dims - number of dimensions in the shape.
/// @param dims - pointer to an array of @a num_dims shape dimensions. If num_dims is 0, this pointer can be NULL.
/// @param data - pointer to static data used for tensor initialization. If the tensor is not statically initialized,
///               this pointer must be is NULL.
/// @param external_id - external ID for the Value. The ID must be within the range of reversed Value IDs specified in
///                      subgraph creation. If the external ID is XNN_INVALID_VALUE_ID, an internal ID will be created
///                      for the Value.
/// @param subgraph - subgraph that will own the created value.
/// @param id_out - pointer to the variable that will be initialized with the Value ID upon successful return.
enum xnn_status xnn_define_tensor_value(
  xnn_subgraph_t subgraph,
  enum xnn_datatype datatype,
  size_t num_dims,
  const size_t* dims,
  const void* data,
  uint32_t external_id,
  uint32_t flags,
  uint32_t* id_out);

/// Define a 2D Convolution node and add it to a subgraph.
///
/// @param input_padding_top - implicit zero-padding above 2D input data.
/// @param input_padding_right - implicit zero-padding to the right of 2D input data.
/// @param input_padding_bottom - implicit zero-padding below 2D input data.
/// @param input_padding_left - implicit zero-padding to the left of 2D input data.
/// @param kernel_height - kernel (filter) height.
/// @param kernel_width - kernel (filter) width.
/// @param subsampling_height - height of subsampling region for convolution output (convolution height stride).
/// @param subsampling_width - width of subsampling region for convolution output (convolution width stride).
/// @param dilation_height - dilation of kernel elements along the height dimension.
/// @param dilation_width - dilation of kernel elements along the width dimension.
/// @param groups - number of convolution groups.
/// @param group_input_channels - number of input channels per group.
/// @param group_output_channels - number of output channels per group.
/// @param output_min - lower bound for clipping output values.
/// @param output_max - upper bound for clipping output values.
/// @param input_id - input tensor ID. Must be a 4D tensor with [N, IH, IW, groups * group_input_channels] dimensions.
/// @param filter_id - filter tensor ID. Must ge a 4D tensor with
///                 [groups * group_output_channels, kernel_height, kernel_width, group_input_channels] dimensions.
/// @param bias_id - bias tensor ID. Must be a 1D tensor with [groups * group_output_channels] dimensions.
/// @param output_id - output tensor ID. Must be a 4D tensor with [N, OH, OW, groups * group_output_channels] dimensions.
enum xnn_status xnn_define_convolution_2d(
  xnn_subgraph_t subgraph,
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
  float output_min,
  float output_max,
  uint32_t input_id,
  uint32_t filter_id,
  uint32_t bias_id,
  uint32_t output_id,
  uint32_t flags);

/// Define a 2D Depthwise Convolution node and add it to a subgraph.
///
/// @param input_padding_top - implicit zero-padding above 2D input data.
/// @param input_padding_right - implicit zero-padding to the right of 2D input data.
/// @param input_padding_bottom - implicit zero-padding below 2D input data.
/// @param input_padding_left - implicit zero-padding to the left of 2D input data.
/// @param kernel_height - kernel (filter) height.
/// @param kernel_width - kernel (filter) width.
/// @param subsampling_height - height of subsampling region for convolution output (convolution height stride).
/// @param subsampling_width - width of subsampling region for convolution output (convolution width stride).
/// @param dilation_height - dilation of kernel elements along the height dimension.
/// @param dilation_width - dilation of kernel elements along the width dimension.
/// @param depth_multiplier - ratio of output channels to input channels.
/// @param input_channels - number of input channels.
/// @param output_min - lower bound for clipping output values.
/// @param output_max - upper bound for clipping output values.
/// @param input_id - input tensor. Must be a 4D tensor with [N, IH, IW, input_channels] dimensions.
/// @param filter_id - filter tensor. Must ge a 4D tensor with
///                 [1, kernel_height, kernel_width, input_channels * depth_multiplier] dimensions.
/// @param bias_id - bias tensor. Must be a 1D tensor with [input_channels * depth_multiplier] dimensions.
/// @param output_id - output tensor. Must be a 4D tensor with [N, OH, OW, input_channels * depth_multiplier] dimensions.
enum xnn_status xnn_define_depthwise_convolution_2d(
  xnn_subgraph_t subgraph,
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
  uint32_t depth_multiplier,
  size_t input_channels,
  float output_min,
  float output_max,
  uint32_t input_id,
  uint32_t filter_id,
  uint32_t bias_id,
  uint32_t output_id,
  uint32_t flags);

typedef struct xnn_runtime* xnn_runtime_t;

enum xnn_status xnn_create_runtime(
  xnn_subgraph_t subgraph,
  xnn_runtime_t* runtime_out);

struct xnn_external_value {
  uint32_t id;
  void* data;
};

enum xnn_status xnn_setup_runtime(
  xnn_runtime_t runtime,
  size_t num_external_values,
  const struct xnn_external_value* external_values);

enum xnn_status xnn_invoke_runtime(
  xnn_runtime_t runtime);

enum xnn_status xnn_delete_runtime(
  xnn_runtime_t runtime);

typedef struct xnn_operator* xnn_operator_t;

enum xnn_status xnn_run_operator(
  xnn_operator_t op,
  pthreadpool_t threadpool);

enum xnn_status xnn_delete_operator(
  xnn_operator_t op);

#ifndef XNN_NO_F32_OPERATORS

enum xnn_status xnn_create_add_nc_f32(
  size_t channels,
  size_t a_stride,
  size_t b_stride,
  size_t sum_stride,
  float sum_min,
  float sum_max,
  uint32_t flags,
  xnn_operator_t* add_op_out);

enum xnn_status xnn_setup_add_nc_f32(
  xnn_operator_t add_op,
  size_t batch_size,
  const float* a,
  const float* b,
  float* sum,
  pthreadpool_t threadpool);

enum xnn_status xnn_create_add_nd_f32(
  float output_min,
  float output_max,
  uint32_t flags,
  xnn_operator_t* add_op_out);

enum xnn_status xnn_setup_add_nd_f32(
  xnn_operator_t add_op,
  size_t num_input1_dims,
  const size_t* input1_shape,
  size_t num_input2_dims,
  const size_t* input2_shape,
  const float* input1,
  const float* input2,
  float* output,
  pthreadpool_t threadpool);

enum xnn_status xnn_create_argmax_pooling2d_nhwc_f32(
  uint32_t input_padding_top,
  uint32_t input_padding_right,
  uint32_t input_padding_bottom,
  uint32_t input_padding_left,
  uint32_t pooling_height,
  uint32_t pooling_width,
  size_t channels,
  size_t input_pixel_stride,
  size_t output_pixel_stride,
  float output_min,
  float output_max,
  uint32_t flags,
  xnn_operator_t* argmax_pooling_op_out);

enum xnn_status xnn_setup_argmax_pooling2d_nhwc_f32(
  xnn_operator_t argmax_pooling_op,
  size_t batch_size,
  size_t input_height,
  size_t input_width,
  const float* input,
  float* output,
  uint32_t* index,
  pthreadpool_t threadpool);

enum xnn_status xnn_create_average_pooling2d_nhwc_f32(
  uint32_t input_padding_top,
  uint32_t input_padding_right,
  uint32_t input_padding_bottom,
  uint32_t input_padding_left,
  uint32_t pooling_height,
  uint32_t pooling_width,
  uint32_t stride_height,
  uint32_t stride_width,
  size_t channels,
  size_t input_pixel_stride,
  size_t output_pixel_stride,
  float output_min,
  float output_max,
  uint32_t flags,
  xnn_operator_t* average_pooling_op_out);

enum xnn_status xnn_setup_average_pooling2d_nhwc_f32(
  xnn_operator_t average_pooling_op,
  size_t batch_size,
  size_t input_height,
  size_t input_width,
  const float* input,
  float* output,
  pthreadpool_t threadpool);

enum xnn_status xnn_create_clamp_nc_f32(
  size_t channels,
  size_t input_stride,
  size_t output_stride,
  float output_min,
  float output_max,
  uint32_t flags,
  xnn_operator_t* clamp_op_out);

enum xnn_status xnn_setup_clamp_nc_f32(
  xnn_operator_t clamp_op,
  size_t batch_size,
  const float* input,
  float* output,
  pthreadpool_t threadpool);

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
  size_t input_pixel_stride,
  size_t output_pixel_stride,
  const float* kernel,
  const float* bias,
  float output_min,
  float output_max,
  uint32_t flags,
  xnn_operator_t* convolution_op_out);

enum xnn_status xnn_setup_convolution2d_nhwc_f32(
  xnn_operator_t convolution_op,
  size_t batch_size,
  size_t input_height,
  size_t input_width,
  const float* input,
  float* output,
  pthreadpool_t threadpool);

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
  xnn_operator_t* deconvolution_op_out);

enum xnn_status xnn_setup_deconvolution2d_nhwc_f32(
  xnn_operator_t deconvolution_op,
  size_t batch_size,
  size_t input_height,
  size_t input_width,
  uint32_t adjustment_height,
  uint32_t adjustment_width,
  const float* input,
  float* output,
  pthreadpool_t threadpool);

enum xnn_status xnn_create_divide_nd_f32(
  float output_min,
  float output_max,
  uint32_t flags,
  xnn_operator_t* divide_op_out);

enum xnn_status xnn_setup_divide_nd_f32(
  xnn_operator_t divide_op,
  size_t num_input1_dims,
  const size_t* input1_shape,
  size_t num_input2_dims,
  const size_t* input2_shape,
  const float* input1,
  const float* input2,
  float* output,
  pthreadpool_t threadpool);

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
  xnn_operator_t* fully_connected_op_out);

enum xnn_status xnn_setup_fully_connected_nc_f32(
  xnn_operator_t fully_connected_op,
  size_t batch_size,
  const float* input,
  float* output,
  pthreadpool_t threadpool);

enum xnn_status xnn_create_global_average_pooling_nwc_f32(
  size_t channels,
  size_t input_stride,
  size_t output_stride,
  float output_min,
  float output_max,
  uint32_t flags,
  xnn_operator_t* global_average_pooling_op_out);

enum xnn_status xnn_setup_global_average_pooling_nwc_f32(
  xnn_operator_t global_average_pooling_op,
  size_t batch_size,
  size_t width,
  const float* input,
  float* output,
  pthreadpool_t threadpool);

enum xnn_status xnn_create_hardswish_nc_f32(
  size_t channels,
  size_t input_stride,
  size_t output_stride,
  uint32_t flags,
  xnn_operator_t* hardswish_op_out);

enum xnn_status xnn_setup_hardswish_nc_f32(
  xnn_operator_t hardswish_op,
  size_t batch_size,
  const float* input,
  float* output,
  pthreadpool_t threadpool);

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
  size_t channels,
  size_t input_pixel_stride,
  size_t output_pixel_stride,
  float output_min,
  float output_max,
  uint32_t flags,
  xnn_operator_t* max_pooling_op_out);

enum xnn_status xnn_setup_max_pooling2d_nhwc_f32(
  xnn_operator_t max_pooling_op,
  size_t batch_size,
  size_t input_height,
  size_t input_width,
  const float* input,
  float* output,
  pthreadpool_t threadpool);

enum xnn_status xnn_create_maximum_nd_f32(
  uint32_t flags,
  xnn_operator_t* maximum_op_out);

enum xnn_status xnn_setup_maximum_nd_f32(
  xnn_operator_t maximum_op,
  size_t num_input1_dims,
  const size_t* input1_shape,
  size_t num_input2_dims,
  const size_t* input2_shape,
  const float* input1,
  const float* input2,
  float* output,
  pthreadpool_t threadpool);

enum xnn_status xnn_create_minimum_nd_f32(
  uint32_t flags,
  xnn_operator_t* minimum_op_out);

enum xnn_status xnn_setup_minimum_nd_f32(
  xnn_operator_t minimum_op,
  size_t num_input1_dims,
  const size_t* input1_shape,
  size_t num_input2_dims,
  const size_t* input2_shape,
  const float* input1,
  const float* input2,
  float* output,
  pthreadpool_t threadpool);

enum xnn_status xnn_create_multiply_nd_f32(
  float output_min,
  float output_max,
  uint32_t flags,
  xnn_operator_t* multiply_op_out);

enum xnn_status xnn_setup_multiply_nd_f32(
  xnn_operator_t multiply_op,
  size_t num_input1_dims,
  const size_t* input1_shape,
  size_t num_input2_dims,
  const size_t* input2_shape,
  const float* input1,
  const float* input2,
  float* output,
  pthreadpool_t threadpool);

enum xnn_status xnn_create_prelu_nc_f32(
  size_t channels,
  size_t input_stride,
  size_t output_stride,
  const float* negative_slope,
  float output_min,
  float output_max,
  uint32_t flags,
  xnn_operator_t* prelu_op_out);

enum xnn_status xnn_setup_prelu_nc_f32(
  xnn_operator_t prelu_op,
  size_t batch_size,
  const float* input,
  float* output,
  pthreadpool_t threadpool);

enum xnn_status xnn_create_resize_bilinear2d_nhwc_f32(
  size_t channels,
  size_t input_pixel_stride,
  size_t output_pixel_stride,
  uint32_t flags,
  xnn_operator_t* resize_op_out);

enum xnn_status xnn_setup_resize_bilinear2d_nhwc_f32(
  xnn_operator_t resize_op,
  size_t batch_size,
  size_t input_height,
  size_t input_width,
  size_t output_height,
  size_t output_width,
  const float* input,
  float* output,
  pthreadpool_t threadpool);

enum xnn_status xnn_create_sigmoid_nc_f32(
  size_t channels,
  size_t input_stride,
  size_t output_stride,
  uint32_t flags,
  xnn_operator_t* sigmoid_op_out);

enum xnn_status xnn_setup_sigmoid_nc_f32(
  xnn_operator_t sigmoid_op,
  size_t batch_size,
  const float* input,
  float* output,
  pthreadpool_t threadpool);

enum xnn_status xnn_create_softmax_nc_f32(
  size_t channels,
  size_t input_stride,
  size_t output_stride,
  uint32_t flags,
  xnn_operator_t* softmax_op_out);

enum xnn_status xnn_setup_softmax_nc_f32(
  xnn_operator_t softmax_op,
  size_t batch_size,
  const float* input,
  float* output,
  pthreadpool_t threadpool);

enum xnn_status xnn_create_subtract_nd_f32(
  float output_min,
  float output_max,
  uint32_t flags,
  xnn_operator_t* subtract_op_out);

enum xnn_status xnn_setup_subtract_nd_f32(
  xnn_operator_t subtract_op,
  size_t num_input1_dims,
  const size_t* input1_shape,
  size_t num_input2_dims,
  const size_t* input2_shape,
  const float* input1,
  const float* input2,
  float* output,
  pthreadpool_t threadpool);

#ifndef XNN_NO_NCHW_OPERATORS

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
  const float* kernel,
  const float* bias,
  float output_min,
  float output_max,
  uint32_t flags,
  xnn_operator_t* convolution_op_out);

enum xnn_status xnn_setup_convolution2d_nchw_f32(
  xnn_operator_t convolution_op,
  size_t batch_size,
  size_t input_batch_stride,
  size_t output_batch_stride,
  size_t input_height,
  size_t input_width,
  const float* input,
  float* output,
  pthreadpool_t threadpool);

enum xnn_status xnn_create_global_average_pooling_ncw_f32(
  size_t channels,
  float output_min,
  float output_max,
  uint32_t flags,
  xnn_operator_t* global_average_pooling_op_out);

enum xnn_status xnn_setup_global_average_pooling_ncw_f32(
  xnn_operator_t global_average_pooling_op,
  size_t batch_size,
  size_t width,
  const float* input,
  float* output,
  pthreadpool_t threadpool);

#endif  // XNN_NO_NCHW_OPERATORS

#endif  // XNN_NO_F32_OPERATORS

#ifndef XNN_NO_X32_OPERATORS

enum xnn_status xnn_create_channel_pad_nc_x32(
  size_t input_channels,
  size_t pad_before_channels,
  size_t pad_after_channels,
  size_t input_stride,
  size_t output_stride,
  const void* pad_value,
  uint32_t flags,
  xnn_operator_t* channel_pad_op_out);

enum xnn_status xnn_setup_channel_pad_nc_x32(
  xnn_operator_t channel_pad_op,
  size_t batch_size,
  const void* input,
  void* output,
  pthreadpool_t threadpool);

enum xnn_status xnn_create_channel_shuffle_nc_x32(
  size_t groups,
  size_t group_channels,
  size_t input_stride,
  size_t output_stride,
  uint32_t flags,
  xnn_operator_t* channel_shuffle_op_out);

enum xnn_status xnn_setup_channel_shuffle_nc_x32(
  xnn_operator_t channel_shuffle_op,
  size_t batch_size,
  const void* input,
  void* output,
  pthreadpool_t threadpool);

enum xnn_status xnn_create_unpooling2d_nhwc_x32(
  uint32_t input_padding_top,
  uint32_t input_padding_right,
  uint32_t input_padding_bottom,
  uint32_t input_padding_left,
  uint32_t pooling_height,
  uint32_t pooling_width,
  size_t channels,
  size_t input_pixel_stride,
  size_t output_pixel_stride,
  uint32_t flags,
  xnn_operator_t* unpooling_op_out);

enum xnn_status xnn_setup_unpooling2d_nhwc_x32(
  xnn_operator_t unpooling_op,
  size_t batch_size,
  size_t input_height,
  size_t input_width,
  const void* input,
  const uint32_t* index,
  void* output,
  pthreadpool_t threadpool);

#endif  // XNN_NO_X32_OPERATORS

#ifndef XNN_NO_Q8_OPERATORS

enum xnn_status xnn_create_add_nc_q8(
  size_t channels,
  size_t a_stride,
  size_t b_stride,
  size_t sum_stride,
  uint8_t a_zero_point,
  float a_scale,
  uint8_t b_zero_point,
  float b_scale,
  uint8_t sum_zero_point,
  float sum_scale,
  uint8_t sum_min,
  uint8_t sum_max,
  uint32_t flags,
  xnn_operator_t* add_op_out);

enum xnn_status xnn_setup_add_nc_q8(
  xnn_operator_t add_op,
  size_t batch_size,
  const uint8_t* a,
  const uint8_t* b,
  uint8_t* sum,
  pthreadpool_t threadpool);

enum xnn_status xnn_create_average_pooling2d_nhwc_q8(
  uint32_t input_padding_top,
  uint32_t input_padding_right,
  uint32_t input_padding_bottom,
  uint32_t input_padding_left,
  uint32_t pooling_height,
  uint32_t pooling_width,
  uint32_t stride_height,
  uint32_t stride_width,
  size_t channels,
  size_t input_pixel_stride,
  size_t output_pixel_stride,
  uint8_t input_zero_point,
  float input_scale,
  uint8_t output_zero_point,
  float output_scale,
  uint8_t output_min,
  uint8_t output_max,
  uint32_t flags,
  xnn_operator_t* average_pooling_op_out);

enum xnn_status xnn_setup_average_pooling2d_nhwc_q8(
  xnn_operator_t average_pooling_op,
  size_t batch_size,
  size_t input_height,
  size_t input_width,
  const uint8_t* input,
  uint8_t* output,
  pthreadpool_t threadpool);

enum xnn_status xnn_create_convolution2d_nhwc_q8(
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
  xnn_operator_t* convolution_op_out);

enum xnn_status xnn_setup_convolution2d_nhwc_q8(
  xnn_operator_t convolution_op,
  size_t batch_size,
  size_t input_height,
  size_t input_width,
  const uint8_t* input,
  uint8_t* output,
  pthreadpool_t threadpool);

enum xnn_status xnn_create_deconvolution2d_nhwc_q8(
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
  xnn_operator_t* deconvolution_op_out);

enum xnn_status xnn_setup_deconvolution2d_nhwc_q8(
  xnn_operator_t deconvolution_op,
  size_t batch_size,
  size_t input_height,
  size_t input_width,
  uint32_t adjustment_height,
  uint32_t adjustment_width,
  const uint8_t* input,
  uint8_t* output,
  pthreadpool_t threadpool);

enum xnn_status xnn_create_fully_connected_nc_q8(
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
  xnn_operator_t* fully_connected_op_out);

enum xnn_status xnn_setup_fully_connected_nc_q8(
  xnn_operator_t fully_connected_op,
  size_t batch_size,
  const uint8_t* input,
  uint8_t* output,
  pthreadpool_t threadpool);

enum xnn_status xnn_create_global_average_pooling_nwc_q8(
  size_t channels,
  size_t input_stride,
  size_t output_stride,
  uint8_t input_zero_point,
  float input_scale,
  uint8_t output_zero_point,
  float output_scale,
  uint8_t output_min,
  uint8_t output_max,
  uint32_t flags,
  xnn_operator_t* global_average_pooling_op_out);

enum xnn_status xnn_setup_global_average_pooling_nwc_q8(
  xnn_operator_t global_average_pooling_op,
  size_t batch_size,
  size_t width,
  const uint8_t* input,
  uint8_t* output,
  pthreadpool_t threadpool);

enum xnn_status xnn_create_leaky_relu_nc_q8(
  size_t channels,
  size_t input_stride,
  size_t output_stride,
  float negative_slope,
  uint8_t input_zero_point,
  float input_scale,
  uint8_t output_zero_point,
  float output_scale,
  uint8_t output_min,
  uint8_t output_max,
  uint32_t flags,
  xnn_operator_t* leaky_relu_op_out);

enum xnn_status xnn_setup_leaky_relu_nc_q8(
  xnn_operator_t leaky_relu_op,
  size_t batch_size,
  const uint8_t* input,
  uint8_t* output,
  pthreadpool_t threadpool);

enum xnn_status xnn_create_sigmoid_nc_q8(
  size_t channels,
  size_t input_stride,
  size_t output_stride,
  uint8_t input_zero_point,
  float input_scale,
  uint8_t output_zero_point,
  float output_scale,
  uint8_t output_min,
  uint8_t output_max,
  uint32_t flags,
  xnn_operator_t* sigmoid_op_out);

enum xnn_status xnn_setup_sigmoid_nc_q8(
  xnn_operator_t sigmoid_op,
  size_t batch_size,
  const uint8_t* input,
  uint8_t* output,
  pthreadpool_t threadpool);

enum xnn_status xnn_create_softmax_nc_q8(
  size_t channels,
  size_t input_stride,
  size_t output_stride,
  float input_scale,
  uint8_t output_zero_point,
  float output_scale,
  uint32_t flags,
  xnn_operator_t* softmax_op_out);

enum xnn_status xnn_setup_softmax_nc_q8(
  xnn_operator_t softmax_op,
  size_t batch_size,
  const uint8_t* input,
  uint8_t* output,
  pthreadpool_t threadpool);

#endif  // XNN_NO_Q8_OPERATORS

#ifndef XNN_NO_U8_OPERATORS

enum xnn_status xnn_create_clamp_nc_u8(
  size_t channels,
  size_t input_stride,
  size_t output_stride,
  uint8_t output_min,
  uint8_t output_max,
  uint32_t flags,
  xnn_operator_t* clamp_op_out);

enum xnn_status xnn_setup_clamp_nc_u8(
  xnn_operator_t clamp_op,
  size_t batch_size,
  const uint8_t* input,
  uint8_t* output,
  pthreadpool_t threadpool);

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
  size_t channels,
  size_t input_pixel_stride,
  size_t output_pixel_stride,
  uint8_t output_min,
  uint8_t output_max,
  uint32_t flags,
  xnn_operator_t* max_pooling_op_out);

enum xnn_status xnn_setup_max_pooling2d_nhwc_u8(
  xnn_operator_t max_pooling_op,
  size_t batch_size,
  size_t input_height,
  size_t input_width,
  const uint8_t* input,
  uint8_t* output,
  pthreadpool_t threadpool);

#endif  // XNN_NO_U8_OPERATORS

#ifndef XNN_NO_X8_OPERATORS

enum xnn_status xnn_create_channel_shuffle_nc_x8(
  size_t groups,
  size_t group_channels,
  size_t input_stride,
  size_t output_stride,
  uint32_t flags,
  xnn_operator_t* channel_shuffle_op_out);

enum xnn_status xnn_setup_channel_shuffle_nc_x8(
  xnn_operator_t channel_shuffle_op,
  size_t batch_size,
  const void* input,
  void* output,
  pthreadpool_t threadpool);

#endif  // XNN_NO_X8_OPERATORS

#ifdef __cplusplus
}  // extern "C"
#endif
