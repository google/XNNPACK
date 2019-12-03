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
#define XNN_MAX_TENSOR_DIMS 4

/// The convolution operator represents a depthwise convolution, and use HWGo layout for filters.
#define XNN_FLAG_DEPTHWISE_CONVOLUTION 0x00000001

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

enum xnn_status xnn_create_softargmax_nc_q8(
    size_t channels,
    size_t input_stride,
    size_t output_stride,
    float input_scale,
    uint8_t output_zero_point,
    float output_scale,
    uint32_t flags,
    xnn_operator_t* softargmax_op_out);

enum xnn_status xnn_setup_softargmax_nc_q8(
    xnn_operator_t softargmax_op,
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
