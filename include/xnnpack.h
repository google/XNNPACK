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
/// The caller must allocate at least this many extra bytes after the tensor data passed to XNNPACK.
///
/// Note: XNNPACK reads, but never writes beyond array bounds.
#define XNN_EXTRA_BYTES 16

/// Maximum number of dimensions in tensor shape.
#define XNN_MAX_TENSOR_DIMS 6

/// Allow sparse inference in a Runtime.
///
/// Note: this flag hints XNNPACK to consider sparse inference, but does not guarantee it.
#define XNN_FLAG_SPARSE_INFERENCE 0x00000001
#define XNN_FLAG_HINT_SPARSE_INFERENCE XNN_FLAG_SPARSE_INFERENCE

/// Allow IEEE FP16 inference in a Runtime.
///
/// Note: this flag hints XNNPACK to consider IEEE FP16 inference, but does not guarantee it.
#define XNN_FLAG_FP16_INFERENCE 0x00000002
#define XNN_FLAG_HINT_FP16_INFERENCE XNN_FLAG_FP16_INFERENCE

/// Force IEEE FP16 inference in a Runtime, and fail if FP16 inference is not possible.
///
/// Note: this flag guarantees that XNNPACK will use IEEE FP16 inference, or fail to create the Runtime object.
/// Warning: on x86 systems FP16 computations will be emulated at a substantial performance cost.
#define XNN_FLAG_FORCE_FP16_INFERENCE 0x00000004

/// Enable timing of each operator's runtime.
#define XNN_FLAG_BASIC_PROFILING 0x00000008

/// The convolution operator represents a depthwise convolution, and use HWGo layout for filters.
#define XNN_FLAG_DEPTHWISE_CONVOLUTION 0x00000001

/// Assume transposed weights in a fully connected operator.
#define XNN_FLAG_TRANSPOSE_WEIGHTS 0x00000001

/// The operator assumes NHWC layout for the input, regardless of the output layout.
#define XNN_FLAG_INPUT_NHWC 0x00000002

/// Match "SAME" padding in TensorFlow. Exact padding values are computed dynamically depending on input size.
#define XNN_FLAG_TENSORFLOW_SAME_PADDING 0x00000004

/// Implicitly flatten and reshape input of a Fully Connected operator into a 2D tensor.
#define XNN_FLAG_TENSORFLOW_RESHAPE_2D 0x00000004

/// Match behaviour of TensorFlow 1.x.
#define XNN_FLAG_TENSORFLOW_LEGACY_MODE 0x00000004

/// Static weights of the FP16 operator are in FP32 format.
#define XNN_FLAG_FP32_STATIC_WEIGHTS 0x00000008

/// Align corners of input and output images in resize operations.
#define XNN_FLAG_ALIGN_CORNERS 0x00000008

/// Yield worker threads of the thread pool to the system scheduler after the inference.
#define XNN_FLAG_YIELD_WORKERS 0x00000010

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
/// XNNPACK must be successfully initialized before use. During initialization, XNNPACK populates internal structures
/// depending on the host processor. Initialization can be time-consuming.
///
/// @param[in] allocator - structure with function pointers to be use for memory allocation and de-allocation.
///                        If this argument is NULL, system-provided memory management functions (e.g. malloc/free)
///                        will be used.
///
/// @retval xnn_status_success - XNNPACK is successfully initialized and ready to use.
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

/// Subgraph is an abstract representation of a neural network model.
/// Subgraph objects are used to define Values (tensors) and Nodes (operators) comprising the model.
typedef struct xnn_subgraph* xnn_subgraph_t;

/// Create a empty Subgraph object.
///
/// @param external_value_ids - number of Value IDs to reserve for communication with external graph representation.
///                             The Subgraph object would avoid creating internal Value IDs in the
///                             [0, reserved_value_ids-1] range.
/// @param flags - binary features of the subgraph. No supported flags are currently defined.
/// @param subgraph_out - pointer to the variable that will be initialized with a handle to the Subgraph object upon
///                       successful return.
enum xnn_status xnn_create_subgraph(
  uint32_t external_value_ids,
  uint32_t flags,
  xnn_subgraph_t* subgraph_out);

/// Destroy a Subgraph object, as well as Values, and Nodes associated with the subgraph.
///
/// @param subgraph - the Subgraph object to destroy.
enum xnn_status xnn_delete_subgraph(
  xnn_subgraph_t subgraph);

#define XNN_VALUE_FLAG_EXTERNAL_INPUT  0x00000001
#define XNN_VALUE_FLAG_EXTERNAL_OUTPUT 0x00000002
#define XNN_VALUE_FLAG_PERSISTENT      0x00000004

#define XNN_INVALID_VALUE_ID UINT32_MAX

/// Type of elements in a Value object.
enum xnn_datatype {
  /// Invalid data type. Valid Values never have this datatype.
  xnn_datatype_invalid = 0,
  /// IEEE754 single-precision floating-point.
  xnn_datatype_fp32 = 1,
  /// IEEE754 half-precision floating-point.
  xnn_datatype_fp16 = 2,
  /// Quantized 8-bit signed integer with shared per-Value quantization parameters.
  xnn_datatype_qint8 = 3,
  /// Quantized 8-bit unsigned integer with shared per-Value quantization parameters.
  xnn_datatype_quint8 = 4,
  /// Quantized 32-bit signed integer with shared per-Value quantization parameters.
  xnn_datatype_qint32 = 5,
  /// Quantized 8-bit signed integer with shared per-channel quantization parameters.
  xnn_datatype_qcint8 = 6,
  /// Quantized 32-bit signed integer with shared per-channel quantization parameters.
  xnn_datatype_qcint32 = 7,
};

/// Define a tensor-type Value and add it to a Subgraph.
///
/// @param subgraph - a Subgraph object that will own the created Value.
/// @param datatype - type of the tensor elements.
/// @param num_dims - number of dimensions in the shape.
/// @param dims - pointer to an array of @a num_dims shape dimensions. If num_dims is 0, this pointer can be NULL.
///               XNNPACK does not keep any pointers to this array after the function returns.
/// @param data - pointer to static data used for tensor initialization. If the tensor is not statically initialized,
///               this pointer must be is NULL. If non-NULL, the life-time of the static data must exceed the life-time
///               of the Subgraph object, and of any Runtime objects created from the Subgraph.
/// @param external_id - external ID for the Value. The ID must be within the range of reversed Value IDs specified on
///                      the Subgraph creation. If the external ID is XNN_INVALID_VALUE_ID, an internal ID will be
///                      created for the Value.
/// @param flags - binary features of the Value. Supported values are any combination of XNN_VALUE_FLAG_EXTERNAL_INPUT
///                and XNN_VALUE_FLAG_EXTERNAL_OUTPUT.
/// @param id_out - pointer to the variable that will be initialized with the Value ID upon successful return. If a
///                 valid @a external_id was provided, the variable will be initialized with the @a external_id value.
enum xnn_status xnn_define_tensor_value(
  xnn_subgraph_t subgraph,
  enum xnn_datatype datatype,
  size_t num_dims,
  const size_t* dims,
  const void* data,
  uint32_t external_id,
  uint32_t flags,
  uint32_t* id_out);

/// Define a quantized tensor-type Value and add it to a Subgraph.
///
/// @param subgraph - a Subgraph object that will own the created Value.
/// @param datatype - type of the tensor elements.
/// @param zero_point - offset from zero to subtract from the quantized elements in the Value.
/// @param scale - multiplication factor to convert quantized elements to real representation.
/// @param num_dims - number of dimensions in the shape.
/// @param dims - pointer to an array of @a num_dims shape dimensions. If num_dims is 0, this pointer can be NULL.
///               XNNPACK does not keep any pointers to this array after the function returns.
/// @param data - pointer to static data used for tensor initialization. If the tensor is not statically initialized,
///               this pointer must be is NULL. If non-NULL, the life-time of the static data must exceed the life-time
///               of the Subgraph object, and of any Runtime objects created from the Subgraph.
/// @param external_id - external ID for the Value. The ID must be within the range of reversed Value IDs specified on
///                      the Subgraph creation. If the external ID is XNN_INVALID_VALUE_ID, an internal ID will be
///                      created for the Value.
/// @param flags - binary features of the Value. Supported values are any combination of XNN_VALUE_FLAG_EXTERNAL_INPUT
///                and XNN_VALUE_FLAG_EXTERNAL_OUTPUT.
/// @param id_out - pointer to the variable that will be initialized with the Value ID upon successful return. If a
///                 valid @a external_id was provided, the variable will be initialized with the @a external_id value.
enum xnn_status xnn_define_quantized_tensor_value(
  xnn_subgraph_t subgraph,
  enum xnn_datatype datatype,
  int32_t zero_point,
  float scale,
  size_t num_dims,
  const size_t* dims,
  const void* data,
  uint32_t external_id,
  uint32_t flags,
  uint32_t* id_out);

/// Define a channelwise quantized tensor-type Value and add it to a Subgraph.
///
/// @param subgraph - a Subgraph object that will own the created Value.
/// @param datatype - type of the tensor elements.
/// @param scale - per-channel multiplication factors to convert quantized elements to real representation.
/// @param num_dims - number of dimensions in the shape.
/// @param channel_dim - index of the channel dimension in the tensor with per-channel quantization parameters.
///                      Typically this is the first dimension (dimension #0) of the filter tensors in the Convolution,
///                      Deconvolution, and Fully Connected operators and the last dimension of the filter tensors in
///                      the Depthwise Convolution operators.
/// @param dims - pointer to an array of @a num_dims shape dimensions. If num_dims is 0, this pointer can be NULL.
///               XNNPACK does not keep any pointers to this array after the function returns.
/// @param data - pointer to static data used for tensor initialization. If the tensor is not statically initialized,
///               this pointer must be is NULL. If non-NULL, the life-time of the static data must exceed the life-time
///               of the Subgraph object, and of any Runtime objects created from the Subgraph.
/// @param external_id - external ID for the Value. The ID must be within the range of reversed Value IDs specified on
///                      the Subgraph creation. If the external ID is XNN_INVALID_VALUE_ID, an internal ID will be
///                      created for the Value.
/// @param flags - binary features of the Value. Supported values are any combination of XNN_VALUE_FLAG_EXTERNAL_INPUT
///                and XNN_VALUE_FLAG_EXTERNAL_OUTPUT.
/// @param id_out - pointer to the variable that will be initialized with the Value ID upon successful return. If a
///                 valid @a external_id was provided, the variable will be initialized with the @a external_id value.
enum xnn_status xnn_define_channelwise_quantized_tensor_value(
  xnn_subgraph_t subgraph,
  enum xnn_datatype datatype,
  const float* scale,
  size_t num_dims,
  size_t channel_dim,
  const size_t* dims,
  const void* data,
  uint32_t external_id,
  uint32_t flags,
  uint32_t* id_out);

/// Define a Convert Node and add it to a Subgraph.
///
/// @param subgraph - a Subgraph object that will own the created Node.
/// @param input_id - Value ID for the input tensor. The input tensor must be defined in the @a subgraph.
/// @param output_id - Value ID for the output tensor. The output tensor must be defined in the @a subgraph, and its
///                    shape must match the shape of the input tensor.
/// @param flags - binary features of the Convert Node. No supported flags are currently defined.
enum xnn_status xnn_define_convert(
  xnn_subgraph_t subgraph,
  uint32_t input_id,
  uint32_t output_id,
  uint32_t flags);

/// Define a 2D Convolution Node and add it to a Subgraph.
///
/// @param subgraph - a Subgraph object that will own the created Node.
/// @param input_padding_top - implicit zero-padding above 2D input data. Must be 0 if XNN_FLAG_TENSORFLOW_SAME_PADDING
///                            flag is specified.
/// @param input_padding_right - implicit zero-padding to the right of 2D input data. Must be 0 if
///                              XNN_FLAG_TENSORFLOW_SAME_PADDING flag is specified.
/// @param input_padding_bottom - implicit zero-padding below 2D input data. Must be 0 if
///                               XNN_FLAG_TENSORFLOW_SAME_PADDING flag is specified.
/// @param input_padding_left - implicit zero-padding to the left of 2D input data. Must be 0 if
///                             XNN_FLAG_TENSORFLOW_SAME_PADDING flag is specified.
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
/// @param input_id - Value ID for the input tensor. The input tensor must be a 4D tensor defined in the @a subgraph
///                   with [N, IH, IW, groups * group_input_channels] dimensions
/// @param filter_id - Value ID for the filter tensor. The filter tensor must ge a 4D tensor defined in the @a subgraph
///                    with [groups * group_output_channels, kernel_height, kernel_width, group_input_channels]
///                    dimensions.
/// @param bias_id - Value ID for the bias tensor, or XNN_INVALID_VALUE_ID for a 2D Convolution Node without a bias. If
///                  present, the bias tensor must be a 1D tensor defined in the @a subgraph with [groups *
///                  group_output_channels] dimensions.
/// @param output_id - Value ID for the output tensor. The output tensor must be a 4D tensor defined in the @a subgraph
///                    with [N, OH, OW, groups * group_output_channels] dimensions.
/// @param flags - binary features of the 2D Convolution Node. The only currently supported values is
///                XNN_FLAG_TENSORFLOW_SAME_PADDING.
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

/// Define a 2D Deconvolution (Transposed Convolution) Node and add it to a Subgraph.
///
/// @param subgraph - a Subgraph object that will own the created Node.
/// @param padding_top - implicit padding above 2D output data.
/// @param padding_right - implicit padding to the right of 2D output data.
/// @param padding_bottom - implicit padding below 2D output data.
/// @param padding_left - implicit padding to the left of 2D output data.
/// @param adjustment_height - additional elements in the bottom of the 2D output data.
/// @param adjustment_width - additional elements to the right of the 2D output data.
/// @param kernel_height - kernel (filter) height.
/// @param kernel_width - kernel (filter) width.
/// @param upsampling_height - height of upsampling region for deconvolution input (deconvolution height stride).
/// @param upsampling_width - width of upsampling region for deconvolution input (deconvolution width stride).
/// @param dilation_height - dilation of kernel elements along the height dimension.
/// @param dilation_width - dilation of kernel elements along the width dimension.
/// @param groups - number of convolution groups.
/// @param group_input_channels - number of input channels per group.
/// @param group_output_channels - number of output channels per group.
/// @param output_min - lower bound for clipping output values.
/// @param output_max - upper bound for clipping output values.
/// @param input_id - Value ID for the input tensor. The input tensor must be a 4D tensor defined in the @a subgraph
///                   with [N, IH, IW, groups * group_input_channels] dimensions
/// @param filter_id - Value ID for the filter tensor. The filter tensor must ge a 4D tensor defined in the @a subgraph
///                    with [groups * group_output_channels, kernel_height, kernel_width, group_input_channels]
///                    dimensions.
/// @param bias_id - Value ID for the bias tensor, or XNN_INVALID_VALUE_ID for a 2D Convolution Node without a bias. If
///                  present, the bias tensor must be a 1D tensor defined in the @a subgraph with
///                  [groups * group_output_channels] dimensions.
/// @param output_id - Value ID for the output tensor. The output tensor must be a 4D tensor defined in the @a subgraph
///                    with [N, OH, OW, groups * group_output_channels] dimensions.
/// @param flags - binary features of the 2D Deconvolution Node. No supported flags are currently defined.
enum xnn_status xnn_define_deconvolution_2d(
  xnn_subgraph_t subgraph,
  uint32_t padding_top,
  uint32_t padding_right,
  uint32_t padding_bottom,
  uint32_t padding_left,
  uint32_t adjustment_height,
  uint32_t adjustment_width,
  uint32_t kernel_height,
  uint32_t kernel_width,
  uint32_t upsampling_height,
  uint32_t upsampling_width,
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

/// Define a 2D Depthwise Convolution Node and add it to a Subgraph.
///
/// @param subgraph - a Subgraph object that will own the created Node.
/// @param input_padding_top - implicit zero-padding above 2D input data. Must be 0 if XNN_FLAG_TENSORFLOW_SAME_PADDING
///                            flag is specified.
/// @param input_padding_right - implicit zero-padding to the right of 2D input data. Must be 0 if
///                              XNN_FLAG_TENSORFLOW_SAME_PADDING flag is specified.
/// @param input_padding_bottom - implicit zero-padding below 2D input data. Must be 0 if
///                               XNN_FLAG_TENSORFLOW_SAME_PADDING flag is specified.
/// @param input_padding_left - implicit zero-padding to the left of 2D input data. Must be 0 if
///                             XNN_FLAG_TENSORFLOW_SAME_PADDING flag is specified.
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
/// @param input_id - Value ID for the input tensor. The input tensor must be a 4D tensor defined in the @a subgraph
///                   with [N, IH, IW, input_channels] dimensions
/// @param filter_id - Value ID for the filter tensor. The filter tensor must ge a 4D tensor defined in the @a subgraph
///                    with [1, kernel_height, kernel_width, input_channels * depth_multiplier] dimensions.
/// @param bias_id - Value ID for the bias tensor, or XNN_INVALID_VALUE_ID for a 2D Depthwise Convolution Node without
///                  a bias. If present, the bias tensor must be a 1D tensor defined in the @a subgraph with
///                  [input_channels * depth_multiplier] dimensions.
/// @param output_id - Value ID for the output tensor. The output tensor must be a 4D tensor defined in the @a subgraph
///                    with [N, OH, OW, input_channels * depth_multiplier] dimensions.
/// @param flags - binary features of the 2D Depthwise Convolution Node. The only currently supported values is
///                XNN_FLAG_TENSORFLOW_SAME_PADDING.
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

/// Define a Depth To Space Node and add it to a Subgraph.
///
/// The Depth To Space Node rearranges data from depth into blocks of spatial data (a reverse transform to
/// Space To Depth). For a given input pixel, an output square of pixels with side @a block_size is formed from values
/// in the corresponding number of its channels. The output depth is therefore @a block_size x @a block_size times
/// smaller than that of the input.
///
/// @param subgraph - a Subgraph object that will own the created Node.
/// @param input_id - Value ID for the input tensor. The input tensor must be a 4D tensor defined in the @a subgraph
///                   with [N, IH, IW, OC * block_size * block_size] dimensions.
/// @param output_id - Value ID for the output tensor. The output tensor must be a 4D tensor defined in the @a subgraph
///                    with [N, IH * block_size, IW * block_size, OC] dimensions.
/// @param block_size - the size of the spatial block.
/// @param flags - binary features of the input_channels Node. No supported flags are currently defined.
enum xnn_status xnn_define_depth_to_space(
  xnn_subgraph_t subgraph,
  uint32_t input_id,
  uint32_t output_id,
  uint32_t block_size,
  uint32_t flags);

/// Define a 1D Global Average Pooling Node and add it to a Subgraph.
///
/// @param subgraph - a Subgraph object that will own the created Node.
/// @param output_min - lower bound for clipping output values.
/// @param output_max - upper bound for clipping output values.
/// @param input_id - Value ID for the input tensor. The input tensor must be a dense tensor with 2 or more dimensions
///                   defined in the @a subgraph. Averaging is performed across the second-innermost dimension.
/// @param output_id - Value ID for the output tensor. The output tensor must be a dense tensor with 2 or more
///                    dimensions defined in the @a subgraph.
/// @param flags - binary features of the 1D Global Average Pooling Node. No supported flags are currently defined.
enum xnn_status xnn_define_global_average_pooling_1d(
  xnn_subgraph_t subgraph,
  float output_min,
  float output_max,
  uint32_t input_id,
  uint32_t output_id,
  uint32_t flags);

/// Define a 2D Global Average Pooling Node and add it to a Subgraph.
///
/// @param subgraph - a Subgraph object that will own the created Node.
/// @param output_min - lower bound for clipping output values.
/// @param output_max - upper bound for clipping output values.
/// @param input_id - Value ID for the input tensor. The input tensor must be a dense tensor with 3 or more dimensions
///                   defined in the @a subgraph. Averaging is performed across the second- and third-innermost
///                   dimensions.
/// @param output_id - Value ID for the output tensor. The output tensor must be a dense tensor with 3 or more
///                    dimensions defined in the @a subgraph.
/// @param flags - binary features of the 2D Global Average Pooling Node. No supported flags are currently defined.
enum xnn_status xnn_define_global_average_pooling_2d(
  xnn_subgraph_t subgraph,
  float output_min,
  float output_max,
  uint32_t input_id,
  uint32_t output_id,
  uint32_t flags);

/// Define a 2D Average Pooling Node and add it to a Subgraph.
///
/// @param subgraph - a Subgraph object that will own the created Node.
/// @param input_padding_top - implicit zero-padding above 2D input data. Must be 0 if XNN_FLAG_TENSORFLOW_SAME_PADDING
///                            flag is specified.
/// @param input_padding_right - implicit zero-padding to the right of 2D input data. Must be 0 if
///                              XNN_FLAG_TENSORFLOW_SAME_PADDING flag is specified.
/// @param input_padding_bottom - implicit zero-padding below 2D input data. Must be 0 if
///                               XNN_FLAG_TENSORFLOW_SAME_PADDING flag is specified.
/// @param input_padding_left - implicit zero-padding to the left of 2D input data. Must be 0 if
///                             XNN_FLAG_TENSORFLOW_SAME_PADDING flag is specified.
/// @param pooling_height - pooling (kernel) height.
/// @param pooling_width - pooling (kernel) width.
/// @param stride_height - displacing of the pooling window in the vertical dimension of the input pixels corresponding
///                        to vertically adjacent output pixels.
/// @param stride_width - displacing of the pooling window in the horizontal dimension of the input pixels corresponding
///                        to horizontally adjacent output pixels.
/// @param output_min - lower bound for clipping output values.
/// @param output_max - upper bound for clipping output values.
/// @param input_id - Value ID for the input tensor. The input tensor must be a 4D tensor defined in the @a subgraph
///                   with [N, IH, IW, channels] dimensions
/// @param output_id - Value ID for the output tensor. The output tensor must be a 4D tensor defined in the @a subgraph
///                    with [N, OH, OW, channels] dimensions.
/// @param flags - binary features of the 2D Average Pooling Node. The only currently supported values is
///                XNN_FLAG_TENSORFLOW_SAME_PADDING.
enum xnn_status xnn_define_average_pooling_2d(
  xnn_subgraph_t subgraph,
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
  uint32_t input_id,
  uint32_t output_id,
  uint32_t flags);

/// Define a Fully Connected Node and add it to a Subgraph.
///
/// @param subgraph - a Subgraph object that will own the created Node.
/// @param output_min - lower bound for clipping output values.
/// @param output_max - upper bound for clipping output values.
/// @param input_id - Value ID for the input tensor. The input tensor must be an N-dimensional tensor defined in the
///                   @a subgraph. If XNN_FLAG_TENSORFLOW_RESHAPE_2D is not specified, the input tensor must be at least
///                   1D and its last dimension must match the last dimension of the filter tensor. In particular, if
///                   input is a 2D tensor, it must have [batch_size, input_channels] dimensions.
///                   If XNN_FLAG_TENSORFLOW_RESHAPE_2D is specified, the number of elements in the input tensor must be
///                   divisible by the input_channels. The tensor will be first flattened into a 1D tensor of
///                   [num_input_elements] dimensions, then reshaped into a 2D tensor of
///                   [num_input_elements / input_channels, input_channels] dimensions where num_input_elements is the
///                   total number of elements in the input tensor.
/// @param filter_id - Value ID for the filter tensor. The filter tensor must a 2D tensor defined in the @a subgraph.
///                    If the XNN_FLAG_TRANSPOSE_WEIGHTS flag is not specified, the filter tensor must have
///                    [output_channels, input_channels] dimensions. If the XNN_FLAG_TRANSPOSE_WEIGHTS flag is
///                    specified, the filter tensor must have [input_channels, output_channels] dimensions.
/// @param bias_id - Value ID for the bias tensor, or XNN_INVALID_VALUE_ID for a Fully Connected Node without a bias.
///                  If present, the bias tensor must be a 1D tensor defined in the @a subgraph with [output_channels]
///                  dimensions.
/// @param output_id - Value ID for the output tensor. The output tensor must be defined in the @a subgraph.
///                    If XNN_FLAG_TENSORFLOW_RESHAPE_2D is not specified, the output tensor must have the same
///                    dimensionality as the input tensor, all its dimensions but the last one must match the
///                    corresponding dimensions of the input tensor, and the last dimensions of the output tensor must
///                    match the first dimension of the filter tensor. In particular, if input is a 2D tensor, output
///                    must be a 2D tensor of [batch_size, output_channels] dimensions.
///                    If XNN_FLAG_TENSORFLOW_RESHAPE_2D is specified, output must be a 2D tensor of
///                    [num_input_elements / input_channels, output_channels] dimensions where num_input_elements is the
///                    total number of elements in the input tensor.
/// @param flags - binary features of the Fully Connected Node. The only currently supported values are
///                XNN_FLAG_TENSORFLOW_RESHAPE_2D and XNN_FLAG_TRANSPOSE_WEIGHTS.
enum xnn_status xnn_define_fully_connected(
  xnn_subgraph_t subgraph,
  float output_min,
  float output_max,
  uint32_t input_id,
  uint32_t filter_id,
  uint32_t bias_id,
  uint32_t output_id,
  uint32_t flags);

/// Define a 2D Max Pooling Node and add it to a Subgraph.
///
/// @param subgraph - a Subgraph object that will own the created Node.
/// @param input_padding_top - implicit zero-padding above 2D input data. Must be 0 if XNN_FLAG_TENSORFLOW_SAME_PADDING
///                            flag is specified.
/// @param input_padding_right - implicit zero-padding to the right of 2D input data. Must be 0 if
///                              XNN_FLAG_TENSORFLOW_SAME_PADDING flag is specified.
/// @param input_padding_bottom - implicit zero-padding below 2D input data. Must be 0 if
///                               XNN_FLAG_TENSORFLOW_SAME_PADDING flag is specified.
/// @param input_padding_left - implicit zero-padding to the left of 2D input data. Must be 0 if
///                             XNN_FLAG_TENSORFLOW_SAME_PADDING flag is specified.
/// @param pooling_height - pooling (kernel) height.
/// @param pooling_width - pooling (kernel) width.
/// @param stride_height - displacing of the pooling window in the vertical dimension of the input pixels corresponding
///                        to vertically adjacent output pixels.
/// @param stride_width - displacing of the pooling window in the horizontal dimension of the input pixels corresponding
///                        to horizontally adjacent output pixels.
/// @param dilation_height - dilation of pooling elements along the height dimension.
/// @param dilation_width - dilation of pooling elements along the width dimension.
/// @param output_min - lower bound for clipping output values.
/// @param output_max - upper bound for clipping output values.
/// @param input_id - Value ID for the input tensor. The input tensor must be a 4D tensor defined in the @a subgraph
///                   with [N, IH, IW, channels] dimensions
/// @param output_id - Value ID for the output tensor. The output tensor must be a 4D tensor defined in the @a subgraph
///                    with [N, OH, OW, channels] dimensions.
/// @param flags - binary features of the 2D Max Pooling Node. The only currently supported values is
///                XNN_FLAG_TENSORFLOW_SAME_PADDING.
enum xnn_status xnn_define_max_pooling_2d(
  xnn_subgraph_t subgraph,
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
  uint32_t input_id,
  uint32_t output_id,
  uint32_t flags);

/// Define a 2D ArgMax Pooling Node and add it to a Subgraph.
///
/// @param subgraph - a Subgraph object that will own the created Node.
/// @param input_padding_top - implicit zero-padding above 2D input data.
/// @param input_padding_right - implicit zero-padding to the right of 2D input data.
/// @param input_padding_bottom - implicit zero-padding below 2D input data.
/// @param input_padding_left - implicit zero-padding to the left of 2D input data.
/// @param pooling_height - pooling (kernel) height. Vertical stride between pooling regions match this value.
/// @param pooling_width - pooling (kernel) width. Horizontal stride between pooling regions match this value.
/// @param input_id - Value ID for the input tensor. The input tensor must be a 4D tensor defined in the @a subgraph
///                   with [N, IH, IW, channels] dimensions
/// @param output_value_id - Value ID for the output tensor with the maximum values in the pools. The output tensor must
///                          be a 4D tensor defined in the @a subgraph with [N, OH, OW, channels] dimensions.
/// @param output_index_id - Value ID for the output tensor with the indexes of the maximum values in the pools. The
///                          output tensor must be a 4D tensor defined in the @a subgraph with [N, OH, OW, channels]
///                          dimensions.
/// @param flags - binary features of the 2D ArgMax Pooling Node. No supported flags are currently defined.
enum xnn_status xnn_define_argmax_pooling_2d(
  xnn_subgraph_t subgraph,
  uint32_t input_padding_top,
  uint32_t input_padding_right,
  uint32_t input_padding_bottom,
  uint32_t input_padding_left,
  uint32_t pooling_height,
  uint32_t pooling_width,
  uint32_t input_id,
  uint32_t output_value_id,
  uint32_t output_index_id,
  uint32_t flags);

/// Define a 2D UnPooling Node and add it to a Subgraph.
///
/// @param subgraph - a Subgraph object that will own the created Node.
/// @param padding_top - implicit padding above 2D output data.
/// @param padding_right - implicit padding to the right of 2D output data.
/// @param padding_bottom - implicit padding below 2D output data.
/// @param padding_left - implicit padding to the left of 2D output data.
/// @param pooling_height - height of the pooling window.
/// @param pooling_width - width of the pooling window.
/// @param input_value_id - Value ID for the input tensor with the max-pooling values to invert. The input value tensor
///                         must be a 4D tensor defined in the @a subgraph with [N, IH, IW, channels] dimensions.
/// @param input_index_id - Value ID for the input tensor with the indices of the per-pool maximum values produced by
///                         a 2D UnPooling Node. The input tensor must be a 4D tensor defined in the @a subgraph with
///                         [N, IH, IW, channels] dimensions.
/// @param output_id - Value ID for the output tensor. The output tensor must be a 4D tensor defined in the @a subgraph
///                    with [N, OH, OW, channels] dimensions.
/// @param flags - binary features of the 2D UnPooling Node. No supported flags are currently defined.
enum xnn_status xnn_define_unpooling_2d(
  xnn_subgraph_t subgraph,
  uint32_t padding_top,
  uint32_t padding_right,
  uint32_t padding_bottom,
  uint32_t padding_left,
  uint32_t pooling_height,
  uint32_t pooling_width,
  uint32_t input_value_id,
  uint32_t input_index_id,
  uint32_t output_id,
  uint32_t flags);

/// Define a 2-Input Add Node and add it to a Subgraph.
///
/// The 2-Input Add Node computes elementwise addition of two tensor inputs with numpy broadcasting rules.
///
/// @param subgraph - a Subgraph object that will own the created Node.
/// @param output_min - lower bound for clipping output values.
/// @param output_max - upper bound for clipping output values.
/// @param input1_id - Value ID for the first input tensor. The input tensor must be an N-dimensional tensor defined in
///                    the @a subgraph with each dimension either equal to the corresponding dimension of the second
///                    input, or equal to 1. In the latter case, the elements of the input tensor are broadcasted along
///                    that dimension.
/// @param input2_id - Value ID for the second input tensor. The input tensor must be an M-dimensional tensor defined in
///                    the @a subgraph with each dimension either equal to the corresponding dimension of the first
///                    input, or equal to 1. In the latter case, the elements of the input tensor are broadcasted along
///                    that dimension.
/// @param output_id - Value ID for the output tensor. The output tensor must be a max(N,M)-dimensional tensor defined
///                    in the @a subgraph with each dimension equal to the maximum between the corresponding dimension
///                    of the two inputs.
/// @param flags - binary features of the Add Node. No supported flags are currently defined.
enum xnn_status xnn_define_add2(
  xnn_subgraph_t subgraph,
  float output_min,
  float output_max,
  uint32_t input1_id,
  uint32_t input2_id,
  uint32_t output_id,
  uint32_t flags);

/// Define a 2-Input Multiply Node and add it to a Subgraph.
///
/// The 2-Input Multiply Node computes elementwise multiplication of two tensor inputs with numpy broadcasting rules.
///
/// @param subgraph - a Subgraph object that will own the created Node.
/// @param output_min - lower bound for clipping output values.
/// @param output_max - upper bound for clipping output values.
/// @param input1_id - Value ID for the first input tensor. The input tensor must be an N-dimensional tensor defined in
///                    the @a subgraph with each dimension either equal to the corresponding dimension of the second
///                    input, or equal to 1. In the latter case, the elements of the input tensor are broadcasted along
///                    that dimension.
/// @param input2_id - Value ID for the second input tensor. The input tensor must be an M-dimensional tensor defined in
///                    the @a subgraph with each dimension either equal to the corresponding dimension of the first
///                    input, or equal to 1. In the latter case, the elements of the input tensor are broadcasted along
///                    that dimension.
/// @param output_id - Value ID for the output tensor. The output tensor must be a max(N,M)-dimensional tensor defined
///                    in the @a subgraph with each dimension equal to the maximum between the corresponding dimension
///                    of the two inputs.
/// @param flags - binary features of the Multiply Node. No supported flags are currently defined.
enum xnn_status xnn_define_multiply2(
  xnn_subgraph_t subgraph,
  float output_min,
  float output_max,
  uint32_t input1_id,
  uint32_t input2_id,
  uint32_t output_id,
  uint32_t flags);

/// Define a Subtract Node and add it to a Subgraph.
///
/// The Subtract Node computes elementwise subtraction of two tensor inputs with numpy broadcasting rules.
///
/// @param subgraph - a Subgraph object that will own the created Node.
/// @param output_min - lower bound for clipping output values.
/// @param output_max - upper bound for clipping output values.
/// @param input1_id - Value ID for the first input tensor. The input tensor must be an N-dimensional tensor defined in
///                    the @a subgraph with each dimension either equal to the corresponding dimension of the second
///                    input, or equal to 1. In the latter case, the elements of the input tensor are broadcasted along
///                    that dimension.
/// @param input2_id - Value ID for the second input tensor. The input tensor must be an M-dimensional tensor defined in
///                    the @a subgraph with each dimension either equal to the corresponding dimension of the first
///                    input, or equal to 1. In the latter case, the elements of the input tensor are broadcasted along
///                    that dimension.
/// @param output_id - Value ID for the output tensor. The output tensor must be a max(N,M)-dimensional tensor defined
///                    in the @a subgraph with each dimension equal to the maximum between the corresponding dimension
///                    of the two inputs.
/// @param flags - binary features of the Subtract Node. No supported flags are currently defined.
enum xnn_status xnn_define_subtract(
  xnn_subgraph_t subgraph,
  float output_min,
  float output_max,
  uint32_t input1_id,
  uint32_t input2_id,
  uint32_t output_id,
  uint32_t flags);

/// Define a Divide Node and add it to a Subgraph.
///
/// The Divide Node computes elementwise division of two tensor inputs with numpy broadcasting rules.
///
/// @param subgraph - a Subgraph object that will own the created Node.
/// @param output_min - lower bound for clipping output values.
/// @param output_max - upper bound for clipping output values.
/// @param input1_id - Value ID for the first input tensor. The input tensor must be an N-dimensional tensor defined in
///                    the @a subgraph with each dimension either equal to the corresponding dimension of the second
///                    input, or equal to 1. In the latter case, the elements of the input tensor are broadcasted along
///                    that dimension.
/// @param input2_id - Value ID for the second input tensor. The input tensor must be an M-dimensional tensor defined in
///                    the @a subgraph with each dimension either equal to the corresponding dimension of the first
///                    input, or equal to 1. In the latter case, the elements of the input tensor are broadcasted along
///                    that dimension.
/// @param output_id - Value ID for the output tensor. The output tensor must be a max(N,M)-dimensional tensor defined
///                    in the @a subgraph with each dimension equal to the maximum between the corresponding dimension
///                    of the two inputs.
/// @param flags - binary features of the Divide Node. No supported flags are currently defined.
enum xnn_status xnn_define_divide(
  xnn_subgraph_t subgraph,
  float output_min,
  float output_max,
  uint32_t input1_id,
  uint32_t input2_id,
  uint32_t output_id,
  uint32_t flags);

/// Define a 2-Input Maximum Node and add it to a Subgraph.
///
/// The 2-Input Maximum Node computes elementwise maximum of two tensor inputs with numpy broadcasting rules.
///
/// @param subgraph - a Subgraph object that will own the created Node.
/// @param input1_id - Value ID for the first input tensor. The input tensor must be an N-dimensional tensor defined in
///                    the @a subgraph with each dimension either equal to the corresponding dimension of the second
///                    input, or equal to 1. In the latter case, the elements of the input tensor are broadcasted along
///                    that dimension.
/// @param input2_id - Value ID for the second input tensor. The input tensor must be an M-dimensional tensor defined in
///                    the @a subgraph with each dimension either equal to the corresponding dimension of the first
///                    input, or equal to 1. In the latter case, the elements of the input tensor are broadcasted along
///                    that dimension.
/// @param output_id - Value ID for the output tensor. The output tensor must be a max(N,M)-dimensional tensor defined
///                    in the @a subgraph with each dimension equal to the maximum between the corresponding dimension
///                    of the two inputs.
/// @param flags - binary features of the Maximum Node. No supported flags are currently defined.
enum xnn_status xnn_define_maximum2(
  xnn_subgraph_t subgraph,
  uint32_t input1_id,
  uint32_t input2_id,
  uint32_t output_id,
  uint32_t flags);

/// Define a 2-Input Minimum Node and add it to a Subgraph.
///
/// The 2-Input Minimum Node computes elementwise minimum of two tensor inputs with numpy broadcasting rules.
///
/// @param subgraph - a Subgraph object that will own the created Node.
/// @param input1_id - Value ID for the first input tensor. The input tensor must be an N-dimensional tensor defined in
///                    the @a subgraph with each dimension either equal to the corresponding dimension of the second
///                    input, or equal to 1. In the latter case, the elements of the input tensor are broadcasted along
///                    that dimension.
/// @param input2_id - Value ID for the second input tensor. The input tensor must be an M-dimensional tensor defined in
///                    the @a subgraph with each dimension either equal to the corresponding dimension of the first
///                    input, or equal to 1. In the latter case, the elements of the input tensor are broadcasted along
///                    that dimension.
/// @param output_id - Value ID for the output tensor. The output tensor must be a max(N,M)-dimensional tensor defined
///                    in the @a subgraph with each dimension equal to the maximum between the corresponding dimension
///                    of the two inputs.
/// @param flags - binary features of the Minimum Node. No supported flags are currently defined.
enum xnn_status xnn_define_minimum2(
  xnn_subgraph_t subgraph,
  uint32_t input1_id,
  uint32_t input2_id,
  uint32_t output_id,
  uint32_t flags);

/// Define a Squared Difference Node and add it to a Subgraph.
///
/// The Squared Difference Node computes elementwise squared difference of two tensor inputs with numpy broadcasting
/// rules.
///
/// @param subgraph - a Subgraph object that will own the created Node.
/// @param input1_id - Value ID for the first input tensor. The input tensor must be an N-dimensional tensor defined in
///                    the @a subgraph with each dimension either equal to the corresponding dimension of the second
///                    input, or equal to 1. In the latter case, the elements of the input tensor are broadcasted along
///                    that dimension.
/// @param input2_id - Value ID for the second input tensor. The input tensor must be an M-dimensional tensor defined in
///                    the @a subgraph with each dimension either equal to the corresponding dimension of the first
///                    input, or equal to 1. In the latter case, the elements of the input tensor are broadcasted along
///                    that dimension.
/// @param output_id - Value ID for the output tensor. The output tensor must be a max(N,M)-dimensional tensor defined
///                    in the @a subgraph with each dimension equal to the maximum between the corresponding dimension
///                    of the two inputs.
/// @param flags - binary features of the Squared Difference Node. No supported flags are currently defined.
enum xnn_status xnn_define_squared_difference(
  xnn_subgraph_t subgraph,
  uint32_t input1_id,
  uint32_t input2_id,
  uint32_t output_id,
  uint32_t flags);

/// Define a Constant Pad Node with static padding specification and add it to a Subgraph.
///
/// @param subgraph - a Subgraph object that will own the created Node.
/// @param pre_paddings - number of padding elements to insert before input elements for every dimension. This array
///                       must have as many elements as the number of dimensions in the input tensor.
/// @param post_paddings - number of padding elements to insert after input elements for every dimension. This array
///                        must have as many elements as the number of dimensions in the input tensor.
/// @param padding_value - constant value used to initialize padding elements.
/// @param input_id - Value ID for the input tensor. The input tensor must be defined in the @a subgraph.
/// @param output_id - Value ID for the output tensor. The output tensor must be defined in the @a subgraph, and its
///                    shape must match the shape of the input tensor with padding.
/// @param flags - binary features of the Constant Pad Node. No supported flags are currently defined.
enum xnn_status xnn_define_static_constant_pad(
  xnn_subgraph_t subgraph,
  const size_t* pre_paddings,
  const size_t* post_paddings,
  float padding_value,
  uint32_t input_id,
  uint32_t output_id,
  uint32_t flags);

/// Define a 2-Input Concatenate Node and add it to a Subgraph.
///
/// The 2-Input Concatenate Node concatenates two tensors along a specified axis.
///
/// @param subgraph - a Subgraph object that will own the created Node.
/// @param axis - the axis to concatenate the two input tensors along
/// @param input1_id - Value ID for the first input tensor. The input tensor must be an N-dimensional tensor defined in
///                    the @a subgraph with each dimension, except the axis, equal to the corresponding dimension of the
///                    second input.
/// @param input2_id - Value ID for the second input tensor. The input tensor must be an N-dimensional tensor defined in
///                    the @a subgraph with each dimension, except the axis, equal to the corresponding dimension of the
///                    first input.
/// @param output_id - Value ID for the output tensor. The output tensor must be a N-dimensional tensor defined
///                    in the @a subgraph with each dimension equal to the dimension of both inputs, except the axis
///                    dimension, where it is the sum of the corresponding dimensions of both inputs.
/// @param flags - binary features of the Concatenate Node. No supported flags are currently defined.
enum xnn_status xnn_define_concatenate2(
  xnn_subgraph_t subgraph,
  size_t axis,
  uint32_t input1_id,
  uint32_t input2_id,
  uint32_t output_id,
  uint32_t flags);

/// Define a 3-Input Concatenate Node and add it to a Subgraph.
///
/// The 3-Input Concatenate Node concatenates three tensors along a specified axis.
///
/// @param subgraph - a Subgraph object that will own the created Node.
/// @param axis - the axis to concatenate the three input tensors along
/// @param input1_id - Value ID for the first input tensor. The input tensor must be an N-dimensional tensor defined in
///                    the @a subgraph with each dimension, except the axis, equal to the corresponding dimension of the
///                    other inputs.
/// @param input2_id - Value ID for the second input tensor. The input tensor must be an N-dimensional tensor defined in
///                    the @a subgraph with each dimension, except the axis, equal to the corresponding dimension of the
///                    other inputs.
/// @param input3_id - Value ID for the third input tensor. The input tensor must be an N-dimensional tensor defined in
///                    the @a subgraph with each dimension, except the axis, equal to the corresponding dimension of the
///                    other inputs.
/// @param output_id - Value ID for the output tensor. The output tensor must be a N-dimensional tensor defined
///                    in the @a subgraph with each dimension equal to the dimension of all inputs, except the axis
///                    dimension, where it is the sum of the corresponding dimensions of all inputs.
/// @param flags - binary features of the Concatenate Node. No supported flags are currently defined.
enum xnn_status xnn_define_concatenate3(
  xnn_subgraph_t subgraph,
  size_t axis,
  uint32_t input1_id,
  uint32_t input2_id,
  uint32_t input3_id,
  uint32_t output_id,
  uint32_t flags);

/// Define a 4-Input Concatenate Node and add it to a Subgraph.
///
/// The 4-Input Concatenate Node concatenates four tensors along a specified axis.
///
/// @param subgraph - a Subgraph object that will own the created Node.
/// @param axis - the axis to concatenate the four input tensors along
/// @param input1_id - Value ID for the first input tensor. The input tensor must be an N-dimensional tensor defined in
///                    the @a subgraph with each dimension, except the axis, equal to the corresponding dimension of the
///                    other inputs.
/// @param input2_id - Value ID for the second input tensor. The input tensor must be an N-dimensional tensor defined in
///                    the @a subgraph with each dimension, except the axis, equal to the corresponding dimension of the
///                    other inputs.
/// @param input3_id - Value ID for the third input tensor. The input tensor must be an N-dimensional tensor defined in
///                    the @a subgraph with each dimension, except the axis, equal to the corresponding dimension of the
///                    other inputs.
/// @param input4_id - Value ID for the fourth input tensor. The input tensor must be an N-dimensional tensor defined in
///                    the @a subgraph with each dimension, except the axis, equal to the corresponding dimension of the
///                    other inputs.
/// @param output_id - Value ID for the output tensor. The output tensor must be a N-dimensional tensor defined
///                    in the @a subgraph with each dimension equal to the dimension of all inputs, except the axis
///                    dimension, where it is the sum of the corresponding dimensions of all inputs.
/// @param flags - binary features of the Concatenate Node. No supported flags are currently defined.
enum xnn_status xnn_define_concatenate4(
  xnn_subgraph_t subgraph,
  size_t axis,
  uint32_t input1_id,
  uint32_t input2_id,
  uint32_t input3_id,
  uint32_t input4_id,
  uint32_t output_id,
  uint32_t flags);

/// Define a Copy Node and add it to a Subgraph.
///
/// The Copy Node copies an input tensor to an output tensor.
///
/// @param subgraph - a Subgraph object that will own the created Node.
/// @param input_id - Value ID for the first input tensor. The input tensor must be defined in the @a subgraph.
/// @param output_id - Value ID for the output tensor. The output tensor must be defined in the @a subgraph, and its
///                    shape must match the shape of the input tensor.
/// @param flags - binary features of the Copy Node. No supported flags are currently defined.
enum xnn_status xnn_define_copy(
  xnn_subgraph_t subgraph,
  uint32_t input_id,
  uint32_t output_id,
  uint32_t flags);

/// Define a 2-Output Split Node and add it to a Subgraph.
///
/// The 2-Output Split Node splits an input tensor into two output tensors along a specified axis evenly.
///
/// @param subgraph - a Subgraph object that will own the created Node.
/// @param split_dim - the dimension to split the input tensor along
/// @param input_id - Value ID for the input tensor. The input tensor must be an N-dimensional tensor defined in the @a
///                   subgraph.
/// @param output1_id - Value ID for the first output tensor. The output tensor must be an N-dimensional tensor defined
///                     in the @a subgraph with each dimension, except the axis, equal to the corresponding dimension
///                     of the second output. The split_dim dimension is half of the input's split_dim.
/// @param output2_id - Value ID for the second output tensor. The output tensor must be an N-dimensional tensor
///                     defined in the @a subgraph with each dimension, except the axis, equal to the corresponding
///                     dimension of the first output. The split_dim dimension is half of the input's split_dim.
/// @param flags - binary features of the Split Node. No supported flags are currently defined.
enum xnn_status xnn_define_even_split2(
  xnn_subgraph_t subgraph,
  size_t split_dim,
  uint32_t input_id,
  uint32_t output1_id,
  uint32_t output2_id,
  uint32_t flags);

/// Define a 3-Output Split Node and add it to a Subgraph.
///
/// The 3-Output Split Node splits an input tensor into three output tensors along a specified axis evenly.
///
/// @param subgraph - a Subgraph object that will own the created Node.
/// @param split_dim - the dimension to split the input tensor along
/// @param input_id - Value ID for the input tensor. The input tensor must be an N-dimensional tensor defined in the @a
///                   subgraph.
/// @param output1_id - Value ID for the first output tensor. The output tensor must be an N-dimensional tensor defined
///                     in the @a subgraph with each dimension, except the axis, equal to the corresponding dimension
///                     of the second and third output. The split_dim dimension is one third of the input's split_dim.
/// @param output2_id - Value ID for the second output tensor. The output tensor must be an N-dimensional tensor
///                     defined in the @a subgraph with each dimension, except the axis, equal to the corresponding
///                     dimension of the first and third output. The split_dim dimension is one third of the input's
///                     split_dim.
/// @param output3_id - Value ID for the third output tensor. The output tensor must be an N-dimensional tensor
///                     defined in the @a subgraph with each dimension, except the axis, equal to the corresponding
///                     dimension of the second and third output. The split_dim dimension is one third of the input's
///                     split_dim.
/// @param flags - binary features of the Split Node. No supported flags are currently defined.
enum xnn_status xnn_define_even_split3(
  xnn_subgraph_t subgraph,
  size_t split_dim,
  uint32_t input_id,
  uint32_t output1_id,
  uint32_t output2_id,
  uint32_t output3_id,
  uint32_t flags);

/// Define a 4-Output Split Node and add it to a Subgraph.
///
/// The 4-Output Split Node splits an input tensor into four output tensors along a specified axis evenly.
///
/// @param subgraph - a Subgraph object that will own the created Node.
/// @param split_dim - the dimension to split the input tensor along
/// @param input_id - Value ID for the input tensor. The input tensor must be an N-dimensional tensor defined in the @a
///                   subgraph.
/// @param output1_id - Value ID for the first output tensor. The output tensor must be an N-dimensional tensor defined
///                     in the @a subgraph with each dimension, except the axis, equal to the corresponding dimension
///                     of the other output tensors. The split_dim dimension is one fourth of the input's split_dim.
/// @param output2_id - Value ID for the second output tensor. The output tensor must be an N-dimensional tensor
///                     defined in the @a subgraph with each dimension, except the axis, equal to the corresponding
///                     dimension of the other output tensors. The split_dim dimension is one fourth of the input's
///                     split_dim.
/// @param output3_id - Value ID for the third output tensor. The output tensor must be an N-dimensional tensor
///                     defined in the @a subgraph with each dimension, except the axis, equal to the corresponding
///                     dimension of the other output tensors. The split_dim dimension is one fourth of the input's
///                     split_dim.
/// @param output4_id - Value ID for the fourth output tensor. The output tensor must be an N-dimensional tensor
///                     defined in the @a subgraph with each dimension, except the axis, equal to the corresponding
///                     dimension of the other output tensors. The split_dim dimension is one fourth of the input's
///                     split_dim.
/// @param flags - binary features of the Split Node. No supported flags are currently defined.
enum xnn_status xnn_define_even_split4(
  xnn_subgraph_t subgraph,
  size_t split_dim,
  uint32_t input_id,
  uint32_t output1_id,
  uint32_t output2_id,
  uint32_t output3_id,
  uint32_t output4_id,
  uint32_t flags);

/// Define a Reshape Node with static shape specification and add it to a Subgraph.
///
/// @param subgraph - a Subgraph object that will own the created Node.
/// @param num_dims - number of shape dimensions in the output tensor.
/// @param new_shape - shape dimensions of the output tensor.
/// @param input_id - Value ID for the input tensor. The input tensor must be defined in the @a subgraph.
/// @param output_id - Value ID for the output tensor. The output tensor must be defined in the @a subgraph, and its
///                    shape must match the shape of the input tensor with padding.
/// @param flags - binary features of the Reshape Node. No supported flags are currently defined.
enum xnn_status xnn_define_static_reshape(
  xnn_subgraph_t subgraph,
  size_t num_dims,
  const size_t* new_shape,
  uint32_t input_id,
  uint32_t output_id,
  uint32_t flags);

/// Define a 2D Resize Bilinear Node with static output height & width specification and add it to a Subgraph.
///
/// @param subgraph - a Subgraph object that will own the created Node.
/// @param new_height - height dimension of the output tensor.
/// @param new_width - width dimension of the output tensor.
/// @param input_id - Value ID for the input tensor. The input tensor must be a 4D tensor defined in the @a subgraph
///                   with [N, H, W, C] dimensions.
/// @param output_id - Value ID for the output tensor. The output tensor must be a 4D tensor defined in the @a subgraph
///                    with [N, new_height, new_width, C] dimensions.
/// @param flags - binary features of the 2D Resize Bilinear Node. The only currently supported values are
///                XNN_FLAG_TENSORFLOW_LEGACY_MODE and XNN_FLAG_ALIGN_CORNERS, which are mutually exclusive.
enum xnn_status xnn_define_static_resize_bilinear_2d(
  xnn_subgraph_t subgraph,
  size_t new_height,
  size_t new_width,
  uint32_t input_id,
  uint32_t output_id,
  uint32_t flags);

/// Define a PReLU (Parametric ReLU) Node and add it to a Subgraph.
///
/// @param subgraph - a Subgraph object that will own the created Node.
/// @param input_id - Value ID for the input tensor. The input tensor must be a 4D tensor defined in the @a subgraph
///                   with [N, H, W, channels] dimensions.
/// @param slope_id - Value ID for the bias tensor. The bias tensor must be a 1D tensor defined in the @a subgraph with
///                   [channels] dimensions.
/// @param output_id - Value ID for the output tensor. The output tensor must be a 4D tensor defined in the @a subgraph
///                    with [N, H, W, channels] dimensions.
/// @param flags - binary features of the PReLU Node. No supported flags are currently defined.
enum xnn_status xnn_define_prelu(
  xnn_subgraph_t subgraph,
  uint32_t input_id,
  uint32_t slope_id,
  uint32_t output_id,
  uint32_t flags);

/// Define a Abs Node and add it to a Subgraph.
///
/// @param subgraph - a Subgraph object that will own the created Node.
/// @param input_id - Value ID for the input tensor. The input tensor must be defined in the @a subgraph.
/// @param output_id - Value ID for the output tensor. The output tensor must be defined in the @a subgraph, and its
///                    shape must match the shape of the input tensor.
/// @param flags - binary features of the Abs Node. No supported flags are currently defined.
enum xnn_status xnn_define_abs(
  xnn_subgraph_t subgraph,
  uint32_t input_id,
  uint32_t output_id,
  uint32_t flags);

/// Define a Bankers' Rounding Node and add it to a Subgraph.
///
/// @param subgraph - a Subgraph object that will own the created Node.
/// @param input_id - Value ID for the input tensor. The input tensor must be defined in the @a subgraph.
/// @param output_id - Value ID for the output tensor. The output tensor must be defined in the @a subgraph, and its
///                    shape must match the shape of the input tensor.
/// @param flags - binary features of the Bankers' Rounding Node. No supported flags are currently defined.
enum xnn_status xnn_define_bankers_rounding(
  xnn_subgraph_t subgraph,
  uint32_t input_id,
  uint32_t output_id,
  uint32_t flags);

/// Define a Ceiling Node and add it to a Subgraph.
///
/// @param subgraph - a Subgraph object that will own the created Node.
/// @param input_id - Value ID for the input tensor. The input tensor must be defined in the @a subgraph.
/// @param output_id - Value ID for the output tensor. The output tensor must be defined in the @a subgraph, and its
///                    shape must match the shape of the input tensor.
/// @param flags - binary features of the Ceiling Node. No supported flags are currently defined.
enum xnn_status xnn_define_ceiling(
  xnn_subgraph_t subgraph,
  uint32_t input_id,
  uint32_t output_id,
  uint32_t flags);

/// Define a Clamp Node and add it to a Subgraph.
///
/// @param subgraph - a Subgraph object that will own the created Node.
/// @param output_min - lower bound for clipping output values.
/// @param output_max - upper bound for clipping output values.
/// @param input_id - Value ID for the input tensor. The input tensor must be defined in the @a subgraph.
/// @param output_id - Value ID for the output tensor. The output tensor must be defined in the @a subgraph, and its
///                    shape must match the shape of the input tensor.
/// @param flags - binary features of the Clamp Node. No supported flags are currently defined.
enum xnn_status xnn_define_clamp(
  xnn_subgraph_t subgraph,
  float output_min,
  float output_max,
  uint32_t input_id,
  uint32_t output_id,
  uint32_t flags);

/// Define an ELU (Exponential Linear Unit) Node and add it to a Subgraph.
///
/// @param subgraph - a Subgraph object that will own the created Node.
/// @param alpha - scale factor for negative output elements.
/// @param input_id - Value ID for the input tensor. The input tensor must be defined in the @a subgraph.
/// @param output_id - Value ID for the output tensor. The output tensor must be defined in the @a subgraph, and its
///                    shape must match the shape of the input tensor.
/// @param flags - binary features of the ELU Node. No supported flags are currently defined.
enum xnn_status xnn_define_elu(
  xnn_subgraph_t subgraph,
  float alpha,
  uint32_t input_id,
  uint32_t output_id,
  uint32_t flags);

/// Define a Floor Node and add it to a Subgraph.
///
/// @param subgraph - a Subgraph object that will own the created Node.
/// @param input_id - Value ID for the input tensor. The input tensor must be defined in the @a subgraph.
/// @param output_id - Value ID for the output tensor. The output tensor must be defined in the @a subgraph, and its
///                    shape must match the shape of the input tensor.
/// @param flags - binary features of the Floor Node. No supported flags are currently defined.
enum xnn_status xnn_define_floor(
  xnn_subgraph_t subgraph,
  uint32_t input_id,
  uint32_t output_id,
  uint32_t flags);

/// Define a HardSwish Node and add it to a Subgraph.
///
/// @param subgraph - a Subgraph object that will own the created Node.
/// @param input_id - Value ID for the input tensor. The input tensor must be defined in the @a subgraph.
/// @param output_id - Value ID for the output tensor. The output tensor must be defined in the @a subgraph, and its
///                    shape must match the shape of the input tensor.
/// @param flags - binary features of the HardSwish Node. No supported flags are currently defined.
enum xnn_status xnn_define_hardswish(
  xnn_subgraph_t subgraph,
  uint32_t input_id,
  uint32_t output_id,
  uint32_t flags);

/// Define a Leaky ReLU Node and add it to a Subgraph.
///
/// @param subgraph - a Subgraph object that will own the created Node.
/// @param negative_slope - scale factor for negative input elements.
/// @param input_id - Value ID for the input tensor. The input tensor must be defined in the @a subgraph.
/// @param output_id - Value ID for the output tensor. The output tensor must be defined in the @a subgraph, and its
///                    shape must match the shape of the input tensor.
/// @param flags - binary features of the Leaky ReLU Node. No supported flags are currently defined.
enum xnn_status xnn_define_leaky_relu(
  xnn_subgraph_t subgraph,
  float negative_slope,
  uint32_t input_id,
  uint32_t output_id,
  uint32_t flags);

/// Define a Negate Node and add it to a Subgraph.
///
/// @param subgraph - a Subgraph object that will own the created Node.
/// @param input_id - Value ID for the input tensor. The input tensor must be defined in the @a subgraph.
/// @param output_id - Value ID for the output tensor. The output tensor must be defined in the @a subgraph, and its
///                    shape must match the shape of the input tensor.
/// @param flags - binary features of the Negate Node. No supported flags are currently defined.
enum xnn_status xnn_define_negate(
  xnn_subgraph_t subgraph,
  uint32_t input_id,
  uint32_t output_id,
  uint32_t flags);

/// Define a Sigmoid Node and add it to a Subgraph.
///
/// @param subgraph - a Subgraph object that will own the created Node.
/// @param input_id - Value ID for the input tensor. The input tensor must be defined in the @a subgraph.
/// @param output_id - Value ID for the output tensor. The output tensor must be defined in the @a subgraph, and its
///                    shape must match the shape of the input tensor.
/// @param flags - binary features of the Sigmoid Node. No supported flags are currently defined.
enum xnn_status xnn_define_sigmoid(
  xnn_subgraph_t subgraph,
  uint32_t input_id,
  uint32_t output_id,
  uint32_t flags);

/// Define a SoftMax Node and add it to a Subgraph.
///
/// @param subgraph - a Subgraph object that will own the created Node.
/// @param input_id - Value ID for the input tensor. The input tensor must be defined in the @a subgraph, and have at
///                   least one dimension.
/// @param output_id - Value ID for the output tensor. The output tensor must be defined in the @a subgraph, and its
///                    shape must match the shape of the input tensor.
/// @param flags - binary features of the SoftMax Node. No supported flags are currently defined.
enum xnn_status xnn_define_softmax(
  xnn_subgraph_t subgraph,
  uint32_t input_id,
  uint32_t output_id,
  uint32_t flags);

/// Define a Space To Depth 2D Node and add it to a Subgraph.
///
/// The Space To Depth 2D Node rearranges blocks of spatial data into blocks (a reverse transform to Depth To Space 2D).
/// For a given input pixel, an output square of pixels with side @a block_size is formed from values in the
/// corresponding number of its channels. The output depth is therefore @a block_size x @a block_size times greater
/// than that of the input.
///
/// @param subgraph - a Subgraph object that will own the created Node.
/// @param block_size - the size of the spatial block.
/// @param input_id - Value ID for the input tensor. The input tensor must be a 4D tensor defined in the @a subgraph
///                   with [N, IH * block_size, IW * block_size, OC] dimensions.
/// @param output_id - Value ID for the output tensor. The output tensor must be a 4D tensor defined in the @a subgraph
///                    with [N, IH, IW, OC * block_size * block_size] dimensions.
/// @param flags - binary features of the input_channels Node. No supported flags are currently defined.
enum xnn_status xnn_define_space_to_depth_2d(
  xnn_subgraph_t subgraph,
  uint32_t block_size,
  uint32_t input_id,
  uint32_t output_id,
  uint32_t flags);

/// Define a Square Node and add it to a Subgraph.
///
/// @param subgraph - a Subgraph object that will own the created Node.
/// @param input_id - Value ID for the input tensor. The input tensor must be defined in the @a subgraph.
/// @param output_id - Value ID for the output tensor. The output tensor must be defined in the @a subgraph, and its
///                    shape must match the shape of the input tensor.
/// @param flags - binary features of the Square Node. No supported flags are currently defined.
enum xnn_status xnn_define_square(
  xnn_subgraph_t subgraph,
  uint32_t input_id,
  uint32_t output_id,
  uint32_t flags);

/// Define a Square Root Node and add it to a Subgraph.
///
/// @param subgraph - a Subgraph object that will own the created Node.
/// @param input_id - Value ID for the input tensor. The input tensor must be defined in the @a subgraph.
/// @param output_id - Value ID for the output tensor. The output tensor must be defined in the @a subgraph, and its
///                    shape must match the shape of the input tensor.
/// @param flags - binary features of the Square Root Node. No supported flags are currently defined.
enum xnn_status xnn_define_square_root(
  xnn_subgraph_t subgraph,
  uint32_t input_id,
  uint32_t output_id,
  uint32_t flags);

/// Define a Static Slice Node add it to a Subgraph.
///
/// @param subgraph - a Subgraph object that will own the created Node.
/// @param num_dims - number of shape dimensions in the input and output tensor.
/// @param offsets - offsets in each dimension of the input tensor. This array must have @a num_dims elements.
/// @param sizes - size of each dimension in output tensor. This array must have @a num_dims elements.
/// @param input_id - Value ID for the input tensor. The input tensor must be defined in the @a subgraph.
/// @param output_id - Value ID for the output tensor. The output tensor must be defined in the @a subgraph, and its
///                    dimensions must match @a sizes.
/// @param flags - binary features of the Static Slice Node. No supported flags are currently defined.
enum xnn_status xnn_define_static_slice(
  xnn_subgraph_t subgraph,
  size_t num_dims,
  const size_t* offsets,
  const size_t* sizes,
  uint32_t input_id,
  uint32_t output_id,
  uint32_t flags);

/// Define a Static Transpose Node and add it to a Subgraph.
///
/// The Static Transpose Node applies a generalized transpose to the input tensor using the permuation in perm.
///
/// @param subgraph - a Subgraph object that will own the created Node.
/// @param input_id - Value ID for the input tensor. The input tensor must be an N-dimensional tensor defined in
///                   the @a subgraph.
/// @param output_id - Value ID for the output tensor. The output tensor must be an N-dimensional tensor defined
///                    in the @a subgraph with each dimension equal to its corresponding permuted input dimension.
/// @param num_dims - the number of permutation dimensions. This must be equal to the number of input dimensions.
/// @param perm - The permutation of the axis of the input tensor. The perm array must must contain 0 to N-1 in the
///               permuted order.
/// @param flags - binary features of the Static Transpose Node. No supported flags are currently defined.
enum xnn_status xnn_define_static_transpose(
  xnn_subgraph_t subgraph,
  size_t num_dims,
  const size_t* perm,
  uint32_t input_id,
  uint32_t output_id,
  uint32_t flags);

/// Weights cache is a cache for packed weights. It can be reused between runtimes.
typedef struct xnn_weights_cache* xnn_weights_cache_t;

enum xnn_status xnn_create_weights_cache(xnn_weights_cache_t* weights_cache_out);

/// Create a weights cache object specifying the initial size of weights cache (in bytes).
/// @size - initial capacity of the weights cache (in bytes), i.e. it can hold size bytes without growing.
/// @param weights_cache_out - pointer to the variable that will be initialized to a handle to the weights cache object
///                            upon successful return. Once created, the weights cache object can be shared between
///                            different Runtime objects.
enum xnn_status xnn_create_weights_cache_with_size(size_t size, xnn_weights_cache_t* weights_cache_out);


/// Weights cache can be finalized in these ways:
enum xnn_weights_cache_finalization_kind {
  /// Weights cache is finalized, no insert operations into the weights cache is allowed, even if the "inserted"
  /// weights already exist in thee cache. Weights cache memory will also be trimmed to page boundary and set to
  /// read-only (to prevent writes).
  xnn_weights_cache_finalization_kind_hard,
  /// Weights cache will be finalized with some extra space at the end, this allows for "inserting" into the cache only
  /// if the weights are already in the cache, and errors on inserting uncached weights. There is memory overhead.
  xnn_weights_cache_finalization_kind_soft,
};

/// Finalizes the weights cache. The kind of finalization is specified by `finalization_kind`.
/// @param weights_cache - the weights cache object to finalize.
/// @param finalization_kind - the kind of finalization.
enum xnn_status xnn_finalize_weights_cache(
  xnn_weights_cache_t weights_cache,
  enum xnn_weights_cache_finalization_kind finalization_kind);

/// Destroy a weights cache object, as well as memory used for the cache.
/// @param weights_cache - the weights cache object to destroy.
enum xnn_status xnn_delete_weights_cache(xnn_weights_cache_t weights_cache);

typedef struct xnn_workspace* xnn_workspace_t;

/// Create a workspace object.
/// @param workspace_out - pointer to the variable that will be initialized to a handle to the workspace object upon
///                        successful return. Once created, the workspace can be shared between different Runtime
///                        objects.
enum xnn_status xnn_create_workspace(xnn_workspace_t* workspace_out);
/// Destroy a workspace object, as well as memory used by the workspace. Object destruction can be deferred until all
/// Runtime objects created with this workspace are destroyed.
/// @param workspace - the workspace object to destroy.
enum xnn_status xnn_release_workspace(xnn_workspace_t workspace);

/// Runtime is a combination of an execution plan for subgraph Nodes and a memory manager for subgraph Values.
typedef struct xnn_runtime* xnn_runtime_t;

enum xnn_profile_info {
  /// Returns a size_t containing the number of operators.
  xnn_profile_info_num_operators,
  /// Returns a char[] containing the null character separated names of all operators.
  xnn_profile_info_operator_name,
  /// Returns a uint64_t[] with the runtimes of all operators in the same order as xnn_profile_info_operator_name.
  xnn_profile_info_operator_timing,
};

/// Return profile information for all operators.
///
/// @param runtime - a Runtime object created with @ref xnn_create_runtime, @ref xnn_create_runtime_v2 or
///                  @ref xnn_create_runtime_v3.
/// @param param_name - type of profile information required.
/// @param param_value_size - the size in bytes of memory pointed to by param_value. If this is not sufficient then
///                           param_value_size_ret will be set to the required size and xnn_status_out_of_memory will be
///                           returned.
/// @param param_value - a pointer to memory location where appropriate values for a given param_value will be written.
/// @param param_value_size_ret - returns number of bytes required to write the result if param_value_size is not
///                               sufficient.
enum xnn_status xnn_get_runtime_profiling_info(xnn_runtime_t runtime,
                                               enum xnn_profile_info param_name,
                                               size_t param_value_size,
                                               void* param_value,
                                               size_t* param_value_size_ret);

/// Create a Runtime object from a subgraph.
///
/// @param subgraph - a Subgraph object with all Values and Nodes that would be handled by the runtime. No Values or
///                   Nodes can be added to the runtime once it is constructed.
/// @param weights_cache - a cache for packed weights. The runtime will look up and reuse packed weights in this cache,
///                        this will reduce memory allocated for packed weights.
/// @param workspace - a workspace to hold internal tensors. The runtime will allocate space used for internal tensors
///                    and track them using workspace. Workspace can be shared and reused across different runtimes. If
///                    workspace is NULL, there will be no sharing: each runtime has its own workspace.
/// @param threadpool - the thread pool to be used for parallelisation of computations in the runtime. If the thread
///                     pool is NULL, the computation would run on the caller thread without parallelization.
/// @param flags - binary features of the runtime. The only currently supported values are
///                XNN_FLAG_HINT_SPARSE_INFERENCE, XNN_FLAG_HINT_FP16_INFERENCE, XNN_FLAG_FORCE_FP16_INFERENCE, and
///                XNN_FLAG_YIELD_WORKERS. If XNN_FLAG_YIELD_WORKERS is specified, worker threads would be yielded to
///                the system scheduler after processing the last operator in the Runtime.
/// @param runtime_out - pointer to the variable that will be initialized with a handle to the Runtime object upon
///                      successful return. Once constructed, the Runtime object is independent of the Subgraph object
///                      used to create it.
enum xnn_status xnn_create_runtime_v4(
  xnn_subgraph_t subgraph,
  xnn_weights_cache_t weights_cache,
  xnn_workspace_t workspace,
  pthreadpool_t threadpool,
  uint32_t flags,
  xnn_runtime_t* runtime_out);

enum xnn_status xnn_create_runtime_v3(
  xnn_subgraph_t subgraph,
  xnn_weights_cache_t weights_cache,
  pthreadpool_t threadpool,
  uint32_t flags,
  xnn_runtime_t* runtime_out);

enum xnn_status xnn_create_runtime_v2(
  xnn_subgraph_t subgraph,
  pthreadpool_t threadpool,
  uint32_t flags,
  xnn_runtime_t* runtime_out);

enum xnn_status xnn_create_runtime(
  xnn_subgraph_t subgraph,
  xnn_runtime_t* runtime_out);

struct xnn_external_value {
  uint32_t id;
  void* data;
};

/// Setup data pointers for external inputs and outputs in a Runtime object.
///
/// @param runtime - a Runtime object created with @ref xnn_create_runtime or @ref xnn_create_runtime_v2.
/// @param num_external_values - the number of external inputs and outputs specified in this call. This number must
///                              match the number of external inputs and outputs in the runtime, i.e. all external
///                              inputs and outputs in the runtime must be specified in one call.
/// @param external_values - array with location information for all external inputs and outputs in the runtime.
enum xnn_status xnn_setup_runtime(
  xnn_runtime_t runtime,
  size_t num_external_values,
  const struct xnn_external_value* external_values);

/// Execute forward pass for all operators in the runtime.
///
/// @param runtime - the Runtime object with the execution plan to invoke.
enum xnn_status xnn_invoke_runtime(
  xnn_runtime_t runtime);

/// Destroy a Runtime object, as well as operators and memory associated with it.
///
/// @param runtime - the Runtime object to destroy.
enum xnn_status xnn_delete_runtime(
  xnn_runtime_t runtime);

typedef struct xnn_operator* xnn_operator_t;

enum xnn_status xnn_run_operator(
  xnn_operator_t op,
  pthreadpool_t threadpool);

enum xnn_status xnn_delete_operator(
  xnn_operator_t op);

#ifndef XNN_NO_F32_OPERATORS

enum xnn_status xnn_create_abs_nc_f32(
  size_t channels,
  size_t input_stride,
  size_t output_stride,
  uint32_t flags,
  xnn_operator_t* abs_op_out);

enum xnn_status xnn_setup_abs_nc_f32(
  xnn_operator_t abs_op,
  size_t batch_size,
  const float* input,
  float* output,
  pthreadpool_t threadpool);

enum xnn_status xnn_run_abs_nc_f32(
  size_t channels,
  size_t input_stride,
  size_t output_stride,
  size_t batch_size,
  const float* input,
  float* output,
  uint32_t flags,
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

enum xnn_status xnn_run_add_nd_f32(
  size_t num_input1_dims,
  const size_t* input1_shape,
  size_t num_input2_dims,
  const size_t* input2_shape,
  const float* input1,
  const float* input2,
  float* output,
  float output_min,
  float output_max,
  uint32_t flags,
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

enum xnn_status xnn_create_bankers_rounding_nc_f32(
  size_t channels,
  size_t input_stride,
  size_t output_stride,
  uint32_t flags,
  xnn_operator_t* rounding_op_out);

enum xnn_status xnn_setup_bankers_rounding_nc_f32(
  xnn_operator_t rounding_op,
  size_t batch_size,
  const float* input,
  float* output,
  pthreadpool_t threadpool);

enum xnn_status xnn_run_bankers_rounding_nc_f32(
  size_t channels,
  size_t input_stride,
  size_t output_stride,
  size_t batch_size,
  const float* input,
  float* output,
  uint32_t flags,
  pthreadpool_t threadpool);

enum xnn_status xnn_create_ceiling_nc_f32(
  size_t channels,
  size_t input_stride,
  size_t output_stride,
  uint32_t flags,
  xnn_operator_t* ceiling_op_out);

enum xnn_status xnn_run_ceiling_nc_f32(
  size_t channels,
  size_t input_stride,
  size_t output_stride,
  size_t batch_size,
  const float* input,
  float* output,
  uint32_t flags,
  pthreadpool_t threadpool);

enum xnn_status xnn_setup_ceiling_nc_f32(
  xnn_operator_t ceiling_op,
  size_t batch_size,
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

enum xnn_status xnn_run_clamp_nc_f32(
  size_t channels,
  size_t input_stride,
  size_t output_stride,
  size_t batch_size,
  const float* input,
  float* output,
  float output_min,
  float output_max,
  uint32_t flags,
  pthreadpool_t threadpool);

typedef const struct xnn_caches* xnn_caches_t;

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
  xnn_caches_t caches,
  xnn_operator_t* convolution_op_out);

// Forward declare.
struct xnn_post_operation;

/// Create a convolution operator with a number of post operations. The
/// convolution operator created using this function does not have output_min
/// and output_max. The list of operators in post_operations will be applied in
/// order. Convolution with post operations is only supported on JIT platforms
/// and when JIT is enabled.
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
    xnn_caches_t caches,
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
  xnn_caches_t caches,
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

enum xnn_status xnn_run_divide_nd_f32(
  size_t num_input1_dims,
  const size_t* input1_shape,
  size_t num_input2_dims,
  const size_t* input2_shape,
  const float* input1,
  const float* input2,
  float* output,
  float output_min,
  float output_max,
  uint32_t flags,
  pthreadpool_t threadpool);

enum xnn_status xnn_create_elu_nc_f32(
  size_t channels,
  size_t input_stride,
  size_t output_stride,
  float alpha,
  uint32_t flags,
  xnn_operator_t* elu_op_out);

enum xnn_status xnn_setup_elu_nc_f32(
  xnn_operator_t elu_op,
  size_t batch_size,
  const float* input,
  float* output,
  pthreadpool_t threadpool);

enum xnn_status xnn_run_elu_nc_f32(
  size_t channels,
  size_t input_stride,
  size_t output_stride,
  size_t batch_size,
  const float* input,
  float* output,
  float alpha,
  uint32_t flags,
  pthreadpool_t threadpool);

enum xnn_status xnn_create_floor_nc_f32(
  size_t channels,
  size_t input_stride,
  size_t output_stride,
  uint32_t flags,
  xnn_operator_t* floor_op_out);

enum xnn_status xnn_setup_floor_nc_f32(
  xnn_operator_t floor_op,
  size_t batch_size,
  const float* input,
  float* output,
  pthreadpool_t threadpool);

enum xnn_status xnn_run_floor_nc_f32(
  size_t channels,
  size_t input_stride,
  size_t output_stride,
  size_t batch_size,
  const float* input,
  float* output,
  uint32_t flags,
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
  const xnn_caches_t caches,
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

enum xnn_status xnn_run_hardswish_nc_f32(
  size_t channels,
  size_t input_stride,
  size_t output_stride,
  size_t batch_size,
  const float* input,
  float* output,
  uint32_t flags,
  pthreadpool_t threadpool);

enum xnn_status xnn_create_leaky_relu_nc_f32(
  size_t channels,
  size_t input_stride,
  size_t output_stride,
  float negative_slope,
  uint32_t flags,
  xnn_operator_t* leaky_relu_op_out);

enum xnn_status xnn_setup_leaky_relu_nc_f32(
  xnn_operator_t leaky_relu_op,
  size_t batch_size,
  const float* input,
  float* output,
  pthreadpool_t threadpool);

enum xnn_status xnn_run_leaky_relu_nc_f32(
  size_t channels,
  size_t input_stride,
  size_t output_stride,
  size_t batch_size,
  const float* input,
  float* output,
  float negative_slope,
  uint32_t flags,
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

enum xnn_status xnn_run_maximum_nd_f32(
  size_t num_input1_dims,
  const size_t* input1_shape,
  size_t num_input2_dims,
  const size_t* input2_shape,
  const float* input1,
  const float* input2,
  float* output,
  float output_min,
  float output_max,
  uint32_t flags,
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

enum xnn_status xnn_run_minimum_nd_f32(
  size_t num_input1_dims,
  const size_t* input1_shape,
  size_t num_input2_dims,
  const size_t* input2_shape,
  const float* input1,
  const float* input2,
  float* output,
  float output_min,
  float output_max,
  uint32_t flags,
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

enum xnn_status xnn_run_multiply_nd_f32(
  size_t num_input1_dims,
  const size_t* input1_shape,
  size_t num_input2_dims,
  const size_t* input2_shape,
  const float* input1,
  const float* input2,
  float* output,
  float output_min,
  float output_max,
  uint32_t flags,
  pthreadpool_t threadpool);

enum xnn_status xnn_create_negate_nc_f32(
  size_t channels,
  size_t input_stride,
  size_t output_stride,
  uint32_t flags,
  xnn_operator_t* negate_op_out);

enum xnn_status xnn_setup_negate_nc_f32(
  xnn_operator_t negate_op,
  size_t batch_size,
  const float* input,
  float* output,
  pthreadpool_t threadpool);

enum xnn_status xnn_run_negate_nc_f32(
  size_t channels,
  size_t input_stride,
  size_t output_stride,
  size_t batch_size,
  const float* input,
  float* output,
  uint32_t flags,
  pthreadpool_t threadpool);

enum xnn_status xnn_create_prelu_nc_f32(
  size_t channels,
  size_t input_stride,
  size_t output_stride,
  const float* negative_slope,
  uint32_t flags,
  xnn_caches_t caches,
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

enum xnn_status xnn_run_sigmoid_nc_f32(
  size_t channels,
  size_t input_stride,
  size_t output_stride,
  size_t batch_size,
  const float* input,
  float* output,
  uint32_t flags,
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

enum xnn_status xnn_create_square_nc_f32(
  size_t channels,
  size_t input_stride,
  size_t output_stride,
  uint32_t flags,
  xnn_operator_t* square_op_out);

enum xnn_status xnn_setup_square_nc_f32(
  xnn_operator_t square_op,
  size_t batch_size,
  const float* input,
  float* output,
  pthreadpool_t threadpool);

enum xnn_status xnn_run_square_nc_f32(
  size_t channels,
  size_t input_stride,
  size_t output_stride,
  size_t batch_size,
  const float* input,
  float* output,
  uint32_t flags,
  pthreadpool_t threadpool);

enum xnn_status xnn_create_square_root_nc_f32(
  size_t channels,
  size_t input_stride,
  size_t output_stride,
  uint32_t flags,
  xnn_operator_t* sqrt_op_out);

enum xnn_status xnn_setup_square_root_nc_f32(
  xnn_operator_t sqrt_op,
  size_t batch_size,
  const float* input,
  float* output,
  pthreadpool_t threadpool);

enum xnn_status xnn_run_square_root_nc_f32(
  size_t channels,
  size_t input_stride,
  size_t output_stride,
  size_t batch_size,
  const float* input,
  float* output,
  uint32_t flags,
  pthreadpool_t threadpool);

enum xnn_status xnn_create_squared_difference_nd_f32(
  uint32_t flags,
  xnn_operator_t* squared_difference_op_out);

enum xnn_status xnn_setup_squared_difference_nd_f32(
  xnn_operator_t squared_difference_op,
  size_t num_input1_dims,
  const size_t* input1_shape,
  size_t num_input2_dims,
  const size_t* input2_shape,
  const float* input1,
  const float* input2,
  float* output,
  pthreadpool_t threadpool);

enum xnn_status xnn_run_squared_difference_nd_f32(
  size_t num_input1_dims,
  const size_t* input1_shape,
  size_t num_input2_dims,
  const size_t* input2_shape,
  const float* input1,
  const float* input2,
  float* output,
  float output_min,
  float output_max,
  uint32_t flags,
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

enum xnn_status xnn_run_subtract_nd_f32(
  size_t num_input1_dims,
  const size_t* input1_shape,
  size_t num_input2_dims,
  const size_t* input2_shape,
  const float* input1,
  const float* input2,
  float* output,
  float output_min,
  float output_max,
  uint32_t flags,
  pthreadpool_t threadpool);

enum xnn_status xnn_create_truncation_nc_f32(
  size_t channels,
  size_t input_stride,
  size_t output_stride,
  uint32_t flags,
  xnn_operator_t* truncation_op_out);

enum xnn_status xnn_setup_truncation_nc_f32(
  xnn_operator_t truncation_op,
  size_t batch_size,
  const float* input,
  float* output,
  pthreadpool_t threadpool);

enum xnn_status xnn_run_truncation_nc_f32(
  size_t channels,
  size_t input_stride,
  size_t output_stride,
  size_t batch_size,
  const float* input,
  float* output,
  uint32_t flags,
  pthreadpool_t threadpool);

#ifndef XNN_NO_NCHW_OPERATORS

enum xnn_status xnn_create_depth_to_space_nchw2nhwc_x32(
  size_t output_channels,
  size_t input_channel_stride,
  size_t output_channel_stride,
  uint32_t block_size,
  uint32_t flags,
  xnn_operator_t* depth_to_space_op_out);

enum xnn_status xnn_setup_depth_to_space_nchw2nhwc_x32(
  xnn_operator_t depth_to_space_op,
  size_t batch_size,
  size_t input_height,
  size_t input_width,
  const void* input,
  void* output,
  pthreadpool_t threadpool);

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
  xnn_caches_t caches,
  xnn_operator_t* convolution_op_out);

enum xnn_status xnn_setup_convolution2d_nchw_f32(
  xnn_operator_t convolution_op,
  size_t batch_size,
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

enum xnn_status xnn_create_resize_bilinear2d_nchw_f32(
  size_t channels,
  size_t input_pixel_stride,
  size_t output_pixel_stride,
  uint32_t flags,
  xnn_operator_t* resize_op_out);

enum xnn_status xnn_setup_resize_bilinear2d_nchw_f32(
  xnn_operator_t resize_op,
  size_t batch_size,
  size_t input_height,
  size_t input_width,
  size_t output_height,
  size_t output_width,
  const float* input,
  float* output,
  pthreadpool_t threadpool);

#endif  // XNN_NO_NCHW_OPERATORS

#endif  // XNN_NO_F32_OPERATORS

#ifndef XNN_NO_X32_OPERATORS

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

enum xnn_status xnn_create_constant_pad_nd_x32(
  const void* padding_value,
  uint32_t flags,
  xnn_operator_t* constant_pad_op_out);

enum xnn_status xnn_setup_constant_pad_nd_x32(
  xnn_operator_t constant_pad_op,
  size_t num_dims,
  const size_t* input_shape,
  const size_t* pre_padding,
  const size_t* post_padding,
  const void* input,
  void* output,
  pthreadpool_t threadpool);

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

enum xnn_status xnn_create_copy_nc_x32(
  size_t channels,
  size_t input_stride,
  size_t output_stride,
  uint32_t flags,
  xnn_operator_t* copy_op_out);

enum xnn_status xnn_setup_copy_nc_x32(
  xnn_operator_t copy_op,
  size_t batch_size,
  const void* input,
  void* output,
  pthreadpool_t threadpool);

enum xnn_status xnn_run_copy_nc_x32(
  size_t channels,
  size_t input_stride,
  size_t output_stride,
  size_t batch_size,
  const uint32_t* input,
  uint32_t* output,
  uint32_t flags,
  pthreadpool_t threadpool);

enum xnn_status xnn_create_depth_to_space_nhwc_x32(
  size_t output_channels,
  size_t input_channel_stride,
  size_t output_channel_stride,
  uint32_t block_size,
  uint32_t flags,
  xnn_operator_t* depth_to_space_op_out);

enum xnn_status xnn_setup_depth_to_space_nhwc_x32(
  xnn_operator_t depth_to_space_op,
  size_t batch_size,
  size_t input_height,
  size_t input_width,
  const void* input,
  void* output,
  pthreadpool_t threadpool);

enum xnn_status xnn_create_slice_nd_x32(
  uint32_t flags,
  xnn_operator_t* slice_op_out);

enum xnn_status xnn_setup_slice_nd_x32(
  xnn_operator_t slice_op,
  size_t num_dims,
  const size_t* input_shape,
  const size_t* offsets,
  const size_t* sizes,
  const void* input,
  void* output,
  pthreadpool_t threadpool);

enum xnn_status xnn_run_slice_nd_x32(
  size_t num_dims,
  const size_t* input_shape,
  const size_t* offsets,
  const size_t* sizes,
  const void* input,
  void* output,
  uint32_t flags,
  pthreadpool_t threadpool);

enum xnn_status xnn_create_space_to_depth_nhwc_x32(
  size_t input_channels,
  size_t input_channel_stride,
  size_t output_channel_stride,
  uint32_t block_size,
  uint32_t flags,
  xnn_operator_t* space_to_depth_op_out);

enum xnn_status xnn_setup_space_to_depth_nhwc_x32(
  xnn_operator_t space_to_depth_op,
  size_t batch_size,
  size_t input_height,
  size_t input_width,
  const void* input,
  void* output,
  pthreadpool_t threadpool);

enum xnn_status xnn_create_transpose_nd_x32(
    uint32_t flags,
    xnn_operator_t* transpose_op_out);

enum xnn_status xnn_setup_transpose_nd_x32(
    xnn_operator_t transpose_op,
    const void* input,
    void* output,
    const size_t num_dims,
    const size_t* input_shape,
    const size_t* output_perm,
    pthreadpool_t threadpool);

enum xnn_status xnn_run_transpose_nd_x32(
    const void* input,
    void* output,
    const size_t num_dims,
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

#ifndef XNN_NO_F16_OPERATORS

enum xnn_status xnn_create_abs_nc_f16(
  size_t channels,
  size_t input_stride,
  size_t output_stride,
  uint32_t flags,
  xnn_operator_t* abs_op_out);

enum xnn_status xnn_setup_abs_nc_f16(
  xnn_operator_t abs_op,
  size_t batch_size,
  const void* input,
  void* output,
  pthreadpool_t threadpool);

enum xnn_status xnn_create_add_nd_f16(
  float output_min,
  float output_max,
  uint32_t flags,
  xnn_operator_t* add_op_out);

enum xnn_status xnn_setup_add_nd_f16(
  xnn_operator_t add_op,
  size_t num_input1_dims,
  const size_t* input1_shape,
  size_t num_input2_dims,
  const size_t* input2_shape,
  const void* input1,
  const void* input2,
  void* output,
  pthreadpool_t threadpool);

enum xnn_status xnn_create_average_pooling2d_nhwc_f16(
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

enum xnn_status xnn_setup_average_pooling2d_nhwc_f16(
  xnn_operator_t average_pooling_op,
  size_t batch_size,
  size_t input_height,
  size_t input_width,
  const void* input,
  void* output,
  pthreadpool_t threadpool);

enum xnn_status xnn_create_bankers_rounding_nc_f16(
  size_t channels,
  size_t input_stride,
  size_t output_stride,
  uint32_t flags,
  xnn_operator_t* rounding_op_out);

enum xnn_status xnn_setup_bankers_rounding_nc_f16(
  xnn_operator_t rounding_op,
  size_t batch_size,
  const void* input,
  void* output,
  pthreadpool_t threadpool);

enum xnn_status xnn_create_ceiling_nc_f16(
  size_t channels,
  size_t input_stride,
  size_t output_stride,
  uint32_t flags,
  xnn_operator_t* ceiling_op_out);

enum xnn_status xnn_setup_ceiling_nc_f16(
  xnn_operator_t ceiling_op,
  size_t batch_size,
  const void* input,
  void* output,
  pthreadpool_t threadpool);

enum xnn_status xnn_create_clamp_nc_f16(
  size_t channels,
  size_t input_stride,
  size_t output_stride,
  float output_min,
  float output_max,
  uint32_t flags,
  xnn_operator_t* clamp_op_out);

enum xnn_status xnn_setup_clamp_nc_f16(
  xnn_operator_t clamp_op,
  size_t batch_size,
  const void* input,
  void* output,
  pthreadpool_t threadpool);

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
  xnn_caches_t caches,
  xnn_operator_t* convolution_op_out);

enum xnn_status xnn_setup_convolution2d_nhwc_f16(
  xnn_operator_t convolution_op,
  size_t batch_size,
  size_t input_height,
  size_t input_width,
  const void* input,
  void* output,
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
  xnn_caches_t caches,
  xnn_operator_t* deconvolution_op_out);

enum xnn_status xnn_setup_deconvolution2d_nhwc_f16(
  xnn_operator_t deconvolution_op,
  size_t batch_size,
  size_t input_height,
  size_t input_width,
  uint32_t adjustment_height,
  uint32_t adjustment_width,
  const void* input,
  void* output,
  pthreadpool_t threadpool);

enum xnn_status xnn_create_divide_nd_f16(
  float output_min,
  float output_max,
  uint32_t flags,
  xnn_operator_t* divide_op_out);

enum xnn_status xnn_setup_divide_nd_f16(
  xnn_operator_t divide_op,
  size_t num_input1_dims,
  const size_t* input1_shape,
  size_t num_input2_dims,
  const size_t* input2_shape,
  const void* input1,
  const void* input2,
  void* output,
  pthreadpool_t threadpool);

enum xnn_status xnn_create_elu_nc_f16(
  size_t channels,
  size_t input_stride,
  size_t output_stride,
  float alpha,
  uint32_t flags,
  xnn_operator_t* elu_op_out);

enum xnn_status xnn_setup_elu_nc_f16(
  xnn_operator_t elu_op,
  size_t batch_size,
  const void* input,
  void* output,
  pthreadpool_t threadpool);

enum xnn_status xnn_create_floor_nc_f16(
  size_t channels,
  size_t input_stride,
  size_t output_stride,
  uint32_t flags,
  xnn_operator_t* floor_op_out);

enum xnn_status xnn_setup_floor_nc_f16(
  xnn_operator_t floor_op,
  size_t batch_size,
  const void* input,
  void* output,
  pthreadpool_t threadpool);

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
  xnn_caches_t caches,
  xnn_operator_t* fully_connected_op_out);

enum xnn_status xnn_setup_fully_connected_nc_f16(
  xnn_operator_t fully_connected_op,
  size_t batch_size,
  const void* input,
  void* output,
  pthreadpool_t threadpool);

enum xnn_status xnn_create_global_average_pooling_nwc_f16(
  size_t channels,
  size_t input_stride,
  size_t output_stride,
  float output_min,
  float output_max,
  uint32_t flags,
  xnn_operator_t* global_average_pooling_op_out);

enum xnn_status xnn_setup_global_average_pooling_nwc_f16(
  xnn_operator_t global_average_pooling_op,
  size_t batch_size,
  size_t width,
  const void* input,
  void* output,
  pthreadpool_t threadpool);

enum xnn_status xnn_create_hardswish_nc_f16(
  size_t channels,
  size_t input_stride,
  size_t output_stride,
  uint32_t flags,
  xnn_operator_t* hardswish_op_out);

enum xnn_status xnn_setup_hardswish_nc_f16(
  xnn_operator_t hardswish_op,
  size_t batch_size,
  const void* input,
  void* output,
  pthreadpool_t threadpool);

enum xnn_status xnn_create_leaky_relu_nc_f16(
  size_t channels,
  size_t input_stride,
  size_t output_stride,
  float negative_slope,
  uint32_t flags,
  xnn_operator_t* leaky_relu_op_out);

enum xnn_status xnn_setup_leaky_relu_nc_f16(
  xnn_operator_t leaky_relu_op,
  size_t batch_size,
  const void* input,
  void* output,
  pthreadpool_t threadpool);

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
  size_t channels,
  size_t input_pixel_stride,
  size_t output_pixel_stride,
  float output_min,
  float output_max,
  uint32_t flags,
  xnn_operator_t* max_pooling_op_out);

enum xnn_status xnn_setup_max_pooling2d_nhwc_f16(
  xnn_operator_t max_pooling_op,
  size_t batch_size,
  size_t input_height,
  size_t input_width,
  const void* input,
  void* output,
  pthreadpool_t threadpool);

enum xnn_status xnn_create_maximum_nd_f16(
  uint32_t flags,
  xnn_operator_t* maximum_op_out);

enum xnn_status xnn_setup_maximum_nd_f16(
  xnn_operator_t maximum_op,
  size_t num_input1_dims,
  const size_t* input1_shape,
  size_t num_input2_dims,
  const size_t* input2_shape,
  const void* input1,
  const void* input2,
  void* output,
  pthreadpool_t threadpool);

enum xnn_status xnn_create_minimum_nd_f16(
  uint32_t flags,
  xnn_operator_t* minimum_op_out);

enum xnn_status xnn_setup_minimum_nd_f16(
  xnn_operator_t minimum_op,
  size_t num_input1_dims,
  const size_t* input1_shape,
  size_t num_input2_dims,
  const size_t* input2_shape,
  const void* input1,
  const void* input2,
  void* output,
  pthreadpool_t threadpool);

enum xnn_status xnn_create_multiply_nd_f16(
  float output_min,
  float output_max,
  uint32_t flags,
  xnn_operator_t* multiply_op_out);

enum xnn_status xnn_setup_multiply_nd_f16(
  xnn_operator_t multiply_op,
  size_t num_input1_dims,
  const size_t* input1_shape,
  size_t num_input2_dims,
  const size_t* input2_shape,
  const void* input1,
  const void* input2,
  void* output,
  pthreadpool_t threadpool);

enum xnn_status xnn_create_negate_nc_f16(
  size_t channels,
  size_t input_stride,
  size_t output_stride,
  uint32_t flags,
  xnn_operator_t* negate_op_out);

enum xnn_status xnn_setup_negate_nc_f16(
  xnn_operator_t negate_op,
  size_t batch_size,
  const void* input,
  void* output,
  pthreadpool_t threadpool);

enum xnn_status xnn_create_prelu_nc_f16(
  size_t channels,
  size_t input_stride,
  size_t output_stride,
  const void* negative_slope,
  uint32_t flags,
  xnn_caches_t caches,
  xnn_operator_t* prelu_op_out);

enum xnn_status xnn_setup_prelu_nc_f16(
  xnn_operator_t prelu_op,
  size_t batch_size,
  const void* input,
  void* output,
  pthreadpool_t threadpool);

enum xnn_status xnn_create_resize_bilinear2d_nhwc_f16(
  size_t channels,
  size_t input_pixel_stride,
  size_t output_pixel_stride,
  uint32_t flags,
  xnn_operator_t* resize_op_out);

enum xnn_status xnn_setup_resize_bilinear2d_nhwc_f16(
  xnn_operator_t resize_op,
  size_t batch_size,
  size_t input_height,
  size_t input_width,
  size_t output_height,
  size_t output_width,
  const void* input,
  void* output,
  pthreadpool_t threadpool);

enum xnn_status xnn_create_sigmoid_nc_f16(
  size_t channels,
  size_t input_stride,
  size_t output_stride,
  uint32_t flags,
  xnn_operator_t* sigmoid_op_out);

enum xnn_status xnn_setup_sigmoid_nc_f16(
  xnn_operator_t sigmoid_op,
  size_t batch_size,
  const void* input,
  void* output,
  pthreadpool_t threadpool);

enum xnn_status xnn_create_softmax_nc_f16(
  size_t channels,
  size_t input_stride,
  size_t output_stride,
  uint32_t flags,
  xnn_operator_t* softmax_op_out);

enum xnn_status xnn_setup_softmax_nc_f16(
  xnn_operator_t softmax_op,
  size_t batch_size,
  const void* input,
  void* output,
  pthreadpool_t threadpool);

enum xnn_status xnn_create_square_nc_f16(
  size_t channels,
  size_t input_stride,
  size_t output_stride,
  uint32_t flags,
  xnn_operator_t* square_op_out);

enum xnn_status xnn_setup_square_nc_f16(
  xnn_operator_t square_op,
  size_t batch_size,
  const void* input,
  void* output,
  pthreadpool_t threadpool);

enum xnn_status xnn_create_square_root_nc_f16(
  size_t channels,
  size_t input_stride,
  size_t output_stride,
  uint32_t flags,
  xnn_operator_t* sqrt_op_out);

enum xnn_status xnn_setup_square_root_nc_f16(
  xnn_operator_t sqrt_op,
  size_t batch_size,
  const void* input,
  void* output,
  pthreadpool_t threadpool);

enum xnn_status xnn_create_squared_difference_nd_f16(
  uint32_t flags,
  xnn_operator_t* squared_difference_op_out);

enum xnn_status xnn_setup_squared_difference_nd_f16(
  xnn_operator_t squared_difference_op,
  size_t num_input1_dims,
  const size_t* input1_shape,
  size_t num_input2_dims,
  const size_t* input2_shape,
  const void* input1,
  const void* input2,
  void* output,
  pthreadpool_t threadpool);

enum xnn_status xnn_create_subtract_nd_f16(
  float output_min,
  float output_max,
  uint32_t flags,
  xnn_operator_t* subtract_op_out);

enum xnn_status xnn_setup_subtract_nd_f16(
  xnn_operator_t subtract_op,
  size_t num_input1_dims,
  const size_t* input1_shape,
  size_t num_input2_dims,
  const size_t* input2_shape,
  const void* input1,
  const void* input2,
  void* output,
  pthreadpool_t threadpool);

enum xnn_status xnn_create_truncation_nc_f16(
  size_t channels,
  size_t input_stride,
  size_t output_stride,
  uint32_t flags,
  xnn_operator_t* truncation_op_out);

enum xnn_status xnn_setup_truncation_nc_f16(
  xnn_operator_t truncation_op,
  size_t batch_size,
  const void* input,
  void* output,
  pthreadpool_t threadpool);

#ifndef XNN_NO_NCHW_OPERATORS

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
  xnn_caches_t caches,
  xnn_operator_t* convolution_op_out);

enum xnn_status xnn_setup_convolution2d_nchw_f16(
  xnn_operator_t convolution_op,
  size_t batch_size,
  size_t input_height,
  size_t input_width,
  const void* input,
  void* output,
  pthreadpool_t threadpool);

enum xnn_status xnn_create_depth_to_space_nchw2nhwc_x16(
  size_t output_channels,
  size_t input_channel_stride,
  size_t output_channel_stride,
  uint32_t block_size,
  uint32_t flags,
  xnn_operator_t* depth_to_space_op_out);

enum xnn_status xnn_setup_depth_to_space_nchw2nhwc_x16(
  xnn_operator_t depth_to_space_op,
  size_t batch_size,
  size_t input_height,
  size_t input_width,
  const void* input,
  void* output,
  pthreadpool_t threadpool);

enum xnn_status xnn_create_global_average_pooling_ncw_f16(
  size_t channels,
  float output_min,
  float output_max,
  uint32_t flags,
  xnn_operator_t* global_average_pooling_op_out);

enum xnn_status xnn_setup_global_average_pooling_ncw_f16(
  xnn_operator_t global_average_pooling_op,
  size_t batch_size,
  size_t width,
  const void* input,
  void* output,
  pthreadpool_t threadpool);

enum xnn_status xnn_create_resize_bilinear2d_nchw_f16(
  size_t channels,
  size_t input_pixel_stride,
  size_t output_pixel_stride,
  uint32_t flags,
  xnn_operator_t* resize_op_out);

enum xnn_status xnn_setup_resize_bilinear2d_nchw_f16(
  xnn_operator_t resize_op,
  size_t batch_size,
  size_t input_height,
  size_t input_width,
  size_t output_height,
  size_t output_width,
  const void* input,
  void* output,
  pthreadpool_t threadpool);

#endif  // XNN_NO_NCHW_OPERATORS

#endif  // XNN_NO_F16_OPERATORS

#ifndef XNN_NO_X16_OPERATORS

enum xnn_status xnn_create_constant_pad_nd_x16(
  const void* padding_value,
  uint32_t flags,
  xnn_operator_t* constant_pad_op_out);

enum xnn_status xnn_setup_constant_pad_nd_x16(
  xnn_operator_t constant_pad_op,
  size_t num_dims,
  const size_t* input_shape,
  const size_t* pre_padding,
  const size_t* post_padding,
  const void* input,
  void* output,
  pthreadpool_t threadpool);

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

enum xnn_status xnn_create_copy_nc_x16(
  size_t channels,
  size_t input_stride,
  size_t output_stride,
  uint32_t flags,
  xnn_operator_t* copy_op_out);

enum xnn_status xnn_setup_copy_nc_x16(
  xnn_operator_t copy_op,
  size_t batch_size,
  const void* input,
  void* output,
  pthreadpool_t threadpool);

enum xnn_status xnn_create_depth_to_space_nhwc_x16(
  size_t output_channels,
  size_t input_channel_stride,
  size_t output_channel_stride,
  uint32_t block_size,
  uint32_t flags,
  xnn_operator_t* depth_to_space_op_out);

enum xnn_status xnn_setup_depth_to_space_nhwc_x16(
  xnn_operator_t depth_to_space_op,
  size_t batch_size,
  size_t input_height,
  size_t input_width,
  const void* input,
  void* output,
  pthreadpool_t threadpool);

enum xnn_status xnn_create_slice_nd_x16(
  uint32_t flags,
  xnn_operator_t* slice_op_out);

enum xnn_status xnn_setup_slice_nd_x16(
  xnn_operator_t slice_op,
  size_t num_dims,
  const size_t* input_shape,
  const size_t* offsets,
  const size_t* sizes,
  const void* input,
  void* output,
  pthreadpool_t threadpool);

enum xnn_status xnn_create_space_to_depth_nhwc_x16(
  size_t input_channels,
  size_t input_channel_stride,
  size_t output_channel_stride,
  uint32_t block_size,
  uint32_t flags,
  xnn_operator_t* space_to_depth_op_out);

enum xnn_status xnn_setup_space_to_depth_nhwc_x16(
  xnn_operator_t space_to_depth_op,
  size_t batch_size,
  size_t input_height,
  size_t input_width,
  const void* input,
  void* output,
  pthreadpool_t threadpool);

enum xnn_status xnn_create_transpose_nd_x16(
    uint32_t flags,
    xnn_operator_t* transpose_op_out);

enum xnn_status xnn_setup_transpose_nd_x16(
    xnn_operator_t transpose_op,
    const void* input,
    void* output,
    const size_t num_dims,
    const size_t* input_shape,
    const size_t* output_perm,
    pthreadpool_t threadpool);

enum xnn_status xnn_run_transpose_nd_x16(
    const void* input,
    void* output,
    const size_t num_dims,
    const size_t* input_shape,
    const size_t* output_perm,
    uint32_t flags,
    pthreadpool_t threadpool);

#endif  // XNN_NO_X16_OPERATORS

#ifndef XNN_NO_QC8_OPERATORS

enum xnn_status xnn_create_convolution2d_nhwc_qc8(
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
  xnn_caches_t caches,
  xnn_operator_t* convolution_op_out);

enum xnn_status xnn_setup_convolution2d_nhwc_qc8(
  xnn_operator_t convolution_op,
  size_t batch_size,
  size_t input_height,
  size_t input_width,
  const int8_t* input,
  int8_t* output,
  pthreadpool_t threadpool);

#endif  // XNN_NO_QC8_OPERATORS

#ifndef XNN_NO_QS8_OPERATORS

enum xnn_status xnn_create_add_nd_qs8(
  int8_t input1_zero_point,
  float input1_scale,
  int8_t input2_zero_point,
  float input2_scale,
  int8_t output_zero_point,
  float output_scale,
  int8_t output_min,
  int8_t output_max,
  uint32_t flags,
  xnn_operator_t* add_op_out);

enum xnn_status xnn_setup_add_nd_qs8(
  xnn_operator_t add_op,
  size_t num_input1_dims,
  const size_t* input1_shape,
  size_t num_input2_dims,
  const size_t* input2_shape,
  const int8_t* input1,
  const int8_t* input2,
  int8_t* output,
  pthreadpool_t threadpool);

enum xnn_status xnn_run_add_nd_qs8(
  size_t num_input1_dims,
  const size_t* input1_shape,
  int8_t input1_zero_point,
  float input1_scale,
  size_t num_input2_dims,
  const size_t* input2_shape,
  int8_t input2_zero_point,
  float input2_scale,
  const int8_t* input1,
  const int8_t* input2,
  int8_t* output,
  int8_t output_zero_point,
  float output_scale,
  int8_t output_min,
  int8_t output_max,
  uint32_t flags,
  pthreadpool_t threadpool);

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
  xnn_caches_t caches,
  xnn_operator_t* convolution_op_out);

enum xnn_status xnn_setup_convolution2d_nhwc_qs8(
  xnn_operator_t convolution_op,
  size_t batch_size,
  size_t input_height,
  size_t input_width,
  const int8_t* input,
  int8_t* output,
  pthreadpool_t threadpool);

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
  xnn_caches_t caches,
  xnn_operator_t* deconvolution_op_out);

enum xnn_status xnn_setup_deconvolution2d_nhwc_qs8(
  xnn_operator_t deconvolution_op,
  size_t batch_size,
  size_t input_height,
  size_t input_width,
  uint32_t adjustment_height,
  uint32_t adjustment_width,
  const int8_t* input,
  int8_t* output,
  pthreadpool_t threadpool);

enum xnn_status xnn_create_elu_nc_qs8(
  size_t channels,
  size_t input_stride,
  size_t output_stride,
  float alpha,
  int8_t input_zero_point,
  float input_scale,
  int8_t output_zero_point,
  float output_scale,
  int8_t output_min,
  int8_t output_max,
  uint32_t flags,
  xnn_operator_t* elu_op_out);

enum xnn_status xnn_setup_elu_nc_qs8(
  xnn_operator_t elu_op,
  size_t batch_size,
  const int8_t* input,
  int8_t* output,
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
  xnn_caches_t caches,
  xnn_operator_t* fully_connected_op_out);

enum xnn_status xnn_setup_fully_connected_nc_qs8(
  xnn_operator_t fully_connected_op,
  size_t batch_size,
  const int8_t* input,
  int8_t* output,
  pthreadpool_t threadpool);

enum xnn_status xnn_create_global_average_pooling_nwc_qs8(
  size_t channels,
  size_t input_stride,
  size_t output_stride,
  int8_t input_zero_point,
  float input_scale,
  int8_t output_zero_point,
  float output_scale,
  int8_t output_min,
  int8_t output_max,
  uint32_t flags,
  xnn_operator_t* global_average_pooling_op_out);

enum xnn_status xnn_setup_global_average_pooling_nwc_qs8(
  xnn_operator_t global_average_pooling_op,
  size_t batch_size,
  size_t width,
  const int8_t* input,
  int8_t* output,
  pthreadpool_t threadpool);

enum xnn_status xnn_create_multiply_nd_qs8(
  int8_t input1_zero_point,
  float input1_scale,
  int8_t input2_zero_point,
  float input2_scale,
  int8_t output_zero_point,
  float output_scale,
  int8_t output_min,
  int8_t output_max,
  uint32_t flags,
  xnn_operator_t* multiply_op_out);

enum xnn_status xnn_setup_multiply_nd_qs8(
  xnn_operator_t multiply_op,
  size_t num_input1_dims,
  const size_t* input1_shape,
  size_t num_input2_dims,
  const size_t* input2_shape,
  const int8_t* input1,
  const int8_t* input2,
  int8_t* output,
  pthreadpool_t threadpool);

enum xnn_status xnn_run_multiply_nd_qs8(
  size_t num_input1_dims,
  const size_t* input1_shape,
  int8_t input1_zero_point,
  float input1_scale,
  size_t num_input2_dims,
  const size_t* input2_shape,
  int8_t input2_zero_point,
  float input2_scale,
  const int8_t* input1,
  const int8_t* input2,
  int8_t* output,
  int8_t output_zero_point,
  float output_scale,
  int8_t output_min,
  int8_t output_max,
  uint32_t flags,
  pthreadpool_t threadpool);

enum xnn_status xnn_create_leaky_relu_nc_qs8(
  size_t channels,
  size_t input_stride,
  size_t output_stride,
  float negative_slope,
  int8_t input_zero_point,
  float input_scale,
  int8_t output_zero_point,
  float output_scale,
  uint32_t flags,
  xnn_operator_t* leaky_relu_op_out);

enum xnn_status xnn_setup_leaky_relu_nc_qs8(
  xnn_operator_t leaky_relu_op,
  size_t batch_size,
  const int8_t* input,
  int8_t* output,
  pthreadpool_t threadpool);

enum xnn_status xnn_create_sigmoid_nc_qs8(
  size_t channels,
  size_t input_stride,
  size_t output_stride,
  int8_t input_zero_point,
  float input_scale,
  int8_t output_zero_point,
  float output_scale,
  int8_t output_min,
  int8_t output_max,
  uint32_t flags,
  xnn_operator_t* sigmoid_op_out);

enum xnn_status xnn_setup_sigmoid_nc_qs8(
  xnn_operator_t sigmoid_op,
  size_t batch_size,
  const int8_t* input,
  int8_t* output,
  pthreadpool_t threadpool);

enum xnn_status xnn_create_subtract_nd_qs8(
  int8_t input1_zero_point,
  float input1_scale,
  int8_t input2_zero_point,
  float input2_scale,
  int8_t output_zero_point,
  float output_scale,
  int8_t output_min,
  int8_t output_max,
  uint32_t flags,
  xnn_operator_t* subtract_op_out);

enum xnn_status xnn_setup_subtract_nd_qs8(
  xnn_operator_t subtract_op,
  size_t num_input1_dims,
  const size_t* input1_shape,
  size_t num_input2_dims,
  const size_t* input2_shape,
  const int8_t* input1,
  const int8_t* input2,
  int8_t* output,
  pthreadpool_t threadpool);

enum xnn_status xnn_run_subtract_nd_qs8(
  size_t num_input1_dims,
  const size_t* input1_shape,
  int8_t input1_zero_point,
  float input1_scale,
  size_t num_input2_dims,
  const size_t* input2_shape,
  int8_t input2_zero_point,
  float input2_scale,
  const int8_t* input1,
  const int8_t* input2,
  int8_t* output,
  int8_t output_zero_point,
  float output_scale,
  int8_t output_min,
  int8_t output_max,
  uint32_t flags,
  pthreadpool_t threadpool);

enum xnn_status xnn_create_tanh_nc_qs8(
  size_t channels,
  size_t input_stride,
  size_t output_stride,
  int8_t input_zero_point,
  float input_scale,
  int8_t output_zero_point,
  float output_scale,
  int8_t output_min,
  int8_t output_max,
  uint32_t flags,
  xnn_operator_t* tanh_op_out);

enum xnn_status xnn_setup_tanh_nc_qs8(
  xnn_operator_t tanh_op,
  size_t batch_size,
  const int8_t* input,
  int8_t* output,
  pthreadpool_t threadpool);

#endif  // XNN_NO_QS8_OPERATORS

#ifndef XNN_NO_QU8_OPERATORS

enum xnn_status xnn_create_add_nd_qu8(
  uint8_t input1_zero_point,
  float input1_scale,
  uint8_t input2_zero_point,
  float input2_scale,
  uint8_t output_zero_point,
  float output_scale,
  uint8_t output_min,
  uint8_t output_max,
  uint32_t flags,
  xnn_operator_t* add_op_out);

enum xnn_status xnn_setup_add_nd_qu8(
  xnn_operator_t add_op,
  size_t num_input1_dims,
  const size_t* input1_shape,
  size_t num_input2_dims,
  const size_t* input2_shape,
  const uint8_t* input1,
  const uint8_t* input2,
  uint8_t* output,
  pthreadpool_t threadpool);

enum xnn_status xnn_run_add_nd_qu8(
  size_t num_input1_dims,
  const size_t* input1_shape,
  uint8_t input1_zero_point,
  float input1_scale,
  size_t num_input2_dims,
  const size_t* input2_shape,
  uint8_t input2_zero_point,
  float input2_scale,
  const uint8_t* input1,
  const uint8_t* input2,
  uint8_t* output,
  uint8_t output_zero_point,
  float output_scale,
  uint8_t output_min,
  uint8_t output_max,
  uint32_t flags,
  pthreadpool_t threadpool);

enum xnn_status xnn_create_average_pooling2d_nhwc_qu8(
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

enum xnn_status xnn_setup_average_pooling2d_nhwc_qu8(
  xnn_operator_t average_pooling_op,
  size_t batch_size,
  size_t input_height,
  size_t input_width,
  const uint8_t* input,
  uint8_t* output,
  pthreadpool_t threadpool);

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
  xnn_caches_t caches,
  xnn_operator_t* convolution_op_out);

enum xnn_status xnn_setup_convolution2d_nhwc_qu8(
  xnn_operator_t convolution_op,
  size_t batch_size,
  size_t input_height,
  size_t input_width,
  const uint8_t* input,
  uint8_t* output,
  pthreadpool_t threadpool);

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
  xnn_caches_t caches,
  xnn_operator_t* deconvolution_op_out);

enum xnn_status xnn_setup_deconvolution2d_nhwc_qu8(
  xnn_operator_t deconvolution_op,
  size_t batch_size,
  size_t input_height,
  size_t input_width,
  uint32_t adjustment_height,
  uint32_t adjustment_width,
  const uint8_t* input,
  uint8_t* output,
  pthreadpool_t threadpool);

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
  xnn_caches_t caches,
  xnn_operator_t* fully_connected_op_out);

enum xnn_status xnn_setup_fully_connected_nc_qu8(
  xnn_operator_t fully_connected_op,
  size_t batch_size,
  const uint8_t* input,
  uint8_t* output,
  pthreadpool_t threadpool);

enum xnn_status xnn_create_global_average_pooling_nwc_qu8(
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

enum xnn_status xnn_setup_global_average_pooling_nwc_qu8(
  xnn_operator_t global_average_pooling_op,
  size_t batch_size,
  size_t width,
  const uint8_t* input,
  uint8_t* output,
  pthreadpool_t threadpool);

enum xnn_status xnn_create_leaky_relu_nc_qu8(
  size_t channels,
  size_t input_stride,
  size_t output_stride,
  float negative_slope,
  uint8_t input_zero_point,
  float input_scale,
  uint8_t output_zero_point,
  float output_scale,
  uint32_t flags,
  xnn_operator_t* leaky_relu_op_out);

enum xnn_status xnn_setup_leaky_relu_nc_qu8(
  xnn_operator_t leaky_relu_op,
  size_t batch_size,
  const uint8_t* input,
  uint8_t* output,
  pthreadpool_t threadpool);

enum xnn_status xnn_create_multiply_nd_qu8(
  uint8_t input1_zero_point,
  float input1_scale,
  uint8_t input2_zero_point,
  float input2_scale,
  uint8_t output_zero_point,
  float output_scale,
  uint8_t output_min,
  uint8_t output_max,
  uint32_t flags,
  xnn_operator_t* multiply_op_out);

enum xnn_status xnn_setup_multiply_nd_qu8(
  xnn_operator_t multiply_op,
  size_t num_input1_dims,
  const size_t* input1_shape,
  size_t num_input2_dims,
  const size_t* input2_shape,
  const uint8_t* input1,
  const uint8_t* input2,
  uint8_t* output,
  pthreadpool_t threadpool);

enum xnn_status xnn_run_multiply_nd_qu8(
  size_t num_input1_dims,
  const size_t* input1_shape,
  uint8_t input1_zero_point,
  float input1_scale,
  size_t num_input2_dims,
  const size_t* input2_shape,
  uint8_t input2_zero_point,
  float input2_scale,
  const uint8_t* input1,
  const uint8_t* input2,
  uint8_t* output,
  uint8_t output_zero_point,
  float output_scale,
  uint8_t output_min,
  uint8_t output_max,
  uint32_t flags,
  pthreadpool_t threadpool);

enum xnn_status xnn_create_sigmoid_nc_qu8(
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

enum xnn_status xnn_setup_sigmoid_nc_qu8(
  xnn_operator_t sigmoid_op,
  size_t batch_size,
  const uint8_t* input,
  uint8_t* output,
  pthreadpool_t threadpool);

enum xnn_status xnn_create_softmax_nc_qu8(
  size_t channels,
  size_t input_stride,
  size_t output_stride,
  float input_scale,
  uint8_t output_zero_point,
  float output_scale,
  uint32_t flags,
  xnn_operator_t* softmax_op_out);

enum xnn_status xnn_setup_softmax_nc_qu8(
  xnn_operator_t softmax_op,
  size_t batch_size,
  const uint8_t* input,
  uint8_t* output,
  pthreadpool_t threadpool);

enum xnn_status xnn_create_subtract_nd_qu8(
  uint8_t input1_zero_point,
  float input1_scale,
  uint8_t input2_zero_point,
  float input2_scale,
  uint8_t output_zero_point,
  float output_scale,
  uint8_t output_min,
  uint8_t output_max,
  uint32_t flags,
  xnn_operator_t* subtract_op_out);

enum xnn_status xnn_setup_subtract_nd_qu8(
  xnn_operator_t subtract_op,
  size_t num_input1_dims,
  const size_t* input1_shape,
  size_t num_input2_dims,
  const size_t* input2_shape,
  const uint8_t* input1,
  const uint8_t* input2,
  uint8_t* output,
  pthreadpool_t threadpool);

enum xnn_status xnn_run_subtract_nd_qu8(
  size_t num_input1_dims,
  const size_t* input1_shape,
  uint8_t input1_zero_point,
  float input1_scale,
  size_t num_input2_dims,
  const size_t* input2_shape,
  uint8_t input2_zero_point,
  float input2_scale,
  const uint8_t* input1,
  const uint8_t* input2,
  uint8_t* output,
  uint8_t output_zero_point,
  float output_scale,
  uint8_t output_min,
  uint8_t output_max,
  uint32_t flags,
  pthreadpool_t threadpool);

enum xnn_status xnn_create_tanh_nc_qu8(
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
  xnn_operator_t* tanh_op_out);

enum xnn_status xnn_setup_tanh_nc_qu8(
  xnn_operator_t tanh_op,
  size_t batch_size,
  const uint8_t* input,
  uint8_t* output,
  pthreadpool_t threadpool);

#endif  // XNN_NO_QU8_OPERATORS

#ifndef XNN_NO_S8_OPERATORS

enum xnn_status xnn_create_clamp_nc_s8(
  size_t channels,
  size_t input_stride,
  size_t output_stride,
  int8_t output_min,
  int8_t output_max,
  uint32_t flags,
  xnn_operator_t* clamp_op_out);

enum xnn_status xnn_setup_clamp_nc_s8(
  xnn_operator_t clamp_op,
  size_t batch_size,
  const int8_t* input,
  int8_t* output,
  pthreadpool_t threadpool);

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
  size_t channels,
  size_t input_pixel_stride,
  size_t output_pixel_stride,
  int8_t output_min,
  int8_t output_max,
  uint32_t flags,
  xnn_operator_t* max_pooling_op_out);

enum xnn_status xnn_setup_max_pooling2d_nhwc_s8(
  xnn_operator_t max_pooling_op,
  size_t batch_size,
  size_t input_height,
  size_t input_width,
  const int8_t* input,
  int8_t* output,
  pthreadpool_t threadpool);

enum xnn_status xnn_create_resize_bilinear2d_nhwc_s8(
  size_t channels,
  size_t input_pixel_stride,
  size_t output_pixel_stride,
  uint32_t flags,
  xnn_operator_t* resize_op_out);

enum xnn_status xnn_setup_resize_bilinear2d_nhwc_s8(
  xnn_operator_t resize_op,
  size_t batch_size,
  size_t input_height,
  size_t input_width,
  size_t output_height,
  size_t output_width,
  const int8_t* input,
  int8_t* output,
  pthreadpool_t threadpool);

#endif  // XNN_NO_S8_OPERATORS

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

enum xnn_status xnn_create_resize_bilinear2d_nhwc_u8(
  size_t channels,
  size_t input_pixel_stride,
  size_t output_pixel_stride,
  uint32_t flags,
  xnn_operator_t* resize_op_out);

enum xnn_status xnn_setup_resize_bilinear2d_nhwc_u8(
  xnn_operator_t resize_op,
  size_t batch_size,
  size_t input_height,
  size_t input_width,
  size_t output_height,
  size_t output_width,
  const uint8_t* input,
  uint8_t* output,
  pthreadpool_t threadpool);

#endif  // XNN_NO_U8_OPERATORS

#ifndef XNN_NO_X8_OPERATORS

enum xnn_status xnn_create_copy_nc_x8(
  size_t channels,
  size_t input_stride,
  size_t output_stride,
  uint32_t flags,
  xnn_operator_t* copy_op_out);

enum xnn_status xnn_setup_copy_nc_x8(
  xnn_operator_t copy_op,
  size_t batch_size,
  const void* input,
  void* output,
  pthreadpool_t threadpool);

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

enum xnn_status xnn_create_constant_pad_nd_x8(
  const void* padding_value,
  uint32_t flags,
  xnn_operator_t* constant_pad_op_out);

enum xnn_status xnn_setup_constant_pad_nd_x8(
  xnn_operator_t constant_pad_op,
  size_t num_dims,
  const size_t* input_shape,
  const size_t* pre_padding,
  const size_t* post_padding,
  const void* input,
  void* output,
  pthreadpool_t threadpool);

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

enum xnn_status xnn_create_depth_to_space_nhwc_x8(
  size_t output_channels,
  size_t input_channel_stride,
  size_t output_channel_stride,
  uint32_t block_size,
  uint32_t flags,
  xnn_operator_t* depth_to_space_op_out);

enum xnn_status xnn_setup_depth_to_space_nhwc_x8(
  xnn_operator_t depth_to_space_op,
  size_t batch_size,
  size_t input_height,
  size_t input_width,
  const void* input,
  void* output,
  pthreadpool_t threadpool);

enum xnn_status xnn_create_slice_nd_x8(
  uint32_t flags,
  xnn_operator_t* slice_op_out);

enum xnn_status xnn_setup_slice_nd_x8(
  xnn_operator_t slice_op,
  size_t num_dims,
  const size_t* input_shape,
  const size_t* offsets,
  const size_t* sizes,
  const void* input,
  void* output,
  pthreadpool_t threadpool);

enum xnn_status xnn_create_space_to_depth_nhwc_x8(
  size_t input_channels,
  size_t input_channel_stride,
  size_t output_channel_stride,
  uint32_t block_size,
  uint32_t flags,
  xnn_operator_t* space_to_depth_op_out);

enum xnn_status xnn_setup_space_to_depth_nhwc_x8(
  xnn_operator_t space_to_depth_op,
  size_t batch_size,
  size_t input_height,
  size_t input_width,
  const void* input,
  void* output,
  pthreadpool_t threadpool);

enum xnn_status xnn_create_transpose_nd_x8(
    uint32_t flags,
    xnn_operator_t* transpose_op_out);

enum xnn_status xnn_setup_transpose_nd_x8(
    xnn_operator_t transpose_op,
    const void* input,
    void* output,
    const size_t num_dims,
    const size_t* input_shape,
    const size_t* output_perm,
    pthreadpool_t threadpool);

enum xnn_status xnn_run_transpose_nd_x8(
    const void* input,
    void* output,
    const size_t num_dims,
    const size_t* input_shape,
    const size_t* output_perm,
    uint32_t flags,
    pthreadpool_t threadpool);

#endif  // XNN_NO_X8_OPERATORS

#ifndef XNN_NO_CVT_OPERATORS

enum xnn_status xnn_create_convert_nc_f16_f32(
  size_t channels,
  size_t input_stride,
  size_t output_stride,
  uint32_t flags,
  xnn_operator_t* convert_op_out);

enum xnn_status xnn_setup_convert_nc_f16_f32(
  xnn_operator_t convert_op,
  size_t batch_size,
  const void* input,
  float* output,
  pthreadpool_t threadpool);

enum xnn_status xnn_run_convert_nc_f16_f32(
  size_t channels,
  size_t input_stride,
  size_t output_stride,
  size_t batch_size,
  const void* input,
  float* output,
  uint32_t flags,
  pthreadpool_t threadpool);

enum xnn_status xnn_create_convert_nc_f32_f16(
  size_t channels,
  size_t input_stride,
  size_t output_stride,
  uint32_t flags,
  xnn_operator_t* convert_op_out);

enum xnn_status xnn_setup_convert_nc_f32_f16(
  xnn_operator_t convert_op,
  size_t batch_size,
  const float* input,
  void* output,
  pthreadpool_t threadpool);

enum xnn_status xnn_run_convert_nc_f32_f16(
  size_t channels,
  size_t input_stride,
  size_t output_stride,
  size_t batch_size,
  const float* input,
  void* output,
  uint32_t flags,
  pthreadpool_t threadpool);

enum xnn_status xnn_create_convert_nc_f32_qs8(
  size_t channels,
  size_t input_stride,
  size_t output_stride,
  float output_scale,
  int8_t output_zero_point,
  int8_t output_min,
  int8_t output_max,
  uint32_t flags,
  xnn_operator_t* convert_op_out);

enum xnn_status xnn_setup_convert_nc_f32_qs8(
  xnn_operator_t convert_op,
  size_t batch_size,
  const float* input,
  int8_t* output,
  pthreadpool_t threadpool);

enum xnn_status xnn_run_convert_nc_f32_qs8(
  size_t channels,
  size_t input_stride,
  size_t output_stride,
  size_t batch_size,
  const float* input,
  int8_t* output,
  float output_scale,
  int8_t output_zero_point,
  uint32_t flags,
  pthreadpool_t threadpool);

enum xnn_status xnn_create_convert_nc_f32_qu8(
  size_t channels,
  size_t input_stride,
  size_t output_stride,
  float output_scale,
  uint8_t output_zero_point,
  uint8_t output_min,
  uint8_t output_max,
  uint32_t flags,
  xnn_operator_t* convert_op_out);

enum xnn_status xnn_setup_convert_nc_f32_qu8(
  xnn_operator_t convert_op,
  size_t batch_size,
  const float* input,
  uint8_t* output,
  pthreadpool_t threadpool);

enum xnn_status xnn_run_convert_nc_f32_qu8(
  size_t channels,
  size_t input_stride,
  size_t output_stride,
  size_t batch_size,
  const float* input,
  uint8_t* output,
  float output_scale,
  uint8_t output_zero_point,
  uint32_t flags,
  pthreadpool_t threadpool);

enum xnn_status xnn_create_convert_nc_qs8(
  size_t channels,
  size_t input_stride,
  size_t output_stride,
  float input_scale,
  int8_t input_zero_point,
  float output_scale,
  int8_t output_zero_point,
  uint32_t flags,
  xnn_operator_t* convert_op_out);

enum xnn_status xnn_setup_convert_nc_qs8(
  xnn_operator_t convert_op,
  size_t batch_size,
  const int8_t* input,
  int8_t* output,
  pthreadpool_t threadpool);

enum xnn_status xnn_create_convert_nc_qs8_f32(
  size_t channels,
  size_t input_stride,
  size_t output_stride,
  float input_scale,
  int8_t input_zero_point,
  uint32_t flags,
  xnn_operator_t* convert_op_out);

enum xnn_status xnn_setup_convert_nc_qs8_f32(
  xnn_operator_t convert_op,
  size_t batch_size,
  const int8_t* input,
  float* output,
  pthreadpool_t threadpool);

enum xnn_status xnn_run_convert_nc_qs8_f32(
  size_t channels,
  size_t input_stride,
  size_t output_stride,
  size_t batch_size,
  const int8_t* input,
  float* output,
  float input_scale,
  int8_t input_zero_point,
  uint32_t flags,
  pthreadpool_t threadpool);

enum xnn_status xnn_create_convert_nc_qu8(
  size_t channels,
  size_t input_stride,
  size_t output_stride,
  float input_scale,
  uint8_t input_zero_point,
  float output_scale,
  uint8_t output_zero_point,
  uint32_t flags,
  xnn_operator_t* convert_op_out);

enum xnn_status xnn_setup_convert_nc_qu8(
  xnn_operator_t convert_op,
  size_t batch_size,
  const uint8_t* input,
  uint8_t* output,
  pthreadpool_t threadpool);

enum xnn_status xnn_create_convert_nc_qu8_f32(
  size_t channels,
  size_t input_stride,
  size_t output_stride,
  float input_scale,
  uint8_t input_zero_point,
  uint32_t flags,
  xnn_operator_t* convert_op_out);

enum xnn_status xnn_setup_convert_nc_qu8_f32(
  xnn_operator_t convert_op,
  size_t batch_size,
  const uint8_t* input,
  float* output,
  pthreadpool_t threadpool);

enum xnn_status xnn_run_convert_nc_qu8_f32(
  size_t channels,
  size_t input_stride,
  size_t output_stride,
  size_t batch_size,
  const uint8_t* input,
  float* output,
  float input_scale,
  uint8_t input_zero_point,
  uint32_t flags,
  pthreadpool_t threadpool);

#endif  // XNN_NO_CVT_OPERATORS

#ifdef __cplusplus
}  // extern "C"
#endif
