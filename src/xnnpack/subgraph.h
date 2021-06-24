// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <stddef.h>
#include <stdint.h>

#include <xnnpack.h>

#define XNN_MAX_INPUTS 3
#define XNN_MAX_OUTPUTS 2

#define XNN_MAX_RUNTIME_INPUTS 2
#define XNN_MAX_RUNTIME_OUTPUTS 2

#define XNN_INVALID_NODE_ID UINT32_MAX

#ifdef __cplusplus
extern "C" {
#endif

struct xnn_shape {
  size_t num_dims;
  size_t dim[XNN_MAX_TENSOR_DIMS];
};

enum xnn_value_type {
  xnn_value_type_invalid = 0,
  xnn_value_type_dense_tensor = 1,
};

enum xnn_layout_type {
  xnn_layout_type_nhwc = 0,
  xnn_layout_type_nchw = 1,
};

/// Abstraction for a collections of elements produced and consumed by nodes.
struct xnn_value {
  /// Unique ID for the value.
  uint32_t id;
  /// Type of the collection of elements.
  ///
  /// Currently only dense tensors are supported.
  /// Other types (e.g. sparse tensors) might be supported in the future.
  enum xnn_value_type type;
  /// Type of elements in the collection.
  enum xnn_datatype datatype;
  /// Per-value quantization parameters.
  struct {
    /// Offset from zero of the quantized elements.
    int32_t zero_point;
    union {
      /// Multiplication factor to convert quantized elements to real representation.
      float scale;
      struct {
        /// Per-channel multiplication factor to convert quantized elements to real representation.
        const float* channelwise_scale;
        /// Index of the channel dimension with per-channel quantization parameters.
        size_t channel_dimension;
      };
    };
  } quantization;
  /// Tensor shape.
  struct xnn_shape shape;
  /// Binary features of the tensor. Supported values are any combination of:
  /// - XNN_VALUE_FLAG_EXTERNAL_INPUT
  /// - XNN_VALUE_FLAG_EXTERNAL_OUTPUT
  uint32_t flags;
  /// Static initialization data. Must be null for non-static values.
  const void* data;
  /// Index of the Subgraph node that produced the value, or XNN_INVALID_NODE_ID is the Value is an external input.
  uint32_t producer;
  /// Index of the first Node that consume the value, or XNN_INVALID_NODE_ID if the Value has no consumers within the
  /// graph (e.g. Value is an external output).
  uint32_t first_consumer;
  /// Number of Nodes that consume the value.
  /// If multiple inputs in a Node refer to this Value as input, the Node is counted as consumer multiple times.
  /// If the Value is an external output, it counts as having an extra consumer.
  uint32_t num_consumers;
  uint32_t num_nchw_compatible_consumers;
  enum xnn_layout_type layout;
};

struct xnn_blob {
  /// Size in bytes.
  size_t size;
  /// Data pointer.
  void* data;
  bool external;
};

enum xnn_node_type {
  xnn_node_type_invalid = 0,
  xnn_node_type_abs,
  xnn_node_type_add2,
  xnn_node_type_argmax_pooling_2d,
  xnn_node_type_average_pooling_2d,
  xnn_node_type_bankers_rounding,
  xnn_node_type_ceiling,
  xnn_node_type_clamp,
  xnn_node_type_convolution_2d,
  xnn_node_type_deconvolution_2d,
  xnn_node_type_depthwise_convolution_2d,
  xnn_node_type_depth_to_space,
  xnn_node_type_divide,
  xnn_node_type_elu,
  xnn_node_type_fully_connected,
  xnn_node_type_floor,
  xnn_node_type_global_average_pooling_2d,
  xnn_node_type_hardswish,
  xnn_node_type_leaky_relu,
  xnn_node_type_max_pooling_2d,
  xnn_node_type_maximum2,
  xnn_node_type_minimum2,
  xnn_node_type_multiply2,
  xnn_node_type_negate,
  xnn_node_type_prelu,
  xnn_node_type_sigmoid,
  xnn_node_type_softmax,
  xnn_node_type_static_constant_pad,
  xnn_node_type_static_reshape,
  xnn_node_type_static_resize_bilinear_2d,
  xnn_node_type_square,
  xnn_node_type_square_root,
  xnn_node_type_squared_difference,
  xnn_node_type_subtract,
  xnn_node_type_unpooling_2d,
};

struct xnn_node {
  enum xnn_node_type type;
  uint32_t id;
  /// Static parameters of the operator node.
  union {
    struct {
      uint32_t input_padding_top;
      uint32_t input_padding_right;
      uint32_t input_padding_bottom;
      uint32_t input_padding_left;
      uint32_t kernel_height;
      uint32_t kernel_width;
      uint32_t subsampling_height;
      uint32_t subsampling_width;
      uint32_t dilation_height;
      uint32_t dilation_width;
      uint32_t groups;
      size_t group_input_channels;
      size_t group_output_channels;
    } convolution_2d;
    struct {
      uint32_t padding_top;
      uint32_t padding_right;
      uint32_t padding_bottom;
      uint32_t padding_left;
      uint32_t adjustment_height;
      uint32_t adjustment_width;
      uint32_t kernel_height;
      uint32_t kernel_width;
      uint32_t upsampling_height;
      uint32_t upsampling_width;
      uint32_t dilation_height;
      uint32_t dilation_width;
      uint32_t groups;
      size_t group_input_channels;
      size_t group_output_channels;
    } deconvolution_2d;
    struct {
      uint32_t input_padding_top;
      uint32_t input_padding_right;
      uint32_t input_padding_bottom;
      uint32_t input_padding_left;
      uint32_t kernel_height;
      uint32_t kernel_width;
      uint32_t subsampling_height;
      uint32_t subsampling_width;
      uint32_t dilation_height;
      uint32_t dilation_width;
      uint32_t depth_multiplier;
      size_t input_channels;
    } depthwise_convolution_2d;
    struct {
      uint32_t block_size;
    } depth_to_space;
    struct {
      uint32_t padding_top;
      uint32_t padding_right;
      uint32_t padding_bottom;
      uint32_t padding_left;
      uint32_t pooling_height;
      uint32_t pooling_width;
      uint32_t stride_height;
      uint32_t stride_width;
      uint32_t dilation_height;
      uint32_t dilation_width;
    } pooling_2d;
    struct {
      float alpha;
    } elu;
    struct {
      float negative_slope;
    } leaky_relu;
    struct {
      size_t pre_paddings[XNN_MAX_TENSOR_DIMS];
      size_t post_paddings[XNN_MAX_TENSOR_DIMS];
      uint32_t padding_value;
    } static_pad;
    struct {
      struct xnn_shape new_shape;
    } static_reshape;
    struct {
      size_t new_height;
      size_t new_width;
    } static_resize;
  } params;
  struct {
    float output_min;
    float output_max;
  } activation;
  /// Value IDs for node inputs.
  uint32_t inputs[XNN_MAX_INPUTS];
  uint32_t num_inputs;
  /// Value IDs for node outputs.
  uint32_t outputs[XNN_MAX_OUTPUTS];
  uint32_t num_outputs;
  uint32_t flags;
  uint32_t layout_flags;
  uint32_t cluster_leader;
  // Number of filter parameters in all 1x1 Convolutions of the sparse cluster.
  // This value is properly initialized only in sparse inference analysis of 1x1 Convolutions.
  size_t num_params;
  // Number of zero filter parameters in all 1x1 Convolutions of the sparse cluster.
  // This value is properly initialized only in sparse inference analysis of 1x1 Convolutions.
  size_t num_zeroes;
};

struct xnn_operator_data {
  xnn_operator_t operator_object;
  size_t batch_size;
  size_t input_height;
  size_t input_width;
  size_t output_height;
  size_t output_width;
  struct xnn_shape shape1;
  struct xnn_shape shape2;
  size_t pre_paddings[XNN_MAX_TENSOR_DIMS];
  size_t post_paddings[XNN_MAX_TENSOR_DIMS];
  uint32_t adjustment_height;
  uint32_t adjustment_width;
  uint32_t inputs[XNN_MAX_RUNTIME_INPUTS];
  uint32_t outputs[XNN_MAX_RUNTIME_OUTPUTS];
};

struct xnn_subgraph {
  /// Number of Value IDs reserved for communication with external graph representation.
  /// Values created during subgraph transformation avoid using IDs in [0, reserved_value_ids-1] range.
  uint32_t external_value_ids;

  uint32_t num_reserved_values;
  uint32_t num_values;
  struct xnn_value* values;

  uint32_t num_reserved_nodes;
  uint32_t num_nodes;
  struct xnn_node* nodes;
};

/// Runtime is a combination of an execution plan for subgraph Nodes and a memory manager for subgraph Values.
struct xnn_runtime {
  uint32_t num_external_values;

  /// List of operators in the execution plan, in execution order.
  struct xnn_operator_data* opdata;
  /// Number of operators in the execution plan.
  size_t num_ops;

  struct xnn_blob* blobs;
  size_t num_blobs;

  void* workspace;

  pthreadpool_t threadpool;
};

struct xnn_value* xnn_subgraph_new_internal_value(xnn_subgraph_t subgraph);

struct xnn_node* xnn_subgraph_new_node(xnn_subgraph_t subgraph);

size_t xnn_tensor_get_size(
  xnn_subgraph_t subgraph,
  uint32_t value_id);

enum xnn_status xnn_subgraph_optimize(xnn_subgraph_t subgraph, uint32_t flags);

void xnn_subgraph_rewrite_for_nchw(xnn_subgraph_t subgraph);

void xnn_node_clear(struct xnn_node* node);
void xnn_value_clear(struct xnn_value* value);


#ifdef __cplusplus
}  // extern "C"
#endif
