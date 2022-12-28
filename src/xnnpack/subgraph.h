// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <stddef.h>
#include <stdint.h>

#include <xnnpack.h>
#include <xnnpack/common.h>
#include <xnnpack/cache.h>
#include <xnnpack/node-type.h>

#if defined(EMSCRIPTEN)
#include <emscripten/emscripten.h>
#elif XNN_PLATFORM_WINDOWS
#include <windows.h>
#else
#include <time.h>
#endif

#define XNN_MAX_INPUTS 4
#define XNN_MAX_OUTPUTS 4

#define XNN_MAX_RUNTIME_INPUTS 4
#define XNN_MAX_RUNTIME_OUTPUTS 4

#define XNN_INVALID_NODE_ID UINT32_MAX

#define XNN_MAX_OPERATOR_OBJECTS 4

/// Disable fusion of nodes in subgraph. Fusion is enabled by default, set this flag to turn it off.
#define XNN_FLAG_NO_OPERATOR_FUSION 0x80000000

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
  /// - XNN_VALUE_FLAG_PERSISTENT
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
  /// Set during analysis in xnn_subgraph_rewrite_for_fp16.
  /// Indicates that this value should be converted to FP16.
  bool fp16_compatible;
  /// Set during analysis in xnn_subgraph_rewrite_for_fp16.
  /// Indicates Value ID of the FP16 variant of this Value.
  uint32_t fp16_id;
  /// Set during analysis in xnn_subgraph_rewrite_for_fp16.
  /// Indicates Value ID of the FP32 variant of this Value.
  uint32_t fp32_id;
};


XNN_INLINE bool xnn_value_is_external(const struct xnn_value* value) {
  return (value->flags & (XNN_VALUE_FLAG_EXTERNAL_INPUT | XNN_VALUE_FLAG_EXTERNAL_OUTPUT)) != 0;
}

XNN_INLINE bool xnn_value_is_external_output(const struct xnn_value* value) {
  return (value->flags & XNN_VALUE_FLAG_EXTERNAL_OUTPUT) != 0;
}

XNN_INLINE bool xnn_value_is_external_input(const struct xnn_value* value) {
  return (value->flags & XNN_VALUE_FLAG_EXTERNAL_INPUT) != 0;
}

XNN_INLINE bool xnn_value_is_internal(const struct xnn_value* value) {
  return (
    (value->flags & (XNN_VALUE_FLAG_EXTERNAL_INPUT | XNN_VALUE_FLAG_EXTERNAL_OUTPUT | XNN_VALUE_FLAG_PERSISTENT)) == 0);
}

XNN_INLINE bool xnn_value_is_persistent(const struct xnn_value* value) {
  return (value->flags & XNN_VALUE_FLAG_PERSISTENT) != 0;
}

XNN_INLINE bool xnn_value_is_valid(const struct xnn_value* value) {
  return value->type != xnn_value_type_invalid;
}

XNN_INLINE bool xnn_value_is_static(const struct xnn_value* value) {
  return value->data != NULL;
}

enum xnn_allocation_type {
  xnn_allocation_type_invalid = 0,
  /// Static data that is provided by caller, needs to outlive the xnn_runtime.
  xnn_allocation_type_static,
  /// Lives in XNNPACK-managed internal workspace.
  xnn_allocation_type_workspace,
  /// Non-static data that is external to the runtime, provided by caller, specified in xnn_setup_runtime.
  xnn_allocation_type_external,
  // Persistent data is internal to XNNPACK-managed workspace, but shared by multiple runtime/subgraph.
  xnn_allocation_type_persistent,
};

struct xnn_blob {
  /// Size in bytes.
  size_t size;
  /// Data pointer.
  void* data;
  enum xnn_allocation_type allocation_type;
};

struct xnn_node;
struct xnn_operator_data;

typedef enum xnn_status (*xnn_create_operator_fn)(
  const struct xnn_node* node,
  const struct xnn_value* values,
  size_t num_values,
  struct xnn_operator_data* opdata,
  const struct xnn_caches* caches);

typedef enum xnn_status (*xnn_setup_operator_fn)(
  const struct xnn_operator_data* opdata,
  const struct xnn_blob* blobs,
  size_t num_blobs,
  pthreadpool_t threadpool);

enum xnn_compute_type {
  xnn_compute_type_invalid = 0,
  xnn_compute_type_fp32,
  xnn_compute_type_fp16,
  xnn_compute_type_qc8,
  xnn_compute_type_qs8,
  xnn_compute_type_qu8,
  xnn_compute_type_fp32_to_fp16,
  xnn_compute_type_fp32_to_qs8,
  xnn_compute_type_fp32_to_qu8,
  xnn_compute_type_fp16_to_fp32,
  xnn_compute_type_qs8_to_fp32,
  xnn_compute_type_qu8_to_fp32,
};

struct xnn_node {
  enum xnn_node_type type;
  uint32_t id;
  enum xnn_compute_type compute_type;
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
    struct {
      size_t axis;
    } concatenate;
    struct {
      size_t axis;
    } even_split;
    struct {
      size_t perm[XNN_MAX_TENSOR_DIMS];
      size_t num_dims;
    } transpose;
    struct {
      size_t num_dims;
      size_t offsets[XNN_MAX_TENSOR_DIMS];
      size_t sizes[XNN_MAX_TENSOR_DIMS];
    } slice;
    struct {
      uint32_t block_size;
    } space_to_depth_2d;
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
  // Factory function to create an operator object from the node.
  xnn_create_operator_fn create;
  // Function to setup an operator using opdata.
  xnn_setup_operator_fn setup;
};

#ifdef __MACH__
typedef uint64_t xnn_timestamp;
#elif __EMSCRIPTEN__
typedef double xnn_timestamp;
#elif XNN_PLATFORM_WINDOWS
typedef LARGE_INTEGER xnn_timestamp;
#else
typedef struct timespec xnn_timestamp;
#endif

struct xnn_operator_data {
  xnn_operator_t operator_objects[XNN_MAX_OPERATOR_OBJECTS];
  xnn_setup_operator_fn setup;
  size_t batch_size;
  size_t input_height;
  size_t input_width;
  size_t output_height;
  size_t output_width;
  struct xnn_shape shape1;
  struct xnn_shape shape2;
  size_t pre_paddings[XNN_MAX_TENSOR_DIMS];
  size_t post_paddings[XNN_MAX_TENSOR_DIMS];
  // TODO(zhin): merge this with pre_paddings/post_paddings to reduce size of this struct.
  size_t offsets[XNN_MAX_TENSOR_DIMS];
  size_t sizes[XNN_MAX_TENSOR_DIMS];
  uint32_t adjustment_height;
  uint32_t adjustment_width;
  uint32_t inputs[XNN_MAX_RUNTIME_INPUTS];
  uint32_t outputs[XNN_MAX_RUNTIME_OUTPUTS];
  xnn_timestamp end_ts[XNN_MAX_OPERATOR_OBJECTS];
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

  struct xnn_workspace* workspace;
  struct xnn_runtime* next_workspace_user;

#if XNN_PLATFORM_JIT
  struct xnn_code_cache code_cache;
#endif // XNN_PLATFORM_JIT

  pthreadpool_t threadpool;

  bool profiling;
  // The start timestamp of the first operator in the subgraph. This is set when profiling is true.
  xnn_timestamp start_ts;
};

struct xnn_value* xnn_subgraph_new_internal_value(xnn_subgraph_t subgraph);

struct xnn_node* xnn_subgraph_new_node(xnn_subgraph_t subgraph);

void xnn_subgraph_add_nodes(xnn_subgraph_t subgraph, size_t num_nodes);

size_t xnn_tensor_get_size(
  xnn_subgraph_t subgraph,
  uint32_t value_id);

// Product of all shape dimensions
size_t xnn_shape_multiply_all_dims(
  const struct xnn_shape shape[1]);

// Product of all shape dimensions, except for the specified number of the last dimensions
size_t xnn_shape_multiply_batch_dims(
  const struct xnn_shape shape[1], size_t num_nonbatch_dims);

// Product of all shape dimensions, except for the last (channel) one
size_t xnn_shape_multiply_non_channel_dims(
  const struct xnn_shape shape[1]);

enum xnn_status xnn_subgraph_optimize(xnn_subgraph_t subgraph, uint32_t flags);

void xnn_subgraph_rewrite_for_nchw(xnn_subgraph_t subgraph);
// Rewrites subgraph for FP16, returns true if success, false if rewrite failed.
bool xnn_subgraph_rewrite_for_fp16(xnn_subgraph_t subgraph);

void xnn_node_clear(struct xnn_node* node);
void xnn_value_clear(struct xnn_value* value);

void xnn_value_copy(struct xnn_value* dst_value, const struct xnn_value* src_value);

void xnn_init_convert_node(
  struct xnn_node* node,
  enum xnn_compute_type compute_type,
  uint32_t input_id,
  uint32_t output_id,
  uint32_t flags);

struct xnn_workspace {
  void* data;
  size_t size;
  struct xnn_runtime* first_user;
  // Workspace will be destroyed in xnn_delete_runtime or xnn_delete_workspace if num_users reaches 0.
  size_t ref_count;
  size_t persistent_size;
};

void xnn_subgraph_analyze_consumers_and_producers(xnn_subgraph_t subgraph);

#ifdef __cplusplus
}  // extern "C"
#endif
