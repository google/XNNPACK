// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <assert.h>
#include <stddef.h>
#include <stdint.h>

#include "include/xnnpack.h"
#include "src/xnnpack/allocation-type.h"
#include "src/xnnpack/common.h"
#include "src/xnnpack/math.h"
#include "src/xnnpack/node-type.h"
#include "src/xnnpack/subgraph_types.h"
#include <pthreadpool.h>

#if defined(EMSCRIPTEN)
#include <emscripten/emscripten.h>
#elif XNN_PLATFORM_WINDOWS
#include <windows.h>
#else
#include <time.h>
#endif

#ifdef __cplusplus
extern "C" {
#endif

#ifdef XNN_SLINKY_AVAILABLE
/// Slinky interface -- unused unless XNN_FLAG_SLINKY_ENABLED is set
enum xnn_status slinky_init_pipeline(xnn_runtime_t runtime);
void slinky_setup_pipeline(xnn_runtime_t runtime);
void slinky_destroy_pipeline(xnn_runtime_t runtime);
enum xnn_status slinky_reshape_pipeline(xnn_runtime_t runtime);
enum xnn_status slinky_invoke_pipeline(xnn_runtime_t runtime);
#endif  // XNN_SLINKY_AVAILABLE

XNN_INLINE static bool xnn_value_is_external(const struct xnn_value* value) {
  return (value->flags & (XNN_VALUE_FLAG_EXTERNAL_INPUT |
                          XNN_VALUE_FLAG_EXTERNAL_OUTPUT)) != 0;
}

XNN_INLINE static bool xnn_value_is_external_output(
    const struct xnn_value* value) {
  return (value->flags & XNN_VALUE_FLAG_EXTERNAL_OUTPUT) != 0;
}

XNN_INLINE static bool xnn_value_is_external_input(
    const struct xnn_value* value) {
  return (value->flags & XNN_VALUE_FLAG_EXTERNAL_INPUT) != 0;
}

XNN_INLINE static bool xnn_value_is_internal(const struct xnn_value* value) {
  return ((value->flags &
           (XNN_VALUE_FLAG_EXTERNAL_INPUT | XNN_VALUE_FLAG_EXTERNAL_OUTPUT |
            XNN_VALUE_FLAG_PERSISTENT)) == 0);
}

XNN_INLINE static bool xnn_value_is_persistent(const struct xnn_value* value) {
  // Treat a value that is both input and output as persistent.
  const uint32_t input_output =
      XNN_VALUE_FLAG_EXTERNAL_INPUT | XNN_VALUE_FLAG_EXTERNAL_OUTPUT;
  return
      (value->flags & input_output) == input_output ||
      value->allocation_type == xnn_allocation_type_persistent;
}

XNN_INLINE static bool xnn_value_is_valid(const struct xnn_value* value) {
  return value->type != xnn_value_type_invalid;
}

XNN_INLINE static bool xnn_value_is_static(const struct xnn_value* value) {
  return value->allocation_type == xnn_allocation_type_static;
}

enum xnn_status xnn_insert_clamp_node(xnn_subgraph_t subgraph, float output_min,
                                      float output_max, struct xnn_node* node);

enum xnn_status xnn_insert_pack_lh_node(xnn_subgraph_t subgraph,
                                        uint32_t input_id, uint32_t* new_id);

struct xnn_value* xnn_subgraph_new_internal_value(xnn_subgraph_t subgraph);

struct xnn_node* xnn_subgraph_new_node(xnn_subgraph_t subgraph);

enum xnn_status xnn_subgraph_add_nodes(xnn_subgraph_t subgraph,
                                       size_t num_nodes);

// Get size of the tensor in bytes (based on dimensions of tensor).
size_t xnn_tensor_get_size(const struct xnn_value* value);

size_t xnn_tensor_get_size_by_id(xnn_subgraph_t subgraph, uint32_t value_id);

XNN_INLINE static size_t xnn_get_rounded_size(size_t size) {
  // We round it to XNN_EXTRA_BYTES to ensure that we can read more than the
  // actual size of the tensor, and round it to allocation alignment to ensure
  // that all tensors and operator workspaces are aligned correctly.
  return round_up_po2(round_up_po2(size, XNN_EXTRA_BYTES),
                      XNN_ALLOCATION_ALIGNMENT);
}

// Returns the size of tensor rounded to appropriate extra bytes and allocation
// alignment.
XNN_INLINE static size_t xnn_tensor_get_rounded_size(
    const struct xnn_value* value) {
  return xnn_get_rounded_size(value->size);
}

// Product of all shape dimensions
size_t xnn_shape_multiply_all_dims(const struct xnn_shape shape[1]);

// Product of all shape dimensions, except for the specified number of the last
// dimensions
size_t xnn_shape_multiply_batch_dims(const struct xnn_shape shape[1],
                                     size_t num_nonbatch_dims);

// Product of all shape dimensions, except for the last (channel) one
size_t xnn_shape_multiply_non_channel_dims(const struct xnn_shape shape[1]);

// Product of n leading dimensions.
size_t xnn_shape_multiply_leading_dims(const struct xnn_shape shape[1],
                                       size_t num_leading_dims);

// Product of trailing dimensions starting from start_dim.
size_t xnn_shape_multiply_trailing_dims(const struct xnn_shape shape[1],
                                        size_t start_dim);

// Get the size in bytes to hold dynamic quant params
size_t xnn_tensor_get_dynamic_quant_param_size(const struct xnn_value* value);

XNN_INLINE static size_t xnn_tensor_get_rounded_dynamic_quant_param_size(
    const struct xnn_value* value) {
  assert(value->datatype == xnn_datatype_qdint8 ||
         value->datatype == xnn_datatype_qduint8);

  // We may read out of bounds for qparams.
  return xnn_get_rounded_size(value->quantization.dynamic_params_size +
                              XNN_EXTRA_QUANTIZATION_PARAMS *
                                  sizeof(struct xnn_quantization_params));
}

enum xnn_status xnn_subgraph_optimize(xnn_subgraph_t subgraph, uint32_t flags);

void xnn_subgraph_rewrite_for_nchw(xnn_subgraph_t subgraph);
// Rewrites subgraph for FP16, returns true if success, false if rewrite failed.
bool xnn_subgraph_rewrite_for_fp16(xnn_subgraph_t subgraph);

void xnn_node_clear(struct xnn_node* node);
void xnn_node_copy(struct xnn_node* dst_node, const struct xnn_node* src_node);
void xnn_value_clear(struct xnn_value* value);

void xnn_value_copy(struct xnn_value* dst_value,
                    const struct xnn_value* src_value);

void xnn_init_convert_node(struct xnn_node* node, uint32_t input_id,
                           uint32_t output_id, uint32_t flags);

struct xnn_workspace {
  void* data;
  size_t size;
  struct xnn_runtime* first_user;
  // Workspace will be destroyed in xnn_delete_runtime or xnn_delete_workspace
  // if num_users reaches 0.
  size_t ref_count;
  size_t persistent_size;
};

void xnn_subgraph_analyze_consumers_and_producers(xnn_subgraph_t subgraph);

enum xnn_status resize_fully_connected_output_tensor(
    const struct xnn_operator_data* opdata, struct xnn_value* values,
    size_t num_values, size_t old_workspace_size, pthreadpool_t threadpool);

XNN_INTERNAL enum xnn_node_type xnn_reduce_operator_to_node_type(
    enum xnn_reduce_operator type);
XNN_INTERNAL enum xnn_reduce_operator xnn_node_type_to_reduce_operator(
    enum xnn_node_type type);

#ifdef __cplusplus
}  // extern "C"
#endif
