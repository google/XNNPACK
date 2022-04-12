// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <xnnpack.h>
#include <xnnpack/log.h>
#include <xnnpack/params.h>
#include <xnnpack/subgraph.h>
#include <xnnpack/subgraph-validation.h>

enum xnn_status xnn_subgraph_check_xnnpack_initialized(enum xnn_node_type node_type)
{
  if ((xnn_params.init_flags & XNN_INIT_FLAG_XNNPACK) == 0) {
    xnn_log_error("failed to define %s operator: XNNPACK is not initialized", xnn_node_type_to_string(node_type));
    return xnn_status_uninitialized;
  }
  return xnn_status_success;
}

enum xnn_status xnn_subgraph_check_input_node_id(enum xnn_node_type node_type, uint32_t input_id, size_t num_values)
{
  if (input_id >= num_values) {
    xnn_log_error(
      "failed to define %s operator with input ID #%" PRIu32 ": invalid Value ID",
      xnn_node_type_to_string(node_type), input_id);
    return xnn_status_invalid_parameter;
  }
  return xnn_status_success;
}

enum xnn_status xnn_subgraph_check_nth_input_node_id(
  enum xnn_node_type node_type,
  uint32_t input_id,
  size_t num_values,
  size_t nth)
{
  if (input_id >= num_values) {
    xnn_log_error(
      "failed to define %s operator with the input %zu ID #%" PRIu32 ": invalid Value ID",
      xnn_node_type_to_string(node_type), nth, input_id);
    return xnn_status_invalid_parameter;
  }
  return xnn_status_success;
}

enum xnn_status xnn_subgraph_check_input_type_dense(
  enum xnn_node_type node_type,
  uint32_t input_id,
  const struct xnn_value* input_value)
{
  if (input_value->type != xnn_value_type_dense_tensor) {
    xnn_log_error(
      "failed to define %s operator with input ID #%" PRIu32 ": unsupported Value type %d (expected dense tensor)",
      xnn_node_type_to_string(node_type), input_id, input_value->type);
    return xnn_status_invalid_parameter;
  }
  return xnn_status_success;
}

enum xnn_status xnn_subgraph_check_nth_input_type_dense(
  enum xnn_node_type node_type,
  uint32_t input_id,
  const struct xnn_value* input_value,
  size_t nth)
{
  if (input_value->type != xnn_value_type_dense_tensor) {
    xnn_log_error(
      "failed to define %s operator with %zu input ID #%" PRIu32 ": unsupported Value type %d (expected dense tensor)",
      xnn_node_type_to_string(node_type), nth, input_id, input_value->type);
    return xnn_status_invalid_parameter;
  }
  return xnn_status_success;
}

enum xnn_status xnn_subgraph_check_output_node_id(enum xnn_node_type node_type, uint32_t output_id, size_t num_values)
{
  if (output_id >= num_values) {
    xnn_log_error(
      "failed to define %s operator with output ID #%" PRIu32 ": invalid Value ID",
      xnn_node_type_to_string(node_type), output_id);
    return xnn_status_invalid_parameter;
  }
  return xnn_status_success;
}

