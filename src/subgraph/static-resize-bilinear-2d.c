// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <math.h>
#include <stddef.h>
#include <stdint.h>

#include <xnnpack.h>
#include <xnnpack/log.h>
#include <xnnpack/params.h>
#include <xnnpack/subgraph.h>


enum xnn_status xnn_define_static_resize_bilinear_2d(
  xnn_subgraph_t subgraph,
  size_t new_height,
  size_t new_width,
  uint32_t input_id,
  uint32_t output_id,
  uint32_t flags)
{
  if ((xnn_params.init_flags & XNN_INIT_FLAG_XNNPACK) == 0) {
    xnn_log_error("failed to define %s operator: XNNPACK is not initialized",
      xnn_node_type_to_string(xnn_node_type_static_resize_bilinear_2d));
    return xnn_status_uninitialized;
  }

  if (new_width == 0 || new_height == 0) {
    xnn_log_error(
      "failed to define %s operator with %zux%zu output: output dimensions must be non-zero",
      xnn_node_type_to_string(xnn_node_type_static_resize_bilinear_2d), new_width, new_height);
    return xnn_status_invalid_parameter;
  }

  if (max(new_width, new_height) >= 16777216) {
    xnn_log_error(
      "failed to define %s operator with %zux%zu output: output dimensions must be below 2**24",
      xnn_node_type_to_string(xnn_node_type_static_resize_bilinear_2d), new_width, new_height);
    return xnn_status_unsupported_parameter;
  }

  const uint32_t supported_flags = XNN_FLAG_TENSORFLOW_LEGACY_MODE | XNN_FLAG_ALIGN_CORNERS;
  const uint32_t invalid_flags = flags & ~supported_flags;
  if (invalid_flags != 0) {
    xnn_log_error(
      "failed to define %s operator with 0x%08" PRIx32 " flags: invalid flags 0x%08" PRIx32,
      xnn_node_type_to_string(xnn_node_type_static_resize_bilinear_2d), flags, invalid_flags);
    return xnn_status_invalid_parameter;
  }

  const uint32_t exclusive_flags = XNN_FLAG_TENSORFLOW_LEGACY_MODE | XNN_FLAG_ALIGN_CORNERS;
  if ((flags & exclusive_flags) == exclusive_flags) {
    xnn_log_error(
      "failed to define %s operator with both XNN_FLAG_TENSORFLOW_LEGACY_MODE and XNN_FLAG_ALIGN_CORNERS flags: "
      "the two flags are mutually exclusive",
      xnn_node_type_to_string(xnn_node_type_static_resize_bilinear_2d));
    return xnn_status_invalid_parameter;
  }

  if (input_id >= subgraph->num_values) {
    xnn_log_error(
      "failed to define %s operator with input ID #%" PRIu32 ": invalid Value ID",
      xnn_node_type_to_string(xnn_node_type_static_resize_bilinear_2d), input_id);
    return xnn_status_invalid_parameter;
  }

  const struct xnn_value* input_value = &subgraph->values[input_id];
  if (input_value->type != xnn_value_type_dense_tensor) {
    xnn_log_error(
      "failed to define %s operator with input ID #%" PRIu32 ": unsupported Value type %d (expected dense tensor)",
      xnn_node_type_to_string(xnn_node_type_static_resize_bilinear_2d), input_id, input_value->type);
    return xnn_status_invalid_parameter;
  }

  switch (input_value->datatype) {
    case xnn_datatype_fp32:
      break;
    default:
      xnn_log_error(
        "failed to define %s operator with input ID #%" PRIu32 ": unsupported Value datatype %s (%d)",
        xnn_node_type_to_string(xnn_node_type_static_resize_bilinear_2d), input_id,
        xnn_datatype_to_string(input_value->datatype), input_value->datatype);
      return xnn_status_invalid_parameter;
  }

  if (output_id >= subgraph->num_values) {
    xnn_log_error(
      "failed to define %s operator with output ID #%" PRIu32 ": invalid Value ID",
      xnn_node_type_to_string(xnn_node_type_static_resize_bilinear_2d), output_id);
    return xnn_status_invalid_parameter;
  }

  const struct xnn_value* output_value = &subgraph->values[output_id];
  if (output_value->type != xnn_value_type_dense_tensor) {
    xnn_log_error(
      "failed to define %s operator with output ID #%" PRIu32 ": unsupported Value type %d (expected dense tensor)",
      xnn_node_type_to_string(xnn_node_type_static_resize_bilinear_2d), output_id, output_value->type);
    return xnn_status_invalid_parameter;
  }

  switch (output_value->datatype) {
    case xnn_datatype_fp32:
      break;
    default:
      xnn_log_error(
        "failed to define %s operator with output ID #%" PRIu32 ": unsupported Value datatype %s (%d)",
        xnn_node_type_to_string(xnn_node_type_static_resize_bilinear_2d), output_id,
        xnn_datatype_to_string(output_value->datatype), output_value->datatype);
      return xnn_status_invalid_parameter;
  }

  struct xnn_node* node = xnn_subgraph_new_node(subgraph);
  if (node == NULL) {
    return xnn_status_out_of_memory;
  }

  node->params.static_resize.new_height = new_height;
  node->params.static_resize.new_width = new_width;

  node->type = xnn_node_type_static_resize_bilinear_2d;
  node->num_inputs = 1;
  node->inputs[0] = input_id;
  node->num_outputs = 1;
  node->outputs[0] = output_id;
  node->flags = flags;

  return xnn_status_success;
}
