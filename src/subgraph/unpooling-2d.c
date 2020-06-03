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
  uint32_t flags)
{
  if (!xnn_params.initialized) {
    xnn_log_error("failed to define UnPooling: XNNPACK is not initialized");
    return xnn_status_uninitialized;
  }

  const uint32_t pooling_size = pooling_height * pooling_width;
  if (pooling_size == 0) {
    xnn_log_error(
      "failed to define UnPooling with %" PRIu32 "x%" PRIu32 " pooling size: "
      "pooling size dimensions must be non-zero",
      pooling_width, pooling_height);
    return xnn_status_invalid_parameter;
  }

  if (pooling_size == 1) {
    xnn_log_error(
      "failed to define UnPooling with 1 pooling element: 1x1 pooling is meaningless");
    return xnn_status_invalid_parameter;
  }

  if (input_value_id >= subgraph->num_values) {
    xnn_log_error(
      "failed to define UnPooling with input value ID #%" PRIu32 ": invalid Value ID",
      input_value_id);
    return xnn_status_invalid_parameter;
  }

  if (input_index_id >= subgraph->num_values) {
    xnn_log_error(
      "failed to define UnPooling with input index ID #%" PRIu32 ": invalid Value ID",
      input_index_id);
    return xnn_status_invalid_parameter;
  }

  if (output_id >= subgraph->num_values) {
    xnn_log_error(
      "failed to define UnPooling with output ID #%" PRIu32 ": invalid Value ID",
      output_id);
    return xnn_status_invalid_parameter;
  }

  struct xnn_node* node = xnn_subgraph_new_node(subgraph);
  if (node == NULL) {
    return xnn_status_out_of_memory;
  }

  node->type = xnn_node_type_unpooling_2d;
  node->params.pooling_2d.padding_top = padding_top;
  node->params.pooling_2d.padding_right = padding_right;
  node->params.pooling_2d.padding_bottom = padding_bottom;
  node->params.pooling_2d.padding_left = padding_left;
  node->params.pooling_2d.pooling_height = pooling_height;
  node->params.pooling_2d.pooling_width = pooling_width;
  node->num_inputs = 2;
  node->inputs[0] = input_value_id;
  node->inputs[1] = input_index_id;
  node->num_outputs = 1;
  node->outputs[0] = output_id;
  node->flags = flags;

  return xnn_status_success;
}
