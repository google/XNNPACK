// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <inttypes.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "xnnpack.h"
#include "xnnpack/log.h"
#include "xnnpack/node-type.h"
#include "xnnpack/operator-type.h"
#include "xnnpack/operator.h"
#include "xnnpack/subgraph-validation.h"
#include "xnnpack/subgraph.h"
#include "pthreadpool.h"

static enum xnn_status create_rope_operator(
  const struct xnn_node* node,
  const struct xnn_value* values,
  size_t num_values,
  struct xnn_operator_data* opdata,
  struct xnn_code_cache* code_cache,
  xnn_weights_cache_t weights_cache)
{
  assert(node->num_inputs == 2);
  assert(node->num_outputs == 1);

  enum xnn_status status;
  switch (node->compute_type) {
    case xnn_compute_type_fp16:
      status = xnn_create_rope_nthc_f16(
        node->params.rope.max_tokens,
        /*flags=*/0,
        &opdata->operator_objects[0]);
      break;
    case xnn_compute_type_fp32:
      status = xnn_create_rope_nthc_f32(
        node->params.rope.max_tokens,
        /*flags=*/0,
        &opdata->operator_objects[0]);
      break;
    default:
      status = xnn_status_invalid_parameter;
  }
  return status;
}

static enum xnn_status reshape_rope_operator(
  struct xnn_operator_data* opdata,
  struct xnn_value* values,
  size_t num_values,
  pthreadpool_t threadpool)
{
  const uint32_t input_id = opdata->inputs[0];
  assert(input_id < num_values);
  const struct xnn_value* input_value = values + input_id;

  const size_t num_input_dims = input_value->shape.num_dims;
  const size_t batch_size = xnn_shape_multiply_batch_dims(&input_value->shape, 3);
  const size_t tokens = input_value->shape.dim[num_input_dims - 3];
  const size_t heads = input_value->shape.dim[num_input_dims - 2];
  const size_t channels = input_value->shape.dim[num_input_dims - 1];

  enum xnn_status status = xnn_status_invalid_state;
  const size_t old_workspace_size = opdata->workspace_size;
  switch (opdata->operator_objects[0]->type) {
    case xnn_operator_type_rope_nthc_f16:
      status = xnn_reshape_rope_nthc_f16(
        opdata->operator_objects[0],
        batch_size,
        tokens,
        heads,
        channels,
        threadpool);
      break;
    case xnn_operator_type_rope_nthc_f32:
      status = xnn_reshape_rope_nthc_f32(
        opdata->operator_objects[0],
        batch_size,
        tokens,
        heads,
        channels,
        threadpool);
      break;
    default:
      return xnn_status_invalid_parameter;
  }
  if (status != xnn_status_success) {
    return status;
  }
  const uint32_t output_id = opdata->outputs[0];
  assert(output_id < num_values);
  struct xnn_value* output_value = values + output_id;

  output_value->shape.num_dims = input_value->shape.num_dims;
  memcpy(output_value->shape.dim, input_value->shape.dim, input_value->shape.num_dims * sizeof(size_t));
  const size_t new_size = xnn_tensor_get_size(output_value);
  if (new_size > output_value->size || opdata->workspace_size > old_workspace_size) {
    output_value->size = new_size;
    return xnn_status_reallocation_required;
  }
  return xnn_status_success;
}

static enum xnn_status setup_rope_operator(
  const struct xnn_operator_data* opdata,
  const struct xnn_value* values,
  size_t num_values,
  pthreadpool_t threadpool)
{
  const uint32_t input_id = opdata->inputs[0];
  assert(input_id != XNN_INVALID_VALUE_ID);
  assert(input_id < num_values);

  const uint32_t weights_id = opdata->inputs[1];
  assert(weights_id != XNN_INVALID_VALUE_ID);
  assert(weights_id < num_values);

  const uint32_t output_id = opdata->outputs[0];
  assert(output_id != XNN_INVALID_VALUE_ID);
  assert(output_id < num_values);

  const struct xnn_value* input_value = values + input_id;
  const void* input_data = input_value->data;
  assert(input_data != NULL);

  const struct xnn_value* weights_value = values + weights_id;
  const void* weights_data = weights_value->data;
  assert(weights_data != NULL);

  const struct xnn_value* output_value = values + output_id;
  void* output_data = output_value->data;
  assert(output_data != NULL);

  switch (opdata->operator_objects[0]->type) {
    case xnn_operator_type_rope_nthc_f16:
      return xnn_setup_rope_nthc_f16(
        opdata->operator_objects[0],
        input_data,
        weights_data,
        output_data);
    case xnn_operator_type_rope_nthc_f32:
      return xnn_setup_rope_nthc_f32(
        opdata->operator_objects[0],
        input_data,
        weights_data,
        output_data);
    default:
      return xnn_status_invalid_parameter;
  }
}

enum xnn_status xnn_define_rope(
  xnn_subgraph_t subgraph,
  size_t max_tokens,
  uint32_t input_id,
  uint32_t weights_id,
  uint32_t output_id,
  uint32_t flags)
{
  enum xnn_status status;
  if ((status = xnn_subgraph_check_xnnpack_initialized(xnn_node_type_rope)) != xnn_status_success) {
    return status;
  }

  if (max_tokens == 0) {
    xnn_log_error(
      "failed to define %s operator with %zu max tokens: maximum number of tokens must be non-zero",
      xnn_node_type_to_string(xnn_node_type_rope), max_tokens);
    return xnn_status_invalid_parameter;
  }

  status = xnn_subgraph_check_input_node_id(xnn_node_type_rope, input_id, subgraph->num_values);
  if (status != xnn_status_success) {
    return status;
  }

  status = xnn_subgraph_check_input_node_id(xnn_node_type_rope, weights_id, subgraph->num_values);
  if (status != xnn_status_success) {
    return status;
  }

  const struct xnn_value* input_value = &subgraph->values[input_id];
  status = xnn_subgraph_check_input_type_dense(xnn_node_type_rope, input_id, input_value);
  if (status != xnn_status_success) {
    return status;
  }

  switch (input_value->datatype) {
    case xnn_datatype_fp16:
    case xnn_datatype_fp32:
      break;
    default:
      xnn_log_error(
        "failed to define %s operator with input ID #%" PRIu32 ": unsupported Value datatype %s (%d)",
        xnn_node_type_to_string(xnn_node_type_rope), input_id,
        xnn_datatype_to_string(input_value->datatype), input_value->datatype);
      return xnn_status_invalid_parameter;
  }

  const struct xnn_value* weights_value = &subgraph->values[weights_id];
  status = xnn_subgraph_check_input_type_dense(xnn_node_type_rope, weights_id, weights_value);
  if (status != xnn_status_success) {
    return status;
  }

  switch (weights_value->datatype) {
    case xnn_datatype_fp16:
    case xnn_datatype_fp32:
      break;
    default:
      xnn_log_error(
        "failed to define %s operator with weights ID #%" PRIu32 ": unsupported Value datatype %s (%d)",
        xnn_node_type_to_string(xnn_node_type_rope), weights_id,
        xnn_datatype_to_string(weights_value->datatype), weights_value->datatype);
      return xnn_status_invalid_parameter;
  }

  status = xnn_subgraph_check_output_node_id(xnn_node_type_rope, output_id, subgraph->num_values);
  if (status != xnn_status_success) {
    return status;
  }

  const struct xnn_value* output_value = &subgraph->values[output_id];
  status = xnn_subgraph_check_output_type_dense(xnn_node_type_rope, output_id, output_value);
  if (status != xnn_status_success) {
    return status;
  }

  enum xnn_compute_type compute_type = xnn_compute_type_invalid;
  switch (output_value->datatype) {
    case xnn_datatype_fp16:
      compute_type = xnn_compute_type_fp16;
      break;
    case xnn_datatype_fp32:
      compute_type = xnn_compute_type_fp32;
      break;
    default:
      xnn_log_error(
        "failed to define %s operator with output ID #%" PRIu32 ": unsupported Value datatype %s (%d)",
        xnn_node_type_to_string(xnn_node_type_rope), output_id,
        xnn_datatype_to_string(output_value->datatype), output_value->datatype);
      return xnn_status_invalid_parameter;
  }

  status = xnn_subgraph_check_datatype_matches(xnn_node_type_subtract, input_id, input_value, output_id, output_value);
  if (status != xnn_status_success) {
    return status;
  }

  struct xnn_node* node = xnn_subgraph_new_node(subgraph);
  if (node == NULL) {
    return xnn_status_out_of_memory;
  }

  node->type = xnn_node_type_rope;
  node->compute_type = compute_type;
  node->params.rope.max_tokens = max_tokens;
  node->num_inputs = 2;
  node->inputs[0] = input_id;
  node->inputs[1] = weights_id;
  node->num_outputs = 1;
  node->outputs[0] = output_id;
  node->flags = flags;

  node->create = create_rope_operator;
  node->reshape = reshape_rope_operator;
  node->setup = setup_rope_operator;

  return xnn_status_success;
}
