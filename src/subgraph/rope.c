// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>

#include <xnnpack.h>
#include <xnnpack/log.h>
#include <xnnpack/operator.h>
#include <xnnpack/params.h>
#include <xnnpack/subgraph.h>
#include <xnnpack/subgraph-validation.h>


static enum xnn_status create_rope_operator(
  const struct xnn_node* node,
  const struct xnn_value* values,
  size_t num_values,
  struct xnn_operator_data* opdata,
  struct xnn_code_cache* code_cache,
  struct xnn_weights_cache* weights_cache)
{
  assert(node->num_inputs == 2);
  const uint32_t input_id = node->inputs[0];
  assert(input_id != XNN_INVALID_VALUE_ID);
  assert(input_id < num_values);

  const uint32_t weights_id = node->inputs[1];
  assert(weights_id != XNN_INVALID_VALUE_ID);
  assert(weights_id < num_values);

  assert(node->num_outputs == 1);
  const uint32_t output_id = node->outputs[0];
  assert(output_id != XNN_INVALID_VALUE_ID);
  assert(output_id < num_values);

  const size_t num_input_dims = values[input_id].shape.num_dims;

  const void* weights_data = values[weights_id].fp32_data != NULL ? values[weights_id].fp32_data : values[weights_id].data;

  assert(node->compute_type == xnn_compute_type_fp32);
  const enum xnn_status status = xnn_create_rope_nthc_f32(
    node->params.rope.max_sequence_size,
    values[input_id].shape.dim[num_input_dims - 1],
    weights_data,
    /*flags=*/0,
    &opdata->operator_objects[0]);
  if (status == xnn_status_success) {
    opdata->batch_size = xnn_shape_multiply_batch_dims(&values[input_id].shape, 3);
    opdata->sequence_size = values[input_id].shape.dim[num_input_dims - 3];
    opdata->heads = values[input_id].shape.dim[num_input_dims - 2];
    opdata->inputs[0] = input_id;
    opdata->inputs[1] = weights_id;
    opdata->outputs[0] = output_id;
  }
  return status;
}

static enum xnn_status reshape_rope_operator(
  struct xnn_operator_data* opdata,
  const struct xnn_value* values,
  size_t num_values,
  pthreadpool_t threadpool)
{
  assert(opdata->operator_objects[0]->type == xnn_operator_type_rope_nthc_f32);
  return xnn_reshape_rope_nthc_f32(
    opdata->operator_objects[0],
    opdata->batch_size,
    opdata->sequence_size,
    opdata->heads,
    threadpool);
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

  const uint32_t output_id = opdata->outputs[0];
  assert(output_id != XNN_INVALID_VALUE_ID);
  assert(output_id < num_values);

  const struct xnn_value* input_value = values + input_id;
  const void* input_data = input_value->data;
  assert(input_data != NULL);

  const struct xnn_value* output_value = values + output_id;
  void* output_data = output_value->data;
  assert(output_data != NULL);

  assert(opdata->operator_objects[0]->type == xnn_operator_type_rope_nthc_f32);
  return xnn_setup_rope_nthc_f32(
    opdata->operator_objects[0],
    input_data,
    output_data);
}

enum xnn_status xnn_define_rope(
  xnn_subgraph_t subgraph,
  size_t max_sequence_size,
  uint32_t input_id,
  uint32_t weights_id,
  uint32_t output_id,
  uint32_t flags)
{
  enum xnn_status status;
  if ((status = xnn_subgraph_check_xnnpack_initialized(xnn_node_type_rope)) != xnn_status_success) {
    return status;
  }

  if (max_sequence_size == 0) {
    xnn_log_error(
      "failed to define %s operator with %zu max sequence size parameter: max sequence size must be non-zero",
      xnn_node_type_to_string(xnn_node_type_rope), max_sequence_size);
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

  if (input_value->datatype != xnn_datatype_fp32) {
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

  if (weights_value->datatype != xnn_datatype_fp32) {
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

  status = xnn_subgraph_check_all_dims_match(xnn_node_type_rope, input_id, input_value, output_id, output_value);
  if (status != xnn_status_success) {
    return status;
  }

  if (output_value->datatype != xnn_datatype_fp32) {
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
  node->compute_type = xnn_compute_type_fp32;
  node->params.rope.max_sequence_size = max_sequence_size;
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
