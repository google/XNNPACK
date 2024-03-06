// Copyright 2020 Google LLC
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


static enum xnn_status create_prelu_operator(
  const struct xnn_node* node,
  const struct xnn_value* values,
  size_t num_values,
  struct xnn_operator_data* opdata,
  struct xnn_code_cache* code_cache,
  xnn_weights_cache_t weights_cache)
{
  assert(node->num_inputs == 2);
  const uint32_t input_id = node->inputs[0];
  assert(input_id != XNN_INVALID_VALUE_ID);
  assert(input_id < num_values);
  const uint32_t slope_id = node->inputs[1];
  assert(slope_id != XNN_INVALID_VALUE_ID);
  assert(slope_id < num_values);

  const void* slope_data = values[slope_id].fp32_data != NULL ? values[slope_id].fp32_data : values[slope_id].data;
  assert(slope_data != NULL);

  assert(node->num_outputs == 1);

  const size_t num_input_dims = values[input_id].shape.num_dims;
  const size_t channel_dim = num_input_dims == 0 ? 1 : values[input_id].shape.dim[num_input_dims - 1];

  enum xnn_status status;
  switch (node->compute_type) {
    case xnn_compute_type_fp16:
      status = xnn_create_prelu_nc_f16(
        channel_dim /* channels */, channel_dim /* input stride */, channel_dim /* output stride */,
        slope_data /* negative slope */,
        node->flags | XNN_FLAG_FP32_STATIC_WEIGHTS,
        code_cache,
        weights_cache,
        &opdata->operator_objects[0]);
      break;
    case xnn_compute_type_fp32:
      status = xnn_create_prelu_nc_f32(
        channel_dim /* channels */, channel_dim /* input stride */, channel_dim /* output stride */,
        slope_data /* negative slope */,
        node->flags,
        code_cache,
        weights_cache,
        &opdata->operator_objects[0]);
      break;
    default:
      XNN_UNREACHABLE;
  }
  return status;
}

static enum xnn_status reshape_prelu_operator(
  struct xnn_operator_data* opdata,
  struct xnn_value* values,
  size_t num_values,
  pthreadpool_t threadpool)
{
  const uint32_t input_id = opdata->inputs[0];
  assert(input_id < num_values);
  const struct xnn_value* input_value = values + input_id;
  const size_t batch_size = xnn_shape_multiply_non_channel_dims(&input_value->shape);

  const size_t old_workspace_size = opdata->workspace_size;
  enum xnn_status status = xnn_status_invalid_state;
  switch (opdata->operator_objects[0]->type) {
    case xnn_operator_type_prelu_nc_f16:
      status = xnn_reshape_prelu_nc_f16(
        opdata->operator_objects[0],
        batch_size,
        threadpool);
        break;
    case xnn_operator_type_prelu_nc_f32:
      status = xnn_reshape_prelu_nc_f32(
        opdata->operator_objects[0],
        batch_size,
        threadpool);
        break;
    default:
      XNN_UNREACHABLE;
  }
  if (status != xnn_status_success) {
    return status;
  }

  const uint32_t output_id = opdata->outputs[0];
  assert(output_id < num_values);
  struct xnn_value* output_value = values + output_id;

  memcpy(output_value->shape.dim, input_value->shape.dim, input_value->shape.num_dims * sizeof(size_t));
  const size_t new_size = xnn_tensor_get_size(output_value);
  if (new_size > output_value->size || opdata->workspace_size > old_workspace_size) {
    output_value->size = new_size;
    return xnn_status_reallocation_required;
  }

  return xnn_status_success;

}

static enum xnn_status setup_prelu_operator(
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

  switch (opdata->operator_objects[0]->type) {
    case xnn_operator_type_prelu_nc_f16:
      return xnn_setup_prelu_nc_f16(
        opdata->operator_objects[0],
        input_data,
        output_data);
    case xnn_operator_type_prelu_nc_f32:
      return xnn_setup_prelu_nc_f32(
        opdata->operator_objects[0],
        input_data,
        output_data);
    default:
      XNN_UNREACHABLE;
  }
}

enum xnn_status xnn_define_prelu(
  xnn_subgraph_t subgraph,
  uint32_t input_id,
  uint32_t slope_id,
  uint32_t output_id,
  uint32_t flags)
{
  enum xnn_status status;
  if ((status = xnn_subgraph_check_xnnpack_initialized(xnn_node_type_prelu)) != xnn_status_success) {
    return status;
  }

  if ((status = xnn_subgraph_check_input_node_id(xnn_node_type_prelu, input_id, subgraph->num_values)) !=
      xnn_status_success) {
    return status;
  }

  const struct xnn_value* input_value = &subgraph->values[input_id];
  status = xnn_subgraph_check_input_type_dense(xnn_node_type_prelu, input_id, input_value);
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
        xnn_node_type_to_string(xnn_node_type_prelu), input_id,
        xnn_datatype_to_string(input_value->datatype), input_value->datatype);
      return xnn_status_invalid_parameter;
  }

  if (slope_id >= subgraph->num_values) {
    xnn_log_error(
      "failed to define %s operator with slope ID #%" PRIu32 ": invalid Value ID",
      xnn_node_type_to_string(xnn_node_type_prelu), slope_id);
    return xnn_status_invalid_parameter;
  }

  const struct xnn_value* slope_value = &subgraph->values[slope_id];
  if (slope_value->type != xnn_value_type_dense_tensor) {
    xnn_log_error(
      "failed to define %s operator with slope ID #%" PRIu32 ": unsupported Value type %d (expected dense tensor)",
      xnn_node_type_to_string(xnn_node_type_prelu), slope_id, slope_value->type);
    return xnn_status_invalid_parameter;
  }

  if (slope_value->data == NULL) {
    xnn_log_error(
      "failed to define %s operator with slope ID #%" PRIu32 ": non-static Value",
      xnn_node_type_to_string(xnn_node_type_prelu), slope_id);
    return xnn_status_invalid_parameter;
  }

  switch (slope_value->datatype) {
    case xnn_datatype_fp32:
      break;
    default:
      xnn_log_error(
        "failed to define %s operator with slope ID #%" PRIu32 ": unsupported Value datatype %s (%d)",
        xnn_node_type_to_string(xnn_node_type_prelu), slope_id,
        xnn_datatype_to_string(slope_value->datatype), slope_value->datatype);
      return xnn_status_invalid_parameter;
  }

  status = xnn_subgraph_check_output_node_id(xnn_node_type_prelu, output_id, subgraph->num_values);
  if (status != xnn_status_success) {
    return status;
  }

  const struct xnn_value* output_value = &subgraph->values[output_id];
  status = xnn_subgraph_check_output_type_dense(xnn_node_type_prelu, output_id, output_value);
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
        xnn_node_type_to_string(xnn_node_type_prelu), output_id,
        xnn_datatype_to_string(output_value->datatype), output_value->datatype);
      return xnn_status_invalid_parameter;
  }

  struct xnn_node* node = xnn_subgraph_new_node(subgraph);
  if (node == NULL) {
    return xnn_status_out_of_memory;
  }

  node->type = xnn_node_type_prelu;
  node->compute_type = compute_type;
  node->num_inputs = 2;
  node->inputs[0] = input_id;
  node->inputs[1] = slope_id;
  node->num_outputs = 1;
  node->outputs[0] = output_id;
  node->flags = flags;

  node->create = create_prelu_operator;
  node->reshape = reshape_prelu_operator;
  node->setup = setup_prelu_operator;

  return xnn_status_success;
}
