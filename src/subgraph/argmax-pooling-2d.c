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


static enum xnn_status create_argmax_pooling_operator(
  const struct xnn_node* node,
  const struct xnn_value* values,
  size_t num_values,
  struct xnn_operator_data* opdata,
  const struct xnn_caches* caches)
{
  assert(node->compute_type == xnn_compute_type_fp32);

  assert(node->num_inputs == 1);
  const uint32_t input_id = node->inputs[0];
  assert(input_id != XNN_INVALID_VALUE_ID);
  assert(input_id < num_values);

  assert(node->num_outputs == 2);
  const uint32_t output_value_id = node->outputs[0];
  assert(output_value_id != XNN_INVALID_VALUE_ID);
  assert(output_value_id < num_values);
  const uint32_t output_index_id = node->outputs[1];
  assert(output_index_id != XNN_INVALID_VALUE_ID);
  assert(output_index_id < num_values);

  const size_t channel_dim = values[input_id].shape.dim[3];
  assert(channel_dim == values[output_value_id].shape.dim[3]);
  assert(channel_dim == values[output_index_id].shape.dim[3]);

  const enum xnn_status status = xnn_create_argmax_pooling2d_nhwc_f32(
    node->params.pooling_2d.padding_top,
    node->params.pooling_2d.padding_right,
    node->params.pooling_2d.padding_bottom,
    node->params.pooling_2d.padding_left,
    node->params.pooling_2d.pooling_height,
    node->params.pooling_2d.pooling_width,
    channel_dim /* channels */, channel_dim /* input stride */, channel_dim /* output stride */,
    node->flags,
    &opdata->operator_objects[0]);
  if (status == xnn_status_success) {
    opdata->batch_size = values[input_id].shape.dim[0];
    opdata->input_height = values[input_id].shape.dim[1];
    opdata->input_width = values[input_id].shape.dim[2];
    opdata->inputs[0] = input_id;
    opdata->outputs[0] = output_value_id;
    opdata->outputs[1] = output_index_id;
  }
  return status;
}

static enum xnn_status setup_argmax_pooling_operator(
  const struct xnn_operator_data* opdata,
  const struct xnn_blob* blobs,
  size_t num_blobs,
  pthreadpool_t threadpool)
{
  const uint32_t input_id = opdata->inputs[0];
  assert(input_id != XNN_INVALID_VALUE_ID);
  assert(input_id < num_blobs);

  const uint32_t output_value_id = opdata->outputs[0];
  assert(output_value_id != XNN_INVALID_VALUE_ID);
  assert(output_value_id < num_blobs);

  const uint32_t output_index_id = opdata->outputs[1];
  assert(output_index_id != XNN_INVALID_VALUE_ID);
  assert(output_index_id < num_blobs);

  const struct xnn_blob* input_blob = blobs + input_id;
  const void* input_data = input_blob->data;
  assert(input_data != NULL);

  const struct xnn_blob* output_value_blob = blobs + output_value_id;
  void* output_value_data = output_value_blob->data;
  assert(output_value_data != NULL);

  const struct xnn_blob* output_index_blob = blobs + output_index_id;
  void* output_index_data = output_index_blob->data;
  assert(output_index_data != NULL);

  return xnn_setup_argmax_pooling2d_nhwc_f32(
    opdata->operator_objects[0],
    opdata->batch_size,
    opdata->input_height,
    opdata->input_width,
    input_data,
    output_value_data,
    output_index_data,
    threadpool);
}

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
  uint32_t flags)
{
  enum xnn_status status;
  if ((status = xnn_subgraph_check_xnnpack_initialized(xnn_node_type_argmax_pooling_2d)) != xnn_status_success) {
    return status;
  }

  const uint32_t pooling_size = pooling_height * pooling_width;
  if (pooling_size == 0) {
    xnn_log_error(
      "failed to define %s operator with %" PRIu32 "x%" PRIu32 " pooling size: "
      "pooling size dimensions must be non-zero",
      xnn_node_type_to_string(xnn_node_type_argmax_pooling_2d), pooling_width, pooling_height);
    return xnn_status_invalid_parameter;
  }

  if (pooling_size == 1) {
    xnn_log_error(
      "failed to define %s operator with 1 pooling element: 1x1 pooling is meaningless",
      xnn_node_type_to_string(xnn_node_type_argmax_pooling_2d));
    return xnn_status_invalid_parameter;
  }

  if ((status = xnn_subgraph_check_input_node_id(xnn_node_type_argmax_pooling_2d, input_id, subgraph->num_values))
      != xnn_status_success) {
    return status;
  }

  const struct xnn_value* input_value = &subgraph->values[input_id];
  status = xnn_subgraph_check_input_type_dense(xnn_node_type_argmax_pooling_2d, input_id, input_value);
  if (status != xnn_status_success) {
    return status;
  }

  switch (input_value->datatype) {
    case xnn_datatype_fp32:
      break;
    default:
      xnn_log_error(
        "failed to define %s operator with input ID #%" PRIu32 ": unsupported Value datatype %s (%d)",
        xnn_node_type_to_string(xnn_node_type_argmax_pooling_2d), input_id,
        xnn_datatype_to_string(input_value->datatype), input_value->datatype);
      return xnn_status_invalid_parameter;
  }

  if (output_value_id >= subgraph->num_values) {
    xnn_log_error(
      "failed to define %s operator with output value ID #%" PRIu32 ": invalid Value ID",
      xnn_node_type_to_string(xnn_node_type_argmax_pooling_2d), output_value_id);
    return xnn_status_invalid_parameter;
  }

  const struct xnn_value* output_value_value = &subgraph->values[output_value_id];
  if (output_value_value->type != xnn_value_type_dense_tensor) {
    xnn_log_error(
      "failed to define %s operator with output value ID #%" PRIu32 ": unsupported Value type %d (expected dense tensor)",
      xnn_node_type_to_string(xnn_node_type_argmax_pooling_2d), output_value_id, output_value_value->type);
    return xnn_status_invalid_parameter;
  }

  switch (output_value_value->datatype) {
    case xnn_datatype_fp32:
      break;
    default:
      xnn_log_error(
        "failed to define %s operator with output value ID #%" PRIu32 ": unsupported Value datatype %s (%d)",
        xnn_node_type_to_string(xnn_node_type_argmax_pooling_2d), output_value_id,
        xnn_datatype_to_string(output_value_value->datatype), output_value_value->datatype);
      return xnn_status_invalid_parameter;
  }

  if (output_index_id >= subgraph->num_values) {
    xnn_log_error(
      "failed to define %s operator with output index ID #%" PRIu32 ": invalid Value ID",
      xnn_node_type_to_string(xnn_node_type_argmax_pooling_2d), output_index_id);
    return xnn_status_invalid_parameter;
  }

  const struct xnn_value* output_index_value = &subgraph->values[output_index_id];
  if (output_index_value->type != xnn_value_type_dense_tensor) {
    xnn_log_error(
      "failed to define %s operator with output index ID #%" PRIu32 ": unsupported Value type %d (expected dense tensor)",
      xnn_node_type_to_string(xnn_node_type_argmax_pooling_2d), output_index_id, output_index_value->type);
    return xnn_status_invalid_parameter;
  }

  struct xnn_node* node = xnn_subgraph_new_node(subgraph);
  if (node == NULL) {
    return xnn_status_out_of_memory;
  }

  node->type = xnn_node_type_argmax_pooling_2d;
  node->compute_type = xnn_compute_type_fp32;
  node->params.pooling_2d.padding_top = input_padding_top;
  node->params.pooling_2d.padding_right = input_padding_right;
  node->params.pooling_2d.padding_bottom = input_padding_bottom;
  node->params.pooling_2d.padding_left = input_padding_left;
  node->params.pooling_2d.pooling_height = pooling_height;
  node->params.pooling_2d.pooling_width = pooling_width;
  node->num_inputs = 1;
  node->inputs[0] = input_id;
  node->num_outputs = 2;
  node->outputs[0] = output_value_id;
  node->outputs[1] = output_index_id;
  node->flags = flags;

  node->create = create_argmax_pooling_operator;
  node->setup = setup_argmax_pooling_operator;

  return xnn_status_success;
}
