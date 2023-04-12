// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include <xnnpack.h>
#include <xnnpack/log.h>
#include <xnnpack/operator.h>
#include <xnnpack/params.h>
#include <xnnpack/subgraph.h>
#include <xnnpack/subgraph-validation.h>

static enum xnn_status create_transpose_operator(
  const struct xnn_node* node,
  const struct xnn_value* values,
  size_t num_values,
  struct xnn_operator_data* opdata,
  const struct xnn_caches* caches)
{
  assert(node->num_inputs == 1);
  const uint32_t input_id = node->inputs[0];
  assert(input_id != XNN_INVALID_VALUE_ID);
  assert(input_id < num_values);

  assert(node->num_outputs == 1);
  const uint32_t output_id = node->outputs[0];
  assert(output_id != XNN_INVALID_VALUE_ID);
  assert(output_id < num_values);

  enum xnn_status status;
  switch (node->compute_type) {
    case xnn_compute_type_fp32:
      status = xnn_create_transpose_nd_x32(node->flags, &opdata->operator_objects[0]);
      break;
    case xnn_compute_type_fp16:
      status = xnn_create_transpose_nd_x16(node->flags, &opdata->operator_objects[0]);
      break;
    case xnn_compute_type_qs8:
    case xnn_compute_type_qu8:
      status = xnn_create_transpose_nd_x8(node->flags, &opdata->operator_objects[0]);
      break;
    default:
      XNN_UNREACHABLE;
  }

  if (status == xnn_status_success) {
    opdata->inputs[0] = input_id;
    opdata->outputs[0] = output_id;
    opdata->shape1.num_dims = node->params.transpose.num_dims;
    opdata->shape2.num_dims = node->params.transpose.num_dims;
    memcpy(opdata->shape1.dim, values[input_id].shape.dim, opdata->shape1.num_dims * sizeof(size_t));
    memcpy(opdata->shape2.dim, node->params.transpose.perm, opdata->shape2.num_dims * sizeof(size_t));
  }

  return status;
}

static enum xnn_status setup_transpose_operator(
  const struct xnn_operator_data* opdata,
  const struct xnn_blob* blobs,
  size_t num_blobs,
  pthreadpool_t threadpool)
{
  const uint32_t input_id = opdata->inputs[0];
  assert(input_id != XNN_INVALID_VALUE_ID);
  assert(input_id < num_blobs);

  const uint32_t output_id = opdata->outputs[0];
  assert(output_id != XNN_INVALID_VALUE_ID);
  assert(output_id < num_blobs);

  const struct xnn_blob* input_blob = blobs + input_id;
  const void* input_data = input_blob->data;
  assert(input_data != NULL);

  const struct xnn_blob* output_blob = blobs + output_id;
  void* output_data = output_blob->data;
  assert(output_data != NULL);

  enum xnn_status status;
   switch (opdata->operator_objects[0]->type) {
    case xnn_operator_type_transpose_nd_x16: {
      status = xnn_setup_transpose_nd_x16(
        opdata->operator_objects[0],
        input_data,
        output_data,
        opdata->shape1.num_dims,
        opdata->shape1.dim,
        opdata->shape2.dim,
        threadpool);
      break;
    }
    case xnn_operator_type_transpose_nd_x32: {
      status = xnn_setup_transpose_nd_x32(
        opdata->operator_objects[0],
        input_data,
        output_data,
        opdata->shape1.num_dims,
        opdata->shape1.dim,
        opdata->shape2.dim,
        threadpool);
      break;
    }
    case xnn_operator_type_transpose_nd_x8: {
      status = xnn_setup_transpose_nd_x8(
        opdata->operator_objects[0],
        input_data,
        output_data,
        opdata->shape1.num_dims,
        opdata->shape1.dim,
        opdata->shape2.dim,
        threadpool);
      break;
    }
    default:
      XNN_UNREACHABLE;
  }

  return status;
}

enum xnn_status xnn_define_static_transpose(
  xnn_subgraph_t subgraph,
  size_t num_dims,
  const size_t* perm,
  uint32_t input_id,
  uint32_t output_id,
  uint32_t flags)
{
  enum xnn_status status;
  if ((status = xnn_subgraph_check_xnnpack_initialized(xnn_node_type_static_transpose)) != xnn_status_success) {
    return status;
  }

  if (num_dims == 0) {
    xnn_log_error(
      "failed to create %s operator with %zu num_dims: num_dims must be non-zero",
      xnn_node_type_to_string(xnn_node_type_static_transpose), num_dims);
    return xnn_status_invalid_parameter;
  }

  if (num_dims > XNN_MAX_TENSOR_DIMS) {
    xnn_log_error(
      "failed to create %s operator with %zu num_dims: num_dims must be <= %d",
      xnn_node_type_to_string(xnn_node_type_static_transpose), num_dims, XNN_MAX_TENSOR_DIMS);
    return xnn_status_invalid_parameter;
  }

  for (size_t i = 0; i < num_dims; ++i) {
    if (perm[i] >= num_dims) {
      xnn_log_error(
          "failed to create %s operator with %zu perm and %zu num_dims: 0 <= perm < num_dims",
          xnn_node_type_to_string(xnn_node_type_static_transpose), perm[i], num_dims);
      return xnn_status_invalid_parameter;
    }
  }

  for (size_t i = 0; i < num_dims - 1; ++i) {
    for (size_t j = i + 1; j < num_dims; ++j) {
      if (perm[i] == perm[j]) {
        xnn_log_error(
            "failed to create %s operator with duplicate entries in perm",
            xnn_node_type_to_string(xnn_node_type_static_transpose));
        return xnn_status_invalid_parameter;
      }
    }
  }

  if ((status = xnn_subgraph_check_input_node_id(xnn_node_type_static_transpose, input_id, subgraph->num_values)) !=
      xnn_status_success) {
    return status;
  }

  const struct xnn_value* input_value = &subgraph->values[input_id];
  status = xnn_subgraph_check_input_type_dense(xnn_node_type_static_transpose, input_id, input_value);
  if (status != xnn_status_success) {
    return status;
  }

  status = xnn_subgraph_check_output_node_id(xnn_node_type_static_transpose, output_id, subgraph->num_values);
  if (status != xnn_status_success) {
    return status;
  }

  const struct xnn_value* output_value = &subgraph->values[output_id];
  status = xnn_subgraph_check_output_type_dense(xnn_node_type_static_transpose, output_id, output_value);
  if (status != xnn_status_success) {
    return status;
  }

  enum xnn_compute_type compute_type = xnn_compute_type_invalid;
  switch (output_value->datatype) {
    case xnn_datatype_fp32:
      compute_type = xnn_compute_type_fp32;
      break;
    case xnn_datatype_qint8:
      compute_type = xnn_compute_type_qs8;
      break;
    case xnn_datatype_quint8:
      compute_type = xnn_compute_type_qu8;
      break;
    default:
      xnn_log_error(
        "failed to define %s operator with output ID #%" PRIu32 ": unsupported Value datatype %s (%d)",
        xnn_node_type_to_string(xnn_node_type_static_transpose), output_id,
        xnn_datatype_to_string(output_value->datatype), output_value->datatype);
      return xnn_status_invalid_parameter;
  }

  switch (input_value->datatype) {
    case xnn_datatype_fp32:
    case xnn_datatype_qint8:
    case xnn_datatype_quint8:
      break;
    default:
      xnn_log_error(
        "failed to define %s operator with input ID #%" PRIu32 ": unsupported Value datatype %s (%d)",
        xnn_node_type_to_string(xnn_node_type_static_transpose), input_id,
        xnn_datatype_to_string(input_value->datatype), input_value->datatype);
      return xnn_status_invalid_parameter;
  }

  status = xnn_subgraph_check_datatype_matches(
    xnn_node_type_static_transpose, input_id, input_value, output_id, output_value);
  if (status != xnn_status_success) {
    return status;
  }

  struct xnn_node* node = xnn_subgraph_new_node(subgraph);
  if (node == NULL) {
    return xnn_status_out_of_memory;
  }

  node->compute_type = compute_type;
  node->inputs[0] = input_id;
  node->flags = flags;
  node->num_inputs = 1;
  node->num_outputs = 1;
  node->outputs[0] = output_id;
  node->type = xnn_node_type_static_transpose;

  node->params.transpose.num_dims = num_dims;
  node->create = create_transpose_operator;
  node->setup = setup_transpose_operator;

  memcpy(node->params.transpose.perm, perm, num_dims * sizeof(size_t));

  return xnn_status_success;
}
