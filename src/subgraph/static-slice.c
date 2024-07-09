// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <inttypes.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "xnnpack.h"
#include "xnnpack/common.h"
#include "xnnpack/log.h"
#include "xnnpack/node-type.h"
#include "xnnpack/operator-type.h"
#include "xnnpack/operator.h"
#include "xnnpack/subgraph-validation.h"
#include "xnnpack/subgraph.h"
#include "pthreadpool.h"

static enum xnn_status create_slice_operator(
    const struct xnn_node* node,
    const struct xnn_value* values,
    size_t num_values,
    struct xnn_operator_data* opdata,
    struct xnn_code_cache* code_cache,
  xnn_weights_cache_t weights_cache)
{
  assert(node->num_inputs == 1);
  assert(node->num_outputs == 1);

  enum xnn_status status;
  switch (node->compute_type) {
    case xnn_compute_type_fp16:
      status = xnn_create_slice_nd_x16(/*flags=*/0, &opdata->operator_objects[0]);
      break;
    case xnn_compute_type_fp32:
      status = xnn_create_slice_nd_x32(/*flags=*/0, &opdata->operator_objects[0]);
      break;
    case xnn_compute_type_qs8:
    case xnn_compute_type_qu8:
      status = xnn_create_slice_nd_x8(/*flags=*/0, &opdata->operator_objects[0]);
      break;
    default:
      XNN_UNREACHABLE;
  }

  if (status == xnn_status_success) {
    const int num_dims = node->params.slice.num_dims;
    opdata->shape2.num_dims = num_dims;
    memcpy(opdata->offsets, node->params.slice.offsets, num_dims * sizeof(size_t));
    memcpy(opdata->sizes, node->params.slice.sizes, num_dims * sizeof(size_t));
  }

  return status;
}

static enum xnn_status reshape_slice_operator(
    struct xnn_operator_data* opdata,
    struct xnn_value* values,
    size_t num_values,
    pthreadpool_t threadpool)
{
  const uint32_t input_id = opdata->inputs[0];
  const uint32_t output_id = opdata->outputs[0];
  assert(input_id < num_values);
  assert(output_id < num_values);
  struct xnn_value* output_value = values + output_id;
  struct xnn_value* input_value = values + input_id;
  const size_t num_dims = input_value->shape.num_dims;
  assert(num_dims == opdata->shape2.num_dims);
  enum xnn_status status = xnn_status_invalid_state;
  const size_t old_workspace_size = opdata->workspace_size;
  switch (opdata->operator_objects[0]->type) {
    case xnn_operator_type_slice_nd_x8:
      status = xnn_reshape_slice_nd_x8(
          opdata->operator_objects[0], num_dims,
          input_value->shape.dim, opdata->offsets, opdata->sizes,
          threadpool);
      break;
    case xnn_operator_type_slice_nd_x16:
      status = xnn_reshape_slice_nd_x16(
          opdata->operator_objects[0], num_dims,
          input_value->shape.dim, opdata->offsets, opdata->sizes,
          threadpool);
      break;
    case xnn_operator_type_slice_nd_x32:
      status = xnn_reshape_slice_nd_x32(
          opdata->operator_objects[0], num_dims,
          input_value->shape.dim, opdata->offsets, opdata->sizes,
          threadpool);
      break;
    default:
      XNN_UNREACHABLE;
  }
  if (status != xnn_status_success) {
    return status;
  }
  output_value->shape.num_dims = num_dims;
  for (size_t i = 0; i < num_dims; ++i) {
    if (opdata->sizes[i] == 0) {
      output_value->shape.dim[i] = input_value->shape.dim[i];
    } else {
      output_value->shape.dim[i] = opdata->sizes[i];
    }
  }
  const size_t new_size = xnn_tensor_get_size(output_value);
  if (new_size > output_value->size || opdata->workspace_size > old_workspace_size) {
    output_value->size = new_size;
    return xnn_status_reallocation_required;
  }
  return xnn_status_success;
}

static enum xnn_status setup_slice_operator(
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
    case xnn_operator_type_slice_nd_x8:
      return xnn_setup_slice_nd_x8(
          opdata->operator_objects[0],
          input_data, output_data);
      break;
    case xnn_operator_type_slice_nd_x16:
      return xnn_setup_slice_nd_x16(
          opdata->operator_objects[0],
          input_data, output_data);
      break;
    case xnn_operator_type_slice_nd_x32:
      return xnn_setup_slice_nd_x32(
          opdata->operator_objects[0],
          input_data, output_data);
      break;
    default:
      XNN_UNREACHABLE;
  }
}

enum xnn_status xnn_define_static_slice(
    xnn_subgraph_t subgraph,
    size_t num_dims,
    const size_t* offsets,
    const size_t* sizes,
    uint32_t input_id,
    uint32_t output_id,
    uint32_t flags)
{
  enum xnn_status status = xnn_subgraph_check_xnnpack_initialized(xnn_node_type_static_slice);
  if (status != xnn_status_success) {
    return status;
  }

  status = xnn_subgraph_check_input_node_id(xnn_node_type_static_slice, input_id, subgraph->num_values);
  if (status != xnn_status_success) {
    return status;
  }

  const struct xnn_value* input_value = &subgraph->values[input_id];
  status = xnn_subgraph_check_input_type_dense(xnn_node_type_static_slice, input_id, input_value);
  if (status != xnn_status_success) {
    return status;
  }

  switch (input_value->datatype) {
    case xnn_datatype_fp16:
    case xnn_datatype_fp32:
    case xnn_datatype_qint8:
    case xnn_datatype_quint8:
      break;
    default:
      xnn_log_error(
        "failed to define %s operator with input ID #%" PRIu32 ": unsupported Value datatype %s (%d)",
        xnn_node_type_to_string(xnn_node_type_static_slice), input_id,
        xnn_datatype_to_string(input_value->datatype), input_value->datatype);
      return xnn_status_invalid_parameter;
  }

  status = xnn_subgraph_check_output_node_id(xnn_node_type_static_slice, output_id, subgraph->num_values);
  if (status != xnn_status_success) {
    return status;
  }

  const struct xnn_value* output_value = &subgraph->values[output_id];
  status = xnn_subgraph_check_output_type_dense(xnn_node_type_static_slice, output_id, output_value);
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
    case xnn_datatype_qint8:
      compute_type = xnn_compute_type_qs8;
      break;
    case xnn_datatype_quint8:
      compute_type = xnn_compute_type_qu8;
      break;
    default:
      xnn_log_error(
        "failed to define %s operator with output ID #%" PRIu32 ": unsupported Value datatype %s (%d)",
        xnn_node_type_to_string(xnn_node_type_static_slice), output_id,
        xnn_datatype_to_string(output_value->datatype), output_value->datatype);
      return xnn_status_invalid_parameter;
  }

  status =
    xnn_subgraph_check_datatype_matches(xnn_node_type_static_slice, input_id, input_value, output_id, output_value);
  if (status != xnn_status_success) {
    return status;
  }

  status = xnn_subgraph_check_quantization_parameter_matches(
      xnn_node_type_static_slice, input_id, input_value, output_id, output_value);
  if (status != xnn_status_success) {
    return status;
  }

  struct xnn_node* node = xnn_subgraph_new_node(subgraph);
  if (node == NULL) {
    return xnn_status_out_of_memory;
  }

  node->type = xnn_node_type_static_slice;
  node->compute_type = compute_type;
  node->num_inputs = 1;
  node->inputs[0] = input_id;
  node->num_outputs = 1;
  node->outputs[0] = output_id;
  node->flags = flags;
  node->params.slice.num_dims = num_dims;
  memcpy(node->params.slice.offsets, offsets, num_dims * sizeof(size_t));
  memcpy(node->params.slice.sizes, sizes, num_dims * sizeof(size_t));

  node->create = create_slice_operator;
  node->reshape = reshape_slice_operator;
  node->setup = setup_slice_operator;

  return xnn_status_success;
}
