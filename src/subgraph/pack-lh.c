// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <inttypes.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "include/xnnpack.h"
#include "src/xnnpack/common.h"
#include "src/xnnpack/internal.h"
#include "src/xnnpack/log.h"
#include "src/xnnpack/node-type.h"
#include "src/xnnpack/operator-type.h"
#include "src/xnnpack/operator.h"
#include "src/xnnpack/subgraph-validation.h"
#include "src/xnnpack/subgraph.h"
#include <pthreadpool.h>

static enum xnn_status create_pack_lh_operator(
  const struct xnn_node* node,
  const struct xnn_runtime_value* values,
  size_t num_values,
  struct xnn_operator_data* opdata,
  xnn_weights_cache_t weights_cache)
{
  assert(node->num_inputs == 1);

  assert(node->num_outputs == 1);

  const uint32_t input_id = node->inputs[0];
  assert(input_id < num_values);
  const struct xnn_runtime_value *input_value = &values[input_id];
  enum xnn_status status;
  switch (input_value->datatype) {
    case xnn_datatype_qint8:
      status = xnn_create_pack_lh_x8(
        node->flags,
        &opdata->operator_objects[0]);
      break;
    case xnn_datatype_fp16:
      status = xnn_create_pack_lh_x16(
        node->flags,
        &opdata->operator_objects[0]);
      break;
    case xnn_datatype_fp32:
      status = xnn_create_pack_lh_x32(
        node->flags,
        &opdata->operator_objects[0]);
      break;
    default:
      XNN_UNREACHABLE;
  }
  return status;
}

static enum xnn_status reshape_pack_lh_operator(
  struct xnn_operator_data* opdata,
  struct xnn_runtime_value* values,
  size_t num_values,
  pthreadpool_t threadpool)
{
  const uint32_t input_id = opdata->inputs[0];
  assert(input_id < num_values);
  const struct xnn_runtime_value* input_value = &values[input_id];
  const uint32_t output_id = opdata->outputs[0];
  assert(output_id < num_values);
  struct xnn_runtime_value* output_value = &values[output_id];

  const size_t num_input_dims = input_value->shape.num_dims;
  const size_t channels =
      num_input_dims < 1 ? 1 : input_value->shape.dim[num_input_dims - 1];
  size_t batch_size =
      num_input_dims < 2 ? 1 : input_value->shape.dim[num_input_dims - 2];
  size_t num_groups = (2 <= num_input_dims)
                          ? xnn_shape_multiply_leading_dims(&input_value->shape,
                                                            num_input_dims - 2)
                          : 1;
  const size_t old_workspace_size = opdata->workspace_size;
  enum xnn_status status = xnn_status_invalid_state;
  size_t output_size_bytes = 0;

  // Squash the group dimension into the batch size if requested.
  if (output_value->flags & XNN_FLAG_SQUASH_GROUPS) {
    batch_size *= num_groups;
    num_groups = 1;
  }

  switch (opdata->operator_objects[0]->type) {
    case xnn_operator_type_pack_lh_x8:
      status = xnn_reshape_pack_lh_x8(
        opdata->operator_objects[0],
        num_groups,
        batch_size,
        channels,
        &output_size_bytes,
        threadpool);
      break;
    case xnn_operator_type_pack_lh_x16:
      status = xnn_reshape_pack_lh_x16(
        opdata->operator_objects[0],
        num_groups,
        batch_size,
        channels,
        &output_size_bytes,
        threadpool);
      break;
    case xnn_operator_type_pack_lh_x32:
      status = xnn_reshape_pack_lh_x32(
        opdata->operator_objects[0],
        num_groups,
        batch_size,
        channels,
        &output_size_bytes,
        threadpool);
      break;
    default:
      XNN_UNREACHABLE;
  }
  if (status != xnn_status_success) {
    return status;
  }
  // output shape is identical to input shape, however the data is packed into
  // an appropriate format for the following operation so the number of bytes
  // required cannot be determined from the shape alone.
  output_value->shape.num_dims = num_input_dims;
  memcpy(output_value->shape.dim, input_value->shape.dim,
         num_input_dims * sizeof(size_t));
  if (output_size_bytes > output_value->size || opdata->workspace_size > old_workspace_size) {
    output_value->size = output_size_bytes;
    return xnn_status_reallocation_required;
  }
  return xnn_status_success;
}

static enum xnn_status setup_pack_lh_operator(
  const struct xnn_operator_data* opdata,
  const struct xnn_runtime_value* values,
  size_t num_values,
  pthreadpool_t threadpool)
{
  const uint32_t input_id = opdata->inputs[0];
  assert(input_id != XNN_INVALID_VALUE_ID);
  assert(input_id < num_values);

  const uint32_t output_id = opdata->outputs[0];
  assert(output_id != XNN_INVALID_VALUE_ID);
  assert(output_id < num_values);

  const struct xnn_runtime_value* input_value = values + input_id;
  const void* input_data = input_value->data;
  assert(input_data != NULL);

  const struct xnn_runtime_value* output_value = values + output_id;
  void* output_data = output_value->data;
  assert(output_data != NULL);

  switch (opdata->operator_objects[0]->type) {
    case xnn_operator_type_pack_lh_x32:
      return xnn_setup_pack_lh_x32(
        opdata->operator_objects[0],
        input_data,
        output_data);
    case xnn_operator_type_pack_lh_x16:
      return xnn_setup_pack_lh_x16(
        opdata->operator_objects[0],
        input_data,
        output_data);
    case xnn_operator_type_pack_lh_x8:
      return xnn_setup_pack_lh_x8(
        opdata->operator_objects[0],
        input_data,
        output_data);
    default:
      XNN_UNREACHABLE;
  }
}

enum xnn_status xnn_define_pack_lh(
  xnn_subgraph_t subgraph,
  uint32_t input_id,
  uint32_t output_id,
  uint32_t flags)
{
  enum xnn_status status;
  if ((status = xnn_subgraph_check_xnnpack_initialized(xnn_node_type_pack_lh)) != xnn_status_success) {
    return status;
  }

  if ((status = xnn_subgraph_check_input_node_id(xnn_node_type_pack_lh, input_id, subgraph->num_values)) !=
      xnn_status_success) {
    return status;
  }

  const struct xnn_value* input_value = &subgraph->values[input_id];
  status = xnn_subgraph_check_input_type_dense(xnn_node_type_pack_lh, input_id, input_value);
  if (status != xnn_status_success) {
    return status;
  }

  switch (input_value->datatype) {
    case xnn_datatype_qint8:
    case xnn_datatype_fp16:
    case xnn_datatype_fp32:
      break;
    default:
      xnn_log_error(
        "failed to define %s operator with input ID #%" PRIu32 ": unsupported Value datatype %s (%d)",
        xnn_node_type_to_string(xnn_node_type_pack_lh), input_id,
        xnn_datatype_to_string(input_value->datatype), input_value->datatype);
      return xnn_status_invalid_parameter;
  }

  status = xnn_subgraph_check_output_node_id(xnn_node_type_pack_lh, output_id, subgraph->num_values);
  if (status != xnn_status_success) {
    return status;
  }

  struct xnn_value* output_value = &subgraph->values[output_id];
  status = xnn_subgraph_check_output_type_dense(xnn_node_type_pack_lh, output_id, output_value);
  if (status != xnn_status_success) {
    return status;
  }

  switch (output_value->datatype) {
    case xnn_datatype_qint8:
      // Coerce the output from `xnn_datatype_qint8` to `xnn_datatype_pqint8` so
      // that the correct GEMM path is taken.
      output_value->datatype = xnn_datatype_pqint8;
      break;
    case xnn_datatype_fp16:
      // Coerce the output from `xnn_datatype_fp16` to `xnn_datatype_pfp16` so
      // that the correct GEMM path is taken.
      output_value->datatype = xnn_datatype_pfp16;
      break;
    case xnn_datatype_fp32:
      // Coerce the output from `xnn_datatype_fp32` to `xnn_datatype_pfp32` so
      // that the correct GEMM path is taken.
      output_value->datatype = xnn_datatype_pfp32;
      break;
    default:
      xnn_log_error(
        "failed to define %s operator with output ID #%" PRIu32 ": unsupported Value datatype %s (%d)",
        xnn_node_type_to_string(xnn_node_type_pack_lh), output_id,
        xnn_datatype_to_string(output_value->datatype), output_value->datatype);
      return xnn_status_invalid_parameter;
  }

  struct xnn_node* node = xnn_subgraph_new_node(subgraph);
  if (node == NULL) {
    return xnn_status_out_of_memory;
  }

  node->type = xnn_node_type_pack_lh;
  node->num_inputs = 1;
  node->inputs[0] = input_id;
  node->num_outputs = 1;
  node->outputs[0] = output_id;
  node->flags = flags;

  node->create = create_pack_lh_operator;
  node->reshape = reshape_pack_lh_operator;
  node->setup = setup_pack_lh_operator;

  return xnn_status_success;
}
