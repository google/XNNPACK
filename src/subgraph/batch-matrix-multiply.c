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


static enum xnn_status create_batch_matrix_multiply_operator(
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
      status = xnn_create_batch_matrix_multiply_nc_f16(node->flags, &opdata->operator_objects[0]);
      break;
    case xnn_compute_type_fp32:
      status = xnn_create_batch_matrix_multiply_nc_f32(node->flags, &opdata->operator_objects[0]);
      break;
    default:
      XNN_UNREACHABLE;
  }
  return status;
}

static enum xnn_status reshape_batch_matrix_multiply_operator(
  struct xnn_operator_data* opdata,
  struct xnn_value* values,
  size_t num_values,
  pthreadpool_t threadpool)
{
  const uint32_t input1_id = opdata->inputs[0];
  assert(input1_id != XNN_INVALID_VALUE_ID);
  assert(input1_id < num_values);
  const uint32_t input2_id = opdata->inputs[1];
  assert(input2_id != XNN_INVALID_VALUE_ID);
  assert(input2_id < num_values);
  const uint32_t output_id = opdata->outputs[0];
  assert(output_id != XNN_INVALID_VALUE_ID);
  assert(output_id < num_values);

  const struct xnn_value* input1 = values + input1_id;
  const struct xnn_value* input2 = values + input2_id;
  struct xnn_value* output = values + output_id;
  // input1: [B, M, K]
  // input2: [B, K, N] or [B, N, K] (transpose_b)
  const size_t m = input1->shape.dim[input1->shape.num_dims - 2];
  const size_t k = input1->shape.dim[input1->shape.num_dims - 1];
  const bool transpose_b = (opdata->flags & XNN_FLAG_TRANSPOSE_B) != 0;
  const size_t n = input2->shape.dim[transpose_b ? input2->shape.num_dims - 2 : input2->shape.num_dims - 1];
  const size_t batch_size = xnn_shape_multiply_batch_dims(&input1->shape, 2);

  const size_t old_workspace_size = opdata->workspace_size;
  enum xnn_status status = xnn_status_invalid_state;
  switch (opdata->operator_objects[0]->type) {
    case xnn_operator_type_batch_matrix_multiply_nc_f16:
      status = xnn_reshape_batch_matrix_multiply_nc_f16(
        opdata->operator_objects[0],
        batch_size,
        m, k, n,
        &opdata->workspace_size, &opdata->workspace_alignment,
        threadpool);
      break;
    case xnn_operator_type_batch_matrix_multiply_nc_f32:
      status = xnn_reshape_batch_matrix_multiply_nc_f32(
        opdata->operator_objects[0],
        batch_size,
        m, k, n,
        &opdata->workspace_size, &opdata->workspace_alignment,
        threadpool);
      break;
    default:
      XNN_UNREACHABLE;
  }
  if (status != xnn_status_success) {
    return status;
  }
  memcpy(output->shape.dim, input1->shape.dim, (input1->shape.num_dims - 2) * sizeof(size_t));
  output->shape.num_dims = max(input1->shape.num_dims, input2->shape.num_dims);
  output->shape.dim[output->shape.num_dims - 2] = m;
  output->shape.dim[output->shape.num_dims - 1] = n;
  const size_t new_size = xnn_tensor_get_size(output);
  if (new_size > output->size || opdata->workspace_size > old_workspace_size) {
    output->size = new_size;
    return xnn_status_reallocation_required;
  }
  return xnn_status_success;
}

static enum xnn_status setup_batch_matrix_multiply_operator(
  const struct xnn_operator_data* opdata,
  const struct xnn_value* values,
  size_t num_values,
  pthreadpool_t threadpool)
{
  const uint32_t input1_id = opdata->inputs[0];
  assert(input1_id != XNN_INVALID_VALUE_ID);
  assert(input1_id < num_values);

  const uint32_t input2_id = opdata->inputs[1];
  assert(input2_id != XNN_INVALID_VALUE_ID);
  assert(input2_id < num_values);

  const uint32_t output_id = opdata->outputs[0];
  assert(output_id != XNN_INVALID_VALUE_ID);
  assert(output_id < num_values);

  const struct xnn_value* input1_value = values + input1_id;
  const void* input1_data = input1_value->data;
  assert(input1_data != NULL);

  const struct xnn_value* input2_value = values + input2_id;
  const void* input2_data = input2_value->data;
  assert(input2_data != NULL);

  const struct xnn_value* output_value = values + output_id;
  void* output_data = output_value->data;
  assert(output_data != NULL);

  switch (opdata->operator_objects[0]->type) {
    case xnn_operator_type_batch_matrix_multiply_nc_f16:
      return xnn_setup_batch_matrix_multiply_nc_f16(
        opdata->operator_objects[0],
        opdata->workspace, input1_data, input2_data, output_data);
    case xnn_operator_type_batch_matrix_multiply_nc_f32:
      return xnn_setup_batch_matrix_multiply_nc_f32(
        opdata->operator_objects[0],
        opdata->workspace, input1_data, input2_data, output_data);
    default:
      XNN_UNREACHABLE;
  }
}

static inline enum xnn_compute_type validate_datatypes(
  enum xnn_datatype input1_datatype,
  enum xnn_datatype input2_datatype,
  enum xnn_datatype output_datatype)
{
  switch (input2_datatype) {
    case xnn_datatype_fp16:
      if (input1_datatype == xnn_datatype_fp16 && output_datatype == xnn_datatype_fp16) {
        return xnn_compute_type_fp16;
      }
      break;
    case xnn_datatype_fp32:
      if (input1_datatype == xnn_datatype_fp32 && output_datatype == xnn_datatype_fp32) {
        return xnn_compute_type_fp32;
      }
      break;
    default:
      XNN_UNREACHABLE;
  }
  return xnn_compute_type_invalid;
}

enum xnn_status xnn_define_batch_matrix_multiply(
  xnn_subgraph_t subgraph,
  uint32_t input1_id,
  uint32_t input2_id,
  uint32_t output_id,
  uint32_t flags)
{
  enum xnn_status status = xnn_subgraph_check_xnnpack_initialized(xnn_node_type_batch_matrix_multiply);
  if (status != xnn_status_success) {
    return status;
  }

  status = xnn_subgraph_check_input_node_id(xnn_node_type_batch_matrix_multiply, input1_id, subgraph->num_values);
  if (status != xnn_status_success) {
    return status;
  }

  const struct xnn_value* input1_value = &subgraph->values[input1_id];
  status = xnn_subgraph_check_input_type_dense(xnn_node_type_batch_matrix_multiply, input1_id, input1_value);
  if (status != xnn_status_success) {
    return status;
  }

  switch (input1_value->datatype) {
    case xnn_datatype_fp16:
    case xnn_datatype_fp32:
      break;
    default:
      xnn_log_error(
        "failed to define %s operator with input1 ID #%" PRIu32 ": unsupported Value datatype %s (%d)",
        xnn_node_type_to_string(xnn_node_type_batch_matrix_multiply), input1_id,
        xnn_datatype_to_string(input1_value->datatype), input1_value->datatype);
      return xnn_status_invalid_parameter;
  }

  if (input1_value->shape.num_dims < 3) {
    xnn_log_error(
      "failed to define %s operator with input1 ID #%" PRIu32
      ": unsupported number of dimension %zu, must be at least 3",
      xnn_node_type_to_string(xnn_node_type_batch_matrix_multiply), input1_id, input1_value->shape.num_dims);
    return xnn_status_invalid_parameter;
  }

  status = xnn_subgraph_check_input_node_id(xnn_node_type_batch_matrix_multiply, input2_id, subgraph->num_values);
  if (status != xnn_status_success) {
    return status;
  }

  const struct xnn_value* input2_value = &subgraph->values[input2_id];

  status = xnn_subgraph_check_input_type_dense(xnn_node_type_batch_matrix_multiply, input2_id, input1_value);
  if (status != xnn_status_success) {
    return status;
  }

  switch (input2_value->datatype) {
    case xnn_datatype_fp16:
    case xnn_datatype_fp32:
      break;
    default:
      xnn_log_error(
        "failed to define %s operator with input2 ID #%" PRIu32 ": unsupported Value datatype %s (%d)",
        xnn_node_type_to_string(xnn_node_type_batch_matrix_multiply), input2_id,
        xnn_datatype_to_string(input2_value->datatype), input2_value->datatype);
      return xnn_status_invalid_parameter;
  }

  if (input2_value->shape.num_dims < 3) {
    xnn_log_error(
      "failed to define %s operator with input2 ID #%" PRIu32
      ": unsupported number of dimension %zu, must be at least 3",
      xnn_node_type_to_string(xnn_node_type_batch_matrix_multiply), input2_id, input2_value->shape.num_dims);
    return xnn_status_invalid_parameter;
  }

  if (input1_value->shape.num_dims != input2_value->shape.num_dims) {
    xnn_log_error(
      "failed to define %s operator with input1 ID #%" PRIu32 " and input2 ID #%" PRIu32
      ": mismatch number of dimension %zu != %zu",
      xnn_node_type_to_string(xnn_node_type_batch_matrix_multiply), input1_id, input2_id, input1_value->shape.num_dims,
      input2_value->shape.num_dims);
    return xnn_status_invalid_parameter;
  }

  status = xnn_subgraph_check_output_node_id(xnn_node_type_batch_matrix_multiply, output_id, subgraph->num_values);
  if (status != xnn_status_success) {
    return status;
  }

  const struct xnn_value* output_value = &subgraph->values[output_id];
  status = xnn_subgraph_check_output_type_dense(xnn_node_type_batch_matrix_multiply, output_id, output_value);
  if (status != xnn_status_success) {
    return status;
  }

  switch (output_value->datatype) {
    case xnn_datatype_fp16:
    case xnn_datatype_fp32:
      break;
    default:
      xnn_log_error(
        "failed to define %s operator with output ID #%" PRIu32 ": unsupported Value datatype %s (%d)",
        xnn_node_type_to_string(xnn_node_type_batch_matrix_multiply), output_id,
        xnn_datatype_to_string(output_value->datatype), output_value->datatype);
      return xnn_status_invalid_parameter;
  }

  if (output_value->shape.num_dims < 3) {
    xnn_log_error(
      "failed to define %s operator with output ID #%" PRIu32
      ": unsupported number of dimension %zu, must be at least 3",
      xnn_node_type_to_string(xnn_node_type_batch_matrix_multiply), output_id, output_value->shape.num_dims);
    return xnn_status_invalid_parameter;
  }

  if (input1_value->shape.num_dims != output_value->shape.num_dims) {
    xnn_log_error(
      "failed to define %s operator with input1 ID #%" PRIu32 " and output ID #%" PRIu32
      ": mismatch number of dimension %zu != %zu",
      xnn_node_type_to_string(xnn_node_type_batch_matrix_multiply), input1_id, output_id, input1_value->shape.num_dims,
      output_value->shape.num_dims);
    return xnn_status_invalid_parameter;
  }

  // Check that all batch dimensions match.
  for (size_t i = 0; i < input1_value->shape.num_dims - 2; i++) {
    if (input1_value->shape.dim[i] != input2_value->shape.dim[i]) {
      xnn_log_error(
        "failed to define %s operator with input1 ID #%" PRIu32 " and input2 ID #%" PRIu32
        ": mismatch at dimension %zu (%zu != %zu)",
        xnn_node_type_to_string(xnn_node_type_batch_matrix_multiply), input1_id, input2_id, i,
        input1_value->shape.dim[i], input2_value->shape.dim[i]);
      return xnn_status_invalid_parameter;
    }
    if (input1_value->shape.dim[i] != output_value->shape.dim[i]) {
      xnn_log_error(
        "failed to define %s operator with input1 ID #%" PRIu32 " and output ID #%" PRIu32
        ": mismatch at dimension %zu (%zu != %zu)",
        xnn_node_type_to_string(xnn_node_type_batch_matrix_multiply), input1_id, output_id, i,
        input1_value->shape.dim[i], output_value->shape.dim[i]);
      return xnn_status_invalid_parameter;
    }
  }

  const bool transpose_b = (flags & XNN_FLAG_TRANSPOSE_B) != 0;
  // Check that K dimension matches.
  const size_t input1_k = input1_value->shape.num_dims - 1;
  const size_t input2_k = transpose_b ? input2_value->shape.num_dims - 1 : input2_value->shape.num_dims - 2;
  if (input1_value->shape.dim[input1_k] != input2_value->shape.dim[input2_k]) {
    xnn_log_error(
        "failed to define %s operator with input1 ID #%" PRIu32 " and input2 ID #%" PRIu32
        ": mismatch at last dimension (%zu != %zu)",
        xnn_node_type_to_string(xnn_node_type_batch_matrix_multiply), input1_id, input2_id,
        input1_value->shape.dim[input1_k], input2_value->shape.dim[input2_k]);
    return xnn_status_invalid_parameter;
  }

  const size_t last_dimension = input1_value->shape.num_dims - 1;
  const size_t input2_n = transpose_b ? input2_value->shape.num_dims - 2 : input2_value->shape.num_dims - 1;
  // Check that output is [M x N].
  if (output_value->shape.dim[last_dimension - 1] != input1_value->shape.dim[last_dimension - 1]) {
    xnn_log_error(
        "failed to define %s operator with output ID #%" PRIu32 " and input1 ID #%" PRIu32
        ": mismatch at second last dimension of output (%zu != %zu)",
        xnn_node_type_to_string(xnn_node_type_batch_matrix_multiply), output_id, input1_id,
        output_value->shape.dim[last_dimension - 1], input1_value->shape.dim[last_dimension - 1]);
    return xnn_status_invalid_parameter;
  }
  if (output_value->shape.dim[last_dimension] != input2_value->shape.dim[input2_n]) {
    xnn_log_error(
        "failed to define %s operator with output ID #%" PRIu32 " and input2 ID #%" PRIu32
        ": mismatch at last dimension of output (%zu != %zu)",
        xnn_node_type_to_string(xnn_node_type_batch_matrix_multiply), output_id, input2_id,
        output_value->shape.dim[last_dimension], input2_value->shape.dim[last_dimension - 1]);
    return xnn_status_invalid_parameter;
  }

  enum xnn_compute_type compute_type = xnn_compute_type_invalid;
  compute_type = validate_datatypes(input1_value->datatype, input2_value->datatype, output_value->datatype);
  if (compute_type == xnn_compute_type_invalid) {
    xnn_log_error(
      "failed to define %s operator with input1 ID #%" PRIu32 ", input2 ID #%" PRIu32 ", and output ID #%" PRIu32
      ": mismatching datatypes across input1 (%s), input2 (%s), and output (%s)",
      xnn_node_type_to_string(xnn_node_type_batch_matrix_multiply), input1_id, input2_id, output_id,
      xnn_datatype_to_string(input1_value->datatype),
      xnn_datatype_to_string(input2_value->datatype),
      xnn_datatype_to_string(output_value->datatype));
    return xnn_status_invalid_parameter;
  }

  struct xnn_node* node = xnn_subgraph_new_node(subgraph);
  if (node == NULL) {
    return xnn_status_out_of_memory;
  }

  node->type = xnn_node_type_batch_matrix_multiply;
  node->compute_type = compute_type;
  node->num_inputs = 2;
  node->inputs[0] = input1_id;
  node->inputs[1] = input2_id;
  node->num_outputs = 1;
  node->outputs[0] = output_id;
  node->flags = flags;

  node->create = create_batch_matrix_multiply_operator;
  node->setup = setup_batch_matrix_multiply_operator;
  node->reshape = reshape_batch_matrix_multiply_operator;

  return xnn_status_success;
}
