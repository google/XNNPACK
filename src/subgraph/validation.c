// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <math.h>

#include <xnnpack.h>
#include <xnnpack/log.h>
#include <xnnpack/node-type.h>
#include <xnnpack/params.h>
#include <xnnpack/subgraph.h>
#include <xnnpack/subgraph-validation.h>

enum xnn_status xnn_subgraph_check_xnnpack_initialized(enum xnn_node_type node_type)
{
  if ((xnn_params.init_flags & XNN_INIT_FLAG_XNNPACK) == 0) {
    xnn_log_error("failed to define %s operator: XNNPACK is not initialized", xnn_node_type_to_string(node_type));
    return xnn_status_uninitialized;
  }
  return xnn_status_success;
}

enum xnn_status xnn_subgraph_check_input_node_id(enum xnn_node_type node_type, uint32_t input_id, size_t num_values)
{
  if (input_id >= num_values) {
    xnn_log_error(
      "failed to define %s operator with input ID #%" PRIu32 ": invalid Value ID",
      xnn_node_type_to_string(node_type), input_id);
    return xnn_status_invalid_parameter;
  }
  return xnn_status_success;
}

enum xnn_status xnn_subgraph_check_nth_input_node_id(
  enum xnn_node_type node_type,
  uint32_t input_id,
  size_t num_values,
  size_t nth)
{
  if (input_id >= num_values) {
    xnn_log_error(
      "failed to define %s operator with the input %zu ID #%" PRIu32 ": invalid Value ID",
      xnn_node_type_to_string(node_type), nth, input_id);
    return xnn_status_invalid_parameter;
  }
  return xnn_status_success;
}

enum xnn_status xnn_subgraph_check_input_type_dense(
  enum xnn_node_type node_type,
  uint32_t input_id,
  const struct xnn_value* input_value)
{
  if (input_value->type != xnn_value_type_dense_tensor) {
    xnn_log_error(
      "failed to define %s operator with input ID #%" PRIu32 ": unsupported Value type %d (expected dense tensor)",
      xnn_node_type_to_string(node_type), input_id, input_value->type);
    return xnn_status_invalid_parameter;
  }
  return xnn_status_success;
}

enum xnn_status xnn_subgraph_check_nth_input_type_dense(
  enum xnn_node_type node_type,
  uint32_t input_id,
  const struct xnn_value* input_value,
  size_t nth)
{
  if (input_value->type != xnn_value_type_dense_tensor) {
    xnn_log_error(
      "failed to define %s operator with %zu input ID #%" PRIu32 ": unsupported Value type %d (expected dense tensor)",
      xnn_node_type_to_string(node_type), nth, input_id, input_value->type);
    return xnn_status_invalid_parameter;
  }
  return xnn_status_success;
}

enum xnn_status xnn_subgraph_check_output_node_id(enum xnn_node_type node_type, uint32_t output_id, size_t num_values)
{
  if (output_id >= num_values) {
    xnn_log_error(
      "failed to define %s operator with output ID #%" PRIu32 ": invalid Value ID",
      xnn_node_type_to_string(node_type), output_id);
    return xnn_status_invalid_parameter;
  }
  return xnn_status_success;
}

enum xnn_status xnn_subgraph_check_output_type_dense(
  enum xnn_node_type node_type,
  uint32_t output_id,
  const struct xnn_value* output_value)
{
  if (output_value->type != xnn_value_type_dense_tensor) {
    xnn_log_error(
      "failed to define %s operator with output ID #%" PRIu32 ": unsupported Value type %d (expected dense tensor)",
      xnn_node_type_to_string(node_type), output_id, output_value->type);
    return xnn_status_invalid_parameter;
  }
  return xnn_status_success;
}

enum xnn_status xnn_subgraph_check_datatype_matches(
  enum xnn_node_type node_type,
  uint32_t input_id,
  const struct xnn_value* input_value,
  uint32_t output_id,
  const struct xnn_value* output_value)
{
  assert(input_value->datatype != xnn_datatype_invalid);
  assert(output_value->datatype != xnn_datatype_invalid);
  if (input_value->datatype != output_value->datatype) {
    xnn_log_error(
      "failed to define %s operator with input ID #%" PRIu32 " and output ID #%" PRIu32
      ": mismatching datatypes across the input (%s) and output (%s)",
      xnn_node_type_to_string(node_type), input_id, output_id,
      xnn_datatype_to_string(input_value->datatype),
      xnn_datatype_to_string(output_value->datatype));
    return xnn_status_invalid_parameter;
  }
  return xnn_status_success;
}

enum xnn_status xnn_subgraph_check_datatype_matches_two_inputs(
  enum xnn_node_type node_type,
  uint32_t input1_id,
  const struct xnn_value* input1_value,
  uint32_t input2_id,
  const struct xnn_value* input2_value,
  uint32_t output_id,
  const struct xnn_value* output_value)
{
  assert(input1_value->datatype != xnn_datatype_invalid);
  assert(input2_value->datatype != xnn_datatype_invalid);
  assert(output_value->datatype != xnn_datatype_invalid);
  if (input1_value->datatype != input2_value->datatype ||
      input1_value->datatype != output_value->datatype)
  {
    xnn_log_error(
      "failed to define %s operator with input IDs #%" PRIu32 " and #%" PRIu32 " and output ID #%" PRIu32
      ": mismatching datatypes across the first input (%s), the second input (%s), and output (%s)",
      xnn_node_type_to_string(node_type), input1_id, input2_id, output_id,
      xnn_datatype_to_string(input1_value->datatype),
      xnn_datatype_to_string(input2_value->datatype),
      xnn_datatype_to_string(output_value->datatype));
    return xnn_status_invalid_parameter;
  }
  return xnn_status_success;
}


enum xnn_status xnn_subgraph_check_output_min_max(enum xnn_node_type node_type, float output_min, float output_max)
{
  if (isnan(output_min)) {
    xnn_log_error(
      "failed to define %s operator with NaN output lower bound: lower bound must be non-NaN",
      xnn_node_type_to_string(node_type));
    return xnn_status_invalid_parameter;
  }

  if (isnan(output_max)) {
    xnn_log_error(
      "failed to define %s operator with NaN output upper bound: upper bound must be non-NaN",
      xnn_node_type_to_string(node_type));
    return xnn_status_invalid_parameter;
  }

  if (output_min > output_max) {
    xnn_log_error(
      "failed to define %s operator with [%.7g, %.7g] output range: lower bound must be less than or equal to upper bound",
      xnn_node_type_to_string(node_type), output_min, output_max);
    return xnn_status_invalid_parameter;
  }
  return xnn_status_success;
}

enum xnn_status xnn_subgraph_check_quantization_parameter_matches(
    enum xnn_node_type node_type,
    uint32_t input_id,
    const struct xnn_value* input_value,
    uint32_t output_id,
    const struct xnn_value* output_value)
{
  if (output_value->datatype == xnn_datatype_qint8 || output_value->datatype == xnn_datatype_quint8) {
    if (input_value->quantization.zero_point != output_value->quantization.zero_point) {
      xnn_log_error(
          "failed to define %s operator with input ID #%" PRIu32 " and output ID #%" PRIu32
          ": mismatching zero point quantization parameter across input (%"PRId32") and output (%"PRId32")",
          xnn_node_type_to_string(node_type), input_id, output_id,
          input_value->quantization.zero_point, output_value->quantization.zero_point);
      return xnn_status_invalid_parameter;
    }
    if (input_value->quantization.scale != output_value->quantization.scale) {
      xnn_log_error(
          "failed to define %s operator with input ID #%" PRIu32 " and output ID #%" PRIu32
          ": mismatching scale quantization parameter across input (%.7g) and output (%.7g)",
          xnn_node_type_to_string(node_type), input_id, output_id,
          input_value->quantization.scale, output_value->quantization.scale);
      return xnn_status_invalid_parameter;
    }
  }
  return xnn_status_success;
}

enum xnn_status xnn_subgraph_check_batch_dims_match(
  enum xnn_node_type node_type,
  uint32_t tensor1_id,
  const struct xnn_value* tensor1_value,
  uint32_t tensor2_id,
  const struct xnn_value* tensor2_value,
  size_t num_batch_dims)
{
  if (tensor1_value->shape.num_dims < num_batch_dims) {
    xnn_log_error(
      "failed to define %s operator with value ID #%" PRIu32 ": number of dimensions of value (%zu) must "
      "be at least %zu", xnn_node_type_to_string(node_type), tensor1_id, tensor1_value->shape.num_dims, num_batch_dims);
  }

  if (tensor2_value->shape.num_dims < num_batch_dims) {
    xnn_log_error(
      "failed to define %s operator with value ID #%" PRIu32 ": number of dimensions of value (%zu) must "
      "be at least %zu", xnn_node_type_to_string(node_type), tensor2_id, tensor2_value->shape.num_dims, num_batch_dims);
  }

  for (size_t i = 0; i < num_batch_dims; i++) {
    if (tensor1_value->shape.dim[i] != tensor2_value->shape.dim[i]) {
      xnn_log_error(
          "failed to define %s operator with value IDs #%" PRIu32 " and #%" PRIu32
          ": mismatch batch size at dimension %zu across first (%zu) and second (%zu) values",
          xnn_node_type_to_string(node_type), tensor1_id, tensor2_id, i, tensor1_value->shape.dim[i],
          tensor2_value->shape.dim[i]);
      return xnn_status_invalid_parameter;
    }
  }

  return xnn_status_success;
}
