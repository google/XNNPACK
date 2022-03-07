// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <stdint.h>  // For size_t.

#include <xnnpack.h>
#include <xnnpack/log.h>
#include <xnnpack/params.h>
#include <xnnpack/subgraph.h>

static size_t calculate_batch_size(const struct xnn_value* input, size_t axis)
{
  size_t batch_size = 1;
  for (size_t i = 0; i < axis; i++) {
    batch_size *= input->shape.dim[i];
  }
  return batch_size;
}

static size_t calculate_input_stride(const struct xnn_value* input, size_t axis)
{
  size_t input_stride = 1;
  for (size_t i = axis; i < input->shape.num_dims; i++) {
    input_stride *= input->shape.dim[i];
  }
  return input_stride;
}

static enum xnn_status create_even_split_operator_helper(
    const struct xnn_node* node,
    size_t channels,
    size_t input_stride,
    size_t output_stride,
    struct xnn_operator_data* opdata,
    size_t index)
{
  switch (node->compute_type) {
    #ifndef XNN_NO_F16_OPERATORS
      case xnn_compute_type_fp16: {
        return xnn_create_copy_nc_x16(
            channels, input_stride, output_stride, node->flags, &opdata->operator_objects[index]);
      }
    #endif  // !defined(XNN_NO_F16_OPERATORS)
    case xnn_compute_type_fp32: {
      return xnn_create_copy_nc_x32(
          channels, input_stride, output_stride, node->flags, &opdata->operator_objects[index]);
    }
    #ifndef XNN_NO_QS8_OPERATORS
      case xnn_compute_type_qs8:
    #endif  // !defined(XNN_NO_QS8_OPERATORS)
    #ifndef XNN_NO_QU8_OPERATORS
      case xnn_compute_type_qu8:
    #endif  // !defined(XNN_NO_QU8_OPERATORS)
    #if !defined(XNN_NO_QS8_OPERATORS) || !defined(XNN_NO_QU8_OPERATORS)
      {
        return xnn_create_copy_nc_x8(
            channels, input_stride, output_stride, node->flags, &opdata->operator_objects[index]);
      }
    #endif  // !defined(XNN_NO_QS8_OPERATORS) || !defined(XNN_NO_QU8_OPERATORS)
    default:
      XNN_UNREACHABLE;
  }
}

static enum xnn_status create_even_split_operator(
  const struct xnn_node* node,
  const struct xnn_value* values,
  size_t num_values,
  struct xnn_operator_data* opdata,
  struct xnn_code_cache* code_cache)
{
  assert(node->num_inputs == 1);
  const uint32_t input_id = node->inputs[0];
  assert(input_id != XNN_INVALID_VALUE_ID);
  assert(input_id < num_values);

  assert(node->num_outputs == 2);
  const uint32_t output1_id = node->outputs[0];
  assert(output1_id != XNN_INVALID_VALUE_ID);
  assert(output1_id < num_values);
  const uint32_t output2_id = node->outputs[1];
  assert(output2_id != XNN_INVALID_VALUE_ID);
  assert(output2_id < num_values);

  const size_t axis = node->params.even_split.axis;
  const size_t batch_size = calculate_batch_size(&values[input_id], axis);
  const size_t input_stride = calculate_input_stride(&values[input_id], axis);
  assert(input_stride % 2 == 0);
  const size_t channels = input_stride / 2;
  const size_t output_stride = channels;

  enum xnn_status status;
  status = create_even_split_operator_helper(node, channels, input_stride, output_stride, opdata, 0);
  if (status != xnn_status_success) {
    return status;
  }
  status = create_even_split_operator_helper(node, channels, input_stride, output_stride, opdata, 1);
  if (status != xnn_status_success) {
    return status;
  }

  opdata->inputs[0] = input_id;
  opdata->outputs[0] = output1_id;
  opdata->outputs[1] = output2_id;
  opdata->batch_size = batch_size;

  return status;
}

static enum xnn_status setup_even_split_operator_helper(
    const size_t channels,
    const void* input_data,
    void* output_data,
    const struct xnn_operator_data* opdata,
    size_t index,
    pthreadpool_t threadpool)
{
  switch (opdata->operator_objects[0]->type) {
    #ifndef XNN_NO_F16_OPERATORS
      case xnn_operator_type_copy_nc_x16: {
        return xnn_setup_copy_nc_x16(
            opdata->operator_objects[index], opdata->batch_size, (const uint16_t*) input_data + index * channels,
            output_data, threadpool);
      }
    #endif  // !defined(XNN_NO_F16_OPERATORS)
    case xnn_operator_type_copy_nc_x32: {
      return xnn_setup_copy_nc_x32(
          opdata->operator_objects[index], opdata->batch_size, (const uint32_t*) input_data + index * channels,
          output_data, threadpool);
    }
    #if !defined(XNN_NO_QS8_OPERATORS) || !defined(XNN_NO_QU8_OPERATORS)
      case xnn_operator_type_copy_nc_x8: {
        return xnn_setup_copy_nc_x8(
            opdata->operator_objects[index], opdata->batch_size, (const uint8_t*) input_data + index * channels,
            output_data, threadpool);
      }
    #endif  // !defined(XNN_NO_QS8_OPERATORS) || !defined(XNN_NO_QU8_OPERATORS)
    default:
      XNN_UNREACHABLE;
  }
}

static enum xnn_status setup_even_split_operator(
  const struct xnn_operator_data* opdata,
  const struct xnn_blob* blobs,
  size_t num_blobs,
  pthreadpool_t threadpool)
{
  const uint32_t input_id = opdata->inputs[0];
  assert(input_id != XNN_INVALID_VALUE_ID);
  assert(input_id < num_blobs);

  const uint32_t output1_id = opdata->outputs[0];
  assert(output1_id != XNN_INVALID_VALUE_ID);
  assert(output1_id < num_blobs);

  const uint32_t output2_id = opdata->outputs[1];
  assert(output2_id != XNN_INVALID_VALUE_ID);
  assert(output2_id < num_blobs);

  const struct xnn_blob* input_blob = blobs + input_id;
  const void* input_data = input_blob->data;
  assert(input_data != NULL);

  const struct xnn_blob* output1_blob = blobs + output1_id;
  void* output1_data = output1_blob->data;
  assert(output1_data != NULL);

  const struct xnn_blob* output2_blob = blobs + output2_id;
  void* output2_data = output2_blob->data;
  assert(output2_data != NULL);

  const size_t channels = opdata->operator_objects[0]->channels;

  enum xnn_status status = setup_even_split_operator_helper(channels, input_data, output1_data, opdata, 0, threadpool);
  if (status != xnn_status_success) {
    return status;
  }
  return setup_even_split_operator_helper(channels, input_data, output2_data, opdata, 1, threadpool);
}

enum xnn_status xnn_define_even_split2(
  xnn_subgraph_t subgraph,
  size_t split_dim,
  uint32_t input_id,
  uint32_t output1_id,
  uint32_t output2_id,
  uint32_t flags)
{
  if ((xnn_params.init_flags & XNN_INIT_FLAG_XNNPACK) == 0) {
    xnn_log_error(
      "failed to define %s operator: XNNPACK is not initialized", xnn_node_type_to_string(xnn_node_type_even_split2));
    return xnn_status_uninitialized;
  }

  if (input_id >= subgraph->num_values) {
    xnn_log_error(
      "failed to define %s operator with the input ID #%" PRIu32 ": invalid Value ID",
      xnn_node_type_to_string(xnn_node_type_even_split2), input_id);
    return xnn_status_invalid_parameter;
  }

  const struct xnn_value* input_value = &subgraph->values[input_id];
  if (input_value->type != xnn_value_type_dense_tensor) {
    xnn_log_error(
      "failed to define %s operator with the input ID #%" PRIu32 ": unsupported Value type %d (expected dense tensor)",
      xnn_node_type_to_string(xnn_node_type_even_split2), input_id, input_value->type);
    return xnn_status_invalid_parameter;
  }

  if (split_dim >= input_value->shape.num_dims) {
    xnn_log_error(
      "failed to define %s operator with the input ID #%" PRIu32
      ": split dimension (%zu) exceeds the number of dimensions (%zu)",
      xnn_node_type_to_string(xnn_node_type_even_split2), input_id, split_dim, input_value->shape.num_dims);
    return xnn_status_invalid_parameter;
  }

  if (output1_id >= subgraph->num_values) {
    xnn_log_error(
      "failed to define %s operator with first output ID #%" PRIu32 ": invalid Value ID",
      xnn_node_type_to_string(xnn_node_type_even_split2), output1_id);
    return xnn_status_invalid_parameter;
  }

  const struct xnn_value* output1_value = &subgraph->values[output1_id];
  if (output1_value->type != xnn_value_type_dense_tensor) {
    xnn_log_error(
      "failed to define %s operator with first output ID #%" PRIu32
      ": unsupported Value type %d (expected dense tensor)",
      xnn_node_type_to_string(xnn_node_type_even_split2), output1_id, output1_value->type);
    return xnn_status_invalid_parameter;
  }

  if (output2_id >= subgraph->num_values) {
    xnn_log_error(
      "failed to define %s operator with second output ID #%" PRIu32 ": invalid Value ID",
      xnn_node_type_to_string(xnn_node_type_even_split2), output2_id);
    return xnn_status_invalid_parameter;
  }

  const struct xnn_value* output2_value = &subgraph->values[output2_id];
  if (output2_value->type != xnn_value_type_dense_tensor) {
    xnn_log_error(
      "failed to define %s operator with second output ID #%" PRIu32
      ": unsupported Value type %d (expected dense tensor)",
      xnn_node_type_to_string(xnn_node_type_even_split2), output2_id, output2_value->type);
    return xnn_status_invalid_parameter;
  }

  if (input_value->shape.num_dims != output1_value->shape.num_dims) {
    xnn_log_error(
      "failed to define %s operator with first output ID #%" PRIu32
      ": mismatch number of dimensions, input has %zu, first output has %zu",
      xnn_node_type_to_string(xnn_node_type_even_split2), output1_id, input_value->shape.num_dims,
      output1_value->shape.num_dims);
    return xnn_status_invalid_parameter;
  }

  if (input_value->shape.num_dims != output2_value->shape.num_dims) {
    xnn_log_error(
      "failed to define %s operator with second output ID #%" PRIu32
      ": mismatch number of dimensions, input has %zu, second output has %zu",
      xnn_node_type_to_string(xnn_node_type_even_split2), output2_id, input_value->shape.num_dims,
      output2_value->shape.num_dims);
    return xnn_status_invalid_parameter;
  }

  for (size_t i = 0; i < output1_value->shape.num_dims; i++) {
    if (output1_value->shape.dim[i] != output2_value->shape.dim[i]) {
      xnn_log_error(
        "failed to defined %s operator with outputs ID #%" PRIu32 " and #%" PRIu32
        ": mismatch dimension, first output has %zu, second output has %zu",
        xnn_node_type_to_string(xnn_node_type_even_split2), output1_id, output2_id, output1_value->shape.dim[i],
        output2_value->shape.dim[i]);
      return xnn_status_invalid_parameter;
    }
  }

  for (size_t i = 0; i < input_value->shape.num_dims; i++) {
    if (i == split_dim) {
      if (input_value->shape.dim[i] != output1_value->shape.dim[i] + output2_value->shape.dim[i]) {
        xnn_log_error(
          "failed to define %s operator with input ID #%" PRIu32 " and output IDs #%" PRIu32 " and #%" PRIu32
          ": mismatch split dimension %zu, input has %zu, sum of output dimensions is %zu",
          xnn_node_type_to_string(xnn_node_type_even_split2), input_id, output1_id, output2_id, i,
          input_value->shape.dim[i], output1_value->shape.dim[i] + output2_value->shape.dim[i]);
        return xnn_status_invalid_parameter;
      }
    }
    else {
      if (input_value->shape.dim[i] != output1_value->shape.dim[i]) {
        xnn_log_error(
          "failed to define %s operator with first output ID #%" PRIu32
          ": mismatch dimension %zu, first output has %zu, input has %zu",
          xnn_node_type_to_string(xnn_node_type_even_split2), output1_id, i, output1_value->shape.dim[i],
          input_value->shape.dim[i]);
        return xnn_status_invalid_parameter;
      }
      if (input_value->shape.dim[i] != output2_value->shape.dim[i]) {
        xnn_log_error(
          "failed to define %s operator with second output ID #%" PRIu32
          ": mismatch dimension %zu, second output has %zu, input has %zu",
          xnn_node_type_to_string(xnn_node_type_even_split2), output2_id, i, output2_value->shape.dim[i],
          input_value->shape.dim[i]);
        return xnn_status_invalid_parameter;
      }
    }
  }

  if (input_value->datatype != output1_value->datatype || output1_value->datatype != output2_value->datatype) {
    xnn_log_error(
      "failed to define %s operator with input ID #%" PRIu32 " and output IDs #%" PRIu32 " and #%" PRIu32
      ": mismatching datatypes across the first input (%s), the second input (%s), and output (%s)",
      xnn_node_type_to_string(xnn_node_type_even_split2), input_id, output1_id, output2_id,
      xnn_datatype_to_string(input_value->datatype), xnn_datatype_to_string(output1_value->datatype),
      xnn_datatype_to_string(output2_value->datatype));
    return xnn_status_invalid_parameter;
  }

  enum xnn_compute_type compute_type = xnn_compute_type_invalid;
  switch (input_value->datatype) {
#ifndef XNN_NO_F16_OPERATORS
    case xnn_datatype_fp16:
      compute_type = xnn_compute_type_fp16;
      break;
#endif  // !defined(XNN_NO_F16_OPERATORS)
    case xnn_datatype_fp32:
      compute_type = xnn_compute_type_fp32;
      break;
#ifndef XNN_NO_QS8_OPERATORS
    case xnn_datatype_qint8:
      compute_type = xnn_compute_type_qs8;
      break;
#endif  // !defined(XNN_NO_QS8_OPERATORS)
#ifndef XNN_NO_QU8_OPERATORS
    case xnn_datatype_quint8:
      compute_type = xnn_compute_type_qu8;
      break;
#endif  // !defined(XNN_NO_QU8_OPERATORS)
    default:
      xnn_log_error(
        "failed to define %s operator with input ID #%" PRIu32 ": unsupported Value datatype %s (%d)",
        xnn_node_type_to_string(xnn_node_type_even_split2), input_id, xnn_datatype_to_string(input_value->datatype),
        input_value->datatype);
      return xnn_status_invalid_parameter;
  }

#if !defined(XNN_NO_QS8_OPERATORS) || !defined(XNN_NO_QU8_OPERATORS)
  if (compute_type == xnn_compute_type_qs8 || compute_type == xnn_compute_type_qu8) {
    if (input_value->quantization.zero_point != output1_value->quantization.zero_point) {
      xnn_log_error(
        "failed to define %s operator with input ID #%" PRIu32 " and output ID #%" PRIu32
        ": mismatching quantization zero point across the input (%d) and the first output (%d)",
        xnn_node_type_to_string(xnn_node_type_concatenate2), input_id, output1_id, input_value->quantization.zero_point,
        output1_value->quantization.zero_point);
      return xnn_status_invalid_parameter;
    }
    if (output1_value->quantization.zero_point != output2_value->quantization.zero_point) {
      xnn_log_error(
        "failed to define %s operator with output IDs #%" PRIu32 " and #%" PRIu32
        ": mismatching quantization zero point across the first output (%d) and second output (%d)",
        xnn_node_type_to_string(xnn_node_type_concatenate2), output1_id, output2_id,
        output1_value->quantization.zero_point, output2_value->quantization.zero_point);
      return xnn_status_invalid_parameter;
    }
    if (input_value->quantization.scale != output1_value->quantization.scale) {
      xnn_log_error(
        "failed to define %s operator with input ID #%" PRIu32 " and output ID #%" PRIu32
        ": mismatching quantization scale across the input (%.7g) and the first output (%.7g)",
        xnn_node_type_to_string(xnn_node_type_concatenate2), input_id, output1_id, input_value->quantization.scale,
        output1_value->quantization.scale);
      return xnn_status_invalid_parameter;
    }
    if (output1_value->quantization.scale != output2_value->quantization.scale) {
      xnn_log_error(
        "failed to define %s operator with output IDs #%" PRIu32 " and #%" PRIu32
        ": mismatching quantization scale across the first output (%.7g) and second output (%.7g)",
        xnn_node_type_to_string(xnn_node_type_concatenate2), output1_id, output2_id, output1_value->quantization.scale,
        output2_value->quantization.scale);
      return xnn_status_invalid_parameter;
    }
  }
#endif  // !defined( XNN_NO_QS8_OPERATORS) || !defined(XNN_NO_QU8_OPERATORS)

  struct xnn_node* node = xnn_subgraph_new_node(subgraph);
  if (node == NULL) {
    return xnn_status_out_of_memory;
  }

  node->params.even_split.axis = split_dim;
  node->type = xnn_node_type_even_split2;
  node->compute_type = compute_type;
  node->num_inputs = 1;
  node->inputs[0] = input_id;
  node->num_outputs = 2;
  node->outputs[0] = output1_id;
  node->outputs[1] = output2_id;
  node->flags = flags;

  node->create = create_even_split_operator;
  node->setup = setup_even_split_operator;

  return xnn_status_success;
}
