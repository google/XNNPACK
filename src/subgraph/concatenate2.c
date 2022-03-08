// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <stdint.h>

#include <xnnpack.h>
#include <xnnpack/log.h>
#include <xnnpack/params.h>
#include <xnnpack/subgraph.h>

static enum xnn_status create_concatenate_operator_helper(
  const struct xnn_node *node,
  size_t channels,
  size_t input_stride,
  size_t output_stride,
  struct xnn_operator_data *opdata,
  size_t index)
{
  switch (node->compute_type) {
    #ifndef XNN_NO_F16_OPERATORS
      case xnn_compute_type_fp16: {
        return xnn_create_copy_nc_x16(channels, input_stride, output_stride, node->flags, &opdata->operator_objects[index]);
      }
    #endif  // !defined(XNN_NO_F16_OPERATORS)
    case xnn_compute_type_fp32: {
      return xnn_create_copy_nc_x32(channels, input_stride, output_stride, node->flags, &opdata->operator_objects[index]);
    }
    #ifndef XNN_NO_QS8_OPERATORS
      case xnn_compute_type_qs8:
    #endif  // !defined(XNN_NO_QS8_OPERATORS)
    #ifndef XNN_NO_QU8_OPERATORS
      case xnn_compute_type_qu8:
    #endif  // !defined(XNN_NO_QU8_OPERATORS)
    #if !defined(XNN_NO_QS8_OPERATORS) || !defined(XNN_NO_QU8_OPERATORS)
    {
      return xnn_create_copy_nc_x8(channels, input_stride, output_stride, node->flags, &opdata->operator_objects[index]);
    }
#endif  // !defined(XNN_NO_QS8_OPERATORS) || !defined(XNN_NO_QU8_OPERATORS)
    default:
      XNN_UNREACHABLE;
  }
}

static enum xnn_status create_concatenate_operator(
  const struct xnn_node* node,
  const struct xnn_value* values,
  size_t num_values,
  struct xnn_operator_data* opdata,
  struct xnn_code_cache* code_cache)
{
  assert(node->num_inputs == 2);
  const uint32_t input1_id = node->inputs[0];
  assert(input1_id != XNN_INVALID_VALUE_ID);
  assert(input1_id < num_values);
  const uint32_t input2_id = node->inputs[1];
  assert(input2_id != XNN_INVALID_VALUE_ID);
  assert(input2_id < num_values);

  assert(node->num_outputs == 1);
  const uint32_t output_id = node->outputs[0];
  assert(output_id != XNN_INVALID_VALUE_ID);
  assert(output_id < num_values);

  const size_t axis = node->params.concatenate.axis;
  size_t batch_size = 1, channels_1 = 1, channels_2 = 1;
  for (size_t i = 0; i < axis; i++) {
    batch_size *= values[output_id].shape.dim[i];
  }

  for (size_t i = axis; i < values[input1_id].shape.num_dims; i++) {
    channels_1 *= values[input1_id].shape.dim[i];
    channels_2 *= values[input2_id].shape.dim[i];
  }
  const size_t output_stride = channels_1 + channels_2;

  enum xnn_status status;
  status = create_concatenate_operator_helper(node, channels_1, channels_1, output_stride, opdata, 0);
  if (status != xnn_status_success) {
    return status;
  }
  status = create_concatenate_operator_helper(node, channels_2, channels_2, output_stride, opdata, 1);
  if (status != xnn_status_success) {
    return status;
  }

  opdata->inputs[0] = input1_id;
  opdata->inputs[1] = input2_id;
  opdata->outputs[0] = output_id;
  opdata->batch_size = batch_size;

  return status;
}

static enum xnn_status setup_concatenate_operator_helper(
  const void* input_data,
  void* output_data,
  const struct xnn_operator_data *opdata,
  size_t index,
  pthreadpool_t threadpool)
{
  // The output pointer of this operator is the sum of all channels of the earlier operators.
  size_t channels = 0;
  for (size_t i = 0; i < index; i++) {
    channels += opdata->operator_objects[i]->channels;
  }

  switch (opdata->operator_objects[index]->type) {
    #ifndef XNN_NO_F16_OPERATORS
      case xnn_operator_type_copy_nc_x16: {
        return xnn_setup_copy_nc_x16(
          opdata->operator_objects[index],
          opdata->batch_size,
          input_data,
          (uint16_t*) output_data + index * channels,
          threadpool);
      }
    #endif  // !defined(XNN_NO_F16_OPERATORS)
    case xnn_operator_type_copy_nc_x32: {
      return xnn_setup_copy_nc_x32(
        opdata->operator_objects[index],
        opdata->batch_size,
        input_data,
        (uint32_t*) output_data + index * channels,
        threadpool);
    }
    #if !defined(XNN_NO_QS8_OPERATORS) || !defined(XNN_NO_QU8_OPERATORS)
      case xnn_operator_type_copy_nc_x8: {
        return xnn_setup_copy_nc_x8(
          opdata->operator_objects[index],
          opdata->batch_size,
          input_data,
          (uint8_t*) output_data + index * channels,
          threadpool);
      }
    #endif  // !defined(XNN_NO_QS8_OPERATORS) || !defined(XNN_NO_QU8_OPERATORS)
    default:
      XNN_UNREACHABLE;
  }
}

static enum xnn_status setup_concatenate_operator(
  const struct xnn_operator_data* opdata,
  const struct xnn_blob* blobs,
  size_t num_blobs,
  pthreadpool_t threadpool)
{
  const uint32_t input1_id = opdata->inputs[0];
  assert(input1_id != XNN_INVALID_VALUE_ID);
  assert(input1_id < num_blobs);

  const uint32_t input2_id = opdata->inputs[1];
  assert(input2_id != XNN_INVALID_VALUE_ID);
  assert(input2_id < num_blobs);

  const uint32_t output_id = opdata->outputs[0];
  assert(output_id != XNN_INVALID_VALUE_ID);
  assert(output_id < num_blobs);

  const struct xnn_blob* input1_blob = blobs + input1_id;
  const void* input1_data = input1_blob->data;
  assert(input1_data != NULL);

  const struct xnn_blob* input2_blob = blobs + input2_id;
  const void* input2_data = input2_blob->data;
  assert(input2_data != NULL);

  const struct xnn_blob* output_blob = blobs + output_id;
  void* output_data = output_blob->data;
  assert(output_data != NULL);

  enum xnn_status status;

  status = setup_concatenate_operator_helper(input1_data, output_data, opdata, 0, threadpool);
  if (status != xnn_status_success) {
    return status;
  }
  return setup_concatenate_operator_helper(input2_data, output_data, opdata, 1, threadpool);
}

enum xnn_status check_input_value(
  xnn_subgraph_t subgraph,
  size_t axis,
  uint32_t input_id,
  uint32_t output_id,
  const char* nth,
  enum xnn_node_type node_type)
{

  if (input_id >= subgraph->num_values) {
    xnn_log_error(
      "failed to define %s operator with the %s input ID #%" PRIu32 ": invalid Value ID",
      xnn_node_type_to_string(node_type), nth, input_id);
    return xnn_status_invalid_parameter;
  }

  const struct xnn_value* input_value = &subgraph->values[input_id];
  if (input_value->type != xnn_value_type_dense_tensor) {
    xnn_log_error(
      "failed to define %s operator with the %s input ID #%" PRIu32
      ": unsupported Value type %d (expected dense tensor)",
      xnn_node_type_to_string(node_type), nth, input_id, input_value->type);
    return xnn_status_invalid_parameter;
  }

  const struct xnn_value* output_value = &subgraph->values[output_id];
  if (input_value->shape.num_dims != output_value->shape.num_dims) {
    xnn_log_error(
      "failed to define %s operator with %s input ID #%" PRIu32
      ": mismatch number of dimensions, %s, input has %zu, output has %zu",
      xnn_node_type_to_string(node_type), nth, input_id, nth, input_value->shape.num_dims,
      output_value->shape.num_dims);
    return xnn_status_invalid_parameter;
  }

  for (size_t i = 0; i < input_value->shape.num_dims; i++) {
    if (i != axis && input_value->shape.dim[i] != output_value->shape.dim[i]) {
      xnn_log_error(
        "failed to define %s operator with input ID #%" PRIu32
        ": mismatch dimension %zu, %s input has %zu, output has %zu",
        xnn_node_type_to_string(node_type), input_id, i, nth, input_value->shape.dim[i], output_value->shape.dim[i]);
      return xnn_status_invalid_parameter;
    }
  }

  if (input_value->datatype != output_value->datatype) {
    xnn_log_error(
      "failed to define %s operator with input ID #%" PRIu32 " and %s output ID #%" PRIu32
      ": mismatching datatypes across the %s input (%s), the output (%s)",
      xnn_node_type_to_string(node_type), input_id, nth, output_id,
      nth, xnn_datatype_to_string(input_value->datatype), xnn_datatype_to_string(output_value->datatype));
    return xnn_status_invalid_parameter;
  }

  return xnn_status_success;
}

#if !defined(XNN_NO_QS8_OPERATORS) || !defined(XNN_NO_QU8_OPERATORS)
enum xnn_status check_input_compute_type(
  xnn_subgraph_t subgraph,
  uint32_t input_id,
  uint32_t output_id,
  const char* nth,
  enum xnn_node_type node_type)
{
  const struct xnn_value* input_value = &subgraph->values[input_id];
  const struct xnn_value* output_value = &subgraph->values[output_id];
  if (input_value->quantization.zero_point != output_value->quantization.zero_point) {
    xnn_log_error(
        "failed to define %s operator with input ID #%" PRIu32 " and output ID #%" PRIu32
        ": mismatching quantization zero point across the %s input (%d) and the output (%d)",
        xnn_node_type_to_string(node_type), input_id, output_id,
        nth, input_value->quantization.zero_point, output_value->quantization.zero_point);
    return xnn_status_invalid_parameter;
  }
  if (input_value->quantization.scale != output_value->quantization.scale) {
    xnn_log_error(
        "failed to define %s operator with input ID #%" PRIu32 " and output ID #%" PRIu32
        ": mismatching quantization scale across the %s input (%.7g) and the output (%.7g)",
        xnn_node_type_to_string(node_type), input_id, output_id,
        nth, input_value->quantization.scale, output_value->quantization.scale);
    return xnn_status_invalid_parameter;
  }
  return xnn_status_success;
}
#endif  // !defined( XNN_NO_QS8_OPERATORS) || !defined(XNN_NO_QU8_OPERATORS)

enum xnn_status xnn_define_concatenate2(
  xnn_subgraph_t subgraph,
  size_t axis,
  uint32_t input1_id,
  uint32_t input2_id,
  uint32_t output_id,
  uint32_t flags)
{
  if ((xnn_params.init_flags & XNN_INIT_FLAG_XNNPACK) == 0) {
    xnn_log_error("failed to define %s operator: XNNPACK is not initialized",
      xnn_node_type_to_string(xnn_node_type_concatenate2));
    return xnn_status_uninitialized;
  }

  if (output_id >= subgraph->num_values) {
    xnn_log_error(
      "failed to define %s operator with output ID #%" PRIu32 ": invalid Value ID",
      xnn_node_type_to_string(xnn_node_type_concatenate2), output_id);
    return xnn_status_invalid_parameter;
  }

  const struct xnn_value* output_value = &subgraph->values[output_id];

  if (output_value->type != xnn_value_type_dense_tensor) {
    xnn_log_error(
      "failed to define %s operator with output ID #%" PRIu32 ": unsupported Value type %d (expected dense tensor)",
      xnn_node_type_to_string(xnn_node_type_concatenate2), output_id, output_value->type);
    return xnn_status_invalid_parameter;
  }

  if (axis >= output_value->shape.num_dims) {
    xnn_log_error(
      "failed to define %s operator with the output ID #%" PRIu32
      ": axis (%zu) exceeds the number of dimensions (%zu)",
      xnn_node_type_to_string(xnn_node_type_concatenate2), output_id, axis, output_value->shape.num_dims);
    return xnn_status_invalid_parameter;
  }

  check_input_value(subgraph, axis, input1_id, output_id, "first", xnn_node_type_concatenate2);
  check_input_value(subgraph, axis, input2_id, output_id, "second", xnn_node_type_concatenate2);

  const struct xnn_value* input1_value = &subgraph->values[input1_id];
  const struct xnn_value* input2_value = &subgraph->values[input2_id];

  if (output_value->shape.dim[axis] != input1_value->shape.dim[axis] + input2_value->shape.dim[axis]) {
    xnn_log_error(
      "failed to define %s operator with output ID #%" PRIu32
      ": mismatch axis dimension %zu, output has %zu, sum of input dimensions is %zu",
      xnn_node_type_to_string(xnn_node_type_concatenate2), output_id, axis, output_value->shape.dim[axis],
      input1_value->shape.dim[axis] + input2_value->shape.dim[axis]);
    return xnn_status_invalid_parameter;
  }

  enum xnn_compute_type compute_type = xnn_compute_type_invalid;
  switch (output_value->datatype) {
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
        "failed to define %s operator with output ID #%" PRIu32 ": unsupported Value datatype %s (%d)",
        xnn_node_type_to_string(xnn_node_type_concatenate2), output_id,
        xnn_datatype_to_string(output_value->datatype), output_value->datatype);
      return xnn_status_invalid_parameter;
  }

  #if !defined(XNN_NO_QS8_OPERATORS) || !defined(XNN_NO_QU8_OPERATORS)
    if (compute_type == xnn_compute_type_qs8 || compute_type == xnn_compute_type_qu8) {
      check_input_compute_type(subgraph, input1_id, output_id, "first", xnn_node_type_concatenate2);
      check_input_compute_type(subgraph, input2_id, output_id, "second", xnn_node_type_concatenate2);
    }
  #endif  // !defined( XNN_NO_QS8_OPERATORS) || !defined(XNN_NO_QU8_OPERATORS)

  struct xnn_node* node = xnn_subgraph_new_node(subgraph);
  if (node == NULL) {
    return xnn_status_out_of_memory;
  }

  node->params.concatenate.axis = axis;
  node->type = xnn_node_type_concatenate2;
  node->compute_type = compute_type;
  node->num_inputs = 2;
  node->inputs[0] = input1_id;
  node->inputs[1] = input2_id;
  node->num_outputs = 1;
  node->outputs[0] = output_id;
  node->flags = flags;

  node->create = create_concatenate_operator;
  node->setup = setup_concatenate_operator;

  return xnn_status_success;
}
