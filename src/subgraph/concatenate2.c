// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <stdint.h>

#include <xnnpack.h>
#include <xnnpack/log.h>
#include <xnnpack/params.h>
#include <xnnpack/subgraph.h>

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
    batch_size *= values[input1_id].shape.dim[i];
  }
  for (size_t i = axis; i < values[input1_id].shape.num_dims; i++) {
    channels_1 *= values[input1_id].shape.dim[i];
    channels_2 *= values[input2_id].shape.dim[i];
  }
  const size_t output_stride = channels_1 + channels_2;

  enum xnn_status status;
  switch (node->compute_type) {
#ifndef XNN_NO_F16_OPERATORS
    case xnn_compute_type_fp16:
    {
      status = xnn_create_copy_nc_x16(channels_1, channels_1, output_stride, node->flags, &opdata->operator_objects[0]);
      if (status != xnn_status_success) {
        break;
      }
      status = xnn_create_copy_nc_x16(channels_2, channels_2, output_stride, node->flags, &opdata->operator_objects[1]);
      break;
    }
#endif  // !defined(XNN_NO_F16_OPERATORS)
    case xnn_compute_type_fp32:
    {
      status = xnn_create_copy_nc_x32(channels_1, channels_1, output_stride, node->flags, &opdata->operator_objects[0]);
      if (status != xnn_status_success) {
        break;
      }
      status = xnn_create_copy_nc_x32(channels_2, channels_2, output_stride, node->flags, &opdata->operator_objects[1]);
      break;
    }
#ifndef XNN_NO_QS8_OPERATORS
    case xnn_compute_type_qs8:
#endif  // !defined(XNN_NO_QS8_OPERATORS)
#ifndef XNN_NO_QU8_OPERATORS
    case xnn_compute_type_qu8:
#endif  // !defined(XNN_NO_QU8_OPERATORS)
#if !defined(XNN_NO_QS8_OPERATORS) || !defined(XNN_NO_QU8_OPERATORS)
    {
      status = xnn_create_copy_nc_x8(channels_1, channels_1, output_stride, node->flags, &opdata->operator_objects[0]);
      if (status != xnn_status_success) {
        break;
      }
      status = xnn_create_copy_nc_x8(channels_2, channels_2, output_stride, node->flags, &opdata->operator_objects[1]);
      break;

    }
#endif  // !defined(XNN_NO_QS8_OPERATORS) || !defined(XNN_NO_QU8_OPERATORS)
    default:
      XNN_UNREACHABLE;
  }

  if (status == xnn_status_success) {
    opdata->inputs[0] = input1_id;
    opdata->inputs[1] = input2_id;
    opdata->outputs[0] = output_id;
    opdata->batch_size = batch_size;
  }

  return status;
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
  size_t channels = opdata->operator_objects[0]->channels;

  switch (opdata->operator_objects[0]->type) {
#ifndef XNN_NO_F16_OPERATORS
    case xnn_operator_type_copy_nc_x16: {
      status = xnn_setup_copy_nc_x16(
          opdata->operator_objects[0],
          opdata->batch_size,
          input1_data,
          output_data,
          threadpool);
      if (status != xnn_status_success) {
        return status;
      }
      status = xnn_setup_copy_nc_x16(
          opdata->operator_objects[1],
          opdata->batch_size,
          input2_data,
          (uint16_t*) output_data + channels,
          threadpool);
      return status;
    }
#endif  // !defined(XNN_NO_F16_OPERATORS)
    case xnn_operator_type_copy_nc_x32: {
      status = xnn_setup_copy_nc_x32(
          opdata->operator_objects[0],
          opdata->batch_size,
          input1_data,
          output_data,
          threadpool);
      if (status != xnn_status_success) {
        return status;
      }
      status = xnn_setup_copy_nc_x32(
          opdata->operator_objects[1],
          opdata->batch_size,
          input2_data,
          (uint32_t *) output_data + channels,
          threadpool);
      return status;
    }
#if !defined(XNN_NO_QS8_OPERATORS) || !defined(XNN_NO_QU8_OPERATORS)
    case xnn_operator_type_copy_nc_x8: {
      status = xnn_setup_copy_nc_x8(
          opdata->operator_objects[0],
          opdata->batch_size,
          input1_data,
          output_data,
          threadpool);
      if (status != xnn_status_success) {
        return status;
      }
      status = xnn_setup_copy_nc_x8(
          opdata->operator_objects[1],
          opdata->batch_size,
          input2_data,
          (uint8_t*) output_data + channels,
          threadpool);
      return status;
    }
#endif  // !defined(XNN_NO_QS8_OPERATORS) || !defined(XNN_NO_QU8_OPERATORS)
    default:
      XNN_UNREACHABLE;
  }
}

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

  if (input1_id >= subgraph->num_values) {
    xnn_log_error(
      "failed to define %s operator with the first input ID #%" PRIu32 ": invalid Value ID",
      xnn_node_type_to_string(xnn_node_type_concatenate2), input1_id);
    return xnn_status_invalid_parameter;
  }

  const struct xnn_value* input1_value = &subgraph->values[input1_id];
  if (input1_value->type != xnn_value_type_dense_tensor) {
    xnn_log_error(
      "failed to define %s operator with the first input ID #%" PRIu32 ": unsupported Value type %d (expected dense tensor)",
      xnn_node_type_to_string(xnn_node_type_concatenate2), input1_id, input1_value->type);
    return xnn_status_invalid_parameter;
  }

  switch (input1_value->datatype) {
    case xnn_datatype_fp32:
#ifndef XNN_NO_QS8_OPERATORS
    case xnn_datatype_qint8:
#endif  // !defined(XNN_NO_QS8_OPERATORS)
#ifndef XNN_NO_QU8_OPERATORS
    case xnn_datatype_quint8:
#endif  // !defined(XNN_NO_QU8_OPERATORS)
      break;
    default:
      xnn_log_error(
        "failed to define %s operator with the first input ID #%" PRIu32 ": unsupported Value datatype %s (%d)",
        xnn_node_type_to_string(xnn_node_type_concatenate2), input1_id,
        xnn_datatype_to_string(input1_value->datatype), input1_value->datatype);
      return xnn_status_invalid_parameter;
  }

  if (axis >= input1_value->shape.num_dims) {
    xnn_log_error(
      "failed to define %s operator with the first input ID #%" PRIu32
      ": axis (%zu) exceeds the number of dimensions (%zu)",
      xnn_node_type_to_string(xnn_node_type_concatenate2), input1_id, axis, input1_value->shape.num_dims);
    return xnn_status_invalid_parameter;
  }

  if (input2_id >= subgraph->num_values) {
    xnn_log_error(
      "failed to define %s operator with the second input ID #%" PRIu32 ": invalid Value ID",
      xnn_node_type_to_string(xnn_node_type_concatenate2), input2_id);
    return xnn_status_invalid_parameter;
  }

  const struct xnn_value* input2_value = &subgraph->values[input2_id];
  if (input2_value->type != xnn_value_type_dense_tensor) {
    xnn_log_error(
      "failed to define %s operator with the second input ID #%" PRIu32 ": unsupported Value type %d (expected dense tensor)",
      xnn_node_type_to_string(xnn_node_type_concatenate2), input2_id, input2_value->type);
    return xnn_status_invalid_parameter;
  }

  switch (input2_value->datatype) {
    case xnn_datatype_fp32:
#ifndef XNN_NO_QS8_OPERATORS
    case xnn_datatype_qint8:
#endif  // !defined(XNN_NO_QS8_OPERATORS)
#ifndef XNN_NO_QU8_OPERATORS
    case xnn_datatype_quint8:
#endif  // !defined(XNN_NO_QU8_OPERATORS)
      break;
    default:
      xnn_log_error(
        "failed to define %s operator with the second input ID #%" PRIu32 ": unsupported Value datatype %s (%d)",
        xnn_node_type_to_string(xnn_node_type_concatenate2), input2_id,
        xnn_datatype_to_string(input2_value->datatype), input2_value->datatype);
      return xnn_status_invalid_parameter;
  }

  if (axis >= input2_value->shape.num_dims) {
    xnn_log_error(
      "failed to define %s operator with the second input ID #%" PRIu32
      ": axis (%zu) exceeds the number of dimensions (%zu)",
      xnn_node_type_to_string(xnn_node_type_concatenate2), input2_id, axis, input2_value->shape.num_dims);
    return xnn_status_invalid_parameter;
  }

  if (input1_value->shape.num_dims != input2_value->shape.num_dims) {
      xnn_log_error(
        "failed to define %s operator with input IDs #%" PRIu32 " and #%" PRIu32
        ": mismatching number of input dimensions %zu and %zu",
        xnn_node_type_to_string(xnn_node_type_concatenate2), input1_id, input2_id,
        input1_value->shape.num_dims, input2_value->shape.num_dims);
    return xnn_status_invalid_parameter;
  }

  for (size_t i = 0; i < input1_value->shape.num_dims; i++) {
    if (i == axis) {
      continue;
    }

    if (input1_value->shape.dim[i] != input2_value->shape.dim[i]) {
      xnn_log_error(
          "failed to define %s operator with input IDs #%" PRIu32 " and #%" PRIu32
          ": mismatch dimension %zu, first input has %zu, second input has %zu",
          xnn_node_type_to_string(xnn_node_type_concatenate2), input1_id, input2_id,
          i, input1_value->shape.dim[i], input2_value->shape.dim[i]);
      return xnn_status_invalid_parameter;
    }
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

  if (input1_value->shape.num_dims != output_value->shape.num_dims) {
      xnn_log_error(
        "failed to define %s operator with output ID #%" PRIu32
        ": mismatch number of dimensions, first input has %zu, output has %zu",
        xnn_node_type_to_string(xnn_node_type_concatenate2), output_id,
        input1_value->shape.num_dims, output_value->shape.num_dims);
    return xnn_status_invalid_parameter;
  }

  for (size_t i = 0; i < output_value->shape.num_dims; i++) {
    if (i == axis) {
      if (output_value->shape.dim[i] != input1_value->shape.dim[i] + input2_value->shape.dim[i]) {
        xnn_log_error(
            "failed to define %s operator with output ID #%" PRIu32
            ": mismatch axis dimension %zu, output has %zu, sum of input dimensions is %zu",
            xnn_node_type_to_string(xnn_node_type_concatenate2), output_id,
            i, output_value->shape.dim[i], input1_value->shape.dim[i] + input2_value->shape.dim[i]);
        return xnn_status_invalid_parameter;
      }
    }

    if (output_value->shape.dim[i] != input1_value->shape.dim[i]) {
      xnn_log_error(
          "failed to define %s operator with output ID #%" PRIu32
          ": mismatch dimension %zu, output has %zu, input has %zu",
          xnn_node_type_to_string(xnn_node_type_concatenate2), output_id,
          i, output_value->shape.dim[i], input1_value->shape.dim[i]);
      return xnn_status_invalid_parameter;
    }
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

  if (input1_value->datatype != input2_value->datatype ||
      input1_value->datatype != output_value->datatype)
  {
    xnn_log_error(
      "failed to define %s operator with input IDs #%" PRIu32 " and #%" PRIu32 " and output ID #%" PRIu32
      ": mismatching datatypes across the first input (%s), the second input (%s), and output (%s)",
      xnn_node_type_to_string(xnn_node_type_concatenate2), input1_id, input2_id, output_id,
      xnn_datatype_to_string(input1_value->datatype),
      xnn_datatype_to_string(input2_value->datatype),
      xnn_datatype_to_string(output_value->datatype));
    return xnn_status_invalid_parameter;
  }

#if !defined(XNN_NO_QS8_OPERATORS) || !defined(XNN_NO_QU8_OPERATORS)
  if (compute_type == xnn_compute_type_qs8 || compute_type == xnn_compute_type_qu8) {
    if (input1_value->quantization.zero_point != input2_value->quantization.zero_point) {
      xnn_log_error(
          "failed to define %s operator with input IDs #%" PRIu32 " and #%" PRIu32
          ": mismatching quantization zero point across the first input (%d) and second input (%d)",
          xnn_node_type_to_string(xnn_node_type_concatenate2), input1_id, input2_id,
          input1_value->quantization.zero_point, input2_value->quantization.zero_point);
      return xnn_status_invalid_parameter;
    }
    if (input1_value->quantization.zero_point != output_value->quantization.zero_point) {
      xnn_log_error(
          "failed to define %s operator with input ID #%" PRIu32 " and output ID #%" PRIu32
          ": mismatching quantization zero point across the first input (%d) and the output (%d)",
          xnn_node_type_to_string(xnn_node_type_concatenate2), input1_id, output_id,
          input1_value->quantization.zero_point, output_value->quantization.zero_point);
      return xnn_status_invalid_parameter;
    }
    if (input1_value->quantization.scale != input2_value->quantization.scale) {
      xnn_log_error(
          "failed to define %s operator with input IDs #%" PRIu32 " and #%" PRIu32
          ": mismatching quantization scale across the first input (%.7g) and second input (%.7g)",
          xnn_node_type_to_string(xnn_node_type_concatenate2), input1_id, input2_id,
          input1_value->quantization.scale, input2_value->quantization.scale);
      return xnn_status_invalid_parameter;
    }
    if (input1_value->quantization.scale != output_value->quantization.scale) {
      xnn_log_error(
          "failed to define %s operator with input ID #%" PRIu32 " and output ID #%" PRIu32
          ": mismatching quantization scale across the first input (%.7g) and the output (%.7g)",
          xnn_node_type_to_string(xnn_node_type_concatenate2), input1_id, output_id,
          input1_value->quantization.scale, output_value->quantization.scale);
      return xnn_status_invalid_parameter;
    }
  }
#endif // !defined( XNN_NO_QS8_OPERATORS) || !defined(XNN_NO_QU8_OPERATORS)

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
