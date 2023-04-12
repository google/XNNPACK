// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <stdint.h>  // For size_t.

#include <xnnpack.h>
#include <xnnpack/log.h>
#include <xnnpack/operator.h>
#include <xnnpack/params.h>
#include <xnnpack/subgraph.h>
#include <xnnpack/subgraph-validation.h>

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
    const uint32_t output_id,
    const struct xnn_node* node,
    size_t channels,
    size_t input_stride,
    size_t output_stride,
    struct xnn_operator_data* opdata,
    size_t index)
{
  if (output_id == XNN_INVALID_VALUE_ID) {
    // Node's output value has been optimized away, don't even create operator object.
    return xnn_status_success;
  }

  switch (node->compute_type) {
    case xnn_compute_type_fp16:
      return xnn_create_copy_nc_x16(
          channels, input_stride, output_stride, node->flags, &opdata->operator_objects[index]);
    case xnn_compute_type_fp32:
      return xnn_create_copy_nc_x32(
          channels, input_stride, output_stride, node->flags, &opdata->operator_objects[index]);
    case xnn_compute_type_qs8:
    case xnn_compute_type_qu8:
      return xnn_create_copy_nc_x8(
          channels, input_stride, output_stride, node->flags, &opdata->operator_objects[index]);
    default:
      XNN_UNREACHABLE;
  }
}

static enum xnn_status create_even_split2_operator(
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

  assert(node->num_outputs == 2);
  uint32_t output1_id = node->outputs[0];
  assert(output1_id != XNN_INVALID_VALUE_ID);
  assert(output1_id < num_values);
  if (values[output1_id].type == xnn_value_type_invalid) {
    output1_id = XNN_INVALID_VALUE_ID;
  }
  uint32_t output2_id = node->outputs[1];
  assert(output2_id != XNN_INVALID_VALUE_ID);
  assert(output2_id < num_values);
  if (values[output2_id].type == xnn_value_type_invalid) {
    output2_id = XNN_INVALID_VALUE_ID;
  }

  const size_t axis = node->params.even_split.axis;
  const size_t batch_size = calculate_batch_size(&values[input_id], axis);
  const size_t input_stride = calculate_input_stride(&values[input_id], axis);
  assert(input_stride % 2 == 0);
  const size_t channels = input_stride / 2;
  const size_t output_stride = channels;

  enum xnn_status status;
  status = create_even_split_operator_helper(output1_id, node, channels, input_stride, output_stride, opdata, 0);
  if (status != xnn_status_success) {
    return status;
  }
  status = create_even_split_operator_helper(output2_id, node, channels, input_stride, output_stride, opdata, 1);
  if (status != xnn_status_success) {
    return status;
  }

  opdata->inputs[0] = input_id;
  opdata->outputs[0] = output1_id;
  opdata->outputs[1] = output2_id;
  opdata->batch_size = batch_size;

  return status;
}

static enum xnn_status create_even_split3_operator(
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

  assert(node->num_outputs == 3);
  uint32_t output1_id = node->outputs[0];
  if (values[output1_id].type == xnn_value_type_invalid) {
    output1_id = XNN_INVALID_VALUE_ID;
  }
  uint32_t output2_id = node->outputs[1];
  if (values[output2_id].type == xnn_value_type_invalid) {
    output2_id = XNN_INVALID_VALUE_ID;
  }
  uint32_t output3_id = node->outputs[2];
  if (values[output3_id].type == xnn_value_type_invalid) {
    output3_id = XNN_INVALID_VALUE_ID;
  }

  const size_t axis = node->params.even_split.axis;
  const size_t batch_size = calculate_batch_size(&values[input_id], axis);
  const size_t input_stride = calculate_input_stride(&values[input_id], axis);
  assert(input_stride % 3 == 0);
  const size_t channels = input_stride / 3;
  const size_t output_stride = channels;

  enum xnn_status status;
  status = create_even_split_operator_helper(output1_id, node, channels, input_stride, output_stride, opdata, 0);
  if (status != xnn_status_success) {
    return status;
  }
  status = create_even_split_operator_helper(output2_id, node, channels, input_stride, output_stride, opdata, 1);
  if (status != xnn_status_success) {
    return status;
  }
  status = create_even_split_operator_helper(output3_id, node, channels, input_stride, output_stride, opdata, 2);
  if (status != xnn_status_success) {
    return status;
  }

  opdata->inputs[0] = input_id;
  opdata->outputs[0] = output1_id;
  opdata->outputs[1] = output2_id;
  opdata->outputs[2] = output3_id;
  opdata->batch_size = batch_size;

  return status;
}

static enum xnn_status create_even_split4_operator(
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

  assert(node->num_outputs == 4);
  uint32_t output1_id = node->outputs[0];
  if (values[output1_id].type == xnn_value_type_invalid) {
    output1_id = XNN_INVALID_VALUE_ID;
  }
  uint32_t output2_id = node->outputs[1];
  if (values[output2_id].type == xnn_value_type_invalid) {
    output2_id = XNN_INVALID_VALUE_ID;
  }
  uint32_t output3_id = node->outputs[2];
  if (values[output3_id].type == xnn_value_type_invalid) {
    output3_id = XNN_INVALID_VALUE_ID;
  }
  uint32_t output4_id = node->outputs[3];
  if (values[output4_id].type == xnn_value_type_invalid) {
    output4_id = XNN_INVALID_VALUE_ID;
  }

  const size_t axis = node->params.even_split.axis;
  const size_t batch_size = calculate_batch_size(&values[input_id], axis);
  const size_t input_stride = calculate_input_stride(&values[input_id], axis);
  assert(input_stride % 4 == 0);
  const size_t channels = input_stride / 4;
  const size_t output_stride = channels;

  enum xnn_status status;
  status = create_even_split_operator_helper(output1_id, node, channels, input_stride, output_stride, opdata, 0);
  if (status != xnn_status_success) {
    return status;
  }
  status = create_even_split_operator_helper(output2_id, node, channels, input_stride, output_stride, opdata, 1);
  if (status != xnn_status_success) {
    return status;
  }
  status = create_even_split_operator_helper(output3_id, node, channels, input_stride, output_stride, opdata, 2);
  if (status != xnn_status_success) {
    return status;
  }
  status = create_even_split_operator_helper(output4_id, node, channels, input_stride, output_stride, opdata, 3);
  if (status != xnn_status_success) {
    return status;
  }

  opdata->inputs[0] = input_id;
  opdata->outputs[0] = output1_id;
  opdata->outputs[1] = output2_id;
  opdata->outputs[2] = output3_id;
  opdata->outputs[3] = output4_id;
  opdata->batch_size = batch_size;

  return status;
}

static enum xnn_status setup_even_split_operator_helper(
  const struct xnn_blob* blobs,
  const uint32_t num_blobs,
  const struct xnn_operator_data* opdata,
  size_t index,
  const void* input_data,
  pthreadpool_t threadpool)
{
  const uint32_t output_id = opdata->outputs[index];
  if (output_id == XNN_INVALID_VALUE_ID) {
    assert(opdata->operator_objects[index] == NULL);
    // output_id was removed during optimization.
    return xnn_status_success;
  }

  const size_t channels = opdata->operator_objects[index]->channels;

  assert(output_id < num_blobs);
  const struct xnn_blob* output_blob = blobs + output_id;
  void* output_data = output_blob->data;
  assert(output_data != NULL);

  switch (opdata->operator_objects[index]->type) {
    case xnn_operator_type_copy_nc_x16:
      return xnn_setup_copy_nc_x16(
        opdata->operator_objects[index], opdata->batch_size, (const uint16_t*) input_data + index * channels,
        output_data, threadpool);
    case xnn_operator_type_copy_nc_x32:
      return xnn_setup_copy_nc_x32(
        opdata->operator_objects[index], opdata->batch_size, (const uint32_t*) input_data + index * channels,
        output_data, threadpool);
    case xnn_operator_type_copy_nc_x8:
      return xnn_setup_copy_nc_x8(
        opdata->operator_objects[index], opdata->batch_size, (const uint8_t*) input_data + index * channels,
        output_data, threadpool);
    default:
      XNN_UNREACHABLE;
  }
}

static enum xnn_status setup_even_split2_operator(
  const struct xnn_operator_data* opdata,
  const struct xnn_blob* blobs,
  size_t num_blobs,
  pthreadpool_t threadpool)
{
  const uint32_t input_id = opdata->inputs[0];
  assert(input_id != XNN_INVALID_VALUE_ID);
  assert(input_id < num_blobs);

  const struct xnn_blob* input_blob = blobs + input_id;
  const void* input_data = input_blob->data;
  assert(input_data != NULL);

  enum xnn_status status = xnn_status_success;

  status = setup_even_split_operator_helper(blobs, num_blobs, opdata, 0, input_data, threadpool);
  if (status != xnn_status_success) {
    return status;
  }
  status = setup_even_split_operator_helper(blobs, num_blobs, opdata, 1, input_data, threadpool);
  if (status != xnn_status_success) {
    return status;
  }

  return status;
}

static enum xnn_status setup_even_split3_operator(
  const struct xnn_operator_data* opdata,
  const struct xnn_blob* blobs,
  size_t num_blobs,
  pthreadpool_t
  threadpool)
{
  const uint32_t input_id = opdata->inputs[0];
  assert(input_id != XNN_INVALID_VALUE_ID);
  assert(input_id < num_blobs);

  const struct xnn_blob* input_blob = blobs + input_id;
  const void* input_data = input_blob->data;
  assert(input_data != NULL);

  enum xnn_status status = xnn_status_success;

  status = setup_even_split_operator_helper(blobs, num_blobs, opdata, 0, input_data, threadpool);
  if (status != xnn_status_success) {
    return status;
  }
  status = setup_even_split_operator_helper(blobs, num_blobs, opdata, 1, input_data, threadpool);
  if (status != xnn_status_success) {
    return status;
  }
  status = setup_even_split_operator_helper(blobs, num_blobs, opdata, 2, input_data, threadpool);
  if (status != xnn_status_success) {
    return status;
  }

  return status;
}

static enum xnn_status setup_even_split4_operator(
  const struct xnn_operator_data* opdata,
  const struct xnn_blob* blobs,
  size_t num_blobs,
  pthreadpool_t
  threadpool)
{
  const uint32_t input_id = opdata->inputs[0];
  assert(input_id != XNN_INVALID_VALUE_ID);
  assert(input_id < num_blobs);

  const struct xnn_blob* input_blob = blobs + input_id;
  const void* input_data = input_blob->data;
  assert(input_data != NULL);

  enum xnn_status status = xnn_status_success;

  status = setup_even_split_operator_helper(blobs, num_blobs, opdata, 0, input_data, threadpool);
  if (status != xnn_status_success) {
    return status;
  }
  status = setup_even_split_operator_helper(blobs, num_blobs, opdata, 1, input_data, threadpool);
  if (status != xnn_status_success) {
    return status;
  }
  status = setup_even_split_operator_helper(blobs, num_blobs, opdata, 2, input_data, threadpool);
  if (status != xnn_status_success) {
    return status;
  }
  status = setup_even_split_operator_helper(blobs, num_blobs, opdata, 3, input_data, threadpool);
  if (status != xnn_status_success) {
    return status;
  }

  return status;
}

enum xnn_status check_output_value(
  xnn_subgraph_t subgraph,
  size_t split_dim,
  uint32_t input_id,
  uint32_t output_id,
  const char* nth,
  enum xnn_node_type node_type)
{
  const struct xnn_value* input_value = &subgraph->values[input_id];
  const struct xnn_value* output_value = &subgraph->values[output_id];
  enum xnn_status status;

  status = xnn_subgraph_check_output_node_id(node_type, output_id, subgraph->num_values);
  if (status != xnn_status_success) {
    return status;
  }

  status = xnn_subgraph_check_output_type_dense(node_type, output_id, output_value);
  if (status != xnn_status_success) {
    return status;
  }

  if (input_value->shape.num_dims != output_value->shape.num_dims) {
    xnn_log_error(
      "failed to define %s operator with %s output ID #%" PRIu32
      ": mismatch number of dimensions, input has %zu, %s output has %zu",
      xnn_node_type_to_string(node_type), nth, output_id, input_value->shape.num_dims,
      nth, output_value->shape.num_dims);
    return xnn_status_invalid_parameter;
  }

  for (size_t i = 0; i < input_value->shape.num_dims; i++) {
    if (i != split_dim && input_value->shape.dim[i] != output_value->shape.dim[i]) {
      xnn_log_error(
        "failed to define %s operator with %s output ID #%" PRIu32
        ": mismatch dimension %zu, %s output has %zu, input has %zu",
        xnn_node_type_to_string(node_type), nth, output_id, i, nth, output_value->shape.dim[i],
        input_value->shape.dim[i]);
      return xnn_status_invalid_parameter;
    }
  }

  status = xnn_subgraph_check_datatype_matches(node_type, input_id, input_value, output_id, output_value);
  if (status != xnn_status_success) {
    return status;
  }

  return xnn_status_success;
}

enum xnn_status check_output_compute_type(
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
      ": mismatching quantization zero point across the input (%" PRId32 ") and the %s output (%" PRId32 ")",
      xnn_node_type_to_string(node_type), input_id, output_id,
      input_value->quantization.zero_point, nth, output_value->quantization.zero_point);
    return xnn_status_invalid_parameter;
  }
  if (input_value->quantization.scale != output_value->quantization.scale) {
    xnn_log_error(
      "failed to define %s operator with input ID #%" PRIu32 " and output ID #%" PRIu32
      ": mismatching quantization scale across the input (%.7g) and the %s output (%.7g)",
      xnn_node_type_to_string(node_type), input_id, output_id, input_value->quantization.scale,
      nth, output_value->quantization.scale);
    return xnn_status_invalid_parameter;
  }
  return xnn_status_success;
}

enum xnn_status xnn_define_even_split_n(
  enum xnn_node_type node_type,
  xnn_subgraph_t subgraph,
  size_t split_dim,
  uint32_t input_id,
  size_t num_outputs,
  const uint32_t* output_ids,
  uint32_t flags)
{
  assert(num_outputs > 1);
  assert(num_outputs < 5);

  enum xnn_status status;
  if ((status = xnn_subgraph_check_xnnpack_initialized(node_type)) != xnn_status_success) {
    return status;
  }

  if ((status = xnn_subgraph_check_input_node_id(node_type, input_id, subgraph->num_values)) != xnn_status_success) {
    return status;
  }

  const struct xnn_value* input_value = &subgraph->values[input_id];
  status = xnn_subgraph_check_input_type_dense(node_type, input_id, input_value);
  if (status != xnn_status_success) {
    return status;
  }

  check_output_value(subgraph, split_dim, input_id, output_ids[0], "first", node_type);
  check_output_value(subgraph, split_dim, input_id, output_ids[1], "second", node_type);

  if (num_outputs > 2) {
    check_output_value(subgraph, split_dim, input_id, output_ids[2], "third", node_type);
  }
  if (num_outputs > 3) {
    check_output_value(subgraph, split_dim, input_id, output_ids[3], "fourth", node_type);
  }

  // Check that the split dimension can be evenly split into outputs.
  if (split_dim >= input_value->shape.num_dims) {
    xnn_log_error(
      "failed to define %s operator with the input ID #%" PRIu32
      ": split dimension (%zu) exceeds the number of dimensions (%zu)",
      xnn_node_type_to_string(node_type), input_id, split_dim, input_value->shape.num_dims);
    return xnn_status_invalid_parameter;
  }

  if (input_value->shape.dim[split_dim] % num_outputs != 0) {
    xnn_log_error(
      "failed to define %s operator with the input ID #%" PRIu32
      ": split dimension %zu has value %zu which cannot be evenly split into %zu",
      xnn_node_type_to_string(node_type), input_id, split_dim, input_value->shape.dim[split_dim], num_outputs);
    return xnn_status_invalid_parameter;
  }

  // Check that the split dimensions of output add up;
  size_t output_dimensions_sum = 0;
  for (size_t i = 0; i < num_outputs; i++) {
    const struct xnn_value* output_value = &subgraph->values[output_ids[i]];
    output_dimensions_sum += output_value->shape.dim[split_dim];
  }

  if (output_dimensions_sum != input_value->shape.dim[split_dim]) {
    xnn_log_error(
      "failed to define %s operator with the input ID #%" PRIu32
      ": input split dimension value (%zu) does not match the sum of output split dimensions value %zu",
      xnn_node_type_to_string(node_type), input_id, input_value->shape.dim[split_dim], output_dimensions_sum);
    return xnn_status_invalid_parameter;
  }

  enum xnn_compute_type compute_type = xnn_compute_type_invalid;
  switch (input_value->datatype) {
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
        "failed to define %s operator with input ID #%" PRIu32 ": unsupported Value datatype %s (%d)",
        xnn_node_type_to_string(node_type), input_id, xnn_datatype_to_string(input_value->datatype),
        input_value->datatype);
      return xnn_status_invalid_parameter;
  }

  if (compute_type == xnn_compute_type_qs8 || compute_type == xnn_compute_type_qu8) {
    check_output_compute_type(subgraph, input_id, output_ids[0], "first", node_type);
    check_output_compute_type(subgraph, input_id, output_ids[1], "second", node_type);
    if (num_outputs > 2) {
      check_output_compute_type(subgraph, input_id, output_ids[2], "third", node_type);
    }
    if (num_outputs > 3) {
      check_output_compute_type(subgraph, input_id, output_ids[3], "fourth", node_type);
    }
  }

  struct xnn_node* node = xnn_subgraph_new_node(subgraph);
  if (node == NULL) {
    return xnn_status_out_of_memory;
  }

  node->params.even_split.axis = split_dim;
  node->type = node_type;
  node->compute_type = compute_type;
  node->num_inputs = 1;
  node->inputs[0] = input_id;
  node->num_outputs = num_outputs;
  node->outputs[0] = output_ids[0];
  node->outputs[1] = output_ids[1];
  switch (num_outputs) {
    case 2:
      node->create = create_even_split2_operator;
      node->setup = setup_even_split2_operator;
      break;
    case 3:
      node->outputs[2] = output_ids[2];
      node->create = create_even_split3_operator;
      node->setup = setup_even_split3_operator;
      break;
    case 4:
      node->outputs[2] = output_ids[2];
      node->outputs[3] = output_ids[3];
      node->create = create_even_split4_operator;
      node->setup = setup_even_split4_operator;
      break;
    default:
      XNN_UNREACHABLE;
  }
  node->flags = flags;

  return xnn_status_success;
};

enum xnn_status xnn_define_even_split2(
  xnn_subgraph_t subgraph,
  size_t split_dim,
  uint32_t input_id,
  uint32_t output1_id,
  uint32_t output2_id,
  uint32_t flags)
{
  const uint32_t output_ids[2] = { output1_id, output2_id };
  return xnn_define_even_split_n(
    xnn_node_type_even_split2, subgraph, split_dim, input_id, XNN_COUNT_OF(output_ids), output_ids, flags);
}

enum xnn_status xnn_define_even_split3(
  xnn_subgraph_t subgraph,
  size_t split_dim,
  uint32_t input_id,
  uint32_t output1_id,
  uint32_t output2_id,
  uint32_t output3_id,
  uint32_t flags)
{
  const uint32_t output_ids[3] = { output1_id, output2_id, output3_id };
  return xnn_define_even_split_n(
    xnn_node_type_even_split3, subgraph, split_dim, input_id, XNN_COUNT_OF(output_ids), output_ids, flags);
}

enum xnn_status xnn_define_even_split4(
  xnn_subgraph_t subgraph,
  size_t split_dim,
  uint32_t input_id,
  uint32_t output1_id,
  uint32_t output2_id,
  uint32_t output3_id,
  uint32_t output4_id,
  uint32_t flags)
{
  const uint32_t output_ids[4] = { output1_id, output2_id, output3_id, output4_id };
  return xnn_define_even_split_n(
    xnn_node_type_even_split4, subgraph, split_dim, input_id, XNN_COUNT_OF(output_ids), output_ids, flags);
}
