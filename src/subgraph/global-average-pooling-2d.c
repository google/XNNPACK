// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <math.h>
#include <stddef.h>
#include <stdint.h>

#include <xnnpack.h>
#include <xnnpack/log.h>
#include <xnnpack/params.h>
#include <xnnpack/subgraph.h>


static enum xnn_status create_global_average_pooling_operator(
  const struct xnn_node* node,
  const struct xnn_value* values,
  size_t num_values,
  struct xnn_operator_data* opdata)
{
  assert(node->num_inputs == 1);
  const uint32_t input_id = node->inputs[0];
  assert(input_id != XNN_INVALID_VALUE_ID);
  assert(input_id < num_values);

  assert(node->num_outputs == 1);
  const uint32_t output_id = node->outputs[0];
  assert(output_id != XNN_INVALID_VALUE_ID);
  assert(output_id < num_values);

  const size_t num_input_dims = values[input_id].shape.num_dims;
  assert(num_input_dims >= 1);
  const size_t channel_dim = values[input_id].shape.dim[num_input_dims - 1];

  enum xnn_status status;
  if (values[node->inputs[0]].layout == xnn_layout_type_nchw) {
    assert(node->compute_type == xnn_compute_type_fp32);
    status = xnn_create_global_average_pooling_ncw_f32(
      channel_dim /* channels */,
      node->activation.output_min,
      node->activation.output_max,
      node->flags,
      &opdata->operator_object);
  } else {
    assert(values[node->inputs[0]].layout == xnn_layout_type_nhwc);
    assert(values[node->outputs[0]].layout == xnn_layout_type_nhwc);
    switch (node->compute_type) {
      case xnn_compute_type_fp32:
        status = xnn_create_global_average_pooling_nwc_f32(
          channel_dim /* channels */, channel_dim /* input stride */, channel_dim /* output stride */,
          node->activation.output_min,
          node->activation.output_max,
          node->flags,
          &opdata->operator_object);
        break;
#ifndef XNN_NO_F16_OPERATORS
      case xnn_compute_type_fp16:
        status = xnn_create_global_average_pooling_nwc_f16(
          channel_dim /* channels */, channel_dim /* input stride */, channel_dim /* output stride */,
          node->activation.output_min,
          node->activation.output_max,
          node->flags,
          &opdata->operator_object);
        break;
#endif  // !defined(XNN_NO_F16_OPERATORS)
#ifndef XNN_NO_QS8_OPERATORS
      case xnn_compute_type_qs8:
      {
        const float output_scale = values[output_id].quantization.scale;
        const int32_t output_zero_point = values[output_id].quantization.zero_point;
        const int8_t output_min =
          (int8_t) lrintf(fminf(fmaxf(node->activation.output_min / output_scale + (float) output_zero_point, -128.0f), 127.0f));
        const int8_t output_max =
          (int8_t) lrintf(fminf(fmaxf(node->activation.output_max / output_scale + (float) output_zero_point, -128.0f), 127.0f));
        status = xnn_create_global_average_pooling_nwc_qs8(
          channel_dim /* channels */, channel_dim /* input stride */, channel_dim /* output stride */,
          (int8_t) values[input_id].quantization.zero_point, values[input_id].quantization.scale,
          (int8_t) values[output_id].quantization.zero_point, values[output_id].quantization.scale,
          output_min,
          output_max,
          node->flags,
          &opdata->operator_object);
        break;
      }
#endif  // !defined(XNN_NO_QS8_OPERATORS)
#ifndef XNN_NO_QU8_OPERATORS
      case xnn_compute_type_qu8:
      {
        const float output_scale = values[output_id].quantization.scale;
        const int32_t output_zero_point = values[output_id].quantization.zero_point;
        const uint8_t output_min =
          (uint8_t) lrintf(fminf(fmaxf(node->activation.output_min / output_scale + (float) output_zero_point, 0.0f), 255.0f));
        const uint8_t output_max =
          (uint8_t) lrintf(fminf(fmaxf(node->activation.output_max / output_scale + (float) output_zero_point, 0.0f), 255.0f));
        status = xnn_create_global_average_pooling_nwc_qu8(
          channel_dim /* channels */, channel_dim /* input stride */, channel_dim /* output stride */,
          (uint8_t) values[input_id].quantization.zero_point, values[input_id].quantization.scale,
          (uint8_t) values[output_id].quantization.zero_point, values[output_id].quantization.scale,
          output_min,
          output_max,
          node->flags,
          &opdata->operator_object);
        break;
      }
#endif  // !defined(XNN_NO_QU8_OPERATORS)
      default:
        XNN_UNREACHABLE;
    }
  }
  if (status == xnn_status_success) {
    opdata->batch_size = values[input_id].shape.dim[0];
    opdata->input_width = values[input_id].shape.dim[1] * values[input_id].shape.dim[2];
    opdata->inputs[0] = input_id;
    opdata->outputs[0] = output_id;
  }
  return status;
}

static enum xnn_status setup_global_average_pooling_operator(
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

  switch (opdata->operator_object->type) {
    case xnn_operator_type_global_average_pooling_ncw_f32:
      return xnn_setup_global_average_pooling_ncw_f32(
        opdata->operator_object,
        opdata->batch_size,
        opdata->input_width,
        input_data,
        output_data,
        threadpool);
      break;
    case xnn_operator_type_global_average_pooling_nwc_f32:
      return xnn_setup_global_average_pooling_nwc_f32(
        opdata->operator_object,
        opdata->batch_size,
        opdata->input_width,
        input_data,
        output_data,
        threadpool);
      break;
#ifndef XNN_NO_F16_OPERATORS
    case xnn_operator_type_global_average_pooling_nwc_f16:
      return xnn_setup_global_average_pooling_nwc_f16(
        opdata->operator_object,
        opdata->batch_size,
        opdata->input_width,
        input_data,
        output_data,
        threadpool);
      break;
#endif  // !defined(XNN_NO_F16_OPERATORS)
#ifndef XNN_NO_QS8_OPERATORS
    case xnn_operator_type_global_average_pooling_nwc_qs8:
      return xnn_setup_global_average_pooling_nwc_qs8(
        opdata->operator_object,
        opdata->batch_size,
        opdata->input_width,
        input_data,
        output_data,
        threadpool);
      break;
#endif  // !defined(XNN_NO_QS8_OPERATORS)
#ifndef XNN_NO_QU8_OPERATORS
    case xnn_operator_type_global_average_pooling_nwc_qu8:
      return xnn_setup_global_average_pooling_nwc_qu8(
        opdata->operator_object,
        opdata->batch_size,
        opdata->input_width,
        input_data,
        output_data,
        threadpool);
      break;
#endif  // !defined(XNN_NO_QU8_OPERATORS)
    default:
      XNN_UNREACHABLE;
  }
}

enum xnn_status xnn_define_global_average_pooling_2d(
  xnn_subgraph_t subgraph,
  float output_min,
  float output_max,
  uint32_t input_id,
  uint32_t output_id,
  uint32_t flags)
{
  if ((xnn_params.init_flags & XNN_INIT_FLAG_XNNPACK) == 0) {
    xnn_log_error("failed to define %s operator: XNNPACK is not initialized",
      xnn_node_type_to_string(xnn_node_type_global_average_pooling_2d));
    return xnn_status_uninitialized;
  }

  if (isnan(output_min)) {
    xnn_log_error(
      "failed to define %s operator with NaN output lower bound: lower bound must be non-NaN",
      xnn_node_type_to_string(xnn_node_type_global_average_pooling_2d));
    return xnn_status_invalid_parameter;
  }

  if (isnan(output_max)) {
    xnn_log_error(
      "failed to define %s operator with NaN output upper bound: upper bound must be non-NaN",
      xnn_node_type_to_string(xnn_node_type_global_average_pooling_2d));
    return xnn_status_invalid_parameter;
  }

  if (output_min >= output_max) {
    xnn_log_error(
      "failed to define %s operator with [%.7g, %.7g] output range: lower bound must be below upper bound",
      xnn_node_type_to_string(xnn_node_type_global_average_pooling_2d), output_min, output_max);
    return xnn_status_invalid_parameter;
  }

  if (input_id >= subgraph->num_values) {
    xnn_log_error(
      "failed to define %s operator with input ID #%" PRIu32 ": invalid Value ID",
      xnn_node_type_to_string(xnn_node_type_global_average_pooling_2d), input_id);
    return xnn_status_invalid_parameter;
  }

  const struct xnn_value* input_value = &subgraph->values[input_id];
  if (input_value->type != xnn_value_type_dense_tensor) {
    xnn_log_error(
      "failed to define %s operator with input ID #%" PRIu32 ": unsupported Value type %d (expected dense tensor)",
      xnn_node_type_to_string(xnn_node_type_global_average_pooling_2d), input_id, input_value->type);
    return xnn_status_invalid_parameter;
  }

  switch (input_value->datatype) {
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
        "failed to define %s operator with input ID #%" PRIu32 ": unsupported Value datatype %s (%d)",
        xnn_node_type_to_string(xnn_node_type_global_average_pooling_2d), input_id,
        xnn_datatype_to_string(input_value->datatype), input_value->datatype);
      return xnn_status_invalid_parameter;
  }

  if (output_id >= subgraph->num_values) {
    xnn_log_error(
      "failed to define %s operator with output ID #%" PRIu32 ": invalid Value ID",
      xnn_node_type_to_string(xnn_node_type_global_average_pooling_2d), output_id);
    return xnn_status_invalid_parameter;
  }

  const struct xnn_value* output_value = &subgraph->values[output_id];
  if (output_value->type != xnn_value_type_dense_tensor) {
    xnn_log_error(
      "failed to define %s operator with output ID #%" PRIu32 ": unsupported Value type %d (expected dense tensor)",
      xnn_node_type_to_string(xnn_node_type_global_average_pooling_2d), output_id, output_value->type);
    return xnn_status_invalid_parameter;
  }

  enum xnn_compute_type compute_type = xnn_compute_type_invalid;
  switch (output_value->datatype) {
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
        xnn_node_type_to_string(xnn_node_type_global_average_pooling_2d), output_id,
        xnn_datatype_to_string(output_value->datatype), output_value->datatype);
      return xnn_status_invalid_parameter;
  }

  if (input_value->datatype != output_value->datatype) {
    xnn_log_error(
      "failed to define %s operator with input ID #%" PRIu32 " and output ID #%" PRIu32
      ": mismatching datatypes across input (%s) and output (%s)",
      xnn_node_type_to_string(xnn_node_type_global_average_pooling_2d), input_id, output_id,
      xnn_datatype_to_string(input_value->datatype),
      xnn_datatype_to_string(output_value->datatype));
    return xnn_status_invalid_parameter;
  }

  struct xnn_node* node = xnn_subgraph_new_node(subgraph);
  if (node == NULL) {
    return xnn_status_out_of_memory;
  }

  node->type = xnn_node_type_global_average_pooling_2d;
  node->compute_type = compute_type;
  node->activation.output_min = output_min;
  node->activation.output_max = output_max;
  node->num_inputs = 1;
  node->inputs[0] = input_id;
  node->num_outputs = 1;
  node->outputs[0] = output_id;
  node->flags = flags;

  node->create = create_global_average_pooling_operator;
  node->setup = setup_global_average_pooling_operator;

  return xnn_status_success;
}
