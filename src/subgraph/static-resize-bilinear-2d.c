// Copyright 2020 Google LLC
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


static enum xnn_status create_resize_bilinear_operator(
  const struct xnn_node* node,
  const struct xnn_value* values,
  size_t num_values,
  struct xnn_operator_data* opdata,
  struct xnn_code_cache* code_cache,
  xnn_weights_cache_t weights_cache)
{
  assert(node->num_inputs == 1);
  const uint32_t input_id = node->inputs[0];
  assert(input_id != XNN_INVALID_VALUE_ID);
  assert(input_id < num_values);

  assert(node->num_outputs == 1);
  const uint32_t output_id = node->outputs[0];
  assert(output_id != XNN_INVALID_VALUE_ID);
  assert(output_id < num_values);
  (void) output_id;  // Silence unused warning, only use in asserts.

  const size_t output_height = node->params.static_resize.new_height;
  const size_t output_width = node->params.static_resize.new_width;
  enum xnn_status status;
  if (values[input_id].layout == xnn_layout_type_nchw) {
    switch (node->compute_type) {
      case xnn_compute_type_fp16:
        status = xnn_create_resize_bilinear2d_nchw_f16(
          output_height, output_width,
          node->flags,
          &opdata->operator_objects[0]);
        break;
      case xnn_compute_type_fp32:
        status = xnn_create_resize_bilinear2d_nchw_f32(
          output_height, output_width,
          node->flags,
          &opdata->operator_objects[0]);
        break;
      default:
        XNN_UNREACHABLE;
    }
  } else {
    assert(values[input_id].layout == xnn_layout_type_nhwc);
    assert(values[output_id].layout == xnn_layout_type_nhwc);
    switch (node->compute_type) {
      case xnn_compute_type_fp16:
        status = xnn_create_resize_bilinear2d_nhwc_f16(
          output_height, output_width,
          node->flags,
          &opdata->operator_objects[0]);
        break;
      case xnn_compute_type_fp32:
        status = xnn_create_resize_bilinear2d_nhwc_f32(
          output_height, output_width,
          node->flags,
          &opdata->operator_objects[0]);
        break;
      case xnn_compute_type_qs8:
        status = xnn_create_resize_bilinear2d_nhwc_s8(
          output_height, output_width,
          node->flags,
          &opdata->operator_objects[0]);
        break;
      case xnn_compute_type_qu8:
        status = xnn_create_resize_bilinear2d_nhwc_u8(
          output_height, output_width,
          node->flags,
          &opdata->operator_objects[0]);
        break;
      default:
        XNN_UNREACHABLE;
    }
  }
  return status;
}

static enum xnn_status reshape_resize_bilinear_operator(
  struct xnn_operator_data* opdata,
  struct xnn_value* values,
  size_t num_values,
  pthreadpool_t threadpool)
{
  const uint32_t input_id = opdata->inputs[0];
  assert(input_id < num_values);
  const size_t batch_size = values[input_id].shape.dim[0];
  const size_t input_height = values[input_id].shape.dim[1];
  const size_t input_width = values[input_id].shape.dim[2];
  const size_t channel_dim = values[input_id].shape.dim[3];
  assert(channel_dim == values[input_id].shape.dim[3]);
  enum xnn_status status = xnn_status_invalid_state;
  const size_t old_workspace_size = opdata->workspace_size;
  switch (opdata->operator_objects[0]->type) {
    case xnn_operator_type_resize_bilinear_nchw_f16:
      status = xnn_reshape_resize_bilinear2d_nchw_f16(
        opdata->operator_objects[0],
        batch_size,
        input_height,
        input_width,
        channel_dim,
        channel_dim,
        channel_dim,
        threadpool);
      break;
    case xnn_operator_type_resize_bilinear_nchw_f32:
      status = xnn_reshape_resize_bilinear2d_nchw_f32(
        opdata->operator_objects[0],
        batch_size,
        input_height,
        input_width,
        channel_dim,
        channel_dim,
        channel_dim,
        threadpool);
      break;
    case xnn_operator_type_resize_bilinear_nhwc_f16:
      status = xnn_reshape_resize_bilinear2d_nhwc_f16(
        opdata->operator_objects[0],
        batch_size,
        input_height,
        input_width,
        channel_dim,
        channel_dim,
        channel_dim,
        &opdata->workspace_size,
        &opdata->workspace_alignment,
        threadpool);
      break;
    case xnn_operator_type_resize_bilinear_nhwc_f32:
      status = xnn_reshape_resize_bilinear2d_nhwc_f32(
        opdata->operator_objects[0],
        batch_size,
        input_height,
        input_width,
        channel_dim,
        channel_dim,
        channel_dim,
        &opdata->workspace_size,
        &opdata->workspace_alignment,
        threadpool);
      break;
    case xnn_operator_type_resize_bilinear_nhwc_s8:
      status = xnn_reshape_resize_bilinear2d_nhwc_s8(
        opdata->operator_objects[0],
        batch_size,
        input_height,
        input_width,
        channel_dim,
        channel_dim,
        channel_dim,
        &opdata->workspace_size,
        &opdata->workspace_alignment,
        threadpool);
      break;
    case xnn_operator_type_resize_bilinear_nhwc_u8:
      status = xnn_reshape_resize_bilinear2d_nhwc_u8(
        opdata->operator_objects[0],
        batch_size,
        input_height,
        input_width,
        channel_dim,
        channel_dim,
        channel_dim,
        &opdata->workspace_size,
        &opdata->workspace_alignment,
        threadpool);
      break;
    default:
      XNN_UNREACHABLE;
  }
  if (status != xnn_status_success) {
    return status;
  }

  const size_t output_height = opdata->operator_objects[0]->output_height;
  const size_t output_width = opdata->operator_objects[0]->output_width;
  const uint32_t output_id = opdata->outputs[0];
  assert(output_id < num_values);
  struct xnn_value* output_value = values + output_id;
  output_value->shape.num_dims = 4;
  output_value->shape.dim[0] = batch_size;
  output_value->shape.dim[1] = output_height;
  output_value->shape.dim[2] = output_width;
  output_value->shape.dim[3] = channel_dim;
  const size_t new_size = xnn_tensor_get_size(output_value);
  if (new_size > output_value->size || opdata->workspace_size > old_workspace_size) {
    output_value->size = new_size;
    return xnn_status_reallocation_required;
  }

  return xnn_status_success;
}

static enum xnn_status setup_resize_bilinear_operator(
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
    case xnn_operator_type_resize_bilinear_nchw_f16:
      return xnn_setup_resize_bilinear2d_nchw_f16(
        opdata->operator_objects[0],
        input_data,
        output_data);
      break;
    case xnn_operator_type_resize_bilinear_nchw_f32:
      return xnn_setup_resize_bilinear2d_nchw_f32(
        opdata->operator_objects[0],
        input_data,
        output_data);
      break;
    case xnn_operator_type_resize_bilinear_nhwc_f16:
      return xnn_setup_resize_bilinear2d_nhwc_f16(
        opdata->operator_objects[0],
        opdata->workspace,
        input_data,
        output_data);
      break;
    case xnn_operator_type_resize_bilinear_nhwc_f32:
      return xnn_setup_resize_bilinear2d_nhwc_f32(
        opdata->operator_objects[0],
        opdata->workspace,
        input_data,
        output_data);
      break;
    case xnn_operator_type_resize_bilinear_nhwc_s8:
      return xnn_setup_resize_bilinear2d_nhwc_s8(
        opdata->operator_objects[0],
        opdata->workspace,
        input_data,
        output_data);
      break;
    case xnn_operator_type_resize_bilinear_nhwc_u8:
      return xnn_setup_resize_bilinear2d_nhwc_u8(
        opdata->operator_objects[0],
        opdata->workspace,
        input_data,
        output_data);
      break;
    default:
      XNN_UNREACHABLE;
  }
}

enum xnn_status xnn_define_static_resize_bilinear_2d(
  xnn_subgraph_t subgraph,
  size_t new_height,
  size_t new_width,
  uint32_t input_id,
  uint32_t output_id,
  uint32_t flags)
{
  enum xnn_status status;
  if ((status = xnn_subgraph_check_xnnpack_initialized(xnn_node_type_static_resize_bilinear_2d)) != xnn_status_success) {
    return status;
  }

  if (new_width == 0 || new_height == 0) {
    xnn_log_error(
      "failed to define %s operator with %zux%zu output: output dimensions must be non-zero",
      xnn_node_type_to_string(xnn_node_type_static_resize_bilinear_2d), new_width, new_height);
    return xnn_status_invalid_parameter;
  }

  if (max(new_width, new_height) >= 16777216) {
    xnn_log_error(
      "failed to define %s operator with %zux%zu output: output dimensions must be below 2**24",
      xnn_node_type_to_string(xnn_node_type_static_resize_bilinear_2d), new_width, new_height);
    return xnn_status_unsupported_parameter;
  }

  const uint32_t supported_flags = XNN_FLAG_TENSORFLOW_LEGACY_MODE |
                                   XNN_FLAG_ALIGN_CORNERS |
                                   XNN_FLAG_TRANSIENT_INDIRECTION_BUFFER;
  const uint32_t invalid_flags = flags & ~supported_flags;
  if (invalid_flags != 0) {
    xnn_log_error(
      "failed to define %s operator with 0x%08" PRIx32 " flags: invalid flags 0x%08" PRIx32,
      xnn_node_type_to_string(xnn_node_type_static_resize_bilinear_2d), flags, invalid_flags);
    return xnn_status_invalid_parameter;
  }

  const uint32_t exclusive_flags = XNN_FLAG_TENSORFLOW_LEGACY_MODE | XNN_FLAG_ALIGN_CORNERS;
  if ((flags & exclusive_flags) == exclusive_flags) {
    xnn_log_error(
      "failed to define %s operator with both XNN_FLAG_TENSORFLOW_LEGACY_MODE and XNN_FLAG_ALIGN_CORNERS flags: "
      "the two flags are mutually exclusive",
      xnn_node_type_to_string(xnn_node_type_static_resize_bilinear_2d));
    return xnn_status_invalid_parameter;
  }

  if ((status = xnn_subgraph_check_input_node_id(xnn_node_type_static_resize_bilinear_2d, input_id, subgraph->num_values)) !=
      xnn_status_success) {
    return status;
  }

  const struct xnn_value* input_value = &subgraph->values[input_id];
  status = xnn_subgraph_check_input_type_dense(xnn_node_type_static_resize_bilinear_2d, input_id, input_value);
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
        xnn_node_type_to_string(xnn_node_type_static_resize_bilinear_2d), input_id,
        xnn_datatype_to_string(input_value->datatype), input_value->datatype);
      return xnn_status_invalid_parameter;
  }

  status = xnn_subgraph_check_output_node_id(xnn_node_type_static_resize_bilinear_2d, output_id, subgraph->num_values);
  if (status != xnn_status_success) {
    return status;
  }

  const struct xnn_value* output_value = &subgraph->values[output_id];
  status = xnn_subgraph_check_output_type_dense(xnn_node_type_static_resize_bilinear_2d, output_id, output_value);
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
        xnn_node_type_to_string(xnn_node_type_static_resize_bilinear_2d), output_id,
        xnn_datatype_to_string(output_value->datatype), output_value->datatype);
      return xnn_status_invalid_parameter;
  }

  status = xnn_subgraph_check_quantization_parameter_matches(
      xnn_node_type_static_resize_bilinear_2d, input_id, input_value, output_id, output_value);
  if (status != xnn_status_success) {
    return status;
  }

  struct xnn_node* node = xnn_subgraph_new_node(subgraph);
  if (node == NULL) {
    return xnn_status_out_of_memory;
  }

  node->params.static_resize.new_height = new_height;
  node->params.static_resize.new_width = new_width;

  node->type = xnn_node_type_static_resize_bilinear_2d;
  node->compute_type = compute_type;
  node->num_inputs = 1;
  node->inputs[0] = input_id;
  node->num_outputs = 1;
  node->outputs[0] = output_id;
  node->flags = flags;

  node->create = create_resize_bilinear_operator;
  node->reshape = reshape_resize_bilinear_operator;
  node->setup = setup_resize_bilinear_operator;

  return xnn_status_success;
}
