// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <inttypes.h>
#include <stddef.h>
#include <stdint.h>

#include "xnnpack.h"
#include "xnnpack/common.h"
#include "xnnpack/config.h"
#include "xnnpack/internal.h"
#include "xnnpack/log.h"
#include "xnnpack/microparams.h"
#include "xnnpack/node-type.h"
#include "xnnpack/operator-type.h"
#include "xnnpack/operator-utils.h"
#include "xnnpack/operator.h"
#include "xnnpack/reshape-helpers.h"
#include "xnnpack/subgraph-validation.h"
#include "xnnpack/subgraph.h"
#include "pthreadpool.h"


static enum xnn_status create_convert_operator(
  const struct xnn_node* node,
  const struct xnn_value* values,
  size_t num_values,
  struct xnn_operator_data* opdata,
  struct xnn_code_cache* code_cache,
  xnn_weights_cache_t weights_cache)
{
  assert(node->num_inputs == 1);
  const uint32_t input_id = node->inputs[0];
  assert(input_id < num_values);
  const struct xnn_value* input_value = values + input_id;

  assert(node->num_outputs == 1);
  const uint32_t output_id = node->outputs[0];
  assert(output_id != XNN_INVALID_VALUE_ID);
  assert(output_id < num_values);
  const struct xnn_value* output_value = values + output_id;

  enum xnn_status status = xnn_status_uninitialized;
  const enum xnn_datatype input_datatype = input_value->datatype;
  const enum xnn_datatype output_datatype = output_value->datatype;
  switch (input_datatype) {
    case xnn_datatype_fp32:
      switch (output_datatype) {
        case xnn_datatype_qdint8:
          status = xnn_create_convert_nc_f32_qd8(
              node->flags,
              &opdata->operator_objects[0]);
          break;
        case xnn_datatype_qduint8:
          status = xnn_create_convert_nc_f32_qdu8(
              node->flags,
              &opdata->operator_objects[0]);
          break;
        case xnn_datatype_qpint8:
          status = xnn_create_convert_nc_f32_qp8(node->flags,
                                                 &opdata->operator_objects[0]);
          break;
        default:
          break;
      }
      break;
    case xnn_datatype_fp16:
      switch (output_datatype) {
        case xnn_datatype_qdint8:
          status = xnn_create_convert_nc_f16_qd8(
            node->flags,
            &opdata->operator_objects[0]);
          break;
        case xnn_datatype_qduint8:
          status = xnn_create_convert_nc_f16_qdu8(
              node->flags,
              &opdata->operator_objects[0]);
          break;
        default:
          break;
      }
      break;
    default:
      break;
  }
  if (status == xnn_status_uninitialized) {
    status = xnn_create_unary_elementwise_nc(xnn_unary_convert, input_datatype,
                                             output_datatype, NULL, NULL, NULL,
                                             node->flags,
                                             &opdata->operator_objects[0]);
  }
  return status;
}

static enum xnn_status reshape_convert_operator(
  struct xnn_operator_data* opdata,
  struct xnn_value* values,
  size_t num_values,
  pthreadpool_t threadpool)
{
  const uint32_t input_id = opdata->inputs[0];
  assert(input_id < num_values);
  const struct xnn_value* input_value = values + input_id;
  const size_t batch_size = xnn_shape_multiply_non_channel_dims(&input_value->shape);
  const size_t num_input_dims = input_value->shape.num_dims;
  const size_t channel_dim = num_input_dims == 0 ? 1 : input_value->shape.dim[num_input_dims - 1];
  const size_t old_workspace_size = opdata->workspace_size;
  enum xnn_status status = xnn_status_invalid_state;

  const uint32_t output_id = opdata->outputs[0];
  assert(output_id < num_values);
  const struct xnn_value* output_value = values + output_id;
  // Channel stride depends on number of non batch dims.
  size_t num_nonbatch_dims = output_value->quantization.num_nonbatch_dims;
  size_t dq_batch_size = xnn_shape_multiply_batch_dims(&input_value->shape, num_nonbatch_dims);
  size_t dq_channel_stride = xnn_shape_multiply_trailing_dims(&input_value->shape, num_input_dims - num_nonbatch_dims);
  switch (opdata->operator_objects[0]->type) {
    case xnn_operator_type_convert_nc_f16_qd8: {
      status = xnn_reshape_convert_nc_f16_qd8(
        opdata->operator_objects[0],
        dq_batch_size,
        /*channels=*/dq_channel_stride, /*input_stride=*/dq_channel_stride,  /*output_stride=*/dq_channel_stride,
        threadpool);
      break;
    }
    case xnn_operator_type_convert_nc_f32_qd8: {
      status = xnn_reshape_convert_nc_f32_qd8(
        opdata->operator_objects[0],
        dq_batch_size,
        /*channels=*/dq_channel_stride, /*input_stride=*/dq_channel_stride,  /*output_stride=*/dq_channel_stride,
        threadpool);
      break;
    }
    case xnn_operator_type_convert_nc_f32_qdu8: {
      status = xnn_reshape_convert_nc_f32_qdu8(
        opdata->operator_objects[0],
        dq_batch_size,
        /*channels=*/dq_channel_stride, /*input_stride=*/dq_channel_stride,  /*output_stride=*/dq_channel_stride,
        threadpool);
      break;
    }
    case xnn_operator_type_convert_nc_f16_qdu8: {
      status = xnn_reshape_convert_nc_f16_qdu8(
        opdata->operator_objects[0],
        dq_batch_size,
        /*channels=*/dq_channel_stride, /*input_stride=*/dq_channel_stride,  /*output_stride=*/dq_channel_stride,
        threadpool);
      break;
    }
    case xnn_operator_type_convert_nc_f32_qp8: {
      num_nonbatch_dims = 1;
      dq_batch_size =
          xnn_shape_multiply_batch_dims(&input_value->shape, num_nonbatch_dims);
      dq_channel_stride = xnn_shape_multiply_trailing_dims(
          &input_value->shape, num_input_dims - num_nonbatch_dims);
      status = xnn_reshape_convert_nc_f32_qp8(
          opdata->operator_objects[0], dq_batch_size,
          /*channels=*/dq_channel_stride, /*input_stride=*/dq_channel_stride,
          threadpool);
      break;
    }
    default:
      status = xnn_reshape_unary_elementwise_nc(
        opdata->operator_objects[0],
        batch_size,
        /*channels=*/channel_dim, /*input_stride=*/channel_dim, /*output_stride=*/channel_dim,
        threadpool);
      break;
  }
  if (status != xnn_status_success) {
    return status;
  }
  return resize_unary_elementwise_output_tensor(opdata, values, num_values, old_workspace_size, threadpool);
}

static enum xnn_status setup_convert_operator(
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
    case xnn_operator_type_convert_nc_f16_qd8:
    {
      void* quantization_params = output_value->quantization.dynamic_params;
      assert(quantization_params != NULL);
      return xnn_setup_convert_nc_f16_qd8(
        opdata->operator_objects[0],
        input_data,
        output_data,
        quantization_params);
    }
    case xnn_operator_type_convert_nc_f32_qd8:
    {
      void* quantization_params = output_value->quantization.dynamic_params;
      assert(quantization_params != NULL);
      return xnn_setup_convert_nc_f32_qd8(
        opdata->operator_objects[0],
        input_data,
        output_data,
        quantization_params);
    }
    case xnn_operator_type_convert_nc_f16_qdu8:
    {
      void* quantization_params = output_value->quantization.dynamic_params;
      assert(quantization_params != NULL);
      return xnn_setup_convert_nc_f16_qdu8(
        opdata->operator_objects[0],
        input_data,
        output_data,
        quantization_params);
    }
    case xnn_operator_type_convert_nc_f32_qdu8:
    {
      void* quantization_params = output_value->quantization.dynamic_params;
      assert(quantization_params != NULL);
      return xnn_setup_convert_nc_f32_qdu8(
        opdata->operator_objects[0],
        input_data,
        output_data,
        quantization_params);
    }
    case xnn_operator_type_convert_nc_f32_qp8:
      return xnn_setup_convert_nc_f32_qp8(opdata->operator_objects[0],
                                          input_data, output_data);
    default:
      return xnn_setup_unary_elementwise_nc(opdata->operator_objects[0], input_data, output_data);
  }
}

void xnn_init_convert_node(
  struct xnn_node* node,
  uint32_t input_id,
  uint32_t output_id,
  uint32_t flags)
{
  node->type = xnn_node_type_convert;
  node->num_inputs = 1;
  node->inputs[0] = input_id;
  node->num_outputs = 1;
  node->outputs[0] = output_id;
  node->flags = flags;

  node->create = create_convert_operator;
  node->reshape = reshape_convert_operator;
  node->setup = setup_convert_operator;
}

static enum xnn_status create_unary_operator(
  const struct xnn_node* node,
  const struct xnn_value* values,
  size_t num_values,
  struct xnn_operator_data* opdata,
  struct xnn_code_cache* code_cache,
  xnn_weights_cache_t weights_cache)
{
  assert(node->num_inputs == 1);
  assert(node->num_outputs == 1);

  const struct xnn_value* value_in = &values[node->inputs[0]];
  const struct xnn_value* value_out = &values[node->outputs[0]];

  struct xnn_quantization_params in_quantization = {
    .scale = value_in->quantization.scale,
    .zero_point = value_in->quantization.zero_point,
  };
  struct xnn_quantization_params out_quantization = {
    .scale = value_out->quantization.scale,
    .zero_point = value_out->quantization.zero_point,
  };

  return xnn_create_unary_elementwise_nc(
    node->unary_operator,
    value_in->datatype,
    value_out->datatype,
    &node->params.unary,
    &in_quantization,
    &out_quantization,
    node->flags,
    &opdata->operator_objects[0]);
}

static enum xnn_status reshape_unary_operator(
  struct xnn_operator_data* opdata,
  struct xnn_value* values,
  size_t num_values,
  pthreadpool_t threadpool)
{
  xnn_operator_t op = opdata->operator_objects[0];
  const uint32_t input_id = opdata->inputs[0];
  assert(input_id < num_values);
  const size_t batch_size = xnn_shape_multiply_non_channel_dims(&values[input_id].shape);
  const size_t num_input_dims = values[input_id].shape.num_dims;
  const size_t channel_dim = num_input_dims == 0 ? 1 : values[input_id].shape.dim[num_input_dims - 1];
  const size_t old_workspace_size = opdata->workspace_size;

  enum xnn_status status = xnn_reshape_unary_elementwise_nc(op, batch_size, channel_dim, channel_dim, channel_dim, threadpool);
  if (status != xnn_status_success) {
    return status;
  }
  return resize_unary_elementwise_output_tensor(opdata, values, num_values, old_workspace_size, threadpool);
}

static enum xnn_status setup_unary_operator(
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

  xnn_operator_t op = opdata->operator_objects[0];

  return xnn_setup_unary_elementwise_nc(op, input_data, output_data);
}

enum xnn_status xnn_define_unary(
  xnn_subgraph_t subgraph,
  enum xnn_unary_operator type,
  const union xnn_unary_params* params,
  uint32_t input_id,
  uint32_t output_id,
  uint32_t flags)
{
  enum xnn_status status;
  if ((status = xnn_subgraph_check_xnnpack_initialized(xnn_node_type_unary_elementwise)) != xnn_status_success) {
    return status;
  }

  if ((status = xnn_subgraph_check_input_node_id(xnn_node_type_unary_elementwise, input_id, subgraph->num_values)) != xnn_status_success) {
    return status;
  }

  if ((status = xnn_subgraph_check_output_node_id(xnn_node_type_unary_elementwise, output_id, subgraph->num_values)) != xnn_status_success) {
    return status;
  }

  switch (type) {
    case xnn_unary_clamp:
    case xnn_unary_leaky_relu:
    case xnn_unary_elu:
      if (!params) {
        xnn_log_error(
          "failed to define %s node with input ID #%" PRIu32 " and output ID #%" PRIu32
          ": missing clamp params",
          xnn_unary_operator_to_string(type), input_id, output_id);
        return xnn_status_invalid_parameter;
      }
    default:
      break;
  }

  const struct xnn_value* input_value = &subgraph->values[input_id];
  status = xnn_subgraph_check_input_type_dense(xnn_node_type_unary_elementwise, input_id, input_value);
  if (status != xnn_status_success) {
    return status;
  }

  const struct xnn_value* output_value = &subgraph->values[output_id];
  status = xnn_subgraph_check_output_type_dense(xnn_node_type_unary_elementwise, output_id, output_value);
  if (status != xnn_status_success) {
    return status;
  }

  if (type == xnn_unary_convert) {
    // Some convert types are not elementwise ops, handle them now.
    if (output_value->datatype == xnn_datatype_qdint8 ||
        output_value->datatype == xnn_datatype_qduint8) {
      struct xnn_node* node = xnn_subgraph_new_node(subgraph);
      if (node == NULL) {
        return xnn_status_out_of_memory;
      }
      xnn_init_convert_node(node, input_id, output_id, flags);
      return xnn_status_success;
    }
  }

  struct xnn_node* node = xnn_subgraph_new_node(subgraph);
  if (node == NULL) {
    return xnn_status_out_of_memory;
  }

  node->type = xnn_node_type_unary_elementwise;
  node->unary_operator = type;
  node->num_inputs = 1;
  node->inputs[0] = input_id;
  node->num_outputs = 1;
  node->outputs[0] = output_id;
  node->flags = flags;
  if (params) {
    node->params.unary = *params;
  }

  if (type == xnn_unary_clamp) {
    assert(params);
    node->activation.output_min = params->clamp.min;
    node->activation.output_max = params->clamp.max;
  }

  node->create = create_unary_operator;
  node->reshape = reshape_unary_operator;
  node->setup = setup_unary_operator;

  return xnn_status_success;
}
