// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <inttypes.h>
#include <stddef.h>
#include <stdint.h>

#include "xnnpack.h"
#include "xnnpack/allocation-type.h"
#include "xnnpack/common.h"
#include "xnnpack/internal.h"
#include "xnnpack/log.h"
#include "xnnpack/node-type.h"
#include "xnnpack/operator-type.h"
#include "xnnpack/operator.h"
#include "xnnpack/requantization.h"
#include "xnnpack/subgraph-validation.h"
#include "xnnpack/subgraph.h"
#include "pthreadpool.h"

static enum xnn_status create_fully_connected_operator(
  const struct xnn_node* node,
  const struct xnn_value* values,
  size_t num_values,
  struct xnn_operator_data* opdata,
  struct xnn_code_cache* code_cache,
  xnn_weights_cache_t weights_cache)
{
  assert(node->num_inputs >= 2);
  assert(node->num_inputs <= 3);
  const uint32_t input_id = node->inputs[0];
  assert(input_id != XNN_INVALID_VALUE_ID);
  assert(input_id < num_values);
  const uint32_t filter_id = node->inputs[1];
  assert(filter_id != XNN_INVALID_VALUE_ID);
  assert(filter_id < num_values);

  assert(node->num_outputs == 1);
  const uint32_t output_id = node->outputs[0];
  assert(output_id != XNN_INVALID_VALUE_ID);
  assert(output_id < num_values);

  size_t output_channels, input_channels;
  if (node->flags & XNN_FLAG_TRANSPOSE_WEIGHTS) {
    input_channels = values[node->inputs[1]].shape.dim[0];
    output_channels = values[node->inputs[1]].shape.dim[1];
  } else {
    output_channels = values[node->inputs[1]].shape.dim[0];
    input_channels = values[node->inputs[1]].shape.dim[1];
  }

  const void* kernel_data = values[filter_id].fp32_data != NULL ? values[filter_id].fp32_data : values[filter_id].data;
  bool has_non_static_weights = (kernel_data == NULL);

  const void* bias_data = NULL;
  if (node->num_inputs > 2) {
    const uint32_t bias_id = node->inputs[2];
    assert(bias_id != XNN_INVALID_VALUE_ID);
    assert(bias_id < num_values);

    bias_data = values[bias_id].fp32_data != NULL ? values[bias_id].fp32_data : values[bias_id].data;
    has_non_static_weights |= (bias_data == NULL);
  }

  enum xnn_status status;
  switch (node->compute_type) {
    case xnn_compute_type_fp16:
      if (has_non_static_weights) {
        status = xnn_create_dynamic_fully_connected_nc_f16(
          node->activation.output_min,
          node->activation.output_max,
          /*flags=*/node->flags,
          &opdata->operator_objects[0]);
      } else {
        status = xnn_create_fully_connected_nc_f16(
          input_channels,
          output_channels,
          /*input_stride=*/input_channels,
          /*output_stride=*/output_channels,
          kernel_data,
          bias_data,
          node->activation.output_min,
          node->activation.output_max,
          node->flags | XNN_FLAG_FP32_STATIC_WEIGHTS,
          code_cache,
          weights_cache,
          &opdata->operator_objects[0]);
      }
      break;
    case xnn_compute_type_fp32:
      if (has_non_static_weights) {
        status = xnn_create_dynamic_fully_connected_nc_f32(
          node->activation.output_min,
          node->activation.output_max,
          /*flags=*/node->flags,
          &opdata->operator_objects[0]);
      } else {
        switch (values[filter_id].datatype) {
          case xnn_datatype_qcint4:
            status = xnn_create_fully_connected_nc_f32_qc4w(
              input_channels,
              output_channels,
              /*input_stride=*/input_channels,
              /*output_stride=*/output_channels,
              values[filter_id].quantization.zero_point,
              values[filter_id].quantization.channelwise_scale,
              kernel_data,
              bias_data,
              node->activation.output_min,
              node->activation.output_max,
              /*flags=*/node->flags,
              code_cache,
              weights_cache,
              &opdata->operator_objects[0]);
            break;
          case xnn_datatype_qcint8:
            status = xnn_create_fully_connected_nc_f32_qc8w(
              input_channels,
              output_channels,
              /*input_stride=*/input_channels,
              /*output_stride=*/output_channels,
              values[filter_id].quantization.channelwise_scale,
              kernel_data,
              bias_data,
              node->activation.output_min,
              node->activation.output_max,
              /*flags=*/node->flags,
              code_cache,
              weights_cache,
              &opdata->operator_objects[0]);
            break;
          case xnn_datatype_fp32:
            status = xnn_create_fully_connected_nc_f32(
              input_channels,
              output_channels,
              /*input_stride=*/input_channels,
              /*output_stride=*/output_channels,
              kernel_data,
              bias_data,
              node->activation.output_min,
              node->activation.output_max,
              /*flags=*/node->flags,
              code_cache,
              weights_cache,
              &opdata->operator_objects[0]);
            break;
          default:
            XNN_UNREACHABLE;
        }
      }
      break;
    case xnn_compute_type_qd8_to_fp16:
    {
      assert(!has_non_static_weights);
      assert(kernel_data != NULL);
      switch (values[filter_id].datatype) {
        case xnn_datatype_qcint4:
          status = xnn_create_fully_connected_nc_qd8_f16_qc4w(
            input_channels,
            output_channels,
            /*input_stride=*/input_channels,
            /*output_stride=*/output_channels,
            /*kernel_zero_point=*/values[filter_id].quantization.zero_point,
            values[filter_id].quantization.channelwise_scale,
            kernel_data,
            bias_data,
            node->activation.output_min,
            node->activation.output_max,
            node->flags,
            code_cache,
            weights_cache,
            &opdata->operator_objects[0]);
          break;
        case xnn_datatype_qbint4:
          status = xnn_create_fully_connected_nc_qd8_f16_qb4w(
            input_channels,
            output_channels,
            /*input_stride=*/input_channels,
            /*output_stride=*/output_channels,
            /*block_size=*/values[filter_id].quantization.block_size,
            /*kernel_zero_point=*/values[filter_id].quantization.zero_point,
            values[filter_id].quantization.blockwise_scale,
            kernel_data,
            bias_data,
            node->activation.output_min,
            node->activation.output_max,
            node->flags,
            code_cache,
            weights_cache,
            &opdata->operator_objects[0]);
          break;
        case xnn_datatype_qcint8:
          status = xnn_create_fully_connected_nc_qd8_f16_qc8w(
            input_channels,
            output_channels,
            /*input_stride=*/input_channels,
            /*output_stride=*/output_channels,
            values[filter_id].quantization.channelwise_scale,
            kernel_data,
            bias_data,
            node->activation.output_min,
            node->activation.output_max,
            node->flags,
            code_cache,
            weights_cache,
            &opdata->operator_objects[0]);
          break;
        default:
          XNN_UNREACHABLE;
      }
      break;
    }
    case xnn_compute_type_qd8_to_fp32:
    {
      assert(!has_non_static_weights);
      assert(kernel_data != NULL);
      switch (values[filter_id].datatype) {
        case xnn_datatype_qcint4:
          status = xnn_create_fully_connected_nc_qd8_f32_qc4w(
            input_channels,
            output_channels,
            /*input_stride=*/input_channels,
            /*output_stride=*/output_channels,
            /*kernel_zero_point=*/values[filter_id].quantization.zero_point,
            values[filter_id].quantization.channelwise_scale,
            kernel_data,
            bias_data,
            node->activation.output_min,
            node->activation.output_max,
            node->flags,
            code_cache,
            weights_cache,
            &opdata->operator_objects[0]);
          break;
        case xnn_datatype_qbint4:
          status = xnn_create_fully_connected_nc_qd8_f32_qb4w(
            input_channels,
            output_channels,
            /*input_stride=*/input_channels,
            /*output_stride=*/output_channels,
            /*block_size=*/values[filter_id].quantization.block_size,
            /*kernel_zero_point=*/values[filter_id].quantization.zero_point,
            values[filter_id].quantization.blockwise_scale,
            kernel_data,
            bias_data,
            node->activation.output_min,
            node->activation.output_max,
            node->flags,
            code_cache,
            weights_cache,
            &opdata->operator_objects[0]);
          break;
        case xnn_datatype_qcint8:
          status = xnn_create_fully_connected_nc_qd8_f32_qc8w(
            input_channels,
            output_channels,
            /*input_stride=*/input_channels,
            /*output_stride=*/output_channels,
            values[filter_id].quantization.channelwise_scale,
            kernel_data,
            bias_data,
            node->activation.output_min,
            node->activation.output_max,
            node->flags,
            code_cache,
            weights_cache,
            &opdata->operator_objects[0]);
          break;
        default:
          XNN_UNREACHABLE;
      }
      break;
    }
    case xnn_compute_type_qp8_to_fp32: {
      assert(!has_non_static_weights);
      assert(kernel_data != NULL);
      switch (values[filter_id].datatype) {
        case xnn_datatype_qcint4:
          status = xnn_create_fully_connected_nc_qp8_f32_qc4w(
              input_channels, output_channels,
              /*input_stride=*/input_channels,
              /*output_stride=*/output_channels,
              /*kernel_zero_point=*/values[filter_id].quantization.zero_point,
              values[filter_id].quantization.channelwise_scale, kernel_data,
              bias_data, node->activation.output_min,
              node->activation.output_max, node->flags, code_cache,
              weights_cache, &opdata->operator_objects[0]);
          break;
        default:
          XNN_UNREACHABLE;
      }
      break;
    }
    case xnn_compute_type_qc8:
    {
      assert(!has_non_static_weights);
      assert(kernel_data != NULL);
      assert(values[filter_id].datatype == xnn_datatype_qcint8);
      const float output_scale = values[output_id].quantization.scale;
      const int32_t output_zero_point = values[output_id].quantization.zero_point;
      const int8_t output_min = xnn_qs8_quantize(node->activation.output_min, output_scale, output_zero_point);
      const int8_t output_max = xnn_qs8_quantize(node->activation.output_max, output_scale, output_zero_point);
      status = xnn_create_fully_connected_nc_qs8_qc8w(
        input_channels,
        output_channels,
        /*input_stride=*/input_channels,
        /*output_stride=*/output_channels,
        (int8_t) values[input_id].quantization.zero_point,
        values[input_id].quantization.scale,
        values[filter_id].quantization.channelwise_scale,
        kernel_data,
        bias_data,
        (int8_t) output_zero_point,
        output_scale, output_min, output_max,
        /*flags=*/node->flags,
        code_cache,
        weights_cache,
        &opdata->operator_objects[0]);
      break;
    }
    case xnn_compute_type_qs8:
    {
      assert(!has_non_static_weights);
      assert(kernel_data != NULL);
      const float output_scale = values[output_id].quantization.scale;
      const int32_t output_zero_point = values[output_id].quantization.zero_point;
      const int8_t output_min = xnn_qs8_quantize(node->activation.output_min, output_scale, output_zero_point);
      const int8_t output_max = xnn_qs8_quantize(node->activation.output_max, output_scale, output_zero_point);
      status = xnn_create_fully_connected_nc_qs8(
        input_channels,
        output_channels,
        /*input_stride=*/input_channels,
        /*output_stride=*/output_channels,
        (int8_t) values[input_id].quantization.zero_point,
        values[input_id].quantization.scale,
        values[filter_id].quantization.scale,
        kernel_data,
        bias_data,
        (int8_t) output_zero_point,
        output_scale, output_min, output_max,
        /*flags=*/node->flags,
        code_cache,
        weights_cache,
        &opdata->operator_objects[0]);
      break;
    }
    case xnn_compute_type_qu8:
    {
      assert(!has_non_static_weights);
      assert(kernel_data != NULL);
      const float output_scale = values[output_id].quantization.scale;
      const int32_t output_zero_point = values[output_id].quantization.zero_point;
      const uint8_t output_min = xnn_qu8_quantize(node->activation.output_min, output_scale, output_zero_point);
      const uint8_t output_max = xnn_qu8_quantize(node->activation.output_max, output_scale, output_zero_point);
      status = xnn_create_fully_connected_nc_qu8(
        input_channels,
        output_channels,
        /*input_stride=*/input_channels,
        /*output_stride=*/output_channels,
        (uint8_t) values[input_id].quantization.zero_point,
        values[input_id].quantization.scale,
        (uint8_t) values[filter_id].quantization.zero_point,
        values[filter_id].quantization.scale,
        kernel_data,
        bias_data,
        (uint8_t) output_zero_point,
        output_scale, output_min, output_max,
        /*flags=*/node->flags,
        code_cache,
        weights_cache,
        &opdata->operator_objects[0]);
      break;
    }
    default:
      XNN_UNREACHABLE;
  }
  return status;
}

enum xnn_status resize_fully_connected_output_tensor(
  const struct xnn_operator_data* opdata,
  struct xnn_value* values,
  size_t num_values,
  size_t old_workspace_size,
  pthreadpool_t threadpool)
{
  const uint32_t filter_id = opdata->inputs[1];
  const struct xnn_value* filter = &values[filter_id];

  const uint32_t output_id = opdata->outputs[0];
  struct xnn_value* output = (struct xnn_value*) &values[output_id];

  const uint32_t input_id = opdata->inputs[0];
  const struct xnn_value* input = &values[input_id];

  bool reshape_2d = opdata->flags & XNN_FLAG_TENSORFLOW_RESHAPE_2D;
  if (reshape_2d) {
    output->shape.num_dims = 2;
  } else {
    output->shape.num_dims = input->shape.num_dims;
  }
  // Infer output channels.
  const uint32_t filter_output_channel_index = (opdata->flags & XNN_FLAG_TRANSPOSE_WEIGHTS) ? 1 : 0;
  output->shape.dim[output->shape.num_dims - 1] = filter->shape.dim[filter_output_channel_index];

  if (reshape_2d) {
    const uint32_t filter_input_channel_index = (opdata->flags & XNN_FLAG_TRANSPOSE_WEIGHTS) ? 0 : 1;
    const size_t num_input_elements = xnn_shape_multiply_all_dims(&input->shape);
    // propogate the input shape to output.
    output->shape.dim[0] = num_input_elements / filter->shape.dim[filter_input_channel_index];
  }
  else {
    // Propagate input shape to output.
    for (size_t cur_dim = 0; cur_dim < input->shape.num_dims - 1; cur_dim++) {
      output->shape.dim[cur_dim] = input->shape.dim[cur_dim];
    }
  }

  const size_t new_size = xnn_tensor_get_size(output);
  if (new_size > output->size || old_workspace_size < opdata->workspace_size) {
    output->size = new_size;
    return xnn_status_reallocation_required;
  }

  return xnn_status_success;
}

static enum xnn_status reshape_fully_connected_operator(
  struct xnn_operator_data* opdata,
  struct xnn_value* values,
  size_t num_values,
  pthreadpool_t threadpool)
{
  const uint32_t input_id = opdata->inputs[0];
  assert(input_id < num_values);
  const uint32_t filter_id = opdata->inputs[1];
  assert(filter_id < num_values);
  const size_t num_input_elements = xnn_shape_multiply_all_dims(&values[input_id].shape);
  size_t output_channels, input_channels;
  if (opdata->flags & XNN_FLAG_TRANSPOSE_WEIGHTS) {
    input_channels = values[filter_id].shape.dim[0];
    output_channels = values[filter_id].shape.dim[1];
  } else {
    output_channels = values[filter_id].shape.dim[0];
    input_channels = values[filter_id].shape.dim[1];
  }
  const size_t batch_size = num_input_elements / input_channels;
  const size_t old_workspace_size = opdata->workspace_size;
  enum xnn_status status = xnn_status_invalid_state;

  switch (opdata->operator_objects[0]->type) {
    case xnn_operator_type_dynamic_fully_connected_nc_f16:
      status = xnn_reshape_dynamic_fully_connected_nc_f16(
        opdata->operator_objects[0],
        batch_size,
        input_channels, output_channels,
        input_channels, output_channels,
        &opdata->workspace_size, &opdata->workspace_alignment,
        threadpool);
      break;
    case xnn_operator_type_dynamic_fully_connected_nc_f32:
      status = xnn_reshape_dynamic_fully_connected_nc_f32(
        opdata->operator_objects[0],
        batch_size,
        input_channels, output_channels,
        input_channels, output_channels,
        &opdata->workspace_size, &opdata->workspace_alignment,
        threadpool);
      break;
    case xnn_operator_type_fully_connected_nc_f16:
      status = xnn_reshape_fully_connected_nc_f16(
        opdata->operator_objects[0],
        batch_size,
        threadpool);
      break;
    case xnn_operator_type_fully_connected_nc_f32:
      status = xnn_reshape_fully_connected_nc_f32(
        opdata->operator_objects[0],
        batch_size,
        threadpool);
      break;
    case xnn_operator_type_fully_connected_nc_f32_qc4w:
      status = xnn_reshape_fully_connected_nc_f32_qc4w(
        opdata->operator_objects[0],
        batch_size,
        threadpool);
      break;
    case xnn_operator_type_fully_connected_nc_f32_qc8w:
      status = xnn_reshape_fully_connected_nc_f32_qc8w(
        opdata->operator_objects[0],
        batch_size,
        threadpool);
      break;
    case xnn_operator_type_fully_connected_nc_qd8_f32_qc4w:
      status = xnn_reshape_fully_connected_nc_qd8_f32_qc4w(
        opdata->operator_objects[0],
        batch_size,
        threadpool);
      break;
    case xnn_operator_type_fully_connected_nc_qd8_f16_qc4w:
      status = xnn_reshape_fully_connected_nc_qd8_f16_qc4w(
        opdata->operator_objects[0],
        batch_size,
        threadpool);
      break;
    case xnn_operator_type_fully_connected_nc_qd8_f16_qb4w:
      status = xnn_reshape_fully_connected_nc_qd8_f16_qb4w(
        opdata->operator_objects[0],
        batch_size,
        threadpool);
      break;
    case xnn_operator_type_fully_connected_nc_qd8_f32_qb4w:
      status = xnn_reshape_fully_connected_nc_qd8_f32_qb4w(
        opdata->operator_objects[0],
        batch_size,
        threadpool);
      break;
    case xnn_operator_type_fully_connected_nc_qd8_f16_qc8w:
      status = xnn_reshape_fully_connected_nc_qd8_f16_qc8w(
        opdata->operator_objects[0],
        batch_size,
        threadpool);
      break;
    case xnn_operator_type_fully_connected_nc_qd8_f32_qc8w:
      status = xnn_reshape_fully_connected_nc_qd8_f32_qc8w(
        opdata->operator_objects[0],
        batch_size,
        threadpool);
      break;
    case xnn_operator_type_fully_connected_nc_qp8_f32_qc4w:
      status = xnn_reshape_fully_connected_nc_qp8_f32_qc4w(
        opdata->operator_objects[0],
        batch_size,
        threadpool);
      break;
    case xnn_operator_type_fully_connected_nc_qs8:
      status = xnn_reshape_fully_connected_nc_qs8(
        opdata->operator_objects[0],
        batch_size,
        threadpool);
      break;
    case xnn_operator_type_fully_connected_nc_qs8_qc8w:
      status = xnn_reshape_fully_connected_nc_qs8_qc8w(
        opdata->operator_objects[0],
        batch_size,
        threadpool);
      break;
    case xnn_operator_type_fully_connected_nc_qu8:
      status = xnn_reshape_fully_connected_nc_qu8(
        opdata->operator_objects[0],
        batch_size,
        threadpool);
      break;
    default:
      XNN_UNREACHABLE;
  }

  if (status != xnn_status_success) {
    return status;
  }
  return resize_fully_connected_output_tensor(opdata, values, num_values, old_workspace_size, threadpool);
}

static enum xnn_status setup_fully_connected_operator(
  const struct xnn_operator_data* opdata,
  const struct xnn_value* values,
  size_t num_values,
  pthreadpool_t threadpool)
{
  const uint32_t input_id = opdata->inputs[0];
  assert(input_id != XNN_INVALID_VALUE_ID);
  assert(input_id < num_values);

  const uint32_t filter_id = opdata->inputs[1];
  assert(filter_id != XNN_INVALID_VALUE_ID);
  assert(filter_id < num_values);

  const uint32_t bias_id = opdata->inputs[2];

  const uint32_t output_id = opdata->outputs[0];
  assert(output_id != XNN_INVALID_VALUE_ID);
  assert(output_id < num_values);

  const struct xnn_value* input_value = values + input_id;
  const void* input_data = input_value->data;
  assert(input_data != NULL);

  const struct xnn_value* kernel_value = values + filter_id;
  bool has_dynamic_weights = kernel_value->allocation_type != xnn_allocation_type_static;
  const void* kernel_data = kernel_value->allocation_type == xnn_allocation_type_static ? NULL : kernel_value->data;

  const void* bias_data = NULL;
  if (opdata->num_inputs > 2) {
    assert(bias_id != XNN_INVALID_VALUE_ID);
    assert(bias_id < num_values);
    const struct xnn_value* bias_value = values + bias_id;
    has_dynamic_weights |= bias_value->allocation_type != xnn_allocation_type_static;
    if (has_dynamic_weights) {
      kernel_data = kernel_value->data;
      bias_data = bias_value->data;
    }
  }

  const struct xnn_value* output_value = values + output_id;
  void* output_data = output_value->data;
  assert(output_data != NULL);

  switch (opdata->operator_objects[0]->type) {
    case xnn_operator_type_dynamic_fully_connected_nc_f16:
      assert(kernel_data != NULL);
      return xnn_setup_dynamic_fully_connected_nc_f16(
        opdata->operator_objects[0],
        opdata->workspace, input_data, kernel_data, bias_data, output_data);
    case xnn_operator_type_dynamic_fully_connected_nc_f32:
      assert(kernel_data != NULL);
      return xnn_setup_dynamic_fully_connected_nc_f32(
        opdata->operator_objects[0],
        opdata->workspace, input_data, kernel_data, bias_data, output_data);
    case xnn_operator_type_fully_connected_nc_f16:
      assert(kernel_data == NULL);
      assert(bias_data == NULL);
      return xnn_setup_fully_connected_nc_f16(
        opdata->operator_objects[0],
        input_data,
        output_data);
    case xnn_operator_type_fully_connected_nc_f32:
      assert(kernel_data == NULL);
      assert(bias_data == NULL);
      return xnn_setup_fully_connected_nc_f32(
        opdata->operator_objects[0],
        input_data,
        output_data);
    case xnn_operator_type_fully_connected_nc_f32_qc4w:
      assert(kernel_data == NULL);
      assert(bias_data == NULL);
      return xnn_setup_fully_connected_nc_f32_qc4w(
        opdata->operator_objects[0],
        input_data,
        output_data);
    case xnn_operator_type_fully_connected_nc_f32_qc8w:
      assert(kernel_data == NULL);
      assert(bias_data == NULL);
      return xnn_setup_fully_connected_nc_f32_qc8w(
        opdata->operator_objects[0],
        input_data,
        output_data);
    case xnn_operator_type_fully_connected_nc_qd8_f32_qc4w:
    {
      const void* quantization_params = input_value->quantization.dynamic_params;
      assert(kernel_data == NULL);
      assert(bias_data == NULL);
      assert(quantization_params != NULL);
      return xnn_setup_fully_connected_nc_qd8_f32_qc4w(
        opdata->operator_objects[0],
        input_data,
        output_data,
        quantization_params);
    }
    case xnn_operator_type_fully_connected_nc_qd8_f16_qc4w:
    {
      const void* quantization_params = input_value->quantization.dynamic_params;
      assert(kernel_data == NULL);
      assert(bias_data == NULL);
      assert(quantization_params != NULL);
      return xnn_setup_fully_connected_nc_qd8_f16_qc4w(
        opdata->operator_objects[0],
        input_data,
        output_data,
        quantization_params);
    }
    case xnn_operator_type_fully_connected_nc_qd8_f32_qb4w:
    {
      const void* quantization_params = input_value->quantization.dynamic_params;
      assert(kernel_data == NULL);
      assert(bias_data == NULL);
      assert(quantization_params != NULL);
      return xnn_setup_fully_connected_nc_qd8_f32_qb4w(
        opdata->operator_objects[0],
        input_data,
        output_data,
        quantization_params);
    }
    case xnn_operator_type_fully_connected_nc_qd8_f16_qb4w:
    {
      const void* quantization_params = input_value->quantization.dynamic_params;
      assert(kernel_data == NULL);
      assert(bias_data == NULL);
      assert(quantization_params != NULL);
      return xnn_setup_fully_connected_nc_qd8_f16_qb4w(
        opdata->operator_objects[0],
        input_data,
        output_data,
        quantization_params);
    }
    case xnn_operator_type_fully_connected_nc_qd8_f16_qc8w:
    {
      const void* quantization_params = input_value->quantization.dynamic_params;
      assert(kernel_data == NULL);
      assert(bias_data == NULL);
      assert(quantization_params != NULL);
      return xnn_setup_fully_connected_nc_qd8_f16_qc8w(
        opdata->operator_objects[0],
        input_data,
        output_data,
        quantization_params);
    }
    case xnn_operator_type_fully_connected_nc_qd8_f32_qc8w:
    {
      const void* quantization_params = input_value->quantization.dynamic_params;
      assert(kernel_data == NULL);
      assert(bias_data == NULL);
      assert(quantization_params != NULL);
      return xnn_setup_fully_connected_nc_qd8_f32_qc8w(
        opdata->operator_objects[0],
        input_data,
        output_data,
        quantization_params);
    }
    case xnn_operator_type_fully_connected_nc_qp8_f32_qc4w:
    {
      assert(kernel_data == NULL);
      assert(bias_data == NULL);
      return xnn_setup_fully_connected_nc_qp8_f32_qc4w(
        opdata->operator_objects[0],
        input_data,
        output_data);
    }
    case xnn_operator_type_fully_connected_nc_qs8:
      assert(kernel_data == NULL);
      assert(bias_data == NULL);
      return xnn_setup_fully_connected_nc_qs8(
        opdata->operator_objects[0],
        input_data,
        output_data);
    case xnn_operator_type_fully_connected_nc_qs8_qc8w:
      assert(kernel_data == NULL);
      assert(bias_data == NULL);
      return xnn_setup_fully_connected_nc_qs8_qc8w(
        opdata->operator_objects[0],
        input_data,
        output_data);
    case xnn_operator_type_fully_connected_nc_qu8:
      assert(kernel_data == NULL);
      assert(bias_data == NULL);
      return xnn_setup_fully_connected_nc_qu8(
        opdata->operator_objects[0],
        input_data,
        output_data);
    default:
      XNN_UNREACHABLE;
  }
}

static inline enum xnn_compute_type validate_datatypes_with_bias(
  enum xnn_datatype input_datatype,
  enum xnn_datatype kernel_datatype,
  enum xnn_datatype bias_datatype,
  enum xnn_datatype output_datatype)
{
  switch (kernel_datatype) {
    case xnn_datatype_fp32:
      if (input_datatype == xnn_datatype_fp32 &&
          bias_datatype == xnn_datatype_fp32 &&
          output_datatype == xnn_datatype_fp32)
      {
        return xnn_compute_type_fp32;
      } else if (input_datatype == xnn_datatype_fp16 &&
          bias_datatype == xnn_datatype_fp32 &&
          output_datatype == xnn_datatype_fp16) {
        // Flag: XNN_FLAG_FP32_STATIC_WEIGHTS
        return xnn_compute_type_fp16;
      }
      break;
    case xnn_datatype_qcint4:
      if (input_datatype == xnn_datatype_fp32 &&
          bias_datatype == xnn_datatype_fp32 &&
          output_datatype == xnn_datatype_fp32)
      {
        return xnn_compute_type_fp32;
      } else if (input_datatype == xnn_datatype_qdint8 &&
          bias_datatype == xnn_datatype_fp32 &&
          output_datatype == xnn_datatype_fp32)
      {
        return xnn_compute_type_qd8_to_fp32;
      } else if (input_datatype == xnn_datatype_qpint8 &&
                 bias_datatype == xnn_datatype_fp32 &&
                 output_datatype == xnn_datatype_fp32) {
        return xnn_compute_type_qp8_to_fp32;
      } else if (input_datatype == xnn_datatype_qdint8 &&
                 bias_datatype == xnn_datatype_fp32 &&
                 output_datatype == xnn_datatype_fp16) {
        return xnn_compute_type_qd8_to_fp16;
      }
      break;
    case xnn_datatype_qbint4:
      if (input_datatype == xnn_datatype_qdint8 &&
          bias_datatype == xnn_datatype_fp32 &&
          output_datatype == xnn_datatype_fp32)
      {
        return xnn_compute_type_qd8_to_fp32;
      } else if (input_datatype == xnn_datatype_qdint8 &&
          bias_datatype == xnn_datatype_fp32 &&
          output_datatype == xnn_datatype_fp16)
      {
        return xnn_compute_type_qd8_to_fp16;
      }
      break;
    case xnn_datatype_qcint8:
      if (input_datatype == xnn_datatype_fp32 &&
          bias_datatype == xnn_datatype_fp32 &&
          output_datatype == xnn_datatype_fp32)
      {
        return xnn_compute_type_fp32;
      } else if (input_datatype == xnn_datatype_qdint8 &&
          bias_datatype == xnn_datatype_fp32 &&
          output_datatype == xnn_datatype_fp32)
      {
        return xnn_compute_type_qd8_to_fp32;
      } else if (input_datatype == xnn_datatype_qpint8 &&
                 bias_datatype == xnn_datatype_fp32 &&
                 output_datatype == xnn_datatype_fp32) {
        return xnn_compute_type_qp8_to_fp32;
      } else if (input_datatype == xnn_datatype_qdint8 &&
                 bias_datatype == xnn_datatype_fp32 &&
                 output_datatype == xnn_datatype_fp16) {
        return xnn_compute_type_qd8_to_fp16;
      } else if (input_datatype == xnn_datatype_qint8 &&
                 bias_datatype == xnn_datatype_qcint32 &&
                 output_datatype == xnn_datatype_qint8) {
        return xnn_compute_type_qc8;
      }
      break;
    case xnn_datatype_qint8:
      if (input_datatype == xnn_datatype_qint8 &&
          bias_datatype == xnn_datatype_qint32 &&
          output_datatype == xnn_datatype_qint8)
      {
        return xnn_compute_type_qs8;
      }
      break;
    case xnn_datatype_quint8:
      if (input_datatype == xnn_datatype_quint8 &&
          bias_datatype == xnn_datatype_qint32 &&
          output_datatype == xnn_datatype_quint8)
      {
        return xnn_compute_type_qu8;
      }
      break;
    default:
      XNN_UNREACHABLE;
  }
  return xnn_compute_type_invalid;
}

static inline enum xnn_compute_type validate_datatypes_without_bias(
  enum xnn_datatype input_datatype,
  enum xnn_datatype kernel_datatype,
  enum xnn_datatype output_datatype)
{
  switch (kernel_datatype) {
    case xnn_datatype_fp32:
      if (input_datatype == xnn_datatype_fp32 && output_datatype == xnn_datatype_fp32) {
        return xnn_compute_type_fp32;
      } else if (input_datatype == xnn_datatype_fp16 && output_datatype == xnn_datatype_fp16) {
        // Flag: XNN_FLAG_FP32_STATIC_WEIGHTS
        return xnn_compute_type_fp16;
      }
      break;
    case xnn_datatype_qcint4:
      if (input_datatype == xnn_datatype_fp32 && output_datatype == xnn_datatype_fp32) {
        return xnn_compute_type_fp32;
      } else if (input_datatype == xnn_datatype_qdint8 && output_datatype == xnn_datatype_fp32) {
        return xnn_compute_type_qd8_to_fp32;
      } else if (input_datatype == xnn_datatype_qpint8 &&
                 output_datatype == xnn_datatype_fp32) {
        return xnn_compute_type_qp8_to_fp32;
      } else if (input_datatype == xnn_datatype_qdint8 &&
                 output_datatype == xnn_datatype_fp16) {
        return xnn_compute_type_qd8_to_fp16;
      }
      break;
    case xnn_datatype_qbint4:
      if (input_datatype == xnn_datatype_qdint8 && output_datatype == xnn_datatype_fp32) {
        return xnn_compute_type_qd8_to_fp32;
      } else if (input_datatype == xnn_datatype_qdint8 && output_datatype == xnn_datatype_fp16) {
        return xnn_compute_type_qd8_to_fp16;
      }
      break;
    case xnn_datatype_qcint8:
      if (input_datatype == xnn_datatype_fp32 && output_datatype == xnn_datatype_fp32) {
        return xnn_compute_type_fp32;
      } else if (input_datatype == xnn_datatype_qdint8 && output_datatype == xnn_datatype_fp32) {
        return xnn_compute_type_qd8_to_fp32;
      } else if (input_datatype == xnn_datatype_qpint8 &&
                 output_datatype == xnn_datatype_fp32) {
        return xnn_compute_type_qp8_to_fp32;
      } else if (input_datatype == xnn_datatype_qdint8 &&
                 output_datatype == xnn_datatype_fp16) {
        return xnn_compute_type_qd8_to_fp16;
      } else if (input_datatype == xnn_datatype_qint8 &&
                 output_datatype == xnn_datatype_qint8) {
        return xnn_compute_type_qc8;
      }
      break;
    case xnn_datatype_qint8:
      if (input_datatype == xnn_datatype_qint8 && output_datatype == xnn_datatype_qint8) {
        return xnn_compute_type_qs8;
      }
      break;
    case xnn_datatype_quint8:
      if (input_datatype == xnn_datatype_quint8 && output_datatype == xnn_datatype_quint8) {
        return xnn_compute_type_qu8;
      }
      break;
    default:
      XNN_UNREACHABLE;
  }
  return xnn_compute_type_invalid;
}

enum xnn_status xnn_define_fully_connected(
  xnn_subgraph_t subgraph,
  float output_min,
  float output_max,
  uint32_t input_id,
  uint32_t filter_id,
  uint32_t bias_id,
  uint32_t output_id,
  uint32_t flags)
{
  enum xnn_status status;
  if ((status = xnn_subgraph_check_xnnpack_initialized(xnn_node_type_fully_connected)) != xnn_status_success) {
    return status;
  }

  status = xnn_subgraph_check_output_min_max(xnn_node_type_fully_connected, output_min, output_max);
  if (status != xnn_status_success) {
    return status;
  }

  if ((status = xnn_subgraph_check_input_node_id(xnn_node_type_fully_connected, input_id, subgraph->num_values)) !=
      xnn_status_success) {
    return status;
  }

  const struct xnn_value* input_value = &subgraph->values[input_id];
  status = xnn_subgraph_check_input_type_dense(xnn_node_type_fully_connected, input_id, input_value);
  if (status != xnn_status_success) {
    return status;
  }

  switch (input_value->datatype) {
    case xnn_datatype_fp16:
    case xnn_datatype_fp32:
    case xnn_datatype_qint8:
    case xnn_datatype_quint8:
    case xnn_datatype_qpint8:
      break;
    case xnn_datatype_qdint8:
      if (input_value->quantization.num_nonbatch_dims > input_value->shape.num_dims) {
        xnn_log_error(
          "failed to define %s operator with input ID #%" PRIu32 ": num_nonbatch_dims (%zu) must be "
          "<= num_dims (%zu)",
          xnn_node_type_to_string(xnn_node_type_fully_connected), input_id,
          input_value->quantization.num_nonbatch_dims, input_value->shape.num_dims);
        return xnn_status_invalid_parameter;
      }
      break;
    default:
      xnn_log_error(
        "failed to define %s operator with input ID #%" PRIu32 ": unsupported Value datatype %s (%d)",
        xnn_node_type_to_string(xnn_node_type_fully_connected), input_id,
        xnn_datatype_to_string(input_value->datatype), input_value->datatype);
      return xnn_status_invalid_parameter;
  }

  if (filter_id >= subgraph->num_values) {
    xnn_log_error(
      "failed to define %s operator with filter ID #%" PRIu32 ": invalid Value ID",
      xnn_node_type_to_string(xnn_node_type_fully_connected), filter_id);
    return xnn_status_invalid_parameter;
  }

  const struct xnn_value* kernel_value = &subgraph->values[filter_id];
  if (kernel_value->type != xnn_value_type_dense_tensor) {
    xnn_log_error(
      "failed to define %s operator with filter ID #%" PRIu32 ": unsupported Value type %d (expected dense tensor)",
      xnn_node_type_to_string(xnn_node_type_fully_connected), filter_id, kernel_value->type);
    return xnn_status_invalid_parameter;
  }

  // Non-static kernel is supported, but only for some data types
  switch (kernel_value->datatype) {
    case xnn_datatype_fp32:
      break;  // non-static kernel is supported
    default:
      if (kernel_value->data == NULL) {
        xnn_log_error(
          "failed to define %s operator with filter ID #%" PRIu32 ": non-static Value",
          xnn_node_type_to_string(xnn_node_type_fully_connected), filter_id);
        return xnn_status_invalid_parameter;
      }
      break;
  }

  // Non-static kernel is supported, but only for some data types
  switch (kernel_value->datatype) {
    case xnn_datatype_fp32:
      break;
    case xnn_datatype_qbint4:
    case xnn_datatype_qcint4:
      if (kernel_value->quantization.zero_point != 8 && kernel_value->quantization.zero_point != 0) {
        xnn_log_error(
          "failed to define %s operator with filter ID #%" PRIu32 ": unsupported quantization zero point %" PRId32
          " for datatype %s, must be equals to 8 (unsigned weights) or 0 (signed weights) ",
          xnn_node_type_to_string(xnn_node_type_fully_connected), filter_id, kernel_value->quantization.zero_point,
          xnn_datatype_to_string(kernel_value->datatype));
        return xnn_status_invalid_parameter;
      }
      break;
    case xnn_datatype_qcint8:
      break;
    case xnn_datatype_qint8:
      if (kernel_value->quantization.zero_point != 0) {
        xnn_log_error(
          "failed to define %s operator with filter ID #%" PRIu32 ": unsupported quantization zero point %" PRId32 " for datatype %s",
          xnn_node_type_to_string(xnn_node_type_fully_connected), filter_id,
          kernel_value->quantization.zero_point, xnn_datatype_to_string(kernel_value->datatype));
      }
      break;
    case xnn_datatype_quint8:
      break;
    default:
      xnn_log_error(
        "failed to define %s operator with filter ID #%" PRIu32 ": unsupported Value datatype %s (%d)",
        xnn_node_type_to_string(xnn_node_type_fully_connected), filter_id,
        xnn_datatype_to_string(kernel_value->datatype), kernel_value->datatype);
      return xnn_status_invalid_parameter;
  }

  const bool is_channelwise_quantized =
    kernel_value->datatype == xnn_datatype_qcint8 || kernel_value->datatype == xnn_datatype_qcint4;

  if (is_channelwise_quantized) {
    const size_t output_channels_dim = ((flags & XNN_FLAG_TRANSPOSE_WEIGHTS) != 0) ?  1 : 0;
    if (kernel_value->quantization.channel_dimension != output_channels_dim) {
      xnn_log_error(
        "failed to define %s operator with filter ID #%" PRIu32 ": invalid channel dimension %zu",
        xnn_node_type_to_string(xnn_node_type_fully_connected), input_id, kernel_value->quantization.channel_dimension);
      return xnn_status_invalid_parameter;
    }
  }

  const bool is_blockwise_quantized = kernel_value->datatype == xnn_datatype_qbint4;

  if (is_blockwise_quantized) {
    // TODO: Unsupported features
    assert ((flags & XNN_FLAG_TRANSPOSE_WEIGHTS) == 0);

    const size_t input_channels_dim = ((flags & XNN_FLAG_TRANSPOSE_WEIGHTS) != 0) ?  0 : 1;
    const size_t output_channels_dim = ((flags & XNN_FLAG_TRANSPOSE_WEIGHTS) != 0) ?  1 : 0;
    if (kernel_value->quantization.channel_dimension_blockwise != output_channels_dim) {
      xnn_log_error(
        "failed to define %s operator with filter ID #%" PRIu32 ": invalid channel dimension %zu",
        xnn_node_type_to_string(xnn_node_type_fully_connected), input_id, kernel_value->quantization.channel_dimension_blockwise);
      return xnn_status_invalid_parameter;
    }
    const size_t input_channels = kernel_value->shape.dim[input_channels_dim];
    if (input_channels % kernel_value->quantization.block_size) {
      xnn_log_error(
        "failed to define %s operator with filter ID #%" PRIu32 ": invalid block size %zu, input_channels %zu",
        xnn_node_type_to_string(xnn_node_type_fully_connected), input_id, kernel_value->quantization.block_size, input_channels);
      return xnn_status_invalid_parameter;
    }
  }

  const struct xnn_value* bias_value = NULL;
  if (bias_id != XNN_INVALID_VALUE_ID) {
    if (bias_id >= subgraph->num_values) {
      xnn_log_error(
        "failed to define %s operator with bias ID #%" PRIu32 ": invalid Value ID",
        xnn_node_type_to_string(xnn_node_type_fully_connected), bias_id);
      return xnn_status_invalid_parameter;
    }

    bias_value = &subgraph->values[bias_id];
    if (bias_value->type != xnn_value_type_dense_tensor) {
      xnn_log_error(
        "failed to define %s operator with bias ID #%" PRIu32 ": unsupported Value type %d (expected dense tensor)",
        xnn_node_type_to_string(xnn_node_type_fully_connected), bias_id, bias_value->type);
      return xnn_status_invalid_parameter;
    }

    // Non-static bias is supported, but only for some data types
    switch (bias_value->datatype) {
      case xnn_datatype_fp32:
        if (is_channelwise_quantized && bias_value->data == NULL) {
          xnn_log_error(
            "failed to define %s operator with bias ID #%" PRIu32 ": non-static Value",
            xnn_node_type_to_string(xnn_node_type_fully_connected), bias_id);
          return xnn_status_invalid_parameter;
        }
        break;  // non-static bias is supported
      default:
        if (bias_value->data == NULL) {
          xnn_log_error(
            "failed to define %s operator with bias ID #%" PRIu32 ": non-static Value",
            xnn_node_type_to_string(xnn_node_type_fully_connected), bias_id);
          return xnn_status_invalid_parameter;
        }
        break;
    }

    switch (bias_value->datatype) {
      case xnn_datatype_fp16:
      case xnn_datatype_fp32:
      case xnn_datatype_qint32:
      case xnn_datatype_qcint32:
        break;
      default:
        xnn_log_error(
          "failed to define %s operator with bias ID #%" PRIu32 ": unsupported Value datatype %s (%d)",
          xnn_node_type_to_string(xnn_node_type_fully_connected), bias_id,
          xnn_datatype_to_string(bias_value->datatype), bias_value->datatype);
        return xnn_status_invalid_parameter;
    }
  }

  status = xnn_subgraph_check_output_node_id(xnn_node_type_fully_connected, output_id, subgraph->num_values);
  if (status != xnn_status_success) {
    return status;
  }

  const struct xnn_value* output_value = &subgraph->values[output_id];
  status = xnn_subgraph_check_output_type_dense(xnn_node_type_fully_connected, output_id, output_value);
  if (status != xnn_status_success) {
    return status;
  }

  switch (output_value->datatype) {
    case xnn_datatype_fp16:
    case xnn_datatype_fp32:
    case xnn_datatype_qint8:
    case xnn_datatype_quint8:
      break;
    default:
      xnn_log_error(
        "failed to define %s operator with output ID #%" PRIu32 ": unsupported Value datatype %s (%d)",
        xnn_node_type_to_string(xnn_node_type_fully_connected), output_id,
        xnn_datatype_to_string(output_value->datatype), output_value->datatype);
      return xnn_status_invalid_parameter;
  }

  enum xnn_compute_type compute_type = xnn_compute_type_invalid;
  if (bias_value != NULL) {
    compute_type = validate_datatypes_with_bias(
      input_value->datatype, kernel_value->datatype, bias_value->datatype, output_value->datatype);
    if (compute_type == xnn_compute_type_invalid) {
      xnn_log_error(
        "failed to define %s operator with input ID #%" PRIu32 ", filter ID #%" PRIu32 ", bias ID #%" PRIu32 ", and output ID #%" PRIu32
        ": mismatching datatypes across input (%s), filter (%s), bias (%s), and output (%s)",
        xnn_node_type_to_string(xnn_node_type_fully_connected), input_id, filter_id, bias_id, output_id,
        xnn_datatype_to_string(input_value->datatype),
        xnn_datatype_to_string(kernel_value->datatype),
        xnn_datatype_to_string(bias_value->datatype),
        xnn_datatype_to_string(output_value->datatype));
      return xnn_status_invalid_parameter;
    }
  } else {
    compute_type = validate_datatypes_without_bias(
      input_value->datatype, kernel_value->datatype, output_value->datatype);
    if (compute_type == xnn_compute_type_invalid) {
      xnn_log_error(
        "failed to define %s operator with input ID #%" PRIu32 ", filter ID #%" PRIu32 ", and output ID #%" PRIu32
        ": mismatching datatypes across input (%s), filter (%s), and output (%s)",
        xnn_node_type_to_string(xnn_node_type_fully_connected), input_id, filter_id, output_id,
        xnn_datatype_to_string(input_value->datatype),
        xnn_datatype_to_string(kernel_value->datatype),
        xnn_datatype_to_string(output_value->datatype));
      return xnn_status_invalid_parameter;
    }
  }

  struct xnn_node* node = xnn_subgraph_new_node(subgraph);
  if (node == NULL) {
    return xnn_status_out_of_memory;
  }

  node->type = xnn_node_type_fully_connected;
  node->compute_type = compute_type;
  node->activation.output_min = output_min;
  node->activation.output_max = output_max;
  node->num_inputs = 2 + (size_t) (bias_id != XNN_INVALID_VALUE_ID);
  node->inputs[0] = input_id;
  node->inputs[1] = filter_id;
  node->inputs[2] = bias_id;
  node->num_outputs = 1;
  node->outputs[0] = output_id;
  node->flags = flags;

  node->create = create_fully_connected_operator;
  node->reshape = reshape_fully_connected_operator;
  node->setup = setup_fully_connected_operator;

  return xnn_status_success;
}
