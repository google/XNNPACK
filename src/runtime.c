// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>

#include <xnnpack.h>
#include <xnnpack/allocator.h>
#include <xnnpack/log.h>
#include <xnnpack/math.h>
#include <xnnpack/operator.h>
#include <xnnpack/params.h>
#include <xnnpack/subgraph.h>


enum xnn_status xnn_create_runtime(
  xnn_subgraph_t subgraph,
  xnn_runtime_t* runtime_out)
{
  return xnn_create_runtime_v2(subgraph, NULL /* threadpool */, 0 /* flags */, runtime_out);
}

enum xnn_status xnn_create_runtime_v2(
  xnn_subgraph_t subgraph,
  pthreadpool_t threadpool,
  uint32_t flags,
  xnn_runtime_t* runtime_out)
{
  struct xnn_runtime* runtime = NULL;
  enum xnn_status status = xnn_status_uninitialized;

  if (!xnn_params.initialized) {
    xnn_log_error("failed to create runtime: XNNPACK is not initialized");
    goto error;
  }

  status = xnn_status_out_of_memory;

  runtime = xnn_allocate_zero_memory(sizeof(struct xnn_runtime));
  if (runtime == NULL) {
    xnn_log_error("failed to allocate %zu bytes for runtime descriptor", sizeof(struct xnn_runtime));
    goto error;
  }

  runtime->ops = xnn_allocate_zero_memory(sizeof(struct xnn_operator_data) * subgraph->num_nodes);
  if (runtime->ops == NULL) {
    xnn_log_error("failed to allocate %zu bytes for opdata descriptors",
      sizeof(struct xnn_operator_data) * subgraph->num_nodes);
    goto error;
  }
  runtime->num_ops = subgraph->num_nodes;

  struct xnn_value* values = subgraph->values;
  for (size_t i = 0; i < subgraph->num_nodes; i++) {
    const struct xnn_node* node = subgraph->nodes + i;
    switch (node->type) {
      case xnn_node_type_add2:
        status = xnn_create_add_nd_f32(
          node->activation.output_min,
          node->activation.output_max,
          node->flags,
          &runtime->ops[i].op);
        if (status != xnn_status_success) {
          goto error;
        }
        runtime->ops[i].shape1.num_dims = values[node->inputs.raw[0]].shape.num_dims;
        runtime->ops[i].shape2.num_dims = values[node->inputs.raw[1]].shape.num_dims;
        memcpy(runtime->ops[i].shape1.dim, values[node->inputs.raw[0]].shape.dim, values[node->inputs.raw[0]].shape.num_dims * sizeof(size_t));
        memcpy(runtime->ops[i].shape2.dim, values[node->inputs.raw[1]].shape.dim, values[node->inputs.raw[1]].shape.num_dims * sizeof(size_t));
        runtime->ops[i].inputs[0] = node->inputs.raw[0];
        runtime->ops[i].inputs[1] = node->inputs.raw[1];
        runtime->ops[i].outputs[0] = node->outputs.raw[0];
        break;
      case xnn_node_type_convolution_2d:
        status = xnn_create_convolution2d_nhwc_f32(
          node->params.convolution_2d.input_padding_top,
          node->params.convolution_2d.input_padding_right,
          node->params.convolution_2d.input_padding_bottom,
          node->params.convolution_2d.input_padding_left,
          node->params.convolution_2d.kernel_height,
          node->params.convolution_2d.kernel_width,
          node->params.convolution_2d.subsampling_height,
          node->params.convolution_2d.subsampling_width,
          node->params.convolution_2d.dilation_height,
          node->params.convolution_2d.dilation_width,
          node->params.convolution_2d.groups,
          node->params.convolution_2d.group_input_channels,
          node->params.convolution_2d.group_output_channels,
          node->params.convolution_2d.group_input_channels * node->params.convolution_2d.groups /* input_pixel_stride */,
          node->params.convolution_2d.group_output_channels * node->params.convolution_2d.groups /* output_pixel_stride */,
          values[node->inputs.convolution_2d.filter].data,
          values[node->inputs.convolution_2d.bias].data,
          node->activation.output_min,
          node->activation.output_max,
          node->flags,
          &runtime->ops[i].op);
        if (status != xnn_status_success) {
          goto error;
        }
        runtime->ops[i].batch_size = values[node->inputs.raw[0]].shape.dim[0];
        runtime->ops[i].input_height = values[node->inputs.raw[0]].shape.dim[1];
        runtime->ops[i].input_width = values[node->inputs.raw[0]].shape.dim[2];
        runtime->ops[i].inputs[0] = node->inputs.raw[0];
        runtime->ops[i].outputs[0] = node->outputs.raw[0];
        break;
      case xnn_node_type_clamp:
        status = xnn_create_clamp_nc_f32(
          values[node->inputs.raw[0]].shape.dim[values[node->inputs.raw[0]].shape.num_dims - 1] /* channels */,
          values[node->inputs.raw[0]].shape.dim[values[node->inputs.raw[0]].shape.num_dims - 1] /* input stride */,
          values[node->inputs.raw[0]].shape.dim[values[node->inputs.raw[0]].shape.num_dims - 1] /* output stride */,
          node->activation.output_min,
          node->activation.output_max,
          node->flags,
          &runtime->ops[i].op);
        if (status != xnn_status_success) {
          goto error;
        }
        runtime->ops[i].batch_size = 1;
        for (size_t i = 0; i + 1 < values[node->inputs.raw[0]].shape.num_dims; i++) {
          runtime->ops[i].batch_size *= values[node->inputs.raw[0]].shape.dim[i];
        }
        runtime->ops[i].inputs[0] = node->inputs.raw[0];
        runtime->ops[i].outputs[0] = node->outputs.raw[0];
        break;
      case xnn_node_type_depthwise_convolution_2d:
        status = xnn_create_convolution2d_nhwc_f32(
          node->params.depthwise_convolution_2d.input_padding_top,
          node->params.depthwise_convolution_2d.input_padding_right,
          node->params.depthwise_convolution_2d.input_padding_bottom,
          node->params.depthwise_convolution_2d.input_padding_left,
          node->params.depthwise_convolution_2d.kernel_height,
          node->params.depthwise_convolution_2d.kernel_width,
          node->params.depthwise_convolution_2d.subsampling_height,
          node->params.depthwise_convolution_2d.subsampling_width,
          node->params.depthwise_convolution_2d.dilation_height,
          node->params.depthwise_convolution_2d.dilation_width,
          node->params.depthwise_convolution_2d.input_channels /* groups */,
          1 /* group_input_channels */,
          node->params.depthwise_convolution_2d.depth_multiplier /* group_output_channels */,
          node->params.depthwise_convolution_2d.input_channels /* input_pixel_stride */,
          node->params.depthwise_convolution_2d.input_channels * node->params.depthwise_convolution_2d.depth_multiplier /* output_pixel_stride */,
          values[node->inputs.convolution_2d.filter].data,
          values[node->inputs.convolution_2d.bias].data,
          node->activation.output_min,
          node->activation.output_max,
          node->flags | XNN_FLAG_DEPTHWISE_CONVOLUTION,
          &runtime->ops[i].op);
        if (status != xnn_status_success) {
          goto error;
        }
        runtime->ops[i].batch_size = values[node->inputs.raw[0]].shape.dim[0];
        runtime->ops[i].input_height = values[node->inputs.raw[0]].shape.dim[1];
        runtime->ops[i].input_width = values[node->inputs.raw[0]].shape.dim[2];
        runtime->ops[i].inputs[0] = node->inputs.raw[0];
        runtime->ops[i].outputs[0] = node->outputs.raw[0];
        break;
      case xnn_node_type_hardswish:
        status = xnn_create_hardswish_nc_f32(
          values[node->inputs.raw[0]].shape.dim[values[node->inputs.raw[0]].shape.num_dims - 1] /* channels */,
          values[node->inputs.raw[0]].shape.dim[values[node->inputs.raw[0]].shape.num_dims - 1] /* input stride */,
          values[node->inputs.raw[0]].shape.dim[values[node->inputs.raw[0]].shape.num_dims - 1] /* output stride */,
          node->flags,
          &runtime->ops[i].op);
        if (status != xnn_status_success) {
          goto error;
        }
        runtime->ops[i].batch_size = 1;
        for (size_t i = 0; i + 1 < values[node->inputs.raw[0]].shape.num_dims; i++) {
          runtime->ops[i].batch_size *= values[node->inputs.raw[0]].shape.dim[i];
        }
        runtime->ops[i].inputs[0] = node->inputs.raw[0];
        runtime->ops[i].outputs[0] = node->outputs.raw[0];
        break;
      case xnn_node_type_multiply2:
        status = xnn_create_multiply_nd_f32(
          node->activation.output_min,
          node->activation.output_max,
          node->flags,
          &runtime->ops[i].op);
        if (status != xnn_status_success) {
          goto error;
        }
        runtime->ops[i].shape1.num_dims = values[node->inputs.raw[0]].shape.num_dims;
        runtime->ops[i].shape2.num_dims = values[node->inputs.raw[1]].shape.num_dims;
        memcpy(runtime->ops[i].shape1.dim, values[node->inputs.raw[0]].shape.dim, values[node->inputs.raw[0]].shape.num_dims * sizeof(size_t));
        memcpy(runtime->ops[i].shape2.dim, values[node->inputs.raw[1]].shape.dim, values[node->inputs.raw[1]].shape.num_dims * sizeof(size_t));
        runtime->ops[i].inputs[0] = node->inputs.raw[0];
        runtime->ops[i].inputs[1] = node->inputs.raw[1];
        runtime->ops[i].outputs[0] = node->outputs.raw[0];
        break;
      case xnn_node_type_prelu:
        status = xnn_create_prelu_nc_f32(
          values[node->inputs.raw[1]].shape.dim[values[node->inputs.raw[1]].shape.num_dims - 1] /* channels */,
          values[node->inputs.raw[1]].shape.dim[values[node->inputs.raw[1]].shape.num_dims - 1] /* input stride */,
          values[node->inputs.raw[1]].shape.dim[values[node->inputs.raw[1]].shape.num_dims - 1] /* output stride */,
          values[node->inputs.raw[1]].data /* negative slope */,
          -INFINITY,
          +INFINITY,
          node->flags,
          &runtime->ops[i].op);
        if (status != xnn_status_success) {
          goto error;
        }
        runtime->ops[i].batch_size = 1;
        for (size_t i = 0; i + 1 < values[node->inputs.raw[0]].shape.num_dims; i++) {
          runtime->ops[i].batch_size *= values[node->inputs.raw[0]].shape.dim[i];
        }
        runtime->ops[i].inputs[0] = node->inputs.raw[0];
        runtime->ops[i].outputs[0] = node->outputs.raw[0];
        break;
      case xnn_node_type_sigmoid:
        status = xnn_create_sigmoid_nc_f32(
          values[node->inputs.raw[0]].shape.dim[values[node->inputs.raw[0]].shape.num_dims - 1] /* channels */,
          values[node->inputs.raw[0]].shape.dim[values[node->inputs.raw[0]].shape.num_dims - 1] /* input stride */,
          values[node->inputs.raw[0]].shape.dim[values[node->inputs.raw[0]].shape.num_dims - 1] /* output stride */,
          node->flags,
          &runtime->ops[i].op);
        if (status != xnn_status_success) {
          goto error;
        }
        runtime->ops[i].batch_size = 1;
        for (size_t i = 0; i + 1 < values[node->inputs.raw[0]].shape.num_dims; i++) {
          runtime->ops[i].batch_size *= values[node->inputs.raw[0]].shape.dim[i];
        }
        runtime->ops[i].inputs[0] = node->inputs.raw[0];
        runtime->ops[i].outputs[0] = node->outputs.raw[0];
        break;
      case xnn_node_type_softmax:
        status = xnn_create_softmax_nc_f32(
          values[node->inputs.raw[0]].shape.dim[values[node->inputs.raw[0]].shape.num_dims - 1] /* channels */,
          values[node->inputs.raw[0]].shape.dim[values[node->inputs.raw[0]].shape.num_dims - 1] /* input stride */,
          values[node->inputs.raw[0]].shape.dim[values[node->inputs.raw[0]].shape.num_dims - 1] /* output stride */,
          node->flags,
          &runtime->ops[i].op);
        if (status != xnn_status_success) {
          goto error;
        }
        runtime->ops[i].batch_size = 1;
        for (size_t i = 0; i + 1 < values[node->inputs.raw[0]].shape.num_dims; i++) {
          runtime->ops[i].batch_size *= values[node->inputs.raw[0]].shape.dim[i];
        }
        runtime->ops[i].inputs[0] = node->inputs.raw[0];
        runtime->ops[i].outputs[0] = node->outputs.raw[0];
        break;
      case xnn_node_type_invalid:
        xnn_log_fatal("unexpected node type %d in node #%zu", node->type, i);
        XNN_UNREACHABLE;
        break;
    }
  }

  runtime->blobs = xnn_allocate_zero_memory(sizeof(struct xnn_blob) * subgraph->num_values);
  if (runtime->blobs == NULL) {
    xnn_log_error("failed to allocate %zu bytes for blob descriptors",
      sizeof(struct xnn_blob) * subgraph->num_values);
    goto error;
  }
  runtime->num_blobs = subgraph->num_values;

  size_t buffer_size = 0;
  for (size_t i = 0; i < subgraph->num_values; i++) {
    const struct xnn_value* value = &subgraph->values[i];
    struct xnn_blob* blob = &runtime->blobs[i];
    if (value->datatype != xnn_datatype_invalid && value->type == xnn_value_type_dense_tensor) {
      blob->size = xnn_tensor_get_size(subgraph, i);
      blob->data = (void*) value->data;
      if (blob->data == NULL) {
        if ((value->flags & (XNN_VALUE_FLAG_EXTERNAL_INPUT | XNN_VALUE_FLAG_EXTERNAL_OUTPUT)) == 0) {
          // Value is purely internal to the runtime, and must be allocated in its workspace.
          buffer_size = round_up_po2(buffer_size + blob->size, XNN_EXTRA_BYTES);
        } else {
          // Value is non-static and external to the runtime: must be specified via a call to xnn_setup_runtime.
          blob->external = true;
        }
      }
    }
  }

  runtime->workspace = xnn_allocate_simd_memory(buffer_size);
  if (runtime->workspace == NULL) {
    xnn_log_error("failed to allocate %zu bytes to runtime workspace", buffer_size);
    goto error;
  }

  size_t buffer_offset = 0;
  for (size_t i = 0; i < subgraph->num_values; i++) {
    const struct xnn_value* value = &subgraph->values[i];
    struct xnn_blob* blob = &runtime->blobs[i];
    if (value->datatype != xnn_datatype_invalid && value->type == xnn_value_type_dense_tensor) {
      if (value->data == NULL && !blob->external) {
        // Value is purely internal to the runtime, allocate it in the workspace.
        blob->data = (void*) ((uintptr_t) runtime->workspace + buffer_offset);
        buffer_offset = round_up_po2(buffer_offset + blob->size, XNN_EXTRA_BYTES);
      }
    }
  }

  runtime->threadpool = threadpool;

  *runtime_out = runtime;
  return xnn_status_success;

error:
  xnn_delete_runtime(runtime);
  return status;
}

enum xnn_status xnn_setup_runtime(
  xnn_runtime_t runtime,
  size_t num_external_values,
  const struct xnn_external_value* external_values)
{
  // Validate inputs without changing internal state.
  // This ensures that runtime stays in consistent state in case validation fails midway.
  for (size_t i = 0; i < num_external_values; i++) {
    const struct xnn_external_value* external_value = &external_values[i];
    const uint32_t value_id = external_value->id;
    if (value_id >= runtime->num_blobs) {
      xnn_log_error("failed to setup runtime: out-of-bounds ID %" PRIu32 " in external value #%zu",
        value_id, i);
      return xnn_status_invalid_parameter;
    }

    const struct xnn_blob* blob = &runtime->blobs[value_id];
    if (!blob->external) {
      xnn_log_error("failed to setup runtime: Value %" PRIu32 " is not external", value_id);
      return xnn_status_invalid_parameter;
    }
  }

  // Apply runtime state changes.
  for (size_t i = 0; i < num_external_values; i++) {
    const struct xnn_external_value* external_value = &external_values[i];
    const uint32_t value_id = external_value->id;
    struct xnn_blob* blob = &runtime->blobs[value_id];
    blob->data = external_value->data;
  }

  for (size_t i = 0; i < runtime->num_ops; i++) {
    const struct xnn_operator_data* op = &runtime->ops[i];
    enum xnn_status status = xnn_status_success;
    switch (op->op->type) {
      case xnn_operator_type_add_nd_f32:
        assert(runtime->blobs[op->inputs[0]].data != NULL);
        assert(runtime->blobs[op->inputs[1]].data != NULL);
        assert(runtime->blobs[op->outputs[0]].data != NULL);
        status = xnn_setup_add_nd_f32(
          op->op,
          op->shape1.num_dims,
          op->shape1.dim,
          op->shape2.num_dims,
          op->shape2.dim,
          runtime->blobs[op->inputs[0]].data,
          runtime->blobs[op->inputs[1]].data,
          runtime->blobs[op->outputs[0]].data,
          runtime->threadpool);
        break;
      case xnn_operator_type_convolution_nhwc_f32:
        assert(runtime->blobs[op->inputs[0]].data != NULL);
        assert(runtime->blobs[op->outputs[0]].data != NULL);
        status = xnn_setup_convolution2d_nhwc_f32(
          op->op,
          op->batch_size,
          op->input_height,
          op->input_width,
          runtime->blobs[op->inputs[0]].data,
          runtime->blobs[op->outputs[0]].data,
          runtime->threadpool);
        break;
      case xnn_operator_type_clamp_nc_f32:
        assert(runtime->blobs[op->inputs[0]].data != NULL);
        assert(runtime->blobs[op->outputs[0]].data != NULL);
        status = xnn_setup_clamp_nc_f32(
          op->op,
          op->batch_size,
          runtime->blobs[op->inputs[0]].data,
          runtime->blobs[op->outputs[0]].data,
          runtime->threadpool);
        break;
      case xnn_operator_type_hardswish_nc_f32:
        assert(runtime->blobs[op->inputs[0]].data != NULL);
        assert(runtime->blobs[op->outputs[0]].data != NULL);
        status = xnn_setup_hardswish_nc_f32(
          op->op,
          op->batch_size,
          runtime->blobs[op->inputs[0]].data,
          runtime->blobs[op->outputs[0]].data,
          runtime->threadpool);
        break;
      case xnn_operator_type_multiply_nd_f32:
        assert(runtime->blobs[op->inputs[0]].data != NULL);
        assert(runtime->blobs[op->inputs[1]].data != NULL);
        assert(runtime->blobs[op->outputs[0]].data != NULL);
        status = xnn_setup_multiply_nd_f32(
          op->op,
          op->shape1.num_dims,
          op->shape1.dim,
          op->shape2.num_dims,
          op->shape2.dim,
          runtime->blobs[op->inputs[0]].data,
          runtime->blobs[op->inputs[1]].data,
          runtime->blobs[op->outputs[0]].data,
          runtime->threadpool);
        break;
      case xnn_operator_type_prelu_nc_f32:
        assert(runtime->blobs[op->inputs[0]].data != NULL);
        assert(runtime->blobs[op->outputs[0]].data != NULL);
        status = xnn_setup_prelu_nc_f32(
          op->op,
          op->batch_size,
          runtime->blobs[op->inputs[0]].data,
          runtime->blobs[op->outputs[0]].data,
          runtime->threadpool);
        break;
      case xnn_operator_type_sigmoid_nc_f32:
        assert(runtime->blobs[op->inputs[0]].data != NULL);
        assert(runtime->blobs[op->outputs[0]].data != NULL);
        status = xnn_setup_sigmoid_nc_f32(
          op->op,
          op->batch_size,
          runtime->blobs[op->inputs[0]].data,
          runtime->blobs[op->outputs[0]].data,
          runtime->threadpool);
        break;
      case xnn_operator_type_softmax_nc_f32:
        assert(runtime->blobs[op->inputs[0]].data != NULL);
        assert(runtime->blobs[op->outputs[0]].data != NULL);
        status = xnn_setup_softmax_nc_f32(
          op->op,
          op->batch_size,
          runtime->blobs[op->inputs[0]].data,
          runtime->blobs[op->outputs[0]].data,
          runtime->threadpool);
        break;
      default:
        xnn_log_fatal("unexpected operator type %d in operator #%zu", op->op->type, i);
        XNN_UNREACHABLE;
    }
    if (status != xnn_status_success) {
      xnn_log_error("failed to setup runtime: error in operator #%zu", i);
      return status;
    }
  }

  return xnn_status_success;
}

enum xnn_status xnn_invoke_runtime(
  xnn_runtime_t runtime)
{
  for (size_t i = 0; i < runtime->num_ops; i++) {
    const enum xnn_status status = xnn_run_operator(runtime->ops[i].op, runtime->threadpool);
    if (status != xnn_status_success) {
      return status;
    }
  }
  return xnn_status_success;
}

enum xnn_status xnn_delete_runtime(
  xnn_runtime_t runtime)
{
  if (runtime != NULL) {
    if (runtime->ops != NULL) {
      for (size_t i = 0; i < runtime->num_ops; i++) {
        xnn_delete_operator(runtime->ops[i].op);
      }
      xnn_release_memory(runtime->ops);

      xnn_release_memory(runtime->blobs);
      xnn_release_memory(runtime->workspace);
    }
    xnn_release_memory(runtime);
  }
  return xnn_status_success;
}
