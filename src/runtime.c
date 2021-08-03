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
#include <xnnpack/memory-planner.h>
#include <xnnpack/operator.h>
#include <xnnpack/params.h>
#include <xnnpack/subgraph.h>


enum xnn_status xnn_create_runtime(
  xnn_subgraph_t subgraph,
  xnn_runtime_t* runtime_out)
{
  return xnn_create_runtime_v2(subgraph, NULL /* threadpool */, 0 /* flags */, runtime_out);
}

// Product of all shape dimensions
static size_t product_all_dims(
  const struct xnn_shape shape[restrict XNN_MIN_ELEMENTS(1)])
{
  size_t batch_size = 1;
  for (size_t i = 0; i < shape->num_dims; i++) {
    batch_size *= shape->dim[i];
  }
  return batch_size;
}

// Product of all shape dimensions, except for the last (channel) one
static size_t product_non_channel_dims(
  const struct xnn_shape shape[restrict XNN_MIN_ELEMENTS(1)])
{
  size_t batch_size = 1;
  for (size_t i = 0; i + 1 < shape->num_dims; i++) {
    batch_size *= shape->dim[i];
  }
  return batch_size;
}

enum xnn_status xnn_create_runtime_v2(
  xnn_subgraph_t subgraph,
  pthreadpool_t threadpool,
  uint32_t flags,
  xnn_runtime_t* runtime_out)
{
  struct xnn_runtime* runtime = NULL;
  enum xnn_status status = xnn_status_uninitialized;

  if ((xnn_params.init_flags & XNN_INIT_FLAG_XNNPACK) == 0) {
    xnn_log_error("failed to create runtime: XNNPACK is not initialized");
    goto error;
  }

  xnn_subgraph_optimize(subgraph, flags & XNN_FLAG_SPARSE_INFERENCE);

  status = xnn_status_out_of_memory;

  runtime = xnn_allocate_zero_memory(sizeof(struct xnn_runtime));
  if (runtime == NULL) {
    xnn_log_error("failed to allocate %zu bytes for runtime descriptor", sizeof(struct xnn_runtime));
    goto error;
  }

  runtime->opdata = xnn_allocate_zero_memory(sizeof(struct xnn_operator_data) * subgraph->num_nodes);
  if (runtime->opdata == NULL) {
    xnn_log_error("failed to allocate %zu bytes for opdata descriptors",
      sizeof(struct xnn_operator_data) * subgraph->num_nodes);
    goto error;
  }
  runtime->num_ops = subgraph->num_nodes;

  if (flags & XNN_FLAG_YIELD_WORKERS) {
    struct xnn_node* last_valid_node = NULL;
    for (size_t i = 0; i < subgraph->num_nodes; i++) {
      struct xnn_node* node = subgraph->nodes + i;
      if (node->type != xnn_node_type_invalid) {
        last_valid_node = node;
      }
    }
    if (last_valid_node != NULL) {
      last_valid_node->flags |= XNN_FLAG_YIELD_WORKERS;
    }
  }

  struct xnn_value* values = subgraph->values;
  for (size_t i = 0; i < subgraph->num_nodes; i++) {
    const struct xnn_node* node = subgraph->nodes + i;
    switch (node->type) {
      case xnn_node_type_invalid:
        // Node was fused
        continue;
      case xnn_node_type_abs:
        status = xnn_create_abs_nc_f32(
          values[node->inputs[0]].shape.dim[values[node->inputs[0]].shape.num_dims - 1] /* channels */,
          values[node->inputs[0]].shape.dim[values[node->inputs[0]].shape.num_dims - 1] /* input stride */,
          values[node->inputs[0]].shape.dim[values[node->inputs[0]].shape.num_dims - 1] /* output stride */,
          node->flags,
          &runtime->opdata[i].operator_object);
        if (status != xnn_status_success) {
          goto error;
        }
        runtime->opdata[i].batch_size = product_non_channel_dims(&values[node->inputs[0]].shape);
        runtime->opdata[i].inputs[0] = node->inputs[0];
        runtime->opdata[i].outputs[0] = node->outputs[0];
        break;
      case xnn_node_type_add2:
        switch (values[node->outputs[0]].datatype) {
          case xnn_datatype_fp32:
            status = xnn_create_add_nd_f32(
              node->activation.output_min,
              node->activation.output_max,
              node->flags,
              &runtime->opdata[i].operator_object);
            break;
#ifndef XNN_NO_QS8_OPERATORS
          case xnn_datatype_qint8:
          {
            const float output_scale = values[node->outputs[0]].quantization.scale;
            const int32_t output_zero_point = values[node->outputs[0]].quantization.zero_point;
            const int8_t output_min =
              (int8_t) lrintf(fminf(fmaxf(node->activation.output_min / output_scale + (float) output_zero_point, -128.0f), 127.0f));
            const int8_t output_max =
              (int8_t) lrintf(fminf(fmaxf(node->activation.output_max / output_scale + (float) output_zero_point, -128.0f), 127.0f));
            status = xnn_create_add_nd_qs8(
              (int8_t) values[node->inputs[0]].quantization.zero_point,
              values[node->inputs[0]].quantization.scale,
              (int8_t) values[node->inputs[1]].quantization.zero_point,
              values[node->inputs[1]].quantization.scale,
              (int8_t) output_zero_point,
              output_scale, output_min, output_max, node->flags,
              &runtime->opdata[i].operator_object);
            break;
          }
#endif  // !defined(XNN_NO_QS8_OPERATORS)
#ifndef XNN_NO_QU8_OPERATORS
          case xnn_datatype_quint8:
          {
            const float output_scale = values[node->outputs[0]].quantization.scale;
            const int32_t output_zero_point = values[node->outputs[0]].quantization.zero_point;
            const uint8_t output_min =
              (uint8_t) lrintf(fminf(fmaxf(node->activation.output_min / output_scale + (float) output_zero_point, 0.0f), 255.0f));
            const uint8_t output_max =
              (uint8_t) lrintf(fminf(fmaxf(node->activation.output_max / output_scale + (float) output_zero_point, 0.0f), 255.0f));
            status = xnn_create_add_nd_qu8(
              (uint8_t) values[node->inputs[0]].quantization.zero_point,
              values[node->inputs[0]].quantization.scale,
              (uint8_t) values[node->inputs[1]].quantization.zero_point,
              values[node->inputs[1]].quantization.scale,
              (uint8_t) output_zero_point,
              output_scale, output_min, output_max, node->flags,
              &runtime->opdata[i].operator_object);
            break;
          }
#endif  // !defined(XNN_NO_QU8_OPERATORS)
          default:
            XNN_UNREACHABLE;
        }
        if (status != xnn_status_success) {
          goto error;
        }
        runtime->opdata[i].shape1.num_dims = values[node->inputs[0]].shape.num_dims;
        runtime->opdata[i].shape2.num_dims = values[node->inputs[1]].shape.num_dims;
        if (values[node->outputs[0]].layout == xnn_layout_type_nchw) {
          assert(values[node->inputs[0]].layout == xnn_layout_type_nchw);
          assert(values[node->inputs[1]].layout == xnn_layout_type_nchw);
          runtime->opdata[i].shape1.dim[0] = values[node->inputs[0]].shape.dim[0];
          runtime->opdata[i].shape1.dim[1] = values[node->inputs[0]].shape.dim[values[node->inputs[0]].shape.num_dims - 1];
          if (values[node->inputs[0]].shape.num_dims > 2) {
            memcpy(&runtime->opdata[i].shape1.dim[2], &values[node->inputs[0]].shape.dim[1], (values[node->inputs[0]].shape.num_dims - 2) * sizeof(size_t));
          }
          runtime->opdata[i].shape2.dim[0] = values[node->inputs[1]].shape.dim[0];
          runtime->opdata[i].shape2.dim[1] = values[node->inputs[1]].shape.dim[values[node->inputs[0]].shape.num_dims - 1];
          if (values[node->inputs[0]].shape.num_dims > 2) {
            memcpy(&runtime->opdata[i].shape2.dim[2], &values[node->inputs[1]].shape.dim[1], (values[node->inputs[1]].shape.num_dims - 2) * sizeof(size_t));
          }
        } else {
          assert(values[node->outputs[0]].layout == xnn_layout_type_nhwc);
          assert(values[node->inputs[0]].layout == xnn_layout_type_nhwc);
          assert(values[node->inputs[1]].layout == xnn_layout_type_nhwc);
          memcpy(runtime->opdata[i].shape1.dim, values[node->inputs[0]].shape.dim, values[node->inputs[0]].shape.num_dims * sizeof(size_t));
          memcpy(runtime->opdata[i].shape2.dim, values[node->inputs[1]].shape.dim, values[node->inputs[1]].shape.num_dims * sizeof(size_t));
        }
        runtime->opdata[i].inputs[0] = node->inputs[0];
        runtime->opdata[i].inputs[1] = node->inputs[1];
        runtime->opdata[i].outputs[0] = node->outputs[0];
        break;
      case xnn_node_type_argmax_pooling_2d:
        status = xnn_create_argmax_pooling2d_nhwc_f32(
          node->params.pooling_2d.padding_top,
          node->params.pooling_2d.padding_right,
          node->params.pooling_2d.padding_bottom,
          node->params.pooling_2d.padding_left,
          node->params.pooling_2d.pooling_height,
          node->params.pooling_2d.pooling_width,
          values[node->inputs[0]].shape.dim[values[node->inputs[0]].shape.num_dims - 1] /* channels */,
          values[node->inputs[0]].shape.dim[values[node->inputs[0]].shape.num_dims - 1] /* input stride */,
          values[node->inputs[0]].shape.dim[values[node->inputs[0]].shape.num_dims - 1] /* output stride */,
          node->flags,
          &runtime->opdata[i].operator_object);
        if (status != xnn_status_success) {
          goto error;
        }
        runtime->opdata[i].batch_size = values[node->inputs[0]].shape.dim[0];
        runtime->opdata[i].input_height = values[node->inputs[0]].shape.dim[1];
        runtime->opdata[i].input_width = values[node->inputs[0]].shape.dim[2];
        runtime->opdata[i].inputs[0] = node->inputs[0];
        runtime->opdata[i].outputs[0] = node->outputs[0];
        runtime->opdata[i].outputs[1] = node->outputs[1];
        break;
      case xnn_node_type_average_pooling_2d:
        status = xnn_create_average_pooling2d_nhwc_f32(
          node->params.pooling_2d.padding_top,
          node->params.pooling_2d.padding_right,
          node->params.pooling_2d.padding_bottom,
          node->params.pooling_2d.padding_left,
          node->params.pooling_2d.pooling_height,
          node->params.pooling_2d.pooling_width,
          node->params.pooling_2d.stride_height,
          node->params.pooling_2d.stride_width,
          values[node->inputs[0]].shape.dim[values[node->inputs[0]].shape.num_dims - 1] /* channels */,
          values[node->inputs[0]].shape.dim[values[node->inputs[0]].shape.num_dims - 1] /* input stride */,
          values[node->inputs[0]].shape.dim[values[node->inputs[0]].shape.num_dims - 1] /* output stride */,
          node->activation.output_min,
          node->activation.output_max,
          node->flags,
          &runtime->opdata[i].operator_object);
        if (status != xnn_status_success) {
          goto error;
        }
        runtime->opdata[i].batch_size = values[node->inputs[0]].shape.dim[0];
        runtime->opdata[i].input_height = values[node->inputs[0]].shape.dim[1];
        runtime->opdata[i].input_width = values[node->inputs[0]].shape.dim[2];
        runtime->opdata[i].inputs[0] = node->inputs[0];
        runtime->opdata[i].outputs[0] = node->outputs[0];
        break;
      case xnn_node_type_bankers_rounding:
        status = xnn_create_bankers_rounding_nc_f32(
          values[node->inputs[0]].shape.dim[values[node->inputs[0]].shape.num_dims - 1] /* channels */,
          values[node->inputs[0]].shape.dim[values[node->inputs[0]].shape.num_dims - 1] /* input stride */,
          values[node->inputs[0]].shape.dim[values[node->inputs[0]].shape.num_dims - 1] /* output stride */,
          node->flags,
          &runtime->opdata[i].operator_object);
        if (status != xnn_status_success) {
          goto error;
        }
        runtime->opdata[i].batch_size = product_non_channel_dims(&values[node->inputs[0]].shape);
        runtime->opdata[i].inputs[0] = node->inputs[0];
        runtime->opdata[i].outputs[0] = node->outputs[0];
        break;
      case xnn_node_type_ceiling:
        status = xnn_create_ceiling_nc_f32(
          values[node->inputs[0]].shape.dim[values[node->inputs[0]].shape.num_dims - 1] /* channels */,
          values[node->inputs[0]].shape.dim[values[node->inputs[0]].shape.num_dims - 1] /* input stride */,
          values[node->inputs[0]].shape.dim[values[node->inputs[0]].shape.num_dims - 1] /* output stride */,
          node->flags,
          &runtime->opdata[i].operator_object);
        if (status != xnn_status_success) {
          goto error;
        }
        runtime->opdata[i].batch_size = product_non_channel_dims(&values[node->inputs[0]].shape);
        runtime->opdata[i].inputs[0] = node->inputs[0];
        runtime->opdata[i].outputs[0] = node->outputs[0];
        break;
      case xnn_node_type_convolution_2d:
      {
        assert(values[node->inputs[1]].data != NULL);
        const void* bias_data = NULL;
        if (node->num_inputs > 2) {
          bias_data = values[node->inputs[2]].data;
          assert(bias_data != NULL);
        }
        if (values[node->outputs[0]].layout == xnn_layout_type_nchw) {
          status = xnn_create_convolution2d_nchw_f32(
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
            values[node->inputs[1]].data,
            bias_data,
            node->activation.output_min,
            node->activation.output_max,
            node->flags | (values[node->inputs[0]].layout == xnn_layout_type_nhwc ? XNN_FLAG_INPUT_NHWC : 0),
            &runtime->opdata[i].operator_object);
        } else {
          assert(values[node->inputs[0]].layout == xnn_layout_type_nhwc);
          assert(values[node->outputs[0]].layout == xnn_layout_type_nhwc);
          switch (values[node->inputs[1]].datatype) {
            case xnn_datatype_fp32:
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
                values[node->inputs[1]].data,
                bias_data,
                node->activation.output_min,
                node->activation.output_max,
                node->flags,
                &runtime->opdata[i].operator_object);
              break;
#ifndef XNN_NO_QS8_OPERATORS
            case xnn_datatype_qint8:
            {
              const float output_scale = values[node->outputs[0]].quantization.scale;
              const int32_t output_zero_point = values[node->outputs[0]].quantization.zero_point;
              const int8_t output_min =
                (int8_t) lrintf(fminf(fmaxf(node->activation.output_min / output_scale + (float) output_zero_point, -128.0f), 127.0f));
              const int8_t output_max =
                (int8_t) lrintf(fminf(fmaxf(node->activation.output_max / output_scale + (float) output_zero_point, -128.0f), 127.0f));
              status = xnn_create_convolution2d_nhwc_qs8(
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
                (int8_t) values[node->inputs[0]].quantization.zero_point,
                values[node->inputs[0]].quantization.scale,
                values[node->inputs[1]].quantization.scale,
                values[node->inputs[1]].data,
                bias_data,
                (int8_t) output_zero_point,
                output_scale, output_min, output_max,
                node->flags,
                &runtime->opdata[i].operator_object);
              break;
            }
            case xnn_datatype_qcint8:
            {
              const float output_scale = values[node->outputs[0]].quantization.scale;
              const int32_t output_zero_point = values[node->outputs[0]].quantization.zero_point;
              const int8_t output_min =
                (int8_t) lrintf(fminf(fmaxf(node->activation.output_min / output_scale + (float) output_zero_point, -128.0f), 127.0f));
              const int8_t output_max =
                (int8_t) lrintf(fminf(fmaxf(node->activation.output_max / output_scale + (float) output_zero_point, -128.0f), 127.0f));
              status = xnn_create_convolution2d_nhwc_qc8(
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
                (int8_t) values[node->inputs[0]].quantization.zero_point,
                values[node->inputs[0]].quantization.scale,
                values[node->inputs[1]].quantization.channelwise_scale,
                values[node->inputs[1]].data,
                bias_data,
                (int8_t) output_zero_point,
                output_scale, output_min, output_max,
                node->flags,
                &runtime->opdata[i].operator_object);
              break;
            }
#endif  // !defined(XNN_NO_QS8_OPERATORS)
#ifndef XNN_NO_QU8_OPERATORS
            case xnn_datatype_quint8:
            {
              const float output_scale = values[node->outputs[0]].quantization.scale;
              const int32_t output_zero_point = values[node->outputs[0]].quantization.zero_point;
              const uint8_t output_min =
                (uint8_t) lrintf(fminf(fmaxf(node->activation.output_min / output_scale + (float) output_zero_point, 0.0f), 255.0f));
              const uint8_t output_max =
                (uint8_t) lrintf(fminf(fmaxf(node->activation.output_max / output_scale + (float) output_zero_point, 0.0f), 255.0f));
              status = xnn_create_convolution2d_nhwc_qu8(
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
                (uint8_t) values[node->inputs[0]].quantization.zero_point,
                values[node->inputs[0]].quantization.scale,
                (uint8_t) values[node->inputs[1]].quantization.zero_point,
                values[node->inputs[1]].quantization.scale,
                values[node->inputs[1]].data,
                bias_data,
                (uint8_t) output_zero_point,
                output_scale, output_min, output_max,
                node->flags,
                &runtime->opdata[i].operator_object);
              break;
            }
#endif  // !defined(XNN_NO_QU8_OPERATORS)
            default:
              XNN_UNREACHABLE;
          }
        }
        if (status != xnn_status_success) {
          goto error;
        }
        runtime->opdata[i].batch_size = values[node->inputs[0]].shape.dim[0];
        runtime->opdata[i].input_height = values[node->inputs[0]].shape.dim[1];
        runtime->opdata[i].input_width = values[node->inputs[0]].shape.dim[2];
        runtime->opdata[i].inputs[0] = node->inputs[0];
        runtime->opdata[i].outputs[0] = node->outputs[0];
        break;
      }
      case xnn_node_type_clamp:
        status = xnn_create_clamp_nc_f32(
          values[node->inputs[0]].shape.dim[values[node->inputs[0]].shape.num_dims - 1] /* channels */,
          values[node->inputs[0]].shape.dim[values[node->inputs[0]].shape.num_dims - 1] /* input stride */,
          values[node->inputs[0]].shape.dim[values[node->inputs[0]].shape.num_dims - 1] /* output stride */,
          node->activation.output_min,
          node->activation.output_max,
          node->flags,
          &runtime->opdata[i].operator_object);
        if (status != xnn_status_success) {
          goto error;
        }
        runtime->opdata[i].batch_size = product_non_channel_dims(&values[node->inputs[0]].shape);
        runtime->opdata[i].inputs[0] = node->inputs[0];
        runtime->opdata[i].outputs[0] = node->outputs[0];
        break;
      case xnn_node_type_deconvolution_2d:
        assert(values[node->inputs[1]].data != NULL);
        assert(values[node->inputs[2]].data != NULL);
        status = xnn_create_deconvolution2d_nhwc_f32(
          node->params.deconvolution_2d.padding_top,
          node->params.deconvolution_2d.padding_right,
          node->params.deconvolution_2d.padding_bottom,
          node->params.deconvolution_2d.padding_left,
          node->params.deconvolution_2d.kernel_height,
          node->params.deconvolution_2d.kernel_width,
          node->params.deconvolution_2d.upsampling_height,
          node->params.deconvolution_2d.upsampling_width,
          node->params.deconvolution_2d.dilation_height,
          node->params.deconvolution_2d.dilation_width,
          node->params.deconvolution_2d.groups,
          node->params.deconvolution_2d.group_input_channels,
          node->params.deconvolution_2d.group_output_channels,
          node->params.deconvolution_2d.group_input_channels * node->params.deconvolution_2d.groups /* input_pixel_stride */,
          node->params.deconvolution_2d.group_output_channels * node->params.deconvolution_2d.groups /* output_pixel_stride */,
          values[node->inputs[1]].data,
          values[node->inputs[2]].data,
          node->activation.output_min,
          node->activation.output_max,
          node->flags,
          &runtime->opdata[i].operator_object);
        if (status != xnn_status_success) {
          goto error;
        }
        runtime->opdata[i].batch_size = values[node->inputs[0]].shape.dim[0];
        runtime->opdata[i].input_height = values[node->inputs[0]].shape.dim[1];
        runtime->opdata[i].input_width = values[node->inputs[0]].shape.dim[2];
        runtime->opdata[i].adjustment_height = node->params.deconvolution_2d.adjustment_height;
        runtime->opdata[i].adjustment_width = node->params.deconvolution_2d.adjustment_width;
        runtime->opdata[i].inputs[0] = node->inputs[0];
        runtime->opdata[i].outputs[0] = node->outputs[0];
        break;
      case xnn_node_type_depthwise_convolution_2d:
      {
        assert(values[node->inputs[1]].data != NULL);
        const void* bias_data = NULL;
        if (node->num_inputs > 2) {
          bias_data = values[node->inputs[2]].data;
          assert(bias_data != NULL);
        }
        if (values[node->outputs[0]].layout == xnn_layout_type_nchw) {
          assert(values[node->inputs[0]].layout == xnn_layout_type_nchw);
          status = xnn_create_convolution2d_nchw_f32(
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
            node->params.depthwise_convolution_2d.input_channels /* input_channel_stride */,
            node->params.depthwise_convolution_2d.input_channels * node->params.depthwise_convolution_2d.depth_multiplier /* output_channel_stride */,
            values[node->inputs[1]].data,
            bias_data,
            node->activation.output_min,
            node->activation.output_max,
            node->flags | XNN_FLAG_DEPTHWISE_CONVOLUTION,
            &runtime->opdata[i].operator_object);
        } else {
          assert(values[node->inputs[0]].layout == xnn_layout_type_nhwc);
          assert(values[node->outputs[0]].layout == xnn_layout_type_nhwc);
          switch (values[node->inputs[1]].datatype) {
            case xnn_datatype_fp32:
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
                node->params.depthwise_convolution_2d.input_channels /* input_channel_stride */,
                node->params.depthwise_convolution_2d.input_channels * node->params.depthwise_convolution_2d.depth_multiplier /* output_channel_stride */,
                values[node->inputs[1]].data,
                bias_data,
                node->activation.output_min,
                node->activation.output_max,
                node->flags | XNN_FLAG_DEPTHWISE_CONVOLUTION,
                &runtime->opdata[i].operator_object);
              break;
#ifndef XNN_NO_QS8_OPERATORS
            case xnn_datatype_qint8:
            {
              const float output_scale = values[node->outputs[0]].quantization.scale;
              const int32_t output_zero_point = values[node->outputs[0]].quantization.zero_point;
              const int8_t output_min =
                (int8_t) lrintf(fminf(fmaxf(node->activation.output_min / output_scale + (float) output_zero_point, -128.0f), 127.0f));
              const int8_t output_max =
                (int8_t) lrintf(fminf(fmaxf(node->activation.output_max / output_scale + (float) output_zero_point, -128.0f), 127.0f));
              status = xnn_create_convolution2d_nhwc_qs8(
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
                node->params.depthwise_convolution_2d.input_channels /* input_channel_stride */,
                node->params.depthwise_convolution_2d.input_channels * node->params.depthwise_convolution_2d.depth_multiplier /* output_channel_stride */,
                (int8_t) values[node->inputs[0]].quantization.zero_point,
                values[node->inputs[0]].quantization.scale,
                values[node->inputs[1]].quantization.scale,
                values[node->inputs[1]].data,
                bias_data,
                (int8_t) output_zero_point,
                output_scale, output_min, output_max,
                node->flags | XNN_FLAG_DEPTHWISE_CONVOLUTION,
                &runtime->opdata[i].operator_object);
              break;
            }
            case xnn_datatype_qcint8:
            {
              const float output_scale = values[node->outputs[0]].quantization.scale;
              const int32_t output_zero_point = values[node->outputs[0]].quantization.zero_point;
              const int8_t output_min =
                (int8_t) lrintf(fminf(fmaxf(node->activation.output_min / output_scale + (float) output_zero_point, -128.0f), 127.0f));
              const int8_t output_max =
                (int8_t) lrintf(fminf(fmaxf(node->activation.output_max / output_scale + (float) output_zero_point, -128.0f), 127.0f));
              status = xnn_create_convolution2d_nhwc_qc8(
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
                node->params.depthwise_convolution_2d.input_channels /* input_channel_stride */,
                node->params.depthwise_convolution_2d.input_channels * node->params.depthwise_convolution_2d.depth_multiplier /* output_channel_stride */,
                (int8_t) values[node->inputs[0]].quantization.zero_point,
                values[node->inputs[0]].quantization.scale,
                values[node->inputs[1]].quantization.channelwise_scale,
                values[node->inputs[1]].data,
                bias_data,
                (int8_t) output_zero_point,
                output_scale, output_min, output_max,
                node->flags | XNN_FLAG_DEPTHWISE_CONVOLUTION,
                &runtime->opdata[i].operator_object);
              break;
            }
#endif  // !defined(XNN_NO_QS8_OPERATORS)
#ifndef XNN_NO_QU8_OPERATORS
            case xnn_datatype_quint8:
            {
              const float output_scale = values[node->outputs[0]].quantization.scale;
              const int32_t output_zero_point = values[node->outputs[0]].quantization.zero_point;
              const uint8_t output_min =
                (uint8_t) lrintf(fminf(fmaxf(node->activation.output_min / output_scale + (float) output_zero_point, 0.0f), 255.0f));
              const uint8_t output_max =
                (uint8_t) lrintf(fminf(fmaxf(node->activation.output_max / output_scale + (float) output_zero_point, 0.0f), 255.0f));
              status = xnn_create_convolution2d_nhwc_qu8(
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
                node->params.depthwise_convolution_2d.input_channels /* input_channel_stride */,
                node->params.depthwise_convolution_2d.input_channels * node->params.depthwise_convolution_2d.depth_multiplier /* output_channel_stride */,
                (uint8_t) values[node->inputs[0]].quantization.zero_point,
                values[node->inputs[0]].quantization.scale,
                (uint8_t) values[node->inputs[1]].quantization.zero_point,
                values[node->inputs[1]].quantization.scale,
                values[node->inputs[1]].data,
                bias_data,
                (uint8_t) output_zero_point,
                output_scale, output_min, output_max,
                node->flags | XNN_FLAG_DEPTHWISE_CONVOLUTION,
                &runtime->opdata[i].operator_object);
              break;
            }
#endif  // !defined(XNN_NO_QU8_OPERATORS)
            default:
              XNN_UNREACHABLE;
          }
        }
        if (status != xnn_status_success) {
          goto error;
        }
        runtime->opdata[i].batch_size = values[node->inputs[0]].shape.dim[0];
        runtime->opdata[i].input_height = values[node->inputs[0]].shape.dim[1];
        runtime->opdata[i].input_width = values[node->inputs[0]].shape.dim[2];

        runtime->opdata[i].inputs[0] = node->inputs[0];
        runtime->opdata[i].outputs[0] = node->outputs[0];
        break;
      }
      case xnn_node_type_depth_to_space:
        status = xnn_status_unsupported_parameter;
        if (values[node->inputs[0]].layout == xnn_layout_type_nchw) {
          assert(values[node->outputs[0]].layout == xnn_layout_type_nhwc);
          status = xnn_create_depth_to_space_nchw2nhwc_x32(
              values[node->outputs[0]].shape.dim[values[node->outputs[0]].shape.num_dims - 1] /* output channels */,
              values[node->inputs[0]].shape.dim[values[node->inputs[0]].shape.num_dims - 1] /* input stride */,
              values[node->outputs[0]].shape.dim[values[node->outputs[0]].shape.num_dims - 1] /* output stride */,
              node->params.depth_to_space.block_size,
              node->flags,
              &runtime->opdata[i].operator_object);
        } else {
          assert(values[node->inputs[0]].layout == xnn_layout_type_nhwc);
          assert(values[node->outputs[0]].layout == xnn_layout_type_nhwc);
          status = xnn_create_depth_to_space_nhwc_x32(
              values[node->outputs[0]].shape.dim[values[node->outputs[0]].shape.num_dims - 1] /* output channels */,
              values[node->inputs[0]].shape.dim[values[node->inputs[0]].shape.num_dims - 1] /* input stride */,
              values[node->outputs[0]].shape.dim[values[node->outputs[0]].shape.num_dims - 1] /* output stride */,
              node->params.depth_to_space.block_size,
              node->flags,
              &runtime->opdata[i].operator_object);
        }
        if (status != xnn_status_success) {
          goto error;
        }
        runtime->opdata[i].batch_size = values[node->inputs[0]].shape.dim[0];
        runtime->opdata[i].input_height = values[node->inputs[0]].shape.dim[1];
        runtime->opdata[i].input_width = values[node->inputs[0]].shape.dim[2];
        runtime->opdata[i].output_height = values[node->outputs[0]].shape.dim[1];
        runtime->opdata[i].output_width = values[node->outputs[0]].shape.dim[2];
        runtime->opdata[i].inputs[0] = node->inputs[0];
        runtime->opdata[i].outputs[0] = node->outputs[0];
        break;
      case xnn_node_type_divide:
        status = xnn_create_divide_nd_f32(
          node->activation.output_min,
          node->activation.output_max,
          node->flags,
          &runtime->opdata[i].operator_object);
        if (status != xnn_status_success) {
          goto error;
        }
        runtime->opdata[i].shape1.num_dims = values[node->inputs[0]].shape.num_dims;
        runtime->opdata[i].shape2.num_dims = values[node->inputs[1]].shape.num_dims;
        memcpy(runtime->opdata[i].shape1.dim, values[node->inputs[0]].shape.dim, values[node->inputs[0]].shape.num_dims * sizeof(size_t));
        memcpy(runtime->opdata[i].shape2.dim, values[node->inputs[1]].shape.dim, values[node->inputs[1]].shape.num_dims * sizeof(size_t));
        runtime->opdata[i].inputs[0] = node->inputs[0];
        runtime->opdata[i].inputs[1] = node->inputs[1];
        runtime->opdata[i].outputs[0] = node->outputs[0];
        break;
      case xnn_node_type_elu:
        status = xnn_create_elu_nc_f32(
          values[node->inputs[0]].shape.dim[values[node->inputs[0]].shape.num_dims - 1] /* channels */,
          values[node->inputs[0]].shape.dim[values[node->inputs[0]].shape.num_dims - 1] /* input stride */,
          values[node->inputs[0]].shape.dim[values[node->inputs[0]].shape.num_dims - 1] /* output stride */,
          node->params.elu.alpha,
          node->flags,
          &runtime->opdata[i].operator_object);
        if (status != xnn_status_success) {
          goto error;
        }
        runtime->opdata[i].batch_size = product_non_channel_dims(&values[node->inputs[0]].shape);
        runtime->opdata[i].inputs[0] = node->inputs[0];
        runtime->opdata[i].outputs[0] = node->outputs[0];
        break;
      case xnn_node_type_fully_connected:
      {
        const size_t num_input_elements = product_all_dims(&values[node->inputs[0]].shape);
        size_t output_channels, input_channels;
        if (node->flags & XNN_FLAG_TRANSPOSE_WEIGHTS) {
          input_channels = values[node->inputs[1]].shape.dim[0];
          output_channels = values[node->inputs[1]].shape.dim[1];
        } else {
          output_channels = values[node->inputs[1]].shape.dim[0];
          input_channels = values[node->inputs[1]].shape.dim[1];
        }
        const void* bias_data = NULL;
        if (node->num_inputs > 2) {
          bias_data = values[node->inputs[2]].data;
        }
        switch (values[node->outputs[0]].datatype) {
          case xnn_datatype_fp32:
            status = xnn_create_fully_connected_nc_f32(
              input_channels,
              output_channels,
              input_channels /* input stride */,
              output_channels /* output stride */,
              values[node->inputs[1]].data,
              bias_data,
              node->activation.output_min,
              node->activation.output_max,
              node->flags /* flags */,
              &runtime->opdata[i].operator_object);
            break;
#ifndef XNN_NO_QS8_OPERATORS
          case xnn_datatype_qint8:
          {
            const float output_scale = values[node->outputs[0]].quantization.scale;
            const int32_t output_zero_point = values[node->outputs[0]].quantization.zero_point;
            const int8_t output_min =
              (int8_t) lrintf(fminf(fmaxf(node->activation.output_min / output_scale + (float) output_zero_point, -128.0f), 127.0f));
            const int8_t output_max =
              (int8_t) lrintf(fminf(fmaxf(node->activation.output_max / output_scale + (float) output_zero_point, -128.0f), 127.0f));
            status = xnn_create_fully_connected_nc_qs8(
              input_channels,
              output_channels,
              input_channels /* input stride */,
              output_channels /* output stride */,
              (int8_t) values[node->inputs[0]].quantization.zero_point,
              values[node->inputs[0]].quantization.scale,
              values[node->inputs[1]].quantization.scale,
              values[node->inputs[1]].data,
              bias_data,
              (int8_t) output_zero_point,
              output_scale, output_min, output_max,
              node->flags /* flags */,
              &runtime->opdata[i].operator_object);
            break;
          }
#endif  // !defined(XNN_NO_QS8_OPERATORS)
#ifndef XNN_NO_QU8_OPERATORS
          case xnn_datatype_quint8:
          {
            const float output_scale = values[node->outputs[0]].quantization.scale;
            const int32_t output_zero_point = values[node->outputs[0]].quantization.zero_point;
            const uint8_t output_min =
              (uint8_t) lrintf(fminf(fmaxf(node->activation.output_min / output_scale + (float) output_zero_point, 0.0f), 255.0f));
            const uint8_t output_max =
              (uint8_t) lrintf(fminf(fmaxf(node->activation.output_max / output_scale + (float) output_zero_point, 0.0f), 255.0f));
            status = xnn_create_fully_connected_nc_qu8(
              input_channels,
              output_channels,
              input_channels /* input stride */,
              output_channels /* output stride */,
              (uint8_t) values[node->inputs[0]].quantization.zero_point,
              values[node->inputs[0]].quantization.scale,
              (uint8_t) values[node->inputs[1]].quantization.zero_point,
              values[node->inputs[1]].quantization.scale,
              values[node->inputs[1]].data,
              bias_data,
              (uint8_t) output_zero_point,
              output_scale, output_min, output_max,
              node->flags /* flags */,
              &runtime->opdata[i].operator_object);
            break;
          }
#endif  // !defined(XNN_NO_QU8_OPERATORS)
          default:
            XNN_UNREACHABLE;
        }
        if (status != xnn_status_success) {
          goto error;
        }
        runtime->opdata[i].batch_size = num_input_elements / input_channels;
        runtime->opdata[i].inputs[0] = node->inputs[0];
        runtime->opdata[i].outputs[0] = node->outputs[0];
        break;
      }
      case xnn_node_type_floor:
        status = xnn_create_floor_nc_f32(
          values[node->inputs[0]].shape.dim[values[node->inputs[0]].shape.num_dims - 1] /* channels */,
          values[node->inputs[0]].shape.dim[values[node->inputs[0]].shape.num_dims - 1] /* input stride */,
          values[node->inputs[0]].shape.dim[values[node->inputs[0]].shape.num_dims - 1] /* output stride */,
          node->flags,
          &runtime->opdata[i].operator_object);
        if (status != xnn_status_success) {
          goto error;
        }
        runtime->opdata[i].batch_size = product_non_channel_dims(&values[node->inputs[0]].shape);
        runtime->opdata[i].inputs[0] = node->inputs[0];
        runtime->opdata[i].outputs[0] = node->outputs[0];
        break;
      case xnn_node_type_global_average_pooling_2d:
        if (values[node->inputs[0]].layout == xnn_layout_type_nchw) {
          status = xnn_create_global_average_pooling_ncw_f32(
            values[node->inputs[0]].shape.dim[values[node->inputs[0]].shape.num_dims - 1] /* channels */,
            node->activation.output_min,
            node->activation.output_max,
            node->flags,
            &runtime->opdata[i].operator_object);
        } else {
          assert(values[node->inputs[0]].layout == xnn_layout_type_nhwc);
          assert(values[node->outputs[0]].layout == xnn_layout_type_nhwc);
          status = xnn_create_global_average_pooling_nwc_f32(
            values[node->inputs[0]].shape.dim[values[node->inputs[0]].shape.num_dims - 1] /* channels */,
            values[node->inputs[0]].shape.dim[values[node->inputs[0]].shape.num_dims - 1] /* input stride */,
            values[node->inputs[0]].shape.dim[values[node->inputs[0]].shape.num_dims - 1] /* output stride */,
            node->activation.output_min,
            node->activation.output_max,
            node->flags,
            &runtime->opdata[i].operator_object);
        }
        if (status != xnn_status_success) {
          goto error;
        }
        runtime->opdata[i].batch_size = values[node->inputs[0]].shape.dim[0];
        runtime->opdata[i].input_width = values[node->inputs[0]].shape.dim[1] * values[node->inputs[0]].shape.dim[2];
        runtime->opdata[i].inputs[0] = node->inputs[0];
        runtime->opdata[i].outputs[0] = node->outputs[0];
        break;
      case xnn_node_type_hardswish:
        status = xnn_create_hardswish_nc_f32(
          values[node->inputs[0]].shape.dim[values[node->inputs[0]].shape.num_dims - 1] /* channels */,
          values[node->inputs[0]].shape.dim[values[node->inputs[0]].shape.num_dims - 1] /* input stride */,
          values[node->inputs[0]].shape.dim[values[node->inputs[0]].shape.num_dims - 1] /* output stride */,
          node->flags,
          &runtime->opdata[i].operator_object);
        if (status != xnn_status_success) {
          goto error;
        }
        runtime->opdata[i].batch_size = product_non_channel_dims(&values[node->inputs[0]].shape);
        runtime->opdata[i].inputs[0] = node->inputs[0];
        runtime->opdata[i].outputs[0] = node->outputs[0];
        break;
      case xnn_node_type_leaky_relu:
        status = xnn_create_leaky_relu_nc_f32(
          values[node->inputs[0]].shape.dim[values[node->inputs[0]].shape.num_dims - 1] /* channels */,
          values[node->inputs[0]].shape.dim[values[node->inputs[0]].shape.num_dims - 1] /* input stride */,
          values[node->inputs[0]].shape.dim[values[node->inputs[0]].shape.num_dims - 1] /* output stride */,
          node->params.leaky_relu.negative_slope,
          node->flags,
          &runtime->opdata[i].operator_object);
        if (status != xnn_status_success) {
          goto error;
        }
        runtime->opdata[i].batch_size = product_non_channel_dims(&values[node->inputs[0]].shape);
        runtime->opdata[i].inputs[0] = node->inputs[0];
        runtime->opdata[i].outputs[0] = node->outputs[0];
        break;
      case xnn_node_type_max_pooling_2d:
        status = xnn_create_max_pooling2d_nhwc_f32(
          node->params.pooling_2d.padding_top,
          node->params.pooling_2d.padding_right,
          node->params.pooling_2d.padding_bottom,
          node->params.pooling_2d.padding_left,
          node->params.pooling_2d.pooling_height,
          node->params.pooling_2d.pooling_width,
          node->params.pooling_2d.stride_height,
          node->params.pooling_2d.stride_width,
          node->params.pooling_2d.dilation_height,
          node->params.pooling_2d.dilation_width,
          values[node->inputs[0]].shape.dim[values[node->inputs[0]].shape.num_dims - 1] /* channels */,
          values[node->inputs[0]].shape.dim[values[node->inputs[0]].shape.num_dims - 1] /* input stride */,
          values[node->inputs[0]].shape.dim[values[node->inputs[0]].shape.num_dims - 1] /* output stride */,
          node->activation.output_min,
          node->activation.output_max,
          node->flags,
          &runtime->opdata[i].operator_object);
        if (status != xnn_status_success) {
          goto error;
        }
        runtime->opdata[i].batch_size = values[node->inputs[0]].shape.dim[0];
        runtime->opdata[i].input_height = values[node->inputs[0]].shape.dim[1];
        runtime->opdata[i].input_width = values[node->inputs[0]].shape.dim[2];
        runtime->opdata[i].inputs[0] = node->inputs[0];
        runtime->opdata[i].outputs[0] = node->outputs[0];
        break;
      case xnn_node_type_maximum2:
        status = xnn_create_maximum_nd_f32(
          node->flags,
          &runtime->opdata[i].operator_object);
        if (status != xnn_status_success) {
          goto error;
        }
        runtime->opdata[i].shape1.num_dims = values[node->inputs[0]].shape.num_dims;
        runtime->opdata[i].shape2.num_dims = values[node->inputs[1]].shape.num_dims;
        memcpy(runtime->opdata[i].shape1.dim, values[node->inputs[0]].shape.dim, values[node->inputs[0]].shape.num_dims * sizeof(size_t));
        memcpy(runtime->opdata[i].shape2.dim, values[node->inputs[1]].shape.dim, values[node->inputs[1]].shape.num_dims * sizeof(size_t));
        runtime->opdata[i].inputs[0] = node->inputs[0];
        runtime->opdata[i].inputs[1] = node->inputs[1];
        runtime->opdata[i].outputs[0] = node->outputs[0];
        break;
      case xnn_node_type_minimum2:
        status = xnn_create_minimum_nd_f32(
          node->flags,
          &runtime->opdata[i].operator_object);
        if (status != xnn_status_success) {
          goto error;
        }
        runtime->opdata[i].shape1.num_dims = values[node->inputs[0]].shape.num_dims;
        runtime->opdata[i].shape2.num_dims = values[node->inputs[1]].shape.num_dims;
        memcpy(runtime->opdata[i].shape1.dim, values[node->inputs[0]].shape.dim, values[node->inputs[0]].shape.num_dims * sizeof(size_t));
        memcpy(runtime->opdata[i].shape2.dim, values[node->inputs[1]].shape.dim, values[node->inputs[1]].shape.num_dims * sizeof(size_t));
        runtime->opdata[i].inputs[0] = node->inputs[0];
        runtime->opdata[i].inputs[1] = node->inputs[1];
        runtime->opdata[i].outputs[0] = node->outputs[0];
        break;
      case xnn_node_type_multiply2:
        switch (values[node->outputs[0]].datatype) {
          case xnn_datatype_fp32:
            status = xnn_create_multiply_nd_f32(
              node->activation.output_min,
              node->activation.output_max,
              node->flags,
              &runtime->opdata[i].operator_object);
            break;
#ifndef XNN_NO_QS8_OPERATORS
          case xnn_datatype_qint8:
          {
            const float output_scale = values[node->outputs[0]].quantization.scale;
            const int32_t output_zero_point = values[node->outputs[0]].quantization.zero_point;
            const int8_t output_min =
              (int8_t) lrintf(fminf(fmaxf(node->activation.output_min / output_scale + (float) output_zero_point, -128.0f), 127.0f));
            const int8_t output_max =
              (int8_t) lrintf(fminf(fmaxf(node->activation.output_max / output_scale + (float) output_zero_point, -128.0f), 127.0f));
            status = xnn_create_multiply_nd_qs8(
              (int8_t) values[node->inputs[0]].quantization.zero_point,
              values[node->inputs[0]].quantization.scale,
              (int8_t) values[node->inputs[1]].quantization.zero_point,
              values[node->inputs[1]].quantization.scale,
              (int8_t) output_zero_point,
              output_scale,
              output_min,
              output_max,
              node->flags,
              &runtime->opdata[i].operator_object);
            break;
          }
#endif  // !defined(XNN_NO_QS8_OPERATORS)
#ifndef XNN_NO_QU8_OPERATORS
          case xnn_datatype_quint8:
          {
            const float output_scale = values[node->outputs[0]].quantization.scale;
            const int32_t output_zero_point = values[node->outputs[0]].quantization.zero_point;
            const uint8_t output_min =
              (uint8_t) lrintf(fminf(fmaxf(node->activation.output_min / output_scale + (float) output_zero_point, 0.0f), 255.0f));
            const uint8_t output_max =
              (uint8_t) lrintf(fminf(fmaxf(node->activation.output_max / output_scale + (float) output_zero_point, 0.0f), 255.0f));
            status = xnn_create_multiply_nd_qu8(
              (uint8_t) values[node->inputs[0]].quantization.zero_point,
              values[node->inputs[0]].quantization.scale,
              (uint8_t) values[node->inputs[1]].quantization.zero_point,
              values[node->inputs[1]].quantization.scale,
              (uint8_t) output_zero_point,
              output_scale,
              output_min,
              output_max,
              node->flags,
              &runtime->opdata[i].operator_object);
            break;
          }
#endif  // !defined(XNN_NO_QU8_OPERATORS)
          default:
            XNN_UNREACHABLE;
        }
        if (status != xnn_status_success) {
          goto error;
        }
        runtime->opdata[i].shape1.num_dims = values[node->inputs[0]].shape.num_dims;
        runtime->opdata[i].shape2.num_dims = values[node->inputs[1]].shape.num_dims;
        if (values[node->outputs[0]].layout == xnn_layout_type_nchw) {
          assert(values[node->inputs[0]].layout == xnn_layout_type_nchw);
          assert(values[node->inputs[1]].layout == xnn_layout_type_nchw);
          runtime->opdata[i].shape1.dim[0] = values[node->inputs[0]].shape.dim[0];
          runtime->opdata[i].shape1.dim[1] = values[node->inputs[0]].shape.dim[values[node->inputs[0]].shape.num_dims - 1];
          if (values[node->inputs[0]].shape.num_dims > 2) {
            memcpy(&runtime->opdata[i].shape1.dim[2], &values[node->inputs[0]].shape.dim[1], (values[node->inputs[0]].shape.num_dims - 2) * sizeof(size_t));
          }
          runtime->opdata[i].shape2.dim[0] = values[node->inputs[1]].shape.dim[0];
          runtime->opdata[i].shape2.dim[1] = values[node->inputs[1]].shape.dim[values[node->inputs[0]].shape.num_dims - 1];
          if (values[node->inputs[0]].shape.num_dims > 2) {
            memcpy(&runtime->opdata[i].shape2.dim[2], &values[node->inputs[1]].shape.dim[1], (values[node->inputs[1]].shape.num_dims - 2) * sizeof(size_t));
          }
        } else {
          assert(values[node->outputs[0]].layout == xnn_layout_type_nhwc);
          assert(values[node->inputs[0]].layout == xnn_layout_type_nhwc);
          assert(values[node->inputs[1]].layout == xnn_layout_type_nhwc);
          memcpy(runtime->opdata[i].shape1.dim, values[node->inputs[0]].shape.dim, values[node->inputs[0]].shape.num_dims * sizeof(size_t));
          memcpy(runtime->opdata[i].shape2.dim, values[node->inputs[1]].shape.dim, values[node->inputs[1]].shape.num_dims * sizeof(size_t));
        }
        runtime->opdata[i].inputs[0] = node->inputs[0];
        runtime->opdata[i].inputs[1] = node->inputs[1];
        runtime->opdata[i].outputs[0] = node->outputs[0];
        break;
      case xnn_node_type_negate:
        status = xnn_create_negate_nc_f32(
          values[node->inputs[0]].shape.dim[values[node->inputs[0]].shape.num_dims - 1] /* channels */,
          values[node->inputs[0]].shape.dim[values[node->inputs[0]].shape.num_dims - 1] /* input stride */,
          values[node->inputs[0]].shape.dim[values[node->inputs[0]].shape.num_dims - 1] /* output stride */,
          node->flags,
          &runtime->opdata[i].operator_object);
        if (status != xnn_status_success) {
          goto error;
        }
        runtime->opdata[i].batch_size = product_non_channel_dims(&values[node->inputs[0]].shape);
        runtime->opdata[i].inputs[0] = node->inputs[0];
        runtime->opdata[i].outputs[0] = node->outputs[0];
        break;
      case xnn_node_type_prelu:
        status = xnn_create_prelu_nc_f32(
          values[node->inputs[1]].shape.dim[values[node->inputs[1]].shape.num_dims - 1] /* channels */,
          values[node->inputs[1]].shape.dim[values[node->inputs[1]].shape.num_dims - 1] /* input stride */,
          values[node->inputs[1]].shape.dim[values[node->inputs[1]].shape.num_dims - 1] /* output stride */,
          values[node->inputs[1]].data /* negative slope */,
          node->flags,
          &runtime->opdata[i].operator_object);
        if (status != xnn_status_success) {
          goto error;
        }
        runtime->opdata[i].batch_size = product_non_channel_dims(&values[node->inputs[0]].shape);
        runtime->opdata[i].inputs[0] = node->inputs[0];
        runtime->opdata[i].outputs[0] = node->outputs[0];
        break;
      case xnn_node_type_sigmoid:
        status = xnn_create_sigmoid_nc_f32(
          values[node->inputs[0]].shape.dim[values[node->inputs[0]].shape.num_dims - 1] /* channels */,
          values[node->inputs[0]].shape.dim[values[node->inputs[0]].shape.num_dims - 1] /* input stride */,
          values[node->inputs[0]].shape.dim[values[node->inputs[0]].shape.num_dims - 1] /* output stride */,
          node->flags,
          &runtime->opdata[i].operator_object);
        if (status != xnn_status_success) {
          goto error;
        }
        runtime->opdata[i].batch_size = product_non_channel_dims(&values[node->inputs[0]].shape);
        runtime->opdata[i].inputs[0] = node->inputs[0];
        runtime->opdata[i].outputs[0] = node->outputs[0];
        break;
      case xnn_node_type_softmax:
        status = xnn_create_softmax_nc_f32(
          values[node->inputs[0]].shape.dim[values[node->inputs[0]].shape.num_dims - 1] /* channels */,
          values[node->inputs[0]].shape.dim[values[node->inputs[0]].shape.num_dims - 1] /* input stride */,
          values[node->inputs[0]].shape.dim[values[node->inputs[0]].shape.num_dims - 1] /* output stride */,
          node->flags,
          &runtime->opdata[i].operator_object);
        if (status != xnn_status_success) {
          goto error;
        }
        runtime->opdata[i].batch_size = product_non_channel_dims(&values[node->inputs[0]].shape);
        runtime->opdata[i].inputs[0] = node->inputs[0];
        runtime->opdata[i].outputs[0] = node->outputs[0];
        break;
      case xnn_node_type_static_constant_pad:
        status = xnn_create_constant_pad_nd_x32(
          &node->params.static_pad.padding_value,
          node->flags,
          &runtime->opdata[i].operator_object);
        if (status != xnn_status_success) {
          goto error;
        }
        runtime->opdata[i].shape1 = values[node->inputs[0]].shape;
        memcpy(runtime->opdata[i].pre_paddings, node->params.static_pad.pre_paddings, sizeof(size_t) * XNN_MAX_TENSOR_DIMS);
        memcpy(runtime->opdata[i].post_paddings, node->params.static_pad.post_paddings, sizeof(size_t) * XNN_MAX_TENSOR_DIMS);
        runtime->opdata[i].inputs[0] = node->inputs[0];
        runtime->opdata[i].outputs[0] = node->outputs[0];
        break;
      case xnn_node_type_static_reshape:
        status = xnn_create_copy_nc_x32(
          1 /* channels */,
          1 /* input stride */,
          1 /* output stride */,
          node->flags,
          &runtime->opdata[i].operator_object);
        if (status != xnn_status_success) {
          goto error;
        }
        runtime->opdata[i].batch_size = product_all_dims(&values[node->inputs[0]].shape);
        runtime->opdata[i].inputs[0] = node->inputs[0];
        runtime->opdata[i].outputs[0] = node->outputs[0];
        break;
      case xnn_node_type_static_resize_bilinear_2d:
        if (values[node->inputs[0]].layout == xnn_layout_type_nchw) {
          status = xnn_create_resize_bilinear2d_nchw_f32(
            values[node->inputs[0]].shape.dim[values[node->inputs[0]].shape.num_dims - 1] /* channels */,
            values[node->inputs[0]].shape.dim[values[node->inputs[0]].shape.num_dims - 1] /* input stride */,
            values[node->inputs[0]].shape.dim[values[node->inputs[0]].shape.num_dims - 1] /* output stride */,
            node->flags,
            &runtime->opdata[i].operator_object);
        } else {
          assert(values[node->inputs[0]].layout == xnn_layout_type_nhwc);
          assert(values[node->outputs[0]].layout == xnn_layout_type_nhwc);
          status = xnn_create_resize_bilinear2d_nhwc_f32(
            values[node->inputs[0]].shape.dim[values[node->inputs[0]].shape.num_dims - 1] /* channels */,
            values[node->inputs[0]].shape.dim[values[node->inputs[0]].shape.num_dims - 1] /* input stride */,
            values[node->inputs[0]].shape.dim[values[node->inputs[0]].shape.num_dims - 1] /* output stride */,
            node->flags,
            &runtime->opdata[i].operator_object);
        }
        if (status != xnn_status_success) {
          goto error;
        }
        runtime->opdata[i].batch_size = values[node->inputs[0]].shape.dim[0];
        runtime->opdata[i].input_height = values[node->inputs[0]].shape.dim[1];
        runtime->opdata[i].input_width = values[node->inputs[0]].shape.dim[2];
        runtime->opdata[i].output_height = values[node->outputs[0]].shape.dim[1];
        runtime->opdata[i].output_width = values[node->outputs[0]].shape.dim[2];
        runtime->opdata[i].inputs[0] = node->inputs[0];
        runtime->opdata[i].outputs[0] = node->outputs[0];
        break;
      case xnn_node_type_square:
        status = xnn_create_square_nc_f32(
          values[node->inputs[0]].shape.dim[values[node->inputs[0]].shape.num_dims - 1] /* channels */,
          values[node->inputs[0]].shape.dim[values[node->inputs[0]].shape.num_dims - 1] /* input stride */,
          values[node->inputs[0]].shape.dim[values[node->inputs[0]].shape.num_dims - 1] /* output stride */,
          node->flags,
          &runtime->opdata[i].operator_object);
        if (status != xnn_status_success) {
          goto error;
        }
        runtime->opdata[i].batch_size = product_non_channel_dims(&values[node->inputs[0]].shape);
        runtime->opdata[i].inputs[0] = node->inputs[0];
        runtime->opdata[i].outputs[0] = node->outputs[0];
        break;
      case xnn_node_type_square_root:
        status = xnn_create_square_root_nc_f32(
          values[node->inputs[0]].shape.dim[values[node->inputs[0]].shape.num_dims - 1] /* channels */,
          values[node->inputs[0]].shape.dim[values[node->inputs[0]].shape.num_dims - 1] /* input stride */,
          values[node->inputs[0]].shape.dim[values[node->inputs[0]].shape.num_dims - 1] /* output stride */,
          node->flags,
          &runtime->opdata[i].operator_object);
        if (status != xnn_status_success) {
          goto error;
        }
        runtime->opdata[i].batch_size = product_non_channel_dims(&values[node->inputs[0]].shape);
        runtime->opdata[i].inputs[0] = node->inputs[0];
        runtime->opdata[i].outputs[0] = node->outputs[0];
        break;
      case xnn_node_type_squared_difference:
        status = xnn_create_squared_difference_nd_f32(
          node->flags,
          &runtime->opdata[i].operator_object);
        if (status != xnn_status_success) {
          goto error;
        }
        runtime->opdata[i].shape1.num_dims = values[node->inputs[0]].shape.num_dims;
        runtime->opdata[i].shape2.num_dims = values[node->inputs[1]].shape.num_dims;
        memcpy(runtime->opdata[i].shape1.dim, values[node->inputs[0]].shape.dim, values[node->inputs[0]].shape.num_dims * sizeof(size_t));
        memcpy(runtime->opdata[i].shape2.dim, values[node->inputs[1]].shape.dim, values[node->inputs[1]].shape.num_dims * sizeof(size_t));
        runtime->opdata[i].inputs[0] = node->inputs[0];
        runtime->opdata[i].inputs[1] = node->inputs[1];
        runtime->opdata[i].outputs[0] = node->outputs[0];
        break;
      case xnn_node_type_subtract:
        status = xnn_create_subtract_nd_f32(
          node->activation.output_min,
          node->activation.output_max,
          node->flags,
          &runtime->opdata[i].operator_object);
        if (status != xnn_status_success) {
          goto error;
        }
        runtime->opdata[i].shape1.num_dims = values[node->inputs[0]].shape.num_dims;
        runtime->opdata[i].shape2.num_dims = values[node->inputs[1]].shape.num_dims;
        memcpy(runtime->opdata[i].shape1.dim, values[node->inputs[0]].shape.dim, values[node->inputs[0]].shape.num_dims * sizeof(size_t));
        memcpy(runtime->opdata[i].shape2.dim, values[node->inputs[1]].shape.dim, values[node->inputs[1]].shape.num_dims * sizeof(size_t));
        runtime->opdata[i].inputs[0] = node->inputs[0];
        runtime->opdata[i].inputs[1] = node->inputs[1];
        runtime->opdata[i].outputs[0] = node->outputs[0];
        break;
      case xnn_node_type_unpooling_2d:
        status = xnn_create_unpooling2d_nhwc_x32(
          node->params.pooling_2d.padding_top,
          node->params.pooling_2d.padding_right,
          node->params.pooling_2d.padding_bottom,
          node->params.pooling_2d.padding_left,
          node->params.pooling_2d.pooling_height,
          node->params.pooling_2d.pooling_width,
          values[node->inputs[0]].shape.dim[values[node->inputs[0]].shape.num_dims - 1] /* channels */,
          values[node->inputs[0]].shape.dim[values[node->inputs[0]].shape.num_dims - 1] /* input stride */,
          values[node->inputs[0]].shape.dim[values[node->inputs[0]].shape.num_dims - 1] /* output stride */,
          node->flags,
          &runtime->opdata[i].operator_object);
        if (status != xnn_status_success) {
          goto error;
        }
        runtime->opdata[i].batch_size = values[node->inputs[0]].shape.dim[0];
        runtime->opdata[i].input_height = values[node->inputs[0]].shape.dim[1];
        runtime->opdata[i].input_width = values[node->inputs[0]].shape.dim[2];
        runtime->opdata[i].inputs[0] = node->inputs[0];
        runtime->opdata[i].inputs[1] = node->inputs[1];
        runtime->opdata[i].outputs[0] = node->outputs[0];
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

  struct xnn_value_allocation_tracker mem_alloc_tracker;
  xnn_init_value_allocation_tracker(&mem_alloc_tracker, subgraph);

  for (uint32_t i = 0; i < subgraph->num_values; i++) {
    const struct xnn_value* value = &subgraph->values[i];
    struct xnn_blob* blob = &runtime->blobs[i];
    if (value->datatype != xnn_datatype_invalid && value->type == xnn_value_type_dense_tensor) {
      blob->size = xnn_tensor_get_size(subgraph, i);
      blob->data = (void*) value->data;
      if (blob->data == NULL) {
        if ((value->flags & (XNN_VALUE_FLAG_EXTERNAL_INPUT | XNN_VALUE_FLAG_EXTERNAL_OUTPUT)) == 0) {
          // Value is purely internal to the runtime, and must be allocated in its workspace.
          xnn_add_value_allocation_tracker(&mem_alloc_tracker, i, round_up_po2(blob->size, XNN_EXTRA_BYTES));
        } else {
          // Value is non-static and external to the runtime: must be specified via a call to xnn_setup_runtime.
          blob->external = true;
        }
      }
    }
  }
  xnn_plan_value_allocation_tracker(&mem_alloc_tracker);

  if (mem_alloc_tracker.mem_arena_size != 0) {
    // XNN_EXTRA_BYTES ensures that out-of-bound reads of intermediate values don't segfault.
    const size_t mem_arena_size = mem_alloc_tracker.mem_arena_size + XNN_EXTRA_BYTES;
    runtime->workspace = xnn_allocate_simd_memory(mem_arena_size);
    if (runtime->workspace == NULL) {
      xnn_log_error("failed to allocate %zu bytes for runtime workspace", mem_arena_size);
      xnn_release_value_allocation_tracker(&mem_alloc_tracker);
      goto error;
    }
    for (size_t i = 0; i < subgraph->num_values; i++) {
      const struct xnn_value* value = &subgraph->values[i];
      struct xnn_blob* blob = &runtime->blobs[i];
      if (value->datatype != xnn_datatype_invalid && value->type == xnn_value_type_dense_tensor) {
        if (value->data == NULL && !blob->external) {
          // Value is purely internal to the runtime, allocate it in the workspace.
          blob->data = (void*) ((uintptr_t) runtime->workspace + mem_alloc_tracker.usage[i].alloc_offset);
        }
      }
    }
  }
  xnn_release_value_allocation_tracker(&mem_alloc_tracker);

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
    const struct xnn_operator_data* opdata = &runtime->opdata[i];
    if (opdata->operator_object == NULL) {
      // Operator was removed during optimization
      continue;
    }

    enum xnn_status status = xnn_status_success;
    switch (opdata->operator_object->type) {
      case xnn_operator_type_abs_nc_f32:
        assert(runtime->blobs[opdata->inputs[0]].data != NULL);
        assert(runtime->blobs[opdata->outputs[0]].data != NULL);
        status = xnn_setup_abs_nc_f32(
          opdata->operator_object,
          opdata->batch_size,
          runtime->blobs[opdata->inputs[0]].data,
          runtime->blobs[opdata->outputs[0]].data,
          runtime->threadpool);
        break;
      case xnn_operator_type_add_nd_f32:
        assert(runtime->blobs[opdata->inputs[0]].data != NULL);
        assert(runtime->blobs[opdata->inputs[1]].data != NULL);
        assert(runtime->blobs[opdata->outputs[0]].data != NULL);
        status = xnn_setup_add_nd_f32(
          opdata->operator_object,
          opdata->shape1.num_dims,
          opdata->shape1.dim,
          opdata->shape2.num_dims,
          opdata->shape2.dim,
          runtime->blobs[opdata->inputs[0]].data,
          runtime->blobs[opdata->inputs[1]].data,
          runtime->blobs[opdata->outputs[0]].data,
          runtime->threadpool);
        break;
#ifndef XNN_NO_QS8_OPERATORS
      case xnn_operator_type_add_nd_qs8:
        assert(runtime->blobs[opdata->inputs[0]].data != NULL);
        assert(runtime->blobs[opdata->inputs[1]].data != NULL);
        assert(runtime->blobs[opdata->outputs[0]].data != NULL);
        status = xnn_setup_add_nd_qs8(
          opdata->operator_object,
          opdata->shape1.num_dims,
          opdata->shape1.dim,
          opdata->shape2.num_dims,
          opdata->shape2.dim,
          runtime->blobs[opdata->inputs[0]].data,
          runtime->blobs[opdata->inputs[1]].data,
          runtime->blobs[opdata->outputs[0]].data,
          runtime->threadpool);
        break;
#endif  // !defined(XNN_NO_QS8_OPERATORS)
#ifndef XNN_NO_QU8_OPERATORS
      case xnn_operator_type_add_nd_qu8:
        assert(runtime->blobs[opdata->inputs[0]].data != NULL);
        assert(runtime->blobs[opdata->inputs[1]].data != NULL);
        assert(runtime->blobs[opdata->outputs[0]].data != NULL);
        status = xnn_setup_add_nd_qu8(
          opdata->operator_object,
          opdata->shape1.num_dims,
          opdata->shape1.dim,
          opdata->shape2.num_dims,
          opdata->shape2.dim,
          runtime->blobs[opdata->inputs[0]].data,
          runtime->blobs[opdata->inputs[1]].data,
          runtime->blobs[opdata->outputs[0]].data,
          runtime->threadpool);
        break;
#endif  // !defined(XNN_NO_QU8_OPERATORS)
      case xnn_operator_type_argmax_pooling_nhwc_f32:
        assert(runtime->blobs[opdata->inputs[0]].data != NULL);
        assert(runtime->blobs[opdata->outputs[0]].data != NULL);
        assert(runtime->blobs[opdata->outputs[1]].data != NULL);
        status = xnn_setup_argmax_pooling2d_nhwc_f32(
          opdata->operator_object,
          opdata->batch_size,
          opdata->input_height,
          opdata->input_width,
          runtime->blobs[opdata->inputs[0]].data,
          runtime->blobs[opdata->outputs[0]].data,
          runtime->blobs[opdata->outputs[1]].data,
          runtime->threadpool);
        break;
      case xnn_operator_type_average_pooling_nhwc_f32:
        assert(runtime->blobs[opdata->inputs[0]].data != NULL);
        assert(runtime->blobs[opdata->outputs[0]].data != NULL);
        status = xnn_setup_average_pooling2d_nhwc_f32(
          opdata->operator_object,
          opdata->batch_size,
          opdata->input_height,
          opdata->input_width,
          runtime->blobs[opdata->inputs[0]].data,
          runtime->blobs[opdata->outputs[0]].data,
          runtime->threadpool);
        break;
      case xnn_operator_type_bankers_rounding_nc_f32:
        assert(runtime->blobs[opdata->inputs[0]].data != NULL);
        assert(runtime->blobs[opdata->outputs[0]].data != NULL);
        status = xnn_setup_bankers_rounding_nc_f32(
          opdata->operator_object,
          opdata->batch_size,
          runtime->blobs[opdata->inputs[0]].data,
          runtime->blobs[opdata->outputs[0]].data,
          runtime->threadpool);
        break;
      case xnn_operator_type_ceiling_nc_f32:
        assert(runtime->blobs[opdata->inputs[0]].data != NULL);
        assert(runtime->blobs[opdata->outputs[0]].data != NULL);
        status = xnn_setup_ceiling_nc_f32(
          opdata->operator_object,
          opdata->batch_size,
          runtime->blobs[opdata->inputs[0]].data,
          runtime->blobs[opdata->outputs[0]].data,
          runtime->threadpool);
        break;
      case xnn_operator_type_constant_pad_nd_x32:
        assert(runtime->blobs[opdata->inputs[0]].data != NULL);
        assert(runtime->blobs[opdata->outputs[0]].data != NULL);
        status = xnn_setup_constant_pad_nd_x32(
          opdata->operator_object,
          opdata->shape1.num_dims,
          opdata->shape1.dim,
          opdata->pre_paddings,
          opdata->post_paddings,
          runtime->blobs[opdata->inputs[0]].data,
          runtime->blobs[opdata->outputs[0]].data,
          runtime->threadpool);
        break;
      case xnn_operator_type_convolution_nchw_f32:
        assert(runtime->blobs[opdata->inputs[0]].data != NULL);
        assert(runtime->blobs[opdata->outputs[0]].data != NULL);
        status = xnn_setup_convolution2d_nchw_f32(
          opdata->operator_object,
          opdata->batch_size,
          opdata->input_height,
          opdata->input_width,
          runtime->blobs[opdata->inputs[0]].data,
          runtime->blobs[opdata->outputs[0]].data,
          runtime->threadpool);
        break;
      case xnn_operator_type_convolution_nhwc_f32:
        assert(runtime->blobs[opdata->inputs[0]].data != NULL);
        assert(runtime->blobs[opdata->outputs[0]].data != NULL);
        status = xnn_setup_convolution2d_nhwc_f32(
          opdata->operator_object,
          opdata->batch_size,
          opdata->input_height,
          opdata->input_width,
          runtime->blobs[opdata->inputs[0]].data,
          runtime->blobs[opdata->outputs[0]].data,
          runtime->threadpool);
        break;
#ifndef XNN_NO_QS8_OPERATORS
      case xnn_operator_type_convolution_nhwc_qc8:
        assert(runtime->blobs[opdata->inputs[0]].data != NULL);
        assert(runtime->blobs[opdata->outputs[0]].data != NULL);
        status = xnn_setup_convolution2d_nhwc_qc8(
          opdata->operator_object,
          opdata->batch_size,
          opdata->input_height,
          opdata->input_width,
          runtime->blobs[opdata->inputs[0]].data,
          runtime->blobs[opdata->outputs[0]].data,
          runtime->threadpool);
        break;
      case xnn_operator_type_convolution_nhwc_qs8:
        assert(runtime->blobs[opdata->inputs[0]].data != NULL);
        assert(runtime->blobs[opdata->outputs[0]].data != NULL);
        status = xnn_setup_convolution2d_nhwc_qs8(
          opdata->operator_object,
          opdata->batch_size,
          opdata->input_height,
          opdata->input_width,
          runtime->blobs[opdata->inputs[0]].data,
          runtime->blobs[opdata->outputs[0]].data,
          runtime->threadpool);
        break;
#endif  // !defined(XNN_NO_QS8_OPERATORS)
#ifndef XNN_NO_QU8_OPERATORS
      case xnn_operator_type_convolution_nhwc_qu8:
        assert(runtime->blobs[opdata->inputs[0]].data != NULL);
        assert(runtime->blobs[opdata->outputs[0]].data != NULL);
        status = xnn_setup_convolution2d_nhwc_qu8(
          opdata->operator_object,
          opdata->batch_size,
          opdata->input_height,
          opdata->input_width,
          runtime->blobs[opdata->inputs[0]].data,
          runtime->blobs[opdata->outputs[0]].data,
          runtime->threadpool);
        break;
#endif  // !defined(XNN_NO_QU8_OPERATORS)
      case xnn_operator_type_copy_nc_x32:
        assert(runtime->blobs[opdata->inputs[0]].data != NULL);
        assert(runtime->blobs[opdata->outputs[0]].data != NULL);
        status = xnn_setup_copy_nc_x32(
          opdata->operator_object,
          opdata->batch_size,
          runtime->blobs[opdata->inputs[0]].data,
          runtime->blobs[opdata->outputs[0]].data,
          runtime->threadpool);
        break;
      case xnn_operator_type_clamp_nc_f32:
        assert(runtime->blobs[opdata->inputs[0]].data != NULL);
        assert(runtime->blobs[opdata->outputs[0]].data != NULL);
        status = xnn_setup_clamp_nc_f32(
          opdata->operator_object,
          opdata->batch_size,
          runtime->blobs[opdata->inputs[0]].data,
          runtime->blobs[opdata->outputs[0]].data,
          runtime->threadpool);
        break;
      case xnn_operator_type_deconvolution_nhwc_f32:
        assert(runtime->blobs[opdata->inputs[0]].data != NULL);
        assert(runtime->blobs[opdata->outputs[0]].data != NULL);
        status = xnn_setup_deconvolution2d_nhwc_f32(
          opdata->operator_object,
          opdata->batch_size,
          opdata->input_height,
          opdata->input_width,
          opdata->adjustment_height,
          opdata->adjustment_width,
          runtime->blobs[opdata->inputs[0]].data,
          runtime->blobs[opdata->outputs[0]].data,
          runtime->threadpool);
        break;
      case xnn_operator_type_depth_to_space_nchw2nhwc_x32:
        assert(runtime->blobs[opdata->inputs[0]].data != NULL);
        assert(runtime->blobs[opdata->outputs[0]].data != NULL);
        status = xnn_setup_depth_to_space_nchw2nhwc_x32(
            opdata->operator_object,
            opdata->batch_size,
            opdata->input_height,
            opdata->input_width,
            runtime->blobs[opdata->inputs[0]].data,
            runtime->blobs[opdata->outputs[0]].data,
            runtime->threadpool);
        break;
      case xnn_operator_type_depth_to_space_nhwc_x32:
        assert(runtime->blobs[opdata->inputs[0]].data != NULL);
        assert(runtime->blobs[opdata->outputs[0]].data != NULL);
        status = xnn_setup_depth_to_space_nhwc_x32(
            opdata->operator_object,
            opdata->batch_size,
            opdata->input_height,
            opdata->input_width,
            runtime->blobs[opdata->inputs[0]].data,
            runtime->blobs[opdata->outputs[0]].data,
            runtime->threadpool);
        break;
      case xnn_operator_type_divide_nd_f32:
        assert(runtime->blobs[opdata->inputs[0]].data != NULL);
        assert(runtime->blobs[opdata->inputs[1]].data != NULL);
        assert(runtime->blobs[opdata->outputs[0]].data != NULL);
        status = xnn_setup_divide_nd_f32(
          opdata->operator_object,
          opdata->shape1.num_dims,
          opdata->shape1.dim,
          opdata->shape2.num_dims,
          opdata->shape2.dim,
          runtime->blobs[opdata->inputs[0]].data,
          runtime->blobs[opdata->inputs[1]].data,
          runtime->blobs[opdata->outputs[0]].data,
          runtime->threadpool);
        break;
      case xnn_operator_type_elu_nc_f32:
        assert(runtime->blobs[opdata->inputs[0]].data != NULL);
        assert(runtime->blobs[opdata->outputs[0]].data != NULL);
        status = xnn_setup_elu_nc_f32(
          opdata->operator_object,
          opdata->batch_size,
          runtime->blobs[opdata->inputs[0]].data,
          runtime->blobs[opdata->outputs[0]].data,
          runtime->threadpool);
        break;
      case xnn_operator_type_fully_connected_nc_f32:
        assert(runtime->blobs[opdata->inputs[0]].data != NULL);
        assert(runtime->blobs[opdata->outputs[0]].data != NULL);
        status = xnn_setup_fully_connected_nc_f32(
          opdata->operator_object,
          opdata->batch_size,
          runtime->blobs[opdata->inputs[0]].data,
          runtime->blobs[opdata->outputs[0]].data,
          runtime->threadpool);
        break;
#ifndef XNN_NO_QS8_OPERATORS
      case xnn_operator_type_fully_connected_nc_qs8:
        assert(runtime->blobs[opdata->inputs[0]].data != NULL);
        assert(runtime->blobs[opdata->outputs[0]].data != NULL);
        status = xnn_setup_fully_connected_nc_qs8(
          opdata->operator_object,
          opdata->batch_size,
          runtime->blobs[opdata->inputs[0]].data,
          runtime->blobs[opdata->outputs[0]].data,
          runtime->threadpool);
        break;
#endif  // !defined(XNN_NO_QS8_OPERATORS)
#ifndef XNN_NO_QU8_OPERATORS
      case xnn_operator_type_fully_connected_nc_qu8:
        assert(runtime->blobs[opdata->inputs[0]].data != NULL);
        assert(runtime->blobs[opdata->outputs[0]].data != NULL);
        status = xnn_setup_fully_connected_nc_qu8(
          opdata->operator_object,
          opdata->batch_size,
          runtime->blobs[opdata->inputs[0]].data,
          runtime->blobs[opdata->outputs[0]].data,
          runtime->threadpool);
        break;
#endif  // !defined(XNN_NO_QU8_OPERATORS)
      case xnn_operator_type_floor_nc_f32:
        assert(runtime->blobs[opdata->inputs[0]].data != NULL);
        assert(runtime->blobs[opdata->outputs[0]].data != NULL);
        status = xnn_setup_floor_nc_f32(
          opdata->operator_object,
          opdata->batch_size,
          runtime->blobs[opdata->inputs[0]].data,
          runtime->blobs[opdata->outputs[0]].data,
          runtime->threadpool);
        break;
      case xnn_operator_type_global_average_pooling_ncw_f32:
        assert(runtime->blobs[opdata->inputs[0]].data != NULL);
        assert(runtime->blobs[opdata->outputs[0]].data != NULL);
        status = xnn_setup_global_average_pooling_ncw_f32(
          opdata->operator_object,
          opdata->batch_size,
          opdata->input_width,
          runtime->blobs[opdata->inputs[0]].data,
          runtime->blobs[opdata->outputs[0]].data,
          runtime->threadpool);
        break;
      case xnn_operator_type_global_average_pooling_nwc_f32:
        assert(runtime->blobs[opdata->inputs[0]].data != NULL);
        assert(runtime->blobs[opdata->outputs[0]].data != NULL);
        status = xnn_setup_global_average_pooling_nwc_f32(
          opdata->operator_object,
          opdata->batch_size,
          opdata->input_width,
          runtime->blobs[opdata->inputs[0]].data,
          runtime->blobs[opdata->outputs[0]].data,
          runtime->threadpool);
        break;
      case xnn_operator_type_hardswish_nc_f32:
        assert(runtime->blobs[opdata->inputs[0]].data != NULL);
        assert(runtime->blobs[opdata->outputs[0]].data != NULL);
        status = xnn_setup_hardswish_nc_f32(
          opdata->operator_object,
          opdata->batch_size,
          runtime->blobs[opdata->inputs[0]].data,
          runtime->blobs[opdata->outputs[0]].data,
          runtime->threadpool);
        break;
      case xnn_operator_type_leaky_relu_nc_f32:
        assert(runtime->blobs[opdata->inputs[0]].data != NULL);
        assert(runtime->blobs[opdata->outputs[0]].data != NULL);
        status = xnn_setup_leaky_relu_nc_f32(
          opdata->operator_object,
          opdata->batch_size,
          runtime->blobs[opdata->inputs[0]].data,
          runtime->blobs[opdata->outputs[0]].data,
          runtime->threadpool);
        break;
      case xnn_operator_type_max_pooling_nhwc_f32:
        assert(runtime->blobs[opdata->inputs[0]].data != NULL);
        assert(runtime->blobs[opdata->outputs[0]].data != NULL);
        status = xnn_setup_max_pooling2d_nhwc_f32(
          opdata->operator_object,
          opdata->batch_size,
          opdata->input_height,
          opdata->input_width,
          runtime->blobs[opdata->inputs[0]].data,
          runtime->blobs[opdata->outputs[0]].data,
          runtime->threadpool);
        break;
      case xnn_operator_type_maximum_nd_f32:
        assert(runtime->blobs[opdata->inputs[0]].data != NULL);
        assert(runtime->blobs[opdata->inputs[1]].data != NULL);
        assert(runtime->blobs[opdata->outputs[0]].data != NULL);
        status = xnn_setup_maximum_nd_f32(
          opdata->operator_object,
          opdata->shape1.num_dims,
          opdata->shape1.dim,
          opdata->shape2.num_dims,
          opdata->shape2.dim,
          runtime->blobs[opdata->inputs[0]].data,
          runtime->blobs[opdata->inputs[1]].data,
          runtime->blobs[opdata->outputs[0]].data,
          runtime->threadpool);
        break;
      case xnn_operator_type_minimum_nd_f32:
        assert(runtime->blobs[opdata->inputs[0]].data != NULL);
        assert(runtime->blobs[opdata->inputs[1]].data != NULL);
        assert(runtime->blobs[opdata->outputs[0]].data != NULL);
        status = xnn_setup_minimum_nd_f32(
          opdata->operator_object,
          opdata->shape1.num_dims,
          opdata->shape1.dim,
          opdata->shape2.num_dims,
          opdata->shape2.dim,
          runtime->blobs[opdata->inputs[0]].data,
          runtime->blobs[opdata->inputs[1]].data,
          runtime->blobs[opdata->outputs[0]].data,
          runtime->threadpool);
        break;
      case xnn_operator_type_multiply_nd_f32:
        assert(runtime->blobs[opdata->inputs[0]].data != NULL);
        assert(runtime->blobs[opdata->inputs[1]].data != NULL);
        assert(runtime->blobs[opdata->outputs[0]].data != NULL);
        status = xnn_setup_multiply_nd_f32(
          opdata->operator_object,
          opdata->shape1.num_dims,
          opdata->shape1.dim,
          opdata->shape2.num_dims,
          opdata->shape2.dim,
          runtime->blobs[opdata->inputs[0]].data,
          runtime->blobs[opdata->inputs[1]].data,
          runtime->blobs[opdata->outputs[0]].data,
          runtime->threadpool);
        break;
#ifndef XNN_NO_QS8_OPERATORS
      case xnn_operator_type_multiply_nd_qs8:
        assert(runtime->blobs[opdata->inputs[0]].data != NULL);
        assert(runtime->blobs[opdata->inputs[1]].data != NULL);
        assert(runtime->blobs[opdata->outputs[0]].data != NULL);
        status = xnn_setup_multiply_nd_qs8(
          opdata->operator_object,
          opdata->shape1.num_dims,
          opdata->shape1.dim,
          opdata->shape2.num_dims,
          opdata->shape2.dim,
          runtime->blobs[opdata->inputs[0]].data,
          runtime->blobs[opdata->inputs[1]].data,
          runtime->blobs[opdata->outputs[0]].data,
          runtime->threadpool);
        break;
#endif  // !defined(XNN_NO_QS8_OPERATORS)
#ifndef XNN_NO_QU8_OPERATORS
      case xnn_operator_type_multiply_nd_qu8:
        assert(runtime->blobs[opdata->inputs[0]].data != NULL);
        assert(runtime->blobs[opdata->inputs[1]].data != NULL);
        assert(runtime->blobs[opdata->outputs[0]].data != NULL);
        status = xnn_setup_multiply_nd_qu8(
          opdata->operator_object,
          opdata->shape1.num_dims,
          opdata->shape1.dim,
          opdata->shape2.num_dims,
          opdata->shape2.dim,
          runtime->blobs[opdata->inputs[0]].data,
          runtime->blobs[opdata->inputs[1]].data,
          runtime->blobs[opdata->outputs[0]].data,
          runtime->threadpool);
        break;
#endif  // !defined(XNN_NO_QU8_OPERATORS)
      case xnn_operator_type_negate_nc_f32:
        assert(runtime->blobs[opdata->inputs[0]].data != NULL);
        assert(runtime->blobs[opdata->outputs[0]].data != NULL);
        status = xnn_setup_negate_nc_f32(
          opdata->operator_object,
          opdata->batch_size,
          runtime->blobs[opdata->inputs[0]].data,
          runtime->blobs[opdata->outputs[0]].data,
          runtime->threadpool);
        break;
      case xnn_operator_type_prelu_nc_f32:
        assert(runtime->blobs[opdata->inputs[0]].data != NULL);
        assert(runtime->blobs[opdata->outputs[0]].data != NULL);
        status = xnn_setup_prelu_nc_f32(
          opdata->operator_object,
          opdata->batch_size,
          runtime->blobs[opdata->inputs[0]].data,
          runtime->blobs[opdata->outputs[0]].data,
          runtime->threadpool);
        break;
      case xnn_operator_type_resize_bilinear_nchw_f32:
        assert(runtime->blobs[opdata->inputs[0]].data != NULL);
        assert(runtime->blobs[opdata->outputs[0]].data != NULL);
        status = xnn_setup_resize_bilinear2d_nchw_f32(
          opdata->operator_object,
          opdata->batch_size,
          opdata->input_height,
          opdata->input_width,
          opdata->output_height,
          opdata->output_width,
          runtime->blobs[opdata->inputs[0]].data,
          runtime->blobs[opdata->outputs[0]].data,
          runtime->threadpool);
        break;
      case xnn_operator_type_resize_bilinear_nhwc_f32:
        assert(runtime->blobs[opdata->inputs[0]].data != NULL);
        assert(runtime->blobs[opdata->outputs[0]].data != NULL);
        status = xnn_setup_resize_bilinear2d_nhwc_f32(
          opdata->operator_object,
          opdata->batch_size,
          opdata->input_height,
          opdata->input_width,
          opdata->output_height,
          opdata->output_width,
          runtime->blobs[opdata->inputs[0]].data,
          runtime->blobs[opdata->outputs[0]].data,
          runtime->threadpool);
        break;
      case xnn_operator_type_sigmoid_nc_f32:
        assert(runtime->blobs[opdata->inputs[0]].data != NULL);
        assert(runtime->blobs[opdata->outputs[0]].data != NULL);
        status = xnn_setup_sigmoid_nc_f32(
          opdata->operator_object,
          opdata->batch_size,
          runtime->blobs[opdata->inputs[0]].data,
          runtime->blobs[opdata->outputs[0]].data,
          runtime->threadpool);
        break;
      case xnn_operator_type_softmax_nc_f32:
        assert(runtime->blobs[opdata->inputs[0]].data != NULL);
        assert(runtime->blobs[opdata->outputs[0]].data != NULL);
        status = xnn_setup_softmax_nc_f32(
          opdata->operator_object,
          opdata->batch_size,
          runtime->blobs[opdata->inputs[0]].data,
          runtime->blobs[opdata->outputs[0]].data,
          runtime->threadpool);
        break;
      case xnn_operator_type_square_nc_f32:
        assert(runtime->blobs[opdata->inputs[0]].data != NULL);
        assert(runtime->blobs[opdata->outputs[0]].data != NULL);
        status = xnn_setup_square_nc_f32(
          opdata->operator_object,
          opdata->batch_size,
          runtime->blobs[opdata->inputs[0]].data,
          runtime->blobs[opdata->outputs[0]].data,
          runtime->threadpool);
        break;
      case xnn_operator_type_square_root_nc_f32:
        assert(runtime->blobs[opdata->inputs[0]].data != NULL);
        assert(runtime->blobs[opdata->outputs[0]].data != NULL);
        status = xnn_setup_square_root_nc_f32(
          opdata->operator_object,
          opdata->batch_size,
          runtime->blobs[opdata->inputs[0]].data,
          runtime->blobs[opdata->outputs[0]].data,
          runtime->threadpool);
        break;
      case xnn_operator_type_squared_difference_nd_f32:
        assert(runtime->blobs[opdata->inputs[0]].data != NULL);
        assert(runtime->blobs[opdata->inputs[1]].data != NULL);
        assert(runtime->blobs[opdata->outputs[0]].data != NULL);
        status = xnn_setup_squared_difference_nd_f32(
          opdata->operator_object,
          opdata->shape1.num_dims,
          opdata->shape1.dim,
          opdata->shape2.num_dims,
          opdata->shape2.dim,
          runtime->blobs[opdata->inputs[0]].data,
          runtime->blobs[opdata->inputs[1]].data,
          runtime->blobs[opdata->outputs[0]].data,
          runtime->threadpool);
        break;
      case xnn_operator_type_subtract_nd_f32:
        assert(runtime->blobs[opdata->inputs[0]].data != NULL);
        assert(runtime->blobs[opdata->inputs[1]].data != NULL);
        assert(runtime->blobs[opdata->outputs[0]].data != NULL);
        status = xnn_setup_subtract_nd_f32(
          opdata->operator_object,
          opdata->shape1.num_dims,
          opdata->shape1.dim,
          opdata->shape2.num_dims,
          opdata->shape2.dim,
          runtime->blobs[opdata->inputs[0]].data,
          runtime->blobs[opdata->inputs[1]].data,
          runtime->blobs[opdata->outputs[0]].data,
          runtime->threadpool);
        break;
      case xnn_operator_type_unpooling_nhwc_x32:
        assert(runtime->blobs[opdata->inputs[0]].data != NULL);
        assert(runtime->blobs[opdata->inputs[1]].data != NULL);
        assert(runtime->blobs[opdata->outputs[0]].data != NULL);
        status = xnn_setup_unpooling2d_nhwc_x32(
          opdata->operator_object,
          opdata->batch_size,
          opdata->input_height,
          opdata->input_width,
          runtime->blobs[opdata->inputs[0]].data,
          runtime->blobs[opdata->inputs[1]].data,
          runtime->blobs[opdata->outputs[0]].data,
          runtime->threadpool);
        break;
      default:
        xnn_log_fatal("unexpected operator type %s in operator #%zu",
          xnn_operator_type_to_string(opdata->operator_object->type), i);
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
    if (runtime->opdata[i].operator_object == NULL) {
      // Operator was removed after fusion
      continue;
    }

    const enum xnn_status status = xnn_run_operator(runtime->opdata[i].operator_object, runtime->threadpool);
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
    if (runtime->opdata != NULL) {
      for (size_t i = 0; i < runtime->num_ops; i++) {
        xnn_delete_operator(runtime->opdata[i].operator_object);
      }
      xnn_release_memory(runtime->opdata);

      xnn_release_memory(runtime->blobs);
      xnn_release_simd_memory(runtime->workspace);
    }
    xnn_release_memory(runtime);
  }
  return xnn_status_success;
}
