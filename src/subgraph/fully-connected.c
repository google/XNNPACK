// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <inttypes.h>
#include <stddef.h>
#include <stdint.h>

#include "include/xnnpack.h"
#include "src/xnnpack/allocation-type.h"
#include "src/xnnpack/common.h"
#include "src/xnnpack/config-types.h"
#include "src/xnnpack/config.h"
#include "src/xnnpack/internal.h"
#include "src/xnnpack/log.h"
#include "src/xnnpack/node-type.h"
#include "src/xnnpack/operator-type.h"
#include "src/xnnpack/operator.h"
#include "src/xnnpack/requantization.h"
#include "src/xnnpack/subgraph-validation.h"
#include "src/xnnpack/subgraph.h"
#include <pthreadpool.h>

// Format is input_type, weights type, output type, (dynamic)?
enum fully_connected_op_type {
  fc_type_invalid = 0,
  fc_type_f16_f16_f16 = 1,
  fc_type_f16_f16_f16_dynamic = 2,
  fc_type_f16_f32_f16 = 3,
  fc_type_f16_f32_f16_dynamic = 4,
  fc_type_qd8_f16_qc4w = 5,
  fc_type_qd8_f16_qb4w = 6,
  fc_type_qd8_f16_qc8w = 7,
  fc_type_f32_f32_f32 = 8,
  fc_type_f32_f32_f32_dynamic = 9,
  fc_type_qd8_f32_qb4w = 10,
  fc_type_f32_f32_qc4w = 11,
  fc_type_qd8_f32_qc4w = 12,
  fc_type_qp8_f32_qc4w = 13,
  fc_type_f32_f32_qc8w = 14,
  fc_type_qd8_f32_qc8w = 15,
  fc_type_qs8_qs8_qc8w = 16,
  fc_type_qs8_qs8_qs8 = 17,
  fc_type_qu8_qu8_qu8 = 18,
  fc_type_qp8_f32_qb4w = 19,
  fc_type_pf32_f32_f32 = 20,
  fc_type_f32_f16_f32 = 21,
  fc_type_qdu8_f16_qc8w = 22,
  fc_type_qdu8_f32_qc8w = 23,
  fc_type_qdu8_f32_qc4w = 24,
  fc_type_qdu8_f32_qb4w = 26,
  fc_type_qdu8_f16_qc4w = 27,
  fc_type_qp8_f32_qc8w = 28,
  fc_type_pf16_f16_f16 = 29,
  fc_type_pqs8_qs8_qc8w = 30,
  fc_type_bf16_bf16_f32 = 31,
  fc_type_pf16_f16_f16_dynamic = 32,
  fc_type_pf32_f32_f32_dynamic = 33,
};

enum fully_connected_op_type get_fully_connected_op_type(
    const struct xnn_value* input_value, const struct xnn_value* filter_value,
    const struct xnn_value* bias_value, const struct xnn_value* output_value) {
  bool has_non_static_weights =
      (filter_value->allocation_type != xnn_allocation_type_static);
  if (bias_value) {
    has_non_static_weights |=
        (bias_value->allocation_type != xnn_allocation_type_static);
  }

  const enum xnn_datatype input_datatype = input_value->datatype;
  const enum xnn_datatype filter_datatype = filter_value->datatype;
  const enum xnn_datatype output_datatype = output_value->datatype;
  switch (output_datatype) {
    case xnn_datatype_fp16:
      switch (filter_datatype) {
        case xnn_datatype_fp16:
          if (has_non_static_weights) {
            switch (input_datatype) {
              case xnn_datatype_fp16:
                return fc_type_f16_f16_f16_dynamic;
              case xnn_datatype_pfp16:
                return fc_type_pf16_f16_f16_dynamic;
              default:
                XNN_UNREACHABLE;
            }
          } else {
            switch (input_datatype) {
              case xnn_datatype_fp16:
                return fc_type_f16_f16_f16;
              case xnn_datatype_pfp16:
                return fc_type_pf16_f16_f16;
              default:
                XNN_UNREACHABLE;
            }
          }
        case xnn_datatype_fp32:
          if (has_non_static_weights) {
            return fc_type_f16_f32_f16_dynamic;
          } else {
            return fc_type_f16_f32_f16;
          }
        case xnn_datatype_qcint4:
          switch (input_datatype) {
            case xnn_datatype_qdint8:
              return fc_type_qd8_f16_qc4w;
            case xnn_datatype_qduint8:
              return fc_type_qdu8_f16_qc4w;
            default:
              XNN_UNREACHABLE;
          }
          break;
        case xnn_datatype_qbint4:
          return fc_type_qd8_f16_qb4w;
        case xnn_datatype_qcint8:
          switch (input_datatype) {
            case xnn_datatype_qdint8:
              return fc_type_qd8_f16_qc8w;
            case xnn_datatype_qduint8:
              return fc_type_qdu8_f16_qc8w;
            default:
              XNN_UNREACHABLE;
          }
          break;
        default:
          XNN_UNREACHABLE;
      }
      break;
    case xnn_datatype_fp32:
      switch (filter_datatype) {
        case xnn_datatype_bf16:
          switch (input_datatype) {
            case xnn_datatype_bf16:
              return fc_type_bf16_bf16_f32;
            default:
              XNN_UNREACHABLE;
          }
        case xnn_datatype_fp16:
          switch (input_datatype) {
            case xnn_datatype_fp32:
              return fc_type_f32_f16_f32;
            default:
              XNN_UNREACHABLE;
          }
        case xnn_datatype_fp32:
          if (has_non_static_weights) {
            switch (input_datatype) {
              case xnn_datatype_fp32:
                return fc_type_f32_f32_f32_dynamic;
              case xnn_datatype_pfp32:
                return fc_type_pf32_f32_f32_dynamic;
              default:
                XNN_UNREACHABLE;
            }
          } else {
            switch (input_datatype) {
              case xnn_datatype_fp32:
                return fc_type_f32_f32_f32;
              case xnn_datatype_pfp32:
                return fc_type_pf32_f32_f32;
              default:
                XNN_UNREACHABLE;
            }
          }
        case xnn_datatype_qbint4:
          switch (input_datatype) {
            case xnn_datatype_qdint8:
              return fc_type_qd8_f32_qb4w;
            case xnn_datatype_qduint8:
              return fc_type_qdu8_f32_qb4w;
            case xnn_datatype_qpint8:
              return fc_type_qp8_f32_qb4w;
            default:
              XNN_UNREACHABLE;
          }
        case xnn_datatype_qcint4:
          switch (input_datatype) {
            case xnn_datatype_fp32:
              return fc_type_f32_f32_qc4w;
            case xnn_datatype_qdint8:
              return fc_type_qd8_f32_qc4w;
            case xnn_datatype_qduint8:
              return fc_type_qdu8_f32_qc4w;
            case xnn_datatype_qpint8:
              return fc_type_qp8_f32_qc4w;
            default:
              XNN_UNREACHABLE;
          }
          break;
        case xnn_datatype_qcint8:
          switch (input_datatype) {
            case xnn_datatype_fp32:
              return fc_type_f32_f32_qc8w;
            case xnn_datatype_qdint8:
              return fc_type_qd8_f32_qc8w;
            case xnn_datatype_qpint8:
              return fc_type_qp8_f32_qc8w;
            case xnn_datatype_qduint8:
              return fc_type_qdu8_f32_qc8w;
            default:
              XNN_UNREACHABLE;
          }
          break;
        default:
          XNN_UNREACHABLE;
      }
      break;
    case xnn_datatype_qint8:
      switch (filter_datatype) {
        case xnn_datatype_qcint8:
          switch (input_datatype) {
            case xnn_datatype_qint8:
              return fc_type_qs8_qs8_qc8w;
            case xnn_datatype_pqint8:
              return fc_type_pqs8_qs8_qc8w;
            default:
              XNN_UNREACHABLE;
          }
          break;
        case xnn_datatype_qint8:
          return fc_type_qs8_qs8_qs8;
        default:
          XNN_UNREACHABLE;
      }
      break;
    case xnn_datatype_quint8:
      return fc_type_qu8_qu8_qu8;
    default:
      XNN_UNREACHABLE;
  }
}

static enum xnn_status create_fully_connected_operator(
    const struct xnn_node* node, const struct xnn_value* values,
    size_t num_values, struct xnn_operator_data* opdata,
    struct xnn_code_cache* code_cache, xnn_weights_cache_t weights_cache) {
  assert(node->num_inputs >= 2);
  assert(node->num_inputs <= 3);

  const uint32_t input_id = node->inputs[0];
  assert(input_id != XNN_INVALID_VALUE_ID);
  assert(input_id < num_values);
  const struct xnn_value* input_value = &values[input_id];

  const uint32_t filter_id = node->inputs[1];
  assert(filter_id != XNN_INVALID_VALUE_ID);
  assert(filter_id < num_values);
  const struct xnn_value* filter_value = &values[filter_id];

  assert(node->num_outputs == 1);
  const uint32_t output_id = node->outputs[0];
  assert(output_id != XNN_INVALID_VALUE_ID);
  assert(output_id < num_values);
  const struct xnn_value* output_value = &values[output_id];

  size_t output_channels, input_channels;
  if (node->flags & XNN_FLAG_TRANSPOSE_WEIGHTS) {
    input_channels = filter_value->shape.dim[0];
    output_channels = filter_value->shape.dim[1];
  } else {
    output_channels = filter_value->shape.dim[0];
    // Note that for convolutions, the filter shape can be `[H, 1, 1, W]`, so we
    // need to look at the last dimension of the filter.
    input_channels = filter_value->shape.dim[filter_value->shape.num_dims - 1];
  }

  const void* kernel_data = filter_value->fp32_data != NULL
                                ? filter_value->fp32_data
                                : filter_value->data;
  bool has_non_static_weights = (kernel_data == NULL);

  const void* bias_data = NULL;
  const struct xnn_value* bias_value = NULL;
  uint32_t bias_id = XNN_INVALID_VALUE_ID;
  if (node->num_inputs > 2) {
    bias_id = node->inputs[2];
    assert(bias_id != XNN_INVALID_VALUE_ID);
    assert(bias_id < num_values);
    bias_value = &values[bias_id];

    bias_data = bias_value->fp32_data != NULL ? bias_value->fp32_data
                                              : bias_value->data;
    has_non_static_weights |= (bias_data == NULL);
  }

  if (output_value->layout == xnn_layout_type_nchw) {
    return create_nchw_convolution(
        /*input_padding_top=*/0,
        /*input_padding_right=*/0,
        /*input_padding_bottom=*/0,
        /*input_padding_left=*/0,
        /*kernel_height=*/1,
        /*kernel_width=*/1,
        /*subsampling_height=*/1,
        /*subsampling_width=*/1,
        /*dilation_height=*/1,
        /*dilation_width=*/1,
        /*groups=*/1,
        /*group_input_channels=*/input_channels,
        /*group_output_channels=*/output_channels, node->activation.output_min,
        node->activation.output_max, node->flags, input_id, filter_id, bias_id,
        output_id, values, kernel_data, bias_data, code_cache, weights_cache,
        opdata);
  }
  enum xnn_status status;
  enum fully_connected_op_type op_type = get_fully_connected_op_type(
      input_value, filter_value, bias_value, output_value);
  xnn_operator_t* fully_connected_op_ptr = &opdata->operator_objects[0];
  switch (op_type) {
    case fc_type_bf16_bf16_f32:
      status = xnn_create_fully_connected_nc_bf16_f32(
          input_channels, output_channels,
          /*input_stride=*/input_channels,
          /*output_stride=*/output_channels, kernel_data, bias_data,
          node->activation.output_min, node->activation.output_max, node->flags,
          code_cache, weights_cache, fully_connected_op_ptr);
      break;
    case fc_type_f16_f16_f16_dynamic:
      status = xnn_create_dynamic_fully_connected_nc_f16(
          node->activation.output_min, node->activation.output_max,
          /*flags=*/node->flags, fully_connected_op_ptr);
      break;
    case fc_type_f16_f16_f16:
      status = xnn_create_fully_connected_nc_f16(
          input_channels, output_channels,
          /*input_stride=*/input_channels,
          /*output_stride=*/output_channels, kernel_data, bias_data,
          node->activation.output_min, node->activation.output_max, node->flags,
          code_cache, weights_cache, fully_connected_op_ptr);
      break;
    case fc_type_f16_f32_f16_dynamic:
      status = xnn_create_dynamic_fully_connected_nc_f16(
          node->activation.output_min, node->activation.output_max,
          node->flags | XNN_FLAG_FP32_STATIC_WEIGHTS, fully_connected_op_ptr);
      break;
    case fc_type_f16_f32_f16:
      status = xnn_create_fully_connected_nc_f16(
          input_channels, output_channels,
          /*input_stride=*/input_channels,
          /*output_stride=*/output_channels, kernel_data, bias_data,
          node->activation.output_min, node->activation.output_max,
          node->flags | XNN_FLAG_FP32_STATIC_WEIGHTS, code_cache, weights_cache,
          fully_connected_op_ptr);
      break;
    case fc_type_qd8_f16_qc4w:
      status = xnn_create_fully_connected_nc_qd8_f16_qc4w(
          input_channels, output_channels,
          /*input_stride=*/input_channels,
          /*output_stride=*/output_channels,
          /*kernel_zero_point=*/filter_value->quantization.zero_point,
          filter_value->quantization.channelwise_scale, kernel_data, bias_data,
          node->activation.output_min, node->activation.output_max, node->flags,
          code_cache, weights_cache, fully_connected_op_ptr);
      break;
    case fc_type_qdu8_f16_qc4w:
      status = xnn_create_fully_connected_nc_qdu8_f16_qc4w(
          input_channels, output_channels,
          /*input_stride=*/input_channels,
          /*output_stride=*/output_channels,
          /*kernel_zero_point=*/filter_value->quantization.zero_point,
          filter_value->quantization.channelwise_scale, kernel_data, bias_data,
          node->activation.output_min, node->activation.output_max, node->flags,
          code_cache, weights_cache, fully_connected_op_ptr);
      break;
    case fc_type_qd8_f16_qb4w:
      status = xnn_create_fully_connected_nc_qd8_f16_qb4w(
          input_channels, output_channels,
          /*input_stride=*/input_channels,
          /*output_stride=*/output_channels,
          /*block_size=*/filter_value->quantization.block_size,
          /*kernel_zero_point=*/filter_value->quantization.zero_point,
          (const uint16_t*)filter_value->quantization.blockwise_scale,
          kernel_data, bias_data, node->activation.output_min,
          node->activation.output_max, node->flags, code_cache, weights_cache,
          fully_connected_op_ptr);
      break;
    case fc_type_qd8_f16_qc8w:
      status = xnn_create_fully_connected_nc_qd8_f16_qc8w(
          input_channels, output_channels,
          /*input_stride=*/input_channels,
          /*output_stride=*/output_channels,
          filter_value->quantization.channelwise_scale, kernel_data, bias_data,
          node->activation.output_min, node->activation.output_max, node->flags,
          code_cache, weights_cache, fully_connected_op_ptr);
      break;
    case fc_type_qdu8_f16_qc8w:
      status = xnn_create_fully_connected_nc_qdu8_f16_qc8w(
          input_channels, output_channels,
          /*input_stride=*/input_channels,
          /*output_stride=*/output_channels,
          filter_value->quantization.channelwise_scale, kernel_data, bias_data,
          node->activation.output_min, node->activation.output_max, node->flags,
          code_cache, weights_cache, fully_connected_op_ptr);
      break;
    case fc_type_f32_f32_f32_dynamic:
      status = xnn_create_dynamic_fully_connected_nc_f32(
          node->activation.output_min, node->activation.output_max,
          /*flags=*/node->flags, fully_connected_op_ptr);
      break;
    case fc_type_f32_f32_f32:
      status = xnn_create_fully_connected_nc_f32(
          input_channels, output_channels,
          /*input_stride=*/input_channels,
          /*output_stride=*/output_channels, kernel_data, bias_data,
          node->activation.output_min, node->activation.output_max,
          /*flags=*/node->flags, code_cache, weights_cache,
          fully_connected_op_ptr);
      break;
    case fc_type_pf16_f16_f16:
      status = xnn_create_fully_connected_nc_pf16(
          input_channels, output_channels,
          /*input_stride=*/input_channels,
          /*output_stride=*/output_channels, kernel_data, bias_data,
          node->activation.output_min, node->activation.output_max,
          /*flags=*/node->flags, code_cache, weights_cache,
          fully_connected_op_ptr);
      break;
    case fc_type_pf16_f16_f16_dynamic:
      status = xnn_create_dynamic_fully_connected_nc_pf16(
          node->activation.output_min, node->activation.output_max,
          /*flags=*/node->flags, fully_connected_op_ptr);
      break;
    case fc_type_pf32_f32_f32:
      status = xnn_create_fully_connected_nc_pf32(
          input_channels, output_channels,
          /*input_stride=*/input_channels,
          /*output_stride=*/output_channels, kernel_data, bias_data,
          node->activation.output_min, node->activation.output_max,
          /*flags=*/node->flags, code_cache, weights_cache,
          fully_connected_op_ptr);
      break;
    case fc_type_pf32_f32_f32_dynamic:
      status = xnn_create_dynamic_fully_connected_nc_pf32(
          node->activation.output_min, node->activation.output_max,
          /*flags=*/node->flags, fully_connected_op_ptr);
      break;
    case fc_type_qd8_f32_qb4w:
      status = xnn_create_fully_connected_nc_qd8_f32_qb4w(
          input_channels, output_channels,
          /*input_stride=*/input_channels,
          /*output_stride=*/output_channels,
          /*block_size=*/filter_value->quantization.block_size,
          /*kernel_zero_point=*/filter_value->quantization.zero_point,
          (const uint16_t*)filter_value->quantization.blockwise_scale,
          kernel_data, bias_data, node->activation.output_min,
          node->activation.output_max, node->flags, code_cache, weights_cache,
          fully_connected_op_ptr);
      break;
    case fc_type_qdu8_f32_qb4w:
      status = xnn_create_fully_connected_nc_qdu8_f32_qb4w(
          input_channels, output_channels,
          /*input_stride=*/input_channels,
          /*output_stride=*/output_channels,
          /*block_size=*/filter_value->quantization.block_size,
          /*kernel_zero_point=*/filter_value->quantization.zero_point,
          (const uint16_t*)filter_value->quantization.blockwise_scale,
          kernel_data, bias_data, node->activation.output_min,
          node->activation.output_max, node->flags, code_cache, weights_cache,
          fully_connected_op_ptr);
      break;
    case fc_type_f32_f16_f32: {
      uint32_t flags = node->flags;
      if (bias_value != NULL && bias_value->datatype == xnn_datatype_fp32) {
        flags |= XNN_FLAG_FP32_STATIC_BIASES;
      }
      status = xnn_create_fully_connected_nc_f32_f16(
          input_channels, output_channels,
          /*input_stride=*/input_channels,
          /*output_stride=*/output_channels, kernel_data, bias_data,
          node->activation.output_min, node->activation.output_max, flags,
          code_cache, weights_cache, fully_connected_op_ptr);
      break;
    }
    case fc_type_qp8_f32_qb4w:
      status = xnn_create_fully_connected_nc_qp8_f32_qb4w(
          input_channels, output_channels,
          /*input_stride=*/input_channels,
          /*output_stride=*/output_channels,
          /*block_size=*/filter_value->quantization.block_size,
          /*kernel_zero_point=*/filter_value->quantization.zero_point,
          (const uint16_t*)filter_value->quantization.blockwise_scale,
          kernel_data, bias_data, node->activation.output_min,
          node->activation.output_max, node->flags, code_cache, weights_cache,
          fully_connected_op_ptr);
      break;
    case fc_type_f32_f32_qc4w:
      status = xnn_create_fully_connected_nc_f32_qc4w(
          input_channels, output_channels,
          /*input_stride=*/input_channels,
          /*output_stride=*/output_channels,
          filter_value->quantization.zero_point,
          filter_value->quantization.channelwise_scale, kernel_data, bias_data,
          node->activation.output_min, node->activation.output_max,
          /*flags=*/node->flags, code_cache, weights_cache,
          fully_connected_op_ptr);
      break;
    case fc_type_qd8_f32_qc4w:
      status = xnn_create_fully_connected_nc_qd8_f32_qc4w(
          input_channels, output_channels,
          /*input_stride=*/input_channels,
          /*output_stride=*/output_channels,
          /*kernel_zero_point=*/filter_value->quantization.zero_point,
          filter_value->quantization.channelwise_scale, kernel_data, bias_data,
          node->activation.output_min, node->activation.output_max, node->flags,
          code_cache, weights_cache, fully_connected_op_ptr);
      break;
    case fc_type_qdu8_f32_qc4w:
      status = xnn_create_fully_connected_nc_qdu8_f32_qc4w(
          input_channels, output_channels,
          /*input_stride=*/input_channels,
          /*output_stride=*/output_channels,
          /*kernel_zero_point=*/filter_value->quantization.zero_point,
          filter_value->quantization.channelwise_scale, kernel_data, bias_data,
          node->activation.output_min, node->activation.output_max, node->flags,
          code_cache, weights_cache, fully_connected_op_ptr);
      break;
    case fc_type_qp8_f32_qc4w:
      status = xnn_create_fully_connected_nc_qp8_f32_qc4w(
          input_channels, output_channels,
          /*input_stride=*/input_channels,
          /*output_stride=*/output_channels,
          /*kernel_zero_point=*/filter_value->quantization.zero_point,
          filter_value->quantization.channelwise_scale, kernel_data, bias_data,
          node->activation.output_min, node->activation.output_max, node->flags,
          code_cache, weights_cache, fully_connected_op_ptr);
      break;
    case fc_type_qp8_f32_qc8w:
      status = xnn_create_fully_connected_nc_qp8_f32_qc8w(
          input_channels, output_channels,
          /*input_stride=*/input_channels,
          /*output_stride=*/output_channels,
          filter_value->quantization.channelwise_scale, kernel_data, bias_data,
          node->activation.output_min, node->activation.output_max, node->flags,
          code_cache, weights_cache, fully_connected_op_ptr);
      break;
    case fc_type_f32_f32_qc8w:
      status = xnn_create_fully_connected_nc_f32_qc8w(
          input_channels, output_channels,
          /*input_stride=*/input_channels,
          /*output_stride=*/output_channels,
          filter_value->quantization.channelwise_scale, kernel_data, bias_data,
          node->activation.output_min, node->activation.output_max,
          /*flags=*/node->flags, code_cache, weights_cache,
          fully_connected_op_ptr);
      break;
    case fc_type_qd8_f32_qc8w:
      status = xnn_create_fully_connected_nc_qd8_f32_qc8w(
          input_channels, output_channels,
          /*input_stride=*/input_channels,
          /*output_stride=*/output_channels,
          filter_value->quantization.channelwise_scale, kernel_data, bias_data,
          node->activation.output_min, node->activation.output_max, node->flags,
          code_cache, weights_cache, fully_connected_op_ptr);
      break;
    case fc_type_qdu8_f32_qc8w:
      status = xnn_create_fully_connected_nc_qdu8_f32_qc8w(
          input_channels, output_channels,
          /*input_stride=*/input_channels,
          /*output_stride=*/output_channels,
          filter_value->quantization.channelwise_scale, kernel_data, bias_data,
          node->activation.output_min, node->activation.output_max, node->flags,
          code_cache, weights_cache, fully_connected_op_ptr);
      break;
    case fc_type_qs8_qs8_qc8w: {
      assert(!has_non_static_weights);
      assert(kernel_data != NULL);
      assert(filter_value->datatype == xnn_datatype_qcint8);
      const float output_scale = output_value->quantization.scale;
      const int32_t output_zero_point = output_value->quantization.zero_point;
      const int8_t output_min = xnn_qs8_quantize(
          node->activation.output_min, output_scale, output_zero_point);
      const int8_t output_max = xnn_qs8_quantize(
          node->activation.output_max, output_scale, output_zero_point);
      status = xnn_create_fully_connected_nc_qs8_qc8w(
          input_channels, output_channels,
          /*input_stride=*/input_channels,
          /*output_stride=*/output_channels,
          (int8_t)input_value->quantization.zero_point,
          input_value->quantization.scale,
          filter_value->quantization.channelwise_scale, kernel_data, bias_data,
          (int8_t)output_zero_point, output_scale, output_min, output_max,
          /*flags=*/node->flags, code_cache, weights_cache,
          fully_connected_op_ptr);
      break;
    }
    case fc_type_pqs8_qs8_qc8w: {
      assert(!has_non_static_weights);
      assert(kernel_data != NULL);
      assert(filter_value->datatype == xnn_datatype_qcint8);
      const float output_scale = output_value->quantization.scale;
      const int32_t output_zero_point = output_value->quantization.zero_point;
      const int8_t output_min = xnn_qs8_quantize(
          node->activation.output_min, output_scale, output_zero_point);
      const int8_t output_max = xnn_qs8_quantize(
          node->activation.output_max, output_scale, output_zero_point);
      status = xnn_create_fully_connected_nc_pqs8_qc8w(
          input_channels, output_channels,
          /*input_stride=*/input_channels,
          /*output_stride=*/output_channels,
          (int8_t)input_value->quantization.zero_point,
          input_value->quantization.scale,
          filter_value->quantization.channelwise_scale, kernel_data, bias_data,
          (int8_t)output_zero_point, output_scale, output_min, output_max,
          /*flags=*/node->flags, code_cache, weights_cache,
          fully_connected_op_ptr);
      break;
    }
    case fc_type_qs8_qs8_qs8: {
      assert(!has_non_static_weights);
      assert(kernel_data != NULL);
      const float output_scale = output_value->quantization.scale;
      const int32_t output_zero_point = output_value->quantization.zero_point;
      const int8_t output_min = xnn_qs8_quantize(
          node->activation.output_min, output_scale, output_zero_point);
      const int8_t output_max = xnn_qs8_quantize(
          node->activation.output_max, output_scale, output_zero_point);
      status = xnn_create_fully_connected_nc_qs8(
          input_channels, output_channels,
          /*input_stride=*/input_channels,
          /*output_stride=*/output_channels,
          (int8_t)input_value->quantization.zero_point,
          input_value->quantization.scale, filter_value->quantization.scale,
          kernel_data, bias_data, (int8_t)output_zero_point, output_scale,
          output_min, output_max,
          /*flags=*/node->flags, code_cache, weights_cache,
          fully_connected_op_ptr);
    } break;
    case fc_type_qu8_qu8_qu8: {
      assert(!has_non_static_weights);
      assert(kernel_data != NULL);
      const float output_scale = output_value->quantization.scale;
      const int32_t output_zero_point = output_value->quantization.zero_point;
      const uint8_t output_min = xnn_qu8_quantize(
          node->activation.output_min, output_scale, output_zero_point);
      const uint8_t output_max = xnn_qu8_quantize(
          node->activation.output_max, output_scale, output_zero_point);
      status = xnn_create_fully_connected_nc_qu8(
          input_channels, output_channels,
          /*input_stride=*/input_channels,
          /*output_stride=*/output_channels,
          (uint8_t)input_value->quantization.zero_point,
          input_value->quantization.scale,
          (uint8_t)filter_value->quantization.zero_point,
          filter_value->quantization.scale, kernel_data, bias_data,
          (uint8_t)output_zero_point, output_scale, output_min, output_max,
          /*flags=*/node->flags, code_cache, weights_cache,
          fully_connected_op_ptr);
      break;
    }
    default:
      XNN_UNREACHABLE;
  }
  return status;
}

enum xnn_status resize_fully_connected_output_tensor(
    const struct xnn_operator_data* opdata, struct xnn_value* values,
    size_t num_values, size_t old_workspace_size, pthreadpool_t threadpool) {
  const uint32_t filter_id = opdata->inputs[1];
  const struct xnn_value* filter = &values[filter_id];

  const uint32_t output_id = opdata->outputs[0];
  struct xnn_value* output = (struct xnn_value*)&values[output_id];

  const uint32_t input_id = opdata->inputs[0];
  const struct xnn_value* input = &values[input_id];

  bool reshape_2d = opdata->flags & XNN_FLAG_TENSORFLOW_RESHAPE_2D;
  if (reshape_2d) {
    output->shape.num_dims = 2;
  } else {
    output->shape.num_dims = input->shape.num_dims;
  }
  // Infer output channels.
  const uint32_t filter_output_channel_index =
      (opdata->flags & XNN_FLAG_TRANSPOSE_WEIGHTS) ? 1 : 0;
  output->shape.dim[output->shape.num_dims - 1] =
      filter->shape.dim[filter_output_channel_index];

  if (reshape_2d) {
    const uint32_t filter_input_channel_index =
        (opdata->flags & XNN_FLAG_TRANSPOSE_WEIGHTS) ? 0 : 1;
    const size_t num_input_elements =
        xnn_shape_multiply_all_dims(&input->shape);
    // propogate the input shape to output.
    output->shape.dim[0] =
        num_input_elements / filter->shape.dim[filter_input_channel_index];
  } else {
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
    struct xnn_operator_data* opdata, struct xnn_value* values,
    size_t num_values, pthreadpool_t threadpool) {
  const uint32_t input_id = opdata->inputs[0];
  assert(input_id < num_values);
  struct xnn_value* input_value = &values[input_id];

  const uint32_t output_id = opdata->outputs[0];
  assert(output_id < num_values);
  struct xnn_value* output_value = &values[output_id];

  const uint32_t filter_id = opdata->inputs[1];
  assert(filter_id < num_values);
  struct xnn_value* filter_value = &values[filter_id];

  if (output_value->layout == xnn_layout_type_nchw) {
    return reshape_convolution_operator(opdata, values, num_values, threadpool);
  }
  const size_t num_input_elements =
      xnn_shape_multiply_all_dims(&input_value->shape);
  size_t output_channels, input_channels;
  if (opdata->flags & XNN_FLAG_TRANSPOSE_WEIGHTS) {
    input_channels = filter_value->shape.dim[0];
    output_channels = filter_value->shape.dim[1];
  } else {
    output_channels = filter_value->shape.dim[0];
    // Note that for convolutions, the filter shape can be `[H, 1, 1, W]`, so we
    // need to look at the last dimension of the filter.
    input_channels = filter_value->shape.dim[filter_value->shape.num_dims - 1];
  }

  const size_t batch_size = num_input_elements / input_channels;
  assert(batch_size * input_channels == num_input_elements);
  const size_t old_workspace_size = opdata->workspace_size;
  enum xnn_status status = xnn_status_invalid_state;

  xnn_operator_t fully_connected_op = opdata->operator_objects[0];
  switch (fully_connected_op->type) {
    case xnn_operator_type_fully_connected_nc_bf16_f32:
      status = xnn_reshape_fully_connected_nc_bf16_f32(fully_connected_op,
                                                  batch_size, threadpool);
      break;
    case xnn_operator_type_dynamic_fully_connected_nc_f16:
      status = xnn_reshape_dynamic_fully_connected_nc_f16(
          fully_connected_op, batch_size, input_channels, output_channels,
          input_channels, output_channels, &opdata->workspace_size,
          &opdata->workspace_alignment, threadpool);
      break;
    case xnn_operator_type_dynamic_fully_connected_nc_pf16:
      status = xnn_reshape_dynamic_fully_connected_nc_pf16(
          fully_connected_op, batch_size, input_channels, output_channels,
          input_channels, output_channels, &opdata->workspace_size,
          &opdata->workspace_alignment, threadpool);
      break;
    case xnn_operator_type_dynamic_fully_connected_nc_f32:
      status = xnn_reshape_dynamic_fully_connected_nc_f32(
          fully_connected_op, batch_size, input_channels, output_channels,
          input_channels, output_channels, &opdata->workspace_size,
          &opdata->workspace_alignment, threadpool);
      break;
    case xnn_operator_type_dynamic_fully_connected_nc_pf32:
      status = xnn_reshape_dynamic_fully_connected_nc_pf32(
          fully_connected_op, batch_size, input_channels, output_channels,
          input_channels, output_channels, &opdata->workspace_size,
          &opdata->workspace_alignment, threadpool);
      break;
    case xnn_operator_type_fully_connected_nc_f16:
      status = xnn_reshape_fully_connected_nc_f16(fully_connected_op,
                                                  batch_size, threadpool);
      break;
    case xnn_operator_type_fully_connected_nc_pf16:
      status = xnn_reshape_fully_connected_nc_pf16(fully_connected_op,
                                                   batch_size, threadpool);
      break;
    case xnn_operator_type_fully_connected_nc_f32:
      status = xnn_reshape_fully_connected_nc_f32(fully_connected_op,
                                                  batch_size, threadpool);
      break;
    case xnn_operator_type_fully_connected_nc_pf32:
      status = xnn_reshape_fully_connected_nc_pf32(fully_connected_op,
                                                   batch_size, threadpool);
      break;
    case xnn_operator_type_fully_connected_nc_f32_qc4w:
      status = xnn_reshape_fully_connected_nc_f32_qc4w(fully_connected_op,
                                                       batch_size, threadpool);
      break;
    case xnn_operator_type_fully_connected_nc_f32_qc8w:
      status = xnn_reshape_fully_connected_nc_f32_qc8w(fully_connected_op,
                                                       batch_size, threadpool);
      break;
    case xnn_operator_type_fully_connected_nc_qd8_f32_qc4w:
      status = xnn_reshape_fully_connected_nc_qd8_f32_qc4w(
          fully_connected_op, batch_size, threadpool);
      break;
    case xnn_operator_type_fully_connected_nc_qdu8_f32_qc4w:
      status = xnn_reshape_fully_connected_nc_qdu8_f32_qc4w(
          fully_connected_op, batch_size, threadpool);
      break;
    case xnn_operator_type_fully_connected_nc_qd8_f16_qc4w:
      status = xnn_reshape_fully_connected_nc_qd8_f16_qc4w(
          fully_connected_op, batch_size, threadpool);
      break;
    case xnn_operator_type_fully_connected_nc_qdu8_f16_qc4w:
      status = xnn_reshape_fully_connected_nc_qdu8_f16_qc4w(
          fully_connected_op, batch_size, threadpool);
      break;
    case xnn_operator_type_fully_connected_nc_qd8_f16_qb4w:
      status = xnn_reshape_fully_connected_nc_qd8_f16_qb4w(
          fully_connected_op, batch_size, threadpool);
      break;
    case xnn_operator_type_fully_connected_nc_qd8_f32_qb4w:
      status = xnn_reshape_fully_connected_nc_qd8_f32_qb4w(
          fully_connected_op, batch_size, threadpool);
      break;
    case xnn_operator_type_fully_connected_nc_qdu8_f32_qb4w:
      status = xnn_reshape_fully_connected_nc_qdu8_f32_qb4w(
          fully_connected_op, batch_size, threadpool);
      break;
    case xnn_operator_type_fully_connected_nc_qd8_f16_qc8w:
      status = xnn_reshape_fully_connected_nc_qd8_f16_qc8w(
          fully_connected_op, batch_size, threadpool);
      break;
    case xnn_operator_type_fully_connected_nc_qdu8_f16_qc8w:
      status = xnn_reshape_fully_connected_nc_qdu8_f16_qc8w(
          fully_connected_op, batch_size, threadpool);
      break;
    case xnn_operator_type_fully_connected_nc_qd8_f32_qc8w:
      status = xnn_reshape_fully_connected_nc_qd8_f32_qc8w(
          fully_connected_op, batch_size, threadpool);
      break;
    case xnn_operator_type_fully_connected_nc_qdu8_f32_qc8w:
      status = xnn_reshape_fully_connected_nc_qdu8_f32_qc8w(
          fully_connected_op, batch_size, threadpool);
      break;
    case xnn_operator_type_fully_connected_nc_qp8_f32_qc4w:
      status = xnn_reshape_fully_connected_nc_qp8_f32_qc4w(
          fully_connected_op, batch_size, threadpool);
      break;
    case xnn_operator_type_fully_connected_nc_qp8_f32_qc8w:
      status = xnn_reshape_fully_connected_nc_qp8_f32_qc8w(
          fully_connected_op, batch_size, threadpool);
      break;
    case xnn_operator_type_fully_connected_nc_qp8_f32_qb4w:
      status = xnn_reshape_fully_connected_nc_qp8_f32_qb4w(
          fully_connected_op, batch_size, threadpool);
      break;
    case xnn_operator_type_fully_connected_nc_qs8:
      status = xnn_reshape_fully_connected_nc_qs8(fully_connected_op,
                                                  batch_size, threadpool);
      break;
    case xnn_operator_type_fully_connected_nc_qs8_qc8w:
      status = xnn_reshape_fully_connected_nc_qs8_qc8w(fully_connected_op,
                                                       batch_size, threadpool);
      break;
    case xnn_operator_type_fully_connected_nc_pqs8_qc8w:
      status = xnn_reshape_fully_connected_nc_pqs8_qc8w(fully_connected_op,
                                                        batch_size, threadpool);
      break;
    case xnn_operator_type_fully_connected_nc_qu8:
      status = xnn_reshape_fully_connected_nc_qu8(fully_connected_op,
                                                  batch_size, threadpool);
      break;
    default:
      XNN_UNREACHABLE;
  }

  if (status != xnn_status_success) {
    return status;
  }
  return resize_fully_connected_output_tensor(opdata, values, num_values,
                                              old_workspace_size, threadpool);
}

static enum xnn_status setup_fully_connected_operator(
    const struct xnn_operator_data* opdata, const struct xnn_value* values,
    size_t num_values, pthreadpool_t threadpool) {
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

  if (values[output_id].layout == xnn_layout_type_nchw) {
    return setup_convolution_operator(opdata, values, num_values, threadpool);
  }
  const struct xnn_value* input_value = values + input_id;
  const void* input_data = input_value->data;
  assert(input_data != NULL);

  const struct xnn_value* kernel_value = values + filter_id;
  bool has_dynamic_weights =
      kernel_value->allocation_type != xnn_allocation_type_static;
  const void* kernel_data =
      kernel_value->allocation_type == xnn_allocation_type_static
          ? NULL
          : kernel_value->data;

  const void* bias_data = NULL;
  if (opdata->num_inputs > 2) {
    assert(bias_id != XNN_INVALID_VALUE_ID);
    assert(bias_id < num_values);
    const struct xnn_value* bias_value = values + bias_id;
    has_dynamic_weights |=
        bias_value->allocation_type != xnn_allocation_type_static;
    if (has_dynamic_weights) {
      kernel_data = kernel_value->data;
      bias_data = bias_value->data;
    }
  }

  const struct xnn_value* output_value = values + output_id;
  void* output_data = output_value->data;
  assert(output_data != NULL);

  xnn_operator_t fully_connected_op = opdata->operator_objects[0];
  switch (fully_connected_op->type) {
    case xnn_operator_type_fully_connected_nc_bf16_f32:
      assert(kernel_data == NULL);
      assert(bias_data == NULL);
      return xnn_setup_fully_connected_nc_bf16_f32(fully_connected_op, input_data,
                                              output_data);
    case xnn_operator_type_dynamic_fully_connected_nc_f16:
      assert(kernel_data != NULL);
      return xnn_setup_dynamic_fully_connected_nc_f16(
          fully_connected_op, opdata->workspace, input_data, kernel_data,
          bias_data, output_data);
    case xnn_operator_type_dynamic_fully_connected_nc_pf16:
      assert(kernel_data != NULL);
      return xnn_setup_dynamic_fully_connected_nc_pf16(
          fully_connected_op, opdata->workspace, input_data, kernel_data,
          bias_data, output_data);
    case xnn_operator_type_dynamic_fully_connected_nc_f32:
      assert(kernel_data != NULL);
      return xnn_setup_dynamic_fully_connected_nc_f32(
          fully_connected_op, opdata->workspace, input_data, kernel_data,
          bias_data, output_data);
    case xnn_operator_type_dynamic_fully_connected_nc_pf32:
      assert(kernel_data != NULL);
      return xnn_setup_dynamic_fully_connected_nc_pf32(
          fully_connected_op, opdata->workspace, input_data, kernel_data,
          bias_data, output_data);
    case xnn_operator_type_fully_connected_nc_f16:
      assert(kernel_data == NULL);
      assert(bias_data == NULL);
      return xnn_setup_fully_connected_nc_f16(fully_connected_op, input_data,
                                              output_data);
    case xnn_operator_type_fully_connected_nc_pf16:
      assert(kernel_data == NULL);
      assert(bias_data == NULL);
      return xnn_setup_fully_connected_nc_pf16(fully_connected_op, input_data,
                                               output_data);
    case xnn_operator_type_fully_connected_nc_f32:
      assert(kernel_data == NULL);
      assert(bias_data == NULL);
      return xnn_setup_fully_connected_nc_f32(fully_connected_op, input_data,
                                              output_data);
    case xnn_operator_type_fully_connected_nc_pf32:
      assert(kernel_data == NULL);
      assert(bias_data == NULL);
      return xnn_setup_fully_connected_nc_pf32(fully_connected_op, input_data,
                                               output_data);
    case xnn_operator_type_fully_connected_nc_f32_qc4w:
      assert(kernel_data == NULL);
      assert(bias_data == NULL);
      return xnn_setup_fully_connected_nc_f32_qc4w(fully_connected_op,
                                                   input_data, output_data);
    case xnn_operator_type_fully_connected_nc_f32_qc8w:
      assert(kernel_data == NULL);
      assert(bias_data == NULL);
      return xnn_setup_fully_connected_nc_f32_qc8w(fully_connected_op,
                                                   input_data, output_data);
    case xnn_operator_type_fully_connected_nc_qd8_f32_qc4w: {
      const void* quantization_params =
          input_value->quantization.dynamic_params;
      assert(kernel_data == NULL);
      assert(bias_data == NULL);
      assert(quantization_params != NULL);
      return xnn_setup_fully_connected_nc_qd8_f32_qc4w(
          fully_connected_op, input_data, output_data, quantization_params);
    }
    case xnn_operator_type_fully_connected_nc_qdu8_f32_qc4w: {
      const void* quantization_params =
          input_value->quantization.dynamic_params;
      assert(kernel_data == NULL);
      assert(bias_data == NULL);
      assert(quantization_params != NULL);
      return xnn_setup_fully_connected_nc_qdu8_f32_qc4w(
          fully_connected_op, input_data, output_data, quantization_params);
    }
    case xnn_operator_type_fully_connected_nc_qd8_f16_qc4w: {
      const void* quantization_params =
          input_value->quantization.dynamic_params;
      assert(kernel_data == NULL);
      assert(bias_data == NULL);
      assert(quantization_params != NULL);
      return xnn_setup_fully_connected_nc_qd8_f16_qc4w(
          fully_connected_op, input_data, output_data, quantization_params);
    }
    case xnn_operator_type_fully_connected_nc_qdu8_f16_qc4w: {
      const void* quantization_params =
          input_value->quantization.dynamic_params;
      assert(kernel_data == NULL);
      assert(bias_data == NULL);
      assert(quantization_params != NULL);
      return xnn_setup_fully_connected_nc_qdu8_f16_qc4w(
          fully_connected_op, input_data, output_data, quantization_params);
    }
    case xnn_operator_type_fully_connected_nc_qd8_f32_qb4w: {
      const void* quantization_params =
          input_value->quantization.dynamic_params;
      assert(kernel_data == NULL);
      assert(bias_data == NULL);
      assert(quantization_params != NULL);
      return xnn_setup_fully_connected_nc_qd8_f32_qb4w(
          fully_connected_op, input_data, output_data, quantization_params);
    }
    case xnn_operator_type_fully_connected_nc_qdu8_f32_qb4w: {
      const void* quantization_params =
          input_value->quantization.dynamic_params;
      assert(kernel_data == NULL);
      assert(bias_data == NULL);
      assert(quantization_params != NULL);
      return xnn_setup_fully_connected_nc_qdu8_f32_qb4w(
          fully_connected_op, input_data, output_data, quantization_params);
    }
    case xnn_operator_type_fully_connected_nc_qd8_f16_qb4w: {
      const void* quantization_params =
          input_value->quantization.dynamic_params;
      assert(kernel_data == NULL);
      assert(bias_data == NULL);
      assert(quantization_params != NULL);
      return xnn_setup_fully_connected_nc_qd8_f16_qb4w(
          fully_connected_op, input_data, output_data, quantization_params);
    }
    case xnn_operator_type_fully_connected_nc_qd8_f16_qc8w: {
      const void* quantization_params =
          input_value->quantization.dynamic_params;
      assert(kernel_data == NULL);
      assert(bias_data == NULL);
      assert(quantization_params != NULL);
      return xnn_setup_fully_connected_nc_qd8_f16_qc8w(
          fully_connected_op, input_data, output_data, quantization_params);
    }
    case xnn_operator_type_fully_connected_nc_qdu8_f16_qc8w: {
      const void* quantization_params =
          input_value->quantization.dynamic_params;
      assert(kernel_data == NULL);
      assert(bias_data == NULL);
      assert(quantization_params != NULL);
      return xnn_setup_fully_connected_nc_qdu8_f16_qc8w(
          fully_connected_op, input_data, output_data, quantization_params);
    }
    case xnn_operator_type_fully_connected_nc_qd8_f32_qc8w: {
      const void* quantization_params =
          input_value->quantization.dynamic_params;
      assert(kernel_data == NULL);
      assert(bias_data == NULL);
      assert(quantization_params != NULL);
      return xnn_setup_fully_connected_nc_qd8_f32_qc8w(
          fully_connected_op, input_data, output_data, quantization_params);
    }
    case xnn_operator_type_fully_connected_nc_qdu8_f32_qc8w: {
      const void* quantization_params =
          input_value->quantization.dynamic_params;
      assert(kernel_data == NULL);
      assert(bias_data == NULL);
      assert(quantization_params != NULL);
      return xnn_setup_fully_connected_nc_qdu8_f32_qc8w(
          fully_connected_op, input_data, output_data, quantization_params);
    }
    case xnn_operator_type_fully_connected_nc_qp8_f32_qc4w: {
      assert(kernel_data == NULL);
      assert(bias_data == NULL);
      return xnn_setup_fully_connected_nc_qp8_f32_qc4w(fully_connected_op,
                                                       input_data, output_data);
    }
    case xnn_operator_type_fully_connected_nc_qp8_f32_qc8w: {
      assert(kernel_data == NULL);
      assert(bias_data == NULL);
      return xnn_setup_fully_connected_nc_qp8_f32_qc8w(fully_connected_op,
                                                       input_data, output_data);
    }
    case xnn_operator_type_fully_connected_nc_qp8_f32_qb4w: {
      assert(kernel_data == NULL);
      assert(bias_data == NULL);
      return xnn_setup_fully_connected_nc_qp8_f32_qb4w(fully_connected_op,
                                                       input_data, output_data);
    }
    case xnn_operator_type_fully_connected_nc_qs8:
      assert(kernel_data == NULL);
      assert(bias_data == NULL);
      return xnn_setup_fully_connected_nc_qs8(fully_connected_op, input_data,
                                              output_data);
    case xnn_operator_type_fully_connected_nc_qs8_qc8w:
      assert(kernel_data == NULL);
      assert(bias_data == NULL);
      return xnn_setup_fully_connected_nc_qs8_qc8w(fully_connected_op,
                                                   input_data, output_data);
    case xnn_operator_type_fully_connected_nc_pqs8_qc8w:
      assert(kernel_data == NULL);
      assert(bias_data == NULL);
      return xnn_setup_fully_connected_nc_pqs8_qc8w(fully_connected_op,
                                                    input_data, output_data);
    case xnn_operator_type_fully_connected_nc_qu8:
      assert(kernel_data == NULL);
      assert(bias_data == NULL);
      return xnn_setup_fully_connected_nc_qu8(fully_connected_op, input_data,
                                              output_data);
    default:
      XNN_UNREACHABLE;
  }
}

static inline bool validate_datatypes_with_bias(
    enum xnn_datatype input_datatype, enum xnn_datatype kernel_datatype,
    enum xnn_datatype bias_datatype, enum xnn_datatype output_datatype) {
  switch (kernel_datatype) {
    case xnn_datatype_bf16:
      if (input_datatype == xnn_datatype_bf16 &&
          bias_datatype == xnn_datatype_fp32 &&
          output_datatype == xnn_datatype_fp32) {
        return true;
      } else {
        return false;
      }
    case xnn_datatype_fp32:
      if (input_datatype == xnn_datatype_fp32 &&
          bias_datatype == xnn_datatype_fp32 &&
          output_datatype == xnn_datatype_fp32) {
        return true;
      } else if (input_datatype == xnn_datatype_fp16 &&
                 bias_datatype == xnn_datatype_fp32 &&
                 output_datatype == xnn_datatype_fp16) {
        // Flag: XNN_FLAG_FP32_STATIC_WEIGHTS
        return true;
      }
      break;
    case xnn_datatype_fp16:
      if (input_datatype == xnn_datatype_fp16 &&
          bias_datatype == xnn_datatype_fp16 &&
          output_datatype == xnn_datatype_fp16) {
        return true;
      } else if (input_datatype == xnn_datatype_fp32 &&
                 bias_datatype == xnn_datatype_fp16 &&
                 output_datatype == xnn_datatype_fp32) {
        return true;
      } else if (input_datatype == xnn_datatype_fp32 &&
                 bias_datatype == xnn_datatype_fp32 &&
                 output_datatype == xnn_datatype_fp32) {
        // Flag: XNN_FLAG_FP32_STATIC_BIASES
        return true;
      }
      break;
    case xnn_datatype_qcint4:
      if (input_datatype == xnn_datatype_fp32 &&
          bias_datatype == xnn_datatype_fp32 &&
          output_datatype == xnn_datatype_fp32) {
        return true;
      } else if (input_datatype == xnn_datatype_qdint8 &&
                 bias_datatype == xnn_datatype_fp32 &&
                 output_datatype == xnn_datatype_fp32) {
        return true;
      } else if (input_datatype == xnn_datatype_qpint8 &&
                 bias_datatype == xnn_datatype_fp32 &&
                 output_datatype == xnn_datatype_fp32) {
        return true;
      } else if (input_datatype == xnn_datatype_qdint8 &&
                 bias_datatype == xnn_datatype_fp32 &&
                 output_datatype == xnn_datatype_fp16) {
        return true;
      }
      break;
    case xnn_datatype_qbint4:
      if (input_datatype == xnn_datatype_qdint8 &&
          bias_datatype == xnn_datatype_fp32 &&
          output_datatype == xnn_datatype_fp32) {
        return true;
      } else if (input_datatype == xnn_datatype_qdint8 &&
                 bias_datatype == xnn_datatype_fp32 &&
                 output_datatype == xnn_datatype_fp16) {
        return true;
      } else if (input_datatype == xnn_datatype_qpint8 &&
                 bias_datatype == xnn_datatype_fp32 &&
                 output_datatype == xnn_datatype_fp32) {
        return true;
      }
      break;
    case xnn_datatype_qcint8:
      if (input_datatype == xnn_datatype_fp32 &&
          bias_datatype == xnn_datatype_fp32 &&
          output_datatype == xnn_datatype_fp32) {
        return true;
      } else if (input_datatype == xnn_datatype_qdint8 &&
                 bias_datatype == xnn_datatype_fp32 &&
                 output_datatype == xnn_datatype_fp32) {
        return true;
      } else if (input_datatype == xnn_datatype_qpint8 &&
                 bias_datatype == xnn_datatype_fp32 &&
                 output_datatype == xnn_datatype_fp32) {
        return true;
      } else if (input_datatype == xnn_datatype_qdint8 &&
                 bias_datatype == xnn_datatype_fp32 &&
                 output_datatype == xnn_datatype_fp16) {
        return true;
      } else if (input_datatype == xnn_datatype_qint8 &&
                 bias_datatype == xnn_datatype_qcint32 &&
                 output_datatype == xnn_datatype_qint8) {
        return true;
      }
      break;
    case xnn_datatype_qint8:
      if (input_datatype == xnn_datatype_qint8 &&
          bias_datatype == xnn_datatype_qint32 &&
          output_datatype == xnn_datatype_qint8) {
        return true;
      }
      break;
    case xnn_datatype_quint8:
      if (input_datatype == xnn_datatype_quint8 &&
          bias_datatype == xnn_datatype_qint32 &&
          output_datatype == xnn_datatype_quint8) {
        return true;
      }
      break;
    default:
      XNN_UNREACHABLE;
  }
  return false;
}

static inline bool validate_datatypes_without_bias(
    enum xnn_datatype input_datatype, enum xnn_datatype kernel_datatype,
    enum xnn_datatype output_datatype) {
  switch (kernel_datatype) {
    case xnn_datatype_bf16:
      if (input_datatype == xnn_datatype_bf16 &&
          output_datatype == xnn_datatype_fp32) {
        return true;
      } else {
        return false;
      }
    case xnn_datatype_fp32:
      if (input_datatype == xnn_datatype_fp32 &&
          output_datatype == xnn_datatype_fp32) {
        return true;
      } else if (input_datatype == xnn_datatype_fp16 &&
                 output_datatype == xnn_datatype_fp16) {
        // Flag: XNN_FLAG_FP32_STATIC_WEIGHTS
        return true;
      }
      break;
    case xnn_datatype_fp16:
      if (input_datatype == xnn_datatype_fp16 &&
          output_datatype == xnn_datatype_fp16) {
        return true;
      } else if (input_datatype == xnn_datatype_fp32 &&
                 output_datatype == xnn_datatype_fp32) {
        return true;
      }
      break;
    case xnn_datatype_qcint4:
      if (input_datatype == xnn_datatype_fp32 &&
          output_datatype == xnn_datatype_fp32) {
        return true;
      } else if (input_datatype == xnn_datatype_qdint8 &&
                 output_datatype == xnn_datatype_fp32) {
        return true;
      } else if (input_datatype == xnn_datatype_qpint8 &&
                 output_datatype == xnn_datatype_fp32) {
        return true;
      } else if (input_datatype == xnn_datatype_qdint8 &&
                 output_datatype == xnn_datatype_fp16) {
        return true;
      }
      break;
    case xnn_datatype_qbint4:
      if (input_datatype == xnn_datatype_qdint8 &&
          output_datatype == xnn_datatype_fp32) {
        return true;
      } else if (input_datatype == xnn_datatype_qdint8 &&
                 output_datatype == xnn_datatype_fp16) {
        return true;
      } else if (input_datatype == xnn_datatype_qpint8 &&
                 output_datatype == xnn_datatype_fp32) {
        return true;
      }
      break;
    case xnn_datatype_qcint8:
      if (input_datatype == xnn_datatype_fp32 &&
          output_datatype == xnn_datatype_fp32) {
        return true;
      } else if (input_datatype == xnn_datatype_qdint8 &&
                 output_datatype == xnn_datatype_fp32) {
        return true;
      } else if (input_datatype == xnn_datatype_qpint8 &&
                 output_datatype == xnn_datatype_fp32) {
        return true;
      } else if (input_datatype == xnn_datatype_qdint8 &&
                 output_datatype == xnn_datatype_fp16) {
        return true;
      } else if (input_datatype == xnn_datatype_qint8 &&
                 output_datatype == xnn_datatype_qint8) {
        return true;
      }
      break;
    case xnn_datatype_qint8:
      if (input_datatype == xnn_datatype_qint8 &&
          output_datatype == xnn_datatype_qint8) {
        return true;
      }
      break;
    case xnn_datatype_quint8:
      if (input_datatype == xnn_datatype_quint8 &&
          output_datatype == xnn_datatype_quint8) {
        return true;
      }
      break;
    default:
      XNN_UNREACHABLE;
  }
  return false;
}

static bool datatype_is_packable(enum xnn_datatype datatype) {
  switch (datatype) {
    case xnn_datatype_fp16:
    case xnn_datatype_fp32:
    case xnn_datatype_qint8:
      return true;
    default:
      return false;
  }
}

enum xnn_status xnn_define_fully_connected(xnn_subgraph_t subgraph,
                                           float output_min, float output_max,
                                           uint32_t input_id,
                                           uint32_t filter_id, uint32_t bias_id,
                                           uint32_t output_id, uint32_t flags) {
  enum xnn_status status;
  if ((status = xnn_subgraph_check_xnnpack_initialized(
           xnn_node_type_fully_connected)) != xnn_status_success) {
    return status;
  }

  status = xnn_subgraph_check_output_min_max(xnn_node_type_fully_connected,
                                             output_min, output_max);
  if (status != xnn_status_success) {
    return status;
  }

  if ((status = xnn_subgraph_check_input_node_id(
           xnn_node_type_fully_connected, input_id, subgraph->num_values)) !=
      xnn_status_success) {
    return status;
  }

  const struct xnn_value* input_value = &subgraph->values[input_id];
  status = xnn_subgraph_check_input_type_dense(xnn_node_type_fully_connected,
                                               input_id, input_value);
  if (status != xnn_status_success) {
    return status;
  }

  switch (input_value->datatype) {
    case xnn_datatype_bf16:
    case xnn_datatype_fp16:
    case xnn_datatype_fp32:
    case xnn_datatype_qint8:
    case xnn_datatype_quint8:
    case xnn_datatype_qpint8:
      break;
    case xnn_datatype_qdint8:
      if (input_value->quantization.num_nonbatch_dims >
          input_value->shape.num_dims) {
        xnn_log_error("failed to define %s operator with input ID #%" PRIu32
                      ": num_nonbatch_dims (%zu) must be "
                      "<= num_dims (%zu)",
                      xnn_node_type_to_string(xnn_node_type_fully_connected),
                      input_id, input_value->quantization.num_nonbatch_dims,
                      input_value->shape.num_dims);
        return xnn_status_invalid_parameter;
      }
      break;
    default:
      xnn_log_error("failed to define %s operator with input ID #%" PRIu32
                    ": unsupported Value datatype %s (%d)",
                    xnn_node_type_to_string(xnn_node_type_fully_connected),
                    input_id, xnn_datatype_to_string(input_value->datatype),
                    input_value->datatype);
      return xnn_status_invalid_parameter;
  }

  if (filter_id >= subgraph->num_values) {
    xnn_log_error("failed to define %s operator with filter ID #%" PRIu32
                  ": invalid Value ID",
                  xnn_node_type_to_string(xnn_node_type_fully_connected),
                  filter_id);
    return xnn_status_invalid_parameter;
  }

  const struct xnn_value* kernel_value = &subgraph->values[filter_id];
  if (kernel_value->type != xnn_value_type_dense_tensor) {
    xnn_log_error("failed to define %s operator with filter ID #%" PRIu32
                  ": unsupported Value type %d (expected dense tensor)",
                  xnn_node_type_to_string(xnn_node_type_fully_connected),
                  filter_id, kernel_value->type);
    return xnn_status_invalid_parameter;
  }

  // Non-static kernel is supported, but only for some data types
  switch (kernel_value->datatype) {
    case xnn_datatype_fp32:
      break;  // non-static kernel is supported
    default:
      if (kernel_value->data == NULL) {
        xnn_log_error("failed to define %s operator with filter ID #%" PRIu32
                      ": non-static Value",
                      xnn_node_type_to_string(xnn_node_type_fully_connected),
                      filter_id);
        return xnn_status_invalid_parameter;
      }
      break;
  }

  // Non-static kernel is supported, but only for some data types
  switch (kernel_value->datatype) {
    case xnn_datatype_bf16:
    case xnn_datatype_fp16:
    case xnn_datatype_fp32:
      break;
    case xnn_datatype_qbint4:
    case xnn_datatype_qcint4:
      if (kernel_value->quantization.zero_point != 8 &&
          kernel_value->quantization.zero_point != 0) {
        xnn_log_error("failed to define %s operator with filter ID #%" PRIu32
                      ": unsupported quantization zero point %" PRId32
                      " for datatype %s, must be equals to 8 (unsigned "
                      "weights) or 0 (signed weights) ",
                      xnn_node_type_to_string(xnn_node_type_fully_connected),
                      filter_id, kernel_value->quantization.zero_point,
                      xnn_datatype_to_string(kernel_value->datatype));
        return xnn_status_invalid_parameter;
      }
      break;
    case xnn_datatype_qcint8:
      break;
    case xnn_datatype_qint8:
      if (kernel_value->quantization.zero_point != 0) {
        xnn_log_error("failed to define %s operator with filter ID #%" PRIu32
                      ": unsupported quantization zero point %" PRId32
                      " for datatype %s",
                      xnn_node_type_to_string(xnn_node_type_fully_connected),
                      filter_id, kernel_value->quantization.zero_point,
                      xnn_datatype_to_string(kernel_value->datatype));
      }
      break;
    case xnn_datatype_quint8:
      break;
    default:
      xnn_log_error("failed to define %s operator with filter ID #%" PRIu32
                    ": unsupported Value datatype %s (%d)",
                    xnn_node_type_to_string(xnn_node_type_fully_connected),
                    filter_id, xnn_datatype_to_string(kernel_value->datatype),
                    kernel_value->datatype);
      return xnn_status_invalid_parameter;
  }

  const bool is_channelwise_quantized =
      kernel_value->datatype == xnn_datatype_qcint8 ||
      kernel_value->datatype == xnn_datatype_qcint4;

  if (is_channelwise_quantized) {
    const size_t output_channels_dim =
        ((flags & XNN_FLAG_TRANSPOSE_WEIGHTS) != 0) ? 1 : 0;
    if (kernel_value->quantization.channel_dimension != output_channels_dim) {
      xnn_log_error("failed to define %s operator with filter ID #%" PRIu32
                    ": invalid channel dimension %zu",
                    xnn_node_type_to_string(xnn_node_type_fully_connected),
                    input_id, kernel_value->quantization.channel_dimension);
      return xnn_status_invalid_parameter;
    }
  }

  const bool is_blockwise_quantized =
      kernel_value->datatype == xnn_datatype_qbint4;

  if (is_blockwise_quantized) {
    // TODO: Unsupported features
    assert((flags & XNN_FLAG_TRANSPOSE_WEIGHTS) == 0);

    const size_t input_channels_dim =
        ((flags & XNN_FLAG_TRANSPOSE_WEIGHTS) != 0) ? 0 : 1;
    const size_t output_channels_dim =
        ((flags & XNN_FLAG_TRANSPOSE_WEIGHTS) != 0) ? 1 : 0;
    if (kernel_value->quantization.channel_dimension_blockwise !=
        output_channels_dim) {
      xnn_log_error("failed to define %s operator with filter ID #%" PRIu32
                    ": invalid channel dimension %zu",
                    xnn_node_type_to_string(xnn_node_type_fully_connected),
                    input_id,
                    kernel_value->quantization.channel_dimension_blockwise);
      return xnn_status_invalid_parameter;
    }
    const size_t input_channels = kernel_value->shape.dim[input_channels_dim];
    if (input_channels % kernel_value->quantization.block_size) {
      xnn_log_error("failed to define %s operator with filter ID #%" PRIu32
                    ": invalid block size %zu, input_channels %zu",
                    xnn_node_type_to_string(xnn_node_type_fully_connected),
                    input_id, kernel_value->quantization.block_size,
                    input_channels);
      return xnn_status_invalid_parameter;
    }
  }

  const struct xnn_value* bias_value = NULL;
  if (bias_id != XNN_INVALID_VALUE_ID) {
    if (bias_id >= subgraph->num_values) {
      xnn_log_error("failed to define %s operator with bias ID #%" PRIu32
                    ": invalid Value ID",
                    xnn_node_type_to_string(xnn_node_type_fully_connected),
                    bias_id);
      return xnn_status_invalid_parameter;
    }

    bias_value = &subgraph->values[bias_id];
    if (bias_value->type != xnn_value_type_dense_tensor) {
      xnn_log_error("failed to define %s operator with bias ID #%" PRIu32
                    ": unsupported Value type %d (expected dense tensor)",
                    xnn_node_type_to_string(xnn_node_type_fully_connected),
                    bias_id, bias_value->type);
      return xnn_status_invalid_parameter;
    }

    // Non-static bias is supported, but only for some data types
    switch (bias_value->datatype) {
      case xnn_datatype_fp32:
        if (is_channelwise_quantized && bias_value->data == NULL) {
          xnn_log_error("failed to define %s operator with bias ID #%" PRIu32
                        ": non-static Value",
                        xnn_node_type_to_string(xnn_node_type_fully_connected),
                        bias_id);
          return xnn_status_invalid_parameter;
        }
        break;  // non-static bias is supported
      default:
        if (bias_value->data == NULL) {
          xnn_log_error("failed to define %s operator with bias ID #%" PRIu32
                        ": non-static Value",
                        xnn_node_type_to_string(xnn_node_type_fully_connected),
                        bias_id);
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
        xnn_log_error("failed to define %s operator with bias ID #%" PRIu32
                      ": unsupported Value datatype %s (%d)",
                      xnn_node_type_to_string(xnn_node_type_fully_connected),
                      bias_id, xnn_datatype_to_string(bias_value->datatype),
                      bias_value->datatype);
        return xnn_status_invalid_parameter;
    }
  }

  status = xnn_subgraph_check_output_node_id(xnn_node_type_fully_connected,
                                             output_id, subgraph->num_values);
  if (status != xnn_status_success) {
    return status;
  }

  const struct xnn_value* output_value = &subgraph->values[output_id];
  status = xnn_subgraph_check_output_type_dense(xnn_node_type_fully_connected,
                                                output_id, output_value);
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
      xnn_log_error("failed to define %s operator with output ID #%" PRIu32
                    ": unsupported Value datatype %s (%d)",
                    xnn_node_type_to_string(xnn_node_type_fully_connected),
                    output_id, xnn_datatype_to_string(output_value->datatype),
                    output_value->datatype);
      return xnn_status_invalid_parameter;
  }

  if (bias_value != NULL) {
    if (!validate_datatypes_with_bias(
            input_value->datatype, kernel_value->datatype, bias_value->datatype,
            output_value->datatype)) {
      xnn_log_error("failed to define %s operator with input ID #%" PRIu32
                    ", filter ID #%" PRIu32 ", bias ID #%" PRIu32
                    ", and output ID #%" PRIu32
                    ": mismatching datatypes across input (%s), filter (%s), "
                    "bias (%s), and output (%s)",
                    xnn_node_type_to_string(xnn_node_type_fully_connected),
                    input_id, filter_id, bias_id, output_id,
                    xnn_datatype_to_string(input_value->datatype),
                    xnn_datatype_to_string(kernel_value->datatype),
                    xnn_datatype_to_string(bias_value->datatype),
                    xnn_datatype_to_string(output_value->datatype));
      return xnn_status_invalid_parameter;
    }
  } else {
    if (!validate_datatypes_without_bias(input_value->datatype,
                                         kernel_value->datatype,
                                         output_value->datatype)) {
      xnn_log_error("failed to define %s operator with input ID #%" PRIu32
                    ", filter ID #%" PRIu32 ", and output ID #%" PRIu32
                    ": mismatching datatypes across input (%s), filter (%s), "
                    "and output (%s)",
                    xnn_node_type_to_string(xnn_node_type_fully_connected),
                    input_id, filter_id, output_id,
                    xnn_datatype_to_string(input_value->datatype),
                    xnn_datatype_to_string(kernel_value->datatype),
                    xnn_datatype_to_string(output_value->datatype));
      return xnn_status_invalid_parameter;
    }
  }

  const enum xnn_datatype input_datatype = input_value->datatype;
  const enum xnn_datatype kernel_datatype = kernel_value->datatype;
  const enum xnn_datatype output_datatype = output_value->datatype;
  if (datatype_is_packable(input_datatype)) {
    if (input_datatype == output_datatype) {
      const struct xnn_gemm_config* gemm_config = NULL;
      switch (input_datatype) {
        case xnn_datatype_fp16:
          if (kernel_datatype == xnn_datatype_fp16) {
            gemm_config = xnn_init_pf16_gemm_config();
          }
          break;
        case xnn_datatype_fp32:
          if (kernel_datatype == xnn_datatype_fp32) {
            gemm_config = xnn_init_pf32_gemm_config();
          }
          break;
        case xnn_datatype_qint8:
          if (kernel_datatype == xnn_datatype_qcint8) {
            gemm_config = xnn_init_pqs8_qc8w_gemm_config();
          }
          break;
        default:
          XNN_UNREACHABLE;
      }
      if (gemm_config != NULL) {
        // Insert a node to pack the LHS.
        uint32_t new_id = XNN_INVALID_VALUE_ID;
        status = xnn_insert_pack_lh_node(subgraph, input_id, &new_id);
        if (status != xnn_status_success) {
          return status;
        }
        input_id = new_id;
        // To prevent issues with packing, coerce the shape of the inputs from
        // `[B, M, K]` to `[B * M, K]` for the fully-connected op.
        subgraph->values[new_id].flags |= XNN_FLAG_SQUASH_GROUPS;
      }
    }
  }
  struct xnn_node* node = xnn_subgraph_new_node(subgraph);
  if (node == NULL) {
    return xnn_status_out_of_memory;
  }

  node->type = xnn_node_type_fully_connected;
  node->activation.output_min = output_min;
  node->activation.output_max = output_max;
  node->num_inputs = 2 + (size_t)(bias_id != XNN_INVALID_VALUE_ID);
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
