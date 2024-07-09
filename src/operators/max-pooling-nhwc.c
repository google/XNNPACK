// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <inttypes.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include <fp16/fp16.h>
#include "xnnpack.h"
#include "xnnpack/allocator.h"
#include "xnnpack/common.h"
#include "xnnpack/compute.h"
#include "xnnpack/config-types.h"
#include "xnnpack/config.h"
#include "xnnpack/indirection.h"
#include "xnnpack/log.h"
#include "xnnpack/math.h"
#include "xnnpack/microparams.h"
#include "xnnpack/operator-type.h"
#include "xnnpack/operator-utils.h"
#include "xnnpack/operator.h"
#include "xnnpack/params.h"
#include "pthreadpool.h"

static inline size_t compute_output_dimension_with_tf_same_padding(
    size_t input_dimension,
    size_t stride_dimension)
{
  return divide_round_up(input_dimension, stride_dimension);
}

static enum xnn_status create_max_pooling2d_nhwc(
    uint32_t input_padding_top,
    uint32_t input_padding_right,
    uint32_t input_padding_bottom,
    uint32_t input_padding_left,
    uint32_t pooling_height,
    uint32_t pooling_width,
    uint32_t stride_height,
    uint32_t stride_width,
    uint32_t dilation_height,
    uint32_t dilation_width,
    uint32_t flags,
    const void* params,
    size_t params_size,
    const struct xnn_maxpool_config* maxpool_config,
    enum xnn_operator_type operator_type,
    xnn_operator_t* max_pooling_op_out)
{
  xnn_operator_t max_pooling_op = NULL;
  enum xnn_status status = xnn_status_uninitialized;

  if ((xnn_params.init_flags & XNN_INIT_FLAG_XNNPACK) == 0) {
    xnn_log_error("failed to setup %s operator: XNNPACK is not initialized",
      xnn_operator_type_to_string(operator_type));
    return xnn_status_uninitialized;
  }

  status = xnn_status_invalid_parameter;

  const uint32_t pooling_size = pooling_height * pooling_width;
  if (pooling_size == 0) {
    xnn_log_error(
      "failed to create %s operator with %" PRIu32 "x%" PRIu32 " pooling size: "
      "pooling size dimensions must be non-zero",
      xnn_operator_type_to_string(operator_type),
      pooling_width, pooling_height);
    goto error;
  }

  if (pooling_size == 1) {
    xnn_log_error(
      "failed to create %s operator with 1 pooling element: 1x1 pooling is meaningless",
      xnn_operator_type_to_string(operator_type));
    goto error;
  }

  if (stride_height == 0 || stride_width == 0) {
    xnn_log_error(
      "failed to create %s operator with %" PRIu32 "x%" PRIu32 " stride: stride dimensions must be non-zero",
      xnn_operator_type_to_string(operator_type), stride_width, stride_height);
    goto error;
  }

  if (dilation_height == 0 || dilation_width == 0) {
    xnn_log_error(
      "failed to create %s operator with %" PRIu32 "x%" PRIu32 " dilation: dilation dimensions must be non-zero",
      xnn_operator_type_to_string(operator_type), dilation_width, dilation_height);
    goto error;
  }

  if (stride_height > pooling_height) {
    xnn_log_error(
      "failed to create %s operator with %" PRIu32 " stride height: must be less than pooling height %" PRIu32,
      xnn_operator_type_to_string(operator_type), stride_height, pooling_height);
    return xnn_status_invalid_parameter;
  }

  if (stride_width > pooling_width) {
    xnn_log_error(
      "failed to create %s operator with %" PRIu32 " stride width: must be less than pooling width %" PRIu32,
      xnn_operator_type_to_string(operator_type), stride_width, pooling_width);
    return xnn_status_invalid_parameter;
  }

  const bool any_padding = (input_padding_left | input_padding_top | input_padding_right | input_padding_bottom) != 0;
  if ((flags & XNN_FLAG_TENSORFLOW_SAME_PADDING) != 0) {
    if (any_padding) {
      xnn_log_error(
        "failed to create %s operator with %" PRIu32 "+%" PRIu32 "x%" PRIu32 "+%" PRIu32" padding: "
        "TensorFlow SAME padding can't be combined with explicit padding specification",
        xnn_operator_type_to_string(operator_type),
        input_padding_top, input_padding_left, input_padding_bottom, input_padding_right);
      goto error;
    }
  }

  status = xnn_status_out_of_memory;

  max_pooling_op = xnn_allocate_zero_simd_memory(sizeof(struct xnn_operator));
  if (max_pooling_op == NULL) {
    xnn_log_error(
      "failed to allocate %zu bytes for %s operator descriptor",
      sizeof(struct xnn_operator), xnn_operator_type_to_string(operator_type));
    goto error;
  }

  max_pooling_op->padding_top = input_padding_top;
  max_pooling_op->padding_right = input_padding_right;
  max_pooling_op->padding_bottom = input_padding_bottom;
  max_pooling_op->padding_left = input_padding_left;

  max_pooling_op->kernel_height = pooling_height;
  max_pooling_op->kernel_width = pooling_width;
  max_pooling_op->stride_height = stride_height;
  max_pooling_op->stride_width = stride_width;
  max_pooling_op->dilation_height = dilation_height;
  max_pooling_op->dilation_width = dilation_width;

  memcpy(&max_pooling_op->params, params, params_size);
  max_pooling_op->type = operator_type;
  max_pooling_op->flags = flags;
  max_pooling_op->maxpool_config = maxpool_config;

  max_pooling_op->state = xnn_run_state_invalid;

  *max_pooling_op_out = max_pooling_op;
  return xnn_status_success;

error:
  xnn_delete_operator(max_pooling_op);
  return status;
}

enum xnn_status xnn_create_max_pooling2d_nhwc_s8(
    uint32_t input_padding_top,
    uint32_t input_padding_right,
    uint32_t input_padding_bottom,
    uint32_t input_padding_left,
    uint32_t pooling_height,
    uint32_t pooling_width,
    uint32_t stride_height,
    uint32_t stride_width,
    uint32_t dilation_height,
    uint32_t dilation_width,
    int8_t output_min,
    int8_t output_max,
    uint32_t flags,
    xnn_operator_t* max_pooling_op_out)
{
  if (output_min > output_max) {
    xnn_log_error(
      "failed to create %s operator with [%" PRId8 ", %" PRId8 "] output range: lower bound must be less than or equal to upper bound",
      xnn_operator_type_to_string(xnn_operator_type_max_pooling_nhwc_s8), output_min, output_max);
    return xnn_status_invalid_parameter;
  }

  const struct xnn_maxpool_config* maxpool_config = xnn_init_s8_maxpool_config();
  assert(maxpool_config != NULL);
  union xnn_s8_minmax_params params;
  maxpool_config->init.s8(&params, output_min, output_max);
  return create_max_pooling2d_nhwc(
    input_padding_top, input_padding_right, input_padding_bottom, input_padding_left,
    pooling_height, pooling_width,
    stride_height, stride_width,
    dilation_height, dilation_width,
    flags,
    &params, sizeof(params),
    maxpool_config,
    xnn_operator_type_max_pooling_nhwc_s8,
    max_pooling_op_out);
}

enum xnn_status xnn_create_max_pooling2d_nhwc_u8(
    uint32_t input_padding_top,
    uint32_t input_padding_right,
    uint32_t input_padding_bottom,
    uint32_t input_padding_left,
    uint32_t pooling_height,
    uint32_t pooling_width,
    uint32_t stride_height,
    uint32_t stride_width,
    uint32_t dilation_height,
    uint32_t dilation_width,
    uint8_t output_min,
    uint8_t output_max,
    uint32_t flags,
    xnn_operator_t* max_pooling_op_out)
{
  if (output_min > output_max) {
    xnn_log_error(
      "failed to create %s operator with [%" PRIu8 ", %" PRIu8 "] output range: lower bound must be less than or equal to upper bound",
      xnn_operator_type_to_string(xnn_operator_type_max_pooling_nhwc_u8), output_min, output_max);
    return xnn_status_invalid_parameter;
  }

  const struct xnn_maxpool_config* maxpool_config = xnn_init_u8_maxpool_config();
  assert(maxpool_config != NULL);
  union xnn_u8_minmax_params params;
  maxpool_config->init.u8(&params, output_min, output_max);
  return create_max_pooling2d_nhwc(
    input_padding_top, input_padding_right, input_padding_bottom, input_padding_left,
    pooling_height, pooling_width,
    stride_height, stride_width,
    dilation_height, dilation_width,
    flags,
    &params, sizeof(params),
    maxpool_config,
    xnn_operator_type_max_pooling_nhwc_u8,
    max_pooling_op_out);
}

enum xnn_status xnn_create_max_pooling2d_nhwc_f32(
    uint32_t input_padding_top,
    uint32_t input_padding_right,
    uint32_t input_padding_bottom,
    uint32_t input_padding_left,
    uint32_t pooling_height,
    uint32_t pooling_width,
    uint32_t stride_height,
    uint32_t stride_width,
    uint32_t dilation_height,
    uint32_t dilation_width,
    float output_min,
    float output_max,
    uint32_t flags,
    xnn_operator_t* max_pooling_op_out)
{
  if (isnan(output_min)) {
    xnn_log_error(
      "failed to create %s with NaN output lower bound: lower bound must be non-NaN",
      xnn_operator_type_to_string(xnn_operator_type_max_pooling_nhwc_f32));
    return xnn_status_invalid_parameter;
  }

  if (isnan(output_max)) {
    xnn_log_error(
      "failed to create %s with NaN output upper bound: upper bound must be non-NaN",
      xnn_operator_type_to_string(xnn_operator_type_max_pooling_nhwc_f32));
    return xnn_status_invalid_parameter;
  }

  if (output_min > output_max) {
    xnn_log_error(
      "failed to create %s operator with [%.7g, %.7g] output range: lower bound must be less than or equal to upper bound",
      xnn_operator_type_to_string(xnn_operator_type_max_pooling_nhwc_f32), output_min, output_max);
    return xnn_status_invalid_parameter;
  }

  const struct xnn_maxpool_config* maxpool_config = xnn_init_f32_maxpool_config();
  if (maxpool_config == NULL) {
    xnn_log_error(
      "failed to create %s operator: unsupported hardware configuration",
      xnn_operator_type_to_string(xnn_operator_type_max_pooling_nhwc_f32));
    return xnn_status_unsupported_hardware;
  }
  union xnn_f32_minmax_params params;
  maxpool_config->init.f32(&params, output_min, output_max);
  return create_max_pooling2d_nhwc(
    input_padding_top, input_padding_right, input_padding_bottom, input_padding_left,
    pooling_height, pooling_width,
    stride_height, stride_width,
    dilation_height, dilation_width,
    flags,
    &params, sizeof(params),
    maxpool_config,
    xnn_operator_type_max_pooling_nhwc_f32,
    max_pooling_op_out);
}

enum xnn_status xnn_create_max_pooling2d_nhwc_f16(
    uint32_t input_padding_top,
    uint32_t input_padding_right,
    uint32_t input_padding_bottom,
    uint32_t input_padding_left,
    uint32_t pooling_height,
    uint32_t pooling_width,
    uint32_t stride_height,
    uint32_t stride_width,
    uint32_t dilation_height,
    uint32_t dilation_width,
    float output_min,
    float output_max,
    uint32_t flags,
    xnn_operator_t* max_pooling_op_out)
{
  if (isnan(output_min)) {
    xnn_log_error(
      "failed to create %s with NaN output lower bound: lower bound must be non-NaN",
      xnn_operator_type_to_string(xnn_operator_type_max_pooling_nhwc_f16));
    return xnn_status_invalid_parameter;
  }

  if (isnan(output_max)) {
    xnn_log_error(
      "failed to create %s with NaN output upper bound: upper bound must be non-NaN",
      xnn_operator_type_to_string(xnn_operator_type_max_pooling_nhwc_f16));
    return xnn_status_invalid_parameter;
  }

  const uint16_t output_min_as_half = fp16_ieee_from_fp32_value(output_min);
  const uint16_t output_max_as_half = fp16_ieee_from_fp32_value(output_max);
  output_min = fp16_ieee_to_fp32_value(output_min_as_half);
  output_max = fp16_ieee_to_fp32_value(output_max_as_half);
  if (output_min > output_max) {
    xnn_log_error(
      "failed to create %s operator with [%.7g, %.7g] output range: lower bound must be less than or equal to upper bound",
      xnn_operator_type_to_string(xnn_operator_type_max_pooling_nhwc_f16), output_min, output_max);
    return xnn_status_invalid_parameter;
  }

  const struct xnn_maxpool_config* maxpool_config = xnn_init_f16_maxpool_config();
  if (maxpool_config == NULL) {
    xnn_log_error("failed to create %s operator: unsupported hardware configuration",
                  xnn_operator_type_to_string(xnn_operator_type_max_pooling_nhwc_f16));
    return xnn_status_unsupported_hardware;
  }

  union xnn_f16_minmax_params params;
  if (maxpool_config->init.f16 != NULL) {
    maxpool_config->init.f16(&params, output_min_as_half, output_max_as_half);
  }
  return create_max_pooling2d_nhwc(
    input_padding_top, input_padding_right, input_padding_bottom, input_padding_left,
    pooling_height, pooling_width,
    stride_height, stride_width,
    dilation_height, dilation_width,
    flags,
    &params, sizeof(params),
    maxpool_config,
    xnn_operator_type_max_pooling_nhwc_f16,
    max_pooling_op_out);
}

static enum xnn_status reshape_max_pooling2d_nhwc(
  xnn_operator_t max_pooling_op,
  enum xnn_operator_type expected_operator_type,
  size_t batch_size,
  size_t input_height,
  size_t input_width,
  size_t channels,
  size_t input_pixel_stride,
  size_t output_pixel_stride,
  uint32_t log2_input_element_size,
  uint32_t log2_output_element_size,
  const struct xnn_maxpool_config maxpool[restrict XNN_MIN_ELEMENTS(1)],
  const void* params,
  size_t params_size,
  size_t* output_height_out,
  size_t* output_width_out,
  pthreadpool_t threadpool)
{
  if (max_pooling_op->type != expected_operator_type) {
    xnn_log_error("failed to reshape operator: operator type mismatch (expected %s, got %s)",
      xnn_operator_type_to_string(expected_operator_type),
      xnn_operator_type_to_string(max_pooling_op->type));
    return xnn_status_invalid_parameter;
  }
  max_pooling_op->state = xnn_run_state_invalid;

  if ((xnn_params.init_flags & XNN_INIT_FLAG_XNNPACK) == 0) {
    xnn_log_error(
      "failed to reshape %s operator: XNNPACK is not initialized",
      xnn_operator_type_to_string(max_pooling_op->type));
    return xnn_status_uninitialized;
  }

  if (input_width == 0 || input_height == 0) {
    xnn_log_error(
      "failed to reshape %s operator with %zux%zu input: input dimensions must be non-zero",
      xnn_operator_type_to_string(max_pooling_op->type), input_width, input_height);
    return xnn_status_invalid_parameter;
  }

  if (channels == 0) {
    xnn_log_error(
      "failed to reshape %s operator with %zu channels: number of channels must be non-zero",
      xnn_operator_type_to_string(expected_operator_type), channels);
    return xnn_status_invalid_parameter;
  }

  if (input_pixel_stride < channels) {
    xnn_log_error(
      "failed to reshape %s operator with input pixel stride of %zu: "
      "stride must be at least as large as the number of channels (%zu)",
      xnn_operator_type_to_string(expected_operator_type), input_pixel_stride, channels);
    return xnn_status_invalid_parameter;
  }

  if (output_pixel_stride < channels) {
    xnn_log_error(
      "failed to reshape %s operator with output pixel stride of %zu: "
      "stride must be at least as large as the number of channels (%zu)",
      xnn_operator_type_to_string(expected_operator_type), output_pixel_stride, channels);
    return xnn_status_invalid_parameter;
  }

  max_pooling_op->channels = channels;
  max_pooling_op->input_pixel_stride = input_pixel_stride;
  max_pooling_op->output_pixel_stride = output_pixel_stride;

  if (batch_size == 0) {
    max_pooling_op->state = xnn_run_state_skip;
    return xnn_status_success;
  }

  max_pooling_op->input_height = input_height;
  max_pooling_op->input_width = input_width;

  if (max_pooling_op->flags & XNN_FLAG_TENSORFLOW_SAME_PADDING) {
    max_pooling_op->output_height = compute_output_dimension_with_tf_same_padding(
        input_height, max_pooling_op->stride_height);
    max_pooling_op->output_width = compute_output_dimension_with_tf_same_padding(
        input_width, max_pooling_op->stride_width);

    const uint32_t effective_kernel_height = (max_pooling_op->kernel_height - 1) * max_pooling_op->dilation_height + 1;
    const uint32_t effective_kernel_width = (max_pooling_op->kernel_width - 1) * max_pooling_op->dilation_width + 1;
    const uint32_t total_padding_height =
      doz((max_pooling_op->output_height - 1) * max_pooling_op->stride_height + effective_kernel_height, input_height);
    const uint32_t total_padding_width =
      doz((max_pooling_op->output_width - 1) * max_pooling_op->stride_width + effective_kernel_width, input_width);
    max_pooling_op->padding_top = total_padding_height / 2;
    max_pooling_op->padding_left = total_padding_width / 2;
    max_pooling_op->padding_bottom = total_padding_height - max_pooling_op->padding_top;
    max_pooling_op->padding_right = total_padding_width - max_pooling_op->padding_left;
  } else {
    max_pooling_op->output_height = xnn_compute_convolution_output_dimension(
        max_pooling_op->padding_top + input_height + max_pooling_op->padding_bottom,
        max_pooling_op->kernel_height,
        max_pooling_op->dilation_height,
        max_pooling_op->stride_height);
    max_pooling_op->output_width = xnn_compute_convolution_output_dimension(
        max_pooling_op->padding_left + input_width + max_pooling_op->padding_right,
        max_pooling_op->kernel_width,
        max_pooling_op->dilation_width,
        max_pooling_op->stride_width);
  }

  if (output_height_out != NULL) {
    *output_height_out = max_pooling_op->output_height;
  }
  if (output_width_out != NULL) {
    *output_width_out = max_pooling_op->output_width;
  }

  const size_t pooling_height = max_pooling_op->kernel_height;
  const size_t pooling_width = max_pooling_op->kernel_width;
  const size_t pooling_size = pooling_height * pooling_width;
  const size_t output_height = max_pooling_op->output_height;
  const size_t output_width = max_pooling_op->output_width;
  const uint32_t first_pass_tile_size = maxpool->first_pass_tile_size;

  const size_t step_width =
    max_pooling_op->dilation_width > 1 ? pooling_width : min(max_pooling_op->stride_width, pooling_width);
  const size_t step_height = pooling_size + (output_width - 1) * step_width * pooling_height;

  if (input_height != max_pooling_op->last_input_height ||
      input_width != max_pooling_op->last_input_width)
  {
    // Micro-kernel may read up to (first_pass_tile_size - 1) elements after the end of indirection buffer.
    const size_t indirection_buffer_size = sizeof(void*) * ((first_pass_tile_size - 1) + output_height * step_height);
    const void** indirection_buffer =
      (const void**) xnn_reallocate_memory(max_pooling_op->indirection_buffer, indirection_buffer_size);
    if (indirection_buffer == NULL) {
      xnn_log_error("failed to allocate %zu bytes for %s operator indirection buffer",
        indirection_buffer_size, xnn_operator_type_to_string(max_pooling_op->type));
      return xnn_status_out_of_memory;
    }
    max_pooling_op->indirection_buffer = indirection_buffer;
    xnn_log_debug("allocated %zu bytes for indirection buffer in %s operator",
      indirection_buffer_size, xnn_operator_type_to_string(max_pooling_op->type));

    // Set a dummy input first, the actual input offset is calculated in setup when we have the input pointer.
    max_pooling_op->input = NULL;

    xnn_indirection_init_maxpool2d(max_pooling_op, step_height, step_width, log2_input_element_size);

    max_pooling_op->last_input = max_pooling_op->input;
    max_pooling_op->last_input_height = input_height;
    max_pooling_op->last_input_width = input_width;
  }

  const uint32_t remainder_pass_tile_size = maxpool->remainder_pass_tile_size;

  const size_t indirect_input_height_stride = step_height * sizeof(void*);
  const size_t output_width_stride = max_pooling_op->output_pixel_stride << log2_output_element_size;
  const size_t output_height_stride = output_width * output_width_stride;
  const size_t multipass_adjustment = round_up(doz(pooling_size, first_pass_tile_size), remainder_pass_tile_size) + first_pass_tile_size;

  max_pooling_op->context.max_pooling = (struct max_pooling_context) {
    .indirect_input = max_pooling_op->indirection_buffer,
    .indirect_input_height_stride = indirect_input_height_stride,
    .input_batch_stride = (input_height * input_width * max_pooling_op->input_pixel_stride) << log2_input_element_size,
    .output_batch_stride = output_height * output_height_stride,
    .output_height_stride = output_height_stride,
    .output_width = output_width,
    .pooling_size = pooling_size,
    .channels = channels,
    .input_increment = (pooling_height * step_width - multipass_adjustment) * sizeof(void*),
    .output_increment = output_width_stride - (channels << log2_output_element_size),
    .ukernel = maxpool->ukernel,
  };
  memcpy(&max_pooling_op->context.max_pooling.params, params, params_size);

  max_pooling_op->compute[0].type = xnn_parallelization_type_2d;
  max_pooling_op->compute[0].task_2d = (pthreadpool_task_2d_t) xnn_compute_max_pooling;
  max_pooling_op->compute[0].range[0] = batch_size;
  max_pooling_op->compute[0].range[1] = output_height;
  max_pooling_op->state = xnn_run_state_needs_setup;

  return xnn_status_success;
}

enum xnn_status xnn_reshape_max_pooling2d_nhwc_s8(
  xnn_operator_t max_pooling_op,
  size_t batch_size,
  size_t input_height,
  size_t input_width,
  size_t channels,
  size_t input_pixel_stride,
  size_t output_pixel_stride,
  size_t* output_height_out,
  size_t* output_width_out,
  pthreadpool_t threadpool)
{
  return reshape_max_pooling2d_nhwc(
    max_pooling_op, xnn_operator_type_max_pooling_nhwc_s8,
    batch_size, input_height, input_width,
    channels, input_pixel_stride, output_pixel_stride,
    /*log2_input_element_size=*/XNN_LOG2_SIZEOF_INT8_T,
    /*log2_output_element_size=*/XNN_LOG2_SIZEOF_INT8_T,
    max_pooling_op->maxpool_config,
    &max_pooling_op->params.s8_minmax, sizeof(max_pooling_op->params.s8_minmax),
    output_height_out, output_width_out,
    threadpool);
}

enum xnn_status xnn_reshape_max_pooling2d_nhwc_u8(
  xnn_operator_t max_pooling_op,
  size_t batch_size,
  size_t input_height,
  size_t input_width,
  size_t channels,
  size_t input_pixel_stride,
  size_t output_pixel_stride,
  size_t* output_height_out,
  size_t* output_width_out,
  pthreadpool_t threadpool)
{
  return reshape_max_pooling2d_nhwc(
    max_pooling_op, xnn_operator_type_max_pooling_nhwc_u8,
    batch_size, input_height, input_width,
    channels, input_pixel_stride, output_pixel_stride,
    /*log2_input_element_size=*/XNN_LOG2_SIZEOF_UINT8_T,
    /*log2_output_element_size=*/XNN_LOG2_SIZEOF_UINT8_T,
    max_pooling_op->maxpool_config,
    &max_pooling_op->params.u8_minmax, sizeof(max_pooling_op->params.u8_minmax),
    output_height_out, output_width_out,
    threadpool);
}

enum xnn_status xnn_reshape_max_pooling2d_nhwc_f16(
  xnn_operator_t max_pooling_op,
  size_t batch_size,
  size_t input_height,
  size_t input_width,
  size_t channels,
  size_t input_pixel_stride,
  size_t output_pixel_stride,
  size_t* output_height_out,
  size_t* output_width_out,
  pthreadpool_t threadpool)
{
  return reshape_max_pooling2d_nhwc(
    max_pooling_op, xnn_operator_type_max_pooling_nhwc_f16,
    batch_size, input_height, input_width,
    channels, input_pixel_stride, output_pixel_stride,
    /*log2_input_element_size=*/XNN_LOG2_SIZEOF_HALF,
    /*log2_output_element_size=*/XNN_LOG2_SIZEOF_HALF,
    max_pooling_op->maxpool_config,
    &max_pooling_op->params.f16_minmax, sizeof(max_pooling_op->params.f16_minmax),
    output_height_out, output_width_out,
    threadpool);
}

enum xnn_status xnn_reshape_max_pooling2d_nhwc_f32(
  xnn_operator_t max_pooling_op,
  size_t batch_size,
  size_t input_height,
  size_t input_width,
  size_t channels,
  size_t input_pixel_stride,
  size_t output_pixel_stride,
  size_t* output_height_out,
  size_t* output_width_out,
  pthreadpool_t threadpool)
{
  return reshape_max_pooling2d_nhwc(
    max_pooling_op, xnn_operator_type_max_pooling_nhwc_f32,
    batch_size, input_height, input_width,
    channels, input_pixel_stride, output_pixel_stride,
    /*log2_input_element_size=*/XNN_LOG2_SIZEOF_FLOAT,
    /*log2_output_element_size=*/XNN_LOG2_SIZEOF_FLOAT,
    max_pooling_op->maxpool_config,
    &max_pooling_op->params.f32_minmax, sizeof(max_pooling_op->params.f32_minmax),
    output_height_out, output_width_out,
    threadpool);
}

static enum xnn_status setup_max_pooling2d_nhwc(
  xnn_operator_t max_pooling_op,
  enum xnn_operator_type expected_operator_type,
  const void* input,
  void* output)
{
  if (max_pooling_op->type != expected_operator_type) {
    xnn_log_error("failed to setup operator: operator type mismatch (expected %s, got %s)",
      xnn_operator_type_to_string(expected_operator_type),
      xnn_operator_type_to_string(max_pooling_op->type));
    return xnn_status_invalid_parameter;
  }

  switch (max_pooling_op->state) {
    case xnn_run_state_skip:
      return xnn_status_success;
    case xnn_run_state_invalid:
      xnn_log_error(
        "failed to setup %s operator: operator has not been reshaped yet",
        xnn_operator_type_to_string(max_pooling_op->type));
      return xnn_status_invalid_state;
    case xnn_run_state_needs_setup:
      // Operator has been reshaped, but not setup, continue with setup.
    case xnn_run_state_ready:
      // Operator has been reshaped, and we are setting up with different pointers.
      break;
  }

  max_pooling_op->context.max_pooling.input_offset = (size_t) ((uintptr_t) input - (uintptr_t) max_pooling_op->last_input);
  max_pooling_op->context.max_pooling.output = output;

  max_pooling_op->state = xnn_run_state_ready;

  return xnn_status_success;
}

enum xnn_status xnn_setup_max_pooling2d_nhwc_s8(
    xnn_operator_t max_pooling_op,
    const int8_t* input,
    int8_t* output)
{
  return setup_max_pooling2d_nhwc(
    max_pooling_op, xnn_operator_type_max_pooling_nhwc_s8,
    input, output);
}

enum xnn_status xnn_setup_max_pooling2d_nhwc_u8(
    xnn_operator_t max_pooling_op,
    const uint8_t* input,
    uint8_t* output)
{
  return setup_max_pooling2d_nhwc(
    max_pooling_op, xnn_operator_type_max_pooling_nhwc_u8,
    input, output);
}

enum xnn_status xnn_setup_max_pooling2d_nhwc_f16(
    xnn_operator_t max_pooling_op,
    const void* input,
    void* output)
{
  return setup_max_pooling2d_nhwc(
    max_pooling_op, xnn_operator_type_max_pooling_nhwc_f16,
    input, output);
}

enum xnn_status xnn_setup_max_pooling2d_nhwc_f32(
    xnn_operator_t max_pooling_op,
    const float* input,
    float* output)
{
  return setup_max_pooling2d_nhwc(
    max_pooling_op, xnn_operator_type_max_pooling_nhwc_f32,
    input, output);
}
