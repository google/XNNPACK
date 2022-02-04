// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include <xnnpack.h>
#include <xnnpack/allocator.h>
#include <xnnpack/log.h>
#include <xnnpack/operator.h>
#include <xnnpack/pack.h>
#include <xnnpack/params-init.h>
#include <xnnpack/params.h>


static enum xnn_status create_prelu_nc(
    size_t channels,
    size_t input_stride,
    size_t output_stride,
    const void* negative_slope,
    uint32_t flags,
    uint32_t log2_weights_element_size,
    xnn_pack_prelu_w_function pack_prelu_w,
    uint32_t datatype_init_flags,
    enum xnn_operator_type operator_type,
    xnn_operator_t* prelu_op_out)
{
  xnn_operator_t prelu_op = NULL;
  enum xnn_status status = xnn_status_uninitialized;

  if ((xnn_params.init_flags & XNN_INIT_FLAG_XNNPACK) == 0) {
    xnn_log_error("failed to setup %s operator: XNNPACK is not initialized",
      xnn_operator_type_to_string(operator_type));
    return xnn_status_uninitialized;
  }

  status = xnn_status_unsupported_hardware;

  if ((xnn_params.init_flags & datatype_init_flags) != datatype_init_flags) {
    xnn_log_error(
      "failed to create %s operator: operations on data type are not supported",
      xnn_operator_type_to_string(operator_type));
    goto error;
  }

  status = xnn_status_invalid_parameter;

  if (channels == 0) {
    xnn_log_error(
      "failed to create %s operator with %zu channels: number of channels must be non-zero",
      xnn_operator_type_to_string(operator_type), channels);
    goto error;
  }

  if (input_stride < channels) {
    xnn_log_error(
      "failed to create %s operator with input element stride of %zu: "
      "stride must be at least as large as the number of channels (%zu)",
      xnn_operator_type_to_string(operator_type), input_stride, channels);
    goto error;
  }

  if (output_stride < channels) {
    xnn_log_error(
      "failed to create %s operator with output element stride of %zu: "
      "stride must be at least as large as the number of channels (%zu)",
      xnn_operator_type_to_string(operator_type), output_stride, channels);
    goto error;
  }

  status = xnn_status_out_of_memory;

  prelu_op = xnn_allocate_zero_simd_memory(sizeof(struct xnn_operator));
  if (prelu_op == NULL) {
    xnn_log_error(
      "failed to allocate %zu bytes for %s operator descriptor",
      sizeof(struct xnn_operator), xnn_operator_type_to_string(operator_type));
    goto error;
  }

  const size_t packed_weights_size = (channels << log2_weights_element_size) + XNN_EXTRA_BYTES;
  prelu_op->packed_weights = xnn_allocate_simd_memory(packed_weights_size);
  if (prelu_op->packed_weights == NULL) {
    xnn_log_error(
      "failed to allocate %zu bytes for %s operator packed weights",
      packed_weights_size, xnn_operator_type_to_string(operator_type));
    goto error;
  }
  pack_prelu_w(channels, negative_slope, prelu_op->packed_weights);

  prelu_op->channels = channels;
  prelu_op->input_pixel_stride = input_stride;
  prelu_op->output_pixel_stride = output_stride;

  prelu_op->type = operator_type;
  prelu_op->flags = flags;

  prelu_op->state = xnn_run_state_invalid;

  *prelu_op_out = prelu_op;
  return xnn_status_success;

error:
  xnn_delete_operator(prelu_op);
  return status;
}


enum xnn_status xnn_create_prelu_nc_f16(
    size_t channels,
    size_t input_stride,
    size_t output_stride,
    const void* negative_slope,
    uint32_t flags,
    xnn_operator_t* prelu_op_out)
{
  xnn_pack_prelu_w_function pack_prelu_w = (xnn_pack_prelu_w_function) xnn_pack_f16_prelu_w;
  if (flags & XNN_FLAG_FP32_STATIC_WEIGHTS) {
    pack_prelu_w = (xnn_pack_prelu_w_function) xnn_pack_f32_to_f16_prelu_w;
  }

  return create_prelu_nc(
    channels, input_stride, output_stride,
    negative_slope, flags,
    1 /* log2(sizeof(uint16_t)) */,
    pack_prelu_w,
    XNN_INIT_FLAG_F16, xnn_operator_type_prelu_nc_f16,
    prelu_op_out);
}

enum xnn_status xnn_create_prelu_nc_f32(
    size_t channels,
    size_t input_stride,
    size_t output_stride,
    const float* negative_slope,
    uint32_t flags,
    xnn_operator_t* prelu_op_out)
{
  return create_prelu_nc(
    channels, input_stride, output_stride,
    negative_slope, flags,
    2 /* log2(sizeof(float)) */,
    (xnn_pack_prelu_w_function) xnn_pack_f32_prelu_w,
    XNN_INIT_FLAG_F32, xnn_operator_type_prelu_nc_f32,
    prelu_op_out);
}

static enum xnn_status setup_prelu_nc(
    xnn_operator_t prelu_op,
    enum xnn_operator_type expected_operator_type,
    size_t batch_size,
    const float* input,
    float* output,
    uint32_t datatype_init_flags,
    uint32_t log2_element_size,
    const struct prelu_parameters prelu[restrict XNN_MIN_ELEMENTS(1)],
    size_t num_threads)
{
  if (prelu_op->type != expected_operator_type) {
    xnn_log_error("failed to setup operator: operator type mismatch (expected %s, got %s)",
      xnn_operator_type_to_string(expected_operator_type),
      xnn_operator_type_to_string(prelu_op->type));
    return xnn_status_invalid_parameter;
  }
  prelu_op->state = xnn_run_state_invalid;

  if ((xnn_params.init_flags & XNN_INIT_FLAG_XNNPACK) == 0) {
    xnn_log_error("failed to setup %s operator: XNNPACK is not initialized",
      xnn_operator_type_to_string(expected_operator_type));
    return xnn_status_uninitialized;
  }

  if ((xnn_params.init_flags & datatype_init_flags) != datatype_init_flags) {
    xnn_log_error("failed to setup %s operator: operations on data type are not supported",
      xnn_operator_type_to_string(expected_operator_type));
    return xnn_status_unsupported_hardware;
  }

  if (batch_size == 0) {
    prelu_op->state = xnn_run_state_skip;
    return xnn_status_success;
  }

  const size_t channels = prelu_op->channels;
  prelu_op->context.prelu = (struct prelu_context) {
    .n = channels << log2_element_size,
    .x = input,
    .x_stride = prelu_op->input_pixel_stride << log2_element_size,
    .w = prelu_op->packed_weights,
    .y = output,
    .y_stride = prelu_op->output_pixel_stride << log2_element_size,
    .ukernel = prelu->ukernel,
  };

  size_t batch_tile = batch_size;
  if (num_threads > 1) {
    const size_t target_tiles_per_thread = 5;
    const size_t max_batch_tile = divide_round_up(batch_size, num_threads * target_tiles_per_thread);
    if (max_batch_tile < batch_tile) {
      const uint32_t row_tile = prelu->row_tile;
      batch_tile = min(batch_tile, divide_round_up(batch_tile, max_batch_tile * row_tile) * row_tile);
    }
  }
  prelu_op->compute.type = xnn_parallelization_type_1d_tile_1d;
  prelu_op->compute.task_1d_tile_1d = (pthreadpool_task_1d_tile_1d_t) xnn_compute_prelu;
  prelu_op->compute.range[0] = batch_size;
  prelu_op->compute.tile[0] = batch_tile;
  prelu_op->state = xnn_run_state_ready;

  return xnn_status_success;
}

enum xnn_status xnn_setup_prelu_nc_f16(
    xnn_operator_t prelu_op,
    size_t batch_size,
    const void* input,
    void* output,
    pthreadpool_t threadpool)
{
  return setup_prelu_nc(
    prelu_op, xnn_operator_type_prelu_nc_f16,
    batch_size, input, output,
    XNN_INIT_FLAG_F16,
    1 /* log2(sizeof(uint16_t)) */,
    &xnn_params.f16.prelu,
    pthreadpool_get_threads_count(threadpool));
}

enum xnn_status xnn_setup_prelu_nc_f32(
    xnn_operator_t prelu_op,
    size_t batch_size,
    const float* input,
    float* output,
    pthreadpool_t threadpool)
{
  return setup_prelu_nc(
    prelu_op, xnn_operator_type_prelu_nc_f32,
    batch_size, input, output,
    XNN_INIT_FLAG_F32,
    2 /* log2(sizeof(float)) */,
    &xnn_params.f32.prelu,
    pthreadpool_get_threads_count(threadpool));
}
