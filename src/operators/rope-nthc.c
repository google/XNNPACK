// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <math.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include <xnnpack.h>
#include <xnnpack/allocator.h>
#include <xnnpack/config.h>
#include <xnnpack/operator.h>
#include <xnnpack/operator-type.h>
#include <xnnpack/log.h>
#include <xnnpack/common.h>
#include <xnnpack/math.h>
#include <xnnpack/params.h>
#include <xnnpack/indirection.h>


static enum xnn_status create_rope_nthc(
    size_t max_tokens,
    uint32_t flags,
    enum xnn_operator_type operator_type,
    const struct xnn_cmul_config* config,
    xnn_operator_t* rope_op_out)
{
  xnn_operator_t rope_op = NULL;
  enum xnn_status status = xnn_status_uninitialized;

  if ((xnn_params.init_flags & XNN_INIT_FLAG_XNNPACK) == 0) {
    xnn_log_error("failed to create %s operator: XNNPACK is not initialized",
      xnn_operator_type_to_string(operator_type));
    goto error;
  }

  status = xnn_status_invalid_parameter;

  if (max_tokens == 0) {
    xnn_log_error(
      "failed to create %s operator with %zu max tokens: maximum number of tokens must be non-zero",
      xnn_operator_type_to_string(operator_type), max_tokens);
    goto error;
  }

  status = xnn_status_out_of_memory;

  rope_op = xnn_allocate_zero_simd_memory(sizeof(struct xnn_operator));
  if (rope_op == NULL) {
    xnn_log_error(
      "failed to allocate %zu bytes for %s operator descriptor",
      sizeof(struct xnn_operator), xnn_operator_type_to_string(operator_type));
    goto error;
  }

  rope_op->max_tokens = max_tokens;

  rope_op->type = operator_type;
  rope_op->flags = flags;
  rope_op->cmul_config = config;

  rope_op->state = xnn_run_state_invalid;

  *rope_op_out = rope_op;
  return xnn_status_success;

error:
  xnn_delete_operator(rope_op);
  return status;
}

enum xnn_status xnn_create_rope_nthc_f16(
  size_t max_tokens,
  uint32_t flags,
  xnn_operator_t* rope_op_out)
{
  const struct xnn_cmul_config* config = xnn_init_f16_cmul_config();
  if (config == NULL) {
    xnn_log_error("failed to create %s operator: unsupported hardware configuration",
                  xnn_operator_type_to_string(xnn_operator_type_rope_nthc_f16));
    return xnn_status_unsupported_hardware;
  }

  return create_rope_nthc(
    max_tokens,
    flags,
    xnn_operator_type_rope_nthc_f16,
    config,
    rope_op_out);
}

enum xnn_status xnn_create_rope_nthc_f32(
  size_t max_tokens,
  uint32_t flags,
  xnn_operator_t* rope_op_out)
{
  const struct xnn_cmul_config* config = xnn_init_f32_cmul_config();
  if (config == NULL) {
    xnn_log_error("failed to create %s operator: unsupported hardware configuration",
                  xnn_operator_type_to_string(xnn_operator_type_rope_nthc_f32));
    return xnn_status_unsupported_hardware;
  }

  return create_rope_nthc(
    max_tokens,
    flags,
    xnn_operator_type_rope_nthc_f32,
    config,
    rope_op_out);
}

static enum xnn_status reshape_rope_nthc(
    xnn_operator_t rope_op,
    enum xnn_operator_type expected_operator_type,
    size_t batch_size,
    size_t tokens,
    size_t heads,
    size_t channels,
    uint32_t log2_data_element_size,
    uint32_t log2_weight_element_size,
    size_t num_threads)
{
  if (rope_op->type != expected_operator_type) {
    xnn_log_error("failed to reshape operator: operator type mismatch (expected %s, got %s)",
      xnn_operator_type_to_string(expected_operator_type),
      xnn_operator_type_to_string(rope_op->type));
    return xnn_status_invalid_parameter;
  }
  rope_op->state = xnn_run_state_invalid;

  if (tokens == 0) {
    xnn_log_error(
      "failed to reshape %s operator with %zu tokens: number of tokens must be non-zero",
      xnn_operator_type_to_string(rope_op->type), tokens);
    return xnn_status_invalid_parameter;
  }

  if (tokens > rope_op->max_tokens) {
    xnn_log_error(
      "failed to reshape %s operator with %zu tokens: number of tokens can not exceed the maximum %zu",
      xnn_operator_type_to_string(rope_op->type), tokens, rope_op->max_tokens);
    return xnn_status_invalid_parameter;
  }

  if (heads == 0) {
    xnn_log_error(
      "failed to reshape %s operator with %zu heads: number of heads must be non-zero",
      xnn_operator_type_to_string(rope_op->type), heads);
    return xnn_status_invalid_parameter;
  }

  if (channels == 0) {
    xnn_log_error(
      "failed to reshape %s operator with %zu channels: number of channels must be non-zero",
      xnn_operator_type_to_string(rope_op->type), channels);
    return xnn_status_invalid_parameter;
  }

  if (channels % 2 != 0) {
    xnn_log_error(
      "failed to reshape %s operator with %zu channels: odd number of channels is not supported",
      xnn_operator_type_to_string(rope_op->type), channels);
    return xnn_status_unsupported_parameter;
  }

  if (batch_size == 0) {
    rope_op->state = xnn_run_state_skip;
    return xnn_status_success;
  }

  const struct xnn_cmul_config* config = rope_op->cmul_config;

  rope_op->context.rope = (struct rope_context) {
    .scaled_channels = (channels / 2) << log2_data_element_size,
    .batch_stride = (tokens * heads * channels) << log2_data_element_size,
    .head_stride = channels << log2_data_element_size,
    .sequence_stride = (heads * channels) << log2_data_element_size,
    .vcmul = config->ukernel,
  };

  rope_op->compute[0].type = xnn_parallelization_type_3d;
  rope_op->compute[0].task_3d = (pthreadpool_task_3d_t) xnn_compute_rope;
  rope_op->compute[0].range[0] = batch_size;
  rope_op->compute[0].range[1] = heads;
  rope_op->compute[0].range[2] = tokens;
  rope_op->state = xnn_run_state_needs_setup;

  return xnn_status_success;
}

enum xnn_status xnn_reshape_rope_nthc_f16(
    xnn_operator_t rope_op,
    size_t batch_size,
    size_t tokens,
    size_t heads,
    size_t channels,
    pthreadpool_t threadpool)
{
  return reshape_rope_nthc(
    rope_op, xnn_operator_type_rope_nthc_f16,
    batch_size, tokens, heads, channels,
    /*log2_data_element_size=*/XNN_LOG2_SIZEOF_HALF,
    /*log2_weight_element_size=*/XNN_LOG2_SIZEOF_HALF,
    pthreadpool_get_threads_count(threadpool));
}

enum xnn_status xnn_reshape_rope_nthc_f32(
    xnn_operator_t rope_op,
    size_t batch_size,
    size_t tokens,
    size_t heads,
    size_t channels,
    pthreadpool_t threadpool)
{
  return reshape_rope_nthc(
    rope_op, xnn_operator_type_rope_nthc_f32,
    batch_size, tokens, heads, channels,
    /*log2_data_element_size=*/XNN_LOG2_SIZEOF_FLOAT,
    /*log2_weight_element_size=*/XNN_LOG2_SIZEOF_FLOAT,
    pthreadpool_get_threads_count(threadpool));
}

static enum xnn_status setup_rope_nthc(
    xnn_operator_t rope_op,
    enum xnn_operator_type expected_operator_type,
    const void* input,
    const void* weights,
    void* output)
{
  if (rope_op->type != expected_operator_type) {
    xnn_log_error("failed to setup operator: operator type mismatch (expected %s, got %s)",
      xnn_operator_type_to_string(expected_operator_type),
      xnn_operator_type_to_string(rope_op->type));
    return xnn_status_invalid_parameter;
  }

  switch (rope_op->state) {
    case xnn_run_state_skip:
      return xnn_status_success;
    case xnn_run_state_invalid:
      xnn_log_error(
        "failed to setup %s operator: operator has not been reshaped yet",
        xnn_operator_type_to_string(rope_op->type));
      return xnn_status_invalid_state;
    case xnn_run_state_needs_setup:
      // Operator has been reshaped, but not setup, continue with setup.
    case xnn_run_state_ready:
      // Operator has been reshaped, and we are setting up with different pointers.
      break;
  }

  rope_op->context.rope.input = input;
  rope_op->context.rope.weights = weights;
  rope_op->context.rope.output = output;
  rope_op->state = xnn_run_state_ready;

  return xnn_status_success;
}

enum xnn_status xnn_setup_rope_nthc_f16(
  xnn_operator_t rope_op,
  const void* input,
  const void* weights,
  void* output)
{
  return setup_rope_nthc(
    rope_op, xnn_operator_type_rope_nthc_f16,
    input, weights, output);
}

enum xnn_status xnn_setup_rope_nthc_f32(
  xnn_operator_t rope_op,
  const float* input,
  const float* weights,
  float* output)
{
  return setup_rope_nthc(
    rope_op, xnn_operator_type_rope_nthc_f32,
    input, weights, output);
}
