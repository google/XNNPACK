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
#include <xnnpack/common.h>
#include <xnnpack/config.h>
#include <xnnpack/cache.h>
#include <xnnpack/log.h>
#include <xnnpack/operator.h>
#include <xnnpack/operator-type.h>
#include <xnnpack/operator-utils.h>
#include <xnnpack/pack.h>
#include <xnnpack/microparams-init.h>
#include <xnnpack/params.h>


static enum xnn_status create_prelu_nc(
    size_t channels,
    size_t input_stride,
    size_t output_stride,
    const void* negative_slope,
    uint32_t flags,
    uint32_t log2_weights_element_size,
    xnn_pack_prelu_w_fn pack_prelu_w,
    enum xnn_operator_type operator_type,
    const struct xnn_prelu_config* prelu_config,
    xnn_code_cache_t code_cache,
    xnn_weights_cache_t weights_cache,
    xnn_operator_t* prelu_op_out)
{
  xnn_operator_t prelu_op = NULL;
  enum xnn_status status = xnn_status_uninitialized;

  if ((xnn_params.init_flags & XNN_INIT_FLAG_XNNPACK) == 0) {
    xnn_log_error("failed to setup %s operator: XNNPACK is not initialized",
      xnn_operator_type_to_string(operator_type));
    return xnn_status_uninitialized;
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

  prelu_op->weights_cache = weights_cache;

  const size_t packed_weights_size = (channels << log2_weights_element_size) + XNN_EXTRA_BYTES;
  const size_t aligned_total_weights_size = round_up_po2(packed_weights_size, XNN_ALLOCATION_ALIGNMENT);
  void* weights_ptr = xnn_get_pointer_to_write_weights(prelu_op, aligned_total_weights_size, 0);
  xnn_log_debug("allocated %zu bytes for packed weights in %s operator",
    aligned_total_weights_size, xnn_operator_type_to_string(operator_type));

  pack_prelu_w(channels, negative_slope, weights_ptr);

  if (use_weights_cache(prelu_op)) {
    struct xnn_weights_cache_look_up_key cache_key;
    cache_key.seed = murmur_hash3(weights_ptr, aligned_total_weights_size, /*seed=*/7);
    cache_key.kernel = negative_slope;
    cache_key.bias = NULL;
    prelu_op->packed_weights.offset = xnn_look_up_or_insert_weights_cache(
        prelu_op->weights_cache, &cache_key, weights_ptr, aligned_total_weights_size);
  }

  prelu_op->channels = channels;
  prelu_op->input_pixel_stride = input_stride;
  prelu_op->output_pixel_stride = output_stride;

  prelu_op->type = operator_type;
  prelu_op->flags = flags;
  prelu_op->prelu_config = prelu_config;

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
    xnn_code_cache_t code_cache,
    xnn_weights_cache_t weights_cache,
    xnn_operator_t* prelu_op_out)
{
  xnn_pack_prelu_w_fn pack_prelu_w = (xnn_pack_prelu_w_fn) xnn_pack_f16_prelu_w;
  if (flags & XNN_FLAG_FP32_STATIC_WEIGHTS) {
    pack_prelu_w = (xnn_pack_prelu_w_fn) xnn_pack_f32_to_f16_prelu_w;
  }

  const struct xnn_prelu_config* prelu_config = xnn_init_f16_prelu_config();
  if (prelu_config == NULL) {
    xnn_log_error("failed to create %s operator: unsupported hardware configuration",
                  xnn_operator_type_to_string(xnn_operator_type_prelu_nc_f16));
    return xnn_status_unsupported_hardware;
  }

  return create_prelu_nc(
    channels, input_stride, output_stride,
    negative_slope, flags,
    /*log2_weights_element_size=*/XNN_LOG2_SIZEOF_HALF,
    pack_prelu_w,
    xnn_operator_type_prelu_nc_f16,
    prelu_config,
    /*code_cache=*/code_cache,
    /*weights_cache=*/weights_cache,
    prelu_op_out);
}

enum xnn_status xnn_create_prelu_nc_f32(
    size_t channels,
    size_t input_stride,
    size_t output_stride,
    const float* negative_slope,
    uint32_t flags,
    xnn_code_cache_t code_cache,
    xnn_weights_cache_t weights_cache,
    xnn_operator_t* prelu_op_out)
{
  const struct xnn_prelu_config* prelu_config = xnn_init_f32_prelu_config();
  if (prelu_config == NULL) {
    xnn_log_error("failed to create %s operator: unsupported hardware configuration",
                  xnn_operator_type_to_string(xnn_operator_type_prelu_nc_f32));
    return xnn_status_unsupported_hardware;
  }

  return create_prelu_nc(
    channels, input_stride, output_stride,
    negative_slope, flags,
    /*log2_weights_element_size=*/XNN_LOG2_SIZEOF_FLOAT,
    (xnn_pack_prelu_w_fn) xnn_pack_f32_prelu_w,
    xnn_operator_type_prelu_nc_f32,
    prelu_config,
    /*code_cache=*/code_cache,
    /*weights_cache=*/weights_cache,
    prelu_op_out);
}

static enum xnn_status reshape_prelu_nc(
    xnn_operator_t prelu_op,
    enum xnn_operator_type expected_operator_type,
    size_t batch_size,
    uint32_t log2_element_size,
    pthreadpool_t threadpool)
{
  if (prelu_op->type != expected_operator_type) {
    xnn_log_error("failed to reshape operator: operator type mismatch (expected %s, got %s)",
      xnn_operator_type_to_string(expected_operator_type),
      xnn_operator_type_to_string(prelu_op->type));
    return xnn_status_invalid_parameter;
  }
  prelu_op->state = xnn_run_state_invalid;

  if ((xnn_params.init_flags & XNN_INIT_FLAG_XNNPACK) == 0) {
    xnn_log_error("failed to reshape %s operator: XNNPACK is not initialized",
      xnn_operator_type_to_string(expected_operator_type));
    return xnn_status_uninitialized;
  }

  if (batch_size == 0) {
    prelu_op->state = xnn_run_state_skip;
    return xnn_status_success;
  }

  const struct xnn_prelu_config* prelu = prelu_op->prelu_config;

  const size_t channels = prelu_op->channels;
  prelu_op->context.prelu = (struct prelu_context) {
    .n = channels << log2_element_size,
    .x_stride = prelu_op->input_pixel_stride << log2_element_size,
    .w = packed_weights(prelu_op),
    .y_stride = prelu_op->output_pixel_stride << log2_element_size,
    .ukernel = prelu->ukernel,
  };

  #if XNN_TEST_MODE
    const size_t batch_tile = prelu->row_tile;
  #else
    size_t batch_tile = batch_size;
    const size_t num_threads = pthreadpool_get_threads_count(threadpool);
    if (num_threads > 1) {
      const size_t target_tiles_per_thread = 5;
      const size_t max_batch_tile = divide_round_up(batch_size, num_threads * target_tiles_per_thread);
      if (max_batch_tile < batch_tile) {
        const uint32_t row_tile = prelu->row_tile;
        batch_tile = min(batch_tile, divide_round_up(batch_tile, max_batch_tile * row_tile) * row_tile);
      }
    }
  #endif
  prelu_op->compute[0].type = xnn_parallelization_type_1d_tile_1d;
  prelu_op->compute[0].task_1d_tile_1d = (pthreadpool_task_1d_tile_1d_t) xnn_compute_prelu;
  prelu_op->compute[0].range[0] = batch_size;
  prelu_op->compute[0].tile[0] = batch_tile;
  prelu_op->state = xnn_run_state_needs_setup;

  return xnn_status_success;
}

enum xnn_status xnn_reshape_prelu_nc_f16(
    xnn_operator_t prelu_op,
    size_t batch_size,
    pthreadpool_t threadpool)
{
  return reshape_prelu_nc(
    prelu_op, xnn_operator_type_prelu_nc_f16,
    batch_size,
    /*log2_element_size=*/XNN_LOG2_SIZEOF_HALF,
    threadpool);
}

enum xnn_status xnn_reshape_prelu_nc_f32(
    xnn_operator_t prelu_op,
    size_t batch_size,
    pthreadpool_t threadpool)
{
  return reshape_prelu_nc(
    prelu_op, xnn_operator_type_prelu_nc_f32,
    batch_size,
    /*log2_element_size=*/XNN_LOG2_SIZEOF_FLOAT,
    threadpool);
}

static enum xnn_status setup_prelu_nc(
    xnn_operator_t prelu_op,
    enum xnn_operator_type expected_operator_type,
    const float* input,
    float* output)
{
  if (prelu_op->type != expected_operator_type) {
    xnn_log_error("failed to setup operator: operator type mismatch (expected %s, got %s)",
      xnn_operator_type_to_string(expected_operator_type),
      xnn_operator_type_to_string(prelu_op->type));
    return xnn_status_invalid_parameter;
  }

  if (prelu_op->weights_cache != NULL && !xnn_weights_cache_is_finalized(prelu_op->weights_cache)) {
    xnn_log_error("failed to setup %s operator: weights cache is not finalized",
      xnn_operator_type_to_string(expected_operator_type));
    return xnn_status_invalid_state;
  }

  switch (prelu_op->state) {
    case xnn_run_state_skip:
      return xnn_status_success;
    case xnn_run_state_invalid:
      xnn_log_error(
        "failed to setup %s operator: operator has not been reshaped yet",
        xnn_operator_type_to_string(prelu_op->type));
      return xnn_status_invalid_state;
    case xnn_run_state_needs_setup:
      // Operator has been reshaped, but not setup, continue with setup.
    case xnn_run_state_ready:
      // Operator has been reshaped, and we are setting up with different pointers.
      break;
  }

  prelu_op->context.prelu.x = input;
  prelu_op->context.prelu.y = output;
  prelu_op->state = xnn_run_state_ready;

  return xnn_status_success;
}

enum xnn_status xnn_setup_prelu_nc_f16(
    xnn_operator_t prelu_op,
    const void* input,
    void* output)
{
  return setup_prelu_nc(
    prelu_op, xnn_operator_type_prelu_nc_f16,
    input, output);
}

enum xnn_status xnn_setup_prelu_nc_f32(
    xnn_operator_t prelu_op,
    const float* input,
    float* output)
{
  return setup_prelu_nc(
    prelu_op, xnn_operator_type_prelu_nc_f32,
    input, output);
}
