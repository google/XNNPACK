// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <math.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include <fp16/fp16.h>

#include <xnnpack.h>
#include <xnnpack/allocator.h>
#include <xnnpack/common.h>
#include <xnnpack/compute.h>
#include <xnnpack/config.h>
#include <xnnpack/log.h>
#include <xnnpack/math.h>
#include <xnnpack/microparams.h>
#include <xnnpack/microparams-init.h>
#include <xnnpack/normalization.h>
#include <xnnpack/operator.h>
#include <xnnpack/operator-type.h>
#include <xnnpack/params.h>


static enum xnn_status create_mean_nd(
    uint32_t flags,
    uint32_t log2_element_size,
    enum xnn_operator_type operator_type,
    const struct xnn_gavgpool_config* gavgpool_config,
    const struct xnn_reduce_config* reduce_config,
    const void* params,
    size_t params_size,
    xnn_operator_t* mean_op_out)
{
  xnn_operator_t mean_op = NULL;
  enum xnn_status status = xnn_status_uninitialized;

  if ((xnn_params.init_flags & XNN_INIT_FLAG_XNNPACK) == 0) {
    xnn_log_error("failed to create %s operator: XNNPACK is not initialized",
      xnn_operator_type_to_string(operator_type));
    goto error;
  }

  status = xnn_status_out_of_memory;

  mean_op = xnn_allocate_zero_simd_memory(sizeof(struct xnn_operator));
  if (mean_op == NULL) {
    xnn_log_error(
      "failed to allocate %zu bytes for %s operator descriptor",
      sizeof(struct xnn_operator), xnn_operator_type_to_string(operator_type));
    goto error;
  }

  mean_op->type = operator_type;
  mean_op->flags = flags;
  mean_op->gavgpool_config = gavgpool_config;
  mean_op->reduce_config = reduce_config;
  if (params_size != 0) {
    memcpy(&mean_op->params, params, params_size);
  }

  mean_op->state = xnn_run_state_invalid;

  *mean_op_out = mean_op;
  return xnn_status_success;

error:
  xnn_delete_operator(mean_op);
  return status;
}

enum xnn_status xnn_create_mean_nd_f16(
  uint32_t flags,
  xnn_operator_t* mean_op_out)
{
  const struct xnn_gavgpool_config* gavgpool_config = xnn_init_f16_gavgpool_config();
  const struct xnn_reduce_config* rsum_config = xnn_init_f16_f32acc_rsum_config();
  if (gavgpool_config == NULL || rsum_config == NULL) {
    xnn_log_error("failed to create %s operator: unsupported hardware configuration",
                  xnn_operator_type_to_string(xnn_operator_type_mean_nd_f16));
    return xnn_status_unsupported_hardware;
  }

  struct {
    union xnn_f16_f32acc_scale_params scale;
    union xnn_f16_scaleminmax_params scaleminmax;
  } params;
  gavgpool_config->init.f16(&params.scaleminmax,
    /*scale=*/UINT16_C(0x3C00)  /* 1.0h */,
    /*output_min=*/UINT16_C(0xFC00)  /* -inf */,
    /*output_max=*/UINT16_C(0x7C00)  /* +inf */);
  rsum_config->init.f16_f32acc_scale(&params.scale,
    /*scale=*/1.0f);
  return create_mean_nd(
    flags,
    /*log2_element_size=*/XNN_LOG2_SIZEOF_HALF,
    xnn_operator_type_mean_nd_f16,
    gavgpool_config, rsum_config,
    &params, sizeof(params),
    mean_op_out);
}

enum xnn_status xnn_create_mean_nd_f32(
  uint32_t flags,
  xnn_operator_t* mean_op_out)
{
  const struct xnn_gavgpool_config* gavgpool_config = xnn_init_f32_gavgpool_config();
  const struct xnn_reduce_config* rsum_config = xnn_init_f32_rsum_config();
  if (gavgpool_config == NULL || rsum_config == NULL) {
    xnn_log_error("failed to create %s operator: unsupported hardware configuration",
                  xnn_operator_type_to_string(xnn_operator_type_mean_nd_f32));
    return xnn_status_unsupported_hardware;
  }

  struct {
    union xnn_f32_scale_params scale;
    union xnn_f32_scaleminmax_params scaleminmax;
  } params;
  gavgpool_config->init.f32(&params.scaleminmax,
    /*scale=*/1.0f, /*output_min=*/-INFINITY, /*output_max=*/INFINITY);
  rsum_config->init.f32_scale(&params.scale,
    /*scale=*/1.0f);
  return create_mean_nd(
    flags,
    /*log2_element_size=*/XNN_LOG2_SIZEOF_FLOAT,
    xnn_operator_type_mean_nd_f32,
    gavgpool_config, rsum_config,
    &params, sizeof(params),
    mean_op_out);
}

static enum xnn_status reshape_mean_nd(
    xnn_operator_t mean_op,
    size_t num_reduction_axes,
    const size_t* reduction_axes,
    size_t num_input_dims,
    const size_t* input_shape,
    size_t* workspace_size,
    size_t* workspace_alignment,
    size_t log2_data_element_size,
    size_t log2_accumulator_element_size,
    enum xnn_operator_type expected_operator_type,
    const void* scaleminmax_params,
    size_t scaleminmax_params_size,
    const void* scale_params,
    size_t scale_params_size,
    void (*update_params)(xnn_operator_t, size_t),
    pthreadpool_t threadpool)
{
  if (mean_op->type != expected_operator_type) {
    xnn_log_error("failed to reshape operator: operator type mismatch (expected %s, got %s)",
      xnn_operator_type_to_string(expected_operator_type),
      xnn_operator_type_to_string(mean_op->type));
    return xnn_status_invalid_parameter;
  }
  mean_op->state = xnn_run_state_invalid;

  if ((xnn_params.init_flags & XNN_INIT_FLAG_XNNPACK) == 0) {
    xnn_log_error("failed to reshape %s operator: XNNPACK is not initialized",
      xnn_operator_type_to_string(mean_op->type));
    return xnn_status_uninitialized;
  }

  if (num_input_dims > XNN_MAX_TENSOR_DIMS) {
    xnn_log_error(
      "failed to reshape %s operator with %zu input dimensions dimensions: "
      "the number of input dimensions must not exceed %d",
      xnn_operator_type_to_string(mean_op->type), num_input_dims, XNN_MAX_TENSOR_DIMS);
    return xnn_status_unsupported_parameter;
  }

  if (num_reduction_axes > num_input_dims) {
    xnn_log_error(
      "failed to reshape %s operator with %zu reduction axes: "
      "the number of reduction axes must not exceed the number of input dimensions %zu",
      xnn_operator_type_to_string(mean_op->type), num_reduction_axes, num_input_dims);
    return xnn_status_invalid_parameter;
  }

  if (num_reduction_axes == 0) {
    xnn_log_error(
      "failed to reshape %s operator with %zu reduction axes: the number of reduction axes must be non-zero",
      xnn_operator_type_to_string(mean_op->type), num_reduction_axes);
    return xnn_status_invalid_parameter;
  }

  for (size_t i = 0; i < num_reduction_axes; i++) {
    if (reduction_axes[i] > num_input_dims) {
      xnn_log_error(
        "failed to reshape %s operator with #%zu reduction axis of %zu: the index is out of bounds for a %zuD input shape",
        xnn_operator_type_to_string(mean_op->type), i, reduction_axes[i], num_input_dims);
      return xnn_status_invalid_parameter;
    }
  }

  size_t normalized_input_shape[XNN_MAX_TENSOR_DIMS];
  assert(num_input_dims <= XNN_MAX_TENSOR_DIMS);
  memcpy(normalized_input_shape, input_shape, num_input_dims * sizeof(size_t));

  size_t normalized_reduction_axes[XNN_MAX_TENSOR_DIMS];
  assert(num_reduction_axes <= XNN_MAX_TENSOR_DIMS);
  memcpy(normalized_reduction_axes, reduction_axes, num_reduction_axes * sizeof(size_t));

  xnn_normalize_reduction(
    &num_reduction_axes, normalized_reduction_axes,
    &num_input_dims, normalized_input_shape);

  if (num_reduction_axes != 1) {
    xnn_log_error(
      "failed to reshape %s operator with %zu normalized reduction axes: only a single post-normalization reduction axis is supported",
      xnn_operator_type_to_string(mean_op->type), num_reduction_axes);
    return xnn_status_invalid_parameter;
  }

  size_t num_input_elements = 1;
  for (size_t i = 0; i < num_input_dims; i++) {
    num_input_elements *= normalized_input_shape[i];
  }

  if (num_input_elements == 0) {
    mean_op->state = xnn_run_state_skip;
    return xnn_status_success;
  }

  const size_t reduction_axis = normalized_reduction_axes[0];
  const size_t axis_dim = normalized_input_shape[reduction_axis];
  const size_t batch_like_dim = reduction_axis == 0 ? 1 : normalized_input_shape[0];

  if (update_params != NULL) {
    update_params(mean_op, axis_dim);
  }

  *workspace_size = 0;
  *workspace_alignment = 1;

  if (reduction_axis + 1 == num_input_dims) {
    // Reduction along the innermost dimension

    mean_op->context.reduce = (struct reduce_context) {
        .input_stride = axis_dim << log2_data_element_size,
        .output_stride = UINT32_C(1) << log2_data_element_size,
        .scaled_elements = axis_dim << log2_data_element_size,
        .ukernel = mean_op->reduce_config->ukernel,
    };
    memcpy(&mean_op->context.reduce.params, scale_params, scale_params_size);

    mean_op->compute[0].task_1d_tile_1d = (pthreadpool_task_1d_tile_1d_t) xnn_compute_reduce;
    mean_op->compute[0].type = xnn_parallelization_type_1d_tile_1d;
    mean_op->compute[0].range[0] = batch_like_dim;
    mean_op->compute[0].tile[0] = 2;
    mean_op->ukernel.type = xnn_microkernel_type_mean;
  } else {
    // Reduction along the non-innermost dimension
    const size_t channel_like_dim = normalized_input_shape[num_input_dims - 1];

    if (mean_op->channels != channel_like_dim) {
      const size_t zero_size = (channel_like_dim << log2_data_element_size) + XNN_EXTRA_BYTES;
      // Note: zero buffer must be SIMD-aligned, so we can't use xnn_reallocate_memory
      xnn_release_simd_memory(mean_op->zero_buffer);
      mean_op->zero_buffer = xnn_allocate_zero_simd_memory(zero_size);
      if (mean_op->zero_buffer == NULL) {
        xnn_log_error(
          "failed to allocate %zu bytes for %s operator zero padding",
          zero_size, xnn_operator_type_to_string(mean_op->type));
        return xnn_status_out_of_memory;
      }
      mean_op->channels = channel_like_dim;
    }

    mean_op->context.global_average_pooling_nwc = (struct global_average_pooling_nwc_context) {
        .zero = mean_op->zero_buffer,
        .input_pixel_stride = channel_like_dim << log2_data_element_size,
        .input_batch_stride = (channel_like_dim * axis_dim) << log2_data_element_size,
        .input_elements = axis_dim,
        .channels = channel_like_dim,
        .output_batch_stride = channel_like_dim << log2_data_element_size,
    };
    memcpy(&mean_op->context.global_average_pooling_nwc.params, scaleminmax_params, scaleminmax_params_size);

    mean_op->compute[0].range[0] = batch_like_dim;
    if (axis_dim <= mean_op->gavgpool_config->row_tile) {
      mean_op->compute[0].type = xnn_parallelization_type_1d;
      mean_op->compute[0].task_1d = (pthreadpool_task_1d_t) xnn_compute_global_average_pooling_nwc_unipass;
      mean_op->context.global_average_pooling_nwc.unipass_ukernel = mean_op->gavgpool_config->unipass;
    } else {
      const size_t multipass_batch_stride = round_up_po2(
          (channel_like_dim + (XNN_MAX_SIMD_SIZE >> log2_data_element_size)) << log2_accumulator_element_size,
          XNN_ALLOCATION_ALIGNMENT);
      const size_t num_threads = pthreadpool_get_threads_count(threadpool);
      *workspace_size = num_threads * multipass_batch_stride;
      *workspace_alignment = XNN_ALLOCATION_ALIGNMENT;
      mean_op->context.global_average_pooling_nwc.multipass_batch_stride = multipass_batch_stride;
      mean_op->compute[0].type = xnn_parallelization_type_1d_with_thread;
      mean_op->compute[0].task_1d_with_thread =
        (pthreadpool_task_1d_with_thread_t) xnn_compute_global_average_pooling_nwc_multipass_with_thread;
      mean_op->context.global_average_pooling_nwc.multipass_ukernel = mean_op->gavgpool_config->multipass;
    }
    mean_op->ukernel.type = xnn_microkernel_type_global_average_pooling;
  }
  mean_op->state = xnn_run_state_needs_setup;

  return xnn_status_success;
}

static void update_params_mean_f16(
  xnn_operator_t mean_op,
  size_t num_elements)
{
  const float scale = 1.0f / (float) (double) num_elements;
  mean_op->reduce_config->init.f16_f32acc_scale(&mean_op->params.f16_f32acc_scale, scale);
  mean_op->gavgpool_config->update.f16(&mean_op->params.f16_scale_minmax, fp16_ieee_from_fp32_value(scale));
}

enum xnn_status xnn_reshape_mean_nd_f16(
    xnn_operator_t mean_op,
    size_t num_reduction_axes,
    const size_t* reduction_axes,
    size_t num_input_dims,
    const size_t* input_shape,
    size_t* workspace_size,
    size_t* workspace_alignment,
    pthreadpool_t threadpool)
{
  return reshape_mean_nd(
    mean_op,
    num_reduction_axes, reduction_axes,
    num_input_dims, input_shape,
    workspace_size, workspace_alignment,
    /*log2_data_element_size=*/XNN_LOG2_SIZEOF_HALF,
    /*log2_accumulator_element_size=*/XNN_LOG2_SIZEOF_HALF,
    xnn_operator_type_mean_nd_f16,
    /*scaleminmax_params=*/&mean_op->params.f16_scale_minmax,
    /*scaleminmax_params_size=*/sizeof(mean_op->params.f16_scale_minmax),
    /*scale_params=*/&mean_op->params.f16_f32acc_scale,
    /*scale_params_size=*/sizeof(mean_op->params.f16_f32acc_scale),
    update_params_mean_f16,
    threadpool);
}

static void update_params_mean_f32(
  xnn_operator_t mean_op,
  size_t num_elements)
{
  const float scale = 1.0f / (float) (double) num_elements;
  mean_op->reduce_config->init.f32_scale(&mean_op->params.f32_scale, scale);
  mean_op->gavgpool_config->update.f32(&mean_op->params.f32_scale_minmax, scale);
}

enum xnn_status xnn_reshape_mean_nd_f32(
    xnn_operator_t mean_op,
    size_t num_reduction_axes,
    const size_t* reduction_axes,
    size_t num_input_dims,
    const size_t* input_shape,
    size_t* workspace_size,
    size_t* workspace_alignment,
    pthreadpool_t threadpool)
{
  return reshape_mean_nd(
    mean_op,
    num_reduction_axes, reduction_axes,
    num_input_dims, input_shape,
    workspace_size, workspace_alignment,
    /*log2_data_element_size=*/XNN_LOG2_SIZEOF_FLOAT,
    /*log2_accumulator_element_size=*/XNN_LOG2_SIZEOF_FLOAT,
    xnn_operator_type_mean_nd_f32,
    /*scaleminmax_params=*/&mean_op->params.f32_scale_minmax,
    /*scaleminmax_params_size=*/sizeof(mean_op->params.f32_scale_minmax),
    /*scale_params=*/&mean_op->params.f32_scale,
    /*scale_params_size=*/sizeof(mean_op->params.f32_scale),
    update_params_mean_f32,
    threadpool);
}

static enum xnn_status setup_mean_nd(
    xnn_operator_t mean_op,
    void* workspace,
    const float* input,
    float* output,
    enum xnn_operator_type expected_operator_type)
{
  if (mean_op->type != expected_operator_type) {
    xnn_log_error("failed to setup operator: operator type mismatch (expected %s, got %s)",
      xnn_operator_type_to_string(expected_operator_type),
      xnn_operator_type_to_string(mean_op->type));
    return xnn_status_invalid_parameter;
  }

  switch (mean_op->state) {
    case xnn_run_state_skip:
      return xnn_status_success;
    case xnn_run_state_invalid:
      xnn_log_error(
        "failed to setup %s operator: operator has not been reshaped yet",
        xnn_operator_type_to_string(mean_op->type));
      return xnn_status_invalid_state;
    case xnn_run_state_needs_setup:
      // Operator has been reshaped, but not setup, continue with setup.
    case xnn_run_state_ready:
      // Operator has been reshaped, and we are setting up with different pointers.
      break;
  }

  if (mean_op->ukernel.type == xnn_microkernel_type_mean) {
    // Reduction along the innermost dimension
    mean_op->context.reduce.input = input;
    mean_op->context.reduce.output = output;
  } else {
    assert(mean_op->ukernel.type == xnn_microkernel_type_global_average_pooling);
    mean_op->context.global_average_pooling_nwc.input = input;
    mean_op->context.global_average_pooling_nwc.output = output;

    if (mean_op->context.global_average_pooling_nwc.multipass_batch_stride != 0 && workspace == NULL) {
      xnn_log_error(
        "failed to setup %s operator: workspace of size %zu required but workspace is NULL",
        xnn_operator_type_to_string(mean_op->type),
        mean_op->context.global_average_pooling_nwc.multipass_batch_stride);
    }
    mean_op->context.global_average_pooling_nwc.multipass_buffer = workspace;
  }
  mean_op->state = xnn_run_state_ready;

  return xnn_status_success;
}

enum xnn_status xnn_setup_mean_nd_f16(
    xnn_operator_t mean_op,
    void* workspace,
    const void* input,
    void* output)
{
  return setup_mean_nd(
    mean_op,
    workspace,
    input, output,
    xnn_operator_type_mean_nd_f16);
}

enum xnn_status xnn_setup_mean_nd_f32(
    xnn_operator_t mean_op,
    void* workspace,
    const float* input,
    float* output)
{
  return setup_mean_nd(
    mean_op,
    workspace,
    input, output,
    xnn_operator_type_mean_nd_f32);
}
