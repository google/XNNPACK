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

#include "xnnpack.h"
#include "xnnpack/allocator.h"
#include "xnnpack/common.h"
#include "xnnpack/compute.h"
#include "xnnpack/config-types.h"
#include "xnnpack/config.h"
#include "xnnpack/log.h"
#include "xnnpack/microkernel-type.h"
#include "xnnpack/microparams.h"
#include "xnnpack/normalization.h"
#include "xnnpack/operator-type.h"
#include "xnnpack/operator.h"
#include "xnnpack/params.h"
#include "pthreadpool.h"

static enum xnn_status create_mean_nd(
    uint32_t flags,
    uint32_t log2_element_size,
    enum xnn_operator_type operator_type,
    const struct xnn_reduce_config* rdsum_config,
    const struct xnn_reduce_config* rsum_config,
    const struct xnn_unary_elementwise_config* cvt_config,
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
  mean_op->rdsum_config = rdsum_config;
  mean_op->rsum_config = rsum_config;
  mean_op->cvt_config = cvt_config;
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
  const struct xnn_reduce_config* rsum_config = xnn_init_f16_f32acc_rsum_config();
  const struct xnn_reduce_config* rdsum_config = xnn_init_f16_f32acc_rdsum_config();
  const struct xnn_unary_elementwise_config* f32_to_f16_cvt_config = xnn_init_f32_to_f16_cvt_config();
  if (rdsum_config == NULL || rsum_config == NULL || f32_to_f16_cvt_config == NULL) {
    xnn_log_error("failed to create %s operator: unsupported hardware configuration",
                  xnn_operator_type_to_string(xnn_operator_type_mean_nd_f16));
    return xnn_status_unsupported_hardware;
  }
  struct f16_f32acc_mean_params params;
  rsum_config->init.f16_f32acc_scale(&params.f16_f32acc_scale, /*scale=*/1.0f);
  return create_mean_nd(
    flags,
    /*log2_element_size=*/XNN_LOG2_SIZEOF_HALF,
    xnn_operator_type_mean_nd_f16,
    rdsum_config, rsum_config, f32_to_f16_cvt_config,
    &params, sizeof(params),
    mean_op_out);
}

enum xnn_status xnn_create_mean_nd_f32(
  uint32_t flags,
  xnn_operator_t* mean_op_out)
{
  const struct xnn_reduce_config* rsum_config = xnn_init_f32_rsum_config();
  const struct xnn_reduce_config* rdsum_config = xnn_init_f32_rdsum_config();
  if (rdsum_config == NULL || rsum_config == NULL) {
    xnn_log_error("failed to create %s operator: unsupported hardware configuration",
                  xnn_operator_type_to_string(xnn_operator_type_mean_nd_f32));
    return xnn_status_unsupported_hardware;
  }

  union xnn_f32_scaleminmax_params params;
  rsum_config->init.f32_scaleminmax(&params, /*scale=*/1.0f, /*min=*/-INFINITY, /*max=*/INFINITY);
  return create_mean_nd(
    flags,
    /*log2_element_size=*/XNN_LOG2_SIZEOF_FLOAT,
    xnn_operator_type_mean_nd_f32,
    rdsum_config, rsum_config, /*cvt_config=*/NULL,
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

  for (size_t i = 1; i < num_reduction_axes; i++) {
    if (reduction_axes[i] <= reduction_axes[i - 1]) {
      xnn_log_error(
        "failed to reshape %s operator with #%zu reduction axis of %zu: the reduction "
        "axes must be in ascending order and unique",
        xnn_operator_type_to_string(mean_op->type), i, reduction_axes[i]);
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

  size_t num_input_elements = 1;
  for (size_t i = 0; i < num_input_dims; i++) {
    num_input_elements *= normalized_input_shape[i];
  }

  if (num_input_elements == 0) {
    mean_op->state = xnn_run_state_skip;
    return xnn_status_success;
  }

  memmove(&normalized_input_shape[XNN_MAX_TENSOR_DIMS - num_input_dims], &normalized_input_shape[0], sizeof(size_t) * num_input_dims);
  for (int i = 0; i < XNN_MAX_TENSOR_DIMS - num_input_dims; ++i) {
    normalized_input_shape[i] = 1;
  }
  mean_op->compute[0].type = xnn_parallelization_type_3d_tile_2d;
  mean_op->ukernel.type = xnn_microkernel_type_mean;
  // Reduction along the innermost dimension.
  if (workspace_alignment != NULL) {
    *workspace_alignment = XNN_ALLOCATION_ALIGNMENT;
  }
  if (normalized_reduction_axes[num_reduction_axes - 1] == num_input_dims - 1) {
    for (int i = 0; i < XNN_MAX_TENSOR_DIMS; ++i) {
    }
    if (workspace_size != NULL) {
      const size_t num_output_elements = normalized_input_shape[0] * normalized_input_shape[2] * normalized_input_shape[4];
      *workspace_size = num_output_elements << log2_accumulator_element_size;
    }
    const size_t scale_dim = normalized_input_shape[1] * normalized_input_shape[3] * normalized_input_shape[5];
    const size_t axis_dim = normalized_input_shape[5];

    if (update_params != NULL) {
      update_params(mean_op, scale_dim);
    }

    mean_op->context.reduce = (struct reduce_context) {
      .channels = axis_dim << log2_data_element_size,
      .ukernel.rsum = mean_op->rsum_config->ukernel,
      .accumulation_element_size = UINT32_C(1) << log2_accumulator_element_size,
      .output_element_size = UINT32_C(1) << log2_data_element_size,
    };
    memcpy(&mean_op->context.reduce.params, scale_params, scale_params_size);

    mean_op->compute[0].task_3d_tile_2d = (pthreadpool_task_3d_tile_2d_t) xnn_compute_contiguous_reduce;
    mean_op->compute[0].range[0] = normalized_input_shape[0];
    mean_op->compute[0].range[1] = normalized_input_shape[2];
    mean_op->compute[0].range[2] = normalized_input_shape[4];
    mean_op->compute[0].tile[0] = 1;
    mean_op->compute[0].tile[1] = 2;
    mean_op->context.reduce.output_stride[XNN_MAX_TENSOR_DIMS / 2 - 1] = 1;
    for (int i = XNN_MAX_TENSOR_DIMS / 2 -  2; i >= 0; --i) {
      mean_op->context.reduce.output_stride[i] =  (mean_op->context.reduce.output_stride[i + 1] * normalized_input_shape[(i + 1) * 2]);
    }
  } else {
    // Reduction along the non-innermost dimension
    const size_t channel_like_dim = normalized_input_shape[XNN_MAX_TENSOR_DIMS - 1];
    if (workspace_size != NULL) {
      const size_t num_output_elements = normalized_input_shape[1] * normalized_input_shape[3] * normalized_input_shape[5];
      *workspace_size = num_output_elements << log2_accumulator_element_size;
    }
    const size_t scale_dim = normalized_input_shape[0] * normalized_input_shape[2] * normalized_input_shape[4];
    const size_t axis_dim = normalized_input_shape[4];

    if (update_params != NULL) {
      update_params(mean_op, scale_dim);
    }
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

    mean_op->context.reduce = (struct reduce_context) {
      .zero = mean_op->zero_buffer,
      .channels = axis_dim,
      .ukernel.rdsum = mean_op->rdsum_config->rd_ukernel,
      .accumulation_element_size = UINT32_C(1) << log2_accumulator_element_size,
      .output_element_size = UINT32_C(1) << log2_data_element_size,
    };
    memcpy(&mean_op->context.reduce.params, scale_params, scale_params_size);
    mean_op->compute[0].task_3d_tile_2d = (pthreadpool_task_3d_tile_2d_t) xnn_compute_discontiguous_reduce;
    mean_op->compute[0].range[0] = normalized_input_shape[1];
    mean_op->compute[0].range[1] = normalized_input_shape[3];
    mean_op->compute[0].range[2] = normalized_input_shape[5];
    mean_op->compute[0].tile[0] = 1;
    mean_op->compute[0].tile[1] = normalized_input_shape[5];
    mean_op->context.reduce.output_stride[XNN_MAX_TENSOR_DIMS / 2 - 1] = 1;
    for (int i = XNN_MAX_TENSOR_DIMS / 2 -  2; i >= 0; --i) {
      mean_op->context.reduce.output_stride[i] = (mean_op->context.reduce.output_stride[i + 1] * normalized_input_shape[(i * 2+3)]);
    }
  }
  mean_op->context.reduce.input_stride[XNN_MAX_TENSOR_DIMS - 1] = (1 << log2_data_element_size);
  if (mean_op->cvt_config) {
    mean_op->context.reduce.cvt_ukernel = mean_op->cvt_config->ukernel;
  }
  for (int i = XNN_MAX_TENSOR_DIMS -  2; i >= 0; --i) {
    mean_op->context.reduce.input_stride[i] =  (mean_op->context.reduce.input_stride[i + 1] * normalized_input_shape[i + 1]);
  }
  memcpy(mean_op->context.reduce.input_shape, normalized_input_shape, XNN_MAX_TENSOR_DIMS * sizeof(size_t));
  mean_op->state = xnn_run_state_needs_setup;

  return xnn_status_success;
}

static void update_params_mean_f16(
  xnn_operator_t mean_op,
  size_t num_elements)
{
  const float scale = 1.0f / (float) (double) num_elements;
  mean_op->rsum_config->init.f16_f32acc_scale(&mean_op->params.mean_params.f16_f32acc_scale, scale);
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
    /*log2_accumulator_element_size=*/XNN_LOG2_SIZEOF_FLOAT,
    xnn_operator_type_mean_nd_f16,
    /*scale_params=*/&mean_op->params.mean_params.f16_f32acc_scale,
    /*scale_params_size=*/sizeof(mean_op->params.mean_params.f16_f32acc_scale),
    update_params_mean_f16,
    threadpool);
}

static void update_params_mean_f32(
  xnn_operator_t mean_op,
  size_t num_elements)
{
  const float scale = 1.0f / (float) (double) num_elements;
  mean_op->rsum_config->init.f32_scaleminmax(&mean_op->params.f32_scaleminmax, scale, -INFINITY, INFINITY);
  mean_op->rdsum_config->init.f32_scaleminmax(&mean_op->params.f32_scaleminmax, scale, -INFINITY, INFINITY);
}

enum xnn_status xnn_reshape_mean_nd_f32(
    xnn_operator_t mean_op,
    size_t num_reduction_axes,
    const size_t* reduction_axes,
    size_t num_input_dims,
    const size_t* input_shape,
    pthreadpool_t threadpool)
{
  return reshape_mean_nd(
    mean_op,
    num_reduction_axes, reduction_axes,
    num_input_dims, input_shape,
    /*workspace_size=*/NULL, /*workspace_alignment=*/NULL,
    /*log2_data_element_size=*/XNN_LOG2_SIZEOF_FLOAT,
    /*log2_accumulator_element_size=*/XNN_LOG2_SIZEOF_FLOAT,
    xnn_operator_type_mean_nd_f32,
    /*scale_params=*/&mean_op->params.f32_scaleminmax,
    /*scale_params_size=*/sizeof(mean_op->params.f32_scaleminmax),
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

  mean_op->context.reduce.input = input;
  mean_op->context.reduce.output = output;
  mean_op->context.reduce.workspace = workspace;
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
    workspace, input, output,
    xnn_operator_type_mean_nd_f16);
}

enum xnn_status xnn_setup_mean_nd_f32(
    xnn_operator_t mean_op,
    const float* input,
    float* output)
{
  return setup_mean_nd(
    mean_op,
    /*workspace=*/NULL, input, output,
    xnn_operator_type_mean_nd_f32);
}
