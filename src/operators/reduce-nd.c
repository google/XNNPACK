// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <inttypes.h>
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
#include "xnnpack/datatype.h"
#include "xnnpack/log.h"
#include "xnnpack/microkernel-type.h"
#include "xnnpack/microparams.h"
#include "xnnpack/normalization.h"
#include "xnnpack/operator-type.h"
#include "xnnpack/operator.h"
#include "xnnpack/params.h"
#include "pthreadpool.h"

static enum xnn_status create_reduce_nd(
    uint32_t flags,
    uint32_t log2_data_element_size,
    uint32_t log2_accumulator_element_size,
    enum xnn_operator_type operator_type,
    const struct xnn_reduce_config* rdsum_config,
    const struct xnn_reduce_config* rsum_config,
    const struct xnn_unary_elementwise_config* cvt_config,
    const struct xnn_unary_elementwise_config* s32_f32_cvt_config,
    const struct xnn_unary_elementwise_config* u32_f32_cvt_config,
    const void* params,
    size_t params_size,
    xnn_operator_t* reduce_op_out)
{
  xnn_operator_t reduce_op = NULL;
  enum xnn_status status = xnn_status_uninitialized;

  if ((xnn_params.init_flags & XNN_INIT_FLAG_XNNPACK) == 0) {
    xnn_log_error("failed to create %s operator: XNNPACK is not initialized",
      xnn_operator_type_to_string(operator_type));
    goto error;
  }

  status = xnn_status_out_of_memory;

  reduce_op = xnn_allocate_zero_simd_memory(sizeof(struct xnn_operator));
  if (reduce_op == NULL) {
    xnn_log_error(
      "failed to allocate %zu bytes for %s operator descriptor",
      sizeof(struct xnn_operator), xnn_operator_type_to_string(operator_type));
    goto error;
  }

  reduce_op->type = operator_type;
  reduce_op->flags = flags;
  reduce_op->rdsum_config = rdsum_config;
  reduce_op->rsum_config = rsum_config;
  reduce_op->cvt_config = cvt_config;
  reduce_op->s32_f32_cvt_config = s32_f32_cvt_config;
  reduce_op->u32_f32_cvt_config = u32_f32_cvt_config;
  reduce_op->reduce.log2_data_element_size = log2_data_element_size;
  reduce_op->reduce.log2_accumulator_element_size = log2_accumulator_element_size;
  if (params_size != 0) {
    memcpy(&reduce_op->params, params, params_size);
  }

  reduce_op->state = xnn_run_state_invalid;

  *reduce_op_out = reduce_op;
  return xnn_status_success;

error:
  xnn_delete_operator(reduce_op);
  return status;
}

static int cmp_value_size_t(const void* a_ptr, const void* b_ptr) {
  const size_t a = *((const size_t*)a_ptr);
  const size_t b = *((const size_t*)b_ptr);
  return (b < a) - (b > a);
}

static enum xnn_status reshape_reduce_nd(
    xnn_operator_t reduce_op, size_t num_reduction_axes,
    const int64_t* reduction_axes, size_t num_input_dims,
    const size_t* input_shape, size_t* workspace_size,
    size_t* workspace_alignment,
    enum xnn_operator_type expected_operator_type,
    pthreadpool_t threadpool) {
  if (reduce_op->type != expected_operator_type) {
    xnn_log_error("failed to reshape operator: operator type mismatch (expected %s, got %s)",
      xnn_operator_type_to_string(expected_operator_type),
      xnn_operator_type_to_string(reduce_op->type));
    return xnn_status_invalid_parameter;
  }
  reduce_op->state = xnn_run_state_invalid;

  if ((xnn_params.init_flags & XNN_INIT_FLAG_XNNPACK) == 0) {
    xnn_log_error("failed to reshape %s operator: XNNPACK is not initialized",
      xnn_operator_type_to_string(reduce_op->type));
    return xnn_status_uninitialized;
  }

  if (num_input_dims > XNN_MAX_TENSOR_DIMS) {
    xnn_log_error(
      "failed to reshape %s operator with %zu input dimensions dimensions: "
      "the number of input dimensions must not exceed %d",
      xnn_operator_type_to_string(reduce_op->type), num_input_dims, XNN_MAX_TENSOR_DIMS);
    return xnn_status_unsupported_parameter;
  }

  if (num_reduction_axes > num_input_dims) {
    xnn_log_error(
      "failed to reshape %s operator with %zu reduction axes: "
      "the number of reduction axes must not exceed the number of input dimensions %zu",
      xnn_operator_type_to_string(reduce_op->type), num_reduction_axes, num_input_dims);
    return xnn_status_invalid_parameter;
  }

  if (num_reduction_axes == 0) {
    xnn_log_error(
      "failed to reshape %s operator with %zu reduction axes: the number of reduction axes must be non-zero",
      xnn_operator_type_to_string(reduce_op->type), num_reduction_axes);
    return xnn_status_invalid_parameter;
  }

  size_t normalized_input_shape[XNN_MAX_TENSOR_DIMS];
  assert(num_input_dims <= XNN_MAX_TENSOR_DIMS);
  memcpy(normalized_input_shape, input_shape, num_input_dims * sizeof(size_t));

  for (size_t i = 0; i < num_reduction_axes; i++) {
    const int64_t signed_num_input_dims = (int64_t)num_input_dims;
    if (signed_num_input_dims <= reduction_axes[i] ||
        reduction_axes[i] < -signed_num_input_dims) {
      xnn_log_error(
          "failed to reshape %s operator with #%zu reduction axis of %" PRIi64
          ": the index is out of bounds for a %zuD input shape",
          xnn_operator_type_to_string(reduce_op->type), i, reduction_axes[i],
          num_input_dims);
      return xnn_status_invalid_parameter;
    }
  }

  size_t normalized_reduction_axes[XNN_MAX_TENSOR_DIMS];
  assert(num_reduction_axes <= XNN_MAX_TENSOR_DIMS);
  for (int i = 0; i < num_reduction_axes; i++) {
    normalized_reduction_axes[i] = 0 <= reduction_axes[i]
                                       ? reduction_axes[i]
                                       : num_input_dims + reduction_axes[i];
  }
  qsort(normalized_reduction_axes, num_reduction_axes, sizeof(size_t),
        cmp_value_size_t);

  for (size_t i = 1; i < num_reduction_axes; i++) {
    if (normalized_reduction_axes[i] <= normalized_reduction_axes[i - 1]) {
      xnn_log_error(
          "failed to reshape %s operator with #%zu reduction axis of %" PRIi64
          ": the reduction axes must be unique",
          xnn_operator_type_to_string(reduce_op->type), i, reduction_axes[i]);
      return xnn_status_invalid_parameter;
    }
  }

  xnn_normalize_reduction(
    &num_reduction_axes, normalized_reduction_axes,
    &num_input_dims, normalized_input_shape);

  size_t num_input_elements = 1;
  for (size_t i = 0; i < num_input_dims; i++) {
    num_input_elements *= normalized_input_shape[i];
  }

  if (num_input_elements == 0) {
    reduce_op->state = xnn_run_state_skip;
    return xnn_status_success;
  }

  memmove(&normalized_input_shape[XNN_MAX_TENSOR_DIMS - num_input_dims], &normalized_input_shape[0], sizeof(size_t) * num_input_dims);
  for (int i = 0; i < XNN_MAX_TENSOR_DIMS - num_input_dims; ++i) {
    normalized_input_shape[i] = 1;
  }
  const uint32_t log2_data_element_size = reduce_op->reduce.log2_data_element_size;
  const uint32_t log2_accumulator_element_size = reduce_op->reduce.log2_accumulator_element_size;
  reduce_op->compute[0].type = xnn_parallelization_type_3d_tile_2d;
  reduce_op->ukernel.type = xnn_microkernel_type_mean;
  // Reduction along the innermost dimension.
  if (workspace_alignment != NULL) {
    *workspace_alignment = XNN_ALLOCATION_ALIGNMENT;
  }
  if (normalized_reduction_axes[num_reduction_axes - 1] == num_input_dims - 1) {
    if (workspace_size != NULL) {
      const size_t num_output_elements = normalized_input_shape[0] * normalized_input_shape[2] * normalized_input_shape[4];
      *workspace_size = (num_output_elements << log2_accumulator_element_size) + XNN_EXTRA_BYTES;
    }
    const size_t scale_dim = normalized_input_shape[1] * normalized_input_shape[3] * normalized_input_shape[5];
    const size_t axis_dim = normalized_input_shape[5];

    if (reduce_op->rsum_config->update != NULL) {
      float scale = 1.0f;
      if (reduce_op->type == xnn_operator_type_mean_nd) {
        scale = 1.0f / scale_dim;
      }
      reduce_op->rsum_config->update(&reduce_op->params.reduce, scale, scale_dim);
    }

    reduce_op->context.reduce = (struct reduce_context) {
      .channels = axis_dim << log2_data_element_size,
      .ukernel.rsum = reduce_op->rsum_config->ukernel,
      .accumulation_element_size = UINT32_C(1) << log2_accumulator_element_size,
      .output_element_size = UINT32_C(1) << log2_data_element_size,
    };
    memcpy(&reduce_op->context.reduce.params, &reduce_op->params.reduce, sizeof(reduce_op->params.reduce));

    reduce_op->compute[0].task_3d_tile_2d = (pthreadpool_task_3d_tile_2d_t) xnn_compute_contiguous_reduce;
    reduce_op->compute[0].range[0] = normalized_input_shape[0];
    reduce_op->compute[0].range[1] = normalized_input_shape[2];
    reduce_op->compute[0].range[2] = normalized_input_shape[4];
    reduce_op->compute[0].tile[0] = 1;
    reduce_op->compute[0].tile[1] = 2;
    reduce_op->context.reduce.output_stride[XNN_MAX_TENSOR_DIMS / 2 - 1] = 1;
    for (int i = XNN_MAX_TENSOR_DIMS / 2 -  2; i >= 0; --i) {
      reduce_op->context.reduce.output_stride[i] =  (reduce_op->context.reduce.output_stride[i + 1] * normalized_input_shape[(i + 1) * 2]);
    }

    if (reduce_op->s32_f32_cvt_config) {
      reduce_op->context.reduce.s32_f32_cvt_ukernel = reduce_op->s32_f32_cvt_config->ukernel;
    }
    if (reduce_op->u32_f32_cvt_config) {
      reduce_op->context.reduce.u32_f32_cvt_ukernel = reduce_op->u32_f32_cvt_config->ukernel;
    }
  } else {
    // Reduction along the non-innermost dimension
    const size_t channel_like_dim = normalized_input_shape[XNN_MAX_TENSOR_DIMS - 1];
    if (workspace_size != NULL) {
      const size_t num_output_elements = normalized_input_shape[1] * normalized_input_shape[3] * normalized_input_shape[5];
      *workspace_size = (num_output_elements << log2_accumulator_element_size) + XNN_EXTRA_BYTES;
    }
    const size_t scale_dim = normalized_input_shape[0] * normalized_input_shape[2] * normalized_input_shape[4];
    const size_t axis_dim = normalized_input_shape[4];

    if (reduce_op->rdsum_config->update != NULL) {
      float scale = 1.0f;
      if (reduce_op->type == xnn_operator_type_mean_nd) {
        scale = 1.0f / scale_dim;
      }
      reduce_op->rdsum_config->update(&reduce_op->params.reduce, scale, scale_dim);
    }
    if (reduce_op->channels != channel_like_dim) {
      const size_t zero_size = (channel_like_dim << log2_data_element_size) + XNN_EXTRA_BYTES;
      // Note: zero buffer must be SIMD-aligned, so we can't use xnn_reallocate_memory
      xnn_release_simd_memory(reduce_op->zero_buffer);
      reduce_op->zero_buffer = xnn_allocate_zero_simd_memory(zero_size);
      if (reduce_op->zero_buffer == NULL) {
        xnn_log_error(
          "failed to allocate %zu bytes for %s operator zero padding",
          zero_size, xnn_operator_type_to_string(reduce_op->type));
        return xnn_status_out_of_memory;
      }
      reduce_op->channels = channel_like_dim;
    }

    reduce_op->context.reduce = (struct reduce_context) {
      .zero = reduce_op->zero_buffer,
      .channels = axis_dim,
      .ukernel.rdsum = reduce_op->rdsum_config->rd_ukernel,
      .accumulation_element_size = UINT32_C(1) << log2_accumulator_element_size,
      .output_element_size = UINT32_C(1) << log2_data_element_size,
    };
    memcpy(&reduce_op->context.reduce.params, &reduce_op->params.reduce, sizeof(reduce_op->params.reduce));
    reduce_op->compute[0].task_3d_tile_2d = (pthreadpool_task_3d_tile_2d_t) xnn_compute_discontiguous_reduce;
    reduce_op->compute[0].range[0] = normalized_input_shape[1];
    reduce_op->compute[0].range[1] = normalized_input_shape[3];
    reduce_op->compute[0].range[2] = normalized_input_shape[5];
    reduce_op->compute[0].tile[0] = 1;
    reduce_op->compute[0].tile[1] = normalized_input_shape[5];
    reduce_op->context.reduce.output_stride[XNN_MAX_TENSOR_DIMS / 2 - 1] = 1;
    for (int i = XNN_MAX_TENSOR_DIMS / 2 -  2; i >= 0; --i) {
      reduce_op->context.reduce.output_stride[i] = (reduce_op->context.reduce.output_stride[i + 1] * normalized_input_shape[(i * 2+3)]);
    }
  }
  reduce_op->context.reduce.input_stride[XNN_MAX_TENSOR_DIMS - 1] = (1 << log2_data_element_size);
  if (reduce_op->cvt_config) {
    reduce_op->context.reduce.cvt_ukernel = reduce_op->cvt_config->ukernel;
  }
  if (reduce_op->s32_f32_cvt_config) {
    reduce_op->context.reduce.s32_f32_cvt_ukernel = reduce_op->s32_f32_cvt_config->ukernel;
  }
  if (reduce_op->u32_f32_cvt_config) {
    reduce_op->context.reduce.u32_f32_cvt_ukernel = reduce_op->u32_f32_cvt_config->ukernel;
  }
  for (int i = XNN_MAX_TENSOR_DIMS -  2; i >= 0; --i) {
    reduce_op->context.reduce.input_stride[i] =  (reduce_op->context.reduce.input_stride[i + 1] * normalized_input_shape[i + 1]);
  }
  memcpy(reduce_op->context.reduce.input_shape, normalized_input_shape, XNN_MAX_TENSOR_DIMS * sizeof(size_t));
  reduce_op->state = xnn_run_state_needs_setup;

  return xnn_status_success;
}

static enum xnn_status setup_reduce_nd(
    xnn_operator_t reduce_op,
    void* workspace,
    const float* input,
    float* output,
    enum xnn_operator_type expected_operator_type)
{
  if (reduce_op->type != expected_operator_type) {
    xnn_log_error("failed to setup operator: operator type mismatch (expected %s, got %s)",
      xnn_operator_type_to_string(expected_operator_type),
      xnn_operator_type_to_string(reduce_op->type));
    return xnn_status_invalid_parameter;
  }

  switch (reduce_op->state) {
    case xnn_run_state_skip:
      return xnn_status_success;
    case xnn_run_state_invalid:
      xnn_log_error(
        "failed to setup %s operator: operator has not been reshaped yet",
        xnn_operator_type_to_string(reduce_op->type));
      return xnn_status_invalid_state;
    case xnn_run_state_needs_setup:
      // Operator has been reshaped, but not setup, continue with setup.
    case xnn_run_state_ready:
      // Operator has been reshaped, and we are setting up with different pointers.
      break;
  }

  reduce_op->context.reduce.input = input;
  reduce_op->context.reduce.output = output;
  reduce_op->context.reduce.workspace = workspace;
  reduce_op->state = xnn_run_state_ready;

  return xnn_status_success;
}

enum xnn_status xnn_create_reduce_nd(
    const enum xnn_reduce_operator reduce_operator_type,
    const enum xnn_datatype datatype,
    const struct xnn_quantization_params* input_quantization,
    const struct xnn_quantization_params* output_quantization,
    uint32_t flags,
    xnn_operator_t* reduce_op_out)
{
  enum xnn_operator_type operator_type = xnn_reduce_operator_to_operator_type(reduce_operator_type);

  // The config initialization functions return NULL on error. Because each
  // datatype uses a different combination of functions, we use an `unused`
  // sentinel value (!= NULL) to have one shared check for successful
  // configuration. The unsued pointers must then be reset to NULL before
  // calling `create_reduce_nd`.
  const struct xnn_unary_elementwise_config* unused = (void*) -1;

  // Load configs.
  const struct xnn_reduce_config* rsum_config = NULL;
  const struct xnn_reduce_config* rdsum_config = NULL;
  const struct xnn_unary_elementwise_config* cvt_config = NULL;
  const struct xnn_unary_elementwise_config* s32_f32_cvt_config = NULL;
  const struct xnn_unary_elementwise_config* u32_f32_cvt_config = NULL;
  uint32_t log2_data_element_size = xnn_datatype_log2_size_bytes(datatype);
  uint32_t log2_accumulator_element_size;
  switch(datatype) {
    case xnn_datatype_fp16: {
      log2_accumulator_element_size = 2;
      rsum_config = xnn_init_f16_f32acc_rsum_config();
      rdsum_config = xnn_init_f16_f32acc_rdsum_config();
      cvt_config = xnn_init_f32_to_f16_cvt_config();
      s32_f32_cvt_config = unused;
      u32_f32_cvt_config = unused;
      break;
    }
    case xnn_datatype_fp32: {
      log2_accumulator_element_size = 2;
      rsum_config = xnn_init_f32_rsum_config();
      rdsum_config = xnn_init_f32_rdsum_config();
      cvt_config = unused;
      s32_f32_cvt_config = unused;
      u32_f32_cvt_config = unused;
      break;
    }
    case xnn_datatype_qint8: { // qs8
      log2_accumulator_element_size = 2;
      rsum_config = xnn_init_qs8_rsum_config();
      rdsum_config = xnn_init_qs8_rdsum_config();
      cvt_config = xnn_init_f32_to_qs8_cvt_config();
      s32_f32_cvt_config = xnn_init_s32_to_f32_cvt_config();
      u32_f32_cvt_config = unused;
      break;
    }
    case xnn_datatype_quint8: { // qu8
      log2_accumulator_element_size = 2;
      rsum_config = xnn_init_qu8_rsum_config();
      rdsum_config = xnn_init_qu8_rdsum_config();
      cvt_config = xnn_init_f32_to_qu8_cvt_config();
      s32_f32_cvt_config = unused;
      // We just use an int32 -> f32 conversion. This means we effectively only
      // have a 31-bit accumulator instead of 32-bit, but that seems insignificant.
      u32_f32_cvt_config = xnn_init_s32_to_f32_cvt_config();
      break;
    }
    default:
      xnn_log_error("failed to create Sum (ND) operator: unsupported data type: %s",
                  xnn_datatype_to_string(datatype));
      return xnn_status_invalid_parameter;
  };

  // Check configs and restore unused pointers to NULL.
  if (rdsum_config == NULL || rsum_config == NULL || cvt_config == NULL ||
      s32_f32_cvt_config == NULL || u32_f32_cvt_config == NULL) {
    xnn_log_error(
        "failed to create %s (%s) operator: unsupported hardware configuration",
        xnn_operator_type_to_string(operator_type), xnn_datatype_to_string(datatype));
    return xnn_status_unsupported_hardware;
  } else {
    cvt_config = cvt_config == unused ? NULL : cvt_config;
    s32_f32_cvt_config = s32_f32_cvt_config == unused ? NULL : s32_f32_cvt_config;
    u32_f32_cvt_config = u32_f32_cvt_config == unused ? NULL : u32_f32_cvt_config;
  }

  struct xnn_reduce_params params;
  size_t params_size = 0;
  // Setup parameters
  if (rsum_config->init) {
    params_size = rsum_config->init(&params, input_quantization, output_quantization);
  }

  return create_reduce_nd(
    flags, log2_data_element_size, log2_accumulator_element_size, operator_type,
    rdsum_config, rsum_config, cvt_config, s32_f32_cvt_config,
    u32_f32_cvt_config, &params, params_size, reduce_op_out);
}

enum xnn_status xnn_reshape_reduce_nd(
    xnn_operator_t reduce_op,
    size_t num_reduction_axes, const int64_t* reduction_axes,
    size_t num_input_dims, const size_t* input_shape, size_t* workspace_size,
    size_t* workspace_alignment, pthreadpool_t threadpool) {
  return reshape_reduce_nd(
      reduce_op, num_reduction_axes, reduction_axes, num_input_dims, input_shape,
      workspace_size, workspace_alignment,
      reduce_op->type,
      threadpool);
}

enum xnn_status xnn_setup_reduce_nd(
    xnn_operator_t reduce_op,
    void* workspace,
    const void* input,
    void* output)
{
  return setup_reduce_nd(
    reduce_op,
    workspace, input, output,
    reduce_op->type);
}
