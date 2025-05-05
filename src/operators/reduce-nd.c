// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <math.h>
#include <inttypes.h>
#include <limits.h>
#include <stddef.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#include "include/xnnpack.h"
#include "src/xnnpack/allocator.h"
#include "src/xnnpack/common.h"
#include "src/xnnpack/compute.h"
#include "src/xnnpack/config-types.h"
#include "src/xnnpack/config.h"
#include "src/xnnpack/operator-utils.h"
#include "src/xnnpack/reference-config.h"
#include "src/xnnpack/datatype.h"
#include "src/xnnpack/log.h"
#include "src/xnnpack/microkernel-type.h"
#include "src/xnnpack/microparams.h"
#include "src/xnnpack/normalization.h"
#include "src/xnnpack/operator-type.h"
#include "src/xnnpack/operator.h"
#include "src/xnnpack/params.h"
#include <pthreadpool.h>

static enum xnn_status create_reduce_nd(
    uint32_t flags,
    uint32_t log2_data_element_size,
    uint32_t log2_accumulator_element_size,
    enum xnn_operator_type operator_type,
    const struct xnn_reduce_config* contiguous_reduce_config,
    const struct xnn_reduce_config* discontiguous_reduce_config,
    const struct xnn_xx_fill_config* fill_config,
    const struct xnn_unary_elementwise_config* cvt_config,
    const void* params,
    size_t params_size,
    const void* cvt_params,
    size_t cvt_params_size,
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
  reduce_op->discontiguous_reduce_config = discontiguous_reduce_config;
  reduce_op->contiguous_reduce_config = contiguous_reduce_config;
  reduce_op->cvt_config = cvt_config;
  reduce_op->fill_config = fill_config;
  reduce_op->reduce.log2_data_element_size = log2_data_element_size;
  reduce_op->reduce.log2_accumulator_element_size = log2_accumulator_element_size;
  reduce_op->reduce.identity_value =
      reduce_op->contiguous_reduce_config->identity_value;

  if (params_size != 0) {
    memcpy(&reduce_op->params, params, params_size);
  }
  if (cvt_params_size != 0) {
    memcpy(&reduce_op->params2, cvt_params, cvt_params_size);
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
    xnn_log_error(
        "failed to reshape operator: operator type mismatch (expected %s, got "
        "%s)",
        xnn_operator_type_to_string(expected_operator_type),
        xnn_operator_type_to_string_v2(reduce_op));
    return xnn_status_invalid_parameter;
  }
  reduce_op->state = xnn_run_state_invalid;

  if ((xnn_params.init_flags & XNN_INIT_FLAG_XNNPACK) == 0) {
    xnn_log_error("failed to reshape %s operator: XNNPACK is not initialized",
                  xnn_operator_type_to_string_v2(reduce_op));
    return xnn_status_uninitialized;
  }

  if (num_input_dims > XNN_MAX_TENSOR_DIMS) {
    xnn_log_error(
        "failed to reshape %s operator with %zu input dimensions dimensions: "
        "the number of input dimensions must not exceed %d",
        xnn_operator_type_to_string_v2(reduce_op), num_input_dims,
        XNN_MAX_TENSOR_DIMS);
    return xnn_status_unsupported_parameter;
  }

  if (num_reduction_axes > num_input_dims) {
    xnn_log_error(
        "failed to reshape %s operator with %zu reduction axes: "
        "the number of reduction axes must not exceed the number of input "
        "dimensions %zu",
        xnn_operator_type_to_string_v2(reduce_op), num_reduction_axes,
        num_input_dims);
    return xnn_status_invalid_parameter;
  }

  if (num_reduction_axes == 0) {
    xnn_log_error(
        "failed to reshape %s operator with %zu reduction axes: the number of "
        "reduction axes must be non-zero",
        xnn_operator_type_to_string_v2(reduce_op), num_reduction_axes);
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
          xnn_operator_type_to_string_v2(reduce_op), i, reduction_axes[i],
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
          xnn_operator_type_to_string_v2(reduce_op), i, reduction_axes[i]);
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
  reduce_op->ukernel.type = xnn_microkernel_type_reduce;
  // Reduction along the innermost dimension.
  if (workspace_alignment != NULL) {
    *workspace_alignment = XNN_ALLOCATION_ALIGNMENT;
  }

  const bool is_minmax = (reduce_op->type == xnn_operator_type_reduce_max_nd ||
                          reduce_op->type == xnn_operator_type_reduce_min_nd);

  size_t num_reduction_elements;
  if (normalized_reduction_axes[num_reduction_axes - 1] == num_input_dims - 1) {
    if (workspace_size != NULL) {
      const size_t num_output_elements = normalized_input_shape[0] * normalized_input_shape[2] * normalized_input_shape[4];
      *workspace_size = (num_output_elements << log2_accumulator_element_size) + XNN_EXTRA_BYTES;
    }
    num_reduction_elements = normalized_input_shape[1] * normalized_input_shape[3] * normalized_input_shape[5];
    const size_t axis_dim = normalized_input_shape[5];

    if (reduce_op->contiguous_reduce_config->update != NULL) {
      float scale = 1.0f;
      if (reduce_op->type == xnn_operator_type_mean_nd) {
        scale = 1.0f / num_reduction_elements;
      }
      reduce_op->contiguous_reduce_config->update(&reduce_op->params.reduce,
                                                  scale);
    }

    reduce_op->context.reduce = (struct reduce_context) {
      .channels = axis_dim << log2_data_element_size,
      .accumulation_element_size = UINT32_C(1) << log2_accumulator_element_size,
      .output_element_size = UINT32_C(1) << log2_data_element_size,
      .identity_value = reduce_op->reduce.identity_value,
      .ukernel.contiguous_reduce = reduce_op->contiguous_reduce_config->ukernel,
    };

    if (is_minmax) {
      reduce_op->context.reduce.fill_ukernel = reduce_op->fill_config->ukernel;
    }

    reduce_op->compute[0].task_3d_tile_2d = (pthreadpool_task_3d_tile_2d_t) xnn_compute_contiguous_reduce;
    reduce_op->compute[0].range[0] = normalized_input_shape[0];
    reduce_op->compute[0].range[1] = normalized_input_shape[2];
    reduce_op->compute[0].range[2] = normalized_input_shape[4];
    reduce_op->compute[0].tile[0] = 1;
    reduce_op->compute[0].tile[1] = 2;
    reduce_op->context.reduce.output_stride[XNN_MAX_TENSOR_DIMS / 2 - 1] = 1;
    for (int i = XNN_MAX_TENSOR_DIMS / 2 -  2; i >= 0; --i) {
      reduce_op->context.reduce.output_stride[i] = (reduce_op->context.reduce.output_stride[i + 1] * normalized_input_shape[(i + 1) * 2]);
    }
  } else {
    // Reduction along the non-innermost dimension
    const size_t channel_like_dim = normalized_input_shape[XNN_MAX_TENSOR_DIMS - 1];
    if (workspace_size != NULL) {
      const size_t num_output_elements = normalized_input_shape[1] * normalized_input_shape[3] * normalized_input_shape[5];
      *workspace_size = (num_output_elements << log2_accumulator_element_size) + XNN_EXTRA_BYTES;
    }
    num_reduction_elements = normalized_input_shape[0] * normalized_input_shape[2] * normalized_input_shape[4];
    const size_t axis_dim = normalized_input_shape[4];

    if (reduce_op->discontiguous_reduce_config->update != NULL) {
      float scale = 1.0f;
      if (reduce_op->type == xnn_operator_type_mean_nd) {
        scale = 1.0f / num_reduction_elements;
      }
      reduce_op->discontiguous_reduce_config->update(&reduce_op->params.reduce,
                                                     scale);
    }
    if (reduce_op->channels != channel_like_dim) {
      const size_t zero_size = (channel_like_dim << log2_data_element_size) + XNN_EXTRA_BYTES;
      // Note: zero buffer must be SIMD-aligned, so we can't use xnn_reallocate_memory
      xnn_release_simd_memory(reduce_op->zero_buffer);
      reduce_op->zero_buffer = xnn_allocate_zero_simd_memory(zero_size);
      if (reduce_op->zero_buffer == NULL) {
        xnn_log_error(
            "failed to allocate %zu bytes for %s operator zero padding",
            zero_size, xnn_operator_type_to_string_v2(reduce_op));
        return xnn_status_out_of_memory;
      }
      reduce_op->channels = channel_like_dim;
    }

    reduce_op->context.reduce = (struct reduce_context) {
      .zero = reduce_op->zero_buffer,
      .channels = axis_dim,
      .ukernel.discontiguous_reduce =
          reduce_op->discontiguous_reduce_config->rd_ukernel,
      .accumulation_element_size = UINT32_C(1) << log2_accumulator_element_size,
      .output_element_size = UINT32_C(1) << log2_data_element_size,
      .identity_value = reduce_op->reduce.identity_value,
    };

    if (is_minmax) {
      reduce_op->context.reduce.fill_ukernel = reduce_op->fill_config->ukernel;
    }

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
  memcpy(&reduce_op->context.reduce.params, &reduce_op->params.reduce, sizeof(reduce_op->params.reduce));
  memcpy(&reduce_op->context.reduce.cvt_params, &reduce_op->params2.unary, sizeof(reduce_op->params2.unary));
  reduce_op->context.reduce.input_stride[XNN_MAX_TENSOR_DIMS - 1] = (1 << log2_data_element_size);
  if (reduce_op->cvt_config) {
    reduce_op->context.reduce.cvt_ukernel = reduce_op->cvt_config->ukernel;
    // int32 is not actually a quantized type, so we need to include the input
    // zero point (multiplied by the number of reduction elements) as part of
    // the computation of the output zero point.
    // The conversion normally looks like:
    //
    //   y = (x - x_zero_point) * x_scale * inv_y_scale + y_zero_point
    //
    // Since this conversion ignores x_zero_point and x_scale, rewrite to:
    //
    //   y = x * x_scale * inv_y_scale - x_zero_point * x_scale * inv_y_scale + y_zero_point
    //
    // Now we can say:
    //
    //   inv_y_scale' = x_scale * inv_y_scale
    //   y_zero_point' = y_zero_point - x_zero_point * x_scale * inv_y_scale
    reduce_op->context.reduce.cvt_params.reference.inv_y_scale =
        reduce_op->context.reduce.params.qs8.scale;
    reduce_op->context.reduce.cvt_params.reference.y_zero_point -=
        ((int32_t) num_reduction_elements *
        reduce_op->context.reduce.cvt_params.reference.x_zero_point) *
        reduce_op->context.reduce.cvt_params.reference.inv_y_scale;
  }
  for (int i = XNN_MAX_TENSOR_DIMS - 2; i >= 0; --i) {
    reduce_op->context.reduce.input_stride[i] = (reduce_op->context.reduce.input_stride[i + 1] * normalized_input_shape[i + 1]);
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
    xnn_log_error(
        "failed to setup operator: operator type mismatch (expected %s, got "
        "%s)",
        xnn_operator_type_to_string(expected_operator_type),
        xnn_operator_type_to_string_v2(reduce_op));
    return xnn_status_invalid_parameter;
  }

  switch (reduce_op->state) {
    case xnn_run_state_skip:
      return xnn_status_success;
    case xnn_run_state_invalid:
      xnn_log_error(
          "failed to setup %s operator: operator has not been reshaped yet",
          xnn_operator_type_to_string_v2(reduce_op));
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
  const struct xnn_unary_elementwise_config* cvt_unused = (void*) -1;
  const struct xnn_xx_fill_config* fill_unused = (void*) -1;

  const bool is_minmax = (operator_type == xnn_operator_type_reduce_max_nd ||
                          operator_type == xnn_operator_type_reduce_min_nd);

  // Load configs.
  const struct xnn_reduce_config* contiguous_config = NULL;
  const struct xnn_reduce_config* discontiguous_config = NULL;
  const struct xnn_unary_elementwise_config* cvt_config = NULL;
  const struct xnn_xx_fill_config* fill_config = NULL;
  uint32_t log2_data_element_size = xnn_datatype_log2_size_bytes(datatype);
  uint32_t log2_accumulator_element_size;
  switch (datatype) {
    case xnn_datatype_fp16: {
      if (is_minmax) {
        log2_accumulator_element_size = 1;
        fill_config = xnn_init_xx_fill_config();
        cvt_config = cvt_unused;

        if (operator_type == xnn_operator_type_reduce_min_nd) {
          contiguous_config = xnn_init_f16_rmin_config();
          discontiguous_config = xnn_init_f16_rdmin_config();
        } else {  // max
          contiguous_config = xnn_init_f16_rmax_config();
          discontiguous_config = xnn_init_f16_rdmax_config();
        }
      } else {
        log2_accumulator_element_size = 2;
        contiguous_config = xnn_init_f16_f32acc_rsum_config();
        discontiguous_config = xnn_init_f16_f32acc_rdsum_config();
        fill_config = fill_unused;
        cvt_config = xnn_init_f32_to_f16_cvt_config();
      }
      break;
    }
    case xnn_datatype_fp32: {
      if (is_minmax) {
        fill_config = xnn_init_xx_fill_config();

        if (operator_type == xnn_operator_type_reduce_min_nd) {
          contiguous_config = xnn_init_f32_rmin_config();
          discontiguous_config = xnn_init_f32_rdmin_config();
        } else {  // max
          contiguous_config = xnn_init_f32_rmax_config();
          discontiguous_config = xnn_init_f32_rdmax_config();
        }
      } else {
        contiguous_config = xnn_init_f32_rsum_config();
        discontiguous_config = xnn_init_f32_rdsum_config();
        fill_config = fill_unused;
      }

      log2_accumulator_element_size = 2;
      cvt_config = cvt_unused;
      break;
    }
    case xnn_datatype_qint8: { // qs8
      if (is_minmax) {
        assert(input_quantization->scale == output_quantization->scale);
        assert(
          input_quantization->zero_point == output_quantization->zero_point);
        log2_accumulator_element_size = 0;
        fill_config = xnn_init_xx_fill_config();
        cvt_config = cvt_unused;

        if (operator_type == xnn_operator_type_reduce_min_nd) {
          contiguous_config = xnn_init_s8_rmin_config();
          discontiguous_config = xnn_init_s8_rdmin_config();
        } else {  // max
          contiguous_config = xnn_init_s8_rmax_config();
          discontiguous_config = xnn_init_s8_rdmax_config();
        }
      } else {
        log2_accumulator_element_size = 2;
        contiguous_config = xnn_init_qs8_rsum_config();
        discontiguous_config = xnn_init_qs8_rdsum_config();
        fill_config = fill_unused;
        cvt_config = xnn_init_unary_reference_config(
          xnn_unary_convert, xnn_datatype_int32, xnn_datatype_qint8);
      }
      break;
    }
    case xnn_datatype_quint8: { // qu8
      if (is_minmax) {
        assert(input_quantization->scale == output_quantization->scale);
        assert(
          input_quantization->zero_point == output_quantization->zero_point);
        log2_accumulator_element_size = 0;
        fill_config = xnn_init_xx_fill_config();
        cvt_config = cvt_unused;

        if (operator_type == xnn_operator_type_reduce_min_nd) {
          contiguous_config = xnn_init_u8_rmin_config();
          discontiguous_config = xnn_init_u8_rdmin_config();
        } else {  // max
          contiguous_config = xnn_init_u8_rmax_config();
          discontiguous_config = xnn_init_u8_rdmax_config();
        }
      } else {
        log2_accumulator_element_size = 2;
        contiguous_config = xnn_init_qu8_rsum_config();
        discontiguous_config = xnn_init_qu8_rdsum_config();
        // We just use an int32 -> qu8 conversion. This means we effectively
        // only have a 31-bit accumulator instead of 32-bit, but that seems
        // insignificant.
        cvt_config = xnn_init_unary_reference_config(
          xnn_unary_convert, xnn_datatype_int32, xnn_datatype_quint8);
        fill_config = fill_unused;
      }
      break;
    }
    default:
      xnn_log_error("failed to create Sum (ND) operator: unsupported data type: %s",
                  xnn_datatype_to_string(datatype));
      return xnn_status_invalid_parameter;
  };

  // Check configs and restore unused pointers to NULL.
  if (contiguous_config == NULL || discontiguous_config == NULL ||
      fill_config == NULL || cvt_config == NULL) {
    xnn_log_error(
        "failed to create %s (%s) operator: unsupported hardware configuration",
        xnn_operator_type_to_string(operator_type), xnn_datatype_to_string(datatype));
    return xnn_status_unsupported_hardware;
  } else {
    fill_config = fill_config == fill_unused ? NULL : fill_config;
    cvt_config = cvt_config == cvt_unused ? NULL : cvt_config;
  }

  struct xnn_reduce_params params;
  size_t params_size = 0;
  // Setup parameters
  if (contiguous_config->init.reduce) {
    params_size = contiguous_config->init.reduce(&params, input_quantization,
                                                 output_quantization);
  }
  union xnn_unary_uparams cvt_params;
  size_t cvt_params_size = 0;
  if (cvt_config && cvt_config->init) {
    cvt_params_size = cvt_config->init(&cvt_params, NULL, input_quantization, output_quantization);
  }

  return create_reduce_nd(
    flags, log2_data_element_size, log2_accumulator_element_size, operator_type,
    contiguous_config, discontiguous_config, fill_config, cvt_config, &params,
    params_size, &cvt_params, cvt_params_size, reduce_op_out);
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
