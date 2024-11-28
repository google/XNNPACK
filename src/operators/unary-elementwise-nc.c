// Copyright 2020 Google LLC
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
#include "xnnpack/datatype.h"
#include "xnnpack/log.h"
#include "xnnpack/math.h"
#include "xnnpack/microfnptr.h"
#include "xnnpack/microparams.h"
#include "xnnpack/node-type.h"
#include "xnnpack/operator-type.h"
#include "xnnpack/operator-utils.h"
#include "xnnpack/operator.h"
#include "xnnpack/params.h"
#include "xnnpack/reference-config.h"
#include "pthreadpool.h"

static enum xnn_status check_op_type(xnn_operator_t op,
                                     enum xnn_operator_type expected_type) {
  if (op->type != expected_type) {
    xnn_log_error(
        "failed to setup operator: operator type mismatch (expected %s, got "
        "%s)",
        xnn_operator_type_to_string(expected_type),
        xnn_operator_type_to_string(op->type));
    return xnn_status_invalid_parameter;
  }
  return xnn_status_success;
}

static enum xnn_status init_lut_op(
    xnn_operator_t op,
    const struct xnn_unary_elementwise_config* reference_config,
    const union xnn_unary_params* params,
    const struct xnn_quantization_params* input_quantization,
    const struct xnn_quantization_params* output_quantization) {
  const int lookup_table_elements = 256;
  op->lookup_table = xnn_allocate_simd_memory(lookup_table_elements * sizeof(uint8_t));
  if (op->lookup_table == NULL) {
    xnn_log_error(
      "failed to allocate %s operator lookup table",
      xnn_operator_type_to_string(op->type));
    return xnn_status_out_of_memory;
  }

  union xnn_unary_uparams uparams;
  if (reference_config->init) {
    reference_config->init(&uparams, params, input_quantization, output_quantization);
  }

  // Run the reference kernel on the lookup table itself to initialize it.
  uint8_t* lookup_table = op->lookup_table;
  for (int i = 0; i < lookup_table_elements; i++) {
    lookup_table[i] = i;
  }
  reference_config->ukernel(lookup_table_elements, lookup_table, lookup_table, &uparams);

  op->lut_config = xnn_init_x8_lut_config();

  op->state = xnn_run_state_invalid;

  return xnn_status_success;
}

static const struct xnn_unary_elementwise_config* get_config(
    enum xnn_unary_operator op_type,
    enum xnn_datatype input_datatype,
    enum xnn_datatype output_datatype,
    const struct xnn_quantization_params* input_quantization,
    const struct xnn_quantization_params* output_quantization) {
  if (input_datatype != output_datatype) {
    if (op_type == xnn_unary_convert) {
      if (input_datatype == xnn_datatype_fp32 && output_datatype == xnn_datatype_fp16) {
        return xnn_init_f32_to_f16_cvt_config();
      } else if (input_datatype == xnn_datatype_fp32 && output_datatype == xnn_datatype_qint8) {
        return xnn_init_f32_to_qs8_cvt_config();
      } else if (input_datatype == xnn_datatype_fp32 && output_datatype == xnn_datatype_quint8) {
        return xnn_init_f32_to_qu8_cvt_config();
      } else if (input_datatype == xnn_datatype_fp16 && output_datatype == xnn_datatype_fp32) {
        return xnn_init_f16_to_f32_cvt_config();
      } else if (input_datatype == xnn_datatype_fp16 && output_datatype == xnn_datatype_qint8) {
        return xnn_init_f16_to_qs8_cvt_config();
      } else if (input_datatype == xnn_datatype_qint8 && output_datatype == xnn_datatype_fp16) {
        return xnn_init_qs8_to_f16_cvt_config();
      } else if (input_datatype == xnn_datatype_qint8 && output_datatype == xnn_datatype_fp32) {
        return xnn_init_qs8_to_f32_cvt_config();
      } else if (input_datatype == xnn_datatype_quint8 && output_datatype == xnn_datatype_fp32) {
        return xnn_init_qu8_to_f32_cvt_config();
      }
    }
    return NULL;
  }
  enum xnn_datatype datatype = output_datatype;
  if (datatype == xnn_datatype_qint8) {
    switch (op_type) {
      case xnn_unary_clamp:
        if (input_quantization->scale != output_quantization->scale ||
            input_quantization->zero_point != output_quantization->zero_point) {
          xnn_log_debug("unsupported operator clamp for datatype QINT8: quantization parameters differ");
          return NULL;
        }
        return xnn_init_s8_clamp_config();
      case xnn_unary_leaky_relu:
        return xnn_init_qs8_lrelu_config();
      case xnn_unary_convert:
        return xnn_init_qs8_cvt_config();
      default:
        return NULL;
    }
  } else if (datatype == xnn_datatype_quint8) {
    switch (op_type) {
      case xnn_unary_clamp:
        if (input_quantization->scale != output_quantization->scale ||
            input_quantization->zero_point != output_quantization->zero_point) {
          xnn_log_debug("unsupported operator clamp for datatype QUINT8: quantization parameters differ");
          return NULL;
        }
        return xnn_init_u8_clamp_config();
      case xnn_unary_leaky_relu:
        return xnn_init_qu8_lrelu_config();
      case xnn_unary_convert:
        return xnn_init_qu8_cvt_config();
      default:
        return NULL;
    }
  } else if (datatype == xnn_datatype_fp16) {
    switch (op_type) {
      case xnn_unary_abs:
        return xnn_init_f16_abs_config();
      case xnn_unary_bankers_rounding:
        return xnn_init_f16_rndne_config();
      case xnn_unary_ceiling:
        return xnn_init_f16_rndu_config();
      case xnn_unary_clamp:
        return xnn_init_f16_clamp_config();
      case xnn_unary_elu:
        return xnn_init_f16_elu_config();
      case xnn_unary_floor:
        return xnn_init_f16_rndd_config();
      case xnn_unary_hardswish:
        return xnn_init_f16_hswish_config();
      case xnn_unary_leaky_relu:
        return xnn_init_f16_lrelu_config();
      case xnn_unary_negate:
        return xnn_init_f16_neg_config();
      case xnn_unary_reciprocal_square_root:
        return xnn_init_f16_rsqrt_config();
      case xnn_unary_sigmoid:
        return xnn_init_f16_sigmoid_config();
      case xnn_unary_square_root:
        return xnn_init_f16_sqrt_config();
      case xnn_unary_square:
        return xnn_init_f16_sqr_config();
      case xnn_unary_tanh:
        return xnn_init_f16_tanh_config();
      default:
        return NULL;
    }
  } else if (datatype == xnn_datatype_fp32) {
    switch (op_type) {
      case xnn_unary_abs:
        return xnn_init_f32_abs_config();
      case xnn_unary_bankers_rounding:
        return xnn_init_f32_rndne_config();
      case xnn_unary_ceiling:
        return xnn_init_f32_rndu_config();
      case xnn_unary_clamp:
        return xnn_init_f32_clamp_config();
      case xnn_unary_elu:
        return xnn_init_f32_elu_config();
      case xnn_unary_exp:
        return xnn_init_f32_exp_config();
      case xnn_unary_floor:
        return xnn_init_f32_rndd_config();
      case xnn_unary_gelu:
        return xnn_init_f32_gelu_config();
      case xnn_unary_hardswish:
        return xnn_init_f32_hswish_config();
      case xnn_unary_leaky_relu:
        return xnn_init_f32_lrelu_config();
      case xnn_unary_log:
        return xnn_init_f32_log_config();
      case xnn_unary_negate:
        return xnn_init_f32_neg_config();
      case xnn_unary_reciprocal_square_root:
        return xnn_init_f32_rsqrt_config();
      case xnn_unary_sigmoid:
        return xnn_init_f32_sigmoid_config();
      case xnn_unary_square_root:
        return xnn_init_f32_sqrt_config();
      case xnn_unary_square:
        return xnn_init_f32_sqr_config();
      case xnn_unary_tanh:
        return xnn_init_f32_tanh_config();
      default:
        return NULL;
    }
  }
  return NULL;
}

static enum xnn_status init_op(
    xnn_operator_t op,
    enum xnn_unary_operator op_type,
    enum xnn_datatype input_datatype,
    enum xnn_datatype output_datatype,
    const union xnn_unary_params* params,
    const struct xnn_quantization_params* input_quantization,
    const struct xnn_quantization_params* output_quantization,
    uint32_t flags) {
  op->type = xnn_operator_type_unary_elementwise;
  op->flags = flags;
  op->unary_elementwise.log2_input_size = xnn_datatype_log2_size_bytes(input_datatype);
  op->unary_elementwise.log2_output_size = xnn_datatype_log2_size_bytes(output_datatype);

  const struct xnn_unary_elementwise_config* config = get_config(op_type, input_datatype, output_datatype, input_quantization, output_quantization);
  if (config) {
    // We have an elementwise config, use it.
    op->unary_elementwise_config = config;
    op->state = xnn_run_state_invalid;

    if (config->init != NULL) {
      config->init(&op->params.unary, params, input_quantization, output_quantization);
    }
    return xnn_status_success;
  }

  // Fall back to reference.
  config = xnn_init_unary_reference_config(op_type, input_datatype, output_datatype);
  if (config) {
    if (xnn_datatype_size_bytes(input_datatype) == 1 && xnn_datatype_size_bytes(output_datatype) == 1) {
      // We can use a LUT for this op.
      return init_lut_op(op, config, params, input_quantization, output_quantization);
    }

    xnn_log_debug(
      "unsupported operator %s for datatypes %s -> %s, falling back to reference kernel",
      xnn_unary_operator_to_string(op_type), xnn_datatype_to_string(input_datatype), xnn_datatype_to_string(output_datatype));

    op->unary_elementwise_config = config;
    op->state = xnn_run_state_invalid;

    if (config->init != NULL) {
      config->init(&op->params.unary, params, input_quantization, output_quantization);
    }
    return xnn_status_success;
  } else {
    xnn_log_error(
      "unsupported operator %s for datatypes %s -> %s, falling back to reference kernel",
      xnn_unary_operator_to_string(op_type), xnn_datatype_to_string(input_datatype), xnn_datatype_to_string(output_datatype));
    return xnn_status_unsupported_parameter;
  }
}

enum xnn_status xnn_create_unary_elementwise_nc(
    enum xnn_unary_operator op_type,
    enum xnn_datatype input_datatype,
    enum xnn_datatype output_datatype,
    const union xnn_unary_params* params,
    const struct xnn_quantization_params* input_quantization,
    const struct xnn_quantization_params* output_quantization,
    uint32_t flags,
    xnn_operator_t* op_out) {
  if ((xnn_params.init_flags & XNN_INIT_FLAG_XNNPACK) == 0) {
    xnn_log_error("failed to create %s operator: XNNPACK is not initialized",
      xnn_unary_operator_to_string(op_type));
    return xnn_status_uninitialized;
  }

  xnn_operator_t op = xnn_allocate_zero_simd_memory(sizeof(struct xnn_operator));
  if (op == NULL) {
    xnn_log_error(
      "failed to allocate %zu bytes for %s operator descriptor",
      sizeof(struct xnn_operator), xnn_unary_operator_to_string(op_type));
    return xnn_status_out_of_memory;
  }

  enum xnn_status status = init_op(op, op_type, input_datatype, output_datatype, params, input_quantization, output_quantization,flags);
  if (status != xnn_status_success) {
    xnn_delete_operator(op);
    return status;
  }

  *op_out = op;
  return xnn_status_success;
}

static bool is_contiguous(xnn_operator_t op)
{
  const size_t channels = op->channels;
  const size_t input_stride = op->input_pixel_stride;
  const size_t output_stride = op->output_pixel_stride;
  const size_t batch_size = op->batch_size;
  return (((input_stride ^ channels) | (output_stride ^ channels)) == 0) || batch_size == 1;
}

enum xnn_status xnn_reshape_unary_elementwise_nc(
  xnn_operator_t op,
  size_t batch_size,
  size_t channels,
  size_t input_stride,
  size_t output_stride,
  pthreadpool_t threadpool) {
  op->state = xnn_run_state_invalid;

  if (batch_size == 0 || channels == 0) {
    op->state = xnn_run_state_skip;
    return xnn_status_success;
  }

  if (input_stride < channels) {
    xnn_log_error(
      "failed to create %s operator with input element stride of %zu: "
      "stride must be at least as large as the number of channels (%zu)",
      xnn_operator_type_to_string(op->type), input_stride, channels);
    return xnn_status_invalid_parameter;
  }

  if (output_stride < channels) {
    xnn_log_error(
      "failed to create %s operator with output element stride of %zu: "
      "stride must be at least as large as the number of channels (%zu)",
      xnn_operator_type_to_string(op->type), output_stride, channels);
    return xnn_status_invalid_parameter;
  }

  op->batch_size = batch_size;
  op->channels = channels;
  op->input_pixel_stride = input_stride;
  op->output_pixel_stride = output_stride;

  if (op->lookup_table) {
    const struct xnn_x8_lut_config* lut_config = op->lut_config;
    if (is_contiguous(op)) {
      op->context.lut_contiguous = (struct lut_contiguous_context) {
        .x_stride = input_stride * sizeof(uint8_t),
        .t = op->lookup_table,
        .y_stride = output_stride * sizeof(uint8_t),
        .ukernel = lut_config->microkernel,
      };

      const size_t range = batch_size * channels * sizeof(uint8_t);
      size_t tile = range;
      if (pthreadpool_get_threads_count(threadpool) > 1) {
        const size_t block_size = 1024;
        tile = block_size * sizeof(uint8_t);
      }

      op->compute[0].type = xnn_parallelization_type_1d_tile_1d;
      op->compute[0].task_1d_tile_1d = (pthreadpool_task_1d_tile_1d_t) xnn_compute_lut_contiguous;
      op->compute[0].range[0] = range;
      op->compute[0].tile[0] = tile;
    } else {
      op->context.lut_strided = (struct lut_strided_context) {
        .n = channels * sizeof(uint8_t),
        .x_stride = input_stride * sizeof(uint8_t),
        .t = op->lookup_table,
        .y_stride = output_stride * sizeof(uint8_t),
        .ukernel = lut_config->microkernel,
      };
      op->compute[0].type = xnn_parallelization_type_1d;
      op->compute[0].task_1d = (pthreadpool_task_1d_t) xnn_compute_lut_strided;
      op->compute[0].range[0] = batch_size;
    }
  } else {
    const xnn_vunary_ukernel_fn ukernel = op->unary_elementwise_config->ukernel;
    const size_t num_threads = pthreadpool_get_threads_count(threadpool);
    if (is_contiguous(op)) {
      const size_t block_size = 4096;

      op->context.univector_contiguous = (struct univector_contiguous_context) {
        .log2_xsize = op->unary_elementwise.log2_input_size,
        .log2_ysize = op->unary_elementwise.log2_output_size,
        .ukernel = ukernel,
      };
      memcpy(&op->context.univector_contiguous.params, &op->params.unary, sizeof(op->params.unary));

      const size_t range = (batch_size * channels) << op->unary_elementwise.log2_input_size;
      op->compute[0].type = xnn_parallelization_type_1d_tile_1d;
      op->compute[0].task_1d_tile_1d = (pthreadpool_task_1d_tile_1d_t) xnn_compute_univector_contiguous;
      op->compute[0].range[0] = range;
      op->compute[0].tile[0] = (num_threads == 1) ? range : block_size;
    } else {
      op->context.univector_strided = (struct univector_strided_context) {
        .n = channels << op->unary_elementwise.log2_input_size,
        .x_stride = input_stride << op->unary_elementwise.log2_input_size,
        .y_stride = output_stride << op->unary_elementwise.log2_output_size,
        .ukernel = ukernel,
      };
      memcpy(&op->context.univector_strided.params, &op->params.unary, sizeof(op->params.unary));

      op->compute[0].type = xnn_parallelization_type_1d_tile_1d;
      op->compute[0].task_1d_tile_1d = (pthreadpool_task_1d_tile_1d_t) xnn_compute_univector_strided;
      op->compute[0].range[0] = batch_size;
      op->compute[0].tile[0] = (num_threads == 1) ? batch_size : 1;
    }
  }
  op->state = xnn_run_state_needs_setup;
  return xnn_status_success;
}

enum xnn_status xnn_setup_unary_elementwise_nc(
    xnn_operator_t op,
    const void* input,
    void* output) {
  switch (op->state) {
    case xnn_run_state_skip:
      return xnn_status_success;
    case xnn_run_state_invalid:
      xnn_log_error(
        "failed to setup %s operator: operator has not been reshaped yet",
        xnn_operator_type_to_string(op->type));
      return xnn_status_invalid_state;
    case xnn_run_state_needs_setup:
      // Operator has been reshaped, but not setup, continue with setup.
    case xnn_run_state_ready:
      // Operator has been reshaped, and we are setting up with different pointers.
      break;
  }

  if (op->lookup_table) {
    if (is_contiguous(op)) {
      op->context.lut_contiguous.x = input;
      op->context.lut_contiguous.y = output;
    } else {
      op->context.lut_strided.x = input;
      op->context.lut_strided.y = output;
    }
  } else {
    if (is_contiguous(op)) {
      op->context.univector_contiguous.x = input;
      op->context.univector_contiguous.y = output;
    } else {
      op->context.univector_strided.x = input;
      op->context.univector_strided.y = output;
    }
  }
  op->state = xnn_run_state_ready;

  return xnn_status_success;
}

enum xnn_status xnn_run_unary_elementwise_nc(
    // create parameters
    enum xnn_unary_operator op_type,
    enum xnn_datatype input_datatype,
    enum xnn_datatype output_datatype,
    const union xnn_unary_params* params,
    const struct xnn_quantization_params* input_quantization,
    const struct xnn_quantization_params* output_quantization,
    uint32_t flags,
    // reshape parameters
    size_t batch_size,
    size_t channels,
    size_t input_stride,
    size_t output_stride,
    pthreadpool_t threadpool,
    // setup parameters
    const void* input,
    void* output) {

  if (batch_size == 0 || channels == 0) {
    return xnn_status_success;
  }

  struct xnn_operator op;
  memset(&op, 0, sizeof(op));

  enum xnn_status status = init_op(&op, op_type, input_datatype, output_datatype, params, input_quantization, output_quantization, flags);
  if (status != xnn_status_success) {
    xnn_destroy_operator(&op);
    return status;
  }

  status = xnn_reshape_unary_elementwise_nc(&op, batch_size, channels, input_stride, output_stride, threadpool);
  if (status != xnn_status_success){
    xnn_destroy_operator(&op);
    return status;
  }

  status = xnn_setup_unary_elementwise_nc(&op, input, output);
  if (status != xnn_status_success){
    xnn_destroy_operator(&op);
    return status;
  }

  status = xnn_run_operator(&op, threadpool);
  xnn_destroy_operator(&op);
  return status;
}

static void init_unary_elementwise_nc(
    uint32_t flags,
    const void* params,
    size_t params_size,
    enum xnn_operator_type operator_type,
    const struct xnn_unary_elementwise_config* unary_elementwise_config,
    xnn_operator_t unary_elementwise_op)
{
  assert(unary_elementwise_config != NULL);
  assert(unary_elementwise_config->ukernel != NULL);

  if (params_size != 0) {
    memcpy(&unary_elementwise_op->params, params, params_size);
  }

  unary_elementwise_op->unary_elementwise_config = unary_elementwise_config;
  unary_elementwise_op->type = operator_type;
  unary_elementwise_op->flags = flags;

  unary_elementwise_op->state = xnn_run_state_invalid;
}

static enum xnn_status create_unary_elementwise_nc(
    uint32_t flags,
    const struct xnn_unary_elementwise_config* unary_elementwise_config,
    const void* params,
    size_t params_size,
    enum xnn_operator_type operator_type,
    xnn_operator_t* unary_elementwise_op_out)
{
  xnn_operator_t unary_elementwise_op = NULL;

  if ((xnn_params.init_flags & XNN_INIT_FLAG_XNNPACK) == 0) {
    xnn_log_error("failed to create %s operator: XNNPACK is not initialized",
      xnn_operator_type_to_string(operator_type));
    return xnn_status_uninitialized;
  }

  if (unary_elementwise_config == NULL) {
    xnn_log_error(
        "failed to create %s operator: unsupported hardware configuration",
        xnn_operator_type_to_string(operator_type));
    return xnn_status_unsupported_hardware;
  }

  unary_elementwise_op = xnn_allocate_zero_simd_memory(sizeof(struct xnn_operator));
  if (unary_elementwise_op == NULL) {
    xnn_log_error(
      "failed to allocate %zu bytes for %s operator descriptor",
      sizeof(struct xnn_operator), xnn_operator_type_to_string(operator_type));
    return xnn_status_out_of_memory;
  }

  init_unary_elementwise_nc(
    flags, params, params_size,
    operator_type, unary_elementwise_config, unary_elementwise_op);

  *unary_elementwise_op_out = unary_elementwise_op;
  return xnn_status_success;
}

static bool is_copy_operator(enum xnn_operator_type operator_type) {
  switch (operator_type) {
    case xnn_operator_type_copy_nc_x8:
    case xnn_operator_type_copy_nc_x16:
    case xnn_operator_type_copy_nc_x32:
      return true;
    default:
      return false;
  }
}

static enum xnn_status reshape_unary_elementwise_nc(
    xnn_operator_t unary_elementwise_op,
    enum xnn_operator_type expected_operator_type,
    size_t batch_size,
    size_t channels,
    size_t input_stride,
    size_t output_stride,
    uint32_t log2_input_size,
    uint32_t log2_output_size,
    const void* params,
    size_t params_size,
    pthreadpool_t threadpool)
{
  if (unary_elementwise_op->type != expected_operator_type) {
    xnn_log_error("failed to setup operator: operator type mismatch (expected %s, got %s)",
      xnn_operator_type_to_string(expected_operator_type),
      xnn_operator_type_to_string(unary_elementwise_op->type));
    return xnn_status_invalid_parameter;
  }
  unary_elementwise_op->state = xnn_run_state_invalid;

  if (batch_size == 0 || channels == 0) {
    unary_elementwise_op->state = xnn_run_state_skip;
    return xnn_status_success;
  }

  if (input_stride < channels) {
    xnn_log_error(
      "failed to create %s operator with input element stride of %zu: "
      "stride must be at least as large as the number of channels (%zu)",
      xnn_operator_type_to_string(unary_elementwise_op->type), input_stride, channels);
    return xnn_status_invalid_parameter;
  }

  if (output_stride < channels) {
    xnn_log_error(
      "failed to create %s operator with output element stride of %zu: "
      "stride must be at least as large as the number of channels (%zu)",
      xnn_operator_type_to_string(unary_elementwise_op->type), output_stride, channels);
    return xnn_status_invalid_parameter;
  }

  unary_elementwise_op->batch_size = batch_size;
  unary_elementwise_op->channels = channels;
  unary_elementwise_op->input_pixel_stride = input_stride;
  unary_elementwise_op->output_pixel_stride = output_stride;

  const xnn_vunary_ukernel_fn ukernel = unary_elementwise_op->unary_elementwise_config->ukernel;
  const size_t num_threads = pthreadpool_get_threads_count(threadpool);
  if ((((input_stride ^ channels) | (output_stride ^ channels)) == 0) || batch_size == 1) {
    const size_t block_size = 4096;

    unary_elementwise_op->context.univector_contiguous = (struct univector_contiguous_context) {
      .log2_xsize = log2_input_size,
      .log2_ysize = log2_output_size,
      .ukernel = ukernel,
    };
    if (params_size != 0) {
      memcpy(&unary_elementwise_op->context.univector_contiguous.params, params, params_size);
    }

    const size_t range = (batch_size * channels) << log2_input_size;
    unary_elementwise_op->compute[0].type = xnn_parallelization_type_1d_tile_1d;
    unary_elementwise_op->compute[0].task_1d_tile_1d = (pthreadpool_task_1d_tile_1d_t) xnn_compute_univector_contiguous;
    unary_elementwise_op->compute[0].range[0] = range;
    unary_elementwise_op->compute[0].tile[0] = (num_threads == 1) ? range : block_size;;
  } else {
    unary_elementwise_op->context.univector_strided = (struct univector_strided_context) {
      .n = channels << log2_input_size,
      .x_stride = input_stride << log2_input_size,
      .y_stride = output_stride << log2_output_size,
      .ukernel = ukernel,
    };
    if (params_size != 0) {
      memcpy(&unary_elementwise_op->context.univector_strided.params, params, params_size);
    }

    unary_elementwise_op->compute[0].type = xnn_parallelization_type_1d_tile_1d;
    unary_elementwise_op->compute[0].task_1d_tile_1d = (pthreadpool_task_1d_tile_1d_t) xnn_compute_univector_strided;
    unary_elementwise_op->compute[0].range[0] = batch_size;
    unary_elementwise_op->compute[0].tile[0] = (num_threads == 1) ? batch_size : 1;
  }
  unary_elementwise_op->state = xnn_run_state_needs_setup;

  return xnn_status_success;
}

static enum xnn_status setup_unary_elementwise_nc(
    xnn_operator_t unary_elementwise_op,
    enum xnn_operator_type expected_operator_type,
    const void* input,
    void* output)
{
  if (unary_elementwise_op->type != expected_operator_type) {
    xnn_log_error("failed to setup operator: operator type mismatch (expected %s, got %s)",
      xnn_operator_type_to_string(expected_operator_type),
      xnn_operator_type_to_string(unary_elementwise_op->type));
    return xnn_status_invalid_parameter;
  }

  switch (unary_elementwise_op->state) {
    case xnn_run_state_skip:
      return xnn_status_success;
    case xnn_run_state_invalid:
      xnn_log_error(
        "failed to setup %s operator: operator has not been reshaped yet",
        xnn_operator_type_to_string(unary_elementwise_op->type));
      return xnn_status_invalid_state;
    case xnn_run_state_needs_setup:
      // Operator has been reshaped, but not setup, continue with setup.
    case xnn_run_state_ready:
      // Operator has been reshaped, and we are setting up with different pointers.
      break;
  }

  if (input == output && is_copy_operator(expected_operator_type)) {
    unary_elementwise_op->state = xnn_run_state_skip;
    return xnn_status_success;
  }

  const size_t channels = unary_elementwise_op->channels;
  const size_t input_stride = unary_elementwise_op->input_pixel_stride;
  const size_t output_stride = unary_elementwise_op->output_pixel_stride;

  if ((((input_stride ^ channels) | (output_stride ^ channels)) == 0) || unary_elementwise_op->batch_size == 1) {
    unary_elementwise_op->context.univector_contiguous.x = input;
    unary_elementwise_op->context.univector_contiguous.y = output;
  } else {
    unary_elementwise_op->context.univector_strided.x = input;
    unary_elementwise_op->context.univector_strided.y = output;
  }
  unary_elementwise_op->state = xnn_run_state_ready;

  return xnn_status_success;
}

enum xnn_status create_convert_nc_f16_qx8(
  uint32_t flags,
  const struct xnn_unary_elementwise_config* cvt_config,
  enum xnn_operator_type expected_operator_type,
  xnn_operator_t* convert_op_out)
{
  const struct xnn_reduce_config* f16_rminmax_config = xnn_init_f16_rminmax_config();
  if (f16_rminmax_config == NULL) {
    xnn_log_error(
        "failed to create %s operator: unsupported hardware configuration",
        xnn_operator_type_to_string(expected_operator_type));
    return xnn_status_unsupported_hardware;
  }

  struct xnn_f16_default_params params;

  enum xnn_status status = create_unary_elementwise_nc(
    flags, cvt_config,
    &params, sizeof(params),
    expected_operator_type, convert_op_out);
  if (status == xnn_status_success) {
    (*convert_op_out)->rminmax_config = f16_rminmax_config;
  }
  return status;
}

enum xnn_status create_convert_nc_f32_qx8(
  uint32_t flags,
  const struct xnn_unary_elementwise_config* cvt_config,
  enum xnn_operator_type expected_operator_type,
  xnn_operator_t* convert_op_out)
{
  const struct xnn_reduce_config* f32_rminmax_config = xnn_init_f32_rminmax_config();
  if (f32_rminmax_config == NULL) {
    xnn_log_error(
        "failed to create %s operator: unsupported hardware configuration",
        xnn_operator_type_to_string(expected_operator_type));
    return xnn_status_unsupported_hardware;
  }

  struct xnn_f32_default_params params;

  enum xnn_status status = create_unary_elementwise_nc(
    flags, cvt_config,
    &params, sizeof(params),
    expected_operator_type, convert_op_out);
  if (status == xnn_status_success) {
    (*convert_op_out)->rminmax_config = f32_rminmax_config;
  }
  return status;
}

enum xnn_status xnn_create_convert_nc_f16_qd8(
  uint32_t flags,
  xnn_operator_t* convert_op_out) {
  return create_convert_nc_f16_qx8(flags, xnn_init_f16_to_qs8_cvt_config(), xnn_operator_type_convert_nc_f16_qd8, convert_op_out);
}

enum xnn_status xnn_create_convert_nc_f16_qdu8(
  uint32_t flags,
  xnn_operator_t* convert_op_out) {
  return create_convert_nc_f16_qx8(flags, xnn_init_f16_to_qu8_cvt_config(), xnn_operator_type_convert_nc_f16_qdu8, convert_op_out);
}

enum xnn_status xnn_create_convert_nc_f32_qd8(
  uint32_t flags,
  xnn_operator_t* convert_op_out) {
  return create_convert_nc_f32_qx8(flags, xnn_init_f32_to_qs8_cvt_config(), xnn_operator_type_convert_nc_f32_qd8, convert_op_out);
}

enum xnn_status xnn_create_convert_nc_f32_qdu8(
  uint32_t flags,
  xnn_operator_t* convert_op_out) {
  return create_convert_nc_f32_qx8(flags, xnn_init_f32_to_qu8_cvt_config(), xnn_operator_type_convert_nc_f32_qdu8, convert_op_out);
}

enum xnn_status xnn_create_convert_nc_f32_qp8(uint32_t flags,
                                              xnn_operator_t* convert_op_out) {
  const struct xnn_reduce_config* f32_rminmax_config =
      xnn_init_f32_rminmax_config();
  if (f32_rminmax_config == NULL) {
    xnn_log_error(
        "failed to create %s operator: unsupported hardware configuration",
        xnn_operator_type_to_string(xnn_operator_type_convert_nc_f32_qp8));
    return xnn_status_unsupported_hardware;
  }

  struct xnn_f32_default_params params;

  enum xnn_status status = create_unary_elementwise_nc(
    flags, xnn_init_f32_to_qp8_cvt_config(),
    &params, sizeof(params),
    xnn_operator_type_convert_nc_f32_qp8, convert_op_out);
  if (status == xnn_status_success) {
    (*convert_op_out)->rminmax_config = f32_rminmax_config;
  }
  return status;
}

enum xnn_status xnn_create_copy_nc_x8(
    uint32_t flags,
    xnn_operator_t* copy_op_out)
{
  return create_unary_elementwise_nc(
    flags, xnn_init_xx_copy_config(),
    /*params=*/NULL, /*params_size=*/0,
    xnn_operator_type_copy_nc_x8, copy_op_out);
}

enum xnn_status xnn_create_copy_nc_x16(
    uint32_t flags,
    xnn_operator_t* copy_op_out)
{
  return create_unary_elementwise_nc(
    flags, xnn_init_xx_copy_config(),
    /*params=*/NULL, /*params_size=*/0,
    xnn_operator_type_copy_nc_x16, copy_op_out);
}

enum xnn_status xnn_create_copy_nc_x32(
    uint32_t flags,
    xnn_operator_t* copy_op_out)
{
  return create_unary_elementwise_nc(
    flags, xnn_init_xx_copy_config(),
    /*params=*/NULL, /*params_size=*/0,
    xnn_operator_type_copy_nc_x32, copy_op_out);
}

enum xnn_status reshape_convert_nc_f16_qx8(
    xnn_operator_t convert_op,
    size_t batch_size,
    size_t channels,
    size_t input_stride,
    size_t output_stride,
    enum xnn_operator_type expected_type,
    pthreadpool_t threadpool)
{
  if (convert_op->type != expected_type) {
    xnn_log_error("failed to setup operator: operator type mismatch (expected %s, got %s)",
      xnn_operator_type_to_string(expected_type),
      xnn_operator_type_to_string(convert_op->type));
    return xnn_status_invalid_parameter;
  }
  convert_op->state = xnn_run_state_invalid;

  if (batch_size == 0) {
    convert_op->state = xnn_run_state_skip;
    return xnn_status_success;
  }

  convert_op->batch_size = batch_size;

  convert_op->context.f16_qd8_convert = (struct f16_qd8_convert_context) {
    .n = channels * sizeof(uint16_t),
    .x_stride = input_stride * sizeof(uint16_t),
    .y_stride = output_stride,
    .batch_size = batch_size,
    .rminmax_ukernel = convert_op->rminmax_config->ukernel,
    .convert_ukernel = convert_op->unary_elementwise_config->ukernel,
    .init_params = convert_op->unary_elementwise_config->init,
  };
  memcpy(&convert_op->context.f16_qd8_convert.params, &convert_op->params.f16_default, sizeof(convert_op->params.f16_default));

  convert_op->compute[0].type = xnn_parallelization_type_1d;
  convert_op->compute[0].task_1d = (pthreadpool_task_1d_t) xnn_compute_f16_qd8_convert;
  switch (expected_type) {
    case xnn_operator_type_convert_nc_f16_qd8:
      convert_op->compute[0].task_1d = (pthreadpool_task_1d_t) xnn_compute_f16_qd8_convert;
      break;
    case xnn_operator_type_convert_nc_f16_qdu8:
      convert_op->compute[0].task_1d = (pthreadpool_task_1d_t) xnn_compute_f16_qdu8_convert;
      break;
    default:
      XNN_UNREACHABLE;
  }
  convert_op->compute[0].range[0] = batch_size;

  convert_op->compute[1].type = xnn_parallelization_type_1d;
  convert_op->compute[1].task_1d = (pthreadpool_task_1d_t) xnn_compute_pad_qd8_params;
  convert_op->compute[1].range[0] = 1;

  convert_op->state = xnn_run_state_needs_setup;

  return xnn_status_success;
}

enum xnn_status reshape_convert_nc_f32_qx8(
    xnn_operator_t convert_op,
    size_t batch_size,
    size_t channels,
    size_t input_stride,
    size_t output_stride,
    enum xnn_operator_type expected_type,
    pthreadpool_t threadpool)
{
  if (convert_op->type != expected_type) {
    xnn_log_error("failed to setup operator: operator type mismatch (expected %s, got %s)",
      xnn_operator_type_to_string(expected_type),
      xnn_operator_type_to_string(convert_op->type));
    return xnn_status_invalid_parameter;
  }
  convert_op->state = xnn_run_state_invalid;

  if (batch_size == 0) {
    convert_op->state = xnn_run_state_skip;
    return xnn_status_success;
  }

  convert_op->batch_size = batch_size;

  convert_op->context.f32_qd8_convert = (struct f32_qd8_convert_context) {
    .n = channels * sizeof(float),
    .x_stride = input_stride * sizeof(float),
    .y_stride = output_stride,
    .batch_size = batch_size,
    .rminmax_ukernel = convert_op->rminmax_config->ukernel,
    .convert_ukernel = convert_op->unary_elementwise_config->ukernel,
    .init_params = convert_op->unary_elementwise_config->init,
  };
  memcpy(&convert_op->context.f32_qd8_convert.params, &convert_op->params.f32_default, sizeof(convert_op->params.f32_default));

  convert_op->compute[0].type = xnn_parallelization_type_1d;
  switch (expected_type) {
    case xnn_operator_type_convert_nc_f32_qd8:
      convert_op->compute[0].task_1d = (pthreadpool_task_1d_t) xnn_compute_f32_qd8_convert;
      break;
    case xnn_operator_type_convert_nc_f32_qdu8:
      convert_op->compute[0].task_1d = (pthreadpool_task_1d_t) xnn_compute_f32_qdu8_convert;
      break;
    default:
      XNN_UNREACHABLE;
  }
  convert_op->compute[0].range[0] = batch_size;

  convert_op->compute[1].type = xnn_parallelization_type_1d;
  convert_op->compute[1].task_1d = (pthreadpool_task_1d_t) xnn_compute_pad_qd8_params;
  convert_op->compute[1].range[0] = 1;

  convert_op->state = xnn_run_state_needs_setup;

  return xnn_status_success;
}

enum xnn_status xnn_reshape_convert_nc_f16_qd8(
    xnn_operator_t convert_op,
    size_t batch_size,
    size_t channels,
    size_t input_stride,
    size_t output_stride,
    pthreadpool_t threadpool)
{
  return reshape_convert_nc_f16_qx8(convert_op, batch_size, channels, input_stride, output_stride, xnn_operator_type_convert_nc_f16_qd8, threadpool);
}

enum xnn_status xnn_reshape_convert_nc_f16_qdu8(
    xnn_operator_t convert_op,
    size_t batch_size,
    size_t channels,
    size_t input_stride,
    size_t output_stride,
    pthreadpool_t threadpool)
{
  return reshape_convert_nc_f16_qx8(convert_op, batch_size, channels, input_stride, output_stride, xnn_operator_type_convert_nc_f16_qdu8, threadpool);
}

enum xnn_status xnn_reshape_convert_nc_f32_qd8(
    xnn_operator_t convert_op,
    size_t batch_size,
    size_t channels,
    size_t input_stride,
    size_t output_stride,
    pthreadpool_t threadpool)
{
  return reshape_convert_nc_f32_qx8(convert_op, batch_size, channels, input_stride, output_stride, xnn_operator_type_convert_nc_f32_qd8, threadpool);
}

enum xnn_status xnn_reshape_convert_nc_f32_qdu8(
    xnn_operator_t convert_op,
    size_t batch_size,
    size_t channels,
    size_t input_stride,
    size_t output_stride,
    pthreadpool_t threadpool)
{
  return reshape_convert_nc_f32_qx8(convert_op, batch_size, channels, input_stride, output_stride, xnn_operator_type_convert_nc_f32_qdu8, threadpool);
}

enum xnn_status xnn_reshape_convert_nc_f32_qp8(xnn_operator_t convert_op,
                                               size_t batch_size,
                                               size_t channels,
                                               size_t input_stride,
                                               pthreadpool_t threadpool) {
  if (convert_op->type != xnn_operator_type_convert_nc_f32_qp8) {
    xnn_log_error(
        "failed to setup operator: operator type mismatch (expected %s, got "
        "%s)",
        xnn_operator_type_to_string(xnn_operator_type_convert_nc_f32_qp8),
        xnn_operator_type_to_string(convert_op->type));
    return xnn_status_invalid_parameter;
  }
  convert_op->state = xnn_run_state_invalid;

  if (batch_size == 0) {
    convert_op->state = xnn_run_state_skip;
    return xnn_status_success;
  }

  convert_op->batch_size = batch_size;

  const struct xnn_gemm_config* gemm_config =
      xnn_init_qp8_f32_qc4w_gemm_config();
  const uint32_t mr_packed = batch_size == 1 ? 1 : gemm_config->mr_packed;
  const uint32_t kr = UINT32_C(1) << gemm_config->log2_kr;
  const uint32_t sr = UINT32_C(1) << gemm_config->log2_sr;

  convert_op->context.f32_qp8_convert = (struct f32_qp8_convert_context){
      .m = batch_size,
      .k = channels,
      .mr = mr_packed,
      .kr = kr,
      .sr = sr,
      .lhs_stride = input_stride * sizeof(float),
      .packq_ukernel = (xnn_x8_packq_f32qp8_ukernel_fn)
                           convert_op->unary_elementwise_config->ukernel,
  };

  // TODO(b/340399245) - Ideally, this should parallelize along `batch` in
  // groups of `mr`.
  convert_op->compute[0].type = xnn_parallelization_type_1d;
  convert_op->compute[0].task_1d =
      (pthreadpool_task_1d_t)xnn_compute_f32_qp8_convert;
  convert_op->compute[0].range[0] = batch_size;

  convert_op->state = xnn_run_state_needs_setup;

  return xnn_status_success;
}

enum xnn_status xnn_reshape_copy_nc_x8(
    xnn_operator_t copy_op,
    size_t batch_size,
    size_t channels,
    size_t input_stride,
    size_t output_stride,
    pthreadpool_t threadpool)
{
  return reshape_unary_elementwise_nc(
    copy_op, xnn_operator_type_copy_nc_x8,
    batch_size,
    channels, input_stride, output_stride,
    /*log2_input_size=*/XNN_LOG2_SIZEOF_UINT8_T,
    /*log2_output_size=*/XNN_LOG2_SIZEOF_UINT8_T,
    /*params=*/NULL, /*params_size=*/0,
    threadpool);
}

enum xnn_status xnn_reshape_copy_nc_x16(
    xnn_operator_t copy_op,
    size_t batch_size,
    size_t channels,
    size_t input_stride,
    size_t output_stride,
    pthreadpool_t threadpool)
{
  return reshape_unary_elementwise_nc(
    copy_op, xnn_operator_type_copy_nc_x16,
    batch_size,
    channels, input_stride, output_stride,
    /*log2_input_size=*/XNN_LOG2_SIZEOF_UINT16_T,
    /*log2_output_size=*/XNN_LOG2_SIZEOF_UINT16_T,
    /*params=*/NULL, /*params_size=*/0,
    threadpool);
}

enum xnn_status xnn_reshape_copy_nc_x32(
    xnn_operator_t copy_op,
    size_t batch_size,
    size_t channels,
    size_t input_stride,
    size_t output_stride,
    pthreadpool_t threadpool)
{
  return reshape_unary_elementwise_nc(
    copy_op, xnn_operator_type_copy_nc_x32,
    batch_size,
    channels, input_stride, output_stride,
    /*log2_input_size=*/XNN_LOG2_SIZEOF_UINT32_T,
    /*log2_output_size=*/XNN_LOG2_SIZEOF_UINT32_T,
    /*params=*/NULL, /*params_size=*/0,
    threadpool);
}

enum xnn_status setup_convert_nc_f16_qx8(
  xnn_operator_t convert_op,
  const void* input,
  void* output,
  enum xnn_operator_type expected_operator_type,
  struct xnn_quantization_params* quantization_params)
{
  if (convert_op->type != expected_operator_type) {
    xnn_log_error("failed to setup operator: operator type mismatch (expected %s, got %s)",
      xnn_operator_type_to_string(expected_operator_type),
      xnn_operator_type_to_string(convert_op->type));
    return xnn_status_invalid_parameter;
  }

  switch (convert_op->state) {
    case xnn_run_state_skip:
      return xnn_status_success;
    case xnn_run_state_invalid:
      xnn_log_error(
        "failed to setup %s operator: operator has not been reshaped yet",
        xnn_operator_type_to_string(convert_op->type));
      return xnn_status_invalid_state;
    case xnn_run_state_needs_setup:
      // Operator has been reshaped, but not setup, continue with setup.
    case xnn_run_state_ready:
      // Operator has been reshaped, and we are setting up with different pointers.
      break;
  }

  convert_op->context.f16_qd8_convert.x = input;
  convert_op->context.f16_qd8_convert.y = output;
  convert_op->context.f16_qd8_convert.quantization_params = (struct xnn_qd8_quantization_params*) quantization_params;
  convert_op->state = xnn_run_state_ready;

  return xnn_status_success;
}

enum xnn_status setup_convert_nc_f32_qx8(
  xnn_operator_t convert_op,
  const float* input,
  void* output,
  enum xnn_operator_type expected_operator_type,
  struct xnn_quantization_params* quantization_params)
{
  if (convert_op->type != expected_operator_type) {
    xnn_log_error("failed to setup operator: operator type mismatch (expected %s, got %s)",
      xnn_operator_type_to_string(expected_operator_type),
      xnn_operator_type_to_string(convert_op->type));
    return xnn_status_invalid_parameter;
  }

  switch (convert_op->state) {
    case xnn_run_state_skip:
      return xnn_status_success;
    case xnn_run_state_invalid:
      xnn_log_error(
        "failed to setup %s operator: operator has not been reshaped yet",
        xnn_operator_type_to_string(convert_op->type));
      return xnn_status_invalid_state;
    case xnn_run_state_needs_setup:
      // Operator has been reshaped, but not setup, continue with setup.
    case xnn_run_state_ready:
      // Operator has been reshaped, and we are setting up with different pointers.
      break;
  }

  convert_op->context.f32_qd8_convert.x = input;
  convert_op->context.f32_qd8_convert.y = output;
  convert_op->context.f32_qd8_convert.quantization_params = (struct xnn_qd8_quantization_params*) quantization_params;

  convert_op->state = xnn_run_state_ready;

  return xnn_status_success;
}

enum xnn_status xnn_setup_convert_nc_f16_qd8(
  xnn_operator_t convert_op,
  const void* input,
  int8_t* output,
  struct xnn_quantization_params* quantization_params)
{
  return setup_convert_nc_f16_qx8(convert_op, input, output, xnn_operator_type_convert_nc_f16_qd8, quantization_params);
}

enum xnn_status xnn_setup_convert_nc_f16_qdu8(
  xnn_operator_t convert_op,
  const void* input,
  uint8_t* output,
  struct xnn_quantization_params* quantization_params)
{
  return setup_convert_nc_f16_qx8(convert_op, input, output, xnn_operator_type_convert_nc_f16_qdu8, quantization_params);
}

enum xnn_status xnn_setup_convert_nc_f32_qd8(
  xnn_operator_t convert_op,
  const float* input,
  int8_t* output,
  struct xnn_quantization_params* quantization_params)
{
  return setup_convert_nc_f32_qx8(convert_op, input, output, xnn_operator_type_convert_nc_f32_qd8, quantization_params);
}

enum xnn_status xnn_setup_convert_nc_f32_qdu8(
  xnn_operator_t convert_op,
  const float* input,
  uint8_t* output,
  struct xnn_quantization_params* quantization_params)
{
  return setup_convert_nc_f32_qx8(convert_op, input, output, xnn_operator_type_convert_nc_f32_qdu8, quantization_params);
}

enum xnn_status xnn_setup_convert_nc_f32_qp8(xnn_operator_t convert_op,
                                             const float* input,
                                             int8_t* output) {
  enum xnn_status status =
      check_op_type(convert_op, xnn_operator_type_convert_nc_f32_qp8);
  if (status != xnn_status_success) {
    return status;
  }

  switch (convert_op->state) {
    case xnn_run_state_skip:
      return xnn_status_success;
    case xnn_run_state_invalid:
      xnn_log_error(
          "failed to setup %s operator: operator has not been reshaped yet",
          xnn_operator_type_to_string(convert_op->type));
      return xnn_status_invalid_state;
    case xnn_run_state_needs_setup:
      // Operator has been reshaped, but not setup, continue with setup.
    case xnn_run_state_ready:
      // Operator has been reshaped, and we are setting up with different
      // pointers.
      break;
  }

  convert_op->context.f32_qp8_convert.lhs = input;
  convert_op->context.f32_qp8_convert.lhs_packed = output;
  convert_op->state = xnn_run_state_ready;

  return xnn_status_success;
}

enum xnn_status xnn_setup_copy_nc_x8(
    xnn_operator_t copy_op,
    const void* input,
    void* output)
{
  return setup_unary_elementwise_nc(
    copy_op, xnn_operator_type_copy_nc_x8,
    input, output);
}

enum xnn_status xnn_setup_copy_nc_x16(
    xnn_operator_t copy_op,
    const void* input,
    void* output)
{
  return setup_unary_elementwise_nc(
    copy_op, xnn_operator_type_copy_nc_x16,
    input, output);
}

enum xnn_status xnn_setup_copy_nc_x32(
    xnn_operator_t copy_op,
    const void* input,
    void* output)
{
  return setup_unary_elementwise_nc(
    copy_op, xnn_operator_type_copy_nc_x32,
    input, output);
}

static enum xnn_status run_unary_elementwise_nc(
    enum xnn_operator_type operator_type,
    size_t channels,
    size_t input_stride,
    size_t output_stride,
    size_t batch_size,
    const void* input,
    void* output,
    const struct xnn_unary_elementwise_config* unary_elementwise_config,
    const void* params,
    size_t params_size,
    uint32_t log2_input_size,
    uint32_t log2_output_size,
    uint32_t flags,
    pthreadpool_t threadpool)
{
  if (unary_elementwise_config == NULL) {
    xnn_log_error(
          "failed to create %s operator: unsupported hardware configuration",
          xnn_operator_type_to_string(operator_type));
    return xnn_status_unsupported_hardware;
  }

  if (channels == 0) {
    xnn_log_error(
      "failed to run %s operator with %zu channels: number of channels must be non-zero",
      xnn_operator_type_to_string(operator_type), channels);
    return xnn_status_invalid_parameter;
  }

  if (input_stride < channels) {
    xnn_log_error(
      "failed to run %s operator with input element stride of %zu: "
      "stride must be at least as large as the number of channels (%zu)",
      xnn_operator_type_to_string(operator_type), input_stride, channels);
    return xnn_status_invalid_parameter;
  }

  if (output_stride < channels) {
    xnn_log_error(
      "failed to run %s operator with output element stride of %zu: "
      "stride must be at least as large as the number of channels (%zu)",
      xnn_operator_type_to_string(operator_type), output_stride, channels);
    return xnn_status_invalid_parameter;
  }

  struct xnn_operator unary_elementwise_op;
  memset(&unary_elementwise_op, 0, sizeof(unary_elementwise_op));

  init_unary_elementwise_nc(
    flags, /*params=*/NULL, /*params_size=*/0,
    operator_type, unary_elementwise_config, &unary_elementwise_op);

  enum xnn_status status = reshape_unary_elementwise_nc(
    &unary_elementwise_op, operator_type,
    batch_size, channels, input_stride, output_stride,
    log2_input_size, log2_output_size,
    params, params_size,
    threadpool);
  if (status != xnn_status_success){
    return status;
  }

  status = setup_unary_elementwise_nc(&unary_elementwise_op, operator_type, input, output);
  if (status != xnn_status_success){
    return status;
  }

  return xnn_run_operator(&unary_elementwise_op, threadpool);
}

enum xnn_status xnn_run_copy_nc_x32(
    size_t channels,
    size_t input_stride,
    size_t output_stride,
    size_t batch_size,
    const uint32_t* input,
    uint32_t* output,
    uint32_t flags,
    pthreadpool_t threadpool)
{
  return run_unary_elementwise_nc(
    xnn_operator_type_copy_nc_x32,
    channels, input_stride, output_stride, batch_size,
    input, output,
    xnn_init_xx_copy_config(), NULL, 0,
    /*log2_input_size=*/XNN_LOG2_SIZEOF_UINT32_T,
    /*log2_output_size=*/XNN_LOG2_SIZEOF_UINT32_T,
    flags,
    threadpool);
}

// This is an implementation of a deprecated API used by an external project.
// After it is updated, this can be removed.
enum xnn_status xnn_run_convert_nc_f32_f16(
    size_t channels,
    size_t input_stride,
    size_t output_stride,
    size_t batch_size,
    const float* input,
    void* output,
    uint32_t flags,
    pthreadpool_t threadpool)
{
  return xnn_run_unary_elementwise_nc(xnn_unary_convert, xnn_datatype_fp32,
                                      xnn_datatype_fp16, NULL, NULL, NULL,
                                      flags, batch_size, channels, input_stride,
                                      output_stride, threadpool, input, output);
}
