// Copyright 2019 Google LLC
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
#include "xnnpack/microparams.h"
#include "xnnpack/operator-type.h"
#include "xnnpack/operator-utils.h"
#include "xnnpack/operator.h"
#include "xnnpack/params.h"
#include "pthreadpool.h"

static const struct xnn_binary_elementwise_config* init_config(
    enum xnn_binary_operator type, enum xnn_datatype datatype, int* sign_b) {
  switch (type) {
    case xnn_binary_add:
      switch (datatype) {
        case xnn_datatype_fp32:
          return xnn_init_f32_vadd_config();
        case xnn_datatype_fp16:
          return xnn_init_f16_vadd_config();
        case xnn_datatype_qint8:
          return xnn_init_qs8_vadd_config();
        case xnn_datatype_quint8:
          return xnn_init_qu8_vadd_config();
        default:
          return NULL;
      }
    case xnn_binary_subtract:
      *sign_b = -1;
      switch (datatype) {
        case xnn_datatype_fp32:
          return xnn_init_f32_vsub_config();
        case xnn_datatype_fp16:
          return xnn_init_f16_vsub_config();
        case xnn_datatype_qint8:
          return xnn_init_qs8_vadd_config();
        case xnn_datatype_quint8:
          return xnn_init_qu8_vadd_config();
        default:
          return NULL;
      }
    case xnn_binary_multiply:
      switch (datatype) {
        case xnn_datatype_fp32:
          return xnn_init_f32_vmul_config();
        case xnn_datatype_fp16:
          return xnn_init_f16_vmul_config();
        case xnn_datatype_qint8:
          return xnn_init_qs8_vmul_config();
        case xnn_datatype_quint8:
          return xnn_init_qu8_vmul_config();
        case xnn_datatype_int32:
          return xnn_init_s32_vmul_config();
        default:
          return NULL;
      }
    case xnn_binary_divide:
      switch (datatype) {
        case xnn_datatype_fp32:
          return xnn_init_f32_vdiv_config();
        case xnn_datatype_fp16:
          return xnn_init_f16_vdiv_config();
        default:
          return NULL;
      }
    case xnn_binary_maximum:
      switch (datatype) {
        case xnn_datatype_fp32:
          return xnn_init_f32_vmax_config();
        case xnn_datatype_fp16:
          return xnn_init_f16_vmax_config();
        default:
          return NULL;
      }
    case xnn_binary_minimum:
      switch (datatype) {
        case xnn_datatype_fp32:
          return xnn_init_f32_vmin_config();
        case xnn_datatype_fp16:
          return xnn_init_f16_vmin_config();
        default:
          return NULL;
      }
    case xnn_binary_copysign:
      switch (datatype) {
        case xnn_datatype_fp32:
          return xnn_init_f32_vcopysign_config();
        default:
          return NULL;
      }
    case xnn_binary_squared_difference:
      switch (datatype) {
        case xnn_datatype_fp32:
          return xnn_init_f32_vsqrdiff_config();
        case xnn_datatype_fp16:
          return xnn_init_f16_vsqrdiff_config();
        default:
          return NULL;
      }
    case xnn_binary_prelu:
      switch (datatype) {
        case xnn_datatype_fp32:
          return xnn_init_f32_vprelu_config();
        case xnn_datatype_fp16:
          return xnn_init_f16_vprelu_config();
        default:
          return NULL;
      }
    default:
      return NULL;
  }
}

static enum xnn_status init_binary_elementwise_nd(
    xnn_operator_t op, enum xnn_binary_operator type,
    enum xnn_datatype datatype,
    const struct xnn_quantization_params* a_quantization,
    const struct xnn_quantization_params* b_quantization,
    const struct xnn_quantization_params* output_quantization, uint32_t flags) {
  if ((xnn_params.init_flags & XNN_INIT_FLAG_XNNPACK) == 0) {
    xnn_log_error("failed to create %s operator: XNNPACK is not initialized",
                  xnn_binary_operator_to_string(type));
    return xnn_status_uninitialized;
  }

  int sign_b = 1;
  const struct xnn_binary_elementwise_config* config =
      init_config(type, datatype, &sign_b);
  if (config == NULL) {
    xnn_log_error(
        "failed to create %s operator: unsupported hardware configuration",
        xnn_binary_operator_to_string(type));
    return xnn_status_unsupported_hardware;
  }

  union xnn_binary_uparams uparams;
  union xnn_binary_uparams uparams2;
  if (config->init != NULL) {
    if (datatype == xnn_datatype_qint8 || datatype == xnn_datatype_quint8) {
      if (!a_quantization || !b_quantization || !output_quantization) {
        xnn_log_error(
            "failed to create %s operator with NULL quantization params",
            xnn_binary_operator_to_string(type));
        return xnn_status_invalid_parameter;
      }
      const float a_scale = a_quantization ? a_quantization->scale : 1.0f;
      const float b_scale = b_quantization ? b_quantization->scale : 1.0f;
      const float output_scale =
          output_quantization ? output_quantization->scale : 1.0f;
      if (a_scale <= 0.0f || !isnormal(a_scale)) {
        xnn_log_error(
            "failed to create %s operator with %.7g input 1 scale: scale must be "
            "finite and positive",
            xnn_binary_operator_to_string(type), a_scale);
        return xnn_status_invalid_parameter;
      }
      if (b_scale <= 0.0f || !isnormal(b_scale)) {
        xnn_log_error(
            "failed to create %s operator with %.7g input 2 scale: scale must be "
            "finite and positive",
            xnn_binary_operator_to_string(type), b_scale);
        return xnn_status_invalid_parameter;
      }
      if (output_scale <= 0.0f || !isnormal(output_scale)) {
        xnn_log_error(
            "failed to create %s operator with %.7g output scale: scale must be "
            "finite and positive",
            xnn_binary_operator_to_string(type), output_scale);
        return xnn_status_invalid_parameter;
      }

      struct xnn_quantization_params b_quantization_with_sign = *b_quantization;
      b_quantization_with_sign.scale *= sign_b;

      config->init(&uparams, a_quantization, &b_quantization_with_sign,
                  output_quantization);
      config->init(&uparams2, &b_quantization_with_sign, a_quantization,
                  output_quantization);
    } else {
      config->init(&uparams, NULL, NULL, NULL);
      config->init(&uparams2, NULL, NULL, NULL);
    }
  }

  if ((xnn_params.init_flags & XNN_INIT_FLAG_XNNPACK) == 0) {
    xnn_log_error("failed to create %s operator: XNNPACK is not initialized",
                  xnn_binary_operator_to_string(type));
    return xnn_status_uninitialized;
  }

  memcpy(&op->params, &uparams, sizeof(uparams));
  memcpy(&op->params2, &uparams2, sizeof(uparams2));

  op->binary_elementwise_config = config;
  op->log2_elementwise_element_size =
      xnn_datatype_log2_size_bytes(datatype);

  op->type = xnn_operator_type_binary_elementwise;
  op->flags = flags;

  op->state = xnn_run_state_invalid;

  return xnn_status_success;
}

enum xnn_status xnn_create_binary_elementwise_nd(
    enum xnn_binary_operator type, enum xnn_datatype datatype,
    const struct xnn_quantization_params* a_quantization,
    const struct xnn_quantization_params* b_quantization,
    const struct xnn_quantization_params* output_quantization, uint32_t flags,
    xnn_operator_t* binary_op_out) {
  if ((xnn_params.init_flags & XNN_INIT_FLAG_XNNPACK) == 0) {
    xnn_log_error("failed to create %s operator: XNNPACK is not initialized",
                  xnn_binary_operator_to_string(type));
    return xnn_status_uninitialized;
  }

  xnn_operator_t op =
      xnn_allocate_zero_simd_memory(sizeof(struct xnn_operator));
  if (op == NULL) {
    xnn_log_error("failed to allocate %zu bytes for %s operator descriptor",
                  sizeof(struct xnn_operator),
                  xnn_binary_operator_to_string(type));
    return xnn_status_out_of_memory;
  }

  enum xnn_status status =
      init_binary_elementwise_nd(op, type, datatype, a_quantization,
                                 b_quantization, output_quantization, flags);
  if (status != xnn_status_success) {
    xnn_release_memory(op);
    return status;
  }

  *binary_op_out = op;
  return xnn_status_success;
}

enum xnn_status xnn_reshape_binary_elementwise_nd(xnn_operator_t op,
                                                  size_t num_input1_dims,
                                                  const size_t* input1_shape,
                                                  size_t num_input2_dims,
                                                  const size_t* input2_shape,
                                                  pthreadpool_t threadpool) {
  op->state = xnn_run_state_invalid;

  if (max(num_input1_dims, num_input2_dims) > XNN_MAX_TENSOR_DIMS) {
    xnn_log_error(
        "failed to reshape %s operator with %zu and %zu dimensions in input "
        "shapes: "
        "the number of input dimensions must not exceed %d",
        xnn_operator_type_to_string(op->type), num_input1_dims, num_input2_dims,
        XNN_MAX_TENSOR_DIMS);
    return xnn_status_unsupported_parameter;
  }

  size_t num_compressed_dims = 0;
  size_t compressed_input1_shape[XNN_MAX_TENSOR_DIMS];
  size_t compressed_input2_shape[XNN_MAX_TENSOR_DIMS];
  size_t compressed_output_shape[XNN_MAX_TENSOR_DIMS];
  for (size_t i = 0; i < XNN_MAX_TENSOR_DIMS; i++) {
    compressed_input1_shape[i] = 1;
    compressed_input2_shape[i] = 1;
    compressed_output_shape[i] = 1;
  }
  bool broadcast_input1 = false;
  bool broadcast_input2 = false;
  bool first_nonunit = true;
  bool degenerate_shape = false;
  const size_t num_common_dims = min(num_input1_dims, num_input2_dims);
  for (size_t i = 1; i <= num_common_dims; i++) {
    const size_t input1_dim = input1_shape[num_input1_dims - i];
    const size_t input2_dim = input2_shape[num_input2_dims - i];
    degenerate_shape |= input1_dim == 0;
    degenerate_shape |= input2_dim == 0;
    if (input1_dim == 1 && input2_dim == 1) {
      continue;
    }
    assert(!broadcast_input1 || !broadcast_input2);

    if (input1_dim == 1) {
      if (!broadcast_input1) {
        broadcast_input1 = true;
        broadcast_input2 = false;
        num_compressed_dims++;
      }
      compressed_input2_shape[num_compressed_dims - 1] *= input2_dim;
      compressed_output_shape[num_compressed_dims - 1] *= input2_dim;
    } else if (input2_dim == 1) {
      if (!broadcast_input2) {
        broadcast_input1 = false;
        broadcast_input2 = true;
        num_compressed_dims++;
      }
      compressed_input1_shape[num_compressed_dims - 1] *= input1_dim;
      compressed_output_shape[num_compressed_dims - 1] *= input1_dim;
    } else if (input1_dim == input2_dim) {
      if (broadcast_input1 || broadcast_input2 || first_nonunit) {
        broadcast_input1 = false;
        broadcast_input2 = false;
        num_compressed_dims++;
      }
      compressed_input1_shape[num_compressed_dims - 1] *= input1_dim;
      compressed_input2_shape[num_compressed_dims - 1] *= input1_dim;
      compressed_output_shape[num_compressed_dims - 1] *= input1_dim;
    } else {
      xnn_log_error(
          "failed to reshape %s operator: "
          "shape dimension #%zu of input1 (%zu) does not match shape dimension "
          "#%zu of input2 (%zu)",
          xnn_operator_type_to_string(op->type), num_input1_dims - i,
          input1_dim, num_input2_dims - i, input2_dim);
      return xnn_status_invalid_parameter;
    }
    first_nonunit = false;
  }
  if (num_input1_dims > num_input2_dims) {
    if (!broadcast_input2) {
      num_compressed_dims++;
    }
    for (size_t i = 0; i < num_input1_dims - num_input2_dims; i++) {
      const size_t input1_dim = input1_shape[i];
      degenerate_shape |= input1_dim == 0;
      compressed_input1_shape[num_compressed_dims - 1] *= input1_dim;
      compressed_output_shape[num_compressed_dims - 1] *= input1_dim;
    }
  } else if (num_input2_dims > num_input1_dims) {
    if (!broadcast_input1) {
      num_compressed_dims++;
    }
    for (size_t i = 0; i < num_input2_dims - num_input1_dims; i++) {
      const size_t input2_dim = input2_shape[i];
      degenerate_shape |= input2_dim == 0;
      compressed_input2_shape[num_compressed_dims - 1] *= input2_dim;
      compressed_output_shape[num_compressed_dims - 1] *= input2_dim;
    }
  }
  num_compressed_dims = max(num_compressed_dims, 1);

  // Early exit without setting up context if any shape dimension is zero.
  if (degenerate_shape) {
    op->state = xnn_run_state_skip;
    return xnn_status_success;
  }

  const uint32_t log2_element_size = op->log2_elementwise_element_size;
  op->context.elementwise_binary = (struct elementwise_binary_context){
      .elements = compressed_output_shape[0] << log2_element_size,
  };
  memcpy(&op->context.elementwise_binary.params, &op->params.binary,
         sizeof(op->params.binary));

  const size_t* compressed_a_shape = compressed_input1_shape;
  const size_t* compressed_b_shape = compressed_input2_shape;
  if (compressed_input1_shape[0] == 1) {
    op->context.elementwise_binary.flip_a_b = true;
    op->context.elementwise_binary.ukernel =
        op->binary_elementwise_config->ropc_ukernel;
    compressed_a_shape = compressed_input2_shape;
    compressed_b_shape = compressed_input1_shape;
    memcpy(&op->context.elementwise_binary.params, &op->params2.binary,
           sizeof(op->params.binary));
  } else if (compressed_input2_shape[0] == 1) {
    op->context.elementwise_binary.ukernel =
        op->binary_elementwise_config->opc_ukernel;
  } else if (compressed_input1_shape[0] == compressed_input2_shape[0]) {
    op->context.elementwise_binary.ukernel =
        op->binary_elementwise_config->op_ukernel;
  }
  size_t a_stride = compressed_a_shape[0];
  size_t b_stride = compressed_b_shape[0];
  size_t y_stride = compressed_output_shape[0];
  for (size_t i = 1; i < num_compressed_dims; i++) {
    if (compressed_a_shape[i] != 1) {
      op->context.elementwise_binary.a_stride[XNN_MAX_TENSOR_DIMS - 1 - i] =
          a_stride << log2_element_size;
    }
    if (compressed_b_shape[i] != 1) {
      op->context.elementwise_binary.b_stride[XNN_MAX_TENSOR_DIMS - 1 - i] =
          b_stride << log2_element_size;
    }
    op->context.elementwise_binary.y_stride[XNN_MAX_TENSOR_DIMS - 1 - i] =
        y_stride << log2_element_size;
    a_stride *= compressed_a_shape[i];
    b_stride *= compressed_b_shape[i];
    y_stride *= compressed_output_shape[i];
  }

  const size_t num_threads = pthreadpool_get_threads_count(threadpool);
  const size_t element_tile = op->binary_elementwise_config->element_tile;
  if (compressed_output_shape[5] == 1) {
    if (compressed_output_shape[4] == 1) {
      if (compressed_output_shape[3] == 1) {
        if (compressed_output_shape[2] == 1) {
          if (compressed_output_shape[1] == 1) {
            op->context.elementwise_binary.a_stride[4] =
                compressed_a_shape[0] == 1 ? 0 : (1 << log2_element_size);
            op->context.elementwise_binary.b_stride[4] =
                compressed_b_shape[0] == 1 ? 0 : (1 << log2_element_size);
            op->context.elementwise_binary.y_stride[4] =
                (1 << log2_element_size);
            op->context.elementwise_binary.elements = (1 << log2_element_size);
            op->compute[0].type = xnn_parallelization_type_1d_tile_1d;
            op->compute[0].task_1d_tile_1d = (pthreadpool_task_1d_tile_1d_t)
                xnn_compute_elementwise_binary_1d_tile;
            op->compute[0].range[0] =
                compressed_output_shape[0] * (1 << log2_element_size);
            op->compute[0].tile[0] =
                max(element_tile,
                    round_up_po2(op->compute[0].range[0] / num_threads,
                                 (1 << log2_element_size)));
          } else {
            op->compute[0].type = xnn_parallelization_type_1d;
            op->compute[0].task_1d =
                (pthreadpool_task_1d_t)xnn_compute_elementwise_binary_1d;
            op->compute[0].range[0] = compressed_output_shape[1];
          }
        } else {
          op->compute[0].type = xnn_parallelization_type_2d;
          op->compute[0].task_2d =
              (pthreadpool_task_2d_t)xnn_compute_elementwise_binary_2d;
          op->compute[0].range[0] = compressed_output_shape[2];
          op->compute[0].range[1] = compressed_output_shape[1];
        }
      } else {
        op->compute[0].type = xnn_parallelization_type_3d;
        op->compute[0].task_3d =
            (pthreadpool_task_3d_t)xnn_compute_elementwise_binary_3d;
        op->compute[0].range[0] = compressed_output_shape[3];
        op->compute[0].range[1] = compressed_output_shape[2];
        op->compute[0].range[2] = compressed_output_shape[1];
      }
    } else {
      op->compute[0].type = xnn_parallelization_type_4d;
      op->compute[0].task_4d =
          (pthreadpool_task_4d_t)xnn_compute_elementwise_binary_4d;
      op->compute[0].range[0] = compressed_output_shape[4];
      op->compute[0].range[1] = compressed_output_shape[3];
      op->compute[0].range[2] = compressed_output_shape[2];
      op->compute[0].range[3] = compressed_output_shape[1];
    }
  } else {
    op->compute[0].type = xnn_parallelization_type_5d;
    op->compute[0].task_5d =
        (pthreadpool_task_5d_t)xnn_compute_elementwise_binary_5d;
    op->compute[0].range[0] = compressed_output_shape[5];
    op->compute[0].range[1] = compressed_output_shape[4];
    op->compute[0].range[2] = compressed_output_shape[3];
    op->compute[0].range[3] = compressed_output_shape[2];
    op->compute[0].range[4] = compressed_output_shape[1];
  }
  op->state = xnn_run_state_needs_setup;

  return xnn_status_success;
}

enum xnn_status xnn_setup_binary_elementwise_nd(xnn_operator_t op,
                                                const void* input1,
                                                const void* input2,
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

  op->context.elementwise_binary.a = input1;
  op->context.elementwise_binary.b = input2;
  op->context.elementwise_binary.y = output;

  if (op->context.elementwise_binary.flip_a_b) {
    op->context.elementwise_binary.a = input2;
    op->context.elementwise_binary.b = input1;
  }

  op->state = xnn_run_state_ready;

  return xnn_status_success;
}

enum xnn_status xnn_run_binary_elementwise_nd(
    enum xnn_binary_operator type, enum xnn_datatype datatype,
    const struct xnn_quantization_params* input1_quantization,
    const struct xnn_quantization_params* input2_quantization,
    const struct xnn_quantization_params* output_quantization, uint32_t flags,
    size_t num_input1_dims, const size_t* input1_shape, size_t num_input2_dims,
    const size_t* input2_shape, const void* input1, const void* input2,
    void* output, pthreadpool_t threadpool) {
  struct xnn_operator op;
  memset(&op, 0, sizeof(op));

  enum xnn_status status = init_binary_elementwise_nd(
      &op, type, datatype, input1_quantization, input2_quantization,
      output_quantization, flags);
  if (status != xnn_status_success) {
    return status;
  }

  status = xnn_reshape_binary_elementwise_nd(&op, num_input1_dims, input1_shape,
                                             num_input2_dims, input2_shape,
                                             threadpool);
  if (status != xnn_status_success) {
    return status;
  }

  status = xnn_setup_binary_elementwise_nd(&op, input1, input2, output);
  if (status != xnn_status_success) {
    return status;
  }

  return xnn_run_operator(&op, threadpool);
}
