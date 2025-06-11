// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <inttypes.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

#include "include/xnnpack.h"
#include "src/xnnpack/allocation-type.h"
#include "src/xnnpack/common.h"
#include "src/xnnpack/config-types.h"
#include "src/xnnpack/config.h"
#include "src/xnnpack/internal.h"
#include "src/xnnpack/log.h"
#include "src/xnnpack/math.h"
#include "src/xnnpack/node-type.h"
#include "src/xnnpack/operator-type.h"
#include "src/xnnpack/operator.h"
#include "src/xnnpack/subgraph-validation.h"
#include "src/xnnpack/subgraph.h"
#include <pthreadpool.h>

static enum xnn_status create_batch_matrix_multiply_operator(
  const struct xnn_node* node,
  const struct xnn_runtime_value* values,
  size_t num_values,
  struct xnn_operator_data* opdata,
  xnn_weights_cache_t weights_cache)
{
  assert(node->num_inputs == 2);
  assert(node->num_outputs == 1);

  enum xnn_status status;

  const uint32_t input_a_id = opdata->inputs[0];
  assert(input_a_id != XNN_INVALID_VALUE_ID);
  assert(input_a_id < num_values);
  const uint32_t input_b_id = opdata->inputs[1];
  assert(input_b_id != XNN_INVALID_VALUE_ID);
  assert(input_b_id < num_values);
  const enum xnn_datatype inputa_datatype = values[input_a_id].datatype;
  const enum xnn_datatype inputb_datatype = values[input_b_id].datatype;

  const struct xnn_runtime_value* input_b = values + input_b_id;
  // Get the shape and size of the second input.
  size_t batch_size_b = 1;
  size_t k = 0;
  size_t n = 0;
  if (xnn_value_is_static(input_b->allocation_type)) {
    if (input_b->shape.num_dims < 2) {
      xnn_log_error(
          "failed to create %s operator with input_b ID #%" PRIu32
          ": unsupported number of dimension %zu, must be at least 2",
          xnn_node_type_to_string(xnn_node_type_batch_matrix_multiply),
          input_b_id, input_b->shape.num_dims);
      return xnn_status_invalid_parameter;
    }
    for (size_t i = 0; i < input_b->shape.num_dims - 2; i++) {
      batch_size_b *= input_b->shape.dim[i];
    }
    k = node->flags & XNN_FLAG_TRANSPOSE_B
        ? input_b->shape.dim[input_b->shape.num_dims - 1]
        : input_b->shape.dim[input_b->shape.num_dims - 2];
    n = node->flags & XNN_FLAG_TRANSPOSE_B
        ? input_b->shape.dim[input_b->shape.num_dims - 2]
        : input_b->shape.dim[input_b->shape.num_dims - 1];

  }
  switch (inputa_datatype) {
    case xnn_datatype_bf16:
      switch (inputb_datatype) {
        case xnn_datatype_bf16: {
          return xnn_create_batch_matrix_multiply_nc_bf16_f32(
                node->flags, &opdata->operator_objects[0]);
        }
        default:
          XNN_UNREACHABLE;
      }
      break;

    case xnn_datatype_fp16:
      switch (inputb_datatype) {
        case xnn_datatype_fp16: {
          // Get the shape and size of the second input.
          if (xnn_value_is_static(input_b->allocation_type)) {
            return xnn_create_batch_matrix_multiply_nc_f16_const_weights(
                batch_size_b, k, n, input_b->data, node->flags,
                &opdata->operator_objects[0]);
          } else {
            return xnn_create_batch_matrix_multiply_nc_f16(
                node->flags, &opdata->operator_objects[0]);
          }
        }
        default:
          XNN_UNREACHABLE;
      }
      break;
    case xnn_datatype_pfp16:
      switch (inputb_datatype) {
        case xnn_datatype_fp16: {
          // Get the shape and size of the second input.
          if (xnn_value_is_static(input_b->allocation_type)) {
            return xnn_create_batch_matrix_multiply_nc_pf16_const_weights(
                batch_size_b, k, n, input_b->data, node->flags,
                &opdata->operator_objects[0]);
          } else {
            return xnn_create_batch_matrix_multiply_nc_pf16(
                node->flags, &opdata->operator_objects[0]);
          }
        }
        default:
          XNN_UNREACHABLE;
      }
      break;
    case xnn_datatype_fp32:
      switch (inputb_datatype) {
        case xnn_datatype_fp32: {
          // Get the shape and size of the second input.
          if (xnn_value_is_static(input_b->allocation_type)) {
            return xnn_create_batch_matrix_multiply_nc_f32_const_weights(
                batch_size_b, k, n, input_b->data, node->flags,
                &opdata->operator_objects[0]);
          } else {
            return xnn_create_batch_matrix_multiply_nc_f32(
                node->flags, &opdata->operator_objects[0]);
          }
        }
        default:
          XNN_UNREACHABLE;
      }
      break;
    case xnn_datatype_pfp32:
      switch (inputb_datatype) {
        case xnn_datatype_fp32: {
          // Get the shape and size of the second input.
          if (xnn_value_is_static(input_b->allocation_type)) {
            return xnn_create_batch_matrix_multiply_nc_pf32_const_weights(
                batch_size_b, k, n, input_b->data, node->flags,
                &opdata->operator_objects[0]);
          } else {
            return xnn_create_batch_matrix_multiply_nc_pf32(
                node->flags, &opdata->operator_objects[0]);
          }
        }
        default:
          XNN_UNREACHABLE;
      }
      break;
    case xnn_datatype_qdint8: {
      switch (inputb_datatype) {
        case xnn_datatype_qcint8:
          status = xnn_create_batch_matrix_multiply_nc_qd8_f32_qc8w(
              batch_size_b, k, n, input_b->data,
              input_b->quantization.channelwise_scale, node->flags,
              &opdata->operator_objects[0]);
          break;
        default:
          XNN_UNREACHABLE;
      }
      break;
    }
    case xnn_datatype_qpint8: {
      switch (inputb_datatype) {
        case xnn_datatype_qcint8:
          status = xnn_create_batch_matrix_multiply_nc_qp8_f32_qc8w(
              batch_size_b, k, n, input_b->data,
              input_b->quantization.channelwise_scale, node->flags,
              &opdata->operator_objects[0]);
          break;
        default:
          XNN_UNREACHABLE;
      }
      break;
    }
    case xnn_datatype_qduint8: {
      switch (inputb_datatype) {
        case xnn_datatype_qcint8:
          status = xnn_create_batch_matrix_multiply_nc_qdu8_f32_qc8w(
              batch_size_b, k, n, input_b->data,
              input_b->quantization.channelwise_scale, node->flags,
              &opdata->operator_objects[0]);
          break;
        default:
          XNN_UNREACHABLE;
      }
      break;
    }
    default:
      XNN_UNREACHABLE;
  }
  return status;
}

static enum xnn_status reshape_batch_matrix_multiply_operator(
  struct xnn_operator_data* opdata,
  struct xnn_runtime_value* values,
  size_t num_values,
  pthreadpool_t threadpool)
{
  const uint32_t input_a_id = opdata->inputs[0];
  assert(input_a_id != XNN_INVALID_VALUE_ID);
  assert(input_a_id < num_values);
  const uint32_t input_b_id = opdata->inputs[1];
  assert(input_b_id != XNN_INVALID_VALUE_ID);
  assert(input_b_id < num_values);
  const uint32_t output_id = opdata->outputs[0];
  assert(output_id != XNN_INVALID_VALUE_ID);
  assert(output_id < num_values);

  // Get the inputs and outputs.
  const struct xnn_runtime_value* input_a = values + input_a_id;
  const struct xnn_runtime_value* input_b = values + input_b_id;
  struct xnn_runtime_value* output = values + output_id;

  // Verify some basic shape properties of the inputs.
  if (input_a->shape.num_dims < 2) {
    xnn_log_error("failed to reshape %s operator with input_a ID #%" PRIu32
                  ": unsupported number of dimension %zu, must be at least 2",
                  xnn_node_type_to_string(xnn_node_type_batch_matrix_multiply),
                  input_a_id, input_a->shape.num_dims);
    return xnn_status_invalid_parameter;
  }
  if (input_b->shape.num_dims < 2) {
    xnn_log_error("failed to reshape %s operator with input_b ID #%" PRIu32
                  ": unsupported number of dimension %zu, must be at least 2",
                  xnn_node_type_to_string(xnn_node_type_batch_matrix_multiply),
                  input_b_id, input_b->shape.num_dims);
    return xnn_status_invalid_parameter;
  }

  // Extract the dimensions of the inputs. Note that if the number of batch
  // dimensions in `input_a` and `input_b` differ, we left-pad the shorter of
  // the two with ones.
  const size_t num_output_dims =
      max(input_a->shape.num_dims, input_b->shape.num_dims);
  const size_t num_batch_dims = num_output_dims - 2;
  size_t padded_dims_a[XNN_MAX_TENSOR_DIMS];
  size_t padded_dims_b[XNN_MAX_TENSOR_DIMS];
  for (int i = 0; i < XNN_MAX_TENSOR_DIMS; i++) {
    padded_dims_a[i] = 1;
    padded_dims_b[i] = 1;
  }
  memcpy(&padded_dims_a[num_output_dims - input_a->shape.num_dims],
         input_a->shape.dim, input_a->shape.num_dims * sizeof(size_t));
  memcpy(&padded_dims_b[num_output_dims - input_b->shape.num_dims],
         input_b->shape.dim, input_b->shape.num_dims * sizeof(size_t));

  // Validate the dimensions.
  // input_a: [B ..., M, K]
  // input_b: [B ..., K, N] or [B ..., N, K] (transpose_b)
  const size_t m = padded_dims_a[num_output_dims - 2];
  const size_t k = padded_dims_a[num_output_dims - 1];
  const bool transpose_b = (opdata->flags & XNN_FLAG_TRANSPOSE_B) != 0;
  const size_t n =
      padded_dims_b[transpose_b ? num_output_dims - 2 : num_output_dims - 1];
  const size_t k_b =
      padded_dims_b[transpose_b ? num_output_dims - 1 : num_output_dims - 2];
  if (k != k_b) {
    xnn_log_error("failed to reshape %s operator with input_a ID #%" PRIu32
                  " and input_b ID #%" PRIu32
                  ": mismatch at last dimension (%zu != %zu)",
                  xnn_node_type_to_string(xnn_node_type_batch_matrix_multiply),
                  input_a_id, input_b_id, k, k_b);
    return xnn_status_invalid_parameter;
  }

  // Validate the batch dimensions. Valid pairs of dimensions are
  //
  //  * `(N, N)`: the values match,
  //  * `(1, N)` or `(N, 1)`: either of the dimensions is `1`,
  for (size_t i = 0; i < num_batch_dims; i++) {
    if (padded_dims_a[i] != 1 && padded_dims_b[i] != 1 &&
        padded_dims_a[i] != padded_dims_b[i]) {
      xnn_log_error(
          "failed to reshape %s operator with input_a ID #%" PRIu32
          " and input_b ID #%" PRIu32
          ": incompatible dimensions %zu (%zu vs. %zu)",
          xnn_node_type_to_string(xnn_node_type_batch_matrix_multiply),
          input_a_id, input_b_id, i, padded_dims_b[i], padded_dims_a[i]);
      return xnn_status_invalid_parameter;
    }
  }

  // Propagate the reshape.
  const size_t old_workspace_size = opdata->workspace_size;
  enum xnn_status status = xnn_status_invalid_state;
  switch (opdata->operator_objects[0]->type) {
    case xnn_operator_type_batch_matrix_multiply_nc_bf16_f32:
      status = xnn_reshape_batch_matrix_multiply_nc_bf16_f32(
          opdata->operator_objects[0], num_batch_dims, padded_dims_a,
          padded_dims_b, m, k, n, &opdata->workspace_size,
          threadpool);
      break;
    case xnn_operator_type_batch_matrix_multiply_nc_f16:
      status = xnn_reshape_batch_matrix_multiply_nc_f16(
          opdata->operator_objects[0], num_batch_dims, padded_dims_a,
          padded_dims_b, m, k, n, &opdata->workspace_size,
          threadpool);
      break;
    case xnn_operator_type_batch_matrix_multiply_nc_f32:
      status = xnn_reshape_batch_matrix_multiply_nc_f32(
          opdata->operator_objects[0], num_batch_dims, padded_dims_a,
          padded_dims_b, m, k, n, &opdata->workspace_size,
          threadpool);
      break;
    case xnn_operator_type_batch_matrix_multiply_nc_pf16:
      status = xnn_reshape_batch_matrix_multiply_nc_pf16(
          opdata->operator_objects[0], num_batch_dims, padded_dims_a,
          padded_dims_b, m, k, n, &opdata->workspace_size,
          threadpool);
      break;
    case xnn_operator_type_batch_matrix_multiply_nc_pf32:
      status = xnn_reshape_batch_matrix_multiply_nc_pf32(
          opdata->operator_objects[0], num_batch_dims, padded_dims_a,
          padded_dims_b, m, k, n, &opdata->workspace_size,
          threadpool);
      break;
    case xnn_operator_type_batch_matrix_multiply_nc_qd8_f32_qc8w:
      status = xnn_reshape_batch_matrix_multiply_nc_qd8_f32_qc8w(
          opdata->operator_objects[0], num_batch_dims, padded_dims_a,
          padded_dims_b, m, k, n, threadpool);
      break;
    case xnn_operator_type_batch_matrix_multiply_nc_qp8_f32_qc8w:
      status = xnn_reshape_batch_matrix_multiply_nc_qp8_f32_qc8w(
          opdata->operator_objects[0], num_batch_dims, padded_dims_a,
          padded_dims_b, m, k, n, threadpool);
      break;
    case xnn_operator_type_batch_matrix_multiply_nc_qdu8_f32_qc8w:
      status = xnn_reshape_batch_matrix_multiply_nc_qdu8_f32_qc8w(
          opdata->operator_objects[0], num_batch_dims, padded_dims_a,
          padded_dims_b, m, k, n, threadpool);
      break;
    default:
      XNN_UNREACHABLE;
  }
  if (status != xnn_status_success) {
    return status;
  }

  // Set the shape of the output tensor.
  for (int i = 0; i < num_batch_dims; i++) {
    output->shape.dim[i] = max(padded_dims_a[i], padded_dims_b[i]);
  }
  output->shape.num_dims = num_output_dims;
  output->shape.dim[num_output_dims - 2] = m;
  output->shape.dim[num_output_dims - 1] = n;
  const size_t new_size = xnn_runtime_tensor_get_size(output);
  if (new_size > output->size || opdata->workspace_size > old_workspace_size) {
    output->size = new_size;
    return xnn_status_reallocation_required;
  }
  return xnn_status_success;
}

static enum xnn_status setup_batch_matrix_multiply_operator(
  const struct xnn_operator_data* opdata,
  const struct xnn_runtime_value* values,
  size_t num_values,
  pthreadpool_t threadpool)
{
  const uint32_t input_a_id = opdata->inputs[0];
  assert(input_a_id != XNN_INVALID_VALUE_ID);
  assert(input_a_id < num_values);

  const uint32_t input_b_id = opdata->inputs[1];
  assert(input_b_id != XNN_INVALID_VALUE_ID);
  assert(input_b_id < num_values);

  const uint32_t output_id = opdata->outputs[0];
  assert(output_id != XNN_INVALID_VALUE_ID);
  assert(output_id < num_values);

  const struct xnn_runtime_value* input_a = values + input_a_id;
  const void* input_a_data = input_a->data;
  assert(input_a_data != NULL);

  const struct xnn_runtime_value* input_b = values + input_b_id;
  const void* input_b_data = input_b->data;
  assert(input_b_data != NULL);

  const struct xnn_runtime_value* output_value = values + output_id;
  void* output_data = output_value->data;
  assert(output_data != NULL);

  switch (opdata->operator_objects[0]->type) {
    case xnn_operator_type_batch_matrix_multiply_nc_bf16_f32:
      return xnn_setup_batch_matrix_multiply_nc_bf16_f32(
          opdata->operator_objects[0], opdata->workspace, input_a_data,
          input_b_data, output_data);
    case xnn_operator_type_batch_matrix_multiply_nc_f16:
      return xnn_setup_batch_matrix_multiply_nc_f16(
          opdata->operator_objects[0], opdata->workspace, input_a_data,
          input_b_data, output_data);
    case xnn_operator_type_batch_matrix_multiply_nc_f32:
      return xnn_setup_batch_matrix_multiply_nc_f32(
          opdata->operator_objects[0], opdata->workspace, input_a_data,
          input_b_data, output_data);
    case xnn_operator_type_batch_matrix_multiply_nc_pf16:
      return xnn_setup_batch_matrix_multiply_nc_pf16(
          opdata->operator_objects[0], opdata->workspace, input_a_data,
          input_b_data, output_data);
    case xnn_operator_type_batch_matrix_multiply_nc_pf32:
      return xnn_setup_batch_matrix_multiply_nc_pf32(
          opdata->operator_objects[0], opdata->workspace, input_a_data,
          input_b_data, output_data);
    case xnn_operator_type_batch_matrix_multiply_nc_qd8_f32_qc8w:
      return xnn_setup_batch_matrix_multiply_nc_qd8_f32_qc8w(
          opdata->operator_objects[0], input_a_data,
          input_a->quantization.dynamic_params, output_data);
    case xnn_operator_type_batch_matrix_multiply_nc_qp8_f32_qc8w:
      return xnn_setup_batch_matrix_multiply_nc_qp8_f32_qc8w(
          opdata->operator_objects[0], input_a_data, output_data);
    case xnn_operator_type_batch_matrix_multiply_nc_qdu8_f32_qc8w:
      return xnn_setup_batch_matrix_multiply_nc_qdu8_f32_qc8w(
          opdata->operator_objects[0], input_a_data,
          input_a->quantization.dynamic_params, output_data);
    default:
      XNN_UNREACHABLE;
  }
}

static inline bool validate_datatypes(
  enum xnn_datatype input1_datatype,
  enum xnn_datatype input2_datatype,
  enum xnn_datatype output_datatype)
{
  switch (input2_datatype) {
    case xnn_datatype_bf16:
      if (input1_datatype == xnn_datatype_bf16 && output_datatype == xnn_datatype_fp32) {
        return true;
      }
      break;
    case xnn_datatype_fp16:
      if (input1_datatype == xnn_datatype_fp16 && output_datatype == xnn_datatype_fp16) {
        return true;
      }
      break;
    case xnn_datatype_fp32:
      if (input1_datatype == xnn_datatype_fp32 && output_datatype == xnn_datatype_fp32) {
        return true;
      }
      break;
    case xnn_datatype_qcint8:
      if (input1_datatype == xnn_datatype_qdint8 &&
          output_datatype == xnn_datatype_fp32) {
        return true;
      }
      break;
    default:
      XNN_UNREACHABLE;
  }
  return false;
}

static bool datatype_is_packable(enum xnn_datatype datatype) {
  switch (datatype) {
    case xnn_datatype_fp16:
    case xnn_datatype_fp32:
      return true;
    default:
      return false;
  }
}

enum xnn_status xnn_define_batch_matrix_multiply(
  xnn_subgraph_t subgraph,
  uint32_t input1_id,
  uint32_t input2_id,
  uint32_t output_id,
  uint32_t flags)
{
  enum xnn_status status = xnn_subgraph_check_xnnpack_initialized(xnn_node_type_batch_matrix_multiply);
  if (status != xnn_status_success) {
    return status;
  }

  status = xnn_subgraph_check_input_node_id(xnn_node_type_batch_matrix_multiply, input1_id, subgraph->num_values);
  if (status != xnn_status_success) {
    return status;
  }

  const struct xnn_value* input1_value = &subgraph->values[input1_id];
  status = xnn_subgraph_check_input_type_dense(xnn_node_type_batch_matrix_multiply, input1_id, input1_value);
  if (status != xnn_status_success) {
    return status;
  }

  switch (input1_value->datatype) {
    case xnn_datatype_bf16:
    case xnn_datatype_fp16:
    case xnn_datatype_fp32:
      break;
    case xnn_datatype_qdint8:
      if (input1_value->quantization.num_nonbatch_dims >
          input1_value->shape.num_dims) {
        xnn_log_error(
            "failed to define %s operator with input ID #%" PRIu32
            ": num_nonbatch_dims (%zu) must be <= num_dims (%zu)",
            xnn_node_type_to_string(xnn_node_type_batch_matrix_multiply),
            input1_id, input1_value->quantization.num_nonbatch_dims,
            input1_value->shape.num_dims);
        return xnn_status_invalid_parameter;
      }
      break;
    default:
      xnn_log_error(
        "failed to define %s operator with input1 ID #%" PRIu32 ": unsupported Value datatype %s (%d)",
        xnn_node_type_to_string(xnn_node_type_batch_matrix_multiply), input1_id,
        xnn_datatype_to_string(input1_value->datatype), input1_value->datatype);
      return xnn_status_invalid_parameter;
  }

  status = xnn_subgraph_check_input_node_id(xnn_node_type_batch_matrix_multiply, input2_id, subgraph->num_values);
  if (status != xnn_status_success) {
    return status;
  }

  const struct xnn_value* input2_value = &subgraph->values[input2_id];

  status = xnn_subgraph_check_input_type_dense(xnn_node_type_batch_matrix_multiply, input2_id, input1_value);
  if (status != xnn_status_success) {
    return status;
  }

  switch (input2_value->datatype) {
    case xnn_datatype_fp16:
    case xnn_datatype_bf16:
    case xnn_datatype_fp32:
      break;
    case xnn_datatype_qcint8:
      // Check that `input2` is static, which is required for this variant.
      if (!xnn_value_is_static(input2_value->allocation_type)) {
        xnn_log_error(
            "failed to define %s operator with input ID #%" PRIu32
            ": %s input must be static (got %s)",
            xnn_node_type_to_string(xnn_node_type_batch_matrix_multiply),
            input2_id, xnn_datatype_to_string(input2_value->datatype),
            xnn_allocation_type_to_string(input2_value->allocation_type));
        return xnn_status_invalid_parameter;
      }
      break;
    default:
      xnn_log_error(
          "failed to define %s operator with input2 ID #%" PRIu32
          ": unsupported Value datatype %s (%d)",
          xnn_node_type_to_string(xnn_node_type_batch_matrix_multiply),
          input2_id, xnn_datatype_to_string(input2_value->datatype),
          input2_value->datatype);
      return xnn_status_invalid_parameter;
  }

  status = xnn_subgraph_check_output_node_id(xnn_node_type_batch_matrix_multiply, output_id, subgraph->num_values);
  if (status != xnn_status_success) {
    return status;
  }

  const struct xnn_value* output_value = &subgraph->values[output_id];
  status = xnn_subgraph_check_output_type_dense(xnn_node_type_batch_matrix_multiply, output_id, output_value);
  if (status != xnn_status_success) {
    return status;
  }

  switch (output_value->datatype) {
    case xnn_datatype_fp16:
    case xnn_datatype_fp32:
      break;
    default:
      xnn_log_error(
        "failed to define %s operator with output ID #%" PRIu32 ": unsupported Value datatype %s (%d)",
        xnn_node_type_to_string(xnn_node_type_batch_matrix_multiply), output_id,
        xnn_datatype_to_string(output_value->datatype), output_value->datatype);
      return xnn_status_invalid_parameter;
  }

  if (!validate_datatypes(input1_value->datatype, input2_value->datatype, output_value->datatype)) {
    xnn_log_error(
      "failed to define %s operator with input1 ID #%" PRIu32 ", input2 ID #%" PRIu32 ", and output ID #%" PRIu32
      ": mismatching datatypes across input1 (%s), input2 (%s), and output (%s)",
      xnn_node_type_to_string(xnn_node_type_batch_matrix_multiply), input1_id, input2_id, output_id,
      xnn_datatype_to_string(input1_value->datatype),
      xnn_datatype_to_string(input2_value->datatype),
      xnn_datatype_to_string(output_value->datatype));
    return xnn_status_invalid_parameter;
  }

  // If supported, convert the input to a packed datatype.
  const enum xnn_datatype input_datatype = input1_value->datatype;
  const enum xnn_datatype output_datatype = output_value->datatype;
  if (datatype_is_packable(input_datatype)) {
    if (input_datatype == output_datatype) {
      const struct xnn_gemm_config* gemm_config = NULL;
      switch (input_datatype) {
        case xnn_datatype_fp16:
          gemm_config = xnn_init_pf16_gemm_config();
          break;
        case xnn_datatype_fp32:
          gemm_config = xnn_init_pf32_gemm_config();
          break;
        default:
          XNN_UNREACHABLE;
      }
      if (gemm_config != NULL) {
        // Insert a node to pack the LHS.
        uint32_t new_id = XNN_INVALID_VALUE_ID;
        status = xnn_insert_pack_lh_node(subgraph, input1_id, &new_id);
        if (status != xnn_status_success) {
          return status;
        }
        input1_id = new_id;
      }
    }
  }

  struct xnn_node* node = xnn_subgraph_new_node(subgraph);
  if (node == NULL) {
    return xnn_status_out_of_memory;
  }

  node->type = xnn_node_type_batch_matrix_multiply;
  node->num_inputs = 2;
  node->inputs[0] = input1_id;
  node->inputs[1] = input2_id;
  node->num_outputs = 1;
  node->outputs[0] = output_id;
  node->flags = flags;

  node->create = create_batch_matrix_multiply_operator;
  node->setup = setup_batch_matrix_multiply_operator;
  node->reshape = reshape_batch_matrix_multiply_operator;

  return xnn_status_success;
}
