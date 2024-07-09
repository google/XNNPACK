// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <assert.h>
#include <inttypes.h>
#include <math.h>
#include <stdint.h>
#include <string.h>

#include "xnnpack.h"
#include "xnnpack/common.h"
#include "xnnpack/log.h"
#include "xnnpack/node-type.h"
#include "xnnpack/operator-type.h"
#include "xnnpack/operator.h"
#include "xnnpack/subgraph-validation.h"
#include "xnnpack/subgraph.h"
#include "pthreadpool.h"

static enum xnn_status create_scaled_dot_product_attention_operator(
  const struct xnn_node* node,
  const struct xnn_value* values,
  size_t num_values,
  struct xnn_operator_data* opdata,
  struct xnn_code_cache* code_cache,
  xnn_weights_cache_t weights_cache)
{
  assert(node->num_inputs == 5);
  assert(node->num_outputs == 1);

  enum xnn_status status;
  switch (node->compute_type) {
    case xnn_compute_type_fp32:
    {
      status = xnn_create_scaled_dot_product_attention_nhtc_f32(
        node->params.scaled_dot_product_attention.cap_type,
        &node->params.scaled_dot_product_attention.cap_tanh_params,
        /*flags=*/0,
        &opdata->operator_objects[0]);
      break;
    }
    case xnn_compute_type_fp16:
    {
      status = xnn_create_scaled_dot_product_attention_nhtc_f16(
        node->params.scaled_dot_product_attention.cap_type,
        &node->params.scaled_dot_product_attention.cap_tanh_params,
        /*flags=*/0,
        &opdata->operator_objects[0]);
      break;
    }
    default:
      XNN_UNREACHABLE;
  }
  return status;
}

static enum xnn_status resize_scaled_dot_product_attention_output_tensor(
  const struct xnn_operator_data* opdata, struct xnn_value* values, size_t num_values, size_t old_workspace_size)
{
  const uint32_t query_id = opdata->inputs[0];
  const struct xnn_value* query = values + query_id;

  const uint32_t value_id = opdata->inputs[2];
  const struct xnn_value* value = values + value_id;

  const uint32_t output_id = opdata->outputs[0];
  struct xnn_value* output = values + output_id;

  const size_t query_batch_size = xnn_shape_multiply_batch_dims(&query->shape, 3);
  const size_t query_num_dims = query->shape.num_dims;
  const size_t query_heads = query->shape.dim[query_num_dims - 3];
  const size_t query_tokens = query->shape.dim[query_num_dims - 2];

  const size_t value_channels = value->shape.dim[value->shape.num_dims - 1];

  const size_t output_batch_size = xnn_shape_multiply_batch_dims(&output->shape, 3);
  const size_t output_num_dims = output->shape.num_dims;

  if (query_num_dims != output_num_dims) {
    xnn_log_error(
      "failed to resize %s operator's output: number of dimensions mismatch, query dims: %zu, output dims: %zu",
        xnn_node_type_to_string(opdata->type), query_num_dims, output_num_dims);
    return xnn_status_invalid_parameter;
  }

  // Update output batch dim(s)
  if (query_batch_size != output_batch_size) {
    for (uint32_t i = 0; i < query_num_dims - 3; ++i) {
      output->shape.dim[i] = query->shape.dim[i];
    }
  }

  // Update output head dim
  output->shape.dim[output_num_dims - 3] = query_heads;

  // Update output token dim
  output->shape.dim[output_num_dims - 2] = query_tokens;

  // Update output channel dim
  output->shape.dim[output_num_dims - 1] = value_channels;

  // Output size after resize
  const size_t new_output_size = xnn_tensor_get_size(output);

  // workspace size after reshape
  const size_t new_workspace_size = opdata->workspace_size;

  if (new_output_size > output->size || new_workspace_size > old_workspace_size) {
    output->size = new_output_size;
    return xnn_status_reallocation_required;
  }

  return xnn_status_success;
}

static enum xnn_status reshape_scaled_dot_product_attention_operator(
  struct xnn_operator_data* opdata,
  struct xnn_value* values,
  size_t num_values,
  pthreadpool_t threadpool)
{
  const uint32_t query_id = opdata->inputs[0];
  assert(query_id != XNN_INVALID_VALUE_ID);
  assert(query_id < num_values);
  const struct xnn_value* query = values + query_id;

  const uint32_t key_id = opdata->inputs[1];
  assert(key_id != XNN_INVALID_VALUE_ID);
  assert(key_id < num_values);
  const struct xnn_value* key = values + key_id;

  const uint32_t value_id = opdata->inputs[2];
  assert(value_id != XNN_INVALID_VALUE_ID);
  assert(value_id < num_values);
  const struct xnn_value* value = values + value_id;

  const uint32_t scale_id = opdata->inputs[3];
  assert(scale_id != XNN_INVALID_VALUE_ID);
  assert(scale_id < num_values);
  const struct xnn_value* scale = values + scale_id;

  const uint32_t mask_id = opdata->inputs[4];
  assert(mask_id != XNN_INVALID_VALUE_ID);
  assert(mask_id < num_values);
  const struct xnn_value* mask = values + mask_id;

  enum xnn_status status = xnn_status_success;

  const size_t query_num_dims = query->shape.num_dims;
  if (query_num_dims < 3) {
    xnn_log_error(
      "failed to reshape %s operator with query ID #%" PRIu32 ": query must have at least 3 dimensions",
      xnn_node_type_to_string(opdata->type), query_id);
    return xnn_status_invalid_parameter;
  }

  const size_t batch_size = xnn_shape_multiply_batch_dims(&query->shape, 3);
  const size_t query_heads = query->shape.dim[query_num_dims - 3];
  const size_t query_tokens = query->shape.dim[query_num_dims - 2];
  const size_t query_channels = query->shape.dim[query_num_dims - 1];

  const size_t key_num_dims = key->shape.num_dims;
  if (key_num_dims < 2) {
    xnn_log_error(
      "failed to reshape %s operator with key ID #%" PRIu32 ": key must have at least 2 dimensions",
      xnn_node_type_to_string(opdata->type), key_id);
    return xnn_status_invalid_parameter;
  }

  if (key_num_dims != query_num_dims && key_num_dims != query_num_dims - 1) {
    xnn_log_error(
      "failed to reshape %s operator with key ID #%" PRIu32 ": key must have either the same number of dimensions as or"
      " 1 less dimension than query", xnn_node_type_to_string(opdata->type), key_id);
    return xnn_status_invalid_parameter;
  }

  const size_t num_batch_dims = query_num_dims - 3;
  const bool is_multi_query = key_num_dims == query_num_dims - 1;
  const size_t key_tokens = key->shape.dim[key_num_dims - 2];
  const size_t key_channels = key->shape.dim[key_num_dims - 1];

  status = xnn_subgraph_check_batch_dims_match(opdata->type, query_id, query, key_id, key, num_batch_dims);
  if (status != xnn_status_success) {
    return status;
  }

  if (!is_multi_query) {
    const size_t key_heads = key->shape.dim[key_num_dims - 3];
    if (key_heads != query_heads) {
      xnn_log_error(
        "failed to reshape %s operator with key ID #%" PRIu32 ": key heads (%zu) must be equals to query heads (%zu)",
        xnn_node_type_to_string(opdata->type), key_id, key_heads, query_heads);
      return xnn_status_invalid_parameter;
    }
  }

  if (key_channels != query_channels) {
    xnn_log_error(
      "failed to reshape %s operator with key ID #%" PRIu32 ": key channels (%zu) must be equals to query channels"
      " (%zu)", xnn_node_type_to_string(opdata->type), key_id, key_channels, query_channels);
    return xnn_status_invalid_parameter;
  }

  const size_t value_num_dims = value->shape.num_dims;
  const size_t value_tokens = value->shape.dim[value_num_dims - 2];
  const size_t value_channels = value->shape.dim[value_num_dims - 1];

  status = xnn_subgraph_check_batch_dims_match(opdata->type, query_id, query, value_id, value, num_batch_dims);
  if (status != xnn_status_success) {
    return status;
  }


  if (!is_multi_query) {
    const size_t value_heads = value->shape.dim[value_num_dims - 3];
    if (value_heads != query_heads) {
      xnn_log_error(
        "failed to reshape %s operator with value ID #%" PRIu32 ": value heads (%zu) must be equals to query (%zu)",
        xnn_node_type_to_string(opdata->type), value_id, value_heads, query_heads);
      return xnn_status_invalid_parameter;
    }

    const size_t key_heads = key->shape.dim[key_num_dims - 3];
    if (key_heads != value_heads) {
      xnn_log_error(
        "failed to reshape %s operator with key ID #%" PRIu32" and value ID #%" PRIu32 ": key heads (%zu) must be "
        "equal to value heads (%zu)", xnn_node_type_to_string(opdata->type), key_id, value_id, key_heads, value_heads);
      return xnn_status_invalid_parameter;
    }
  }

  if (key_tokens != value_tokens) {
    xnn_log_error(
      "failed to reshape %s operator with key ID #%" PRIu32" and value ID #%" PRIu32 ": key tokens (%zu) must be equal "
      "to value tokens (%zu)", xnn_node_type_to_string(opdata->type), key_id, value_id, key_tokens, value_tokens);
    return xnn_status_invalid_parameter;
  }

  if (scale->shape.dim[0] != query_channels) {
    xnn_log_error(
      "failed to reshape %s operator with scale ID #%" PRIu32 ": scale channels (%zu) must be equal to query channels "
      "(%zu)", xnn_node_type_to_string(opdata->type), scale_id, scale->shape.dim[0], query_channels);
    return xnn_status_invalid_parameter;
  }

  if (mask->shape.dim[0] != query_tokens) {
    xnn_log_error(
      "failed to reshape %s operator with mask ID #%" PRIu32 ": mask query tokens (%zu) must be equal to query tokens "
      "(%zu)", xnn_node_type_to_string(opdata->type), mask_id, mask->shape.dim[0], query_tokens);
    return xnn_status_invalid_parameter;
  }

  if (mask->shape.dim[1] != key_tokens) {
    xnn_log_error(
      "failed to reshape %s operator with mask ID #%" PRIu32 ": mask key/value tokens (%zu) must be equal to key/value "
      "tokens (%zu)", xnn_node_type_to_string(opdata->type), mask_id, mask->shape.dim[1], key_tokens);
    return xnn_status_invalid_parameter;
  }

  const uint32_t output_id = opdata->outputs[0];
  assert(output_id != XNN_INVALID_VALUE_ID);
  assert(output_id < num_values);
  const struct xnn_value* output = values + output_id;

  const size_t output_num_dims = output->shape.num_dims;
  const size_t output_heads = output->shape.dim[output_num_dims - 3];
  const size_t output_tokens = output->shape.dim[output_num_dims - 2];
  const size_t output_channels = output->shape.dim[output_num_dims - 1];

  status = xnn_subgraph_check_batch_dims_match(opdata->type, query_id, query, output_id, output, num_batch_dims);
  if (status != xnn_status_success) {
    return status;
  }

  if (output_heads != query_heads) {
    xnn_log_error(
      "failed to reshape %s operator with output ID #%" PRIu32 ": output heads (%zu) must be equals to query heads"
      " (%zu)", xnn_node_type_to_string(opdata->type), output_id, output_heads, query_heads);
    return xnn_status_invalid_parameter;
  }

  if (output_tokens != query_tokens) {
    xnn_log_error(
      "failed to reshape %s operator with output ID #%" PRIu32 ": output tokens (%zu) must be equals to query tokens"
      " (%zu)", xnn_node_type_to_string(opdata->type), output_id, output_tokens, query_tokens);
    return xnn_status_invalid_parameter;
  }


  if (output_channels != value_channels) {
    xnn_log_error(
      "failed to reshape %s operator with output ID #%" PRIu32 ": output channels (%zu) must be equals to value "
      "channels (%zu)", xnn_node_type_to_string(opdata->type), output_id, output_channels, value_channels);
    return xnn_status_invalid_parameter;
  }

  const size_t key_heads = is_multi_query ? 1 : key->shape.dim[key_num_dims - 3];
  const size_t old_workspace_size = opdata->workspace_size;
  status = xnn_status_invalid_state;

  switch (opdata->operator_objects[0]->type) {
    case xnn_operator_type_scaled_dot_product_attention_nhtc_f32:
      status = xnn_reshape_scaled_dot_product_attention_nhtc_f32(
        opdata->operator_objects[0],
        batch_size,
        query_heads,
        query_tokens,
        key_heads,
        key_tokens,
        query_channels,
        value_channels,
        &opdata->workspace_size,
        &opdata->workspace_alignment,
        threadpool);
      break;
    case xnn_operator_type_scaled_dot_product_attention_nhtc_f16:
      status = xnn_reshape_scaled_dot_product_attention_nhtc_f16(
        opdata->operator_objects[0],
        batch_size,
        query_heads,
        query_tokens,
        key_heads,
        key_tokens,
        query_channels,
        value_channels,
        &opdata->workspace_size,
        &opdata->workspace_alignment,
        threadpool);
      break;
    default:
      XNN_UNREACHABLE;
  }

  if (status != xnn_status_success) {
    return status;
  }

  // Resize the output tensor.
  return resize_scaled_dot_product_attention_output_tensor(opdata, values, num_values, old_workspace_size);
}

static enum xnn_status setup_scaled_dot_product_attention_operator(
  const struct xnn_operator_data* opdata,
  const struct xnn_value* values,
  size_t num_values,
  pthreadpool_t threadpool)
{
  const uint32_t query_id = opdata->inputs[0];
  assert(query_id != XNN_INVALID_VALUE_ID);
  assert(query_id < num_values);
  const struct xnn_value* query = values + query_id;
  const void* query_data = query->data;
  assert(query_data != NULL);

  const uint32_t key_id = opdata->inputs[1];
  assert(key_id != XNN_INVALID_VALUE_ID);
  assert(key_id < num_values);
  const struct xnn_value* key = values + key_id;
  const void* key_data = key->data;
  assert(key_data != NULL);

  const uint32_t value_id = opdata->inputs[2];
  assert(value_id != XNN_INVALID_VALUE_ID);
  assert(value_id < num_values);
  const struct xnn_value* value = values + value_id;
  const void* attention_value_data = value->data;
  assert(attention_value_data != NULL);

  const uint32_t scale_id = opdata->inputs[3];
  assert(scale_id != XNN_INVALID_VALUE_ID);
  assert(scale_id < num_values);
  const struct xnn_value* scale = values + scale_id;
  const void* scale_data = scale->data;
  assert(scale_data != NULL);

  const uint32_t mask_id = opdata->inputs[4];
  assert(mask_id != XNN_INVALID_VALUE_ID);
  assert(mask_id < num_values);
  const struct xnn_value* mask = values + mask_id;
  const void* mask_data = mask->data;
  assert(mask_data != NULL);

  const uint32_t output_id = opdata->outputs[0];
  assert(output_id != XNN_INVALID_VALUE_ID);
  assert(output_id < num_values);
  const struct xnn_value* output = values + output_id;
  void* output_data = output->data;
  assert(output_data != NULL);

  switch (opdata->operator_objects[0]->type) {
    case xnn_operator_type_scaled_dot_product_attention_nhtc_f32:
      return xnn_setup_scaled_dot_product_attention_nhtc_f32(
        opdata->operator_objects[0],
        opdata->workspace,
        query_data,
        key_data,
        attention_value_data,
        scale_data,
        mask_data,
        output_data);
    case xnn_operator_type_scaled_dot_product_attention_nhtc_f16:
      return xnn_setup_scaled_dot_product_attention_nhtc_f16(
        opdata->operator_objects[0],
        opdata->workspace,
        query_data,
        key_data,
        attention_value_data,
        scale_data,
        mask_data,
        output_data);
    default:
      XNN_UNREACHABLE;
  }
}

static enum xnn_status check_inputs(
  xnn_subgraph_t subgraph,
  uint32_t input_id)
{
  const enum xnn_node_type node_type = xnn_node_type_scaled_dot_product_attention;
  enum xnn_status status = xnn_subgraph_check_input_node_id(node_type, input_id, subgraph->num_values);
  if (status != xnn_status_success) {
    return status;
  }

  const struct xnn_value* input_value = &subgraph->values[input_id];
  status = xnn_subgraph_check_input_type_dense(node_type, input_id, input_value);
  if (status != xnn_status_success) {
    return status;
  }

  switch (input_value->datatype) {
    case xnn_datatype_fp16:
    case xnn_datatype_fp32:
      break;
    default:
      xnn_log_error(
        "failed to define %s operator with input ID #%" PRIu32 ": unsupported Value datatype %s (%d)",
        xnn_node_type_to_string(node_type), input_id,
        xnn_datatype_to_string(input_value->datatype), input_value->datatype);
      return xnn_status_invalid_parameter;
  }

  return status;
}

enum xnn_status xnn_define_scaled_dot_product_attention(
  xnn_subgraph_t subgraph,
  enum xnn_attention_logits_cap_type cap_type,
  const void* cap_params,
  uint32_t query_id,
  uint32_t key_id,
  uint32_t value_id,
  uint32_t scale_id,
  uint32_t mask_id,
  uint32_t output_id,
  uint32_t flags)
{
  const enum xnn_node_type node_type = xnn_node_type_scaled_dot_product_attention;
  enum xnn_status status = xnn_subgraph_check_xnnpack_initialized(node_type);
  if (status != xnn_status_success) {
    return status;
  }

  if (cap_type == xnn_attention_logits_cap_type_tanh) {
    const struct xnn_attention_logits_cap_tanh_params* p =
      (const struct xnn_attention_logits_cap_tanh_params*) cap_params;
    const float cap_value = p->cap;
    if (!isfinite(cap_value) || cap_value <= 0.0f) {
      xnn_log_error("failed to define %s operator with logits cap tanh: cap (%f) must be finite and positive",
                    xnn_node_type_to_string(node_type), cap_value);
      return xnn_status_invalid_parameter;
    }
  }

  // Query is [N, H, T, C].
  status = check_inputs(subgraph, query_id);
  if (status != xnn_status_success) {
    return status;
  }
  const struct xnn_value* query = &subgraph->values[query_id];

  // Query must have at least 3 dimensions.
  const size_t query_num_dims = query->shape.num_dims;
  if (query_num_dims < 3) {
    xnn_log_error(
      "failed to define %s operator with query ID #%" PRIu32 ": query must have at least 3 dimensions",
      xnn_node_type_to_string(node_type), query_id);
    return xnn_status_invalid_parameter;
  }

  const size_t heads = query->shape.dim[query_num_dims - 3];
  const size_t query_tokens = query->shape.dim[query_num_dims - 2];
  const size_t channels = query->shape.dim[query_num_dims - 1];

  // Key can be [N, H, U, C] (multi-head) or [N, U, C] (multi-query).
  status = check_inputs(subgraph, key_id);
  if (status != xnn_status_success) {
    return status;
  }
  const struct xnn_value* key = &subgraph->values[key_id];

  // Key must have at least 2 dimensions.
  const size_t key_num_dims = key->shape.num_dims;
  if (key_num_dims < 2) {
    xnn_log_error(
      "failed to define %s operator with key ID #%" PRIu32 ": key must have at least 2 dimensions",
      xnn_node_type_to_string(node_type), key_id);
    return xnn_status_invalid_parameter;
  }

  // Key must have either the same dimensions as query or 1 less.
  if (key_num_dims != query_num_dims && key_num_dims != query_num_dims - 1) {
    xnn_log_error(
      "failed to define %s operator with key ID #%" PRIu32 ": key (%zu) must have either the same dimensions as or 1 "
      "less than query (%zu)", xnn_node_type_to_string(node_type), key_id, key_num_dims, query_num_dims);
    return xnn_status_invalid_parameter;
  }

  const size_t num_batch_dims = query_num_dims - 3;
  const bool is_multi_query = key_num_dims == query_num_dims - 1;

  status = xnn_subgraph_check_batch_dims_match(node_type, query_id, query, key_id, key, num_batch_dims);
  if (status != xnn_status_success) {
    return status;
  }

  if (!is_multi_query) {
    const size_t key_heads_dim = key_num_dims - 3;
    // Key heads must match query.
    if (key->shape.dim[key_heads_dim] != heads) {
      xnn_log_error(
        "failed to define %s operator with key ID #%" PRIu32 ": key heads (%zu) must be equal to query heads (%zu)",
        xnn_node_type_to_string(node_type), key_id, key->shape.dim[key_heads_dim], heads);
      return xnn_status_invalid_parameter;
    }
  }

  const size_t key_channels_dim = key_num_dims - 1;
  // Key channels must match query.
  if (key->shape.dim[key_channels_dim] != channels) {
    xnn_log_error(
      "failed to define %s operator with key ID #%" PRIu32 ": key channels (%zu) must be equal to query channels "
      "(%zu)", xnn_node_type_to_string(node_type), key_id, key->shape.dim[key_channels_dim], channels);
    return xnn_status_invalid_parameter;
  }

  // Value can be [N, H, U, D] (multi-head) or [N, U, D] (multi-query).
  status = check_inputs(subgraph, value_id);
  if (status != xnn_status_success) {
    return status;
  }
  const struct xnn_value* value = &subgraph->values[value_id];

  // Value must have at least 2 dimensions.
  const size_t value_num_dims = value->shape.num_dims;
  if (value_num_dims < 2) {
    xnn_log_error(
      "failed to define %s operator with value ID #%" PRIu32 ": value must have at least 2 dimensions",
      xnn_node_type_to_string(node_type), value_id);
    return xnn_status_invalid_parameter;
  }

  // Value must have the same dimensions as key.
  if (value_num_dims != key_num_dims) {
    xnn_log_error(
      "failed to define %s operator with value ID #%" PRIu32 ": value (%zu) must have the same dimensions as key (%zu)",
      xnn_node_type_to_string(node_type), value_id, value_num_dims, key_num_dims);
    return xnn_status_invalid_parameter;
  }

  status = xnn_subgraph_check_batch_dims_match(node_type, query_id, query, value_id, value, num_batch_dims);
  if (status != xnn_status_success) {
    return status;
  }

  // Value heads must match query.
  if (!is_multi_query) {
    const size_t value_heads_dim = value_num_dims - 3;
    if (value->shape.dim[value_heads_dim] != heads) {
      xnn_log_error(
        "failed to define %s operator with value ID #%" PRIu32 ": value heads (%zu) must be equal to query heads (%zu)",
        xnn_node_type_to_string(node_type), value_id, value->shape.dim[value_heads_dim], heads);
      return xnn_status_invalid_parameter;
    }
  }

  // Check that key and value have the same tokens.
  const size_t key_tokens = key->shape.dim[key_num_dims - 2];
  const size_t value_tokens = value->shape.dim[value_num_dims - 2];
  if (key_tokens != value_tokens) {
    xnn_log_error(
      "failed to define %s operator with key ID #%" PRIu32 " and value ID #%" PRIu32 ": key tokens (%zu) must be equal "
      "to value tokens (%zu)", xnn_node_type_to_string(node_type), key_id, value_id, key_tokens, value_tokens);
    return xnn_status_invalid_parameter;
  }

  // Scale is [C].
  status = check_inputs(subgraph, scale_id);
  if (status != xnn_status_success) {
    return status;
  }
  const struct xnn_value* scale = &subgraph->values[scale_id];

  // Scale must have 1 dimension.
  if (scale->shape.num_dims != 1) {
    xnn_log_error(
      "failed to define %s operator with scale ID #%" PRIu32 ": scale must have only 1 dimension, found %zu",
        xnn_node_type_to_string(node_type), scale_id, scale->shape.num_dims);
    return xnn_status_invalid_parameter;
  }

  // Scale channels must match query channels.
  if (scale->shape.dim[0] != channels) {
    xnn_log_error(
      "failed to define %s operator with scale ID #%" PRIu32 ": scale channels (%zu) must be equal to query channels "
      "(%zu)", xnn_node_type_to_string(node_type), scale_id, scale->shape.dim[0], channels);
    return xnn_status_invalid_parameter;
  }

  // Mask is [T, U].
  status = check_inputs(subgraph, mask_id);
  if (status != xnn_status_success) {
    return status;
  }
  const struct xnn_value* mask = &subgraph->values[mask_id];

  // Mask must have 2 dimensions.
  if (mask->shape.num_dims != 2) {
    xnn_log_error(
      "failed to define %s operator with mask ID #%" PRIu32 ": mask must have only 2 dimension, found %zu",
      xnn_node_type_to_string(node_type), mask_id, mask->shape.num_dims);
    return xnn_status_invalid_parameter;
  }

  // Mask query tokens must match query tokens.
  if (mask->shape.dim[0] != query_tokens) {
    xnn_log_error(
      "failed to define %s operator with mask ID #%" PRIu32 ": mask query tokens (%zu) must match query (%zu)",
      xnn_node_type_to_string(node_type), mask_id, mask->shape.dim[0], query_tokens);
    return xnn_status_invalid_parameter;
  }

  // Mask key/value tokens must match key/value tokens.
  if (mask->shape.dim[1] != key_tokens) {
    xnn_log_error(
      "failed to define %s operator with mask ID #%" PRIu32 ": mask key/value tokens (%zu) must match key/value (%zu)",
      xnn_node_type_to_string(node_type), mask_id, mask->shape.dim[1], key_tokens);
    return xnn_status_invalid_parameter;
  }

  status = xnn_subgraph_check_output_node_id(node_type, output_id, subgraph->num_values);
  if (status != xnn_status_success) {
    return status;
  }

  const struct xnn_value* output = &subgraph->values[output_id];
  status = xnn_subgraph_check_output_type_dense(node_type, output_id, output);
  if (status != xnn_status_success) {
    return status;
  }

  // Output must have at least 3 dimensions.
  const size_t output_num_dims = output->shape.num_dims;
  if (output_num_dims < 3) {
    xnn_log_error(
      "failed to define %s operator with output ID #%" PRIu32 ": output must have at least 3 dimensions",
      xnn_node_type_to_string(node_type), output_id);
    return xnn_status_invalid_parameter;
  }

  // Output must have the same dimensions as query.
  if (output_num_dims != query_num_dims) {
    xnn_log_error(
      "failed to define %s operator with output ID #%" PRIu32 ": output (%zu) must have the same dimensions as query "
      "(%zu)", xnn_node_type_to_string(node_type), output_id, output_num_dims, query_num_dims);
    return xnn_status_invalid_parameter;
  }

  // Output batch size must match query.
  status = xnn_subgraph_check_batch_dims_match(node_type, query_id, query, output_id, output, num_batch_dims);
  if (status != xnn_status_success) {
    return status;
  }

  // Output heads must match query.
  const size_t output_heads = output->shape.dim[output_num_dims - 3];
  if (output_heads != heads) {
    xnn_log_error(
      "failed to define %s operator with output ID #%" PRIu32 ": output heads (%zu) must be equal to query heads (%zu)",
      xnn_node_type_to_string(node_type), output_id, output_heads, heads);
    return xnn_status_invalid_parameter;
  }

  // Output tokens must match query.
  const size_t output_tokens = output->shape.dim[output_num_dims - 2];
  if (output_tokens != query_tokens) {
    xnn_log_error(
      "failed to define %s operator with output ID #%" PRIu32 ": output tokens (%zu) must match query (%zu)",
      xnn_node_type_to_string(node_type), output_id, output_tokens, query_tokens);
    return xnn_status_invalid_parameter;
  }

  // Output channels must match value channels.
  const size_t value_channels = value->shape.dim[value_num_dims - 1];
  const size_t output_channels = output->shape.dim[output_num_dims - 1];
  if (output_channels != value_channels) {
    xnn_log_error(
      "failed to define %s operator with output ID #%" PRIu32 ": output channels (%zu) must be equal to value channels "
      "(%zu)", xnn_node_type_to_string(node_type), output_id, output_channels, value_channels);
    return xnn_status_invalid_parameter;
  }

  enum xnn_compute_type compute_type = xnn_compute_type_invalid;
  switch (output->datatype) {
    case xnn_datatype_fp16:
      compute_type = xnn_compute_type_fp16;
      break;
    case xnn_datatype_fp32:
      compute_type = xnn_compute_type_fp32;
      break;
    default:
      xnn_log_error(
        "failed to define %s operator with output ID #%" PRIu32 ": unsupported Value datatype %s (%d)",
        xnn_node_type_to_string(node_type), output_id,
        xnn_datatype_to_string(output->datatype), output->datatype);
      return xnn_status_invalid_parameter;
  }

  struct xnn_node* node = xnn_subgraph_new_node(subgraph);
  if (node == NULL) {
    return xnn_status_out_of_memory;
  }

  node->type = node_type;
  node->compute_type = compute_type;
  node->params.scaled_dot_product_attention.cap_type = cap_type;
  if (cap_type == xnn_attention_logits_cap_type_tanh) {
    memcpy(&node->params.scaled_dot_product_attention.cap_tanh_params, cap_params,
           sizeof(struct xnn_attention_logits_cap_tanh_params));
  }
  node->num_inputs = 5;
  node->inputs[0] = query_id;
  node->inputs[1] = key_id;
  node->inputs[2] = value_id;
  node->inputs[3] = scale_id;
  node->inputs[4] = mask_id;
  node->num_outputs = 1;
  node->outputs[0] = output_id;
  node->flags = flags;

  node->create = create_scaled_dot_product_attention_operator;
  node->reshape = reshape_scaled_dot_product_attention_operator;
  node->setup = setup_scaled_dot_product_attention_operator;

  return xnn_status_success;
}
