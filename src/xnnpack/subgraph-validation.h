// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include "xnnpack.h"
#include "xnnpack/subgraph.h"

#ifdef __cplusplus
extern "C" {
#endif

enum xnn_status xnn_subgraph_check_xnnpack_initialized(enum xnn_node_type node_type);
enum xnn_status xnn_subgraph_check_input_node_id(enum xnn_node_type node_type, uint32_t input_id, size_t num_values);
enum xnn_status xnn_subgraph_check_nth_input_node_id(
  enum xnn_node_type node_type,
  uint32_t input_id,
  size_t num_values,
  size_t nth);
enum xnn_status xnn_subgraph_check_input_type_dense(
  enum xnn_node_type node_type,
  uint32_t input_id,
  const struct xnn_value* input_value);
enum xnn_status xnn_subgraph_check_nth_input_type_dense(
  enum xnn_node_type node_type,
  uint32_t input_id,
  const struct xnn_value* input_value,
  size_t nth);
enum xnn_status xnn_subgraph_check_output_node_id(enum xnn_node_type node_type, uint32_t output_id, size_t num_values);
enum xnn_status xnn_subgraph_check_output_type_dense(
  enum xnn_node_type node_type,
  uint32_t output_id,
  const struct xnn_value* output_value);
enum xnn_status xnn_subgraph_check_datatype_matches(
  enum xnn_node_type node_type,
  uint32_t input_id,
  const struct xnn_value* input_value,
  uint32_t output_id,
  const struct xnn_value* output_value);
enum xnn_status xnn_subgraph_check_datatype_matches_two_inputs(
  enum xnn_node_type node_type,
  uint32_t input1_id,
  const struct xnn_value* input1_value,
  uint32_t input2_id,
  const struct xnn_value* input2_value,
  uint32_t output_id,
  const struct xnn_value* output_value);
enum xnn_status xnn_subgraph_check_output_min_max(enum xnn_node_type node_type, float output_min, float output_max);

enum xnn_status xnn_subgraph_check_quantization_parameter_matches(
  enum xnn_node_type node_type,
  uint32_t input_id,
  const struct xnn_value* input_value,
  uint32_t output_id,
  const struct xnn_value* output_value);

// Check that two tensors have the same batch dimensions.
enum xnn_status xnn_subgraph_check_batch_dims_match(
  enum xnn_node_type node_type,
  uint32_t tensor1_id,
  const struct xnn_value* tensor1_value,
  uint32_t tensor2_id,
  const struct xnn_value* tensor2_value,
  size_t num_batch_dims);

#ifdef __cplusplus
}  // extern "C"
#endif
