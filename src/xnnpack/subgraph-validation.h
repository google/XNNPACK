// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <xnnpack.h>
#include <xnnpack/subgraph.h>

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

#ifdef __cplusplus
}  // extern "C"
#endif
