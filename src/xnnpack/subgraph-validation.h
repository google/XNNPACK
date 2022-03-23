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

bool xnn_subgraph_xnnpack_initialized(enum xnn_node_type node_type);
bool xnn_subgraph_valid_input_id(enum xnn_node_type node_type, uint32_t input_id, size_t num_values);
bool xnn_subgraph_valid_nth_input_id(enum xnn_node_type node_type, uint32_t input_id, size_t num_values, size_t nth);

#ifdef __cplusplus
}  // extern "C"
#endif
