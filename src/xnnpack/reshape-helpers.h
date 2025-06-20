// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#ifndef XNNPACK_SRC_XNNPACK_RESHAPE_HELPERS_H_
#define XNNPACK_SRC_XNNPACK_RESHAPE_HELPERS_H_

#include "include/xnnpack.h"
#include "src/xnnpack/subgraph.h"

enum xnn_status resize_unary_elementwise_output_tensor(
    const struct xnn_operator_data* opdata, struct xnn_runtime_value* values,
    size_t num_values, size_t old_workspace_size, pthreadpool_t threadpool);

enum xnn_status resize_binary_elementwise_output_tensor(
    const struct xnn_operator_data* opdata, struct xnn_runtime_value* values,
    size_t num_values, size_t old_workspace_size, pthreadpool_t threadpool);

#endif  // XNNPACK_SRC_XNNPACK_RESHAPE_HELPERS_H_
