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

#ifdef __cplusplus
}  // extern "C"
#endif
