// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#ifndef THIRD_PARTY_XNNPACK_SRC_SUBGRAPH_SUBGRAPH_UTILS_H_
#define THIRD_PARTY_XNNPACK_SRC_SUBGRAPH_SUBGRAPH_UTILS_H_

#include <stdio.h>

#include "include/xnnpack.h"

#ifdef __cplusplus
extern "C" {
#endif

// Prints a list of the subgraph's values and nodes to the given `out`.
void xnn_subgraph_dump(xnn_subgraph_t subgraph, FILE* out);

#ifdef __cplusplus
}  // extern "C"
#endif

#endif  // THIRD_PARTY_XNNPACK_SRC_SUBGRAPH_SUBGRAPH_UTILS_H_
