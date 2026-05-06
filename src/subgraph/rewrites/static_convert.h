// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
#ifndef XNNPACK_SRC_SUBGRAPH_REWRITES_CONSTANT_FOLDING_H_
#define XNNPACK_SRC_SUBGRAPH_REWRITES_CONSTANT_FOLDING_H_

#include "include/xnnpack.h"

#ifdef __cplusplus
extern "C" {
#endif

// Computes static converts and replaces them with their result.
enum xnn_status xnn_subgraph_constant_fold_converts(xnn_subgraph_t subgraph);

// Updates the weight cache with data aliases for static values that were
// converted during a previous call to `xnn_subgraph_constant_fold_converts`.
enum xnn_status xnn_subgraph_alias_constant_folded_data(
    xnn_subgraph_t subgraph, xnn_weights_cache_t cache);

#ifdef __cplusplus
}
#endif

#endif  // XNNPACK_SRC_SUBGRAPH_REWRITES_CONSTANT_FOLDING_H_
