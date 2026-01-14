// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#ifndef XNNPACK_YNNPACK_SUBGRAPH_FUSION_H_
#define XNNPACK_YNNPACK_SUBGRAPH_FUSION_H_

#include <cstdint>
#include <map>
#include <unordered_set>
#include <vector>

#include "ynnpack/include/ynnpack.h"
#include "ynnpack/subgraph/subgraph.h"

struct subgraph_analysis {
  std::map<uint32_t, ynn_node*> producers;
  std::map<uint32_t, std::vector<ynn_node*>> consumers;

  explicit subgraph_analysis(ynn_subgraph& subgraph);
};

struct subgraph_candidate {
  uint32_t size = 0;
  uint32_t input_id = YNN_INVALID_VALUE_ID;
  uint32_t output_id = YNN_INVALID_VALUE_ID;
  std::unordered_set<uint32_t> values;
  std::unordered_set<const ynn_node*> nodes;
};

// Finds a chain of unary or binary-with-constant nodes ending at `node`.
// The `node` output must be int8 or uint8. The input to the start of the chain
// must be the same as `node`'s output.
// Intermediate nodes may have any type and may have multiple consumers as long
// as the output is still used by `node`.
subgraph_candidate find_subgraph_for_unary_lut(ynn_subgraph& subgraph,
                                               ynn_node& node,
                                               subgraph_analysis& analysis);

#endif  // XNNPACK_YNNPACK_SUBGRAPH_FUSION_H_
