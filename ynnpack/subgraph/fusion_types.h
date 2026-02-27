// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#ifndef XNNPACK_YNNPACK_SUBGRAPH_FUSION_TYPES_H_
#define XNNPACK_YNNPACK_SUBGRAPH_FUSION_TYPES_H_

#include <cstdint>
#include <map>
#include <vector>

#include "ynnpack/include/ynnpack.h"
#include "ynnpack/subgraph/subgraph.h"

struct subgraph_analysis {
  std::map<uint32_t, ynn_node*> producers;
  std::map<uint32_t, std::vector<ynn_node*>> consumers;

  ynn_node* producer_of(uint32_t id) {
    auto i = producers.find(id);
    return i != producers.end() ? i->second : nullptr;
  }

  explicit subgraph_analysis(ynn_subgraph& subgraph);
};

#endif  // XNNPACK_YNNPACK_SUBGRAPH_FUSION_TYPES_H_
