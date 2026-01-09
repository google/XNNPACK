// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <cstdint>
#include <functional>
#include <memory>

#include "include/xnnpack.h"
#include <benchmark/benchmark.h>

namespace xnnpack {

using unique_subgraph_ptr =
    std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)>;

unique_subgraph_ptr CreateUniqueSubgraph(uint32_t num_external_values,
                                         uint32_t external_value_flags);

void RunBenchmark(benchmark::State& state,
                  std::function<xnn_subgraph_t()> model_factory,
                  uint32_t extra_flags = 0);

}  // namespace xnnpack
