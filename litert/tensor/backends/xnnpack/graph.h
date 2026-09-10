/* Copyright 2026 Google LLC.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#ifndef LITERT_TENSOR_BACKENDS_XNNPACK_GRAPH_H_
#define LITERT_TENSOR_BACKENDS_XNNPACK_GRAPH_H_

#include <cstdint>
#include <memory>
#include <utility>

#include "include/xnnpack.h"
#include "absl/status/status.h"
#include "litert/tensor/backends/common_nnpack/graph.h"
#include "litert/tensor/backends/xnnpack/utils.h"
#include "litert/tensor/utils/macros.h"

namespace litert::tensor {

class XnnpackGraph : public NnpackGraph {
 public:
  struct XnnSubgraphDeleter {
    void operator()(xnn_subgraph_t sg) const { xnn_delete_subgraph(sg); }
  };
  using UniquePtr = std::unique_ptr<xnn_subgraph, XnnSubgraphDeleter>;

  explicit XnnpackGraph(UniquePtr subgraph = nullptr)
      : subgraph_(std::move(subgraph)) {}

  XnnpackGraph(XnnpackGraph&&) = default;
  XnnpackGraph& operator=(XnnpackGraph&&) = default;

  absl::Status ResetSubgraph(uint32_t external_value_ids, uint32_t flags) {
    xnn_subgraph_t subgraph = nullptr;
    LRT_TENSOR_RETURN_IF_ERROR(
        xnn_create_subgraph(external_value_ids, flags, &subgraph));
    subgraph_.reset(subgraph);
    return absl::OkStatus();
  }

  ~XnnpackGraph() override = default;

  xnn_subgraph_t GetSubgraph() const { return subgraph_.get(); }
  xnn_subgraph_t ReleaseSubgraph() { return subgraph_.release(); }

 private:
  UniquePtr subgraph_;
};

}  // namespace litert::tensor

#endif  // LITERT_TENSOR_BACKENDS_XNNPACK_GRAPH_H_
