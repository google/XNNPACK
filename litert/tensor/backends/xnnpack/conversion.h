/* Copyright 2025 Google LLC.

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

#ifndef LITERT_TENSOR_BACKENDS_XNNPACK_CONVERSION_H_
#define LITERT_TENSOR_BACKENDS_XNNPACK_CONVERSION_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "include/xnnpack.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "litert/tensor/backends/common_nnpack/conversion.h"
#include "litert/tensor/backends/common_nnpack/graph.h"
#include "litert/tensor/backends/xnnpack/graph.h"
#include "litert/tensor/buffer.h"
#include "litert/tensor/datatypes.h"
#include "litert/tensor/internal/graph.h"
#include "litert/tensor/internal/type_id.h"
#include "litert/tensor/tensor.h"

namespace litert::tensor {

class XnnpackBuildContext;

// Base class for XNNPACK operations.
class XnnpackOperation : public graph::BackendExtension {
 public:
  internal::TypeId GetTypeId() const override {
    return internal::TypeId::Get<XnnpackOperation>();
  }
  // Converts the operation to XNNPACK.
  virtual absl::Status ToXnnpack(const graph::Operation& op,
                                 XnnpackBuildContext& ctx) const = 0;
};

class XnnpackBuildContext : public NnpackBuildContext {
 public:
  using NnpackBuildContext::NnpackBuildContext;

  absl::string_view BackendName() const override { return "XNNPACK"; }
  uint32_t FlagExternalInput() const override {
    return XNN_VALUE_FLAG_EXTERNAL_INPUT;
  }
  uint32_t FlagExternalOutput() const override {
    return XNN_VALUE_FLAG_EXTERNAL_OUTPUT;
  }

  xnn_subgraph_t subgraph() {
    return static_cast<XnnpackGraph&>(*graph_).GetSubgraph();
  }

  void SetWarmFcBuffers(absl::flat_hash_set<const Buffer*> warm_fc_buffers,
                        size_t max_warm_rows, size_t max_warm_blocks) {
    warm_fc_buffers_ = std::move(warm_fc_buffers);
    warm_max_rows_ = max_warm_rows;
    warm_max_blocks_ = max_warm_blocks;
  }

  absl::flat_hash_map<uint32_t, uint32_t>& qdint8_cache() {
    return qdint8_cache_;
  }

 protected:
  absl::Status EnsureInitialized() override;
  std::unique_ptr<NnpackGraph> CreateEmptyGraph() override;
  absl::Status CreateSubgraph(size_t external_value_ids,
                              uint32_t flags) override;
  absl::Status DefineTensorValue(const graph::Tensor& tensor,
                                 NnpackValue& value) override;
  absl::Status DefineConstantTensor(Type datatype,
                                    absl::Span<const size_t> shape,
                                    const void* data, uint32_t* id) override;
  absl::Status LowerOp(const graph::Operation& op) override;

 private:
  absl::flat_hash_set<const Buffer*> warm_fc_buffers_;
  absl::flat_hash_map<uint32_t, uint32_t> qdint8_cache_;
  size_t warm_max_rows_ = 0;
  size_t warm_max_blocks_ = 0;
  const float* warm_dummy_f32_scales_ = nullptr;
  const void* warm_dummy_bf16_scales_ = nullptr;
};

// Builds an XNNPACK graph from the given outputs.
absl::StatusOr<std::unique_ptr<XnnpackGraph>> BuildXnnpackGraph(
    std::vector<TensorHandle> outputs);

// Builds an XNNPACK graph assuming a warm `xnn_weights_cache_t`: constant 2D
// `FullyConnected` weight tensors with FP32/FP16 activations are defined with
// synthetic non-null pointers and dummy positive scales without locking or
// materializing lazy buffers.
absl::StatusOr<std::unique_ptr<XnnpackGraph>> BuildXnnpackGraphWithWarmFcCache(
    std::vector<TensorHandle> outputs);

}  // namespace litert::tensor

#endif  // LITERT_TENSOR_BACKENDS_XNNPACK_CONVERSION_H_
