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

#ifndef LITERT_TENSOR_BACKENDS_XNNPACK_ARITHMETIC_H_
#define LITERT_TENSOR_BACKENDS_XNNPACK_ARITHMETIC_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "include/xnnpack.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "litert/tensor/arithmetic_graph.h"
#include "litert/tensor/buffer.h"
#include "litert/tensor/internal/graph.h"
#include "litert/tensor/internal/mixin.h"

struct xnn_subgraph;

namespace litert::tensor {

// Tag to identify the XNNPACK mixin.
struct XnnpackMixinTag {};

struct XnnpackValue {
  graph::TensorInformation info;
  uint32_t id = XNN_INVALID_VALUE_ID;
  uint32_t flags = 0;
  LockedBufferSpan<const std::byte> data =
      LockedBufferSpan<const std::byte>::Empty();
};

class XnnpackGraph;
class TensorHandle;

// Context for building an XNNPACK subgraph.
class XnnpackBuildContext {
 public:
  explicit XnnpackBuildContext(
      std::vector<TensorHandle> outputs,
      absl::flat_hash_map<graph::Tensor, uint32_t> external_ids = {});
  ~XnnpackBuildContext();
  absl::Status Init();
  absl::StatusOr<std::unique_ptr<XnnpackGraph>> Finalize();
  // Defines a tensor in the XNNPACK subgraph.
  absl::StatusOr<uint32_t> DefineValue(const graph::Tensor& tensor);
  // Returns the XNNPACK subgraph.
  ::xnn_subgraph* subgraph();

 private:
  xnn_subgraph* subgraph_ = nullptr;
  std::vector<graph::Tensor> outputs_;
  std::vector<XnnpackValue> values_;
  absl::flat_hash_map<graph::Tensor, size_t> tensor_index_;
  absl::flat_hash_set<graph::Tensor> external_outputs_;
  absl::flat_hash_map<graph::Tensor, uint32_t> external_ids_;
  std::vector<std::vector<float>> dequantized_buffers_;

  friend absl::StatusOr<std::unique_ptr<XnnpackGraph>> BuildXnnpackGraph(
      std::vector<TensorHandle> outputs);
};

// Base class for XNNPACK operations.
class XnnpackOperation : virtual public graph::Operation {
 public:
  // Converts the operation to XNNPACK.
  virtual absl::Status ToXnnpack(XnnpackBuildContext& ctx) const = 0;
};

namespace graph {

// XNNPACK mixin for the Add operation.
template <>
class OpMixin<AddOperationTag, XnnpackMixinTag> : public XnnpackOperation {
 public:
  // Converts the Add operation to XNNPACK.
  absl::Status ToXnnpack(XnnpackBuildContext& ctx) const override;
};

template <>
class OpMixin<MulOperationTag, XnnpackMixinTag> : public XnnpackOperation {
 public:
  absl::Status ToXnnpack(XnnpackBuildContext& ctx) const override;
};

template <>
class OpMixin<SubOperationTag, XnnpackMixinTag> : public XnnpackOperation {
 public:
  absl::Status ToXnnpack(XnnpackBuildContext& ctx) const override;
};

template <>
class OpMixin<DivOperationTag, XnnpackMixinTag> : public XnnpackOperation {
 public:
  absl::Status ToXnnpack(XnnpackBuildContext& ctx) const override;
};

template <>
class OpMixin<MaximumOperationTag, XnnpackMixinTag> : public XnnpackOperation {
 public:
  absl::Status ToXnnpack(XnnpackBuildContext& ctx) const override;
};

template <>
class OpMixin<MinimumOperationTag, XnnpackMixinTag> : public XnnpackOperation {
 public:
  absl::Status ToXnnpack(XnnpackBuildContext& ctx) const override;
};

template <>
class OpMixin<PowOperationTag, XnnpackMixinTag> : public XnnpackOperation {
 public:
  absl::Status ToXnnpack(XnnpackBuildContext& ctx) const override;
};

template <>
class OpMixin<AbsOperationTag, XnnpackMixinTag> : public XnnpackOperation {
 public:
  absl::Status ToXnnpack(XnnpackBuildContext& ctx) const override;
};

template <>
class OpMixin<SquareOperationTag, XnnpackMixinTag> : public XnnpackOperation {
 public:
  absl::Status ToXnnpack(XnnpackBuildContext& ctx) const override;
};

template <>
class OpMixin<RsqrtOperationTag, XnnpackMixinTag> : public XnnpackOperation {
 public:
  absl::Status ToXnnpack(XnnpackBuildContext& ctx) const override;
};

template <>
class OpMixin<SqrtOperationTag, XnnpackMixinTag> : public XnnpackOperation {
 public:
  absl::Status ToXnnpack(XnnpackBuildContext& ctx) const override;
};

template <>
class OpMixin<ExpOperationTag, XnnpackMixinTag> : public XnnpackOperation {
 public:
  absl::Status ToXnnpack(XnnpackBuildContext& ctx) const override;
};

template <>
class OpMixin<LogOperationTag, XnnpackMixinTag> : public XnnpackOperation {
 public:
  absl::Status ToXnnpack(XnnpackBuildContext& ctx) const override;
};

template <>
class OpMixin<CeilOperationTag, XnnpackMixinTag> : public XnnpackOperation {
 public:
  absl::Status ToXnnpack(XnnpackBuildContext& ctx) const override;
};

template <>
class OpMixin<FloorOperationTag, XnnpackMixinTag> : public XnnpackOperation {
 public:
  absl::Status ToXnnpack(XnnpackBuildContext& ctx) const override;
};

template <>
class OpMixin<SignOperationTag, XnnpackMixinTag> : public XnnpackOperation {
 public:
  absl::Status ToXnnpack(XnnpackBuildContext& ctx) const override;
};

template <>
class OpMixin<RoundOperationTag, XnnpackMixinTag> : public XnnpackOperation {
 public:
  absl::Status ToXnnpack(XnnpackBuildContext& ctx) const override;
};

template <>
class OpMixin<NegOperationTag, XnnpackMixinTag> : public XnnpackOperation {
 public:
  absl::Status ToXnnpack(XnnpackBuildContext& ctx) const override;
};

template <>
class OpMixin<TanhOperationTag, XnnpackMixinTag> : public XnnpackOperation {
 public:
  absl::Status ToXnnpack(XnnpackBuildContext& ctx) const override;
};

template <>
class OpMixin<LogisticOperationTag, XnnpackMixinTag> : public XnnpackOperation {
 public:
  absl::Status ToXnnpack(XnnpackBuildContext& ctx) const override;
};

template <>
class OpMixin<CosOperationTag, XnnpackMixinTag> : public XnnpackOperation {
 public:
  absl::Status ToXnnpack(XnnpackBuildContext& ctx) const override;
};

template <>
class OpMixin<CastOperationTag, XnnpackMixinTag> : public XnnpackOperation {
 public:
  absl::Status ToXnnpack(XnnpackBuildContext& ctx) const override;
};

template <>
class OpMixin<ReluOperationTag, XnnpackMixinTag> : public XnnpackOperation {
 public:
  absl::Status ToXnnpack(XnnpackBuildContext& ctx) const override;
};

template <>
class OpMixin<Relu6OperationTag, XnnpackMixinTag> : public XnnpackOperation {
 public:
  absl::Status ToXnnpack(XnnpackBuildContext& ctx) const override;
};

template <>
class OpMixin<LeakyReluOperationTag, XnnpackMixinTag>
    : public XnnpackOperation {
 public:
  absl::Status ToXnnpack(XnnpackBuildContext& ctx) const override;
};

template <>
class OpMixin<EluOperationTag, XnnpackMixinTag> : public XnnpackOperation {
 public:
  absl::Status ToXnnpack(XnnpackBuildContext& ctx) const override;
};

template <>
class OpMixin<HardSwishOperationTag, XnnpackMixinTag>
    : public XnnpackOperation {
 public:
  absl::Status ToXnnpack(XnnpackBuildContext& ctx) const override;
};

template <>
class OpMixin<PReluOperationTag, XnnpackMixinTag> : public XnnpackOperation {
 public:
  absl::Status ToXnnpack(XnnpackBuildContext& ctx) const override;
};

template <>
class OpMixin<L2NormalizationOperationTag, XnnpackMixinTag>
    : public XnnpackOperation {
 public:
  absl::Status ToXnnpack(XnnpackBuildContext& ctx) const override;
};

template <>
class OpMixin<SinOperationTag, XnnpackMixinTag> : public XnnpackOperation {
 public:
  absl::Status ToXnnpack(XnnpackBuildContext& ctx) const override;
};

template <>
class OpMixin<GeluOperationTag, XnnpackMixinTag> : public XnnpackOperation {
 public:
  absl::Status ToXnnpack(XnnpackBuildContext& ctx) const override;
};

template <>
class OpMixin<SoftmaxOperationTag, XnnpackMixinTag> : public XnnpackOperation {
 public:
  absl::Status ToXnnpack(XnnpackBuildContext& ctx) const override;
};

template <>
class OpMixin<AveragePool2DOperationTag, XnnpackMixinTag>
    : public XnnpackOperation {
 public:
  absl::Status ToXnnpack(XnnpackBuildContext& ctx) const override;
};

template <>
class OpMixin<MaxPool2DOperationTag, XnnpackMixinTag>
    : public XnnpackOperation {
 public:
  absl::Status ToXnnpack(XnnpackBuildContext& ctx) const override;
};

template <>
class OpMixin<Conv2DOperationTag, XnnpackMixinTag> : public XnnpackOperation {
 public:
  absl::Status ToXnnpack(XnnpackBuildContext& ctx) const override;
};

template <>
class OpMixin<DepthwiseConv2DOperationTag, XnnpackMixinTag>
    : public XnnpackOperation {
 public:
  absl::Status ToXnnpack(XnnpackBuildContext& ctx) const override;
};

template <>
class OpMixin<FullyConnectedOperationTag, XnnpackMixinTag>
    : public XnnpackOperation {
 public:
  absl::Status ToXnnpack(XnnpackBuildContext& ctx) const override;
};

template <>
class OpMixin<BatchMatMulOperationTag, XnnpackMixinTag>
    : public XnnpackOperation {
 public:
  absl::Status ToXnnpack(XnnpackBuildContext& ctx) const override;
};

template <>
class OpMixin<TransposeOperationTag, XnnpackMixinTag>
    : public XnnpackOperation {
 public:
  absl::Status ToXnnpack(XnnpackBuildContext& ctx) const override;
};

template <>
class OpMixin<MeanOperationTag, XnnpackMixinTag> : public XnnpackOperation {
 public:
  absl::Status ToXnnpack(XnnpackBuildContext& ctx) const override;
};

template <>
class OpMixin<SliceOperationTag, XnnpackMixinTag> : public XnnpackOperation {
 public:
  absl::Status ToXnnpack(XnnpackBuildContext& ctx) const override;
};

template <>
class OpMixin<ConcatenationOperationTag, XnnpackMixinTag>
    : public XnnpackOperation {
 public:
  absl::Status ToXnnpack(XnnpackBuildContext& ctx) const override;
};

template <>
class OpMixin<ReshapeOperationTag, XnnpackMixinTag> : public XnnpackOperation {
 public:
  absl::Status ToXnnpack(XnnpackBuildContext& ctx) const override;
};

template <>
class OpMixin<SqueezeOperationTag, XnnpackMixinTag> : public XnnpackOperation {
 public:
  absl::Status ToXnnpack(XnnpackBuildContext& ctx) const override;
};

template <>
class OpMixin<ExpandDimsOperationTag, XnnpackMixinTag>
    : public XnnpackOperation {
 public:
  absl::Status ToXnnpack(XnnpackBuildContext& ctx) const override;
};

template <>
class OpMixin<TileOperationTag, XnnpackMixinTag> : public XnnpackOperation {
 public:
  absl::Status ToXnnpack(XnnpackBuildContext& ctx) const override;
};

template <>
class OpMixin<ResizeBilinearOperationTag, XnnpackMixinTag>
    : public XnnpackOperation {
 public:
  absl::Status ToXnnpack(XnnpackBuildContext& ctx) const override;
};

template <>
class OpMixin<ResizeNearestNeighborOperationTag, XnnpackMixinTag>
    : public XnnpackOperation {
 public:
  absl::Status ToXnnpack(XnnpackBuildContext& ctx) const override;
};

template <>
class OpMixin<TransposeConvOperationTag, XnnpackMixinTag>
    : public XnnpackOperation {
 public:
  absl::Status ToXnnpack(XnnpackBuildContext& ctx) const override;
};

template <>
class OpMixin<TransposeConv2DOperationTag, XnnpackMixinTag>
    : public XnnpackOperation {
 public:
  absl::Status ToXnnpack(XnnpackBuildContext& ctx) const override;
};

template <>
class OpMixin<GatherOperationTag, XnnpackMixinTag> : public XnnpackOperation {
 public:
  absl::Status ToXnnpack(XnnpackBuildContext& ctx) const override;
};

template <>
class OpMixin<SpaceToDepthOperationTag, XnnpackMixinTag>
    : public XnnpackOperation {
 public:
  absl::Status ToXnnpack(XnnpackBuildContext& ctx) const override;
};

template <>
class OpMixin<DepthToSpaceOperationTag, XnnpackMixinTag>
    : public XnnpackOperation {
 public:
  absl::Status ToXnnpack(XnnpackBuildContext& ctx) const override;
};

template <>
class OpMixin<SplitOperationTag, XnnpackMixinTag> : public XnnpackOperation {
 public:
  absl::Status ToXnnpack(XnnpackBuildContext& ctx) const override;
};
}  // namespace graph

}  // namespace litert::tensor

#endif  // LITERT_TENSOR_BACKENDS_XNNPACK_ARITHMETIC_H_
