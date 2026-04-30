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

#include "litert/tensor/backends/xnnpack/conversion.h"

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <memory>
#include <utility>
#include <vector>

#include "include/xnnpack.h"
#include "absl/base/call_once.h"
#include "absl/base/no_destructor.h"
#include "absl/container/flat_hash_map.h"
#include "absl/container/flat_hash_set.h"
#include "absl/hash/hash.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/str_cat.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "litert/tensor/backends/xnnpack/arithmetic.h"
#include "litert/tensor/backends/xnnpack/utils.h"
#include "litert/tensor/datatypes.h"
#include "litert/tensor/internal/graph.h"
#include "litert/tensor/internal/graph_traversal.h"
#include "litert/tensor/tensor.h"
#include "litert/tensor/utils/macros.h"

namespace litert::tensor {
namespace {

absl::once_flag g_xnn_init_once;
absl::NoDestructor<absl::Status> g_xnn_init_status(absl::OkStatus());

absl::Status EnsureXnnInitialized() {
  absl::call_once(g_xnn_init_once, []() {
    *g_xnn_init_status =
        XnnStatusToAbsl(xnn_initialize(nullptr), "xnn_initialize");
  });
  return *g_xnn_init_status;
}

xnn_datatype GetXnnpackType(const XnnpackValue& value) {
  switch (value.info.type) {
    case Type::kUnknown:
    case Type::kBOOL:
    case Type::kI2:
    case Type::kI4:
    case Type::kI8: {
      if (value.info.quantization) {
        if (auto it =
                value.info.quantization->As<PerChannelAffineQuantization>();
            it.ok()) {
          return it->scales.size() > 1 ? xnn_datatype_qcint8
                                       : xnn_datatype_qint8;
        }
      }
      break;
    }
    case Type::kI16:
    case Type::kI32:
    case Type::kI64:
    case Type::kU4:
    case Type::kU8:
    case Type::kU16:
    case Type::kU32:
    case Type::kU64:
    case Type::kFP16:
      return xnn_datatype_fp16;
    case Type::kFP32:
      return xnn_datatype_fp32;
    case Type::kFP64:
      break;
    case Type::kBF16:
      return xnn_datatype_bf16;
  }
  return xnn_datatype_invalid;
}

// TODO: b/493560478 - Decide whether to delete this from here.
[[maybe_unused]]
size_t ByteSize(const graph::TensorInformation& info) {
  return BufferSize(info.type, info.GetSize());
}

absl::StatusOr<size_t> NumElements(const graph::TensorInformation& info) {
  size_t num_elements = 1;
  for (int32_t dim : info.shape) {
    if (dim < 0) {
      return absl::InvalidArgumentError("Negative tensor dimension.");
    }
    if (dim == 0) {
      return static_cast<size_t>(0);
    }
    if (num_elements >
        std::numeric_limits<size_t>::max() / static_cast<size_t>(dim)) {
      return absl::InvalidArgumentError("Tensor element count overflow.");
    }
    num_elements *= static_cast<size_t>(dim);
  }
  return num_elements;
}

enum class QuantParamMode {
  kScalar,
  kPerChannelDim0,
  kPerElement,
};

absl::StatusOr<QuantParamMode> DetermineQuantParamMode(
    size_t param_size, const graph::TensorInformation& info,
    size_t num_elements, absl::string_view param_name) {
  if (param_size == 1) {
    return QuantParamMode::kScalar;
  }
  if (param_size == num_elements) {
    return QuantParamMode::kPerElement;
  }
  if (!info.shape.empty() && info.shape[0] > 0 &&
      param_size == static_cast<size_t>(info.shape[0])) {
    return QuantParamMode::kPerChannelDim0;
  }
  return absl::InvalidArgumentError(
      absl::StrCat("Unsupported ", param_name, " size for tensor ", info.name));
}

size_t QuantParamIndex(QuantParamMode mode, size_t element_index,
                       size_t channel_index) {
  switch (mode) {
    case QuantParamMode::kScalar:
      return 0;
    case QuantParamMode::kPerChannelDim0:
      return channel_index;
    case QuantParamMode::kPerElement:
      return element_index;
  }
  return 0;
}

// TODO: b/493560478 - Decide whether to delete this from here.
[[maybe_unused]]
absl::StatusOr<std::vector<float>> DequantizeInt8ConstantTensor(
    const graph::TensorInformation& info,
    absl::Span<const std::byte> raw_data) {
  if (info.quantization == nullptr) {
    return absl::InvalidArgumentError(
        "INT8 constant tensor missing quantization metadata.");
  }
  LRT_TENSOR_ASSIGN_OR_RETURN(
      const auto& quantization,
      info.quantization->As<const PerChannelAffineQuantization>());
  if (quantization.quantized_dimension != 0) {
    return absl::UnimplementedError(
        absl::StrCat("Only quantized_dimension=0 is supported for XNNPACK "
                     "dequantization, got ",
                     quantization.quantized_dimension));
  }
  if (quantization.scales.empty() || quantization.zero_points.empty()) {
    return absl::InvalidArgumentError(
        "INT8 constant tensor has empty quantization params.");
  }

  LRT_TENSOR_ASSIGN_OR_RETURN(auto num_elements, NumElements(info));
  if (raw_data.size() != num_elements * sizeof(int8_t)) {
    return absl::InvalidArgumentError(absl::StrCat(
        "INT8 constant tensor byte size mismatch for ", info.name));
  }

  LRT_TENSOR_ASSIGN_OR_RETURN(
      auto scale_mode, DetermineQuantParamMode(quantization.scales.size(), info,
                                               num_elements, "scale"));
  LRT_TENSOR_ASSIGN_OR_RETURN(
      auto zero_point_mode,
      DetermineQuantParamMode(quantization.zero_points.size(), info,
                              num_elements, "zero_point"));

  size_t channel_size = 0;
  if (!info.shape.empty() && info.shape[0] > 0) {
    channel_size = num_elements / static_cast<size_t>(info.shape[0]);
  }

  std::vector<float> values(num_elements);
  const int8_t* quantized_data =
      reinterpret_cast<const int8_t*>(raw_data.data());
  for (size_t i = 0; i < num_elements; ++i) {
    const size_t channel_index = channel_size == 0 ? 0 : i / channel_size;
    const size_t scale_index = QuantParamIndex(scale_mode, i, channel_index);
    const size_t zp_index = QuantParamIndex(zero_point_mode, i, channel_index);
    values[i] = static_cast<float>(
                    static_cast<int32_t>(quantized_data[i]) -
                    static_cast<int32_t>(quantization.zero_points[zp_index])) *
                quantization.scales[scale_index];
  }
  return values;
}

}  // namespace

XnnpackGraph::XnnpackGraph(
    xnn_subgraph* subgraph, std::vector<XnnpackValue> values,
    absl::flat_hash_map<graph::Tensor, size_t> tensor_index,
    absl::flat_hash_set<graph::Tensor> external_outputs,
    std::vector<std::vector<float>> dequantized_buffers)
    : subgraph_(subgraph),
      values_(std::move(values)),
      tensor_index_(std::move(tensor_index)),
      external_outputs_(std::move(external_outputs)),
      dequantized_buffers_(std::move(dequantized_buffers)) {}

XnnpackGraph::~XnnpackGraph() {
  if (subgraph_ != nullptr) {
    xnn_delete_subgraph(subgraph_);
  }
}

absl::StatusOr<size_t> XnnpackGraph::Lookup(const graph::Tensor& tensor) const {
  auto it = tensor_index_.find(tensor);
  if (it == tensor_index_.end()) {
    return absl::NotFoundError("Tensor not part of XNNPACK graph");
  }
  return it->second;
}

XnnpackBuildContext::XnnpackBuildContext(
    std::vector<TensorHandle> outputs,
    absl::flat_hash_map<graph::Tensor, uint32_t> external_ids)
    : external_ids_(std::move(external_ids)) {
  outputs_.reserve(outputs.size());
  for (TensorHandle& handle : outputs) {
    outputs_.push_back(handle.GetRaw());
    external_outputs_.insert(handle.GetRaw());
  }
}

XnnpackBuildContext::~XnnpackBuildContext() {
  if (subgraph_ != nullptr) {
    xnn_delete_subgraph(subgraph_);
    subgraph_ = nullptr;
  }
}

absl::Status XnnpackBuildContext::Init() {
  LRT_TENSOR_RETURN_IF_ERROR(EnsureXnnInitialized());
  const size_t reserved_external_ids = external_ids_.size();
  LRT_TENSOR_RETURN_IF_ERROR(xnn_create_subgraph(
      /*external_value_ids=*/reserved_external_ids, /*flags=*/0, &subgraph_))
      << "xnn_create_subgraph";
  return absl::OkStatus();
}

absl::StatusOr<std::unique_ptr<XnnpackGraph>> XnnpackBuildContext::Finalize() {
  for (const graph::Tensor& tensor : outputs_) {
    LRT_TENSOR_RETURN_IF_ERROR(DefineValue(tensor));
  }

  auto* subgraph = subgraph_;
  subgraph_ = nullptr;

  return std::make_unique<XnnpackGraph>(
      subgraph, std::move(values_), std::move(tensor_index_),
      std::move(external_outputs_), std::move(dequantized_buffers_));
}

absl::StatusOr<std::unique_ptr<XnnpackGraph>> BuildXnnpackGraph(
    std::vector<TensorHandle> outputs) {
  LRT_TENSOR_ASSIGN_OR_RETURN(auto plan, GetExecutionPlan(outputs));

  absl::flat_hash_set<graph::Tensor> external_tensors;
  external_tensors.reserve(32);
  for (const TensorHandle& out : outputs) {
    external_tensors.insert(out.GetRaw());
  }
  for (const graph::Operation* op : plan) {
    for (const graph::Tensor& t : op->inputs) {
      auto info_or = graph::GetInfo(t);
      if (!info_or.ok()) {
        continue;
      }
      const graph::TensorInformation& info = *info_or;
      if (info.buffer != nullptr) {
        continue;
      }
      auto producer_or = graph::GetProducer(t);
      if (producer_or.ok() && *producer_or != nullptr) {
        continue;
      }
      external_tensors.insert(t);
    }
  }

  // Assign stable reserved external IDs to each external value.
  std::vector<graph::Tensor> external_sorted(external_tensors.begin(),
                                             external_tensors.end());
  // TODO: b/493560478 - This isn't a stable sort. The hash values of pointers
  // are not consistent between runs.
  std::sort(external_sorted.begin(), external_sorted.end(),
            [](const graph::Tensor& a, const graph::Tensor& b) {
              return absl::HashOf(a) < absl::HashOf(b);
            });
  absl::flat_hash_map<graph::Tensor, uint32_t> external_ids;
  external_ids.reserve(external_sorted.size());
  for (uint32_t i = 0; i < external_sorted.size(); ++i) {
    external_ids.emplace(external_sorted[i], i);
  }

  XnnpackBuildContext ctx(std::move(outputs), std::move(external_ids));
  LRT_TENSOR_RETURN_IF_ERROR(ctx.Init());

  for (const graph::Operation* op : plan) {
    auto xnn_op = dynamic_cast<const XnnpackOperation*>(op);
    if (xnn_op == nullptr) {
      return absl::InvalidArgumentError(
          absl::StrCat("Operation ", op->GetName(),
                       " does not implement XnnpackOperation."));
    }
    LRT_TENSOR_RETURN_IF_ERROR(xnn_op->ToXnnpack(ctx))
        << "Failed to convert " << op->GetName() << " to XNNPack.";
  }

  return ctx.Finalize();
}

absl::StatusOr<uint32_t> XnnpackBuildContext::DefineValue(
    const graph::Tensor& tensor) {
  if (auto it = tensor_index_.find(tensor); it != tensor_index_.end()) {
    return values_[it->second].id;
  }

  LRT_TENSOR_ASSIGN_OR_RETURN(const graph::TensorInformation& info,
                              graph::GetInfo(tensor));
  LRT_TENSOR_ASSIGN_OR_RETURN(std::shared_ptr<graph::Operation> & producer,
                              graph::GetProducer(tensor));
  const bool is_external_input = info.buffer == nullptr && producer == nullptr;
  const bool is_external_output = external_outputs_.contains(tensor);
  const bool is_external = is_external_input || is_external_output;

  XnnpackValue value{
      .info = info,
      .flags = (is_external_input ? XNN_VALUE_FLAG_EXTERNAL_INPUT : 0u) |
               (is_external_output ? XNN_VALUE_FLAG_EXTERNAL_OUTPUT : 0u),
  };

  if (info.buffer && !is_external) {
    value.data = info.buffer->Lock();
  }

  std::vector<size_t> dims(info.shape.begin(), info.shape.end());
  const void* data_ptr = value.data.data();

  uint32_t external_id = XNN_INVALID_VALUE_ID;
  if (is_external) {
    auto it = external_ids_.find(tensor);
    if (it == external_ids_.end()) {
      return absl::InternalError(
          "External tensor missing reserved external ID");
    }
    external_id = it->second;
  }

  if (!info.quantization) {
    LRT_TENSOR_RETURN_IF_ERROR(
        xnn_define_tensor_value(subgraph_, GetXnnpackType(value), dims.size(),
                                dims.empty() ? nullptr : dims.data(), data_ptr,
                                external_id, value.flags, &value.id))
        << "Could not define a new tensor value.";
  } else {
    auto maybe_pcq = info.quantization->As<PerChannelAffineQuantization>();
    if (maybe_pcq.ok()) {
      const auto& pcq = maybe_pcq.value();
      if (pcq.scales.size() == 1) {
        LRT_TENSOR_RETURN_IF_ERROR(xnn_define_quantized_tensor_value(
            subgraph_, GetXnnpackType(value), pcq.zero_points[0], pcq.scales[0],
            dims.size(), dims.empty() ? nullptr : dims.data(), data_ptr,
            external_id, value.flags, &value.id))
            << "Could not define a new quantized tensor value.";
      } else {
        bool all_zeros = true;
        for (int64_t zp : pcq.zero_points) {
          if (zp != 0) {
            all_zeros = false;
            break;
          }
        }
        if (!all_zeros) {
          LRT_TENSOR_ASSIGN_OR_RETURN(
              std::vector<float> f32_data,
              DequantizeInt8ConstantTensor(info, value.data));
          dequantized_buffers_.push_back(std::move(f32_data));
          data_ptr = dequantized_buffers_.back().data();
          LRT_TENSOR_RETURN_IF_ERROR(xnn_define_tensor_value(
              subgraph_, xnn_datatype_fp32, dims.size(),
              dims.empty() ? nullptr : dims.data(), data_ptr, external_id,
              value.flags, &value.id))
              << "Could not define a new tensor value after dequantization.";
        } else {
          LRT_TENSOR_RETURN_IF_ERROR(
              xnn_define_channelwise_quantized_tensor_value_v3(
                  subgraph_, GetXnnpackType(value), /*zero_point=*/0,
                  pcq.scales.data(), dims.size(), pcq.quantized_dimension,
                  dims.empty() ? nullptr : dims.data(), data_ptr, external_id,
                  value.flags, &value.id, /*channelwise_zero_point=*/nullptr))
              << "Could not define a new channelwise quantized tensor value.";
        }
      }
    } else {
      return absl::UnimplementedError("Unsupported quantization type.");
    }
  }

  tensor_index_.emplace(tensor, values_.size());
  values_.emplace_back(std::move(value));
  return values_.back().id;
}

::xnn_subgraph* XnnpackBuildContext::subgraph() { return subgraph_; }

}  // namespace litert::tensor
