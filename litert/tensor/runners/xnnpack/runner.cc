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

#include "litert/tensor/runners/xnnpack/runner.h"

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <utility>
#include <vector>

#include "include/xnnpack.h"
#include "absl/algorithm/container.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/strings/string_view.h"
#include "absl/types/span.h"
#include "litert/tensor/backends/xnnpack/arithmetic.h"
#include "litert/tensor/backends/xnnpack/conversion.h"
#include "litert/tensor/backends/xnnpack/utils.h"
#include "litert/tensor/buffer.h"
#include "litert/tensor/datatypes.h"
#include "litert/tensor/internal/graph.h"
#include "litert/tensor/tensor.h"
#include "litert/tensor/utils/macros.h"
#include <pthreadpool.h>

namespace litert::tensor {
namespace {

size_t ByteSize(const graph::TensorInformation& info) {
  return BufferSize(info.type, info.GetSize());
}

absl::StatusOr<std::vector<size_t>> ToXnnDims(
    const std::vector<int32_t>& shape) {
  std::vector<size_t> dims;
  dims.reserve(shape.size());
  for (const int32_t d : shape) {
    if (d < 0) {
      return absl::InvalidArgumentError("Negative dimension is not supported");
    }
    dims.push_back(static_cast<size_t>(d));
  }
  return dims;
}

}  // namespace

// Creates an XnnpackRunner from a list of output tensors.
absl::StatusOr<XnnpackRunner> XnnpackRunner::Create(
    std::vector<TensorHandle> outputs) {
  // Build the XNNPACK graph from the output tensors.
  LRT_TENSOR_ASSIGN_OR_RETURN(std::unique_ptr<XnnpackGraph> graph,
                              BuildXnnpackGraph(std::move(outputs)));
  return XnnpackRunner(std::move(graph));
}

XnnpackRunner::XnnpackRunner(std::unique_ptr<XnnpackGraph> graph)
    : graph_(std::move(graph)) {}

// Sets the input data for a given tensor.
absl::Status XnnpackRunner::SetInput(const graph::Tensor& tensor,
                                     absl::Span<const std::byte> data) {
  // Find the index of the tensor in the graph.
  LRT_TENSOR_ASSIGN_OR_RETURN(size_t index, graph_->Lookup(tensor));
  XnnpackValue& value = graph_->mutable_values()[index];
  // Check if the tensor is marked as an external input.
  if ((value.flags & XNN_VALUE_FLAG_EXTERNAL_INPUT) == 0) {
    return absl::InvalidArgumentError("Tensor is not marked as external input");
  }
  if (ByteSize(value.info) != data.size()) {
    return absl::InvalidArgumentError("Mismatched input size");
  }
  external_buffers_[value.id].assign(data.begin(), data.end());
  return absl::OkStatus();
}

absl::Status XnnpackRunner::SetInput(const TensorHandle& tensor,
                                     absl::Span<const std::byte> data) {
  return SetInput(tensor.GetRaw(), data);
}

absl::Status XnnpackRunner::ReshapeInput(const graph::Tensor& tensor,
                                         absl::Span<const int32_t> shape) {
  LRT_TENSOR_ASSIGN_OR_RETURN(size_t index, graph_->Lookup(tensor));
  XnnpackValue& value = graph_->mutable_values()[index];
  if ((value.flags & XNN_VALUE_FLAG_EXTERNAL_INPUT) == 0) {
    return absl::InvalidArgumentError("Tensor is not marked as external input");
  }
  value.info.shape.assign(shape.begin(), shape.end());
  external_buffers_[value.id].resize(ByteSize(value.info));
  return absl::OkStatus();
}

absl::Status XnnpackRunner::ReshapeInput(const TensorHandle& tensor,
                                         absl::Span<const int32_t> shape) {
  return ReshapeInput(tensor.GetRaw(), shape);
}

absl::Status XnnpackRunner::WriteInput(const graph::Tensor& tensor,
                                       size_t offset_bytes,
                                       absl::Span<const std::byte> data) {
  LRT_TENSOR_ASSIGN_OR_RETURN(size_t index, graph_->Lookup(tensor));
  XnnpackValue& value = graph_->mutable_values()[index];
  if ((value.flags & XNN_VALUE_FLAG_EXTERNAL_INPUT) == 0) {
    return absl::InvalidArgumentError("Tensor is not marked as external input");
  }
  std::vector<std::byte>& buffer = external_buffers_[value.id];
  if (offset_bytes > buffer.size() ||
      data.size() > buffer.size() - offset_bytes) {
    return absl::InvalidArgumentError("WriteInput out of bounds");
  }
  absl::c_copy(data, buffer.begin() + offset_bytes);
  return absl::OkStatus();
}

absl::Status XnnpackRunner::WriteInput(const TensorHandle& tensor,
                                       size_t offset_bytes,
                                       absl::Span<const std::byte> data) {
  return WriteInput(tensor.GetRaw(), offset_bytes, data);
}

// Runs the XNNPACK graph.
absl::Status XnnpackRunner::Run() {
  // Create the XNNPACK runtime if it doesn't exist.
  if (runtime_ == nullptr) {
    if (num_threads_ > 1) {
      threadpool_.reset(pthreadpool_create(num_threads_));
    }
    xnn_runtime* raw_runtime = nullptr;
    LRT_TENSOR_RETURN_IF_ERROR(
        xnn_create_runtime_v3(graph_->subgraph(), /*weights_cache=*/nullptr,
                              /*threadpool=*/threadpool_.get(),
                              /*flags=*/0, &raw_runtime));
    runtime_.reset(raw_runtime);
  }

  // Reshape external inputs to match the current host-side tensor shapes.
  for (XnnpackValue& value : graph_->mutable_values()) {
    if ((value.flags & XNN_VALUE_FLAG_EXTERNAL_INPUT) == 0) {
      continue;
    }
    LRT_TENSOR_ASSIGN_OR_RETURN(const std::vector<size_t> dims,
                                ToXnnDims(value.info.shape));
    LRT_TENSOR_RETURN_IF_ERROR(
        xnn_reshape_external_value(runtime_.get(), value.id, dims.size(),
                                   dims.empty() ? nullptr : dims.data()))
        << "xnn_reshape_external_value";

    external_buffers_[value.id].resize(ByteSize(value.info));
  }

  // Reshape the runtime (propagates input shapes through the graph).
  LRT_TENSOR_RETURN_IF_ERROR(xnn_reshape_runtime(runtime_.get()));

  // Resize external output buffers to match the runtime-determined shapes.
  for (XnnpackValue& value : graph_->mutable_values()) {
    if ((value.flags & XNN_VALUE_FLAG_EXTERNAL_OUTPUT) == 0) {
      continue;
    }

    size_t num_dims = 0;
    std::array<size_t, XNN_MAX_TENSOR_DIMS> dims{};
    LRT_TENSOR_RETURN_IF_ERROR(xnn_get_external_value_shape(
        runtime_.get(), value.id, &num_dims, dims.data()))
        << "xnn_get_external_value_shape";

    value.info.shape.clear();
    value.info.shape.reserve(num_dims);
    for (size_t i = 0; i < num_dims; ++i) {
      value.info.shape.push_back(static_cast<int32_t>(dims[i]));
    }

    external_buffers_[value.id].resize(ByteSize(value.info));
  }

  // Prepare external values for the runtime.
  std::vector<xnn_external_value> externals;
  externals.reserve(graph_->values().size());
  for (XnnpackValue& value : graph_->mutable_values()) {
    if (value.flags == 0) {
      continue;
    }
    std::vector<std::byte>& buffer = external_buffers_[value.id];
    if (buffer.empty()) {
      return absl::FailedPreconditionError(
          "External value missing host buffer");
    }
    externals.push_back(
        {.id = value.id, .data = static_cast<void*>(buffer.data())});
  }

  // Setup the runtime with the external values.
  LRT_TENSOR_RETURN_IF_ERROR(
      xnn_setup_runtime_v2(runtime_.get(), externals.size(), externals.data()));
  // Invoke the runtime.
  return XnnStatusToAbsl(xnn_invoke_runtime(runtime_.get()),
                         "xnn_invoke_runtime");
}

// Reads the output data for a given tensor.
absl::StatusOr<LockedBufferSpan<const std::byte>> XnnpackRunner::ReadOutput(
    const graph::Tensor& tensor) const {
  // Find the index of the tensor in the graph.
  LRT_TENSOR_ASSIGN_OR_RETURN(size_t index, graph_->Lookup(tensor));
  const XnnpackValue& value = graph_->values()[index];
  // Check if the tensor is marked as an external output.
  if ((value.flags & XNN_VALUE_FLAG_EXTERNAL_OUTPUT) == 0) {
    return absl::InvalidArgumentError("Tensor is not marked as output");
  }
  const auto buffer_it = external_buffers_.find(value.id);
  if (buffer_it == external_buffers_.end()) {
    return absl::FailedPreconditionError("Tensor is not an output buffer");
  }
  if (buffer_it->second.empty()) {
    return absl::FailedPreconditionError("No buffer available for output");
  }
  return LockedBufferSpan<const std::byte>(
      buffer_it->second.data(), [](const std::byte*) {},
      buffer_it->second.size());
}

absl::StatusOr<LockedBufferSpan<const std::byte>> XnnpackRunner::ReadOutput(
    const TensorHandle& tensor) const {
  return ReadOutput(tensor.GetRaw());
}

}  // namespace litert::tensor
