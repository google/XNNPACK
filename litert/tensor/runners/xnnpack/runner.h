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

#ifndef LITERT_TENSOR_RUNNERS_XNNPACK_RUNNER_H_
#define LITERT_TENSOR_RUNNERS_XNNPACK_RUNNER_H_

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "include/xnnpack.h"
#include "absl/container/flat_hash_map.h"
#include "absl/status/status.h"
#include "absl/status/statusor.h"
#include "absl/types/span.h"
#include "litert/tensor/backends/xnnpack/conversion.h"
#include "litert/tensor/buffer.h"
#include "litert/tensor/internal/graph.h"
#include "litert/tensor/tensor.h"
#include <pthreadpool.h>

struct xnn_runtime;

namespace litert::tensor {
class XnnpackRunnerTest_ConstantsAreNotBoundAsExternals_Test;

// XnnpackRunner is a class that runs an XNNPACK graph.
class XnnpackRunner {
  friend class XnnpackRunnerTest_ConstantsAreNotBoundAsExternals_Test;

 public:
  // Creates an XnnpackRunner from a list of output tensors.
  static absl::StatusOr<XnnpackRunner> Create(
      std::vector<TensorHandle> outputs);

  // Sets the number of threads.
  //
  // Warning: this must be called before the first call to `Run`.
  void SetNumThreads(size_t num_threads) { num_threads_ = num_threads; }

  // Sets the input data for a given tensor.
  absl::Status SetInput(const graph::Tensor& tensor,
                        absl::Span<const std::byte> data);
  absl::Status SetInput(const TensorHandle& tensor,
                        absl::Span<const std::byte> data);
  // Updates the shape for an external input tensor.
  absl::Status ReshapeInput(const graph::Tensor& tensor,
                            absl::Span<const int32_t> shape);
  absl::Status ReshapeInput(const TensorHandle& tensor,
                            absl::Span<const int32_t> shape);
  // Writes a slice of bytes into an external input tensor's host buffer.
  absl::Status WriteInput(const graph::Tensor& tensor, size_t offset_bytes,
                          absl::Span<const std::byte> data);
  absl::Status WriteInput(const TensorHandle& tensor, size_t offset_bytes,
                          absl::Span<const std::byte> data);
  // Runs the XNNPACK graph.
  absl::Status Run();
  // Reads the output data for a given tensor.
  absl::StatusOr<LockedBufferSpan<const std::byte>> ReadOutput(
      const graph::Tensor& tensor) const;
  absl::StatusOr<LockedBufferSpan<const std::byte>> ReadOutput(
      const TensorHandle& tensor) const;

 private:
  explicit XnnpackRunner(std::unique_ptr<XnnpackGraph> graph);

#define TENSOR_API_UNIQUE_PTR_WITH_DELETER(NAME, TYPE, DEL_FUNC) \
  struct NAME##Deleter {                                         \
    void operator()(TYPE* data) const {                          \
      if (data) {                                                \
        DEL_FUNC(data);                                          \
      }                                                          \
    }                                                            \
  };                                                             \
  using NAME##Ptr = std::unique_ptr<TYPE, NAME##Deleter>

  TENSOR_API_UNIQUE_PTR_WITH_DELETER(Runtime, xnn_runtime, xnn_delete_runtime);
  TENSOR_API_UNIQUE_PTR_WITH_DELETER(Threadpool, pthreadpool,
                                     pthreadpool_destroy);
#undef TENSOR_API_UNIQUE_PTR_WITH_DELETER

  RuntimePtr runtime_ = nullptr;
  std::unique_ptr<XnnpackGraph> graph_;
  absl::flat_hash_map<uint32_t, std::vector<std::byte>> external_buffers_;
  ThreadpoolPtr threadpool_ = nullptr;
  size_t num_threads_ = 1;
};

}  // namespace litert::tensor

#endif  // LITERT_TENSOR_RUNNERS_XNNPACK_RUNNER_H_
