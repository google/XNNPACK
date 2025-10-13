// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#ifndef XNNPACK_YNNPACK_SUBGRAPH_TEST_SUBGRAPH_BUILDER_H_
#define XNNPACK_YNNPACK_SUBGRAPH_TEST_SUBGRAPH_BUILDER_H_

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <memory>
#include <utility>
#include <vector>

#include "ynnpack/base/test/tensor.h"
#include "ynnpack/base/type.h"
#include "ynnpack/include/ynnpack.h"

namespace ynn {

class TestScheduler;

// This type allows describing tensors as just a rank (with no dimensions), or
// a fully specified shape.
struct TensorShape {
  // Constructs a rank 0 (scalar) shape.
  TensorShape() = default;

  // Constructs a shape with a specified rank but unknown dimensions.
  TensorShape(size_t rank) : rank_(rank) {}  // NOLINT

  // Constructs a shape with specific dimensions.
  TensorShape(std::vector<size_t> dims) : dims_(std::move(dims)) {}  // NOLINT
  TensorShape(const std::initializer_list<size_t>& dims)
      : dims_(dims) {}  // NOLINT

  size_t Rank() const { return std::max(rank_, dims_.size()); }
  const size_t* Dims() const { return dims_.empty() ? nullptr : dims_.data(); }

 private:
  // If this is non-zero, the shape is dynamic with this rank.
  size_t rank_ = 0;

  // If this is non-empty, the shape is static with these extents.
  std::vector<size_t> dims_;
};

class SubgraphBuilder {
 public:
  explicit SubgraphBuilder(size_t external_value_count, uint32_t flags = 0);

  SubgraphBuilder& AddTensor(ynn_type type, TensorShape shape, uint32_t& id,
                             const void* data = nullptr,
                             uint32_t zero_point_id = YNN_INVALID_VALUE_ID,
                             uint32_t scale_id = YNN_INVALID_VALUE_ID,
                             uint32_t flags = 0);
  SubgraphBuilder& AddTensor(ynn_type type, TensorShape shape, uint32_t& id,
                             const void* data,
                             quantization_params scalar_quantization,
                             uint32_t flags = 0);

  SubgraphBuilder& AddInput(ynn_type type, TensorShape shape, uint32_t id,
                            uint32_t zero_point_id = YNN_INVALID_VALUE_ID,
                            uint32_t scale_id = YNN_INVALID_VALUE_ID,
                            uint32_t flags = 0) {
    return AddTensor(type, shape, id, /*data=*/nullptr, zero_point_id, scale_id,
                     flags | YNN_VALUE_FLAG_EXTERNAL_INPUT);
  }
  SubgraphBuilder& AddInput(ynn_type type, TensorShape shape, uint32_t id,
                            quantization_params scalar_quantization,
                            uint32_t flags = 0) {
    return AddTensor(type, shape, id, /*data=*/nullptr, scalar_quantization,
                     flags | YNN_VALUE_FLAG_EXTERNAL_INPUT);
  }
  SubgraphBuilder& AddOutput(ynn_type type, TensorShape shape, uint32_t id,
                             uint32_t zero_point_id = YNN_INVALID_VALUE_ID,
                             uint32_t scale_id = YNN_INVALID_VALUE_ID,
                             uint32_t flags = 0) {
    return AddTensor(type, shape, id, /*data=*/nullptr, zero_point_id, scale_id,
                     flags | YNN_VALUE_FLAG_EXTERNAL_OUTPUT);
  }
  SubgraphBuilder& AddOutput(ynn_type type, TensorShape shape, uint32_t id,
                             quantization_params scalar_quantization,
                             uint32_t flags = 0) {
    return AddTensor(type, shape, id, /*data=*/nullptr, scalar_quantization,
                     flags | YNN_VALUE_FLAG_EXTERNAL_OUTPUT);
  }

  template <typename T>
  SubgraphBuilder& AddTensor(const Tensor<T>& tensor, uint32_t id,
                             quantization_params quantization = {},
                             uint32_t flags = 0) {
    std::vector<size_t> extents(tensor.extents());
    if (!extents.empty()) {
      // Convert physical shape of tensor to logical shape for YNNPACK API.
      extents.back() *= type_element_count(type_of<T>());
    }
    return AddTensor(type_of<T>(), extents, id, tensor.data(), quantization,
                     flags);
  }

  template <typename T>
  SubgraphBuilder& AddScalar(T value, uint32_t id,
                             quantization_params quantization = {},
                             uint32_t flags = 0) {
    return AddTensor(type_of<T>(), {}, id, &value, quantization,
                     flags | YNN_VALUE_FLAG_COPY_DATA);
  }

  SubgraphBuilder& AddUnary(ynn_unary_operator op, uint32_t input_id,
                            uint32_t output_id, uint32_t flags = 0);
  SubgraphBuilder& AddBinary(ynn_binary_operator op, uint32_t input_a_id,
                             uint32_t input_b_id, uint32_t output_id,
                             uint32_t flags = 0);

  SubgraphBuilder& AddBroadcast(const std::vector<size_t>& shape,
                                uint32_t input_id, uint32_t output_id,
                                uint32_t flags = 0);
  SubgraphBuilder& AddExpandDims(const std::vector<int32_t>& axes,
                                 uint32_t input_id, uint32_t output_id,
                                 uint32_t flags = 0);
  SubgraphBuilder& AddReshape(const std::vector<size_t>& shape,
                              uint32_t input_id, uint32_t output_id,
                              uint32_t flags = 0);
  SubgraphBuilder& AddCopy(uint32_t input_id, uint32_t output_id,
                           uint32_t flags = 0);

  SubgraphBuilder& AddFuseDim(int32_t first_dim, size_t num_dims,
                              uint32_t input_id, uint32_t output_id);

  SubgraphBuilder& AddFuseDims(const std::vector<int32_t>& axes,
                               uint32_t input_id, uint32_t output_id);

  SubgraphBuilder& AddSplitDim(int32_t axis, const std::vector<size_t>& splits,
                               uint32_t input_id, uint32_t output_id);

  SubgraphBuilder& AddConcatenate(int32_t axis,
                                  const std::vector<uint32_t>& input_ids,
                                  uint32_t output_id);

  SubgraphBuilder& AddStack(int32_t axis,
                            const std::vector<uint32_t>& input_ids,
                            uint32_t output_id);

  SubgraphBuilder& AddEvenSplit(int32_t axis, uint32_t input_id,
                                const std::vector<uint32_t>& output_ids);

  SubgraphBuilder& AddTranspose(const std::vector<int32_t>& perm,
                                uint32_t input_id, uint32_t output_id);

  SubgraphBuilder& AddSlice(const std::vector<int32_t>& axes,
                            const std::vector<int64_t>& begins,
                            const std::vector<int64_t>& ends,
                            const std::vector<int64_t>& strides,
                            uint32_t input_id, uint32_t output_id,
                            uint32_t flags = 0);

  SubgraphBuilder& AddBroadcastLike(const std::vector<int32_t>& axes,
                                    uint32_t input_id, uint32_t template_id,
                                    uint32_t output_id);

  SubgraphBuilder& AddBroadcast(const std::vector<int32_t>& axes,
                                uint32_t input_id, uint32_t output_id);

  SubgraphBuilder& AddStencilCopy(const std::vector<int32_t>& stencil_axes,
                                  const std::vector<int32_t>& new_axes,
                                  const std::vector<size_t>& stencil_dims,
                                  const std::vector<size_t>& stencil_strides,
                                  const std::vector<size_t>& stencil_dilations,
                                  uint32_t input_id, uint32_t padding_id,
                                  uint32_t output_id);

  SubgraphBuilder& AddPad(const std::vector<int32_t>& axes,
                          const std::vector<int64_t>& pre_paddings,
                          const std::vector<int64_t>& post_paddings,
                          uint32_t input_id, uint32_t padding_id,
                          uint32_t output_id);

  SubgraphBuilder& AddDot(size_t num_k_dims, uint32_t input_a_id,
                          uint32_t input_b_id, uint32_t input_c_id,
                          uint32_t output_id);

  SubgraphBuilder& AddReduce(ynn_reduce_operator op,
                             const std::vector<int32_t>& reduce_axes,
                             uint32_t input_a_id, uint32_t input_b_id,
                             uint32_t output_id, uint32_t flags = 0);

  SubgraphBuilder& AddGetTensorShape(const std::vector<int32_t>& axes,
                                     ynn_type type, size_t rank,
                                     uint32_t input_id, uint32_t output_id,
                                     uint32_t flags = 0);

  template <typename T>
  uint32_t DefineScalar(T value) {
    uint32_t id = YNN_INVALID_VALUE_ID;
    status_ = ynn_define_tensor_value(subgraph_.get(), type_of<T>(), 0,
                                      /*dims=*/nullptr, &value,
                                      /*zero_point_id=*/YNN_INVALID_VALUE_ID,
                                      /*scale_id=*/YNN_INVALID_VALUE_ID,
                                      /*flags=*/YNN_VALUE_FLAG_COPY_DATA, &id);
    return id;
  }

  // Define a quantized scalar with a dequantized value of `value`.
  template <typename T>
  uint32_t DefineScalar(float value, quantization_params quantization) {
    uint32_t id = YNN_INVALID_VALUE_ID;
    T quantized_value = quantize<T>(value, quantization);
    status_ = ynn_define_tensor_value(subgraph_.get(), type_of<T>(), 0,
                                      /*dims=*/nullptr, &quantized_value,
                                      DefineScalar<T>(quantization.zero_point),
                                      DefineScalar<T>(quantization.scale),
                                      /*flags=*/YNN_VALUE_FLAG_COPY_DATA, &id);
    return id;
  }

  ynn_subgraph_t GetSubgraph() const { return subgraph_.get(); }

 private:
  ynn_status status_;
  std::unique_ptr<ynn_subgraph, decltype(&ynn_delete_subgraph)> subgraph_;
};

class Runtime {
 public:
  Runtime(ynn_subgraph_t subgraph, TestScheduler* scheduler = nullptr,
          uint32_t flags = 0);

  Runtime& ReshapeExternalTensor(const TensorShape& shape, void* data,
                                 uint32_t id);
  Runtime& SetupExternalTensor(void* data, uint32_t id);

  Runtime& ReshapeRuntime();
  Runtime& InvokeRuntime();

  std::vector<size_t> GetExternalTensorShape(uint32_t id);

  ynn_status Status() const { return status_; }

 private:
  ynn_status status_;
  std::unique_ptr<ynn_runtime, decltype(&ynn_delete_runtime)> runtime_;
  std::unique_ptr<ynn_threadpool, decltype(&ynn_delete_threadpool)> threadpool_;
};

}  // namespace ynn

#endif  // XNNPACK_YNNPACK_SUBGRAPH_TEST_SUBGRAPH_BUILDER_H_
