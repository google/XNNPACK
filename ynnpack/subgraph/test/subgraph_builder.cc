// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "ynnpack/subgraph/test/subgraph_builder.h"

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include "ynnpack/base/type.h"
#include "ynnpack/include/ynnpack.h"
#include "ynnpack/subgraph/test/scheduler.h"

namespace ynn {

SubgraphBuilder::SubgraphBuilder(size_t external_value_count, uint32_t flags)
    : subgraph_(nullptr, ynn_delete_subgraph) {
  ynn_subgraph_t subgraph;
  status_ = ynn_create_subgraph(external_value_count, flags, &subgraph);
  subgraph_ = std::unique_ptr<ynn_subgraph, decltype(&ynn_delete_subgraph)>(
      subgraph, ynn_delete_subgraph);
}

SubgraphBuilder& SubgraphBuilder::AddTensor(ynn_type type, TensorShape shape,
                                            uint32_t& id, const void* data,
                                            uint32_t zero_point_id,
                                            uint32_t scale_id, uint32_t flags) {
  assert(status_ == ynn_status_success);
  status_ = ynn_define_tensor_value(
      subgraph_.get(), type, shape.Rank(), shape.Dims(),
      /*data=*/data, zero_point_id, scale_id, flags, &id);
  return *this;
}

SubgraphBuilder& SubgraphBuilder::AddTensor(
    ynn_type type, TensorShape shape, uint32_t& id, const void* data,
    quantization_params scalar_quantization, uint32_t flags) {
  assert(status_ == ynn_status_success);
  uint32_t scale_id = YNN_INVALID_VALUE_ID;
  uint32_t zero_point_id = YNN_INVALID_VALUE_ID;
  if (scalar_quantization.scale != 1.0f) {
    scale_id = DefineScalar<float>(scalar_quantization.scale);
  }
  if (scalar_quantization.zero_point != 0) {
    zero_point_id = DefineScalar<int32_t>(scalar_quantization.zero_point);
  }
  status_ =
      ynn_define_tensor_value(subgraph_.get(), type, shape.Rank(), shape.Dims(),
                              data, zero_point_id, scale_id, flags, &id);
  return *this;
}

SubgraphBuilder& SubgraphBuilder::AddUnary(ynn_unary_operator op,
                                           uint32_t input_id,
                                           uint32_t output_id, uint32_t flags) {
  assert(status_ == ynn_status_success);
  status_ = ynn_define_unary(subgraph_.get(), op, input_id, &output_id, flags);
  return *this;
}
SubgraphBuilder& SubgraphBuilder::AddBinary(ynn_binary_operator op,
                                            uint32_t input_a_id,
                                            uint32_t input_b_id,
                                            uint32_t output_id,
                                            uint32_t flags) {
  assert(status_ == ynn_status_success);
  status_ = ynn_define_binary(subgraph_.get(), op, input_a_id, input_b_id,
                              &output_id, flags);
  return *this;
}

SubgraphBuilder& SubgraphBuilder::AddBroadcast(const std::vector<size_t>& shape,
                                               uint32_t input_id,
                                               uint32_t output_id,
                                               uint32_t flags) {
  assert(status_ == ynn_status_success);
  status_ = ynn_define_static_broadcast(
      subgraph_.get(), shape.size(), shape.data(), input_id, &output_id, flags);
  return *this;
}

SubgraphBuilder& SubgraphBuilder::AddExpandDims(
    const std::vector<int32_t>& new_axes, uint32_t input_id, uint32_t output_id,
    uint32_t flags) {
  assert(status_ == ynn_status_success);
  status_ = ynn_define_static_expand_dims(subgraph_.get(), new_axes.size(),
                                          new_axes.data(), input_id, &output_id,
                                          flags);
  return *this;
}

SubgraphBuilder& SubgraphBuilder::AddReshape(const std::vector<size_t>& shape,
                                             uint32_t input_id,
                                             uint32_t output_id,
                                             uint32_t flags) {
  assert(status_ == ynn_status_success);
  status_ = ynn_define_static_reshape(
      subgraph_.get(), shape.size(), shape.data(), input_id, &output_id, flags);
  return *this;
}

SubgraphBuilder& SubgraphBuilder::AddCopy(uint32_t input_id, uint32_t output_id,
                                          uint32_t flags) {
  assert(status_ == ynn_status_success);
  status_ = ynn_define_copy(subgraph_.get(), input_id, &output_id, flags);
  return *this;
}

SubgraphBuilder& SubgraphBuilder::AddFuseDim(int32_t first_dim, size_t num_dims,
                                             uint32_t input_id,
                                             uint32_t output_id) {
  assert(status_ == ynn_status_success);
  status_ = ynn_define_fuse_dim(subgraph_.get(), first_dim, num_dims, input_id,
                                &output_id,
                                /*flags=*/0);
  return *this;
}

SubgraphBuilder& SubgraphBuilder::AddFuseDims(const std::vector<int32_t>& axes,
                                              uint32_t input_id,
                                              uint32_t output_id) {
  assert(status_ == ynn_status_success);
  status_ = ynn_define_fuse_dims(subgraph_.get(), axes.size(), axes.data(),
                                 input_id, &output_id,
                                 /*flags=*/0);
  return *this;
}

SubgraphBuilder& SubgraphBuilder::AddSplitDim(int32_t axis,
                                              const std::vector<size_t>& splits,
                                              uint32_t input_id,
                                              uint32_t output_id) {
  assert(status_ == ynn_status_success);
  status_ = ynn_define_split_dim(subgraph_.get(), axis, splits.size(),
                                 splits.data(), input_id, &output_id,
                                 /*flags=*/0);
  return *this;
}

SubgraphBuilder& SubgraphBuilder::AddConcatenate(
    int32_t axis, const std::vector<uint32_t>& input_ids, uint32_t output_id) {
  assert(status_ == ynn_status_success);
  status_ = ynn_define_concatenate(subgraph_.get(), axis, input_ids.size(),
                                   input_ids.data(), &output_id,
                                   /*flags=*/0);
  return *this;
}

SubgraphBuilder& SubgraphBuilder::AddStack(
    int32_t axis, const std::vector<uint32_t>& input_ids, uint32_t output_id) {
  assert(status_ == ynn_status_success);
  status_ = ynn_define_stack(subgraph_.get(), axis, input_ids.size(),
                             input_ids.data(), &output_id,
                             /*flags=*/0);
  return *this;
}

SubgraphBuilder& SubgraphBuilder::AddEvenSplit(
    int32_t axis, uint32_t input_id, const std::vector<uint32_t>& output_ids) {
  assert(status_ == ynn_status_success);
  status_ =
      ynn_define_even_split(subgraph_.get(), axis, input_id, output_ids.size(),
                            std::vector<uint32_t>(output_ids).data(),
                            /*flags=*/0);
  return *this;
}

SubgraphBuilder& SubgraphBuilder::AddTranspose(const std::vector<int32_t>& perm,
                                               uint32_t input_id,
                                               uint32_t output_id) {
  assert(status_ == ynn_status_success);
  status_ = ynn_define_static_transpose(subgraph_.get(), perm.size(),
                                        perm.data(), input_id, &output_id,
                                        /*flags=*/0);
  return *this;
}

SubgraphBuilder& SubgraphBuilder::AddSlice(const std::vector<int32_t>& axes,
                                           const std::vector<int64_t>& begins,
                                           const std::vector<int64_t>& ends,
                                           const std::vector<int64_t>& strides,
                                           uint32_t input_id,
                                           uint32_t output_id, uint32_t flags) {
  assert(status_ == ynn_status_success);
  status_ = ynn_define_static_slice(subgraph_.get(), axes.size(), axes.data(),
                                    begins.data(), ends.data(), strides.data(),
                                    input_id, &output_id, flags);
  return *this;
}

SubgraphBuilder& SubgraphBuilder::AddBroadcast(const std::vector<int32_t>& axes,
                                               uint32_t input_id,
                                               uint32_t output_id) {
  assert(status_ == ynn_status_success);
  status_ = ynn_define_broadcast(subgraph_.get(), axes.size(), axes.data(),
                                 input_id, &output_id, /*flags=*/0);
  return *this;
}

SubgraphBuilder& SubgraphBuilder::AddBroadcastLike(
    const std::vector<int32_t>& axes, uint32_t input_id, uint32_t template_id,
    uint32_t output_id) {
  assert(status_ == ynn_status_success);
  status_ =
      ynn_define_broadcast_like(subgraph_.get(), axes.size(), axes.data(),
                                input_id, template_id, &output_id, /*flags=*/0);
  return *this;
}

SubgraphBuilder& SubgraphBuilder::AddStencilCopy(
    const std::vector<int32_t>& stencil_axes,
    const std::vector<int32_t>& new_axes,
    const std::vector<size_t>& stencil_dims,
    const std::vector<size_t>& stencil_strides,
    const std::vector<size_t>& stencil_dilations, uint32_t input_id,
    uint32_t padding_id, uint32_t output_id) {
  assert(status_ == ynn_status_success);
  assert(stencil_axes.size() == stencil_dims.size());
  assert(stencil_axes.size() == new_axes.size());
  assert(stencil_axes.size() == stencil_strides.size());
  assert(stencil_axes.size() == stencil_dilations.size());
  status_ = ynn_define_stencil_copy(
      subgraph_.get(), stencil_axes.size(), stencil_axes.data(),
      new_axes.data(), stencil_dims.data(), stencil_strides.data(),
      stencil_dilations.data(), input_id, padding_id, &output_id, /*flags=*/0);
  return *this;
}

SubgraphBuilder& SubgraphBuilder::AddPad(
    const std::vector<int32_t>& axes, const std::vector<int64_t>& pre_paddings,
    const std::vector<int64_t>& post_paddings, uint32_t input_id,
    uint32_t padding_id, uint32_t output_id) {
  assert(pre_paddings.size() == post_paddings.size());
  assert(status_ == ynn_status_success);
  status_ = ynn_define_static_pad(subgraph_.get(), axes.size(), axes.data(),
                                  pre_paddings.data(), post_paddings.data(),
                                  input_id, padding_id, &output_id,
                                  /*flags=*/0);
  return *this;
}

SubgraphBuilder& SubgraphBuilder::AddDot(size_t num_k_dims, uint32_t input_a_id,
                                         uint32_t input_b_id,
                                         uint32_t input_c_id,
                                         uint32_t output_id) {
  assert(status_ == ynn_status_success);
  status_ = ynn_define_dot(subgraph_.get(), num_k_dims, input_a_id, input_b_id,
                           input_c_id, &output_id, /*flags=*/0);
  return *this;
}

SubgraphBuilder& SubgraphBuilder::AddReduce(
    ynn_reduce_operator op, const std::vector<int32_t>& reduce_axes,
    uint32_t input_a_id, uint32_t input_b_id, uint32_t output_id,
    uint32_t flags) {
  assert(status_ == ynn_status_success);
  status_ = ynn_define_reduce(subgraph_.get(), op, reduce_axes.size(),
                              reduce_axes.data(), input_a_id, input_b_id,
                              &output_id, flags);
  return *this;
}

SubgraphBuilder& SubgraphBuilder::AddGetTensorShape(
    const std::vector<int32_t>& axes, ynn_type type, size_t rank,
    uint32_t input_id, uint32_t output_id, uint32_t flags) {
  assert(status_ == ynn_status_success);
  status_ =
      ynn_define_get_tensor_shape(subgraph_.get(), axes.size(), axes.data(),
                                  type, rank, input_id, &output_id, flags);
  return *this;
}

Runtime::Runtime(ynn_subgraph_t subgraph, TestScheduler* scheduler,
                 uint32_t flags)
    : runtime_(nullptr, ynn_delete_runtime),
      threadpool_(nullptr, ynn_delete_threadpool) {
  ynn_runtime_t runtime;

  if (scheduler) {
    ynn_threadpool_t threadpool;
    status_ = ynn_create_threadpool(scheduler->scheduler(), scheduler,
                                    /*flags=*/0, &threadpool);
    if (status_ != ynn_status_success) {
      return;
    }
    threadpool_ =
        std::unique_ptr<ynn_threadpool, decltype(&ynn_delete_threadpool)>(
            threadpool, ynn_delete_threadpool);
  }

  status_ = ynn_optimize_subgraph(subgraph, threadpool_.get(), 0);
  if (status_ != ynn_status_success) {
    return;
  }

  status_ = ynn_create_runtime(subgraph, threadpool_.get(), flags, &runtime);
  runtime_ = std::unique_ptr<ynn_runtime, decltype(&ynn_delete_runtime)>(
      runtime, ynn_delete_runtime);
}

Runtime& Runtime::ReshapeExternalTensor(const TensorShape& shape, void* data,
                                        uint32_t id) {
  assert(status_ == ynn_status_success);
  status_ = ynn_set_external_value_shape(runtime_.get(), id, shape.Rank(),
                                         shape.Dims());
  status_ = ynn_set_external_value_data(runtime_.get(), id, data);
  return *this;
}
Runtime& Runtime::SetupExternalTensor(void* data, uint32_t id) {
  assert(status_ == ynn_status_success);
  status_ = ynn_set_external_value_data(runtime_.get(), id, data);
  return *this;
}

Runtime& Runtime::ReshapeRuntime() {
  assert(status_ == ynn_status_success);
  status_ = ynn_reshape_runtime(runtime_.get());
  return *this;
}

Runtime& Runtime::InvokeRuntime() {
  assert(status_ == ynn_status_success);
  status_ = ynn_invoke_runtime(runtime_.get());
  return *this;
}

std::vector<size_t> Runtime::GetExternalTensorShape(uint32_t id) {
  assert(status_ == ynn_status_success);
  std::vector<size_t> shape(YNN_MAX_TENSOR_RANK);
  size_t rank = YNN_MAX_TENSOR_RANK;
  status_ =
      ynn_get_external_value_shape(runtime_.get(), id, &rank, shape.data());
  shape.resize(rank);
  return shape;
}

}  // namespace ynn
