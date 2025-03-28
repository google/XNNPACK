// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <memory>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "include/xnnpack.h"
#include "src/xnnpack/subgraph.h"
#include "test/subgraph/runtime-flags.h"
#include "test/subgraph/subgraph-tester.h"

namespace xnnpack {

class RuntimeTester : public SubgraphTester {
 public:
  using SubgraphTester::SubgraphTester;

  template <typename T>
  xnnpack::Buffer<T> RunWithFusion() {
    Run();
    xnnpack::Buffer<char>& tensor = this->buffers_.at(this->output_id_);
    xnnpack::Buffer<float> output =
        xnnpack::Buffer<float>(tensor.size() / sizeof(float));
    std::memcpy(output.data(), tensor.data(), tensor.size());
    return output;
  }

  template <typename T>
  xnnpack::Buffer<T> RunWithoutFusion() {
    Run(XNN_FLAG_NO_OPERATOR_FUSION | xnn_test_runtime_flags());
    xnnpack::Buffer<char>& tensor = this->buffers_.at(this->output_id_);
    xnnpack::Buffer<float> output =
        xnnpack::Buffer<float>(tensor.size() / sizeof(float));
    memcpy(output.data(), tensor.data(), tensor.size());
    return output;
  }

  template <typename T>
  xnnpack::Buffer<T> RepeatRun() {
    xnnpack::Buffer<char>& tensor = this->buffers_.at(this->output_id_);
    xnn_invoke_runtime(Runtime());
    xnnpack::Buffer<float> output =
        xnnpack::Buffer<float>(tensor.size() / sizeof(float));
    memcpy(output.data(), tensor.data(), tensor.size());
    return output;
  }

  void CreateRuntime(uint32_t flags) {
    xnn_runtime_t runtime = nullptr;
    ASSERT_EQ(xnn_status_success,
              xnn_create_runtime_v3(this->subgraph_.get(), nullptr, nullptr,
                                    flags, &runtime));
    ASSERT_NE(nullptr, runtime);
    runtime_.reset(runtime);
  }

  void SetupRuntime() {
    auto& output = buffers_[output_id_];
    // Scramble output tensor.
    std::fill(output.begin(), output.end(), 0xA8);

    std::vector<xnn_external_value> externals;
    externals.reserve(this->external_tensors_.size());
    for (auto it = this->external_tensors_.begin();
         it != this->external_tensors_.end(); ++it) {
      externals.push_back(xnn_external_value{it->first, it->second});
    }

    ASSERT_EQ(xnn_status_success,
              xnn_setup_runtime(Runtime(), externals.size(), externals.data()));
    externals_ = std::move(externals);
  }

  void SetupRuntimeV2() {
    auto& output = buffers_[output_id_];
    // Scramble output tensor.
    std::fill(output.begin(), output.end(), 0xA8);

    std::vector<xnn_external_value> externals;
    externals.reserve(this->external_tensors_.size());
    for (auto it = this->external_tensors_.begin();
         it != this->external_tensors_.end(); ++it) {
      externals.push_back(xnn_external_value{it->first, it->second});
    }

    ASSERT_EQ(
        xnn_status_success,
        xnn_setup_runtime_v2(Runtime(), externals.size(), externals.data()));
    externals_ = std::move(externals);
  }

  size_t NumOperators() {
    size_t count = 0;
    for (size_t i = 0; i < runtime_->num_ops; i++) {
      if (runtime_->opdata[i].operator_objects[0] != nullptr) {
        count++;
      }
    }
    return count;
  }

  xnn_runtime_t Runtime() const { return runtime_.get(); }

  void ReshapeInput(const std::vector<size_t>& dims, uint32_t external_id) {
    size_t num_elements = NumElements(dims);
    xnnpack::Buffer<char> input(num_elements * sizeof(float) +
                                XNN_EXTRA_BYTES * sizeof(char));
    float* data = reinterpret_cast<float*>(input.data());
    std::generate(data, data + num_elements, [&]() { return f32dist(rng_); });
    ReshapeExternalTensor(dims, input.data(), external_id);
    buffers_[external_id] = std::move(input);
  }

  void ReshapeRuntime() {
    xnn_status status = xnn_reshape_runtime(Runtime());
    EXPECT_EQ(status, xnn_status_success);
    std::vector<size_t> output_dims(XNN_MAX_TENSOR_DIMS);
    size_t num_dims;
    status = xnn_get_external_value_shape(Runtime(), output_id_, &num_dims,
                                          output_dims.data());
    output_dims.resize(num_dims);
    EXPECT_EQ(status, xnn_status_success);
    buffers_[output_id_] =
        xnnpack::Buffer<char>(NumElements(output_dims) * sizeof(float));
    external_tensors_[output_id_] = buffers_[output_id_].data();
  }

 private:
  void Run(uint32_t flags = xnn_test_runtime_flags()) {
    CreateRuntime(flags);
    SetupRuntime();

    ASSERT_EQ(
        xnn_status_success,
        xnn_setup_runtime(Runtime(), externals_.size(), externals_.data()));
    ASSERT_EQ(xnn_status_success, xnn_invoke_runtime(Runtime()));
  };

  std::vector<xnn_external_value> externals_;
};

}  // namespace xnnpack
