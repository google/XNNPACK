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
#include <vector>

#include <gtest/gtest.h>
#include "xnnpack.h"
#include "xnnpack/subgraph.h"
#include "subgraph-tester.h"

namespace xnnpack {

class RuntimeTester : public SubgraphTester {
 public:
  using SubgraphTester::SubgraphTester;

  template<typename T>
  std::vector<T> RunWithFusion() {
    Run();
    std::vector<char>& tensor = this->external_tensors_.at(this->output_id_);
    std::vector<float> output = std::vector<float>(tensor.size() / sizeof(float));
    std::memcpy(output.data(), tensor.data(), tensor.size());
    return output;
  }

  template<typename T>
  std::vector<T> RunWithoutFusion() {
    Run(XNN_FLAG_NO_OPERATOR_FUSION);
    std::vector<char>& tensor = this->external_tensors_.at(this->output_id_);
    std::vector<float> output = std::vector<float>(tensor.size() / sizeof(float));
    memcpy(output.data(), tensor.data(), tensor.size());
    return output;
  }

  template<typename T>
  std::vector<T> RepeatRun() {
    std::vector<char>& tensor = this->external_tensors_.at(this->output_id_);
    xnn_invoke_runtime(Runtime());
    std::vector<float> output = std::vector<float>(tensor.size() / sizeof(float));
    memcpy(output.data(), tensor.data(), tensor.size());
    return output;
  }

  void CreateRuntime(uint32_t flags = 0) {
    xnn_runtime_t runtime = nullptr;
    ASSERT_EQ(xnn_status_success, xnn_create_runtime_v3(this->subgraph_.get(), nullptr, nullptr, flags, &runtime));
    ASSERT_NE(nullptr, runtime);
    runtime_.reset(runtime);
  }

  void SetupRuntime() {
    std::vector<xnn_external_value> externals;
    for (auto it = this->external_tensors_.begin(); it != this->external_tensors_.end(); ++it) {
      if (it->first == this->output_id_) {
        // Scramble output tensor.
        std::fill(it->second.begin(), it->second.end(), 0xA8);
      }
      externals.push_back(xnn_external_value{it->first, it->second.data()});
    }

    ASSERT_EQ(xnn_status_success, xnn_setup_runtime(Runtime(), externals.size(), externals.data()));
    externals_ = externals;
  }

  void SetupRuntimeV2() {
    std::vector<xnn_external_value> externals;
    for (auto it = this->external_tensors_.begin(); it != this->external_tensors_.end(); ++it) {
      if (it->first == this->output_id_) {
        // Scramble output tensor.
        std::fill(it->second.begin(), it->second.end(), 0xA8);
      }
      externals.push_back(xnn_external_value{it->first, it->second.data()});
    }

    ASSERT_EQ(xnn_status_success, xnn_setup_runtime_v2(Runtime(), externals.size(), externals.data()));
    externals_ = externals;
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

  xnn_runtime_t Runtime() const {
    return runtime_.get();
  }

  void ReshapeInput(const std::vector<size_t>& dims, uint32_t external_id) {
    xnn_status status = xnn_reshape_external_value(Runtime(), external_id, dims.size(), dims.data());
    EXPECT_EQ(status, xnn_status_success);
    size_t num_elements = NumElements(dims);
    auto input = std::vector<char>(num_elements * sizeof(float) + XNN_EXTRA_BYTES * sizeof(char));
    float* data = reinterpret_cast<float*>(input.data());
    std::generate(data, data + num_elements, [&]() { return f32dist(rng_); });
    external_tensors_[external_id] = input;
  }

  void ReshapeRuntime() {
    xnn_status status = xnn_reshape_runtime(Runtime());
    EXPECT_EQ(status, xnn_status_success);
    std::vector<size_t> output_dims(XNN_MAX_TENSOR_DIMS);
    size_t num_dims;
    status = xnn_get_external_value_shape(Runtime(), output_id_, &num_dims, output_dims.data());
    output_dims.resize(num_dims);
    EXPECT_EQ(status, xnn_status_success);
    external_tensors_[output_id_].resize(NumElements(output_dims) * sizeof(float));
  }

 private:
  void Run(uint32_t flags = 0) {
    CreateRuntime(flags);
    SetupRuntime();

    ASSERT_EQ(xnn_status_success, xnn_setup_runtime(Runtime(), externals_.size(), externals_.data()));
    ASSERT_EQ(xnn_status_success, xnn_invoke_runtime(Runtime()));
  };

  std::unique_ptr<xnn_runtime, decltype(&xnn_delete_runtime)> runtime_{nullptr, xnn_delete_runtime};
  std::vector<xnn_external_value> externals_;
};

}  // namespace xnnpack
