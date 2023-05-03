// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdlib>
#include <cstring>
#include <vector>

#include <xnnpack.h>
#include <xnnpack/subgraph.h>

#include "subgraph-tester.h"

namespace xnnpack {

class RuntimeTester : public SubgraphTester {
 public:
  using SubgraphTester::SubgraphTester;

  template<typename T>
  inline std::vector<T> RunWithFusion() {
    Run();
    std::vector<char>& tensor = this->external_tensors_.at(this->output_id_);
    std::vector<float> output = std::vector<float>(tensor.size() / sizeof(float));
    std::memcpy(output.data(), tensor.data(), tensor.size());
    return output;
  }

  template<typename T>
  inline std::vector<T> RunWithoutFusion() {
    Run(XNN_FLAG_NO_OPERATOR_FUSION);
    std::vector<char>& tensor = this->external_tensors_.at(this->output_id_);
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
