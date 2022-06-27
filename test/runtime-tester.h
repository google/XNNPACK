// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <xnnpack.h>
#include <xnnpack/subgraph.h>

#include "subgraph-tester.h"

template<typename T>
class RuntimeTester : public SubgraphTester<T> {
 public:
  using SubgraphTester<T>::SubgraphTester;

  inline std::vector<T> RunWithFusion() {
    Run();
    std::vector<T>& output = this->external_tensors_.at(this->output_id);
    return output;
  }

  inline std::vector<T> RunWithoutFusion() {
    Run(XNN_FLAG_NO_OPERATOR_FUSION);
    std::vector<T>& output = this->external_tensors_.at(this->output_id);
    return output;
  }

  size_t NumOperators() {
    size_t count = 0;
    for (size_t i = 0; i < runtime_->num_ops; i++) {
      if (runtime_->opdata[i].operator_objects[0] != NULL) {
        count++;
      }
    }
    return count;
  }

 private:
  void ScrambleOutput(std::vector<float>& output) {
    std::fill(output.begin(), output.end(), std::nanf(""));
  }

  void Run(uint32_t flags = 0) {
    xnn_runtime_t runtime = nullptr;
    ASSERT_EQ(xnn_status_success, xnn_create_runtime_v3(this->subgraph_.get(), nullptr, nullptr, flags, &runtime));
    ASSERT_NE(nullptr, runtime);
    runtime_.reset(runtime);

    std::vector<xnn_external_value> externals;
    for (auto it = this->external_tensors_.begin(); it != this->external_tensors_.end(); ++it) {
      if (it->first == this->output_id) {
        // Scramble output data.
        ScrambleOutput(it->second);
      }
      externals.push_back(xnn_external_value{it->first, it->second.data()});
    }

    ASSERT_EQ(xnn_status_success, xnn_setup_runtime(runtime, externals.size(), externals.data()));
    ASSERT_EQ(xnn_status_success, xnn_invoke_runtime(runtime));
  };

  std::unique_ptr<xnn_runtime, decltype(&xnn_delete_runtime)> runtime_{nullptr, xnn_delete_runtime};
};
