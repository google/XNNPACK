// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <random>
#include <tuple>
#include <vector>

#include <gtest/gtest.h>
#include "xnnpack.h"
#include "xnnpack/math.h"
#include "xnnpack/microfnptr.h"
#include "xnnpack/microparams.h"
#include "xnnpack/buffer.h"
#include "replicable_random_device.h"

class ReduceMicrokernelTester {
 public:
  enum class OpType {
    Max,
    Min,
    MinMax,
  };

  ReduceMicrokernelTester& batch_size(size_t batch_size) {
    assert(batch_size != 0);
    this->batch_size_ = batch_size;
    return *this;
  }

  size_t batch_size() const {
    return this->batch_size_;
  }

  ReduceMicrokernelTester& iterations(size_t iterations) {
    this->iterations_ = iterations;
    return *this;
  }

  size_t iterations() const {
    return this->iterations_;
  }

  void Test(xnn_f16_reduce_ukernel_fn reduce, OpType op_type, xnn_init_f16_default_params_fn init_params = nullptr) const {
    xnnpack::ReplicableRandomDevice rng;
    std::uniform_real_distribution<float> f32dist(-1.0f, 1.0f);

    xnnpack::Buffer<xnn_float16> input(batch_size() + XNN_EXTRA_BYTES / sizeof(xnn_float16));
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(input.begin(), input.end(), [&]() { return f32dist(rng); });

      // Compute reference results.
      xnnpack::Buffer<xnn_float16>::iterator min, max;
      std::tie(min, max) = std::minmax_element(input.begin(), input.begin() + batch_size());

      // Prepare parameters.
      xnn_f16_default_params params;
      if (init_params != nullptr) {
        init_params(&params);
      }

      // Call optimized micro-kernel.
      xnn_float16 output[2];
      reduce(batch_size() * sizeof(xnn_float16), input.data(), output, init_params != nullptr ? &params : nullptr);

      // Verify results.
      switch (op_type) {
        case OpType::Max:
          EXPECT_EQ(output[0], *max)
              << "with batch " << batch_size();
          break;
        case OpType::Min:
          EXPECT_EQ(output[0], *min)
              << "with batch " << batch_size();
          break;
        case OpType::MinMax:
          EXPECT_EQ(output[0], *min)
              << "with batch " << batch_size();
          EXPECT_EQ(output[1], *max)
              << "with batch " << batch_size();
          break;
      }
    }
  }

  void Test(xnn_f32_reduce_ukernel_fn reduce, OpType op_type, xnn_init_f32_default_params_fn init_params = nullptr) const {
    xnnpack::ReplicableRandomDevice rng;
    std::uniform_real_distribution<float> f32dist(-1.0f, 1.0f);

    xnnpack::Buffer<float> input(batch_size() + XNN_EXTRA_BYTES / sizeof(float));
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(input.begin(), input.end(), [&]() { return f32dist(rng); });

      // Compute reference results.
      xnnpack::Buffer<float>::iterator min, max;
      std::tie(min, max) = std::minmax_element(input.begin(), input.begin() + batch_size());

      // Prepare parameters.
      xnn_f32_default_params params;
      if (init_params != nullptr) {
        init_params(&params);
      }

      // Call optimized micro-kernel.
      float output[2] = {std::nanf(""), std::nanf("")};
      reduce(batch_size() * sizeof(float), input.data(), output, init_params != nullptr ? &params : nullptr);

      // Verify results.
      switch (op_type) {
        case OpType::Max:
          EXPECT_EQ(output[0], *max)
              << "with batch " << batch_size();
          break;
        case OpType::Min:
          EXPECT_EQ(output[0], *min)
              << "with batch " << batch_size();
          break;
        case OpType::MinMax:
          EXPECT_EQ(output[0], *min)
              << "with batch " << batch_size();
          EXPECT_EQ(output[1], *max)
              << "with batch " << batch_size();
          break;
      }
    }
  }

  void Test(xnn_u8_reduce_ukernel_fn reduce, OpType op_type) const {
    xnnpack::ReplicableRandomDevice rng;
    std::uniform_int_distribution<int32_t> u8dist(
      std::numeric_limits<uint8_t>::min(), std::numeric_limits<uint8_t>::max());

    xnnpack::Buffer<uint8_t> input(batch_size() + XNN_EXTRA_BYTES / sizeof(uint8_t));
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(input.begin(), input.end(), [&]() { return u8dist(rng); });

      // Compute reference results.
      xnnpack::Buffer<uint8_t>::iterator min, max;
      std::tie(min, max) = std::minmax_element(input.begin(), input.begin() + batch_size());

      // Call optimized micro-kernel.
      uint8_t output[2] = {0xAA, 0xAA};
      reduce(batch_size() * sizeof(uint8_t), input.data(), output, nullptr);

      // Verify results.
      switch (op_type) {
        case OpType::Max:
          EXPECT_EQ(output[0], *max)
              << "with batch " << batch_size();
          break;
        case OpType::Min:
          EXPECT_EQ(output[0], *min)
              << "with batch " << batch_size();
          break;
        case OpType::MinMax:
          EXPECT_EQ(output[0], *min)
              << "with batch " << batch_size();
          EXPECT_EQ(output[1], *max)
              << "with batch " << batch_size();
          break;
      }
    }
  }

 private:
  size_t batch_size_{1};
  size_t iterations_{15};
};
