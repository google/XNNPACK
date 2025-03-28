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

#include <gtest/gtest.h>
#include "include/xnnpack.h"
#include "src/xnnpack/buffer.h"
#include "src/xnnpack/common.h"
#include "src/xnnpack/math.h"
#include "src/xnnpack/microfnptr.h"
#include "src/xnnpack/microparams.h"
#include "test/replicable_random_device.h"

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

  size_t batch_size() const { return this->batch_size_; }

  ReduceMicrokernelTester& iterations(size_t iterations) {
    this->iterations_ = iterations;
    return *this;
  }

  size_t iterations() const { return this->iterations_; }

  ReduceMicrokernelTester& rows(size_t rows) {
    assert(rows != 0);
    this->rows_ = rows;
    return *this;
  }

  size_t rows() const { return this->rows_; }

  ReduceMicrokernelTester& channels(size_t channels) {
    assert(channels != 0);
    this->channels_ = channels;
    return *this;
  }

  size_t channels() const { return this->channels_; }

  ReduceMicrokernelTester& input_stride(size_t input_stride) {
    assert(input_stride != 0);
    this->input_stride_ = input_stride;
    return *this;
  }

  size_t input_stride() const {
    if (this->input_stride_ == 0) {
      return channels();
    } else {
      assert(this->input_stride_ >= channels());
      return this->input_stride_;
    }
  }

  void Test(xnn_f16_reduce_ukernel_fn reduce, OpType op_type,
            xnn_init_f16_default_params_fn init_params = nullptr) const {
    xnnpack::ReplicableRandomDevice rng;
    std::uniform_real_distribution<float> f32dist(-1.0f, 1.0f);

    xnnpack::Buffer<xnn_float16> input(batch_size() +
                                       XNN_EXTRA_BYTES / sizeof(xnn_float16));
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate_n(input.begin(), batch_size(),
                      [&]() { return f32dist(rng); });

      // Compute reference results.
      xnn_float16 min_value = input[0];
      xnn_float16 max_value = input[0];
      for (size_t i = 0; i < batch_size(); ++i) {
        min_value = std::min<float>(min_value, input[i]);
        max_value = std::max<float>(max_value, input[i]);
      }

      // Prepare parameters.
      xnn_f16_default_params params;
      if (init_params != nullptr) {
        init_params(&params);
      }

      // Call optimized micro-kernel.
      xnn_float16 output[2];
      output[0] = f32dist(rng);
      min_value = std::min<float>(min_value, output[0]);
      if (op_type == OpType::MinMax) {
        output[1] = f32dist(rng);
        max_value = std::max<float>(max_value, output[1]);
      } else {
        max_value = std::max<float>(max_value, output[0]);
      }
      reduce(batch_size() * sizeof(xnn_float16), input.data(), output,
             init_params != nullptr ? &params : nullptr);

      // Verify results.
      switch (op_type) {
        case OpType::Max:
          EXPECT_EQ(output[0], max_value) << "with batch " << batch_size();
          break;
        case OpType::Min:
          EXPECT_EQ(output[0], min_value) << "with batch " << batch_size();
          break;
        case OpType::MinMax:
          EXPECT_EQ(output[0], min_value) << "with batch " << batch_size();
          EXPECT_EQ(output[1], max_value) << "with batch " << batch_size();
          break;
      }
    }
  }

  void Test(xnn_f32_reduce_ukernel_fn reduce, OpType op_type,
            xnn_init_f32_default_params_fn init_params = nullptr) const {
    xnnpack::ReplicableRandomDevice rng;
    std::uniform_real_distribution<float> f32dist(-1.0f, 1.0f);

    xnnpack::Buffer<float> input(batch_size() +
                                 XNN_EXTRA_BYTES / sizeof(float));
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(input.begin(), input.end(), [&]() { return f32dist(rng); });

      // Compute reference results.
      xnnpack::Buffer<float>::iterator min_it, max_it;
      std::tie(min_it, max_it) =
          std::minmax_element(input.begin(), input.begin() + batch_size());

      // Prepare parameters.
      xnn_f32_default_params params;
      if (init_params != nullptr) {
        init_params(&params);
      }

      // Call optimized micro-kernel.
      float output[2];
      float min_value = *min_it;
      float max_value = *max_it;
      output[0] = f32dist(rng);
      min_value = std::min(min_value, output[0]);
      if (op_type == OpType::MinMax) {
        output[1] = f32dist(rng);
        max_value = std::max(max_value, output[1]);
      } else {
        max_value = std::max(max_value, output[0]);
      }
      reduce(batch_size() * sizeof(float), input.data(), output,
             init_params != nullptr ? &params : nullptr);

      // Verify results.
      switch (op_type) {
        case OpType::Max:
          EXPECT_EQ(output[0], max_value) << "with batch " << batch_size();
          break;
        case OpType::Min:
          EXPECT_EQ(output[0], min_value) << "with batch " << batch_size();
          break;
        case OpType::MinMax:
          EXPECT_EQ(output[0], min_value) << "with batch " << batch_size();
          EXPECT_EQ(output[1], max_value) << "with batch " << batch_size();
          break;
      }
    }
  }

  void Test(xnn_s8_reduce_ukernel_fn reduce, OpType op_type) const {
    xnnpack::ReplicableRandomDevice rng;
    std::uniform_int_distribution<int32_t> i8dist(
        std::numeric_limits<int8_t>::min(), std::numeric_limits<int8_t>::max());

    xnnpack::Buffer<int8_t> input(batch_size() +
                                  XNN_EXTRA_BYTES / sizeof(int8_t));
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate_n(input.data(), batch_size(),
                      [&]() { return i8dist(rng); });

      // Compute reference results.
      xnnpack::Buffer<int8_t>::iterator min_it, max_it;
      std::tie(min_it, max_it) =
          std::minmax_element(input.begin(), input.begin() + batch_size());

      // Call optimized micro-kernel.
      int8_t output[2];
      output[0] = i8dist(rng);
      int8_t min_value = *min_it;
      int8_t max_value = *max_it;
      min_value = std::min(min_value, output[0]);
      if (op_type == OpType::MinMax) {
        output[1] = i8dist(rng);
        max_value = std::max(max_value, output[1]);
      } else {
        max_value = std::max(max_value, output[0]);
      }
      reduce(batch_size() * sizeof(int8_t), input.data(), output, nullptr);

      // Verify results.
      switch (op_type) {
        case OpType::Max:
          EXPECT_EQ(output[0], max_value) << "with batch " << batch_size();
          break;
        case OpType::Min:
          EXPECT_EQ(output[0], min_value) << "with batch " << batch_size();
          break;
        case OpType::MinMax:
          EXPECT_EQ(output[0], min_value) << "with batch " << batch_size();
          EXPECT_EQ(output[1], max_value) << "with batch " << batch_size();
          break;
      }
    }
  }

  void Test(xnn_f32_rdminmax_ukernel_fn reduce, OpType op_type) const {
    xnnpack::ReplicableRandomDevice rng;
    std::uniform_real_distribution<float> f32dist(-1.0f, 1.0f);

    xnnpack::Buffer<float> input((rows() - 1) * input_stride() + channels() +
                                 XNN_EXTRA_BYTES / sizeof(float));
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate_n(input.data(), (rows() - 1) * input_stride() + channels(),
                      [&]() { return f32dist(rng); });

      xnnpack::Buffer<float> output(channels());
      xnnpack::Buffer<float> output_ref(channels());

      std::generate_n(output.begin(), channels(),
                      [&]() { return f32dist(rng); });

      // Compute reference results.
      switch (op_type) {
        case OpType::Max:
          for (size_t c = 0; c < channels(); ++c) {
            float max_value = output[c];
            for (size_t n = 0; n < rows(); ++n) {
              max_value = std::max(max_value, input[n * input_stride() + c]);
            }
            output_ref[c] = max_value;
          }
          break;
        case OpType::Min:
          for (size_t c = 0; c < channels(); ++c) {
            float min_value = output[c];
            for (size_t n = 0; n < rows(); ++n) {
              min_value = std::min(min_value, input[n * input_stride() + c]);
            }
            output_ref[c] = min_value;
          }
          break;
        default:
          XNN_UNREACHABLE;
      }

      // Call optimized micro-kernel.
      reduce(rows(), channels(), input.data(), input_stride() * sizeof(float),
             /*zero=*/nullptr, output.data(), /*params=*/nullptr);

      // Verify results.
      for (size_t c = 0; c < channels(); c++) {
        ASSERT_NEAR(output[c], output_ref[c], std::abs(output_ref[c]) * 1.0e-6f)
            << "at position " << c << ", rows = " << rows()
            << ", channels = " << channels();
      }
    }
  }

  void Test(xnn_f16_rdminmax_ukernel_fn reduce, OpType op_type) const {
    xnnpack::ReplicableRandomDevice rng;
    std::uniform_real_distribution<float> f32dist(-1.0f, 1.0f);

    xnnpack::Buffer<xnn_float16> input((rows() - 1) * input_stride() +
                                       channels() +
                                       XNN_EXTRA_BYTES / sizeof(xnn_float16));
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate_n(input.data(), (rows() - 1) * input_stride() + channels(),
                      [&]() { return f32dist(rng); });

      xnnpack::Buffer<xnn_float16> output(channels());
      xnnpack::Buffer<xnn_float16> output_ref(channels());

      std::generate_n(output.begin(), channels(),
                      [&]() { return f32dist(rng); });

      // Compute reference results.
      switch (op_type) {
        case OpType::Max:
          for (size_t c = 0; c < channels(); ++c) {
            xnn_float16 max_value = output[c];
            for (size_t n = 0; n < rows(); ++n) {
              max_value =
                  std::max<float>(max_value, input[n * input_stride() + c]);
            }
            output_ref[c] = max_value;
          }
          break;
        case OpType::Min:
          for (size_t c = 0; c < channels(); ++c) {
            xnn_float16 min_value = output[c];
            for (size_t n = 0; n < rows(); ++n) {
              min_value =
                  std::min<float>(min_value, input[n * input_stride() + c]);
            }
            output_ref[c] = min_value;
          }
          break;
        default:
          XNN_UNREACHABLE;
      }

      // Call optimized micro-kernel.
      reduce(rows(), channels(), input.data(),
             input_stride() * sizeof(xnn_float16), /*zero=*/nullptr,
             output.data(), /*params=*/nullptr);

      // Verify results.
      for (size_t c = 0; c < channels(); c++) {
        ASSERT_EQ(output[c], output_ref[c])
            << "at position " << c << ", rows = " << rows()
            << ", channels = " << channels();
      }
    }
  }

  void Test(xnn_s8_rdminmax_ukernel_fn reduce, OpType op_type) const {
    xnnpack::ReplicableRandomDevice rng;
    std::uniform_int_distribution<int32_t> i8dist(
        std::numeric_limits<int8_t>::min(), std::numeric_limits<int8_t>::max());

    xnnpack::Buffer<int8_t> input((rows() - 1) * input_stride() + channels() +
                                  XNN_EXTRA_BYTES / sizeof(int8_t));
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate_n(input.data(), (rows() - 1) * input_stride() + channels(),
                      [&]() { return i8dist(rng); });

      xnnpack::Buffer<int8_t> output(channels());
      xnnpack::Buffer<int8_t> output_ref(channels());

      std::generate_n(output.begin(), channels(),
                      [&]() { return i8dist(rng); });

      // Compute reference results.
      switch (op_type) {
        case OpType::Max:
          for (size_t c = 0; c < channels(); ++c) {
            int8_t max_value = output[c];
            for (size_t n = 0; n < rows(); ++n) {
              max_value = std::max(max_value, input[n * input_stride() + c]);
            }
            output_ref[c] = max_value;
          }
          break;
        case OpType::Min:
          for (size_t c = 0; c < channels(); ++c) {
            int8_t min_value = output[c];
            for (size_t n = 0; n < rows(); ++n) {
              min_value = std::min(min_value, input[n * input_stride() + c]);
            }
            output_ref[c] = min_value;
          }
          break;
        default:
          XNN_UNREACHABLE;
      }

      // Call optimized micro-kernel.
      reduce(rows(), channels(), input.data(), input_stride() * sizeof(int8_t),
             /*zero=*/nullptr, output.data(), /*params=*/nullptr);

      // Verify results.
      for (size_t c = 0; c < channels(); c++) {
        ASSERT_EQ(output[c], output_ref[c])
            << "at position " << c << ", rows = " << rows()
            << ", channels = " << channels();
      }
    }
  }

  void Test(xnn_u8_reduce_ukernel_fn reduce, OpType op_type) const {
    xnnpack::ReplicableRandomDevice rng;
    std::uniform_int_distribution<int32_t> u8dist(
        std::numeric_limits<uint8_t>::min(),
        std::numeric_limits<uint8_t>::max());

    xnnpack::Buffer<uint8_t> input(batch_size() +
                                   XNN_EXTRA_BYTES / sizeof(uint8_t));
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(input.begin(), input.end(), [&]() { return u8dist(rng); });

      // Compute reference results.
      xnnpack::Buffer<uint8_t>::iterator min_it, max_it;
      std::tie(min_it, max_it) =
          std::minmax_element(input.begin(), input.begin() + batch_size());

      // Call optimized micro-kernel.
      uint8_t output[2];
      output[0] = u8dist(rng);
      uint8_t min_value = *min_it;
      uint8_t max_value = *max_it;
      min_value = std::min(min_value, output[0]);
      if (op_type == OpType::MinMax) {
        output[1] = u8dist(rng);
        max_value = std::max(max_value, output[1]);
      } else {
        max_value = std::max(max_value, output[0]);
      }
      reduce(batch_size() * sizeof(uint8_t), input.data(), output, nullptr);

      // Verify results.
      switch (op_type) {
        case OpType::Max:
          EXPECT_EQ(output[0], max_value) << "with batch " << batch_size();
          break;
        case OpType::Min:
          EXPECT_EQ(output[0], min_value) << "with batch " << batch_size();
          break;
        case OpType::MinMax:
          EXPECT_EQ(output[0], min_value) << "with batch " << batch_size();
          EXPECT_EQ(output[1], max_value) << "with batch " << batch_size();
          break;
      }
    }
  }

  void Test(xnn_u8_rdminmax_ukernel_fn reduce, OpType op_type) const {
    xnnpack::ReplicableRandomDevice rng;
    std::uniform_int_distribution<int32_t> u8dist(
        std::numeric_limits<uint8_t>::min(),
        std::numeric_limits<uint8_t>::max());

    xnnpack::Buffer<uint8_t> input((rows() - 1) * input_stride() + channels() +
                                   XNN_EXTRA_BYTES / sizeof(int8_t));
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate_n(input.data(), (rows() - 1) * input_stride() + channels(),
                      [&]() { return u8dist(rng); });

      xnnpack::Buffer<uint8_t> output(channels());
      xnnpack::Buffer<uint8_t> output_ref(channels());

      std::generate_n(output.begin(), channels(),
                      [&]() { return u8dist(rng); });

      // Compute reference results.
      switch (op_type) {
        case OpType::Max:
          for (size_t c = 0; c < channels(); ++c) {
            uint8_t max_value = output[c];
            for (size_t n = 0; n < rows(); ++n) {
              max_value = std::max(max_value, input[n * input_stride() + c]);
            }
            output_ref[c] = max_value;
          }
          break;
        case OpType::Min:
          for (size_t c = 0; c < channels(); ++c) {
            uint8_t min_value = output[c];
            for (size_t n = 0; n < rows(); ++n) {
              min_value = std::min(min_value, input[n * input_stride() + c]);
            }
            output_ref[c] = min_value;
          }
          break;
        default:
          XNN_UNREACHABLE;
      }

      // Call optimized micro-kernel.
      reduce(rows(), channels(), input.data(), input_stride() * sizeof(uint8_t),
             /*zero=*/nullptr, output.data(), /*params=*/nullptr);

      // Verify results.
      for (size_t c = 0; c < channels(); c++) {
        ASSERT_EQ(output[c], output_ref[c])
            << "at position " << c << ", rows = " << rows()
            << ", channels = " << channels();
      }
    }
  }

 private:
  size_t rows_{1};
  size_t channels_{1};
  size_t input_stride_{0};
  size_t batch_size_{1};
  size_t iterations_{15};
};
