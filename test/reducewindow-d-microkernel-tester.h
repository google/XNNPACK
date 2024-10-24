// Copyright 2024 Google LLC
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

#include "replicable_random_device.h"
#include "xnnpack.h"
#include "xnnpack/microfnptr.h"
#include "xnnpack/microparams.h"

class RWDMicrokernelTester {
 public:
  enum class OpType {
    Sum,
  };

  RWDMicrokernelTester& rows(size_t rows) {
    assert(rows != 0);
    this->rows_ = rows;
    return *this;
  }

  size_t rows() const {
    return this->rows_;
  }

  RWDMicrokernelTester& channels(size_t channels) {
    assert(channels != 0);
    this->channels_ = channels;
    return *this;
  }

  size_t channels() const {
    return this->channels_;
  }

  RWDMicrokernelTester& channel_tile(size_t channel_tile) {
    assert(channel_tile != 0);
    this->channel_tile_ = channel_tile;
    return *this;
  }

  size_t channel_tile() const {
    return this->channel_tile_;
  }
 RWDMicrokernelTester& init_value(float init_value) {
    assert(init_value != 0);
    this->init_value_ = init_value;
    return *this;
  }

  float init_value() const {
    return this->init_value_;
  }

 RWDMicrokernelTester& padding(int64_t* padding) {
    assert(padding != 0);
    this->padding_[0] = padding[0];
    this->padding_[1] = padding[1];
    return *this;
  }

  int64_t padding_high() const {
    return this->padding_[0];
  }

  int64_t padding_low() const {
    return this->padding_[1];
  }

 RWDMicrokernelTester& base_dilation(int64_t base_dilation) {
    assert(base_dilation != 0);
    this->base_dilation_ = base_dilation;
    return *this;
  }

  int64_t base_dilation() const {
    return this->base_dilation_;
  }

 RWDMicrokernelTester& window_dilation(int64_t window_dilation) {
    assert(window_dilation != 0);
    this->window_dilation_ = window_dilation;
    return *this;
  }

  int64_t window_dilation() const {
    return this->window_dilation_;
  }

 RWDMicrokernelTester& window_dimension(int64_t window_dimension) {
    assert(window_dimension != 0);
    this->window_dimension_ = window_dimension;
    return *this;
  }

  int64_t window_dimension() const {
    return this->window_dimension_;
  }

 RWDMicrokernelTester& window_stride(int64_t window_stride) {
    assert(window_stride != 0);
    this->window_stride_ = window_stride;
    return *this;
  }

  int64_t window_stride() const {
    return this->window_stride_;
  }

  RWDMicrokernelTester& iterations(size_t iterations) {
    this->iterations_ = iterations;
    return *this;
  }

  size_t iterations() const {
    return this->iterations_;
  }

  void Test(xnn_f32_rwd_ukernel_fn reduce_window, OpType op_type, xnn_init_f32_default_params_fn init_params = nullptr) const {
    xnnpack::ReplicableRandomDevice rng;
    std::uniform_real_distribution<float> f32dist;

    std::vector<float> input(rows() * channels() + XNN_EXTRA_BYTES / sizeof(float));
    
    int64_t padding[2] = {padding_high(), padding_low()};
    int64_t size = rows();
    int64_t padded_size = size + (size - 1) * (base_dilation() - 1) + padding[0] + padding[1];
    int64_t output_size = (padded_size - (window_dimension() - 1) * window_dilation() - 1) / window_stride() + 1;

    std::vector<float> output_ref(output_size * channels());
    std::vector<float> output(output_size * channels());

    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(input.begin(), input.end(), [&]() { return f32dist(rng); });

      // Compute reference results.
      for (int64_t j = 0; j < output_size; j++) {
        for (int64_t i = 0; i < channels(); i++) {
          float sum = init_value();
          for (int64_t k = 0; k < window_dimension(); k++) {
            int64_t window_row = j * window_stride() + k * window_dilation();
            if (window_row < padding[0] || window_row >= padded_size - padding[1] || 
              (window_row - padding[0]) % base_dilation() != 0) {
                sum += init_value();
                continue;
            }
            window_row = (window_row - padding[0]) / base_dilation();
            sum += input[window_row * channels() + i];
          }
          output_ref[j * channels() + i] = sum;
        }
      }

      // Prepare parameters.
      xnn_f32_default_params params;
      if (init_params != nullptr) {
        init_params(&params);
      }

      // Call optimized micro-kernel.
      reduce_window(rows(), channels(), input.data(), init_value(), padding, base_dilation(),
          window_dilation(), window_dimension(), window_stride(), output.data(), init_params != nullptr ? &params : nullptr);

      // Verify results.
      switch (op_type) {
        case OpType::Sum:
        for (size_t i = 0; i < output_size * channels(); i++){
          EXPECT_EQ(output[i], output_ref[i])<< "with batch " << rows();
        }
        break;
      }
    }
  }

 private:
  size_t rows_{1};
  size_t channels_{1};
  size_t channel_tile_{1};
  size_t iterations_{15};
  float init_value_{0.0};
  int64_t padding_[2] = {1,1};
  int64_t base_dilation_{1};
  int64_t window_dimension_{1};
  int64_t window_dilation_{1};
  int64_t window_stride_{1};
};

#define XNN_TEST_RWD_CHANNEL_EQ_ROW_EQ(ukernel, ...)               \
  TEST(ukernel, channels_eq_1_2pass_fulltile) {                    \
    RWDMicrokernelTester()                                         \
      .rows(1)                                                     \
      .channels(1)                                                 \
      .Test(__VA_ARGS__);                                          \
  }

#define XNN_TEST_RWD_CHANNEL_EQ_ROW_GT(ukernel, ...)               \
  TEST(ukernel, channels_eq_1_multipass_fulltile) {                \
    for (size_t rows = 1; rows <= 5; rows += 1) {                  \
      RWDMicrokernelTester()                                       \
        .rows(rows)                                                \
        .channels(1)                                               \
        .Test(__VA_ARGS__);                                        \
    }                                                              \
  }

#define XNN_TEST_RWD_CHANNEL_GT_ROW_EQ(ukernel, ...)               \
  TEST(ukernel, channels_div_1_2pass_fulltile) {                   \
    for (size_t channels = 2; channels < 8; channels += 1) {       \
      RWDMicrokernelTester()                                       \
        .rows(1)                                                   \
        .channels(channels)                                        \
        .Test(__VA_ARGS__);                                        \
    }                                                              \
  }

#define XNN_TEST_RWD_CHANNEL_GT_ROW_GT(ukernel, ...)               \
  TEST(ukernel, channels_div_1_multipass_fulltile) {               \
    for (size_t channels = 2; channels <= 10; channels += 1) {     \
      for (size_t rows = 1; rows <= 5; rows += 1) {                \
        RWDMicrokernelTester()                                     \
          .rows(rows)                                              \
          .channels(channels)                                      \
          .Test(__VA_ARGS__);                                      \
      }                                                            \
    }                                                              \
  }