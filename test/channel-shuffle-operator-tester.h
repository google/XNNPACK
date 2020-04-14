// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <gtest/gtest.h>

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdlib>
#include <functional>
#include <limits>
#include <random>
#include <vector>

#include <xnnpack.h>


class ChannelShuffleOperatorTester {
 public:
  inline ChannelShuffleOperatorTester& groups(size_t groups) {
    assert(groups != 0);
    this->groups_ = groups;
    return *this;
  }

  inline size_t groups() const {
    return this->groups_;
  }

  inline ChannelShuffleOperatorTester& group_channels(size_t group_channels) {
    assert(group_channels != 0);
    this->group_channels_ = group_channels;
    return *this;
  }

  inline size_t group_channels() const {
    return this->group_channels_;
  }

  inline size_t channels() const {
    return groups() * group_channels();
  }

  inline ChannelShuffleOperatorTester& input_stride(size_t input_stride) {
    assert(input_stride != 0);
    this->input_stride_ = input_stride;
    return *this;
  }

  inline size_t input_stride() const {
    if (this->input_stride_ == 0) {
      return channels();
    } else {
      assert(this->input_stride_ >= channels());
      return this->input_stride_;
    }
  }

  inline ChannelShuffleOperatorTester& output_stride(size_t output_stride) {
    assert(output_stride != 0);
    this->output_stride_ = output_stride;
    return *this;
  }

  inline size_t output_stride() const {
    if (this->output_stride_ == 0) {
      return channels();
    } else {
      assert(this->output_stride_ >= channels());
      return this->output_stride_;
    }
  }

  inline ChannelShuffleOperatorTester& batch_size(size_t batch_size) {
    assert(batch_size != 0);
    this->batch_size_ = batch_size;
    return *this;
  }

  inline size_t batch_size() const {
    return this->batch_size_;
  }

  inline ChannelShuffleOperatorTester& iterations(size_t iterations) {
    this->iterations_ = iterations;
    return *this;
  }

  inline size_t iterations() const {
    return this->iterations_;
  }

  void TestX8() const {
    std::random_device random_device;
    auto rng = std::mt19937(random_device());
    auto u8rng = std::bind(std::uniform_int_distribution<uint32_t>(0, std::numeric_limits<uint8_t>::max()), rng);

    std::vector<uint8_t> input(XNN_EXTRA_BYTES / sizeof(uint8_t) + (batch_size() - 1) * input_stride() + channels());
    std::vector<uint8_t> output((batch_size() - 1) * output_stride() + channels());
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(input.begin(), input.end(), std::ref(u8rng));
      std::fill(output.begin(), output.end(), 0xA5);

      // Create, setup, run, and destroy Channel Shuffle operator.
      ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
      xnn_operator_t channel_shuffle_op = nullptr;

      ASSERT_EQ(xnn_status_success,
        xnn_create_channel_shuffle_nc_x8(
          groups(), group_channels(),
          input_stride(), output_stride(),
          0, &channel_shuffle_op));
      ASSERT_NE(nullptr, channel_shuffle_op);

      // Smart pointer to automatically delete channel_shuffle_op.
      std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_channel_shuffle_op(channel_shuffle_op, xnn_delete_operator);

      ASSERT_EQ(xnn_status_success,
        xnn_setup_channel_shuffle_nc_x8(
          channel_shuffle_op,
          batch_size(),
          input.data(), output.data(),
          nullptr /* thread pool */));

      ASSERT_EQ(xnn_status_success,
        xnn_run_operator(channel_shuffle_op, nullptr /* thread pool */));

      // Verify results.
      for (size_t i = 0; i < batch_size(); i++) {
        for (size_t g = 0; g < groups(); g++) {
          for (size_t c = 0; c < group_channels(); c++) {
            ASSERT_EQ(uint32_t(input[i * input_stride() + g * group_channels() + c]),
                uint32_t(output[i * output_stride() + c * groups() + g]))
              << "batch index " << i << ", group " << g << ", channel " << c;
          }
        }
      }
    }
  }

  void TestX32() const {
    std::random_device random_device;
    auto rng = std::mt19937(random_device());
    auto f32rng = std::bind(std::uniform_real_distribution<float>(), rng);

    std::vector<float> input(XNN_EXTRA_BYTES / sizeof(float) + (batch_size() - 1) * input_stride() + channels());
    std::vector<float> output((batch_size() - 1) * output_stride() + channels());
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(input.begin(), input.end(), std::ref(f32rng));
      std::fill(output.begin(), output.end(), std::nanf(""));

      // Create, setup, run, and destroy Channel Shuffle operator.
      ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
      xnn_operator_t channel_shuffle_op = nullptr;

      ASSERT_EQ(xnn_status_success,
        xnn_create_channel_shuffle_nc_x32(
          groups(), group_channels(),
          input_stride(), output_stride(),
          0, &channel_shuffle_op));
      ASSERT_NE(nullptr, channel_shuffle_op);

      // Smart pointer to automatically delete channel_shuffle_op.
      std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_channel_shuffle_op(channel_shuffle_op, xnn_delete_operator);

      ASSERT_EQ(xnn_status_success,
        xnn_setup_channel_shuffle_nc_x32(
          channel_shuffle_op,
          batch_size(),
          input.data(), output.data(),
          nullptr /* thread pool */));

      ASSERT_EQ(xnn_status_success,
        xnn_run_operator(channel_shuffle_op, nullptr /* thread pool */));

      // Verify results.
      for (size_t i = 0; i < batch_size(); i++) {
        for (size_t g = 0; g < groups(); g++) {
          for (size_t c = 0; c < group_channels(); c++) {
            ASSERT_EQ(input[i * input_stride() + g * group_channels() + c],
                output[i * output_stride() + c * groups() + g])
              << "batch index " << i << ", group " << g << ", channel " << c;
          }
        }
      }
    }
  }

 private:
  size_t groups_{1};
  size_t group_channels_{1};
  size_t batch_size_{1};
  size_t input_stride_{0};
  size_t output_stride_{0};
  size_t iterations_{15};
};
