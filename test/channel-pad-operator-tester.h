// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <gtest/gtest.h>

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <functional>
#include <random>
#include <vector>

#include <xnnpack.h>


class ChannelPadOperatorTester {
 public:
  inline ChannelPadOperatorTester& batch_size(size_t batch_size) {
    assert(batch_size != 0);
    this->batch_size_ = batch_size;
    return *this;
  }

  inline size_t batch_size() const {
    return this->batch_size_;
  }

  inline ChannelPadOperatorTester& input_channels(size_t input_channels) {
    assert(input_channels != 0);
    this->input_channels_ = input_channels;
    return *this;
  }

  inline size_t input_channels() const {
    return this->input_channels_;
  }

  inline ChannelPadOperatorTester& pad_before(size_t pad_before) {
    this->pad_before_ = pad_before;
    return *this;
  }

  inline size_t pad_before() const {
    return this->pad_before_;
  }

  inline ChannelPadOperatorTester& pad_after(size_t pad_after) {
    this->pad_after_ = pad_after;
    return *this;
  }

  inline size_t pad_after() const {
    return this->pad_after_;
  }

  inline size_t output_channels() const {
    return pad_before() + input_channels() + pad_after();
  }

  inline ChannelPadOperatorTester& input_stride(size_t input_stride) {
    assert(input_stride != 0);
    this->input_stride_ = input_stride;
    return *this;
  }

  inline size_t input_stride() const {
    if (this->input_stride_ == 0) {
      return this->input_channels_;
    } else {
      assert(this->input_stride_ >= this->input_channels_);
      return this->input_stride_;
    }
  }

  inline ChannelPadOperatorTester& output_stride(size_t output_stride) {
    assert(output_stride != 0);
    this->output_stride_ = output_stride;
    return *this;
  }

  inline size_t output_stride() const {
    if (this->output_stride_ == 0) {
      return output_channels();
    } else {
      assert(this->output_stride_ >= output_channels());
      return this->output_stride_;
    }
  }

  inline ChannelPadOperatorTester& iterations(size_t iterations) {
    this->iterations_ = iterations;
    return *this;
  }

  inline size_t iterations() const {
    return this->iterations_;
  }

  void TestX32() const {
    std::random_device random_device;
    auto rng = std::mt19937(random_device());
    auto u32rng = std::bind(std::uniform_int_distribution<uint32_t>(), rng);

    const uint32_t pad_value = u32rng();
    std::vector<uint32_t> input(input_channels() + (batch_size() - 1) * input_stride() + XNN_EXTRA_BYTES / sizeof(uint32_t));
    std::vector<uint32_t> output(output_channels() + (batch_size() - 1) * output_stride());
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(input.begin(), input.end(), std::ref(u32rng));
      std::generate(output.begin(), output.end(), std::ref(u32rng));

      // Create, setup, run, and destroy Channel Pad operator.
      ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
      xnn_operator_t channel_pad_op = nullptr;

      ASSERT_EQ(xnn_status_success,
        xnn_create_channel_pad_nc_x32(
          input_channels(), pad_before(), pad_after(),
          input_stride(), output_stride(),
          &pad_value,
          0 /* flags */,
          &channel_pad_op));
      ASSERT_NE(nullptr, channel_pad_op);

      // Smart pointer to automatically delete channel_pad_op.
      std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_channel_pad_op(channel_pad_op, xnn_delete_operator);

      ASSERT_EQ(xnn_status_success,
        xnn_setup_channel_pad_nc_x32(
          channel_pad_op,
          batch_size(),
          input.data(), output.data(),
          nullptr /* thread pool */));

      ASSERT_EQ(xnn_status_success,
        xnn_run_operator(channel_pad_op, nullptr /* thread pool */));

      // Verify results.
      for (size_t i = 0; i < batch_size(); i++) {
        for (size_t k = 0; k < pad_before(); k++) {
          ASSERT_EQ(pad_value,
            output[i * output_stride() + k]);
        }
        for (size_t k = 0; k < input_channels(); k++) {
          ASSERT_EQ(input[i * input_stride() + k],
            output[i * output_stride() + pad_before() + k]);
        }
        for (size_t k = 0; k < pad_after(); k++) {
          ASSERT_EQ(pad_value,
            output[i * output_stride() + pad_before() + input_channels() + k]);
        }
      }
    }
  }

 private:
  size_t batch_size_{1};
  size_t input_channels_{1};
  size_t pad_before_{0};
  size_t pad_after_{0};
  size_t input_stride_{0};
  size_t output_stride_{0};
  size_t iterations_{15};
};
