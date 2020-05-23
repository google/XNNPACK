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
#include <random>
#include <vector>

#include <xnnpack.h>
#include <xnnpack/params.h>


class PadMicrokernelTester {
 public:
  inline PadMicrokernelTester& rows(size_t rows) {
    assert(rows != 0);
    this->rows_ = rows;
    return *this;
  }

  inline size_t rows() const {
    return this->rows_;
  }

  inline PadMicrokernelTester& input_channels(size_t input_channels) {
    assert(input_channels != 0);
    this->input_channels_ = input_channels;
    return *this;
  }

  inline size_t input_channels() const {
    return this->input_channels_;
  }

  inline PadMicrokernelTester& pre_padding(size_t pre_padding) {
    this->pre_padding_ = pre_padding;
    return *this;
  }

  inline size_t pre_padding() const {
    return this->pre_padding_;
  }

  inline PadMicrokernelTester& post_padding(size_t post_padding) {
    this->post_padding_ = post_padding;
    return *this;
  }

  inline size_t post_padding() const {
    return this->post_padding_;
  }

  inline size_t output_channels() const {
    return pre_padding() + input_channels() + post_padding();
  }

  inline PadMicrokernelTester& input_stride(size_t input_stride) {
    assert(input_stride != 0);
    this->input_stride_ = input_stride;
    return *this;
  }

  inline size_t input_stride() const {
    if (this->input_stride_ == 0) {
      return input_channels();
    } else {
      assert(this->input_stride_ >= input_channels());
      return this->input_stride_;
    }
  }

  inline PadMicrokernelTester& output_stride(size_t output_stride) {
    assert(output_stride != 0);
    this->output_stride_ = output_stride;
    return *this;
  }

  inline size_t output_stride() const {
    if (this->output_stride_ == 0) {
      return pre_padding() + input_channels() + post_padding();
    } else {
      assert(this->output_stride_ >= pre_padding() + input_channels() + post_padding());
      return this->output_stride_;
    }
  }

  inline PadMicrokernelTester& iterations(size_t iterations) {
    this->iterations_ = iterations;
    return *this;
  }

  inline size_t iterations() const {
    return this->iterations_;
  }

  void Test(xnn_x32_pad_ukernel_function pad) const {
    std::random_device random_device;
    auto rng = std::mt19937(random_device());
    auto u32rng = std::bind(std::uniform_int_distribution<uint32_t>(), rng);

    std::vector<uint32_t> input(input_channels() + (rows() - 1) * input_stride() + XNN_EXTRA_BYTES / sizeof(uint32_t));
    std::vector<uint32_t> output((pre_padding() + input_channels() + post_padding()) + (rows() - 1) * output_stride());
    std::vector<uint32_t> output_ref(rows() * (pre_padding() + input_channels() + post_padding()));
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(input.begin(), input.end(), std::ref(u32rng));
      std::generate(output.begin(), output.end(), std::ref(u32rng));
      const uint32_t fill_value = u32rng();

      // Compute reference results.
      std::fill(output_ref.begin(), output_ref.end(), fill_value);
      for (size_t i = 0; i < rows(); i++) {
        std::copy(
          &input[i * input_stride()],
          &input[i * input_stride() + input_channels()],
          &output_ref[i * output_channels() + pre_padding()]);
      }

      // Call optimized micro-kernel.
      pad(
        rows(),
        input_channels() * sizeof(uint32_t),
        pre_padding() * sizeof(uint32_t),
        post_padding() * sizeof(uint32_t),
        &fill_value,
        input.data(), input_stride() * sizeof(uint32_t),
        output.data(), output_stride() * sizeof(uint32_t));

      // Verify results.
      for (size_t i = 0; i < rows(); i++) {
        for (size_t c = 0; c < output_channels(); c++) {
          ASSERT_EQ(output_ref[i * output_channels() + c], output[i * output_stride() + c])
            << "at row " << i << " / " << rows() << ", channel " << i << " / " << output_channels()
            << " (" << pre_padding() << " + " << input_channels() << " + " << post_padding() << ")"
            << ", fill value = " << fill_value;
        }
      }
    }
  }

 private:
  size_t rows_{1};
  size_t input_channels_{1};
  size_t pre_padding_{0};
  size_t post_padding_{0};
  size_t input_stride_{0};
  size_t output_stride_{0};
  size_t iterations_{15};
};
