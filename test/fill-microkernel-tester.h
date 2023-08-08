// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <gtest/gtest.h>

#include <array>
#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdlib>
#include <functional>
#include <iomanip>
#include <ios>
#include <random>
#include <vector>

#include <xnnpack.h>
#include <xnnpack/microfnptr.h>


class FillMicrokernelTester {
 public:
  inline FillMicrokernelTester& rows(size_t rows) {
    assert(rows != 0);
    this->rows_ = rows;
    return *this;
  }

  inline size_t rows() const {
    return this->rows_;
  }

  inline FillMicrokernelTester& channels(size_t channels) {
    assert(channels != 0);
    this->channels_ = channels;
    return *this;
  }

  inline size_t channels() const {
    return this->channels_;
  }

  inline FillMicrokernelTester& output_stride(size_t output_stride) {
    assert(output_stride != 0);
    this->output_stride_ = output_stride;
    return *this;
  }

  inline size_t output_stride() const {
    if (this->output_stride_ == 0) {
      return channels();
    } else {
      return this->output_stride_;
    }
  }

  inline FillMicrokernelTester& iterations(size_t iterations) {
    this->iterations_ = iterations;
    return *this;
  }

  inline size_t iterations() const {
    return this->iterations_;
  }

  void Test(xnn_fill_ukernel_fn fill) const {
    ASSERT_GE(output_stride(), channels());

    std::random_device random_device;
    auto rng = std::mt19937(random_device());
    auto u8rng = std::bind(std::uniform_int_distribution<uint32_t>(0, std::numeric_limits<uint8_t>::max()), rng);

    std::vector<uint8_t> output((rows() - 1) * output_stride() + channels());
    std::vector<uint8_t> output_copy(output.size());
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(output.begin(), output.end(), std::ref(u8rng));
      std::copy(output.cbegin(), output.cend(), output_copy.begin());
      std::array<uint8_t, 4> fill_pattern;
      std::generate(fill_pattern.begin(), fill_pattern.end(), std::ref(u8rng));
      uint32_t fill_value = 0;
      memcpy(&fill_value, fill_pattern.data(), sizeof(fill_value));

      // Call optimized micro-kernel.
      fill(
        rows(),
        channels() * sizeof(uint8_t),
        output.data(),
        output_stride() * sizeof(uint8_t),
        fill_value);

      // Verify results.
      for (size_t i = 0; i < rows(); i++) {
        for (size_t c = 0; c < channels(); c++) {
          EXPECT_EQ(uint32_t(output[i * output_stride() + c]), uint32_t(fill_pattern[c % fill_pattern.size()]))
            << "at row " << i << " / " << rows()
            << ", channel " << c << " / " << channels()
            << ", fill value 0x" << std::hex << std::setw(8) << std::setfill('0') << fill_value
            << ", output value 0x" << std::hex << std::setw(8) << std::setfill('0') << output[i * output_stride() + c];
        }
      }
      for (size_t i = 0; i + 1 < rows(); i++) {
        for (size_t c = channels(); c < output_stride(); c++) {
          EXPECT_EQ(uint32_t(output[i * output_stride() + c]), uint32_t(output_copy[i * output_stride() + c]))
            << "at row " << i << " / " << rows()
            << ", channel " << c << " / " << channels()
            << ", original value 0x" << std::hex << std::setw(8) << std::setfill('0')
            << output_copy[i * output_stride() + c]
            << ", output value 0x" << std::hex << std::setw(8) << std::setfill('0') << output[i * output_stride() + c];
        }
      }
    }
  }

 private:
  size_t rows_{1};
  size_t channels_{1};
  size_t output_stride_{0};
  size_t iterations_{15};
};
