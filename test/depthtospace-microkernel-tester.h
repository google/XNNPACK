// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <gtest/gtest.h>

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <random>
#include <vector>

#include <xnnpack.h>
#include <xnnpack/params.h>


class DepthToSpaceMicrokernelTester {
 public:
  inline DepthToSpaceMicrokernelTester& output_channels(size_t output_channels) {
    assert(output_channels != 0);
    this->output_channels_ = output_channels;
    return *this;
  }

  inline size_t output_channels() const {
    return this->output_channels_;
  }

  inline size_t input_channels() const {
    return this->output_channels() * this->block_size() * this->block_size();
  }

  inline DepthToSpaceMicrokernelTester& input_height(size_t input_height) {
    assert(input_height != 0);
    this->input_height_ = input_height;
    return *this;
  }

  inline size_t input_height() const {
    return this->input_height_;
  }

  inline size_t output_height() const {
    return this->input_height() * this->block_size();
  }

  inline DepthToSpaceMicrokernelTester& input_width(size_t input_width) {
    assert(input_width != 0);
    this->input_width_ = input_width;
    return *this;
  }

  inline size_t input_width() const {
    return this->input_width_;
  }

  inline size_t output_width() const {
    return this->input_width() * this->block_size();
  }

  inline DepthToSpaceMicrokernelTester& block_size(size_t block_size) {
    assert(block_size != 0);
    this->block_size_ = block_size;
    return *this;
  }

  inline size_t block_size() const {
    return this->block_size_;
  }

  inline DepthToSpaceMicrokernelTester& output_channel_stride(size_t output_channel_stride) {
    assert(output_channel_stride != 0);
    this->output_channel_stride_ = output_channel_stride;
    return *this;
  }

  inline size_t output_channel_stride() const {
    if (this->output_channel_stride_ != 0) {
      return this->output_channel_stride_;
    } else {
      return this->output_channels();
    }
  }

  inline DepthToSpaceMicrokernelTester& iterations(size_t iterations) {
    this->iterations_ = iterations;
    return *this;
  }

  inline size_t iterations() const {
    return this->iterations_;
  }

  void Test(xnn_x32_depthtospace2d_chw2hwc_ukernel_function depthtospace2d) const {
    ASSERT_GE(block_size(), 2);

    std::random_device random_device;
    auto rng = std::mt19937(random_device());
    auto u32rng = std::bind(std::uniform_int_distribution<uint32_t>(), rng);

    std::vector<uint32_t> input(input_channels() * input_height() * input_width());
    std::vector<uint32_t> output((output_height() * output_width() - 1) * output_channel_stride() + output_channels());

    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(input.begin(), input.end(), std::ref(u32rng));

      // Call optimized micro-kernel.
      depthtospace2d(
        output_channels(),
        input_height(),
        input_width(),
        block_size(),
        input.data(),
        output.data(),
        output_channel_stride());

      // Verify results.
      for (size_t iy = 0; iy < input_height(); iy++) {
        for (size_t by = 0; by < block_size(); by++) {
          for (size_t ix = 0; ix < input_width(); ix++) {
            for (size_t bx = 0; bx < block_size(); bx++) {
              for (size_t oc = 0; oc < output_channels(); oc++) {
                const size_t input_index =
                  (((by * block_size() + bx) * output_channels() + oc) * input_height() + iy) * input_width() + ix;
                const size_t output_index =
                  ((iy * block_size() + by) * output_width() + ix * block_size() + bx) * output_channel_stride() + oc;
                ASSERT_EQ(output[output_index], input[input_index])
                  << "input x: " << ix << " / " << input_width()
                  << ", input y: " << iy << " / " << input_height()
                  << ", block x: " << bx << " / " << block_size()
                  << ", block y: " << by << " / " << block_size()
                  << ", output channel: " << oc << " / " << output_channels()
                  << ", output stride: " << output_channel_stride();
              }
            }
          }
        }
      }
    }
  }

 private:
  size_t output_channels_{1};
  size_t input_height_{1};
  size_t input_width_{1};
  size_t block_size_{2};
  size_t output_channel_stride_{0};
  size_t iterations_{3};
};
