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

  inline DepthToSpaceMicrokernelTester& element_size(size_t element_size) {
    assert(element_size != 0);
    this->element_size_ = element_size;
    return *this;
  }

  inline size_t element_size() const {
    return this->element_size_;
  }

  inline DepthToSpaceMicrokernelTester& input_channel_stride(size_t input_channel_stride) {
    assert(input_channel_stride != 0);
    this->input_channel_stride_ = input_channel_stride;
    return *this;
  }

  inline size_t input_channel_stride() const {
    if (this->input_channel_stride_ != 0) {
      return this->input_channel_stride_;
    } else {
      return this->input_height() * this->input_width() * this->element_size();
    }
  }

  inline DepthToSpaceMicrokernelTester& input_height_stride(size_t input_height_stride) {
    assert(input_height_stride != 0);
    this->input_height_stride_ = input_height_stride;
    return *this;
  }

  inline size_t input_height_stride() const {
    if (this->input_height_stride_ != 0) {
      return this->input_height_stride_;
    } else {
      return this->input_width() * this->element_size();
    }
  }

  inline DepthToSpaceMicrokernelTester& output_height_stride(size_t output_height_stride) {
    assert(output_height_stride != 0);
    this->output_height_stride_ = output_height_stride;
    return *this;
  }

  inline size_t output_height_stride() const {
    if (this->output_height_stride_ != 0) {
      return this->output_height_stride_;
    } else {
      return this->output_width() * this->output_channels() * this->element_size();
    }
  }

  inline DepthToSpaceMicrokernelTester& output_width_stride(size_t output_width_stride) {
    assert(output_width_stride != 0);
    this->output_width_stride_ = output_width_stride;
    return *this;
  }

  inline size_t output_width_stride() const {
    if (this->output_width_stride_ != 0) {
      return this->output_width_stride_;
    } else {
      return this->output_channels() * this->element_size();
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
    ASSERT_EQ(element_size(), sizeof(uint32_t));
    ASSERT_GE(block_size(), 2);
    ASSERT_GE(input_channel_stride(), input_height() * input_height_stride());
    ASSERT_GE(input_height_stride(), input_width() * element_size());
    ASSERT_GE(output_height_stride(), input_width() * block_size() * output_width_stride());
    ASSERT_GE(output_width_stride(), output_channels() * element_size());

    std::random_device random_device;
    auto rng = std::mt19937(random_device());
    auto u32rng = std::bind(std::uniform_int_distribution<uint32_t>(), rng);

    const size_t input_byte_size =
        (input_channels() - 1) * input_channel_stride() +
        (input_height() - 1) * input_height_stride() +
        input_width() * element_size();
    ASSERT_EQ(input_byte_size % element_size(), 0);
    std::vector<uint32_t> input(input_byte_size / element_size());

    const size_t output_byte_size =
        (output_height() - 1) * output_height_stride() +
        (output_width() - 1) * output_width_stride() +
        output_channels() * element_size();
    ASSERT_EQ(output_byte_size % element_size(), 0);
    std::vector<uint32_t> output(output_byte_size / element_size());

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
        input_channel_stride(),
        input_height_stride(),
        output_height_stride(),
        output_width_stride());

      // Verify results.
      for (size_t iy = 0; iy < input_height(); ++iy) {
        for (size_t by = 0; by < block_size(); ++by) {
          for (size_t ix = 0; ix < input_width(); ++ix) {
            for (size_t bx = 0; bx < block_size(); ++bx) {
              for (size_t c = 0; c < output_channels(); ++c) {
                size_t input_offset =
                    (c * block_size() * block_size() + by * block_size() + bx) * input_channel_stride() +
                    iy * input_height_stride() +
                    ix * element_size();
                ASSERT_EQ(input_offset % element_size(), 0);
                ASSERT_LT(input_offset / element_size(), input.size());

                size_t output_offset =
                    (iy * block_size() + by) * output_height_stride() +
                    (ix * block_size() + bx) * output_width_stride() +
                    c * element_size();
                ASSERT_EQ(output_offset % element_size(), 0);
                ASSERT_LT(output_offset / element_size(), output.size());

                ASSERT_EQ(output[output_offset / element_size()],
                          input[input_offset / element_size()])
                    << "iy = " << iy << ", " << "by = " << by << ", "
                    << "ix = " << ix << ", " << "bx = " << bx << ", "
                    << "c = " << c;
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
  size_t element_size_{4};
  size_t input_channel_stride_{0};
  size_t input_height_stride_{0};
  size_t output_height_stride_{0};
  size_t output_width_stride_{0};
  size_t iterations_{3};
};
