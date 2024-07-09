// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <memory>
#include <numeric>
#include <vector>

#include <gtest/gtest.h>
#include "xnnpack.h"

class SpaceToDepthOperatorTester {
 public:
  SpaceToDepthOperatorTester& input_size(size_t input_height, size_t input_width) {
    assert(input_height >= 1);
    assert(input_width >= 1);
    this->input_height_ = input_height;
    this->input_width_ = input_width;
    return *this;
  }

  SpaceToDepthOperatorTester& input_height(size_t input_height) {
    assert(input_height >= 1);
    this->input_height_ = input_height;
    return *this;
  }

  size_t input_height() const {
    return this->input_height_;
  }

  SpaceToDepthOperatorTester& input_width(size_t input_width) {
    assert(input_width >= 1);
    this->input_width_ = input_width;
    return *this;
  }

  size_t input_width() const {
    return this->input_width_;
  }

  size_t output_height() const {
    assert(input_height() % block_size() == 0);
    return input_height() / block_size();
  }

  size_t output_width() const {
    assert(input_width() % block_size() == 0);
    return input_width() / block_size();
  }

  SpaceToDepthOperatorTester& block_size(size_t block_size) {
    assert(block_size >= 2);
    this->block_size_ = block_size;
    return *this;
  }

  size_t block_size() const {
    return this->block_size_;
  }

  SpaceToDepthOperatorTester& input_channels(size_t input_channels) {
    assert(input_channels != 0);
    this->input_channels_ = input_channels;
    return *this;
  }

  size_t input_channels() const {
    return this->input_channels_;
  }

  size_t output_channels() const {
    return input_channels() * block_size() * block_size();
  }

  SpaceToDepthOperatorTester& batch_size(size_t batch_size) {
    assert(batch_size != 0);
    this->batch_size_ = batch_size;
    return *this;
  }

  size_t batch_size() const {
    return this->batch_size_;
  }

  size_t input_channels_stride() const {
    if (this->input_channels_stride_ == 0) {
      return input_channels();
    } else {
      assert(this->input_channels_stride_ >= input_channels());
      return this->input_channels_stride_;
    }
  }

  size_t output_channels_stride() const {
    if (this->output_channels_stride_ == 0) {
      return output_channels();
    } else {
      assert(this->output_channels_stride_ >= output_channels());
      return this->output_channels_stride_;
    }
  }

  SpaceToDepthOperatorTester& iterations(size_t iterations) {
    this->iterations_ = iterations;
    return *this;
  }

  size_t iterations() const {
    return this->iterations_;
  }

  void TestNHWCxX8() const {
    std::vector<int8_t> input(
      (batch_size() * input_height() * input_width() - 1) * input_channels_stride()
      + input_channels() + XNN_EXTRA_BYTES / (sizeof(int8_t)));
    std::vector<int8_t> output(
      (batch_size() * output_height() * output_width() - 1) * output_channels_stride() + output_channels());
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::iota(input.begin(), input.end(), 0);
      std::fill(output.begin(), output.end(), INT8_C(0xAF));

      // Create, setup, run, and destroy Depth To Space operator.
      ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
      xnn_operator_t space_to_depth_op = nullptr;

      ASSERT_EQ(xnn_status_success,
                xnn_create_space_to_depth_nhwc_x8(
                    block_size(), 0, &space_to_depth_op));
      ASSERT_NE(nullptr, space_to_depth_op);

      // Smart pointer to automatically delete space_to_depth_op.
      std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_space_to_depth_op(space_to_depth_op, xnn_delete_operator);

      ASSERT_EQ(xnn_status_success,
                xnn_reshape_space_to_depth_nhwc_x8(
                    space_to_depth_op,
                    batch_size(), input_height(), input_width(), input_channels(),
                    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr, /*output_channels_out=*/nullptr,
                    /*threadpool=*/nullptr));

      ASSERT_EQ(xnn_status_success,
                xnn_setup_space_to_depth_nhwc_x8(
                    space_to_depth_op,
                    input.data(), output.data()));

      ASSERT_EQ(xnn_status_success,
        xnn_run_operator(space_to_depth_op, /*threadpool=*/nullptr));

      // Verify results.
      for (size_t i = 0; i < batch_size(); i++) {
        for (size_t iy = 0; iy < output_height(); iy++) {
          for (size_t ix = 0; ix < output_width(); ix++) {
            for (size_t by = 0; by < block_size(); by++) {
              for (size_t bx = 0; bx < block_size(); bx++) {
                for (size_t oc = 0; oc < input_channels(); oc++) {
                  const size_t input_index = oc
                      + bx * input_channels_stride()
                      + ix * block_size() * input_channels_stride()
                      + by * output_width() * block_size() * input_channels_stride()
                      + iy * block_size() * output_width() * block_size() * input_channels_stride()
                      + i * output_height() * block_size() * output_width() * block_size() * input_channels_stride();
                  const size_t output_index = oc
                      + bx * input_channels()
                      + by * input_channels() * block_size()
                      + ix * output_channels_stride()
                      + iy * output_width() * output_channels_stride()
                      + i * output_height() * output_width() * output_channels_stride();

                  ASSERT_EQ(int32_t(output[output_index]), int32_t(input[input_index]))
                    << "batch: " << i << " / " << batch_size()
                    << ", output x: " << ix << " / " << output_width()
                    << ", output y: " << iy << " / " << output_height()
                    << ", block x: " << bx << " / " << block_size()
                    << ", block y: " << by << " / " << block_size()
                    << ", input channel: " << oc << " / " << input_channels()
                    << ", input stride: " << input_channels_stride()
                    << ", output stride: " << output_channels_stride();
                }
              }
            }
          }
        }
      }
    }
  }

  void TestNHWCxX16() const {
    std::vector<int16_t> input(
      (batch_size() * input_height() * input_width() - 1) * input_channels_stride()
      + input_channels() + XNN_EXTRA_BYTES / (sizeof(int16_t)));
    std::vector<int16_t> output(
      (batch_size() * output_height() * output_width() - 1) * output_channels_stride() + output_channels());
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::iota(input.begin(), input.end(), 0);
      std::fill(output.begin(), output.end(), INT16_C(0xDEAD));

      // Create, setup, run, and destroy Depth To Space operator.
      ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
      xnn_operator_t space_to_depth_op = nullptr;

      ASSERT_EQ(xnn_status_success,
                xnn_create_space_to_depth_nhwc_x16(
                    block_size(), 0, &space_to_depth_op));
      ASSERT_NE(nullptr, space_to_depth_op);

      // Smart pointer to automatically delete space_to_depth_op.
      std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_space_to_depth_op(space_to_depth_op, xnn_delete_operator);

      ASSERT_EQ(xnn_status_success,
                xnn_reshape_space_to_depth_nhwc_x16(
                    space_to_depth_op,
                    batch_size(), input_height(), input_width(), input_channels(),
                    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr, /*output_channels_out=*/nullptr,
                    /*threadpool=*/nullptr));

      ASSERT_EQ(xnn_status_success,
                xnn_setup_space_to_depth_nhwc_x16(
                    space_to_depth_op,
                    input.data(), output.data()));

      ASSERT_EQ(xnn_status_success,
        xnn_run_operator(space_to_depth_op, /*threadpool=*/nullptr));

      // Verify results.
      for (size_t i = 0; i < batch_size(); i++) {
        for (size_t iy = 0; iy < output_height(); iy++) {
          for (size_t ix = 0; ix < output_width(); ix++) {
            for (size_t by = 0; by < block_size(); by++) {
              for (size_t bx = 0; bx < block_size(); bx++) {
                for (size_t oc = 0; oc < input_channels(); oc++) {
                  const size_t input_index = oc
                      + bx * input_channels_stride()
                      + ix * block_size() * input_channels_stride()
                      + by * output_width() * block_size() * input_channels_stride()
                      + iy * block_size() * output_width() * block_size() * input_channels_stride()
                      + i * output_height() * block_size() * output_width() * block_size() * input_channels_stride();
                  const size_t output_index = oc
                      + bx * input_channels()
                      + by * input_channels() * block_size()
                      + ix * output_channels_stride()
                      + iy * output_width() * output_channels_stride()
                      + i * output_height() * output_width() * output_channels_stride();

                  ASSERT_EQ(int32_t(output[output_index]), int32_t(input[input_index]))
                    << "batch: " << i << " / " << batch_size()
                    << ", output x: " << ix << " / " << output_width()
                    << ", output y: " << iy << " / " << output_height()
                    << ", block x: " << bx << " / " << block_size()
                    << ", block y: " << by << " / " << block_size()
                    << ", input channel: " << oc << " / " << input_channels()
                    << ", input stride: " << input_channels_stride()
                    << ", output stride: " << output_channels_stride();
                }
              }
            }
          }
        }
      }
    }
  }

  void TestNHWCxX32() const {
    std::vector<int32_t> input(
      (batch_size() * input_height() * input_width() - 1) * input_channels_stride()
      + input_channels() + XNN_EXTRA_BYTES / (sizeof(int32_t)));
    std::vector<int32_t> output(
      (batch_size() * output_height() * output_width() - 1) * output_channels_stride() + output_channels());
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::iota(input.begin(), input.end(), 0);
      std::fill(output.begin(), output.end(), INT32_C(0xDEADBEEF));

      // Create, setup, run, and destroy Depth To Space operator.
      ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
      xnn_operator_t space_to_depth_op = nullptr;

      ASSERT_EQ(xnn_status_success,
                xnn_create_space_to_depth_nhwc_x32(
                    block_size(), 0, &space_to_depth_op));
      ASSERT_NE(nullptr, space_to_depth_op);

      // Smart pointer to automatically delete space_to_depth_op.
      std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_space_to_depth_op(space_to_depth_op, xnn_delete_operator);

      ASSERT_EQ(xnn_status_success,
                xnn_reshape_space_to_depth_nhwc_x32(
                    space_to_depth_op,
                    batch_size(), input_height(), input_width(), input_channels(),
                    /*output_height_out=*/nullptr, /*output_width_out=*/nullptr, /*output_channels_out=*/nullptr,
                    /*threadpool=*/nullptr));

      ASSERT_EQ(xnn_status_success,
                xnn_setup_space_to_depth_nhwc_x32(
                    space_to_depth_op,
                    input.data(), output.data()));

      ASSERT_EQ(xnn_status_success,
        xnn_run_operator(space_to_depth_op, /*threadpool=*/nullptr));

      // Verify results.
      for (size_t i = 0; i < batch_size(); i++) {
        for (size_t iy = 0; iy < output_height(); iy++) {
          for (size_t ix = 0; ix < output_width(); ix++) {
            for (size_t by = 0; by < block_size(); by++) {
              for (size_t bx = 0; bx < block_size(); bx++) {
                for (size_t oc = 0; oc < input_channels(); oc++) {
                  const size_t input_index = oc
                      + bx * input_channels_stride()
                      + ix * block_size() * input_channels_stride()
                      + by * output_width() * block_size() * input_channels_stride()
                      + iy * block_size() * output_width() * block_size() * input_channels_stride()
                      + i * output_height() * block_size() * output_width() * block_size() * input_channels_stride();
                  const size_t output_index = oc
                      + bx * input_channels()
                      + by * input_channels() * block_size()
                      + ix * output_channels_stride()
                      + iy * output_width() * output_channels_stride()
                      + i * output_height() * output_width() * output_channels_stride();

                  ASSERT_EQ(int32_t(output[output_index]), int32_t(input[input_index]))
                    << "batch: " << i << " / " << batch_size()
                    << ", output x: " << ix << " / " << output_width()
                    << ", output y: " << iy << " / " << output_height()
                    << ", block x: " << bx << " / " << block_size()
                    << ", block y: " << by << " / " << block_size()
                    << ", input channel: " << oc << " / " << input_channels()
                    << ", input stride: " << input_channels_stride()
                    << ", output stride: " << output_channels_stride();
                }
              }
            }
          }
        }
      }
    }
  }

 private:
  size_t input_height_{1};
  size_t input_width_{1};
  size_t input_channels_{1};
  size_t block_size_{2};
  size_t batch_size_{1};
  size_t input_channels_stride_{0};
  size_t output_channels_stride_{0};
  size_t iterations_{1};
};
