// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <functional>
#include <memory>
#include <random>
#include <vector>

#include <gtest/gtest.h>
#include "xnnpack.h"
#include "replicable_random_device.h"

class UnpoolingOperatorTester {
 public:
  UnpoolingOperatorTester& padding(uint32_t padding) {
    this->padding_top_ = padding;
    this->padding_right_ = padding;
    this->padding_bottom_ = padding;
    this->padding_left_ = padding;
    return *this;
  }

  UnpoolingOperatorTester& padding(uint32_t padding_height, uint32_t padding_width) {
    this->padding_top_ = padding_height;
    this->padding_right_ = padding_width;
    this->padding_bottom_ = padding_height;
    this->padding_left_ = padding_width;
    return *this;
  }

  UnpoolingOperatorTester& padding_height(uint32_t padding_height) {
    this->padding_top_ = padding_height;
    this->padding_bottom_ = padding_height;
    return *this;
  }

  UnpoolingOperatorTester& padding_width(uint32_t padding_width) {
    this->padding_right_ = padding_width;
    this->padding_left_ = padding_width;
    return *this;
  }

  UnpoolingOperatorTester& padding_top(uint32_t padding_top) {
    this->padding_top_ = padding_top;
    return *this;
  }

  uint32_t padding_top() const {
    return this->padding_top_;
  }

  UnpoolingOperatorTester& padding_right(uint32_t padding_right) {
    this->padding_right_ = padding_right;
    return *this;
  }

  uint32_t padding_right() const {
    return this->padding_right_;
  }

  UnpoolingOperatorTester& padding_bottom(uint32_t padding_bottom) {
    this->padding_bottom_ = padding_bottom;
    return *this;
  }

  uint32_t padding_bottom() const {
    return this->padding_bottom_;
  }

  UnpoolingOperatorTester& padding_left(uint32_t padding_left) {
    this->padding_left_ = padding_left;
    return *this;
  }

  uint32_t padding_left() const {
    return this->padding_left_;
  }

  UnpoolingOperatorTester& input_size(size_t input_height, size_t input_width) {
    assert(input_height >= 1);
    assert(input_width >= 1);
    this->input_height_ = input_height;
    this->input_width_ = input_width;
    return *this;
  }

  UnpoolingOperatorTester& input_height(size_t input_height) {
    assert(input_height >= 1);
    this->input_height_ = input_height;
    return *this;
  }

  size_t input_height() const {
    return this->input_height_;
  }

  UnpoolingOperatorTester& input_width(size_t input_width) {
    assert(input_width >= 1);
    this->input_width_ = input_width;
    return *this;
  }

  size_t input_width() const {
    return this->input_width_;
  }

  UnpoolingOperatorTester& channels(size_t channels) {
    assert(channels != 0);
    this->channels_ = channels;
    return *this;
  }

  size_t channels() const {
    return this->channels_;
  }

  UnpoolingOperatorTester& batch_size(size_t batch_size) {
    assert(batch_size != 0);
    this->batch_size_ = batch_size;
    return *this;
  }

  size_t batch_size() const {
    return this->batch_size_;
  }

  UnpoolingOperatorTester& pooling_size(uint32_t pooling_size) {
    assert(pooling_size >= 1);
    this->pooling_height_ = pooling_size;
    this->pooling_width_ = pooling_size;
    return *this;
  }

  UnpoolingOperatorTester& pooling_size(uint32_t pooling_height, uint32_t pooling_width) {
    assert(pooling_height >= 1);
    assert(pooling_width >= 1);
    this->pooling_height_ = pooling_height;
    this->pooling_width_ = pooling_width;
    return *this;
  }

  UnpoolingOperatorTester& pooling_height(uint32_t pooling_height) {
    assert(pooling_height >= 1);
    this->pooling_height_ = pooling_height;
    return *this;
  }

  uint32_t pooling_height() const {
    return this->pooling_height_;
  }

  UnpoolingOperatorTester& pooling_width(uint32_t pooling_width) {
    assert(pooling_width >= 1);
    this->pooling_width_ = pooling_width;
    return *this;
  }

  uint32_t pooling_width() const {
    return this->pooling_width_;
  }

  size_t output_height() const {
    const size_t padding_height = padding_top() + padding_bottom();
    return std::max<size_t>(input_height() * pooling_height(), padding_height) - padding_height;
  }

  size_t output_width() const {
    const size_t padding_width = padding_left() + padding_right();
    return std::max<size_t>(input_width() * pooling_width(), padding_width) - padding_width;
  }

  UnpoolingOperatorTester& input_pixel_stride(size_t input_pixel_stride) {
    assert(input_pixel_stride != 0);
    this->input_pixel_stride_ = input_pixel_stride;
    return *this;
  }

  size_t input_pixel_stride() const {
    if (this->input_pixel_stride_ == 0) {
      return channels();
    } else {
      assert(this->input_pixel_stride_ >= channels());
      return this->input_pixel_stride_;
    }
  }

  UnpoolingOperatorTester& output_pixel_stride(size_t output_pixel_stride) {
    assert(output_pixel_stride != 0);
    this->output_pixel_stride_ = output_pixel_stride;
    return *this;
  }

  size_t output_pixel_stride() const {
    if (this->output_pixel_stride_ == 0) {
      return channels();
    } else {
      assert(this->output_pixel_stride_ >= channels());
      return this->output_pixel_stride_;
    }
  }

  UnpoolingOperatorTester& next_input_size(uint32_t next_input_height, uint32_t next_input_width) {
    assert(next_input_height >= 1);
    assert(next_input_width >= 1);
    this->next_input_height_ = next_input_height;
    this->next_input_width_ = next_input_width;
    return *this;
  }

  UnpoolingOperatorTester& next_input_height(uint32_t next_input_height) {
    assert(next_input_height >= 1);
    this->next_input_height_ = next_input_height;
    return *this;
  }

  uint32_t next_input_height() const {
    if (this->next_input_height_ == 0) {
      return input_height();
    } else {
      return this->next_input_height_;
    }
  }

  UnpoolingOperatorTester& next_input_width(uint32_t next_input_width) {
    assert(next_input_width >= 1);
    this->next_input_width_ = next_input_width;
    return *this;
  }

  uint32_t next_input_width() const {
    if (this->next_input_width_ == 0) {
      return input_width();
    } else {
      return this->next_input_width_;
    }
  }

  size_t next_output_height() const {
    const size_t padding_height = padding_top() + padding_bottom();
    return std::max<size_t>(next_input_height() * pooling_height(), padding_height) - padding_height;
  }

  size_t next_output_width() const {
    const size_t padding_width = padding_left() + padding_right();
    return std::max<size_t>(next_input_width() * pooling_width(), padding_width) - padding_width;
  }

  UnpoolingOperatorTester& next_batch_size(size_t next_batch_size) {
    assert(next_batch_size >= 1);
    this->next_batch_size_ = next_batch_size;
    return *this;
  }

  size_t next_batch_size() const {
    if (this->next_batch_size_ == 0) {
      return batch_size();
    } else {
      return this->next_batch_size_;
    }
  }

  UnpoolingOperatorTester& iterations(size_t iterations) {
    this->iterations_ = iterations;
    return *this;
  }

  size_t iterations() const {
    return this->iterations_;
  }

  void TestX32() const {
    xnnpack::ReplicableRandomDevice rng;
    auto u32rng = std::bind(std::uniform_int_distribution<uint32_t>(), std::ref(rng));
    auto idx_rng = std::bind(std::uniform_int_distribution<uint32_t>(0, pooling_height() * pooling_width() - 1), std::ref(rng));

    std::vector<uint32_t> input((batch_size() * input_height() * input_width() - 1) * input_pixel_stride() + channels());
    std::vector<uint32_t> index(batch_size() * input_height() * input_width() * channels());
    std::vector<uint32_t> output((batch_size() * output_height() * output_width() - 1) * output_pixel_stride() + channels());
    std::vector<uint32_t> output_ref(batch_size() * output_height() * output_width() * channels());
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(input.begin(), input.end(), std::ref(u32rng));
      std::generate(index.begin(), index.end(), std::ref(idx_rng));
      std::generate(output.begin(), output.end(), std::ref(u32rng));

      // Compute reference results.
      std::fill(output_ref.begin(), output_ref.end(), 0);
      for (size_t i = 0; i < batch_size(); i++) {
        for (size_t iy = 0; iy < input_height(); iy++) {
          for (size_t ix = 0; ix < input_width(); ix++) {
            for (size_t c = 0; c < channels(); c++) {
              const uint32_t pooling_index = index[((i * input_height() + iy) * input_width() + ix) * channels() + c];
              const uint32_t py = pooling_index % pooling_height();
              const uint32_t px = pooling_index / pooling_height();
              const size_t oy = std::min<size_t>(std::max<size_t>(iy * pooling_height() + py, padding_top()) - padding_top(), output_height() - 1);
              const size_t ox = std::min<size_t>(std::max<size_t>(ix * pooling_width() + px, padding_left()) - padding_left(), output_width() - 1);
              output_ref[((i * output_height() + oy) * output_width() + ox) * channels() + c] =
                input[((i * input_height() + iy) * input_width() + ix) * input_pixel_stride() + c];
            }
          }
        }
      }

      // Create, setup, run, and destroy Unpooling operator.
      ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
      xnn_operator_t unpooling_op = nullptr;

      ASSERT_EQ(xnn_status_success,
        xnn_create_unpooling2d_nhwc_x32(
          padding_top(), padding_right(), padding_bottom(), padding_left(),
          pooling_height(), pooling_width(),
          channels(), input_pixel_stride(), output_pixel_stride(),
          0, &unpooling_op));
      ASSERT_NE(nullptr, unpooling_op);

      // Smart pointer to automatically delete unpooling_op.
      std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_unpooling_op(unpooling_op, xnn_delete_operator);

      ASSERT_EQ(xnn_status_success,
        xnn_reshape_unpooling2d_nhwc_x32(
          unpooling_op,
          batch_size(), input_height(), input_width(),
          /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
          /*threadpool=*/nullptr));

      ASSERT_EQ(xnn_status_success,
        xnn_setup_unpooling2d_nhwc_x32(
          unpooling_op,
          input.data(), index.data(), output.data()));

      ASSERT_EQ(xnn_status_success,
        xnn_run_operator(unpooling_op, /*threadpool=*/nullptr));

      // Verify results.
      for (size_t i = 0; i < batch_size(); i++) {
        for (size_t c = 0; c < channels(); c++) {
          for (size_t y = 0; y < output_height(); y++) {
            for (size_t x = 0; x < output_width(); x++) {
              EXPECT_EQ(output_ref[((i * output_height() + y) * output_width() + x) * channels() + c],
                output[((i * output_height() + y) * output_width() + x) * output_pixel_stride() + c]) <<
                "in batch index " << i << ", pixel (" << y << ", " << x << "), channel " << c;
            }
          }
        }
      }
    }
  }

  void TestSetupX32() const {
    xnnpack::ReplicableRandomDevice rng;
    auto u32rng = std::bind(std::uniform_int_distribution<uint32_t>(), std::ref(rng));
    auto idx_rng = std::bind(std::uniform_int_distribution<uint32_t>(0, pooling_height() * pooling_width() - 1), std::ref(rng));

    std::vector<uint32_t> input(std::max<size_t>(
      (batch_size() * input_height() * input_width() - 1) * input_pixel_stride() + channels(),
      (next_batch_size() * next_input_height() * next_input_width() - 1) * input_pixel_stride() + channels()));
    std::vector<uint32_t> index(std::max<size_t>(
      batch_size() * input_height() * input_width() * channels(),
      next_batch_size() * next_input_height() * next_input_width() * channels()));
    std::vector<uint32_t> output(std::max<size_t>(
      (batch_size() * output_height() * output_width() - 1) * output_pixel_stride() + channels(),
      (next_batch_size() * next_output_height() * next_output_width() - 1) * output_pixel_stride() * channels()));
    std::vector<uint32_t> output_ref(batch_size() * output_height() * output_width() * channels());
    std::vector<uint32_t> next_output_ref(next_batch_size() * next_output_height() * next_output_width() * channels());

    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(input.begin(), input.end(), std::ref(u32rng));
      std::generate(index.begin(), index.end(), std::ref(idx_rng));
      std::generate(output.begin(), output.end(), std::ref(u32rng));

      // Compute reference results.
      std::fill(output_ref.begin(), output_ref.end(), 0);
      for (size_t i = 0; i < batch_size(); i++) {
        for (size_t iy = 0; iy < input_height(); iy++) {
          for (size_t ix = 0; ix < input_width(); ix++) {
            for (size_t c = 0; c < channels(); c++) {
              const uint32_t pooling_index = index[((i * input_height() + iy) * input_width() + ix) * channels() + c];
              const uint32_t py = pooling_index % pooling_height();
              const uint32_t px = pooling_index / pooling_height();
              const size_t oy = std::min<size_t>(std::max<size_t>(iy * pooling_height() + py, padding_top()) - padding_top(), output_height() - 1);
              const size_t ox = std::min<size_t>(std::max<size_t>(ix * pooling_width() + px, padding_left()) - padding_left(), output_width() - 1);
              output_ref[((i * output_height() + oy) * output_width() + ox) * channels() + c] =
                input[((i * input_height() + iy) * input_width() + ix) * input_pixel_stride() + c];
            }
          }
        }
      }

      // Create, setup, and run Unpooling operator once.
      ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
      xnn_operator_t unpooling_op = nullptr;

      ASSERT_EQ(xnn_status_success,
        xnn_create_unpooling2d_nhwc_x32(
          padding_top(), padding_right(), padding_bottom(), padding_left(),
          pooling_height(), pooling_width(),
          channels(), input_pixel_stride(), output_pixel_stride(),
          0, &unpooling_op));
      ASSERT_NE(nullptr, unpooling_op);

      // Smart pointer to automatically delete unpooling_op.
      std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_unpooling_op(unpooling_op, xnn_delete_operator);

      ASSERT_EQ(xnn_status_success,
        xnn_reshape_unpooling2d_nhwc_x32(
          unpooling_op,
          batch_size(), input_height(), input_width(),
          /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
          /*threadpool=*/nullptr));

      ASSERT_EQ(xnn_status_success,
        xnn_setup_unpooling2d_nhwc_x32(
          unpooling_op,
          input.data(), index.data(), output.data()));

      ASSERT_EQ(xnn_status_success,
        xnn_run_operator(unpooling_op, /*threadpool=*/nullptr));

      // Verify results of the first run.
      for (size_t i = 0; i < batch_size(); i++) {
        for (size_t c = 0; c < channels(); c++) {
          for (size_t y = 0; y < output_height(); y++) {
            for (size_t x = 0; x < output_width(); x++) {
              EXPECT_EQ(output_ref[((i * output_height() + y) * output_width() + x) * channels() + c],
                output[((i * output_height() + y) * output_width() + x) * output_pixel_stride() + c]) <<
                "in batch index " << i << ", pixel (" << y << ", " << x << "), channel " << c;
            }
          }
        }
      }

      // Re-generate data for the second run.
      std::generate(input.begin(), input.end(), std::ref(u32rng));
      std::generate(index.begin(), index.end(), std::ref(idx_rng));
      std::generate(output.begin(), output.end(), std::ref(u32rng));

      // Compute reference results for the second run, including clamping.
      std::fill(next_output_ref.begin(), next_output_ref.end(), 0);
      for (size_t i = 0; i < next_batch_size(); i++) {
        for (size_t iy = 0; iy < next_input_height(); iy++) {
          for (size_t ix = 0; ix < next_input_width(); ix++) {
            for (size_t c = 0; c < channels(); c++) {
              const uint32_t pooling_index = index[((i * next_input_height() + iy) * next_input_width() + ix) * channels() + c];
              const uint32_t py = pooling_index % pooling_height();
              const uint32_t px = pooling_index / pooling_height();
              const size_t oy = std::min<size_t>(std::max<size_t>(iy * pooling_height() + py, padding_top()) - padding_top(), next_output_height() - 1);
              const size_t ox = std::min<size_t>(std::max<size_t>(ix * pooling_width() + px, padding_left()) - padding_left(), next_output_width() - 1);
              next_output_ref[((i * next_output_height() + oy) * next_output_width() + ox) * channels() + c] =
                input[((i * next_input_height() + iy) * next_input_width() + ix) * input_pixel_stride() + c];
            }
          }
        }
      }

      // Setup and run Max Pooling operator the second time, and destroy the operator.
      ASSERT_EQ(xnn_status_success,
        xnn_reshape_unpooling2d_nhwc_x32(
          unpooling_op,
          next_batch_size(), next_input_height(), next_input_width(),
          /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
          /*threadpool=*/nullptr));

      ASSERT_EQ(xnn_status_success,
        xnn_setup_unpooling2d_nhwc_x32(
          unpooling_op,
          input.data(), index.data(), output.data()));

      ASSERT_EQ(xnn_status_success,
        xnn_run_operator(unpooling_op, /*threadpool=*/nullptr));

      // Verify results of the second run.
      for (size_t i = 0; i < next_batch_size(); i++) {
        for (size_t c = 0; c < channels(); c++) {
          for (size_t y = 0; y < next_output_height(); y++) {
            for (size_t x = 0; x < next_output_width(); x++) {
              EXPECT_EQ(next_output_ref[((i * next_output_height() + y) * next_output_width() + x) * channels() + c],
                output[((i * next_output_height() + y) * next_output_width() + x) * output_pixel_stride() + c]) <<
                "in batch index " << i << ", pixel (" << y << ", " << x << "), channel " << c;
            }
          }
        }
      }
    }
  }

 private:
  uint32_t padding_top_{0};
  uint32_t padding_right_{0};
  uint32_t padding_bottom_{0};
  uint32_t padding_left_{0};
  size_t input_height_{1};
  size_t input_width_{1};
  size_t channels_{1};
  size_t batch_size_{1};
  size_t input_pixel_stride_{0};
  size_t output_pixel_stride_{0};
  uint32_t pooling_height_{1};
  uint32_t pooling_width_{1};
  size_t next_input_height_{0};
  size_t next_input_width_{0};
  size_t next_batch_size_{0};
  size_t iterations_{1};
};
