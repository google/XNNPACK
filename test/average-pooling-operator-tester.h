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
#include <cmath>
#include <cassert>
#include <cstddef>
#include <cstdlib>
#include <functional>
#include <limits>
#include <random>
#include <vector>

#include <xnnpack.h>


class AveragePoolingOperatorTester {
 public:
  inline AveragePoolingOperatorTester& padding_tf_same(bool padding_same) {
    if (padding_same) {
      assert(padding_top() == 0);
      assert(padding_left() == 0);
      assert(padding_bottom() == 0);
      assert(padding_right() == 0);
    }
    this->padding_tf_same_ = padding_same;
    return *this;
  }

  inline bool padding_tf_same() const {
    return this->padding_tf_same_;
  }

  inline AveragePoolingOperatorTester& padding(uint32_t padding) {
    assert(!padding_tf_same());
    this->padding_top_ = padding;
    this->padding_right_ = padding;
    this->padding_bottom_ = padding;
    this->padding_left_ = padding;
    return *this;
  }

  inline AveragePoolingOperatorTester& padding(uint32_t padding_height, uint32_t padding_width) {
    assert(!padding_tf_same());
    this->padding_top_ = padding_height;
    this->padding_right_ = padding_width;
    this->padding_bottom_ = padding_height;
    this->padding_left_ = padding_width;
    return *this;
  }

  inline AveragePoolingOperatorTester& padding_height(uint32_t padding_height) {
    assert(!padding_tf_same());
    this->padding_top_ = padding_height;
    this->padding_bottom_ = padding_height;
    return *this;
  }

  inline AveragePoolingOperatorTester& padding_width(uint32_t padding_width) {
    assert(!padding_tf_same());
    this->padding_right_ = padding_width;
    this->padding_left_ = padding_width;
    return *this;
  }

  inline AveragePoolingOperatorTester& padding_top(uint32_t padding_top) {
    assert(!padding_tf_same());
    this->padding_top_ = padding_top;
    return *this;
  }

  inline uint32_t padding_top() const {
    if (padding_tf_same()) {
      const uint32_t total_padding_height =
        (output_height() - 1) * stride_height() + pooling_height() - input_height();
      return total_padding_height / 2;
    } else {
      return this->padding_top_;
    }
  }

  inline AveragePoolingOperatorTester& padding_left(uint32_t padding_left) {
    assert(!padding_tf_same());
    this->padding_left_ = padding_left;
    return *this;
  }

  inline uint32_t padding_left() const {
    if (padding_tf_same()) {
      const uint32_t total_padding_width =
        (output_width() - 1) * stride_width() + pooling_width() - input_width();
      return total_padding_width / 2;
    } else {
      return this->padding_left_;
    }
  }

  inline AveragePoolingOperatorTester& padding_bottom(uint32_t padding_bottom) {
    assert(!padding_tf_same());
    this->padding_bottom_ = padding_bottom;
    return *this;
  }

  inline uint32_t padding_bottom() const {
    if (padding_tf_same()) {
      const uint32_t total_padding_height =
        (output_height() - 1) * stride_height() + pooling_height() - input_height();
      return total_padding_height - total_padding_height / 2;
    } else {
      return this->padding_bottom_;
    }
  }

  inline AveragePoolingOperatorTester& padding_right(uint32_t padding_right) {
    assert(!padding_tf_same());
    this->padding_right_ = padding_right;
    return *this;
  }

  inline uint32_t padding_right() const {
    if (padding_tf_same()) {
      const uint32_t total_padding_width =
        (output_width() - 1) * stride_width() + pooling_width() - input_width();
      return total_padding_width - total_padding_width / 2;
    } else {
      return this->padding_right_;
    }
  }

  inline AveragePoolingOperatorTester& input_size(size_t input_height, size_t input_width) {
    assert(input_height >= 1);
    assert(input_width >= 1);
    this->input_height_ = input_height;
    this->input_width_ = input_width;
    return *this;
  }

  inline AveragePoolingOperatorTester& input_height(size_t input_height) {
    assert(input_height >= 1);
    this->input_height_ = input_height;
    return *this;
  }

  inline size_t input_height() const {
    return this->input_height_;
  }

  inline AveragePoolingOperatorTester& input_width(size_t input_width) {
    assert(input_width >= 1);
    this->input_width_ = input_width;
    return *this;
  }

  inline size_t input_width() const {
    return this->input_width_;
  }

  inline AveragePoolingOperatorTester& channels(size_t channels) {
    assert(channels != 0);
    this->channels_ = channels;
    return *this;
  }

  inline size_t channels() const {
    return this->channels_;
  }

  inline AveragePoolingOperatorTester& batch_size(size_t batch_size) {
    assert(batch_size != 0);
    this->batch_size_ = batch_size;
    return *this;
  }

  inline size_t batch_size() const {
    return this->batch_size_;
  }

  inline AveragePoolingOperatorTester& pooling_size(uint32_t pooling_size) {
    assert(pooling_size >= 1);
    this->pooling_height_ = pooling_size;
    this->pooling_width_ = pooling_size;
    return *this;
  }

  inline AveragePoolingOperatorTester& pooling_size(uint32_t pooling_height, uint32_t pooling_width) {
    assert(pooling_height >= 1);
    assert(pooling_width >= 1);
    this->pooling_height_ = pooling_height;
    this->pooling_width_ = pooling_width;
    return *this;
  }

  inline AveragePoolingOperatorTester& pooling_height(uint32_t pooling_height) {
    assert(pooling_height >= 1);
    this->pooling_height_ = pooling_height;
    return *this;
  }

  inline uint32_t pooling_height() const {
    return this->pooling_height_;
  }

  inline AveragePoolingOperatorTester& pooling_width(uint32_t pooling_width) {
    assert(pooling_width >= 1);
    this->pooling_width_ = pooling_width;
    return *this;
  }

  inline uint32_t pooling_width() const {
    return this->pooling_width_;
  }

  inline AveragePoolingOperatorTester& stride(uint32_t stride) {
    assert(stride >= 1);
    this->stride_height_ = stride;
    this->stride_width_ = stride;
    return *this;
  }

  inline AveragePoolingOperatorTester& stride(uint32_t stride_height, uint32_t stride_width) {
    assert(stride_height >= 1);
    assert(stride_width >= 1);
    this->stride_height_ = stride_height;
    this->stride_width_ = stride_width;
    return *this;
  }

  inline AveragePoolingOperatorTester& stride_height(uint32_t stride_height) {
    assert(stride_height >= 1);
    this->stride_height_ = stride_height;
    return *this;
  }

  inline uint32_t stride_height() const {
    return this->stride_height_;
  }

  inline AveragePoolingOperatorTester& stride_width(uint32_t stride_width) {
    assert(stride_width >= 1);
    this->stride_width_ = stride_width;
    return *this;
  }

  inline uint32_t stride_width() const {
    return this->stride_width_;
  }

  inline size_t output_height() const {
    if (padding_tf_same()) {
      return (input_height() + stride_height() - 1) / stride_height();
    } else {
      const size_t padded_input_height = padding_top() + input_height() + padding_bottom();
      if (padded_input_height <= pooling_height()) {
        return 1;
      } else {
        return (padded_input_height - pooling_height()) / stride_height() + 1;
      }
    }
  }

  inline size_t output_width() const {
    if (padding_tf_same()) {
      return (input_width() + stride_width() - 1) / stride_width();
    } else {
      const size_t padded_input_width = padding_left() + input_width() + padding_right();
      if (padded_input_width <= pooling_width()) {
        return 1;
      } else {
        return (padded_input_width - pooling_width()) / stride_width() + 1;
      }
    }
  }

  inline AveragePoolingOperatorTester& input_pixel_stride(size_t input_pixel_stride) {
    assert(input_pixel_stride != 0);
    this->input_pixel_stride_ = input_pixel_stride;
    return *this;
  }

  inline size_t input_pixel_stride() const {
    if (this->input_pixel_stride_ == 0) {
      return channels();
    } else {
      assert(this->input_pixel_stride_ >= channels());
      return this->input_pixel_stride_;
    }
  }

  inline AveragePoolingOperatorTester& output_pixel_stride(size_t output_pixel_stride) {
    assert(output_pixel_stride != 0);
    this->output_pixel_stride_ = output_pixel_stride;
    return *this;
  }

  inline size_t output_pixel_stride() const {
    if (this->output_pixel_stride_ == 0) {
      return channels();
    } else {
      assert(this->output_pixel_stride_ >= channels());
      return this->output_pixel_stride_;
    }
  }

  inline AveragePoolingOperatorTester& next_input_size(uint32_t next_input_height, uint32_t next_input_width) {
    assert(next_input_height >= 1);
    assert(next_input_width >= 1);
    this->next_input_height_ = next_input_height;
    this->next_input_width_ = next_input_width;
    return *this;
  }

  inline AveragePoolingOperatorTester& next_input_height(uint32_t next_input_height) {
    assert(next_input_height >= 1);
    this->next_input_height_ = next_input_height;
    return *this;
  }

  inline uint32_t next_input_height() const {
    if (this->next_input_height_ == 0) {
      return input_height();
    } else {
      return this->next_input_height_;
    }
  }

  inline AveragePoolingOperatorTester& next_input_width(uint32_t next_input_width) {
    assert(next_input_width >= 1);
    this->next_input_width_ = next_input_width;
    return *this;
  }

  inline uint32_t next_input_width() const {
    if (this->next_input_width_ == 0) {
      return input_width();
    } else {
      return this->next_input_width_;
    }
  }

  inline size_t next_output_height() const {
    const size_t padded_next_input_height = padding_top() + next_input_height() + padding_bottom();
    if (padded_next_input_height <= pooling_height()) {
      return 1;
    } else {
      return (padded_next_input_height - pooling_height()) / stride_height() + 1;
    }
  }

  inline size_t next_output_width() const {
    const size_t padded_next_input_width = padding_left() + next_input_width() + padding_right();
    if (padded_next_input_width <= pooling_width()) {
      return 1;
    } else {
      return (padded_next_input_width - pooling_width()) / stride_width() + 1;
    }
  }

  inline AveragePoolingOperatorTester& next_batch_size(size_t next_batch_size) {
    assert(next_batch_size >= 1);
    this->next_batch_size_ = next_batch_size;
    return *this;
  }

  inline size_t next_batch_size() const {
    if (this->next_batch_size_ == 0) {
      return batch_size();
    } else {
      return this->next_batch_size_;
    }
  }

  inline AveragePoolingOperatorTester& input_scale(float input_scale) {
    assert(input_scale > 0.0f);
    assert(std::isnormal(input_scale));
    this->input_scale_ = input_scale;
    return *this;
  }

  inline float input_scale() const {
    return this->input_scale_;
  }

  inline AveragePoolingOperatorTester& input_zero_point(uint8_t input_zero_point) {
    this->input_zero_point_ = input_zero_point;
    return *this;
  }

  inline uint8_t input_zero_point() const {
    return this->input_zero_point_;
  }

  inline AveragePoolingOperatorTester& output_scale(float output_scale) {
    assert(output_scale > 0.0f);
    assert(std::isnormal(output_scale));
    this->output_scale_ = output_scale;
    return *this;
  }

  inline float output_scale() const {
    return this->output_scale_;
  }

  inline AveragePoolingOperatorTester& output_zero_point(uint8_t output_zero_point) {
    this->output_zero_point_ = output_zero_point;
    return *this;
  }

  inline uint8_t output_zero_point() const {
    return this->output_zero_point_;
  }

  inline AveragePoolingOperatorTester& qmin(uint8_t qmin) {
    this->qmin_ = qmin;
    return *this;
  }

  inline uint8_t qmin() const {
    return this->qmin_;
  }

  inline AveragePoolingOperatorTester& qmax(uint8_t qmax) {
    this->qmax_ = qmax;
    return *this;
  }

  inline uint8_t qmax() const {
    return this->qmax_;
  }

  inline AveragePoolingOperatorTester& iterations(size_t iterations) {
    this->iterations_ = iterations;
    return *this;
  }

  inline size_t iterations() const {
    return this->iterations_;
  }

  void TestQ8() const {
    std::random_device random_device;
    auto rng = std::mt19937(random_device());
    auto u8rng = std::bind(std::uniform_int_distribution<uint32_t>(0, std::numeric_limits<uint8_t>::max()), rng);

    std::vector<uint8_t> input((batch_size() * input_height() * input_width() - 1) * input_pixel_stride() + channels() + XNN_EXTRA_BYTES / sizeof(uint8_t));
    std::vector<uint8_t> output((batch_size() * output_height() * output_width() - 1) * output_pixel_stride() + channels());
    std::vector<float> output_ref(batch_size() * output_height() * output_width() * channels());
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(input.begin(), input.end(), std::ref(u8rng));
      std::fill(output.begin(), output.end(), 0xA5);

      // Compute reference results.
      const double scale = double(input_scale()) / (double(output_scale()) * double(pooling_height() * pooling_width()));
      for (size_t i = 0; i < batch_size(); i++) {
        for (size_t oy = 0; oy < output_height(); oy++) {
          for (size_t ox = 0; ox < output_width(); ox++) {
            for (size_t c = 0; c < channels(); c++) {
              double acc = 0.0f;
              for (size_t py = 0; py < pooling_height(); py++) {
                const size_t iy = oy * stride_height() + py - padding_top();
                for (size_t px = 0; px < pooling_width(); px++) {
                  const size_t ix = ox * stride_width() + px - padding_left();
                  if (ix < input_width() && iy < input_height()) {
                    acc += double(int32_t(input[((i * input_height() + iy) * input_width() + ix) * input_pixel_stride() + c]) - int32_t(input_zero_point()));
                  }
                }
              }
              output_ref[((i * output_height() + oy) * output_width() + ox) * channels() + c] = float(acc * scale + double(output_zero_point()));
              output_ref[((i * output_height() + oy) * output_width() + ox) * channels() + c] =
                std::min<float>(output_ref[((i * output_height() + oy) * output_width() + ox) * channels() + c], float(qmax()));
              output_ref[((i * output_height() + oy) * output_width() + ox) * channels() + c] =
                std::max<float>(output_ref[((i * output_height() + oy) * output_width() + ox) * channels() + c], float(qmin()));
            }
          }
        }
      }

      // Create, setup, run, and destroy Average Pooling operator.
      ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
      xnn_operator_t average_pooling_op = nullptr;

      ASSERT_EQ(xnn_status_success,
        xnn_create_average_pooling2d_nhwc_q8(
          padding_top(), padding_right(), padding_bottom(), padding_left(),
          pooling_height(), pooling_width(),
          stride_height(), stride_width(),
          channels(), input_pixel_stride(), output_pixel_stride(),
          input_zero_point(), input_scale(),
          output_zero_point(), output_scale(),
          qmin(), qmax(),
          0, &average_pooling_op));
      ASSERT_NE(nullptr, average_pooling_op);

      // Smart pointer to automatically delete average_pooling_op.
      std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_average_pooling_op(average_pooling_op, xnn_delete_operator);

      ASSERT_EQ(xnn_status_success,
        xnn_setup_average_pooling2d_nhwc_q8(
          average_pooling_op,
          batch_size(), input_height(), input_width(),
          input.data(), output.data(),
          nullptr /* thread pool */));

      ASSERT_EQ(xnn_status_success,
        xnn_run_operator(average_pooling_op, nullptr /* thread pool */));

      // Verify results.
      for (size_t i = 0; i < batch_size(); i++) {
        for (size_t y = 0; y < output_height(); y++) {
          for (size_t x = 0; x < output_width(); x++) {
            for (size_t c = 0; c < channels(); c++) {
              ASSERT_LE(uint32_t(output[((i * output_height() + y) * output_width() + x) * output_pixel_stride() + c]), uint32_t(qmax()));
              ASSERT_GE(uint32_t(output[((i * output_height() + y) * output_width() + x) * output_pixel_stride() + c]), uint32_t(qmin()));
              ASSERT_NEAR(float(int32_t(output[((i * output_height() + y) * output_width() + x) * output_pixel_stride() + c])),
                output_ref[((i * output_height() + y) * output_width() + x) * channels() + c], 0.80f) <<
                "in batch index " << i << ", pixel (" << y << ", " << x << "), channel " << c;
            }
          }
        }
      }
    }
  }

  void TestF32() const {
    std::random_device random_device;
    auto rng = std::mt19937(random_device());
    auto f32rng = std::bind(std::uniform_real_distribution<float>(), rng);

    std::vector<float> input((batch_size() * input_height() * input_width() - 1) * input_pixel_stride() + channels() + XNN_EXTRA_BYTES / sizeof(float));
    std::vector<float> output((batch_size() * output_height() * output_width() - 1) * output_pixel_stride() + channels());
    std::vector<float> output_ref(batch_size() * output_height() * output_width() * channels());
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(input.begin(), input.end(), std::ref(f32rng));
      std::fill(output.begin(), output.end(), std::nanf(""));

      // Compute reference results, without clamping.
      for (size_t i = 0; i < batch_size(); i++) {
        for (size_t oy = 0; oy < output_height(); oy++) {
          for (size_t ox = 0; ox < output_width(); ox++) {
            for (size_t c = 0; c < channels(); c++) {
              float acc = 0.0f;
              int32_t n = 0;
              for (size_t py = 0; py < pooling_height(); py++) {
                const size_t iy = oy * stride_height() + py - padding_top();
                for (size_t px = 0; px < pooling_width(); px++) {
                  const size_t ix = ox * stride_width() + px - padding_left();
                  if (ix < input_width() && iy < input_height()) {
                    acc += input[((i * input_height() + iy) * input_width() + ix) * input_pixel_stride() + c];
                    n += 1;
                  }
                }
              }
              output_ref[((i * output_height() + oy) * output_width() + ox) * channels() + c] = acc / float(n);
            }
          }
        }
      }

      // Compute clamping parameters.
      const float accumulated_min = *std::min_element(output_ref.cbegin(), output_ref.cend());
      const float accumulated_max = *std::max_element(output_ref.cbegin(), output_ref.cend());
      const float accumulated_range = accumulated_max - accumulated_min;
      const float output_min = accumulated_range == 0.0f ?
        -std::numeric_limits<float>::infinity() :
        accumulated_min + accumulated_range / 255.0f * float(qmin());
      const float output_max = accumulated_range == 0.0f ?
        +std::numeric_limits<float>::infinity() :
        accumulated_max - accumulated_range / 255.0f * float(255 - qmax());

      // Clamp reference results.
      for (float& value : output_ref) {
        value = std::max(std::min(value, output_max), output_min);
      }

      // Create, setup, run, and destroy Average Pooling operator.
      ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
      xnn_operator_t average_pooling_op = nullptr;

      ASSERT_EQ(xnn_status_success,
        xnn_create_average_pooling2d_nhwc_f32(
          padding_top(), padding_right(), padding_bottom(), padding_left(),
          pooling_height(), pooling_width(),
          stride_height(), stride_width(),
          channels(), input_pixel_stride(), output_pixel_stride(),
          output_min, output_max,
          0, &average_pooling_op));
      ASSERT_NE(nullptr, average_pooling_op);

      // Smart pointer to automatically delete average_pooling_op.
      std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_average_pooling_op(average_pooling_op, xnn_delete_operator);

      ASSERT_EQ(xnn_status_success,
        xnn_setup_average_pooling2d_nhwc_f32(
          average_pooling_op,
          batch_size(), input_height(), input_width(),
          input.data(), output.data(),
          nullptr /* thread pool */));

      ASSERT_EQ(xnn_status_success,
        xnn_run_operator(average_pooling_op, nullptr /* thread pool */));

      // Verify results.
      for (size_t i = 0; i < batch_size(); i++) {
        for (size_t y = 0; y < output_height(); y++) {
          for (size_t x = 0; x < output_width(); x++) {
            for (size_t c = 0; c < channels(); c++) {
              ASSERT_LE(output[((i * output_height() + y) * output_width() + x) * output_pixel_stride() + c], output_max);
              ASSERT_GE(output[((i * output_height() + y) * output_width() + x) * output_pixel_stride() + c], output_min);
              ASSERT_NEAR(output[((i * output_height() + y) * output_width() + x) * output_pixel_stride() + c],
                  output_ref[((i * output_height() + y) * output_width() + x) * channels() + c],
                  std::abs(output_ref[((i * output_height() + y) * output_width() + x) * channels() + c]) * 1.0e-6f) <<
                "in batch index " << i << ", pixel (" << y << ", " << x << "), channel " << c;
            }
          }
        }
      }
    }
  }

  void TestSetupQ8() const {
    std::random_device random_device;
    auto rng = std::mt19937(random_device());
    auto u8rng = std::bind(std::uniform_int_distribution<uint32_t>(0, std::numeric_limits<uint8_t>::max()), rng);

    std::vector<uint8_t> input(XNN_EXTRA_BYTES / sizeof(uint8_t) + std::max(
      (batch_size() * input_height() * input_width() - 1) * input_pixel_stride() + channels(),
      (next_batch_size() * next_input_height() * next_input_width() - 1) * input_pixel_stride() + channels()));
    std::vector<uint8_t> output(std::max(
      (batch_size() * output_height() * output_width() - 1) * output_pixel_stride() + channels(),
      (next_batch_size() * next_output_height() * next_output_width() - 1) * output_pixel_stride() + channels()));
    std::vector<float> output_ref(batch_size() * output_height() * output_width() * channels());
    std::vector<float> next_output_ref(next_batch_size() * next_output_height() * next_output_width() * channels());
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(input.begin(), input.end(), std::ref(u8rng));
      std::fill(output.begin(), output.end(), 0xA5);

      // Compute reference results.
      const double scale = double(input_scale()) / (double(output_scale()) * double(pooling_height() * pooling_width()));
      for (size_t i = 0; i < batch_size(); i++) {
        for (size_t oy = 0; oy < output_height(); oy++) {
          for (size_t ox = 0; ox < output_width(); ox++) {
            for (size_t c = 0; c < channels(); c++) {
              double acc = 0.0f;
              for (size_t py = 0; py < pooling_height(); py++) {
                const size_t iy = oy * stride_height() + py - padding_top();
                for (size_t px = 0; px < pooling_width(); px++) {
                  const size_t ix = ox * stride_width() + px - padding_left();
                  if (ix < input_width() && iy < input_height()) {
                    acc += double(int32_t(input[((i * input_height() + iy) * input_width() + ix) * input_pixel_stride() + c]) - int32_t(input_zero_point()));
                  }
                }
              }
              output_ref[((i * output_height() + oy) * output_width() + ox) * channels() + c] = float(acc * scale + double(output_zero_point()));
              output_ref[((i * output_height() + oy) * output_width() + ox) * channels() + c] =
                std::min<float>(output_ref[((i * output_height() + oy) * output_width() + ox) * channels() + c], float(qmax()));
              output_ref[((i * output_height() + oy) * output_width() + ox) * channels() + c] =
                std::max<float>(output_ref[((i * output_height() + oy) * output_width() + ox) * channels() + c], float(qmin()));
            }
          }
        }
      }

      // Create, setup, and run Average Pooling operator once.
      ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
      xnn_operator_t average_pooling_op = nullptr;

      ASSERT_EQ(xnn_status_success,
        xnn_create_average_pooling2d_nhwc_q8(
          padding_top(), padding_right(), padding_bottom(), padding_left(),
          pooling_height(), pooling_width(),
          stride_height(), stride_width(),
          channels(), input_pixel_stride(), output_pixel_stride(),
          input_zero_point(), input_scale(),
          output_zero_point(), output_scale(),
          qmin(), qmax(),
          0, &average_pooling_op));
      ASSERT_NE(nullptr, average_pooling_op);

      ASSERT_EQ(xnn_status_success,
        xnn_setup_average_pooling2d_nhwc_q8(
          average_pooling_op,
          batch_size(), input_height(), input_width(),
          input.data(), output.data(),
          nullptr /* thread pool */));

      ASSERT_EQ(xnn_status_success,
        xnn_run_operator(average_pooling_op, nullptr /* thread pool */));

      // Verify results of the first run.
      for (size_t i = 0; i < batch_size(); i++) {
        for (size_t y = 0; y < output_height(); y++) {
          for (size_t x = 0; x < output_width(); x++) {
            for (size_t c = 0; c < channels(); c++) {
              ASSERT_LE(uint32_t(output[((i * output_height() + y) * output_width() + x) * output_pixel_stride() + c]), uint32_t(qmax()));
              ASSERT_GE(uint32_t(output[((i * output_height() + y) * output_width() + x) * output_pixel_stride() + c]), uint32_t(qmin()));
              ASSERT_NEAR(float(int32_t(output[((i * output_height() + y) * output_width() + x) * output_pixel_stride() + c])),
                output_ref[((i * output_height() + y) * output_width() + x) * channels() + c], 0.80f) <<
                "in batch index " << i << ", pixel (" << y << ", " << x << "), channel " << c;
            }
          }
        }
      }

      // Re-generate data for the second run.
      std::generate(input.begin(), input.end(), std::ref(u8rng));
      std::fill(output.begin(), output.end(), 0xA5);

      // Compute reference results for the second run.
      for (size_t i = 0; i < next_batch_size(); i++) {
        for (size_t oy = 0; oy < next_output_height(); oy++) {
          for (size_t ox = 0; ox < next_output_width(); ox++) {
            for (size_t c = 0; c < channels(); c++) {
              double acc = 0.0f;
              for (size_t py = 0; py < pooling_height(); py++) {
                const size_t iy = oy * stride_height() + py - padding_top();
                for (size_t px = 0; px < pooling_width(); px++) {
                  const size_t ix = ox * stride_width() + px - padding_left();
                  if (ix < next_input_width() && iy < next_input_height()) {
                    acc += double(int32_t(input[((i * next_input_height() + iy) * next_input_width() + ix) * input_pixel_stride() + c]) - int32_t(input_zero_point()));
                  }
                }
              }
              next_output_ref[((i * next_output_height() + oy) * next_output_width() + ox) * channels() + c] = float(acc * scale + double(output_zero_point()));
              next_output_ref[((i * next_output_height() + oy) * next_output_width() + ox) * channels() + c] =
                std::min<float>(next_output_ref[((i * next_output_height() + oy) * next_output_width() + ox) * channels() + c], float(qmax()));
              next_output_ref[((i * next_output_height() + oy) * next_output_width() + ox) * channels() + c] =
                std::max<float>(next_output_ref[((i * next_output_height() + oy) * next_output_width() + ox) * channels() + c], float(qmin()));
            }
          }
        }
      }

      // Setup and run Average Pooling operator the second time, and destroy the operator.
      ASSERT_EQ(xnn_status_success,
        xnn_setup_average_pooling2d_nhwc_q8(
          average_pooling_op,
          next_batch_size(), next_input_height(), next_input_width(),
          input.data(), output.data(),
          nullptr /* thread pool */));

      ASSERT_EQ(xnn_status_success,
        xnn_run_operator(average_pooling_op, nullptr /* thread pool */));

      ASSERT_EQ(xnn_status_success,
        xnn_delete_operator(average_pooling_op));
      average_pooling_op = nullptr;

      // Verify results of the second run.
      for (size_t i = 0; i < next_batch_size(); i++) {
        for (size_t y = 0; y < next_output_height(); y++) {
          for (size_t x = 0; x < next_output_width(); x++) {
            for (size_t c = 0; c < channels(); c++) {
              ASSERT_LE(uint32_t(output[((i * next_output_height() + y) * next_output_width() + x) * output_pixel_stride() + c]), uint32_t(qmax()));
              ASSERT_GE(uint32_t(output[((i * next_output_height() + y) * next_output_width() + x) * output_pixel_stride() + c]), uint32_t(qmin()));
              ASSERT_NEAR(float(int32_t(output[((i * next_output_height() + y) * next_output_width() + x) * output_pixel_stride() + c])),
                next_output_ref[((i * next_output_height() + y) * next_output_width() + x) * channels() + c], 0.80f) <<
                "in batch index " << i << ", pixel (" << y << ", " << x << "), channel " << c;
            }
          }
        }
      }
    }
  }

  void TestSetupF32() const {
    std::random_device random_device;
    auto rng = std::mt19937(random_device());
    auto f32rng = std::bind(std::uniform_real_distribution<float>(), rng);

    std::vector<float> input(XNN_EXTRA_BYTES / sizeof(float) + std::max(
      (batch_size() * input_height() * input_width() - 1) * input_pixel_stride() + channels(),
      (next_batch_size() * next_input_height() * next_input_width() - 1) * input_pixel_stride() + channels()));
    std::vector<float> output(std::max(
      (batch_size() * output_height() * output_width() - 1) * output_pixel_stride() + channels(),
      (next_batch_size() * next_output_height() * next_output_width() - 1) * output_pixel_stride() + channels()));
    std::vector<float> output_ref(batch_size() * output_height() * output_width() * channels());
    std::vector<float> next_output_ref(next_batch_size() * next_output_height() * next_output_width() * channels());
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(input.begin(), input.end(), std::ref(f32rng));
      std::fill(output.begin(), output.end(), std::nanf(""));

      // Compute reference results, without clamping.
      for (size_t i = 0; i < batch_size(); i++) {
        for (size_t oy = 0; oy < output_height(); oy++) {
          for (size_t ox = 0; ox < output_width(); ox++) {
            for (size_t c = 0; c < channels(); c++) {
              float acc = 0.0f;
              size_t n = 0;
              for (size_t py = 0; py < pooling_height(); py++) {
                const size_t iy = oy * stride_height() + py - padding_top();
                for (size_t px = 0; px < pooling_width(); px++) {
                  const size_t ix = ox * stride_width() + px - padding_left();
                  if (ix < input_width() && iy < input_height()) {
                    acc += input[((i * input_height() + iy) * input_width() + ix) * input_pixel_stride() + c];
                    n += 1;
                  }
                }
              }
              output_ref[((i * output_height() + oy) * output_width() + ox) * channels() + c] = acc / float(n);
            }
          }
        }
      }

      // Compute clamping parameters.
      const float accumulated_min = *std::min_element(output_ref.cbegin(), output_ref.cend());
      const float accumulated_max = *std::max_element(output_ref.cbegin(), output_ref.cend());
      const float accumulated_range = accumulated_max - accumulated_min;
      const float output_min = accumulated_range == 0.0f ?
        -std::numeric_limits<float>::infinity() :
        accumulated_min + accumulated_range / 255.0f * float(qmin());
      const float output_max = accumulated_range == 0.0f ?
        +std::numeric_limits<float>::infinity() :
        accumulated_max - accumulated_range / 255.0f * float(255 - qmax());

      // Clamp reference results.
      for (float& value : output_ref) {
        value = std::max(std::min(value, output_max), output_min);
      }

      // Create, setup, and run Average Pooling operator once.
      ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
      xnn_operator_t average_pooling_op = nullptr;

      ASSERT_EQ(xnn_status_success,
        xnn_create_average_pooling2d_nhwc_f32(
          padding_top(), padding_right(), padding_bottom(), padding_left(),
          pooling_height(), pooling_width(),
          stride_height(), stride_width(),
          channels(), input_pixel_stride(), output_pixel_stride(),
          output_min, output_max,
          0, &average_pooling_op));
      ASSERT_NE(nullptr, average_pooling_op);

      ASSERT_EQ(xnn_status_success,
        xnn_setup_average_pooling2d_nhwc_f32(
          average_pooling_op,
          batch_size(), input_height(), input_width(),
          input.data(), output.data(),
          nullptr /* thread pool */));

      ASSERT_EQ(xnn_status_success,
        xnn_run_operator(average_pooling_op, nullptr /* thread pool */));

      // Verify results of the first run.
      for (size_t i = 0; i < batch_size(); i++) {
        for (size_t y = 0; y < output_height(); y++) {
          for (size_t x = 0; x < output_width(); x++) {
            for (size_t c = 0; c < channels(); c++) {
              ASSERT_LE(output[((i * output_height() + y) * output_width() + x) * output_pixel_stride() + c], output_max);
              ASSERT_GE(output[((i * output_height() + y) * output_width() + x) * output_pixel_stride() + c], output_min);
              ASSERT_NEAR(output[((i * output_height() + y) * output_width() + x) * output_pixel_stride() + c],
                  output_ref[((i * output_height() + y) * output_width() + x) * channels() + c],
                  std::abs(output_ref[((i * output_height() + y) * output_width() + x) * channels() + c]) * 1.0e-6f) <<
                "in batch index " << i << ", pixel (" << y << ", " << x << "), channel " << c;
            }
          }
        }
      }

      // Re-generate data for the second run.
      std::generate(input.begin(), input.end(), std::ref(f32rng));
      std::fill(output.begin(), output.end(), std::nanf(""));

      // Compute reference results for the second run.
      for (size_t i = 0; i < next_batch_size(); i++) {
        for (size_t oy = 0; oy < next_output_height(); oy++) {
          for (size_t ox = 0; ox < next_output_width(); ox++) {
            for (size_t c = 0; c < channels(); c++) {
              float acc = 0.0f;
              int32_t n = 0;
              for (size_t py = 0; py < pooling_height(); py++) {
                const size_t iy = oy * stride_height() + py - padding_top();
                for (size_t px = 0; px < pooling_width(); px++) {
                  const size_t ix = ox * stride_width() + px - padding_left();
                  if (ix < next_input_width() && iy < next_input_height()) {
                    acc += input[((i * next_input_height() + iy) * next_input_width() + ix) * input_pixel_stride() + c];
                    n += 1;
                  }
                }
              }
              next_output_ref[((i * next_output_height() + oy) * next_output_width() + ox) * channels() + c] =
                std::max(std::min(acc / float(n), output_max), output_min);
            }
          }
        }
      }

      // Setup and run Average Pooling operator the second time, and destroy the operator.
      ASSERT_EQ(xnn_status_success,
        xnn_setup_average_pooling2d_nhwc_f32(
          average_pooling_op,
          next_batch_size(), next_input_height(), next_input_width(),
          input.data(), output.data(),
          nullptr /* thread pool */));

      ASSERT_EQ(xnn_status_success,
        xnn_run_operator(average_pooling_op, nullptr /* thread pool */));

      ASSERT_EQ(xnn_status_success,
        xnn_delete_operator(average_pooling_op));
      average_pooling_op = nullptr;

      // Verify results of the second run.
      for (size_t i = 0; i < next_batch_size(); i++) {
        for (size_t y = 0; y < next_output_height(); y++) {
          for (size_t x = 0; x < next_output_width(); x++) {
            for (size_t c = 0; c < channels(); c++) {
              ASSERT_LE(output[((i * next_output_height() + y) * next_output_width() + x) * output_pixel_stride() + c], output_max);
              ASSERT_GE(output[((i * next_output_height() + y) * next_output_width() + x) * output_pixel_stride() + c], output_min);
              ASSERT_NEAR(output[((i * next_output_height() + y) * next_output_width() + x) * output_pixel_stride() + c],
                  next_output_ref[((i * next_output_height() + y) * next_output_width() + x) * channels() + c],
                  std::abs(next_output_ref[((i * next_output_height() + y) * next_output_width() + x) * channels() + c]) * 1.0e-6f) <<
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
  bool padding_tf_same_{false};
  size_t input_height_{1};
  size_t input_width_{1};
  size_t channels_{1};
  size_t batch_size_{1};
  size_t input_pixel_stride_{0};
  size_t output_pixel_stride_{0};
  uint32_t pooling_height_{1};
  uint32_t pooling_width_{1};
  uint32_t stride_height_{1};
  uint32_t stride_width_{1};
  size_t next_input_height_{0};
  size_t next_input_width_{0};
  size_t next_batch_size_{0};
  float input_scale_{1.0f};
  float output_scale_{1.0f};
  uint8_t input_zero_point_{121};
  uint8_t output_zero_point_{133};
  uint8_t qmin_{0};
  uint8_t qmax_{255};
  size_t iterations_{1};
};
