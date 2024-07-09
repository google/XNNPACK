// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// Copyright 2019 Google LLC
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
#include <memory>
#include <random>
#include <vector>

#include <gtest/gtest.h>
#include <fp16/fp16.h>
#include "xnnpack.h"
#include "replicable_random_device.h"

class MaxPoolingOperatorTester {
 public:
  MaxPoolingOperatorTester& padding_tf_same(bool padding_same) {
    if (padding_same) {
      assert(padding_top() == 0);
      assert(padding_left() == 0);
      assert(padding_bottom() == 0);
      assert(padding_right() == 0);
    }
    this->padding_tf_same_ = padding_same;
    return *this;
  }

  bool padding_tf_same() const {
    return this->padding_tf_same_;
  }

  MaxPoolingOperatorTester& padding(uint32_t padding) {
    assert(!padding_tf_same());
    this->padding_top_ = padding;
    this->padding_right_ = padding;
    this->padding_bottom_ = padding;
    this->padding_left_ = padding;
    return *this;
  }

  MaxPoolingOperatorTester& padding(uint32_t padding_height, uint32_t padding_width) {
    assert(!padding_tf_same());
    this->padding_top_ = padding_height;
    this->padding_right_ = padding_width;
    this->padding_bottom_ = padding_height;
    this->padding_left_ = padding_width;
    return *this;
  }

  MaxPoolingOperatorTester& padding_height(uint32_t padding_height) {
    assert(!padding_tf_same());
    this->padding_top_ = padding_height;
    this->padding_bottom_ = padding_height;
    return *this;
  }

  MaxPoolingOperatorTester& padding_width(uint32_t padding_width) {
    assert(!padding_tf_same());
    this->padding_right_ = padding_width;
    this->padding_left_ = padding_width;
    return *this;
  }

  MaxPoolingOperatorTester& padding_top(uint32_t padding_top) {
    assert(!padding_tf_same());
    this->padding_top_ = padding_top;
    return *this;
  }

  uint32_t padding_top() const {
    if (padding_tf_same()) {
      const uint32_t total_padding_height =
        (output_height() - 1) * stride_height() + dilated_pooling_height() - input_height();
      return total_padding_height / 2;
    } else {
      return this->padding_top_;
    }
  }

  MaxPoolingOperatorTester& padding_left(uint32_t padding_left) {
    assert(!padding_tf_same());
    this->padding_left_ = padding_left;
    return *this;
  }

  uint32_t padding_left() const {
    if (padding_tf_same()) {
      const uint32_t total_padding_width =
        (output_width() - 1) * stride_width() + dilated_pooling_width() - input_width();
      return total_padding_width / 2;
    } else {
      return this->padding_left_;
    }
  }

  MaxPoolingOperatorTester& padding_bottom(uint32_t padding_bottom) {
    assert(!padding_tf_same());
    this->padding_bottom_ = padding_bottom;
    return *this;
  }

  uint32_t padding_bottom() const {
    if (padding_tf_same()) {
      const uint32_t total_padding_height =
        (output_height() - 1) * stride_height() + dilated_pooling_height() - input_height();
      return total_padding_height - total_padding_height / 2;
    } else {
      return this->padding_bottom_;
    }
  }

  MaxPoolingOperatorTester& padding_right(uint32_t padding_right) {
    assert(!padding_tf_same());
    this->padding_right_ = padding_right;
    return *this;
  }

  uint32_t padding_right() const {
    if (padding_tf_same()) {
      const uint32_t total_padding_width =
        (output_width() - 1) * stride_width() + dilated_pooling_width() - input_width();
      return total_padding_width - total_padding_width / 2;
    } else {
      return this->padding_right_;
    }
  }

  MaxPoolingOperatorTester& input_size(size_t input_height, size_t input_width) {
    assert(input_height >= 1);
    assert(input_width >= 1);
    this->input_height_ = input_height;
    this->input_width_ = input_width;
    return *this;
  }

  MaxPoolingOperatorTester& input_height(size_t input_height) {
    assert(input_height >= 1);
    this->input_height_ = input_height;
    return *this;
  }

  size_t input_height() const {
    return this->input_height_;
  }

  MaxPoolingOperatorTester& input_width(size_t input_width) {
    assert(input_width >= 1);
    this->input_width_ = input_width;
    return *this;
  }

  size_t input_width() const {
    return this->input_width_;
  }

  MaxPoolingOperatorTester& channels(size_t channels) {
    assert(channels != 0);
    this->channels_ = channels;
    return *this;
  }

  size_t channels() const {
    return this->channels_;
  }

  MaxPoolingOperatorTester& batch_size(size_t batch_size) {
    assert(batch_size != 0);
    this->batch_size_ = batch_size;
    return *this;
  }

  size_t batch_size() const {
    return this->batch_size_;
  }

  MaxPoolingOperatorTester& pooling_size(uint32_t pooling_size) {
    assert(pooling_size >= 1);
    this->pooling_height_ = pooling_size;
    this->pooling_width_ = pooling_size;
    return *this;
  }

  MaxPoolingOperatorTester& pooling_size(uint32_t pooling_height, uint32_t pooling_width) {
    assert(pooling_height >= 1);
    assert(pooling_width >= 1);
    this->pooling_height_ = pooling_height;
    this->pooling_width_ = pooling_width;
    return *this;
  }

  MaxPoolingOperatorTester& pooling_height(uint32_t pooling_height) {
    assert(pooling_height >= 1);
    this->pooling_height_ = pooling_height;
    return *this;
  }

  uint32_t pooling_height() const {
    return this->pooling_height_;
  }

  MaxPoolingOperatorTester& pooling_width(uint32_t pooling_width) {
    assert(pooling_width >= 1);
    this->pooling_width_ = pooling_width;
    return *this;
  }

  uint32_t pooling_width() const {
    return this->pooling_width_;
  }

  MaxPoolingOperatorTester& stride(uint32_t stride) {
    assert(stride >= 1);
    this->stride_height_ = stride;
    this->stride_width_ = stride;
    return *this;
  }

  MaxPoolingOperatorTester& stride(uint32_t stride_height, uint32_t stride_width) {
    assert(stride_height >= 1);
    assert(stride_width >= 1);
    this->stride_height_ = stride_height;
    this->stride_width_ = stride_width;
    return *this;
  }

  MaxPoolingOperatorTester& stride_height(uint32_t stride_height) {
    assert(stride_height >= 1);
    this->stride_height_ = stride_height;
    return *this;
  }

  uint32_t stride_height() const {
    return this->stride_height_;
  }

  MaxPoolingOperatorTester& stride_width(uint32_t stride_width) {
    assert(stride_width >= 1);
    this->stride_width_ = stride_width;
    return *this;
  }

  uint32_t stride_width() const {
    return this->stride_width_;
  }

  MaxPoolingOperatorTester& dilation(uint32_t dilation) {
    assert(dilation >= 1);
    this->dilation_height_ = dilation;
    this->dilation_width_ = dilation;
    return *this;
  }

  MaxPoolingOperatorTester& dilation(uint32_t dilation_height, uint32_t dilation_width) {
    assert(dilation_height >= 1);
    assert(dilation_width >= 1);
    this->dilation_height_ = dilation_height;
    this->dilation_width_ = dilation_width;
    return *this;
  }

  MaxPoolingOperatorTester& dilation_height(uint32_t dilation_height) {
    assert(dilation_height >= 1);
    this->dilation_height_ = dilation_height;
    return *this;
  }

  uint32_t dilation_height() const {
    return this->dilation_height_;
  }

  MaxPoolingOperatorTester& dilation_width(uint32_t dilation_width) {
    assert(dilation_width >= 1);
    this->dilation_width_ = dilation_width;
    return *this;
  }

  uint32_t dilation_width() const {
    return this->dilation_width_;
  }

  uint32_t dilated_pooling_height() const {
    return (pooling_height() - 1) * dilation_height() + 1;
  }

  uint32_t dilated_pooling_width() const {
    return (pooling_width() - 1) * dilation_width() + 1;
  }

  size_t output_height() const {
    if (padding_tf_same()) {
      return (input_height() + stride_height() - 1) / stride_height();
    } else {
      const size_t padded_input_height = padding_top() + input_height() + padding_bottom();
      if (padded_input_height <= dilated_pooling_height()) {
        return 1;
      } else {
        return (padded_input_height - dilated_pooling_height()) / stride_height() + 1;
      }
    }
  }

  size_t output_width() const {
    if (padding_tf_same()) {
      return (input_width() + stride_width() - 1) / stride_width();
    } else {
      const size_t padded_input_width = padding_left() + input_width() + padding_right();
      if (padded_input_width <= dilated_pooling_width()) {
        return 1;
      } else {
        return (padded_input_width - dilated_pooling_width()) / stride_width() + 1;
      }
    }
  }

  MaxPoolingOperatorTester& input_pixel_stride(size_t input_pixel_stride) {
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

  MaxPoolingOperatorTester& output_pixel_stride(size_t output_pixel_stride) {
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

  MaxPoolingOperatorTester& next_input_size(uint32_t next_input_height, uint32_t next_input_width) {
    assert(next_input_height >= 1);
    assert(next_input_width >= 1);
    this->next_input_height_ = next_input_height;
    this->next_input_width_ = next_input_width;
    return *this;
  }

  MaxPoolingOperatorTester& next_input_height(uint32_t next_input_height) {
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

  MaxPoolingOperatorTester& next_input_width(uint32_t next_input_width) {
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
    const size_t padded_next_input_height = padding_top() + next_input_height() + padding_bottom();
    if (padded_next_input_height <= dilated_pooling_height()) {
      return 1;
    } else {
      return (padded_next_input_height - dilated_pooling_height()) / stride_height() + 1;
    }
  }

  size_t next_output_width() const {
    const size_t padded_next_input_width = padding_left() + next_input_width() + padding_right();
    if (padded_next_input_width <= dilated_pooling_width()) {
      return 1;
    } else {
      return (padded_next_input_width - dilated_pooling_width()) / stride_width() + 1;
    }
  }

  MaxPoolingOperatorTester& next_batch_size(size_t next_batch_size) {
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

  MaxPoolingOperatorTester& qmin(uint8_t qmin) {
    this->qmin_ = qmin;
    return *this;
  }

  uint8_t qmin() const {
    return this->qmin_;
  }

  MaxPoolingOperatorTester& qmax(uint8_t qmax) {
    this->qmax_ = qmax;
    return *this;
  }

  uint8_t qmax() const {
    return this->qmax_;
  }

  MaxPoolingOperatorTester& iterations(size_t iterations) {
    this->iterations_ = iterations;
    return *this;
  }

  size_t iterations() const {
    return this->iterations_;
  }

  void TestS8() const {
    xnnpack::ReplicableRandomDevice rng;
    std::uniform_int_distribution<int32_t> i8dist(
      std::numeric_limits<int8_t>::min(), std::numeric_limits<int8_t>::max());

    std::vector<int8_t> input((batch_size() * input_height() * input_width() - 1) * input_pixel_stride() + channels() + XNN_EXTRA_BYTES / sizeof(int8_t));
    std::vector<int8_t> output((batch_size() * output_height() * output_width() - 1) * output_pixel_stride() + channels() + XNN_EXTRA_BYTES / sizeof(int8_t));
    std::vector<int8_t> output_ref(batch_size() * output_height() * output_width() * channels());
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(input.begin(), input.end(), [&]() { return i8dist(rng); });
      std::fill(output.begin(), output.end(), 0xA5);

      // Compute reference results.
      for (size_t i = 0; i < batch_size(); i++) {
        for (size_t oy = 0; oy < output_height(); oy++) {
          for (size_t ox = 0; ox < output_width(); ox++) {
            for (size_t c = 0; c < channels(); c++) {
              int8_t max_value = std::numeric_limits<int8_t>::min();
              for (size_t py = 0; py < pooling_height(); py++) {
                const size_t iy = oy * stride_height() + py * dilation_height() - padding_top();
                for (size_t px = 0; px < pooling_width(); px++) {
                  const size_t ix = ox * stride_width() + px * dilation_width() - padding_left();
                  if (ix < input_width() && iy < input_height()) {
                    max_value = std::max(max_value,
                      input[((i * input_height() + iy) * input_width() + ix) * input_pixel_stride() + c]);
                  }
                }
              }
              max_value = std::min(max_value, int8_t(qmax() - 0x80));
              max_value = std::max(max_value, int8_t(qmin() - 0x80));
              output_ref[((i * output_height() + oy) * output_width() + ox) * channels() + c] = max_value;
            }
          }
        }
      }

      // Create, setup, run, and destroy Max Pooling operator.
      ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
      xnn_operator_t max_pooling_op = nullptr;

      ASSERT_EQ(xnn_status_success,
        xnn_create_max_pooling2d_nhwc_s8(
          padding_tf_same() ? 0 : padding_top(), padding_tf_same() ? 0 : padding_right(),
          padding_tf_same() ? 0 : padding_bottom(), padding_tf_same() ? 0 : padding_left(),
          pooling_height(), pooling_width(),
          stride_height(), stride_width(),
          dilation_height(), dilation_width(),
          int8_t(qmin() - 0x80), int8_t(qmax() - 0x80),
          padding_tf_same() ? XNN_FLAG_TENSORFLOW_SAME_PADDING : 0,
          &max_pooling_op));
      ASSERT_NE(nullptr, max_pooling_op);

      // Smart pointer to automatically delete max_pooling_op.
      std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_max_pooling_op(max_pooling_op, xnn_delete_operator);

      ASSERT_EQ(xnn_status_success,
        xnn_reshape_max_pooling2d_nhwc_s8(
          max_pooling_op,
          batch_size(), input_height(), input_width(),
          channels(), input_pixel_stride(), output_pixel_stride(),
          /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
          /*threadpool=*/nullptr));

      ASSERT_EQ(xnn_status_success,
        xnn_setup_max_pooling2d_nhwc_s8(
          max_pooling_op,
          input.data(), output.data()));

      ASSERT_EQ(xnn_status_success,
        xnn_run_operator(max_pooling_op, /*threadpool=*/nullptr));

      // Verify results.
      for (size_t i = 0; i < batch_size(); i++) {
        for (size_t y = 0; y < output_height(); y++) {
          for (size_t x = 0; x < output_width(); x++) {
            for (size_t c = 0; c < channels(); c++) {
              EXPECT_LE(int32_t(output[((i * output_height() + y) * output_width() + x) * output_pixel_stride() + c]), int32_t(qmax() - 0x80));
              EXPECT_GE(int32_t(output[((i * output_height() + y) * output_width() + x) * output_pixel_stride() + c]), int32_t(qmin() - 0x80));
              EXPECT_EQ(int32_t(output_ref[((i * output_height() + y) * output_width() + x) * channels() + c]),
                int32_t(output[((i * output_height() + y) * output_width() + x) * output_pixel_stride() + c])) <<
                "in batch index " << i << ", pixel (" << y << ", " << x << "), channel " << c;
            }
          }
        }
      }
    }
  }

  void TestU8() const {
    xnnpack::ReplicableRandomDevice rng;
    std::uniform_int_distribution<int32_t> u8dist(
      std::numeric_limits<uint8_t>::min(), std::numeric_limits<uint8_t>::max());

    std::vector<uint8_t> input((batch_size() * input_height() * input_width() - 1) * input_pixel_stride() + channels() + XNN_EXTRA_BYTES / sizeof(uint8_t));
    std::vector<uint8_t> output((batch_size() * output_height() * output_width() - 1) * output_pixel_stride() + channels() + XNN_EXTRA_BYTES / sizeof(uint8_t));
    std::vector<uint8_t> output_ref(batch_size() * output_height() * output_width() * channels());
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(input.begin(), input.end(), [&]() { return u8dist(rng); });
      std::fill(output.begin(), output.end(), 0xA5);

      // Compute reference results.
      for (size_t i = 0; i < batch_size(); i++) {
        for (size_t oy = 0; oy < output_height(); oy++) {
          for (size_t ox = 0; ox < output_width(); ox++) {
            for (size_t c = 0; c < channels(); c++) {
              uint8_t max_value = 0;
              for (size_t py = 0; py < pooling_height(); py++) {
                const size_t iy = oy * stride_height() + py * dilation_height() - padding_top();
                for (size_t px = 0; px < pooling_width(); px++) {
                  const size_t ix = ox * stride_width() + px * dilation_width() - padding_left();
                  if (ix < input_width() && iy < input_height()) {
                    max_value = std::max(max_value,
                      input[((i * input_height() + iy) * input_width() + ix) * input_pixel_stride() + c]);
                  }
                }
              }
              max_value = std::min(max_value, qmax());
              max_value = std::max(max_value, qmin());
              output_ref[((i * output_height() + oy) * output_width() + ox) * channels() + c] = max_value;
            }
          }
        }
      }

      // Create, setup, run, and destroy Max Pooling operator.
      ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
      xnn_operator_t max_pooling_op = nullptr;

      ASSERT_EQ(xnn_status_success,
        xnn_create_max_pooling2d_nhwc_u8(
          padding_tf_same() ? 0 : padding_top(), padding_tf_same() ? 0 : padding_right(),
          padding_tf_same() ? 0 : padding_bottom(), padding_tf_same() ? 0 : padding_left(),
          pooling_height(), pooling_width(),
          stride_height(), stride_width(),
          dilation_height(), dilation_width(),
          qmin(), qmax(),
          padding_tf_same() ? XNN_FLAG_TENSORFLOW_SAME_PADDING : 0,
          &max_pooling_op));
      ASSERT_NE(nullptr, max_pooling_op);

      // Smart pointer to automatically delete max_pooling_op.
      std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_max_pooling_op(max_pooling_op, xnn_delete_operator);

      ASSERT_EQ(xnn_status_success,
        xnn_reshape_max_pooling2d_nhwc_u8(
          max_pooling_op,
          batch_size(), input_height(), input_width(),
          channels(), input_pixel_stride(), output_pixel_stride(),
          /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
          /*threadpool=*/nullptr));

      ASSERT_EQ(xnn_status_success,
        xnn_setup_max_pooling2d_nhwc_u8(
          max_pooling_op,
          input.data(), output.data()));

      ASSERT_EQ(xnn_status_success,
        xnn_run_operator(max_pooling_op, /*threadpool=*/nullptr));

      // Verify results.
      for (size_t i = 0; i < batch_size(); i++) {
        for (size_t y = 0; y < output_height(); y++) {
          for (size_t x = 0; x < output_width(); x++) {
            for (size_t c = 0; c < channels(); c++) {
              EXPECT_LE(uint32_t(output[((i * output_height() + y) * output_width() + x) * output_pixel_stride() + c]), uint32_t(qmax()));
              EXPECT_GE(uint32_t(output[((i * output_height() + y) * output_width() + x) * output_pixel_stride() + c]), uint32_t(qmin()));
              EXPECT_EQ(uint32_t(output_ref[((i * output_height() + y) * output_width() + x) * channels() + c]),
                uint32_t(output[((i * output_height() + y) * output_width() + x) * output_pixel_stride() + c])) <<
                "in batch index " << i << ", pixel (" << y << ", " << x << "), channel " << c;
            }
          }
        }
      }
    }
  }

  void TestF16() const {
    xnnpack::ReplicableRandomDevice rng;
    // Note: we need to avoid FP16 denormals in the generated tensor because they might be processed differently in
    // native vs emulated arithmetics, and we use exact comparison to verify the results against reference.
    std::uniform_real_distribution<float> f32dist(0.001f, 1.0f);

    std::vector<uint16_t> input((batch_size() * input_height() * input_width() - 1) * input_pixel_stride() + channels() + XNN_EXTRA_BYTES / sizeof(uint16_t));
    std::vector<uint16_t> output((batch_size() * output_height() * output_width() - 1) * output_pixel_stride() + channels() + XNN_EXTRA_BYTES / sizeof(uint16_t));
    std::vector<float> output_ref(batch_size() * output_height() * output_width() * channels());
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(input.begin(), input.end(), [&]() { return fp16_ieee_from_fp32_value(f32dist(rng)); });
      std::fill(output.begin(), output.end(), UINT16_C(0x7E00) /* NaN */);

      // Compute reference results, without clamping.
      for (size_t i = 0; i < batch_size(); i++) {
        for (size_t oy = 0; oy < output_height(); oy++) {
          for (size_t ox = 0; ox < output_width(); ox++) {
            for (size_t c = 0; c < channels(); c++) {
              float max_value = -std::numeric_limits<float>::infinity();
              for (size_t py = 0; py < pooling_height(); py++) {
                const size_t iy = oy * stride_height() + py * dilation_height() - padding_top();
                for (size_t px = 0; px < pooling_width(); px++) {
                  const size_t ix = ox * stride_width() + px * dilation_width() - padding_left();
                  if (ix < input_width() && iy < input_height()) {
                    max_value = std::max(max_value,
                      fp16_ieee_to_fp32_value(input[((i * input_height() + iy) * input_width() + ix) * input_pixel_stride() + c]));
                  }
                }
              }
              output_ref[((i * output_height() + oy) * output_width() + ox) * channels() + c] = max_value;
            }
          }
        }
      }

      // Compute clamping parameters.
      const float accumulated_min = *std::min_element(output_ref.cbegin(), output_ref.cend());
      const float accumulated_max = *std::max_element(output_ref.cbegin(), output_ref.cend());
      const float accumulated_range = accumulated_max - accumulated_min;
      float output_min = accumulated_min + accumulated_range / 255.0f * float(qmin());
      float output_max = accumulated_max - accumulated_range / 255.0f * float(255 - qmax());
      output_min = fp16_ieee_to_fp32_value(fp16_ieee_from_fp32_value(output_min));
      output_max = fp16_ieee_to_fp32_value(fp16_ieee_from_fp32_value(output_max));
      if (accumulated_range == 0.0f) {
        output_min = -std::numeric_limits<float>::infinity();
        output_max = +std::numeric_limits<float>::infinity();
      }
      if (qmin() == std::numeric_limits<uint8_t>::min()) {
        output_min = -std::numeric_limits<float>::infinity();
      }
      if (qmax() == std::numeric_limits<uint8_t>::max()) {
        output_max = +std::numeric_limits<float>::infinity();
      }

      // Clamp reference results.
      for (float& value : output_ref) {
        value = std::max(std::min(value, output_max), output_min);
      }

      // Create, setup, run, and destroy Max Pooling operator.
      ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
      xnn_operator_t max_pooling_op = nullptr;

      const xnn_status status = xnn_create_max_pooling2d_nhwc_f16(
          padding_tf_same() ? 0 : padding_top(), padding_tf_same() ? 0 : padding_right(),
          padding_tf_same() ? 0 : padding_bottom(), padding_tf_same() ? 0 : padding_left(),
          pooling_height(), pooling_width(),
          stride_height(), stride_width(),
          dilation_height(), dilation_width(),
          output_min, output_max,
          padding_tf_same() ? XNN_FLAG_TENSORFLOW_SAME_PADDING : 0,
          &max_pooling_op);
      if (status == xnn_status_unsupported_hardware) {
        GTEST_SKIP();
      }
      ASSERT_EQ(xnn_status_success, status);
      ASSERT_NE(nullptr, max_pooling_op);

      // Smart pointer to automatically delete max_pooling_op.
      std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_max_pooling_op(max_pooling_op, xnn_delete_operator);

      ASSERT_EQ(xnn_status_success,
        xnn_reshape_max_pooling2d_nhwc_f16(
          max_pooling_op,
          batch_size(), input_height(), input_width(),
          channels(), input_pixel_stride(), output_pixel_stride(),
          /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
          /*threadpool=*/nullptr));

      ASSERT_EQ(xnn_status_success,
        xnn_setup_max_pooling2d_nhwc_f16(
          max_pooling_op,
          input.data(), output.data()));

      ASSERT_EQ(xnn_status_success,
        xnn_run_operator(max_pooling_op, /*threadpool=*/nullptr));

      // Verify results.
      for (size_t i = 0; i < batch_size(); i++) {
        for (size_t y = 0; y < output_height(); y++) {
          for (size_t x = 0; x < output_width(); x++) {
            for (size_t c = 0; c < channels(); c++) {
              EXPECT_LE(fp16_ieee_to_fp32_value(output[((i * output_height() + y) * output_width() + x) * output_pixel_stride() + c]), output_max);
              EXPECT_GE(fp16_ieee_to_fp32_value(output[((i * output_height() + y) * output_width() + x) * output_pixel_stride() + c]), output_min);
              EXPECT_EQ(
                  fp16_ieee_to_fp32_value(output[((i * output_height() + y) * output_width() + x) * output_pixel_stride() + c]),
                  output_ref[((i * output_height() + y) * output_width() + x) * channels() + c]) <<
                "in batch index " << i << ", pixel (" << y << ", " << x << "), channel " << c
                << ", min = " << output_min << ", max = " << output_max;
            }
          }
        }
      }
    }
  }

  void TestF32() const {
    xnnpack::ReplicableRandomDevice rng;
    std::uniform_real_distribution<float> f32dist;

    std::vector<float> input((batch_size() * input_height() * input_width() - 1) * input_pixel_stride() + channels() + XNN_EXTRA_BYTES / sizeof(float));
    std::vector<float> output((batch_size() * output_height() * output_width() - 1) * output_pixel_stride() + channels() + XNN_EXTRA_BYTES / sizeof(float));
    std::vector<float> output_ref(batch_size() * output_height() * output_width() * channels());
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(input.begin(), input.end(), [&]() { return f32dist(rng); });
      std::fill(output.begin(), output.end(), nanf(""));

      // Compute reference results, without clamping.
      for (size_t i = 0; i < batch_size(); i++) {
        for (size_t oy = 0; oy < output_height(); oy++) {
          for (size_t ox = 0; ox < output_width(); ox++) {
            for (size_t c = 0; c < channels(); c++) {
              float max_value = -std::numeric_limits<float>::infinity();
              for (size_t py = 0; py < pooling_height(); py++) {
                const size_t iy = oy * stride_height() + py * dilation_height() - padding_top();
                for (size_t px = 0; px < pooling_width(); px++) {
                  const size_t ix = ox * stride_width() + px * dilation_width() - padding_left();
                  if (ix < input_width() && iy < input_height()) {
                    max_value = std::max(max_value,
                      input[((i * input_height() + iy) * input_width() + ix) * input_pixel_stride() + c]);
                  }
                }
              }
              output_ref[((i * output_height() + oy) * output_width() + ox) * channels() + c] = max_value;
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

      // Create, setup, run, and destroy Max Pooling operator.
      ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
      xnn_operator_t max_pooling_op = nullptr;

      ASSERT_EQ(xnn_status_success,
        xnn_create_max_pooling2d_nhwc_f32(
          padding_tf_same() ? 0 : padding_top(), padding_tf_same() ? 0 : padding_right(),
          padding_tf_same() ? 0 : padding_bottom(), padding_tf_same() ? 0 : padding_left(),
          pooling_height(), pooling_width(),
          stride_height(), stride_width(),
          dilation_height(), dilation_width(),
          output_min, output_max,
          padding_tf_same() ? XNN_FLAG_TENSORFLOW_SAME_PADDING : 0,
          &max_pooling_op));
      ASSERT_NE(nullptr, max_pooling_op);

      // Smart pointer to automatically delete max_pooling_op.
      std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_max_pooling_op(max_pooling_op, xnn_delete_operator);

      ASSERT_EQ(xnn_status_success,
        xnn_reshape_max_pooling2d_nhwc_f32(
          max_pooling_op,
          batch_size(), input_height(), input_width(),
          channels(), input_pixel_stride(), output_pixel_stride(),
          /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
          /*threadpool=*/nullptr));

      ASSERT_EQ(xnn_status_success,
        xnn_setup_max_pooling2d_nhwc_f32(
          max_pooling_op,
          input.data(), output.data()));

      ASSERT_EQ(xnn_status_success,
        xnn_run_operator(max_pooling_op, /*threadpool=*/nullptr));

      // Verify results.
      for (size_t i = 0; i < batch_size(); i++) {
        for (size_t y = 0; y < output_height(); y++) {
          for (size_t x = 0; x < output_width(); x++) {
            for (size_t c = 0; c < channels(); c++) {
              EXPECT_LE(output[((i * output_height() + y) * output_width() + x) * output_pixel_stride() + c], output_max);
              EXPECT_GE(output[((i * output_height() + y) * output_width() + x) * output_pixel_stride() + c], output_min);
              EXPECT_EQ(output_ref[((i * output_height() + y) * output_width() + x) * channels() + c],
                output[((i * output_height() + y) * output_width() + x) * output_pixel_stride() + c]) <<
                "in batch index " << i << ", pixel (" << y << ", " << x << "), channel " << c
                << ", min = " << output_min << ", max = " << output_max;
            }
          }
        }
      }
    }
  }

  void TestSetupS8() const {
    xnnpack::ReplicableRandomDevice rng;
    std::uniform_int_distribution<int32_t> i8dist(
      std::numeric_limits<int8_t>::min(), std::numeric_limits<int8_t>::max());

    std::vector<int8_t> input(XNN_EXTRA_BYTES / sizeof(int8_t) + std::max<size_t>(
      (batch_size() * input_height() * input_width() - 1) * input_pixel_stride() + channels(),
      (next_batch_size() * next_input_height() * next_input_width() - 1) * input_pixel_stride() + channels()));
    std::vector<int8_t> output(XNN_EXTRA_BYTES / sizeof(int8_t) + std::max<size_t>(
      (batch_size() * output_height() * output_width() - 1) * output_pixel_stride() + channels(),
      (next_batch_size() * next_output_height() * next_output_width() - 1) * output_pixel_stride() + channels()));
    std::vector<float> output_ref(batch_size() * output_height() * output_width() * channels());
    std::vector<float> next_output_ref(next_batch_size() * next_output_height() * next_output_width() * channels());
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(input.begin(), input.end(), [&]() { return i8dist(rng); });
      std::fill(output.begin(), output.end(), INT8_C(0xA5));

      // Compute reference results.
      for (size_t i = 0; i < batch_size(); i++) {
        for (size_t oy = 0; oy < output_height(); oy++) {
          for (size_t ox = 0; ox < output_width(); ox++) {
            for (size_t c = 0; c < channels(); c++) {
              int8_t max_value = std::numeric_limits<int8_t>::min();
              for (size_t py = 0; py < pooling_height(); py++) {
                const size_t iy = oy * stride_height() + py * dilation_height() - padding_top();
                for (size_t px = 0; px < pooling_width(); px++) {
                  const size_t ix = ox * stride_width() + px * dilation_width() - padding_left();
                  if (ix < input_width() && iy < input_height()) {
                    max_value = std::max(max_value,
                      input[((i * input_height() + iy) * input_width() + ix) * input_pixel_stride() + c]);
                  }
                }
              }
              max_value = std::min(max_value, int8_t(qmax() - 0x80));
              max_value = std::max(max_value, int8_t(qmin() - 0x80));
              output_ref[((i * output_height() + oy) * output_width() + ox) * channels() + c] = max_value;
            }
          }
        }
      }

      // Create, setup, and run Max Pooling operator once.
      ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
      xnn_operator_t max_pooling_op = nullptr;

      ASSERT_EQ(xnn_status_success,
        xnn_create_max_pooling2d_nhwc_s8(
          padding_top(), padding_right(), padding_bottom(), padding_left(),
          pooling_height(), pooling_width(),
          stride_height(), stride_width(),
          dilation_height(), dilation_width(),
          int8_t(qmin() - 0x80), int8_t(qmax() - 0x80),
          0, &max_pooling_op));
      ASSERT_NE(nullptr, max_pooling_op);

      // Smart pointer to automatically delete max_pooling_op.
      std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_max_pooling_op(max_pooling_op, xnn_delete_operator);

      ASSERT_EQ(xnn_status_success,
        xnn_reshape_max_pooling2d_nhwc_s8(
          max_pooling_op,
          batch_size(), input_height(), input_width(),
          channels(), input_pixel_stride(), output_pixel_stride(),
          /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
          /*threadpool=*/nullptr));

      ASSERT_EQ(xnn_status_success,
        xnn_setup_max_pooling2d_nhwc_s8(
          max_pooling_op,
          input.data(), output.data()));

      ASSERT_EQ(xnn_status_success,
        xnn_run_operator(max_pooling_op, /*threadpool=*/nullptr));

      // Verify results of the first run.
      for (size_t i = 0; i < batch_size(); i++) {
        for (size_t y = 0; y < output_height(); y++) {
          for (size_t x = 0; x < output_width(); x++) {
            for (size_t c = 0; c < channels(); c++) {
              EXPECT_LE(int32_t(output[((i * output_height() + y) * output_width() + x) * output_pixel_stride() + c]), int32_t(qmax() - 0x80));
              EXPECT_GE(int32_t(output[((i * output_height() + y) * output_width() + x) * output_pixel_stride() + c]), int32_t(qmin() - 0x80));
              EXPECT_EQ(int32_t(output_ref[((i * output_height() + y) * output_width() + x) * channels() + c]),
                int32_t(output[((i * output_height() + y) * output_width() + x) * output_pixel_stride() + c])) <<
                "in batch index " << i << ", pixel (" << y << ", " << x << "), channel " << c;
            }
          }
        }
      }

      // Re-generate data for the second run.
      std::generate(input.begin(), input.end(), [&]() { return i8dist(rng); });
      std::fill(output.begin(), output.end(), 0xA5);

      // Compute reference results for the second run.
      for (size_t i = 0; i < next_batch_size(); i++) {
        for (size_t oy = 0; oy < next_output_height(); oy++) {
          for (size_t ox = 0; ox < next_output_width(); ox++) {
            for (size_t c = 0; c < channels(); c++) {
              int8_t max_value = std::numeric_limits<int8_t>::min();
              for (size_t py = 0; py < pooling_height(); py++) {
                const size_t iy = oy * stride_height() + py * dilation_height() - padding_top();
                for (size_t px = 0; px < pooling_width(); px++) {
                  const size_t ix = ox * stride_width() + px * dilation_width() - padding_left();
                  if (ix < next_input_width() && iy < next_input_height()) {
                    max_value = std::max(max_value,
                      input[((i * next_input_height() + iy) * next_input_width() + ix) * input_pixel_stride() + c]);
                  }
                }
              }
              max_value = std::min(max_value, int8_t(qmax() - 0x80));
              max_value = std::max(max_value, int8_t(qmin() - 0x80));
              next_output_ref[((i * next_output_height() + oy) * next_output_width() + ox) * channels() + c] = max_value;
            }
          }
        }
      }

      // Setup and run Max Pooling operator the second time, and destroy the operator.
      ASSERT_EQ(xnn_status_success,
        xnn_reshape_max_pooling2d_nhwc_s8(
          max_pooling_op,
          next_batch_size(), next_input_height(), next_input_width(),
          channels(), input_pixel_stride(), output_pixel_stride(),
          /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
          /*threadpool=*/nullptr));
      ASSERT_EQ(xnn_status_success,
        xnn_setup_max_pooling2d_nhwc_s8(
          max_pooling_op,
          input.data(), output.data()));

      ASSERT_EQ(xnn_status_success,
        xnn_run_operator(max_pooling_op, /*threadpool=*/nullptr));

      // Verify results of the second run.
      for (size_t i = 0; i < next_batch_size(); i++) {
        for (size_t y = 0; y < next_output_height(); y++) {
          for (size_t x = 0; x < next_output_width(); x++) {
            for (size_t c = 0; c < channels(); c++) {
              EXPECT_LE(int32_t(output[((i * next_output_height() + y) * next_output_width() + x) * output_pixel_stride() + c]), int32_t(qmax() - 0x80));
              EXPECT_GE(int32_t(output[((i * next_output_height() + y) * next_output_width() + x) * output_pixel_stride() + c]), int32_t(qmin() - 0x80));
              EXPECT_EQ(int32_t(next_output_ref[((i * next_output_height() + y) * next_output_width() + x) * channels() + c]),
                int32_t(output[((i * next_output_height() + y) * next_output_width() + x) * output_pixel_stride() + c])) <<
                "in batch index " << i << ", pixel (" << y << ", " << x << "), channel " << c;
            }
          }
        }
      }
    }
  }

  void TestSetupU8() const {
    xnnpack::ReplicableRandomDevice rng;
    std::uniform_int_distribution<int32_t> u8dist(
      std::numeric_limits<uint8_t>::min(), std::numeric_limits<uint8_t>::max());

    std::vector<uint8_t> input(XNN_EXTRA_BYTES / sizeof(uint8_t) + std::max<size_t>(
      (batch_size() * input_height() * input_width() - 1) * input_pixel_stride() + channels(),
      (next_batch_size() * next_input_height() * next_input_width() - 1) * input_pixel_stride() + channels()));
    std::vector<uint8_t> output(XNN_EXTRA_BYTES / sizeof(uint8_t) + std::max<size_t>(
      (batch_size() * output_height() * output_width() - 1) * output_pixel_stride() + channels(),
      (next_batch_size() * next_output_height() * next_output_width() - 1) * output_pixel_stride() + channels()));
    std::vector<float> output_ref(batch_size() * output_height() * output_width() * channels());
    std::vector<float> next_output_ref(next_batch_size() * next_output_height() * next_output_width() * channels());
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(input.begin(), input.end(), [&]() { return u8dist(rng); });
      std::fill(output.begin(), output.end(), 0xA5);

      // Compute reference results.
      for (size_t i = 0; i < batch_size(); i++) {
        for (size_t oy = 0; oy < output_height(); oy++) {
          for (size_t ox = 0; ox < output_width(); ox++) {
            for (size_t c = 0; c < channels(); c++) {
              uint8_t max_value = 0;
              for (size_t py = 0; py < pooling_height(); py++) {
                const size_t iy = oy * stride_height() + py * dilation_height() - padding_top();
                for (size_t px = 0; px < pooling_width(); px++) {
                  const size_t ix = ox * stride_width() + px * dilation_width() - padding_left();
                  if (ix < input_width() && iy < input_height()) {
                    max_value = std::max(max_value,
                      input[((i * input_height() + iy) * input_width() + ix) * input_pixel_stride() + c]);
                  }
                }
              }
              max_value = std::min(max_value, qmax());
              max_value = std::max(max_value, qmin());
              output_ref[((i * output_height() + oy) * output_width() + ox) * channels() + c] = max_value;
            }
          }
        }
      }

      // Create, setup, and run Max Pooling operator once.
      ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
      xnn_operator_t max_pooling_op = nullptr;

      ASSERT_EQ(xnn_status_success,
        xnn_create_max_pooling2d_nhwc_u8(
          padding_top(), padding_right(), padding_bottom(), padding_left(),
          pooling_height(), pooling_width(),
          stride_height(), stride_width(),
          dilation_height(), dilation_width(),
          qmin(), qmax(),
          0, &max_pooling_op));
      ASSERT_NE(nullptr, max_pooling_op);

      // Smart pointer to automatically delete max_pooling_op.
      std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_max_pooling_op(max_pooling_op, xnn_delete_operator);

      ASSERT_EQ(xnn_status_success,
        xnn_reshape_max_pooling2d_nhwc_u8(
          max_pooling_op,
          batch_size(), input_height(), input_width(),
          channels(), input_pixel_stride(), output_pixel_stride(),
          /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
          /*threadpool=*/nullptr));

      ASSERT_EQ(xnn_status_success,
        xnn_setup_max_pooling2d_nhwc_u8(
          max_pooling_op,
          input.data(), output.data()));

      ASSERT_EQ(xnn_status_success,
        xnn_run_operator(max_pooling_op, /*threadpool=*/nullptr));

      // Verify results of the first run.
      for (size_t i = 0; i < batch_size(); i++) {
        for (size_t y = 0; y < output_height(); y++) {
          for (size_t x = 0; x < output_width(); x++) {
            for (size_t c = 0; c < channels(); c++) {
              EXPECT_LE(uint32_t(output[((i * output_height() + y) * output_width() + x) * output_pixel_stride() + c]), uint32_t(qmax()));
              EXPECT_GE(uint32_t(output[((i * output_height() + y) * output_width() + x) * output_pixel_stride() + c]), uint32_t(qmin()));
              EXPECT_EQ(uint32_t(output_ref[((i * output_height() + y) * output_width() + x) * channels() + c]),
                uint32_t(output[((i * output_height() + y) * output_width() + x) * output_pixel_stride() + c])) <<
                "in batch index " << i << ", pixel (" << y << ", " << x << "), channel " << c;
            }
          }
        }
      }

      // Re-generate data for the second run.
      std::generate(input.begin(), input.end(), [&]() { return u8dist(rng); });
      std::fill(output.begin(), output.end(), 0xA5);

      // Compute reference results for the second run.
      for (size_t i = 0; i < next_batch_size(); i++) {
        for (size_t oy = 0; oy < next_output_height(); oy++) {
          for (size_t ox = 0; ox < next_output_width(); ox++) {
            for (size_t c = 0; c < channels(); c++) {
              uint8_t max_value = 0;
              for (size_t py = 0; py < pooling_height(); py++) {
                const size_t iy = oy * stride_height() + py * dilation_height() - padding_top();
                for (size_t px = 0; px < pooling_width(); px++) {
                  const size_t ix = ox * stride_width() + px * dilation_width() - padding_left();
                  if (ix < next_input_width() && iy < next_input_height()) {
                    max_value = std::max(max_value,
                      input[((i * next_input_height() + iy) * next_input_width() + ix) * input_pixel_stride() + c]);
                  }
                }
              }
              max_value = std::min(max_value, qmax());
              max_value = std::max(max_value, qmin());
              next_output_ref[((i * next_output_height() + oy) * next_output_width() + ox) * channels() + c] = max_value;
            }
          }
        }
      }

      // Setup and run Max Pooling operator the second time, and destroy the operator.
      ASSERT_EQ(xnn_status_success,
        xnn_reshape_max_pooling2d_nhwc_u8(
          max_pooling_op,
          next_batch_size(), next_input_height(), next_input_width(),
          channels(), input_pixel_stride(), output_pixel_stride(),
          /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
          /*threadpool=*/nullptr));
      ASSERT_EQ(xnn_status_success,
        xnn_setup_max_pooling2d_nhwc_u8(
          max_pooling_op,
          input.data(), output.data()));

      ASSERT_EQ(xnn_status_success,
        xnn_run_operator(max_pooling_op, /*threadpool=*/nullptr));

      // Verify results of the second run.
      for (size_t i = 0; i < next_batch_size(); i++) {
        for (size_t y = 0; y < next_output_height(); y++) {
          for (size_t x = 0; x < next_output_width(); x++) {
            for (size_t c = 0; c < channels(); c++) {
              EXPECT_LE(uint32_t(output[((i * next_output_height() + y) * next_output_width() + x) * output_pixel_stride() + c]), uint32_t(qmax()));
              EXPECT_GE(uint32_t(output[((i * next_output_height() + y) * next_output_width() + x) * output_pixel_stride() + c]), uint32_t(qmin()));
              EXPECT_EQ(uint32_t(next_output_ref[((i * next_output_height() + y) * next_output_width() + x) * channels() + c]),
                uint32_t(output[((i * next_output_height() + y) * next_output_width() + x) * output_pixel_stride() + c])) <<
                "in batch index " << i << ", pixel (" << y << ", " << x << "), channel " << c;
            }
          }
        }
      }
    }
  }

  void TestSetupF16() const {
    xnnpack::ReplicableRandomDevice rng;
    // Note: we need to avoid FP16 denormals in the generated tensor because they might be processed differently in
    // native vs emulated arithmetics, and we use exact comparison to verify the results against reference.
    std::uniform_real_distribution<float> f32dist(0.001f, 1.0f);

    std::vector<uint16_t> input(XNN_EXTRA_BYTES / sizeof(uint16_t) + std::max<size_t>(
      (batch_size() * input_height() * input_width() - 1) * input_pixel_stride() + channels(),
      (next_batch_size() * next_input_height() * next_input_width() - 1) * input_pixel_stride() + channels()));
    std::vector<uint16_t> output(XNN_EXTRA_BYTES / sizeof(uint16_t) + std::max<size_t>(
      (batch_size() * output_height() * output_width() - 1) * output_pixel_stride() + channels(),
      (next_batch_size() * next_output_height() * next_output_width() - 1) * output_pixel_stride() + channels()));
    std::vector<float> output_ref(batch_size() * output_height() * output_width() * channels());
    std::vector<float> next_output_ref(next_batch_size() * next_output_height() * next_output_width() * channels());
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(input.begin(), input.end(), [&]() { return fp16_ieee_from_fp32_value(f32dist(rng)); });
      std::fill(output.begin(), output.end(), UINT16_C(0x7E00) /* NaN */);

      // Compute reference results, without clamping.
      for (size_t i = 0; i < batch_size(); i++) {
        for (size_t oy = 0; oy < output_height(); oy++) {
          for (size_t ox = 0; ox < output_width(); ox++) {
            for (size_t c = 0; c < channels(); c++) {
              float max_value = -std::numeric_limits<float>::infinity();
              for (size_t py = 0; py < pooling_height(); py++) {
                const size_t iy = oy * stride_height() + py * dilation_height() - padding_top();
                for (size_t px = 0; px < pooling_width(); px++) {
                  const size_t ix = ox * stride_width() + px * dilation_width() - padding_left();
                  if (ix < input_width() && iy < input_height()) {
                    max_value = std::max(max_value,
                      fp16_ieee_to_fp32_value(input[((i * input_height() + iy) * input_width() + ix) * input_pixel_stride() + c]));
                  }
                }
              }
              output_ref[((i * output_height() + oy) * output_width() + ox) * channels() + c] = max_value;
            }
          }
        }
      }

      // Compute clamping parameters.
      const float accumulated_min = *std::min_element(output_ref.cbegin(), output_ref.cend());
      const float accumulated_max = *std::max_element(output_ref.cbegin(), output_ref.cend());
      const float accumulated_range = accumulated_max - accumulated_min;
      float output_min = accumulated_min + accumulated_range / 255.0f * float(qmin());
      float output_max = accumulated_max - accumulated_range / 255.0f * float(255 - qmax());
      output_min = fp16_ieee_to_fp32_value(fp16_ieee_from_fp32_value(output_min));
      output_max = fp16_ieee_to_fp32_value(fp16_ieee_from_fp32_value(output_max));
      if (accumulated_range == 0.0f) {
        output_min = -std::numeric_limits<float>::infinity();
        output_max = +std::numeric_limits<float>::infinity();
      }
      if (qmin() == std::numeric_limits<uint8_t>::min()) {
        output_min = -std::numeric_limits<float>::infinity();
      }
      if (qmax() == std::numeric_limits<uint8_t>::max()) {
        output_max = +std::numeric_limits<float>::infinity();
      }

      // Clamp reference results.
      for (float& value : output_ref) {
        value = std::max(std::min(value, output_max), output_min);
      }

      // Create, setup, and run Max Pooling operator once.
      ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
      xnn_operator_t max_pooling_op = nullptr;

      const xnn_status status = xnn_create_max_pooling2d_nhwc_f16(
          padding_top(), padding_right(), padding_bottom(), padding_left(),
          pooling_height(), pooling_width(),
          stride_height(), stride_width(),
          dilation_height(), dilation_width(),
          output_min, output_max,
          0, &max_pooling_op);
      if (status == xnn_status_unsupported_hardware) {
        GTEST_SKIP();
      }
      ASSERT_EQ(xnn_status_success, status);
      ASSERT_NE(nullptr, max_pooling_op);

      // Smart pointer to automatically delete max_pooling_op.
      std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_max_pooling_op(max_pooling_op, xnn_delete_operator);

      ASSERT_EQ(xnn_status_success,
        xnn_reshape_max_pooling2d_nhwc_f16(
          max_pooling_op,
          batch_size(), input_height(), input_width(),
          channels(), input_pixel_stride(), output_pixel_stride(),
          /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
          /*threadpool=*/nullptr));
      ASSERT_EQ(xnn_status_success,
        xnn_setup_max_pooling2d_nhwc_f16(
          max_pooling_op,
          input.data(), output.data()));

      ASSERT_EQ(xnn_status_success,
        xnn_run_operator(max_pooling_op, /*threadpool=*/nullptr));

      // Verify results of the first run.
      for (size_t i = 0; i < batch_size(); i++) {
        for (size_t y = 0; y < output_height(); y++) {
          for (size_t x = 0; x < output_width(); x++) {
            for (size_t c = 0; c < channels(); c++) {
              EXPECT_LE(fp16_ieee_to_fp32_value(output[((i * output_height() + y) * output_width() + x) * output_pixel_stride() + c]), output_max);
              EXPECT_GE(fp16_ieee_to_fp32_value(output[((i * output_height() + y) * output_width() + x) * output_pixel_stride() + c]), output_min);
              EXPECT_EQ(
                  fp16_ieee_to_fp32_value(output[((i * output_height() + y) * output_width() + x) * output_pixel_stride() + c]),
                  output_ref[((i * output_height() + y) * output_width() + x) * channels() + c]) <<
                "in batch index " << i << ", pixel (" << y << ", " << x << "), channel " << c
                << ", min = " << output_min << ", max = " << output_max;
            }
          }
        }
      }

      // Re-generate data for the second run.
      std::generate(input.begin(), input.end(), [&]() { return fp16_ieee_from_fp32_value(f32dist(rng)); });
      std::fill(output.begin(), output.end(), UINT16_C(0x7E00) /* NaN */);

      // Compute reference results for the second run, including clamping.
      for (size_t i = 0; i < next_batch_size(); i++) {
        for (size_t oy = 0; oy < next_output_height(); oy++) {
          for (size_t ox = 0; ox < next_output_width(); ox++) {
            for (size_t c = 0; c < channels(); c++) {
              float max_value = -std::numeric_limits<float>::infinity();
              for (size_t py = 0; py < pooling_height(); py++) {
                const size_t iy = oy * stride_height() + py * dilation_height() - padding_top();
                for (size_t px = 0; px < pooling_width(); px++) {
                  const size_t ix = ox * stride_width() + px * dilation_width() - padding_left();
                  if (ix < next_input_width() && iy < next_input_height()) {
                    max_value = std::max(max_value,
                      fp16_ieee_to_fp32_value(input[((i * next_input_height() + iy) * next_input_width() + ix) * input_pixel_stride() + c]));
                  }
                }
              }
              max_value = std::min(max_value, output_max);
              max_value = std::max(max_value, output_min);
              next_output_ref[((i * next_output_height() + oy) * next_output_width() + ox) * channels() + c] = max_value;
            }
          }
        }
      }

      // Setup and run Max Pooling operator the second time, and destroy the operator.
      ASSERT_EQ(xnn_status_success,
        xnn_reshape_max_pooling2d_nhwc_f16(
          max_pooling_op,
          next_batch_size(), next_input_height(), next_input_width(),
          channels(), input_pixel_stride(), output_pixel_stride(),
          /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
          /*threadpool=*/nullptr));
      ASSERT_EQ(xnn_status_success,
        xnn_setup_max_pooling2d_nhwc_f16(
          max_pooling_op,
          input.data(), output.data()));

      ASSERT_EQ(xnn_status_success,
        xnn_run_operator(max_pooling_op, /*threadpool=*/nullptr));

      // Verify results of the second run.
      for (size_t i = 0; i < next_batch_size(); i++) {
        for (size_t y = 0; y < next_output_height(); y++) {
          for (size_t x = 0; x < next_output_width(); x++) {
            for (size_t c = 0; c < channels(); c++) {
              EXPECT_LE(fp16_ieee_to_fp32_value(output[((i * next_output_height() + y) * next_output_width() + x) * output_pixel_stride() + c]), output_max);
              EXPECT_GE(fp16_ieee_to_fp32_value(output[((i * next_output_height() + y) * next_output_width() + x) * output_pixel_stride() + c]), output_min);
              EXPECT_EQ(
                  fp16_ieee_to_fp32_value(output[((i * next_output_height() + y) * next_output_width() + x) * output_pixel_stride() + c]),
                  next_output_ref[((i * next_output_height() + y) * next_output_width() + x) * channels() + c]) <<
                "in batch index " << i << ", pixel (" << y << ", " << x << "), channel " << c
                << ", min = " << output_min << ", max = " << output_max;
            }
          }
        }
      }
    }
  }

  void TestSetupF32() const {
    xnnpack::ReplicableRandomDevice rng;
    std::uniform_real_distribution<float> f32dist;

    std::vector<float> input(XNN_EXTRA_BYTES / sizeof(float) + std::max<size_t>(
      (batch_size() * input_height() * input_width() - 1) * input_pixel_stride() + channels(),
      (next_batch_size() * next_input_height() * next_input_width() - 1) * input_pixel_stride() + channels()));
    std::vector<float> output(XNN_EXTRA_BYTES / sizeof(float) + std::max<size_t>(
      (batch_size() * output_height() * output_width() - 1) * output_pixel_stride() + channels(),
      (next_batch_size() * next_output_height() * next_output_width() - 1) * output_pixel_stride() + channels()));
    std::vector<float> output_ref(batch_size() * output_height() * output_width() * channels());
    std::vector<float> next_output_ref(next_batch_size() * next_output_height() * next_output_width() * channels());
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(input.begin(), input.end(), [&]() { return f32dist(rng); });
      std::fill(output.begin(), output.end(), nanf(""));

      // Compute reference results, without clamping.
      for (size_t i = 0; i < batch_size(); i++) {
        for (size_t oy = 0; oy < output_height(); oy++) {
          for (size_t ox = 0; ox < output_width(); ox++) {
            for (size_t c = 0; c < channels(); c++) {
              float max_value = -std::numeric_limits<float>::infinity();
              for (size_t py = 0; py < pooling_height(); py++) {
                const size_t iy = oy * stride_height() + py * dilation_height() - padding_top();
                for (size_t px = 0; px < pooling_width(); px++) {
                  const size_t ix = ox * stride_width() + px * dilation_width() - padding_left();
                  if (ix < input_width() && iy < input_height()) {
                    max_value = std::max(max_value,
                      input[((i * input_height() + iy) * input_width() + ix) * input_pixel_stride() + c]);
                  }
                }
              }
              output_ref[((i * output_height() + oy) * output_width() + ox) * channels() + c] = max_value;
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

      // Create, setup, and run Max Pooling operator once.
      ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
      xnn_operator_t max_pooling_op = nullptr;

      ASSERT_EQ(xnn_status_success,
        xnn_create_max_pooling2d_nhwc_f32(
          padding_top(), padding_right(), padding_bottom(), padding_left(),
          pooling_height(), pooling_width(),
          stride_height(), stride_width(),
          dilation_height(), dilation_width(),
          output_min, output_max,
          0, &max_pooling_op));
      ASSERT_NE(nullptr, max_pooling_op);

      // Smart pointer to automatically delete max_pooling_op.
      std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_max_pooling_op(max_pooling_op, xnn_delete_operator);

      ASSERT_EQ(xnn_status_success,
        xnn_reshape_max_pooling2d_nhwc_f32(
          max_pooling_op,
          batch_size(), input_height(), input_width(),
          channels(), input_pixel_stride(), output_pixel_stride(),
          /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
          /*threadpool=*/nullptr));
      ASSERT_EQ(xnn_status_success,
        xnn_setup_max_pooling2d_nhwc_f32(
          max_pooling_op,
          input.data(), output.data()));

      ASSERT_EQ(xnn_status_success,
        xnn_run_operator(max_pooling_op, /*threadpool=*/nullptr));

      // Verify results of the first run.
      for (size_t i = 0; i < batch_size(); i++) {
        for (size_t y = 0; y < output_height(); y++) {
          for (size_t x = 0; x < output_width(); x++) {
            for (size_t c = 0; c < channels(); c++) {
              EXPECT_LE(output[((i * output_height() + y) * output_width() + x) * output_pixel_stride() + c], output_max);
              EXPECT_GE(output[((i * output_height() + y) * output_width() + x) * output_pixel_stride() + c], output_min);
              EXPECT_EQ(output_ref[((i * output_height() + y) * output_width() + x) * channels() + c],
                output[((i * output_height() + y) * output_width() + x) * output_pixel_stride() + c]) <<
                "in batch index " << i << ", pixel (" << y << ", " << x << "), channel " << c;
            }
          }
        }
      }

      // Re-generate data for the second run.
      std::generate(input.begin(), input.end(), [&]() { return f32dist(rng); });
      std::fill(output.begin(), output.end(), std::nanf(""));

      // Compute reference results for the second run, including clamping.
      for (size_t i = 0; i < next_batch_size(); i++) {
        for (size_t oy = 0; oy < next_output_height(); oy++) {
          for (size_t ox = 0; ox < next_output_width(); ox++) {
            for (size_t c = 0; c < channels(); c++) {
              float max_value = -std::numeric_limits<float>::infinity();
              for (size_t py = 0; py < pooling_height(); py++) {
                const size_t iy = oy * stride_height() + py * dilation_height() - padding_top();
                for (size_t px = 0; px < pooling_width(); px++) {
                  const size_t ix = ox * stride_width() + px * dilation_width() - padding_left();
                  if (ix < next_input_width() && iy < next_input_height()) {
                    max_value = std::max(max_value,
                      input[((i * next_input_height() + iy) * next_input_width() + ix) * input_pixel_stride() + c]);
                  }
                }
              }
              max_value = std::min(max_value, output_max);
              max_value = std::max(max_value, output_min);
              next_output_ref[((i * next_output_height() + oy) * next_output_width() + ox) * channels() + c] = max_value;
            }
          }
        }
      }

      // Setup and run Max Pooling operator the second time, and destroy the operator.
      ASSERT_EQ(xnn_status_success,
        xnn_reshape_max_pooling2d_nhwc_f32(
          max_pooling_op,
          next_batch_size(), next_input_height(), next_input_width(),
          channels(), input_pixel_stride(), output_pixel_stride(),
          /*output_height_out=*/nullptr, /*output_width_out=*/nullptr,
          /*threadpool=*/nullptr));
      ASSERT_EQ(xnn_status_success,
        xnn_setup_max_pooling2d_nhwc_f32(
          max_pooling_op,
          input.data(), output.data()));

      ASSERT_EQ(xnn_status_success,
        xnn_run_operator(max_pooling_op, /*threadpool=*/nullptr));

      // Verify results of the second run.
      for (size_t i = 0; i < next_batch_size(); i++) {
        for (size_t y = 0; y < next_output_height(); y++) {
          for (size_t x = 0; x < next_output_width(); x++) {
            for (size_t c = 0; c < channels(); c++) {
              EXPECT_LE(output[((i * next_output_height() + y) * next_output_width() + x) * output_pixel_stride() + c], output_max);
              EXPECT_GE(output[((i * next_output_height() + y) * next_output_width() + x) * output_pixel_stride() + c], output_min);
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
  uint32_t dilation_height_{1};
  uint32_t dilation_width_{1};
  size_t next_input_height_{0};
  size_t next_input_width_{0};
  size_t next_batch_size_{0};
  uint8_t qmin_{0};
  uint8_t qmax_{255};
  size_t iterations_{1};
};
