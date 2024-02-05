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
#include <numeric>
#include <random>
#include <vector>

#include <fp16/fp16.h>

#include <xnnpack.h>
#include <xnnpack/aligned-allocator.h>
#include <xnnpack/common.h>


class ResizeBilinearOperatorTester {
 public:
  inline ResizeBilinearOperatorTester& input_size(size_t input_height, size_t input_width) {
    assert(input_height >= 1);
    assert(input_width >= 1);
    this->input_height_ = input_height;
    this->input_width_ = input_width;
    return *this;
  }

  inline ResizeBilinearOperatorTester& input_height(size_t input_height) {
    assert(input_height >= 1);
    this->input_height_ = input_height;
    return *this;
  }

  inline size_t input_height() const {
    return this->input_height_;
  }

  inline ResizeBilinearOperatorTester& input_width(size_t input_width) {
    assert(input_width >= 1);
    this->input_width_ = input_width;
    return *this;
  }

  inline size_t input_width() const {
    return this->input_width_;
  }

  inline ResizeBilinearOperatorTester& output_size(size_t output_height, size_t output_width) {
    assert(output_height >= 1);
    assert(output_width >= 1);
    this->output_height_ = output_height;
    this->output_width_ = output_width;
    return *this;
  }

  inline ResizeBilinearOperatorTester& output_height(size_t output_height) {
    assert(output_height >= 1);
    this->output_height_ = output_height;
    return *this;
  }

  inline size_t output_height() const {
    return this->output_height_;
  }

  inline ResizeBilinearOperatorTester& output_width(size_t output_width) {
    assert(output_width >= 1);
    this->output_width_ = output_width;
    return *this;
  }

  inline size_t output_width() const {
    return this->output_width_;
  }

  inline float height_scale() const {
    if (align_corners() && output_height() > 1) {
      return float(input_height() - 1) / float(output_height() - 1);
    } else {
      return float(input_height()) / float(output_height());
    }
  }

  inline float width_scale() const {
    if (align_corners() && output_width() > 1) {
      return float(input_width() - 1) / float(output_width() - 1);
    } else {
      return float(input_width()) / float(output_width());
    }
  }

  inline ResizeBilinearOperatorTester& channels(size_t channels) {
    assert(channels != 0);
    this->channels_ = channels;
    return *this;
  }

  inline size_t channels() const {
    return this->channels_;
  }

  inline ResizeBilinearOperatorTester& batch_size(size_t batch_size) {
    assert(batch_size != 0);
    this->batch_size_ = batch_size;
    return *this;
  }

  inline size_t batch_size() const {
    return this->batch_size_;
  }

  inline ResizeBilinearOperatorTester& input_pixel_stride(size_t input_pixel_stride) {
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

  inline ResizeBilinearOperatorTester& output_pixel_stride(size_t output_pixel_stride) {
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

  inline ResizeBilinearOperatorTester& next_input_size(uint32_t next_input_height, uint32_t next_input_width) {
    assert(next_input_height >= 1);
    assert(next_input_width >= 1);
    this->next_input_height_ = next_input_height;
    this->next_input_width_ = next_input_width;
    return *this;
  }

  inline ResizeBilinearOperatorTester& next_input_height(uint32_t next_input_height) {
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

  inline ResizeBilinearOperatorTester& next_input_width(uint32_t next_input_width) {
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

  inline ResizeBilinearOperatorTester& next_batch_size(size_t next_batch_size) {
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

  inline ResizeBilinearOperatorTester& align_corners(bool align_corners) {
    this->align_corners_ = align_corners;
    return *this;
  }

  inline bool align_corners() const {
    return this->align_corners_;
  }

  inline ResizeBilinearOperatorTester& tf_legacy_mode(bool tf_legacy_mode) {
    this->tf_legacy_mode_ = tf_legacy_mode;
    return *this;
  }

  inline bool tf_legacy_mode() const {
    return this->tf_legacy_mode_;
  }

  inline ResizeBilinearOperatorTester& iterations(size_t iterations) {
    this->iterations_ = iterations;
    return *this;
  }

  inline size_t iterations() const {
    return this->iterations_;
  }

  inline ResizeBilinearOperatorTester& transient_indirection_buffer(bool transient_indirection_buffer) {
    this->transient_indirection_buffer_ = transient_indirection_buffer;
    return *this;
  }

  inline bool transient_indirection_buffer() const {
    return this->transient_indirection_buffer_;
  }

  void TestNHWCxF16() const {
    if (align_corners()) {
      ASSERT_FALSE(tf_legacy_mode());
    }

    std::random_device random_device;
    auto rng = std::mt19937(random_device());
    std::uniform_real_distribution<float> f32dist;

    std::vector<uint16_t> input(XNN_EXTRA_BYTES / sizeof(uint16_t) +
      (batch_size() * input_height() * input_width() - 1) * input_pixel_stride() + channels());
    std::vector<uint16_t> output((batch_size() * output_height() * output_width() - 1) * output_pixel_stride() + channels());
    std::vector<float> output_ref(batch_size() * output_height() * output_width() * channels());
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(input.begin(), input.end(), [&]() { return fp16_ieee_from_fp32_value(f32dist(rng)); });
      std::fill(output.begin(), output.end(), UINT16_C(0x7E00) /* NaN */);

      // Compute reference results.
      const float offset = (tf_legacy_mode() || align_corners()) ? 0.0f : 0.5f;
      for (size_t batch_index = 0; batch_index < batch_size(); batch_index++) {
        for (size_t output_y = 0; output_y < output_height(); output_y++) {
          const float input_y = (float(output_y) + offset) * height_scale() - offset;
          const int64_t input_y_top = std::max<int64_t>(int64_t(std::floor(input_y)), 0);
          const int64_t input_y_bottom = std::min<int64_t>(int64_t(std::ceil(input_y)), input_height() - 1);
          const float y_alpha = fp16_ieee_to_fp32_value(fp16_ieee_from_fp32_value(input_y - std::floor(input_y)));
          for (size_t output_x = 0; output_x < output_width(); output_x++) {
            const float input_x = (float(output_x) + offset) * width_scale() - offset;
            const int64_t input_x_left = std::max<int64_t>(int64_t(std::floor(input_x)), 0);
            const int64_t input_x_right = std::min<int64_t>(int64_t(std::ceil(input_x)), input_width() - 1);
            const float x_alpha = fp16_ieee_to_fp32_value(fp16_ieee_from_fp32_value(input_x - std::floor(input_x)));
            for (size_t c = 0; c < channels(); c++) {
              output_ref[((batch_index * output_height() + output_y) * output_width() + output_x) * channels() + c] =
                fp16_ieee_to_fp32_value(input[((batch_index * input_height() + input_y_top) * input_width() + input_x_left) * input_pixel_stride() + c]) * (1.0f - y_alpha) * (1.0f - x_alpha) +
                fp16_ieee_to_fp32_value(input[((batch_index * input_height() + input_y_top) * input_width() + input_x_right) * input_pixel_stride() + c]) * (1.0f - y_alpha) * x_alpha +
                fp16_ieee_to_fp32_value(input[((batch_index * input_height() + input_y_bottom) * input_width() + input_x_left) * input_pixel_stride() + c]) * y_alpha * (1.0f - x_alpha) +
                fp16_ieee_to_fp32_value(input[((batch_index * input_height() + input_y_bottom) * input_width() + input_x_right) * input_pixel_stride() + c]) * y_alpha * x_alpha;
            }
          }
        }
      }

      // Create, setup, run, and destroy Resize Bilinear operator.
      ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
      xnn_operator_t resize_bilinear_op = nullptr;

      uint32_t flags = 0;
      if (align_corners()) {
        flags |= XNN_FLAG_ALIGN_CORNERS;
      }
      if (tf_legacy_mode()) {
        flags |= XNN_FLAG_TENSORFLOW_LEGACY_MODE;
      }
      if (transient_indirection_buffer()) {
        flags |= XNN_FLAG_TRANSIENT_INDIRECTION_BUFFER;
      }
      const xnn_status status = xnn_create_resize_bilinear2d_nhwc_f16(
          output_height(), output_width(), flags,
          &resize_bilinear_op);
      if (status == xnn_status_unsupported_hardware) {
        GTEST_SKIP();
      }
      ASSERT_EQ(xnn_status_success, status);
      ASSERT_NE(nullptr, resize_bilinear_op);

      // Smart pointer to automatically delete resize_bilinear_op.
      std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_resize_bilinear_op(resize_bilinear_op, xnn_delete_operator);

      size_t workspace_size = transient_indirection_buffer() ? 0 : SIZE_MAX;
      size_t workspace_alignment = SIZE_MAX;
      ASSERT_EQ(xnn_status_success,
        xnn_reshape_resize_bilinear2d_nhwc_f16(
          resize_bilinear_op,
          batch_size(), input_height(), input_width(),
          channels(), input_pixel_stride(), output_pixel_stride(),
          &workspace_size, &workspace_alignment,
          /*threadpool=*/nullptr));
      std::vector<char, AlignedAllocator<char, XNN_ALLOCATION_ALIGNMENT>> workspace(workspace_size);
      std::iota(workspace.begin(), workspace.end(), 0);
      if (transient_indirection_buffer()) {
        ASSERT_NE(workspace_size, 0);
        ASSERT_EQ(workspace_alignment, XNN_ALLOCATION_ALIGNMENT);
        ASSERT_EQ(xnn_status_success,
          xnn_setup_resize_bilinear2d_nhwc_f16(
            resize_bilinear_op,
            workspace.data(), input.data(), output.data()));
      } else {
        ASSERT_EQ(workspace_size, 0);
        ASSERT_EQ(workspace_alignment, 1);
        ASSERT_EQ(xnn_status_success,
          xnn_setup_resize_bilinear2d_nhwc_f16(
            resize_bilinear_op,
            /*workspace=*/nullptr, input.data(), output.data()));
      }

      ASSERT_EQ(xnn_status_success,
        xnn_run_operator(resize_bilinear_op, /*threadpool=*/nullptr));

      // Verify results.
      for (size_t i = 0; i < batch_size(); i++) {
        for (size_t y = 0; y < output_height(); y++) {
          for (size_t x = 0; x < output_width(); x++) {
            for (size_t c = 0; c < channels(); c++) {
              ASSERT_NEAR(
                  fp16_ieee_to_fp32_value(output[((i * output_height() + y) * output_width() + x) * output_pixel_stride() + c]),
                  output_ref[((i * output_height() + y) * output_width() + x) * channels() + c],
                  std::max(1.0e-4f, std::abs(output_ref[((i * output_height() + y) * output_width() + x) * channels() + c]) * 1.0e-2f)) <<
                "in batch index " << i << ", pixel (" << y << ", " << x << "), channel " << c;
            }
          }
        }
      }
    }
  }

  void TestNHWCxF32() const {
    if (align_corners()) {
      ASSERT_FALSE(tf_legacy_mode());
    }

    std::random_device random_device;
    auto rng = std::mt19937(random_device());
    std::uniform_real_distribution<float> f32dist;

    std::vector<float> input((batch_size() * input_height() * input_width() - 1) * input_pixel_stride() + channels() + XNN_EXTRA_BYTES / sizeof(float));
    std::vector<float> output((batch_size() * output_height() * output_width() - 1) * output_pixel_stride() + channels());
    std::vector<float> output_ref(batch_size() * output_height() * output_width() * channels());
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(input.begin(), input.end(), [&]() { return f32dist(rng); });
      std::fill(output.begin(), output.end(), std::nanf(""));

      // Compute reference results.
      const float offset = (tf_legacy_mode() || align_corners()) ? 0.0f : 0.5f;
      for (size_t batch_index = 0; batch_index < batch_size(); batch_index++) {
        for (size_t output_y = 0; output_y < output_height(); output_y++) {
          const float input_y = (float(output_y) + offset) * height_scale() - offset;
          const int64_t input_y_top = std::max<int64_t>(int64_t(std::floor(input_y)), 0);
          const int64_t input_y_bottom = std::min<int64_t>(int64_t(std::ceil(input_y)), input_height() - 1);
          const float y_alpha = input_y - std::floor(input_y);
          for (size_t output_x = 0; output_x < output_width(); output_x++) {
            const float input_x = (float(output_x) + offset) * width_scale() - offset;
            const int64_t input_x_left = std::max<int64_t>(int64_t(std::floor(input_x)), 0);
            const int64_t input_x_right = std::min<int64_t>(int64_t(std::ceil(input_x)), input_width() - 1);
            const float x_alpha = input_x - std::floor(input_x);
            for (size_t c = 0; c < channels(); c++) {
              output_ref[((batch_index * output_height() + output_y) * output_width() + output_x) * channels() + c] =
                input[((batch_index * input_height() + input_y_top) * input_width() + input_x_left) * input_pixel_stride() + c] * (1.0f - y_alpha) * (1.0f - x_alpha) +
                input[((batch_index * input_height() + input_y_top) * input_width() + input_x_right) * input_pixel_stride() + c] * (1.0f - y_alpha) * x_alpha +
                input[((batch_index * input_height() + input_y_bottom) * input_width() + input_x_left) * input_pixel_stride() + c] * y_alpha * (1.0f - x_alpha) +
                input[((batch_index * input_height() + input_y_bottom) * input_width() + input_x_right) * input_pixel_stride() + c] * y_alpha * x_alpha;
            }
          }
        }
      }

      // Create, setup, run, and destroy Resize Bilinear operator.
      ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
      xnn_operator_t resize_bilinear_op = nullptr;

      uint32_t flags = 0;
      if (align_corners()) {
        flags |= XNN_FLAG_ALIGN_CORNERS;
      }
      if (tf_legacy_mode()) {
        flags |= XNN_FLAG_TENSORFLOW_LEGACY_MODE;
      }
      if (transient_indirection_buffer()) {
        flags |= XNN_FLAG_TRANSIENT_INDIRECTION_BUFFER;
      }
      ASSERT_EQ(xnn_status_success,
        xnn_create_resize_bilinear2d_nhwc_f32(
          output_height(), output_width(), flags,
          &resize_bilinear_op));
      ASSERT_NE(nullptr, resize_bilinear_op);

      // Smart pointer to automatically delete resize_bilinear_op.
      std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_resize_bilinear_op(resize_bilinear_op, xnn_delete_operator);

      size_t workspace_size = transient_indirection_buffer() ? 0 : SIZE_MAX;
      size_t workspace_alignment = SIZE_MAX;
      ASSERT_EQ(xnn_status_success,
        xnn_reshape_resize_bilinear2d_nhwc_f32(
          resize_bilinear_op,
          batch_size(), input_height(), input_width(),
          channels(), input_pixel_stride(), output_pixel_stride(),
          &workspace_size, &workspace_alignment,
          /*threadpool=*/nullptr));

      std::vector<char, AlignedAllocator<char, XNN_ALLOCATION_ALIGNMENT>> workspace(workspace_size);
      std::iota(workspace.begin(), workspace.end(), 0);
      if (transient_indirection_buffer()) {
        ASSERT_NE(workspace_size, 0);
        ASSERT_EQ(workspace_alignment, XNN_ALLOCATION_ALIGNMENT);
        ASSERT_EQ(xnn_status_success,
          xnn_setup_resize_bilinear2d_nhwc_f32(
            resize_bilinear_op,
            workspace.data(), input.data(), output.data()));
      } else {
        ASSERT_EQ(workspace_size, 0);
        ASSERT_EQ(workspace_alignment, 1);
        ASSERT_EQ(xnn_status_success,
          xnn_setup_resize_bilinear2d_nhwc_f32(
            resize_bilinear_op,
            /*workspace=*/nullptr, input.data(), output.data()));
      }

      ASSERT_EQ(xnn_status_success,
        xnn_run_operator(resize_bilinear_op, /*threadpool=*/nullptr));

      // Verify results.
      for (size_t i = 0; i < batch_size(); i++) {
        for (size_t y = 0; y < output_height(); y++) {
          for (size_t x = 0; x < output_width(); x++) {
            for (size_t c = 0; c < channels(); c++) {
              ASSERT_NEAR(output[((i * output_height() + y) * output_width() + x) * output_pixel_stride() + c],
                  output_ref[((i * output_height() + y) * output_width() + x) * channels() + c],
                  std::abs(output_ref[((i * output_height() + y) * output_width() + x) * channels() + c]) * 1.0e-5f) <<
                "in batch index " << i << ", pixel (" << y << ", " << x << "), channel " << c;
            }
          }
        }
      }
    }
  }

  void TestNHWCxS8() const {
    if (align_corners()) {
      ASSERT_FALSE(tf_legacy_mode());
    }

    std::random_device random_device;
    auto rng = std::mt19937(random_device());
    std::uniform_int_distribution<int32_t> i8dist(
      std::numeric_limits<int8_t>::min(), std::numeric_limits<int8_t>::max());

    std::vector<int8_t> input((batch_size() * input_height() * input_width() - 1) * input_pixel_stride() + channels() + XNN_EXTRA_BYTES / sizeof(int8_t));
    std::vector<int8_t> output((batch_size() * output_height() * output_width() - 1) * output_pixel_stride() + channels());
    std::vector<float> output_ref(batch_size() * output_height() * output_width() * channels());
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(input.begin(), input.end(), [&]() { return i8dist(rng); });
      std::fill(output.begin(), output.end(), INT8_C(0xA5));

      // Compute reference results.
      const float offset = (tf_legacy_mode() || align_corners()) ? 0.0f : 0.5f;
      for (size_t batch_index = 0; batch_index < batch_size(); batch_index++) {
        for (size_t output_y = 0; output_y < output_height(); output_y++) {
          const float input_y = (float(output_y) + offset) * height_scale() - offset;
          const int64_t input_y_top = std::max<int64_t>(int64_t(std::floor(input_y)), 0);
          const int64_t input_y_bottom = std::min<int64_t>(int64_t(std::ceil(input_y)), input_height() - 1);
          const float y_alpha = input_y - std::floor(input_y);
          for (size_t output_x = 0; output_x < output_width(); output_x++) {
            const float input_x = (float(output_x) + offset) * width_scale() - offset;
            const int64_t input_x_left = std::max<int64_t>(int64_t(std::floor(input_x)), 0);
            const int64_t input_x_right = std::min<int64_t>(int64_t(std::ceil(input_x)), input_width() - 1);
            const float x_alpha = input_x - std::floor(input_x);
            for (size_t c = 0; c < channels(); c++) {
              output_ref[((batch_index * output_height() + output_y) * output_width() + output_x) * channels() + c] =
                float(int32_t(input[((batch_index * input_height() + input_y_top) * input_width() + input_x_left) * input_pixel_stride() + c])) * (1.0f - y_alpha) * (1.0f - x_alpha) +
                float(int32_t(input[((batch_index * input_height() + input_y_top) * input_width() + input_x_right) * input_pixel_stride() + c])) * (1.0f - y_alpha) * x_alpha +
                float(int32_t(input[((batch_index * input_height() + input_y_bottom) * input_width() + input_x_left) * input_pixel_stride() + c])) * y_alpha * (1.0f - x_alpha) +
                float(int32_t(input[((batch_index * input_height() + input_y_bottom) * input_width() + input_x_right) * input_pixel_stride() + c])) * y_alpha * x_alpha;
            }
          }
        }
      }

      // Create, setup, run, and destroy Resize Bilinear operator.
      ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
      xnn_operator_t resize_bilinear_op = nullptr;

      uint32_t flags = 0;
      if (align_corners()) {
        flags |= XNN_FLAG_ALIGN_CORNERS;
      }
      if (tf_legacy_mode()) {
        flags |= XNN_FLAG_TENSORFLOW_LEGACY_MODE;
      }
      if (transient_indirection_buffer()) {
        flags |= XNN_FLAG_TRANSIENT_INDIRECTION_BUFFER;
      }
      ASSERT_EQ(xnn_status_success,
        xnn_create_resize_bilinear2d_nhwc_s8(
          output_height(), output_width(), flags,
          &resize_bilinear_op));
      ASSERT_NE(nullptr, resize_bilinear_op);

      // Smart pointer to automatically delete resize_bilinear_op.
      std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_resize_bilinear_op(resize_bilinear_op, xnn_delete_operator);

      size_t workspace_size = transient_indirection_buffer() ? 0 : SIZE_MAX;
      size_t workspace_alignment = SIZE_MAX;
      ASSERT_EQ(xnn_status_success,
        xnn_reshape_resize_bilinear2d_nhwc_s8(
          resize_bilinear_op,
          batch_size(), input_height(), input_width(),
          channels(), input_pixel_stride(), output_pixel_stride(),
          &workspace_size, &workspace_alignment,
          /*threadpool=*/nullptr));

      std::vector<char, AlignedAllocator<char, XNN_ALLOCATION_ALIGNMENT>> workspace(workspace_size);
      std::iota(workspace.begin(), workspace.end(), 0);
      if (transient_indirection_buffer()) {
        ASSERT_NE(workspace_size, 0);
        ASSERT_EQ(workspace_alignment, XNN_ALLOCATION_ALIGNMENT);
        ASSERT_EQ(xnn_status_success,
          xnn_setup_resize_bilinear2d_nhwc_s8(
            resize_bilinear_op,
            workspace.data(), input.data(), output.data()));
      } else {
        ASSERT_EQ(workspace_size, 0);
        ASSERT_EQ(workspace_alignment, 1);
        ASSERT_EQ(xnn_status_success,
          xnn_setup_resize_bilinear2d_nhwc_s8(
            resize_bilinear_op,
            /*workspace=*/nullptr, input.data(), output.data()));
      }

      ASSERT_EQ(xnn_status_success,
        xnn_run_operator(resize_bilinear_op, /*threadpool=*/nullptr));

      // Verify results.
      for (size_t i = 0; i < batch_size(); i++) {
        for (size_t y = 0; y < output_height(); y++) {
          for (size_t x = 0; x < output_width(); x++) {
            for (size_t c = 0; c < channels(); c++) {
              ASSERT_NEAR(
                  float(int32_t(output[((i * output_height() + y) * output_width() + x) * output_pixel_stride() + c])),
                  output_ref[((i * output_height() + y) * output_width() + x) * channels() + c],
                  0.6f) <<
                "in batch index " << i << ", pixel (" << y << ", " << x << "), channel " << c;
            }
          }
        }
      }
    }
  }

  void TestNHWCxU8() const {
    if (align_corners()) {
      ASSERT_FALSE(tf_legacy_mode());
    }

    std::random_device random_device;
    auto rng = std::mt19937(random_device());
    std::uniform_int_distribution<int32_t> u8dist(
      std::numeric_limits<uint8_t>::min(), std::numeric_limits<uint8_t>::max());

    std::vector<uint8_t> input((batch_size() * input_height() * input_width() - 1) * input_pixel_stride() + channels() + XNN_EXTRA_BYTES / sizeof(uint8_t));
    std::vector<uint8_t> output((batch_size() * output_height() * output_width() - 1) * output_pixel_stride() + channels());
    std::vector<float> output_ref(batch_size() * output_height() * output_width() * channels());
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(input.begin(), input.end(), [&]() { return u8dist(rng); });
      std::fill(output.begin(), output.end(), UINT8_C(0xA5));

      // Compute reference results.
      const float offset = (tf_legacy_mode() || align_corners()) ? 0.0f : 0.5f;
      for (size_t batch_index = 0; batch_index < batch_size(); batch_index++) {
        for (size_t output_y = 0; output_y < output_height(); output_y++) {
          const float input_y = (float(output_y) + offset) * height_scale() - offset;
          const int64_t input_y_top = std::max<int64_t>(int64_t(std::floor(input_y)), 0);
          const int64_t input_y_bottom = std::min<int64_t>(int64_t(std::ceil(input_y)), input_height() - 1);
          const float y_alpha = input_y - std::floor(input_y);
          for (size_t output_x = 0; output_x < output_width(); output_x++) {
            const float input_x = (float(output_x) + offset) * width_scale() - offset;
            const int64_t input_x_left = std::max<int64_t>(int64_t(std::floor(input_x)), 0);
            const int64_t input_x_right = std::min<int64_t>(int64_t(std::ceil(input_x)), input_width() - 1);
            const float x_alpha = input_x - std::floor(input_x);
            for (size_t c = 0; c < channels(); c++) {
              output_ref[((batch_index * output_height() + output_y) * output_width() + output_x) * channels() + c] =
                float(int32_t(input[((batch_index * input_height() + input_y_top) * input_width() + input_x_left) * input_pixel_stride() + c])) * (1.0f - y_alpha) * (1.0f - x_alpha) +
                float(int32_t(input[((batch_index * input_height() + input_y_top) * input_width() + input_x_right) * input_pixel_stride() + c])) * (1.0f - y_alpha) * x_alpha +
                float(int32_t(input[((batch_index * input_height() + input_y_bottom) * input_width() + input_x_left) * input_pixel_stride() + c])) * y_alpha * (1.0f - x_alpha) +
                float(int32_t(input[((batch_index * input_height() + input_y_bottom) * input_width() + input_x_right) * input_pixel_stride() + c])) * y_alpha * x_alpha;
            }
          }
        }
      }

      // Create, setup, run, and destroy Resize Bilinear operator.
      ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
      xnn_operator_t resize_bilinear_op = nullptr;

      uint32_t flags = 0;
      if (align_corners()) {
        flags |= XNN_FLAG_ALIGN_CORNERS;
      }
      if (tf_legacy_mode()) {
        flags |= XNN_FLAG_TENSORFLOW_LEGACY_MODE;
      }
      if (transient_indirection_buffer()) {
        flags |= XNN_FLAG_TRANSIENT_INDIRECTION_BUFFER;
      }
      ASSERT_EQ(xnn_status_success,
        xnn_create_resize_bilinear2d_nhwc_u8(
          output_height(), output_width(), flags,
          &resize_bilinear_op));
      ASSERT_NE(nullptr, resize_bilinear_op);

      // Smart pointer to automatically delete resize_bilinear_op.
      std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_resize_bilinear_op(resize_bilinear_op, xnn_delete_operator);

      size_t workspace_size = transient_indirection_buffer() ? 0 : SIZE_MAX;
      size_t workspace_alignment = SIZE_MAX;
      ASSERT_EQ(xnn_status_success,
        xnn_reshape_resize_bilinear2d_nhwc_u8(
          resize_bilinear_op,
          batch_size(), input_height(), input_width(),
          channels(), input_pixel_stride(), output_pixel_stride(),
          &workspace_size, &workspace_alignment,
          /*threadpool=*/nullptr));

      std::vector<char, AlignedAllocator<char, XNN_ALLOCATION_ALIGNMENT>> workspace(workspace_size);
      std::iota(workspace.begin(), workspace.end(), 0);
      if (transient_indirection_buffer()) {
        ASSERT_NE(workspace_size, 0);
        ASSERT_EQ(workspace_alignment, XNN_ALLOCATION_ALIGNMENT);
        ASSERT_EQ(xnn_status_success,
          xnn_setup_resize_bilinear2d_nhwc_u8(
            resize_bilinear_op,
            workspace.data(), input.data(), output.data()));
      } else {
        ASSERT_EQ(workspace_size, 0);
        ASSERT_EQ(workspace_alignment, 1);
        ASSERT_EQ(xnn_status_success,
          xnn_setup_resize_bilinear2d_nhwc_u8(
            resize_bilinear_op,
            /*workspace=*/nullptr, input.data(), output.data()));
      }

      ASSERT_EQ(xnn_status_success,
        xnn_run_operator(resize_bilinear_op, /*threadpool=*/nullptr));

      // Verify results.
      for (size_t i = 0; i < batch_size(); i++) {
        for (size_t y = 0; y < output_height(); y++) {
          for (size_t x = 0; x < output_width(); x++) {
            for (size_t c = 0; c < channels(); c++) {
              ASSERT_NEAR(
                  float(int32_t(output[((i * output_height() + y) * output_width() + x) * output_pixel_stride() + c])),
                  output_ref[((i * output_height() + y) * output_width() + x) * channels() + c],
                  0.6f) <<
                "in batch index " << i << ", pixel (" << y << ", " << x << "), channel " << c;
            }
          }
        }
      }
    }
  }

  void TestNCHWxF16() const {
    if (align_corners()) {
      ASSERT_FALSE(tf_legacy_mode());
    }

    std::random_device random_device;
    auto rng = std::mt19937(random_device());
    std::uniform_real_distribution<float> f32dist;

    std::vector<uint16_t> input((batch_size() * input_height() * input_width() - 1) * input_pixel_stride() + channels() + XNN_EXTRA_BYTES / sizeof(uint16_t));
    std::vector<uint16_t> output((batch_size() * output_height() * output_width() - 1) * output_pixel_stride() + channels());
    std::vector<float> output_ref(batch_size() * output_height() * output_width() * channels());
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(input.begin(), input.end(), [&]() { return fp16_ieee_from_fp32_value(f32dist(rng)); });
      std::fill(output.begin(), output.end(), UINT16_C(0x7E00) /* NaN */);

      // Compute reference results.
      const float offset = (tf_legacy_mode() || align_corners()) ? 0.0f : 0.5f;
      const int64_t input_num_pixels = input_height() * input_width();
      const int64_t input_num_elements = input_num_pixels * input_pixel_stride();
      const int64_t output_num_pixels = output_height() * output_width();
      const int64_t output_num_elements = output_num_pixels * channels();
      for (size_t batch_index = 0; batch_index < batch_size(); batch_index++) {
        for (size_t output_y = 0; output_y < output_height(); output_y++) {
          const float input_y = (float(output_y) + offset) * height_scale() - offset;
          const int64_t input_y_top = std::max<int64_t>(int64_t(std::floor(input_y)), 0);
          const int64_t input_y_bottom = std::min<int64_t>(int64_t(std::ceil(input_y)), input_height() - 1);
          const float y_alpha = fp16_ieee_to_fp32_value(fp16_ieee_from_fp32_value(input_y - std::floor(input_y)));
          for (size_t output_x = 0; output_x < output_width(); output_x++) {
            const float input_x = (float(output_x) + offset) * width_scale() - offset;
            const int64_t input_x_left = std::max<int64_t>(int64_t(std::floor(input_x)), 0);
            const int64_t input_x_right = std::min<int64_t>(int64_t(std::ceil(input_x)), input_width() - 1);
            const float x_alpha = fp16_ieee_to_fp32_value(fp16_ieee_from_fp32_value(input_x - std::floor(input_x)));
            for (size_t c = 0; c < channels(); c++) {
              output_ref[batch_index * output_num_elements + c * output_num_pixels + output_y * output_width() + output_x] =
                fp16_ieee_to_fp32_value(input[batch_index * input_num_elements + c * input_num_pixels + input_y_top * input_width() + input_x_left]) * (1.0f - y_alpha) * (1.0f - x_alpha) +
                fp16_ieee_to_fp32_value(input[batch_index * input_num_elements + c * input_num_pixels + input_y_top * input_width() + input_x_right]) * (1.0f - y_alpha) * x_alpha +
                fp16_ieee_to_fp32_value(input[batch_index * input_num_elements + c * input_num_pixels + input_y_bottom * input_width() + input_x_left]) * y_alpha * (1.0f - x_alpha) +
                fp16_ieee_to_fp32_value(input[batch_index * input_num_elements + c * input_num_pixels + input_y_bottom * input_width() + input_x_right]) * y_alpha * x_alpha;
            }
          }
        }
      }

      // Create, setup, run, and destroy Resize Bilinear operator.
      ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
      xnn_operator_t resize_bilinear_op = nullptr;

      const xnn_status status = xnn_create_resize_bilinear2d_nchw_f16(
          output_height(), output_width(),
          (align_corners() ? XNN_FLAG_ALIGN_CORNERS : 0) | (tf_legacy_mode() ? XNN_FLAG_TENSORFLOW_LEGACY_MODE : 0),
          &resize_bilinear_op);
      if (status == xnn_status_unsupported_hardware) {
        GTEST_SKIP();
      }
      ASSERT_EQ(xnn_status_success, status);
      ASSERT_NE(nullptr, resize_bilinear_op);

      // Smart pointer to automatically delete resize_bilinear_op.
      std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_resize_bilinear_op(resize_bilinear_op, xnn_delete_operator);

      ASSERT_EQ(xnn_status_success,
        xnn_reshape_resize_bilinear2d_nchw_f16(
          resize_bilinear_op,
          batch_size(), input_height(), input_width(),
          channels(), input_pixel_stride(), output_pixel_stride(),
          /*threadpool=*/nullptr));

      ASSERT_EQ(xnn_status_success,
        xnn_setup_resize_bilinear2d_nchw_f16(
          resize_bilinear_op,
          input.data(), output.data()));

      ASSERT_EQ(xnn_status_success,
        xnn_run_operator(resize_bilinear_op, /*threadpool=*/nullptr));

      // Verify results.
      for (size_t i = 0; i < batch_size(); i++) {
        for (size_t y = 0; y < output_height(); y++) {
          for (size_t x = 0; x < output_width(); x++) {
            for (size_t c = 0; c < channels(); c++) {
              ASSERT_NEAR(fp16_ieee_to_fp32_value(output[i * output_num_elements +  c * output_num_pixels + y * output_width() + x]),
                  output_ref[i * output_num_elements +  c * output_num_pixels + y * output_width() + x],
                  std::max(1.0e-3f, std::abs(output_ref[i * output_num_elements +  c * output_num_pixels + y * output_width() + x]) * 1.0e-2f)) <<
                "in batch index " << i << ", pixel (" << y << ", " << x << "), channel " << c;
            }
          }
        }
      }
    }
  }

  void TestNCHWxF32() const {
    if (align_corners()) {
      ASSERT_FALSE(tf_legacy_mode());
    }

    std::random_device random_device;
    auto rng = std::mt19937(random_device());
    std::uniform_real_distribution<float> f32dist;

    std::vector<float> input((batch_size() * input_height() * input_width() - 1) * input_pixel_stride() + channels() + XNN_EXTRA_BYTES / sizeof(float));
    std::vector<float> output((batch_size() * output_height() * output_width() - 1) * output_pixel_stride() + channels());
    std::vector<float> output_ref(batch_size() * output_height() * output_width() * channels());
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(input.begin(), input.end(), [&]() { return f32dist(rng); });
      std::fill(output.begin(), output.end(), std::nanf(""));

      // Compute reference results.
      const float offset = (tf_legacy_mode() || align_corners()) ? 0.0f : 0.5f;
      const int64_t input_num_pixels = input_height() * input_width();
      const int64_t input_num_elements = input_num_pixels * input_pixel_stride();
      const int64_t output_num_pixels = output_height() * output_width();
      const int64_t output_num_elements = output_num_pixels * channels();
      for (size_t batch_index = 0; batch_index < batch_size(); batch_index++) {
        for (size_t output_y = 0; output_y < output_height(); output_y++) {
          const float input_y = (float(output_y) + offset) * height_scale() - offset;
          const int64_t input_y_top = std::max<int64_t>(int64_t(std::floor(input_y)), 0);
          const int64_t input_y_bottom = std::min<int64_t>(int64_t(std::ceil(input_y)), input_height() - 1);
          const float y_alpha = input_y - std::floor(input_y);
          for (size_t output_x = 0; output_x < output_width(); output_x++) {
            const float input_x = (float(output_x) + offset) * width_scale() - offset;
            const int64_t input_x_left = std::max<int64_t>(int64_t(std::floor(input_x)), 0);
            const int64_t input_x_right = std::min<int64_t>(int64_t(std::ceil(input_x)), input_width() - 1);
            const float x_alpha = input_x - std::floor(input_x);
            for (size_t c = 0; c < channels(); c++) {
              output_ref[batch_index * output_num_elements + c * output_num_pixels + output_y * output_width() + output_x] =
                input[batch_index * input_num_elements + c * input_num_pixels + input_y_top * input_width() + input_x_left] * (1.0f - y_alpha) * (1.0f - x_alpha) +
                input[batch_index * input_num_elements + c * input_num_pixels + input_y_top * input_width() + input_x_right] * (1.0f - y_alpha) * x_alpha +
                input[batch_index * input_num_elements + c * input_num_pixels + input_y_bottom * input_width() + input_x_left] * y_alpha * (1.0f - x_alpha) +
                input[batch_index * input_num_elements + c * input_num_pixels + input_y_bottom * input_width() + input_x_right] * y_alpha * x_alpha;
            }
          }
        }
      }

      // Create, setup, run, and destroy Resize Bilinear operator.
      ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
      xnn_operator_t resize_bilinear_op = nullptr;

      const xnn_status status =
        xnn_create_resize_bilinear2d_nchw_f32(
          output_height(), output_width(),
          (align_corners() ? XNN_FLAG_ALIGN_CORNERS : 0) | (tf_legacy_mode() ? XNN_FLAG_TENSORFLOW_LEGACY_MODE : 0),
          &resize_bilinear_op);
      if (status == xnn_status_unsupported_hardware) {
        GTEST_SKIP();
      }
      ASSERT_NE(nullptr, resize_bilinear_op);

      // Smart pointer to automatically delete resize_bilinear_op.
      std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_resize_bilinear_op(resize_bilinear_op, xnn_delete_operator);

      ASSERT_EQ(xnn_status_success,
        xnn_reshape_resize_bilinear2d_nchw_f32(
          resize_bilinear_op,
          batch_size(), input_height(), input_width(),
          channels(), input_pixel_stride(), output_pixel_stride(),
          /*threadpool=*/nullptr));

      ASSERT_EQ(xnn_status_success,
        xnn_setup_resize_bilinear2d_nchw_f32(
          resize_bilinear_op,
          input.data(), output.data()));

      ASSERT_EQ(xnn_status_success,
        xnn_run_operator(resize_bilinear_op, /*threadpool=*/nullptr));

      // Verify results.
      for (size_t i = 0; i < batch_size(); i++) {
        for (size_t y = 0; y < output_height(); y++) {
          for (size_t x = 0; x < output_width(); x++) {
            for (size_t c = 0; c < channels(); c++) {
              ASSERT_NEAR(output[i * output_num_elements +  c * output_num_pixels + y * output_width() + x],
                  output_ref[i * output_num_elements +  c * output_num_pixels + y * output_width() + x],
                  1.0e-6f) <<
                "in batch index " << i << ", pixel (" << y << ", " << x << "), channel " << c;
            }
          }
        }
      }
    }
  }

 private:
  size_t input_height_{1};
  size_t input_width_{1};
  size_t output_height_{1};
  size_t output_width_{1};
  size_t channels_{1};
  size_t batch_size_{1};
  size_t input_pixel_stride_{0};
  size_t output_pixel_stride_{0};
  size_t next_input_height_{0};
  size_t next_input_width_{0};
  size_t next_batch_size_{0};
  bool align_corners_{false};
  bool tf_legacy_mode_{false};
  size_t iterations_{1};
  bool transient_indirection_buffer_{false};
};
