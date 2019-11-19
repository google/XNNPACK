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
#include <random>
#include <vector>

#include <xnnpack.h>


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

      // Compute reference results.
      const float offset = tf_legacy_mode() ? 0.0f : 0.5f;
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

      ASSERT_EQ(xnn_status_success,
        xnn_create_resize_bilinear2d_nhwc_f32(
          channels(), input_pixel_stride(), output_pixel_stride(),
          (align_corners() ? XNN_FLAG_ALIGN_CORNERS : 0) | (tf_legacy_mode() ? XNN_FLAG_TENSORFLOW_LEGACY_MODE : 0),
          &resize_bilinear_op));
      ASSERT_NE(nullptr, resize_bilinear_op);

      // Smart pointer to automatically delete resize_bilinear_op.
      std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_resize_bilinear_op(resize_bilinear_op, xnn_delete_operator);

      ASSERT_EQ(xnn_status_success,
        xnn_setup_resize_bilinear2d_nhwc_f32(
          resize_bilinear_op,
          batch_size(), input_height(), input_width(),
          output_height(), output_width(),
          input.data(), output.data(),
          nullptr /* thread pool */));

      ASSERT_EQ(xnn_status_success,
        xnn_run_operator(resize_bilinear_op, nullptr /* thread pool */));

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

  // void TestSetupF32() const {
  //   std::random_device random_device;
  //   auto rng = std::mt19937(random_device());
  //   auto f32rng = std::bind(std::uniform_real_distribution<float>(), rng);

  //   std::vector<float> input(XNN_EXTRA_BYTES / sizeof(float) + std::max(
  //     (batch_size() * input_height() * input_width() - 1) * input_pixel_stride() + channels(),
  //     (next_batch_size() * next_input_height() * next_input_width() - 1) * input_pixel_stride() + channels()));
  //   std::vector<float> output(std::max(
  //     (batch_size() * output_height() * output_width() - 1) * output_pixel_stride() + channels(),
  //     (next_batch_size() * next_output_height() * next_output_width() - 1) * output_pixel_stride() + channels()));
  //   std::vector<float> output_ref(batch_size() * output_height() * output_width() * channels());
  //   std::vector<float> next_output_ref(next_batch_size() * next_output_height() * next_output_width() * channels());
  //   for (size_t iteration = 0; iteration < iterations(); iteration++) {
  //     std::generate(input.begin(), input.end(), std::ref(f32rng));
  //     std::fill(output.begin(), output.end(), std::nanf(""));

  //     // Compute reference results, without clamping.
  //     for (size_t batch_index = 0; batch_index < batch_size(); batch_index++) {
  //       for (size_t output_y = 0; output_y < output_height(); output_y++) {
  //         for (size_t output_x = 0; output_x < output_width(); output_x++) {
  //           for (size_t c = 0; c < channels(); c++) {
  //             float acc = 0.0f;
  //             size_t n = 0;
  //             for (size_t py = 0; py < pooling_height(); py++) {
  //               const size_t iy = output_y * stride_height() + py - padding_top();
  //               for (size_t px = 0; px < pooling_width(); px++) {
  //                 const size_t input_x = output_x * stride_width() + px - padding_left();
  //                 if (input_x < input_width() && iy < input_height()) {
  //                   acc += input[((batch_index * input_height() + iy) * input_width() + input_x) * input_pixel_stride() + c];
  //                   n += 1;
  //                 }
  //               }
  //             }
  //             output_ref[((batch_index * output_height() + output_y) * output_width() + output_x) * channels() + c] = acc / float(n);
  //           }
  //         }
  //       }
  //     }

  //     // Compute clamping parameters.
  //     const float accumulated_min = *std::min_element(output_ref.cbegin(), output_ref.cend());
  //     const float accumulated_max = *std::max_element(output_ref.cbegin(), output_ref.cend());
  //     const float accumulated_range = accumulated_max - accumulated_min;
  //     const float output_min = accumulated_range == 0.0f ?
  //       -std::numeric_limits<float>::infinity() :
  //       accumulated_min + accumulated_range / 255.0f * float(qmin());
  //     const float output_max = accumulated_range == 0.0f ?
  //       +std::numeric_limits<float>::infinity() :
  //       accumulated_max - accumulated_range / 255.0f * float(255 - qmax());

  //     // Clamp reference results.
  //     for (float& value : output_ref) {
  //       value = std::max(std::min(value, output_max), output_min);
  //     }

  //     // Create, setup, and run Average Pooling operator once.
  //     ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  //     xnn_operator_t resize_bilinear_op = nullptr;

  //     ASSERT_EQ(xnn_status_success,
  //       xnn_create_average_pooling2d_nhwc_f32(
  //         padding_top(), padding_right(), padding_bottom(), padding_left(),
  //         pooling_height(), pooling_width(),
  //         stride_height(), stride_width(),
  //         channels(), input_pixel_stride(), output_pixel_stride(),
  //         output_min, output_max,
  //         0, &resize_bilinear_op));
  //     ASSERT_NE(nullptr, resize_bilinear_op);

  //     ASSERT_EQ(xnn_status_success,
  //       xnn_setup_average_pooling2d_nhwc_f32(
  //         resize_bilinear_op,
  //         batch_size(), input_height(), input_width(),
  //         input.data(), output.data(),
  //         nullptr /* thread pool */));

  //     ASSERT_EQ(xnn_status_success,
  //       xnn_run_operator(resize_bilinear_op, nullptr /* thread pool */));

  //     // Verify results of the first run.
  //     for (size_t batch_index = 0; batch_index < batch_size(); batch_index++) {
  //       for (size_t y = 0; y < output_height(); y++) {
  //         for (size_t x = 0; x < output_width(); x++) {
  //           for (size_t c = 0; c < channels(); c++) {
  //             ASSERT_LE(output[((batch_index * output_height() + y) * output_width() + x) * output_pixel_stride() + c], output_max);
  //             ASSERT_GE(output[((batch_index * output_height() + y) * output_width() + x) * output_pixel_stride() + c], output_min);
  //             ASSERT_NEAR(output[((batch_index * output_height() + y) * output_width() + x) * output_pixel_stride() + c],
  //                 output_ref[((batch_index * output_height() + y) * output_width() + x) * channels() + c],
  //                 std::abs(output_ref[((batch_index * output_height() + y) * output_width() + x) * channels() + c]) * 1.0e-6f) <<
  //               "in batch index " << batch_index << ", pixel (" << y << ", " << x << "), channel " << c;
  //           }
  //         }
  //       }
  //     }

  //     // Re-generate data for the second run.
  //     std::generate(input.begin(), input.end(), std::ref(f32rng));
  //     std::fill(output.begin(), output.end(), std::nanf(""));

  //     // Compute reference results for the second run.
  //     for (size_t batch_index = 0; batch_index < next_batch_size(); batch_index++) {
  //       for (size_t output_y = 0; output_y < next_output_height(); output_y++) {
  //         for (size_t output_x = 0; output_x < next_output_width(); output_x++) {
  //           for (size_t c = 0; c < channels(); c++) {
  //             float acc = 0.0f;
  //             int32_t n = 0;
  //             for (size_t py = 0; py < pooling_height(); py++) {
  //               const size_t iy = output_y * stride_height() + py - padding_top();
  //               for (size_t px = 0; px < pooling_width(); px++) {
  //                 const size_t input_x = output_x * stride_width() + px - padding_left();
  //                 if (input_x < next_input_width() && iy < next_input_height()) {
  //                   acc += input[((batch_index * next_input_height() + iy) * next_input_width() + input_x) * input_pixel_stride() + c];
  //                   n += 1;
  //                 }
  //               }
  //             }
  //             next_output_ref[((batch_index * next_output_height() + output_y) * next_output_width() + output_x) * channels() + c] =
  //               std::max(std::min(acc / float(n), output_max), output_min);
  //           }
  //         }
  //       }
  //     }

  //     // Setup and run Average Pooling operator the second time, and destroutput_y the operator.
  //     ASSERT_EQ(xnn_status_success,
  //       xnn_setup_average_pooling2d_nhwc_f32(
  //         resize_bilinear_op,
  //         next_batch_size(), next_input_height(), next_input_width(),
  //         input.data(), output.data(),
  //         nullptr /* thread pool */));

  //     ASSERT_EQ(xnn_status_success,
  //       xnn_run_operator(resize_bilinear_op, nullptr /* thread pool */));

  //     ASSERT_EQ(xnn_status_success,
  //       xnn_delete_operator(resize_bilinear_op));
  //     resize_bilinear_op = nullptr;

  //     // Verify results of the second run.
  //     for (size_t batch_index = 0; batch_index < next_batch_size(); batch_index++) {
  //       for (size_t y = 0; y < next_output_height(); y++) {
  //         for (size_t x = 0; x < next_output_width(); x++) {
  //           for (size_t c = 0; c < channels(); c++) {
  //             ASSERT_LE(output[((batch_index * next_output_height() + y) * next_output_width() + x) * output_pixel_stride() + c], output_max);
  //             ASSERT_GE(output[((batch_index * next_output_height() + y) * next_output_width() + x) * output_pixel_stride() + c], output_min);
  //             ASSERT_NEAR(output[((batch_index * next_output_height() + y) * next_output_width() + x) * output_pixel_stride() + c],
  //                 next_output_ref[((batch_index * next_output_height() + y) * next_output_width() + x) * channels() + c],
  //                 std::abs(next_output_ref[((batch_index * next_output_height() + y) * next_output_width() + x) * channels() + c]) * 1.0e-6f) <<
  //               "in batch index " << batch_index << ", pixel (" << y << ", " << x << "), channel " << c;
  //           }
  //         }
  //       }
  //     }
  //   }
  // }

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
};
