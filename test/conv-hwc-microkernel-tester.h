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
#include <random>
#include <vector>

#include <gtest/gtest.h>
#include "xnnpack.h"
#include "xnnpack/aligned-allocator.h"
#include "xnnpack/microfnptr.h"
#include "xnnpack/microparams.h"
#include "xnnpack/pack.h"
#include "replicable_random_device.h"

class ConvHWCMicrokernelTester {
 public:
  ConvHWCMicrokernelTester& output_channels_tile(
      uint32_t output_channels_tile) {
    this->output_channels_tile_ = output_channels_tile;
    return *this;
  }

  uint32_t output_channels_tile() const {
    return this->output_channels_tile_;
  }

  ConvHWCMicrokernelTester& padding(uint32_t padding) {
    this->padding_top_ = padding;
    this->padding_right_ = padding;
    this->padding_bottom_ = padding;
    this->padding_left_ = padding;
    return *this;
  }

  ConvHWCMicrokernelTester& padding_height(uint32_t padding_height) {
    this->padding_top_ = padding_height;
    this->padding_bottom_ = padding_height;
    return *this;
  }

  ConvHWCMicrokernelTester& padding_width(uint32_t padding_width) {
    this->padding_right_ = padding_width;
    this->padding_left_ = padding_width;
    return *this;
  }

  ConvHWCMicrokernelTester& padding_top(uint32_t padding_top) {
    this->padding_top_ = padding_top;
    return *this;
  }

  uint32_t padding_top() const {
    return this->padding_top_;
  }

  ConvHWCMicrokernelTester& padding_right(uint32_t padding_right) {
    this->padding_right_ = padding_right;
    return *this;
  }

  uint32_t padding_right() const {
    return this->padding_right_;
  }

  ConvHWCMicrokernelTester& padding_bottom(uint32_t padding_bottom) {
    this->padding_bottom_ = padding_bottom;
    return *this;
  }

  uint32_t padding_bottom() const {
    return this->padding_bottom_;
  }

  ConvHWCMicrokernelTester& padding_left(uint32_t padding_left) {
    this->padding_left_ = padding_left;
    return *this;
  }

  uint32_t padding_left() const {
    return this->padding_left_;
  }

  ConvHWCMicrokernelTester& input_size(uint32_t input_height, uint32_t input_width) {
    assert(input_height >= 1);
    assert(input_width >= 1);
    this->input_height_ = input_height;
    this->input_width_ = input_width;
    return *this;
  }

  ConvHWCMicrokernelTester& input_height(uint32_t input_height) {
    assert(input_height >= 1);
    this->input_height_ = input_height;
    return *this;
  }

  uint32_t input_height() const {
    return this->input_height_;
  }

  ConvHWCMicrokernelTester& input_width(uint32_t input_width) {
    assert(input_width >= 1);
    this->input_width_ = input_width;
    return *this;
  }

  uint32_t input_width() const {
    return this->input_width_;
  }

  ConvHWCMicrokernelTester& input_channels(size_t input_channels) {
    assert(input_channels >= 1);
    this->input_channels_ = input_channels;
    return *this;
  }

  size_t input_channels() const {
    return this->input_channels_;
  }

  ConvHWCMicrokernelTester& output_channels(size_t output_channels) {
    assert(output_channels >= 1);
    this->output_channels_ = output_channels;
    return *this;
  }

  size_t output_channels() const {
    return this->output_channels_;
  }

  size_t packed_output_channels() const {
    return output_channels() % output_channels_tile() == 0 ? output_channels() : output_channels() / output_channels_tile() * output_channels_tile() + output_channels_tile();
  }

  ConvHWCMicrokernelTester& batch_size(size_t batch_size) {
    assert(batch_size >= 1);
    this->batch_size_ = batch_size;
    return *this;
  }

  size_t batch_size() const {
    return this->batch_size_;
  }

  ConvHWCMicrokernelTester& kernel_size(uint32_t kernel_size) {
    assert(kernel_size >= 1);
    this->kernel_height_ = kernel_size;
    this->kernel_width_ = kernel_size;
    return *this;
  }

  ConvHWCMicrokernelTester& kernel_height(uint32_t kernel_height) {
    assert(kernel_height >= 1);
    this->kernel_height_ = kernel_height;
    return *this;
  }

  uint32_t kernel_height() const {
    return this->kernel_height_;
  }

  ConvHWCMicrokernelTester& kernel_width(uint32_t kernel_width) {
    assert(kernel_width >= 1);
    this->kernel_width_ = kernel_width;
    return *this;
  }

  uint32_t kernel_width() const {
    return this->kernel_width_;
  }

  ConvHWCMicrokernelTester& subsampling(uint32_t subsampling) {
    assert(subsampling >= 1);
    this->subsampling_height_ = subsampling;
    this->subsampling_width_ = subsampling;
    return *this;
  }

  ConvHWCMicrokernelTester& subsampling_height(uint32_t subsampling_height) {
    assert(subsampling_height >= 1);
    this->subsampling_height_ = subsampling_height;
    return *this;
  }

  uint32_t subsampling_height() const {
    return this->subsampling_height_;
  }

  ConvHWCMicrokernelTester& subsampling_width(uint32_t subsampling_width) {
    assert(subsampling_width >= 1);
    this->subsampling_width_ = subsampling_width;
    return *this;
  }

  uint32_t subsampling_width() const {
    return this->subsampling_width_;
  }

  ConvHWCMicrokernelTester& output_y_start(uint32_t output_y_start) {
    this->output_y_start_ = output_y_start;
    return *this;
  }

  uint32_t output_y_start() const {
    return this->output_y_start_;
  }

  ConvHWCMicrokernelTester& output_y_end(uint32_t output_y_end) {
    this->output_y_end_ = output_y_end;
    return *this;
  }

  uint32_t output_y_end() const {
    if (this->output_y_end_ == std::numeric_limits<uint32_t>::max()) {
      return output_height();
    } else {
      return this->output_y_end_;
    }
  }

  size_t input_pixel_stride() const {
    return input_channels();
  }

  size_t output_pixel_stride() const {
    return output_channels();
  }

  size_t output_height() const {
    const size_t padded_input_height = padding_top() + input_height() + padding_bottom();
    return (std::max<size_t>(padded_input_height + subsampling_height(), kernel_height()) - kernel_height())
      / subsampling_height();
  }

  size_t output_width() const {
    const size_t padded_input_width = padding_left() + input_width() + padding_right();
    return (std::max<size_t>(padded_input_width + subsampling_width(), kernel_width()) - kernel_width())
      / subsampling_width();
  }

  ConvHWCMicrokernelTester& qmin(uint8_t qmin) {
    this->qmin_ = qmin;
    return *this;
  }

  uint8_t qmin() const {
    return this->qmin_;
  }

  ConvHWCMicrokernelTester& qmax(uint8_t qmax) {
    this->qmax_ = qmax;
    return *this;
  }

  uint8_t qmax() const {
    return this->qmax_;
  }

  ConvHWCMicrokernelTester& iterations(size_t iterations) {
    this->iterations_ = iterations;
    return *this;
  }

  size_t iterations() const {
    return this->iterations_;
  }

  void Test(xnn_f32_conv_hwc_ukernel_fn conv, xnn_init_f32_minmax_params_fn init_params) const {
    ASSERT_LT(output_y_start(), output_height());
    ASSERT_LE(output_y_end(), output_height());
    ASSERT_GT(output_y_end(), output_y_start());
    ASSERT_GE(output_width(), 1);
    ASSERT_GE(output_height(), 1);

    xnnpack::ReplicableRandomDevice rng;
    std::uniform_real_distribution<float> f32dist(0.1f, 1.0f);

    std::vector<float> input(XNN_EXTRA_BYTES / sizeof(float) +
      batch_size() * ((input_height() * input_width() - 1) * input_pixel_stride() + input_channels()));
    std::vector<float> zero(XNN_EXTRA_BYTES / sizeof(float) + input_width() * input_channels());
    std::vector<float> kernel(output_channels() * kernel_height() * kernel_width() * input_channels());
    std::vector<float> bias(output_channels());
    std::vector<float> output(batch_size() * ((output_height() * output_width() - 1) * output_pixel_stride() + output_channels()));
    std::vector<float> output_ref(batch_size() * output_height() * output_width() * output_channels());
    std::vector<float, AlignedAllocator<float, 64>> packed_weights((input_channels() * kernel_height() * kernel_width() + 1) * packed_output_channels());

    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(input.begin(), input.end(), [&]() { return f32dist(rng); });
      std::generate(kernel.begin(), kernel.end(), [&]() { return f32dist(rng); });
      std::generate(bias.begin(), bias.end(), [&]() { return f32dist(rng); });
      std::fill(output.begin(), output.end(), nanf(""));
      std::fill(packed_weights.begin(), packed_weights.end(), 0.0f);

      xnn_pack_f32_dconv_oki_w(
        output_channels(),
        input_channels(),
        output_channels_tile(),
        kernel_height(), kernel_width(),
        kernel.data(), bias.data(), packed_weights.data(), nullptr);

      // Compute reference results, without clamping.
      for (size_t i = 0; i < batch_size(); i++) {
        for (size_t oy = 0; oy < output_height(); oy++) {
          for (size_t ox = 0; ox < output_width(); ox++) {
            for (size_t oc = 0; oc < output_channels(); oc++) {
              float acc = bias[oc];
              for (size_t ky = 0; ky < kernel_height(); ky++) {
                const size_t iy = oy * subsampling_height() + ky - padding_top();
                if (iy < input_height()) {
                  for (size_t kx = 0; kx < kernel_width(); kx++) {
                    const size_t ix = ox * subsampling_width() + kx - padding_left();
                    if (ix < input_width()) {
                      for (size_t ic = 0; ic < input_channels(); ic++) {
                        acc +=
                          input[((i * input_height() + iy) * input_width() + ix) * input_pixel_stride() + ic] *
                          kernel[((oc * kernel_height() + ky) * kernel_width() + kx) * input_channels() + ic];
                      }
                    }
                  }
                }
              }
              output_ref[((i * output_height() + oy) * output_width() + ox) * output_channels() + oc] = acc;
            }
          }
        }
      }

      // Compute clamping parameters.
      const float accumulated_min = *std::min_element(output_ref.cbegin(), output_ref.cend());
      const float accumulated_max = *std::max_element(output_ref.cbegin(), output_ref.cend());

      const float output_min = accumulated_min + (accumulated_max - accumulated_min) / 255.0f * float(qmin());
      const float output_max = accumulated_max - (accumulated_max - accumulated_min) / 255.0f * float(255 - qmax());

      // Clamp reference results.
      for (float& value : output_ref) {
        value = std::max(std::min(value, output_max), output_min);
      }

      // Prepare parameters.
      xnn_f32_minmax_params params;
      init_params(&params, output_min, output_max);

      // Call optimized micro-kernel.
      conv(
        input_height(), input_width(),
        output_y_start(), output_y_end(),
        input.data(), zero.data(), packed_weights.data(), output.data(),
        padding_top(), output_channels(),
        output_pixel_stride() * output_width() * sizeof(float),
        output_pixel_stride() * sizeof(float),
        &params);

      // Verify results.
      for (size_t i = 0; i < batch_size(); i++) {
        for (size_t y = output_y_start(); y < output_y_end(); y++) {
          for (size_t x = 0; x < output_width(); x++) {
            for (size_t c = 0; c < output_channels(); c++) {
              EXPECT_GE(output[((i * output_height() + y) * output_width() + x) * output_pixel_stride() + c], output_min)
                << "(x, y) = (" << x << ", " << y << "), channel = " << c;
              EXPECT_LE(output[((i * output_height() + y) * output_width() + x) * output_pixel_stride() + c], output_max)
                << "(x, y) = (" << x << ", " << y << "), channel = " << c;
              EXPECT_NEAR(
                  output_ref[((i * output_height() + y) * output_width() + x) * output_channels() + c],
                  output[((i * output_height() + y) * output_width() + x) * output_pixel_stride() + c],
                  1.0e-4 * std::abs(output_ref[((i * output_height() + y) * output_width() + x) * output_channels() + c]))
                << "(x, y) = (" << x << ", " << y << "), channel = " << c;
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
  size_t input_channels_{1};
  size_t output_channels_{1};
  uint32_t output_channels_tile_{1};
  size_t batch_size_{1};
  uint32_t kernel_height_{1};
  uint32_t kernel_width_{1};
  uint32_t subsampling_height_{1};
  uint32_t subsampling_width_{1};
  uint32_t output_y_start_{0};
  uint32_t output_y_end_{std::numeric_limits<uint32_t>::max()};
  uint8_t qmin_{0};
  uint8_t qmax_{255};
  size_t iterations_{1};
};
