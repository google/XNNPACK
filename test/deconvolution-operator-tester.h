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
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <functional>
#include <limits>
#include <random>
#include <vector>

#include <xnnpack.h>

namespace {

template<class T>
inline T doz(T a, T b) {
  return a > b ? a - b : T(0);
}

}  // namespace

class DeconvolutionOperatorTester {
 public:
  inline DeconvolutionOperatorTester& padding_tf_same(bool padding_same) {
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

  inline DeconvolutionOperatorTester& padding(uint32_t padding) {
    assert(!padding_tf_same());
    this->padding_top_ = padding;
    this->padding_right_ = padding;
    this->padding_bottom_ = padding;
    this->padding_left_ = padding;
    return *this;
  }

  inline DeconvolutionOperatorTester& padding(uint32_t padding_height, uint32_t padding_width) {
    assert(!padding_tf_same());
    this->padding_top_ = padding_height;
    this->padding_right_ = padding_width;
    this->padding_bottom_ = padding_height;
    this->padding_left_ = padding_width;
    return *this;
  }

  inline DeconvolutionOperatorTester& padding_height(uint32_t padding_height) {
    assert(!padding_tf_same());
    this->padding_top_ = padding_height;
    this->padding_bottom_ = padding_height;
    return *this;
  }

  inline uint32_t padding_height() const {
    if (padding_tf_same()) {
      return doz(dilated_kernel_height() - 1, static_cast<uint32_t>((input_height() - 1) % stride_height()));
    } else {
      return this->padding_top_ + this->padding_bottom_;
    }
  }

  inline DeconvolutionOperatorTester& padding_width(uint32_t padding_width) {
    assert(!padding_tf_same());
    this->padding_right_ = padding_width;
    this->padding_left_ = padding_width;
    return *this;
  }

  inline uint32_t padding_width() const {
    if (padding_tf_same()) {
      return doz(dilated_kernel_width() - 1, static_cast<uint32_t>((input_width() - 1) % stride_width()));
    } else {
      return this->padding_left_ + this->padding_right_;
    }
  }

  inline DeconvolutionOperatorTester& padding_top(uint32_t padding_top) {
    assert(!padding_tf_same());
    this->padding_top_ = padding_top;
    return *this;
  }

  inline uint32_t padding_top() const {
    if (padding_tf_same()) {
      return padding_height() / 2;
    } else {
      return this->padding_top_;
    }
  }

  inline DeconvolutionOperatorTester& padding_right(uint32_t padding_right) {
    assert(!padding_tf_same());
    this->padding_right_ = padding_right;
    return *this;
  }

  inline uint32_t padding_right() const {
    if (padding_tf_same()) {
      return padding_width() - padding_left();
    } else {
      return this->padding_right_;
    }
  }

  inline DeconvolutionOperatorTester& padding_bottom(uint32_t padding_bottom) {
    assert(!padding_tf_same());
    this->padding_bottom_ = padding_bottom;
    return *this;
  }

  inline uint32_t padding_bottom() const {
    if (padding_tf_same()) {
      return padding_height() - padding_top();
    } else {
      return this->padding_bottom_;
    }
  }

  inline DeconvolutionOperatorTester& padding_left(uint32_t padding_left) {
    assert(!padding_tf_same());
    this->padding_left_ = padding_left;
    return *this;
  }

  inline uint32_t padding_left() const {
    if (padding_tf_same()) {
      return padding_width() / 2;
    } else {
      return this->padding_left_;
    }
  }

  inline DeconvolutionOperatorTester& adjustment_height(uint32_t adjustment_height) {
    this->adjustment_height_ = adjustment_height;
    return *this;
  }

  inline uint32_t adjustment_height() const {
    return this->adjustment_height_;
  }

  inline DeconvolutionOperatorTester& adjustment_width(uint32_t adjustment_width) {
    this->adjustment_width_ = adjustment_width;
    return *this;
  }

  inline uint32_t adjustment_width() const {
    return this->adjustment_width_;
  }

  inline DeconvolutionOperatorTester& input_size(uint32_t input_height, uint32_t input_width) {
    assert(input_height >= 1);
    assert(input_width >= 1);
    this->input_height_ = input_height;
    this->input_width_ = input_width;
    return *this;
  }

  inline DeconvolutionOperatorTester& input_height(uint32_t input_height) {
    assert(input_height >= 1);
    this->input_height_ = input_height;
    return *this;
  }

  inline uint32_t input_height() const {
    return this->input_height_;
  }

  inline DeconvolutionOperatorTester& input_width(uint32_t input_width) {
    assert(input_width >= 1);
    this->input_width_ = input_width;
    return *this;
  }

  inline uint32_t input_width() const {
    return this->input_width_;
  }

  inline DeconvolutionOperatorTester& groups(uint32_t groups) {
    assert(groups >= 1);
    this->groups_ = groups;
    return *this;
  }

  inline uint32_t groups() const {
    return this->groups_;
  }

  inline DeconvolutionOperatorTester& group_input_channels(size_t group_input_channels) {
    assert(group_input_channels >= 1);
    this->group_input_channels_ = group_input_channels;
    return *this;
  }

  inline size_t group_input_channels() const {
    return this->group_input_channels_;
  }

  inline DeconvolutionOperatorTester& group_output_channels(size_t group_output_channels) {
    assert(group_output_channels >= 1);
    this->group_output_channels_ = group_output_channels;
    return *this;
  }

  inline size_t group_output_channels() const {
    return this->group_output_channels_;
  }

  inline DeconvolutionOperatorTester& batch_size(size_t batch_size) {
    assert(batch_size >= 1);
    this->batch_size_ = batch_size;
    return *this;
  }

  inline size_t batch_size() const {
    return this->batch_size_;
  }

  inline DeconvolutionOperatorTester& kernel_size(uint32_t kernel_size) {
    assert(kernel_size >= 1);
    this->kernel_height_ = kernel_size;
    this->kernel_width_ = kernel_size;
    return *this;
  }

  inline DeconvolutionOperatorTester& kernel_size(uint32_t kernel_height, uint32_t kernel_width) {
    assert(kernel_height >= 1);
    assert(kernel_width >= 1);
    this->kernel_height_ = kernel_height;
    this->kernel_width_ = kernel_width;
    return *this;
  }

  inline DeconvolutionOperatorTester& kernel_height(uint32_t kernel_height) {
    assert(kernel_height >= 1);
    this->kernel_height_ = kernel_height;
    return *this;
  }

  inline uint32_t kernel_height() const {
    return this->kernel_height_;
  }

  inline DeconvolutionOperatorTester& kernel_width(uint32_t kernel_width) {
    assert(kernel_width >= 1);
    this->kernel_width_ = kernel_width;
    return *this;
  }

  inline uint32_t kernel_width() const {
    return this->kernel_width_;
  }

  inline DeconvolutionOperatorTester& dilation(uint32_t dilation) {
    assert(dilation >= 1);
    this->dilation_height_ = dilation;
    this->dilation_width_ = dilation;
    return *this;
  }

  inline DeconvolutionOperatorTester& dilation(uint32_t dilation_height, uint32_t dilation_width) {
    assert(dilation_height >= 1);
    assert(dilation_width >= 1);
    this->dilation_height_ = dilation_height;
    this->dilation_width_ = dilation_width;
    return *this;
  }

  inline DeconvolutionOperatorTester& dilation_height(uint32_t dilation_height) {
    assert(dilation_height >= 1);
    this->dilation_height_ = dilation_height;
    return *this;
  }

  inline uint32_t dilation_height() const {
    return this->dilation_height_;
  }

  inline DeconvolutionOperatorTester& dilation_width(uint32_t dilation_width) {
    assert(dilation_width >= 1);
    this->dilation_width_ = dilation_width;
    return *this;
  }

  inline uint32_t dilation_width() const {
    return this->dilation_width_;
  }

  inline DeconvolutionOperatorTester& stride(uint32_t stride) {
    assert(stride >= 1);
    this->stride_height_ = stride;
    this->stride_width_ = stride;
    return *this;
  }

  inline DeconvolutionOperatorTester& stride(uint32_t stride_height, uint32_t stride_width) {
    assert(stride_height >= 1);
    assert(stride_width >= 1);
    this->stride_height_ = stride_height;
    this->stride_width_ = stride_width;
    return *this;
  }

  inline DeconvolutionOperatorTester& stride_height(uint32_t stride_height) {
    assert(stride_height >= 1);
    this->stride_height_ = stride_height;
    return *this;
  }

  inline uint32_t stride_height() const {
    return this->stride_height_;
  }

  inline DeconvolutionOperatorTester& stride_width(uint32_t stride_width) {
    assert(stride_width >= 1);
    this->stride_width_ = stride_width;
    return *this;
  }

  inline uint32_t stride_width() const {
    return this->stride_width_;
  }

  inline DeconvolutionOperatorTester& input_pixel_stride(size_t input_pixel_stride) {
    assert(input_pixel_stride >= 1);
    this->input_pixel_stride_ = input_pixel_stride;
    return *this;
  }

  inline size_t input_pixel_stride() const {
    if (this->input_pixel_stride_ == 0) {
      return group_input_channels() * groups();
    } else {
      assert(this->input_pixel_stride_ >= group_input_channels() * groups());
      return this->input_pixel_stride_;
    }
  }

  inline DeconvolutionOperatorTester& output_pixel_stride(size_t output_pixel_stride) {
    assert(output_pixel_stride >= 1);
    this->output_pixel_stride_ = output_pixel_stride;
    return *this;
  }

  inline size_t output_pixel_stride() const {
    if (this->output_pixel_stride_ == 0) {
      return group_output_channels() * groups();
    } else {
      assert(this->output_pixel_stride_ >= group_output_channels() * groups());
      return this->output_pixel_stride_;
    }
  }

  inline uint32_t dilated_kernel_height() const {
    return (kernel_height() - 1) * dilation_height() + 1;
  }

  inline uint32_t dilated_kernel_width() const {
    return (kernel_width() - 1) * dilation_width() + 1;
  }

  inline size_t output_height() const {
    return stride_height() * (input_height() - 1) + adjustment_height() + dilated_kernel_height() - padding_height();
  }

  inline size_t output_width() const {
    return stride_width() * (input_width() - 1) + adjustment_width() + dilated_kernel_width() - padding_width();
  }

  inline DeconvolutionOperatorTester& next_input_size(uint32_t next_input_height, uint32_t next_input_width) {
    assert(next_input_height >= 1);
    assert(next_input_width >= 1);
    this->next_input_height_ = next_input_height;
    this->next_input_width_ = next_input_width;
    return *this;
  }

  inline DeconvolutionOperatorTester& next_input_height(uint32_t next_input_height) {
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

  inline DeconvolutionOperatorTester& next_input_width(uint32_t next_input_width) {
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
    return stride_height() * (next_input_height() - 1) + adjustment_height() + dilated_kernel_height() - padding_height();
  }

  inline size_t next_output_width() const {
    return stride_width() * (next_input_width() - 1) + adjustment_width() + dilated_kernel_width() - padding_width();
  }

  inline DeconvolutionOperatorTester& next_batch_size(size_t next_batch_size) {
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

  inline DeconvolutionOperatorTester& qmin(uint8_t qmin) {
    this->qmin_ = qmin;
    return *this;
  }

  inline uint8_t qmin() const {
    return this->qmin_;
  }

  inline DeconvolutionOperatorTester& qmax(uint8_t qmax) {
    this->qmax_ = qmax;
    return *this;
  }

  inline uint8_t qmax() const {
    return this->qmax_;
  }

  inline DeconvolutionOperatorTester& has_bias(bool has_bias) {
    this->has_bias_ = has_bias;
    return *this;
  }

  inline bool has_bias() const {
    return this->has_bias_;
  }

  inline DeconvolutionOperatorTester& iterations(size_t iterations) {
    this->iterations_ = iterations;
    return *this;
  }

  inline size_t iterations() const {
    return this->iterations_;
  }

  void TestQ8() const {
    std::random_device random_device;
    auto rng = std::mt19937(random_device());
    auto s32rng = std::bind(std::uniform_int_distribution<int32_t>(-10000, 10000), rng);
    auto u8rng = std::bind(std::uniform_int_distribution<uint32_t>(0, std::numeric_limits<uint8_t>::max()), rng);

    std::vector<uint8_t> input(XNN_EXTRA_BYTES / sizeof(uint8_t) +
      (batch_size() * input_height() * input_width() - 1) * input_pixel_stride() + groups() * group_input_channels());
    std::vector<uint8_t> kernel(groups() * group_output_channels() * kernel_height() * kernel_width() * group_input_channels());
    std::vector<int32_t> bias(groups() * group_output_channels());
    std::vector<uint8_t> output((batch_size() * output_height() * output_width() - 1) * output_pixel_stride() + groups() * group_output_channels());
    std::vector<int32_t> accumulators(batch_size() * output_height() * output_width() * groups() * group_output_channels());
    std::vector<double> output_ref(batch_size() * output_height() * output_width() * groups() * group_output_channels());

    const uint8_t input_zero_point = 127;
    const uint8_t kernel_zero_point = 127;

    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(input.begin(), input.end(), std::ref(u8rng));
      std::generate(kernel.begin(), kernel.end(), std::ref(u8rng));
      std::generate(bias.begin(), bias.end(), std::ref(s32rng));
      std::fill(output.begin(), output.end(), 0xA5);

      // Compute reference results, without renormalization.
      if (has_bias()) {
        for (size_t i = 0; i < batch_size(); i++) {
          for (size_t oy = 0; oy < output_height(); oy++) {
            for (size_t ox = 0; ox < output_width(); ox++) {
              for (size_t g = 0; g < groups(); g++) {
                for (size_t oc = 0; oc < group_output_channels(); oc++) {
                  accumulators[(((i * output_height() + oy) * output_width() + ox) * groups() + g) * group_output_channels() + oc] =
                    bias[g * group_output_channels() + oc];
                }
              }
            }
          }
        }
      } else {
        std::fill(accumulators.begin(), accumulators.end(), 0);
      }
      for (size_t i = 0; i < batch_size(); i++) {
        for (size_t oy = 0; oy < output_height(); oy++) {
          for (size_t ox = 0; ox < output_width(); ox++) {
            for (size_t ky = 0; ky < kernel_height(); ky++) {
              const size_t y = oy + padding_top() - ky * dilation_height();
              const size_t iy = y / stride_height();
              if (iy * stride_height() == y && iy < input_height()) {
                for (size_t kx = 0; kx < kernel_width(); kx++) {
                  const size_t x = ox + padding_left() - kx * dilation_width();
                  const size_t ix = x / stride_width();
                  if (ix * stride_width() == x && ix < input_width()) {
                    for (size_t g = 0; g < groups(); g++) {
                      for (size_t oc = 0; oc < group_output_channels(); oc++) {
                        for (size_t ic = 0; ic < group_input_channels(); ic++) {
                          accumulators[(((i * output_height() + oy) * output_width() + ox) * groups() + g) * group_output_channels() + oc] +=
                            (int32_t(input[((i * input_height() + iy) * input_width() + ix) * input_pixel_stride() + g * group_input_channels() + ic]) - int32_t(input_zero_point)) *
                            (int32_t(kernel[(((g * group_output_channels() + oc) * kernel_height() + ky) * kernel_width() + kx) * group_input_channels() + ic]) - int32_t(kernel_zero_point));
                        }
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }

      // Compute renormalization parameters.
      const int32_t accumulated_min = *std::min_element(accumulators.cbegin(), accumulators.cend());
      const int32_t accumulated_max = *std::max_element(accumulators.cbegin(), accumulators.cend());

      const double output_scale = double(uint32_t(accumulated_max - accumulated_min)) / 255.0;
      const uint8_t output_zero_point = uint8_t(std::max(std::min(
        lrint(127.5 - 0.5 * double(accumulated_min + accumulated_max) / output_scale),
        long(std::numeric_limits<uint8_t>::max())), long(std::numeric_limits<uint8_t>::min())));

      // Renormalize reference results.
      std::transform(accumulators.cbegin(), accumulators.cend(), output_ref.begin(),
        [this, output_scale, output_zero_point](int32_t x) -> double {
          return std::max<double>(std::min<double>(double(x) / output_scale, double(qmax()) - output_zero_point), double(qmin()) - output_zero_point);
        });

      // Create, setup, run, and destroy Deconvolution operator.
      ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
      xnn_operator_t deconvolution_op = nullptr;

      ASSERT_EQ(xnn_status_success,
        xnn_create_deconvolution2d_nhwc_q8(
          padding_tf_same() ? 0 : padding_top(), padding_tf_same() ? 0 : padding_right(),
          padding_tf_same() ? 0 : padding_bottom(), padding_tf_same() ? 0 : padding_left(),
          kernel_height(), kernel_width(),
          stride_height(), stride_width(),
          dilation_height(), dilation_width(),
          groups(), group_input_channels(), group_output_channels(),
          input_pixel_stride(), output_pixel_stride(),
          input_zero_point, 1.0f /* input scale */,
          kernel_zero_point, 1.0f /* kernel scale */,
          kernel.data(), has_bias() ? bias.data() : nullptr,
          output_zero_point, output_scale, qmin(), qmax(),
          padding_tf_same() ? XNN_FLAG_TENSORFLOW_SAME_PADDING : 0,
          &deconvolution_op));

      // Smart pointer to automatically delete deconvolution_op.
      std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_deconvolution_op(deconvolution_op, xnn_delete_operator);

      ASSERT_EQ(xnn_status_success,
        xnn_setup_deconvolution2d_nhwc_q8(
          deconvolution_op,
          batch_size(), input_height(), input_width(),
          adjustment_height(), adjustment_width(),
          input.data(), output.data(),
          nullptr /* thread pool */));

      ASSERT_EQ(xnn_status_success,
        xnn_run_operator(deconvolution_op, nullptr /* thread pool */));

      // Verify results.
      for (size_t i = 0; i < batch_size(); i++) {
        for (size_t y = 0; y < output_height(); y++) {
          for (size_t x = 0; x < output_width(); x++) {
            for (size_t g = 0; g < groups(); g++) {
              for (size_t c = 0; c < group_output_channels(); c++) {
                ASSERT_LE(int32_t(output[((i * output_height() + y) * output_width() + x) * output_pixel_stride() + g * group_output_channels() + c]), int32_t(qmax()))
                     << "(x, y) = (" << x << ", " << y << "), group = " << g << ", channel = " << c;
                ASSERT_GE(int32_t(output[((i * output_height() + y) * output_width() + x) * output_pixel_stride() + g * group_output_channels() + c]), int32_t(qmin()))
                     << "(x, y) = (" << x << ", " << y << "), group = " << g << ", channel = " << c;
                ASSERT_NEAR(
                    output_ref[(((i * output_height() + y) * output_width() + x) * groups() + g) * group_output_channels() + c],
                    double(output[((i * output_height() + y) * output_width() + x) * output_pixel_stride() + g * group_output_channels() + c]) - double(output_zero_point),
                    0.9)
                  << "(x, y) = (" << x << ", " << y << "), group = " << g << ", channel = " << c;
              }
            }
          }
        }
      }
    }
  }

  void TestF32() const {
    std::random_device random_device;
    auto rng = std::mt19937(random_device());
    auto f32rng = std::bind(std::uniform_real_distribution<float>(0.1f, 1.0f), rng);

    std::vector<float> input(XNN_EXTRA_BYTES / sizeof(float) +
      (batch_size() * input_height() * input_width() - 1) * input_pixel_stride() + groups() * group_input_channels());
    std::vector<float> kernel(groups() * group_output_channels() * kernel_height() * kernel_width() * group_input_channels());
    std::vector<float> bias(groups() * group_output_channels());
    std::vector<float> output((batch_size() * output_height() * output_width() - 1) * output_pixel_stride() + groups() * group_output_channels());
    std::vector<float> output_ref(batch_size() * output_height() * output_width() * groups() * group_output_channels());

    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(input.begin(), input.end(), std::ref(f32rng));
      std::generate(kernel.begin(), kernel.end(), std::ref(f32rng));
      std::generate(bias.begin(), bias.end(), std::ref(f32rng));
      std::fill(output.begin(), output.end(), nanf(""));
      std::fill(output_ref.begin(), output_ref.end(), 0.0f);

      // Compute reference results, without clamping.
      if (has_bias()) {
        for (size_t i = 0; i < batch_size(); i++) {
          for (size_t oy = 0; oy < output_height(); oy++) {
            for (size_t ox = 0; ox < output_width(); ox++) {
              for (size_t g = 0; g < groups(); g++) {
                for (size_t oc = 0; oc < group_output_channels(); oc++) {
                  output_ref[(((i * output_height() + oy) * output_width() + ox) * groups() + g) * group_output_channels() + oc] =
                    bias[g * group_output_channels() + oc];
                }
              }
            }
          }
        }
      } else {
        std::fill(output_ref.begin(), output_ref.end(), 0.0f);
      }
      for (size_t i = 0; i < batch_size(); i++) {
        for (size_t oy = 0; oy < output_height(); oy++) {
          for (size_t ox = 0; ox < output_width(); ox++) {
            for (size_t ky = 0; ky < kernel_height(); ky++) {
              const size_t y = oy + padding_top() - ky * dilation_height();
              const size_t iy = y / stride_height();
              if (iy * stride_height() == y && iy < input_height()) {
                for (size_t kx = 0; kx < kernel_width(); kx++) {
                  const size_t x = ox + padding_left() - kx * dilation_width();
                  const size_t ix = x / stride_width();
                  if (ix * stride_width() == x && ix < input_width()) {
                    for (size_t g = 0; g < groups(); g++) {
                      for (size_t oc = 0; oc < group_output_channels(); oc++) {
                        for (size_t ic = 0; ic < group_input_channels(); ic++) {
                          output_ref[(((i * output_height() + oy) * output_width() + ox) * groups() + g) * group_output_channels() + oc] +=
                            input[((i * input_height() + iy) * input_width() + ix) * input_pixel_stride() + g * group_input_channels() + ic] *
                            kernel[(((g * group_output_channels() + oc) * kernel_height() + ky) * kernel_width() + kx) * group_input_channels() + ic];
                        }
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }

      // Compute clamping parameters.
      const float accumulated_min = *std::min_element(output_ref.cbegin(), output_ref.cend());
      const float accumulated_max = *std::max_element(output_ref.cbegin(), output_ref.cend());

      const float output_min = qmin() == 0 ? -std::numeric_limits<float>::infinity() :
        accumulated_min + (accumulated_max - accumulated_min) / 255.0f * float(qmin());
      const float output_max = qmax() == 255 ? std::numeric_limits<float>::infinity() :
        accumulated_max - (accumulated_max - accumulated_min) / 255.0f * float(255 - qmax());

      // Clamp reference results.
      for (float& value : output_ref) {
        value = std::max(std::min(value, output_max), output_min);
      }

      // Create, setup, run, and destroy Deconvolution operator.
      ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
      xnn_operator_t deconvolution_op = nullptr;

      ASSERT_EQ(xnn_status_success,
        xnn_create_deconvolution2d_nhwc_f32(
          padding_tf_same() ? 0 : padding_top(), padding_tf_same() ? 0 : padding_right(),
          padding_tf_same() ? 0 : padding_bottom(), padding_tf_same() ? 0 : padding_left(),
          kernel_height(), kernel_width(),
          stride_height(), stride_width(),
          dilation_height(), dilation_width(),
          groups(), group_input_channels(), group_output_channels(),
          input_pixel_stride(), output_pixel_stride(),
          kernel.data(), has_bias() ? bias.data() : nullptr,
          output_min, output_max,
          padding_tf_same() ? XNN_FLAG_TENSORFLOW_SAME_PADDING : 0,
          &deconvolution_op));

      // Smart pointer to automatically delete deconvolution_op.
      std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_deconvolution_op(deconvolution_op, xnn_delete_operator);

      ASSERT_EQ(xnn_status_success,
        xnn_setup_deconvolution2d_nhwc_f32(
          deconvolution_op,
          batch_size(), input_height(), input_width(),
          adjustment_height(), adjustment_width(),
          input.data(), output.data(),
          nullptr /* thread pool */));

      ASSERT_EQ(xnn_status_success,
        xnn_run_operator(deconvolution_op, nullptr /* thread pool */));

      // Verify results.
      for (size_t i = 0; i < batch_size(); i++) {
        for (size_t y = 0; y < output_height(); y++) {
          for (size_t x = 0; x < output_width(); x++) {
            for (size_t g = 0; g < groups(); g++) {
              for (size_t c = 0; c < group_output_channels(); c++) {
                ASSERT_GE(output[((i * output_height() + y) * output_width() + x) * output_pixel_stride() + g * group_output_channels() + c], output_min)
                  << "(x, y) = (" << x << ", " << y << "), group = " << g << ", channel = " << c;
                ASSERT_LE(output[((i * output_height() + y) * output_width() + x) * output_pixel_stride() + g * group_output_channels() + c], output_max)
                  << "(x, y) = (" << x << ", " << y << "), group = " << g << ", channel = " << c;
                ASSERT_NEAR(
                    output_ref[(((i * output_height() + y) * output_width() + x) * groups() + g) * group_output_channels() + c],
                    output[((i * output_height() + y) * output_width() + x) * output_pixel_stride() + g * group_output_channels() + c],
                    1.0e-4 * std::abs(output_ref[(((i * output_height() + y) * output_width() + x) * groups() + g) * group_output_channels() + c]))
                  << "(x, y) = (" << x << ", " << y << "), group = " << g << ", channel = " << c;
              }
            }
          }
        }
      }
    }
  }

  void TestSetupQ8() const {
    std::random_device random_device;
    auto rng = std::mt19937(random_device());
    auto s32rng = std::bind(std::uniform_int_distribution<int32_t>(-10000, 10000), rng);
    auto u8rng = std::bind(std::uniform_int_distribution<uint32_t>(0, std::numeric_limits<uint8_t>::max()), rng);

    std::vector<uint8_t> input(XNN_EXTRA_BYTES / sizeof(uint8_t) + std::max(
      (batch_size() * input_height() * input_width() - 1) * input_pixel_stride() + groups() * group_input_channels(),
      (next_batch_size() * next_input_height() * next_input_width() - 1) * input_pixel_stride() + groups() * group_input_channels()));
    std::vector<uint8_t> kernel(groups() * group_output_channels() * kernel_height() * kernel_width() * group_input_channels());
    std::vector<int32_t> bias(groups() * group_output_channels());
    std::vector<uint8_t> output(std::max(
      (batch_size() * output_height() * output_width() - 1) * output_pixel_stride() + groups() * group_output_channels(),
      (next_batch_size() * next_output_height() * next_output_width() - 1) * output_pixel_stride() + groups() * group_output_channels()));
    std::vector<int32_t> accumulators(batch_size() * output_height() * output_width() * groups() * group_output_channels());
    std::vector<double> output_ref(batch_size() * output_height() * output_width() * groups() * group_output_channels());
    std::vector<int32_t> next_accumulators(next_batch_size() * next_output_height() * next_output_width() * groups() * group_output_channels());
    std::vector<double> next_output_ref(next_batch_size() * next_output_height() * next_output_width() * groups() * group_output_channels());

    const uint8_t input_zero_point = 127;
    const uint8_t kernel_zero_point = 127;

    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(input.begin(), input.end(), std::ref(u8rng));
      std::generate(kernel.begin(), kernel.end(), std::ref(u8rng));
      std::generate(bias.begin(), bias.end(), std::ref(s32rng));
      std::fill(output.begin(), output.end(), 0xA5);

      // Compute reference results, without renormalization.
      if (has_bias()) {
        for (size_t i = 0; i < batch_size(); i++) {
          for (size_t oy = 0; oy < output_height(); oy++) {
            for (size_t ox = 0; ox < output_width(); ox++) {
              for (size_t g = 0; g < groups(); g++) {
                for (size_t oc = 0; oc < group_output_channels(); oc++) {
                  accumulators[(((i * output_height() + oy) * output_width() + ox) * groups() + g) * group_output_channels() + oc] =
                    bias[g * group_output_channels() + oc];
                }
              }
            }
          }
        }
      } else {
        std::fill(accumulators.begin(), accumulators.end(), 0);
      }
      for (size_t i = 0; i < batch_size(); i++) {
        for (size_t oy = 0; oy < output_height(); oy++) {
          for (size_t ox = 0; ox < output_width(); ox++) {
            for (size_t ky = 0; ky < kernel_height(); ky++) {
              const size_t y = oy + padding_top() - ky * dilation_height();
              const size_t iy = y / stride_height();
              if (iy * stride_height() == y && iy < input_height()) {
                for (size_t kx = 0; kx < kernel_width(); kx++) {
                  const size_t x = ox + padding_left() - kx * dilation_width();
                  const size_t ix = x / stride_width();
                  if (ix * stride_width() == x && ix < input_width()) {
                    for (size_t g = 0; g < groups(); g++) {
                      for (size_t oc = 0; oc < group_output_channels(); oc++) {
                        for (size_t ic = 0; ic < group_input_channels(); ic++) {
                          accumulators[(((i * output_height() + oy) * output_width() + ox) * groups() + g) * group_output_channels() + oc] +=
                            (int32_t(input[((i * input_height() + iy) * input_width() + ix) * input_pixel_stride() + g * group_input_channels() + ic]) - int32_t(input_zero_point)) *
                            (int32_t(kernel[(((g * group_output_channels() + oc) * kernel_height() + ky) * kernel_width() + kx) * group_input_channels() + ic]) - int32_t(kernel_zero_point));
                        }
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }

      // Compute renormalization parameters.
      const int32_t accumulated_min = *std::min_element(accumulators.cbegin(), accumulators.cend());
      const int32_t accumulated_max = *std::max_element(accumulators.cbegin(), accumulators.cend());

      const double output_scale = double(uint32_t(accumulated_max - accumulated_min)) / 255.0;
      const uint8_t output_zero_point = uint8_t(std::max(std::min(
        lrint(127.5 - 0.5 * double(accumulated_min + accumulated_max) / output_scale),
        long(std::numeric_limits<uint8_t>::max())), long(std::numeric_limits<uint8_t>::min())));

      // Renormalize reference results.
      std::transform(accumulators.cbegin(), accumulators.cend(), output_ref.begin(),
        [this, output_scale, output_zero_point](int32_t x) -> double {
          return std::max<double>(std::min<double>(double(x) / output_scale, double(qmax()) - output_zero_point), double(qmin()) - output_zero_point);
        });

      // Create, setup, and run Deconvolution operator once.
      ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
      xnn_operator_t deconvolution_op = nullptr;

      ASSERT_EQ(xnn_status_success,
        xnn_create_deconvolution2d_nhwc_q8(
          padding_top(), padding_right(), padding_bottom(), padding_left(),
          kernel_height(), kernel_width(),
          stride_height(), stride_width(),
          dilation_height(), dilation_width(),
          groups(), group_input_channels(), group_output_channels(),
          input_pixel_stride(), output_pixel_stride(),
          input_zero_point, 1.0f /* input scale */,
          kernel_zero_point, 1.0f /* kernel scale */,
          kernel.data(), has_bias() ? bias.data() : nullptr,
          output_zero_point, output_scale, qmin(), qmax(),
          0, &deconvolution_op));

      // Smart pointer to automatically delete deconvolution_op.
      std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_deconvolution_op(deconvolution_op, xnn_delete_operator);

      ASSERT_EQ(xnn_status_success,
        xnn_setup_deconvolution2d_nhwc_q8(
          deconvolution_op,
          batch_size(), input_height(), input_width(),
          adjustment_height(), adjustment_width(),
          input.data(), output.data(),
          nullptr /* thread pool */));

      ASSERT_EQ(xnn_status_success,
        xnn_run_operator(deconvolution_op, nullptr /* thread pool */));

      // Verify results of the first run.
      for (size_t i = 0; i < batch_size(); i++) {
        for (size_t y = 0; y < output_height(); y++) {
          for (size_t x = 0; x < output_width(); x++) {
            for (size_t g = 0; g < groups(); g++) {
              for (size_t c = 0; c < group_output_channels(); c++) {
                ASSERT_LE(int32_t(output[((i * output_height() + y) * output_width() + x) * output_pixel_stride() + g * group_output_channels() + c]), int32_t(qmax()))
                     << "(x, y) = (" << x << ", " << y << "), group = " << g << ", channel = " << c;
                ASSERT_GE(int32_t(output[((i * output_height() + y) * output_width() + x) * output_pixel_stride() + g * group_output_channels() + c]), int32_t(qmin()))
                     << "(x, y) = (" << x << ", " << y << "), group = " << g << ", channel = " << c;
                ASSERT_NEAR(
                    output_ref[(((i * output_height() + y) * output_width() + x) * groups() + g) * group_output_channels() + c],
                    double(output[((i * output_height() + y) * output_width() + x) * output_pixel_stride() + g * group_output_channels() + c]) - double(output_zero_point),
                    0.9)
                  << "(x, y) = (" << x << ", " << y << "), group = " << g << ", channel = " << c;
              }
            }
          }
        }
      }

      // Re-generate data for the second run.
      std::generate(input.begin(), input.end(), std::ref(u8rng));
      std::fill(output.begin(), output.end(), 0xA5);

      // Compute reference results for the second run, including renormalization.
      if (has_bias()) {
        for (size_t i = 0; i < next_batch_size(); i++) {
          for (size_t oy = 0; oy < next_output_height(); oy++) {
            for (size_t ox = 0; ox < next_output_width(); ox++) {
              for (size_t g = 0; g < groups(); g++) {
                for (size_t oc = 0; oc < group_output_channels(); oc++) {
                  next_accumulators[(((i * next_output_height() + oy) * next_output_width() + ox) * groups() + g) * group_output_channels() + oc] =
                    bias[g * group_output_channels() + oc];
                }
              }
            }
          }
        }
      } else {
        std::fill(next_accumulators.begin(), next_accumulators.end(), 0);
      }
      for (size_t i = 0; i < next_batch_size(); i++) {
        for (size_t oy = 0; oy < next_output_height(); oy++) {
          for (size_t ox = 0; ox < next_output_width(); ox++) {
            for (size_t ky = 0; ky < kernel_height(); ky++) {
              const size_t y = oy + padding_top() - ky * dilation_height();
              const size_t iy = y / stride_height();
              if (iy * stride_height() == y && iy < next_input_height()) {
                for (size_t kx = 0; kx < kernel_width(); kx++) {
                  const size_t x = ox + padding_left() - kx * dilation_width();
                  const size_t ix = x / stride_width();
                  if (ix * stride_width() == x && ix < next_input_width()) {
                    for (size_t g = 0; g < groups(); g++) {
                      for (size_t oc = 0; oc < group_output_channels(); oc++) {
                        for (size_t ic = 0; ic < group_input_channels(); ic++) {
                          next_accumulators[(((i * next_output_height() + oy) * next_output_width() + ox) * groups() + g) * group_output_channels() + oc] +=
                            (int32_t(input[((i * next_input_height() + iy) * next_input_width() + ix) * input_pixel_stride() + g * group_input_channels() + ic]) - int32_t(input_zero_point)) *
                            (int32_t(kernel[(((g * group_output_channels() + oc) * kernel_height() + ky) * kernel_width() + kx) * group_input_channels() + ic]) - int32_t(kernel_zero_point));
                        }
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
      std::transform(next_accumulators.cbegin(), next_accumulators.cend(), next_output_ref.begin(),
        [this, output_scale, output_zero_point](int32_t x) -> double {
          return std::max<double>(std::min<double>(double(x) / output_scale, double(qmax()) - output_zero_point), double(qmin()) - output_zero_point);
        });

      // Setup and run Deconvolution operator the second time, and destroy the operator.
      ASSERT_EQ(xnn_status_success,
        xnn_setup_deconvolution2d_nhwc_q8(
          deconvolution_op,
          next_batch_size(), next_input_height(), next_input_width(),
          adjustment_height(), adjustment_width(),
          input.data(), output.data(),
          nullptr /* thread pool */));

      ASSERT_EQ(xnn_status_success,
        xnn_run_operator(deconvolution_op, nullptr /* thread pool */));

      // Verify results of the second run.
      for (size_t i = 0; i < next_batch_size(); i++) {
        for (size_t y = 0; y < next_output_height(); y++) {
          for (size_t x = 0; x < next_output_width(); x++) {
            for (size_t g = 0; g < groups(); g++) {
              for (size_t c = 0; c < group_output_channels(); c++) {
                ASSERT_LE(int32_t(output[((i * next_output_height() + y) * next_output_width() + x) * output_pixel_stride() + g * group_output_channels() + c]), int32_t(qmax()))
                     << "(x, y) = (" << x << ", " << y << "), group = " << g << ", channel = " << c;
                ASSERT_GE(int32_t(output[((i * next_output_height() + y) * next_output_width() + x) * output_pixel_stride() + g * group_output_channels() + c]), int32_t(qmin()))
                     << "(x, y) = (" << x << ", " << y << "), group = " << g << ", channel = " << c;
                ASSERT_NEAR(
                    next_output_ref[(((i * next_output_height() + y) * next_output_width() + x) * groups() + g) * group_output_channels() + c],
                    double(output[((i * next_output_height() + y) * next_output_width() + x) * output_pixel_stride() + g * group_output_channels() + c]) - double(output_zero_point),
                    0.9)
                  << "(x, y) = (" << x << ", " << y << "), group = " << g << ", channel = " << c;
              }
            }
          }
        }
      }
    }
  }

  void TestSetupF32() const {
    std::random_device random_device;
    auto rng = std::mt19937(random_device());
    auto f32rng = std::bind(std::uniform_real_distribution<float>(0.1f, 1.0f), rng);

    std::vector<float> input(XNN_EXTRA_BYTES / sizeof(float) + std::max(
      (batch_size() * input_height() * input_width() - 1) * input_pixel_stride() + groups() * group_input_channels(),
      (next_batch_size() * next_input_height() * next_input_width() - 1) * input_pixel_stride() + groups() * group_input_channels()));
    std::vector<float> kernel(groups() * group_output_channels() * kernel_height() * kernel_width() * group_input_channels());
    std::vector<float> bias(groups() * group_output_channels());
    std::vector<float> output(std::max(
      (batch_size() * output_height() * output_width() - 1) * output_pixel_stride() + groups() * group_output_channels(),
      (next_batch_size() * next_output_height() * next_output_width() - 1) * output_pixel_stride() + groups() * group_output_channels()));
    std::vector<float> output_ref(batch_size() * output_height() * output_width() * groups() * group_output_channels());
    std::vector<float> next_output_ref(next_batch_size() * next_output_height() * next_output_width() * groups() * group_output_channels());

    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(input.begin(), input.end(), std::ref(f32rng));
      std::generate(kernel.begin(), kernel.end(), std::ref(f32rng));
      std::generate(bias.begin(), bias.end(), std::ref(f32rng));
      std::fill(output.begin(), output.end(), nanf(""));

      // Compute reference results, without clamping.
      if (has_bias()) {
        for (size_t i = 0; i < batch_size(); i++) {
          for (size_t oy = 0; oy < output_height(); oy++) {
            for (size_t ox = 0; ox < output_width(); ox++) {
              for (size_t g = 0; g < groups(); g++) {
                for (size_t oc = 0; oc < group_output_channels(); oc++) {
                  output_ref[(((i * output_height() + oy) * output_width() + ox) * groups() + g) * group_output_channels() + oc] =
                    bias[g * group_output_channels() + oc];
                }
              }
            }
          }
        }
      } else {
        std::fill(output_ref.begin(), output_ref.end(), 0.0f);
      }
      for (size_t i = 0; i < batch_size(); i++) {
        for (size_t oy = 0; oy < output_height(); oy++) {
          for (size_t ox = 0; ox < output_width(); ox++) {
            for (size_t ky = 0; ky < kernel_height(); ky++) {
              const size_t y = oy + padding_top() - ky * dilation_height();
              const size_t iy = y / stride_height();
              if (iy * stride_height() == y && iy < input_height()) {
                for (size_t kx = 0; kx < kernel_width(); kx++) {
                  const size_t x = ox + padding_left() - kx * dilation_width();
                  const size_t ix = x / stride_width();
                  if (ix * stride_width() == x && ix < input_width()) {
                    for (size_t g = 0; g < groups(); g++) {
                      for (size_t oc = 0; oc < group_output_channels(); oc++) {
                        for (size_t ic = 0; ic < group_input_channels(); ic++) {
                          output_ref[(((i * output_height() + oy) * output_width() + ox) * groups() + g) * group_output_channels() + oc] +=
                            input[((i * input_height() + iy) * input_width() + ix) * input_pixel_stride() + g * group_input_channels() + ic] *
                            kernel[(((g * group_output_channels() + oc) * kernel_height() + ky) * kernel_width() + kx) * group_input_channels() + ic];
                        }
                      }
                    }
                  }
                }
              }
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

      // Create, setup, and run Deconvolution operator once.
      ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
      xnn_operator_t deconvolution_op = nullptr;

      ASSERT_EQ(xnn_status_success,
        xnn_create_deconvolution2d_nhwc_f32(
          padding_top(), padding_right(), padding_bottom(), padding_left(),
          kernel_height(), kernel_width(),
          stride_height(), stride_width(),
          dilation_height(), dilation_width(),
          groups(), group_input_channels(), group_output_channels(),
          input_pixel_stride(), output_pixel_stride(),
          kernel.data(), has_bias() ? bias.data() : nullptr,
          output_min, output_max,
          0, &deconvolution_op));

      // Smart pointer to automatically delete deconvolution_op.
      std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_deconvolution_op(deconvolution_op, xnn_delete_operator);

      ASSERT_EQ(xnn_status_success,
        xnn_setup_deconvolution2d_nhwc_f32(
          deconvolution_op,
          batch_size(), input_height(), input_width(),
          adjustment_height(), adjustment_width(),
          input.data(), output.data(),
          nullptr /* thread pool */));

      ASSERT_EQ(xnn_status_success,
        xnn_run_operator(deconvolution_op, nullptr /* thread pool */));

      // Verify results of the first run.
      for (size_t i = 0; i < batch_size(); i++) {
        for (size_t y = 0; y < output_height(); y++) {
          for (size_t x = 0; x < output_width(); x++) {
            for (size_t g = 0; g < groups(); g++) {
              for (size_t c = 0; c < group_output_channels(); c++) {
                ASSERT_GE(output[((i * output_height() + y) * output_width() + x) * output_pixel_stride() + g * group_output_channels() + c], output_min)
                  << "(x, y) = (" << x << ", " << y << "), group = " << g << ", channel = " << c;
                ASSERT_LE(output[((i * output_height() + y) * output_width() + x) * output_pixel_stride() + g * group_output_channels() + c], output_max)
                  << "(x, y) = (" << x << ", " << y << "), group = " << g << ", channel = " << c;
                ASSERT_NEAR(
                    output_ref[(((i * output_height() + y) * output_width() + x) * groups() + g) * group_output_channels() + c],
                    output[((i * output_height() + y) * output_width() + x) * output_pixel_stride() + g * group_output_channels() + c],
                    1.0e-4 * std::abs(output_ref[(((i * output_height() + y) * output_width() + x) * groups() + g) * group_output_channels() + c]))
                  << "(x, y) = (" << x << ", " << y << "), group = " << g << ", channel = " << c;
              }
            }
          }
        }
      }

      // Re-generate data for the second run.
      std::generate(input.begin(), input.end(), std::ref(f32rng));
      std::fill(output.begin(), output.end(), nanf(""));

      // Compute reference results for the second run, including clamping.
      if (has_bias()) {
        for (size_t i = 0; i < next_batch_size(); i++) {
          for (size_t oy = 0; oy < next_output_height(); oy++) {
            for (size_t ox = 0; ox < next_output_width(); ox++) {
              for (size_t g = 0; g < groups(); g++) {
                for (size_t oc = 0; oc < group_output_channels(); oc++) {
                  next_output_ref[(((i * next_output_height() + oy) * next_output_width() + ox) * groups() + g) * group_output_channels() + oc] =
                    bias[g * group_output_channels() + oc];
                }
              }
            }
          }
        }
      } else {
        std::fill(next_output_ref.begin(), next_output_ref.end(), 0.0f);
      }
      for (size_t i = 0; i < next_batch_size(); i++) {
        for (size_t oy = 0; oy < next_output_height(); oy++) {
          for (size_t ox = 0; ox < next_output_width(); ox++) {
            for (size_t ky = 0; ky < kernel_height(); ky++) {
              const size_t y = oy + padding_top() - ky * dilation_height();
              const size_t iy = y / stride_height();
              if (iy * stride_height() == y && iy < next_input_height()) {
                for (size_t kx = 0; kx < kernel_width(); kx++) {
                  const size_t x = ox + padding_left() - kx * dilation_width();
                  const size_t ix = x / stride_width();
                  if (ix * stride_width() == x && ix < next_input_width()) {
                    for (size_t g = 0; g < groups(); g++) {
                      for (size_t oc = 0; oc < group_output_channels(); oc++) {
                        for (size_t ic = 0; ic < group_input_channels(); ic++) {
                          next_output_ref[(((i * next_output_height() + oy) * next_output_width() + ox) * groups() + g) * group_output_channels() + oc] +=
                            input[((i * next_input_height() + iy) * next_input_width() + ix) * input_pixel_stride() + g * group_input_channels() + ic] *
                            kernel[(((g * group_output_channels() + oc) * kernel_height() + ky) * kernel_width() + kx) * group_input_channels() + ic];
                        }
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
      for (float& value : next_output_ref) {
        value = std::max(std::min(value, output_max), output_min);
      }

      // Setup and run Deconvolution operator the second time, and destroy the operator.
      ASSERT_EQ(xnn_status_success,
        xnn_setup_deconvolution2d_nhwc_f32(
          deconvolution_op,
          next_batch_size(), next_input_height(), next_input_width(),
          adjustment_height(), adjustment_width(),
          input.data(), output.data(),
          nullptr /* thread pool */));

      ASSERT_EQ(xnn_status_success,
        xnn_run_operator(deconvolution_op, nullptr /* thread pool */));

      // Verify results of the second run.
      for (size_t i = 0; i < next_batch_size(); i++) {
        for (size_t y = 0; y < next_output_height(); y++) {
          for (size_t x = 0; x < next_output_width(); x++) {
            for (size_t g = 0; g < groups(); g++) {
              for (size_t c = 0; c < group_output_channels(); c++) {
                ASSERT_GE(output[((i * next_output_height() + y) * next_output_width() + x) * output_pixel_stride() + g * group_output_channels() + c], output_min)
                  << "(x, y) = (" << x << ", " << y << "), group = " << g << ", channel = " << c;
                ASSERT_LE(output[((i * next_output_height() + y) * next_output_width() + x) * output_pixel_stride() + g * group_output_channels() + c], output_max)
                  << "(x, y) = (" << x << ", " << y << "), group = " << g << ", channel = " << c;
                ASSERT_NEAR(
                    next_output_ref[(((i * next_output_height() + y) * next_output_width() + x) * groups() + g) * group_output_channels() + c],
                    output[((i * next_output_height() + y) * next_output_width() + x) * output_pixel_stride() + g * group_output_channels() + c],
                    1.0e-4 * std::abs(next_output_ref[(((i * next_output_height() + y) * next_output_width() + x) * groups() + g) * group_output_channels() + c]))
                  << "(x, y) = (" << x << ", " << y << "), group = " << g << ", channel = " << c;
              }
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
  uint32_t groups_{1};
  size_t group_input_channels_{1};
  size_t input_pixel_stride_{0};
  size_t group_output_channels_{1};
  size_t output_pixel_stride_{0};
  size_t batch_size_{1};
  uint32_t kernel_height_{1};
  uint32_t kernel_width_{1};
  uint32_t adjustment_height_{0};
  uint32_t adjustment_width_{0};
  uint32_t dilation_height_{1};
  uint32_t dilation_width_{1};
  uint32_t stride_height_{1};
  uint32_t stride_width_{1};
  size_t next_input_height_{0};
  size_t next_input_width_{0};
  size_t next_batch_size_{0};
  uint8_t qmin_{0};
  uint8_t qmax_{255};
  bool has_bias_{true};
  size_t iterations_{1};
};
