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
#include <random>
#include <vector>

#include <xnnpack.h>


class ConvolutionSpNCHWOperatorTester {
 public:
  inline ConvolutionSpNCHWOperatorTester& padding(uint32_t padding) {
    this->padding_top_ = padding;
    this->padding_right_ = padding;
    this->padding_bottom_ = padding;
    this->padding_left_ = padding;
    return *this;
  }

  inline ConvolutionSpNCHWOperatorTester& padding(uint32_t padding_height, uint32_t padding_width) {
    this->padding_top_ = padding_height;
    this->padding_right_ = padding_width;
    this->padding_bottom_ = padding_height;
    this->padding_left_ = padding_width;
    return *this;
  }

  inline ConvolutionSpNCHWOperatorTester& padding_height(uint32_t padding_height) {
    this->padding_top_ = padding_height;
    this->padding_bottom_ = padding_height;
    return *this;
  }

  inline ConvolutionSpNCHWOperatorTester& padding_width(uint32_t padding_width) {
    this->padding_right_ = padding_width;
    this->padding_left_ = padding_width;
    return *this;
  }

  inline ConvolutionSpNCHWOperatorTester& padding_top(uint32_t padding_top) {
    this->padding_top_ = padding_top;
    return *this;
  }

  inline uint32_t padding_top() const {
    return this->padding_top_;
  }

  inline ConvolutionSpNCHWOperatorTester& padding_right(uint32_t padding_right) {
    this->padding_right_ = padding_right;
    return *this;
  }

  inline uint32_t padding_right() const {
    return this->padding_right_;
  }

  inline ConvolutionSpNCHWOperatorTester& padding_bottom(uint32_t padding_bottom) {
    this->padding_bottom_ = padding_bottom;
    return *this;
  }

  inline uint32_t padding_bottom() const {
    return this->padding_bottom_;
  }

  inline ConvolutionSpNCHWOperatorTester& padding_left(uint32_t padding_left) {
    this->padding_left_ = padding_left;
    return *this;
  }

  inline uint32_t padding_left() const {
    return this->padding_left_;
  }

  inline ConvolutionSpNCHWOperatorTester& input_size(uint32_t input_height, uint32_t input_width) {
    assert(input_height >= 1);
    assert(input_width >= 1);
    this->input_height_ = input_height;
    this->input_width_ = input_width;
    return *this;
  }

  inline ConvolutionSpNCHWOperatorTester& input_height(uint32_t input_height) {
    assert(input_height >= 1);
    this->input_height_ = input_height;
    return *this;
  }

  inline uint32_t input_height() const {
    return this->input_height_;
  }

  inline ConvolutionSpNCHWOperatorTester& input_width(uint32_t input_width) {
    assert(input_width >= 1);
    this->input_width_ = input_width;
    return *this;
  }

  inline uint32_t input_width() const {
    return this->input_width_;
  }

  inline ConvolutionSpNCHWOperatorTester& groups(uint32_t groups) {
    assert(groups >= 1);
    this->groups_ = groups;
    return *this;
  }

  inline uint32_t groups() const {
    return this->groups_;
  }

  inline ConvolutionSpNCHWOperatorTester& group_input_channels(size_t group_input_channels) {
    assert(group_input_channels >= 1);
    this->group_input_channels_ = group_input_channels;
    return *this;
  }

  inline size_t group_input_channels() const {
    return this->group_input_channels_;
  }

  inline ConvolutionSpNCHWOperatorTester& group_output_channels(size_t group_output_channels) {
    assert(group_output_channels >= 1);
    this->group_output_channels_ = group_output_channels;
    return *this;
  }

  inline size_t group_output_channels() const {
    return this->group_output_channels_;
  }

  inline ConvolutionSpNCHWOperatorTester& batch_size(size_t batch_size) {
    assert(batch_size >= 1);
    this->batch_size_ = batch_size;
    return *this;
  }

  inline size_t batch_size() const {
    return this->batch_size_;
  }

  inline ConvolutionSpNCHWOperatorTester& kernel_size(uint32_t kernel_size) {
    assert(kernel_size >= 1);
    this->kernel_height_ = kernel_size;
    this->kernel_width_ = kernel_size;
    return *this;
  }

  inline ConvolutionSpNCHWOperatorTester& kernel_size(uint32_t kernel_height, uint32_t kernel_width) {
    assert(kernel_height >= 1);
    assert(kernel_width >= 1);
    this->kernel_height_ = kernel_height;
    this->kernel_width_ = kernel_width;
    return *this;
  }

  inline ConvolutionSpNCHWOperatorTester& kernel_height(uint32_t kernel_height) {
    assert(kernel_height >= 1);
    this->kernel_height_ = kernel_height;
    return *this;
  }

  inline uint32_t kernel_height() const {
    return this->kernel_height_;
  }

  inline ConvolutionSpNCHWOperatorTester& kernel_width(uint32_t kernel_width) {
    assert(kernel_width >= 1);
    this->kernel_width_ = kernel_width;
    return *this;
  }

  inline uint32_t kernel_width() const {
    return this->kernel_width_;
  }

  inline ConvolutionSpNCHWOperatorTester& dilation(uint32_t dilation) {
    assert(dilation >= 1);
    this->dilation_height_ = dilation;
    this->dilation_width_ = dilation;
    return *this;
  }

  inline ConvolutionSpNCHWOperatorTester& dilation(uint32_t dilation_height, uint32_t dilation_width) {
    assert(dilation_height >= 1);
    assert(dilation_width >= 1);
    this->dilation_height_ = dilation_height;
    this->dilation_width_ = dilation_width;
    return *this;
  }

  inline ConvolutionSpNCHWOperatorTester& dilation_height(uint32_t dilation_height) {
    assert(dilation_height >= 1);
    this->dilation_height_ = dilation_height;
    return *this;
  }

  inline uint32_t dilation_height() const {
    return this->dilation_height_;
  }

  inline ConvolutionSpNCHWOperatorTester& dilation_width(uint32_t dilation_width) {
    assert(dilation_width >= 1);
    this->dilation_width_ = dilation_width;
    return *this;
  }

  inline uint32_t dilation_width() const {
    return this->dilation_width_;
  }

  inline ConvolutionSpNCHWOperatorTester& subsampling(uint32_t subsampling) {
    assert(subsampling >= 1);
    this->subsampling_height_ = subsampling;
    this->subsampling_width_ = subsampling;
    return *this;
  }

  inline ConvolutionSpNCHWOperatorTester& subsampling(uint32_t subsampling_height, uint32_t subsampling_width) {
    assert(subsampling_height >= 1);
    assert(subsampling_width >= 1);
    this->subsampling_height_ = subsampling_height;
    this->subsampling_width_ = subsampling_width;
    return *this;
  }

  inline ConvolutionSpNCHWOperatorTester& subsampling_height(uint32_t subsampling_height) {
    assert(subsampling_height >= 1);
    this->subsampling_height_ = subsampling_height;
    return *this;
  }

  inline uint32_t subsampling_height() const {
    return this->subsampling_height_;
  }

  inline ConvolutionSpNCHWOperatorTester& subsampling_width(uint32_t subsampling_width) {
    assert(subsampling_width >= 1);
    this->subsampling_width_ = subsampling_width;
    return *this;
  }

  inline uint32_t subsampling_width() const {
    return this->subsampling_width_;
  }

  inline ConvolutionSpNCHWOperatorTester& input_batch_stride(size_t input_batch_stride) {
    assert(input_batch_stride >= 1);
    this->input_batch_stride_ = input_batch_stride;
    return *this;
  }

  inline size_t input_batch_stride() const {
    if (this->input_batch_stride_ == 0) {
      return groups() * group_input_channels() * input_height() * input_width();
    } else {
      assert(this->input_batch_stride_ >= groups() * group_input_channels() * input_height() * input_width());
      return this->input_batch_stride_;
    }
  }

  inline ConvolutionSpNCHWOperatorTester& output_batch_stride(size_t output_batch_stride) {
    assert(output_batch_stride >= 1);
    this->output_batch_stride_ = output_batch_stride;
    return *this;
  }

  inline size_t output_batch_stride() const {
    if (this->output_batch_stride_ == 0) {
      return groups() * group_output_channels() * output_height() * output_width();
    } else {
      assert(this->output_batch_stride_ >= groups() * group_output_channels() * output_height() * output_width());
      return this->output_batch_stride_;
    }
  }

  inline uint32_t dilated_kernel_height() const {
    return (kernel_height() - 1) * dilation_height() + 1;
  }

  inline uint32_t dilated_kernel_width() const {
    return (kernel_width() - 1) * dilation_width() + 1;
  }

  inline size_t output_height() const {
    const size_t padded_input_height = padding_top() + input_height() + padding_bottom();
    if (padded_input_height <= dilated_kernel_height()) {
      return 1;
    } else {
      return (padded_input_height - dilated_kernel_height()) / subsampling_height() + 1;
    }
  }

  inline size_t output_width() const {
    const size_t padded_input_width = padding_left() + input_width() + padding_right();
    if (padded_input_width <= dilated_kernel_width()) {
      return 1;
    } else {
      return (padded_input_width - dilated_kernel_width()) / subsampling_width() + 1;
    }
  }

  inline ConvolutionSpNCHWOperatorTester& nhwc_input(bool nhwc_input) {
    this->nhwc_input_ = nhwc_input;
    return *this;
  }

  inline bool nhwc_input() const {
    return this->nhwc_input_;
  }

  inline ConvolutionSpNCHWOperatorTester& sparsity(float sparsity) {
    this->sparsity_ = sparsity;
    return *this;
  }

  inline float sparsity() const {
    return this->sparsity_;
  }

  inline ConvolutionSpNCHWOperatorTester& qmin(uint8_t qmin) {
    this->qmin_ = qmin;
    return *this;
  }

  inline uint8_t qmin() const {
    return this->qmin_;
  }

  inline ConvolutionSpNCHWOperatorTester& qmax(uint8_t qmax) {
    this->qmax_ = qmax;
    return *this;
  }

  inline uint8_t qmax() const {
    return this->qmax_;
  }

  inline ConvolutionSpNCHWOperatorTester& depthwise_layout(bool depthwise_layout) {
    this->depthwise_layout_ = depthwise_layout;
    return *this;
  }

  inline bool depthwise_layout() const {
    return this->depthwise_layout_;
  }

  inline ConvolutionSpNCHWOperatorTester& has_bias(bool has_bias) {
    this->has_bias_ = has_bias;
    return *this;
  }

  inline bool has_bias() const {
    return this->has_bias_;
  }

  inline ConvolutionSpNCHWOperatorTester& iterations(size_t iterations) {
    this->iterations_ = iterations;
    return *this;
  }

  inline size_t iterations() const {
    return this->iterations_;
  }

  void TestF32() const {
    ASSERT_FALSE(depthwise_layout());

    std::random_device random_device;
    auto rng = std::mt19937(random_device());
    auto f32rng = std::bind(std::uniform_real_distribution<float>(0.1f, 1.0f), rng);
    auto prng = std::bind(std::uniform_real_distribution<float>(), rng);

    std::vector<float> input(XNN_EXTRA_BYTES / sizeof(float) +
      batch_size() * input_batch_stride() + groups() * group_input_channels() * input_height() * input_width());
    std::vector<float> kernel(
      groups() * group_output_channels() * kernel_height() * kernel_width() * group_input_channels());
    std::vector<float> bias(groups() * group_output_channels());
    std::vector<float> output(
      batch_size() * output_batch_stride() + groups() * group_output_channels() * output_height() * output_width());
    std::vector<float> output_ref(batch_size() * groups() * group_output_channels() * output_height() * output_width());

    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(input.begin(), input.end(), std::ref(f32rng));
      std::generate(kernel.begin(), kernel.end(), std::ref(f32rng));
      for (float& k : kernel) {
        if (prng() <= sparsity()) {
          k = 0.0f;
        }
      }
      std::generate(bias.begin(), bias.end(), std::ref(f32rng));
      std::fill(output.begin(), output.end(), nanf(""));

      // Compute reference results, without clamping.
      if (has_bias()) {
        for (size_t i = 0; i < batch_size(); i++) {
          for (size_t oy = 0; oy < output_height(); oy++) {
            for (size_t ox = 0; ox < output_width(); ox++) {
              for (size_t g = 0; g < groups(); g++) {
                for (size_t oc = 0; oc < group_output_channels(); oc++) {
                  output_ref[(((i * groups() + g) * group_output_channels() + oc) * output_height() + oy) * output_width() + ox] =
                    bias[g * group_output_channels() + oc];
                }
              }
            }
          }
        }
      } else {
        std::fill(output_ref.begin(), output_ref.end(), 0.0f);
      }
      if (nhwc_input()) {
        for (size_t i = 0; i < batch_size(); i++) {
          for (size_t oy = 0; oy < output_height(); oy++) {
            for (size_t ox = 0; ox < output_width(); ox++) {
              for (size_t ky = 0; ky < kernel_height(); ky++) {
                const size_t iy = oy * subsampling_height() + ky * dilation_height() - padding_top();
                if (iy < input_height()) {
                  for (size_t kx = 0; kx < kernel_width(); kx++) {
                    const size_t ix = ox * subsampling_width() + kx * dilation_width() - padding_left();
                    if (ix < input_width()) {
                      for (size_t g = 0; g < groups(); g++) {
                        for (size_t oc = 0; oc < group_output_channels(); oc++) {
                          for (size_t ic = 0; ic < group_input_channels(); ic++) {
                            output_ref[(((i * groups() + g) * group_output_channels() + oc) * output_height() + oy) * output_width() + ox] +=
                              input[((((i * input_height() + iy) * input_width() + ix) * groups() + g) * group_input_channels() + ic)] *
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
      } else {
        for (size_t i = 0; i < batch_size(); i++) {
          for (size_t oy = 0; oy < output_height(); oy++) {
            for (size_t ox = 0; ox < output_width(); ox++) {
              for (size_t ky = 0; ky < kernel_height(); ky++) {
                const size_t iy = oy * subsampling_height() + ky * dilation_height() - padding_top();
                if (iy < input_height()) {
                  for (size_t kx = 0; kx < kernel_width(); kx++) {
                    const size_t ix = ox * subsampling_width() + kx * dilation_width() - padding_left();
                    if (ix < input_width()) {
                      for (size_t g = 0; g < groups(); g++) {
                        for (size_t oc = 0; oc < group_output_channels(); oc++) {
                          for (size_t ic = 0; ic < group_input_channels(); ic++) {
                            output_ref[(((i * groups() + g) * group_output_channels() + oc) * output_height() + oy) * output_width() + ox] +=
                              input[i * input_batch_stride() +
                                    ((g * group_input_channels() + ic) * input_height() + iy) * input_width() + ix] *
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

      // Create, setup, run, and destroy Convolution operator.
      ASSERT_EQ(xnn_status_success, xnn_initialize());
      xnn_operator_t convolution_op = nullptr;

      xnn_status status = xnn_create_convolution2d_spnchw_f32(
        padding_top(), padding_right(), padding_bottom(), padding_left(),
        kernel_height(), kernel_width(),
        subsampling_height(), subsampling_width(),
        dilation_height(), dilation_width(),
        groups(), group_input_channels(), group_output_channels(),
        kernel.data(), has_bias() ? bias.data() : nullptr,
        output_min, output_max,
        (depthwise_layout() ? XNN_FLAG_DEPTHWISE_CONVOLUTION : 0) | (nhwc_input() ? XNN_FLAG_INPUT_NHWC : 0),
        &convolution_op);
      if (status == xnn_status_unsupported_parameter) {
        GTEST_SKIP();
      }
      ASSERT_EQ(xnn_status_success, status);

      // Smart pointer to automatically delete convolution_op.
      std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_convolution_op(convolution_op, xnn_delete_operator);

      ASSERT_EQ(xnn_status_success,
        xnn_setup_convolution2d_spnchw_f32(
          convolution_op,
          batch_size(), input_batch_stride(), output_batch_stride(), input_height(), input_width(),
          input.data(), output.data(),
          nullptr /* thread pool */));

      ASSERT_EQ(xnn_status_success,
        xnn_run_operator(convolution_op, nullptr /* thread pool */));

      // Verify results.
      for (size_t i = 0; i < batch_size(); i++) {
        for (size_t y = 0; y < output_height(); y++) {
          for (size_t x = 0; x < output_width(); x++) {
            for (size_t g = 0; g < groups(); g++) {
              for (size_t c = 0; c < group_output_channels(); c++) {
                ASSERT_GE(output[i * output_batch_stride() + ((g * group_output_channels() + c) * output_height() + y) * output_width() + x], output_min)
                  << "(x, y) = (" << x << ", " << y << "), group = " << g << ", channel = " << c << ", image = " << i;
                ASSERT_LE(output[i * output_batch_stride() + ((g * group_output_channels() + c) * output_height() + y) * output_width() + x], output_max)
                  << "(x, y) = (" << x << ", " << y << "), group = " << g << ", channel = " << c << ", image = " << i;
                ASSERT_NEAR(
                    output_ref[(((i * groups() + g) * group_output_channels() + c) * output_height() + y) * output_width() + x],
                    output[i * output_batch_stride() + ((g * group_output_channels() + c) * output_height() + y) * output_width() + x],
                    1.0e-4 * std::abs(output_ref[(((i * groups() + g) * group_output_channels() + c) * output_height() + y) * output_width() + x]))
                  << "(x, y) = (" << x << ", " << y << "), group = " << g << ", channel = " << c << ", image = " << i;
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
  size_t input_height_{1};
  size_t input_width_{1};
  uint32_t groups_{1};
  size_t group_input_channels_{1};
  size_t input_batch_stride_{0};
  size_t group_output_channels_{1};
  size_t output_batch_stride_{0};
  size_t batch_size_{1};
  uint32_t kernel_height_{1};
  uint32_t kernel_width_{1};
  uint32_t dilation_height_{1};
  uint32_t dilation_width_{1};
  uint32_t subsampling_height_{1};
  uint32_t subsampling_width_{1};
  bool nhwc_input_{false};
  float sparsity_{0.5f};
  uint8_t qmin_{0};
  uint8_t qmax_{255};
  bool depthwise_layout_{false};
  bool has_bias_{true};
  size_t iterations_{1};
};
