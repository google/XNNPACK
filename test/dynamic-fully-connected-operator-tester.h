// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <gtest/gtest.h>

#include <cassert>
#include <cstddef>
#include <cstdlib>
#include <algorithm>
#include <cmath>
#include <limits>
#include <random>
#include <vector>

#include <fp16/fp16.h>

#include <xnnpack.h>
#include <xnnpack/aligned-allocator.h>
#include <xnnpack/common.h>


class DynamicFullyConnectedOperatorTester {
 public:
  inline DynamicFullyConnectedOperatorTester& input_channels(size_t input_channels) {
    assert(input_channels >= 1);
    this->input_channels_ = input_channels;
    return *this;
  }

  inline size_t input_channels() const {
    return this->input_channels_;
  }

  inline DynamicFullyConnectedOperatorTester& output_channels(size_t output_channels) {
    assert(output_channels >= 1);
    this->output_channels_ = output_channels;
    return *this;
  }

  inline size_t output_channels() const {
    return this->output_channels_;
  }

  inline DynamicFullyConnectedOperatorTester& batch_size(size_t batch_size) {
    assert(batch_size >= 1);
    this->batch_size_ = batch_size;
    return *this;
  }

  inline size_t batch_size() const {
    return this->batch_size_;
  }

  inline DynamicFullyConnectedOperatorTester& input_stride(size_t input_stride) {
    assert(input_stride >= 1);
    this->input_stride_ = input_stride;
    return *this;
  }

  inline size_t input_stride() const {
    if (this->input_stride_ == 0) {
      return input_channels();
    } else {
      assert(this->input_stride_ >= input_channels());
      return this->input_stride_;
    }
  }

  inline DynamicFullyConnectedOperatorTester& output_stride(size_t output_stride) {
    assert(output_stride >= 1);
    this->output_stride_ = output_stride;
    return *this;
  }

  inline size_t output_stride() const {
    if (this->output_stride_ == 0) {
      return output_channels();
    } else {
      assert(this->output_stride_ >= output_channels());
      return this->output_stride_;
    }
  }

  inline DynamicFullyConnectedOperatorTester& qmin(uint8_t qmin) {
    this->qmin_ = qmin;
    return *this;
  }

  inline uint8_t qmin() const {
    return this->qmin_;
  }

  inline DynamicFullyConnectedOperatorTester& qmax(uint8_t qmax) {
    this->qmax_ = qmax;
    return *this;
  }

  inline uint8_t qmax() const {
    return this->qmax_;
  }

  inline DynamicFullyConnectedOperatorTester& transpose_weights(bool transpose_weights) {
    this->transpose_weights_ = transpose_weights;
    return *this;
  }

  inline bool transpose_weights() const {
    return this->transpose_weights_;
  }

  inline DynamicFullyConnectedOperatorTester& has_bias(bool has_bias) {
    this->has_bias_ = has_bias;
    return *this;
  }

  inline bool has_bias() const {
    return this->has_bias_;
  }

  inline DynamicFullyConnectedOperatorTester& iterations(size_t iterations) {
    this->iterations_ = iterations;
    return *this;
  }

  inline size_t iterations() const {
    return this->iterations_;
  }

  inline uint32_t flags() const {
    uint32_t flags = 0;
    if (transpose_weights()) {
      flags |= XNN_FLAG_TRANSPOSE_WEIGHTS;
    }
    return flags;
  };

  void TestF16() const {
    std::random_device random_device;
    auto rng = std::mt19937(random_device());
    std::uniform_real_distribution<float> f32dist(0.1f, 1.0f);

    std::vector<uint16_t> input(XNN_EXTRA_BYTES / sizeof(uint16_t) +
      (batch_size() - 1) * input_stride() + input_channels());
    std::vector<uint16_t> kernel(output_channels() * input_channels());
    std::vector<float> kernel_as_float(kernel.size());
    std::vector<uint16_t> bias(output_channels());
    std::vector<float> bias_as_float(bias.size());
    std::vector<uint16_t> output((batch_size() - 1) * output_stride() + output_channels());
    std::vector<float> output_ref(batch_size() * output_channels());

    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(input.begin(), input.end(), [&]() { return fp16_ieee_from_fp32_value(f32dist(rng)); });
      std::generate(kernel.begin(), kernel.end(), [&]() { return fp16_ieee_from_fp32_value(f32dist(rng)); });
      std::transform(kernel.cbegin(), kernel.cend(), kernel_as_float.begin(), fp16_ieee_to_fp32_value);
      std::generate(bias.begin(), bias.end(), [&]() { return fp16_ieee_from_fp32_value(f32dist(rng)); });
      std::transform(bias.cbegin(), bias.cend(), bias_as_float.begin(), fp16_ieee_to_fp32_value);
      std::fill(output.begin(), output.end(), UINT16_C(0x7E00) /* NaN */);

      // Compute reference results.
      if (has_bias()) {
        for (size_t i = 0; i < batch_size(); i++) {
          for (size_t oc = 0; oc < output_channels(); oc++) {
            output_ref[i * output_channels() + oc] = fp16_ieee_to_fp32_value(bias[oc]);
          }
        }
      } else {
        std::fill(output_ref.begin(), output_ref.end(), 0.0f);
      }

      if (transpose_weights()) {
        for (size_t i = 0; i < batch_size(); i++) {
          for (size_t oc = 0; oc < output_channels(); oc++) {
            for (size_t ic = 0; ic < input_channels(); ic++) {
              output_ref[i * output_channels() + oc] +=
                  fp16_ieee_to_fp32_value(input[i * input_stride() + ic]) *
                  fp16_ieee_to_fp32_value(kernel[ic * output_channels() + oc]);
            }
          }
        }
      } else {
        for (size_t i = 0; i < batch_size(); i++) {
          for (size_t oc = 0; oc < output_channels(); oc++) {
            for (size_t ic = 0; ic < input_channels(); ic++) {
              output_ref[i * output_channels() + oc] +=
                  fp16_ieee_to_fp32_value(input[i * input_stride() + ic]) *
                  fp16_ieee_to_fp32_value(kernel[oc * input_channels() + ic]);
            }
          }
        }
      }

      // Compute clamping parameters.
      const float accumulated_min = *std::min_element(output_ref.cbegin(), output_ref.cend());
      const float accumulated_max = *std::max_element(output_ref.cbegin(), output_ref.cend());
      const float accumulated_range = accumulated_max - accumulated_min;
      const float scaled_min = fp16_ieee_to_fp32_value(fp16_ieee_from_fp32_value(accumulated_min + accumulated_range / 255.0f * float(qmin())));
      const float scaled_max = fp16_ieee_to_fp32_value(fp16_ieee_from_fp32_value(accumulated_max - accumulated_range / 255.0f * float(255 - qmax())));
      const float output_min = scaled_min == scaled_max ? -std::numeric_limits<float>::infinity() : scaled_min;
      const float output_max = scaled_min == scaled_max ? +std::numeric_limits<float>::infinity() : scaled_max;

      // Clamp reference results.
      for (float& value : output_ref) {
        value = std::max(std::min(value, output_max), output_min);
      }

      // Create, setup, run, and destroy Dynamic Fully Connected operator.
      ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
      xnn_operator_t dynamic_fully_connected_op = nullptr;

      const xnn_status status = xnn_create_dynamic_fully_connected_nc_f16(
          output_min, output_max, flags(), &dynamic_fully_connected_op);
      if (status == xnn_status_unsupported_hardware) {
        GTEST_SKIP();
      }
      ASSERT_EQ(xnn_status_success, status);
      ASSERT_NE(nullptr, dynamic_fully_connected_op);

      // Smart pointer to automatically delete dynamic_fully_connected_op.
      std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_dynamic_fully_connected_op(
        dynamic_fully_connected_op, xnn_delete_operator);

      size_t workspace_size = 0;
      size_t workspace_alignment = 0;
      ASSERT_EQ(
        xnn_status_success,
        xnn_reshape_dynamic_fully_connected_nc_f16(
          dynamic_fully_connected_op, batch_size(), input_channels(), output_channels(), input_stride(),
          output_stride(), &workspace_size, &workspace_alignment, /*threadpool=*/nullptr));
      ASSERT_NE(workspace_size, 0);
      ASSERT_LE(workspace_alignment, XNN_ALLOCATION_ALIGNMENT);
      std::vector<char, AlignedAllocator<char, XNN_ALLOCATION_ALIGNMENT>> workspace(workspace_size);

      ASSERT_EQ(
        xnn_status_success, xnn_setup_dynamic_fully_connected_nc_f16(
                              dynamic_fully_connected_op, workspace.data(), input.data(), kernel.data(),
                              has_bias() ? bias.data() : nullptr, output.data()));

      ASSERT_EQ(xnn_status_success, xnn_run_operator(dynamic_fully_connected_op, /*threadpool=*/nullptr));

      VerifyF16(output, output_ref, output_max, output_min);
    }
  }

  void VerifyF16(const std::vector<uint16_t>& output,
                 const std::vector<float>& output_ref,
                 const float output_max,
                 const float output_min) const {
    for (size_t i = 0; i < batch_size(); i++) {
      for (size_t c = 0; c < output_channels(); c++) {
        ASSERT_LE(fp16_ieee_to_fp32_value(output[i * output_stride() + c]), output_max)
          << "batch index = " << i << ", channel = " << c;
        ASSERT_GE(fp16_ieee_to_fp32_value(output[i * output_stride() + c]), output_min)
          << "batch index = " << i << ", channel = " << c;
        EXPECT_NEAR(
            output_ref[i * output_channels() + c],
            fp16_ieee_to_fp32_value(output[i * output_stride() + c]),
            1.0e-2f * std::abs(output_ref[i * output_channels() + c]))
          << "batch index = " << i << ", channel = " << c;
      }
    }
  }

  void TestF32() const {
    std::random_device random_device;
    auto rng = std::mt19937(random_device());
    std::uniform_real_distribution<float> f32dist(0.1f, 1.0f);

    std::vector<float> input(XNN_EXTRA_BYTES / sizeof(float) +
      (batch_size() - 1) * input_stride() + input_channels());
    std::vector<float> kernel(output_channels() * input_channels());
    std::vector<float> bias(output_channels());
    std::vector<float> output((batch_size() - 1) * output_stride() + output_channels());
    std::vector<float> output_ref(batch_size() * output_channels());

    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(input.begin(), input.end(), [&]() { return f32dist(rng); });
      std::generate(kernel.begin(), kernel.end(), [&]() { return f32dist(rng); });
      std::generate(bias.begin(), bias.end(), [&]() { return f32dist(rng); });

      // Compute reference results.
      if (has_bias()) {
        for (size_t i = 0; i < batch_size(); i++) {
          for (size_t oc = 0; oc < output_channels(); oc++) {
            output_ref[i * output_channels() + oc] = bias[oc];
          }
        }
      } else {
        std::fill(output_ref.begin(), output_ref.end(), 0.0f);
      }
      if (transpose_weights()) {
        for (size_t i = 0; i < batch_size(); i++) {
          for (size_t oc = 0; oc < output_channels(); oc++) {
            for (size_t ic = 0; ic < input_channels(); ic++) {
              output_ref[i * output_channels() + oc] +=
                  input[i * input_stride() + ic] * kernel[ic * output_channels() + oc];
            }
          }
        }
      } else {
        for (size_t i = 0; i < batch_size(); i++) {
          for (size_t oc = 0; oc < output_channels(); oc++) {
            for (size_t ic = 0; ic < input_channels(); ic++) {
              output_ref[i * output_channels() + oc] +=
                  input[i * input_stride() + ic] * kernel[oc * input_channels() + ic];
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

      // Create, setup, run, and destroy Fully Connected operator.
      ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
      xnn_operator_t dynamic_fully_connected_op = nullptr;

      const xnn_status status = xnn_create_dynamic_fully_connected_nc_f32(
          output_min, output_max, flags(), &dynamic_fully_connected_op);
      if (status == xnn_status_unsupported_hardware) {
        GTEST_SKIP();
      }
      ASSERT_EQ(xnn_status_success, status);
      ASSERT_NE(nullptr, dynamic_fully_connected_op);

      // Smart pointer to automatically delete dynamic_fully_connected_op.
      std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_dynamic_fully_connected_op(
        dynamic_fully_connected_op, xnn_delete_operator);

      size_t workspace_size = 0;
      size_t workspace_alignment = 0;
      ASSERT_EQ(
        xnn_status_success,
        xnn_reshape_dynamic_fully_connected_nc_f32(
          dynamic_fully_connected_op, batch_size(), input_channels(), output_channels(), input_stride(),
          output_stride(), &workspace_size, &workspace_alignment, /*threadpool=*/nullptr));
      ASSERT_NE(workspace_size, 0);
      ASSERT_LE(workspace_alignment, XNN_ALLOCATION_ALIGNMENT);
      std::vector<char, AlignedAllocator<char, XNN_ALLOCATION_ALIGNMENT>> workspace(workspace_size);

      ASSERT_EQ(
        xnn_status_success, xnn_setup_dynamic_fully_connected_nc_f32(
                              dynamic_fully_connected_op, workspace.data(), input.data(), kernel.data(),
                              has_bias() ? bias.data() : nullptr, output.data()));

      ASSERT_EQ(xnn_status_success, xnn_run_operator(dynamic_fully_connected_op, /*threadpool=*/nullptr));

      VerifyF32(output, output_ref, output_max, output_min);
    }
  }

  void VerifyF32(const std::vector<float>& output,
                 const std::vector<float>& output_ref,
                 float output_max,
                 float output_min) const
  {
    // Verify results.
    for (size_t i = 0; i < batch_size(); i++) {
      for (size_t c = 0; c < output_channels(); c++) {
        ASSERT_LE(output[i * output_stride() + c], output_max)
            << "batch index = " << i << ", channel = " << c;
        ASSERT_GE(output[i * output_stride() + c], output_min)
            << "batch index = " << i << ", channel = " << c;
        EXPECT_NEAR(output_ref[i * output_channels() + c],
                    output[i * output_stride() + c],
                    1.0e-4f * std::abs(output_ref[i * output_channels() + c]))
            << "batch index = " << i << ", channel = " << c;
      }
    }
  }

 private:
  size_t input_channels_{1};
  size_t input_stride_{0};
  size_t output_channels_{1};
  size_t output_stride_{0};
  size_t batch_size_{1};
  uint8_t qmin_{0};
  uint8_t qmax_{255};
  bool transpose_weights_{false};
  bool has_bias_{true};
  size_t iterations_{1};
};
