// Copyright (c) Facebook, Inc. and its affiliates.
// All rights reserved.
//
// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <gtest/gtest.h>

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


class GlobalSumPoolingOperatorTester {
 public:
  inline GlobalSumPoolingOperatorTester& channels(size_t channels) {
    assert(channels != 0);
    this->channels_ = channels;
    return *this;
  }

  inline size_t channels() const {
    return this->channels_;
  }

  inline GlobalSumPoolingOperatorTester& width(size_t width) {
    assert(width != 0);
    this->width_ = width;
    return *this;
  }

  inline size_t width() const {
    return this->width_;
  }

  inline GlobalSumPoolingOperatorTester& input_stride(size_t input_stride) {
    assert(input_stride != 0);
    this->input_stride_ = input_stride;
    return *this;
  }

  inline size_t input_stride() const {
    if (this->input_stride_ == 0) {
      return channels();
    } else {
      assert(this->input_stride_ >= channels());
      return this->input_stride_;
    }
  }

  inline GlobalSumPoolingOperatorTester& output_stride(size_t output_stride) {
    assert(output_stride != 0);
    this->output_stride_ = output_stride;
    return *this;
  }

  inline size_t output_stride() const {
    if (this->output_stride_ == 0) {
      return channels();
    } else {
      assert(this->output_stride_ >= channels());
      return this->output_stride_;
    }
  }

  inline GlobalSumPoolingOperatorTester& batch_size(size_t batch_size) {
    assert(batch_size != 0);
    this->batch_size_ = batch_size;
    return *this;
  }

  inline size_t batch_size() const {
    return this->batch_size_;
  }

  inline GlobalSumPoolingOperatorTester& qmin(uint8_t qmin) {
    this->qmin_ = qmin;
    return *this;
  }

  inline uint8_t qmin() const {
    return this->qmin_;
  }

  inline GlobalSumPoolingOperatorTester& qmax(uint8_t qmax) {
    this->qmax_ = qmax;
    return *this;
  }

  inline uint8_t qmax() const {
    return this->qmax_;
  }

  inline GlobalSumPoolingOperatorTester& iterations(size_t iterations) {
    this->iterations_ = iterations;
    return *this;
  }

  inline size_t iterations() const {
    return this->iterations_;
  }

  void TestNWCxF16() const {
    std::random_device random_device;
    auto rng = std::mt19937(random_device());
    std::uniform_real_distribution<float> f32dist(1.0e-3f, 1.0f);

    std::vector<uint16_t> input((batch_size() * width() - 1) * input_stride() + channels() + XNN_EXTRA_BYTES / sizeof(uint16_t));
    std::vector<uint16_t> output(batch_size() * output_stride());
    std::vector<float> output_ref(batch_size() * channels());
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(input.begin(), input.end(), [&]() { return fp16_ieee_from_fp32_value(f32dist(rng)); });
      std::fill(output.begin(), output.end(), UINT16_C(0x7E00) /* NaN */);

      // Compute reference results, without clamping.
      for (size_t i = 0; i < batch_size(); i++) {
        for (size_t j = 0; j < channels(); j++) {
          float acc = 0.0f;
          for (size_t k = 0; k < width(); k++) {
            acc += fp16_ieee_to_fp32_value(input[(i * width() + k) * input_stride() + j]);
          }
          output_ref[i * channels() + j] = acc;
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

      // Create, setup, run, and destroy Global Sum Pooling operator.
      ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
      xnn_operator_t global_sum_pooling_op = nullptr;

      xnn_status status = xnn_create_global_sum_pooling_nwc_f16(
          output_min, output_max,
          0, &global_sum_pooling_op);
      if (status == xnn_status_unsupported_hardware) {
        GTEST_SKIP();
      }
      ASSERT_EQ(xnn_status_success, status);
      ASSERT_NE(nullptr, global_sum_pooling_op);

      // Smart pointer to automatically delete global_sum_pooling_op.
      std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_global_sum_pooling_op(global_sum_pooling_op, xnn_delete_operator);

      size_t workspace_size = 0;
      size_t workspace_alignment = 0;
      ASSERT_EQ(xnn_status_success,
        xnn_reshape_global_sum_pooling_nwc_f16(
          global_sum_pooling_op,
          batch_size(), width(),
          channels(), input_stride(), output_stride(),
          &workspace_size, &workspace_alignment,
          /*threadpool=*/nullptr));

      ASSERT_LE(workspace_alignment, XNN_ALLOCATION_ALIGNMENT);
      std::vector<char, AlignedAllocator<char, XNN_ALLOCATION_ALIGNMENT>> workspace(workspace_size);

      ASSERT_EQ(xnn_status_success,
        xnn_setup_global_sum_pooling_nwc_f16(
          global_sum_pooling_op,
          workspace.data(),
          input.data(), output.data()));

      ASSERT_EQ(xnn_status_success,
        xnn_run_operator(global_sum_pooling_op, /*threadpool=*/nullptr));

      // Verify results.
      for (size_t i = 0; i < batch_size(); i++) {
        for (size_t c = 0; c < channels(); c++) {
          EXPECT_LE(fp16_ieee_to_fp32_value(output[i * output_stride() + c]), output_max);
          EXPECT_GE(fp16_ieee_to_fp32_value(output[i * output_stride() + c]), output_min);
          EXPECT_NEAR(fp16_ieee_to_fp32_value(output[i * output_stride() + c]), output_ref[i * channels() + c], std::max(1.0e-4f, std::abs(output_ref[i * channels() + c]) * 1.0e-2f))
            << "at batch index " << i << " / " << batch_size()
            << ", channel " << c << " / " << channels();
        }
      }
    }
  }

  void TestNWCxF32() const {
    std::random_device random_device;
    auto rng = std::mt19937(random_device());
    std::uniform_real_distribution<float> f32dist;

    std::vector<float> input((batch_size() * width() - 1) * input_stride() + channels() + XNN_EXTRA_BYTES / sizeof(float));
    std::vector<float> output(batch_size() * output_stride());
    std::vector<float> output_ref(batch_size() * channels());
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(input.begin(), input.end(), [&]() { return f32dist(rng); });
      std::fill(output.begin(), output.end(), std::nanf(""));

      // Compute reference results, without clamping.
      for (size_t i = 0; i < batch_size(); i++) {
        for (size_t j = 0; j < channels(); j++) {
          float acc = 0.0f;
          for (size_t k = 0; k < width(); k++) {
            acc += input[(i * width() + k) * input_stride() + j];
          }
          output_ref[i * channels() + j] = acc;
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

      // Create, setup, run, and destroy Global Sum Pooling operator.
      ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
      xnn_operator_t global_sum_pooling_op = nullptr;

      xnn_status status = xnn_create_global_sum_pooling_nwc_f32(
          output_min, output_max,
          0, &global_sum_pooling_op);
      if (status == xnn_status_unsupported_hardware) {
        GTEST_SKIP();
      }
      ASSERT_EQ(xnn_status_success, status);
      ASSERT_NE(nullptr, global_sum_pooling_op);

      // Smart pointer to automatically delete global_sum_pooling_op.
      std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_global_sum_pooling_op(global_sum_pooling_op, xnn_delete_operator);

      size_t workspace_size = 0;
      size_t workspace_alignment = 0;
      ASSERT_EQ(xnn_status_success,
        xnn_reshape_global_sum_pooling_nwc_f32(
          global_sum_pooling_op,
          batch_size(), width(),
          channels(), input_stride(), output_stride(),
          &workspace_size, &workspace_alignment,
          /*threadpool=*/nullptr));

      ASSERT_LE(workspace_alignment, XNN_ALLOCATION_ALIGNMENT);
      std::vector<char, AlignedAllocator<char, XNN_ALLOCATION_ALIGNMENT>> workspace(workspace_size);

      ASSERT_EQ(xnn_status_success,
        xnn_setup_global_sum_pooling_nwc_f32(
          global_sum_pooling_op,
          workspace.data(),
          input.data(), output.data()));

      ASSERT_EQ(xnn_status_success,
        xnn_run_operator(global_sum_pooling_op, /*threadpool=*/nullptr));

      // Verify results.
      for (size_t i = 0; i < batch_size(); i++) {
        for (size_t c = 0; c < channels(); c++) {
          EXPECT_LE(output[i * output_stride() + c], output_max);
          EXPECT_GE(output[i * output_stride() + c], output_min);
          EXPECT_NEAR(output[i * output_stride() + c], output_ref[i * channels() + c], std::abs(output_ref[i * channels() + c]) * 1.0e-6f)
            << "at batch index " << i << " / " << batch_size()
            << ", channel " << c << " / " << channels();
        }
      }
    }
  }

 private:
  size_t batch_size_{1};
  size_t width_{1};
  size_t channels_{1};
  size_t input_stride_{0};
  size_t output_stride_{0};
  uint8_t qmin_{0};
  uint8_t qmax_{255};
  size_t iterations_{1};
};
