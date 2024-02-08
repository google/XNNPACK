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
#include <memory>
#include <random>
#include <vector>

#include <fp16/fp16.h>
#include <pthreadpool.h>

#include <xnnpack.h>
#include <xnnpack/aligned-allocator.h>
#include <xnnpack/common.h>


class GlobalAveragePoolingOperatorTester {
 public:
  inline GlobalAveragePoolingOperatorTester& channels(size_t channels) {
    assert(channels != 0);
    this->channels_ = channels;
    return *this;
  }

  inline size_t channels() const {
    return this->channels_;
  }

  inline GlobalAveragePoolingOperatorTester& width(size_t width) {
    assert(width != 0);
    this->width_ = width;
    return *this;
  }

  inline size_t width() const {
    return this->width_;
  }

  inline GlobalAveragePoolingOperatorTester& input_stride(size_t input_stride) {
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

  inline GlobalAveragePoolingOperatorTester& output_stride(size_t output_stride) {
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

  inline GlobalAveragePoolingOperatorTester& batch_size(size_t batch_size) {
    assert(batch_size != 0);
    this->batch_size_ = batch_size;
    return *this;
  }

  inline size_t batch_size() const {
    return this->batch_size_;
  }

  inline GlobalAveragePoolingOperatorTester& input_scale(float input_scale) {
    assert(input_scale > 0.0f);
    assert(std::isnormal(input_scale));
    this->input_scale_ = input_scale;
    return *this;
  }

  inline float input_scale() const {
    return this->input_scale_;
  }

  inline GlobalAveragePoolingOperatorTester& input_zero_point(uint8_t input_zero_point) {
    this->input_zero_point_ = input_zero_point;
    return *this;
  }

  inline uint8_t input_zero_point() const {
    return this->input_zero_point_;
  }

  inline GlobalAveragePoolingOperatorTester& output_scale(float output_scale) {
    assert(output_scale > 0.0f);
    assert(std::isnormal(output_scale));
    this->output_scale_ = output_scale;
    return *this;
  }

  inline float output_scale() const {
    return this->output_scale_;
  }

  inline GlobalAveragePoolingOperatorTester& output_zero_point(uint8_t output_zero_point) {
    this->output_zero_point_ = output_zero_point;
    return *this;
  }

  inline uint8_t output_zero_point() const {
    return this->output_zero_point_;
  }

  inline GlobalAveragePoolingOperatorTester& qmin(uint8_t qmin) {
    this->qmin_ = qmin;
    return *this;
  }

  inline uint8_t qmin() const {
    return this->qmin_;
  }

  inline GlobalAveragePoolingOperatorTester& qmax(uint8_t qmax) {
    this->qmax_ = qmax;
    return *this;
  }

  inline uint8_t qmax() const {
    return this->qmax_;
  }

  inline GlobalAveragePoolingOperatorTester& multithreaded(size_t multithreaded) {
    this->multithreaded_ = multithreaded;
    return *this;
  }

  inline size_t multithreaded() const {
    return this->multithreaded_;
  }

  size_t num_threads() const {
    // Do not spin up excessive number of threads for tests.
    return multithreaded() ? 5 : 1;
  }

  inline GlobalAveragePoolingOperatorTester& iterations(size_t iterations) {
    this->iterations_ = iterations;
    return *this;
  }

  inline size_t iterations() const {
    return this->iterations_;
  }

  void TestNWCxQU8() const {
    std::random_device random_device;
    auto rng = std::mt19937(random_device());
    std::uniform_int_distribution<int32_t> u8dist(
      std::numeric_limits<uint8_t>::min(), std::numeric_limits<uint8_t>::max());

    std::vector<uint8_t> input((batch_size() * width() - 1) * input_stride() + channels() + XNN_EXTRA_BYTES / sizeof(uint8_t));
    std::vector<uint8_t> output(batch_size() * output_stride());
    std::vector<float> output_ref(batch_size() * channels());
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::unique_ptr<pthreadpool, decltype(&pthreadpool_destroy)> auto_threadpool{nullptr, pthreadpool_destroy};
      if (multithreaded()) {
        const pthreadpool_t threadpool = pthreadpool_create(num_threads());
        if (pthreadpool_get_threads_count(threadpool) <= 1) {
          GTEST_SKIP();
        } else {
          auto_threadpool.reset(threadpool);
        }
      }

      std::generate(input.begin(), input.end(), [&]() { return u8dist(rng); });
      std::fill(output.begin(), output.end(), UINT8_C(0xA5));

      // Compute reference results.
      const double scale = double(input_scale()) / (double(width()) * double(output_scale()));
      for (size_t i = 0; i < batch_size(); i++) {
        for (size_t j = 0; j < channels(); j++) {
          double acc = 0.0f;
          for (size_t k = 0; k < width(); k++) {
            acc += double(int32_t(input[(i * width() + k) * input_stride() + j]) - int32_t(input_zero_point()));
          }
          output_ref[i * channels() + j] = float(acc * scale + double(output_zero_point()));
          output_ref[i * channels() + j] = std::min<float>(output_ref[i * channels() + j], float(qmax()));
          output_ref[i * channels() + j] = std::max<float>(output_ref[i * channels() + j], float(qmin()));
        }
      }

      // Create, setup, run, and destroy Global Average Pooling operator.
      ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
      xnn_operator_t global_average_pooling_op = nullptr;

      xnn_status status = xnn_create_global_average_pooling_nwc_qu8(
          input_zero_point(), input_scale(),
          output_zero_point(), output_scale(),
          qmin(), qmax(),
          0, &global_average_pooling_op);
      if (status == xnn_status_unsupported_hardware) {
        GTEST_SKIP();
      }
      ASSERT_EQ(xnn_status_success, status);
      ASSERT_NE(nullptr, global_average_pooling_op);

      // Smart pointer to automatically delete global_average_pooling_op.
      std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_global_average_pooling_op(global_average_pooling_op, xnn_delete_operator);

      size_t workspace_size = 0;
      size_t workspace_alignment = 0;
      ASSERT_EQ(xnn_status_success,
        xnn_reshape_global_average_pooling_nwc_qu8(
          global_average_pooling_op,
          batch_size(), width(),
          channels(), input_stride(), output_stride(),
          &workspace_size, &workspace_alignment,
          auto_threadpool.get()));

      ASSERT_LE(workspace_alignment, XNN_ALLOCATION_ALIGNMENT);
      std::vector<char, AlignedAllocator<char, XNN_ALLOCATION_ALIGNMENT>> workspace(workspace_size);

      ASSERT_EQ(xnn_status_success,
        xnn_setup_global_average_pooling_nwc_qu8(
          global_average_pooling_op,
          workspace.data(),
          input.data(), output.data()));

      ASSERT_EQ(xnn_status_success,
        xnn_run_operator(global_average_pooling_op, auto_threadpool.get()));

      // Verify results.
      for (size_t i = 0; i < batch_size(); i++) {
        for (size_t c = 0; c < channels(); c++) {
          EXPECT_LE(uint32_t(output[i * output_stride() + c]), uint32_t(qmax()));
          EXPECT_GE(uint32_t(output[i * output_stride() + c]), uint32_t(qmin()));
          EXPECT_NEAR(float(int32_t(output[i * output_stride() + c])), output_ref[i * channels() + c], 0.80f)
            << "at batch index " << i << " / " << batch_size()
            << ", channel " << c << " / " << channels();
        }
      }
    }
  }

  void TestNWCxQS8() const {
    std::random_device random_device;
    auto rng = std::mt19937(random_device());
    std::uniform_int_distribution<int32_t> i8dist(
      std::numeric_limits<int8_t>::min(), std::numeric_limits<int8_t>::max());

    std::vector<int8_t> input((batch_size() * width() - 1) * input_stride() + channels() + XNN_EXTRA_BYTES / sizeof(int8_t));
    std::vector<int8_t> output(batch_size() * output_stride());
    std::vector<float> output_ref(batch_size() * channels());
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::unique_ptr<pthreadpool, decltype(&pthreadpool_destroy)> auto_threadpool{nullptr, pthreadpool_destroy};
      if (multithreaded()) {
        const pthreadpool_t threadpool = pthreadpool_create(num_threads());
        if (pthreadpool_get_threads_count(threadpool) <= 1) {
          GTEST_SKIP();
        } else {
          auto_threadpool.reset(threadpool);
        }
      }

      std::generate(input.begin(), input.end(), [&]() { return i8dist(rng); });
      std::fill(output.begin(), output.end(), INT8_C(0xA5));

      // Compute reference results.
      const double scale = double(input_scale()) / (double(width()) * double(output_scale()));
      for (size_t i = 0; i < batch_size(); i++) {
        for (size_t j = 0; j < channels(); j++) {
          double acc = 0.0f;
          for (size_t k = 0; k < width(); k++) {
            acc += double(int32_t(input[(i * width() + k) * input_stride() + j]) - int32_t(input_zero_point() - 0x80));
          }
          output_ref[i * channels() + j] = float(acc * scale + double(output_zero_point() - 0x80));
          output_ref[i * channels() + j] = std::min<float>(output_ref[i * channels() + j], float(qmax() - 0x80));
          output_ref[i * channels() + j] = std::max<float>(output_ref[i * channels() + j], float(qmin() - 0x80));
        }
      }

      // Create, setup, run, and destroy Global Average Pooling operator.
      ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
      xnn_operator_t global_average_pooling_op = nullptr;

      xnn_status status = xnn_create_global_average_pooling_nwc_qs8(
          int8_t(input_zero_point() - 0x80), input_scale(),
          int8_t(output_zero_point() - 0x80), output_scale(),
          int8_t(qmin() - 0x80), int8_t(qmax() - 0x80),
          0, &global_average_pooling_op);
      if (status == xnn_status_unsupported_hardware) {
        GTEST_SKIP();
      }
      ASSERT_EQ(xnn_status_success, status);
      ASSERT_NE(nullptr, global_average_pooling_op);

      // Smart pointer to automatically delete global_average_pooling_op.
      std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_global_average_pooling_op(global_average_pooling_op, xnn_delete_operator);

      size_t workspace_size = 0;
      size_t workspace_alignment = 0;
      ASSERT_EQ(xnn_status_success,
        xnn_reshape_global_average_pooling_nwc_qs8(
          global_average_pooling_op,
          batch_size(), width(),
          channels(), input_stride(), output_stride(),
          &workspace_size, &workspace_alignment,
          auto_threadpool.get()));

      ASSERT_LE(workspace_alignment, XNN_ALLOCATION_ALIGNMENT);
      std::vector<char, AlignedAllocator<char, XNN_ALLOCATION_ALIGNMENT>> workspace(workspace_size);

      ASSERT_EQ(xnn_status_success,
        xnn_setup_global_average_pooling_nwc_qs8(
          global_average_pooling_op,
          workspace.data(),
          input.data(), output.data()));

      ASSERT_EQ(xnn_status_success,
        xnn_run_operator(global_average_pooling_op, auto_threadpool.get()));

      // Verify results.
      for (size_t i = 0; i < batch_size(); i++) {
        for (size_t c = 0; c < channels(); c++) {
          EXPECT_LE(int32_t(output[i * output_stride() + c]), int32_t(qmax() - 0x80));
          EXPECT_GE(int32_t(output[i * output_stride() + c]), int32_t(qmin() - 0x80));
          EXPECT_NEAR(float(int32_t(output[i * output_stride() + c])), output_ref[i * channels() + c], 0.80f)
            << "at batch index " << i << " / " << batch_size()
            << ", channel " << c << " / " << channels();
        }
      }
    }
  }

  void TestNWCxF16() const {
    std::random_device random_device;
    auto rng = std::mt19937(random_device());
    std::uniform_real_distribution<float> f32dist(1.0e-3f, 1.0f);

    std::vector<uint16_t> input((batch_size() * width() - 1) * input_stride() + channels() + XNN_EXTRA_BYTES / sizeof(uint16_t));
    std::vector<uint16_t> output(batch_size() * output_stride());
    std::vector<float> output_ref(batch_size() * channels());
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::unique_ptr<pthreadpool, decltype(&pthreadpool_destroy)> auto_threadpool{nullptr, pthreadpool_destroy};
      if (multithreaded()) {
        const pthreadpool_t threadpool = pthreadpool_create(num_threads());
        if (pthreadpool_get_threads_count(threadpool) <= 1) {
          GTEST_SKIP();
        } else {
          auto_threadpool.reset(threadpool);
        }
      }

      std::generate(input.begin(), input.end(), [&]() { return fp16_ieee_from_fp32_value(f32dist(rng)); });
      std::fill(output.begin(), output.end(), UINT16_C(0x7E00) /* NaN */);

      // Compute reference results, without clamping.
      for (size_t i = 0; i < batch_size(); i++) {
        for (size_t j = 0; j < channels(); j++) {
          float acc = 0.0f;
          for (size_t k = 0; k < width(); k++) {
            acc += fp16_ieee_to_fp32_value(input[(i * width() + k) * input_stride() + j]);
          }
          output_ref[i * channels() + j] = acc / float(width());
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

      // Create, setup, run, and destroy Global Average Pooling operator.
      ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
      xnn_operator_t global_average_pooling_op = nullptr;

      xnn_status status = xnn_create_global_average_pooling_nwc_f16(
          output_min, output_max,
          0, &global_average_pooling_op);
      if (status == xnn_status_unsupported_hardware) {
        GTEST_SKIP();
      }
      ASSERT_EQ(xnn_status_success, status);
      ASSERT_NE(nullptr, global_average_pooling_op);

      // Smart pointer to automatically delete global_average_pooling_op.
      std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_global_average_pooling_op(global_average_pooling_op, xnn_delete_operator);

      size_t workspace_size = 0;
      size_t workspace_alignment = 0;
      ASSERT_EQ(xnn_status_success,
        xnn_reshape_global_average_pooling_nwc_f16(
          global_average_pooling_op,
          batch_size(), width(),
          channels(), input_stride(), output_stride(),
          &workspace_size, &workspace_alignment,
          auto_threadpool.get()));

      ASSERT_LE(workspace_alignment, XNN_ALLOCATION_ALIGNMENT);
      std::vector<char, AlignedAllocator<char, XNN_ALLOCATION_ALIGNMENT>> workspace(workspace_size);

      ASSERT_EQ(xnn_status_success,
        xnn_setup_global_average_pooling_nwc_f16(
          global_average_pooling_op,
          workspace.data(),
          input.data(), output.data()));

      ASSERT_EQ(xnn_status_success,
        xnn_run_operator(global_average_pooling_op, auto_threadpool.get()));

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
      std::unique_ptr<pthreadpool, decltype(&pthreadpool_destroy)> auto_threadpool{nullptr, pthreadpool_destroy};
      if (multithreaded()) {
        const pthreadpool_t threadpool = pthreadpool_create(num_threads());
        if (pthreadpool_get_threads_count(threadpool) <= 1) {
          GTEST_SKIP();
        } else {
          auto_threadpool.reset(threadpool);
        }
      }

      std::generate(input.begin(), input.end(), [&]() { return f32dist(rng); });
      std::fill(output.begin(), output.end(), std::nanf(""));

      // Compute reference results, without clamping.
      for (size_t i = 0; i < batch_size(); i++) {
        for (size_t j = 0; j < channels(); j++) {
          float acc = 0.0f;
          for (size_t k = 0; k < width(); k++) {
            acc += input[(i * width() + k) * input_stride() + j];
          }
          output_ref[i * channels() + j] = acc / float(width());
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

      // Create, setup, run, and destroy Global Average Pooling operator.
      ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
      xnn_operator_t global_average_pooling_op = nullptr;

      xnn_status status = xnn_create_global_average_pooling_nwc_f32(
          output_min, output_max,
          0, &global_average_pooling_op);
      if (status == xnn_status_unsupported_hardware) {
        GTEST_SKIP();
      }
      ASSERT_EQ(xnn_status_success, status);
      ASSERT_NE(nullptr, global_average_pooling_op);

      // Smart pointer to automatically delete global_average_pooling_op.
      std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_global_average_pooling_op(global_average_pooling_op, xnn_delete_operator);

      size_t workspace_size = 0;
      size_t workspace_alignment = 0;
      ASSERT_EQ(xnn_status_success,
        xnn_reshape_global_average_pooling_nwc_f32(
          global_average_pooling_op,
          batch_size(), width(),
          channels(), input_stride(), output_stride(),
          &workspace_size, &workspace_alignment,
          auto_threadpool.get()));

      ASSERT_LE(workspace_alignment, XNN_ALLOCATION_ALIGNMENT);
      std::vector<char, AlignedAllocator<char, XNN_ALLOCATION_ALIGNMENT>> workspace(workspace_size);

      ASSERT_EQ(xnn_status_success,
        xnn_setup_global_average_pooling_nwc_f32(
          global_average_pooling_op,
          workspace.data(),
          input.data(), output.data()));

      ASSERT_EQ(xnn_status_success,
        xnn_run_operator(global_average_pooling_op, auto_threadpool.get()));

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

  void TestNCWxF16() const {
    std::random_device random_device;
    auto rng = std::mt19937(random_device());
    std::uniform_real_distribution<float> f32dist(1.0e-3f, 1.0f);

    std::vector<uint16_t> input(batch_size() * channels() * width() + XNN_EXTRA_BYTES / sizeof(uint16_t));
    std::vector<uint16_t> output(batch_size() * channels());
    std::vector<float> output_ref(batch_size() * channels());
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::unique_ptr<pthreadpool, decltype(&pthreadpool_destroy)> auto_threadpool{nullptr, pthreadpool_destroy};
      if (multithreaded()) {
        const pthreadpool_t threadpool = pthreadpool_create(num_threads());
        if (pthreadpool_get_threads_count(threadpool) <= 1) {
          GTEST_SKIP();
        } else {
          auto_threadpool.reset(threadpool);
        }
      }

      std::generate(input.begin(), input.end(), [&]() { return fp16_ieee_from_fp32_value(f32dist(rng)); });
      std::fill(output.begin(), output.end(), UINT16_C(0x7E00) /* NaN */);

      // Compute reference results, without clamping.
      for (size_t i = 0; i < batch_size(); i++) {
        for (size_t j = 0; j < channels(); j++) {
          float acc = 0.0f;
          for (size_t k = 0; k < width(); k++) {
            acc += fp16_ieee_to_fp32_value(input[(i * channels() + j) * width() + k]);
          }
          output_ref[i * channels() + j] = acc / float(width());
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

      // Create, setup, run, and destroy Global Average Pooling operator.
      ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
      xnn_operator_t global_average_pooling_op = nullptr;

      xnn_status status = xnn_create_global_average_pooling_ncw_f16(
        output_min, output_max, 0, &global_average_pooling_op);
      if (status == xnn_status_unsupported_hardware) {
        GTEST_SKIP();
      }
      ASSERT_EQ(xnn_status_success, status);

      // Smart pointer to automatically delete global_average_pooling_op.
      std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_global_average_pooling_op(global_average_pooling_op, xnn_delete_operator);

      ASSERT_EQ(xnn_status_success,
        xnn_reshape_global_average_pooling_ncw_f16(
          global_average_pooling_op,
          batch_size(), width(), channels(),
          auto_threadpool.get()));

      ASSERT_EQ(xnn_status_success,
        xnn_setup_global_average_pooling_ncw_f16(
          global_average_pooling_op,
          input.data(), output.data()));

      ASSERT_EQ(xnn_status_success,
        xnn_run_operator(global_average_pooling_op, auto_threadpool.get()));

      // Verify results.
      for (size_t i = 0; i < batch_size(); i++) {
        for (size_t c = 0; c < channels(); c++) {
          EXPECT_LE(fp16_ieee_to_fp32_value(output[i * channels() + c]), output_max);
          EXPECT_GE(fp16_ieee_to_fp32_value(output[i * channels() + c]), output_min);
          EXPECT_NEAR(fp16_ieee_to_fp32_value(output[i * channels() + c]), output_ref[i * channels() + c], std::max(1.0e-4f, std::abs(output_ref[i * channels() + c]) * 1.0e-2f))
            << "at batch index " << i << " / " << batch_size()
            << ", channel " << c << " / " << channels();
        }
      }
    }
  }

  void TestNCWxF32() const {
    std::random_device random_device;
    auto rng = std::mt19937(random_device());
    std::uniform_real_distribution<float> f32dist;

    std::vector<float> input(batch_size() * channels() * width() + XNN_EXTRA_BYTES / sizeof(float));
    std::vector<float> output(batch_size() * channels());
    std::vector<float> output_ref(batch_size() * channels());
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::unique_ptr<pthreadpool, decltype(&pthreadpool_destroy)> auto_threadpool{nullptr, pthreadpool_destroy};
      if (multithreaded()) {
        const pthreadpool_t threadpool = pthreadpool_create(num_threads());
        if (pthreadpool_get_threads_count(threadpool) <= 1) {
          GTEST_SKIP();
        } else {
          auto_threadpool.reset(threadpool);
        }
      }

      std::generate(input.begin(), input.end(), [&]() { return f32dist(rng); });
      std::fill(output.begin(), output.end(), std::nanf(""));

      // Compute reference results, without clamping.
      for (size_t i = 0; i < batch_size(); i++) {
        for (size_t j = 0; j < channels(); j++) {
          float acc = 0.0f;
          for (size_t k = 0; k < width(); k++) {
            acc += input[(i * channels() + j) * width() + k];
          }
          output_ref[i * channels() + j] = acc / float(width());
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

      // Create, setup, run, and destroy Global Average Pooling operator.
      ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
      xnn_operator_t global_average_pooling_op = nullptr;

      xnn_status status = xnn_create_global_average_pooling_ncw_f32(
        output_min, output_max, 0, &global_average_pooling_op);
      if (status == xnn_status_unsupported_hardware) {
        GTEST_SKIP();
      }
      ASSERT_EQ(xnn_status_success, status);

      // Smart pointer to automatically delete global_average_pooling_op.
      std::unique_ptr<xnn_operator, decltype(&xnn_delete_operator)> auto_global_average_pooling_op(global_average_pooling_op, xnn_delete_operator);

      ASSERT_EQ(xnn_status_success,
        xnn_reshape_global_average_pooling_ncw_f32(
          global_average_pooling_op,
          batch_size(), width(), channels(),
          auto_threadpool.get()));

      ASSERT_EQ(xnn_status_success,
        xnn_setup_global_average_pooling_ncw_f32(
          global_average_pooling_op,
          input.data(), output.data()));

      ASSERT_EQ(xnn_status_success,
        xnn_run_operator(global_average_pooling_op, auto_threadpool.get()));

      // Verify results.
      for (size_t i = 0; i < batch_size(); i++) {
        for (size_t c = 0; c < channels(); c++) {
          EXPECT_LE(output[i * channels() + c], output_max);
          EXPECT_GE(output[i * channels() + c], output_min);
          EXPECT_NEAR(output[i * channels() + c], output_ref[i * channels() + c], std::abs(output_ref[i * channels() + c]) * 1.0e-5f)
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
  float input_scale_{1.0f};
  float output_scale_{1.0f};
  uint8_t input_zero_point_{121};
  uint8_t output_zero_point_{133};
  uint8_t qmin_{0};
  uint8_t qmax_{255};
  bool multithreaded_{false};
  size_t iterations_{1};
};
