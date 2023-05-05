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
#include <cstddef>
#include <cstdlib>
#include <limits>
#include <random>
#include <vector>

#include <fp16/fp16.h>

#include <xnnpack.h>
#include <xnnpack/microparams-init.h>
#include <xnnpack/microfnptr.h>


class MaxPoolMicrokernelTester {
 public:
  inline MaxPoolMicrokernelTester& output_pixels(size_t output_pixels) {
    assert(output_pixels != 0);
    this->output_pixels_ = output_pixels;
    return *this;
  }

  inline size_t output_pixels() const {
    return this->output_pixels_;
  }

  inline MaxPoolMicrokernelTester& step(size_t step) {
    assert(step != 0);
    this->step_ = step;
    return *this;
  }

  inline size_t step() const {
    return this->step_;
  }

  inline MaxPoolMicrokernelTester& input_offset(size_t input_offset) {
    assert(input_offset != 0);
    this->input_offset_ = input_offset;
    return *this;
  }

  inline size_t input_offset() const {
    return this->input_offset_;
  }

  inline MaxPoolMicrokernelTester& pooling_elements(size_t pooling_elements) {
    assert(pooling_elements != 0);
    this->pooling_elements_ = pooling_elements;
    return *this;
  }

  inline size_t pooling_elements() const {
    return this->pooling_elements_;
  }

  inline size_t packed_pooling_elements() const {
    if (pooling_elements() <= primary_pooling_tile()) {
      return primary_pooling_tile();
    } else {
      return (pooling_elements() - primary_pooling_tile()) % incremental_pooling_tile() == 0 ? pooling_elements() : ((pooling_elements() - primary_pooling_tile()) / incremental_pooling_tile() + 1) * incremental_pooling_tile() + primary_pooling_tile();
    }
  }

  inline MaxPoolMicrokernelTester& pooling_tile(size_t primary_tile, size_t incremental_tile) {
    assert(primary_tile != 0);
    this->primary_pooling_tile_ = primary_tile;
    this->incremental_pooling_tile_ = incremental_tile;
    return *this;
  }

  inline MaxPoolMicrokernelTester& primary_pooling_tile(size_t primary_pooling_tile) {
    assert(primary_pooling_tile != 0);
    this->primary_pooling_tile_ = primary_pooling_tile;
    return *this;
  }

  inline size_t primary_pooling_tile() const {
    return this->primary_pooling_tile_;
  }

  inline MaxPoolMicrokernelTester& incremental_pooling_tile(size_t incremental_pooling_tile) {
    assert(incremental_pooling_tile != 0);
    this->incremental_pooling_tile_ = incremental_pooling_tile;
    return *this;
  }

  inline size_t incremental_pooling_tile() const {
    return this->incremental_pooling_tile_;
  }

  inline MaxPoolMicrokernelTester& channels(size_t channels) {
    assert(channels != 0);
    this->channels_ = channels;
    return *this;
  }

  inline size_t channels() const {
    return this->channels_;
  }

  inline MaxPoolMicrokernelTester& output_stride(size_t output_stride) {
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

  inline MaxPoolMicrokernelTester& qmin(int16_t qmin) {
    this->qmin_ = qmin;
    return *this;
  }

  inline int16_t qmin() const {
    return this->qmin_;
  }

  inline MaxPoolMicrokernelTester& qmax(int16_t qmax) {
    this->qmax_ = qmax;
    return *this;
  }

  inline int16_t qmax() const {
    return this->qmax_;
  }

  inline MaxPoolMicrokernelTester& iterations(size_t iterations) {
    this->iterations_ = iterations;
    return *this;
  }

  inline size_t iterations() const {
    return this->iterations_;
  }

  void Test(xnn_s8_maxpool_ukernel_fn maxpool, xnn_init_s8_minmax_params_fn init_params) const {
    ASSERT_GE(qmin(), std::numeric_limits<int8_t>::min());
    ASSERT_LE(qmax(), std::numeric_limits<int8_t>::max());
    ASSERT_LT(qmin(), qmax());

    std::random_device random_device;
    auto rng = std::mt19937(random_device());
    std::uniform_int_distribution<int32_t> i8dist(
      std::numeric_limits<int8_t>::min(), std::numeric_limits<int8_t>::max());

    std::vector<const int8_t*> indirect_input((output_pixels() - 1) * step() + packed_pooling_elements());
    std::vector<int8_t> input(XNN_EXTRA_BYTES / sizeof(int8_t) +
      indirect_input.size() * channels());
    std::vector<int8_t> output(XNN_EXTRA_BYTES / sizeof(int8_t) +
      (output_pixels() - 1) * output_stride() + channels());
    std::vector<int8_t> output_ref(output_pixels() * channels());
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      do {
        std::generate(input.begin(), input.end(), [&]() { return i8dist(rng); });
      } while (input.size() > 1 && *std::max_element(input.cbegin(), input.cend()) == *std::min_element(input.cbegin(), input.cend()));
      std::fill(output.begin(), output.end(), INT8_C(0xA5));

      for (size_t i = 0; i < (output_pixels() - 1) * step() + pooling_elements(); i++) {
        indirect_input[i] = input.data() + i * channels() - input_offset();
      }
      std::shuffle(indirect_input.begin(),
        indirect_input.begin() + (output_pixels() - 1) * step() + pooling_elements(), rng);

      // Prepare parameters.
      xnn_s8_minmax_params params;
      init_params(&params, static_cast<int8_t>(qmin()), static_cast<int8_t>(qmax()));

      // Compute reference results.
      for (size_t x = 0; x < output_pixels(); x++) {
        for (size_t c = 0; c < channels(); c++) {
          int8_t max_value = std::numeric_limits<int8_t>::min();
          for (size_t p = 0; p < pooling_elements(); p++) {
            max_value = std::max(max_value, indirect_input[x * step() + p][c + input_offset()]);
          }
          max_value = std::min(max_value, static_cast<int8_t>(qmax()));
          max_value = std::max(max_value, static_cast<int8_t>(qmin()));
          output_ref[x * channels() + c] = max_value;
        }
      }

      // Call optimized micro-kernel.
      maxpool(output_pixels(), pooling_elements(), channels(),
        indirect_input.data(), input_offset() * sizeof(int8_t), output.data(),
        (step() - packed_pooling_elements()) * sizeof(void*),
        (output_stride() - channels()) * sizeof(int8_t),
        &params);

      // Verify results.
      for (size_t x = 0; x < output_pixels(); x++) {
        for (size_t c = 0; c < channels(); c++) {
          ASSERT_GE(int16_t(output[x * output_stride() + c]), qmin())
            << "at pixel " << x << " / " << output_pixels() << ", channel " << c << " / " << channels()
            << ", pooling elements = " << pooling_elements() << ", step = " << step()
            << ", input offset = " << input_offset();
          ASSERT_LE(int16_t(output[x * output_stride() + c]), qmax())
            << "at pixel " << x << " / " << output_pixels() << ", channel " << c << " / " << channels()
            << ", pooling elements = " << pooling_elements() << ", step = " << step()
            << ", input offset = " << input_offset();
          EXPECT_EQ(int32_t(output_ref[x * channels() + c]), int32_t(output[x * output_stride() + c]))
            << "at pixel " << x << " / " << output_pixels() << ", channel " << c << " / " << channels()
            << ", pooling elements = " << pooling_elements() << ", step = " << step()
            << ", input offset = " << input_offset();
        }
      }
    }
  }

  void Test(xnn_u8_maxpool_ukernel_fn maxpool, xnn_init_u8_minmax_params_fn init_params) const {
    ASSERT_GE(qmin(), std::numeric_limits<uint8_t>::min());
    ASSERT_LE(qmax(), std::numeric_limits<uint8_t>::max());
    ASSERT_LT(qmin(), qmax());

    std::random_device random_device;
    auto rng = std::mt19937(random_device());
    std::uniform_int_distribution<int32_t> u8dist(
      std::numeric_limits<uint8_t>::min(), std::numeric_limits<uint8_t>::max());

    std::vector<const uint8_t*> indirect_input((output_pixels() - 1) * step() + packed_pooling_elements());
    std::vector<uint8_t> input(XNN_EXTRA_BYTES / sizeof(uint8_t) +
      indirect_input.size() * channels());
    std::vector<uint8_t> output(XNN_EXTRA_BYTES / sizeof(uint8_t) +
      (output_pixels() - 1) * output_stride() + channels());
    std::vector<uint8_t> output_ref(output_pixels() * channels());
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      do {
        std::generate(input.begin(), input.end(), [&]() { return u8dist(rng); });
      } while (input.size() > 1 && *std::max_element(input.cbegin(), input.cend()) == *std::min_element(input.cbegin(), input.cend()));
      std::fill(output.begin(), output.end(), UINT8_C(0xA5));

      for (size_t i = 0; i < (output_pixels() - 1) * step() + pooling_elements(); i++) {
        indirect_input[i] = input.data() + i * channels() - input_offset();
      }
      std::shuffle(indirect_input.begin(),
        indirect_input.begin() + (output_pixels() - 1) * step() + pooling_elements(), rng);

      // Prepare parameters.
      xnn_u8_minmax_params params;
      init_params(&params, static_cast<uint8_t>(qmin()), static_cast<uint8_t>(qmax()));

      // Compute reference results.
      for (size_t x = 0; x < output_pixels(); x++) {
        for (size_t c = 0; c < channels(); c++) {
          uint8_t max_value = 0;
          for (size_t p = 0; p < pooling_elements(); p++) {
            max_value = std::max(max_value, indirect_input[x * step() + p][c + input_offset()]);
          }
          max_value = std::min(max_value, static_cast<uint8_t>(qmax()));
          max_value = std::max(max_value, static_cast<uint8_t>(qmin()));
          output_ref[x * channels() + c] = max_value;
        }
      }

      // Call optimized micro-kernel.
      maxpool(output_pixels(), pooling_elements(), channels(),
        indirect_input.data(), input_offset() * sizeof(uint8_t), output.data(),
        (step() - packed_pooling_elements()) * sizeof(void*),
        (output_stride() - channels()) * sizeof(uint8_t),
        &params);

      // Verify results.
      for (size_t x = 0; x < output_pixels(); x++) {
        for (size_t c = 0; c < channels(); c++) {
          ASSERT_GE(int16_t(output[x * output_stride() + c]), qmin())
            << "at pixel " << x << " / " << output_pixels() << ", channel " << c << " / " << channels()
            << ", pooling elements = " << pooling_elements() << ", step = " << step()
            << ", input offset = " << input_offset();
          ASSERT_LE(int16_t(output[x * output_stride() + c]), qmax())
            << "at pixel " << x << " / " << output_pixels() << ", channel " << c << " / " << channels()
            << ", pooling elements = " << pooling_elements() << ", step = " << step()
            << ", input offset = " << input_offset();
          EXPECT_EQ(int32_t(output_ref[x * channels() + c]), int32_t(output[x * output_stride() + c]))
            << "at pixel " << x << " / " << output_pixels() << ", channel " << c << " / " << channels()
            << ", pooling elements = " << pooling_elements() << ", step = " << step()
            << ", input offset = " << input_offset();
        }
      }
    }
  }

  void Test(xnn_f16_maxpool_ukernel_fn maxpool, xnn_init_f16_minmax_params_fn init_params) const {
    ASSERT_LT(qmin(), qmax());

    std::random_device random_device;
    auto rng = std::mt19937(random_device());
    std::uniform_real_distribution<float> f32dist(-1.0f, 1.0f);

    std::vector<const uint16_t*> indirect_input((output_pixels() - 1) * step() + packed_pooling_elements());
    std::vector<uint16_t> input(XNN_EXTRA_BYTES / sizeof(uint16_t) +
      ((output_pixels() - 1) * step() + pooling_elements()) * channels());
    std::vector<uint16_t> output(XNN_EXTRA_BYTES / sizeof(uint16_t) +
      (output_pixels() - 1) * output_stride() + channels());
    std::vector<float> output_ref(output_pixels() * channels());
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(input.begin(), input.end(), [&]() { return fp16_ieee_from_fp32_value(f32dist(rng)); });
      std::fill(output.begin(), output.end(), UINT16_C(0x7E00) /* NaN */);

      for (size_t i = 0; i < (output_pixels() - 1) * step() + pooling_elements(); i++) {
        indirect_input[i] = input.data() + i * channels() - input_offset();
      }
      std::shuffle(indirect_input.begin(),
        indirect_input.begin() + (output_pixels() - 1) * step() + pooling_elements(), rng);

      // Compute reference results, without clamping.
      for (size_t x = 0; x < output_pixels(); x++) {
        for (size_t c = 0; c < channels(); c++) {
          float max_value = -std::numeric_limits<float>::infinity();
          for (size_t p = 0; p < pooling_elements(); p++) {
            max_value = std::max(max_value, fp16_ieee_to_fp32_value(indirect_input[x * step() + p][c + input_offset()]));
          }
          output_ref[x * channels() + c] = max_value;
        }
      }

      // Compute clamping parameters.
      const float accumulated_min = *std::min_element(output_ref.cbegin(), output_ref.cend());
      const float accumulated_max = *std::max_element(output_ref.cbegin(), output_ref.cend());
      const float accumulated_range = accumulated_max - accumulated_min;
      float output_min = accumulated_min + accumulated_range *
        (static_cast<float>(qmin() - std::numeric_limits<int16_t>::min()) /
          static_cast<float>(std::numeric_limits<int16_t>::max() - std::numeric_limits<int16_t>::min()));
      if (qmin() == std::numeric_limits<int16_t>::min()) {
        output_min = -std::numeric_limits<float>::infinity();
      }
      float output_max = accumulated_max - accumulated_range *
        (static_cast<float>(std::numeric_limits<int16_t>::max() - qmax()) /
          static_cast<float>(std::numeric_limits<int16_t>::max() - std::numeric_limits<int16_t>::min()));
      if (qmax() == std::numeric_limits<int16_t>::max()) {
        output_max = +std::numeric_limits<float>::infinity();
      }
      output_min = fp16_ieee_to_fp32_value(fp16_ieee_from_fp32_value(output_min));
      output_max = fp16_ieee_to_fp32_value(fp16_ieee_from_fp32_value(output_max));

      // Prepare parameters.
      xnn_f16_minmax_params params;
      init_params(&params, fp16_ieee_from_fp32_value(output_min), fp16_ieee_from_fp32_value(output_max));

      // Clamp reference results.
      for (float& output_value : output_ref) {
        output_value = std::max(std::min(output_value, output_max), output_min);
      }

      // Call optimized micro-kernel.
      maxpool(output_pixels(), pooling_elements(), channels(),
        reinterpret_cast<const void**>(indirect_input.data()), input_offset() * sizeof(uint16_t), output.data(),
        (step() - packed_pooling_elements()) * sizeof(void*),
        (output_stride() - channels()) * sizeof(uint16_t),
        &params);

      // Verify results.
      for (size_t x = 0; x < output_pixels(); x++) {
        for (size_t c = 0; c < channels(); c++) {
          ASSERT_GE(fp16_ieee_to_fp32_value(output[x * output_stride() + c]), output_min)
            << "at pixel " << x << " / " << output_pixels() << ", channel " << c << " / " << channels()
            << ", pooling elements = " << pooling_elements() << ", step = " << step()
            << ", input offset = " << input_offset();
          ASSERT_LE(fp16_ieee_to_fp32_value(output[x * output_stride() + c]), output_max)
            << "at pixel " << x << " / " << output_pixels() << ", channel " << c << " / " << channels()
            << ", pooling elements = " << pooling_elements() << ", step = " << step()
            << ", input offset = " << input_offset();
          EXPECT_EQ(fp16_ieee_to_fp32_value(output[x * output_stride() + c]), output_ref[x * channels() + c])
            << "at pixel " << x << " / " << output_pixels() << ", channel " << c << " / " << channels()
            << ", pooling elements = " << pooling_elements() << ", step = " << step()
            << ", input offset = " << input_offset();
        }
      }
    }
  }

  void Test(xnn_f32_maxpool_ukernel_fn maxpool, xnn_init_f32_minmax_params_fn init_params) const {
    ASSERT_LT(qmin(), qmax());

    std::random_device random_device;
    auto rng = std::mt19937(random_device());
    std::uniform_real_distribution<float> f32dist(-1.0f, 1.0f);

    std::vector<const float*> indirect_input((output_pixels() - 1) * step() + packed_pooling_elements());
    std::vector<float> input(XNN_EXTRA_BYTES / sizeof(float) +
      ((output_pixels() - 1) * step() + pooling_elements()) * channels());
    std::vector<float> output(XNN_EXTRA_BYTES / sizeof(float) +
      (output_pixels() - 1) * output_stride() + channels());
    std::vector<float> output_ref(output_pixels() * channels());
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(input.begin(), input.end(), [&]() { return f32dist(rng); });
      std::fill(output.begin(), output.end(), nanf(""));

      for (size_t i = 0; i < (output_pixels() - 1) * step() + pooling_elements(); i++) {
        indirect_input[i] = input.data() + i * channels() - input_offset();
      }
      std::shuffle(indirect_input.begin(),
        indirect_input.begin() + (output_pixels() - 1) * step() + pooling_elements(), rng);

      // Compute reference results, without clamping.
      for (size_t x = 0; x < output_pixels(); x++) {
        for (size_t c = 0; c < channels(); c++) {
          float max_value = -std::numeric_limits<float>::infinity();
          for (size_t p = 0; p < pooling_elements(); p++) {
            max_value = std::max(max_value, indirect_input[x * step() + p][c + input_offset()]);
          }
          output_ref[x * channels() + c] = max_value;
        }
      }

      // Compute clamping parameters.
      const float accumulated_min = *std::min_element(output_ref.cbegin(), output_ref.cend());
      const float accumulated_max = *std::max_element(output_ref.cbegin(), output_ref.cend());
      const float accumulated_range = accumulated_max - accumulated_min;
      float output_min = accumulated_min + accumulated_range *
        (static_cast<float>(qmin() - std::numeric_limits<int16_t>::min()) /
          static_cast<float>(std::numeric_limits<int16_t>::max() - std::numeric_limits<int16_t>::min()));
      if (qmin() == std::numeric_limits<int16_t>::min()) {
        output_min = -std::numeric_limits<float>::infinity();
      }
      float output_max = accumulated_max - accumulated_range *
        (static_cast<float>(std::numeric_limits<int16_t>::max() - qmax()) /
          static_cast<float>(std::numeric_limits<int16_t>::max() - std::numeric_limits<int16_t>::min()));
      if (qmax() == std::numeric_limits<int16_t>::max()) {
        output_max = +std::numeric_limits<float>::infinity();
      }

      // Prepare parameters.
      xnn_f32_minmax_params params;
      init_params(&params, output_min, output_max);

      // Clamp reference results.
      for (float& output_value : output_ref) {
        output_value = std::max(std::min(output_value, output_max), output_min);
      }

      // Call optimized micro-kernel.
      maxpool(output_pixels(), pooling_elements(), channels(),
        indirect_input.data(), input_offset() * sizeof(float), output.data(),
        (step() - packed_pooling_elements()) * sizeof(void*),
        (output_stride() - channels()) * sizeof(float),
        &params);

      // Verify results.
      for (size_t x = 0; x < output_pixels(); x++) {
        for (size_t c = 0; c < channels(); c++) {
          ASSERT_GE(output[x * output_stride() + c], output_min)
            << "at pixel " << x << " / " << output_pixels() << ", channel " << c << " / " << channels()
            << ", pooling elements = " << pooling_elements() << ", step = " << step()
            << ", input offset = " << input_offset();
          ASSERT_LE(output[x * output_stride() + c], output_max)
            << "at pixel " << x << " / " << output_pixels() << ", channel " << c << " / " << channels()
            << ", pooling elements = " << pooling_elements() << ", step = " << step()
            << ", input offset = " << input_offset();
          EXPECT_EQ(output_ref[x * channels() + c], output[x * output_stride() + c])
            << "at pixel " << x << " / " << output_pixels() << ", channel " << c << " / " << channels()
            << ", pooling elements = " << pooling_elements() << ", step = " << step()
            << ", input offset = " << input_offset();
        }
      }
    }
  }

 private:
  size_t output_pixels_{1};
  size_t pooling_elements_{1};
  size_t channels_{1};
  size_t input_offset_{0};
  size_t step_{1};
  size_t primary_pooling_tile_{1};
  size_t incremental_pooling_tile_{1};
  size_t output_stride_{0};
  int16_t qmin_{std::numeric_limits<int16_t>::min()};
  int16_t qmax_{std::numeric_limits<int16_t>::max()};
  size_t iterations_{3};
};
