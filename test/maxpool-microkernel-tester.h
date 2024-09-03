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
#include <random>
#include <vector>

#include <gtest/gtest.h>
#include "xnnpack.h"
#include "xnnpack/math.h"
#include "xnnpack/microfnptr.h"
#include "xnnpack/microparams.h"
#include "next_prime.h"
#include "replicable_random_device.h"

template<typename CType>
class MaxPoolMicrokernelTester {
 public:
  MaxPoolMicrokernelTester<CType>() {
    if (std::is_same<CType, uint8_t>::value || std::is_same<CType, int8_t>::value) {
        this->qmin_ = std::numeric_limits<CType>::min();
        this->qmax_ = std::numeric_limits<CType>::max();
    }
  }

  MaxPoolMicrokernelTester& output_pixels(size_t output_pixels) {
    assert(output_pixels != 0);
    this->output_pixels_ = output_pixels;
    return *this;
  }

  size_t output_pixels() const {
    return this->output_pixels_;
  }

  MaxPoolMicrokernelTester& step(size_t step) {
    assert(step != 0);
    this->step_ = step;
    return *this;
  }

  size_t step() const {
    return this->step_;
  }

  MaxPoolMicrokernelTester& input_offset(size_t input_offset) {
    assert(input_offset != 0);
    this->input_offset_ = input_offset;
    return *this;
  }

  size_t input_offset() const {
    return this->input_offset_;
  }

  MaxPoolMicrokernelTester& pooling_elements(size_t pooling_elements) {
    assert(pooling_elements != 0);
    this->pooling_elements_ = pooling_elements;
    return *this;
  }

  size_t pooling_elements() const {
    return this->pooling_elements_;
  }

  size_t packed_pooling_elements() const {
    if (pooling_elements() <= primary_pooling_tile()) {
      return primary_pooling_tile();
    } else {
      return (pooling_elements() - primary_pooling_tile()) % incremental_pooling_tile() == 0 ? pooling_elements() : ((pooling_elements() - primary_pooling_tile()) / incremental_pooling_tile() + 1) * incremental_pooling_tile() + primary_pooling_tile();
    }
  }

  MaxPoolMicrokernelTester& pooling_tile(size_t primary_tile, size_t incremental_tile) {
    assert(primary_tile != 0);
    this->primary_pooling_tile_ = primary_tile;
    this->incremental_pooling_tile_ = incremental_tile;
    return *this;
  }

  MaxPoolMicrokernelTester& primary_pooling_tile(size_t primary_pooling_tile) {
    assert(primary_pooling_tile != 0);
    this->primary_pooling_tile_ = primary_pooling_tile;
    return *this;
  }

  size_t primary_pooling_tile() const {
    return this->primary_pooling_tile_;
  }

  MaxPoolMicrokernelTester& incremental_pooling_tile(size_t incremental_pooling_tile) {
    assert(incremental_pooling_tile != 0);
    this->incremental_pooling_tile_ = incremental_pooling_tile;
    return *this;
  }

  size_t incremental_pooling_tile() const {
    return this->incremental_pooling_tile_;
  }

  MaxPoolMicrokernelTester& channels(size_t channels) {
    assert(channels != 0);
    this->channels_ = channels;
    return *this;
  }

  size_t channels() const {
    return this->channels_;
  }

  MaxPoolMicrokernelTester& output_stride(size_t output_stride) {
    assert(output_stride != 0);
    this->output_stride_ = output_stride;
    return *this;
  }

  size_t output_stride() const {
    if (this->output_stride_ == 0) {
      return channels();
    } else {
      assert(this->output_stride_ >= channels());
      return this->output_stride_;
    }
  }

  MaxPoolMicrokernelTester& qmin(int16_t qmin) {
    this->qmin_ = qmin;
    return *this;
  }

  int16_t qmin() const {
    return this->qmin_;
  }

  MaxPoolMicrokernelTester& qmax(int16_t qmax) {
    this->qmax_ = qmax;
    return *this;
  }

  int16_t qmax() const {
    return this->qmax_;
  }

  MaxPoolMicrokernelTester& iterations(size_t iterations) {
    this->iterations_ = iterations;
    return *this;
  }

  size_t iterations() const {
    return this->iterations_;
  }

  void Test(xnn_s8_maxpool_ukernel_fn maxpool, xnn_init_s8_minmax_params_fn init_params) const {
    ASSERT_GE(qmin(), std::numeric_limits<int8_t>::min());
    ASSERT_LE(qmax(), std::numeric_limits<int8_t>::max());
    ASSERT_LT(qmin(), qmax());

    xnnpack::ReplicableRandomDevice rng;
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

    xnnpack::ReplicableRandomDevice rng;
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

    xnnpack::ReplicableRandomDevice rng;
    std::uniform_real_distribution<float> f32dist(-1.0f, 1.0f);

    std::vector<const xnn_float16*> indirect_input((output_pixels() - 1) * step() + packed_pooling_elements());
    std::vector<xnn_float16> input(XNN_EXTRA_BYTES / sizeof(xnn_float16) +
      ((output_pixels() - 1) * step() + pooling_elements()) * channels());
    std::vector<xnn_float16> output(XNN_EXTRA_BYTES / sizeof(xnn_float16) +
      (output_pixels() - 1) * output_stride() + channels());
    std::vector<float> output_ref(output_pixels() * channels());
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(input.begin(), input.end(), [&]() { return xnn_float16_from_float(f32dist(rng)); });
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
            max_value = std::max(max_value, xnn_float16_to_float(indirect_input[x * step() + p][c + input_offset()]));
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
      output_min = xnn_float16_to_float(xnn_float16_from_float(output_min));
      output_max = xnn_float16_to_float(xnn_float16_from_float(output_max));

      // Prepare parameters.
      xnn_f16_minmax_params params;
      init_params(&params, xnn_float16_from_float(output_min), xnn_float16_from_float(output_max));

      // Clamp reference results.
      for (float& output_value : output_ref) {
        output_value = std::max(std::min(output_value, output_max), output_min);
      }

      // Call optimized micro-kernel.
      maxpool(output_pixels(), pooling_elements(), channels(),
        reinterpret_cast<const xnn_float16**>(indirect_input.data()), input_offset() * sizeof(xnn_float16), output.data(),
        (step() - packed_pooling_elements()) * sizeof(void*),
        (output_stride() - channels()) * sizeof(xnn_float16),
        &params);

      // Verify results.
      for (size_t x = 0; x < output_pixels(); x++) {
        for (size_t c = 0; c < channels(); c++) {
          ASSERT_GE(xnn_float16_to_float(output[x * output_stride() + c]), output_min)
            << "at pixel " << x << " / " << output_pixels() << ", channel " << c << " / " << channels()
            << ", pooling elements = " << pooling_elements() << ", step = " << step()
            << ", input offset = " << input_offset();
          ASSERT_LE(xnn_float16_to_float(output[x * output_stride() + c]), output_max)
            << "at pixel " << x << " / " << output_pixels() << ", channel " << c << " / " << channels()
            << ", pooling elements = " << pooling_elements() << ", step = " << step()
            << ", input offset = " << input_offset();
          EXPECT_EQ(xnn_float16_to_float(output[x * output_stride() + c]), output_ref[x * channels() + c])
            << "at pixel " << x << " / " << output_pixels() << ", channel " << c << " / " << channels()
            << ", pooling elements = " << pooling_elements() << ", step = " << step()
            << ", input offset = " << input_offset();
        }
      }
    }
  }

  void Test(xnn_f32_maxpool_ukernel_fn maxpool, xnn_init_f32_minmax_params_fn init_params) const {
    ASSERT_LT(qmin(), qmax());

    xnnpack::ReplicableRandomDevice rng;
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

template<typename KernelFn_, typename ParamsFn_>
struct XnnMaxpoolTestParam {
  using KernelFn = KernelFn_;
  using ParamsFn = ParamsFn_;

  const char *name;
  KernelFn kernel_fn;
  ParamsFn params_fn;
  uint64_t arch_flags;
  size_t channel_tile, channel_scaled_tile, primary_tile, incremental_tile;
  int16_t qmin, qmax;
};

template<typename KernelFn_, typename ParamsFn_, typename CType>
class XnnMaxpoolTest : public testing::TestWithParam<XnnMaxpoolTestParam<KernelFn_, ParamsFn_>> {
protected:
  const XnnMaxpoolTestParam<KernelFn_, ParamsFn_>& TestParam() const {
    return this->GetParam();
  }

  void channels_eq_channel_tile_unipass_fulltile() {
    TEST_REQUIRES_ARCH_FLAGS(TestParam().arch_flags);
    const size_t channel_tile = TestParam().channel_scaled_tile;
    MaxPoolMicrokernelTester<CType>()
      .pooling_elements(TestParam().primary_tile)
      .pooling_tile(TestParam().primary_tile, TestParam().incremental_tile)
      .channels(channel_tile)
      .Test(TestParam().kernel_fn, TestParam().params_fn);
  }

  void channels_eq_channel_tile_unipass_fulltile_with_input_offset() {
    TEST_REQUIRES_ARCH_FLAGS(TestParam().arch_flags);
    const size_t channel_tile = TestParam().channel_scaled_tile;
    MaxPoolMicrokernelTester<CType>()
      .pooling_elements(TestParam().primary_tile)
      .pooling_tile(TestParam().primary_tile, TestParam().incremental_tile)
      .channels(channel_tile)
      .input_offset(TestParam().channel_scaled_tile == TestParam().channel_tile ?
                    xnnpack::NextPrime(TestParam().channel_tile + 1) :
                    channel_tile + 1)
      .Test(TestParam().kernel_fn, TestParam().params_fn);
  }

  void channels_eq_channel_tile_unipass_fulltile_with_qmin() {
    TEST_REQUIRES_ARCH_FLAGS(TestParam().arch_flags);
    const size_t channel_tile = TestParam().channel_scaled_tile;
    MaxPoolMicrokernelTester<CType>()
      .pooling_elements(TestParam().primary_tile)
      .pooling_tile(TestParam().primary_tile, TestParam().incremental_tile)
      .channels(channel_tile)
      .qmin(TestParam().qmin)
      .Test(TestParam().kernel_fn, TestParam().params_fn);
  }

  void channels_eq_channel_tile_unipass_fulltile_with_qmax() {
    TEST_REQUIRES_ARCH_FLAGS(TestParam().arch_flags);
    const size_t channel_tile = TestParam().channel_scaled_tile;
    MaxPoolMicrokernelTester<CType>()
      .pooling_elements(TestParam().primary_tile)
      .pooling_tile(TestParam().primary_tile, TestParam().incremental_tile)
      .channels(channel_tile)
      .qmax(TestParam().qmax)
      .Test(TestParam().kernel_fn, TestParam().params_fn);
  }

  void channels_eq_channel_tile_unipass_subtile() {
    TEST_REQUIRES_ARCH_FLAGS(TestParam().arch_flags);
    const size_t channel_tile = TestParam().channel_scaled_tile;
    for (size_t pooling_elements = 2; pooling_elements < TestParam().primary_tile; pooling_elements++) {
      MaxPoolMicrokernelTester<CType>()
        .pooling_elements(pooling_elements)
        .pooling_tile(TestParam().primary_tile, TestParam().incremental_tile)
        .channels(channel_tile)
        .Test(TestParam().kernel_fn, TestParam().params_fn);
    }
  }

  void channels_eq_channel_tile_unipass_subtile_with_input_offset() {
    TEST_REQUIRES_ARCH_FLAGS(TestParam().arch_flags);
    const size_t channel_tile = TestParam().channel_scaled_tile;
    for (size_t pooling_elements = 2; pooling_elements < TestParam().primary_tile; pooling_elements++) {
      MaxPoolMicrokernelTester<CType>()
        .pooling_elements(pooling_elements)
        .pooling_tile(TestParam().primary_tile, TestParam().incremental_tile)
        .channels(channel_tile)
        .input_offset(TestParam().channel_scaled_tile == TestParam().channel_tile ?
                      xnnpack::NextPrime(TestParam().channel_tile + 1) :
                      channel_tile + 1)
        .Test(TestParam().kernel_fn, TestParam().params_fn);
    }
  }

  void channels_div_channel_tile_unipass_fulltile() {
    TEST_REQUIRES_ARCH_FLAGS(TestParam().arch_flags);
    if (TestParam().channel_tile <= 1 || TestParam().channel_tile == TestParam().channel_scaled_tile) {
        GTEST_SKIP();
    }
    if (TestParam().channel_scaled_tile == TestParam().channel_tile) {
      for (size_t channels = TestParam().channel_tile * 2; channels < TestParam().channel_tile * 8; channels += TestParam().channel_tile) {
        MaxPoolMicrokernelTester<CType>()
          .pooling_elements(TestParam().primary_tile)
          .pooling_tile(TestParam().primary_tile, TestParam().incremental_tile)
          .channels(channels)
          .Test(TestParam().kernel_fn, TestParam().params_fn);
      }
    } else {
      const size_t channel_tile = TestParam().channel_scaled_tile;
      for (size_t channels = channel_tile*2; channels < channel_tile*8; channels += channel_tile) {
        MaxPoolMicrokernelTester<CType>()
          .pooling_elements(TestParam().primary_tile)
          .pooling_tile(TestParam().primary_tile, TestParam().incremental_tile)
          .channels(channels)
          .Test(TestParam().kernel_fn, TestParam().params_fn);
      }
    }
  }

  void channels_div_channel_tile_unipass_fulltile_with_input_offset() {
    TEST_REQUIRES_ARCH_FLAGS(TestParam().arch_flags);
    if (TestParam().channel_tile <= 1 || TestParam().channel_tile == TestParam().channel_scaled_tile) {
        GTEST_SKIP();
    }
    if (TestParam().channel_scaled_tile == TestParam().channel_tile) {
      for (size_t channels = TestParam().channel_tile * 2; channels < TestParam().channel_tile * 8; channels += TestParam().channel_tile) {
        MaxPoolMicrokernelTester<CType>()
          .pooling_elements(TestParam().primary_tile)
          .pooling_tile(TestParam().primary_tile, TestParam().incremental_tile)
          .channels(channels)
          .input_offset(xnnpack::NextPrime(TestParam().channel_tile * 8))
          .Test(TestParam().kernel_fn, TestParam().params_fn);
      }
    } else {
      const size_t channel_tile = TestParam().channel_scaled_tile;
      for (size_t channels = channel_tile*2; channels < channel_tile*8; channels += channel_tile) {
        MaxPoolMicrokernelTester<CType>()
          .pooling_elements(TestParam().primary_tile)
          .pooling_tile(TestParam().primary_tile, TestParam().incremental_tile)
          .channels(channels)
          .input_offset(channel_tile*8)
          .Test(TestParam().kernel_fn, TestParam().params_fn);
      }
    }
  }

  void channels_div_channel_tile_unipass_fulltile_with_qmin() {
    TEST_REQUIRES_ARCH_FLAGS(TestParam().arch_flags);
    if (TestParam().channel_tile <= 1 || TestParam().channel_tile == TestParam().channel_scaled_tile) {
        GTEST_SKIP();
    }
    if (TestParam().channel_scaled_tile == TestParam().channel_tile) {
      for (size_t channels = TestParam().channel_tile * 2; channels < TestParam().channel_tile * 8; channels += TestParam().channel_tile) {
        MaxPoolMicrokernelTester<CType>()
          .pooling_elements(TestParam().primary_tile)
          .pooling_tile(TestParam().primary_tile, TestParam().incremental_tile)
          .channels(channels)
          .qmin(TestParam().qmin)
          .Test(TestParam().kernel_fn, TestParam().params_fn);
      }
    } else {
      const size_t channel_tile = TestParam().channel_scaled_tile;
      for (size_t channels = channel_tile*2; channels < channel_tile*8; channels += channel_tile) {
        MaxPoolMicrokernelTester<CType>()
          .pooling_elements(TestParam().primary_tile)
          .pooling_tile(TestParam().primary_tile, TestParam().incremental_tile)
          .channels(channels)
          .qmin(TestParam().qmin)
          .Test(TestParam().kernel_fn, TestParam().params_fn);
      }
    }
  }

  void channels_div_channel_tile_unipass_fulltile_with_qmax() {
    TEST_REQUIRES_ARCH_FLAGS(TestParam().arch_flags);
    if (TestParam().channel_tile <= 1 || TestParam().channel_tile == TestParam().channel_scaled_tile) {
        GTEST_SKIP();
    }
    if (TestParam().channel_scaled_tile == TestParam().channel_tile) {
      for (size_t channels = TestParam().channel_tile * 2; channels < TestParam().channel_tile * 8; channels += TestParam().channel_tile) {
        MaxPoolMicrokernelTester<CType>()
          .pooling_elements(TestParam().primary_tile)
          .pooling_tile(TestParam().primary_tile, TestParam().incremental_tile)
          .channels(channels)
          .qmax(TestParam().qmax)
          .Test(TestParam().kernel_fn, TestParam().params_fn);
      }
    } else {
      const size_t channel_tile = TestParam().channel_scaled_tile;
      for (size_t channels = channel_tile*2; channels < channel_tile*8; channels += channel_tile) {
        MaxPoolMicrokernelTester<CType>()
          .pooling_elements(TestParam().primary_tile)
          .pooling_tile(TestParam().primary_tile, TestParam().incremental_tile)
          .channels(channels)
          .qmax(TestParam().qmax)
          .Test(TestParam().kernel_fn, TestParam().params_fn);
      }
    }
  }

  void channels_div_channel_tile_unipass_subtile() {
    TEST_REQUIRES_ARCH_FLAGS(TestParam().arch_flags);
    if (TestParam().channel_tile <= 1 || TestParam().channel_tile == TestParam().channel_scaled_tile) {
        GTEST_SKIP();
    }
    for (size_t pooling_elements = 2; pooling_elements < TestParam().primary_tile; pooling_elements++) {
      if (TestParam().channel_scaled_tile == TestParam().channel_tile) {
        for (size_t channels = TestParam().channel_tile * 2; channels < TestParam().channel_tile * 8; channels += TestParam().channel_tile) {
          MaxPoolMicrokernelTester<CType>()
            .pooling_elements(pooling_elements)
            .pooling_tile(TestParam().primary_tile, TestParam().incremental_tile)
            .channels(channels)
            .Test(TestParam().kernel_fn, TestParam().params_fn);
        }
      } else {
        const size_t channel_tile = TestParam().channel_scaled_tile;
        for (size_t channels = channel_tile*2; channels < channel_tile*8; channels += channel_tile) {
          MaxPoolMicrokernelTester<CType>()
            .pooling_elements(pooling_elements)
            .pooling_tile(TestParam().primary_tile, TestParam().incremental_tile)
            .channels(channels)
            .Test(TestParam().kernel_fn, TestParam().params_fn);
        }
      }
    }
  }

  void channels_div_channel_tile_unipass_subtile_with_input_offset() {
    TEST_REQUIRES_ARCH_FLAGS(TestParam().arch_flags);
    if (TestParam().channel_tile <= 1 || TestParam().channel_tile == TestParam().channel_scaled_tile) {
        GTEST_SKIP();
    }
    for (size_t pooling_elements = 2; pooling_elements < TestParam().primary_tile; pooling_elements++) {
      if (TestParam().channel_scaled_tile == TestParam().channel_tile) {
        for (size_t channels = TestParam().channel_tile * 2; channels < TestParam().channel_tile * 8; channels += TestParam().channel_tile) {
          MaxPoolMicrokernelTester<CType>()
            .pooling_elements(pooling_elements)
            .pooling_tile(TestParam().primary_tile, TestParam().incremental_tile)
            .channels(channels)
            .input_offset(xnnpack::NextPrime(TestParam().channel_tile * 8))
            .Test(TestParam().kernel_fn, TestParam().params_fn);
        }
      } else {
        const size_t channel_tile = TestParam().channel_scaled_tile;
        for (size_t channels = channel_tile*2; channels < channel_tile*8; channels += channel_tile) {
          MaxPoolMicrokernelTester<CType>()
            .pooling_elements(pooling_elements)
            .pooling_tile(TestParam().primary_tile, TestParam().incremental_tile)
            .channels(channels)
            .input_offset(channel_tile*8)
            .Test(TestParam().kernel_fn, TestParam().params_fn);
        }
      }
    }
  }

  void channels_lt_channel_tile_unipass_fulltile() {
    TEST_REQUIRES_ARCH_FLAGS(TestParam().arch_flags);
    if (TestParam().channel_tile <= 1 || TestParam().channel_tile == TestParam().channel_scaled_tile) {
        GTEST_SKIP();
    }
    const size_t channel_tile = TestParam().channel_scaled_tile;
    for (size_t channels = 1; channels < channel_tile; channels++) {
      MaxPoolMicrokernelTester<CType>()
        .pooling_elements(TestParam().primary_tile)
        .pooling_tile(TestParam().primary_tile, TestParam().incremental_tile)
        .channels(channels)
        .Test(TestParam().kernel_fn, TestParam().params_fn);
    }
  }

  void channels_lt_channel_tile_unipass_fulltile_with_input_offset() {
    TEST_REQUIRES_ARCH_FLAGS(TestParam().arch_flags);
    if (TestParam().channel_tile <= 1 || TestParam().channel_tile == TestParam().channel_scaled_tile) {
        GTEST_SKIP();
    }
    const size_t channel_tile = TestParam().channel_scaled_tile;
    for (size_t channels = 1; channels < channel_tile; channels++) {
      MaxPoolMicrokernelTester<CType>()
        .pooling_elements(TestParam().primary_tile)
        .pooling_tile(TestParam().primary_tile, TestParam().incremental_tile)
        .channels(channels)
        .input_offset(TestParam().channel_scaled_tile == TestParam().channel_tile ?
                      xnnpack::NextPrime(TestParam().channel_tile) :
                      channel_tile)
        .Test(TestParam().kernel_fn, TestParam().params_fn);
    }
  }

  void channels_lt_channel_tile_unipass_fulltile_with_qmin() {
    TEST_REQUIRES_ARCH_FLAGS(TestParam().arch_flags);
    if (TestParam().channel_tile <= 1 || TestParam().channel_tile == TestParam().channel_scaled_tile) {
        GTEST_SKIP();
    }
    const size_t channel_tile = TestParam().channel_scaled_tile;
    for (size_t channels = 1; channels < channel_tile; channels++) {
      MaxPoolMicrokernelTester<CType>()
        .pooling_elements(TestParam().primary_tile)
        .pooling_tile(TestParam().primary_tile, TestParam().incremental_tile)
        .channels(channels)
        .qmin(TestParam().qmin)
        .Test(TestParam().kernel_fn, TestParam().params_fn);
    }
  }

  void channels_lt_channel_tile_unipass_fulltile_with_qmax() {
    TEST_REQUIRES_ARCH_FLAGS(TestParam().arch_flags);
    if (TestParam().channel_tile <= 1 || TestParam().channel_tile == TestParam().channel_scaled_tile) {
        GTEST_SKIP();
    }
    const size_t channel_tile = TestParam().channel_scaled_tile;
    for (size_t channels = 1; channels < channel_tile; channels++) {
      MaxPoolMicrokernelTester<CType>()
        .pooling_elements(TestParam().primary_tile)
        .pooling_tile(TestParam().primary_tile, TestParam().incremental_tile)
        .channels(channels)
        .qmax(TestParam().qmax)
        .Test(TestParam().kernel_fn, TestParam().params_fn);
    }
  }

  void channels_lt_channel_tile_unipass_subtile() {
    TEST_REQUIRES_ARCH_FLAGS(TestParam().arch_flags);
    if (TestParam().channel_tile <= 1 || TestParam().channel_tile == TestParam().channel_scaled_tile) {
        GTEST_SKIP();
    }
    const size_t channel_tile = TestParam().channel_scaled_tile;
    for (size_t pooling_elements = 2; pooling_elements < TestParam().primary_tile; pooling_elements++) {
      for (size_t channels = 1; channels < channel_tile; channels++) {
        MaxPoolMicrokernelTester<CType>()
          .pooling_elements(pooling_elements)
          .pooling_tile(TestParam().primary_tile, TestParam().incremental_tile)
          .channels(channels)
          .Test(TestParam().kernel_fn, TestParam().params_fn);
      }
    }
  }

  void channels_lt_channel_tile_unipass_subtile_with_input_offset() {
    TEST_REQUIRES_ARCH_FLAGS(TestParam().arch_flags);
    if (TestParam().channel_tile <= 1 || TestParam().channel_tile == TestParam().channel_scaled_tile) {
        GTEST_SKIP();
    }
    const size_t channel_tile = TestParam().channel_scaled_tile;
    for (size_t pooling_elements = 2; pooling_elements < TestParam().primary_tile; pooling_elements++) {
      for (size_t channels = 1; channels < channel_tile; channels++) {
        MaxPoolMicrokernelTester<CType>()
          .pooling_elements(pooling_elements)
          .pooling_tile(TestParam().primary_tile, TestParam().incremental_tile)
          .channels(channels)
          .input_offset(TestParam().channel_scaled_tile == TestParam().channel_tile ?
                        xnnpack::NextPrime(TestParam().channel_tile) :
                        channel_tile)
          .Test(TestParam().kernel_fn, TestParam().params_fn);
      }
    }
  }

  void channels_gt_channel_tile_unipass_fulltile() {
    TEST_REQUIRES_ARCH_FLAGS(TestParam().arch_flags);
    if (TestParam().channel_scaled_tile == TestParam().channel_tile) {
      for (size_t channels = TestParam().channel_tile + 1; channels < (TestParam().channel_tile == 1 ? 10 : TestParam().channel_tile * 2); channels++) {
        MaxPoolMicrokernelTester<CType>()
          .pooling_elements(TestParam().primary_tile)
          .pooling_tile(TestParam().primary_tile, TestParam().incremental_tile)
          .channels(channels)
          .Test(TestParam().kernel_fn, TestParam().params_fn);
      }
    } else {
      const size_t channel_tile = TestParam().channel_scaled_tile;
      for (size_t channels = channel_tile+1; channels < channel_tile*2; channels++) {
        MaxPoolMicrokernelTester<CType>()
          .pooling_elements(TestParam().primary_tile)
          .pooling_tile(TestParam().primary_tile, TestParam().incremental_tile)
          .channels(channels)
          .Test(TestParam().kernel_fn, TestParam().params_fn);
      }
    }
  }

  void channels_gt_channel_tile_unipass_fulltile_with_input_offset() {
    TEST_REQUIRES_ARCH_FLAGS(TestParam().arch_flags);
    if (TestParam().channel_scaled_tile == TestParam().channel_tile) {
      for (size_t channels = TestParam().channel_tile + 1; channels < (TestParam().channel_tile == 1 ? 10 : TestParam().channel_tile * 2); channels++) {
        MaxPoolMicrokernelTester<CType>()
          .pooling_elements(TestParam().primary_tile)
          .pooling_tile(TestParam().primary_tile, TestParam().incremental_tile)
          .channels(channels)
          .input_offset(xnnpack::NextPrime(TestParam().channel_tile * 2))
          .Test(TestParam().kernel_fn, TestParam().params_fn);
      }
    } else {
      const size_t channel_tile = TestParam().channel_scaled_tile;
      for (size_t channels = channel_tile+1; channels < channel_tile*2; channels++) {
        MaxPoolMicrokernelTester<CType>()
          .pooling_elements(TestParam().primary_tile)
          .pooling_tile(TestParam().primary_tile, TestParam().incremental_tile)
          .channels(channels)
          .input_offset(channel_tile*2)
          .Test(TestParam().kernel_fn, TestParam().params_fn);
      }
    }
  }

  void channels_gt_channel_tile_unipass_fulltile_with_qmin() {
    TEST_REQUIRES_ARCH_FLAGS(TestParam().arch_flags);
    if (TestParam().channel_scaled_tile == TestParam().channel_tile) {
      for (size_t channels = TestParam().channel_tile + 1; channels < (TestParam().channel_tile == 1 ? 10 : TestParam().channel_tile * 2); channels++) {
        MaxPoolMicrokernelTester<CType>()
          .pooling_elements(TestParam().primary_tile)
          .pooling_tile(TestParam().primary_tile, TestParam().incremental_tile)
          .channels(channels)
          .qmin(TestParam().qmin)
          .Test(TestParam().kernel_fn, TestParam().params_fn);
      }
    } else {
      const size_t channel_tile = TestParam().channel_scaled_tile;
      for (size_t channels = channel_tile+1; channels < channel_tile*2; channels++) {
        MaxPoolMicrokernelTester<CType>()
          .pooling_elements(TestParam().primary_tile)
          .pooling_tile(TestParam().primary_tile, TestParam().incremental_tile)
          .channels(channels)
          .qmin(TestParam().qmin)
          .Test(TestParam().kernel_fn, TestParam().params_fn);
      }
    }
  }

  void channels_gt_channel_tile_unipass_fulltile_with_qmax() {
    TEST_REQUIRES_ARCH_FLAGS(TestParam().arch_flags);
    if (TestParam().channel_scaled_tile == TestParam().channel_tile) {
      for (size_t channels = TestParam().channel_tile + 1; channels < (TestParam().channel_tile == 1 ? 10 : TestParam().channel_tile * 2); channels++) {
        MaxPoolMicrokernelTester<CType>()
          .pooling_elements(TestParam().primary_tile)
          .pooling_tile(TestParam().primary_tile, TestParam().incremental_tile)
          .channels(channels)
          .qmax(TestParam().qmax)
          .Test(TestParam().kernel_fn, TestParam().params_fn);
      }
    } else {
      const size_t channel_tile = TestParam().channel_scaled_tile;
      for (size_t channels = channel_tile+1; channels < channel_tile*2; channels++) {
        MaxPoolMicrokernelTester<CType>()
          .pooling_elements(TestParam().primary_tile)
          .pooling_tile(TestParam().primary_tile, TestParam().incremental_tile)
          .channels(channels)
          .qmax(TestParam().qmax)
          .Test(TestParam().kernel_fn, TestParam().params_fn);
      }
    }
  }

  void channels_gt_channel_tile_unipass_subtile() {
    TEST_REQUIRES_ARCH_FLAGS(TestParam().arch_flags);
    for (size_t pooling_elements = 2; pooling_elements < TestParam().primary_tile; pooling_elements++) {
      if (TestParam().channel_scaled_tile == TestParam().channel_tile) {
        for (size_t channels = TestParam().channel_tile + 1; channels < (TestParam().channel_tile == 1 ? 10 : TestParam().channel_tile * 2); channels++) {
          MaxPoolMicrokernelTester<CType>()
            .pooling_elements(pooling_elements)
            .pooling_tile(TestParam().primary_tile, TestParam().incremental_tile)
            .channels(channels)
            .Test(TestParam().kernel_fn, TestParam().params_fn);
        }
      } else {
        const size_t channel_tile = TestParam().channel_scaled_tile;
        for (size_t channels = channel_tile+1; channels < channel_tile*2; channels++) {
          MaxPoolMicrokernelTester<CType>()
            .pooling_elements(pooling_elements)
            .pooling_tile(TestParam().primary_tile, TestParam().incremental_tile)
            .channels(channels)
            .Test(TestParam().kernel_fn, TestParam().params_fn);
        }
      }
    }
  }

  void channels_gt_channel_tile_unipass_subtile_with_input_offset() {
    TEST_REQUIRES_ARCH_FLAGS(TestParam().arch_flags);
    for (size_t pooling_elements = 2; pooling_elements < TestParam().primary_tile; pooling_elements++) {
      if (TestParam().channel_scaled_tile == TestParam().channel_tile) {
        for (size_t channels = TestParam().channel_tile + 1; channels < (TestParam().channel_tile == 1 ? 10 : TestParam().channel_tile * 2); channels++) {
          MaxPoolMicrokernelTester<CType>()
            .pooling_elements(pooling_elements)
            .pooling_tile(TestParam().primary_tile, TestParam().incremental_tile)
            .channels(channels)
            .input_offset(xnnpack::NextPrime(TestParam().channel_tile * 2))
            .Test(TestParam().kernel_fn, TestParam().params_fn);
        }
      } else {
        const size_t channel_tile = TestParam().channel_scaled_tile;
        for (size_t channels = channel_tile+1; channels < channel_tile*2; channels++) {
          MaxPoolMicrokernelTester<CType>()
            .pooling_elements(pooling_elements)
            .pooling_tile(TestParam().primary_tile, TestParam().incremental_tile)
            .channels(channels)
            .input_offset(channel_tile*2)
            .Test(TestParam().kernel_fn, TestParam().params_fn);
        }
      }
    }
  }

  void channels_eq_channel_tile_twopass_fulltile() {
    TEST_REQUIRES_ARCH_FLAGS(TestParam().arch_flags);
    const size_t channel_tile = TestParam().channel_scaled_tile;
    MaxPoolMicrokernelTester<CType>()
      .pooling_elements(TestParam().primary_tile + TestParam().incremental_tile)
      .pooling_tile(TestParam().primary_tile, TestParam().incremental_tile)
      .channels(channel_tile)
      .Test(TestParam().kernel_fn, TestParam().params_fn);
  }

  void channels_eq_channel_tile_twopass_fulltile_with_input_offset() {
    TEST_REQUIRES_ARCH_FLAGS(TestParam().arch_flags);
    const size_t channel_tile = TestParam().channel_scaled_tile;
    MaxPoolMicrokernelTester<CType>()
      .pooling_elements(TestParam().primary_tile + TestParam().incremental_tile)
      .pooling_tile(TestParam().primary_tile, TestParam().incremental_tile)
      .channels(channel_tile)
      .input_offset(TestParam().channel_scaled_tile == TestParam().channel_tile ?
                    xnnpack::NextPrime(TestParam().channel_tile + 1) :
                    channel_tile + 1)
      .Test(TestParam().kernel_fn, TestParam().params_fn);
  }

  void channels_eq_channel_tile_twopass_fulltile_with_qmin() {
    TEST_REQUIRES_ARCH_FLAGS(TestParam().arch_flags);
    const size_t channel_tile = TestParam().channel_scaled_tile;
    MaxPoolMicrokernelTester<CType>()
      .pooling_elements(TestParam().primary_tile + TestParam().incremental_tile)
      .pooling_tile(TestParam().primary_tile, TestParam().incremental_tile)
      .channels(channel_tile)
      .qmin(TestParam().qmin)
      .Test(TestParam().kernel_fn, TestParam().params_fn);
  }

  void channels_eq_channel_tile_twopass_fulltile_with_qmax() {
    TEST_REQUIRES_ARCH_FLAGS(TestParam().arch_flags);
    const size_t channel_tile = TestParam().channel_scaled_tile;
    MaxPoolMicrokernelTester<CType>()
      .pooling_elements(TestParam().primary_tile + TestParam().incremental_tile)
      .pooling_tile(TestParam().primary_tile, TestParam().incremental_tile)
      .channels(channel_tile)
      .qmax(TestParam().qmax)
      .Test(TestParam().kernel_fn, TestParam().params_fn);
  }

  void channels_eq_channel_tile_twopass_subtile() {
    TEST_REQUIRES_ARCH_FLAGS(TestParam().arch_flags);
    const size_t channel_tile = TestParam().channel_scaled_tile;
    for (size_t pooling_elements = TestParam().primary_tile + 1; pooling_elements < TestParam().primary_tile + TestParam().incremental_tile; pooling_elements++) {
      MaxPoolMicrokernelTester<CType>()
        .pooling_elements(pooling_elements)
        .pooling_tile(TestParam().primary_tile, TestParam().incremental_tile)
        .channels(channel_tile)
        .Test(TestParam().kernel_fn, TestParam().params_fn);
    }
  }

  void channels_eq_channel_tile_twopass_subtile_with_input_offset() {
    TEST_REQUIRES_ARCH_FLAGS(TestParam().arch_flags);
    const size_t channel_tile = TestParam().channel_scaled_tile;
    for (size_t pooling_elements = TestParam().primary_tile + 1; pooling_elements < TestParam().primary_tile + TestParam().incremental_tile; pooling_elements++) {
      MaxPoolMicrokernelTester<CType>()
        .pooling_elements(pooling_elements)
        .pooling_tile(TestParam().primary_tile, TestParam().incremental_tile)
        .channels(channel_tile)
        .input_offset(TestParam().channel_scaled_tile == TestParam().channel_tile ?
                      xnnpack::NextPrime(TestParam().channel_tile + 1) :
                      channel_tile + 1)
        .Test(TestParam().kernel_fn, TestParam().params_fn);
    }
  }

  void channels_div_channel_tile_twopass_fulltile() {
    TEST_REQUIRES_ARCH_FLAGS(TestParam().arch_flags);
    if (TestParam().channel_tile <= 1 || TestParam().channel_tile == TestParam().channel_scaled_tile) {
        GTEST_SKIP();
    }
    if (TestParam().channel_scaled_tile == TestParam().channel_tile) {
      for (size_t channels = TestParam().channel_tile * 2; channels < TestParam().channel_tile * 8; channels += TestParam().channel_tile) {
        MaxPoolMicrokernelTester<CType>()
          .pooling_elements(TestParam().primary_tile + TestParam().incremental_tile)
          .pooling_tile(TestParam().primary_tile, TestParam().incremental_tile)
          .channels(channels)
          .Test(TestParam().kernel_fn, TestParam().params_fn);
      }
    } else {
      const size_t channel_tile = TestParam().channel_scaled_tile;
      for (size_t channels = channel_tile*2; channels < channel_tile*8; channels += channel_tile) {
        MaxPoolMicrokernelTester<CType>()
          .pooling_elements(TestParam().primary_tile + TestParam().incremental_tile)
          .pooling_tile(TestParam().primary_tile, TestParam().incremental_tile)
          .channels(channels)
          .Test(TestParam().kernel_fn, TestParam().params_fn);
      }
    }
  }

  void channels_div_channel_tile_twopass_fulltile_with_input_offset() {
    TEST_REQUIRES_ARCH_FLAGS(TestParam().arch_flags);
    if (TestParam().channel_tile <= 1 || TestParam().channel_tile == TestParam().channel_scaled_tile) {
        GTEST_SKIP();
    }
    if (TestParam().channel_scaled_tile == TestParam().channel_tile) {
      for (size_t channels = TestParam().channel_tile * 2; channels < TestParam().channel_tile * 8; channels += TestParam().channel_tile) {
        MaxPoolMicrokernelTester<CType>()
          .pooling_elements(TestParam().primary_tile + TestParam().incremental_tile)
          .pooling_tile(TestParam().primary_tile, TestParam().incremental_tile)
          .channels(channels)
          .input_offset(xnnpack::NextPrime(TestParam().channel_tile * 5))
          .Test(TestParam().kernel_fn, TestParam().params_fn);
      }
    } else {
      const size_t channel_tile = TestParam().channel_scaled_tile;
      for (size_t channels = channel_tile*2; channels < channel_tile*8; channels += channel_tile) {
        MaxPoolMicrokernelTester<CType>()
          .pooling_elements(TestParam().primary_tile + TestParam().incremental_tile)
          .pooling_tile(TestParam().primary_tile, TestParam().incremental_tile)
          .channels(channels)
          .input_offset(channel_tile*5)
          .Test(TestParam().kernel_fn, TestParam().params_fn);
      }
    }
  }

  void channels_div_channel_tile_twopass_fulltile_with_qmin() {
    TEST_REQUIRES_ARCH_FLAGS(TestParam().arch_flags);
    if (TestParam().channel_tile <= 1 || TestParam().channel_tile == TestParam().channel_scaled_tile) {
        GTEST_SKIP();
    }
    if (TestParam().channel_scaled_tile == TestParam().channel_tile) {
      for (size_t channels = TestParam().channel_tile * 2; channels < TestParam().channel_tile * 8; channels += TestParam().channel_tile) {
        MaxPoolMicrokernelTester<CType>()
          .pooling_elements(TestParam().primary_tile + TestParam().incremental_tile)
          .pooling_tile(TestParam().primary_tile, TestParam().incremental_tile)
          .channels(channels)
          .qmin(TestParam().qmin)
          .Test(TestParam().kernel_fn, TestParam().params_fn);
      }
    } else {
      const size_t channel_tile = TestParam().channel_scaled_tile;
      for (size_t channels = channel_tile*2; channels < channel_tile*8; channels += channel_tile) {
        MaxPoolMicrokernelTester<CType>()
          .pooling_elements(TestParam().primary_tile + TestParam().incremental_tile)
          .pooling_tile(TestParam().primary_tile, TestParam().incremental_tile)
          .channels(channels)
          .qmin(TestParam().qmin)
          .Test(TestParam().kernel_fn, TestParam().params_fn);
      }
    }
  }

  void channels_div_channel_tile_twopass_fulltile_with_qmax() {
    TEST_REQUIRES_ARCH_FLAGS(TestParam().arch_flags);
    if (TestParam().channel_tile <= 1 || TestParam().channel_tile == TestParam().channel_scaled_tile) {
        GTEST_SKIP();
    }
    if (TestParam().channel_scaled_tile == TestParam().channel_tile) {
      for (size_t channels = TestParam().channel_tile * 2; channels < TestParam().channel_tile * 8; channels += TestParam().channel_tile) {
        MaxPoolMicrokernelTester<CType>()
          .pooling_elements(TestParam().primary_tile + TestParam().incremental_tile)
          .pooling_tile(TestParam().primary_tile, TestParam().incremental_tile)
          .channels(channels)
          .qmax(TestParam().qmax)
          .Test(TestParam().kernel_fn, TestParam().params_fn);
      }
    } else {
      const size_t channel_tile = TestParam().channel_scaled_tile;
      for (size_t channels = channel_tile*2; channels < channel_tile*8; channels += channel_tile) {
        MaxPoolMicrokernelTester<CType>()
          .pooling_elements(TestParam().primary_tile + TestParam().incremental_tile)
          .pooling_tile(TestParam().primary_tile, TestParam().incremental_tile)
          .channels(channels)
          .qmax(TestParam().qmax)
          .Test(TestParam().kernel_fn, TestParam().params_fn);
      }
    }
  }

  void channels_div_channel_tile_twopass_subtile() {
    TEST_REQUIRES_ARCH_FLAGS(TestParam().arch_flags);
    if (TestParam().channel_tile <= 1 || TestParam().channel_tile == TestParam().channel_scaled_tile) {
        GTEST_SKIP();
    }
    for (size_t pooling_elements = TestParam().primary_tile + 1; pooling_elements < TestParam().primary_tile + TestParam().incremental_tile; pooling_elements++) {
      if (TestParam().channel_scaled_tile == TestParam().channel_tile) {
        for (size_t channels = TestParam().channel_tile * 2; channels < TestParam().channel_tile * 8; channels += TestParam().channel_tile) {
          MaxPoolMicrokernelTester<CType>()
            .pooling_elements(pooling_elements)
            .pooling_tile(TestParam().primary_tile, TestParam().incremental_tile)
            .channels(channels)
            .Test(TestParam().kernel_fn, TestParam().params_fn);
        }
      } else {
        const size_t channel_tile = TestParam().channel_scaled_tile;
        for (size_t channels = channel_tile*2; channels < channel_tile*8; channels += channel_tile) {
          MaxPoolMicrokernelTester<CType>()
            .pooling_elements(pooling_elements)
            .pooling_tile(TestParam().primary_tile, TestParam().incremental_tile)
            .channels(channels)
            .Test(TestParam().kernel_fn, TestParam().params_fn);
        }
      }
    }
  }

  void channels_div_channel_tile_twopass_subtile_with_input_offset() {
    TEST_REQUIRES_ARCH_FLAGS(TestParam().arch_flags);
    if (TestParam().channel_tile <= 1 || TestParam().channel_tile == TestParam().channel_scaled_tile) {
        GTEST_SKIP();
    }
    for (size_t pooling_elements = TestParam().primary_tile + 1; pooling_elements < TestParam().primary_tile + TestParam().incremental_tile; pooling_elements++) {
      if (TestParam().channel_scaled_tile == TestParam().channel_tile) {
        for (size_t channels = TestParam().channel_tile * 2; channels < TestParam().channel_tile * 8; channels += TestParam().channel_tile) {
          MaxPoolMicrokernelTester<CType>()
            .pooling_elements(pooling_elements)
            .pooling_tile(TestParam().primary_tile, TestParam().incremental_tile)
            .channels(channels)
            .input_offset(xnnpack::NextPrime(TestParam().channel_tile * 8))
            .Test(TestParam().kernel_fn, TestParam().params_fn);
        }
      } else {
        const size_t channel_tile = TestParam().channel_scaled_tile;
        for (size_t channels = channel_tile*2; channels < channel_tile*8; channels += channel_tile) {
          MaxPoolMicrokernelTester<CType>()
            .pooling_elements(pooling_elements)
            .pooling_tile(TestParam().primary_tile, TestParam().incremental_tile)
            .channels(channels)
            .input_offset(channel_tile*8)
            .Test(TestParam().kernel_fn, TestParam().params_fn);
        }
      }
    }
  }

  void channels_lt_channel_tile_twopass_fulltile() {
    TEST_REQUIRES_ARCH_FLAGS(TestParam().arch_flags);
    if (TestParam().channel_tile <= 1 || TestParam().channel_tile == TestParam().channel_scaled_tile) {
        GTEST_SKIP();
    }
    const size_t channel_tile = TestParam().channel_scaled_tile;
    for (size_t channels = 1; channels < channel_tile; channels++) {
      MaxPoolMicrokernelTester<CType>()
        .pooling_elements(TestParam().primary_tile + TestParam().incremental_tile)
        .pooling_tile(TestParam().primary_tile, TestParam().incremental_tile)
        .channels(channels)
        .Test(TestParam().kernel_fn, TestParam().params_fn);
    }
  }

  void channels_lt_channel_tile_twopass_fulltile_with_input_offset() {
    TEST_REQUIRES_ARCH_FLAGS(TestParam().arch_flags);
    if (TestParam().channel_tile <= 1 || TestParam().channel_tile == TestParam().channel_scaled_tile) {
        GTEST_SKIP();
    }
    const size_t channel_tile = TestParam().channel_scaled_tile;
    for (size_t channels = 1; channels < channel_tile; channels++) {
      MaxPoolMicrokernelTester<CType>()
        .pooling_elements(TestParam().primary_tile + TestParam().incremental_tile)
        .pooling_tile(TestParam().primary_tile, TestParam().incremental_tile)
        .channels(channels)
        .input_offset(TestParam().channel_scaled_tile == TestParam().channel_tile ?
                      xnnpack::NextPrime(TestParam().channel_tile) :
                      channel_tile)
        .Test(TestParam().kernel_fn, TestParam().params_fn);
    }
  }

  void channels_lt_channel_tile_twopass_fulltile_with_qmin() {
    TEST_REQUIRES_ARCH_FLAGS(TestParam().arch_flags);
    if (TestParam().channel_tile <= 1 || TestParam().channel_tile == TestParam().channel_scaled_tile) {
        GTEST_SKIP();
    }
    const size_t channel_tile = TestParam().channel_scaled_tile;
    for (size_t channels = 1; channels < channel_tile; channels++) {
      MaxPoolMicrokernelTester<CType>()
        .pooling_elements(TestParam().primary_tile + TestParam().incremental_tile)
        .pooling_tile(TestParam().primary_tile, TestParam().incremental_tile)
        .channels(channels)
        .qmin(TestParam().qmin)
        .Test(TestParam().kernel_fn, TestParam().params_fn);
    }
  }

  void channels_lt_channel_tile_twopass_fulltile_with_qmax() {
    TEST_REQUIRES_ARCH_FLAGS(TestParam().arch_flags);
    if (TestParam().channel_tile <= 1 || TestParam().channel_tile == TestParam().channel_scaled_tile) {
        GTEST_SKIP();
    }
    const size_t channel_tile = TestParam().channel_scaled_tile;
    for (size_t channels = 1; channels < channel_tile; channels++) {
      MaxPoolMicrokernelTester<CType>()
        .pooling_elements(TestParam().primary_tile + TestParam().incremental_tile)
        .pooling_tile(TestParam().primary_tile, TestParam().incremental_tile)
        .channels(channels)
        .qmax(TestParam().qmax)
        .Test(TestParam().kernel_fn, TestParam().params_fn);
    }
  }

  void channels_lt_channel_tile_twopass_subtile() {
    TEST_REQUIRES_ARCH_FLAGS(TestParam().arch_flags);
    if (TestParam().channel_tile <= 1 || TestParam().channel_tile == TestParam().channel_scaled_tile) {
        GTEST_SKIP();
    }
    const size_t channel_tile = TestParam().channel_scaled_tile;
    for (size_t pooling_elements = TestParam().primary_tile + 1; pooling_elements < TestParam().primary_tile + TestParam().incremental_tile; pooling_elements++) {
      for (size_t channels = 1; channels < channel_tile; channels++) {
        MaxPoolMicrokernelTester<CType>()
          .pooling_elements(pooling_elements)
          .pooling_tile(TestParam().primary_tile, TestParam().incremental_tile)
          .channels(channels)
          .Test(TestParam().kernel_fn, TestParam().params_fn);
      }
    }
  }

  void channels_lt_channel_tile_twopass_subtile_with_input_offset() {
    TEST_REQUIRES_ARCH_FLAGS(TestParam().arch_flags);
    if (TestParam().channel_tile <= 1 || TestParam().channel_tile == TestParam().channel_scaled_tile) {
        GTEST_SKIP();
    }
    const size_t channel_tile = TestParam().channel_scaled_tile;
    for (size_t pooling_elements = TestParam().primary_tile + 1; pooling_elements < TestParam().primary_tile + TestParam().incremental_tile; pooling_elements++) {
      for (size_t channels = 1; channels < channel_tile; channels++) {
        MaxPoolMicrokernelTester<CType>()
          .pooling_elements(pooling_elements)
          .pooling_tile(TestParam().primary_tile, TestParam().incremental_tile)
          .channels(channels)
          .input_offset(TestParam().channel_scaled_tile == TestParam().channel_tile ?
                        xnnpack::NextPrime(TestParam().channel_tile) :
                        channel_tile)
          .Test(TestParam().kernel_fn, TestParam().params_fn);
      }
    }
  }

  void channels_gt_channel_tile_twopass_fulltile() {
    TEST_REQUIRES_ARCH_FLAGS(TestParam().arch_flags);
    if (TestParam().channel_scaled_tile == TestParam().channel_tile) {
      for (size_t channels = TestParam().channel_tile + 1; channels < (TestParam().channel_tile == 1 ? 10 : TestParam().channel_tile * 2); channels++) {
        MaxPoolMicrokernelTester<CType>()
          .pooling_elements(TestParam().primary_tile + TestParam().incremental_tile)
          .pooling_tile(TestParam().primary_tile, TestParam().incremental_tile)
          .channels(channels)
          .Test(TestParam().kernel_fn, TestParam().params_fn);
      }
    } else {
      const size_t channel_tile = TestParam().channel_scaled_tile;
      for (size_t channels = channel_tile+1; channels < channel_tile*2; channels++) {
        MaxPoolMicrokernelTester<CType>()
          .pooling_elements(TestParam().primary_tile + TestParam().incremental_tile)
          .pooling_tile(TestParam().primary_tile, TestParam().incremental_tile)
          .channels(channels)
          .Test(TestParam().kernel_fn, TestParam().params_fn);
      }
    }
  }

  void channels_gt_channel_tile_twopass_fulltile_with_input_offset() {
    TEST_REQUIRES_ARCH_FLAGS(TestParam().arch_flags);
    if (TestParam().channel_scaled_tile == TestParam().channel_tile) {
      for (size_t channels = TestParam().channel_tile + 1; channels < (TestParam().channel_tile == 1 ? 10 : TestParam().channel_tile * 2); channels++) {
        MaxPoolMicrokernelTester<CType>()
          .pooling_elements(TestParam().primary_tile + TestParam().incremental_tile)
          .pooling_tile(TestParam().primary_tile, TestParam().incremental_tile)
          .channels(channels)
          .input_offset(xnnpack::NextPrime(TestParam().channel_tile * 2))
          .Test(TestParam().kernel_fn, TestParam().params_fn);
      }
    } else {
      const size_t channel_tile = TestParam().channel_scaled_tile;
      for (size_t channels = channel_tile+1; channels < channel_tile*2; channels++) {
        MaxPoolMicrokernelTester<CType>()
          .pooling_elements(TestParam().primary_tile + TestParam().incremental_tile)
          .pooling_tile(TestParam().primary_tile, TestParam().incremental_tile)
          .channels(channels)
          .input_offset(channel_tile*2)
          .Test(TestParam().kernel_fn, TestParam().params_fn);
      }
    }
  }

  void channels_gt_channel_tile_twopass_fulltile_with_qmin() {
    TEST_REQUIRES_ARCH_FLAGS(TestParam().arch_flags);
    if (TestParam().channel_scaled_tile == TestParam().channel_tile) {
      for (size_t channels = TestParam().channel_tile + 1; channels < (TestParam().channel_tile == 1 ? 10 : TestParam().channel_tile * 2); channels++) {
        MaxPoolMicrokernelTester<CType>()
          .pooling_elements(TestParam().primary_tile + TestParam().incremental_tile)
          .pooling_tile(TestParam().primary_tile, TestParam().incremental_tile)
          .channels(channels)
          .qmin(TestParam().qmin)
          .Test(TestParam().kernel_fn, TestParam().params_fn);
      }
    } else {
      const size_t channel_tile = TestParam().channel_scaled_tile;
      for (size_t channels = channel_tile+1; channels < channel_tile*2; channels++) {
        MaxPoolMicrokernelTester<CType>()
          .pooling_elements(TestParam().primary_tile + TestParam().incremental_tile)
          .pooling_tile(TestParam().primary_tile, TestParam().incremental_tile)
          .channels(channels)
          .qmin(TestParam().qmin)
          .Test(TestParam().kernel_fn, TestParam().params_fn);
      }
    }
  }

  void channels_gt_channel_tile_twopass_fulltile_with_qmax() {
    TEST_REQUIRES_ARCH_FLAGS(TestParam().arch_flags);
    if (TestParam().channel_scaled_tile == TestParam().channel_tile) {
      for (size_t channels = TestParam().channel_tile + 1; channels < (TestParam().channel_tile == 1 ? 10 : TestParam().channel_tile * 2); channels++) {
        MaxPoolMicrokernelTester<CType>()
          .pooling_elements(TestParam().primary_tile + TestParam().incremental_tile)
          .pooling_tile(TestParam().primary_tile, TestParam().incremental_tile)
          .channels(channels)
          .qmax(TestParam().qmax)
          .Test(TestParam().kernel_fn, TestParam().params_fn);
      }
    } else {
      const size_t channel_tile = TestParam().channel_scaled_tile;
      for (size_t channels = channel_tile+1; channels < channel_tile*2; channels++) {
        MaxPoolMicrokernelTester<CType>()
          .pooling_elements(TestParam().primary_tile + TestParam().incremental_tile)
          .pooling_tile(TestParam().primary_tile, TestParam().incremental_tile)
          .channels(channels)
          .qmax(TestParam().qmax)
          .Test(TestParam().kernel_fn, TestParam().params_fn);
      }
    }
  }

  void channels_gt_channel_tile_twopass_subtile() {
    TEST_REQUIRES_ARCH_FLAGS(TestParam().arch_flags);
    for (size_t pooling_elements = TestParam().primary_tile + 1; pooling_elements < TestParam().primary_tile + TestParam().incremental_tile; pooling_elements++) {
      if (TestParam().channel_scaled_tile == TestParam().channel_tile) {
        for (size_t channels = TestParam().channel_tile + 1; channels < (TestParam().channel_tile == 1 ? 10 : TestParam().channel_tile * 2); channels++) {
          MaxPoolMicrokernelTester<CType>()
            .pooling_elements(pooling_elements)
            .pooling_tile(TestParam().primary_tile, TestParam().incremental_tile)
            .channels(channels)
            .Test(TestParam().kernel_fn, TestParam().params_fn);
        }
      } else {
        const size_t channel_tile = TestParam().channel_scaled_tile;
        for (size_t channels = channel_tile+1; channels < channel_tile*2; channels++) {
          MaxPoolMicrokernelTester<CType>()
            .pooling_elements(pooling_elements)
            .pooling_tile(TestParam().primary_tile, TestParam().incremental_tile)
            .channels(channels)
            .Test(TestParam().kernel_fn, TestParam().params_fn);
        }
      }
    }
  }

  void channels_gt_channel_tile_twopass_subtile_with_input_offset() {
    TEST_REQUIRES_ARCH_FLAGS(TestParam().arch_flags);
    for (size_t pooling_elements = TestParam().primary_tile + 1; pooling_elements < TestParam().primary_tile + TestParam().incremental_tile; pooling_elements++) {
      if (TestParam().channel_scaled_tile == TestParam().channel_tile) {
        for (size_t channels = TestParam().channel_tile + 1; channels < (TestParam().channel_tile == 1 ? 10 : TestParam().channel_tile * 2); channels++) {
          MaxPoolMicrokernelTester<CType>()
            .pooling_elements(pooling_elements)
            .pooling_tile(TestParam().primary_tile, TestParam().incremental_tile)
            .channels(channels)
            .input_offset(xnnpack::NextPrime(TestParam().channel_tile * 2))
            .Test(TestParam().kernel_fn, TestParam().params_fn);
        }
      } else {
        const size_t channel_tile = TestParam().channel_scaled_tile;
        for (size_t channels = channel_tile+1; channels < channel_tile*2; channels++) {
          MaxPoolMicrokernelTester<CType>()
            .pooling_elements(pooling_elements)
            .pooling_tile(TestParam().primary_tile, TestParam().incremental_tile)
            .channels(channels)
            .input_offset(channel_tile*2)
            .Test(TestParam().kernel_fn, TestParam().params_fn);
        }
      }
    }
  }

  void channels_eq_channel_tile_multipass() {
    TEST_REQUIRES_ARCH_FLAGS(TestParam().arch_flags);
    const size_t channel_tile = TestParam().channel_scaled_tile;
    for (size_t pooling_elements = TestParam().primary_tile + TestParam().incremental_tile;
        pooling_elements <= TestParam().primary_tile + TestParam().incremental_tile * 3;
        pooling_elements = xnnpack::NextPrime(pooling_elements)) {
      MaxPoolMicrokernelTester<CType>()
        .pooling_elements(pooling_elements)
        .pooling_tile(TestParam().primary_tile, TestParam().incremental_tile)
        .channels(channel_tile)
        .Test(TestParam().kernel_fn, TestParam().params_fn);
    }
  }

  void channels_eq_channel_tile_multipass_with_input_offset() {
    TEST_REQUIRES_ARCH_FLAGS(TestParam().arch_flags);
    const size_t channel_tile = TestParam().channel_scaled_tile;
    for (size_t pooling_elements = TestParam().primary_tile + TestParam().incremental_tile;
        pooling_elements <= TestParam().primary_tile + TestParam().incremental_tile * 3;
        pooling_elements = xnnpack::NextPrime(pooling_elements)) {
      MaxPoolMicrokernelTester<CType>()
        .pooling_elements(pooling_elements)
        .pooling_tile(TestParam().primary_tile, TestParam().incremental_tile)
        .channels(channel_tile)
        .input_offset(TestParam().channel_scaled_tile == TestParam().channel_tile ?
                      xnnpack::NextPrime(TestParam().channel_tile + 1) :
                      channel_tile+1)
        .Test(TestParam().kernel_fn, TestParam().params_fn);
    }
  }

  void channels_eq_channel_tile_multipass_with_qmin() {
    TEST_REQUIRES_ARCH_FLAGS(TestParam().arch_flags);
    const size_t channel_tile = TestParam().channel_scaled_tile;
    for (size_t pooling_elements = TestParam().primary_tile + TestParam().incremental_tile;
        pooling_elements <= TestParam().primary_tile + TestParam().incremental_tile * 3;
        pooling_elements = xnnpack::NextPrime(pooling_elements)) {
      MaxPoolMicrokernelTester<CType>()
        .pooling_elements(pooling_elements)
        .pooling_tile(TestParam().primary_tile, TestParam().incremental_tile)
        .channels(channel_tile)
        .qmin(TestParam().qmin)
        .Test(TestParam().kernel_fn, TestParam().params_fn);
    }
  }

  void channels_eq_channel_tile_multipass_with_qmax() {
    TEST_REQUIRES_ARCH_FLAGS(TestParam().arch_flags);
    const size_t channel_tile = TestParam().channel_scaled_tile;
    for (size_t pooling_elements = TestParam().primary_tile + TestParam().incremental_tile;
        pooling_elements <= TestParam().primary_tile + TestParam().incremental_tile * 3;
        pooling_elements = xnnpack::NextPrime(pooling_elements)) {
      MaxPoolMicrokernelTester<CType>()
        .pooling_elements(pooling_elements)
        .pooling_tile(TestParam().primary_tile, TestParam().incremental_tile)
        .channels(channel_tile)
        .qmax(TestParam().qmax)
        .Test(TestParam().kernel_fn, TestParam().params_fn);
    }
  }

  void channels_div_channel_tile_multipass() {
    TEST_REQUIRES_ARCH_FLAGS(TestParam().arch_flags);
    if (TestParam().channel_tile <= 1 || TestParam().channel_tile == TestParam().channel_scaled_tile) {
        GTEST_SKIP();
    }
    for (size_t pooling_elements = TestParam().primary_tile + TestParam().incremental_tile;
        pooling_elements <= TestParam().primary_tile + TestParam().incremental_tile * 3;
        pooling_elements = xnnpack::NextPrime(pooling_elements)) {
      if (TestParam().channel_scaled_tile == TestParam().channel_tile) {
        for (size_t channels = TestParam().channel_tile * 2; channels < TestParam().channel_tile * 8; channels += TestParam().channel_tile) {
          MaxPoolMicrokernelTester<CType>()
            .pooling_elements(pooling_elements)
            .pooling_tile(TestParam().primary_tile, TestParam().incremental_tile)
            .channels(channels)
            .Test(TestParam().kernel_fn, TestParam().params_fn);
        }
      } else {
        const size_t channel_tile = TestParam().channel_scaled_tile;
        for (size_t channels = channel_tile*2; channels < channel_tile*8; channels += channel_tile) {
          MaxPoolMicrokernelTester<CType>()
            .pooling_elements(pooling_elements)
            .pooling_tile(TestParam().primary_tile, TestParam().incremental_tile)
            .channels(channels)
            .Test(TestParam().kernel_fn, TestParam().params_fn);
        }
      }
    }
  }

  void channels_div_channel_tile_multipass_with_input_offset() {
    TEST_REQUIRES_ARCH_FLAGS(TestParam().arch_flags);
    if (TestParam().channel_tile <= 1 || TestParam().channel_tile == TestParam().channel_scaled_tile) {
        GTEST_SKIP();
    }
    for (size_t pooling_elements = TestParam().primary_tile + TestParam().incremental_tile;
        pooling_elements <= TestParam().primary_tile + TestParam().incremental_tile * 3;
        pooling_elements = xnnpack::NextPrime(pooling_elements)) {
      if (TestParam().channel_scaled_tile == TestParam().channel_tile) {
        for (size_t channels = TestParam().channel_tile * 2; channels < TestParam().channel_tile * 8; channels += TestParam().channel_tile) {
          MaxPoolMicrokernelTester<CType>()
            .pooling_elements(pooling_elements)
            .pooling_tile(TestParam().primary_tile, TestParam().incremental_tile)
            .channels(channels)
            .input_offset(xnnpack::NextPrime(TestParam().channel_tile * 8))
            .Test(TestParam().kernel_fn, TestParam().params_fn);
        }
      } else {
        const size_t channel_tile = TestParam().channel_scaled_tile;
        for (size_t channels = channel_tile*2; channels < channel_tile*8; channels += channel_tile) {
          MaxPoolMicrokernelTester<CType>()
            .pooling_elements(pooling_elements)
            .pooling_tile(TestParam().primary_tile, TestParam().incremental_tile)
            .channels(channels)
            .input_offset(channel_tile*8)
            .Test(TestParam().kernel_fn, TestParam().params_fn);
        }
      }
    }
  }

  void channels_div_channel_tile_multipass_with_qmin() {
    TEST_REQUIRES_ARCH_FLAGS(TestParam().arch_flags);
    if (TestParam().channel_tile <= 1 || TestParam().channel_tile == TestParam().channel_scaled_tile) {
        GTEST_SKIP();
    }
    for (size_t pooling_elements = TestParam().primary_tile + TestParam().incremental_tile;
        pooling_elements <= TestParam().primary_tile + TestParam().incremental_tile * 3;
        pooling_elements = xnnpack::NextPrime(pooling_elements)) {
      if (TestParam().channel_scaled_tile == TestParam().channel_tile) {
        for (size_t channels = TestParam().channel_tile * 2; channels < TestParam().channel_tile * 8; channels += TestParam().channel_tile) {
          MaxPoolMicrokernelTester<CType>()
            .pooling_elements(pooling_elements)
            .pooling_tile(TestParam().primary_tile, TestParam().incremental_tile)
            .channels(channels)
            .qmin(TestParam().qmin)
            .Test(TestParam().kernel_fn, TestParam().params_fn);
        }
      } else {
        const size_t channel_tile = TestParam().channel_scaled_tile;
        for (size_t channels = channel_tile*2; channels < channel_tile*8; channels += channel_tile) {
          MaxPoolMicrokernelTester<CType>()
            .pooling_elements(pooling_elements)
            .pooling_tile(TestParam().primary_tile, TestParam().incremental_tile)
            .channels(channels)
            .qmin(TestParam().qmin)
            .Test(TestParam().kernel_fn, TestParam().params_fn);
        }
      }
    }
  }

  void channels_div_channel_tile_multipass_with_qmax() {
    TEST_REQUIRES_ARCH_FLAGS(TestParam().arch_flags);
    if (TestParam().channel_tile <= 1 || TestParam().channel_tile == TestParam().channel_scaled_tile) {
        GTEST_SKIP();
    }
    for (size_t pooling_elements = TestParam().primary_tile + TestParam().incremental_tile;
        pooling_elements <= TestParam().primary_tile + TestParam().incremental_tile * 3;
        pooling_elements = xnnpack::NextPrime(pooling_elements)) {
      if (TestParam().channel_scaled_tile == TestParam().channel_tile) {
        for (size_t channels = TestParam().channel_tile * 2; channels < TestParam().channel_tile * 8; channels += TestParam().channel_tile) {
          MaxPoolMicrokernelTester<CType>()
            .pooling_elements(pooling_elements)
            .pooling_tile(TestParam().primary_tile, TestParam().incremental_tile)
            .channels(channels)
            .qmax(TestParam().qmax)
            .Test(TestParam().kernel_fn, TestParam().params_fn);
        }
      } else {
        const size_t channel_tile = TestParam().channel_scaled_tile;
        for (size_t channels = channel_tile*2; channels < channel_tile*8; channels += channel_tile) {
          MaxPoolMicrokernelTester<CType>()
            .pooling_elements(pooling_elements)
            .pooling_tile(TestParam().primary_tile, TestParam().incremental_tile)
            .channels(channels)
            .qmax(TestParam().qmax)
            .Test(TestParam().kernel_fn, TestParam().params_fn);
        }
      }
    }
  }

  void channels_lt_channel_tile_multipass() {
    TEST_REQUIRES_ARCH_FLAGS(TestParam().arch_flags);
    if (TestParam().channel_tile <= 1 || TestParam().channel_tile == TestParam().channel_scaled_tile) {
        GTEST_SKIP();
    }
    const size_t channel_tile = TestParam().channel_scaled_tile;
    for (size_t pooling_elements = TestParam().primary_tile + TestParam().incremental_tile;
        pooling_elements <= TestParam().primary_tile + TestParam().incremental_tile * 3;
        pooling_elements = xnnpack::NextPrime(pooling_elements)) {
      for (size_t channels = 1; channels < channel_tile; channels++) {
        MaxPoolMicrokernelTester<CType>()
          .pooling_elements(pooling_elements)
          .pooling_tile(TestParam().primary_tile, TestParam().incremental_tile)
          .channels(channels)
          .Test(TestParam().kernel_fn, TestParam().params_fn);
      }
    }
  }

  void channels_lt_channel_tile_multipass_with_input_offset() {
    TEST_REQUIRES_ARCH_FLAGS(TestParam().arch_flags);
    if (TestParam().channel_tile <= 1 || TestParam().channel_tile == TestParam().channel_scaled_tile) {
        GTEST_SKIP();
    }
    const size_t channel_tile = TestParam().channel_scaled_tile;
    for (size_t pooling_elements = TestParam().primary_tile + TestParam().incremental_tile;
        pooling_elements <= TestParam().primary_tile + TestParam().incremental_tile * 3;
        pooling_elements = xnnpack::NextPrime(pooling_elements)) {
      for (size_t channels = 1; channels < channel_tile; channels++) {
        MaxPoolMicrokernelTester<CType>()
          .pooling_elements(pooling_elements)
          .pooling_tile(TestParam().primary_tile, TestParam().incremental_tile)
          .channels(channels)
          .input_offset(channel_tile)
          .Test(TestParam().kernel_fn, TestParam().params_fn);
      }
    }
  }

  void channels_lt_channel_tile_multipass_with_qmin() {
    TEST_REQUIRES_ARCH_FLAGS(TestParam().arch_flags);
    if (TestParam().channel_tile <= 1 || TestParam().channel_tile == TestParam().channel_scaled_tile) {
        GTEST_SKIP();
    }
    const size_t channel_tile = TestParam().channel_scaled_tile;
    for (size_t pooling_elements = TestParam().primary_tile + TestParam().incremental_tile;
        pooling_elements <= TestParam().primary_tile + TestParam().incremental_tile * 3;
        pooling_elements = xnnpack::NextPrime(pooling_elements)) {
      for (size_t channels = 1; channels < channel_tile; channels++) {
        MaxPoolMicrokernelTester<CType>()
          .pooling_elements(pooling_elements)
          .pooling_tile(TestParam().primary_tile, TestParam().incremental_tile)
          .channels(channels)
          .qmin(TestParam().qmin)
          .Test(TestParam().kernel_fn, TestParam().params_fn);
      }
    }
  }

  void channels_lt_channel_tile_multipass_with_qmax() {
    TEST_REQUIRES_ARCH_FLAGS(TestParam().arch_flags);
    if (TestParam().channel_tile <= 1 || TestParam().channel_tile == TestParam().channel_scaled_tile) {
        GTEST_SKIP();
    }
    const size_t channel_tile = TestParam().channel_scaled_tile;
    for (size_t pooling_elements = TestParam().primary_tile + TestParam().incremental_tile;
        pooling_elements <= TestParam().primary_tile + TestParam().incremental_tile * 3;
        pooling_elements = xnnpack::NextPrime(pooling_elements)) {
      for (size_t channels = 1; channels < channel_tile; channels++) {
        MaxPoolMicrokernelTester<CType>()
          .pooling_elements(pooling_elements)
          .pooling_tile(TestParam().primary_tile, TestParam().incremental_tile)
          .channels(channels)
          .qmax(TestParam().qmax)
          .Test(TestParam().kernel_fn, TestParam().params_fn);
      }
    }
  }

  void channels_gt_channel_tile_multipass() {
    TEST_REQUIRES_ARCH_FLAGS(TestParam().arch_flags);
    for (size_t pooling_elements = TestParam().primary_tile + TestParam().incremental_tile;
        pooling_elements <= TestParam().primary_tile + TestParam().incremental_tile * 3;
        pooling_elements = xnnpack::NextPrime(pooling_elements)) {
      if (TestParam().channel_scaled_tile == TestParam().channel_tile) {
        for (size_t channels = TestParam().channel_tile + 1; channels < (TestParam().channel_tile == 1 ? 10 : TestParam().channel_tile * 2); channels++) {
          MaxPoolMicrokernelTester<CType>()
            .pooling_elements(pooling_elements)
            .pooling_tile(TestParam().primary_tile, TestParam().incremental_tile)
            .channels(channels)
            .Test(TestParam().kernel_fn, TestParam().params_fn);
        }
      } else {
        const size_t channel_tile = TestParam().channel_scaled_tile;
        for (size_t channels = channel_tile+1; channels < channel_tile*2; channels++) {
          MaxPoolMicrokernelTester<CType>()
            .pooling_elements(pooling_elements)
            .pooling_tile(TestParam().primary_tile, TestParam().incremental_tile)
            .channels(channels)
            .Test(TestParam().kernel_fn, TestParam().params_fn);
        }
      }
    }
  }

  void channels_gt_channel_tile_multipass_with_input_offset() {
    TEST_REQUIRES_ARCH_FLAGS(TestParam().arch_flags);
    for (size_t pooling_elements = TestParam().primary_tile + TestParam().incremental_tile;
        pooling_elements <= TestParam().primary_tile + TestParam().incremental_tile * 3;
        pooling_elements = xnnpack::NextPrime(pooling_elements)) {
      if (TestParam().channel_scaled_tile == TestParam().channel_tile) {
        for (size_t channels = TestParam().channel_tile + 1; channels < (TestParam().channel_tile == 1 ? 10 : TestParam().channel_tile * 2); channels++) {
          MaxPoolMicrokernelTester<CType>()
            .pooling_elements(pooling_elements)
            .pooling_tile(TestParam().primary_tile, TestParam().incremental_tile)
            .channels(channels)
            .input_offset(xnnpack::NextPrime(TestParam().channel_tile * 2))
            .Test(TestParam().kernel_fn, TestParam().params_fn);
        }
      } else {
        const size_t channel_tile = TestParam().channel_scaled_tile;
        for (size_t channels = channel_tile+1; channels < channel_tile*2; channels++) {
          MaxPoolMicrokernelTester<CType>()
            .pooling_elements(pooling_elements)
            .pooling_tile(TestParam().primary_tile, TestParam().incremental_tile)
            .channels(channels)
            .input_offset(channel_tile*2)
            .Test(TestParam().kernel_fn, TestParam().params_fn);
        }
      }
    }
  }

  void channels_gt_channel_tile_multipass_with_qmin() {
    TEST_REQUIRES_ARCH_FLAGS(TestParam().arch_flags);
    for (size_t pooling_elements = TestParam().primary_tile + TestParam().incremental_tile;
        pooling_elements <= TestParam().primary_tile + TestParam().incremental_tile * 3;
        pooling_elements = xnnpack::NextPrime(pooling_elements)) {
      if (TestParam().channel_scaled_tile == TestParam().channel_tile) {
        for (size_t channels = TestParam().channel_tile + 1; channels < (TestParam().channel_tile == 1 ? 10 : TestParam().channel_tile * 2); channels++) {
          MaxPoolMicrokernelTester<CType>()
            .pooling_elements(pooling_elements)
            .pooling_tile(TestParam().primary_tile, TestParam().incremental_tile)
            .channels(channels)
            .qmin(TestParam().qmin)
            .Test(TestParam().kernel_fn, TestParam().params_fn);
        }
      } else {
        const size_t channel_tile = TestParam().channel_scaled_tile;
        for (size_t channels = channel_tile+1; channels < channel_tile*2; channels++) {
          MaxPoolMicrokernelTester<CType>()
            .pooling_elements(pooling_elements)
            .pooling_tile(TestParam().primary_tile, TestParam().incremental_tile)
            .channels(channels)
            .qmin(TestParam().qmin)
            .Test(TestParam().kernel_fn, TestParam().params_fn);
        }
      }
    }
  }

  void channels_gt_channel_tile_multipass_with_qmax() {
    TEST_REQUIRES_ARCH_FLAGS(TestParam().arch_flags);
    for (size_t pooling_elements = TestParam().primary_tile + TestParam().incremental_tile;
        pooling_elements <= TestParam().primary_tile + TestParam().incremental_tile * 3;
        pooling_elements = xnnpack::NextPrime(pooling_elements)) {
      if (TestParam().channel_scaled_tile == TestParam().channel_tile) {
        for (size_t channels = TestParam().channel_tile + 1; channels < (TestParam().channel_tile == 1 ? 10 : TestParam().channel_tile * 2); channels++) {
          MaxPoolMicrokernelTester<CType>()
            .pooling_elements(pooling_elements)
            .pooling_tile(TestParam().primary_tile, TestParam().incremental_tile)
            .channels(channels)
            .qmax(TestParam().qmax)
            .Test(TestParam().kernel_fn, TestParam().params_fn);
        }
      } else {
        const size_t channel_tile = TestParam().channel_scaled_tile;
        for (size_t channels = channel_tile+1; channels < channel_tile*2; channels++) {
          MaxPoolMicrokernelTester<CType>()
            .pooling_elements(pooling_elements)
            .pooling_tile(TestParam().primary_tile, TestParam().incremental_tile)
            .channels(channels)
            .qmax(TestParam().qmax)
            .Test(TestParam().kernel_fn, TestParam().params_fn);
        }
      }
    }
  }

  void few_output_pixels() {
    TEST_REQUIRES_ARCH_FLAGS(TestParam().arch_flags);
    for (size_t output_pixels = 2; output_pixels <= 5; output_pixels++) {
      for (size_t pooling_elements : std::vector<size_t>{{2, TestParam().primary_tile, TestParam().primary_tile + TestParam().incremental_tile - 1}}) {
        if (TestParam().channel_scaled_tile == TestParam().channel_tile) {
          for (size_t channels = 1; channels <= TestParam().channel_tile * 5; channels += std::max<size_t>(1, TestParam().channel_tile - 1)) {
            MaxPoolMicrokernelTester<CType>()
              .output_pixels(output_pixels)
              .pooling_elements(pooling_elements)
              .pooling_tile(TestParam().primary_tile, TestParam().incremental_tile)
              .channels(channels)
              .Test(TestParam().kernel_fn, TestParam().params_fn);
          }
        } else {
          const size_t channel_tile = TestParam().channel_scaled_tile;
          for (size_t channels = 1; channels <= channel_tile*5; channels += channel_tile-1) {
            MaxPoolMicrokernelTester<CType>()
              .output_pixels(output_pixels)
              .pooling_elements(pooling_elements)
              .pooling_tile(TestParam().primary_tile, TestParam().incremental_tile)
              .channels(channels)
              .Test(TestParam().kernel_fn, TestParam().params_fn);
          }
        }
      }
    }
  }

  void few_output_pixels_with_input_offset() {
    TEST_REQUIRES_ARCH_FLAGS(TestParam().arch_flags);
    for (size_t output_pixels = 2; output_pixels <= 5; output_pixels++) {
      for (size_t pooling_elements : std::vector<size_t>{{2, TestParam().primary_tile, TestParam().primary_tile + TestParam().incremental_tile - 1}}) {
        if (TestParam().channel_scaled_tile == TestParam().channel_tile) {
          for (size_t channels = 1; channels <= TestParam().channel_tile * 5; channels += std::max<size_t>(1, TestParam().channel_tile - 1)) {
            MaxPoolMicrokernelTester<CType>()
              .output_pixels(output_pixels)
              .pooling_elements(pooling_elements)
              .pooling_tile(TestParam().primary_tile, TestParam().incremental_tile)
              .channels(channels)
              .input_offset(xnnpack::NextPrime(TestParam().channel_tile * 5 + 1))
              .Test(TestParam().kernel_fn, TestParam().params_fn);
          }
        } else {
          const size_t channel_tile = TestParam().channel_scaled_tile;
          for (size_t channels = 1; channels <= channel_tile*5; channels += channel_tile-1) {
            MaxPoolMicrokernelTester<CType>()
              .output_pixels(output_pixels)
              .pooling_elements(pooling_elements)
              .pooling_tile(TestParam().primary_tile, TestParam().incremental_tile)
              .channels(channels)
              .input_offset(channel_tile*5+1)
              .Test(TestParam().kernel_fn, TestParam().params_fn);
          }
        }
      }
    }
  }

  void few_output_pixels_with_qmin() {
    TEST_REQUIRES_ARCH_FLAGS(TestParam().arch_flags);
    for (size_t output_pixels = 2; output_pixels <= 5; output_pixels++) {
      for (size_t pooling_elements : std::vector<size_t>{{2, TestParam().primary_tile, TestParam().primary_tile + TestParam().incremental_tile - 1}}) {
        if (TestParam().channel_scaled_tile == TestParam().channel_tile) {
          for (size_t channels = 1; channels <= TestParam().channel_tile * 5; channels += std::max<size_t>(1, TestParam().channel_tile - 1)) {
            MaxPoolMicrokernelTester<CType>()
              .output_pixels(output_pixels)
              .pooling_elements(pooling_elements)
              .pooling_tile(TestParam().primary_tile, TestParam().incremental_tile)
              .channels(channels)
              .qmin(TestParam().qmin)
              .Test(TestParam().kernel_fn, TestParam().params_fn);
          }
        } else {
          const size_t channel_tile = TestParam().channel_scaled_tile;
          for (size_t channels = 1; channels <= channel_tile*5; channels += channel_tile-1) {
            MaxPoolMicrokernelTester<CType>()
              .output_pixels(output_pixels)
              .pooling_elements(pooling_elements)
              .pooling_tile(TestParam().primary_tile, TestParam().incremental_tile)
              .channels(channels)
              .qmin(TestParam().qmin)
              .Test(TestParam().kernel_fn, TestParam().params_fn);
          }
        }
      }
    }
  }

  void few_output_pixels_with_qmax() {
    TEST_REQUIRES_ARCH_FLAGS(TestParam().arch_flags);
    for (size_t output_pixels = 2; output_pixels <= 5; output_pixels++) {
      for (size_t pooling_elements : std::vector<size_t>{{2, TestParam().primary_tile, TestParam().primary_tile + TestParam().incremental_tile - 1}}) {
        if (TestParam().channel_scaled_tile == TestParam().channel_tile) {
          for (size_t channels = 1; channels <= TestParam().channel_tile * 5; channels += std::max<size_t>(1, TestParam().channel_tile - 1)) {
            MaxPoolMicrokernelTester<CType>()
              .output_pixels(output_pixels)
              .pooling_elements(pooling_elements)
              .pooling_tile(TestParam().primary_tile, TestParam().incremental_tile)
              .channels(channels)
              .qmax(TestParam().qmax)
              .Test(TestParam().kernel_fn, TestParam().params_fn);
          }
        } else {
          const size_t channel_tile = TestParam().channel_scaled_tile;
          for (size_t channels = 1; channels <= channel_tile*5; channels += channel_tile-1) {
            MaxPoolMicrokernelTester<CType>()
              .output_pixels(output_pixels)
              .pooling_elements(pooling_elements)
              .pooling_tile(TestParam().primary_tile, TestParam().incremental_tile)
              .channels(channels)
              .qmax(TestParam().qmax)
              .Test(TestParam().kernel_fn, TestParam().params_fn);
          }
        }
      }
    }
  }

  void few_output_pixels_with_output_stride() {
    TEST_REQUIRES_ARCH_FLAGS(TestParam().arch_flags);
    for (size_t output_pixels = 2; output_pixels <= 5; output_pixels++) {
      for (size_t pooling_elements : std::vector<size_t>{{2, TestParam().primary_tile, TestParam().primary_tile + TestParam().incremental_tile - 1}}) {
        if (TestParam().channel_scaled_tile == TestParam().channel_tile) {
          for (size_t channels = 1; channels <= TestParam().channel_tile * 5; channels += std::max<size_t>(1, TestParam().channel_tile - 1)) {
            MaxPoolMicrokernelTester<CType>()
              .output_pixels(output_pixels)
              .pooling_elements(pooling_elements)
              .pooling_tile(TestParam().primary_tile, TestParam().incremental_tile)
              .channels(channels)
              .input_offset(xnnpack::NextPrime(TestParam().channel_tile * 5 + 1))
              .Test(TestParam().kernel_fn, TestParam().params_fn);
          }
        } else {
          const size_t channel_tile = TestParam().channel_scaled_tile;
          for (size_t channels = 1; channels <= channel_tile*5; channels += channel_tile-1) {
            MaxPoolMicrokernelTester<CType>()
              .output_pixels(output_pixels)
              .pooling_elements(pooling_elements)
              .pooling_tile(TestParam().primary_tile, TestParam().incremental_tile)
              .channels(channels)
              .output_stride(channel_tile*5+1)
              .Test(TestParam().kernel_fn, TestParam().params_fn);
          }
        }
      }
    }
  }

  void few_output_pixels_with_step() {
    TEST_REQUIRES_ARCH_FLAGS(TestParam().arch_flags);
    for (size_t output_pixels = 2; output_pixels <= 5; output_pixels++) {
      for (size_t pooling_elements : std::vector<size_t>{{2, TestParam().primary_tile, TestParam().primary_tile + TestParam().incremental_tile - 1}}) {
        if (TestParam().channel_scaled_tile == TestParam().channel_tile) {
          for (size_t channels = 1; channels <= TestParam().channel_tile * 5; channels += std::max<size_t>(1, TestParam().channel_tile - 1)) {
            for (size_t step = 2; step <= pooling_elements; step++) {
              MaxPoolMicrokernelTester<CType>()
                .output_pixels(output_pixels)
                .pooling_elements(pooling_elements)
                .pooling_tile(TestParam().primary_tile, TestParam().incremental_tile)
                .step(step)
                .channels(channels)
                .input_offset(xnnpack::NextPrime(TestParam().channel_tile * 5 + 1))
                .Test(TestParam().kernel_fn, TestParam().params_fn);
            }
          }
        } else {
          const size_t channel_tile = TestParam().channel_scaled_tile;
          for (size_t channels = 1; channels <= channel_tile*5; channels += channel_tile-1) {
            for (size_t step = 2; step <= pooling_elements; step = xnnpack::NextPrime(step)) {
              MaxPoolMicrokernelTester<CType>()
                .output_pixels(output_pixels)
                .pooling_elements(pooling_elements)
                .pooling_tile(TestParam().primary_tile, TestParam().incremental_tile)
                .step(step)
                .channels(channels)
                .output_stride(channel_tile*5+1)
                .Test(TestParam().kernel_fn, TestParam().params_fn);
            }
          }
        }
      }
    }
  }
};

