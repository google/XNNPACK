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
#include <functional>
#include <random>
#include <vector>

#include <fp16/fp16.h>

#include <xnnpack.h>
#include <xnnpack/aligned-allocator.h>
#include <xnnpack/pack.h>
#include <xnnpack/microfnptr.h>
#include <xnnpack/microparams-init.h>


class VMulCAddCMicrokernelTester {
 public:
  inline VMulCAddCMicrokernelTester& channel_tile(size_t channel_tile) {
    this->channel_tile_ = channel_tile;
    return *this;
  }

  inline size_t channel_tile() const {
    return this->channel_tile_;
  }

  inline VMulCAddCMicrokernelTester& channels(size_t channels) {
    assert(channels != 0);
    this->channels_ = channels;
    return *this;
  }

  inline size_t channels() const {
    return this->channels_;
  }

  inline size_t packed_channels() const {
    return channels() % channel_tile() == 0 ? channels() : (channels() / channel_tile() + 1) * channel_tile();
  }

  inline VMulCAddCMicrokernelTester& rows(size_t rows) {
    assert(rows != 0);
    this->rows_ = rows;
    return *this;
  }

  inline size_t rows() const {
    return this->rows_;
  }

  inline VMulCAddCMicrokernelTester& input_stride(size_t input_stride) {
    this->input_stride_ = input_stride;
    return *this;
  }

  inline size_t input_stride() const {
    return this->input_stride_ == 0 ? channels() : this->input_stride_;
  }

  inline VMulCAddCMicrokernelTester& output_stride(size_t output_stride) {
    this->output_stride_ = output_stride;
    return *this;
  }

  inline size_t output_stride() const {
    return this->output_stride_ == 0 ? channels() : this->output_stride_;
  }

  inline VMulCAddCMicrokernelTester& inplace(bool inplace) {
    this->inplace_ = inplace;
    return *this;
  }

  inline bool inplace() const {
    return this->inplace_;
  }

  inline VMulCAddCMicrokernelTester& qmin(uint8_t qmin) {
    this->qmin_ = qmin;
    return *this;
  }

  inline uint8_t qmin() const {
    return this->qmin_;
  }

  inline VMulCAddCMicrokernelTester& qmax(uint8_t qmax) {
    this->qmax_ = qmax;
    return *this;
  }

  inline uint8_t qmax() const {
    return this->qmax_;
  }

  inline VMulCAddCMicrokernelTester& iterations(size_t iterations) {
    this->iterations_ = iterations;
    return *this;
  }

  inline size_t iterations() const {
    return this->iterations_;
  }

  void Test(xnn_f16_vmulcaddc_ukernel_fn vmulcaddc, xnn_init_f16_minmax_params_fn init_params) const {
    std::random_device random_device;
    auto rng = std::mt19937(random_device());
    std::uniform_real_distribution<float> f32dist;

    if (inplace()) {
      ASSERT_EQ(input_stride(), output_stride());
    }

    std::vector<uint16_t> x((rows() - 1) * input_stride() + channels() + XNN_EXTRA_BYTES / sizeof(uint16_t));
    std::vector<uint16_t> scale(channels());
    std::vector<uint16_t> bias(channels());
    std::vector<uint16_t, AlignedAllocator<uint16_t, 64>> packed_w(packed_channels() * 2);
    std::vector<uint16_t> y((rows() - 1) * output_stride() + channels() + (inplace() ? XNN_EXTRA_BYTES / sizeof(uint16_t) : 0));
    std::vector<float> y_ref(rows() * channels());

    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(scale.begin(), scale.end(), [&]() { return fp16_ieee_from_fp32_value(f32dist(rng)); });
      std::generate(bias.begin(), bias.end(), [&]() { return fp16_ieee_from_fp32_value(f32dist(rng)); });
      std::generate(x.begin(), x.end(), [&]() { return fp16_ieee_from_fp32_value(f32dist(rng)); });
      if (inplace()) {
        std::copy(x.cbegin(), x.cend(), y.begin());
      } else {
        std::fill(y.begin(), y.end(), UINT16_C(0x7E00) /* NaN */);
      }
      const uint16_t* x_data = inplace() ? y.data() : x.data();

      std::fill(packed_w.begin(), packed_w.end(), UINT16_C(0x7E00) /* NaN */);
      xnn_pack_f16_vmulcaddc_w(channels(), channel_tile(),
        scale.data(), bias.data(), packed_w.data(), nullptr);

      // Compute reference results.
      for (size_t i = 0; i < rows(); i++) {
        for (size_t j = 0; j < channels(); j++) {
          y_ref[i * channels() + j] = fp16_ieee_to_fp32_value(x_data[i * input_stride() + j]) * fp16_ieee_to_fp32_value(scale[j]) + fp16_ieee_to_fp32_value(bias[j]);
        }
      }
      const float accumulated_min = *std::min_element(y_ref.cbegin(), y_ref.cend());
      const float accumulated_max = *std::max_element(y_ref.cbegin(), y_ref.cend());
      const float accumulated_range = accumulated_max - accumulated_min;
      const float y_max = fp16_ieee_to_fp32_value(fp16_ieee_from_fp32_value(accumulated_max - accumulated_range / 255.0f * float(255 - qmax())));
      const float y_min = fp16_ieee_to_fp32_value(fp16_ieee_from_fp32_value(accumulated_min + accumulated_range / 255.0f * float(qmin())));

      for (float& y_value : y_ref) {
        y_value = std::max(std::min(y_value, y_max), y_min);
      }

      // Prepare parameters.
      xnn_f16_minmax_params params;
      init_params(&params, fp16_ieee_from_fp32_value(y_min), fp16_ieee_from_fp32_value(y_max));

      // Call optimized micro-kernel.
      vmulcaddc(rows(), channels() * sizeof(uint16_t),
        x_data, input_stride() * sizeof(uint16_t),
        packed_w.data(),
        y.data(), output_stride() * sizeof(uint16_t),
        &params);

      // Verify results.
      for (size_t i = 0; i < rows(); i++) {
        for (size_t j = 0; j < channels(); j++) {
          EXPECT_NEAR(fp16_ieee_to_fp32_value(y[i * output_stride() + j]), y_ref[i * channels() + j], std::max(1.0e-4f, std::abs(y_ref[i * channels() + j]) * 1.0e-2f))
            << "at pixel " << i << " / " << rows()
            << ", channel = " << j << " / " << channels();
        }
      }
    }
  }

  void Test(xnn_f32_vmulcaddc_ukernel_fn vmulcaddc, xnn_init_f32_minmax_params_fn init_params) const {
    std::random_device random_device;
    auto rng = std::mt19937(random_device());
    std::uniform_real_distribution<float> f32dist;

    if (inplace()) {
      ASSERT_EQ(input_stride(), output_stride());
    }

    std::vector<float> x((rows() - 1) * input_stride() + channels() + XNN_EXTRA_BYTES / sizeof(float));
    std::vector<float> scale(channels());
    std::vector<float> bias(channels());
    std::vector<float, AlignedAllocator<float, 64>> packed_w(packed_channels() * 2);
    std::vector<float> y((rows() - 1) * output_stride() + channels() + (inplace() ? XNN_EXTRA_BYTES / sizeof(float) : 0));
    std::vector<float> y_ref(rows() * channels());
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(scale.begin(), scale.end(), [&]() { return f32dist(rng); });
      std::generate(bias.begin(), bias.end(), [&]() { return f32dist(rng); });
      std::generate(x.begin(), x.end(), [&]() { return f32dist(rng); });
      if (inplace()) {
        std::copy(x.cbegin(), x.cend(), y.begin());
      } else {
        std::fill(y.begin(), y.end(), nanf(""));
      }
      const float* x_data = inplace() ? y.data() : x.data();

      std::fill(packed_w.begin(), packed_w.end(), nanf(""));
      xnn_pack_f32_vmulcaddc_w(channels(), channel_tile(),
        scale.data(), bias.data(), packed_w.data(), nullptr);

      // Compute reference results.
      for (size_t i = 0; i < rows(); i++) {
        for (size_t j = 0; j < channels(); j++) {
          y_ref[i * channels() + j] = x_data[i * input_stride() + j] * scale[j] + bias[j];
        }
      }
      const float accumulated_min = *std::min_element(y_ref.cbegin(), y_ref.cend());
      const float accumulated_max = *std::max_element(y_ref.cbegin(), y_ref.cend());
      const float accumulated_range = accumulated_max - accumulated_min;
      const float y_max = accumulated_max - accumulated_range / 255.0f * float(255 - qmax());
      const float y_min = accumulated_min + accumulated_range / 255.0f * float(qmin());
      for (float& y_value : y_ref) {
        y_value = std::max<float>(std::min<float>(y_value, y_max), y_min);
      }

      // Prepare parameters.
      xnn_f32_minmax_params params;
      init_params(&params, y_min, y_max);

      // Call optimized micro-kernel.
      vmulcaddc(rows(), channels() * sizeof(float),
        x_data, input_stride() * sizeof(float),
        packed_w.data(),
        y.data(), output_stride() * sizeof(float),
        &params);

      // Verify results.
      for (size_t i = 0; i < rows(); i++) {
        for (size_t j = 0; j < channels(); j++) {
          EXPECT_NEAR(y[i * output_stride() + j], y_ref[i * channels() + j], std::abs(y_ref[i * channels() + j]) * 1.0e-6f)
            << "at pixel " << i << " / " << rows()
            << ", channel = " << j << " / " << channels();
        }
      }
    }
  }

 private:
  size_t channel_tile_{1};
  size_t channels_{1};
  size_t rows_{1};
  size_t input_stride_{0};
  size_t output_stride_{0};
  bool inplace_{false};
  uint8_t qmin_{0};
  uint8_t qmax_{255};
  size_t iterations_{15};
};
