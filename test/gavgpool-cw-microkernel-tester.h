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
#include <random>
#include <vector>

#include <gtest/gtest.h>
#include <fp16/fp16.h>
#include "xnnpack.h"
#include "xnnpack/microfnptr.h"
#include "xnnpack/microparams.h"
#include "replicable_random_device.h"

class GAvgPoolCWMicrokernelTester {
 public:
  GAvgPoolCWMicrokernelTester& elements(size_t elements) {
    assert(elements != 0);
    this->elements_ = elements;
    return *this;
  }

  size_t elements() const {
    return this->elements_;
  }

  GAvgPoolCWMicrokernelTester& channels(size_t channels) {
    assert(channels != 0);
    this->channels_ = channels;
    return *this;
  }

  size_t channels() const {
    return this->channels_;
  }

  GAvgPoolCWMicrokernelTester& qmin(uint8_t qmin) {
    this->qmin_ = qmin;
    return *this;
  }

  uint8_t qmin() const {
    return this->qmin_;
  }

  GAvgPoolCWMicrokernelTester& qmax(uint8_t qmax) {
    this->qmax_ = qmax;
    return *this;
  }

  uint8_t qmax() const {
    return this->qmax_;
  }

  GAvgPoolCWMicrokernelTester& iterations(size_t iterations) {
    this->iterations_ = iterations;
    return *this;
  }

  size_t iterations() const {
    return this->iterations_;
  }


  void Test(xnn_f32_gavgpool_cw_ukernel_fn gavgpool, xnn_init_f32_gavgpool_params_fn init_params) const {
    xnnpack::ReplicableRandomDevice rng;
    std::uniform_real_distribution<float> f32dist;

    std::vector<float> x(elements() * channels() + XNN_EXTRA_BYTES / sizeof(float));
    std::vector<float> y(channels());
    std::vector<float> y_ref(channels());
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      std::generate(x.begin(), x.end(), [&]() { return f32dist(rng); });
      std::fill(y.begin(), y.end(), std::nanf(""));

      // Compute reference results, without clamping.
      for (size_t i = 0; i < channels(); i++) {
        float acc = 0.0f;
        for (size_t j = 0; j < elements(); j++) {
          acc += x[i * elements() + j];
        }
        y_ref[i] = acc / float(elements());
      }

      // Compute clamping parameters.
      const float accumulated_min = *std::min_element(y_ref.cbegin(), y_ref.cend());
      const float accumulated_max = *std::max_element(y_ref.cbegin(), y_ref.cend());
      const float accumulated_range = accumulated_max - accumulated_min;
      const float y_min = accumulated_min + float(qmin()) / 255.0f * accumulated_range;
      const float y_max = accumulated_max - float(255 - qmax()) / 255.0f * accumulated_range;

      // Prepare parameters.
      union xnn_f32_gavgpool_params params;
      init_params(&params, 1.0f / float(elements()), y_min, y_max, elements());

      // Clamp reference results.
      for (float& y_value : y_ref) {
        y_value = std::max(std::min(y_value, y_max), y_min);
      }

      // Call optimized micro-kernel.
      gavgpool(elements() * sizeof(float), channels(), x.data(), y.data(), &params);

      // Verify results.
      for (size_t i = 0; i < channels(); i++) {
        EXPECT_LE(y[i], y_max)
          << "at position " << i << ", elements = " << elements() << ", channels = " << channels();
        EXPECT_GE(y[i], y_min)
          << "at position " << i << ", elements = " << elements() << ", channels = " << channels();
        EXPECT_NEAR(y[i], y_ref[i], std::abs(y_ref[i]) * 1.0e-6f)
          << "at position " << i << ", elements = " << elements() << ", channels = " << channels();
      }
    }
  }

void Test(xnn_f16_gavgpool_cw_ukernel_fn gavgpool, xnn_init_f16_gavgpool_neon_params_fn init_params) const {
  xnnpack::ReplicableRandomDevice rng;
  std::uniform_real_distribution<float> f32dist(0.1f, 10.0f);

  std::vector<uint16_t> x(elements() * channels() +
                          XNN_EXTRA_BYTES / sizeof(uint16_t));
  std::vector<uint16_t> y(channels());
  std::vector<float> y_ref(channels());
  for (size_t iteration = 0; iteration < iterations(); iteration++) {
    std::generate(x.begin(), x.end(),
                  [&]() { return fp16_ieee_from_fp32_value(f32dist(rng)); });
    std::fill(y.begin(), y.end(), UINT16_C(0x7E00) /* NaN */);

    // Compute reference results, without clamping.
    for (size_t i = 0; i < channels(); i++) {
      float acc = 0.0f;
      for (size_t j = 0; j < elements(); j++) {
        acc += fp16_ieee_to_fp32_value(x[i * elements() + j]);
      }
      y_ref[i] = acc / float(elements());
    }

    // Compute clamping parameters.
    const float accumulated_min =
        *std::min_element(y_ref.cbegin(), y_ref.cend());
    const float accumulated_max =
        *std::max_element(y_ref.cbegin(), y_ref.cend());
    const float accumulated_range = accumulated_max - accumulated_min;
    const float y_min = fp16_ieee_to_fp32_value(fp16_ieee_from_fp32_value(
        accumulated_min + accumulated_range / 255.0f * float(qmin())));
    const float y_max = fp16_ieee_to_fp32_value(fp16_ieee_from_fp32_value(
        accumulated_max - accumulated_range / 255.0f * float(255 - qmax())));

    // Prepare parameters.
    union xnn_f16_gavgpool_params params;
    init_params(&params, fp16_ieee_from_fp32_value(1.0f / float(elements())),
                fp16_ieee_from_fp32_value(y_min),
                fp16_ieee_from_fp32_value(y_max), elements());

    // Clamp reference results.
    for (float& y_value : y_ref) {
      y_value = std::max(std::min(y_value, y_max), y_min);
    }

    // Call optimized micro-kernel.
    gavgpool(elements() * sizeof(uint16_t), channels(), x.data(), y.data(),
             &params);

    // Verify results.
    for (size_t i = 0; i < channels(); i++) {
      EXPECT_LE(fp16_ieee_to_fp32_value(y[i]), y_max)
          << "at position " << i << ", elements = " << elements()
          << ", channels = " << channels();
      EXPECT_GE(fp16_ieee_to_fp32_value(y[i]), y_min)
          << "at position " << i << ", elements = " << elements()
          << ", channels = " << channels();
      EXPECT_NEAR(fp16_ieee_to_fp32_value(y[i]), y_ref[i],
                  1.0e-2f * std::abs(y_ref[i]))
          << "at position " << i << ", elements = " << elements()
          << ", channels = " << channels();
    }
  }
}

 private:
  size_t elements_{1};
  size_t channels_{1};
  uint8_t qmin_{0};
  uint8_t qmax_{255};
  size_t iterations_{15};
};
