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
#include <cfloat>
#include <cmath>
#include <cstddef>
#include <cstdlib>
#include <functional>
#include <limits>
#include <random>
#include <vector>

#include <xnnpack/params.h>
#include <xnnpack/requantization-stubs.h>
#include <xnnpack/scalar-utils.h>


class RequantizationTester {
 public:
  inline RequantizationTester& s(uint32_t s) {
    this->s_ = s;
    return *this;
  }

  inline uint32_t s() const {
    return this->s_;
  }

  inline float scale() const {
    return ldexpf(1.0f, -s());
  }

  inline RequantizationTester& zero_point(int32_t zero_point) {
    this->zero_point_ = zero_point;
    return *this;
  }

  inline int32_t zero_point() const {
    return this->zero_point_;
  }

  inline RequantizationTester& qmin(uint8_t qmin) {
    this->qmin_ = qmin;
    return *this;
  }

  inline uint8_t qmin() const {
    return this->qmin_;
  }

  inline RequantizationTester& qmax(uint8_t qmax) {
    this->qmax_ = qmax;
    return *this;
  }

  inline uint8_t qmax() const {
    return this->qmax_;
  }

  inline RequantizationTester& iterations(size_t iterations) {
    this->iterations_ = iterations;
    return *this;
  }

  inline size_t iterations() const {
    return this->iterations_;
  }

  /*
   * Test that requantization of numbers ((i - zero point) * 2**s) with
   * - scale = exp2(-s)
   * - zero point in [0, 255]
   * - no output clamping
   * produces exactly i, provided that ((i - zero point) * 2**s) does not overflow.
   */
  void TestExactDivideByPO2(requantization_function requantize) const {
    ASSERT_GE(zero_point(), 0);
    ASSERT_LE(zero_point(), 255);

    /* Note: need s >= 1 to ensure scale = exp2(-s) < 1.0 */
    ASSERT_GE(s(), 1);
    ASSERT_LT(s(), 32);

    std::vector<int32_t> inputs(256);
    std::vector<uint8_t> outputs(inputs.size());
    const int32_t maxI = (uint32_t(std::numeric_limits<int32_t>::max()) >> s()) + zero_point();
    const int32_t minI = -(-uint32_t(std::numeric_limits<int32_t>::min()) >> s()) + zero_point();
    for (int32_t i = 0; i < 256; i++) {
      const int32_t clampedI = std::max(minI, std::min(maxI, i));
      inputs[i] = int32_t(uint32_t(clampedI - zero_point()) << s());
    }
    requantize(inputs.size(), inputs.data(),
        scale(), zero_point(), qmin(), qmax(),
        outputs.data());
    for (int32_t i = 0; i < 256; i++) {
      const int32_t clampedI = std::max(minI, std::min(maxI, i));
      ASSERT_EQ(clampedI, outputs[i]) << "i = " << i << ", clamped i = " << clampedI <<
        ", min i = " << minI << ", max i = " << maxI <<
        ", s = " << s() << ", zero point = " << zero_point();
    }
  }

  /*
   * Test that requantization of numbers (i * 2**s + sign(i - zero point) * 2**(s-1)) with
   * - scale = exp2(-s)
   * - zero point in [1, 255]
   * - no output clamping
   * produces exactly i, provided that ((i - zero point) * 2**s) does not overflow.
   */
  void TestDivideByPO2WithRoundingUp(requantization_function requantize) {
    ASSERT_GE(zero_point(), 0);
    ASSERT_LE(zero_point(), 255);

    /* Note: need s >= 1 to ensure scale = exp2(-s) < 1.0 */
    ASSERT_GE(s(), 1);
    ASSERT_LT(s(), 32);

    std::vector<int32_t> inputs(256);
    std::vector<uint8_t> outputs(inputs.size());
    for (int32_t i = 0; i < 256; i++) {
      const int64_t input = RequantizationTester::ShiftLeft(i - zero_point(), s()) -
        (INT64_C(1) << (s() - 1)) + (int64_t) (i <= zero_point());
      inputs[i] = int32_t(input);
    }
    requantize(inputs.size(), inputs.data(),
        scale(), zero_point(), qmin(), qmax(),
        outputs.data());
    for (int32_t i = 0; i < 256; i++) {
      const int64_t input = RequantizationTester::ShiftLeft(i - zero_point(), s()) -
        (INT64_C(1) << (s() - 1)) + (int64_t) (i <= zero_point());
      if (int32_t(input) == input) {
        ASSERT_EQ(i, uint32_t(outputs[i])) << "i = " << i << ", input = " << input <<
          ", s = " << s() << ", zero point = " << zero_point();
      }
    }
  }

  /*
   * Test that requantization of numbers (i * 2**s + sign(i - zero point) * 2**(s-1)) with
   * - scale = exp2(-s)
   * - zero point in [1, 255]
   * - no output clamping
   * produces exactly i, provided that ((i - zero point) * 2**s) does not overflow.
   */
  void TestDivideByPO2WithRoundingDown(requantization_function requantize) {
    ASSERT_GE(zero_point(), 0);
    ASSERT_LE(zero_point(), 255);

    /* Note: need s >= 1 to ensure scale = exp2(-s) < 1.0 */
    ASSERT_GE(s(), 1);
    ASSERT_LT(s(), 32);

    std::vector<int32_t> inputs(256);
    std::vector<uint8_t> outputs(inputs.size());
    for (int32_t i = 0; i < 256; i++) {
      const int64_t input = RequantizationTester::ShiftLeft(i - zero_point(), s()) +
        (INT64_C(1) << (s() - 1)) - (int64_t) (i >= zero_point());
      inputs[i] = int32_t(input);
    }
    requantize(inputs.size(), inputs.data(),
        scale(), zero_point(), qmin(), qmax(),
        outputs.data());
    for (int32_t i = 0; i < 256; i++) {
      const int64_t input = RequantizationTester::ShiftLeft(i - zero_point(), s()) +
        (INT64_C(1) << (s() - 1)) - (int64_t) (i >= zero_point());
      if (int32_t(input) == input) {
        ASSERT_EQ(i, uint32_t(outputs[i])) << "i = " << i << ", input = " << input <<
          ", s = " << s() << ", zero point = " << zero_point();
      }
    }
  }

  void TestDivideByPO2WithRoundingAway(requantization_function requantize) {
    ASSERT_GE(zero_point(), 0);
    ASSERT_LE(zero_point(), 255);

    /* Note: need s >= 1 to ensure scale = exp2(-s) < 1.0 */
    ASSERT_GE(s(), 1);
    ASSERT_LT(s(), 32);

    std::vector<int32_t> inputs(256);
    std::vector<uint8_t> outputs(inputs.size());
    for (int32_t i = 0; i < 256; i++) {
      int64_t input = RequantizationTester::ShiftLeft(i - zero_point(), s());
      if (input > 0) {
        input -= INT64_C(1) << (s() - 1);
      } else if (input < 0) {
        input += INT64_C(1) << (s() - 1);
      }
      inputs[i] = int32_t(input);
    }
    requantize(inputs.size(), inputs.data(),
        scale(), zero_point(), qmin(), qmax(),
        outputs.data());
    for (uint32_t i = 0; i < 256; i++) {
      int64_t input = RequantizationTester::ShiftLeft(i - zero_point(), s());
      if (input > 0) {
        input -= INT64_C(1) << (s() - 1);
      } else if (input < 0) {
        input += INT64_C(1) << (s() - 1);
      }
      if (int32_t(input) == input) {
        ASSERT_EQ(i, uint32_t(outputs[i])) << "i = " << i << ", input = " << input <<
          ", s = " << s() << ", zero point = " << zero_point();
      }
    }
  }

  void TestSpecialCases(requantization_function requantize) {
    std::vector<int32_t> inputs(256);
    std::vector<uint8_t> outputs(inputs.size());

    std::fill(inputs.begin(), inputs.end(), std::numeric_limits<int32_t>::min());
    for (int32_t zero_point = 0; zero_point < 256; zero_point++) {
      requantize(
          inputs.size(),
          inputs.data(),
          ldexpf(1.0f, -32) /* scale */,
          zero_point /* zero point */,
          std::numeric_limits<uint8_t>::min(),
          std::numeric_limits<uint8_t>::max(),
          outputs.data());
      ASSERT_EQ(std::max(int32_t(0), zero_point - 1), *std::min_element(outputs.cbegin(), outputs.cend()));
    }

    std::fill(inputs.begin(), inputs.end(), std::numeric_limits<int32_t>::max());
    requantize(
        inputs.size(),
        inputs.data(),
        0x1.FFFFFEp-1f /* scale */,
        std::numeric_limits<uint8_t>::max() /* zero point */,
        std::numeric_limits<uint8_t>::min(),
        std::numeric_limits<uint8_t>::max(),
        outputs.data());
    for (size_t i = 0; i < inputs.size(); i++) {
      ASSERT_EQ(std::numeric_limits<uint8_t>::max(), outputs[i]);
    }
  }

  void TestRandomCasesPrecise(requantization_function requantize) {
    std::random_device random_device;
    std::mt19937 rng(random_device());
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      auto u8rng = std::bind(std::uniform_int_distribution<uint32_t>(0, std::numeric_limits<uint8_t>::max()), rng);

      std::vector<int32_t> inputs(4096);
      std::vector<uint8_t> outputs(inputs.size());

      const uint8_t zero_point = UINT8_C(128);
      std::uniform_real_distribution<float> scale_distribution(0x1.000000p-23f, 0x1.FFFFFEp-1f);
      const float scale = scale_distribution(rng);
      for (size_t i = 0; i < inputs.size(); i++) {
        const uint8_t approximate_output = u8rng();
        const int32_t input = int32_t(double(approximate_output) / double(scale));
        inputs[i] = input;
      }

      requantize(
        inputs.size(), inputs.data(), scale, zero_point,
        std::numeric_limits<uint8_t>::min(),
        std::numeric_limits<uint8_t>::max(),
        outputs.data());

      /* Ensure that outputs are not all identical, as in this case Test doesn't validate much */
      ASSERT_NE(
        *std::max_element(outputs.cbegin(), outputs.cend()),
        *std::min_element(outputs.cbegin(), outputs.cend()));

      for (size_t i = 0; i < inputs.size(); i++) {
        const uint8_t reference_output =
          scalar_requantize_precise(
            inputs[i], scale, zero_point,
            std::numeric_limits<uint8_t>::min(),
            std::numeric_limits<uint8_t>::max());
        ASSERT_EQ(uint32_t(reference_output), uint32_t(outputs[i]));
      }
    }
  }

  void TestRandomCasesApproximate(requantization_function requantize) {
    std::random_device random_device;
    std::mt19937 rng(random_device());
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      auto u8rng = std::bind(std::uniform_int_distribution<uint32_t>(0, std::numeric_limits<uint8_t>::max()), rng);

      std::vector<int32_t> inputs(4096);
      std::vector<uint8_t> outputs(inputs.size());

      const uint8_t zero_point = UINT8_C(128);
      std::uniform_real_distribution<float> scale_distribution(0x1.000000p-23f, 0x1.FFFFFEp-1f);
      const float scale = scale_distribution(rng);
      for (size_t i = 0; i < inputs.size(); i++) {
        const uint8_t approximate_output = u8rng();
        const int32_t input = int32_t(double(approximate_output) / double(scale));
        inputs[i] = input;
      }

      requantize(
        inputs.size(), inputs.data(), scale, zero_point,
        std::numeric_limits<uint8_t>::min(),
        std::numeric_limits<uint8_t>::max(),
        outputs.data());

      /* Ensure that outputs are not all identical, as in this case Test doesn't validate much */
      ASSERT_NE(
        *std::max_element(outputs.cbegin(), outputs.cend()),
        *std::min_element(outputs.cbegin(), outputs.cend()));

      for (size_t i = 0; i < inputs.size(); i++) {
        const double reference_output =
          RequantizationTester::RequantizeApproximate(
            inputs[i], scale, zero_point,
            std::numeric_limits<uint8_t>::min(),
            std::numeric_limits<uint8_t>::max());
        ASSERT_LE(fabs(reference_output - double(outputs[i])), 0.55) <<
          "input = " << inputs[i] <<
          ", output = " << uint32_t(outputs[i]) << ", reference output = " << reference_output;
      }
    }
  }

  void TestRandomCasesAgainstReference(requantization_function requantize, requantization_function requantize_reference) {
    std::random_device random_device;
    std::mt19937 rng(random_device());
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      auto u8rng = std::bind(std::uniform_int_distribution<uint32_t>(0, std::numeric_limits<uint8_t>::max()), rng);

      std::vector<int32_t> inputs(4096);
      std::vector<uint8_t> outputs(inputs.size());
      std::vector<uint8_t> reference_outputs(inputs.size());

      const uint8_t zero_point = UINT8_C(128);
      std::uniform_real_distribution<float> scale_distribution(0x1.000000p-23f, 0x1.FFFFFEp-1f);
      const float scale = scale_distribution(rng);
      for (size_t i = 0; i < inputs.size(); i++) {
        const uint8_t approximate_output = u8rng();
        const int32_t input = int32_t(double(approximate_output) / double(scale));
        inputs[i] = input;
      }

      requantize(
        inputs.size(), inputs.data(), scale, zero_point,
        std::numeric_limits<uint8_t>::min(),
        std::numeric_limits<uint8_t>::max(),
        outputs.data());

      requantize_reference(
        inputs.size(), inputs.data(), scale, zero_point,
        std::numeric_limits<uint8_t>::min(),
        std::numeric_limits<uint8_t>::max(),
        reference_outputs.data());

      /* Ensure that outputs are not all identical, as in this case Test doesn't validate much */
      ASSERT_NE(
        *std::max_element(outputs.cbegin(), outputs.cend()),
        *std::min_element(outputs.cbegin(), outputs.cend()));

      for (size_t i = 0; i < inputs.size(); i++) {
        ASSERT_EQ(uint32_t(reference_outputs[i]), uint32_t(outputs[i]));
      }
    }
  }

  static inline int64_t ShiftLeft(int64_t w, uint32_t n) {
    return (int64_t) ((uint64_t) w << n);
  }

  static inline double RequantizeApproximate(
    int32_t value,
    float scale,
    uint8_t zero_point,
    uint8_t qmin,
    uint8_t qmax)
  {
    assert(scale < 1.0f);
    assert(scale >= 0x1.0p-32f);

    double clamped_value = double(value) * double(scale) + double(zero_point);

    const double fmin = double(qmin);
    if (clamped_value < fmin) {
      clamped_value = fmin;
    }

    const double fmax = double(qmax);
    if (clamped_value > fmax) {
      clamped_value = fmax;
    }

    return clamped_value;
  }

 private:
  size_t zero_point_{0};
  size_t s_{1};
  uint8_t qmin_{std::numeric_limits<uint8_t>::min()};
  uint8_t qmax_{std::numeric_limits<uint8_t>::max()};
  size_t iterations_{1};
};
