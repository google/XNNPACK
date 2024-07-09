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
#include <functional>
#include <limits>
#include <random>
#include <vector>

#include <gtest/gtest.h>
#include "xnnpack/requantization-stubs.h"
#include "xnnpack/requantization.h"
#include "replicable_random_device.h"

class RequantizationTester {
 public:
  RequantizationTester& s(uint32_t s) {
    this->s_ = s;
    return *this;
  }

  uint32_t s() const {
    return this->s_;
  }

  float scale() const {
    return ldexpf(1.0f, -s());
  }

  RequantizationTester& zero_point(int32_t zero_point) {
    this->zero_point_ = zero_point;
    return *this;
  }

  int32_t zero_point() const {
    return this->zero_point_;
  }

  RequantizationTester& qmin(int16_t qmin) {
    this->qmin_ = qmin;
    return *this;
  }

  int16_t qmin() const {
    return this->qmin_;
  }

  RequantizationTester& qmax(int16_t qmax) {
    this->qmax_ = qmax;
    return *this;
  }

  int16_t qmax() const {
    return this->qmax_;
  }

  RequantizationTester& iterations(size_t iterations) {
    this->iterations_ = iterations;
    return *this;
  }

  size_t iterations() const {
    return this->iterations_;
  }

  /*
   * Test that requantization of numbers ((i - zero point) * 2**s) with
   * - scale = exp2(-s)
   * - zero point in [0, 255]
   * - no output clamping
   * produces exactly i, provided that ((i - zero point) * 2**s) does not overflow.
   */
  void TestExactDivideByPO2(xnn_qu8_requantization_fn requantize) const {
    ASSERT_GE(zero_point(), std::numeric_limits<uint8_t>::min());
    ASSERT_LE(zero_point(), std::numeric_limits<uint8_t>::max());
    ASSERT_GE(qmin(), std::numeric_limits<uint8_t>::min());
    ASSERT_LE(qmin(), std::numeric_limits<uint8_t>::max());
    ASSERT_GE(qmax(), std::numeric_limits<uint8_t>::min());
    ASSERT_LE(qmax(), std::numeric_limits<uint8_t>::max());
    ASSERT_LT(qmin(), qmax());

    /* Note: need s >= 1 to ensure scale = exp2(-s) < 1.0 */
    ASSERT_GE(s(), 1);
    ASSERT_LT(s(), 32);

    std::vector<int32_t> inputs(256);
    std::vector<uint8_t> outputs(inputs.size());
    const int32_t max_i = (uint32_t(std::numeric_limits<int32_t>::max()) >> s()) + zero_point();
    const int32_t min_i = -(-uint32_t(std::numeric_limits<int32_t>::min()) >> s()) + zero_point();
    for (int32_t i = 0; i <= std::numeric_limits<uint8_t>::max(); i++) {
      const int32_t clamped_i = std::max(min_i, std::min(max_i, i));
      inputs[i] = int32_t(uint32_t(clamped_i - zero_point()) << s());
    }
    requantize(inputs.size(), inputs.data(),
        scale(), zero_point(), qmin(), qmax(),
        outputs.data());
    for (int32_t i = 0; i <= std::numeric_limits<uint8_t>::max(); i++) {
      const int32_t clamped_i = std::max(min_i, std::min(max_i, i));
      ASSERT_EQ(uint32_t(clamped_i), uint32_t(outputs[i]))
        << "i = " << i << ", clamped i = " << clamped_i << ", input = " << inputs[i]
        << ", min i = " << min_i << ", max i = " << max_i
        << ", s = " << s() << ", zero point = " << zero_point();
    }
  }

  /*
   * Test that requantization of numbers ((i - zero point) * 2**s) with
   * - scale = exp2(-s)
   * - zero point in [-128, 127]
   * - no output clamping
   * produces exactly i, provided that ((i - zero point) * 2**s) does not overflow.
   */
  void TestExactDivideByPO2(xnn_qs8_requantization_fn requantize) const {
    ASSERT_GE(zero_point(), std::numeric_limits<int8_t>::min());
    ASSERT_LE(zero_point(), std::numeric_limits<int8_t>::max());
    ASSERT_GE(qmin(), std::numeric_limits<int8_t>::min());
    ASSERT_LE(qmin(), std::numeric_limits<int8_t>::max());
    ASSERT_GE(qmax(), std::numeric_limits<int8_t>::min());
    ASSERT_LE(qmax(), std::numeric_limits<int8_t>::max());
    ASSERT_LT(qmin(), qmax());

    /* Note: need s >= 1 to ensure scale = exp2(-s) < 1.0 */
    ASSERT_GE(s(), 1);
    ASSERT_LT(s(), 32);

    std::vector<int32_t> inputs(256);
    std::vector<int8_t> outputs(inputs.size());
    const int32_t max_i = (uint32_t(std::numeric_limits<int32_t>::max()) >> s()) + zero_point();
    const int32_t min_i = -(-uint32_t(std::numeric_limits<int32_t>::min()) >> s()) + zero_point();
    for (int32_t i = std::numeric_limits<int8_t>::min(); i <= std::numeric_limits<int8_t>::max(); i++) {
      const int32_t clamped_i = std::max(min_i, std::min(max_i, i));
      inputs[i - std::numeric_limits<int8_t>::min()] = int32_t(uint32_t(clamped_i - zero_point()) << s());
    }
    requantize(inputs.size(), inputs.data(),
        scale(), zero_point(), qmin(), qmax(),
        outputs.data());
    for (int32_t i = std::numeric_limits<int8_t>::min(); i <= std::numeric_limits<int8_t>::max(); i++) {
      const int32_t clamped_i = std::max(min_i, std::min(max_i, i));
      ASSERT_EQ(clamped_i, int32_t(outputs[i - std::numeric_limits<int8_t>::min()]))
        << "i = " << i << ", clamped i = " << clamped_i
        << ", input = " << inputs[i - std::numeric_limits<int8_t>::min()]
        << ", min i = " << min_i << ", max i = " << max_i
        << ", s = " << s() << ", zero point = " << zero_point();
    }
  }

  /*
   * Test that requantization of numbers ((i - zero point) * 2**s - 2**(s-1) + 1) with
   * - scale = exp2(-s)
   * - zero point in [1, 255]
   * - no output clamping
   * produces exactly i, provided that ((i - zero point) * 2**s) does not overflow.
   */
  void TestDivideByPO2WithRoundingUp(xnn_qu8_requantization_fn requantize) {
    ASSERT_GE(zero_point(), std::numeric_limits<uint8_t>::min());
    ASSERT_LE(zero_point(), std::numeric_limits<uint8_t>::max());
    ASSERT_GE(qmin(), std::numeric_limits<uint8_t>::min());
    ASSERT_LE(qmin(), std::numeric_limits<uint8_t>::max());
    ASSERT_GE(qmax(), std::numeric_limits<uint8_t>::min());
    ASSERT_LE(qmax(), std::numeric_limits<uint8_t>::max());
    ASSERT_LT(qmin(), qmax());

    /* Note: need s >= 1 to ensure scale = exp2(-s) < 1.0 */
    ASSERT_GE(s(), 1);
    ASSERT_LT(s(), 32);

    std::vector<int32_t> inputs(256);
    std::vector<uint8_t> outputs(inputs.size());
    for (int32_t i = 0; i <= std::numeric_limits<uint8_t>::max(); i++) {
      const int64_t input = RequantizationTester::ShiftLeft(i - zero_point(), s()) -
        (INT64_C(1) << (s() - 1)) + 1;
      inputs[i] = int32_t(input);
    }
    requantize(inputs.size(), inputs.data(),
        scale(), zero_point(), qmin(), qmax(),
        outputs.data());
    for (int32_t i = 0; i <= std::numeric_limits<uint8_t>::max(); i++) {
      const int64_t input = RequantizationTester::ShiftLeft(i - zero_point(), s()) -
        (INT64_C(1) << (s() - 1)) + 1;
      if (int32_t(input) == input) {
        ASSERT_EQ(i, int32_t(outputs[i]))
          << "i = " << i << ", input = " << input
          << ", s = " << s() << ", zero point = " << zero_point();
      }
    }
  }

  /*
   * Test that requantization of numbers ((i - zero point) * 2**s - 2**(s-1) + 1) with
   * - scale = exp2(-s)
   * - zero point in [-128, 127]
   * - no output clamping
   * produces exactly i, provided that ((i - zero point) * 2**s) does not overflow.
   */
  void TestDivideByPO2WithRoundingUp(xnn_qs8_requantization_fn requantize) {
    ASSERT_GE(zero_point(), std::numeric_limits<int8_t>::min());
    ASSERT_LE(zero_point(), std::numeric_limits<int8_t>::max());
    ASSERT_GE(qmin(), std::numeric_limits<int8_t>::min());
    ASSERT_LE(qmin(), std::numeric_limits<int8_t>::max());
    ASSERT_GE(qmax(), std::numeric_limits<int8_t>::min());
    ASSERT_LE(qmax(), std::numeric_limits<int8_t>::max());
    ASSERT_LT(qmin(), qmax());

    /* Note: need s >= 1 to ensure scale = exp2(-s) < 1.0 */
    ASSERT_GE(s(), 1);
    ASSERT_LT(s(), 32);

    std::vector<int32_t> inputs(256);
    std::vector<int8_t> outputs(inputs.size());
    for (int32_t i = std::numeric_limits<int8_t>::min(); i <= std::numeric_limits<int8_t>::max(); i++) {
      const int64_t input = RequantizationTester::ShiftLeft(i - zero_point(), s()) -
        (INT64_C(1) << (s() - 1)) + 1;
      inputs[i - std::numeric_limits<int8_t>::min()] = int32_t(input);
    }
    requantize(inputs.size(), inputs.data(),
        scale(), zero_point(), qmin(), qmax(),
        outputs.data());
    for (int32_t i = std::numeric_limits<int8_t>::min(); i <= std::numeric_limits<int8_t>::max(); i++) {
      const int64_t input = RequantizationTester::ShiftLeft(i - zero_point(), s()) -
        (INT64_C(1) << (s() - 1)) + 1;
      if (int32_t(input) == input) {
        ASSERT_EQ(i, int32_t(outputs[i - std::numeric_limits<int8_t>::min()]))
          << "i = " << i << ", input = " << input
          << ", s = " << s() << ", zero point = " << zero_point();
      }
    }
  }

  /*
   * Test that requantization of numbers ((i - zero point) * 2**s + 2**(s-1) - 1) with
   * - scale = exp2(-s)
   * - zero point in [1, 255]
   * - no output clamping
   * produces exactly i, provided that ((i - zero point) * 2**s) does not overflow.
   */
  void TestDivideByPO2WithRoundingDown(xnn_qu8_requantization_fn requantize) {
    ASSERT_GE(zero_point(), std::numeric_limits<uint8_t>::min());
    ASSERT_LE(zero_point(), std::numeric_limits<uint8_t>::max());
    ASSERT_GE(qmin(), std::numeric_limits<uint8_t>::min());
    ASSERT_LE(qmin(), std::numeric_limits<uint8_t>::max());
    ASSERT_GE(qmax(), std::numeric_limits<uint8_t>::min());
    ASSERT_LE(qmax(), std::numeric_limits<uint8_t>::max());
    ASSERT_LT(qmin(), qmax());

    /* Note: need s >= 1 to ensure scale = exp2(-s) < 1.0 */
    ASSERT_GE(s(), 1);
    ASSERT_LT(s(), 32);

    std::vector<int32_t> inputs(256);
    std::vector<uint8_t> outputs(inputs.size());
    for (int32_t i = 0; i <= std::numeric_limits<uint8_t>::max(); i++) {
      const int64_t input = RequantizationTester::ShiftLeft(i - zero_point(), s()) +
        (INT64_C(1) << (s() - 1)) - 1;
      inputs[i] = int32_t(input);
    }
    requantize(inputs.size(), inputs.data(),
        scale(), zero_point(), qmin(), qmax(),
        outputs.data());
    for (int32_t i = 0; i <= std::numeric_limits<uint8_t>::max(); i++) {
      const int64_t input = RequantizationTester::ShiftLeft(i - zero_point(), s()) +
        (INT64_C(1) << (s() - 1)) - 1;
      if (int32_t(input) == input) {
        ASSERT_EQ(i, int32_t(outputs[i]))
          << "i = " << i << ", input = " << input
          << ", s = " << s() << ", zero point = " << zero_point();
      }
    }
  }

  /*
   * Test that requantization of numbers ((i - zero point) * 2**s + 2**(s-1) - 1) with
   * - scale = exp2(-s)
   * - zero point in [-128, 127]
   * - no output clamping
   * produces exactly i, provided that ((i - zero point) * 2**s) does not overflow.
   */
  void TestDivideByPO2WithRoundingDown(xnn_qs8_requantization_fn requantize) {
    ASSERT_GE(zero_point(), std::numeric_limits<int8_t>::min());
    ASSERT_LE(zero_point(), std::numeric_limits<int8_t>::max());
    ASSERT_GE(qmin(), std::numeric_limits<int8_t>::min());
    ASSERT_LE(qmin(), std::numeric_limits<int8_t>::max());
    ASSERT_GE(qmax(), std::numeric_limits<int8_t>::min());
    ASSERT_LE(qmax(), std::numeric_limits<int8_t>::max());
    ASSERT_LT(qmin(), qmax());

    /* Note: need s >= 1 to ensure scale = exp2(-s) < 1.0 */
    ASSERT_GE(s(), 1);
    ASSERT_LT(s(), 32);

    std::vector<int32_t> inputs(256);
    std::vector<int8_t> outputs(inputs.size());
    for (int32_t i = std::numeric_limits<int8_t>::min(); i <= std::numeric_limits<int8_t>::max(); i++) {
      const int64_t input = RequantizationTester::ShiftLeft(i - zero_point(), s()) +
        (INT64_C(1) << (s() - 1)) - 1;
      inputs[i - std::numeric_limits<int8_t>::min()] = int32_t(input);
    }
    requantize(inputs.size(), inputs.data(),
        scale(), zero_point(), qmin(), qmax(),
        outputs.data());
    for (int32_t i = std::numeric_limits<int8_t>::min(); i <= std::numeric_limits<int8_t>::max(); i++) {
      const int64_t input = RequantizationTester::ShiftLeft(i - zero_point(), s()) +
        (INT64_C(1) << (s() - 1)) - 1;
      if (int32_t(input) == input) {
        ASSERT_EQ(i, int32_t(outputs[i - std::numeric_limits<int8_t>::min()]))
          << "i = " << i << ", input = " << input
          << ", s = " << s() << ", zero point = " << zero_point();
      }
    }
  }

  void TestDivideByPO2WithRoundingTiesAway(xnn_qu8_requantization_fn requantize) {
    ASSERT_GE(zero_point(), std::numeric_limits<uint8_t>::min());
    ASSERT_LE(zero_point(), std::numeric_limits<uint8_t>::max());
    ASSERT_GE(qmin(), std::numeric_limits<uint8_t>::min());
    ASSERT_LE(qmin(), std::numeric_limits<uint8_t>::max());
    ASSERT_GE(qmax(), std::numeric_limits<uint8_t>::min());
    ASSERT_LE(qmax(), std::numeric_limits<uint8_t>::max());
    ASSERT_LT(qmin(), qmax());

    /* Note: need s >= 1 to ensure scale = exp2(-s) < 1.0 */
    ASSERT_GE(s(), 1);
    ASSERT_LT(s(), 32);

    std::vector<int32_t> inputs(256);
    std::vector<uint8_t> outputs(inputs.size());
    for (int32_t i = 0; i <= std::numeric_limits<uint8_t>::max(); i++) {
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
    for (int32_t i = 0; i <= std::numeric_limits<uint8_t>::max(); i++) {
      int64_t input = RequantizationTester::ShiftLeft(i - zero_point(), s());
      if (input > 0) {
        input -= INT64_C(1) << (s() - 1);
      } else if (input < 0) {
        input += INT64_C(1) << (s() - 1);
      }
      if (int32_t(input) == input) {
        ASSERT_EQ(i, int32_t(outputs[i]))
          << "i = " << i << ", input = " << input
          << ", s = " << s() << ", zero point = " << zero_point();
      }
    }
  }

  void TestDivideByPO2WithRoundingTiesAway(xnn_qs8_requantization_fn requantize) {
    ASSERT_GE(zero_point(), std::numeric_limits<int8_t>::min());
    ASSERT_LE(zero_point(), std::numeric_limits<int8_t>::max());
    ASSERT_GE(qmin(), std::numeric_limits<int8_t>::min());
    ASSERT_LE(qmin(), std::numeric_limits<int8_t>::max());
    ASSERT_GE(qmax(), std::numeric_limits<int8_t>::min());
    ASSERT_LE(qmax(), std::numeric_limits<int8_t>::max());
    ASSERT_LT(qmin(), qmax());

    /* Note: need s >= 1 to ensure scale = exp2(-s) < 1.0 */
    ASSERT_GE(s(), 1);
    ASSERT_LT(s(), 32);

    std::vector<int32_t> inputs(256);
    std::vector<int8_t> outputs(inputs.size());
    for (int32_t i = std::numeric_limits<int8_t>::min(); i <= std::numeric_limits<int8_t>::max(); i++) {
      int64_t input = RequantizationTester::ShiftLeft(i - zero_point(), s());
      if (input > 0) {
        input -= INT64_C(1) << (s() - 1);
      } else if (input < 0) {
        input += INT64_C(1) << (s() - 1);
      }
      inputs[i - std::numeric_limits<int8_t>::min()] = int32_t(input);
    }
    requantize(inputs.size(), inputs.data(),
        scale(), zero_point(), qmin(), qmax(),
        outputs.data());
    for (int32_t i = std::numeric_limits<int8_t>::min(); i <= std::numeric_limits<int8_t>::max(); i++) {
      int64_t input = RequantizationTester::ShiftLeft(i - zero_point(), s());
      if (input > 0) {
        input -= INT64_C(1) << (s() - 1);
      } else if (input < 0) {
        input += INT64_C(1) << (s() - 1);
      }
      if (int32_t(input) == input) {
        ASSERT_EQ(i, int32_t(outputs[i - std::numeric_limits<int8_t>::min()]))
          << "i = " << i << ", input = " << input
          << ", s = " << s() << ", zero point = " << zero_point();
      }
    }
  }

  void TestDivideByPO2WithRoundingTiesUp(xnn_qs8_requantization_fn requantize) {
    ASSERT_GE(zero_point(), std::numeric_limits<int8_t>::min());
    ASSERT_LE(zero_point(), std::numeric_limits<int8_t>::max());
    ASSERT_GE(qmin(), std::numeric_limits<int8_t>::min());
    ASSERT_LE(qmin(), std::numeric_limits<int8_t>::max());
    ASSERT_GE(qmax(), std::numeric_limits<int8_t>::min());
    ASSERT_LE(qmax(), std::numeric_limits<int8_t>::max());
    ASSERT_LT(qmin(), qmax());

    /* Note: need s >= 1 to ensure scale = exp2(-s) < 1.0 */
    ASSERT_GE(s(), 1);
    ASSERT_LT(s(), 32);

    std::vector<int32_t> inputs(256);
    std::vector<int8_t> outputs(inputs.size());
    for (int32_t i = std::numeric_limits<int8_t>::min(); i <= std::numeric_limits<int8_t>::max(); i++) {
      int64_t input = RequantizationTester::ShiftLeft(i - zero_point(), s());
      input -= INT64_C(1) << (s() - 1);
      inputs[i - std::numeric_limits<int8_t>::min()] = int32_t(input);
    }
    requantize(inputs.size(), inputs.data(),
        scale(), zero_point(), qmin(), qmax(),
        outputs.data());
    for (int32_t i = std::numeric_limits<int8_t>::min(); i <= std::numeric_limits<int8_t>::max(); i++) {
      int64_t input = RequantizationTester::ShiftLeft(i - zero_point(), s());
      input -= INT64_C(1) << (s() - 1);
      if (int32_t(input) == input) {
        ASSERT_EQ(i, int32_t(outputs[i - std::numeric_limits<int8_t>::min()]))
          << "i = " << i << ", input = " << input
          << ", s = " << s() << ", zero point = " << zero_point();
      }
    }
  }

  void TestSpecialCases(xnn_qu8_requantization_fn requantize) {
    ASSERT_GE(qmin(), std::numeric_limits<uint8_t>::min());
    ASSERT_LE(qmin(), std::numeric_limits<uint8_t>::max());
    ASSERT_GE(qmax(), std::numeric_limits<uint8_t>::min());
    ASSERT_LE(qmax(), std::numeric_limits<uint8_t>::max());
    ASSERT_LT(qmin(), qmax());

    std::vector<int32_t> inputs(256);
    std::vector<uint8_t> outputs(inputs.size());

    std::fill(inputs.begin(), inputs.end(), std::numeric_limits<int32_t>::min());
    for (int32_t zero_point = 0; zero_point <= std::numeric_limits<uint8_t>::max(); zero_point++) {
      requantize(
          inputs.size(),
          inputs.data(),
          ldexpf(1.0f, -32) /* scale */,
          zero_point /* zero point */,
          std::numeric_limits<uint8_t>::min(),
          std::numeric_limits<uint8_t>::max(),
          outputs.data());
      for (size_t i = 0; i < outputs.size(); i++) {
        ASSERT_EQ(std::max(int32_t(int32_t(std::numeric_limits<uint8_t>::min())), zero_point - 1), int32_t(outputs[i]));
      }
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
    for (size_t i = 0; i < outputs.size(); i++) {
      ASSERT_EQ(std::numeric_limits<uint8_t>::max(), int32_t(outputs[i]));
    }
  }

  void TestSpecialCases(xnn_qs8_requantization_fn requantize) {
    ASSERT_GE(qmin(), std::numeric_limits<int8_t>::min());
    ASSERT_LE(qmin(), std::numeric_limits<int8_t>::max());
    ASSERT_GE(qmax(), std::numeric_limits<int8_t>::min());
    ASSERT_LE(qmax(), std::numeric_limits<int8_t>::max());
    ASSERT_LT(qmin(), qmax());

    std::vector<int32_t> inputs(256);
    std::vector<int8_t> outputs(inputs.size());

    std::fill(inputs.begin(), inputs.end(), std::numeric_limits<int32_t>::min());
    for (int32_t zero_point = std::numeric_limits<int8_t>::min();
         zero_point <= std::numeric_limits<int8_t>::max();
         zero_point++)
    {
      requantize(
          inputs.size(),
          inputs.data(),
          ldexpf(1.0f, -32) /* scale */,
          zero_point,
          std::numeric_limits<int8_t>::min(),
          std::numeric_limits<int8_t>::max(),
          outputs.data());
      for (size_t i = 0; i < outputs.size(); i++) {
        ASSERT_EQ(std::max(int32_t(std::numeric_limits<int8_t>::min()), zero_point - 1), int32_t(outputs[i]));
      }
    }

    std::fill(inputs.begin(), inputs.end(), std::numeric_limits<int32_t>::max());
    requantize(
        inputs.size(),
        inputs.data(),
        0x1.FFFFFEp-1f /* scale */,
        std::numeric_limits<int8_t>::max() /* zero point */,
        std::numeric_limits<int8_t>::min(),
        std::numeric_limits<int8_t>::max(),
        outputs.data());
    for (size_t i = 0; i < outputs.size(); i++) {
      ASSERT_EQ(std::numeric_limits<int8_t>::max(), int32_t(outputs[i]));
    }
  }

  void TestRandomCasesRoundToNearestTiesAway(xnn_qu8_requantization_fn requantize) {
    ASSERT_GE(zero_point(), std::numeric_limits<uint8_t>::min());
    ASSERT_LE(zero_point(), std::numeric_limits<uint8_t>::max());
    ASSERT_GE(qmin(), std::numeric_limits<uint8_t>::min());
    ASSERT_LE(qmin(), std::numeric_limits<uint8_t>::max());
    ASSERT_GE(qmax(), std::numeric_limits<uint8_t>::min());
    ASSERT_LE(qmax(), std::numeric_limits<uint8_t>::max());
    ASSERT_LT(qmin(), qmax());

    xnnpack::ReplicableRandomDevice rng;
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      auto u8rng =
        std::bind(std::uniform_int_distribution<uint32_t>(0, std::numeric_limits<uint8_t>::max()), std::ref(rng));

      std::vector<int32_t> inputs(4096);
      std::vector<uint8_t> outputs(inputs.size());

      std::uniform_real_distribution<float> scale_distribution(0x1.000000p-23f, 0x1.FFFFFEp-1f);
      const float scale = scale_distribution(rng);
      for (size_t i = 0; i < inputs.size(); i++) {
        const uint8_t approximate_output = std::min(std::max(uint8_t(u8rng()), uint8_t(qmin())), uint8_t(qmax()));
        const int32_t input = int32_t(double(approximate_output) / double(scale));
        inputs[i] = input;
      }

      requantize(
        inputs.size(), inputs.data(), scale, zero_point(), qmin(), qmax(),
        outputs.data());

      /* Ensure that outputs are not all identical, as in this case the test doesn't validate much */
      ASSERT_NE(
        *std::max_element(outputs.cbegin(), outputs.cend()),
        *std::min_element(outputs.cbegin(), outputs.cend()));

      for (size_t i = 0; i < inputs.size(); i++) {
        const uint8_t reference_output = xnn_qu8_requantize_rndna(
          inputs[i], scale, zero_point(), qmin(), qmax());
        ASSERT_EQ(uint32_t(reference_output), uint32_t(outputs[i]));
      }
    }
  }

  void TestRandomCasesRoundToNearestTiesAway(xnn_qs8_requantization_fn requantize) {
    ASSERT_GE(zero_point(), std::numeric_limits<int8_t>::min());
    ASSERT_LE(zero_point(), std::numeric_limits<int8_t>::max());
    ASSERT_GE(qmin(), std::numeric_limits<int8_t>::min());
    ASSERT_LE(qmin(), std::numeric_limits<int8_t>::max());
    ASSERT_GE(qmax(), std::numeric_limits<int8_t>::min());
    ASSERT_LE(qmax(), std::numeric_limits<int8_t>::max());
    ASSERT_LT(qmin(), qmax());

    xnnpack::ReplicableRandomDevice rng;
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      auto i8rng = std::bind(
        std::uniform_int_distribution<int32_t>(std::numeric_limits<int8_t>::min(), std::numeric_limits<int8_t>::max()), std::ref(rng));

      std::vector<int32_t> inputs(4096);
      std::vector<int8_t> outputs(inputs.size());

      std::uniform_real_distribution<float> scale_distribution(0x1.000000p-23f, 0x1.FFFFFEp-1f);
      const float scale = scale_distribution(rng);
      for (size_t i = 0; i < inputs.size(); i++) {
        const int8_t approximate_output = std::min(std::max(int8_t(i8rng()), int8_t(qmin())), int8_t(qmax()));
        const int32_t input = int32_t(double(approximate_output) / double(scale));
        inputs[i] = input;
      }

      requantize(
        inputs.size(), inputs.data(), scale, zero_point(), qmin(), qmax(),
        outputs.data());

      /* Ensure that outputs are not all identical, as in this case the test doesn't validate much */
      ASSERT_NE(
        *std::max_element(outputs.cbegin(), outputs.cend()),
        *std::min_element(outputs.cbegin(), outputs.cend()));

      for (size_t i = 0; i < inputs.size(); i++) {
        const int8_t reference_output = xnn_qs8_requantize_rndna(
          inputs[i], scale, zero_point(), qmin(), qmax());
        ASSERT_EQ(int32_t(reference_output), int32_t(outputs[i]));
      }
    }
  }

  void TestRandomCasesRoundToNearestTiesUp(xnn_qs8_requantization_fn requantize) {
    ASSERT_GE(zero_point(), std::numeric_limits<int8_t>::min());
    ASSERT_LE(zero_point(), std::numeric_limits<int8_t>::max());
    ASSERT_GE(qmin(), std::numeric_limits<int8_t>::min());
    ASSERT_LE(qmin(), std::numeric_limits<int8_t>::max());
    ASSERT_GE(qmax(), std::numeric_limits<int8_t>::min());
    ASSERT_LE(qmax(), std::numeric_limits<int8_t>::max());
    ASSERT_LT(qmin(), qmax());

    xnnpack::ReplicableRandomDevice rng;
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      auto i8rng = std::bind(
        std::uniform_int_distribution<int32_t>(std::numeric_limits<int8_t>::min(), std::numeric_limits<int8_t>::max()), std::ref(rng));

      std::vector<int32_t> inputs(4096);
      std::vector<int8_t> outputs(inputs.size());

      std::uniform_real_distribution<float> scale_distribution(0x1.000000p-23f, 0x1.FFFFFEp-1f);
      const float scale = scale_distribution(rng);
      for (size_t i = 0; i < inputs.size(); i++) {
        const int8_t approximate_output = std::min(std::max(int8_t(i8rng()), int8_t(qmin())), int8_t(qmax()));
        const int32_t input = int32_t(double(approximate_output) / double(scale));
        inputs[i] = input;
      }

      requantize(
        inputs.size(), inputs.data(), scale, zero_point(), qmin(), qmax(),
        outputs.data());

      /* Ensure that outputs are not all identical, as in this case the test doesn't validate much */
      ASSERT_NE(
        *std::max_element(outputs.cbegin(), outputs.cend()),
        *std::min_element(outputs.cbegin(), outputs.cend()));

      for (size_t i = 0; i < inputs.size(); i++) {
        const int8_t reference_output = xnn_qs8_requantize_rndnu(
          inputs[i], scale, zero_point(), qmin(), qmax());
        ASSERT_EQ(int32_t(reference_output), int32_t(outputs[i]));
      }
    }
  }

  void TestRandomCasesApproximate(xnn_qu8_requantization_fn requantize) {
    ASSERT_GE(zero_point(), std::numeric_limits<uint8_t>::min());
    ASSERT_LE(zero_point(), std::numeric_limits<uint8_t>::max());
    ASSERT_GE(qmin(), std::numeric_limits<uint8_t>::min());
    ASSERT_LE(qmin(), std::numeric_limits<uint8_t>::max());
    ASSERT_GE(qmax(), std::numeric_limits<uint8_t>::min());
    ASSERT_LE(qmax(), std::numeric_limits<uint8_t>::max());
    ASSERT_LT(qmin(), qmax());

    xnnpack::ReplicableRandomDevice rng;
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      auto u8rng =
        std::bind(std::uniform_int_distribution<uint32_t>(0, std::numeric_limits<uint8_t>::max()), std::ref(rng));

      std::vector<int32_t> inputs(4096);
      std::vector<uint8_t> outputs(inputs.size());

      std::uniform_real_distribution<float> scale_distribution(0x1.000000p-23f, 0x1.FFFFFEp-1f);
      const float scale = scale_distribution(rng);
      for (size_t i = 0; i < inputs.size(); i++) {
        const uint8_t approximate_output = std::min(std::max(uint8_t(u8rng()), uint8_t(qmin())), uint8_t(qmax()));
        const int32_t input = int32_t(double(approximate_output) / double(scale));
        inputs[i] = input;
      }

      requantize(
        inputs.size(), inputs.data(), scale, zero_point(), qmin(), qmax(),
        outputs.data());

      /* Ensure that outputs are not all identical, as in this case Test doesn't validate much */
      ASSERT_NE(
        *std::max_element(outputs.cbegin(), outputs.cend()),
        *std::min_element(outputs.cbegin(), outputs.cend()));

      for (size_t i = 0; i < inputs.size(); i++) {
        const double reference_output = RequantizationTester::RequantizeApproximate(
          inputs[i], scale, uint8_t(zero_point()), uint8_t(qmin()), uint8_t(qmax()));
        ASSERT_LE(std::abs(reference_output - double(outputs[i])), 0.55)
          << "input = " << inputs[i] << ", output = " << int32_t(outputs[i])
          << ", reference output = " << reference_output;
      }
    }
  }

  void TestRandomCasesApproximate(xnn_qs8_requantization_fn requantize) {
    ASSERT_GE(zero_point(), std::numeric_limits<int8_t>::min());
    ASSERT_LE(zero_point(), std::numeric_limits<int8_t>::max());
    ASSERT_GE(qmin(), std::numeric_limits<int8_t>::min());
    ASSERT_LE(qmin(), std::numeric_limits<int8_t>::max());
    ASSERT_GE(qmax(), std::numeric_limits<int8_t>::min());
    ASSERT_LE(qmax(), std::numeric_limits<int8_t>::max());
    ASSERT_LT(qmin(), qmax());

    xnnpack::ReplicableRandomDevice rng;
    for (size_t iteration = 0; iteration < iterations(); iteration++) {
      auto i8rng = std::bind(
        std::uniform_int_distribution<int32_t>(std::numeric_limits<int8_t>::min(), std::numeric_limits<int8_t>::max()), std::ref(rng));

      std::vector<int32_t> inputs(4096);
      std::vector<int8_t> outputs(inputs.size());

      std::uniform_real_distribution<float> scale_distribution(0x1.000000p-23f, 0x1.FFFFFEp-1f);
      const float scale = scale_distribution(rng);
      for (size_t i = 0; i < inputs.size(); i++) {
        const int8_t approximate_output = std::min(std::max(int8_t(i8rng()), int8_t(qmin())), int8_t(qmax()));
        const int32_t input = int32_t(double(approximate_output) / double(scale));
        inputs[i] = input;
      }

      requantize(
        inputs.size(), inputs.data(), scale, zero_point(), qmin(), qmax(),
        outputs.data());

      /* Ensure that outputs are not all identical, as in this case Test doesn't validate much */
      ASSERT_NE(
        *std::max_element(outputs.cbegin(), outputs.cend()),
        *std::min_element(outputs.cbegin(), outputs.cend()));

      for (size_t i = 0; i < inputs.size(); i++) {
        const double reference_output = RequantizationTester::RequantizeApproximate(
          inputs[i], scale, int8_t(zero_point()), int8_t(qmin()), int8_t(qmax()));
        ASSERT_LE(std::abs(reference_output - double(outputs[i])), 0.55)
          << "input = " << inputs[i] << ", output = " << int32_t(outputs[i])
          << ", reference output = " << reference_output;
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

    return std::min(std::max(double(value) * double(scale) + double(zero_point), double(qmin)), double(qmax));
  }

  static inline double RequantizeApproximate(
    int32_t value,
    float scale,
    int8_t zero_point,
    int8_t qmin,
    int8_t qmax)
  {
    assert(scale < 1.0f);
    assert(scale >= 0x1.0p-32f);

    return std::min(std::max(double(value) * double(scale) + double(zero_point), double(qmin)), double(qmax));
  }

 private:
  uint32_t s_{1};
  int32_t zero_point_{0};
  int16_t qmin_{std::numeric_limits<int16_t>::min()};
  int16_t qmax_{std::numeric_limits<int16_t>::max()};
  size_t iterations_{1};
};
