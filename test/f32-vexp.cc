// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: test/f32-vexp.yaml
//   Generator: tools/generate-vunary-test.py


#include <array>
#include <cmath>
#include <cstdint>
#include <cstddef>
#include <limits>

#include <gtest/gtest.h>
#include "xnnpack.h"
#include "xnnpack/common.h"
#include "xnnpack/isa-checks.h"
#include "xnnpack/microparams-init.h"
#include "xnnpack/microparams.h"
#include "xnnpack/vunary.h"
#include "next_prime.h"
#include "vunary-microkernel-tester.h"


TEST(F32_VEXP__SCALAR_EXP_U1, batch_eq_1) {
  VUnaryMicrokernelTester()
    .batch_size(1)
    .TestExp(xnn_f32_vexp_ukernel__scalar_exp_u1);
}

TEST(F32_VEXP__SCALAR_EXP_U1, batch_gt_1) {
  const size_t batch_step = 1;
  for (size_t batch_size = batch_step + 1; batch_size < 10; batch_size++) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .TestExp(xnn_f32_vexp_ukernel__scalar_exp_u1);
  }
}

TEST(F32_VEXP__SCALAR_EXP_U1, inplace) {
  const size_t batch_step = 1;
  for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 1) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .inplace(true)
      .TestExp(xnn_f32_vexp_ukernel__scalar_exp_u1);
  }
}

TEST(F32_VEXP__SCALAR_EXP_U1, special_values) {
  constexpr size_t num_elements = 3;
  constexpr size_t buffered_size =
      num_elements + XNN_EXTRA_BYTES / sizeof(float);
  std::array<float, buffered_size> inputs =
      {0.0f, -1e3f, 1e3f};
  std::array<float, num_elements> expected =
      {1.0f, 0.0f, INFINITY};
  std::array<float, buffered_size> outputs;
  xnn_f32_vexp_ukernel__scalar_exp_u1(
      num_elements * sizeof(float), inputs.data(), outputs.data(), nullptr);
  for (int i = 0; i < num_elements; i++) {
    if (std::isfinite(expected[i])) {
      EXPECT_NEAR(
          expected[i], outputs[i],
          1 * std::abs(expected[i]) * std::numeric_limits<float>::epsilon())
          << "for input " << inputs[i];
    } else {
      EXPECT_EQ(std::fpclassify(expected[i]), std::fpclassify(outputs[i]))
          << "for input " << inputs[i] << " and output " << outputs[i]
          << " (FP_INFINITE=" << FP_INFINITE << ", FP_NAN=" << FP_NAN
          << ", FP_NORMAL=" << FP_NORMAL << ", FP_SUBNORMAL=" << FP_SUBNORMAL
          << ", FP_ZERO=" << FP_ZERO << ")";
    }
  }
}

TEST(F32_VEXP__SCALAR_EXP_U2, batch_eq_2) {
  VUnaryMicrokernelTester()
    .batch_size(2)
    .TestExp(xnn_f32_vexp_ukernel__scalar_exp_u2);
}

TEST(F32_VEXP__SCALAR_EXP_U2, batch_div_2) {
  const size_t batch_step = 2;
  for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .TestExp(xnn_f32_vexp_ukernel__scalar_exp_u2);
  }
}

TEST(F32_VEXP__SCALAR_EXP_U2, batch_lt_2) {
  const size_t batch_step = 2;
  for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .TestExp(xnn_f32_vexp_ukernel__scalar_exp_u2);
  }
}

TEST(F32_VEXP__SCALAR_EXP_U2, batch_gt_2) {
  const size_t batch_step = 2;
  for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .TestExp(xnn_f32_vexp_ukernel__scalar_exp_u2);
  }
}

TEST(F32_VEXP__SCALAR_EXP_U2, inplace) {
  const size_t batch_step = 2;
  for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 1) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .inplace(true)
      .TestExp(xnn_f32_vexp_ukernel__scalar_exp_u2);
  }
}

TEST(F32_VEXP__SCALAR_EXP_U2, special_values) {
  constexpr size_t num_elements = 3;
  constexpr size_t buffered_size =
      num_elements + XNN_EXTRA_BYTES / sizeof(float);
  std::array<float, buffered_size> inputs =
      {0.0f, -1e3f, 1e3f};
  std::array<float, num_elements> expected =
      {1.0f, 0.0f, INFINITY};
  std::array<float, buffered_size> outputs;
  xnn_f32_vexp_ukernel__scalar_exp_u2(
      num_elements * sizeof(float), inputs.data(), outputs.data(), nullptr);
  for (int i = 0; i < num_elements; i++) {
    if (std::isfinite(expected[i])) {
      EXPECT_NEAR(
          expected[i], outputs[i],
          1 * std::abs(expected[i]) * std::numeric_limits<float>::epsilon())
          << "for input " << inputs[i];
    } else {
      EXPECT_EQ(std::fpclassify(expected[i]), std::fpclassify(outputs[i]))
          << "for input " << inputs[i] << " and output " << outputs[i]
          << " (FP_INFINITE=" << FP_INFINITE << ", FP_NAN=" << FP_NAN
          << ", FP_NORMAL=" << FP_NORMAL << ", FP_SUBNORMAL=" << FP_SUBNORMAL
          << ", FP_ZERO=" << FP_ZERO << ")";
    }
  }
}

TEST(F32_VEXP__SCALAR_EXP_U4, batch_eq_4) {
  VUnaryMicrokernelTester()
    .batch_size(4)
    .TestExp(xnn_f32_vexp_ukernel__scalar_exp_u4);
}

TEST(F32_VEXP__SCALAR_EXP_U4, batch_div_4) {
  const size_t batch_step = 4;
  for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .TestExp(xnn_f32_vexp_ukernel__scalar_exp_u4);
  }
}

TEST(F32_VEXP__SCALAR_EXP_U4, batch_lt_4) {
  const size_t batch_step = 4;
  for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .TestExp(xnn_f32_vexp_ukernel__scalar_exp_u4);
  }
}

TEST(F32_VEXP__SCALAR_EXP_U4, batch_gt_4) {
  const size_t batch_step = 4;
  for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .TestExp(xnn_f32_vexp_ukernel__scalar_exp_u4);
  }
}

TEST(F32_VEXP__SCALAR_EXP_U4, inplace) {
  const size_t batch_step = 4;
  for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 3) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .inplace(true)
      .TestExp(xnn_f32_vexp_ukernel__scalar_exp_u4);
  }
}

TEST(F32_VEXP__SCALAR_EXP_U4, special_values) {
  constexpr size_t num_elements = 3;
  constexpr size_t buffered_size =
      num_elements + XNN_EXTRA_BYTES / sizeof(float);
  std::array<float, buffered_size> inputs =
      {0.0f, -1e3f, 1e3f};
  std::array<float, num_elements> expected =
      {1.0f, 0.0f, INFINITY};
  std::array<float, buffered_size> outputs;
  xnn_f32_vexp_ukernel__scalar_exp_u4(
      num_elements * sizeof(float), inputs.data(), outputs.data(), nullptr);
  for (int i = 0; i < num_elements; i++) {
    if (std::isfinite(expected[i])) {
      EXPECT_NEAR(
          expected[i], outputs[i],
          1 * std::abs(expected[i]) * std::numeric_limits<float>::epsilon())
          << "for input " << inputs[i];
    } else {
      EXPECT_EQ(std::fpclassify(expected[i]), std::fpclassify(outputs[i]))
          << "for input " << inputs[i] << " and output " << outputs[i]
          << " (FP_INFINITE=" << FP_INFINITE << ", FP_NAN=" << FP_NAN
          << ", FP_NORMAL=" << FP_NORMAL << ", FP_SUBNORMAL=" << FP_SUBNORMAL
          << ", FP_ZERO=" << FP_ZERO << ")";
    }
  }
}