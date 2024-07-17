// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: test/f32-vlog.yaml
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


TEST(F32_VLOG__SCALAR_LOG_U1, batch_eq_1) {
  VUnaryMicrokernelTester()
    .batch_size(1)
    .TestLog(xnn_f32_vlog_ukernel__scalar_log_u1);
}

TEST(F32_VLOG__SCALAR_LOG_U1, batch_gt_1) {
  const size_t batch_step = 1;
  for (size_t batch_size = batch_step + 1; batch_size < 10; batch_size++) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .TestLog(xnn_f32_vlog_ukernel__scalar_log_u1);
  }
}

TEST(F32_VLOG__SCALAR_LOG_U1, inplace) {
  const size_t batch_step = 1;
  for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 1) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .inplace(true)
      .TestLog(xnn_f32_vlog_ukernel__scalar_log_u1);
  }
}

TEST(F32_VLOG__SCALAR_LOG_U1, special_values) {
  constexpr size_t num_elements = 4;
  constexpr size_t buffered_size =
      num_elements + XNN_EXTRA_BYTES / sizeof(float);
  std::array<float, buffered_size> inputs =
      {1.0f, -1.0f, 0.0f, -0.0f};
  std::array<float, num_elements> expected =
      {0.0f, NAN, -INFINITY, -INFINITY};
  std::array<float, buffered_size> outputs;
  xnn_f32_vlog_ukernel__scalar_log_u1(
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

TEST(F32_VLOG__SCALAR_LOG_U2, batch_eq_2) {
  VUnaryMicrokernelTester()
    .batch_size(2)
    .TestLog(xnn_f32_vlog_ukernel__scalar_log_u2);
}

TEST(F32_VLOG__SCALAR_LOG_U2, batch_div_2) {
  const size_t batch_step = 2;
  for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .TestLog(xnn_f32_vlog_ukernel__scalar_log_u2);
  }
}

TEST(F32_VLOG__SCALAR_LOG_U2, batch_lt_2) {
  const size_t batch_step = 2;
  for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .TestLog(xnn_f32_vlog_ukernel__scalar_log_u2);
  }
}

TEST(F32_VLOG__SCALAR_LOG_U2, batch_gt_2) {
  const size_t batch_step = 2;
  for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .TestLog(xnn_f32_vlog_ukernel__scalar_log_u2);
  }
}

TEST(F32_VLOG__SCALAR_LOG_U2, inplace) {
  const size_t batch_step = 2;
  for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 1) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .inplace(true)
      .TestLog(xnn_f32_vlog_ukernel__scalar_log_u2);
  }
}

TEST(F32_VLOG__SCALAR_LOG_U2, special_values) {
  constexpr size_t num_elements = 4;
  constexpr size_t buffered_size =
      num_elements + XNN_EXTRA_BYTES / sizeof(float);
  std::array<float, buffered_size> inputs =
      {1.0f, -1.0f, 0.0f, -0.0f};
  std::array<float, num_elements> expected =
      {0.0f, NAN, -INFINITY, -INFINITY};
  std::array<float, buffered_size> outputs;
  xnn_f32_vlog_ukernel__scalar_log_u2(
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

TEST(F32_VLOG__SCALAR_LOG_U4, batch_eq_4) {
  VUnaryMicrokernelTester()
    .batch_size(4)
    .TestLog(xnn_f32_vlog_ukernel__scalar_log_u4);
}

TEST(F32_VLOG__SCALAR_LOG_U4, batch_div_4) {
  const size_t batch_step = 4;
  for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .TestLog(xnn_f32_vlog_ukernel__scalar_log_u4);
  }
}

TEST(F32_VLOG__SCALAR_LOG_U4, batch_lt_4) {
  const size_t batch_step = 4;
  for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .TestLog(xnn_f32_vlog_ukernel__scalar_log_u4);
  }
}

TEST(F32_VLOG__SCALAR_LOG_U4, batch_gt_4) {
  const size_t batch_step = 4;
  for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .TestLog(xnn_f32_vlog_ukernel__scalar_log_u4);
  }
}

TEST(F32_VLOG__SCALAR_LOG_U4, inplace) {
  const size_t batch_step = 4;
  for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 3) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .inplace(true)
      .TestLog(xnn_f32_vlog_ukernel__scalar_log_u4);
  }
}

TEST(F32_VLOG__SCALAR_LOG_U4, special_values) {
  constexpr size_t num_elements = 4;
  constexpr size_t buffered_size =
      num_elements + XNN_EXTRA_BYTES / sizeof(float);
  std::array<float, buffered_size> inputs =
      {1.0f, -1.0f, 0.0f, -0.0f};
  std::array<float, num_elements> expected =
      {0.0f, NAN, -INFINITY, -INFINITY};
  std::array<float, buffered_size> outputs;
  xnn_f32_vlog_ukernel__scalar_log_u4(
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

TEST(F32_VLOG__SCALAR_RATIONAL_3_3_DIV_U1, batch_eq_1) {
  VUnaryMicrokernelTester()
    .batch_size(1)
    .TestLog(xnn_f32_vlog_ukernel__scalar_rational_3_3_div_u1);
}

TEST(F32_VLOG__SCALAR_RATIONAL_3_3_DIV_U1, batch_gt_1) {
  const size_t batch_step = 1;
  for (size_t batch_size = batch_step + 1; batch_size < 10; batch_size++) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .TestLog(xnn_f32_vlog_ukernel__scalar_rational_3_3_div_u1);
  }
}

TEST(F32_VLOG__SCALAR_RATIONAL_3_3_DIV_U1, inplace) {
  const size_t batch_step = 1;
  for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 1) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .inplace(true)
      .TestLog(xnn_f32_vlog_ukernel__scalar_rational_3_3_div_u1);
  }
}

TEST(F32_VLOG__SCALAR_RATIONAL_3_3_DIV_U1, special_values) {
  constexpr size_t num_elements = 4;
  constexpr size_t buffered_size =
      num_elements + XNN_EXTRA_BYTES / sizeof(float);
  std::array<float, buffered_size> inputs =
      {1.0f, -1.0f, 0.0f, -0.0f};
  std::array<float, num_elements> expected =
      {0.0f, NAN, -INFINITY, -INFINITY};
  std::array<float, buffered_size> outputs;
  xnn_f32_vlog_ukernel__scalar_rational_3_3_div_u1(
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

TEST(F32_VLOG__SCALAR_RATIONAL_3_3_DIV_U2, batch_eq_2) {
  VUnaryMicrokernelTester()
    .batch_size(2)
    .TestLog(xnn_f32_vlog_ukernel__scalar_rational_3_3_div_u2);
}

TEST(F32_VLOG__SCALAR_RATIONAL_3_3_DIV_U2, batch_div_2) {
  const size_t batch_step = 2;
  for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .TestLog(xnn_f32_vlog_ukernel__scalar_rational_3_3_div_u2);
  }
}

TEST(F32_VLOG__SCALAR_RATIONAL_3_3_DIV_U2, batch_lt_2) {
  const size_t batch_step = 2;
  for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .TestLog(xnn_f32_vlog_ukernel__scalar_rational_3_3_div_u2);
  }
}

TEST(F32_VLOG__SCALAR_RATIONAL_3_3_DIV_U2, batch_gt_2) {
  const size_t batch_step = 2;
  for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .TestLog(xnn_f32_vlog_ukernel__scalar_rational_3_3_div_u2);
  }
}

TEST(F32_VLOG__SCALAR_RATIONAL_3_3_DIV_U2, inplace) {
  const size_t batch_step = 2;
  for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 1) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .inplace(true)
      .TestLog(xnn_f32_vlog_ukernel__scalar_rational_3_3_div_u2);
  }
}

TEST(F32_VLOG__SCALAR_RATIONAL_3_3_DIV_U2, special_values) {
  constexpr size_t num_elements = 4;
  constexpr size_t buffered_size =
      num_elements + XNN_EXTRA_BYTES / sizeof(float);
  std::array<float, buffered_size> inputs =
      {1.0f, -1.0f, 0.0f, -0.0f};
  std::array<float, num_elements> expected =
      {0.0f, NAN, -INFINITY, -INFINITY};
  std::array<float, buffered_size> outputs;
  xnn_f32_vlog_ukernel__scalar_rational_3_3_div_u2(
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

TEST(F32_VLOG__SCALAR_RATIONAL_3_3_DIV_U4, batch_eq_4) {
  VUnaryMicrokernelTester()
    .batch_size(4)
    .TestLog(xnn_f32_vlog_ukernel__scalar_rational_3_3_div_u4);
}

TEST(F32_VLOG__SCALAR_RATIONAL_3_3_DIV_U4, batch_div_4) {
  const size_t batch_step = 4;
  for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .TestLog(xnn_f32_vlog_ukernel__scalar_rational_3_3_div_u4);
  }
}

TEST(F32_VLOG__SCALAR_RATIONAL_3_3_DIV_U4, batch_lt_4) {
  const size_t batch_step = 4;
  for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .TestLog(xnn_f32_vlog_ukernel__scalar_rational_3_3_div_u4);
  }
}

TEST(F32_VLOG__SCALAR_RATIONAL_3_3_DIV_U4, batch_gt_4) {
  const size_t batch_step = 4;
  for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .TestLog(xnn_f32_vlog_ukernel__scalar_rational_3_3_div_u4);
  }
}

TEST(F32_VLOG__SCALAR_RATIONAL_3_3_DIV_U4, inplace) {
  const size_t batch_step = 4;
  for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 3) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .inplace(true)
      .TestLog(xnn_f32_vlog_ukernel__scalar_rational_3_3_div_u4);
  }
}

TEST(F32_VLOG__SCALAR_RATIONAL_3_3_DIV_U4, special_values) {
  constexpr size_t num_elements = 4;
  constexpr size_t buffered_size =
      num_elements + XNN_EXTRA_BYTES / sizeof(float);
  std::array<float, buffered_size> inputs =
      {1.0f, -1.0f, 0.0f, -0.0f};
  std::array<float, num_elements> expected =
      {0.0f, NAN, -INFINITY, -INFINITY};
  std::array<float, buffered_size> outputs;
  xnn_f32_vlog_ukernel__scalar_rational_3_3_div_u4(
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

TEST(F32_VLOG__SCALAR_RATIONAL_3_3_DIV_U8, batch_eq_8) {
  VUnaryMicrokernelTester()
    .batch_size(8)
    .TestLog(xnn_f32_vlog_ukernel__scalar_rational_3_3_div_u8);
}

TEST(F32_VLOG__SCALAR_RATIONAL_3_3_DIV_U8, batch_div_8) {
  const size_t batch_step = 8;
  for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .TestLog(xnn_f32_vlog_ukernel__scalar_rational_3_3_div_u8);
  }
}

TEST(F32_VLOG__SCALAR_RATIONAL_3_3_DIV_U8, batch_lt_8) {
  const size_t batch_step = 8;
  for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .TestLog(xnn_f32_vlog_ukernel__scalar_rational_3_3_div_u8);
  }
}

TEST(F32_VLOG__SCALAR_RATIONAL_3_3_DIV_U8, batch_gt_8) {
  const size_t batch_step = 8;
  for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .TestLog(xnn_f32_vlog_ukernel__scalar_rational_3_3_div_u8);
  }
}

TEST(F32_VLOG__SCALAR_RATIONAL_3_3_DIV_U8, inplace) {
  const size_t batch_step = 8;
  for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 7) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .inplace(true)
      .TestLog(xnn_f32_vlog_ukernel__scalar_rational_3_3_div_u8);
  }
}

TEST(F32_VLOG__SCALAR_RATIONAL_3_3_DIV_U8, special_values) {
  constexpr size_t num_elements = 4;
  constexpr size_t buffered_size =
      num_elements + XNN_EXTRA_BYTES / sizeof(float);
  std::array<float, buffered_size> inputs =
      {1.0f, -1.0f, 0.0f, -0.0f};
  std::array<float, num_elements> expected =
      {0.0f, NAN, -INFINITY, -INFINITY};
  std::array<float, buffered_size> outputs;
  xnn_f32_vlog_ukernel__scalar_rational_3_3_div_u8(
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

#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VLOG__SSE2_RATIONAL_3_3_DIV_U4, batch_eq_4) {
    TEST_REQUIRES_X86_SSE2;
    VUnaryMicrokernelTester()
      .batch_size(4)
      .TestLog(xnn_f32_vlog_ukernel__sse2_rational_3_3_div_u4);
  }

  TEST(F32_VLOG__SSE2_RATIONAL_3_3_DIV_U4, batch_div_4) {
    TEST_REQUIRES_X86_SSE2;
    const size_t batch_step = 4;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .TestLog(xnn_f32_vlog_ukernel__sse2_rational_3_3_div_u4);
    }
  }

  TEST(F32_VLOG__SSE2_RATIONAL_3_3_DIV_U4, batch_lt_4) {
    TEST_REQUIRES_X86_SSE2;
    const size_t batch_step = 4;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .TestLog(xnn_f32_vlog_ukernel__sse2_rational_3_3_div_u4);
    }
  }

  TEST(F32_VLOG__SSE2_RATIONAL_3_3_DIV_U4, batch_gt_4) {
    TEST_REQUIRES_X86_SSE2;
    const size_t batch_step = 4;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .TestLog(xnn_f32_vlog_ukernel__sse2_rational_3_3_div_u4);
    }
  }

  TEST(F32_VLOG__SSE2_RATIONAL_3_3_DIV_U4, inplace) {
    TEST_REQUIRES_X86_SSE2;
    const size_t batch_step = 4;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 3) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .TestLog(xnn_f32_vlog_ukernel__sse2_rational_3_3_div_u4);
    }
  }

  TEST(F32_VLOG__SSE2_RATIONAL_3_3_DIV_U4, special_values) {
    TEST_REQUIRES_X86_SSE2;
    constexpr size_t num_elements = 4;
    constexpr size_t buffered_size =
        num_elements + XNN_EXTRA_BYTES / sizeof(float);
    std::array<float, buffered_size> inputs =
        {1.0f, -1.0f, 0.0f, -0.0f};
    std::array<float, num_elements> expected =
        {0.0f, NAN, -INFINITY, -INFINITY};
    std::array<float, buffered_size> outputs;
    xnn_f32_vlog_ukernel__sse2_rational_3_3_div_u4(
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
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VLOG__SSE2_RATIONAL_3_3_DIV_U8, batch_eq_8) {
    TEST_REQUIRES_X86_SSE2;
    VUnaryMicrokernelTester()
      .batch_size(8)
      .TestLog(xnn_f32_vlog_ukernel__sse2_rational_3_3_div_u8);
  }

  TEST(F32_VLOG__SSE2_RATIONAL_3_3_DIV_U8, batch_div_8) {
    TEST_REQUIRES_X86_SSE2;
    const size_t batch_step = 8;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .TestLog(xnn_f32_vlog_ukernel__sse2_rational_3_3_div_u8);
    }
  }

  TEST(F32_VLOG__SSE2_RATIONAL_3_3_DIV_U8, batch_lt_8) {
    TEST_REQUIRES_X86_SSE2;
    const size_t batch_step = 8;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .TestLog(xnn_f32_vlog_ukernel__sse2_rational_3_3_div_u8);
    }
  }

  TEST(F32_VLOG__SSE2_RATIONAL_3_3_DIV_U8, batch_gt_8) {
    TEST_REQUIRES_X86_SSE2;
    const size_t batch_step = 8;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .TestLog(xnn_f32_vlog_ukernel__sse2_rational_3_3_div_u8);
    }
  }

  TEST(F32_VLOG__SSE2_RATIONAL_3_3_DIV_U8, inplace) {
    TEST_REQUIRES_X86_SSE2;
    const size_t batch_step = 8;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .TestLog(xnn_f32_vlog_ukernel__sse2_rational_3_3_div_u8);
    }
  }

  TEST(F32_VLOG__SSE2_RATIONAL_3_3_DIV_U8, special_values) {
    TEST_REQUIRES_X86_SSE2;
    constexpr size_t num_elements = 4;
    constexpr size_t buffered_size =
        num_elements + XNN_EXTRA_BYTES / sizeof(float);
    std::array<float, buffered_size> inputs =
        {1.0f, -1.0f, 0.0f, -0.0f};
    std::array<float, num_elements> expected =
        {0.0f, NAN, -INFINITY, -INFINITY};
    std::array<float, buffered_size> outputs;
    xnn_f32_vlog_ukernel__sse2_rational_3_3_div_u8(
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
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VLOG__SSE2_RATIONAL_3_3_DIV_U12, batch_eq_12) {
    TEST_REQUIRES_X86_SSE2;
    VUnaryMicrokernelTester()
      .batch_size(12)
      .TestLog(xnn_f32_vlog_ukernel__sse2_rational_3_3_div_u12);
  }

  TEST(F32_VLOG__SSE2_RATIONAL_3_3_DIV_U12, batch_div_12) {
    TEST_REQUIRES_X86_SSE2;
    const size_t batch_step = 12;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .TestLog(xnn_f32_vlog_ukernel__sse2_rational_3_3_div_u12);
    }
  }

  TEST(F32_VLOG__SSE2_RATIONAL_3_3_DIV_U12, batch_lt_12) {
    TEST_REQUIRES_X86_SSE2;
    const size_t batch_step = 12;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .TestLog(xnn_f32_vlog_ukernel__sse2_rational_3_3_div_u12);
    }
  }

  TEST(F32_VLOG__SSE2_RATIONAL_3_3_DIV_U12, batch_gt_12) {
    TEST_REQUIRES_X86_SSE2;
    const size_t batch_step = 12;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .TestLog(xnn_f32_vlog_ukernel__sse2_rational_3_3_div_u12);
    }
  }

  TEST(F32_VLOG__SSE2_RATIONAL_3_3_DIV_U12, inplace) {
    TEST_REQUIRES_X86_SSE2;
    const size_t batch_step = 12;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 11) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .TestLog(xnn_f32_vlog_ukernel__sse2_rational_3_3_div_u12);
    }
  }

  TEST(F32_VLOG__SSE2_RATIONAL_3_3_DIV_U12, special_values) {
    TEST_REQUIRES_X86_SSE2;
    constexpr size_t num_elements = 4;
    constexpr size_t buffered_size =
        num_elements + XNN_EXTRA_BYTES / sizeof(float);
    std::array<float, buffered_size> inputs =
        {1.0f, -1.0f, 0.0f, -0.0f};
    std::array<float, num_elements> expected =
        {0.0f, NAN, -INFINITY, -INFINITY};
    std::array<float, buffered_size> outputs;
    xnn_f32_vlog_ukernel__sse2_rational_3_3_div_u12(
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
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VLOG__SSE2_RATIONAL_3_3_DIV_U16, batch_eq_16) {
    TEST_REQUIRES_X86_SSE2;
    VUnaryMicrokernelTester()
      .batch_size(16)
      .TestLog(xnn_f32_vlog_ukernel__sse2_rational_3_3_div_u16);
  }

  TEST(F32_VLOG__SSE2_RATIONAL_3_3_DIV_U16, batch_div_16) {
    TEST_REQUIRES_X86_SSE2;
    const size_t batch_step = 16;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .TestLog(xnn_f32_vlog_ukernel__sse2_rational_3_3_div_u16);
    }
  }

  TEST(F32_VLOG__SSE2_RATIONAL_3_3_DIV_U16, batch_lt_16) {
    TEST_REQUIRES_X86_SSE2;
    const size_t batch_step = 16;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .TestLog(xnn_f32_vlog_ukernel__sse2_rational_3_3_div_u16);
    }
  }

  TEST(F32_VLOG__SSE2_RATIONAL_3_3_DIV_U16, batch_gt_16) {
    TEST_REQUIRES_X86_SSE2;
    const size_t batch_step = 16;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .TestLog(xnn_f32_vlog_ukernel__sse2_rational_3_3_div_u16);
    }
  }

  TEST(F32_VLOG__SSE2_RATIONAL_3_3_DIV_U16, inplace) {
    TEST_REQUIRES_X86_SSE2;
    const size_t batch_step = 16;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .TestLog(xnn_f32_vlog_ukernel__sse2_rational_3_3_div_u16);
    }
  }

  TEST(F32_VLOG__SSE2_RATIONAL_3_3_DIV_U16, special_values) {
    TEST_REQUIRES_X86_SSE2;
    constexpr size_t num_elements = 4;
    constexpr size_t buffered_size =
        num_elements + XNN_EXTRA_BYTES / sizeof(float);
    std::array<float, buffered_size> inputs =
        {1.0f, -1.0f, 0.0f, -0.0f};
    std::array<float, num_elements> expected =
        {0.0f, NAN, -INFINITY, -INFINITY};
    std::array<float, buffered_size> outputs;
    xnn_f32_vlog_ukernel__sse2_rational_3_3_div_u16(
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
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VLOG__AVX2_RATIONAL_3_3_DIV_U8, batch_eq_8) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(8)
      .TestLog(xnn_f32_vlog_ukernel__avx2_rational_3_3_div_u8);
  }

  TEST(F32_VLOG__AVX2_RATIONAL_3_3_DIV_U8, batch_div_8) {
    TEST_REQUIRES_X86_AVX2;
    const size_t batch_step = 8;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .TestLog(xnn_f32_vlog_ukernel__avx2_rational_3_3_div_u8);
    }
  }

  TEST(F32_VLOG__AVX2_RATIONAL_3_3_DIV_U8, batch_lt_8) {
    TEST_REQUIRES_X86_AVX2;
    const size_t batch_step = 8;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .TestLog(xnn_f32_vlog_ukernel__avx2_rational_3_3_div_u8);
    }
  }

  TEST(F32_VLOG__AVX2_RATIONAL_3_3_DIV_U8, batch_gt_8) {
    TEST_REQUIRES_X86_AVX2;
    const size_t batch_step = 8;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .TestLog(xnn_f32_vlog_ukernel__avx2_rational_3_3_div_u8);
    }
  }

  TEST(F32_VLOG__AVX2_RATIONAL_3_3_DIV_U8, inplace) {
    TEST_REQUIRES_X86_AVX2;
    const size_t batch_step = 8;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .TestLog(xnn_f32_vlog_ukernel__avx2_rational_3_3_div_u8);
    }
  }

  TEST(F32_VLOG__AVX2_RATIONAL_3_3_DIV_U8, special_values) {
    TEST_REQUIRES_X86_AVX2;
    constexpr size_t num_elements = 4;
    constexpr size_t buffered_size =
        num_elements + XNN_EXTRA_BYTES / sizeof(float);
    std::array<float, buffered_size> inputs =
        {1.0f, -1.0f, 0.0f, -0.0f};
    std::array<float, num_elements> expected =
        {0.0f, NAN, -INFINITY, -INFINITY};
    std::array<float, buffered_size> outputs;
    xnn_f32_vlog_ukernel__avx2_rational_3_3_div_u8(
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
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VLOG__AVX2_RATIONAL_3_3_DIV_U16, batch_eq_16) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(16)
      .TestLog(xnn_f32_vlog_ukernel__avx2_rational_3_3_div_u16);
  }

  TEST(F32_VLOG__AVX2_RATIONAL_3_3_DIV_U16, batch_div_16) {
    TEST_REQUIRES_X86_AVX2;
    const size_t batch_step = 16;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .TestLog(xnn_f32_vlog_ukernel__avx2_rational_3_3_div_u16);
    }
  }

  TEST(F32_VLOG__AVX2_RATIONAL_3_3_DIV_U16, batch_lt_16) {
    TEST_REQUIRES_X86_AVX2;
    const size_t batch_step = 16;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .TestLog(xnn_f32_vlog_ukernel__avx2_rational_3_3_div_u16);
    }
  }

  TEST(F32_VLOG__AVX2_RATIONAL_3_3_DIV_U16, batch_gt_16) {
    TEST_REQUIRES_X86_AVX2;
    const size_t batch_step = 16;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .TestLog(xnn_f32_vlog_ukernel__avx2_rational_3_3_div_u16);
    }
  }

  TEST(F32_VLOG__AVX2_RATIONAL_3_3_DIV_U16, inplace) {
    TEST_REQUIRES_X86_AVX2;
    const size_t batch_step = 16;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .TestLog(xnn_f32_vlog_ukernel__avx2_rational_3_3_div_u16);
    }
  }

  TEST(F32_VLOG__AVX2_RATIONAL_3_3_DIV_U16, special_values) {
    TEST_REQUIRES_X86_AVX2;
    constexpr size_t num_elements = 4;
    constexpr size_t buffered_size =
        num_elements + XNN_EXTRA_BYTES / sizeof(float);
    std::array<float, buffered_size> inputs =
        {1.0f, -1.0f, 0.0f, -0.0f};
    std::array<float, num_elements> expected =
        {0.0f, NAN, -INFINITY, -INFINITY};
    std::array<float, buffered_size> outputs;
    xnn_f32_vlog_ukernel__avx2_rational_3_3_div_u16(
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
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VLOG__AVX2_RATIONAL_3_3_DIV_U24, batch_eq_24) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(24)
      .TestLog(xnn_f32_vlog_ukernel__avx2_rational_3_3_div_u24);
  }

  TEST(F32_VLOG__AVX2_RATIONAL_3_3_DIV_U24, batch_div_24) {
    TEST_REQUIRES_X86_AVX2;
    const size_t batch_step = 24;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .TestLog(xnn_f32_vlog_ukernel__avx2_rational_3_3_div_u24);
    }
  }

  TEST(F32_VLOG__AVX2_RATIONAL_3_3_DIV_U24, batch_lt_24) {
    TEST_REQUIRES_X86_AVX2;
    const size_t batch_step = 24;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .TestLog(xnn_f32_vlog_ukernel__avx2_rational_3_3_div_u24);
    }
  }

  TEST(F32_VLOG__AVX2_RATIONAL_3_3_DIV_U24, batch_gt_24) {
    TEST_REQUIRES_X86_AVX2;
    const size_t batch_step = 24;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .TestLog(xnn_f32_vlog_ukernel__avx2_rational_3_3_div_u24);
    }
  }

  TEST(F32_VLOG__AVX2_RATIONAL_3_3_DIV_U24, inplace) {
    TEST_REQUIRES_X86_AVX2;
    const size_t batch_step = 24;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 23) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .TestLog(xnn_f32_vlog_ukernel__avx2_rational_3_3_div_u24);
    }
  }

  TEST(F32_VLOG__AVX2_RATIONAL_3_3_DIV_U24, special_values) {
    TEST_REQUIRES_X86_AVX2;
    constexpr size_t num_elements = 4;
    constexpr size_t buffered_size =
        num_elements + XNN_EXTRA_BYTES / sizeof(float);
    std::array<float, buffered_size> inputs =
        {1.0f, -1.0f, 0.0f, -0.0f};
    std::array<float, num_elements> expected =
        {0.0f, NAN, -INFINITY, -INFINITY};
    std::array<float, buffered_size> outputs;
    xnn_f32_vlog_ukernel__avx2_rational_3_3_div_u24(
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
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VLOG__AVX2_RATIONAL_3_3_DIV_U32, batch_eq_32) {
    TEST_REQUIRES_X86_AVX2;
    VUnaryMicrokernelTester()
      .batch_size(32)
      .TestLog(xnn_f32_vlog_ukernel__avx2_rational_3_3_div_u32);
  }

  TEST(F32_VLOG__AVX2_RATIONAL_3_3_DIV_U32, batch_div_32) {
    TEST_REQUIRES_X86_AVX2;
    const size_t batch_step = 32;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .TestLog(xnn_f32_vlog_ukernel__avx2_rational_3_3_div_u32);
    }
  }

  TEST(F32_VLOG__AVX2_RATIONAL_3_3_DIV_U32, batch_lt_32) {
    TEST_REQUIRES_X86_AVX2;
    const size_t batch_step = 32;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .TestLog(xnn_f32_vlog_ukernel__avx2_rational_3_3_div_u32);
    }
  }

  TEST(F32_VLOG__AVX2_RATIONAL_3_3_DIV_U32, batch_gt_32) {
    TEST_REQUIRES_X86_AVX2;
    const size_t batch_step = 32;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .TestLog(xnn_f32_vlog_ukernel__avx2_rational_3_3_div_u32);
    }
  }

  TEST(F32_VLOG__AVX2_RATIONAL_3_3_DIV_U32, inplace) {
    TEST_REQUIRES_X86_AVX2;
    const size_t batch_step = 32;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 31) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .TestLog(xnn_f32_vlog_ukernel__avx2_rational_3_3_div_u32);
    }
  }

  TEST(F32_VLOG__AVX2_RATIONAL_3_3_DIV_U32, special_values) {
    TEST_REQUIRES_X86_AVX2;
    constexpr size_t num_elements = 4;
    constexpr size_t buffered_size =
        num_elements + XNN_EXTRA_BYTES / sizeof(float);
    std::array<float, buffered_size> inputs =
        {1.0f, -1.0f, 0.0f, -0.0f};
    std::array<float, num_elements> expected =
        {0.0f, NAN, -INFINITY, -INFINITY};
    std::array<float, buffered_size> outputs;
    xnn_f32_vlog_ukernel__avx2_rational_3_3_div_u32(
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
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VLOG__FMA3_RATIONAL_3_3_DIV_U8, batch_eq_8) {
    TEST_REQUIRES_X86_FMA3;
    VUnaryMicrokernelTester()
      .batch_size(8)
      .TestLog(xnn_f32_vlog_ukernel__fma3_rational_3_3_div_u8);
  }

  TEST(F32_VLOG__FMA3_RATIONAL_3_3_DIV_U8, batch_div_8) {
    TEST_REQUIRES_X86_FMA3;
    const size_t batch_step = 8;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .TestLog(xnn_f32_vlog_ukernel__fma3_rational_3_3_div_u8);
    }
  }

  TEST(F32_VLOG__FMA3_RATIONAL_3_3_DIV_U8, batch_lt_8) {
    TEST_REQUIRES_X86_FMA3;
    const size_t batch_step = 8;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .TestLog(xnn_f32_vlog_ukernel__fma3_rational_3_3_div_u8);
    }
  }

  TEST(F32_VLOG__FMA3_RATIONAL_3_3_DIV_U8, batch_gt_8) {
    TEST_REQUIRES_X86_FMA3;
    const size_t batch_step = 8;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .TestLog(xnn_f32_vlog_ukernel__fma3_rational_3_3_div_u8);
    }
  }

  TEST(F32_VLOG__FMA3_RATIONAL_3_3_DIV_U8, inplace) {
    TEST_REQUIRES_X86_FMA3;
    const size_t batch_step = 8;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .TestLog(xnn_f32_vlog_ukernel__fma3_rational_3_3_div_u8);
    }
  }

  TEST(F32_VLOG__FMA3_RATIONAL_3_3_DIV_U8, special_values) {
    TEST_REQUIRES_X86_FMA3;
    constexpr size_t num_elements = 4;
    constexpr size_t buffered_size =
        num_elements + XNN_EXTRA_BYTES / sizeof(float);
    std::array<float, buffered_size> inputs =
        {1.0f, -1.0f, 0.0f, -0.0f};
    std::array<float, num_elements> expected =
        {0.0f, NAN, -INFINITY, -INFINITY};
    std::array<float, buffered_size> outputs;
    xnn_f32_vlog_ukernel__fma3_rational_3_3_div_u8(
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
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VLOG__FMA3_RATIONAL_3_3_DIV_U16, batch_eq_16) {
    TEST_REQUIRES_X86_FMA3;
    VUnaryMicrokernelTester()
      .batch_size(16)
      .TestLog(xnn_f32_vlog_ukernel__fma3_rational_3_3_div_u16);
  }

  TEST(F32_VLOG__FMA3_RATIONAL_3_3_DIV_U16, batch_div_16) {
    TEST_REQUIRES_X86_FMA3;
    const size_t batch_step = 16;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .TestLog(xnn_f32_vlog_ukernel__fma3_rational_3_3_div_u16);
    }
  }

  TEST(F32_VLOG__FMA3_RATIONAL_3_3_DIV_U16, batch_lt_16) {
    TEST_REQUIRES_X86_FMA3;
    const size_t batch_step = 16;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .TestLog(xnn_f32_vlog_ukernel__fma3_rational_3_3_div_u16);
    }
  }

  TEST(F32_VLOG__FMA3_RATIONAL_3_3_DIV_U16, batch_gt_16) {
    TEST_REQUIRES_X86_FMA3;
    const size_t batch_step = 16;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .TestLog(xnn_f32_vlog_ukernel__fma3_rational_3_3_div_u16);
    }
  }

  TEST(F32_VLOG__FMA3_RATIONAL_3_3_DIV_U16, inplace) {
    TEST_REQUIRES_X86_FMA3;
    const size_t batch_step = 16;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .TestLog(xnn_f32_vlog_ukernel__fma3_rational_3_3_div_u16);
    }
  }

  TEST(F32_VLOG__FMA3_RATIONAL_3_3_DIV_U16, special_values) {
    TEST_REQUIRES_X86_FMA3;
    constexpr size_t num_elements = 4;
    constexpr size_t buffered_size =
        num_elements + XNN_EXTRA_BYTES / sizeof(float);
    std::array<float, buffered_size> inputs =
        {1.0f, -1.0f, 0.0f, -0.0f};
    std::array<float, num_elements> expected =
        {0.0f, NAN, -INFINITY, -INFINITY};
    std::array<float, buffered_size> outputs;
    xnn_f32_vlog_ukernel__fma3_rational_3_3_div_u16(
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
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VLOG__FMA3_RATIONAL_3_3_DIV_U24, batch_eq_24) {
    TEST_REQUIRES_X86_FMA3;
    VUnaryMicrokernelTester()
      .batch_size(24)
      .TestLog(xnn_f32_vlog_ukernel__fma3_rational_3_3_div_u24);
  }

  TEST(F32_VLOG__FMA3_RATIONAL_3_3_DIV_U24, batch_div_24) {
    TEST_REQUIRES_X86_FMA3;
    const size_t batch_step = 24;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .TestLog(xnn_f32_vlog_ukernel__fma3_rational_3_3_div_u24);
    }
  }

  TEST(F32_VLOG__FMA3_RATIONAL_3_3_DIV_U24, batch_lt_24) {
    TEST_REQUIRES_X86_FMA3;
    const size_t batch_step = 24;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .TestLog(xnn_f32_vlog_ukernel__fma3_rational_3_3_div_u24);
    }
  }

  TEST(F32_VLOG__FMA3_RATIONAL_3_3_DIV_U24, batch_gt_24) {
    TEST_REQUIRES_X86_FMA3;
    const size_t batch_step = 24;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .TestLog(xnn_f32_vlog_ukernel__fma3_rational_3_3_div_u24);
    }
  }

  TEST(F32_VLOG__FMA3_RATIONAL_3_3_DIV_U24, inplace) {
    TEST_REQUIRES_X86_FMA3;
    const size_t batch_step = 24;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 23) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .TestLog(xnn_f32_vlog_ukernel__fma3_rational_3_3_div_u24);
    }
  }

  TEST(F32_VLOG__FMA3_RATIONAL_3_3_DIV_U24, special_values) {
    TEST_REQUIRES_X86_FMA3;
    constexpr size_t num_elements = 4;
    constexpr size_t buffered_size =
        num_elements + XNN_EXTRA_BYTES / sizeof(float);
    std::array<float, buffered_size> inputs =
        {1.0f, -1.0f, 0.0f, -0.0f};
    std::array<float, num_elements> expected =
        {0.0f, NAN, -INFINITY, -INFINITY};
    std::array<float, buffered_size> outputs;
    xnn_f32_vlog_ukernel__fma3_rational_3_3_div_u24(
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
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VLOG__FMA3_RATIONAL_3_3_DIV_U32, batch_eq_32) {
    TEST_REQUIRES_X86_FMA3;
    VUnaryMicrokernelTester()
      .batch_size(32)
      .TestLog(xnn_f32_vlog_ukernel__fma3_rational_3_3_div_u32);
  }

  TEST(F32_VLOG__FMA3_RATIONAL_3_3_DIV_U32, batch_div_32) {
    TEST_REQUIRES_X86_FMA3;
    const size_t batch_step = 32;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .TestLog(xnn_f32_vlog_ukernel__fma3_rational_3_3_div_u32);
    }
  }

  TEST(F32_VLOG__FMA3_RATIONAL_3_3_DIV_U32, batch_lt_32) {
    TEST_REQUIRES_X86_FMA3;
    const size_t batch_step = 32;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .TestLog(xnn_f32_vlog_ukernel__fma3_rational_3_3_div_u32);
    }
  }

  TEST(F32_VLOG__FMA3_RATIONAL_3_3_DIV_U32, batch_gt_32) {
    TEST_REQUIRES_X86_FMA3;
    const size_t batch_step = 32;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .TestLog(xnn_f32_vlog_ukernel__fma3_rational_3_3_div_u32);
    }
  }

  TEST(F32_VLOG__FMA3_RATIONAL_3_3_DIV_U32, inplace) {
    TEST_REQUIRES_X86_FMA3;
    const size_t batch_step = 32;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 31) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .TestLog(xnn_f32_vlog_ukernel__fma3_rational_3_3_div_u32);
    }
  }

  TEST(F32_VLOG__FMA3_RATIONAL_3_3_DIV_U32, special_values) {
    TEST_REQUIRES_X86_FMA3;
    constexpr size_t num_elements = 4;
    constexpr size_t buffered_size =
        num_elements + XNN_EXTRA_BYTES / sizeof(float);
    std::array<float, buffered_size> inputs =
        {1.0f, -1.0f, 0.0f, -0.0f};
    std::array<float, num_elements> expected =
        {0.0f, NAN, -INFINITY, -INFINITY};
    std::array<float, buffered_size> outputs;
    xnn_f32_vlog_ukernel__fma3_rational_3_3_div_u32(
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
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VLOG__FMA3_RATIONAL_3_3_NR_U8, batch_eq_8) {
    TEST_REQUIRES_X86_FMA3;
    VUnaryMicrokernelTester()
      .batch_size(8)
      .TestLog(xnn_f32_vlog_ukernel__fma3_rational_3_3_nr_u8);
  }

  TEST(F32_VLOG__FMA3_RATIONAL_3_3_NR_U8, batch_div_8) {
    TEST_REQUIRES_X86_FMA3;
    const size_t batch_step = 8;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .TestLog(xnn_f32_vlog_ukernel__fma3_rational_3_3_nr_u8);
    }
  }

  TEST(F32_VLOG__FMA3_RATIONAL_3_3_NR_U8, batch_lt_8) {
    TEST_REQUIRES_X86_FMA3;
    const size_t batch_step = 8;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .TestLog(xnn_f32_vlog_ukernel__fma3_rational_3_3_nr_u8);
    }
  }

  TEST(F32_VLOG__FMA3_RATIONAL_3_3_NR_U8, batch_gt_8) {
    TEST_REQUIRES_X86_FMA3;
    const size_t batch_step = 8;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .TestLog(xnn_f32_vlog_ukernel__fma3_rational_3_3_nr_u8);
    }
  }

  TEST(F32_VLOG__FMA3_RATIONAL_3_3_NR_U8, inplace) {
    TEST_REQUIRES_X86_FMA3;
    const size_t batch_step = 8;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .TestLog(xnn_f32_vlog_ukernel__fma3_rational_3_3_nr_u8);
    }
  }

  TEST(F32_VLOG__FMA3_RATIONAL_3_3_NR_U8, special_values) {
    TEST_REQUIRES_X86_FMA3;
    constexpr size_t num_elements = 4;
    constexpr size_t buffered_size =
        num_elements + XNN_EXTRA_BYTES / sizeof(float);
    std::array<float, buffered_size> inputs =
        {1.0f, -1.0f, 0.0f, -0.0f};
    std::array<float, num_elements> expected =
        {0.0f, NAN, -INFINITY, -INFINITY};
    std::array<float, buffered_size> outputs;
    xnn_f32_vlog_ukernel__fma3_rational_3_3_nr_u8(
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
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VLOG__FMA3_RATIONAL_3_3_NR_U16, batch_eq_16) {
    TEST_REQUIRES_X86_FMA3;
    VUnaryMicrokernelTester()
      .batch_size(16)
      .TestLog(xnn_f32_vlog_ukernel__fma3_rational_3_3_nr_u16);
  }

  TEST(F32_VLOG__FMA3_RATIONAL_3_3_NR_U16, batch_div_16) {
    TEST_REQUIRES_X86_FMA3;
    const size_t batch_step = 16;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .TestLog(xnn_f32_vlog_ukernel__fma3_rational_3_3_nr_u16);
    }
  }

  TEST(F32_VLOG__FMA3_RATIONAL_3_3_NR_U16, batch_lt_16) {
    TEST_REQUIRES_X86_FMA3;
    const size_t batch_step = 16;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .TestLog(xnn_f32_vlog_ukernel__fma3_rational_3_3_nr_u16);
    }
  }

  TEST(F32_VLOG__FMA3_RATIONAL_3_3_NR_U16, batch_gt_16) {
    TEST_REQUIRES_X86_FMA3;
    const size_t batch_step = 16;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .TestLog(xnn_f32_vlog_ukernel__fma3_rational_3_3_nr_u16);
    }
  }

  TEST(F32_VLOG__FMA3_RATIONAL_3_3_NR_U16, inplace) {
    TEST_REQUIRES_X86_FMA3;
    const size_t batch_step = 16;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .TestLog(xnn_f32_vlog_ukernel__fma3_rational_3_3_nr_u16);
    }
  }

  TEST(F32_VLOG__FMA3_RATIONAL_3_3_NR_U16, special_values) {
    TEST_REQUIRES_X86_FMA3;
    constexpr size_t num_elements = 4;
    constexpr size_t buffered_size =
        num_elements + XNN_EXTRA_BYTES / sizeof(float);
    std::array<float, buffered_size> inputs =
        {1.0f, -1.0f, 0.0f, -0.0f};
    std::array<float, num_elements> expected =
        {0.0f, NAN, -INFINITY, -INFINITY};
    std::array<float, buffered_size> outputs;
    xnn_f32_vlog_ukernel__fma3_rational_3_3_nr_u16(
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
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VLOG__FMA3_RATIONAL_3_3_NR_U24, batch_eq_24) {
    TEST_REQUIRES_X86_FMA3;
    VUnaryMicrokernelTester()
      .batch_size(24)
      .TestLog(xnn_f32_vlog_ukernel__fma3_rational_3_3_nr_u24);
  }

  TEST(F32_VLOG__FMA3_RATIONAL_3_3_NR_U24, batch_div_24) {
    TEST_REQUIRES_X86_FMA3;
    const size_t batch_step = 24;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .TestLog(xnn_f32_vlog_ukernel__fma3_rational_3_3_nr_u24);
    }
  }

  TEST(F32_VLOG__FMA3_RATIONAL_3_3_NR_U24, batch_lt_24) {
    TEST_REQUIRES_X86_FMA3;
    const size_t batch_step = 24;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .TestLog(xnn_f32_vlog_ukernel__fma3_rational_3_3_nr_u24);
    }
  }

  TEST(F32_VLOG__FMA3_RATIONAL_3_3_NR_U24, batch_gt_24) {
    TEST_REQUIRES_X86_FMA3;
    const size_t batch_step = 24;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .TestLog(xnn_f32_vlog_ukernel__fma3_rational_3_3_nr_u24);
    }
  }

  TEST(F32_VLOG__FMA3_RATIONAL_3_3_NR_U24, inplace) {
    TEST_REQUIRES_X86_FMA3;
    const size_t batch_step = 24;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 23) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .TestLog(xnn_f32_vlog_ukernel__fma3_rational_3_3_nr_u24);
    }
  }

  TEST(F32_VLOG__FMA3_RATIONAL_3_3_NR_U24, special_values) {
    TEST_REQUIRES_X86_FMA3;
    constexpr size_t num_elements = 4;
    constexpr size_t buffered_size =
        num_elements + XNN_EXTRA_BYTES / sizeof(float);
    std::array<float, buffered_size> inputs =
        {1.0f, -1.0f, 0.0f, -0.0f};
    std::array<float, num_elements> expected =
        {0.0f, NAN, -INFINITY, -INFINITY};
    std::array<float, buffered_size> outputs;
    xnn_f32_vlog_ukernel__fma3_rational_3_3_nr_u24(
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
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VLOG__FMA3_RATIONAL_3_3_NR_U32, batch_eq_32) {
    TEST_REQUIRES_X86_FMA3;
    VUnaryMicrokernelTester()
      .batch_size(32)
      .TestLog(xnn_f32_vlog_ukernel__fma3_rational_3_3_nr_u32);
  }

  TEST(F32_VLOG__FMA3_RATIONAL_3_3_NR_U32, batch_div_32) {
    TEST_REQUIRES_X86_FMA3;
    const size_t batch_step = 32;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .TestLog(xnn_f32_vlog_ukernel__fma3_rational_3_3_nr_u32);
    }
  }

  TEST(F32_VLOG__FMA3_RATIONAL_3_3_NR_U32, batch_lt_32) {
    TEST_REQUIRES_X86_FMA3;
    const size_t batch_step = 32;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .TestLog(xnn_f32_vlog_ukernel__fma3_rational_3_3_nr_u32);
    }
  }

  TEST(F32_VLOG__FMA3_RATIONAL_3_3_NR_U32, batch_gt_32) {
    TEST_REQUIRES_X86_FMA3;
    const size_t batch_step = 32;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .TestLog(xnn_f32_vlog_ukernel__fma3_rational_3_3_nr_u32);
    }
  }

  TEST(F32_VLOG__FMA3_RATIONAL_3_3_NR_U32, inplace) {
    TEST_REQUIRES_X86_FMA3;
    const size_t batch_step = 32;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 31) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .TestLog(xnn_f32_vlog_ukernel__fma3_rational_3_3_nr_u32);
    }
  }

  TEST(F32_VLOG__FMA3_RATIONAL_3_3_NR_U32, special_values) {
    TEST_REQUIRES_X86_FMA3;
    constexpr size_t num_elements = 4;
    constexpr size_t buffered_size =
        num_elements + XNN_EXTRA_BYTES / sizeof(float);
    std::array<float, buffered_size> inputs =
        {1.0f, -1.0f, 0.0f, -0.0f};
    std::array<float, num_elements> expected =
        {0.0f, NAN, -INFINITY, -INFINITY};
    std::array<float, buffered_size> outputs;
    xnn_f32_vlog_ukernel__fma3_rational_3_3_nr_u32(
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
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VLOG__AVX512F_RATIONAL_3_3_DIV_U16, batch_eq_16) {
    TEST_REQUIRES_X86_AVX512F;
    VUnaryMicrokernelTester()
      .batch_size(16)
      .TestLog(xnn_f32_vlog_ukernel__avx512f_rational_3_3_div_u16);
  }

  TEST(F32_VLOG__AVX512F_RATIONAL_3_3_DIV_U16, batch_div_16) {
    TEST_REQUIRES_X86_AVX512F;
    const size_t batch_step = 16;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .TestLog(xnn_f32_vlog_ukernel__avx512f_rational_3_3_div_u16);
    }
  }

  TEST(F32_VLOG__AVX512F_RATIONAL_3_3_DIV_U16, batch_lt_16) {
    TEST_REQUIRES_X86_AVX512F;
    const size_t batch_step = 16;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .TestLog(xnn_f32_vlog_ukernel__avx512f_rational_3_3_div_u16);
    }
  }

  TEST(F32_VLOG__AVX512F_RATIONAL_3_3_DIV_U16, batch_gt_16) {
    TEST_REQUIRES_X86_AVX512F;
    const size_t batch_step = 16;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .TestLog(xnn_f32_vlog_ukernel__avx512f_rational_3_3_div_u16);
    }
  }

  TEST(F32_VLOG__AVX512F_RATIONAL_3_3_DIV_U16, inplace) {
    TEST_REQUIRES_X86_AVX512F;
    const size_t batch_step = 16;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .TestLog(xnn_f32_vlog_ukernel__avx512f_rational_3_3_div_u16);
    }
  }

  TEST(F32_VLOG__AVX512F_RATIONAL_3_3_DIV_U16, special_values) {
    TEST_REQUIRES_X86_AVX512F;
    constexpr size_t num_elements = 4;
    constexpr size_t buffered_size =
        num_elements + XNN_EXTRA_BYTES / sizeof(float);
    std::array<float, buffered_size> inputs =
        {1.0f, -1.0f, 0.0f, -0.0f};
    std::array<float, num_elements> expected =
        {0.0f, NAN, -INFINITY, -INFINITY};
    std::array<float, buffered_size> outputs;
    xnn_f32_vlog_ukernel__avx512f_rational_3_3_div_u16(
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
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VLOG__AVX512F_RATIONAL_3_3_DIV_U32, batch_eq_32) {
    TEST_REQUIRES_X86_AVX512F;
    VUnaryMicrokernelTester()
      .batch_size(32)
      .TestLog(xnn_f32_vlog_ukernel__avx512f_rational_3_3_div_u32);
  }

  TEST(F32_VLOG__AVX512F_RATIONAL_3_3_DIV_U32, batch_div_32) {
    TEST_REQUIRES_X86_AVX512F;
    const size_t batch_step = 32;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .TestLog(xnn_f32_vlog_ukernel__avx512f_rational_3_3_div_u32);
    }
  }

  TEST(F32_VLOG__AVX512F_RATIONAL_3_3_DIV_U32, batch_lt_32) {
    TEST_REQUIRES_X86_AVX512F;
    const size_t batch_step = 32;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .TestLog(xnn_f32_vlog_ukernel__avx512f_rational_3_3_div_u32);
    }
  }

  TEST(F32_VLOG__AVX512F_RATIONAL_3_3_DIV_U32, batch_gt_32) {
    TEST_REQUIRES_X86_AVX512F;
    const size_t batch_step = 32;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .TestLog(xnn_f32_vlog_ukernel__avx512f_rational_3_3_div_u32);
    }
  }

  TEST(F32_VLOG__AVX512F_RATIONAL_3_3_DIV_U32, inplace) {
    TEST_REQUIRES_X86_AVX512F;
    const size_t batch_step = 32;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 31) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .TestLog(xnn_f32_vlog_ukernel__avx512f_rational_3_3_div_u32);
    }
  }

  TEST(F32_VLOG__AVX512F_RATIONAL_3_3_DIV_U32, special_values) {
    TEST_REQUIRES_X86_AVX512F;
    constexpr size_t num_elements = 4;
    constexpr size_t buffered_size =
        num_elements + XNN_EXTRA_BYTES / sizeof(float);
    std::array<float, buffered_size> inputs =
        {1.0f, -1.0f, 0.0f, -0.0f};
    std::array<float, num_elements> expected =
        {0.0f, NAN, -INFINITY, -INFINITY};
    std::array<float, buffered_size> outputs;
    xnn_f32_vlog_ukernel__avx512f_rational_3_3_div_u32(
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
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VLOG__AVX512F_RATIONAL_3_3_DIV_U48, batch_eq_48) {
    TEST_REQUIRES_X86_AVX512F;
    VUnaryMicrokernelTester()
      .batch_size(48)
      .TestLog(xnn_f32_vlog_ukernel__avx512f_rational_3_3_div_u48);
  }

  TEST(F32_VLOG__AVX512F_RATIONAL_3_3_DIV_U48, batch_div_48) {
    TEST_REQUIRES_X86_AVX512F;
    const size_t batch_step = 48;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .TestLog(xnn_f32_vlog_ukernel__avx512f_rational_3_3_div_u48);
    }
  }

  TEST(F32_VLOG__AVX512F_RATIONAL_3_3_DIV_U48, batch_lt_48) {
    TEST_REQUIRES_X86_AVX512F;
    const size_t batch_step = 48;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .TestLog(xnn_f32_vlog_ukernel__avx512f_rational_3_3_div_u48);
    }
  }

  TEST(F32_VLOG__AVX512F_RATIONAL_3_3_DIV_U48, batch_gt_48) {
    TEST_REQUIRES_X86_AVX512F;
    const size_t batch_step = 48;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .TestLog(xnn_f32_vlog_ukernel__avx512f_rational_3_3_div_u48);
    }
  }

  TEST(F32_VLOG__AVX512F_RATIONAL_3_3_DIV_U48, inplace) {
    TEST_REQUIRES_X86_AVX512F;
    const size_t batch_step = 48;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 47) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .TestLog(xnn_f32_vlog_ukernel__avx512f_rational_3_3_div_u48);
    }
  }

  TEST(F32_VLOG__AVX512F_RATIONAL_3_3_DIV_U48, special_values) {
    TEST_REQUIRES_X86_AVX512F;
    constexpr size_t num_elements = 4;
    constexpr size_t buffered_size =
        num_elements + XNN_EXTRA_BYTES / sizeof(float);
    std::array<float, buffered_size> inputs =
        {1.0f, -1.0f, 0.0f, -0.0f};
    std::array<float, num_elements> expected =
        {0.0f, NAN, -INFINITY, -INFINITY};
    std::array<float, buffered_size> outputs;
    xnn_f32_vlog_ukernel__avx512f_rational_3_3_div_u48(
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
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VLOG__AVX512F_RATIONAL_3_3_DIV_U64, batch_eq_64) {
    TEST_REQUIRES_X86_AVX512F;
    VUnaryMicrokernelTester()
      .batch_size(64)
      .TestLog(xnn_f32_vlog_ukernel__avx512f_rational_3_3_div_u64);
  }

  TEST(F32_VLOG__AVX512F_RATIONAL_3_3_DIV_U64, batch_div_64) {
    TEST_REQUIRES_X86_AVX512F;
    const size_t batch_step = 64;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .TestLog(xnn_f32_vlog_ukernel__avx512f_rational_3_3_div_u64);
    }
  }

  TEST(F32_VLOG__AVX512F_RATIONAL_3_3_DIV_U64, batch_lt_64) {
    TEST_REQUIRES_X86_AVX512F;
    const size_t batch_step = 64;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .TestLog(xnn_f32_vlog_ukernel__avx512f_rational_3_3_div_u64);
    }
  }

  TEST(F32_VLOG__AVX512F_RATIONAL_3_3_DIV_U64, batch_gt_64) {
    TEST_REQUIRES_X86_AVX512F;
    const size_t batch_step = 64;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .TestLog(xnn_f32_vlog_ukernel__avx512f_rational_3_3_div_u64);
    }
  }

  TEST(F32_VLOG__AVX512F_RATIONAL_3_3_DIV_U64, inplace) {
    TEST_REQUIRES_X86_AVX512F;
    const size_t batch_step = 64;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 63) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .TestLog(xnn_f32_vlog_ukernel__avx512f_rational_3_3_div_u64);
    }
  }

  TEST(F32_VLOG__AVX512F_RATIONAL_3_3_DIV_U64, special_values) {
    TEST_REQUIRES_X86_AVX512F;
    constexpr size_t num_elements = 4;
    constexpr size_t buffered_size =
        num_elements + XNN_EXTRA_BYTES / sizeof(float);
    std::array<float, buffered_size> inputs =
        {1.0f, -1.0f, 0.0f, -0.0f};
    std::array<float, num_elements> expected =
        {0.0f, NAN, -INFINITY, -INFINITY};
    std::array<float, buffered_size> outputs;
    xnn_f32_vlog_ukernel__avx512f_rational_3_3_div_u64(
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
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VLOG__AVX512F_RATIONAL_3_3_NR_U16, batch_eq_16) {
    TEST_REQUIRES_X86_AVX512F;
    VUnaryMicrokernelTester()
      .batch_size(16)
      .TestLog(xnn_f32_vlog_ukernel__avx512f_rational_3_3_nr_u16);
  }

  TEST(F32_VLOG__AVX512F_RATIONAL_3_3_NR_U16, batch_div_16) {
    TEST_REQUIRES_X86_AVX512F;
    const size_t batch_step = 16;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .TestLog(xnn_f32_vlog_ukernel__avx512f_rational_3_3_nr_u16);
    }
  }

  TEST(F32_VLOG__AVX512F_RATIONAL_3_3_NR_U16, batch_lt_16) {
    TEST_REQUIRES_X86_AVX512F;
    const size_t batch_step = 16;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .TestLog(xnn_f32_vlog_ukernel__avx512f_rational_3_3_nr_u16);
    }
  }

  TEST(F32_VLOG__AVX512F_RATIONAL_3_3_NR_U16, batch_gt_16) {
    TEST_REQUIRES_X86_AVX512F;
    const size_t batch_step = 16;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .TestLog(xnn_f32_vlog_ukernel__avx512f_rational_3_3_nr_u16);
    }
  }

  TEST(F32_VLOG__AVX512F_RATIONAL_3_3_NR_U16, inplace) {
    TEST_REQUIRES_X86_AVX512F;
    const size_t batch_step = 16;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .TestLog(xnn_f32_vlog_ukernel__avx512f_rational_3_3_nr_u16);
    }
  }

  TEST(F32_VLOG__AVX512F_RATIONAL_3_3_NR_U16, special_values) {
    TEST_REQUIRES_X86_AVX512F;
    constexpr size_t num_elements = 4;
    constexpr size_t buffered_size =
        num_elements + XNN_EXTRA_BYTES / sizeof(float);
    std::array<float, buffered_size> inputs =
        {1.0f, -1.0f, 0.0f, -0.0f};
    std::array<float, num_elements> expected =
        {0.0f, NAN, -INFINITY, -INFINITY};
    std::array<float, buffered_size> outputs;
    xnn_f32_vlog_ukernel__avx512f_rational_3_3_nr_u16(
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
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VLOG__AVX512F_RATIONAL_3_3_NR_U32, batch_eq_32) {
    TEST_REQUIRES_X86_AVX512F;
    VUnaryMicrokernelTester()
      .batch_size(32)
      .TestLog(xnn_f32_vlog_ukernel__avx512f_rational_3_3_nr_u32);
  }

  TEST(F32_VLOG__AVX512F_RATIONAL_3_3_NR_U32, batch_div_32) {
    TEST_REQUIRES_X86_AVX512F;
    const size_t batch_step = 32;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .TestLog(xnn_f32_vlog_ukernel__avx512f_rational_3_3_nr_u32);
    }
  }

  TEST(F32_VLOG__AVX512F_RATIONAL_3_3_NR_U32, batch_lt_32) {
    TEST_REQUIRES_X86_AVX512F;
    const size_t batch_step = 32;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .TestLog(xnn_f32_vlog_ukernel__avx512f_rational_3_3_nr_u32);
    }
  }

  TEST(F32_VLOG__AVX512F_RATIONAL_3_3_NR_U32, batch_gt_32) {
    TEST_REQUIRES_X86_AVX512F;
    const size_t batch_step = 32;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .TestLog(xnn_f32_vlog_ukernel__avx512f_rational_3_3_nr_u32);
    }
  }

  TEST(F32_VLOG__AVX512F_RATIONAL_3_3_NR_U32, inplace) {
    TEST_REQUIRES_X86_AVX512F;
    const size_t batch_step = 32;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 31) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .TestLog(xnn_f32_vlog_ukernel__avx512f_rational_3_3_nr_u32);
    }
  }

  TEST(F32_VLOG__AVX512F_RATIONAL_3_3_NR_U32, special_values) {
    TEST_REQUIRES_X86_AVX512F;
    constexpr size_t num_elements = 4;
    constexpr size_t buffered_size =
        num_elements + XNN_EXTRA_BYTES / sizeof(float);
    std::array<float, buffered_size> inputs =
        {1.0f, -1.0f, 0.0f, -0.0f};
    std::array<float, num_elements> expected =
        {0.0f, NAN, -INFINITY, -INFINITY};
    std::array<float, buffered_size> outputs;
    xnn_f32_vlog_ukernel__avx512f_rational_3_3_nr_u32(
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
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VLOG__AVX512F_RATIONAL_3_3_NR_U48, batch_eq_48) {
    TEST_REQUIRES_X86_AVX512F;
    VUnaryMicrokernelTester()
      .batch_size(48)
      .TestLog(xnn_f32_vlog_ukernel__avx512f_rational_3_3_nr_u48);
  }

  TEST(F32_VLOG__AVX512F_RATIONAL_3_3_NR_U48, batch_div_48) {
    TEST_REQUIRES_X86_AVX512F;
    const size_t batch_step = 48;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .TestLog(xnn_f32_vlog_ukernel__avx512f_rational_3_3_nr_u48);
    }
  }

  TEST(F32_VLOG__AVX512F_RATIONAL_3_3_NR_U48, batch_lt_48) {
    TEST_REQUIRES_X86_AVX512F;
    const size_t batch_step = 48;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .TestLog(xnn_f32_vlog_ukernel__avx512f_rational_3_3_nr_u48);
    }
  }

  TEST(F32_VLOG__AVX512F_RATIONAL_3_3_NR_U48, batch_gt_48) {
    TEST_REQUIRES_X86_AVX512F;
    const size_t batch_step = 48;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .TestLog(xnn_f32_vlog_ukernel__avx512f_rational_3_3_nr_u48);
    }
  }

  TEST(F32_VLOG__AVX512F_RATIONAL_3_3_NR_U48, inplace) {
    TEST_REQUIRES_X86_AVX512F;
    const size_t batch_step = 48;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 47) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .TestLog(xnn_f32_vlog_ukernel__avx512f_rational_3_3_nr_u48);
    }
  }

  TEST(F32_VLOG__AVX512F_RATIONAL_3_3_NR_U48, special_values) {
    TEST_REQUIRES_X86_AVX512F;
    constexpr size_t num_elements = 4;
    constexpr size_t buffered_size =
        num_elements + XNN_EXTRA_BYTES / sizeof(float);
    std::array<float, buffered_size> inputs =
        {1.0f, -1.0f, 0.0f, -0.0f};
    std::array<float, num_elements> expected =
        {0.0f, NAN, -INFINITY, -INFINITY};
    std::array<float, buffered_size> outputs;
    xnn_f32_vlog_ukernel__avx512f_rational_3_3_nr_u48(
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
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VLOG__AVX512F_RATIONAL_3_3_NR_U64, batch_eq_64) {
    TEST_REQUIRES_X86_AVX512F;
    VUnaryMicrokernelTester()
      .batch_size(64)
      .TestLog(xnn_f32_vlog_ukernel__avx512f_rational_3_3_nr_u64);
  }

  TEST(F32_VLOG__AVX512F_RATIONAL_3_3_NR_U64, batch_div_64) {
    TEST_REQUIRES_X86_AVX512F;
    const size_t batch_step = 64;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .TestLog(xnn_f32_vlog_ukernel__avx512f_rational_3_3_nr_u64);
    }
  }

  TEST(F32_VLOG__AVX512F_RATIONAL_3_3_NR_U64, batch_lt_64) {
    TEST_REQUIRES_X86_AVX512F;
    const size_t batch_step = 64;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .TestLog(xnn_f32_vlog_ukernel__avx512f_rational_3_3_nr_u64);
    }
  }

  TEST(F32_VLOG__AVX512F_RATIONAL_3_3_NR_U64, batch_gt_64) {
    TEST_REQUIRES_X86_AVX512F;
    const size_t batch_step = 64;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .TestLog(xnn_f32_vlog_ukernel__avx512f_rational_3_3_nr_u64);
    }
  }

  TEST(F32_VLOG__AVX512F_RATIONAL_3_3_NR_U64, inplace) {
    TEST_REQUIRES_X86_AVX512F;
    const size_t batch_step = 64;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 63) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .TestLog(xnn_f32_vlog_ukernel__avx512f_rational_3_3_nr_u64);
    }
  }

  TEST(F32_VLOG__AVX512F_RATIONAL_3_3_NR_U64, special_values) {
    TEST_REQUIRES_X86_AVX512F;
    constexpr size_t num_elements = 4;
    constexpr size_t buffered_size =
        num_elements + XNN_EXTRA_BYTES / sizeof(float);
    std::array<float, buffered_size> inputs =
        {1.0f, -1.0f, 0.0f, -0.0f};
    std::array<float, num_elements> expected =
        {0.0f, NAN, -INFINITY, -INFINITY};
    std::array<float, buffered_size> outputs;
    xnn_f32_vlog_ukernel__avx512f_rational_3_3_nr_u64(
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
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VLOG__NEON_RATIONAL_3_3_DIV_U4, batch_eq_4) {
    TEST_REQUIRES_ARM_NEON;
    VUnaryMicrokernelTester()
      .batch_size(4)
      .TestLog(xnn_f32_vlog_ukernel__neon_rational_3_3_div_u4);
  }

  TEST(F32_VLOG__NEON_RATIONAL_3_3_DIV_U4, batch_div_4) {
    TEST_REQUIRES_ARM_NEON;
    const size_t batch_step = 4;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .TestLog(xnn_f32_vlog_ukernel__neon_rational_3_3_div_u4);
    }
  }

  TEST(F32_VLOG__NEON_RATIONAL_3_3_DIV_U4, batch_lt_4) {
    TEST_REQUIRES_ARM_NEON;
    const size_t batch_step = 4;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .TestLog(xnn_f32_vlog_ukernel__neon_rational_3_3_div_u4);
    }
  }

  TEST(F32_VLOG__NEON_RATIONAL_3_3_DIV_U4, batch_gt_4) {
    TEST_REQUIRES_ARM_NEON;
    const size_t batch_step = 4;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .TestLog(xnn_f32_vlog_ukernel__neon_rational_3_3_div_u4);
    }
  }

  TEST(F32_VLOG__NEON_RATIONAL_3_3_DIV_U4, inplace) {
    TEST_REQUIRES_ARM_NEON;
    const size_t batch_step = 4;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 3) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .TestLog(xnn_f32_vlog_ukernel__neon_rational_3_3_div_u4);
    }
  }

  TEST(F32_VLOG__NEON_RATIONAL_3_3_DIV_U4, special_values) {
    TEST_REQUIRES_ARM_NEON;
    constexpr size_t num_elements = 4;
    constexpr size_t buffered_size =
        num_elements + XNN_EXTRA_BYTES / sizeof(float);
    std::array<float, buffered_size> inputs =
        {1.0f, -1.0f, 0.0f, -0.0f};
    std::array<float, num_elements> expected =
        {0.0f, NAN, -INFINITY, -INFINITY};
    std::array<float, buffered_size> outputs;
    xnn_f32_vlog_ukernel__neon_rational_3_3_div_u4(
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
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VLOG__NEON_RATIONAL_3_3_DIV_U8, batch_eq_8) {
    TEST_REQUIRES_ARM_NEON;
    VUnaryMicrokernelTester()
      .batch_size(8)
      .TestLog(xnn_f32_vlog_ukernel__neon_rational_3_3_div_u8);
  }

  TEST(F32_VLOG__NEON_RATIONAL_3_3_DIV_U8, batch_div_8) {
    TEST_REQUIRES_ARM_NEON;
    const size_t batch_step = 8;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .TestLog(xnn_f32_vlog_ukernel__neon_rational_3_3_div_u8);
    }
  }

  TEST(F32_VLOG__NEON_RATIONAL_3_3_DIV_U8, batch_lt_8) {
    TEST_REQUIRES_ARM_NEON;
    const size_t batch_step = 8;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .TestLog(xnn_f32_vlog_ukernel__neon_rational_3_3_div_u8);
    }
  }

  TEST(F32_VLOG__NEON_RATIONAL_3_3_DIV_U8, batch_gt_8) {
    TEST_REQUIRES_ARM_NEON;
    const size_t batch_step = 8;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .TestLog(xnn_f32_vlog_ukernel__neon_rational_3_3_div_u8);
    }
  }

  TEST(F32_VLOG__NEON_RATIONAL_3_3_DIV_U8, inplace) {
    TEST_REQUIRES_ARM_NEON;
    const size_t batch_step = 8;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .TestLog(xnn_f32_vlog_ukernel__neon_rational_3_3_div_u8);
    }
  }

  TEST(F32_VLOG__NEON_RATIONAL_3_3_DIV_U8, special_values) {
    TEST_REQUIRES_ARM_NEON;
    constexpr size_t num_elements = 4;
    constexpr size_t buffered_size =
        num_elements + XNN_EXTRA_BYTES / sizeof(float);
    std::array<float, buffered_size> inputs =
        {1.0f, -1.0f, 0.0f, -0.0f};
    std::array<float, num_elements> expected =
        {0.0f, NAN, -INFINITY, -INFINITY};
    std::array<float, buffered_size> outputs;
    xnn_f32_vlog_ukernel__neon_rational_3_3_div_u8(
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
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VLOG__NEON_RATIONAL_3_3_DIV_U12, batch_eq_12) {
    TEST_REQUIRES_ARM_NEON;
    VUnaryMicrokernelTester()
      .batch_size(12)
      .TestLog(xnn_f32_vlog_ukernel__neon_rational_3_3_div_u12);
  }

  TEST(F32_VLOG__NEON_RATIONAL_3_3_DIV_U12, batch_div_12) {
    TEST_REQUIRES_ARM_NEON;
    const size_t batch_step = 12;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .TestLog(xnn_f32_vlog_ukernel__neon_rational_3_3_div_u12);
    }
  }

  TEST(F32_VLOG__NEON_RATIONAL_3_3_DIV_U12, batch_lt_12) {
    TEST_REQUIRES_ARM_NEON;
    const size_t batch_step = 12;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .TestLog(xnn_f32_vlog_ukernel__neon_rational_3_3_div_u12);
    }
  }

  TEST(F32_VLOG__NEON_RATIONAL_3_3_DIV_U12, batch_gt_12) {
    TEST_REQUIRES_ARM_NEON;
    const size_t batch_step = 12;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .TestLog(xnn_f32_vlog_ukernel__neon_rational_3_3_div_u12);
    }
  }

  TEST(F32_VLOG__NEON_RATIONAL_3_3_DIV_U12, inplace) {
    TEST_REQUIRES_ARM_NEON;
    const size_t batch_step = 12;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 11) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .TestLog(xnn_f32_vlog_ukernel__neon_rational_3_3_div_u12);
    }
  }

  TEST(F32_VLOG__NEON_RATIONAL_3_3_DIV_U12, special_values) {
    TEST_REQUIRES_ARM_NEON;
    constexpr size_t num_elements = 4;
    constexpr size_t buffered_size =
        num_elements + XNN_EXTRA_BYTES / sizeof(float);
    std::array<float, buffered_size> inputs =
        {1.0f, -1.0f, 0.0f, -0.0f};
    std::array<float, num_elements> expected =
        {0.0f, NAN, -INFINITY, -INFINITY};
    std::array<float, buffered_size> outputs;
    xnn_f32_vlog_ukernel__neon_rational_3_3_div_u12(
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
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_VLOG__NEON_RATIONAL_3_3_DIV_U16, batch_eq_16) {
    TEST_REQUIRES_ARM_NEON;
    VUnaryMicrokernelTester()
      .batch_size(16)
      .TestLog(xnn_f32_vlog_ukernel__neon_rational_3_3_div_u16);
  }

  TEST(F32_VLOG__NEON_RATIONAL_3_3_DIV_U16, batch_div_16) {
    TEST_REQUIRES_ARM_NEON;
    const size_t batch_step = 16;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .TestLog(xnn_f32_vlog_ukernel__neon_rational_3_3_div_u16);
    }
  }

  TEST(F32_VLOG__NEON_RATIONAL_3_3_DIV_U16, batch_lt_16) {
    TEST_REQUIRES_ARM_NEON;
    const size_t batch_step = 16;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .TestLog(xnn_f32_vlog_ukernel__neon_rational_3_3_div_u16);
    }
  }

  TEST(F32_VLOG__NEON_RATIONAL_3_3_DIV_U16, batch_gt_16) {
    TEST_REQUIRES_ARM_NEON;
    const size_t batch_step = 16;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .TestLog(xnn_f32_vlog_ukernel__neon_rational_3_3_div_u16);
    }
  }

  TEST(F32_VLOG__NEON_RATIONAL_3_3_DIV_U16, inplace) {
    TEST_REQUIRES_ARM_NEON;
    const size_t batch_step = 16;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .TestLog(xnn_f32_vlog_ukernel__neon_rational_3_3_div_u16);
    }
  }

  TEST(F32_VLOG__NEON_RATIONAL_3_3_DIV_U16, special_values) {
    TEST_REQUIRES_ARM_NEON;
    constexpr size_t num_elements = 4;
    constexpr size_t buffered_size =
        num_elements + XNN_EXTRA_BYTES / sizeof(float);
    std::array<float, buffered_size> inputs =
        {1.0f, -1.0f, 0.0f, -0.0f};
    std::array<float, num_elements> expected =
        {0.0f, NAN, -INFINITY, -INFINITY};
    std::array<float, buffered_size> outputs;
    xnn_f32_vlog_ukernel__neon_rational_3_3_div_u16(
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
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_VLOG__WASMSIMD_RATIONAL_3_3_DIV_U4, batch_eq_4) {
    VUnaryMicrokernelTester()
      .batch_size(4)
      .TestLog(xnn_f32_vlog_ukernel__wasmsimd_rational_3_3_div_u4);
  }

  TEST(F32_VLOG__WASMSIMD_RATIONAL_3_3_DIV_U4, batch_div_4) {
    const size_t batch_step = 4;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .TestLog(xnn_f32_vlog_ukernel__wasmsimd_rational_3_3_div_u4);
    }
  }

  TEST(F32_VLOG__WASMSIMD_RATIONAL_3_3_DIV_U4, batch_lt_4) {
    const size_t batch_step = 4;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .TestLog(xnn_f32_vlog_ukernel__wasmsimd_rational_3_3_div_u4);
    }
  }

  TEST(F32_VLOG__WASMSIMD_RATIONAL_3_3_DIV_U4, batch_gt_4) {
    const size_t batch_step = 4;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .TestLog(xnn_f32_vlog_ukernel__wasmsimd_rational_3_3_div_u4);
    }
  }

  TEST(F32_VLOG__WASMSIMD_RATIONAL_3_3_DIV_U4, inplace) {
    const size_t batch_step = 4;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 3) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .TestLog(xnn_f32_vlog_ukernel__wasmsimd_rational_3_3_div_u4);
    }
  }

  TEST(F32_VLOG__WASMSIMD_RATIONAL_3_3_DIV_U4, special_values) {
    constexpr size_t num_elements = 4;
    constexpr size_t buffered_size =
        num_elements + XNN_EXTRA_BYTES / sizeof(float);
    std::array<float, buffered_size> inputs =
        {1.0f, -1.0f, 0.0f, -0.0f};
    std::array<float, num_elements> expected =
        {0.0f, NAN, -INFINITY, -INFINITY};
    std::array<float, buffered_size> outputs;
    xnn_f32_vlog_ukernel__wasmsimd_rational_3_3_div_u4(
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
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_VLOG__WASMSIMD_RATIONAL_3_3_DIV_U8, batch_eq_8) {
    VUnaryMicrokernelTester()
      .batch_size(8)
      .TestLog(xnn_f32_vlog_ukernel__wasmsimd_rational_3_3_div_u8);
  }

  TEST(F32_VLOG__WASMSIMD_RATIONAL_3_3_DIV_U8, batch_div_8) {
    const size_t batch_step = 8;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .TestLog(xnn_f32_vlog_ukernel__wasmsimd_rational_3_3_div_u8);
    }
  }

  TEST(F32_VLOG__WASMSIMD_RATIONAL_3_3_DIV_U8, batch_lt_8) {
    const size_t batch_step = 8;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .TestLog(xnn_f32_vlog_ukernel__wasmsimd_rational_3_3_div_u8);
    }
  }

  TEST(F32_VLOG__WASMSIMD_RATIONAL_3_3_DIV_U8, batch_gt_8) {
    const size_t batch_step = 8;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .TestLog(xnn_f32_vlog_ukernel__wasmsimd_rational_3_3_div_u8);
    }
  }

  TEST(F32_VLOG__WASMSIMD_RATIONAL_3_3_DIV_U8, inplace) {
    const size_t batch_step = 8;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .TestLog(xnn_f32_vlog_ukernel__wasmsimd_rational_3_3_div_u8);
    }
  }

  TEST(F32_VLOG__WASMSIMD_RATIONAL_3_3_DIV_U8, special_values) {
    constexpr size_t num_elements = 4;
    constexpr size_t buffered_size =
        num_elements + XNN_EXTRA_BYTES / sizeof(float);
    std::array<float, buffered_size> inputs =
        {1.0f, -1.0f, 0.0f, -0.0f};
    std::array<float, num_elements> expected =
        {0.0f, NAN, -INFINITY, -INFINITY};
    std::array<float, buffered_size> outputs;
    xnn_f32_vlog_ukernel__wasmsimd_rational_3_3_div_u8(
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
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_VLOG__WASMSIMD_RATIONAL_3_3_DIV_U12, batch_eq_12) {
    VUnaryMicrokernelTester()
      .batch_size(12)
      .TestLog(xnn_f32_vlog_ukernel__wasmsimd_rational_3_3_div_u12);
  }

  TEST(F32_VLOG__WASMSIMD_RATIONAL_3_3_DIV_U12, batch_div_12) {
    const size_t batch_step = 12;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .TestLog(xnn_f32_vlog_ukernel__wasmsimd_rational_3_3_div_u12);
    }
  }

  TEST(F32_VLOG__WASMSIMD_RATIONAL_3_3_DIV_U12, batch_lt_12) {
    const size_t batch_step = 12;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .TestLog(xnn_f32_vlog_ukernel__wasmsimd_rational_3_3_div_u12);
    }
  }

  TEST(F32_VLOG__WASMSIMD_RATIONAL_3_3_DIV_U12, batch_gt_12) {
    const size_t batch_step = 12;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .TestLog(xnn_f32_vlog_ukernel__wasmsimd_rational_3_3_div_u12);
    }
  }

  TEST(F32_VLOG__WASMSIMD_RATIONAL_3_3_DIV_U12, inplace) {
    const size_t batch_step = 12;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 11) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .TestLog(xnn_f32_vlog_ukernel__wasmsimd_rational_3_3_div_u12);
    }
  }

  TEST(F32_VLOG__WASMSIMD_RATIONAL_3_3_DIV_U12, special_values) {
    constexpr size_t num_elements = 4;
    constexpr size_t buffered_size =
        num_elements + XNN_EXTRA_BYTES / sizeof(float);
    std::array<float, buffered_size> inputs =
        {1.0f, -1.0f, 0.0f, -0.0f};
    std::array<float, num_elements> expected =
        {0.0f, NAN, -INFINITY, -INFINITY};
    std::array<float, buffered_size> outputs;
    xnn_f32_vlog_ukernel__wasmsimd_rational_3_3_div_u12(
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
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_VLOG__WASMSIMD_RATIONAL_3_3_DIV_U16, batch_eq_16) {
    VUnaryMicrokernelTester()
      .batch_size(16)
      .TestLog(xnn_f32_vlog_ukernel__wasmsimd_rational_3_3_div_u16);
  }

  TEST(F32_VLOG__WASMSIMD_RATIONAL_3_3_DIV_U16, batch_div_16) {
    const size_t batch_step = 16;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .TestLog(xnn_f32_vlog_ukernel__wasmsimd_rational_3_3_div_u16);
    }
  }

  TEST(F32_VLOG__WASMSIMD_RATIONAL_3_3_DIV_U16, batch_lt_16) {
    const size_t batch_step = 16;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .TestLog(xnn_f32_vlog_ukernel__wasmsimd_rational_3_3_div_u16);
    }
  }

  TEST(F32_VLOG__WASMSIMD_RATIONAL_3_3_DIV_U16, batch_gt_16) {
    const size_t batch_step = 16;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .TestLog(xnn_f32_vlog_ukernel__wasmsimd_rational_3_3_div_u16);
    }
  }

  TEST(F32_VLOG__WASMSIMD_RATIONAL_3_3_DIV_U16, inplace) {
    const size_t batch_step = 16;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .TestLog(xnn_f32_vlog_ukernel__wasmsimd_rational_3_3_div_u16);
    }
  }

  TEST(F32_VLOG__WASMSIMD_RATIONAL_3_3_DIV_U16, special_values) {
    constexpr size_t num_elements = 4;
    constexpr size_t buffered_size =
        num_elements + XNN_EXTRA_BYTES / sizeof(float);
    std::array<float, buffered_size> inputs =
        {1.0f, -1.0f, 0.0f, -0.0f};
    std::array<float, num_elements> expected =
        {0.0f, NAN, -INFINITY, -INFINITY};
    std::array<float, buffered_size> outputs;
    xnn_f32_vlog_ukernel__wasmsimd_rational_3_3_div_u16(
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
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
