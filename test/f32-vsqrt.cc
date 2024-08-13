// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: test/f32-vsqrt.yaml
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


#if XNN_ARCH_ARM64
  TEST(F32_VSQRT__AARCH64_NEON_SQRT_U4, batch_eq_4) {
    TEST_REQUIRES_ARM_NEON;
    VUnaryMicrokernelTester()
      .batch_size(4)
      .Test(xnn_f32_vsqrt_ukernel__aarch64_neon_sqrt_u4);
  }

  TEST(F32_VSQRT__AARCH64_NEON_SQRT_U4, batch_div_4) {
    TEST_REQUIRES_ARM_NEON;
    const size_t batch_step = 4;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsqrt_ukernel__aarch64_neon_sqrt_u4);
    }
  }

  TEST(F32_VSQRT__AARCH64_NEON_SQRT_U4, batch_lt_4) {
    TEST_REQUIRES_ARM_NEON;
    const size_t batch_step = 4;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsqrt_ukernel__aarch64_neon_sqrt_u4);
    }
  }

  TEST(F32_VSQRT__AARCH64_NEON_SQRT_U4, batch_gt_4) {
    TEST_REQUIRES_ARM_NEON;
    const size_t batch_step = 4;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsqrt_ukernel__aarch64_neon_sqrt_u4);
    }
  }

  TEST(F32_VSQRT__AARCH64_NEON_SQRT_U4, inplace) {
    TEST_REQUIRES_ARM_NEON;
    const size_t batch_step = 4;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 3) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsqrt_ukernel__aarch64_neon_sqrt_u4);
    }
  }

  TEST(F32_VSQRT__AARCH64_NEON_SQRT_U4, special_values) {
    TEST_REQUIRES_ARM_NEON;
    constexpr size_t num_elements = 4;
    constexpr size_t buffered_size =
        num_elements + XNN_EXTRA_BYTES / sizeof(float);
    std::array<float, buffered_size> inputs =
        {0.0f, -0.0f, 1.0f, -1.0f};
    std::array<float, num_elements> expected =
        {0.0f, -0.0f, 1.0f, NAN};
    std::array<float, buffered_size> outputs;
    xnn_f32_vsqrt_ukernel__aarch64_neon_sqrt_u4(
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
#endif  // XNN_ARCH_ARM64


#if XNN_ARCH_ARM64
  TEST(F32_VSQRT__AARCH64_NEON_SQRT_U8, batch_eq_8) {
    TEST_REQUIRES_ARM_NEON;
    VUnaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f32_vsqrt_ukernel__aarch64_neon_sqrt_u8);
  }

  TEST(F32_VSQRT__AARCH64_NEON_SQRT_U8, batch_div_8) {
    TEST_REQUIRES_ARM_NEON;
    const size_t batch_step = 8;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsqrt_ukernel__aarch64_neon_sqrt_u8);
    }
  }

  TEST(F32_VSQRT__AARCH64_NEON_SQRT_U8, batch_lt_8) {
    TEST_REQUIRES_ARM_NEON;
    const size_t batch_step = 8;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsqrt_ukernel__aarch64_neon_sqrt_u8);
    }
  }

  TEST(F32_VSQRT__AARCH64_NEON_SQRT_U8, batch_gt_8) {
    TEST_REQUIRES_ARM_NEON;
    const size_t batch_step = 8;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsqrt_ukernel__aarch64_neon_sqrt_u8);
    }
  }

  TEST(F32_VSQRT__AARCH64_NEON_SQRT_U8, inplace) {
    TEST_REQUIRES_ARM_NEON;
    const size_t batch_step = 8;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsqrt_ukernel__aarch64_neon_sqrt_u8);
    }
  }

  TEST(F32_VSQRT__AARCH64_NEON_SQRT_U8, special_values) {
    TEST_REQUIRES_ARM_NEON;
    constexpr size_t num_elements = 4;
    constexpr size_t buffered_size =
        num_elements + XNN_EXTRA_BYTES / sizeof(float);
    std::array<float, buffered_size> inputs =
        {0.0f, -0.0f, 1.0f, -1.0f};
    std::array<float, num_elements> expected =
        {0.0f, -0.0f, 1.0f, NAN};
    std::array<float, buffered_size> outputs;
    xnn_f32_vsqrt_ukernel__aarch64_neon_sqrt_u8(
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
#endif  // XNN_ARCH_ARM64


#if XNN_ARCH_ARM64
  TEST(F32_VSQRT__AARCH64_NEON_SQRT_U16, batch_eq_16) {
    TEST_REQUIRES_ARM_NEON;
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_vsqrt_ukernel__aarch64_neon_sqrt_u16);
  }

  TEST(F32_VSQRT__AARCH64_NEON_SQRT_U16, batch_div_16) {
    TEST_REQUIRES_ARM_NEON;
    const size_t batch_step = 16;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsqrt_ukernel__aarch64_neon_sqrt_u16);
    }
  }

  TEST(F32_VSQRT__AARCH64_NEON_SQRT_U16, batch_lt_16) {
    TEST_REQUIRES_ARM_NEON;
    const size_t batch_step = 16;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsqrt_ukernel__aarch64_neon_sqrt_u16);
    }
  }

  TEST(F32_VSQRT__AARCH64_NEON_SQRT_U16, batch_gt_16) {
    TEST_REQUIRES_ARM_NEON;
    const size_t batch_step = 16;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsqrt_ukernel__aarch64_neon_sqrt_u16);
    }
  }

  TEST(F32_VSQRT__AARCH64_NEON_SQRT_U16, inplace) {
    TEST_REQUIRES_ARM_NEON;
    const size_t batch_step = 16;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsqrt_ukernel__aarch64_neon_sqrt_u16);
    }
  }

  TEST(F32_VSQRT__AARCH64_NEON_SQRT_U16, special_values) {
    TEST_REQUIRES_ARM_NEON;
    constexpr size_t num_elements = 4;
    constexpr size_t buffered_size =
        num_elements + XNN_EXTRA_BYTES / sizeof(float);
    std::array<float, buffered_size> inputs =
        {0.0f, -0.0f, 1.0f, -1.0f};
    std::array<float, num_elements> expected =
        {0.0f, -0.0f, 1.0f, NAN};
    std::array<float, buffered_size> outputs;
    xnn_f32_vsqrt_ukernel__aarch64_neon_sqrt_u16(
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
#endif  // XNN_ARCH_ARM64


#if XNN_ENABLE_RISCV_VECTOR && XNN_ARCH_RISCV
  TEST(F32_VSQRT__RVV_SQRT_U1V, batch_eq_1v) {
    TEST_REQUIRES_RISCV_VECTOR;
    VUnaryMicrokernelTester()
      .batch_size(1 * xnn_init_hardware_config()->vlenb / sizeof(float))
      .Test(xnn_f32_vsqrt_ukernel__rvv_sqrt_u1v);
  }

  TEST(F32_VSQRT__RVV_SQRT_U1V, batch_div_1v) {
    TEST_REQUIRES_RISCV_VECTOR;
    const size_t batch_step = 1 * xnn_init_hardware_config()->vlenb / sizeof(float);
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsqrt_ukernel__rvv_sqrt_u1v);
    }
  }

  TEST(F32_VSQRT__RVV_SQRT_U1V, batch_lt_1v) {
    TEST_REQUIRES_RISCV_VECTOR;
    const size_t batch_step = 1 * xnn_init_hardware_config()->vlenb / sizeof(float);
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsqrt_ukernel__rvv_sqrt_u1v);
    }
  }

  TEST(F32_VSQRT__RVV_SQRT_U1V, batch_gt_1v) {
    TEST_REQUIRES_RISCV_VECTOR;
    const size_t batch_step = 1 * xnn_init_hardware_config()->vlenb / sizeof(float);
    for (size_t batch_size = batch_step + 1; batch_size < 10; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsqrt_ukernel__rvv_sqrt_u1v);
    }
  }

  TEST(F32_VSQRT__RVV_SQRT_U1V, inplace) {
    TEST_REQUIRES_RISCV_VECTOR;
    const size_t batch_step = 1 * xnn_init_hardware_config()->vlenb / sizeof(float);
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 1) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsqrt_ukernel__rvv_sqrt_u1v);
    }
  }

  TEST(F32_VSQRT__RVV_SQRT_U1V, special_values) {
    TEST_REQUIRES_RISCV_VECTOR;
    constexpr size_t num_elements = 4;
    constexpr size_t buffered_size =
        num_elements + XNN_EXTRA_BYTES / sizeof(float);
    std::array<float, buffered_size> inputs =
        {0.0f, -0.0f, 1.0f, -1.0f};
    std::array<float, num_elements> expected =
        {0.0f, -0.0f, 1.0f, NAN};
    std::array<float, buffered_size> outputs;
    xnn_f32_vsqrt_ukernel__rvv_sqrt_u1v(
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
#endif  // XNN_ENABLE_RISCV_VECTOR && XNN_ARCH_RISCV


#if XNN_ENABLE_RISCV_VECTOR && XNN_ARCH_RISCV
  TEST(F32_VSQRT__RVV_SQRT_U2V, batch_eq_2v) {
    TEST_REQUIRES_RISCV_VECTOR;
    VUnaryMicrokernelTester()
      .batch_size(2 * xnn_init_hardware_config()->vlenb / sizeof(float))
      .Test(xnn_f32_vsqrt_ukernel__rvv_sqrt_u2v);
  }

  TEST(F32_VSQRT__RVV_SQRT_U2V, batch_div_2v) {
    TEST_REQUIRES_RISCV_VECTOR;
    const size_t batch_step = 2 * xnn_init_hardware_config()->vlenb / sizeof(float);
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsqrt_ukernel__rvv_sqrt_u2v);
    }
  }

  TEST(F32_VSQRT__RVV_SQRT_U2V, batch_lt_2v) {
    TEST_REQUIRES_RISCV_VECTOR;
    const size_t batch_step = 2 * xnn_init_hardware_config()->vlenb / sizeof(float);
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsqrt_ukernel__rvv_sqrt_u2v);
    }
  }

  TEST(F32_VSQRT__RVV_SQRT_U2V, batch_gt_2v) {
    TEST_REQUIRES_RISCV_VECTOR;
    const size_t batch_step = 2 * xnn_init_hardware_config()->vlenb / sizeof(float);
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsqrt_ukernel__rvv_sqrt_u2v);
    }
  }

  TEST(F32_VSQRT__RVV_SQRT_U2V, inplace) {
    TEST_REQUIRES_RISCV_VECTOR;
    const size_t batch_step = 2 * xnn_init_hardware_config()->vlenb / sizeof(float);
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 1) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsqrt_ukernel__rvv_sqrt_u2v);
    }
  }

  TEST(F32_VSQRT__RVV_SQRT_U2V, special_values) {
    TEST_REQUIRES_RISCV_VECTOR;
    constexpr size_t num_elements = 4;
    constexpr size_t buffered_size =
        num_elements + XNN_EXTRA_BYTES / sizeof(float);
    std::array<float, buffered_size> inputs =
        {0.0f, -0.0f, 1.0f, -1.0f};
    std::array<float, num_elements> expected =
        {0.0f, -0.0f, 1.0f, NAN};
    std::array<float, buffered_size> outputs;
    xnn_f32_vsqrt_ukernel__rvv_sqrt_u2v(
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
#endif  // XNN_ENABLE_RISCV_VECTOR && XNN_ARCH_RISCV


#if XNN_ENABLE_RISCV_VECTOR && XNN_ARCH_RISCV
  TEST(F32_VSQRT__RVV_SQRT_U4V, batch_eq_4v) {
    TEST_REQUIRES_RISCV_VECTOR;
    VUnaryMicrokernelTester()
      .batch_size(4 * xnn_init_hardware_config()->vlenb / sizeof(float))
      .Test(xnn_f32_vsqrt_ukernel__rvv_sqrt_u4v);
  }

  TEST(F32_VSQRT__RVV_SQRT_U4V, batch_div_4v) {
    TEST_REQUIRES_RISCV_VECTOR;
    const size_t batch_step = 4 * xnn_init_hardware_config()->vlenb / sizeof(float);
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsqrt_ukernel__rvv_sqrt_u4v);
    }
  }

  TEST(F32_VSQRT__RVV_SQRT_U4V, batch_lt_4v) {
    TEST_REQUIRES_RISCV_VECTOR;
    const size_t batch_step = 4 * xnn_init_hardware_config()->vlenb / sizeof(float);
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsqrt_ukernel__rvv_sqrt_u4v);
    }
  }

  TEST(F32_VSQRT__RVV_SQRT_U4V, batch_gt_4v) {
    TEST_REQUIRES_RISCV_VECTOR;
    const size_t batch_step = 4 * xnn_init_hardware_config()->vlenb / sizeof(float);
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsqrt_ukernel__rvv_sqrt_u4v);
    }
  }

  TEST(F32_VSQRT__RVV_SQRT_U4V, inplace) {
    TEST_REQUIRES_RISCV_VECTOR;
    const size_t batch_step = 4 * xnn_init_hardware_config()->vlenb / sizeof(float);
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 3) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsqrt_ukernel__rvv_sqrt_u4v);
    }
  }

  TEST(F32_VSQRT__RVV_SQRT_U4V, special_values) {
    TEST_REQUIRES_RISCV_VECTOR;
    constexpr size_t num_elements = 4;
    constexpr size_t buffered_size =
        num_elements + XNN_EXTRA_BYTES / sizeof(float);
    std::array<float, buffered_size> inputs =
        {0.0f, -0.0f, 1.0f, -1.0f};
    std::array<float, num_elements> expected =
        {0.0f, -0.0f, 1.0f, NAN};
    std::array<float, buffered_size> outputs;
    xnn_f32_vsqrt_ukernel__rvv_sqrt_u4v(
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
#endif  // XNN_ENABLE_RISCV_VECTOR && XNN_ARCH_RISCV


#if XNN_ENABLE_RISCV_VECTOR && XNN_ARCH_RISCV
  TEST(F32_VSQRT__RVV_SQRT_U8V, batch_eq_8v) {
    TEST_REQUIRES_RISCV_VECTOR;
    VUnaryMicrokernelTester()
      .batch_size(8 * xnn_init_hardware_config()->vlenb / sizeof(float))
      .Test(xnn_f32_vsqrt_ukernel__rvv_sqrt_u8v);
  }

  TEST(F32_VSQRT__RVV_SQRT_U8V, batch_div_8v) {
    TEST_REQUIRES_RISCV_VECTOR;
    const size_t batch_step = 8 * xnn_init_hardware_config()->vlenb / sizeof(float);
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsqrt_ukernel__rvv_sqrt_u8v);
    }
  }

  TEST(F32_VSQRT__RVV_SQRT_U8V, batch_lt_8v) {
    TEST_REQUIRES_RISCV_VECTOR;
    const size_t batch_step = 8 * xnn_init_hardware_config()->vlenb / sizeof(float);
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsqrt_ukernel__rvv_sqrt_u8v);
    }
  }

  TEST(F32_VSQRT__RVV_SQRT_U8V, batch_gt_8v) {
    TEST_REQUIRES_RISCV_VECTOR;
    const size_t batch_step = 8 * xnn_init_hardware_config()->vlenb / sizeof(float);
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsqrt_ukernel__rvv_sqrt_u8v);
    }
  }

  TEST(F32_VSQRT__RVV_SQRT_U8V, inplace) {
    TEST_REQUIRES_RISCV_VECTOR;
    const size_t batch_step = 8 * xnn_init_hardware_config()->vlenb / sizeof(float);
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsqrt_ukernel__rvv_sqrt_u8v);
    }
  }

  TEST(F32_VSQRT__RVV_SQRT_U8V, special_values) {
    TEST_REQUIRES_RISCV_VECTOR;
    constexpr size_t num_elements = 4;
    constexpr size_t buffered_size =
        num_elements + XNN_EXTRA_BYTES / sizeof(float);
    std::array<float, buffered_size> inputs =
        {0.0f, -0.0f, 1.0f, -1.0f};
    std::array<float, num_elements> expected =
        {0.0f, -0.0f, 1.0f, NAN};
    std::array<float, buffered_size> outputs;
    xnn_f32_vsqrt_ukernel__rvv_sqrt_u8v(
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
#endif  // XNN_ENABLE_RISCV_VECTOR && XNN_ARCH_RISCV


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_VSQRT__SSE_SQRT_U4, batch_eq_4) {
    TEST_REQUIRES_X86_SSE;
    VUnaryMicrokernelTester()
      .batch_size(4)
      .Test(xnn_f32_vsqrt_ukernel__sse_sqrt_u4);
  }

  TEST(F32_VSQRT__SSE_SQRT_U4, batch_div_4) {
    TEST_REQUIRES_X86_SSE;
    const size_t batch_step = 4;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsqrt_ukernel__sse_sqrt_u4);
    }
  }

  TEST(F32_VSQRT__SSE_SQRT_U4, batch_lt_4) {
    TEST_REQUIRES_X86_SSE;
    const size_t batch_step = 4;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsqrt_ukernel__sse_sqrt_u4);
    }
  }

  TEST(F32_VSQRT__SSE_SQRT_U4, batch_gt_4) {
    TEST_REQUIRES_X86_SSE;
    const size_t batch_step = 4;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsqrt_ukernel__sse_sqrt_u4);
    }
  }

  TEST(F32_VSQRT__SSE_SQRT_U4, inplace) {
    TEST_REQUIRES_X86_SSE;
    const size_t batch_step = 4;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 3) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsqrt_ukernel__sse_sqrt_u4);
    }
  }

  TEST(F32_VSQRT__SSE_SQRT_U4, special_values) {
    TEST_REQUIRES_X86_SSE;
    constexpr size_t num_elements = 4;
    constexpr size_t buffered_size =
        num_elements + XNN_EXTRA_BYTES / sizeof(float);
    std::array<float, buffered_size> inputs =
        {0.0f, -0.0f, 1.0f, -1.0f};
    std::array<float, num_elements> expected =
        {0.0f, -0.0f, 1.0f, NAN};
    std::array<float, buffered_size> outputs;
    xnn_f32_vsqrt_ukernel__sse_sqrt_u4(
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
  TEST(F32_VSQRT__SSE_SQRT_U8, batch_eq_8) {
    TEST_REQUIRES_X86_SSE;
    VUnaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f32_vsqrt_ukernel__sse_sqrt_u8);
  }

  TEST(F32_VSQRT__SSE_SQRT_U8, batch_div_8) {
    TEST_REQUIRES_X86_SSE;
    const size_t batch_step = 8;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsqrt_ukernel__sse_sqrt_u8);
    }
  }

  TEST(F32_VSQRT__SSE_SQRT_U8, batch_lt_8) {
    TEST_REQUIRES_X86_SSE;
    const size_t batch_step = 8;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsqrt_ukernel__sse_sqrt_u8);
    }
  }

  TEST(F32_VSQRT__SSE_SQRT_U8, batch_gt_8) {
    TEST_REQUIRES_X86_SSE;
    const size_t batch_step = 8;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsqrt_ukernel__sse_sqrt_u8);
    }
  }

  TEST(F32_VSQRT__SSE_SQRT_U8, inplace) {
    TEST_REQUIRES_X86_SSE;
    const size_t batch_step = 8;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsqrt_ukernel__sse_sqrt_u8);
    }
  }

  TEST(F32_VSQRT__SSE_SQRT_U8, special_values) {
    TEST_REQUIRES_X86_SSE;
    constexpr size_t num_elements = 4;
    constexpr size_t buffered_size =
        num_elements + XNN_EXTRA_BYTES / sizeof(float);
    std::array<float, buffered_size> inputs =
        {0.0f, -0.0f, 1.0f, -1.0f};
    std::array<float, num_elements> expected =
        {0.0f, -0.0f, 1.0f, NAN};
    std::array<float, buffered_size> outputs;
    xnn_f32_vsqrt_ukernel__sse_sqrt_u8(
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
  TEST(F32_VSQRT__SSE_SQRT_U16, batch_eq_16) {
    TEST_REQUIRES_X86_SSE;
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_vsqrt_ukernel__sse_sqrt_u16);
  }

  TEST(F32_VSQRT__SSE_SQRT_U16, batch_div_16) {
    TEST_REQUIRES_X86_SSE;
    const size_t batch_step = 16;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsqrt_ukernel__sse_sqrt_u16);
    }
  }

  TEST(F32_VSQRT__SSE_SQRT_U16, batch_lt_16) {
    TEST_REQUIRES_X86_SSE;
    const size_t batch_step = 16;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsqrt_ukernel__sse_sqrt_u16);
    }
  }

  TEST(F32_VSQRT__SSE_SQRT_U16, batch_gt_16) {
    TEST_REQUIRES_X86_SSE;
    const size_t batch_step = 16;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsqrt_ukernel__sse_sqrt_u16);
    }
  }

  TEST(F32_VSQRT__SSE_SQRT_U16, inplace) {
    TEST_REQUIRES_X86_SSE;
    const size_t batch_step = 16;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsqrt_ukernel__sse_sqrt_u16);
    }
  }

  TEST(F32_VSQRT__SSE_SQRT_U16, special_values) {
    TEST_REQUIRES_X86_SSE;
    constexpr size_t num_elements = 4;
    constexpr size_t buffered_size =
        num_elements + XNN_EXTRA_BYTES / sizeof(float);
    std::array<float, buffered_size> inputs =
        {0.0f, -0.0f, 1.0f, -1.0f};
    std::array<float, num_elements> expected =
        {0.0f, -0.0f, 1.0f, NAN};
    std::array<float, buffered_size> outputs;
    xnn_f32_vsqrt_ukernel__sse_sqrt_u16(
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
  TEST(F32_VSQRT__SSE_RSQRT_U4, batch_eq_4) {
    TEST_REQUIRES_X86_SSE;
    VUnaryMicrokernelTester()
      .batch_size(4)
      .Test(xnn_f32_vsqrt_ukernel__sse_rsqrt_u4);
  }

  TEST(F32_VSQRT__SSE_RSQRT_U4, batch_div_4) {
    TEST_REQUIRES_X86_SSE;
    const size_t batch_step = 4;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsqrt_ukernel__sse_rsqrt_u4);
    }
  }

  TEST(F32_VSQRT__SSE_RSQRT_U4, batch_lt_4) {
    TEST_REQUIRES_X86_SSE;
    const size_t batch_step = 4;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsqrt_ukernel__sse_rsqrt_u4);
    }
  }

  TEST(F32_VSQRT__SSE_RSQRT_U4, batch_gt_4) {
    TEST_REQUIRES_X86_SSE;
    const size_t batch_step = 4;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsqrt_ukernel__sse_rsqrt_u4);
    }
  }

  TEST(F32_VSQRT__SSE_RSQRT_U4, inplace) {
    TEST_REQUIRES_X86_SSE;
    const size_t batch_step = 4;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 3) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsqrt_ukernel__sse_rsqrt_u4);
    }
  }

  TEST(F32_VSQRT__SSE_RSQRT_U4, special_values) {
    TEST_REQUIRES_X86_SSE;
    constexpr size_t num_elements = 4;
    constexpr size_t buffered_size =
        num_elements + XNN_EXTRA_BYTES / sizeof(float);
    std::array<float, buffered_size> inputs =
        {0.0f, -0.0f, 1.0f, -1.0f};
    std::array<float, num_elements> expected =
        {0.0f, -0.0f, 1.0f, NAN};
    std::array<float, buffered_size> outputs;
    xnn_f32_vsqrt_ukernel__sse_rsqrt_u4(
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
  TEST(F32_VSQRT__SSE_RSQRT_U8, batch_eq_8) {
    TEST_REQUIRES_X86_SSE;
    VUnaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f32_vsqrt_ukernel__sse_rsqrt_u8);
  }

  TEST(F32_VSQRT__SSE_RSQRT_U8, batch_div_8) {
    TEST_REQUIRES_X86_SSE;
    const size_t batch_step = 8;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsqrt_ukernel__sse_rsqrt_u8);
    }
  }

  TEST(F32_VSQRT__SSE_RSQRT_U8, batch_lt_8) {
    TEST_REQUIRES_X86_SSE;
    const size_t batch_step = 8;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsqrt_ukernel__sse_rsqrt_u8);
    }
  }

  TEST(F32_VSQRT__SSE_RSQRT_U8, batch_gt_8) {
    TEST_REQUIRES_X86_SSE;
    const size_t batch_step = 8;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsqrt_ukernel__sse_rsqrt_u8);
    }
  }

  TEST(F32_VSQRT__SSE_RSQRT_U8, inplace) {
    TEST_REQUIRES_X86_SSE;
    const size_t batch_step = 8;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsqrt_ukernel__sse_rsqrt_u8);
    }
  }

  TEST(F32_VSQRT__SSE_RSQRT_U8, special_values) {
    TEST_REQUIRES_X86_SSE;
    constexpr size_t num_elements = 4;
    constexpr size_t buffered_size =
        num_elements + XNN_EXTRA_BYTES / sizeof(float);
    std::array<float, buffered_size> inputs =
        {0.0f, -0.0f, 1.0f, -1.0f};
    std::array<float, num_elements> expected =
        {0.0f, -0.0f, 1.0f, NAN};
    std::array<float, buffered_size> outputs;
    xnn_f32_vsqrt_ukernel__sse_rsqrt_u8(
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
  TEST(F32_VSQRT__SSE_RSQRT_U12, batch_eq_12) {
    TEST_REQUIRES_X86_SSE;
    VUnaryMicrokernelTester()
      .batch_size(12)
      .Test(xnn_f32_vsqrt_ukernel__sse_rsqrt_u12);
  }

  TEST(F32_VSQRT__SSE_RSQRT_U12, batch_div_12) {
    TEST_REQUIRES_X86_SSE;
    const size_t batch_step = 12;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsqrt_ukernel__sse_rsqrt_u12);
    }
  }

  TEST(F32_VSQRT__SSE_RSQRT_U12, batch_lt_12) {
    TEST_REQUIRES_X86_SSE;
    const size_t batch_step = 12;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsqrt_ukernel__sse_rsqrt_u12);
    }
  }

  TEST(F32_VSQRT__SSE_RSQRT_U12, batch_gt_12) {
    TEST_REQUIRES_X86_SSE;
    const size_t batch_step = 12;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsqrt_ukernel__sse_rsqrt_u12);
    }
  }

  TEST(F32_VSQRT__SSE_RSQRT_U12, inplace) {
    TEST_REQUIRES_X86_SSE;
    const size_t batch_step = 12;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 11) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsqrt_ukernel__sse_rsqrt_u12);
    }
  }

  TEST(F32_VSQRT__SSE_RSQRT_U12, special_values) {
    TEST_REQUIRES_X86_SSE;
    constexpr size_t num_elements = 4;
    constexpr size_t buffered_size =
        num_elements + XNN_EXTRA_BYTES / sizeof(float);
    std::array<float, buffered_size> inputs =
        {0.0f, -0.0f, 1.0f, -1.0f};
    std::array<float, num_elements> expected =
        {0.0f, -0.0f, 1.0f, NAN};
    std::array<float, buffered_size> outputs;
    xnn_f32_vsqrt_ukernel__sse_rsqrt_u12(
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
  TEST(F32_VSQRT__AVX_SQRT_U8, batch_eq_8) {
    TEST_REQUIRES_X86_AVX;
    VUnaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f32_vsqrt_ukernel__avx_sqrt_u8);
  }

  TEST(F32_VSQRT__AVX_SQRT_U8, batch_div_8) {
    TEST_REQUIRES_X86_AVX;
    const size_t batch_step = 8;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsqrt_ukernel__avx_sqrt_u8);
    }
  }

  TEST(F32_VSQRT__AVX_SQRT_U8, batch_lt_8) {
    TEST_REQUIRES_X86_AVX;
    const size_t batch_step = 8;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsqrt_ukernel__avx_sqrt_u8);
    }
  }

  TEST(F32_VSQRT__AVX_SQRT_U8, batch_gt_8) {
    TEST_REQUIRES_X86_AVX;
    const size_t batch_step = 8;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsqrt_ukernel__avx_sqrt_u8);
    }
  }

  TEST(F32_VSQRT__AVX_SQRT_U8, inplace) {
    TEST_REQUIRES_X86_AVX;
    const size_t batch_step = 8;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsqrt_ukernel__avx_sqrt_u8);
    }
  }

  TEST(F32_VSQRT__AVX_SQRT_U8, special_values) {
    TEST_REQUIRES_X86_AVX;
    constexpr size_t num_elements = 4;
    constexpr size_t buffered_size =
        num_elements + XNN_EXTRA_BYTES / sizeof(float);
    std::array<float, buffered_size> inputs =
        {0.0f, -0.0f, 1.0f, -1.0f};
    std::array<float, num_elements> expected =
        {0.0f, -0.0f, 1.0f, NAN};
    std::array<float, buffered_size> outputs;
    xnn_f32_vsqrt_ukernel__avx_sqrt_u8(
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
  TEST(F32_VSQRT__AVX_SQRT_U16, batch_eq_16) {
    TEST_REQUIRES_X86_AVX;
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_vsqrt_ukernel__avx_sqrt_u16);
  }

  TEST(F32_VSQRT__AVX_SQRT_U16, batch_div_16) {
    TEST_REQUIRES_X86_AVX;
    const size_t batch_step = 16;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsqrt_ukernel__avx_sqrt_u16);
    }
  }

  TEST(F32_VSQRT__AVX_SQRT_U16, batch_lt_16) {
    TEST_REQUIRES_X86_AVX;
    const size_t batch_step = 16;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsqrt_ukernel__avx_sqrt_u16);
    }
  }

  TEST(F32_VSQRT__AVX_SQRT_U16, batch_gt_16) {
    TEST_REQUIRES_X86_AVX;
    const size_t batch_step = 16;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsqrt_ukernel__avx_sqrt_u16);
    }
  }

  TEST(F32_VSQRT__AVX_SQRT_U16, inplace) {
    TEST_REQUIRES_X86_AVX;
    const size_t batch_step = 16;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsqrt_ukernel__avx_sqrt_u16);
    }
  }

  TEST(F32_VSQRT__AVX_SQRT_U16, special_values) {
    TEST_REQUIRES_X86_AVX;
    constexpr size_t num_elements = 4;
    constexpr size_t buffered_size =
        num_elements + XNN_EXTRA_BYTES / sizeof(float);
    std::array<float, buffered_size> inputs =
        {0.0f, -0.0f, 1.0f, -1.0f};
    std::array<float, num_elements> expected =
        {0.0f, -0.0f, 1.0f, NAN};
    std::array<float, buffered_size> outputs;
    xnn_f32_vsqrt_ukernel__avx_sqrt_u16(
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
  TEST(F32_VSQRT__AVX_SQRT_U32, batch_eq_32) {
    TEST_REQUIRES_X86_AVX;
    VUnaryMicrokernelTester()
      .batch_size(32)
      .Test(xnn_f32_vsqrt_ukernel__avx_sqrt_u32);
  }

  TEST(F32_VSQRT__AVX_SQRT_U32, batch_div_32) {
    TEST_REQUIRES_X86_AVX;
    const size_t batch_step = 32;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsqrt_ukernel__avx_sqrt_u32);
    }
  }

  TEST(F32_VSQRT__AVX_SQRT_U32, batch_lt_32) {
    TEST_REQUIRES_X86_AVX;
    const size_t batch_step = 32;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsqrt_ukernel__avx_sqrt_u32);
    }
  }

  TEST(F32_VSQRT__AVX_SQRT_U32, batch_gt_32) {
    TEST_REQUIRES_X86_AVX;
    const size_t batch_step = 32;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsqrt_ukernel__avx_sqrt_u32);
    }
  }

  TEST(F32_VSQRT__AVX_SQRT_U32, inplace) {
    TEST_REQUIRES_X86_AVX;
    const size_t batch_step = 32;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 31) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsqrt_ukernel__avx_sqrt_u32);
    }
  }

  TEST(F32_VSQRT__AVX_SQRT_U32, special_values) {
    TEST_REQUIRES_X86_AVX;
    constexpr size_t num_elements = 4;
    constexpr size_t buffered_size =
        num_elements + XNN_EXTRA_BYTES / sizeof(float);
    std::array<float, buffered_size> inputs =
        {0.0f, -0.0f, 1.0f, -1.0f};
    std::array<float, num_elements> expected =
        {0.0f, -0.0f, 1.0f, NAN};
    std::array<float, buffered_size> outputs;
    xnn_f32_vsqrt_ukernel__avx_sqrt_u32(
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
  TEST(F32_VSQRT__AVX_RSQRT_U8, batch_eq_8) {
    TEST_REQUIRES_X86_AVX;
    VUnaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f32_vsqrt_ukernel__avx_rsqrt_u8);
  }

  TEST(F32_VSQRT__AVX_RSQRT_U8, batch_div_8) {
    TEST_REQUIRES_X86_AVX;
    const size_t batch_step = 8;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsqrt_ukernel__avx_rsqrt_u8);
    }
  }

  TEST(F32_VSQRT__AVX_RSQRT_U8, batch_lt_8) {
    TEST_REQUIRES_X86_AVX;
    const size_t batch_step = 8;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsqrt_ukernel__avx_rsqrt_u8);
    }
  }

  TEST(F32_VSQRT__AVX_RSQRT_U8, batch_gt_8) {
    TEST_REQUIRES_X86_AVX;
    const size_t batch_step = 8;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsqrt_ukernel__avx_rsqrt_u8);
    }
  }

  TEST(F32_VSQRT__AVX_RSQRT_U8, inplace) {
    TEST_REQUIRES_X86_AVX;
    const size_t batch_step = 8;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsqrt_ukernel__avx_rsqrt_u8);
    }
  }

  TEST(F32_VSQRT__AVX_RSQRT_U8, special_values) {
    TEST_REQUIRES_X86_AVX;
    constexpr size_t num_elements = 4;
    constexpr size_t buffered_size =
        num_elements + XNN_EXTRA_BYTES / sizeof(float);
    std::array<float, buffered_size> inputs =
        {0.0f, -0.0f, 1.0f, -1.0f};
    std::array<float, num_elements> expected =
        {0.0f, -0.0f, 1.0f, NAN};
    std::array<float, buffered_size> outputs;
    xnn_f32_vsqrt_ukernel__avx_rsqrt_u8(
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
  TEST(F32_VSQRT__AVX_RSQRT_U16, batch_eq_16) {
    TEST_REQUIRES_X86_AVX;
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_vsqrt_ukernel__avx_rsqrt_u16);
  }

  TEST(F32_VSQRT__AVX_RSQRT_U16, batch_div_16) {
    TEST_REQUIRES_X86_AVX;
    const size_t batch_step = 16;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsqrt_ukernel__avx_rsqrt_u16);
    }
  }

  TEST(F32_VSQRT__AVX_RSQRT_U16, batch_lt_16) {
    TEST_REQUIRES_X86_AVX;
    const size_t batch_step = 16;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsqrt_ukernel__avx_rsqrt_u16);
    }
  }

  TEST(F32_VSQRT__AVX_RSQRT_U16, batch_gt_16) {
    TEST_REQUIRES_X86_AVX;
    const size_t batch_step = 16;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsqrt_ukernel__avx_rsqrt_u16);
    }
  }

  TEST(F32_VSQRT__AVX_RSQRT_U16, inplace) {
    TEST_REQUIRES_X86_AVX;
    const size_t batch_step = 16;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsqrt_ukernel__avx_rsqrt_u16);
    }
  }

  TEST(F32_VSQRT__AVX_RSQRT_U16, special_values) {
    TEST_REQUIRES_X86_AVX;
    constexpr size_t num_elements = 4;
    constexpr size_t buffered_size =
        num_elements + XNN_EXTRA_BYTES / sizeof(float);
    std::array<float, buffered_size> inputs =
        {0.0f, -0.0f, 1.0f, -1.0f};
    std::array<float, num_elements> expected =
        {0.0f, -0.0f, 1.0f, NAN};
    std::array<float, buffered_size> outputs;
    xnn_f32_vsqrt_ukernel__avx_rsqrt_u16(
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
  TEST(F32_VSQRT__AVX_RSQRT_U32, batch_eq_32) {
    TEST_REQUIRES_X86_AVX;
    VUnaryMicrokernelTester()
      .batch_size(32)
      .Test(xnn_f32_vsqrt_ukernel__avx_rsqrt_u32);
  }

  TEST(F32_VSQRT__AVX_RSQRT_U32, batch_div_32) {
    TEST_REQUIRES_X86_AVX;
    const size_t batch_step = 32;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsqrt_ukernel__avx_rsqrt_u32);
    }
  }

  TEST(F32_VSQRT__AVX_RSQRT_U32, batch_lt_32) {
    TEST_REQUIRES_X86_AVX;
    const size_t batch_step = 32;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsqrt_ukernel__avx_rsqrt_u32);
    }
  }

  TEST(F32_VSQRT__AVX_RSQRT_U32, batch_gt_32) {
    TEST_REQUIRES_X86_AVX;
    const size_t batch_step = 32;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsqrt_ukernel__avx_rsqrt_u32);
    }
  }

  TEST(F32_VSQRT__AVX_RSQRT_U32, inplace) {
    TEST_REQUIRES_X86_AVX;
    const size_t batch_step = 32;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 31) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsqrt_ukernel__avx_rsqrt_u32);
    }
  }

  TEST(F32_VSQRT__AVX_RSQRT_U32, special_values) {
    TEST_REQUIRES_X86_AVX;
    constexpr size_t num_elements = 4;
    constexpr size_t buffered_size =
        num_elements + XNN_EXTRA_BYTES / sizeof(float);
    std::array<float, buffered_size> inputs =
        {0.0f, -0.0f, 1.0f, -1.0f};
    std::array<float, num_elements> expected =
        {0.0f, -0.0f, 1.0f, NAN};
    std::array<float, buffered_size> outputs;
    xnn_f32_vsqrt_ukernel__avx_rsqrt_u32(
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
  TEST(F32_VSQRT__FMA3_RSQRT_U8, batch_eq_8) {
    TEST_REQUIRES_X86_FMA3;
    VUnaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f32_vsqrt_ukernel__fma3_rsqrt_u8);
  }

  TEST(F32_VSQRT__FMA3_RSQRT_U8, batch_div_8) {
    TEST_REQUIRES_X86_FMA3;
    const size_t batch_step = 8;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsqrt_ukernel__fma3_rsqrt_u8);
    }
  }

  TEST(F32_VSQRT__FMA3_RSQRT_U8, batch_lt_8) {
    TEST_REQUIRES_X86_FMA3;
    const size_t batch_step = 8;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsqrt_ukernel__fma3_rsqrt_u8);
    }
  }

  TEST(F32_VSQRT__FMA3_RSQRT_U8, batch_gt_8) {
    TEST_REQUIRES_X86_FMA3;
    const size_t batch_step = 8;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsqrt_ukernel__fma3_rsqrt_u8);
    }
  }

  TEST(F32_VSQRT__FMA3_RSQRT_U8, inplace) {
    TEST_REQUIRES_X86_FMA3;
    const size_t batch_step = 8;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsqrt_ukernel__fma3_rsqrt_u8);
    }
  }

  TEST(F32_VSQRT__FMA3_RSQRT_U8, special_values) {
    TEST_REQUIRES_X86_FMA3;
    constexpr size_t num_elements = 4;
    constexpr size_t buffered_size =
        num_elements + XNN_EXTRA_BYTES / sizeof(float);
    std::array<float, buffered_size> inputs =
        {0.0f, -0.0f, 1.0f, -1.0f};
    std::array<float, num_elements> expected =
        {0.0f, -0.0f, 1.0f, NAN};
    std::array<float, buffered_size> outputs;
    xnn_f32_vsqrt_ukernel__fma3_rsqrt_u8(
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
  TEST(F32_VSQRT__FMA3_RSQRT_U16, batch_eq_16) {
    TEST_REQUIRES_X86_FMA3;
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_vsqrt_ukernel__fma3_rsqrt_u16);
  }

  TEST(F32_VSQRT__FMA3_RSQRT_U16, batch_div_16) {
    TEST_REQUIRES_X86_FMA3;
    const size_t batch_step = 16;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsqrt_ukernel__fma3_rsqrt_u16);
    }
  }

  TEST(F32_VSQRT__FMA3_RSQRT_U16, batch_lt_16) {
    TEST_REQUIRES_X86_FMA3;
    const size_t batch_step = 16;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsqrt_ukernel__fma3_rsqrt_u16);
    }
  }

  TEST(F32_VSQRT__FMA3_RSQRT_U16, batch_gt_16) {
    TEST_REQUIRES_X86_FMA3;
    const size_t batch_step = 16;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsqrt_ukernel__fma3_rsqrt_u16);
    }
  }

  TEST(F32_VSQRT__FMA3_RSQRT_U16, inplace) {
    TEST_REQUIRES_X86_FMA3;
    const size_t batch_step = 16;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsqrt_ukernel__fma3_rsqrt_u16);
    }
  }

  TEST(F32_VSQRT__FMA3_RSQRT_U16, special_values) {
    TEST_REQUIRES_X86_FMA3;
    constexpr size_t num_elements = 4;
    constexpr size_t buffered_size =
        num_elements + XNN_EXTRA_BYTES / sizeof(float);
    std::array<float, buffered_size> inputs =
        {0.0f, -0.0f, 1.0f, -1.0f};
    std::array<float, num_elements> expected =
        {0.0f, -0.0f, 1.0f, NAN};
    std::array<float, buffered_size> outputs;
    xnn_f32_vsqrt_ukernel__fma3_rsqrt_u16(
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
  TEST(F32_VSQRT__FMA3_RSQRT_U32, batch_eq_32) {
    TEST_REQUIRES_X86_FMA3;
    VUnaryMicrokernelTester()
      .batch_size(32)
      .Test(xnn_f32_vsqrt_ukernel__fma3_rsqrt_u32);
  }

  TEST(F32_VSQRT__FMA3_RSQRT_U32, batch_div_32) {
    TEST_REQUIRES_X86_FMA3;
    const size_t batch_step = 32;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsqrt_ukernel__fma3_rsqrt_u32);
    }
  }

  TEST(F32_VSQRT__FMA3_RSQRT_U32, batch_lt_32) {
    TEST_REQUIRES_X86_FMA3;
    const size_t batch_step = 32;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsqrt_ukernel__fma3_rsqrt_u32);
    }
  }

  TEST(F32_VSQRT__FMA3_RSQRT_U32, batch_gt_32) {
    TEST_REQUIRES_X86_FMA3;
    const size_t batch_step = 32;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsqrt_ukernel__fma3_rsqrt_u32);
    }
  }

  TEST(F32_VSQRT__FMA3_RSQRT_U32, inplace) {
    TEST_REQUIRES_X86_FMA3;
    const size_t batch_step = 32;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 31) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsqrt_ukernel__fma3_rsqrt_u32);
    }
  }

  TEST(F32_VSQRT__FMA3_RSQRT_U32, special_values) {
    TEST_REQUIRES_X86_FMA3;
    constexpr size_t num_elements = 4;
    constexpr size_t buffered_size =
        num_elements + XNN_EXTRA_BYTES / sizeof(float);
    std::array<float, buffered_size> inputs =
        {0.0f, -0.0f, 1.0f, -1.0f};
    std::array<float, num_elements> expected =
        {0.0f, -0.0f, 1.0f, NAN};
    std::array<float, buffered_size> outputs;
    xnn_f32_vsqrt_ukernel__fma3_rsqrt_u32(
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
  TEST(F32_VSQRT__AVX512F_RSQRT_U16, batch_eq_16) {
    TEST_REQUIRES_X86_AVX512F;
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_vsqrt_ukernel__avx512f_rsqrt_u16);
  }

  TEST(F32_VSQRT__AVX512F_RSQRT_U16, batch_div_16) {
    TEST_REQUIRES_X86_AVX512F;
    const size_t batch_step = 16;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsqrt_ukernel__avx512f_rsqrt_u16);
    }
  }

  TEST(F32_VSQRT__AVX512F_RSQRT_U16, batch_lt_16) {
    TEST_REQUIRES_X86_AVX512F;
    const size_t batch_step = 16;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsqrt_ukernel__avx512f_rsqrt_u16);
    }
  }

  TEST(F32_VSQRT__AVX512F_RSQRT_U16, batch_gt_16) {
    TEST_REQUIRES_X86_AVX512F;
    const size_t batch_step = 16;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsqrt_ukernel__avx512f_rsqrt_u16);
    }
  }

  TEST(F32_VSQRT__AVX512F_RSQRT_U16, inplace) {
    TEST_REQUIRES_X86_AVX512F;
    const size_t batch_step = 16;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsqrt_ukernel__avx512f_rsqrt_u16);
    }
  }

  TEST(F32_VSQRT__AVX512F_RSQRT_U16, special_values) {
    TEST_REQUIRES_X86_AVX512F;
    constexpr size_t num_elements = 4;
    constexpr size_t buffered_size =
        num_elements + XNN_EXTRA_BYTES / sizeof(float);
    std::array<float, buffered_size> inputs =
        {0.0f, -0.0f, 1.0f, -1.0f};
    std::array<float, num_elements> expected =
        {0.0f, -0.0f, 1.0f, NAN};
    std::array<float, buffered_size> outputs;
    xnn_f32_vsqrt_ukernel__avx512f_rsqrt_u16(
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
  TEST(F32_VSQRT__AVX512F_RSQRT_U32, batch_eq_32) {
    TEST_REQUIRES_X86_AVX512F;
    VUnaryMicrokernelTester()
      .batch_size(32)
      .Test(xnn_f32_vsqrt_ukernel__avx512f_rsqrt_u32);
  }

  TEST(F32_VSQRT__AVX512F_RSQRT_U32, batch_div_32) {
    TEST_REQUIRES_X86_AVX512F;
    const size_t batch_step = 32;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsqrt_ukernel__avx512f_rsqrt_u32);
    }
  }

  TEST(F32_VSQRT__AVX512F_RSQRT_U32, batch_lt_32) {
    TEST_REQUIRES_X86_AVX512F;
    const size_t batch_step = 32;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsqrt_ukernel__avx512f_rsqrt_u32);
    }
  }

  TEST(F32_VSQRT__AVX512F_RSQRT_U32, batch_gt_32) {
    TEST_REQUIRES_X86_AVX512F;
    const size_t batch_step = 32;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsqrt_ukernel__avx512f_rsqrt_u32);
    }
  }

  TEST(F32_VSQRT__AVX512F_RSQRT_U32, inplace) {
    TEST_REQUIRES_X86_AVX512F;
    const size_t batch_step = 32;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 31) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsqrt_ukernel__avx512f_rsqrt_u32);
    }
  }

  TEST(F32_VSQRT__AVX512F_RSQRT_U32, special_values) {
    TEST_REQUIRES_X86_AVX512F;
    constexpr size_t num_elements = 4;
    constexpr size_t buffered_size =
        num_elements + XNN_EXTRA_BYTES / sizeof(float);
    std::array<float, buffered_size> inputs =
        {0.0f, -0.0f, 1.0f, -1.0f};
    std::array<float, num_elements> expected =
        {0.0f, -0.0f, 1.0f, NAN};
    std::array<float, buffered_size> outputs;
    xnn_f32_vsqrt_ukernel__avx512f_rsqrt_u32(
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
  TEST(F32_VSQRT__AVX512F_RSQRT_U48, batch_eq_48) {
    TEST_REQUIRES_X86_AVX512F;
    VUnaryMicrokernelTester()
      .batch_size(48)
      .Test(xnn_f32_vsqrt_ukernel__avx512f_rsqrt_u48);
  }

  TEST(F32_VSQRT__AVX512F_RSQRT_U48, batch_div_48) {
    TEST_REQUIRES_X86_AVX512F;
    const size_t batch_step = 48;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsqrt_ukernel__avx512f_rsqrt_u48);
    }
  }

  TEST(F32_VSQRT__AVX512F_RSQRT_U48, batch_lt_48) {
    TEST_REQUIRES_X86_AVX512F;
    const size_t batch_step = 48;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsqrt_ukernel__avx512f_rsqrt_u48);
    }
  }

  TEST(F32_VSQRT__AVX512F_RSQRT_U48, batch_gt_48) {
    TEST_REQUIRES_X86_AVX512F;
    const size_t batch_step = 48;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsqrt_ukernel__avx512f_rsqrt_u48);
    }
  }

  TEST(F32_VSQRT__AVX512F_RSQRT_U48, inplace) {
    TEST_REQUIRES_X86_AVX512F;
    const size_t batch_step = 48;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 47) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsqrt_ukernel__avx512f_rsqrt_u48);
    }
  }

  TEST(F32_VSQRT__AVX512F_RSQRT_U48, special_values) {
    TEST_REQUIRES_X86_AVX512F;
    constexpr size_t num_elements = 4;
    constexpr size_t buffered_size =
        num_elements + XNN_EXTRA_BYTES / sizeof(float);
    std::array<float, buffered_size> inputs =
        {0.0f, -0.0f, 1.0f, -1.0f};
    std::array<float, num_elements> expected =
        {0.0f, -0.0f, 1.0f, NAN};
    std::array<float, buffered_size> outputs;
    xnn_f32_vsqrt_ukernel__avx512f_rsqrt_u48(
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


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_VSQRT__WASMSIMD_SQRT_U4, batch_eq_4) {
    VUnaryMicrokernelTester()
      .batch_size(4)
      .Test(xnn_f32_vsqrt_ukernel__wasmsimd_sqrt_u4);
  }

  TEST(F32_VSQRT__WASMSIMD_SQRT_U4, batch_div_4) {
    const size_t batch_step = 4;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsqrt_ukernel__wasmsimd_sqrt_u4);
    }
  }

  TEST(F32_VSQRT__WASMSIMD_SQRT_U4, batch_lt_4) {
    const size_t batch_step = 4;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsqrt_ukernel__wasmsimd_sqrt_u4);
    }
  }

  TEST(F32_VSQRT__WASMSIMD_SQRT_U4, batch_gt_4) {
    const size_t batch_step = 4;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsqrt_ukernel__wasmsimd_sqrt_u4);
    }
  }

  TEST(F32_VSQRT__WASMSIMD_SQRT_U4, inplace) {
    const size_t batch_step = 4;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 3) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsqrt_ukernel__wasmsimd_sqrt_u4);
    }
  }

  TEST(F32_VSQRT__WASMSIMD_SQRT_U4, special_values) {
    constexpr size_t num_elements = 4;
    constexpr size_t buffered_size =
        num_elements + XNN_EXTRA_BYTES / sizeof(float);
    std::array<float, buffered_size> inputs =
        {0.0f, -0.0f, 1.0f, -1.0f};
    std::array<float, num_elements> expected =
        {0.0f, -0.0f, 1.0f, NAN};
    std::array<float, buffered_size> outputs;
    xnn_f32_vsqrt_ukernel__wasmsimd_sqrt_u4(
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
  TEST(F32_VSQRT__WASMSIMD_SQRT_U8, batch_eq_8) {
    VUnaryMicrokernelTester()
      .batch_size(8)
      .Test(xnn_f32_vsqrt_ukernel__wasmsimd_sqrt_u8);
  }

  TEST(F32_VSQRT__WASMSIMD_SQRT_U8, batch_div_8) {
    const size_t batch_step = 8;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsqrt_ukernel__wasmsimd_sqrt_u8);
    }
  }

  TEST(F32_VSQRT__WASMSIMD_SQRT_U8, batch_lt_8) {
    const size_t batch_step = 8;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsqrt_ukernel__wasmsimd_sqrt_u8);
    }
  }

  TEST(F32_VSQRT__WASMSIMD_SQRT_U8, batch_gt_8) {
    const size_t batch_step = 8;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsqrt_ukernel__wasmsimd_sqrt_u8);
    }
  }

  TEST(F32_VSQRT__WASMSIMD_SQRT_U8, inplace) {
    const size_t batch_step = 8;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 7) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsqrt_ukernel__wasmsimd_sqrt_u8);
    }
  }

  TEST(F32_VSQRT__WASMSIMD_SQRT_U8, special_values) {
    constexpr size_t num_elements = 4;
    constexpr size_t buffered_size =
        num_elements + XNN_EXTRA_BYTES / sizeof(float);
    std::array<float, buffered_size> inputs =
        {0.0f, -0.0f, 1.0f, -1.0f};
    std::array<float, num_elements> expected =
        {0.0f, -0.0f, 1.0f, NAN};
    std::array<float, buffered_size> outputs;
    xnn_f32_vsqrt_ukernel__wasmsimd_sqrt_u8(
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
  TEST(F32_VSQRT__WASMSIMD_SQRT_U16, batch_eq_16) {
    VUnaryMicrokernelTester()
      .batch_size(16)
      .Test(xnn_f32_vsqrt_ukernel__wasmsimd_sqrt_u16);
  }

  TEST(F32_VSQRT__WASMSIMD_SQRT_U16, batch_div_16) {
    const size_t batch_step = 16;
    for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsqrt_ukernel__wasmsimd_sqrt_u16);
    }
  }

  TEST(F32_VSQRT__WASMSIMD_SQRT_U16, batch_lt_16) {
    const size_t batch_step = 16;
    for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsqrt_ukernel__wasmsimd_sqrt_u16);
    }
  }

  TEST(F32_VSQRT__WASMSIMD_SQRT_U16, batch_gt_16) {
    const size_t batch_step = 16;
    for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .Test(xnn_f32_vsqrt_ukernel__wasmsimd_sqrt_u16);
    }
  }

  TEST(F32_VSQRT__WASMSIMD_SQRT_U16, inplace) {
    const size_t batch_step = 16;
    for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 15) {
      VUnaryMicrokernelTester()
        .batch_size(batch_size)
        .inplace(true)
        .Test(xnn_f32_vsqrt_ukernel__wasmsimd_sqrt_u16);
    }
  }

  TEST(F32_VSQRT__WASMSIMD_SQRT_U16, special_values) {
    constexpr size_t num_elements = 4;
    constexpr size_t buffered_size =
        num_elements + XNN_EXTRA_BYTES / sizeof(float);
    std::array<float, buffered_size> inputs =
        {0.0f, -0.0f, 1.0f, -1.0f};
    std::array<float, num_elements> expected =
        {0.0f, -0.0f, 1.0f, NAN};
    std::array<float, buffered_size> outputs;
    xnn_f32_vsqrt_ukernel__wasmsimd_sqrt_u16(
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


TEST(F32_VSQRT__SCALAR_SQRT_U1, batch_eq_1) {
  VUnaryMicrokernelTester()
    .batch_size(1)
    .Test(xnn_f32_vsqrt_ukernel__scalar_sqrt_u1);
}

TEST(F32_VSQRT__SCALAR_SQRT_U1, batch_gt_1) {
  const size_t batch_step = 1;
  for (size_t batch_size = batch_step + 1; batch_size < 10; batch_size++) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_vsqrt_ukernel__scalar_sqrt_u1);
  }
}

TEST(F32_VSQRT__SCALAR_SQRT_U1, inplace) {
  const size_t batch_step = 1;
  for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 1) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .inplace(true)
      .Test(xnn_f32_vsqrt_ukernel__scalar_sqrt_u1);
  }
}

TEST(F32_VSQRT__SCALAR_SQRT_U1, special_values) {
  constexpr size_t num_elements = 4;
  constexpr size_t buffered_size =
      num_elements + XNN_EXTRA_BYTES / sizeof(float);
  std::array<float, buffered_size> inputs =
      {0.0f, -0.0f, 1.0f, -1.0f};
  std::array<float, num_elements> expected =
      {0.0f, -0.0f, 1.0f, NAN};
  std::array<float, buffered_size> outputs;
  xnn_f32_vsqrt_ukernel__scalar_sqrt_u1(
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

TEST(F32_VSQRT__SCALAR_SQRT_U2, batch_eq_2) {
  VUnaryMicrokernelTester()
    .batch_size(2)
    .Test(xnn_f32_vsqrt_ukernel__scalar_sqrt_u2);
}

TEST(F32_VSQRT__SCALAR_SQRT_U2, batch_div_2) {
  const size_t batch_step = 2;
  for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_vsqrt_ukernel__scalar_sqrt_u2);
  }
}

TEST(F32_VSQRT__SCALAR_SQRT_U2, batch_lt_2) {
  const size_t batch_step = 2;
  for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_vsqrt_ukernel__scalar_sqrt_u2);
  }
}

TEST(F32_VSQRT__SCALAR_SQRT_U2, batch_gt_2) {
  const size_t batch_step = 2;
  for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_vsqrt_ukernel__scalar_sqrt_u2);
  }
}

TEST(F32_VSQRT__SCALAR_SQRT_U2, inplace) {
  const size_t batch_step = 2;
  for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 1) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .inplace(true)
      .Test(xnn_f32_vsqrt_ukernel__scalar_sqrt_u2);
  }
}

TEST(F32_VSQRT__SCALAR_SQRT_U2, special_values) {
  constexpr size_t num_elements = 4;
  constexpr size_t buffered_size =
      num_elements + XNN_EXTRA_BYTES / sizeof(float);
  std::array<float, buffered_size> inputs =
      {0.0f, -0.0f, 1.0f, -1.0f};
  std::array<float, num_elements> expected =
      {0.0f, -0.0f, 1.0f, NAN};
  std::array<float, buffered_size> outputs;
  xnn_f32_vsqrt_ukernel__scalar_sqrt_u2(
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

TEST(F32_VSQRT__SCALAR_SQRT_U4, batch_eq_4) {
  VUnaryMicrokernelTester()
    .batch_size(4)
    .Test(xnn_f32_vsqrt_ukernel__scalar_sqrt_u4);
}

TEST(F32_VSQRT__SCALAR_SQRT_U4, batch_div_4) {
  const size_t batch_step = 4;
  for (size_t batch_size = 2 * batch_step; batch_size < 10 * batch_step; batch_size += batch_step) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_vsqrt_ukernel__scalar_sqrt_u4);
  }
}

TEST(F32_VSQRT__SCALAR_SQRT_U4, batch_lt_4) {
  const size_t batch_step = 4;
  for (size_t batch_size = 1; batch_size < batch_step; batch_size++) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_vsqrt_ukernel__scalar_sqrt_u4);
  }
}

TEST(F32_VSQRT__SCALAR_SQRT_U4, batch_gt_4) {
  const size_t batch_step = 4;
  for (size_t batch_size = batch_step + 1; batch_size < 2 * batch_step; batch_size++) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .Test(xnn_f32_vsqrt_ukernel__scalar_sqrt_u4);
  }
}

TEST(F32_VSQRT__SCALAR_SQRT_U4, inplace) {
  const size_t batch_step = 4;
  for (size_t batch_size = 1; batch_size <= batch_step; batch_size += 3) {
    VUnaryMicrokernelTester()
      .batch_size(batch_size)
      .inplace(true)
      .Test(xnn_f32_vsqrt_ukernel__scalar_sqrt_u4);
  }
}

TEST(F32_VSQRT__SCALAR_SQRT_U4, special_values) {
  constexpr size_t num_elements = 4;
  constexpr size_t buffered_size =
      num_elements + XNN_EXTRA_BYTES / sizeof(float);
  std::array<float, buffered_size> inputs =
      {0.0f, -0.0f, 1.0f, -1.0f};
  std::array<float, num_elements> expected =
      {0.0f, -0.0f, 1.0f, NAN};
  std::array<float, buffered_size> outputs;
  xnn_f32_vsqrt_ukernel__scalar_sqrt_u4(
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