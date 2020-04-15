// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: test/f32-raddstoreexpminusmax.yaml
//   Generator: tools/generate-raddstoreexpminusmax-test.py


#include <gtest/gtest.h>

#include <xnnpack/common.h>
#include <xnnpack/isa-checks.h>

#include <xnnpack/raddstoreexpminusmax.h>
#include "raddstoreexpminusmax-microkernel-tester.h"


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_RADDSTOREEXPMINUSMAX__NEON_P5_X4, elements_eq_4) {
    TEST_REQUIRES_ARM_NEON;
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(4)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__neon_p5_x4);
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEON_P5_X4, elements_div_4) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t elements = 8; elements < 40; elements += 4) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neon_p5_x4);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEON_P5_X4, elements_lt_4) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t elements = 1; elements < 4; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neon_p5_x4);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEON_P5_X4, elements_gt_4) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t elements = 5; elements < 8; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neon_p5_x4);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_RADDSTOREEXPMINUSMAX__NEON_P5_X8, elements_eq_8) {
    TEST_REQUIRES_ARM_NEON;
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(8)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__neon_p5_x8);
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEON_P5_X8, elements_div_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t elements = 16; elements < 80; elements += 8) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neon_p5_x8);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEON_P5_X8, elements_lt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t elements = 1; elements < 8; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neon_p5_x8);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEON_P5_X8, elements_gt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t elements = 9; elements < 16; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neon_p5_x8);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_RADDSTOREEXPMINUSMAX__NEON_P5_X8_ACC2, elements_eq_8) {
    TEST_REQUIRES_ARM_NEON;
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(8)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__neon_p5_x8_acc2);
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEON_P5_X8_ACC2, elements_div_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t elements = 16; elements < 80; elements += 8) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neon_p5_x8_acc2);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEON_P5_X8_ACC2, elements_lt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t elements = 1; elements < 8; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neon_p5_x8_acc2);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEON_P5_X8_ACC2, elements_gt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t elements = 9; elements < 16; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neon_p5_x8_acc2);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_RADDSTOREEXPMINUSMAX__NEON_P5_X12, elements_eq_12) {
    TEST_REQUIRES_ARM_NEON;
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(12)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__neon_p5_x12);
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEON_P5_X12, elements_div_12) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t elements = 24; elements < 120; elements += 12) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neon_p5_x12);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEON_P5_X12, elements_lt_12) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t elements = 1; elements < 12; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neon_p5_x12);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEON_P5_X12, elements_gt_12) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t elements = 13; elements < 24; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neon_p5_x12);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_RADDSTOREEXPMINUSMAX__NEON_P5_X12_ACC2, elements_eq_12) {
    TEST_REQUIRES_ARM_NEON;
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(12)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__neon_p5_x12_acc2);
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEON_P5_X12_ACC2, elements_div_12) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t elements = 24; elements < 120; elements += 12) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neon_p5_x12_acc2);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEON_P5_X12_ACC2, elements_lt_12) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t elements = 1; elements < 12; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neon_p5_x12_acc2);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEON_P5_X12_ACC2, elements_gt_12) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t elements = 13; elements < 24; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neon_p5_x12_acc2);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_RADDSTOREEXPMINUSMAX__NEON_P5_X12_ACC3, elements_eq_12) {
    TEST_REQUIRES_ARM_NEON;
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(12)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__neon_p5_x12_acc3);
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEON_P5_X12_ACC3, elements_div_12) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t elements = 24; elements < 120; elements += 12) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neon_p5_x12_acc3);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEON_P5_X12_ACC3, elements_lt_12) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t elements = 1; elements < 12; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neon_p5_x12_acc3);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEON_P5_X12_ACC3, elements_gt_12) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t elements = 13; elements < 24; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neon_p5_x12_acc3);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_RADDSTOREEXPMINUSMAX__NEON_P5_X16, elements_eq_16) {
    TEST_REQUIRES_ARM_NEON;
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(16)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__neon_p5_x16);
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEON_P5_X16, elements_div_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t elements = 32; elements < 160; elements += 16) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neon_p5_x16);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEON_P5_X16, elements_lt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t elements = 1; elements < 16; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neon_p5_x16);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEON_P5_X16, elements_gt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t elements = 17; elements < 32; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neon_p5_x16);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_RADDSTOREEXPMINUSMAX__NEON_P5_X16_ACC2, elements_eq_16) {
    TEST_REQUIRES_ARM_NEON;
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(16)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__neon_p5_x16_acc2);
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEON_P5_X16_ACC2, elements_div_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t elements = 32; elements < 160; elements += 16) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neon_p5_x16_acc2);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEON_P5_X16_ACC2, elements_lt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t elements = 1; elements < 16; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neon_p5_x16_acc2);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEON_P5_X16_ACC2, elements_gt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t elements = 17; elements < 32; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neon_p5_x16_acc2);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_RADDSTOREEXPMINUSMAX__NEON_P5_X16_ACC4, elements_eq_16) {
    TEST_REQUIRES_ARM_NEON;
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(16)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__neon_p5_x16_acc4);
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEON_P5_X16_ACC4, elements_div_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t elements = 32; elements < 160; elements += 16) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neon_p5_x16_acc4);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEON_P5_X16_ACC4, elements_lt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t elements = 1; elements < 16; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neon_p5_x16_acc4);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEON_P5_X16_ACC4, elements_gt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t elements = 17; elements < 32; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neon_p5_x16_acc4);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_RADDSTOREEXPMINUSMAX__NEON_P5_X20, elements_eq_20) {
    TEST_REQUIRES_ARM_NEON;
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(20)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__neon_p5_x20);
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEON_P5_X20, elements_div_20) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t elements = 40; elements < 200; elements += 20) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neon_p5_x20);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEON_P5_X20, elements_lt_20) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t elements = 1; elements < 20; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neon_p5_x20);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEON_P5_X20, elements_gt_20) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t elements = 21; elements < 40; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neon_p5_x20);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_RADDSTOREEXPMINUSMAX__NEON_P5_X20_ACC2, elements_eq_20) {
    TEST_REQUIRES_ARM_NEON;
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(20)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__neon_p5_x20_acc2);
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEON_P5_X20_ACC2, elements_div_20) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t elements = 40; elements < 200; elements += 20) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neon_p5_x20_acc2);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEON_P5_X20_ACC2, elements_lt_20) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t elements = 1; elements < 20; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neon_p5_x20_acc2);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEON_P5_X20_ACC2, elements_gt_20) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t elements = 21; elements < 40; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neon_p5_x20_acc2);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_RADDSTOREEXPMINUSMAX__NEON_P5_X20_ACC5, elements_eq_20) {
    TEST_REQUIRES_ARM_NEON;
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(20)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__neon_p5_x20_acc5);
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEON_P5_X20_ACC5, elements_div_20) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t elements = 40; elements < 200; elements += 20) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neon_p5_x20_acc5);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEON_P5_X20_ACC5, elements_lt_20) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t elements = 1; elements < 20; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neon_p5_x20_acc5);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEON_P5_X20_ACC5, elements_gt_20) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t elements = 21; elements < 40; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neon_p5_x20_acc5);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_RADDSTOREEXPMINUSMAX__NEON_LUT64_P2_X4, elements_eq_4) {
    TEST_REQUIRES_ARM_NEON;
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(4)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__neon_lut64_p2_x4);
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEON_LUT64_P2_X4, elements_div_4) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t elements = 8; elements < 40; elements += 4) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neon_lut64_p2_x4);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEON_LUT64_P2_X4, elements_lt_4) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t elements = 1; elements < 4; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neon_lut64_p2_x4);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEON_LUT64_P2_X4, elements_gt_4) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t elements = 5; elements < 8; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neon_lut64_p2_x4);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_RADDSTOREEXPMINUSMAX__NEON_LUT64_P2_X8, elements_eq_8) {
    TEST_REQUIRES_ARM_NEON;
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(8)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__neon_lut64_p2_x8);
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEON_LUT64_P2_X8, elements_div_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t elements = 16; elements < 80; elements += 8) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neon_lut64_p2_x8);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEON_LUT64_P2_X8, elements_lt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t elements = 1; elements < 8; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neon_lut64_p2_x8);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEON_LUT64_P2_X8, elements_gt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t elements = 9; elements < 16; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neon_lut64_p2_x8);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_RADDSTOREEXPMINUSMAX__NEON_LUT64_P2_X8_ACC2, elements_eq_8) {
    TEST_REQUIRES_ARM_NEON;
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(8)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__neon_lut64_p2_x8_acc2);
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEON_LUT64_P2_X8_ACC2, elements_div_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t elements = 16; elements < 80; elements += 8) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neon_lut64_p2_x8_acc2);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEON_LUT64_P2_X8_ACC2, elements_lt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t elements = 1; elements < 8; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neon_lut64_p2_x8_acc2);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEON_LUT64_P2_X8_ACC2, elements_gt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t elements = 9; elements < 16; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neon_lut64_p2_x8_acc2);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_RADDSTOREEXPMINUSMAX__NEON_LUT64_P2_X12, elements_eq_12) {
    TEST_REQUIRES_ARM_NEON;
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(12)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__neon_lut64_p2_x12);
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEON_LUT64_P2_X12, elements_div_12) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t elements = 24; elements < 120; elements += 12) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neon_lut64_p2_x12);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEON_LUT64_P2_X12, elements_lt_12) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t elements = 1; elements < 12; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neon_lut64_p2_x12);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEON_LUT64_P2_X12, elements_gt_12) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t elements = 13; elements < 24; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neon_lut64_p2_x12);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_RADDSTOREEXPMINUSMAX__NEON_LUT64_P2_X12_ACC2, elements_eq_12) {
    TEST_REQUIRES_ARM_NEON;
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(12)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__neon_lut64_p2_x12_acc2);
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEON_LUT64_P2_X12_ACC2, elements_div_12) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t elements = 24; elements < 120; elements += 12) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neon_lut64_p2_x12_acc2);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEON_LUT64_P2_X12_ACC2, elements_lt_12) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t elements = 1; elements < 12; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neon_lut64_p2_x12_acc2);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEON_LUT64_P2_X12_ACC2, elements_gt_12) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t elements = 13; elements < 24; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neon_lut64_p2_x12_acc2);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_RADDSTOREEXPMINUSMAX__NEON_LUT64_P2_X12_ACC3, elements_eq_12) {
    TEST_REQUIRES_ARM_NEON;
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(12)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__neon_lut64_p2_x12_acc3);
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEON_LUT64_P2_X12_ACC3, elements_div_12) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t elements = 24; elements < 120; elements += 12) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neon_lut64_p2_x12_acc3);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEON_LUT64_P2_X12_ACC3, elements_lt_12) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t elements = 1; elements < 12; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neon_lut64_p2_x12_acc3);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEON_LUT64_P2_X12_ACC3, elements_gt_12) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t elements = 13; elements < 24; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neon_lut64_p2_x12_acc3);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_RADDSTOREEXPMINUSMAX__NEON_LUT64_P2_X16, elements_eq_16) {
    TEST_REQUIRES_ARM_NEON;
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(16)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__neon_lut64_p2_x16);
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEON_LUT64_P2_X16, elements_div_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t elements = 32; elements < 160; elements += 16) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neon_lut64_p2_x16);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEON_LUT64_P2_X16, elements_lt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t elements = 1; elements < 16; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neon_lut64_p2_x16);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEON_LUT64_P2_X16, elements_gt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t elements = 17; elements < 32; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neon_lut64_p2_x16);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_RADDSTOREEXPMINUSMAX__NEON_LUT64_P2_X16_ACC2, elements_eq_16) {
    TEST_REQUIRES_ARM_NEON;
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(16)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__neon_lut64_p2_x16_acc2);
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEON_LUT64_P2_X16_ACC2, elements_div_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t elements = 32; elements < 160; elements += 16) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neon_lut64_p2_x16_acc2);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEON_LUT64_P2_X16_ACC2, elements_lt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t elements = 1; elements < 16; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neon_lut64_p2_x16_acc2);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEON_LUT64_P2_X16_ACC2, elements_gt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t elements = 17; elements < 32; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neon_lut64_p2_x16_acc2);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_RADDSTOREEXPMINUSMAX__NEON_LUT64_P2_X16_ACC4, elements_eq_16) {
    TEST_REQUIRES_ARM_NEON;
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(16)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__neon_lut64_p2_x16_acc4);
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEON_LUT64_P2_X16_ACC4, elements_div_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t elements = 32; elements < 160; elements += 16) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neon_lut64_p2_x16_acc4);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEON_LUT64_P2_X16_ACC4, elements_lt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t elements = 1; elements < 16; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neon_lut64_p2_x16_acc4);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEON_LUT64_P2_X16_ACC4, elements_gt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t elements = 17; elements < 32; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neon_lut64_p2_x16_acc4);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_RADDSTOREEXPMINUSMAX__NEON_LUT64_P2_X20, elements_eq_20) {
    TEST_REQUIRES_ARM_NEON;
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(20)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__neon_lut64_p2_x20);
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEON_LUT64_P2_X20, elements_div_20) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t elements = 40; elements < 200; elements += 20) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neon_lut64_p2_x20);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEON_LUT64_P2_X20, elements_lt_20) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t elements = 1; elements < 20; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neon_lut64_p2_x20);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEON_LUT64_P2_X20, elements_gt_20) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t elements = 21; elements < 40; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neon_lut64_p2_x20);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_RADDSTOREEXPMINUSMAX__NEON_LUT64_P2_X20_ACC2, elements_eq_20) {
    TEST_REQUIRES_ARM_NEON;
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(20)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__neon_lut64_p2_x20_acc2);
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEON_LUT64_P2_X20_ACC2, elements_div_20) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t elements = 40; elements < 200; elements += 20) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neon_lut64_p2_x20_acc2);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEON_LUT64_P2_X20_ACC2, elements_lt_20) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t elements = 1; elements < 20; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neon_lut64_p2_x20_acc2);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEON_LUT64_P2_X20_ACC2, elements_gt_20) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t elements = 21; elements < 40; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neon_lut64_p2_x20_acc2);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_RADDSTOREEXPMINUSMAX__NEON_LUT64_P2_X20_ACC5, elements_eq_20) {
    TEST_REQUIRES_ARM_NEON;
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(20)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__neon_lut64_p2_x20_acc5);
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEON_LUT64_P2_X20_ACC5, elements_div_20) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t elements = 40; elements < 200; elements += 20) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neon_lut64_p2_x20_acc5);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEON_LUT64_P2_X20_ACC5, elements_lt_20) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t elements = 1; elements < 20; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neon_lut64_p2_x20_acc5);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEON_LUT64_P2_X20_ACC5, elements_gt_20) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t elements = 21; elements < 40; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neon_lut64_p2_x20_acc5);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_RADDSTOREEXPMINUSMAX__NEONFMA_P5_X4, elements_eq_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(4)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__neonfma_p5_x4);
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEONFMA_P5_X4, elements_div_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t elements = 8; elements < 40; elements += 4) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neonfma_p5_x4);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEONFMA_P5_X4, elements_lt_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t elements = 1; elements < 4; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neonfma_p5_x4);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEONFMA_P5_X4, elements_gt_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t elements = 5; elements < 8; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neonfma_p5_x4);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_RADDSTOREEXPMINUSMAX__NEONFMA_P5_X8, elements_eq_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(8)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__neonfma_p5_x8);
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEONFMA_P5_X8, elements_div_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t elements = 16; elements < 80; elements += 8) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neonfma_p5_x8);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEONFMA_P5_X8, elements_lt_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t elements = 1; elements < 8; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neonfma_p5_x8);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEONFMA_P5_X8, elements_gt_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t elements = 9; elements < 16; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neonfma_p5_x8);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_RADDSTOREEXPMINUSMAX__NEONFMA_P5_X8_ACC2, elements_eq_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(8)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__neonfma_p5_x8_acc2);
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEONFMA_P5_X8_ACC2, elements_div_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t elements = 16; elements < 80; elements += 8) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neonfma_p5_x8_acc2);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEONFMA_P5_X8_ACC2, elements_lt_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t elements = 1; elements < 8; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neonfma_p5_x8_acc2);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEONFMA_P5_X8_ACC2, elements_gt_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t elements = 9; elements < 16; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neonfma_p5_x8_acc2);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_RADDSTOREEXPMINUSMAX__NEONFMA_P5_X12, elements_eq_12) {
    TEST_REQUIRES_ARM_NEON_FMA;
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(12)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__neonfma_p5_x12);
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEONFMA_P5_X12, elements_div_12) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t elements = 24; elements < 120; elements += 12) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neonfma_p5_x12);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEONFMA_P5_X12, elements_lt_12) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t elements = 1; elements < 12; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neonfma_p5_x12);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEONFMA_P5_X12, elements_gt_12) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t elements = 13; elements < 24; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neonfma_p5_x12);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_RADDSTOREEXPMINUSMAX__NEONFMA_P5_X12_ACC2, elements_eq_12) {
    TEST_REQUIRES_ARM_NEON_FMA;
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(12)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__neonfma_p5_x12_acc2);
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEONFMA_P5_X12_ACC2, elements_div_12) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t elements = 24; elements < 120; elements += 12) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neonfma_p5_x12_acc2);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEONFMA_P5_X12_ACC2, elements_lt_12) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t elements = 1; elements < 12; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neonfma_p5_x12_acc2);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEONFMA_P5_X12_ACC2, elements_gt_12) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t elements = 13; elements < 24; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neonfma_p5_x12_acc2);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_RADDSTOREEXPMINUSMAX__NEONFMA_P5_X12_ACC3, elements_eq_12) {
    TEST_REQUIRES_ARM_NEON_FMA;
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(12)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__neonfma_p5_x12_acc3);
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEONFMA_P5_X12_ACC3, elements_div_12) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t elements = 24; elements < 120; elements += 12) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neonfma_p5_x12_acc3);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEONFMA_P5_X12_ACC3, elements_lt_12) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t elements = 1; elements < 12; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neonfma_p5_x12_acc3);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEONFMA_P5_X12_ACC3, elements_gt_12) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t elements = 13; elements < 24; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neonfma_p5_x12_acc3);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_RADDSTOREEXPMINUSMAX__NEONFMA_P5_X16, elements_eq_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(16)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__neonfma_p5_x16);
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEONFMA_P5_X16, elements_div_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t elements = 32; elements < 160; elements += 16) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neonfma_p5_x16);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEONFMA_P5_X16, elements_lt_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t elements = 1; elements < 16; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neonfma_p5_x16);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEONFMA_P5_X16, elements_gt_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t elements = 17; elements < 32; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neonfma_p5_x16);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_RADDSTOREEXPMINUSMAX__NEONFMA_P5_X16_ACC2, elements_eq_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(16)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__neonfma_p5_x16_acc2);
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEONFMA_P5_X16_ACC2, elements_div_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t elements = 32; elements < 160; elements += 16) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neonfma_p5_x16_acc2);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEONFMA_P5_X16_ACC2, elements_lt_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t elements = 1; elements < 16; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neonfma_p5_x16_acc2);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEONFMA_P5_X16_ACC2, elements_gt_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t elements = 17; elements < 32; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neonfma_p5_x16_acc2);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_RADDSTOREEXPMINUSMAX__NEONFMA_P5_X16_ACC4, elements_eq_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(16)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__neonfma_p5_x16_acc4);
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEONFMA_P5_X16_ACC4, elements_div_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t elements = 32; elements < 160; elements += 16) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neonfma_p5_x16_acc4);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEONFMA_P5_X16_ACC4, elements_lt_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t elements = 1; elements < 16; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neonfma_p5_x16_acc4);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEONFMA_P5_X16_ACC4, elements_gt_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t elements = 17; elements < 32; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neonfma_p5_x16_acc4);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_RADDSTOREEXPMINUSMAX__NEONFMA_P5_X20, elements_eq_20) {
    TEST_REQUIRES_ARM_NEON_FMA;
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(20)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__neonfma_p5_x20);
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEONFMA_P5_X20, elements_div_20) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t elements = 40; elements < 200; elements += 20) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neonfma_p5_x20);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEONFMA_P5_X20, elements_lt_20) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t elements = 1; elements < 20; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neonfma_p5_x20);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEONFMA_P5_X20, elements_gt_20) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t elements = 21; elements < 40; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neonfma_p5_x20);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_RADDSTOREEXPMINUSMAX__NEONFMA_P5_X20_ACC2, elements_eq_20) {
    TEST_REQUIRES_ARM_NEON_FMA;
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(20)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__neonfma_p5_x20_acc2);
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEONFMA_P5_X20_ACC2, elements_div_20) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t elements = 40; elements < 200; elements += 20) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neonfma_p5_x20_acc2);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEONFMA_P5_X20_ACC2, elements_lt_20) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t elements = 1; elements < 20; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neonfma_p5_x20_acc2);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEONFMA_P5_X20_ACC2, elements_gt_20) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t elements = 21; elements < 40; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neonfma_p5_x20_acc2);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_RADDSTOREEXPMINUSMAX__NEONFMA_P5_X20_ACC5, elements_eq_20) {
    TEST_REQUIRES_ARM_NEON_FMA;
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(20)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__neonfma_p5_x20_acc5);
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEONFMA_P5_X20_ACC5, elements_div_20) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t elements = 40; elements < 200; elements += 20) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neonfma_p5_x20_acc5);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEONFMA_P5_X20_ACC5, elements_lt_20) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t elements = 1; elements < 20; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neonfma_p5_x20_acc5);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEONFMA_P5_X20_ACC5, elements_gt_20) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t elements = 21; elements < 40; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neonfma_p5_x20_acc5);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_RADDSTOREEXPMINUSMAX__NEONFMA_LUT64_P2_X4, elements_eq_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(4)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__neonfma_lut64_p2_x4);
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEONFMA_LUT64_P2_X4, elements_div_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t elements = 8; elements < 40; elements += 4) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neonfma_lut64_p2_x4);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEONFMA_LUT64_P2_X4, elements_lt_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t elements = 1; elements < 4; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neonfma_lut64_p2_x4);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEONFMA_LUT64_P2_X4, elements_gt_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t elements = 5; elements < 8; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neonfma_lut64_p2_x4);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_RADDSTOREEXPMINUSMAX__NEONFMA_LUT64_P2_X8, elements_eq_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(8)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__neonfma_lut64_p2_x8);
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEONFMA_LUT64_P2_X8, elements_div_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t elements = 16; elements < 80; elements += 8) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neonfma_lut64_p2_x8);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEONFMA_LUT64_P2_X8, elements_lt_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t elements = 1; elements < 8; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neonfma_lut64_p2_x8);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEONFMA_LUT64_P2_X8, elements_gt_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t elements = 9; elements < 16; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neonfma_lut64_p2_x8);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_RADDSTOREEXPMINUSMAX__NEONFMA_LUT64_P2_X8_ACC2, elements_eq_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(8)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__neonfma_lut64_p2_x8_acc2);
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEONFMA_LUT64_P2_X8_ACC2, elements_div_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t elements = 16; elements < 80; elements += 8) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neonfma_lut64_p2_x8_acc2);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEONFMA_LUT64_P2_X8_ACC2, elements_lt_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t elements = 1; elements < 8; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neonfma_lut64_p2_x8_acc2);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEONFMA_LUT64_P2_X8_ACC2, elements_gt_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t elements = 9; elements < 16; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neonfma_lut64_p2_x8_acc2);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_RADDSTOREEXPMINUSMAX__NEONFMA_LUT64_P2_X12, elements_eq_12) {
    TEST_REQUIRES_ARM_NEON_FMA;
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(12)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__neonfma_lut64_p2_x12);
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEONFMA_LUT64_P2_X12, elements_div_12) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t elements = 24; elements < 120; elements += 12) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neonfma_lut64_p2_x12);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEONFMA_LUT64_P2_X12, elements_lt_12) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t elements = 1; elements < 12; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neonfma_lut64_p2_x12);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEONFMA_LUT64_P2_X12, elements_gt_12) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t elements = 13; elements < 24; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neonfma_lut64_p2_x12);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_RADDSTOREEXPMINUSMAX__NEONFMA_LUT64_P2_X12_ACC2, elements_eq_12) {
    TEST_REQUIRES_ARM_NEON_FMA;
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(12)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__neonfma_lut64_p2_x12_acc2);
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEONFMA_LUT64_P2_X12_ACC2, elements_div_12) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t elements = 24; elements < 120; elements += 12) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neonfma_lut64_p2_x12_acc2);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEONFMA_LUT64_P2_X12_ACC2, elements_lt_12) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t elements = 1; elements < 12; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neonfma_lut64_p2_x12_acc2);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEONFMA_LUT64_P2_X12_ACC2, elements_gt_12) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t elements = 13; elements < 24; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neonfma_lut64_p2_x12_acc2);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_RADDSTOREEXPMINUSMAX__NEONFMA_LUT64_P2_X12_ACC3, elements_eq_12) {
    TEST_REQUIRES_ARM_NEON_FMA;
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(12)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__neonfma_lut64_p2_x12_acc3);
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEONFMA_LUT64_P2_X12_ACC3, elements_div_12) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t elements = 24; elements < 120; elements += 12) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neonfma_lut64_p2_x12_acc3);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEONFMA_LUT64_P2_X12_ACC3, elements_lt_12) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t elements = 1; elements < 12; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neonfma_lut64_p2_x12_acc3);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEONFMA_LUT64_P2_X12_ACC3, elements_gt_12) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t elements = 13; elements < 24; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neonfma_lut64_p2_x12_acc3);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_RADDSTOREEXPMINUSMAX__NEONFMA_LUT64_P2_X16, elements_eq_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(16)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__neonfma_lut64_p2_x16);
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEONFMA_LUT64_P2_X16, elements_div_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t elements = 32; elements < 160; elements += 16) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neonfma_lut64_p2_x16);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEONFMA_LUT64_P2_X16, elements_lt_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t elements = 1; elements < 16; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neonfma_lut64_p2_x16);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEONFMA_LUT64_P2_X16, elements_gt_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t elements = 17; elements < 32; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neonfma_lut64_p2_x16);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_RADDSTOREEXPMINUSMAX__NEONFMA_LUT64_P2_X16_ACC2, elements_eq_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(16)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__neonfma_lut64_p2_x16_acc2);
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEONFMA_LUT64_P2_X16_ACC2, elements_div_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t elements = 32; elements < 160; elements += 16) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neonfma_lut64_p2_x16_acc2);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEONFMA_LUT64_P2_X16_ACC2, elements_lt_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t elements = 1; elements < 16; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neonfma_lut64_p2_x16_acc2);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEONFMA_LUT64_P2_X16_ACC2, elements_gt_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t elements = 17; elements < 32; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neonfma_lut64_p2_x16_acc2);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_RADDSTOREEXPMINUSMAX__NEONFMA_LUT64_P2_X16_ACC4, elements_eq_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(16)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__neonfma_lut64_p2_x16_acc4);
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEONFMA_LUT64_P2_X16_ACC4, elements_div_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t elements = 32; elements < 160; elements += 16) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neonfma_lut64_p2_x16_acc4);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEONFMA_LUT64_P2_X16_ACC4, elements_lt_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t elements = 1; elements < 16; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neonfma_lut64_p2_x16_acc4);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEONFMA_LUT64_P2_X16_ACC4, elements_gt_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t elements = 17; elements < 32; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neonfma_lut64_p2_x16_acc4);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_RADDSTOREEXPMINUSMAX__NEONFMA_LUT64_P2_X20, elements_eq_20) {
    TEST_REQUIRES_ARM_NEON_FMA;
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(20)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__neonfma_lut64_p2_x20);
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEONFMA_LUT64_P2_X20, elements_div_20) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t elements = 40; elements < 200; elements += 20) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neonfma_lut64_p2_x20);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEONFMA_LUT64_P2_X20, elements_lt_20) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t elements = 1; elements < 20; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neonfma_lut64_p2_x20);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEONFMA_LUT64_P2_X20, elements_gt_20) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t elements = 21; elements < 40; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neonfma_lut64_p2_x20);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_RADDSTOREEXPMINUSMAX__NEONFMA_LUT64_P2_X20_ACC2, elements_eq_20) {
    TEST_REQUIRES_ARM_NEON_FMA;
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(20)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__neonfma_lut64_p2_x20_acc2);
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEONFMA_LUT64_P2_X20_ACC2, elements_div_20) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t elements = 40; elements < 200; elements += 20) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neonfma_lut64_p2_x20_acc2);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEONFMA_LUT64_P2_X20_ACC2, elements_lt_20) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t elements = 1; elements < 20; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neonfma_lut64_p2_x20_acc2);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEONFMA_LUT64_P2_X20_ACC2, elements_gt_20) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t elements = 21; elements < 40; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neonfma_lut64_p2_x20_acc2);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_RADDSTOREEXPMINUSMAX__NEONFMA_LUT64_P2_X20_ACC5, elements_eq_20) {
    TEST_REQUIRES_ARM_NEON_FMA;
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(20)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__neonfma_lut64_p2_x20_acc5);
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEONFMA_LUT64_P2_X20_ACC5, elements_div_20) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t elements = 40; elements < 200; elements += 20) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neonfma_lut64_p2_x20_acc5);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEONFMA_LUT64_P2_X20_ACC5, elements_lt_20) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t elements = 1; elements < 20; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neonfma_lut64_p2_x20_acc5);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEONFMA_LUT64_P2_X20_ACC5, elements_gt_20) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t elements = 21; elements < 40; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neonfma_lut64_p2_x20_acc5);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_RADDSTOREEXPMINUSMAX__SSE2_P5_X4, elements_eq_4) {
    TEST_REQUIRES_X86_SSE2;
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(4)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__sse2_p5_x4);
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__SSE2_P5_X4, elements_div_4) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t elements = 8; elements < 40; elements += 4) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__sse2_p5_x4);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__SSE2_P5_X4, elements_lt_4) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t elements = 1; elements < 4; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__sse2_p5_x4);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__SSE2_P5_X4, elements_gt_4) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t elements = 5; elements < 8; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__sse2_p5_x4);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_RADDSTOREEXPMINUSMAX__SSE2_P5_X8, elements_eq_8) {
    TEST_REQUIRES_X86_SSE2;
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(8)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__sse2_p5_x8);
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__SSE2_P5_X8, elements_div_8) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t elements = 16; elements < 80; elements += 8) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__sse2_p5_x8);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__SSE2_P5_X8, elements_lt_8) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t elements = 1; elements < 8; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__sse2_p5_x8);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__SSE2_P5_X8, elements_gt_8) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t elements = 9; elements < 16; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__sse2_p5_x8);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_RADDSTOREEXPMINUSMAX__SSE2_P5_X8_ACC2, elements_eq_8) {
    TEST_REQUIRES_X86_SSE2;
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(8)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__sse2_p5_x8_acc2);
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__SSE2_P5_X8_ACC2, elements_div_8) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t elements = 16; elements < 80; elements += 8) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__sse2_p5_x8_acc2);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__SSE2_P5_X8_ACC2, elements_lt_8) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t elements = 1; elements < 8; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__sse2_p5_x8_acc2);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__SSE2_P5_X8_ACC2, elements_gt_8) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t elements = 9; elements < 16; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__sse2_p5_x8_acc2);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_RADDSTOREEXPMINUSMAX__SSE2_P5_X12, elements_eq_12) {
    TEST_REQUIRES_X86_SSE2;
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(12)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__sse2_p5_x12);
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__SSE2_P5_X12, elements_div_12) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t elements = 24; elements < 120; elements += 12) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__sse2_p5_x12);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__SSE2_P5_X12, elements_lt_12) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t elements = 1; elements < 12; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__sse2_p5_x12);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__SSE2_P5_X12, elements_gt_12) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t elements = 13; elements < 24; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__sse2_p5_x12);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_RADDSTOREEXPMINUSMAX__SSE2_P5_X12_ACC2, elements_eq_12) {
    TEST_REQUIRES_X86_SSE2;
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(12)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__sse2_p5_x12_acc2);
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__SSE2_P5_X12_ACC2, elements_div_12) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t elements = 24; elements < 120; elements += 12) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__sse2_p5_x12_acc2);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__SSE2_P5_X12_ACC2, elements_lt_12) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t elements = 1; elements < 12; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__sse2_p5_x12_acc2);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__SSE2_P5_X12_ACC2, elements_gt_12) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t elements = 13; elements < 24; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__sse2_p5_x12_acc2);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_RADDSTOREEXPMINUSMAX__SSE2_P5_X12_ACC3, elements_eq_12) {
    TEST_REQUIRES_X86_SSE2;
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(12)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__sse2_p5_x12_acc3);
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__SSE2_P5_X12_ACC3, elements_div_12) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t elements = 24; elements < 120; elements += 12) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__sse2_p5_x12_acc3);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__SSE2_P5_X12_ACC3, elements_lt_12) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t elements = 1; elements < 12; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__sse2_p5_x12_acc3);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__SSE2_P5_X12_ACC3, elements_gt_12) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t elements = 13; elements < 24; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__sse2_p5_x12_acc3);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_RADDSTOREEXPMINUSMAX__SSE2_P5_X16, elements_eq_16) {
    TEST_REQUIRES_X86_SSE2;
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(16)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__sse2_p5_x16);
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__SSE2_P5_X16, elements_div_16) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t elements = 32; elements < 160; elements += 16) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__sse2_p5_x16);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__SSE2_P5_X16, elements_lt_16) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t elements = 1; elements < 16; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__sse2_p5_x16);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__SSE2_P5_X16, elements_gt_16) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t elements = 17; elements < 32; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__sse2_p5_x16);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_RADDSTOREEXPMINUSMAX__SSE2_P5_X16_ACC2, elements_eq_16) {
    TEST_REQUIRES_X86_SSE2;
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(16)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__sse2_p5_x16_acc2);
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__SSE2_P5_X16_ACC2, elements_div_16) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t elements = 32; elements < 160; elements += 16) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__sse2_p5_x16_acc2);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__SSE2_P5_X16_ACC2, elements_lt_16) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t elements = 1; elements < 16; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__sse2_p5_x16_acc2);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__SSE2_P5_X16_ACC2, elements_gt_16) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t elements = 17; elements < 32; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__sse2_p5_x16_acc2);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_RADDSTOREEXPMINUSMAX__SSE2_P5_X16_ACC4, elements_eq_16) {
    TEST_REQUIRES_X86_SSE2;
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(16)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__sse2_p5_x16_acc4);
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__SSE2_P5_X16_ACC4, elements_div_16) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t elements = 32; elements < 160; elements += 16) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__sse2_p5_x16_acc4);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__SSE2_P5_X16_ACC4, elements_lt_16) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t elements = 1; elements < 16; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__sse2_p5_x16_acc4);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__SSE2_P5_X16_ACC4, elements_gt_16) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t elements = 17; elements < 32; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__sse2_p5_x16_acc4);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_RADDSTOREEXPMINUSMAX__SSE2_P5_X20, elements_eq_20) {
    TEST_REQUIRES_X86_SSE2;
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(20)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__sse2_p5_x20);
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__SSE2_P5_X20, elements_div_20) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t elements = 40; elements < 200; elements += 20) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__sse2_p5_x20);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__SSE2_P5_X20, elements_lt_20) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t elements = 1; elements < 20; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__sse2_p5_x20);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__SSE2_P5_X20, elements_gt_20) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t elements = 21; elements < 40; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__sse2_p5_x20);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_RADDSTOREEXPMINUSMAX__SSE2_P5_X20_ACC2, elements_eq_20) {
    TEST_REQUIRES_X86_SSE2;
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(20)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__sse2_p5_x20_acc2);
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__SSE2_P5_X20_ACC2, elements_div_20) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t elements = 40; elements < 200; elements += 20) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__sse2_p5_x20_acc2);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__SSE2_P5_X20_ACC2, elements_lt_20) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t elements = 1; elements < 20; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__sse2_p5_x20_acc2);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__SSE2_P5_X20_ACC2, elements_gt_20) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t elements = 21; elements < 40; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__sse2_p5_x20_acc2);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_RADDSTOREEXPMINUSMAX__SSE2_P5_X20_ACC5, elements_eq_20) {
    TEST_REQUIRES_X86_SSE2;
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(20)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__sse2_p5_x20_acc5);
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__SSE2_P5_X20_ACC5, elements_div_20) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t elements = 40; elements < 200; elements += 20) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__sse2_p5_x20_acc5);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__SSE2_P5_X20_ACC5, elements_lt_20) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t elements = 1; elements < 20; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__sse2_p5_x20_acc5);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__SSE2_P5_X20_ACC5, elements_gt_20) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t elements = 21; elements < 40; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__sse2_p5_x20_acc5);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_RADDSTOREEXPMINUSMAX__AVX2_P5_X64, elements_eq_64) {
    TEST_REQUIRES_X86_AVX2;
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(64)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__avx2_p5_x64);
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__AVX2_P5_X64, elements_div_64) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t elements = 128; elements < 640; elements += 64) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__avx2_p5_x64);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__AVX2_P5_X64, elements_lt_64) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t elements = 1; elements < 64; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__avx2_p5_x64);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__AVX2_P5_X64, elements_gt_64) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t elements = 65; elements < 128; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__avx2_p5_x64);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_RADDSTOREEXPMINUSMAX__AVX2_P5_X64_ACC2, elements_eq_64) {
    TEST_REQUIRES_X86_AVX2;
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(64)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__avx2_p5_x64_acc2);
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__AVX2_P5_X64_ACC2, elements_div_64) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t elements = 128; elements < 640; elements += 64) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__avx2_p5_x64_acc2);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__AVX2_P5_X64_ACC2, elements_lt_64) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t elements = 1; elements < 64; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__avx2_p5_x64_acc2);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__AVX2_P5_X64_ACC2, elements_gt_64) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t elements = 65; elements < 128; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__avx2_p5_x64_acc2);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_RADDSTOREEXPMINUSMAX__AVX2_P5_X64_ACC4, elements_eq_64) {
    TEST_REQUIRES_X86_AVX2;
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(64)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__avx2_p5_x64_acc4);
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__AVX2_P5_X64_ACC4, elements_div_64) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t elements = 128; elements < 640; elements += 64) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__avx2_p5_x64_acc4);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__AVX2_P5_X64_ACC4, elements_lt_64) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t elements = 1; elements < 64; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__avx2_p5_x64_acc4);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__AVX2_P5_X64_ACC4, elements_gt_64) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t elements = 65; elements < 128; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__avx2_p5_x64_acc4);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_RADDSTOREEXPMINUSMAX__AVX2_P5_X72, elements_eq_72) {
    TEST_REQUIRES_X86_AVX2;
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(72)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__avx2_p5_x72);
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__AVX2_P5_X72, elements_div_72) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t elements = 144; elements < 720; elements += 72) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__avx2_p5_x72);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__AVX2_P5_X72, elements_lt_72) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t elements = 1; elements < 72; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__avx2_p5_x72);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__AVX2_P5_X72, elements_gt_72) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t elements = 73; elements < 144; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__avx2_p5_x72);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_RADDSTOREEXPMINUSMAX__AVX2_P5_X72_ACC3, elements_eq_72) {
    TEST_REQUIRES_X86_AVX2;
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(72)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__avx2_p5_x72_acc3);
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__AVX2_P5_X72_ACC3, elements_div_72) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t elements = 144; elements < 720; elements += 72) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__avx2_p5_x72_acc3);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__AVX2_P5_X72_ACC3, elements_lt_72) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t elements = 1; elements < 72; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__avx2_p5_x72_acc3);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__AVX2_P5_X72_ACC3, elements_gt_72) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t elements = 73; elements < 144; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__avx2_p5_x72_acc3);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_RADDSTOREEXPMINUSMAX__AVX2_P5_X80, elements_eq_80) {
    TEST_REQUIRES_X86_AVX2;
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(80)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__avx2_p5_x80);
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__AVX2_P5_X80, elements_div_80) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t elements = 160; elements < 800; elements += 80) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__avx2_p5_x80);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__AVX2_P5_X80, elements_lt_80) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t elements = 1; elements < 80; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__avx2_p5_x80);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__AVX2_P5_X80, elements_gt_80) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t elements = 81; elements < 160; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__avx2_p5_x80);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_RADDSTOREEXPMINUSMAX__AVX2_P5_X80_ACC2, elements_eq_80) {
    TEST_REQUIRES_X86_AVX2;
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(80)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__avx2_p5_x80_acc2);
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__AVX2_P5_X80_ACC2, elements_div_80) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t elements = 160; elements < 800; elements += 80) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__avx2_p5_x80_acc2);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__AVX2_P5_X80_ACC2, elements_lt_80) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t elements = 1; elements < 80; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__avx2_p5_x80_acc2);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__AVX2_P5_X80_ACC2, elements_gt_80) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t elements = 81; elements < 160; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__avx2_p5_x80_acc2);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_RADDSTOREEXPMINUSMAX__AVX2_P5_X80_ACC5, elements_eq_80) {
    TEST_REQUIRES_X86_AVX2;
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(80)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__avx2_p5_x80_acc5);
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__AVX2_P5_X80_ACC5, elements_div_80) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t elements = 160; elements < 800; elements += 80) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__avx2_p5_x80_acc5);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__AVX2_P5_X80_ACC5, elements_lt_80) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t elements = 1; elements < 80; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__avx2_p5_x80_acc5);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__AVX2_P5_X80_ACC5, elements_gt_80) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t elements = 81; elements < 160; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__avx2_p5_x80_acc5);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_RADDSTOREEXPMINUSMAX__AVX2_P5_X96, elements_eq_96) {
    TEST_REQUIRES_X86_AVX2;
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(96)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__avx2_p5_x96);
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__AVX2_P5_X96, elements_div_96) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t elements = 192; elements < 960; elements += 96) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__avx2_p5_x96);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__AVX2_P5_X96, elements_lt_96) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t elements = 1; elements < 96; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__avx2_p5_x96);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__AVX2_P5_X96, elements_gt_96) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t elements = 97; elements < 192; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__avx2_p5_x96);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_RADDSTOREEXPMINUSMAX__AVX2_P5_X96_ACC2, elements_eq_96) {
    TEST_REQUIRES_X86_AVX2;
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(96)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__avx2_p5_x96_acc2);
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__AVX2_P5_X96_ACC2, elements_div_96) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t elements = 192; elements < 960; elements += 96) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__avx2_p5_x96_acc2);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__AVX2_P5_X96_ACC2, elements_lt_96) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t elements = 1; elements < 96; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__avx2_p5_x96_acc2);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__AVX2_P5_X96_ACC2, elements_gt_96) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t elements = 97; elements < 192; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__avx2_p5_x96_acc2);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_RADDSTOREEXPMINUSMAX__AVX2_P5_X96_ACC3, elements_eq_96) {
    TEST_REQUIRES_X86_AVX2;
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(96)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__avx2_p5_x96_acc3);
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__AVX2_P5_X96_ACC3, elements_div_96) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t elements = 192; elements < 960; elements += 96) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__avx2_p5_x96_acc3);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__AVX2_P5_X96_ACC3, elements_lt_96) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t elements = 1; elements < 96; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__avx2_p5_x96_acc3);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__AVX2_P5_X96_ACC3, elements_gt_96) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t elements = 97; elements < 192; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__avx2_p5_x96_acc3);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_RADDSTOREEXPMINUSMAX__AVX2_P5_X96_ACC6, elements_eq_96) {
    TEST_REQUIRES_X86_AVX2;
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(96)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__avx2_p5_x96_acc6);
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__AVX2_P5_X96_ACC6, elements_div_96) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t elements = 192; elements < 960; elements += 96) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__avx2_p5_x96_acc6);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__AVX2_P5_X96_ACC6, elements_lt_96) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t elements = 1; elements < 96; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__avx2_p5_x96_acc6);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__AVX2_P5_X96_ACC6, elements_gt_96) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t elements = 97; elements < 192; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__avx2_p5_x96_acc6);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_RADDSTOREEXPMINUSMAX__AVX512F_P5_SCALEF_X128, elements_eq_128) {
    TEST_REQUIRES_X86_AVX512F;
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(128)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__avx512f_p5_scalef_x128);
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__AVX512F_P5_SCALEF_X128, elements_div_128) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t elements = 256; elements < 1280; elements += 128) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__avx512f_p5_scalef_x128);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__AVX512F_P5_SCALEF_X128, elements_lt_128) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t elements = 1; elements < 128; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__avx512f_p5_scalef_x128);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__AVX512F_P5_SCALEF_X128, elements_gt_128) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t elements = 129; elements < 256; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__avx512f_p5_scalef_x128);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_RADDSTOREEXPMINUSMAX__AVX512F_P5_SCALEF_X128_ACC2, elements_eq_128) {
    TEST_REQUIRES_X86_AVX512F;
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(128)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__avx512f_p5_scalef_x128_acc2);
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__AVX512F_P5_SCALEF_X128_ACC2, elements_div_128) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t elements = 256; elements < 1280; elements += 128) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__avx512f_p5_scalef_x128_acc2);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__AVX512F_P5_SCALEF_X128_ACC2, elements_lt_128) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t elements = 1; elements < 128; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__avx512f_p5_scalef_x128_acc2);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__AVX512F_P5_SCALEF_X128_ACC2, elements_gt_128) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t elements = 129; elements < 256; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__avx512f_p5_scalef_x128_acc2);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_RADDSTOREEXPMINUSMAX__AVX512F_P5_SCALEF_X128_ACC4, elements_eq_128) {
    TEST_REQUIRES_X86_AVX512F;
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(128)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__avx512f_p5_scalef_x128_acc4);
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__AVX512F_P5_SCALEF_X128_ACC4, elements_div_128) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t elements = 256; elements < 1280; elements += 128) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__avx512f_p5_scalef_x128_acc4);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__AVX512F_P5_SCALEF_X128_ACC4, elements_lt_128) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t elements = 1; elements < 128; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__avx512f_p5_scalef_x128_acc4);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__AVX512F_P5_SCALEF_X128_ACC4, elements_gt_128) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t elements = 129; elements < 256; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__avx512f_p5_scalef_x128_acc4);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_RADDSTOREEXPMINUSMAX__AVX512F_P5_SCALEF_X144, elements_eq_144) {
    TEST_REQUIRES_X86_AVX512F;
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(144)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__avx512f_p5_scalef_x144);
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__AVX512F_P5_SCALEF_X144, elements_div_144) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t elements = 288; elements < 1440; elements += 144) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__avx512f_p5_scalef_x144);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__AVX512F_P5_SCALEF_X144, elements_lt_144) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t elements = 1; elements < 144; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__avx512f_p5_scalef_x144);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__AVX512F_P5_SCALEF_X144, elements_gt_144) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t elements = 145; elements < 288; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__avx512f_p5_scalef_x144);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_RADDSTOREEXPMINUSMAX__AVX512F_P5_SCALEF_X144_ACC3, elements_eq_144) {
    TEST_REQUIRES_X86_AVX512F;
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(144)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__avx512f_p5_scalef_x144_acc3);
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__AVX512F_P5_SCALEF_X144_ACC3, elements_div_144) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t elements = 288; elements < 1440; elements += 144) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__avx512f_p5_scalef_x144_acc3);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__AVX512F_P5_SCALEF_X144_ACC3, elements_lt_144) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t elements = 1; elements < 144; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__avx512f_p5_scalef_x144_acc3);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__AVX512F_P5_SCALEF_X144_ACC3, elements_gt_144) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t elements = 145; elements < 288; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__avx512f_p5_scalef_x144_acc3);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_RADDSTOREEXPMINUSMAX__AVX512F_P5_SCALEF_X160, elements_eq_160) {
    TEST_REQUIRES_X86_AVX512F;
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(160)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__avx512f_p5_scalef_x160);
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__AVX512F_P5_SCALEF_X160, elements_div_160) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t elements = 320; elements < 1600; elements += 160) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__avx512f_p5_scalef_x160);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__AVX512F_P5_SCALEF_X160, elements_lt_160) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t elements = 1; elements < 160; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__avx512f_p5_scalef_x160);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__AVX512F_P5_SCALEF_X160, elements_gt_160) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t elements = 161; elements < 320; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__avx512f_p5_scalef_x160);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_RADDSTOREEXPMINUSMAX__AVX512F_P5_SCALEF_X160_ACC2, elements_eq_160) {
    TEST_REQUIRES_X86_AVX512F;
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(160)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__avx512f_p5_scalef_x160_acc2);
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__AVX512F_P5_SCALEF_X160_ACC2, elements_div_160) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t elements = 320; elements < 1600; elements += 160) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__avx512f_p5_scalef_x160_acc2);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__AVX512F_P5_SCALEF_X160_ACC2, elements_lt_160) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t elements = 1; elements < 160; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__avx512f_p5_scalef_x160_acc2);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__AVX512F_P5_SCALEF_X160_ACC2, elements_gt_160) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t elements = 161; elements < 320; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__avx512f_p5_scalef_x160_acc2);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_RADDSTOREEXPMINUSMAX__AVX512F_P5_SCALEF_X160_ACC5, elements_eq_160) {
    TEST_REQUIRES_X86_AVX512F;
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(160)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__avx512f_p5_scalef_x160_acc5);
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__AVX512F_P5_SCALEF_X160_ACC5, elements_div_160) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t elements = 320; elements < 1600; elements += 160) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__avx512f_p5_scalef_x160_acc5);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__AVX512F_P5_SCALEF_X160_ACC5, elements_lt_160) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t elements = 1; elements < 160; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__avx512f_p5_scalef_x160_acc5);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__AVX512F_P5_SCALEF_X160_ACC5, elements_gt_160) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t elements = 161; elements < 320; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__avx512f_p5_scalef_x160_acc5);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_RADDSTOREEXPMINUSMAX__AVX512F_P5_SCALEF_X192, elements_eq_192) {
    TEST_REQUIRES_X86_AVX512F;
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(192)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__avx512f_p5_scalef_x192);
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__AVX512F_P5_SCALEF_X192, elements_div_192) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t elements = 384; elements < 1920; elements += 192) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__avx512f_p5_scalef_x192);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__AVX512F_P5_SCALEF_X192, elements_lt_192) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t elements = 1; elements < 192; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__avx512f_p5_scalef_x192);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__AVX512F_P5_SCALEF_X192, elements_gt_192) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t elements = 193; elements < 384; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__avx512f_p5_scalef_x192);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_RADDSTOREEXPMINUSMAX__AVX512F_P5_SCALEF_X192_ACC2, elements_eq_192) {
    TEST_REQUIRES_X86_AVX512F;
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(192)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__avx512f_p5_scalef_x192_acc2);
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__AVX512F_P5_SCALEF_X192_ACC2, elements_div_192) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t elements = 384; elements < 1920; elements += 192) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__avx512f_p5_scalef_x192_acc2);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__AVX512F_P5_SCALEF_X192_ACC2, elements_lt_192) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t elements = 1; elements < 192; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__avx512f_p5_scalef_x192_acc2);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__AVX512F_P5_SCALEF_X192_ACC2, elements_gt_192) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t elements = 193; elements < 384; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__avx512f_p5_scalef_x192_acc2);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_RADDSTOREEXPMINUSMAX__AVX512F_P5_SCALEF_X192_ACC3, elements_eq_192) {
    TEST_REQUIRES_X86_AVX512F;
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(192)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__avx512f_p5_scalef_x192_acc3);
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__AVX512F_P5_SCALEF_X192_ACC3, elements_div_192) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t elements = 384; elements < 1920; elements += 192) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__avx512f_p5_scalef_x192_acc3);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__AVX512F_P5_SCALEF_X192_ACC3, elements_lt_192) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t elements = 1; elements < 192; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__avx512f_p5_scalef_x192_acc3);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__AVX512F_P5_SCALEF_X192_ACC3, elements_gt_192) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t elements = 193; elements < 384; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__avx512f_p5_scalef_x192_acc3);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_RADDSTOREEXPMINUSMAX__AVX512F_P5_SCALEF_X192_ACC6, elements_eq_192) {
    TEST_REQUIRES_X86_AVX512F;
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(192)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__avx512f_p5_scalef_x192_acc6);
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__AVX512F_P5_SCALEF_X192_ACC6, elements_div_192) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t elements = 384; elements < 1920; elements += 192) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__avx512f_p5_scalef_x192_acc6);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__AVX512F_P5_SCALEF_X192_ACC6, elements_lt_192) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t elements = 1; elements < 192; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__avx512f_p5_scalef_x192_acc6);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__AVX512F_P5_SCALEF_X192_ACC6, elements_gt_192) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t elements = 193; elements < 384; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__avx512f_p5_scalef_x192_acc6);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if !XNN_ARCH_ASMJS && !XNN_ARCH_WASM && !XNN_COMPILER_MSVC && !XNN_COMPILER_ICC
  TEST(F32_RADDSTOREEXPMINUSMAX__PSIMD_P5_X4, elements_eq_4) {
    TEST_REQUIRES_PSIMD;
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(4)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__psimd_p5_x4);
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__PSIMD_P5_X4, elements_div_4) {
    TEST_REQUIRES_PSIMD;
    for (size_t elements = 8; elements < 40; elements += 4) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__psimd_p5_x4);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__PSIMD_P5_X4, elements_lt_4) {
    TEST_REQUIRES_PSIMD;
    for (size_t elements = 1; elements < 4; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__psimd_p5_x4);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__PSIMD_P5_X4, elements_gt_4) {
    TEST_REQUIRES_PSIMD;
    for (size_t elements = 5; elements < 8; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__psimd_p5_x4);
    }
  }
#endif  // !XNN_ARCH_ASMJS && !XNN_ARCH_WASM && !XNN_COMPILER_MSVC && !XNN_COMPILER_ICC


#if !XNN_ARCH_ASMJS && !XNN_ARCH_WASM && !XNN_COMPILER_MSVC && !XNN_COMPILER_ICC
  TEST(F32_RADDSTOREEXPMINUSMAX__PSIMD_P5_X8, elements_eq_8) {
    TEST_REQUIRES_PSIMD;
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(8)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__psimd_p5_x8);
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__PSIMD_P5_X8, elements_div_8) {
    TEST_REQUIRES_PSIMD;
    for (size_t elements = 16; elements < 80; elements += 8) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__psimd_p5_x8);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__PSIMD_P5_X8, elements_lt_8) {
    TEST_REQUIRES_PSIMD;
    for (size_t elements = 1; elements < 8; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__psimd_p5_x8);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__PSIMD_P5_X8, elements_gt_8) {
    TEST_REQUIRES_PSIMD;
    for (size_t elements = 9; elements < 16; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__psimd_p5_x8);
    }
  }
#endif  // !XNN_ARCH_ASMJS && !XNN_ARCH_WASM && !XNN_COMPILER_MSVC && !XNN_COMPILER_ICC


#if !XNN_ARCH_ASMJS && !XNN_ARCH_WASM && !XNN_COMPILER_MSVC && !XNN_COMPILER_ICC
  TEST(F32_RADDSTOREEXPMINUSMAX__PSIMD_P5_X8_ACC2, elements_eq_8) {
    TEST_REQUIRES_PSIMD;
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(8)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__psimd_p5_x8_acc2);
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__PSIMD_P5_X8_ACC2, elements_div_8) {
    TEST_REQUIRES_PSIMD;
    for (size_t elements = 16; elements < 80; elements += 8) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__psimd_p5_x8_acc2);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__PSIMD_P5_X8_ACC2, elements_lt_8) {
    TEST_REQUIRES_PSIMD;
    for (size_t elements = 1; elements < 8; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__psimd_p5_x8_acc2);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__PSIMD_P5_X8_ACC2, elements_gt_8) {
    TEST_REQUIRES_PSIMD;
    for (size_t elements = 9; elements < 16; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__psimd_p5_x8_acc2);
    }
  }
#endif  // !XNN_ARCH_ASMJS && !XNN_ARCH_WASM && !XNN_COMPILER_MSVC && !XNN_COMPILER_ICC


#if !XNN_ARCH_ASMJS && !XNN_ARCH_WASM && !XNN_COMPILER_MSVC && !XNN_COMPILER_ICC
  TEST(F32_RADDSTOREEXPMINUSMAX__PSIMD_P5_X12, elements_eq_12) {
    TEST_REQUIRES_PSIMD;
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(12)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__psimd_p5_x12);
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__PSIMD_P5_X12, elements_div_12) {
    TEST_REQUIRES_PSIMD;
    for (size_t elements = 24; elements < 120; elements += 12) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__psimd_p5_x12);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__PSIMD_P5_X12, elements_lt_12) {
    TEST_REQUIRES_PSIMD;
    for (size_t elements = 1; elements < 12; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__psimd_p5_x12);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__PSIMD_P5_X12, elements_gt_12) {
    TEST_REQUIRES_PSIMD;
    for (size_t elements = 13; elements < 24; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__psimd_p5_x12);
    }
  }
#endif  // !XNN_ARCH_ASMJS && !XNN_ARCH_WASM && !XNN_COMPILER_MSVC && !XNN_COMPILER_ICC


#if !XNN_ARCH_ASMJS && !XNN_ARCH_WASM && !XNN_COMPILER_MSVC && !XNN_COMPILER_ICC
  TEST(F32_RADDSTOREEXPMINUSMAX__PSIMD_P5_X12_ACC2, elements_eq_12) {
    TEST_REQUIRES_PSIMD;
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(12)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__psimd_p5_x12_acc2);
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__PSIMD_P5_X12_ACC2, elements_div_12) {
    TEST_REQUIRES_PSIMD;
    for (size_t elements = 24; elements < 120; elements += 12) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__psimd_p5_x12_acc2);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__PSIMD_P5_X12_ACC2, elements_lt_12) {
    TEST_REQUIRES_PSIMD;
    for (size_t elements = 1; elements < 12; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__psimd_p5_x12_acc2);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__PSIMD_P5_X12_ACC2, elements_gt_12) {
    TEST_REQUIRES_PSIMD;
    for (size_t elements = 13; elements < 24; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__psimd_p5_x12_acc2);
    }
  }
#endif  // !XNN_ARCH_ASMJS && !XNN_ARCH_WASM && !XNN_COMPILER_MSVC && !XNN_COMPILER_ICC


#if !XNN_ARCH_ASMJS && !XNN_ARCH_WASM && !XNN_COMPILER_MSVC && !XNN_COMPILER_ICC
  TEST(F32_RADDSTOREEXPMINUSMAX__PSIMD_P5_X12_ACC3, elements_eq_12) {
    TEST_REQUIRES_PSIMD;
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(12)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__psimd_p5_x12_acc3);
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__PSIMD_P5_X12_ACC3, elements_div_12) {
    TEST_REQUIRES_PSIMD;
    for (size_t elements = 24; elements < 120; elements += 12) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__psimd_p5_x12_acc3);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__PSIMD_P5_X12_ACC3, elements_lt_12) {
    TEST_REQUIRES_PSIMD;
    for (size_t elements = 1; elements < 12; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__psimd_p5_x12_acc3);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__PSIMD_P5_X12_ACC3, elements_gt_12) {
    TEST_REQUIRES_PSIMD;
    for (size_t elements = 13; elements < 24; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__psimd_p5_x12_acc3);
    }
  }
#endif  // !XNN_ARCH_ASMJS && !XNN_ARCH_WASM && !XNN_COMPILER_MSVC && !XNN_COMPILER_ICC


#if !XNN_ARCH_ASMJS && !XNN_ARCH_WASM && !XNN_COMPILER_MSVC && !XNN_COMPILER_ICC
  TEST(F32_RADDSTOREEXPMINUSMAX__PSIMD_P5_X16, elements_eq_16) {
    TEST_REQUIRES_PSIMD;
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(16)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__psimd_p5_x16);
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__PSIMD_P5_X16, elements_div_16) {
    TEST_REQUIRES_PSIMD;
    for (size_t elements = 32; elements < 160; elements += 16) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__psimd_p5_x16);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__PSIMD_P5_X16, elements_lt_16) {
    TEST_REQUIRES_PSIMD;
    for (size_t elements = 1; elements < 16; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__psimd_p5_x16);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__PSIMD_P5_X16, elements_gt_16) {
    TEST_REQUIRES_PSIMD;
    for (size_t elements = 17; elements < 32; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__psimd_p5_x16);
    }
  }
#endif  // !XNN_ARCH_ASMJS && !XNN_ARCH_WASM && !XNN_COMPILER_MSVC && !XNN_COMPILER_ICC


#if !XNN_ARCH_ASMJS && !XNN_ARCH_WASM && !XNN_COMPILER_MSVC && !XNN_COMPILER_ICC
  TEST(F32_RADDSTOREEXPMINUSMAX__PSIMD_P5_X16_ACC2, elements_eq_16) {
    TEST_REQUIRES_PSIMD;
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(16)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__psimd_p5_x16_acc2);
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__PSIMD_P5_X16_ACC2, elements_div_16) {
    TEST_REQUIRES_PSIMD;
    for (size_t elements = 32; elements < 160; elements += 16) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__psimd_p5_x16_acc2);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__PSIMD_P5_X16_ACC2, elements_lt_16) {
    TEST_REQUIRES_PSIMD;
    for (size_t elements = 1; elements < 16; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__psimd_p5_x16_acc2);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__PSIMD_P5_X16_ACC2, elements_gt_16) {
    TEST_REQUIRES_PSIMD;
    for (size_t elements = 17; elements < 32; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__psimd_p5_x16_acc2);
    }
  }
#endif  // !XNN_ARCH_ASMJS && !XNN_ARCH_WASM && !XNN_COMPILER_MSVC && !XNN_COMPILER_ICC


#if !XNN_ARCH_ASMJS && !XNN_ARCH_WASM && !XNN_COMPILER_MSVC && !XNN_COMPILER_ICC
  TEST(F32_RADDSTOREEXPMINUSMAX__PSIMD_P5_X16_ACC4, elements_eq_16) {
    TEST_REQUIRES_PSIMD;
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(16)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__psimd_p5_x16_acc4);
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__PSIMD_P5_X16_ACC4, elements_div_16) {
    TEST_REQUIRES_PSIMD;
    for (size_t elements = 32; elements < 160; elements += 16) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__psimd_p5_x16_acc4);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__PSIMD_P5_X16_ACC4, elements_lt_16) {
    TEST_REQUIRES_PSIMD;
    for (size_t elements = 1; elements < 16; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__psimd_p5_x16_acc4);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__PSIMD_P5_X16_ACC4, elements_gt_16) {
    TEST_REQUIRES_PSIMD;
    for (size_t elements = 17; elements < 32; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__psimd_p5_x16_acc4);
    }
  }
#endif  // !XNN_ARCH_ASMJS && !XNN_ARCH_WASM && !XNN_COMPILER_MSVC && !XNN_COMPILER_ICC


#if !XNN_ARCH_ASMJS && !XNN_ARCH_WASM && !XNN_COMPILER_MSVC && !XNN_COMPILER_ICC
  TEST(F32_RADDSTOREEXPMINUSMAX__PSIMD_P5_X20, elements_eq_20) {
    TEST_REQUIRES_PSIMD;
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(20)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__psimd_p5_x20);
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__PSIMD_P5_X20, elements_div_20) {
    TEST_REQUIRES_PSIMD;
    for (size_t elements = 40; elements < 200; elements += 20) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__psimd_p5_x20);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__PSIMD_P5_X20, elements_lt_20) {
    TEST_REQUIRES_PSIMD;
    for (size_t elements = 1; elements < 20; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__psimd_p5_x20);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__PSIMD_P5_X20, elements_gt_20) {
    TEST_REQUIRES_PSIMD;
    for (size_t elements = 21; elements < 40; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__psimd_p5_x20);
    }
  }
#endif  // !XNN_ARCH_ASMJS && !XNN_ARCH_WASM && !XNN_COMPILER_MSVC && !XNN_COMPILER_ICC


#if !XNN_ARCH_ASMJS && !XNN_ARCH_WASM && !XNN_COMPILER_MSVC && !XNN_COMPILER_ICC
  TEST(F32_RADDSTOREEXPMINUSMAX__PSIMD_P5_X20_ACC2, elements_eq_20) {
    TEST_REQUIRES_PSIMD;
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(20)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__psimd_p5_x20_acc2);
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__PSIMD_P5_X20_ACC2, elements_div_20) {
    TEST_REQUIRES_PSIMD;
    for (size_t elements = 40; elements < 200; elements += 20) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__psimd_p5_x20_acc2);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__PSIMD_P5_X20_ACC2, elements_lt_20) {
    TEST_REQUIRES_PSIMD;
    for (size_t elements = 1; elements < 20; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__psimd_p5_x20_acc2);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__PSIMD_P5_X20_ACC2, elements_gt_20) {
    TEST_REQUIRES_PSIMD;
    for (size_t elements = 21; elements < 40; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__psimd_p5_x20_acc2);
    }
  }
#endif  // !XNN_ARCH_ASMJS && !XNN_ARCH_WASM && !XNN_COMPILER_MSVC && !XNN_COMPILER_ICC


#if !XNN_ARCH_ASMJS && !XNN_ARCH_WASM && !XNN_COMPILER_MSVC && !XNN_COMPILER_ICC
  TEST(F32_RADDSTOREEXPMINUSMAX__PSIMD_P5_X20_ACC5, elements_eq_20) {
    TEST_REQUIRES_PSIMD;
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(20)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__psimd_p5_x20_acc5);
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__PSIMD_P5_X20_ACC5, elements_div_20) {
    TEST_REQUIRES_PSIMD;
    for (size_t elements = 40; elements < 200; elements += 20) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__psimd_p5_x20_acc5);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__PSIMD_P5_X20_ACC5, elements_lt_20) {
    TEST_REQUIRES_PSIMD;
    for (size_t elements = 1; elements < 20; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__psimd_p5_x20_acc5);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__PSIMD_P5_X20_ACC5, elements_gt_20) {
    TEST_REQUIRES_PSIMD;
    for (size_t elements = 21; elements < 40; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__psimd_p5_x20_acc5);
    }
  }
#endif  // !XNN_ARCH_ASMJS && !XNN_ARCH_WASM && !XNN_COMPILER_MSVC && !XNN_COMPILER_ICC


TEST(F32_RADDSTOREEXPMINUSMAX__SCALAR_P5_X1, elements_eq_1) {
  RAddStoreExpMinusMaxMicrokernelTester()
    .elements(1)
    .Test(xnn_f32_raddstoreexpminusmax_ukernel__scalar_p5_x1);
}

TEST(F32_RADDSTOREEXPMINUSMAX__SCALAR_P5_X1, elements_gt_1) {
  for (size_t elements = 2; elements < 10; elements++) {
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(elements)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__scalar_p5_x1);
  }
}

TEST(F32_RADDSTOREEXPMINUSMAX__SCALAR_P5_X2, elements_eq_2) {
  RAddStoreExpMinusMaxMicrokernelTester()
    .elements(2)
    .Test(xnn_f32_raddstoreexpminusmax_ukernel__scalar_p5_x2);
}

TEST(F32_RADDSTOREEXPMINUSMAX__SCALAR_P5_X2, elements_div_2) {
  for (size_t elements = 4; elements < 20; elements += 2) {
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(elements)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__scalar_p5_x2);
  }
}

TEST(F32_RADDSTOREEXPMINUSMAX__SCALAR_P5_X2, elements_lt_2) {
  for (size_t elements = 1; elements < 2; elements++) {
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(elements)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__scalar_p5_x2);
  }
}

TEST(F32_RADDSTOREEXPMINUSMAX__SCALAR_P5_X2, elements_gt_2) {
  for (size_t elements = 3; elements < 4; elements++) {
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(elements)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__scalar_p5_x2);
  }
}

TEST(F32_RADDSTOREEXPMINUSMAX__SCALAR_P5_X2_ACC2, elements_eq_2) {
  RAddStoreExpMinusMaxMicrokernelTester()
    .elements(2)
    .Test(xnn_f32_raddstoreexpminusmax_ukernel__scalar_p5_x2_acc2);
}

TEST(F32_RADDSTOREEXPMINUSMAX__SCALAR_P5_X2_ACC2, elements_div_2) {
  for (size_t elements = 4; elements < 20; elements += 2) {
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(elements)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__scalar_p5_x2_acc2);
  }
}

TEST(F32_RADDSTOREEXPMINUSMAX__SCALAR_P5_X2_ACC2, elements_lt_2) {
  for (size_t elements = 1; elements < 2; elements++) {
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(elements)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__scalar_p5_x2_acc2);
  }
}

TEST(F32_RADDSTOREEXPMINUSMAX__SCALAR_P5_X2_ACC2, elements_gt_2) {
  for (size_t elements = 3; elements < 4; elements++) {
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(elements)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__scalar_p5_x2_acc2);
  }
}

TEST(F32_RADDSTOREEXPMINUSMAX__SCALAR_P5_X4, elements_eq_4) {
  RAddStoreExpMinusMaxMicrokernelTester()
    .elements(4)
    .Test(xnn_f32_raddstoreexpminusmax_ukernel__scalar_p5_x4);
}

TEST(F32_RADDSTOREEXPMINUSMAX__SCALAR_P5_X4, elements_div_4) {
  for (size_t elements = 8; elements < 40; elements += 4) {
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(elements)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__scalar_p5_x4);
  }
}

TEST(F32_RADDSTOREEXPMINUSMAX__SCALAR_P5_X4, elements_lt_4) {
  for (size_t elements = 1; elements < 4; elements++) {
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(elements)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__scalar_p5_x4);
  }
}

TEST(F32_RADDSTOREEXPMINUSMAX__SCALAR_P5_X4, elements_gt_4) {
  for (size_t elements = 5; elements < 8; elements++) {
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(elements)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__scalar_p5_x4);
  }
}

TEST(F32_RADDSTOREEXPMINUSMAX__SCALAR_P5_X4_ACC2, elements_eq_4) {
  RAddStoreExpMinusMaxMicrokernelTester()
    .elements(4)
    .Test(xnn_f32_raddstoreexpminusmax_ukernel__scalar_p5_x4_acc2);
}

TEST(F32_RADDSTOREEXPMINUSMAX__SCALAR_P5_X4_ACC2, elements_div_4) {
  for (size_t elements = 8; elements < 40; elements += 4) {
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(elements)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__scalar_p5_x4_acc2);
  }
}

TEST(F32_RADDSTOREEXPMINUSMAX__SCALAR_P5_X4_ACC2, elements_lt_4) {
  for (size_t elements = 1; elements < 4; elements++) {
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(elements)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__scalar_p5_x4_acc2);
  }
}

TEST(F32_RADDSTOREEXPMINUSMAX__SCALAR_P5_X4_ACC2, elements_gt_4) {
  for (size_t elements = 5; elements < 8; elements++) {
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(elements)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__scalar_p5_x4_acc2);
  }
}

TEST(F32_RADDSTOREEXPMINUSMAX__SCALAR_P5_X4_ACC4, elements_eq_4) {
  RAddStoreExpMinusMaxMicrokernelTester()
    .elements(4)
    .Test(xnn_f32_raddstoreexpminusmax_ukernel__scalar_p5_x4_acc4);
}

TEST(F32_RADDSTOREEXPMINUSMAX__SCALAR_P5_X4_ACC4, elements_div_4) {
  for (size_t elements = 8; elements < 40; elements += 4) {
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(elements)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__scalar_p5_x4_acc4);
  }
}

TEST(F32_RADDSTOREEXPMINUSMAX__SCALAR_P5_X4_ACC4, elements_lt_4) {
  for (size_t elements = 1; elements < 4; elements++) {
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(elements)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__scalar_p5_x4_acc4);
  }
}

TEST(F32_RADDSTOREEXPMINUSMAX__SCALAR_P5_X4_ACC4, elements_gt_4) {
  for (size_t elements = 5; elements < 8; elements++) {
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(elements)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__scalar_p5_x4_acc4);
  }
}

TEST(F32_RADDSTOREEXPMINUSMAX__SCALAR_LUT64_P2_X1, elements_eq_1) {
  RAddStoreExpMinusMaxMicrokernelTester()
    .elements(1)
    .Test(xnn_f32_raddstoreexpminusmax_ukernel__scalar_lut64_p2_x1);
}

TEST(F32_RADDSTOREEXPMINUSMAX__SCALAR_LUT64_P2_X1, elements_gt_1) {
  for (size_t elements = 2; elements < 10; elements++) {
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(elements)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__scalar_lut64_p2_x1);
  }
}

TEST(F32_RADDSTOREEXPMINUSMAX__SCALAR_LUT64_P2_X2, elements_eq_2) {
  RAddStoreExpMinusMaxMicrokernelTester()
    .elements(2)
    .Test(xnn_f32_raddstoreexpminusmax_ukernel__scalar_lut64_p2_x2);
}

TEST(F32_RADDSTOREEXPMINUSMAX__SCALAR_LUT64_P2_X2, elements_div_2) {
  for (size_t elements = 4; elements < 20; elements += 2) {
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(elements)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__scalar_lut64_p2_x2);
  }
}

TEST(F32_RADDSTOREEXPMINUSMAX__SCALAR_LUT64_P2_X2, elements_lt_2) {
  for (size_t elements = 1; elements < 2; elements++) {
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(elements)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__scalar_lut64_p2_x2);
  }
}

TEST(F32_RADDSTOREEXPMINUSMAX__SCALAR_LUT64_P2_X2, elements_gt_2) {
  for (size_t elements = 3; elements < 4; elements++) {
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(elements)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__scalar_lut64_p2_x2);
  }
}

TEST(F32_RADDSTOREEXPMINUSMAX__SCALAR_LUT64_P2_X2_ACC2, elements_eq_2) {
  RAddStoreExpMinusMaxMicrokernelTester()
    .elements(2)
    .Test(xnn_f32_raddstoreexpminusmax_ukernel__scalar_lut64_p2_x2_acc2);
}

TEST(F32_RADDSTOREEXPMINUSMAX__SCALAR_LUT64_P2_X2_ACC2, elements_div_2) {
  for (size_t elements = 4; elements < 20; elements += 2) {
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(elements)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__scalar_lut64_p2_x2_acc2);
  }
}

TEST(F32_RADDSTOREEXPMINUSMAX__SCALAR_LUT64_P2_X2_ACC2, elements_lt_2) {
  for (size_t elements = 1; elements < 2; elements++) {
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(elements)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__scalar_lut64_p2_x2_acc2);
  }
}

TEST(F32_RADDSTOREEXPMINUSMAX__SCALAR_LUT64_P2_X2_ACC2, elements_gt_2) {
  for (size_t elements = 3; elements < 4; elements++) {
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(elements)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__scalar_lut64_p2_x2_acc2);
  }
}

TEST(F32_RADDSTOREEXPMINUSMAX__SCALAR_LUT64_P2_X4, elements_eq_4) {
  RAddStoreExpMinusMaxMicrokernelTester()
    .elements(4)
    .Test(xnn_f32_raddstoreexpminusmax_ukernel__scalar_lut64_p2_x4);
}

TEST(F32_RADDSTOREEXPMINUSMAX__SCALAR_LUT64_P2_X4, elements_div_4) {
  for (size_t elements = 8; elements < 40; elements += 4) {
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(elements)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__scalar_lut64_p2_x4);
  }
}

TEST(F32_RADDSTOREEXPMINUSMAX__SCALAR_LUT64_P2_X4, elements_lt_4) {
  for (size_t elements = 1; elements < 4; elements++) {
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(elements)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__scalar_lut64_p2_x4);
  }
}

TEST(F32_RADDSTOREEXPMINUSMAX__SCALAR_LUT64_P2_X4, elements_gt_4) {
  for (size_t elements = 5; elements < 8; elements++) {
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(elements)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__scalar_lut64_p2_x4);
  }
}

TEST(F32_RADDSTOREEXPMINUSMAX__SCALAR_LUT64_P2_X4_ACC2, elements_eq_4) {
  RAddStoreExpMinusMaxMicrokernelTester()
    .elements(4)
    .Test(xnn_f32_raddstoreexpminusmax_ukernel__scalar_lut64_p2_x4_acc2);
}

TEST(F32_RADDSTOREEXPMINUSMAX__SCALAR_LUT64_P2_X4_ACC2, elements_div_4) {
  for (size_t elements = 8; elements < 40; elements += 4) {
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(elements)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__scalar_lut64_p2_x4_acc2);
  }
}

TEST(F32_RADDSTOREEXPMINUSMAX__SCALAR_LUT64_P2_X4_ACC2, elements_lt_4) {
  for (size_t elements = 1; elements < 4; elements++) {
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(elements)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__scalar_lut64_p2_x4_acc2);
  }
}

TEST(F32_RADDSTOREEXPMINUSMAX__SCALAR_LUT64_P2_X4_ACC2, elements_gt_4) {
  for (size_t elements = 5; elements < 8; elements++) {
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(elements)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__scalar_lut64_p2_x4_acc2);
  }
}

TEST(F32_RADDSTOREEXPMINUSMAX__SCALAR_LUT64_P2_X4_ACC4, elements_eq_4) {
  RAddStoreExpMinusMaxMicrokernelTester()
    .elements(4)
    .Test(xnn_f32_raddstoreexpminusmax_ukernel__scalar_lut64_p2_x4_acc4);
}

TEST(F32_RADDSTOREEXPMINUSMAX__SCALAR_LUT64_P2_X4_ACC4, elements_div_4) {
  for (size_t elements = 8; elements < 40; elements += 4) {
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(elements)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__scalar_lut64_p2_x4_acc4);
  }
}

TEST(F32_RADDSTOREEXPMINUSMAX__SCALAR_LUT64_P2_X4_ACC4, elements_lt_4) {
  for (size_t elements = 1; elements < 4; elements++) {
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(elements)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__scalar_lut64_p2_x4_acc4);
  }
}

TEST(F32_RADDSTOREEXPMINUSMAX__SCALAR_LUT64_P2_X4_ACC4, elements_gt_4) {
  for (size_t elements = 5; elements < 8; elements++) {
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(elements)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__scalar_lut64_p2_x4_acc4);
  }
}