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
  TEST(F32_RADDSTOREEXPMINUSMAX__NEON_RR2_LUT64_P2_U4, elements_eq_4) {
    TEST_REQUIRES_ARM_NEON;
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(4)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__neon_rr2_lut64_p2_u4, xnn_init_f32_expminus_neon_rr2_lut64_p2_params);
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEON_RR2_LUT64_P2_U4, elements_div_4) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t elements = 8; elements < 40; elements += 4) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neon_rr2_lut64_p2_u4, xnn_init_f32_expminus_neon_rr2_lut64_p2_params);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEON_RR2_LUT64_P2_U4, elements_lt_4) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t elements = 1; elements < 4; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neon_rr2_lut64_p2_u4, xnn_init_f32_expminus_neon_rr2_lut64_p2_params);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEON_RR2_LUT64_P2_U4, elements_gt_4) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t elements = 5; elements < 8; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neon_rr2_lut64_p2_u4, xnn_init_f32_expminus_neon_rr2_lut64_p2_params);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_RADDSTOREEXPMINUSMAX__NEON_RR2_LUT64_P2_U8, elements_eq_8) {
    TEST_REQUIRES_ARM_NEON;
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(8)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__neon_rr2_lut64_p2_u8, xnn_init_f32_expminus_neon_rr2_lut64_p2_params);
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEON_RR2_LUT64_P2_U8, elements_div_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t elements = 16; elements < 80; elements += 8) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neon_rr2_lut64_p2_u8, xnn_init_f32_expminus_neon_rr2_lut64_p2_params);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEON_RR2_LUT64_P2_U8, elements_lt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t elements = 1; elements < 8; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neon_rr2_lut64_p2_u8, xnn_init_f32_expminus_neon_rr2_lut64_p2_params);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEON_RR2_LUT64_P2_U8, elements_gt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t elements = 9; elements < 16; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neon_rr2_lut64_p2_u8, xnn_init_f32_expminus_neon_rr2_lut64_p2_params);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_RADDSTOREEXPMINUSMAX__NEON_RR2_LUT64_P2_U8_ACC2, elements_eq_8) {
    TEST_REQUIRES_ARM_NEON;
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(8)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__neon_rr2_lut64_p2_u8_acc2, xnn_init_f32_expminus_neon_rr2_lut64_p2_params);
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEON_RR2_LUT64_P2_U8_ACC2, elements_div_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t elements = 16; elements < 80; elements += 8) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neon_rr2_lut64_p2_u8_acc2, xnn_init_f32_expminus_neon_rr2_lut64_p2_params);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEON_RR2_LUT64_P2_U8_ACC2, elements_lt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t elements = 1; elements < 8; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neon_rr2_lut64_p2_u8_acc2, xnn_init_f32_expminus_neon_rr2_lut64_p2_params);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEON_RR2_LUT64_P2_U8_ACC2, elements_gt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t elements = 9; elements < 16; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neon_rr2_lut64_p2_u8_acc2, xnn_init_f32_expminus_neon_rr2_lut64_p2_params);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_RADDSTOREEXPMINUSMAX__NEON_RR2_LUT64_P2_U12, elements_eq_12) {
    TEST_REQUIRES_ARM_NEON;
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(12)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__neon_rr2_lut64_p2_u12, xnn_init_f32_expminus_neon_rr2_lut64_p2_params);
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEON_RR2_LUT64_P2_U12, elements_div_12) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t elements = 24; elements < 120; elements += 12) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neon_rr2_lut64_p2_u12, xnn_init_f32_expminus_neon_rr2_lut64_p2_params);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEON_RR2_LUT64_P2_U12, elements_lt_12) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t elements = 1; elements < 12; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neon_rr2_lut64_p2_u12, xnn_init_f32_expminus_neon_rr2_lut64_p2_params);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEON_RR2_LUT64_P2_U12, elements_gt_12) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t elements = 13; elements < 24; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neon_rr2_lut64_p2_u12, xnn_init_f32_expminus_neon_rr2_lut64_p2_params);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_RADDSTOREEXPMINUSMAX__NEON_RR2_LUT64_P2_U12_ACC2, elements_eq_12) {
    TEST_REQUIRES_ARM_NEON;
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(12)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__neon_rr2_lut64_p2_u12_acc2, xnn_init_f32_expminus_neon_rr2_lut64_p2_params);
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEON_RR2_LUT64_P2_U12_ACC2, elements_div_12) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t elements = 24; elements < 120; elements += 12) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neon_rr2_lut64_p2_u12_acc2, xnn_init_f32_expminus_neon_rr2_lut64_p2_params);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEON_RR2_LUT64_P2_U12_ACC2, elements_lt_12) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t elements = 1; elements < 12; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neon_rr2_lut64_p2_u12_acc2, xnn_init_f32_expminus_neon_rr2_lut64_p2_params);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEON_RR2_LUT64_P2_U12_ACC2, elements_gt_12) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t elements = 13; elements < 24; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neon_rr2_lut64_p2_u12_acc2, xnn_init_f32_expminus_neon_rr2_lut64_p2_params);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_RADDSTOREEXPMINUSMAX__NEON_RR2_LUT64_P2_U12_ACC3, elements_eq_12) {
    TEST_REQUIRES_ARM_NEON;
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(12)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__neon_rr2_lut64_p2_u12_acc3, xnn_init_f32_expminus_neon_rr2_lut64_p2_params);
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEON_RR2_LUT64_P2_U12_ACC3, elements_div_12) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t elements = 24; elements < 120; elements += 12) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neon_rr2_lut64_p2_u12_acc3, xnn_init_f32_expminus_neon_rr2_lut64_p2_params);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEON_RR2_LUT64_P2_U12_ACC3, elements_lt_12) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t elements = 1; elements < 12; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neon_rr2_lut64_p2_u12_acc3, xnn_init_f32_expminus_neon_rr2_lut64_p2_params);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEON_RR2_LUT64_P2_U12_ACC3, elements_gt_12) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t elements = 13; elements < 24; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neon_rr2_lut64_p2_u12_acc3, xnn_init_f32_expminus_neon_rr2_lut64_p2_params);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_RADDSTOREEXPMINUSMAX__NEON_RR2_LUT64_P2_U16, elements_eq_16) {
    TEST_REQUIRES_ARM_NEON;
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(16)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__neon_rr2_lut64_p2_u16, xnn_init_f32_expminus_neon_rr2_lut64_p2_params);
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEON_RR2_LUT64_P2_U16, elements_div_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t elements = 32; elements < 160; elements += 16) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neon_rr2_lut64_p2_u16, xnn_init_f32_expminus_neon_rr2_lut64_p2_params);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEON_RR2_LUT64_P2_U16, elements_lt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t elements = 1; elements < 16; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neon_rr2_lut64_p2_u16, xnn_init_f32_expminus_neon_rr2_lut64_p2_params);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEON_RR2_LUT64_P2_U16, elements_gt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t elements = 17; elements < 32; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neon_rr2_lut64_p2_u16, xnn_init_f32_expminus_neon_rr2_lut64_p2_params);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_RADDSTOREEXPMINUSMAX__NEON_RR2_LUT64_P2_U16_ACC2, elements_eq_16) {
    TEST_REQUIRES_ARM_NEON;
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(16)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__neon_rr2_lut64_p2_u16_acc2, xnn_init_f32_expminus_neon_rr2_lut64_p2_params);
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEON_RR2_LUT64_P2_U16_ACC2, elements_div_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t elements = 32; elements < 160; elements += 16) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neon_rr2_lut64_p2_u16_acc2, xnn_init_f32_expminus_neon_rr2_lut64_p2_params);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEON_RR2_LUT64_P2_U16_ACC2, elements_lt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t elements = 1; elements < 16; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neon_rr2_lut64_p2_u16_acc2, xnn_init_f32_expminus_neon_rr2_lut64_p2_params);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEON_RR2_LUT64_P2_U16_ACC2, elements_gt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t elements = 17; elements < 32; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neon_rr2_lut64_p2_u16_acc2, xnn_init_f32_expminus_neon_rr2_lut64_p2_params);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_RADDSTOREEXPMINUSMAX__NEON_RR2_LUT64_P2_U16_ACC4, elements_eq_16) {
    TEST_REQUIRES_ARM_NEON;
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(16)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__neon_rr2_lut64_p2_u16_acc4, xnn_init_f32_expminus_neon_rr2_lut64_p2_params);
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEON_RR2_LUT64_P2_U16_ACC4, elements_div_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t elements = 32; elements < 160; elements += 16) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neon_rr2_lut64_p2_u16_acc4, xnn_init_f32_expminus_neon_rr2_lut64_p2_params);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEON_RR2_LUT64_P2_U16_ACC4, elements_lt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t elements = 1; elements < 16; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neon_rr2_lut64_p2_u16_acc4, xnn_init_f32_expminus_neon_rr2_lut64_p2_params);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEON_RR2_LUT64_P2_U16_ACC4, elements_gt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t elements = 17; elements < 32; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neon_rr2_lut64_p2_u16_acc4, xnn_init_f32_expminus_neon_rr2_lut64_p2_params);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_RADDSTOREEXPMINUSMAX__NEON_RR2_LUT64_P2_U20, elements_eq_20) {
    TEST_REQUIRES_ARM_NEON;
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(20)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__neon_rr2_lut64_p2_u20, xnn_init_f32_expminus_neon_rr2_lut64_p2_params);
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEON_RR2_LUT64_P2_U20, elements_div_20) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t elements = 40; elements < 200; elements += 20) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neon_rr2_lut64_p2_u20, xnn_init_f32_expminus_neon_rr2_lut64_p2_params);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEON_RR2_LUT64_P2_U20, elements_lt_20) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t elements = 1; elements < 20; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neon_rr2_lut64_p2_u20, xnn_init_f32_expminus_neon_rr2_lut64_p2_params);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEON_RR2_LUT64_P2_U20, elements_gt_20) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t elements = 21; elements < 40; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neon_rr2_lut64_p2_u20, xnn_init_f32_expminus_neon_rr2_lut64_p2_params);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_RADDSTOREEXPMINUSMAX__NEON_RR2_LUT64_P2_U20_ACC2, elements_eq_20) {
    TEST_REQUIRES_ARM_NEON;
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(20)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__neon_rr2_lut64_p2_u20_acc2, xnn_init_f32_expminus_neon_rr2_lut64_p2_params);
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEON_RR2_LUT64_P2_U20_ACC2, elements_div_20) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t elements = 40; elements < 200; elements += 20) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neon_rr2_lut64_p2_u20_acc2, xnn_init_f32_expminus_neon_rr2_lut64_p2_params);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEON_RR2_LUT64_P2_U20_ACC2, elements_lt_20) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t elements = 1; elements < 20; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neon_rr2_lut64_p2_u20_acc2, xnn_init_f32_expminus_neon_rr2_lut64_p2_params);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEON_RR2_LUT64_P2_U20_ACC2, elements_gt_20) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t elements = 21; elements < 40; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neon_rr2_lut64_p2_u20_acc2, xnn_init_f32_expminus_neon_rr2_lut64_p2_params);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_RADDSTOREEXPMINUSMAX__NEON_RR2_LUT64_P2_U20_ACC5, elements_eq_20) {
    TEST_REQUIRES_ARM_NEON;
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(20)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__neon_rr2_lut64_p2_u20_acc5, xnn_init_f32_expminus_neon_rr2_lut64_p2_params);
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEON_RR2_LUT64_P2_U20_ACC5, elements_div_20) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t elements = 40; elements < 200; elements += 20) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neon_rr2_lut64_p2_u20_acc5, xnn_init_f32_expminus_neon_rr2_lut64_p2_params);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEON_RR2_LUT64_P2_U20_ACC5, elements_lt_20) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t elements = 1; elements < 20; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neon_rr2_lut64_p2_u20_acc5, xnn_init_f32_expminus_neon_rr2_lut64_p2_params);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEON_RR2_LUT64_P2_U20_ACC5, elements_gt_20) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t elements = 21; elements < 40; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neon_rr2_lut64_p2_u20_acc5, xnn_init_f32_expminus_neon_rr2_lut64_p2_params);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_RADDSTOREEXPMINUSMAX__NEON_RR2_P5_U4, elements_eq_4) {
    TEST_REQUIRES_ARM_NEON;
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(4)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__neon_rr2_p5_u4, xnn_init_f32_expminus_neon_rr2_p5_params);
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEON_RR2_P5_U4, elements_div_4) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t elements = 8; elements < 40; elements += 4) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neon_rr2_p5_u4, xnn_init_f32_expminus_neon_rr2_p5_params);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEON_RR2_P5_U4, elements_lt_4) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t elements = 1; elements < 4; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neon_rr2_p5_u4, xnn_init_f32_expminus_neon_rr2_p5_params);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEON_RR2_P5_U4, elements_gt_4) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t elements = 5; elements < 8; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neon_rr2_p5_u4, xnn_init_f32_expminus_neon_rr2_p5_params);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_RADDSTOREEXPMINUSMAX__NEON_RR2_P5_U8, elements_eq_8) {
    TEST_REQUIRES_ARM_NEON;
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(8)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__neon_rr2_p5_u8, xnn_init_f32_expminus_neon_rr2_p5_params);
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEON_RR2_P5_U8, elements_div_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t elements = 16; elements < 80; elements += 8) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neon_rr2_p5_u8, xnn_init_f32_expminus_neon_rr2_p5_params);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEON_RR2_P5_U8, elements_lt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t elements = 1; elements < 8; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neon_rr2_p5_u8, xnn_init_f32_expminus_neon_rr2_p5_params);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEON_RR2_P5_U8, elements_gt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t elements = 9; elements < 16; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neon_rr2_p5_u8, xnn_init_f32_expminus_neon_rr2_p5_params);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_RADDSTOREEXPMINUSMAX__NEON_RR2_P5_U8_ACC2, elements_eq_8) {
    TEST_REQUIRES_ARM_NEON;
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(8)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__neon_rr2_p5_u8_acc2, xnn_init_f32_expminus_neon_rr2_p5_params);
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEON_RR2_P5_U8_ACC2, elements_div_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t elements = 16; elements < 80; elements += 8) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neon_rr2_p5_u8_acc2, xnn_init_f32_expminus_neon_rr2_p5_params);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEON_RR2_P5_U8_ACC2, elements_lt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t elements = 1; elements < 8; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neon_rr2_p5_u8_acc2, xnn_init_f32_expminus_neon_rr2_p5_params);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEON_RR2_P5_U8_ACC2, elements_gt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t elements = 9; elements < 16; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neon_rr2_p5_u8_acc2, xnn_init_f32_expminus_neon_rr2_p5_params);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_RADDSTOREEXPMINUSMAX__NEON_RR2_P5_U12, elements_eq_12) {
    TEST_REQUIRES_ARM_NEON;
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(12)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__neon_rr2_p5_u12, xnn_init_f32_expminus_neon_rr2_p5_params);
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEON_RR2_P5_U12, elements_div_12) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t elements = 24; elements < 120; elements += 12) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neon_rr2_p5_u12, xnn_init_f32_expminus_neon_rr2_p5_params);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEON_RR2_P5_U12, elements_lt_12) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t elements = 1; elements < 12; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neon_rr2_p5_u12, xnn_init_f32_expminus_neon_rr2_p5_params);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEON_RR2_P5_U12, elements_gt_12) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t elements = 13; elements < 24; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neon_rr2_p5_u12, xnn_init_f32_expminus_neon_rr2_p5_params);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_RADDSTOREEXPMINUSMAX__NEON_RR2_P5_U12_ACC2, elements_eq_12) {
    TEST_REQUIRES_ARM_NEON;
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(12)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__neon_rr2_p5_u12_acc2, xnn_init_f32_expminus_neon_rr2_p5_params);
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEON_RR2_P5_U12_ACC2, elements_div_12) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t elements = 24; elements < 120; elements += 12) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neon_rr2_p5_u12_acc2, xnn_init_f32_expminus_neon_rr2_p5_params);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEON_RR2_P5_U12_ACC2, elements_lt_12) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t elements = 1; elements < 12; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neon_rr2_p5_u12_acc2, xnn_init_f32_expminus_neon_rr2_p5_params);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEON_RR2_P5_U12_ACC2, elements_gt_12) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t elements = 13; elements < 24; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neon_rr2_p5_u12_acc2, xnn_init_f32_expminus_neon_rr2_p5_params);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_RADDSTOREEXPMINUSMAX__NEON_RR2_P5_U12_ACC3, elements_eq_12) {
    TEST_REQUIRES_ARM_NEON;
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(12)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__neon_rr2_p5_u12_acc3, xnn_init_f32_expminus_neon_rr2_p5_params);
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEON_RR2_P5_U12_ACC3, elements_div_12) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t elements = 24; elements < 120; elements += 12) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neon_rr2_p5_u12_acc3, xnn_init_f32_expminus_neon_rr2_p5_params);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEON_RR2_P5_U12_ACC3, elements_lt_12) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t elements = 1; elements < 12; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neon_rr2_p5_u12_acc3, xnn_init_f32_expminus_neon_rr2_p5_params);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEON_RR2_P5_U12_ACC3, elements_gt_12) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t elements = 13; elements < 24; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neon_rr2_p5_u12_acc3, xnn_init_f32_expminus_neon_rr2_p5_params);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_RADDSTOREEXPMINUSMAX__NEON_RR2_P5_U16, elements_eq_16) {
    TEST_REQUIRES_ARM_NEON;
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(16)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__neon_rr2_p5_u16, xnn_init_f32_expminus_neon_rr2_p5_params);
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEON_RR2_P5_U16, elements_div_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t elements = 32; elements < 160; elements += 16) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neon_rr2_p5_u16, xnn_init_f32_expminus_neon_rr2_p5_params);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEON_RR2_P5_U16, elements_lt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t elements = 1; elements < 16; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neon_rr2_p5_u16, xnn_init_f32_expminus_neon_rr2_p5_params);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEON_RR2_P5_U16, elements_gt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t elements = 17; elements < 32; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neon_rr2_p5_u16, xnn_init_f32_expminus_neon_rr2_p5_params);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_RADDSTOREEXPMINUSMAX__NEON_RR2_P5_U16_ACC2, elements_eq_16) {
    TEST_REQUIRES_ARM_NEON;
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(16)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__neon_rr2_p5_u16_acc2, xnn_init_f32_expminus_neon_rr2_p5_params);
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEON_RR2_P5_U16_ACC2, elements_div_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t elements = 32; elements < 160; elements += 16) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neon_rr2_p5_u16_acc2, xnn_init_f32_expminus_neon_rr2_p5_params);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEON_RR2_P5_U16_ACC2, elements_lt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t elements = 1; elements < 16; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neon_rr2_p5_u16_acc2, xnn_init_f32_expminus_neon_rr2_p5_params);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEON_RR2_P5_U16_ACC2, elements_gt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t elements = 17; elements < 32; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neon_rr2_p5_u16_acc2, xnn_init_f32_expminus_neon_rr2_p5_params);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_RADDSTOREEXPMINUSMAX__NEON_RR2_P5_U16_ACC4, elements_eq_16) {
    TEST_REQUIRES_ARM_NEON;
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(16)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__neon_rr2_p5_u16_acc4, xnn_init_f32_expminus_neon_rr2_p5_params);
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEON_RR2_P5_U16_ACC4, elements_div_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t elements = 32; elements < 160; elements += 16) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neon_rr2_p5_u16_acc4, xnn_init_f32_expminus_neon_rr2_p5_params);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEON_RR2_P5_U16_ACC4, elements_lt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t elements = 1; elements < 16; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neon_rr2_p5_u16_acc4, xnn_init_f32_expminus_neon_rr2_p5_params);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEON_RR2_P5_U16_ACC4, elements_gt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t elements = 17; elements < 32; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neon_rr2_p5_u16_acc4, xnn_init_f32_expminus_neon_rr2_p5_params);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_RADDSTOREEXPMINUSMAX__NEON_RR2_P5_U20, elements_eq_20) {
    TEST_REQUIRES_ARM_NEON;
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(20)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__neon_rr2_p5_u20, xnn_init_f32_expminus_neon_rr2_p5_params);
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEON_RR2_P5_U20, elements_div_20) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t elements = 40; elements < 200; elements += 20) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neon_rr2_p5_u20, xnn_init_f32_expminus_neon_rr2_p5_params);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEON_RR2_P5_U20, elements_lt_20) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t elements = 1; elements < 20; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neon_rr2_p5_u20, xnn_init_f32_expminus_neon_rr2_p5_params);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEON_RR2_P5_U20, elements_gt_20) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t elements = 21; elements < 40; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neon_rr2_p5_u20, xnn_init_f32_expminus_neon_rr2_p5_params);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_RADDSTOREEXPMINUSMAX__NEON_RR2_P5_U20_ACC2, elements_eq_20) {
    TEST_REQUIRES_ARM_NEON;
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(20)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__neon_rr2_p5_u20_acc2, xnn_init_f32_expminus_neon_rr2_p5_params);
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEON_RR2_P5_U20_ACC2, elements_div_20) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t elements = 40; elements < 200; elements += 20) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neon_rr2_p5_u20_acc2, xnn_init_f32_expminus_neon_rr2_p5_params);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEON_RR2_P5_U20_ACC2, elements_lt_20) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t elements = 1; elements < 20; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neon_rr2_p5_u20_acc2, xnn_init_f32_expminus_neon_rr2_p5_params);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEON_RR2_P5_U20_ACC2, elements_gt_20) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t elements = 21; elements < 40; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neon_rr2_p5_u20_acc2, xnn_init_f32_expminus_neon_rr2_p5_params);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_RADDSTOREEXPMINUSMAX__NEON_RR2_P5_U20_ACC5, elements_eq_20) {
    TEST_REQUIRES_ARM_NEON;
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(20)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__neon_rr2_p5_u20_acc5, xnn_init_f32_expminus_neon_rr2_p5_params);
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEON_RR2_P5_U20_ACC5, elements_div_20) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t elements = 40; elements < 200; elements += 20) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neon_rr2_p5_u20_acc5, xnn_init_f32_expminus_neon_rr2_p5_params);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEON_RR2_P5_U20_ACC5, elements_lt_20) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t elements = 1; elements < 20; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neon_rr2_p5_u20_acc5, xnn_init_f32_expminus_neon_rr2_p5_params);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEON_RR2_P5_U20_ACC5, elements_gt_20) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t elements = 21; elements < 40; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neon_rr2_p5_u20_acc5, xnn_init_f32_expminus_neon_rr2_p5_params);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_RADDSTOREEXPMINUSMAX__NEONFMA_RR1_LUT64_P2_U4, elements_eq_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(4)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__neonfma_rr1_lut64_p2_u4, xnn_init_f32_expminus_neonfma_rr1_lut64_p2_params);
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEONFMA_RR1_LUT64_P2_U4, elements_div_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t elements = 8; elements < 40; elements += 4) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neonfma_rr1_lut64_p2_u4, xnn_init_f32_expminus_neonfma_rr1_lut64_p2_params);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEONFMA_RR1_LUT64_P2_U4, elements_lt_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t elements = 1; elements < 4; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neonfma_rr1_lut64_p2_u4, xnn_init_f32_expminus_neonfma_rr1_lut64_p2_params);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEONFMA_RR1_LUT64_P2_U4, elements_gt_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t elements = 5; elements < 8; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neonfma_rr1_lut64_p2_u4, xnn_init_f32_expminus_neonfma_rr1_lut64_p2_params);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_RADDSTOREEXPMINUSMAX__NEONFMA_RR1_LUT64_P2_U8, elements_eq_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(8)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__neonfma_rr1_lut64_p2_u8, xnn_init_f32_expminus_neonfma_rr1_lut64_p2_params);
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEONFMA_RR1_LUT64_P2_U8, elements_div_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t elements = 16; elements < 80; elements += 8) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neonfma_rr1_lut64_p2_u8, xnn_init_f32_expminus_neonfma_rr1_lut64_p2_params);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEONFMA_RR1_LUT64_P2_U8, elements_lt_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t elements = 1; elements < 8; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neonfma_rr1_lut64_p2_u8, xnn_init_f32_expminus_neonfma_rr1_lut64_p2_params);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEONFMA_RR1_LUT64_P2_U8, elements_gt_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t elements = 9; elements < 16; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neonfma_rr1_lut64_p2_u8, xnn_init_f32_expminus_neonfma_rr1_lut64_p2_params);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_RADDSTOREEXPMINUSMAX__NEONFMA_RR1_LUT64_P2_U8_ACC2, elements_eq_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(8)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__neonfma_rr1_lut64_p2_u8_acc2, xnn_init_f32_expminus_neonfma_rr1_lut64_p2_params);
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEONFMA_RR1_LUT64_P2_U8_ACC2, elements_div_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t elements = 16; elements < 80; elements += 8) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neonfma_rr1_lut64_p2_u8_acc2, xnn_init_f32_expminus_neonfma_rr1_lut64_p2_params);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEONFMA_RR1_LUT64_P2_U8_ACC2, elements_lt_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t elements = 1; elements < 8; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neonfma_rr1_lut64_p2_u8_acc2, xnn_init_f32_expminus_neonfma_rr1_lut64_p2_params);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEONFMA_RR1_LUT64_P2_U8_ACC2, elements_gt_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t elements = 9; elements < 16; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neonfma_rr1_lut64_p2_u8_acc2, xnn_init_f32_expminus_neonfma_rr1_lut64_p2_params);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_RADDSTOREEXPMINUSMAX__NEONFMA_RR1_LUT64_P2_U12, elements_eq_12) {
    TEST_REQUIRES_ARM_NEON_FMA;
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(12)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__neonfma_rr1_lut64_p2_u12, xnn_init_f32_expminus_neonfma_rr1_lut64_p2_params);
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEONFMA_RR1_LUT64_P2_U12, elements_div_12) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t elements = 24; elements < 120; elements += 12) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neonfma_rr1_lut64_p2_u12, xnn_init_f32_expminus_neonfma_rr1_lut64_p2_params);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEONFMA_RR1_LUT64_P2_U12, elements_lt_12) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t elements = 1; elements < 12; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neonfma_rr1_lut64_p2_u12, xnn_init_f32_expminus_neonfma_rr1_lut64_p2_params);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEONFMA_RR1_LUT64_P2_U12, elements_gt_12) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t elements = 13; elements < 24; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neonfma_rr1_lut64_p2_u12, xnn_init_f32_expminus_neonfma_rr1_lut64_p2_params);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_RADDSTOREEXPMINUSMAX__NEONFMA_RR1_LUT64_P2_U12_ACC2, elements_eq_12) {
    TEST_REQUIRES_ARM_NEON_FMA;
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(12)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__neonfma_rr1_lut64_p2_u12_acc2, xnn_init_f32_expminus_neonfma_rr1_lut64_p2_params);
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEONFMA_RR1_LUT64_P2_U12_ACC2, elements_div_12) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t elements = 24; elements < 120; elements += 12) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neonfma_rr1_lut64_p2_u12_acc2, xnn_init_f32_expminus_neonfma_rr1_lut64_p2_params);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEONFMA_RR1_LUT64_P2_U12_ACC2, elements_lt_12) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t elements = 1; elements < 12; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neonfma_rr1_lut64_p2_u12_acc2, xnn_init_f32_expminus_neonfma_rr1_lut64_p2_params);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEONFMA_RR1_LUT64_P2_U12_ACC2, elements_gt_12) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t elements = 13; elements < 24; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neonfma_rr1_lut64_p2_u12_acc2, xnn_init_f32_expminus_neonfma_rr1_lut64_p2_params);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_RADDSTOREEXPMINUSMAX__NEONFMA_RR1_LUT64_P2_U12_ACC3, elements_eq_12) {
    TEST_REQUIRES_ARM_NEON_FMA;
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(12)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__neonfma_rr1_lut64_p2_u12_acc3, xnn_init_f32_expminus_neonfma_rr1_lut64_p2_params);
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEONFMA_RR1_LUT64_P2_U12_ACC3, elements_div_12) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t elements = 24; elements < 120; elements += 12) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neonfma_rr1_lut64_p2_u12_acc3, xnn_init_f32_expminus_neonfma_rr1_lut64_p2_params);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEONFMA_RR1_LUT64_P2_U12_ACC3, elements_lt_12) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t elements = 1; elements < 12; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neonfma_rr1_lut64_p2_u12_acc3, xnn_init_f32_expminus_neonfma_rr1_lut64_p2_params);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEONFMA_RR1_LUT64_P2_U12_ACC3, elements_gt_12) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t elements = 13; elements < 24; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neonfma_rr1_lut64_p2_u12_acc3, xnn_init_f32_expminus_neonfma_rr1_lut64_p2_params);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_RADDSTOREEXPMINUSMAX__NEONFMA_RR1_LUT64_P2_U16, elements_eq_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(16)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__neonfma_rr1_lut64_p2_u16, xnn_init_f32_expminus_neonfma_rr1_lut64_p2_params);
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEONFMA_RR1_LUT64_P2_U16, elements_div_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t elements = 32; elements < 160; elements += 16) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neonfma_rr1_lut64_p2_u16, xnn_init_f32_expminus_neonfma_rr1_lut64_p2_params);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEONFMA_RR1_LUT64_P2_U16, elements_lt_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t elements = 1; elements < 16; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neonfma_rr1_lut64_p2_u16, xnn_init_f32_expminus_neonfma_rr1_lut64_p2_params);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEONFMA_RR1_LUT64_P2_U16, elements_gt_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t elements = 17; elements < 32; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neonfma_rr1_lut64_p2_u16, xnn_init_f32_expminus_neonfma_rr1_lut64_p2_params);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_RADDSTOREEXPMINUSMAX__NEONFMA_RR1_LUT64_P2_U16_ACC2, elements_eq_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(16)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__neonfma_rr1_lut64_p2_u16_acc2, xnn_init_f32_expminus_neonfma_rr1_lut64_p2_params);
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEONFMA_RR1_LUT64_P2_U16_ACC2, elements_div_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t elements = 32; elements < 160; elements += 16) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neonfma_rr1_lut64_p2_u16_acc2, xnn_init_f32_expminus_neonfma_rr1_lut64_p2_params);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEONFMA_RR1_LUT64_P2_U16_ACC2, elements_lt_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t elements = 1; elements < 16; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neonfma_rr1_lut64_p2_u16_acc2, xnn_init_f32_expminus_neonfma_rr1_lut64_p2_params);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEONFMA_RR1_LUT64_P2_U16_ACC2, elements_gt_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t elements = 17; elements < 32; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neonfma_rr1_lut64_p2_u16_acc2, xnn_init_f32_expminus_neonfma_rr1_lut64_p2_params);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_RADDSTOREEXPMINUSMAX__NEONFMA_RR1_LUT64_P2_U16_ACC4, elements_eq_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(16)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__neonfma_rr1_lut64_p2_u16_acc4, xnn_init_f32_expminus_neonfma_rr1_lut64_p2_params);
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEONFMA_RR1_LUT64_P2_U16_ACC4, elements_div_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t elements = 32; elements < 160; elements += 16) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neonfma_rr1_lut64_p2_u16_acc4, xnn_init_f32_expminus_neonfma_rr1_lut64_p2_params);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEONFMA_RR1_LUT64_P2_U16_ACC4, elements_lt_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t elements = 1; elements < 16; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neonfma_rr1_lut64_p2_u16_acc4, xnn_init_f32_expminus_neonfma_rr1_lut64_p2_params);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEONFMA_RR1_LUT64_P2_U16_ACC4, elements_gt_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t elements = 17; elements < 32; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neonfma_rr1_lut64_p2_u16_acc4, xnn_init_f32_expminus_neonfma_rr1_lut64_p2_params);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_RADDSTOREEXPMINUSMAX__NEONFMA_RR1_LUT64_P2_U20, elements_eq_20) {
    TEST_REQUIRES_ARM_NEON_FMA;
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(20)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__neonfma_rr1_lut64_p2_u20, xnn_init_f32_expminus_neonfma_rr1_lut64_p2_params);
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEONFMA_RR1_LUT64_P2_U20, elements_div_20) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t elements = 40; elements < 200; elements += 20) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neonfma_rr1_lut64_p2_u20, xnn_init_f32_expminus_neonfma_rr1_lut64_p2_params);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEONFMA_RR1_LUT64_P2_U20, elements_lt_20) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t elements = 1; elements < 20; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neonfma_rr1_lut64_p2_u20, xnn_init_f32_expminus_neonfma_rr1_lut64_p2_params);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEONFMA_RR1_LUT64_P2_U20, elements_gt_20) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t elements = 21; elements < 40; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neonfma_rr1_lut64_p2_u20, xnn_init_f32_expminus_neonfma_rr1_lut64_p2_params);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_RADDSTOREEXPMINUSMAX__NEONFMA_RR1_LUT64_P2_U20_ACC2, elements_eq_20) {
    TEST_REQUIRES_ARM_NEON_FMA;
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(20)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__neonfma_rr1_lut64_p2_u20_acc2, xnn_init_f32_expminus_neonfma_rr1_lut64_p2_params);
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEONFMA_RR1_LUT64_P2_U20_ACC2, elements_div_20) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t elements = 40; elements < 200; elements += 20) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neonfma_rr1_lut64_p2_u20_acc2, xnn_init_f32_expminus_neonfma_rr1_lut64_p2_params);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEONFMA_RR1_LUT64_P2_U20_ACC2, elements_lt_20) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t elements = 1; elements < 20; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neonfma_rr1_lut64_p2_u20_acc2, xnn_init_f32_expminus_neonfma_rr1_lut64_p2_params);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEONFMA_RR1_LUT64_P2_U20_ACC2, elements_gt_20) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t elements = 21; elements < 40; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neonfma_rr1_lut64_p2_u20_acc2, xnn_init_f32_expminus_neonfma_rr1_lut64_p2_params);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_RADDSTOREEXPMINUSMAX__NEONFMA_RR1_LUT64_P2_U20_ACC5, elements_eq_20) {
    TEST_REQUIRES_ARM_NEON_FMA;
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(20)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__neonfma_rr1_lut64_p2_u20_acc5, xnn_init_f32_expminus_neonfma_rr1_lut64_p2_params);
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEONFMA_RR1_LUT64_P2_U20_ACC5, elements_div_20) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t elements = 40; elements < 200; elements += 20) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neonfma_rr1_lut64_p2_u20_acc5, xnn_init_f32_expminus_neonfma_rr1_lut64_p2_params);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEONFMA_RR1_LUT64_P2_U20_ACC5, elements_lt_20) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t elements = 1; elements < 20; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neonfma_rr1_lut64_p2_u20_acc5, xnn_init_f32_expminus_neonfma_rr1_lut64_p2_params);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEONFMA_RR1_LUT64_P2_U20_ACC5, elements_gt_20) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t elements = 21; elements < 40; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neonfma_rr1_lut64_p2_u20_acc5, xnn_init_f32_expminus_neonfma_rr1_lut64_p2_params);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_RADDSTOREEXPMINUSMAX__NEONFMA_RR1_P5_U4, elements_eq_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(4)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__neonfma_rr1_p5_u4, xnn_init_f32_expminus_neonfma_rr1_p5_params);
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEONFMA_RR1_P5_U4, elements_div_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t elements = 8; elements < 40; elements += 4) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neonfma_rr1_p5_u4, xnn_init_f32_expminus_neonfma_rr1_p5_params);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEONFMA_RR1_P5_U4, elements_lt_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t elements = 1; elements < 4; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neonfma_rr1_p5_u4, xnn_init_f32_expminus_neonfma_rr1_p5_params);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEONFMA_RR1_P5_U4, elements_gt_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t elements = 5; elements < 8; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neonfma_rr1_p5_u4, xnn_init_f32_expminus_neonfma_rr1_p5_params);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_RADDSTOREEXPMINUSMAX__NEONFMA_RR1_P5_U8, elements_eq_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(8)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__neonfma_rr1_p5_u8, xnn_init_f32_expminus_neonfma_rr1_p5_params);
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEONFMA_RR1_P5_U8, elements_div_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t elements = 16; elements < 80; elements += 8) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neonfma_rr1_p5_u8, xnn_init_f32_expminus_neonfma_rr1_p5_params);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEONFMA_RR1_P5_U8, elements_lt_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t elements = 1; elements < 8; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neonfma_rr1_p5_u8, xnn_init_f32_expminus_neonfma_rr1_p5_params);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEONFMA_RR1_P5_U8, elements_gt_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t elements = 9; elements < 16; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neonfma_rr1_p5_u8, xnn_init_f32_expminus_neonfma_rr1_p5_params);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_RADDSTOREEXPMINUSMAX__NEONFMA_RR1_P5_U8_ACC2, elements_eq_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(8)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__neonfma_rr1_p5_u8_acc2, xnn_init_f32_expminus_neonfma_rr1_p5_params);
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEONFMA_RR1_P5_U8_ACC2, elements_div_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t elements = 16; elements < 80; elements += 8) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neonfma_rr1_p5_u8_acc2, xnn_init_f32_expminus_neonfma_rr1_p5_params);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEONFMA_RR1_P5_U8_ACC2, elements_lt_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t elements = 1; elements < 8; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neonfma_rr1_p5_u8_acc2, xnn_init_f32_expminus_neonfma_rr1_p5_params);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEONFMA_RR1_P5_U8_ACC2, elements_gt_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t elements = 9; elements < 16; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neonfma_rr1_p5_u8_acc2, xnn_init_f32_expminus_neonfma_rr1_p5_params);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_RADDSTOREEXPMINUSMAX__NEONFMA_RR1_P5_U12, elements_eq_12) {
    TEST_REQUIRES_ARM_NEON_FMA;
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(12)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__neonfma_rr1_p5_u12, xnn_init_f32_expminus_neonfma_rr1_p5_params);
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEONFMA_RR1_P5_U12, elements_div_12) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t elements = 24; elements < 120; elements += 12) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neonfma_rr1_p5_u12, xnn_init_f32_expminus_neonfma_rr1_p5_params);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEONFMA_RR1_P5_U12, elements_lt_12) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t elements = 1; elements < 12; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neonfma_rr1_p5_u12, xnn_init_f32_expminus_neonfma_rr1_p5_params);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEONFMA_RR1_P5_U12, elements_gt_12) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t elements = 13; elements < 24; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neonfma_rr1_p5_u12, xnn_init_f32_expminus_neonfma_rr1_p5_params);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_RADDSTOREEXPMINUSMAX__NEONFMA_RR1_P5_U12_ACC2, elements_eq_12) {
    TEST_REQUIRES_ARM_NEON_FMA;
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(12)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__neonfma_rr1_p5_u12_acc2, xnn_init_f32_expminus_neonfma_rr1_p5_params);
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEONFMA_RR1_P5_U12_ACC2, elements_div_12) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t elements = 24; elements < 120; elements += 12) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neonfma_rr1_p5_u12_acc2, xnn_init_f32_expminus_neonfma_rr1_p5_params);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEONFMA_RR1_P5_U12_ACC2, elements_lt_12) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t elements = 1; elements < 12; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neonfma_rr1_p5_u12_acc2, xnn_init_f32_expminus_neonfma_rr1_p5_params);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEONFMA_RR1_P5_U12_ACC2, elements_gt_12) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t elements = 13; elements < 24; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neonfma_rr1_p5_u12_acc2, xnn_init_f32_expminus_neonfma_rr1_p5_params);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_RADDSTOREEXPMINUSMAX__NEONFMA_RR1_P5_U12_ACC3, elements_eq_12) {
    TEST_REQUIRES_ARM_NEON_FMA;
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(12)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__neonfma_rr1_p5_u12_acc3, xnn_init_f32_expminus_neonfma_rr1_p5_params);
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEONFMA_RR1_P5_U12_ACC3, elements_div_12) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t elements = 24; elements < 120; elements += 12) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neonfma_rr1_p5_u12_acc3, xnn_init_f32_expminus_neonfma_rr1_p5_params);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEONFMA_RR1_P5_U12_ACC3, elements_lt_12) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t elements = 1; elements < 12; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neonfma_rr1_p5_u12_acc3, xnn_init_f32_expminus_neonfma_rr1_p5_params);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEONFMA_RR1_P5_U12_ACC3, elements_gt_12) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t elements = 13; elements < 24; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neonfma_rr1_p5_u12_acc3, xnn_init_f32_expminus_neonfma_rr1_p5_params);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_RADDSTOREEXPMINUSMAX__NEONFMA_RR1_P5_U16, elements_eq_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(16)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__neonfma_rr1_p5_u16, xnn_init_f32_expminus_neonfma_rr1_p5_params);
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEONFMA_RR1_P5_U16, elements_div_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t elements = 32; elements < 160; elements += 16) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neonfma_rr1_p5_u16, xnn_init_f32_expminus_neonfma_rr1_p5_params);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEONFMA_RR1_P5_U16, elements_lt_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t elements = 1; elements < 16; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neonfma_rr1_p5_u16, xnn_init_f32_expminus_neonfma_rr1_p5_params);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEONFMA_RR1_P5_U16, elements_gt_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t elements = 17; elements < 32; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neonfma_rr1_p5_u16, xnn_init_f32_expminus_neonfma_rr1_p5_params);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_RADDSTOREEXPMINUSMAX__NEONFMA_RR1_P5_U16_ACC2, elements_eq_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(16)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__neonfma_rr1_p5_u16_acc2, xnn_init_f32_expminus_neonfma_rr1_p5_params);
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEONFMA_RR1_P5_U16_ACC2, elements_div_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t elements = 32; elements < 160; elements += 16) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neonfma_rr1_p5_u16_acc2, xnn_init_f32_expminus_neonfma_rr1_p5_params);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEONFMA_RR1_P5_U16_ACC2, elements_lt_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t elements = 1; elements < 16; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neonfma_rr1_p5_u16_acc2, xnn_init_f32_expminus_neonfma_rr1_p5_params);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEONFMA_RR1_P5_U16_ACC2, elements_gt_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t elements = 17; elements < 32; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neonfma_rr1_p5_u16_acc2, xnn_init_f32_expminus_neonfma_rr1_p5_params);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_RADDSTOREEXPMINUSMAX__NEONFMA_RR1_P5_U16_ACC4, elements_eq_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(16)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__neonfma_rr1_p5_u16_acc4, xnn_init_f32_expminus_neonfma_rr1_p5_params);
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEONFMA_RR1_P5_U16_ACC4, elements_div_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t elements = 32; elements < 160; elements += 16) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neonfma_rr1_p5_u16_acc4, xnn_init_f32_expminus_neonfma_rr1_p5_params);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEONFMA_RR1_P5_U16_ACC4, elements_lt_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t elements = 1; elements < 16; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neonfma_rr1_p5_u16_acc4, xnn_init_f32_expminus_neonfma_rr1_p5_params);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEONFMA_RR1_P5_U16_ACC4, elements_gt_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t elements = 17; elements < 32; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neonfma_rr1_p5_u16_acc4, xnn_init_f32_expminus_neonfma_rr1_p5_params);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_RADDSTOREEXPMINUSMAX__NEONFMA_RR1_P5_U20, elements_eq_20) {
    TEST_REQUIRES_ARM_NEON_FMA;
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(20)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__neonfma_rr1_p5_u20, xnn_init_f32_expminus_neonfma_rr1_p5_params);
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEONFMA_RR1_P5_U20, elements_div_20) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t elements = 40; elements < 200; elements += 20) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neonfma_rr1_p5_u20, xnn_init_f32_expminus_neonfma_rr1_p5_params);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEONFMA_RR1_P5_U20, elements_lt_20) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t elements = 1; elements < 20; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neonfma_rr1_p5_u20, xnn_init_f32_expminus_neonfma_rr1_p5_params);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEONFMA_RR1_P5_U20, elements_gt_20) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t elements = 21; elements < 40; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neonfma_rr1_p5_u20, xnn_init_f32_expminus_neonfma_rr1_p5_params);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_RADDSTOREEXPMINUSMAX__NEONFMA_RR1_P5_U20_ACC2, elements_eq_20) {
    TEST_REQUIRES_ARM_NEON_FMA;
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(20)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__neonfma_rr1_p5_u20_acc2, xnn_init_f32_expminus_neonfma_rr1_p5_params);
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEONFMA_RR1_P5_U20_ACC2, elements_div_20) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t elements = 40; elements < 200; elements += 20) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neonfma_rr1_p5_u20_acc2, xnn_init_f32_expminus_neonfma_rr1_p5_params);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEONFMA_RR1_P5_U20_ACC2, elements_lt_20) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t elements = 1; elements < 20; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neonfma_rr1_p5_u20_acc2, xnn_init_f32_expminus_neonfma_rr1_p5_params);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEONFMA_RR1_P5_U20_ACC2, elements_gt_20) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t elements = 21; elements < 40; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neonfma_rr1_p5_u20_acc2, xnn_init_f32_expminus_neonfma_rr1_p5_params);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_RADDSTOREEXPMINUSMAX__NEONFMA_RR1_P5_U20_ACC5, elements_eq_20) {
    TEST_REQUIRES_ARM_NEON_FMA;
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(20)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__neonfma_rr1_p5_u20_acc5, xnn_init_f32_expminus_neonfma_rr1_p5_params);
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEONFMA_RR1_P5_U20_ACC5, elements_div_20) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t elements = 40; elements < 200; elements += 20) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neonfma_rr1_p5_u20_acc5, xnn_init_f32_expminus_neonfma_rr1_p5_params);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEONFMA_RR1_P5_U20_ACC5, elements_lt_20) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t elements = 1; elements < 20; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neonfma_rr1_p5_u20_acc5, xnn_init_f32_expminus_neonfma_rr1_p5_params);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEONFMA_RR1_P5_U20_ACC5, elements_gt_20) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t elements = 21; elements < 40; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neonfma_rr1_p5_u20_acc5, xnn_init_f32_expminus_neonfma_rr1_p5_params);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ENABLE_RISCV_VECTOR && XNN_ARCH_RISCV
  TEST(F32_RADDSTOREEXPMINUSMAX__RVV_RR2_P6_U2V, elements_eq_2v) {
    TEST_REQUIRES_RISCV_VECTOR;
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(2 * xnn_init_hardware_config()->vlenb / sizeof(float))
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__rvv_rr2_p6_u2v, xnn_init_f32_expminus_rvv_rr2_p6_params);
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__RVV_RR2_P6_U2V, elements_div_2v) {
    TEST_REQUIRES_RISCV_VECTOR;
    for (size_t elements = 4 * xnn_init_hardware_config()->vlenb / sizeof(float);
                elements < 20 * xnn_init_hardware_config()->vlenb / sizeof(float);
                elements += 2 * xnn_init_hardware_config()->vlenb / sizeof(float)) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__rvv_rr2_p6_u2v, xnn_init_f32_expminus_rvv_rr2_p6_params);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__RVV_RR2_P6_U2V, elements_lt_2v) {
    TEST_REQUIRES_RISCV_VECTOR;
    for (size_t elements = 1;
                elements < 2 * xnn_init_hardware_config()->vlenb / sizeof(float);
                elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__rvv_rr2_p6_u2v, xnn_init_f32_expminus_rvv_rr2_p6_params);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__RVV_RR2_P6_U2V, elements_gt_2v) {
    TEST_REQUIRES_RISCV_VECTOR;
    for (size_t elements = 2 * xnn_init_hardware_config()->vlenb / sizeof(float) + 1;
                elements < 4 * xnn_init_hardware_config()->vlenb / sizeof(float);
                elements += 4) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__rvv_rr2_p6_u2v, xnn_init_f32_expminus_rvv_rr2_p6_params);
    }
  }
#endif  // XNN_ENABLE_RISCV_VECTOR && XNN_ARCH_RISCV


#if XNN_ENABLE_RISCV_VECTOR && XNN_ARCH_RISCV
  TEST(F32_RADDSTOREEXPMINUSMAX__RVV_RR2_P6_U4V, elements_eq_4v) {
    TEST_REQUIRES_RISCV_VECTOR;
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(4 * xnn_init_hardware_config()->vlenb / sizeof(float))
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__rvv_rr2_p6_u4v, xnn_init_f32_expminus_rvv_rr2_p6_params);
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__RVV_RR2_P6_U4V, elements_div_4v) {
    TEST_REQUIRES_RISCV_VECTOR;
    for (size_t elements = 8 * xnn_init_hardware_config()->vlenb / sizeof(float);
                elements < 40 * xnn_init_hardware_config()->vlenb / sizeof(float);
                elements += 4 * xnn_init_hardware_config()->vlenb / sizeof(float)) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__rvv_rr2_p6_u4v, xnn_init_f32_expminus_rvv_rr2_p6_params);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__RVV_RR2_P6_U4V, elements_lt_4v) {
    TEST_REQUIRES_RISCV_VECTOR;
    for (size_t elements = 1;
                elements < 4 * xnn_init_hardware_config()->vlenb / sizeof(float);
                elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__rvv_rr2_p6_u4v, xnn_init_f32_expminus_rvv_rr2_p6_params);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__RVV_RR2_P6_U4V, elements_gt_4v) {
    TEST_REQUIRES_RISCV_VECTOR;
    for (size_t elements = 4 * xnn_init_hardware_config()->vlenb / sizeof(float) + 1;
                elements < 8 * xnn_init_hardware_config()->vlenb / sizeof(float);
                elements += 8) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__rvv_rr2_p6_u4v, xnn_init_f32_expminus_rvv_rr2_p6_params);
    }
  }
#endif  // XNN_ENABLE_RISCV_VECTOR && XNN_ARCH_RISCV


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_RADDSTOREEXPMINUSMAX__SSE2_RR2_P5_U4, elements_eq_4) {
    TEST_REQUIRES_X86_SSE2;
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(4)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__sse2_rr2_p5_u4, xnn_init_f32_expminus_sse2_rr2_p5_params);
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__SSE2_RR2_P5_U4, elements_div_4) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t elements = 8; elements < 40; elements += 4) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__sse2_rr2_p5_u4, xnn_init_f32_expminus_sse2_rr2_p5_params);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__SSE2_RR2_P5_U4, elements_lt_4) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t elements = 1; elements < 4; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__sse2_rr2_p5_u4, xnn_init_f32_expminus_sse2_rr2_p5_params);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__SSE2_RR2_P5_U4, elements_gt_4) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t elements = 5; elements < 8; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__sse2_rr2_p5_u4, xnn_init_f32_expminus_sse2_rr2_p5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_RADDSTOREEXPMINUSMAX__SSE2_RR2_P5_U8, elements_eq_8) {
    TEST_REQUIRES_X86_SSE2;
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(8)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__sse2_rr2_p5_u8, xnn_init_f32_expminus_sse2_rr2_p5_params);
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__SSE2_RR2_P5_U8, elements_div_8) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t elements = 16; elements < 80; elements += 8) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__sse2_rr2_p5_u8, xnn_init_f32_expminus_sse2_rr2_p5_params);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__SSE2_RR2_P5_U8, elements_lt_8) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t elements = 1; elements < 8; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__sse2_rr2_p5_u8, xnn_init_f32_expminus_sse2_rr2_p5_params);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__SSE2_RR2_P5_U8, elements_gt_8) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t elements = 9; elements < 16; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__sse2_rr2_p5_u8, xnn_init_f32_expminus_sse2_rr2_p5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_RADDSTOREEXPMINUSMAX__SSE2_RR2_P5_U8_ACC2, elements_eq_8) {
    TEST_REQUIRES_X86_SSE2;
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(8)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__sse2_rr2_p5_u8_acc2, xnn_init_f32_expminus_sse2_rr2_p5_params);
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__SSE2_RR2_P5_U8_ACC2, elements_div_8) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t elements = 16; elements < 80; elements += 8) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__sse2_rr2_p5_u8_acc2, xnn_init_f32_expminus_sse2_rr2_p5_params);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__SSE2_RR2_P5_U8_ACC2, elements_lt_8) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t elements = 1; elements < 8; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__sse2_rr2_p5_u8_acc2, xnn_init_f32_expminus_sse2_rr2_p5_params);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__SSE2_RR2_P5_U8_ACC2, elements_gt_8) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t elements = 9; elements < 16; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__sse2_rr2_p5_u8_acc2, xnn_init_f32_expminus_sse2_rr2_p5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_RADDSTOREEXPMINUSMAX__SSE2_RR2_P5_U12, elements_eq_12) {
    TEST_REQUIRES_X86_SSE2;
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(12)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__sse2_rr2_p5_u12, xnn_init_f32_expminus_sse2_rr2_p5_params);
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__SSE2_RR2_P5_U12, elements_div_12) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t elements = 24; elements < 120; elements += 12) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__sse2_rr2_p5_u12, xnn_init_f32_expminus_sse2_rr2_p5_params);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__SSE2_RR2_P5_U12, elements_lt_12) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t elements = 1; elements < 12; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__sse2_rr2_p5_u12, xnn_init_f32_expminus_sse2_rr2_p5_params);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__SSE2_RR2_P5_U12, elements_gt_12) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t elements = 13; elements < 24; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__sse2_rr2_p5_u12, xnn_init_f32_expminus_sse2_rr2_p5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_RADDSTOREEXPMINUSMAX__SSE2_RR2_P5_U12_ACC2, elements_eq_12) {
    TEST_REQUIRES_X86_SSE2;
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(12)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__sse2_rr2_p5_u12_acc2, xnn_init_f32_expminus_sse2_rr2_p5_params);
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__SSE2_RR2_P5_U12_ACC2, elements_div_12) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t elements = 24; elements < 120; elements += 12) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__sse2_rr2_p5_u12_acc2, xnn_init_f32_expminus_sse2_rr2_p5_params);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__SSE2_RR2_P5_U12_ACC2, elements_lt_12) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t elements = 1; elements < 12; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__sse2_rr2_p5_u12_acc2, xnn_init_f32_expminus_sse2_rr2_p5_params);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__SSE2_RR2_P5_U12_ACC2, elements_gt_12) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t elements = 13; elements < 24; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__sse2_rr2_p5_u12_acc2, xnn_init_f32_expminus_sse2_rr2_p5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_RADDSTOREEXPMINUSMAX__SSE2_RR2_P5_U12_ACC3, elements_eq_12) {
    TEST_REQUIRES_X86_SSE2;
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(12)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__sse2_rr2_p5_u12_acc3, xnn_init_f32_expminus_sse2_rr2_p5_params);
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__SSE2_RR2_P5_U12_ACC3, elements_div_12) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t elements = 24; elements < 120; elements += 12) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__sse2_rr2_p5_u12_acc3, xnn_init_f32_expminus_sse2_rr2_p5_params);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__SSE2_RR2_P5_U12_ACC3, elements_lt_12) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t elements = 1; elements < 12; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__sse2_rr2_p5_u12_acc3, xnn_init_f32_expminus_sse2_rr2_p5_params);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__SSE2_RR2_P5_U12_ACC3, elements_gt_12) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t elements = 13; elements < 24; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__sse2_rr2_p5_u12_acc3, xnn_init_f32_expminus_sse2_rr2_p5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_RADDSTOREEXPMINUSMAX__SSE2_RR2_P5_U16, elements_eq_16) {
    TEST_REQUIRES_X86_SSE2;
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(16)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__sse2_rr2_p5_u16, xnn_init_f32_expminus_sse2_rr2_p5_params);
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__SSE2_RR2_P5_U16, elements_div_16) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t elements = 32; elements < 160; elements += 16) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__sse2_rr2_p5_u16, xnn_init_f32_expminus_sse2_rr2_p5_params);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__SSE2_RR2_P5_U16, elements_lt_16) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t elements = 1; elements < 16; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__sse2_rr2_p5_u16, xnn_init_f32_expminus_sse2_rr2_p5_params);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__SSE2_RR2_P5_U16, elements_gt_16) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t elements = 17; elements < 32; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__sse2_rr2_p5_u16, xnn_init_f32_expminus_sse2_rr2_p5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_RADDSTOREEXPMINUSMAX__SSE2_RR2_P5_U16_ACC2, elements_eq_16) {
    TEST_REQUIRES_X86_SSE2;
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(16)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__sse2_rr2_p5_u16_acc2, xnn_init_f32_expminus_sse2_rr2_p5_params);
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__SSE2_RR2_P5_U16_ACC2, elements_div_16) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t elements = 32; elements < 160; elements += 16) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__sse2_rr2_p5_u16_acc2, xnn_init_f32_expminus_sse2_rr2_p5_params);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__SSE2_RR2_P5_U16_ACC2, elements_lt_16) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t elements = 1; elements < 16; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__sse2_rr2_p5_u16_acc2, xnn_init_f32_expminus_sse2_rr2_p5_params);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__SSE2_RR2_P5_U16_ACC2, elements_gt_16) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t elements = 17; elements < 32; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__sse2_rr2_p5_u16_acc2, xnn_init_f32_expminus_sse2_rr2_p5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_RADDSTOREEXPMINUSMAX__SSE2_RR2_P5_U16_ACC4, elements_eq_16) {
    TEST_REQUIRES_X86_SSE2;
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(16)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__sse2_rr2_p5_u16_acc4, xnn_init_f32_expminus_sse2_rr2_p5_params);
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__SSE2_RR2_P5_U16_ACC4, elements_div_16) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t elements = 32; elements < 160; elements += 16) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__sse2_rr2_p5_u16_acc4, xnn_init_f32_expminus_sse2_rr2_p5_params);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__SSE2_RR2_P5_U16_ACC4, elements_lt_16) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t elements = 1; elements < 16; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__sse2_rr2_p5_u16_acc4, xnn_init_f32_expminus_sse2_rr2_p5_params);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__SSE2_RR2_P5_U16_ACC4, elements_gt_16) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t elements = 17; elements < 32; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__sse2_rr2_p5_u16_acc4, xnn_init_f32_expminus_sse2_rr2_p5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_RADDSTOREEXPMINUSMAX__SSE2_RR2_P5_U20, elements_eq_20) {
    TEST_REQUIRES_X86_SSE2;
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(20)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__sse2_rr2_p5_u20, xnn_init_f32_expminus_sse2_rr2_p5_params);
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__SSE2_RR2_P5_U20, elements_div_20) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t elements = 40; elements < 200; elements += 20) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__sse2_rr2_p5_u20, xnn_init_f32_expminus_sse2_rr2_p5_params);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__SSE2_RR2_P5_U20, elements_lt_20) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t elements = 1; elements < 20; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__sse2_rr2_p5_u20, xnn_init_f32_expminus_sse2_rr2_p5_params);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__SSE2_RR2_P5_U20, elements_gt_20) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t elements = 21; elements < 40; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__sse2_rr2_p5_u20, xnn_init_f32_expminus_sse2_rr2_p5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_RADDSTOREEXPMINUSMAX__SSE2_RR2_P5_U20_ACC2, elements_eq_20) {
    TEST_REQUIRES_X86_SSE2;
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(20)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__sse2_rr2_p5_u20_acc2, xnn_init_f32_expminus_sse2_rr2_p5_params);
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__SSE2_RR2_P5_U20_ACC2, elements_div_20) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t elements = 40; elements < 200; elements += 20) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__sse2_rr2_p5_u20_acc2, xnn_init_f32_expminus_sse2_rr2_p5_params);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__SSE2_RR2_P5_U20_ACC2, elements_lt_20) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t elements = 1; elements < 20; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__sse2_rr2_p5_u20_acc2, xnn_init_f32_expminus_sse2_rr2_p5_params);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__SSE2_RR2_P5_U20_ACC2, elements_gt_20) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t elements = 21; elements < 40; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__sse2_rr2_p5_u20_acc2, xnn_init_f32_expminus_sse2_rr2_p5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_RADDSTOREEXPMINUSMAX__SSE2_RR2_P5_U20_ACC5, elements_eq_20) {
    TEST_REQUIRES_X86_SSE2;
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(20)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__sse2_rr2_p5_u20_acc5, xnn_init_f32_expminus_sse2_rr2_p5_params);
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__SSE2_RR2_P5_U20_ACC5, elements_div_20) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t elements = 40; elements < 200; elements += 20) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__sse2_rr2_p5_u20_acc5, xnn_init_f32_expminus_sse2_rr2_p5_params);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__SSE2_RR2_P5_U20_ACC5, elements_lt_20) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t elements = 1; elements < 20; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__sse2_rr2_p5_u20_acc5, xnn_init_f32_expminus_sse2_rr2_p5_params);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__SSE2_RR2_P5_U20_ACC5, elements_gt_20) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t elements = 21; elements < 40; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__sse2_rr2_p5_u20_acc5, xnn_init_f32_expminus_sse2_rr2_p5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_RADDSTOREEXPMINUSMAX__AVX2_RR1_P5_U64, elements_eq_64) {
    TEST_REQUIRES_X86_AVX2;
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(64)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__avx2_rr1_p5_u64, xnn_init_f32_expminus_avx2_rr1_p5_params);
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__AVX2_RR1_P5_U64, elements_div_64) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t elements = 128; elements < 640; elements += 64) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__avx2_rr1_p5_u64, xnn_init_f32_expminus_avx2_rr1_p5_params);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__AVX2_RR1_P5_U64, elements_lt_64) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t elements = 1; elements < 64; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__avx2_rr1_p5_u64, xnn_init_f32_expminus_avx2_rr1_p5_params);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__AVX2_RR1_P5_U64, elements_gt_64) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t elements = 65; elements < 128; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__avx2_rr1_p5_u64, xnn_init_f32_expminus_avx2_rr1_p5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_RADDSTOREEXPMINUSMAX__AVX2_RR1_P5_U64_ACC2, elements_eq_64) {
    TEST_REQUIRES_X86_AVX2;
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(64)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__avx2_rr1_p5_u64_acc2, xnn_init_f32_expminus_avx2_rr1_p5_params);
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__AVX2_RR1_P5_U64_ACC2, elements_div_64) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t elements = 128; elements < 640; elements += 64) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__avx2_rr1_p5_u64_acc2, xnn_init_f32_expminus_avx2_rr1_p5_params);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__AVX2_RR1_P5_U64_ACC2, elements_lt_64) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t elements = 1; elements < 64; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__avx2_rr1_p5_u64_acc2, xnn_init_f32_expminus_avx2_rr1_p5_params);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__AVX2_RR1_P5_U64_ACC2, elements_gt_64) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t elements = 65; elements < 128; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__avx2_rr1_p5_u64_acc2, xnn_init_f32_expminus_avx2_rr1_p5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_RADDSTOREEXPMINUSMAX__AVX2_RR1_P5_U64_ACC4, elements_eq_64) {
    TEST_REQUIRES_X86_AVX2;
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(64)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__avx2_rr1_p5_u64_acc4, xnn_init_f32_expminus_avx2_rr1_p5_params);
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__AVX2_RR1_P5_U64_ACC4, elements_div_64) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t elements = 128; elements < 640; elements += 64) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__avx2_rr1_p5_u64_acc4, xnn_init_f32_expminus_avx2_rr1_p5_params);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__AVX2_RR1_P5_U64_ACC4, elements_lt_64) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t elements = 1; elements < 64; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__avx2_rr1_p5_u64_acc4, xnn_init_f32_expminus_avx2_rr1_p5_params);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__AVX2_RR1_P5_U64_ACC4, elements_gt_64) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t elements = 65; elements < 128; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__avx2_rr1_p5_u64_acc4, xnn_init_f32_expminus_avx2_rr1_p5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_RADDSTOREEXPMINUSMAX__AVX2_RR1_P5_U72, elements_eq_72) {
    TEST_REQUIRES_X86_AVX2;
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(72)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__avx2_rr1_p5_u72, xnn_init_f32_expminus_avx2_rr1_p5_params);
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__AVX2_RR1_P5_U72, elements_div_72) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t elements = 144; elements < 720; elements += 72) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__avx2_rr1_p5_u72, xnn_init_f32_expminus_avx2_rr1_p5_params);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__AVX2_RR1_P5_U72, elements_lt_72) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t elements = 1; elements < 72; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__avx2_rr1_p5_u72, xnn_init_f32_expminus_avx2_rr1_p5_params);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__AVX2_RR1_P5_U72, elements_gt_72) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t elements = 73; elements < 144; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__avx2_rr1_p5_u72, xnn_init_f32_expminus_avx2_rr1_p5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_RADDSTOREEXPMINUSMAX__AVX2_RR1_P5_U72_ACC3, elements_eq_72) {
    TEST_REQUIRES_X86_AVX2;
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(72)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__avx2_rr1_p5_u72_acc3, xnn_init_f32_expminus_avx2_rr1_p5_params);
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__AVX2_RR1_P5_U72_ACC3, elements_div_72) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t elements = 144; elements < 720; elements += 72) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__avx2_rr1_p5_u72_acc3, xnn_init_f32_expminus_avx2_rr1_p5_params);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__AVX2_RR1_P5_U72_ACC3, elements_lt_72) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t elements = 1; elements < 72; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__avx2_rr1_p5_u72_acc3, xnn_init_f32_expminus_avx2_rr1_p5_params);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__AVX2_RR1_P5_U72_ACC3, elements_gt_72) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t elements = 73; elements < 144; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__avx2_rr1_p5_u72_acc3, xnn_init_f32_expminus_avx2_rr1_p5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_RADDSTOREEXPMINUSMAX__AVX2_RR1_P5_U80, elements_eq_80) {
    TEST_REQUIRES_X86_AVX2;
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(80)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__avx2_rr1_p5_u80, xnn_init_f32_expminus_avx2_rr1_p5_params);
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__AVX2_RR1_P5_U80, elements_div_80) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t elements = 160; elements < 800; elements += 80) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__avx2_rr1_p5_u80, xnn_init_f32_expminus_avx2_rr1_p5_params);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__AVX2_RR1_P5_U80, elements_lt_80) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t elements = 1; elements < 80; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__avx2_rr1_p5_u80, xnn_init_f32_expminus_avx2_rr1_p5_params);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__AVX2_RR1_P5_U80, elements_gt_80) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t elements = 81; elements < 160; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__avx2_rr1_p5_u80, xnn_init_f32_expminus_avx2_rr1_p5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_RADDSTOREEXPMINUSMAX__AVX2_RR1_P5_U80_ACC2, elements_eq_80) {
    TEST_REQUIRES_X86_AVX2;
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(80)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__avx2_rr1_p5_u80_acc2, xnn_init_f32_expminus_avx2_rr1_p5_params);
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__AVX2_RR1_P5_U80_ACC2, elements_div_80) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t elements = 160; elements < 800; elements += 80) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__avx2_rr1_p5_u80_acc2, xnn_init_f32_expminus_avx2_rr1_p5_params);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__AVX2_RR1_P5_U80_ACC2, elements_lt_80) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t elements = 1; elements < 80; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__avx2_rr1_p5_u80_acc2, xnn_init_f32_expminus_avx2_rr1_p5_params);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__AVX2_RR1_P5_U80_ACC2, elements_gt_80) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t elements = 81; elements < 160; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__avx2_rr1_p5_u80_acc2, xnn_init_f32_expminus_avx2_rr1_p5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_RADDSTOREEXPMINUSMAX__AVX2_RR1_P5_U80_ACC5, elements_eq_80) {
    TEST_REQUIRES_X86_AVX2;
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(80)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__avx2_rr1_p5_u80_acc5, xnn_init_f32_expminus_avx2_rr1_p5_params);
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__AVX2_RR1_P5_U80_ACC5, elements_div_80) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t elements = 160; elements < 800; elements += 80) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__avx2_rr1_p5_u80_acc5, xnn_init_f32_expminus_avx2_rr1_p5_params);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__AVX2_RR1_P5_U80_ACC5, elements_lt_80) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t elements = 1; elements < 80; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__avx2_rr1_p5_u80_acc5, xnn_init_f32_expminus_avx2_rr1_p5_params);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__AVX2_RR1_P5_U80_ACC5, elements_gt_80) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t elements = 81; elements < 160; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__avx2_rr1_p5_u80_acc5, xnn_init_f32_expminus_avx2_rr1_p5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_RADDSTOREEXPMINUSMAX__AVX2_RR1_P5_U96, elements_eq_96) {
    TEST_REQUIRES_X86_AVX2;
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(96)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__avx2_rr1_p5_u96, xnn_init_f32_expminus_avx2_rr1_p5_params);
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__AVX2_RR1_P5_U96, elements_div_96) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t elements = 192; elements < 960; elements += 96) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__avx2_rr1_p5_u96, xnn_init_f32_expminus_avx2_rr1_p5_params);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__AVX2_RR1_P5_U96, elements_lt_96) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t elements = 1; elements < 96; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__avx2_rr1_p5_u96, xnn_init_f32_expminus_avx2_rr1_p5_params);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__AVX2_RR1_P5_U96, elements_gt_96) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t elements = 97; elements < 192; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__avx2_rr1_p5_u96, xnn_init_f32_expminus_avx2_rr1_p5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_RADDSTOREEXPMINUSMAX__AVX2_RR1_P5_U96_ACC2, elements_eq_96) {
    TEST_REQUIRES_X86_AVX2;
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(96)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__avx2_rr1_p5_u96_acc2, xnn_init_f32_expminus_avx2_rr1_p5_params);
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__AVX2_RR1_P5_U96_ACC2, elements_div_96) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t elements = 192; elements < 960; elements += 96) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__avx2_rr1_p5_u96_acc2, xnn_init_f32_expminus_avx2_rr1_p5_params);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__AVX2_RR1_P5_U96_ACC2, elements_lt_96) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t elements = 1; elements < 96; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__avx2_rr1_p5_u96_acc2, xnn_init_f32_expminus_avx2_rr1_p5_params);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__AVX2_RR1_P5_U96_ACC2, elements_gt_96) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t elements = 97; elements < 192; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__avx2_rr1_p5_u96_acc2, xnn_init_f32_expminus_avx2_rr1_p5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_RADDSTOREEXPMINUSMAX__AVX2_RR1_P5_U96_ACC3, elements_eq_96) {
    TEST_REQUIRES_X86_AVX2;
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(96)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__avx2_rr1_p5_u96_acc3, xnn_init_f32_expminus_avx2_rr1_p5_params);
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__AVX2_RR1_P5_U96_ACC3, elements_div_96) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t elements = 192; elements < 960; elements += 96) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__avx2_rr1_p5_u96_acc3, xnn_init_f32_expminus_avx2_rr1_p5_params);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__AVX2_RR1_P5_U96_ACC3, elements_lt_96) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t elements = 1; elements < 96; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__avx2_rr1_p5_u96_acc3, xnn_init_f32_expminus_avx2_rr1_p5_params);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__AVX2_RR1_P5_U96_ACC3, elements_gt_96) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t elements = 97; elements < 192; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__avx2_rr1_p5_u96_acc3, xnn_init_f32_expminus_avx2_rr1_p5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_RADDSTOREEXPMINUSMAX__AVX2_RR1_P5_U96_ACC6, elements_eq_96) {
    TEST_REQUIRES_X86_AVX2;
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(96)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__avx2_rr1_p5_u96_acc6, xnn_init_f32_expminus_avx2_rr1_p5_params);
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__AVX2_RR1_P5_U96_ACC6, elements_div_96) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t elements = 192; elements < 960; elements += 96) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__avx2_rr1_p5_u96_acc6, xnn_init_f32_expminus_avx2_rr1_p5_params);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__AVX2_RR1_P5_U96_ACC6, elements_lt_96) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t elements = 1; elements < 96; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__avx2_rr1_p5_u96_acc6, xnn_init_f32_expminus_avx2_rr1_p5_params);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__AVX2_RR1_P5_U96_ACC6, elements_gt_96) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t elements = 97; elements < 192; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__avx2_rr1_p5_u96_acc6, xnn_init_f32_expminus_avx2_rr1_p5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_RADDSTOREEXPMINUSMAX__AVX512F_RR1_P5_SCALEF_U128, elements_eq_128) {
    TEST_REQUIRES_X86_AVX512F;
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(128)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__avx512f_rr1_p5_scalef_u128, xnn_init_f32_expminus_avx512_rr1_p5_params);
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__AVX512F_RR1_P5_SCALEF_U128, elements_div_128) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t elements = 256; elements < 1280; elements += 128) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__avx512f_rr1_p5_scalef_u128, xnn_init_f32_expminus_avx512_rr1_p5_params);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__AVX512F_RR1_P5_SCALEF_U128, elements_lt_128) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t elements = 1; elements < 128; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__avx512f_rr1_p5_scalef_u128, xnn_init_f32_expminus_avx512_rr1_p5_params);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__AVX512F_RR1_P5_SCALEF_U128, elements_gt_128) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t elements = 129; elements < 256; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__avx512f_rr1_p5_scalef_u128, xnn_init_f32_expminus_avx512_rr1_p5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_RADDSTOREEXPMINUSMAX__AVX512F_RR1_P5_SCALEF_U128_ACC2, elements_eq_128) {
    TEST_REQUIRES_X86_AVX512F;
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(128)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__avx512f_rr1_p5_scalef_u128_acc2, xnn_init_f32_expminus_avx512_rr1_p5_params);
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__AVX512F_RR1_P5_SCALEF_U128_ACC2, elements_div_128) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t elements = 256; elements < 1280; elements += 128) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__avx512f_rr1_p5_scalef_u128_acc2, xnn_init_f32_expminus_avx512_rr1_p5_params);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__AVX512F_RR1_P5_SCALEF_U128_ACC2, elements_lt_128) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t elements = 1; elements < 128; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__avx512f_rr1_p5_scalef_u128_acc2, xnn_init_f32_expminus_avx512_rr1_p5_params);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__AVX512F_RR1_P5_SCALEF_U128_ACC2, elements_gt_128) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t elements = 129; elements < 256; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__avx512f_rr1_p5_scalef_u128_acc2, xnn_init_f32_expminus_avx512_rr1_p5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_RADDSTOREEXPMINUSMAX__AVX512F_RR1_P5_SCALEF_U128_ACC4, elements_eq_128) {
    TEST_REQUIRES_X86_AVX512F;
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(128)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__avx512f_rr1_p5_scalef_u128_acc4, xnn_init_f32_expminus_avx512_rr1_p5_params);
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__AVX512F_RR1_P5_SCALEF_U128_ACC4, elements_div_128) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t elements = 256; elements < 1280; elements += 128) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__avx512f_rr1_p5_scalef_u128_acc4, xnn_init_f32_expminus_avx512_rr1_p5_params);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__AVX512F_RR1_P5_SCALEF_U128_ACC4, elements_lt_128) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t elements = 1; elements < 128; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__avx512f_rr1_p5_scalef_u128_acc4, xnn_init_f32_expminus_avx512_rr1_p5_params);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__AVX512F_RR1_P5_SCALEF_U128_ACC4, elements_gt_128) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t elements = 129; elements < 256; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__avx512f_rr1_p5_scalef_u128_acc4, xnn_init_f32_expminus_avx512_rr1_p5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_RADDSTOREEXPMINUSMAX__AVX512F_RR1_P5_SCALEF_U144, elements_eq_144) {
    TEST_REQUIRES_X86_AVX512F;
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(144)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__avx512f_rr1_p5_scalef_u144, xnn_init_f32_expminus_avx512_rr1_p5_params);
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__AVX512F_RR1_P5_SCALEF_U144, elements_div_144) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t elements = 288; elements < 1440; elements += 144) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__avx512f_rr1_p5_scalef_u144, xnn_init_f32_expminus_avx512_rr1_p5_params);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__AVX512F_RR1_P5_SCALEF_U144, elements_lt_144) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t elements = 1; elements < 144; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__avx512f_rr1_p5_scalef_u144, xnn_init_f32_expminus_avx512_rr1_p5_params);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__AVX512F_RR1_P5_SCALEF_U144, elements_gt_144) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t elements = 145; elements < 288; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__avx512f_rr1_p5_scalef_u144, xnn_init_f32_expminus_avx512_rr1_p5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_RADDSTOREEXPMINUSMAX__AVX512F_RR1_P5_SCALEF_U144_ACC3, elements_eq_144) {
    TEST_REQUIRES_X86_AVX512F;
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(144)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__avx512f_rr1_p5_scalef_u144_acc3, xnn_init_f32_expminus_avx512_rr1_p5_params);
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__AVX512F_RR1_P5_SCALEF_U144_ACC3, elements_div_144) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t elements = 288; elements < 1440; elements += 144) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__avx512f_rr1_p5_scalef_u144_acc3, xnn_init_f32_expminus_avx512_rr1_p5_params);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__AVX512F_RR1_P5_SCALEF_U144_ACC3, elements_lt_144) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t elements = 1; elements < 144; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__avx512f_rr1_p5_scalef_u144_acc3, xnn_init_f32_expminus_avx512_rr1_p5_params);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__AVX512F_RR1_P5_SCALEF_U144_ACC3, elements_gt_144) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t elements = 145; elements < 288; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__avx512f_rr1_p5_scalef_u144_acc3, xnn_init_f32_expminus_avx512_rr1_p5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_RADDSTOREEXPMINUSMAX__AVX512F_RR1_P5_SCALEF_U160, elements_eq_160) {
    TEST_REQUIRES_X86_AVX512F;
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(160)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__avx512f_rr1_p5_scalef_u160, xnn_init_f32_expminus_avx512_rr1_p5_params);
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__AVX512F_RR1_P5_SCALEF_U160, elements_div_160) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t elements = 320; elements < 1600; elements += 160) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__avx512f_rr1_p5_scalef_u160, xnn_init_f32_expminus_avx512_rr1_p5_params);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__AVX512F_RR1_P5_SCALEF_U160, elements_lt_160) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t elements = 1; elements < 160; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__avx512f_rr1_p5_scalef_u160, xnn_init_f32_expminus_avx512_rr1_p5_params);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__AVX512F_RR1_P5_SCALEF_U160, elements_gt_160) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t elements = 161; elements < 320; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__avx512f_rr1_p5_scalef_u160, xnn_init_f32_expminus_avx512_rr1_p5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_RADDSTOREEXPMINUSMAX__AVX512F_RR1_P5_SCALEF_U160_ACC2, elements_eq_160) {
    TEST_REQUIRES_X86_AVX512F;
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(160)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__avx512f_rr1_p5_scalef_u160_acc2, xnn_init_f32_expminus_avx512_rr1_p5_params);
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__AVX512F_RR1_P5_SCALEF_U160_ACC2, elements_div_160) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t elements = 320; elements < 1600; elements += 160) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__avx512f_rr1_p5_scalef_u160_acc2, xnn_init_f32_expminus_avx512_rr1_p5_params);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__AVX512F_RR1_P5_SCALEF_U160_ACC2, elements_lt_160) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t elements = 1; elements < 160; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__avx512f_rr1_p5_scalef_u160_acc2, xnn_init_f32_expminus_avx512_rr1_p5_params);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__AVX512F_RR1_P5_SCALEF_U160_ACC2, elements_gt_160) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t elements = 161; elements < 320; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__avx512f_rr1_p5_scalef_u160_acc2, xnn_init_f32_expminus_avx512_rr1_p5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_RADDSTOREEXPMINUSMAX__AVX512F_RR1_P5_SCALEF_U160_ACC5, elements_eq_160) {
    TEST_REQUIRES_X86_AVX512F;
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(160)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__avx512f_rr1_p5_scalef_u160_acc5, xnn_init_f32_expminus_avx512_rr1_p5_params);
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__AVX512F_RR1_P5_SCALEF_U160_ACC5, elements_div_160) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t elements = 320; elements < 1600; elements += 160) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__avx512f_rr1_p5_scalef_u160_acc5, xnn_init_f32_expminus_avx512_rr1_p5_params);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__AVX512F_RR1_P5_SCALEF_U160_ACC5, elements_lt_160) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t elements = 1; elements < 160; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__avx512f_rr1_p5_scalef_u160_acc5, xnn_init_f32_expminus_avx512_rr1_p5_params);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__AVX512F_RR1_P5_SCALEF_U160_ACC5, elements_gt_160) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t elements = 161; elements < 320; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__avx512f_rr1_p5_scalef_u160_acc5, xnn_init_f32_expminus_avx512_rr1_p5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_RADDSTOREEXPMINUSMAX__AVX512F_RR1_P5_SCALEF_U192, elements_eq_192) {
    TEST_REQUIRES_X86_AVX512F;
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(192)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__avx512f_rr1_p5_scalef_u192, xnn_init_f32_expminus_avx512_rr1_p5_params);
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__AVX512F_RR1_P5_SCALEF_U192, elements_div_192) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t elements = 384; elements < 1920; elements += 192) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__avx512f_rr1_p5_scalef_u192, xnn_init_f32_expminus_avx512_rr1_p5_params);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__AVX512F_RR1_P5_SCALEF_U192, elements_lt_192) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t elements = 1; elements < 192; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__avx512f_rr1_p5_scalef_u192, xnn_init_f32_expminus_avx512_rr1_p5_params);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__AVX512F_RR1_P5_SCALEF_U192, elements_gt_192) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t elements = 193; elements < 384; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__avx512f_rr1_p5_scalef_u192, xnn_init_f32_expminus_avx512_rr1_p5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_RADDSTOREEXPMINUSMAX__AVX512F_RR1_P5_SCALEF_U192_ACC2, elements_eq_192) {
    TEST_REQUIRES_X86_AVX512F;
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(192)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__avx512f_rr1_p5_scalef_u192_acc2, xnn_init_f32_expminus_avx512_rr1_p5_params);
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__AVX512F_RR1_P5_SCALEF_U192_ACC2, elements_div_192) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t elements = 384; elements < 1920; elements += 192) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__avx512f_rr1_p5_scalef_u192_acc2, xnn_init_f32_expminus_avx512_rr1_p5_params);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__AVX512F_RR1_P5_SCALEF_U192_ACC2, elements_lt_192) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t elements = 1; elements < 192; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__avx512f_rr1_p5_scalef_u192_acc2, xnn_init_f32_expminus_avx512_rr1_p5_params);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__AVX512F_RR1_P5_SCALEF_U192_ACC2, elements_gt_192) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t elements = 193; elements < 384; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__avx512f_rr1_p5_scalef_u192_acc2, xnn_init_f32_expminus_avx512_rr1_p5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_RADDSTOREEXPMINUSMAX__AVX512F_RR1_P5_SCALEF_U192_ACC3, elements_eq_192) {
    TEST_REQUIRES_X86_AVX512F;
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(192)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__avx512f_rr1_p5_scalef_u192_acc3, xnn_init_f32_expminus_avx512_rr1_p5_params);
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__AVX512F_RR1_P5_SCALEF_U192_ACC3, elements_div_192) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t elements = 384; elements < 1920; elements += 192) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__avx512f_rr1_p5_scalef_u192_acc3, xnn_init_f32_expminus_avx512_rr1_p5_params);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__AVX512F_RR1_P5_SCALEF_U192_ACC3, elements_lt_192) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t elements = 1; elements < 192; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__avx512f_rr1_p5_scalef_u192_acc3, xnn_init_f32_expminus_avx512_rr1_p5_params);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__AVX512F_RR1_P5_SCALEF_U192_ACC3, elements_gt_192) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t elements = 193; elements < 384; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__avx512f_rr1_p5_scalef_u192_acc3, xnn_init_f32_expminus_avx512_rr1_p5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_RADDSTOREEXPMINUSMAX__AVX512F_RR1_P5_SCALEF_U192_ACC6, elements_eq_192) {
    TEST_REQUIRES_X86_AVX512F;
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(192)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__avx512f_rr1_p5_scalef_u192_acc6, xnn_init_f32_expminus_avx512_rr1_p5_params);
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__AVX512F_RR1_P5_SCALEF_U192_ACC6, elements_div_192) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t elements = 384; elements < 1920; elements += 192) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__avx512f_rr1_p5_scalef_u192_acc6, xnn_init_f32_expminus_avx512_rr1_p5_params);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__AVX512F_RR1_P5_SCALEF_U192_ACC6, elements_lt_192) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t elements = 1; elements < 192; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__avx512f_rr1_p5_scalef_u192_acc6, xnn_init_f32_expminus_avx512_rr1_p5_params);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__AVX512F_RR1_P5_SCALEF_U192_ACC6, elements_gt_192) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t elements = 193; elements < 384; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__avx512f_rr1_p5_scalef_u192_acc6, xnn_init_f32_expminus_avx512_rr1_p5_params);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_RADDSTOREEXPMINUSMAX__WASMSIMD_RR2_P5_U4, elements_eq_4) {
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(4)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__wasmsimd_rr2_p5_u4, xnn_init_f32_expminus_wasmsimd_rr2_p5_params);
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__WASMSIMD_RR2_P5_U4, elements_div_4) {
    for (size_t elements = 8; elements < 40; elements += 4) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__wasmsimd_rr2_p5_u4, xnn_init_f32_expminus_wasmsimd_rr2_p5_params);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__WASMSIMD_RR2_P5_U4, elements_lt_4) {
    for (size_t elements = 1; elements < 4; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__wasmsimd_rr2_p5_u4, xnn_init_f32_expminus_wasmsimd_rr2_p5_params);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__WASMSIMD_RR2_P5_U4, elements_gt_4) {
    for (size_t elements = 5; elements < 8; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__wasmsimd_rr2_p5_u4, xnn_init_f32_expminus_wasmsimd_rr2_p5_params);
    }
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_RADDSTOREEXPMINUSMAX__WASMSIMD_RR2_P5_U8, elements_eq_8) {
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(8)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__wasmsimd_rr2_p5_u8, xnn_init_f32_expminus_wasmsimd_rr2_p5_params);
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__WASMSIMD_RR2_P5_U8, elements_div_8) {
    for (size_t elements = 16; elements < 80; elements += 8) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__wasmsimd_rr2_p5_u8, xnn_init_f32_expminus_wasmsimd_rr2_p5_params);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__WASMSIMD_RR2_P5_U8, elements_lt_8) {
    for (size_t elements = 1; elements < 8; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__wasmsimd_rr2_p5_u8, xnn_init_f32_expminus_wasmsimd_rr2_p5_params);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__WASMSIMD_RR2_P5_U8, elements_gt_8) {
    for (size_t elements = 9; elements < 16; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__wasmsimd_rr2_p5_u8, xnn_init_f32_expminus_wasmsimd_rr2_p5_params);
    }
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_RADDSTOREEXPMINUSMAX__WASMSIMD_RR2_P5_U8_ACC2, elements_eq_8) {
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(8)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__wasmsimd_rr2_p5_u8_acc2, xnn_init_f32_expminus_wasmsimd_rr2_p5_params);
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__WASMSIMD_RR2_P5_U8_ACC2, elements_div_8) {
    for (size_t elements = 16; elements < 80; elements += 8) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__wasmsimd_rr2_p5_u8_acc2, xnn_init_f32_expminus_wasmsimd_rr2_p5_params);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__WASMSIMD_RR2_P5_U8_ACC2, elements_lt_8) {
    for (size_t elements = 1; elements < 8; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__wasmsimd_rr2_p5_u8_acc2, xnn_init_f32_expminus_wasmsimd_rr2_p5_params);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__WASMSIMD_RR2_P5_U8_ACC2, elements_gt_8) {
    for (size_t elements = 9; elements < 16; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__wasmsimd_rr2_p5_u8_acc2, xnn_init_f32_expminus_wasmsimd_rr2_p5_params);
    }
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_RADDSTOREEXPMINUSMAX__WASMSIMD_RR2_P5_U12, elements_eq_12) {
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(12)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__wasmsimd_rr2_p5_u12, xnn_init_f32_expminus_wasmsimd_rr2_p5_params);
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__WASMSIMD_RR2_P5_U12, elements_div_12) {
    for (size_t elements = 24; elements < 120; elements += 12) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__wasmsimd_rr2_p5_u12, xnn_init_f32_expminus_wasmsimd_rr2_p5_params);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__WASMSIMD_RR2_P5_U12, elements_lt_12) {
    for (size_t elements = 1; elements < 12; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__wasmsimd_rr2_p5_u12, xnn_init_f32_expminus_wasmsimd_rr2_p5_params);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__WASMSIMD_RR2_P5_U12, elements_gt_12) {
    for (size_t elements = 13; elements < 24; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__wasmsimd_rr2_p5_u12, xnn_init_f32_expminus_wasmsimd_rr2_p5_params);
    }
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_RADDSTOREEXPMINUSMAX__WASMSIMD_RR2_P5_U12_ACC2, elements_eq_12) {
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(12)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__wasmsimd_rr2_p5_u12_acc2, xnn_init_f32_expminus_wasmsimd_rr2_p5_params);
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__WASMSIMD_RR2_P5_U12_ACC2, elements_div_12) {
    for (size_t elements = 24; elements < 120; elements += 12) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__wasmsimd_rr2_p5_u12_acc2, xnn_init_f32_expminus_wasmsimd_rr2_p5_params);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__WASMSIMD_RR2_P5_U12_ACC2, elements_lt_12) {
    for (size_t elements = 1; elements < 12; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__wasmsimd_rr2_p5_u12_acc2, xnn_init_f32_expminus_wasmsimd_rr2_p5_params);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__WASMSIMD_RR2_P5_U12_ACC2, elements_gt_12) {
    for (size_t elements = 13; elements < 24; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__wasmsimd_rr2_p5_u12_acc2, xnn_init_f32_expminus_wasmsimd_rr2_p5_params);
    }
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_RADDSTOREEXPMINUSMAX__WASMSIMD_RR2_P5_U12_ACC3, elements_eq_12) {
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(12)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__wasmsimd_rr2_p5_u12_acc3, xnn_init_f32_expminus_wasmsimd_rr2_p5_params);
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__WASMSIMD_RR2_P5_U12_ACC3, elements_div_12) {
    for (size_t elements = 24; elements < 120; elements += 12) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__wasmsimd_rr2_p5_u12_acc3, xnn_init_f32_expminus_wasmsimd_rr2_p5_params);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__WASMSIMD_RR2_P5_U12_ACC3, elements_lt_12) {
    for (size_t elements = 1; elements < 12; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__wasmsimd_rr2_p5_u12_acc3, xnn_init_f32_expminus_wasmsimd_rr2_p5_params);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__WASMSIMD_RR2_P5_U12_ACC3, elements_gt_12) {
    for (size_t elements = 13; elements < 24; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__wasmsimd_rr2_p5_u12_acc3, xnn_init_f32_expminus_wasmsimd_rr2_p5_params);
    }
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_RADDSTOREEXPMINUSMAX__WASMSIMD_RR2_P5_U16, elements_eq_16) {
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(16)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__wasmsimd_rr2_p5_u16, xnn_init_f32_expminus_wasmsimd_rr2_p5_params);
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__WASMSIMD_RR2_P5_U16, elements_div_16) {
    for (size_t elements = 32; elements < 160; elements += 16) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__wasmsimd_rr2_p5_u16, xnn_init_f32_expminus_wasmsimd_rr2_p5_params);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__WASMSIMD_RR2_P5_U16, elements_lt_16) {
    for (size_t elements = 1; elements < 16; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__wasmsimd_rr2_p5_u16, xnn_init_f32_expminus_wasmsimd_rr2_p5_params);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__WASMSIMD_RR2_P5_U16, elements_gt_16) {
    for (size_t elements = 17; elements < 32; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__wasmsimd_rr2_p5_u16, xnn_init_f32_expminus_wasmsimd_rr2_p5_params);
    }
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_RADDSTOREEXPMINUSMAX__WASMSIMD_RR2_P5_U16_ACC2, elements_eq_16) {
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(16)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__wasmsimd_rr2_p5_u16_acc2, xnn_init_f32_expminus_wasmsimd_rr2_p5_params);
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__WASMSIMD_RR2_P5_U16_ACC2, elements_div_16) {
    for (size_t elements = 32; elements < 160; elements += 16) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__wasmsimd_rr2_p5_u16_acc2, xnn_init_f32_expminus_wasmsimd_rr2_p5_params);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__WASMSIMD_RR2_P5_U16_ACC2, elements_lt_16) {
    for (size_t elements = 1; elements < 16; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__wasmsimd_rr2_p5_u16_acc2, xnn_init_f32_expminus_wasmsimd_rr2_p5_params);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__WASMSIMD_RR2_P5_U16_ACC2, elements_gt_16) {
    for (size_t elements = 17; elements < 32; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__wasmsimd_rr2_p5_u16_acc2, xnn_init_f32_expminus_wasmsimd_rr2_p5_params);
    }
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_RADDSTOREEXPMINUSMAX__WASMSIMD_RR2_P5_U16_ACC4, elements_eq_16) {
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(16)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__wasmsimd_rr2_p5_u16_acc4, xnn_init_f32_expminus_wasmsimd_rr2_p5_params);
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__WASMSIMD_RR2_P5_U16_ACC4, elements_div_16) {
    for (size_t elements = 32; elements < 160; elements += 16) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__wasmsimd_rr2_p5_u16_acc4, xnn_init_f32_expminus_wasmsimd_rr2_p5_params);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__WASMSIMD_RR2_P5_U16_ACC4, elements_lt_16) {
    for (size_t elements = 1; elements < 16; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__wasmsimd_rr2_p5_u16_acc4, xnn_init_f32_expminus_wasmsimd_rr2_p5_params);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__WASMSIMD_RR2_P5_U16_ACC4, elements_gt_16) {
    for (size_t elements = 17; elements < 32; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__wasmsimd_rr2_p5_u16_acc4, xnn_init_f32_expminus_wasmsimd_rr2_p5_params);
    }
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_RADDSTOREEXPMINUSMAX__WASMSIMD_RR2_P5_U20, elements_eq_20) {
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(20)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__wasmsimd_rr2_p5_u20, xnn_init_f32_expminus_wasmsimd_rr2_p5_params);
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__WASMSIMD_RR2_P5_U20, elements_div_20) {
    for (size_t elements = 40; elements < 200; elements += 20) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__wasmsimd_rr2_p5_u20, xnn_init_f32_expminus_wasmsimd_rr2_p5_params);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__WASMSIMD_RR2_P5_U20, elements_lt_20) {
    for (size_t elements = 1; elements < 20; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__wasmsimd_rr2_p5_u20, xnn_init_f32_expminus_wasmsimd_rr2_p5_params);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__WASMSIMD_RR2_P5_U20, elements_gt_20) {
    for (size_t elements = 21; elements < 40; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__wasmsimd_rr2_p5_u20, xnn_init_f32_expminus_wasmsimd_rr2_p5_params);
    }
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_RADDSTOREEXPMINUSMAX__WASMSIMD_RR2_P5_U20_ACC2, elements_eq_20) {
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(20)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__wasmsimd_rr2_p5_u20_acc2, xnn_init_f32_expminus_wasmsimd_rr2_p5_params);
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__WASMSIMD_RR2_P5_U20_ACC2, elements_div_20) {
    for (size_t elements = 40; elements < 200; elements += 20) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__wasmsimd_rr2_p5_u20_acc2, xnn_init_f32_expminus_wasmsimd_rr2_p5_params);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__WASMSIMD_RR2_P5_U20_ACC2, elements_lt_20) {
    for (size_t elements = 1; elements < 20; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__wasmsimd_rr2_p5_u20_acc2, xnn_init_f32_expminus_wasmsimd_rr2_p5_params);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__WASMSIMD_RR2_P5_U20_ACC2, elements_gt_20) {
    for (size_t elements = 21; elements < 40; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__wasmsimd_rr2_p5_u20_acc2, xnn_init_f32_expminus_wasmsimd_rr2_p5_params);
    }
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_RADDSTOREEXPMINUSMAX__WASMSIMD_RR2_P5_U20_ACC5, elements_eq_20) {
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(20)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__wasmsimd_rr2_p5_u20_acc5, xnn_init_f32_expminus_wasmsimd_rr2_p5_params);
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__WASMSIMD_RR2_P5_U20_ACC5, elements_div_20) {
    for (size_t elements = 40; elements < 200; elements += 20) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__wasmsimd_rr2_p5_u20_acc5, xnn_init_f32_expminus_wasmsimd_rr2_p5_params);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__WASMSIMD_RR2_P5_U20_ACC5, elements_lt_20) {
    for (size_t elements = 1; elements < 20; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__wasmsimd_rr2_p5_u20_acc5, xnn_init_f32_expminus_wasmsimd_rr2_p5_params);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__WASMSIMD_RR2_P5_U20_ACC5, elements_gt_20) {
    for (size_t elements = 21; elements < 40; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__wasmsimd_rr2_p5_u20_acc5, xnn_init_f32_expminus_wasmsimd_rr2_p5_params);
    }
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_RADDSTOREEXPMINUSMAX__WASMRELAXEDSIMD_RR2_P5_U4, elements_eq_4) {
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(4)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__wasmrelaxedsimd_rr2_p5_u4, xnn_init_f32_expminus_wasmsimd_rr2_p5_params);
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__WASMRELAXEDSIMD_RR2_P5_U4, elements_div_4) {
    for (size_t elements = 8; elements < 40; elements += 4) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__wasmrelaxedsimd_rr2_p5_u4, xnn_init_f32_expminus_wasmsimd_rr2_p5_params);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__WASMRELAXEDSIMD_RR2_P5_U4, elements_lt_4) {
    for (size_t elements = 1; elements < 4; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__wasmrelaxedsimd_rr2_p5_u4, xnn_init_f32_expminus_wasmsimd_rr2_p5_params);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__WASMRELAXEDSIMD_RR2_P5_U4, elements_gt_4) {
    for (size_t elements = 5; elements < 8; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__wasmrelaxedsimd_rr2_p5_u4, xnn_init_f32_expminus_wasmsimd_rr2_p5_params);
    }
  }
#endif  // XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_RADDSTOREEXPMINUSMAX__WASMRELAXEDSIMD_RR2_P5_U8, elements_eq_8) {
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(8)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__wasmrelaxedsimd_rr2_p5_u8, xnn_init_f32_expminus_wasmsimd_rr2_p5_params);
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__WASMRELAXEDSIMD_RR2_P5_U8, elements_div_8) {
    for (size_t elements = 16; elements < 80; elements += 8) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__wasmrelaxedsimd_rr2_p5_u8, xnn_init_f32_expminus_wasmsimd_rr2_p5_params);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__WASMRELAXEDSIMD_RR2_P5_U8, elements_lt_8) {
    for (size_t elements = 1; elements < 8; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__wasmrelaxedsimd_rr2_p5_u8, xnn_init_f32_expminus_wasmsimd_rr2_p5_params);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__WASMRELAXEDSIMD_RR2_P5_U8, elements_gt_8) {
    for (size_t elements = 9; elements < 16; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__wasmrelaxedsimd_rr2_p5_u8, xnn_init_f32_expminus_wasmsimd_rr2_p5_params);
    }
  }
#endif  // XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_RADDSTOREEXPMINUSMAX__WASMRELAXEDSIMD_RR2_P5_U8_ACC2, elements_eq_8) {
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(8)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__wasmrelaxedsimd_rr2_p5_u8_acc2, xnn_init_f32_expminus_wasmsimd_rr2_p5_params);
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__WASMRELAXEDSIMD_RR2_P5_U8_ACC2, elements_div_8) {
    for (size_t elements = 16; elements < 80; elements += 8) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__wasmrelaxedsimd_rr2_p5_u8_acc2, xnn_init_f32_expminus_wasmsimd_rr2_p5_params);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__WASMRELAXEDSIMD_RR2_P5_U8_ACC2, elements_lt_8) {
    for (size_t elements = 1; elements < 8; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__wasmrelaxedsimd_rr2_p5_u8_acc2, xnn_init_f32_expminus_wasmsimd_rr2_p5_params);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__WASMRELAXEDSIMD_RR2_P5_U8_ACC2, elements_gt_8) {
    for (size_t elements = 9; elements < 16; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__wasmrelaxedsimd_rr2_p5_u8_acc2, xnn_init_f32_expminus_wasmsimd_rr2_p5_params);
    }
  }
#endif  // XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_RADDSTOREEXPMINUSMAX__WASMRELAXEDSIMD_RR2_P5_U12, elements_eq_12) {
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(12)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__wasmrelaxedsimd_rr2_p5_u12, xnn_init_f32_expminus_wasmsimd_rr2_p5_params);
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__WASMRELAXEDSIMD_RR2_P5_U12, elements_div_12) {
    for (size_t elements = 24; elements < 120; elements += 12) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__wasmrelaxedsimd_rr2_p5_u12, xnn_init_f32_expminus_wasmsimd_rr2_p5_params);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__WASMRELAXEDSIMD_RR2_P5_U12, elements_lt_12) {
    for (size_t elements = 1; elements < 12; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__wasmrelaxedsimd_rr2_p5_u12, xnn_init_f32_expminus_wasmsimd_rr2_p5_params);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__WASMRELAXEDSIMD_RR2_P5_U12, elements_gt_12) {
    for (size_t elements = 13; elements < 24; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__wasmrelaxedsimd_rr2_p5_u12, xnn_init_f32_expminus_wasmsimd_rr2_p5_params);
    }
  }
#endif  // XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_RADDSTOREEXPMINUSMAX__WASMRELAXEDSIMD_RR2_P5_U12_ACC2, elements_eq_12) {
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(12)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__wasmrelaxedsimd_rr2_p5_u12_acc2, xnn_init_f32_expminus_wasmsimd_rr2_p5_params);
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__WASMRELAXEDSIMD_RR2_P5_U12_ACC2, elements_div_12) {
    for (size_t elements = 24; elements < 120; elements += 12) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__wasmrelaxedsimd_rr2_p5_u12_acc2, xnn_init_f32_expminus_wasmsimd_rr2_p5_params);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__WASMRELAXEDSIMD_RR2_P5_U12_ACC2, elements_lt_12) {
    for (size_t elements = 1; elements < 12; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__wasmrelaxedsimd_rr2_p5_u12_acc2, xnn_init_f32_expminus_wasmsimd_rr2_p5_params);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__WASMRELAXEDSIMD_RR2_P5_U12_ACC2, elements_gt_12) {
    for (size_t elements = 13; elements < 24; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__wasmrelaxedsimd_rr2_p5_u12_acc2, xnn_init_f32_expminus_wasmsimd_rr2_p5_params);
    }
  }
#endif  // XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_RADDSTOREEXPMINUSMAX__WASMRELAXEDSIMD_RR2_P5_U12_ACC3, elements_eq_12) {
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(12)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__wasmrelaxedsimd_rr2_p5_u12_acc3, xnn_init_f32_expminus_wasmsimd_rr2_p5_params);
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__WASMRELAXEDSIMD_RR2_P5_U12_ACC3, elements_div_12) {
    for (size_t elements = 24; elements < 120; elements += 12) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__wasmrelaxedsimd_rr2_p5_u12_acc3, xnn_init_f32_expminus_wasmsimd_rr2_p5_params);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__WASMRELAXEDSIMD_RR2_P5_U12_ACC3, elements_lt_12) {
    for (size_t elements = 1; elements < 12; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__wasmrelaxedsimd_rr2_p5_u12_acc3, xnn_init_f32_expminus_wasmsimd_rr2_p5_params);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__WASMRELAXEDSIMD_RR2_P5_U12_ACC3, elements_gt_12) {
    for (size_t elements = 13; elements < 24; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__wasmrelaxedsimd_rr2_p5_u12_acc3, xnn_init_f32_expminus_wasmsimd_rr2_p5_params);
    }
  }
#endif  // XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_RADDSTOREEXPMINUSMAX__WASMRELAXEDSIMD_RR2_P5_U16, elements_eq_16) {
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(16)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__wasmrelaxedsimd_rr2_p5_u16, xnn_init_f32_expminus_wasmsimd_rr2_p5_params);
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__WASMRELAXEDSIMD_RR2_P5_U16, elements_div_16) {
    for (size_t elements = 32; elements < 160; elements += 16) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__wasmrelaxedsimd_rr2_p5_u16, xnn_init_f32_expminus_wasmsimd_rr2_p5_params);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__WASMRELAXEDSIMD_RR2_P5_U16, elements_lt_16) {
    for (size_t elements = 1; elements < 16; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__wasmrelaxedsimd_rr2_p5_u16, xnn_init_f32_expminus_wasmsimd_rr2_p5_params);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__WASMRELAXEDSIMD_RR2_P5_U16, elements_gt_16) {
    for (size_t elements = 17; elements < 32; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__wasmrelaxedsimd_rr2_p5_u16, xnn_init_f32_expminus_wasmsimd_rr2_p5_params);
    }
  }
#endif  // XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_RADDSTOREEXPMINUSMAX__WASMRELAXEDSIMD_RR2_P5_U16_ACC2, elements_eq_16) {
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(16)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__wasmrelaxedsimd_rr2_p5_u16_acc2, xnn_init_f32_expminus_wasmsimd_rr2_p5_params);
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__WASMRELAXEDSIMD_RR2_P5_U16_ACC2, elements_div_16) {
    for (size_t elements = 32; elements < 160; elements += 16) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__wasmrelaxedsimd_rr2_p5_u16_acc2, xnn_init_f32_expminus_wasmsimd_rr2_p5_params);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__WASMRELAXEDSIMD_RR2_P5_U16_ACC2, elements_lt_16) {
    for (size_t elements = 1; elements < 16; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__wasmrelaxedsimd_rr2_p5_u16_acc2, xnn_init_f32_expminus_wasmsimd_rr2_p5_params);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__WASMRELAXEDSIMD_RR2_P5_U16_ACC2, elements_gt_16) {
    for (size_t elements = 17; elements < 32; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__wasmrelaxedsimd_rr2_p5_u16_acc2, xnn_init_f32_expminus_wasmsimd_rr2_p5_params);
    }
  }
#endif  // XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_RADDSTOREEXPMINUSMAX__WASMRELAXEDSIMD_RR2_P5_U16_ACC4, elements_eq_16) {
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(16)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__wasmrelaxedsimd_rr2_p5_u16_acc4, xnn_init_f32_expminus_wasmsimd_rr2_p5_params);
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__WASMRELAXEDSIMD_RR2_P5_U16_ACC4, elements_div_16) {
    for (size_t elements = 32; elements < 160; elements += 16) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__wasmrelaxedsimd_rr2_p5_u16_acc4, xnn_init_f32_expminus_wasmsimd_rr2_p5_params);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__WASMRELAXEDSIMD_RR2_P5_U16_ACC4, elements_lt_16) {
    for (size_t elements = 1; elements < 16; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__wasmrelaxedsimd_rr2_p5_u16_acc4, xnn_init_f32_expminus_wasmsimd_rr2_p5_params);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__WASMRELAXEDSIMD_RR2_P5_U16_ACC4, elements_gt_16) {
    for (size_t elements = 17; elements < 32; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__wasmrelaxedsimd_rr2_p5_u16_acc4, xnn_init_f32_expminus_wasmsimd_rr2_p5_params);
    }
  }
#endif  // XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_RADDSTOREEXPMINUSMAX__WASMRELAXEDSIMD_RR2_P5_U20, elements_eq_20) {
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(20)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__wasmrelaxedsimd_rr2_p5_u20, xnn_init_f32_expminus_wasmsimd_rr2_p5_params);
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__WASMRELAXEDSIMD_RR2_P5_U20, elements_div_20) {
    for (size_t elements = 40; elements < 200; elements += 20) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__wasmrelaxedsimd_rr2_p5_u20, xnn_init_f32_expminus_wasmsimd_rr2_p5_params);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__WASMRELAXEDSIMD_RR2_P5_U20, elements_lt_20) {
    for (size_t elements = 1; elements < 20; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__wasmrelaxedsimd_rr2_p5_u20, xnn_init_f32_expminus_wasmsimd_rr2_p5_params);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__WASMRELAXEDSIMD_RR2_P5_U20, elements_gt_20) {
    for (size_t elements = 21; elements < 40; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__wasmrelaxedsimd_rr2_p5_u20, xnn_init_f32_expminus_wasmsimd_rr2_p5_params);
    }
  }
#endif  // XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_RADDSTOREEXPMINUSMAX__WASMRELAXEDSIMD_RR2_P5_U20_ACC2, elements_eq_20) {
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(20)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__wasmrelaxedsimd_rr2_p5_u20_acc2, xnn_init_f32_expminus_wasmsimd_rr2_p5_params);
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__WASMRELAXEDSIMD_RR2_P5_U20_ACC2, elements_div_20) {
    for (size_t elements = 40; elements < 200; elements += 20) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__wasmrelaxedsimd_rr2_p5_u20_acc2, xnn_init_f32_expminus_wasmsimd_rr2_p5_params);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__WASMRELAXEDSIMD_RR2_P5_U20_ACC2, elements_lt_20) {
    for (size_t elements = 1; elements < 20; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__wasmrelaxedsimd_rr2_p5_u20_acc2, xnn_init_f32_expminus_wasmsimd_rr2_p5_params);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__WASMRELAXEDSIMD_RR2_P5_U20_ACC2, elements_gt_20) {
    for (size_t elements = 21; elements < 40; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__wasmrelaxedsimd_rr2_p5_u20_acc2, xnn_init_f32_expminus_wasmsimd_rr2_p5_params);
    }
  }
#endif  // XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_RADDSTOREEXPMINUSMAX__WASMRELAXEDSIMD_RR2_P5_U20_ACC5, elements_eq_20) {
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(20)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__wasmrelaxedsimd_rr2_p5_u20_acc5, xnn_init_f32_expminus_wasmsimd_rr2_p5_params);
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__WASMRELAXEDSIMD_RR2_P5_U20_ACC5, elements_div_20) {
    for (size_t elements = 40; elements < 200; elements += 20) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__wasmrelaxedsimd_rr2_p5_u20_acc5, xnn_init_f32_expminus_wasmsimd_rr2_p5_params);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__WASMRELAXEDSIMD_RR2_P5_U20_ACC5, elements_lt_20) {
    for (size_t elements = 1; elements < 20; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__wasmrelaxedsimd_rr2_p5_u20_acc5, xnn_init_f32_expminus_wasmsimd_rr2_p5_params);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__WASMRELAXEDSIMD_RR2_P5_U20_ACC5, elements_gt_20) {
    for (size_t elements = 21; elements < 40; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__wasmrelaxedsimd_rr2_p5_u20_acc5, xnn_init_f32_expminus_wasmsimd_rr2_p5_params);
    }
  }
#endif  // XNN_ARCH_WASMRELAXEDSIMD


TEST(F32_RADDSTOREEXPMINUSMAX__SCALAR_RR2_LUT64_P2_U1, elements_eq_1) {
  RAddStoreExpMinusMaxMicrokernelTester()
    .elements(1)
    .Test(xnn_f32_raddstoreexpminusmax_ukernel__scalar_rr2_lut64_p2_u1, xnn_init_f32_expminus_scalar_rr2_lut64_p2_params);
}

TEST(F32_RADDSTOREEXPMINUSMAX__SCALAR_RR2_LUT64_P2_U1, elements_gt_1) {
  for (size_t elements = 2; elements < 10; elements++) {
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(elements)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__scalar_rr2_lut64_p2_u1, xnn_init_f32_expminus_scalar_rr2_lut64_p2_params);
  }
}

TEST(F32_RADDSTOREEXPMINUSMAX__SCALAR_RR2_LUT64_P2_U2, elements_eq_2) {
  RAddStoreExpMinusMaxMicrokernelTester()
    .elements(2)
    .Test(xnn_f32_raddstoreexpminusmax_ukernel__scalar_rr2_lut64_p2_u2, xnn_init_f32_expminus_scalar_rr2_lut64_p2_params);
}

TEST(F32_RADDSTOREEXPMINUSMAX__SCALAR_RR2_LUT64_P2_U2, elements_div_2) {
  for (size_t elements = 4; elements < 20; elements += 2) {
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(elements)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__scalar_rr2_lut64_p2_u2, xnn_init_f32_expminus_scalar_rr2_lut64_p2_params);
  }
}

TEST(F32_RADDSTOREEXPMINUSMAX__SCALAR_RR2_LUT64_P2_U2, elements_lt_2) {
  for (size_t elements = 1; elements < 2; elements++) {
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(elements)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__scalar_rr2_lut64_p2_u2, xnn_init_f32_expminus_scalar_rr2_lut64_p2_params);
  }
}

TEST(F32_RADDSTOREEXPMINUSMAX__SCALAR_RR2_LUT64_P2_U2, elements_gt_2) {
  for (size_t elements = 3; elements < 4; elements++) {
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(elements)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__scalar_rr2_lut64_p2_u2, xnn_init_f32_expminus_scalar_rr2_lut64_p2_params);
  }
}

TEST(F32_RADDSTOREEXPMINUSMAX__SCALAR_RR2_LUT64_P2_U2_ACC2, elements_eq_2) {
  RAddStoreExpMinusMaxMicrokernelTester()
    .elements(2)
    .Test(xnn_f32_raddstoreexpminusmax_ukernel__scalar_rr2_lut64_p2_u2_acc2, xnn_init_f32_expminus_scalar_rr2_lut64_p2_params);
}

TEST(F32_RADDSTOREEXPMINUSMAX__SCALAR_RR2_LUT64_P2_U2_ACC2, elements_div_2) {
  for (size_t elements = 4; elements < 20; elements += 2) {
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(elements)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__scalar_rr2_lut64_p2_u2_acc2, xnn_init_f32_expminus_scalar_rr2_lut64_p2_params);
  }
}

TEST(F32_RADDSTOREEXPMINUSMAX__SCALAR_RR2_LUT64_P2_U2_ACC2, elements_lt_2) {
  for (size_t elements = 1; elements < 2; elements++) {
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(elements)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__scalar_rr2_lut64_p2_u2_acc2, xnn_init_f32_expminus_scalar_rr2_lut64_p2_params);
  }
}

TEST(F32_RADDSTOREEXPMINUSMAX__SCALAR_RR2_LUT64_P2_U2_ACC2, elements_gt_2) {
  for (size_t elements = 3; elements < 4; elements++) {
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(elements)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__scalar_rr2_lut64_p2_u2_acc2, xnn_init_f32_expminus_scalar_rr2_lut64_p2_params);
  }
}

TEST(F32_RADDSTOREEXPMINUSMAX__SCALAR_RR2_LUT64_P2_U4, elements_eq_4) {
  RAddStoreExpMinusMaxMicrokernelTester()
    .elements(4)
    .Test(xnn_f32_raddstoreexpminusmax_ukernel__scalar_rr2_lut64_p2_u4, xnn_init_f32_expminus_scalar_rr2_lut64_p2_params);
}

TEST(F32_RADDSTOREEXPMINUSMAX__SCALAR_RR2_LUT64_P2_U4, elements_div_4) {
  for (size_t elements = 8; elements < 40; elements += 4) {
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(elements)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__scalar_rr2_lut64_p2_u4, xnn_init_f32_expminus_scalar_rr2_lut64_p2_params);
  }
}

TEST(F32_RADDSTOREEXPMINUSMAX__SCALAR_RR2_LUT64_P2_U4, elements_lt_4) {
  for (size_t elements = 1; elements < 4; elements++) {
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(elements)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__scalar_rr2_lut64_p2_u4, xnn_init_f32_expminus_scalar_rr2_lut64_p2_params);
  }
}

TEST(F32_RADDSTOREEXPMINUSMAX__SCALAR_RR2_LUT64_P2_U4, elements_gt_4) {
  for (size_t elements = 5; elements < 8; elements++) {
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(elements)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__scalar_rr2_lut64_p2_u4, xnn_init_f32_expminus_scalar_rr2_lut64_p2_params);
  }
}

TEST(F32_RADDSTOREEXPMINUSMAX__SCALAR_RR2_LUT64_P2_U4_ACC2, elements_eq_4) {
  RAddStoreExpMinusMaxMicrokernelTester()
    .elements(4)
    .Test(xnn_f32_raddstoreexpminusmax_ukernel__scalar_rr2_lut64_p2_u4_acc2, xnn_init_f32_expminus_scalar_rr2_lut64_p2_params);
}

TEST(F32_RADDSTOREEXPMINUSMAX__SCALAR_RR2_LUT64_P2_U4_ACC2, elements_div_4) {
  for (size_t elements = 8; elements < 40; elements += 4) {
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(elements)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__scalar_rr2_lut64_p2_u4_acc2, xnn_init_f32_expminus_scalar_rr2_lut64_p2_params);
  }
}

TEST(F32_RADDSTOREEXPMINUSMAX__SCALAR_RR2_LUT64_P2_U4_ACC2, elements_lt_4) {
  for (size_t elements = 1; elements < 4; elements++) {
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(elements)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__scalar_rr2_lut64_p2_u4_acc2, xnn_init_f32_expminus_scalar_rr2_lut64_p2_params);
  }
}

TEST(F32_RADDSTOREEXPMINUSMAX__SCALAR_RR2_LUT64_P2_U4_ACC2, elements_gt_4) {
  for (size_t elements = 5; elements < 8; elements++) {
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(elements)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__scalar_rr2_lut64_p2_u4_acc2, xnn_init_f32_expminus_scalar_rr2_lut64_p2_params);
  }
}

TEST(F32_RADDSTOREEXPMINUSMAX__SCALAR_RR2_LUT64_P2_U4_ACC4, elements_eq_4) {
  RAddStoreExpMinusMaxMicrokernelTester()
    .elements(4)
    .Test(xnn_f32_raddstoreexpminusmax_ukernel__scalar_rr2_lut64_p2_u4_acc4, xnn_init_f32_expminus_scalar_rr2_lut64_p2_params);
}

TEST(F32_RADDSTOREEXPMINUSMAX__SCALAR_RR2_LUT64_P2_U4_ACC4, elements_div_4) {
  for (size_t elements = 8; elements < 40; elements += 4) {
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(elements)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__scalar_rr2_lut64_p2_u4_acc4, xnn_init_f32_expminus_scalar_rr2_lut64_p2_params);
  }
}

TEST(F32_RADDSTOREEXPMINUSMAX__SCALAR_RR2_LUT64_P2_U4_ACC4, elements_lt_4) {
  for (size_t elements = 1; elements < 4; elements++) {
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(elements)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__scalar_rr2_lut64_p2_u4_acc4, xnn_init_f32_expminus_scalar_rr2_lut64_p2_params);
  }
}

TEST(F32_RADDSTOREEXPMINUSMAX__SCALAR_RR2_LUT64_P2_U4_ACC4, elements_gt_4) {
  for (size_t elements = 5; elements < 8; elements++) {
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(elements)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__scalar_rr2_lut64_p2_u4_acc4, xnn_init_f32_expminus_scalar_rr2_lut64_p2_params);
  }
}

TEST(F32_RADDSTOREEXPMINUSMAX__SCALAR_RR2_P5_U1, elements_eq_1) {
  RAddStoreExpMinusMaxMicrokernelTester()
    .elements(1)
    .Test(xnn_f32_raddstoreexpminusmax_ukernel__scalar_rr2_p5_u1, xnn_init_f32_expminus_scalar_rr2_p5_params);
}

TEST(F32_RADDSTOREEXPMINUSMAX__SCALAR_RR2_P5_U1, elements_gt_1) {
  for (size_t elements = 2; elements < 10; elements++) {
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(elements)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__scalar_rr2_p5_u1, xnn_init_f32_expminus_scalar_rr2_p5_params);
  }
}

TEST(F32_RADDSTOREEXPMINUSMAX__SCALAR_RR2_P5_U2, elements_eq_2) {
  RAddStoreExpMinusMaxMicrokernelTester()
    .elements(2)
    .Test(xnn_f32_raddstoreexpminusmax_ukernel__scalar_rr2_p5_u2, xnn_init_f32_expminus_scalar_rr2_p5_params);
}

TEST(F32_RADDSTOREEXPMINUSMAX__SCALAR_RR2_P5_U2, elements_div_2) {
  for (size_t elements = 4; elements < 20; elements += 2) {
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(elements)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__scalar_rr2_p5_u2, xnn_init_f32_expminus_scalar_rr2_p5_params);
  }
}

TEST(F32_RADDSTOREEXPMINUSMAX__SCALAR_RR2_P5_U2, elements_lt_2) {
  for (size_t elements = 1; elements < 2; elements++) {
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(elements)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__scalar_rr2_p5_u2, xnn_init_f32_expminus_scalar_rr2_p5_params);
  }
}

TEST(F32_RADDSTOREEXPMINUSMAX__SCALAR_RR2_P5_U2, elements_gt_2) {
  for (size_t elements = 3; elements < 4; elements++) {
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(elements)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__scalar_rr2_p5_u2, xnn_init_f32_expminus_scalar_rr2_p5_params);
  }
}

TEST(F32_RADDSTOREEXPMINUSMAX__SCALAR_RR2_P5_U2_ACC2, elements_eq_2) {
  RAddStoreExpMinusMaxMicrokernelTester()
    .elements(2)
    .Test(xnn_f32_raddstoreexpminusmax_ukernel__scalar_rr2_p5_u2_acc2, xnn_init_f32_expminus_scalar_rr2_p5_params);
}

TEST(F32_RADDSTOREEXPMINUSMAX__SCALAR_RR2_P5_U2_ACC2, elements_div_2) {
  for (size_t elements = 4; elements < 20; elements += 2) {
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(elements)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__scalar_rr2_p5_u2_acc2, xnn_init_f32_expminus_scalar_rr2_p5_params);
  }
}

TEST(F32_RADDSTOREEXPMINUSMAX__SCALAR_RR2_P5_U2_ACC2, elements_lt_2) {
  for (size_t elements = 1; elements < 2; elements++) {
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(elements)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__scalar_rr2_p5_u2_acc2, xnn_init_f32_expminus_scalar_rr2_p5_params);
  }
}

TEST(F32_RADDSTOREEXPMINUSMAX__SCALAR_RR2_P5_U2_ACC2, elements_gt_2) {
  for (size_t elements = 3; elements < 4; elements++) {
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(elements)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__scalar_rr2_p5_u2_acc2, xnn_init_f32_expminus_scalar_rr2_p5_params);
  }
}

TEST(F32_RADDSTOREEXPMINUSMAX__SCALAR_RR2_P5_U4, elements_eq_4) {
  RAddStoreExpMinusMaxMicrokernelTester()
    .elements(4)
    .Test(xnn_f32_raddstoreexpminusmax_ukernel__scalar_rr2_p5_u4, xnn_init_f32_expminus_scalar_rr2_p5_params);
}

TEST(F32_RADDSTOREEXPMINUSMAX__SCALAR_RR2_P5_U4, elements_div_4) {
  for (size_t elements = 8; elements < 40; elements += 4) {
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(elements)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__scalar_rr2_p5_u4, xnn_init_f32_expminus_scalar_rr2_p5_params);
  }
}

TEST(F32_RADDSTOREEXPMINUSMAX__SCALAR_RR2_P5_U4, elements_lt_4) {
  for (size_t elements = 1; elements < 4; elements++) {
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(elements)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__scalar_rr2_p5_u4, xnn_init_f32_expminus_scalar_rr2_p5_params);
  }
}

TEST(F32_RADDSTOREEXPMINUSMAX__SCALAR_RR2_P5_U4, elements_gt_4) {
  for (size_t elements = 5; elements < 8; elements++) {
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(elements)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__scalar_rr2_p5_u4, xnn_init_f32_expminus_scalar_rr2_p5_params);
  }
}

TEST(F32_RADDSTOREEXPMINUSMAX__SCALAR_RR2_P5_U4_ACC2, elements_eq_4) {
  RAddStoreExpMinusMaxMicrokernelTester()
    .elements(4)
    .Test(xnn_f32_raddstoreexpminusmax_ukernel__scalar_rr2_p5_u4_acc2, xnn_init_f32_expminus_scalar_rr2_p5_params);
}

TEST(F32_RADDSTOREEXPMINUSMAX__SCALAR_RR2_P5_U4_ACC2, elements_div_4) {
  for (size_t elements = 8; elements < 40; elements += 4) {
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(elements)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__scalar_rr2_p5_u4_acc2, xnn_init_f32_expminus_scalar_rr2_p5_params);
  }
}

TEST(F32_RADDSTOREEXPMINUSMAX__SCALAR_RR2_P5_U4_ACC2, elements_lt_4) {
  for (size_t elements = 1; elements < 4; elements++) {
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(elements)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__scalar_rr2_p5_u4_acc2, xnn_init_f32_expminus_scalar_rr2_p5_params);
  }
}

TEST(F32_RADDSTOREEXPMINUSMAX__SCALAR_RR2_P5_U4_ACC2, elements_gt_4) {
  for (size_t elements = 5; elements < 8; elements++) {
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(elements)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__scalar_rr2_p5_u4_acc2, xnn_init_f32_expminus_scalar_rr2_p5_params);
  }
}

TEST(F32_RADDSTOREEXPMINUSMAX__SCALAR_RR2_P5_U4_ACC4, elements_eq_4) {
  RAddStoreExpMinusMaxMicrokernelTester()
    .elements(4)
    .Test(xnn_f32_raddstoreexpminusmax_ukernel__scalar_rr2_p5_u4_acc4, xnn_init_f32_expminus_scalar_rr2_p5_params);
}

TEST(F32_RADDSTOREEXPMINUSMAX__SCALAR_RR2_P5_U4_ACC4, elements_div_4) {
  for (size_t elements = 8; elements < 40; elements += 4) {
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(elements)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__scalar_rr2_p5_u4_acc4, xnn_init_f32_expminus_scalar_rr2_p5_params);
  }
}

TEST(F32_RADDSTOREEXPMINUSMAX__SCALAR_RR2_P5_U4_ACC4, elements_lt_4) {
  for (size_t elements = 1; elements < 4; elements++) {
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(elements)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__scalar_rr2_p5_u4_acc4, xnn_init_f32_expminus_scalar_rr2_p5_params);
  }
}

TEST(F32_RADDSTOREEXPMINUSMAX__SCALAR_RR2_P5_U4_ACC4, elements_gt_4) {
  for (size_t elements = 5; elements < 8; elements++) {
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(elements)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__scalar_rr2_p5_u4_acc4, xnn_init_f32_expminus_scalar_rr2_p5_params);
  }
}