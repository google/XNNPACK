// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: test/f32-raddstoreexpminusmax.yaml
//   Generator: tools/generate-raddstoreexpminusmax-test.py


#include <gtest/gtest.h>
#include "xnnpack/common.h"
#include "xnnpack/isa-checks.h"
#include "xnnpack/microparams-init.h"
#include "xnnpack/raddstoreexpminusmax.h"
#include "raddstoreexpminusmax-microkernel-tester.h"


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_RADDSTOREEXPMINUSMAX__NEON_RR2_LUT64_P2_U4, elements_eq_4) {
    TEST_REQUIRES_ARM_NEON;
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(4)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__neon_rr2_lut64_p2_u4, nullptr);
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEON_RR2_LUT64_P2_U4, elements_div_4) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t elements = 8; elements < 40; elements += 4) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neon_rr2_lut64_p2_u4, nullptr);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEON_RR2_LUT64_P2_U4, elements_lt_4) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t elements = 1; elements < 4; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neon_rr2_lut64_p2_u4, nullptr);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEON_RR2_LUT64_P2_U4, elements_gt_4) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t elements = 5; elements < 8; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neon_rr2_lut64_p2_u4, nullptr);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_RADDSTOREEXPMINUSMAX__NEON_RR2_LUT64_P2_U8_ACC2, elements_eq_8) {
    TEST_REQUIRES_ARM_NEON;
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(8)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__neon_rr2_lut64_p2_u8_acc2, nullptr);
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEON_RR2_LUT64_P2_U8_ACC2, elements_div_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t elements = 16; elements < 80; elements += 8) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neon_rr2_lut64_p2_u8_acc2, nullptr);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEON_RR2_LUT64_P2_U8_ACC2, elements_lt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t elements = 1; elements < 8; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neon_rr2_lut64_p2_u8_acc2, nullptr);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEON_RR2_LUT64_P2_U8_ACC2, elements_gt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t elements = 9; elements < 16; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neon_rr2_lut64_p2_u8_acc2, nullptr);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_RADDSTOREEXPMINUSMAX__NEON_RR2_LUT64_P2_U16_ACC2, elements_eq_16) {
    TEST_REQUIRES_ARM_NEON;
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(16)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__neon_rr2_lut64_p2_u16_acc2, nullptr);
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEON_RR2_LUT64_P2_U16_ACC2, elements_div_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t elements = 32; elements < 160; elements += 16) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neon_rr2_lut64_p2_u16_acc2, nullptr);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEON_RR2_LUT64_P2_U16_ACC2, elements_lt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t elements = 1; elements < 16; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neon_rr2_lut64_p2_u16_acc2, nullptr);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEON_RR2_LUT64_P2_U16_ACC2, elements_gt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t elements = 17; elements < 32; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neon_rr2_lut64_p2_u16_acc2, nullptr);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_RADDSTOREEXPMINUSMAX__NEON_RR2_LUT64_P2_U16_ACC4, elements_eq_16) {
    TEST_REQUIRES_ARM_NEON;
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(16)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__neon_rr2_lut64_p2_u16_acc4, nullptr);
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEON_RR2_LUT64_P2_U16_ACC4, elements_div_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t elements = 32; elements < 160; elements += 16) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neon_rr2_lut64_p2_u16_acc4, nullptr);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEON_RR2_LUT64_P2_U16_ACC4, elements_lt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t elements = 1; elements < 16; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neon_rr2_lut64_p2_u16_acc4, nullptr);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEON_RR2_LUT64_P2_U16_ACC4, elements_gt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t elements = 17; elements < 32; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neon_rr2_lut64_p2_u16_acc4, nullptr);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_RADDSTOREEXPMINUSMAX__NEON_RR2_P5_U4, elements_eq_4) {
    TEST_REQUIRES_ARM_NEON;
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(4)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__neon_rr2_p5_u4, nullptr);
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEON_RR2_P5_U4, elements_div_4) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t elements = 8; elements < 40; elements += 4) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neon_rr2_p5_u4, nullptr);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEON_RR2_P5_U4, elements_lt_4) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t elements = 1; elements < 4; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neon_rr2_p5_u4, nullptr);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEON_RR2_P5_U4, elements_gt_4) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t elements = 5; elements < 8; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neon_rr2_p5_u4, nullptr);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_RADDSTOREEXPMINUSMAX__NEON_RR2_P5_U8_ACC2, elements_eq_8) {
    TEST_REQUIRES_ARM_NEON;
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(8)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__neon_rr2_p5_u8_acc2, nullptr);
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEON_RR2_P5_U8_ACC2, elements_div_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t elements = 16; elements < 80; elements += 8) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neon_rr2_p5_u8_acc2, nullptr);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEON_RR2_P5_U8_ACC2, elements_lt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t elements = 1; elements < 8; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neon_rr2_p5_u8_acc2, nullptr);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEON_RR2_P5_U8_ACC2, elements_gt_8) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t elements = 9; elements < 16; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neon_rr2_p5_u8_acc2, nullptr);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_RADDSTOREEXPMINUSMAX__NEON_RR2_P5_U16_ACC2, elements_eq_16) {
    TEST_REQUIRES_ARM_NEON;
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(16)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__neon_rr2_p5_u16_acc2, nullptr);
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEON_RR2_P5_U16_ACC2, elements_div_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t elements = 32; elements < 160; elements += 16) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neon_rr2_p5_u16_acc2, nullptr);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEON_RR2_P5_U16_ACC2, elements_lt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t elements = 1; elements < 16; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neon_rr2_p5_u16_acc2, nullptr);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEON_RR2_P5_U16_ACC2, elements_gt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t elements = 17; elements < 32; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neon_rr2_p5_u16_acc2, nullptr);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_RADDSTOREEXPMINUSMAX__NEON_RR2_P5_U16_ACC4, elements_eq_16) {
    TEST_REQUIRES_ARM_NEON;
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(16)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__neon_rr2_p5_u16_acc4, nullptr);
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEON_RR2_P5_U16_ACC4, elements_div_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t elements = 32; elements < 160; elements += 16) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neon_rr2_p5_u16_acc4, nullptr);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEON_RR2_P5_U16_ACC4, elements_lt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t elements = 1; elements < 16; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neon_rr2_p5_u16_acc4, nullptr);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEON_RR2_P5_U16_ACC4, elements_gt_16) {
    TEST_REQUIRES_ARM_NEON;
    for (size_t elements = 17; elements < 32; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neon_rr2_p5_u16_acc4, nullptr);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_RADDSTOREEXPMINUSMAX__NEONFMA_RR1_LUT64_P2_U4, elements_eq_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(4)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__neonfma_rr1_lut64_p2_u4, nullptr);
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEONFMA_RR1_LUT64_P2_U4, elements_div_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t elements = 8; elements < 40; elements += 4) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neonfma_rr1_lut64_p2_u4, nullptr);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEONFMA_RR1_LUT64_P2_U4, elements_lt_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t elements = 1; elements < 4; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neonfma_rr1_lut64_p2_u4, nullptr);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEONFMA_RR1_LUT64_P2_U4, elements_gt_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t elements = 5; elements < 8; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neonfma_rr1_lut64_p2_u4, nullptr);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_RADDSTOREEXPMINUSMAX__NEONFMA_RR1_LUT64_P2_U8_ACC2, elements_eq_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(8)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__neonfma_rr1_lut64_p2_u8_acc2, nullptr);
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEONFMA_RR1_LUT64_P2_U8_ACC2, elements_div_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t elements = 16; elements < 80; elements += 8) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neonfma_rr1_lut64_p2_u8_acc2, nullptr);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEONFMA_RR1_LUT64_P2_U8_ACC2, elements_lt_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t elements = 1; elements < 8; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neonfma_rr1_lut64_p2_u8_acc2, nullptr);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEONFMA_RR1_LUT64_P2_U8_ACC2, elements_gt_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t elements = 9; elements < 16; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neonfma_rr1_lut64_p2_u8_acc2, nullptr);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_RADDSTOREEXPMINUSMAX__NEONFMA_RR1_LUT64_P2_U16_ACC2, elements_eq_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(16)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__neonfma_rr1_lut64_p2_u16_acc2, nullptr);
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEONFMA_RR1_LUT64_P2_U16_ACC2, elements_div_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t elements = 32; elements < 160; elements += 16) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neonfma_rr1_lut64_p2_u16_acc2, nullptr);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEONFMA_RR1_LUT64_P2_U16_ACC2, elements_lt_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t elements = 1; elements < 16; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neonfma_rr1_lut64_p2_u16_acc2, nullptr);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEONFMA_RR1_LUT64_P2_U16_ACC2, elements_gt_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t elements = 17; elements < 32; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neonfma_rr1_lut64_p2_u16_acc2, nullptr);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_RADDSTOREEXPMINUSMAX__NEONFMA_RR1_LUT64_P2_U16_ACC4, elements_eq_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(16)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__neonfma_rr1_lut64_p2_u16_acc4, nullptr);
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEONFMA_RR1_LUT64_P2_U16_ACC4, elements_div_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t elements = 32; elements < 160; elements += 16) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neonfma_rr1_lut64_p2_u16_acc4, nullptr);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEONFMA_RR1_LUT64_P2_U16_ACC4, elements_lt_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t elements = 1; elements < 16; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neonfma_rr1_lut64_p2_u16_acc4, nullptr);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEONFMA_RR1_LUT64_P2_U16_ACC4, elements_gt_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t elements = 17; elements < 32; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neonfma_rr1_lut64_p2_u16_acc4, nullptr);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_RADDSTOREEXPMINUSMAX__NEONFMA_RR1_P5_U4, elements_eq_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(4)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__neonfma_rr1_p5_u4, nullptr);
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEONFMA_RR1_P5_U4, elements_div_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t elements = 8; elements < 40; elements += 4) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neonfma_rr1_p5_u4, nullptr);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEONFMA_RR1_P5_U4, elements_lt_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t elements = 1; elements < 4; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neonfma_rr1_p5_u4, nullptr);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEONFMA_RR1_P5_U4, elements_gt_4) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t elements = 5; elements < 8; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neonfma_rr1_p5_u4, nullptr);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_RADDSTOREEXPMINUSMAX__NEONFMA_RR1_P5_U8_ACC2, elements_eq_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(8)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__neonfma_rr1_p5_u8_acc2, nullptr);
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEONFMA_RR1_P5_U8_ACC2, elements_div_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t elements = 16; elements < 80; elements += 8) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neonfma_rr1_p5_u8_acc2, nullptr);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEONFMA_RR1_P5_U8_ACC2, elements_lt_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t elements = 1; elements < 8; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neonfma_rr1_p5_u8_acc2, nullptr);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEONFMA_RR1_P5_U8_ACC2, elements_gt_8) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t elements = 9; elements < 16; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neonfma_rr1_p5_u8_acc2, nullptr);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_RADDSTOREEXPMINUSMAX__NEONFMA_RR1_P5_U16_ACC2, elements_eq_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(16)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__neonfma_rr1_p5_u16_acc2, nullptr);
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEONFMA_RR1_P5_U16_ACC2, elements_div_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t elements = 32; elements < 160; elements += 16) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neonfma_rr1_p5_u16_acc2, nullptr);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEONFMA_RR1_P5_U16_ACC2, elements_lt_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t elements = 1; elements < 16; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neonfma_rr1_p5_u16_acc2, nullptr);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEONFMA_RR1_P5_U16_ACC2, elements_gt_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t elements = 17; elements < 32; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neonfma_rr1_p5_u16_acc2, nullptr);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  TEST(F32_RADDSTOREEXPMINUSMAX__NEONFMA_RR1_P5_U16_ACC4, elements_eq_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(16)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__neonfma_rr1_p5_u16_acc4, nullptr);
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEONFMA_RR1_P5_U16_ACC4, elements_div_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t elements = 32; elements < 160; elements += 16) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neonfma_rr1_p5_u16_acc4, nullptr);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEONFMA_RR1_P5_U16_ACC4, elements_lt_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t elements = 1; elements < 16; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neonfma_rr1_p5_u16_acc4, nullptr);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__NEONFMA_RR1_P5_U16_ACC4, elements_gt_16) {
    TEST_REQUIRES_ARM_NEON_FMA;
    for (size_t elements = 17; elements < 32; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__neonfma_rr1_p5_u16_acc4, nullptr);
    }
  }
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64


#if XNN_ENABLE_RISCV_VECTOR && XNN_ARCH_RISCV
  TEST(F32_RADDSTOREEXPMINUSMAX__RVV_RR2_P6_U2V, elements_eq_2v) {
    TEST_REQUIRES_RISCV_VECTOR;
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(2 * xnn_init_hardware_config()->vlenb / sizeof(float))
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__rvv_rr2_p6_u2v, nullptr);
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__RVV_RR2_P6_U2V, elements_div_2v) {
    TEST_REQUIRES_RISCV_VECTOR;
    for (size_t elements = 4 * xnn_init_hardware_config()->vlenb / sizeof(float);
                elements < 20 * xnn_init_hardware_config()->vlenb / sizeof(float);
                elements += 2 * xnn_init_hardware_config()->vlenb / sizeof(float)) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__rvv_rr2_p6_u2v, nullptr);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__RVV_RR2_P6_U2V, elements_lt_2v) {
    TEST_REQUIRES_RISCV_VECTOR;
    for (size_t elements = 1;
                elements < 2 * xnn_init_hardware_config()->vlenb / sizeof(float);
                elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__rvv_rr2_p6_u2v, nullptr);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__RVV_RR2_P6_U2V, elements_gt_2v) {
    TEST_REQUIRES_RISCV_VECTOR;
    for (size_t elements = 2 * xnn_init_hardware_config()->vlenb / sizeof(float) + 1;
                elements < 4 * xnn_init_hardware_config()->vlenb / sizeof(float);
                elements += 4) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__rvv_rr2_p6_u2v, nullptr);
    }
  }
#endif  // XNN_ENABLE_RISCV_VECTOR && XNN_ARCH_RISCV


#if XNN_ENABLE_RISCV_VECTOR && XNN_ARCH_RISCV
  TEST(F32_RADDSTOREEXPMINUSMAX__RVV_RR2_P6_U4V, elements_eq_4v) {
    TEST_REQUIRES_RISCV_VECTOR;
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(4 * xnn_init_hardware_config()->vlenb / sizeof(float))
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__rvv_rr2_p6_u4v, nullptr);
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__RVV_RR2_P6_U4V, elements_div_4v) {
    TEST_REQUIRES_RISCV_VECTOR;
    for (size_t elements = 8 * xnn_init_hardware_config()->vlenb / sizeof(float);
                elements < 40 * xnn_init_hardware_config()->vlenb / sizeof(float);
                elements += 4 * xnn_init_hardware_config()->vlenb / sizeof(float)) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__rvv_rr2_p6_u4v, nullptr);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__RVV_RR2_P6_U4V, elements_lt_4v) {
    TEST_REQUIRES_RISCV_VECTOR;
    for (size_t elements = 1;
                elements < 4 * xnn_init_hardware_config()->vlenb / sizeof(float);
                elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__rvv_rr2_p6_u4v, nullptr);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__RVV_RR2_P6_U4V, elements_gt_4v) {
    TEST_REQUIRES_RISCV_VECTOR;
    for (size_t elements = 4 * xnn_init_hardware_config()->vlenb / sizeof(float) + 1;
                elements < 8 * xnn_init_hardware_config()->vlenb / sizeof(float);
                elements += 8) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__rvv_rr2_p6_u4v, nullptr);
    }
  }
#endif  // XNN_ENABLE_RISCV_VECTOR && XNN_ARCH_RISCV


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_RADDSTOREEXPMINUSMAX__SSE2_RR2_P5_U4, elements_eq_4) {
    TEST_REQUIRES_X86_SSE2;
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(4)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__sse2_rr2_p5_u4, nullptr);
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__SSE2_RR2_P5_U4, elements_div_4) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t elements = 8; elements < 40; elements += 4) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__sse2_rr2_p5_u4, nullptr);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__SSE2_RR2_P5_U4, elements_lt_4) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t elements = 1; elements < 4; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__sse2_rr2_p5_u4, nullptr);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__SSE2_RR2_P5_U4, elements_gt_4) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t elements = 5; elements < 8; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__sse2_rr2_p5_u4, nullptr);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_RADDSTOREEXPMINUSMAX__SSE2_RR2_P5_U8_ACC2, elements_eq_8) {
    TEST_REQUIRES_X86_SSE2;
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(8)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__sse2_rr2_p5_u8_acc2, nullptr);
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__SSE2_RR2_P5_U8_ACC2, elements_div_8) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t elements = 16; elements < 80; elements += 8) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__sse2_rr2_p5_u8_acc2, nullptr);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__SSE2_RR2_P5_U8_ACC2, elements_lt_8) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t elements = 1; elements < 8; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__sse2_rr2_p5_u8_acc2, nullptr);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__SSE2_RR2_P5_U8_ACC2, elements_gt_8) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t elements = 9; elements < 16; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__sse2_rr2_p5_u8_acc2, nullptr);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_RADDSTOREEXPMINUSMAX__SSE2_RR2_P5_U16_ACC2, elements_eq_16) {
    TEST_REQUIRES_X86_SSE2;
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(16)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__sse2_rr2_p5_u16_acc2, nullptr);
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__SSE2_RR2_P5_U16_ACC2, elements_div_16) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t elements = 32; elements < 160; elements += 16) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__sse2_rr2_p5_u16_acc2, nullptr);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__SSE2_RR2_P5_U16_ACC2, elements_lt_16) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t elements = 1; elements < 16; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__sse2_rr2_p5_u16_acc2, nullptr);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__SSE2_RR2_P5_U16_ACC2, elements_gt_16) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t elements = 17; elements < 32; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__sse2_rr2_p5_u16_acc2, nullptr);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_RADDSTOREEXPMINUSMAX__SSE2_RR2_P5_U16_ACC4, elements_eq_16) {
    TEST_REQUIRES_X86_SSE2;
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(16)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__sse2_rr2_p5_u16_acc4, nullptr);
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__SSE2_RR2_P5_U16_ACC4, elements_div_16) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t elements = 32; elements < 160; elements += 16) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__sse2_rr2_p5_u16_acc4, nullptr);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__SSE2_RR2_P5_U16_ACC4, elements_lt_16) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t elements = 1; elements < 16; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__sse2_rr2_p5_u16_acc4, nullptr);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__SSE2_RR2_P5_U16_ACC4, elements_gt_16) {
    TEST_REQUIRES_X86_SSE2;
    for (size_t elements = 17; elements < 32; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__sse2_rr2_p5_u16_acc4, nullptr);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_RADDSTOREEXPMINUSMAX__AVX2_RR1_P5_U8, elements_eq_8) {
    TEST_REQUIRES_X86_AVX2;
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(8)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__avx2_rr1_p5_u8, nullptr);
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__AVX2_RR1_P5_U8, elements_div_8) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t elements = 16; elements < 80; elements += 8) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__avx2_rr1_p5_u8, nullptr);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__AVX2_RR1_P5_U8, elements_lt_8) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t elements = 1; elements < 8; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__avx2_rr1_p5_u8, nullptr);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__AVX2_RR1_P5_U8, elements_gt_8) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t elements = 9; elements < 16; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__avx2_rr1_p5_u8, nullptr);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_RADDSTOREEXPMINUSMAX__AVX2_RR1_P5_U16_ACC2, elements_eq_16) {
    TEST_REQUIRES_X86_AVX2;
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(16)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__avx2_rr1_p5_u16_acc2, nullptr);
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__AVX2_RR1_P5_U16_ACC2, elements_div_16) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t elements = 32; elements < 160; elements += 16) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__avx2_rr1_p5_u16_acc2, nullptr);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__AVX2_RR1_P5_U16_ACC2, elements_lt_16) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t elements = 1; elements < 16; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__avx2_rr1_p5_u16_acc2, nullptr);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__AVX2_RR1_P5_U16_ACC2, elements_gt_16) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t elements = 17; elements < 32; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__avx2_rr1_p5_u16_acc2, nullptr);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_RADDSTOREEXPMINUSMAX__AVX2_RR1_P5_U32_ACC2, elements_eq_32) {
    TEST_REQUIRES_X86_AVX2;
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(32)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__avx2_rr1_p5_u32_acc2, nullptr);
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__AVX2_RR1_P5_U32_ACC2, elements_div_32) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t elements = 64; elements < 320; elements += 32) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__avx2_rr1_p5_u32_acc2, nullptr);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__AVX2_RR1_P5_U32_ACC2, elements_lt_32) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t elements = 1; elements < 32; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__avx2_rr1_p5_u32_acc2, nullptr);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__AVX2_RR1_P5_U32_ACC2, elements_gt_32) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t elements = 33; elements < 64; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__avx2_rr1_p5_u32_acc2, nullptr);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_RADDSTOREEXPMINUSMAX__AVX2_RR1_P5_U32_ACC4, elements_eq_32) {
    TEST_REQUIRES_X86_AVX2;
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(32)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__avx2_rr1_p5_u32_acc4, nullptr);
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__AVX2_RR1_P5_U32_ACC4, elements_div_32) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t elements = 64; elements < 320; elements += 32) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__avx2_rr1_p5_u32_acc4, nullptr);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__AVX2_RR1_P5_U32_ACC4, elements_lt_32) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t elements = 1; elements < 32; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__avx2_rr1_p5_u32_acc4, nullptr);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__AVX2_RR1_P5_U32_ACC4, elements_gt_32) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t elements = 33; elements < 64; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__avx2_rr1_p5_u32_acc4, nullptr);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_RADDSTOREEXPMINUSMAX__AVX2_RR2_P5_U8, elements_eq_8) {
    TEST_REQUIRES_X86_AVX2;
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(8)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__avx2_rr2_p5_u8, nullptr);
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__AVX2_RR2_P5_U8, elements_div_8) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t elements = 16; elements < 80; elements += 8) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__avx2_rr2_p5_u8, nullptr);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__AVX2_RR2_P5_U8, elements_lt_8) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t elements = 1; elements < 8; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__avx2_rr2_p5_u8, nullptr);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__AVX2_RR2_P5_U8, elements_gt_8) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t elements = 9; elements < 16; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__avx2_rr2_p5_u8, nullptr);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_RADDSTOREEXPMINUSMAX__AVX2_RR2_P5_U16_ACC2, elements_eq_16) {
    TEST_REQUIRES_X86_AVX2;
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(16)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__avx2_rr2_p5_u16_acc2, nullptr);
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__AVX2_RR2_P5_U16_ACC2, elements_div_16) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t elements = 32; elements < 160; elements += 16) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__avx2_rr2_p5_u16_acc2, nullptr);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__AVX2_RR2_P5_U16_ACC2, elements_lt_16) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t elements = 1; elements < 16; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__avx2_rr2_p5_u16_acc2, nullptr);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__AVX2_RR2_P5_U16_ACC2, elements_gt_16) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t elements = 17; elements < 32; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__avx2_rr2_p5_u16_acc2, nullptr);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_RADDSTOREEXPMINUSMAX__AVX2_RR2_P5_U32_ACC2, elements_eq_32) {
    TEST_REQUIRES_X86_AVX2;
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(32)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__avx2_rr2_p5_u32_acc2, nullptr);
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__AVX2_RR2_P5_U32_ACC2, elements_div_32) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t elements = 64; elements < 320; elements += 32) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__avx2_rr2_p5_u32_acc2, nullptr);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__AVX2_RR2_P5_U32_ACC2, elements_lt_32) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t elements = 1; elements < 32; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__avx2_rr2_p5_u32_acc2, nullptr);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__AVX2_RR2_P5_U32_ACC2, elements_gt_32) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t elements = 33; elements < 64; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__avx2_rr2_p5_u32_acc2, nullptr);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_RADDSTOREEXPMINUSMAX__AVX2_RR2_P5_U32_ACC4, elements_eq_32) {
    TEST_REQUIRES_X86_AVX2;
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(32)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__avx2_rr2_p5_u32_acc4, nullptr);
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__AVX2_RR2_P5_U32_ACC4, elements_div_32) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t elements = 64; elements < 320; elements += 32) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__avx2_rr2_p5_u32_acc4, nullptr);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__AVX2_RR2_P5_U32_ACC4, elements_lt_32) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t elements = 1; elements < 32; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__avx2_rr2_p5_u32_acc4, nullptr);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__AVX2_RR2_P5_U32_ACC4, elements_gt_32) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t elements = 33; elements < 64; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__avx2_rr2_p5_u32_acc4, nullptr);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ENABLE_AVX256SKX && (XNN_ARCH_X86 || XNN_ARCH_X86_64)
  TEST(F32_RADDSTOREEXPMINUSMAX__AVX256SKX_RR2_P5_U8, elements_eq_8) {
    TEST_REQUIRES_X86_AVX256SKX;
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(8)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__avx256skx_rr2_p5_u8, nullptr);
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__AVX256SKX_RR2_P5_U8, elements_div_8) {
    TEST_REQUIRES_X86_AVX256SKX;
    for (size_t elements = 16; elements < 80; elements += 8) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__avx256skx_rr2_p5_u8, nullptr);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__AVX256SKX_RR2_P5_U8, elements_lt_8) {
    TEST_REQUIRES_X86_AVX256SKX;
    for (size_t elements = 1; elements < 8; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__avx256skx_rr2_p5_u8, nullptr);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__AVX256SKX_RR2_P5_U8, elements_gt_8) {
    TEST_REQUIRES_X86_AVX256SKX;
    for (size_t elements = 9; elements < 16; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__avx256skx_rr2_p5_u8, nullptr);
    }
  }
#endif  // XNN_ENABLE_AVX256SKX && (XNN_ARCH_X86 || XNN_ARCH_X86_64)


#if XNN_ENABLE_AVX256SKX && (XNN_ARCH_X86 || XNN_ARCH_X86_64)
  TEST(F32_RADDSTOREEXPMINUSMAX__AVX256SKX_RR2_P5_U16_ACC2, elements_eq_16) {
    TEST_REQUIRES_X86_AVX256SKX;
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(16)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__avx256skx_rr2_p5_u16_acc2, nullptr);
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__AVX256SKX_RR2_P5_U16_ACC2, elements_div_16) {
    TEST_REQUIRES_X86_AVX256SKX;
    for (size_t elements = 32; elements < 160; elements += 16) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__avx256skx_rr2_p5_u16_acc2, nullptr);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__AVX256SKX_RR2_P5_U16_ACC2, elements_lt_16) {
    TEST_REQUIRES_X86_AVX256SKX;
    for (size_t elements = 1; elements < 16; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__avx256skx_rr2_p5_u16_acc2, nullptr);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__AVX256SKX_RR2_P5_U16_ACC2, elements_gt_16) {
    TEST_REQUIRES_X86_AVX256SKX;
    for (size_t elements = 17; elements < 32; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__avx256skx_rr2_p5_u16_acc2, nullptr);
    }
  }
#endif  // XNN_ENABLE_AVX256SKX && (XNN_ARCH_X86 || XNN_ARCH_X86_64)


#if XNN_ENABLE_AVX256SKX && (XNN_ARCH_X86 || XNN_ARCH_X86_64)
  TEST(F32_RADDSTOREEXPMINUSMAX__AVX256SKX_RR2_P5_U32_ACC2, elements_eq_32) {
    TEST_REQUIRES_X86_AVX256SKX;
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(32)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__avx256skx_rr2_p5_u32_acc2, nullptr);
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__AVX256SKX_RR2_P5_U32_ACC2, elements_div_32) {
    TEST_REQUIRES_X86_AVX256SKX;
    for (size_t elements = 64; elements < 320; elements += 32) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__avx256skx_rr2_p5_u32_acc2, nullptr);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__AVX256SKX_RR2_P5_U32_ACC2, elements_lt_32) {
    TEST_REQUIRES_X86_AVX256SKX;
    for (size_t elements = 1; elements < 32; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__avx256skx_rr2_p5_u32_acc2, nullptr);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__AVX256SKX_RR2_P5_U32_ACC2, elements_gt_32) {
    TEST_REQUIRES_X86_AVX256SKX;
    for (size_t elements = 33; elements < 64; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__avx256skx_rr2_p5_u32_acc2, nullptr);
    }
  }
#endif  // XNN_ENABLE_AVX256SKX && (XNN_ARCH_X86 || XNN_ARCH_X86_64)


#if XNN_ENABLE_AVX256SKX && (XNN_ARCH_X86 || XNN_ARCH_X86_64)
  TEST(F32_RADDSTOREEXPMINUSMAX__AVX256SKX_RR2_P5_U32_ACC4, elements_eq_32) {
    TEST_REQUIRES_X86_AVX256SKX;
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(32)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__avx256skx_rr2_p5_u32_acc4, nullptr);
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__AVX256SKX_RR2_P5_U32_ACC4, elements_div_32) {
    TEST_REQUIRES_X86_AVX256SKX;
    for (size_t elements = 64; elements < 320; elements += 32) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__avx256skx_rr2_p5_u32_acc4, nullptr);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__AVX256SKX_RR2_P5_U32_ACC4, elements_lt_32) {
    TEST_REQUIRES_X86_AVX256SKX;
    for (size_t elements = 1; elements < 32; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__avx256skx_rr2_p5_u32_acc4, nullptr);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__AVX256SKX_RR2_P5_U32_ACC4, elements_gt_32) {
    TEST_REQUIRES_X86_AVX256SKX;
    for (size_t elements = 33; elements < 64; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__avx256skx_rr2_p5_u32_acc4, nullptr);
    }
  }
#endif  // XNN_ENABLE_AVX256SKX && (XNN_ARCH_X86 || XNN_ARCH_X86_64)


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_RADDSTOREEXPMINUSMAX__AVX512F_RR1_P5_SCALEF_U16, elements_eq_16) {
    TEST_REQUIRES_X86_AVX512F;
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(16)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__avx512f_rr1_p5_scalef_u16, nullptr);
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__AVX512F_RR1_P5_SCALEF_U16, elements_div_16) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t elements = 32; elements < 160; elements += 16) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__avx512f_rr1_p5_scalef_u16, nullptr);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__AVX512F_RR1_P5_SCALEF_U16, elements_lt_16) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t elements = 1; elements < 16; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__avx512f_rr1_p5_scalef_u16, nullptr);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__AVX512F_RR1_P5_SCALEF_U16, elements_gt_16) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t elements = 17; elements < 32; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__avx512f_rr1_p5_scalef_u16, nullptr);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_RADDSTOREEXPMINUSMAX__AVX512F_RR1_P5_SCALEF_U32_ACC2, elements_eq_32) {
    TEST_REQUIRES_X86_AVX512F;
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(32)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__avx512f_rr1_p5_scalef_u32_acc2, nullptr);
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__AVX512F_RR1_P5_SCALEF_U32_ACC2, elements_div_32) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t elements = 64; elements < 320; elements += 32) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__avx512f_rr1_p5_scalef_u32_acc2, nullptr);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__AVX512F_RR1_P5_SCALEF_U32_ACC2, elements_lt_32) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t elements = 1; elements < 32; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__avx512f_rr1_p5_scalef_u32_acc2, nullptr);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__AVX512F_RR1_P5_SCALEF_U32_ACC2, elements_gt_32) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t elements = 33; elements < 64; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__avx512f_rr1_p5_scalef_u32_acc2, nullptr);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_RADDSTOREEXPMINUSMAX__AVX512F_RR1_P5_SCALEF_U64_ACC2, elements_eq_64) {
    TEST_REQUIRES_X86_AVX512F;
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(64)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__avx512f_rr1_p5_scalef_u64_acc2, nullptr);
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__AVX512F_RR1_P5_SCALEF_U64_ACC2, elements_div_64) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t elements = 128; elements < 640; elements += 64) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__avx512f_rr1_p5_scalef_u64_acc2, nullptr);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__AVX512F_RR1_P5_SCALEF_U64_ACC2, elements_lt_64) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t elements = 1; elements < 64; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__avx512f_rr1_p5_scalef_u64_acc2, nullptr);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__AVX512F_RR1_P5_SCALEF_U64_ACC2, elements_gt_64) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t elements = 65; elements < 128; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__avx512f_rr1_p5_scalef_u64_acc2, nullptr);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_RADDSTOREEXPMINUSMAX__AVX512F_RR1_P5_SCALEF_U64_ACC4, elements_eq_64) {
    TEST_REQUIRES_X86_AVX512F;
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(64)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__avx512f_rr1_p5_scalef_u64_acc4, nullptr);
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__AVX512F_RR1_P5_SCALEF_U64_ACC4, elements_div_64) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t elements = 128; elements < 640; elements += 64) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__avx512f_rr1_p5_scalef_u64_acc4, nullptr);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__AVX512F_RR1_P5_SCALEF_U64_ACC4, elements_lt_64) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t elements = 1; elements < 64; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__avx512f_rr1_p5_scalef_u64_acc4, nullptr);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__AVX512F_RR1_P5_SCALEF_U64_ACC4, elements_gt_64) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t elements = 65; elements < 128; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__avx512f_rr1_p5_scalef_u64_acc4, nullptr);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_RADDSTOREEXPMINUSMAX__AVX512F_RR2_P5_U16, elements_eq_16) {
    TEST_REQUIRES_X86_AVX512F;
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(16)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__avx512f_rr2_p5_u16, nullptr);
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__AVX512F_RR2_P5_U16, elements_div_16) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t elements = 32; elements < 160; elements += 16) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__avx512f_rr2_p5_u16, nullptr);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__AVX512F_RR2_P5_U16, elements_lt_16) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t elements = 1; elements < 16; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__avx512f_rr2_p5_u16, nullptr);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__AVX512F_RR2_P5_U16, elements_gt_16) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t elements = 17; elements < 32; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__avx512f_rr2_p5_u16, nullptr);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_RADDSTOREEXPMINUSMAX__AVX512F_RR2_P5_U32_ACC2, elements_eq_32) {
    TEST_REQUIRES_X86_AVX512F;
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(32)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__avx512f_rr2_p5_u32_acc2, nullptr);
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__AVX512F_RR2_P5_U32_ACC2, elements_div_32) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t elements = 64; elements < 320; elements += 32) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__avx512f_rr2_p5_u32_acc2, nullptr);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__AVX512F_RR2_P5_U32_ACC2, elements_lt_32) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t elements = 1; elements < 32; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__avx512f_rr2_p5_u32_acc2, nullptr);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__AVX512F_RR2_P5_U32_ACC2, elements_gt_32) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t elements = 33; elements < 64; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__avx512f_rr2_p5_u32_acc2, nullptr);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_RADDSTOREEXPMINUSMAX__AVX512F_RR2_P5_U64_ACC2, elements_eq_64) {
    TEST_REQUIRES_X86_AVX512F;
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(64)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__avx512f_rr2_p5_u64_acc2, nullptr);
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__AVX512F_RR2_P5_U64_ACC2, elements_div_64) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t elements = 128; elements < 640; elements += 64) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__avx512f_rr2_p5_u64_acc2, nullptr);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__AVX512F_RR2_P5_U64_ACC2, elements_lt_64) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t elements = 1; elements < 64; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__avx512f_rr2_p5_u64_acc2, nullptr);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__AVX512F_RR2_P5_U64_ACC2, elements_gt_64) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t elements = 65; elements < 128; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__avx512f_rr2_p5_u64_acc2, nullptr);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F32_RADDSTOREEXPMINUSMAX__AVX512F_RR2_P5_U64_ACC4, elements_eq_64) {
    TEST_REQUIRES_X86_AVX512F;
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(64)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__avx512f_rr2_p5_u64_acc4, nullptr);
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__AVX512F_RR2_P5_U64_ACC4, elements_div_64) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t elements = 128; elements < 640; elements += 64) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__avx512f_rr2_p5_u64_acc4, nullptr);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__AVX512F_RR2_P5_U64_ACC4, elements_lt_64) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t elements = 1; elements < 64; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__avx512f_rr2_p5_u64_acc4, nullptr);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__AVX512F_RR2_P5_U64_ACC4, elements_gt_64) {
    TEST_REQUIRES_X86_AVX512F;
    for (size_t elements = 65; elements < 128; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__avx512f_rr2_p5_u64_acc4, nullptr);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_RADDSTOREEXPMINUSMAX__WASMSIMD_RR2_P5_U4, elements_eq_4) {
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(4)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__wasmsimd_rr2_p5_u4, nullptr);
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__WASMSIMD_RR2_P5_U4, elements_div_4) {
    for (size_t elements = 8; elements < 40; elements += 4) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__wasmsimd_rr2_p5_u4, nullptr);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__WASMSIMD_RR2_P5_U4, elements_lt_4) {
    for (size_t elements = 1; elements < 4; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__wasmsimd_rr2_p5_u4, nullptr);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__WASMSIMD_RR2_P5_U4, elements_gt_4) {
    for (size_t elements = 5; elements < 8; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__wasmsimd_rr2_p5_u4, nullptr);
    }
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_RADDSTOREEXPMINUSMAX__WASMSIMD_RR2_P5_U8_ACC2, elements_eq_8) {
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(8)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__wasmsimd_rr2_p5_u8_acc2, nullptr);
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__WASMSIMD_RR2_P5_U8_ACC2, elements_div_8) {
    for (size_t elements = 16; elements < 80; elements += 8) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__wasmsimd_rr2_p5_u8_acc2, nullptr);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__WASMSIMD_RR2_P5_U8_ACC2, elements_lt_8) {
    for (size_t elements = 1; elements < 8; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__wasmsimd_rr2_p5_u8_acc2, nullptr);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__WASMSIMD_RR2_P5_U8_ACC2, elements_gt_8) {
    for (size_t elements = 9; elements < 16; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__wasmsimd_rr2_p5_u8_acc2, nullptr);
    }
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_RADDSTOREEXPMINUSMAX__WASMSIMD_RR2_P5_U16_ACC2, elements_eq_16) {
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(16)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__wasmsimd_rr2_p5_u16_acc2, nullptr);
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__WASMSIMD_RR2_P5_U16_ACC2, elements_div_16) {
    for (size_t elements = 32; elements < 160; elements += 16) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__wasmsimd_rr2_p5_u16_acc2, nullptr);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__WASMSIMD_RR2_P5_U16_ACC2, elements_lt_16) {
    for (size_t elements = 1; elements < 16; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__wasmsimd_rr2_p5_u16_acc2, nullptr);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__WASMSIMD_RR2_P5_U16_ACC2, elements_gt_16) {
    for (size_t elements = 17; elements < 32; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__wasmsimd_rr2_p5_u16_acc2, nullptr);
    }
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_RADDSTOREEXPMINUSMAX__WASMSIMD_RR2_P5_U16_ACC4, elements_eq_16) {
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(16)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__wasmsimd_rr2_p5_u16_acc4, nullptr);
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__WASMSIMD_RR2_P5_U16_ACC4, elements_div_16) {
    for (size_t elements = 32; elements < 160; elements += 16) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__wasmsimd_rr2_p5_u16_acc4, nullptr);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__WASMSIMD_RR2_P5_U16_ACC4, elements_lt_16) {
    for (size_t elements = 1; elements < 16; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__wasmsimd_rr2_p5_u16_acc4, nullptr);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__WASMSIMD_RR2_P5_U16_ACC4, elements_gt_16) {
    for (size_t elements = 17; elements < 32; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__wasmsimd_rr2_p5_u16_acc4, nullptr);
    }
  }
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_RADDSTOREEXPMINUSMAX__WASMRELAXEDSIMD_RR2_P5_U4, elements_eq_4) {
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(4)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__wasmrelaxedsimd_rr2_p5_u4, nullptr);
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__WASMRELAXEDSIMD_RR2_P5_U4, elements_div_4) {
    for (size_t elements = 8; elements < 40; elements += 4) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__wasmrelaxedsimd_rr2_p5_u4, nullptr);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__WASMRELAXEDSIMD_RR2_P5_U4, elements_lt_4) {
    for (size_t elements = 1; elements < 4; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__wasmrelaxedsimd_rr2_p5_u4, nullptr);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__WASMRELAXEDSIMD_RR2_P5_U4, elements_gt_4) {
    for (size_t elements = 5; elements < 8; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__wasmrelaxedsimd_rr2_p5_u4, nullptr);
    }
  }
#endif  // XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_RADDSTOREEXPMINUSMAX__WASMRELAXEDSIMD_RR2_P5_U8_ACC2, elements_eq_8) {
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(8)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__wasmrelaxedsimd_rr2_p5_u8_acc2, nullptr);
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__WASMRELAXEDSIMD_RR2_P5_U8_ACC2, elements_div_8) {
    for (size_t elements = 16; elements < 80; elements += 8) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__wasmrelaxedsimd_rr2_p5_u8_acc2, nullptr);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__WASMRELAXEDSIMD_RR2_P5_U8_ACC2, elements_lt_8) {
    for (size_t elements = 1; elements < 8; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__wasmrelaxedsimd_rr2_p5_u8_acc2, nullptr);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__WASMRELAXEDSIMD_RR2_P5_U8_ACC2, elements_gt_8) {
    for (size_t elements = 9; elements < 16; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__wasmrelaxedsimd_rr2_p5_u8_acc2, nullptr);
    }
  }
#endif  // XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_RADDSTOREEXPMINUSMAX__WASMRELAXEDSIMD_RR2_P5_U16_ACC2, elements_eq_16) {
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(16)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__wasmrelaxedsimd_rr2_p5_u16_acc2, nullptr);
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__WASMRELAXEDSIMD_RR2_P5_U16_ACC2, elements_div_16) {
    for (size_t elements = 32; elements < 160; elements += 16) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__wasmrelaxedsimd_rr2_p5_u16_acc2, nullptr);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__WASMRELAXEDSIMD_RR2_P5_U16_ACC2, elements_lt_16) {
    for (size_t elements = 1; elements < 16; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__wasmrelaxedsimd_rr2_p5_u16_acc2, nullptr);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__WASMRELAXEDSIMD_RR2_P5_U16_ACC2, elements_gt_16) {
    for (size_t elements = 17; elements < 32; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__wasmrelaxedsimd_rr2_p5_u16_acc2, nullptr);
    }
  }
#endif  // XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ARCH_WASMRELAXEDSIMD
  TEST(F32_RADDSTOREEXPMINUSMAX__WASMRELAXEDSIMD_RR2_P5_U16_ACC4, elements_eq_16) {
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(16)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__wasmrelaxedsimd_rr2_p5_u16_acc4, nullptr);
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__WASMRELAXEDSIMD_RR2_P5_U16_ACC4, elements_div_16) {
    for (size_t elements = 32; elements < 160; elements += 16) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__wasmrelaxedsimd_rr2_p5_u16_acc4, nullptr);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__WASMRELAXEDSIMD_RR2_P5_U16_ACC4, elements_lt_16) {
    for (size_t elements = 1; elements < 16; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__wasmrelaxedsimd_rr2_p5_u16_acc4, nullptr);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__WASMRELAXEDSIMD_RR2_P5_U16_ACC4, elements_gt_16) {
    for (size_t elements = 17; elements < 32; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__wasmrelaxedsimd_rr2_p5_u16_acc4, nullptr);
    }
  }
#endif  // XNN_ARCH_WASMRELAXEDSIMD


#if XNN_ENABLE_HVX && XNN_ARCH_HEXAGON
  TEST(F32_RADDSTOREEXPMINUSMAX__HVX_RR2_P5_U32, elements_eq_32) {
    TEST_REQUIRES_HVX;
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(32)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__hvx_rr2_p5_u32, nullptr);
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__HVX_RR2_P5_U32, elements_div_32) {
    TEST_REQUIRES_HVX;
    for (size_t elements = 64; elements < 320; elements += 32) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__hvx_rr2_p5_u32, nullptr);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__HVX_RR2_P5_U32, elements_lt_32) {
    TEST_REQUIRES_HVX;
    for (size_t elements = 1; elements < 32; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__hvx_rr2_p5_u32, nullptr);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__HVX_RR2_P5_U32, elements_gt_32) {
    TEST_REQUIRES_HVX;
    for (size_t elements = 33; elements < 64; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__hvx_rr2_p5_u32, nullptr);
    }
  }
#endif  // XNN_ENABLE_HVX && XNN_ARCH_HEXAGON


#if XNN_ENABLE_HVX && XNN_ARCH_HEXAGON
  TEST(F32_RADDSTOREEXPMINUSMAX__HVX_RR2_P5_U64_ACC2, elements_eq_64) {
    TEST_REQUIRES_HVX;
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(64)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__hvx_rr2_p5_u64_acc2, nullptr);
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__HVX_RR2_P5_U64_ACC2, elements_div_64) {
    TEST_REQUIRES_HVX;
    for (size_t elements = 128; elements < 640; elements += 64) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__hvx_rr2_p5_u64_acc2, nullptr);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__HVX_RR2_P5_U64_ACC2, elements_lt_64) {
    TEST_REQUIRES_HVX;
    for (size_t elements = 1; elements < 64; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__hvx_rr2_p5_u64_acc2, nullptr);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__HVX_RR2_P5_U64_ACC2, elements_gt_64) {
    TEST_REQUIRES_HVX;
    for (size_t elements = 65; elements < 128; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__hvx_rr2_p5_u64_acc2, nullptr);
    }
  }
#endif  // XNN_ENABLE_HVX && XNN_ARCH_HEXAGON


#if XNN_ENABLE_HVX && XNN_ARCH_HEXAGON
  TEST(F32_RADDSTOREEXPMINUSMAX__HVX_RR2_P5_U128_ACC2, elements_eq_128) {
    TEST_REQUIRES_HVX;
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(128)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__hvx_rr2_p5_u128_acc2, nullptr);
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__HVX_RR2_P5_U128_ACC2, elements_div_128) {
    TEST_REQUIRES_HVX;
    for (size_t elements = 256; elements < 1280; elements += 128) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__hvx_rr2_p5_u128_acc2, nullptr);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__HVX_RR2_P5_U128_ACC2, elements_lt_128) {
    TEST_REQUIRES_HVX;
    for (size_t elements = 1; elements < 128; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__hvx_rr2_p5_u128_acc2, nullptr);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__HVX_RR2_P5_U128_ACC2, elements_gt_128) {
    TEST_REQUIRES_HVX;
    for (size_t elements = 129; elements < 256; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__hvx_rr2_p5_u128_acc2, nullptr);
    }
  }
#endif  // XNN_ENABLE_HVX && XNN_ARCH_HEXAGON


#if XNN_ENABLE_HVX && XNN_ARCH_HEXAGON
  TEST(F32_RADDSTOREEXPMINUSMAX__HVX_RR2_P5_U128_ACC4, elements_eq_128) {
    TEST_REQUIRES_HVX;
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(128)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__hvx_rr2_p5_u128_acc4, nullptr);
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__HVX_RR2_P5_U128_ACC4, elements_div_128) {
    TEST_REQUIRES_HVX;
    for (size_t elements = 256; elements < 1280; elements += 128) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__hvx_rr2_p5_u128_acc4, nullptr);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__HVX_RR2_P5_U128_ACC4, elements_lt_128) {
    TEST_REQUIRES_HVX;
    for (size_t elements = 1; elements < 128; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__hvx_rr2_p5_u128_acc4, nullptr);
    }
  }

  TEST(F32_RADDSTOREEXPMINUSMAX__HVX_RR2_P5_U128_ACC4, elements_gt_128) {
    TEST_REQUIRES_HVX;
    for (size_t elements = 129; elements < 256; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f32_raddstoreexpminusmax_ukernel__hvx_rr2_p5_u128_acc4, nullptr);
    }
  }
#endif  // XNN_ENABLE_HVX && XNN_ARCH_HEXAGON


TEST(F32_RADDSTOREEXPMINUSMAX__SCALAR_RR2_LUT64_P2_U1, elements_eq_1) {
  RAddStoreExpMinusMaxMicrokernelTester()
    .elements(1)
    .Test(xnn_f32_raddstoreexpminusmax_ukernel__scalar_rr2_lut64_p2_u1, nullptr);
}

TEST(F32_RADDSTOREEXPMINUSMAX__SCALAR_RR2_LUT64_P2_U1, elements_gt_1) {
  for (size_t elements = 2; elements < 10; elements++) {
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(elements)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__scalar_rr2_lut64_p2_u1, nullptr);
  }
}

TEST(F32_RADDSTOREEXPMINUSMAX__SCALAR_RR2_LUT64_P2_U2_ACC2, elements_eq_2) {
  RAddStoreExpMinusMaxMicrokernelTester()
    .elements(2)
    .Test(xnn_f32_raddstoreexpminusmax_ukernel__scalar_rr2_lut64_p2_u2_acc2, nullptr);
}

TEST(F32_RADDSTOREEXPMINUSMAX__SCALAR_RR2_LUT64_P2_U2_ACC2, elements_div_2) {
  for (size_t elements = 4; elements < 20; elements += 2) {
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(elements)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__scalar_rr2_lut64_p2_u2_acc2, nullptr);
  }
}

TEST(F32_RADDSTOREEXPMINUSMAX__SCALAR_RR2_LUT64_P2_U2_ACC2, elements_lt_2) {
  for (size_t elements = 1; elements < 2; elements++) {
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(elements)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__scalar_rr2_lut64_p2_u2_acc2, nullptr);
  }
}

TEST(F32_RADDSTOREEXPMINUSMAX__SCALAR_RR2_LUT64_P2_U2_ACC2, elements_gt_2) {
  for (size_t elements = 3; elements < 4; elements++) {
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(elements)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__scalar_rr2_lut64_p2_u2_acc2, nullptr);
  }
}

TEST(F32_RADDSTOREEXPMINUSMAX__SCALAR_RR2_LUT64_P2_U4_ACC2, elements_eq_4) {
  RAddStoreExpMinusMaxMicrokernelTester()
    .elements(4)
    .Test(xnn_f32_raddstoreexpminusmax_ukernel__scalar_rr2_lut64_p2_u4_acc2, nullptr);
}

TEST(F32_RADDSTOREEXPMINUSMAX__SCALAR_RR2_LUT64_P2_U4_ACC2, elements_div_4) {
  for (size_t elements = 8; elements < 40; elements += 4) {
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(elements)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__scalar_rr2_lut64_p2_u4_acc2, nullptr);
  }
}

TEST(F32_RADDSTOREEXPMINUSMAX__SCALAR_RR2_LUT64_P2_U4_ACC2, elements_lt_4) {
  for (size_t elements = 1; elements < 4; elements++) {
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(elements)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__scalar_rr2_lut64_p2_u4_acc2, nullptr);
  }
}

TEST(F32_RADDSTOREEXPMINUSMAX__SCALAR_RR2_LUT64_P2_U4_ACC2, elements_gt_4) {
  for (size_t elements = 5; elements < 8; elements++) {
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(elements)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__scalar_rr2_lut64_p2_u4_acc2, nullptr);
  }
}

TEST(F32_RADDSTOREEXPMINUSMAX__SCALAR_RR2_LUT64_P2_U4_ACC4, elements_eq_4) {
  RAddStoreExpMinusMaxMicrokernelTester()
    .elements(4)
    .Test(xnn_f32_raddstoreexpminusmax_ukernel__scalar_rr2_lut64_p2_u4_acc4, nullptr);
}

TEST(F32_RADDSTOREEXPMINUSMAX__SCALAR_RR2_LUT64_P2_U4_ACC4, elements_div_4) {
  for (size_t elements = 8; elements < 40; elements += 4) {
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(elements)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__scalar_rr2_lut64_p2_u4_acc4, nullptr);
  }
}

TEST(F32_RADDSTOREEXPMINUSMAX__SCALAR_RR2_LUT64_P2_U4_ACC4, elements_lt_4) {
  for (size_t elements = 1; elements < 4; elements++) {
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(elements)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__scalar_rr2_lut64_p2_u4_acc4, nullptr);
  }
}

TEST(F32_RADDSTOREEXPMINUSMAX__SCALAR_RR2_LUT64_P2_U4_ACC4, elements_gt_4) {
  for (size_t elements = 5; elements < 8; elements++) {
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(elements)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__scalar_rr2_lut64_p2_u4_acc4, nullptr);
  }
}

TEST(F32_RADDSTOREEXPMINUSMAX__SCALAR_RR2_P5_U1, elements_eq_1) {
  RAddStoreExpMinusMaxMicrokernelTester()
    .elements(1)
    .Test(xnn_f32_raddstoreexpminusmax_ukernel__scalar_rr2_p5_u1, nullptr);
}

TEST(F32_RADDSTOREEXPMINUSMAX__SCALAR_RR2_P5_U1, elements_gt_1) {
  for (size_t elements = 2; elements < 10; elements++) {
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(elements)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__scalar_rr2_p5_u1, nullptr);
  }
}

TEST(F32_RADDSTOREEXPMINUSMAX__SCALAR_RR2_P5_U2_ACC2, elements_eq_2) {
  RAddStoreExpMinusMaxMicrokernelTester()
    .elements(2)
    .Test(xnn_f32_raddstoreexpminusmax_ukernel__scalar_rr2_p5_u2_acc2, nullptr);
}

TEST(F32_RADDSTOREEXPMINUSMAX__SCALAR_RR2_P5_U2_ACC2, elements_div_2) {
  for (size_t elements = 4; elements < 20; elements += 2) {
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(elements)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__scalar_rr2_p5_u2_acc2, nullptr);
  }
}

TEST(F32_RADDSTOREEXPMINUSMAX__SCALAR_RR2_P5_U2_ACC2, elements_lt_2) {
  for (size_t elements = 1; elements < 2; elements++) {
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(elements)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__scalar_rr2_p5_u2_acc2, nullptr);
  }
}

TEST(F32_RADDSTOREEXPMINUSMAX__SCALAR_RR2_P5_U2_ACC2, elements_gt_2) {
  for (size_t elements = 3; elements < 4; elements++) {
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(elements)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__scalar_rr2_p5_u2_acc2, nullptr);
  }
}

TEST(F32_RADDSTOREEXPMINUSMAX__SCALAR_RR2_P5_U4_ACC2, elements_eq_4) {
  RAddStoreExpMinusMaxMicrokernelTester()
    .elements(4)
    .Test(xnn_f32_raddstoreexpminusmax_ukernel__scalar_rr2_p5_u4_acc2, nullptr);
}

TEST(F32_RADDSTOREEXPMINUSMAX__SCALAR_RR2_P5_U4_ACC2, elements_div_4) {
  for (size_t elements = 8; elements < 40; elements += 4) {
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(elements)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__scalar_rr2_p5_u4_acc2, nullptr);
  }
}

TEST(F32_RADDSTOREEXPMINUSMAX__SCALAR_RR2_P5_U4_ACC2, elements_lt_4) {
  for (size_t elements = 1; elements < 4; elements++) {
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(elements)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__scalar_rr2_p5_u4_acc2, nullptr);
  }
}

TEST(F32_RADDSTOREEXPMINUSMAX__SCALAR_RR2_P5_U4_ACC2, elements_gt_4) {
  for (size_t elements = 5; elements < 8; elements++) {
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(elements)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__scalar_rr2_p5_u4_acc2, nullptr);
  }
}

TEST(F32_RADDSTOREEXPMINUSMAX__SCALAR_RR2_P5_U4_ACC4, elements_eq_4) {
  RAddStoreExpMinusMaxMicrokernelTester()
    .elements(4)
    .Test(xnn_f32_raddstoreexpminusmax_ukernel__scalar_rr2_p5_u4_acc4, nullptr);
}

TEST(F32_RADDSTOREEXPMINUSMAX__SCALAR_RR2_P5_U4_ACC4, elements_div_4) {
  for (size_t elements = 8; elements < 40; elements += 4) {
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(elements)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__scalar_rr2_p5_u4_acc4, nullptr);
  }
}

TEST(F32_RADDSTOREEXPMINUSMAX__SCALAR_RR2_P5_U4_ACC4, elements_lt_4) {
  for (size_t elements = 1; elements < 4; elements++) {
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(elements)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__scalar_rr2_p5_u4_acc4, nullptr);
  }
}

TEST(F32_RADDSTOREEXPMINUSMAX__SCALAR_RR2_P5_U4_ACC4, elements_gt_4) {
  for (size_t elements = 5; elements < 8; elements++) {
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(elements)
      .Test(xnn_f32_raddstoreexpminusmax_ukernel__scalar_rr2_p5_u4_acc4, nullptr);
  }
}