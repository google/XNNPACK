// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: test/f16-raddstoreexpminusmax.yaml
//   Generator: tools/generate-raddstoreexpminusmax-test.py


#include <gtest/gtest.h>
#include "xnnpack/common.h"
#include "xnnpack/isa-checks.h"
#include "xnnpack/microparams-init.h"
#include "xnnpack/raddstoreexpminusmax.h"
#include "raddstoreexpminusmax-microkernel-tester.h"


#if XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  TEST(F16_RADDSTOREEXPMINUSMAX__NEONFP16ARITH_RR2_P2_U32, elements_eq_32) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(32)
      .Test(xnn_f16_raddstoreexpminusmax_ukernel__neonfp16arith_rr2_p2_u32, nullptr);
  }

  TEST(F16_RADDSTOREEXPMINUSMAX__NEONFP16ARITH_RR2_P2_U32, elements_div_32) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t elements = 64; elements < 320; elements += 32) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f16_raddstoreexpminusmax_ukernel__neonfp16arith_rr2_p2_u32, nullptr);
    }
  }

  TEST(F16_RADDSTOREEXPMINUSMAX__NEONFP16ARITH_RR2_P2_U32, elements_lt_32) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t elements = 1; elements < 32; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f16_raddstoreexpminusmax_ukernel__neonfp16arith_rr2_p2_u32, nullptr);
    }
  }

  TEST(F16_RADDSTOREEXPMINUSMAX__NEONFP16ARITH_RR2_P2_U32, elements_gt_32) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t elements = 33; elements < 64; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f16_raddstoreexpminusmax_ukernel__neonfp16arith_rr2_p2_u32, nullptr);
    }
  }
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  TEST(F16_RADDSTOREEXPMINUSMAX__NEONFP16ARITH_RR2_P2_U32_ACC2, elements_eq_32) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(32)
      .Test(xnn_f16_raddstoreexpminusmax_ukernel__neonfp16arith_rr2_p2_u32_acc2, nullptr);
  }

  TEST(F16_RADDSTOREEXPMINUSMAX__NEONFP16ARITH_RR2_P2_U32_ACC2, elements_div_32) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t elements = 64; elements < 320; elements += 32) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f16_raddstoreexpminusmax_ukernel__neonfp16arith_rr2_p2_u32_acc2, nullptr);
    }
  }

  TEST(F16_RADDSTOREEXPMINUSMAX__NEONFP16ARITH_RR2_P2_U32_ACC2, elements_lt_32) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t elements = 1; elements < 32; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f16_raddstoreexpminusmax_ukernel__neonfp16arith_rr2_p2_u32_acc2, nullptr);
    }
  }

  TEST(F16_RADDSTOREEXPMINUSMAX__NEONFP16ARITH_RR2_P2_U32_ACC2, elements_gt_32) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t elements = 33; elements < 64; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f16_raddstoreexpminusmax_ukernel__neonfp16arith_rr2_p2_u32_acc2, nullptr);
    }
  }
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  TEST(F16_RADDSTOREEXPMINUSMAX__NEONFP16ARITH_RR2_P2_U32_ACC4, elements_eq_32) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(32)
      .Test(xnn_f16_raddstoreexpminusmax_ukernel__neonfp16arith_rr2_p2_u32_acc4, nullptr);
  }

  TEST(F16_RADDSTOREEXPMINUSMAX__NEONFP16ARITH_RR2_P2_U32_ACC4, elements_div_32) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t elements = 64; elements < 320; elements += 32) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f16_raddstoreexpminusmax_ukernel__neonfp16arith_rr2_p2_u32_acc4, nullptr);
    }
  }

  TEST(F16_RADDSTOREEXPMINUSMAX__NEONFP16ARITH_RR2_P2_U32_ACC4, elements_lt_32) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t elements = 1; elements < 32; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f16_raddstoreexpminusmax_ukernel__neonfp16arith_rr2_p2_u32_acc4, nullptr);
    }
  }

  TEST(F16_RADDSTOREEXPMINUSMAX__NEONFP16ARITH_RR2_P2_U32_ACC4, elements_gt_32) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t elements = 33; elements < 64; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f16_raddstoreexpminusmax_ukernel__neonfp16arith_rr2_p2_u32_acc4, nullptr);
    }
  }
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  TEST(F16_RADDSTOREEXPMINUSMAX__NEONFP16ARITH_RR2_P2_U40, elements_eq_40) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(40)
      .Test(xnn_f16_raddstoreexpminusmax_ukernel__neonfp16arith_rr2_p2_u40, nullptr);
  }

  TEST(F16_RADDSTOREEXPMINUSMAX__NEONFP16ARITH_RR2_P2_U40, elements_div_40) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t elements = 80; elements < 400; elements += 40) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f16_raddstoreexpminusmax_ukernel__neonfp16arith_rr2_p2_u40, nullptr);
    }
  }

  TEST(F16_RADDSTOREEXPMINUSMAX__NEONFP16ARITH_RR2_P2_U40, elements_lt_40) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t elements = 1; elements < 40; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f16_raddstoreexpminusmax_ukernel__neonfp16arith_rr2_p2_u40, nullptr);
    }
  }

  TEST(F16_RADDSTOREEXPMINUSMAX__NEONFP16ARITH_RR2_P2_U40, elements_gt_40) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t elements = 41; elements < 80; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f16_raddstoreexpminusmax_ukernel__neonfp16arith_rr2_p2_u40, nullptr);
    }
  }
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  TEST(F16_RADDSTOREEXPMINUSMAX__NEONFP16ARITH_RR2_P2_U40_ACC2, elements_eq_40) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(40)
      .Test(xnn_f16_raddstoreexpminusmax_ukernel__neonfp16arith_rr2_p2_u40_acc2, nullptr);
  }

  TEST(F16_RADDSTOREEXPMINUSMAX__NEONFP16ARITH_RR2_P2_U40_ACC2, elements_div_40) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t elements = 80; elements < 400; elements += 40) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f16_raddstoreexpminusmax_ukernel__neonfp16arith_rr2_p2_u40_acc2, nullptr);
    }
  }

  TEST(F16_RADDSTOREEXPMINUSMAX__NEONFP16ARITH_RR2_P2_U40_ACC2, elements_lt_40) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t elements = 1; elements < 40; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f16_raddstoreexpminusmax_ukernel__neonfp16arith_rr2_p2_u40_acc2, nullptr);
    }
  }

  TEST(F16_RADDSTOREEXPMINUSMAX__NEONFP16ARITH_RR2_P2_U40_ACC2, elements_gt_40) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t elements = 41; elements < 80; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f16_raddstoreexpminusmax_ukernel__neonfp16arith_rr2_p2_u40_acc2, nullptr);
    }
  }
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  TEST(F16_RADDSTOREEXPMINUSMAX__NEONFP16ARITH_RR2_P2_U40_ACC5, elements_eq_40) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(40)
      .Test(xnn_f16_raddstoreexpminusmax_ukernel__neonfp16arith_rr2_p2_u40_acc5, nullptr);
  }

  TEST(F16_RADDSTOREEXPMINUSMAX__NEONFP16ARITH_RR2_P2_U40_ACC5, elements_div_40) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t elements = 80; elements < 400; elements += 40) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f16_raddstoreexpminusmax_ukernel__neonfp16arith_rr2_p2_u40_acc5, nullptr);
    }
  }

  TEST(F16_RADDSTOREEXPMINUSMAX__NEONFP16ARITH_RR2_P2_U40_ACC5, elements_lt_40) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t elements = 1; elements < 40; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f16_raddstoreexpminusmax_ukernel__neonfp16arith_rr2_p2_u40_acc5, nullptr);
    }
  }

  TEST(F16_RADDSTOREEXPMINUSMAX__NEONFP16ARITH_RR2_P2_U40_ACC5, elements_gt_40) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t elements = 41; elements < 80; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f16_raddstoreexpminusmax_ukernel__neonfp16arith_rr2_p2_u40_acc5, nullptr);
    }
  }
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  TEST(F16_RADDSTOREEXPMINUSMAX__NEONFP16ARITH_RR2_P2_U48, elements_eq_48) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(48)
      .Test(xnn_f16_raddstoreexpminusmax_ukernel__neonfp16arith_rr2_p2_u48, nullptr);
  }

  TEST(F16_RADDSTOREEXPMINUSMAX__NEONFP16ARITH_RR2_P2_U48, elements_div_48) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t elements = 96; elements < 480; elements += 48) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f16_raddstoreexpminusmax_ukernel__neonfp16arith_rr2_p2_u48, nullptr);
    }
  }

  TEST(F16_RADDSTOREEXPMINUSMAX__NEONFP16ARITH_RR2_P2_U48, elements_lt_48) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t elements = 1; elements < 48; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f16_raddstoreexpminusmax_ukernel__neonfp16arith_rr2_p2_u48, nullptr);
    }
  }

  TEST(F16_RADDSTOREEXPMINUSMAX__NEONFP16ARITH_RR2_P2_U48, elements_gt_48) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t elements = 49; elements < 96; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f16_raddstoreexpminusmax_ukernel__neonfp16arith_rr2_p2_u48, nullptr);
    }
  }
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  TEST(F16_RADDSTOREEXPMINUSMAX__NEONFP16ARITH_RR2_P2_U48_ACC2, elements_eq_48) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(48)
      .Test(xnn_f16_raddstoreexpminusmax_ukernel__neonfp16arith_rr2_p2_u48_acc2, nullptr);
  }

  TEST(F16_RADDSTOREEXPMINUSMAX__NEONFP16ARITH_RR2_P2_U48_ACC2, elements_div_48) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t elements = 96; elements < 480; elements += 48) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f16_raddstoreexpminusmax_ukernel__neonfp16arith_rr2_p2_u48_acc2, nullptr);
    }
  }

  TEST(F16_RADDSTOREEXPMINUSMAX__NEONFP16ARITH_RR2_P2_U48_ACC2, elements_lt_48) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t elements = 1; elements < 48; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f16_raddstoreexpminusmax_ukernel__neonfp16arith_rr2_p2_u48_acc2, nullptr);
    }
  }

  TEST(F16_RADDSTOREEXPMINUSMAX__NEONFP16ARITH_RR2_P2_U48_ACC2, elements_gt_48) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t elements = 49; elements < 96; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f16_raddstoreexpminusmax_ukernel__neonfp16arith_rr2_p2_u48_acc2, nullptr);
    }
  }
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  TEST(F16_RADDSTOREEXPMINUSMAX__NEONFP16ARITH_RR2_P2_U48_ACC3, elements_eq_48) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(48)
      .Test(xnn_f16_raddstoreexpminusmax_ukernel__neonfp16arith_rr2_p2_u48_acc3, nullptr);
  }

  TEST(F16_RADDSTOREEXPMINUSMAX__NEONFP16ARITH_RR2_P2_U48_ACC3, elements_div_48) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t elements = 96; elements < 480; elements += 48) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f16_raddstoreexpminusmax_ukernel__neonfp16arith_rr2_p2_u48_acc3, nullptr);
    }
  }

  TEST(F16_RADDSTOREEXPMINUSMAX__NEONFP16ARITH_RR2_P2_U48_ACC3, elements_lt_48) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t elements = 1; elements < 48; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f16_raddstoreexpminusmax_ukernel__neonfp16arith_rr2_p2_u48_acc3, nullptr);
    }
  }

  TEST(F16_RADDSTOREEXPMINUSMAX__NEONFP16ARITH_RR2_P2_U48_ACC3, elements_gt_48) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t elements = 49; elements < 96; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f16_raddstoreexpminusmax_ukernel__neonfp16arith_rr2_p2_u48_acc3, nullptr);
    }
  }
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  TEST(F16_RADDSTOREEXPMINUSMAX__NEONFP16ARITH_RR2_P2_U64, elements_eq_64) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(64)
      .Test(xnn_f16_raddstoreexpminusmax_ukernel__neonfp16arith_rr2_p2_u64, nullptr);
  }

  TEST(F16_RADDSTOREEXPMINUSMAX__NEONFP16ARITH_RR2_P2_U64, elements_div_64) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t elements = 128; elements < 640; elements += 64) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f16_raddstoreexpminusmax_ukernel__neonfp16arith_rr2_p2_u64, nullptr);
    }
  }

  TEST(F16_RADDSTOREEXPMINUSMAX__NEONFP16ARITH_RR2_P2_U64, elements_lt_64) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t elements = 1; elements < 64; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f16_raddstoreexpminusmax_ukernel__neonfp16arith_rr2_p2_u64, nullptr);
    }
  }

  TEST(F16_RADDSTOREEXPMINUSMAX__NEONFP16ARITH_RR2_P2_U64, elements_gt_64) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t elements = 65; elements < 128; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f16_raddstoreexpminusmax_ukernel__neonfp16arith_rr2_p2_u64, nullptr);
    }
  }
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  TEST(F16_RADDSTOREEXPMINUSMAX__NEONFP16ARITH_RR2_P2_U64_ACC2, elements_eq_64) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(64)
      .Test(xnn_f16_raddstoreexpminusmax_ukernel__neonfp16arith_rr2_p2_u64_acc2, nullptr);
  }

  TEST(F16_RADDSTOREEXPMINUSMAX__NEONFP16ARITH_RR2_P2_U64_ACC2, elements_div_64) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t elements = 128; elements < 640; elements += 64) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f16_raddstoreexpminusmax_ukernel__neonfp16arith_rr2_p2_u64_acc2, nullptr);
    }
  }

  TEST(F16_RADDSTOREEXPMINUSMAX__NEONFP16ARITH_RR2_P2_U64_ACC2, elements_lt_64) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t elements = 1; elements < 64; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f16_raddstoreexpminusmax_ukernel__neonfp16arith_rr2_p2_u64_acc2, nullptr);
    }
  }

  TEST(F16_RADDSTOREEXPMINUSMAX__NEONFP16ARITH_RR2_P2_U64_ACC2, elements_gt_64) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t elements = 65; elements < 128; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f16_raddstoreexpminusmax_ukernel__neonfp16arith_rr2_p2_u64_acc2, nullptr);
    }
  }
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  TEST(F16_RADDSTOREEXPMINUSMAX__NEONFP16ARITH_RR2_P2_U64_ACC4, elements_eq_64) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(64)
      .Test(xnn_f16_raddstoreexpminusmax_ukernel__neonfp16arith_rr2_p2_u64_acc4, nullptr);
  }

  TEST(F16_RADDSTOREEXPMINUSMAX__NEONFP16ARITH_RR2_P2_U64_ACC4, elements_div_64) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t elements = 128; elements < 640; elements += 64) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f16_raddstoreexpminusmax_ukernel__neonfp16arith_rr2_p2_u64_acc4, nullptr);
    }
  }

  TEST(F16_RADDSTOREEXPMINUSMAX__NEONFP16ARITH_RR2_P2_U64_ACC4, elements_lt_64) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t elements = 1; elements < 64; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f16_raddstoreexpminusmax_ukernel__neonfp16arith_rr2_p2_u64_acc4, nullptr);
    }
  }

  TEST(F16_RADDSTOREEXPMINUSMAX__NEONFP16ARITH_RR2_P2_U64_ACC4, elements_gt_64) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t elements = 65; elements < 128; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f16_raddstoreexpminusmax_ukernel__neonfp16arith_rr2_p2_u64_acc4, nullptr);
    }
  }
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  TEST(F16_RADDSTOREEXPMINUSMAX__NEONFP16ARITH_RR2_P2_U72, elements_eq_72) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(72)
      .Test(xnn_f16_raddstoreexpminusmax_ukernel__neonfp16arith_rr2_p2_u72, nullptr);
  }

  TEST(F16_RADDSTOREEXPMINUSMAX__NEONFP16ARITH_RR2_P2_U72, elements_div_72) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t elements = 144; elements < 720; elements += 72) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f16_raddstoreexpminusmax_ukernel__neonfp16arith_rr2_p2_u72, nullptr);
    }
  }

  TEST(F16_RADDSTOREEXPMINUSMAX__NEONFP16ARITH_RR2_P2_U72, elements_lt_72) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t elements = 1; elements < 72; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f16_raddstoreexpminusmax_ukernel__neonfp16arith_rr2_p2_u72, nullptr);
    }
  }

  TEST(F16_RADDSTOREEXPMINUSMAX__NEONFP16ARITH_RR2_P2_U72, elements_gt_72) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t elements = 73; elements < 144; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f16_raddstoreexpminusmax_ukernel__neonfp16arith_rr2_p2_u72, nullptr);
    }
  }
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  TEST(F16_RADDSTOREEXPMINUSMAX__NEONFP16ARITH_RR2_P2_U72_ACC3, elements_eq_72) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(72)
      .Test(xnn_f16_raddstoreexpminusmax_ukernel__neonfp16arith_rr2_p2_u72_acc3, nullptr);
  }

  TEST(F16_RADDSTOREEXPMINUSMAX__NEONFP16ARITH_RR2_P2_U72_ACC3, elements_div_72) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t elements = 144; elements < 720; elements += 72) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f16_raddstoreexpminusmax_ukernel__neonfp16arith_rr2_p2_u72_acc3, nullptr);
    }
  }

  TEST(F16_RADDSTOREEXPMINUSMAX__NEONFP16ARITH_RR2_P2_U72_ACC3, elements_lt_72) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t elements = 1; elements < 72; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f16_raddstoreexpminusmax_ukernel__neonfp16arith_rr2_p2_u72_acc3, nullptr);
    }
  }

  TEST(F16_RADDSTOREEXPMINUSMAX__NEONFP16ARITH_RR2_P2_U72_ACC3, elements_gt_72) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t elements = 73; elements < 144; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f16_raddstoreexpminusmax_ukernel__neonfp16arith_rr2_p2_u72_acc3, nullptr);
    }
  }
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  TEST(F16_RADDSTOREEXPMINUSMAX__NEONFP16ARITH_RR2_P2_U80, elements_eq_80) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(80)
      .Test(xnn_f16_raddstoreexpminusmax_ukernel__neonfp16arith_rr2_p2_u80, nullptr);
  }

  TEST(F16_RADDSTOREEXPMINUSMAX__NEONFP16ARITH_RR2_P2_U80, elements_div_80) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t elements = 160; elements < 800; elements += 80) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f16_raddstoreexpminusmax_ukernel__neonfp16arith_rr2_p2_u80, nullptr);
    }
  }

  TEST(F16_RADDSTOREEXPMINUSMAX__NEONFP16ARITH_RR2_P2_U80, elements_lt_80) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t elements = 1; elements < 80; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f16_raddstoreexpminusmax_ukernel__neonfp16arith_rr2_p2_u80, nullptr);
    }
  }

  TEST(F16_RADDSTOREEXPMINUSMAX__NEONFP16ARITH_RR2_P2_U80, elements_gt_80) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t elements = 81; elements < 160; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f16_raddstoreexpminusmax_ukernel__neonfp16arith_rr2_p2_u80, nullptr);
    }
  }
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  TEST(F16_RADDSTOREEXPMINUSMAX__NEONFP16ARITH_RR2_P2_U80_ACC2, elements_eq_80) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(80)
      .Test(xnn_f16_raddstoreexpminusmax_ukernel__neonfp16arith_rr2_p2_u80_acc2, nullptr);
  }

  TEST(F16_RADDSTOREEXPMINUSMAX__NEONFP16ARITH_RR2_P2_U80_ACC2, elements_div_80) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t elements = 160; elements < 800; elements += 80) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f16_raddstoreexpminusmax_ukernel__neonfp16arith_rr2_p2_u80_acc2, nullptr);
    }
  }

  TEST(F16_RADDSTOREEXPMINUSMAX__NEONFP16ARITH_RR2_P2_U80_ACC2, elements_lt_80) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t elements = 1; elements < 80; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f16_raddstoreexpminusmax_ukernel__neonfp16arith_rr2_p2_u80_acc2, nullptr);
    }
  }

  TEST(F16_RADDSTOREEXPMINUSMAX__NEONFP16ARITH_RR2_P2_U80_ACC2, elements_gt_80) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t elements = 81; elements < 160; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f16_raddstoreexpminusmax_ukernel__neonfp16arith_rr2_p2_u80_acc2, nullptr);
    }
  }
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  TEST(F16_RADDSTOREEXPMINUSMAX__NEONFP16ARITH_RR2_P2_U80_ACC5, elements_eq_80) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(80)
      .Test(xnn_f16_raddstoreexpminusmax_ukernel__neonfp16arith_rr2_p2_u80_acc5, nullptr);
  }

  TEST(F16_RADDSTOREEXPMINUSMAX__NEONFP16ARITH_RR2_P2_U80_ACC5, elements_div_80) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t elements = 160; elements < 800; elements += 80) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f16_raddstoreexpminusmax_ukernel__neonfp16arith_rr2_p2_u80_acc5, nullptr);
    }
  }

  TEST(F16_RADDSTOREEXPMINUSMAX__NEONFP16ARITH_RR2_P2_U80_ACC5, elements_lt_80) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t elements = 1; elements < 80; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f16_raddstoreexpminusmax_ukernel__neonfp16arith_rr2_p2_u80_acc5, nullptr);
    }
  }

  TEST(F16_RADDSTOREEXPMINUSMAX__NEONFP16ARITH_RR2_P2_U80_ACC5, elements_gt_80) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t elements = 81; elements < 160; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f16_raddstoreexpminusmax_ukernel__neonfp16arith_rr2_p2_u80_acc5, nullptr);
    }
  }
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  TEST(F16_RADDSTOREEXPMINUSMAX__NEONFP16ARITH_RR2_P2_U96, elements_eq_96) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(96)
      .Test(xnn_f16_raddstoreexpminusmax_ukernel__neonfp16arith_rr2_p2_u96, nullptr);
  }

  TEST(F16_RADDSTOREEXPMINUSMAX__NEONFP16ARITH_RR2_P2_U96, elements_div_96) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t elements = 192; elements < 960; elements += 96) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f16_raddstoreexpminusmax_ukernel__neonfp16arith_rr2_p2_u96, nullptr);
    }
  }

  TEST(F16_RADDSTOREEXPMINUSMAX__NEONFP16ARITH_RR2_P2_U96, elements_lt_96) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t elements = 1; elements < 96; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f16_raddstoreexpminusmax_ukernel__neonfp16arith_rr2_p2_u96, nullptr);
    }
  }

  TEST(F16_RADDSTOREEXPMINUSMAX__NEONFP16ARITH_RR2_P2_U96, elements_gt_96) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t elements = 97; elements < 192; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f16_raddstoreexpminusmax_ukernel__neonfp16arith_rr2_p2_u96, nullptr);
    }
  }
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  TEST(F16_RADDSTOREEXPMINUSMAX__NEONFP16ARITH_RR2_P2_U96_ACC2, elements_eq_96) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(96)
      .Test(xnn_f16_raddstoreexpminusmax_ukernel__neonfp16arith_rr2_p2_u96_acc2, nullptr);
  }

  TEST(F16_RADDSTOREEXPMINUSMAX__NEONFP16ARITH_RR2_P2_U96_ACC2, elements_div_96) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t elements = 192; elements < 960; elements += 96) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f16_raddstoreexpminusmax_ukernel__neonfp16arith_rr2_p2_u96_acc2, nullptr);
    }
  }

  TEST(F16_RADDSTOREEXPMINUSMAX__NEONFP16ARITH_RR2_P2_U96_ACC2, elements_lt_96) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t elements = 1; elements < 96; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f16_raddstoreexpminusmax_ukernel__neonfp16arith_rr2_p2_u96_acc2, nullptr);
    }
  }

  TEST(F16_RADDSTOREEXPMINUSMAX__NEONFP16ARITH_RR2_P2_U96_ACC2, elements_gt_96) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t elements = 97; elements < 192; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f16_raddstoreexpminusmax_ukernel__neonfp16arith_rr2_p2_u96_acc2, nullptr);
    }
  }
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  TEST(F16_RADDSTOREEXPMINUSMAX__NEONFP16ARITH_RR2_P2_U96_ACC3, elements_eq_96) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(96)
      .Test(xnn_f16_raddstoreexpminusmax_ukernel__neonfp16arith_rr2_p2_u96_acc3, nullptr);
  }

  TEST(F16_RADDSTOREEXPMINUSMAX__NEONFP16ARITH_RR2_P2_U96_ACC3, elements_div_96) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t elements = 192; elements < 960; elements += 96) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f16_raddstoreexpminusmax_ukernel__neonfp16arith_rr2_p2_u96_acc3, nullptr);
    }
  }

  TEST(F16_RADDSTOREEXPMINUSMAX__NEONFP16ARITH_RR2_P2_U96_ACC3, elements_lt_96) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t elements = 1; elements < 96; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f16_raddstoreexpminusmax_ukernel__neonfp16arith_rr2_p2_u96_acc3, nullptr);
    }
  }

  TEST(F16_RADDSTOREEXPMINUSMAX__NEONFP16ARITH_RR2_P2_U96_ACC3, elements_gt_96) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t elements = 97; elements < 192; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f16_raddstoreexpminusmax_ukernel__neonfp16arith_rr2_p2_u96_acc3, nullptr);
    }
  }
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  TEST(F16_RADDSTOREEXPMINUSMAX__NEONFP16ARITH_RR2_P2_U96_ACC6, elements_eq_96) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(96)
      .Test(xnn_f16_raddstoreexpminusmax_ukernel__neonfp16arith_rr2_p2_u96_acc6, nullptr);
  }

  TEST(F16_RADDSTOREEXPMINUSMAX__NEONFP16ARITH_RR2_P2_U96_ACC6, elements_div_96) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t elements = 192; elements < 960; elements += 96) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f16_raddstoreexpminusmax_ukernel__neonfp16arith_rr2_p2_u96_acc6, nullptr);
    }
  }

  TEST(F16_RADDSTOREEXPMINUSMAX__NEONFP16ARITH_RR2_P2_U96_ACC6, elements_lt_96) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t elements = 1; elements < 96; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f16_raddstoreexpminusmax_ukernel__neonfp16arith_rr2_p2_u96_acc6, nullptr);
    }
  }

  TEST(F16_RADDSTOREEXPMINUSMAX__NEONFP16ARITH_RR2_P2_U96_ACC6, elements_gt_96) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t elements = 97; elements < 192; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f16_raddstoreexpminusmax_ukernel__neonfp16arith_rr2_p2_u96_acc6, nullptr);
    }
  }
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_RADDSTOREEXPMINUSMAX__AVX2_RR1_P2_U16, elements_eq_16) {
    TEST_REQUIRES_X86_AVX2;
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(16)
      .Test(xnn_f16_raddstoreexpminusmax_ukernel__avx2_rr1_p2_u16, nullptr);
  }

  TEST(F16_RADDSTOREEXPMINUSMAX__AVX2_RR1_P2_U16, elements_div_16) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t elements = 32; elements < 160; elements += 16) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f16_raddstoreexpminusmax_ukernel__avx2_rr1_p2_u16, nullptr);
    }
  }

  TEST(F16_RADDSTOREEXPMINUSMAX__AVX2_RR1_P2_U16, elements_lt_16) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t elements = 1; elements < 16; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f16_raddstoreexpminusmax_ukernel__avx2_rr1_p2_u16, nullptr);
    }
  }

  TEST(F16_RADDSTOREEXPMINUSMAX__AVX2_RR1_P2_U16, elements_gt_16) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t elements = 17; elements < 32; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f16_raddstoreexpminusmax_ukernel__avx2_rr1_p2_u16, nullptr);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_RADDSTOREEXPMINUSMAX__AVX2_RR1_P2_U16_ACC2, elements_eq_16) {
    TEST_REQUIRES_X86_AVX2;
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(16)
      .Test(xnn_f16_raddstoreexpminusmax_ukernel__avx2_rr1_p2_u16_acc2, nullptr);
  }

  TEST(F16_RADDSTOREEXPMINUSMAX__AVX2_RR1_P2_U16_ACC2, elements_div_16) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t elements = 32; elements < 160; elements += 16) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f16_raddstoreexpminusmax_ukernel__avx2_rr1_p2_u16_acc2, nullptr);
    }
  }

  TEST(F16_RADDSTOREEXPMINUSMAX__AVX2_RR1_P2_U16_ACC2, elements_lt_16) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t elements = 1; elements < 16; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f16_raddstoreexpminusmax_ukernel__avx2_rr1_p2_u16_acc2, nullptr);
    }
  }

  TEST(F16_RADDSTOREEXPMINUSMAX__AVX2_RR1_P2_U16_ACC2, elements_gt_16) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t elements = 17; elements < 32; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f16_raddstoreexpminusmax_ukernel__avx2_rr1_p2_u16_acc2, nullptr);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_RADDSTOREEXPMINUSMAX__AVX2_RR1_P2_U32, elements_eq_32) {
    TEST_REQUIRES_X86_AVX2;
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(32)
      .Test(xnn_f16_raddstoreexpminusmax_ukernel__avx2_rr1_p2_u32, nullptr);
  }

  TEST(F16_RADDSTOREEXPMINUSMAX__AVX2_RR1_P2_U32, elements_div_32) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t elements = 64; elements < 320; elements += 32) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f16_raddstoreexpminusmax_ukernel__avx2_rr1_p2_u32, nullptr);
    }
  }

  TEST(F16_RADDSTOREEXPMINUSMAX__AVX2_RR1_P2_U32, elements_lt_32) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t elements = 1; elements < 32; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f16_raddstoreexpminusmax_ukernel__avx2_rr1_p2_u32, nullptr);
    }
  }

  TEST(F16_RADDSTOREEXPMINUSMAX__AVX2_RR1_P2_U32, elements_gt_32) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t elements = 33; elements < 64; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f16_raddstoreexpminusmax_ukernel__avx2_rr1_p2_u32, nullptr);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_RADDSTOREEXPMINUSMAX__AVX2_RR1_P2_U32_ACC2, elements_eq_32) {
    TEST_REQUIRES_X86_AVX2;
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(32)
      .Test(xnn_f16_raddstoreexpminusmax_ukernel__avx2_rr1_p2_u32_acc2, nullptr);
  }

  TEST(F16_RADDSTOREEXPMINUSMAX__AVX2_RR1_P2_U32_ACC2, elements_div_32) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t elements = 64; elements < 320; elements += 32) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f16_raddstoreexpminusmax_ukernel__avx2_rr1_p2_u32_acc2, nullptr);
    }
  }

  TEST(F16_RADDSTOREEXPMINUSMAX__AVX2_RR1_P2_U32_ACC2, elements_lt_32) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t elements = 1; elements < 32; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f16_raddstoreexpminusmax_ukernel__avx2_rr1_p2_u32_acc2, nullptr);
    }
  }

  TEST(F16_RADDSTOREEXPMINUSMAX__AVX2_RR1_P2_U32_ACC2, elements_gt_32) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t elements = 33; elements < 64; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f16_raddstoreexpminusmax_ukernel__avx2_rr1_p2_u32_acc2, nullptr);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_RADDSTOREEXPMINUSMAX__AVX2_RR1_P2_U32_ACC4, elements_eq_32) {
    TEST_REQUIRES_X86_AVX2;
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(32)
      .Test(xnn_f16_raddstoreexpminusmax_ukernel__avx2_rr1_p2_u32_acc4, nullptr);
  }

  TEST(F16_RADDSTOREEXPMINUSMAX__AVX2_RR1_P2_U32_ACC4, elements_div_32) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t elements = 64; elements < 320; elements += 32) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f16_raddstoreexpminusmax_ukernel__avx2_rr1_p2_u32_acc4, nullptr);
    }
  }

  TEST(F16_RADDSTOREEXPMINUSMAX__AVX2_RR1_P2_U32_ACC4, elements_lt_32) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t elements = 1; elements < 32; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f16_raddstoreexpminusmax_ukernel__avx2_rr1_p2_u32_acc4, nullptr);
    }
  }

  TEST(F16_RADDSTOREEXPMINUSMAX__AVX2_RR1_P2_U32_ACC4, elements_gt_32) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t elements = 33; elements < 64; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f16_raddstoreexpminusmax_ukernel__avx2_rr1_p2_u32_acc4, nullptr);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_RADDSTOREEXPMINUSMAX__AVX2_RR1_P2_U40, elements_eq_40) {
    TEST_REQUIRES_X86_AVX2;
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(40)
      .Test(xnn_f16_raddstoreexpminusmax_ukernel__avx2_rr1_p2_u40, nullptr);
  }

  TEST(F16_RADDSTOREEXPMINUSMAX__AVX2_RR1_P2_U40, elements_div_40) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t elements = 80; elements < 400; elements += 40) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f16_raddstoreexpminusmax_ukernel__avx2_rr1_p2_u40, nullptr);
    }
  }

  TEST(F16_RADDSTOREEXPMINUSMAX__AVX2_RR1_P2_U40, elements_lt_40) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t elements = 1; elements < 40; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f16_raddstoreexpminusmax_ukernel__avx2_rr1_p2_u40, nullptr);
    }
  }

  TEST(F16_RADDSTOREEXPMINUSMAX__AVX2_RR1_P2_U40, elements_gt_40) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t elements = 41; elements < 80; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f16_raddstoreexpminusmax_ukernel__avx2_rr1_p2_u40, nullptr);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_RADDSTOREEXPMINUSMAX__AVX2_RR1_P2_U40_ACC2, elements_eq_40) {
    TEST_REQUIRES_X86_AVX2;
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(40)
      .Test(xnn_f16_raddstoreexpminusmax_ukernel__avx2_rr1_p2_u40_acc2, nullptr);
  }

  TEST(F16_RADDSTOREEXPMINUSMAX__AVX2_RR1_P2_U40_ACC2, elements_div_40) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t elements = 80; elements < 400; elements += 40) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f16_raddstoreexpminusmax_ukernel__avx2_rr1_p2_u40_acc2, nullptr);
    }
  }

  TEST(F16_RADDSTOREEXPMINUSMAX__AVX2_RR1_P2_U40_ACC2, elements_lt_40) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t elements = 1; elements < 40; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f16_raddstoreexpminusmax_ukernel__avx2_rr1_p2_u40_acc2, nullptr);
    }
  }

  TEST(F16_RADDSTOREEXPMINUSMAX__AVX2_RR1_P2_U40_ACC2, elements_gt_40) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t elements = 41; elements < 80; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f16_raddstoreexpminusmax_ukernel__avx2_rr1_p2_u40_acc2, nullptr);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_RADDSTOREEXPMINUSMAX__AVX2_RR1_P2_U40_ACC5, elements_eq_40) {
    TEST_REQUIRES_X86_AVX2;
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(40)
      .Test(xnn_f16_raddstoreexpminusmax_ukernel__avx2_rr1_p2_u40_acc5, nullptr);
  }

  TEST(F16_RADDSTOREEXPMINUSMAX__AVX2_RR1_P2_U40_ACC5, elements_div_40) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t elements = 80; elements < 400; elements += 40) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f16_raddstoreexpminusmax_ukernel__avx2_rr1_p2_u40_acc5, nullptr);
    }
  }

  TEST(F16_RADDSTOREEXPMINUSMAX__AVX2_RR1_P2_U40_ACC5, elements_lt_40) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t elements = 1; elements < 40; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f16_raddstoreexpminusmax_ukernel__avx2_rr1_p2_u40_acc5, nullptr);
    }
  }

  TEST(F16_RADDSTOREEXPMINUSMAX__AVX2_RR1_P2_U40_ACC5, elements_gt_40) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t elements = 41; elements < 80; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f16_raddstoreexpminusmax_ukernel__avx2_rr1_p2_u40_acc5, nullptr);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_RADDSTOREEXPMINUSMAX__AVX2_RR1_P2_U48, elements_eq_48) {
    TEST_REQUIRES_X86_AVX2;
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(48)
      .Test(xnn_f16_raddstoreexpminusmax_ukernel__avx2_rr1_p2_u48, nullptr);
  }

  TEST(F16_RADDSTOREEXPMINUSMAX__AVX2_RR1_P2_U48, elements_div_48) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t elements = 96; elements < 480; elements += 48) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f16_raddstoreexpminusmax_ukernel__avx2_rr1_p2_u48, nullptr);
    }
  }

  TEST(F16_RADDSTOREEXPMINUSMAX__AVX2_RR1_P2_U48, elements_lt_48) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t elements = 1; elements < 48; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f16_raddstoreexpminusmax_ukernel__avx2_rr1_p2_u48, nullptr);
    }
  }

  TEST(F16_RADDSTOREEXPMINUSMAX__AVX2_RR1_P2_U48, elements_gt_48) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t elements = 49; elements < 96; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f16_raddstoreexpminusmax_ukernel__avx2_rr1_p2_u48, nullptr);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_RADDSTOREEXPMINUSMAX__AVX2_RR1_P2_U48_ACC2, elements_eq_48) {
    TEST_REQUIRES_X86_AVX2;
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(48)
      .Test(xnn_f16_raddstoreexpminusmax_ukernel__avx2_rr1_p2_u48_acc2, nullptr);
  }

  TEST(F16_RADDSTOREEXPMINUSMAX__AVX2_RR1_P2_U48_ACC2, elements_div_48) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t elements = 96; elements < 480; elements += 48) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f16_raddstoreexpminusmax_ukernel__avx2_rr1_p2_u48_acc2, nullptr);
    }
  }

  TEST(F16_RADDSTOREEXPMINUSMAX__AVX2_RR1_P2_U48_ACC2, elements_lt_48) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t elements = 1; elements < 48; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f16_raddstoreexpminusmax_ukernel__avx2_rr1_p2_u48_acc2, nullptr);
    }
  }

  TEST(F16_RADDSTOREEXPMINUSMAX__AVX2_RR1_P2_U48_ACC2, elements_gt_48) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t elements = 49; elements < 96; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f16_raddstoreexpminusmax_ukernel__avx2_rr1_p2_u48_acc2, nullptr);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_RADDSTOREEXPMINUSMAX__AVX2_RR1_P2_U48_ACC3, elements_eq_48) {
    TEST_REQUIRES_X86_AVX2;
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(48)
      .Test(xnn_f16_raddstoreexpminusmax_ukernel__avx2_rr1_p2_u48_acc3, nullptr);
  }

  TEST(F16_RADDSTOREEXPMINUSMAX__AVX2_RR1_P2_U48_ACC3, elements_div_48) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t elements = 96; elements < 480; elements += 48) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f16_raddstoreexpminusmax_ukernel__avx2_rr1_p2_u48_acc3, nullptr);
    }
  }

  TEST(F16_RADDSTOREEXPMINUSMAX__AVX2_RR1_P2_U48_ACC3, elements_lt_48) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t elements = 1; elements < 48; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f16_raddstoreexpminusmax_ukernel__avx2_rr1_p2_u48_acc3, nullptr);
    }
  }

  TEST(F16_RADDSTOREEXPMINUSMAX__AVX2_RR1_P2_U48_ACC3, elements_gt_48) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t elements = 49; elements < 96; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f16_raddstoreexpminusmax_ukernel__avx2_rr1_p2_u48_acc3, nullptr);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_RADDSTOREEXPMINUSMAX__AVX2_RR1_P2_U64, elements_eq_64) {
    TEST_REQUIRES_X86_AVX2;
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(64)
      .Test(xnn_f16_raddstoreexpminusmax_ukernel__avx2_rr1_p2_u64, nullptr);
  }

  TEST(F16_RADDSTOREEXPMINUSMAX__AVX2_RR1_P2_U64, elements_div_64) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t elements = 128; elements < 640; elements += 64) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f16_raddstoreexpminusmax_ukernel__avx2_rr1_p2_u64, nullptr);
    }
  }

  TEST(F16_RADDSTOREEXPMINUSMAX__AVX2_RR1_P2_U64, elements_lt_64) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t elements = 1; elements < 64; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f16_raddstoreexpminusmax_ukernel__avx2_rr1_p2_u64, nullptr);
    }
  }

  TEST(F16_RADDSTOREEXPMINUSMAX__AVX2_RR1_P2_U64, elements_gt_64) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t elements = 65; elements < 128; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f16_raddstoreexpminusmax_ukernel__avx2_rr1_p2_u64, nullptr);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_RADDSTOREEXPMINUSMAX__AVX2_RR1_P2_U64_ACC2, elements_eq_64) {
    TEST_REQUIRES_X86_AVX2;
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(64)
      .Test(xnn_f16_raddstoreexpminusmax_ukernel__avx2_rr1_p2_u64_acc2, nullptr);
  }

  TEST(F16_RADDSTOREEXPMINUSMAX__AVX2_RR1_P2_U64_ACC2, elements_div_64) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t elements = 128; elements < 640; elements += 64) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f16_raddstoreexpminusmax_ukernel__avx2_rr1_p2_u64_acc2, nullptr);
    }
  }

  TEST(F16_RADDSTOREEXPMINUSMAX__AVX2_RR1_P2_U64_ACC2, elements_lt_64) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t elements = 1; elements < 64; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f16_raddstoreexpminusmax_ukernel__avx2_rr1_p2_u64_acc2, nullptr);
    }
  }

  TEST(F16_RADDSTOREEXPMINUSMAX__AVX2_RR1_P2_U64_ACC2, elements_gt_64) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t elements = 65; elements < 128; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f16_raddstoreexpminusmax_ukernel__avx2_rr1_p2_u64_acc2, nullptr);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_RADDSTOREEXPMINUSMAX__AVX2_RR1_P2_U64_ACC4, elements_eq_64) {
    TEST_REQUIRES_X86_AVX2;
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(64)
      .Test(xnn_f16_raddstoreexpminusmax_ukernel__avx2_rr1_p2_u64_acc4, nullptr);
  }

  TEST(F16_RADDSTOREEXPMINUSMAX__AVX2_RR1_P2_U64_ACC4, elements_div_64) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t elements = 128; elements < 640; elements += 64) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f16_raddstoreexpminusmax_ukernel__avx2_rr1_p2_u64_acc4, nullptr);
    }
  }

  TEST(F16_RADDSTOREEXPMINUSMAX__AVX2_RR1_P2_U64_ACC4, elements_lt_64) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t elements = 1; elements < 64; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f16_raddstoreexpminusmax_ukernel__avx2_rr1_p2_u64_acc4, nullptr);
    }
  }

  TEST(F16_RADDSTOREEXPMINUSMAX__AVX2_RR1_P2_U64_ACC4, elements_gt_64) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t elements = 65; elements < 128; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f16_raddstoreexpminusmax_ukernel__avx2_rr1_p2_u64_acc4, nullptr);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_RADDSTOREEXPMINUSMAX__AVX2_RR1_P2_U72, elements_eq_72) {
    TEST_REQUIRES_X86_AVX2;
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(72)
      .Test(xnn_f16_raddstoreexpminusmax_ukernel__avx2_rr1_p2_u72, nullptr);
  }

  TEST(F16_RADDSTOREEXPMINUSMAX__AVX2_RR1_P2_U72, elements_div_72) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t elements = 144; elements < 720; elements += 72) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f16_raddstoreexpminusmax_ukernel__avx2_rr1_p2_u72, nullptr);
    }
  }

  TEST(F16_RADDSTOREEXPMINUSMAX__AVX2_RR1_P2_U72, elements_lt_72) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t elements = 1; elements < 72; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f16_raddstoreexpminusmax_ukernel__avx2_rr1_p2_u72, nullptr);
    }
  }

  TEST(F16_RADDSTOREEXPMINUSMAX__AVX2_RR1_P2_U72, elements_gt_72) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t elements = 73; elements < 144; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f16_raddstoreexpminusmax_ukernel__avx2_rr1_p2_u72, nullptr);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_RADDSTOREEXPMINUSMAX__AVX2_RR1_P2_U72_ACC3, elements_eq_72) {
    TEST_REQUIRES_X86_AVX2;
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(72)
      .Test(xnn_f16_raddstoreexpminusmax_ukernel__avx2_rr1_p2_u72_acc3, nullptr);
  }

  TEST(F16_RADDSTOREEXPMINUSMAX__AVX2_RR1_P2_U72_ACC3, elements_div_72) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t elements = 144; elements < 720; elements += 72) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f16_raddstoreexpminusmax_ukernel__avx2_rr1_p2_u72_acc3, nullptr);
    }
  }

  TEST(F16_RADDSTOREEXPMINUSMAX__AVX2_RR1_P2_U72_ACC3, elements_lt_72) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t elements = 1; elements < 72; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f16_raddstoreexpminusmax_ukernel__avx2_rr1_p2_u72_acc3, nullptr);
    }
  }

  TEST(F16_RADDSTOREEXPMINUSMAX__AVX2_RR1_P2_U72_ACC3, elements_gt_72) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t elements = 73; elements < 144; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f16_raddstoreexpminusmax_ukernel__avx2_rr1_p2_u72_acc3, nullptr);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_RADDSTOREEXPMINUSMAX__AVX2_RR1_P2_U80, elements_eq_80) {
    TEST_REQUIRES_X86_AVX2;
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(80)
      .Test(xnn_f16_raddstoreexpminusmax_ukernel__avx2_rr1_p2_u80, nullptr);
  }

  TEST(F16_RADDSTOREEXPMINUSMAX__AVX2_RR1_P2_U80, elements_div_80) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t elements = 160; elements < 800; elements += 80) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f16_raddstoreexpminusmax_ukernel__avx2_rr1_p2_u80, nullptr);
    }
  }

  TEST(F16_RADDSTOREEXPMINUSMAX__AVX2_RR1_P2_U80, elements_lt_80) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t elements = 1; elements < 80; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f16_raddstoreexpminusmax_ukernel__avx2_rr1_p2_u80, nullptr);
    }
  }

  TEST(F16_RADDSTOREEXPMINUSMAX__AVX2_RR1_P2_U80, elements_gt_80) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t elements = 81; elements < 160; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f16_raddstoreexpminusmax_ukernel__avx2_rr1_p2_u80, nullptr);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_RADDSTOREEXPMINUSMAX__AVX2_RR1_P2_U80_ACC2, elements_eq_80) {
    TEST_REQUIRES_X86_AVX2;
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(80)
      .Test(xnn_f16_raddstoreexpminusmax_ukernel__avx2_rr1_p2_u80_acc2, nullptr);
  }

  TEST(F16_RADDSTOREEXPMINUSMAX__AVX2_RR1_P2_U80_ACC2, elements_div_80) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t elements = 160; elements < 800; elements += 80) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f16_raddstoreexpminusmax_ukernel__avx2_rr1_p2_u80_acc2, nullptr);
    }
  }

  TEST(F16_RADDSTOREEXPMINUSMAX__AVX2_RR1_P2_U80_ACC2, elements_lt_80) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t elements = 1; elements < 80; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f16_raddstoreexpminusmax_ukernel__avx2_rr1_p2_u80_acc2, nullptr);
    }
  }

  TEST(F16_RADDSTOREEXPMINUSMAX__AVX2_RR1_P2_U80_ACC2, elements_gt_80) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t elements = 81; elements < 160; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f16_raddstoreexpminusmax_ukernel__avx2_rr1_p2_u80_acc2, nullptr);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_RADDSTOREEXPMINUSMAX__AVX2_RR1_P2_U80_ACC5, elements_eq_80) {
    TEST_REQUIRES_X86_AVX2;
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(80)
      .Test(xnn_f16_raddstoreexpminusmax_ukernel__avx2_rr1_p2_u80_acc5, nullptr);
  }

  TEST(F16_RADDSTOREEXPMINUSMAX__AVX2_RR1_P2_U80_ACC5, elements_div_80) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t elements = 160; elements < 800; elements += 80) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f16_raddstoreexpminusmax_ukernel__avx2_rr1_p2_u80_acc5, nullptr);
    }
  }

  TEST(F16_RADDSTOREEXPMINUSMAX__AVX2_RR1_P2_U80_ACC5, elements_lt_80) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t elements = 1; elements < 80; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f16_raddstoreexpminusmax_ukernel__avx2_rr1_p2_u80_acc5, nullptr);
    }
  }

  TEST(F16_RADDSTOREEXPMINUSMAX__AVX2_RR1_P2_U80_ACC5, elements_gt_80) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t elements = 81; elements < 160; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f16_raddstoreexpminusmax_ukernel__avx2_rr1_p2_u80_acc5, nullptr);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_RADDSTOREEXPMINUSMAX__AVX2_RR1_P2_U96, elements_eq_96) {
    TEST_REQUIRES_X86_AVX2;
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(96)
      .Test(xnn_f16_raddstoreexpminusmax_ukernel__avx2_rr1_p2_u96, nullptr);
  }

  TEST(F16_RADDSTOREEXPMINUSMAX__AVX2_RR1_P2_U96, elements_div_96) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t elements = 192; elements < 960; elements += 96) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f16_raddstoreexpminusmax_ukernel__avx2_rr1_p2_u96, nullptr);
    }
  }

  TEST(F16_RADDSTOREEXPMINUSMAX__AVX2_RR1_P2_U96, elements_lt_96) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t elements = 1; elements < 96; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f16_raddstoreexpminusmax_ukernel__avx2_rr1_p2_u96, nullptr);
    }
  }

  TEST(F16_RADDSTOREEXPMINUSMAX__AVX2_RR1_P2_U96, elements_gt_96) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t elements = 97; elements < 192; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f16_raddstoreexpminusmax_ukernel__avx2_rr1_p2_u96, nullptr);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_RADDSTOREEXPMINUSMAX__AVX2_RR1_P2_U96_ACC2, elements_eq_96) {
    TEST_REQUIRES_X86_AVX2;
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(96)
      .Test(xnn_f16_raddstoreexpminusmax_ukernel__avx2_rr1_p2_u96_acc2, nullptr);
  }

  TEST(F16_RADDSTOREEXPMINUSMAX__AVX2_RR1_P2_U96_ACC2, elements_div_96) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t elements = 192; elements < 960; elements += 96) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f16_raddstoreexpminusmax_ukernel__avx2_rr1_p2_u96_acc2, nullptr);
    }
  }

  TEST(F16_RADDSTOREEXPMINUSMAX__AVX2_RR1_P2_U96_ACC2, elements_lt_96) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t elements = 1; elements < 96; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f16_raddstoreexpminusmax_ukernel__avx2_rr1_p2_u96_acc2, nullptr);
    }
  }

  TEST(F16_RADDSTOREEXPMINUSMAX__AVX2_RR1_P2_U96_ACC2, elements_gt_96) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t elements = 97; elements < 192; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f16_raddstoreexpminusmax_ukernel__avx2_rr1_p2_u96_acc2, nullptr);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_RADDSTOREEXPMINUSMAX__AVX2_RR1_P2_U96_ACC3, elements_eq_96) {
    TEST_REQUIRES_X86_AVX2;
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(96)
      .Test(xnn_f16_raddstoreexpminusmax_ukernel__avx2_rr1_p2_u96_acc3, nullptr);
  }

  TEST(F16_RADDSTOREEXPMINUSMAX__AVX2_RR1_P2_U96_ACC3, elements_div_96) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t elements = 192; elements < 960; elements += 96) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f16_raddstoreexpminusmax_ukernel__avx2_rr1_p2_u96_acc3, nullptr);
    }
  }

  TEST(F16_RADDSTOREEXPMINUSMAX__AVX2_RR1_P2_U96_ACC3, elements_lt_96) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t elements = 1; elements < 96; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f16_raddstoreexpminusmax_ukernel__avx2_rr1_p2_u96_acc3, nullptr);
    }
  }

  TEST(F16_RADDSTOREEXPMINUSMAX__AVX2_RR1_P2_U96_ACC3, elements_gt_96) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t elements = 97; elements < 192; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f16_raddstoreexpminusmax_ukernel__avx2_rr1_p2_u96_acc3, nullptr);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64


#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  TEST(F16_RADDSTOREEXPMINUSMAX__AVX2_RR1_P2_U96_ACC6, elements_eq_96) {
    TEST_REQUIRES_X86_AVX2;
    RAddStoreExpMinusMaxMicrokernelTester()
      .elements(96)
      .Test(xnn_f16_raddstoreexpminusmax_ukernel__avx2_rr1_p2_u96_acc6, nullptr);
  }

  TEST(F16_RADDSTOREEXPMINUSMAX__AVX2_RR1_P2_U96_ACC6, elements_div_96) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t elements = 192; elements < 960; elements += 96) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f16_raddstoreexpminusmax_ukernel__avx2_rr1_p2_u96_acc6, nullptr);
    }
  }

  TEST(F16_RADDSTOREEXPMINUSMAX__AVX2_RR1_P2_U96_ACC6, elements_lt_96) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t elements = 1; elements < 96; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f16_raddstoreexpminusmax_ukernel__avx2_rr1_p2_u96_acc6, nullptr);
    }
  }

  TEST(F16_RADDSTOREEXPMINUSMAX__AVX2_RR1_P2_U96_ACC6, elements_gt_96) {
    TEST_REQUIRES_X86_AVX2;
    for (size_t elements = 97; elements < 192; elements++) {
      RAddStoreExpMinusMaxMicrokernelTester()
        .elements(elements)
        .Test(xnn_f16_raddstoreexpminusmax_ukernel__avx2_rr1_p2_u96_acc6, nullptr);
    }
  }
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64
