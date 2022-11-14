// Copyright 2019 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <gtest/gtest.h>

#include <xnnpack/common.h>
#include <xnnpack/isa-checks.h>

#include <xnnpack/gavgpool.h>
#include "gavgpool-cw-microkernel-tester.h"


#if XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  TEST(F16_GAVGPOOL_CW__NEONFP16ARITH_X8, elements_eq_8) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    GAvgPoolCWMicrokernelTester()
      .elements(8)
      .channels(4)
      .Test(xnn_f16_gavgpool_cw_ukernel__neonfp16arith_x8, xnn_init_f16_gavgpool_neonfp16arith_params);
  }

  TEST(F16_GAVGPOOL_CW__NEONFP16ARITH_X8, channels_eq_8) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    GAvgPoolCWMicrokernelTester()
      .elements(8)
      .channels(8)
      .Test(xnn_f16_gavgpool_cw_ukernel__neonfp16arith_x8, xnn_init_f16_gavgpool_neonfp16arith_params);
  }

  TEST(F16_GAVGPOOL_CW__NEONFP16ARITH_X8, elements_div_8) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t elements = 8; elements < 32; elements += 8) {
      GAvgPoolCWMicrokernelTester()
        .elements(elements)
        .channels(4)
        .Test(xnn_f16_gavgpool_cw_ukernel__neonfp16arith_x8, xnn_init_f16_gavgpool_neonfp16arith_params);
    }
  }

  TEST(F16_GAVGPOOL_CW__NEONFP16ARITH_X8, elements_lt_8) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t elements = 1; elements < 8; elements++) {
      GAvgPoolCWMicrokernelTester()
        .elements(elements)
        .channels(4)
        .Test(xnn_f16_gavgpool_cw_ukernel__neonfp16arith_x8, xnn_init_f16_gavgpool_neonfp16arith_params);
    }
  }

  TEST(F16_GAVGPOOL_CW__NEONFP16ARITH_X8, elements_gt_8) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t elements = 9; elements < 16; elements++) {
      GAvgPoolCWMicrokernelTester()
        .elements(elements)
        .channels(4)
        .Test(xnn_f16_gavgpool_cw_ukernel__neonfp16arith_x8, xnn_init_f16_gavgpool_neonfp16arith_params);
    }
  }

  TEST(F16_GAVGPOOL_CW__NEONFP16ARITH_X8, channels_lt_4) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t channels = 1; channels < 4; channels++) {
      for (size_t elements = 1; elements < 16; elements += 3) {
        GAvgPoolCWMicrokernelTester()
          .elements(elements)
          .channels(channels)
          .Test(xnn_f16_gavgpool_cw_ukernel__neonfp16arith_x8, xnn_init_f16_gavgpool_neonfp16arith_params);
      }
    }
  }

  TEST(F16_GAVGPOOL_CW__NEONFP16ARITH_X8, channels_gt_4) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t channels = 5; channels < 8; channels++) {
      for (size_t elements = 1; elements < 16; elements += 3) {
        GAvgPoolCWMicrokernelTester()
          .elements(elements)
          .channels(channels)
          .Test(xnn_f16_gavgpool_cw_ukernel__neonfp16arith_x8, xnn_init_f16_gavgpool_neonfp16arith_params);
      }
    }
  }

  TEST(F16_GAVGPOOL_CW__NEONFP16ARITH_X8, channels_div_4) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t channels = 8; channels <= 16; channels += 4) {
      for (size_t elements = 1; elements < 16; elements += 3) {
        GAvgPoolCWMicrokernelTester()
          .elements(elements)
          .channels(channels)
          .Test(xnn_f16_gavgpool_cw_ukernel__neonfp16arith_x8, xnn_init_f16_gavgpool_neonfp16arith_params);
      }
    }
  }

  TEST(F16_GAVGPOOL_CW__NEONFP16ARITH_X8, qmin) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t elements = 1; elements < 16; elements += 3) {
      GAvgPoolCWMicrokernelTester()
        .elements(elements)
        .channels(4)
        .qmin(128)
        .Test(xnn_f16_gavgpool_cw_ukernel__neonfp16arith_x8, xnn_init_f16_gavgpool_neonfp16arith_params);
    }
  }

  TEST(F16_GAVGPOOL_CW__NEONFP16ARITH_X8, qmax) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    for (size_t elements = 1; elements < 16; elements += 3) {
      GAvgPoolCWMicrokernelTester()
        .elements(elements)
        .channels(4)
        .qmax(128)
        .Test(xnn_f16_gavgpool_cw_ukernel__neonfp16arith_x8, xnn_init_f16_gavgpool_neonfp16arith_params);
    }
  }

#endif  // XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
