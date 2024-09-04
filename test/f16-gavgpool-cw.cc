// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.
//
// Auto-generated file. Do not edit!
//   Specification: test/f16-gavgpool-cw.yaml
//   Generator: tools/generate-gavgpool-cw-test.py


#include <gtest/gtest.h>
#include "xnnpack/common.h"
#include "xnnpack/gavgpool.h"
#include "xnnpack/isa-checks.h"
#include "xnnpack/microparams-init.h"
#include "gavgpool-cw-microkernel-tester.h"


#if XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
  TEST(F16_GAVGPOOL_CW__NEONFP16ARITH_U8, elements_eq_8) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    const size_t element_tile = 8;
    GAvgPoolCWMicrokernelTester()
      .elements(element_tile)
      .channels(1)
      .Test(xnn_f16_gavgpool_cw_ukernel__neonfp16arith_u8, xnn_init_f16_gavgpool_scalar_params);
  }

  TEST(F16_GAVGPOOL_CW__NEONFP16ARITH_U8, elements_gt_8) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    const size_t element_tile = 8;
    for (size_t elements = element_tile+1; elements < element_tile*2; elements++) {
      GAvgPoolCWMicrokernelTester()
        .elements(elements)
        .channels(1)
        .Test(xnn_f16_gavgpool_cw_ukernel__neonfp16arith_u8, xnn_init_f16_gavgpool_scalar_params);
    }
  }

  TEST(F16_GAVGPOOL_CW__NEONFP16ARITH_U8, elements_lt_8) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    const size_t element_tile = 8;
    for (size_t elements = 1; elements < element_tile; elements++) {
      GAvgPoolCWMicrokernelTester()
        .elements(elements)
        .channels(1)
        .Test(xnn_f16_gavgpool_cw_ukernel__neonfp16arith_u8, xnn_init_f16_gavgpool_scalar_params);
    }
  }

  TEST(F16_GAVGPOOL_CW__NEONFP16ARITH_U8, elements_div_8) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    const size_t element_tile = 8;
    for (size_t elements = element_tile*2; elements < element_tile*5; elements += element_tile) {
      GAvgPoolCWMicrokernelTester()
        .elements(elements)
        .channels(1)
        .Test(xnn_f16_gavgpool_cw_ukernel__neonfp16arith_u8, xnn_init_f16_gavgpool_scalar_params);
    }
  }

  TEST(F16_GAVGPOOL_CW__NEONFP16ARITH_U8, channels_gt_1) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    const size_t element_tile = 8;
    for (size_t channels = 2; channels < 4; channels++) {
      GAvgPoolCWMicrokernelTester()
        .elements(element_tile)
        .channels(channels)
        .Test(xnn_f16_gavgpool_cw_ukernel__neonfp16arith_u8, xnn_init_f16_gavgpool_scalar_params);
    }
  }

  TEST(F16_GAVGPOOL_CW__NEONFP16ARITH_U8, qmin) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    const size_t element_tile = 8;
    for (size_t elements = 1; elements < element_tile*2; elements += 3) {
      GAvgPoolCWMicrokernelTester()
        .elements(elements)
        .channels(4)
        .qmin(128)
        .Test(xnn_f16_gavgpool_cw_ukernel__neonfp16arith_u8, xnn_init_f16_gavgpool_scalar_params);
    }
  }

  TEST(F16_GAVGPOOL_CW__NEONFP16ARITH_U8, qmax) {
    TEST_REQUIRES_ARM_NEON_FP16_ARITH;
    const size_t element_tile = 8;
    for (size_t elements = 1; elements < element_tile*2; elements += 3) {
      GAvgPoolCWMicrokernelTester()
        .elements(elements)
        .channels(4)
        .qmax(128)
        .Test(xnn_f16_gavgpool_cw_ukernel__neonfp16arith_u8, xnn_init_f16_gavgpool_scalar_params);
    }
  }
#endif  // XNN_ENABLE_ARM_FP16_VECTOR && (XNN_ARCH_ARM || XNN_ARCH_ARM64)
