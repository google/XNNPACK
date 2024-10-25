// Copyright 2021 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <gtest/gtest.h>
#include "xnnpack/common.h"
#include "xnnpack/isa-checks.h"
#include "xnnpack/lut.h"
#include "lut-microkernel-tester.h"

#define XNN_TEST_LUT_BATCH_EQ(arch_flags, ukernel, batch_tile, ...)                                                  \
  TEST(ukernel, batch_eq)                                                                                            \
  {                                                                                                                  \
    LUTMicrokernelTester().batch_size(batch_tile).Test(ukernel);                                                     \
  }

#define XNN_TEST_LUT_BATCH_DIV(arch_flags, ukernel, batch_tile, ...)                                                 \
  TEST(ukernel, batch_div)                                                                                           \
  {                                                                                                                  \
    LUTMicrokernelTester().batch_size(batch_tile).Test(ukernel);                                                     \
  }

#define XNN_TEST_LUT_BATCH_LT(arch_flags, ukernel, batch_tile, ...)                                                  \
  TEST(ukernel, batch_lt)                                                                                            \
  {                                                                                                                  \
    for (size_t batch= 1; batch < batch_tile; batch++) {                                                             \
      LUTMicrokernelTester().batch_size(batch).Test(ukernel);                                                        \
    }                                                                                                                \
  }

#define XNN_TEST_LUT_BATCH_GT(arch_flags, ukernel, batch_tile, ...)                                                  \
  TEST(ukernel, batch_gt)                                                                                            \
  {                                                                                                                  \
    for (size_t batch = 2 * batch_tile; batch < 10 * batch_tile; batch += batch_tile) {                              \
      LUTMicrokernelTester().batch_size(batch).Test(ukernel);                                                        \
    }                                                                                                                \
  }

#define XNN_TEST_LUT_BATCH_IP(arch_flags, ukernel, batch_tile, ...)                                                  \
  TEST(ukernel, batch_ip)                                                                                            \
  {                                                                                                                  \
    for (size_t batch = 2 * batch_tile; batch < 10 * batch_tile; batch += batch_tile) {                              \
      LUTMicrokernelTester().batch_size(batch).inplace(true).Test(ukernel);                                          \
    }                                                                                                                \
  }

#define XNN_UKERNEL_WITH_PARAMS(arch_flags, ukernel, batch_size, datatype, params_type, init_params) XNN_TEST_LUT_BATCH_EQ(arch_flags, ukernel, batch_size, inplace);  \
XNN_TEST_LUT_BATCH_DIV(arch_flags, ukernel,  batch_size, inplace);                                                                                                     \
XNN_TEST_LUT_BATCH_LT(arch_flags, ukernel, batch_size, inplace);                                                                                                       \
XNN_TEST_LUT_BATCH_GT(arch_flags, ukernel, batch_size, inplace);                                                                                                       \
XNN_TEST_LUT_BATCH_IP(arch_flags, ukernel, batch_size, inplace);
#include "x8-lut/x8-lut.h"
#undef XNN_UKERNEL_WITH_PARAMS
