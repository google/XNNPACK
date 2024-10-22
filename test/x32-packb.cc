// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.


#include <string>

#include <gtest/gtest.h>
#include "xnnpack/common.h"
#include "xnnpack/isa-checks.h"
#include "xnnpack/packb.h"
#include "xnnpack/zerob.h"
#include "packb-microkernel-tester.h"

namespace {

struct XnnTestParam {
  const char *name;
  PackBMicrokernelTester::Kernel kernel;
  uint64_t arch_flags;
  size_t channel_tile, channel_subtile, channel_round;
};

class XnnTest : public testing::TestWithParam<XnnTestParam> {};

std::string GetTestName(const testing::TestParamInfo<XnnTest::ParamType>& info) {
  return info.param.name;
}

#define XNN_UKERNEL(arch_flags, ukernel, channel_tile, channel_subtile, channel_round) \
  { #ukernel, PackBMicrokernelTester::Kernel{ukernel}, arch_flags, channel_tile, channel_subtile, channel_round },

const XnnTestParam xnn_test_params[] = {
#include "x32-packb/x32-packb.h"
#include "x32-zerob/x32-zerob.h"
};

#undef XNN_UKERNEL

}  // namespace

TEST_P(XnnTest, n_eq_channel_tile) {
  TEST_REQUIRES_ARCH_FLAGS(GetParam().arch_flags);
  for (size_t k = 1; k < 4; k++) {
    PackBMicrokernelTester()
      .channels(GetParam().channel_tile)
      .kernel_tile(k)
      .channel_tile(GetParam().channel_tile)
      .channel_subtile(GetParam().channel_subtile)
      .channel_round(GetParam().channel_round)
      .Test(GetParam().kernel);
  }
}

TEST_P(XnnTest, n_div_channel_tile) {
  TEST_REQUIRES_ARCH_FLAGS(GetParam().arch_flags);
  if (GetParam().channel_tile <= 1) {
    GTEST_SKIP();
  }
  for (size_t k = 1; k < 4; k++) {
    PackBMicrokernelTester()
      .channels(GetParam().channel_tile * 2)
      .kernel_tile(k)
      .channel_tile(GetParam().channel_tile)
      .channel_subtile(GetParam().channel_subtile)
      .channel_round(GetParam().channel_round)
      .Test(GetParam().kernel);
  }
}

TEST_P(XnnTest, n_lt_channel_tile) {
  TEST_REQUIRES_ARCH_FLAGS(GetParam().arch_flags);
  if (GetParam().channel_tile <= 1) {
    GTEST_SKIP();
  }
  for (size_t k = 1; k < 4; k++) {
    for (size_t n = 1; n < GetParam().channel_tile; n++) {
      PackBMicrokernelTester()
        .channels(n)
        .kernel_tile(k)
        .channel_tile(GetParam().channel_tile)
        .channel_subtile(GetParam().channel_subtile)
        .channel_round(GetParam().channel_round)
        .Test(GetParam().kernel);
    }
  }
}

TEST_P(XnnTest, n_gt_channel_tile) {
  TEST_REQUIRES_ARCH_FLAGS(GetParam().arch_flags);
  for (size_t k = 1; k < 4; k++) {
    for (size_t n = GetParam().channel_tile + 1; n < (GetParam().channel_tile == 1 ? 10 : GetParam().channel_tile * 2); n++) {
      PackBMicrokernelTester()
        .channels(n)
        .kernel_tile(k)
        .channel_tile(GetParam().channel_tile)
        .channel_subtile(GetParam().channel_subtile)
        .channel_round(GetParam().channel_round)
        .Test(GetParam().kernel);
    }
  }
}

TEST_P(XnnTest, groups_gt_1) {
  TEST_REQUIRES_ARCH_FLAGS(GetParam().arch_flags);
  for (size_t g = 2; g <= 3; g++) {
    for (size_t kernel_tile = 1; kernel_tile < 4; kernel_tile++) {
      for (size_t n = GetParam().channel_tile + 1; n < (GetParam().channel_tile == 1 ? 10 : GetParam().channel_tile * 2); n++) {
        PackBMicrokernelTester()
          .groups(g)
          .channels(n)
          .kernel_tile(kernel_tile)
          .channel_tile(GetParam().channel_tile)
          .channel_subtile(GetParam().channel_subtile)
          .channel_round(GetParam().channel_round)
          .Test(GetParam().kernel);
      }
    }
  }
}
INSTANTIATE_TEST_SUITE_P(x32_packb,
                         XnnTest,
                         testing::ValuesIn(xnn_test_params),
                         GetTestName);

