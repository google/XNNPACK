// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <string>

#include <gtest/gtest.h>
#include "xnnpack/common.h"
#include "xnnpack/isa-checks.h"
#include "xnnpack/packx.h"
#include "next_prime.h"
#include "pack-microkernel-tester.h"

namespace {

struct XnnTestParam {
  const char *name;
  xnn_x32_packx_ukernel_fn ukernel;
  uint64_t arch_flags;
  size_t k, mr;
};

class XnnTest : public testing::TestWithParam<XnnTestParam> {
};

std::string GetTestName(const testing::TestParamInfo<XnnTest::ParamType>& info) {
  return info.param.name;
}

#define XNN_UKERNEL(arch_flags, ukernel, k, mr) \
  { #ukernel, ukernel, arch_flags, k, mr },

const XnnTestParam xnn_test_params[] = {
#include "src/x32-packx/x32-packx.h"
};

#undef XNN_UKERNEL

}  // namespace

TEST_P(XnnTest, k_eq_kblock) {
  TEST_REQUIRES_ARCH_FLAGS(GetParam().arch_flags);
  PackMicrokernelTester()
    .mr(GetParam().mr)
    .m(GetParam().mr)
    .k(GetParam().k)
    .Test(GetParam().ukernel);
}

TEST_P(XnnTest, k_eq_kblock_subtile) {
  TEST_REQUIRES_ARCH_FLAGS(GetParam().arch_flags);
  for (size_t m = 1; m <= GetParam().mr; m++) {
    PackMicrokernelTester()
      .mr(GetParam().mr)
      .m(m)
      .k(GetParam().k)
      .Test(GetParam().ukernel);
  }
}

TEST_P(XnnTest, k_lt_kblock) {
  if (GetParam().k == 1) {
    GTEST_SKIP();
  }
  TEST_REQUIRES_ARCH_FLAGS(GetParam().arch_flags);
  for (size_t k = 1; k < GetParam().k; k++) {
    PackMicrokernelTester()
      .mr(GetParam().mr)
      .m(GetParam().mr)
      .k(k)
      .Test(GetParam().ukernel);
  }
}

TEST_P(XnnTest, k_lt_kblock_subtile) {
  if (GetParam().k == 1) {
    GTEST_SKIP();
  }
  TEST_REQUIRES_ARCH_FLAGS(GetParam().arch_flags);
  for (size_t k = 1; k < GetParam().k; k++) {
    for (size_t m = 1; m <= GetParam().mr; m++) {
      PackMicrokernelTester()
        .mr(GetParam().mr)
        .m(m)
        .k(k)
        .Test(GetParam().ukernel);
    }
  }
}

TEST_P(XnnTest, k_gt_kblock) {
  TEST_REQUIRES_ARCH_FLAGS(GetParam().arch_flags);
  for (size_t k = GetParam().k + 1; k < (GetParam().k == 1 ? 10 : GetParam().k * 2); k++) {
    PackMicrokernelTester()
      .mr(GetParam().mr)
      .m(GetParam().mr)
      .k(k)
      .Test(GetParam().ukernel);
  }
}

TEST_P(XnnTest, k_gt_kblock_subtile) {
  TEST_REQUIRES_ARCH_FLAGS(GetParam().arch_flags);
  for (size_t k = GetParam().k + 1; k < (GetParam().k == 1 ? 10 : GetParam().k * 2); k++) {
    for (size_t m = 1; m <= GetParam().mr; m++) {
      PackMicrokernelTester()
        .mr(GetParam().mr)
        .m(m)
        .k(k)
        .Test(GetParam().ukernel);
    }
  }
}

TEST_P(XnnTest, k_div_kblock) {
  if (GetParam().k <= 1) {
    GTEST_SKIP();
  }
  TEST_REQUIRES_ARCH_FLAGS(GetParam().arch_flags);
  for (size_t k = GetParam().k * 2; k < GetParam().k * 10; k += 4) {
    PackMicrokernelTester()
      .mr(GetParam().mr)
      .m(GetParam().mr)
      .k(k)
      .Test(GetParam().ukernel);
  }
}

TEST_P(XnnTest, k_div_kblock_subtile) {
  if (GetParam().k <= 1) {
    GTEST_SKIP();
  }
  TEST_REQUIRES_ARCH_FLAGS(GetParam().arch_flags);
  for (size_t k = GetParam().k * 2; k < GetParam().k * 10; k += 4) {
    for (size_t m = 1; m <= GetParam().mr; m++) {
      PackMicrokernelTester()
        .mr(GetParam().mr)
        .m(m)
        .k(k)
        .Test(GetParam().ukernel);
    }
  }
}

TEST_P(XnnTest, strided_x) {
  TEST_REQUIRES_ARCH_FLAGS(GetParam().arch_flags);
  for (size_t k = 1; k <= GetParam().k * 5; k += GetParam().k + 1) {
    PackMicrokernelTester()
      .mr(GetParam().mr)
      .m(GetParam().mr)
      .k(k)
      .x_stride(xnnpack::NextPrime(GetParam().k * 5 + 1))
      .Test(GetParam().ukernel);
  }
}
INSTANTIATE_TEST_SUITE_P(x32_packx,
                         XnnTest,
                         testing::ValuesIn(xnn_test_params),
                         GetTestName);
