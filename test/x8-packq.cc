// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <cstddef>
#include <string>

#include <gtest/gtest.h>
#include "src/xnnpack/common.h"
#include "src/xnnpack/isa-checks.h"
#include "src/xnnpack/packq.h"
#include "test/packq-microkernel-tester.h"

namespace {

struct XnnTestParam {
  const char* name;
  xnn_x8_packq_f32qp8_ukernel_fn ukernel;
  uint64_t arch_flags;
  int unroll;
};

class XnnTest : public testing::TestWithParam<XnnTestParam> {};

std::string GetTestName(
    const testing::TestParamInfo<XnnTest::ParamType>& info) {
  return info.param.name;
}

#define XNN_UKERNEL(arch_flags, ukernel, unroll) \
  {#ukernel, ukernel, arch_flags, unroll},

const XnnTestParam xnn_test_params[] = {
#include "src/x8-packq/x8-packq.h"
};

#undef XNN_UKERNEL

}  // namespace

namespace xnnpack {

TEST_P(XnnTest, k_div_kr_m_div_mr) {
  TEST_REQUIRES_ARCH_FLAGS(GetParam().arch_flags);
  for (size_t kr : {1, 2, 4}) {
    for (size_t mr = 1; mr <= 4; mr++) {
      PackQMicrokernelTester()
          .m(mr * GetParam().unroll * 10)
          .k(kr * GetParam().unroll * 10)
          .mr(mr)
          .kr(kr)
          .Test(GetParam().ukernel);
    }
  }
}

TEST_P(XnnTest, k_div_kr_m_lt_mr) {
  TEST_REQUIRES_ARCH_FLAGS(GetParam().arch_flags);
  for (size_t kr : {1, 2, 4}) {
    for (size_t mr = 2; mr <= 4; mr++) {
      PackQMicrokernelTester()
          .m(mr - 1)
          .k(kr * GetParam().unroll * 10)
          .mr(mr)
          .kr(kr)
          .Test(GetParam().ukernel);
    }
  }
}

INSTANTIATE_TEST_SUITE_P(x8_packq, XnnTest, testing::ValuesIn(xnn_test_params),
                         GetTestName);

}  // namespace xnnpack
