// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.


#include <cstddef>
#include <string>

#include <gtest/gtest.h>
#include "xnnpack/common.h"
#include "xnnpack/isa-checks.h"
#include "xnnpack/packq.h"
#include "packq-microkernel-tester.h"

struct XnnTestParam {
  const char *name;
  bool (*isa_check)();
  xnn_x8_packq_f32qp8_ukernel_fn fn;
  int unroll;
};

class XnnTest : public testing::TestWithParam<XnnTestParam> {
};

std::string GetTestName(const testing::TestParamInfo<XnnTest::ParamType>& info) {
  return info.param.name;
}

const XnnTestParam xnn_test_params[] = {
  { "X8_PACKQ_F32QP8__SCALAR_U1", []() { return true; }, xnn_x8_packq_f32qp8_ukernel__scalar_u1, /*unroll=*/1 },
#if XNN_ARCH_ARM64
#if XNN_ENABLE_KLEIDIAI
  { "X8_PACKQ_F32QP8__AARCH64_NEON_U2", []() { return TEST_REQUIRES_ARM_NEON_VALUE; }, xnn_x8_packq_f32qp8_ukernel__aarch64_neon_u2, /*unroll=*/2 }
#endif  // XNN_ENABLE_KLEIDIAI
#endif  // XNN_ARCH_ARM64
};

namespace xnnpack {

TEST_P(XnnTest, k_div_kr_m_div_mr) {
  if (!GetParam().isa_check()) {
    GTEST_SKIP();
  }
  for (size_t kr = 1; kr <= 4; kr++) {
    for (size_t mr = 1; mr <= 4; mr++) {
      PackQMicrokernelTester()
          .m(mr * GetParam().unroll * 10)
          .k(kr * GetParam().unroll * 10)
          .mr(mr)
          .kr(kr)
          .Test(GetParam().fn);
    }
  }
}

TEST_P(XnnTest, k_div_kr_m_div_mr_kr_div_sr) {
  if (!GetParam().isa_check()) {
    GTEST_SKIP();
  }
  for (size_t sr = 1; sr <= 4; sr++) {
    for (size_t kr = sr; kr <= 4 * sr; kr += sr) {
      for (size_t mr = 1; mr <= 4; mr++) {
        PackQMicrokernelTester()
            .m(mr * GetParam().unroll * 10)
            .k(kr * GetParam().unroll * 10)
            .mr(mr)
            .kr(kr)
            .sr(sr)
            .Test(GetParam().fn);
      }
    }
  }
}

TEST_P(XnnTest, k_div_kr_m_lt_mr) {
  if (!GetParam().isa_check()) {
    GTEST_SKIP();
  }
  for (size_t kr = 1; kr <= 4; kr++) {
    for (size_t mr = 2; mr <= 4; mr++) {
      PackQMicrokernelTester()
          .m(mr - 1)
          .k(kr * GetParam().unroll * 10)
          .mr(mr)
          .kr(kr)
          .Test(GetParam().fn);
    }
  }
}

TEST_P(XnnTest, k_div_kr_m_gt_mr) {
  if (!GetParam().isa_check()) {
    GTEST_SKIP();
  }
  for (size_t kr = 1; kr <= 4; kr++) {
    for (size_t mr = 2; mr <= 4; mr++) {
      PackQMicrokernelTester()
          .m(2 * mr + 1)
          .k(kr * GetParam().unroll * 10)
          .mr(mr)
          .kr(kr)
          .Test(GetParam().fn);
    }
  }
}

INSTANTIATE_TEST_SUITE_P(x8_packq,
                         XnnTest,
                         testing::ValuesIn(xnn_test_params),
                         GetTestName);

} // namespace xnnpack

