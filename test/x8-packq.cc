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
#include "src/x8-packq/x8-packq.inc"
};

#undef XNN_UKERNEL


TEST_P(XnnTest, k_div_kr_m_div_mr) {
  TEST_REQUIRES_ARCH_FLAGS(GetParam().arch_flags);
  for (size_t kr : {1, 2, 4}) {
    for (size_t mr = 1; mr <= 4; mr++) {
      xnnpack::PackQMicrokernelTester()
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
      xnnpack::PackQMicrokernelTester()
          .m(mr - 1)
          .k(kr * GetParam().unroll * 10)
          .mr(mr)
          .kr(kr)
          .Test(GetParam().ukernel);
    }
  }
}

constexpr size_t kKr4Ks[] = {1,  2,  3,  4,  5,  7,  8,   15,  16,
                             17, 28, 29, 31, 32, 33, 36,  60,  63,
                             64, 65, 96, 97, 127, 128, 129, 200, 257};

void TestKr4(const XnnTestParam& param, size_t m, size_t k, size_t mr) {
  xnnpack::PackQMicrokernelTester()
      .m(m)
      .k(k)
      .mr(mr)
      .kr(4)
      .Test(param.ukernel);
}

TEST_P(XnnTest, kr_eq_4_mr_eq_1_m_eq_1) {
  TEST_REQUIRES_ARCH_FLAGS(GetParam().arch_flags);
  for (size_t k : kKr4Ks) {
    TestKr4(GetParam(), 1, k, 1);
  }
}

TEST_P(XnnTest, kr_eq_4_mr_eq_32_m_le_16) {
  TEST_REQUIRES_ARCH_FLAGS(GetParam().arch_flags);
  for (size_t m = 1; m <= 16; m++) {
    for (size_t k : kKr4Ks) {
      TestKr4(GetParam(), m, k, 32);
    }
  }
}

TEST_P(XnnTest, kr_eq_4_mr_eq_32_m_gt_16) {
  TEST_REQUIRES_ARCH_FLAGS(GetParam().arch_flags);
  for (size_t m = 17; m <= 32; m++) {
    for (size_t k : kKr4Ks) {
      TestKr4(GetParam(), m, k, 32);
    }
  }
}

TEST_P(XnnTest, kr_eq_4_mr_eq_32_m_gt_32) {
  TEST_REQUIRES_ARCH_FLAGS(GetParam().arch_flags);
  for (size_t m : {33, 47, 64, 65, 100}) {
    for (size_t k : kKr4Ks) {
      TestKr4(GetParam(), m, k, 32);
    }
  }
}

TEST_P(XnnTest, mr_kr_sr_mixed) {
  TEST_REQUIRES_ARCH_FLAGS(GetParam().arch_flags);
  for (size_t sr : {1, 2}) {
    for (size_t kr : {4, 8, 16}) {
      for (size_t mr : {1, 4, 8, 16}) {
        for (size_t m : {static_cast<size_t>(1), mr - 1, mr, 2 * mr + 1}) {
          if (m == 0) {
            continue;
          }
          for (size_t k : {1, 15, 33, 100}) {
            xnnpack::PackQMicrokernelTester()
                .m(m)
                .k(k)
                .mr(mr)
                .kr(kr)
                .sr(sr)
                .Test(GetParam().ukernel);
          }
        }
      }
    }
  }
}

INSTANTIATE_TEST_SUITE_P(x8_packq, XnnTest, testing::ValuesIn(xnn_test_params),
                         GetTestName);

#if XNN_ENABLE_ARM_SME2 && XNN_ENABLE_ARM_SME2_ACLE
void TestSme2Kr4(size_t m, size_t k, size_t mr) {
  xnnpack::PackQMicrokernelTester()
      .m(m)
      .k(k)
      .mr(mr)
      .kr(4)
      .check_k_padding(false)
      .Test(xnn_x8_packq_f32qp8_ukernel__sme2);
}

TEST(x8_packq_sme2, kr_eq_4_mr_eq_1_m_eq_1) {
  TEST_REQUIRES_ARCH_FLAGS(xnn_arch_arm_sme2);
  for (size_t k : kKr4Ks) {
    TestSme2Kr4(1, k, 1);
  }
}

TEST(x8_packq_sme2, kr_eq_4_mr_eq_32_m_le_16) {
  TEST_REQUIRES_ARCH_FLAGS(xnn_arch_arm_sme2);
  for (size_t m = 1; m <= 16; m++) {
    for (size_t k : kKr4Ks) {
      TestSme2Kr4(m, k, 32);
    }
  }
}

TEST(x8_packq_sme2, kr_eq_4_mr_eq_32_m_gt_16) {
  TEST_REQUIRES_ARCH_FLAGS(xnn_arch_arm_sme2);
  for (size_t m = 17; m <= 32; m++) {
    for (size_t k : kKr4Ks) {
      TestSme2Kr4(m, k, 32);
    }
  }
}

TEST(x8_packq_sme2, kr_eq_4_mr_eq_32_m_gt_32) {
  TEST_REQUIRES_ARCH_FLAGS(xnn_arch_arm_sme2);
  for (size_t m : {33, 47, 64, 65, 100}) {
    for (size_t k : kKr4Ks) {
      TestSme2Kr4(m, k, 32);
    }
  }
}
#endif  // XNN_ENABLE_ARM_SME2 && XNN_ENABLE_ARM_SME2_ACLE

}  // namespace
