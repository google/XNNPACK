// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.


#include <string>

#include <gtest/gtest.h>
#include "xnnpack/common.h"
#include "xnnpack/isa-checks.h"
#include "xnnpack/packw.h"
#include "next_prime.h"
#include "packw-microkernel-tester.h"

namespace {

struct XnnTestParam {
  const char *name;
  xnn_x8_packw_gemm_goi_ukernel_fn ukernel;
  uint64_t arch_flags;
  size_t nr, kr, sr, kblock, nr_scale;
};

class XnnTest : public testing::TestWithParam<XnnTestParam> {
};

std::string GetTestName(const testing::TestParamInfo<XnnTest::ParamType>& info) {
  return info.param.name;
}

#define XNN_UKERNEL(arch_flags, ukernel, nr, kr, sr, kblock, nr_scale) \
  { #ukernel, ukernel, arch_flags, nr, kr, sr, kblock, nr_scale },

const XnnTestParam xnn_test_params[] = {
#include "x8-packw/x8-packw.h"
};

#undef XNN_UKERNEL

}  // namespace

TEST_P(XnnTest, null_bias) {
  TEST_REQUIRES_ARCH_FLAGS(GetParam().arch_flags);
  PackWMicrokernelTester()
    .nullbias(true)
    .n(GetParam().nr * GetParam().nr_scale)
    .k(GetParam().kblock)
    .nr(GetParam().nr * GetParam().nr_scale)
    .kr(GetParam().kr)
    .sr(GetParam().sr)
    .Test(GetParam().ukernel);
}

TEST_P(XnnTest, k_eq_kblock) {
  TEST_REQUIRES_ARCH_FLAGS(GetParam().arch_flags);
  PackWMicrokernelTester()
    .n(GetParam().nr * GetParam().nr_scale)
    .k(GetParam().kblock)
    .nr(GetParam().nr * GetParam().nr_scale)
    .kr(GetParam().kr)
    .sr(GetParam().sr)
    .Test(GetParam().ukernel);
}

TEST_P(XnnTest, k_div_kblock) {
  TEST_REQUIRES_ARCH_FLAGS(GetParam().arch_flags);
  for (size_t k = GetParam().kblock; k < GetParam().kblock * 5; k += GetParam().kblock) {
    PackWMicrokernelTester()
      .n(GetParam().nr * GetParam().nr_scale)
      .k(k)
      .nr(GetParam().nr * GetParam().nr_scale)
      .kr(GetParam().kr)
      .sr(GetParam().sr)
      .Test(GetParam().ukernel);
  }
}

TEST_P(XnnTest, k_lt_kblock) {
  if (GetParam().kblock <= 1) {
    GTEST_SKIP();
  }
  TEST_REQUIRES_ARCH_FLAGS(GetParam().arch_flags);
  for (size_t k = 1; k < GetParam().kblock; k++) {
    PackWMicrokernelTester()
      .n(GetParam().nr * GetParam().nr_scale)
      .k(k)
      .nr(GetParam().nr * GetParam().nr_scale)
      .kr(GetParam().kr)
      .sr(GetParam().sr)
      .Test(GetParam().ukernel);
  }
}

TEST_P(XnnTest, k_gt_kblock) {
  TEST_REQUIRES_ARCH_FLAGS(GetParam().arch_flags);
  for (size_t k = GetParam().kblock + 1; k < GetParam().kblock * 5; k = xnnpack::NextPrime(k + 1)) {
    PackWMicrokernelTester()
      .n(GetParam().nr * GetParam().nr_scale)
      .k(k)
      .nr(GetParam().nr * GetParam().nr_scale)
      .kr(GetParam().kr)
      .sr(GetParam().sr)
      .Test(GetParam().ukernel);
  }
}

TEST_P(XnnTest, n_eq_1) {
  if (GetParam().nr <= 1 || GetParam().nr_scale != 1) {
    GTEST_SKIP();
  }
  TEST_REQUIRES_ARCH_FLAGS(GetParam().arch_flags);
  PackWMicrokernelTester()
    .n(1 * GetParam().nr_scale)
    .k(GetParam().kblock)
    .nr(GetParam().nr * GetParam().nr_scale)
    .kr(GetParam().kr)
    .sr(GetParam().sr)
    .Test(GetParam().ukernel);
}

TEST_P(XnnTest, n_div_nr) {
  TEST_REQUIRES_ARCH_FLAGS(GetParam().arch_flags);
  for (size_t n = GetParam().nr; n < GetParam().nr * 5; n += GetParam().nr) {
    PackWMicrokernelTester()
      .n(n * GetParam().nr_scale)
      .k(GetParam().kblock)
      .nr(GetParam().nr * GetParam().nr_scale)
      .kr(GetParam().kr)
      .sr(GetParam().sr)
      .Test(GetParam().ukernel);
  }
}

TEST_P(XnnTest, n_lt_nr) {
  if (GetParam().nr <= 1 || GetParam().nr_scale != 1) {
    GTEST_SKIP();
  }
  TEST_REQUIRES_ARCH_FLAGS(GetParam().arch_flags);
  for (size_t n = 1; n < GetParam().nr * GetParam().nr_scale; n++) {
    PackWMicrokernelTester()
      .n(n)
      .k(GetParam().kblock)
      .nr(GetParam().nr * GetParam().nr_scale)
      .kr(GetParam().kr)
      .sr(GetParam().sr)
      .Test(GetParam().ukernel);
  }
}

TEST_P(XnnTest, n_gt_nr) {
  TEST_REQUIRES_ARCH_FLAGS(GetParam().arch_flags);
  for (size_t n = GetParam().nr * GetParam().nr_scale; n < GetParam().nr * GetParam().nr_scale * 5; n = xnnpack::NextPrime(n + 1)) {
    PackWMicrokernelTester()
      .n(n)
      .k(xnnpack::NextPrime(GetParam().kblock + 1))
      .nr(GetParam().nr * GetParam().nr_scale)
      .kr(GetParam().kr)
      .sr(GetParam().sr)
      .Test(GetParam().ukernel);
  }
}
INSTANTIATE_TEST_SUITE_P(x8_packw,
                         XnnTest,
                         testing::ValuesIn(xnn_test_params),
                         GetTestName);
