// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.


#include <string>

#include <gtest/gtest.h>
#include "xnnpack/common.h"
#include "xnnpack/isa-checks.h"
#include "xnnpack/packw.h"
#include "packw-microkernel-tester.h"

namespace {

struct XnnTestQS8Param {
  const char *name;
  xnn_qs8_packw_gemm_goi_ukernel_fn ukernel;
  uint64_t arch_flags;
  size_t nr, kr, sr, kblock, nr_scale;
};

class XnnTestQS8 : public testing::TestWithParam<XnnTestQS8Param> {
};

std::string GetTestQS8Name(const testing::TestParamInfo<XnnTestQS8::ParamType>& info) {
  return info.param.name;
}

#define XNN_QS8_UKERNEL(arch_flags, ukernel, nr, kr, sr, kblock, nr_scale) \
  { #ukernel, ukernel, arch_flags, nr, kr, sr, kblock, nr_scale },

const XnnTestQS8Param xnn_test_qs8_params[] = {
#include "src/qs8-packw/qs8-packw.h"
};

#undef XNN_QS8_UKERNEL

}  // namespace

TEST_P(XnnTestQS8, k_eq_kblock) {
  TEST_REQUIRES_ARCH_FLAGS(GetParam().arch_flags);
  PackWMicrokernelTester()
    .n(GetParam().nr * GetParam().nr_scale)
    .k(GetParam().kblock)
    .nr(GetParam().nr * GetParam().nr_scale)
    .kr(GetParam().kr)
    .sr(GetParam().sr)
    .Test(GetParam().ukernel);
}

TEST_P(XnnTestQS8, k_div_kblock) {
  if (GetParam().kblock <= 1) {
    GTEST_SKIP();
  }
  TEST_REQUIRES_ARCH_FLAGS(GetParam().arch_flags);
  PackWMicrokernelTester()
    .n(GetParam().nr * GetParam().nr_scale)
    .k(GetParam().kblock * 5)
    .nr(GetParam().nr * GetParam().nr_scale)
    .kr(GetParam().kr)
    .sr(GetParam().sr)
    .Test(GetParam().ukernel);
}

TEST_P(XnnTestQS8, k_lt_kblock) {
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

TEST_P(XnnTestQS8, k_gt_kblock) {
  TEST_REQUIRES_ARCH_FLAGS(GetParam().arch_flags);
  for (size_t k = GetParam().kblock + 1; k < (GetParam().kblock == 1 ? 4 : GetParam().kblock * 2); k++) {
    PackWMicrokernelTester()
      .n(GetParam().nr * GetParam().nr_scale)
      .k(k)
      .nr(GetParam().nr * GetParam().nr_scale)
      .kr(GetParam().kr)
      .sr(GetParam().sr)
      .Test(GetParam().ukernel);
  }
}

TEST_P(XnnTestQS8, n_eq_nr) {
  TEST_REQUIRES_ARCH_FLAGS(GetParam().arch_flags);
  for (size_t k = 1; k < (GetParam().kblock == 1 ? 4 : GetParam().kblock * 2); k++) {
    PackWMicrokernelTester()
      .n(GetParam().nr * GetParam().nr_scale)
      .k(k)
      .nr(GetParam().nr * GetParam().nr_scale)
      .kr(GetParam().kr)
      .sr(GetParam().sr)
      .Test(GetParam().ukernel);
  }
}

INSTANTIATE_TEST_SUITE_P(qs8_packw,
                         XnnTestQS8,
                         testing::ValuesIn(xnn_test_qs8_params),
                         GetTestQS8Name);

