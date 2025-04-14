// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.


#include <string>

#include <gtest/gtest.h>
#include "src/xnnpack/common.h"
#include "src/xnnpack/isa-checks.h"
#include "src/xnnpack/packw.h"
#include "test/next_prime.h"
#include "test/packw-microkernel-tester.h"

namespace {

struct XnnTestQB4Param {
  const char *name;
  xnn_qb4_packw_gemm_goi_ukernel_fn ukernel;
  uint64_t arch_flags;
  size_t nr, kr, sr, kblock, bl, nr_scale, izp;
};

class XnnTestQB4 : public testing::TestWithParam<XnnTestQB4Param> {
};

std::string GetTestQB4Name(const testing::TestParamInfo<XnnTestQB4::ParamType>& info) {
  return info.param.name;
}

#define XNN_QB4_UKERNEL(arch_flags, ukernel, nr, kr, sr, kblock, bl, nr_scale, izp) \
  { #ukernel, ukernel, arch_flags, nr, kr, sr, kblock, bl, nr_scale, izp },

const XnnTestQB4Param xnn_test_qb4_params[] = {
#include "src/qb4-packw/qb4-packw.h"
};

#undef XNN_QB4_UKERNEL

TEST_P(XnnTestQB4, null_bias) {
  TEST_REQUIRES_ARCH_FLAGS(GetParam().arch_flags);
  PackWMicrokernelTester()
    .nullbias(true)
    .n(GetParam().nr * GetParam().nr_scale)
    .k(GetParam().kblock)
    .nr(GetParam().nr * GetParam().nr_scale)
    .kr(GetParam().kr)
    .sr(GetParam().sr)
    .bl(GetParam().bl)
    .izp(GetParam().izp)
    .Test(GetParam().ukernel);
}

TEST_P(XnnTestQB4, bias) {
  TEST_REQUIRES_ARCH_FLAGS(GetParam().arch_flags);
  PackWMicrokernelTester()
    .nullbias(false)
    .n(GetParam().nr * GetParam().nr_scale)
    .k(GetParam().kblock)
    .nr(GetParam().nr * GetParam().nr_scale)
    .kr(GetParam().kr)
    .sr(GetParam().sr)
    .bl(GetParam().bl)
    .izp(GetParam().izp)
    .Test(GetParam().ukernel);
}

TEST_P(XnnTestQB4, kb_gt_bl_no_bias) {
  TEST_REQUIRES_ARCH_FLAGS(GetParam().arch_flags);
  for(int kb_scale = 2; kb_scale < 7; kb_scale++){
    PackWMicrokernelTester()
      .nullbias(true)
      .n(GetParam().nr * GetParam().nr_scale)
      .k(GetParam().kblock * kb_scale)
      .nr(GetParam().nr * GetParam().nr_scale)
      .kr(GetParam().kr)
      .sr(GetParam().sr)
      .bl(GetParam().bl)
      .izp(GetParam().izp)
      .Test(GetParam().ukernel);
  }
}

TEST_P(XnnTestQB4, kb_gt_bl_bias) {
  TEST_REQUIRES_ARCH_FLAGS(GetParam().arch_flags);
  for(int kb_scale = 2; kb_scale < 7; kb_scale++){
    PackWMicrokernelTester()
      .nullbias(false)
      .n(GetParam().nr * GetParam().nr_scale)
      .k(GetParam().kblock * kb_scale)
      .nr(GetParam().nr * GetParam().nr_scale)
      .kr(GetParam().kr)
      .sr(GetParam().sr)
      .bl(GetParam().bl)
      .izp(GetParam().izp)
      .Test(GetParam().ukernel);
  }
}

TEST_P(XnnTestQB4, nr_divides_nc) {
  TEST_REQUIRES_ARCH_FLAGS(GetParam().arch_flags);
  PackWMicrokernelTester()
    .nullbias(true)
    .n(GetParam().nr * GetParam().nr_scale * 2)
    .k(GetParam().kblock)
    .nr(GetParam().nr * GetParam().nr_scale)
    .kr(GetParam().kr)
    .sr(GetParam().sr)
    .bl(GetParam().bl)
    .izp(GetParam().izp)
    .Test(GetParam().ukernel);
}

TEST_P(XnnTestQB4, nr_divides_nc_with_bias) {
  TEST_REQUIRES_ARCH_FLAGS(GetParam().arch_flags);
  PackWMicrokernelTester()
    .nullbias(false)
    .n(GetParam().nr * GetParam().nr_scale * 2)
    .k(GetParam().kblock)
    .nr(GetParam().nr * GetParam().nr_scale)
    .kr(GetParam().kr)
    .sr(GetParam().sr)
    .bl(GetParam().bl)
    .izp(GetParam().izp)
    .Test(GetParam().ukernel);
}

TEST_P(XnnTestQB4, nc_gt_nr) {
  TEST_REQUIRES_ARCH_FLAGS(GetParam().arch_flags);
  for(size_t ni = 1; ni < GetParam().nr; ++ni){
    PackWMicrokernelTester()
      .nullbias(false)
      .n(ni)
      .k(GetParam().kblock)
      .nr(GetParam().nr * GetParam().nr_scale)
      .kr(GetParam().kr)
      .sr(GetParam().sr)
      .bl(GetParam().bl)
      .izp(GetParam().izp)
      .Test(GetParam().ukernel);
  }
}


INSTANTIATE_TEST_SUITE_P(qb4_packw,
                         XnnTestQB4,
                         testing::ValuesIn(xnn_test_qb4_params),
                         GetTestQB4Name);


}  // namespace
