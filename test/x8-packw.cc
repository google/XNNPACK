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

struct XnnTestParam {
  const char *name;
  bool (*isa_check)();
  xnn_x8_packw_gemm_goi_ukernel_fn fn;
  size_t nr, kr, sr, kblock, nr_scale;
};

class XnnTest : public testing::TestWithParam<XnnTestParam> {
};

std::string GetTestName(const testing::TestParamInfo<XnnTest::ParamType>& info) {
  return info.param.name;
}

const XnnTestParam xnn_test_params[] = {
  { "X8_PACKW_GEMM_GOI_X2__SCALAR_INT_U2", []() { return true; }, xnn_x8_packw_gemm_goi_ukernel_x2__scalar_int_u2, /*NR=*/2, /*KR=*/1, /*SR=*/1, /*KBLOCK=*/2, /*NR_SCALE=*/1 },
  { "X8_PACKW_GEMM_GOI_X4__SCALAR_INT_U2", []() { return true; }, xnn_x8_packw_gemm_goi_ukernel_x4__scalar_int_u2, /*NR=*/4, /*KR=*/1, /*SR=*/1, /*KBLOCK=*/2, /*NR_SCALE=*/1 },
  { "X8_PACKW_GEMM_GOI_X8__SCALAR_INT_U2", []() { return true; }, xnn_x8_packw_gemm_goi_ukernel_x8__scalar_int_u2, /*NR=*/8, /*KR=*/1, /*SR=*/1, /*KBLOCK=*/2, /*NR_SCALE=*/1 },
  { "X8_PACKW_GEMM_GOI_X16__SCALAR_INT_U2", []() { return true; }, xnn_x8_packw_gemm_goi_ukernel_x16__scalar_int_u2, /*NR=*/16, /*KR=*/1, /*SR=*/1, /*KBLOCK=*/2, /*NR_SCALE=*/1 },
  { "X8_PACKW_GEMM_GOI_X32__SCALAR_INT_U2", []() { return true; }, xnn_x8_packw_gemm_goi_ukernel_x32__scalar_int_u2, /*NR=*/32, /*KR=*/1, /*SR=*/1, /*KBLOCK=*/2, /*NR_SCALE=*/1 },
  { "X8_PACKW_GEMM_GOI_X2__SCALAR_INT_U4", []() { return true; }, xnn_x8_packw_gemm_goi_ukernel_x2__scalar_int_u4, /*NR=*/2, /*KR=*/1, /*SR=*/1, /*KBLOCK=*/4, /*NR_SCALE=*/1 },
  { "X8_PACKW_GEMM_GOI_X4__SCALAR_INT_U4", []() { return true; }, xnn_x8_packw_gemm_goi_ukernel_x4__scalar_int_u4, /*NR=*/4, /*KR=*/1, /*SR=*/1, /*KBLOCK=*/4, /*NR_SCALE=*/1 },
  { "X8_PACKW_GEMM_GOI_X8__SCALAR_INT_U4", []() { return true; }, xnn_x8_packw_gemm_goi_ukernel_x8__scalar_int_u4, /*NR=*/8, /*KR=*/1, /*SR=*/1, /*KBLOCK=*/4, /*NR_SCALE=*/1 },
  { "X8_PACKW_GEMM_GOI_X16__SCALAR_INT_U4", []() { return true; }, xnn_x8_packw_gemm_goi_ukernel_x16__scalar_int_u4, /*NR=*/16, /*KR=*/1, /*SR=*/1, /*KBLOCK=*/4, /*NR_SCALE=*/1 },
  { "X8_PACKW_GEMM_GOI_X32__SCALAR_INT_U4", []() { return true; }, xnn_x8_packw_gemm_goi_ukernel_x32__scalar_int_u4, /*NR=*/32, /*KR=*/1, /*SR=*/1, /*KBLOCK=*/4, /*NR_SCALE=*/1 }
};

TEST_P(XnnTest, k_eq_kblock) {
  if (!GetParam().isa_check()) {
    GTEST_SKIP();
  }
  PackWMicrokernelTester()
    .n(GetParam().nr * GetParam().nr_scale)
    .k(GetParam().kblock)
    .nr(GetParam().nr * GetParam().nr_scale)
    .kr(GetParam().kr)
    .sr(GetParam().sr)
    .Test(GetParam().fn);
}

TEST_P(XnnTest, k_div_kblock) {
  if (GetParam().kblock <= 1) {
    GTEST_SKIP();
  }
  if (!GetParam().isa_check()) {
    GTEST_SKIP();
  }
  PackWMicrokernelTester()
    .n(GetParam().nr * GetParam().nr_scale)
    .k(GetParam().kblock * 5)
    .nr(GetParam().nr * GetParam().nr_scale)
    .kr(GetParam().kr)
    .sr(GetParam().sr)
    .Test(GetParam().fn);
}

TEST_P(XnnTest, k_lt_kblock) {
  if (GetParam().kblock <= 1) {
    GTEST_SKIP();
  }
  if (!GetParam().isa_check()) {
    GTEST_SKIP();
  }
  for (size_t k = 1; k < GetParam().kblock; k++) {
    PackWMicrokernelTester()
      .n(GetParam().nr * GetParam().nr_scale)
      .k(k)
      .nr(GetParam().nr * GetParam().nr_scale)
      .kr(GetParam().kr)
      .sr(GetParam().sr)
      .Test(GetParam().fn);
  }
}

TEST_P(XnnTest, k_gt_kblock) {
  if (!GetParam().isa_check()) {
    GTEST_SKIP();
  }
  for (size_t k = GetParam().kblock + 1; k < (GetParam().kblock == 1 ? 4 : GetParam().kblock * 2); k++) {
    PackWMicrokernelTester()
      .n(GetParam().nr * GetParam().nr_scale)
      .k(k)
      .nr(GetParam().nr * GetParam().nr_scale)
      .kr(GetParam().kr)
      .sr(GetParam().sr)
      .Test(GetParam().fn);
  }
}

TEST_P(XnnTest, n_eq_nr) {
  if (!GetParam().isa_check()) {
    GTEST_SKIP();
  }
  for (size_t k = 1; k < (GetParam().kblock == 1 ? 4 : GetParam().kblock * 2); k++) {
    PackWMicrokernelTester()
      .n(GetParam().nr * GetParam().nr_scale)
      .k(k)
      .nr(GetParam().nr * GetParam().nr_scale)
      .kr(GetParam().kr)
      .sr(GetParam().sr)
      .Test(GetParam().fn);
  }
}

TEST_P(XnnTest, n_div_nr) {
  if (GetParam().nr <= 1 || GetParam().nr_scale != 1) {
    GTEST_SKIP();
  }
  if (!GetParam().isa_check()) {
    GTEST_SKIP();
  }
  for (size_t k = 1; k < (GetParam().kblock == 1 ? 4 : GetParam().kblock * 2); k++) {
    PackWMicrokernelTester()
      .n(GetParam().nr * 2 * GetParam().nr_scale)
      .k(k)
      .nr(GetParam().nr * GetParam().nr_scale)
      .kr(GetParam().kr)
      .sr(GetParam().sr)
      .Test(GetParam().fn);
  }
}

TEST_P(XnnTest, n_lt_nr) {
  if (GetParam().nr <= 1 || GetParam().nr_scale != 1) {
    GTEST_SKIP();
  }
  if (!GetParam().isa_check()) {
    GTEST_SKIP();
  }
  for (size_t k = 1; k < (GetParam().kblock == 1 ? 4 : GetParam().kblock * 2); k++) {
    for (size_t n = 1; n < GetParam().nr * GetParam().nr_scale; n++) {
      PackWMicrokernelTester()
        .n(n)
        .k(k)
        .nr(GetParam().nr * GetParam().nr_scale)
        .kr(GetParam().kr)
        .sr(GetParam().sr)
        .Test(GetParam().fn);
    }
  }
}

TEST_P(XnnTest, n_gt_nr) {
  if (!GetParam().isa_check()) {
    GTEST_SKIP();
  }
  for (size_t k = 1; k < (GetParam().kblock == 1 ? 4 : GetParam().kblock * 2); k++) {
    if (GetParam().nr_scale == 1) {
      for (size_t n = GetParam().nr + 1; n < (GetParam().nr == 1 ? 4 : GetParam().nr * 2); n++) {
        PackWMicrokernelTester()
          .n(n)
          .k(k)
          .nr(GetParam().nr)
          .kr(GetParam().kr)
          .sr(GetParam().sr)
          .Test(GetParam().fn);
      }
    } else {
      for (size_t n = (GetParam().nr + 1) * GetParam().nr_scale;
                  n < (GetParam().nr == 1 ? 4 : GetParam().nr * 2) * GetParam().nr_scale;
                  n += 1 * GetParam().nr_scale) {
        PackWMicrokernelTester()
          .n(n)
          .k(k)
          .nr(GetParam().nr * GetParam().nr_scale)
          .kr(GetParam().kr)
          .sr(GetParam().sr)
          .Test(GetParam().fn);
      }
    }
  }
}

TEST_P(XnnTest, g_gt_1) {
  if (!GetParam().isa_check()) {
    GTEST_SKIP();
  }
  for (size_t g = 2; g <= 3; g++) {
    for (size_t k = 1; k < (GetParam().kblock == 1 ? 4 : GetParam().kblock * 2); k++) {
      if (GetParam().nr_scale == 1) {
        for (size_t n = GetParam().nr + 1; n < (GetParam().nr == 1 ? 4 : GetParam().nr * 2); n++) {
          PackWMicrokernelTester()
            .g(g)
            .n(n)
            .k(k)
            .nr(GetParam().nr)
            .kr(GetParam().kr)
            .sr(GetParam().sr)
            .Test(GetParam().fn);
        }
      } else {
        for (size_t n = (GetParam().nr + 1) * GetParam().nr_scale;
                    n < (GetParam().nr == 1 ? 4 : GetParam().nr * 2) * GetParam().nr_scale;
                    n += 1 * GetParam().nr_scale) {
          PackWMicrokernelTester()
            .g(g)
            .n(n)
            .k(k)
            .nr(GetParam().nr * GetParam().nr_scale)
            .kr(GetParam().kr)
            .sr(GetParam().sr)
            .Test(GetParam().fn);
        }
      }
    }
  }
}

TEST_P(XnnTest, null_bias) {
  if (!GetParam().isa_check()) {
    GTEST_SKIP();
  }
  for (size_t g = 2; g <= 3; g++) {
    for (size_t k = 1; k < (GetParam().kblock == 1 ? 4 : GetParam().kblock * 2); k++) {
      if (GetParam().nr_scale == 1) {
        for (size_t n = GetParam().nr + 1; n < (GetParam().nr == 1 ? 4 : GetParam().nr * 2); n++) {
          PackWMicrokernelTester()
            .nullbias(true)
            .g(g)
            .n(n)
            .k(k)
            .nr(GetParam().nr)
            .kr(GetParam().kr)
            .sr(GetParam().sr)
            .Test(GetParam().fn);
        }
      } else {
        for (size_t n = (GetParam().nr + 1) * GetParam().nr_scale;
                    n < (GetParam().nr == 1 ? 4 : GetParam().nr * 2) * GetParam().nr_scale;
                    n += 1 * GetParam().nr_scale) {
          PackWMicrokernelTester()
            .nullbias(true)
            .g(g)
            .n(n)
            .k(k)
            .nr(GetParam().nr * GetParam().nr_scale)
            .kr(GetParam().kr)
            .sr(GetParam().sr)
            .Test(GetParam().fn);
        }
      }
    }
  }
}
INSTANTIATE_TEST_SUITE_P(x8_packw,
                         XnnTest,
                         testing::ValuesIn(xnn_test_params),
                         GetTestName);

