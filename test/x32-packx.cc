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

struct XnnTestParam {
  const char *name;
  bool (*isa_check)();
  xnn_x32_packx_ukernel_fn fn;
  size_t k, mr;
};

class XnnTest : public testing::TestWithParam<XnnTestParam> {
};

std::string GetTestName(const testing::TestParamInfo<XnnTest::ParamType>& info) {
  return info.param.name;
}

const XnnTestParam xnn_test_params[] = {
#if XNN_ARCH_ARM || XNN_ARCH_ARM64
  { "X32_PACKX_4X__NEON_ST4_X4", []() { return TEST_REQUIRES_ARM_NEON_VALUE; }, xnn_x32_packx_ukernel_4x__neon_st4_x4, /*k=*/4, /*mr=*/4 },
  { "X32_PACKX_4X__NEON_ST4_X4_PRFM", []() { return TEST_REQUIRES_ARM_NEON_VALUE; }, xnn_x32_packx_ukernel_4x__neon_st4_x4_prfm, /*k=*/4, /*mr=*/4 },
  { "X32_PACKX_4X__NEON_ST4_X8", []() { return TEST_REQUIRES_ARM_NEON_VALUE; }, xnn_x32_packx_ukernel_4x__neon_st4_x8, /*k=*/8, /*mr=*/4 },
  { "X32_PACKX_4X__NEON_ST4_X8_PRFM", []() { return TEST_REQUIRES_ARM_NEON_VALUE; }, xnn_x32_packx_ukernel_4x__neon_st4_x8_prfm, /*k=*/8, /*mr=*/4 },
  { "X32_PACKX_8X__NEON_ST4_X4", []() { return TEST_REQUIRES_ARM_NEON_VALUE; }, xnn_x32_packx_ukernel_8x__neon_st4_x4, /*k=*/4, /*mr=*/8 },
  { "X32_PACKX_8X__NEON_ST4_X4_PRFM", []() { return TEST_REQUIRES_ARM_NEON_VALUE; }, xnn_x32_packx_ukernel_8x__neon_st4_x4_prfm, /*k=*/4, /*mr=*/8 },
  { "X32_PACKX_8X__NEON_ST4_X8", []() { return TEST_REQUIRES_ARM_NEON_VALUE; }, xnn_x32_packx_ukernel_8x__neon_st4_x8, /*k=*/8, /*mr=*/8 },
  { "X32_PACKX_8X__NEON_ST4_X8_PRFM", []() { return TEST_REQUIRES_ARM_NEON_VALUE; }, xnn_x32_packx_ukernel_8x__neon_st4_x8_prfm, /*k=*/8, /*mr=*/8 },
#endif  // XNN_ARCH_ARM || XNN_ARCH_ARM64
#if XNN_ARCH_X86 || XNN_ARCH_X86_64
  { "X32_PACKX_4X__SSE", []() { return TEST_REQUIRES_X86_SSE_VALUE; }, xnn_x32_packx_ukernel_4x__sse, /*k=*/4, /*mr=*/4 },
#endif  // XNN_ARCH_X86 || XNN_ARCH_X86_64
#if XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  { "X32_PACKX_4X__WASMSIMD", []() { return true; }, xnn_x32_packx_ukernel_4x__wasmsimd, /*k=*/4, /*mr=*/4 },
#endif  // XNN_ARCH_WASMSIMD || XNN_ARCH_WASMRELAXEDSIMD
  { "X32_PACKX_2X__SCALAR", []() { return true; }, xnn_x32_packx_ukernel_2x__scalar, /*k=*/1, /*mr=*/2 },
  { "X32_PACKX_3X__SCALAR", []() { return true; }, xnn_x32_packx_ukernel_3x__scalar, /*k=*/1, /*mr=*/3 },
  { "X32_PACKX_4X__SCALAR", []() { return true; }, xnn_x32_packx_ukernel_4x__scalar, /*k=*/1, /*mr=*/4 },
};

TEST_P(XnnTest, k_eq_kblock) {
  if (!GetParam().isa_check()) {
    GTEST_SKIP();
  }
  PackMicrokernelTester()
    .mr(GetParam().mr)
    .m(GetParam().mr)
    .k(GetParam().k)
    .Test(GetParam().fn);
}

TEST_P(XnnTest, k_eq_kblock_subtile) {
  if (!GetParam().isa_check()) {
    GTEST_SKIP();
  }
  for (size_t m = 1; m <= GetParam().mr; m++) {
    PackMicrokernelTester()
      .mr(GetParam().mr)
      .m(m)
      .k(GetParam().k)
      .Test(GetParam().fn);
  }
}

TEST_P(XnnTest, k_lt_kblock) {
  if (GetParam().k == 1) {
    GTEST_SKIP();
  }
  if (!GetParam().isa_check()) {
    GTEST_SKIP();
  }
  for (size_t k = 1; k < GetParam().k; k++) {
    PackMicrokernelTester()
      .mr(GetParam().mr)
      .m(GetParam().mr)
      .k(k)
      .Test(GetParam().fn);
  }
}

TEST_P(XnnTest, k_lt_kblock_subtile) {
  if (GetParam().k == 1) {
    GTEST_SKIP();
  }
  if (!GetParam().isa_check()) {
    GTEST_SKIP();
  }
  for (size_t k = 1; k < GetParam().k; k++) {
    for (size_t m = 1; m <= GetParam().mr; m++) {
      PackMicrokernelTester()
        .mr(GetParam().mr)
        .m(m)
        .k(k)
        .Test(GetParam().fn);
    }
  }
}

TEST_P(XnnTest, k_gt_kblock) {
  if (!GetParam().isa_check()) {
    GTEST_SKIP();
  }
  for (size_t k = GetParam().k + 1; k < (GetParam().k == 1 ? 10 : GetParam().k * 2); k++) {
    PackMicrokernelTester()
      .mr(GetParam().mr)
      .m(GetParam().mr)
      .k(k)
      .Test(GetParam().fn);
  }
}

TEST_P(XnnTest, k_gt_kblock_subtile) {
  if (!GetParam().isa_check()) {
    GTEST_SKIP();
  }
  for (size_t k = GetParam().k + 1; k < (GetParam().k == 1 ? 10 : GetParam().k * 2); k++) {
    for (size_t m = 1; m <= GetParam().mr; m++) {
      PackMicrokernelTester()
        .mr(GetParam().mr)
        .m(m)
        .k(k)
        .Test(GetParam().fn);
    }
  }
}

TEST_P(XnnTest, k_div_kblock) {
  if (GetParam().k <= 1) {
    GTEST_SKIP();
  }
  if (!GetParam().isa_check()) {
    GTEST_SKIP();
  }
  for (size_t k = GetParam().k * 2; k < GetParam().k * 10; k += 4) {
    PackMicrokernelTester()
      .mr(GetParam().mr)
      .m(GetParam().mr)
      .k(k)
      .Test(GetParam().fn);
  }
}

TEST_P(XnnTest, k_div_kblock_subtile) {
  if (GetParam().k <= 1) {
    GTEST_SKIP();
  }
  if (!GetParam().isa_check()) {
    GTEST_SKIP();
  }
  for (size_t k = GetParam().k * 2; k < GetParam().k * 10; k += 4) {
    for (size_t m = 1; m <= GetParam().mr; m++) {
      PackMicrokernelTester()
        .mr(GetParam().mr)
        .m(m)
        .k(k)
        .Test(GetParam().fn);
    }
  }
}

TEST_P(XnnTest, strided_x) {
  if (!GetParam().isa_check()) {
    GTEST_SKIP();
  }
  for (size_t k = 1; k <= GetParam().k * 5; k += GetParam().k + 1) {
    PackMicrokernelTester()
      .mr(GetParam().mr)
      .m(GetParam().mr)
      .k(k)
      .x_stride(xnnpack::NextPrime(GetParam().k * 5 + 1))
      .Test(GetParam().fn);
  }
}
INSTANTIATE_TEST_SUITE_P(x32_packx,
                         XnnTest,
                         testing::ValuesIn(xnn_test_params),
                         GetTestName);
