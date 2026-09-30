// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <cstddef>
#include <cstdint>
#include <string>
#include <vector>

#include <gtest/gtest.h>
#include "src/xnnpack/common.h"
#include "src/xnnpack/isa-checks.h"
#include "src/xnnpack/microfnptr.h"
#include "src/xnnpack/packw.h"
#include "test/next_prime.h"
#include "test/packw-microkernel-tester.h"

namespace {

struct XnnTestQU8Param {
  const char* name;
  xnn_qu8_packw_gemm_goi_ukernel_fn ukernel;
  uint64_t arch_flags;
  size_t nr, kr, sr, kblock, nr_scale;
};

class XnnTestQU8 : public testing::TestWithParam<XnnTestQU8Param> {};

std::string GetTestQU8Name(
    const testing::TestParamInfo<XnnTestQU8::ParamType>& info) {
  return info.param.name;
}

#define XNN_QU8_UKERNEL(arch_flags, ukernel, nr, kr, sr, kblock, nr_scale) \
  {#ukernel, ukernel, arch_flags, nr, kr, sr, kblock, nr_scale},

// The list is empty on targets without x86 AVX2/AVX256VNNI kernels.
const std::vector<XnnTestQU8Param> xnn_test_qu8_params = {
#include "src/qu8-packw/qu8-packw.inc"
};

#undef XNN_QU8_UKERNEL

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(XnnTestQU8);

PackWMicrokernelTester Tester(const XnnTestQU8Param& p) {
  return PackWMicrokernelTester().nr(p.nr * p.nr_scale).kr(p.kr).sr(p.sr);
}

TEST_P(XnnTestQU8, null_bias) {
  TEST_REQUIRES_ARCH_FLAGS(GetParam().arch_flags);
  Tester(GetParam())
      .nullbias(true)
      .n(GetParam().nr * GetParam().nr_scale)
      .k(GetParam().kblock)
      .Test(GetParam().ukernel);
}

TEST_P(XnnTestQU8, k_eq_kblock) {
  TEST_REQUIRES_ARCH_FLAGS(GetParam().arch_flags);
  Tester(GetParam())
      .n(GetParam().nr * GetParam().nr_scale)
      .k(GetParam().kblock)
      .Test(GetParam().ukernel);
}

TEST_P(XnnTestQU8, n_stride) {
  TEST_REQUIRES_ARCH_FLAGS(GetParam().arch_flags);
  Tester(GetParam())
      .n(GetParam().nr * GetParam().nr_scale * 2 + 2)
      .k(GetParam().kblock + 1)
      .n_stride(GetParam().kblock + 17)
      .Test(GetParam().ukernel);
}

TEST_P(XnnTestQU8, k_div_kblock) {
  TEST_REQUIRES_ARCH_FLAGS(GetParam().arch_flags);
  for (size_t k = GetParam().kblock; k < GetParam().kblock * 5;
       k += GetParam().kblock) {
    Tester(GetParam())
        .n(GetParam().nr * GetParam().nr_scale)
        .k(k)
        .Test(GetParam().ukernel);
  }
}

TEST_P(XnnTestQU8, k_lt_kblock) {
  TEST_REQUIRES_ARCH_FLAGS(GetParam().arch_flags);
  for (size_t k = 1; k < GetParam().kblock; k++) {
    Tester(GetParam())
        .n(GetParam().nr * GetParam().nr_scale)
        .k(k)
        .Test(GetParam().ukernel);
  }
}

TEST_P(XnnTestQU8, k_gt_kblock) {
  TEST_REQUIRES_ARCH_FLAGS(GetParam().arch_flags);
  for (size_t k = GetParam().kblock + 1; k < GetParam().kblock * 5;
       k = xnnpack::NextPrime(k + 1)) {
    Tester(GetParam())
        .n(GetParam().nr * GetParam().nr_scale)
        .k(k)
        .Test(GetParam().ukernel);
  }
}

TEST_P(XnnTestQU8, k_large) {
  TEST_REQUIRES_ARCH_FLAGS(GetParam().arch_flags);
  // Large K exercises the 32-byte main loop and ksum magnitude.
  for (size_t k : {size_t{255}, size_t{1024}, size_t{4099}}) {
    Tester(GetParam())
        .n(GetParam().nr * GetParam().nr_scale + 1)
        .k(k)
        .Test(GetParam().ukernel);
  }
}

TEST_P(XnnTestQU8, n_div_nr) {
  TEST_REQUIRES_ARCH_FLAGS(GetParam().arch_flags);
  for (size_t n = GetParam().nr; n < GetParam().nr * 5; n += GetParam().nr) {
    Tester(GetParam())
        .n(n * GetParam().nr_scale)
        .k(GetParam().kblock)
        .Test(GetParam().ukernel);
  }
}

TEST_P(XnnTestQU8, n_div_nr_null_bias) {
  TEST_REQUIRES_ARCH_FLAGS(GetParam().arch_flags);
  for (size_t n = GetParam().nr; n < GetParam().nr * 5; n += GetParam().nr) {
    Tester(GetParam())
        .nullbias(true)
        .n(n * GetParam().nr_scale)
        .k(GetParam().kblock)
        .Test(GetParam().ukernel);
  }
}

TEST_P(XnnTestQU8, n_lt_nr) {
  TEST_REQUIRES_ARCH_FLAGS(GetParam().arch_flags);
  for (size_t n = 1; n < GetParam().nr * GetParam().nr_scale; n++) {
    Tester(GetParam())
        .n(n)
        .k(xnnpack::NextPrime(GetParam().kblock + 1))
        .Test(GetParam().ukernel);
  }
}

TEST_P(XnnTestQU8, n_gt_nr) {
  TEST_REQUIRES_ARCH_FLAGS(GetParam().arch_flags);
  for (size_t n = GetParam().nr * GetParam().nr_scale;
       n < GetParam().nr * GetParam().nr_scale * 5;
       n = xnnpack::NextPrime(n + 1)) {
    Tester(GetParam())
        .n(n)
        .k(xnnpack::NextPrime(GetParam().kblock + 1))
        .Test(GetParam().ukernel);
  }
}

INSTANTIATE_TEST_SUITE_P(qu8_packw, XnnTestQU8,
                         testing::ValuesIn(xnn_test_qu8_params),
                         GetTestQU8Name);

}  // namespace
