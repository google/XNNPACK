// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <cstddef>
#include <cstdint>

#ifndef XNN_PACKBTEST_PARAM_DEFINED
#define XNN_PACKBTEST_PARAM_DEFINED
template<typename KernelFn>
struct XnnPackBTestParam {
  const char *name;
  KernelFn kernel_fn;
  uint64_t arch_flags;
  size_t channel_tile, channel_subtile, channel_round;
};
#endif

#ifdef XNN_TEST_SUITE_NAME

TEST_P(XNN_TEST_SUITE_NAME, n_eq_channel_tile) {
  TEST_REQUIRES_ARCH_FLAGS(GetParam().arch_flags);
  for (size_t k = 1; k < 4; k++) {
    PackBMicrokernelTester()
      .channels(GetParam().channel_tile)
      .kernel_tile(k)
      .channel_tile(GetParam().channel_tile)
      .channel_subtile(GetParam().channel_subtile)
      .channel_round(GetParam().channel_round)
      .Test(GetParam().kernel_fn);
  }
}

TEST_P(XNN_TEST_SUITE_NAME, n_div_channel_tile) {
  TEST_REQUIRES_ARCH_FLAGS(GetParam().arch_flags);
  if (GetParam().channel_tile <= 1) {
    GTEST_SKIP();
  }
  for (size_t k = 1; k < 4; k++) {
    PackBMicrokernelTester()
      .channels(GetParam().channel_tile * 2)
      .kernel_tile(k)
      .channel_tile(GetParam().channel_tile)
      .channel_subtile(GetParam().channel_subtile)
      .channel_round(GetParam().channel_round)
      .Test(GetParam().kernel_fn);
  }
}

TEST_P(XNN_TEST_SUITE_NAME, n_lt_channel_tile) {
  TEST_REQUIRES_ARCH_FLAGS(GetParam().arch_flags);
  if (GetParam().channel_tile <= 1) {
    GTEST_SKIP();
  }
  for (size_t k = 1; k < 4; k++) {
    for (size_t n = 1; n < GetParam().channel_tile; n++) {
      PackBMicrokernelTester()
        .channels(n)
        .kernel_tile(k)
        .channel_tile(GetParam().channel_tile)
        .channel_subtile(GetParam().channel_subtile)
        .channel_round(GetParam().channel_round)
        .Test(GetParam().kernel_fn);
    }
  }
}

TEST_P(XNN_TEST_SUITE_NAME, n_gt_channel_tile) {
  TEST_REQUIRES_ARCH_FLAGS(GetParam().arch_flags);
  const size_t n_end = (GetParam().channel_tile == 1 ? 10 : GetParam().channel_tile * 2);
  for (size_t k = 1; k < 4; k++) {
    for (size_t n = GetParam().channel_tile + 1; n < n_end; n++) {
      PackBMicrokernelTester()
        .channels(n)
        .kernel_tile(k)
        .channel_tile(GetParam().channel_tile)
        .channel_subtile(GetParam().channel_subtile)
        .channel_round(GetParam().channel_round)
        .Test(GetParam().kernel_fn);
    }
  }
}

TEST_P(XNN_TEST_SUITE_NAME, groups_gt_1) {
  TEST_REQUIRES_ARCH_FLAGS(GetParam().arch_flags);
  const size_t n_end = (GetParam().channel_tile == 1 ? 10 : GetParam().channel_tile * 2);
  for (size_t g = 2; g <= 3; g++) {
    for (size_t kernel_tile = 1; kernel_tile < 4; kernel_tile++) {
      for (size_t n = GetParam().channel_tile + 1; n < n_end; n++) {
        PackBMicrokernelTester()
          .groups(g)
          .channels(n)
          .kernel_tile(kernel_tile)
          .channel_tile(GetParam().channel_tile)
          .channel_subtile(GetParam().channel_subtile)
          .channel_round(GetParam().channel_round)
          .Test(GetParam().kernel_fn);
      }
    }
  }
}

#endif  // defined XNN_TEST_SUITE_NAME
