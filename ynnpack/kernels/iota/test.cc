// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "ynnpack/base/arch.h"
#include "ynnpack/base/test/util.h"
#include "ynnpack/kernels/iota/iota.h"

namespace ynn {
namespace {

template <typename T>
void run_iota_test(uint64_t arch_flags, iota_kernel_fn kernel, size_t n,
                   T begin, T stride) {
  if (!is_arch_supported(arch_flags)) {
    GTEST_SKIP();
  }

  std::vector<T> output(n);
  kernel(n, &begin, &stride, output.data());

  std::vector<T> expected(n);
  for (size_t i = 0; i < n; ++i) {
    expected[i] = begin + stride * static_cast<T>(i);
  }

  EXPECT_THAT(output, testing::ElementsAreArray(expected));
}

class IotaTest : public testing::TestWithParam<size_t> {};

#define YNN_IOTA_KERNEL(arch, name, type)                           \
  using name##_test = IotaTest;                                     \
  TEST_P(name##_test, test) {                                       \
    run_iota_test<type>(arch, name, GetParam(), (type)10, (type)3); \
  }                                                                 \
  INSTANTIATE_TEST_SUITE_P(name, name##_test,                       \
                           testing::ValuesIn(simd_sizes_up_to(1024)));

#include "ynnpack/kernels/iota/kernels.inc"
#undef YNN_IOTA_KERNEL

}  // namespace
}  // namespace ynn
