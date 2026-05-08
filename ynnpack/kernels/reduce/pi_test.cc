// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

#include <gtest/gtest.h>
#include "ynnpack/base/arch.h"
#include "ynnpack/base/type.h"
#include "ynnpack/kernels/reduce/reduce.h"

namespace ynn {
namespace {

template <typename AT, typename CT>
void TestPi(reduce_kernel_fn kernel, bool is_k1) {
  if (!std::is_floating_point_v<AT>) {
    GTEST_SKIP();
  }

  const size_t n = 1000000;
  std::vector<AT> terms(n);

  // pi = sum_{i=0}^{n} a/((i+b)*(i + c))
  const double a = 0.5;
  const double b = 0.25;
  const double c = 0.75;
  for (size_t i = 0; i < n; ++i) {
    terms[i] = static_cast<AT>(a / ((i + b) * (i + c)));
  }

  CT result = 0;
  if (is_k1) {
    kernel(1, n, 0, terms.data(), 0, &result);
  } else {
    kernel(1, n, sizeof(AT), terms.data(), 0, &result);
  }

  const double pi = 3.14159265358979323846;
  EXPECT_NEAR(result, pi, 1e-6);
}

#define TEST_PI_KERNEL(arch, name, type_a, type_c, is_k1) \
  TEST(PiSummation, name) {                               \
    if (!is_arch_supported(arch)) GTEST_SKIP();           \
    TestPi<type_a, type_c>(name, is_k1);                  \
  }

#undef YNN_REDUCE_K1_KERNEL
#define YNN_REDUCE_K1_KERNEL(arch_flags, name, type_a, type_c) \
  TEST_PI_KERNEL(arch_flags, name, type_a, type_c, true)

#undef YNN_REDUCE_KN_KERNEL
#define YNN_REDUCE_KN_KERNEL(arch_flags, name, type_a, type_c) \
  TEST_PI_KERNEL(arch_flags, name, type_a, type_c, false)

#include "ynnpack/kernels/reduce/sum.inc"

}  // namespace
}  // namespace ynn
