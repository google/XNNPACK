// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <cstddef>
#include <cstdint>
#include <vector>

#include <gtest/gtest.h>
#include "ynnpack/base/arch.h"
#include "ynnpack/base/type.h"
#include "ynnpack/kernels/reduce/reduce.h"

namespace ynn {
namespace {

const double pi = 3.14159265358979323846;

template <typename AT, typename CT>
void pi_sum(uint64_t arch_flags, reduce_kernel_fn kernel) {
  if (!is_arch_supported(arch_flags)) {
    GTEST_SKIP();
    return;
  }

  // sum_{i=0}^n 0.5/((i + 0.25)*(i + 0.75)) converges to pi
  const size_t n = 1000000;
  std::vector<AT> terms(n);
  for (size_t i = 0; i < n; ++i) {
    terms[i] = static_cast<AT>(0.5 / ((i + 0.25) * (i + 0.75)));
  }

  CT result = 0;
  kernel(1, n, sizeof(AT), terms.data(), &result, nullptr);

  // Getting this sum to within a tight tolerance of pi requires good numerical
  // behavior from the kernel.
  const CT expected = static_cast<CT>(pi);
  EXPECT_NEAR(result, expected,
              std::max<double>(1e-6, type_info<AT>::epsilon()));
}

template <typename AT, typename CT>
void pi_sum_squared(uint64_t arch_flags, reduce_kernel_fn kernel) {
  if (!is_arch_supported(arch_flags)) {
    GTEST_SKIP();
    return;
  }

  // sum_{i=1}^n 1/i^2 converges to pi^2/6
  const size_t n = 1000000;
  std::vector<AT> terms(n);
  for (size_t i = 0; i < n; ++i) {
    terms[i] = static_cast<AT>(1.0 / (i + 1));
  }

  CT result = 0;
  kernel(1, n, sizeof(AT), terms.data(), &result, nullptr);

  // Getting this sum to within a tight tolerance of pi requires good numerical
  // behavior from the kernel.
  const CT expected = static_cast<CT>(pi * pi / 6.0);
  EXPECT_NEAR(result, expected,
              std::max<double>(2e-6, type_info<AT>::epsilon()));
}

#define YNN_REDUCE_KERNEL(arch_flags, name, k_dim, type_a, type_c) \
  TEST(pi_sum, name) { pi_sum<type_a, type_c>(arch_flags, name); }
#include "ynnpack/kernels/reduce/sum.inc"
#undef YNN_REDUCE_KERNEL

#define YNN_REDUCE_KERNEL(arch_flags, name, k_dim, type_a, type_c) \
  TEST(pi_sum_squared, name) {                                     \
    pi_sum_squared<type_a, type_c>(arch_flags, name);              \
  }
#include "ynnpack/kernels/reduce/sum_squared.inc"
#undef YNN_REDUCE_KERNEL

}  // namespace
}  // namespace ynn
