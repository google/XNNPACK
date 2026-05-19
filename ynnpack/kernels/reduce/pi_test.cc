// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <cstddef>
#include <cstdint>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "ynnpack/base/arch.h"
#include "ynnpack/base/type.h"
#include "ynnpack/kernels/reduce/reduce.h"

namespace ynn {
namespace {

template <typename AT, typename CT>
void pi_sum(uint64_t arch_flags, reduce_kernel_fn kernel) {
  if (!is_arch_supported(arch_flags)) {
    GTEST_SKIP();
    return;
  }

  // sum_{i=0}^n 0.5/((i + 0.25)*(i + 0.75)) converges to pi. This is some of
  // the partial sums as computed by Mathematica.
  const std::pair<size_t, double> expected_sum[] = {
      {10, 3.09162380666783863168},
      {100, 3.13659268483881675041},
      {1000, 3.14109265362104322869},
      {10000, 3.14154265358982448846},
      {100000, 3.14158765358979326971},
      {1000000, 3.14159215358979323849},
  };

  const double tolerance = std::max<double>(type_info<AT>::epsilon(),
                                 type_info<CT>::epsilon() * 8);

  for (const auto& [n, expected] : expected_sum) {
    std::vector<AT> terms(n);
    for (size_t i = 0; i < n; ++i) {
      terms[i] = static_cast<AT>(0.5 / ((i + 0.25) * (i + 0.75)));
    }

    CT result = 0;
    kernel(1, n, sizeof(AT), terms.data(), &result, nullptr);

    EXPECT_NEAR(result, expected, tolerance);
  }
}

template <typename AT, typename CT>
void pi_sum_squared(uint64_t arch_flags, reduce_kernel_fn kernel) {
  if (!is_arch_supported(arch_flags)) {
    GTEST_SKIP();
    return;
  }

  // sum_{i=1}^n 1/i^2 converges to pi^2/6. This is some of the partial sums as
  // computed by Mathematica.
  const std::pair<size_t, double> expected_sum[] = {
      {10, 1.54976773116654069035},
      {100, 1.63498390018489286508},
      {1000, 1.64393456668155980314},
      {10000, 1.64483407184805976981},
      {100000, 1.64492406689822626981},
      {1000000, 1.64493306684872643631},
  };

  const double tolerance =
      std::max<double>(type_info<AT>::epsilon(), type_info<CT>::epsilon() * 4);

  for (const auto& [n, expected] : expected_sum) {
    std::vector<AT> terms(n);
    for (size_t i = 0; i < n; ++i) {
      terms[i] = static_cast<AT>(1.0 / (i + 1));
    }

    CT result = 0;
    kernel(1, n, sizeof(AT), terms.data(), &result, nullptr);

    EXPECT_NEAR(result, expected, tolerance);
  }
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
