// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <tuple>

#include <gtest/gtest.h>
#include "ynnpack/base/test/buffer.h"
#include "ynnpack/base/test/util.h"
#include "ynnpack/base/to_string.h"
#include "ynnpack/base/type.h"
#include "ynnpack/kernels/lut/lut.h"

using testing::Combine;
using testing::Values;
using testing::ValuesIn;
using ynn::to_string;  // NOLINT(misc-unused-using-decls)

namespace ynn {

struct KernelInfo {
  lut_kernel_fn kernel;
};

const char* to_string(const KernelInfo& info) { return ""; }

template <typename A, typename X>
void TestImpl(A, X, lut_kernel_fn kernel, size_t n) {
  Buffer<A> a(n);
  Buffer<X> lut(1 << (sizeof(A) * 8));
  Buffer<X> x(n);

  std::iota(lut.begin(), lut.end(), 0);
  std::reverse(lut.begin(), lut.end());
  std::iota(a.begin(), a.end(), 0);

  kernel(n, a.data(), lut.data(), x.data());

  for (size_t i = 0; i < n; ++i) {
    ASSERT_EQ(x[i], lut[a[i]]);
  }
}

// Don't try to instantiate LUT tests for large a types.
template <typename F>
constexpr decltype(auto) SwitchLutTypes(multi_type type, F&& f) {
  switch (type) {
    case multi_type::int8:
      return std::forward<F>(f)(int8_t(), int8_t());
    case multi_type::uint8:
      return std::forward<F>(f)(uint8_t(), uint8_t());
    case multi_type::int8_int32:
      return std::forward<F>(f)(int8_t(), int32_t());
    case multi_type::uint8_int32:
      return std::forward<F>(f)(uint8_t(), int32_t());
    default:
      YNN_UNREACHABLE;
  }
}

class LutTest : public testing::TestWithParam<
                    std::tuple<multi_type, KernelInfo, size_t>> {};

TEST_P(LutTest, kernel) {
  multi_type type = std::get<0>(GetParam());
  lut_kernel_fn kernel = std::get<1>(GetParam()).kernel;
  size_t n = std::get<2>(GetParam());
  SwitchLutTypes(type, [&](auto type_a, auto type_x) {
    TestImpl(type_a, type_x, kernel, n);
  });
}

#define YNN_LUT_KERNEL(arch_flags, kernel, type_a, type_x)                   \
  INSTANTIATE_TEST_SUITE_P(                                                  \
      kernel, LutTest,                                                       \
      Combine(Values(multi_type_of(type_a(), type_x())),                     \
              Values(KernelInfo{kernel}), ValuesIn(simd_sizes_up_to(2048))), \
      test_param_to_string<LutTest::ParamType>);
#include "ynnpack/kernels/lut/kernels.inc"
#undef YNN_ELEMENTWISE_KERNEL

}  // namespace ynn
