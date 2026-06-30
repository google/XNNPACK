// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <tuple>
#include <utility>

#include <gtest/gtest.h>
#include "ynnpack/base/algorithm.h"
#include "ynnpack/base/arch.h"
#include "ynnpack/base/test/buffer.h"
#include "ynnpack/base/test/util.h"
#include "ynnpack/base/to_string.h"
#include "ynnpack/base/type.h"
#include "ynnpack/include/ynnpack.h"
#include "ynnpack/kernels/lut/lut.h"

using testing::Combine;
using testing::Values;
using testing::ValuesIn;
using ynn::to_string;  // NOLINT(misc-unused-using-decls)

namespace ynn {

struct KernelInfo {
  uint64_t arch_flags;
  lut_kernel_fn kernel;
};

const char* to_string(const KernelInfo& info) { return ""; }

template <typename Idx, typename Out>
void TestImpl(lut_kernel_fn kernel, size_t n, size_t lut_size) {
  Buffer<Idx> idx(n);
  Buffer<Out> lut(lut_size);
  Buffer<Out> out(n);

  std::iota(lut.begin(), lut.end(), 0);
  std::reverse(lut.begin(), lut.end());
  for (size_t i = 0; i < n; ++i) {
    using Element = typename type_info<Idx>::element_type;
    idx[i] = static_cast<Element>(i);
  }

  bool result = kernel(n, idx.data(), lut_size, lut.data(), out.data());

  if (any_n(n, [&](size_t i) { return idx[i] < 0 || idx[i] >= lut_size; })) {
    ASSERT_FALSE(result);
  } else {
    ASSERT_TRUE(result);
    for (size_t i = 0; i < n; ++i) {
      ASSERT_EQ(out[i], lut[idx[i]]);
    }
  }
}

template <typename Idx>
void TestImpl(Idx, size_t elem_size_bits, lut_kernel_fn kernel, size_t n,
              size_t lut_size) {
  switch (elem_size_bits) {
    case 8:
      TestImpl<Idx, uint8_t>(kernel, n, lut_size);
      break;
    case 16:
      TestImpl<Idx, uint16_t>(kernel, n, lut_size);
      break;
    case 32:
      TestImpl<Idx, uint32_t>(kernel, n, lut_size);
      break;
    default:
      FAIL() << "Unsupported element size: " << elem_size_bits;
  }
}

template <typename F>
constexpr decltype(auto) SwitchLutIdxType(ynn_type type, F&& f) {
  switch (type) {
    case ynn_type_int8:
      return std::forward<F>(f)(int8_t());
    case ynn_type_uint8:
      return std::forward<F>(f)(uint8_t());
    case ynn_type_int32:
      return std::forward<F>(f)(int32_t());
    case ynn_type_int2:
      return std::forward<F>(f)(int2x4());
    case ynn_type_uint2:
      return std::forward<F>(f)(uint2x4());
    case ynn_type_int4:
      return std::forward<F>(f)(int4x2());
    case ynn_type_uint4:
      return std::forward<F>(f)(uint4x2());
    default:
      YNN_UNREACHABLE;
  }
}

class LutTest : public testing::TestWithParam<
                    std::tuple<ynn_type, size_t, KernelInfo, size_t, size_t>> {
};

TEST_P(LutTest, kernel) {
  KernelInfo kernel_info = std::get<2>(GetParam());
  if (!is_arch_supported(kernel_info.arch_flags)) {
    GTEST_SKIP();
  }
  ynn_type type = std::get<0>(GetParam());
  size_t elem_size_bits = std::get<1>(GetParam());
  lut_kernel_fn kernel = kernel_info.kernel;
  size_t n = std::get<3>(GetParam());
  size_t lut_size = std::get<4>(GetParam());
  SwitchLutIdxType(type, [&](auto idx) {
    TestImpl(idx, elem_size_bits, kernel, n, lut_size);
  });
}

#define YNN_LUT_KERNEL(arch_flags, kernel, idx_type, elem_size_bits) \
  INSTANTIATE_TEST_SUITE_P(                                          \
      kernel, LutTest,                                               \
      Combine(Values(type_of<idx_type>()), Values(elem_size_bits),   \
              Values(KernelInfo{arch_flags, kernel}),                \
              ValuesIn(simd_sizes_up_to(2048)),                      \
              Values(3, 4, 15, 16, 255, 256)),                       \
      test_param_to_string<LutTest::ParamType>);
#include "ynnpack/kernels/lut/kernels.inc"
#undef YNN_LUT_KERNEL

}  // namespace ynn
