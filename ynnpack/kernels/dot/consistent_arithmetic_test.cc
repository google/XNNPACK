// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <ostream>
#include <set>

#include <gtest/gtest.h>
#include "ynnpack/base/arch.h"  // IWYU pragma: keep
#include "ynnpack/base/test/buffer.h"
#include "ynnpack/base/test/fuzz_test.h"
#include "ynnpack/base/test/random.h"
#include "ynnpack/base/test/tensor.h"
#include "ynnpack/base/test/util.h"
#include "ynnpack/base/type.h"
#include "ynnpack/kernels/dot/dot.h"
#include "ynnpack/kernels/dot/pack_test_tensor.h"

namespace ynn {

struct DotShape {
  size_t m, n, k;

  DotShape with_m(size_t m) const { return DotShape{m, n, k}; }
  DotShape with_n(size_t n) const { return DotShape{m, n, k}; }
  DotShape with_k(size_t k) const { return DotShape{m, n, k}; }
};

std::ostream& operator<<(std::ostream& os, const DotShape& shape) {
  return os << shape.m << "x" << shape.n << "x" << shape.k;
}

struct KernelInfo {
  uint64_t arch_flags;
  dot_kernel_fn kernel;
  const char* name;
  DotShape block_shape;
  size_t tile_m;
  size_t tile_n;
  size_t tile_k;
  uint32_t flags;
  multi_type type;
};

KernelInfo all_kernels[] = {
#define YNN_DOT_KERNEL(arch_flags, name, block_m, block_n, block_k, tile_m, \
                       tile_n, tile_k, flags, a_type, b_type, c_type)       \
  KernelInfo{arch_flags,                                                    \
             name,                                                          \
             #name,                                                         \
             {block_m, block_n, block_k},                                   \
             tile_m,                                                        \
             tile_n,                                                        \
             tile_k,                                                        \
             flags,                                                         \
             multi_type_of(a_type(), b_type(), c_type())},
#include "ynnpack/kernels/dot/kernels.inc"
#undef YNN_DOT_KERNEL
};

// Get the alignment that satisfies the requirement of any dot kernel.
size_t get_max_alignment() {
  size_t result = 1;
  for (const KernelInfo& kernel : all_kernels) {
    result = std::max(result, kernel.tile_n * kernel.tile_k);
  }
  return result;
}

template <typename AT, typename BT, typename CT>
void TestMatMul(AT, BT, CT, size_t k) {
  using B_info = type_info<BT>;

  ReplicableRandomDevice rng;
  // We want a large range, but not so large that our outputs are likely to be
  // Inf/NaN.
  const float max_abs_value = std::sqrt(type_info<CT>::max()) / (k / 4);
  TypeGenerator<AT> a_gen(-max_abs_value, max_abs_value);
  TypeGenerator<BT> b_gen(-max_abs_value, max_abs_value);
  TypeGenerator<CT> c_gen(-max_abs_value, max_abs_value);

  // The consistency of a kernel is mostly an issue for:
  // - The reduction order
  // - Whether fma is used or not
  // So we should be able to just use one shape that works for every kernel.
  const size_t m = 1;
  const size_t n = 4096;

  Tensor<AT> a({m, k});
  Tensor<BT> b({k, n / B_info::element_count()},
               Alignment({.bytes = get_max_alignment()}));
  Tensor<CT> init_c({m, n});
  a.generate([&]() { return a_gen(rng); });
  b.generate([&]() { return b_gen(rng); });
  init_c.generate([&]() { return c_gen(rng); });

  Tensor<CT> c;
  int consistent_kernels = 0;
  for (const KernelInfo& kernel : all_kernels) {
    if (kernel.type != multi_type_of(AT(), BT(), CT())) {
      continue;
    }
    if (!is_arch_supported(kernel.arch_flags)) {
      std::cout << "Skipping unsupported kernel " << kernel.name << std::endl;
      continue;
    }
    if (!(kernel.flags & dot_flag::consistent_arithmetic)) {
      std::cout << "Skipping inconsistent arithmetic kernel " << kernel.name
                << std::endl;
      continue;
    }
    std::cout << "Considering kernel " << kernel.name << std::endl;
    ++consistent_kernels;

    const size_t tile_m = kernel.tile_m;
    const size_t tile_n = kernel.tile_n;
    const size_t tile_k = kernel.tile_k;

    Tensor<CT> kernel_c = init_c.deep_copy();

    // dot kernels require B's k and n dimensions to be aligned to tile_k,
    // tile_n. The kernel might also require b to be packed (tile_k > 1).
    Tensor<BT> packed_b = pack_b(b, tile_k, tile_n);
    Tensor<AT> packed_a = (kernel.flags & dot_flag::transpose_a)
                              ? transpose_a(a, tile_m, tile_k)
                              : a;

    kernel.kernel(m, n, 1, 1, k, packed_a.stride(0) * sizeof(AT), 0, 0,
                  packed_a.base(), 0, 0,
                  packed_b.stride(0) * sizeof(BT) / tile_k, packed_b.base(),
                  kernel_c.stride(0) * sizeof(CT), kernel_c.base(),
                  kernel_c.stride(0) * sizeof(CT), kernel_c.base());

    if (c.base()) {
      int finite = 0;
      for (const auto& i : EnumerateIndices({m, n})) {
        bool c_finite = std::isfinite(static_cast<float>(c(i)));
        bool kernel_c_finite = std::isfinite(static_cast<float>(kernel_c(i)));
        if (c_finite && kernel_c_finite) {
          ASSERT_EQ(c(i), kernel_c(i));
          finite++;
        } else {
          ASSERT_EQ(c_finite, kernel_c_finite);
        }
      }
      // Make sure the result wasn't entirely Inf/NaN.
      ASSERT_GE(finite * 2, m * n);
    } else {
      c = kernel_c;
    }
  }
  ASSERT_GT(consistent_kernels, 0) << "No consistent_arithmetic kernels found.";
}

const char* to_string(const KernelInfo& param) { return ""; }

class Dot : public ::testing::TestWithParam<multi_type> {};

TEST_P(Dot, Consistent) {
  multi_type type = GetParam();
  SwitchThreeTypes(type, [&](auto a_type, auto b_type, auto c_type) {
    TestMatMul(a_type, b_type, c_type, 256);
  });
}

std::set<multi_type> get_kernel_types() {
  std::set<multi_type> types;
  for (const auto& kernel : all_kernels) {
    types.insert(kernel.type);
  }
  return types;
}

INSTANTIATE_TEST_SUITE_P(
    Dot, Dot, testing::ValuesIn(get_kernel_types()),
    [](const testing::TestParamInfo<Dot::ParamType>& info) {
      return to_string(info.param);
    });

}  // namespace ynn
