// Copyright 2025 Google LLC
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
#include <string>
#include <tuple>
#include <vector>

#include <gtest/gtest.h>
#include "ynnpack/base/arch.h"
#include "ynnpack/base/test/fuzz_test.h"
#include "ynnpack/base/test/random.h"
#include "ynnpack/base/test/tensor.h"
#include "ynnpack/base/test/util.h"
#include "ynnpack/base/to_string.h"
#include "ynnpack/base/type.h"
#include "ynnpack/include/ynnpack.h"
#include "ynnpack/kernels/reduce/reduce.h"

namespace ynn {

struct ReduceShape {
  size_t n;
  size_t k1;
  size_t k2;
  size_t k3;
};

std::ostream& operator<<(std::ostream& os, const ReduceShape& shape) {
  return os << "n=" << shape.n << " k={" << shape.k3 << "," << shape.k2 << ","
            << shape.k1 << "}";
}

struct KernelInfo {
  uint64_t arch_flags;
  unary_reduce_kernel_fn kernel;
  const char* name;
  ynn_reduce_operator op;
  multi_type type;
};

KernelInfo all_kernels[] = {
#define YNN_UNARY_REDUCE_KERNEL(arch_flags, name, type_a, type_c) \
  KernelInfo{arch_flags, name, #name, ynn_reduce_sum,             \
             multi_type_of(type_a(), type_c())},
#include "ynnpack/kernels/reduce/sum.inc"
#undef YNN_UNARY_REDUCE_KERNEL
};

template <typename AT, typename CT>
void TestReduce(AT, CT, ynn_reduce_operator op) {
  ReplicableRandomDevice rng;

  constexpr ReduceShape shapes[] = {
      {256, 16, 16, 1},
      {256, 1, 1, 256},
      {256, 1, 4, 256},
  };

  for (const ReduceShape& shape : shapes) {
    const size_t n = shape.n;
    const size_t k1 = shape.k1;
    const size_t k2 = shape.k2;
    const size_t k3 = shape.k3;

    const size_t k = k3 * k2 * k1;
    const float max_abs_value = type_info<CT>::max() / (k / 4);
    TypeGenerator<AT> a_gen(-max_abs_value, max_abs_value);
    TypeGenerator<CT> c_gen(-max_abs_value, max_abs_value);

    Tensor<AT> a({n, k3, k2, k1});
    Tensor<CT> init_c({n});

    a.generate([&]() { return a_gen(rng); });
    init_c.generate([&]() { return c_gen(rng); });

    Tensor<CT> c;
    const char* reference_kernel_name = nullptr;
    for (const KernelInfo& kernel : all_kernels) {
      if (kernel.op != op) {
        continue;
      }
      if (kernel.type != multi_type_of(AT(), CT())) {
        continue;
      }
      if (!is_arch_supported(kernel.arch_flags)) {
        continue;
      }

      // TODO(b/460621873): Make the remaining kernels consistent with
      // AVX/AVX512.
      if (strstr(kernel.name, "sum_fp32_avx") == nullptr) {
        continue;
      }

      Tensor<CT> kernel_c = init_c.deep_copy();

      kernel.kernel(n, k3, k2, k1, a.stride(0) * sizeof(AT),
                    a.stride(1) * sizeof(AT), a.stride(2) * sizeof(AT),
                    a.base(), 0, kernel_c.base());

      if (c.base()) {
        int finite = 0;
        for (size_t i = 0; i < n; ++i) {
          bool c_finite = std::isfinite(static_cast<float>(c(i)));
          bool kernel_c_finite = std::isfinite(static_cast<float>(kernel_c(i)));
          if (c_finite && kernel_c_finite) {
            ASSERT_EQ(c(i), kernel_c(i))
                << "Mismatch between " << reference_kernel_name << " and "
                << kernel.name << " at index " << i << " shape=" << shape;
            ++finite;
          } else {
            ASSERT_EQ(c_finite, kernel_c_finite)
                << "Mismatch between " << reference_kernel_name << " and "
                << kernel.name << " at index " << i << " shape=" << shape;
          }
        }
        // Make sure the result wasn't entirely Inf/NaN.
        ASSERT_GE(2 * finite, n);
      } else {
        c = kernel_c;
        reference_kernel_name = kernel.name;
      }
    }
  }
}

class Sum : public ::testing::TestWithParam<
                std::tuple<ynn_reduce_operator, multi_type>> {};

TEST_P(Sum, Consistent) {
  ynn_reduce_operator op = std::get<0>(GetParam());
  multi_type type = std::get<1>(GetParam());
  SwitchTwoTypes(
      type, [&](auto a_type, auto c_type) { TestReduce(a_type, c_type, op); });
}

std::set<std::tuple<ynn_reduce_operator, multi_type>> get_kernel_types() {
  std::set<std::tuple<ynn_reduce_operator, multi_type>> types;
  for (const auto& kernel : all_kernels) {
    types.insert({kernel.op, kernel.type});
  }
  return types;
}

INSTANTIATE_TEST_SUITE_P(
    Sum, Sum, testing::ValuesIn(get_kernel_types()),
    [](const testing::TestParamInfo<Sum::ParamType>& info) {
      return std::string(to_string(std::get<0>(info.param))) + "_" +
             std::string(to_string(std::get<1>(info.param)));
    });

}  // namespace ynn
