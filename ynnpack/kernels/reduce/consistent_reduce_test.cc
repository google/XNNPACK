// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

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
  reduce_kernel_fn kernel;
  const char* name;
  ynn_reduce_operator op;
  multi_type type;
  bool is_k1;
};

KernelInfo all_kernels[] = {
#define YNN_REDUCE_K1_KERNEL(arch_flags, name, type_a, type_c)                \
  KernelInfo{                                                                 \
      arch_flags, name, #name, current_op, multi_type_of(type_a(), type_c()), \
      true},
#define YNN_REDUCE_KN_KERNEL(arch_flags, name, type_a, type_c)                \
  KernelInfo{                                                                 \
      arch_flags, name, #name, current_op, multi_type_of(type_a(), type_c()), \
      false},

#define current_op ynn_reduce_sum
#include "ynnpack/kernels/reduce/sum.inc"
#undef current_op

#define current_op ynn_reduce_sum_squared
#include "ynnpack/kernels/reduce/sum_squared.inc"
#undef current_op

#define current_op ynn_reduce_min
#include "ynnpack/kernels/reduce/min.inc"
#undef current_op

#define current_op ynn_reduce_max
#include "ynnpack/kernels/reduce/max.inc"
#undef current_op

#define current_op ynn_reduce_min_max
#include "ynnpack/kernels/reduce/min_max.inc"
#undef current_op

#undef YNN_REDUCE_K1_KERNEL
#undef YNN_REDUCE_KN_KERNEL
};

template <typename AT, typename CT>
void TestReduce(AT, CT, ynn_reduce_operator op, size_t n, size_t k,
                bool is_k1) {
  ReplicableRandomDevice rng;

  // Choose the range of values in a and c such that infinite results should be
  // rare.
  const double max_abs_value = type_info<CT>::max() / k;
  const double max_abs_a_value =
      op == ynn_reduce_sum_squared ? std::sqrt(max_abs_value) : max_abs_value;

  Tensor<AT> a_max = is_k1 ? Tensor<AT>({n, k}) : Tensor<AT>({k, n});
  const size_t c_n = op == ynn_reduce_min_max ? 2 : 1;
  Tensor<CT> init_c({c_n, n});

  fill_random(a_max.data(), a_max.size(), rng, -max_abs_a_value,
              max_abs_a_value);
  fill_random(init_c.data(), init_c.size(), rng, -max_abs_value, max_abs_value);

  Tensor<AT> a = is_k1 ? a_max : a_max.transpose({1, 0});

  Tensor<CT> c;
  const char* reference_kernel_name = nullptr;
  for (const KernelInfo& kernel : all_kernels) {
    if (kernel.op != op) {
      continue;
    }
    if (kernel.type != multi_type_of(AT(), CT())) {
      continue;
    }
    if (kernel.is_k1 != is_k1) {
      continue;
    }
    if (!is_arch_supported(kernel.arch_flags)) {
      continue;
    }

    Tensor<CT> kernel_c = init_c.deep_copy();

    kernel.kernel(n, k, a.stride_bytes(is_k1 ? 0 : 1), a.base(),
                  init_c.stride_bytes(0), kernel_c.base());

    if (c.base()) {
      int finite = 0;
      for (size_t i = 0; i < c_n * n; ++i) {
        bool c_finite = std::isfinite(c[i]);
        bool kernel_c_finite = std::isfinite(kernel_c[i]);
        if (c_finite && kernel_c_finite) {
          ASSERT_EQ(c[i], kernel_c[i])
              << "Mismatch between " << reference_kernel_name << " and "
              << kernel.name << " at index " << i << " n=" << n << " k=" << k;
          ++finite;
        } else {
          ASSERT_EQ(c_finite, kernel_c_finite)
              << "Mismatch between " << reference_kernel_name << " and "
              << kernel.name << " at index " << i << " n=" << n << " k=" << k;
        }
      }
      // Make sure the result wasn't entirely Inf/NaN.
      ASSERT_GE(2 * finite, c_n * n);
    } else {
      c = kernel_c;
      reference_kernel_name = kernel.name;
    }
  }
}

class Reduce : public ::testing::TestWithParam<
                   std::tuple<ynn_reduce_operator, multi_type>> {};

TEST_P(Reduce, Consistent_k1) {
  ynn_reduce_operator op = std::get<0>(GetParam());
  multi_type type = std::get<1>(GetParam());
  SwitchTwoTypes(type, [&](auto a_type, auto c_type) {
    TestReduce(a_type, c_type, op, 4096, 256, true);
  });
}

TEST_P(Reduce, Consistent_kn) {
  ynn_reduce_operator op = std::get<0>(GetParam());
  multi_type type = std::get<1>(GetParam());
  SwitchTwoTypes(type, [&](auto a_type, auto c_type) {
    TestReduce(a_type, c_type, op, 256, 16, false);
  });
}

std::set<std::tuple<ynn_reduce_operator, multi_type>> get_kernel_types() {
  std::set<std::tuple<ynn_reduce_operator, multi_type>> types;
  for (const auto& kernel : all_kernels) {
    types.insert({kernel.op, kernel.type});
  }
  return types;
}

INSTANTIATE_TEST_SUITE_P(
    Reduce, Reduce, testing::ValuesIn(get_kernel_types()),
    [](const testing::TestParamInfo<Reduce::ParamType>& info) {
      return std::string(to_string(std::get<0>(info.param))) + "_" +
             std::string(to_string(std::get<1>(info.param)));
    });

}  // namespace ynn
