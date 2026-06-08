// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <cassert>
#include <cstdint>
#include <cstring>
#include <set>
#include <string>
#include <tuple>
#include <vector>

#include <gtest/gtest.h>
#include "ynnpack/base/arch.h"
#include "ynnpack/base/arithmetic.h"
#include "ynnpack/base/base.h"
#include "ynnpack/base/bfloat16.h"
#include "ynnpack/base/half.h"
#include "ynnpack/base/test/fuzz_test.h"
#include "ynnpack/base/test/random.h"
#include "ynnpack/base/test/tensor.h"
#include "ynnpack/base/to_string.h"
#include "ynnpack/base/type.h"
#include "ynnpack/include/ynnpack.h"
#include "ynnpack/kernels/unary/unary.h"

namespace ynn {
namespace {

template <typename F>
constexpr decltype(auto) SwitchType(ynn_type type, F&& f) {
  switch (type) {
    case ynn_type_int32:
      return std::forward<F>(f)(int32_t());
    case ynn_type_int8:
      return std::forward<F>(f)(int8_t());
    case ynn_type_uint8:
      return std::forward<F>(f)(uint8_t());
    case ynn_type_fp16:
      return std::forward<F>(f)(half());
    case ynn_type_bf16:
      return std::forward<F>(f)(bfloat16());
    case ynn_type_fp32:
      return std::forward<F>(f)(float());
    case ynn_type_fp64:
      return std::forward<F>(f)(double());
    case ynn_type_int4:
      return std::forward<F>(f)(int4x2());
    case ynn_type_int2:
      return std::forward<F>(f)(int2x4());
    default:
      YNN_UNREACHABLE;
  }
}

struct KernelInfo {
  uint64_t arch_flags;
  unary_kernel_fn kernel;
  const char* name;
  ynn_unary_operator op;
  uint32_t flags;
  ynn_type a_type;
  ynn_type x_type;
};

std::vector<KernelInfo> get_all_kernels() {
  std::vector<KernelInfo> kernels;
#define YNN_ELEMENTWISE_KERNEL(arch_val, name, op_type, flags_val, type_a,  \
                               type_x)                                      \
  kernels.push_back({arch_val, name, #name, ynn_unary_##op_type, flags_val, \
                     type_of<type_a>(), type_of<type_x>()});
#include "ynnpack/kernels/unary/kernels.inc"
#undef YNN_ELEMENTWISE_KERNEL

  std::set<ynn_unary_operator> ops;
  for (const KernelInfo& kernel : kernels) {
    ops.insert(kernel.op);
  }

  return kernels;
}

template <typename AT, typename XT>
void TestUnary(AT, XT, ynn_unary_operator op, size_t n) {
  ReplicableRandomDevice rng;

  Tensor<AT> a({n});

  // Most kernels work fine with random values.
  fill_random(a.data(), a.size(), rng);

  const auto all_kernels = get_all_kernels();
  const char* reference_kernel_name = nullptr;
  Tensor<XT> reference_x;

  // Add the reference kernel to the set of kernels to test if it is consistent.
  struct TestKernelInfo {
    unary_kernel_fn kernel;
    const char* name;
  };
  std::vector<TestKernelInfo> kernels_to_test;
  if (unary_kernel_fn ref_kernel = get_unary_reference_kernel(
          op, type_of<XT>(), unary_flag::consistent_arithmetic)) {
    kernels_to_test.push_back({ref_kernel, "reference"});
  }

  for (const KernelInfo& kernel : all_kernels) {
    if (kernel.op != op) continue;
    if (!is_arch_supported(kernel.arch_flags)) continue;
    if (kernel.a_type != type_of<AT>() || kernel.x_type != type_of<XT>()) {
      continue;
    }
    if (!(kernel.flags & unary_flag::consistent_arithmetic)) continue;
    kernels_to_test.push_back({kernel.kernel, kernel.name});
  }

  for (const auto& kernel : kernels_to_test) {
    Tensor<XT> kernel_x({n});
    unary_params params = get_unary_params(op);
    // For poly3, set some non-zero coefficients.
    if (op == ynn_unary_poly3) {
      params.poly3.c0 = 1.0f;
      params.poly3.c1 = 0.5f;
      params.poly3.c2 = 0.25f;
      params.poly3.c3 = 0.125f;
    }

    kernel.kernel(1, n, 0, a.base(), 0, kernel_x.base(), &params);

    if (reference_x.base()) {
      for (size_t i = 0; i < n; ++i) {
        if (isnan(reference_x[i])) {
          ASSERT_TRUE(isnan(kernel_x[i]))
              << "kernel `" << kernel.name
              << "` is inconsistent with reference kernel `"
              << reference_kernel_name << "`";
        } else {
          ASSERT_EQ(reference_x[i], kernel_x[i])
              << "a[i]=" << a[i] << ", kernel `" << kernel.name
              << "` is inconsistent with reference kernel `"
              << reference_kernel_name << "`";
        }
      }
    } else {
      reference_x = kernel_x.deep_copy();
      reference_kernel_name = kernel.name;
    }
  }
}

class ConsistentUnary
    : public ::testing::TestWithParam<
          std::tuple<ynn_unary_operator, ynn_type, ynn_type>> {};

TEST_P(ConsistentUnary, Consistent) {
  ynn_unary_operator op = std::get<0>(GetParam());
  ynn_type a_type = std::get<1>(GetParam());
  ynn_type x_type = std::get<2>(GetParam());
  SwitchType(a_type, [&](auto a_t) {
    SwitchType(x_type, [&](auto x_t) { TestUnary(a_t, x_t, op, 1024 * 64); });
  });
}

std::set<std::tuple<ynn_unary_operator, ynn_type, ynn_type>>
get_consistent_kernel_types() {
  std::set<std::tuple<ynn_unary_operator, ynn_type, ynn_type>> types;
  for (const auto& kernel : get_all_kernels()) {
    if (kernel.flags & unary_flag::consistent_arithmetic) {
      types.insert({kernel.op, kernel.a_type, kernel.x_type});
    }
  }
  return types;
}

INSTANTIATE_TEST_SUITE_P(
    Unary, ConsistentUnary, testing::ValuesIn(get_consistent_kernel_types()),
    [](const testing::TestParamInfo<ConsistentUnary::ParamType>& info) {
      return std::string(to_string(std::get<0>(info.param))) + "_" +
             std::string(to_string(std::get<1>(info.param))) + "_" +
             std::string(to_string(std::get<2>(info.param)));
    });

#if !defined(YNN_ARCH_X86) && !defined(YNN_ARCH_ARM)
GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(ConsistentUnary);
#endif  // !defined(YNN_ARCH_X86) && !defined(YNN_ARCH_ARM)

}  // namespace
}  // namespace ynn
