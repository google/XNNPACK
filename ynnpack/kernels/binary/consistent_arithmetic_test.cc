// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <cstddef>
#include <cstdint>
#include <set>
#include <string>
#include <tuple>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "ynnpack/base/arch.h"
#include "ynnpack/base/arithmetic.h"
#include "ynnpack/base/base.h"
#include "ynnpack/base/bfloat16.h"
#include "ynnpack/base/fp8.h"
#include "ynnpack/base/half.h"
#include "ynnpack/base/test/fuzz_test.h"
#include "ynnpack/base/test/random.h"
#include "ynnpack/base/test/tensor.h"
#include "ynnpack/base/to_string.h"
#include "ynnpack/base/type.h"
#include "ynnpack/include/ynnpack.h"
#include "ynnpack/kernels/binary/binary.h"

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
    case ynn_type_fp8_e5m2:
      return std::forward<F>(f)(fp8_e5m2());
    case ynn_type_fp8_e4m3:
      return std::forward<F>(f)(fp8_e4m3());
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
  binary_kernel_fn kernel;
  const char* name;
  ynn_binary_operator op;
  uint32_t flags;
  ynn_type a_type;
  ynn_type b_type;
  ynn_type x_type;
};

std::vector<KernelInfo> get_all_kernels() {
  std::vector<KernelInfo> kernels;
#define YNN_ELEMENTWISE_KERNEL(arch_val, name, op_type, flags_val, type_a,   \
                               type_b, type_x)                               \
  kernels.push_back({arch_val, name, #name, ynn_binary_##op_type, flags_val, \
                     type_of<type_a>(), type_of<type_b>(),                   \
                     type_of<type_x>()});
#include "ynnpack/kernels/binary/kernels.inc"
#undef YNN_ELEMENTWISE_KERNEL
  return kernels;
}

template <typename AT, typename BT, typename XT>
void TestBinary(AT, BT, XT, ynn_binary_operator op, size_t n) {
  ReplicableRandomDevice rng;

  Tensor<AT> a({1, n});
  Tensor<BT> b({1, n});

  fill_random(a.data(), a.size(), rng);
  fill_random(b.data(), b.size(), rng);

  const auto all_kernels = get_all_kernels();
  const char* reference_kernel_name = nullptr;
  Tensor<XT> reference_x;

  struct TestKernelInfo {
    binary_kernel_fn kernel;
    const char* name;
  };
  std::vector<TestKernelInfo> kernels_to_test;
  if (type_of<AT>() == type_of<XT>() && type_of<BT>() == type_of<XT>()) {
    if (binary_kernel_fn ref_kernel = get_binary_reference_kernel(
            op, type_of<XT>(), binary_flag::consistent_arithmetic)) {
      kernels_to_test.push_back({ref_kernel, "reference"});
    }
  }

  for (const KernelInfo& kernel : all_kernels) {
    if (kernel.op != op) continue;
    if (!is_arch_supported(kernel.arch_flags)) continue;
    if (kernel.a_type != type_of<AT>() || kernel.b_type != type_of<BT>() ||
        kernel.x_type != type_of<XT>()) {
      continue;
    }
    if (!(kernel.flags & binary_flag::consistent_arithmetic)) continue;
    kernels_to_test.push_back({kernel.kernel, kernel.name});
  }

  for (const auto& kernel : kernels_to_test) {
    Tensor<XT> kernel_x({1, n});

    kernel.kernel(1, n, 0, a.stride_bytes(1), a.base(), 0, b.stride_bytes(1),
                  b.base(), 0, kernel_x.base(), nullptr);

    if (reference_x.base()) {
      for (size_t i = 0; i < n; ++i) {
        if (isnan(reference_x(0, i))) {
          ASSERT_TRUE(isnan(kernel_x(0, i)))
              << "kernel `" << kernel.name
              << "` is inconsistent with reference kernel `"
              << reference_kernel_name << "`";
        } else {
          ASSERT_EQ(reference_x(0, i), kernel_x(0, i))
              << "a[i]=" << a(0, i) << ", b[i]=" << b(0, i) << ", kernel `"
              << kernel.name << "` is inconsistent with reference kernel `"
              << reference_kernel_name << "`";
        }
      }
    } else {
      reference_x = kernel_x.deep_copy();
      reference_kernel_name = kernel.name;
    }
  }
}

class ConsistentBinary
    : public ::testing::TestWithParam<
          std::tuple<ynn_binary_operator, ynn_type, ynn_type, ynn_type>> {};

TEST_P(ConsistentBinary, Consistent) {
  ynn_binary_operator op = std::get<0>(GetParam());
  ynn_type a_type = std::get<1>(GetParam());
  ynn_type b_type = std::get<2>(GetParam());
  ynn_type x_type = std::get<3>(GetParam());
  constexpr size_t n = 4096;
  if (a_type == b_type && b_type == x_type) {
    SwitchType(x_type, [&](auto t) { TestBinary(t, t, t, op, n); });
  } else if (a_type == ynn_type_bf16 && b_type == ynn_type_fp32 &&
             x_type == ynn_type_fp32) {
    TestBinary(bfloat16(), float(), float(), op, n);
  } else if (a_type == ynn_type_bf16 && b_type == ynn_type_fp32 &&
             x_type == ynn_type_bf16) {
    TestBinary(bfloat16(), float(), bfloat16(), op, n);
  } else if (a_type == ynn_type_fp32 && b_type == ynn_type_bf16 &&
             x_type == ynn_type_bf16) {
    TestBinary(float(), bfloat16(), bfloat16(), op, n);
  } else if (a_type == ynn_type_int32 && b_type == ynn_type_fp32 &&
             x_type == ynn_type_fp32) {
    TestBinary(int32_t(), float(), float(), op, n);
  } else {
    FAIL() << "Unsupported type combination: " << a_type << ", " << b_type
           << ", " << x_type;
  }
}

std::set<std::tuple<ynn_binary_operator, ynn_type, ynn_type, ynn_type>>
get_consistent_kernel_types() {
  std::set<std::tuple<ynn_binary_operator, ynn_type, ynn_type, ynn_type>> types;
  for (const auto& kernel : get_all_kernels()) {
    if (kernel.flags & binary_flag::consistent_arithmetic) {
      types.insert({kernel.op, kernel.a_type, kernel.b_type, kernel.x_type});
    }
  }
  return types;
}

INSTANTIATE_TEST_SUITE_P(
    Binary, ConsistentBinary, testing::ValuesIn(get_consistent_kernel_types()),
    [](const testing::TestParamInfo<ConsistentBinary::ParamType>& info) {
      return std::string(to_string(std::get<0>(info.param))) + "_" +
             std::string(to_string(std::get<1>(info.param))) + "_" +
             std::string(to_string(std::get<2>(info.param))) + "_" +
             std::string(to_string(std::get<3>(info.param)));
    });

GTEST_ALLOW_UNINSTANTIATED_PARAMETERIZED_TEST(ConsistentBinary);

}  // namespace
}  // namespace ynn
