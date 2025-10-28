// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <type_traits>
#include <vector>

#include <gtest/gtest.h>
#include "ynnpack/base/arch.h"  // IWYU pragma: keep
#include "ynnpack/base/arithmetic.h"
#include "ynnpack/base/base.h"
#include "ynnpack/base/bfloat16.h"
#include "ynnpack/base/test/buffer.h"
#include "ynnpack/base/test/fuzz_test.h"
#include "ynnpack/base/test/random.h"
#include "ynnpack/base/test/tensor.h"
#include "ynnpack/base/test/util.h"
#include "ynnpack/base/type.h"
#include "ynnpack/kernels/reduce/reduce.h"

namespace ynn {

enum class ReduceOp {
  kSum,
  kSumSquared,
  kMin,
  kMax,
  kMinMax,
};

size_t OutputRows(ReduceOp op) {
  switch (op) {
    case ReduceOp::kSum:
    case ReduceOp::kMin:
    case ReduceOp::kMax:
    case ReduceOp::kSumSquared:
      return 1;
    case ReduceOp::kMinMax:
      return 2;
  }
  YNN_UNREACHABLE;
  return 0;
}

template <typename T>
float Tolerance(ReduceOp op, size_t k, float max_abs_value) {
  switch (op) {
    case ReduceOp::kSum:
      return epsilon(type_of<T>()) * k * max_abs_value * 3;
    case ReduceOp::kSumSquared:
      return epsilon(type_of<T>()) * k * max_abs_value * max_abs_value * 6;
    case ReduceOp::kMin:
    case ReduceOp::kMax:
    case ReduceOp::kMinMax:
      return 0.0f;
  }
  YNN_UNREACHABLE;
  return 0.0f;
}

template <typename AT, typename CT>
void Reference(Tensor<AT> a, Tensor<CT> c, ReduceOp op) {
  // This helper allows omitting 2 of the 3 k dimensions. Canonicalize to 3 k
  // dimensions here.
  while (a.rank() < 4) {
    a = a.expand_dims({1});
  }

  ASSERT_EQ(a.extent(0), c.extent(1));
  size_t K3 = a.extent(1);
  size_t K2 = a.extent(2);
  size_t K1 = a.extent(3);
  size_t N = c.extent(1);
  for (size_t k3 = 0; k3 < K3; ++k3) {
    for (size_t k2 = 0; k2 < K2; ++k2) {
      for (size_t k1 = 0; k1 < K1; ++k1) {
        for (size_t j = 0; j < N; ++j) {
          CT a_j = static_cast<CT>(a(j, k3, k2, k1));

          // Let us find bfloat/half overload of min/max.
          using std::max;
          using std::min;

          switch (op) {
            case ReduceOp::kSum:
              c(0, j) = c(0, j) + a_j;
              break;
            case ReduceOp::kSumSquared:
              c(0, j) = c(0, j) + a_j * a_j;
              break;
            case ReduceOp::kMin:
              c(0, j) = min(c(0, j), a_j);
              break;
            case ReduceOp::kMax:
              c(0, j) = max(c(0, j), a_j);
              break;
            case ReduceOp::kMinMax:
              c(0, j) = min(c(0, j), a_j);
              c(1, j) = max(c(1, j), a_j);
              break;
            default:
              YNN_UNREACHABLE;
          }
        }
      }
    }
  }
}

template <typename AT, typename CT>
void TestUnaryReduce(AT, CT, size_t n, size_t k3, size_t k2, size_t k1,
                     size_t pad_n, ReduceOp op, unary_reduce_kernel_fn kernel) {
  ReplicableRandomDevice rng;

  const float max_abs_value = 10.0f;
  TypeGenerator<AT> a_gen(-max_abs_value, max_abs_value, quantization_params{});
  TypeGenerator<CT> c_gen(-max_abs_value, max_abs_value, quantization_params{});

  Tensor<AT> a({n, k3, k2, k1});
  Tensor<CT> c({OutputRows(op), n + pad_n});

  c = c.crop_padding({0, 0}, {0, pad_n});

  a.generate([&]() { return a_gen(rng); });

  // Fill the output with some random data, and copy it for the reference
  // result before running the kernel, which updates it in place.
  c.generate([&]() { return c_gen(rng); });
  Tensor<CT> expected = c.deep_copy();

  kernel(n, k3, k2, k1, a.stride(0) * sizeof(AT), a.stride(1) * sizeof(AT),
         a.stride(2) * sizeof(AT), a.base(), c.stride(0) * sizeof(CT),
         c.base());

  // Verify results.
  Reference(a, expected, op);
  for (const auto& i : EnumerateIndices(c.extents())) {
    if (std::is_integral<CT>::value) {
      ASSERT_EQ(c(i), expected(i))
          << "shape=" << n << "x" << k3 << "x" << k2 << "x" << k1;
    } else {
      const float tolerance =
          Tolerance<CT>(op, k3 * k2 * k1 + 1, max_abs_value);
      ASSERT_NEAR(c(i), expected(i), tolerance)
          << "shape=" << n << "x" << k3 << "x" << k2 << "x" << k1;
    }
  }
}

struct KernelParam {
  uint64_t arch_flags;
  unary_reduce_kernel_fn kernel;
  ReduceOp op;
  multi_type type;
};

const char* to_string(const KernelParam& param) { return ""; }

class UnaryReduce : public ::testing::TestWithParam<KernelParam> {};

const size_t max_dim = 128;
const auto n_values = simd_sizes_up_to(max_dim);
const auto k_values = simd_sizes_up_to(max_dim);
const size_t no_pad_n = 0;

TEST_P(UnaryReduce, n) {
  KernelParam kernel = GetParam();
  if (!is_arch_supported(kernel.arch_flags)) GTEST_SKIP();
  SwitchTwoTypes(kernel.type, [&](auto a_type, auto c_type) {
    for (size_t n : n_values) {
      TestUnaryReduce(a_type, c_type, n, 1, 1, max_dim, no_pad_n, kernel.op,
                      kernel.kernel);
      TestUnaryReduce(a_type, c_type, n, 1, max_dim, 1, no_pad_n, kernel.op,
                      kernel.kernel);
      TestUnaryReduce(a_type, c_type, n, max_dim, 1, 1, no_pad_n, kernel.op,
                      kernel.kernel);
    }
  });
}

TEST_P(UnaryReduce, n_padded) {
  KernelParam kernel = GetParam();
  if (!is_arch_supported(kernel.arch_flags)) GTEST_SKIP();
  SwitchTwoTypes(kernel.type, [&](auto a_type, auto c_type) {
    for (size_t pad_n : {1, 2, 3}) {
      TestUnaryReduce(a_type, c_type, max_dim, 1, 1, max_dim, pad_n, kernel.op,
                      kernel.kernel);
    }
  });
}

TEST_P(UnaryReduce, k1) {
  KernelParam kernel = GetParam();
  if (!is_arch_supported(kernel.arch_flags)) GTEST_SKIP();
  SwitchTwoTypes(kernel.type, [&](auto a_type, auto c_type) {
    for (size_t k1 : k_values) {
      TestUnaryReduce(a_type, c_type, max_dim, 1, 1, k1, no_pad_n, kernel.op,
                      kernel.kernel);
    }
  });
}

TEST_P(UnaryReduce, k2) {
  KernelParam kernel = GetParam();
  if (!is_arch_supported(kernel.arch_flags)) GTEST_SKIP();
  SwitchTwoTypes(kernel.type, [&](auto a_type, auto c_type) {
    for (size_t k2 : k_values) {
      TestUnaryReduce(a_type, c_type, max_dim, 1, k2, 1, no_pad_n, kernel.op,
                      kernel.kernel);
    }
  });
}

TEST_P(UnaryReduce, k3) {
  KernelParam kernel = GetParam();
  if (!is_arch_supported(kernel.arch_flags)) GTEST_SKIP();
  SwitchTwoTypes(kernel.type, [&](auto a_type, auto c_type) {
    for (size_t k3 : k_values) {
      TestUnaryReduce(a_type, c_type, max_dim, k3, 1, 1, no_pad_n, kernel.op,
                      kernel.kernel);
    }
  });
}

#define TEST_REDUCE_KERNEL(op, arch_flags, name, a_type, c_type) \
  INSTANTIATE_TEST_SUITE_P(                                      \
      name, UnaryReduce,                                         \
      testing::Values(KernelParam{arch_flags, name, op,          \
                                  multi_type_of(a_type(), c_type())}));

#define YNN_UNARY_REDUCE_KERNEL(arch_flags, name, a_type, c_type) \
  TEST_REDUCE_KERNEL(ReduceOp::kSum, arch_flags, name, a_type, c_type);
#include "ynnpack/kernels/reduce/sum.inc"
#undef YNN_UNARY_REDUCE_KERNEL

#define YNN_UNARY_REDUCE_KERNEL(arch_flags, name, a_type, c_type) \
  TEST_REDUCE_KERNEL(ReduceOp::kSumSquared, arch_flags, name, a_type, c_type);
#include "ynnpack/kernels/reduce/sum_squared.inc"
#undef YNN_UNARY_REDUCE_KERNEL

#define YNN_UNARY_REDUCE_KERNEL(arch_flags, name, a_type, c_type) \
  TEST_REDUCE_KERNEL(ReduceOp::kMin, arch_flags, name, a_type, c_type);
#include "ynnpack/kernels/reduce/min.inc"
#undef YNN_UNARY_REDUCE_KERNEL

#define YNN_UNARY_REDUCE_KERNEL(arch_flags, name, a_type, c_type) \
  TEST_REDUCE_KERNEL(ReduceOp::kMax, arch_flags, name, a_type, c_type);
#include "ynnpack/kernels/reduce/max.inc"
#undef YNN_UNARY_REDUCE_KERNEL

#define YNN_UNARY_REDUCE_KERNEL(arch_flags, name, a_type, c_type) \
  TEST_REDUCE_KERNEL(ReduceOp::kMinMax, arch_flags, name, a_type, c_type);
#include "ynnpack/kernels/reduce/min_max.inc"
#undef YNN_UNARY_REDUCE_KERNEL

}  // namespace ynn
