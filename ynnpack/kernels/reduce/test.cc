// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <vector>

#include <gtest/gtest.h>
#include "ynnpack/base/arch.h"  // IWYU pragma: keep
#include "ynnpack/base/base.h"
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
      return epsilon(type_of<T>()) * k * max_abs_value * 5;
    case ReduceOp::kSumSquared:
      return epsilon(type_of<T>()) * k * max_abs_value * max_abs_value * 10;
    case ReduceOp::kMin:
    case ReduceOp::kMax:
    case ReduceOp::kMinMax:
      return 0.0f;
  }
  YNN_UNREACHABLE;
  return 0.0f;
}

template <typename AT, typename CT>
YNN_ALWAYS_INLINE void ReduceRow(ReduceOp op, CT* c_0, CT* c_1, const AT* a,
                                 size_t N, size_t K1, size_t a_stride_n) {
  for (size_t j = 0; j < N; ++j) {
    for (size_t k1 = 0; k1 < K1; ++k1) {
      CT a_j = static_cast<CT>(a[k1]);

      switch (op) {
        case ReduceOp::kSum:
          c_0[j] = c_0[j] + a_j;
          break;
        case ReduceOp::kSumSquared:
          c_0[j] = c_0[j] + a_j * a_j;
          break;
        case ReduceOp::kMin:
          c_0[j] = std::min(c_0[j], a_j);
          break;
        case ReduceOp::kMax:
          c_0[j] = std::max(c_0[j], a_j);
          break;
        case ReduceOp::kMinMax:
          c_0[j] = std::min(c_0[j], a_j);
          c_1[j] = std::max(c_1[j], a_j);
          break;
        default:
          YNN_UNREACHABLE;
      }
    }
    a += a_stride_n;
  }
}

template <typename X4x2>
YNN_ALWAYS_INLINE void ReduceRowInt4Sum(ReduceOp op, int32_t* c_0,
                                        int32_t* /*c_1*/, const X4x2* a,
                                        size_t N, size_t K1,
                                        size_t a_stride_n) {
  for (size_t j = 0; j < N; ++j) {
    for (size_t k1 = 0; k1 < K1; ++k1) {
      int32_t v0 = static_cast<int32_t>(a[k1].get(0));
      int32_t v1 = static_cast<int32_t>(a[k1].get(1));
      switch (op) {
        case ReduceOp::kSum:
          c_0[j] += v0 + v1;
          break;
        case ReduceOp::kSumSquared:
          c_0[j] += v0 * v0 + v1 * v1;
          break;
        default:
          YNN_UNREACHABLE;
      }
    }
    a += a_stride_n;
  }
}

YNN_ALWAYS_INLINE void ReduceRow(ReduceOp op, int32_t* c_0, int32_t* c_1,
                                 const int4x2* a, size_t N, size_t K1,
                                 size_t a_stride_n) {
  ReduceRowInt4Sum(op, c_0, c_1, a, N, K1, a_stride_n);
}

YNN_ALWAYS_INLINE void ReduceRow(ReduceOp op, int32_t* c_0, int32_t* c_1,
                                 const uint4x2* a, size_t N, size_t K1,
                                 size_t a_stride_n) {
  ReduceRowInt4Sum(op, c_0, c_1, a, N, K1, a_stride_n);
}

template <typename AT, typename CT>
void Reference(Tensor<AT> a, Tensor<CT> c, ReduceOp op) {
  // This helper allows omitting 2 of the 3 k dimensions. Canonicalize to 3 k
  // dimensions here.
  while (a.rank() < 4) {
    a = a.expand_dims({1});
  }

  ASSERT_EQ(a.extent(0), c.extent(1));
  const size_t a_stride_n = a.stride(0);
  const size_t a_stride_k1 = a.stride(3);
  const size_t K3 = a.extent(1);
  const size_t K2 = a.extent(2);
  const size_t K1 = a.extent(3);
  const size_t N = c.extent(1);
  CT* c_0 = &c(0, 0);
  CT* c_1 = &c(1, 0);
  for (size_t k3 = 0; k3 < K3; ++k3) {
    for (size_t k2 = 0; k2 < K2; ++k2) {
      if (a_stride_n == 1) {
        // Move the loop over k1 outside the loop over n (and call the "kernel"
        // with K1=1).
        const AT* a_k1 = &a(0, k3, k2, 0);
        for (size_t k1 = 0; k1 < K1; ++k1) {
          ReduceRow(op, c_0, c_1, a_k1, N, /*K1=*/1,
                    /*a_stride_n=*/1);
          a_k1 += a_stride_k1;
        }
      } else {
        ReduceRow(op, c_0, c_1, &a(0, k3, k2, 0), N, K1, a_stride_n);
      }
    }
  }
}

const float max_abs_value = 10.0f;

template <typename AT, typename CT>
void TestUnaryReduce(Tensor<AT> a, Tensor<CT> c,
                     ReduceOp op, unary_reduce_kernel_fn kernel) {
  const size_t n = a.extent(0);
  const size_t k3 = a.extent(1);
  const size_t k2 = a.extent(2);
  const size_t k1 = a.extent(3);

  // We will modify c
  c = c.deep_copy();

  // Fill the output with some random data, and copy it for the
  // reference result before running the kernel, which updates it in
  // place.
  Tensor<CT> expected = c.deep_copy();

  kernel(n, k3, k2, k1, a.stride(0) * sizeof(AT),
          a.stride(1) * sizeof(AT), a.stride(2) * sizeof(AT), a.base(),
          c.stride(0) * sizeof(CT), c.base());

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

template <typename AT, typename CT>
void TestUnaryReduce(AT, CT, std::vector<size_t> ns, std::vector<size_t> k3s,
                     std::vector<size_t> k2s, std::vector<size_t> k1s,
                     ReduceOp op, unary_reduce_kernel_fn kernel) {
  ReplicableRandomDevice rng;

  TypeGenerator<AT> a_gen(-max_abs_value, max_abs_value, quantization_params{});
  TypeGenerator<CT> c_gen(-max_abs_value, max_abs_value, quantization_params{});

  // Get the max size of the buffer we want to test.
  const size_t max_n = *std::max_element(ns.begin(), ns.end());
  const size_t max_k1 = *std::max_element(k1s.begin(), k1s.end());
  const size_t max_k2 = *std::max_element(k2s.begin(), k2s.end());
  const size_t max_k3 = *std::max_element(k3s.begin(), k3s.end());

  // We want n to be contiguous if k1 is 1, so allocate it in that order, and
  // transpose it after.
  Tensor<AT> a_max({max_k3, max_k2, max_n, max_k1});
  a_max = a_max.transpose({2, 0, 1, 3});
  Tensor<CT> c_max({OutputRows(op), max_n});
  a_max.generate([&]() { return a_gen(rng); });
  c_max.generate([&]() { return c_gen(rng); });

  for (size_t n : ns) {
    for (size_t k3 : k3s) {
      for (size_t k2 : k2s) {
        for (size_t k1 : k1s) {
          Tensor<AT> a = a_max.crop({0, 0, 0, 0}, {n, k3, k2, k1});
          Tensor<CT> c = c_max.crop({0, 0}, {OutputRows(op), n});
          TestUnaryReduce(a, c, op, kernel);
        }
      }
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

const size_t max_dim = 512;
const auto n_values = simd_sizes_up_to(max_dim);
const auto k_values = simd_sizes_up_to(max_dim);

TEST_P(UnaryReduce, n) {
  KernelParam kernel = GetParam();
  if (!is_arch_supported(kernel.arch_flags)) GTEST_SKIP();
  SwitchTwoTypes(kernel.type, [&](auto a_type, auto c_type) {
    TestUnaryReduce(a_type, c_type, n_values, {1}, {1}, {max_dim},
                    kernel.op, kernel.kernel);
    TestUnaryReduce(a_type, c_type, n_values, {1}, {max_dim}, {1},
                    kernel.op, kernel.kernel);
    TestUnaryReduce(a_type, c_type, n_values, {max_dim}, {1}, {1},
                    kernel.op, kernel.kernel);
  });
}

TEST_P(UnaryReduce, k1) {
  KernelParam kernel = GetParam();
  if (!is_arch_supported(kernel.arch_flags)) GTEST_SKIP();
  SwitchTwoTypes(kernel.type, [&](auto a_type, auto c_type) {
    TestUnaryReduce(a_type, c_type, {max_dim}, {1}, {1}, k_values, kernel.op,
                    kernel.kernel);
  });
}

TEST_P(UnaryReduce, k2) {
  KernelParam kernel = GetParam();
  if (!is_arch_supported(kernel.arch_flags)) GTEST_SKIP();
  SwitchTwoTypes(kernel.type, [&](auto a_type, auto c_type) {
    TestUnaryReduce(a_type, c_type, {max_dim}, {1}, k_values, {1}, kernel.op,
                    kernel.kernel);
  });
}

TEST_P(UnaryReduce, k3) {
  KernelParam kernel = GetParam();
  if (!is_arch_supported(kernel.arch_flags)) GTEST_SKIP();
  SwitchTwoTypes(kernel.type, [&](auto a_type, auto c_type) {
    TestUnaryReduce(a_type, c_type, {max_dim}, k_values, {1}, {1}, kernel.op,
                    kernel.kernel);
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
