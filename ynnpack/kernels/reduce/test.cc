// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "ynnpack/base/arch.h"  // IWYU pragma: keep
#include "ynnpack/base/arithmetic.h"
#include "ynnpack/base/base.h"
#include "ynnpack/base/span.h"
#include "ynnpack/base/test/fuzz_test.h"
#include "ynnpack/base/test/random.h"
#include "ynnpack/base/test/tensor.h"
#include "ynnpack/base/test/util.h"
#include "ynnpack/base/type.h"
#include "ynnpack/kernels/reduce/reduce.h"

namespace ynn {

using testing::ElementsAreArray;
using testing::FloatNear;
using testing::Pointwise;

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
      return type_info<T>::epsilon() * k * max_abs_value;
    case ReduceOp::kSumSquared:
      return type_info<T>::epsilon() * k * max_abs_value * max_abs_value;
    case ReduceOp::kMin:
    case ReduceOp::kMax:
    case ReduceOp::kMinMax:
      return 0.0f;
  }
  YNN_UNREACHABLE;
  return 0.0f;
}

template <typename T>
span<T> row(Tensor<T> t, size_t i) {
  assert(t.stride(1) == 1);
  return span<T>(&t(i, 0), t.extent(1));
}

template <typename AT, typename CT>
YNN_ALWAYS_INLINE void ReduceRow(ReduceOp op, size_t n, const AT* a,
                                 size_t c_row_stride, CT* c_0, CT* c_1) {
  // Get the previous row.
  const CT* prev_c_0 = c_0 - c_row_stride;
  const CT* prev_c_1 = c_1 - c_row_stride;

  // Compute the current row.
  for (size_t i = 0; i < n; ++i) {
    CT a_i = static_cast<CT>(a[i]);

    switch (op) {
      case ReduceOp::kSum:
        c_0[i] = prev_c_0[i];
        kahan_sum(a_i, c_0[i], c_1[i]);
        break;
      case ReduceOp::kSumSquared:
        c_0[i] = prev_c_0[i];
        kahan_sum(static_cast<CT>(a_i * a_i), c_0[i], c_1[i]);
        break;
      case ReduceOp::kMin:
        c_0[i] = std::min(prev_c_0[i], a_i);
        break;
      case ReduceOp::kMax:
        c_0[i] = std::max(prev_c_0[i], a_i);
        break;
      case ReduceOp::kMinMax:
        c_0[i] = std::min(prev_c_0[i], a_i);
        c_1[i] = std::max(prev_c_1[i], a_i);
        break;
      default:
        YNN_UNREACHABLE;
    }
  }
}

const float max_abs_value = 255;

template <typename AT, typename CT>
void TestUnaryReduce(Tensor<AT> a, Tensor<CT> c, Tensor<CT> expected,
                     ReduceOp op, reduce_kernel_fn kernel, bool is_k1) {
  const size_t n = is_k1 ? a.extent(0) : a.extent(1);
  const size_t k = is_k1 ? a.extent(1) : a.extent(0);

  // We will modify c
  c = c.deep_copy();

  void* x0 = c.base();
  void* x1 = offset_bytes(c.base(), c.stride_bytes(0));

  kernel(n, k, a.stride_bytes(0), a.base(), x0, x1);

  ASSERT_EQ(c.rank(), 2);
  for (size_t i = 0; i < c.extent(0); ++i) {
    if (is_integral<CT>::value) {
      EXPECT_THAT(row(c, i), ElementsAreArray(row(expected, i)))
          << "shape=" << n << "x" << k << " row=" << i;
    } else {
      const float tolerance = Tolerance<CT>(op, k + 1, max_abs_value);
      EXPECT_THAT(row(c, i), Pointwise(FloatNear(tolerance), row(expected, i)))
          << "shape=" << n << "x" << k << " row=" << i;
    }
  }
}

template <typename AT, typename CT>
void TestUnaryReduce(AT, CT, std::vector<size_t> ns, std::vector<size_t> ks,
                     ReduceOp op, reduce_kernel_fn kernel, int k_dim) {
  ReplicableRandomDevice rng;

  // Get the max size of the buffer we want to test.
  const size_t max_n = *std::max_element(ns.begin(), ns.end());
  const size_t max_k = *std::max_element(ks.begin(), ks.end());

  const size_t c_m = OutputRows(op);

  // To prepare the buffers for the test, we always construct the buffers with
  // the reduction dimension being the row dimension.
  Tensor<AT> a({max_k, max_n});
  Tensor<CT> x({c_m, max_n});
  fill_random(a.data(), a.size(), rng, -max_abs_value, max_abs_value);
  fill_random(x.data(), x.size(), rng, -max_abs_value, max_abs_value);

  // Precompute the expected result, such that `expected_max(:, k, :)` is the
  // result of the reduction of `a` from 0 to k. This is the "prefix sum" in the
  // case of sum reductions.
  Tensor<CT> expected({c_m, max_k + 1, max_n + 1});
  expected.slice(1, 0).remove_dim(1).crop({0, 0}, {c_m, max_n}).assign(x);
  std::vector<CT> error(max_n, static_cast<CT>(0));
  for (size_t i = 1; i <= a.extent(0); ++i) {
    CT* c0 = &expected(0, i, 0);
    // For 2-row outputs, we pass the second row. Otherwise we pass `error`,
    // which we use for Kahan summation.
    CT* c1 = c_m == 2 ? &expected(1, i, 0) : error.data();
    ReduceRow(op, a.extent(1), &a(i - 1, 0), expected.stride(1), c0, c1);
  }

  if (k_dim == reduce_dim::k1) {
    // If this is a k1 kernel, we need to transpose a.
    a = a.transpose({1, 0}).deep_copy();
  }

  int n_dim = k_dim;
  k_dim = 1 - k_dim;

  std::vector<size_t> a_shape = a.shape();
  std::vector<size_t> x_shape = x.shape();

  for (size_t k : ks) {
    a_shape[k_dim] = k;
    Tensor<CT> expected_k =
        expected.slice(1, k).remove_dim(1).crop({0, 0}, {c_m, max_n});
    for (size_t n : ns) {
      a_shape[n_dim] = n;
      x_shape[1] = n;

      a.set_shape(a_shape, a.strides());
      x.set_shape(x_shape, x.strides());
      expected_k.set_shape({c_m, n}, expected_k.strides());

      TestUnaryReduce(a, x, expected_k, op, kernel, k_dim);
    }
  }
}

struct KernelParam {
  uint64_t arch_flags;
  reduce_kernel_fn kernel;
  ReduceOp op;
  multi_type type;
  int k_dim;
};

const char* to_string(const KernelParam& param) { return ""; }

class UnaryReduce : public ::testing::TestWithParam<KernelParam> {};

constexpr size_t max_k_dim_bytes = 1024;
constexpr size_t max_n_dim = 64;

TEST_P(UnaryReduce, k) {
  KernelParam kernel = GetParam();
  if (!is_arch_supported(kernel.arch_flags)) GTEST_SKIP();

  SwitchTwoTypes(kernel.type, [&](auto a_type, auto c_type) {
    constexpr size_t max_k_dim = max_k_dim_bytes / sizeof(a_type);
    const auto k_values = simd_sizes_up_to(max_k_dim);
    TestUnaryReduce(a_type, c_type, {max_n_dim}, k_values, kernel.op,
                    kernel.kernel, kernel.k_dim);
  });
}

TEST_P(UnaryReduce, n) {
  KernelParam kernel = GetParam();
  if (!is_arch_supported(kernel.arch_flags)) GTEST_SKIP();

  SwitchTwoTypes(kernel.type, [&](auto a_type, auto c_type) {
    constexpr size_t max_k_dim = max_k_dim_bytes / sizeof(a_type);
    std::vector<size_t> n_values = simd_sizes_up_to(max_n_dim);
    TestUnaryReduce(a_type, c_type, n_values, {max_k_dim}, kernel.op,
                    kernel.kernel, kernel.k_dim);
  });
}

#define YNN_REDUCE_KERNEL(arch_flags, name, k_dim, a_type, c_type) \
  INSTANTIATE_TEST_SUITE_P(                                        \
      name, UnaryReduce,                                           \
      testing::Values(KernelParam{arch_flags, name, current_op,    \
                                  multi_type_of(a_type(), c_type()), k_dim}));

#define current_op ReduceOp::kSum
#include "ynnpack/kernels/reduce/sum.inc"
#undef current_op

#define current_op ReduceOp::kSumSquared
#include "ynnpack/kernels/reduce/sum_squared.inc"
#undef current_op

#define current_op ReduceOp::kMin
#include "ynnpack/kernels/reduce/min.inc"
#undef current_op

#define current_op ReduceOp::kMax
#include "ynnpack/kernels/reduce/max.inc"
#undef current_op

#define current_op ReduceOp::kMinMax
#include "ynnpack/kernels/reduce/min_max.inc"
#undef current_op

}  // namespace ynn
