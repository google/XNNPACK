// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "ynnpack/base/arch.h"  // IWYU pragma: keep
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
      return type_info<T>::epsilon() * k * max_abs_value * 5;
    case ReduceOp::kSumSquared:
      return type_info<T>::epsilon() * k * max_abs_value * max_abs_value * 10;
    case ReduceOp::kMin:
    case ReduceOp::kMax:
    case ReduceOp::kMinMax:
      return 0.0f;
  }
  YNN_UNREACHABLE;
  return 0.0f;
}

const float max_abs_value = 10.0f;

template <typename T>
span<T> row(Tensor<T> t, size_t i) {
  return span<T>(&t(i, 0), t.extent(1));
}

template <typename AT, typename CT>
YNN_ALWAYS_INLINE void ReduceRow(ReduceOp op, CT* c_0, CT* c_1, const AT* a,
                                 size_t k, size_t a_stride_k) {
  for (size_t j = 0; j < k; ++j) {
    CT a_j = static_cast<CT>(*offset_bytes(a, j * a_stride_k));

    switch (op) {
      case ReduceOp::kSum:
        c_0[0] = c_0[0] + a_j;
        break;
      case ReduceOp::kSumSquared:
        c_0[0] = c_0[0] + a_j * a_j;
        break;
      case ReduceOp::kMin:
        c_0[0] = std::min(c_0[0], a_j);
        break;
      case ReduceOp::kMax:
        c_0[0] = std::max(c_0[0], a_j);
        break;
      case ReduceOp::kMinMax:
        c_0[0] = std::min(c_0[0], a_j);
        c_1[0] = std::max(c_1[0], a_j);
        break;
      default:
        YNN_UNREACHABLE;
    }
  }
}

template <typename AT, typename CT>
void Reference(ReduceOp op, size_t n, size_t k, size_t a_stride_n,
               size_t a_stride_k, const AT* a, size_t c_stride_m, CT* c,
               bool is_k1) {
  for (size_t i = 0; i < n; ++i) {
    CT* c_0 = offset_bytes(c, i * sizeof(CT));
    CT* c_1 = offset_bytes(c, i * sizeof(CT) + c_stride_m);
    ReduceRow(op, c_0, c_1, offset_bytes(a, i * a_stride_n), k,
              is_k1 ? sizeof(AT) : a_stride_k);
  }
}

template <typename AT, typename CT>
void TestUnaryReduce(Tensor<AT> a, Tensor<CT> c, ReduceOp op,
                     reduce_kernel_fn kernel, bool is_k1) {
  const size_t n = a.extent(0);
  const size_t k = a.extent(1);

  // We will modify c
  c = c.deep_copy();

  // Fill the output with some random data, and copy it for the
  // reference result before running the kernel, which updates it in
  // place.
  Tensor<CT> expected = c.deep_copy();

  if (is_k1) {
    kernel(n, k, a.stride_bytes(0), a.base(), c.stride_bytes(0), c.base());

    // Verify results.
    Reference(op, n, k, a.stride_bytes(0), a.stride_bytes(1), a.base(),
              expected.stride_bytes(0), expected.base(), is_k1);
  } else {
    kernel(k, n, a.stride_bytes(0), a.base(), c.stride_bytes(0), c.base());

    // Verify results.
    Reference(op, k, n, a.stride_bytes(1), a.stride_bytes(0), a.base(),
              expected.stride_bytes(0), expected.base(), is_k1);
  }

  ASSERT_EQ(c.rank(), 2);
  for (size_t i = 0; i < c.extent(0); ++i) {
    if (is_integral<CT>::value) {
      EXPECT_THAT(row(c, i), ElementsAreArray(row(expected, i)))
          << "shape=" << n << "x" << k << " row=" << i;
    } else {
      const float tolerance =
          Tolerance<CT>(op, (is_k1 ? k : n) + 1, max_abs_value);
      EXPECT_THAT(row(c, i), Pointwise(FloatNear(tolerance), row(expected, i)))
          << "shape=" << n << "x" << k << " row=" << i;
    }
  }
}

template <typename AT, typename CT>
void TestUnaryReduce(AT, CT, std::vector<size_t> ns, std::vector<size_t> ks,
                     ReduceOp op, reduce_kernel_fn kernel, bool is_k1) {
  ReplicableRandomDevice rng;

  // Get the max size of the buffer we want to test.
  const size_t max_n = *std::max_element(ns.begin(), ns.end());
  const size_t max_k = *std::max_element(ks.begin(), ks.end());

  Tensor<AT> a_max({max_n, max_k});
  Tensor<CT> c_max({OutputRows(op), is_k1 ? max_n : max_k});
  fill_random(a_max.data(), a_max.size(), rng, -max_abs_value, max_abs_value);
  fill_random(c_max.data(), c_max.size(), rng, -max_abs_value, max_abs_value);

  for (size_t n : ns) {
    for (size_t k : ks) {
      Tensor<AT> a = a_max.crop({0, 0}, {n, k});
      Tensor<CT> c = c_max.crop({0, 0}, {OutputRows(op), is_k1 ? n : k});
      TestUnaryReduce(a, c, op, kernel, is_k1);
    }
  }
}

struct KernelParam {
  uint64_t arch_flags;
  reduce_kernel_fn kernel;
  ReduceOp op;
  multi_type type;
  bool is_k1;
};

const char* to_string(const KernelParam& param) { return ""; }

class UnaryReduce : public ::testing::TestWithParam<KernelParam> {};

constexpr size_t max_k_dim_bytes = 256;
constexpr size_t max_n_dim = 64;

TEST_P(UnaryReduce, k) {
  KernelParam kernel = GetParam();
  if (!is_arch_supported(kernel.arch_flags)) GTEST_SKIP();
  SwitchTwoTypes(kernel.type, [&](auto a_type, auto c_type) {
    const size_t max_k_dim = max_k_dim_bytes / sizeof(a_type);
    const auto k_values = simd_sizes_up_to(max_k_dim);
    TestUnaryReduce(a_type, c_type, {max_n_dim}, k_values, kernel.op,
                    kernel.kernel, kernel.is_k1);
  });
}

TEST_P(UnaryReduce, n) {
  KernelParam kernel = GetParam();
  std::vector<size_t> ns = simd_sizes_up_to(max_n_dim);
  if (!is_arch_supported(kernel.arch_flags)) GTEST_SKIP();
  SwitchTwoTypes(kernel.type, [&](auto a_type, auto c_type) {
    constexpr size_t max_k_dim = max_k_dim_bytes / sizeof(a_type);
    TestUnaryReduce(a_type, c_type, ns, {max_k_dim}, kernel.op, kernel.kernel,
                    kernel.is_k1);
  });
}

#define TEST_REDUCE_KERNEL(op, arch_flags, name, a_type, c_type, is_k1) \
  INSTANTIATE_TEST_SUITE_P(                                             \
      name, UnaryReduce,                                                \
      testing::Values(KernelParam{arch_flags, name, op,                 \
                                  multi_type_of(a_type(), c_type()), is_k1}));

#define YNN_REDUCE_K1_KERNEL(arch_flags, name, a_type, c_type) \
  TEST_REDUCE_KERNEL(current_op, arch_flags, name, a_type, c_type, true);
#define YNN_REDUCE_KN_KERNEL(arch_flags, name, a_type, c_type) \
  TEST_REDUCE_KERNEL(current_op, arch_flags, name, a_type, c_type, false);

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
