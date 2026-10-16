// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <gtest/gtest.h>

#include <cstddef>
#include <cstdint>
#include <string>
#include <tuple>
#include <type_traits>
#include <variant>
#include <vector>

#include "ynnpack/base/bfloat16.h"
#include "ynnpack/base/test/tensor.h"
#include "ynnpack/base/type.h"
#include "ynnpack/include/ynnpack.h"
#include "ynnpack/subgraph/subgraph.h"
#include "ynnpack/subgraph/test/scheduler.h"
#include "ynnpack/subgraph/test/subgraph_builder.h"

namespace ynn {
namespace {

// XLA:CPU lowers `reduce(dot(a, broadcast(x)))` (e.g. from the gradient of a
// JAX function) to this subgraph:
//
//   x_expanded = static_expand_dims(x, ...)
//   b          = static_broadcast(x_expanded, {batch, k, n})
//   ab         = dot(a, b)
//   out        = reduce_sum(ab, {2})
//
// Slinky implements the broadcast by aliasing its input with a stride of 0 in
// the broadcast dimensions, unless the consumer requires otherwise. The dot
// must handle a B that is a broadcast in n, in k, or in both, whether or not it
// decides to pack B.
enum class BroadcastKind {
  // b[i, k, n] = x: stride 0 in both k and n.
  kScalar,
  // b[i, k, n] = x[i, k]: stride 0 in n.
  kN,
  // b[i, k, n] = x[i, n]: stride 0 in k.
  kK,
  // b = transpose(broadcast(x, {batch, n, k})): stride 0 in both k and n. A
  // transposed B is always packed (the transpose is aliased into the packing),
  // so this covers `pack_b` independently of the kernel selection heuristics.
  kScalarTransposed,
};

std::string ToString(BroadcastKind kind) {
  switch (kind) {
    case BroadcastKind::kScalar:
      return "Scalar";
    case BroadcastKind::kN:
      return "N";
    case BroadcastKind::kK:
      return "K";
    case BroadcastKind::kScalarTransposed:
      return "ScalarTransposed";
  }
  return "";
}

enum class Type { kFp32, kFp64, kBf16 };

std::string ToString(Type type) {
  switch (type) {
    case Type::kFp32:
      return "fp32";
    case Type::kFp64:
      return "fp64";
    case Type::kBf16:
      return "bf16";
  }
  return "";
}

struct Shape {
  size_t batch, m, k, n;
};

using Params = std::tuple<Type, BroadcastKind, Shape, int>;

// The values of `x` and `a` are small multiples of 0.5, so all of the products
// and sums below are exact in every type we test.
float XValue(size_t i, size_t j) {
  return 0.5f * static_cast<float>(static_cast<int>((i * 5 + j * 3) % 7) - 3);
}
float AValue(size_t i, size_t j, size_t k) {
  return static_cast<float>(static_cast<int>((i * 7 + j * 3 + k) % 11) - 5);
}

// Returns true if the B operand of the dot producing `dot_id` is packed.
bool IsBPacked(const ynn_subgraph& subgraph, uint32_t dot_id) {
  const ynn_node* dot = subgraph.get_producer(dot_id);
  if (!dot) return false;
  const ynn_node* b = subgraph.get_producer(dot->inputs[1]);
  return b && std::holds_alternative<ynn_node::pack_b>(b->op);
}

template <typename T>
void TestBatchedDotBroadcastReduce(BroadcastKind kind, const Shape& shape,
                                   int threads) {
  // The type of the dot and reduction (bf16 x bf16 -> fp32).
  using C = std::conditional_t<std::is_same_v<T, double>, double, float>;
  const size_t batch = shape.batch;
  const size_t m = shape.m;
  const size_t k = shape.k;
  const size_t n = shape.n;

  const std::vector<size_t> a_shape = {batch, m, k};
  const std::vector<size_t> b_shape = {batch, k, n};
  const std::vector<size_t> ab_shape = {batch, m, n};
  const std::vector<size_t> out_shape = {batch, m};
  std::vector<size_t> x_shape;
  switch (kind) {
    case BroadcastKind::kScalar:
    case BroadcastKind::kScalarTransposed:
      x_shape = {};
      break;
    case BroadcastKind::kN:
      x_shape = {batch, k};
      break;
    case BroadcastKind::kK:
      x_shape = {batch, n};
      break;
  }

  SubgraphBuilder subgraph(3);
  const uint32_t a_id = 0;
  const uint32_t x_id = 1;
  const uint32_t out_id = 2;
  subgraph.AddInput(type_of<T>(), a_shape, a_id)
      .AddInput(type_of<T>(), x_shape, x_id)
      .AddOutput(type_of<C>(), out_shape, out_id);

  uint32_t x_expanded_id = YNN_INVALID_VALUE_ID;
  uint32_t b_id = YNN_INVALID_VALUE_ID;
  uint32_t ab_id = YNN_INVALID_VALUE_ID;
  uint32_t init_id = YNN_INVALID_VALUE_ID;
  subgraph.AddTensor(type_of<T>(), 3, x_expanded_id)
      .AddTensor(type_of<T>(), b_shape, b_id)
      .AddTensor(type_of<C>(), ab_shape, ab_id);
  subgraph.AddScalar<C>(0, init_id);

  switch (kind) {
    case BroadcastKind::kScalar:
      subgraph.AddExpandDims({0, 1, 2}, x_id, x_expanded_id)
          .AddStaticBroadcast(b_shape, x_expanded_id, b_id);
      break;
    case BroadcastKind::kN:
      subgraph.AddExpandDims({2}, x_id, x_expanded_id)
          .AddStaticBroadcast(b_shape, x_expanded_id, b_id);
      break;
    case BroadcastKind::kK:
      subgraph.AddExpandDims({1}, x_id, x_expanded_id)
          .AddStaticBroadcast(b_shape, x_expanded_id, b_id);
      break;
    case BroadcastKind::kScalarTransposed: {
      uint32_t b_t_id = YNN_INVALID_VALUE_ID;
      subgraph.AddTensor(type_of<T>(), {batch, n, k}, b_t_id);
      subgraph.AddExpandDims({0, 1, 2}, x_id, x_expanded_id)
          .AddStaticBroadcast({batch, n, k}, x_expanded_id, b_t_id)
          .AddTranspose({0, 2, 1}, b_t_id, b_id);
      break;
    }
  }
  subgraph.AddDot(/*num_k_dims=*/1, a_id, b_id, YNN_INVALID_VALUE_ID, ab_id)
      .AddReduce(ynn_reduce_sum, {2}, ab_id, init_id, out_id);

  // Whether B is packed depends on the kernels and cost models available on
  // the target, except for a transposed B, which is always packed.
  const bool b_packed = IsBPacked(*subgraph.GetSubgraph(), ab_id);
  ::testing::Test::RecordProperty("b_packed", b_packed ? "true" : "false");
  if (kind == BroadcastKind::kScalarTransposed) {
    ASSERT_TRUE(b_packed);
  }

  TestScheduler scheduler(threads);
  Runtime runtime(subgraph.GetSubgraph(), &scheduler);
  ASSERT_EQ(runtime.Status(), ynn_status_success);

  Tensor<T> a(a_shape);
  for (size_t i = 0; i < batch; ++i) {
    for (size_t j = 0; j < m; ++j) {
      for (size_t l = 0; l < k; ++l) {
        a(i, j, l) = static_cast<T>(AValue(i, j, l));
      }
    }
  }
  Tensor<T> x(x_shape);
  if (x_shape.empty()) {
    x.data()[0] = static_cast<T>(0.5f);
  } else {
    for (size_t i = 0; i < x_shape[0]; ++i) {
      for (size_t j = 0; j < x_shape[1]; ++j) {
        x(i, j) = static_cast<T>(XValue(i, j));
      }
    }
  }
  auto b_value = [&](size_t i, size_t l, size_t j) -> double {
    switch (kind) {
      case BroadcastKind::kScalar:
      case BroadcastKind::kScalarTransposed:
        return 0.5;
      case BroadcastKind::kN:
        return XValue(i, l);
      case BroadcastKind::kK:
        return XValue(i, j);
    }
    return 0.0;
  };

  runtime.ReshapeExternalTensor(a_shape, a.data(), a_id)
      .ReshapeExternalTensor(x_shape, x.data(), x_id)
      .ReshapeRuntime();
  ASSERT_EQ(runtime.Status(), ynn_status_success);
  ASSERT_EQ(runtime.GetExternalTensorShape(out_id), out_shape);

  Tensor<C> out(out_shape);
  out.fill(static_cast<C>(-12345));
  runtime.SetupExternalTensor(out.data(), out_id).InvokeRuntime();
  ASSERT_EQ(runtime.Status(), ynn_status_success);

  // out[i, j] = sum_n sum_k a[i, j, k] * b[i, k, n]
  for (size_t i = 0; i < batch; ++i) {
    for (size_t j = 0; j < m; ++j) {
      double expected = 0.0;
      for (size_t l = 0; l < k; ++l) {
        for (size_t o = 0; o < n; ++o) {
          expected += AValue(i, j, l) * b_value(i, l, o);
        }
      }
      ASSERT_EQ(static_cast<double>(out(i, j)), expected)
          << "at batch " << i << ", row " << j;
    }
  }
}

class BatchedDotBroadcastReduceTest : public ::testing::TestWithParam<Params> {
};

TEST_P(BatchedDotBroadcastReduceTest, Sum) {
  const auto& [type, kind, shape, threads] = GetParam();
  switch (type) {
    case Type::kFp32:
      TestBatchedDotBroadcastReduce<float>(kind, shape, threads);
      break;
    case Type::kFp64:
      TestBatchedDotBroadcastReduce<double>(kind, shape, threads);
      break;
    case Type::kBf16:
      TestBatchedDotBroadcastReduce<bfloat16>(kind, shape, threads);
      break;
  }
}

INSTANTIATE_TEST_SUITE_P(
    BatchedDotBroadcastReduce, BatchedDotBroadcastReduceTest,
    ::testing::Combine(
        ::testing::Values(Type::kFp32, Type::kFp64, Type::kBf16),
        ::testing::Values(BroadcastKind::kScalar, BroadcastKind::kN,
                          BroadcastKind::kK, BroadcastKind::kScalarTransposed),
        // {batch, m, k, n}
        ::testing::Values(Shape{1, 16, 16, 16}, Shape{5, 60, 60, 60},
                          Shape{3, 7, 24, 40}, Shape{2, 48, 70, 20}),
        ::testing::Values(1, 8)),
    [](const ::testing::TestParamInfo<Params>& info) {
      const Shape& shape = std::get<2>(info.param);
      return ToString(std::get<0>(info.param)) + "_" +
             ToString(std::get<1>(info.param)) + "_" +
             std::to_string(shape.batch) + "x" + std::to_string(shape.m) + "x" +
             std::to_string(shape.k) + "x" + std::to_string(shape.n) +
             "_threads" + std::to_string(std::get<3>(info.param));
    });

}  // namespace
}  // namespace ynn
