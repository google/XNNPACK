// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <random>
#include <tuple>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "ynnpack/base/test/fuzz_test.h"
#include "ynnpack/base/test/random.h"
#include "ynnpack/base/test/tensor.h"
#include "ynnpack/base/test/util.h"
#include "ynnpack/base/type.h"
#include "ynnpack/include/ynnpack.h"
#include "ynnpack/subgraph/test/subgraph_builder.h"

namespace ynn {

using testing::ElementsAre;

template <typename T>
void RunIota(T begin, const std::vector<T>& stride, Tensor<T>& output) {
  const size_t rank = output.rank();

  // Define subgraph
  SubgraphBuilder subgraph(3);
  const uint32_t begin_id = 0;
  const uint32_t stride_id = 1;
  const uint32_t output_id = 2;

  subgraph.AddInput(type_of<T>(), 0, begin_id)
      .AddInput(type_of<T>(), 1, stride_id)
      .AddOutput(type_of<T>(), rank, output_id);

  subgraph.AddIota(type_of<T>(), output.shape(), begin_id, stride_id,
                   output_id);

  Runtime runtime(subgraph.GetSubgraph());
  ASSERT_EQ(runtime.Status(), ynn_status_success);

  runtime.ReshapeExternalTensor(0, &begin, begin_id)
      .ReshapeExternalTensor({rank}, const_cast<T*>(stride.data()), stride_id)
      .ReshapeRuntime();
  ASSERT_EQ(runtime.GetExternalTensorShape(output_id), output.shape());

  runtime.SetupExternalTensor(output.data(), output_id);
  runtime.InvokeRuntime();
  ASSERT_EQ(runtime.Status(), ynn_status_success);
}

template <typename T>
void RunFill(T begin, Tensor<T>& output) {
  const size_t rank = output.rank();

  // Define subgraph
  SubgraphBuilder subgraph(2);
  const uint32_t begin_id = 0;
  const uint32_t output_id = 1;

  subgraph.AddInput(type_of<T>(), 0, begin_id)
      .AddOutput(type_of<T>(), rank, output_id);

  subgraph.AddIota(type_of<T>(), output.shape(), begin_id, YNN_INVALID_VALUE_ID,
                   output_id);

  Runtime runtime(subgraph.GetSubgraph());
  ASSERT_EQ(runtime.Status(), ynn_status_success);

  runtime.ReshapeExternalTensor(0, &begin, begin_id).ReshapeRuntime();
  ASSERT_EQ(runtime.GetExternalTensorShape(output_id), output.shape());

  runtime.SetupExternalTensor(output.data(), output_id);
  runtime.InvokeRuntime();
  ASSERT_EQ(runtime.Status(), ynn_status_success);
}

template <typename T>
void RunFill(Tensor<T>& output) {
  const size_t rank = output.rank();

  // Define subgraph
  SubgraphBuilder subgraph(1);
  const uint32_t output_id = 0;

  subgraph.AddOutput(type_of<T>(), rank, output_id);

  subgraph.AddIota(type_of<T>(), output.shape(), YNN_INVALID_VALUE_ID,
                   YNN_INVALID_VALUE_ID, output_id);

  Runtime runtime(subgraph.GetSubgraph());
  ASSERT_EQ(runtime.Status(), ynn_status_success);

  runtime.ReshapeRuntime();
  ASSERT_EQ(runtime.GetExternalTensorShape(output_id), output.shape());

  runtime.SetupExternalTensor(output.data(), output_id);
  runtime.InvokeRuntime();
  ASSERT_EQ(runtime.Status(), ynn_status_success);
}

template <typename T>
class iota : public testing::Test {};

TYPED_TEST_SUITE_P(iota);

TYPED_TEST_P(iota, rank0) {
  Tensor<TypeParam> output(std::vector<size_t>{});
  RunIota<TypeParam>(3, {}, output);
  ASSERT_THAT(output, ElementsAre(3));
}

TYPED_TEST_P(iota, rank1_stride1) {
  Tensor<TypeParam> output({5});
  RunIota<TypeParam>(2, {1}, output);
  ASSERT_THAT(output, ElementsAre(2, 3, 4, 5, 6));
}

TYPED_TEST_P(iota, rank1_stride3) {
  Tensor<TypeParam> output({5});
  RunIota<TypeParam>(2, {3}, output);
  ASSERT_THAT(output, ElementsAre(2, 5, 8, 11, 14));
}

TYPED_TEST_P(iota, rank2_stride1_0) {
  Tensor<TypeParam> output({4, 3});
  RunIota<TypeParam>(3, {0, 1}, output);
  ASSERT_THAT(output, ElementsAre(3, 4, 5, 3, 4, 5, 3, 4, 5, 3, 4, 5));
}

TYPED_TEST_P(iota, rank2_flat) {
  Tensor<TypeParam> output({3, 3});
  RunIota<TypeParam>(2, {3, 1}, output);
  ASSERT_THAT(output, ElementsAre(2, 3, 4, 5, 6, 7, 8, 9, 10));
}

TYPED_TEST_P(iota, rank2_fill_zero) {
  Tensor<TypeParam> output({3, 2});
  RunFill<TypeParam>(output);
  ASSERT_THAT(output, ElementsAre(0, 0, 0, 0, 0, 0));
}

TYPED_TEST_P(iota, rank2_fill_one) {
  Tensor<TypeParam> output({3, 2});
  RunFill<TypeParam>(1, output);
  ASSERT_THAT(output, ElementsAre(1, 1, 1, 1, 1, 1));
}

TYPED_TEST_P(iota, random) {
  using T = TypeParam;
  ReplicableRandomDevice rng;
  std::uniform_int_distribution<size_t> rank_dist(0, 4);
  std::uniform_int_distribution<size_t> dim_dist(1, 9);

  for (auto _ : FuzzTest(std::chrono::milliseconds(250))) {
    size_t rank = rank_dist(rng);
    std::vector<size_t> shape = random_shape(rng, rank);

    T begin = random_value<T>(rng);
    std::vector<T> stride(rank);
    fill_random(stride.data(), stride.size(), rng, -10, 10);

    Tensor<T> output(shape);
    RunIota(begin, stride, output);

    for (auto i : EnumerateIndices(shape)) {
      T expected = begin;
      for (int d = 0; d < rank; ++d) {
        expected = expected + i[d] * stride[d];
      }
#ifdef YNN_ARCH_ARM
      // ARM SIMD is not consistent with scalar arithmetic.
      ASSERT_NEAR(output(i), expected, 1e-3f);
#else
      ASSERT_EQ(output(i), expected);
#endif
    }
  }
}

REGISTER_TYPED_TEST_SUITE_P(iota, rank0, rank1_stride1, rank1_stride3,
                            rank2_stride1_0, rank2_flat, rank2_fill_zero,
                            rank2_fill_one, random);

using types = testing::Types<int32_t, float>;

INSTANTIATE_TYPED_TEST_SUITE_P(test, iota, types);

}  // namespace ynn
