// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <cstddef>
#include <cstdint>
#include <type_traits>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "ynnpack/base/arithmetic.h"
#include "ynnpack/base/bfloat16.h"
#include "ynnpack/base/half.h"
#include "ynnpack/base/test/fuzz_test.h"
#include "ynnpack/base/test/random.h"
#include "ynnpack/base/test/tensor.h"
#include "ynnpack/base/type.h"
#include "ynnpack/include/ynnpack.h"
#include "ynnpack/subgraph/test/scheduler.h"
#include "ynnpack/subgraph/test/subgraph_builder.h"

namespace ynn {
namespace {

using ::testing::ElementsAreArray;

template <typename T>
class TransposeSplitTailAndUnitDimTest : public ::testing::Test {};

using TestTypes = ::testing::Types<int2x4, uint2x4, int4x2, uint4x2, int8_t,
                                   uint8_t, half, bfloat16, float>;
TYPED_TEST_SUITE(TransposeSplitTailAndUnitDimTest, TestTypes);

TYPED_TEST(TransposeSplitTailAndUnitDimTest, UnitDimAndSplitTail) {
  using T = TypeParam;
  constexpr size_t elem_count = type_info<T>::element_count();
  TestScheduler scheduler(3);
  ReplicableRandomDevice rng;

  auto run_transpose = [&](const std::vector<size_t>& input_shape,
                           const std::vector<int32_t>& permutation) {
    SubgraphBuilder subgraph(2);
    subgraph.AddInput(type_of<T>(), input_shape, 0)
        .AddOutput(type_of<T>(), permutation.size(), 1)
        .AddTranspose(permutation, 0, 1);
    Runtime runtime(subgraph.GetSubgraph(), &scheduler);
    ASSERT_EQ(runtime.Status(), ynn_status_success);
    Tensor<T> input(input_shape);
    fill_random(input.data(), input.size(), rng);
    Tensor<T> expected = input.transpose(permutation).deep_copy();
    runtime.ReshapeExternalTensor(input_shape, input.data(), 0)
        .ReshapeRuntime();
    Tensor<T> output(expected.extents());
    runtime.SetupExternalTensor(output.data(), 1).InvokeRuntime();
    EXPECT_THAT(output, ElementsAreArray(expected));
  };

  // Triggers optimize_dims dropping a unit dim while permutation has rank 3.
  if constexpr (elem_count == 1) {
    run_transpose({17, 1, 32}, {2, 0, 1});
  }
  const size_t dim = align_up<size_t>(17, elem_count);
  run_transpose({dim, 1, dim}, {1, 2, 0});
  run_transpose({16, 32, 4}, {1, 0, 2});
  run_transpose({dim, dim, dim}, {0, 2, 1});

  // Triggers splits[1] == 1 in a 3D transpose with large dim 0 and tests the
  // copy fallback when fused with a dot that splits dim 0 to extent 1.
  if constexpr (std::is_same_v<T, float>) {
    run_transpose({128, 32, 1024}, {2, 0, 1});

    const size_t m = 17, k = 2049, n = 32;
    SubgraphBuilder subgraph(3);
    uint32_t transposed_a_id = YNN_INVALID_VALUE_ID;
    subgraph.AddInput(type_of<float>(), {k, m}, 0)
        .AddInput(type_of<float>(), {k, n}, 1)
        .AddOutput(type_of<float>(), {m, n}, 2)
        .AddTensor(type_of<float>(), {m, k}, transposed_a_id)
        .AddTranspose({1, 0}, 0, transposed_a_id)
        .AddDot(1, transposed_a_id, 1, YNN_INVALID_VALUE_ID, 2);
    Runtime runtime(subgraph.GetSubgraph(), &scheduler);
    ASSERT_EQ(runtime.Status(), ynn_status_success);
    Tensor<float> input_a({k, m});
    Tensor<float> input_b({k, n});
    input_a.fill(1.0f);
    input_b.fill(1.0f);
    Tensor<float> output({m, n});
    runtime.ReshapeExternalTensor({k, m}, input_a.data(), 0)
        .ReshapeExternalTensor({k, n}, input_b.data(), 1)
        .ReshapeRuntime()
        .SetupExternalTensor(output.data(), 2)
        .InvokeRuntime();
    Tensor<float> expected({m, n});
    expected.fill(static_cast<float>(k));
    EXPECT_THAT(output, ElementsAreArray(expected));
  }
}

}  // namespace
}  // namespace ynn
