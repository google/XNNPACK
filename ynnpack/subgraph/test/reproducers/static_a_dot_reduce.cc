// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <cstddef>
#include <cstdint>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "ynnpack/base/test/tensor.h"
#include "ynnpack/base/type.h"
#include "ynnpack/include/ynnpack.h"
#include "ynnpack/subgraph/test/scheduler.h"
#include "ynnpack/subgraph/test/subgraph_builder.h"

namespace ynn {
namespace {

using ::testing::ElementsAreArray;

// Computes sum(dot(a, b), axis=0) where `a` is a constant. Dot kernels that
// want a transposed `a` constant fold the transpose, which requires strides
// that are not those of a dense array
// (https://github.com/jax-ml/jax/issues/42651).
void TestStaticADotReduce(size_t m, size_t k, size_t n, bool transpose_b) {
  Tensor<float> a({m, k});
  Tensor<float> b(transpose_b ? std::vector<size_t>{n, k}
                              : std::vector<size_t>{k, n});
  // Use small integers so the result is exact.
  int next = 0;
  a.generate([&]() { return static_cast<float>(next++ % 7 - 3); });
  b.generate([&]() { return static_cast<float>(next++ % 5 - 2); });

  const uint32_t a_id = 0;
  const uint32_t b_id = 1;
  const uint32_t output_id = 2;
  uint32_t dot_b_id = b_id;
  uint32_t dot_id = YNN_INVALID_VALUE_ID;

  SubgraphBuilder subgraph(3);
  subgraph.AddTensor(a, a_id)
      .AddInput(type_of<float>(), b.extents(), b_id)
      .AddOutput(type_of<float>(), 1, output_id)
      .AddTensor(type_of<float>(), 2, dot_id);
  if (transpose_b) {
    dot_b_id = YNN_INVALID_VALUE_ID;
    subgraph.AddTensor(type_of<float>(), 2, dot_b_id)
        .AddTranspose({1, 0}, b_id, dot_b_id);
  }
  subgraph.AddDot(1, a_id, dot_b_id, YNN_INVALID_VALUE_ID, dot_id)
      .AddReduce(ynn_reduce_sum, {0}, dot_id, YNN_INVALID_VALUE_ID, output_id);

  TestScheduler scheduler(3);
  Runtime runtime(subgraph.GetSubgraph(), &scheduler);
  ASSERT_EQ(runtime.Status(), ynn_status_success);

  Tensor<float> output({n});
  runtime.ReshapeExternalTensor(b.extents(), b.data(), b_id)
      .ReshapeRuntime()
      .SetupExternalTensor(output.data(), output_id)
      .InvokeRuntime();
  ASSERT_EQ(runtime.Status(), ynn_status_success);

  Tensor<float> expected({n});
  for (size_t j = 0; j < n; ++j) {
    float sum = 0.0f;
    for (size_t i = 0; i < m; ++i) {
      for (size_t l = 0; l < k; ++l) {
        sum += a(i, l) * (transpose_b ? b(j, l) : b(l, j));
      }
    }
    expected(j) = sum;
  }
  EXPECT_THAT(output, ElementsAreArray(expected));
}

TEST(StaticADotReduce, Dot) { TestStaticADotReduce(16, 16, 256, false); }

TEST(StaticADotReduce, TransposedB) { TestStaticADotReduce(13, 13, 316, true); }

}  // namespace
}  // namespace ynn
