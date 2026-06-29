// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <numeric>
#include <vector>

#include <gtest/gtest.h>
#include "ynnpack/include/ynnpack.h"
#include "ynnpack/subgraph/test/subgraph_builder.h"

namespace ynn {

TEST(subgraphs, input_reused) {
  // This subgraph computes x = abs(a) - a.
  const uint32_t a_id = 0;
  const uint32_t x_id = 1;
  SubgraphBuilder builder(2);
  uint32_t a_sq_id = YNN_INVALID_VALUE_ID;
  builder.AddInput(ynn_type_fp32, 1, a_id)
      .AddOutput(ynn_type_fp32, 1, x_id)
      .AddTensor(ynn_type_fp32, 1, a_sq_id);
  builder.AddUnary(ynn_unary_abs, a_id, a_sq_id)
      .AddBinary(ynn_binary_subtract, a_sq_id, a_id, x_id);

  Runtime runtime(builder.GetSubgraph());

  const size_t n = 10;
  std::vector<float> a(n);
  std::iota(a.begin(), a.end(), -4);

  // Copy a and use it for both the input and output, which requires that we
  // don't try to alias abs(a) with the output.
  std::vector<float> x(a);

  runtime.ReshapeExternalTensor({n}, x.data(), a_id)
      .ReshapeExternalTensor({n}, x.data(), x_id)
      .ReshapeRuntime()
      .InvokeRuntime();
  ASSERT_EQ(runtime.Status(), ynn_status_success);

  for (size_t i = 0; i < n; ++i) {
    ASSERT_EQ(x[i], std::abs(a[i]) - a[i]);
  }
}

TEST(subgraphs, static_tensor_cache) {
  SubgraphBuilder builder(0);

  uint32_t id1 = YNN_INVALID_VALUE_ID;
  uint32_t id2 = YNN_INVALID_VALUE_ID;
  uint32_t id3 = YNN_INVALID_VALUE_ID;
  uint32_t id4 = YNN_INVALID_VALUE_ID;

  float data1[] = {1.0f, 2.0f, 3.0f};
  float data2[] = {1.0f, 2.0f, 3.0f};
  float data3[] = {1.0f, 2.0f, 4.0f};
  double data4[] = {1.0f, 2.0f, 3.0f};

  builder.AddTensor(ynn_type_fp32, {3}, id1, data1, YNN_VALUE_FLAG_COPY_DATA);
  builder.AddTensor(ynn_type_fp32, {3}, id2, data2, YNN_VALUE_FLAG_COPY_DATA);
  builder.AddTensor(ynn_type_fp32, {3}, id3, data3, YNN_VALUE_FLAG_COPY_DATA);
  builder.AddTensor(ynn_type_fp64, {3}, id4, data4, YNN_VALUE_FLAG_COPY_DATA);

  EXPECT_EQ(id1, id2);
  EXPECT_NE(id1, id3);
  EXPECT_NE(id1, id4);

  // Test with dims = nullptr (scalar)
  uint32_t id5 = YNN_INVALID_VALUE_ID;
  uint32_t id6 = YNN_INVALID_VALUE_ID;
  float scalar1 = 5.0f;
  float scalar2 = 5.0f;
  float scalar3 = 6.0f;
  builder.AddTensor(ynn_type_fp32, {}, id5, &scalar1, YNN_VALUE_FLAG_COPY_DATA);
  builder.AddTensor(ynn_type_fp32, {}, id6, &scalar2, YNN_VALUE_FLAG_COPY_DATA);
  EXPECT_EQ(id5, id6);

  uint32_t id7 = YNN_INVALID_VALUE_ID;
  builder.AddTensor(ynn_type_fp32, {}, id7, &scalar3, YNN_VALUE_FLAG_COPY_DATA);
  EXPECT_NE(id5, id7);
}

}  // namespace ynn
