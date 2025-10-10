// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <cstdint>

#include <gtest/gtest.h>
#include "ynnpack/include/ynnpack.h"
#include "ynnpack/subgraph/test/subgraph_builder.h"

namespace ynn {

TEST(Errors, bad_binary_shape) {
  const uint32_t a_id = 0;
  const uint32_t b_id = 1;
  const uint32_t x_id = 2;
  SubgraphBuilder subgraph(3);
  subgraph.AddInput(ynn_type_fp32, 2, a_id)
      .AddInput(ynn_type_fp32, 2, b_id)
      .AddOutput(ynn_type_fp32, 2, x_id)
      .AddBinary(ynn_binary_add, a_id, b_id, x_id);

  Runtime runtime(subgraph.GetSubgraph());
  ASSERT_EQ(runtime.Status(), ynn_status_success);

  runtime.ReshapeExternalTensor({10, 20}, nullptr, a_id)
      .ReshapeExternalTensor({5, 20}, nullptr, b_id);

  ASSERT_EQ(runtime.ReshapeRuntime().Status(), ynn_status_error);
}

TEST(Errors, bad_dot_shape) {
  const uint32_t a_id = 0;
  const uint32_t b_id = 1;
  const uint32_t x_id = 2;
  SubgraphBuilder subgraph(3);
  subgraph.AddInput(ynn_type_fp32, 4, a_id)
      .AddInput(ynn_type_fp32, 4, b_id)
      .AddOutput(ynn_type_fp32, 4, x_id)
      .AddDot(2, a_id, b_id, YNN_INVALID_VALUE_ID, x_id);

  Runtime runtime(subgraph.GetSubgraph());
  ASSERT_EQ(runtime.Status(), ynn_status_success);

  runtime.ReshapeExternalTensor({5, 10, 20, 30}, nullptr, a_id)
      .ReshapeExternalTensor({1, 2, 3, 4}, nullptr, b_id);

  ASSERT_EQ(runtime.ReshapeRuntime().Status(), ynn_status_error);
}

}  // namespace ynn
