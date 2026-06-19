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

TEST(Errors, broadcast_axis_out_of_bounds) {
  const uint32_t in_id = 0;
  SubgraphBuilder subgraph(1);
  subgraph.AddInput(ynn_type_fp32, 3, in_id);

  // An axis far outside [-rank, rank) maps to a slinky dim past the input's
  // dimensions. Such a dimension is an implicit broadcast, so broadcasting it
  // is a no-op: it must be skipped rather than indexing the axes_set bitset out
  // of range.
  const int32_t axes[] = {-100};
  uint32_t output_id = YNN_INVALID_VALUE_ID;
  EXPECT_EQ(ynn_define_broadcast(subgraph.GetSubgraph(), /*num_axes=*/1, axes,
                                 in_id, &output_id, /*flags=*/0),
            ynn_status_success);
  EXPECT_EQ(output_id, in_id);
}

}  // namespace ynn
