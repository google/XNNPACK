// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <cstdint>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "ynnpack/base/type.h"
#include "ynnpack/include/ynnpack.h"
#include "ynnpack/subgraph/subgraph.h"
#include "ynnpack/subgraph/test/matchers.h"
#include "ynnpack/subgraph/test/subgraph_builder.h"

namespace ynn {

using ::testing::AllOf;
using ::testing::Not;

TEST(invalidate_dead_values, one_dead_value) {
  uint32_t temp_id = YNN_INVALID_VALUE_ID;

  SubgraphBuilder builder(0);
  builder.AddTensor(type_of<float>(), 1, temp_id);

  ynn_subgraph_t subgraph = builder.GetSubgraph();
  subgraph->invalidate_dead_values();

  EXPECT_THAT(subgraph,
              AllOf(Not(HasValidValueId(temp_id)), HasValidValueCount(0)));
}

TEST(invalidate_dead_values, transitive) {
  uint32_t a_id = YNN_INVALID_VALUE_ID;
  uint32_t b_id = YNN_INVALID_VALUE_ID;

  SubgraphBuilder builder(0);
  builder.AddTensor(type_of<float>(), 1, a_id)
      .AddTensor(type_of<float>(), 1, b_id);

  builder.AddUnary(ynn_unary_negate, a_id, b_id);

  ynn_subgraph_t subgraph = builder.GetSubgraph();
  subgraph->invalidate_dead_values();

  // b is dead, which makes a dead too.
  EXPECT_THAT(*subgraph,
              AllOf(Not(HasValidValueId(a_id)), Not(HasValidValueId(b_id)),
                    HasValidValueCount(0)));
}

TEST(invalidate_dead_values, outputs_are_not_dead) {
  const uint32_t output_id = 0;
  const uint32_t input_id = 1;

  uint32_t temp_id = YNN_INVALID_VALUE_ID;

  SubgraphBuilder builder(2);
  builder.AddOutput(type_of<float>(), 1, output_id)
      .AddTensor(type_of<float>(), 1, temp_id)
      .AddInput(type_of<float>(), 1, input_id);

  builder.AddUnary(ynn_unary_square, input_id, temp_id);
  builder.AddUnary(ynn_unary_negate, temp_id, output_id);

  ynn_subgraph_t subgraph = builder.GetSubgraph();
  subgraph->invalidate_dead_values();

  EXPECT_THAT(subgraph, HasValidValueCount(3));
}

}  // namespace ynn
