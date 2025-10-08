// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "ynnpack/base/type.h"
#include "ynnpack/include/ynnpack.h"
#include "ynnpack/subgraph/subgraph.h"
#include "ynnpack/subgraph/test/subgraph_builder.h"

using testing::ElementsAre;

namespace ynn {

int valid_node_count(ynn_subgraph_t subgraph) {
  return std::count_if(subgraph->nodes.begin(), subgraph->nodes.end(),
                       [](const ynn_node& node) { return node.is_valid(); });
}

// This should return `std::span` when it is generally available.
template <typename T>
std::vector<T> make_span(const T* data, size_t size) {
  return std::vector<T>(data, data + size);
}

TEST(fold_constants, simple) {
  const uint32_t input_id = 0;
  const uint32_t output_id = 1;
  uint32_t constant_id = 2;
  uint32_t temp_id = YNN_INVALID_VALUE_ID;

  constexpr float constant_values[] = {0.0f, 1.0f, 2.0f, 3.0f};
  SubgraphBuilder builder(3);
  builder.AddInput(type_of<float>(), 1, input_id)
      .AddOutput(type_of<float>(), 1, output_id)
      .AddTensor(type_of<float>(), {4}, constant_id, &constant_values)
      .AddTensor(type_of<float>(), 1, temp_id);

  const uint32_t one_id = builder.DefineScalar(1.0f);
  builder.AddBinary(ynn_binary_add, constant_id, one_id, temp_id)
      .AddBinary(ynn_binary_add, input_id, temp_id, output_id);

  ynn_subgraph_t subgraph = builder.GetSubgraph();
  subgraph->fold_constants(nullptr);

  const ynn_value& constant = subgraph->value(temp_id);
  ASSERT_TRUE(subgraph->value(constant_id).is_valid());
  ASSERT_TRUE(constant.is_static());
  ASSERT_THAT(make_span(constant.data->cast<float>().base(), 4),
              ElementsAre(1.0f, 2.0f, 3.0f, 4.0f));
  ASSERT_EQ(valid_node_count(subgraph), 1);

  subgraph->invalidate_dead_values();
  ASSERT_FALSE(subgraph->value(constant_id).is_valid());
}

TEST(fold_constants, transitive) {
  const uint32_t input_id = 0;
  const uint32_t output_id = 1;
  uint32_t constant_id = 2;
  uint32_t temp_id = YNN_INVALID_VALUE_ID;
  uint32_t temp2_id = YNN_INVALID_VALUE_ID;

  constexpr float constant_values[] = {0.0f, 1.0f, 2.0f, 3.0f};
  SubgraphBuilder builder(3);
  builder.AddInput(type_of<float>(), 1, input_id)
      .AddOutput(type_of<float>(), 1, output_id)
      .AddTensor(type_of<float>(), {4}, constant_id, &constant_values)
      .AddTensor(type_of<float>(), 1, temp_id)
      .AddTensor(type_of<float>(), 1, temp2_id);

  const uint32_t one_id = builder.DefineScalar(1.0f);
  const uint32_t two_id = builder.DefineScalar(2.0f);
  builder.AddBinary(ynn_binary_multiply, constant_id, two_id, temp2_id)
      .AddBinary(ynn_binary_add, temp2_id, one_id, temp_id)
      .AddBinary(ynn_binary_add, input_id, temp_id, output_id);

  ynn_subgraph_t subgraph = builder.GetSubgraph();
  subgraph->fold_constants(nullptr);

  const ynn_value& constant = subgraph->value(temp_id);
  ASSERT_TRUE(subgraph->value(constant_id).is_valid());
  ASSERT_TRUE(constant.is_static());
  ASSERT_THAT(make_span(constant.data->cast<float>().base(), 4),
              ElementsAre(1.0f, 3.0f, 5.0f, 7.0f));
  ASSERT_EQ(valid_node_count(subgraph), 1);

  subgraph->invalidate_dead_values();
  ASSERT_FALSE(subgraph->value(constant_id).is_valid());
  ASSERT_FALSE(subgraph->value(temp2_id).is_valid());
}

}  // namespace ynn
