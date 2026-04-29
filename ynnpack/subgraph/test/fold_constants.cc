// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <cstddef>
#include <cstdint>
#include <numeric>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "ynnpack/base/half.h"
#include "ynnpack/base/type.h"
#include "ynnpack/include/ynnpack.h"
#include "ynnpack/subgraph/subgraph.h"
#include "ynnpack/subgraph/test/matchers.h"
#include "ynnpack/subgraph/test/subgraph_builder.h"

namespace ynn {

using ::testing::ElementsAre;
using ::testing::Not;

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
  ASSERT_THAT(subgraph, HasValidValueId(constant_id));
  EXPECT_TRUE(constant.is_static());
  EXPECT_THAT(ValuesIn<float>(constant), ElementsAre(1.0f, 2.0f, 3.0f, 4.0f));
  ASSERT_THAT(subgraph, HasValidNodeCount(1));

  subgraph->invalidate_dead_values();
  EXPECT_THAT(subgraph, Not(HasValidValueId(constant_id)));
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
  ASSERT_THAT(subgraph, HasValidValueId(constant_id));
  EXPECT_TRUE(constant.is_static());
  EXPECT_THAT(ValuesIn<float>(constant), ElementsAre(1.0f, 3.0f, 5.0f, 7.0f));
  ASSERT_THAT(subgraph, HasValidNodeCount(1));

  subgraph->invalidate_dead_values();
  EXPECT_THAT(subgraph, Not(HasValidValueId(constant_id)));
  EXPECT_THAT(subgraph, Not(HasValidValueId(temp2_id)));
}

TEST(fold_constants, scalar_fp16_to_fp32_folded) {
  const uint32_t input_id = 0;
  const uint32_t output_id = 1;
  uint32_t fp16_id = YNN_INVALID_VALUE_ID;
  uint32_t fp32_id = YNN_INVALID_VALUE_ID;

  half constant_value = 1.0f;
  SubgraphBuilder builder(2);
  builder.AddInput(type_of<float>(), {}, input_id)
      .AddOutput(type_of<float>(), {}, output_id)
      .AddTensor(type_of<half>(), {}, fp16_id, &constant_value)
      .AddTensor(type_of<float>(), {}, fp32_id);

  builder.AddConvert(fp16_id, type_of<float>(), fp32_id)
      .AddBinary(ynn_binary_add, input_id, fp32_id, output_id);

  ynn_subgraph_t subgraph = builder.GetSubgraph();
  subgraph->fold_constants(nullptr);

  EXPECT_TRUE(subgraph->value(fp32_id).is_static());
  EXPECT_THAT(ValuesIn<float>(subgraph->value(fp32_id)), ElementsAre(1.0f));
  EXPECT_THAT(subgraph, HasValidNodeCount(1));
}

TEST(fold_constants, vector_fp16_to_fp32_not_folded) {
  const uint32_t input_id = 0;
  const uint32_t output_id = 1;
  uint32_t fp16_id = YNN_INVALID_VALUE_ID;
  uint32_t fp32_id = YNN_INVALID_VALUE_ID;

  half constant_values[256];
  std::iota(constant_values, constant_values + 256, 0.0f);
  SubgraphBuilder builder(2);
  builder.AddInput(type_of<float>(), {256}, input_id)
      .AddOutput(type_of<float>(), {256}, output_id)
      .AddTensor(type_of<half>(), {256}, fp16_id, &constant_values)
      .AddTensor(type_of<float>(), {256}, fp32_id);

  builder.AddConvert(fp16_id, type_of<float>(), fp32_id)
      .AddBinary(ynn_binary_add, input_id, fp32_id, output_id);

  ynn_subgraph_t subgraph = builder.GetSubgraph();
  subgraph->fold_constants(nullptr);

  EXPECT_FALSE(subgraph->value(fp32_id).is_static());
  EXPECT_THAT(subgraph, HasValidNodeCount(2));
}

TEST(fold_constants, iota_square_folding) {
  for (size_t size : {2, 16}) {
    SubgraphBuilder builder(1);
    const uint32_t begin_id = builder.DefineScalar(0.0f);
    const uint32_t stride_id = builder.DefineScalar(1.0f);
    uint32_t iota_id = YNN_INVALID_VALUE_ID;
    uint32_t square_id = YNN_INVALID_VALUE_ID;
    const uint32_t output_id = 0;

    builder.AddTensor(type_of<float>(), {size, size}, iota_id)
        .AddTensor(type_of<float>(), {size, size}, square_id)
        .AddOutput(type_of<float>(), {size, size}, output_id);

    builder
        .AddIota(type_of<float>(), {size, size}, begin_id, stride_id, iota_id)
        .AddBinary(ynn_binary_multiply, iota_id, iota_id, square_id)
        .AddBinary(ynn_binary_add, square_id, builder.DefineScalar(1.0f),
                   output_id);

    ynn_subgraph_t subgraph = builder.GetSubgraph();
    subgraph->fold_constants(nullptr);

    const ynn_value& square_val = subgraph->value(square_id);
    if (size == 2) {
      EXPECT_TRUE(square_val.is_static()) << size;
    } else {
      EXPECT_FALSE(square_val.is_static()) << size;
    }
  }
}

}  // namespace ynn
