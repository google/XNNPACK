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
#include "ynnpack/subgraph/dot.h"
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

TEST(fold_constants, caching_shared_pointer) {
  constexpr float constant_values[] = {0.0f, 1.0f, 2.0f, 3.0f};

  slinky::raw_buffer_ptr data1;
  {
    const uint32_t input_id = 0;
    const uint32_t output_id = 1;
    uint32_t constant_id = 2;
    uint32_t temp_id = YNN_INVALID_VALUE_ID;

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
    ASSERT_TRUE(constant.is_static());
    data1 = constant.data;
  }

  slinky::raw_buffer_ptr data2;
  {
    const uint32_t input_id = 0;
    const uint32_t output_id = 1;
    uint32_t constant_id = 2;
    uint32_t temp_id = YNN_INVALID_VALUE_ID;

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
    ASSERT_TRUE(constant.is_static());
    data2 = constant.data;
  }

  EXPECT_EQ(data1.get(), data2.get());
}

TEST(fold_constants, caching_copied_value) {
  constexpr float constant_values1[] = {0.0f, 1.0f, 2.0f, 3.0f};
  constexpr float constant_values2[] = {0.0f, 1.0f, 2.0f, 3.0f};

  slinky::raw_buffer_ptr data1;
  {
    const uint32_t input_id = 0;
    const uint32_t output_id = 1;
    uint32_t constant_id = 2;
    uint32_t temp_id = YNN_INVALID_VALUE_ID;

    SubgraphBuilder builder(3);
    builder.AddInput(type_of<float>(), 1, input_id)
        .AddOutput(type_of<float>(), 1, output_id)
        .AddTensor(type_of<float>(), {4}, constant_id, &constant_values1,
                   YNN_VALUE_FLAG_COPY_DATA)
        .AddTensor(type_of<float>(), 1, temp_id);

    const uint32_t one_id = builder.DefineScalar(1.0f);
    builder.AddBinary(ynn_binary_add, constant_id, one_id, temp_id)
        .AddBinary(ynn_binary_add, input_id, temp_id, output_id);

    ynn_subgraph_t subgraph = builder.GetSubgraph();
    subgraph->fold_constants(nullptr);

    const ynn_value& constant = subgraph->value(temp_id);
    ASSERT_TRUE(constant.is_static());
    data1 = constant.data;
  }

  slinky::raw_buffer_ptr data2;
  {
    const uint32_t input_id = 0;
    const uint32_t output_id = 1;
    uint32_t constant_id = 2;
    uint32_t temp_id = YNN_INVALID_VALUE_ID;

    SubgraphBuilder builder(3);
    builder.AddInput(type_of<float>(), 1, input_id)
        .AddOutput(type_of<float>(), 1, output_id)
        .AddTensor(type_of<float>(), {4}, constant_id, &constant_values2,
                   YNN_VALUE_FLAG_COPY_DATA)
        .AddTensor(type_of<float>(), 1, temp_id);

    const uint32_t one_id = builder.DefineScalar(1.0f);
    builder.AddBinary(ynn_binary_add, constant_id, one_id, temp_id)
        .AddBinary(ynn_binary_add, input_id, temp_id, output_id);

    ynn_subgraph_t subgraph = builder.GetSubgraph();
    subgraph->fold_constants(nullptr);

    const ynn_value& constant = subgraph->value(temp_id);
    ASSERT_TRUE(constant.is_static());
    data2 = constant.data;
  }

  EXPECT_EQ(data1.get(), data2.get());
}

TEST(fold_constants, mix_cached_uncached) {
  // Run 1: Evaluate and cache temp1 and temp2.
  {
    const uint32_t input_id = 0;
    const uint32_t output_id = 1;
    uint32_t const1_id = 2;
    uint32_t const2_id = 3;
    uint32_t temp1_id = YNN_INVALID_VALUE_ID;
    uint32_t temp2_id = YNN_INVALID_VALUE_ID;

    float const1_val = 1.0f;
    float const2_val = 10.0f;

    SubgraphBuilder builder(4);
    builder.AddInput(type_of<float>(), 1, input_id)
        .AddOutput(type_of<float>(), 1, output_id)
        .AddTensor(type_of<float>(), {}, const1_id, &const1_val)
        .AddTensor(type_of<float>(), {}, const2_id, &const2_val)
        .AddTensor(type_of<float>(), 1, temp1_id)
        .AddTensor(type_of<float>(), 1, temp2_id);

    uint32_t one = builder.DefineScalar(1.0f);
    uint32_t two = builder.DefineScalar(2.0f);

    uint32_t temp3_id = YNN_INVALID_VALUE_ID;
    builder.AddTensor(type_of<float>(), 1, temp3_id);
    builder.AddBinary(ynn_binary_add, const1_id, one, temp1_id)
        .AddBinary(ynn_binary_add, const2_id, two, temp2_id)
        .AddBinary(ynn_binary_add, input_id, temp1_id, temp3_id)
        .AddBinary(ynn_binary_add, temp3_id, temp2_id, output_id);

    ynn_subgraph_t subgraph = builder.GetSubgraph();
    subgraph->fold_constants(nullptr);

    EXPECT_TRUE(subgraph->value(temp1_id).is_static());
    EXPECT_TRUE(subgraph->value(temp2_id).is_static());
  }

  // Run 2: const1 is same (temp1 cached), const2 is different (temp2 uncached).
  {
    const uint32_t input_id = 0;
    const uint32_t output_id = 1;
    uint32_t const1_id = 2;
    uint32_t const2_id = 3;
    uint32_t temp1_id = YNN_INVALID_VALUE_ID;
    uint32_t temp2_id = YNN_INVALID_VALUE_ID;

    float const1_val = 1.0f;   // same
    float const2_val = 20.0f;  // changed

    SubgraphBuilder builder(4);
    builder.AddInput(type_of<float>(), 1, input_id)
        .AddOutput(type_of<float>(), 1, output_id)
        .AddTensor(type_of<float>(), {}, const1_id, &const1_val)
        .AddTensor(type_of<float>(), {}, const2_id, &const2_val)
        .AddTensor(type_of<float>(), 1, temp1_id)
        .AddTensor(type_of<float>(), 1, temp2_id);

    uint32_t one = builder.DefineScalar(1.0f);
    uint32_t two = builder.DefineScalar(2.0f);

    uint32_t temp3_id = YNN_INVALID_VALUE_ID;
    builder.AddTensor(type_of<float>(), 1, temp3_id);
    builder.AddBinary(ynn_binary_add, const1_id, one, temp1_id)
        .AddBinary(ynn_binary_add, const2_id, two, temp2_id)
        .AddBinary(ynn_binary_add, input_id, temp1_id, temp3_id)
        .AddBinary(ynn_binary_add, temp3_id, temp2_id, output_id);

    ynn_subgraph_t subgraph = builder.GetSubgraph();
    // This should not crash!
    subgraph->fold_constants(nullptr);

    EXPECT_TRUE(subgraph->value(temp1_id).is_static());
    EXPECT_TRUE(subgraph->value(temp2_id).is_static());
    EXPECT_THAT(ValuesIn<float>(subgraph->value(temp1_id)), ElementsAre(2.0f));
    EXPECT_THAT(ValuesIn<float>(subgraph->value(temp2_id)), ElementsAre(22.0f));
  }
}

TEST(fold_constants, static_transpose) {
  constexpr float constant_values[] = {0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f};

  // A transpose that rearranges the contiguous (last) dimension cannot alias
  // its input (`alias == false`), so it executes a transpose kernel and should
  // be constant-folded.
  {
    const uint32_t input_id = 0;
    const uint32_t output_id = 1;
    uint32_t constant_id = 2;
    uint32_t transposed_id = YNN_INVALID_VALUE_ID;

    SubgraphBuilder builder(3);
    builder.AddInput(type_of<float>(), {3, 2}, input_id)
        .AddOutput(type_of<float>(), {3, 2}, output_id)
        .AddTensor(type_of<float>(), {2, 3}, constant_id, &constant_values)
        .AddTensor(type_of<float>(), {3, 2}, transposed_id);

    builder.AddTranspose({1, 0}, constant_id, transposed_id)
        .AddBinary(ynn_binary_add, input_id, transposed_id, output_id);

    ynn_subgraph_t subgraph = builder.GetSubgraph();
    subgraph->fold_constants(nullptr);

    const ynn_value& transposed = subgraph->value(transposed_id);
    EXPECT_TRUE(transposed.is_static());
    EXPECT_THAT(ValuesIn<float>(transposed),
                ElementsAre(0.0f, 3.0f, 1.0f, 4.0f, 2.0f, 5.0f));
    EXPECT_THAT(subgraph, HasValidNodeCount(1));
  }

  // A transpose that preserves the contiguous (last) dimension aliases its
  // input (`alias == true`), so it is free and should not be constant-folded.
  {
    const uint32_t input_id = 0;
    const uint32_t output_id = 1;
    uint32_t constant_id = 2;
    uint32_t transposed_id = YNN_INVALID_VALUE_ID;

    SubgraphBuilder builder(3);
    builder.AddInput(type_of<float>(), {3, 1, 2}, input_id)
        .AddOutput(type_of<float>(), {3, 1, 2}, output_id)
        .AddTensor(type_of<float>(), {1, 3, 2}, constant_id, &constant_values)
        .AddTensor(type_of<float>(), {3, 1, 2}, transposed_id);

    builder.AddTranspose({1, 0, 2}, constant_id, transposed_id)
        .AddBinary(ynn_binary_add, input_id, transposed_id, output_id);

    ynn_subgraph_t subgraph = builder.GetSubgraph();
    subgraph->fold_constants(nullptr);

    EXPECT_FALSE(subgraph->value(transposed_id).is_static());
    EXPECT_THAT(subgraph, HasValidNodeCount(2));
  }
}

TEST(fold_constants, symbolic_output_shape) {
  const uint32_t input_id = 0;
  const uint32_t output_id = 1;
  const uint32_t constant_id = 2;
  uint32_t broadcast_id = YNN_INVALID_VALUE_ID;
  uint32_t transposed_id = YNN_INVALID_VALUE_ID;

  SubgraphBuilder builder(3);
  builder.AddInput(type_of<float>(), 2, input_id)
      .AddOutput(type_of<float>(), 4, output_id)
      .AddScalar(1.0f, constant_id)
      .AddTensor(type_of<float>(), 2, broadcast_id)
      .AddTensor(type_of<float>(), 4, transposed_id);

  builder.AddStaticBroadcast({1, 1}, constant_id, broadcast_id);

  ynn_subgraph_t subgraph = builder.GetSubgraph();
  subgraph->value(broadcast_id).extents[0] =
      subgraph->value(input_id).extent(0);

  ynn_node transpose;
  define_transpose_a(*subgraph, transpose, /*tile_m=*/4, /*tile_k=*/4,
                     /*m_dim=*/1, broadcast_id, transposed_id);
  subgraph->add_node(std::move(transpose));

  builder.AddCopy(transposed_id, output_id);

  subgraph->fold_constants(nullptr);

  EXPECT_FALSE(subgraph->value(broadcast_id).is_static());
  EXPECT_FALSE(subgraph->value(transposed_id).is_static());
  EXPECT_THAT(subgraph, HasValidNodeCount(3));
}

TEST(fold_constants, cache_distinguishes_output_type) {
  constexpr float constant_values[] = {1.5f, -2.5f, 3.0f, 4.0f};

  uint32_t half_const_id = 1;
  uint32_t half_folded_id = YNN_INVALID_VALUE_ID;
  SubgraphBuilder half_builder(2);
  half_builder.AddOutput(type_of<half>(), {4}, 0)
      .AddTensor(type_of<float>(), {4}, half_const_id, &constant_values)
      .AddTensor(type_of<half>(), {4}, half_folded_id);
  half_builder.AddConvert(half_const_id, type_of<half>(), half_folded_id)
      .AddCopy(half_folded_id, 0);
  half_builder.GetSubgraph()->fold_constants(nullptr);

  uint32_t int_const_id = 1;
  uint32_t int_folded_id = YNN_INVALID_VALUE_ID;
  SubgraphBuilder int_builder(2);
  int_builder.AddOutput(type_of<int32_t>(), {4}, 0)
      .AddTensor(type_of<float>(), {4}, int_const_id, &constant_values)
      .AddTensor(type_of<int32_t>(), {4}, int_folded_id);
  int_builder.AddConvert(int_const_id, type_of<int32_t>(), int_folded_id)
      .AddCopy(int_folded_id, 0);
  int_builder.GetSubgraph()->fold_constants(nullptr);

  const ynn_value& half_val = half_builder.GetSubgraph()->value(half_folded_id);
  const ynn_value& int_val = int_builder.GetSubgraph()->value(int_folded_id);
  ASSERT_TRUE(half_val.is_static());
  ASSERT_TRUE(int_val.is_static());
  EXPECT_EQ(half_val.type, ynn_type_fp16);
  EXPECT_EQ(int_val.type, ynn_type_int32);
  EXPECT_EQ(half_val.data->elem_size, sizeof(half));
  EXPECT_EQ(int_val.data->elem_size, sizeof(int32_t));
  EXPECT_THAT(ValuesIn<int32_t>(int_val), ElementsAre(2, -2, 3, 4));
}

TEST(fold_constants, cache_distinguishes_output_shape) {
  uint32_t folded_1d_id = YNN_INVALID_VALUE_ID;
  SubgraphBuilder builder_1d(1);
  uint32_t begin_1d = builder_1d.DefineScalar(int32_t{0});
  uint32_t stride_1d = builder_1d.DefineScalar(int32_t{1});
  builder_1d.AddOutput(type_of<int32_t>(), {4}, 0)
      .AddTensor(type_of<int32_t>(), {4}, folded_1d_id)
      .AddIota(type_of<int32_t>(), {4}, begin_1d, stride_1d, folded_1d_id)
      .AddCopy(folded_1d_id, 0);
  builder_1d.GetSubgraph()->fold_constants(nullptr);

  uint32_t folded_2d_id = YNN_INVALID_VALUE_ID;
  SubgraphBuilder builder_2d(1);
  uint32_t begin_2d = builder_2d.DefineScalar(int32_t{0});
  uint32_t stride_2d = builder_2d.DefineScalar(int32_t{1});
  builder_2d.AddOutput(type_of<int32_t>(), {2, 4}, 0)
      .AddTensor(type_of<int32_t>(), {2, 4}, folded_2d_id)
      .AddIota(type_of<int32_t>(), {2, 4}, begin_2d, stride_2d, folded_2d_id)
      .AddCopy(folded_2d_id, 0);
  builder_2d.GetSubgraph()->fold_constants(nullptr);

  const ynn_value& val_1d = builder_1d.GetSubgraph()->value(folded_1d_id);
  const ynn_value& val_2d = builder_2d.GetSubgraph()->value(folded_2d_id);
  ASSERT_TRUE(val_1d.is_static());
  ASSERT_TRUE(val_2d.is_static());
  EXPECT_EQ(val_1d.data->rank, 1);
  EXPECT_EQ(val_2d.data->rank, 2);
  EXPECT_THAT(ValuesIn<int32_t>(val_1d), ElementsAre(0, 1, 2, 3));
  EXPECT_THAT(ValuesIn<int32_t>(val_2d),
              ElementsAre(0, 1, 2, 3, 1, 2, 3, 4));
}

TEST(fold_constants, cache_distinguishes_pack_b_params) {
  constexpr size_t K2 = 4;
  constexpr size_t K1 = 16;
  constexpr size_t N = 16;
  constexpr size_t M = 4;
  std::vector<int8_t> b_data(K2 * K1 * N, 1);

  SubgraphBuilder builder1(2);
  uint32_t b1_id = YNN_INVALID_VALUE_ID;
  builder1.AddInput(type_of<int8_t>(), {K2, M, K1}, 0)
      .AddOutput(type_of<int32_t>(), {K2, M, N}, 1)
      .AddTensor(type_of<int8_t>(), {K2, K1, N}, b1_id, b_data.data())
      .AddDot(/*num_k_dims=*/1, 0, b1_id, YNN_INVALID_VALUE_ID, 1);
  ynn_subgraph_t sg1 = builder1.GetSubgraph();
  ASSERT_EQ(sg1->fold_constants(nullptr), ynn_status_success);

  SubgraphBuilder builder2(2);
  uint32_t b2_id = YNN_INVALID_VALUE_ID;
  builder2.AddInput(type_of<int8_t>(), {M, K2, K1}, 0)
      .AddOutput(type_of<int32_t>(), {M, N}, 1)
      .AddTensor(type_of<int8_t>(), {K2, K1, N}, b2_id, b_data.data())
      .AddDot(/*num_k_dims=*/2, 0, b2_id, YNN_INVALID_VALUE_ID, 1);
  ynn_subgraph_t sg2 = builder2.GetSubgraph();
  ASSERT_EQ(sg2->fold_constants(nullptr), ynn_status_success);

  const ynn_node& dot1 = ProducerOf(1, *sg1);
  const ynn_node& dot2 = ProducerOf(1, *sg2);
  const ynn_value& packed_b1 = sg1->value(dot1.inputs[1]);
  const ynn_value& packed_b2 = sg2->value(dot2.inputs[1]);
  ASSERT_TRUE(packed_b1.is_static());
  ASSERT_TRUE(packed_b2.is_static());
  EXPECT_NE(packed_b1.data.get(), packed_b2.data.get());
}

}  // namespace ynn
