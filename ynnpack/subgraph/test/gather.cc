// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <array>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <numeric>
#include <tuple>
#include <type_traits>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "ynnpack/base/bfloat16.h"
#include "ynnpack/base/half.h"
#include "ynnpack/base/test/buffer.h"
#include "ynnpack/base/type.h"
#include "ynnpack/include/ynnpack.h"
#include "ynnpack/subgraph/test/subgraph_builder.h"

namespace ynn {
namespace {

template <typename T, typename IndexType>
void TestGather(std::vector<int32_t> axes, std::vector<size_t> input_shape,
                std::vector<T> input_data, std::vector<size_t> index_shape,
                std::vector<int32_t> index_data,
                std::vector<size_t> expected_output_shape,
                std::vector<T> expected_output_data,
                bool expect_success = true) {
  SubgraphBuilder subgraph(3);
  uint32_t input_id = 0;
  uint32_t index_id = 1;
  uint32_t output_id = 2;

  subgraph.AddInput(type_of<T>(), input_shape, input_id)
      .AddInput(type_of<IndexType>(), index_shape, index_id)
      .AddOutput(type_of<T>(), expected_output_shape, output_id);

  subgraph.AddGather(axes, expected_output_shape.size(), input_id, index_id,
                     output_id);

  Runtime runtime(subgraph.GetSubgraph());
  ASSERT_EQ(runtime.Status(), ynn_status_success);

  runtime.ReshapeExternalTensor(input_shape, input_data.data(), input_id);

  Buffer<IndexType> index_buffer(index_data.size());
  for (size_t i = 0; i < index_data.size(); ++i) {
    index_buffer[i] = index_data[i];
  }
  runtime.ReshapeExternalTensor(index_shape, index_buffer.data(), index_id);

  runtime.ReshapeRuntime();
  ASSERT_EQ(runtime.Status(), ynn_status_success);

  ASSERT_EQ(runtime.GetExternalTensorShape(output_id), expected_output_shape);

  if (expect_success) {
    std::vector<T> output_data(expected_output_data.size());
    runtime.SetupExternalTensor(output_data.data(), output_id).InvokeRuntime();
    EXPECT_EQ(runtime.Status(), ynn_status_success);
    EXPECT_THAT(output_data, testing::ElementsAreArray(expected_output_data));
  } else {
    size_t output_size = std::accumulate(expected_output_shape.begin(),
                                         expected_output_shape.end(), size_t{1},
                                         std::multiplies<size_t>());
    std::vector<T> output_data(output_size);
    runtime.SetupExternalTensor(output_data.data(), output_id).InvokeRuntime();
    EXPECT_EQ(runtime.Status(), ynn_status_error);
  }
}

template <typename T, typename IndexType>
constexpr bool is_supported_sub_byte() {
  if constexpr (type_info<IndexType>::element_count() == 1) {
    return true;
  }
  return sizeof(T) == 1;
}

template <typename T>
class GatherTest : public ::testing::Test {
 protected:
  using InputType = typename std::tuple_element<0, T>::type;
  using IndexType = typename std::tuple_element<1, T>::type;
};

using GatherTestTypes = ::testing::Types<
    std::tuple<int8_t, int8_t>, std::tuple<int8_t, uint8_t>,
    std::tuple<int8_t, int32_t>, std::tuple<int8_t, uint2x4>,
    std::tuple<int8_t, uint4x2>, std::tuple<int32_t, int8_t>,
    std::tuple<int32_t, uint8_t>, std::tuple<int32_t, int32_t>,
    std::tuple<int32_t, uint2x4>, std::tuple<int32_t, uint4x2>,
    std::tuple<float, int8_t>, std::tuple<float, uint8_t>,
    std::tuple<float, int32_t>, std::tuple<float, uint2x4>,
    std::tuple<float, uint4x2>, std::tuple<half, int8_t>,
    std::tuple<half, uint8_t>, std::tuple<half, int32_t>,
    std::tuple<half, uint2x4>, std::tuple<half, uint4x2>,
    std::tuple<bfloat16, int8_t>, std::tuple<bfloat16, uint8_t>,
    std::tuple<bfloat16, int32_t>, std::tuple<bfloat16, uint2x4>,
    std::tuple<bfloat16, uint4x2> >;
TYPED_TEST_SUITE(GatherTest, GatherTestTypes);

TYPED_TEST(GatherTest, Index0D) {
  using T = typename TestFixture::InputType;
  using IndexType = typename TestFixture::IndexType;

  if constexpr (type_info<IndexType>::element_count() != 1) {
    GTEST_SKIP() << "0D index not supported for sub-byte types";
  }

  // 4. 1D input, 0D index
  TestGather<T, IndexType>(
      /*axes=*/{0},
      /*input_shape=*/{3}, /*input_data=*/{1, 2, 3},
      /*index_shape=*/{}, /*index_data=*/{1},
      /*expected_output_shape=*/{}, /*expected_output_data=*/{2});
}

TYPED_TEST(GatherTest, Index1D) {
  using T = typename TestFixture::InputType;
  using IndexType = typename TestFixture::IndexType;

  if constexpr (!is_supported_sub_byte<T, IndexType>()) {
    GTEST_SKIP() << "Sub-byte index only supported with 8-bit elements";
  }

  // 1D input, 1D index (aligned)
  TestGather<T, IndexType>(
      /*axes=*/{0},
      /*input_shape=*/{3}, /*input_data=*/{1, 2, 3},
      /*index_shape=*/{4}, /*index_data=*/{1, 2, 0, 1},
      /*expected_output_shape=*/{4}, /*expected_output_data=*/{2, 3, 1, 2});
}

TYPED_TEST(GatherTest, Index2D) {
  using T = typename TestFixture::InputType;
  using IndexType = typename TestFixture::IndexType;

  if constexpr (!is_supported_sub_byte<T, IndexType>()) {
    GTEST_SKIP() << "Sub-byte index only supported with 8-bit elements";
  }

  // 1D input, 2D index (aligned)
  TestGather<T, IndexType>(
      /*axes=*/{0},
      /*input_shape=*/{3}, /*input_data=*/{1, 2, 3},
      /*index_shape=*/{2, 4}, /*index_data=*/{1, 2, 0, 2, 0, 1, 1, 2},
      /*expected_output_shape=*/{2, 4},
      /*expected_output_data=*/{2, 3, 1, 3, 1, 2, 2, 3});
}

TYPED_TEST(GatherTest, Input2DIndex2D) {
  using T = typename TestFixture::InputType;
  using IndexType = typename TestFixture::IndexType;

  if constexpr (type_info<IndexType>::element_count() != 1) {
    GTEST_SKIP() << "Multi-dimensional gather not supported for sub-byte types "
                    "(requires LUT path)";
  }

  // 2D input, 2D index, axis = 0 (aligned)
  TestGather<T, IndexType>(
      /*axes=*/{0},
      /*input_shape=*/{3, 1, 4},
      /*input_data=*/{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12},
      /*index_shape=*/{2, 3, 4},
      /*index_data=*/
      {2, 0, 1, 0, 1, 2, 0, 1, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0},
      /*expected_output_shape=*/{2, 3, 4},
      /*expected_output_data=*/
      {9, 2,  7, 4, 5, 10, 3, 8,  1, 6, 11, 4,
       5, 10, 3, 8, 9, 2,  7, 12, 1, 6, 11, 4});
}

TYPED_TEST(GatherTest, IndexBroadcasting) {
  using T = typename TestFixture::InputType;
  using IndexType = typename TestFixture::IndexType;

  if constexpr (type_info<IndexType>::element_count() != 1) {
    GTEST_SKIP() << "Index broadcasting not supported for sub-byte types "
                    "(requires LUT path)";
  }

  // Index broadcasting (aligned)
  TestGather<T, IndexType>(
      /*axes=*/{0},
      /*input_shape=*/{3, 2, 4},
      /*input_data=*/
      {1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12,
       13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24},
      /*index_shape=*/{2, 1, 4}, /*index_data=*/{2, 0, 1, 0, 1, 2, 0, 1},
      /*expected_output_shape=*/{2, 2, 4},
      /*expected_output_data=*/
      {17, 2, 11, 4, 21, 6, 15, 8, 9, 18, 3, 12, 13, 22, 7, 16});
}

TYPED_TEST(GatherTest, InputBroadcasting) {
  using T = typename TestFixture::InputType;
  using IndexType = typename TestFixture::IndexType;

  if constexpr (type_info<IndexType>::element_count() != 1) {
    GTEST_SKIP() << "Input broadcasting not supported for sub-byte types "
                    "(requires LUT path)";
  }

  // Input broadcasting (aligned)
  TestGather<T, IndexType>(
      /*axes=*/{0},
      /*input_shape=*/{3, 1, 4},
      /*input_data=*/{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12},
      /*index_shape=*/{2, 3, 4},
      /*index_data=*/
      {2, 0, 1, 0, 1, 2, 0, 1, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2, 0},
      /*expected_output_shape=*/{2, 3, 4},
      /*expected_output_data=*/
      {9, 2,  7, 4, 5, 10, 3, 8,  1, 6, 11, 4,
       5, 10, 3, 8, 9, 2,  7, 12, 1, 6, 11, 4});
}

TYPED_TEST(GatherTest, OutOfBounds) {
  using T = typename TestFixture::InputType;
  using IndexType = typename TestFixture::IndexType;

  if constexpr (!is_supported_sub_byte<T, IndexType>()) {
    GTEST_SKIP() << "Sub-byte index only supported with 8-bit elements";
  }

  // 1D input, 1D index (aligned). Fast path (scalar gather) if axis is 0.
  // input_shape = {3}, valid indexes are 0, 1, 2.
  TestGather<T, IndexType>(
      /*axes=*/{0},
      /*input_shape=*/{3}, /*input_data=*/{1, 2, 3},
      /*index_shape=*/{4}, /*index_data=*/{1, 3, 0, 1},  // 3 is out of bounds
      /*expected_output_shape=*/{4}, /*expected_output_data=*/{},
      /*expect_success=*/false);

  if constexpr (std::is_signed_v<IndexType>) {
    TestGather<T, IndexType>(
        /*axes=*/{0},
        /*input_shape=*/{3}, /*input_data=*/{1, 2, 3},
        /*index_shape=*/{4},
        /*index_data=*/{-1, 1, 0, 1},  // -1 is out of bounds
        /*expected_output_shape=*/{4}, /*expected_output_data=*/{},
        /*expect_success=*/false);
  }
}

TYPED_TEST(GatherTest, OutOfBounds2D) {
  using T = typename TestFixture::InputType;
  using IndexType = typename TestFixture::IndexType;

  if constexpr (type_info<IndexType>::element_count() != 1) {
    GTEST_SKIP() << "2D OutOfBounds not supported for sub-byte types (requires "
                    "LUT path)";
  }

  // 2D input, 2D index, axis = 0 (aligned).
  // input_shape = {3, 1, 4}, valid indexes for axis 0 are 0, 1, 2.
  TestGather<T, IndexType>(
      /*axes=*/{0},
      /*input_shape=*/{3, 1, 4},
      /*input_data=*/{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12},
      /*index_shape=*/{2, 2, 4},
      /*index_data=*/
      {1, 0, 0, 0, 3, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 0},  // 3 is out of bounds
      /*expected_output_shape=*/{2, 2, 4}, /*expected_output_data=*/{},
      /*expect_success=*/false);

  // 2D input, 2D index, axis = 1 (aligned).
  // input_shape = {3, 3, 4}, valid indexes for axis 1 are 0, 1, 2.
  TestGather<T, IndexType>(
      /*axes=*/{1},
      /*input_shape=*/{3, 3, 4},
      /*input_data=*/
      {1,  2,  3,  4,  5,  6,  7,  8,  9,  10, 11, 12, 13, 14, 15, 16, 17, 18,
       19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35, 36},
      /*index_shape=*/{1, 3, 4},
      /*index_data=*/{0, 3, 1, 1, 2, 0, 0, 0, 0, 0, 0, 0},  // 3 is out of
                                                            // bounds
      /*expected_output_shape=*/{3, 3, 4}, /*expected_output_data=*/{},
      /*expect_success=*/false);
}

TYPED_TEST(GatherTest, MultiAxis) {
  using T = typename TestFixture::InputType;
  using IndexType = typename TestFixture::IndexType;

  if constexpr (type_info<IndexType>::element_count() != 1) {
    GTEST_SKIP() << "MultiAxis gather not supported for sub-byte types "
                    "(requires LUT path)";
  }

  // 2D input, gather both axes (aligned).
  // input_shape = {3, 3}
  // axes = {0, 1}
  TestGather<T, IndexType>(
      /*axes=*/{0, 1},
      /*input_shape=*/{3, 3},
      /*input_data=*/{1, 2, 3, 4, 5, 6, 7, 8, 9},
      /*index_shape=*/{2, 4},
      /*index_data=*/
      {
          0, 1, 2, 0,  // coordinates for axis 0
          0, 2, 1, 1   // coordinates for axis 1
      },
      /*expected_output_shape=*/{4},
      /*expected_output_data=*/{1, 6, 8, 2});
}

TYPED_TEST(GatherTest, NumAxes1NotOmitted) {
  using T = typename TestFixture::InputType;
  using IndexType = typename TestFixture::IndexType;

  if constexpr (!is_supported_sub_byte<T, IndexType>()) {
    GTEST_SKIP() << "Sub-byte index only supported with 8-bit elements";
  }

  // 1D input, 1D gather (num_axes = 1), index has coordinate dim of size 1
  // (aligned). input_shape = {3} axes = {0}
  TestGather<T, IndexType>(
      /*axes=*/{0},
      /*input_shape=*/{3}, /*input_data=*/{1, 2, 3},
      /*index_shape=*/{1, 4}, /*index_data=*/{2, 0, 1, 2},
      /*expected_output_shape=*/{4}, /*expected_output_data=*/{3, 1, 2, 3});
}

TEST(GatherTestStandalone, IntermediateInputAndOutputPermutation) {
  // Graph: input -> Negate -> temp1 -> Gather -> temp2 -> Abs -> output
  uint32_t temp1_id = YNN_INVALID_VALUE_ID;
  uint32_t temp2_id = YNN_INVALID_VALUE_ID;
  const uint32_t input_id = 0;
  const uint32_t index_id = 1;
  const uint32_t output_id = 2;

  SubgraphBuilder subgraph(3);
  subgraph.AddInput(ynn_type_fp32, {4}, input_id)
      .AddInput(ynn_type_int32, {4}, index_id)
      .AddTensor(ynn_type_fp32, 1, temp1_id)
      .AddTensor(ynn_type_fp32, 1, temp2_id)
      .AddOutput(ynn_type_fp32, {4}, output_id);

  subgraph.AddUnary(ynn_unary_negate, input_id, temp1_id);
  subgraph.AddGather({0}, /*output_rank=*/1, temp1_id, index_id, temp2_id);
  subgraph.AddUnary(ynn_unary_abs, temp2_id, output_id);

  Runtime runtime(subgraph.GetSubgraph());
  ASSERT_EQ(runtime.Status(), ynn_status_success);

  std::vector<float> input_data = {10.0f, 20.0f, 30.0f, 40.0f};
  std::vector<int32_t> index_data = {3, 2, 1, 0};
  runtime.ReshapeExternalTensor({4}, input_data.data(), input_id);
  runtime.ReshapeExternalTensor({4}, index_data.data(), index_id);

  runtime.ReshapeRuntime();
  ASSERT_EQ(runtime.Status(), ynn_status_success);

  std::vector<float> output_data(4);
  runtime.SetupExternalTensor(output_data.data(), output_id).InvokeRuntime();
  EXPECT_EQ(runtime.Status(), ynn_status_success);

  std::vector<float> expected_output = {40.0f, 30.0f, 20.0f, 10.0f};
  EXPECT_THAT(output_data, testing::ElementsAreArray(expected_output));
}

TEST(GatherTestStandalone, MultiAxisInferredOutputRank) {
  const uint32_t input_id = 0;
  const uint32_t index_id = 1;
  uint32_t gathered_id = YNN_INVALID_VALUE_ID;
  const uint32_t output_id = 2;

  SubgraphBuilder subgraph(3);
  subgraph.AddInput(ynn_type_fp32, {3, 3}, input_id)
      .AddInput(ynn_type_int32, {2, 4}, index_id)
      .AddOutput(ynn_type_fp32, {4}, output_id);

  const int32_t axes[2] = {0, 1};
  ASSERT_EQ(ynn_define_gather(subgraph.GetSubgraph(), /*num_axes=*/2, axes,
                              YNN_UNKNOWN_RANK, input_id, index_id,
                              &gathered_id, /*flags=*/0),
            ynn_status_success);
  subgraph.AddCopy(gathered_id, output_id);

  Runtime runtime(subgraph.GetSubgraph());
  ASSERT_EQ(runtime.Status(), ynn_status_success);

  std::vector<float> input_data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f,
                                   6.0f, 7.0f, 8.0f, 9.0f};
  std::vector<int32_t> index_data = {0, 1, 2, 0, 0, 2, 1, 1};
  std::vector<float> output_data(4, 0.0f);
  runtime.ReshapeExternalTensor({3, 3}, input_data.data(), input_id)
      .ReshapeExternalTensor({2, 4}, index_data.data(), index_id)
      .SetupExternalTensor(output_data.data(), output_id)
      .ReshapeRuntime()
      .InvokeRuntime();
  ASSERT_EQ(runtime.Status(), ynn_status_success);
  EXPECT_THAT(output_data, testing::ElementsAre(1.0f, 6.0f, 8.0f, 2.0f));

  SubgraphBuilder bad_subgraph(2);
  uint32_t bad_out_id = YNN_INVALID_VALUE_ID;
  bad_subgraph.AddInput(ynn_type_fp32, {3, 3, 5}, 0)
      .AddInput(ynn_type_int32, {4}, 1);
  EXPECT_EQ(ynn_define_gather(bad_subgraph.GetSubgraph(), /*num_axes=*/2, axes,
                              YNN_UNKNOWN_RANK, 0, 1, &bad_out_id,
                              /*flags=*/0),
            ynn_status_invalid_parameter);

  SubgraphBuilder bad_coord_subgraph(3);
  uint32_t bad_coord_id = YNN_INVALID_VALUE_ID;
  bad_coord_subgraph.AddInput(ynn_type_fp32, {3, 3}, 0)
      .AddInput(ynn_type_int32, TensorShape(2), 1)
      .AddOutput(ynn_type_fp32, TensorShape(1), 2);
  ASSERT_EQ(ynn_define_gather(bad_coord_subgraph.GetSubgraph(), /*num_axes=*/2,
                              axes, YNN_UNKNOWN_RANK, 0, 1, &bad_coord_id,
                              /*flags=*/0),
            ynn_status_success);
  bad_coord_subgraph.AddCopy(bad_coord_id, 2);
  std::vector<int32_t> short_index(4, 0);
  Runtime bad_coord_runtime(bad_coord_subgraph.GetSubgraph());
  bad_coord_runtime.ReshapeExternalTensor({3, 3}, input_data.data(), 0)
      .ReshapeExternalTensor({1, 4}, short_index.data(), 1)
      .SetupExternalTensor(output_data.data(), 2)
      .ReshapeRuntime();
  EXPECT_NE(bad_coord_runtime.Status(), ynn_status_success);

  SubgraphBuilder static_bad_coord(2);
  uint32_t static_bad_id = YNN_INVALID_VALUE_ID;
  static_bad_coord.AddInput(ynn_type_fp32, {3, 3}, 0)
      .AddInput(ynn_type_int32, {1, 4}, 1);
  EXPECT_EQ(ynn_define_gather(static_bad_coord.GetSubgraph(), /*num_axes=*/2,
                              axes, YNN_UNKNOWN_RANK, 0, 1, &static_bad_id,
                              /*flags=*/0),
            ynn_status_invalid_parameter);

  for (const auto& dup_axes :
       {std::array<int32_t, 2>{0, 0}, std::array<int32_t, 2>{0, -2}}) {
    SubgraphBuilder dup_subgraph(2);
    uint32_t dup_out_id = YNN_INVALID_VALUE_ID;
    dup_subgraph.AddInput(ynn_type_fp32, {3, 3}, 0)
        .AddInput(ynn_type_int32, {2, 4}, 1);
    EXPECT_EQ(ynn_define_gather(dup_subgraph.GetSubgraph(), /*num_axes=*/2,
                                dup_axes.data(), YNN_UNKNOWN_RANK, 0, 1,
                                &dup_out_id, /*flags=*/0),
              ynn_status_invalid_parameter);
    EXPECT_EQ(dup_out_id, YNN_INVALID_VALUE_ID);
  }
}

TEST(GatherTestStandalone, InternalInputLowerOutputRankAndValidation) {
  // Internal 2D input gathered into 1D output (`output_rank < input.rank()`),
  // both via multi-axis gather and via 0D scalar index on axis 0.
  {
    SubgraphBuilder subgraph(3);
    uint32_t neg_id = YNN_INVALID_VALUE_ID;
    subgraph.AddInput(ynn_type_fp32, {3, 3}, 0)
        .AddInput(ynn_type_int32, {2, 4}, 1)
        .AddTensor(ynn_type_fp32, 2, neg_id)
        .AddOutput(ynn_type_fp32, {4}, 2);
    subgraph.AddUnary(ynn_unary_negate, 0, neg_id);
    const int32_t axes[2] = {0, 1};
    uint32_t gathered_id = YNN_INVALID_VALUE_ID;
    ASSERT_EQ(ynn_define_gather(subgraph.GetSubgraph(), /*num_axes=*/2, axes,
                                YNN_UNKNOWN_RANK, neg_id, 1, &gathered_id, 0),
              ynn_status_success);
    subgraph.AddCopy(gathered_id, 2);

    std::vector<float> input_data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f,
                                     6.0f, 7.0f, 8.0f, 9.0f};
    std::vector<int32_t> index_data = {0, 1, 2, 0, 0, 2, 1, 1};
    std::vector<float> output_data(4, 0.0f);
    Runtime runtime(subgraph.GetSubgraph());
    ASSERT_EQ(runtime.Status(), ynn_status_success);
    runtime.ReshapeExternalTensor({3, 3}, input_data.data(), 0)
        .ReshapeExternalTensor({2, 4}, index_data.data(), 1)
        .SetupExternalTensor(output_data.data(), 2)
        .ReshapeRuntime()
        .InvokeRuntime();
    ASSERT_EQ(runtime.Status(), ynn_status_success);
    EXPECT_THAT(output_data, testing::ElementsAre(-1.0f, -6.0f, -8.0f, -2.0f));
  }
  {
    SubgraphBuilder subgraph(3);
    uint32_t neg_id = YNN_INVALID_VALUE_ID;
    subgraph.AddInput(ynn_type_fp32, {3, 4}, 0)
        .AddInput(ynn_type_int32, {}, 1)
        .AddTensor(ynn_type_fp32, 2, neg_id)
        .AddOutput(ynn_type_fp32, {4}, 2);
    subgraph.AddUnary(ynn_unary_negate, 0, neg_id);
    const int32_t axis0 = 0;
    uint32_t gathered_id = YNN_INVALID_VALUE_ID;
    ASSERT_EQ(ynn_define_gather(subgraph.GetSubgraph(), /*num_axes=*/1, &axis0,
                                YNN_UNKNOWN_RANK, neg_id, 1, &gathered_id, 0),
              ynn_status_success);
    subgraph.AddCopy(gathered_id, 2);

    std::vector<float> input_data = {1.0f, 2.0f, 3.0f, 4.0f,  5.0f,  6.0f,
                                     7.0f, 8.0f, 9.0f, 10.0f, 11.0f, 12.0f};
    int32_t index_scalar = 2;
    std::vector<float> output_data(4, 0.0f);
    Runtime runtime(subgraph.GetSubgraph());
    ASSERT_EQ(runtime.Status(), ynn_status_success);
    runtime.ReshapeExternalTensor({3, 4}, input_data.data(), 0)
        .ReshapeExternalTensor({}, &index_scalar, 1)
        .SetupExternalTensor(output_data.data(), 2)
        .ReshapeRuntime()
        .InvokeRuntime();
    ASSERT_EQ(runtime.Status(), ynn_status_success);
    EXPECT_THAT(output_data,
                testing::ElementsAre(-9.0f, -10.0f, -11.0f, -12.0f));
  }

  // Explicit `output_rank` rank/coordinate validation for `num_axes == 1`.
  {
    const int32_t axis0 = 0;
    SubgraphBuilder rank_too_high(2);
    uint32_t out_id = YNN_INVALID_VALUE_ID;
    rank_too_high.AddInput(ynn_type_int8, {3}, 0)
        .AddInput(ynn_type_int32, {1, 2, 4}, 1);
    EXPECT_EQ(ynn_define_gather(rank_too_high.GetSubgraph(), /*num_axes=*/1,
                                &axis0, /*output_rank=*/1, 0, 1, &out_id, 0),
              ynn_status_invalid_parameter);

    SubgraphBuilder bad_single_coord(2);
    bad_single_coord.AddInput(ynn_type_int8, {3}, 0)
        .AddInput(ynn_type_int32, {2, 4}, 1);
    EXPECT_EQ(ynn_define_gather(bad_single_coord.GetSubgraph(), /*num_axes=*/1,
                                &axis0, /*output_rank=*/1, 0, 1, &out_id, 0),
              ynn_status_invalid_parameter);

    SubgraphBuilder ungathered_exceeds(2);
    ungathered_exceeds.AddInput(ynn_type_fp32, {5, 3}, 0)
        .AddInput(ynn_type_int32, {4}, 1);
    const int32_t axis1 = 1;
    EXPECT_EQ(
        ynn_define_gather(ungathered_exceeds.GetSubgraph(), /*num_axes=*/1,
                          &axis1, /*output_rank=*/1, 0, 1, &out_id, 0),
        ynn_status_invalid_parameter);

    for (size_t bad_rank :
         {size_t{YNN_MAX_TENSOR_RANK + 1},
          std::numeric_limits<size_t>::max() - 1, size_t{1} << 31}) {
      SubgraphBuilder bad_rank_subgraph(2);
      uint32_t bad_rank_out = YNN_INVALID_VALUE_ID;
      bad_rank_subgraph.AddInput(ynn_type_fp32, {3}, 0)
          .AddInput(ynn_type_int32, {}, 1);
      EXPECT_EQ(ynn_define_gather(bad_rank_subgraph.GetSubgraph(),
                                  /*num_axes=*/1, &axis0, bad_rank, 0, 1,
                                  &bad_rank_out, 0),
                ynn_status_unsupported_parameter);
      EXPECT_EQ(bad_rank_out, YNN_INVALID_VALUE_ID);
    }
  }
}

TEST(GatherTestStandalone, ScalarOutputAndExternalOutputRankInference) {
  // Explicit `output_rank = 0` with `num_axes = 1` and `{1}` coordinate index
  // must preserve 0D scalar output (both float general path and int8 LUT path).
  TestGather<float, int32_t>(
      /*axes=*/{0}, /*input_shape=*/{3}, /*input_data=*/{1.0f, 2.0f, 3.0f},
      /*index_shape=*/{1}, /*index_data=*/{2},
      /*expected_output_shape=*/{}, /*expected_output_data=*/{3.0f});
  TestGather<int8_t, int32_t>(
      /*axes=*/{0}, /*input_shape=*/{3}, /*input_data=*/{10, 20, 30},
      /*index_shape=*/{1}, /*index_data=*/{1},
      /*expected_output_shape=*/{}, /*expected_output_data=*/{20});

  // `YNN_UNKNOWN_RANK` into a pre-defined external output preserves the
  // output's declared rank when `index` includes a non-omitted size-1
  // coordinate dim.
  {
    const uint32_t input_id = 0;
    const uint32_t index_id = 1;
    uint32_t output_id = 2;
    SubgraphBuilder subgraph(3);
    subgraph.AddInput(ynn_type_fp32, {3}, input_id)
        .AddInput(ynn_type_int32, {1, 4}, index_id)
        .AddOutput(ynn_type_fp32, {4}, output_id);
    const int32_t axis0 = 0;
    ASSERT_EQ(
        ynn_define_gather(subgraph.GetSubgraph(), /*num_axes=*/1, &axis0,
                          YNN_UNKNOWN_RANK, input_id, index_id, &output_id, 0),
        ynn_status_success);

    Runtime runtime(subgraph.GetSubgraph());
    ASSERT_EQ(runtime.Status(), ynn_status_success);
    std::vector<float> input_data = {10.0f, 20.0f, 30.0f};
    std::vector<int32_t> index_data = {2, 0, 1, 2};
    std::vector<float> output_data(4, 0.0f);
    runtime.ReshapeExternalTensor({3}, input_data.data(), input_id)
        .ReshapeExternalTensor({1, 4}, index_data.data(), index_id)
        .ReshapeRuntime();
    ASSERT_EQ(runtime.Status(), ynn_status_success);
    EXPECT_THAT(runtime.GetExternalTensorShape(output_id),
                testing::ElementsAre(4));
    runtime.SetupExternalTensor(output_data.data(), output_id).InvokeRuntime();
    ASSERT_EQ(runtime.Status(), ynn_status_success);
    EXPECT_THAT(output_data, testing::ElementsAre(30.0f, 10.0f, 20.0f, 30.0f));
  }
  {
    const uint32_t input_id = 0;
    const uint32_t index_id = 1;
    uint32_t output_id = 2;
    SubgraphBuilder subgraph(3);
    subgraph.AddInput(ynn_type_fp32, {3}, input_id)
        .AddInput(ynn_type_int32, {1}, index_id)
        .AddOutput(ynn_type_fp32, {}, output_id);
    const int32_t axis0 = 0;
    ASSERT_EQ(
        ynn_define_gather(subgraph.GetSubgraph(), /*num_axes=*/1, &axis0,
                          YNN_UNKNOWN_RANK, input_id, index_id, &output_id, 0),
        ynn_status_success);

    Runtime runtime(subgraph.GetSubgraph());
    ASSERT_EQ(runtime.Status(), ynn_status_success);
    std::vector<float> input_data = {10.0f, 20.0f, 30.0f};
    int32_t index_data = 1;
    float output_data = 0.0f;
    runtime.ReshapeExternalTensor({3}, input_data.data(), input_id)
        .ReshapeExternalTensor({1}, &index_data, index_id)
        .ReshapeRuntime();
    ASSERT_EQ(runtime.Status(), ynn_status_success);
    EXPECT_TRUE(runtime.GetExternalTensorShape(output_id).empty());
    runtime.SetupExternalTensor(&output_data, output_id).InvokeRuntime();
    ASSERT_EQ(runtime.Status(), ynn_status_success);
    EXPECT_EQ(output_data, 20.0f);
  }
}

TEST(GatherTestStandalone, LeadingUnitInputAndOutputRankMismatch) {
  // Multi-axis gather with leading extent-1 ungathered dim in `input` and
  // `YNN_UNKNOWN_RANK` into an internal output drops the trailing size-1 dim
  // to match `index.rank() - 1`.
  {
    const uint32_t input_id = 0;
    const uint32_t index_id = 1;
    uint32_t gathered_id = YNN_INVALID_VALUE_ID;
    const uint32_t output_id = 2;
    SubgraphBuilder subgraph(3);
    subgraph.AddInput(ynn_type_fp32, {1, 3, 3}, input_id)
        .AddInput(ynn_type_int32, {2, 4}, index_id)
        .AddOutput(ynn_type_fp32, {4}, output_id);
    const int32_t axes[2] = {1, 2};
    ASSERT_EQ(ynn_define_gather(subgraph.GetSubgraph(), /*num_axes=*/2, axes,
                                YNN_UNKNOWN_RANK, input_id, index_id,
                                &gathered_id, /*flags=*/0),
              ynn_status_success);
    subgraph.AddCopy(gathered_id, output_id);

    std::vector<float> input_data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f,
                                     6.0f, 7.0f, 8.0f, 9.0f};
    std::vector<int32_t> index_data = {0, 1, 2, 0, 0, 2, 1, 1};
    std::vector<float> output_data(4, 0.0f);
    Runtime runtime(subgraph.GetSubgraph());
    ASSERT_EQ(runtime.Status(), ynn_status_success);
    runtime.ReshapeExternalTensor({1, 3, 3}, input_data.data(), input_id)
        .ReshapeExternalTensor({2, 4}, index_data.data(), index_id)
        .ReshapeRuntime();
    ASSERT_EQ(runtime.Status(), ynn_status_success);
    EXPECT_THAT(runtime.GetExternalTensorShape(output_id),
                testing::ElementsAre(4));
    runtime.SetupExternalTensor(output_data.data(), output_id).InvokeRuntime();
    ASSERT_EQ(runtime.Status(), ynn_status_success);
    EXPECT_THAT(output_data, testing::ElementsAre(1.0f, 6.0f, 8.0f, 2.0f));
  }

  // Explicit `output_rank` must match a pre-defined external output's rank.
  {
    SubgraphBuilder mismatch_subgraph(3);
    uint32_t out_id = 2;
    mismatch_subgraph.AddInput(ynn_type_fp32, {3}, 0)
        .AddInput(ynn_type_int32, {4}, 1)
        .AddOutput(ynn_type_fp32, {4}, out_id);
    const int32_t axis0 = 0;
    EXPECT_EQ(ynn_define_gather(mismatch_subgraph.GetSubgraph(), /*num_axes=*/1,
                                &axis0, /*output_rank=*/2, 0, 1, &out_id,
                                /*flags=*/0),
              ynn_status_invalid_parameter);
  }
}

}  // namespace
}  // namespace ynn
