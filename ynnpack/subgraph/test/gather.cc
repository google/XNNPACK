// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <tuple>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "ynnpack/base/bfloat16.h"
#include "ynnpack/base/half.h"
#include "ynnpack/base/type.h"
#include "ynnpack/include/ynnpack.h"
#include "ynnpack/subgraph/test/subgraph_builder.h"

namespace ynn {
namespace {

template <typename T, typename IndexType>
void TestGather(int32_t axis, std::vector<size_t> input_shape,
                std::vector<T> input_data, std::vector<size_t> index_shape,
                std::vector<IndexType> index_data,
                std::vector<size_t> expected_output_shape,
                std::vector<T> expected_output_data) {
  SubgraphBuilder subgraph(3);
  uint32_t input_id = 0;
  uint32_t index_id = 1;
  uint32_t output_id = 2;

  subgraph.AddInput(type_of<T>(), input_shape, input_id)
      .AddInput(type_of<IndexType>(), index_shape, index_id)
      .AddOutput(type_of<T>(), expected_output_shape, output_id)
      .AddGather(axis, input_id, index_id, output_id);

  Runtime runtime(subgraph.GetSubgraph());
  ASSERT_EQ(runtime.Status(), ynn_status_success);

  runtime.ReshapeExternalTensor(input_shape, input_data.data(), input_id);
  runtime.ReshapeExternalTensor(index_shape, index_data.data(), index_id);
  runtime.ReshapeRuntime();

  ASSERT_EQ(runtime.GetExternalTensorShape(output_id), expected_output_shape);

  std::vector<T> output_data(expected_output_data.size());
  runtime.SetupExternalTensor(output_data.data(), output_id).InvokeRuntime();

  EXPECT_THAT(output_data, testing::ElementsAreArray(expected_output_data));
}

template <typename T>
class GatherTest : public ::testing::Test {
 protected:
  using InputType = typename std::tuple_element<0, T>::type;
  using IndexType = typename std::tuple_element<1, T>::type;
};

using GatherTestTypes =
    ::testing::Types<std::tuple<int8_t, int8_t>, std::tuple<int8_t, uint8_t>,
                     std::tuple<int8_t, int32_t>, std::tuple<int32_t, int8_t>,
                     std::tuple<int32_t, uint8_t>, std::tuple<int32_t, int32_t>,
                     std::tuple<float, int8_t>, std::tuple<float, uint8_t>,
                     std::tuple<float, int32_t>, std::tuple<half, int8_t>,
                     std::tuple<half, uint8_t>, std::tuple<half, int32_t>,
                     std::tuple<bfloat16, int8_t>,
                     std::tuple<bfloat16, uint8_t>,
                     std::tuple<bfloat16, int32_t> >;
TYPED_TEST_SUITE(GatherTest, GatherTestTypes);

TYPED_TEST(GatherTest, Index0D) {
  using T = typename TestFixture::InputType;
  using IndexType = typename TestFixture::IndexType;

  // 4. 1D input, 0D index
  TestGather<T, IndexType>(
      /*axis=*/0,
      /*input_shape=*/{3}, /*input_data=*/{1, 2, 3},
      /*index_shape=*/{}, /*index_data=*/{1},
      /*expected_output_shape=*/{}, /*expected_output_data=*/{2});
}

TYPED_TEST(GatherTest, Index1D) {
  using T = typename TestFixture::InputType;
  using IndexType = typename TestFixture::IndexType;

  // 5. 1D input, 1D index
  TestGather<T, IndexType>(
      /*axis=*/0,
      /*input_shape=*/{3}, /*input_data=*/{1, 2, 3},
      /*index_shape=*/{2}, /*index_data=*/{2, 0},
      /*expected_output_shape=*/{2}, /*expected_output_data=*/{3, 1});
}

TYPED_TEST(GatherTest, Index2D) {
  using T = typename TestFixture::InputType;
  using IndexType = typename TestFixture::IndexType;

  // 6. 1D input, 2D index
  TestGather<T, IndexType>(
      /*axis=*/0,
      /*input_shape=*/{3}, /*input_data=*/{1, 2, 3},
      /*index_shape=*/{2, 3}, /*index_data=*/{2, 0, 1, 1, 2, 0},
      /*expected_output_shape=*/{2, 3},
      /*expected_output_data=*/{3, 1, 2, 2, 3, 1});
}

TYPED_TEST(GatherTest, Input2DIndex2D) {
  using T = typename TestFixture::InputType;
  using IndexType = typename TestFixture::IndexType;

  // 9. 2D input, 2D index
  TestGather<T, IndexType>(
      /*axis=*/0,
      /*input_shape=*/{3, 3}, /*input_data=*/{1, 2, 3, 4, 5, 6, 7, 8, 9},
      /*index_shape=*/{2, 3}, /*index_data=*/{1, 0, 2, 1, 2, 0},
      /*expected_output_shape=*/{2, 3},
      /*expected_output_data=*/{4, 2, 9, 4, 8, 3});
}

TYPED_TEST(GatherTest, IndexBroadcasting) {
  using T = typename TestFixture::InputType;
  using IndexType = typename TestFixture::IndexType;

  // Index broadcasting: index shape {2, 1} broadcasted to match input shape {2,
  // 3} (excluding axis 0)
  TestGather<T, IndexType>(
      /*axis=*/0,
      /*input_shape=*/{3, 3}, /*input_data=*/{1, 2, 3, 4, 5, 6, 7, 8, 9},
      /*index_shape=*/{2, 1}, /*index_data=*/{2, 0},
      /*expected_output_shape=*/{2, 3},
      /*expected_output_data=*/{7, 8, 9, 1, 2, 3});
}

TYPED_TEST(GatherTest, InputBroadcasting) {
  using T = typename TestFixture::InputType;
  using IndexType = typename TestFixture::IndexType;

  // Input broadcasting: input shape {2, 1} broadcasted to match index shape {2,
  // 3} (excluding axis 0)
  TestGather<T, IndexType>(
      /*axis=*/0,
      /*input_shape=*/{3, 1}, /*input_data=*/{1, 2, 3},
      /*index_shape=*/{2, 3}, /*index_data=*/{2, 0, 1, 0, 2, 0},
      /*expected_output_shape=*/{2, 3},
      /*expected_output_data=*/{3, 1, 2, 1, 3, 1});
}

}  // namespace
}  // namespace ynn
