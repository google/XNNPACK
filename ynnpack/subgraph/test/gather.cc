// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <functional>
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
void TestGather(int32_t axis, std::vector<size_t> input_shape,
                std::vector<T> input_data, std::vector<size_t> index_shape,
                std::vector<int32_t> index_data,
                std::vector<size_t> expected_output_shape,
                std::vector<T> expected_output_data,
                bool expect_success = true) {
  TestGather<T, IndexType>(std::vector<int32_t>{axis}, input_shape, input_data,
                           index_shape, index_data, expected_output_shape,
                           expected_output_data, expect_success);
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
      /*axis=*/0,
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
      /*axis=*/0,
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
      /*axis=*/0,
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
      /*axis=*/0,
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
      /*axis=*/0,
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
      /*axis=*/0,
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
      /*axis=*/0,
      /*input_shape=*/{3}, /*input_data=*/{1, 2, 3},
      /*index_shape=*/{4}, /*index_data=*/{1, 3, 0, 1},  // 3 is out of bounds
      /*expected_output_shape=*/{4}, /*expected_output_data=*/{},
      /*expect_success=*/false);

  if constexpr (std::is_signed_v<IndexType>) {
    TestGather<T, IndexType>(
        /*axis=*/0,
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
      /*axis=*/0,
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
      /*axis=*/1,
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

}  // namespace
}  // namespace ynn
