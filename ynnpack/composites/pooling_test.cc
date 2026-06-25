// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <string>
#include <tuple>
#include <type_traits>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "ynnpack/base/bfloat16.h"
#include "ynnpack/base/half.h"
#include "ynnpack/base/type.h"
#include "ynnpack/composites/composites.h"
#include "ynnpack/composites/util.h"
#include "ynnpack/include/ynnpack.h"

namespace ynn {
namespace {

template <typename T>
void VerifyAveragePool2D(bool padding_same, size_t filter_height,
                         size_t filter_width, size_t stride_height,
                         size_t stride_width, const std::vector<float>& x_fp32,
                         const std::vector<size_t>& x_shape,
                         const std::vector<float>& expected_fp32,
                         const std::vector<size_t>& expected_shape) {
  subgraph_ptr subgraph = create_subgraph(2, 0);
  ASSERT_NE(subgraph, nullptr);

  ynn_type tensor_type = type_of<T>();

  uint32_t x_id = 0;
  ASSERT_EQ(ynn_define_tensor(subgraph.get(), tensor_type, 4, nullptr, nullptr,
                              YNN_VALUE_FLAG_EXTERNAL_INPUT, &x_id),
            ynn_status_success);

  uint32_t y_id = 1;
  ASSERT_EQ(ynn_define_tensor(subgraph.get(), tensor_type, 4, nullptr, nullptr,
                              YNN_VALUE_FLAG_EXTERNAL_OUTPUT, &y_id),
            ynn_status_success);

  ASSERT_EQ(define_average_pool_2d(subgraph.get(), x_id, tensor_type,
                                   padding_same, filter_height, filter_width,
                                   stride_height, stride_width, y_id),
            ynn_status_success);

  runtime_ptr runtime = create_runtime(subgraph, nullptr, 0);
  ASSERT_NE(runtime, nullptr);

  ASSERT_EQ(
      ynn_set_external_value_shape(runtime.get(), x_id, 4, x_shape.data()),
      ynn_status_success);

  std::vector<T> x(x_fp32.size());
  std::copy_n(x_fp32.begin(), x_fp32.size(), x.begin());
  std::vector<T> y(expected_fp32.size(), T(0.0f));

  ASSERT_EQ(ynn_set_external_value_data(runtime.get(), x_id, x.data()),
            ynn_status_success);
  ASSERT_EQ(ynn_set_external_value_data(runtime.get(), y_id, y.data()),
            ynn_status_success);
  ASSERT_EQ(ynn_reshape_runtime(runtime.get()), ynn_status_success);
  ASSERT_EQ(ynn_invoke_runtime(runtime.get()), ynn_status_success);

  std::vector<float> y_fp32(y.size());
  std::copy_n(y.begin(), y.size(), y_fp32.begin());

  float tolerance = 1e-5f;
  if constexpr (std::is_same_v<T, half>) {
    tolerance = 1e-2f;
  } else if constexpr (std::is_same_v<T, bfloat16>) {
    tolerance = 5e-2f;
  }

  for (size_t i = 0; i < y_fp32.size(); ++i) {
    EXPECT_NEAR(y_fp32[i], expected_fp32[i], tolerance) << "at index " << i;
  }
}

void VerifyAveragePool2D(ynn_type type, bool padding_same, size_t filter_height,
                         size_t filter_width, size_t stride_height,
                         size_t stride_width, const std::vector<float>& x_fp32,
                         const std::vector<size_t>& x_shape,
                         const std::vector<float>& expected_fp32,
                         const std::vector<size_t>& expected_shape) {
  if (type == ynn_type_fp32) {
    VerifyAveragePool2D<float>(padding_same, filter_height, filter_width,
                               stride_height, stride_width, x_fp32, x_shape,
                               expected_fp32, expected_shape);
  } else if (type == ynn_type_fp16) {
    VerifyAveragePool2D<half>(padding_same, filter_height, filter_width,
                              stride_height, stride_width, x_fp32, x_shape,
                              expected_fp32, expected_shape);
  } else if (type == ynn_type_bf16) {
    VerifyAveragePool2D<bfloat16>(padding_same, filter_height, filter_width,
                                  stride_height, stride_width, x_fp32, x_shape,
                                  expected_fp32, expected_shape);
  } else {
    FAIL() << "Unsupported type";
  }
}

class PoolingTest : public ::testing::TestWithParam<ynn_type> {};

TEST_P(PoolingTest, AveragePool2DValid) {
  ynn_type type = GetParam();

  // clang-format off
  std::vector<float> x_data = {
      1.0f,  2.0f,  3.0f,  4.0f,
      5.0f,  6.0f,  7.0f,  8.0f,
      9.0f, 10.0f, 11.0f, 12.0f,
     13.0f, 14.0f, 15.0f, 16.0f,
  };
  std::vector<float> expected = {
      3.5f,  4.5f,  5.5f,
      7.5f,  8.5f,  9.5f,
     11.5f, 12.5f, 13.5f,
  };
  // clang-format on

  VerifyAveragePool2D(type, /*padding_same=*/false, /*filter_height=*/2,
                      /*filter_width=*/2, /*stride_height=*/1,
                      /*stride_width=*/1, x_data, {1, 4, 4, 1}, expected,
                      {1, 3, 3, 1});
}

TEST_P(PoolingTest, AveragePool2DSame) {
  ynn_type type = GetParam();

  // clang-format off
  std::vector<float> x_data = {
      1.0f, 2.0f, 3.0f,
      4.0f, 5.0f, 6.0f,
      7.0f, 8.0f, 9.0f,
  };
  std::vector<float> expected = {
      3.0f, 3.5f, 4.0f,
      4.5f, 5.0f, 5.5f,
      6.0f, 6.5f, 7.0f,
  };
  // clang-format on

  VerifyAveragePool2D(type, /*padding_same=*/true, /*filter_height=*/3,
                      /*filter_width=*/3, /*stride_height=*/1,
                      /*stride_width=*/1, x_data, {1, 3, 3, 1}, expected,
                      {1, 3, 3, 1});
}

TEST_P(PoolingTest, AveragePool2DValidStride2) {
  ynn_type type = GetParam();

  // clang-format off
  std::vector<float> x_data = {
      1.0f,  2.0f,  3.0f,  4.0f,
      5.0f,  6.0f,  7.0f,  8.0f,
      9.0f, 10.0f, 11.0f, 12.0f,
     13.0f, 14.0f, 15.0f, 16.0f,
  };
  std::vector<float> expected = {
      3.5f,  5.5f,
     11.5f, 13.5f,
  };
  // clang-format on

  VerifyAveragePool2D(type, /*padding_same=*/false, /*filter_height=*/2,
                      /*filter_width=*/2, /*stride_height=*/2,
                      /*stride_width=*/2, x_data, {1, 4, 4, 1}, expected,
                      {1, 2, 2, 1});
}

TEST_P(PoolingTest, AveragePool2DSameStride2) {
  ynn_type type = GetParam();

  // clang-format off
  std::vector<float> x_data = {
      1.0f, 2.0f, 3.0f,
      4.0f, 5.0f, 6.0f,
      7.0f, 8.0f, 9.0f,
  };
  std::vector<float> expected = {
      3.0f, 4.0f,
      6.0f, 7.0f,
  };
  // clang-format on

  VerifyAveragePool2D(type, /*padding_same=*/true, /*filter_height=*/3,
                      /*filter_width=*/3, /*stride_height=*/2,
                      /*stride_width=*/2, x_data, {1, 3, 3, 1}, expected,
                      {1, 2, 2, 1});
}

INSTANTIATE_TEST_SUITE_P(
    Pooling, PoolingTest,
    ::testing::Values(ynn_type_fp32, ynn_type_fp16, ynn_type_bf16),
    [](const testing::TestParamInfo<PoolingTest::ParamType>& info) {
      return std::string(to_string(info.param));
    });

}  // namespace
}  // namespace ynn
