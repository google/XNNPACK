// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <cstddef>
#include <cstdint>
#include <tuple>
#include <utility>
#include <vector>

#include <gtest/gtest.h>
#include "ynnpack/base/base.h"
#include "ynnpack/base/bfloat16.h"
#include "ynnpack/base/half.h"
#include "ynnpack/base/to_string.h"
#include "ynnpack/base/test/tensor.h"
#include "ynnpack/base/test/util.h"
#include "ynnpack/include/ynnpack.h"
#include "ynnpack/subgraph/test/subgraph_builder.h"

using ynn::to_string;  // NOLINT

namespace ynn {
namespace {

template <typename T>
void TestImpl(T, ynn_type target_type) {
  // Define subgraph
  const uint32_t min_max_id = 0;
  uint32_t scale_id = 1;
  uint32_t zero_point_id = 2;

  // This test works by defining some scales and zero points, setting the input
  // to the min/max values that produce those scales/zero points, and then
  // checking the result of the operation matches what we defined,
  const float scales[8] = {0.125f, 0.25f, 0.5f, 1.0f, 2.0f, 4.0f, 8.0f, 16.0f};
  std::vector<size_t> output_shape = {8, 256};

  std::vector<size_t> min_max_shape = output_shape;
  // Shape of min_max has leading dimension 2.
  min_max_shape.insert(min_max_shape.begin(), 2);

  SubgraphBuilder subgraph(3);
  subgraph.AddInput(type_of<T>(), 3, min_max_id)
      .AddOutput(ynn_type_fp32, 2, scale_id)
      .AddOutput(ynn_type_int32, 2, zero_point_id);

  ynn_status status = ynn_define_dynamic_quantization(
      subgraph.GetSubgraph(), min_max_id, target_type, &zero_point_id,
      &scale_id, /*flags=*/0);
  ASSERT_EQ(status, ynn_status_success);

  Runtime runtime(subgraph.GetSubgraph());
  ASSERT_EQ(runtime.Status(), ynn_status_success);

  Tensor<T> min_max(min_max_shape);
  Tensor<float> expected_scale(output_shape);
  Tensor<int32_t> expected_zero_point(output_shape);

  // Generate expected scale and zero point, and compute min_max.
  for (int zp = 0; zp < output_shape[1]; ++zp) {
    for (int s_idx = 0; s_idx < output_shape[0]; ++s_idx) {
      const float s = scales[s_idx];
      min_max(0, s_idx, zp) = static_cast<T>(-zp * s);
      min_max(1, s_idx, zp) = static_cast<T>((255 - zp) * s);
    }
  }

  runtime.ReshapeExternalTensor(min_max_shape, min_max.base(), min_max_id)
      .ReshapeRuntime();

  ASSERT_EQ(runtime.GetExternalTensorShape(scale_id), output_shape);
  ASSERT_EQ(runtime.GetExternalTensorShape(zero_point_id), output_shape);

  Tensor<float> scale(output_shape);
  Tensor<int32_t> zero_point(output_shape);

  runtime.SetupExternalTensor(scale.base(), scale_id)
      .SetupExternalTensor(zero_point.base(), zero_point_id)
      .InvokeRuntime();

  for (int zp = 0; zp < output_shape[1]; ++zp) {
    for (int s_idx = 0; s_idx < output_shape[0]; ++s_idx) {
      // Compare with some tolerance.
      EXPECT_NEAR(scale(s_idx, zp), scales[s_idx], 1e-5);
      const int32_t expected_zp = target_type == ynn_type_uint8 ? zp : zp - 128;
      EXPECT_EQ(zero_point(s_idx, zp), expected_zp);
    }
  }
}

template <typename F>
constexpr decltype(auto) SwitchType(ynn_type type, F&& f) {
  switch (type) {
    case ynn_type_fp32:
      return std::forward<F>(f)(float());
    case ynn_type_fp16:
      return std::forward<F>(f)(half());
    case ynn_type_bf16:
      return std::forward<F>(f)(bfloat16());
    default:
      YNN_UNREACHABLE;
  }
}

class DynamicQuantizationTest
    : public testing::TestWithParam<std::tuple<ynn_type, ynn_type>> {};

TEST_P(DynamicQuantizationTest, Run) {
  SwitchType(std::get<0>(GetParam()), [&](auto a_type) {
    TestImpl(a_type, std::get<1>(GetParam()));
  });
}

INSTANTIATE_TEST_SUITE_P(
    Test, DynamicQuantizationTest,
    testing::Combine(testing::Values(ynn_type_fp32, ynn_type_fp16,
                                     ynn_type_bf16),
                     testing::Values(ynn_type_int8, ynn_type_uint8)),
    [](const testing::TestParamInfo<DynamicQuantizationTest::ParamType>& info) {
      return test_param_to_string(info);
    });

}  // namespace
}  // namespace ynn
