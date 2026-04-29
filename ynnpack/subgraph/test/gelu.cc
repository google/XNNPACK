// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "ynnpack/include/ynnpack.h"
#include "ynnpack/subgraph/subgraph.h"
#include "ynnpack/subgraph/test/matchers.h"
#include "ynnpack/subgraph/test/subgraph_builder.h"

namespace ynn {

using ::testing::AllOf;

float gelu(float x) {
  return x * 0.5f * (1.0f + std::erf(x / std::sqrt(2.0f)));
}

float approxgelu(float x) {
  const float c1 = static_cast<float>(std::sqrt(2.0f / M_PI));
  const float c3 = 0.044715f;
  return x * 0.5f * (1.0f + std::tanh(c1 * x * (1.0f + c3 * x * x)));
}

TEST(subgraph, gelu) {
  const uint32_t x_id = 0;
  const uint32_t y_id = 1;
  const uint32_t sqrt2_over_2_id = 2;
  const uint32_t half_id = 3;
  SubgraphBuilder builder(4);

  uint32_t x_scaled_id = YNN_INVALID_VALUE_ID;
  uint32_t erf_id = YNN_INVALID_VALUE_ID;
  uint32_t half_erf_id = YNN_INVALID_VALUE_ID;
  uint32_t half_erf_plus_half_id = YNN_INVALID_VALUE_ID;

  builder.AddInput(ynn_type_fp32, 1, x_id)
      .AddOutput(ynn_type_fp32, 1, y_id)
      .AddScalar(std::sqrt(2.0f) / 2.0f, sqrt2_over_2_id)
      .AddScalar(0.5f, half_id)
      .AddTensor(ynn_type_fp32, 1, x_scaled_id)
      .AddTensor(ynn_type_fp32, 1, erf_id)
      .AddTensor(ynn_type_fp32, 1, half_erf_id)
      .AddTensor(ynn_type_fp32, 1, half_erf_plus_half_id);

  builder.AddBinary(ynn_binary_multiply, x_id, sqrt2_over_2_id, x_scaled_id)
      .AddUnary(ynn_unary_erf, x_scaled_id, erf_id)
      .AddBinary(ynn_binary_multiply, erf_id, half_id, half_erf_id)
      .AddBinary(ynn_binary_add, half_erf_id, half_id, half_erf_plus_half_id)
      .AddBinary(ynn_binary_multiply, x_id, half_erf_plus_half_id, y_id);

  ynn_subgraph& subgraph = *builder.GetSubgraph();

  subgraph.fusion();
  subgraph.invalidate_dead_values();

  // This graph should have fused into two nodes.
  ASSERT_THAT(subgraph, AllOf(HasValidNodeCount(2), HasValidValueCount(3)));

  Runtime runtime(&subgraph);
  ASSERT_EQ(runtime.Status(), ynn_status_success);

  const size_t n = 100;
  std::vector<float> x(n);
  std::vector<float> y(n);
  for (size_t i = 0; i < n; ++i) {
    x[i] = static_cast<float>(i) / 10.0f - 5.0f;
  }

  runtime.ReshapeExternalTensor({n}, x.data(), x_id)
      .SetupExternalTensor(y.data(), y_id)
      .ReshapeRuntime()
      .InvokeRuntime();
  ASSERT_EQ(runtime.Status(), ynn_status_success);

  for (size_t i = 0; i < n; ++i) {
    EXPECT_NEAR(y[i], gelu(x[i]), 1e-5f);
  }
}

TEST(subgraph, approxgelu) {
  const uint32_t x_id = 0;
  const uint32_t y_id = 1;
  const uint32_t sqrt2_over_2_id = 2;
  const uint32_t half_id = 3;
  SubgraphBuilder builder(4);

  uint32_t x_poly_id = YNN_INVALID_VALUE_ID;
  uint32_t tanh_id = YNN_INVALID_VALUE_ID;
  uint32_t half_tanh_id = YNN_INVALID_VALUE_ID;
  uint32_t half_tanh_plus_half_id = YNN_INVALID_VALUE_ID;

  builder.AddInput(ynn_type_fp32, 1, x_id)
      .AddOutput(ynn_type_fp32, 1, y_id)
      .AddScalar(std::sqrt(2.0f) / 2.0f, sqrt2_over_2_id)
      .AddScalar(0.5f, half_id)
      .AddTensor(ynn_type_fp32, 1, x_poly_id)
      .AddTensor(ynn_type_fp32, 1, tanh_id)
      .AddTensor(ynn_type_fp32, 1, half_tanh_id)
      .AddTensor(ynn_type_fp32, 1, half_tanh_plus_half_id);

  // (x / 2) * (1 + tanh(sqrt(2 / pi) * (x + 0.044715 * x^3)))
  const float c1 = static_cast<float>(std::sqrt(2.0f / M_PI));
  const float c3 = 0.044715f;
  builder.AddPolynomial({0.0f, c1, 0.0f, c1 * c3}, x_id, x_poly_id)
      .AddUnary(ynn_unary_tanh, x_poly_id, tanh_id)
      .AddBinary(ynn_binary_multiply, tanh_id, half_id, half_tanh_id)
      .AddBinary(ynn_binary_add, half_tanh_id, half_id, half_tanh_plus_half_id)
      .AddBinary(ynn_binary_multiply, x_id, half_tanh_plus_half_id, y_id);

  ynn_subgraph& subgraph = *builder.GetSubgraph();

  subgraph.fusion();
  subgraph.invalidate_dead_values();

  // This graph should have fused into three nodes.
  ASSERT_THAT(subgraph, AllOf(HasValidNodeCount(3), HasValidValueCount(4)));

  Runtime runtime(&subgraph);
  ASSERT_EQ(runtime.Status(), ynn_status_success);

  const size_t n = 100;
  std::vector<float> x(n);
  std::vector<float> y(n);
  for (size_t i = 0; i < n; ++i) {
    x[i] = static_cast<float>(i) / 10.0f - 5.0f;
  }

  runtime.ReshapeExternalTensor({n}, x.data(), x_id)
      .SetupExternalTensor(y.data(), y_id)
      .ReshapeRuntime()
      .InvokeRuntime();
  ASSERT_EQ(runtime.Status(), ynn_status_success);

  for (size_t i = 0; i < n; ++i) {
    EXPECT_NEAR(y[i], approxgelu(x[i]), 1e-5f);
  }
}

}  // namespace ynn
