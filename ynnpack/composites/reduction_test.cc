// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "ynnpack/composites/composites.h"
#include "ynnpack/composites/util.h"
#include "ynnpack/include/ynnpack.h"

namespace ynn {
namespace {

TEST(ReductionTest, QuantizedSum) {
  subgraph_ptr subgraph = create_subgraph(2, 0);
  ASSERT_NE(subgraph, nullptr);

  uint32_t x_id = 0;
  ASSERT_EQ(ynn_define_tensor(subgraph.get(), ynn_type_int8, 2, nullptr,
                              nullptr, YNN_VALUE_FLAG_EXTERNAL_INPUT, &x_id),
            ynn_status_success);

  uint32_t y_id = 1;
  ASSERT_EQ(ynn_define_tensor(subgraph.get(), ynn_type_int8, 1, nullptr,
                              nullptr, YNN_VALUE_FLAG_EXTERNAL_OUTPUT, &y_id),
            ynn_status_success);

  int32_t input_zp = 5;
  float input_scale = 0.2f;
  int32_t output_zp = -2;
  float output_scale = 0.5f;
  int32_t axes[] = {1};

  uint32_t input_zp_id = YNN_INVALID_VALUE_ID;
  ASSERT_EQ(define_constant(subgraph.get(), input_zp, input_zp_id),
            ynn_status_success);
  uint32_t input_scale_id = YNN_INVALID_VALUE_ID;
  ASSERT_EQ(define_constant(subgraph.get(), input_scale, input_scale_id),
            ynn_status_success);

  uint32_t output_zp_id = YNN_INVALID_VALUE_ID;
  ASSERT_EQ(define_constant(subgraph.get(), output_zp, output_zp_id),
            ynn_status_success);
  uint32_t output_scale_id = YNN_INVALID_VALUE_ID;
  ASSERT_EQ(define_constant(subgraph.get(), output_scale, output_scale_id),
            ynn_status_success);

  ASSERT_EQ(define_reduce_sum(
                subgraph.get(), 1, axes, x_id, input_zp_id, input_scale_id,
                /*keep_dims=*/false, /*mean=*/false, /*squared=*/false,
                ynn_type_int8, output_zp_id, output_scale_id, y_id),
            ynn_status_success);

  runtime_ptr runtime = create_runtime(subgraph, nullptr, 0);
  ASSERT_NE(runtime, nullptr);

  // Input shape: 2x3
  size_t x_shape[] = {2, 3};
  ASSERT_EQ(ynn_set_external_value_shape(runtime.get(), x_id, 2, x_shape),
            ynn_status_success);

  // Input data (quantized)
  // Row 0: [10, 20, 30] -> float: [(10-5)*0.2, (20-5)*0.2, (30-5)*0.2] =
  // [1.0, 3.0, 5.0] -> sum = 9.0 Row 1: [-5, 0, 5]   -> float: [(-5-5)*0.2,
  // (0-5)*0.2, (5-5)*0.2]   = [-2.0, -1.0, 0.0] -> sum = -3.0
  std::vector<int8_t> x_data = {10, 20, 30, -5, 0, 5};
  std::vector<int8_t> y_data(2, 0);

  ASSERT_EQ(ynn_set_external_value_data(runtime.get(), x_id, x_data.data()),
            ynn_status_success);
  ASSERT_EQ(ynn_set_external_value_data(runtime.get(), y_id, y_data.data()),
            ynn_status_success);
  ASSERT_EQ(ynn_reshape_runtime(runtime.get()), ynn_status_success);
  ASSERT_EQ(ynn_invoke_runtime(runtime.get()), ynn_status_success);

  // Expected output:
  // Row 0: sum = 9.0  -> quantized: round(9.0/0.5) - 2 = 18 - 2 = 16
  // Row 1: sum = -3.0 -> quantized: round(-3.0/0.5) - 2 = -6 - 2 = -8
  EXPECT_EQ(y_data[0], 16);
  EXPECT_EQ(y_data[1], -8);
}

TEST(ReductionTest, QuantizedMean) {
  subgraph_ptr subgraph = create_subgraph(2, 0);
  ASSERT_NE(subgraph, nullptr);

  uint32_t x_id = 0;
  ASSERT_EQ(ynn_define_tensor(subgraph.get(), ynn_type_int8, 2, nullptr,
                              nullptr, YNN_VALUE_FLAG_EXTERNAL_INPUT, &x_id),
            ynn_status_success);

  uint32_t y_id = 1;
  ASSERT_EQ(ynn_define_tensor(subgraph.get(), ynn_type_int8, 1, nullptr,
                              nullptr, YNN_VALUE_FLAG_EXTERNAL_OUTPUT, &y_id),
            ynn_status_success);

  int32_t input_zp = 5;
  float input_scale = 0.2f;
  int32_t output_zp = -2;
  float output_scale = 0.1f;
  int32_t axes[] = {1};

  uint32_t input_zp_id = YNN_INVALID_VALUE_ID;
  ASSERT_EQ(define_constant(subgraph.get(), input_zp, input_zp_id),
            ynn_status_success);
  uint32_t input_scale_id = YNN_INVALID_VALUE_ID;
  ASSERT_EQ(define_constant(subgraph.get(), input_scale, input_scale_id),
            ynn_status_success);

  uint32_t output_zp_id = YNN_INVALID_VALUE_ID;
  ASSERT_EQ(define_constant(subgraph.get(), output_zp, output_zp_id),
            ynn_status_success);
  uint32_t output_scale_id = YNN_INVALID_VALUE_ID;
  ASSERT_EQ(define_constant(subgraph.get(), output_scale, output_scale_id),
            ynn_status_success);

  ASSERT_EQ(define_reduce_sum(
                subgraph.get(), 1, axes, x_id, input_zp_id, input_scale_id,
                /*keep_dims=*/false, /*mean=*/true, /*squared=*/false,
                ynn_type_int8, output_zp_id, output_scale_id, y_id),
            ynn_status_success);

  runtime_ptr runtime = create_runtime(subgraph, nullptr, 0);
  ASSERT_NE(runtime, nullptr);

  // Input shape: 2x3
  size_t x_shape[] = {2, 3};
  ASSERT_EQ(ynn_set_external_value_shape(runtime.get(), x_id, 2, x_shape),
            ynn_status_success);

  // Input data (quantized)
  // Row 0: [10, 20, 30] -> float: [1.0, 3.0, 5.0] -> mean = 3.0
  // Row 1: [-5, 0, 5]   -> float: [-2.0, -1.0, 0.0] -> mean = -1.0
  std::vector<int8_t> x_data = {10, 20, 30, -5, 0, 5};
  std::vector<int8_t> y_data(2, 0);

  ASSERT_EQ(ynn_set_external_value_data(runtime.get(), x_id, x_data.data()),
            ynn_status_success);
  ASSERT_EQ(ynn_set_external_value_data(runtime.get(), y_id, y_data.data()),
            ynn_status_success);
  ASSERT_EQ(ynn_reshape_runtime(runtime.get()), ynn_status_success);
  ASSERT_EQ(ynn_invoke_runtime(runtime.get()), ynn_status_success);

  // Expected output:
  // Row 0: mean = 3.0  -> quantized: round(3.0/0.1) - 2 = 30 - 2 = 28
  // Row 1: mean = -1.0 -> quantized: round(-1.0/0.1) - 2 = -10 - 2 = -12
  EXPECT_EQ(y_data[0], 28);
  EXPECT_EQ(y_data[1], -12);
}

TEST(ReductionTest, FloatSum) {
  subgraph_ptr subgraph = create_subgraph(2, 0);
  ASSERT_NE(subgraph, nullptr);

  uint32_t x_id = 0;
  ASSERT_EQ(ynn_define_tensor(subgraph.get(), ynn_type_fp32, 2, nullptr,
                              nullptr, YNN_VALUE_FLAG_EXTERNAL_INPUT, &x_id),
            ynn_status_success);

  uint32_t y_id = 1;
  ASSERT_EQ(ynn_define_tensor(subgraph.get(), ynn_type_fp32, 1, nullptr,
                              nullptr, YNN_VALUE_FLAG_EXTERNAL_OUTPUT, &y_id),
            ynn_status_success);

  int32_t axes[] = {1};

  ASSERT_EQ(define_reduce_sum(subgraph.get(), 1, axes, x_id,
                              YNN_INVALID_VALUE_ID, YNN_INVALID_VALUE_ID,
                              /*keep_dims=*/false, /*mean=*/false,
                              /*squared=*/false, ynn_type_fp32,
                              YNN_INVALID_VALUE_ID, YNN_INVALID_VALUE_ID, y_id),
            ynn_status_success);

  runtime_ptr runtime = create_runtime(subgraph, nullptr, 0);
  ASSERT_NE(runtime, nullptr);

  // Input shape: 2x3
  size_t x_shape[] = {2, 3};
  ASSERT_EQ(ynn_set_external_value_shape(runtime.get(), x_id, 2, x_shape),
            ynn_status_success);

  std::vector<float> x_data = {1.0f, 2.0f, 3.0f, -1.0f, 0.0f, 1.0f};
  std::vector<float> y_data(2, 0.0f);

  ASSERT_EQ(ynn_set_external_value_data(runtime.get(), x_id, x_data.data()),
            ynn_status_success);
  ASSERT_EQ(ynn_set_external_value_data(runtime.get(), y_id, y_data.data()),
            ynn_status_success);
  ASSERT_EQ(ynn_reshape_runtime(runtime.get()), ynn_status_success);
  ASSERT_EQ(ynn_invoke_runtime(runtime.get()), ynn_status_success);

  EXPECT_FLOAT_EQ(y_data[0], 6.0f);
  EXPECT_FLOAT_EQ(y_data[1], 0.0f);
}

TEST(ReductionTest, FloatMean) {
  subgraph_ptr subgraph = create_subgraph(2, 0);
  ASSERT_NE(subgraph, nullptr);

  uint32_t x_id = 0;
  ASSERT_EQ(ynn_define_tensor(subgraph.get(), ynn_type_fp32, 2, nullptr,
                              nullptr, YNN_VALUE_FLAG_EXTERNAL_INPUT, &x_id),
            ynn_status_success);

  uint32_t y_id = 1;
  ASSERT_EQ(ynn_define_tensor(subgraph.get(), ynn_type_fp32, 1, nullptr,
                              nullptr, YNN_VALUE_FLAG_EXTERNAL_OUTPUT, &y_id),
            ynn_status_success);

  int32_t axes[] = {1};

  ASSERT_EQ(define_reduce_sum(subgraph.get(), 1, axes, x_id,
                              YNN_INVALID_VALUE_ID, YNN_INVALID_VALUE_ID,
                              /*keep_dims=*/false, /*mean=*/true,
                              /*squared=*/false, ynn_type_fp32,
                              YNN_INVALID_VALUE_ID, YNN_INVALID_VALUE_ID, y_id),
            ynn_status_success);

  runtime_ptr runtime = create_runtime(subgraph, nullptr, 0);
  ASSERT_NE(runtime, nullptr);

  // Input shape: 2x3
  size_t x_shape[] = {2, 3};
  ASSERT_EQ(ynn_set_external_value_shape(runtime.get(), x_id, 2, x_shape),
            ynn_status_success);

  std::vector<float> x_data = {1.0f, 2.0f, 3.0f, -1.0f, 0.0f, 1.0f};
  std::vector<float> y_data(2, 0.0f);

  ASSERT_EQ(ynn_set_external_value_data(runtime.get(), x_id, x_data.data()),
            ynn_status_success);
  ASSERT_EQ(ynn_set_external_value_data(runtime.get(), y_id, y_data.data()),
            ynn_status_success);
  ASSERT_EQ(ynn_reshape_runtime(runtime.get()), ynn_status_success);
  ASSERT_EQ(ynn_invoke_runtime(runtime.get()), ynn_status_success);

  EXPECT_FLOAT_EQ(y_data[0], 2.0f);
  EXPECT_FLOAT_EQ(y_data[1], 0.0f);
}

TEST(ReductionTest, FloatSumSquared) {
  subgraph_ptr subgraph = create_subgraph(2, 0);
  ASSERT_NE(subgraph, nullptr);

  uint32_t x_id = 0;
  ASSERT_EQ(ynn_define_tensor(subgraph.get(), ynn_type_fp32, 2, nullptr,
                              nullptr, YNN_VALUE_FLAG_EXTERNAL_INPUT, &x_id),
            ynn_status_success);

  uint32_t y_id = 1;
  ASSERT_EQ(ynn_define_tensor(subgraph.get(), ynn_type_fp32, 1, nullptr,
                              nullptr, YNN_VALUE_FLAG_EXTERNAL_OUTPUT, &y_id),
            ynn_status_success);

  int32_t axes[] = {1};

  ASSERT_EQ(define_reduce_sum(subgraph.get(), 1, axes, x_id,
                              YNN_INVALID_VALUE_ID, YNN_INVALID_VALUE_ID,
                              /*keep_dims=*/false, /*mean=*/false,
                              /*squared=*/true, ynn_type_fp32,
                              YNN_INVALID_VALUE_ID, YNN_INVALID_VALUE_ID, y_id),
            ynn_status_success);

  runtime_ptr runtime = create_runtime(subgraph, nullptr, 0);
  ASSERT_NE(runtime, nullptr);

  // Input shape: 2x3
  size_t x_shape[] = {2, 3};
  ASSERT_EQ(ynn_set_external_value_shape(runtime.get(), x_id, 2, x_shape),
            ynn_status_success);

  // 1^2 + 2^2 + 3^2 = 1 + 4 + 9 = 14
  // (-1)^2 + 0^2 + 1^2 = 1 + 0 + 1 = 2
  std::vector<float> x_data = {1.0f, 2.0f, 3.0f, -1.0f, 0.0f, 1.0f};
  std::vector<float> y_data(2, 0.0f);

  ASSERT_EQ(ynn_set_external_value_data(runtime.get(), x_id, x_data.data()),
            ynn_status_success);
  ASSERT_EQ(ynn_set_external_value_data(runtime.get(), y_id, y_data.data()),
            ynn_status_success);
  ASSERT_EQ(ynn_reshape_runtime(runtime.get()), ynn_status_success);
  ASSERT_EQ(ynn_invoke_runtime(runtime.get()), ynn_status_success);

  EXPECT_FLOAT_EQ(y_data[0], 14.0f);
  EXPECT_FLOAT_EQ(y_data[1], 2.0f);
}

TEST(ReductionTest, FloatMeanSquared) {
  subgraph_ptr subgraph = create_subgraph(2, 0);
  ASSERT_NE(subgraph, nullptr);

  uint32_t x_id = 0;
  ASSERT_EQ(ynn_define_tensor(subgraph.get(), ynn_type_fp32, 2, nullptr,
                              nullptr, YNN_VALUE_FLAG_EXTERNAL_INPUT, &x_id),
            ynn_status_success);

  uint32_t y_id = 1;
  ASSERT_EQ(ynn_define_tensor(subgraph.get(), ynn_type_fp32, 1, nullptr,
                              nullptr, YNN_VALUE_FLAG_EXTERNAL_OUTPUT, &y_id),
            ynn_status_success);

  int32_t axes[] = {1};

  ASSERT_EQ(define_reduce_sum(subgraph.get(), 1, axes, x_id,
                              YNN_INVALID_VALUE_ID, YNN_INVALID_VALUE_ID,
                              /*keep_dims=*/false, /*mean=*/true,
                              /*squared=*/true, ynn_type_fp32,
                              YNN_INVALID_VALUE_ID, YNN_INVALID_VALUE_ID, y_id),
            ynn_status_success);

  runtime_ptr runtime = create_runtime(subgraph, nullptr, 0);
  ASSERT_NE(runtime, nullptr);

  // Input shape: 2x3
  size_t x_shape[] = {2, 3};
  ASSERT_EQ(ynn_set_external_value_shape(runtime.get(), x_id, 2, x_shape),
            ynn_status_success);

  // (1^2 + 2^2 + 3^2) / 3 = 14 / 3 = 4.6666666...
  // ((-1)^2 + 0^2 + 1^2) / 3 = 2 / 3 = 0.6666666...
  std::vector<float> x_data = {1.0f, 2.0f, 3.0f, -1.0f, 0.0f, 1.0f};
  std::vector<float> y_data(2, 0.0f);

  ASSERT_EQ(ynn_set_external_value_data(runtime.get(), x_id, x_data.data()),
            ynn_status_success);
  ASSERT_EQ(ynn_set_external_value_data(runtime.get(), y_id, y_data.data()),
            ynn_status_success);
  ASSERT_EQ(ynn_reshape_runtime(runtime.get()), ynn_status_success);
  ASSERT_EQ(ynn_invoke_runtime(runtime.get()), ynn_status_success);

  EXPECT_NEAR(y_data[0], 14.0f / 3.0f, 1e-5);
  EXPECT_NEAR(y_data[1], 2.0f / 3.0f, 1e-5);
}

}  // namespace
}  // namespace ynn
