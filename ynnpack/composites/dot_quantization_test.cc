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

TEST(DotQuantizationTest, Basic) {
  subgraph_ptr subgraph =
      create_subgraph(8, 0);  // 6 inputs + 2 outputs = 8 external values
  ASSERT_NE(subgraph, nullptr);

  // Define external inputs
  uint32_t a_id = 0;
  ASSERT_EQ(ynn_define_tensor(subgraph.get(), ynn_type_int8, 2, nullptr,
                              nullptr, YNN_VALUE_FLAG_EXTERNAL_INPUT, &a_id),
            ynn_status_success);

  uint32_t b_id = 1;
  ASSERT_EQ(ynn_define_tensor(subgraph.get(), ynn_type_int8, 2, nullptr,
                              nullptr, YNN_VALUE_FLAG_EXTERNAL_INPUT, &b_id),
            ynn_status_success);

  uint32_t a_zp_id = 2;
  ASSERT_EQ(ynn_define_tensor(subgraph.get(), ynn_type_int32, 2, nullptr,
                              nullptr, YNN_VALUE_FLAG_EXTERNAL_INPUT, &a_zp_id),
            ynn_status_success);

  uint32_t b_zp_id = 3;
  ASSERT_EQ(ynn_define_tensor(subgraph.get(), ynn_type_int32, 2, nullptr,
                              nullptr, YNN_VALUE_FLAG_EXTERNAL_INPUT, &b_zp_id),
            ynn_status_success);

  uint32_t a_scale_id = 4;
  ASSERT_EQ(
      ynn_define_tensor(subgraph.get(), ynn_type_fp32, 2, nullptr, nullptr,
                        YNN_VALUE_FLAG_EXTERNAL_INPUT, &a_scale_id),
      ynn_status_success);

  uint32_t b_scale_id = 5;
  ASSERT_EQ(
      ynn_define_tensor(subgraph.get(), ynn_type_fp32, 2, nullptr, nullptr,
                        YNN_VALUE_FLAG_EXTERNAL_INPUT, &b_scale_id),
      ynn_status_success);

  // Define external outputs
  uint32_t zp_out_id = 6;
  ASSERT_EQ(
      ynn_define_tensor(subgraph.get(), ynn_type_int32, 2, nullptr, nullptr,
                        YNN_VALUE_FLAG_EXTERNAL_OUTPUT, &zp_out_id),
      ynn_status_success);

  uint32_t scale_out_id = 7;
  ASSERT_EQ(
      ynn_define_tensor(subgraph.get(), ynn_type_fp32, 2, nullptr, nullptr,
                        YNN_VALUE_FLAG_EXTERNAL_OUTPUT, &scale_out_id),
      ynn_status_success);

  // Call define_dot_quantization
  uint32_t zp_id = YNN_INVALID_VALUE_ID;
  uint32_t scale_id = YNN_INVALID_VALUE_ID;
  ASSERT_EQ(
      define_dot_quantization(subgraph.get(), 1, a_id, a_zp_id, a_scale_id,
                              b_id, b_zp_id, b_scale_id, zp_id, scale_id),
      ynn_status_success);

  // Copy internal outputs to external outputs by adding 0
  uint32_t zero_int32_id = YNN_INVALID_VALUE_ID;
  ASSERT_EQ(define_constant(subgraph.get(), 0, zero_int32_id),
            ynn_status_success);
  uint32_t zero_fp32_id = YNN_INVALID_VALUE_ID;
  ASSERT_EQ(define_constant(subgraph.get(), 0.0f, zero_fp32_id),
            ynn_status_success);

  ASSERT_EQ(ynn_define_binary(subgraph.get(), ynn_binary_add, zp_id,
                              zero_int32_id, &zp_out_id, 0),
            ynn_status_success);
  ASSERT_EQ(ynn_define_binary(subgraph.get(), ynn_binary_add, scale_id,
                              zero_fp32_id, &scale_out_id, 0),
            ynn_status_success);

  // Create runtime
  runtime_ptr runtime = create_runtime(subgraph, nullptr, 0);
  ASSERT_NE(runtime, nullptr);

  // Set shapes
  size_t a_shape[] = {2, 3};
  size_t b_shape[] = {3, 4};
  size_t a_zp_shape[] = {2, 1};
  size_t b_zp_shape[] = {1, 4};
  size_t a_scale_shape[] = {2, 1};
  size_t b_scale_shape[] = {1, 4};

  ASSERT_EQ(ynn_set_external_value_shape(runtime.get(), a_id, 2, a_shape),
            ynn_status_success);
  ASSERT_EQ(ynn_set_external_value_shape(runtime.get(), b_id, 2, b_shape),
            ynn_status_success);
  ASSERT_EQ(ynn_set_external_value_shape(runtime.get(), a_zp_id, 2, a_zp_shape),
            ynn_status_success);
  ASSERT_EQ(ynn_set_external_value_shape(runtime.get(), b_zp_id, 2, b_zp_shape),
            ynn_status_success);
  ASSERT_EQ(
      ynn_set_external_value_shape(runtime.get(), a_scale_id, 2, a_scale_shape),
      ynn_status_success);
  ASSERT_EQ(
      ynn_set_external_value_shape(runtime.get(), b_scale_id, 2, b_scale_shape),
      ynn_status_success);

  // Set data
  std::vector<int8_t> a_data = {1, 2, 3, -1, 0, 1};
  std::vector<int8_t> b_data = {2, 1, 0, -1, 0, 1, 2, 3, 1, 1, 1, 1};
  std::vector<int32_t> a_zp_data = {1, -2};
  std::vector<int32_t> b_zp_data = {0, 2, -1, 1};
  std::vector<float> a_scale_data = {0.1f, 0.2f};
  std::vector<float> b_scale_data = {0.5f, 1.0f, 2.0f, 0.2f};

  std::vector<int32_t> zp_out_data(8, 0);
  std::vector<float> scale_out_data(8, 0.0f);

  ASSERT_EQ(ynn_set_external_value_data(runtime.get(), a_id, a_data.data()),
            ynn_status_success);
  ASSERT_EQ(ynn_set_external_value_data(runtime.get(), b_id, b_data.data()),
            ynn_status_success);
  ASSERT_EQ(
      ynn_set_external_value_data(runtime.get(), a_zp_id, a_zp_data.data()),
      ynn_status_success);
  ASSERT_EQ(
      ynn_set_external_value_data(runtime.get(), b_zp_id, b_zp_data.data()),
      ynn_status_success);
  ASSERT_EQ(ynn_set_external_value_data(runtime.get(), a_scale_id,
                                        a_scale_data.data()),
            ynn_status_success);
  ASSERT_EQ(ynn_set_external_value_data(runtime.get(), b_scale_id,
                                        b_scale_data.data()),
            ynn_status_success);
  ASSERT_EQ(
      ynn_set_external_value_data(runtime.get(), zp_out_id, zp_out_data.data()),
      ynn_status_success);
  ASSERT_EQ(ynn_set_external_value_data(runtime.get(), scale_out_id,
                                        scale_out_data.data()),
            ynn_status_success);

  // Run
  ASSERT_EQ(ynn_reshape_runtime(runtime.get()), ynn_status_success);
  ASSERT_EQ(ynn_invoke_runtime(runtime.get()), ynn_status_success);

  // Verify zero point
  std::vector<int32_t> expected_zp = {3, 9, 0, 6, -6, 6, -12, 0};
  for (size_t i = 0; i < 8; ++i) {
    EXPECT_EQ(zp_out_data[i], expected_zp[i]) << "at index " << i;
  }

  // Verify scale
  std::vector<float> expected_scale = {0.05f, 0.1f, 0.2f, 0.02f,
                                       0.1f,  0.2f, 0.4f, 0.04f};
  for (size_t i = 0; i < 8; ++i) {
    EXPECT_NEAR(scale_out_data[i], expected_scale[i], 1e-6) << "at index " << i;
  }
}

}  // namespace
}  // namespace ynn
