// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include <gtest/gtest.h>
#include "include/xnnpack.h"
#include "src/xnnpack/datatype.h"
#include "src/xnnpack/internal.h"
#include "src/xnnpack/subgraph.h"

namespace xnnpack {

#ifndef XNNPACK_USE_YNNPACK
TEST(PackLH, ReshapeOverflowInputElements) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));

  xnn_subgraph_t subgraph = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_subgraph(2, 0, &subgraph));
  std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)> auto_subgraph(
      subgraph, xnn_delete_subgraph);

  uint32_t input_id = XNN_INVALID_VALUE_ID;
  const size_t input_dims[2] = {2, 4};
  ASSERT_EQ(
      xnn_status_success,
      xnn_define_tensor_value(
          subgraph, xnn_datatype_fp32, 2, input_dims, nullptr,
          /*external_id=*/0, XNN_VALUE_FLAG_EXTERNAL_INPUT, &input_id));

  uint32_t output_id = XNN_INVALID_VALUE_ID;
  const size_t output_dims[2] = {2, 4};
  ASSERT_EQ(
      xnn_status_success,
      xnn_define_tensor_value(
          subgraph, xnn_datatype_fp32, 2, output_dims, nullptr,
          /*external_id=*/1, XNN_VALUE_FLAG_EXTERNAL_OUTPUT, &output_id));

  ASSERT_EQ(
      xnn_status_success,
      xnn_define_pack_lh(subgraph, input_id, output_id, 0));

  xnn_runtime_t runtime = nullptr;
  const xnn_status status =
      xnn_create_runtime_v4(subgraph, nullptr, nullptr, nullptr, 0, &runtime);
  if (status == xnn_status_unsupported_hardware) {
    GTEST_SKIP();
  }
  ASSERT_EQ(xnn_status_success, status);
  std::unique_ptr<xnn_runtime, decltype(&xnn_delete_runtime)> auto_runtime(
      runtime, xnn_delete_runtime);

  runtime->values[input_id].shape.num_dims = 2;
  runtime->values[input_id].shape.dim[0] = SIZE_MAX;
  runtime->values[input_id].shape.dim[1] = 4;

  const enum xnn_status reshape_status = xnn_reshape_runtime(runtime);
  EXPECT_EQ(xnn_status_invalid_parameter, reshape_status);
}

TEST(PackLH, DefineRejectsUnsupportedDatatype) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));

  xnn_subgraph_t subgraph = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_subgraph(2, 0, &subgraph));
  std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)> auto_subgraph(
      subgraph, xnn_delete_subgraph);

  uint32_t input_id = XNN_INVALID_VALUE_ID;
  const size_t input_dims[2] = {2, 4};
  ASSERT_EQ(
      xnn_status_success,
      xnn_define_tensor_value(
          subgraph, xnn_datatype_int32, 2, input_dims, nullptr,
          /*external_id=*/0, XNN_VALUE_FLAG_EXTERNAL_INPUT, &input_id));

  uint32_t output_id = XNN_INVALID_VALUE_ID;
  ASSERT_EQ(
      xnn_status_success,
      xnn_define_tensor_value(
          subgraph, xnn_datatype_int32, 2, input_dims, nullptr,
          /*external_id=*/1, XNN_VALUE_FLAG_EXTERNAL_OUTPUT, &output_id));

  EXPECT_EQ(
      xnn_status_invalid_parameter,
      xnn_define_pack_lh(subgraph, input_id, output_id, 0));
}
#endif  // XNNPACK_USE_YNNPACK

}  // namespace xnnpack
