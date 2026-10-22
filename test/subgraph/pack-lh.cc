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
#include "src/xnnpack/config.h"
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

TEST(PackLH, ReshapeOverflowInputStride) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));

  xnn_operator_t pack_lh_op = nullptr;
  const enum xnn_status create_status =
      xnn_create_pack_lh_x8(/*flags=*/0, &pack_lh_op);
  if (create_status == xnn_status_unsupported_hardware) {
    GTEST_SKIP();
  }
  ASSERT_EQ(xnn_status_success, create_status);
  ASSERT_NE(pack_lh_op, nullptr);

  size_t output_size_bytes = 0;
  EXPECT_EQ(
      xnn_status_out_of_memory,
      xnn_reshape_pack_lh_x8(pack_lh_op, /*num_groups=*/1, /*batch_size=*/1,
                             /*channels=*/SIZE_MAX,
                             &output_size_bytes, /*threadpool=*/nullptr));
  EXPECT_EQ(SIZE_MAX, output_size_bytes);

  EXPECT_EQ(xnn_status_success, xnn_delete_operator(pack_lh_op));
}

TEST(PackLH, RuntimeTensorSizeOverflow) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));
  const struct xnn_gemm_config* gemm_config =
      xnn_init_qp8_f32_qc8w_gemm_config();
  if (gemm_config == nullptr) {
    GTEST_SKIP();
  }

  xnn_runtime_value value = {};
  value.type = xnn_value_type_dense_tensor;
  value.datatype = xnn_datatype_qpint8;
  value.gemm_config = gemm_config;
  value.shape.num_dims = 3;
  value.shape.dim[0] = 2;
  value.shape.dim[1] = (SIZE_MAX / 2) + 1;
  value.shape.dim[2] = 1;
  value.flags = XNN_FLAG_SQUASH_GROUPS;

  EXPECT_EQ(SIZE_MAX, xnn_runtime_tensor_get_size(&value));
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
