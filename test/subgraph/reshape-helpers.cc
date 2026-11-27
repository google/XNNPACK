// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <cstddef>

#include <gtest/gtest.h>
#include "include/xnnpack.h"
#include "src/xnnpack/common.h"
extern "C" {
#include "src/xnnpack/reshape-helpers.h"
}
#include "src/xnnpack/subgraph.h"

namespace xnnpack {
namespace {

TEST(ResizeHelpers, UnaryAcceptsMaximumRank) {
  xnn_operator_data opdata = {};
  opdata.inputs[0] = 0;
  opdata.outputs[0] = 1;

  xnn_runtime_value values[2] = {};
  values[0].shape.num_dims = XNN_MAX_TENSOR_DIMS;
  values[0].size = sizeof(float);
  values[1].size = sizeof(float);
  values[1].type = xnn_value_type_dense_tensor;
  values[1].datatype = xnn_datatype_fp32;
  for (size_t i = 0; i < XNN_MAX_TENSOR_DIMS; ++i) {
    values[0].shape.dim[i] = 1;
  }

  EXPECT_EQ(xnn_status_success,
            resize_unary_elementwise_output_tensor(
                &opdata, values, 2, /*old_workspace_size=*/0, nullptr));
  EXPECT_EQ(values[1].shape.num_dims, XNN_MAX_TENSOR_DIMS);
}

TEST(ResizeHelpers, UnaryRejectsOversizedRank) {
  xnn_operator_data opdata = {};
  opdata.inputs[0] = 0;
  opdata.outputs[0] = 1;

  xnn_runtime_value values[2] = {};
  values[0].shape.num_dims = XNN_MAX_TENSOR_DIMS + 1;
  values[0].size = sizeof(float);
  values[1].size = sizeof(float);
  values[1].type = xnn_value_type_dense_tensor;
  values[1].datatype = xnn_datatype_fp32;
  for (size_t i = 0; i < XNN_MAX_TENSOR_DIMS; ++i) {
    values[0].shape.dim[i] = 1;
  }

  EXPECT_EQ(xnn_status_unsupported_parameter,
            resize_unary_elementwise_output_tensor(
                &opdata, values, 2, /*old_workspace_size=*/0, nullptr));
}

}  // namespace
}  // namespace xnnpack
