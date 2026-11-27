// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <vector>

#include <gtest/gtest.h>
#include "ynnpack/base/bfloat16.h"
#include "ynnpack/base/test/tensor.h"
#include "ynnpack/include/ynnpack.h"
#include "ynnpack/subgraph/test/subgraph_builder.h"

namespace ynn {
namespace {

// When a static_broadcast has multiple consumers, fusion cannot commute a
// consumer's unary op (such as a bf16->fp32 convert inserted by
// ynn_define_binary) before the broadcast. The unary op then receives a buffer
// whose innermost dimension has stride 0 (not contiguous), which previously
// triggered an assertion failure in debug builds and out-of-bounds reads in
// release builds.
TEST(UnaryBroadcastTest, MultiConsumerStaticBroadcastBf16Maximum) {
  constexpr size_t kRows = 256;
  constexpr size_t kCols = 16;
  const std::vector<size_t> scalar_shape = {1, 1};
  const std::vector<size_t> shape = {kRows, kCols};

  SubgraphBuilder subgraph(5);
  const uint32_t x_id = 0;
  const uint32_t y_id = 1;
  const uint32_t z_id = 2;
  const uint32_t out0_id = 3;
  const uint32_t out1_id = 4;

  subgraph.AddInput(ynn_type_bf16, shape, x_id)
      .AddInput(ynn_type_bf16, shape, y_id)
      .AddInput(ynn_type_bf16, scalar_shape, z_id)
      .AddOutput(ynn_type_bf16, shape, out0_id)
      .AddOutput(ynn_type_bf16, shape, out1_id);

  uint32_t z_bcast_id = YNN_INVALID_VALUE_ID;
  subgraph.AddTensor(ynn_type_bf16, shape, z_bcast_id);

  subgraph.AddStaticBroadcast(shape, z_id, z_bcast_id)
      .AddBinary(ynn_binary_max, x_id, z_bcast_id, out0_id)
      .AddBinary(ynn_binary_max, y_id, z_bcast_id, out1_id);

  Runtime runtime(subgraph.GetSubgraph());
  ASSERT_EQ(runtime.Status(), ynn_status_success);

  Tensor<bfloat16> x(shape);
  Tensor<bfloat16> y(shape);
  Tensor<bfloat16> z(scalar_shape);
  for (size_t i = 0; i < kRows; ++i) {
    for (size_t j = 0; j < kCols; ++j) {
      x(i, j) = static_cast<bfloat16>(static_cast<float>(i % 7) - 3.0f);
      y(i, j) = static_cast<bfloat16>(static_cast<float>(j % 5) - 2.0f);
    }
  }
  z(0, 0) = static_cast<bfloat16>(0.0f);

  Tensor<bfloat16> out0(shape);
  Tensor<bfloat16> out1(shape);

  runtime.ReshapeExternalTensor(shape, x.data(), x_id)
      .ReshapeExternalTensor(shape, y.data(), y_id)
      .ReshapeExternalTensor(scalar_shape, z.data(), z_id)
      .ReshapeRuntime();
  ASSERT_EQ(runtime.Status(), ynn_status_success);

  runtime.SetupExternalTensor(out0.data(), out0_id)
      .SetupExternalTensor(out1.data(), out1_id)
      .InvokeRuntime();
  ASSERT_EQ(runtime.Status(), ynn_status_success);

  for (size_t i = 0; i < kRows; ++i) {
    for (size_t j = 0; j < kCols; ++j) {
      EXPECT_EQ(static_cast<float>(out0(i, j)),
                std::max(static_cast<float>(x(i, j)), 0.0f));
      EXPECT_EQ(static_cast<float>(out1(i, j)),
                std::max(static_cast<float>(y(i, j)), 0.0f));
    }
  }
}

}  // namespace
}  // namespace ynn
