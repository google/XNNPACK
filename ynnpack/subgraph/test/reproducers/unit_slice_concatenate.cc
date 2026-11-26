// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <cstddef>
#include <cstdint>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "ynnpack/base/test/tensor.h"
#include "ynnpack/include/ynnpack.h"
#include "ynnpack/subgraph/test/subgraph_builder.h"

namespace ynn {
namespace {

using ::testing::ElementsAreArray;

// Swaps the two halves of the innermost dimension of a [rows, 2 * width]
// input with static slices and a concatenation, then negates the result so
// that the concatenation output is an intermediate value Slinky can alias the
// slices into.
Tensor<float> SwapHalvesAndNegate(Tensor<float>& input, size_t width,
                                  bool optimize) {
  const size_t rows = input.extents()[0];
  const int64_t w = static_cast<int64_t>(width);
  uint32_t lo_id = YNN_INVALID_VALUE_ID;
  uint32_t hi_id = YNN_INVALID_VALUE_ID;
  uint32_t swapped_id = YNN_INVALID_VALUE_ID;
  SubgraphBuilder subgraph(2);
  subgraph.AddInput(ynn_type_fp32, {rows, 2 * width}, 0)
      .AddOutput(ynn_type_fp32, {rows, 2 * width}, 1)
      .AddTensor(ynn_type_fp32, {rows, width}, lo_id)
      .AddTensor(ynn_type_fp32, {rows, width}, hi_id)
      .AddTensor(ynn_type_fp32, {rows, 2 * width}, swapped_id)
      .AddSlice({1}, {0}, {w}, {1}, 0, lo_id)
      .AddSlice({1}, {w}, {2 * w}, {1}, 0, hi_id)
      .AddConcatenate(1, {hi_id, lo_id}, swapped_id)
      .AddUnary(ynn_unary_negate, swapped_id, 1);
  Runtime runtime(subgraph.GetSubgraph(), /*scheduler=*/nullptr, /*flags=*/0,
                  optimize);
  EXPECT_EQ(runtime.Status(), ynn_status_success);
  Tensor<float> output({rows, 2 * width});
  runtime.ReshapeExternalTensor(input.extents(), input.base(), 0)
      .ReshapeRuntime()
      .SetupExternalTensor(output.base(), 1)
      .InvokeRuntime();
  return output;
}

TEST(UnitSliceConcatenate, SwapHalves) {
  for (size_t width : {1, 2, 3}) {
    for (size_t rows : {1, 2, 5}) {
      for (bool optimize : {false, true}) {
        Tensor<float> input({rows, 2 * width});
        std::vector<float> expected;
        for (size_t r = 0; r < rows; ++r) {
          for (size_t c = 0; c < 2 * width; ++c) {
            input(r, c) = static_cast<float>(r * 2 * width + c + 1);
          }
          for (size_t c = 0; c < width; ++c) {
            expected.push_back(-input(r, width + c));
          }
          for (size_t c = 0; c < width; ++c) {
            expected.push_back(-input(r, c));
          }
        }
        EXPECT_THAT(SwapHalvesAndNegate(input, width, optimize),
                    ElementsAreArray(expected))
            << "width=" << width << " rows=" << rows
            << " optimize=" << optimize;
      }
    }
  }
}

}  // namespace
}  // namespace ynn
