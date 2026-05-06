// Copyright 2026 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <array>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "include/xnnpack.h"
#include "litert/tensor/arithmetic.h"
#include "litert/tensor/backends/xnnpack/arithmetic.h"
#include "litert/tensor/backends/xnnpack/conversion.h"
#include "litert/tensor/datatypes.h"
#include "litert/tensor/tensor.h"
#include "litert/tensor/utils/matchers.h"

namespace xnnpack {
namespace {

namespace lrt = ::litert::tensor;
using XTensor = lrt::Tensor<lrt::XnnpackMixinTag>;

using testing::ElementsAreArray;

TEST(PlanningTest, ReshapingToBroadcastWorks) {
  std::unique_ptr<lrt::XnnpackGraph> graph;
  uint32_t a_id = XNN_INVALID_VALUE_ID;
  uint32_t b_id = XNN_INVALID_VALUE_ID;
  uint32_t c_id = XNN_INVALID_VALUE_ID;

  {
    XTensor a({.type = lrt::Type::kI16, .shape = {3, 3}});
    XTensor b({.type = lrt::Type::kI16, .shape = {3, 3}});

    XTensor c = Add(Cast(a, lrt::Type::kFP32), Cast(b, lrt::Type::kFP32));
    c.SetShape({3, 3});
    c = Cast(c, lrt::Type::kI16);

    LRT_TENSOR_ASSERT_OK_AND_ASSIGN(graph, lrt::BuildXnnpackGraph({c}));
    LRT_TENSOR_ASSERT_OK_AND_ASSIGN(const size_t a_idx, graph->Lookup(a));
    LRT_TENSOR_ASSERT_OK_AND_ASSIGN(const size_t b_idx, graph->Lookup(b));
    LRT_TENSOR_ASSERT_OK_AND_ASSIGN(const size_t c_idx, graph->Lookup(c));
    a_id = graph->values()[a_idx].id;
    b_id = graph->values()[b_idx].id;
    c_id = graph->values()[c_idx].id;
  }

  xnn_runtime_t runtime;
  xnn_create_runtime_v4(graph->subgraph(), /*weights_cache=*/nullptr,
                        /*workspace=*/nullptr, /*threadpool=*/nullptr,
                        /*flags=*/0, &runtime);

  std::array<uint16_t, 9> a_data{1, 2, 3, 4, 5, 6, 7, 8, 9};
  std::array<uint16_t, 9> b_data{1, 2, 3, 4, 5, 6, 7, 8, 9};
  std::array<uint16_t, 9> c_data{};

  std::array<xnn_external_value, 3> values{
      xnn_external_value{.id = a_id, .data = a_data.data()},
      xnn_external_value{.id = b_id, .data = b_data.data()},
      xnn_external_value{.id = c_id, .data = c_data.data()},
  };

  xnn_reshape_runtime(runtime);
  xnn_setup_runtime_v2(runtime, values.size(), values.data());
  xnn_invoke_runtime(runtime);

  EXPECT_THAT(c_data, ElementsAreArray({2, 4, 6, 8, 10, 12, 14, 16, 18}));

  // Change `a` so that the operation now needs a broadcast. The internal buffer
  // after the cast can't be reused by the add op to write its output to.
  std::array<size_t, 2> new_a_dims{3, 1};
  xnn_reshape_external_value(runtime, a_id, new_a_dims.size(),
                             new_a_dims.data());

  xnn_reshape_runtime(runtime);
  xnn_setup_runtime_v2(runtime, values.size(), values.data());
  xnn_invoke_runtime(runtime);

  EXPECT_THAT(c_data, ElementsAreArray({2, 3, 4, 6, 7, 8, 10, 11, 12}));
}

}  // namespace
}  // namespace xnnpack
