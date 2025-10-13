// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include "ynnpack/xnnpack/dynamic_quantization.h"

#include <cassert>
#include <cstddef>
#include <cstdint>
#include <numeric>
#include <tuple>
#include <vector>

#include <gtest/gtest.h>
#include "ynnpack/base/base.h"
#include "ynnpack/base/half.h"
#include "ynnpack/base/test/fuzz_test.h"
#include "ynnpack/base/test/random.h"
#include "ynnpack/base/test/tensor.h"
#include "ynnpack/base/test/util.h"
#include "ynnpack/base/type.h"
#include "ynnpack/include/ynnpack.h"
#include "ynnpack/subgraph/test/subgraph_builder.h"

using ynn::to_string;  // NOLINT(misc-unused-using-decls)

namespace ynn {

template <typename T>
void TestImpl(T, size_t rank) {
  ReplicableRandomDevice rng;
  TypeGenerator<T> input_gen(type_info<T>::min() / 4.0f,
                             type_info<T>::max() / 4.0f);

  for (int32_t num_nonbatch_axes = 1; num_nonbatch_axes <= rank;
       ++num_nonbatch_axes) {
    // Define subgraph
    const uint32_t input_id = 0;
    const uint32_t output_id = 1;
    const uint32_t scale_id = 2;
    const uint32_t zero_point_id = 3;
    SubgraphBuilder subgraph(4);
    subgraph.AddInput(type_of<T>(), rank, input_id)
        .AddOutput(type_of<int8_t>(), rank, output_id, zero_point_id, scale_id)
        .AddOutput(type_of<float>(), rank, scale_id)
        .AddOutput(type_of<int32_t>(), rank, zero_point_id);
    ynn::compute_qd8_params(subgraph.GetSubgraph(), num_nonbatch_axes, input_id,
                            output_id);
    subgraph.AddUnary(ynn_unary_convert, input_id, output_id);

    Runtime runtime(subgraph.GetSubgraph());
    ASSERT_EQ(runtime.Status(), ynn_status_success);

    for (int reshape = 0; reshape < 2; ++reshape) {
      std::vector<size_t> shape = random_shape(rng, rank);

      Tensor<T> input(shape);
      input.generate([&]() { return input_gen(rng); });

      // Check reshaped shape is correct
      std::vector<size_t> params_shape(shape);
      for (size_t i = 0; i < num_nonbatch_axes; ++i) {
        params_shape[params_shape.size() - 1 - i] = 1;
      }
      runtime.ReshapeExternalTensor(shape, input.base(), input_id)
          .ReshapeRuntime();
      ASSERT_EQ(runtime.GetExternalTensorShape(output_id), shape);
      ASSERT_EQ(runtime.GetExternalTensorShape(scale_id), params_shape);
      ASSERT_EQ(runtime.GetExternalTensorShape(zero_point_id), params_shape);

      // Run subgraph
      Tensor<int8_t> output(shape);
      Tensor<int32_t> zero_point(params_shape);
      Tensor<float> scale(params_shape);
      runtime.SetupExternalTensor(output.base(), output_id)
          .SetupExternalTensor(zero_point.base(), zero_point_id)
          .SetupExternalTensor(scale.base(), scale_id)
          .InvokeRuntime();

      // Verify results.
      broadcast_extent_1(scale);
      broadcast_extent_1(zero_point);
      for (const auto& i : EnumerateIndices(shape)) {
        ASSERT_NEAR(quantize<int8_t>(input(i), 1.0f / scale(i), zero_point(i)),
                    output(i), 1);
      }
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
    default:
      YNN_UNREACHABLE;
  }
}

class ConvertQd8 : public testing::TestWithParam<std::tuple<ynn_type, int>> {};

TEST_P(ConvertQd8, Scalar) {
  SwitchType(std::get<0>(GetParam()),
             [&](auto a_type) { TestImpl(a_type, std::get<1>(GetParam())); });
}

TEST_P(ConvertQd8, Vector) {
  SwitchType(std::get<0>(GetParam()),
             [&](auto a_type) { TestImpl(a_type, std::get<1>(GetParam())); });
}

INSTANTIATE_TEST_SUITE_P(
    Test, ConvertQd8,
    testing::Combine(testing::Values(ynn_type_fp32, ynn_type_fp16),
                     testing::Range(1, YNN_MAX_TENSOR_RANK)),
    [](const testing::TestParamInfo<ConvertQd8::ParamType>& info) {
      return test_param_to_string(info);
    });

}  // namespace ynn
