// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <random>
#include <tuple>
#include <vector>

#include <gtest/gtest.h>
#include "ynnpack/base/test/fuzz_test.h"
#include "ynnpack/base/test/random.h"
#include "ynnpack/base/test/tensor.h"
#include "ynnpack/base/test/util.h"
#include "ynnpack/base/to_string.h"
#include "ynnpack/base/type.h"
#include "ynnpack/include/ynnpack.h"
#include "ynnpack/subgraph/test/subgraph_builder.h"

// This needs to be in the global namespace for argument dependent lookup to
// work.
using ynn::to_string;  // NOLINT

namespace ynn {

using ::testing::Combine;
using ::testing::ValuesIn;

float EvalPolynomial(const std::vector<float>& coefficients, float x) {
  float y = 0.0f;
  float x_i = 1.0f;
  for (float c : coefficients) {
    y += c * x_i;
    x_i *= x;
  }
  return y;
}

template <typename A, typename X>
void TestPolynomial(A, X) {
  ReplicableRandomDevice rng;
  std::uniform_int_distribution<size_t> rank_dist(1, YNN_MAX_TENSOR_RANK);
  std::uniform_int_distribution<size_t> degree_dist(0, 3);
  const float max_abs_coefficient = 2.0f;
  const float max_abs_value = 10.0f;
  std::uniform_real_distribution<float> coeff_dist(-max_abs_coefficient,
                                                   max_abs_coefficient);
  for (auto _ : FuzzTest(std::chrono::milliseconds(100))) {
    size_t rank = rank_dist(rng);
    constexpr size_t max_size = 1024;
    const size_t max_dim = static_cast<size_t>(std::ceil(
        std::pow(static_cast<double>(max_size),
                 1.0 / static_cast<double>(std::max<size_t>(1, rank)))));
    quantization_params a_quantization = random_quantization(A(), rng);
    quantization_params output_quantization = random_quantization(X(), rng);

    const int degree = degree_dist(rng);
    std::vector<float> coefficients(degree + 1);
    for (float& c : coefficients) {
      c = coeff_dist(rng);
    }

    SubgraphBuilder subgraph(2);
    subgraph.AddInput(type_of<A>(), rank, 0, a_quantization)
        .AddOutput(type_of<X>(), rank, 1, output_quantization)
        .AddPolynomial(coefficients, 0, 1);

    Runtime runtime(subgraph.GetSubgraph());
    ASSERT_EQ(runtime.Status(), ynn_status_success);

    for (int reshape = 0; reshape < 2; ++reshape) {
      std::vector<size_t> shape = random_shape(rng, rank, 1, max_dim);

      Tensor<A> a(shape);
      Tensor<X> output(shape);

      fill_random(a.data(), a.size(), rng, -max_abs_value, max_abs_value,
                  a_quantization);

      runtime.ReshapeExternalTensor(shape, a.data(), 0).ReshapeRuntime();

      ASSERT_EQ(runtime.GetExternalTensorShape(1), shape);

      runtime.SetupExternalTensor(output.data(), 1).InvokeRuntime();
      for (const auto& i : EnumerateIndices(output.extents())) {
        float a_i = dequantize(a(i), a_quantization);
        float expected = EvalPolynomial(coefficients, a_i);
        if (is_quantized<X>()) {
          expected = quantize<X>(expected, output_quantization);
          ASSERT_NEAR(expected, output(i), 1);
        } else {
          const float tolerance = epsilon(type_of<X>()) * max_abs_value *
                                  std::pow(max_abs_value, degree);
          ASSERT_NEAR(expected, output(i), tolerance);
        }
      }
    }
  }
}

class Polynomial : public testing::TestWithParam<std::tuple<ynn_type>> {};

TEST_P(Polynomial, op) {
  ynn_type type = std::get<0>(GetParam());
  SwitchRealType(type, [&](auto type) { TestPolynomial(type, type); });
}

const ynn_type all_real_types[] = {
    ynn_type_int8, ynn_type_uint8, ynn_type_fp16, ynn_type_bf16, ynn_type_fp32,
};

INSTANTIATE_TEST_SUITE_P(UnaryTest, Polynomial,
                         Combine(ValuesIn(all_real_types)),
                         test_param_to_string<Polynomial::ParamType>);

}  // namespace ynn
