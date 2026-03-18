// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <random>
#include <type_traits>
#include <vector>

#include <gtest/gtest.h>
#include "ynnpack/base/test/fuzz_test.h"
#include "ynnpack/base/test/random.h"
#include "ynnpack/base/test/tensor.h"
#include "ynnpack/base/type.h"
#include "ynnpack/include/ynnpack.h"
#include "ynnpack/subgraph/test/subgraph_builder.h"

namespace ynn {
namespace {

template <typename A, typename X, typename Index>
void TestLut() {
  ReplicableRandomDevice rng;
  std::uniform_int_distribution<size_t> rank_dist(1, YNN_MAX_TENSOR_RANK);
  for (auto _ : FuzzTest(std::chrono::milliseconds(100))) {
    size_t rank = rank_dist(rng);
    // We want the total number of elements to be reasonable, so choose max_dim
    // such that a random shape of rank `p.rank` produces this max size.
    constexpr size_t max_size = 1024;
    const size_t max_dim = static_cast<size_t>(std::ceil(
        std::pow(static_cast<double>(max_size),
                 1.0 / static_cast<double>(std::max<size_t>(1, rank)))));
    quantization_params a_quantization = random_quantization(A(), rng);
    quantization_params output_quantization = random_quantization(X(), rng);

    std::vector<X> lut_data(256);
    std::uniform_int_distribution<int> lut_dist(std::numeric_limits<X>::min(),
                                                std::numeric_limits<X>::max());
    std::generate(lut_data.begin(), lut_data.end(),
                  [&]() { return static_cast<X>(lut_dist(rng)); });

    SubgraphBuilder subgraph(2);
    uint32_t input_id = 0;
    uint32_t output_id = 1;
    uint32_t lut_id = YNN_INVALID_VALUE_ID;

    subgraph.AddInput(type_of<A>(), rank, input_id, a_quantization)
        .AddOutput(type_of<X>(), rank, output_id, output_quantization)
        .AddTensor(type_of<X>(), {256}, lut_id, lut_data.data());

    ASSERT_EQ(
        ynn_define_lut(subgraph.GetSubgraph(), input_id, lut_id, &output_id, 0),
        ynn_status_success);

    Runtime runtime(subgraph.GetSubgraph());
    ASSERT_EQ(runtime.Status(), ynn_status_success);

    for (int reshape = 0; reshape < 2; ++reshape) {
      std::vector<size_t> shape = random_shape(rng, rank, 1, max_dim);

      Tensor<A> a(shape);
      Tensor<X> output(shape);

      std::uniform_int_distribution<int> a_dist(std::numeric_limits<A>::min(),
                                                std::numeric_limits<A>::max());
      std::generate(a.data(), a.data() + a.size(),
                    [&]() { return static_cast<A>(a_dist(rng)); });

      runtime.ReshapeExternalTensor(shape, a.data(), input_id).ReshapeRuntime();
      ASSERT_EQ(runtime.GetExternalTensorShape(output_id), shape);
      runtime.SetupExternalTensor(output.data(), output_id).InvokeRuntime();

      for (size_t i = 0; i < a.size(); ++i) {
        size_t index = static_cast<Index>(a.data()[i]);
        ASSERT_EQ(output.data()[i], lut_data[index]);
      }
    }
  }
}

TEST(LutTest, LutUint8) { TestLut<uint8_t, uint8_t, uint8_t>(); }

TEST(LutTest, LutInt8) { TestLut<int8_t, int8_t, uint8_t>(); }

}  // namespace
}  // namespace ynn
