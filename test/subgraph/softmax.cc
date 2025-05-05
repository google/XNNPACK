// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <cassert>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "include/xnnpack.h"
#include "src/xnnpack/buffer.h"
#include "src/xnnpack/datatype.h"
#include "src/xnnpack/math.h"
#include "test/replicable_random_device.h"
#include "test/subgraph/subgraph-tester.h"

namespace xnnpack {

template <typename T>
Tensor<T> softmax(Tensor<T> x) {
  Tensor<T> y(x.extents());
  std::vector<size_t> batch_dims = x.extents();
  size_t channels = x.extents().back();
  batch_dims.pop_back();
  for (std::vector<size_t> i : EnumerateIndices(batch_dims)) {
    i.push_back(0);
    const T* x_i = &x(i);
    T* y_i = &y(i);
    // softmax(x) = softmax(x - C) for any C. To avoid computing exp of large
    // values which overflow, we use C = max(x).
    double max = *std::max_element(x_i, x_i + channels);
    double sum_exp = 0.0;
    for (size_t c = 0; c < channels; c++) {
      sum_exp += std::exp(static_cast<double>(x_i[c]) - max);
    }
    for (size_t c = 0; c < channels; c++) {
      y_i[c] = std::exp(static_cast<double>(x_i[c]) - max) / sum_exp;
    }
  }
  return y;
}

template <typename T>
void TestImpl(size_t rank) {
  ReplicableRandomDevice rng;

  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));

  // Define subgraph
  SubgraphTester subgraph(2);
  subgraph.AddInputTensor(rank, xnn_datatype_of<T>(), 0)
      .AddOutputTensor(rank, xnn_datatype_of<T>(), 1)
      .AddSoftmax(0, 1);
  xnn_status status = subgraph.CreateRuntime();
  if (status == xnn_status_unsupported_hardware) {
    GTEST_SKIP();
    return;
  }

  for (auto _ : FuzzTest(std::chrono::milliseconds(500))) {
    std::vector<size_t> shape = random_shape(rng, rank);

    Tensor<T> input(shape, xnnpack::XnnExtraBytes);
    DatatypeGenerator<T> generator(-20.0f, 20.0f);
    input.generate([&]() { return generator(rng); });

    Tensor<T> expected = softmax(input);

    // Check reshaped shape is correct
    subgraph.ReshapeExternalTensor(shape, input.base(), 0).ReshapeRuntime();
    ASSERT_EQ(subgraph.GetExternalTensorShape(1), expected.extents());

    // Run subgraph
    // Softmax reads from the output assuming XNN_EXTRA_BYTES exist.
    Tensor<T> output(expected.extents(), xnnpack::XnnExtraBytes);
    subgraph.SetupExternalTensor(output.base(), 1)
        .SetupRuntime()
        .InvokeRuntime();

    // Verify results.
    const float tolerance = sizeof(T) == 2 ? 1.0e-2f : 1.0e-4f;
    ASSERT_THAT(output,
                testing::Pointwise(testing::FloatNear(tolerance), expected));
  }
}

template <typename T>
class Softmax : public ::testing::TestWithParam<int> {};

using SoftmaxF16 = Softmax<xnn_float16>;
using SoftmaxF32 = Softmax<float>;

TEST_P(SoftmaxF16, test) { TestImpl<xnn_float16>(GetParam()); }
TEST_P(SoftmaxF32, test) { TestImpl<float>(GetParam()); }

auto rank_params = testing::Range(1, XNN_MAX_TENSOR_DIMS);
INSTANTIATE_TEST_SUITE_P(Softmax, SoftmaxF16, rank_params);
INSTANTIATE_TEST_SUITE_P(Softmax, SoftmaxF32, rank_params);

}  // namespace xnnpack
