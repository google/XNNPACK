// Copyright 2023 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstddef>
#include <cstdlib>
#include <random>
#include <vector>

#include <gtest/gtest.h>
#include "include/xnnpack.h"
#include "src/xnnpack/buffer.h"
#include "src/xnnpack/datatype.h"
#include "src/xnnpack/math.h"
#include "test/replicable_random_device.h"
#include "test/subgraph/subgraph-tester.h"

namespace xnnpack {

namespace {

static const float kMaxR = 10.0f;
static const float kMaxI = 1.0;

};  // namespace

template <typename T>
Tensor<T> ReferenceImpl(Tensor<T> x, Tensor<T> w) {
  assert(x.rank() == 5);
  const size_t batch_size = x.extents()[0];
  const size_t tokens = x.extents()[1];
  const size_t heads = x.extents()[2];
  const size_t channels = x.extents()[4];

  Tensor<T> y({batch_size, tokens, heads, 2, channels});
  for (size_t n = 0; n < batch_size; n++) {
    for (size_t t = 0; t < tokens; ++t) {
      for (size_t h = 0; h < heads; ++h) {
        for (size_t c = 0; c < channels; ++c) {
          y(n, t, h, 0, c) =
              x(n, t, h, 0, c) * w(t, 0, c) - x(n, t, h, 1, c) * w(t, 1, c);
          y(n, t, h, 1, c) =
              x(n, t, h, 0, c) * w(t, 1, c) + x(n, t, h, 1, c) * w(t, 0, c);
        }
      }
    }
  }
  return y;
}

template <typename T>
void TestImpl() {
  ReplicableRandomDevice rng;

  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));

  // Define subgraph
  SubgraphTester subgraph(3);
  subgraph.AddInputTensor(4, xnn_datatype_of<T>(), 0)
      .AddInputTensor(2, xnn_datatype_of<T>(), 1)
      .AddOutputTensor(4, xnn_datatype_of<T>(), 2)
      .AddRoPE(0, 1, 2);
  xnn_status status = subgraph.CreateRuntime();
  if (status == xnn_status_unsupported_hardware) {
    GTEST_SKIP();
    return;
  }

  for (auto _ : FuzzTest(std::chrono::milliseconds(1000))) {
    std::vector<size_t> shape = random_shape(rng, 4);
    const size_t batch_size = shape[0];
    const size_t tokens = shape[1];
    const size_t heads = shape[2];
    const size_t channels = shape[3];
    shape[3] *= 2;
    const size_t max_tokens =
        tokens + std::uniform_int_distribution<>(0, 10)(rng);

    // The last dimension is split into 2 dimensions {re, im}, channels
    Tensor<T> input({batch_size, tokens, heads, 2, channels}, XnnExtraBytes);
    Tensor<T> weights({max_tokens, 2, channels}, XnnExtraBytes);
    DatatypeGenerator<T> gen_r(1.0f, kMaxR);
    DatatypeGenerator<T> gen_i(0.01f, kMaxI);
    input.slice(3, 0).generate([&]() { return gen_r(rng); });
    input.slice(3, 1).generate([&]() { return gen_i(rng); });
    weights.slice(1, 0).generate([&]() { return gen_r(rng); });
    weights.slice(1, 1).generate([&]() { return gen_i(rng); });

    Tensor<T> expected = ReferenceImpl(input, weights);

    // Check reshaped shape is correct
    subgraph.ReshapeExternalTensor(shape, input.base(), 0)
        .ReshapeExternalTensor({max_tokens, channels * 2}, weights.base(), 1)
        .ReshapeRuntime();
    ASSERT_EQ(subgraph.GetExternalTensorShape(2), shape);

    // Run subgraph
    Tensor<T> output({batch_size, tokens, heads, 2, channels});
    subgraph.SetupExternalTensor(output.base(), 2)
        .SetupRuntime()
        .InvokeRuntime();

    // Verify results.
    const float max_input_val = std::max(kMaxR, kMaxR);
    const float abs_tol =
        max_input_val * max_input_val * xnnpack::epsilon(xnn_datatype_of<T>());
    for (const auto& i : EnumerateIndices(output.extents())) {
      ASSERT_NEAR(output(i), expected(i), abs_tol);
    }
  }
}

TEST(RoPEF16, test) { TestImpl<xnn_float16>(); }
TEST(RoPEF32, test) { TestImpl<float>(); }

}  // namespace xnnpack
