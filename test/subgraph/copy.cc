// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <cassert>
#include <chrono>
#include <cstddef>
#include <cstdint>
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
void TestImpl(size_t rank) {
  ReplicableRandomDevice rng;

  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));

  for (auto _ : FuzzTest(std::chrono::milliseconds(250))) {
    xnn_quantization_params quantization =
        random_quantization(xnn_datatype_of<T>(), rng);

    // Define subgraph
    SubgraphTester subgraph(2);
    subgraph.AddInputTensor(rank, xnn_datatype_of<T>(), quantization, 0)
        .AddOutputTensor(rank, xnn_datatype_of<T>(), quantization, 1)
        .AddCopy(0, 1)
        .CreateRuntime();

    for (int reshape = 0; reshape < 2; ++reshape) {
      std::vector<size_t> shape = random_shape(rng, rank);

      Tensor<T> input(shape, xnnpack::XnnExtraBytes);
      DatatypeGenerator<T> generator(quantization);
      input.generate([&]() { return generator(rng); });

      // Check reshaped shape is correct
      subgraph.ReshapeExternalTensor(shape, input.base(), 0).ReshapeRuntime();
      ASSERT_EQ(subgraph.GetExternalTensorShape(1), input.extents());

      // Run subgraph
      Tensor<T> output(input.extents());
      subgraph.SetupExternalTensor(output.base(), 1)
          .SetupRuntime()
          .InvokeRuntime();

      // Verify results.
      ASSERT_THAT(output, testing::ElementsAreArray(input));
    }
  }
}

template <typename T>
class Copy : public ::testing::TestWithParam<int> {};

using CopyQS8 = Copy<quantized<int8_t>>;
using CopyQU8 = Copy<quantized<uint8_t>>;
using CopyBF16 = Copy<xnn_bfloat16>;
using CopyF16 = Copy<xnn_float16>;
using CopyF32 = Copy<float>;

TEST_P(CopyQS8, test) { TestImpl<quantized<int8_t>>(GetParam()); }
TEST_P(CopyQU8, test) { TestImpl<quantized<uint8_t>>(GetParam()); }
TEST_P(CopyBF16, test) { TestImpl<xnn_bfloat16>(GetParam()); }
TEST_P(CopyF16, test) { TestImpl<xnn_float16>(GetParam()); }
TEST_P(CopyF32, test) { TestImpl<float>(GetParam()); }

auto rank_params = testing::Range(1, XNN_MAX_TENSOR_DIMS);
INSTANTIATE_TEST_SUITE_P(Copy, CopyQS8, rank_params);
INSTANTIATE_TEST_SUITE_P(Copy, CopyQU8, rank_params);
INSTANTIATE_TEST_SUITE_P(Copy, CopyBF16, rank_params);
INSTANTIATE_TEST_SUITE_P(Copy, CopyF16, rank_params);
INSTANTIATE_TEST_SUITE_P(Copy, CopyF32, rank_params);

}  // namespace xnnpack
