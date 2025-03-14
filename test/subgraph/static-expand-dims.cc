// Copyright 2024 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <numeric>
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

std::vector<size_t> mask_to_axes(uint32_t mask) {
  std::vector<size_t> axes;
  for (uint32_t i = 0; i < XNN_MAX_TENSOR_DIMS; ++i) {
    if (mask & (1 << i)) {
      axes.push_back(i);
    }
  }
  return axes;
}

template <typename T>
void TestImpl(size_t rank) {
  ReplicableRandomDevice rng;

  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));

  std::vector<size_t> perm(rank);
  std::iota(perm.begin(), perm.end(), 0);

  for (uint32_t mask = 1; mask < (1 << XNN_MAX_TENSOR_DIMS); ++mask) {
    std::vector<size_t> new_axes = mask_to_axes(mask);
    if (rank + new_axes.size() > XNN_MAX_TENSOR_DIMS) {
      continue;
    }
    if (std::any_of(new_axes.begin(), new_axes.end(),
                    [&](size_t i) { return i >= rank + new_axes.size(); })) {
      continue;
    }

    xnn_quantization_params quantization =
        random_quantization(xnn_datatype_of<T>(), rng);

    // Define subgraph
    SubgraphTester subgraph(2);
    subgraph.AddInputTensor(rank, xnn_datatype_of<T>(), quantization, 0)
        .AddOutputTensor(rank, xnn_datatype_of<T>(), quantization, 1)
        .AddExpandDims(new_axes, 0, 1)
        .CreateRuntime();

    for (int reshape = 0; reshape < 2; ++reshape) {
      std::vector<size_t> shape = random_shape(rng, rank);

      Tensor<T> input(shape, PaddingBytes{XNN_EXTRA_BYTES});
      DatatypeGenerator<T> generator(quantization);
      input.generate([&]() { return generator(rng); });

      Tensor<T> expected = input.expand_dims(new_axes);

      // Check reshape is correct
      subgraph.ReshapeExternalTensor(shape, input.base(), 0).ReshapeRuntime();
      ASSERT_EQ(subgraph.GetExternalTensorShape(1), expected.extents());

      // Run subgraph
      Tensor<T> output(expected.extents());
      subgraph.SetupExternalTensor(output.base(), 1)
          .SetupRuntime()
          .InvokeRuntime();

      // Verify results.
      ASSERT_THAT(output, testing::ElementsAreArray(expected));
    }
  }
}

template <typename T>
class ExpandDims : public ::testing::TestWithParam<int> {};

using ExpandDimsQS8 = ExpandDims<quantized<int8_t>>;
using ExpandDimsQU8 = ExpandDims<quantized<uint8_t>>;
using ExpandDimsF16 = ExpandDims<xnn_float16>;
using ExpandDimsF32 = ExpandDims<float>;

TEST_P(ExpandDimsQS8, test) { TestImpl<quantized<int8_t>>(GetParam()); }
TEST_P(ExpandDimsQU8, test) { TestImpl<quantized<uint8_t>>(GetParam()); }
TEST_P(ExpandDimsF16, test) { TestImpl<xnn_float16>(GetParam()); }
TEST_P(ExpandDimsF32, test) { TestImpl<float>(GetParam()); }

auto rank_params = testing::Range(0, XNN_MAX_TENSOR_DIMS - 1);
INSTANTIATE_TEST_SUITE_P(ExpandDims, ExpandDimsQS8, rank_params);
INSTANTIATE_TEST_SUITE_P(ExpandDims, ExpandDimsQU8, rank_params);
INSTANTIATE_TEST_SUITE_P(ExpandDims, ExpandDimsF16, rank_params);
INSTANTIATE_TEST_SUITE_P(ExpandDims, ExpandDimsF32, rank_params);

}  // namespace xnnpack
