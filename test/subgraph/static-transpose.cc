// Copyright 2022 Google LLC
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

template <typename T>
void TestImpl(size_t rank) {
  ReplicableRandomDevice rng;

  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));

  std::vector<size_t> perm(rank);
  std::iota(perm.begin(), perm.end(), 0);

  do {
    xnn_quantization_params quantization =
        random_quantization(xnn_datatype_of<T>(), rng);

    // Define subgraph
    SubgraphTester subgraph(2);
    subgraph.AddInputTensor(rank, xnn_datatype_of<T>(), quantization, 0)
        .AddOutputTensor(rank, xnn_datatype_of<T>(), quantization, 1)
        .AddTranspose(perm, 0, 1)
        .CreateRuntime();

    for (int reshape = 0; reshape < 2; ++reshape) {
      std::vector<size_t> shape = random_shape(rng, rank);

      Tensor<T> input(shape, PaddingBytes{XNN_EXTRA_BYTES});
      DatatypeGenerator<T> generator(quantization);
      input.generate([&]() { return generator(rng); });

      // Make a deep copy of the expected result so it is contiguous.
      Tensor<T> expected = input.transpose(perm).deep_copy();

      // Check reshaped shape is correct
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
  } while (std::next_permutation(perm.begin(), perm.end()));
}

template <typename T>
class Transpose : public ::testing::TestWithParam<int> {};

using TransposeQS8 = Transpose<quantized<int8_t>>;
using TransposeQU8 = Transpose<quantized<uint8_t>>;
using TransposeF16 = Transpose<xnn_float16>;
using TransposeF32 = Transpose<float>;

TEST_P(TransposeQS8, test) { TestImpl<quantized<int8_t>>(GetParam()); }
TEST_P(TransposeQU8, test) { TestImpl<quantized<uint8_t>>(GetParam()); }
TEST_P(TransposeF16, test) { TestImpl<xnn_float16>(GetParam()); }
TEST_P(TransposeF32, test) { TestImpl<float>(GetParam()); }

auto rank_params = testing::Range(1, XNN_MAX_TENSOR_DIMS);
INSTANTIATE_TEST_SUITE_P(Transpose, TransposeQS8, rank_params);
INSTANTIATE_TEST_SUITE_P(Transpose, TransposeQU8, rank_params);
INSTANTIATE_TEST_SUITE_P(Transpose, TransposeF16, rank_params);
INSTANTIATE_TEST_SUITE_P(Transpose, TransposeF32, rank_params);

}  // namespace xnnpack
