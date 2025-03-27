// Copyright 2025 Google LLC
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
void FuseAndSplit() {
  ReplicableRandomDevice rng;

  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));

  for (auto _ : FuzzTest(std::chrono::milliseconds(1000))) {
    xnn_quantization_params quantization =
        random_quantization(xnn_datatype_of<T>(), rng);

    const auto input_shape = random_shape(rng);
    Tensor<T> input(input_shape, PaddingBytes{XNN_EXTRA_BYTES});
    DatatypeGenerator<T> generator(quantization);
    input.generate([&]() { return generator(rng); });

    for (size_t left = 0; left < input_shape.size(); left++) {
      for (size_t right = left + 1; right < input_shape.size(); right++) {
        std::vector<size_t> splits(&input_shape[left], &input_shape[right] + 1);
        size_t split_dims_size = 1;
        for (size_t dim : splits) {
          split_dims_size *= dim;
        }

        //
        // Fuse dimensions `[left, right]` of the input.
        //

        // Define the fusion subgraph.
        Tensor<T> fused({input.size()});
        SubgraphTester fuse_subgraph(2);
        fuse_subgraph
            .AddInputTensor(input.shape(), input.data(), quantization,
                            /*external_id=*/0)
            .AddOutputTensor(fused.shape(), fused.data(), quantization,
                             /*external_id=*/1)
            .AddFuseDims(left, splits.size(), /*input_id=*/0, /*output_id=*/1);
        ASSERT_EQ(xnn_status_success, fuse_subgraph.CreateRuntime());

        // Reshape the thing.
        fuse_subgraph.ReshapeRuntime();
        auto fused_shape = fuse_subgraph.GetExternalTensorShape(1);

        // Check that the fused shape is correct.
        size_t fused_size = 1;
        for (size_t dim : fused_shape) {
          fused_size *= dim;
        }
        ASSERT_EQ(input.size(), fused_size);
        for (size_t k = 0; k < left; k++) {
          ASSERT_EQ(input_shape[k], fused_shape[k]);
        }
        ASSERT_EQ(split_dims_size, fused_shape[left]);
        for (size_t k = right + 1; k < input_shape.size(); k++) {
          ASSERT_EQ(input_shape[k], fused_shape[k - right + left]);
        }

        // Randomize the input and run the fusion subgraph.
        input.generate([&]() { return generator(rng); });
        fuse_subgraph.SetupRuntime().InvokeRuntime();

        // Verify results.
        ASSERT_THAT(fused, testing::ElementsAreArray(input));

        //
        // Split the `left` axis back into the original dimensions.
        //

        // Maybe set one of the splits to zero.
        const size_t idx = rng() % input_shape.size();
        if (idx < splits.size()) {
          splits[idx] = 0;
        }

        // Define the split subgraph.
        Tensor<T> split({input.size()});
        SubgraphTester split_subgraph(2);
        split_subgraph
            .AddInputTensor(fused_shape, fused.data(), quantization,
                            /*external_id=*/0)
            .AddOutputTensor(split.shape(), split.data(), quantization,
                             /*external_id=*/1)
            .AddSplitDim(left, splits, /*input_id=*/0, /*output_id=*/1);
        ASSERT_EQ(xnn_status_success, split_subgraph.CreateRuntime());

        // Reshape the thing.
        split_subgraph.ReshapeRuntime();
        auto split_shape = split_subgraph.GetExternalTensorShape(1);

        // Check that the split shape is correct.
        ASSERT_EQ(input_shape, split_shape);

        // Run the split subgraph.
        split_subgraph.SetupRuntime().InvokeRuntime();

        // Verify results.
        ASSERT_THAT(split, testing::ElementsAreArray(input));

        //
        // Fuse and Split the input back into the original dimensions in
        // consecutive ops..
        //

        // Define the fuse_and_split subgraph.
        Tensor<T> output({input.size()});
        SubgraphTester fuse_and_split_subgraph(3);
        fuse_and_split_subgraph
            .AddInputTensor(input_shape, input.data(), quantization,
                            /*external_id=*/0)
            .AddOutputTensor(fused.shape(), fused.data(), quantization,
                             /*external_id=*/1)
            .AddOutputTensor(output.shape(), output.data(), quantization,
                             /*external_id=*/2)
            .AddFuseDims(left, splits.size(), /*input_id=*/0, /*output_id=*/1)
            .AddSplitDim(left, splits, /*input_id=*/1, /*output_id=*/2);
        ASSERT_EQ(xnn_status_success, fuse_and_split_subgraph.CreateRuntime());

        // Reshape the thing.
        fuse_and_split_subgraph.ReshapeRuntime();
        auto output_shape = fuse_and_split_subgraph.GetExternalTensorShape(2);

        // Check that the output shape is correct.
        ASSERT_EQ(input_shape, output_shape);

        // Randomize the input and run the fuse and split subgraph.
        input.generate([&]() { return generator(rng); });
        fuse_and_split_subgraph.SetupRuntime().InvokeRuntime();

        // Verify results.
        ASSERT_THAT(output, testing::ElementsAreArray(input));
      }
    }
  }
}

TEST(FuseAndSplitQS8, test) { FuseAndSplit<quantized<int8_t>>(); }
TEST(FuseAndSplitQU8, test) { FuseAndSplit<quantized<uint8_t>>(); }
TEST(FuseAndSplitBF16, test) { FuseAndSplit<xnn_bfloat16>(); }
TEST(FuseAndSplitF16, test) { FuseAndSplit<xnn_float16>(); }
TEST(FuseAndSplitF32, test) { FuseAndSplit<float>(); }

}  // namespace xnnpack
