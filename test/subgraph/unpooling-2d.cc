// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <algorithm>
#include <cassert>
#include <cstddef>
#include <cstdint>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "include/xnnpack.h"
#include "src/xnnpack/buffer.h"
#include "src/xnnpack/datatype.h"
#include "test/replicable_random_device.h"
#include "test/subgraph/stencil.h"
#include "test/subgraph/subgraph-tester.h"

using testing::ElementsAreArray;

namespace xnnpack {

template <typename T>
Tensor<T> ReferenceImpl(Tensor<T> value, Tensor<int32_t> index,
                        const StencilParams& kh, const StencilParams& kw) {
  Tensor<T> output({value.extent(0), kh.input_extent(value.extent(1)),
                    kw.input_extent(value.extent(2)), value.extent(3)});

  assert(kw.padding() == 0);
  assert(kh.padding() == 0);

  output.fill(0);
  for (size_t i = 0; i < output.extent(0); i++) {
    for (size_t y = 0; y < value.extent(1); y++) {
      for (size_t x = 0; x < value.extent(2); x++) {
        for (size_t c = 0; c < output.extent(3); c++) {
          const uint32_t dy = index(i, y, x, c) % kh.size;
          const uint32_t dx = index(i, y, x, c) / kh.size;
          output(i, y * kh.size + dy, x * kw.size + dx, c) = value(i, y, x, c);
        }
      }
    }
  }
  return output;
}

template <typename T>
void TestImpl() {
  ReplicableRandomDevice rng;

  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));

  for (int rep = 0; rep < 100; ++rep) {
    StencilParams kw = random_stencil_params(rng, /*max_dilation=*/1);
    StencilParams kh = random_stencil_params(rng, /*max_dilation=*/1);
    // argmax pooling is weird... stride = kernel extent.
    kw.stride = kw.size;
    kh.stride = kh.size;
    // And no padding
    kw.padding_min = 0;
    kw.padding_max = 0;
    kh.padding_min = 0;
    kh.padding_max = 0;

    // Define subgraph
    SubgraphTester subgraph(3);
    subgraph.AddInputTensor(4, xnn_datatype_of<T>(), 0)
        .AddInputTensor(4, xnn_datatype_int32, 1)
        .AddOutputTensor(4, xnn_datatype_of<T>(), 2)
        .AddUnpooling2D(kh.padding_min, kw.padding_max, kh.padding_max,
                        kw.padding_min, kh.size, kw.size, 0, 1, 2);
    xnn_status status = subgraph.CreateRuntime();
    if (status == xnn_status_unsupported_hardware) {
      GTEST_SKIP();
      return;
    }

    for (int reshape = 0; reshape < 2; ++reshape) {
      std::vector<size_t> input_shape = random_shape(rng, 4);
      input_shape[0] = 1;
      input_shape[1] += kh.dilated_kernel_extent();
      input_shape[2] += kw.dilated_kernel_extent();
      input_shape[3] = 1;

      std::vector<size_t> output_shape = {
          input_shape[0],
          kh.input_extent(input_shape[1]),
          kw.input_extent(input_shape[2]),
          input_shape[3],
      };

      Tensor<T> value(input_shape, XnnExtraBytes);
      Tensor<int32_t> index(input_shape, XnnExtraBytes);
      DatatypeGenerator<T> value_gen(-10.0f, 20.0f);
      DatatypeGenerator<int32_t> index_gen(0, kh.size * kw.size - 1);
      value.generate([&]() { return value_gen(rng); });
      index.generate([&]() { return index_gen(rng); });

      subgraph.ReshapeExternalTensor(input_shape, value.base(), 0)
          .ReshapeExternalTensor(input_shape, index.base(), 1)
          .ReshapeRuntime();
      ASSERT_EQ(subgraph.GetExternalTensorShape(2), output_shape)
          << "output_shape=" << index_to_string(output_shape)
          << ", input_shape=" << index_to_string(input_shape) << ", kh=" << kh
          << ", kw=" << kw;

      // Run subgraph
      Tensor<T> output(output_shape);
      subgraph.SetupExternalTensor(output.base(), 2)
          .SetupRuntime()
          .InvokeRuntime();

      // Verify results.
      Tensor<T> expected = ReferenceImpl(value, index, kh, kw);
      ASSERT_THAT(output, ElementsAreArray(expected))
          << "output_shape=" << index_to_string(output_shape)
          << ", input_shape=" << index_to_string(input_shape) << ", kh=" << kh
          << ", kw=" << kw;
    }
  }
}

TEST(Unpooling2DF32, test) { TestImpl<float>(); }

}  // namespace xnnpack
