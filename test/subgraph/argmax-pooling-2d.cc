// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <cassert>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <vector>

#include <gtest/gtest.h>
#include "include/xnnpack.h"
#include "src/xnnpack/buffer.h"
#include "src/xnnpack/datatype.h"
#include "test/replicable_random_device.h"
#include "test/subgraph/stencil.h"
#include "test/subgraph/subgraph-tester.h"

namespace xnnpack {

template <typename T>
std::tuple<Tensor<T>, Tensor<int32_t>> ReferenceImpl(Tensor<T> input,
                                                     const StencilParams& kh,
                                                     const StencilParams& kw) {
  Tensor<T> value({input.extent(0), kh.output_extent(input.extent(1)),
                   kw.output_extent(input.extent(2)), input.extent(3)});
  Tensor<int32_t> index(value.extents());
  value.fill(NumericLimits<T>::min());
  index.fill(NumericLimits<int32_t>::min());

  // Pad the input
  input = input.pad({0, kh.padding_min, kw.padding_min, 0},
                    {0, kh.padding_max, kw.padding_max, 0});

  input = make_stencil_dim(input, 2, kw);
  input = make_stencil_dim(input, 1, kh);
  for (size_t n = 0; n < value.extent(0); ++n) {
    for (size_t y = 0; y < value.extent(1); ++y) {
      for (size_t x = 0; x < value.extent(2); ++x) {
        for (size_t c = 0; c < value.extent(3); ++c) {
          T& value_nyxc = value(n, y, x, c);
          int32_t& index_nyxc = index(n, y, x, c);
          for (size_t dy = 0; dy < kh.size; ++dy) {
            for (size_t dx = 0; dx < kw.size; ++dx) {
              T input_nyxc = input(n, y, dy, x, dx, c);
              if (input_nyxc > value_nyxc) {
                value_nyxc = input_nyxc;
                index_nyxc = dx * kh.size + dy;
              }
            }
          }
        }
      }
    }
  }

  return {value, index};
}

template <typename T>
void TestImpl() {
  ReplicableRandomDevice rng;

  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));

  for (auto _ : FuzzTest(std::chrono::milliseconds(1000))) {
    StencilParams kw = random_stencil_params(rng);
    StencilParams kh = random_stencil_params(rng);
    kw.dilation = 1;
    kh.dilation = 1;
    // argmax pooling is weird... stride = kernel extent.
    kw.stride = kw.size;
    kh.stride = kh.size;

    // Define subgraph
    SubgraphTester subgraph(3);
    subgraph.AddInputTensor(4, xnn_datatype_of<T>(), 0)
        .AddOutputTensor(4, xnn_datatype_of<T>(), 1)
        .AddOutputTensor(4, xnn_datatype_int32, 2)
        .AddArgMaxPooling2D(kh.padding_min, kw.padding_max, kh.padding_max,
                            kw.padding_min, kh.size, kw.size, 0, 1, 2);
    xnn_status status = subgraph.CreateRuntime();
    if (status == xnn_status_unsupported_hardware) {
      GTEST_SKIP();
      return;
    }

    for (int reshape = 0; reshape < 2; ++reshape) {
      std::vector<size_t> output_shape = random_shape(rng, 4);

      std::vector<size_t> input_shape = {
          output_shape[0],
          kh.input_extent(output_shape[1]),
          kw.input_extent(output_shape[2]),
          output_shape[3],
      };

      // TODO(b/404587443): Fix XNNPACK's pooling implementation so this hack is
      // not necessary.
      if (kh.result_is_identity(input_shape[1], output_shape[1]) ||
          kw.result_is_identity(input_shape[2], output_shape[2])) {
        continue;
      }

      Tensor<T> input(input_shape, PaddingBytes{XNN_EXTRA_BYTES});
      DatatypeGenerator<T> gen(-10.0f, 20.0f);
      input.generate([&]() { return gen(rng); });

      subgraph.ReshapeExternalTensor(input_shape, input.base(), 0)
          .ReshapeRuntime();
      ASSERT_EQ(subgraph.GetExternalTensorShape(1), output_shape)
          << "output_shape=" << index_to_string(output_shape)
          << ", input_shape=" << index_to_string(input_shape) << ", kh=" << kh
          << ", kw=" << kw;
      ASSERT_EQ(subgraph.GetExternalTensorShape(2), output_shape)
          << "output_shape=" << index_to_string(output_shape)
          << ", input_shape=" << index_to_string(input_shape) << ", kh=" << kh
          << ", kw=" << kw;

      // Run subgraph
      Tensor<T> value(output_shape);
      Tensor<int32_t> index(output_shape);
      subgraph.SetupExternalTensor(value.base(), 1)
          .SetupExternalTensor(index.base(), 2)
          .SetupRuntime()
          .InvokeRuntime();

      // Verify results.
      Tensor<T> expected_value;
      Tensor<int32_t> expected_index;
      std::tie(expected_value, expected_index) = ReferenceImpl(input, kh, kw);

      int index_matches = 0;
      for (const auto& i : EnumerateIndices(output_shape)) {
        ASSERT_EQ(value(i), expected_value(i))
            << "output_shape=" << index_to_string(output_shape)
            << ", input_shape=" << index_to_string(input_shape) << ", kh=" << kh
            << ", kw=" << kw;

        index_matches += index(i) == expected_index(i);
      }
      // In the case of ties, it's unreasonable to match XNNPACK's logic for
      // which index is considered the argmax. The indirection buffer logic is
      // very messy and complicated (b/404587443). I think it's better to keep
      // this reference implementation clean and simple, and just check that the
      // index agrees most of the time.
      if (output_shape.size() > 10) {
        ASSERT_GE(index_matches, output_shape.size() / 2)
            << "output_shape=" << index_to_string(output_shape)
            << ", input_shape=" << index_to_string(input_shape) << ", kh=" << kh
            << ", kw=" << kw;
      }
    }
  }
}

TEST(ArgMaxPooling2DF32, test) { TestImpl<float>(); }

}  // namespace xnnpack
