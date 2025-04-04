// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <cassert>
#include <chrono>
#include <cstddef>
#include <vector>

#include <gtest/gtest.h>
#include "include/xnnpack.h"
#include "src/xnnpack/buffer.h"
#include "src/xnnpack/datatype.h"
#include "src/xnnpack/math.h"
#include "test/replicable_random_device.h"
#include "test/subgraph/stencil.h"
#include "test/subgraph/subgraph-tester.h"

namespace xnnpack {

template <typename T>
Tensor<float> ReferenceImpl(Tensor<T> input, const StencilParams& kh,
                            const StencilParams& kw) {
  Tensor<float> output({input.extent(0), kh.output_extent(input.extent(1)),
                        kw.output_extent(input.extent(2)), input.extent(3)});
  output.fill(0.0f);

  // Make a buffer to compute the sum of the kernels.
  Tensor<float> ones({input.extent(1), input.extent(2)});
  ones.fill(1.0f);

  // Pad the input and the ones
  input = input.pad(0.0f, {0, kh.padding_min, kw.padding_min, 0},
                    {0, kh.padding_max, kw.padding_max, 0});
  ones = ones.pad(0.0f, {kh.padding_min, kw.padding_min},
                  {kh.padding_max, kw.padding_max});

  input = make_stencil_dim(input, 2, kw);
  input = make_stencil_dim(input, 1, kh);
  ones = make_stencil_dim(ones, 1, kw);
  ones = make_stencil_dim(ones, 0, kh);

  for (size_t y = 0; y < output.extent(1); ++y) {
    for (size_t x = 0; x < output.extent(2); ++x) {
      float kernel_sum = 0.0f;
      for (size_t dy = 0; dy < kh.size; ++dy) {
        for (size_t dx = 0; dx < kw.size; ++dx) {
          kernel_sum += ones(y, dy, x, dx);
        }
      }
      for (size_t n = 0; n < output.extent(0); ++n) {
        for (size_t c = 0; c < output.extent(3); ++c) {
          for (size_t dy = 0; dy < kh.size; ++dy) {
            for (size_t dx = 0; dx < kw.size; ++dx) {
              output(n, y, x, c) += input(n, y, dy, x, dx, c);
            }
          }
        }
        for (size_t c = 0; c < output.extent(3); ++c) {
          output(n, y, x, c) /= kernel_sum;
        }
      }
    }
  }

  // Crop off the 1s we padded.
  return output;
}

template <typename T>
void TestImpl() {
  ReplicableRandomDevice rng;

  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));

  for (auto _ : FuzzTest(std::chrono::milliseconds(1000))) {
    // TODO(b/406664150): Test kernels bigger than 5.
    StencilParams kw =
        random_stencil_params(rng, /*max_dilation=*/1, /*max_kernel_size=*/5);
    StencilParams kh =
        random_stencil_params(rng, /*max_dilation=*/1, /*max_kernel_size=*/5);

    // Define subgraph
    SubgraphTester subgraph(2);
    subgraph.AddInputTensor(4, xnn_datatype_of<T>(), 0)
        .AddOutputTensor(4, xnn_datatype_of<T>(), 1)
        .AddAveragePooling2D(kh.padding_min, kw.padding_max, kh.padding_max,
                             kw.padding_min, kh.size, kw.size, kh.stride,
                             kw.stride, 0, 1);
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

      if (input_shape[1] <= kh.padding() || input_shape[2] <= kw.padding()) {
        // TODO(b/406664150): This is a hack around a horrible long-standing bug
        // where xnn_indirection_init_dwconv2d_compressed produces incorrect
        // results if the input is smaller than the padded area. If we hit this
        // case, just try again.
        break;
      }

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
          << ", input_shape=" << index_to_string(input_shape) << ", kh=" << kh
          << ", kw=" << kw;

      // Run subgraph
      Tensor<T> output(output_shape);
      subgraph.SetupExternalTensor(output.base(), 1)
          .SetupRuntime()
          .InvokeRuntime();

      // Verify results.
      Tensor<float> expected = ReferenceImpl(input, kh, kw);
      const float tolerance = 10.0f * xnnpack::epsilon(xnn_datatype_of<T>());
      for (const auto& i : EnumerateIndices(output.extents())) {
        ASSERT_NEAR(output(i), expected(i),
                    tolerance * (1.0f + std::abs(expected(i))))
            << "input_shape=" << index_to_string(input_shape)
            << ", output_shape=" << index_to_string(output_shape)
            << ", i=" << index_to_string(i) << ", kh=" << kh << ", kw=" << kw;
      }
    }
  }
}

TEST(AveragePooling2DF16, test) { TestImpl<xnn_float16>(); }
TEST(AveragePooling2DF32, test) { TestImpl<float>(); }

}  // namespace xnnpack
