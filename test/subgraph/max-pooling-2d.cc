// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cstddef>
#include <cstdint>
#include <random>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "include/xnnpack.h"
#include "src/xnnpack/buffer.h"
#include "src/xnnpack/datatype.h"
#include "src/xnnpack/math.h"
#include "test/replicable_random_device.h"
#include "test/subgraph/stencil.h"
#include "test/subgraph/subgraph-tester.h"

using testing::FloatNear;
using testing::Pointwise;

namespace xnnpack {

template <typename T>
Tensor<T> ReferenceImpl(Tensor<T> input, const StencilParams& kh,
                        const StencilParams& kw) {
  Tensor<T> output({input.extent(0), kh.output_extent(input.extent(1)),
                    kw.output_extent(input.extent(2)), input.extent(3)});

  // Pad the input
  size_t h_padding_max =
      std::max(kh.padding_max, kh.dilated_kernel_extent() - 1 - kh.padding_min);
  size_t w_padding_max =
      std::max(kw.padding_max, kw.dilated_kernel_extent() - 1 - kw.padding_min);
  Tensor<T> padded =
      input.pad(NumericLimits<T>::min(), {0, kh.padding_min, kw.padding_min, 0},
                {0, h_padding_max, w_padding_max, 0});

  padded = make_stencil_dim(padded, 2, kw);
  padded = make_stencil_dim(padded, 1, kh);
  for (size_t n = 0; n < output.extent(0); ++n) {
    for (size_t y = 0; y < output.extent(1); ++y) {
      for (size_t x = 0; x < output.extent(2); ++x) {
        for (size_t c = 0; c < output.extent(3); ++c) {
          T& output_nyxc = output(n, y, x, c);
          output_nyxc = NumericLimits<T>::max_identity();
          for (size_t dy = 0; dy < kh.size; ++dy) {
            for (size_t dx = 0; dx < kw.size; ++dx) {
              output_nyxc = std::max(output_nyxc, padded(n, y, dy, x, dx, c));
            }
          }
        }
      }
    }
  }

  return output;
}

template <typename T>
void TestImpl() {
  ReplicableRandomDevice rng;
  std::bernoulli_distribution bool_dist(0.5);

  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));

  xnn_quantization_params quantization = {0, 1.0f};

  for (auto _ : FuzzTest(std::chrono::milliseconds(1000))) {
    StencilParams kw = random_stencil_params(rng);
    StencilParams kh = random_stencil_params(rng);

    const bool same_padding = bool_dist(rng);

    uint32_t flags = 0;
    if (same_padding) {
      flags |= XNN_FLAG_TENSORFLOW_SAME_PADDING;
      kw.padding_min = kw.padding_max = 0;
      kh.padding_min = kh.padding_max = 0;
    }

    // Define subgraph
    SubgraphTester subgraph(2);
    subgraph.AddInputTensor(4, xnn_datatype_of<T>(), quantization, 0)
        .AddOutputTensor(4, xnn_datatype_of<T>(), quantization, 1)
        .AddMaxPooling2D(kh.padding_min, kw.padding_max, kh.padding_max,
                         kw.padding_min, kh.size, kw.size, kh.stride, kw.stride,
                         kh.dilation, kw.dilation, 0, 1, flags);
    xnn_status status = subgraph.CreateRuntime();
    if (status == xnn_status_unsupported_hardware) {
      GTEST_SKIP();
      return;
    }

    for (int reshape = 0; reshape < 2; ++reshape) {
      std::vector<size_t> output_shape = random_shape(rng, 4);

      std::vector<size_t> input_shape = {
          output_shape[0],
          kh.input_extent(output_shape[1], same_padding),
          kw.input_extent(output_shape[2], same_padding),
          output_shape[3],
      };

      if (same_padding) {
        kh.compute_tf_same_padding(input_shape[1]);
        kw.compute_tf_same_padding(input_shape[2]);
      }

      // TODO(b/404587443): Fix XNNPACK's pooling implementation so this hack is
      // not necessary.
      if (kh.result_is_identity(input_shape[1], output_shape[1]) ||
          kw.result_is_identity(input_shape[2], output_shape[2])) {
        continue;
      }

      Tensor<T> input(input_shape, XnnExtraBytes);
      DatatypeGenerator<T> gen(-100.0f, 100.0f, quantization);
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
      Tensor<T> expected = ReferenceImpl(input, kh, kw);
      // This test should be exact, but it needs a tolerance because kernels
      // that use fp16 arithmetic might flush denormals to 0, but our reference
      // code might not.
      ASSERT_THAT(output,
                  Pointwise(FloatNear(epsilon(xnn_datatype_of<T>())), expected))
          << "output_shape=" << index_to_string(output_shape)
          << ", input_shape=" << index_to_string(input_shape) << ", kh=" << kh
          << ", kw=" << kw;
    }
  }
}

TEST(MaxPooling2DQS8, test) { TestImpl<quantized<int8_t>>(); }
TEST(MaxPooling2DQU8, test) { TestImpl<quantized<uint8_t>>(); }
TEST(MaxPooling2DF16, test) { TestImpl<xnn_float16>(); }
TEST(MaxPooling2DF32, test) { TestImpl<float>(); }

}  // namespace xnnpack
