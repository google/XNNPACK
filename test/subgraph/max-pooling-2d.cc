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
#include <type_traits>
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

TEST(MaxPooling2DQU8, IndirectionBufferSizeOverflow32BitSubgraph) {
  if (sizeof(void*) != 4) {
    GTEST_SKIP() << "requires 32-bit pointers";
  }

  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr));

  xnn_subgraph_t subgraph = nullptr;
  ASSERT_EQ(xnn_status_success,
            xnn_create_subgraph(/*external_value_ids=*/2, /*flags=*/0,
                                &subgraph));
  std::unique_ptr<std::remove_pointer<xnn_subgraph_t>::type,
                  decltype(&xnn_delete_subgraph)>
      auto_subgraph(subgraph, xnn_delete_subgraph);

  const size_t dynamic_shape[4] = {0, 0, 0, 0};
  uint32_t input_id = XNN_INVALID_VALUE_ID;
  ASSERT_EQ(xnn_status_success,
            xnn_define_quantized_tensor_value(
                subgraph, xnn_datatype_quint8, /*zero_point=*/0,
                /*scale=*/1.0f, /*num_dims=*/4, dynamic_shape,
                /*data=*/nullptr, /*external_id=*/0,
                XNN_VALUE_FLAG_EXTERNAL_INPUT, &input_id));
  ASSERT_EQ(input_id, 0u);

  uint32_t output_id = XNN_INVALID_VALUE_ID;
  ASSERT_EQ(xnn_status_success,
            xnn_define_quantized_tensor_value(
                subgraph, xnn_datatype_quint8, /*zero_point=*/0,
                /*scale=*/1.0f, /*num_dims=*/4, dynamic_shape,
                /*data=*/nullptr, /*external_id=*/1,
                XNN_VALUE_FLAG_EXTERNAL_OUTPUT, &output_id));
  ASSERT_EQ(output_id, 1u);

  ASSERT_EQ(xnn_status_success,
            xnn_define_max_pooling_2d(
                subgraph,
                /*input_padding_top=*/0, /*input_padding_right=*/0,
                /*input_padding_bottom=*/0, /*input_padding_left=*/0,
                /*pooling_height=*/3, /*pooling_width=*/1,
                /*stride_height=*/1, /*stride_width=*/1,
                /*dilation_height=*/1, /*dilation_width=*/1,
                /*output_min=*/0.0f, /*output_max=*/255.0f, input_id,
                output_id, /*flags=*/0));

  xnn_runtime_t runtime = nullptr;
  const enum xnn_status create_status = xnn_create_runtime_v4(
      subgraph, /*weights_cache=*/nullptr, /*workspace=*/nullptr,
      /*threadpool=*/nullptr, /*flags=*/0, &runtime);
  if (create_status == xnn_status_unsupported_hardware) {
    GTEST_SKIP() << "maxpool runtime unsupported on this target";
  }
  ASSERT_EQ(xnn_status_success, create_status);
  std::unique_ptr<std::remove_pointer<xnn_runtime_t>::type,
                  decltype(&xnn_delete_runtime)>
      auto_runtime(runtime, xnn_delete_runtime);

  const size_t input_shape[4] = {1, 1, 357913941, 1};
  ASSERT_EQ(xnn_status_success,
            xnn_reshape_external_value(runtime, /*external_id=*/0,
                                       /*num_dims=*/4, input_shape));

  const enum xnn_status status = xnn_reshape_runtime(runtime);
  EXPECT_EQ(xnn_status_out_of_memory, status);
}

}  // namespace xnnpack
