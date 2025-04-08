// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <random>
#include <type_traits>
#include <vector>

#include <gtest/gtest.h>
#include "include/xnnpack.h"
#include "src/xnnpack/buffer.h"
#include "src/xnnpack/datatype.h"
#include "src/xnnpack/math.h"
#include "test/replicable_random_device.h"
#include "test/subgraph/subgraph-tester.h"

namespace xnnpack {

template <typename Rng>
uint32_t random_flags(Rng& rng) {
  std::uniform_int_distribution<> flags_dist(0, 3);
  switch (flags_dist(rng)) {
    case 1:
      return XNN_FLAG_TENSORFLOW_LEGACY_MODE;
    case 2:
      return XNN_FLAG_ALIGN_CORNERS;
    default:
      return 0;
  }
}

template <typename T>
Tensor<T> ReferenceImpl(const Tensor<T>& input, size_t new_height,
                        size_t new_width, uint32_t flags) {
  Tensor<T> output({input.extent(0), new_height, new_width, input.extent(3)});

  const bool align_corners = flags & XNN_FLAG_ALIGN_CORNERS;
  const bool tensorflow_legacy = flags & XNN_FLAG_TENSORFLOW_LEGACY_MODE;

  const int input_width = input.extent(2);
  const int input_height = input.extent(1);
  const int output_width = output.extent(2);
  const int output_height = output.extent(1);

  const int width_adjustment = align_corners && output.extent(2) != 1;
  const int height_adjustment = align_corners && output.extent(1) != 1;
  const float width_scale = static_cast<float>(input_width - width_adjustment) /
                            static_cast<float>(output_width - width_adjustment);
  const float height_scale =
      static_cast<float>(input_height - height_adjustment) /
      static_cast<float>(output_height - height_adjustment);

  const float height_offset =
      tensorflow_legacy || align_corners ? 0.0f : 0.5f * height_scale - 0.5f;
  const float width_offset =
      tensorflow_legacy || align_corners ? 0.0f : 0.5f * width_scale - 0.5f;

  for (size_t y = 0; y < output.extent(1); ++y) {
    const float iy = y * height_scale + height_offset;
    int y0 = std::floor(iy);
    int y1 = y0 + 1;
    float alpha_y = iy - y0;
    y0 = std::min(std::max(y0, 0), input_height - 1);
    y1 = std::min(std::max(y1, 0), input_height - 1);
    for (size_t x = 0; x < output.extent(2); ++x) {
      const float ix = x * width_scale + width_offset;
      int x0 = std::floor(ix);
      int x1 = x0 + 1;
      float alpha_x = ix - x0;
      x0 = std::min(std::max(x0, 0), input_width - 1);
      x1 = std::min(std::max(x1, 0), input_width - 1);
      for (size_t n = 0; n < output.extent(0); ++n) {
        for (size_t c = 0; c < output.extent(3); ++c) {
          float output_nyxc =
              input(n, y0, x0, c) * ((1.0f - alpha_x) * (1.0f - alpha_y)) +
              input(n, y1, x0, c) * ((1.0f - alpha_x) * alpha_y) +
              input(n, y0, x1, c) * (alpha_x * (1.0f - alpha_y)) +
              input(n, y1, x1, c) * (alpha_x * alpha_y);
          if (std::is_integral<typename unwrap_quantized<T>::type>::value) {
            output(n, y, x, c) = round_float_to_int<T>(output_nyxc);
          } else {
            output(n, y, x, c) = output_nyxc;
          }
        }
      }
    }
  }
  return output;
}

// Bilinear resize can be really poorly behaved numerically (hard to test) if
// the input data has sharp details. Limit the input to [-1, 1] to avoid this.
template <typename T>
DatatypeGenerator<T> MakeDatatypeGenerator(T) {
  return DatatypeGenerator<T>(-1.0f, 1.0f);
}

// For quantized types, generate the full range of the type.
template <typename T>
DatatypeGenerator<quantized<T>> MakeDatatypeGenerator(quantized<T>) {
  return DatatypeGenerator<quantized<T>>();
}

template <typename T>
void TestImpl() {
  ReplicableRandomDevice rng;

  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));

  for (auto _ : FuzzTest(std::chrono::milliseconds(1000))) {
    xnn_quantization_params quantization =
        random_quantization(xnn_datatype_of<T>(), rng);

    std::uniform_int_distribution<size_t> width_dist(1, 128);
    std::uniform_int_distribution<size_t> height_dist(1, 9);
    const size_t new_width = width_dist(rng);
    const size_t new_height = height_dist(rng);

    // Define subgraph
    const uint32_t flags = random_flags(rng);
    SubgraphTester subgraph(2);
    subgraph.AddInputTensor(4, xnn_datatype_of<T>(), quantization, 0)
        .AddOutputTensor(4, xnn_datatype_of<T>(), quantization, 1)
        .AddResizeBilinear(new_height, new_width, 0, 1, flags);
    xnn_status status = subgraph.CreateRuntime();
    if (status == xnn_status_unsupported_hardware) {
      GTEST_SKIP();
      return;
    }

    for (int reshape = 0; reshape < 2; ++reshape) {
      std::vector<size_t> input_shape = random_shape(rng, 4);
      Tensor<T> input(input_shape, XnnExtraBytes);
      DatatypeGenerator<T> generator = MakeDatatypeGenerator(T());
      input.generate([&]() { return generator(rng); });

      Tensor<T> expected =
          ReferenceImpl(input, new_height, new_width, flags);

      // Check reshaped shape is correct
      subgraph.ReshapeExternalTensor(input_shape, input.base(), 0)
          .ReshapeRuntime();
      ASSERT_EQ(subgraph.GetExternalTensorShape(1), expected.extents());

      // Run subgraph
      Tensor<T> output(expected.extents());
      subgraph.SetupExternalTensor(output.base(), 1)
          .SetupRuntime()
          .InvokeRuntime();

      // Verify results.
      for (const auto& i : EnumerateIndices(output.extents())) {
        if (is_quantized<T>()) {
          ASSERT_NEAR(expected(i), output(i), 1)
              << "input_shape=" << index_to_string(input.extents())
              << ", new_width=" << new_width << ", new_height=" << new_height
              << ", flags=" << flags;
        } else {
          const float expected_i = expected(i);
          const float tolerance = (std::abs(expected_i) + 1.0f) *
                                  (epsilon(xnn_datatype_of<T>()) * 20.0f);
          ASSERT_NEAR(expected_i, output(i), tolerance)
              << "input_shape=" << index_to_string(input.extents())
              << ", new_width=" << new_width << ", new_height=" << new_height
              << ", flags=" << flags;
        }
      }
    }
  }
}

TEST(ResizeBilinearQS8, test) { TestImpl<quantized<int8_t>>(); }
TEST(ResizeBilinearQU8, test) { TestImpl<quantized<uint8_t>>(); }
TEST(ResizeBilinearF16, test) { TestImpl<xnn_float16>(); }
TEST(ResizeBilinearF32, test) { TestImpl<float>(); }

}  // namespace xnnpack
