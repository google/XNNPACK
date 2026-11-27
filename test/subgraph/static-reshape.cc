// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <cassert>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <memory>
#include <random>
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

template <typename Rng>
size_t random_factor(Rng& rng, size_t n) {
  std::uniform_int_distribution<size_t> dist(
      1, static_cast<size_t>(std::ceil(std::sqrt(n))));

  for (size_t i = 0; i < 100; ++i) {
    size_t factor = dist(rng);
    if (n % factor == 0) {
      return factor;
    }
  }
  return n;
}

template <typename Rng>
std::vector<size_t> random_reshape(Rng& rng, size_t n) {
  std::vector<size_t> new_shape;
  while (new_shape.size() + 1 < XNN_MAX_TENSOR_DIMS && n > 1) {
    size_t factor = random_factor(rng, n);
    new_shape.push_back(factor);
    n /= factor;
  }
  if (n != 1) {
    new_shape.push_back(n);
  }
  return new_shape;
}

template <typename T>
void TestImpl() {
  ReplicableRandomDevice rng;
  std::bernoulli_distribution bool_dist(0.5);

  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));

  for (auto _ : FuzzTest(std::chrono::milliseconds(1000))) {
    xnn_quantization_params quantization =
        random_quantization(xnn_datatype_of<T>(), rng);

    Tensor<T> output(random_shape(rng));
    Tensor<T> input(random_reshape(rng, output.size()), xnnpack::XnnExtraBytes);
    DatatypeGenerator<T> generator(quantization);
    input.generate([&]() { return generator(rng); });

    std::vector<size_t> dims = output.shape();

    // Randomly set one dimension to 0, indicating reshape should deduce the
    // size of that dimension.
    if (!dims.empty() && bool_dist(rng)) {
      std::uniform_int_distribution<> dim_dist(0, dims.size() - 1);
      dims[dim_dist(rng)] = 0;
    }

    // Define subgraph
    SubgraphTester subgraph(2);
    subgraph.AddInputTensor(input.extents(), input.data(), quantization, 0)
        .AddOutputTensor(output.extents(), output.data(), quantization, 1)
        .AddReshape(dims, 0, 1);
    ASSERT_EQ(xnn_status_success, subgraph.CreateRuntime());
    subgraph.ReshapeRuntime();

    // Check reshaped shape is correct
    ASSERT_EQ(subgraph.GetExternalTensorShape(1), output.extents());

    // Run subgraph
    subgraph.SetupRuntime().InvokeRuntime();

    // Verify results.
    ASSERT_THAT(output, testing::ElementsAreArray(input));
  }
}

TEST(ReshapeQS8, test) { TestImpl<quantized<int8_t>>(); }
TEST(ReshapeQU8, test) { TestImpl<quantized<uint8_t>>(); }
TEST(ReshapeBF16, test) { TestImpl<xnn_bfloat16>(); }
TEST(ReshapeF16, test) { TestImpl<xnn_float16>(); }
TEST(ReshapeF32, test) { TestImpl<float>(); }

TEST(ReshapeF32, static_input_element_count_mismatch) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));

  xnn_subgraph_t subgraph = nullptr;
  ASSERT_EQ(xnn_status_success, xnn_create_subgraph(2, 0, &subgraph));
  std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)> auto_subgraph(
      subgraph, xnn_delete_subgraph);

  const size_t dims[1] = {64};
  uint32_t input_id = XNN_INVALID_VALUE_ID;
  ASSERT_EQ(xnn_status_success,
            xnn_define_tensor_value(subgraph, xnn_datatype_fp32, 1, dims,
                                    nullptr, /*external_id=*/0,
                                    XNN_VALUE_FLAG_EXTERNAL_INPUT, &input_id));
  const size_t output_dims[1] = {128};
  uint32_t output_id = XNN_INVALID_VALUE_ID;
  ASSERT_EQ(xnn_status_success,
            xnn_define_tensor_value(
                subgraph, xnn_datatype_fp32, 1, output_dims, nullptr,
                /*external_id=*/1, XNN_VALUE_FLAG_EXTERNAL_OUTPUT, &output_id));

  Tensor<float> static_data({4}, XnnExtraBytes);
  static_data.fill(1.0f);
  uint32_t static_id = XNN_INVALID_VALUE_ID;
  ASSERT_EQ(
      xnn_status_success,
      xnn_define_tensor_value(subgraph, xnn_datatype_fp32, static_data.rank(),
                              static_data.extents().data(), static_data.base(),
                              XNN_INVALID_VALUE_ID, /*flags=*/0, &static_id));
  uint32_t reshaped_id = XNN_INVALID_VALUE_ID;
  ASSERT_EQ(xnn_status_success,
            xnn_define_tensor_value(subgraph, xnn_datatype_fp32, 1, dims,
                                    nullptr, XNN_INVALID_VALUE_ID,
                                    /*flags=*/0, &reshaped_id));
  ASSERT_EQ(xnn_status_success,
            xnn_define_static_reshape(subgraph, 1, dims, static_id, reshaped_id,
                                      /*flags=*/0));
  const uint32_t inputs[2] = {input_id, reshaped_id};
  ASSERT_EQ(xnn_status_success,
            xnn_define_concatenate(subgraph, /*axis=*/0, 2, inputs, output_id,
                                   /*flags=*/0));

  xnn_runtime_t runtime = nullptr;
  ASSERT_EQ(xnn_status_success,
            xnn_create_runtime_v4(subgraph, nullptr, nullptr, nullptr,
                                  /*flags=*/0, &runtime));
  std::unique_ptr<xnn_runtime, decltype(&xnn_delete_runtime)> auto_runtime(
      runtime, xnn_delete_runtime);
  EXPECT_EQ(xnn_status_invalid_parameter, xnn_reshape_runtime(runtime));
}

}  // namespace xnnpack
