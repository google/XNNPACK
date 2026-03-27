// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <chrono>
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

namespace {

std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)>
CreateSubgraphWithExternalInputAndOutput(uint32_t* input_id, uint32_t* output_id) {
  xnn_subgraph_t subgraph = nullptr;
  EXPECT_EQ(xnn_status_success,
            xnn_create_subgraph(/*external_value_ids=*/2, /*flags=*/0,
                                &subgraph));
  std::unique_ptr<xnn_subgraph, decltype(&xnn_delete_subgraph)> auto_subgraph(
      subgraph, xnn_delete_subgraph);
  if (subgraph == nullptr) {
    return auto_subgraph;
  }

  const size_t dims[1] = {1};
  EXPECT_EQ(xnn_status_success,
            xnn_define_tensor_value(subgraph, xnn_datatype_fp32,
                                    /*num_dims=*/1, dims, /*data=*/nullptr,
                                    /*external_id=*/0,
                                    XNN_VALUE_FLAG_EXTERNAL_INPUT, input_id));
  EXPECT_EQ(xnn_status_success,
            xnn_define_tensor_value(subgraph, xnn_datatype_fp32,
                                    /*num_dims=*/1, dims, /*data=*/nullptr,
                                    /*external_id=*/1,
                                    XNN_VALUE_FLAG_EXTERNAL_OUTPUT, output_id));
  return auto_subgraph;
}

}  // namespace

template <typename T>
void TestImpl(size_t rank) {
  xnnpack::ReplicableRandomDevice rng;

  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));

  for (auto _ : FuzzTest(std::chrono::milliseconds(500))) {
    std::vector<size_t> dims = random_shape(rng, rank);

    std::vector<int64_t> begins(dims.size());
    std::vector<int64_t> ends(dims.size());
    for (size_t i = 0; i < dims.size(); i++) {
      // Test out of bounds slices too.
      const int64_t range = dims[i] * 2;
      auto begin_dist =
          std::uniform_int_distribution<int64_t>(-range, range - 1);
      begins[i] = begin_dist(rng);
      std::uniform_int_distribution<int64_t> end_dist;
      if (begins[i] < 0) {
        // Negative begin, negative end
        end_dist = std::uniform_int_distribution<int64_t>(begins[i], 0);
      } else if (rng() % 2 == 0) {
        // Positive begin, negative end
        end_dist =
            std::uniform_int_distribution<int64_t>(begins[i] - range, 0);
      } else {
        // Positive begin, positive end
        end_dist = std::uniform_int_distribution<int64_t>(begins[i], range);
      }
      ends[i] = end_dist(rng);
    }

    xnn_quantization_params quantization =
        random_quantization(xnn_datatype_of<T>(), rng);

    std::vector<int64_t> strides(dims.size(), 1);
    // Define subgraph
    xnnpack::SubgraphTester subgraph(2);
    subgraph.AddInputTensor(rank, xnn_datatype_of<T>(), quantization, 0)
        .AddOutputTensor(rank, xnn_datatype_of<T>(), quantization, 1)
        .AddSlice(begins, ends, strides, 0, 1)
        .CreateRuntime();

    for (int reshape = 0; reshape < 2; ++reshape) {
      std::vector<size_t> shape = random_shape(rng, rank);
      for (size_t i = 0; i < rank; ++i) {
        shape[i] += dims[i];
      }

      xnnpack::Tensor<T> input(shape, xnnpack::XnnExtraBytes);
      DatatypeGenerator<T> generator(quantization);
      input.generate([&]() { return generator(rng); });

      // Make a deep copy so the expected result is contiguous.
      xnnpack::Tensor<T> expected = input.slice(begins, ends).deep_copy();

      // Check reshape is correct
      subgraph.ReshapeExternalTensor(shape, input.base(), 0).ReshapeRuntime();
      ASSERT_EQ(subgraph.GetExternalTensorShape(1), expected.extents());

      // Run subgraph
      xnnpack::Tensor<T> output(expected.extents());
      subgraph.SetupExternalTensor(output.base(), 1)
          .SetupRuntime()
          .InvokeRuntime();

      // Verify results.
      ASSERT_THAT(output, testing::ElementsAreArray(expected));
    }
  }
}

template <typename T>
class Slice : public ::testing::TestWithParam<int> {};

using SliceQS8 = Slice<quantized<int8_t>>;
using SliceQU8 = Slice<quantized<uint8_t>>;
using SliceF16 = Slice<xnn_float16>;
using SliceF32 = Slice<float>;

TEST_P(SliceQS8, test) { TestImpl<quantized<int8_t>>(GetParam()); }
TEST_P(SliceQU8, test) { TestImpl<quantized<uint8_t>>(GetParam()); }
TEST_P(SliceF16, test) { TestImpl<xnn_float16>(GetParam()); }
TEST_P(SliceF32, test) { TestImpl<float>(GetParam()); }

auto rank_params = testing::Range(1, XNN_MAX_TENSOR_DIMS);
INSTANTIATE_TEST_SUITE_P(Slice, SliceQS8, rank_params);
INSTANTIATE_TEST_SUITE_P(Slice, SliceQU8, rank_params);
INSTANTIATE_TEST_SUITE_P(Slice, SliceF16, rank_params);
INSTANTIATE_TEST_SUITE_P(Slice, SliceF32, rank_params);

TEST(Slice, rejects_rank_above_max_tensor_dims) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));

  uint32_t input_id = XNN_INVALID_VALUE_ID;
  uint32_t output_id = XNN_INVALID_VALUE_ID;
  auto subgraph =
      CreateSubgraphWithExternalInputAndOutput(&input_id, &output_id);
  ASSERT_NE(subgraph, nullptr);

  std::vector<size_t> offsets(XNN_MAX_TENSOR_DIMS + 1, 0);
  std::vector<size_t> sizes(XNN_MAX_TENSOR_DIMS + 1, 1);
  EXPECT_EQ(xnn_status_unsupported_parameter,
            xnn_define_static_slice(subgraph.get(), offsets.size(),
                                    offsets.data(), sizes.data(), input_id,
                                    output_id, /*flags=*/0));
}

TEST(Slice, v2_rejects_rank_above_max_tensor_dims) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));

  uint32_t input_id = XNN_INVALID_VALUE_ID;
  uint32_t output_id = XNN_INVALID_VALUE_ID;
  auto subgraph =
      CreateSubgraphWithExternalInputAndOutput(&input_id, &output_id);
  ASSERT_NE(subgraph, nullptr);

  std::vector<int64_t> offsets(XNN_MAX_TENSOR_DIMS + 1, 0);
  std::vector<size_t> sizes(XNN_MAX_TENSOR_DIMS + 1, 1);
  EXPECT_EQ(xnn_status_unsupported_parameter,
            xnn_define_static_slice_v2(subgraph.get(), offsets.size(),
                                       offsets.data(), sizes.data(), input_id,
                                       output_id, /*flags=*/0));
}

TEST(Slice, v3_rejects_rank_above_max_tensor_dims) {
  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));

  uint32_t input_id = XNN_INVALID_VALUE_ID;
  uint32_t output_id = XNN_INVALID_VALUE_ID;
  auto subgraph =
      CreateSubgraphWithExternalInputAndOutput(&input_id, &output_id);
  ASSERT_NE(subgraph, nullptr);

  std::vector<int64_t> begins(XNN_MAX_TENSOR_DIMS + 1, 0);
  std::vector<int64_t> ends(XNN_MAX_TENSOR_DIMS + 1, 1);
  std::vector<int64_t> strides(XNN_MAX_TENSOR_DIMS + 1, 1);
  EXPECT_EQ(xnn_status_unsupported_parameter,
            xnn_define_static_slice_v3(subgraph.get(), begins.size(),
                                       begins.data(), ends.data(),
                                       strides.data(), input_id, output_id,
                                       /*flags=*/0));
}

}  // namespace xnnpack
