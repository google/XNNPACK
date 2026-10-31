// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <numeric>
#include <random>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "ynnpack/base/test/fuzz_test.h"
#include "ynnpack/base/test/random.h"
#include "ynnpack/base/test/tensor.h"
#include "ynnpack/base/type.h"
#include "ynnpack/include/ynnpack.h"
#include "ynnpack/subgraph/test/subgraph_builder.h"

using ynn::to_string;  // NOLINT(misc-unused-using-decls)

namespace ynn {

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
  while (new_shape.size() + 1 < YNN_MAX_TENSOR_RANK && n > 1) {
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
void TestImpl(T) {
  ReplicableRandomDevice rng;
  std::bernoulli_distribution bool_dist(0.5);

  for (auto _ : FuzzTest(std::chrono::milliseconds(1000))) {
    Tensor<T> output(random_shape(rng));
    Tensor<T> input(random_reshape(rng, output.size()));
    fill_random(input.data(), input.size(), rng);

    std::vector<size_t> dims = output.shape();

    // Randomly set one dimension to 0, indicating reshape should deduce the
    // size of that dimension.
    if (!dims.empty() && bool_dist(rng)) {
      std::uniform_int_distribution<> dim_dist(0, dims.size() - 1);
      dims[dim_dist(rng)] = 0;
    }

    // Define subgraph
    SubgraphBuilder subgraph(2);
    subgraph.AddInput(type_of<T>(), input.extents(), 0)
        .AddOutput(type_of<T>(), output.extents(), 1)
        .AddReshape(dims, 0, 1);

    Runtime runtime(subgraph.GetSubgraph());
    ASSERT_EQ(runtime.Status(), ynn_status_success);

    runtime.ReshapeExternalTensor(input.extents(), input.base(), 0)
        .ReshapeRuntime();

    // Check reshaped shape is correct
    ASSERT_EQ(runtime.GetExternalTensorShape(1), output.extents());

    // Run subgraph
    runtime.SetupExternalTensor(output.base(), 1).InvokeRuntime();

    // Verify results.
    ASSERT_THAT(output, testing::ElementsAreArray(input));
  }
}

class Reshape : public ::testing::TestWithParam<ynn_type> {};

TEST_P(Reshape, test) {
  SwitchRealType(GetParam(), [&](auto type) { TestImpl(type); });
}

INSTANTIATE_TEST_SUITE_P(Reshape, Reshape,
                         testing::Values(ynn_type_int8, ynn_type_uint8,
                                         ynn_type_fp16, ynn_type_bf16,
                                         ynn_type_fp32),
                         [](auto p) { return to_string(p.param); });

// Like random_reshape, but with at most `max_rank` dimensions.
template <typename Rng>
std::vector<size_t> random_reshape(Rng& rng, size_t n, size_t max_rank) {
  std::vector<size_t> new_shape = random_reshape(rng, n);
  while (new_shape.size() > max_rank) {
    new_shape[new_shape.size() - 2] *= new_shape.back();
    new_shape.pop_back();
  }
  return new_shape;
}

// Add a random reshape-like op to `subgraph` that reshapes `input_id` (which
// has shape `shape`). Updates `shape` to the shape of the output, and returns
// the id of the output.
template <typename Rng>
uint32_t AddRandomReshape(Rng& rng, SubgraphBuilder& subgraph,
                          std::vector<size_t>& shape, uint32_t input_id) {
  const size_t rank = shape.size();
  const size_t size = std::accumulate(shape.begin(), shape.end(), size_t{1},
                                      std::multiplies<>());
  std::uniform_int_distribution<int> op_dist(0, 3);
  std::uniform_int_distribution<size_t> axis_dist(
      1, std::max<size_t>(rank, 2) - 1);
  int op = op_dist(rng);
  if (rank < 2 && (op == 1 || op == 2)) {
    // These ops need to modify a dimension other than the outermost.
    op = 3;
  }
  uint32_t output_id = YNN_INVALID_VALUE_ID;
  switch (op) {
    case 0: {
      // Reshape everything but the outermost dimension.
      std::vector<size_t> new_shape =
          random_reshape(rng, size / shape[0], YNN_MAX_TENSOR_RANK - 1);
      new_shape.insert(new_shape.begin(), shape[0]);
      subgraph.AddTensor(ynn_type_fp32, TensorShape(new_shape.size()),
                         output_id);
      subgraph.AddReshape(new_shape, input_id, output_id);
      shape = new_shape;
      break;
    }
    case 1: {
      // Split one dimension.
      const size_t axis = axis_dist(rng);
      std::vector<size_t> splits =
          random_reshape(rng, shape[axis], YNN_MAX_TENSOR_RANK + 1 - rank);
      if (splits.empty()) {
        splits.push_back(1);
      }
      subgraph.AddTensor(ynn_type_fp32, TensorShape(rank + splits.size() - 1),
                         output_id);
      subgraph.AddSplitDim(axis, splits, input_id, output_id);
      shape.erase(shape.begin() + axis);
      shape.insert(shape.begin() + axis, splits.begin(), splits.end());
      break;
    }
    case 2: {
      // Fuse some dimensions.
      const size_t axis = axis_dist(rng);
      std::uniform_int_distribution<size_t> count_dist(1, rank - axis);
      const size_t count = count_dist(rng);
      subgraph.AddTensor(ynn_type_fp32, TensorShape(rank - count + 1),
                         output_id);
      subgraph.AddFuseDim(axis, count, input_id, output_id);
      const size_t fused =
          std::accumulate(shape.begin() + axis, shape.begin() + axis + count,
                          size_t{1}, std::multiplies<>());
      shape.erase(shape.begin() + axis, shape.begin() + axis + count);
      shape.insert(shape.begin() + axis, fused);
      break;
    }
    default: {
      // Reshape everything.
      std::vector<size_t> new_shape =
          random_reshape(rng, size, YNN_MAX_TENSOR_RANK);
      subgraph.AddTensor(ynn_type_fp32, TensorShape(new_shape.size()),
                         output_id);
      subgraph.AddReshape(new_shape, input_id, output_id);
      shape = new_shape;
      break;
    }
  }
  return output_id;
}

// Test chains of reshapes (which may be combined by fusion) with producers and
// consumers.
TEST(Reshape, ReshapesBetweenElementwise) {
  ReplicableRandomDevice rng;
  std::bernoulli_distribution bool_dist(0.5);
  std::uniform_int_distribution<size_t> outer_dist(1, 300);
  std::uniform_int_distribution<size_t> inner_dist(1, 2000);
  std::uniform_int_distribution<int> num_reshapes_dist(1, 3);

  for (auto _ : FuzzTest(std::chrono::milliseconds(2000))) {
    std::vector<size_t> shape =
        random_reshape(rng, inner_dist(rng), YNN_MAX_TENSOR_RANK - 1);
    shape.insert(shape.begin(), outer_dist(rng));
    const std::vector<size_t> input_shape = shape;
    const bool static_shape = bool_dist(rng);

    const uint32_t input_id = 0;
    const uint32_t output_id = 1;
    SubgraphBuilder subgraph(2);
    subgraph.AddInput(ynn_type_fp32,
                      static_shape ? TensorShape(input_shape)
                                   : TensorShape(input_shape.size()),
                      input_id);

    // All of the elementwise ops are abs, and the input is positive, so the
    // output should be equal to the input.
    uint32_t x_id = YNN_INVALID_VALUE_ID;
    subgraph.AddTensor(ynn_type_fp32, TensorShape(shape.size()), x_id)
        .AddUnary(ynn_unary_abs, input_id, x_id);

    const int num_reshapes = num_reshapes_dist(rng);
    for (int i = 0; i < num_reshapes; ++i) {
      if (i > 0 && bool_dist(rng)) {
        // Add an op between reshapes.
        uint32_t abs_id = YNN_INVALID_VALUE_ID;
        subgraph.AddTensor(ynn_type_fp32, TensorShape(shape.size()), abs_id)
            .AddUnary(ynn_unary_abs, x_id, abs_id);
        x_id = abs_id;
      }
      x_id = AddRandomReshape(rng, subgraph, shape, x_id);
    }

    subgraph.AddOutput(ynn_type_fp32, TensorShape(shape.size()), output_id)
        .AddUnary(ynn_unary_abs, x_id, output_id);

    Runtime runtime(subgraph.GetSubgraph());
    ASSERT_EQ(runtime.Status(), ynn_status_success);

    Tensor<float> input(input_shape);
    std::uniform_real_distribution<float> value_dist(1.0f, 10.0f);
    for (size_t i = 0; i < input.size(); ++i) {
      input.data()[i] = value_dist(rng);
    }
    runtime.ReshapeExternalTensor(input_shape, input.base(), input_id)
        .ReshapeRuntime();
    ASSERT_EQ(runtime.GetExternalTensorShape(output_id), shape);

    Tensor<float> output(shape);
    runtime.SetupExternalTensor(output.base(), output_id).InvokeRuntime();
    ASSERT_THAT(output, testing::ElementsAreArray(input));
  }
}

// A reshape followed by a slice could cause the internal allocation to be too
// small, because the reshape was writing its full output into an allocation
// sized only for the sliced output.
TEST(Reshape, ReshapeSliceRegression) {
  std::vector<size_t> input_shape = {112};
  std::vector<size_t> reshape_shape = {7, 16};
  std::vector<size_t> output_shape = {100};

  Tensor<float> input(input_shape);
  Tensor<float> other(reshape_shape);
  Tensor<float> output(output_shape);

  std::fill(input.data(), input.data() + input.size(), 1.0f);
  std::fill(other.data(), other.data() + other.size(), 2.0f);

  uint32_t input_id = 0;
  uint32_t other_id = 1;
  uint32_t output_id = 2;

  // Intermediate values.
  uint32_t reshape1_id = YNN_INVALID_VALUE_ID;
  uint32_t add_id = YNN_INVALID_VALUE_ID;
  uint32_t reshape2_id = YNN_INVALID_VALUE_ID;

  SubgraphBuilder subgraph(3);
  subgraph.AddInput(ynn_type_fp32, input_shape, input_id)
      .AddInput(ynn_type_fp32, reshape_shape, other_id)
      .AddOutput(ynn_type_fp32, output_shape, output_id)
      .AddTensor(ynn_type_fp32, 2, reshape1_id)
      .AddTensor(ynn_type_fp32, 2, add_id)
      .AddTensor(ynn_type_fp32, 1, reshape2_id)
      .AddReshape(reshape_shape, input_id, reshape1_id)
      .AddBinary(ynn_binary_add, reshape1_id, other_id, add_id)
      .AddReshape(input_shape, add_id, reshape2_id)
      .AddSlice({0}, {0}, {100}, {1}, reshape2_id, output_id);

  Runtime runtime(subgraph.GetSubgraph());
  ASSERT_EQ(runtime.Status(), ynn_status_success);

  runtime.ReshapeExternalTensor(input_shape, input.base(), input_id)
      .ReshapeExternalTensor(reshape_shape, other.base(), other_id)
      .ReshapeRuntime();

  runtime.SetupExternalTensor(output.base(), output_id).InvokeRuntime();

  for (size_t i = 0; i < output.size(); ++i) {
    ASSERT_EQ(output.data()[i], 3.0f);
  }
}

TEST(Reshape, DotBitcastSliceReduceRegression) {
  std::vector<size_t> p1_shape = {400, 64};
  std::vector<size_t> p2_shape = {64, 2};
  std::vector<size_t> dot_shape = {400, 2};
  std::vector<size_t> bitcast_shape = {100, 4, 2};
  std::vector<size_t> slice_shape = {100, 3, 2};

  Tensor<float> p1(p1_shape);
  Tensor<float> p2(p2_shape);
  Tensor<float> output(slice_shape);

  ReplicableRandomDevice rng;
  fill_random(p1.data(), p1.size(), rng);
  fill_random(p2.data(), p2.size(), rng);

  uint32_t p1_id = 0;
  uint32_t p2_id = 1;
  uint32_t output_id = 2;

  uint32_t dot_id = YNN_INVALID_VALUE_ID;
  uint32_t bitcast_id = YNN_INVALID_VALUE_ID;
  uint32_t slice_id = YNN_INVALID_VALUE_ID;
  uint32_t reduce_id = YNN_INVALID_VALUE_ID;

  SubgraphBuilder subgraph(3);
  subgraph.AddInput(ynn_type_fp32, p1_shape, p1_id)
      .AddInput(ynn_type_fp32, p2_shape, p2_id)
      .AddOutput(ynn_type_fp32, slice_shape, output_id)
      .AddTensor(ynn_type_fp32, dot_shape, dot_id)
      .AddTensor(ynn_type_fp32, bitcast_shape, bitcast_id)
      .AddTensor(ynn_type_fp32, slice_shape, slice_id)
      .AddTensor(ynn_type_fp32, {100, 3, 1}, reduce_id)
      .AddDot(1, p1_id, p2_id, YNN_INVALID_VALUE_ID, dot_id)
      .AddReshape(bitcast_shape, dot_id, bitcast_id)
      .AddSlice({1}, {0}, {3}, {1}, bitcast_id, slice_id)
      .AddReduce(ynn_reduce_max, {2}, slice_id, YNN_INVALID_VALUE_ID, reduce_id,
                 YNN_NODE_FLAG_KEEP_DIMS)
      .AddBinary(ynn_binary_subtract, slice_id, reduce_id, output_id);

  Runtime runtime(subgraph.GetSubgraph());
  ASSERT_EQ(runtime.Status(), ynn_status_success);

  runtime.ReshapeExternalTensor(p1_shape, p1.base(), p1_id)
      .ReshapeExternalTensor(p2_shape, p2.base(), p2_id)
      .ReshapeRuntime();

  runtime.SetupExternalTensor(output.base(), output_id).InvokeRuntime();
}

}  // namespace ynn
