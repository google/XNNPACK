// Copyright 2025 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <algorithm>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <map>
#include <numeric>
#include <random>
#include <tuple>
#include <utility>
#include <vector>

#include <gmock/gmock.h>
#include <gtest/gtest.h>
#include "include/xnnpack.h"
#include "src/subgraph/subgraph-utils.h"
#include "src/xnnpack/buffer.h"
#include "src/xnnpack/datatype.h"
#include "src/xnnpack/node-type.h"
#include "src/xnnpack/subgraph.h"
#include "test/replicable_random_device.h"
#include "test/subgraph/runtime-flags.h"
#include "test/subgraph/subgraph-tester.h"

namespace xnnpack {

namespace {

enum kTensorIDs {
  input_id = 0,
  output_id,
  extra_external_value_id,
  num_external_values,
};

template <typename T>
static Tensor<T> add_input_tensor(ReplicableRandomDevice& rng,
                                  SubgraphTester& subgraph,
                                  const TensorShape shape, uint32_t id) {
  Tensor<T> tensor(shape.dims, xnnpack::XnnExtraBytes);
  DatatypeGenerator<T> generator;
  tensor.generate([&]() { return generator(rng); });
  subgraph.AddInputTensor(shape, tensor.base(), id);
  return tensor;
}

template <typename T>
static std::tuple<Tensor<T>, uint32_t> add_static_tensor(
    ReplicableRandomDevice& rng, SubgraphTester& subgraph,
    const TensorShape shape, double min = 0.0, double max = 1.0) {
  Tensor<T> static_tensor(shape.dims, xnnpack::XnnExtraBytes);
  if (min < max) {
    DatatypeGenerator<T> generator(min, max);
    static_tensor.generate([&]() { return generator(rng); });
  } else {
    std::fill(static_tensor.begin(), static_tensor.end(), static_cast<T>(min));
  }
  uint32_t static_value_id = XNN_INVALID_VALUE_ID;
  subgraph.AddInternalStaticTensor(shape, xnn_datatype_of<T>(),
                                   &static_value_id, static_tensor.base(),
                                   /*flags=*/0);
  return {static_tensor, static_value_id};
}

template <typename T>
static uint32_t add_internal_dynamic_tensor(SubgraphTester& subgraph,
                                            TensorShape shape) {
  uint32_t value_id = XNN_INVALID_VALUE_ID;
  subgraph.AddInternalDynamicTensor(shape, xnn_datatype_of<T>(), &value_id,
                                    /*flags=*/0);
  return value_id;
}

static std::vector<size_t> create_random_extra_dims(ReplicableRandomDevice& rng,
                                                    size_t rank) {
  std::vector<size_t> indices(rank + 1);
  std::iota(indices.begin(), indices.end(), 0);
  std::shuffle(indices.begin(), indices.end(), rng);
  indices.resize(std::uniform_int_distribution<int>(
      1, std::min<int>(XNN_MAX_TENSOR_DIMS - rank, indices.size()))(rng));
  std::sort(indices.begin(), indices.end());
  return indices;
}

template <class T, class U>
std::pair<T, T> random_swap(ReplicableRandomDevice& rng, T a, U b) {
  return (rng() % 2) ? std::pair<T, T>{a, b} : std::pair<T, T>{b, a};
}
}  // namespace

template <typename F>
void RewriteTestImpl(
    size_t rank, F populate, int expected_size_diff,
    const std::map<enum xnn_node_type, int> expected_node_type_counts = {}) {
  ReplicableRandomDevice rng;
  std::uniform_int_distribution<size_t> dim_dist(1, 9);

  ASSERT_EQ(xnn_status_success, xnn_initialize(nullptr /* allocator */));

  for (auto _ : FuzzTest(std::chrono::milliseconds(100))) {
    std::vector<size_t> input_shape = random_shape(rng, rank);

    // Create the subgraph inputs.
    DatatypeGenerator<float> generator(-1, 1);
    Tensor<float> input(input_shape, xnnpack::XnnExtraBytes);
    input.generate([&]() { return generator(rng); });

    // Define the subgraph.
    SubgraphTester subgraph(num_external_values);
    subgraph.AddInputTensor(input_shape, xnn_datatype_of<float>(), input_id);
    subgraph.AddOutputTensor(rank, xnn_datatype_of<float>(), output_id);
    populate(rng, subgraph);

    // Create a runtime without any rewrites.
    ASSERT_EQ(subgraph.CreateRuntime(
                  /*threadpool=*/nullptr,
                  xnn_test_runtime_flags() | XNN_FLAG_NO_OPERATOR_FUSION),
              xnn_status_success);
    const size_t num_nodes_orig = subgraph.NumNodes();

    // Generate the `dot` files of the before/after subgraphs.
    xnn_subgraph_log_dot_debug(subgraph.Subgraph());

    // Attach the inputs.
    subgraph.ReshapeExternalTensor(input_shape, input.base(), input_id)
        .ReshapeRuntime();
    ASSERT_EQ(subgraph.Status(), xnn_status_success);

    // Run subgraph without rewrites.
    Tensor<float> expected(subgraph.GetExternalTensorShape(output_id));
    subgraph.SetupExternalTensor(expected.base(), output_id)
        .SetupRuntime()
        .InvokeRuntime();

    // Create a runtime with rewrites.
    ASSERT_EQ(subgraph.CreateRuntime(), xnn_status_success);
    const size_t num_nodes_rewritten = subgraph.NumNodes();

    // Generate the `dot` files of the before/after subgraphs.
    xnn_subgraph_log_dot_debug(subgraph.Subgraph());

    // Attach the inputs.
    subgraph.ReshapeExternalTensor(input_shape, input.base(), input_id)
        .ReshapeRuntime();
    ASSERT_EQ(subgraph.Status(), xnn_status_success);

    // Check the node type counts in the rewritten subgraph.
    for (const auto& entry : expected_node_type_counts) {
      const auto& node_type = entry.first;
      const auto& expected_count = entry.second;
      size_t count = 0;
      for (int k = 0; k < subgraph.NumNodes(); k++) {
        count += (subgraph.Node(k)->type == node_type);
      }
      ASSERT_EQ(count, expected_count)
          << "Unexpected number of " << xnn_node_type_to_string(node_type)
          << " nodes (expected " << expected_count << ", got " << count << ").";
    }

    // Run subgraph without rewrites.
    Tensor<float> output(subgraph.GetExternalTensorShape(output_id));
    subgraph.SetupExternalTensor(output.base(), output_id)
        .SetupRuntime()
        .InvokeRuntime();

    // Verify results.
    ASSERT_EQ(num_nodes_orig + expected_size_diff, num_nodes_rewritten);
    ASSERT_THAT(output, testing::ElementsAreArray(expected));
  }
}

class RewriteShapesTest : public ::testing::TestWithParam<int> {};

TEST_P(RewriteShapesTest, DoesNotElideLoadBearingNoOpReshape) {
  // Before: http://graphviz/svg?graph_id=70ed6e3e52006d5be4a45a6d34dbff4d
  // After:  http://graphviz/svg?graph_id=2b154ade83049ce71d6ee3df4869d947

  // Do not completely elide "load-bearing" nodes that are on the critical path
  // for the output, e.g. a single node between an input and an output value.
  RewriteTestImpl(
      GetParam(),
      [&](ReplicableRandomDevice& rng, SubgraphTester& subgraph) {
        const TensorShape input_shape(&subgraph.Value(input_id)->shape);

        // Reshape the input tensor to a different shape.
        std::vector<size_t> output_shape = input_shape.dims;
        std::shuffle(output_shape.begin(), output_shape.end(), rng);
        subgraph.AddReshape(output_shape, input_id, output_id);
      },
      /*expected_size_diff=*/0);
}

TEST_P(RewriteShapesTest, DoesNotElideLoadBearingReshape) {
  // Before: http://graphviz/svg?graph_id=07d2a6e306317442403eea8be205dd00
  // After:  http://graphviz/svg?graph_id=36243dbf9870447fd2973c555bc14846

  // Do not completely elide "load-bearing" nodes that are on the critical path
  // for the output, e.g. a single node between an input and an output value.
  RewriteTestImpl(
      GetParam(),
      [&](ReplicableRandomDevice& rng, SubgraphTester& subgraph) {
        const TensorShape input_shape(&subgraph.Value(input_id)->shape);

        // Reshape the input tensor to the same shape.
        subgraph.AddReshape(input_shape.dims, input_id, output_id);
      },
      /*expected_size_diff=*/0);
}

TEST_P(RewriteShapesTest, DoesNotElideLoadBearingExpandDims) {
  // Do not completely elide "load-bearing" nodes that are on the critical path
  // for the output, e.g. a single node between an input and an output value.

  // Before: http://graphviz/svg?graph_id=6ba9fdce4d427ef86f75eed5ab6b58db
  // After:  http://graphviz/svg?graph_id=6ba9fdce4d427ef86f75eed5ab6b58db
  RewriteTestImpl(
      GetParam(),
      [&](ReplicableRandomDevice& rng, SubgraphTester& subgraph) {
        const TensorShape input_shape(&subgraph.Value(input_id)->shape);

        // Reshape the input tensor to the same shape.
        subgraph.AddExpandDims({0}, input_id, output_id);
      },
      /*expected_size_diff=*/0);
}

TEST_P(RewriteShapesTest, ElidesReshapeOfStaticValue) {
  // Before: http://graphviz/svg?graph_id=
  // After:  http://graphviz/svg?graph_id=

  // Keep static and external tensor data in this scope so that it lives for the
  // duration of the test.
  Tensor<float> static_tensor;

  RewriteTestImpl(
      GetParam(),
      [&](ReplicableRandomDevice& rng, SubgraphTester& subgraph) {
        const TensorShape input_shape(&subgraph.Value(input_id)->shape);

        // Create a static value with a different shape from input1.
        uint32_t static_value_id = XNN_INVALID_VALUE_ID;
        std::vector<size_t> static_value_shape = input_shape.dims;
        std::shuffle(static_value_shape.begin(), static_value_shape.end(), rng);
        std::tie(static_tensor, static_value_id) =
            add_static_tensor<float>(rng, subgraph, static_value_shape);

        // Reshape the static tensor to the same shape as input.
        uint32_t reshaped_value_id =
            add_internal_dynamic_tensor<float>(subgraph, input_shape);
        subgraph.AddReshape(input_shape.dims, static_value_id,
                            reshaped_value_id);

        // Add input1 and the reshaped static value, with the result going to
        // output1.
        auto inputs = random_swap(rng, reshaped_value_id, input_id);
        subgraph.AddAddition(inputs.first, inputs.second, output_id);
      },
      /*expected_size_diff=*/-1);
}

TEST_P(RewriteShapesTest, DoesNotElideNoOpReshapeOfNonStaticShapeValue) {
  // Before: http://graphviz/svg?graph_id=e01a9ed68f996d5fa7a269859dab27c1
  // After:  http://graphviz/svg?graph_id=b115b67bfff6b24e251d3b6393c3ad51

  // Keep static and external tensor data in this scope so that it lives for the
  // duration of the test.
  Tensor<float> extra_input_tensor;

  RewriteTestImpl(
      GetParam(),
      [&](ReplicableRandomDevice& rng, SubgraphTester& subgraph) {
        const TensorShape input_shape(&subgraph.Value(input_id)->shape);

        // Create a non-static value with a different shape from input1.
        extra_input_tensor = add_input_tensor<float>(rng, subgraph, input_shape,
                                                     extra_external_value_id);

        // Reshape the static tensor to the same shape as input.
        uint32_t reshaped_value_id =
            add_internal_dynamic_tensor<float>(subgraph, input_shape);
        subgraph.AddReshape(input_shape.dims, extra_external_value_id,
                            reshaped_value_id);

        // Add input1 and the reshaped static value, with the result going to
        // output1.
        auto inputs = random_swap(rng, reshaped_value_id, input_id);
        subgraph.AddAddition(inputs.first, inputs.second, output_id);
      },
      /*expected_size_diff=*/0);
}

TEST_P(RewriteShapesTest, DoesNotElidesReshapeOfStaticShapeNonStaticValue) {
  // Before: http://graphviz/svg?graph_id=9bbc8511d1acdd044f9721c6446fcf76
  // After:  http://graphviz/svg?graph_id=170e0171589df4f3daed4a6cf58b1013

  // Keep static tensor data in this scope so that it lives for the duration of
  // the test.
  Tensor<float> static_tensor;

  RewriteTestImpl(
      GetParam(),
      [&](ReplicableRandomDevice& rng, SubgraphTester& subgraph) {
        const TensorShape input_shape(&subgraph.Value(input_id)->shape);

        // Create a static value with a different shape from input1.
        uint32_t static_value_id = XNN_INVALID_VALUE_ID;
        std::vector<size_t> static_value_shape = input_shape.dims;
        std::shuffle(static_value_shape.begin(), static_value_shape.end(), rng);
        std::tie(static_tensor, static_value_id) =
            add_static_tensor<float>(rng, subgraph, static_value_shape);

        // Square the static tensor.
        uint32_t squared_value_id =
            add_internal_dynamic_tensor<float>(subgraph, static_value_shape);
        subgraph.AddUnary(xnn_unary_square, /*params=*/nullptr, static_value_id,
                          squared_value_id);

        // Reshape the squared value to the same shape as input.
        uint32_t reshaped_value_id =
            add_internal_dynamic_tensor<float>(subgraph, input_shape);
        subgraph.AddReshape(input_shape.dims, squared_value_id,
                            reshaped_value_id);

        // Add input1 and the reshaped static value, with the result going to
        // output1.
        auto inputs = random_swap(rng, reshaped_value_id, input_id);
        subgraph.AddAddition(inputs.first, inputs.second, output_id);
      },
      /*expected_size_diff=*/0);
}

TEST_P(RewriteShapesTest, CompactsASequenceOfReshapes) {
  // Before: http://graphviz/svg?graph_id=4d6e5f1aca96f1c822640758ce9d44a3
  // After:  http://graphviz/svg?graph_id=cb4e2777edc370177a53a7c479f8cf63

  RewriteTestImpl(
      GetParam(),
      [&](ReplicableRandomDevice& rng, SubgraphTester& subgraph) {
        const TensorShape input_shape(&subgraph.Value(input_id)->shape);

        // Reshape the input value.
        std::vector<size_t> shuffled_shape = input_shape.dims;
        std::shuffle(shuffled_shape.begin(), shuffled_shape.end(), rng);
        uint32_t reshaped_value_id =
            add_internal_dynamic_tensor<float>(subgraph, shuffled_shape);
        subgraph.AddReshape(shuffled_shape, input_id, reshaped_value_id);

        // Reshape the input value again.
        std::shuffle(shuffled_shape.begin(), shuffled_shape.end(), rng);
        uint32_t rereshaped_value_id =
            add_internal_dynamic_tensor<float>(subgraph, shuffled_shape);
        subgraph.AddReshape(shuffled_shape, reshaped_value_id,
                            rereshaped_value_id);

        // Reshape the input tensor back to its original shape.
        subgraph.AddReshape(input_shape.dims, rereshaped_value_id, output_id);
      },
      /*expected_size_diff=*/-2);
}

TEST_P(RewriteShapesTest, CompactsASequenceOfReshapesAndExpandDims) {
  // Before: http://graphviz/svg?graph_id=33663b56011261cfcbc4211e81594424
  // After:  http://graphviz/svg?graph_id=cb4e2777edc370177a53a7c479f8cf63

  RewriteTestImpl(
      GetParam(),
      [&](ReplicableRandomDevice& rng, SubgraphTester& subgraph) {
        const TensorShape input_shape(&subgraph.Value(input_id)->shape);

        // Reshape the input value.
        std::vector<size_t> shuffled_shape = input_shape.dims;
        std::shuffle(shuffled_shape.begin(), shuffled_shape.end(), rng);
        uint32_t reshaped_value_id =
            add_internal_dynamic_tensor<float>(subgraph, shuffled_shape);
        subgraph.AddReshape(shuffled_shape, input_id, reshaped_value_id);

        // Expand the dims..
        std::vector<size_t> extra_dims =
            create_random_extra_dims(rng, input_shape.dims.size());
        uint32_t expanded_value_id = add_internal_dynamic_tensor<float>(
            subgraph, input_shape.dims.size() + extra_dims.size());
        subgraph.AddExpandDims(extra_dims, reshaped_value_id,
                               expanded_value_id);

        // Reshape the input tensor back to its original shape.
        subgraph.AddReshape(input_shape.dims, expanded_value_id, output_id);
      },
      /*expected_size_diff=*/-2);
}

TEST_P(RewriteShapesTest, CompactsASequenceOfExpandDimsAndReshape) {
  // Before: http://graphviz/svg?graph_id=7ba16b5afacc5547bf34a816eeb886de
  // After:  http://graphviz/svg?graph_id=cb4e2777edc370177a53a7c479f8cf63

  RewriteTestImpl(
      GetParam(),
      [&](ReplicableRandomDevice& rng, SubgraphTester& subgraph) {
        const TensorShape input_shape(&subgraph.Value(input_id)->shape);

        // Expand the dims.
        std::vector<size_t> extra_dims =
            create_random_extra_dims(rng, input_shape.dims.size());
        uint32_t expanded_value_id = add_internal_dynamic_tensor<float>(
            subgraph, input_shape.dims.size() + extra_dims.size());
        subgraph.AddExpandDims(extra_dims, input_id, expanded_value_id);

        // Reshape the input value.
        std::vector<size_t> shuffled_shape = input_shape.dims;
        std::shuffle(shuffled_shape.begin(), shuffled_shape.end(), rng);
        uint32_t reshaped_value_id =
            add_internal_dynamic_tensor<float>(subgraph, shuffled_shape);
        subgraph.AddReshape(shuffled_shape, expanded_value_id,
                            reshaped_value_id);

        // Reshape the input tensor back to its original shape.
        subgraph.AddReshape(input_shape.dims, reshaped_value_id, output_id);
      },
      /*expected_size_diff=*/-2);
}

class RewriteClampsTest : public ::testing::TestWithParam<int> {};

TEST_P(RewriteClampsTest, DoesNotElideLoadBearingNoOpClamp) {
  // Do not completely elide "load-bearing" nodes that are on the critical path
  // for the output, e.g. a single node between an input and an output value.

  // Before: http://graphviz/svg?graph_id=d5a47c648d482268bd9634ddbd9b5bb3
  // After:  http://graphviz/svg?graph_id=f59bd53338e154e7f2de38205fd872a3
  RewriteTestImpl(
      GetParam(),
      [&](ReplicableRandomDevice& rng, SubgraphTester& subgraph) {
        // Add a no-op clamp between the input and output.
        subgraph.AddClamp(-INFINITY, INFINITY, input_id, output_id);
      },
      /*expected_size_diff=*/0);
}

TEST_P(RewriteClampsTest, ElidesNoOpClamp) {
  // Before: http://graphviz/svg?graph_id=586fa00cb00e1f40407afd7560ce3062
  // After:  http://graphviz/svg?graph_id=1559de2e06bbf33675fa75a9d2b620fc

  RewriteTestImpl(
      GetParam(),
      [&](ReplicableRandomDevice& rng, SubgraphTester& subgraph) {
        const TensorShape input_shape(&subgraph.Value(input_id)->shape);

        // Add a clamp in the range `[-INFINITY, INFINITY]`.
        uint32_t clamped_value_id =
            add_internal_dynamic_tensor<float>(subgraph, input_shape);
        subgraph.AddClamp(-INFINITY, INFINITY, input_id, clamped_value_id);

        // Output the sum of the input and the clamped values.
        auto inputs = random_swap(rng, clamped_value_id, input_id);
        subgraph.AddAddition(inputs.first, inputs.second, output_id);
      },
      /*expected_size_diff=*/-1);
}

TEST_P(RewriteClampsTest, RewritesMinMaxWithStaticArgsToClamp) {
  // Before: http://graphviz/svg?graph_id=e1f8cc218995d70e56df1d90d766ae1e
  // After:  http://graphviz/svg?graph_id=25f295ed7b3ffd925d84664eda876d4f

  // Keep static and external tensor data in this scope so that it lives for the
  // duration of the test.
  Tensor<float> static_min_tensor;
  Tensor<float> static_max_tensor;

  RewriteTestImpl(
      GetParam(),
      [&](ReplicableRandomDevice& rng, SubgraphTester& subgraph) {
        const TensorShape input_shape(&subgraph.Value(input_id)->shape);

        // Add scalar static tensors for the min/max values.
        uint32_t static_min_value_id;
        uint32_t static_max_value_id;
        std::tie(static_min_tensor, static_min_value_id) =
            add_static_tensor<float>(rng, subgraph, {1});
        std::tie(static_max_tensor, static_max_value_id) =
            add_static_tensor<float>(rng, subgraph, {1});

        // Add the binary `minimum` op.
        uint32_t min_capped_value_id =
            add_internal_dynamic_tensor<float>(subgraph, input_shape);
        auto inputs = random_swap(rng, static_min_value_id, input_id);
        subgraph.AddBinary(xnn_binary_minimum, /*params=*/nullptr, inputs.first,
                           inputs.second, min_capped_value_id);

        // Add the binary `maximum` op.
        inputs = random_swap(rng, min_capped_value_id, static_max_value_id);
        subgraph.AddBinary(xnn_binary_maximum, /*params=*/nullptr, inputs.first,
                           inputs.second, output_id);
      },
      /*expected_size_diff=*/-1);
}

TEST_P(RewriteClampsTest, RewritesMaxMinWithStaticArgsToClamp) {
  // Before: http://graphviz/svg?graph_id=add31d982bca6736fdedd5d4beb9f9a9
  // After:  http://graphviz/svg?graph_id=224372cf4e8d6c80954989ec9424d518

  // Keep static and external tensor data in this scope so that it lives for the
  // duration of the test.
  Tensor<float> static_min_tensor;
  Tensor<float> static_max_tensor;

  RewriteTestImpl(
      GetParam(),
      [&](ReplicableRandomDevice& rng, SubgraphTester& subgraph) {
        const TensorShape input_shape(&subgraph.Value(input_id)->shape);

        // Add scalar static tensors for the min/max values.
        uint32_t static_min_value_id;
        uint32_t static_max_value_id;
        std::tie(static_min_tensor, static_min_value_id) =
            add_static_tensor<float>(rng, subgraph, /*shape=*/{1});
        std::tie(static_max_tensor, static_max_value_id) =
            add_static_tensor<float>(rng, subgraph, /*shape=*/{1});

        // Add the binary `maximum` op.
        uint32_t max_capped_value_id =
            add_internal_dynamic_tensor<float>(subgraph, input_shape);
        auto inputs = random_swap(rng, static_max_value_id, input_id);
        subgraph.AddBinary(xnn_binary_maximum, /*params=*/nullptr, inputs.first,
                           inputs.second, max_capped_value_id);

        // Add the binary `minimum` op.
        inputs = random_swap(rng, max_capped_value_id, static_min_value_id);
        subgraph.AddBinary(xnn_binary_minimum, /*params=*/nullptr, inputs.first,
                           inputs.second, output_id);
      },
      /*expected_size_diff=*/-1);
}

TEST_P(RewriteClampsTest, RewritesSequenceOfClampsToSingleClamps) {
  // Before: http://graphviz/svg?graph_id=b8ef42499710517f35589ba303b49368
  // After:  http://graphviz/svg?graph_id=1839f8e7e8307d49473d22a25f3f02d3

  RewriteTestImpl(
      GetParam(),
      [&](ReplicableRandomDevice& rng, SubgraphTester& subgraph) {
        const TensorShape input_shape(&subgraph.Value(input_id)->shape);
        std::uniform_real_distribution<float> rng_fp32(0.0, 0.5);

        // Add a first clamp in the range `[[0, 0.5), [0.5, 1)]`.
        uint32_t clamped_value_id =
            add_internal_dynamic_tensor<float>(subgraph, input_shape);
        subgraph.AddClamp(rng_fp32(rng), 0.5 + rng_fp32(rng), input_id,
                          clamped_value_id);

        // Add a second clamp in the same range.
        subgraph.AddClamp(rng_fp32(rng), 0.5 + rng_fp32(rng), clamped_value_id,
                          output_id);
      },
      /*expected_size_diff=*/-1);
}

TEST_P(RewriteClampsTest,
       RewritesSequenceOfPotentiallyDisjointClampsToSingleClamps) {
  // Before: http://graphviz/svg?graph_id=bb678dca5038ec09bb06ef3ae50cff62
  // After:  http://graphviz/svg?graph_id=0b196f5c2e4a12fbae00f88b9bc1b02d

  RewriteTestImpl(
      GetParam(),
      [&](ReplicableRandomDevice& rng, SubgraphTester& subgraph) {
        const TensorShape input_shape(&subgraph.Value(input_id)->shape);
        std::uniform_real_distribution<float> rng_fp32(0.0, 2.0);

        // Add a first clamp in the range `[-1, 1]`.
        uint32_t clamped_value_id =
            add_internal_dynamic_tensor<float>(subgraph, input_shape);
        subgraph.AddClamp(rng_fp32(rng) - 1.0, rng_fp32(rng) - 1.0, input_id,
                          clamped_value_id);

        // Add a second clamp in the same range.
        subgraph.AddClamp(rng_fp32(rng) - 1.0, rng_fp32(rng) - 1.0,
                          clamped_value_id, output_id);
      },
      /*expected_size_diff=*/-1);
}

TEST_P(RewriteClampsTest, DoesNotRewriteSharedSequenceOfClamps) {
  // Before: http://graphviz/svg?graph_id=0866534a3ceb0488c24dcfe5c4171c2e
  // After:  http://graphviz/svg?graph_id=0866534a3ceb0488c24dcfe5c4171c2e

  RewriteTestImpl(
      GetParam(),
      [&](ReplicableRandomDevice& rng, SubgraphTester& subgraph) {
        const TensorShape input_shape(&subgraph.Value(input_id)->shape);
        std::uniform_real_distribution<float> rng_fp32(0.0, 0.5);

        // Add a first clamp in the range `[[0, 0.5), [0.5, 1)]`.
        uint32_t first_clamped_value_id =
            add_internal_dynamic_tensor<float>(subgraph, input_shape);
        subgraph.AddClamp(rng_fp32(rng), 0.5 + rng_fp32(rng), input_id,
                          first_clamped_value_id);

        // Add a second clamp in the same range.
        uint32_t second_clamped_value_id =
            add_internal_dynamic_tensor<float>(subgraph, input_shape);
        subgraph.AddClamp(rng_fp32(rng), 0.5 + rng_fp32(rng),
                          first_clamped_value_id, second_clamped_value_id);

        // Output the sum of the first and second clamped values.
        auto inputs =
            random_swap(rng, first_clamped_value_id, second_clamped_value_id);
        subgraph.AddAddition(inputs.first, inputs.second, output_id);
      },
      /*expected_size_diff=*/0);
}

class RewriteArithmeticTest : public ::testing::TestWithParam<int> {};

TEST_P(RewriteArithmeticTest, ElidesNoOpStaticShapeMul) {
  // Before: http://graphviz/svg?graph_id=25f957a09522fb3f746b8f99dbfe3e1b
  // After:  http://graphviz/svg?graph_id=fa2930c3248753347fb03067287181f3

  // Keep static and external tensor data in this scope so that it lives for the
  // duration of the test.
  Tensor<float> static_one_tensor;

  RewriteTestImpl(
      GetParam(),
      [&](ReplicableRandomDevice& rng, SubgraphTester& subgraph) {
        const TensorShape input_shape(&subgraph.Value(input_id)->shape);

        // Reshape the input so that its shape is static.
        uint32_t reshaped_input_value_id =
            add_internal_dynamic_tensor<float>(subgraph, input_shape);
        subgraph.AddReshape(input_shape.dims, input_id,
                            reshaped_input_value_id);

        // Add a scalar static tensor with the value `1.0`.
        uint32_t static_one_value_id;
        std::tie(static_one_tensor, static_one_value_id) =
            add_static_tensor<float>(rng, subgraph, /*shape=*/{1}, 1.0, 1.0);

        // Add the binary `multiply` op with the constant 1.0.
        auto inputs =
            random_swap(rng, static_one_value_id, reshaped_input_value_id);
        subgraph.AddBinary(xnn_binary_multiply, /*params=*/nullptr,
                           inputs.first, inputs.second, output_id);
      },
      /*expected_size_diff=*/-1,
      /*expected_node_type_counts=*/
      {{xnn_node_type_static_reshape, 1},
       {xnn_node_type_binary_elementwise, 0}});
}

TEST_P(RewriteArithmeticTest, DoesNotElidesNoOpDynamicShapeMul) {
  // Before: http://graphviz/svg?graph_id=cfdea1b65dc94d8a7943816872385608
  // After:  http://graphviz/svg?graph_id=cfdea1b65dc94d8a7943816872385608

  // Keep static and external tensor data in this scope so that it lives for the
  // duration of the test.
  Tensor<float> static_one_tensor;

  RewriteTestImpl(
      GetParam(),
      [&](ReplicableRandomDevice& rng, SubgraphTester& subgraph) {
        const TensorShape input_shape(&subgraph.Value(input_id)->shape);

        // Add a scalar static tensor with the value `1.0`.
        uint32_t static_one_value_id;
        std::tie(static_one_tensor, static_one_value_id) =
            add_static_tensor<float>(rng, subgraph, /*shape=*/{1}, 1.0, 1.0);

        // Add the binary `multiply` op with the constant 1.0.
        auto inputs = random_swap(rng, static_one_value_id, input_id);
        subgraph.AddBinary(xnn_binary_multiply, /*params=*/nullptr,
                           inputs.first, inputs.second, output_id);
      },
      /*expected_size_diff=*/0);
}

TEST_P(RewriteArithmeticTest, ElidesNoOpStaticShapeDiv) {
  // Before: http://graphviz/svg?graph_id=0cc474ad547e47a1ff2912c930108b34
  // After:  http://graphviz/svg?graph_id=708ef740be1366e7d4be0dbb0af2f299

  // Keep static and external tensor data in this scope so that it lives for the
  // duration of the test.
  Tensor<float> static_one_tensor;

  RewriteTestImpl(
      GetParam(),
      [&](ReplicableRandomDevice& rng, SubgraphTester& subgraph) {
        const TensorShape input_shape(&subgraph.Value(input_id)->shape);

        // Reshape the input so that its shape is static.
        uint32_t reshaped_input_value_id =
            add_internal_dynamic_tensor<float>(subgraph, input_shape);
        subgraph.AddReshape(input_shape.dims, input_id,
                            reshaped_input_value_id);

        // Add a scalar static tensor with the value `1.0`.
        uint32_t static_one_value_id;
        std::tie(static_one_tensor, static_one_value_id) =
            add_static_tensor<float>(rng, subgraph, /*shape=*/{1}, 1.0, 1.0);

        // Add the binary `divide` op by the constant 1.0.
        subgraph.AddBinary(xnn_binary_divide, /*params=*/nullptr,
                           reshaped_input_value_id, static_one_value_id,
                           output_id);
      },
      /*expected_size_diff=*/-1,
      /*expected_node_type_counts=*/
      {{xnn_node_type_static_reshape, 1},
       {xnn_node_type_binary_elementwise, 0}});
}

TEST_P(RewriteArithmeticTest, DoesNotElidesNoOpDynamicShapeDiv) {
  // Before: http://graphviz/svg?graph_id=7e0c24b47afbb1da4a1bb73ef060f18f
  // After:  http://graphviz/svg?graph_id=7e0c24b47afbb1da4a1bb73ef060f18f

  // Keep static and external tensor data in this scope so that it lives for the
  // duration of the test.
  Tensor<float> static_one_tensor;

  RewriteTestImpl(
      GetParam(),
      [&](ReplicableRandomDevice& rng, SubgraphTester& subgraph) {
        const TensorShape input_shape(&subgraph.Value(input_id)->shape);

        // Add a scalar static tensor with the value `1.0`.
        uint32_t static_one_value_id;
        std::tie(static_one_tensor, static_one_value_id) =
            add_static_tensor<float>(rng, subgraph, /*shape=*/{1}, 1.0, 1.0);

        // Add the binary `divide` op by the constant 1.0.
        subgraph.AddBinary(xnn_binary_divide, /*params=*/nullptr, input_id,
                           static_one_value_id, output_id);
      },
      /*expected_size_diff=*/0);
}

TEST_P(RewriteArithmeticTest, ElidesNoOpStaticShapeAdd) {
  // Before: http://graphviz/svg?graph_id=db2d7f834a222758b462022e17540aff
  // After:  http://graphviz/svg?graph_id=4c4cc998e06e2c455c9cf99d7cf4ca92

  // Keep static and external tensor data in this scope so that it lives for the
  // duration of the test.
  Tensor<float> static_zero_tensor;

  RewriteTestImpl(
      GetParam(),
      [&](ReplicableRandomDevice& rng, SubgraphTester& subgraph) {
        const TensorShape input_shape(&subgraph.Value(input_id)->shape);

        // Reshape the input so that its shape is static.
        uint32_t reshaped_input_value_id =
            add_internal_dynamic_tensor<float>(subgraph, input_shape);
        subgraph.AddReshape(input_shape.dims, input_id,
                            reshaped_input_value_id);

        // Add a scalar static tensor with the value `0.0`.
        uint32_t static_zero_value_id;
        std::tie(static_zero_tensor, static_zero_value_id) =
            add_static_tensor<float>(rng, subgraph, /*shape=*/{1}, 0.0, 0.0);

        // Add the binary `add` op with the constant 0.0.
        auto inputs =
            random_swap(rng, static_zero_value_id, reshaped_input_value_id);
        subgraph.AddBinary(xnn_binary_add, /*params=*/nullptr, inputs.first,
                           inputs.second, output_id);
      },
      /*expected_size_diff=*/-1,
      /*expected_node_type_counts=*/
      {{xnn_node_type_static_reshape, 1},
       {xnn_node_type_binary_elementwise, 0}});
}

TEST_P(RewriteArithmeticTest, DoesNotElidesNoOpDynamicShapeAdd) {
  // Before: http://graphviz/svg?graph_id=a588ed09f437700e7585ce49c8830f0c
  // After:  http://graphviz/svg?graph_id=a588ed09f437700e7585ce49c8830f0c

  // Keep static and external tensor data in this scope so that it lives for the
  // duration of the test.
  Tensor<float> static_zero_tensor;

  RewriteTestImpl(
      GetParam(),
      [&](ReplicableRandomDevice& rng, SubgraphTester& subgraph) {
        const TensorShape input_shape(&subgraph.Value(input_id)->shape);

        // Add a scalar static tensor with the value `0.0`.
        uint32_t static_zero_value_id;
        std::tie(static_zero_tensor, static_zero_value_id) =
            add_static_tensor<float>(rng, subgraph, /*shape=*/{1}, 0.0, 0.0);

        // Add the binary `add` op with the constant 0.0.
        auto inputs = random_swap(rng, static_zero_value_id, input_id);
        subgraph.AddBinary(xnn_binary_add, /*params=*/nullptr, inputs.first,
                           inputs.second, output_id);
      },
      /*expected_size_diff=*/0);
}

TEST_P(RewriteArithmeticTest, ElidesNoOpStaticShapeSub) {
  // Before: http://graphviz/svg?graph_id=0b5f0a7662e3326c71050df429dfc66b
  // After:  http://graphviz/svg?graph_id=114a34e687181d3b30d51c06d844c3f4

  // Keep static and external tensor data in this scope so that it lives for the
  // duration of the test.
  Tensor<float> static_zero_tensor;

  RewriteTestImpl(
      GetParam(),
      [&](ReplicableRandomDevice& rng, SubgraphTester& subgraph) {
        const TensorShape input_shape(&subgraph.Value(input_id)->shape);

        // Reshape the input so that its shape is static.
        uint32_t reshaped_input_value_id =
            add_internal_dynamic_tensor<float>(subgraph, input_shape);
        subgraph.AddReshape(input_shape.dims, input_id,
                            reshaped_input_value_id);

        // Add a scalar static tensor with the value `0.0`.
        uint32_t static_zero_value_id;
        std::tie(static_zero_tensor, static_zero_value_id) =
            add_static_tensor<float>(rng, subgraph, /*shape=*/{1}, 0.0, 0.0);

        // Add the binary `subtract` op with the constant 0.0.
        subgraph.AddBinary(xnn_binary_subtract, /*params=*/nullptr,
                           reshaped_input_value_id, static_zero_value_id,
                           output_id);
      },
      /*expected_size_diff=*/-1,
      /*expected_node_type_counts=*/
      {{xnn_node_type_static_reshape, 1},
       {xnn_node_type_binary_elementwise, 0}});
}

TEST_P(RewriteArithmeticTest, DoesNotElidesNoOpDynamicShapeSub) {
  // Before: http://graphviz/svg?graph_id=3f933623f480d77e676f6a0eaff4d774
  // After:  http://graphviz/svg?graph_id=3f933623f480d77e676f6a0eaff4d774

  // Keep static and external tensor data in this scope so that it lives for the
  // duration of the test.
  Tensor<float> static_zero_tensor;

  RewriteTestImpl(
      GetParam(),
      [&](ReplicableRandomDevice& rng, SubgraphTester& subgraph) {
        const TensorShape input_shape(&subgraph.Value(input_id)->shape);

        // Add a scalar static tensor with the value `0.0`.
        uint32_t static_zero_value_id;
        std::tie(static_zero_tensor, static_zero_value_id) =
            add_static_tensor<float>(rng, subgraph, /*shape=*/{1}, 0.0, 0.0);

        // Add the binary `subtract` op with the constant 0.0.
        subgraph.AddBinary(xnn_binary_subtract, /*params=*/nullptr, input_id,
                           static_zero_value_id, output_id);
      },
      /*expected_size_diff=*/0);
}

TEST_P(RewriteArithmeticTest, RewritesZeroMinusXToNegX) {
  // Before: http://graphviz/svg?graph_id=ef0145d0692eb1737136c6d7a83fb630
  // After:  http://graphviz/svg?graph_id=17720fc6d2f7af918b9ca0fee58d3d23

  // Keep static and external tensor data in this scope so that it lives for the
  // duration of the test.
  Tensor<float> static_zero_tensor;

  RewriteTestImpl(
      GetParam(),
      [&](ReplicableRandomDevice& rng, SubgraphTester& subgraph) {
        const TensorShape input_shape(&subgraph.Value(input_id)->shape);

        // Add a scalar static tensor with the value `0.0`.
        uint32_t static_zero_value_id;
        std::tie(static_zero_tensor, static_zero_value_id) =
            add_static_tensor<float>(rng, subgraph, /*shape=*/{1}, 0.0, 0.0);

        // Add the binary `sub` op with the constant 0.0.
        subgraph.AddBinary(xnn_binary_subtract, /*params=*/nullptr,
                           static_zero_value_id, input_id, output_id);
      },
      /*expected_size_diff=*/0,
      /*expected_node_type_counts=*/
      {{xnn_node_type_binary_elementwise, 0},
       {xnn_node_type_unary_elementwise, 1}});
}

TEST_P(RewriteArithmeticTest, ElidesNoOpChainOfStaticShapeMulZeroAdd) {
  // Before: http://graphviz/svg?graph_id=449b1fd2ba1f74285d40ff3b9200bd8a
  // After:  http://graphviz/svg?graph_id=7b87a703f4182b82c831fe775945507b

  // Keep static and external tensor data in this scope so that it lives for the
  // duration of the test.
  Tensor<float> static_zero_tensor;

  RewriteTestImpl(
      GetParam(),
      [&](ReplicableRandomDevice& rng, SubgraphTester& subgraph) {
        const TensorShape input_shape(&subgraph.Value(input_id)->shape);

        // Reshape the input so that its shape is static.
        uint32_t reshaped_input_value_id =
            add_internal_dynamic_tensor<float>(subgraph, input_shape);
        subgraph.AddReshape(input_shape.dims, input_id,
                            reshaped_input_value_id);

        // Add a scalar static tensor with the value `0.0`.
        uint32_t static_zero_value_id;
        std::tie(static_zero_tensor, static_zero_value_id) =
            add_static_tensor<float>(rng, subgraph, /*shape=*/{1}, 0.0, 0.0);

        // Add the binary `multiply` op with the constant 0.0.
        uint32_t dynamic_zero_value_id =
            add_internal_dynamic_tensor<float>(subgraph, input_shape);
        auto inputs =
            random_swap(rng, static_zero_value_id, reshaped_input_value_id);
        subgraph.AddBinary(xnn_binary_multiply, /*params=*/nullptr,
                           inputs.first, inputs.second, dynamic_zero_value_id);

        // Add the binary `add` op with the input and the dynamic zero value.
        inputs =
            random_swap(rng, dynamic_zero_value_id, reshaped_input_value_id);
        subgraph.AddBinary(xnn_binary_add, /*params=*/nullptr, inputs.first,
                           inputs.second, output_id);
      },
      /*expected_size_diff=*/-2,
      /*expected_node_type_counts=*/
      {{xnn_node_type_static_reshape, 1},
       {xnn_node_type_binary_elementwise, 0}});
}

TEST_P(RewriteArithmeticTest, DoesNotElidesNoOpChainOfDynamicShapeMulZeroAdd) {
  // Before: http://graphviz/svg?graph_id=cb06ee49752d456931d230df2ed58b5c
  // After:  http://graphviz/svg?graph_id=2461d800783badde77c4a7f602f9841e

  // Keep static and external tensor data in this scope so that it lives for the
  // duration of the test.
  Tensor<float> static_zero_tensor;

  RewriteTestImpl(
      GetParam(),
      [&](ReplicableRandomDevice& rng, SubgraphTester& subgraph) {
        const TensorShape input_shape(&subgraph.Value(input_id)->shape);

        // Add a scalar static tensor with the value `0.0`.
        uint32_t static_zero_value_id;
        std::tie(static_zero_tensor, static_zero_value_id) =
            add_static_tensor<float>(rng, subgraph, /*shape=*/{1}, 0.0, 0.0);

        // Add the binary `multiply` op with the constant 0.0.
        uint32_t dynamic_zero_value_id =
            add_internal_dynamic_tensor<float>(subgraph, input_shape);
        auto inputs = random_swap(rng, static_zero_value_id, input_id);
        subgraph.AddBinary(xnn_binary_multiply, /*params=*/nullptr,
                           inputs.first, inputs.second, dynamic_zero_value_id);

        // Add the binary `add` op with the input and the dynamic zero value.
        inputs = random_swap(rng, dynamic_zero_value_id, input_id);
        subgraph.AddBinary(xnn_binary_add, /*params=*/nullptr, inputs.first,
                           inputs.second, output_id);
      },
      /*expected_size_diff=*/0);
}

TEST_P(RewriteArithmeticTest, ElidesNoOpChainOfStaticShapeDivOneMul) {
  // Before: http://graphviz/svg?graph_id=6419999d0359fb72134e11072f53bd18
  // After:  http://graphviz/svg?graph_id=024d9ed9e3edc599416a21fa94d7cd75

  // Keep static and external tensor data in this scope so that it lives for the
  // duration of the test.
  Tensor<float> static_one_tensor;

  RewriteTestImpl(
      GetParam(),
      [&](ReplicableRandomDevice& rng, SubgraphTester& subgraph) {
        const TensorShape input_shape(&subgraph.Value(input_id)->shape);

        // Reshape the input so that its shape is static.
        uint32_t reshaped_input_value_id =
            add_internal_dynamic_tensor<float>(subgraph, input_shape);
        subgraph.AddReshape(input_shape.dims, input_id,
                            reshaped_input_value_id);

        // Add a scalar static tensor with the value `1.0`.
        uint32_t static_one_value_id;
        std::tie(static_one_tensor, static_one_value_id) =
            add_static_tensor<float>(rng, subgraph, /*shape=*/{1}, 1.0, 1.0);

        // Add the static `1.0` to the absolute value of the inputs to make sure
        // they are non-negative
        uint32_t abs_value_id =
            add_internal_dynamic_tensor<float>(subgraph, input_shape);
        uint32_t shifted_value_id =
            add_internal_dynamic_tensor<float>(subgraph, input_shape);
        subgraph.AddUnary(xnn_unary_abs, /*params=*/nullptr,
                          reshaped_input_value_id, abs_value_id);
        auto inputs = random_swap(rng, abs_value_id, static_one_value_id);
        subgraph.AddAddition(inputs.first, inputs.second, shifted_value_id);

        // Add the binary `div(x, x)` op.
        uint32_t dynamic_one_value_id =
            add_internal_dynamic_tensor<float>(subgraph, input_shape);
        subgraph.AddBinary(xnn_binary_divide, /*params=*/nullptr,
                           shifted_value_id, shifted_value_id,
                           dynamic_one_value_id);

        // Add the binary `mul` op with the input and the dynamic one value.
        inputs =
            random_swap(rng, dynamic_one_value_id, reshaped_input_value_id);
        subgraph.AddBinary(xnn_binary_multiply, /*params=*/nullptr,
                           inputs.first, inputs.second, output_id);
      },
      /*expected_size_diff=*/-4,
      /*expected_node_type_counts=*/
      {{xnn_node_type_static_reshape, 1},
       {xnn_node_type_binary_elementwise, 0},
       {xnn_node_type_unary_elementwise, 0}});
}

TEST_P(RewriteArithmeticTest, DoesNotElidesNoOpChainOfDynamicShapeDivOneMul) {
  // Before: http://graphviz/svg?graph_id=b8f48c6a1b4f7728292479805a88649d
  // After:  http://graphviz/svg?graph_id=6a8748d633737a172b104b8e7cbb8f84

  // Keep static and external tensor data in this scope so that it lives for the
  // duration of the test.
  Tensor<float> static_one_tensor;

  RewriteTestImpl(
      GetParam(),
      [&](ReplicableRandomDevice& rng, SubgraphTester& subgraph) {
        const TensorShape input_shape(&subgraph.Value(input_id)->shape);

        // Add a scalar static tensor with the value `1.0`.
        uint32_t static_one_value_id;
        std::tie(static_one_tensor, static_one_value_id) =
            add_static_tensor<float>(rng, subgraph, /*shape=*/{1}, 1.0, 1.0);

        // Add the static `1.0` to the absolute value of the inputs to make sure
        // they are non-negative
        uint32_t abs_value_id =
            add_internal_dynamic_tensor<float>(subgraph, input_shape);
        uint32_t shifted_value_id =
            add_internal_dynamic_tensor<float>(subgraph, input_shape);
        subgraph.AddUnary(xnn_unary_abs, /*params=*/nullptr, input_id,
                          abs_value_id);
        auto inputs = random_swap(rng, abs_value_id, static_one_value_id);
        subgraph.AddAddition(inputs.first, inputs.second, shifted_value_id);

        // Add the binary `div(x, x)` op.
        uint32_t dynamic_one_value_id =
            add_internal_dynamic_tensor<float>(subgraph, input_shape);
        subgraph.AddBinary(xnn_binary_divide, /*params=*/nullptr,
                           shifted_value_id, shifted_value_id,
                           dynamic_one_value_id);

        // Add the binary `mul` op with the input and the dynamic one value.
        inputs = random_swap(rng, dynamic_one_value_id, input_id);
        subgraph.AddBinary(xnn_binary_multiply, /*params=*/nullptr,
                           inputs.first, inputs.second, output_id);
      },
      /*expected_size_diff=*/0);
}

TEST_P(RewriteArithmeticTest, RewritesAddOfNegValue) {
  // Before: http://graphviz/svg?graph_id=6753c33e9c52cc6eb7805fdd0c4ca7a8
  // After:  http://graphviz/svg?graph_id=447b56eb34529358a8c6a13e745c7480

  RewriteTestImpl(
      GetParam(),
      [&](ReplicableRandomDevice& rng, SubgraphTester& subgraph) {
        const TensorShape input_shape(&subgraph.Value(input_id)->shape);

        uint32_t exp_value_id =
            add_internal_dynamic_tensor<float>(subgraph, input_shape);
        subgraph.AddUnary(xnn_unary_exp, /*params=*/nullptr, input_id,
                          exp_value_id);

        uint32_t neg_exp_value_id =
            add_internal_dynamic_tensor<float>(subgraph, input_shape);
        subgraph.AddUnary(xnn_unary_negate, /*params=*/nullptr, exp_value_id,
                          neg_exp_value_id);

        // Add the negated value to the original input.
        auto inputs = random_swap(rng, neg_exp_value_id, input_id);
        subgraph.AddBinary(xnn_binary_add, /*params=*/nullptr, inputs.first,
                           inputs.second, output_id);
      },
      /*expected_size_diff=*/-1,
      /*expected_node_type_counts=*/
      {{xnn_node_type_binary_elementwise, 1},
       {xnn_node_type_unary_elementwise, 1}});
}

TEST_P(RewriteArithmeticTest, RewritesSubOfNegValue) {
  // Before: http://graphviz/svg?graph_id=ab77ab2722fddade4328eaeb3fc4be29
  // After:  http://graphviz/svg?graph_id=17befda00622b6b95b191be2cbc48433

  RewriteTestImpl(
      GetParam(),
      [&](ReplicableRandomDevice& rng, SubgraphTester& subgraph) {
        const TensorShape input_shape(&subgraph.Value(input_id)->shape);

        uint32_t exp_value_id =
            add_internal_dynamic_tensor<float>(subgraph, input_shape);
        subgraph.AddUnary(xnn_unary_exp, /*params=*/nullptr, input_id,
                          exp_value_id);

        uint32_t neg_exp_value_id =
            add_internal_dynamic_tensor<float>(subgraph, input_shape);
        subgraph.AddUnary(xnn_unary_negate, /*params=*/nullptr, exp_value_id,
                          neg_exp_value_id);

        // Subgract the negated value from the original input.
        subgraph.AddBinary(xnn_binary_subtract, /*params=*/nullptr, input_id,
                           neg_exp_value_id, output_id);
      },
      /*expected_size_diff=*/-1,
      /*expected_node_type_counts=*/
      {{xnn_node_type_binary_elementwise, 1},
       {xnn_node_type_unary_elementwise, 1}});
}

TEST_P(RewriteArithmeticTest, RewritesMulOfNegValues) {
  // Before: http://graphviz/svg?graph_id=0b1396a7440ffe4c834c5c7bd1f3deb8
  // After:  http://graphviz/svg?graph_id=59384ecf271274273ec6eca886385971

  RewriteTestImpl(
      GetParam(),
      [&](ReplicableRandomDevice& rng, SubgraphTester& subgraph) {
        const TensorShape input_shape(&subgraph.Value(input_id)->shape);

        uint32_t exp_value_id =
            add_internal_dynamic_tensor<float>(subgraph, input_shape);
        subgraph.AddUnary(xnn_unary_exp, /*params=*/nullptr, input_id,
                          exp_value_id);

        uint32_t neg_exp_value_id =
            add_internal_dynamic_tensor<float>(subgraph, input_shape);
        subgraph.AddUnary(xnn_unary_negate, /*params=*/nullptr, exp_value_id,
                          neg_exp_value_id);

        uint32_t neg_input_value_id =
            add_internal_dynamic_tensor<float>(subgraph, input_shape);
        subgraph.AddUnary(xnn_unary_negate, /*params=*/nullptr, input_id,
                          neg_input_value_id);

        // Multiply the two negated values.
        auto inputs = random_swap(rng, neg_exp_value_id, neg_input_value_id);
        subgraph.AddBinary(xnn_binary_multiply, /*params=*/nullptr,
                           inputs.first, inputs.second, output_id);
      },
      /*expected_size_diff=*/-2,
      /*expected_node_type_counts=*/
      {{xnn_node_type_binary_elementwise, 1},
       {xnn_node_type_unary_elementwise, 1}});
}

TEST_P(RewriteArithmeticTest, RewritesDivOfNegValues) {
  // Before: http://graphviz/svg?graph_id=df395a8dcb42714392bce7ebd5e65665
  // After:  http://graphviz/svg?graph_id=735166d40051a734892eb394d3caeccb

  RewriteTestImpl(
      GetParam(),
      [&](ReplicableRandomDevice& rng, SubgraphTester& subgraph) {
        const TensorShape input_shape(&subgraph.Value(input_id)->shape);

        uint32_t exp_value_id =
            add_internal_dynamic_tensor<float>(subgraph, input_shape);
        subgraph.AddUnary(xnn_unary_exp, /*params=*/nullptr, input_id,
                          exp_value_id);

        uint32_t neg_exp_value_id =
            add_internal_dynamic_tensor<float>(subgraph, input_shape);
        subgraph.AddUnary(xnn_unary_negate, /*params=*/nullptr, exp_value_id,
                          neg_exp_value_id);

        uint32_t neg_input_value_id =
            add_internal_dynamic_tensor<float>(subgraph, input_shape);
        subgraph.AddUnary(xnn_unary_negate, /*params=*/nullptr, input_id,
                          neg_input_value_id);

        // Divide the two negated values.
        subgraph.AddBinary(xnn_binary_divide, /*params=*/nullptr,
                           neg_input_value_id, neg_exp_value_id, output_id);
      },
      /*expected_size_diff=*/-2,
      /*expected_node_type_counts=*/
      {{xnn_node_type_binary_elementwise, 1},
       {xnn_node_type_unary_elementwise, 1}});
}

INSTANTIATE_TEST_SUITE_P(Rewrite, RewriteShapesTest,
                         testing::Range(0, XNN_MAX_TENSOR_DIMS));
INSTANTIATE_TEST_SUITE_P(Rewrite, RewriteClampsTest, testing::Values(0, 1, 3));
INSTANTIATE_TEST_SUITE_P(Rewrite, RewriteArithmeticTest,
                         testing::Values(0, 1, 3));

}  // namespace xnnpack
