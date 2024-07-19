// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#pragma once

#include <algorithm>
#include <array>
#include <cstddef>
#include <cstdint>
#include <functional>
#include <limits>
#include <numeric>
#include <random>
#include <vector>

#include <gtest/gtest.h>
#include "xnnpack.h"
#include "xnnpack/node-type.h"
#include "xnnpack/operator.h"
#include "xnnpack/requantization.h"
#include "xnnpack/subgraph.h"
#include "replicable_random_device.h"

template <typename T> class BinaryTest : public ::testing::Test {
 protected:
  BinaryTest() {
    shape_dist = std::uniform_int_distribution<size_t>(0, XNN_MAX_TENSOR_DIMS);
    dim_dist = std::uniform_int_distribution<size_t>(1, 9);
    f32dist = std::uniform_real_distribution<float>(0.01f, 1.0f);
    i8dist =
      std::uniform_int_distribution<int32_t>(std::numeric_limits<int8_t>::min(), std::numeric_limits<int8_t>::max());
    u8dist =
      std::uniform_int_distribution<int32_t>(std::numeric_limits<uint8_t>::min(), std::numeric_limits<uint8_t>::max());
    scale_dist = std::uniform_real_distribution<float>(0.1f, 5.0f);
    s32dist = std::uniform_int_distribution<int32_t>(std::numeric_limits<int32_t>::min(), std::numeric_limits<int32_t>::max());
  }

  void SetUp() override
  {
    std::vector<size_t> input1_shape = RandomShape();
    std::vector<size_t> input2_shape;
    std::vector<size_t> output_shape;
    // Create input dimensions.
    // Create input 2 with an equal or larger number of dimensions.
    const size_t input2_num_dims = std::uniform_int_distribution<size_t>(input1_shape.size(), XNN_MAX_TENSOR_DIMS)(rng);
    input2_shape = RandomShape(input2_num_dims);
    // Ensure that the inputs dimensions match.
    std::copy_backward(input1_shape.begin(), input1_shape.end(), input2_shape.end());

    // Choose a random dimension to broadcast for each input.
    const size_t input1_broadcast_dim = std::uniform_int_distribution<size_t>(0, input1_shape.size())(rng);
    if (input1_broadcast_dim < input1_shape.size()) {
      input1_shape[input1_broadcast_dim] = 1;
    }
    const size_t input2_broadcast_dim = std::uniform_int_distribution<size_t>(0, input2_shape.size())(rng);
    if (input2_broadcast_dim < input2_shape.size()) {
      input2_shape[input2_broadcast_dim] = 1;
    }
    input1_dims.resize(XNN_MAX_TENSOR_DIMS);
    input2_dims.resize(XNN_MAX_TENSOR_DIMS);
    output_dims.resize(XNN_MAX_TENSOR_DIMS);

    // Calculate generalized shapes.
    std::fill(input1_dims.begin(), input1_dims.end(), 1);
    std::fill(input2_dims.begin(), input2_dims.end(), 1);
    std::copy_backward(input1_shape.cbegin(), input1_shape.cend(), input1_dims.end());
    std::copy_backward(input2_shape.cbegin(), input2_shape.cend(), input2_dims.end());
    for (size_t i = 0; i < XNN_MAX_TENSOR_DIMS; i++) {
      if (input1_dims[i] != 1 && input2_dims[i] != 1) {
        ASSERT_EQ(input1_dims[i], input2_dims[i]) << "i: " << i;
      }
      output_dims[i] = std::max(input1_dims[i], input2_dims[i]);
    }

    if (f32dist(rng) < 0.5f) {
      RemoveLeadingOnes(input1_dims);
    }
    if (f32dist(rng) < 0.5f) {
      RemoveLeadingOnes(input2_dims);
    }
    while (output_dims.size() > std::max(input1_dims.size(), input2_dims.size())) {
      output_dims.erase(output_dims.begin());
    }

    input1 = std::vector<T>(XNN_EXTRA_BYTES / sizeof(T) + NumElements(input1_shape));
    input2 = std::vector<T>(XNN_EXTRA_BYTES / sizeof(T) + NumElements(input2_shape));
    operator_output = std::vector<T>(NumElements(output_dims));
    subgraph_output = std::vector<T>(operator_output.size());
  }

  std::vector<size_t> RandomShape(size_t num_dims)
  {
    std::vector<size_t> dims(num_dims);
    std::generate(dims.begin(), dims.end(), [&] { return dim_dist(rng); });
    return dims;
  }

  std::vector<size_t> RandomShape() { return RandomShape(shape_dist(rng)); }

  size_t NumElements(std::vector<size_t>& dims)
  {
    return std::accumulate(dims.begin(), dims.end(), size_t(1), std::multiplies<size_t>());
  }

  size_t NumElements(std::array<size_t, XNN_MAX_TENSOR_DIMS>& dims)
  {
    return std::accumulate(dims.begin(), dims.end(), size_t(1), std::multiplies<size_t>());
  }

  void RemoveLeadingOnes(std::vector<size_t>& dims) {
    while (!dims.empty()) {
      if (dims.front() == 1) {
        dims.erase(dims.begin());
      } else {
        break;
      }
    }
  }

  xnnpack::ReplicableRandomDevice rng;
  std::uniform_int_distribution<size_t> shape_dist;
  std::uniform_int_distribution<size_t> dim_dist;
  std::uniform_real_distribution<float> f32dist;
  std::uniform_real_distribution<float> scale_dist;
  std::uniform_int_distribution<int32_t> i8dist;
  std::uniform_int_distribution<int32_t> u8dist;
  std::uniform_int_distribution<int32_t> s32dist;


  T output_min = std::numeric_limits<T>::min();
  T output_max = std::numeric_limits<T>::max();

  std::vector<size_t> input1_dims;
  std::vector<size_t> input2_dims;
  std::vector<size_t> output_dims;

  std::vector<T> input1;
  std::vector<T> input2;
  std::vector<T> operator_output;
  std::vector<T> subgraph_output;
};
