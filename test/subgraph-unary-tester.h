// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <array>
#include <cstdint>
#include <cstddef>
#include <limits>
#include <memory>
#include <random>

#include <xnnpack.h>
#include <xnnpack/node-type.h>
#include <xnnpack/operator.h>
#include <xnnpack/requantization.h>
#include <xnnpack/subgraph.h>

#include <gtest/gtest.h>

template <typename T> class UnaryTest : public ::testing::Test {
protected:
  UnaryTest()
  {
    random_device = std::unique_ptr<std::random_device>(new std::random_device());
    rng = std::mt19937((*random_device)());
    shape_dist = std::uniform_int_distribution<size_t>(0, XNN_MAX_TENSOR_DIMS);
    dim_dist = std::uniform_int_distribution<size_t>(1, 9);
    i8dist =
      std::uniform_int_distribution<int32_t>(std::numeric_limits<int8_t>::min(), std::numeric_limits<int8_t>::max());
    u8dist =
      std::uniform_int_distribution<int32_t>(std::numeric_limits<uint8_t>::min(), std::numeric_limits<uint8_t>::max());
    scale_dist = std::uniform_real_distribution<float>(0.1f, 10.0f);
    dims = RandomShape();
    channels = dims.empty() ? 1 : dims.back();
    xnn_shape shape = {
      .num_dims = dims.size(),
    };
    memcpy(shape.dim, dims.data(), dims.size() * sizeof(size_t));
    batch_size = xnn_shape_multiply_non_channel_dims(&shape);
    num_output_elements = batch_size * channels;

    input = std::vector<T>(num_output_elements + XNN_EXTRA_BYTES / sizeof(T));
    operator_output = std::vector<T>(num_output_elements);
    subgraph_output = std::vector<T>(num_output_elements);
  }

  std::vector<size_t> RandomShape() {
    std::vector<size_t> dims(shape_dist(rng));
    std::generate(dims.begin(), dims.end(), [&] { return dim_dist(rng); });
    return dims;
  }

  std::unique_ptr<std::random_device> random_device;
  std::mt19937 rng;
  std::uniform_int_distribution<size_t> shape_dist;
  std::uniform_int_distribution<size_t> dim_dist;
  std::uniform_real_distribution<float> scale_dist;
  std::uniform_int_distribution<int32_t> i8dist;
  std::uniform_int_distribution<int32_t> u8dist;

  std::vector<size_t> dims;

  uint32_t input_id;
  uint32_t output_id;

  size_t channels;
  size_t batch_size;
  size_t num_output_elements;

  std::vector<T> input;
  std::vector<T> operator_output;
  std::vector<T> subgraph_output;
};

