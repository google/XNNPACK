// Copyright 2022 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <algorithm>
#include <array>
#include <cstdint>
#include <cstddef>
#include <limits>
#include <memory>
#include <numeric>
#include <random>

#include <xnnpack.h>
#include <xnnpack/node-type.h>
#include <xnnpack/operator.h>
#include <xnnpack/requantization.h>
#include <xnnpack/subgraph.h>

#include <gtest/gtest.h>

template <typename InputType, typename OutputType = InputType, size_t min_dim = 0> class UnaryTest : public ::testing::Test {
protected:
  UnaryTest()
  {
    random_device = std::unique_ptr<std::random_device>(new std::random_device());
    rng = std::mt19937((*random_device)());
    shape_dist = std::uniform_int_distribution<size_t>(min_dim, XNN_MAX_TENSOR_DIMS);
    dim_dist = std::uniform_int_distribution<size_t>(1, 9);
    i8dist =
      std::uniform_int_distribution<int32_t>(std::numeric_limits<int8_t>::min(), std::numeric_limits<int8_t>::max());
    u8dist =
      std::uniform_int_distribution<int32_t>(std::numeric_limits<uint8_t>::min(), std::numeric_limits<uint8_t>::max());
    u32dist = std::uniform_int_distribution<uint32_t>();
    scale_dist = std::uniform_real_distribution<float>(0.1f, 10.0f);
    f32dist = std::uniform_real_distribution<float>(0.01f, 1.0f);
    dims = RandomShape();
    channels = dims.empty() ? 1 : dims.back();
    xnn_shape shape = {
      .num_dims = dims.size(),
    };
    memcpy(shape.dim, dims.data(), dims.size() * sizeof(size_t));
    batch_size = xnn_shape_multiply_non_channel_dims(&shape);
    num_output_elements = batch_size * channels;
    scale = scale_dist(rng);
    signed_zero_point = i8dist(rng);
    unsigned_zero_point = u8dist(rng);

    input = std::vector<InputType>(num_output_elements + XNN_EXTRA_BYTES / sizeof(InputType));
    operator_output = std::vector<OutputType>(num_output_elements);
    subgraph_output = std::vector<OutputType>(num_output_elements);
  }

  std::vector<size_t> RandomShape() {
    std::vector<size_t> dims(shape_dist(rng));
    std::generate(dims.begin(), dims.end(), [&] { return dim_dist(rng); });
    return dims;
  }

  static size_t NumElements(const std::vector<size_t>& dims)
  {
    return std::accumulate(dims.begin(), dims.end(), size_t(1), std::multiplies<size_t>());
  }

  std::unique_ptr<std::random_device> random_device;
  std::mt19937 rng;
  std::uniform_int_distribution<size_t> shape_dist;
  std::uniform_int_distribution<size_t> dim_dist;
  std::uniform_real_distribution<float> scale_dist;
  std::uniform_int_distribution<int32_t> i8dist;
  std::uniform_int_distribution<int32_t> u8dist;
  std::uniform_int_distribution<uint32_t> u32dist;
  std::uniform_real_distribution<float> f32dist;

  std::vector<size_t> dims;

  uint32_t input_id;
  uint32_t output_id;

  size_t channels;
  size_t batch_size;
  size_t num_output_elements;
  float scale;
  int32_t signed_zero_point;
  int32_t unsigned_zero_point;

  std::vector<InputType> input;
  std::vector<OutputType> operator_output;
  std::vector<OutputType> subgraph_output;
};
