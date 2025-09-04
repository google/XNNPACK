// Copyright 2020 Google LLC
//
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree.

#include <array>
#include <cstddef>
#include <cstdint>
#include <iostream>
#include <limits>
#include <vector>

#include "bench/subgraph/benchmark.h"
#include "include/xnnpack.h"
#include <benchmark/benchmark.h>

namespace models {

// Compute the layer norm of [m x n x k] tensors, where the mean and variance
// are computed over the dimensions in `norm_mask`. This computation is
// equivalent to: input_mean = mean(input, norm_mask) (input - input_mean) /
// sqrt(mean(squared_difference(input, input_mean), norm_mask) + epsilon) *
// weight + bias
//
// Where `mean(x, norm_mask)` means computing the mean of the dimensions in the
// `norm_mask`.
xnn_subgraph_t FP32LayerNorm(size_t m, size_t n, size_t k, uint32_t norm_mask) {
  xnn_status status;
  xnn_subgraph_t subgraph = nullptr;
  status = xnn_create_subgraph(/*num_external_values=*/2, 0, &subgraph);
  if (status != xnn_status_success) {
    std::cerr << "failed to create subgraph" << std::endl;
    return nullptr;
  }

  std::array<size_t, 3> dims = {{m, n, k}};
  std::array<size_t, 3> reduction_dims = dims;
  for (size_t i = 0; i < reduction_dims.size(); ++i) {
    if ((norm_mask & (1 << i)) != 0) {
      reduction_dims[i] = 1;
    }
  }

  uint32_t input = XNN_INVALID_VALUE_ID;
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, dims.size(), dims.data(),
      /*data=*/nullptr, 0, XNN_VALUE_FLAG_EXTERNAL_INPUT, &input);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor input" << std::endl;
    return nullptr;
  }

  uint32_t output = XNN_INVALID_VALUE_ID;
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, dims.size(), dims.data(),
      /*data=*/nullptr, 1, /*flags=*/XNN_VALUE_FLAG_EXTERNAL_OUTPUT, &output);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor output" << std::endl;
    return nullptr;
  }

  uint32_t mean = XNN_INVALID_VALUE_ID;
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, reduction_dims.size(), reduction_dims.data(),
      /*data=*/nullptr, XNN_INVALID_VALUE_ID, /*flags=*/0, &mean);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor mean" << std::endl;
    return nullptr;
  }

  std::vector<size_t> reduction_axes;
  reduction_axes.reserve(reduction_dims.size());
  for (size_t i = 0; i < reduction_dims.size(); ++i) {
    if ((norm_mask & (1 << i)) != 0) {
      reduction_axes.push_back(i);
    }
  }
  status =
      xnn_define_static_reduce(subgraph, xnn_reduce_mean, reduction_axes.size(),
                               reduction_axes.data(), input, mean,
                               /*flags=*/XNN_FLAG_KEEP_DIMS);
  if (status != xnn_status_success) {
    std::cerr << "failed to create reduce mean" << std::endl;
    return nullptr;
  }

  uint32_t input_minus_mean = XNN_INVALID_VALUE_ID;
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, dims.size(), dims.data(),
      /*data=*/nullptr, XNN_INVALID_VALUE_ID, /*flags=*/0, &input_minus_mean);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor input_minus_mean" << std::endl;
    return nullptr;
  }

  xnn_binary_params params = {-std::numeric_limits<float>::infinity(),
                              std::numeric_limits<float>::infinity()};
  status = xnn_define_binary(subgraph, xnn_binary_subtract, &params, input,
                             mean, input_minus_mean,
                             /*flags=*/0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create binary subtract" << std::endl;
    return nullptr;
  }

  uint32_t sqr_diff = XNN_INVALID_VALUE_ID;
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, dims.size(), dims.data(),
      /*data=*/nullptr, XNN_INVALID_VALUE_ID, /*flags=*/0, &sqr_diff);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor variance_squared" << std::endl;
    return nullptr;
  }

  status = xnn_define_binary(subgraph, xnn_binary_squared_difference, &params,
                             input, mean, sqr_diff,
                             /*flags=*/0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create binary squared difference" << std::endl;
    return nullptr;
  }

  uint32_t variance = XNN_INVALID_VALUE_ID;
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, reduction_dims.size(), reduction_dims.data(),
      /*data=*/nullptr, XNN_INVALID_VALUE_ID, /*flags=*/0, &variance);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor variance" << std::endl;
    return nullptr;
  }

  status =
      xnn_define_static_reduce(subgraph, xnn_reduce_mean, reduction_axes.size(),
                               reduction_axes.data(), sqr_diff, variance,
                               /*flags=*/XNN_FLAG_KEEP_DIMS);
  if (status != xnn_status_success) {
    std::cerr << "failed to create reduce mean" << std::endl;
    return nullptr;
  }

  uint32_t variance_plus_epsilon = XNN_INVALID_VALUE_ID;
  status = xnn_define_tensor_value(subgraph, xnn_datatype_fp32,
                                   reduction_dims.size(), reduction_dims.data(),
                                   /*data=*/nullptr, XNN_INVALID_VALUE_ID,
                                   /*flags=*/0, &variance_plus_epsilon);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor variance_plus_epsilon" << std::endl;
    return nullptr;
  }

  static float epsilon_value = 1e-3f;
  uint32_t epsilon = XNN_INVALID_VALUE_ID;
  status = xnn_define_tensor_value(subgraph, xnn_datatype_fp32, 0, nullptr,
                                   &epsilon_value, XNN_INVALID_VALUE_ID,
                                   /*flags=*/0, &epsilon);

  status = xnn_define_binary(subgraph, xnn_binary_add, &params, variance,
                             epsilon, variance_plus_epsilon,
                             /*flags=*/0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create binary add" << std::endl;
    return nullptr;
  }

  uint32_t stddev = XNN_INVALID_VALUE_ID;
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, reduction_dims.size(), reduction_dims.data(),
      /*data=*/nullptr, XNN_INVALID_VALUE_ID, /*flags=*/0, &stddev);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor stddev" << std::endl;
    return nullptr;
  }

  status = xnn_define_unary(subgraph, xnn_unary_square_root, nullptr,
                            variance_plus_epsilon, stddev, /*flags=*/0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create unary square root" << std::endl;
    return nullptr;
  }

  uint32_t normalized = XNN_INVALID_VALUE_ID;
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, dims.size(), dims.data(),
      /*data=*/nullptr, XNN_INVALID_VALUE_ID, /*flags=*/0, &normalized);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor normalized" << std::endl;
    return nullptr;
  }

  status = xnn_define_binary(subgraph, xnn_binary_divide, &params,
                             input_minus_mean, stddev, normalized,
                             /*flags=*/0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create binary divide" << std::endl;
    return nullptr;
  }

  static float weight_value = 2.0f;
  uint32_t weight = XNN_INVALID_VALUE_ID;
  status = xnn_define_tensor_value(subgraph, xnn_datatype_fp32, 0, nullptr,
                                   &weight_value, XNN_INVALID_VALUE_ID,
                                   /*flags=*/0, &weight);

  uint32_t normalized_weight = XNN_INVALID_VALUE_ID;
  status = xnn_define_tensor_value(
      subgraph, xnn_datatype_fp32, dims.size(), dims.data(),
      /*data=*/nullptr, XNN_INVALID_VALUE_ID, /*flags=*/0, &normalized_weight);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor normalized_weight" << std::endl;
    return nullptr;
  }

  status = xnn_define_binary(subgraph, xnn_binary_multiply, &params, normalized,
                             weight, normalized_weight,
                             /*flags=*/0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create binary multiply" << std::endl;
    return nullptr;
  }

  static float bias_value = 0.1f;
  uint32_t bias = XNN_INVALID_VALUE_ID;
  status = xnn_define_tensor_value(subgraph, xnn_datatype_fp32, 0, nullptr,
                                   &bias_value, XNN_INVALID_VALUE_ID,
                                   /*flags=*/0, &bias);

  status = xnn_define_binary(subgraph, xnn_binary_add, &params,
                             normalized_weight, bias, output,
                             /*flags=*/0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create binary multiply" << std::endl;
    return nullptr;
  }

  return subgraph;
}

}  // namespace models

static void FP32LayerNorm(benchmark::State& state) {
  xnnpack::RunBenchmark(state, [&state]() {
    return models::FP32LayerNorm(state.range(0), state.range(1), state.range(2),
                         state.range(3));
  });
}

static void LayerNormArguments(benchmark::internal::Benchmark* b) {
  b->ArgNames({"M", "N", "K", "NormMask"});
  for (int norm_mask = 1; norm_mask < 8; norm_mask++) {
    b->Args({128, 256, 512, norm_mask});
  }
}

BENCHMARK(FP32LayerNorm)
    ->Unit(benchmark::kMicrosecond)
    ->MeasureProcessCPUTime()
    ->UseRealTime()
    ->Apply(LayerNormArguments);
