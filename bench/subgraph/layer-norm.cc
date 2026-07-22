// Copyright 2026 Google LLC
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
#include "src/xnnpack/datatype.h"
#include "src/xnnpack/math.h"
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
template <typename T>
xnn_subgraph_t LayerNormImpl(size_t m, size_t n, size_t k, uint32_t norm_mask) {
  xnn_status status;
  const xnn_datatype datatype = xnn_datatype_of<T>();
  const xnn_datatype math_datatype = xnn_datatype_fp32;
  auto subgraph = xnnpack::CreateUniqueSubgraph(/*num_external_values=*/2, 0);
  if (!subgraph) {
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
      subgraph.get(), datatype, dims.size(), dims.data(),
      /*data=*/nullptr, 0, XNN_VALUE_FLAG_EXTERNAL_INPUT, &input);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor input" << std::endl;
    return nullptr;
  }

  uint32_t output = XNN_INVALID_VALUE_ID;
  status = xnn_define_tensor_value(
      subgraph.get(), datatype, dims.size(), dims.data(),
      /*data=*/nullptr, 1, /*flags=*/XNN_VALUE_FLAG_EXTERNAL_OUTPUT, &output);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor output" << std::endl;
    return nullptr;
  }

  std::vector<size_t> reduction_axes;
  reduction_axes.reserve(reduction_dims.size());
  for (size_t i = 0; i < reduction_dims.size(); ++i) {
    if ((norm_mask & (1 << i)) != 0) {
      reduction_axes.push_back(i);
    }
  }

  xnn_binary_params params = {-std::numeric_limits<float>::infinity(),
                              std::numeric_limits<float>::infinity()};

  // 1. Convert input to math_datatype (FP32) if necessary
  uint32_t math_input = input;
  if (datatype != math_datatype) {
    status = xnn_define_tensor_value(
        subgraph.get(), math_datatype, dims.size(), dims.data(),
        /*data=*/nullptr, XNN_INVALID_VALUE_ID, 0, &math_input);
    if (status != xnn_status_success) {
      std::cerr << "failed to create tensor math_input" << std::endl;
      return nullptr;
    }
    status = xnn_define_unary(subgraph.get(), xnn_unary_convert, nullptr, input,
                              math_input, /*flags=*/0);
    if (status != xnn_status_success) {
      std::cerr << "failed to create unary convert input->math" << std::endl;
      return nullptr;
    }
  }

  // 2. Compute mean in math_datatype (FP32)
  uint32_t mean = XNN_INVALID_VALUE_ID;
  status = xnn_define_tensor_value(
      subgraph.get(), math_datatype, reduction_dims.size(),
      reduction_dims.data(), /*data=*/nullptr, XNN_INVALID_VALUE_ID, 0, &mean);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor mean" << std::endl;
    return nullptr;
  }
  status = xnn_define_static_reduce(subgraph.get(), xnn_reduce_mean,
                                    reduction_axes.size(),
                                    reduction_axes.data(), math_input, mean,
                                    /*flags=*/XNN_FLAG_KEEP_DIMS);
  if (status != xnn_status_success) {
    std::cerr << "failed to create reduce mean" << std::endl;
    return nullptr;
  }

  // 3. Compute squared difference in math_datatype (FP32)
  uint32_t sqr_diff = XNN_INVALID_VALUE_ID;
  status = xnn_define_tensor_value(
      subgraph.get(), math_datatype, dims.size(), dims.data(),
      /*data=*/nullptr, XNN_INVALID_VALUE_ID, 0, &sqr_diff);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor sqr_diff" << std::endl;
    return nullptr;
  }
  status = xnn_define_binary(subgraph.get(), xnn_binary_squared_difference,
                             &params, math_input, mean, sqr_diff, /*flags=*/0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create binary squared difference" << std::endl;
    return nullptr;
  }

  // 4. Compute variance in math_datatype (FP32)
  uint32_t variance = XNN_INVALID_VALUE_ID;
  status = xnn_define_tensor_value(subgraph.get(), math_datatype,
                                   reduction_dims.size(), reduction_dims.data(),
                                   /*data=*/nullptr, XNN_INVALID_VALUE_ID, 0,
                                   &variance);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor variance" << std::endl;
    return nullptr;
  }
  status = xnn_define_static_reduce(subgraph.get(), xnn_reduce_mean,
                                    reduction_axes.size(),
                                    reduction_axes.data(), sqr_diff, variance,
                                    /*flags=*/XNN_FLAG_KEEP_DIMS);
  if (status != xnn_status_success) {
    std::cerr << "failed to create reduce mean (variance)" << std::endl;
    return nullptr;
  }

  // 5. Compute variance + epsilon in math_datatype (FP32)
  uint32_t variance_plus_epsilon = XNN_INVALID_VALUE_ID;
  status = xnn_define_tensor_value(subgraph.get(), math_datatype,
                                   reduction_dims.size(), reduction_dims.data(),
                                   /*data=*/nullptr, XNN_INVALID_VALUE_ID, 0,
                                   &variance_plus_epsilon);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor variance_plus_epsilon" << std::endl;
    return nullptr;
  }

  static const float epsilon_value = 1e-3f;
  uint32_t epsilon = XNN_INVALID_VALUE_ID;
  status = xnn_define_tensor_value(subgraph.get(), math_datatype, 0, nullptr,
                                   &epsilon_value, XNN_INVALID_VALUE_ID,
                                   /*flags=*/0, &epsilon);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor epsilon" << std::endl;
    return nullptr;
  }

  status = xnn_define_binary(subgraph.get(), xnn_binary_add, &params, variance,
                             epsilon, variance_plus_epsilon,
                             /*flags=*/0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create binary add" << std::endl;
    return nullptr;
  }

  // 6. Compute stddev (sqrt) in math_datatype (FP32)
  uint32_t stddev_math = XNN_INVALID_VALUE_ID;
  status = xnn_define_tensor_value(subgraph.get(), math_datatype,
                                   reduction_dims.size(), reduction_dims.data(),
                                   /*data=*/nullptr, XNN_INVALID_VALUE_ID, 0,
                                   &stddev_math);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor stddev_math" << std::endl;
    return nullptr;
  }
  status = xnn_define_unary(subgraph.get(), xnn_unary_square_root, nullptr,
                            variance_plus_epsilon, stddev_math, /*flags=*/0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create unary square root" << std::endl;
    return nullptr;
  }

  // Transition to datatype (FP16 or FP32)

  // 7. Convert mean to datatype if necessary, and compute input - mean in
  // datatype
  uint32_t reduced_mean = mean;
  if (datatype != math_datatype) {
    status = xnn_define_tensor_value(
        subgraph.get(), datatype, reduction_dims.size(), reduction_dims.data(),
        /*data=*/nullptr, XNN_INVALID_VALUE_ID, 0, &reduced_mean);
    if (status != xnn_status_success) {
      std::cerr << "failed to create tensor reduced_mean" << std::endl;
      return nullptr;
    }
    status = xnn_define_unary(subgraph.get(), xnn_unary_convert, nullptr, mean,
                              reduced_mean, /*flags=*/0);
    if (status != xnn_status_success) {
      std::cerr << "failed to create unary convert mean->datatype" << std::endl;
      return nullptr;
    }
  }

  uint32_t reduced_input_minus_mean = XNN_INVALID_VALUE_ID;
  status = xnn_define_tensor_value(
      subgraph.get(), datatype, dims.size(), dims.data(),
      /*data=*/nullptr, XNN_INVALID_VALUE_ID, 0, &reduced_input_minus_mean);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor reduced_input_minus_mean"
              << std::endl;
    return nullptr;
  }
  status =
      xnn_define_binary(subgraph.get(), xnn_binary_subtract, &params, input,
                        reduced_mean, reduced_input_minus_mean, /*flags=*/0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create binary subtract (reduced)" << std::endl;
    return nullptr;
  }

  // 8. Convert stddev_math to datatype if necessary
  uint32_t stddev = stddev_math;
  if (datatype != math_datatype) {
    status = xnn_define_tensor_value(
        subgraph.get(), datatype, reduction_dims.size(), reduction_dims.data(),
        /*data=*/nullptr, XNN_INVALID_VALUE_ID, 0, &stddev);
    if (status != xnn_status_success) {
      std::cerr << "failed to create tensor stddev" << std::endl;
      return nullptr;
    }
    status = xnn_define_unary(subgraph.get(), xnn_unary_convert, nullptr,
                              stddev_math, stddev, /*flags=*/0);
    if (status != xnn_status_success) {
      std::cerr << "failed to create unary convert stddev->datatype"
                << std::endl;
      return nullptr;
    }
  }

  // 9. Compute normalized (divide) in datatype
  uint32_t normalized = XNN_INVALID_VALUE_ID;
  status = xnn_define_tensor_value(
      subgraph.get(), datatype, dims.size(), dims.data(),
      /*data=*/nullptr, XNN_INVALID_VALUE_ID, 0, &normalized);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor normalized" << std::endl;
    return nullptr;
  }
  status = xnn_define_binary(subgraph.get(), xnn_binary_divide, &params,
                             reduced_input_minus_mean, stddev, normalized,
                             /*flags=*/0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create binary divide" << std::endl;
    return nullptr;
  }

  // 10. Compute normalized * weight in datatype
  static T weight_value = (T)2.0f;
  uint32_t weight = XNN_INVALID_VALUE_ID;
  status = xnn_define_tensor_value(subgraph.get(), datatype, 0, nullptr,
                                   &weight_value, XNN_INVALID_VALUE_ID,
                                   /*flags=*/0, &weight);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor weight" << std::endl;
    return nullptr;
  }

  uint32_t normalized_weight = XNN_INVALID_VALUE_ID;
  status = xnn_define_tensor_value(
      subgraph.get(), datatype, dims.size(), dims.data(),
      /*data=*/nullptr, XNN_INVALID_VALUE_ID, 0, &normalized_weight);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor normalized_weight" << std::endl;
    return nullptr;
  }
  status =
      xnn_define_binary(subgraph.get(), xnn_binary_multiply, &params,
                        normalized, weight, normalized_weight, /*flags=*/0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create binary multiply" << std::endl;
    return nullptr;
  }

  // 11. Compute output = normalized_weight + bias in datatype
  static T bias_value = (T)0.1f;
  uint32_t bias = XNN_INVALID_VALUE_ID;
  status = xnn_define_tensor_value(subgraph.get(), datatype, 0, nullptr,
                                   &bias_value, XNN_INVALID_VALUE_ID,
                                   /*flags=*/0, &bias);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor bias" << std::endl;
    return nullptr;
  }

  status = xnn_define_binary(subgraph.get(), xnn_binary_add, &params,
                             normalized_weight, bias, output, /*flags=*/0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create binary add (output)" << std::endl;
    return nullptr;
  }

  return subgraph.release();
}
xnn_subgraph_t FP32LayerNorm(size_t m, size_t n, size_t k, uint32_t norm_mask) {
  return LayerNormImpl<float>(m, n, k, norm_mask);
}

xnn_subgraph_t FP16LayerNorm(size_t m, size_t n, size_t k, uint32_t norm_mask) {
  return LayerNormImpl<xnn_float16>(m, n, k, norm_mask);
}

}  // namespace models

static void FP32LayerNorm(benchmark::State& state) {
  xnnpack::RunBenchmark(state, [&state]() {
    return models::FP32LayerNorm(state.range(0), state.range(1), state.range(2),
                                 state.range(3));
  });
}
static void FP16LayerNorm(benchmark::State& state) {
  xnnpack::RunBenchmark(state, [&state]() {
    return models::FP16LayerNorm(state.range(0), state.range(1), state.range(2),
                                 state.range(3));
  });
}

static void LayerNormArguments(benchmark::Benchmark* b) {
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
BENCHMARK(FP16LayerNorm)
    ->Unit(benchmark::kMicrosecond)
    ->MeasureProcessCPUTime()
    ->UseRealTime()
    ->Apply(LayerNormArguments);
