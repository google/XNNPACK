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
#include <benchmark/benchmark.h>

namespace models {

// Compute the RMS norm of [m x n x k] tensors, where the mean
// is computed over the dimensions in `norm_mask`. This computation is
// equivalent to: input / sqrt(mean(input^2, norm_mask))
//
// Where `mean(x, norm_mask)` means computing the mean of the dimensions in the
// `norm_mask`.
xnn_subgraph_t FP32RMSNorm(size_t m, size_t n, size_t k, uint32_t norm_mask) {
  xnn_status status;
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
      subgraph.get(), xnn_datatype_fp32, dims.size(), dims.data(),
      /*data=*/nullptr, 0, XNN_VALUE_FLAG_EXTERNAL_INPUT, &input);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor input" << std::endl;
    return nullptr;
  }

  uint32_t output = XNN_INVALID_VALUE_ID;
  status = xnn_define_tensor_value(
      subgraph.get(), xnn_datatype_fp32, dims.size(), dims.data(),
      /*data=*/nullptr, 1, /*flags=*/XNN_VALUE_FLAG_EXTERNAL_OUTPUT, &output);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor output" << std::endl;
    return nullptr;
  }

  uint32_t input_sq = XNN_INVALID_VALUE_ID;
  status = xnn_define_tensor_value(subgraph.get(), xnn_datatype_fp32,
                                   dims.size(), dims.data(),
                                   /*data=*/nullptr, XNN_INVALID_VALUE_ID,
                                   /*flags=*/0, &input_sq);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor input_sq" << std::endl;
    return nullptr;
  }

  status = xnn_define_unary(subgraph.get(), xnn_unary_square, nullptr, input,
                            input_sq, /*flags=*/0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create unary square" << std::endl;
    return nullptr;
  }

  uint32_t mean_sq = XNN_INVALID_VALUE_ID;
  status = xnn_define_tensor_value(subgraph.get(), xnn_datatype_fp32,
                                   reduction_dims.size(), reduction_dims.data(),
                                   /*data=*/nullptr, XNN_INVALID_VALUE_ID,
                                   /*flags=*/0, &mean_sq);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor mean_sq" << std::endl;
    return nullptr;
  }

  std::vector<size_t> reduction_axes;
  reduction_axes.reserve(reduction_dims.size());
  for (size_t i = 0; i < reduction_dims.size(); ++i) {
    if ((norm_mask & (1 << i)) != 0) {
      reduction_axes.push_back(i);
    }
  }
  status = xnn_define_static_reduce(subgraph.get(), xnn_reduce_mean,
                                    reduction_axes.size(),
                                    reduction_axes.data(), input_sq, mean_sq,
                                    /*flags=*/XNN_FLAG_KEEP_DIMS);
  if (status != xnn_status_success) {
    std::cerr << "failed to create reduce mean" << std::endl;
    return nullptr;
  }

  uint32_t inv_rms = XNN_INVALID_VALUE_ID;
  status = xnn_define_tensor_value(subgraph.get(), xnn_datatype_fp32,
                                   reduction_dims.size(), reduction_dims.data(),
                                   /*data=*/nullptr, XNN_INVALID_VALUE_ID,
                                   /*flags=*/0, &inv_rms);
  if (status != xnn_status_success) {
    std::cerr << "failed to create tensor inv_rms" << std::endl;
    return nullptr;
  }

  status = xnn_define_unary(subgraph.get(), xnn_unary_reciprocal_square_root,
                            nullptr, mean_sq, inv_rms, /*flags=*/0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create unary reciprocal square root" << std::endl;
    return nullptr;
  }

  xnn_binary_params params = {-std::numeric_limits<float>::infinity(),
                              std::numeric_limits<float>::infinity()};
  status = xnn_define_binary(subgraph.get(), xnn_binary_multiply, &params,
                             input, inv_rms, output,
                             /*flags=*/0);
  if (status != xnn_status_success) {
    std::cerr << "failed to create binary multiply" << std::endl;
    return nullptr;
  }

  return subgraph.release();
}

}  // namespace models

static void FP32RMSNorm(benchmark::State& state) {
  xnnpack::RunBenchmark(state, [&state]() {
    return models::FP32RMSNorm(state.range(0), state.range(1), state.range(2),
                               state.range(3));
  });
}

static void RMSNormArguments(benchmark::Benchmark* b) {
  b->ArgNames({"M", "N", "K", "NormMask"});
  for (int norm_mask = 1; norm_mask < 8; norm_mask++) {
    b->Args({128, 256, 512, norm_mask});
  }
}

BENCHMARK(FP32RMSNorm)
    ->Unit(benchmark::kMicrosecond)
    ->MeasureProcessCPUTime()
    ->UseRealTime()
    ->Apply(RMSNormArguments);
